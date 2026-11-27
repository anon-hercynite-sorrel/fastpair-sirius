# =============================================================================
# Copyright 2026, Sirius Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License
# is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing permissions and limitations under
# the License.
# =============================================================================

"""Compare two or three lane directories using the pre-declared change rule.

A change is meaningful when abs(candidate_median - baseline_median) is at least
5% of baseline_median AND exceeds 3*sqrt(baseline_std**2 + candidate_std**2).
Stddev is the sample standard deviation of hot repetitions, not a standard error.
At least two hot repetitions per lane are required for a meaningful-change claim.
The union of fallback, disk and validation-failed queries is excluded from every
aggregate. Tier differences are disclosed separately and excluded from codec
speedups. Comparisons without validation are explicitly unvalidated. Shared
extension/config/pre-SQL/performance-environment/DuckDB-memory fingerprints must
match; plans and layouts may differ.
"""

import argparse
import csv
import itertools
import json
import math
from pathlib import Path

from clickbench_bench import DECODE_FIELDS, PROFILE_FIELDS, geomean
from clickbench_queries import QUERY_COLUMNS


def meaningful_change(baseline, candidate):
    delta = candidate["hot_median_s"] - baseline["hot_median_s"]
    relative = delta / baseline["hot_median_s"]
    sigma = math.hypot(baseline["hot_std_s"], candidate["hot_std_s"])
    return relative, abs(relative) >= 0.05 - 1e-12 and abs(delta) > 3 * sigma


def load_lane(directory):
    directory = Path(directory)
    summary = json.loads((directory / "summary.json").read_text())
    with (directory / "results.csv").open() as f:
        records = list(csv.DictReader(f))
    rows = {}
    for record in records:
        q = int(record["query"])
        if q in rows:
            raise ValueError(f"duplicate query {q} in {directory}")
        rows[q] = {
            **record,
            **{
                k: float(record[k])
                for k in (
                    "cold_s",
                    "hot_median_s",
                    "hot_min_s",
                    "hot_mean_s",
                    "hot_std_s",
                )
            },
        }
        rows[q]["fallback"] = record["fallback"].lower() == "true"
        rows[q]["disk"] = record["disk"].lower() == "true"
        rows[q]["served_by"] = json.loads(record["served_by"])
        for key in (*PROFILE_FIELDS, *DECODE_FIELDS):
            value = record.get(key)
            rows[q][key] = float(value) if value else None
            if rows[q][key] is not None and (
                not math.isfinite(rows[q][key])
                or (key != "remainder_ms" and rows[q][key] < 0)
            ):
                raise ValueError(f"invalid {key} for query {q}")
        if not rows[q]["served_by"]:
            raise ValueError(f"missing pin assignment audit for query {q}")
        if (
            any(
                not math.isfinite(rows[q][k]) or rows[q][k] < 0
                for k in ("hot_median_s", "hot_std_s")
            )
            or rows[q]["hot_median_s"] <= 0
        ):
            raise ValueError(f"invalid timing for query {q}")
    with (directory / "footprint.csv").open() as f:
        footprint = list(csv.DictReader(f))
    return {"summary": summary, "rows": rows, "footprint": footprint}


def footprint_totals(rows):
    result = {}
    for tier in ("gpu", "host"):
        selected = [r for r in rows if r["tier"] == tier]
        result[tier] = (
            None
            if any(not r["resident_bytes"] for r in selected)
            else sum(int(r["resident_bytes"]) for r in selected)
        )
    return result


def served_tiers(row):
    return {a["tier"] for a in row["served_by"]}


def comparison_fingerprint(summary):
    """Compare execution knobs; lane-local config/log/output paths are metadata."""
    shared = (
        "extension_sha256",
        "config_sha256",
        "pre_sql",
        "duckdb_memory_limit",
    )
    for key in (*shared, "perf_stack_env"):
        if key not in summary:
            raise ValueError(f"missing comparison fingerprint: {key}")
    env = summary["perf_stack_env"]
    return {
        **{key: summary[key] for key in shared},
        "source": summary.get("source", "parquet"),
        "duckdb_path": (
            summary.get("duckdb_path") if summary.get("source") == "table" else None
        ),
        "profile": summary.get("profile", False),
        "perf_stack_env": {
            key: env[key]
            for key in sorted(env)
            if key == "LD_PRELOAD"
            or key.startswith(("SIRIUS_EXP_", "SIRIUS_LATE_MAT", "SIRIUS_PIN_TIER"))
        },
    }


def free_text_queries():
    columns = {"URL", "Title", "Referer", "SearchPhrase", "OriginalURL"}
    return {q for q, used in QUERY_COLUMNS.items() if used & columns}


def amdahl_penalties(base, lane):
    """Signed differences between hot medians; the first lane is the control."""
    wall = lane["hot_median_s"] * 1000
    wall_penalty = wall - base["hot_median_s"] * 1000
    if lane["exec_ms"] is None or base["exec_ms"] is None:
        return None, wall_penalty, None
    exec_penalty = lane["exec_ms"] - base["exec_ms"]
    denominator = wall - exec_penalty
    return (
        exec_penalty,
        wall_penalty,
        wall / denominator if denominator > 0 else None,
    )


def positive_geomean(values):
    """A geometric mean of signed penalties is undefined if any is <= zero."""
    values = list(values)
    return (
        geomean(values)
        if values and all(v is not None and v > 0 for v in values)
        else None
    )


def print_breakdown(lanes, eligible):
    def display(value):
        return "n/a" if value is None else f"{value:.3f}"

    print("\nHot median breakdown (ms; wall remains the headline)")
    print(
        "Non-execution floor = planner + optimizer + physical + remainder; "
        "optimizer extension is already inside optimizer."
    )
    print(
        "Decode sum counts concurrent chunk work at face value, not elapsed query wall time."
    )
    print(
        "query | lane | wall_ms | non_execution_floor_ms | exec_ms | decode_ms_sum | status"
    )
    for q in sorted(lanes[0]["rows"]):
        for lane in lanes:
            row = lane["rows"][q]
            parts = [
                row[k]
                for k in (
                    "planner_ms",
                    "optimizer_ms",
                    "physical_ms",
                    "remainder_ms",
                )
            ]
            floor = sum(parts) if all(p is not None for p in parts) else None
            status = "eligible" if q in eligible else "excluded from codec aggregates"
            print(
                f"q{q} | {lane['summary']['lane']} | {display(row['hot_median_s'] * 1000)} | "
                f"{display(floor)} | {display(row['exec_ms'])} | {display(row['decode_ms_sum'])} | {status}"
            )

    base = lanes[0]
    base_name = base["summary"]["lane"]
    selected = sorted(eligible & free_text_queries())
    print(f"Free-text geomean query set: {selected}")
    print(
        "Nonpositive/missing penalties make their geomean n/a; signed per-query penalties are retained."
    )
    print(
        f"Amdahl control: {base_name} (first lane; supply the uncompressed lane first)"
    )
    print(
        "Bound = wall_lane / (wall_lane - exec_penalty); nonpositive denominator is n/a."
    )
    for lane in lanes[1:]:
        name = lane["summary"]["lane"]
        print(f"\n{name} relative to {base_name}")
        print(
            "query | lane | baseline_lane | exec_penalty_ms | wall_penalty_ms | zero_decode_penalty_speedup_bound | status"
        )
        penalties = {}
        for q in sorted(base["rows"]):
            penalties[q] = amdahl_penalties(base["rows"][q], lane["rows"][q])
            status = "eligible" if q in eligible else "excluded from codec aggregates"
            print(
                f"q{q} | {name} | {base_name} | "
                + " | ".join(display(v) for v in penalties[q])
                + f" | {status}"
            )
        for index, label in enumerate(
            (
                "exec_penalty_ms",
                "wall_penalty_ms",
                "zero_decode_penalty_speedup_bound",
            )
        ):
            values = [penalties[q][index] for q in selected]
            nonpositive = sum(v is not None and v <= 0 for v in values)
            missing = sum(v is None for v in values)
            print(
                f"{name} relative to {base_name}: free-text geomean {label}="
                f"{display(positive_geomean(values))} (queries={len(selected)}, "
                f"nonpositive={nonpositive}, missing={missing})"
            )


def report(directories, allow_compression_failure=False, breakdown=False):
    lanes = [load_lane(d) for d in directories]
    if len({lane["summary"]["lane"] for lane in lanes}) != len(lanes):
        raise ValueError("lane names must be distinct")
    if not allow_compression_failure and any(
        l["summary"]["compression_failed"] for l in lanes
    ):
        raise ValueError(
            "pin compression failed; use --allow-compression-failure to inspect this comparison"
        )
    queries = set(lanes[0]["rows"])
    baseline_fingerprint = comparison_fingerprint(lanes[0]["summary"])
    for lane in lanes:
        fingerprint = comparison_fingerprint(lane["summary"])
        if set(lane["rows"]) != queries:
            raise ValueError("lanes must contain the same query subset")
        if lane["summary"]["parquet"] != lanes[0]["summary"]["parquet"]:
            raise ValueError("lane inputs differ; compare the same Parquet file/glob")
        for key in baseline_fingerprint:
            if fingerprint[key] != baseline_fingerprint[key]:
                raise ValueError(f"comparison fingerprint differs: {key}")
    if len({(Path(d) / "effective_queries.sql").read_text() for d in directories}) != 1:
        raise ValueError("effective SQL differs across lanes")
    excluded = {}
    for lane in lanes:
        summary = lane["summary"]
        failed = {int(k.split(":")[0]) for k in summary.get("validation_failures", {})}
        for q, row in lane["rows"].items():
            reasons = []
            if row["fallback"] or q in summary["fallback_queries"]:
                reasons.append("fallback")
            if row["disk"] or q in summary.get("disk_queries", []):
                reasons.append("disk")
            if row["validated"].lower() == "false" or q in failed:
                reasons.append("validation failed")
            if reasons:
                excluded.setdefault(q, []).extend(
                    f"{summary['lane']}: {r}" for r in reasons
                )
    tier_differences = {
        q
        for q in queries
        if len({tuple(sorted(served_tiers(l["rows"][q]))) for l in lanes}) > 1
        or any(len(served_tiers(l["rows"][q])) > 1 for l in lanes)
    }
    eligible = queries - excluded.keys() - tier_differences
    print(
        "Variant reminder: Q17 adds ordered top-k; Q24 equals Q26 and weights it twice in the geomean."
    )
    print(f"Excluded queries (fallback, disk, validation failure): {excluded}")
    print(
        f"Layout differences (excluded from codec aggregates): {sorted(tier_differences)}"
    )
    print(f"Geomean query set: {sorted(eligible)}")
    if any(l["summary"]["compression_failed"] for l in lanes):
        print(
            "COMPRESSION FAILURE OVERRIDE: timings include pins with failed compression."
        )
    all_validated = all(
        l["summary"].get("validate") is True
        and all(
            r["validated"].lower() == "true"
            for q, r in l["rows"].items()
            if q in eligible
        )
        for l in lanes
    )
    for lane in lanes:
        summary = lane["summary"]
        value = geomean(lane["rows"][q]["hot_median_s"] for q in eligible)
        status = "validated" if all_validated else "UNVALIDATED"
        print(
            f"{summary['lane']}: hot geomean_s={value} ({status}), pin_s={summary['pin_seconds']}, "
            f"resident_bytes={footprint_totals(lane['footprint'])} (None=unavailable)"
        )
    for base, candidate in itertools.combinations(lanes, 2):
        a, b = base["summary"]["lane"], candidate["summary"]["lane"]
        validated = (
            base["summary"].get("validate") is True
            and candidate["summary"].get("validate") is True
            and all(
                base["rows"][q]["validated"].lower() == "true"
                and candidate["rows"][q]["validated"].lower() == "true"
                for q in eligible
            )
        )
        status = (
            "validated"
            if validated
            else "UNVALIDATED (--no-validate or missing validation)"
        )
        print(f"\n{b} relative to {a} [{status}]: negative time change is faster")
        print("query | time_change | assessment | layout_difference")
        for q in sorted(queries):
            relative, meaningful = meaningful_change(
                base["rows"][q], candidate["rows"][q]
            )
            enough = (
                min(
                    base["summary"]["hot_reps"],
                    candidate["summary"]["hot_reps"],
                )
                >= 2
            )
            tiers_a, tiers_b = served_tiers(base["rows"][q]), served_tiers(
                candidate["rows"][q]
            )
            layout = (
                f"{','.join(sorted(tiers_a))} -> {','.join(sorted(tiers_b))}"
                if q in tier_differences
                else ""
            )
            reason = (
                "; ".join(excluded[q])
                if q in excluded
                else (
                    "layout difference"
                    if q in tier_differences
                    else (
                        "unvalidated"
                        if not validated
                        else (
                            "fewer than 2 hot reps"
                            if not enough
                            else "meaningful" if meaningful else "below rule"
                        )
                    )
                )
            )
            change = "excluded" if q in excluded else f"{relative:+.2%}"
            print(f"q{q} | {change} | {reason} | {layout}")
        speedup = geomean(
            base["rows"][q]["hot_median_s"] / candidate["rows"][q]["hot_median_s"]
            for q in eligible
        )
        print(f"Geomean speedup {a}/{b}: {speedup} [{status}]")
    if breakdown:
        print_breakdown(lanes, eligible)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("lanes", nargs="+")
    parser.add_argument("--allow-compression-failure", action="store_true")
    parser.add_argument("--breakdown", action="store_true")
    args = parser.parse_args()
    if len(args.lanes) not in (2, 3):
        parser.error("supply two or three lane output directories")
    report(args.lanes, args.allow_compression_failure, args.breakdown)


if __name__ == "__main__":
    main()
