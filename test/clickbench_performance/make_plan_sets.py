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

"""Generate lane plans from real simpatico explore output and sample classes.

Preferred input is the CLI's --frontier-out TSV: column,dtype,picked,ratio,
comp_gbps,decomp_gbps,old_wall_comp_gbps,old_wall_decomp_gbps,compressed_bytes,plan.
The plan field flattens newlines with '; ', not JSON or CSV escaping. Filter out
OnPair/FSST for the general baseline, then select max ratio with decompress >=
250 GB/s; if none qualify, select fastest decode (the explorer's exact fallback).
Low-cardinality string choices must start with dictionary. Numerics and low-card
plans are byte-identical in all lanes. Free text uses identity/general/onpair_16
with delta+ANS on row_offsets (byte offsets). No training parameters are overridden.

Also accepts stdout '# column: ...' / '# comp: ... decomp: ...' plan blocks
separated by '---'. Those contain ONLY the CLI's picked plan: run explore with
--score pareto --decomp-floor 250. OnPair is not explorable in the registry.
A TSV frontier is required if the stdout pick is not dictionary for
a low-card string (or comes from a different search including OnPair). OOM/untimed identity blocks are rejected, not silently scored.
Plans cover every full-table column in schema order, including unqueried columns.

Free-text general lane: the explorer's Pareto picks for URL/Referer/Title on the
10M sample are dictionary-rooted (Snappy or GDeflate on the keys, Snappy on Title's
indices), but Simpatico's dictionary operator admits
a chunk only while its estimated distinct fraction is below 0.5
(dictionary_compressor.cu, kDictMaxCardFraction) and pin_table latches compression
off for every later chunk once one chunk is refused; on the full hits that happened
at chunk 10 of 27. With --free-text-bench (CLI bench CSV plus its plan file, one
per codec) the general lane's free-text candidates are those measured str_split
plans only, selected by the same rule; the plan comment records the exclusion.
Bench plans must be rooted at str_split (the family the comment promises); any
other root, dictionary included, is rejected. Note that the frontier also holds
dictionary-free str_split plans above the decode floor (nvCOMP ANS on the chars
for URL/Referer/Title/OriginalURL, raw chars on SearchPhrase; 1.4x to 1.7x at
336-378 GB/s); restricting to the bench candidates is a declared change of
candidate set, not a consequence of the rule.

Explorer lane: the fourth plan set applies the same rule to the frontier itself
with the dictionary-rooted plans removed (the exclusion above) and OnPair
excluded, so its free-text plans are what the explorer would pick among the
plans that survive the full data. Numerics and low-card plans are the shared
ones; the plan comment records the candidate set.

--onpair-bench (one CLI bench CSV:PLAN measuring the onpair lane's exact free-text
plan) replaces the onpair lane's "not yet measured" header with the measured ratio
and throughput, and records what the same selection rule picks when OnPair is
allowed into the candidate set (the general bench candidates, or the explorer
frontier without a bench, plus OnPair). The lane's plan does not change; the
comment states whether the rule itself would have chosen it.
"""

import argparse
import csv
import io
import math
import re
from pathlib import Path

from clickbench_bench import write_csv
from clickbench_queries import SCHEMA_COLUMNS, sql_literal

import duckdb

FREE_TEXT = frozenset(("URL", "Referer", "Title", "SearchPhrase", "OriginalURL"))
ONPAIR = """input -> onpair_16 -> codes, row_offsets, dict_bytes, dict_offsets, batch_prefixes
onpair_16.row_offsets -> delta -> differences
onpair_16.row_offsets.differences -> ans"""


def parse_explore(text):
    """Return column -> list of timed candidate dicts, from actual CLI formats."""
    candidates = {}
    if text.startswith("column\tdtype\tpicked\tratio\t"):
        for row in csv.DictReader(
            io.StringIO(text), delimiter="\t", quoting=csv.QUOTE_NONE
        ):
            candidate = {
                "dtype": row["dtype"],
                "ratio": float(row["ratio"]),
                "comp": float(row["comp_gbps"]),
                "decomp": float(row["decomp_gbps"]),
                "plan": row["plan"].replace("; ", "\n").strip(),
            }
            candidates.setdefault(row["column"], []).append(candidate)
    else:
        for block in re.split(r"(?m)^---\s*$", text):
            meta = re.search(
                r"(?m)^# column: (\S+)\s+dtype: (\S+)\s+ratio: ([\d.]+)x\s+depth: \d+",
                block,
            )
            if not meta:
                if block.strip():
                    raise ValueError(
                        "not a simpatico explore stdout block or frontier TSV"
                    )
                continue
            speed = re.search(
                r"(?m)^# comp: ([\d.]+) GB/s\s+decomp: ([\d.]+) GB/s", block
            )
            if not speed:
                raise ValueError(
                    f"untimed/OOM explore block for {meta[1]}; rerun explore"
                )
            plan = "\n".join(
                line.strip()
                for line in block.splitlines()
                if line.strip() and not line.startswith("#")
            )
            candidates.setdefault(meta[1], []).append(
                {
                    "dtype": meta[2],
                    "ratio": float(meta[3]),
                    "comp": float(speed[1]),
                    "decomp": float(speed[2]),
                    "plan": plan,
                }
            )
    if not candidates:
        raise ValueError("empty explore output")
    for column, choices in candidates.items():
        for c in choices:
            if not c["plan"].startswith("input -> ") or any(
                not math.isfinite(c[k]) or c[k] <= 0
                for k in ("ratio", "comp", "decomp")
            ):
                raise ValueError(f"invalid/untimed explore candidate for {column}")
    return candidates


def choose_plan(candidates, column_class, floor=250.0, allow_onpair=False):
    """Pre-declared selection rules, identical in every lane.

    numeric / free_text: the SF1000 rule, max ratio among candidates decoding at
    >= floor GB/s, else the fastest decode (the explorer's own fallback).
    low_cardinality: max ratio among dictionary candidates with the floor waived.
    These columns are small (short strings, few distinct values), so their absolute
    decode time is negligible while their raw-indices dictionary form barely
    compresses; applying the floor would pick that form or identity. A column whose
    frontier holds no dictionary candidate at all (near-empty columns, where the
    explorer's str_split cascades dominate every dictionary plan) takes the
    frontier's max-ratio candidate; the pick carries a note for the plan comment.
    """
    allowed = [
        c
        for c in candidates
        if allow_onpair
        or not re.search(r"\b(?:onpair(?:_\d+)?|fsst)\b", c["plan"], re.IGNORECASE)
    ]
    if not allowed:
        raise ValueError(
            f"no general-purpose {column_class} candidate; supply the explore frontier TSV"
        )
    if column_class == "low_cardinality":
        dictionary = [
            c for c in allowed if re.match(r"input\s*->\s*dictionary\b", c["plan"])
        ]
        if dictionary:
            return {**max(dictionary, key=lambda c: c["ratio"]), "note": ""}
        return {
            **max(allowed, key=lambda c: c["ratio"]),
            "note": "no dictionary candidate on the frontier; frontier max-ratio plan",
        }
    above = [c for c in allowed if c["decomp"] >= floor]
    # max() preserves CLI frontier order for exactly equal scores.
    pick = (
        max(above, key=lambda c: c["ratio"])
        if above
        else max(allowed, key=lambda c: c["decomp"])
    )
    return {
        **pick,
        "note": "" if above else "no candidate met 250 GB/s; fastest-decode fallback",
    }


def parse_bench(csv_text, plan_text, root="str_split"):
    """Column -> [candidate] from a CLI bench CSV and the plan file it measured.

    Plans must be rooted at `root` (str_split for the general-purpose bench,
    onpair_16 for the onpair lane's own measurement).
    """
    plans = {}
    for block in re.split(r"(?m)^---\s*$", plan_text):
        meta = re.search(r"(?m)^# column: (\S+)\s+dtype: (\S+)", block)
        if not meta:
            if block.strip():
                raise ValueError("not a CLI bench plan file")
            continue
        plans[meta[1]] = (
            meta[2],
            "\n".join(
                line.strip()
                for line in block.splitlines()
                if line.strip() and not line.startswith("#")
            ),
        )
    candidates = {}
    for row in csv.DictReader(io.StringIO(csv_text)):
        name = row["column"]
        if name == "TOTAL":
            continue
        if name not in plans:
            raise ValueError(f"bench CSV column {name} has no plan block")
        if row["verify_ok"] != "1":
            raise ValueError(f"bench candidate for {name} did not verify")
        dtype, plan = plans[name]
        if not re.match(rf"input\s*->\s*{re.escape(root)}\b", plan):
            raise ValueError(
                f"bench plan for {name} is not rooted at {root}: {plan.splitlines()[0]}"
            )
        candidate = {
            "dtype": dtype,
            "ratio": float(row["ratio"]),
            "comp": float(row["compress_gbps_median"]),
            "decomp": float(row["decompress_gbps_median"]),
            "plan": plan,
        }
        if not plan.startswith("input -> ") or any(
            not math.isfinite(candidate[k]) or candidate[k] <= 0
            for k in ("ratio", "comp", "decomp")
        ):
            raise ValueError(f"invalid bench candidate for {name}")
        candidates.setdefault(name, []).append(candidate)
    if not candidates:
        raise ValueError("empty bench CSV")
    return candidates


def classify(schema, counts, rows):
    if rows <= 0:
        raise ValueError("classification requires a non-empty sample")
    result = []
    for name, dtype in schema:
        distinct = int(counts[name])
        if not 0 <= distinct <= rows:
            raise ValueError(f"invalid distinct count for {name}")
        is_string = dtype.upper().startswith(("TEXT", "VARCHAR", "CHAR", "STRING"))
        kind = (
            "numeric"
            if not is_string
            else "low_cardinality" if distinct < rows * 0.01 else "free_text"
        )
        result.append(
            dict(column=name, **{"class": kind}, distinct=distinct, rows=rows)
        )
    return result


def measure_classes(parquet):
    with duckdb.connect() as con:
        con.execute("SET preserve_insertion_order = true")
        source = f"read_parquet({sql_literal(parquet)})"
        rows = con.execute(f"SELECT COUNT(*) FROM {source}").fetchone()[0]
        # One column at a time bounds COUNT(DISTINCT)'s live hash sets.
        counts = {
            name: con.execute(
                f'SELECT COUNT(DISTINCT "{name}") FROM {source}'
            ).fetchone()[0]
            for name, _ in SCHEMA_COLUMNS
        }
    return classify(SCHEMA_COLUMNS, counts, rows)


def render_plan_sets(
    candidates, classes, schema=SCHEMA_COLUMNS, free_text_bench=None, onpair_bench=None
):
    by_name = {c["column"]: c for c in classes}
    columns = [name for name, _ in schema]
    if (
        len(by_name) != len(classes)
        or set(by_name) != set(columns)
        or set(candidates) != set(columns)
    ):
        raise ValueError(
            "classes and explore output must each cover exactly the full schema"
        )
    outputs = {lane: [] for lane in ("uncompressed", "general", "onpair", "explorer")}
    for name in columns:
        cls = by_name[name]["class"]
        if cls not in ("numeric", "low_cardinality", "free_text"):
            raise ValueError(f"unknown class for {name}: {cls}")
        pick = choose_plan(candidates[name], cls)
        bench_note = ""
        explorer_pick = None
        if cls == "free_text":
            frontier = [
                c
                for c in candidates[name]
                if not re.match(r"input\s*->\s*dictionary\b", c["plan"])
            ]
            if not frontier:
                raise ValueError(f"no dictionary-free frontier candidate for {name}")
            explorer_pick = {
                **choose_plan(frontier, cls),
                "candidates_note": (
                    "explorer candidates: frontier plans without a dictionary root"
                    " (dictionary excluded: per-chunk admission below 0.5 distinct"
                    " fraction, failure latches compression off)"
                ),
            }
        if cls == "free_text" and free_text_bench is not None:
            if name not in free_text_bench:
                raise ValueError(f"no --free-text-bench candidates for {name}")
            pick = choose_plan(free_text_bench[name], cls)
            bench_note = (
                "\n# general-purpose candidates: measured str_split plans only"
                " (dictionary excluded: per-chunk admission below 0.5 distinct"
                " fraction, failure latches compression off)"
            )
        onpair_measured = None
        if cls == "free_text" and onpair_bench is not None:
            if name not in onpair_bench:
                raise ValueError(f"no --onpair-bench candidate for {name}")
            measured = [c for c in onpair_bench[name] if c["plan"] == ONPAIR]
            if len(measured) != 1:
                raise ValueError(
                    f"--onpair-bench must measure exactly the onpair lane plan for {name}"
                )
            onpair_measured = measured[0]
            pool = (free_text_bench or candidates)[name] + [onpair_measured]
            rule = choose_plan(pool, cls, allow_onpair=True)
            rule_pick = rule["plan"].splitlines()[0]
            onpair_measured = {
                **onpair_measured,
                "rule_note": (
                    "rule over all measured candidates (general"
                    f"{' bench' if free_text_bench else ' frontier'} + onpair):"
                    f" picks {rule_pick} ({rule['ratio']:.4f}x, {rule['decomp']:.2f} GB/s)"
                    + (f"; {rule['note']}" if rule.get("note") else "")
                ),
            }
        for lane, blocks in outputs.items():
            plan = pick["plan"]
            if cls == "free_text":
                plan = {
                    "uncompressed": "input -> identity",
                    "general": pick["plan"],
                    "onpair": ONPAIR,
                    "explorer": explorer_pick["plan"],
                }[lane]
            changed = plan != pick["plan"]
            comment = f"# column: {name}  dtype: {pick['dtype']}  class: {cls}"
            if cls == "free_text" and lane == "explorer":
                comment += (
                    f"  ratio: {explorer_pick['ratio']:.4f}x\n# comp:"
                    f" {explorer_pick['comp']:.2f} GB/s  decomp:"
                    f" {explorer_pick['decomp']:.2f} GB/s"
                )
                if explorer_pick.get("note"):
                    comment += f"\n# {explorer_pick['note']}"
                comment += f"\n# {explorer_pick['candidates_note']}"
            elif changed and lane == "onpair" and onpair_measured is not None:
                comment += (
                    f"  ratio: {onpair_measured['ratio']:.4f}x\n# comp:"
                    f" {onpair_measured['comp']:.2f} GB/s  decomp:"
                    f" {onpair_measured['decomp']:.2f} GB/s (CLI bench of this plan)"
                    f"\n# {onpair_measured['rule_note']}"
                )
            elif changed:
                comment += (
                    f"\n# lane override: {lane}; ratio and throughput not yet measured"
                )
            else:
                comment += f"  ratio: {pick['ratio']:.4f}x\n# comp: {pick['comp']:.2f} GB/s  decomp: {pick['decomp']:.2f} GB/s"
                if pick.get("note"):
                    comment += f"\n# {pick['note']}"
                comment += bench_note
            blocks.append(comment + "\n" + plan)
    return {lane: "\n\n---\n".join(blocks) + "\n" for lane, blocks in outputs.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--explore", required=True, help="CLI stdout plan blocks or --frontier-out TSV"
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--parquet",
        help="10M-row classification sample (never the replicated/full set)",
    )
    source.add_argument("--classes", help="reuse previously measured classes.csv")
    parser.add_argument(
        "--output", default=str(Path(__file__).resolve().parent / "plans/clickbench")
    )
    parser.add_argument(
        "--free-text-bench",
        action="append",
        default=[],
        metavar="CSV:PLAN",
        help="CLI bench CSV and the plan file it measured; repeat per codec. When"
        " given, the general lane's free-text plans are chosen among these only",
    )
    parser.add_argument(
        "--onpair-bench",
        metavar="CSV:PLAN",
        help="CLI bench CSV and plan file measuring the onpair lane's exact free-text"
        " plan; fills the onpair lane's header and records the rule's pick with"
        " OnPair allowed",
    )
    args = parser.parse_args()
    onpair_bench = None
    if args.onpair_bench:
        csv_path, _, plan_path = args.onpair_bench.partition(":")
        if not plan_path:
            raise ValueError("--onpair-bench takes CSV:PLAN")
        onpair_bench = parse_bench(
            Path(csv_path).read_text(), Path(plan_path).read_text(), root="onpair_16"
        )
    free_text_bench = None
    if args.free_text_bench:
        free_text_bench = {}
        for spec in args.free_text_bench:
            csv_path, _, plan_path = spec.partition(":")
            if not plan_path:
                raise ValueError("--free-text-bench takes CSV:PLAN")
            for name, choices in parse_bench(
                Path(csv_path).read_text(), Path(plan_path).read_text()
            ).items():
                free_text_bench.setdefault(name, []).extend(choices)
    if args.classes:
        with Path(args.classes).open() as f:
            recorded = list(csv.DictReader(f))
        counts = {r["column"]: int(r["distinct"]) for r in recorded}
        row_counts = {int(r["rows"]) for r in recorded}
        if len(row_counts) != 1 or len(counts) != len(recorded):
            raise ValueError(
                "classes must describe one sample, without duplicate columns"
            )
        classes = classify(SCHEMA_COLUMNS, counts, row_counts.pop())
        if any(a["class"] != b["class"] for a, b in zip(classes, recorded)):
            raise ValueError("recorded classes disagree with the <1% rule/schema order")
    else:
        classes = measure_classes(args.parquet)
    free = {c["column"] for c in classes if c["class"] == "free_text"}
    if free != FREE_TEXT:
        raise ValueError(
            f"sample classes differ from the pre-declared five free-text columns: {sorted(free)}"
        )
    outputs = render_plan_sets(
        parse_explore(Path(args.explore).read_text()),
        classes,
        free_text_bench=free_text_bench,
        onpair_bench=onpair_bench,
    )
    destination = Path(args.output)
    for lane, text in outputs.items():
        (destination / lane).mkdir(parents=True, exist_ok=True)
        (destination / lane / "hits.txt").write_text(text)
    write_csv(
        destination / "classes.csv", ["column", "class", "distinct", "rows"], classes
    )


if __name__ == "__main__":
    main()
