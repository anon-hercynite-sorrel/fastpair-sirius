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

"""Single-table ClickBench lane runner; query numbers are 0-based.

FROM hits uses read_parquet(the exact pin source) by default, or a persistent
DuckDB-native table with --source table --duckdb-path. --config is passed as
SIRIUS_CONFIG_FILE before LOAD, matching the TPC-H harness. SIRIUS_PRE_SQL is
applied before the explicit lane settings, so it cannot change the lane.

The parent never loads Sirius. A GPU child pins and runs q0..q42 sequentially,
fetching every result inside the wall timer. After that child EXITS (releasing
Sirius pools), a separate, extension-free DuckDB child validates EVERY repetition.
Comparison preserves row order; floats use rel_tol=1e-4, abs_tol=1e-6.

Fallback audit uses SIRIUS_LOG_BACKEND=spdlog and SIRIUS_LOG_LEVEL=info, forced by
the runner. C++ transparent execution counters have no SQL accessor. Untimed
SET VARIABLE string sentinels bracket each repetition in QueryBegin INFO records;
planning fallback and runtime replay WARN/ERROR records are associated with that
window after process exit flushes logs. A window without a positive transparent
GPU execution record is conservatively excluded too (e.g. optimizer-only COUNT).
Missing sentinels fail the audit, rather than certifying GPU execution.

Outputs include per-repetition pin assignments (served_by, a JSON list of entry/tier
objects), per-column resident/uncompressed footprints, and a fingerprint of the
extension, effective YAML, pre-SQL, plans, layout and git revision. Windows with
no pin assignment are disk-served and excluded from aggregate claims, alongside
fallback and validation failures. Mixed layouts duplicate every queried column
on host so each scan can be served by one entry.
"""

import argparse
import csv
import glob
import hashlib
import json
import math
import os
import pickle
import re
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

from clickbench_queries import (
    QUERIES,
    QUERY_COLUMNS,
    render_query,
    sql_identifier,
    sql_literal,
    union_columns_by_table,
)

import duckdb

ROOT = Path(__file__).resolve().parents[2]
EXTENSION = (
    ROOT / "sirius-duckdb/build/release/extension/sirius/sirius.duckdb_extension"
)
PROFILE_FIELDS = (
    "planner_ms",
    "optimizer_ms",
    "optimizer_ext_ms",
    "physical_ms",
    "exec_ms",
    "profiler_total_ms",
    "remainder_ms",
)
DECODE_FIELDS = ("decode_ms_sum", "decode_calls")


def write_csv(path, fields, rows):
    with Path(path).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def parse_queries(value):
    result = set()
    try:
        for part in value.split(","):
            bounds = [int(x) for x in part.split("-")]
            if len(bounds) == 1:
                result.add(bounds[0])
            elif len(bounds) == 2 and bounds[0] <= bounds[1]:
                result.update(range(bounds[0], bounds[1] + 1))
            else:
                raise ValueError
        if not result or result - QUERIES.keys():
            raise ValueError
    except ValueError:
        raise argparse.ArgumentTypeError(
            "queries must be comma-separated 0-based numbers/ranges in 0-42"
        ) from None
    return sorted(result)


def load_pin_layout(path=None):
    columns = union_columns_by_table()["hits"]
    entries = (
        json.loads(Path(path).read_text())
        if path
        else [{"name": "hits", "tier": "gpu", "cols": columns}]
    )
    if not isinstance(entries, list) or not entries:
        raise ValueError("pin layout must be a non-empty list")
    seen_names, covered, resolved = set(), set(), []
    for entry in entries:
        if not isinstance(entry, dict):
            # Keep the layout validator's existing ValueError interface.
            raise ValueError("pin entries must be objects")  # noqa: TRY004
        name, tier, cols = (entry.get(k) for k in ("name", "tier", "cols"))
        if name not in ("hits", "main.hits") or name in seen_names:
            raise ValueError("pin names must be unique hits or main.hits")
        if tier not in ("gpu", "host"):
            raise ValueError("pin tier must be gpu or host")
        if (
            not isinstance(cols, list)
            or not cols
            or any(not isinstance(c, str) for c in cols)
        ):
            raise ValueError("cols must be a non-empty list of column names")
        if len(set(cols)) != len(cols) or set(cols) - set(columns):
            raise ValueError("unknown or duplicate pin columns")
        batch_bytes = entry.get("scan_task_batch_bytes")
        if batch_bytes is not None and (
            isinstance(batch_bytes, bool) or not isinstance(batch_bytes, int) or batch_bytes <= 0
        ):
            raise ValueError("scan_task_batch_bytes must be a positive integer")
        seen_names.add(name)
        covered.update(cols)
        resolved.append(
            {"name": name, "tier": tier, "cols": cols}
            | ({} if batch_bytes is None else {"scan_task_batch_bytes": batch_bytes})
        )
    if covered != set(columns):
        raise ValueError(
            f"pin layout missing queried columns: {sorted(set(columns) - covered)}"
        )
    for q, requested in QUERY_COLUMNS.items():
        if not any(set(requested) <= set(e["cols"]) for e in resolved):
            raise ValueError(f"query {q} is not servable by any single pin entry")
    return resolved


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def plan_hashes(directory):
    if not directory:
        return {}
    root = Path(directory)
    return {
        str(p.relative_to(root)): file_sha256(p)
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


def effective_config(source, output, gpu_usage_limit_bytes=None):
    """Mirror Sirius's config discovery, then load the private recorded copy."""
    if source is None:
        source = os.environ.get("SIRIUS_CONFIG_FILE")
    if source is None:
        source = next(
            (
                p
                for p in (
                    Path.cwd() / "sirius.yaml",
                    Path.home() / ".sirius/sirius.yaml",
                )
                if p.exists()
            ),
            None,
        )
    text = Path(source).read_text() if source is not None else "sirius: {}\n"
    if gpu_usage_limit_bytes is not None:
        if gpu_usage_limit_bytes <= 0:
            raise ValueError("gpu-usage-limit-bytes must be positive")
        import yaml

        config = yaml.safe_load(text) or {}
        gpu = (
            config.setdefault("sirius", {})
            .setdefault("memory", {})
            .setdefault("gpu", {})
        )
        gpu.pop("usage_limit_fraction", None)
        gpu["usage_limit_bytes"] = gpu_usage_limit_bytes
        text = yaml.safe_dump(config, sort_keys=False)
    target = Path(output) / "sirius-config.yaml"
    target.write_text(text)
    return str(target.resolve())


def git_revision():
    """HEAD of the checkout, or the staged revision when the tree is a git archive
    (stage-rev.sh writes `head: <sha>` into SYNCED_FROM), else $SIRIUS_GIT_REV or
    'unknown'. Provenance only; it never gates a comparison."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (subprocess.CalledProcessError, OSError):
        pass
    synced = ROOT / "SYNCED_FROM"
    if synced.is_file():
        for line in synced.read_text().splitlines():
            if line.startswith("head:"):
                return line.split(":", 1)[1].strip()
    return os.environ.get("SIRIUS_GIT_REV", "unknown")


def perf_stack_env():
    keys = {
        "LD_PRELOAD",
        "SIRIUS_PRE_SQL",
        "SIRIUS_CONFIG_FILE",
        "SIRIUS_LOG_BACKEND",
        "SIRIUS_LOG_LEVEL",
    }
    keys.update(
        k
        for k in os.environ
        if k.startswith(("SIRIUS_EXP_", "SIRIUS_LATE_MAT", "SIRIUS_PIN_TIER"))
    )
    # Keep empty values: an exported-but-empty SIRIUS_EXP_* gate (e.g.
    # SIRIUS_EXP_FUSED_SCAN_FILTER="") enables the engine gate, unset disables it.
    return {k: os.environ[k] for k in sorted(keys) if k in os.environ}


def timed_query(con, statement, timeout_s):
    timer = threading.Timer(timeout_s, con.interrupt) if timeout_s else None
    if timer:
        timer.daemon = True
        timer.start()
    start = time.perf_counter()
    try:
        rows = con.execute(statement).fetchall()
        elapsed = time.perf_counter() - start
    finally:
        if timer:
            timer.cancel()
            timer.join()
    return elapsed, rows


def parse_profile(profile, wall_seconds):
    """Sum exclusive operator timings throughout the tree, including collectors.

    Sirius appears as one EXTENSION child; CPU fallback may have many children.
    Operator times on parallel CPU threads can exceed wall time, so the remainder
    is signed. optimizer_ext_ms is a subset of optimizer_ms, never added twice.
    Also accept the flat metrics emitted by DuckDB 1.x.
    """

    def milliseconds(value):
        value = float(value)
        if not math.isfinite(value) or value < 0:
            raise ValueError("invalid profiler timing")
        return value * 1000

    def operator_ms(nodes, timing_key):
        return sum(
            milliseconds(node[timing_key])
            + operator_ms(node.get("children", []), timing_key)
            for node in nodes
        )

    if "query" in profile:
        values = {
            "planner_ms": milliseconds(profile["planner"]["total_time"]),
            "optimizer_ms": milliseconds(profile["optimizer"]["total_time"]),
            "optimizer_ext_ms": milliseconds(profile["optimizer"].get("extension", 0)),
            "physical_ms": milliseconds(profile["physical_planner"]["total_time"]),
            "exec_ms": operator_ms(profile["operator"], "timing"),
            "profiler_total_ms": milliseconds(profile["query"]["total_time"]),
        }
    else:
        values = {
            "planner_ms": milliseconds(profile["planner"]),
            "optimizer_ms": milliseconds(profile["all_optimizers"]),
            "optimizer_ext_ms": milliseconds(profile.get("optimizer_extension", 0)),
            "physical_ms": milliseconds(profile["physical_planner"]),
            "exec_ms": operator_ms(profile["children"], "operator_timing"),
            "profiler_total_ms": milliseconds(profile["latency"]),
        }
    values["remainder_ms"] = wall_seconds * 1000 - sum(
        values[k] for k in ("planner_ms", "optimizer_ms", "physical_ms", "exec_ms")
    )
    return values


def profiled_query(con, statement, timeout_s, profile_path=None):
    """Keep profiler setup/teardown and JSON parsing outside the wall timer."""
    if profile_path is None:
        elapsed, rows = timed_query(con, statement, timeout_s)
        return elapsed, rows, dict.fromkeys(PROFILE_FIELDS)
    # Default SELECT coverage skips metadata-only COUNT's COLUMN_DATA_SCAN.
    con.execute("SET profiling_coverage = 'ALL'").fetchall()
    con.execute("PRAGMA enable_profiling = 'json'").fetchall()
    try:
        # DuckDB 2.x validates the suffix against the enabled profiler type.
        con.execute(f"SET profiling_output = {sql_literal(profile_path)}").fetchall()
        elapsed, rows = timed_query(con, statement, timeout_s)
    finally:
        con.execute("PRAGMA disable_profiling").fetchall()
    return (
        elapsed,
        rows,
        parse_profile(json.loads(profile_path.read_text()), elapsed),
    )


def compare_rows(cpu_rows, gpu_rows):
    if len(cpu_rows) != len(gpu_rows):
        return f"row count: CPU={len(cpu_rows)}, GPU={len(gpu_rows)}"
    for i, (cpu, gpu) in enumerate(zip(cpu_rows, gpu_rows)):
        if len(cpu) != len(gpu):
            return f"row {i}: column count differs"
        for j, (a, b) in enumerate(zip(cpu, gpu)):
            if (
                isinstance(a, float)
                and isinstance(b, (float, int))
                or isinstance(b, float)
                and isinstance(a, int)
            ):
                equal = (math.isnan(a) and math.isnan(b)) or math.isclose(
                    a, b, rel_tol=1e-4, abs_tol=1e-6
                )
            else:
                equal = a == b
            if not equal:
                return f"row {i}, column {j}: CPU={a!r}, GPU={b!r}"
    return None


def stage_plans(source, output, layout):
    if not source:
        return None
    plan = Path(source, "hits.txt").read_text()
    destination = Path(output, "effective_plans")
    destination.mkdir()
    for entry in layout:
        (destination / f"{entry['name']}.txt").write_text(plan)
    return str(destination.resolve())


def prepare_table(spec):
    """Create/checkpoint the native source before either worker or any wall timer.

    An existing main.hits is reused verbatim, even if the Parquet input changed.
    The caller must choose another database path to load another dataset.
    Record its catalog in spec for the worker identity check and qualified oracle.
    """
    with duckdb.connect(database=spec["duckdb_path"]) as con:
        spec["table_catalog"] = con.execute("SELECT current_database()").fetchone()[0]
        con.execute(f"SET memory_limit = {sql_literal(spec['duckdb_memory_limit'])}")
        table = con.execute(
            "SELECT table_type FROM information_schema.tables "
            "WHERE table_catalog = current_database() AND table_schema = 'main' "
            "AND table_name = 'hits'"
        ).fetchone()
        if table is not None and table[0] != "BASE TABLE":
            raise ValueError("table source requires a native main.hits base table")
        created = table is None
        if created:
            con.execute(
                f"CREATE TABLE hits AS FROM read_parquet({sql_literal(spec['parquet'])})"
            )
        con.execute("CHECKPOINT")
    return created


def source_database(spec):
    return (
        spec["duckdb_path"] if spec.get("source", "parquet") == "table" else ":memory:"
    )


def query_sql(spec, q, *, oracle=False):
    source = spec.get("source", "parquet")
    return render_query(
        QUERIES[q],
        spec["parquet"],
        source,
        spec["table_catalog"] if oracle and source == "table" else None,
    )


def verify_table_identity(con, spec):
    """Check native source resolution after pre-SQL, before pins or query timers.

    DuckDB searches temp.main first, then the current catalog/schema. With that
    default at the prepared main schema and its persistent hits present, both
    hits and main.hits resolve there regardless of later search-path entries.
    """
    catalog = spec["table_catalog"]
    # Every built-in is catalog-qualified: a user macro defined by pre-SQL
    # resolves before the system function of the same name and could otherwise
    # supply the evidence this guard relies on.
    context = con.execute(
        "SELECT system.main.current_database(), system.main.current_schema(), "
        "system.main.current_setting('search_path')"
    ).fetchone()
    tables = con.execute(
        "SELECT d.path FROM system.main.duckdb_tables() t "
        "JOIN system.main.duckdb_databases() d ON t.database_oid = d.database_oid "
        "WHERE t.database_name = ? AND t.schema_name = 'main' "
        "AND system.main.lower(t.table_name) = 'hits' AND NOT t.temporary "
        "AND NOT d.internal AND d.type = 'duckdb'",
        [catalog],
    ).fetchall()
    shadows = con.execute(
        "SELECT database_name, schema_name, table_name AS name "
        "FROM system.main.duckdb_tables() "
        "WHERE temporary AND system.main.lower(table_name) = 'hits' "
        "UNION ALL "
        "SELECT database_name, schema_name, view_name AS name "
        "FROM system.main.duckdb_views() "
        "WHERE temporary AND system.main.lower(view_name) = 'hits'"
    ).fetchall()
    path = Path(spec["duckdb_path"]).resolve()
    if (
        context[:2] != (catalog, "main")
        or len(tables) != 1
        or not tables[0][0]
        or Path(tables[0][0]).resolve() != path
        or shadows
    ):
        raise ValueError(
            "native table source identity mismatch after SIRIUS_PRE_SQL: "
            "hits and main.hits must both resolve to persistent "
            f"{sql_identifier(catalog)}.main.hits in {path}; "
            f"current catalog/schema/search_path={context!r}, "
            f"persistent table paths={tables!r}, temporary hits={shadows!r}"
        )


def open_benchmark_db(spec):
    con = duckdb.connect(
        database=source_database(spec),
        config={"allow_unsigned_extensions": "true"},
    )
    con.execute(f"SET memory_limit = {sql_literal(spec['duckdb_memory_limit'])}")
    con.execute(f"LOAD {sql_literal(spec['extension'])}")
    pre_sql = os.environ.get("SIRIUS_PRE_SQL", "").strip()
    if pre_sql:
        con.execute(pre_sql).fetchall()
    con.execute("SET gpu_execution = true")
    con.execute("SET enable_duckdb_fallback = true")
    con.execute(
        f"SET pin_table_compression = {'true' if spec['effective_plan_dir'] else 'false'}"
    )
    if spec["effective_plan_dir"]:
        con.execute(
            f"SET pin_table_input_compression_plan_dir = {sql_literal(spec['effective_plan_dir'])}"
        )
    # Pre-SQL must not enable profiling for pins/markers or --no-profile runs.
    con.execute("PRAGMA disable_profiling")
    if spec.get("profile", False):
        con.execute("SET profiling_mode = 'detailed'")
        con.execute("PRAGMA disable_profiling")
    return con


def pin_tables(con, spec):
    pins = []
    for entry in spec["layout"]:
        cols = ", ".join(sql_literal(c) for c in entry["cols"])
        source = (
            "format = 'duckdb', "
            if spec.get("source", "parquet") == "table"
            else f"{sql_literal(spec['parquet'])}, "
        )
        statement = (
            f"CALL pin_table({source}"
            f"name = {sql_literal(entry['name'])}, tier = {sql_literal(entry['tier'])}, cols = [{cols}])"
        )
        # A per-entry batch size pins the entry as a chosen number of chunks (gen_pin_layout
        # --chunks); the session setting goes back to its previous value for the queries.
        batch_bytes = entry.get("scan_task_batch_bytes")
        if batch_bytes is not None:
            previous = con.execute(
                "SELECT current_setting('scan_task_batch_size')"
            ).fetchone()[0]
            con.execute(f"SET scan_task_batch_size = {int(batch_bytes)}")
        try:
            seconds, _ = timed_query(con, statement, spec["timeout_s"])
        finally:
            if batch_bytes is not None:
                con.execute(f"SET scan_task_batch_size = {int(previous)}")
        pins.append(
            {
                "entry": entry["name"],
                "tier": entry["tier"],
                "cols": json.dumps(entry["cols"]),
                "seconds": seconds,
                "batch_bytes": "" if batch_bytes is None else batch_bytes,
            }
        )
    return pins


def cuda_profiler_library():
    """The CUDA runtime of the active environment, for cudaProfilerStart/Stop."""
    import ctypes

    prefix = os.environ.get("CONDA_PREFIX", sys.prefix)
    for pattern in ("targets/x86_64-linux/lib/libcudart.so.*", "lib/libcudart.so.*"):
        found = sorted(glob.glob(os.path.join(prefix, pattern)))
        if found:
            return ctypes.CDLL(found[-1])
    return ctypes.CDLL("libcudart.so")


def gpu_worker(spec_path):
    spec = json.loads(Path(spec_path).read_text())
    output = Path(spec["run_dir"])
    # scan_task_batch_size is one of Sirius's internal options, registered at LOAD only under
    # this switch. The engine also consults the switch per query (test injections, extra
    # counters), so it is cleared once the pins are in place and the queries run without it.
    batch_sized = any("scan_task_batch_bytes" in entry for entry in spec.get("layout", []))
    if batch_sized:
        os.environ["SIRIUS_ENABLE_TEST_OPTIONS"] = "1"
    con = open_benchmark_db(spec)
    try:
        if spec.get("source", "parquet") == "table":
            verify_table_identity(con, spec)
        pins = pin_tables(con, spec)
        if batch_sized:
            del os.environ["SIRIUS_ENABLE_TEST_OPTIONS"]
        write_csv(
            output / "pins.csv", ["entry", "tier", "cols", "seconds", "batch_bytes"], pins
        )
        if spec.get("profile", False):
            (output / "profiles").mkdir()
        repetitions, stored_rows = [], {}
        # Under `nsys profile --capture-range=cudaProfilerApi`, capture only the query loop, not the pins.
        cudart = cuda_profiler_library() if os.environ.get("CLICKBENCH_CUDA_PROFILER") == "1" else None
        if cudart is not None:
            cudart.cudaProfilerStart()
        for q in spec["queries"]:
            statement = query_sql(spec, q)
            for rep in range(spec["hot_reps"] + 1):
                con.execute(
                    f"SET VARIABLE clickbench_marker = 'clickbench_begin_q{q}_r{rep}'"
                ).fetchall()
                profile_path = (
                    output / "profiles" / f"q{q}_r{rep}.json"
                    if spec.get("profile", False)
                    else None
                )
                elapsed, rows, breakdown = profiled_query(
                    con, statement, spec["timeout_s"], profile_path
                )
                con.execute(
                    f"SET VARIABLE clickbench_marker = 'clickbench_end_q{q}_r{rep}'"
                ).fetchall()
                repetitions.append(
                    dict(
                        query=q,
                        rep=rep,
                        phase="cold" if rep == 0 else "hot",
                        seconds=elapsed,
                        rows=len(rows),
                        **breakdown,
                    )
                )
                if spec["validate"]:
                    stored_rows[q, rep] = rows
        if cudart is not None:
            cudart.cudaProfilerStop()
        write_json(output / "_repetitions.json", repetitions)
        if spec["validate"]:
            with (output / "_gpu_rows.pickle").open("wb") as f:
                pickle.dump(stored_rows, f, protocol=pickle.HIGHEST_PROTOCOL)
    finally:
        con.close()
    # The parent waits for this process to EXIT, not merely unpin/close.


def validation_worker(spec_path):
    spec = json.loads(Path(spec_path).read_text())
    output = Path(spec["run_dir"])
    with (output / "_gpu_rows.pickle").open("rb") as f:
        stored_rows = pickle.load(f)
    verdicts = {}
    with duckdb.connect(
        database=source_database(spec),
        read_only=spec.get("source", "parquet") == "table",
    ) as con:
        con.execute(f"SET memory_limit = {sql_literal(spec['duckdb_memory_limit'])}")
        for q in spec["queries"]:
            _, cpu_rows = timed_query(
                con, query_sql(spec, q, oracle=True), spec["timeout_s"]
            )
            for rep in range(spec["hot_reps"] + 1):
                verdicts[f"{q}:{rep}"] = compare_rows(cpu_rows, stored_rows[q, rep])
    write_json(output / "_validation.json", verdicts)


def parse_decode_line(line):
    if "[compression] decode " not in line:
        return None
    match = re.search(
        r"\[compression\] decode path=(device|host) cols=(\d+) rows=(\d+) ms=(\d+\.\d+)(?:\s|$)",
        line,
    )
    if match is None:
        raise ValueError(f"malformed decode record: {line}")
    path, cols, rows, ms = match.groups()
    return {"path": path, "cols": int(cols), "rows": int(rows), "ms": float(ms)}


def audit_log(log_text, expected):
    """Associate INFO planning and WARN runtime fallback with sentinel windows.

    Nested/internal QueryBegin records leave the outer sentinel window intact.
    Positive execution evidence is required; a missing log is not a clean audit.
    """
    windows, active = {}, None
    marker = re.compile(r"clickbench_(begin|end)_q(\d+)_r(\d+)")
    for line in log_text.splitlines():
        match = marker.search(line) if "QueryBegin:" in line else None
        if match:
            kind, q, rep = match.groups()
            key = int(q), int(rep)
            if kind == "begin":
                if active is not None or key in windows:
                    raise ValueError("overlapping/duplicate audit window")
                windows[key] = {
                    "fallback": False,
                    "gpu": False,
                    "ended": False,
                    "assignments": set(),
                    "decode_ms_sum": 0.0,
                    "decode_calls": 0,
                }
                active = key
            else:
                if active != key:
                    raise ValueError("audit window end without matching begin")
                windows[key]["ended"] = True
                active = None
        elif active is not None:
            window = windows[active]
            if (
                "Transparent execution fallback" in line
                or "falling back to DuckDB CPU" in line
                or "SiriusExecuteQuery error:" in line
            ):
                window["fallback"] = True
            if "Transparent GPU execution: executing query" in line:
                window["gpu"] = True
            assignment = re.search(
                r"assigned pinned entry '([^']+)' to operator '[^']+' tier=(\w+) columns=(\d+)",
                line,
            )
            if assignment:
                window["assignments"].add((assignment[1], assignment[2].lower()))
            decode = parse_decode_line(line)
            if decode is not None:
                window["decode_ms_sum"] += decode["ms"]
                window["decode_calls"] += 1
    if set(windows) != set(expected) or any(not w["ended"] for w in windows.values()):
        raise ValueError("incomplete Sirius INFO log: missing repetition sentinels")
    return {
        key: {
            "fallback": w["fallback"] or not w["gpu"],
            "disk": not w["assignments"] and not w["fallback"] and w["gpu"],
            "served_by": [
                {"entry": name, "tier": tier} for name, tier in sorted(w["assignments"])
            ]
            or [
                {
                    "entry": None,
                    "tier": "disk" if w["gpu"] and not w["fallback"] else "cpu",
                }
            ],
            "audit": (
                "cpu_fallback"
                if w["fallback"]
                else "gpu" if w["gpu"] else "cpu_or_unobserved"
            ),
            "decode_ms_sum": w["decode_ms_sum"],
            "decode_calls": w["decode_calls"],
        }
        for key, w in windows.items()
    }


def footprint_rows(layout, log_text=""):
    """Sum retained chunks, never disk bytes or table totals, by entry/tier/column.

    Unknown/missing column records stay blank, including partial log coverage.
    Resident payload excludes headers, pool rounding and query workspace.
    """
    pattern = re.compile(
        r"\[pin_table\] resident entry='([^']+)' tier=(\w+) batch=(\d+) "
        r"column=(\S+) resident_bytes=(\d+) uncompressed_bytes=(\d+) compressed=(?:true|false|0|1)"
    )
    totals, seen, batches = {}, set(), {}
    for match in pattern.finditer(log_text):
        entry, tier, batch, col, resident, raw = match.groups()
        key = (entry, tier.lower(), col)
        if (*key, batch) in seen:
            raise ValueError(f"duplicate resident record: {key}, batch {batch}")
        seen.add((*key, batch))
        batches.setdefault((entry, tier.lower()), set()).add(batch)
        values = totals.setdefault(key, [0, 0])
        values[0] += int(resident)
        values[1] += int(raw)
    for key in totals:
        entry_batches = batches[key[:2]]
        if entry_batches != {str(i) for i in range(len(entry_batches))} or any(
            (*key, batch) not in seen for batch in entry_batches
        ):
            totals[key] = ["", ""]
    return [
        {
            "column": col,
            "tier": e["tier"],
            "entry": e["name"],
            "resident_bytes": totals.get((e["name"], e["tier"], col), ["", ""])[0],
            "uncompressed_bytes": totals.get((e["name"], e["tier"], col), ["", ""])[1],
        }
        for e in layout
        for col in e["cols"]
    ]


def pin_diagnostics(log_text):
    tags = (
        "[pin_table]",
        "[materialize_all_batches_compressed]",
        "[materialize_pin_to_host]",
    )
    warnings = [
        line
        for line in log_text.splitlines()
        if any(tag in line for tag in tags)
        and re.search(r"\[(?:warning|warn)\]", line, re.IGNORECASE)
    ]
    rejected_plan = re.compile(
        r"\[pin_table\].*(?:pinning uncompressed|"
        r"cannot (?:inspect|open|read) plan file|cannot scan plan dir|"
        r"plan file .* is empty)"
    )
    return {
        "pin_warnings": warnings,
        "compression_failed": any(
            ("compression failed:" in line and any(tag in line for tag in tags[1:]))
            or rejected_plan.search(line)
            for line in warnings
        ),
    }


def exclusion_reasons(results):
    reasons = {}
    for r in results:
        why = []
        if r["fallback"]:
            why.append("fallback")
        if r["disk"]:
            why.append("disk")
        if r["validated"] is False:
            why.append("validation failed")
        if why:
            reasons[r["query"]] = why
    return reasons


def geomean(values):
    values = list(values)
    return math.exp(statistics.mean(math.log(v) for v in values)) if values else None


def summarize(repetitions, verdicts):
    results = []
    for q in sorted({r["query"] for r in repetitions}):
        reps = [r for r in repetitions if r["query"] == q]
        hot = [r["seconds"] for r in reps if r["rep"] > 0]
        results.append(
            dict(
                query=q,
                cold_s=reps[0]["seconds"],
                hot_median_s=statistics.median(hot),
                hot_min_s=min(hot),
                hot_mean_s=statistics.mean(hot),
                hot_std_s=statistics.stdev(hot) if len(hot) > 1 else 0.0,
                rows=reps[0]["rows"],
                fallback=any(r["fallback"] for r in reps),
                disk=any(r["disk"] for r in reps),
                served_by=json.dumps(
                    [
                        {"entry": name, "tier": tier}
                        for name, tier in sorted(
                            {
                                (a["entry"], a["tier"])
                                for r in reps
                                for a in r["served_by"]
                            },
                            key=lambda pair: (pair[1], pair[0] or ""),
                        )
                    ]
                ),
                validated=(
                    all(verdicts[f"{q}:{r['rep']}"] is None for r in reps)
                    if verdicts is not None
                    else ""
                ),
                **{
                    field: (
                        statistics.median(r[field] for r in reps if r["rep"] > 0)
                        if all(r.get(field) is not None for r in reps if r["rep"] > 0)
                        else None
                    )
                    for field in (*PROFILE_FIELDS, *DECODE_FIELDS)
                },
            )
        )
    return results


def run_child(mode, spec_path, env, console):
    with Path(console).open("w") as log:
        subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                mode,
                str(spec_path),
            ],
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )


def write_run_info(output, spec):
    write_json(output / "_spec.json", spec)
    (output / "run_info.txt").write_text(json.dumps(spec, indent=2) + "\n")
    (output / "effective_queries.sql").write_text(
        "\n".join(query_sql(spec, q) for q in spec["queries"]) + "\n"
    )


def run(args):
    output = Path(args.output, args.lane).resolve()
    layout = load_pin_layout(args.pin_layout)
    output.mkdir(parents=True, exist_ok=False)
    logs = output / "log_dir"
    logs.mkdir()
    env = dict(
        os.environ,
        SIRIUS_LOG_DIR=str(logs),
        SIRIUS_LOG_BACKEND="spdlog",
        SIRIUS_LOG_LEVEL="info",
    )
    env["SIRIUS_CONFIG_FILE"] = effective_config(
        args.config, output, args.gpu_usage_limit_bytes
    )
    spec = vars(args).copy()
    spec.setdefault("source", "parquet")
    spec.setdefault("profile", True)
    spec["duckdb_path"] = (
        str(Path(args.duckdb_path).expanduser().resolve())
        if spec["source"] == "table"
        else None
    )
    spec.update(
        run_dir=str(output),
        layout=layout,
        effective_plan_dir=stage_plans(args.plan_dir, output, layout),
        parquet=os.path.abspath(os.path.expanduser(args.parquet)),
        plan_dir=str(Path(args.plan_dir).resolve()) if args.plan_dir else None,
        extension=str(Path(args.extension).resolve(strict=True)),
        extension_sha256=file_sha256(args.extension),
        config_sha256=file_sha256(env["SIRIUS_CONFIG_FILE"]),
        config_yaml=Path(env["SIRIUS_CONFIG_FILE"]).read_text(),
        pre_sql=env.get("SIRIUS_PRE_SQL", "").strip(),
        plan_hashes=plan_hashes(args.plan_dir),
        config=env.get("SIRIUS_CONFIG_FILE"),
        git_rev=git_revision(),
        perf_stack_env={
            **perf_stack_env(),
            **{k: env[k] for k in ("SIRIUS_LOG_BACKEND", "SIRIUS_LOG_LEVEL")},
        },
    )
    if spec["config"]:
        spec["perf_stack_env"]["SIRIUS_CONFIG_FILE"] = spec["config"]
    spec["duckdb_table_created"] = (
        prepare_table(spec) if spec["source"] == "table" else None
    )
    write_run_info(output, spec)
    spec_path = output / "_spec.json"
    run_child("--gpu-worker", spec_path, env, output / "gpu_worker.log")
    repetitions = json.loads((output / "_repetitions.json").read_text())
    log_text = "\n".join(
        p.read_text(errors="replace") for p in sorted(logs.glob("*.log"))
    )
    audit = audit_log(log_text, [(r["query"], r["rep"]) for r in repetitions])
    for rep in repetitions:
        rep.update(audit[rep["query"], rep["rep"]])
        for field in PROFILE_FIELDS:
            rep.setdefault(field, None)
    verdicts = None
    if args.validate:
        cpu_env = {k: v for k, v in env.items() if not k.startswith("SIRIUS_")}
        run_child(
            "--validate-worker",
            spec_path,
            cpu_env,
            output / "validation_worker.log",
        )
        verdicts = json.loads((output / "_validation.json").read_text())
    for rep in repetitions:
        rep["validated"] = (
            verdicts[f"{rep['query']}:{rep['rep']}"] is None
            if verdicts is not None
            else ""
        )
    results = summarize(repetitions, verdicts)
    write_json(output / "_repetitions.json", repetitions)
    excluded = exclusion_reasons(results)
    footprints = footprint_rows(layout, log_text)
    write_csv(
        output / "repetitions.csv",
        [
            "query",
            "rep",
            "phase",
            "seconds",
            "rows",
            "fallback",
            "disk",
            "served_by",
            "audit",
            "validated",
            *PROFILE_FIELDS,
            *DECODE_FIELDS,
        ],
        [{**r, "served_by": json.dumps(r["served_by"])} for r in repetitions],
    )
    write_csv(output / "results.csv", list(results[0]), results)
    write_csv(
        output / "footprint.csv",
        ["column", "tier", "resident_bytes", "uncompressed_bytes", "entry"],
        footprints,
    )
    with (output / "pins.csv").open() as f:
        pins = list(csv.DictReader(f))
    summary = {
        **spec,
        "aggregation_validation": ("validated" if args.validate else "unvalidated"),
        "hot_geomean_s": geomean(
            r["hot_median_s"] for r in results if r["query"] not in excluded
        ),
        "fallback_queries": [r["query"] for r in results if r["fallback"]],
        "disk_queries": [r["query"] for r in results if r["disk"]],
        "excluded_queries": excluded,
        "served_tier_counts": {
            tier: sum(
                any(a["tier"] == tier for a in r["served_by"]) for r in repetitions
            )
            for tier in ("gpu", "host", "disk", "cpu")
        },
        "effective_plan_hashes": plan_hashes(spec["effective_plan_dir"]),
        "pin_seconds": sum(float(p["seconds"]) for p in pins),
        "footprint_status": (
            "measured"
            if all(r["resident_bytes"] != "" for r in footprints)
            else "incomplete resident byte logs"
        ),
        "compression_coverage": [
            line for line in log_text.splitlines() if "chunk(s) compressed" in line
        ],
        **pin_diagnostics(log_text),
        "validation_failures": {
            k: v for k, v in (verdicts or {}).items() if v is not None
        },
    }
    write_json(output / "summary.json", summary)
    print(
        json.dumps(
            {
                k: summary[k]
                for k in (
                    "lane",
                    "hot_geomean_s",
                    "aggregation_validation",
                    "fallback_queries",
                    "excluded_queries",
                    "compression_failed",
                    "pin_seconds",
                    "footprint_status",
                    "validation_failures",
                )
            },
            indent=2,
        )
    )
    return 1 if summary["validation_failures"] else 0


def main():
    if len(sys.argv) == 3 and sys.argv[1] in (
        "--gpu-worker",
        "--validate-worker",
    ):
        (gpu_worker if sys.argv[1] == "--gpu-worker" else validation_worker)(
            sys.argv[2]
        )
        return 0
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--parquet", required=True)
    parser.add_argument("--source", choices=("parquet", "table"), default="parquet")
    parser.add_argument("--duckdb-path", help="persistent database for --source table")
    parser.add_argument(
        "--no-profile", dest="profile", action="store_false", default=True
    )
    parser.add_argument("--lane", required=True)
    parser.add_argument("--plan-dir")
    parser.add_argument("--pin-layout")
    parser.add_argument("--hot-reps", type=int, default=10)
    parser.add_argument("--queries", type=parse_queries, default=list(QUERIES))
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--validate", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--timeout-s", type=float, default=600)
    parser.add_argument("--config", default=os.environ.get("SIRIUS_CONFIG_FILE"))
    parser.add_argument("--gpu-usage-limit-bytes", type=int)
    parser.add_argument("--duckdb-memory-limit", default="32GB")
    parser.add_argument(
        "--extension",
        default=os.environ.get("SIRIUS_EXTENSION_PATH", str(EXTENSION)),
    )
    args = parser.parse_args()
    if args.source == "table" and (
        not args.duckdb_path or args.duckdb_path == ":memory:"
    ):
        parser.error("--source table requires --duckdb-path to a persistent database")
    if args.hot_reps < 1 or not math.isfinite(args.timeout_s) or args.timeout_s <= 0:
        parser.error("hot-reps must be >= 1 and timeout-s must be finite and positive")
    if args.gpu_usage_limit_bytes is not None and args.gpu_usage_limit_bytes <= 0:
        parser.error("gpu-usage-limit-bytes must be positive")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", args.lane):
        parser.error("lane must be a simple directory name")
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
