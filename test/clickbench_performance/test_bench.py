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

import argparse
import csv
import json
import os
import pickle
from pathlib import Path

import clickbench_bench as bench
import pytest
from clickbench_queries import QUERIES, parquet_query

import duckdb


def test_query_subsets():
    assert bench.parse_queries("0,5-7,42,5") == [0, 5, 6, 7, 42]
    for value in ("", "43", "1-0", "-1", "1-2-3", "x"):
        with pytest.raises(argparse.ArgumentTypeError):
            bench.parse_queries(value)


def test_ordered_comparison_and_float_tolerance():
    assert bench.compare_rows([(1.0, "a", None)], [(1.00001, "a", None)]) is None
    assert (
        bench.compare_rows(
            [(float("nan"), float("inf"))], [(float("nan"), float("inf"))]
        )
        is None
    )
    assert bench.compare_rows([(1,), (2,)], [(2,), (1,)]) is not None
    assert bench.compare_rows([(1,)], []) is not None
    assert bench.compare_rows([(1,)], [(1, 2)]) is not None
    assert bench.compare_rows([(float("inf"),)], [(1.0,)]) is not None


def log_window(q, rep, body):
    return (
        f"[info] QueryBegin: instance=0x01 connection=1 query=1 SQL: SET VARIABLE clickbench_marker = 'clickbench_begin_q{q}_r{rep}'\n"
        f"[info] QueryBegin: SQL: SELECT COUNT(*) FROM read_parquet('hits.parquet')\n{body}\n"
        f"[info] QueryBegin: SQL: SET VARIABLE clickbench_marker = 'clickbench_end_q{q}_r{rep}'\n"
    )


def test_fallback_audit_catches_planning_runtime_and_unobserved():
    log = log_window(0, 0, "[info] Transparent GPU execution: executing query")
    log += log_window(0, 1, "[info] Transparent execution fallback (unsupported): op")
    log += log_window(
        1,
        0,
        "[info] Transparent GPU execution: executing query\n[warning] Transparent execution: GPU execution failed at runtime; falling back to DuckDB CPU within the same transaction. GPU error: OOM\n[info] Transparent GPU execution: query completed",
    )
    log += log_window(1, 1, "[info] query optimized to constant")
    audit = bench.audit_log(log, [(0, 0), (0, 1), (1, 0), (1, 1)])
    assert [audit[k]["fallback"] for k in audit] == [False, True, True, True]
    assert audit[1, 1]["audit"] == "cpu_or_unobserved"
    with pytest.raises(ValueError, match="incomplete"):
        bench.audit_log("", [(0, 0)])
    with pytest.raises(ValueError):
        bench.audit_log(log + log, [(0, 0), (0, 1), (1, 0), (1, 1)])


class Connection:
    def __init__(self):
        self.statements = []

    def execute(self, statement):
        self.statements.append(statement)
        return self

    def fetchall(self):
        return []

    def interrupt(self):
        pass


def test_setup_and_pin_sql(monkeypatch):
    con = Connection()
    monkeypatch.setattr(bench.duckdb, "connect", lambda **kwargs: con)
    monkeypatch.setenv(
        "SIRIUS_PRE_SQL", "SET threads=4; SET pin_table_compression=true"
    )
    spec = {
        "duckdb_memory_limit": "2GB",
        "extension": "sirius.duckdb_extension",
        "effective_plan_dir": None,
        "layout": [{"name": "main.hits", "tier": "host", "cols": ["URL", "Title"]}],
        "parquet": "/tmp/hit's/*.parquet",
        "timeout_s": 0,
    }
    assert bench.open_benchmark_db(spec) is con
    assert "SET pin_table_compression = false" in con.statements
    assert con.statements[-1] == "PRAGMA disable_profiling"
    pins = bench.pin_tables(con, spec)
    assert (
        "pin_table('/tmp/hit''s/*.parquet', name = 'main.hits', tier = 'host', cols = ['URL', 'Title'])"
        in con.statements[-1]
    )
    assert pins[0]["seconds"] >= 0
    assert json.loads(pins[0]["cols"]) == ["URL", "Title"]


def test_private_plan_aliases(tmp_path):
    source, output = tmp_path / "source", tmp_path / "output"
    source.mkdir()
    output.mkdir()
    (source / "hits.txt").write_text("input -> identity\n")
    target = Path(bench.stage_plans(source, output, bench.load_pin_layout()))
    assert (target / "hits.txt").read_text() == "input -> identity\n"
    # Two exact names require identical plan files, not an uncompressed host pin.
    output2 = tmp_path / "output2"
    output2.mkdir()
    target = Path(
        bench.stage_plans(source, output2, [{"name": "hits"}, {"name": "main.hits"}])
    )
    assert (target / "main.hits.txt").read_text() == (target / "hits.txt").read_text()


def test_validation_child_and_every_repetition(tmp_path, tiny_parquet):
    qs = [0, 2, 5, 23, 28]
    spec = {
        "parquet": str(tiny_parquet),
        "run_dir": str(tmp_path),
        "queries": qs,
        "hot_reps": 2,
        "timeout_s": 10,
        "duckdb_memory_limit": "1GB",
    }
    stored = {}
    with duckdb.connect() as con:
        for q in qs:
            rows = con.execute(parquet_query(QUERIES[q], tiny_parquet)).fetchall()
            for rep in range(3):
                stored[q, rep] = rows
    stored[0, 1] = [(99,)]
    with (tmp_path / "_gpu_rows.pickle").open("wb") as f:
        pickle.dump(stored, f)
    path = tmp_path / "_spec.json"
    bench.write_json(path, spec)
    env = {k: v for k, v in os.environ.items() if not k.startswith("SIRIUS_")}
    bench.run_child("--validate-worker", path, env, tmp_path / "validation_worker.log")
    verdicts = json.loads((tmp_path / "_validation.json").read_text())
    assert len(verdicts) == len(qs) * 3
    assert [k for k, v in verdicts.items() if v is not None] == ["0:1"]


def test_summary_cold_hot_and_sample_stddev():
    reps = [
        {
            "query": 2,
            "rep": i,
            "seconds": x,
            "rows": 1,
            "fallback": i == 0,
            "disk": False,
            "served_by": [{"entry": "hits", "tier": "gpu"}],
        }
        for i, x in enumerate((100, 1, 2, 3))
    ]
    result = bench.summarize(reps, None)[0]
    assert result == dict(
        query=2,
        cold_s=100,
        hot_median_s=2,
        hot_min_s=1,
        hot_mean_s=2,
        hot_std_s=1,
        rows=1,
        fallback=True,
        disk=False,
        served_by=json.dumps([{"entry": "hits", "tier": "gpu"}]),
        validated="",
        **dict.fromkeys((*bench.PROFILE_FIELDS, *bench.DECODE_FIELDS)),
    )
    assert bench.geomean([1, 4]) == 2
    assert bench.geomean([]) is None


@pytest.mark.parametrize("failure", ["fallback", "disk", "validation"])
@pytest.mark.parametrize("source", ["parquet", "table-created", "table-reused"])
def test_runner_outputs_and_child_order(
    tmp_path, tiny_parquet, monkeypatch, failure, source
):
    calls = []

    def child(mode, spec_path, env, console):
        calls.append(mode)
        spec = json.loads(Path(spec_path).read_text())
        output = Path(spec["run_dir"])
        if mode == "--gpu-worker":
            assert env["SIRIUS_LOG_LEVEL"] == "info"
            assert env["SIRIUS_LOG_BACKEND"] == "spdlog"
            bench.write_csv(
                output / "pins.csv",
                ["entry", "tier", "cols", "seconds"],
                [{"entry": "hits", "tier": "gpu", "cols": '["URL"]', "seconds": 7}],
            )
            reps = [
                {
                    "query": q,
                    "rep": i,
                    "phase": "cold" if i == 0 else "hot",
                    "seconds": 10 if i == 0 else i,
                    "rows": 1,
                }
                for q in (0, 1)
                for i in range(3)
            ]
            bench.write_json(output / "_repetitions.json", reps)
            text = "".join(
                log_window(
                    r["query"],
                    r["rep"],
                    (
                        "[info] Transparent execution fallback: unsupported"
                        if r["query"] == 1 and failure == "fallback"
                        else (
                            "[info] Transparent GPU execution: executing query"
                            if r["query"] == 1 and failure == "disk"
                            else "[info] Transparent GPU execution: executing query\n[info] [sirius_scan_manager] assigned pinned entry 'hits' to operator '1' tier=GPU columns=105"
                        )
                    )
                    + f"\n[info] [compression] decode path=device cols=1 rows=1 ms={r['rep'] + .25:.3f}",
                )
                for r in reps
            )
            (output / "log_dir/sirius.log").write_text(text)
        else:
            assert calls == ["--gpu-worker", "--validate-worker"]
            assert not any(k.startswith("SIRIUS_") for k in env)
            bench.write_json(
                output / "_validation.json",
                {
                    f"{q}:{i}": (
                        "wrong result"
                        if failure == "validation" and (q, i) == (1, 1)
                        else None
                    )
                    for q in (0, 1)
                    for i in range(3)
                },
            )

    monkeypatch.setattr(bench, "run_child", child)
    extension = tmp_path / "sirius.duckdb_extension"
    extension.write_bytes(b"fixture extension")
    source_config = tmp_path / "source.yaml"
    source_config.write_text(
        "sirius:\n  memory:\n    gpu:\n      usage_limit_fraction: 0.8\n"
    )
    args = argparse.Namespace(
        output=str(tmp_path),
        lane="onpair",
        pin_layout=None,
        plan_dir=None,
        parquet=str(tiny_parquet),
        source="parquet" if source == "parquet" else "table",
        duckdb_path=str(tmp_path / "native.duckdb"),
        queries=[0, 1],
        hot_reps=2,
        timeout_s=10,
        validate=True,
        config=str(source_config),
        extension=str(extension),
        gpu_usage_limit_bytes=123456,
        duckdb_memory_limit="1GB",
    )
    if source == "table-reused":
        assert bench.prepare_table(vars(args)) is True
    assert bench.run(args) == (1 if failure == "validation" else 0)
    summary = json.loads((tmp_path / "onpair/summary.json").read_text())
    created = None if source == "parquet" else source == "table-created"
    assert summary["source"] == args.source
    assert summary["duckdb_table_created"] is created
    info = json.loads((tmp_path / "onpair/run_info.txt").read_text())
    assert info["duckdb_table_created"] is created
    assert info["duckdb_path"] == (None if source == "parquet" else args.duckdb_path)
    reps = json.loads((tmp_path / "onpair/_repetitions.json").read_text())
    assert [r["decode_ms_sum"] for r in reps] == [0.25, 1.25, 2.25] * 2
    assert all(r["decode_calls"] == 1 for r in reps)
    with (tmp_path / "onpair/results.csv").open() as f:
        results = list(csv.DictReader(f))
    assert all(float(r["decode_ms_sum"]) == 1.75 for r in results)
    assert all(float(r["decode_calls"]) == 1 for r in results)
    assert summary["fallback_queries"] == ([1] if failure == "fallback" else [])
    assert summary["excluded_queries"] == {
        "1": ["validation failed" if failure == "validation" else failure]
    }
    assert summary["extension_sha256"] == bench.file_sha256(extension)
    assert summary["config_sha256"] == bench.file_sha256(summary["config"])
    assert summary["config_yaml"] == Path(summary["config"]).read_text()
    assert summary["gpu_usage_limit_bytes"] == 123456
    assert "usage_limit_bytes: 123456" in summary["config_yaml"]
    assert "usage_limit_fraction" not in summary["config_yaml"]
    assert "usage_limit_fraction" in source_config.read_text()
    assert summary["served_tier_counts"]["gpu"] == (6 if failure == "validation" else 3)
    assert summary["hot_geomean_s"] == 1.5
    assert summary["pin_seconds"] == 7
    for name in (
        "results.csv",
        "repetitions.csv",
        "footprint.csv",
        "run_info.txt",
        "effective_queries.sql",
    ):
        assert (tmp_path / "onpair" / name).exists()


def test_worker_execution_with_plain_duckdb_substitute(
    tmp_path, tiny_parquet, monkeypatch
):
    """Exercise repetition storage and marker SQL without loading an extension."""
    monkeypatch.setattr(bench, "open_benchmark_db", lambda spec: duckdb.connect())
    monkeypatch.setattr(bench, "pin_tables", lambda con, spec: [])
    spec = {
        "parquet": str(tiny_parquet),
        "run_dir": str(tmp_path),
        "queries": [0, 23],
        "hot_reps": 2,
        "timeout_s": 10,
        "validate": True,
    }
    spec_path = tmp_path / "_spec.json"
    bench.write_json(spec_path, spec)
    bench.gpu_worker(spec_path)
    repetitions = json.loads((tmp_path / "_repetitions.json").read_text())
    assert [(r["query"], r["rep"], r["rows"]) for r in repetitions] == [
        (q, r, 1) for q in (0, 23) for r in range(3)
    ]
    with (tmp_path / "_gpu_rows.pickle").open("rb") as f:
        rows = pickle.load(f)
    assert len(rows[23, 0][0]) == 105
    assert rows[0, 2] == [(1,)]
