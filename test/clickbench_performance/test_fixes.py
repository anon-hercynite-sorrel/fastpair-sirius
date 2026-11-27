# Copyright 2026, Sirius Contributors.
# SPDX-License-Identifier: Apache-2.0

"""CPU regressions for the reviewed placement, accounting and comparison contract."""

import csv
import json
from pathlib import Path

import clickbench_bench as bench
import pytest
import yaml
from clickbench_queries import QUERY_COLUMNS, union_columns_by_table
from gen_pin_layout import generate_layout, read_footprint
from report import report
from test_bench import log_window
from test_tools import make_lane

ASSIGNED_GPU = "[info] [sirius_scan_manager] assigned pinned entry 'hits' to operator '7' tier=GPU columns=20"
ASSIGNED_HOST = "[info] [sirius_scan_manager] assigned pinned entry 'main.hits' to operator '8' tier=HOST columns=105"
EXECUTING = "[info] Transparent GPU execution: executing query\n"


def update_summary(path, **values):
    summary = json.loads((path / "summary.json").read_text())
    summary.update(values)
    bench.write_json(path / "summary.json", summary)


def update_result(path, q, **values):
    with (path / "results.csv").open() as f:
        records = list(csv.DictReader(f))
    for record in records:
        if int(record["query"]) == q:
            record.update(values)
    bench.write_csv(path / "results.csv", list(records[0]), records)


def test_mixed_layout_serves_each_query_from_one_entry(tmp_path):
    sizes = {col: i + 1 for i, col in enumerate(union_columns_by_table()["hits"])}
    layout = generate_layout(sizes, 200)
    gpu, host = (set(entry["cols"]) for entry in layout)
    gpu_queries = {q for q, cols in QUERY_COLUMNS.items() if set(cols) <= gpu}
    assert gpu_queries and gpu_queries != set(QUERY_COLUMNS)
    served_gpu = set()
    for q, cols in QUERY_COLUMNS.items():
        # Match the engine's GPU preference over a complete host superset.
        chosen = next(entry for entry in layout if set(cols) <= set(entry["cols"]))
        if chosen["tier"] == "gpu":
            served_gpu.add(q)
        assert set(cols) <= host
    assert served_gpu == gpu_queries
    path = tmp_path / "layout.json"
    bench.write_json(path, layout)
    assert bench.load_pin_layout(path) == layout
    # The old split has complete union coverage, but Q23 cannot use a single entry.
    split = [
        {"name": "hits", "tier": "gpu", "cols": sorted(gpu)},
        {"name": "main.hits", "tier": "host", "cols": sorted(host - gpu)},
    ]
    bench.write_json(path, split)
    with pytest.raises(ValueError, match="single pin entry"):
        bench.load_pin_layout(path)


@pytest.mark.parametrize("budget,tiers", [(0, ["host"]), (100000, ["gpu", "host"])])
def test_layout_extremes_keep_complete_host(budget, tiers):
    sizes = {col: 1 for col in union_columns_by_table()["hits"]}
    layout = generate_layout(sizes, budget)
    assert [entry["tier"] for entry in layout] == tiers
    assert set(layout[-1]["cols"]) == set(sizes)


def test_assignments_are_windowed_and_disk_is_excluded():
    log = ASSIGNED_HOST + "\n"  # Pin/planning outside a repetition must not leak in.
    log += log_window(1, 0, EXECUTING + ASSIGNED_GPU + "\n" + ASSIGNED_GPU)
    log += log_window(1, 1, EXECUTING + ASSIGNED_HOST)
    log += log_window(2, 0, EXECUTING)
    log += log_window(3, 0, EXECUTING + ASSIGNED_GPU + "\nfalling back to DuckDB CPU")
    audit = bench.audit_log(log, [(1, 0), (1, 1), (2, 0), (3, 0)])
    assert audit[1, 0]["served_by"] == [{"entry": "hits", "tier": "gpu"}]
    assert audit[1, 1]["served_by"] == [{"entry": "main.hits", "tier": "host"}]
    assert audit[2, 0]["disk"] and not audit[2, 0]["fallback"]
    assert audit[2, 0]["served_by"] == [{"entry": None, "tier": "disk"}]
    assert audit[3, 0]["fallback"] and not audit[3, 0]["disk"]


def test_exclusion_union_and_runner_geomean():
    results = [
        {
            "query": q,
            "fallback": q in (1, 4),
            "disk": q in (2, 4),
            "validated": q not in (3, 4),
            "hot_median_s": 2 if q == 0 else 999,
        }
        for q in range(5)
    ]
    excluded = bench.exclusion_reasons(results)
    assert excluded == {
        1: ["fallback"],
        2: ["disk"],
        3: ["validation failed"],
        4: ["fallback", "disk", "validation failed"],
    }
    assert (
        bench.geomean(r["hot_median_s"] for r in results if r["query"] not in excluded)
        == 2
    )


@pytest.mark.parametrize(
    "tag", ["materialize_all_batches_compressed", "materialize_pin_to_host"]
)
def test_actual_engine_codec_failure_warning(tag):
    warning = (
        f"[2026-11-05 12:34:56] [warning] [{tag}] compression failed: invalid plan; "
        "pinning the remainder of this table uncompressed (the failure latches compression "
        "off for every later chunk)"
    )
    result = bench.pin_diagnostics(warning + "\n[warning] unrelated query warning")
    assert result == {"pin_warnings": [warning], "compression_failed": True}
    assert not bench.pin_diagnostics(
        f"[info] [{tag}] pin 'hits': 1/2 chunk(s) compressed"
    )["compression_failed"]


@pytest.mark.parametrize(
    "message",
    [
        "'hits': compression was requested but pin_table_input_compression_plan_dir is empty; pinning uncompressed",
        "cannot inspect plan file '/plans/hits.txt': Permission denied",
        "cannot open plan file '/plans/hits.txt'",
        "cannot read plan file '/plans/hits.txt'",
        "plan file '/plans/hits.txt' is empty",
        "cannot scan plan dir '/plans': No such file or directory",
        "'hits': plan file does not cover all pinned columns; pinning uncompressed",
        "'hits': compression was requested but no readable nonempty plan file was found in '/plans'; pinning uncompressed",
    ],
)
def test_actual_engine_plan_rejection_warning(message, lanes):
    warning = f"[2026-11-05 12:34:56] [warning] [pin_table] {message}"
    diagnostics = bench.pin_diagnostics(warning)
    assert diagnostics == {"pin_warnings": [warning], "compression_failed": True}
    a, b = lanes
    update_summary(b, **diagnostics)
    with pytest.raises(ValueError, match="allow-compression-failure"):
        report([a, b])


def test_other_pin_warnings_do_not_imply_compression_failure():
    warning = "[warning] [pin_table] some unrelated warning"
    assert bench.pin_diagnostics(warning) == {
        "pin_warnings": [warning],
        "compression_failed": False,
    }


def test_footprint_parser_sums_retained_batches_by_entry_and_tier(tmp_path):
    layout = [
        {"name": "hits", "tier": "gpu", "cols": ["URL", "Title"]},
        {"name": "main.hits", "tier": "host", "cols": ["URL"]},
    ]
    lines = "[info] [pin_table] resident entry='hits' tier=gpu batch=0 column=URL resident_bytes=10 uncompressed_bytes=100 compressed=true\n[info] [pin_table] resident entry='hits' tier=gpu batch=1 column=URL resident_bytes=20 uncompressed_bytes=200 compressed=false\n[info] [pin_table] resident entry='hits' tier=gpu batch=0 column=Title resident_bytes=2 uncompressed_bytes=10 compressed=true\n[info] [pin_table] resident entry='hits' tier=gpu batch=1 column=Title resident_bytes=4 uncompressed_bytes=20 compressed=true\n[info] [pin_table] resident entry='main.hits' tier=host batch=0 column=URL resident_bytes=99 uncompressed_bytes=100 compressed=false\n[info] parquet disk bytes=10000"
    rows = bench.footprint_rows(layout, lines)
    assert [(r["resident_bytes"], r["uncompressed_bytes"]) for r in rows] == [
        (30, 300),
        (6, 30),
        (99, 100),
    ]
    path = tmp_path / "footprint.csv"
    bench.write_csv(path, list(rows[0]), rows[:2])
    assert read_footprint(path) == {"URL": 30, "Title": 6}
    with pytest.raises(ValueError, match="duplicate resident"):
        bench.footprint_rows(layout, lines + "\n" + lines)
    missing = bench.footprint_rows(layout, "")
    assert all(r["resident_bytes"] == "" for r in missing)
    partial = bench.footprint_rows(
        layout,
        "\n".join(
            line for line in lines.splitlines() if "batch=1 column=Title" not in line
        ),
    )
    assert partial[0]["resident_bytes"] == 30
    assert partial[1]["resident_bytes"] == ""


# Snapshot of a reference Sirius config for portable CPU tests.
G4_CONFIG = """# Engine config for the ClickBench harness on the G4 (RTX PRO 6000 Server Edition, 96 GB;
# g4-standard-48: 48 vCPU, 176 GB RAM). Identical in every lane. Derived from
# test/cpp/integration/integration.yaml with the pools sized for this box; the GPU usage limit
# is the budget knob the layout generator fills, overridden per run by the harness.
sirius:
  topology:
    num_gpus: 1
  memory:
    gpu:
      usage_limit_fraction: 0.9
      reservation_limit_fraction: 1.0
    host:
      capacity_bytes: 120000000000
      initial_number_pools: 8
      pool_size: 512
      block_size: 4194304
  executor:
    pipeline:
      num_threads: 8
    task_creator:
      num_threads: 4
    downgrade:
      num_threads: 1
      monitor_period: 10ms
  operator_params:
    scan_task_batch_size: 2GB
    max_sort_partition_bytes: 0
    hash_partition_bytes: 4GB
    concat_batch_bytes: 1GB
    max_build_hash_table_bytes: 8GB
"""


@pytest.mark.parametrize(
    "original",
    [
        (bench.ROOT / "test/cpp/integration/integration.yaml").read_text(),
        G4_CONFIG,
    ],
    ids=["repository-integration", "g4"],
)
def test_gpu_usage_limit_yaml_rewrite(tmp_path, original):
    source = tmp_path / "original.yaml"
    source.write_text(original)
    output = tmp_path / "derived"
    output.mkdir()
    # Without an override the exact source config is preserved.
    assert Path(bench.effective_config(source, output)).read_text() == original
    effective = Path(bench.effective_config(source, output, 123456))
    config = yaml.safe_load(effective.read_text())
    # src/sirius_config.cpp's root reader rejects every key except sirius;
    # gpu_mem_config rejects simultaneous usage_limit_bytes/fraction keys.
    assert set(config) == {"sirius"}
    gpu = config["sirius"]["memory"]["gpu"]
    assert gpu["usage_limit_bytes"] == 123456
    assert "usage_limit_fraction" not in gpu
    assert not {"reservation_limit_bytes", "reservation_limit_fraction"} <= gpu.keys()
    expected = yaml.safe_load(original)
    expected["sirius"]["memory"]["gpu"].pop("usage_limit_fraction")
    expected["sirius"]["memory"]["gpu"]["usage_limit_bytes"] = 123456
    assert config == expected
    assert source.read_text() == original
    # A second rewrite replaces an existing absolute limit without adding a fraction.
    second = tmp_path / "second"
    second.mkdir()
    rewritten = yaml.safe_load(
        Path(bench.effective_config(effective, second, 654321)).read_text()
    )
    expected["sirius"]["memory"]["gpu"]["usage_limit_bytes"] = 654321
    assert rewritten == expected
    with pytest.raises(ValueError, match="positive"):
        bench.effective_config(source, output, 0)


@pytest.mark.parametrize("limit", [None, 123456])
def test_missing_config_uses_sirius_root(tmp_path, monkeypatch, limit):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.delenv("SIRIUS_CONFIG_FILE", raising=False)
    config = yaml.safe_load(
        Path(bench.effective_config(None, tmp_path, limit)).read_text()
    )
    assert config == (
        {"sirius": {}}
        if limit is None
        else {"sirius": {"memory": {"gpu": {"usage_limit_bytes": limit}}}}
    )


def test_config_discovery_and_file_fingerprints(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("SIRIUS_CONFIG_FILE", raising=False)
    source = tmp_path / "sirius.yaml"
    source.write_text("sirius: {}\n")
    out = tmp_path / "out"
    out.mkdir()
    effective = Path(bench.effective_config(None, out))
    assert effective.read_bytes() == source.read_bytes()
    assert bench.file_sha256(effective) == bench.file_sha256(source)
    assert bench.plan_hashes(out) == {"sirius-config.yaml": bench.file_sha256(source)}


@pytest.fixture
def lanes(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    make_lane(a, "uncompressed", 1, [])
    make_lane(b, "onpair", 0.5, [])
    return a, b


@pytest.mark.parametrize(
    "reason,values",
    [
        ("validation failed", {"validated": False}),
        (
            "disk",
            {"disk": True, "served_by": json.dumps([{"entry": None, "tier": "disk"}])},
        ),
        ("fallback", {"fallback": True}),
    ],
)
def test_report_excludes_failure_union_from_every_aggregate(
    lanes, capsys, reason, values
):
    a, b = lanes
    update_result(b, 1, hot_median_s=0.0001, **values)
    report([a, b])
    text = capsys.readouterr().out
    assert reason in text
    assert "Geomean query set: [0]" in text
    assert "uncompressed: hot geomean_s=1.0" in text
    assert "onpair: hot geomean_s=0.5" in text
    assert "Geomean speedup uncompressed/onpair: 2.0" in text
    assert "q1 | excluded" in text


def test_report_marks_disabled_validation(lanes, capsys):
    a, b = lanes
    update_summary(b, validate=False)
    for q in (0, 1):
        update_result(b, q, validated="")
    report([a, b])
    text = capsys.readouterr().out
    assert "UNVALIDATED (--no-validate or missing validation)" in text
    assert "| meaningful |" not in text
    assert "Q17 adds ordered top-k; Q24 equals Q26" in text


def test_report_separates_layout_differences(lanes, capsys):
    a, b = lanes
    update_result(b, 1, served_by=json.dumps([{"entry": "main.hits", "tier": "host"}]))
    report([a, b])
    text = capsys.readouterr().out
    assert "query | time_change | assessment | layout_difference" in text
    assert "q1 | -50.00% | layout difference | gpu -> host" in text
    assert "Geomean query set: [0]" in text


def test_report_refuses_compression_failure_without_override(lanes, capsys):
    a, b = lanes
    update_summary(b, compression_failed=True)
    with pytest.raises(ValueError, match="allow-compression-failure"):
        report([a, b])
    report([a, b], allow_compression_failure=True)
    assert "COMPRESSION FAILURE OVERRIDE" in capsys.readouterr().out


@pytest.mark.parametrize(
    "key,value",
    [
        ("extension_sha256", "other-build"),
        ("config_sha256", "other-budget"),
        ("pre_sql", "SET threads=1"),
        ("duckdb_memory_limit", "2GB"),
    ],
)
def test_report_rejects_shared_fingerprint_mismatch(lanes, key, value):
    a, b = lanes
    update_summary(b, **{key: value})
    with pytest.raises(ValueError, match=f"fingerprint differs: {key}"):
        report([a, b])


@pytest.mark.parametrize(
    "key",
    [
        "SIRIUS_EXP_LATE_MAT",
        "SIRIUS_EXP_FUSED_SCAN",
        "SIRIUS_LATE_MAT_MAX_ROWS",
        "SIRIUS_PIN_TIER_DEFAULT",
        "LD_PRELOAD",
    ],
)
@pytest.mark.parametrize("baseline", [None, "0", ""])
def test_report_rejects_performance_environment_mismatch(lanes, key, baseline):
    a, b = lanes
    update_summary(a, perf_stack_env={} if baseline is None else {key: baseline})
    update_summary(b, perf_stack_env={key: "1"})
    with pytest.raises(ValueError, match="fingerprint differs: perf_stack_env"):
        report([a, b])


def test_report_normalizes_environment_and_excludes_output_paths(lanes, capsys):
    a, b = lanes
    env = {"SIRIUS_EXP_LATE_MAT": "1", "LD_PRELOAD": "/lib/libcudf.so"}
    update_summary(a, perf_stack_env={**env, "SIRIUS_EXP_FUSED_SCAN_FILTER": ""})
    update_summary(
        b,
        perf_stack_env={
            "LD_PRELOAD": env["LD_PRELOAD"],
            "SIRIUS_EXP_LATE_MAT": "1",
            "SIRIUS_EXP_FUSED_SCAN_FILTER": "",
            "SIRIUS_CONFIG_FILE": "/output/b/sirius-config.yaml",
            "SIRIUS_LOG_DIR": "/output/b/log_dir",
            "SIRIUS_LOG_LEVEL": "info",
        },
    )
    report([a, b])
    assert "Geomean speedup uncompressed/onpair: 2.0" in capsys.readouterr().out


def test_report_distinguishes_empty_gate_from_unset(lanes):
    # SIRIUS_EXP_FUSED_SCAN_FILTER="" enables the engine gate; unset disables it.
    a, b = lanes
    update_summary(a, perf_stack_env={"SIRIUS_EXP_FUSED_SCAN_FILTER": ""})
    update_summary(b, perf_stack_env={})
    with pytest.raises(ValueError, match="fingerprint differs: perf_stack_env"):
        report([a, b])


def test_perf_stack_env_keeps_empty_gate(monkeypatch):
    monkeypatch.setenv("SIRIUS_EXP_FUSED_SCAN_FILTER", "")
    monkeypatch.delenv("SIRIUS_EXP_OTHER", raising=False)
    env = bench.perf_stack_env()
    assert env["SIRIUS_EXP_FUSED_SCAN_FILTER"] == ""
    assert "SIRIUS_EXP_OTHER" not in env


@pytest.mark.parametrize("key", ["perf_stack_env", "duckdb_memory_limit"])
def test_report_rejects_missing_performance_fingerprint(lanes, key):
    a, b = lanes
    summary = json.loads((b / "summary.json").read_text())
    del summary[key]
    bench.write_json(b / "summary.json", summary)
    with pytest.raises(ValueError, match=f"missing comparison fingerprint: {key}"):
        report([a, b])


def test_report_allows_declared_plan_and_layout_differences(lanes, capsys):
    a, b = lanes
    update_summary(a, layout=["a"], plan_hashes={"hits.txt": "a"}, git_rev="rev-a")
    update_summary(b, layout=["b"], plan_hashes={"hits.txt": "b"}, git_rev="rev-b")
    report([a, b])
    assert "Geomean speedup uncompressed/onpair: 2.0" in capsys.readouterr().out
