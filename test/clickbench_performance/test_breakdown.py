# Copyright 2026, Sirius Contributors.
# SPDX-License-Identifier: Apache-2.0

"""CPU checks for native sources, profiler phases and decode-work accounting."""

import csv
import json
import pickle
import sys
from pathlib import Path

import clickbench_bench as bench
import pytest
from clickbench_queries import QUERIES, QUERY_COLUMNS, render_query
from gen_pin_layout import generate_layout
from report import (
    amdahl_penalties,
    free_text_queries,
    positive_geomean,
    report,
)
from test_bench import Connection, log_window
from test_fixes import update_result, update_summary
from test_tools import make_lane

import duckdb


def native_spec(tmp_path, tiny_parquet):
    return {
        "source": "table",
        "duckdb_path": str(tmp_path / "native.duckdb"),
        "duckdb_memory_limit": "1GB",
        "parquet": str(tiny_parquet),
        "run_dir": str(tmp_path),
        "queries": [0, 20, 23, 28],
        "hot_reps": 2,
        "timeout_s": 10,
        "validate": True,
        "profile": True,
    }


def test_render_table_and_default_parquet():
    for query in QUERIES.values():
        assert render_query(query, "unused.parquet", "table") == query
        assert "FROM hits" not in render_query(query, "/tmp/hit's.parquet")
        assert "read_parquet('/tmp/hit''s.parquet')" in render_query(
            query, "/tmp/hit's.parquet"
        )
    with pytest.raises(ValueError, match="unknown source"):
        render_query(QUERIES[0], "unused", "view")


def test_native_pin_sql_and_mixed_layout_plan_aliases(tmp_path):
    from clickbench_queries import union_columns_by_table

    columns = union_columns_by_table()["hits"]
    layout = generate_layout(dict.fromkeys(columns, 1), 20)
    path = tmp_path / "layout.json"
    bench.write_json(path, layout)
    assert bench.load_pin_layout(path) == layout
    con = Connection()
    spec = {
        "source": "table",
        "layout": layout,
        "parquet": "never-read.parquet",
        "timeout_s": 0,
    }
    bench.pin_tables(con, spec)
    assert len(con.statements) == 2
    for statement, entry in zip(con.statements, layout):
        assert statement.startswith("CALL pin_table(format = 'duckdb', ")
        assert f"name = '{entry['name']}', tier = '{entry['tier']}'" in statement
        assert "parquet" not in statement
    assert layout[-1]["name"] == "main.hits"
    source = tmp_path / "plans"
    source.mkdir()
    (source / "hits.txt").write_text("input -> identity\n")
    plans = Path(bench.stage_plans(source, tmp_path, layout))
    assert (plans / "hits.txt").read_bytes() == (plans / "main.hits.txt").read_bytes()


def test_native_table_creation_and_reuse(tmp_path, tiny_parquet):
    spec = native_spec(tmp_path, tiny_parquet)
    assert bench.prepare_table(spec) is True
    # Reuse the database even if the original Parquet is no longer accessible.
    spec["parquet"] = "missing.parquet"
    assert bench.prepare_table(spec) is False
    with duckdb.connect(spec["duckdb_path"], read_only=True) as con:
        assert con.execute("FROM hits").fetchall()
        assert con.execute("SELECT COUNT(*) FROM hits").fetchone() == (1,)


def test_native_source_rejects_view(tmp_path, tiny_parquet):
    spec = native_spec(tmp_path, tiny_parquet)
    with duckdb.connect(spec["duckdb_path"]) as con:
        con.execute("CREATE VIEW hits AS SELECT 1")
    with pytest.raises(ValueError, match="base table"):
        bench.prepare_table(spec)


def test_native_gpu_connection_uses_file(monkeypatch):
    con = Connection()
    calls = []

    def connect(**kwargs):
        calls.append(kwargs)
        return con

    monkeypatch.setattr(bench.duckdb, "connect", connect)
    spec = {
        "source": "table",
        "duckdb_path": "native.duckdb",
        "duckdb_memory_limit": "1GB",
        "extension": "sirius",
        "effective_plan_dir": None,
        "profile": True,
    }
    bench.open_benchmark_db(spec)
    assert calls == [
        {
            "database": "native.duckdb",
            "config": {"allow_unsigned_extensions": "true"},
        }
    ]
    assert con.statements.index(
        "SET profiling_mode = 'detailed'"
    ) > con.statements.index("LOAD 'sirius'")
    assert con.statements[-1] == "PRAGMA disable_profiling"


@pytest.mark.parametrize("profile", [True, False])
def test_native_workers_and_profile_files(tmp_path, tiny_parquet, monkeypatch, profile):
    spec = native_spec(tmp_path, tiny_parquet)
    spec["profile"] = profile
    assert bench.prepare_table(spec)
    spec["parquet"] = "missing.parquet"  # Neither worker may scan Parquet.

    def cpu_substitute(spec):
        con = duckdb.connect(bench.source_database(spec))
        if spec["profile"]:
            con.execute("SET profiling_mode = 'detailed'")
            con.execute("PRAGMA disable_profiling")
        return con

    monkeypatch.setattr(bench, "open_benchmark_db", cpu_substitute)
    monkeypatch.setattr(bench, "pin_tables", lambda con, spec: [])
    path = tmp_path / "_spec.json"
    bench.write_run_info(tmp_path, spec)
    bench.gpu_worker(path)
    # This opens the same persistent file read-only, without loading Sirius.
    bench.validation_worker(path)
    verdicts = json.loads((tmp_path / "_validation.json").read_text())
    assert len(verdicts) == 12 and not any(verdicts.values())
    reps = json.loads((tmp_path / "_repetitions.json").read_text())
    assert len(reps) == 12
    for rep in reps:
        assert rep["phase"] == ("cold" if rep["rep"] == 0 else "hot")
        file = tmp_path / "profiles" / f"q{rep['query']}_r{rep['rep']}.json"
        assert file.is_file() == profile
        if profile:
            data = json.loads(file.read_text())
            sql = data["query"]["sql"] if "query" in data else data["query_name"]
            assert sql == bench.query_sql(spec, rep["query"])
            assert rep["profiler_total_ms"] > 0
            parts = sum(
                rep[k]
                for k in (
                    "planner_ms",
                    "optimizer_ms",
                    "physical_ms",
                    "exec_ms",
                    "remainder_ms",
                )
            )
            assert parts == pytest.approx(rep["seconds"] * 1000)
        else:
            assert all(rep[k] is None for k in bench.PROFILE_FIELDS)
    assert (tmp_path / "profiles").exists() == profile
    assert "read_parquet" not in (tmp_path / "effective_queries.sql").read_text()


def substitute_sirius(monkeypatch):
    """Run real worker setup/pre-SQL, skipping only extension-specific commands."""
    connect = duckdb.connect
    statements = []

    class PlainConnection:
        def __init__(self, **kwargs):
            self.con = connect(**kwargs)

        def execute(self, statement, *args):
            statements.append(statement)
            if statement.startswith(
                (
                    "LOAD ",
                    "SET gpu_execution = ",
                    "SET enable_duckdb_fallback = ",
                    "SET pin_table_compression = ",
                )
            ):
                return self.con
            return self.con.execute(statement, *args)

        def __getattr__(self, name):
            return getattr(self.con, name)

    monkeypatch.setattr(bench.duckdb, "connect", PlainConnection)
    return statements


@pytest.mark.parametrize(
    "change",
    [
        "use",
        "catalog-search-path",
        "schema",
        "schema-search-path",
        "temp-table",
        "temp-view",
        "catalog-replacement",
        "missing-table",
        "persistent-view",
        "macro-current-database",
        "macro-duckdb-tables",
    ],
)
def test_native_worker_rejects_source_switch_before_timing(
    tmp_path, tiny_parquet, monkeypatch, change
):
    spec = native_spec(tmp_path, tiny_parquet)
    spec.update(queries=[20], extension="sirius", effective_plan_dir=None)
    assert bench.prepare_table(spec)
    alternate = {**spec, "duckdb_path": str(tmp_path / "alternate.duckdb")}
    assert bench.prepare_table(alternate)
    for database, extra_rows in (
        (spec["duckdb_path"], 99),
        (alternate["duckdb_path"], 0),
    ):
        with duckdb.connect(database) as con:
            con.execute("UPDATE hits SET URL = 'https://example.com/'")
            con.execute(
                f"INSERT INTO hits SELECT hits.* FROM hits, range({extra_rows})"
            )
            assert con.execute("SELECT COUNT(*) FROM hits").fetchone() == (
                extra_rows + 1,
            )
            assert con.execute(bench.query_sql(spec, 20)).fetchall() == [(0,)]

    # Comparing Q20 results cannot reveal the different tables (100 rows vs 1).
    attach = f"ATTACH {bench.sql_literal(alternate['duckdb_path'])} AS alternate; "
    shadow = "CREATE SCHEMA shadow; CREATE TABLE shadow.hits AS FROM hits LIMIT 1; "
    changes = {
        "use": attach + "USE alternate",
        "catalog-search-path": attach
        + "SET search_path = 'alternate.main,native.main'",
        "schema": shadow + "SET schema = 'shadow'",
        "schema-search-path": shadow + "SET search_path = 'shadow,main'",
        "temp-table": "CREATE TEMP TABLE HiTs AS FROM main.hits LIMIT 1",
        "temp-view": "CREATE TEMP VIEW HiTs AS FROM main.hits LIMIT 1",
        "catalog-replacement": (
            "ATTACH ':memory:' AS staging; USE staging; DETACH native; "
            f"ATTACH {bench.sql_literal(alternate['duckdb_path'])} AS native; USE native"
        ),
        "missing-table": "DROP TABLE hits",
        "persistent-view": "ALTER TABLE hits RENAME TO original; CREATE VIEW hits AS FROM original",
        # A user macro resolves before the system function of the same name.
        "macro-current-database": (
            "CREATE TEMP MACRO current_database() AS 'native'; " + attach + "USE alternate"
        ),
        "macro-duckdb-tables": (
            "CREATE TEMP MACRO duckdb_tables() AS TABLE "
            "SELECT * FROM system.main.duckdb_tables() WHERE NOT temporary; "
            "CREATE TEMP TABLE hits AS FROM native.main.hits LIMIT 1"
        ),
    }
    monkeypatch.setenv("SIRIUS_PRE_SQL", changes[change])
    statements = substitute_sirius(monkeypatch)

    def unexpected(*args):
        pytest.fail("source identity must be rejected before any pin or timer")

    monkeypatch.setattr(bench, "pin_tables", unexpected)
    monkeypatch.setattr(bench, "timed_query", unexpected)
    path = tmp_path / "_spec.json"
    bench.write_json(path, spec)
    with pytest.raises(
        ValueError, match="native table source identity mismatch after SIRIUS_PRE_SQL"
    ):
        bench.gpu_worker(path)
    assert not any("clickbench_begin" in statement for statement in statements)
    assert not (tmp_path / "pins.csv").exists()
    assert not (tmp_path / "_repetitions.json").exists()


def test_native_worker_allows_unshadowed_search_path(
    tmp_path, tiny_parquet, monkeypatch
):
    spec = native_spec(tmp_path, tiny_parquet)
    spec.update(
        queries=[20], extension="sirius", effective_plan_dir=None, profile=False
    )
    assert bench.prepare_table(spec)
    monkeypatch.setenv(
        "SIRIUS_PRE_SQL",
        "CREATE SCHEMA later; CREATE TABLE later.hits AS FROM hits LIMIT 0; "
        "SET search_path = 'native.main,native.later'; SET threads = 1",
    )
    substitute_sirius(monkeypatch)
    monkeypatch.setattr(bench, "pin_tables", lambda con, spec: [])
    path = tmp_path / "_spec.json"
    bench.write_json(path, spec)
    bench.gpu_worker(path)
    assert len(json.loads((tmp_path / "_repetitions.json").read_text())) == 3


def test_native_oracle_qualifies_catalog_despite_connection_state(
    tmp_path, tiny_parquet, monkeypatch
):
    spec = native_spec(tmp_path, tiny_parquet)
    spec.update(queries=[0], duckdb_path=str(tmp_path / 'native"quoted.duckdb'))
    assert bench.prepare_table(spec)
    assert spec["table_catalog"] == 'native"quoted'
    with (tmp_path / "_gpu_rows.pickle").open("wb") as f:
        pickle.dump({(0, rep): [(1,)] for rep in range(3)}, f)
    alternate = tmp_path / "alternate.duckdb"
    with duckdb.connect(str(alternate)) as con:
        con.execute("CREATE TABLE hits AS SELECT 1 FROM range(5)")
    connect = duckdb.connect

    def changed_context(**kwargs):
        con = connect(**kwargs)
        con.execute(
            f"ATTACH {bench.sql_literal(alternate)} AS alternate (READ_ONLY); USE alternate; "
            "CREATE TEMP TABLE hits AS SELECT 1 FROM range(10)"
        )
        assert con.execute(QUERIES[0]).fetchall() == [(10,)]
        return con

    monkeypatch.setattr(bench.duckdb, "connect", changed_context)
    sql = bench.query_sql(spec, 0, oracle=True)
    assert 'FROM "native""quoted".main.hits' in sql
    assert bench.query_sql(spec, 0) == QUERIES[0]
    path = tmp_path / "_spec.json"
    bench.write_json(path, spec)
    bench.validation_worker(path)
    assert json.loads((tmp_path / "_validation.json").read_text()) == dict.fromkeys(
        ("0:0", "0:1", "0:2")
    )


def test_real_profile_fixture():
    path = Path(__file__).with_name("fixtures") / "q20-uncompressed.json"
    values = bench.parse_profile(json.loads(path.read_text()), 0.202)
    assert values == pytest.approx(
        {
            "planner_ms": 44.17187,
            "optimizer_ms": 57.24451,
            "optimizer_ext_ms": 56.19817,
            "physical_ms": 0.05091,
            "exec_ms": 39.926851,
            "profiler_total_ms": 201.557211,
            "remainder_ms": 60.605859,
        }
    )


def test_profile_fallback_tree_and_flat_format():
    profile = {
        "planner": {"total_time": 0.01},
        "optimizer": {"total_time": 0.02},
        "physical_planner": {"total_time": 0.03},
        "query": {"total_time": 0.10},
        "operator": [
            {"timing": 0.04, "children": [{"timing": 0.05}, {"timing": 0.06}]},
            {"timing": 0.07},
        ],
    }
    parsed = bench.parse_profile(profile, 0.10)
    assert parsed["exec_ms"] == pytest.approx(220)
    assert parsed["optimizer_ext_ms"] == 0
    assert parsed["remainder_ms"] == pytest.approx(-180)
    flat = {
        "planner": 0.01,
        "all_optimizers": 0.02,
        "physical_planner": 0.03,
        "latency": 0.10,
        "children": [
            {"operator_timing": 0.04, "children": [{"operator_timing": 0.05}]}
        ],
    }
    assert bench.parse_profile(flat, 0.10)["exec_ms"] == pytest.approx(90)
    with pytest.raises(KeyError):
        bench.parse_profile({}, 0.10)
    profile["planner"]["total_time"] = float("nan")
    with pytest.raises(ValueError, match="invalid profiler"):
        bench.parse_profile(profile, 0.10)


def test_profiling_disabled_after_query_error(tmp_path):
    with duckdb.connect() as con:
        with pytest.raises(duckdb.BinderException):
            bench.profiled_query(con, "SELECT absent", 0, tmp_path / "bad.json")
        assert con.execute("SELECT current_setting('enable_profiling')").fetchone() == (
            None,
        )


def decode(path, ms):
    return f"[info] [compression] decode path={path} cols=2 rows=4000 ms={ms:.3f}"


def test_decode_line_and_window_sums():
    assert bench.parse_decode_line(decode("host", 1.25)) == {
        "path": "host",
        "cols": 2,
        "rows": 4000,
        "ms": 1.25,
    }
    assert bench.parse_decode_line("unrelated log") is None
    executing = "[info] Transparent GPU execution: executing query\n"
    assignment = "[info] assigned pinned entry 'main.hits' to operator '1' tier=HOST columns=105\n"
    body = executing + assignment + decode("device", 1.25) + "\n" + decode("host", 2.5)
    log = (
        decode("host", 999)
        + "\n"
        + log_window(20, 0, body)
        + decode("device", 999)
        + "\n"
    )
    log += log_window(20, 1, executing + decode("device", 0.125))
    log += log_window(20, 2, executing)
    result = bench.audit_log(log, [(20, 0), (20, 1), (20, 2)])
    assert result[20, 0]["decode_ms_sum"] == 3.75
    assert result[20, 0]["decode_calls"] == 2
    assert result[20, 0]["served_by"] == [{"entry": "main.hits", "tier": "host"}]
    assert result[20, 1]["decode_ms_sum"] == 0.125
    assert result[20, 2]["decode_ms_sum"] == result[20, 2]["decode_calls"] == 0
    with pytest.raises(ValueError, match="malformed decode"):
        bench.parse_decode_line(decode("device", -1))


def test_phase_hot_medians_exclude_cold():
    reps = [
        dict(
            query=20,
            rep=i,
            seconds=i + 1,
            rows=1,
            fallback=False,
            disk=False,
            served_by=[],
            **dict.fromkeys((*bench.PROFILE_FIELDS, *bench.DECODE_FIELDS), v),
        )
        for i, v in enumerate((999, 2, 4))
    ]
    result = bench.summarize(reps, None)[0]
    assert all(result[k] == 3 for k in (*bench.PROFILE_FIELDS, *bench.DECODE_FIELDS))
    reps[1]["exec_ms"] = None
    assert bench.summarize(reps, None)[0]["exec_ms"] is None


def test_table_path_required_cli(monkeypatch):
    args = [
        "clickbench_bench.py",
        "--parquet",
        "unused",
        "--lane",
        "test",
        "--output",
        "unused",
        "--source",
        "table",
    ]
    for extra in ([], ["--duckdb-path", ":memory:"]):
        monkeypatch.setattr(sys, "argv", args + extra)
        with pytest.raises(SystemExit) as error:
            bench.main()
        assert error.value.code == 2


def breakdown_lane(path, name, delta):
    make_lane(path, name, 1, [])
    with (path / "results.csv").open() as f:
        rows = list(csv.DictReader(f))
    for row, q in zip(rows, (20, 37)):
        row.update(
            query=q,
            hot_median_s=(200 + delta) / 1000,
            planner_ms=40,
            optimizer_ms=55,
            optimizer_ext_ms=50,
            physical_ms=5,
            exec_ms=40 + delta,
            profiler_total_ms=200 + delta,
            remainder_ms=60,
            decode_ms_sum=10 + delta,
            decode_calls=27,
        )
    bench.write_csv(path / "results.csv", list(rows[0]), rows)
    update_summary(path, source="table", duckdb_path="native.duckdb", profile=True)


def test_report_breakdown_two_lanes(tmp_path, capsys):
    base, lane = tmp_path / "base", tmp_path / "onpair"
    breakdown_lane(base, "uncompressed", 0)
    breakdown_lane(lane, "onpair", 20)
    report([base, lane], breakdown=True)
    text = capsys.readouterr().out
    assert "q20 | onpair | 220.000 | 160.000 | 60.000 | 30.000 | eligible" in text
    assert "q20 | onpair | uncompressed | 20.000 | 20.000 | 1.100 | eligible" in text
    assert "Free-text geomean query set: [20, 37]" in text
    assert (
        "onpair relative to uncompressed: free-text geomean exec_penalty_ms=20.000"
        in text
    )
    assert (
        "onpair relative to uncompressed: free-text geomean wall_penalty_ms=20.000"
        in text
    )
    assert "free-text geomean zero_decode_penalty_speedup_bound=1.100" in text
    assert "lane A" not in text
    # Existing exclusions apply to the breakdown aggregates too.
    update_result(lane, 37, fallback=True)
    report([base, lane], breakdown=True)
    assert "Free-text geomean query set: [20]" in capsys.readouterr().out


def test_breakdown_signed_penalties_and_missing_profiles(tmp_path, capsys):
    base, lane = tmp_path / "base", tmp_path / "onpair"
    breakdown_lane(base, "uncompressed", 0)
    breakdown_lane(lane, "onpair", 20)
    update_result(lane, 20, exec_ms=30, hot_median_s=0.19)
    update_result(lane, 37, **dict.fromkeys(bench.PROFILE_FIELDS, ""))
    report([base, lane], breakdown=True)
    text = capsys.readouterr().out
    assert "q20 | onpair | uncompressed | -10.000 | -10.000 | 0.950" in text
    assert "q37 | onpair | 220.000 | n/a | n/a | 30.000" in text
    assert "geomean exec_penalty_ms=n/a (queries=2, nonpositive=1, missing=1)" in text
    assert positive_geomean([1, 4]) == 2
    assert positive_geomean([0, 4]) is None
    assert amdahl_penalties(
        {"exec_ms": 0, "hot_median_s": 0.1}, {"exec_ms": 300, "hot_median_s": 0.2}
    ) == (300, 100, None)


def test_free_text_set_uses_dependencies():
    expected = {
        q
        for q, cols in QUERY_COLUMNS.items()
        if cols & {"URL", "Title", "Referer", "SearchPhrase", "OriginalURL"}
    }
    assert free_text_queries() == expected
    assert 23 in expected and 1 not in expected


@pytest.mark.parametrize(
    "key,value",
    [
        ("source", "parquet"),
        ("duckdb_path", "another.duckdb"),
        ("profile", False),
    ],
)
def test_report_source_and_profile_fingerprints(tmp_path, key, value):
    base, lane = tmp_path / "base", tmp_path / "onpair"
    breakdown_lane(base, "uncompressed", 0)
    breakdown_lane(lane, "onpair", 20)
    update_summary(lane, **{key: value})
    with pytest.raises(ValueError, match=f"fingerprint differs: {key}"):
        report([base, lane], breakdown=True)
