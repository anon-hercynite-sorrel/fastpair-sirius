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

import json
from pathlib import Path

import pytest
from clickbench_bench import footprint_rows, load_pin_layout, write_csv
from clickbench_queries import union_columns_by_table
from gen_pin_layout import (
    annotate_batch_bytes,
    batch_bytes_for_chunks,
    chunk_count,
    generate_layout,
    query_frequencies,
    read_footprint,
)
from make_plan_sets import (
    ONPAIR,
    choose_plan,
    classify,
    parse_explore,
    render_plan_sets,
)
from report import footprint_totals, meaningful_change, report


@pytest.mark.parametrize(
    "budget,expected_gpu", [(1, []), (100, ["b", "a", "c"]), (5, ["b"])]
)
def test_layout_budgets_and_byte_tie(budget, expected_gpu):
    layout = generate_layout({"a": 8, "b": 4, "c": 2}, budget, {"a": 2, "b": 2, "c": 1})
    assert next((e["cols"] for e in layout if e["tier"] == "gpu"), []) == expected_gpu
    assert {c for e in layout for c in e["cols"]} == {"a", "b", "c"}
    assert all(
        e["name"] == ("hits" if e["tier"] == "gpu" else "main.hits") for e in layout
    )


def test_layout_skips_large_column_and_breaks_exact_ties_by_name():
    layout = generate_layout({"z": 10, "b": 2, "a": 2}, 2, {"z": 3, "b": 1, "a": 1})
    assert layout[0]["cols"] == ["a"]
    assert layout[1]["cols"] == ["a", "b", "z"]


def test_layout_round_trip_and_rejections(tmp_path):
    cols = union_columns_by_table()["hits"]
    sizes = {c: 1 for c in cols}
    path = tmp_path / "layout.json"
    layout = generate_layout(sizes, 20)
    path.write_text(json.dumps(layout))
    assert load_pin_layout(path) == layout
    for invalid in (
        [],
        [{"name": "elsewhere.hits", "tier": "gpu", "cols": cols}],
        [{"name": "hits", "tier": "gpu", "cols": cols[:-1]}],
        [{"name": "hits", "tier": "disk", "cols": cols}],
        [{"name": "hits", "tier": "gpu", "cols": cols + cols[:1]}],
        layout + layout[:1],
    ):
        path.write_text(json.dumps(invalid))
        with pytest.raises(ValueError):
            load_pin_layout(path)
    with pytest.raises(ValueError):
        generate_layout({"a": 1}, -1, {"a": 1})
    with pytest.raises(ValueError):
        generate_layout({}, 1, {"a": 1})
    assert query_frequencies()["URL"] > query_frequencies()["OriginalURL"] == 1


def test_batch_bytes_pin_exact_chunk_counts(tmp_path):
    # Ten row groups of 100 rows: a 4-byte column costs 412 per group, a VARCHAR column its
    # parquet size plus 400 offsets and 12 validity bytes.
    types = {"a": "INTEGER", "s": "VARCHAR"}
    row_groups = [(100, {"a": 999, "s": 1000 + 50 * i}) for i in range(10)]
    costs = [412 + 1412 + 50 * i for i in range(10)]
    assert chunk_count(costs, sum(costs)) == 1
    assert chunk_count(costs, max(costs)) == 10
    for chunks in range(1, 11):
        cap = batch_bytes_for_chunks(row_groups, types, ["a", "s"], chunks)
        assert chunk_count(costs, cap) == chunks
    # the middle of the interval: a few bytes either way keeps the count
    cap = batch_bytes_for_chunks(row_groups, types, ["a", "s"], 5)
    assert chunk_count(costs, cap - 100) == chunk_count(costs, cap + 100) == 5
    with pytest.raises(ValueError):
        batch_bytes_for_chunks(row_groups, types, ["a"], 11)
    with pytest.raises(ValueError):
        batch_bytes_for_chunks(row_groups, {"a": "STRUCT"}, ["a"], 2)
    layout = annotate_batch_bytes(
        [{"name": "hits", "tier": "gpu", "cols": ["a"]}], row_groups, types, 5
    )
    assert chunk_count([412] * 10, layout[0]["scan_task_batch_bytes"]) == 5
    sizes = {c: 1 for c in union_columns_by_table()["hits"]}
    full = generate_layout(sizes, 10**9)
    full[0]["scan_task_batch_bytes"] = 123
    path = tmp_path / "layout.json"
    path.write_text(json.dumps(full))
    assert load_pin_layout(path) == full
    for bad in (0, -1, 1.5, True, "2GB"):
        full[0]["scan_task_batch_bytes"] = bad
        path.write_text(json.dumps(full))
        with pytest.raises(ValueError):
            load_pin_layout(path)


def test_footprint_requires_resident_column_sizes(tmp_path):
    path = tmp_path / "footprint.csv"
    fields = ["column", "tier", "resident_bytes", "uncompressed_bytes", "entry"]
    write_csv(path, fields, footprint_rows(load_pin_layout()))
    with pytest.raises(ValueError, match="per-column"):
        read_footprint(path)
    assert footprint_totals(footprint_rows(load_pin_layout())) == {
        "gpu": None,
        "host": 0,
    }
    write_csv(
        path,
        fields,
        [
            {
                "column": "URL",
                "tier": "gpu",
                "resident_bytes": 42,
                "uncompressed_bytes": 100,
                "entry": "hits",
            }
        ],
    )
    assert read_footprint(path) == {"URL": 42}


STDOUT = """# column: n  dtype: i64  ratio: 2.000x  depth: 1
# comp: 50.00 GB/s  decomp: 300.00 GB/s
# old_wall_comp: 4.00 GB/s  old_wall_decomp: 5.00 GB/s
input -> bitpack -> chunk_min, chunk_count, chunk_bits, packed

---
# column: low  dtype: str  ratio: 8.000x  depth: 2
# comp: 10.00 GB/s  decomp: 400.00 GB/s
input -> dictionary -> keys_offsets, keys_chars, indices
dictionary.indices -> ans

---
# column: URL  dtype: str  ratio: 3.000x  depth: 2
# comp: 2.00 GB/s  decomp: 280.00 GB/s
input -> str_split -> offsets, chars
str_split.chars -> snappy
"""


def test_explore_stdout_and_three_way_split():
    candidates = parse_explore(STDOUT)
    schema = [("n", "BIGINT"), ("low", "TEXT"), ("URL", "TEXT")]
    classes = classify(schema, {"n": 100, "low": 9, "URL": 10}, 1000)
    assert [c["class"] for c in classes] == ["numeric", "low_cardinality", "free_text"]
    plans = render_plan_sets(candidates, classes, schema)
    for lane in plans:
        assert plans[lane].count("# column:") == 3
        assert plans[lane].count("---") == 2
    blocks = {lane: text.split("---") for lane, text in plans.items()}
    assert set(blocks) == {"uncompressed", "general", "onpair", "explorer"}
    assert (
        blocks["general"][:2]
        == blocks["onpair"][:2]
        == blocks["uncompressed"][:2]
        == blocks["explorer"][:2]
    )
    assert "input -> identity" in blocks["uncompressed"][2]
    assert "str_split.chars -> snappy" in blocks["general"][2]
    assert ONPAIR in blocks["onpair"][2]
    assert "ratio and throughput not yet measured" in blocks["onpair"][2]
    # The frontier's only free-text plan is dictionary-free: the explorer lane takes it.
    assert "str_split.chars -> snappy" in blocks["explorer"][2]
    assert "frontier plans without a dictionary root" in blocks["explorer"][2]
    assert "ratio: 3.0000x" in blocks["explorer"][2]


def test_actual_cli_frontier_tsv_selection():
    text = "column\tdtype\tpicked\tratio\tcomp_gbps\tdecomp_gbps\told_wall_comp_gbps\told_wall_decomp_gbps\tcompressed_bytes\tplan\n"
    text += "URL\tstr\t1\t9.0\t1.0\t100.0\t0\t0\t99\tinput -> str_split -> offsets, chars; str_split.chars -> ans\n"
    text += "URL\tstr\t0\t3.0\t1.0\t250.0\t0\t0\t99\tinput -> str_split -> offsets, chars; str_split.chars -> snappy\n"
    text += "URL\tstr\t0\t4.0\t1.0\t350.0\t0\t0\t99\tinput -> onpair_16 -> codes, row_offsets, dict_bytes, dict_offsets, batch_prefixes\n"
    choices = parse_explore(text)["URL"]
    pick = choose_plan(choices, "free_text")
    assert pick["ratio"] == 3
    assert "\nstr_split.chars -> snappy" in pick["plan"]
    assert choose_plan(choices[:1], "free_text")["decomp"] == 100
    # Low cardinality without any dictionary candidate: frontier max ratio, noted.
    low = choose_plan(choices, "low_cardinality")
    assert low["ratio"] == 9 and "no dictionary candidate" in low["note"]
    # With dictionary candidates: max ratio among them, decode floor waived.
    text2 = text
    text2 += "URL\tstr\t0\t1.04\t1.0\t134.0\t0\t0\t99\tinput -> dictionary -> keys_offsets, keys_chars, indices\n"
    text2 += "URL\tstr\t0\t80.0\t1.0\t109.0\t0\t0\t99\tinput -> dictionary -> keys_offsets, keys_chars, indices; dictionary.indices -> bitcomp\n"
    low = choose_plan(parse_explore(text2)["URL"], "low_cardinality")
    assert low["ratio"] == 80 and low["note"] == ""
    with pytest.raises(ValueError, match="no general-purpose"):
        choose_plan(choices[2:], "free_text")
    with pytest.raises(ValueError, match="untimed/OOM"):
        parse_explore(
            "# column: n  dtype: i64  ratio: 1.000x  depth: 0\ninput -> identity\n"
        )
    with pytest.raises(ValueError):
        render_plan_sets(parse_explore(STDOUT), [], [("n", "BIGINT")])
    with pytest.raises(ValueError):
        classify([("x", "TEXT")], {"x": 0}, 0)


BENCH_CSV = """column,dtype,rows,input_bytes,compressed_bytes,ratio,compress_ms_min,compress_ms_median,compress_ms_mean,compress_gbps_median,decompress_ms_min,decompress_ms_median,decompress_ms_mean,decompress_gbps_median,verify_ok
URL,str,10,100,30,3.3,1,1,1,30.0,1,1,1,{decomp},1
TOTAL,(all),10,100,30,3.3,1,1,1,30.0,1,1,1,{decomp},1
"""
BENCH_PLAN = """# column: URL  dtype: str
input -> str_split -> offsets, chars
str_split.offsets -> delta -> differences
str_split.offsets.differences -> ans
str_split.chars -> {codec}
---
"""


def test_free_text_bench_candidates_replace_dictionary_pick():
    from make_plan_sets import parse_bench

    bench = {}
    for codec, decomp in (("snappy", "245.0"), ("deflate", "150.0")):
        for name, choices in parse_bench(
            BENCH_CSV.format(decomp=decomp), BENCH_PLAN.format(codec=codec)
        ).items():
            bench.setdefault(name, []).extend(choices)
    assert [c["decomp"] for c in bench["URL"]] == [245.0, 150.0]
    # Explorer frontier: dictionary + snappy keys is the max-ratio pick above the floor.
    text = "column\tdtype\tpicked\tratio\tcomp_gbps\tdecomp_gbps\told_wall_comp_gbps\told_wall_decomp_gbps\tcompressed_bytes\tplan\n"
    text += "URL\tstr\t1\t4.6\t46.0\t294.0\t0\t0\t99\tinput -> dictionary -> keys_offsets, keys_chars, indices; dictionary.keys_chars -> snappy\n"
    # ... and the dictionary-free str_split plan above the floor that the explorer lane must pick.
    text += "URL\tstr\t0\t1.43\t170.0\t346.0\t0\t0\t300\tinput -> str_split -> offsets, chars; str_split.offsets -> bitpack -> chunk_min, chunk_count, chunk_bits, packed; str_split.chars -> ans\n"
    candidates = parse_explore(text)
    schema = [("URL", "TEXT")]
    classes = classify(schema, {"URL": 500}, 1000)
    plain = render_plan_sets(candidates, classes, schema)
    assert "input -> dictionary" in plain["general"]
    explorer_plan_lines = [
        line for line in plain["explorer"].splitlines() if not line.startswith("#")
    ]
    assert "str_split.chars -> ans" in explorer_plan_lines
    assert not any("dictionary" in line for line in explorer_plan_lines)
    assert "ratio: 1.4300x" in plain["explorer"]
    assert "without a dictionary root" in plain["explorer"]
    dictionary_only = "\n".join(text.splitlines()[:2]) + "\n"
    with pytest.raises(ValueError, match="no dictionary-free frontier candidate"):
        render_plan_sets(parse_explore(dictionary_only), classes, schema)
    # With bench candidates neither meets 250 GB/s: fastest decode (snappy), noted.
    with_bench = render_plan_sets(candidates, classes, schema, free_text_bench=bench)
    assert "str_split.chars -> snappy" in with_bench["general"]
    general_plan_lines = [
        line for line in with_bench["general"].splitlines() if not line.startswith("#")
    ]
    assert not any("dictionary" in line for line in general_plan_lines)
    assert "dictionary excluded" in with_bench["general"]
    assert "fastest-decode fallback" in with_bench["general"]
    assert with_bench["uncompressed"] == plain["uncompressed"]
    assert with_bench["onpair"] == plain["onpair"]
    with pytest.raises(ValueError, match="no --free-text-bench"):
        render_plan_sets(candidates, classes, schema, free_text_bench={})
    with pytest.raises(ValueError, match="did not verify"):
        parse_bench(
            BENCH_CSV.format(decomp="1").replace(",1\n", ",0\n"),
            BENCH_PLAN.format(codec="lz4"),
        )
    with pytest.raises(ValueError, match="not rooted at str_split"):
        parse_bench(
            BENCH_CSV.format(decomp="300"),
            "# column: URL  dtype: str\ninput -> dictionary -> keys_offsets, keys_chars, indices\ndictionary.keys_chars -> snappy\n---\n",
        )
    with pytest.raises(ValueError, match="no plan block"):
        parse_bench(
            BENCH_CSV.format(decomp="1"),
            BENCH_PLAN.format(codec="lz4").replace("URL", "Title"),
        )


ONPAIR_BENCH_PLAN = """# column: URL  dtype: str
input -> onpair_16 -> codes, row_offsets, dict_bytes, dict_offsets, batch_prefixes
onpair_16.row_offsets -> delta -> differences
onpair_16.row_offsets.differences -> ans
---
"""


def test_onpair_bench_fills_header_and_records_rule_pick():
    from make_plan_sets import parse_bench

    bench = {}
    for codec, decomp in (("snappy", "245.0"), ("lz4", "230.0")):
        for name, choices in parse_bench(
            BENCH_CSV.format(decomp=decomp), BENCH_PLAN.format(codec=codec)
        ).items():
            bench.setdefault(name, []).extend(choices)
    onpair = parse_bench(
        BENCH_CSV.format(decomp="398.0").replace(",3.3,", ",4.2,"),
        ONPAIR_BENCH_PLAN,
        root="onpair_16",
    )
    assert onpair["URL"][0]["plan"] == ONPAIR and onpair["URL"][0]["ratio"] == 4.2
    text = "column\tdtype\tpicked\tratio\tcomp_gbps\tdecomp_gbps\told_wall_comp_gbps\told_wall_decomp_gbps\tcompressed_bytes\tplan\n"
    text += "URL\tstr\t1\t4.6\t46.0\t294.0\t0\t0\t99\tinput -> dictionary -> keys_offsets, keys_chars, indices; dictionary.keys_chars -> snappy\n"
    text += "URL\tstr\t0\t1.43\t170.0\t346.0\t0\t0\t300\tinput -> str_split -> offsets, chars; str_split.chars -> ans\n"
    candidates = parse_explore(text)
    schema = [("URL", "TEXT")]
    classes = classify(schema, {"URL": 500}, 1000)
    plain = render_plan_sets(candidates, classes, schema, free_text_bench=bench)
    with_onpair = render_plan_sets(
        candidates, classes, schema, free_text_bench=bench, onpair_bench=onpair
    )
    # The onpair lane's plan is unchanged; only its header gains the measurement.
    strip = lambda s: [l for l in s.splitlines() if not l.startswith("#")]
    assert strip(with_onpair["onpair"]) == strip(plain["onpair"])
    assert with_onpair["general"] == plain["general"]
    assert "not yet measured" not in with_onpair["onpair"]
    assert "ratio: 4.2000x" in with_onpair["onpair"]
    assert "decomp: 398.00 GB/s (CLI bench of this plan)" in with_onpair["onpair"]
    # Neither general codec meets the floor, OnPair does: the rule picks OnPair.
    assert (
        "rule over all measured candidates (general bench + onpair): picks"
        " input -> onpair_16 -> codes, row_offsets, dict_bytes, dict_offsets,"
        " batch_prefixes (4.2000x, 398.00 GB/s)"
    ) in with_onpair["onpair"]
    # A faster general codec above the floor with a higher ratio wins the rule instead.
    strong = parse_bench(
        BENCH_CSV.format(decomp="400.0").replace(",3.3,", ",5.0,"),
        BENCH_PLAN.format(codec="deflate"),
    )
    rule_general = render_plan_sets(
        candidates,
        classes,
        schema,
        free_text_bench={"URL": bench["URL"] + strong["URL"]},
        onpair_bench=onpair,
    )
    assert (
        "picks input -> str_split -> offsets, chars (5.0000x" in rule_general["onpair"]
    )
    # Without a general bench the rule runs over the explorer frontier plus OnPair.
    frontier_rule = render_plan_sets(candidates, classes, schema, onpair_bench=onpair)
    assert (
        "(general frontier + onpair): picks input -> dictionary"
        in frontier_rule["onpair"]
    )
    with pytest.raises(ValueError, match="no --onpair-bench"):
        render_plan_sets(candidates, classes, schema, onpair_bench={})
    with pytest.raises(ValueError, match="exactly the onpair lane plan"):
        render_plan_sets(
            candidates,
            classes,
            schema,
            onpair_bench={
                "URL": [{**onpair["URL"][0], "plan": "input -> onpair_16 -> codes"}]
            },
        )
    with pytest.raises(ValueError, match="not rooted at onpair_16"):
        parse_bench(
            BENCH_CSV.format(decomp="1"),
            BENCH_PLAN.format(codec="lz4"),
            root="onpair_16",
        )


def test_meaningful_change_rule():
    baseline = {"hot_median_s": 1, "hot_std_s": 0.01}
    assert meaningful_change(baseline, {"hot_median_s": 0.95, "hot_std_s": 0.01})[1]
    assert not meaningful_change(baseline, {"hot_median_s": 0.96, "hot_std_s": 0.001})[
        1
    ]
    assert not meaningful_change(baseline, {"hot_median_s": 0.90, "hot_std_s": 0.1})[1]
    assert meaningful_change(baseline, {"hot_median_s": 1.2, "hot_std_s": 0.01})[1]


def make_lane(path, name, scale, fallback):
    path.mkdir()
    (path / "summary.json").write_text(
        json.dumps(
            {
                "lane": name,
                "parquet": "hits.parquet",
                "hot_reps": 10,
                "fallback_queries": fallback,
                "pin_seconds": 2,
                "validate": True,
                "compression_failed": False,
                "extension_sha256": "same-extension",
                "config_sha256": "same-config",
                "pre_sql": "",
                "perf_stack_env": {},
                "duckdb_memory_limit": "8GB",
            }
        )
    )
    fields = [
        "query",
        "cold_s",
        "hot_median_s",
        "hot_min_s",
        "hot_mean_s",
        "hot_std_s",
        "rows",
        "fallback",
        "disk",
        "served_by",
        "validated",
    ]
    rows = [
        {
            "query": q,
            "cold_s": 2,
            "hot_median_s": scale * (q + 1),
            "hot_min_s": scale,
            "hot_mean_s": scale,
            "hot_std_s": 0.001,
            "rows": 1,
            "fallback": q in fallback,
            "disk": False,
            "served_by": json.dumps([{"entry": "hits", "tier": "gpu"}]),
            "validated": True,
        }
        for q in (0, 1)
    ]
    write_csv(path / "results.csv", fields, rows)
    write_csv(
        path / "footprint.csv",
        ["column", "tier", "resident_bytes"],
        [{"column": "URL", "tier": "gpu", "resident_bytes": 10}],
    )
    (path / "effective_queries.sql").write_text(
        "SELECT count(*) FROM read_parquet('hits.parquet');\n"
    )


def test_report_common_fallback_exclusion(tmp_path, capsys):
    a, b, c = (tmp_path / x for x in ("a", "b", "c"))
    make_lane(a, "uncompressed", 1, [])
    make_lane(b, "general", 2, [1])
    make_lane(c, "onpair", 0.9, [])
    report([a, b, c])
    text = capsys.readouterr().out
    assert "1: ['general: fallback']" in text
    assert "Geomean query set: [0]" in text
    assert "Geomean speedup uncompressed/general: 0.5" in text
    assert "q1 | excluded | general: fallback" in text
    assert "resident_bytes={'gpu': 10, 'host': 0}" in text
    (c / "effective_queries.sql").write_text("SELECT 2;\n")
    with pytest.raises(ValueError, match="SQL differs"):
        report([a, c])


def test_plan_generator_cli_covers_full_schema(tmp_path):
    """Small, hand-written-per-column explore output; no GPU/data files."""
    import subprocess
    import sys

    from clickbench_queries import SCHEMA_COLUMNS
    from make_plan_sets import FREE_TEXT

    rows, counts, blocks = 1000, {}, []
    for name, dtype in SCHEMA_COLUMNS:
        string = dtype.startswith(("TEXT", "VARCHAR", "CHAR"))
        counts[name] = 20 if name in FREE_TEXT else 1 if string else 100
        dsl = (
            "input -> str_split -> offsets, chars\nstr_split.chars -> snappy"
            if name in FREE_TEXT
            else (
                "input -> dictionary -> keys_offsets, keys_chars, indices"
                if string
                else "input -> bitpack -> chunk_min, chunk_count, chunk_bits, packed"
            )
        )
        blocks.append(
            f"# column: {name}  dtype: {'str' if string else 'i64'}  ratio: 2.000x  depth: 1\n# comp: 20.00 GB/s  decomp: 300.00 GB/s\n{dsl}"
        )
    classes = classify(SCHEMA_COLUMNS, counts, rows)
    class_path = tmp_path / "classes.csv"
    write_csv(class_path, ["column", "class", "distinct", "rows"], classes)
    explore_path = tmp_path / "explore.txt"
    explore_path.write_text("\n---\n".join(blocks) + "\n")
    output = tmp_path / "plans"
    subprocess.run(
        [
            sys.executable,
            str(Path(__file__).with_name("make_plan_sets.py")),
            "--explore",
            str(explore_path),
            "--classes",
            str(class_path),
            "--output",
            str(output),
        ],
        check=True,
    )
    shared = []
    for lane in ("uncompressed", "general", "onpair"):
        text = (output / lane / "hits.txt").read_text()
        assert text.count("# column:") == 105
        assert [b.splitlines()[0].split()[2] for b in text.split("\n\n---\n")] == [
            name for name, _ in SCHEMA_COLUMNS
        ]
        shared.append(
            [
                b
                for b in text.split("\n\n---\n")
                if b.splitlines()[0].split()[2] not in FREE_TEXT
            ]
        )
    assert shared[0] == shared[1] == shared[2]
    assert (output / "classes.csv").read_text() == class_path.read_text()
