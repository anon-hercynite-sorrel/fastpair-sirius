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

"""Binding and dependency checks against an empty DuckDB hits schema."""

from pathlib import Path

import pytest
from clickbench_queries import (
    QUERIES,
    QUERY_COLUMNS,
    SCHEMA,
    SCHEMA_COLUMNS,
    parquet_query,
    union_columns_by_table,
)

import duckdb


@pytest.fixture(scope="module")
def empty_schema():
    con = duckdb.connect()
    con.execute(SCHEMA)
    yield con
    con.close()


def test_inventory(empty_schema):
    assert list(QUERIES) == list(range(43))
    assert len(QUERIES) == len(QUERY_COLUMNS) == 43
    assert len(SCHEMA_COLUMNS) == 105
    actual = [(r[0], r[1]) for r in empty_schema.execute("DESCRIBE hits").fetchall()]
    expected = [
        (
            name,
            str(
                empty_schema.execute(f"SELECT CAST(NULL AS {dtype})").description[0][1]
            ),
        )
        for name, dtype in SCHEMA_COLUMNS
    ]
    assert actual == expected
    assert set(union_columns_by_table()["hits"]) == {n for n, _ in SCHEMA_COLUMNS}
    assert QUERY_COLUMNS[23] == set(union_columns_by_table()["hits"])
    assert "SearchPhrase" not in QUERY_COLUMNS[15]
    assert "OriginalURL" in QUERY_COLUMNS[23]


@pytest.mark.parametrize("q", range(43))
def test_query_binds_and_columns_match(empty_schema, q):
    empty_schema.execute("EXPLAIN " + QUERIES[q]).fetchall()
    if q == 23:
        # SELECT * references all columns by its projection, even when exclusion
        # would still bind. Check the result schema independently.
        assert {
            d[0] for d in empty_schema.execute(QUERIES[q]).description
        } == QUERY_COLUMNS[q]
        return
    referenced = set()
    for col, _ in SCHEMA_COLUMNS:
        # Binding, not an optimized empty-table scan, detects filter/sort inputs
        # too. Missing inputs fail even when their output would be optimized out.
        without = f'WITH hits AS (SELECT * EXCLUDE ("{col}") FROM main.hits) '
        try:
            empty_schema.execute("EXPLAIN " + without + QUERIES[q]).fetchall()
        except duckdb.BinderException:
            referenced.add(col)
    assert referenced == QUERY_COLUMNS[q]


def test_rewrite_escapes_path_and_keeps_sql():
    rendered = parquet_query(QUERIES[28], "/tmp/hit's/*.parquet")
    assert "FROM read_parquet('/tmp/hit''s/*.parquet')" in rendered
    assert r"'\1'" in rendered
    assert "FROM hits" not in rendered


def test_original_schema_is_unmodified():
    assert SCHEMA.startswith("CREATE TABLE hits")
    assert Path(__file__).with_name("queries.sql").read_text().count("\n") == 43
