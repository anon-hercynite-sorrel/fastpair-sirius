# Copyright 2026, Sirius Contributors.
# SPDX-License-Identifier: Apache-2.0

"""Shared CPU-only ClickBench data fixture."""

import pytest
from clickbench_queries import SCHEMA, SCHEMA_COLUMNS

import duckdb


@pytest.fixture
def tiny_parquet(tmp_path):
    path = tmp_path / "hits.parquet"
    with duckdb.connect() as con:
        con.execute(SCHEMA)
        values = []
        for name, dtype in SCHEMA_COLUMNS:
            if name == "URL":
                value = "'https://google.com/'"
            elif dtype.startswith(("TEXT", "VARCHAR", "CHAR")):
                value = "''"
            elif dtype == "TIMESTAMP":
                value = "TIMESTAMP '2013-07-15 00:00:00'"
            elif dtype == "DATE":
                value = "DATE '2013-07-15'"
            else:
                value = "1"
            values.append(value)
        con.execute("INSERT INTO hits SELECT " + ",".join(values))
        con.execute(f"COPY hits TO '{path}' (FORMAT PARQUET)")
    return path
