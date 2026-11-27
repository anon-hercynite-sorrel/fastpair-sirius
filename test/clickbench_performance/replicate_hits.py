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

"""Write k verbatim row copies for the capacity sweep (large; run on the box).

Each file is a fresh DuckDB COPY of the same source, preserving input row order.
Rows/values are copied verbatim; Parquet encoding need not be byte-identical.
Never overwrite an existing destination. No script imports or tests create the
real sweep data.
"""

import argparse
from pathlib import Path

from clickbench_queries import sql_literal

import duckdb


def replicate(parquet, output, copies):
    if copies not in (2, 4):
        raise ValueError("the pre-declared replication factors are 2 and 4")
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=False)
    with duckdb.connect() as con:
        con.execute("SET threads = 1")
        con.execute("SET preserve_insertion_order = true")
        for i in range(copies):
            con.execute(
                f"COPY (SELECT * FROM read_parquet({sql_literal(parquet)})) TO {sql_literal(destination / f'part_{i}.parquet')} (FORMAT PARQUET)"
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet", required=True)
    parser.add_argument("--copies", type=int, choices=(2, 4), required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    replicate(args.parquet, args.output, args.copies)


if __name__ == "__main__":
    main()
