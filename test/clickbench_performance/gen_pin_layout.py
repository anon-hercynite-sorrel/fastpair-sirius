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

"""Place columns by query frequency descending, bytes ascending, name ascending.

Consume an all-GPU footprint.csv with exact per-column resident bytes. Entry-only
or unavailable measurements are rejected: distributing a table total across
columns would silently change the experiment. A column that does not fit stays
off GPU; later, smaller columns may still fit. The host entry includes EVERY
queried column, duplicating GPU columns in host memory: a scan cannot combine
entries and must find its entire column set in a single pin. GPU is preferred
when both entries can serve. An empty GPU selection emits only the host entry. Budget is resident payload,
not allocator headroom; reserve engine workspace separately in the Sirius YAML.

With --chunks N and --parquet FILE every entry also carries scan_task_batch_bytes: the
batch size that makes the engine's parquet batch coalescer emit exactly N chunks for
that entry's columns, so entries with few columns are not pinned as a few large chunks
(per-query engine time has a fixed per-chunk part, which would otherwise favor the
smaller layouts). The coalescer accumulates row groups until the next one would push the
batch over the cap, sizing a row group as rows x width + rows/8 (validity) per
fixed-width column and parquet uncompressed size + rows x 4 (offsets) + rows/8 per
variable-width column.
"""

import argparse
import csv
import json
from pathlib import Path

from clickbench_queries import QUERY_COLUMNS, union_columns_by_table

# Bytes per row of the fixed-width DuckDB types, as the engine's parquet batch estimator
# sizes them; VARIABLE_WIDTH types cost their parquet uncompressed size plus offsets.
FIXED_WIDTH = {
    "BOOLEAN": 1,
    "TINYINT": 1,
    "UTINYINT": 1,
    "SMALLINT": 2,
    "USMALLINT": 2,
    "INTEGER": 4,
    "UINTEGER": 4,
    "BIGINT": 8,
    "UBIGINT": 8,
    "HUGEINT": 16,
    "UHUGEINT": 16,
    "FLOAT": 4,
    "DOUBLE": 8,
    "DATE": 4,
    "TIME": 8,
    "TIMESTAMP": 8,
    "TIMESTAMP_S": 8,
    "TIMESTAMP_MS": 8,
    "TIMESTAMP_NS": 8,
    "TIMESTAMP WITH TIME ZONE": 8,
}
VARIABLE_WIDTH = {"VARCHAR", "BLOB"}


def query_frequencies():
    return {
        c: sum(c in cols for cols in QUERY_COLUMNS.values())
        for c in union_columns_by_table()["hits"]
    }


def read_footprint(path):
    sizes = {}
    with Path(path).open() as f:
        for row in csv.DictReader(f):
            col = row["column"]
            if col == "*" or not row["resident_bytes"]:
                raise ValueError(
                    "layout requires per-column resident bytes; engine entry totals/unavailable sizes are insufficient"
                )
            if row["tier"] != "gpu" or col in sizes:
                raise ValueError("expected one all-GPU footprint row per column")
            size = int(row["resident_bytes"])
            if size < 0:
                raise ValueError("negative footprint")
            sizes[col] = size
    return sizes


def read_row_groups(parquet):
    """Row groups of a parquet file in file order, each (rows, {column: parquet uncompressed
    bytes}), plus the DuckDB type of every column."""
    import duckdb

    con = duckdb.connect()
    types = dict(
        con.execute(
            "SELECT column_name, column_type FROM (DESCRIBE SELECT * FROM read_parquet(?))",
            [parquet],
        ).fetchall()
    )
    groups = {}
    for rg, rows, col, size in con.execute(
        "SELECT row_group_id, row_group_num_rows, path_in_schema, total_uncompressed_size "
        "FROM parquet_metadata(?)",
        [parquet],
    ).fetchall():
        groups.setdefault(rg, (rows, {}))[1][col] = size
    return [groups[rg] for rg in sorted(groups)], types


def row_group_cost(rows, sizes, types, cols):
    total = 0
    for col in cols:
        kind = types[col]
        if kind in FIXED_WIDTH:
            total += rows * FIXED_WIDTH[kind] + rows // 8
        elif kind in VARIABLE_WIDTH:
            total += sizes[col] + rows * 4 + rows // 8
        else:
            raise ValueError(f"no batch size rule for column {col} of type {kind}")
    return total


def chunk_count(costs, cap):
    count, acc = 1, 0
    for i, cost in enumerate(costs):
        if i and acc + cost > cap:
            count, acc = count + 1, 0
        acc += cost
    return count


def batch_bytes_for_chunks(row_groups, types, cols, chunks):
    """The batch size, in the middle of the interval of sizes that make the coalescer emit
    exactly `chunks` chunks for `cols`, so a small difference in the engine's own estimate
    does not move the count."""
    costs = [row_group_cost(rows, sizes, types, cols) for rows, sizes in row_groups]
    if not 1 <= chunks <= len(costs):
        raise ValueError(f"chunks must be between 1 and {len(costs)} row groups")

    def smallest_cap_with_at_most(target):
        lo, hi = 0, sum(costs)
        while lo < hi:
            mid = (lo + hi) // 2
            if chunk_count(costs, mid) <= target:
                hi = mid
            else:
                lo = mid + 1
        return lo

    low = smallest_cap_with_at_most(chunks)
    high = smallest_cap_with_at_most(chunks - 1) if chunks > 1 else sum(costs) + 1
    if chunk_count(costs, low) != chunks or high <= low:
        raise ValueError(f"no batch size yields exactly {chunks} chunks")
    return (low + high - 1) // 2


def annotate_batch_bytes(layout, row_groups, types, chunks):
    for entry in layout:
        entry["scan_task_batch_bytes"] = batch_bytes_for_chunks(
            row_groups, types, entry["cols"], chunks
        )
    return layout


ORDERS = ("hot", "file")


def generate_layout(sizes, budget, frequencies=None, order="hot"):
    """order: "hot" places columns by query frequency descending, bytes ascending; "file"
    places them in footprint (parquet column) order, first come first pinned, which is what
    an engine admitting pins in arrival order would do."""
    frequencies = query_frequencies() if frequencies is None else frequencies
    if budget < 0 or set(sizes) != set(frequencies):
        raise ValueError(
            "budget must be nonnegative and footprint must cover exactly the pinnable columns"
        )
    if any(not isinstance(v, int) or v < 0 for v in sizes.values()):
        raise ValueError("resident byte sizes must be nonnegative integers")
    if order not in ORDERS:
        raise ValueError(f"order must be one of {ORDERS}")
    position = {c: i for i, c in enumerate(sizes)}
    key = (
        (lambda c: (-frequencies[c], sizes[c], c)) if order == "hot" else (lambda c: position[c])
    )
    gpu, used = [], 0
    for col in sorted(sizes, key=key):
        if used + sizes[col] <= budget:
            gpu.append(col)
            used += sizes[col]
    return ([{"name": "hits", "tier": "gpu", "cols": gpu}] if gpu else []) + [
        {"name": "main.hits", "tier": "host", "cols": sorted(sizes)}
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("footprint")
    parser.add_argument("--budget-bytes", type=int, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--order",
        choices=ORDERS,
        default="hot",
        help="hot: query frequency first (default); file: parquet column order, first come first pinned",
    )
    parser.add_argument(
        "--chunks",
        type=int,
        help="give every entry the scan_task_batch_bytes that pins it as exactly this many chunks",
    )
    parser.add_argument("--parquet", help="the table's parquet file (required with --chunks)")
    args = parser.parse_args()
    if (args.chunks is None) != (args.parquet is None):
        parser.error("--chunks and --parquet go together")
    layout = generate_layout(read_footprint(args.footprint), args.budget_bytes, order=args.order)
    if args.chunks is not None:
        annotate_batch_bytes(layout, *read_row_groups(args.parquet), args.chunks)
    Path(args.output).write_text(json.dumps(layout, indent=2) + "\n")


if __name__ == "__main__":
    main()
