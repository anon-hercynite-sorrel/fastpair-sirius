# ClickBench OnPair evaluation

This directory runs 43 ClickBench queries, numbered **0–42**, over one local
Parquet file or a quoted glob, or a persistent DuckDB table loaded from that input.
Python uses DuckDB and PyYAML (for budget-derived YAML); tests additionally use pytest.
CPU dependencies are listed in `requirements.txt`. No GPU or
external data is needed for the tests:

```bash
pixi run python -m pytest test/clickbench_performance -q
```

`create.sql` is recovered verbatim from
`9f30b9ca^:scripts/clickbench_runner/create.sql`. `queries.sql` takes all 43 SQL
texts from `9f30b9ca^:test/sql/clickbench-sirius.test`, including its aliases and
tie breakers. Each query occupies one line. [QUERIES.md](QUERIES.md) records every
edit from the deleted runner. Q7 retains count ties; Q23 can retain ties when
EventTime and WatchID coincide but other projected values differ. Validation
reports these differences; it never sorts rows to hide them.

## Prepare the plans on the GPU box

From the built repository on the G4 box (currently `~/sirius`), with the 10M-row
sample at `~/data/hits.parquet`:

```bash
cd ~/sirius
mkdir -p ~/results/clickbench-setup
pixi run build/release/simpatico_codegen/simpatico explore \
  --input "$HOME/data/hits.parquet" --score pareto --decomp-floor 250 \
  --frontier-out "$HOME/results/clickbench-setup/frontier.tsv" \
  > "$HOME/results/clickbench-setup/explore.txt" \
  2> "$HOME/results/clickbench-setup/explore.log"
pixi run python test/clickbench_performance/make_plan_sets.py \
  --explore "$HOME/results/clickbench-setup/frontier.tsv" \
  --parquet "$HOME/data/hits.parquet" \
  --output test/clickbench_performance/plans/clickbench
```

Use a fresh frontier filename: the CLI **appends** across invocations. It emits
TSV, with multiline DSL flattened using `; `. Its stdout instead contains
`# column: ... dtype: ... ratio: ... depth: ...`, throughput comments, DSL, and
`---` separators; both formats are accepted. The frontier allows selection among
alternatives instead of trusting a lone stdout pick. OOM/untimed blocks and
missing columns are errors. Current `operator_registry.cpp` marks OnPair as
non-explorable, so this search supplies the general-purpose baseline. The parser
also excludes OnPair/FSST candidates from an externally supplied frontier.

The generator measures exact distinct counts using plain DuckDB on this sample,
one column at a time. It writes `classes.csv` with column, class, distinct, rows.
`--classes <previous classes.csv>` reuses those measurements and rechecks the rule.
Strings below 1% distinct/rows are low-cardinality; the other strings must
be exactly URL, Referer, Title, SearchPhrase, OriginalURL. Numerics include the
date/time columns. All 105 schema columns receive plan blocks, in schema order.

It writes `plans/clickbench/{uncompressed,general,onpair}/hits.txt`:

- Shared numerics: identical explore plans in every lane, maximum ratio among
  candidates decoding at least 250 GB/s. If no candidate reaches 250 GB/s, choose
  fastest decode, matching the explorer's fallback; the plan comment records this.
- Shared low-cardinality strings: maximum ratio among the frontier's dictionary
  candidates with the decode floor waived (these columns are small, so their
  decode time is negligible, while their raw-indices dictionary form barely
  compresses and would be the floor's pick). A column whose frontier has no
  dictionary candidate at all (near-empty columns such as Params or SocialAction,
  where str_split cascades dominate) takes the frontier's max-ratio plan; the plan
  comment records this.
- Free text: identity / selected general plan / OnPair-16 with row offsets.
  Pass `--free-text-bench gp-snappy.csv:plans/gp-snappy.txt` (repeat per codec,
  from the CLI bench) to choose the general lane's free-text plans among measured
  str_split plans only: the explorer's dictionary picks are refused by Simpatico
  on chunks above 0.5 estimated distinct fraction, and one refusal pins the rest
  of the table uncompressed (seen at chunk 10/27 of the full hits).
  compressed by delta followed by ANS. OnPair's trainer uses bridge defaults
  (threshold 0.2, seed 42); this script introduces no training knob.

The `uncompressed` lane keeps **only free text** uncompressed; shared columns
still use their common compression plans. Omitting `--plan-dir` below creates a
separate, wholly uncompressed-pin control. Generated plans and classes are not
fabricated or checked in before the actual explore run.

The sample's Parquet column order must match `create.sql`, and strings/date/time
must have the intended logical types. Use a normalized Parquet input if the
original download instead stores unannotated binary or integer timestamps;
queries, pinning, exploration and validation must all read that same input.

## Run one lane on the G4

The existing integration YAML is a conservative first-run config (one GPU, 40%
GPU usage fraction, 32 GB host pool). Use the same hardware-appropriate YAML for
all lanes. The SF1000 reproduction YAML targets a different GPU and host size.
`--config` becomes `SIRIUS_CONFIG_FILE` before loading the extension, as in the
TPC-H harness and `bench/sf1000-repro/run.sh`.

```bash
cd ~/sirius
pixi run python test/clickbench_performance/clickbench_bench.py \
  --parquet "$HOME/data/hits.parquet" \
  --lane onpair \
  --plan-dir test/clickbench_performance/plans/clickbench/onpair \
  --config test/cpp/integration/integration.yaml \
  --hot-reps 10 --validate --timeout-s 600 --duckdb-memory-limit 8GB \
  --output "$HOME/results/clickbench-10m"
```

Repeat with lane/plan directory `uncompressed` and `general`. Output lane
directories must be new, preventing old logs from contaminating an audit.
`--queries 0,5,12-14,20-28` selects a diagnostic subset; default is all 43.
The layout still must cover the full query union. Q23 projects every column,
so that union is all 105 columns. `--extension` or `SIRIUS_EXTENSION_PATH` can
select a different build; default is
`sirius-duckdb/build/release/extension/sirius/sirius.duckdb_extension`.

`--source parquet` is the default. The runner rewrites `FROM hits` to
`FROM read_parquet('<exact pin source>')`,
following the documented Parquet pin matching. It calls
`pin_table('<source>', name='<name>', tier='<tier>', cols=[...])` for each layout
entry. Pin time includes materialization, compression and OnPair training;
it is reported separately from queries. A supplied layout is a JSON list:

```json
[
  {"name": "hits", "tier": "gpu", "cols": ["URL", "UserID"]},
  {"name": "main.hits", "tier": "host", "cols": ["URL", "UserID", "Title"]}
]
```

This small example illustrates the shape and deliberately omits columns; the
runner rejects it for missing coverage. Real layouts must assign every queried
column at least once, serve each query from a single entry, use unique allowed
names, and use only gpu/host tiers.
Without a layout, all query columns are pinned on the GPU. For compression with
a mixed layout the runner stages identical `hits.txt` and `main.hits.txt` files:
engine plan lookup compares the complete filename stem to the pin name.

### DuckDB-native table source

Use the same queries and plans with a persistent DuckDB database to remove the
per-query Parquet bind/metadata read from the measurement:

```bash
pixi run python test/clickbench_performance/clickbench_bench.py \
  --source table --duckdb-path "$HOME/data/clickbench.duckdb" \
  --parquet "$HOME/data/hits.parquet" --lane onpair \
  --plan-dir test/clickbench_performance/plans/clickbench/onpair \
  --config test/cpp/integration/integration.yaml \
  --hot-reps 10 --output "$HOME/results/clickbench-table"
```

`--duckdb-path` is required with `table` and must name a persistent file.
Before the workers start, plain DuckDB opens/creates this database and, if
`main.hits` is absent, executes `CREATE TABLE hits AS FROM read_parquet(...)`,
then `CHECKPOINT`. An existing native `main.hits` is reused verbatim; it is not
refreshed or checked against the supplied Parquet. Use a new database path to
load a different dataset. `--parquet` remains required as the creation input and
provenance. `run_info.txt` and `summary.json` record `source`, the resolved
`duckdb_path`, and `duckdb_table_created` (true on creation, false on reuse, null
for Parquet). Creation/checkpoint time is outside query and pin timings.

Both workers open that same file sequentially. The GPU worker loads Sirius;
the CPU oracle opens the file read-only without Sirius. One SQL renderer serves
both workers and `effective_queries.sql`: table mode keeps `FROM hits` and uses
a native scan. Pins use `CALL pin_table(name => 'hits', format => 'duckdb',
tier => 'gpu', cols => [...])`, without a positional Parquet path.

Budget layouts from `gen_pin_layout.py` work unchanged: the GPU entry is `hits`
and the host entry is **`main.hits`**. Both names resolve to the same native
table, but are distinct pin registry keys. The engine matches scans by the
resolved catalog/schema/table, the table OID and row-group collection identity,
then column coverage. GPU preference and the complete host superset still apply.
Plans still come from `hits.txt`; mixed layouts stage the `main.hits.txt` alias
as for Parquet. Native pins emit the same `[pin_table] resident entry='hits'`
(or `'main.hits'`) records and `assigned pinned entry ... tier=GPU|HOST` records,
so footprint and `served_by` audits use the same parsers.

`SIRIUS_PRE_SQL` runs after LOAD, before the explicit lane settings and pins.
The runner enforces GPU execution, enables auditable CPU fallback, and enables
pin compression only when `--plan-dir` is given. Use the same optional performance
settings in all lanes. The effective settings environment is recorded.
The runner discovers YAML using
Sirius's order (`--config`/`SIRIUS_CONFIG_FILE`, `./sirius.yaml`,
`~/.sirius/sirius.yaml`), copies it into the lane directory, and loads that copy.
With no YAML it writes `{}`, selecting the engine defaults. For a sweep,
`--gpu-usage-limit-bytes N` writes a derived copy with
`memory.gpu.usage_limit_bytes: N` and removes the mutually exclusive
`usage_limit_fraction`. The source YAML stays intact. Use the same N across lanes;
layout budgets count resident columns, while this limit also covers workspace.

For each query in numerical order, repetition 0 is cold (its first execution
after pinning); repetitions 1–10 are hot. This does not flush device caches or
JIT caches between queries. Wall time includes full result fetching. A query or
pin timeout interrupts the connection and fails the run; it is not a fast timing.

## Results and validation

Each `<output>/<lane>/` contains:

| File | Contents |
| --- | --- |
| `results.csv` | Wall statistics, rows, fallback, disk, served_by, validated, plus hot medians of the phase/decode columns below |
| `repetitions.csv` | query, rep, phase, seconds, rows, fallback, disk, served_by, audit, validated, plus per-repetition phase/decode columns |
| `pins.csv` | entry, tier, cols (JSON list), seconds |
| `footprint.csv` | column, tier, resident_bytes, uncompressed_bytes, entry; blank sizes mean unavailable |
| `summary.json` | lane, extension SHA256, copied YAML and SHA256, pre-SQL, source/effective plan hashes, layout, git revision, eligible hot geomean, exclusion reasons, per-tier repetition counts, validation status, pin total, compression_failed, coverage and warnings |
| `log_dir/` | Original Sirius logs, directly written here and flushed when the GPU child exits |
| `gpu_worker.log`, `validation_worker.log` | Child console diagnostics |
| `effective_queries.sql`, `run_info.txt` | Effective SQL and run provenance |
| `effective_plans/`, `sirius-config.yaml` | Copies of effective plans and YAML (including discovered/default or derived YAML) |
| `profiles/q<query>_r<rep>.json` | Original detailed DuckDB JSON profile for every cold/hot repetition; absent with `--no-profile` |

Internal `_spec.json`, `_repetitions.json`, `_gpu_rows.pickle` and
`_validation.json` preserve worker inputs/results for diagnosis. Pickles are
local harness output, not an external interchange format. Boolean CSV values
are `True`/`False`; blank `validated` means `--no-validate`. Standard deviation is
the sample standard deviation (`n-1`; zero with one hot repetition).

The parent never loads Sirius. The GPU child exits before validation begins,
releasing its memory pools. A fresh plain DuckDB child runs each selected query
on the same source (Parquet or native table) and compares it to **every cold and hot result**. Row order
is exact, floats use relative tolerance 1e-4 and absolute tolerance 1e-6, paired
NaNs match, and NULLs must match. Mismatches leave results and return exit code 1.
`--no-validate` skips this phase; it cannot establish correctness.

Fallback audit requires INFO logging and the spdlog file backend; the runner
forces both. C++ counters exist but have no SQL accessor at this revision.
Untimed `SET VARIABLE` markers bracket each repetition in QueryBegin records;
planning `Transparent execution fallback` and runtime `falling back to DuckDB
CPU` messages mark fallback. Missing markers fail the audit. Windows without a
positive `Transparent GPU execution: executing query` marker are conservatively
marked `cpu_or_unobserved` and included in the fallback exclusion list (including
queries fully answered by metadata/constant folding). This prevents unsupported
queries or disabled interception from masquerading as GPU speedups.

Each repetition's `served_by` is a JSON list of `{entry, tier}` assignments from
its sentinel window. A GPU-executed window with no assignment is `disk`, and is
excluded just like fallback. Fallback without an assignment is labeled `cpu`.
`results.csv` records all distinct assignments across repetitions; tier counts in
`summary.json` count repetitions served by each tier, including the cold one.
A repetition with multiple scan assignments can count in more than one tier.

Inspect compression coverage and pin warnings before using timings. The engine
can retain uncompressed chunks below its size/ratio threshold or after a codec
failure; those are reported separately from CPU query fallback.

### Per-repetition timing and decode work

`seconds` in `repetitions.csv` and `_repetitions.json` remains the wall time of
execute plus fetching every result. `hot_median_s` remains the headline in
`results.csv`. By default each cold/hot query enables detailed JSON profiling
with full operator coverage (including metadata-only COUNT),
writes `profiles/q<query>_r<rep>.json`, then disables profiling before the end
marker. Setup, teardown and JSON parsing are outside the wall timer; collecting
profile metrics is inside it. `--no-profile` skips collection and leaves phase
columns blank/null. It still records wall time and audits decode logs.

| Column (ms unless stated) | Meaning |
| --- | --- |
| `planner_ms` | `planner.total_time`, including bind |
| `optimizer_ms` | `optimizer.total_time` |
| `optimizer_ext_ms` | `optimizer.extension`, already included in optimizer total |
| `physical_ms` | `physical_planner.total_time` |
| `exec_ms` | Sum of exclusive operator timings throughout the `operator[]` tree, including `RESULT_COLLECTOR` and its `EXTENSION` child for Sirius; all operators for CPU fallback |
| `profiler_total_ms` | `query.total_time` |
| `remainder_ms` | wall ms minus (planner + optimizer + physical + exec); signed, never clamped |
| `decode_ms_sum` | Sum of `[compression] decode path=device\|host cols=... rows=... ms=...` records within this repetition's QueryBegin marker window |
| `decode_calls` | Number of those decode records (count, not ms) |

All these values are stored per repetition in `_repetitions.json` and
`repetitions.csv`; `results.csv` takes each column's median over hot repetitions.
Independent medians need not add exactly to median wall time. CPU operator
times can overlap across threads and exceed elapsed wall time, making the
remainder negative. Fallback/disk/validation exclusions remain in force.

The engine measures host wall time around `decompress_chunk` on the device
path, and reconstruction plus decode on the host path (including H2D transport).
Decode sessions synchronize and check status words before returning; the INFO
record adds no extra synchronization. The generated uncompressed lane's identity
free-text plan uses the same compressed representation/converter as the other
lanes, so identity copies are counted too. A wholly raw pin (`--plan-dir`
omitted), or a chunk retained raw due to size/ratio gates, bypasses the converter
and produces no decode records. A zero count means no observed calls; older
extensions without the new instrumentation also produce zero records.

Concurrent chunk decodes on separate streams count at face value in the sum.
This measures decode work, **not the wall time it occupied**. The wall-visible
penalty is the execution-phase difference between a compressed lane and the
uncompressed lane on the same box. The profiler itself can affect timing:
compare hot wall medians/stddev for identical queries/settings with profiling
and `--no-profile` in separate output directories. CPU tests verify capture and
teardown; agreement within noise on the G4 GPU box still needs measurement.

## Footprints and budget sweeps

The engine emits one INFO record per retained column per pin batch, for compressed
and raw GPU/host chunks. `footprint.csv` sums these by column, tier and entry.
GPU compressed sizes come from the reconstructed retained columns; compressed
host sizes sum serialized column payloads. Raw GPU sizes use cuDF allocation
sizes; raw host sizes use retained column buffers. Uncompressed sizes use the
compression-input cuDF allocation size (after any configured carrier narrowing).
These measurements exclude structural headers, host pool block rounding,
GPU slab alignment gaps, allocator bookkeeping and query workspace. They are
column storage measurements; leave headroom for those other allocations.
Missing resident records remain blank and cannot generate a budget layout.

Start with one all-GPU measurement run per lane using the default layout, then:

```bash
pixi run python test/clickbench_performance/gen_pin_layout.py \
  "$HOME/results/clickbench-10m/onpair/footprint.csv" \
  --budget-bytes 20000000000 --output /tmp/onpair-layout.json
```

The pre-declared placement order is query frequency descending, resident bytes
ascending, then column name. Walk that order, placing a column on GPU only when
it fits; continue past larger columns that do not fit. `hits` contains the GPU
subset. `main.hits` on host contains **every queried column**, including those
on GPU. That duplicates GPU columns in host memory because a scan must find its
entire column set in one entry; Sirius cannot combine entries. The engine
prefers GPU, then host, then the smallest column superset, then entry name.
Thus exactly queries whose columns are a subset of `hits` use GPU storage;
others use host storage (query execution itself is still on GPU). At zero budget
only the host entry is emitted; at a full budget both complete entries are emitted.
The generator requires a complete all-GPU per-column footprint. Layout loading
checks both union coverage and serviceability of each query by a single entry.

Use `--pin-layout /tmp/onpair-layout.json` and `--gpu-usage-limit-bytes N` for
each sweep run. The all-GPU measurement and sweep must use the same plan and pin
settings; reserve workspace beyond the column budget. Per-entry compression
coverage and measured sweep footprints reveal any chunking or ratio changes.

Replication is prepared once, on the data box, for k=2 or k=4:

```bash
pixi run python test/clickbench_performance/replicate_hits.py \
  --parquet "$HOME/data/hits.parquet" --copies 2 \
  --output "$HOME/data/hits-2x"
```

Then use `--parquet "$HOME/data/hits-2x/*.parquet"` in all lanes. Each output
part uses DuckDB COPY with preserved row order and a single writer thread;
values/rows are verbatim, Parquet encoding need not be byte-identical. Existing
directories are never overwritten. No replication runs in the unit tests or on
this development machine. Validation on a replicated input currently runs
DuckDB on that input; derivation from the 1x oracle for very large sweeps is
future work, so do not mistake `--no-validate` for that derivation.

## Compare lanes

```bash
pixi run python test/clickbench_performance/report.py \
  "$HOME/results/clickbench-10m/uncompressed" \
  "$HOME/results/clickbench-10m/general" \
  "$HOME/results/clickbench-10m/onpair"
```

Two or three lane directories are accepted. Input paths, query subsets, effective
SQL, source mode, native database path, profiling setting, extension SHA256,
effective YAML SHA256, pre-SQL, DuckDB memory limit and
execution-affecting environment (`SIRIUS_EXP_*`, `SIRIUS_LATE_MAT*`,
`SIRIUS_PIN_TIER*`, `LD_PRELOAD`) must agree. Empty environment values are treated
as unset; config/log/output paths are excluded from the environment comparison.
Plan hashes and layouts are the declared differences; git revisions are recorded, while the
loaded extension hash identifies the build. Compression-failed lanes are refused
unless `--allow-compression-failure` is passed, which prints an override notice.

The union of fallback, disk-served and validation-failed queries in any lane is
printed with reasons and excluded from every lane geomean and pairwise speedup.
Tier differences, including a query changing tiers between repetitions, appear
in their own `layout_difference` column and are excluded from codec aggregates.
All other queries, including unaffected ones, remain in the geomean. Any pair
with `--no-validate` on either side is labeled **UNVALIDATED**, with no meaningful
per-query claims. Footprints and pin time are printed per lane; unknown sizes
remain `None`. Each report reminds readers that Q17 adds ordered top-k and Q24
is identical to Q26, weighting that workload twice in this variant's geomean.

A change is pre-declared meaningful when both hold:

- Absolute median time change is at least 5% of the baseline median.
- Absolute time difference exceeds `3 * sqrt(std_baseline^2 + std_candidate^2)`.

These are hot-repetition sample standard deviations, **not** standard errors.
At least two hot repetitions per lane are needed for a meaningful-change claim.
Negative time changes are improvements. Cold/pin timings never enter this rule.
The report does not turn CPU fallback, failed validation, or missing sizes into
performance claims.

Add `--breakdown` to print hot medians per query and lane:
`wall_ms | non_execution_floor_ms | exec_ms | decode_ms_sum`, where the floor
is planner + optimizer + physical + remainder. Supply the **uncompressed lane
first**. For every subsequent named lane the report also prints:

- `exec_penalty_ms = exec_lane - exec_base`.
- `wall_penalty_ms = wall_lane - wall_base`.
- `zero_decode_penalty_speedup_bound = wall_lane / (wall_lane - exec_penalty)`:
  the estimated speedup if the extra execution cost vanished with the floor
  fixed. This attributes the entire execution difference to decode; it is not
  derived from `decode_ms_sum`.

The free-text query set is derived from `QUERY_COLUMNS` in
`clickbench_queries.py`: queries touching URL, Title, Referer, SearchPhrase or
OriginalURL. The report prints geometric means of both penalties and the bound
over the eligible intersection with that set, applying the usual exclusions.
Signed per-query penalties are retained. If any selected value is missing or
nonpositive, its geometric mean is `n/a` with counts, rather than silently
dropping those queries. A nonpositive bound denominator is also `n/a`.
Runs with `--no-profile` or older CSVs show unavailable phases/bounds as `n/a`;
wall-only comparisons continue to work. Compare profiling on/off wall medians
directly: `report.py` requires matching profiling settings across codec lanes.
