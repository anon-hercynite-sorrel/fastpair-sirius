# onpair-bridge

C ABI over the [`onpair`](https://crates.io/crates/onpair) crate for Simpatico's `onpair`
operator. The operator trains and encodes on the host at pin time through
`onpair_bridge_encode32/64` and stores five compact channels: `codes`, `row_offsets`
(the input's byte offsets, INT32/INT64), `dict_bytes`, `dict_offsets`, and `batch_prefixes`
(UINT64 decoded byte offsets computed on the GPU at pin time). By default each batch contains
128 tokens. Setting `SIRIUS_EXP_ONPAIR_K=6` before compression selects 192-token batches
and the six-token-per-lane decoder. Decoding uses the batch size stored with the column.
The bridge still returns token offsets in its own `row_offsets`; the operator does not
upload them. GPU decode is the operator's own kernel. The CPU decode here backs
round-trip tests and `simpatico compress --verify`.

- `include/onpair_bridge.h` — the API; `src/lib.rs` documents the encoded layout.
- `cargo test` — round-trip tests through the C ABI (empty, all-empty, single row, rows longer
  than one 128-token decode batch, sliced offsets, int64 offsets, bad input).
- `ctest/run.sh <file>...` — builds the staticlib and a C program that round-trips a
  newline-delimited text file and prints encode/decode throughput and ratio.

Built by CMake through Corrosion (see `cmake/onpair_bridge.cmake`); plain `cargo` works too.

Training runs once over the full column (the crate samples internally). Parsing shares the
trained `Parser` by reference across scoped standard-library threads; `Parser` is `Sync`.
`onpair_bridge_config.threads` defaults to 0 (`std::thread::available_parallelism()`, or 1
if unavailable); 1 forces serial parsing. The number of balanced, contiguous row chunks is
`min(thread_cap, max(1, num_rows / 65536))`, so every parallel chunk has at least 64K rows.
Chunks preserve row order and produce the same dictionary, codes, and rebased token offsets
as serial encoding for a fixed training seed. The encoded format is unchanged.

The C benchmark accepts `<file> [bits] [threshold] [threads]`. Its auto mode resolves the cap
from the POSIX online CPU count, caps it by the same minimum chunk size, and passes that
explicit count to the bridge so `threads=` reports the actual parsing worker count
(1 means parsing on the caller). Run the built `target/roundtrip` with a final argument of
1 for a serial baseline or a larger count for parallel throughput comparisons.
