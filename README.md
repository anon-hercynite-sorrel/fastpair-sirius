# FastPair-Sirius

This repository contains the Sirius GPU SQL engine integration used for the end-to-end ClickBench
evaluation (Section 5.4) of *FastPair: GPU-Optimized String Decoding*.

Companion repositories:
- [fastpair-artifacts](https://github.com/anon-hercynite-sorrel/fastpair-artifacts): recorded
  measurements, figure generators, and verification scripts.
- [fastpair-harness](https://github.com/anon-hercynite-sorrel/fastpair-harness): standalone
  FastPair CUDA kernels and microbenchmark harness.

## Key additions over upstream Sirius

- FastPair (OnPair-16, `K=6` / 192-token batches) string compression and GPU decoding under
  `src/compression/simpatico_codegen/` (`onpair_compressor.cu`, `onpair_kernels.cuh`,
  `onpair_k6.cuh`, `onpair_like.cu`, and `rust/onpair-bridge/`).
- Chunk-selective string decoding for nvCOMP codecs (Snappy, LZ4, GDeflate, gANS) and selectable
  Blackwell Decompression Engine (DE) routing for Snappy and LZ4.
- End-to-end ClickBench benchmark harness under [test/clickbench_performance/](test/clickbench_performance/README.md).
- `cucascade/` is vendored directly in this snapshot (including CUDA driver init and MIG NVML UUID
  fixes); initialize the remaining upstream submodules with `git submodule update --init --recursive`.

## Building

See [docs/building.md](docs/building.md) and [test/clickbench_performance/README.md](test/clickbench_performance/README.md)
for build and benchmark instructions (`pixi run make`).
