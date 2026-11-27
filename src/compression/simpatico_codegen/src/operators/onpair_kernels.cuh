// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
#pragma once

// Ported from vortex-cuda/kernels/src/onpair_shmem_4tpt_split8read.cu
// and vortex-cuda/cub/kernels/onpair.cu. Kernel bodies, launch bounds,
// split dictionary layout, and 128-token batches are preserved. Only u16
// codes are instantiated; the C ABI host wrappers are internal C++ helpers.
#include "operators/onpair_layout.cuh"

#include <cub/cub.cuh>
#include <cuda.h>
#include <cuda_runtime.h>

#include <limits.h>
#include <stdint.h>

namespace simpatico::onpair_detail {

// OnPair decompress — 4 tokens/thread, split-read dictionary.
//
// One warp decodes one 128-token batch (4 tokens per lane) in four phases:
//
// 1. Load   — each lane fetches its 4 codes, their first-8 dictionary bytes
//             (`dict_s8`), and their lengths.
// 2. Scan   — a warp prefix-scan of the lengths positions every token within
//             the batch and yields the batch's total decoded size.
// 3. Stage  — token bytes are gathered into a per-warp shared staging buffer;
//             only the rare `len > 8` tokens touch the full 16-byte-row
//             `dict_padded`.
// 4. Drain  — the staged bytes stream to global output as an aligned `uint4`
//             body between a byte head and tail.
//
// Baseline `onpair_shmem_4tpt` is L1/TEX-cache-request bound on the per-token
// 16-byte `uint4` gather into the 64 KB padded dict, where the dict L1 hit rate
// is only ~31% (the 64 KB dict thrashes against the streaming codes/output).
//
// Most tokens are short (mean dict len ~6). This variant reads the common case
// from the **32 KB** `dict_s8` array (first 8 bytes/entry, `uint2`) and only
// touches the 64 KB `dict_padded` for the rare `len > 8` tokens. Halving the
// hot dict working set aims to raise the dict L1 hit rate, cutting L2 sectors
// and L1/TEX-request pressure. As a bonus, holding `uint2 lo[4]` (32 B) instead
// of `uint4 t[4]` (64 B) lowers register pressure.

#ifndef WARPS_PER_BLOCK_MAX
#define WARPS_PER_BLOCK_MAX 16u
#endif
// The launch bounds are an exception, not a pattern: this kernel was tuned on
// GH200 where NCU showed it register-capped at 64 (50% occupancy); bounding to
// 2 blocks/SM was benchmarked as the best trade against spills. Re-evaluate on
// new architectures rather than copying this to other kernels.
#ifndef ONPAIR_LAUNCH_BOUNDS
#define ONPAIR_LAUNCH_BOUNDS __launch_bounds__(512, 2)
#endif
#define WARP_BUF_BYTES 2080u

__device__ inline uint32_t onpair_warp_inclusive_scan_u32(uint32_t x, uint32_t lane)
{
  constexpr unsigned mask = 0xffffffffu;
#pragma unroll
  for (uint8_t offset = 1; offset < 32; offset <<= 1) {
    uint32_t y = __shfl_up_sync(mask, x, offset);
    if (lane >= offset) { x += y; }
  }
  return x;
}

// One lane's slice of a 128-token batch: 4 tokens, strided a warp apart.
struct OnPairTokens {
  // First 8 dictionary bytes of each token (the common-case read).
  uint2 lo[4];
  // Dictionary codes, kept for the rare `len > 8` high-byte gather.
  uint32_t code[4];
  // Decoded token lengths.
  uint32_t len[4];
};

// Phase 1 — load this lane's 4 (code, dict_s8 bytes, length) triples. Tokens
// outside the visible window load as empty. `CodeT` is the code stream's
// element type (u16 natively, u8 when the compressor narrowed the codes).
template <typename CodeT>
__device__ inline OnPairTokens onpair_load_tokens(const CodeT* __restrict codes,
                                                  const uint8_t* __restrict dict_s8,
                                                  const uint8_t* __restrict lens,
                                                  uint64_t base_i,
                                                  uint64_t token_start,
                                                  uint64_t token_end)
{
  OnPairTokens t;
#pragma unroll
  for (uint8_t k = 0; k < 4; ++k) {
    const uint64_t i = base_i + (uint64_t)(k * 32);
    if (i >= token_start && i < token_end) {
      const uint32_t code = (uint32_t)codes[i];
      t.code[k]           = code;
      t.lo[k]             = *reinterpret_cast<const uint2*>(dict_s8 + (size_t)code * 8u);
      t.len[k]            = (uint32_t)lens[code];
    } else {
      t.code[k] = 0u;
      t.lo[k]   = make_uint2(0u, 0u);
      t.len[k]  = 0u;
    }
  }
  return t;
}

// Phase 2 — position every token within the batch: `excl[k]` is the exclusive
// prefix (the token's staging offset) via 4 chained warp scans of the lengths.
// Returns the batch's total decoded byte count.
__device__ inline uint32_t onpair_scan_offsets(const uint32_t (&len)[4],
                                               uint32_t lane,
                                               uint32_t (&excl)[4])
{
  constexpr unsigned mask = 0xffffffffu;
  uint32_t acc_base       = 0u;
#pragma unroll
  for (uint8_t k = 0; k < 4; ++k) {
    const uint32_t incl = onpair_warp_inclusive_scan_u32(len[k], lane);
    excl[k]             = acc_base + (incl - len[k]);
    acc_base += __shfl_sync(mask, incl, 31);
  }
  return acc_base;
}

// Phase 3 — gather each token's bytes into the warp's shared staging buffer at
// its scanned offset. The common case writes only the 8 `dict_s8` bytes
// already in registers; tokens longer than 8 bytes take the rare path through
// the full padded dictionary.
__device__ inline void onpair_stage_tokens(const OnPairTokens& t,
                                           const uint32_t (&excl)[4],
                                           const uint8_t* __restrict dict_padded,
                                           uint8_t* __restrict s_buf)
{
#pragma unroll
  for (uint8_t k = 0; k < 4; ++k) {
    const uint32_t len = t.len[k];
    if (len == 0u) { continue; }
    const uint32_t base = excl[k];
    const uint8_t* lob  = reinterpret_cast<const uint8_t*>(&t.lo[k]);
    const uint32_t nlo  = len < 8u ? len : 8u;
#pragma unroll
    for (uint8_t j = 0; j < 8; ++j) {
      if (j < nlo) { s_buf[base + j] = lob[j]; }
    }
    if (len > 8u) {
      // Rare path: high bytes from the full padded dict.
      const uint2 hi = *reinterpret_cast<const uint2*>(dict_padded + (size_t)t.code[k] * 16u + 8u);
      const uint8_t* hib = reinterpret_cast<const uint8_t*>(&hi);
#pragma unroll
      for (uint8_t j = 0; j < 8; ++j) {
        if (8u + j < len) { s_buf[base + 8 + j] = hib[j]; }
      }
    }
  }
}

// Phase 4 — copy the staged batch to global output: a byte head up to the
// first 16-aligned output address, an aligned `uint4` body with streaming
// stores, and a byte tail. The caller offset `s_buf` so shared and global
// 16-byte alignment phases match.
__device__ inline void onpair_drain(const uint8_t* __restrict s_buf,
                                    uint8_t* __restrict output_bytes,
                                    uint64_t out_start,
                                    uint32_t head_pre,
                                    uint32_t warp_total,
                                    uint32_t lane)
{
  const uint32_t head = head_pre < warp_total ? head_pre : warp_total;
  if (lane < head) { output_bytes[out_start + (uint64_t)lane] = s_buf[lane]; }
  if (head >= warp_total) { return; }

  const uint32_t body_chunks = (warp_total - head) >> 4;
  for (uint8_t k = (uint8_t)lane; k < body_chunks; k += 32u) {
    const uint32_t off = head + k * 16u;
    const uint4 v      = *reinterpret_cast<const uint4*>(s_buf + off);
    __stcs(reinterpret_cast<uint4*>(output_bytes + out_start + off), v);
  }

  const uint32_t tail_start = head + (body_chunks << 4);
  if (lane < warp_total - tail_start) {
    output_bytes[out_start + (uint64_t)tail_start + (uint64_t)lane] = s_buf[tail_start + lane];
  }
}

template <typename CodeT>
__device__ inline void onpair_decode_body(const CodeT* __restrict codes,
                                          const uint64_t* __restrict chunk_offsets,
                                          const uint8_t* __restrict dict_s8,
                                          const uint8_t* __restrict dict_padded,
                                          const uint8_t* __restrict lens,
                                          uint8_t* __restrict output_bytes,
                                          uint64_t token_start,
                                          uint64_t token_end,
                                          uint64_t first_batch,
                                          uint64_t byte_start)
{
  const uint32_t lane    = threadIdx.x & 31;
  const uint32_t warp_id = threadIdx.x >> 5;
  const uint64_t window_chunk =
    (uint64_t)blockIdx.x * (uint64_t)(blockDim.x >> 5) + (uint64_t)warp_id;
  const uint64_t chunk = first_batch + window_chunk;
  if (chunk * 128u >= token_end) { return; }

  __shared__ __align__(16) uint8_t s_buf_all[WARPS_PER_BLOCK_MAX * WARP_BUF_BYTES];
  uint8_t* s_buf_base = &s_buf_all[warp_id * WARP_BUF_BYTES];

  const OnPairTokens t =
    onpair_load_tokens(codes, dict_s8, lens, chunk * 128u + (uint64_t)lane, token_start, token_end);

  uint32_t excl[4];
  const uint32_t warp_total = onpair_scan_offsets(t.len, lane, excl);

  // Offset the staging buffer by (out_start % 16) so the drain's global
  // 16-byte stores land aligned when copied from 16-aligned shared reads.
  const uint64_t out_start = chunk == first_batch ? 0 : chunk_offsets[chunk] - byte_start;
  const uint32_t head_pre  = (16u - (uint32_t)(out_start & 15u)) & 15u;
  uint8_t* s_buf           = s_buf_base + ((16u - head_pre) & 15u);

  onpair_stage_tokens(t, excl, dict_padded, s_buf);
  __syncwarp();

  onpair_drain(s_buf, output_bytes, out_start, head_pre, warp_total, lane);
}

__global__ ONPAIR_LAUNCH_BOUNDS void onpair_shmem_4tpt_split8read_u16(
  const uint16_t* __restrict codes,
  const uint64_t* __restrict chunk_offsets,
  const uint8_t* __restrict dict_s8,
  const uint8_t* __restrict dict_padded,
  const uint8_t* __restrict lens,
  uint8_t* __restrict output_bytes,
  uint64_t token_start,
  uint64_t token_end,
  uint64_t first_batch,
  uint64_t byte_start)
{
  onpair_decode_body<uint16_t>(codes,
                               chunk_offsets,
                               dict_s8,
                               dict_padded,
                               lens,
                               output_bytes,
                               token_start,
                               token_end,
                               first_batch,
                               byte_start);
}

// Selected batches keep the full decoder's warp geometry and staging layout. Only the batch
// lookup and destination base differ; all four token/decode phases are shared above.
__global__ ONPAIR_LAUNCH_BOUNDS void onpair_decode_marked_u16(
  const uint16_t* __restrict codes,
  const uint8_t* __restrict dict_s8,
  const uint8_t* __restrict dict_padded,
  const uint8_t* __restrict lens,
  const uint32_t* __restrict marked,
  const uint64_t* __restrict scratch_offsets,
  uint32_t marked_count,
  uint64_t total_tokens,
  uint8_t* __restrict output_bytes)
{
  const uint32_t lane    = threadIdx.x & 31;
  const uint32_t warp_id = threadIdx.x >> 5;
  const uint32_t i       = blockIdx.x * (blockDim.x >> 5) + warp_id;
  if (i >= marked_count) return;
  const uint64_t chunk       = marked[i];
  const uint64_t token_start = chunk * 128u;
  const uint64_t token_end = token_start + 128u < total_tokens ? token_start + 128u : total_tokens;
  __shared__ __align__(16) uint8_t s_buf_all[WARPS_PER_BLOCK_MAX * WARP_BUF_BYTES];
  uint8_t* s_buf_base = &s_buf_all[warp_id * WARP_BUF_BYTES];
  const OnPairTokens t =
    onpair_load_tokens(codes, dict_s8, lens, token_start + (uint64_t)lane, token_start, token_end);
  uint32_t excl[4];
  const uint32_t warp_total = onpair_scan_offsets(t.len, lane, excl);
  // select_batches validated these immutable batch envelopes before its sizing readback.
  const uint64_t out_start = scratch_offsets[i];
  const uint32_t head_pre  = (16u - (uint32_t)(out_start & 15u)) & 15u;
  uint8_t* s_buf           = s_buf_base + ((16u - head_pre) & 15u);
  onpair_stage_tokens(t, excl, dict_padded, s_buf);
  __syncwarp();
  onpair_drain(s_buf, output_bytes, out_start, head_pre, warp_total, lane);
}

#undef WARP_BUF_BYTES
#undef ONPAIR_LAUNCH_BOUNDS
#undef WARPS_PER_BLOCK_MAX

// OnPair per-batch output-offsets regeneration, fused into a single sweep
// kernel: every warp reduces one batch's decoded size from the codes
// and the length LUT, the first warp scans the block's batch sums, and CUB's
// decoupled look-back (`ScanTileState` + `TilePrefixCallbackOp` — the same
// machinery `cub::DeviceScan` is built on) resolves the running prefix of all
// preceding tiles in-kernel. The per-batch sizes live only in registers and
// shared memory; only the offsets are written to global memory.
//
// `onpair_batch_offsets` is the sole entry point. The tile-state init launch
// it performs first is CUB's own requirement — `DeviceScan` runs the identical
// init internally before its sweep.

// One warp reduces one batch. A 512-thread sweep tile covers 16 batches.
constexpr uint32_t ONPAIR_BLOCK_THREADS    = 512;
constexpr uint32_t ONPAIR_WARPS_PER_BLOCK  = ONPAIR_BLOCK_THREADS / 32;

using OnPairTileState = cub::ScanTileState<uint64_t>;

// Version-stable u64 addition functor (avoids deprecated cub thread operators).
struct SumU64 {
  __host__ __device__ inline uint64_t operator()(uint64_t a, uint64_t b) const { return a + b; }
};

using OnPairPrefixOp = cub::TilePrefixCallbackOp<uint64_t, SumU64, OnPairTileState>;

__global__ void onpair_batch_offsets_init(OnPairTileState tile_state, int num_tiles)
{ tile_state.InitializeStatus(num_tiles); }

// The fused sweep, instantiated for the two code widths OnPair stores (u16
// natively, u8 when the compressor narrowed the codes). A code outside the
// dictionary raises `status` to 1 and contributes zero bytes: the host must
// check the flag before trusting the offsets and before launching the decode
// kernel, whose dictionary gathers are unchecked.
template <int K, typename CodeT>
__device__ __forceinline__ void onpair_batch_offsets_sweep_impl(const CodeT* __restrict codes,
                                                                const uint8_t* __restrict lens,
                                                                uint32_t dict_size,
                                                                uint64_t total_tokens,
                                                                uint64_t* __restrict chunk_offsets,
                                                                uint32_t* __restrict status,
                                                                OnPairTileState tile_state,
                                                                int64_t num_batches)
{
  const uint32_t lane = threadIdx.x & 31;
  const uint32_t warp = threadIdx.x >> 5;
  const int tile_idx  = static_cast<int>(blockIdx.x);
  const int64_t batch = (int64_t)tile_idx * ONPAIR_WARPS_PER_BLOCK + (int64_t)warp;

  // Warp-parallel reduction of this batch's decoded size. Code reads are
  // lane-consecutive (coalesced); the length LUT is small and
  // cache-resident. Batches past the end (last tile) contribute zero.
  uint32_t batch_bytes = 0;
  if (batch < num_batches) {
    const uint64_t base = (uint64_t)batch * (uint64_t)(32 * K);
#pragma unroll
    for (uint8_t token = 0; token < K; ++token) {
      const uint64_t i = base + (uint64_t)lane + (uint64_t)(token * 32);
      if (i < total_tokens) {
        const uint32_t code = (uint32_t)codes[i];
        if (code < dict_size) {
          batch_bytes += token_length<K>(lens, code);
        } else {
          atomicMax(status, 1u);
        }
      }
    }
  }
#pragma unroll
  for (uint8_t offset = 16; offset > 0; offset >>= 1) {
    batch_bytes += __shfl_down_sync(0xffffffffu, batch_bytes, offset);
  }

  __shared__ uint64_t warp_sums[ONPAIR_WARPS_PER_BLOCK];
  __shared__ uint64_t warp_excl[ONPAIR_WARPS_PER_BLOCK];
  __shared__ typename OnPairPrefixOp::TempStorage prefix_storage;
  if (lane == 0) { warp_sums[warp] = (uint64_t)batch_bytes; }
  __syncthreads();

  // The first warp scans the tile's batch sums and resolves the running
  // prefix of all preceding tiles via decoupled look-back.
  if (warp == 0) {
    const uint64_t lane_sum = (lane < ONPAIR_WARPS_PER_BLOCK) ? warp_sums[lane] : 0;
    uint64_t inclusive      = lane_sum;
#pragma unroll
    for (uint8_t offset = 1; offset < 32; offset <<= 1) {
      const uint64_t shifted = __shfl_up_sync(0xffffffffu, inclusive, offset);
      if (lane >= offset) { inclusive += shifted; }
    }
    const uint64_t aggregate = __shfl_sync(0xffffffffu, inclusive, 31);

    uint64_t prefix = 0;
    if (tile_idx == 0) {
      if (lane == 0) { tile_state.SetInclusive(0, aggregate); }
    } else {
      // Collective over the first warp, as BlockScan would invoke it.
      // The callback's return value is only defined in lane 0 (its
      // look-back window reduction is a WarpReduce); broadcast it, the
      // same way BlockScan shares the prefix before applying it.
      OnPairPrefixOp prefix_op(tile_state, prefix_storage, SumU64(), tile_idx);
      const uint64_t lane0_prefix = prefix_op(aggregate);
      prefix                      = __shfl_sync(0xffffffffu, lane0_prefix, 0);
    }
    if (lane < ONPAIR_WARPS_PER_BLOCK) { warp_excl[lane] = prefix + inclusive - lane_sum; }
  }
  __syncthreads();

  if (lane == 0 && batch < num_batches) {
    chunk_offsets[batch] = warp_excl[warp];
    // The trailing slot holds the total decoded byte count.
    if (batch == num_batches - 1) {
      chunk_offsets[num_batches] = warp_excl[warp] + warp_sums[warp];
    }
  }
}

template <typename CodeT>
__global__ __launch_bounds__(ONPAIR_BLOCK_THREADS) void onpair_batch_offsets_sweep(
  const CodeT* __restrict codes,
  const uint8_t* __restrict lens,
  uint32_t dict_size,
  uint64_t total_tokens,
  uint64_t* __restrict chunk_offsets,
  uint32_t* __restrict status,
  OnPairTileState tile_state,
  int64_t num_batches)
{
  onpair_batch_offsets_sweep_impl<4>(
    codes, lens, dict_size, total_tokens, chunk_offsets, status, tile_state, num_batches);
}

template <typename CodeT>
__global__ __launch_bounds__(ONPAIR_BLOCK_THREADS) void onpair_batch_offsets_sweep_k6(
  const CodeT* __restrict codes,
  const uint8_t* __restrict lens,
  uint32_t dict_size,
  uint64_t total_tokens,
  uint64_t* __restrict chunk_offsets,
  uint32_t* __restrict status,
  OnPairTileState tile_state,
  int64_t num_batches)
{
  onpair_batch_offsets_sweep_impl<6>(
    codes, lens, dict_size, total_tokens, chunk_offsets, status, tile_state, num_batches);
}

int onpair_num_tiles(int64_t num_batches)
{ return static_cast<int>((num_batches + ONPAIR_WARPS_PER_BLOCK - 1) / ONPAIR_WARPS_PER_BLOCK); }

// Query the look-back tile-state storage for `num_batches` batches.
cudaError_t onpair_batch_offsets_temp_size(size_t* temp_bytes, int64_t num_batches)
{
  if (num_batches < 0 || num_batches / ONPAIR_WARPS_PER_BLOCK >= INT_MAX) {
    return cudaErrorInvalidValue;
  }
  return OnPairTileState::AllocationSize(onpair_num_tiles(num_batches), *temp_bytes);
}

// Regenerate the OnPair decode kernel's per-batch output offsets on `stream`
// in one fused sweep, writing `chunk_offsets[0..num_batches]` where
// `chunk_offsets[b]` is the decoded byte count preceding batch `b` and
// `chunk_offsets[num_batches]` is the total. `code_width` selects the code
// stream's element size in bytes (1 or 2), so narrowed codes never need a
// widening pass. A code outside the dictionary raises `*status` to 1 and
// contributes zero bytes; the caller must check the flag before trusting the
// offsets.
cudaError_t onpair_batch_offsets(void* d_temp,
                                 size_t temp_bytes,
                                 const void* codes,
                                 uint32_t code_width,
                                 const uint8_t* lens,
                                 uint32_t dict_size,
                                 uint64_t total_tokens,
                                 uint64_t* chunk_offsets,
                                 uint32_t* status,
                                 int64_t num_batches,
                                 uint16_t tokens_per_batch,
                                 cudaStream_t stream)
{
  if (num_batches <= 0 || num_batches / ONPAIR_WARPS_PER_BLOCK >= INT_MAX) {
    return cudaErrorInvalidValue;
  }
  const int num_tiles = onpair_num_tiles(num_batches);

  OnPairTileState tile_state;
  cudaError_t err = tile_state.Init(num_tiles, d_temp, temp_bytes);
  if (err != cudaSuccess) { return err; }

  constexpr int INIT_THREADS = 128;
  const int init_blocks      = (num_tiles + INIT_THREADS - 1) / INIT_THREADS;
  onpair_batch_offsets_init<<<init_blocks, INIT_THREADS, 0, stream>>>(tile_state, num_tiles);
  err = cudaGetLastError();
  if (err != cudaSuccess) { return err; }

  if (code_width != 2) return cudaErrorInvalidValue;
  auto const kernel = tokens_per_batch == 192 ? onpair_batch_offsets_sweep_k6<uint16_t>
                                              : onpair_batch_offsets_sweep<uint16_t>;
  kernel<<<num_tiles, ONPAIR_BLOCK_THREADS, 0, stream>>>(static_cast<const uint16_t*>(codes),
                                                         lens,
                                                         dict_size,
                                                         total_tokens,
                                                         chunk_offsets,
                                                         status,
                                                         tile_state,
                                                         num_batches);
  return cudaGetLastError();
}

}  // namespace simpatico::onpair_detail
