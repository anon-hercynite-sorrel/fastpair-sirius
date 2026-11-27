// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#pragma once

#include "operators/onpair_layout.cuh"

#include <cuda_runtime.h>

namespace simpatico::onpair_detail::k6 {

// FastPair onpair_decompress_tpt.cuh, configuration onpair_dg_k6_t256_b4.
// Each warp stages up to 16 bytes/token plus alignment padding. Block size
// controls both the launch bound and the number of per-warp shared arrays.
constexpr unsigned tokens_per_thread = 6;
constexpr unsigned block_threads     = 256;
constexpr unsigned warps_per_block   = block_threads / 32;
constexpr unsigned tokens_per_warp   = tokens_per_thread * 32;
constexpr unsigned warp_buf_bytes    = tokens_per_warp * 16 + 32;
// A high request encodes its destination in bits [27:16], limiting K to 8.
static_assert(16 * (tokens_per_warp - 1) + 8 <= 0xfff);

__device__ inline uint64_t warp_scan_four_u16_prefixes(uint64_t x, int lane)
{
  constexpr unsigned mask = 0xffffffffu;
#pragma unroll
  for (int offset = 1; offset < 32; offset <<= 1) {
    const uint64_t y = __shfl_up_sync(mask, x, offset);
    if (lane >= offset) { x += y; }
  }
  return x;
}

// Layout: code[15:0], shared destination[27:16], high_length-1[30:28].
__device__ inline uint32_t pack_high_request(uint32_t code,
                                             uint32_t destination,
                                             uint32_t high_length)
{ return code | (destination << 16u) | ((high_length - 1u) << 28u); }

__device__ inline void emit_high_bytes(uint8_t* s_buf,
                                       uint32_t destination,
                                       uint32_t high_length,
                                       const uint2& high)
{
  const uint8_t* bytes = reinterpret_cast<const uint8_t*>(&high);
#pragma unroll
  for (int byte = 0; byte < 8; ++byte) {
    if (byte < (int)high_length) { s_buf[destination + (uint32_t)byte] = bytes[byte]; }
  }
}

// Selected warps use compact scratch offsets and fetch their source batch from marked.
// Both instantiations use the same dictionary gathers, queue, staging and drain.
template <bool Selected>
__global__ __launch_bounds__(block_threads, 4) void decode(const uint16_t* __restrict codes,
                                                           const uint64_t* __restrict chunk_offsets,
                                                           const uint8_t* __restrict dict_s8_lo,
                                                           const uint8_t* __restrict dict_s8_hi,
                                                           const uint8_t* __restrict packed_lens,
                                                           uint8_t* __restrict output_bytes,
                                                           uint64_t total_tokens,
                                                           const uint32_t* __restrict marked,
                                                           uint32_t marked_count)
{
  constexpr unsigned mask = 0xffffffffu;
  const int lane          = threadIdx.x & 31;
  const uint32_t warp_id  = threadIdx.x >> 5;
  if (warp_id >= warps_per_block) return;
  const uint64_t warp = (uint64_t)blockIdx.x * (blockDim.x >> 5) + warp_id;
  uint64_t chunk      = warp;
  if constexpr (Selected) {
    if (warp >= marked_count) return;
    chunk = marked[warp];
  } else if (chunk * tokens_per_warp >= total_tokens) {
    return;
  }

  __shared__ __align__(16) uint8_t s_buf_all[warps_per_block * warp_buf_bytes];
  __shared__ __align__(16) uint32_t s_requests[warps_per_block][tokens_per_warp];
  uint32_t* requests  = s_requests[warp_id];
  uint8_t* s_buf_base = &s_buf_all[warp_id * warp_buf_bytes];

  const uint64_t base_i = chunk * tokens_per_warp + (uint64_t)lane;
  uint2 lo[tokens_per_thread];
  uint32_t code[tokens_per_thread];
  uint32_t len[tokens_per_thread];
#pragma unroll
  for (int k = 0; k < (int)tokens_per_thread; ++k) {
    const uint64_t i = base_i + (uint64_t)(k * 32);
    if (i < total_tokens) {
      code[k] = (uint32_t)codes[i];
      lo[k]   = *reinterpret_cast<const uint2*>(dict_s8_lo + (size_t)code[k] * 8u);
      len[k]  = token_length<6>(packed_lens, code[k]);
    } else {
      code[k] = 0u;
      lo[k]   = make_uint2(0u, 0u);
      len[k]  = 0u;
    }
  }

  constexpr uint64_t field_mask = 0xffffull;
  static_assert(32u * 16u <= field_mask, "packed fields must hold a full plane prefix");
  uint32_t excl[tokens_per_thread];
  uint32_t acc_base = 0u;
#pragma unroll
  for (int group = 0; group < (int)tokens_per_thread; group += 4) {
    uint64_t packed = 0u;
#pragma unroll
    for (int field = 0; field < 4; ++field) {
      const int k = group + field;
      if (k < (int)tokens_per_thread) { packed |= (uint64_t)len[k] << ((uint32_t)field * 16u); }
    }
    packed                       = warp_scan_four_u16_prefixes(packed, lane);
    const uint64_t packed_totals = __shfl_sync(mask, packed, 31);
#pragma unroll
    for (int field = 0; field < 4; ++field) {
      const int k = group + field;
      if (k < (int)tokens_per_thread) {
        const uint32_t shift       = (uint32_t)field * 16u;
        const uint32_t incl        = (uint32_t)((packed >> shift) & field_mask);
        const uint32_t plane_total = (uint32_t)((packed_totals >> shift) & field_mask);
        excl[k]                    = acc_base + incl - len[k];
        acc_base += plane_total;
      }
    }
  }
  const uint32_t warp_total = acc_base;

  const uint64_t out_start = chunk_offsets[warp];
  const uint32_t head_pre  = (16u - (uint32_t)(out_start & 15u)) & 15u;
  uint8_t* s_buf           = s_buf_base + ((16u - head_pre) & 15u);

  // Compact long-token requests in plane order so each lane can fetch a high half.
  uint32_t high_count = 0u;
#pragma unroll
  for (int k = 0; k < (int)tokens_per_thread; ++k) {
    const bool needs_high      = len[k] > 8u;
    const uint32_t needs_mask  = __ballot_sync(mask, needs_high);
    const uint32_t lower_lanes = lane == 0 ? 0u : ((1u << (uint32_t)lane) - 1u);
    const uint32_t rank        = __popc(needs_mask & lower_lanes);
    if (needs_high) {
      requests[high_count + rank] = pack_high_request(code[k], excl[k] + 8u, len[k] - 8u);
    }
    high_count += __popc(needs_mask);
  }
  __syncwarp();

  // H=1: issue one round of high-plane loads before emitting the low bytes.
  const bool held      = (uint32_t)lane < high_count;
  uint32_t destination = 0, high_length = 0;
  uint2 high = make_uint2(0u, 0u);
  if (held) {
    const uint32_t request = requests[lane];
    destination            = (request >> 16u) & 0xfffu;
    high_length            = (request >> 28u) + 1u;
    high = *reinterpret_cast<const uint2*>(dict_s8_hi + (size_t)(request & 0xffffu) * 8u);
  }

  // The first high value is deliberately not consumed until all owners have
  // emitted their low bytes, creating independent instructions after LDG.
#pragma unroll
  for (int k = 0; k < (int)tokens_per_thread; ++k) {
    const uint32_t low_length = len[k] < 8u ? len[k] : 8u;
    const uint8_t* bytes      = reinterpret_cast<const uint8_t*>(&lo[k]);
#pragma unroll
    for (int byte = 0; byte < 8; ++byte) {
      if (byte < (int)low_length) { s_buf[excl[k] + (uint32_t)byte] = bytes[byte]; }
    }
  }

  if (held) emit_high_bytes(s_buf, destination, high_length, high);

  // The remaining rounds retain the baseline's dense queue drain.
#pragma unroll
  for (uint32_t round = 1u; round < tokens_per_thread; ++round) {
    const uint32_t request_idx = (uint32_t)lane + round * 32u;
    if (request_idx < high_count) {
      const uint32_t request       = requests[request_idx];
      const uint32_t selected_code = request & 0xffffu;
      const uint32_t destination   = (request >> 16u) & 0xfffu;
      const uint32_t high_length   = (request >> 28u) + 1u;
      const uint2 high = *reinterpret_cast<const uint2*>(dict_s8_hi + (size_t)selected_code * 8u);
      emit_high_bytes(s_buf, destination, high_length, high);
    }
  }
  __syncwarp();

  const uint32_t head = head_pre < warp_total ? head_pre : warp_total;
  if ((uint32_t)lane < head) { output_bytes[out_start + (uint64_t)lane] = s_buf[lane]; }
  if (head >= warp_total) { return; }

  const uint32_t body_chunks = (warp_total - head) >> 4;
  for (uint32_t k = (uint32_t)lane; k < body_chunks; k += 32u) {
    const uint32_t off = head + k * 16u;
    const uint4 value  = *reinterpret_cast<const uint4*>(s_buf + off);
    __stcs(reinterpret_cast<uint4*>(output_bytes + out_start + off), value);
  }

  const uint32_t tail_start = head + (body_chunks << 4);
  if ((uint32_t)lane < warp_total - tail_start) {
    output_bytes[out_start + (uint64_t)tail_start + (uint64_t)lane] = s_buf[tail_start + lane];
  }
}

}  // namespace simpatico::onpair_detail::k6
