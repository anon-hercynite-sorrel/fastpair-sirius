// SPDX-License-Identifier: Apache-2.0
#include "codegen/util/cuda_check.hpp"
#include "decode/decode_session.hpp"
#include "nvcomp_chunk_select.hpp"

#include <cudf/null_mask.hpp>

#include <rmm/device_buffer.hpp>

#include <cub/device/device_select.cuh>
#include <cuda_runtime.h>
#include <thrust/iterator/counting_iterator.h>

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

namespace simpatico {
namespace detail {

namespace {

// Device header ahead of the compacted ids, read back with them in one copy.
struct chunk_select_header {
  std::uint32_t status      = 0;  // nonzero: a survivor's range left the chars buffer
  std::uint32_t count       = 0;  // marked chunks
  std::int64_t output_bytes = 0;
};
static_assert(sizeof(chunk_select_header) == 16);

constexpr int kThreads = 256;

unsigned grid_for(std::int64_t items)
{
  return static_cast<unsigned>(
    std::clamp<std::int64_t>((items + kThreads - 1) / kThreads, 1, 8192));
}

__global__ void gather_validity(std::uint32_t const* source,
                                std::int32_t const* indices,
                                std::int64_t survivors,
                                std::uint32_t* mask,
                                std::int32_t* lengths)
{
  auto const thread = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x;
  auto const lane   = threadIdx.x % 32;
  for (auto base = thread - lane; base < survivors;
       base += static_cast<std::int64_t>(gridDim.x) * blockDim.x) {
    auto const row = base + lane;
    bool valid     = false;
    if (row < survivors) {
      auto const index = indices[row];
      valid            = (source[index / 32] & (1u << (index % 32))) != 0;
      if (!valid) lengths[row] = 0;
    }
    auto const bits = __ballot_sync(0xffffffffu, valid);
    if (lane == 0) mask[base / 32] = bits;
  }
}

__global__ void mark_survivor_chunks(std::int64_t const* __restrict__ starts,
                                     std::int32_t const* __restrict__ lengths,
                                     std::int64_t survivors,
                                     std::uint64_t chars_bytes,
                                     std::uint64_t chunk_bytes,
                                     std::uint32_t* __restrict__ flags,
                                     chunk_select_header* header)
{
  for (std::int64_t j = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x;
       j < survivors;
       j += static_cast<std::int64_t>(gridDim.x) * blockDim.x) {
    auto const length = lengths[j];
    if (length == 0) continue;
    auto const start = starts[j];
    if (length < 0 || start < 0 || static_cast<std::uint64_t>(start) > chars_bytes ||
        static_cast<std::uint64_t>(length) > chars_bytes - static_cast<std::uint64_t>(start)) {
      atomicOr(&header->status, 1u);
      continue;
    }
    auto const first = static_cast<std::uint64_t>(start) / chunk_bytes;
    auto const last  = (static_cast<std::uint64_t>(start) + length - 1) / chunk_bytes;
    for (auto c = first; c <= last; ++c)
      atomicOr(&flags[c], 1u);
  }
}

__global__ void remap_survivor_starts(std::int64_t* __restrict__ starts,
                                      std::int32_t const* __restrict__ lengths,
                                      std::int32_t const* __restrict__ output_offsets,
                                      std::int64_t survivors,
                                      std::uint64_t chunk_bytes,
                                      std::uint32_t const* __restrict__ ids,
                                      chunk_select_header* header)
{
  auto const thread = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x;
  if (thread == 0) header->output_bytes = output_offsets[survivors];
  // A rejected range may lie past the marked chunks; the host refuses the decode instead.
  if (header->status != 0) return;
  auto const count = header->count;
  for (std::int64_t j = thread; j < survivors;
       j += static_cast<std::int64_t>(gridDim.x) * blockDim.x) {
    if (lengths[j] == 0) {
      starts[j] = 0;
      continue;
    }
    auto const start = static_cast<std::uint64_t>(starts[j]);
    auto const first = start / chunk_bytes;
    // The ids are ascending and include `first`, so its slot is its lower bound.
    std::uint32_t lo = 0;
    std::uint32_t hi = count;
    while (lo < hi) {
      auto const mid = lo + (hi - lo) / 2;
      if (ids[mid] < first)
        lo = mid + 1;
      else
        hi = mid;
    }
    starts[j] = static_cast<std::int64_t>(lo * chunk_bytes + (start - first * chunk_bytes));
  }
}

}  // namespace

rmm::device_buffer select_string_validity(validity_sidecar const& validity,
                                          sirius::codegen::selection_mask const& selection,
                                          std::int32_t* lengths,
                                          cudf::size_type& null_count,
                                          decode_frame& frame)
{
  auto const survivors = static_cast<cudf::size_type>(selection.survivor_count);
  auto const stream    = frame.stream();
  if (validity.kind == validity_kind::all_null) {
    throw_if_cuda_error(cudaMemsetAsync(lengths, 0, sizeof(*lengths) * survivors, stream.get()),
                        "selected strings: clear null lengths");
    null_count = survivors;
    return cudf::create_null_mask(survivors, cudf::mask_state::ALL_NULL, stream, frame.mr());
  }
  rmm::device_buffer indices(sizeof(std::int32_t) * survivors, stream, frame.scratch_mr());
  auto* rows = static_cast<std::int32_t*>(indices.data());
  sirius::codegen::mask_to_row_indices(selection, rows, stream);
  auto mask =
    cudf::create_null_mask(survivors, cudf::mask_state::UNINITIALIZED, stream, frame.mr());
  auto* bits = static_cast<cudf::bitmask_type*>(mask.data());
  gather_validity<<<grid_for(survivors), kThreads, 0, stream.get()>>>(
    static_cast<std::uint32_t const*>(validity.mask.data()), rows, survivors, bits, lengths);
  throw_if_cuda_error(cudaGetLastError(), "selected strings: gather validity");
  null_count = cudf::null_count(bits, 0, survivors, stream);
  return mask;
}

selected_frame_chunks select_frame_chunks(std::int64_t* source_offsets,
                                          std::int32_t const* lengths,
                                          std::int32_t const* output_offsets,
                                          std::int64_t survivors,
                                          std::size_t chars_bytes,
                                          std::size_t chunk_bytes,
                                          decode_frame& frame)
{
  if (chars_bytes && chunk_bytes == 0) throw std::invalid_argument("chunk select: zero chunk size");
  auto const chunks = chars_bytes == 0 ? std::size_t{0} : 1 + (chars_bytes - 1) / chunk_bytes;
  if (chunks > std::numeric_limits<std::uint32_t>::max())
    throw std::invalid_argument("chunk select: too many frame chunks");
  auto const stream = frame.stream();
  auto const mr     = frame.scratch_mr();
  auto const ids_at = sizeof(chunk_select_header);
  rmm::device_buffer selected(ids_at + chunks * sizeof(std::uint32_t), stream, mr);
  auto* const header = static_cast<chunk_select_header*>(selected.data());
  auto* const ids =
    reinterpret_cast<std::uint32_t*>(static_cast<std::byte*>(selected.data()) + ids_at);
  throw_if_cuda_error(cudaMemsetAsync(header, 0, sizeof(chunk_select_header), stream.get()),
                      "chunk select: clear header");
  rmm::device_buffer flags(std::max<std::size_t>(chunks, 1) * sizeof(std::uint32_t), stream, mr);
  throw_if_cuda_error(cudaMemsetAsync(flags.data(), 0, flags.size(), stream.get()),
                      "chunk select: clear chunk flags");
  if (survivors > 0) {
    mark_survivor_chunks<<<grid_for(survivors), kThreads, 0, stream.get()>>>(
      source_offsets,
      lengths,
      survivors,
      chars_bytes,
      chunk_bytes,
      static_cast<std::uint32_t*>(flags.data()),
      header);
    throw_if_cuda_error(cudaGetLastError(), "chunk select: mark survivor chunks");
  }
  if (chunks > 0) {
    auto const items = thrust::make_counting_iterator(std::uint32_t{0});
    auto select      = [&](void* temp, std::size_t& bytes) {
      return cub::DeviceSelect::Flagged(temp,
                                        bytes,
                                        items,
                                        static_cast<std::uint32_t const*>(flags.data()),
                                        ids,
                                        &header->count,
                                        static_cast<std::int64_t>(chunks),
                                        stream.get());
    };
    std::size_t temp_bytes = 0;
    throw_if_cuda_error(select(nullptr, temp_bytes), "chunk select: compaction scratch size");
    rmm::device_buffer temp(temp_bytes, stream, mr);
    throw_if_cuda_error(select(temp.data(), temp_bytes), "chunk select: compact marked chunks");
  }
  remap_survivor_starts<<<grid_for(survivors), kThreads, 0, stream.get()>>>(
    source_offsets, lengths, output_offsets, survivors, chunk_bytes, ids, header);
  throw_if_cuda_error(cudaGetLastError(), "chunk select: remap survivor starts");

  std::vector<std::byte> host(selected.size());
  frame.read_bytes(host.data(), selected.data(), host.size());
  chunk_select_header totals;
  std::memcpy(&totals, host.data(), sizeof(totals));
  if (totals.status != 0 || totals.count > chunks)
    throw std::runtime_error("chunk select: survivor byte range outside the chars frame");
  if (totals.output_bytes < 0) throw std::runtime_error("decode: selected string size overflow");
  selected_frame_chunks result;
  result.output_bytes = totals.output_bytes;
  result.ids.resize(totals.count);
  if (totals.count > 0)
    std::memcpy(result.ids.data(), host.data() + ids_at, totals.count * sizeof(std::uint32_t));
  return result;
}

}  // namespace detail
}  // namespace simpatico
