// SPDX-License-Identifier: Apache-2.0

#include "codegen/plan/bitjoin_layout.hpp"
#include "codegen/plan/plan_tree.hpp"
#include "codegen/util/cuda_check.hpp"
#include "decode/decode_session.hpp"
#include "onpair_bridge.h"
#include "operators/onpair_compressor.hpp"
#include "operators/onpair_k6.cuh"
#include "operators/onpair_kernels.cuh"
#include "util/host_observation.hpp"
#include "util/onpair_k.hpp"

#include <cudf/strings/strings_column_view.hpp>

#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>

namespace simpatico {
namespace {

constexpr auto kColumnLimit = std::numeric_limits<cudf::size_type>::max();

cudf::size_type column_size(std::uint64_t size)
{
  // cuDF's element-count limit also applies to UINT16 codes, independently of
  // the row byte-offset width. Never truncate a bridge buffer when adopting it.
  if (size > static_cast<std::uint64_t>(kColumnLimit))
    throw std::length_error("onpair: channel exceeds cuDF's INT32 element-count limit");
  return static_cast<cudf::size_type>(size);
}

std::unique_ptr<cudf::column> upload(cudf::type_id type,
                                     void const* data,
                                     std::uint64_t size,
                                     ::cuda::stream_ref stream,
                                     rmm::device_async_resource_ref mr)
{
  auto col = cudf::make_fixed_width_column(
    cudf::data_type{type}, column_size(size), cudf::mask_state::UNALLOCATED, stream, mr);
  if (size != 0)
    throw_if_cuda_error(cudaMemcpyAsync(col->mutable_view().head<void>(),
                                        data,
                                        size * cudf::size_of(col->type()),
                                        cudaMemcpyHostToDevice,
                                        stream.get()),
                        "onpair: upload channel");
  return col;
}

__device__ bool invalid_entry(std::uint32_t code,
                              std::uint32_t start,
                              std::uint32_t end,
                              std::uint32_t dict_size,
                              std::uint32_t bytes_size)
{
  return (code == 0 && start != 0) || end <= start || end - start > 16 || end > bytes_size ||
         (code + 1 == dict_size && bytes_size - end > 16);
}

__global__ void stage_dictionary(std::uint8_t const* bytes,
                                 std::uint32_t const* offsets,
                                 std::uint32_t dict_size,
                                 std::uint32_t bytes_size,
                                 std::uint8_t* s8,
                                 std::uint8_t* padded,
                                 std::uint8_t* lengths,
                                 std::uint32_t* status)
{
  auto const code = blockIdx.x * blockDim.x + threadIdx.x;
  if (code >= dict_size) return;
  auto const start = offsets[code];
  auto const end   = offsets[code + 1];
  // Same layout as stage_dict in vortex-cuda/src/kernel/encodings/onpair.rs.
  // Fail on corrupt persisted dictionaries before any unchecked byte gather. The
  // bridge read-pads dict_bytes by up to 16 bytes past the last token (the CPU
  // decoder needs it), so only trailing bytes beyond that are unreferenced.
  if (invalid_entry(code, start, end, dict_size, bytes_size)) {
    lengths[code] = 0;
    atomicExch(status, 1u);
    return;
  }
  auto const len = end - start;
  lengths[code]  = static_cast<std::uint8_t>(len);
#pragma unroll
  for (int j = 0; j < 16; ++j) {
    auto const b          = j < len ? bytes[start + j] : std::uint8_t{0};
    padded[code * 16 + j] = b;
    if (j < 8) s8[code * 8 + j] = b;
  }
}

// Each thread owns one packed length byte, including the final odd entry.
__global__ void stage_dictionary_k6(std::uint8_t const* bytes,
                                    std::uint32_t const* offsets,
                                    std::uint32_t dict_size,
                                    std::uint32_t bytes_size,
                                    std::uint8_t* low,
                                    std::uint8_t* high,
                                    std::uint8_t* lengths,
                                    std::uint32_t* status)
{
  auto const pair = blockIdx.x * blockDim.x + threadIdx.x;
  if (2 * pair >= dict_size) return;
  std::uint8_t packed = 0;
#pragma unroll
  for (unsigned half = 0; half < 2; ++half) {
    auto const code = 2 * pair + half;
    if (code >= dict_size) break;
    auto const start = offsets[code];
    auto const end   = offsets[code + 1];
    if (invalid_entry(code, start, end, dict_size, bytes_size)) {
      atomicExch(status, 1u);
      continue;
    }
    auto const len = end - start;
    packed |= (len - 1) << (4 * half);
#pragma unroll
    for (unsigned j = 0; j < 8; ++j) {
      low[code * 8 + j]  = j < len ? bytes[start + j] : 0;
      high[code * 8 + j] = 8 + j < len ? bytes[start + 8 + j] : 0;
    }
  }
  lengths[pair] = packed;
}

// Construction-time code validation is safe even when dictionary staging rejected an entry.
template <int K>
__device__ __forceinline__ void validate_codes_impl(std::uint16_t const* codes,
                                                    std::uint64_t count,
                                                    std::uint8_t const* lengths,
                                                    std::uint32_t dict_size,
                                                    std::uint32_t* status,
                                                    unsigned long long* decoded_bytes)
{
  auto const i             = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  unsigned long long bytes = 0;
  if (i < count) {
    auto const code = codes[i];
    if (code >= dict_size)
      atomicExch(status, 1u);
    else
      bytes = onpair_detail::token_length<K>(lengths, code);
  }
  using reduction = cub::BlockReduce<unsigned long long, 256>;
  __shared__ reduction::TempStorage scratch;
  auto const total = reduction(scratch).Sum(bytes);
  if (threadIdx.x == 0) atomicAdd(decoded_bytes, total);
}

__global__ void validate_codes(std::uint16_t const* codes,
                               std::uint64_t count,
                               std::uint8_t const* lengths,
                               std::uint32_t dict_size,
                               std::uint32_t* status,
                               unsigned long long* decoded_bytes)
{ validate_codes_impl<4>(codes, count, lengths, dict_size, status, decoded_bytes); }

__global__ void validate_codes_k6(std::uint16_t const* codes,
                                  std::uint64_t count,
                                  std::uint8_t const* lengths,
                                  std::uint32_t dict_size,
                                  std::uint32_t* status,
                                  unsigned long long* decoded_bytes)
{ validate_codes_impl<6>(codes, count, lengths, dict_size, status, decoded_bytes); }

template <typename ByteOffset>
__global__ void validate_rows(ByteOffset const* offsets,
                              std::uint64_t count,
                              std::uint64_t decoded_bytes,
                              std::uint32_t* status)
{
  auto const i = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= count) return;
  auto const boundary = static_cast<std::uint64_t>(offsets[i]);
  if (boundary > decoded_bytes || (i == 0 && boundary != 0) ||
      (i + 1 == count && boundary != decoded_bytes) || (i > 0 && offsets[i] < offsets[i - 1]))
    atomicExch(status, 1u);
}

template <typename ByteOffset>
__global__ void copy_validated_offsets(ByteOffset const* source,
                                       ByteOffset* destination,
                                       std::uint64_t count,
                                       std::uint32_t const* status)
{
  auto const i = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < count) destination[i] = *status == 0 ? source[i] : ByteOffset{0};
}

// Compare every stored batch_prefixes entry (decoded byte offsets), including
// the origin and final total. Monotonicity alone cannot protect codec writes.
__global__ void validate_batch_prefixes(std::uint64_t const* stored,
                                        std::uint64_t const* computed,
                                        std::uint64_t count,
                                        std::uint32_t* status)
{
  auto const i = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < count && stored[i] != computed[i]) atomicExch(status, 1u);
}

template <typename ByteOffset>
__global__ void selected_row_meta(ByteOffset const* rows,
                                  std::int32_t const* indices,
                                  std::uint32_t survivors,
                                  std::uint32_t num_rows,
                                  std::uint64_t decoded_bytes,
                                  std::int64_t* lengths,
                                  std::int64_t* source,
                                  std::uint32_t* status)
{
  auto const j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j >= survivors) return;
  lengths[j]   = 0;
  source[j]    = 0;
  auto const r = indices[j];
  if (r < 0 || r >= num_rows || (j > 0 && r <= indices[j - 1])) {
    atomicExch(status, 1u);
    return;
  }
  auto const start = static_cast<std::uint64_t>(rows[r]);
  auto const end   = static_cast<std::uint64_t>(rows[r + 1]);
  // The batch envelope is the entire validated decoded byte domain, not a single batch:
  // a row may cross arbitrarily many token batches.
  if (start > end || end > decoded_bytes) {
    atomicExch(status, 1u);
    return;
  }
  lengths[j] = static_cast<std::int64_t>(end - start);
  source[j]  = static_cast<std::int64_t>(start);
}

__device__ std::uint32_t first_byte_batch(std::uint64_t const* prefixes,
                                          std::uint32_t batches,
                                          std::uint64_t byte)
{
  // upper_bound(byte) - 1; an exact boundary belongs to the following batch.
  std::uint32_t lo = 0, hi = batches;
  while (lo < hi) {
    auto const mid = lo + (hi - lo) / 2;
    if (prefixes[mid + 1] <= byte)
      lo = mid + 1;
    else
      hi = mid;
  }
  return lo;
}

__global__ void mark_selected_batches(std::int64_t const* lengths,
                                      std::int64_t const* source,
                                      std::uint32_t survivors,
                                      std::uint64_t const* prefixes,
                                      std::uint32_t batches,
                                      std::uint8_t* flags,
                                      std::uint32_t const* status)
{
  auto const j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j >= survivors || *status != 0 || lengths[j] == 0) return;
  auto const start = static_cast<std::uint64_t>(source[j]);
  auto const end   = start + lengths[j];
  for (auto k = first_byte_batch(prefixes, batches, start); k < batches && prefixes[k] < end; ++k) {
    // CUDA atomics operate on words. Each byte flag owns one bit in a padded word, so overlapping
    // rows and adjacent batches cannot lose updates (or race on a plain byte store).
    atomicOr(reinterpret_cast<std::uint32_t*>(flags) + k / 4, 1u << ((k % 4) * 8));
  }
}

struct marked_byte_length {
  std::uint32_t const* marked;
  std::uint32_t const* count;
  std::uint64_t const* prefixes;
  __host__ __device__ std::uint64_t operator()(std::uint32_t i) const
  {
    if (i >= *count) return 0;
    auto const batch = marked[i];
    return prefixes[batch + 1] - prefixes[batch];
  }
};

__global__ void map_selected_sources(std::int64_t* lengths,
                                     std::int64_t const* source,
                                     std::uint32_t survivors,
                                     std::uint64_t const* prefixes,
                                     std::uint32_t batches,
                                     std::uint32_t const* marked,
                                     std::uint32_t const* count,
                                     std::uint64_t const* scratch_offsets,
                                     std::int64_t* destination,
                                     std::uint32_t* status)
{
  auto const j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j >= survivors) return;
  destination[j] = 0;
  if (lengths[j] == 0) return;  // includes zero-length rows at the final byte boundary
  auto const start = static_cast<std::uint64_t>(source[j]);
  auto const batch = first_byte_batch(prefixes, batches, start);
  std::uint32_t lo = 0, hi = *count;
  while (lo < hi) {
    auto const mid = lo + (hi - lo) / 2;
    if (marked[mid] < batch)
      lo = mid + 1;
    else
      hi = mid;
  }
  if (lo >= *count || marked[lo] != batch) {
    lengths[j] = 0;
    atomicExch(status, 1u);
    return;
  }
  auto const offset = scratch_offsets[lo] + start - prefixes[batch];
  if (offset > scratch_offsets[*count] ||
      static_cast<std::uint64_t>(lengths[j]) > scratch_offsets[*count] - offset) {
    lengths[j] = 0;
    atomicExch(status, 1u);
    return;
  }
  destination[j] = static_cast<std::int64_t>(offset);
}

struct selected_batch_totals {
  std::uint64_t bytes;
  std::int64_t output_bytes;
  std::uint32_t count;
  std::uint32_t status;
};

// Validate the sparse decoder's byte envelopes before the sole sizing/status observation.
// Codes, dictionary planes, and prefixes were validated at preparation and are immutable.
template <int K>
__device__ __forceinline__ void validate_selected_batches_impl(std::uint16_t const* codes,
                                                               std::uint8_t const* lengths,
                                                               std::uint64_t total_tokens,
                                                               std::uint64_t const* prefixes,
                                                               std::uint32_t const* marked,
                                                               std::uint32_t const* count,
                                                               std::uint64_t const* scratch_offsets,
                                                               std::uint32_t* status)
{
  auto const lane = threadIdx.x & 31;
  auto const i    = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
  if (i >= *count) return;
  auto const batch    = marked[i];
  auto const start    = static_cast<std::uint64_t>(batch) * (32 * K);
  std::uint32_t bytes = 0;
#pragma unroll
  for (int k = 0; k < K; ++k) {
    auto const token = start + lane + k * 32;
    if (token < total_tokens) bytes += onpair_detail::token_length<K>(lengths, codes[token]);
  }
#pragma unroll
  for (int delta = 16; delta > 0; delta /= 2)
    bytes += __shfl_down_sync(0xffffffffu, bytes, delta);
  if (lane == 0 && (bytes != prefixes[batch + 1] - prefixes[batch] ||
                    bytes != scratch_offsets[i + 1] - scratch_offsets[i] || bytes > 32u * K * 16))
    atomicExch(status, 1u);
}

__global__ void validate_selected_batches(std::uint16_t const* codes,
                                          std::uint8_t const* lengths,
                                          std::uint64_t total_tokens,
                                          std::uint64_t const* prefixes,
                                          std::uint32_t const* marked,
                                          std::uint32_t const* count,
                                          std::uint64_t const* scratch_offsets,
                                          std::uint32_t* status)
{
  validate_selected_batches_impl<4>(
    codes, lengths, total_tokens, prefixes, marked, count, scratch_offsets, status);
}

__global__ void validate_selected_batches_k6(std::uint16_t const* codes,
                                             std::uint8_t const* lengths,
                                             std::uint64_t total_tokens,
                                             std::uint64_t const* prefixes,
                                             std::uint32_t const* marked,
                                             std::uint32_t const* count,
                                             std::uint64_t const* scratch_offsets,
                                             std::uint32_t* status)
{
  validate_selected_batches_impl<6>(
    codes, lengths, total_tokens, prefixes, marked, count, scratch_offsets, status);
}

__global__ void collect_selected_totals(std::uint64_t const* offsets,
                                        std::int64_t const* output_offsets,
                                        std::uint32_t survivors,
                                        std::uint32_t const* count,
                                        std::uint32_t const* status,
                                        selected_batch_totals* totals)
{
  *totals = {offsets[*count], output_offsets[survivors], *count, *status};
}

__global__ void selected_char_copy64(std::uint8_t const* chars,
                                     std::int64_t const* source_offsets,
                                     std::int64_t const* output_offsets,
                                     std::int64_t survivors,
                                     std::uint8_t* output)
{
  for (std::int64_t j = blockIdx.x; j < survivors; j += gridDim.x) {
    auto const src = source_offsets[j];
    auto const dst = output_offsets[j];
    auto const len = output_offsets[j + 1] - dst;
    for (std::int64_t b = threadIdx.x; b < len; b += blockDim.x)
      output[dst + b] = chars[src + b];
  }
}

}  // namespace

void copy_onpair_selected_chars64(void const* chars,
                                  std::int64_t const* source_offsets,
                                  std::int64_t const* output_offsets,
                                  std::int64_t survivors,
                                  void* output,
                                  decode_frame& frame)
{
  if (survivors <= 0) return;
  selected_char_copy64<<<std::min<std::int64_t>(8192, survivors), 256, 0, frame.stream().get()>>>(
    static_cast<std::uint8_t const*>(chars),
    source_offsets,
    output_offsets,
    survivors,
    static_cast<std::uint8_t*>(output));
  throw_if_cuda_error(cudaGetLastError(), "onpair: copy selected large-string chars");
}

onpair_selected_batches onpair_compressed_representation::select_batches(cudf::column_view rows,
                                                                         cudf::column_view indices,
                                                                         decode_frame& frame) const
{
  if (rows.size() != static_cast<std::int64_t>(num_rows) + 1 || rows.null_count() != 0 ||
      (rows.type().id() != cudf::type_id::INT32 && rows.type().id() != cudf::type_id::INT64))
    throw std::invalid_argument("onpair: invalid selected row byte offsets");
  auto const stream    = frame.stream();
  auto const mr        = frame.scratch_mr();
  auto const survivors = static_cast<std::uint32_t>(indices.size());
  auto const batches   = batch_count();
  onpair_selected_batches selected;
  selected.source_offsets = rmm::device_buffer(survivors * sizeof(std::int64_t), stream, mr);
  // The public route handles zero survivors before reconstructing offsets or submitting work.
  if (survivors == 0) return selected;
  rmm::device_buffer length_storage((survivors + 1ULL) * sizeof(std::int64_t), stream, mr);
  auto* lengths           = static_cast<std::int64_t*>(length_storage.data());
  // A small source guarantees the wide scan will be cast to INT32 before delivery.
  auto const offsets_mr =
    metadata_.decoded_bytes <= std::numeric_limits<std::int32_t>::max() ? mr : frame.mr();
  selected.output_offsets = cudf::make_fixed_width_column(cudf::data_type{cudf::type_id::INT64},
                                                          survivors + 1,
                                                          cudf::mask_state::UNALLOCATED,
                                                          stream,
                                                          offsets_mr);
  rmm::device_buffer status(sizeof(std::uint32_t), stream, mr);
  auto* flag = static_cast<std::uint32_t*>(status.data());
  throw_if_cuda_error(cudaMemsetAsync(flag, 0, status.size(), stream.get()),
                      "onpair: initialize selected status");
  rmm::device_buffer source(survivors * sizeof(std::int64_t), stream, mr);
  auto meta = [&]<typename Offset>() {
    auto const count = static_cast<std::uint64_t>(num_rows) + 1;
    validate_rows<<<(count + 255) / 256, 256, 0, stream.get()>>>(
      rows.data<Offset>(), count, metadata_.decoded_bytes, flag);
    throw_if_cuda_error(cudaGetLastError(), "onpair: validate selected row byte offsets");
    selected_row_meta<<<(survivors + 255) / 256, 256, 0, stream.get()>>>(
      rows.data<Offset>(),
      indices.data<std::int32_t>(),
      survivors,
      num_rows,
      metadata_.decoded_bytes,
      lengths,
      static_cast<std::int64_t*>(source.data()),
      flag);
    throw_if_cuda_error(cudaGetLastError(), "onpair: selected survivor meta");
  };
  if (rows.type().id() == cudf::type_id::INT64)
    meta.template operator()<std::int64_t>();
  else
    meta.template operator()<std::int32_t>();
  throw_if_cuda_error(cudaMemsetAsync(lengths + survivors, 0, sizeof(std::int64_t), stream.get()),
                      "onpair: clear selected length tail");

  selected.marked = rmm::device_buffer(batches * sizeof(std::uint32_t), stream, mr);
  selected.scratch_offsets =
    rmm::device_buffer((batches + 1ULL) * sizeof(std::uint64_t), stream, mr);
  rmm::device_buffer count(sizeof(std::uint32_t), stream, mr);
  auto* count_data     = static_cast<std::uint32_t*>(count.data());
  auto* offsets        = static_cast<std::uint64_t*>(selected.scratch_offsets.data());
  auto const* prefixes = views_[4].data<std::uint64_t>();
  if (batches == 0) {
    throw_if_cuda_error(cudaMemsetAsync(count_data, 0, count.size(), stream.get()),
                        "onpair: empty marked count");
    throw_if_cuda_error(cudaMemsetAsync(offsets, 0, selected.scratch_offsets.size(), stream.get()),
                        "onpair: empty scratch offsets");
  } else {
    rmm::device_buffer flags((batches + 3ULL) / 4 * 4, stream, mr);
    throw_if_cuda_error(cudaMemsetAsync(flags.data(), 0, flags.size(), stream.get()),
                        "onpair: clear batch flags");
    mark_selected_batches<<<(survivors + 255) / 256, 256, 0, stream.get()>>>(
      lengths,
      static_cast<std::int64_t const*>(source.data()),
      survivors,
      prefixes,
      batches,
      static_cast<std::uint8_t*>(flags.data()),
      flag);
    throw_if_cuda_error(cudaGetLastError(), "onpair: mark selected batches");
    auto const ids = thrust::make_counting_iterator(std::uint32_t{0});
    auto select    = [&](void* temp, std::size_t& bytes) {
      return cub::DeviceSelect::Flagged(temp,
                                        bytes,
                                        ids,
                                        static_cast<std::uint8_t const*>(flags.data()),
                                        static_cast<std::uint32_t*>(selected.marked.data()),
                                        count_data,
                                        batches,
                                        stream.get());
    };
    std::size_t select_bytes = 0;
    throw_if_cuda_error(select(nullptr, select_bytes), "onpair: batch select scratch size");
    rmm::device_buffer select_temp(select_bytes, stream, mr);
    throw_if_cuda_error(select(select_temp.data(), select_bytes), "onpair: select marked batches");
    auto const sizes = thrust::make_transform_iterator(
      ids,
      marked_byte_length{
        static_cast<std::uint32_t const*>(selected.marked.data()), count_data, prefixes});
    auto scan = [&](void* temp, std::size_t& bytes) {
      // The zero tail through batches+1 permits one totals/status readback. Only
      // offsets[0..marked_count] are consumed; no D2H count is needed to size the scan.
      return cub::DeviceScan::ExclusiveSum(temp, bytes, sizes, offsets, batches + 1, stream.get());
    };
    std::size_t scan_bytes = 0;
    throw_if_cuda_error(scan(nullptr, scan_bytes), "onpair: scratch scan size");
    rmm::device_buffer scan_temp(scan_bytes, stream, mr);
    throw_if_cuda_error(scan(scan_temp.data(), scan_bytes), "onpair: scan marked batch bytes");
  }
  map_selected_sources<<<(survivors + 255) / 256, 256, 0, stream.get()>>>(
    lengths,
    static_cast<std::int64_t const*>(source.data()),
    survivors,
    prefixes,
    batches,
    static_cast<std::uint32_t const*>(selected.marked.data()),
    count_data,
    offsets,
    static_cast<std::int64_t*>(selected.source_offsets.data()),
    flag);
  throw_if_cuda_error(cudaGetLastError(), "onpair: map selected scratch sources");
  auto* output_offsets = selected.output_offsets->mutable_view().data<std::int64_t>();
  auto scan_output     = [&](void* temp, std::size_t& bytes) {
    return cub::DeviceScan::ExclusiveSum(
      temp, bytes, lengths, output_offsets, survivors + 1, stream.get());
  };
  std::size_t output_scan_bytes = 0;
  throw_if_cuda_error(scan_output(nullptr, output_scan_bytes), "onpair: output scan size");
  rmm::device_buffer output_scan_temp(output_scan_bytes, stream, mr);
  throw_if_cuda_error(scan_output(output_scan_temp.data(), output_scan_bytes),
                      "onpair: scan selected output offsets");
  if (batches != 0) {
    auto const& dictionary = prepared_ ? *prepared_ : *this;
    auto const kernel =
      tokens_per_batch() == 192 ? validate_selected_batches_k6 : validate_selected_batches;
    kernel<<<(batches + 15) / 16, 512, 0, stream.get()>>>(
      views_[0].data<std::uint16_t>(),
      static_cast<std::uint8_t const*>(dictionary.lengths_.data()),
      views_[0].size(),
      prefixes,
      static_cast<std::uint32_t const*>(selected.marked.data()),
      count_data,
      offsets,
      flag);
    throw_if_cuda_error(cudaGetLastError(), "onpair: validate selected batch envelopes");
  }
  rmm::device_buffer totals(sizeof(selected_batch_totals), stream, mr);
  collect_selected_totals<<<1, 1, 0, stream.get()>>>(
    offsets,
    output_offsets,
    survivors,
    count_data,
    flag,
    static_cast<selected_batch_totals*>(totals.data()));
  throw_if_cuda_error(cudaGetLastError(), "onpair: collect selected totals");
  auto const host = frame.read_scalar(static_cast<selected_batch_totals const*>(totals.data()));
  if (host.status != 0 || host.count > batches || host.bytes > metadata_.decoded_bytes ||
      host.output_bytes < 0 ||
      static_cast<std::uint64_t>(host.output_bytes) > metadata_.decoded_bytes)
    throw std::runtime_error("onpair: corrupt selected row offsets, indices, or batch envelope");
  selected.count         = host.count;
  selected.scratch_bytes = host.bytes;
  selected.output_bytes  = host.output_bytes;
  return selected;
}

void onpair_compressed_representation::decode_batches(onpair_selected_batches const& selected,
                                                      void* scratch,
                                                      decode_frame& frame) const
{
  if (selected.count == 0) return;
  auto const& dictionary = prepared_ ? *prepared_ : *this;
  using namespace onpair_detail;
  if (tokens_per_batch() == k6::tokens_per_warp) {
    k6::decode<true>
      <<<(selected.count + k6::warps_per_block - 1) / k6::warps_per_block,
         k6::block_threads,
         0,
         frame.stream().get()>>>(views_[0].data<std::uint16_t>(),
                                 static_cast<std::uint64_t const*>(selected.scratch_offsets.data()),
                                 static_cast<std::uint8_t const*>(dictionary.dict_s8_.data()),
                                 static_cast<std::uint8_t const*>(dictionary.dict_tail_.data()),
                                 static_cast<std::uint8_t const*>(dictionary.lengths_.data()),
                                 static_cast<std::uint8_t*>(scratch),
                                 views_[0].size(),
                                 static_cast<std::uint32_t const*>(selected.marked.data()),
                                 selected.count);
  } else {
    onpair_decode_marked_u16<<<(selected.count + 15) / 16, 512, 0, frame.stream().get()>>>(
      views_[0].data<std::uint16_t>(),
      static_cast<std::uint8_t const*>(dictionary.dict_s8_.data()),
      static_cast<std::uint8_t const*>(dictionary.dict_tail_.data()),
      static_cast<std::uint8_t const*>(dictionary.lengths_.data()),
      static_cast<std::uint32_t const*>(selected.marked.data()),
      static_cast<std::uint64_t const*>(selected.scratch_offsets.data()),
      selected.count,
      views_[0].size(),
      static_cast<std::uint8_t*>(scratch));
  }
  throw_if_cuda_error(cudaGetLastError(), "onpair: decode marked batches");
}

onpair_compressed_representation::onpair_compressed_representation(
  leaf_meta::onpair metadata,
  std::vector<std::unique_ptr<cudf::column>> channels,
  ::cuda::stream_ref stream,
  rmm::device_async_resource_ref mr,
  bool initialize_batch_prefixes)
  : metadata_(metadata)
{
  if (channels.size() != 5 ||
      std::any_of(channels.begin(), channels.end(), [](auto const& c) { return !c; }))
    throw std::invalid_argument("onpair: missing channels");
  for (std::size_t i = 0; i < 5; ++i)
    views_[i] = channels[i]->view();
  channels_ = std::move(channels);
  validate_and_stage(stream, mr, initialize_batch_prefixes);
}

onpair_compressed_representation::onpair_compressed_representation(
  leaf_meta::onpair metadata,
  std::array<cudf::column_view, 5> views,
  std::vector<std::unique_ptr<cudf::column>> owners,
  ::cuda::stream_ref stream,
  rmm::device_async_resource_ref mr,
  std::optional<rmm::device_async_resource_ref> scratch_mr)
  : metadata_(metadata), views_(std::move(views))
{
  if (owners.size() != 5) throw std::invalid_argument("onpair: missing channel owners");
  channels_ = std::move(owners);
  validate_and_stage(stream, mr, false, scratch_mr);
}

onpair_compressed_representation::onpair_compressed_representation(
  onpair_compressed_representation const& prepared, std::unique_ptr<cudf::column> rows)
  : metadata_(prepared.metadata_), views_(prepared.views_), prepared_(&prepared)
{
  if (!prepared.preparation_only_ || !rows || rows->size() != metadata_.num_rows + 1 ||
      (rows->type().id() != cudf::type_id::INT32 && rows->type().id() != cudf::type_id::INT64) ||
      rows->null_count() != 0)
    throw std::invalid_argument("onpair: invalid reconstructed row byte offsets");
  original_type = cudf::data_type{cudf::type_id::STRING};
  num_rows      = static_cast<cudf::size_type>(metadata_.num_rows);
  views_[1]     = rows->view();
  channels_.resize(5);
  channels_[1] = std::move(rows);
}

void onpair_compressed_representation::validate_and_stage(
  ::cuda::stream_ref stream,
  rmm::device_async_resource_ref mr,
  bool initialize_batch_prefixes,
  std::optional<rmm::device_async_resource_ref> scratch_mr)
{
  auto const& metadata = metadata_;
  if (metadata.original_type_id != static_cast<std::int32_t>(cudf::type_id::STRING) ||
      metadata.dict_bits < 9 || metadata.dict_bits > 16 || metadata.num_rows >= kColumnLimit ||
      (metadata.tokens_per_batch != 128 && metadata.tokens_per_batch != 192))
    throw std::invalid_argument("onpair: invalid metadata");
  auto const offset_type = views_[1].type().id();
  if (views_[0].type().id() != cudf::type_id::UINT16 ||
      (offset_type != cudf::type_id::INT32 && offset_type != cudf::type_id::INT64) ||
      views_[2].type().id() != cudf::type_id::UINT8 ||
      views_[3].type().id() != cudf::type_id::UINT32 ||
      views_[4].type().id() != cudf::type_id::UINT64 || views_[1].size() != metadata.num_rows + 1 ||
      views_[3].size() < 1 ||
      std::any_of(views_.begin(), views_.end(), [](auto const& c) { return c.null_count(); }))
    throw std::invalid_argument("onpair: invalid channel types, sizes, or validity");
  auto const tokens    = static_cast<std::uint64_t>(views_[0].size());
  auto const num_batches = batch_count();
  if (views_[4].size() != num_batches + 1)
    throw std::invalid_argument("onpair: invalid batch_prefixes size (byte offsets)");
  if (offset_type == cudf::type_id::INT32 &&
      metadata.decoded_bytes > std::numeric_limits<std::int32_t>::max())
    throw std::invalid_argument(
      "onpair: INT32 row byte offsets cannot represent decoded_bytes above INT32_MAX");
  auto const dict_size = static_cast<std::uint32_t>(views_[3].size() - 1);
  if (dict_size > (1u << metadata.dict_bits) || (tokens != 0 && dict_size == 0) ||
      metadata.decoded_bytes < tokens || metadata.decoded_bytes > tokens * 16 ||
      (metadata.num_rows == 0 && tokens != 0) || (dict_size == 0 && views_[2].size() != 0))
    throw std::invalid_argument("onpair: dictionary or decoded byte count is inconsistent");
  original_type = cudf::data_type{cudf::type_id::STRING};
  num_rows      = static_cast<cudf::size_type>(metadata.num_rows);
  rmm::device_buffer status(sizeof(std::uint32_t), stream, scratch_mr.value_or(mr));
  rmm::device_buffer sum(sizeof(unsigned long long), stream, scratch_mr.value_or(mr));
  throw_if_cuda_error(cudaMemsetAsync(status.data(), 0, status.size(), stream.get()),
                      "onpair: initialize validation status");
  throw_if_cuda_error(cudaMemsetAsync(sum.data(), 0, sum.size(), stream.get()),
                      "onpair: initialize validation sum");
  auto* flag = static_cast<std::uint32_t*>(status.data());
  if (dict_size != 0) {
    bool const k6      = tokens_per_batch() == 192;
    dict_s8_           = rmm::device_buffer(dict_size * 8u, stream, mr);
    dict_tail_         = rmm::device_buffer(dict_size * (k6 ? 8u : 16u), stream, mr);
    lengths_           = rmm::device_buffer(k6 ? (dict_size + 1) / 2 : dict_size, stream, mr);
    auto const stage   = k6 ? stage_dictionary_k6 : stage_dictionary;
    auto const threads = k6 ? (dict_size + 1) / 2 : dict_size;
    stage<<<(threads + 255) / 256, 256, 0, stream.get()>>>(
      views_[2].data<std::uint8_t>(),
      views_[3].data<std::uint32_t>(),
      dict_size,
      static_cast<std::uint32_t>(views_[2].size()),
      static_cast<std::uint8_t*>(dict_s8_.data()),
      static_cast<std::uint8_t*>(dict_tail_.data()),
      static_cast<std::uint8_t*>(lengths_.data()),
      flag);
    throw_if_cuda_error(cudaGetLastError(), "onpair: stage dictionary");
  }
  if (tokens != 0) {
    auto const kernel = tokens_per_batch() == 192 ? validate_codes_k6 : validate_codes;
    kernel<<<(tokens + 255) / 256, 256, 0, stream.get()>>>(
      views_[0].data<std::uint16_t>(),
      tokens,
      static_cast<std::uint8_t const*>(lengths_.data()),
      dict_size,
      flag,
      static_cast<unsigned long long*>(sum.data()));
    throw_if_cuda_error(cudaGetLastError(), "onpair: validate codes");
  }
  auto const count = metadata.num_rows + 1;
  if (offset_type == cudf::type_id::INT64)
    validate_rows<<<(count + 255) / 256, 256, 0, stream.get()>>>(
      views_[1].data<std::int64_t>(), count, metadata.decoded_bytes, flag);
  else
    validate_rows<<<(count + 255) / 256, 256, 0, stream.get()>>>(
      views_[1].data<std::int32_t>(), count, metadata.decoded_bytes, flag);
  throw_if_cuda_error(cudaGetLastError(), "onpair: validate row byte offsets");
  // Pin time only: stage/validate codes before the sweep, which also safely
  // flags out-of-dictionary codes without gathering their lengths.
  using namespace onpair_detail;
  rmm::device_buffer computed(
    (num_batches + 1) * sizeof(std::uint64_t), stream, scratch_mr.value_or(mr));
  if (tokens == 0) {
    throw_if_cuda_error(cudaMemsetAsync(computed.data(), 0, computed.size(), stream.get()),
                        "onpair: empty batch byte prefixes");
  } else {
    std::size_t temp_bytes = 0;
    throw_if_cuda_error(onpair_batch_offsets_temp_size(&temp_bytes, num_batches),
                        "onpair: batch prefixes scratch size");
    rmm::device_buffer temp(temp_bytes, stream, scratch_mr.value_or(mr));
    throw_if_cuda_error(onpair_batch_offsets(temp.data(),
                                             temp_bytes,
                                             views_[0].data<std::uint16_t>(),
                                             2,
                                             static_cast<std::uint8_t const*>(lengths_.data()),
                                             dict_size,
                                             tokens,
                                             static_cast<std::uint64_t*>(computed.data()),
                                             flag,
                                             num_batches,
                                             tokens_per_batch(),
                                             stream.get()),
                        "onpair: compute batch byte prefixes");
  }
  if (initialize_batch_prefixes)
    throw_if_cuda_error(cudaMemcpyAsync(channels_[4]->mutable_view().data<std::uint64_t>(),
                                        computed.data(),
                                        computed.size(),
                                        cudaMemcpyDeviceToDevice,
                                        stream.get()),
                        "onpair: store batch byte prefixes");
  validate_batch_prefixes<<<(num_batches + 1 + 255) / 256, 256, 0, stream.get()>>>(
    views_[4].data<std::uint64_t>(),
    static_cast<std::uint64_t const*>(computed.data()),
    num_batches + 1,
    flag);
  throw_if_cuda_error(cudaGetLastError(), "onpair: validate batch byte prefixes");
  std::uint32_t host_status;
  unsigned long long host_sum;
  // Host observations belong to construction, never to decompress().
  std::uint32_t first = 0;
  std::array<device_read, 3> reads{{{&host_status, status.data(), sizeof(host_status)},
                                    {&host_sum, sum.data(), sizeof(host_sum)},
                                    {&first, views_[3].data<std::uint32_t>(), sizeof(first)}}};
  read_device_bytes_completed(reads, stream);
  if (first != 0 || host_status != 0 || host_sum != metadata.decoded_bytes)
    throw std::invalid_argument(
      "onpair: corrupt dictionary, codes, row byte offsets, batch byte prefixes, or decoded size "
      "(dict_offsets[0]=" +
      std::to_string(first) + ", status=" + std::to_string(host_status) + ", token bytes=" +
      std::to_string(host_sum) + ", decoded_bytes=" + std::to_string(metadata.decoded_bytes) +
      ", tokens=" + std::to_string(tokens) + ", dict_size=" + std::to_string(dict_size) +
      ", rows=" + std::to_string(metadata.num_rows) + ")");
}

void onpair_compressed_representation::finish_preparation(PlanNode const& node)
{
  if (node.output_names != op_info(OpId::OnPair).channels)
    throw std::invalid_argument("onpair: invalid output channels during preparation");
  for (std::size_t i : {0u, 2u, 3u, 4u}) {
    auto const* terminal = terminal_identity_channel(node, op_info(OpId::OnPair).channels[i]);
    if (terminal) {
      views_[i] = terminal->view();
      channels_[i].reset();
    }
  }
  channels_[1].reset();
  views_[1]         = cudf::column_view{};
  preparation_only_ = true;
}

std::vector<compressible_output> onpair_compressed_representation::named_channels(
  ::cuda::stream_ref) const
{
  if (preparation_only_) return {};
  std::vector<compressible_output> result;
  auto const& names = op_info(OpId::OnPair).channels;
  for (std::size_t i = 0; i < 5; ++i)
    result.push_back({names[i], views_[i]});
  return result;
}

size_t onpair_compressed_representation::resident_size_bytes(::cuda::stream_ref stream) const
{
  return compressed_representation::resident_size_bytes(stream) + dict_s8_.size() +
         dict_tail_.size() + lengths_.size();
}

std::unique_ptr<cudf::column> onpair_compressed_representation::decompress(
  decode_frame& frame) const
{
  if (preparation_only_)
    throw std::logic_error("onpair: prepared plan state requires reconstructed row byte offsets");
  auto const stream       = frame.stream();
  auto const mr           = frame.mr();
  auto const total_tokens = static_cast<std::uint64_t>(views_[0].size());
  auto const& dictionary  = prepared_ ? *prepared_ : *this;
  std::uint32_t const* row_status = nullptr;
  // A reconstructed row_offsets channel contains byte offsets and has not had
  // construction-time validation. Defer its status to session completion.
  if (prepared_ || total_tokens == 0) {
    rmm::device_buffer status(sizeof(std::uint32_t), stream, frame.scratch_mr());
    throw_if_cuda_error(cudaMemsetAsync(status.data(), 0, status.size(), stream.get()),
                        "onpair: initialize row byte offsets validation");
    auto const count = static_cast<std::uint64_t>(num_rows) + 1;
    auto* flag       = static_cast<std::uint32_t*>(status.data());
    if (views_[1].type().id() == cudf::type_id::INT64)
      validate_rows<<<(count + 255) / 256, 256, 0, stream.get()>>>(
        views_[1].data<std::int64_t>(), count, metadata_.decoded_bytes, flag);
    else
      validate_rows<<<(count + 255) / 256, 256, 0, stream.get()>>>(
        views_[1].data<std::int32_t>(), count, metadata_.decoded_bytes, flag);
    throw_if_cuda_error(cudaGetLastError(), "onpair: validate reconstructed row byte offsets");
    row_status = flag;
    frame.check_device_status(std::move(status), "onpair: corrupt reconstructed row byte offsets");
  }
  if (num_rows == 0) return cudf::make_empty_column(cudf::data_type{cudf::type_id::STRING});
  // Copying keeps decompress const and repeatable for both stored and rebuilt reps.
  std::unique_ptr<cudf::column> offsets;
  if (row_status) {
    offsets = cudf::make_fixed_width_column(
      views_[1].type(), views_[1].size(), cudf::mask_state::UNALLOCATED, stream, mr);
    auto const count = static_cast<std::uint64_t>(num_rows) + 1;
    // A separate launch on the validation stream observes the completed status.
    // Same-request consumers must see safe spans before session completion throws.
    if (views_[1].type().id() == cudf::type_id::INT64)
      copy_validated_offsets<<<(count + 255) / 256, 256, 0, stream.get()>>>(
        views_[1].data<std::int64_t>(),
        offsets->mutable_view().data<std::int64_t>(),
        count,
        row_status);
    else
      copy_validated_offsets<<<(count + 255) / 256, 256, 0, stream.get()>>>(
        views_[1].data<std::int32_t>(),
        offsets->mutable_view().data<std::int32_t>(),
        count,
        row_status);
    throw_if_cuda_error(cudaGetLastError(), "onpair: copy validated row byte offsets");
  } else {
    offsets = std::make_unique<cudf::column>(views_[1], stream, mr);
  }
  rmm::device_buffer chars(metadata_.decoded_bytes, stream, mr);
  if (total_tokens == 0) {
    throw_if_cuda_error(
      cudaMemsetAsync(offsets->mutable_view().head<void>(),
                      0,
                      (static_cast<std::size_t>(num_rows) + 1) * cudf::size_of(views_[1].type()),
                      stream.get()),
      "onpair: empty row byte offsets");
  } else {
    using namespace onpair_detail;
    auto const num_batches = batch_count();
    if (tokens_per_batch() == k6::tokens_per_warp) {
      k6::decode<false>
        <<<(num_batches + k6::warps_per_block - 1) / k6::warps_per_block,
           k6::block_threads,
           0,
           stream.get()>>>(views_[0].data<std::uint16_t>(),
                           views_[4].data<std::uint64_t>(),
                           static_cast<std::uint8_t const*>(dictionary.dict_s8_.data()),
                           static_cast<std::uint8_t const*>(dictionary.dict_tail_.data()),
                           static_cast<std::uint8_t const*>(dictionary.lengths_.data()),
                           static_cast<std::uint8_t*>(chars.data()),
                           total_tokens,
                           nullptr,
                           0);
    } else {
      onpair_shmem_4tpt_split8read_u16<<<(num_batches + 15) / 16, 512, 0, stream.get()>>>(
        views_[0].data<std::uint16_t>(),
        views_[4].data<std::uint64_t>(),
        static_cast<std::uint8_t const*>(dictionary.dict_s8_.data()),
        static_cast<std::uint8_t const*>(dictionary.dict_tail_.data()),
        static_cast<std::uint8_t const*>(dictionary.lengths_.data()),
        static_cast<std::uint8_t*>(chars.data()),
        0,
        total_tokens,
        0,
        0);
    }
    throw_if_cuda_error(cudaGetLastError(), "onpair: decode");
  }
  return cudf::make_strings_column(
    num_rows,
    std::move(offsets),
    std::move(chars),
    0,
    cudf::create_null_mask(0, cudf::mask_state::UNALLOCATED, stream, mr));
}

std::unique_ptr<compressed_representation> onpair_compressor::compress(
  cudf::column_view column, ::cuda::stream_ref stream, rmm::device_async_resource_ref mr)
{
  if (column.type().id() != cudf::type_id::STRING)
    throw std::invalid_argument("onpair: column must be STRING");
  if (column.size() == kColumnLimit)
    throw std::length_error("onpair: row byte offsets exceed cuDF's element-count limit");
  leaf_meta::onpair metadata{0,
                             0,
                             dict_bits_,
                             static_cast<std::int32_t>(column.type().id()),
                             static_cast<std::uint16_t>(32 * onpair_compression_k())};
  if (column.size() == 0) {
    // A childless empty STRING has no offsets child. Do not train/store a
    // dictionary: the canonical form is {[], [0], [], [0], [0]}.
    std::vector<std::unique_ptr<cudf::column>> channels;
    auto zero = [&](cudf::type_id type, cudf::size_type size) {
      auto col = cudf::make_fixed_width_column(
        cudf::data_type{type}, size, cudf::mask_state::UNALLOCATED, stream, mr);
      if (size != 0)
        throw_if_cuda_error(
          cudaMemsetAsync(
            col->mutable_view().head<void>(), 0, size * cudf::size_of(col->type()), stream.get()),
          "onpair: initialize empty channel");
      return col;
    };
    channels.push_back(zero(cudf::type_id::UINT16, 0));
    channels.push_back(zero(cudf::type_id::INT32, 1));
    channels.push_back(zero(cudf::type_id::UINT8, 0));
    channels.push_back(zero(cudf::type_id::UINT32, 1));
    channels.push_back(zero(cudf::type_id::UINT64, 1));
    auto rep =
      std::make_unique<onpair_compressed_representation>(metadata, std::move(channels), stream, mr);
    throw_if_cuda_error(cudaStreamSynchronize(stream.get()), "onpair: complete empty channels");
    return rep;
  }
  bool const input_wide =
    cudf::strings_column_view(column).offsets().type().id() == cudf::type_id::INT64;
  std::unique_ptr<cudf::column> owned;
  if (column.offset() != 0 ||
      cudf::strings_column_view(column).offsets().size() != column.size() + 1) {
    owned  = copy_column_view(column, stream, mr);
    column = owned->view();
  }
  cudf::strings_column_view scv(column);
  auto const rows        = static_cast<std::uint64_t>(column.size());
  auto const offset_type = scv.offsets().type().id();
  if (offset_type != cudf::type_id::INT32 && offset_type != cudf::type_id::INT64)
    throw std::invalid_argument("onpair: string offsets must be INT32 or INT64");
  auto const chars_bytes = static_cast<std::uint64_t>(scv.chars_size(stream));
  std::vector<std::uint8_t> host_chars(chars_bytes);
  std::vector<std::int32_t> offsets32(offset_type == cudf::type_id::INT32 ? rows + 1 : 0);
  std::vector<std::int64_t> offsets64(offset_type == cudf::type_id::INT64 ? rows + 1 : 0);
  try {
    if (chars_bytes != 0) {
      throw_if_cuda_error(cudaMemcpyAsync(host_chars.data(),
                                          scv.chars_begin(stream),
                                          chars_bytes,
                                          cudaMemcpyDeviceToHost,
                                          stream.get()),
                          "onpair: download chars");
    }
    auto* host_offsets = offset_type == cudf::type_id::INT64 ? static_cast<void*>(offsets64.data())
                                                             : static_cast<void*>(offsets32.data());
    throw_if_cuda_error(cudaMemcpyAsync(host_offsets,
                                        scv.offsets().head<void>(),
                                        (rows + 1) * (offset_type == cudf::type_id::INT64 ? 8 : 4),
                                        cudaMemcpyDeviceToHost,
                                        stream.get()),
                        "onpair: download offsets");
    throw_if_cuda_error(cudaStreamSynchronize(stream.get()), "onpair: complete download");
  } catch (...) {
    // A preceding download may still reference these host vectors.
    (void)cudaStreamSynchronize(stream.get());
    throw;
  }

  // cuDF's gather can narrow the offsets while normalizing a small slice of
  // a large-string column. Preserve the original input's width for encode64.
  auto const use64 = input_wide || offset_type == cudf::type_id::INT64;
  if (use64 && offsets64.empty()) offsets64.assign(offsets32.begin(), offsets32.end());
  auto const decoded_bytes = static_cast<std::uint64_t>(
    use64 ? offsets64.back() - offsets64.front() : offsets32.back() - offsets32.front());

  auto config               = onpair_bridge_config_default();  // default host parallelism
  config.max_dict_bits      = dict_bits_;
  onpair_bridge_column* raw = nullptr;
  auto const status =
    use64
      ? onpair_bridge_encode64(host_chars.data(), chars_bytes, offsets64.data(), rows, config, &raw)
      : onpair_bridge_encode32(
          host_chars.data(), chars_bytes, offsets32.data(), rows, config, &raw);
  std::unique_ptr<onpair_bridge_column, decltype(&onpair_bridge_column_free)> encoded(
    raw, &onpair_bridge_column_free);
  if (status != ONPAIR_BRIDGE_OK || !encoded)
    throw std::runtime_error("onpair: bridge encode failed (status " + std::to_string(status) +
                             ")");
  // The bridge consumes absolute boundaries against the original chars base.
  // Only then rebase the stored byte offsets to the decoded chars allocation.
  auto rebase = [](auto& offsets) {
    auto const origin = offsets.front();
    for (auto& offset : offsets)
      offset -= origin;
  };
  if (use64)
    rebase(offsets64);
  else
    rebase(offsets32);
  auto const& e = *encoded;
  std::vector<std::unique_ptr<cudf::column>> channels;
  channels.reserve(5);
  try {
    channels.push_back(upload(cudf::type_id::UINT16, e.codes, e.codes_len, stream, mr));
    // row_offsets stores the input's normalized BYTE offsets, preserving its width.
    // The bridge's e.row_offsets still contains token offsets; it is not uploaded.
    channels.push_back(upload(use64 ? cudf::type_id::INT64 : cudf::type_id::INT32,
                              use64 ? static_cast<void const*>(offsets64.data())
                                    : static_cast<void const*>(offsets32.data()),
                              rows + 1,
                              stream,
                              mr));
    channels.push_back(upload(cudf::type_id::UINT8, e.dict_bytes, e.dict_bytes_len, stream, mr));
    channels.push_back(
      upload(cudf::type_id::UINT32, e.dict_offsets, e.dict_offsets_len, stream, mr));
    channels.push_back(
      cudf::make_fixed_width_column(cudf::data_type{cudf::type_id::UINT64},
                                    column_size(metadata.batch_count(e.codes_len) + 1),
                                    cudf::mask_state::UNALLOCATED,
                                    stream,
                                    mr));
    metadata.num_rows      = rows;
    metadata.decoded_bytes = decoded_bytes;
    auto rep               = std::make_unique<onpair_compressed_representation>(
      metadata, std::move(channels), stream, mr, true);
    // Host bridge buffers must remain alive until all uploads have completed.
    throw_if_cuda_error(cudaStreamSynchronize(stream.get()), "onpair: complete upload");
    return rep;
  } catch (...) {
    (void)cudaStreamSynchronize(stream.get());
    throw;
  }
}

}  // namespace simpatico
