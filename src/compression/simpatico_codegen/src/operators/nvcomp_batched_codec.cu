// SPDX-License-Identifier: Apache-2.0
// Implementation of the low-level (batched) nvcomp codec driver. See the header
// for the frame format and rationale.

#include "codegen/util/cuda_check.hpp"
#include "decode/decode_session.hpp"
#include "nvcomp_batched_codec.hpp"
#include "util/host_observation.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cstring>
#include <new>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace simpatico {
namespace detail {

namespace {

inline std::size_t align_up(std::size_t x, std::size_t a) { return (x + a - 1) / a * a; }

// nvcomp-status check stays local; CUDA-error checks use the shared
// throw_if_cuda_error (cuda_check.hpp).
inline void check(nvcompStatus_t s, char const* what)
{
  if (s != nvcompSuccess) {
    throw std::runtime_error(std::string("nvcomp batched: ") + what + " failed (status " +
                             std::to_string(static_cast<int>(s)) + ")");
  }
}

// One block per chunk; the block's threads cooperatively copy the chunk's
// compressed bytes from the padded scratch layout into the compact frame.
__global__ void scatter_chunks_kernel(std::uint8_t const* __restrict__ src,
                                      std::size_t src_stride,
                                      std::uint8_t* __restrict__ dst,
                                      std::size_t const* __restrict__ dst_offsets,
                                      std::size_t const* __restrict__ sizes,
                                      std::size_t num_chunks)
{
  std::size_t const chunk = blockIdx.x;
  if (chunk >= num_chunks) return;
  std::uint8_t const* s = src + chunk * src_stride;
  std::uint8_t* d       = dst + dst_offsets[chunk];
  std::size_t const n   = sizes[chunk];
  for (std::size_t i = threadIdx.x; i < n; i += blockDim.x)
    d[i] = s[i];
}

}  // namespace

std::pair<std::unique_ptr<rmm::device_buffer>, nvcomp_frame_layout> batched_compress_bytes(
  batched_codec_ops const& ops,
  void const* src,
  std::size_t n_bytes,
  ::cuda::stream_ref stream,
  rmm::device_async_resource_ref mr)
{
  cudaStream_t const s = stream.get();
  if (n_bytes == 0)
    return {std::make_unique<rmm::device_buffer>(0, stream, mr), nvcomp_frame_layout{}};

  std::size_t const chunk      = ops.chunk_size;
  std::size_t const num_chunks = (n_bytes + chunk - 1) / chunk;

  // Uncompressed chunk pointers/sizes into the source (host-built, uploaded).
  std::vector<void const*> h_uncomp_ptrs(num_chunks);
  std::vector<std::size_t> h_uncomp_bytes(num_chunks);
  auto const* base = static_cast<std::uint8_t const*>(src);
  for (std::size_t i = 0; i < num_chunks; ++i) {
    h_uncomp_ptrs[i]  = base + i * chunk;
    h_uncomp_bytes[i] = std::min(chunk, n_bytes - i * chunk);
  }

  std::size_t max_out = 0;
  check(ops.compress_get_max_output(chunk, &max_out), "compress_get_max_output");
  std::size_t const out_stride = align_up(max_out, kFrameAlign);

  // Per-chunk output slots (worst-case sized, padded for alignment).
  rmm::device_buffer scratch(out_stride * num_chunks, stream, mr);
  std::vector<void*> h_comp_ptrs(num_chunks);
  for (std::size_t i = 0; i < num_chunks; ++i)
    h_comp_ptrs[i] = static_cast<std::uint8_t*>(scratch.data()) + i * out_stride;

  std::size_t temp_bytes = 0;
  check(ops.compress_get_temp_size(num_chunks, chunk, &temp_bytes, n_bytes, s),
        "compress_get_temp_size");
  rmm::device_buffer temp(temp_bytes, stream, mr);

  // Coalesce the per-chunk metadata arrays into ONE device allocation instead of six
  // tiny ones (uncomp ptrs/bytes, comp ptrs/bytes, dst offsets, statuses). nvcomp only
  // needs each array base to be 8-byte aligned, which sub-regions at 8*num_chunks
  // strides satisfy; this cuts the per-op device-allocation count from ~6 to 1.
  std::size_t const sz_ptr           = num_chunks * sizeof(void*);
  std::size_t const sz_size          = num_chunks * sizeof(std::size_t);
  std::size_t const sz_stat          = num_chunks * sizeof(nvcompStatus_t);
  std::size_t const off_uncomp_ptrs  = 0;
  std::size_t const off_uncomp_bytes = off_uncomp_ptrs + sz_ptr;
  std::size_t const off_comp_ptrs    = off_uncomp_bytes + sz_size;
  std::size_t const off_comp_bytes   = off_comp_ptrs + sz_ptr;
  std::size_t const off_dst_off      = off_comp_bytes + sz_size;
  std::size_t const off_statuses     = off_dst_off + sz_size;
  rmm::device_buffer meta(off_statuses + sz_stat, stream, mr);
  auto* const meta_base      = static_cast<std::uint8_t*>(meta.data());
  void* const d_uncomp_ptrs  = meta_base + off_uncomp_ptrs;
  void* const d_uncomp_bytes = meta_base + off_uncomp_bytes;
  void* const d_comp_ptrs    = meta_base + off_comp_ptrs;
  void* const d_comp_bytes   = meta_base + off_comp_bytes;
  void* const d_dst_off      = meta_base + off_dst_off;  // filled after the size readback
  void* const d_statuses     = meta_base + off_statuses;
  throw_if_cuda_error(
    cudaMemcpyAsync(
      d_uncomp_ptrs, h_uncomp_ptrs.data(), num_chunks * sizeof(void*), cudaMemcpyHostToDevice, s),
    "H2D uncomp ptrs");
  throw_if_cuda_error(cudaMemcpyAsync(d_uncomp_bytes,
                                      h_uncomp_bytes.data(),
                                      num_chunks * sizeof(std::size_t),
                                      cudaMemcpyHostToDevice,
                                      s),
                      "H2D uncomp bytes");
  throw_if_cuda_error(
    cudaMemcpyAsync(
      d_comp_ptrs, h_comp_ptrs.data(), num_chunks * sizeof(void*), cudaMemcpyHostToDevice, s),
    "H2D comp ptrs");

  check(ops.compress_async(static_cast<void const* const*>(d_uncomp_ptrs),
                           static_cast<std::size_t const*>(d_uncomp_bytes),
                           chunk,
                           num_chunks,
                           temp.data(),
                           temp_bytes,
                           static_cast<void* const*>(d_comp_ptrs),
                           static_cast<std::size_t*>(d_comp_bytes),
                           static_cast<nvcompStatus_t*>(d_statuses),
                           s),
        "compress_async");

  std::vector<std::size_t> h_comp_bytes(num_chunks);
  throw_if_cuda_error(cudaMemcpyAsync(h_comp_bytes.data(),
                                      d_comp_bytes,
                                      num_chunks * sizeof(std::size_t),
                                      cudaMemcpyDeviceToHost,
                                      s),
                      "D2H comp bytes");
  throw_if_cuda_error(cudaStreamSynchronize(s), "sync after compress");

  // Lay out the compact frame: header + per-chunk sizes, then 256-aligned chunks.
  std::size_t const header = align_up(16 + num_chunks * sizeof(std::size_t), kFrameAlign);
  std::vector<std::size_t> h_dst_off(num_chunks);
  std::size_t cursor = header;
  for (std::size_t i = 0; i < num_chunks; ++i) {
    if (h_comp_bytes[i] == 0 || h_comp_bytes[i] > max_out)
      throw std::runtime_error("nvcomp batched: invalid compressed chunk size");
    h_dst_off[i] = cursor;
    cursor       = align_up(cursor + h_comp_bytes[i], kFrameAlign);
  }
  std::size_t const frame_size = h_dst_off[num_chunks - 1] + h_comp_bytes[num_chunks - 1];

  auto frame = std::make_unique<rmm::device_buffer>(frame_size, stream, mr);

  // Header + sizes assembled on host, one H2D.
  std::vector<std::uint8_t> h_hdr(header, 0);
  std::uint64_t const nc = num_chunks;
  std::uint64_t const cs = chunk;
  std::memcpy(h_hdr.data(), &nc, sizeof(nc));
  std::memcpy(h_hdr.data() + 8, &cs, sizeof(cs));
  std::memcpy(h_hdr.data() + 16, h_comp_bytes.data(), num_chunks * sizeof(std::size_t));
  throw_if_cuda_error(
    cudaMemcpyAsync(frame->data(), h_hdr.data(), header, cudaMemcpyHostToDevice, s),
    "H2D frame header");

  throw_if_cuda_error(
    cudaMemcpyAsync(
      d_dst_off, h_dst_off.data(), num_chunks * sizeof(std::size_t), cudaMemcpyHostToDevice, s),
    "H2D dst offsets");

  scatter_chunks_kernel<<<static_cast<unsigned>(num_chunks), 256, 0, s>>>(
    static_cast<std::uint8_t const*>(scratch.data()),
    out_stride,
    static_cast<std::uint8_t*>(frame->data()),
    static_cast<std::size_t const*>(d_dst_off),
    static_cast<std::size_t const*>(d_comp_bytes),
    num_chunks);
  throw_if_cuda_error(cudaGetLastError(), "scatter kernel launch");
  throw_if_cuda_error(cudaStreamSynchronize(s), "sync after scatter");

  return {std::move(frame), nvcomp_frame_layout{chunk, std::move(h_comp_bytes), frame_size}};
}

template <typename Read>
static std::array<std::uint64_t, 2> read_frame_header(void const* frame,
                                                      std::size_t frame_size,
                                                      std::size_t out_bytes,
                                                      Read read)
{
  if (frame_size < 16) throw std::runtime_error("nvcomp batched: truncated frame header");
  std::array<std::uint64_t, 2> hdr{};
  read(hdr.data(), frame, sizeof(hdr));
  auto const [blocks, chunk] = hdr;
  if (blocks == 0 || chunk == 0 || blocks != 1 + (out_bytes - 1) / chunk ||
      blocks > (frame_size - 16) / sizeof(std::size_t))
    throw std::runtime_error("nvcomp batched: corrupt frame header");
  return hdr;
}

std::size_t read_batched_frame_chunk_size(void const* frame,
                                          std::size_t frame_size,
                                          std::size_t out_bytes,
                                          ::cuda::stream_ref stream)
{
  if (out_bytes == 0) return 0;
  return read_frame_header(
    frame, frame_size, out_bytes, [&](void* dst, void const* src, std::size_t bytes) {
      read_device_bytes_completed(dst, src, bytes, stream);
    })[1];
}

template <typename Read>
static nvcomp_frame_layout parse_frame_layout(void const* frame,
                                              std::size_t frame_size,
                                              std::size_t out_bytes,
                                              Read read)
{
  nvcomp_frame_layout layout;
  if (out_bytes == 0) return layout;
  auto const* fbase = static_cast<std::uint8_t const*>(frame);
  auto const [num_chunks, chunk] = read_frame_header(frame, frame_size, out_bytes, read);
  layout.frame_bytes = frame_size;
  layout.chunk_bytes = chunk;
  layout.compressed_bytes.resize(num_chunks);
  read(layout.compressed_bytes.data(), fbase + 16, num_chunks * sizeof(std::size_t));
  std::size_t cursor = align_up(16 + num_chunks * sizeof(std::size_t), kFrameAlign);
  for (auto bytes : layout.compressed_bytes) {
    if (cursor > frame_size || bytes == 0 || bytes > frame_size - cursor)
      throw std::runtime_error("nvcomp batched: corrupt compressed chunk sizes");
    cursor = align_up(cursor + bytes, kFrameAlign);
  }
  return layout;
}

nvcomp_frame_layout read_host_batched_frame_layout(void const* frame,
                                                   std::size_t frame_size,
                                                   std::size_t out_bytes)
{ return parse_frame_layout(frame, frame_size, out_bytes, std::memcpy); }

nvcomp_frame_layout read_batched_frame_layout(void const* frame,
                                              std::size_t frame_size,
                                              std::size_t out_bytes,
                                              decode_frame& owner)
{
  return parse_frame_layout(
    frame, frame_size, out_bytes, [&](void* dst, void const* src, size_t n) {
      owner.read_bytes(dst, src, n);
    });
}

namespace {

// Upload one batched decompression's per-chunk tables and submit it. `fill(i, comp_ptr,
// comp_bytes, uncomp_bytes, uncomp_ptr)` describes entry i, and is called for i = 0..count-1 in
// order. `chunk` bounds every entry's uncompressed bytes and `total_uncomp` is their sum.
template <typename Fill>
std::size_t submit_batched_decompress(batched_codec_ops const& ops,
                                      std::size_t count,
                                      std::size_t chunk,
                                      std::size_t total_uncomp,
                                      decode_frame& owner,
                                      std::span<std::uint32_t const> selected_ids,
                                      Fill fill)
{
  cudaStream_t const s = owner.stream().get();
  // One device allocation holds the per-chunk tables (comp ptrs/bytes, uncomp bytes/ptrs) plus the
  // actual-size and status outputs some codecs require. Sub-regions at 8*count strides are 8-byte
  // aligned, which is all nvcomp needs. The four input tables are assembled in one frame-owned
  // host block laid out like the device block and uploaded with one copy.
  std::size_t const sz_ptr           = count * sizeof(void*);
  std::size_t const sz_size          = count * sizeof(std::size_t);
  std::size_t const sz_stat          = count * sizeof(nvcompStatus_t);
  std::size_t const off_comp_ptrs    = 0;
  std::size_t const off_comp_bytes   = off_comp_ptrs + sz_ptr;
  std::size_t const off_uncomp_bytes = off_comp_bytes + sz_size;
  std::size_t const off_uncomp_ptrs  = off_uncomp_bytes + sz_size;
  std::size_t const off_actual       = off_uncomp_ptrs + sz_ptr;
  std::size_t const off_statuses     = off_actual + sz_size;
  // Pointer and size tables share one 8-byte stride, so every sub-region stays aligned.
  static_assert(sizeof(void*) == sizeof(std::size_t) && alignof(void*) <= sizeof(std::size_t));
  auto const host = owner.host_array<std::byte>(off_actual);
  // The byte allocation implicitly creates the table elements; launder reaches them.
  auto* const h_comp_ptrs =
    std::launder(reinterpret_cast<void const**>(host.data() + off_comp_ptrs));
  auto* const h_comp_bytes =
    std::launder(reinterpret_cast<std::size_t*>(host.data() + off_comp_bytes));
  auto* const h_uncomp_bytes =
    std::launder(reinterpret_cast<std::size_t*>(host.data() + off_uncomp_bytes));
  auto* const h_uncomp_ptrs = std::launder(reinterpret_cast<void**>(host.data() + off_uncomp_ptrs));
  for (std::size_t i = 0; i < count; ++i)
    fill(i, h_comp_ptrs[i], h_comp_bytes[i], h_uncomp_bytes[i], h_uncomp_ptrs[i]);

  rmm::device_buffer meta(off_statuses + sz_stat, owner.stream(), owner.scratch_mr());
  auto* const meta_base = static_cast<std::uint8_t*>(meta.data());
  throw_if_cuda_error(
    cudaMemcpyAsync(meta_base, host.data(), host.size(), cudaMemcpyHostToDevice, s),
    "H2D chunk tables");
  void* const d_comp_ptrs    = meta_base + off_comp_ptrs;
  void* const d_comp_bytes   = meta_base + off_comp_bytes;
  void* const d_uncomp_bytes = meta_base + off_uncomp_bytes;
  void* const d_uncomp_ptrs  = meta_base + off_uncomp_ptrs;
  void* const d_actual   = meta_base + off_actual;
  void* const d_statuses = meta_base + off_statuses;

  std::size_t temp_bytes = 0;
  check(ops.decompress_get_temp_size(count, chunk, &temp_bytes, total_uncomp, s),
        "decompress_get_temp_size");
  rmm::device_buffer temp(temp_bytes, owner.stream(), owner.scratch_mr());

  check(ops.decompress_async(static_cast<void const* const*>(d_comp_ptrs),
                             static_cast<std::size_t const*>(d_comp_bytes),
                             static_cast<std::size_t const*>(d_uncomp_bytes),
                             static_cast<std::size_t*>(d_actual),
                             count,
                             temp.data(),
                             temp_bytes,
                             static_cast<void* const*>(d_uncomp_ptrs),
                             static_cast<nvcompStatus_t*>(d_statuses),
                             s),
        "decompress_async");
  if (!selected_ids.empty()) {
    // The result tables are adjacent. Observe completion on the submitting stream before the
    // caller can use any bytes in the selected chunks.
    std::vector<std::byte> results(sz_size + sz_stat);
    owner.read_bytes(results.data(), d_actual, results.size());
    for (std::size_t i = 0; i < count; ++i) {
      nvcompStatus_t status;
      std::memcpy(&status, results.data() + sz_size + i * sizeof(status), sizeof(status));
      if (status != nvcompSuccess) {
        throw std::runtime_error("nvcomp batched: selected block " +
                                 std::to_string(selected_ids[i]) + " failed (status " +
                                 std::to_string(static_cast<int>(status)) + ")");
      }
      std::size_t actual;
      std::memcpy(&actual, results.data() + i * sizeof(actual), sizeof(actual));
      if (actual != h_uncomp_bytes[i]) {
        throw std::runtime_error(
          "nvcomp batched: selected block " + std::to_string(selected_ids[i]) + " decoded " +
          std::to_string(actual) + " of " + std::to_string(h_uncomp_bytes[i]) + " bytes");
      }
    }
  }
  return count;
}

void check_layout(nvcomp_frame_layout const& layout, std::size_t frame_size, std::size_t out_bytes)
{
  auto const num_chunks = layout.compressed_bytes.size();
  auto const chunk      = layout.chunk_bytes;
  if (layout.frame_bytes != frame_size || chunk == 0 || num_chunks != 1 + (out_bytes - 1) / chunk)
    throw std::runtime_error("nvcomp batched: cached layout does not match frame or output size");
}

}  // namespace

void batched_decompress_bytes(batched_codec_ops const& ops,
                              void const* frame,
                              std::size_t frame_size,
                              void* dst,
                              std::size_t out_bytes,
                              decode_frame& owner,
                              nvcomp_frame_layout const* layout)
{
  if (out_bytes == 0) return;

  auto const* fbase = static_cast<std::uint8_t const*>(frame);

  nvcomp_frame_layout local;
  if (!layout) {
    local  = read_batched_frame_layout(frame, frame_size, out_bytes, owner);
    layout = &local;
  }
  check_layout(*layout, frame_size, out_bytes);
  std::size_t const num_chunks = layout->compressed_bytes.size();
  std::size_t const chunk      = layout->chunk_bytes;

  auto* d            = static_cast<std::uint8_t*>(dst);
  std::size_t cursor = align_up(16 + num_chunks * sizeof(std::size_t), kFrameAlign);
  submit_batched_decompress(ops,
                            num_chunks,
                            chunk,
                            out_bytes,
                            owner,
                            {},
                            [&](std::size_t i,
                                void const*& comp_ptr,
                                std::size_t& comp_bytes,
                                std::size_t& uncomp_bytes,
                                void*& uncomp_ptr) {
                              comp_bytes   = layout->compressed_bytes[i];
                              comp_ptr     = fbase + cursor;
                              cursor       = align_up(cursor + comp_bytes, kFrameAlign);
                              uncomp_bytes = std::min(chunk, out_bytes - i * chunk);
                              uncomp_ptr   = d + i * chunk;
                            });
}

std::size_t batched_decompress_chunks(batched_codec_ops const& ops,
                                      void const* frame,
                                      std::size_t frame_size,
                                      std::size_t out_bytes,
                                      nvcomp_frame_layout const& layout,
                                      std::span<std::uint32_t const> chunk_ids,
                                      void* dst,
                                      decode_frame& owner)
{
  if (chunk_ids.empty()) return 0;
  if (out_bytes == 0)
    throw std::runtime_error("nvcomp batched: chunk selected from an empty frame");
  check_layout(layout, frame_size, out_bytes);
  std::size_t const num_chunks = layout.compressed_bytes.size();
  std::size_t const chunk      = layout.chunk_bytes;
  // Chunk payloads follow one another at kFrameAlign, so a chunk's position depends on the sizes
  // of all chunks before it, selected or not.
  std::vector<std::size_t> positions(num_chunks);
  std::size_t cursor = align_up(16 + num_chunks * sizeof(std::size_t), kFrameAlign);
  for (std::size_t i = 0; i < num_chunks; ++i) {
    positions[i] = cursor;
    cursor       = align_up(cursor + layout.compressed_bytes[i], kFrameAlign);
  }
  for (std::size_t k = 0; k < chunk_ids.size(); ++k) {
    if (chunk_ids[k] >= num_chunks || (k > 0 && chunk_ids[k] <= chunk_ids[k - 1]))
      throw std::runtime_error("nvcomp batched: selected chunk ids are not ascending frame chunks");
  }
  auto const* fbase       = static_cast<std::uint8_t const*>(frame);
  auto* d                 = static_cast<std::uint8_t*>(dst);
  std::size_t total_bytes = 0;
  for (auto const id : chunk_ids)
    total_bytes += std::min(chunk, out_bytes - id * chunk);
  return submit_batched_decompress(ops,
                                   chunk_ids.size(),
                                   chunk,
                                   total_bytes,
                                   owner,
                                   chunk_ids,
                                   [&](std::size_t k,
                                       void const*& comp_ptr,
                                       std::size_t& comp_bytes,
                                       std::size_t& uncomp_bytes,
                                       void*& uncomp_ptr) {
                                     std::size_t const id = chunk_ids[k];
                                     comp_bytes           = layout.compressed_bytes[id];
                                     comp_ptr             = fbase + positions[id];
                                     uncomp_bytes         = std::min(chunk, out_bytes - id * chunk);
                                     uncomp_ptr           = d + k * chunk;
                                   });
}

}  // namespace detail
}  // namespace simpatico
