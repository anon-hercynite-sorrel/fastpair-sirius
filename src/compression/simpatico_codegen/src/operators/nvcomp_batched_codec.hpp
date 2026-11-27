// SPDX-License-Identifier: Apache-2.0
// Low-level (batched) nvcomp codec driver — turns a contiguous device byte
// range into a self-describing "frame" and back. Used by the byte-stream nvcomp
// codecs (LZ4, Snappy, GDeflate, Cascaded, ANS, bitcomp) through the shared
// helpers in nvcomp_simple_compressor.hpp.
//
// Why the batched API and not nvcomp's high-level Manager: it leaves every
// device allocation to the caller (RMM), so we keep memory management in our own
// hands — an OOM throws cleanly at our alloc site instead of nvcomp faulting a
// kernel and corrupting the CUDA context.
//
// A `batched_codec_ops` bundle carries a codec's type-erased entry points with
// its compress/decompress opts pre-bound.
//
//   frame layout (little-endian, x86-64 only):
//     u64  num_chunks
//     u64  chunk_size                 // uncompressed bytes per chunk (last is smaller)
//     u64  comp_sizes[num_chunks]     // actual compressed bytes per chunk
//     <pad to 256>
//     compressed chunk data, each chunk 256-byte (kFrameAlign) aligned

#pragma once
#include "util/nvcomp_frame_layout.hpp"

#include <rmm/device_buffer.hpp>
#include <rmm/resource_ref.hpp>

#include <cuda/stream>
#include <cuda_runtime.h>

#include <nvcomp/shared_types.h>
#include <nvcomp/version.h>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <span>
#include <utility>

namespace simpatico {
class decode_frame;
namespace detail {

#if NVCOMP_VER_MAJOR >= 6
#define SIMPATICO_NVCOMP_GET_TEMP_SIZE(name, stream, ...) name(__VA_ARGS__, stream)
#else
#define SIMPATICO_NVCOMP_GET_TEMP_SIZE(name, stream, ...) name##Async(__VA_ARGS__)
#endif

// Chunk offsets within a frame (and per-chunk output slots) are aligned to this.
inline constexpr std::size_t kFrameAlign = 256;

// Type-erased bundle of a codec's batched entry points, with opts pre-bound.
// Pure functions of their arguments (no stream/device state), so a single
// instance is safe to share across threads and calls.
struct batched_codec_ops {
  std::size_t chunk_size = 64 * 1024;  // uncompressed bytes per chunk

  std::function<nvcompStatus_t(std::size_t num_chunks,
                               std::size_t max_uncomp_chunk,
                               std::size_t* temp_bytes,
                               std::size_t max_total_uncomp,
                               cudaStream_t stream)>
    compress_get_temp_size;

  std::function<nvcompStatus_t(std::size_t max_uncomp_chunk, std::size_t* max_out)>
    compress_get_max_output;

  std::function<nvcompStatus_t(void const* const* uncomp_ptrs,
                               std::size_t const* uncomp_bytes,
                               std::size_t max_uncomp_chunk,
                               std::size_t num_chunks,
                               void* temp,
                               std::size_t temp_bytes,
                               void* const* comp_ptrs,
                               std::size_t* comp_bytes,
                               nvcompStatus_t* statuses,
                               cudaStream_t stream)>
    compress_async;

  std::function<nvcompStatus_t(std::size_t num_chunks,
                               std::size_t max_uncomp_chunk,
                               std::size_t* temp_bytes,
                               std::size_t max_total_uncomp,
                               cudaStream_t stream)>
    decompress_get_temp_size;

  std::function<nvcompStatus_t(void const* const* comp_ptrs,
                               std::size_t const* comp_bytes,
                               std::size_t const* uncomp_buffer_bytes,
                               std::size_t* actual_uncomp_bytes,
                               std::size_t num_chunks,
                               void* temp,
                               std::size_t temp_bytes,
                               void* const* uncomp_ptrs,
                               nvcompStatus_t* statuses,
                               cudaStream_t stream)>
    decompress_async;
};

// Compress `n_bytes` from device `src` into a self-describing frame. Returns the
// frame buffer and its host layout (including the exact byte size). Throws std::bad_alloc on OOM
// (from RMM) or std::runtime_error on an nvcomp/CUDA error. Enqueued on `stream`; the frame is
// coherent on return (the function synchronizes internally).
std::pair<std::unique_ptr<rmm::device_buffer>, nvcomp_frame_layout> batched_compress_bytes(
  batched_codec_ops const& ops,
  void const* src,
  std::size_t n_bytes,
  ::cuda::stream_ref stream,
  rmm::device_async_resource_ref mr);

// Decompress a frame produced by batched_compress_bytes into device `dst`, which
// must hold exactly `out_bytes` (the original uncompressed size). No-op when
// `out_bytes == 0`. Enqueued on `owner.stream()`; the chunk tables uploaded from
// the host live in `owner` until session completion.
void batched_decompress_bytes(batched_codec_ops const& ops,
                              void const* frame,
                              std::size_t frame_size,
                              void* dst,
                              std::size_t out_bytes,
                              decode_frame& owner,
                              nvcomp_frame_layout const* layout = nullptr);

// Decompress only the frame chunks @p chunk_ids (ascending) into consecutive @p layout.chunk_bytes
// slots of device @p dst, slot k receiving chunk chunk_ids[k]; @p dst must hold chunk_ids.size()
// slots (the final slot only that chunk's bytes when it is the frame's partial last chunk).
// @p out_bytes is the frame's whole uncompressed size. No-op for no chunks. Enqueued like
// batched_decompress_bytes. Returns the number of blocks submitted and verified.
std::size_t batched_decompress_chunks(batched_codec_ops const& ops,
                                      void const* frame,
                                      std::size_t frame_size,
                                      std::size_t out_bytes,
                                      nvcomp_frame_layout const& layout,
                                      std::span<std::uint32_t const> chunk_ids,
                                      void* dst,
                                      decode_frame& owner);

// The entry points of the codecs whose frames a selected string decode addresses by chunk.
batched_codec_ops const& snappy_batched_ops();
batched_codec_ops const& ans_batched_ops();
batched_codec_ops const& lz4_batched_ops();
batched_codec_ops const& deflate_batched_ops();

std::size_t read_batched_frame_chunk_size(void const* frame,
                                          std::size_t frame_size,
                                          std::size_t out_bytes,
                                          ::cuda::stream_ref stream);
nvcomp_frame_layout read_batched_frame_layout(void const* frame,
                                              std::size_t frame_size,
                                              std::size_t out_bytes,
                                              decode_frame& owner);

}  // namespace detail
}  // namespace simpatico
