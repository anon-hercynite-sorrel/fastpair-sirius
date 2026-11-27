// SPDX-License-Identifier: Apache-2.0
// nvcomp Snappy compressor (low-level batched API).
// Compress/decompress bodies are shared via nvcomp_simple_compressor.hpp; the
// frame format and all device allocations live in nvcomp_batched_codec.{hpp,cu}.

#include "nvcomp_batched_codec.hpp"
#include "nvcomp_simple_compressor.hpp"
#include "util/snappy_backend.hpp"

#include <nvcomp/snappy.h>

namespace simpatico {

namespace {

constexpr size_t kSnappyChunkSize = 64 * 1024;

detail::batched_codec_ops const& snappy_ops()
{
  static detail::batched_codec_ops const ops = [] {
    nvcompBatchedSnappyCompressOpts_t copts   = nvcompBatchedSnappyCompressDefaultOpts;
    nvcompBatchedSnappyDecompressOpts_t dopts = nvcompBatchedSnappyDecompressDefaultOpts;
    // Preserve the vendor defaults, including reserved[56]; override only these two fields.
    dopts.backend                             = snappy_backend().value;
    dopts.sort_before_hw_decompress           = 0;

    detail::batched_codec_ops o;
    o.chunk_size             = kSnappyChunkSize;
    o.compress_get_temp_size = [copts](
                                 size_t nc, size_t mc, size_t* tb, size_t mt, cudaStream_t stream) {
      return SIMPATICO_NVCOMP_GET_TEMP_SIZE(
        nvcompBatchedSnappyCompressGetTempSize, stream, nc, mc, copts, tb, mt);
    };
    o.compress_get_max_output = [copts](size_t mc, size_t* mo) {
      return nvcompBatchedSnappyCompressGetMaxOutputChunkSize(mc, copts, mo);
    };
    o.compress_async = [copts](void const* const* up,
                               size_t const* ub,
                               size_t mc,
                               size_t nc,
                               void* t,
                               size_t tb,
                               void* const* cp,
                               size_t* cb,
                               nvcompStatus_t* st,
                               cudaStream_t s) {
      return nvcompBatchedSnappyCompressAsync(up, ub, mc, nc, t, tb, cp, cb, copts, st, s);
    };
    o.decompress_get_temp_size =
      [dopts](size_t nc, size_t mc, size_t* tb, size_t mt, cudaStream_t stream) {
        log_snappy_decompress(nc, mt);
        return check_snappy_decompress(
          SIMPATICO_NVCOMP_GET_TEMP_SIZE(
            nvcompBatchedSnappyDecompressGetTempSize, stream, nc, mc, dopts, tb, mt),
          "decompress_get_temp_size");
      };
    o.decompress_async = [dopts](void const* const* cp,
                                 size_t const* cb,
                                 size_t const* ubuf,
                                 size_t* actual,
                                 size_t nc,
                                 void* t,
                                 size_t tb,
                                 void* const* up,
                                 nvcompStatus_t* st,
                                 cudaStream_t s) {
      return check_snappy_decompress(
        nvcompBatchedSnappyDecompressAsync(cp, cb, ubuf, actual, nc, t, tb, up, dopts, st, s),
        "decompress_async");
    };
    return o;
  }();
  return ops;
}

}  // namespace

std::unique_ptr<cudf::column> snappy_compressed_representation::decompress(
  decode_frame& frame) const
{
  auto output          = detail::nvcomp_decompress_impl(snappy_ops(), *this, frame);
  frame.snappy_output_ = output->view().head<void>();
  return output;
}

std::unique_ptr<compressed_representation> snappy_compressor::compress(
  cudf::column_view col, ::cuda::stream_ref stream, rmm::device_async_resource_ref mr)
{
  return detail::nvcomp_compress_impl<snappy_compressed_representation>(
    snappy_ops, col, stream, mr);
}

namespace detail {
batched_codec_ops const& snappy_batched_ops() { return snappy_ops(); }
}  // namespace detail

}  // namespace simpatico
