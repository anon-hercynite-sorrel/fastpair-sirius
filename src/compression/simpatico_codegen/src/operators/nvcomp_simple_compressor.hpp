// SPDX-License-Identifier: Apache-2.0
// Shared compress / decompress helpers for the byte-stream nvcomp codecs
// (LZ4, Snappy, GDeflate, Cascaded, ANS, bitcomp).
//
// All codecs drive nvcomp through the low-level *batched* API via a
// `batched_codec_ops` bundle (see nvcomp_batched_codec.hpp). That API leaves
// every device allocation to us (RMM), so nvcomp never allocates its own memory
// and an out-of-memory condition throws cleanly instead of faulting a kernel and
// corrupting the CUDA context. Each codec's .cu file builds its ops once and
// calls the fixed-width helpers here (and, for STRING-capable codecs, the
// primitives in nvcomp_string_support.hpp).

#pragma once

#include "codegen/plan/representation.hpp"
#include "decode/decode_session.hpp"
#include "nvcomp_batched_codec.hpp"

#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/traits.hpp>

#include <rmm/device_buffer.hpp>

#include <cuda/stream>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <utility>

namespace simpatico {
namespace detail {

// Each codec supplies its representation type and any options needed to reconstruct it.
template <typename Rep, typename GetOps, typename... Options>
std::unique_ptr<Rep> nvcomp_compress_impl(GetOps get_ops,
                                          cudf::column_view const& col,
                                          ::cuda::stream_ref stream,
                                          rmm::device_async_resource_ref mr,
                                          Options... options)
{
  if (col.size() == 0) {
    return std::make_unique<Rep>(
      col.type(), 0, std::make_unique<rmm::device_buffer>(0, stream, mr), 0, 0, options...);
  }
  if (!cudf::is_fixed_width(col.type())) {
    throw std::runtime_error("nvcomp simple compressor: only fixed-width columns supported");
  }
  auto const elem_size         = static_cast<std::size_t>(cudf::size_of(col.type()));
  auto const uncompressed_size = static_cast<std::size_t>(col.size()) * elem_size;
  auto const* src              = static_cast<std::uint8_t const*>(col.head<void>()) +
                                 static_cast<std::size_t>(col.offset()) * elem_size;
  auto [buffer, layout] = batched_compress_bytes(get_ops(), src, uncompressed_size, stream, mr);
  auto rep              = std::make_unique<Rep>(
    col.type(), col.size(), std::move(buffer), layout.frame_bytes, uncompressed_size, options...);
  rep->frame_layout = std::make_shared<nvcomp_frame_layout const>(std::move(layout));
  return rep;
}

inline std::unique_ptr<cudf::column> nvcomp_decompress_impl(batched_codec_ops const& ops,
                                                            nvcomp_payload_rep const& rep,
                                                            decode_frame& frame)
{
  auto output = cudf::make_fixed_width_column(
    rep.original_type, rep.num_rows, cudf::mask_state::UNALLOCATED, frame.stream(), frame.mr());
  if (rep.num_rows == 0 || rep.payload_data() == nullptr || rep.payload_size() == 0) return output;

  auto const out_bytes = static_cast<std::size_t>(rep.num_rows) * cudf::size_of(rep.original_type);
  batched_decompress_bytes(ops,
                           rep.payload_data(),
                           rep.payload_size(),
                           output->mutable_view().head<void>(),
                           out_bytes,
                           frame,
                           rep.frame_layout.get());
  return output;
}

}  // namespace detail
}  // namespace simpatico
