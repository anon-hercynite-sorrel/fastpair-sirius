// SPDX-License-Identifier: Apache-2.0
// Survivor-selective chunk addressing for string chars stored as a batched nvCOMP frame: which
// fixed-size chunks a selection touches, and where each survivor's bytes land once only those
// chunks are decompressed, side by side, in ascending chunk order.

#pragma once

#include "codegen/plan/validity.hpp"
#include "codegen/selection/selection.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace simpatico {
class decode_frame;
namespace detail {

// Gather the sidecar in survivor order and clear null rows' lengths before scanning offsets.
rmm::device_buffer select_string_validity(validity_sidecar const& validity,
                                          sirius::codegen::selection_mask const& selection,
                                          std::int32_t* lengths,
                                          cudf::size_type& null_count,
                                          decode_frame& frame);

struct selected_frame_chunks {
  std::vector<std::uint32_t> ids;  ///< Marked chunk ids, ascending.
  std::int64_t output_bytes = 0;   ///< output_offsets[survivors]: the survivors' total bytes.
};

/// Mark every @p chunk_bytes chunk of a @p chars_bytes chars buffer that holds a byte of a
/// survivor and compact the marked ids. Survivor j covers @p lengths[j] bytes from
/// @p source_offsets[j]. Because a survivor's chunks are consecutive and all marked, they occupy
/// consecutive slots of a buffer holding the marked chunks in order, so each nonempty survivor's
/// start is rewritten in place to slot(first chunk) * chunk_bytes + its offset in that chunk; an
/// empty survivor's start becomes 0. One readback returns the ids and @p output_offsets[survivors].
///
/// @throw std::runtime_error if a survivor's range leaves the chars buffer or its total overflows
selected_frame_chunks select_frame_chunks(std::int64_t* source_offsets,
                                          std::int32_t const* lengths,
                                          std::int32_t const* output_offsets,
                                          std::int64_t survivors,
                                          std::size_t chars_bytes,
                                          std::size_t chunk_bytes,
                                          decode_frame& frame);

}  // namespace detail
}  // namespace simpatico
