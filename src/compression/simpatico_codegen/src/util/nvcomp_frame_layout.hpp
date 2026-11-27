// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstddef>
#include <vector>

namespace simpatico::detail {

// Host-only immutable frame metadata, prepared before plan publication. No device pointers:
// tables are rebuilt against the current payload/output addresses on each decode.
struct nvcomp_frame_layout {
  std::size_t chunk_bytes = 0;
  std::vector<std::size_t> compressed_bytes;
  std::size_t frame_bytes = 0;  // extent these lengths describe, independent of the payload address
};

// Parse and validate a frame already resident in host memory during file loading.
nvcomp_frame_layout read_host_batched_frame_layout(void const* frame,
                                                   std::size_t frame_size,
                                                   std::size_t out_bytes);

}  // namespace simpatico::detail
