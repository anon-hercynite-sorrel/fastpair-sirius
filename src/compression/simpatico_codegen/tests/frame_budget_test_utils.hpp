// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "operators/nvcomp_batched_codec.hpp"
#include "test_utils.hpp"

#include <stdexcept>
#include <string_view>

inline simpatico::detail::batched_codec_ops const& frame_test_ops(std::string_view codec)
{
  using namespace simpatico::detail;
  if (codec == "snappy") return snappy_batched_ops();
  if (codec == "ans") return ans_batched_ops();
  if (codec == "lz4") return lz4_batched_ops();
  if (codec == "deflate") return deflate_batched_ops();
  throw std::invalid_argument("unknown test frame codec");
}

inline std::size_t frame_budget(std::size_t rows,
                                std::size_t chars,
                                ::cuda::stream_ref stream,
                                std::string_view codec = "snappy",
                                std::size_t block_size = 65536)
{
  auto const blocks     = (chars + block_size - 1) / block_size;
  std::size_t workspace = 0;
  if (blocks) {
    expect(frame_test_ops(codec).decompress_get_temp_size(
             blocks, block_size, &workspace, chars, stream.get()) == nvcompSuccess,
           "nvCOMP workspace query failed");
  }
  // Survivor lengths, source positions, wide offsets, validity gather indices and bitmask.
  auto const survivors = (rows + 1) * 12 + rows * 12 + ((rows + 511) / 512) * 64;
  // nvCOMP pointers and sizes, status, block marks and compacted block indices.
  return chars + survivors + blocks * (5 * sizeof(void*) + sizeof(nvcompStatus_t) + 8) + workspace;
}
