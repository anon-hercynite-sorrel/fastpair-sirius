// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "codegen/plan/operator_registry.hpp"
#include "util/nvcomp_frame_layout.hpp"

#include <cudf/types.hpp>

#include <cstdint>
#include <memory>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

namespace simpatico {

// Per-leaf decode metadata reported by compressed_representation::describe_meta.
namespace leaf_meta {

struct none {};

// ALP-RD: right_bw (and the K=8 dict) are COLUMN-WIDE — chosen once and applied
// to every row, keeping right_parts fixed-width with a uniform shift/mask
// decode. The cwida/ALP reference instead fits these per rowgroup (1024 values).
// Per-chunk right_bw would cut exceptions on heterogeneous columns but needs a
// [num_chunks] right_bw array, an 8×num_chunks dict, and a chunk-indexed decode.
struct alp_rd {
  std::uint8_t right_bw = 0;
};

// Shared by codecs using the batched frame. Only size and type are serialized.
// The host layout describes the immutable frame independently of its device address.
struct nvcomp_batched {
  std::uint64_t uncompressed_size = 0;
  std::int32_t original_type_id   = 0;
  std::shared_ptr<detail::nvcomp_frame_layout const> frame_layout{};
};

struct ans : nvcomp_batched {
  // 5: pre-v6 payload; 6: v6 payload. Independent of the .hpln layout version.
  std::uint8_t payload_format = 0;  // Unspecified metadata is rejected.
};

struct bitcomp : nvcomp_batched {
  std::int32_t algorithm = 0;
};

struct nvcomp_cascaded : nvcomp_batched {
  std::int32_t num_deltas     = 1;
  std::int32_t num_RLEs       = 2;
  std::int32_t use_bp         = 1;
  std::uint8_t payload_format = 0;  // Same format identifiers as ANS.
};

struct snappy : nvcomp_batched {};
struct lz4 : nvcomp_batched {};
struct deflate : nvcomp_batched {};

// Five compact channels: codes, row_offsets (byte offsets), dict_bytes,
// dict_offsets, and batch_prefixes (decoded byte offsets at warp batch boundaries).
struct onpair {
  constexpr onpair(std::uint64_t rows  = 0,
                   std::uint64_t bytes = 0,
                   std::uint8_t bits   = 12,
                   std::int32_t type   = static_cast<std::int32_t>(cudf::type_id::STRING),
                   std::uint16_t batch = 128)
    : num_rows(rows),
      decoded_bytes(bytes),
      dict_bits(bits),
      tokens_per_batch(batch),
      original_type_id(type)
  {
  }

  std::uint64_t num_rows;
  std::uint64_t decoded_bytes;
  std::uint8_t dict_bits;
  std::uint16_t tokens_per_batch;  // Uses the original padding before original_type_id.
  std::int32_t original_type_id;

  std::uint32_t batch_count(std::uint64_t tokens) const
  {
    return static_cast<std::uint32_t>(tokens_per_batch == 192 ? (tokens + 191) / 192
                                                              : (tokens + 127) / 128);
  }
};

}  // namespace leaf_meta

using leaf_meta_v = std::variant<leaf_meta::none,
                                 leaf_meta::alp_rd,
                                 leaf_meta::ans,
                                 leaf_meta::bitcomp,
                                 leaf_meta::nvcomp_cascaded,
                                 leaf_meta::snappy,
                                 leaf_meta::lz4,
                                 leaf_meta::deflate,
                                 leaf_meta::onpair>;

// Only these leaf kinds carry the self-describing batched frame. Match the metadata
// tag as well as the operator before interpreting the frame layout.
template <typename Metadata>
auto batched_meta(OpId kind, Metadata& meta) -> std::conditional_t<std::is_const_v<Metadata>,
                                                                   leaf_meta::nvcomp_batched const*,
                                                                   leaf_meta::nvcomp_batched*>
{
  switch (kind) {
    case OpId::Ans: return std::get_if<leaf_meta::ans>(&meta);
    case OpId::Bitcomp: return std::get_if<leaf_meta::bitcomp>(&meta);
    case OpId::NvcompCascaded: return std::get_if<leaf_meta::nvcomp_cascaded>(&meta);
    case OpId::Snappy: return std::get_if<leaf_meta::snappy>(&meta);
    case OpId::Lz4: return std::get_if<leaf_meta::lz4>(&meta);
    case OpId::Deflate: return std::get_if<leaf_meta::deflate>(&meta);
    default: return nullptr;
  }
}

// ---------------------------------------------------------------------------
// Per-buffer descriptor used by compressed_table::describe() and the IO layer.
// ---------------------------------------------------------------------------

struct leaf_buffer_desc {
  std::string name;
  std::uint8_t type_tag    = 0;
  std::uint64_t num_rows   = 0;
  std::uint64_t size_bytes = 0;
  const void* device_ptr   = nullptr;
};

// `slot` value marking a leaf as its node's own rep (rather than a terminal
// output-channel rep, which uses the output-port index).
inline constexpr std::int32_t kSelfRepSlot = -1;

// Descriptor for one compressed leaf (one rep slot in the PlanTree), located
// structurally: `node_index` is its owning node, `slot` is kSelfRepSlot for the
// node's own rep or an output-port index for a terminal channel rep. Produced
// by compressed_table::describe(); consumed by the file writer and read path.
struct leaf_desc {
  std::uint32_t node_index = 0;
  std::int32_t slot        = kSelfRepSlot;
  OpId kind                = OpId::Unknown;
  std::uint8_t type_tag    = 0;  // decoded column dtype tag
  // This node's own output length (elements). For a nested fused subtree this is
  // the channel length, which can be far smaller than the enclosing column's row
  // count; the codegen decode grid is ceil(num_rows / kChunkSize), so it MUST use
  // this per-node value rather than the column row count (see rep_from_leaf_desc).
  std::uint64_t num_rows = 0;
  leaf_meta_v meta{leaf_meta::none{}};
  std::vector<leaf_buffer_desc> buffers;  // channel buffers in named_channels() order
};

// ---------------------------------------------------------------------------
// Type-tag encoding. One table drives both directions; 255 = unsupported type.
// ---------------------------------------------------------------------------

inline constexpr std::pair<cudf::type_id, std::uint8_t> kTypeTags[] = {
  {cudf::type_id::INT8, 0},
  {cudf::type_id::INT16, 1},
  {cudf::type_id::INT32, 2},
  {cudf::type_id::INT64, 3},
  {cudf::type_id::UINT8, 4},
  {cudf::type_id::UINT16, 5},
  {cudf::type_id::UINT32, 6},
  {cudf::type_id::UINT64, 7},
  {cudf::type_id::FLOAT32, 8},
  {cudf::type_id::FLOAT64, 9},
  {cudf::type_id::STRING, 10},
  // Fixed-point types. Their scale is not captured by the tag alone (it is
  // serialized alongside the column dtype); tag_to_dtype returns scale 0.
  {cudf::type_id::DECIMAL32, 11},
  {cudf::type_id::DECIMAL64, 12},
  {cudf::type_id::DECIMAL128, 13},
  // Chrono types, compressed as their same-width integer storage (DATE32 =
  // days as int32; the timestamp/duration families as int64 except *_DAYS);
  // the tag restores the logical type on read (apply_stored_dtype).
  {cudf::type_id::TIMESTAMP_DAYS, 14},
  {cudf::type_id::TIMESTAMP_SECONDS, 15},
  {cudf::type_id::TIMESTAMP_MILLISECONDS, 16},
  {cudf::type_id::TIMESTAMP_MICROSECONDS, 17},
  {cudf::type_id::TIMESTAMP_NANOSECONDS, 18},
  {cudf::type_id::DURATION_DAYS, 19},
  {cudf::type_id::DURATION_SECONDS, 20},
  {cudf::type_id::DURATION_MILLISECONDS, 21},
  {cudf::type_id::DURATION_MICROSECONDS, 22},
  {cudf::type_id::DURATION_NANOSECONDS, 23},
  {cudf::type_id::BOOL8, 24},
};

inline std::uint8_t dtype_to_tag(cudf::data_type dt) noexcept
{
  for (auto const& [id, tag] : kTypeTags)
    if (id == dt.id()) return tag;
  return 255;
}

inline cudf::data_type tag_to_dtype(std::uint8_t tag) noexcept
{
  for (auto const& [id, t] : kTypeTags)
    if (t == tag) return cudf::data_type{id};
  return cudf::data_type{cudf::type_id::EMPTY};
}

}  // namespace simpatico
