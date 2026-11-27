// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "codegen/plan/representation.hpp"

#include <array>
#include <cstdint>
#include <optional>

namespace simpatico {

struct PlanNode;
struct PlanTree;

// Transient, stream-ordered selected-decode state; never stored in a plan or serialized.
struct onpair_selected_batches {
  rmm::device_buffer marked;
  rmm::device_buffer scratch_offsets;
  rmm::device_buffer source_offsets;
  std::unique_ptr<cudf::column> output_offsets;  // INT64 exclusive scan, including the total
  std::uint32_t count       = 0;
  std::size_t scratch_bytes = 0;
  std::size_t output_bytes  = 0;
};

// Large-string counterpart of launch_masked_char_copy; the str_split INT32 ABI stays unchanged.
void copy_onpair_selected_chars64(void const* chars,
                                  std::int64_t const* source_offsets,
                                  std::int64_t const* output_offsets,
                                  std::int64_t survivors,
                                  void* output,
                                  decode_frame& frame);

// Called only before publication, including after loading a persisted plan.
void prepare_onpair_nodes(PlanTree& tree,
                          ::cuda::stream_ref stream,
                          rmm::device_async_resource_ref mr,
                          std::optional<rmm::device_async_resource_ref> scratch_mr = {});

// OnPair node preparations completed in this process: pin time, a read that does not defer, and
// the request-local preparations of deferred or unpublished plans. A diagnostic for tests.
std::uint64_t onpair_preparation_count() noexcept;

// Prepares a plan whose OnPair nodes a host read left unprepared; false (and no work) when every
// OnPair node is already prepared, as in every published plan.
bool prepare_deferred_onpair(PlanTree& tree,
                             ::cuda::stream_ref stream,
                             rmm::device_async_resource_ref mr,
                             std::optional<rmm::device_async_resource_ref> scratch_mr = {});

struct onpair_compressed_representation : standalone_compressed_representation {
  // Five compact channels: codes (UINT16), row_offsets (INT32/INT64 byte offsets),
  // dict_bytes (UINT8), dict_offsets (UINT32), batch_prefixes (UINT64 byte offsets).
  // Fresh compression initializes batch_prefixes from the pin-time device sweep.
  onpair_compressed_representation(leaf_meta::onpair metadata,
                                   std::vector<std::unique_ptr<cudf::column>> channels,
                                   ::cuda::stream_ref stream,
                                   rmm::device_async_resource_ref mr,
                                   bool initialize_batch_prefixes = false);

  // Pin-time preparation can borrow identity leaves and own decoded sub-plan channels.
  onpair_compressed_representation(leaf_meta::onpair metadata,
                                   std::array<cudf::column_view, 5> views,
                                   std::vector<std::unique_ptr<cudf::column>> owners,
                                   ::cuda::stream_ref stream,
                                   rmm::device_async_resource_ref mr,
                                   std::optional<rmm::device_async_resource_ref> scratch_mr = {});

  // Scan-time reconstruction owns row_offsets (byte offsets) and borrows the
  // immutable prepared dictionary, codes, and batch_prefixes (byte offsets).
  onpair_compressed_representation(onpair_compressed_representation const& prepared,
                                   std::unique_ptr<cudf::column> row_offsets);

  void finish_preparation(PlanNode const& node);
  std::vector<compressible_output> named_channels(::cuda::stream_ref stream) const override;
  size_t resident_size_bytes(::cuda::stream_ref stream) const override;

  OpId kind() const override { return OpId::OnPair; }
  leaf_meta_v describe_meta() const override { return metadata_; }
  [[nodiscard]] std::unique_ptr<cudf::column> decompress(decode_frame& frame) const override;

  [[nodiscard]] std::unique_ptr<cudf::column> decompress_predicate(decode_predicate const& pred,
                                                                   decode_frame& frame) const;

  cudf::column_view row_byte_offsets() const { return views_[1]; }
  std::uint16_t tokens_per_batch() const { return metadata_.tokens_per_batch; }
  std::uint32_t batch_count() const { return metadata_.batch_count(views_[0].size()); }
  onpair_selected_batches select_batches(cudf::column_view rows,
                                         cudf::column_view indices,
                                         decode_frame& frame) const;
  void decode_batches(onpair_selected_batches const& selected,
                      void* scratch,
                      decode_frame& frame) const;

 private:
  leaf_meta::onpair metadata_;
  std::array<cudf::column_view, 5> views_;
  onpair_compressed_representation const* prepared_ = nullptr;
  bool preparation_only_                            = false;
  void validate_and_stage(::cuda::stream_ref stream,
                          rmm::device_async_resource_ref mr,
                          bool initialize_batch_prefixes                           = false,
                          std::optional<rmm::device_async_resource_ref> scratch_mr = {});
  // Immutable after construction. Only compact channels are named/serialized;
  // these split dictionary planes are staged once for this representation.
  rmm::device_buffer dict_s8_;
  rmm::device_buffer dict_tail_;  // K=4 padded 16-byte entries, K=6 dense high 8 bytes
  rmm::device_buffer lengths_;    // K=4 byte lengths, K=6 packed nibbles
};

struct onpair_compressor : compressor {
  explicit onpair_compressor(std::uint8_t dict_bits = 12) : dict_bits_(dict_bits)
  {
    if (dict_bits < 9 || dict_bits > 16)
      throw std::invalid_argument("onpair: dictionary bits must be in 9..16");
  }

  std::unique_ptr<compressed_representation> compress(cudf::column_view column,
                                                      ::cuda::stream_ref stream,
                                                      rmm::device_async_resource_ref mr) override;

 private:
  std::uint8_t dict_bits_;
};

}  // namespace simpatico
