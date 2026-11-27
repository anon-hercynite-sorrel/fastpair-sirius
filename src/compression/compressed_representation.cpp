/*
 * Copyright 2026, Sirius Contributors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "compressed_representation.hpp"

#include "api/compressed_table_io.hpp"
#include "compression/decompression_pushdown_policy.hpp"
#include "decode/decode_scratch_resource.hpp"
#include "device_compressed_blob.hpp"
#include "memory/size_arithmetic.hpp"
#include "util/decoded_like.hpp"

#include <rmm/aligned.hpp>

#include <cuda_runtime.h>

#include <cucascade/error.hpp>

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <utility>

namespace sirius {

namespace {
template <typename Representation, typename FrameScratch, typename NeedsDecodedLike>
std::size_t selection_scratch(Representation const& rep,
                              per_column_byte_sizes const* sizes,
                              std::size_t pin_bytes,
                              std::size_t pin_payload,
                              bool pending_source,
                              FrameScratch frame_scratch,
                              NeedsDecodedLike needs_decoded_like)
{
  using memory::saturating_add;
  using memory::saturating_mul;
  if (!decompression_pushdown_enabled()) return 0;
  auto const& scan = rep.pushdown_scan();
  if (!pending_source && !rep.visibility_mask().has_mask() && (!scan || scan->request().empty()))
    return 0;

  auto const& projection = rep.selected_indices();
  auto const width       = rep.column_names().size();
  auto const count       = projection ? projection->size() : width;
  auto const decoded_slot = [&](std::size_t slot) {
    return needs_decoded_like(projection ? (*projection)[slot] : slot);
  };
  bool decoded_like = false;
  if (simpatico::decoded_like_enabled() && scan) {
    for (std::size_t i = 0; i < std::min(count, scan->request().columns.size()); ++i) {
      if (!scan->request().columns[i].likes.empty() && decoded_slot(i)) {
        decoded_like = true;
        break;
      }
    }
  }
  if (!simpatico::chunk_selective_enabled() && !decoded_like) return 0;
  bool const exact =
    sizes && sizes->uncompressed.size() == width && sizes->compressed.size() == width;
  std::vector<std::size_t> occurrences(!exact && projection ? width : 0);
  std::size_t frames = 0, logical = 0, copies = 1;
  for (std::size_t i = 0; i < count; ++i) {
    auto const physical = projection ? (*projection)[i] : i;
    frames              = saturating_add(frames, frame_scratch(physical));
    if (exact)
      logical = saturating_add(logical, sizes->uncompressed[physical]);
    else if (projection)
      copies = std::max(copies, ++occurrences[physical]);
  }
  if (!frames && !decoded_like) return 0;
  if (!exact) logical = saturating_mul(copies, pin_bytes);

  // Full-width input and gathered output may coexist for every projected column. Credit the
  // output bytes already in the ordinary reservation, including older pins' proportional estimate.
  auto const output = rep.get_uncompressed_data_size_in_bytes();
  auto scratch =
    saturating_add(frames, saturating_add(logical, logical - std::min(logical, output)));
  if (projection && !exact) {
    // The legacy estimate apportions the uploaded payload by column count too.
    auto const payload = saturating_mul(copies, pin_payload);
    scratch = saturating_add(scratch, payload - std::min(payload, rep.get_size_in_bytes()));
  }
  if (scan) {
    for (std::size_t slot = 0; slot < scan->request().columns.size(); ++slot) {
      auto const& column = scan->request().columns[slot];
      if (!column.equals_any.empty()) {
        auto bytes = saturating_mul(saturating_add(column.equals_any.size(), 1), sizeof(int32_t));
        for (auto const& literal : column.equals_any)
          bytes = saturating_add(bytes, literal.size());
        scratch = saturating_add(scratch, bytes);
      }
      if (!column.likes.empty()) {
        if (decoded_like && slot < count && decoded_slot(slot)) {
          // Host plans are probed only after upload. Budget either LIKE route. The retained
          // full column can coexist with decoded channels, the gathering output and row indices.
          // For old pins the entire logical chunk bounds any single selected column.
          auto const physical = projection ? (*projection)[slot] : slot;
          auto const full     = exact ? sizes->uncompressed[physical] : pin_bytes;
          scratch             = saturating_add(scratch, saturating_mul(full, 2));
          // cuDF LIKE and NOT each allocate BOOL8 plus validity. Packing, gather indices,
          // string offsets (possibly INT64), and scan workspace are all row-sized.
          scratch = saturating_add(scratch, saturating_mul(rep.num_rows() + 1, 32));
        }
        // OnPair validates dict_size <= 2^dict_bits and dict_bits <= 16 on upload.
        constexpr std::size_t max_entries = 1u << 16;
        // The K=4 layout (8 + 16 + 1 bytes/entry) bounds either prepared dictionary.
        // K=6 uses 8 + 8 bytes/entry and one length byte per pair.
        scratch = saturating_add(scratch, 25 * max_entries);
        scratch = saturating_add(scratch, saturating_mul(rep.num_rows() + 1, sizeof(uint32_t)));
        // An OnPair value decode can prepare the dictionary even with code matching disabled,
        // but only code-level matching builds literal transition tables.
        if (decoded_like && !simpatico::onpair_like_enabled()) continue;
        for (auto const& like : column.likes)
          for (auto const& literal : like.literals)
            scratch = saturating_add(
              scratch,
              saturating_mul(saturating_add(literal.size(), 1), max_entries * sizeof(uint16_t)));
      }
    }
  }
  // Reserve all mask slots. Another split may change the latch or a join may publish before decode.
  auto const words = codegen::selection_mask::AllocWordsFor(rep.num_rows());
  auto const mask =
    rmm::align_up(std::size_t(words) * sizeof(std::uint32_t), rmm::CUDA_ALLOCATION_ALIGNMENT);
  scratch = saturating_add(scratch, saturating_mul(kMaxSelectionSources, mask));
  // Tiny columns still allocate aligned parameter tables, count/scan workspaces and output buffers.
  scratch = saturating_add(
    scratch, saturating_mul(count, simpatico::decode_scratch_resource::small_workspace_bytes));
  if (codegen::decode_scratch_arena_enabled())
    scratch = saturating_add(scratch, simpatico::decode_scratch_resource::max_bytes);
  return scratch;
}
}  // namespace

std::size_t compressed_host_representation::selected_decode_scratch_bytes(bool pending_source) const
{
  auto const frames = simpatico::pinned_frame_scratch(nvcomp_metadata());
  if (frames.empty() && !simpatico::decoded_like_enabled()) return 0;
  if (!frames.empty() && frames.size() != column_names().size())
    throw std::logic_error("pin selection metadata does not match the columns");
  return selection_scratch(
    *this,
    _column_sizes.get(),
    _pin_logical_bytes,
    _blob->payload_bytes,
    pending_source,
    [&](std::size_t i) { return frames.empty() ? 0 : frames[i]; },
    [](std::size_t) { return true; });  // Host plans are probed after upload.
}

std::size_t compressed_device_representation::selected_decode_scratch_bytes(
  bool pending_source) const
{
  if (!has_table()) return 0;
  return selection_scratch(
    *this,
    _column_sizes.get(),
    _pin_logical_bytes,
    _blob->leaves.size(),
    pending_source,
    [&](std::size_t i) { return simpatico::frame_scratch_bytes(table().columns.at(i)); },
    [&](std::size_t i) {
      auto const& col = table().columns.at(i);
      return col.dtype.id() == cudf::type_id::STRING && col.plan_tree &&
             !simpatico::probe_column(*col.plan_tree).can_answer_like;
    });
}

// ── Block-aware pinned copies ────────────────────────────────────────────────

void copy_device_to_pinned_blocks(
  const void* src_device,
  cucascade::memory::fixed_size_host_memory_resource::multiple_blocks_allocation& dst,
  std::uint64_t dst_offset,
  std::size_t size,
  ::cuda::stream_ref stream)
{
  if (size == 0) return;
  const std::size_t bs = dst.block_size();
  std::size_t d_idx    = dst_offset / bs;
  std::size_t d_off    = dst_offset % bs;
  const auto* src      = static_cast<const std::byte*>(src_device);
  std::size_t copied   = 0;
  while (copied < size) {
    const std::size_t chunk = std::min(size - copied, bs - d_off);
    CUCASCADE_CUDA_TRY(cudaMemcpyAsync(
      dst.at(d_idx).data() + d_off, src + copied, chunk, cudaMemcpyDeviceToHost, stream.get()));
    copied += chunk;
    d_off += chunk;
    if (d_off == bs) {
      ++d_idx;
      d_off = 0;
    }
  }
}

void copy_pinned_blocks_to_device(
  const cucascade::memory::fixed_size_host_memory_resource::multiple_blocks_allocation& src,
  std::uint64_t src_offset,
  void* dst_device,
  std::size_t size,
  ::cuda::stream_ref stream)
{
  if (size == 0) return;
  const std::size_t bs = src.block_size();
  std::size_t s_idx    = src_offset / bs;
  std::size_t s_off    = src_offset % bs;
  auto* dst            = static_cast<std::byte*>(dst_device);
  std::size_t copied   = 0;
  while (copied < size) {
    const std::size_t chunk = std::min(size - copied, bs - s_off);
    CUCASCADE_CUDA_TRY(cudaMemcpyAsync(
      dst + copied, src.at(s_idx).data() + s_off, chunk, cudaMemcpyHostToDevice, stream.get()));
    copied += chunk;
    s_off += chunk;
    if (s_off == bs) {
      ++s_idx;
      s_off = 0;
    }
  }
}

// ── Owning constructor ───────────────────────────────────────────────────────

compressed_host_representation::compressed_host_representation(
  cucascade::memory::memory_space& memory_space,
  std::shared_ptr<pinned_compressed_blob> blob,
  std::vector<std::string> column_names,
  std::size_t compressed_bytes,
  std::size_t uncompressed_bytes,
  std::int64_t num_rows,
  std::shared_ptr<const per_column_byte_sizes> column_sizes)
  : simpatico_compressed_representation(memory_space),
    _blob(std::move(blob)),
    _column_names(std::move(column_names)),
    _compressed_bytes(compressed_bytes),
    _uncompressed_bytes(uncompressed_bytes),
    _pin_logical_bytes(uncompressed_bytes),
    _num_rows(num_rows),
    _column_sizes(std::move(column_sizes))
{
}

// ── Sharing constructor (private) ────────────────────────────────────────────

compressed_host_representation::compressed_host_representation(
  cucascade::memory::memory_space& memory_space,
  std::shared_ptr<pinned_compressed_blob> blob,
  std::vector<std::string> column_names,
  std::size_t compressed_bytes,
  std::size_t uncompressed_bytes,
  std::size_t pin_logical_bytes,
  std::int64_t num_rows,
  std::optional<std::vector<std::size_t>> selected_indices,
  std::shared_ptr<const per_column_byte_sizes> column_sizes)
  : simpatico_compressed_representation(memory_space),
    _blob(std::move(blob)),
    _column_names(std::move(column_names)),
    _compressed_bytes(compressed_bytes),
    _uncompressed_bytes(uncompressed_bytes),
    _pin_logical_bytes(pin_logical_bytes),
    _num_rows(num_rows),
    _selected_indices(std::move(selected_indices)),
    _column_sizes(std::move(column_sizes))
{
}

// ── idata_representation interface ───────────────────────────────────────────

std::unique_ptr<cucascade::idata_representation> compressed_host_representation::clone(
  ::cuda::stream_ref /*stream*/) const
{
  // Share the same backing blob — no byte copy needed.
  auto copy = std::unique_ptr<compressed_host_representation>(
    new compressed_host_representation(get_memory_space(),
                                       _blob,
                                       _column_names,
                                       _compressed_bytes,
                                       _uncompressed_bytes,
                                       _pin_logical_bytes,
                                       _num_rows,
                                       _selected_indices,
                                       _column_sizes));
  // The request is indexed by the selected column list, which the clone shares.
  copy->set_pushdown_scan(_pushdown_scan);
  copy->set_visibility_mask(_visibility_mask);
  return copy;
}

// ── Projection ───────────────────────────────────────────────────────────────

namespace {

/// Bytes of the selected columns: exact sum when the pin recorded per-column
/// sizes, otherwise the totals scaled by the selected fraction of columns.
std::pair<std::size_t, std::size_t> projected_bytes(
  std::span<const std::size_t> absolute,
  const std::shared_ptr<const per_column_byte_sizes>& sizes,
  std::size_t total_columns,
  std::size_t total_compressed,
  std::size_t total_uncompressed)
{
  if (sizes && sizes->compressed.size() == total_columns &&
      sizes->uncompressed.size() == total_columns) {
    std::size_t compressed   = 0;
    std::size_t uncompressed = 0;
    for (auto idx : absolute) {
      compressed   = memory::saturating_add(compressed, sizes->compressed[idx]);
      uncompressed = memory::saturating_add(uncompressed, sizes->uncompressed[idx]);
    }
    return {compressed, uncompressed};
  }
  if (total_columns == 0) { return {0, 0}; }
  const auto selected = absolute.size();
  return {total_compressed * selected / total_columns,
          total_uncompressed * selected / total_columns};
}

}  // namespace

std::unique_ptr<compressed_host_representation> compressed_host_representation::select_columns(
  std::span<const std::size_t> indices) const
{
  // Build absolute indices into _column_names, respecting any existing projection.
  std::vector<std::size_t> absolute;
  absolute.reserve(indices.size());
  for (auto idx : indices) {
    if (_selected_indices.has_value()) {
      if (idx >= _selected_indices->size()) {
        throw std::out_of_range(
          "[compressed_host_representation::select_columns] index out of range");
      }
      absolute.push_back((*_selected_indices)[idx]);
    } else {
      if (idx >= _column_names.size()) {
        throw std::out_of_range(
          "[compressed_host_representation::select_columns] index out of range");
      }
      absolute.push_back(idx);
    }
  }

  auto const [compressed, uncompressed] = projected_bytes(
    absolute, _column_sizes, _column_names.size(), _compressed_bytes, _uncompressed_bytes);

  return std::unique_ptr<compressed_host_representation>(
    new compressed_host_representation(get_memory_space(),
                                       _blob,
                                       _column_names,
                                       compressed,
                                       uncompressed,
                                       _pin_logical_bytes,
                                       _num_rows,
                                       std::move(absolute),
                                       _column_sizes));
}

// ── compressed_device_representation ─────────────────────────────────────────

namespace {

// Resolve caller-relative column indices to absolute indices into the chunk's
// full column list, honoring any projection already applied. Shared by the host
// and device select_columns() implementations.
std::vector<std::size_t> resolve_absolute_indices(
  std::span<const std::size_t> indices,
  const std::optional<std::vector<std::size_t>>& existing_selection,
  std::size_t num_all_columns)
{
  std::vector<std::size_t> absolute;
  absolute.reserve(indices.size());
  for (auto idx : indices) {
    if (existing_selection.has_value()) {
      if (idx >= existing_selection->size()) {
        throw std::out_of_range(
          "[compressed_device_representation::select_columns] index out of range");
      }
      absolute.push_back((*existing_selection)[idx]);
    } else {
      if (idx >= num_all_columns) {
        throw std::out_of_range(
          "[compressed_device_representation::select_columns] index out of range");
      }
      absolute.push_back(idx);
    }
  }
  return absolute;
}

}  // namespace

compressed_device_representation::compressed_device_representation(
  cucascade::memory::memory_space& memory_space,
  std::shared_ptr<compressed_device_blob> blob,
  std::vector<std::string> column_names,
  std::size_t compressed_bytes,
  std::size_t uncompressed_bytes,
  std::int64_t num_rows,
  std::shared_ptr<const per_column_byte_sizes> column_sizes)
  : simpatico_compressed_representation(memory_space),
    _blob(std::move(blob)),
    _column_names(std::move(column_names)),
    _compressed_bytes(compressed_bytes),
    _uncompressed_bytes(uncompressed_bytes),
    _pin_logical_bytes(uncompressed_bytes),
    _num_rows(num_rows),
    _column_sizes(std::move(column_sizes))
{
}

compressed_device_representation::compressed_device_representation(
  cucascade::memory::memory_space& memory_space,
  std::shared_ptr<compressed_device_blob> blob,
  std::vector<std::string> column_names,
  std::size_t compressed_bytes,
  std::size_t uncompressed_bytes,
  std::size_t pin_logical_bytes,
  std::int64_t num_rows,
  std::optional<std::vector<std::size_t>> selected_indices,
  std::shared_ptr<const per_column_byte_sizes> column_sizes)
  : simpatico_compressed_representation(memory_space),
    _blob(std::move(blob)),
    _column_names(std::move(column_names)),
    _compressed_bytes(compressed_bytes),
    _uncompressed_bytes(uncompressed_bytes),
    _pin_logical_bytes(pin_logical_bytes),
    _num_rows(num_rows),
    _selected_indices(std::move(selected_indices)),
    _column_sizes(std::move(column_sizes))
{
}

bool compressed_device_representation::has_table() const noexcept { return _blob != nullptr; }

const simpatico::compressed_table& compressed_device_representation::table() const noexcept
{
  return _blob->table;
}

std::unique_ptr<cucascade::idata_representation> compressed_device_representation::clone(
  ::cuda::stream_ref /*stream*/) const
{
  // Share the same cached blob — no byte copy needed.
  auto copy = std::unique_ptr<compressed_device_representation>(
    new compressed_device_representation(get_memory_space(),
                                         _blob,
                                         _column_names,
                                         _compressed_bytes,
                                         _uncompressed_bytes,
                                         _pin_logical_bytes,
                                         _num_rows,
                                         _selected_indices,
                                         _column_sizes));
  // The request is indexed by the selected column list, which the clone shares.
  copy->set_string_view_eligible(_string_view_eligible);
  copy->set_pushdown_scan(_pushdown_scan);
  copy->set_visibility_mask(_visibility_mask);
  return copy;
}

std::unique_ptr<compressed_device_representation> compressed_device_representation::select_columns(
  std::span<const std::size_t> indices) const
{
  auto absolute = resolve_absolute_indices(indices, _selected_indices, _column_names.size());

  auto const [compressed, uncompressed] = projected_bytes(
    absolute, _column_sizes, _column_names.size(), _compressed_bytes, _uncompressed_bytes);

  return std::unique_ptr<compressed_device_representation>(
    new compressed_device_representation(get_memory_space(),
                                         _blob,
                                         _column_names,
                                         compressed,
                                         uncompressed,
                                         _pin_logical_bytes,
                                         _num_rows,
                                         std::move(absolute),
                                         _column_sizes));
}

}  // namespace sirius
