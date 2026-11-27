// SPDX-License-Identifier: Apache-2.0
#include "api/compressed_leaf_slab.hpp"
#include "api/compressed_table_io.hpp"
#include "api/simpatico_codegen.hpp"
#include "codegen/plan/plan_interpreter.hpp"
#include "codegen/plan/plan_tree.hpp"
#include "codegen/plan/representation.hpp"
#include "codegen/selection/selection.hpp"
#include "decode/decode_session.hpp"
#include "decode_session_test_access.hpp"
#include "frame_budget_test_utils.hpp"
#include "operators/nvcomp_batched_codec.hpp"
#include "snappy_test_utils.hpp"
#include "util/chunk_selective.hpp"
#include "util/snappy_backend.hpp"

#include <cudf/copying.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/strings/utilities.hpp>
#include <cudf/unary.hpp>
#include <cudf/utilities/traits.hpp>

#include <rmm/cuda_stream.hpp>
#include <rmm/mr/cuda_async_memory_resource.hpp>
#include <rmm/mr/per_device_resource.hpp>

#include <cuda/stream>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <numeric>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

using sirius::codegen::selection_mask;

namespace {

int g_failures = 0;

#define REQUIRE_MSG(cond, ...)           \
  do {                                   \
    if (!(cond)) {                       \
      std::fprintf(stderr, "FAIL: ");    \
      std::fprintf(stderr, __VA_ARGS__); \
      std::fprintf(stderr, "\n");        \
      ++g_failures;                      \
      return false;                      \
    }                                    \
  } while (0)

constexpr std::size_t kFrameChunk = 64 * 1024;
constexpr std::array kFrameCodecs{"snappy", "ans", "lz4", "deflate"};

std::vector<simpatico::chunk_selective_stats> decode_stats;
int snappy_jobs     = 0;
int info_messages   = 0;
bool logged_enabled = false;
std::ofstream result_file;

std::uint64_t splitmix64(std::uint64_t& s)
{
  std::uint64_t z = (s += 0x9e3779b97f4a7c15ULL);
  z               = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
  z               = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
  return z ^ (z >> 31);
}

std::unique_ptr<cudf::column> upload_indices(std::vector<std::int32_t> const& indices,
                                             ::cuda::stream_ref stream)
{
  auto column = cudf::make_fixed_width_column(cudf::data_type{cudf::type_id::INT32},
                                              static_cast<cudf::size_type>(indices.size()),
                                              cudf::mask_state::UNALLOCATED,
                                              stream);
  if (!indices.empty())
    cuda_check(cudaMemcpyAsync(column->mutable_view().head<void>(),
                               indices.data(),
                               indices.size() * sizeof(std::int32_t),
                               cudaMemcpyHostToDevice,
                               stream.get()));
  stream.sync();
  return column;
}

// Owning, unsliced decoder outputs: compare every offset and char, including null rows.
std::string result_bytes(cudf::column_view col, ::cuda::stream_ref stream)
{
  std::string bytes;
  auto const rows = col.size();
  bytes.append(reinterpret_cast<char const*>(&rows), sizeof(rows));
  auto const nulls = col.null_count();
  bytes.append(reinterpret_cast<char const*>(&nulls), sizeof(nulls));
  auto const children = col.num_children();
  bytes.append(reinterpret_cast<char const*>(&children), sizeof(children));
  if (children == 0) {
    expect(rows == 0, "nonempty STRING has no offsets");
    return bytes;
  }
  for (bool valid : host_validity_bits(col))
    bytes.push_back(valid);
  cudf::strings_column_view strings(col);
  auto const offsets = strings.offsets();
  auto const type    = offsets.type().id();
  bytes.append(reinterpret_cast<char const*>(&type), sizeof(type));
  auto append_device = [&](void const* src, std::size_t size) {
    if (size == 0) return;
    auto const pos = bytes.size();
    bytes.resize(pos + size);
    cuda_check(
      cudaMemcpyAsync(bytes.data() + pos, src, size, cudaMemcpyDeviceToHost, stream.get()));
    stream.sync();
  };
  append_device(offsets.head<void>(), offsets.size() * cudf::size_of(offsets.type()));
  append_device(strings.chars_begin(stream), strings.chars_size(stream));
  return bytes;
}

simpatico::compressed_table upload(simpatico::compressed_table const& encoded,
                                   simpatico::compressed_leaf_slab& slab,
                                   ::cuda::stream_ref stream,
                                   rmm::device_async_resource_ref mr,
                                   bool one_shot_without_metadata = false)
{
  std::vector<std::uint8_t> header;
  std::vector<simpatico::payload_buffer_ref> buffers;
  std::uint64_t bytes = 0;
  auto error = simpatico::build_compressed_table_header(encoded, header, buffers, bytes, stream);
  expect(error.empty(), error);
  auto const metadata =
    one_shot_without_metadata ? nullptr : simpatico::collect_nvcomp_pin_metadata(encoded);
  std::vector<std::uint8_t> host(bytes);
  for (auto const& b : buffers) {
    if (b.size_bytes)
      cuda_check(cudaMemcpyAsync(
        host.data() + b.offset, b.device_ptr, b.size_bytes, cudaMemcpyDeviceToHost, stream.get()));
  }
  stream.sync();
  auto rebuilt = simpatico::read_uploaded_compressed_table(
    header,
    [&](std::uint64_t offset, std::size_t size, void* dst, ::cuda::stream_ref lane) {
      if (size)
        cuda_check(
          cudaMemcpyAsync(dst, host.data() + offset, size, cudaMemcpyHostToDevice, lane.get()));
    },
    std::nullopt,
    stream,
    mr,
    slab,
    &error,
    {},
    metadata.get(),
    !one_shot_without_metadata);
  stream.sync();
  expect(error.empty(), error);
  return rebuilt;
}

void check_block_errors(simpatico::nvcomp_payload_rep const& rep,
                        ::cuda::stream_ref stream,
                        rmm::device_async_resource_ref mr)
{
  namespace detail = simpatico::detail;
  std::array<std::uint32_t, 2> ids{0, 2};
  rmm::device_buffer dst(ids.size() * kFrameChunk, stream, mr);
  for (bool short_output : {false, true}) {
    detail::batched_codec_ops ops;
    ops.decompress_get_temp_size = [](auto, auto, auto* bytes, auto, auto) {
      *bytes = 0;
      return nvcompSuccess;
    };
    ops.decompress_async = [&](auto,
                               auto,
                               auto expected,
                               auto actual,
                               auto count,
                               auto,
                               auto,
                               auto,
                               auto statuses,
                               cudaStream_t lane) {
      expect(count == ids.size(), "wrong nvCOMP submission count");
      cuda_check(
        cudaMemcpyAsync(actual, expected, count * sizeof(*actual), cudaMemcpyDeviceToDevice, lane));
      static_assert(nvcompSuccess == 0);
      cuda_check(cudaMemsetAsync(statuses, 0, count * sizeof(*statuses), lane));
      if (short_output) {
        cuda_check(cudaMemsetAsync(actual + 1, 0, sizeof(*actual), lane));
      } else {
        static nvcompStatus_t const failed = nvcompErrorCannotDecompress;
        cuda_check(
          cudaMemcpyAsync(statuses + 1, &failed, sizeof(failed), cudaMemcpyHostToDevice, lane));
      }
      return nvcompSuccess;
    };
    std::array streams{stream};
    simpatico::decode_session session{streams, mr};
    auto& frame = simpatico::decode_session_test_access::frame(session);
    try {
      detail::batched_decompress_chunks(ops,
                                        rep.payload_data(),
                                        rep.payload_size(),
                                        rep.uncompressed_size,
                                        *rep.frame_layout,
                                        ids,
                                        dst.data(),
                                        frame);
    } catch (std::runtime_error const& error) {
      expect(std::strstr(error.what(), short_output ? "block 2 decoded 0" : "block 2 failed"),
             error.what());
      continue;
    }
    throw std::runtime_error("selected decode accepted invalid per-block results");
  }
}

enum class variant {
  standard,
  all_null,
  offsets_tail,
  wide_offsets,
  small_blocks,
  small_blocks_no_metadata,
  empty_strings
};

bool run_case(std::string const& codec,
              bool deep,
              bool terminal,
              bool cached,
              variant mode                      = variant::standard,
              rmm::device_async_resource_ref mr = rmm::mr::get_current_device_resource_ref())
{
  rmm::cuda_stream owned_stream(rmm::cuda_stream::flags::non_blocking);
  ::cuda::stream_ref const stream = owned_stream;
  bool const all_null             = mode == variant::all_null;
  bool const wide                 = mode == variant::wide_offsets;
  bool const offsets_tail         = mode == variant::offsets_tail;
  bool const no_metadata          = mode == variant::small_blocks_no_metadata;
  bool const small_blocks         = mode == variant::small_blocks || no_metadata;
  auto const label = codec + (deep ? "/delta" : "/bitpack") + (terminal ? "/output" : "/rep") +
                     (cached ? "/cached" : "/uncached") + (all_null ? "/all-null" : "") +
                     (wide ? "/int64" : "") + (offsets_tail ? "/snappy-offsets" : "") +
                     (small_blocks ? "/16KiB" : "") + (no_metadata ? "/no-metadata" : "") +
                     (mode == variant::empty_strings ? "/empty" : "");
  auto const block_bytes = small_blocks ? kFrameChunk / 4 : kFrameChunk;
  constexpr int n  = 2200;
  // Row 0 ends exactly at block 1. Rows 0 and 2 select blocks 0 and 2, skipping block 1.
  // The guard rows force a nonzero parent offset and a nonzero starting byte address.
  std::vector<std::string> values(n + 2);
  std::vector<bool> valid(n + 2, true);
  values.front() = "prefix";
  values.back()  = "suffix";
  std::vector<std::size_t> offsets{0};
  for (int r = 0; r < n; ++r) {
    auto& value        = values[r + 1];
    std::size_t length = r < 2 ? block_bytes : (r == 5 ? 3 * block_bytes + 9 : 1 + r % 17);
    if (r == 3 || mode == variant::empty_strings) length = 0;
    if (wide && r == 6) length = block_bytes - 1;
    for (std::size_t i = 0; i < length; ++i)
      value.push_back('a' + (7 * r + i % 23) % 26);
    valid[r + 1] = !all_null && (cached || r % 41 != 4);
    offsets.push_back(offsets.back() + (valid[r + 1] ? length : 0));
  }
  auto const chars_bytes = offsets.back();
  auto const chunks      = (chars_bytes + block_bytes - 1) / block_bytes;
  REQUIRE_MSG(chars_bytes == 0 || (chunks >= 6 && chars_bytes % block_bytes != 0),
              "[%s] expected a partial final block",
              label.c_str());
  auto input = make_strings_column(values, valid, stream);
  if (small_blocks) {
    // Keep the bytes of the spanning row in the frame while making that row NULL.
    constexpr int spanning_row = 5 + 1;  // Prefix guard precedes the slice.
    auto const nulls           = input->null_count();
    cudf::set_null_mask(
      input->mutable_view().null_mask(), spanning_row, spanning_row + 1, false, stream);
    input->set_null_count(nulls + 1);
    valid[spanning_row] = false;
  }
  if (wide) {
    auto const nulls = input->null_count();
    auto parts       = input->release();
    auto offsets64 =
      cudf::cast(parts.children.front()->view(), cudf::data_type{cudf::type_id::INT64}, stream, mr);
    input = cudf::make_strings_column(
      n + 2, std::move(offsets64), std::move(*parts.data), nulls, std::move(*parts.null_mask));
  }
  auto const sliced = cudf::slice(input->view(), {1, n + 1}, stream).front();
  std::string dsl   = "input -> str_split -> offsets, chars\nstr_split.chars -> " + codec +
                      (terminal ? " -> output\n" : "\n");
  dsl += deep ? "str_split.offsets -> delta -> differences\nstr_split.offsets.differences"
              : "str_split.offsets";
  dsl += " -> bitpack -> chunk_min, chunk_count, chunk_bits, packed\n";
  if (offsets_tail) dsl += "str_split.offsets.differences.packed -> snappy\n";
  std::vector<std::int32_t> keys(n);
  std::iota(keys.begin(), keys.end(), 0);
  auto key_column = upload_indices(keys, stream);
  auto encoded    = simpatico::compress_with_plan(
    cudf::table_view{{key_column->view(), sliced}}, "input -> bitpack\n---\n" + dsl, stream, mr);
  auto& nodes = encoded.columns.at(1).plan_tree->nodes;
  auto it =
    std::find_if(nodes.begin(), nodes.end(), [&](auto const& node) { return node.op == codec; });
  REQUIRE_MSG(it != nodes.end(), "[%s] missing codec node", label.c_str());
  auto const nid = std::distance(nodes.begin(), it);
  auto const kind = simpatico::op_id_from_name(codec).value();
  auto* metadata  = simpatico::batched_meta(kind, it->meta);
  REQUIRE_MSG(metadata, "[%s] missing codec metadata", label.c_str());
  if (small_blocks) {
    namespace detail = simpatico::detail;
    auto& rep        = dynamic_cast<simpatico::nvcomp_payload_rep&>(*it->rep);
    auto ops         = frame_test_ops(codec);
    ops.chunk_size   = block_bytes;
    REQUIRE_MSG(
      rep.uncompressed_size == chars_bytes, "[%s] missing physical NULL bytes", label.c_str());
    rmm::device_buffer source(
      cudf::strings_column_view(input->view()).chars_begin(stream) + values.front().size(),
      chars_bytes,
      stream,
      mr);
    auto [bytes, layout] =
      detail::batched_compress_bytes(ops, source.data(), chars_bytes, stream, mr);
    rep.channels_[0] =
      std::make_unique<cudf::column>(cudf::data_type{cudf::type_id::UINT8},
                                     static_cast<cudf::size_type>(layout.frame_bytes),
                                     std::move(*bytes),
                                     rmm::device_buffer{},
                                     0);
    metadata->frame_layout = rep.frame_layout =
      std::make_shared<detail::nvcomp_frame_layout>(std::move(layout));
  }
  auto const before_upload =
    simpatico::selected_string_scratch_bytes(*encoded.columns[1].plan_tree, n);
  bool const enabled = simpatico::chunk_selective_enabled();
  auto const expected_budget =
    enabled ? frame_budget(n, chars_bytes, stream, codec, block_bytes) : 0;
  REQUIRE_MSG(
    before_upload == expected_budget, "[%s] wrong compressed frame budget", label.c_str());
  if (!cached) metadata->frame_layout.reset();

  simpatico::compressed_leaf_slab slab;
  auto rebuilt     = upload(encoded, slab, stream, mr, no_metadata);
  auto& plan       = *rebuilt.columns.at(1).plan_tree;
  auto& node       = plan.nodes.at(nid);
  auto* payload    = dynamic_cast<simpatico::nvcomp_payload_rep*>(node.rep.get());
  auto const* meta = simpatico::batched_meta(kind, node.meta);
  REQUIRE_MSG(meta, "[%s] missing uploaded codec metadata", label.c_str());
  REQUIRE_MSG(terminal ? !node.rep && simpatico::terminal_identity_channel(node, "output")
                       : payload && node.output_names.empty(),
              "[%s] wrong storage after upload",
              label.c_str());
  auto const& layout = payload ? payload->frame_layout : meta->frame_layout;
  REQUIRE_MSG(static_cast<bool>(layout) == (cached && chunks > 0) &&
                (!layout ||
                 (layout->chunk_bytes == block_bytes && layout->compressed_bytes.size() == chunks)),
              "[%s] wrong layout after upload",
              label.c_str());

  if (no_metadata) {
    REQUIRE_MSG(plan.framed_chars_chunk_bytes == 0,
                "[%s] one-shot upload read the frame header",
                label.c_str());
  } else if (enabled) {
    REQUIRE_MSG(plan.framed_chars_chunk_bytes == (chars_bytes ? block_bytes : 0) &&
                  expected_budget == simpatico::selected_string_scratch_bytes(plan, n),
                "[%s] frame geometry or admission estimate changed after upload",
                label.c_str());
  }
  using sirius::codegen::decode_route;
  auto const caps = simpatico::probe_column(plan);
  REQUIRE_MSG(caps.compact_route == (enabled ? decode_route::str_split : decode_route::full),
              "[%s] unexpected route or frame reservation",
              label.c_str());
  if (enabled && deep && payload && layout) {
    check_block_errors(*payload, stream, mr);
    auto const type        = payload->original_type;
    payload->original_type = cudf::data_type{cudf::type_id::UINT32};
    REQUIRE_MSG(simpatico::probe_column(plan).compact_route == decode_route::full,
                "[%s] widened chars carrier accepted",
                label.c_str());
    payload->original_type = type;
  }
  auto full = simpatico::decode_one(simpatico::column_decode_request{std::cref(plan)}, stream, mr);
  stream.sync();
  if (wide)
    REQUIRE_MSG(full->view().child(0).type().id() == cudf::type_id::INT64,
                "[%s] expected full decode to preserve INT64 input offsets",
                label.c_str());
  REQUIRE_MSG(strings_equal(full->view(), sliced, stream),
              "[%s] full decode differs from sliced input",
              label.c_str());

  std::vector<std::int32_t> every(n);
  std::iota(every.begin(), every.end(), 0);
  std::vector<std::int32_t> random;
  std::uint64_t seed = 0x5eed;
  for (auto r : every)
    if (splitmix64(seed) % 100 < 7) random.push_back(r);
  using pattern = std::pair<char const*, std::vector<std::int32_t>>;
  for (auto const& [name, indices] : std::vector<pattern>{{"zero", {}},
                                                          {"exact-end", {0}},
                                                          {"below-boundary", {6}},
                                                          {"nonadjacent", {0, 2}},
                                                          {"spanning", {5}},
                                                          {"final-partial", {n - 1}},
                                                          {"empty", {3}},
                                                          {"nullable", {3, 4, 45}},
                                                          {"every", every},
                                                          {"random", random}}) {
    std::vector<std::uint32_t> words(selection_mask::WordsFor(n), 0);
    for (auto r : indices)
      words[r / 32] |= 1u << (r % 32);
    auto const survivors = static_cast<std::int64_t>(indices.size());
    std::set<std::size_t> marked;
    for (auto const r : indices)
      if (valid[r + 1] && offsets[r + 1] > offsets[r])
        for (auto c = static_cast<std::size_t>(offsets[r]) / block_bytes;
             c <= static_cast<std::size_t>(offsets[r + 1] - 1) / block_bytes;
             ++c)
          marked.insert(c);
    std::size_t marked_bytes = 0;
    for (auto const c : marked)
      marked_bytes += std::min(block_bytes, chars_bytes - c * block_bytes);

    auto index_col = upload_indices(indices, stream);
    auto gathered = cudf::gather(cudf::table_view{{full->view()}},
                                 index_col->view(),
                                 cudf::out_of_bounds_policy::DONT_CHECK,
                                 stream,
                                 mr);
    stream.sync();
    auto reference = result_bytes(gathered->view().column(0), stream);
    namespace sc   = sirius::codegen;
    sc::scan_filter_request request;
    request.filters.push_back({1, {0, n - 1}});
    request.routes           = {caps.compact_route, decode_route::bitpack_mask};
    request.keep_mask_words  = words.data();
    request.keep_mask_rows   = n;
    request.delivered_prefix = 1;
    sc::scan_filter_result result;
    decode_stats.clear();
    std::array<std::size_t, 2> projection{1, 0};
    simpatico::snappy_backend_log_scope const observe_offsets{
      [](bool info, char const*, std::size_t, std::size_t) {
        if (!info) ++snappy_jobs;
      }};
    snappy_jobs   = 0;
    auto selected = simpatico::decompress_scan_filter(
      rebuilt, projection, request, result, std::array{stream}, stream, mr);
    if (offsets_tail)
      REQUIRE_MSG((snappy_jobs > 0) == (survivors > 0),
                  "[%s] unexpected offsets decode (%s)",
                  label.c_str(),
                  name);
    if (wide && indices.size() == 1 && (indices.front() == 0 || indices.front() == 6)) {
      auto const expected_width =
        indices.front() == 0 ? cudf::type_id::INT64 : cudf::type_id::INT32;
      REQUIRE_MSG(gathered->view().column(0).child(0).type().id() == expected_width,
                  "[%s] gather did not exercise the promotion boundary (%s)",
                  label.c_str(),
                  name);
    }
    bool const should_apply = enabled || (plan.validity.empty() && survivors <= n / 10);
    REQUIRE_MSG(
      result.applied == should_apply, "[%s] unexpected scan decision (%s)", label.c_str(), name);
    if (should_apply) {
      REQUIRE_MSG(result.routes == std::vector<decode_route>{caps.compact_route} &&
                    result.survivor_count == survivors && result.keep_mask_applied,
                  "[%s] unexpected scan route (%s)",
                  label.c_str(),
                  name);
      REQUIRE_MSG(result.zero_skipped_full_columns == (!enabled && survivors == 0 ? 1u : 0u),
                  "[%s] unexpected zero-skip count (%s)",
                  label.c_str(),
                  name);
    } else {
      REQUIRE_MSG(
        result.status == (plan.validity.empty() ? sc::scan_filter_status::declined_unselective
                                                : sc::scan_filter_status::refused),
        "[%s] unexpected full-decode fallback (%s)",
        label.c_str(),
        name);
      selected = cudf::gather(
        selected->view(), index_col->view(), cudf::out_of_bounds_policy::DONT_CHECK, stream, mr);
    }
    stream.sync();
    auto actual = result_bytes(selected->view().column(0), stream);
    REQUIRE_MSG(actual == reference,
                "[%s] selected bytes, offsets or validity differ (%s)",
                label.c_str(),
                name);
    if (result_file.is_open()) result_file.write(actual.data(), actual.size());
    if (!enabled) {
      REQUIRE_MSG(
        decode_stats.empty(), "[%s] switch-off ran a chunk-selective decode", label.c_str());
      continue;
    }
    REQUIRE_MSG(
      decode_stats.size() == 1 && decode_stats[0].column_index == 1 && decode_stats[0].codec &&
        codec == decode_stats[0].codec && decode_stats[0].rows == n &&
        decode_stats[0].survivors == survivors && decode_stats[0].marked_chunks == marked.size() &&
        decode_stats[0].total_chunks == chunks && decode_stats[0].decoded_bytes == marked_bytes,
      "[%s] pattern %s: chunk accounting differs from the host oracle "
      "(marked %u/%u, oracle %zu/%zu)",
      label.c_str(),
      name,
      decode_stats.empty() ? 0u : decode_stats[0].marked_chunks,
      decode_stats.empty() ? 0u : decode_stats[0].total_chunks,
      marked.size(),
      chunks);
  }
  std::printf("PASS: chunk-selective %s (enabled=%d)\n", label.c_str(), enabled);
  return true;
}

}  // namespace

int main(int argc, char** argv)
{
  if (argc == 2 && std::strcmp(argv[1], "--invalid") == 0) {
    try {
      (void)simpatico::chunk_selective_enabled();
    } catch (std::invalid_argument const& e) {
      if (std::strstr(e.what(), "SIRIUS_EXP_CHUNK_SELECTIVE")) {
        std::puts("PASS: invalid chunk-selective switch rejected");
        return 0;
      }
    }
    std::fputs("FAIL: invalid chunk-selective switch accepted\n", stderr);
    return 1;
  }
  simpatico::chunk_selective_log_scope const logging{
    [](bool info, bool enabled, simpatico::chunk_selective_stats const& stats) {
      if (info) {
        ++info_messages;
        logged_enabled = enabled;
      } else
        decode_stats.push_back(stats);
    }};
  auto const* value   = std::getenv("SIRIUS_EXP_CHUNK_SELECTIVE");
  bool const expected = value && std::strcmp(value, "1") == 0;
  bool const enabled  = simpatico::chunk_selective_enabled();
  setenv("SIRIUS_EXP_CHUNK_SELECTIVE", "invalid-after-first-use", 1);
  if (enabled != expected || simpatico::chunk_selective_enabled() != enabled ||
      info_messages != static_cast<int>(enabled) || logged_enabled != enabled) {
    std::fputs("FAIL: chunk-selective unexpected switch value or log count\n", stderr);
    return 1;
  }

  int device_count = 0;
  auto const status = cudaGetDeviceCount(&device_count);
  if (status == cudaErrorNoDevice || (status == cudaSuccess && device_count == 0)) {
    std::puts("SKIP: no CUDA device");
    return 77;
  }
  if (status != cudaSuccess) {
    std::fprintf(stderr, "FAIL: cudaGetDeviceCount: %s\n", cudaGetErrorString(status));
    return 1;
  }
  if (cudaSetDevice(0) != cudaSuccess) {
    std::fprintf(stderr, "FAIL: cudaSetDevice(0) failed\n");
    return 1;
  }
  try {
    if (argc == 3 && std::strcmp(argv[1], "--write-probe") == 0) {
      auto stream = cudf::get_default_stream();
      auto input  = make_strings_table({"x", "miss"}, {true, true}, stream);
      auto table  = simpatico::compress_with_plan(
        input->view(),
        "input -> str_split -> offsets, chars\n"
        "str_split.chars -> snappy\n"
        "str_split.offsets -> delta -> differences\n"
        "str_split.offsets.differences -> bitpack -> chunk_min, chunk_count, chunk_bits, packed\n",
        stream,
        rmm::mr::get_current_device_resource_ref());
      auto error = simpatico::write_compressed_table(table, argv[2], stream);
      expect(error.empty(), error);
      return 0;
    }
    if (argc == 3 && std::strcmp(argv[1], "--probe-off") == 0) {
      auto stream = cudf::get_default_stream();
      auto mr     = rmm::mr::get_current_device_resource_ref();
      std::string error;
      auto table = simpatico::read_compressed_table(argv[2], stream, mr, &error);
      expect(error.empty(), error);
      simpatico::compressed_leaf_slab slab;
      auto rebuilt     = upload(table, slab, stream, mr);
      auto const& plan = *rebuilt.columns.at(0).plan_tree;
      auto const node  = std::find_if(
        plan.nodes.begin(), plan.nodes.end(), [](auto const& n) { return n.op == "snappy"; });
      expect(node != plan.nodes.end(), "missing uploaded Snappy node");
      auto const* rep = dynamic_cast<simpatico::nvcomp_payload_rep const*>(node->rep.get());
      expect(rep && rep->payload_size() && rep->payload_data() && rep->uncompressed_size == 5,
             "missing uploaded Snappy frame");
      expect(!enabled &&
               simpatico::probe_column(plan).compact_route == sirius::codegen::decode_route::full,
             "switch-off admitted a framed string");
      return 0;
    }
    if (argc == 3 && std::strcmp(argv[1], "--record") == 0) {
      result_file.exceptions(std::ios::failbit | std::ios::badbit);
      result_file.open(argv[2], std::ios::binary | std::ios::trunc);
    }
    setenv("SIRIUS_EXP_FUSED_SCAN_FILTER", "1", 1);
    setenv("SIRIUS_EXP_FUSED_SCAN_MAX_SEL", "1", 1);
    setenv("SIRIUS_EXP_FUSED_SCAN_TIERB_MAX_SEL", "0.10", 1);
    setenv("SIRIUS_EXP_FUSED_SCAN_ZERO_SKIP", "1", 1);
    if (argc == 2 && std::strcmp(argv[1], "--wide") == 0) {
      expect(cudf::strings::is_large_strings_enabled() &&
               cudf::strings::get_offset64_threshold() == kFrameChunk,
             "large-string threshold fixture misconfigured");
      for (auto const* codec : kFrameCodecs)
        run_case(codec, true, false, true, variant::wide_offsets);
      return g_failures ? 1 : 0;
    }
    if (argc == 2 && std::strcmp(argv[1], "--hardware") == 0) {
      if (!snappy_test_device_has_de()) return 77;
      rmm::mr::cuda_async_memory_resource pool;
      run_case(
        "snappy", true, false, true, variant::standard, rmm::device_async_resource_ref{pool});
      return g_failures ? 1 : 0;
    }
    for (auto const* codec : {"ans", "lz4", "deflate"})
      run_case(codec, true, false, true, variant::offsets_tail);
    for (auto const* codec : kFrameCodecs) {
      for (bool cached : {false, true}) {
        run_case(codec, true, false, cached);
        run_case(codec, true, true, cached);
      }
      run_case(codec, false, false, true);
      run_case(codec, true, false, false, variant::all_null);
      run_case(codec, true, false, false, variant::small_blocks);
      run_case(codec, true, false, false, variant::small_blocks_no_metadata);
      run_case(codec, true, false, true, variant::empty_strings);
    }
    if (result_file.is_open()) result_file.close();
  } catch (std::exception const& e) {
    std::fprintf(stderr, "FAIL: unhandled exception: %s\n", e.what());
    return 1;
  }
  if (g_failures > 0) {
    std::fprintf(stderr, "%d check(s) failed\n", g_failures);
    return 1;
  }
  std::printf("ALL PASS\n");
  return 0;
}
