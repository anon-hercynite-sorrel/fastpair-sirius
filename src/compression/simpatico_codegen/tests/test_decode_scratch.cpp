// SPDX-License-Identifier: Apache-2.0
#include "api/simpatico_codegen.hpp"
#include "codegen/plan/plan_tree.hpp"
#include "codegen/selection/decompression_pushdown_policy.hpp"
#include "decode/decode_scratch_resource.hpp"
#include "test_utils.hpp"

#include <rmm/cuda_stream.hpp>
#include <rmm/mr/cuda_async_memory_resource.hpp>

#include <array>
#include <cstdio>
#include <fstream>
#include <limits>
#include <numeric>
#include <string>

namespace {
namespace sc = sirius::codegen;

class current_resource_scope {
 public:
  explicit current_resource_scope(rmm::device_async_resource_ref mr)
    : previous_(
        cudf::set_current_device_resource(cuda::mr::any_resource<cuda::mr::device_accessible>{mr}))
  {
  }
  ~current_resource_scope() { cudf::set_current_device_resource(std::move(previous_)); }

 private:
  cuda::mr::any_resource<cuda::mr::device_accessible> previous_;
};

class counting_resource {
 public:
  struct counts {
    std::size_t allocations = 0, frees = 0, live = 0, peak = 0, failures = 0;
    bool non_oom_failure = false;
    std::size_t limit    = std::numeric_limits<std::size_t>::max();
  };
  explicit counting_resource(rmm::device_async_resource_ref upstream) : upstream_(upstream) {}
  std::shared_ptr<counts> observed = std::make_shared<counts>();
  void* allocate(cuda::stream_ref stream, std::size_t bytes, std::size_t alignment)
  {
    if (observed->non_oom_failure) {
      ++observed->failures;
      throw rmm::bad_alloc("decode scratch non-OOM sentinel");
    }
    if (bytes > observed->limit - observed->live) {
      ++observed->failures;
      throw rmm::out_of_memory("decode scratch test allocation limit");
    }
    auto* pointer = upstream_.allocate(stream, bytes, alignment);
    if (pointer && bytes) {
      ++observed->allocations;
      observed->live += bytes;
      observed->peak = std::max(observed->peak, observed->live);
    }
    return pointer;
  }
  void deallocate(cuda::stream_ref stream,
                  void* pointer,
                  std::size_t bytes,
                  std::size_t alignment) noexcept
  {
    if (pointer && bytes) {
      ++observed->frees;
      observed->live -= bytes;
    }
    upstream_.deallocate(stream, pointer, bytes, alignment);
  }
  void* allocate_sync(std::size_t bytes, std::size_t alignment)
  {
    auto* pointer = allocate(cuda::stream_ref{cudaStream_t{nullptr}}, bytes, alignment);
    cuda_check(cudaStreamSynchronize(nullptr));
    return pointer;
  }
  void deallocate_sync(void* pointer, std::size_t bytes, std::size_t alignment) noexcept
  {
    deallocate(cuda::stream_ref{cudaStream_t{nullptr}}, pointer, bytes, alignment);
    (void)cudaStreamSynchronize(nullptr);
  }
  bool operator==(counting_resource const& other) const noexcept
  { return observed == other.observed; }
  friend void get_property(counting_resource const&, cuda::mr::device_accessible) noexcept {}

 private:
  rmm::device_async_resource_ref upstream_;
};

void write_buffer(std::ostream& out,
                  rmm::device_buffer& buffer,
                  rmm::device_async_resource_ref mr,
                  cuda::stream_ref stream)
{
  auto const bytes = buffer.size();
  out.write(reinterpret_cast<char const*>(&bytes), sizeof(bytes));
  if (bytes == 0) return;
  expect(buffer.memory_resource() == mr, "escaped buffer retains the caller's resource");
  std::vector<char> host(bytes);
  cuda_check(
    cudaMemcpyAsync(host.data(), buffer.data(), bytes, cudaMemcpyDeviceToHost, stream.get()));
  stream.sync();
  out.write(host.data(), host.size());
}

void write_column(std::ostream& out,
                  cudf::column& column,
                  rmm::device_async_resource_ref mr,
                  cuda::stream_ref stream)
{
  auto const type = column.type().id();
  auto const rows = column.size();
  out.write(reinterpret_cast<char const*>(&type), sizeof(type));
  out.write(reinterpret_cast<char const*>(&rows), sizeof(rows));
  auto contents = column.release();
  if (contents.data) write_buffer(out, *contents.data, mr, stream);
  if (contents.null_mask) write_buffer(out, *contents.null_mask, mr, stream);
  for (auto& child : contents.children)
    write_column(out, *child, mr, stream);
}

void arena_ordering(rmm::device_async_resource_ref upstream,
                    std::span<const cuda::stream_ref> lanes)
{
  counting_resource counted(upstream);
  {
    auto const before = simpatico::decode_scratch_resource::statistics();
    simpatico::decode_scratch_resource arena(counted, lanes, 4099);
    expect(arena.backed_lane_count() == 0, "supplied-but-unused lanes have no backing storage");
    expect(counted.observed->allocations == 0, "unused lanes allocate no backing");
    rmm::device_buffer first(1024, lanes.front(), arena);
    rmm::device_buffer second(1024, lanes.front(), arena);
    expect(counted.observed->allocations == 1, "two scratch buffers share one backing allocation");
    expect(arena.backed_lane_count() == 1, "only the allocating lane holds backing storage");
    cuda_check(cudaMemsetAsync(first.data(), 7, first.size(), lanes.front().get()));
    auto ready = lanes.front().record_event();
    lanes.back().wait(ready);
    cuda_check(cudaMemsetAsync(second.data(), 9, second.size(), lanes.back().get()));
    auto* reused = second.data();
    second.set_stream(lanes.back());
    second = rmm::device_buffer{};
    expect(simpatico::decode_scratch_resource::statistics().cross_stream_retirements -
               before.cross_stream_retirements ==
             (lanes.front() == lanes.back() ? 0 : 1),
           "only the deliberate cross-stream free takes the event-ordered fallback");
    rmm::device_buffer third(1024, lanes.front(), arena);
    expect(third.data() == reused, "freed slice is reused after its consumer stream");
    cuda_check(cudaMemsetAsync(third.data(), 11, third.size(), lanes.front().get()));
    std::array<unsigned char, 1024> bytes;
    cuda_check(cudaMemcpyAsync(
      bytes.data(), third.data(), bytes.size(), cudaMemcpyDeviceToHost, lanes.front().get()));
    lanes.front().sync();
    expect(std::all_of(bytes.begin(), bytes.end(), [](auto byte) { return byte == 11; }),
           "cross-stream retirement precedes reuse");
    rmm::device_buffer oversized(
      simpatico::decode_scratch_resource::max_bytes + 1, lanes.front(), arena);
    expect(counted.observed->allocations == 2, "oversized scratch goes directly upstream");
    expect(arena.backed_lane_count() == 1, "reuse and cross-stream consumers add no backing lanes");
    std::size_t used_lanes = 1;
    for (auto lane : lanes.subspan(1)) {
      rmm::device_buffer small(1024, lane, arena);
      expect(arena.backed_lane_count() == ++used_lanes,
             "backing storage is engaged once per allocating lane");
    }
  }
  expect(counted.observed->live == 0, "arena releases backing and direct allocations");
  counted.observed->limit = 2048;
  {
    simpatico::decode_scratch_resource arena(counted, lanes, 4099);
    rmm::device_buffer small(1024, lanes.front(), arena);
    rmm::device_buffer another(1024, lanes.front(), arena);
    expect(counted.observed->failures == 1,
           "refused backing disables the lane; later requests go directly upstream");
  }
  expect(counted.observed->live == 0, "pressure fallback releases its allocation");
  counted.observed->non_oom_failure = true;
  auto const failures               = counted.observed->failures;
  bool propagated                   = false;
  try {
    simpatico::decode_scratch_resource arena(counted, lanes, 4099);
    rmm::device_buffer small(1024, lanes.front(), arena);
  } catch (rmm::bad_alloc const& e) {
    propagated = std::string(e.what()).find("non-OOM sentinel") != std::string::npos;
  }
  expect(propagated && counted.observed->failures == failures + 1,
         "non-OOM allocation errors propagate without retry");
}

template <typename T>
std::unique_ptr<cudf::column> upload(std::vector<T> const& values,
                                     cudf::type_id type,
                                     cuda::stream_ref stream,
                                     rmm::device_async_resource_ref mr)
{
  rmm::device_buffer data(values.data(), values.size() * sizeof(T), stream, mr);
  stream.sync();
  return std::make_unique<cudf::column>(
    cudf::data_type{type}, values.size(), std::move(data), rmm::device_buffer{}, 0);
}

// A valid deferred DAG where decoding row-offset differences also materializes dict_offsets.
// Use exactly one 1024-element delta block so the two bitjoin inputs have matching lengths.
// All 1023 dictionary entries spell "a"; only code zero is used. This keeps the fixture independent
// of the CPU encoder's dictionary training and reproduces the preparation ownership bug exactly.
simpatico::compressed_table shared_preparation(cuda::stream_ref stream,
                                               rmm::device_async_resource_ref mr)
{
  using namespace simpatico;
  auto tree = plan_tree_from_dsl(
    "input -> onpair_16 -> codes, row_offsets, dict_bytes, dict_offsets, batch_prefixes\n"
    "onpair_16.row_offsets -> delta -> differences\n"
    "onpair_16.row_offsets.differences_31:0, onpair_16.dict_offsets_31:0 -> bitjoin_u64\n");
  expect(tree.has_value(), "parse shared preparation DAG");
  std::vector<std::int32_t> offsets(1024);
  std::iota(offsets.begin(), offsets.end(), 0);
  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(upload(offsets, cudf::type_id::INT32, stream, mr));
  cudf::table input(std::move(columns));
  auto delta = compress_with_plan(input.view(), "input -> delta -> differences\n", stream, mr);
  auto& delta_node = delta.columns[0].plan_tree->nodes[1];
  auto& target     = tree->nodes[2];
  target.rep       = std::move(delta_node.rep);
  auto raw         = std::move(delta_node.channels.begin()->second);
  auto* fused      = dynamic_cast<codegen_fused_representation*>(raw.get());
  expect(fused != nullptr, "delta raw tail is fused storage");
  std::vector<std::uint32_t> differences;
  for (auto& [name, column] : fused->buffers) {
    if (name != "data") continue;
    differences.resize(column->size());
    cuda_check(cudaMemcpy(differences.data(),
                          column->view().head<void>(),
                          differences.size() * sizeof(std::uint32_t),
                          cudaMemcpyDeviceToHost));
    column.reset();
  }
  std::erase_if(fused->buffers, [](auto const& entry) { return !entry.second; });
  target.channels.emplace(target.output_paths[0], std::move(raw));
  expect(differences.size() == offsets.size(), "bitjoin inputs have matching lengths");
  std::vector<std::uint64_t> packed(offsets.size());
  for (std::size_t i = 0; i < packed.size(); ++i)
    // bitjoin places its first input in the most significant bits (bitjoin_layout.cpp).
    packed[i] = (static_cast<std::uint64_t>(differences[i]) << 32) | i;
  tree->nodes[3].rep = std::make_unique<identity_compressed_representation>(
    upload(packed, cudf::type_id::UINT64, stream, mr));
  auto& root = tree->nodes[1];
  root.meta  = leaf_meta::onpair{1023, 1023, 16, static_cast<std::int32_t>(cudf::type_id::STRING)};
  auto terminal = [&](std::size_t port, std::unique_ptr<cudf::column> column) {
    root.channels.emplace(root.output_paths[port],
                          std::make_unique<identity_compressed_representation>(std::move(column)));
  };
  terminal(0, upload(std::vector<std::uint16_t>(1023, 0), cudf::type_id::UINT16, stream, mr));
  terminal(2, upload(std::vector<std::uint8_t>(1023, 'a'), cudf::type_id::UINT8, stream, mr));
  terminal(4,
           upload(std::vector<std::uint64_t>{0, 128, 256, 384, 512, 640, 768, 896, 1023},
                  cudf::type_id::UINT64,
                  stream,
                  mr));
  compressed_table table;
  table.columns.push_back({{},
                           cudf::data_type{cudf::type_id::STRING},
                           1023,
                           std::make_unique<PlanTree>(std::move(*tree))});
  return table;
}

std::vector<std::string> url_rows()
{
  std::vector<std::string> rows;
  rows.reserve(1'000'003);
  // Unique paths/query strings, varying lengths and domains; one in ten contains the LIKE literal.
  for (std::size_t i = 0; i < 1'000'003; ++i) {
    auto const hash = (i * 2654435761ULL) ^ (i >> 7);
    rows.push_back("https://" + std::string(i % 10 == 0 ? "google" : "example") +
                   std::to_string(i % 97) + ".org/" + std::string(i % 31, 'a' + i % 26) +
                   "/article/" + std::to_string(i) + "?session=" + std::to_string(hash) +
                   "&page=" + std::to_string(i % 1009));
  }
  return rows;
}
}  // namespace

int main(int argc, char** argv)
{
  try {
    expect(argc == 4 || argc == 5, "usage: test_decode_scratch MODE OUTPUT LANES [BYTE_LIMIT]");
    cuda_check(cudaSetDevice(0));
    rmm::cuda_stream first, second, third, fourth;
    std::array<cuda::stream_ref, 4> all_lanes{first, second, third, fourth};
    auto const lanes  = std::span{all_lanes}.first(std::stoul(argv[3]));
    auto const stream = lanes.front();
    rmm::mr::cuda_async_memory_resource upstream;
    auto const mr = rmm::device_async_resource_ref{upstream};
    arena_ordering(mr, lanes);

    std::string const mode = argv[1];
    bool const large       = mode.starts_with("url_");
    bool const deferred    = mode == "deferred" || mode == "shared_deferred";
    bool const capacity    = mode == "capacity_projection";
    bool const plain = mode == "projection" || mode == "url_projection" || capacity || deferred;
    bool const above = mode == "above";
    bool const zero  = mode == "zero";
    // INT32 reconstructed offsets fill exactly one lane at these row counts.
    auto const row_count =
      capacity
        ? simpatico::decode_scratch_resource::max_bytes / lanes.size() / sizeof(std::int32_t) - 1
        : 4099;
    auto rows =
      large ? url_rows() : std::vector<std::string>(row_count, std::string(128, 'x') + " other");
    if (!large) {
      rows[3]    = "google";
      rows[4000] = "google-x";
    }
    if (mode == "shared_deferred") rows.assign(1023, "a");
    auto input = make_strings_table(rows, {}, stream);
    auto stored =
      mode == "shared_deferred"
        ? shared_preparation(stream, mr)
        : simpatico::compress_with_plan(
            input->view(),
            "input -> onpair_16 -> codes, row_offsets, dict_bytes, dict_offsets, batch_prefixes\n"
            "onpair_16.row_offsets -> delta -> differences\n"
            "onpair_16.row_offsets.differences -> bitpack\n",
            stream,
            mr);
    if (deferred)
      for (auto& node : stored.columns[0].plan_tree->nodes)
        node.prepared_rep.reset();
    stream.sync();
    auto ready = stream.record_event();
    for (auto lane : lanes)
      lane.wait(ready);

    counting_resource counted(mr);
    if (argc == 5) counted.observed->limit = std::stoull(argv[4]);
    auto const decode_mr = rmm::device_async_resource_ref{counted};
    // A single boundary projection must save allocations without relying on reuse across columns.
    // Other cases repeat projections to exercise lanes without compressing identical copies.
    std::vector<std::size_t> selected(capacity ? 1 : large ? 4 : 2, 0);
    sc::scan_filter_request request;
    request.routes.assign(selected.size(),
                          mode == "full" ? sc::decode_route::full : sc::decode_route::onpair);
    if (!plain) {
      request.like_filters = {{0, {{zero ? "absent" : above ? "x" : "google"}, false}}};
      if (above) {
        request.like_filters[0].column = 1;
        request.delivered_prefix       = 1;
        request.like_bool8_columns     = {1};
      } else if (!large) {
        request.like_filters.push_back({1, {{zero ? "absent" : "google"}, false}});
      }
    }
    sc::scan_filter_result result;
    std::unique_ptr<cudf::table> output;
    auto const arena_before = simpatico::decode_scratch_resource::statistics();
    {
      current_resource_scope current{decode_mr};
      output = simpatico::decompress_scan_filter(
        stored, selected, request, result, lanes, stream, decode_mr);
    }
    auto const arena_after = simpatico::decode_scratch_resource::statistics();
    auto const retirements =
      arena_after.cross_stream_retirements - arena_before.cross_stream_retirements;
    expect(retirements == 0, "scan paths never retire arena slices on another stream");
    expect(arena_after.live_backing_bytes == arena_before.live_backing_bytes,
           "no arena backing remains after the call, while output and preparation are live");
    stream.sync();
    auto const allocations = counted.observed->allocations;
    auto const peak        = counted.observed->peak;
    expect(result.applied != plain && result.like_mask_delivered == above,
           "the requested plain/compacted/BOOL8 path ran");
    std::vector<std::string> expected_rows;
    for (auto const& row : rows)
      if (plain || above || (!zero && row.find("google") != std::string::npos))
        expected_rows.push_back(row);
    if (large && !plain)
      expect(result.survivor_count == 100001, "URL LIKE retains approximately ten percent");
    auto expected = make_strings_column(expected_rows, {}, stream);
    expect(output->num_columns() == selected.size(), "all requested columns delivered");
    for (cudf::size_type i = 0; i < output->num_columns(); ++i) {
      if (above && i == 1) {
        std::vector<std::uint8_t> answer(rows.size());
        cuda_check(cudaMemcpy(answer.data(),
                              output->view().column(i).head<void>(),
                              answer.size(),
                              cudaMemcpyDeviceToHost));
        for (std::size_t row = 0; row < rows.size(); ++row)
          expect(answer[row] == (rows[row].find('x') != std::string::npos), "BOOL8 LIKE values");
      } else {
        expect(columns_equal_any(output->view().column(i), expected->view(), stream),
               "decoded strings match expected rows byte for byte");
      }
    }
    std::ofstream bytes(argv[2], std::ios::binary);
    expect(bytes.good(), "open result file");
    for (auto& column : output->release())
      write_column(bytes, *column, decode_mr, stream);
    write_buffer(bytes, result.mask_words, decode_mr, stream);
    write_buffer(bytes, result.chunk_offsets, decode_mr, stream);
    write_buffer(bytes, result.row_indices, decode_mr, stream);
    result = {};
    // Deferred preparation escapes into the plan and must remain usable after arena destruction.
    if (deferred) {
      auto again = simpatico::decompress(stored, stream, mr);
      expect(columns_equal_any(again->view().column(0), input->view().column(0), stream),
             "published preparation outlives the decode arena");
      // Inspect owned preparation channels before dropping them; a memoized dict_offsets
      // sibling must use the caller MR even though row offsets used scratch in the same walk.
      for (auto& node : stored.columns[0].plan_tree->nodes) {
        if (node.prepared_rep)
          for (auto& channel : node.prepared_rep->channels_)
            if (channel) write_column(bytes, *channel, decode_mr, stream);
        node.prepared_rep.reset();
      }
    }
    expect(counted.observed->live == 0 && counted.observed->allocations == counted.observed->frees,
           "all measured allocations freed through the live caller resource");
    expect(bytes.good(), "write result file");
    std::printf("allocations=%zu peak=%zu retirements=%zu backing=%zu frees=%zu failures=%zu\n",
                allocations,
                peak,
                retirements,
                arena_after.live_backing_bytes - arena_before.live_backing_bytes,
                counted.observed->frees,
                counted.observed->failures);
    return 0;
  } catch (std::exception const& e) {
    std::fprintf(stderr, "test_decode_scratch: FAIL: %s\n", e.what());
    return 1;
  }
}
