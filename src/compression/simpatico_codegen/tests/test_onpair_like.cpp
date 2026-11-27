// SPDX-License-Identifier: Apache-2.0
#include "api/compressed_table_io.hpp"
#include "api/simpatico_codegen.hpp"
#include "codegen/plan/bitjoin_layout.hpp"
#include "codegen/plan/plan_tree.hpp"
#include "codegen/selection/decompression_pushdown_policy.hpp"
#include "decode/decode_session.hpp"
#include "operators/onpair_compressor.hpp"
#include "test_utils.hpp"
#include "util/host_observation.hpp"
#include "util/onpair_like.hpp"

#include <cudf/strings/contains.hpp>
#include <cudf/unary.hpp>

#include <rmm/cuda_stream.hpp>
#include <rmm/mr/cuda_async_memory_resource.hpp>
#include <rmm/mr/statistics_resource_adaptor.hpp>

#include <unistd.h>

#include <algorithm>
#include <array>
#include <cstdio>
#include <numeric>
#include <random>
#include <span>
#include <string_view>

// Observe only explicit calls in the static decoder, outside fixture setup/reference work.
namespace like_observation {
struct counts {
  std::size_t events = 0, readbacks = 0, synchronizations = 0, blocking_groups = 0;
  bool operator==(counts const&) const = default;
};
thread_local bool active = false;
thread_local counts calls;
thread_local bool pending_readback = false;
}  // namespace like_observation
extern "C" cudaError_t __real_cudaEventCreate(cudaEvent_t*);
extern "C" cudaError_t __real_cudaStreamSynchronize(cudaStream_t);
extern "C" cudaError_t __real_cudaMemcpyAsync(
  void*, void const*, std::size_t, cudaMemcpyKind, cudaStream_t);
extern "C" cudaError_t __wrap_cudaEventCreate(cudaEvent_t* event)
{
  if (like_observation::active) ++like_observation::calls.events;
  return __real_cudaEventCreate(event);
}
extern "C" cudaError_t __wrap_cudaStreamSynchronize(cudaStream_t stream)
{
  if (like_observation::active) {
    ++like_observation::calls.synchronizations;
    if (like_observation::pending_readback) {
      ++like_observation::calls.blocking_groups;
      like_observation::pending_readback = false;
    }
  }
  return __real_cudaStreamSynchronize(stream);
}
extern "C" cudaError_t __wrap_cudaMemcpyAsync(
  void* dst, void const* src, std::size_t size, cudaMemcpyKind kind, cudaStream_t stream)
{
  if (like_observation::active && kind == cudaMemcpyDeviceToHost) {
    ++like_observation::calls.readbacks;
    like_observation::pending_readback = true;
  }
  return __real_cudaMemcpyAsync(dst, src, size, kind, stream);
}

namespace {
namespace sc = sirius::codegen;
using simpatico::like_predicate;

std::size_t log_count = 0;
std::uint16_t expected_width = 0;
void log_stats(bool info, bool enabled, simpatico::onpair_like_stats const& stats)
{
  if (info) return;
  expect(enabled && stats.table_ms >= 0 && stats.row_ms >= 0 && stats.matches <= stats.rows,
         "LIKE completed timing and match diagnostics");
  if (expected_width) expect(stats.tokens_per_batch == expected_width, "LIKE column batch width");
  ++log_count;
}

template <typename T>
std::vector<T> read(cudf::column_view column, ::cuda::stream_ref stream)
{
  std::vector<T> values(column.size());
  if (!values.empty())
    simpatico::read_device_bytes_completed(
      values.data(), column.data<T>(), values.size() * sizeof(T), stream);
  return values;
}

std::unique_ptr<cudf::column> reference(cudf::column_view strings,
                                        like_predicate const& pattern,
                                        ::cuda::stream_ref stream,
                                        rmm::device_async_resource_ref mr)
{
  std::string text = "%";
  for (auto const& literal : pattern.literals)
    text += literal + "%";
  auto result = cudf::strings::like(
    cudf::strings_column_view(strings), std::string_view(text), std::string_view(), stream, mr);
  if (pattern.invert)
    result = cudf::unary_operation(result->view(), cudf::unary_operator::NOT, stream, mr);
  return result;
}

void check(simpatico::PlanTree const& tree,
           cudf::column_view decoded,
           like_predicate const& pattern,
           ::cuda::stream_ref stream,
           rmm::device_async_resource_ref mr)
{
  simpatico::decode_predicate predicate;
  predicate.likes = {pattern};
  auto expected   = reference(decoded, pattern, stream, mr);
  auto actual     = simpatico::decode_one(
    {std::cref(tree), simpatico::predicate_result{predicate, {}}, {}}, stream, mr);
  expect(read<std::uint8_t>(actual->view(), stream) == read<std::uint8_t>(expected->view(), stream),
         "OnPair token LIKE differs from cuDF LIKE on full decode");
}

std::vector<std::string> corpus()
{
  std::vector<std::string> rows{"",
                                "",
                                "abcd",
                                "abxcd",
                                "cdab",
                                "ab",
                                "cd",
                                "",
                                "ababababababababababababac",
                                "specialrequests",
                                "special requests",
                                "requests special",
                                "abcdefghijklmnopqrstuvwxyz012345",
                                "google",
                                "Google",
                                ".google.",
                                "xgoogley"};
  // Valid UTF-8, with multibyte characters immediately around ASCII literals.
  // Non-ASCII bytes cannot accidentally satisfy any of the accepted literal bytes.
  rows.insert(rows.end(),
              {"éab中cd🙂",
               "λgoogleé",
               "🙂Google中",
               "中.google.λ",
               "éab🙂",
               "中cdλabé",
               "é特殊 requests🙂",
               "λspecial中requestsé"});
  rows.push_back(std::string(8192, 'a') + "bac");
  rows.push_back(std::string(4096, 'x') + "special" + std::string(8192, 'y') + "requests");
  std::mt19937 random(29);
  std::string alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789./-";
  for (int r = 0; r < 4096; ++r) {
    std::string value;
    for (unsigned n = random() % 257; n; --n)
      value += alphabet[random() % alphabet.size()];
    if (r % 7 == 0) value = "ab" + value + "cd";
    if (r % 13 == 0) value += "special requests";
    rows.push_back(std::move(value));
  }
  // Repetitions train real long dictionary entries, with whole-token matches and tails.
  for (int i = 0; i < 1024; ++i)
    rows.push_back("abcdefghijklmnopabcdefghijklmnopabcd");
  rows.insert(rows.end(), {"", "", ""});
  return rows;
}

// Take literal bytes from real code boundaries: the test asserts actual 2/3/4-token
// coverage instead of assuming a particular trained dictionary segmentation.
std::vector<like_predicate> token_patterns(simpatico::compressed_representation const& rep,
                                           ::cuda::stream_ref stream)
{
  auto channels = rep.named_channels(stream);
  auto codes    = read<std::uint16_t>(channels[0].view, stream);
  auto rows     = read<std::int32_t>(channels[1].view, stream);
  auto bytes    = read<std::uint8_t>(channels[2].view, stream);
  auto offsets  = read<std::uint32_t>(channels[3].view, stream);
  std::vector<like_predicate> result;
  for (int count = 1; count <= 4; ++count) {
    std::size_t row    = 0;
    std::uint64_t byte = 0;
    bool found         = false;
    for (std::size_t start = 0; start + count <= codes.size(); ++start) {
      while (row + 1 < rows.size() && rows[row + 1] <= byte)
        ++row;
      std::string literal;
      for (int j = 0; j < count; ++j) {
        auto code = codes[start + j];
        literal.append(reinterpret_cast<char const*>(bytes.data() + offsets[code]),
                       offsets[code + 1] - offsets[code]);
      }
      if (!literal.empty() && literal.size() <= 32 && row + 1 < rows.size() &&
          byte + literal.size() <= rows[row + 1]) {
        result.push_back({{literal}, false});
        found = true;
        break;
      }
      byte += offsets[codes[start] + 1] - offsets[codes[start]];
    }
    expect(found, "genuine dictionary must furnish a 1/2/3/4-token literal");
  }
  // Exercise a FIRST completion at the largest representable token end when trained.
  for (std::size_t code = 0; code + 1 < offsets.size(); ++code) {
    auto const length = offsets[code + 1] - offsets[code];
    if (length == 16) {
      result.push_back(
        {{std::string(reinterpret_cast<char const*>(bytes.data() + offsets[code]), 16)}, false});
      break;
    }
  }
  return result;
}

void correctness(::cuda::stream_ref stream, rmm::device_async_resource_ref mr)
{
  auto input = make_strings_table(corpus(), {}, stream);
  for (std::string const op : {"onpair_16"}) {
    auto stored = simpatico::compress_with_plan(input->view(), "input -> " + op + "\n", stream, mr);
    auto& tree  = *stored.columns[0].plan_tree;
    expect(simpatico::probe_column(tree).can_answer_like, "non-null OnPair answers LIKE");
    auto decoded  = simpatico::decompress(stored, stream, mr);
    auto patterns = token_patterns(*tree.nodes[1].rep, stream);
    for (auto p : std::vector<like_predicate>{{{"a"}, false},
                                              {{"abcdefghijklmnopqrstuvwxyz012345"}, false},
                                              {{"ab", "cd"}, false},
                                              {{"a", "b", "c"}, false},
                                              {{"a", "b", "c", "d"}, false},
                                              {{"special", "requests"}, false},
                                              {{"ababac"}, false},
                                              {{"google"}, false},
                                              {{"Google"}, false},
                                              {{".google."}, false},
                                              {{"not-present"}, false}})
      patterns.push_back(p);
    for (auto pattern : patterns) {
      check(tree, decoded->view().column(0), pattern, stream, mr);
      pattern.invert = true;
      check(tree, decoded->view().column(0), pattern, stream, mr);
    }
  }
  // Reconstructed byte offsets (the prepared root borrows dictionary/codes), and
  // genuine INT64 offsets, both with and without a delta/bitpack sub-plan.
  for (bool wide : {false, true}) {
    auto strings = make_strings_column(
      {"", "abcd", "abxcd", "cdab", "", std::string(9000, 'x'), ""}, {}, stream);
    if (wide) {
      auto offsets  = cudf::cast(cudf::strings_column_view(strings->view()).offsets(),
                                 cudf::data_type{cudf::type_id::INT64},
                                 stream,
                                 mr);
      auto contents = strings->release();
      strings       = cudf::make_strings_column(
        7,
        std::move(offsets),
        std::move(*contents.data),
        0,
        cudf::create_null_mask(0, cudf::mask_state::UNALLOCATED, stream, mr));
    }
    for (bool compressed_offsets : {false, true}) {
      std::string plan = "input -> onpair_16\n";
      if (compressed_offsets)
        plan =
          "input -> onpair_16 -> codes, row_offsets, dict_bytes, dict_offsets, batch_prefixes\n"
          "onpair_16.row_offsets -> delta -> differences\n"
          "onpair_16.row_offsets.differences -> bitpack\n";
      auto stored =
        simpatico::compress_with_plan(cudf::table_view{{strings->view()}}, plan, stream, mr);
      auto decoded = simpatico::decompress(stored, stream, mr);
      check(
        *stored.columns[0].plan_tree, decoded->view().column(0), {{"ab", "cd"}, false}, stream, mr);
    }
  }
  for (auto rows : std::vector<std::vector<std::string>>{{}, {"", "", ""}}) {
    auto empty  = make_strings_table(rows, {}, stream);
    auto stored = simpatico::compress_with_plan(empty->view(), "input -> onpair_16\n", stream, mr);
    check(*stored.columns[0].plan_tree, empty->view().column(0), {{"a"}, true}, stream, mr);
  }
}

template <typename T>
std::unique_ptr<cudf::column> upload(std::vector<T> const& values,
                                     cudf::type_id type,
                                     ::cuda::stream_ref stream,
                                     rmm::device_async_resource_ref mr)
{
  auto column = cudf::make_fixed_width_column(
    cudf::data_type{type}, values.size(), cudf::mask_state::UNALLOCATED, stream, mr);
  if (!values.empty())
    cuda_check(cudaMemcpyAsync(column->mutable_view().template head<void>(),
                               values.data(),
                               values.size() * sizeof(T),
                               cudaMemcpyHostToDevice,
                               stream.get()));
  stream.sync();
  return column;
}

void deterministic_continuation(::cuda::stream_ref stream,
                                rmm::device_async_resource_ref mr,
                                std::uint16_t width = 128)
{
  // Fix the token segmentation so training can never remove the edge cases:
  // "abcd" is one token; a second row is [abcdefghijklmnop][qrst].
  std::string const dictionary = "abcdefghijklmnopqrstabcd";
  std::vector<std::unique_ptr<cudf::column>> channels;
  channels.push_back(upload<std::uint16_t>({2, 0, 1, 0, 1, 0}, cudf::type_id::UINT16, stream, mr));
  channels.push_back(
    upload<std::int32_t>({0, 4, 24, 40, 60, 60}, cudf::type_id::INT32, stream, mr));
  channels.push_back(upload(std::vector<std::uint8_t>(dictionary.begin(), dictionary.end()),
                            cudf::type_id::UINT8,
                            stream,
                            mr));
  channels.push_back(upload<std::uint32_t>({0, 16, 20, 24}, cudf::type_id::UINT32, stream, mr));
  channels.push_back(upload<std::uint64_t>({0, 60}, cudf::type_id::UINT64, stream, mr));
  auto parsed = simpatico::plan_tree_from_dsl("input -> onpair_16\n");
  expect(parsed.has_value(), "deterministic LIKE fixture plan");
  auto& root = parsed->nodes[1];
  simpatico::leaf_meta::onpair metadata{5, 60, 16};
  metadata.tokens_per_batch = width;
  root.rep                  = std::make_unique<simpatico::onpair_compressed_representation>(
    metadata, std::move(channels), stream, mr);
  root.meta = root.rep->describe_meta();
  auto decoded =
    simpatico::decode_one({std::cref(*parsed), simpatico::value_result{}, {}}, stream, mr);
  for (auto pattern :
       std::vector<like_predicate>{{{"ab", "cd"}, false}, {{"abcdefghijklmnop", "qrst"}, false}}) {
    check(*parsed, decoded->view(), pattern, stream, mr);
    pattern.invert = true;
    check(*parsed, decoded->view(), pattern, stream, mr);
  }
  simpatico::decode_predicate pred;
  pred.likes  = {{{"abcdefghijklmnop", "qrst"}, false}};
  auto answer = simpatico::decode_one(
    {std::cref(*parsed), simpatico::predicate_result{pred, {}}, {}}, stream, mr);
  expect(read<std::uint8_t>(answer->view(), stream) == std::vector<std::uint8_t>({0, 1, 0, 0, 0}),
         "end offset 16 continues at the next token, without wrapping to byte zero");
}

void k6_like_batch_boundary(::cuda::stream_ref stream, rmm::device_async_resource_ref mr)
{
  // Only tokens 191 and 192 spell "ab". Keep the codes fixed so training cannot merge them.
  std::vector<std::uint16_t> codes(193, 0);
  codes[191] = 1;
  codes[192] = 2;
  std::vector<std::unique_ptr<cudf::column>> channels;
  channels.push_back(upload(codes, cudf::type_id::UINT16, stream, mr));
  channels.push_back(upload<std::int32_t>({0, 191, 193}, cudf::type_id::INT32, stream, mr));
  channels.push_back(upload<std::uint8_t>({'x', 'a', 'b'}, cudf::type_id::UINT8, stream, mr));
  channels.push_back(upload<std::uint32_t>({0, 1, 2, 3}, cudf::type_id::UINT32, stream, mr));
  channels.push_back(upload<std::uint64_t>({0, 192, 193}, cudf::type_id::UINT64, stream, mr));
  auto tree = simpatico::plan_tree_from_dsl("input -> onpair_16\n");
  expect(tree.has_value(), "K=6 LIKE boundary plan");
  auto& root = tree->nodes[1];
  simpatico::leaf_meta::onpair metadata{2, 193, 16};
  metadata.tokens_per_batch = 192;
  root.meta                 = metadata;
  root.rep                  = std::make_unique<simpatico::onpair_compressed_representation>(
    metadata, std::move(channels), stream, mr);
  auto decoded =
    simpatico::decode_one({std::cref(*tree), simpatico::value_result{}, {}}, stream, mr);
  auto oracle = make_strings_column({std::string(191, 'x'), "ab"}, {}, stream);
  expect(strings_equal(decoded->view(), oracle->view(), stream), "K=6 LIKE fixture byte offsets");
  check(*tree, decoded->view(), {{"ab"}, false}, stream, mr);
  check(*tree, decoded->view(), {{"ab"}, true}, stream, mr);
}

void instrumentation_and_shared_bounds(::cuda::stream_ref stream, rmm::device_async_resource_ref mr)
{
  auto input  = make_strings_table({"abcd", "abxcd", "cdab", "", "x"}, {}, stream);
  auto stored = simpatico::compress_with_plan(input->view(), "input -> onpair_16\n", stream, mr);
  auto const& tree = *stored.columns[0].plan_tree;
  simpatico::decode_predicate pred;
  pred.likes   = {{{"ab", "cd"}, false}};
  auto observe = [&](simpatico::onpair_like_log_fn logger, bool stats) {
    simpatico::onpair_like_log_scope log{logger, stats};
    like_observation::calls  = {};
    like_observation::active = true;
    auto output              = simpatico::decode_one(
      {std::cref(tree), simpatico::predicate_result{pred, {}}, {}}, stream, mr);
    like_observation::active = false;
    return std::pair{like_observation::calls, std::move(output)};
  };
  auto baseline = observe(nullptr, false);
  auto quiet    = observe(log_stats, false);  // INFO callback installed, DEBUG disabled
  expect(quiet.first == baseline.first,
         "INFO-only LIKE adds no events, readbacks or synchronization");
  auto debug = observe(log_stats, true);
  expect(debug.first.events == baseline.first.events + 3 &&
           debug.first.readbacks == baseline.first.readbacks + 1 &&
           debug.first.synchronizations == baseline.first.synchronizations + 1,
         "DEBUG alone collects three events and the match readback");
  pred.likes.push_back({{"x"}, true});
  auto grouped = observe(log_stats, false);
  expect(grouped.first == baseline.first,
         "two LIKEs share one bounds status readback and no cross-stream event");
  expect(read<std::uint8_t>(grouped.second->view(), stream) ==
           std::vector<std::uint8_t>({1, 0, 0, 0, 0}),
         "shared bounds preserve independent inversion and conjunction");
  // Corruption must still fail before any grouped result is published.
  auto channels        = tree.nodes[1].rep->named_channels(stream);
  std::int32_t invalid = -1;
  cuda_check(cudaMemcpyAsync(const_cast<void*>(channels[1].view.head<void>()),
                             &invalid,
                             sizeof(invalid),
                             cudaMemcpyHostToDevice,
                             stream.get()));
  stream.sync();
  simpatico::onpair_like_log_scope quiet_log{log_stats, false};
  expect(throws<std::invalid_argument>([&] {
           (void)simpatico::decode_one(
             {std::cref(tree), simpatico::predicate_result{pred, {}}, {}}, stream, mr);
         }),
         "shared bounds retain corrupt-row validation");
}

void corrupt_bounds(::cuda::stream_ref stream, rmm::device_async_resource_ref mr)
{
  auto strings = make_strings_table(
    std::vector<std::string>(1024, "abcdefghijklmnopabcdefghijklmnop"), {}, stream);
  auto stored = simpatico::compress_with_plan(strings->view(), "input -> onpair_16\n", stream, mr);
  auto& tree  = *stored.columns[0].plan_tree;
  auto channels   = tree.nodes[1].rep->named_channels(stream);
  auto codes      = read<std::uint16_t>(channels[0].view, stream);
  auto dictionary = read<std::uint32_t>(channels[3].view, stream);
  auto rows       = read<std::int32_t>(channels[1].view, stream);
  // Put the first interior row boundary one byte into an actual multi-byte token.
  std::uint32_t byte = 0;
  bool changed       = false;
  for (auto code : codes) {
    auto length = dictionary[code + 1] - dictionary[code];
    if (length > 1 && byte + 1 < static_cast<std::uint32_t>(rows[2])) {
      rows[1] = byte + 1;
      changed = true;
      break;
    }
    byte += length;
  }
  expect(changed, "trained real bridge must provide a multi-byte token");
  cuda_check(cudaMemcpyAsync(const_cast<void*>(channels[1].view.head<void>()),
                             rows.data(),
                             rows.size() * sizeof(rows[0]),
                             cudaMemcpyHostToDevice,
                             stream.get()));
  stream.sync();
  simpatico::decode_predicate predicate;
  predicate.likes = {like_predicate{{"a"}, false}};
  bool rejected   = false;
  try {
    (void)simpatico::decode_one(
      {std::cref(tree), simpatico::predicate_result{predicate, {}}, {}}, stream, mr);
  } catch (std::invalid_argument const& error) {
    rejected = std::string(error.what()).find("onpair") != std::string::npos;
  }
  expect(rejected, "corrupt row start inside token must throw naming onpair");
  cuda_check(cudaGetLastError());
}

void preflight(::cuda::stream_ref stream, rmm::device_async_resource_ref mr, bool off)
{
  auto strings = make_strings_table({"abcd", "", "google"}, {}, stream);
  // The selective route and LIKE answering are keyed to the production onpair_16 root; other
  // dictionary widths are ordinary (full-decode) OnPair roots.
  for (std::string const op : {"identity", "dictionary", "onpair_9", "onpair_16"}) {
    auto stored =
      simpatico::compress_with_plan(strings->view(), "input -> " + op + "\n", stream, mr);
    auto const& tree   = *stored.columns[0].plan_tree;
    bool const allowed = !off && op == "onpair_16";
    expect(simpatico::probe_column(tree).can_answer_like == allowed, "LIKE capability/root/switch");
    simpatico::decode_predicate pred;
    pred.likes = {like_predicate{{"ab"}, false}};
    std::string error;
    auto result = simpatico::decompress_column(tree, stream, mr, &error, &pred);
    expect(static_cast<bool>(result) == allowed, "LIKE preflight refuses without full decoding");
    if (!allowed) expect(error.find("onpair") != std::string::npos, "LIKE refusal reason");
    pred.equals_any = {"abcd"};
    expect(!simpatico::decompress_column(tree, stream, mr, &error, &pred),
           "two predicate forms refused");
    if (off) {
      sc::scan_filter_request request;
      request.routes = {simpatico::probe_column(tree).compact_route};
      request.like_filters.push_back({0, {{"ab"}, false}});
      request.row_count_only = true;  // OFF must refuse the carrier along with answering LIKE.
      sc::scan_filter_result status;
      std::array<std::size_t, 1> selected{0};
      std::array<::cuda::stream_ref, 1> lanes{stream};
      auto refused =
        simpatico::decompress_scan_filter(stored, selected, request, status, lanes, stream, mr);
      expect(!status.applied &&
               columns_equal_any(refused->view().column(0), strings->view().column(0), stream),
             "switch off refuses fused LIKE, returns same bytes");
      auto normal = simpatico::decompress(stored, stream, mr);
      expect(columns_equal_any(normal->view().column(0), strings->view().column(0), stream),
             "switch off preserves ordinary values");
    }
  }
  auto nullable = make_strings_table({"abcd", ""}, {true, false}, stream);
  auto stored = simpatico::compress_with_plan(nullable->view(), "input -> onpair_16\n", stream, mr);
  expect(!simpatico::probe_column(*stored.columns[0].plan_tree).can_answer_like,
         "nullable LIKE refused");
}

void wave(::cuda::stream_ref stream, rmm::device_async_resource_ref mr)
{
  std::vector<std::string> rows{"abcd", "abxcd", "cdab", "", "xx", "abcd", "x"};
  std::vector<std::unique_ptr<cudf::column>> columns;
  std::vector<std::int32_t> numbers(rows.size());
  std::iota(numbers.begin(), numbers.end(), 0);
  auto numeric = cudf::make_fixed_width_column(
    cudf::data_type{cudf::type_id::INT32}, rows.size(), cudf::mask_state::UNALLOCATED, stream, mr);
  cuda_check(cudaMemcpyAsync(numeric->mutable_view().head<void>(),
                             numbers.data(),
                             numbers.size() * sizeof(numbers[0]),
                             cudaMemcpyHostToDevice,
                             stream.get()));
  columns.push_back(std::move(numeric));
  columns.push_back(make_strings_column(rows, {}, stream));
  cudf::table input{std::move(columns)};
  auto stored = simpatico::compress_with_plan(
    input.view(), "input -> bitpack\n---\ninput -> onpair_16\n", stream, mr);
  std::array<std::size_t, 2> selected{0, 1};
  rmm::cuda_stream other;
  stream.sync();  // stored inputs must be ready on both borrowed lanes
  std::array<::cuda::stream_ref, 2> lanes{stream, other};
  for (bool invert : {false, true})
    for (bool projected : {false, true})
      for (bool range : {false, true}) {
        sc::scan_filter_request request;
        request.routes = {sc::decode_route::bitpack_mask, sc::decode_route::onpair};
        request.like_filters.push_back({1, {{"ab", "cd"}, invert}});
        if (!projected) {
          request.delivered_prefix   = 1;
          request.like_bool8_columns = {1};
        }
        if (range) request.filters.push_back({0, {1, 5}});
        sc::scan_filter_result result;
        auto output =
          simpatico::decompress_scan_filter(stored, selected, request, result, lanes, stream, mr);
        std::vector<std::int32_t> kept;
        std::vector<std::string> kept_strings;
        for (std::size_t i = 0; i < rows.size(); ++i) {
          bool hit = rows[i] == "abcd" || rows[i] == "abxcd";
          if ((hit != invert) && (!range || (i >= 1 && i <= 5))) {
            kept.push_back(i);
            kept_strings.push_back(rows[i]);
          }
        }
        bool const above_cap =
          sc::onpair_like_mask_above_cap_enabled() &&
          double(kept.size()) / rows.size() > sc::decompression_pushdown_max_selectivity();
        if (above_cap) {
          expect(read<std::int32_t>(output->view().column(0), stream) == numbers &&
                   output->num_columns() == 2 && output->num_rows() == rows.size(),
                 "above-cap LIKE preserves every delivered value's full row domain");
          if (projected) {
            expect(!result.applied && !result.like_mask_delivered &&
                     columns_equal_any(output->view().column(1), input.view().column(1), stream),
                   "projected high-survival LIKE declines to original STRING values");
          } else {
            std::vector<std::uint8_t> answers;
            for (auto const& row : rows)
              answers.push_back((row == "abcd" || row == "abxcd") != invert);
            expect(result.applied && result.like_mask_delivered &&
                     result.like_bool8_columns == std::vector<std::size_t>{1} &&
                     read<std::uint8_t>(output->view().column(1), stream) == answers &&
                     result.row_indices.is_empty(),
                   "filter-only high-survival LIKE delivers full BOOL8, leaving ranges residual");
          }
          continue;
        }
        expect(result.applied && !result.like_mask_delivered &&
                 read<std::int32_t>(output->view().column(0), stream) == kept,
               "LIKE ballot and optional range select exact rows");
        expect(output->num_columns() == (projected ? 2 : 1),
               "filter-only LIKE omitted from wave 2");
        if (projected) {
          auto expected = make_strings_column(kept_strings, {}, stream);
          expect(columns_equal_any(output->view().column(1), expected->view(), stream),
                 "projected LIKE column delivers compacted STRING values");
        }
      }
  for (bool count_only : {false, true}) {
    sc::scan_filter_request request;
    request.routes         = {sc::decode_route::bitpack_mask, sc::decode_route::onpair};
    request.like_filters   = {{1, {{"ab", "cd"}, false}}, {1, {{"x"}, true}}};
    request.row_count_only = count_only;
    sc::scan_filter_result result;
    auto output =
      simpatico::decompress_scan_filter(stored, selected, request, result, lanes, stream, mr);
    expect(result.applied && output->num_rows() == 2, "same-column LIKE conjunction in a wave");
    if (count_only) {
      expect(output->num_columns() == 1 &&
               output->view().column(0).type().id() == cudf::type_id::INT8 &&
               result.row_indices.is_empty() && result.routes.empty(),
             "all-filter-only wave carries only the count, without string or indices");
    } else {
      expect(
        read<std::int32_t>(output->view().column(0), stream) == std::vector<std::int32_t>({0, 5}),
        "grouped LIKE mask keeps projected values");
    }
  }
  // Q20 shape: the only selected column is the filter string, without an output prefix.
  for (auto pattern : std::vector<like_predicate>{
         {{"ab", "cd"}, false}, {{"absent"}, false}, {{"absent"}, true}}) {
    sc::scan_filter_request request;
    request.routes         = {sc::decode_route::onpair};
    request.like_filters   = {{0, pattern}};
    request.row_count_only = true;
    sc::scan_filter_result result;
    std::array<std::size_t, 1> only_string{1};
    auto output =
      simpatico::decompress_scan_filter(stored, only_string, request, result, lanes, stream, mr);
    auto expected = pattern.literals[0] == "ab" ? 3 : pattern.invert ? 7 : 0;
    expect(result.applied && output->num_columns() == 1 && output->num_rows() == expected &&
             output->view().column(0).type().id() == cudf::type_id::INT8 &&
             result.row_indices.is_empty(),
           "count-only LIKE preserves zero/partial/all survivor cardinality without strings");
  }
  // Source cap is checked before work; refusal leaves ordinary string values.
  sc::scan_filter_request request;
  request.routes = {sc::decode_route::bitpack_mask, sc::decode_route::onpair};
  for (int i = 0; i < 9; ++i)
    request.like_filters.push_back({1, {{"ab"}, false}});
  sc::scan_filter_result result;
  auto output =
    simpatico::decompress_scan_filter(stored, selected, request, result, lanes, stream, mr);
  expect(!result.applied && output->num_rows() == input.num_rows() &&
           output->view().column(1).type().id() == cudf::type_id::STRING,
         "too many LIKE sources refuse to ordinary values");
}
// Counts allocations from the test mr, including compacted outputs retained by earlier batches.
// One lane makes queued frees and subsequent allocations FIFO, so this is the allocator's live
// requested-byte peak, independent of the decoder's staging estimates/counters.
class peak_tracking_resource {
 public:
  struct counts {
    std::size_t live = 0, peak = 0;
  };
  explicit peak_tracking_resource(rmm::device_async_resource_ref upstream) : upstream_(upstream) {}
  std::shared_ptr<counts> observed = std::make_shared<counts>();
  void* allocate(cuda::stream_ref stream, std::size_t bytes, std::size_t alignment)
  {
    auto* p = upstream_.allocate(stream, bytes, alignment);
    if (p) {
      observed->live += bytes;
      observed->peak = std::max(observed->peak, observed->live);
    }
    return p;
  }
  void deallocate(cuda::stream_ref stream,
                  void* p,
                  std::size_t bytes,
                  std::size_t alignment) noexcept
  {
    if (p) observed->live -= bytes;
    upstream_.deallocate(stream, p, bytes, alignment);
  }
  void* allocate_sync(std::size_t bytes, std::size_t alignment)
  {
    auto* p = allocate(cuda::stream_ref{cudaStream_t{nullptr}}, bytes, alignment);
    cuda_check(cudaStreamSynchronize(nullptr));
    return p;
  }
  void deallocate_sync(void* p, std::size_t bytes, std::size_t alignment) noexcept
  {
    deallocate(cuda::stream_ref{cudaStream_t{nullptr}}, p, bytes, alignment);
    (void)cudaStreamSynchronize(nullptr);
  }
  bool operator==(peak_tracking_resource const& other) const noexcept
  { return observed == other.observed; }
  friend void get_property(peak_tracking_resource const&, cuda::mr::device_accessible) noexcept {}

 private:
  rmm::device_async_resource_ref upstream_;
};

void ordered_peak_regression(::cuda::stream_ref stream, rmm::device_async_resource_ref mr)
{
  constexpr std::size_t mib      = 1024 * 1024;
  constexpr cudf::size_type rows = 8 * 1024 * 1024;
  // Exact review counterexample: 128 MiB -> 8 MiB, then 80 x (16 MiB -> 1 MiB).
  // Repeated projection slots avoid storing eighty identical inputs outside the measured window.
  std::vector<std::unique_ptr<cudf::column>> input_columns;
  for (auto type :
       {cudf::data_type{cudf::type_id::DECIMAL128, 0}, cudf::data_type{cudf::type_id::INT16}}) {
    auto column =
      cudf::make_fixed_width_column(type, rows, cudf::mask_state::UNALLOCATED, stream, mr);
    cuda_check(cudaMemsetAsync(
      column->mutable_view().head<void>(), 0, rows * cudf::size_of(type), stream.get()));
    input_columns.push_back(std::move(column));
  }
  std::vector<std::int32_t> keys(rows);
  for (std::size_t i = 0; i < keys.size(); ++i)
    keys[i] = i % 16;
  input_columns.push_back(upload(keys, cudf::type_id::INT32, stream, mr));
  cudf::table input{std::move(input_columns)};
  auto stored = simpatico::compress_with_plan(
    input.view(), "input -> identity\n---\ninput -> identity\n---\ninput -> bitpack\n", stream, mr);
  stream.sync();
  std::vector<std::size_t> selected{0};
  selected.insert(selected.end(), 80, 1);
  selected.push_back(2);
  sc::scan_filter_request request;
  request.delivered_prefix = 81;
  request.routes.assign(81, sc::decode_route::full);
  request.routes.push_back(sc::decode_route::bitpack_mask);
  request.filters = {{81, {0, 0}}};
  peak_tracking_resource baseline_mr{mr}, staged_mr{mr};
  sc::scan_filter_result baseline_result, staged_result;
  request.staged_byte_budget = 0;
  auto baseline              = simpatico::decompress_scan_filter(
    stored, selected, request, baseline_result, std::span{&stream, 1}, stream, baseline_mr);
  request.staged_byte_budget = 64 * mib;
  auto staged                = simpatico::decompress_scan_filter(
    stored, selected, request, staged_result, std::span{&stream, 1}, stream, staged_mr);
  expect(baseline_result.applied && staged_result.applied && baseline->num_rows() == rows / 16 &&
           staged->num_rows() == rows / 16 && baseline->num_columns() == 81 &&
           staged->num_columns() == 81,
         "peak regression preserves 1/16 selection and all 81 outputs");
  expect(baseline_mr.observed->peak >= 136 * mib,
         "allocator sees the initial 128 MiB full plus 8 MiB compacted output");
  expect(staged_mr.observed->peak <= baseline_mr.observed->peak + 64 * mib,
         "allocator peak includes retained outputs and stays within old peak plus 64 MiB");
  if (sc::decompression_pushdown_batched_wave2_enabled())
    expect(staged_result.wave2_syncs == 0 && staged_result.wave2_readbacks == 0,
           "fixed-width batches add no metadata or per-batch host observations");
  for (cudf::size_type i = 0; i < 81; ++i)
    expect(columns_equal_any(baseline->view().column(i), staged->view().column(i), stream),
           "every staged column equals its interleaved baseline in traversal order");
  std::printf("ordered wave2 allocator peak: baseline=%zu staged=%zu allowance=%zu\n",
              baseline_mr.observed->peak,
              staged_mr.observed->peak,
              64 * mib);
  baseline.reset();
  staged.reset();
  baseline_result = {};
  staged_result   = {};
  expect(baseline_mr.observed->live == 0 && staged_mr.observed->live == 0,
         "peak tracking resources see no leaked output or temporary allocations");
}

void wave2_regressions(::cuda::stream_ref stream, rmm::device_async_resource_ref mr)
{
  // Production URL root and offsets tail; cache construction is outside the observation window.
  std::vector<std::string> rows(4099, "other");
  rows[3]     = "google";
  rows[4000]  = "google-x";
  auto input  = make_strings_table(rows, {}, stream);
  auto stored = simpatico::compress_with_plan(
    input->view(),
    "input -> onpair_16 -> codes, row_offsets, dict_bytes, dict_offsets, batch_prefixes\n"
    "onpair_16.row_offsets -> delta -> differences\n"
    "onpair_16.row_offsets.differences -> ans\n",
    stream,
    mr);
  auto const& tree     = *stored.columns[0].plan_tree;
  bool const batched   = sc::decompression_pushdown_batched_wave2_enabled();
  bool const zero_skip = sc::decompression_pushdown_full_route_zero_skip_enabled();
  auto ans_layout      = [](simpatico::PlanTree const& plan) {
    auto ans = std::find_if(
      plan.nodes.begin(), plan.nodes.end(), [](auto const& node) { return node.op == "ans"; });
    expect(ans != plan.nodes.end(), "offsets plan has an ANS node");
    return std::get<simpatico::leaf_meta::ans>(ans->meta).frame_layout;
  };
  auto const fresh_layout = ans_layout(tree);
  expect(fresh_layout != nullptr, "compressed ANS offsets retain their encoder layout");
  auto const* root = dynamic_cast<simpatico::onpair_compressed_representation const*>(
    tree.nodes[1].prepared_rep.get());
  expect(root && root->row_byte_offsets().type().id() == cudf::type_id::EMPTY,
         "metadata caching must not retain full decoded row offsets");
  std::vector<std::uint32_t> words(sc::selection_mask::AllocWordsFor(rows.size()), 0);
  words[3 / 32] |= 1u << (3 % 32);
  words[4000 / 32] |= 1u << (4000 % 32);
  rmm::device_buffer mask_words(words.data(), words.size() * sizeof(words[0]), stream, mr);
  rmm::device_buffer offsets((sc::selection_mask::ChunksFor(rows.size()) + 1) * 4, stream, mr);
  sc::selection_mask mask{static_cast<std::uint32_t*>(mask_words.data()),
                          static_cast<std::int64_t>(rows.size()),
                          -1,
                          static_cast<std::uint32_t*>(offsets.data())};
  sc::run_selection_cnt(mask, stream, mr);
  // Reload the URL offsets tail from its persisted bytes, without a pin sidecar.
  struct persisted_file {
    char path[64] = "/tmp/simpatico_like_reload_XXXXXX";
    persisted_file()
    {
      auto fd = mkstemp(path);
      if (fd < 0) throw std::runtime_error("reload test: mkstemp failed");
      ::close(fd);
    }
    ~persisted_file() { std::remove(path); }
  } file;
  expect(simpatico::write_compressed_table(stored, file.path, stream).empty(),
         "persist ANS offsets plan");
  std::string error;
  auto loaded = simpatico::read_compressed_table(file.path, stream, mr, &error);
  expect(error.empty() && loaded.num_columns() == 1, "reload persisted ANS offsets plan");
  auto const& loaded_tree = *loaded.columns[0].plan_tree;
  auto const loaded_layout = ans_layout(loaded_tree);
  expect(loaded_layout && loaded_layout != fresh_layout, "reload owns a fresh immutable layout");
  auto const before = simpatico::host_observations.readbacks;
  simpatico::prepare_onpair_nodes(*loaded.columns[0].plan_tree, stream, mr);
  expect(
    ans_layout(loaded_tree) == loaded_layout && simpatico::host_observations.readbacks == before,
    "OnPair preparation preserves cached ANS metadata without reading the frame");
  auto const* loaded_root = dynamic_cast<simpatico::onpair_compressed_representation const*>(
    loaded_tree.nodes[1].prepared_rep.get());
  expect(loaded_root && loaded_root->row_byte_offsets().type().id() == cudf::type_id::EMPTY,
         "reload cache retains metadata, not decoded full-width offsets");
  for (auto const* source : {&stored, &loaded}) {
    for (int repeat = 0; repeat < 2; ++repeat) {
      like_observation::calls            = {};
      like_observation::pending_readback = false;
      like_observation::active           = true;
      auto selected = simpatico::decompress_column_compacted(*source, 0, mask, stream, mr);
      like_observation::active = false;
      expect(like_observation::calls.readbacks == 1,
             "selected decode reads only the survivor byte count");
      expect(like_observation::calls.blocking_groups == 1,
             "selected observation groups exclude completion drains");
      auto expected = make_strings_column({"google", "google-x"}, {}, stream);
      expect(columns_equal_any(selected->view(), expected->view(), stream),
             "cached ANS offsets preserve exact survivor bytes after reload and reuse");
    }
  }

  // Interleave heterogeneous full routes with selective routes. The table gather must restore
  // request order, preserve empty selections, and work with one or multiple pool lanes.
  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(upload<std::int32_t>(
    std::vector<std::int32_t>(rows.size(), 17), cudf::type_id::INT32, stream, mr));
  columns.push_back(make_strings_column(rows, {}, stream));
  columns.push_back(upload<std::int16_t>(
    std::vector<std::int16_t>(rows.size(), 29), cudf::type_id::INT16, stream, mr));
  columns.push_back(make_strings_column(rows, {}, stream));
  cudf::table mixed_input{std::move(columns)};
  auto mixed = simpatico::compress_with_plan(mixed_input.view(),
                                             "input -> identity\n---\ninput -> str_split\n---\n"
                                             "input -> bitpack\n---\ninput -> onpair_16\n",
                                             stream,
                                             mr);
  rmm::cuda_stream other;
  stream.sync();
  std::array<std::size_t, 4> selected{0, 1, 2, 3};
  for (int lane_count : {1, 2}) {
    std::array<::cuda::stream_ref, 2> lanes{stream, other};
    for (bool empty : {false, true}) {
      for (std::size_t budget :
           {std::size_t{64 * 1024 * 1024}, std::size_t{17000}, std::size_t{0}}) {
        sc::scan_filter_request request;
        request.staged_byte_budget = budget;
        request.routes             = {sc::decode_route::full,
                                      sc::decode_route::full,
                                      sc::decode_route::full,
                                      sc::decode_route::onpair};
        request.like_filters       = {{3, {{empty ? "absent" : "google"}, false}}};
        sc::scan_filter_result result;
        auto output =
          simpatico::decompress_scan_filter(mixed,
                                            selected,
                                            request,
                                            result,
                                            std::span{lanes.data(), std::size_t(lane_count)},
                                            stream,
                                            mr);
        expect(
          result.applied && output->num_rows() == (empty ? 0 : 2) && output->num_columns() == 4,
          "staged heterogeneous full routes preserve cardinality and width");
        expect(result.peak_staged_bytes <= budget, "retained full buffers stay within byte cap");
        // With no survivors and the zero skip on, the three full routes return typed empty
        // columns without a decode, so neither staging nor interleaving runs.
        bool const skipped = empty && zero_skip;
        expect(result.zero_skipped_full_columns == (skipped ? 3u : 0u),
               "zero-survivor full routes skip exactly when the switch is on");
        if (skipped) {
          expect(result.full_decode_batches == 0 && result.interleaved_full_columns == 0,
                 "skipped full routes neither stage nor interleave");
        } else {
          if (batched && budget == 17000) {
            // INT32 + INT16 total 24594 bytes exceeds the cap; the string is oversized.
            expect(result.full_decode_batches == 2 && result.interleaved_full_columns == 1,
                   "small budget splits staging and interleaves an oversized column");
          }
          if (!batched || budget == 0)
            expect(result.full_decode_batches == 0 && result.interleaved_full_columns == 3,
                   "batching off and zero budget retain per-column decode/gather");
        }
        auto strings = make_strings_column(
          empty ? std::vector<std::string>{} : std::vector<std::string>{"google", "google-x"},
          {},
          stream);
        expect(columns_equal_any(output->view().column(1), strings->view(), stream) &&
                 columns_equal_any(output->view().column(3), strings->view(), stream),
               "staged full gather and selected OnPair deliver identical string bytes");
        expect(read<std::int32_t>(output->view().column(0), stream) ==
                   std::vector<std::int32_t>(empty ? 0 : 2, 17) &&
                 read<std::int16_t>(output->view().column(2), stream) ==
                   std::vector<std::int16_t>(empty ? 0 : 2, 29),
               "staged full numeric routes keep their output positions");
      }
    }
  }
}

void cost_regression(::cuda::stream_ref stream, rmm::device_async_resource_ref mr)
{
  std::vector<std::string> rows(4099, "other");
  rows[3] = "google";
  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(upload<std::int32_t>(
    std::vector<std::int32_t>(rows.size(), 17), cudf::type_id::INT32, stream, mr));
  columns.push_back(make_strings_column(rows, {}, stream));
  cudf::table input{std::move(columns)};
  auto stored = simpatico::compress_with_plan(
    input.view(), "input -> identity\n---\ninput -> onpair_16\n", stream, mr);
  // Repeated projection entries are legal and let this small fixture exercise 105 delivered slots.
  std::vector<std::size_t> selected(100, 0);
  selected.insert(selected.end(), 5, 1);
  sc::scan_filter_request request;
  request.routes.assign(100, sc::decode_route::full);
  request.routes.insert(request.routes.end(), 5, sc::decode_route::onpair);
  request.like_filters = {{100, {{"google"}, false}}};
  sc::scan_filter_result result;
  auto output = simpatico::decompress_scan_filter(
    stored, selected, request, result, std::span{&stream, 1}, stream, mr);
  bool const cost_rule = sc::decompression_pushdown_cost_rule_enabled();
  expect(result.applied != cost_rule && output->num_columns() == 105 &&
           output->num_rows() == (cost_rule ? rows.size() : 1),
         "cost switch toggles the wide projection between ordinary decode and compaction");
  expect(result.status ==
           (cost_rule ? sc::scan_filter_status::declined_cost : sc::scan_filter_status::applied),
         "wide projection reports cost refusal, not an unselective generation");
  auto expected =
    make_strings_column(cost_rule ? rows : std::vector<std::string>{"google"}, {}, stream);
  expect(columns_equal_any(output->view().column(104), expected->view(), stream),
         "cost refusal delivers ordinary STRING values for residual LIKE");
}

void mask_above_cap_regressions(::cuda::stream_ref stream, rmm::device_async_resource_ref mr)
{
  bool const mask_enabled = sc::onpair_like_mask_above_cap_enabled();
  std::vector<std::string> comments(1000, "ordinary order comment");
  for (int i = 0; i < 11; ++i)
    comments[i] = "special requests";
  std::vector<std::int32_t> keys(comments.size());
  std::iota(keys.begin(), keys.end(), 0);
  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(upload(keys, cudf::type_id::INT32, stream, mr));
  columns.push_back(upload(keys, cudf::type_id::INT32, stream, mr));
  columns.push_back(make_strings_column(comments, {}, stream));
  cudf::table input{std::move(columns)};
  auto stored = simpatico::compress_with_plan(
    input.view(), "input -> bitpack\n---\ninput -> bitpack\n---\ninput -> onpair_16\n", stream, mr);
  std::array<std::size_t, 3> selected{0, 1, 2};
  for (bool projected : {false, true}) {
    for (bool keep_mask : {false, true}) {
      sc::scan_filter_request request;
      request.routes = {
        sc::decode_route::bitpack_mask, sc::decode_route::bitpack_mask, sc::decode_route::onpair};
      request.like_filters = {{2, {{"special", "requests"}, true}}, {2, {{"absent"}, true}}};
      if (!projected) {
        request.delivered_prefix   = 2;
        request.like_bool8_columns = {2};
      }
      std::vector<std::uint32_t> visibility((comments.size() + 31) / 32, 0);
      for (std::size_t i = 0; i < comments.size(); ++i)
        if (i % 2 == 0) visibility[i / 32] |= 1u << (i % 32);
      if (keep_mask) {
        request.keep_mask_words = visibility.data();
        request.keep_mask_rows  = comments.size();
      }
      sc::scan_filter_result result;
      auto output = simpatico::decompress_scan_filter(
        stored, selected, request, result, std::span{&stream, 1}, stream, mr);
      if (mask_enabled) {
        expect(output->num_columns() == 3 && output->num_rows() == comments.size() &&
                 result.row_indices.is_empty() && !result.keep_mask_applied &&
                 read<std::int32_t>(output->view().column(0), stream) == keys &&
                 read<std::int32_t>(output->view().column(1), stream) == keys,
               "98.9 percent LIKE delivers INT32 values full-width, without survivor indices");
        if (projected) {
          expect(!result.applied && !result.like_mask_delivered &&
                   columns_equal_any(output->view().column(2), input.view().column(2), stream),
                 "projected LIKE above cap declines to ordinary decode");
        } else if (keep_mask) {
          // The consumer applies an outstanding positional mask before returning an
          // UNFILTERED table, which drops predicate-answer metadata: BOOL8 is declined
          // and the LIKE column reaches the caller as ordinary STRING values.
          expect(!result.applied && !result.like_mask_delivered &&
                   result.status == sc::scan_filter_status::declined_unselective &&
                   result.like_bool8_columns.empty() &&
                   columns_equal_any(output->view().column(2), input.view().column(2), stream),
                 "outstanding keep mask declines BOOL8 LIKE delivery to ordinary decode");
        } else {
          std::vector<std::uint8_t> expected(comments.size(), 1);
          std::fill_n(expected.begin(), 11, 0);
          expect(result.applied && result.like_mask_delivered &&
                   result.like_bool8_columns == std::vector<std::size_t>{2} &&
                   read<std::uint8_t>(output->view().column(2), stream) == expected &&
                   result.wave2_readbacks == 0,
                 "full BOOL8 is LIKE-only with no outstanding visibility mask");
          if (sc::decompression_pushdown_batched_wave2_enabled())
            expect(result.wave2_syncs == 0,
                   "fixed-width BOOL8 delivery separates final completion from host observations");
        }
      } else {
        std::vector<std::int32_t> kept;
        for (std::size_t i = 11; i < comments.size(); ++i)
          if (!keep_mask || i % 2 == 0) kept.push_back(i);
        expect(result.applied && !result.like_mask_delivered &&
                 output->num_columns() == (projected ? 3 : 2) &&
                 read<std::int32_t>(output->view().column(0), stream) == kept &&
                 result.keep_mask_applied == keep_mask,
               "mask switch off restores high-survival compaction and filter-only drop");
      }
    }
  }
}
}  // namespace

// A per-batch host read can leave an OnPair column unprepared. Every decode that reaches it must
// prepare a request-local copy, and a filtered decode that keeps no row must not prepare it at all.
void deferred_preparation(::cuda::stream_ref stream,
                          rmm::device_async_resource_ref mr,
                          char const* offsets_plan)
{
  std::vector<std::string> rows(4099, "other");
  rows[3]    = "google";
  rows[4000] = "google-x";
  std::vector<std::int32_t> numbers(rows.size());
  std::iota(numbers.begin(), numbers.end(), 0);
  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(upload<std::int32_t>(numbers, cudf::type_id::INT32, stream, mr));
  columns.push_back(make_strings_column(rows, {}, stream));
  columns.push_back(make_strings_column(rows, {}, stream));  // a LIKE source (above-cap case)
  cudf::table input{std::move(columns)};
  std::string const onpair_plan =
    std::string{
      "input -> onpair_16 -> codes, row_offsets, dict_bytes, dict_offsets, batch_prefixes\n"} +
    offsets_plan;
  auto stored = simpatico::compress_with_plan(
    input.view(), "input -> bitpack\n---\n" + onpair_plan + "---\n" + onpair_plan, stream, mr);
  std::vector<std::uint8_t> header;
  std::vector<simpatico::payload_buffer_ref> buffers;
  std::uint64_t bytes = 0;
  auto error = simpatico::build_compressed_table_header(stored, header, buffers, bytes, stream);
  expect(error.empty(), "deferred: header");
  std::vector<std::uint8_t> payload(bytes);
  for (auto const& b : buffers) {
    if (b.size_bytes != 0)
      cuda_check(cudaMemcpyAsync(payload.data() + b.offset,
                                 b.device_ptr,
                                 b.size_bytes,
                                 cudaMemcpyDeviceToHost,
                                 stream.get()));
  }
  stream.sync();
  simpatico::payload_fetch_fn fetch =
    [&](std::uint64_t offset, std::size_t size, void* dst, ::cuda::stream_ref s) {
      cuda_check(
        cudaMemcpyAsync(dst, payload.data() + offset, size, cudaMemcpyHostToDevice, s.get()));
    };
  std::array<std::size_t, 2> const selected{0, 1};
  // Reads the subset and reports the bytes the read itself allocated.
  auto read = [&](std::span<const bool> defer) {
    rmm::mr::statistics_resource_adaptor stats{
      cuda::mr::any_resource<cuda::mr::device_accessible>{mr}};
    std::string read_error;
    auto table = simpatico::read_compressed_table_subset_from_memory(
      header, fetch, selected, stream, rmm::device_async_resource_ref{stats}, &read_error, defer);
    stream.sync();
    expect(read_error.empty() && table.num_columns() == 2, "deferred: subset read");
    return std::pair{std::move(table), stats.get_bytes_counter().total};
  };
  auto const before_eager         = simpatico::onpair_preparation_count();
  auto [eager, eager_bytes]       = read({});
  auto const before_deferred      = simpatico::onpair_preparation_count();
  std::array<bool, 2> const defer = {false, true};
  auto [deferred, deferred_bytes] = read(defer);
  expect(before_deferred == before_eager + 1, "deferred: an eager read prepares its OnPair column");
  expect(simpatico::onpair_preparation_count() == before_deferred,
         "deferred: a deferred read prepares nothing");
  std::array<bool, 1> const mismatched = {true};
  std::string mismatch_error;
  (void)simpatico::read_compressed_table_subset_from_memory(
    header, fetch, selected, stream, mr, &mismatch_error, mismatched);
  expect(!mismatch_error.empty(), "deferred: a flag span must match the selected columns");
  auto const& tree = *deferred.columns[1].plan_tree;
  expect(eager.columns[1].plan_tree->nodes[1].prepared_rep != nullptr,
         "deferred: an eager read keeps the preparation");
  expect(tree.nodes[1].prepared_rep == nullptr, "deferred: the deferred column stays unprepared");
  expect(deferred_bytes < eager_bytes,
         "deferred: preparation's decoded channels are not allocated by the read");

  // A bare decode_one request (no table facade) prepares inside the request and publishes nothing.
  // Plain decode and a LIKE answer prepare on demand, once each, and match the input.
  auto const before_plain = simpatico::onpair_preparation_count();
  auto plain = simpatico::decode_one(simpatico::column_decode_request{std::cref(tree)}, stream, mr);
  expect(simpatico::onpair_preparation_count() == before_plain + 1,
         "deferred: a plain decode prepares once");
  expect(tree.nodes[1].prepared_rep == nullptr, "deferred: a bare request publishes nothing");
  auto all = make_strings_column(rows, {}, stream);
  expect(columns_equal_any(plain->view(), all->view(), stream), "deferred: plain decode");
  check(tree, plain->view(), like_predicate{{"google"}, false}, stream, mr);
  check(tree, plain->view(), like_predicate{{"google"}, true}, stream, mr);

  rmm::cuda_stream other;
  std::array<::cuda::stream_ref, 2> lanes{stream, other};
  bool const zero_skip = sc::decompression_pushdown_full_route_zero_skip_enabled();
  struct filter_case {
    char const* name;
    sc::decode_route route;
    std::int64_t lo, hi;
    std::vector<std::string> expected;  // ignored when declined: every row
    bool declined;
  };
  for (auto const& c :
       {filter_case{"no survivors", sc::decode_route::onpair, 5000, 6000, {}, false},
        filter_case{"no survivors, full route", sc::decode_route::full, 5000, 6000, {}, false},
        filter_case{"survivors", sc::decode_route::onpair, 3, 3, {"google"}, false},
        filter_case{"survivors, full route", sc::decode_route::full, 3, 3, {"google"}, false},
        filter_case{"declined", sc::decode_route::onpair, 0, 4098, {}, true}}) {
    std::string const label = std::string{"deferred ("} + c.name + "): ";
    // A decode that reaches a deferred column prepares it in place, so each case reads afresh.
    auto [batch, batch_bytes] = read(defer);
    (void)batch_bytes;
    auto const& batch_tree = *batch.columns[1].plan_tree;
    sc::scan_filter_request request;
    request.routes = {sc::decode_route::bitpack_mask, c.route};
    request.filters.push_back({0, {c.lo, c.hi}});
    rmm::mr::statistics_resource_adaptor stats{
      cuda::mr::any_resource<cuda::mr::device_accessible>{mr}};
    sc::scan_filter_result result;
    auto const before = simpatico::onpair_preparation_count();
    auto output       = simpatico::decompress_scan_filter(
      batch, selected, request, result, lanes, stream, rmm::device_async_resource_ref{stats});
    stream.sync();
    auto const prepared = simpatico::onpair_preparation_count() - before;
    if (c.declined) {
      expect(!result.applied && result.status == sc::scan_filter_status::declined_unselective &&
               output->num_rows() == static_cast<cudf::size_type>(rows.size()),
             label + "declined batch decodes plainly");
      expect(columns_equal_any(output->view().column(1), all->view(), stream),
             label + "plain values");
      expect(prepared == 1, label + "the plain decode prepares once");
      expect(batch_tree.nodes[1].prepared_rep != nullptr,
             label + "the plain decode prepares before submitting, not inside a lane request");
      continue;
    }
    expect(result.applied && output->num_rows() == static_cast<cudf::size_type>(c.expected.size()),
           label + "filtered shape");
    auto expected = make_strings_column(c.expected, {}, stream);
    expect(columns_equal_any(output->view().column(1), expected->view(), stream),
           label + "filtered values");
    // An empty batch prepares only when a full-route decode runs (zero skip off).
    bool const decodes = !c.expected.empty() || (c.route == sc::decode_route::full && !zero_skip);
    expect(prepared == (decodes ? 1u : 0u), label + "preparations match the decodes that ran");
    // Wave 2 prepares what it will decode before its first submission; a skipped column stays
    // unprepared.
    expect((batch_tree.nodes[1].prepared_rep != nullptr) == decodes,
           label + "wave 2 prepares exactly the outputs it decodes, up front");
    if (!c.expected.empty()) continue;
    expect(result.zero_skipped_full_columns ==
             (c.route == sc::decode_route::full && zero_skip ? 1u : 0u),
           label + "zero-skip counter");
    // Independently of the counters: beyond what the same selection allocates for the filter
    // column alone, a batch that prepares allocates at least the decoded row byte offsets, and one
    // that does not stays below.
    auto filter_only   = request;
    filter_only.routes = {sc::decode_route::bitpack_mask};
    std::array<std::size_t, 1> const filter_column{0};
    rmm::mr::statistics_resource_adaptor baseline{
      cuda::mr::any_resource<cuda::mr::device_accessible>{mr}};
    sc::scan_filter_result baseline_result;
    (void)simpatico::decompress_scan_filter(deferred,
                                            filter_column,
                                            filter_only,
                                            baseline_result,
                                            lanes,
                                            stream,
                                            rmm::device_async_resource_ref{baseline});
    stream.sync();
    auto const extra         = stats.get_bytes_counter().total - baseline.get_bytes_counter().total;
    auto const offsets_bytes = static_cast<std::int64_t>((rows.size() + 1) * sizeof(std::int32_t));
    expect(baseline_result.applied && (decodes ? extra >= offsets_bytes : extra < offsets_bytes),
           label + "allocations show whether preparation ran (" + std::to_string(extra) +
             " bytes beyond the filter column, offsets are " + std::to_string(offsets_bytes) + ")");
  }

  // Above the cap, a filter-only LIKE source (eager, as the engine reads it) delivers BOOL8 and the
  // deferred value output decodes full width: it too is prepared before submission. With the
  // BOOL8 switch off the request compacts instead (every row survives) and drops the filter-only
  // column; wave 2 prepares the value output the same way.
  {
    std::string const label = "deferred (LIKE above cap): ";
    std::array<std::size_t, 3> const with_source{0, 1, 2};
    std::array<bool, 3> const defer_value{false, true, false};
    std::string read_error;
    auto batch = simpatico::read_compressed_table_subset_from_memory(
      header, fetch, with_source, stream, mr, &read_error, defer_value);
    stream.sync();
    expect(read_error.empty() && batch.num_columns() == 3, label + "subset read");
    expect(batch.columns[2].plan_tree->nodes[1].prepared_rep != nullptr &&
             batch.columns[1].plan_tree->nodes[1].prepared_rep == nullptr,
           label + "only the value output is deferred");
    sc::scan_filter_request request;
    request.routes = {
      sc::decode_route::bitpack_mask, sc::decode_route::onpair, sc::decode_route::onpair};
    request.like_filters       = {{2, {{"o"}, false}}};  // every row
    request.delivered_prefix   = 2;
    request.like_bool8_columns = {2};
    sc::scan_filter_result result;
    auto const before = simpatico::onpair_preparation_count();
    auto output =
      simpatico::decompress_scan_filter(batch, with_source, request, result, lanes, stream, mr);
    stream.sync();
    bool const bool8 = sc::onpair_like_mask_above_cap_enabled();
    expect(result.applied && result.like_mask_delivered == bool8 &&
             output->num_columns() == (bool8 ? 3 : 2) &&
             output->num_rows() == static_cast<cudf::size_type>(rows.size()),
           label + (bool8 ? "BOOL8 delivery" : "compacted, filter-only source dropped"));
    expect(columns_equal_any(output->view().column(1), all->view(), stream),
           label + "full-width values");
    expect(simpatico::onpair_preparation_count() == before + 1 &&
             batch.columns[1].plan_tree->nodes[1].prepared_rep != nullptr,
           label + "the value output is prepared once, in place");
  }
}

void deferred_preparation(::cuda::stream_ref stream, rmm::device_async_resource_ref mr)
{
  // Stored offsets (borrowed by preparation), the benchmark's delta -> bitpack tail, and an ANS
  // tail (whose frame layouts a deferred read does not cache).
  for (char const* offsets : {"",
                              "onpair_16.row_offsets -> delta -> differences\n"
                              "onpair_16.row_offsets.differences -> bitpack\n",
                              "onpair_16.row_offsets -> delta -> differences\n"
                              "onpair_16.row_offsets.differences -> ans\n"})
    deferred_preparation(stream, mr, offsets);
}

int main(int argc, char** argv)
{
  try {
    cuda_check(cudaSetDevice(0));
    rmm::cuda_stream stream;
    rmm::mr::cuda_async_memory_resource resource;
    auto const mr = rmm::device_async_resource_ref{resource};
    simpatico::onpair_like_log_scope log{log_stats};
    if (argc == 2 && std::string(argv[1]) == "--mask") {
      mask_above_cap_regressions(stream, mr);
      std::puts("test_onpair_like mask: PASS");
      return 0;
    }
    if (argc == 2 && std::string(argv[1]) == "--cost") {
      cost_regression(stream, mr);
      std::puts("test_onpair_like cost: PASS");
      return 0;
    }
    if (argc == 2 && std::string(argv[1]) == "--deferred") {
      deferred_preparation(stream, mr);
      std::puts("test_onpair_like deferred: PASS");
      return 0;
    }
    if (argc == 2 && std::string(argv[1]) == "--wave2") {
      wave2_regressions(stream, mr);
      ordered_peak_regression(stream, mr);
      std::puts("test_onpair_like wave2: PASS");
      return 0;
    }
    if (argc == 2 && std::string_view{argv[1]} == "--k6-batch-boundary") {
      expected_width = 192;
      k6_like_batch_boundary(stream, mr);
      return 0;
    }
    bool const k6 = argc == 2 && std::string_view{argv[1]} == "--k6";
    if (k6) expected_width = 192;
    bool const off = argc == 2 && std::string(argv[1]) == "--off";
    expect(simpatico::onpair_like_enabled() != off, "test process switch");
    preflight(stream, mr, off);
    if (!off) {
      correctness(stream, mr);
      deterministic_continuation(stream, mr, k6 ? 192 : 128);
      instrumentation_and_shared_bounds(stream, mr);
      corrupt_bounds(stream, mr);
      wave(stream, mr);
      expect(log_count > 0, "LIKE completion diagnostics emitted");
    }
    stream.synchronize();
    std::puts("test_onpair_like: PASS");
    return 0;
  } catch (std::exception const& error) {
    std::fprintf(stderr, "test_onpair_like: FAIL: %s\n", error.what());
    return 1;
  }
}
