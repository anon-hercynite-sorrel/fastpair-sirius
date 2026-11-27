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

/**
 * @file test_fused_scan_filter_delivery.cpp
 * @brief Verify filtered output prefixes and retention of columns needed by residual filters.
 *
 * Codec cases check delivered_prefix; engine cases drop trailing filter-only columns only when the
 * scan's whole filter was applied. Join-filter acceptance or refusal does not complete a partially
 * applied scan filter. Nullable selected values decline compaction and preserve the original rows
 * and validity.
 */

#include "api/simpatico_codegen.hpp"
#include "codegen/plan/plan_interpreter.hpp"
#include "codegen/selection/decompression_pushdown_policy.hpp"
#include "codegen/selection/selection.hpp"
#include "codegen/util/stream_pool.hpp"
#include "scan/fused_scan_filter_test_utils.hpp"

#include <cudf/column/column_view.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/strings/contains.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/unary.hpp>
#include <cudf/utilities/bit.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <cuda_runtime.h>

#include <catch.hpp>
#include <compression/compressed_scan.hpp>
#include <cucascade/memory/memory_space.hpp>
#include <cucascade/memory/reservation_aware_resource_adaptor.hpp>
#include <expression_evaluator/like_multiliteral.hpp>
#include <util/decoded_like.hpp>
#include <util/host_observation.hpp>
#include <utils/like_matcher_trace.hpp>
#include <utils/log_test_utils.hpp>

#include <algorithm>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <string>
#include <utility>
#include <vector>

using namespace sirius::test::fused_scan_filter;

TEST_CASE("Fused LIKE cost rejects recorded Q23 routes and accepts exempt narrow policy shapes",
          "[scan][onpair_like][fused_scan_filter][cost_rule]")
{
  namespace sc = sirius::codegen;
  std::vector<sc::decode_route> wide(68, sc::decode_route::full);
  wide.insert(wide.end(), 27, sc::decode_route::bitpack_mask);
  wide.insert(wide.end(), 1, sc::decode_route::dict_codes);
  wide.insert(wide.end(), 4, sc::decode_route::str_split);
  wide.insert(wide.end(), 5, sc::decode_route::onpair);
  REQUIRE(wide.size() == 105);
  sc::fused_scan_cost_constants c;  // Explicit defaults, independent of G4 env overrides.
  // First 27 recorded Q23 (rows, survivors) pairs from clickbench-17-0a9852da.
  std::vector<std::pair<std::int64_t, std::int64_t>> chunks{
    {4424861, 434}, {3809280, 1299}, {4794356, 292},  {4056989, 148}, {4300800, 1094},
    {3319630, 361}, {3809280, 167},  {4055040, 168},  {2457600, 253}, {4177920, 293},
    {4300800, 396}, {3440640, 242},  {2591688, 2592}, {4055040, 183}, {2951044, 1387},
    {3811604, 240}, {3194880, 289},  {4300800, 357},  {3814698, 566}, {3072000, 454},
    {4177920, 482}, {4177920, 363},  {3569737, 700},  {3194880, 390}, {3441525, 905},
    {3689261, 941}, {3007304, 915}};
  for (auto [rows, survivors] : chunks) {
    for (double fraction : {0.0, double(survivors) / rows}) {
      auto cost = sc::estimate_fused_scan_cost(wide, rows, fraction, 1, c);
      CHECK(cost.compact_ms == Catch::Approx(31.59 + 68 * rows * fraction * 1e-8));
      CHECK(cost.plain_ms ==
            Catch::Approx(0.15 + rows * 1e-7 + 37 * rows * (1 - fraction) * 1.5e-8));
      CHECK_FALSE(cost.profitable());
      CHECK(sc::fused_scan_cost_refused(wide, rows, fraction, 1, c) ==
            sc::decompression_pushdown_cost_rule_enabled());
    }
  }
  // These assertions prove policy acceptance for <=4 delivered columns across route types,
  // selectivities and coefficients. They do not assert an arithmetic profitability bound.
  for (std::size_t width = 1; width <= 4; ++width) {
    for (auto route : {sc::decode_route::full,
                       sc::decode_route::onpair,
                       sc::decode_route::str_split,
                       sc::decode_route::dict_codes,
                       sc::decode_route::bitpack_mask,
                       sc::decode_route::delta_mask}) {
      std::vector<sc::decode_route> narrow(width, route);
      for (double per_column : {0.0, 300.0, 2000.0}) {
        c.per_column_us = per_column;
        for (double fraction : {0.0, 150.0 / 4'000'000, 500.0 / 4'000'000})
          CHECK_FALSE(sc::fused_scan_cost_refused(narrow, 4'000'000, fraction, 1, c));
      }
    }
  }
  // Explicitly demonstrate the distinction: policy accepts even this unprofitable estimate.
  c.per_column_us         = 2000;
  auto const narrow_full  = std::vector<sc::decode_route>(4, sc::decode_route::full);
  auto const unprofitable = sc::estimate_fused_scan_cost(narrow_full, 4'000'000, 0.0001, 1, c);
  CHECK(unprofitable.compact_ms == Catch::Approx(8.000016));
  CHECK(unprofitable.plain_ms == Catch::Approx(0.55));
  CHECK_FALSE(unprofitable.profitable());
  CHECK_FALSE(sc::fused_scan_cost_refused(narrow_full, 4'000'000, 0.0001, 1, c));
  // Disabling the exemption exposes the formula; COST_RULE=0 still bypasses it.
  c.narrow_columns = 0;
  CHECK(sc::fused_scan_cost_refused(narrow_full, 4'000'000, 0.0001, 1, c) ==
        sc::decompression_pushdown_cost_rule_enabled());
}

TEST_CASE("Decoded LIKE pays the match cost on both scan routes",
          "[scan][fused_scan_filter][cost_rule]")
{
  namespace sc = sirius::codegen;
  std::vector<sc::decode_route> routes(5, sc::decode_route::full);
  sc::fused_scan_cost_constants c;
  c.per_like_us     = 2000;
  auto const codes  = sc::estimate_fused_scan_cost(routes, 200, 0, 1, c);
  auto const values = sc::estimate_fused_scan_cost(routes, 200, 0, 1, c, 1);
  CHECK(values.plain_ms == codes.plain_ms);
  CHECK(values.compact_ms == Catch::Approx(codes.compact_ms + values.plain_ms));
  CHECK_FALSE(sc::fused_scan_cost_refused(routes, 200, 0, 1, c));
  CHECK(sc::fused_scan_cost_refused(routes, 200, 0, 1, c, 1) ==
        sc::decompression_pushdown_cost_rule_enabled());
}

TEST_CASE("Fused LIKE cost bounds overrides and declines non-finite estimates",
          "[scan][fused_scan_filter][cost_rule]")
{
  namespace sc = sirius::codegen;
  for (double maximum : {sc::fused_scan_max_fixed_us,
                         sc::fused_scan_max_row_ns,
                         sc::fused_scan_max_narrow_columns}) {
    CHECK(sc::parse_fused_scan_cost_coefficient("0", maximum) == 0);
    CHECK(sc::parse_fused_scan_cost_coefficient(std::to_string(maximum), maximum) == maximum);
    for (auto text : {"", "-1", "nan", "inf", "1e308", "1e309", "1garbage"})
      CHECK_THROWS_AS(sc::parse_fused_scan_cost_coefficient(text, maximum), std::invalid_argument);
    CHECK_THROWS_AS(sc::parse_fused_scan_cost_coefficient(std::to_string(maximum + 1), maximum),
                    std::invalid_argument);
  }
  std::vector<sc::decode_route> routes(5, sc::decode_route::full);
  routes.push_back(sc::decode_route::onpair);
  for (double fraction : {0.0, 1.0}) {
    for (double value : {0.0, sc::fused_scan_max_row_ns}) {
      sc::fused_scan_cost_constants c;
      c.per_column_us = c.per_selected_string_us = c.per_like_us = value;
      c.per_row_ns = c.per_full_survivor_ns = c.per_avoided_row_ns = value;
      auto cost = sc::estimate_fused_scan_cost(routes,
                                               std::numeric_limits<std::int64_t>::max(),
                                               fraction,
                                               std::numeric_limits<std::size_t>::max(),
                                               c);
      CHECK(cost.finite());
      CHECK(cost.compact_ms >= 0);
      CHECK(cost.plain_ms >= 0);
    }
  }
  // Explicit constants bypass parsing to exercise the defensive arithmetic check on either side.
  for (int side : {0, 1, 2}) {
    sc::fused_scan_cost_constants c;
    c.narrow_columns = 0;
    if (side != 1) {
      c.per_column_us        = std::numeric_limits<double>::max();
      c.per_full_survivor_ns = std::numeric_limits<double>::max();
    }
    if (side != 0) c.per_row_ns = std::numeric_limits<double>::max();
    auto cost = sc::estimate_fused_scan_cost(routes, 4'000'000, 0.01, 1, c);
    CHECK_FALSE(cost.finite());
    CHECK_FALSE(cost.profitable());
    CHECK_FALSE(sc::fused_scan_cost_refused(routes, 4'000'000, 0.01, 1, c));
  }
  sc::fused_scan_cost nonfinite[] = {
    {std::numeric_limits<double>::quiet_NaN(), 1},
    {1, std::numeric_limits<double>::quiet_NaN()},
    {std::numeric_limits<double>::infinity(), std::numeric_limits<double>::infinity()}};
  for (auto cost : nonfinite) {
    CHECK_FALSE(cost.finite());
    CHECK_FALSE(cost.profitable());
  }
}

namespace {

namespace sc = sirius::codegen;

constexpr char const* kEqualitySourcePlans =
  "input -> bitpack -> chunk_min, chunk_count, chunk_bits, packed\n"
  "---\n"
  "input -> bitpack -> chunk_min, chunk_count, chunk_bits, packed\n"
  "---\n"
  "input -> dictionary -> keys_offsets, keys_chars, indices\n"
  "dictionary.indices -> bitpack -> chunk_min, chunk_count, chunk_bits, packed\n";

/// "n" is dictionary-encoded with bitpacked codes, which decodes on the compacted dict_codes route.
constexpr char const* kStringSourcePlans =
  "input -> bitpack -> chunk_min, chunk_count, chunk_bits, packed\n"
  "---\n"
  "input -> dictionary -> keys_offsets, keys_chars, indices\n"
  "dictionary.indices -> bitpack -> chunk_min, chunk_count, chunk_bits, packed\n"
  "---\n"
  "input -> bitpack -> chunk_min, chunk_count, chunk_bits, packed\n";

/// The validity of "n" travels beside its encoded column (plan/validity.cpp), so the plan names no
/// null_mask channel; "n" decodes full width and is then gathered to the survivors.
constexpr char const* kNullableStringSourcePlans =
  "input -> bitpack -> chunk_min, chunk_count, chunk_bits, packed\n"
  "---\n"
  "input -> dictionary -> keys_offsets, keys_chars, indices\n"
  "dictionary.indices -> bitpack -> chunk_min, chunk_count, chunk_bits, packed\n"
  "---\n"
  "input -> bitpack -> chunk_min, chunk_count, chunk_bits, packed\n";

bool every_row(cudf::size_type) { return true; }

bool null_in_nullable_source(cudf::size_type i) { return i % 7 == 0; }

simpatico::compressed_table compress_equality_source(::cuda::stream_ref stream)
{
  std::vector<std::int32_t> v(kRows);
  std::vector<std::int32_t> k(kRows);
  std::vector<std::string> s(kRows);
  for (cudf::size_type i = 0; i < kRows; ++i) {
    v[static_cast<std::size_t>(i)] = i % 100;
    k[static_cast<std::size_t>(i)] = i % 50;
    s[static_cast<std::size_t>(i)] = key_string(i);
  }
  std::vector<std::unique_ptr<cudf::column>> cols;
  cols.push_back(upload_column(v, stream));
  cols.push_back(upload_column(k, stream));
  cols.push_back(upload_strings(s, stream));
  stream.sync();
  cudf::table const table{std::move(cols)};
  return simpatico::compress_with_plan(
    table.view(), kEqualitySourcePlans, stream, cudf::get_current_device_resource_ref());
}

simpatico::compressed_table compress_string_source(bool nullable, ::cuda::stream_ref stream)
{
  std::vector<std::int32_t> v(kRows);
  std::vector<std::string> n(kRows);
  std::vector<std::int32_t> f(kRows);
  std::vector<bool> valid;
  if (nullable) { valid.resize(kRows); }
  for (cudf::size_type i = 0; i < kRows; ++i) {
    auto const row     = static_cast<std::size_t>(i);
    bool const is_null = nullable && null_in_nullable_source(i);
    v[row]             = i % 100;
    n[row]             = is_null ? std::string{} : key_string(i);
    f[row]             = i % kFilterModulus;
    if (nullable) { valid[row] = !is_null; }
  }
  std::vector<std::unique_ptr<cudf::column>> cols;
  cols.push_back(upload_column(v, stream));
  cols.push_back(upload_strings(n, stream, valid));
  cols.push_back(upload_column(f, stream));
  stream.sync();
  cudf::table const table{std::move(cols)};
  return simpatico::compress_with_plan(table.view(),
                                       nullable ? kNullableStringSourcePlans : kStringSourcePlans,
                                       stream,
                                       cudf::get_current_device_resource_ref());
}

/// The rows of a string source that @p keep selects, column by column ("n" is nullopt where null).
struct string_source_rows {
  std::vector<std::int32_t> v;
  std::vector<std::optional<std::string>> n;
  std::vector<std::int32_t> f;
};

string_source_rows string_rows_where(std::predicate<cudf::size_type> auto keep, bool nullable)
{
  string_source_rows rows;
  for (cudf::size_type i = 0; i < kRows; ++i) {
    if (keep(i)) {
      rows.v.push_back(i % 100);
      rows.n.push_back(nullable && null_in_nullable_source(i)
                         ? std::nullopt
                         : std::optional<std::string>{key_string(i)});
      rows.f.push_back(i % kFilterModulus);
    }
  }
  return rows;
}

/// The values of STRING column @p col, with nullopt for each null row.
std::vector<std::optional<std::string>> nullable_strings_to_host(cudf::column_view const& col,
                                                                 ::cuda::stream_ref stream)
{
  auto const values = strings_to_host(col, stream);
  std::vector<std::optional<std::string>> out(values.begin(), values.end());
  if (!col.nullable()) { return out; }
  std::vector<cudf::bitmask_type> words(
    static_cast<std::size_t>(cudf::num_bitmask_words(col.offset() + col.size())));
  REQUIRE(cudaMemcpyAsync(words.data(),
                          col.null_mask(),
                          words.size() * sizeof(cudf::bitmask_type),
                          cudaMemcpyDeviceToHost,
                          stream.get()) == cudaSuccess);
  stream.sync();
  for (cudf::size_type i = 0; i < col.size(); ++i) {
    if (!cudf::bit_is_set(words.data(), col.offset() + i)) {
      out[static_cast<std::size_t>(i)].reset();
    }
  }
  return out;
}

/// @p out holds exactly the first @p width columns of @p rows, values included.
void require_columns(cudf::table_view out,
                     source_rows const& rows,
                     std::size_t width,
                     ::cuda::stream_ref stream)
{
  REQUIRE(static_cast<std::size_t>(out.num_columns()) == width);
  CHECK(column_to_host(out.column(0), stream) == rows.v);
  CHECK(column_to_host(out.column(1), stream) == rows.k);
  if (width > kOutputWidth) { CHECK(column_to_host(out.column(2), stream) == rows.f); }
}

/// @p out holds exactly the first @p width columns of @p rows, values and nulls included.
void require_columns(cudf::table_view out,
                     string_source_rows const& rows,
                     std::size_t width,
                     ::cuda::stream_ref stream)
{
  REQUIRE(static_cast<std::size_t>(out.num_columns()) == width);
  CHECK(column_to_host(out.column(0), stream) == rows.v);
  CHECK(nullable_strings_to_host(out.column(1), stream) == rows.n);
  if (width > kOutputWidth) { CHECK(column_to_host(out.column(2), stream) == rows.f); }
}

/// A codec request whose only source is the range on "f", with every column on its bitpack route.
sc::scan_filter_request range_request(std::optional<std::size_t> delivered_prefix,
                                      sc::range_predicate range = {kFilterLo, kFilterHi})
{
  sc::scan_filter_request request;
  request.filters.push_back({2, range});
  request.routes = {
    sc::decode_route::bitpack_mask, sc::decode_route::bitpack_mask, sc::decode_route::bitpack_mask};
  request.delivered_prefix = delivered_prefix;
  return request;
}

/// A codec request whose only source is an equality on "s" over @p keys, delivering "v" and "k".
sc::scan_filter_request equality_request(std::vector<std::string> keys)
{
  sc::scan_filter_request request;
  request.bool8_filters.push_back({2, std::move(keys)});
  request.routes = {
    sc::decode_route::bitpack_mask, sc::decode_route::bitpack_mask, sc::decode_route::dict_codes};
  request.delivered_prefix = kOutputWidth;
  return request;
}

struct decode_environment {
  using tracking = cucascade::memory::reservation_aware_resource_adaptor::AllocationTrackingScope;

  static cucascade::memory::gpu_memory_space_config config(bool per_stream)
  {
    int device = -1;
    REQUIRE(cudaGetDevice(&device) == cudaSuccess);
    cucascade::memory::gpu_memory_space_config config;
    config.device_id              = device;
    config.memory_capacity        = 16U << 20;
    config.per_stream_reservation = per_stream;
    return config;
  }

  explicit decode_environment(::cuda::stream_ref stream, bool per_stream = false)
    : space{config(per_stream)},
      stream{stream},
      tracking_scope{per_stream ? tracking::PER_STREAM : tracking::PER_THREAD}
  {
    auto* allocator =
      space.get_memory_resource_as<cucascade::memory::reservation_aware_resource_adaptor>();
    REQUIRE(allocator != nullptr);
    auto reservation = space.make_reservation(8U << 20);
    REQUIRE(reservation != nullptr);
    REQUIRE(allocator->attach_reservation_to_tracker(stream, std::move(reservation)));
  }

  ~decode_environment()
  {
    space.get_memory_resource_as<cucascade::memory::reservation_aware_resource_adaptor>()
      ->reset_stream_reservation(stream);
  }

  cucascade::memory::memory_space space;
  ::cuda::stream_ref stream;
  tracking tracking_scope;
};

sirius::decompress_result decompress_with(decode_environment& environment,
                                          simpatico::compressed_table const& chunk,
                                          sirius::pushdown_request request,
                                          ::cuda::stream_ref stream)
{
  sirius::decompression_pushdown_scan const scan{std::move(request)};
  return sirius::decompress_chunk(chunk,
                                  kSelected,
                                  &scan,
                                  sirius::decode_visibility_mask{},
                                  environment.space,
                                  environment.tracking_scope,
                                  stream,
                                  environment.space.get_default_allocator());
}

/// Join filters per selected slot, as a scan attaches them to a batch.
using slot_probes = std::vector<std::vector<sirius::membership_probe>>;

/// A join filter backed by @p keys and ordered by @p selectivity_rank, which defaults to the rank a
/// snapshot gives an exact hash set.
sirius::membership_probe key_set_probe(
  std::shared_ptr<sirius::op::sirius_dynamic_in_list_filter> keys,
  std::uint8_t selectivity_rank = 1)
{
  auto const num_keys = static_cast<std::uint64_t>(keys->size());
  return {[keys = std::move(keys)](cudf::column_view const& column,
                                   std::uint32_t const* prior_mask_words,
                                   ::cuda::stream_ref s,
                                   rmm::device_async_resource_ref mr) {
            return keys->compute_mask(column, prior_mask_words, /*device_id=*/0, s, mr);
          },
          selectivity_rank,
          num_keys};
}

/// A join filter that declines every batch, as one does when the slot's type is outside its key
/// family.
sirius::membership_probe declining_probe()
{
  return {[](cudf::column_view const&,
             std::uint32_t const*,
             ::cuda::stream_ref,
             rmm::device_async_resource_ref) { return std::unique_ptr<cudf::column>{}; },
          /*selectivity_rank=*/1,
          /*num_keys=*/10};
}

/// @p probe as the only join filter, on @p slot.
slot_probes one_probe(std::size_t slot, sirius::membership_probe probe)
{
  slot_probes probes(kSelected.size());
  probes[slot].push_back(std::move(probe));
  return probes;
}

/// Decodes @p chunk under @p request with @p probes attached through
/// `decompression_pushdown_scan::with_membership_probes`, the way a scan refreshes a batch's join
/// filters.
sirius::decompress_result decompress_with_probes(
  decode_environment& environment,
  simpatico::compressed_table const& chunk,
  sirius::pushdown_request request,
  slot_probes probes,
  ::cuda::stream_ref stream,
  sirius::decode_visibility_mask const& keep_mask    = {},
  sirius::membership_measurement_request measurement = {})
{
  auto const scan = sirius::decompression_pushdown_scan{std::move(request)}.with_membership_probes(
    std::move(probes), /*generation=*/1, measurement);
  return sirius::decompress_chunk(chunk,
                                  kSelected,
                                  scan.get(),
                                  keep_mask,
                                  environment.space,
                                  environment.tracking_scope,
                                  stream,
                                  environment.space.get_default_allocator());
}

/// The codec and the engine read the same env gate; an explicit SIRIUS_EXP_FUSED_SCAN_FILTER=0
/// turns every filtered decode into a plain one.
bool gate_off()
{
  if (sc::decompression_pushdown_enabled()) { return false; }
  WARN("filtered-decode env gate off in this process; skipping delivery coverage");
  return true;
}

}  // namespace

//===----------------------------------------------------------------------===//
// Codec: simpatico::decompress_scan_filter
//===----------------------------------------------------------------------===//

TEST_CASE("an applied filtered decode delivers only its prefix", "[fused_scan_filter][delivery]")
{
  if (gate_off()) { return; }
  ::cuda::stream_ref const stream = cudf::get_default_stream();
  auto const mr                   = cudf::get_current_device_resource_ref();
  auto const ct                   = compress_range_source(stream);
  simpatico::stream_pool pool;
  REQUIRE(pool.init(4));

  sc::scan_filter_result result;
  auto out = simpatico::decompress_scan_filter(
    ct, kSelected, range_request(kOutputWidth), result, pool, stream, mr);

  REQUIRE(result.applied);
  auto const expected = rows_where(in_filter_range);
  REQUIRE(result.survivor_count == static_cast<std::int64_t>(expected.v.size()));
  CHECK(result.routes.size() == kOutputWidth);
  require_columns(out->view(), expected, kOutputWidth, stream);
}

TEST_CASE("a filtered decode with no survivors delivers empty prefix columns",
          "[fused_scan_filter][delivery]")
{
  if (gate_off()) { return; }
  ::cuda::stream_ref const stream = cudf::get_default_stream();
  auto const mr                   = cudf::get_current_device_resource_ref();
  auto const ct                   = compress_range_source(stream);
  simpatico::stream_pool pool;
  REQUIRE(pool.init(4));

  // "f" never exceeds 999, so this range keeps nothing.
  sc::scan_filter_result result;
  auto out = simpatico::decompress_scan_filter(
    ct, kSelected, range_request(kOutputWidth, {2000, 3000}), result, pool, stream, mr);

  REQUIRE(result.applied);
  CHECK(result.survivor_count == 0);
  REQUIRE(static_cast<std::size_t>(out->num_columns()) == kOutputWidth);
  for (cudf::size_type c = 0; c < out->num_columns(); ++c) {
    CHECK(out->view().column(c).size() == 0);
    CHECK(out->view().column(c).type() == cudf::data_type{cudf::type_id::INT32});
  }
}

TEST_CASE("a filtered decode over a null-masked column returns every selected column",
          "[fused_scan_filter][delivery]")
{
  if (gate_off()) { return; }
  ::cuda::stream_ref const stream = cudf::get_default_stream();
  auto const mr                   = cudf::get_current_device_resource_ref();
  auto const ct                   = compress_string_source(/*nullable=*/true, stream);
  simpatico::stream_pool pool;
  REQUIRE(pool.init(4));

  // A compacted decode cannot carry the stored bitmask of "n" (it spans the whole chunk), so the
  // filtered decode is refused before any device work and every column decodes in full.
  auto request      = range_request(kOutputWidth);
  request.routes[1] = sc::decode_route::full;
  sc::scan_filter_result result;
  auto out = simpatico::decompress_scan_filter(ct, kSelected, request, result, pool, stream, mr);

  REQUIRE_FALSE(result.applied);
  REQUIRE(result.status == sc::scan_filter_status::refused);
  require_columns(
    out->view(), string_rows_where(every_row, /*nullable=*/true), kSelected.size(), stream);
}

TEST_CASE("a delivered prefix outside [1, selected] refuses the filtered decode",
          "[fused_scan_filter][delivery]")
{
  // With the gate off every request is refused, which would pass this case for the wrong reason.
  if (gate_off()) { return; }
  ::cuda::stream_ref const stream = cudf::get_default_stream();
  auto const mr                   = cudf::get_current_device_resource_ref();
  auto const ct                   = compress_range_source(stream);
  simpatico::stream_pool pool;
  REQUIRE(pool.init(4));

  auto const prefix = GENERATE(std::size_t{0}, std::size_t{4});
  CAPTURE(prefix);
  sc::scan_filter_result result;
  auto out = simpatico::decompress_scan_filter(
    ct, kSelected, range_request(prefix), result, pool, stream, mr);

  REQUIRE_FALSE(result.applied);
  CHECK(result.status == sc::scan_filter_status::refused);
  require_columns(out->view(), rows_where(every_row), kSelected.size(), stream);
}

TEST_CASE("an equality source past the delivered prefix is applied but not delivered",
          "[fused_scan_filter][delivery]")
{
  if (gate_off()) { return; }
  ::cuda::stream_ref const stream = cudf::get_default_stream();
  auto const mr                   = cudf::get_current_device_resource_ref();
  auto const ct                   = compress_equality_source(stream);
  simpatico::stream_pool pool;
  REQUIRE(pool.init(4));

  sc::scan_filter_result result;
  auto out = simpatico::decompress_scan_filter(
    ct, kSelected, equality_request({key_string(5)}), result, pool, stream, mr);

  REQUIRE(result.applied);
  auto const expected = rows_where([](cudf::size_type i) { return i % 50 == 5; });
  REQUIRE(result.survivor_count == static_cast<std::int64_t>(expected.v.size()));
  // The BOOL8 answer belongs to the dropped slot, so nothing delivered is BOOL8.
  require_columns(out->view(), expected, kOutputWidth, stream);
}

TEST_CASE("an applied decode builds no survivor map when only a dropped column would read it",
          "[fused_scan_filter][delivery]")
{
  if (gate_off()) { return; }
  ::cuda::stream_ref const stream = cudf::get_default_stream();
  auto const mr                   = cudf::get_current_device_resource_ref();
  auto const ct                   = compress_equality_source(stream);
  simpatico::stream_pool pool;
  REQUIRE(pool.init(4));

  // Ten of the fifty keys keep 20% of the rows. Above the index-walk ceiling the delivered bitpack
  // columns walk the mask bits, which leaves the dropped BOOL8 answer as the map's only reader.
  REQUIRE(sc::decompression_pushdown_index_walk_max_selectivity() < 0.2);
  std::vector<std::string> keys;
  for (cudf::size_type k = 0; k < 10; ++k) {
    keys.push_back(key_string(k));
  }
  sc::scan_filter_result result;
  auto out = simpatico::decompress_scan_filter(
    ct, kSelected, equality_request(std::move(keys)), result, pool, stream, mr);

  REQUIRE(result.applied);
  CHECK(result.row_indices.size() == 0);
  auto const expected = rows_where([](cudf::size_type i) { return i % 50 < 10; });
  REQUIRE(result.survivor_count == static_cast<std::int64_t>(expected.v.size()));
  require_columns(out->view(), expected, kOutputWidth, stream);
}

TEST_CASE("a delivered prefix covering every selected column changes nothing",
          "[fused_scan_filter][delivery]")
{
  if (gate_off()) { return; }
  ::cuda::stream_ref const stream = cudf::get_default_stream();
  auto const mr                   = cudf::get_current_device_resource_ref();
  auto const ct                   = compress_range_source(stream);
  simpatico::stream_pool pool;
  REQUIRE(pool.init(4));

  sc::scan_filter_result unset_result;
  auto unset = simpatico::decompress_scan_filter(
    ct, kSelected, range_request(std::nullopt), unset_result, pool, stream, mr);
  sc::scan_filter_result full_result;
  auto full = simpatico::decompress_scan_filter(
    ct, kSelected, range_request(kSelected.size()), full_result, pool, stream, mr);

  REQUIRE(unset_result.applied);
  REQUIRE(full_result.applied);
  CHECK(full_result.survivor_count == unset_result.survivor_count);
  CHECK(full_result.routes == unset_result.routes);
  auto const expected = rows_where(in_filter_range);
  require_columns(unset->view(), expected, kSelected.size(), stream);
  require_columns(full->view(), expected, kSelected.size(), stream);
}

//===----------------------------------------------------------------------===//
// Engine: sirius::decompress_chunk
//===----------------------------------------------------------------------===//

TEST_CASE("Decoded LIKE shares matcher dispatch and preserves per-conjunct counts",
          "[scan][decoded_like_sql]")
{
  if (gate_off()) return;
  auto const stream = cudf::get_default_stream();
  auto const mr     = cudf::get_current_device_resource_ref();
  decode_environment environment{stream};
  std::string const unicode = "café";  // legal substring outside the ASCII SWAR alphabet
  auto ids                  = upload_column<std::int32_t>({0, 1, 2, 3, 4, 5}, stream);
  auto keys                 = upload_strings({"red" + unicode, "red", unicode, "", "red", "other"},
                             stream,
                                             {true, true, true, true, false, true});
  REQUIRE(nullable_strings_to_host(keys->view(), stream) ==
          std::vector<std::optional<std::string>>{
            "red" + unicode, "red", unicode, "", std::nullopt, "other"});
  auto stored = simpatico::compress_with_plan(cudf::table_view{{ids->view(), keys->view()}},
                                              "input -> bitpack\n---\ninput -> str_split\n",
                                              stream,
                                              mr);
  std::array<std::size_t, 2> const selected{0, 1};
  for (bool swar : {false, true}) {
    for (bool count_only : {false, true}) {
      CAPTURE(swar, count_only);
      sirius::pushdown_request request;
      request.columns.resize(2);
      // Supply typed decoder predicates directly. SQL admission continues to leave
      // non-ASCII LIKE patterns with the residual evaluator.
      request.columns[1].likes          = {{{"red"}, false}, {{unicode}, true}};
      request.ranges_cover_whole_filter = true;
      if (count_only)
        request.row_count_only = true;
      else
        request.output_prefix_width = 1;
      if (swar) request.decoded_like_cache = std::make_shared<sirius::like_multiliteral_cache>();
      auto scan = sirius::decompression_pushdown_scan{request}.for_chunk(stored, selected);
      sirius::test::scoped_recording_log_sink logs{"info"};
      auto decoded       = sirius::decompress_chunk(stored,
                                              selected,
                                              scan.get(),
                                                    {},
                                              environment.space,
                                              environment.tracking_scope,
                                              stream,
                                              environment.space.get_default_allocator());
      bool const applied = simpatico::decoded_like_enabled();
      REQUIRE(decoded.outcome.row_filtered == applied);
      CHECK(decoded.outcome.row_count_carrier == (applied && count_only));
      std::vector<std::string> routes;
      for (auto const& record : logs.records()) {
        if (!record.message.starts_with("[decoded-like]")) continue;
        auto pos = record.message.find("matcher=");
        REQUIRE(pos != std::string::npos);
        routes.push_back(record.message.substr(pos));
      }
      if (applied) {
        REQUIRE(routes == std::vector<std::string>{swar ? "matcher=swar rows=6 matches=2"
                                                        : "matcher=cudf_like rows=6 matches=2",
                                                   "matcher=cudf_like rows=6 matches=3"});
        REQUIRE(decoded.table->num_columns() == 1);
      } else {
        REQUIRE(routes.empty());
        REQUIRE(decoded.table->num_columns() == 2);
        for (auto const& like : request.columns[1].likes) {
          auto view  = decoded.table->view();
          auto flags = cudf::strings::like(cudf::strings_column_view{view.column(1)},
                                           "%" + like.literals[0] + "%",
                                           "",
                                           stream,
                                           mr);
          if (like.invert)
            flags = cudf::unary_operation(flags->view(), cudf::unary_operator::NOT, stream, mr);
          auto filtered = cudf::apply_boolean_mask(view, flags->view(), stream, mr);
          // The full decode owns buffers on its producer lanes. Finish the residual's
          // reads on this stream before replacing their owner.
          stream.sync();
          decoded.table = std::move(filtered);
        }
      }
      REQUIRE(decoded.table->num_rows() == 1);
      if (applied && count_only)
        CHECK(decoded.table->view().column(0).type().id() == cudf::type_id::INT8);
      else
        CHECK(column_to_host(decoded.table->view().column(0), stream) ==
              std::vector<std::int32_t>{1});
    }
  }
}

TEST_CASE("Count-only decoded LIKE combines two producer lanes across a partial final chunk",
          "[scan][decoded_like_sql][decoded_like_trace]")
{
  if (gate_off()) return;
  auto const stream = cudf::get_default_stream();
  auto const mr     = cudf::get_current_device_resource_ref();
  // Row 1026 survives both predicates. A dropped tail or counted padding changes the total.
  constexpr int n = sc::SELECTION_CHUNK_ROWS + 3;
  std::vector<std::string> left(n), right(n);
  for (int i = 0; i < n; ++i) {
    left[i]  = i % 2 == 0 ? "red" : "-";
    right[i] = i % 3 == 0 ? "blue" : "-";
  }
  auto a      = upload_strings(left, stream);
  auto b      = upload_strings(right, stream);
  auto stored = simpatico::compress_with_plan(cudf::table_view{{a->view(), b->view()}},
                                              "input -> str_split\n---\ninput -> str_split\n",
                                              stream,
                                              mr);
  stream.sync();
  simpatico::stream_pool pool;
  REQUIRE(pool.init(2));
  std::array<std::size_t, 2> const selected{0, 1};
  sc::scan_filter_request request;
  // Raw str_split offsets have no bitpack/delta child. Use the same compact-route
  // probe as the engine, including its full-decode answer for this plan.
  for (auto index : selected)
    request.routes.push_back(
      simpatico::probe_column(*stored.columns[index].plan_tree).compact_route);
  request.like_filters   = {{0, {{"red"}, false}}, {1, {{"blue"}, false}}};
  request.row_count_only = true;
  std::vector<cudaStream_t> lanes;
  request.try_like_swar =
    [&](auto const& input, std::string_view sql, bool invert, auto lane, auto resource) {
      lanes.push_back(lane.get());
      auto const parsed = sirius::classify_like_multiliteral(sql, {});
      REQUIRE(parsed.has_value());
      return sirius::like_multiliteral(input, *parsed, invert, lane, resource);
    };
  // The codec emits these synchronous notifications after the combined D2H readback.
  static thread_local std::vector<std::pair<std::size_t, std::int64_t>> counts;
  counts.clear();
  request.log_decoded_like = [](std::size_t column,
                                std::string const&,
                                std::string_view matcher,
                                std::int64_t rows,
                                std::int64_t matches) {
    CHECK(rows == n);
    CHECK(matcher == "swar");
    counts.emplace_back(column, matches);
  };
  sc::scan_filter_result result;
  std::unique_ptr<cudf::table> carrier;
  auto const reads_before = simpatico::host_observations.readbacks;
  bool const enabled      = simpatico::decoded_like_enabled();
  {
    sirius::test::like_matcher_trace trace{enabled ? "count_only" : "count_off"};
    carrier =
      simpatico::decompress_scan_filter(stored, selected, request, result, pool, stream, mr);
  }
  if (!enabled) {
    REQUIRE_FALSE(result.applied);
    CHECK(lanes.empty());
    CHECK(counts.empty());
    REQUIRE(carrier->num_columns() == 2);
    REQUIRE(carrier->num_rows() == n);
    CHECK(strings_to_host(carrier->view().column(0), stream) == left);
    CHECK(strings_to_host(carrier->view().column(1), stream) == right);
    return;
  }
  REQUIRE(result.applied);
  REQUIRE(lanes.size() == 2);
  CHECK(lanes[0] != lanes[1]);
  CHECK(simpatico::host_observations.readbacks - reads_before == 1);
  CHECK(counts == decltype(counts){{0, 514}, {1, 343}});
  CHECK(result.survivor_count == 172);
  // Check the intersected mask through the final padded word. Only rows divisible
  // by six survive, including the last real row in the second 1024-row strip.
  std::vector<std::uint32_t> expected_words(sc::selection_mask::AllocWordsFor(n), 0);
  for (int i = 0; i < n; i += 6)
    expected_words[i / 32] |= 1U << (i % 32);
  std::vector<std::uint32_t> actual_words(expected_words.size());
  REQUIRE(result.mask_words.size() == actual_words.size() * sizeof(std::uint32_t));
  REQUIRE(cudaMemcpyAsync(actual_words.data(),
                          result.mask_words.data(),
                          result.mask_words.size(),
                          cudaMemcpyDeviceToHost,
                          stream.get()) == cudaSuccess);
  stream.sync();
  CHECK(actual_words == expected_words);
  REQUIRE(carrier->num_columns() == 1);
  CHECK(carrier->num_rows() == result.survivor_count);
  CHECK(carrier->view().column(0).type().id() == cudf::type_id::INT8);
  CHECK(result.chunk_offsets.size() == 0);
  CHECK(result.row_indices.size() == 0);
}

TEST_CASE("OnPair LIKE engine outcomes distinguish values, dropped suffix and count carrier",
          "[scan][onpair_like][fused_scan_filter][delivery]")
{
  if (gate_off() || !simpatico::onpair_like_enabled()) return;
  auto const stream = cudf::get_default_stream();
  decode_environment environment{stream};
  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(upload_column<std::int32_t>({0, 1, 2, 3, 4, 5, 6, 7}, stream));
  columns.push_back(
    upload_strings({"google", "other", "google-x", "", "a", "b", "c", "d"}, stream));
  columns.push_back(upload_column<std::int32_t>({7, 8, 9, 10, 11, 12, 13, 14}, stream));
  stream.sync();
  cudf::table input{std::move(columns)};
  auto stored = simpatico::compress_with_plan(
    input.view(),
    "input -> bitpack\n---\ninput -> onpair_16\n---\ninput -> bitpack\n",
    stream,
    cudf::get_current_device_resource_ref());
  sirius::pushdown_request request;
  request.columns.resize(3);
  request.columns[1].likes          = {{{"google"}, false}};
  request.ranges_cover_whole_filter = true;
  SECTION("projected LIKE stays STRING and advertises no BOOL8 substitution") {}
  SECTION("filter-only LIKE can be dropped behind the output prefix")
  { request.output_prefix_width = 1; }
  SECTION("all-filter-only decode delivers a count carrier") { request.row_count_only = true; }
  SECTION("partial coverage keeps values even when count-only was offered")
  {
    request.row_count_only            = true;
    request.ranges_cover_whole_filter = false;
  }
  auto decoded = decompress_with(environment, stored, request, stream);
  CHECK(decoded.outcome.predicate_columns.empty());
  CHECK_FALSE(decoded.outcome.predicates_enforced);  // reserved for equality BOOL8 answers
  CHECK(decoded.outcome.enforced_like_columns == std::vector<std::size_t>{1});
  CHECK(decoded.outcome.row_filtered == request.ranges_cover_whole_filter);
  REQUIRE(decoded.table->num_rows() == 2);
  if (request.row_count_only && request.ranges_cover_whole_filter) {
    CHECK(decoded.outcome.row_count_carrier);
    CHECK(decoded.outcome.filter_only_columns_dropped);
    REQUIRE(decoded.table->num_columns() == 1);
    CHECK(decoded.table->view().column(0).type().id() == cudf::type_id::INT8);
  } else if (request.output_prefix_width) {
    CHECK(decoded.outcome.filter_only_columns_dropped);
    REQUIRE(decoded.table->num_columns() == 1);
  } else {
    CHECK_FALSE(decoded.outcome.row_count_carrier);
    CHECK_FALSE(decoded.outcome.filter_only_columns_dropped);
    REQUIRE(decoded.table->num_columns() == 3);
    CHECK(strings_to_host(decoded.table->view().column(1), stream) ==
          std::vector<std::string>{"google", "google-x"});
  }
}

TEST_CASE("High-survival filter-only LIKE delivers BOOL8 and full-width INT32 values",
          "[scan][onpair_like][fused_scan_filter][delivery][engine_bool8]")
{
  if (gate_off() || !simpatico::onpair_like_enabled()) return;
  // 989 of 1000 rows survive. The unittest binary lifts the selectivity cap at static
  // initialization (test_gpu_execution_cast_date_predicates.cpp); the BOOL8 contract needs the
  // production cap, which the engine_like_bool8 CTest registrations set explicitly.
  if (sc::decompression_pushdown_max_selectivity() >= 0.989)
    SKIP("needs SIRIUS_EXP_FUSED_SCAN_MAX_SEL below 0.989 (default 0.35)");
  auto const stream = cudf::get_default_stream();
  decode_environment environment{stream};
  std::vector<std::int32_t> keys(1000);
  std::iota(keys.begin(), keys.end(), 0);
  std::vector<std::string> comments(1000, "ordinary order comment");
  std::fill_n(comments.begin(), 11, "special requests");
  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(upload_column(keys, stream));
  columns.push_back(upload_column(keys, stream));
  columns.push_back(upload_strings(comments, stream));
  stream.sync();
  cudf::table input{std::move(columns)};
  auto stored = simpatico::compress_with_plan(
    input.view(),
    "input -> bitpack\n---\ninput -> bitpack\n---\ninput -> onpair_16\n",
    stream,
    cudf::get_current_device_resource_ref());
  sirius::pushdown_request request;
  request.columns.resize(3);
  request.columns[2].likes          = {{{"special", "requests"}, true}};
  request.ranges_cover_whole_filter = true;
  request.output_prefix_width       = 2;
  bool values_required              = false;
  sirius::decode_visibility_mask keep_mask;
  SECTION("filter-only column may substitute BOOL8") {}
  SECTION("projected LIKE keeps STRING values")
  {
    request.output_prefix_width.reset();
    values_required = true;
  }
  SECTION("partial static coverage keeps STRING values")
  {
    request.ranges_cover_whole_filter = false;
    values_required                   = true;
  }
  SECTION("outstanding MVCC mask declines BOOL8 before metadata-dropping materialization")
  {
    // All-visible is still an outstanding positional mask. Preserve STRING for the residual.
    auto const word_count = (comments.size() + 31) / 32;
    auto words            = std::make_shared<std::uint32_t[]>(word_count, 0xffffffffU);
    if (comments.size() % 32 != 0) words[word_count - 1] &= (1U << (comments.size() % 32)) - 1;
    keep_mask.words     = std::move(words);  // the mask type holds const words
    keep_mask.row_count = comments.size();
    values_required = true;
  }
  sirius::decompression_pushdown_scan decode_scan{request};
  auto decoded = sirius::decompress_chunk(stored,
                                          kSelected,
                                          &decode_scan,
                                          keep_mask,
                                          environment.space,
                                          environment.tracking_scope,
                                          stream,
                                          environment.space.get_default_allocator());
  if (sc::onpair_like_mask_above_cap_enabled() && values_required) {
    REQUIRE(decoded.table->num_rows() == 1000);
    REQUIRE(decoded.table->num_columns() == 3);
    CHECK_FALSE(decoded.outcome.row_filtered);
    CHECK(decoded.outcome.selection_unprofitable);
    CHECK(decoded.outcome.predicate_columns.empty());
    CHECK(decoded.outcome.enforced_like_columns.empty());
    CHECK(strings_to_host(decoded.table->view().column(2), stream) == comments);
  } else if (sc::onpair_like_mask_above_cap_enabled()) {
    REQUIRE(decoded.table->num_columns() == 3);
    REQUIRE(decoded.table->num_rows() == 1000);
    CHECK_FALSE(decoded.outcome.row_filtered);
    CHECK_FALSE(decoded.outcome.selection_unprofitable);
    CHECK_FALSE(decoded.outcome.predicates_enforced);
    CHECK_FALSE(decoded.outcome.filter_only_columns_dropped);
    CHECK(decoded.outcome.predicate_columns == std::vector<std::size_t>{2});
    CHECK(decoded.outcome.enforced_like_columns.empty());
    CHECK(decoded.table->view().column(2).type().id() == cudf::type_id::BOOL8);
    std::vector<std::uint8_t> answers(1000, 1);
    std::fill_n(answers.begin(), 11, 0);
    CHECK(column_to_host<std::uint8_t>(decoded.table->view().column(2),
                                       stream,
                                       cudf::data_type{cudf::type_id::BOOL8}) == answers);
    for (cudf::size_type column : {0, 1})
      CHECK(column_to_host(decoded.table->view().column(column), stream) == keys);
  } else {
    REQUIRE(decoded.table->num_columns() == (values_required && !keep_mask.has_mask() ? 3 : 2));
    REQUIRE(decoded.table->num_rows() == 989);
    CHECK(decoded.outcome.row_filtered == request.ranges_cover_whole_filter);
    CHECK(decoded.outcome.filter_only_columns_dropped ==
          (!values_required || keep_mask.has_mask()));
    CHECK(decoded.outcome.predicate_columns.empty());
    CHECK(decoded.outcome.enforced_like_columns == std::vector<std::size_t>{2});
  }
}

TEST_CASE("a decode carrying the whole filter drops the trailing pure-filter column",
          "[fused_scan_filter][delivery]")
{
  if (gate_off()) { return; }
  ::cuda::stream_ref const stream = cudf::get_default_stream();
  auto const per_stream           = GENERATE(false, true);
  CAPTURE(per_stream);
  decode_environment environment{stream, per_stream};
  auto const ct = compress_range_source(stream);

  auto const decoded = decompress_with(environment, ct, covered_request(kOutputWidth), stream);

  CHECK(decoded.outcome.row_filtered);
  CHECK(decoded.outcome.filter_only_columns_dropped);
  CHECK(decoded.outcome.predicate_columns.empty());
  require_columns(decoded.table->view(), rows_where(in_filter_range), kOutputWidth, stream);
}

TEST_CASE("a scan latched unselective still compacts a chunk with no survivors",
          "[fused_scan_filter][delivery]")
{
  if (gate_off()) { return; }
  ::cuda::stream_ref const stream = cudf::get_default_stream();
  decode_environment environment{stream};
  auto const ct = compress_range_source(stream);
  auto const latched =
    sirius::decompression_pushdown_scan{covered_request(kOutputWidth)}.for_latched_scan();
  if (!sc::decompression_pushdown_latched_zero_check_enabled()) {
    // The old latch: with only a range to drop, nothing is left to ask for.
    CHECK(latched == nullptr);
    return;
  }
  REQUIRE(latched);
  CHECK(latched->request().zero_survivor_only);
  CHECK_FALSE(latched->request().row_selection_disabled);
  CHECK(latched->for_latched_scan()->request().zero_survivor_only);
  auto decode = [&](sirius::decompression_pushdown_scan const& scan) {
    return sirius::decompress_chunk(ct,
                                    kSelected,
                                    &scan,
                                    sirius::decode_visibility_mask{},
                                    environment.space,
                                    environment.tracking_scope,
                                    stream,
                                    environment.space.get_default_allocator());
  };

  // A chunk with survivors declines and comes back whole for the scan to filter.
  auto const kept = decode(*latched);
  CHECK_FALSE(kept.outcome.row_filtered);
  CHECK(kept.outcome.selection_unprofitable);
  require_columns(kept.table->view(), rows_where(every_row), kSelected.size(), stream);

  // "f" never exceeds 999, so this range keeps nothing: the latched scan still applies it.
  auto empty_request             = covered_request(kOutputWidth);
  empty_request.columns[2].range = sirius::decode_range{2000, 3000};
  auto const latched_empty =
    sirius::decompression_pushdown_scan{std::move(empty_request)}.for_latched_scan();
  auto const dropped = decode(*latched_empty);
  CHECK(dropped.outcome.row_filtered);
  CHECK_FALSE(dropped.outcome.selection_unprofitable);
  CHECK(dropped.table->num_rows() == 0);
  CHECK(static_cast<std::size_t>(dropped.table->num_columns()) == kOutputWidth);
}

TEST_CASE("a decode with a join filter still drops the pure-filter column",
          "[fused_scan_filter][delivery]")
{
  if (gate_off()) { return; }
  ::cuda::stream_ref const stream = cudf::get_default_stream();
  decode_environment environment{stream};
  auto const ct = compress_range_source(stream);

  // A join filter is not part of the scan's own filter, so the range alone decides coverage.
  auto const decoded = decompress_with_probes(environment,
                                              ct,
                                              covered_request(kOutputWidth),
                                              one_probe(0, key_set_probe(make_key_set(stream))),
                                              stream);

  CHECK(decoded.outcome.row_filtered);
  CHECK(decoded.outcome.filter_only_columns_dropped);
  CHECK(decoded.outcome.predicate_columns.empty());
  // The join filter keeps v = i % 100 in its key set {0, 5, ..., 45}.
  auto const expected =
    rows_where([](cudf::size_type i) { return in_filter_range(i) && in_key_set(i % 100); });
  require_columns(decoded.table->view(), expected, kOutputWidth, stream);
}

TEST_CASE("a join filter past the delivered prefix is applied but not delivered",
          "[fused_scan_filter][delivery]")
{
  if (gate_off()) { return; }
  ::cuda::stream_ref const stream = cudf::get_default_stream();
  decode_environment environment{stream};
  auto const ct = compress_range_source(stream);

  // The join filter tests the pure-filter column "f" while the selection is built; "f" is still
  // dropped from the delivered columns.
  auto const decoded = decompress_with_probes(environment,
                                              ct,
                                              covered_request(kOutputWidth),
                                              one_probe(2, key_set_probe(make_key_set(stream))),
                                              stream);

  CHECK(decoded.outcome.row_filtered);
  CHECK(decoded.outcome.filter_only_columns_dropped);
  auto const expected = rows_where(
    [](cudf::size_type i) { return in_filter_range(i) && in_key_set(i % kFilterModulus); });
  require_columns(decoded.table->view(), expected, kOutputWidth, stream);
}

TEST_CASE("a join filter that declines leaves the range's survivors and the drop",
          "[fused_scan_filter][delivery]")
{
  if (gate_off()) { return; }
  ::cuda::stream_ref const stream = cudf::get_default_stream();
  decode_environment environment{stream};
  auto const ct = compress_range_source(stream);

  auto const decoded = decompress_with_probes(
    environment, ct, covered_request(kOutputWidth), one_probe(0, declining_probe()), stream);

  // The declined join filter removed nothing, so every survivor is a row of the range.
  CHECK(decoded.outcome.row_filtered);
  CHECK(decoded.outcome.filter_only_columns_dropped);
  require_columns(decoded.table->view(), rows_where(in_filter_range), kOutputWidth, stream);
}

TEST_CASE("a decode whose only source is a join filter keeps every column",
          "[fused_scan_filter][delivery]")
{
  if (gate_off()) { return; }
  ::cuda::stream_ref const stream = cudf::get_default_stream();
  decode_environment environment{stream};
  auto const ct = compress_range_source(stream);

  sirius::pushdown_request request;
  request.columns.resize(kSelected.size());
  request.output_prefix_width = kOutputWidth;
  SECTION("the request claims no coverage") { request.ranges_cover_whole_filter = false; }
  // Coverage needs a range the decode evaluates; a claim with none is not enough.
  SECTION("the request claims coverage with no range") { request.ranges_cover_whole_filter = true; }
  auto const decoded = decompress_with_probes(
    environment, ct, std::move(request), one_probe(0, key_set_probe(make_key_set(stream))), stream);

  CHECK_FALSE(decoded.outcome.row_filtered);
  CHECK_FALSE(decoded.outcome.filter_only_columns_dropped);
  auto const expected = rows_where([](cudf::size_type i) { return in_key_set(i % 100); });
  require_columns(decoded.table->view(), expected, kSelected.size(), stream);
}

TEST_CASE("a join filter does not complete a partly carried filter",
          "[fused_scan_filter][delivery]")
{
  if (gate_off()) { return; }
  ::cuda::stream_ref const stream = cudf::get_default_stream();
  decode_environment environment{stream};
  auto const ct = compress_range_source(stream);

  auto const in_range_and_keys = [](cudf::size_type i) {
    return in_filter_range(i) && in_key_set(i % 100);
  };
  auto request = covered_request(kOutputWidth);
  SECTION("the ranges are not the scan's whole filter")
  { request.ranges_cover_whole_filter = false; }
  // A provably empty conjunct is dropped from the decode and left to the scan, which rejects
  // every row.
  SECTION("a conjunct is dropped") { request.columns[1].range = sirius::decode_range{5, 3}; }
  auto const decoded = decompress_with_probes(
    environment, ct, std::move(request), one_probe(0, key_set_probe(make_key_set(stream))), stream);

  CHECK_FALSE(decoded.outcome.row_filtered);
  CHECK_FALSE(decoded.outcome.filter_only_columns_dropped);
  require_columns(decoded.table->view(), rows_where(in_range_and_keys), kSelected.size(), stream);
}

TEST_CASE("a nullable-selection refusal with a join filter keeps every row and column",
          "[fused_scan_filter][delivery]")
{
  if (gate_off()) { return; }
  ::cuda::stream_ref const stream = cudf::get_default_stream();
  decode_environment environment{stream};
  auto const ct = compress_string_source(/*nullable=*/true, stream);

  auto const decoded = decompress_with_probes(environment,
                                              ct,
                                              covered_request(kOutputWidth),
                                              one_probe(0, key_set_probe(make_key_set(stream))),
                                              stream,
                                              {},
                                              {.measure = true, .complete = true});

  CHECK_FALSE(decoded.outcome.row_filtered);
  CHECK_FALSE(decoded.outcome.filter_only_columns_dropped);
  // The refusal returns every row, so it measured nothing and reports the join filter as skipped.
  CHECK_FALSE(decoded.outcome.membership_keep.has_value());
  CHECK(decoded.outcome.membership_skipped);
  require_columns(decoded.table->view(),
                  string_rows_where(every_row, /*nullable=*/true),
                  kSelected.size(),
                  stream);
}

TEST_CASE("a decode with a join filter and a visibility mask consumes both",
          "[fused_scan_filter][delivery]")
{
  if (gate_off()) { return; }
  ::cuda::stream_ref const stream = cudf::get_default_stream();
  decode_environment environment{stream};
  auto const ct = compress_range_source(stream);

  // Keep every even row.
  static_assert(kRows % 32 == 0);
  sirius::decode_visibility_mask keep_even;
  keep_even.words     = std::make_shared<std::uint32_t[]>(kRows / 32, 0x55555555U);
  keep_even.row_count = kRows;
  auto const decoded  = decompress_with_probes(environment,
                                               ct,
                                               covered_request(kOutputWidth),
                                               one_probe(0, key_set_probe(make_key_set(stream))),
                                               stream,
                                               keep_even);

  CHECK(decoded.outcome.row_filtered);
  CHECK(decoded.outcome.visibility_mask_applied);
  CHECK(decoded.outcome.filter_only_columns_dropped);
  auto const expected = rows_where(
    [](cudf::size_type i) { return i % 2 == 0 && in_filter_range(i) && in_key_set(i % 100); });
  require_columns(decoded.table->view(), expected, kOutputWidth, stream);
}

TEST_CASE("join filters past the decode's cap leave the drop in place",
          "[fused_scan_filter][delivery]")
{
  if (gate_off()) { return; }
  INFO(
    "fused_gate_armer sets SIRIUS_EXP_FUSED_SCAN_MAX_MEMBER=2 unless the environment overrides it");
  REQUIRE(sc::decompression_pushdown_max_membership_sources() == 2);
  ::cuda::stream_ref const stream = cudf::get_default_stream();
  decode_environment environment{stream};
  auto const ct = compress_range_source(stream);

  // Listed out of rank order: the decode sorts them strongest first and carries the first two.
  slot_probes probes(kSelected.size());
  probes[1].push_back(key_set_probe(make_key_set(stream, /*step=*/3), /*selectivity_rank=*/2));
  probes[0].push_back(key_set_probe(make_key_set(stream, /*step=*/5), /*selectivity_rank=*/0));
  probes[1].push_back(key_set_probe(make_key_set(stream, /*step=*/2), /*selectivity_rank=*/1));
  auto const decoded = decompress_with_probes(environment,
                                              ct,
                                              covered_request(kOutputWidth),
                                              std::move(probes),
                                              stream,
                                              {},
                                              {.measure = true, .complete = true});

  CHECK(decoded.outcome.row_filtered);
  CHECK(decoded.outcome.filter_only_columns_dropped);
  CHECK_FALSE(decoded.outcome.membership_skipped);
  // The capped multiples-of-3 filter on "k" did not run.
  auto const expected = rows_where([](cudf::size_type i) {
    return in_filter_range(i) && in_key_set(i % 100) && in_key_set(i % 50, 2);
  });
  require_columns(decoded.table->view(), expected, kOutputWidth, stream);
  // Two of the three requested probes ran, so the measurement cannot describe the whole set.
  REQUIRE(decoded.outcome.membership_keep.has_value());
  CHECK(decoded.outcome.membership_keep->probes.size() == 2);
  CHECK(decoded.outcome.membership_keep->rows_after == expected.v.size());
  CHECK_FALSE(decoded.outcome.membership_keep->complete);
}

TEST_CASE("a measuring decode reports what its join filters kept",
          "[fused_scan_filter][delivery][membership_measurement]")
{
  if (gate_off()) { return; }
  ::cuda::stream_ref const stream = cudf::get_default_stream();
  decode_environment environment{stream};
  auto const ct = compress_range_source(stream);
  sirius::membership_measurement_request const measure{.measure = true, .complete = true};
  auto const count_rows = [](auto predicate) { return rows_where(predicate).v.size(); };
  using probe_keep      = sirius::pushdown_outcome::membership_keep_observation::probe_keep;
  auto const tagged     = [](sirius::membership_probe probe, std::size_t tag) {
    probe.tag = tag;
    return probe;
  };
  auto const require_probe =
    [](probe_keep const& probe, std::size_t tag, std::size_t rows_before, std::size_t rows_after) {
      CHECK(probe.tag == tag);
      CHECK(probe.rows_before == rows_before);
      CHECK(probe.rows_after == rows_after);
    };
  // "v" = i % 100 in the multiples of 5 below 50, and "k" = i % 50 even.
  auto const in_v_keys      = [](cudf::size_type i) { return in_key_set(i % 100); };
  auto const in_k_keys      = [](cudf::size_type i) { return in_key_set(i % 50, 2); };
  auto const v_and_k_probes = [&] {
    slot_probes probes(kSelected.size());
    probes[0].push_back(tagged(key_set_probe(make_key_set(stream)), 7));
    probes[1].push_back(tagged(key_set_probe(make_key_set(stream, /*step=*/2)), 9));
    return probes;
  };

  SECTION("a join filter that is the only source is evaluated on every row")
  {
    sirius::pushdown_request request;
    request.columns.resize(kSelected.size());
    auto const decoded =
      decompress_with_probes(environment,
                             ct,
                             std::move(request),
                             one_probe(0, tagged(key_set_probe(make_key_set(stream)), 3)),
                             stream,
                             {},
                             measure);
    REQUIRE(decoded.outcome.membership_keep.has_value());
    auto const& keep = *decoded.outcome.membership_keep;
    CHECK(keep.generation == 1);
    CHECK(keep.rows_before == static_cast<std::size_t>(kRows));
    CHECK(keep.rows_after == count_rows(in_v_keys));
    CHECK(keep.complete);
    REQUIRE(keep.probes.size() == 1);
    require_probe(keep.probes[0], 3, kRows, count_rows(in_v_keys));
    CHECK_FALSE(decoded.outcome.membership_skipped);
  }
  SECTION("concurrent join filters each report what they kept of the earlier ones' survivors")
  {
    sirius::pushdown_request request;
    request.columns.resize(kSelected.size());
    auto const decoded = decompress_with_probes(
      environment, ct, std::move(request), v_and_k_probes(), stream, {}, measure);
    REQUIRE(decoded.outcome.membership_keep.has_value());
    auto const& keep = *decoded.outcome.membership_keep;
    auto const both  = count_rows([&](cudf::size_type i) { return in_v_keys(i) && in_k_keys(i); });
    CHECK(keep.rows_before == static_cast<std::size_t>(kRows));
    CHECK(keep.rows_after == both);
    // The set with fewer keys runs first.
    REQUIRE(keep.probes.size() == 2);
    require_probe(keep.probes[0], 7, kRows, count_rows(in_v_keys));
    require_probe(keep.probes[1], 9, count_rows(in_v_keys), both);
  }
  SECTION("a join filter after a range is evaluated on the range's survivors")
  {
    auto const decoded = decompress_with_probes(environment,
                                                ct,
                                                covered_request(kOutputWidth),
                                                one_probe(0, key_set_probe(make_key_set(stream))),
                                                stream,
                                                {},
                                                measure);
    REQUIRE(decoded.outcome.membership_keep.has_value());
    auto const kept =
      count_rows([&](cudf::size_type i) { return in_filter_range(i) && in_v_keys(i); });
    CHECK(decoded.outcome.membership_keep->rows_before == count_rows(in_filter_range));
    CHECK(decoded.outcome.membership_keep->rows_after == kept);
    CHECK(decoded.outcome.membership_keep->complete);
    REQUIRE(decoded.outcome.membership_keep->probes.size() == 1);
    require_probe(decoded.outcome.membership_keep->probes[0], 0, count_rows(in_filter_range), kept);
  }
  SECTION("cascaded join filters each report what they kept of the earlier sources' survivors")
  {
    auto const decoded = decompress_with_probes(
      environment, ct, covered_request(kOutputWidth), v_and_k_probes(), stream, {}, measure);
    REQUIRE(decoded.outcome.membership_keep.has_value());
    auto const& keep  = *decoded.outcome.membership_keep;
    auto const ranged = count_rows(in_filter_range);
    auto const first =
      count_rows([&](cudf::size_type i) { return in_filter_range(i) && in_v_keys(i); });
    auto const both = count_rows(
      [&](cudf::size_type i) { return in_filter_range(i) && in_v_keys(i) && in_k_keys(i); });
    CHECK(keep.rows_before == ranged);
    CHECK(keep.rows_after == both);
    REQUIRE(keep.probes.size() == 2);
    require_probe(keep.probes[0], 7, ranged, first);
    require_probe(keep.probes[1], 9, first, both);
  }
  SECTION("a join filter after a visibility mask is evaluated on the visible rows")
  {
    // Keep every even row.
    static_assert(kRows % 32 == 0);
    sirius::decode_visibility_mask keep_even;
    keep_even.words     = std::make_shared<std::uint32_t[]>(kRows / 32, 0x55555555U);
    keep_even.row_count = kRows;
    sirius::pushdown_request request;
    request.columns.resize(kSelected.size());
    auto const decoded = decompress_with_probes(environment,
                                                ct,
                                                std::move(request),
                                                one_probe(0, key_set_probe(make_key_set(stream))),
                                                stream,
                                                keep_even,
                                                measure);
    REQUIRE(decoded.outcome.membership_keep.has_value());
    auto const visible = count_rows([](cudf::size_type i) { return i % 2 == 0; });
    auto const kept    = count_rows([&](cudf::size_type i) { return i % 2 == 0 && in_v_keys(i); });
    CHECK(decoded.outcome.membership_keep->rows_before == visible);
    CHECK(decoded.outcome.membership_keep->rows_after == kept);
    REQUIRE(decoded.outcome.membership_keep->probes.size() == 1);
    require_probe(decoded.outcome.membership_keep->probes[0], 0, visible, kept);
  }
  SECTION("a join filter after an equality is evaluated on the equality's survivors")
  {
    auto const equality_ct = compress_equality_source(stream);
    sirius::pushdown_request request;
    request.columns.resize(kSelected.size());
    request.columns[2].equals_any = {key_string(5), key_string(10), key_string(15)};
    auto const in_equality        = [](cudf::size_type i) {
      auto const s = i % 50;
      return s == 5 || s == 10 || s == 15;
    };
    auto const decoded = decompress_with_probes(environment,
                                                equality_ct,
                                                std::move(request),
                                                one_probe(0, key_set_probe(make_key_set(stream))),
                                                stream,
                                                {},
                                                measure);
    REQUIRE(decoded.outcome.membership_keep.has_value());
    auto const matching = count_rows(in_equality);
    auto const kept = count_rows([&](cudf::size_type i) { return in_equality(i) && in_v_keys(i); });
    CHECK(decoded.outcome.membership_keep->rows_before == matching);
    CHECK(decoded.outcome.membership_keep->rows_after == kept);
    REQUIRE(decoded.outcome.membership_keep->probes.size() == 1);
    require_probe(decoded.outcome.membership_keep->probes[0], 0, matching, kept);
  }
  SECTION("a declined join filter makes the measurement incomplete")
  {
    slot_probes probes(kSelected.size());
    probes[0].push_back(key_set_probe(make_key_set(stream)));
    probes[1].push_back(declining_probe());
    auto const decoded = decompress_with_probes(
      environment, ct, covered_request(kOutputWidth), std::move(probes), stream, {}, measure);
    REQUIRE(decoded.outcome.membership_keep.has_value());
    CHECK(decoded.outcome.membership_keep->rows_after ==
          count_rows([&](cudf::size_type i) { return in_filter_range(i) && in_v_keys(i); }));
    CHECK_FALSE(decoded.outcome.membership_keep->complete);
    CHECK(decoded.outcome.membership_keep->probes.size() == 1);
  }
  SECTION("a decode whose every join filter declined measures nothing")
  {
    auto const decoded = decompress_with_probes(environment,
                                                ct,
                                                covered_request(kOutputWidth),
                                                one_probe(0, declining_probe()),
                                                stream,
                                                {},
                                                measure);
    CHECK_FALSE(decoded.outcome.membership_keep.has_value());
    CHECK(decoded.outcome.membership_skipped);
  }
  SECTION("a decode that was not asked measures nothing")
  {
    auto const decoded = decompress_with_probes(environment,
                                                ct,
                                                covered_request(kOutputWidth),
                                                one_probe(0, key_set_probe(make_key_set(stream))),
                                                stream);
    CHECK_FALSE(decoded.outcome.membership_keep.has_value());
    CHECK_FALSE(decoded.outcome.membership_skipped);
  }
}

TEST_CASE("a decode carrying part of the filter keeps every column",
          "[fused_scan_filter][delivery]")
{
  if (gate_off()) { return; }
  ::cuda::stream_ref const stream = cudf::get_default_stream();
  decode_environment environment{stream};
  auto const ct = compress_range_source(stream);

  auto request                      = covered_request(kOutputWidth);
  request.ranges_cover_whole_filter = false;
  auto const decoded                = decompress_with(environment, ct, std::move(request), stream);

  CHECK_FALSE(decoded.outcome.row_filtered);
  CHECK_FALSE(decoded.outcome.filter_only_columns_dropped);
  require_columns(decoded.table->view(), rows_where(in_filter_range), kSelected.size(), stream);
}

TEST_CASE("a decode without an output prefix keeps every column", "[fused_scan_filter][delivery]")
{
  if (gate_off()) { return; }
  ::cuda::stream_ref const stream = cudf::get_default_stream();
  decode_environment environment{stream};
  auto const ct = compress_range_source(stream);

  auto const decoded = decompress_with(environment, ct, covered_request(std::nullopt), stream);

  CHECK(decoded.outcome.row_filtered);
  CHECK_FALSE(decoded.outcome.filter_only_columns_dropped);
  require_columns(decoded.table->view(), rows_where(in_filter_range), kSelected.size(), stream);
}

TEST_CASE("each chunk of one scan decides its own delivery", "[fused_scan_filter][delivery]")
{
  if (gate_off()) { return; }
  ::cuda::stream_ref const stream = cudf::get_default_stream();
  decode_environment environment{stream};
  auto const mr       = environment.space.get_default_allocator();
  auto const clean    = compress_string_source(/*nullable=*/false, stream);
  auto const nullable = compress_string_source(/*nullable=*/true, stream);
  sirius::decompression_pushdown_scan const scan{covered_request(kOutputWidth)};

  auto const dropped = sirius::decompress_chunk(clean,
                                                kSelected,
                                                &scan,
                                                sirius::decode_visibility_mask{},
                                                environment.space,
                                                environment.tracking_scope,
                                                stream,
                                                mr);
  CHECK(dropped.outcome.row_filtered);
  CHECK(dropped.outcome.filter_only_columns_dropped);
  require_columns(dropped.table->view(),
                  string_rows_where(in_filter_range, /*nullable=*/false),
                  kOutputWidth,
                  stream);

  // The same request over a chunk whose "n" is null-masked refuses nullable selection after
  // filtering, so the scan gets the whole chunk back and filters it itself.
  auto const kept = sirius::decompress_chunk(nullable,
                                             kSelected,
                                             &scan,
                                             sirius::decode_visibility_mask{},
                                             environment.space,
                                             environment.tracking_scope,
                                             stream,
                                             mr);
  CHECK_FALSE(kept.outcome.row_filtered);
  CHECK_FALSE(kept.outcome.filter_only_columns_dropped);
  require_columns(
    kept.table->view(), string_rows_where(every_row, /*nullable=*/true), kSelected.size(), stream);
}

TEST_CASE("an equality-only pure-filter column past the prefix is dropped with its answer",
          "[fused_scan_filter][delivery]")
{
  if (gate_off()) { return; }
  ::cuda::stream_ref const stream = cudf::get_default_stream();
  decode_environment environment{stream};
  auto const ct = compress_equality_source(stream);

  // The range sits on output column "v"; "s" is read only by its equality.
  sirius::pushdown_request request;
  request.columns.resize(kSelected.size());
  request.columns[0].range          = sirius::decode_range{0, 49};
  request.columns[2].equals_any     = {key_string(5)};
  request.ranges_cover_whole_filter = true;
  request.output_prefix_width       = kOutputWidth;
  auto const decoded                = decompress_with(environment, ct, std::move(request), stream);

  CHECK(decoded.outcome.row_filtered);
  CHECK(decoded.outcome.filter_only_columns_dropped);
  CHECK(decoded.outcome.predicate_columns.empty());
  CHECK(decoded.outcome.predicates_enforced);
  auto const expected = rows_where([](cudf::size_type i) { return i % 100 <= 49 && i % 50 == 5; });
  require_columns(decoded.table->view(), expected, kOutputWidth, stream);
}
