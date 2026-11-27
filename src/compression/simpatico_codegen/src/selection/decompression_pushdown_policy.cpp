// SPDX-License-Identifier: Apache-2.0

#include "codegen/selection/decompression_pushdown_policy.hpp"

#include "util/experimental_policy.hpp"

#include <cmath>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <string_view>

namespace sirius::codegen {

namespace {

/// "Set and not exactly 0" — the contract every boolean knob here uses.
bool env_flag(char const* name)
{
  char const* value = std::getenv(name);
  return value != nullptr && std::string_view{value} != "0";
}

/// A positive double, or @p fallback when unset, unparsable or <= 0. Rejecting
/// 0 is what makes "set it tiny" a kill switch rather than a silent default.
double env_fraction(char const* name, double fallback)
{
  char const* value = std::getenv(name);
  if (value == nullptr || *value == '\0') { return fallback; }
  char* end      = nullptr;
  double const d = std::strtod(value, &end);
  return (end != value && d > 0.0) ? d : fallback;
}

}  // namespace

bool decompression_pushdown_enabled()
{
  static bool const enabled = env_flag("SIRIUS_EXP_FUSED_SCAN_FILTER");
  return enabled;
}

bool decompression_pushdown_diag_enabled()
{
  static bool const enabled = env_flag("SIRIUS_EXP_FUSED_SCAN_DIAG");
  return enabled;
}

bool decompression_pushdown_batched_wave2_enabled()
{
  static simpatico::experimental_policy policy{"SIRIUS_EXP_FUSED_SCAN_BATCHED_WAVE2",
                                               "fused-scan-batched-wave2"};
  return policy.enabled();
}

bool decompression_pushdown_full_route_zero_skip_enabled()
{
  static simpatico::experimental_policy policy{"SIRIUS_EXP_FUSED_SCAN_ZERO_SKIP",
                                               "fused-scan-zero-skip"};
  return policy.enabled();
}

bool decompression_pushdown_latched_zero_check_enabled()
{
  static simpatico::experimental_policy policy{"SIRIUS_EXP_FUSED_SCAN_LATCH_ZERO_CHECK",
                                               "fused-scan-latch-zero-check"};
  return policy.enabled();
}

bool decode_scratch_arena_enabled()
{
  static simpatico::experimental_policy policy{"SIRIUS_EXP_DECODE_SCRATCH_ARENA",
                                               "decode-scratch-arena"};
  return policy.enabled();
}

bool onpair_defer_preparation_enabled()
{
  static simpatico::experimental_policy policy{"SIRIUS_EXP_ONPAIR_DEFER_PREPARE",
                                               "onpair-defer-prepare"};
  return policy.enabled();
}

std::size_t decompression_pushdown_stage_bytes()
{
  static std::size_t const bytes = [] {
    auto const* value = std::getenv("SIRIUS_EXP_FUSED_SCAN_STAGE_BYTES");
    if (!value) return std::size_t{64 * 1024 * 1024};
    char* end         = nullptr;
    auto const parsed = std::strtoull(value, &end, 10);
    if (value[0] == '-' || end == value || *end != '\0')
      throw std::invalid_argument("SIRIUS_EXP_FUSED_SCAN_STAGE_BYTES: expected byte count");
    return static_cast<std::size_t>(parsed);
  }();
  return bytes;
}

bool decompression_pushdown_cost_rule_enabled()
{
  static simpatico::experimental_policy policy{"SIRIUS_EXP_FUSED_SCAN_COST_RULE",
                                               "fused-scan-cost-rule"};
  return policy.enabled();
}

bool onpair_like_mask_above_cap_enabled()
{
  static simpatico::experimental_policy policy{"SIRIUS_EXP_ONPAIR_LIKE_MASK_ABOVE_CAP",
                                               "onpair-like-mask-above-cap"};
  return policy.enabled();
}

double parse_fused_scan_cost_coefficient(std::string_view text, double maximum)
{
  std::string const value{text};
  char* end         = nullptr;
  auto const number = std::strtod(value.c_str(), &end);
  if (end == value.c_str() || end != value.c_str() + value.size() || !std::isfinite(number) ||
      number < 0 || number > maximum)
    throw std::invalid_argument("expected finite cost in [0," + std::to_string(maximum) + "]");
  return number;
}

fused_scan_cost_constants const& fused_scan_cost_configuration()
{
  static auto const constants = [] {
    auto read = [](char const* name, double fallback, double maximum) {
      auto const* value = std::getenv(name);
      if (!value) return fallback;
      try {
        return parse_fused_scan_cost_coefficient(value, maximum);
      } catch (std::invalid_argument const& e) {
        throw std::invalid_argument(std::string(name) + ": " + e.what());
      }
    };
    fused_scan_cost_constants c;
    c.per_column_us =
      read("SIRIUS_EXP_FUSED_SCAN_COST_PER_COLUMN_US", c.per_column_us, fused_scan_max_fixed_us);
    c.per_selected_string_us = read("SIRIUS_EXP_FUSED_SCAN_COST_PER_SELECTED_STRING_US",
                                    c.per_selected_string_us,
                                    fused_scan_max_fixed_us);
    c.per_like_us =
      read("SIRIUS_EXP_FUSED_SCAN_COST_PER_LIKE_US", c.per_like_us, fused_scan_max_fixed_us);
    c.per_row_ns =
      read("SIRIUS_EXP_FUSED_SCAN_COST_PER_ROW_NS", c.per_row_ns, fused_scan_max_row_ns);
    c.per_full_survivor_ns = read("SIRIUS_EXP_FUSED_SCAN_COST_PER_FULL_SURVIVOR_NS",
                                  c.per_full_survivor_ns,
                                  fused_scan_max_row_ns);
    c.per_avoided_row_ns   = read(
      "SIRIUS_EXP_FUSED_SCAN_COST_PER_AVOIDED_ROW_NS", c.per_avoided_row_ns, fused_scan_max_row_ns);
    c.narrow_columns = read(
      "SIRIUS_EXP_FUSED_SCAN_COST_NARROW_COLUMNS", c.narrow_columns, fused_scan_max_narrow_columns);
    return c;
  }();
  return constants;
}

fused_scan_cost estimate_fused_scan_cost(std::span<decode_route const> routes,
                                         std::int64_t rows,
                                         double survivor_fraction,
                                         std::size_t like_sources,
                                         fused_scan_cost_constants const& c,
                                         std::size_t decoded_like_sources)
{
  // 300 us/delivered column/chunk is the measured pre-staging incremental cost (90 vs
  // 57 ms for 105 columns). G4 must replace this with the measured staged cost. Common full
  // decode and final survivor writes cancel; selected string sizing and full gathers remain.
  double compact = c.per_column_us * 1e-3 * routes.size();
  auto const match = c.per_like_us * 1e-3 + rows * c.per_row_ns * 1e-6;
  double plain     = match * like_sources;
  compact += match * decoded_like_sources;
  for (auto route : routes) {
    if (route == decode_route::str_split || route == decode_route::onpair)
      compact += c.per_selected_string_us * 1e-3;
    if (route == decode_route::full)
      compact += rows * survivor_fraction * c.per_full_survivor_ns * 1e-6;
    else
      plain += rows * (1.0 - survivor_fraction) * c.per_avoided_row_ns * 1e-6;
  }
  return {compact, plain};
}

bool fused_scan_cost_refused(std::span<decode_route const> routes,
                             std::int64_t rows,
                             double survivor_fraction,
                             std::size_t like_sources,
                             fused_scan_cost_constants const& c,
                             std::size_t decoded_like_sources)
{
  if (!decompression_pushdown_cost_rule_enabled() || like_sources == 0 ||
      routes.size() <= c.narrow_columns)
    return false;
  auto const cost = estimate_fused_scan_cost(
    routes, rows, survivor_fraction, like_sources, c, decoded_like_sources);
  // A failed estimate cannot establish either profitability or a cost refusal. Keep the policy
  // that applied before this rule (including its full-route/selectivity guards).
  return cost.finite() && !cost.profitable();
}

double decompression_pushdown_max_selectivity()
{
  static double const value = env_fraction("SIRIUS_EXP_FUSED_SCAN_MAX_SEL", 0.35);
  return value;
}

double decompression_pushdown_full_route_max_selectivity()
{
  static double const value = env_fraction("SIRIUS_EXP_FUSED_SCAN_TIERB_MAX_SEL", 0.10);
  return value;
}

double decompression_pushdown_index_walk_max_selectivity()
{
  static double const value = env_fraction("SIRIUS_EXP_FUSED_SCAN_K4_MAX_SEL", 0.15);
  return value;
}

std::size_t decompression_pushdown_max_membership_sources()
{
  static std::size_t const cap = [] {
    char const* value = std::getenv("SIRIUS_EXP_FUSED_SCAN_MAX_MEMBER");
    if (value == nullptr || *value == '\0') { return static_cast<std::size_t>(1); }
    char* end              = nullptr;
    long long const parsed = std::strtoll(value, &end, 10);
    return (end != value && parsed >= 0) ? static_cast<std::size_t>(parsed)
                                         : static_cast<std::size_t>(1);
  }();
  return cap;
}

}  // namespace sirius::codegen
