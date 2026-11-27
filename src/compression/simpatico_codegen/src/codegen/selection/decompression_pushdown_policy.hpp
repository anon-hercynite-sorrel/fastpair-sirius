// SPDX-License-Identifier: Apache-2.0
//
// Every knob that decides whether, and how hard, a decode filters.
//
// One reader per knob, shared by everything that has to agree on it: the scan
// (does it bother analysing its filter?), the memory estimator (will the batch
// come back smaller?) and the decode itself. They used to be copied per
// translation unit, in three different parse styles, and had already drifted —
// one reader accepted only "1" where the decode accepts anything but "0", so a
// value like "true" turned the feature on in one layer and off in another.
//
// All are read once and cached: the gate check sits on the per-batch path.
//
// TODO: these should be Sirius parameters, not environment variables. As env
// vars they are invisible to duckdb_settings(), cannot be set per session, and
// — because the values cache on first read — cannot be varied within one test
// binary, which is why some decode paths can only be reached by a unit test
// rather than end to end. The obstacle is layering: this library has no DuckDB
// dependency, so the values cannot be pulled from the setting registry here.
// They would have to arrive from Sirius, either pushed down once per query or
// passed into the decode call as a policy value. Left as env while the feature
// is experimental; revisit before it ships.

#pragma once

#include "codegen/selection/selection.hpp"

#include <cmath>
#include <cstddef>
#include <span>
#include <string_view>

namespace sirius::codegen {

/// Master gate (SIRIUS_EXP_FUSED_SCAN_FILTER). Set and not exactly "0" = on.
bool decompression_pushdown_enabled();

/// Promote the decision trace to INFO / stderr (SIRIUS_EXP_FUSED_SCAN_DIAG),
/// same "set and not exactly 0" contract.
///
/// The trace is permanent tooling, not temporary instrumentation: it records
/// every accept/decline decision, and raising the level is the first move
/// whenever a batch quietly falls back to a plain decode.
bool decompression_pushdown_diag_enabled();

/// Stage full-route decodes before table gather and cache immutable ANS frame metadata at pin time.
/// SIRIUS_EXP_FUSED_SCAN_BATCHED_WAVE2, default on.
bool decompression_pushdown_batched_wave2_enabled();
std::size_t decompression_pushdown_stage_bytes();

/// Return a typed empty column for a `full`-route output when the chunk has no survivors, instead
/// of decoding full width and gathering nothing. Covers fixed-width and STRING outputs of
/// non-nullable plans; other outputs keep the decode + gather path.
/// SIRIUS_EXP_FUSED_SCAN_ZERO_SKIP, default on.
bool decompression_pushdown_full_route_zero_skip_enabled();

/// After a scan reports declined_unselective, its later batches still run wave 1 to find chunks
/// with no survivors (scan_filter_request::zero_survivor_only) instead of dropping row selection
/// outright. A chunk with survivors declines and takes the plain decode, as before.
/// SIRIUS_EXP_FUSED_SCAN_LATCH_ZERO_CHECK, default on.
bool decompression_pushdown_latched_zero_check_enabled();

/// A per-batch read of a host- or disk-resident pin leaves OnPair columns unprepared unless the
/// scan evaluates LIKE on them; the decode that needs a column prepares it, so a batch whose filter
/// keeps no row never does. SIRIUS_EXP_ONPAIR_DEFER_PREPARE, default on.
bool onpair_defer_preparation_enabled();

/// Per-call bounded decode scratch suballocation; SIRIUS_EXP_DECODE_SCRATCH_ARENA, default on.
bool decode_scratch_arena_enabled();

/// Compare estimated compaction overhead with saved decode/filter work; default on.
bool decompression_pushdown_cost_rule_enabled();

/// SIRIUS_EXP_ONPAIR_LIKE_MASK_ABOVE_CAP, default on. Filter-only LIKE answers above the
/// compaction cap are delivered full-width as BOOL8; projected LIKE declines to ordinary decode.
bool onpair_like_mask_above_cap_enabled();

// Calibration envelopes: <= 1 second per fixed contribution, <= 1 ms per row. These
// deliberately exceed measured costs by orders of magnitude while bounding products for INT64
// row counts and size_t route/source counts. The measured exemption covers at most four columns.
inline constexpr double fused_scan_max_fixed_us       = 1'000'000;
inline constexpr double fused_scan_max_row_ns         = 1'000'000;
inline constexpr double fused_scan_max_narrow_columns = 4;

// Shared with parser tests; throws for malformed, non-finite or out-of-range overrides.
double parse_fused_scan_cost_coefficient(std::string_view text, double maximum);

struct fused_scan_cost_constants {
  double per_column_us          = 300.0;
  double per_selected_string_us = 10.0;
  double per_like_us            = 150.0;
  double per_row_ns             = 0.1;
  double per_full_survivor_ns   = 0.01;
  double per_avoided_row_ns     = 0.015;
  // Policy exemption: number of DELIVERED columns at or below which the cost rule accepts
  // the fused route without an arithmetic profitability requirement. On G4, at <= 4 delivered
  // columns the fused route's fixed cost is below plain decode of any string column at these
  // measured row counts. Preserve those measured narrow-query wins while calibrating wide
  // projections. SIRIUS_EXP_FUSED_SCAN_COST_NARROW_COLUMNS is in [0,4]; 0 disables the exemption.
  // Coefficient overrides do not turn this policy acceptance into an arithmetic guarantee.
  double narrow_columns = 4;
};

// Read once. All six coefficients have SIRIUS_EXP_FUSED_SCAN_COST_* overrides.
fused_scan_cost_constants const& fused_scan_cost_configuration();

struct fused_scan_cost {
  double compact_ms;
  double plain_ms;
  bool finite() const { return std::isfinite(compact_ms) && std::isfinite(plain_ms); }
  bool profitable() const { return finite() && compact_ms <= plain_ms; }
};

/// Incremental costs after cancelling full decodes common to both paths. The zero-survival
/// estimate is a lower bound, allowing a chunk to decline before wave 1 when even that loses.
fused_scan_cost estimate_fused_scan_cost(
  std::span<decode_route const> routes,
  std::int64_t rows,
  double survivor_fraction,
  std::size_t like_sources,
  fused_scan_cost_constants const& constants = fused_scan_cost_configuration(),
  std::size_t decoded_like_sources           = 0);

// Both pre-wave and post-count decisions use this rule. A non-finite estimate disables the
// cost rule for this decision, leaving the pre-rule route/selectivity policy in force.
bool fused_scan_cost_refused(
  std::span<decode_route const> routes,
  std::int64_t rows,
  double survivor_fraction,
  std::size_t like_sources,
  fused_scan_cost_constants const& constants = fused_scan_cost_configuration(),
  std::size_t decoded_like_sources           = 0);

/// Surviving-row fraction above which the decode gives compaction up and
/// produces ordinary full-width columns (SIRIUS_EXP_FUSED_SCAN_MAX_SEL,
/// default 0.35). Every batch that DOES come back compacted is bounded by it,
/// which is what makes sizing a memory reservation off it sound.
///
/// Measured: wins at sel <= .152, losses by .526; the mask walk costs about the
/// plain decode at .5.
double decompression_pushdown_max_selectivity();

/// The same, for a batch with any full-width output
/// (SIRIUS_EXP_FUSED_SCAN_TIERB_MAX_SEL, default 0.10): a full decode plus
/// gather costs about the unfiltered path, so the win is the compacted batch
/// (and, when the decode carries the whole filter, the skipped post-filter),
/// which only pays off at low selectivity.
double decompression_pushdown_full_route_max_selectivity();

/// Surviving-row fraction at or below which walking the survivor index list
/// beats walking the mask bits (SIRIUS_EXP_FUSED_SCAN_K4_MAX_SEL, default
/// 0.15). The microbench crossover sits at 15-50% depending on bit width, so
/// this is the conservative edge. Setting it tiny is the effective kill switch
/// for the index walk (the parse requires > 0).
double decompression_pushdown_index_walk_max_selectivity();

/// How many join-filter probes one decode carries
/// (SIRIUS_EXP_FUSED_SCAN_MAX_MEMBER, default 1).
///
/// Wave-1 probes run at FULL width, so k probes cost k*N rows of probe+adapter work, while the
/// downstream operator's cascade costs ~1*N (later probes see compacted survivors). Measured with 3
/// probes: the added probe-side cost outweighed the compaction win. One probe is always
/// ~volume-neutral against the cascade's first probe, so the compaction win survives. Sources
/// beyond the cap are DROPPED, which is sound because a join filter is optional: the join stays
/// authoritative, so a dropped probe only lets through rows the join rejects. The caller must order
/// the probes by ascending expected keep-rate so the kept prefix is the strongest.
std::size_t decompression_pushdown_max_membership_sources();

}  // namespace sirius::codegen
