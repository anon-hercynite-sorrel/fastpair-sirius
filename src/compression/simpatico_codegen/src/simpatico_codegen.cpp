// SPDX-License-Identifier: Apache-2.0
#include "api/simpatico_codegen.hpp"

#include "codegen/plan/plan_interpreter.hpp"
#include "codegen/plan/representation.hpp"
#include "codegen/selection/decompression_pushdown_policy.hpp"
#include "codegen/selection/selection.hpp"
#include "codegen/util/cuda_check.hpp"
#include "codegen/util/nvtx.hpp"
#include "codegen/util/stream_pool.hpp"
#include "decode/decode_scratch_resource.hpp"
#include "decode/decode_session.hpp"
#ifdef SIMPATICO_ONPAIR
#include "operators/onpair_compressor.hpp"
#endif
#include "util/decoded_like.hpp"
#include "util/host_observation.hpp"

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/copying.hpp>
#include <cudf/detail/utilities/stream_pool.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/traits.hpp>

#include <rmm/device_buffer.hpp>

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <limits>
#ifndef SIMPATICO_USE_CUCASCADE_STREAM_POOL
#include <map>
#include <mutex>
#endif
#include <numeric>
#include <optional>
#include <ranges>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace simpatico {

namespace {

// cuDF 26.08 joins RMM stream views; 26.10+ joins CUDA stream refs.
using cudf_pool_stream =
  decltype(cudf::detail::fork_streams(cudf::get_default_stream(), 0))::value_type;

// ── Internal helpers for the public compress/decompress API ───────────────────
// (Formerly compress_internals.hpp; this TU is the only consumer.)

class plan_error : public std::runtime_error {
 public:
  explicit plan_error(std::string const& msg) : std::runtime_error(msg) {}
};

// A host-only upper bound on retained decoded buffers. Unknown variable-width layouts use
// the interleaved path: compressed bytes are never a bound on decompressed string bytes.
std::optional<std::size_t> staged_column_bytes(compressed_column const& column)
{
  auto const rows = static_cast<std::size_t>(column.num_rows);
  if (cudf::is_fixed_width(column.dtype)) return rows * cudf::size_of(column.dtype);
  if (column.dtype.id() != cudf::type_id::STRING) return std::nullopt;
  for (auto const& node : column.plan_tree->nodes) {
    if (node.input_sources.size() != 1 || node.input_sources[0] != ValueId{0, 0}) continue;
    // INT64 offsets also bound decoders that choose INT32 offsets.
    auto const offsets = (rows + 1) * sizeof(std::int64_t);
    if (auto const* meta = std::get_if<leaf_meta::onpair>(&node.meta))
      return offsets + meta->decoded_bytes;
    if (auto const* split =
          dynamic_cast<str_split_compressed_representation const*>(node.rep.get());
        split && split->channels_.size() == 2)
      return offsets + static_cast<std::size_t>(split->channels_[1]->size()) *
                         cudf::size_of(split->channels_[1]->type());
  }
  return std::nullopt;
}

// Move every buffer's eventual free to the stream that last consumes it, preserving its mr.
// A table gather may read children (offsets/chars) as well as the parent data and null mask.
void release_column_on(cudf::column& column, ::cuda::stream_ref stream)
{
  auto contents = column.release();
  if (contents.data) contents.data->set_stream(stream);
  if (contents.null_mask) contents.null_mask->set_stream(stream);
  for (auto& child : contents.children)
    release_column_on(*child, stream);
  // Destruction queues frees on the gather stream. The input column is no longer needed;
  // rebuilding it would be both unnecessary and incompatible with cudf::column's deleted
  // move assignment operator.
}

std::string trim_plan_block(std::string s)
{
  while (!s.empty() &&
         (s.back() == '\n' || s.back() == '\r' || s.back() == ' ' || s.back() == '\t'))
    s.pop_back();
  size_t start = 0;
  while (start < s.size() && (s[start] == ' ' || s[start] == '\t'))
    ++start;
  return s.substr(start);
}

// Split a multi-column DSL string on "---" separators, skipping blank lines
// and comment lines (beginning with '#'). Each returned block is trimmed.
std::vector<std::string> split_plan_dsl_impl(std::string_view plan_dsl)
{
  std::vector<std::string> plans;
  std::string current;
  size_t i = 0;
  while (i < plan_dsl.size()) {
    size_t line_end = plan_dsl.find('\n', i);
    if (line_end == std::string_view::npos) line_end = plan_dsl.size();
    std::string_view line = plan_dsl.substr(i, line_end - i);
    if (!line.empty() && line.back() == '\r') line.remove_suffix(1);

    std::string_view trimmed = line;
    while (!trimmed.empty() && trimmed.front() == ' ')
      trimmed.remove_prefix(1);
    while (!trimmed.empty() && trimmed.back() == ' ')
      trimmed.remove_suffix(1);

    if (trimmed == "---") {
      auto block = trim_plan_block(current);
      if (!block.empty()) plans.push_back(std::move(block));
      current.clear();
    } else if (!trimmed.empty() && trimmed.front() != '#') {
      current.append(trimmed);
      current.push_back('\n');
    }
    i = (line_end == plan_dsl.size()) ? plan_dsl.size() : line_end + 1;
  }
  auto block = trim_plan_block(current);
  if (!block.empty()) plans.push_back(std::move(block));
  return plans;
}

void validate_plan_count(size_t plan_count, int table_columns)
{
  if (plan_count != static_cast<size_t>(table_columns)) {
    throw plan_error("plan count (" + std::to_string(plan_count) +
                     ") does not match table.num_columns() (" + std::to_string(table_columns) +
                     ")");
  }
}

void validate_column_names(std::vector<std::string> const& column_names, size_t num_columns)
{
  if (!column_names.empty() && column_names.size() != num_columns) {
    throw plan_error("column_names size (" + std::to_string(column_names.size()) +
                     ") does not match num_columns (" + std::to_string(num_columns) + ")");
  }
}

#ifndef SIMPATICO_USE_CUCASCADE_STREAM_POOL
// Process-lifetime cache of CUDA streams for the internal `int column_threads`
// overloads. These overloads have no caller-owned pool, yet the objects they
// return (a cudf::table, or a compressed_table whose leaf buffers live in cudf
// columns) record the stream they were built on for their eventual async free.
// If that stream were a per-call stream_pool destroyed on return, freeing the
// result later would deallocate on a dangling stream handle — a use-after-free
// with an async memory resource. Leasing from a cache that NEVER destroys its
// streams keeps every recorded handle valid for the process lifetime, so the
// result is safe to free by any stream (including the RMM default) with no
// external rebinding. Streams are recycled between calls, so this also avoids
// per-call stream create/destroy churn.
// CUDA streams are device-bound, so recycled streams are keyed by device.
class stream_cache {
 public:
  // The caller must have `device` current when new streams are created.
  std::vector<cudaStream_t> checkout(int device, size_t n)
  {
    std::vector<cudaStream_t> out;
    out.reserve(n);
    std::lock_guard<std::mutex> lock(mu_);
    auto& free_list = free_[device];
    while (out.size() < n && !free_list.empty()) {
      out.push_back(free_list.back());
      free_list.pop_back();
    }
    while (out.size() < n) {
      cudaStream_t s{};
      if (cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking) != cudaSuccess) break;
      out.push_back(s);
    }
    return out;
  }

  // Return streams to the same device list they were checked out from.
  void check_in(int device, std::vector<cudaStream_t>& streams)
  {
    std::lock_guard<std::mutex> lock(mu_);
    auto& free_list = free_[device];
    free_list.insert(free_list.end(), streams.begin(), streams.end());
    streams.clear();
  }

 private:
  std::mutex mu_;
  std::map<int, std::vector<cudaStream_t>> free_;
};

stream_cache& global_stream_cache()
{
  static stream_cache cache;
  return cache;
}

#endif

// RAII lease of max(1, column_threads) streams for an internal-parallel call.
// Both backends keep stream owners alive after the call so returned buffers
// can safely use their recorded allocation streams for asynchronous frees.
// Concurrent leases get disjoint streams, so sync_all only touches this call's
// streams. Capture the device for the standalone cache's checkout/check-in.
struct leased_pool {
  stream_pool pool;
  int device = 0;

  explicit leased_pool(int column_threads)
  {
    if (cudaGetDevice(&device) != cudaSuccess)
      throw plan_error("failed to query the current device for the internal stream lease");

#ifdef SIMPATICO_USE_CUCASCADE_STREAM_POOL
    if (!pool.init(static_cast<size_t>(std::max(1, column_threads))))
      throw plan_error("failed to lease internal streams");
#else
    pool.streams =
      global_stream_cache().checkout(device, static_cast<size_t>(std::max(1, column_threads)));

    if (pool.streams.empty()) throw plan_error("failed to lease internal streams");
#endif
  }

  ~leased_pool()
  {
    // Completed compression/decode scopes already surfaced asynchronous errors.
    // A lease destructor cannot throw.
#ifndef SIMPATICO_USE_CUCASCADE_STREAM_POOL
    (void)pool.sync_all();
    global_stream_cache().check_in(
      device, pool.streams);  // leaves pool.streams empty; ~stream_pool is a no-op
#endif
  }

  leased_pool(const leased_pool&)            = delete;
  leased_pool& operator=(const leased_pool&) = delete;
};

// Submit `body(i, stream)` for every index in [0, n_items) across the pool
// streams from the calling thread (round-robin), then synchronise all streams.
// No worker threads are spawned: CUDA stream submission is asynchronous, so
// the GPU can overlap column work across pool streams while the CPU submits
// serially. All allocations happen on the calling thread, keeping
// cuCascade's per-thread memory-reservation accounting correct.
template <typename Body>
void run_column_workers(size_t n_items, stream_pool& pool, Body&& body)
{
  size_t const n_streams = pool.streams.size();
  if (n_streams == 0) throw plan_error("stream_pool has no streams");
  std::exception_ptr first_exception;
  for (size_t i = 0; i < n_items; ++i) {
    ::cuda::stream_ref s{pool.streams[i % n_streams]};
    try {
      body(i, s);
    } catch (...) {
      if (!first_exception) first_exception = std::current_exception();
      break;
    }
  }
  cudaError_t sync_err = pool.sync_all();
  if (first_exception) std::rethrow_exception(first_exception);
  if (sync_err != cudaSuccess) {
    throw plan_error(std::string("column worker stream sync failed: ") +
                     cudaGetErrorString(sync_err));
  }
}

compressed_table compress_columns_parallel(cudf::table_view table,
                                           std::vector<std::string> const& plans,
                                           stream_pool& pool,
                                           rmm::device_async_resource_ref mr,
                                           std::vector<std::string> const& column_names)
{
  compressed_table out;
  out.columns.resize(plans.size());
  run_column_workers(plans.size(), pool, [&](size_t i, ::cuda::stream_ref stream) {
    std::string err;
    auto plan_tree =
      compress_column(table.column(static_cast<cudf::size_type>(i)), plans[i], stream, mr, &err);
    if (!plan_tree) throw plan_error(err.empty() ? "compress failed" : err);
    compressed_column col;
    col.dtype     = table.column(static_cast<cudf::size_type>(i)).type();
    col.num_rows  = table.num_rows();
    col.plan_tree = std::move(plan_tree);
    if (!column_names.empty()) col.name = column_names[i];
    out.columns[i] = std::move(col);
  });
  return out;
}

// Every table facade submits the same typed requests to one session. Streams and the explicit
// allocator are borrowed by the session.
std::unique_ptr<cudf::table> decode_table_columns(
  compressed_table const& table,
  std::span<std::size_t const> selected,
  std::span<decode_predicate const> predicates,
  std::span<const ::cuda::stream_ref> streams,
  rmm::device_async_resource_ref mr,
  std::optional<rmm::device_async_resource_ref> scratch = {})
{
  for (auto index : selected) {
    if (index >= table.columns.size()) throw plan_error("selected column index out of range");
    if (!table.columns[index].plan_tree) throw plan_error("decompress: column missing plan tree");
  }
  prepare_deferred_onpair(table, selected, streams, mr, scratch);
  decode_session session{streams, mr, scratch.value_or(mr)};
  for (std::size_t i = 0; i < selected.size(); ++i) {
    auto const& column = table.columns[selected[i]];
    column_decode_request request{std::cref(*column.plan_tree), value_result{column.dtype}, {}};
    if (!predicates.empty() && predicates[i].active()) {
      request.result = predicate_result{predicates[i], {}};
    }
    session.append(std::move(request));
  }
  return std::make_unique<cudf::table>(session.finish());
}

std::vector<std::size_t> all_columns(compressed_table const& table)
{
  std::vector<std::size_t> selected(table.num_columns());
  std::iota(selected.begin(), selected.end(), std::size_t{0});
  return selected;
}

// ── Filtering while decoding (env gate SIRIUS_EXP_FUSED_SCAN_FILTER) ────────
//
// Two waves inside one converter call:
//   wave 1: ballot the filter columns into mask words, round-robin on the pool
//           streams; stream 0 waits on the others (events, no host sync),
//           AND-combines, counts (per-chunk popcount + CUB scan -> chunk_offsets)
//           and D2H's the survivor count — the one added host sync, and it
//           gates wave-2 allocations. Membership probes are wave-1 sources
//           only when there is no range, BOOL8, or keep-mask source;
//           otherwise they run after the combine and before the count, one
//           at a time on stream 0, each given the running combined mask as
//           its prior.
//   wave 2: compactable columns decode straight to survivor width; the rest
//           decode full width and gather to survivor rows on their own
//           streams, in parallel, each waiting first for the gather map
//           (mask -> int32 row indices) built once on stream 0.
// Semantic preconditions and selectivity policy may decline to ordinary decode.
// Execution failures propagate after checked cleanup, preserving engine OOM recovery.
//
// Enable policy, in two stages:
//   Statically, before any device work: a column asking for a compacted route
//           must probe as that route; `full` is admitted for anything, with its
//           economics deferred below. Range-filter sources are exempt and
//           forced to bitpack_mask.
//   Dynamically, once the survivor count is known — both regimes report
//           declined_unselective, so the caller can remember it for the scan:
//           - any `full` output: proceed iff survivors/rows <= TIERB_MAX_SEL
//             (0.10);
//           - otherwise: give compaction up above MAX_SEL (0.35), unless a
//             dict_codes output is present (that gather wins at every
//             selectivity).
//           Giving up = masks dropped, ordinary decode, batch not row-filtered.

// The index walk is reachable from here (its launcher and the
// decode_selection.prefer_index_decode routing are both live). Effective kill
// switch: set SIRIUS_EXP_FUSED_SCAN_K4_MAX_SEL to a tiny value (the parse
// requires > 0).

// Both waves submit typed session requests through the same plan interpreter.
// probe_column describes row-route semantics, never asynchronous eligibility.

// A semantic precondition or completed selectivity decision may decline this
// optimization. Execution failures drain the phase and propagate unchanged;
// allocation, compilation, and device errors never become ordinary-decode retries.
std::optional<std::vector<std::unique_ptr<cudf::column>>> try_decompress_fused(
  compressed_table const& table,
  std::span<const std::size_t> selected,
  sirius::codegen::scan_filter_request const& request,
  sirius::codegen::scan_filter_result& result,
  std::vector<std::unique_ptr<cudf::column>>& decoded_likes,
  std::span<const ::cuda::stream_ref> streams,
  rmm::device_async_resource_ref mr,
  rmm::device_async_resource_ref scratch_mr)
{
  namespace sc = sirius::codegen;
  auto const gather_mr = [&] {
    // Preserve cuDF's lazy current-resource lookup at each actual gather when the gate is off.
    return scratch_mr == mr ? cudf::memory_resources{mr} : cudf::memory_resources{mr, scratch_mr};
  };

  // Preconditions — all checked before touching the device. A refusal is a
  // normal-path decision (the caller runs the unfiltered decode, byte-identical
  // to what it would have run anyway), not an error; the reason line only
  // appears under SIRIUS_EXP_FUSED_SCAN_DIAG.
  //
  // Most of these are structural — index bounds, arity, chunk geometry, the
  // source cap — and protect the kernels from a request that would fault or
  // address outside a packed region.
  //
  // The four that call probe_column are different in kind and worth keeping
  // deliberately. The caller narrows its request with the SAME probe before
  // sending it, so these cannot disagree with it; they are assertions on the
  // boundary, not a second opinion. They stay because the failure they catch is
  // a WRONG MASK rather than a crash — render a range ballot over a plan that
  // is not bitpack-rooted and it reads the wrong bits and silently drops the
  // wrong rows — and because the caller's guarantee holds only through a chain
  // of reasoning across several files (a request reaches us only via
  // decompression_pushdown_scan::for_chunk, whose narrowing is what makes the claim true).
  // They are host-side plan-tree walks, once per batch, against device work
  // measured in milliseconds.
  auto refuse = [](char const* why) {
    if (sc::decompression_pushdown_diag_enabled())
      std::fprintf(stderr, "simpatico: filtered decode refused: %s\n", why);
    return std::nullopt;
  };
  if (!sc::decompression_pushdown_enabled()) return refuse("env gate off");
  size_t const k_range = request.filters.size();
  size_t const k_bool8 = request.bool8_filters.size();
  size_t const k_like  = request.like_filters.size();
  // Membership cap (drop-tail, sound — see max_membership_sources).
  size_t const k_member    = std::min(request.membership_filters.size(),
                                      sc::decompression_pushdown_max_membership_sources());
  size_t const k_total     = k_range + k_bool8 + k_like + k_member;
  result.source_generation = request.source_generation;  // echoed on every outcome
  if (k_member < request.membership_filters.size() && sc::decompression_pushdown_diag_enabled()) {
    std::fprintf(stderr,
                 "simpatico: filtered decode membership sources capped %zu -> %zu "
                 "(SIRIUS_EXP_FUSED_SCAN_MAX_MEMBER)\n",
                 request.membership_filters.size(),
                 k_member);
  }
  if (k_total == 0) return refuse("no mask directives (no range/bool8/membership)");
  // The keep mask is a combine source, never a directive: with no real filter the
  // survivor rate is just the visible-row fraction, so compaction cannot pay off.
  bool const has_keep_mask = request.keep_mask_words != nullptr;
  if (k_total + (has_keep_mask ? 1 : 0) > 8) return refuse("more than 8 mask sources");
  if (request.routes.size() != selected.size())
    return refuse("request.routes not parallel to selected");
  std::size_t const delivered = request.delivered_prefix.value_or(selected.size());
  if (delivered == 0 || delivered > selected.size())
    return refuse("delivered_prefix outside [1, selected.size()]");
  if (request.row_count_only && (k_like == 0 || request.delivered_prefix))
    return refuse("row-count carrier requires LIKE and no delivered prefix");
  if (streams.empty()) return refuse("no streams supplied");
  int64_t const num_rows = table.num_rows();
  if (num_rows <= 0) return refuse("num_rows <= 0");
  if (num_rows > std::numeric_limits<std::int32_t>::max())
    return refuse("num_rows > INT32_MAX (int32 row indices)");
  if (has_keep_mask && request.keep_mask_rows != num_rows)
    return refuse("keep mask row count != batch rows (positional mask misaligned)");
  // Code-level LIKE has priority. Only a decoded LIKE slot can consume nullable values
  // independently of its compact route: wave 2 gathers the already-valid STRING column.
  std::vector<std::size_t> decoded_slots;
  std::size_t decoded_patterns = 0;
  for (auto const& directive : request.like_filters) {
    if (directive.column >= selected.size())
      return refuse("LIKE directive has an invalid column or pattern");
    auto const index = selected[directive.column];
    if (index >= table.columns.size() || !table.columns[index].plan_tree)
      return refuse("LIKE directive has no column plan");
    auto const& column = table.columns[index];
    if (probe_column(*column.plan_tree).can_answer_like) {
      if (!directive.pattern.valid()) return refuse("invalid code-level LIKE pattern");
      continue;
    }
    if (!directive.pattern.valid(/*require_ascii=*/false))
      return refuse("invalid decoded LIKE pattern");
    if (!decoded_like_enabled() || column.dtype.id() != cudf::type_id::STRING)
      return refuse("no enabled LIKE route");
    ++decoded_patterns;
    if (std::ranges::find(decoded_slots, directive.column) == decoded_slots.end())
      decoded_slots.push_back(directive.column);
  }
  auto const decodes_like = [&](std::size_t slot) {
    return std::ranges::find(decoded_slots, slot) != decoded_slots.end();
  };
  for (std::size_t i = 0; i < selected.size(); ++i) {
    auto const idx = selected[i];
    if (idx >= table.columns.size()) return refuse("selected column index out of range");
    auto const& col = table.columns[idx];
    if (!col.plan_tree || col.num_rows != num_rows)
      return refuse("selected column missing plan_tree or row-count mismatch");
    if (!col.plan_tree->validity.empty() && !decodes_like(i) &&
        (request.routes[i] != sc::decode_route::str_split ||
         probe_column(*col.plan_tree).compact_route != sc::decode_route::str_split))
      return refuse("selected column is nullable");
  }
  for (auto const& f : request.filters) {
    if (f.column >= selected.size()) return refuse("filter directive column out of range");
    if (f.pred.lo > f.pred.hi) return refuse("empty predicate range (lo > hi)");
    // can_produce_mask, not merely "decodes compacted": a dictionary column
    // decodes compacted but cannot ballot a numeric range, and one arriving as
    // a range source would be a latent wrong-mask gate.
    if (!probe_column(*table.columns[selected[f.column]].plan_tree).can_produce_mask())
      return refuse("filter column plan is not a bitpack-rooted range source");
  }
  for (auto const& b : request.bool8_filters) {
    if (b.column >= selected.size()) return refuse("bool8 directive column out of range");
    if (b.equals_any.empty()) return refuse("bool8 directive with empty equals_any");
    // The BOOL8 source rides the shipped dict-code pushdown: dictionary-rooted
    // plans only (the generic fallback would full-decode + compare — no win).
    if (!probe_column(*table.columns[selected[b.column]].plan_tree).can_answer_equality)
      return refuse("bool8 filter column plan not dictionary-rooted");
  }
  for (size_t mi = 0; mi < k_member; ++mi) {  // only the kept (capped) prefix
    auto const& m = request.membership_filters[mi];
    if (m.column >= selected.size()) return refuse("membership directive column out of range");
    if (!m.probe) return refuse("membership directive with an empty probe");
    // No plan-shape constraint: the key column decodes full width in wave 1
    // (any decodable plan) and takes its own tier in wave 2 like any output.
  }

  // A column answered off a dictionary delivers gather(bool8_fullwidth,
  // row_indices) — the compacted BOOL8 answer at the slot, no value decode, no
  // string re-compare downstream. These slots bypass the tier check below
  // (delivery is route-independent) and are EXCLUDED from the `full` selectivity
  // regime (a 1 B/row gather is cheap, not a full decode plus gather).
  std::vector<char> is_bool8_slot(selected.size(), 0);
  std::vector<int> bool8_of_slot(selected.size(), -1);
  for (size_t b = 0; b < k_bool8; ++b) {
    if (decodes_like(request.bool8_filters[b].column))
      return refuse("conflicting LIKE and BOOL8 value slot");
    if (is_bool8_slot[request.bool8_filters[b].column])
      return refuse("duplicate bool8 directives on one column");
    is_bool8_slot[request.bool8_filters[b].column] = 1;
    bool8_of_slot[request.bool8_filters[b].column] = static_cast<int>(b);
  }

  // Static check, zero-cost: a requested compacted route must match the plan's
  // own; `full` is admitted for anything and takes the wave-2 full-decode +
  // survivor-gather path, with its economics enforced once the survivor count
  // is known (a full-width decode plus gather costs about the unfiltered path,
  // so only low-selectivity batches pay off; measured losses at high
  // selectivity: q1 +43.5%, q5 +6.2%). Range-filter source columns are exempt
  // from the check (probed bitpack-rooted above) and forced to bitpack_mask;
  // dictionary-answered sources get no exemption — they carry their own
  // route.
  std::vector<sc::decode_route> routes(request.routes.begin(), request.routes.end());
  for (auto const& f : request.filters)
    routes[f.column] = sc::decode_route::bitpack_mask;
  for (size_t i = 0; i < selected.size(); ++i) {
    if (is_bool8_slot[i]) continue;  // the slot's output is the compacted BOOL8 answer
    // `full` is always available — every plan decodes full width, and a
    // null-masked column declines the batch rather than corrupting it. Any
    // other requested route must be the one this plan
    // supports; one probe answers that, so a route and a capability cannot
    // disagree.
    if (routes[i] == sc::decode_route::full) { continue; }
    if (routes[i] != probe_column(*table.columns[selected[i]].plan_tree).compact_route) {
      return refuse("requested decode route does not match the column's plan shape");
    }
  }

  // A delivered decoded-LIKE value costs a survivor gather. Its match is paid on either path.
  for (auto slot : decoded_slots)
    routes[slot] = sc::decode_route::full;
  auto const cost_refuses = [&](double fraction) {
    if (request.row_count_only || !sc::fused_scan_cost_refused(std::span{routes.data(), delivered},
                                                               num_rows,
                                                               fraction,
                                                               k_like,
                                                               sc::fused_scan_cost_configuration(),
                                                               decoded_patterns))
      return false;
    auto const cost = sc::estimate_fused_scan_cost(std::span{routes.data(), delivered},
                                                   num_rows,
                                                   fraction,
                                                   k_like,
                                                   sc::fused_scan_cost_configuration(),
                                                   decoded_patterns);
    if (sc::decompression_pushdown_diag_enabled())
      std::fprintf(stderr,
                   "[decompression-pushdown] reason=cost columns=%zu sel=%.6f "
                   "compact_ms=%.4f plain_ms=%.4f\n",
                   delivered,
                   fraction,
                   cost.compact_ms,
                   cost.plain_ms);
    result.status   = sc::scan_filter_status::declined_cost;
    result.num_rows = num_rows;  // the pre-wave refusal has no survivor count, but knows its domain
    return true;
  };
  // At zero survivors the rule gives compaction every benefit. If even that loses, skip wave 1
  // too: this chunk pays precisely the LIKE-off decode followed by its ordinary residual filter.
  if (cost_refuses(0.0)) return refuse("reason=cost");

  if (sc::decompression_pushdown_diag_enabled()) {
    std::string line = "simpatico: filtered decode wave-1 sources:";
    for (auto const& f : request.filters) {
      line += " range(col ";
      line += std::to_string(f.column) + " [" + std::to_string(f.pred.lo) + "," +
              std::to_string(f.pred.hi) + "])";
    }
    for (size_t mi = 0; mi < k_member; ++mi) {
      line += " member(col " + std::to_string(request.membership_filters[mi].column) + ")";
    }
    for (auto const& b : request.bool8_filters) {
      line += " bool8(col " + std::to_string(b.column) + " eq#" +
              std::to_string(b.equals_any.size()) + ")";
    }
    for (auto const& directive : request.like_filters) {
      line += " like(col " + std::to_string(directive.column) + " literals#" +
              std::to_string(directive.pattern.literals.size()) +
              " invert=" + std::to_string(directive.pattern.invert) + ")";
    }
    if (has_keep_mask) { line += " keep-mask"; }
    line += " rows=" + std::to_string(num_rows);
    std::fprintf(stderr, "%s\n", line.c_str());
  }

  // Declared before the try so the mid-flight catch can synchronize the streams BEFORE these
  // buffers unwind (their stream-ordered frees must not race the combine's cross-stream reads).
  std::vector<rmm::device_buffer> per_filter;
  rmm::device_buffer keep_mask_dev;
  // Wave-1 column results in submission order: BOOL8 answers and retained STRING values.
  std::vector<std::unique_ptr<cudf::column>> wave1_columns(request.bool8_filters.size());
  std::vector<::cuda::stream_ref> bool8_lanes;
  // Probes that declined this chunk and were stood down to the AND identity.
  size_t declined_members = 0;
  auto reset_result       = [&](sc::scan_filter_status status) {
    result                   = sc::scan_filter_result{};
    result.status            = status;
    result.source_generation = request.source_generation;
  };
  try {
    int64_t const nc          = sc::selection_mask::ChunksFor(num_rows);
    int64_t const alloc_words = sc::selection_mask::AllocWordsFor(num_rows);
    auto const mask_bytes     = static_cast<std::size_t>(alloc_words) * sizeof(std::uint32_t);
    ::cuda::stream_ref s0     = streams.front();
    auto const same_stream    = [](::cuda::stream_ref stream) {
      return [stream](::cuda::stream_ref other) { return other.get() == stream.get(); };
    };

    bool const count_only = request.row_count_only && !decoded_slots.empty();
    result.num_rows   = num_rows;
    result.mask_words = rmm::device_buffer(mask_bytes, s0, mr);
    if (!count_only)
      result.chunk_offsets =
        rmm::device_buffer(static_cast<std::size_t>(nc + 1) * sizeof(std::uint32_t), s0, mr);
    auto* combined = static_cast<std::uint32_t*>(result.mask_words.data());

    // ── Wave 1: one mask source per request, on the session's lanes in rotation. The first
    // request's lane is s0, so source 0 writes straight into the combined buffer (allocated on s0);
    // later sources write per-filter buffers allocated on the lane their request is assigned. Range
    // conjuncts run the range ballot; equality conjuncts run the BOOL8 pushdown and ballot it.
    per_filter.reserve(k_total > 1 ? k_total - 1 : 0);
    std::vector<std::uint32_t const*> mask_ptrs;
    mask_ptrs.reserve(k_total + 1);  // +1: the optional positional keep mask
    // Lanes other than s0 that produced a source, which the combine on s0 must wait for.
    std::vector<cudf_pool_stream> producer_lanes;
    if (!decoded_slots.empty()) {
      std::vector<std::size_t> physical;
      for (auto slot : decoded_slots)
        physical.push_back(selected[slot]);
      prepare_deferred_onpair(table, physical, streams, mr, scratch_mr);
    }
    decode_session wave1{streams, mr, scratch_mr};
    auto next_destination = [&] {
      auto const lane = wave1.next_stream();
      if (lane.get() != s0.get() &&
          std::none_of(producer_lanes.begin(), producer_lanes.end(), [&](auto other) {
            return static_cast<::cuda::stream_ref>(other).get() == lane.get();
          }))
        producer_lanes.emplace_back(lane);
      auto* words = mask_ptrs.empty()
                      ? combined
                      : static_cast<std::uint32_t*>(
                          per_filter.emplace_back(mask_bytes, lane, scratch_mr).data());
      mask_ptrs.push_back(words);
      return mask_destination{words, num_rows};
    };

    for (auto const& directive : request.filters) {
      auto const& column = table.columns[selected[directive.column]];
      if (wave1.append(mask_decode_request{
            *column.plan_tree, directive.pred, next_destination()}) != mask_source_status::ACCEPTED)
        throw plan_error("filtered decode: preflighted range source declined");
    }
    for (auto const& directive : request.bool8_filters) {
      auto const& column = table.columns[selected[directive.column]];
      bool8_lanes.push_back(wave1.next_stream());
      wave1.append(column_decode_request{
        std::cref(*column.plan_tree),
        predicate_result{decode_predicate{directive.equals_any}, next_destination()},
        {}});
    }
    std::vector<decoded_like_observation> like_observations;
    if (request.log_decoded_like) like_observations.reserve(decoded_patterns);
    std::vector<bool> submitted_like(selected.size(), false);
    std::vector<std::size_t> like_answer_of_slot(selected.size(), 0);
    std::size_t like_answer_count = k_bool8;
    for (auto const& directive : request.like_filters) {
      if (submitted_like[directive.column]) continue;
      submitted_like[directive.column]      = true;
      like_answer_of_slot[directive.column] = like_answer_count++;
      auto const& column                    = table.columns[selected[directive.column]];
      if (decodes_like(directive.column)) {
        decoded_like_result values{request.try_like_swar, {}};
        for (auto const& other : request.like_filters) {
          if (other.column != directive.column) continue;
          auto dest         = next_destination();
          auto* observation = request.log_decoded_like
                                ? &like_observations.emplace_back(decoded_like_observation{
                                    selected[directive.column], dest.words, {}})
                                : nullptr;
          values.filters.push_back({other.pattern, dest, observation});
        }
        wave1.append(column_decode_request{std::cref(*column.plan_tree), std::move(values), {}});
        continue;
      }
      decode_predicate predicate;
      for (auto const& other : request.like_filters)
        if (other.column == directive.column) predicate.likes.push_back(other.pattern);
      wave1.append(column_decode_request{std::cref(*column.plan_tree),
                                         predicate_result{std::move(predicate), next_destination()},
                                         {}});
    }
    // With another mask source to prime a prior from (a static range/bool8 strip or the
    // positional keep mask), membership probes run sequentially on s0 after the combine, so dead
    // rows skip the set/Bloom lookup. Otherwise they run concurrently here, prior-free.
    bool const sequential_membership =
      k_member > 0 && (k_range + k_bool8 + k_like > 0 || has_keep_mask);
    // Which membership probes ran, by request position, for the measurement.
    std::vector<bool> member_ran(k_member, false);
    for (size_t m = 0; !sequential_membership && m < k_member; ++m) {
      auto const& directive = request.membership_filters[m];
      auto const& column    = table.columns[selected[directive.column]];
      auto const status     = wave1.append(mask_decode_request{
        *column.plan_tree,
        membership_source{directive.probe,
                          column.dtype,
                          nullptr,
                          routes[directive.column] == sc::decode_route::str_split},
        next_destination()});
      declined_members += status == mask_source_status::DECLINED;
      member_ran[m] = status != mask_source_status::DECLINED;
    }

    // Order the combine on s0 after every other lane that produced a source.
    bool const probe_decoded_keys =
      !decoded_slots.empty() &&
      std::ranges::any_of(std::span{request.membership_filters}.first(k_member),
                          [&](auto const& filter) { return decodes_like(filter.column); });
    if (probe_decoded_keys)
      wave1_columns =
        wave1.handoff(s0);  // Orders the cascade's reads after the full value decodes.
    else if (!producer_lanes.empty())
      cudf::detail::join_streams(producer_lanes, s0);

    // ── Keep mask, uploaded before the combine so it primes the cascade's prior. No ballot
    // producer zeroes its tail (selection.hpp:52), so the gap between the host words and
    // alloc_words is memset here. Upload on s0 keeps the combine ordered behind it. When no source
    // has written `combined` yet, the keep mask lands there directly.
    if (has_keep_mask) {
      auto const host_words   = static_cast<std::size_t>((num_rows + 31) / 32);
      std::uint32_t* keep_dst = combined;
      if (!mask_ptrs.empty()) {
        keep_mask_dev = rmm::device_buffer(
          static_cast<std::size_t>(alloc_words) * sizeof(std::uint32_t), s0, scratch_mr);
        keep_dst = static_cast<std::uint32_t*>(keep_mask_dev.data());
      }
      if (static_cast<int64_t>(host_words) < alloc_words &&
          cudaMemsetAsync(
            keep_dst + host_words,
            0,
            (static_cast<std::size_t>(alloc_words) - host_words) * sizeof(std::uint32_t),
            s0.get()) != cudaSuccess) {
        throw plan_error("fused scan-filter: keep-mask padding memset failed");
      }
      if (cudaMemcpyAsync(keep_dst,
                          request.keep_mask_words,
                          host_words * sizeof(std::uint32_t),
                          cudaMemcpyHostToDevice,
                          s0.get()) != cudaSuccess) {
        throw plan_error("fused scan-filter: keep-mask upload failed");
      }
      mask_ptrs.push_back(keep_dst);
    }

    // ── Membership measurement counters on s0, copied back by the CNT wave's host sync. Counter m
    // holds the rows that pass every probe up to m (concurrent probes; their masks are counted
    // before the combine overwrites source 0), or, for the cascade, counter 0 holds its input and
    // counter m + 1 the rows that pass every probe up to m.
    bool const measure_members = request.measure_membership && k_member > 0;
    // A sole decoded LIKE already has its match count in CNT. With other sources,
    // count each LIKE before the combine and return those counts with the same host wait.
    bool const individual_like_counts  = k_total > 1 || has_keep_mask;
    std::size_t const membership_count = measure_members ? k_member + 1 : 0;
    auto const like_count              = individual_like_counts ? like_observations.size() : 0;
    std::vector<std::uint32_t> counters(membership_count + like_count + (count_only ? 1 : 0));
    rmm::device_buffer counter_storage;
    if (!counters.empty()) {
      counter_storage = rmm::device_buffer(counters.size() * sizeof(std::uint32_t), s0, scratch_mr);
      throw_if_cuda_error(
        cudaMemsetAsync(counter_storage.data(), 0, counter_storage.size(), s0.get()),
        "filtered decode: counter reset");
    }
    auto* counter_data = static_cast<std::uint32_t*>(counter_storage.data());
    for (size_t i = 0; i < like_count; ++i)
      sc::count_masked_rows(
        &like_observations[i].words, 1, alloc_words, counter_data + membership_count + i, s0);
    if (measure_members) {
      std::vector<std::uint32_t const*> ran_masks;
      for (size_t m = 0; !sequential_membership && m < k_member; ++m) {
        if (!member_ran[m]) continue;
        ran_masks.push_back(mask_ptrs[m]);
        sc::count_masked_rows(
          ran_masks.data(), static_cast<int>(ran_masks.size()), alloc_words, counter_data + m, s0);
      }
    }

    // ── Combine on stream 0.
    if (mask_ptrs.size() > 1) {
      sc::combine_masks_and(
        combined, mask_ptrs.data(), static_cast<int>(mask_ptrs.size()), alloc_words, s0);
    }

    // ── Sequential membership cascade, a single-lane session on s0 behind the combine. Each probe
    // takes `combined` as its prior and its result is ANDed back in, so the next probe sees the
    // tightened mask. A declining probe leaves `combined` untouched, the AND identity. The scratch
    // strip is written and read only on s0, so its stream-ordered free cannot race a reader.
    std::optional<decode_session> cascade;
    if (sequential_membership) {
      std::array<std::uint32_t const*, 1> const cascade_input{combined};
      if (measure_members) {
        sc::count_masked_rows(cascade_input.data(), 1, alloc_words, counter_data, s0);
      }
      cascade.emplace(std::span{&s0, 1}, mr, scratch_mr);
      rmm::device_buffer membership_scratch(mask_bytes, s0, scratch_mr);
      auto* member_words = static_cast<std::uint32_t*>(membership_scratch.data());
      std::array<std::uint32_t const*, 2> const and_sources{combined, member_words};
      for (size_t m = 0; m < k_member; ++m) {
        auto const& directive = request.membership_filters[m];
        auto const& column    = table.columns[selected[directive.column]];
        if (cascade->append(mask_decode_request{
              *column.plan_tree,
              membership_source{directive.probe,
                                column.dtype,
                                combined,
                                routes[directive.column] == sc::decode_route::str_split,
                                decodes_like(directive.column)
                                  ? wave1_columns[like_answer_of_slot[directive.column]].get()
                                  : nullptr},
              {member_words, num_rows}}) == mask_source_status::DECLINED) {
          ++declined_members;
          continue;
        }
        member_ran[m] = true;
        sc::combine_masks_and(combined, and_sources.data(), 2, alloc_words, s0);
        if (measure_members) {
          sc::count_masked_rows(cascade_input.data(), 1, alloc_words, counter_data + m + 1, s0);
        }
      }
    }

    // An all-ones source is the AND identity only while a real source still
    // zeroes the tail bits past num_rows, which CNT and the gather require. If
    // every source declined there is nothing left to filter by anyway, so take
    // the plain decode instead of counting a mask that is all ones. On a cascade
    // primed only by the keep mask, the survivor rate is just the visible-row
    // fraction, which the caller applies itself on the plain path.
    if (declined_members == k_total) {
      // Never count the padded all-ones masks: there is no real source to clear
      // the tail. This is an explicit policy decline after completed work.
      (void)wave1.finish();
      if (cascade) (void)cascade->finish();
      reset_result(sc::scan_filter_status::refused);
      return refuse("every membership source declined");
    }

    if (declined_members > 0 && sc::decompression_pushdown_diag_enabled()) {
      std::fprintf(stderr,
                   "simpatico: %zu of %zu membership probe(s) declined this chunk (rows=%lld); the "
                   "decode carries the rest and the join filters the remainder\n",
                   declined_members,
                   k_member,
                   static_cast<long long>(num_rows));
    }

    // ── CNT on stream 0. run_selection_cnt host-syncs s0 once (the survivor
    // count gates wave-2 allocations); after it returns, every wave-1 kernel,
    // the combine and the cascade have completed, so per_filter teardown is safe.
    sc::selection_mask sel{
      combined, num_rows, -1, static_cast<std::uint32_t*>(result.chunk_offsets.data())};
    if (count_only) {
      // COUNT(*) needs cardinality only. Skip chunk prefix sums and their CUB scratch.
      std::uint32_t const* source = combined;
      sc::count_masked_rows(&source, 1, alloc_words, counter_data + counters.size() - 1, s0);
      read_device_bytes_completed(counters.data(), counter_data, counter_storage.size(), s0);
      sel.survivor_count = counters.back();
    } else {
      sc::run_selection_cnt(sel, s0, scratch_mr, {counter_data, counters});
    }
    for (std::size_t i = 0; i < like_observations.size(); ++i) {
      auto const& obs = like_observations[i];
      request.log_decoded_like(
        obs.column,
        obs.codec,
        obs.matcher,
        num_rows,
        individual_like_counts ? counters[membership_count + i] : sel.survivor_count);
    }
    // Allocate before the first request's existing finish/drain so even an
    // async allocation is ready for the caller's stream when we return.
    std::unique_ptr<cudf::column> count_carrier;
    if (request.row_count_only)
      count_carrier =
        cudf::make_fixed_width_column(cudf::data_type{cudf::type_id::INT8},
                                      static_cast<cudf::size_type>(sel.survivor_count),
                                      cudf::mask_state::UNALLOCATED,
                                      s0,
                                      mr);
    // CNT is the genuine host observation. finish also checks external phase work after request
    // submission before publishing the dual-delivery BOOL8 owners.
    auto completed = wave1.finish();
    if (!probe_decoded_keys) wave1_columns = std::move(completed);
    if (!decoded_slots.empty()) {
      decoded_likes.resize(selected.size());
      for (auto slot : decoded_slots)
        decoded_likes[slot] = std::move(wave1_columns[like_answer_of_slot[slot]]);
    }
    if (cascade) (void)cascade->finish();
    result.survivor_count            = sel.survivor_count;
    result.membership_probes_applied = k_member - declined_members;
    if (measure_members && declined_members < k_member) {
      // Each probe saw the rows every earlier probe that ran kept.
      int64_t rows_in           = sequential_membership ? counters[0] : num_rows;
      result.membership_rows_in = rows_in;
      result.membership_probe_rows.resize(request.membership_filters.size());
      for (size_t m = 0; m < k_member; ++m) {
        if (!member_ran[m]) continue;
        int64_t const rows_out          = counters[sequential_membership ? m + 1 : m];
        result.membership_probe_rows[m] = {rows_in, rows_out};
        rows_in                         = rows_out;
      }
      result.membership_rows_out = sel.survivor_count;
    }
    per_filter.clear();
    keep_mask_dev = rmm::device_buffer{};  // combine consumed it; CNT synced s0

    auto decline_unselective = [&] {
      // The membership measurement stays valid: the probes ran on every row before the decode
      // declined.
      auto measured = std::move(result);
      reset_result(sc::scan_filter_status::declined_unselective);
      result.membership_probes_applied = measured.membership_probes_applied;
      result.membership_rows_in        = measured.membership_rows_in;
      result.membership_rows_out       = measured.membership_rows_out;
      result.membership_probe_rows     = std::move(measured.membership_probe_rows);
    };
    if (request.zero_survivor_only && sel.survivor_count > 0) {
      decline_unselective();
      return std::nullopt;
    }

    if (request.row_count_only) {
      // cuDF tables obtain cardinality from a column. No consumer reads this
      // carrier's bytes: every original slot is filter-only, all static filters
      // are complete, and COUNT(*) only reads num_rows(). Do not materialize
      // strings or survivor indices just to preserve that cardinality.
      std::vector<std::unique_ptr<cudf::column>> carrier;
      carrier.push_back(std::move(count_carrier));
      result.applied           = true;
      result.status            = sc::scan_filter_status::applied;
      result.keep_mask_applied = has_keep_mask;
      return carrier;
    }

    // Selectivity guard, now that the survivor count is known; two regimes:
    //  * `full` outputs present: proceed only at sel <= the full-route threshold
    //    (default 0.10) — a full decode plus gather costs about the unfiltered
    //    path, so only a near-empty survivor set pays for the compacted batch.
    //  * no `full`: the 0.35 write-skip threshold (the mask walk costs about
    //    the plain decode at sel .5) covering bitpack_mask / delta_mask AND
    //    str_split (whose char-gather savings are dictionary-like in shape but
    //    weak at ~1-char widths — deliberately NOT exempt until measurements
    //    say otherwise), with only a dict_codes output
    //    exempting the batch (it wins 2.1-2.6x at ALL selectivities — the
    //    string-materialization savings are survivor-count-independent).
    // Both regimes explicitly decline after completed wave-1 work, reporting
    // declined_unselective for the rest of this scan/filter generation.
    // A compressed LIKE source already paid for its exact answer. When every
    // delivered value has a compacted route, preserve that saving at high keep
    // rates too (especially NOT LIKE). Full-value outputs retain their guard.
    bool any_dict_gather = k_like > 0;
    bool any_full        = false;
    for (size_t i = 0; i < (k_like ? delivered : routes.size()); ++i) {
      if (is_bool8_slot[i])
        continue;  // dictionary-answered slots: a 1 B/row gather, excluded
                   // from the full-route regime
      any_dict_gather |= routes[i] == sc::decode_route::dict_codes;
      any_full |= routes[i] == sc::decode_route::full;
    }
    // Decoded LIKE has already materialized full strings. Keep both ordinary ceilings
    // for these requests, including filter-only LIKE with dictionary outputs.
    if (!decoded_slots.empty()) any_dict_gather = false;
    double const sel_frac = static_cast<double>(sel.survivor_count) / static_cast<double>(num_rows);
    if (k_like > 0 && decoded_slots.empty() && sc::onpair_like_mask_above_cap_enabled() &&
        sel_frac > sc::decompression_pushdown_max_selectivity()) {
      bool const can_substitute =
        !has_keep_mask &&
        std::all_of(
          request.like_filters.begin(), request.like_filters.end(), [&](auto const& directive) {
            return std::find(request.like_bool8_columns.begin(),
                             request.like_bool8_columns.end(),
                             directive.column) != request.like_bool8_columns.end();
          });
      if (!can_substitute) {
        // No consumer contract for projected values plus an extra mask slot. Do not replace a
        // projected STRING or mark a full-width output as row-filtered. An outstanding
        // positional mask also forbids BOOL8: gpu_ingestible applies it before returning
        // an ordinary UNFILTERED table, which drops predicate-answer metadata.
        result.status = sc::scan_filter_status::declined_unselective;
        return refuse(has_keep_mask
                        ? "outstanding keep mask forbids BOOL8 LIKE delivery"
                        : "projected or partially covered LIKE above cap requires ordinary decode");
      }
      auto const observations_before = host_observations;
      {
        // As in wave 2: prepare deferred OnPair value outputs before the first submission.
        std::vector<std::size_t> decoded;
        for (std::size_t i = 0; i < selected.size(); ++i)
          if (!submitted_like[i] && !is_bool8_slot[i]) decoded.push_back(selected[i]);
        prepare_deferred_onpair(table, decoded, streams, mr, scratch_mr);
      }
      std::vector<std::unique_ptr<cudf::column>> columns(selected.size());
      std::vector<std::size_t> positions;
      decode_session full{streams, mr, scratch_mr};
      for (std::size_t i = 0; i < selected.size(); ++i) {
        if (submitted_like[i]) {
          columns[i] = std::move(wave1_columns[like_answer_of_slot[i]]);
          result.like_bool8_columns.push_back(i);
        } else if (is_bool8_slot[i]) {
          columns[i] = std::move(wave1_columns[bool8_of_slot[i]]);
        } else {
          auto const& column = table.columns[selected[i]];
          positions.push_back(i);
          full.append(
            column_decode_request{std::cref(*column.plan_tree), value_result{column.dtype}, {}});
        }
      }
      bool const batched = sc::decompression_pushdown_batched_wave2_enabled();
      auto decoded       = batched ? full.handoff(s0) : full.finish();
      if (batched) {
        auto const completion_before = host_observations.synchronizations;
        throw_if_cuda_error(synchronize_distinct({&s0, 1}), "filtered decode: BOOL8 completion");
        result.wave2_completion_syncs = host_observations.synchronizations - completion_before;
        (void)full.finish();  // validate retained statuses and release host state after the handoff
      }
      for (std::size_t i = 0; i < positions.size(); ++i)
        columns[positions[i]] = std::move(decoded[i]);
      result.routes.assign(selected.size(), sc::decode_route::full);
      result.wave2_syncs = host_observations.synchronizations -
                           observations_before.synchronizations - result.wave2_completion_syncs;
      result.wave2_readbacks     = host_observations.readbacks - observations_before.readbacks;
      result.like_mask_delivered = true;
      result.applied             = true;
      result.status              = sc::scan_filter_status::applied;
      // The full row domain still needs the residual, dynamic probes, and any positional keep
      // mask. Their wave-1 measurements are valid, but none were applied to the delivered rows.
      return columns;
    }
    if (cost_refuses(sel_frac)) return refuse("reason=cost");
    bool const give_up =
      any_full ? sel_frac > sc::decompression_pushdown_full_route_max_selectivity()
               : (sel_frac > sc::decompression_pushdown_max_selectivity() && !any_dict_gather);
    if (give_up) {
      double const threshold = any_full ? sc::decompression_pushdown_full_route_max_selectivity()
                                        : sc::decompression_pushdown_max_selectivity();
      char const* env_name =
        any_full ? "SIRIUS_EXP_FUSED_SCAN_TIERB_MAX_SEL" : "SIRIUS_EXP_FUSED_SCAN_MAX_SEL";
      if (sc::decompression_pushdown_diag_enabled()) {
        std::fprintf(stderr,
                     "simpatico: filtered decode gave compaction up: sel=%.4f > %.4f (%s, "
                     "survivors=%lld/%lld)\n",
                     sel_frac,
                     threshold,
                     env_name,
                     static_cast<long long>(sel.survivor_count),
                     static_cast<long long>(num_rows));
      }
      decline_unselective();
      return std::nullopt;
    }

    // Per-batch enumeration pick for bitpack_mask outputs: walk the survivor
    // index list below the crossover, walk the mask bits above it. delta_mask
    // always walks the mask (the index walk rejects delta roots at render); the
    // dictionary route is unchanged.
    bool any_bitpack_mask = false;
    for (auto const t : routes)
      any_bitpack_mask |= t == sc::decode_route::bitpack_mask;
    bool const index_walk_pick =
      any_bitpack_mask && sel_frac <= sc::decompression_pushdown_index_walk_max_selectivity();
    if (sc::decompression_pushdown_diag_enabled()) {
      std::fprintf(stderr,
                   "simpatico: filtered decode row enumeration: %s (sel=%.4f, max=%.4f)\n",
                   index_walk_pick ? "index list" : "mask bits",
                   sel_frac,
                   sc::decompression_pushdown_index_walk_max_selectivity());
    }

    // Survivor indices are built on s0 and shared by the delivered columns that consume them.
    result.routes.assign(routes.begin(), routes.begin() + static_cast<std::ptrdiff_t>(delivered));
    bool const needs_survivor_indices =
      std::ranges::any_of(std::views::iota(std::size_t{0}, delivered), [&](std::size_t i) {
        if (sel.survivor_count == num_rows && decodes_like(i)) return false;
        return is_bool8_slot[i] || routes[i] == sc::decode_route::full ||
#ifdef SIMPATICO_ONPAIR
               routes[i] == sc::decode_route::onpair ||
#endif
               (index_walk_pick && routes[i] == sc::decode_route::bitpack_mask);
      });
    bool const indices_needed = needs_survivor_indices && sel.survivor_count > 0;
    cudf::column_view survivor_indices{
      cudf::data_type{cudf::type_id::INT32}, 0, nullptr, nullptr, 0};
    if (indices_needed) {
      result.row_indices = rmm::device_buffer(
        static_cast<std::size_t>(sel.survivor_count) * sizeof(std::int32_t), s0, mr);
      sc::mask_to_row_indices(sel, static_cast<std::int32_t*>(result.row_indices.data()), s0);
      survivor_indices = cudf::column_view{cudf::data_type{cudf::type_id::INT32},
                                           static_cast<cudf::size_type>(sel.survivor_count),
                                           result.row_indices.data(),
                                           nullptr,
                                           0};
    }
    // Order a consuming lane after the indices kernel with one device-side wait; streams are FIFO,
    // so it covers every later launch on that lane.
    std::vector<::cuda::stream_ref> index_lanes{s0};
    auto const wait_for_indices = [&](::cuda::stream_ref lane) {
      if (!indices_needed || std::any_of(index_lanes.begin(), index_lanes.end(), same_stream(lane)))
        return;
      cudf_pool_stream const producer{s0};
      cudf::detail::join_streams({&producer, 1}, lane);
      index_lanes.push_back(lane);
    };

    // Wave 2 decodes only the delivered prefix to survivor rows. Columns past it may have supplied
    // wave-1 filters but need no value decode or gather. Compacted routes decode compacted in the
    // kernel; `full` routes decode full width and gather in the same request, releasing the
    // full-width column on its lane once the gather is queued. A dictionary answer gathers its
    // wave-1 BOOL8 on the lane that allocated it.
    std::vector<std::unique_ptr<cudf::column>> columns(delivered);
    auto const observations_before = host_observations;
    bool const batched_wave2       = sc::decompression_pushdown_batched_wave2_enabled();
    // Retain at most budget full-width bytes, and visit every output in traversal order.
    // In particular an oversized column must run before later batches retain their compacted
    // outputs: those outputs are live allocations too (the old peak + budget bound).
    auto const budget =
      batched_wave2 ? request.staged_byte_budget.value_or(sc::decompression_pushdown_stage_bytes())
                    : 0;
    std::vector<std::size_t> positions;
    std::size_t bytes = 0;
    std::unique_ptr<decode_session> full_wave;
    // Sessions retain host uploads and kernel handles until the single public-return completion.
    // Full-width device buffers, however, release immediately after their gather is queued.
    std::vector<std::unique_ptr<decode_session>> pending_sessions;
    auto order_next_batch = [&] {
      cudf_pool_stream const gather{s0};
      for (auto lane : streams)
        if (lane.get() != s0.get()) cudf::detail::join_streams({&gather, 1}, lane);
    };
    auto flush_batch = [&] {
      if (!full_wave) return;
      auto full_columns = full_wave->handoff(s0);
      ++result.full_decode_batches;
      result.peak_staged_bytes = std::max(result.peak_staged_bytes, bytes);
      std::vector<cudf::column_view> views;
      for (auto const& full : full_columns)
        views.push_back(full->view());
      try {
        auto gathered = cudf::gather(cudf::table_view{views},
                                     survivor_indices,
                                     cudf::out_of_bounds_policy::DONT_CHECK,
                                     s0,
                                     gather_mr())
                          ->release();
        for (auto& full : full_columns)
          release_column_on(*full, s0);
        full_columns.clear();  // stream-ordered frees follow the gather, including child buffers
        // Future producers wait after those frees, so batches cannot overlap retained full inputs.
        order_next_batch();
        for (std::size_t j = 0; j < gathered.size(); ++j)
          columns[positions[j]] = std::move(gathered[j]);
      } catch (...) {
        synchronize_distinct_or_log({&s0, 1}, "filtered decode: table gather cleanup");
        throw;
      }
      pending_sessions.push_back(std::move(full_wave));
      positions.clear();
      bytes = 0;
    };
    std::unique_ptr<decode_session> wave2;
    std::vector<std::size_t> decoded_positions;
    if (!batched_wave2) wave2 = std::make_unique<decode_session>(streams, mr, scratch_mr);
    // With no survivors, a `full` route would decode the whole chunk only to gather zero rows.
    // Compacted routes already return early inside their kernels; this gives `full` routes (every
    // general-purpose codec, e.g. Snappy or ANS string bytes) the same chunk skip. Nullable plans
    // keep the existing decode path unchanged, and nested or dictionary outputs keep it because a
    // typed empty column for them needs children.
    bool const zero_skip =
      sel.survivor_count == 0 && sc::decompression_pushdown_full_route_zero_skip_enabled();
    auto const skips_decode = [&](std::size_t i) {
      if (!zero_skip || is_bool8_slot[i] || result.routes[i] != sc::decode_route::full)
        return false;
      auto const& column = table.columns[selected[i]];
      auto const dtype   = column.dtype;
      return column.plan_tree->validity.empty() &&
             (cudf::is_fixed_width(dtype) || dtype.id() == cudf::type_id::STRING);
    };
    {
      // Prepare, before the first submission, the deferred OnPair outputs this wave decodes. With
      // no survivors a compacted route returns before it prepares, so only decoded full routes
      // need it.
      std::vector<std::size_t> decoded;
      for (std::size_t i = 0; i < delivered; ++i) {
        if (is_bool8_slot[i] || decodes_like(i) || skips_decode(i) ||
            (sel.survivor_count == 0 && result.routes[i] != sc::decode_route::full))
          continue;
        decoded.push_back(selected[i]);
      }
      prepare_deferred_onpair(table, decoded, streams, mr, scratch_mr);
    }
    for (std::size_t i = 0; i < delivered; ++i) {
      if (decodes_like(i)) {
        if (sel.survivor_count == num_rows) {
          columns[i] = std::move(decoded_likes[i]);
          continue;
        }
        // Wave 1 is complete. Gather on the index producer's stream and release after its read.
        columns[i] = std::move(cudf::gather(cudf::table_view{{decoded_likes[i]->view()}},
                                            survivor_indices,
                                            cudf::out_of_bounds_policy::DONT_CHECK,
                                            s0,
                                            gather_mr())
                                 ->release()
                                 .front());
        release_column_on(*decoded_likes[i], s0);
        decoded_likes[i].reset();
        continue;
      }
      if (skips_decode(i)) {
        columns[i] = cudf::make_empty_column(table.columns[selected[i]].dtype);
        ++result.zero_skipped_full_columns;
        continue;
      }
      if (batched_wave2 && !is_bool8_slot[i] && result.routes[i] == sc::decode_route::full) {
        auto const& column = table.columns[selected[i]];
        auto const size    = staged_column_bytes(column);
        if (size && *size <= budget && budget != 0) {
          if (*size > budget - bytes) flush_batch();
          // These outputs move from their producer lanes to the gather stream. Keep them
          // upstream; order_next_batch already joins their frees back to the producer lanes.
          if (!full_wave) full_wave = std::make_unique<decode_session>(streams, mr, scratch_mr);
          full_wave->append(
            column_decode_request{std::cref(*column.plan_tree), value_result{column.dtype}, {}});
          positions.push_back(i);
          bytes += *size;
          continue;
        }
      }
      // Unknown/oversized full results and selective routes execute at their own position.
      flush_batch();
      if (!is_bool8_slot[i] && result.routes[i] == sc::decode_route::full)
        ++result.interleaved_full_columns;
      if (is_bool8_slot[i]) {
        auto const slot = static_cast<std::size_t>(bool8_of_slot[i]);
        auto const lane = bool8_lanes[slot];
        wait_for_indices(lane);
        columns[i] = std::move(cudf::gather(cudf::table_view{{wave1_columns[slot]->view()}},
                                            survivor_indices,
                                            cudf::out_of_bounds_policy::DONT_CHECK,
                                            lane,
                                            gather_mr())
                                 ->release()
                                 .front());
        if (std::none_of(index_lanes.begin(), index_lanes.end(), same_stream(lane)))
          index_lanes.push_back(lane);
        result.routes[i] = sc::decode_route::full;
        continue;
      }
      if (batched_wave2) wave2 = std::make_unique<decode_session>(streams, mr, scratch_mr);
      auto const& column = table.columns[selected[i]];
      decode_selection selection;
      selection.column_index       = selected[i];
      selection.mask               = &sel;
      selection.survivor_count     = sel.survivor_count;
      selection.survivor_indices   = survivor_indices;
      selection.route              = result.routes[i];
      selection.enumerate_by_index = index_walk_pick &&
                                     selection.route == sc::decode_route::bitpack_mask &&
                                     sel.survivor_count > 0;
      // Forward the caller's restore type only for a lossless widening of the stored type: same
      // DECIMAL scale, a wider signed integer, or TIMESTAMP_DAYS for a narrowed DATE.
      // Unsigned storage is excluded: the JIT decoder has no unsigned element types.
      if (i < request.output_types.size() && request.output_types[i].has_value()) {
        auto const stored = column.dtype;
        auto const target = *request.output_types[i];
        bool const decimal_widening =
          cudf::is_fixed_point(stored) && cudf::is_fixed_point(target) &&
          stored.scale() == target.scale() && cudf::size_of(target) > cudf::size_of(stored);
        bool const integer_widening =
          cudf::is_integral_not_bool(stored) && !cudf::is_unsigned(stored) &&
          ((cudf::is_integral_not_bool(target) && !cudf::is_unsigned(target)) ||
           target.id() == cudf::type_id::TIMESTAMP_DAYS) &&
          cudf::size_of(target) > cudf::size_of(stored);
        if (decimal_widening || integer_widening) { selection.output_type = target; }
      }
      bool consumes_indices =
        selection.route == sc::decode_route::full || selection.enumerate_by_index;
#ifdef SIMPATICO_ONPAIR
      consumes_indices = consumes_indices || selection.route == sc::decode_route::onpair;
#endif
      if (consumes_indices) wait_for_indices(wave2->next_stream());
      wave2->append(column_decode_request{std::cref(*column.plan_tree),
                                          value_result{column.dtype},
                                          validated_selection(*column.plan_tree, selection)});
      if (batched_wave2) {
        columns[i] = std::move(wave2->handoff(s0).front());
        order_next_batch();
        pending_sessions.push_back(std::move(wave2));
      } else
        decoded_positions.push_back(i);
    }
    flush_batch();
    if (!batched_wave2) {
      auto decoded = wave2->finish();
      for (std::size_t i = 0; i < decoded.size(); ++i)
        columns[decoded_positions[i]] = std::move(decoded[i]);
    }
    // handoff() has joined producer work to s0. BOOL8 gathers are external to sessions.
    // Keep the public completed-return contract with one final drain; report that boundary
    // separately from genuine metadata/status observations, rather than hiding it in syncs.
    auto const completion_before = host_observations.synchronizations;
    throw_if_cuda_error(synchronize_distinct(index_lanes), "filtered decode: gather completion");
    auto const completion_syncs = host_observations.synchronizations - completion_before;
    for (auto& session : pending_sessions)
      (void)session->finish();
    result.wave2_completion_syncs = completion_syncs;
    result.wave2_syncs =
      host_observations.synchronizations - observations_before.synchronizations - completion_syncs;
    result.wave2_readbacks = host_observations.readbacks - observations_before.readbacks;

    result.applied           = true;
    result.status            = sc::scan_filter_status::applied;
    result.keep_mask_applied = has_keep_mask;
    return columns;
  } catch (unsupported_nullable_selection const&) {
    // Nulls in selected values or predicate ballots require a clean drain before declining row
    // selection. A failed drain is a failure.
    if (auto const status = synchronize_distinct(streams); status != cudaSuccess) {
      reset_result(sc::scan_filter_status::failed);
      throw_if_cuda_error(status, "simpatico: filtered decode cleanup");
    }
    reset_result(sc::scan_filter_status::refused);
    return refuse("row selection on a null-masked column is not supported");
  } catch (...) {
    // Phase-owned masks outlive this drain, including work queued after session-owned decode
    // work. Preserve the original exception and OOM subtype.
    synchronize_distinct_or_log(streams, "simpatico: filtered decode cleanup failed");
    reset_result(sc::scan_filter_status::failed);
    throw;
  }
}

}  // namespace

// ── compressed_table ─────────────────────────────────────────────────────────

std::int64_t compressed_table::num_rows() const
{ return columns.empty() ? 0 : columns.front().num_rows; }

std::unique_ptr<cudf::table> compressed_table::decompress(::cuda::stream_ref stream,
                                                          rmm::device_async_resource_ref mr) const
{ return simpatico::decompress(*this, stream, mr); }

// ── split_plan_dsl ────────────────────────────────────────────────────────────

std::vector<std::string> split_plan_dsl(std::string_view plan_dsl)
{ return split_plan_dsl_impl(plan_dsl); }

// ── compress_with_plan ────────────────────────────────────────────────────────

namespace {
// Split the per-column plan DSL and validate it against the table + names.
// Shared preamble of all three compress_with_plan overloads.
std::vector<std::string> split_and_validate_plans(std::string_view plan_dsl,
                                                  cudf::table_view table,
                                                  std::vector<std::string> const& column_names)
{
  auto plans = split_plan_dsl_impl(plan_dsl);
  validate_plan_count(plans.size(), table.num_columns());
  validate_column_names(column_names, plans.size());
  // Sliced column views (offset != 0) are supported: every encode kernel reads
  // data<T>() (= head<T>() + offset) rather than head<T>() so the correct
  // elements are compressed regardless of the view's allocation base.
  return plans;
}
}  // namespace

compressed_table compress_with_plan(cudf::table_view table,
                                    std::string_view plan_dsl,
                                    ::cuda::stream_ref stream,
                                    rmm::device_async_resource_ref mr,
                                    std::vector<std::string> column_names)
{
  nvtx_scoped_range nvtx_range{"simpatico::compress_table[serial]"};
  auto plans = split_and_validate_plans(plan_dsl, table, column_names);

  compressed_table out;
  out.columns.reserve(plans.size());
  for (size_t i = 0; i < plans.size(); ++i) {
    std::string err;
    auto plan_tree =
      compress_column(table.column(static_cast<cudf::size_type>(i)), plans[i], stream, mr, &err);
    if (!plan_tree) throw plan_error(err.empty() ? "compress failed" : err);
    compressed_column col;
    col.dtype     = table.column(static_cast<cudf::size_type>(i)).type();
    col.num_rows  = table.num_rows();
    col.plan_tree = std::move(plan_tree);
    if (!column_names.empty()) col.name = column_names[i];
    out.columns.push_back(std::move(col));
  }
  return out;
}

compressed_table compress_with_plan(cudf::table_view table,
                                    std::string_view plan_dsl,
                                    int column_threads,
                                    rmm::device_async_resource_ref mr,
                                    std::vector<std::string> column_names)
{
  nvtx_scoped_range nvtx_range{"simpatico::compress_table[threads]"};
  auto plans = split_and_validate_plans(plan_dsl, table, column_names);
  leased_pool lp(column_threads);
  return compress_columns_parallel(table, plans, lp.pool, mr, column_names);
}

compressed_table compress_with_plan(cudf::table_view table,
                                    std::string_view plan_dsl,
                                    simpatico::stream_pool& pool,
                                    rmm::device_async_resource_ref mr,
                                    std::vector<std::string> column_names)
{
  nvtx_scoped_range nvtx_range{"simpatico::compress_table[pool]"};
  auto plans = split_and_validate_plans(plan_dsl, table, column_names);
  return compress_columns_parallel(table, plans, pool, mr, column_names);
}

// ── decompress ────────────────────────────────────────────────────────────────

std::unique_ptr<cudf::table> decompress(const compressed_table& table,
                                        ::cuda::stream_ref stream,
                                        rmm::device_async_resource_ref mr)
{
  nvtx_scoped_range nvtx_range{"simpatico::decompress_table[serial]"};
  return decode_table_columns(table, all_columns(table), {}, {&stream, 1}, mr);
}

std::unique_ptr<cudf::table> decompress(const compressed_table& table,
                                        int column_threads,
                                        rmm::device_async_resource_ref mr)
{
  nvtx_scoped_range nvtx_range{"simpatico::decompress_table[threads]"};
  leased_pool lp(column_threads);
  return decode_table_columns(table, all_columns(table), {}, lp.pool.refs(), mr);
}

std::unique_ptr<cudf::table> decompress(const compressed_table& table,
                                        simpatico::stream_pool& pool,
                                        rmm::device_async_resource_ref mr)
{
  nvtx_scoped_range nvtx_range{"simpatico::decompress_table[pool]"};
  return decode_table_columns(table, all_columns(table), {}, pool.refs(), mr);
}

std::unique_ptr<cudf::table> decompress(const compressed_table& table,
                                        std::span<const std::size_t> selected_columns,
                                        ::cuda::stream_ref stream,
                                        rmm::device_async_resource_ref mr)
{
  nvtx_scoped_range nvtx_range{"simpatico::decompress_table[selected,serial]"};
  return decode_table_columns(table, selected_columns, {}, {&stream, 1}, mr);
}

namespace {

// The plan tree of one column, or nullptr with `error_out` set.
PlanTree const* column_plan(const compressed_table& table,
                            std::size_t column_index,
                            std::string* error_out,
                            char const* what)
{
  if (column_index >= table.columns.size()) {
    if (error_out) { *error_out = std::string(what) + ": column index out of range"; }
    return nullptr;
  }
  auto const& col = table.columns[column_index];
  if (!col.plan_tree) {
    if (error_out) { *error_out = std::string(what) + ": column has no plan tree"; }
    return nullptr;
  }
  return col.plan_tree.get();
}

std::unique_ptr<cudf::column> decode_complete_column(PlanTree const& plan,
                                                     cudf::data_type stored_type,
                                                     ::cuda::stream_ref stream,
                                                     rmm::device_async_resource_ref mr,
                                                     decode_selection const* selection,
                                                     std::string* error_out)
{
  column_decode_request request{std::cref(plan), value_result{stored_type}, {}};
  if (selection) {
    try {
      request.selection.emplace(plan, *selection);
    } catch (std::invalid_argument const& error) {
      if (error_out) *error_out = error.what();
      return nullptr;
    }
  }
  return decode_one(request, stream, mr);
}

}  // namespace

std::unique_ptr<cudf::column> decompress_column_rows(const compressed_table& table,
                                                     std::size_t column_index,
                                                     sirius::codegen::chunk_row_set const& rows,
                                                     ::cuda::stream_ref stream,
                                                     rmm::device_async_resource_ref mr,
                                                     std::string* error_out)
{
  auto const* tree = column_plan(table, column_index, error_out, "decompress_column_rows");
  if (tree == nullptr) { return nullptr; }
  if (rows.num_rows < 0 || rows.num_rows > std::numeric_limits<cudf::size_type>::max() ||
      !rows.valid()) {
    if (error_out) { *error_out = "decompress_column_rows: the row set is not valid"; }
    return nullptr;
  }
  // Only the bitpack compacted decode reads a row set, and discovering that
  // costs no device work.
  if (probe_column(*tree).compact_route != sirius::codegen::decode_route::bitpack_mask) {
    if (error_out) {
      *error_out = "decompress_column_rows: this plan has no random-access decode for a row set";
    }
    return nullptr;
  }

  decode_selection sel;
  sel.rows           = &rows;
  sel.survivor_count = rows.num_survivors;
  sel.route          = sirius::codegen::decode_route::bitpack_mask;

  return decode_complete_column(
    *tree, table.columns[column_index].dtype, stream, mr, &sel, error_out);
}

std::unique_ptr<cudf::column> decompress_column_compacted(
  const compressed_table& table,
  std::size_t column_index,
  sirius::codegen::selection_mask const& mask,
  ::cuda::stream_ref stream,
  rmm::device_async_resource_ref mr,
  std::string* error_out)
{
  auto const* tree = column_plan(table, column_index, error_out, "decompress_column_compacted");
  if (tree == nullptr) { return nullptr; }
  if (mask.words == nullptr || mask.chunk_offsets == nullptr || mask.survivor_count < 0) {
    if (error_out) {
      *error_out = "decompress_column_compacted: the mask has not been counted (no chunk_offsets)";
    }
    return nullptr;
  }
  auto const route = probe_column(*tree).compact_route;
  if (route == sirius::codegen::decode_route::full) {
    if (error_out) { *error_out = "decompress_column_compacted: this plan has no compacted route"; }
    return nullptr;
  }

  decode_selection sel;
  sel.column_index   = column_index;
  sel.mask           = &mask;
  sel.survivor_count = mask.survivor_count;
  sel.route          = route;

#ifdef SIMPATICO_ONPAIR
  rmm::device_buffer indices;
  if (route == sirius::codegen::decode_route::onpair) {
    if (mask.num_rows != table.columns[column_index].num_rows ||
        mask.num_rows > std::numeric_limits<cudf::size_type>::max() ||
        mask.survivor_count > mask.num_rows) {
      if (error_out) *error_out = "onpair: inconsistent compacted mask row domain or count";
      return nullptr;
    }
    if (mask.survivor_count > 0) {
      indices = rmm::device_buffer(mask.survivor_count * sizeof(std::int32_t), stream, mr);
      sirius::codegen::mask_to_row_indices(
        mask, static_cast<std::int32_t*>(indices.data()), stream);
    }
    sel.survivor_indices = cudf::column_view{cudf::data_type{cudf::type_id::INT32},
                                             static_cast<cudf::size_type>(mask.survivor_count),
                                             indices.data(),
                                             nullptr,
                                             0};
  }
#endif

  return decode_complete_column(
    *tree, table.columns[column_index].dtype, stream, mr, &sel, error_out);
}

std::unique_ptr<cudf::column> decompress_column_full(const compressed_table& table,
                                                     std::size_t column_index,
                                                     ::cuda::stream_ref stream,
                                                     rmm::device_async_resource_ref mr,
                                                     std::string* error_out)
{
  auto const* tree = column_plan(table, column_index, error_out, "decompress_column_full");
  if (tree == nullptr) { return nullptr; }
  return decode_complete_column(
    *tree, table.columns[column_index].dtype, stream, mr, nullptr, error_out);
}

std::unique_ptr<cudf::table> decompress(const compressed_table& table,
                                        std::span<const std::size_t> selected_columns,
                                        int column_threads,
                                        rmm::device_async_resource_ref mr)
{
  nvtx_scoped_range nvtx_range{"simpatico::decompress_table[selected,threads]"};
  leased_pool lp(column_threads);
  return decode_table_columns(table, selected_columns, {}, lp.pool.refs(), mr);
}

std::unique_ptr<cudf::table> decompress(const compressed_table& table,
                                        std::span<const std::size_t> selected_columns,
                                        simpatico::stream_pool& pool,
                                        rmm::device_async_resource_ref mr)
{ return decompress(table, selected_columns, pool.refs(), mr); }

std::unique_ptr<cudf::table> decompress(const compressed_table& table,
                                        std::span<const std::size_t> selected_columns,
                                        std::span<const decode_predicate> predicates,
                                        simpatico::stream_pool& pool,
                                        rmm::device_async_resource_ref mr)
{ return decompress(table, selected_columns, predicates, pool.refs(), mr); }

std::unique_ptr<cudf::table> decompress(const compressed_table& table,
                                        std::span<const std::size_t> selected_columns,
                                        std::span<const ::cuda::stream_ref> streams,
                                        rmm::device_async_resource_ref mr)
{
  nvtx_scoped_range nvtx_range{"simpatico::decompress_table[selected,streams]"};
  return decode_table_columns(table, selected_columns, {}, streams, mr);
}

std::unique_ptr<cudf::table> decompress(const compressed_table& table,
                                        std::span<const std::size_t> selected_columns,
                                        std::span<const decode_predicate> predicates,
                                        std::span<const ::cuda::stream_ref> streams,
                                        rmm::device_async_resource_ref mr)
{
  nvtx_scoped_range nvtx_range{"simpatico::decompress_table[selected,predicated,streams]"};
  if (predicates.size() != selected_columns.size()) {
    throw plan_error("decompress: predicates and selected_columns must be the same length");
  }
  return decode_table_columns(table, selected_columns, predicates, streams, mr);
}

void prepare_deferred_onpair(compressed_table const& table,
                             std::span<const std::size_t> selected_columns,
                             std::span<const ::cuda::stream_ref> streams,
                             rmm::device_async_resource_ref mr,
                             std::optional<rmm::device_async_resource_ref> scratch_mr)
{
#ifdef SIMPATICO_ONPAIR
  if (streams.empty()) return;
  bool prepared = false;
  for (auto index : selected_columns) {
    if (index < table.columns.size() && table.columns[index].plan_tree)
      prepared |=
        prepare_deferred_onpair(*table.columns[index].plan_tree, streams.front(), mr, scratch_mr);
  }
  if (!prepared) return;
  auto const ready = streams.front().record_event();
  for (auto lane : streams.subspan(1))
    if (lane.get() != streams.front().get()) lane.wait(ready);
#else
  (void)table;
  (void)selected_columns;
  (void)streams;
  (void)mr;
  (void)scratch_mr;
#endif
}

std::unique_ptr<cudf::table> decompress_scan_filter(
  const compressed_table& table,
  std::span<const std::size_t> selected_columns,
  sirius::codegen::scan_filter_request const& request,
  sirius::codegen::scan_filter_result& result,
  std::span<const ::cuda::stream_ref> streams,
  ::cuda::stream_ref /*stream*/,
  rmm::device_async_resource_ref mr)
{
  nvtx_scoped_range nvtx_range{"simpatico::decompress_table[scan_filter,streams]"};
  std::optional<decode_scratch_resource> scratch;
  if (sirius::codegen::decode_scratch_arena_enabled())
    scratch.emplace(
      mr, streams, static_cast<std::size_t>(std::max<std::int64_t>(0, table.num_rows())));
  // Arena slack is bounded by 1 MiB per scan call. Accept that bound: retrying the entire
  // decode could replay probes/callbacks and retain preparation published by the first attempt.
  // Only a failed backing-block allocation locally falls back to an exact upstream request.
  auto const scratch_mr = scratch ? rmm::device_async_resource_ref{*scratch} : mr;
  result = sirius::codegen::scan_filter_result{};
  std::vector<std::unique_ptr<cudf::column>> decoded_likes;
  if (auto cols = try_decompress_fused(
        table, selected_columns, request, result, decoded_likes, streams, mr, scratch_mr)) {
    if (sirius::codegen::decompression_pushdown_diag_enabled()) {
      int n_a = 0, n_delta = 0, n_dict = 0, n_str_split = 0, n_onpair = 0, n_b = 0;
      for (auto const t : result.routes) {
        n_a += t == sirius::codegen::decode_route::bitpack_mask;
        n_delta += t == sirius::codegen::decode_route::delta_mask;
        n_dict += t == sirius::codegen::decode_route::dict_codes;
        n_str_split += t == sirius::codegen::decode_route::str_split;
        n_onpair += t == sirius::codegen::decode_route::onpair;
        n_b += t == sirius::codegen::decode_route::full;
      }
      std::fprintf(
        stderr,
        "simpatico: filtered decode applied: survivors=%lld/%lld "
        "routes bitpack=%d delta=%d dict=%d str_split=%d onpair=%d full=%d sources=%zu "
        "delivered=%zu/%zu like_delivery=%s zero_skipped_full=%zu\n",
        static_cast<long long>(result.survivor_count),
        static_cast<long long>(result.num_rows),
        n_a,
        n_delta,
        n_dict,
        n_str_split,
        n_onpair,
        n_b,
        request.filters.size() + request.bool8_filters.size() + request.like_filters.size(),
        result.routes.size(),
        selected_columns.size(),
        result.like_mask_delivered ? "bool8" : "compact",
        result.zero_skipped_full_columns);
    }
    // Compacted delivery is survivor-sized. Above-cap BOOL8 delivery instead retains the full
    // row domain and tells the caller which slots need residual mask evaluation.
    return std::make_unique<cudf::table>(std::move(*cols));
  }
  // Gate off / nothing requested / explicit completed policy refusal:
  // exactly the unfiltered path — with one obligation: when the request routed
  // dictionary-answered equalities (bool8_filters) into the mask, the fallback
  // must NOT be a plain decode, or every declined batch would silently lose the
  // BOOL8-substitution win (q19 -21.4%). Re-express them as ordinary
  // decode_predicates so the fallback degrades to the substitution behaviour,
  // never below it.
  std::vector<decode_predicate> predicates;
  if (!request.bool8_filters.empty()) {
    predicates.resize(selected_columns.size());
    for (auto const& b : request.bool8_filters)
      if (b.column < predicates.size()) predicates[b.column].equals_any = b.equals_any;
  }
  if (decoded_likes.empty())
    return decode_table_columns(table, selected_columns, predicates, streams, mr, scratch_mr);

  // A post-wave policy refusal returns the original row domain. Reuse its decoded LIKE values
  // here too, leaving their predicates pending for the ordinary residual.
  std::vector<std::size_t> missing;
  std::vector<decode_predicate> remaining;
  for (std::size_t i = 0; i < selected_columns.size(); ++i) {
    if (decoded_likes[i]) continue;
    missing.push_back(selected_columns[i]);
    if (!predicates.empty()) remaining.push_back(predicates[i]);
  }
  if (!missing.empty()) {
    auto rest = decode_table_columns(table, missing, remaining, streams, mr, scratch_mr)->release();
    auto next = rest.begin();
    for (auto& column : decoded_likes)
      if (!column) column = std::move(*next++);
  }
  return std::make_unique<cudf::table>(std::move(decoded_likes));
}

std::unique_ptr<cudf::table> decompress_scan_filter(
  const compressed_table& table,
  std::span<const std::size_t> selected_columns,
  sirius::codegen::scan_filter_request const& request,
  sirius::codegen::scan_filter_result& result,
  simpatico::stream_pool& pool,
  ::cuda::stream_ref stream,
  rmm::device_async_resource_ref mr,
  std::string* /*error_out*/)
{
  return decompress_scan_filter(table, selected_columns, request, result, pool.refs(), stream, mr);
}

}  // namespace simpatico
