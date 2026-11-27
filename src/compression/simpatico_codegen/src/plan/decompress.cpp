// SPDX-License-Identifier: Apache-2.0
#include "codegen/bridge/fused_tree_build.hpp"
#include "codegen/codegen_bridge.hpp"
#include "codegen/decode/masked_launch.hpp"
#include "codegen/plan/bitjoin_layout.hpp"
#include "codegen/plan/plan_interpreter.hpp"
#include "codegen/plan/validity.hpp"
#include "codegen/selection/decompression_pushdown_policy.hpp"
#include "codegen/util/cuda_check.hpp"
#include "codegen/util/nvtx.hpp"
#include "decode/decode_session.hpp"
#include "operators/constant_width_offsets.hpp"
#include "operators/nvcomp_batched_codec.hpp"
#include "operators/nvcomp_chunk_select.hpp"
#include "util/cudf_like.hpp"
#include "util/decoded_like.hpp"
#include "util/snappy_backend.hpp"
#ifdef SIMPATICO_ONPAIR
#include "operators/onpair_compressor.hpp"
#include "util/onpair_like.hpp"
#include "util/onpair_selective.hpp"
#endif

#include <cudf/aggregation.hpp>
#include <cudf/binaryop.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf/reduction.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/strings/utilities.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/transform.hpp>
#include <cudf/types.hpp>
#include <cudf/unary.hpp>
#include <cudf/utilities/traits.hpp>

#include <rmm/device_buffer.hpp>
#include <rmm/resource_ref.hpp>

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdio>
#include <limits>
#include <optional>
#include <stdexcept>
#include <unordered_map>

// The high-level JIT decode bridge is defined here alongside DecodeWalk because
// buffer binding may recursively materialize entropy tails. Its low-level
// launch_decode_fused_tree counterpart lives in codegen_runtime.cpp.

namespace simpatico {
// The compress driver (the recursive CompressWalk, compress_column) lives
// in plan/compress.cpp.
// This file owns the decode driver
// (decompress_column). The two halves share the bitjoin_layout helpers
// (including copy_column_view{,_as_uint8}) and reconstruct_representation
// (plan/representation_factory.cpp).

namespace {

void validate_plan(PlanTree const& tree);

// === codegen decode for fused subtrees ===
//
// Codegen compress writes one ``compressed_representation`` per node in a
// fused subtree, keyed by the node's plan path (``"input"`` /
// ``"delta.differences"`` / ...).  Decode recovers the whole subtree's op
// kinds + buffers and reconstructs the root column with a single codegen
// kernel launch.
//
// The per-node op kind comes from the PlanTree node's `op` field; the per-node
// device buffers come from ``compressed_representation::named_channels()``.
// Nothing per-kind lives in the walker — adding a fused op is a decode kernel,
// not new walker code.

const char* codegen_dtype_str_for(cudf::data_type type)
{
  switch (type.id()) {
    case cudf::type_id::INT8: return "int8";
    case cudf::type_id::UINT8: return "uint8";
    case cudf::type_id::INT16: return "int16";
    case cudf::type_id::UINT16: return "uint16";
    case cudf::type_id::INT32: return "int32";
    case cudf::type_id::INT64: return "int64";
    case cudf::type_id::UINT32: return "uint32";
    case cudf::type_id::UINT64: return "uint64";
    case cudf::type_id::FLOAT32: return "float32";
    case cudf::type_id::FLOAT64: return "float64";
    // The narrower decimals and the chrono types are stored as their integer
    // storage type, so they arrive here already as INT32/INT64. DECIMAL128 is
    // the exception: cudf has no 128-bit integer type_id, so its storage column
    // stays DECIMAL128 and has to be named here.
    case cudf::type_id::DECIMAL128: return "int128";
    default: return nullptr;
  }
}

// Map a DSL compressor name to its kind string if it is a codegen (fusable)
// op, or return empty if it isn't. The returned string equals the DSL op name
// so it can be used directly as the key into consumed_slots().
std::string codegen_kind_for_compressor(std::string const& c)
{
  if (c == "bitpack") return "bitpack";
  if (c == "delta") return "delta";
  if (c == "rle") return "rle";
  if (c == "for") return "for";
  if (c == "zigzag") return "zigzag";
  return {};
}

// The per-op buffer slots the decode binder reads from each rep, in order.
// Keys match the DSL op names (lowercase) returned by codegen_kind_for_compressor,
// plus "RawFused" for the synthetic raw-passthrough rep (no DSL counterpart).
// Note bitpack's 5th manifest slot (bp_offsets) is a decode-only transient
// synthesized by ``synthesize_decode_transients``, NOT bound here.  Empty
// list → unknown kind.
std::vector<std::string> consumed_slots(std::string const& kind)
{
  if (kind == "bitpack") return {"chunk_min", "chunk_count", "chunk_bits", "packed"};
  if (kind == "delta") return {"delta_first"};
  if (kind == "rle") return {"rle_runs_offsets"};
  if (kind == "for") return {"references"};
  if (kind == "zigzag") return {"zigzag"};
  if (kind == "RawFused") return {"data", "offsets"};
  return {};
}

// Interprets one request's plan and owns its structural memo, keyed by ValueId. Only values keyed
// by codegen nodes are bound into fused launches, and they stay in the memo until the walk ends. A
// non-codegen node takes ownership of the values it rebuilds from, including its stored terminal
// channels, for its rebuilt representation or selected decode; bind() refuses such values.
// run() moves the final value out.
class DecodeWalk {
 public:
  DecodeWalk(PlanTree const& tree,
             decode_frame& frame,
             decode_predicate const* pred,
             decode_selection const* sel,
             PlanTree* consumable = nullptr);
  // Materialize `consumer` and return the decoded `value` it consumes, for a fused region to bind.
  cudf::column const* bind(NodeId consumer, ValueId value);
  std::unique_ptr<cudf::column> run();
  rmm::device_async_resource_ref value_mr(ValueId value) const;
#ifdef SIMPATICO_ONPAIR
  // The caller publishes this rep at pin time or keeps it local to a selected decode. With
  // @p keep_rows, the decoded row byte offsets move there instead of being released, for a
  // request-local decode of a plan whose preparation was deferred.
  std::unique_ptr<onpair_compressed_representation> prepare_onpair_node(
    NodeId nid, std::unique_ptr<cudf::column>* keep_rows = nullptr);
  cudf::column_view onpair_row_offsets(NodeId nid);
#endif

 private:
  // Decode the values that `nid` encoded into the memo: every input of a bitjoin, otherwise its
  // single input.
  void materialize(NodeId nid);
  [[nodiscard]] cudf::column const* find_memo(std::uint64_t key) const;
  cudf::column const* store(std::uint64_t key, std::unique_ptr<cudf::column> value);
  // The memo entry of a terminal channel stored in the plan, decoded unless it is already there, so
  // a specialization that declines after inspecting it leaves it for the general route.
  std::unique_ptr<cudf::column>& terminal_entry(ValueId value,
                                                compressed_representation const& rep);
  std::unique_ptr<cudf::column> consume(ValueId value);
  std::unique_ptr<cudf::column> decode_leaf(compressed_representation const& rep, NodeId nid);
  void decode_bitjoin(NodeId nid);
  std::unique_ptr<cudf::column> materialize_fused_node(NodeId nid,
                                                       decode_selection const* node_sel);

  /// True when @p nid produces the column's final value and a predicate is
  /// pending — the one place a rep may answer the predicate instead of decoding.
  [[nodiscard]] bool predicate_applies_to(NodeId nid) const;

  /// True when @p nid is THE node whose fused decode consumes the pending
  /// selection: the (0,0)-producing bitpack region, or — for the
  /// dictionary-gather mode — the bitpack region producing the dictionary's `indices`
  /// value. Inner fused subtrees (entropy tails, dictionary keys_offsets, ...)
  /// hold metadata that is NOT row-aligned with the column and must decode
  /// full; the precomputed @c sel_target pins the exact consumer.
  [[nodiscard]] bool selection_applies_to(NodeId nid) const;

  PlanTree const& tree;
  PlanTree* consumable;
  decode_frame& frame;
  /// Borrowed; null when the caller wants the column itself.
  decode_predicate const* pred = nullptr;
  /// Borrowed decode-time row selection; null on the default path.
  decode_selection const* sel = nullptr;
  /// The one NodeId selection_applies_to accepts; tree.nodes.size() = none.
  NodeId sel_target;
  /// Set once a rep has answered `pred`, so run() knows not to compare again.
  bool predicate_resolved = false;
  // A present entry holding no column was consumed; requesting it again is an error.
  std::unordered_map<std::uint64_t, std::unique_ptr<cudf::column>> memo;
  std::unordered_map<std::uint64_t, std::size_t> remaining_consumers;
#ifdef SIMPATICO_ONPAIR
  // Only the selected root uses these offsets. Never publish them in a pinned representation.
  std::unique_ptr<cudf::column> selected_onpair_rows;
#endif
};

std::string value_label(ValueId v)
{
  return "(" + std::to_string(v.node) + "," + std::to_string(v.channel) + ")";
}

// Returns the rep for node nid, or nullptr if the node has none.
compressed_representation const* node_rep(NodeId nid, PlanTree const& tree)
{
  if (nid < tree.nodes.size()) return tree.nodes[nid].rep.get();
  return nullptr;
}

// Returns the output port name for `path` on `node`: output_names[i] where
// output_paths[i] == path. Falls back to `path` if not found.
std::string port_for_output_path(PlanNode const& node, std::string const& path)
{
  for (std::size_t i = 0; i < node.output_paths.size(); ++i) {
    if (node.output_paths[i] == path) return node.output_names[i];
  }
  return path;
}

// Per-slot element width for a LabeledBuffer. Fixed-width metadata slots carry
// their own widths; value-typed slots (chunk_min/delta_first/data/
// rle_run_values) carry the column's element width.
std::size_t elem_size_for_slot(std::string const& slot, std::size_t element_size)
{
  if (slot == "chunk_count" || slot == "rle_runs_offsets" || slot == "offsets")
    return sizeof(std::int32_t);
  if (slot == "chunk_bits") return sizeof(std::uint8_t);
  if (slot == "packed") return sizeof(std::uint32_t);
  return element_size;  // chunk_min, delta_first, data, rle_run_values
}

// name → view map of a rep's channels, for repeated per-slot lookups.
std::unordered_map<std::string, cudf::column_view> channels_by_name(
  compressed_representation const& rep, ::cuda::stream_ref stream)
{
  std::unordered_map<std::string, cudf::column_view> by_name;
  for (auto const& o : rep.named_channels(stream))
    by_name.emplace(o.name, o.view);
  return by_name;
}

// Bind the ``data``/``offsets`` slots of a synthesized Raw passthrough leaf
// at preorder *node_id*. The Raw leaf has no PlanTree op of its own; its
// bytes live in a RawFused rep parked on the parent node's ``channels``.
//
// Two cases depending on whether the channel was entropy-tail-routed at encode:
//
//   Terminal (no downstream consumer): the RawFused rep holds both ``data``
//   and ``offsets``; bind them directly.
//
//   Entropy-tail (data was routed to a downstream non-fused op, e.g. ans):
//   the RawFused rep holds only ``offsets``; ``data`` is resolved through ``materialize`` for the
//   downstream PlanTree child node (the non-fused op that compressed the raw bytes) and the
//   parent's output value. The result is a view into the shared memo, which keeps it until the walk
//   ends.
//
// `data_elem_size` is the leaf's lane width (see lane_elem_sizes).
void bind_raw_passthrough_buffers(std::int32_t node_id,
                                  NodeId parent_id,
                                  std::string const& parent_op,
                                  std::string const& parent_channel,
                                  PlanTree const& tree,
                                  std::size_t data_elem_size,
                                  ::cuda::stream_ref stream,
                                  codegen::jit::LabeledBuffers& labeled,
                                  DecodeWalk& walk)
{
  // The fused-tree builder always records the materialized channel name on the
  // raw-passthrough origin (differences / runs / values / deltas / zigzag).
  std::string const& channel_name = parent_channel;

  // Locate the RawFused rep in the parent's channels.
  compressed_representation const* rep = nullptr;
  if (parent_id < tree.nodes.size()) {
    auto const& parent_node = tree.nodes[parent_id];
    for (auto const& [path, crep] : parent_node.channels) {
      if (!crep) continue;
      std::string port = port_for_output_path(parent_node, path);
      if (port == channel_name) {
        rep = crep.get();
        break;
      }
    }
  }
  if (rep == nullptr) {
    throw std::runtime_error("codegen decode: RawFused passthrough rep missing at " + parent_op +
                             " node " + std::to_string(parent_id) + " (channel '" + channel_name +
                             "')");
  }

  auto by_name = channels_by_name(*rep, stream);

  for (auto const& slot : consumed_slots("RawFused")) {
    if (slot == "data" && by_name.find(slot) == by_name.end()) {
      // Entropy-tail: data was stripped from the rep at encode time and
      // compressed by a downstream non-fused op.  Find that child node and
      // resolve (decompress) its bytes back to the raw element array.
      if (parent_id >= tree.nodes.size()) {
        throw std::runtime_error("codegen decode: invalid parent_id for entropy-tail data resolve");
      }
      auto const& parent_node = tree.nodes[parent_id];
      NodeId child_id         = static_cast<NodeId>(tree.nodes.size());
      for (auto const& e : parent_node.children) {
        if (e.channel == channel_name) {
          child_id = e.child;
          break;
        }
      }
      if (child_id >= tree.nodes.size()) {
        throw std::runtime_error("codegen decode: no child edge '" + channel_name + "' on parent " +
                                 std::to_string(parent_id) + " for entropy-tail resolve");
      }
      auto const port = output_port(tree, parent_id, channel_name);
      if (!port) {
        throw std::runtime_error("codegen decode: node " + std::to_string(parent_id) +
                                 " does not output routed channel '" + channel_name + "'");
      }
      cudf::column_view dv = walk.bind(child_id, ValueId{parent_id, *port})->view();
      labeled[codegen::jit::buffer_key(node_id, "data")] = {
        dv.head<void>(), static_cast<std::size_t>(dv.size()), data_elem_size};
      continue;
    }
    auto bit = by_name.find(slot);
    if (bit == by_name.end()) {
      throw std::runtime_error("codegen decode: RawFused leaf missing slot '" + slot + "'");
    }
    labeled[codegen::jit::buffer_key(node_id, slot)] = {
      bit->second.head<void>(),
      static_cast<std::size_t>(bit->second.size()),
      elem_size_for_slot(slot, data_elem_size)};
  }
}

// Bind the device buffers for ONE real fused op node (bitpack / delta / rle)
// at preorder *node_id* into *labeled*. Buffers come from the node's
// rep ``named_channels()`` in per-op CONSUMED-slot order (``consumed_slots``);
// every rep is dense, so decode always uses the Compact gather.
//
// Entropy-tail-routed channels — a CONSUMED slot consumed downstream by another op (e.g. ``…packed
// -> snappy``, ``…packed -> bitcomp -> ans``, or a codegen tail ``…chunk_min -> zigzag``), detected
// as a child edge — are RESOLVED here via ``materialize`` (the downstream subtree), which returns a
// view of this node's output value in the shared memo, kept there until the walk ends. An identity
// NO-OP terminal (``…chunk_min -> identity``) leaves the bytes inside THIS rep and is bound
// directly.
void bind_real_node_buffers(std::int32_t node_id,
                            NodeId plan_node,
                            PlanTree const& tree,
                            std::size_t element_size,
                            ::cuda::stream_ref stream,
                            codegen::jit::LabeledBuffers& labeled,
                            DecodeWalk& walk)
{
  PlanNode const& node                  = tree.nodes[plan_node];
  compressed_representation const* repr = node_rep(plan_node, tree);

  std::string kind = codegen_kind_for_compressor(node.op);
  if (kind.empty()) {
    throw std::runtime_error("codegen decode: non-codegen op at node " + std::to_string(plan_node) +
                             " ('" + node.op + "')");
  }
  if (repr == nullptr) {
    throw std::runtime_error("codegen decode: missing rep at node " + std::to_string(plan_node));
  }

  auto by_name = channels_by_name(*repr, stream);

  std::unordered_map<std::string, NodeId> edge_by_channel;
  edge_by_channel.reserve(node.children.size());
  for (auto const& e : node.children)
    edge_by_channel.emplace(e.channel, e.child);

  auto const slots = consumed_slots(kind);
  if (slots.empty())
    throw std::runtime_error("codegen decode: no consumed-slot list for '" + kind + "'");
  for (auto const& slot : slots) {
    auto eit                      = edge_by_channel.find(slot);
    const bool has_edge           = eit != edge_by_channel.end();
    const bool downstream_has_rep = has_edge && node_rep(eit->second, tree) != nullptr;

    const void* ptr = nullptr;
    std::size_t len = 0;
    auto bit        = by_name.find(slot);
    if (bit != by_name.end() && !downstream_has_rep) {
      // Slot lives directly in this node's rep — bind it straight.
      ptr = bit->second.head<void>();
      len = static_cast<std::size_t>(bit->second.size());
    } else if (has_edge) {
      // Tail-routed slot (a downstream codegen region OR non-codegen rep consumes it): materialize
      // the downstream's output — a view into the shared memo, which keeps it until the walk ends.
      // One path for both, no empty-map special case (e.g. …bitpack -> chunk_min -> zigzag resolves
      // the nested codegen tail through the same memo).
      auto const port = output_port(tree, plan_node, slot);
      if (!port) {
        throw std::runtime_error("codegen decode: node " + std::to_string(plan_node) +
                                 " does not output routed slot '" + slot + "'");
      }
      auto const v = walk.bind(eit->second, ValueId{plan_node, *port})->view();
      ptr          = v.head<void>();
      len          = static_cast<std::size_t>(v.size());
    } else {
      throw std::runtime_error("codegen decode: missing buffer for slot '" + slot + "' at node " +
                               std::to_string(plan_node));
    }
    labeled[codegen::jit::buffer_key(node_id, slot)] = {
      ptr, len, elem_size_for_slot(slot, element_size)};
  }
}

// Each fused node's lane width, as both renderers type it: Rle's `runs` are int32 run
// counts, and every other child inherits its parent's width (the root's is the column's).
std::unordered_map<codegen::jit::FusedTree const*, std::size_t> lane_elem_sizes(
  codegen::jit::FusedTree const& root, std::size_t element_size)
{
  std::unordered_map<codegen::jit::FusedTree const*, std::size_t> sizes;
  auto const visit =
    [&](auto const& self, codegen::jit::FusedTree const& node, std::size_t width) -> void {
    sizes.emplace(&node, width);
    for (auto const& [name, child] : node.children) {
      bool const runs = node.op == codegen::OpKind::Rle && name == "runs";
      self(self, *child, runs ? sizeof(std::int32_t) : width);
    }
  };
  visit(visit, root, element_size);
  return sizes;
}

// Bind every node's device buffers from an already-built fused subtree in the
// builder's DFS-preorder (preorder index == rendered kernel node_id). The
// structural shape (op kinds, children, node-id order) is the builder's
// responsibility — shared with the encode bridge — so this binder only sources
// the per-node reps/buffers. Decode is Compact-only.
void bind_fused_subtree(BuiltFusedTree const& built,
                        PlanTree const& tree,
                        std::size_t element_size,
                        ::cuda::stream_ref stream,
                        codegen::jit::LabeledBuffers& labeled,
                        DecodeWalk& walk)
{
  auto const lane_sizes = lane_elem_sizes(*built.tree, element_size);
  for (std::int32_t node_id = 0; node_id < static_cast<std::int32_t>(built.preorder.size());
       ++node_id) {
    auto const& origin = built.preorder[node_id];
    // ZigZag stores nothing (it rewrites the lane value inline and recurses);
    // the decode renderer emits no params for it, so there is no buffer to
    // bind. Skip it — the child binds its own buffers.
    if (origin.node != nullptr && origin.node->op == codegen::OpKind::Zigzag) { continue; }
    if (origin.is_raw_passthrough) {
      bind_raw_passthrough_buffers(node_id,
                                   origin.parent_node,
                                   origin.parent_op,
                                   origin.parent_channel,
                                   tree,
                                   lane_sizes.at(origin.node),
                                   stream,
                                   labeled,
                                   walk);
    } else {
      bind_real_node_buffers(node_id, origin.plan_node, tree, element_size, stream, labeled, walk);
    }
  }
}

// The shared prologue of every fused-region decode launch (plain, mask-out,
// mask-consume): the built FusedTree, the region's decoded metadata, and the
// bound device buffers.
struct bound_fused_region {
  BuiltFusedTree built;
  cudf::data_type root_type{cudf::type_id::EMPTY};
  cudf::size_type num_rows = 0;
  const char* dtype        = nullptr;
  codegen::jit::LabeledBuffers labeled;
};

// Build one codegen-fused subtree, resolve its metadata, and bind its device buffers (keyed by
// DFS-preorder node_id) directly from the node-owned reps. Entropy-tail-routed channels are
// materialized into the caller's shared memo, which keeps them until the walk ends.
//
// Some intermediate fuse nodes store nothing and own no rep; their children
// own the reps. In that case, the first non-null rep in the fused preorder
// provides the decoded type and num_rows.
bound_fused_region bind_fused_region(PlanTree const& tree,
                                     NodeId root_nid,
                                     DecodeWalk& walk,
                                     ::cuda::stream_ref stream)
{
  auto built = build_fused_tree(tree, root_nid);
  if (!built) {
    throw std::runtime_error("codegen decode: no valid fusable region rooted at node " +
                             std::to_string(root_nid));
  }

  compressed_representation const* root_repr = node_rep(root_nid, tree);
  if (root_repr == nullptr) {
    // A ZigZag root stores no rep; find the first non-null rep in the
    // already-built preorder for metadata, else a Raw passthrough's rep parked
    // in its parent's channels (ZigZag over a Raw leaf owns no other).
    for (auto const& origin : built->preorder) {
      if (!origin.is_raw_passthrough && origin.plan_node < tree.nodes.size()) {
        root_repr = node_rep(origin.plan_node, tree);
        if (root_repr) break;
      }
    }
    for (auto const& origin : built->preorder) {
      if (root_repr != nullptr) break;
      if (!origin.is_raw_passthrough || origin.parent_node >= tree.nodes.size()) continue;
      auto const& parent = tree.nodes[origin.parent_node];
      for (auto const& [path, crep] : parent.channels) {
        if (crep && port_for_output_path(parent, path) == origin.parent_channel) {
          root_repr = crep.get();
          break;
        }
      }
    }
    if (root_repr == nullptr) {
      throw std::runtime_error("codegen decompress: no rep at root node " +
                               std::to_string(root_nid));
    }
  }

  bound_fused_region region;
  region.root_type = root_repr->decoded_type();
  region.num_rows  = root_repr->num_rows;
  region.dtype     = codegen_dtype_str_for(region.root_type);
  if (region.dtype == nullptr) {
    throw std::runtime_error("codegen decompress: unsupported root dtype");
  }

  const std::size_t element_size = static_cast<std::size_t>(cudf::size_of(region.root_type));
  bind_fused_subtree(*built, tree, element_size, stream, region.labeled, walk);
  region.built = std::move(*built);
  return region;
}

bool uses_index_walk(PlanTree const& tree, decode_selection const& sel);

/// The element type to store when a column decoded as @p decoded may be written directly as
/// @p target, or nullptr when it may not. Allowed: a strictly wider signed integer, a DECIMAL of
/// the same scale and greater width, or a wider DECIMAL or TIMESTAMP_DAYS from a narrower raw
/// signed integer (the decode's representation of a stored DECIMAL or narrowed DATE; the caller
/// checks the stored type's scale, see decompress_scan_filter). Each is a lossless sign-extending
/// widening, so the stored value keeps its meaning.
char const* widened_element_dtype(cudf::data_type decoded, cudf::data_type target)
{
  auto const is_signed_int = [](cudf::type_id id) {
    return id == cudf::type_id::INT8 || id == cudf::type_id::INT16 || id == cudf::type_id::INT32 ||
           id == cudf::type_id::INT64;
  };
  auto const is_decimal = [](cudf::type_id id) {
    return id == cudf::type_id::DECIMAL32 || id == cudf::type_id::DECIMAL64;
  };
  if (!cudf::is_fixed_width(decoded) || !cudf::is_fixed_width(target) ||
      cudf::size_of(target) <= cudf::size_of(decoded)) {
    return nullptr;
  }
  bool const same_family =
    (is_signed_int(decoded.id()) && is_signed_int(target.id())) ||
    (is_decimal(decoded.id()) && is_decimal(target.id()) && decoded.scale() == target.scale()) ||
    (is_signed_int(decoded.id()) &&
     (is_decimal(target.id()) || target.id() == cudf::type_id::TIMESTAMP_DAYS));
  if (!same_family) { return nullptr; }
  switch (cudf::size_of(target)) {
    case 4: return "int32_t";
    case 8: return "int64_t";
    default: return nullptr;
  }
}

// Bind one region, allocate its output, and queue the decode launch.
std::unique_ptr<cudf::column> decode_fused_subtree_impl(PlanTree const& tree,
                                                        NodeId root_nid,
                                                        DecodeWalk& walk,
                                                        decode_frame& frame,
                                                        decode_selection const* sel,
                                                        rmm::device_async_resource_ref output_mr)
{
  auto region         = bind_fused_region(tree, root_nid, walk, frame.stream());
  auto const num_rows = region.num_rows;
  bool const masked   = sel && sel->active();
  if (masked && sel->survivor_count > num_rows) {
    throw std::runtime_error("decode selection exceeds the column row count");
  }
  auto const out_rows = masked ? static_cast<cudf::size_type>(sel->survivor_count) : num_rows;
  // A bitpack-rooted compacted decode can store each value directly in the wider type the caller
  // will restore the column to, which saves a separate widening pass over the column.
  char const* const out_dtype    = masked && root_nid < tree.nodes.size() &&
                                    tree.nodes[root_nid].op == "bitpack" &&
                                    sel->output_type.has_value()
                                     ? widened_element_dtype(region.root_type, *sel->output_type)
                                     : nullptr;
  cudf::data_type const out_type = out_dtype != nullptr ? *sel->output_type : region.root_type;
  auto output                    = cudf::make_fixed_width_column(
    out_type, out_rows, cudf::mask_state::UNALLOCATED, frame.stream(), output_mr);
  if (out_rows == 0) return output;
  auto const& built   = region.built;
  auto const& labeled = region.labeled;
  auto const* dtype   = region.dtype;
  auto* const out     = output->mutable_view().head<void>();
  if (!masked) {
    launch_decode_fused_tree(*built.tree, labeled, dtype, num_rows, out, frame);
    return output;
  }
  // validated_selection checked the selection's own consistency; only the column's row count is
  // new here.
  if (sel->rows) {
    if (sel->rows->num_rows != num_rows) {
      throw std::runtime_error("decode row set does not match the selected column");
    }
    sirius::codegen::selection_mask const hollow{nullptr, num_rows, sel->survivor_count, nullptr};
    launch_decode_fused_tree_compacted(*built.tree,
                                       labeled,
                                       dtype,
                                       num_rows,
                                       hollow,
                                       row_enumeration{nullptr, sel->rows},
                                       out,
                                       frame,
                                       out_dtype);
    return output;
  }
  bool const by_index = uses_index_walk(tree, *sel);
  launch_decode_fused_tree_compacted(
    *built.tree,
    labeled,
    dtype,
    num_rows,
    *sel->mask,
    row_enumeration{by_index ? sel->survivor_indices.data<std::int32_t>() : nullptr, nullptr},
    out,
    frame,
    out_dtype);
  return output;
}

// Rep holding a bitjoin node's own (packed) output leaf: its node rep, or the
// terminal channel parked for output port 0.
compressed_representation const* bitjoin_packed_rep(PlanNode const& node)
{
  if (node.rep) return node.rep.get();
  if (!node.output_paths.empty()) {
    auto cit = node.channels.find(node.output_paths[0]);
    if (cit != node.channels.end() && cit->second) return cit->second.get();
  }
  return nullptr;
}

rmm::device_async_resource_ref DecodeWalk::value_mr(ValueId value) const
{
#ifdef SIMPATICO_ONPAIR
  if (value.node < tree.nodes.size()) {
    auto const& producer = tree.nodes[value.node];
    // Only this exact value is discarded by preparation and every scan-time reconstruction.
    // Recursive materialization (notably bitjoin siblings such as dict_offsets) can publish
    // other memo values, so it must never inherit the row-offset allocator.
    if (op_id_from_name(producer.op) == OpId::OnPair &&
        value.channel < producer.output_names.size() &&
        producer.output_names[value.channel] == "row_offsets")
      return frame.scratch_mr();
  }
#else
  (void)value;
#endif
  return frame.mr();
}

cudf::column const* DecodeWalk::find_memo(std::uint64_t key) const
{
  auto const found = memo.find(key);
  return found == memo.end() ? nullptr : found->second.get();
}

cudf::column const* DecodeWalk::store(std::uint64_t key, std::unique_ptr<cudf::column> value)
{
  if (!value) throw std::runtime_error("decode: leaf returned no column");
  auto const [entry, inserted] = memo.try_emplace(key, std::move(value));
  if (!inserted) throw std::runtime_error("decode: memo value produced twice");
  return entry->second.get();
}

// Transfers a memoised value to the inverse of its producer. Shared values are copied until their
// last consumer; a sole/last consumer takes ownership. An empty entry is deliberately retained
// after a move so any accidental re-request is a deterministic runtime error rather than a silent
// re-decode.
std::unique_ptr<cudf::column> DecodeWalk::consume(ValueId value)
{
  auto const key   = value_id_key(value);
  auto const found = memo.find(key);
  if (found == memo.end() || !found->second) {
    throw std::runtime_error("decode: unresolved or consumed memo value " + value_label(value));
  }
  auto count_it = remaining_consumers.find(key);
  if (count_it == remaining_consumers.end() || count_it->second == 0) {
    throw std::runtime_error("decode: memo value has no remaining consumer " + value_label(value));
  }
  if (--count_it->second == 0) return std::move(found->second);
  return std::make_unique<cudf::column>(found->second->view(), frame.stream(), value_mr(value));
}

std::unique_ptr<cudf::column>& DecodeWalk::terminal_entry(ValueId value,
                                                          compressed_representation const& rep)
{
  auto const key = value_id_key(value);
  if (auto const found = memo.find(key); found != memo.end()) {
    if (!found->second)
      throw std::runtime_error("decode: memo value already consumed " + value_label(value));
    return found->second;
  }
  // A terminal row-offset identity can be copied directly with its value-specific allocator.
  // Other representations keep their ordinary output resource and choose internal scratch.
  auto decoded = rep.kind() == OpId::Identity && rep.channels_.size() == 1 && rep.channels_[0] &&
                     value_mr(value) != frame.mr()
                   ? std::make_unique<cudf::column>(
                       rep.channels_.front()->view(), frame.stream(), value_mr(value))
                   : decode_standalone(rep, frame);
  if (!decoded) throw std::runtime_error("decode: leaf returned no column");
  return memo.emplace(key, std::move(decoded)).first->second;
}

cudf::column const* DecodeWalk::bind(NodeId consumer, ValueId value)
{
  if (value.node >= tree.nodes.size() || !is_codegen_compressor(tree.nodes[value.node].op)) {
    throw std::logic_error("decode: fused region binds a value it does not produce " +
                           value_label(value));
  }
  // Look the value up first: materializing a bitjoin checks only its first input, which that
  // input's producer may already have taken over.
  auto const key = value_id_key(value);
  if (!memo.contains(key)) materialize(consumer);
  if (auto const* column = find_memo(key)) return column;
  if (memo.contains(key))
    throw std::runtime_error("decode: memo value already consumed " + value_label(value));
  throw std::runtime_error("decode: node " + std::to_string(consumer) +
                           " does not consume bound value " + value_label(value));
}

// A dictionary producing the column's final value may answer the predicate from its keys.
std::unique_ptr<cudf::column> DecodeWalk::decode_leaf(compressed_representation const& rep,
                                                      NodeId nid)
{
  if (predicate_applies_to(nid)) {
#ifdef SIMPATICO_ONPAIR
    if (!pred->likes.empty()) {
      if (auto const* onpair = dynamic_cast<onpair_compressed_representation const*>(&rep)) {
        auto hits          = onpair->decompress_predicate(*pred, frame);
        predicate_resolved = true;
        return hits;
      }
      throw std::invalid_argument("onpair: LIKE requires an OnPair root");
    }
#endif
    if (auto const* dict = dynamic_cast<dictionary_compressed_representation const*>(&rep)) {
      if (auto hits = dict->decompress_predicate(*pred, frame)) {
        predicate_resolved = true;
        return hits;
      }
    }
  }
  return decode_standalone(rep, frame);
}

// Split a bitjoin node's packed leaf back into its input field values, keyed in
// `memo` by each input's structural ValueId. Fields sharing a source value are
// OR-ed into one column (a source may receive several bit ranges).
void DecodeWalk::decode_bitjoin(NodeId nid)
{
  auto const& node = tree.nodes[nid];
  auto const* rep  = bitjoin_packed_rep(node);
  if (!node.attrs.bitjoin || !rep)
    throw std::runtime_error("bitjoin decode: missing layout or packed rep");
  auto const packed      = decode_standalone(*rep, frame);
  auto const packed_view = packed->view();
  std::vector<std::optional<bit_range>> input_ranges;
  for (auto const& ref : node.attrs.bitjoin->inputs)
    input_ranges.push_back(ref.range);
  bitjoin_layout layout;
  std::string error;
  if (!resolve_bitjoin_layout(node.op, node.input_sources.size(), input_ranges, &layout, &error)) {
    throw std::runtime_error(error);
  }
  struct field_ref {
    std::uint32_t width, src_lo, dst_lo;
  };

  // Group the fields by the source value each targets (a source may collect
  // several bit ranges), keyed structurally by ValueId.
  std::unordered_map<std::uint64_t, std::vector<field_ref>> by_src;
  for (std::size_t i = 0; i < node.input_sources.size(); ++i) {
    by_src[value_id_key(node.input_sources[i])].push_back(
      {layout.widths[i], layout.src_los[i], layout.dst_los[i]});
  }
  for (auto const& [key, refs] : by_src) {
    std::uint32_t top = 0;
    for (auto const& ref : refs)
      top = std::max(top, ref.src_lo + ref.width);
    auto const type = top <= 8    ? cudf::type_id::UINT8
                      : top <= 16 ? cudf::type_id::UINT16
                      : top <= 32 ? cudf::type_id::UINT32
                                  : cudf::type_id::UINT64;
    auto output     = cudf::make_fixed_width_column(
      cudf::data_type{type},
      packed_view.size(),
      cudf::mask_state::UNALLOCATED,
      frame.stream(),
      value_mr(ValueId{static_cast<NodeId>(key >> 8), static_cast<ChannelId>(key & 0xff)}));
    throw_if_cuda_error(
      cudaMemsetAsync(output->mutable_view().head<void>(),
                      0,
                      static_cast<std::size_t>(packed_view.size()) * cudf::size_of(output->type()),
                      frame.stream().get()),
      "bitjoin decode: clear field");
    for (auto const& ref : refs) {
      launch_bitjoin_field(output->mutable_view(),
                           packed_view,
                           static_cast<int>(ref.dst_lo),
                           static_cast<int>(ref.src_lo),
                           ref.width,
                           frame.stream().get());
    }
    store(key, std::move(output));
  }
}

std::unique_ptr<cudf::column> DecodeWalk::materialize_fused_node(NodeId nid,
                                                                 decode_selection const* node_sel)
{
  auto const& sources = tree.nodes[nid].input_sources;
  auto const value    = sources.empty() ? ValueId{nid, 0} : sources.front();
  return decode_fused_subtree_impl(tree, nid, *this, frame, node_sel, value_mr(value));
}

void DecodeWalk::materialize(NodeId nid)
{
  auto const& node   = tree.nodes.at(nid);
  auto const primary = node.input_sources.empty() ? ValueId{nid, 0} : node.input_sources.front();
  auto const key     = value_id_key(primary);
  if (auto const found = memo.find(key); found != memo.end()) {
    if (!found->second)
      throw std::runtime_error("decode: memo value already consumed " + value_label(primary));
    return;
  }
  // A bitjoin decodes every input value it consumes, not only the first.
  if (node.attrs.bitjoin) {
    decode_bitjoin(nid);
    return;
  }
  if (is_codegen_compressor(node.op)) {
    store(key, materialize_fused_node(nid, selection_applies_to(nid) ? sel : nullptr));
    return;
  }
  if (node.rep) {
    if (consumable) {
      if (auto* strings =
            dynamic_cast<str_split_compressed_representation*>(consumable->nodes[nid].rep.get())) {
        store(key, strings->decompress_consuming(frame));
        return;
      }
    }
    store(key, decode_leaf(*node.rep, nid));
    return;
  }
#ifdef SIMPATICO_ONPAIR
  if (node.prepared_rep && node.prepared_rep->kind() == OpId::OnPair) {
    // row_offsets contains decoded byte offsets, rebuilt from its sub-plan or identity leaf.
    auto const port = output_port(tree, nid, "row_offsets");
    if (!port) throw std::invalid_argument("onpair: missing row_offsets output");
    ValueId const value{nid, *port};
    auto child = std::find_if(node.children.begin(), node.children.end(), [](PlanEdge const& edge) {
      return edge.channel == "row_offsets";
    });
    std::unique_ptr<cudf::column> rows;
    if (child != node.children.end()) {
      if (!memo.contains(value_id_key(value))) materialize(child->child);
      rows = consume(value);
    } else {
      auto const channel = node.channels.find(node.output_paths[*port]);
      if (channel == node.channels.end() || !channel->second)
        throw std::invalid_argument("onpair: missing row_offsets channel");
      rows = std::move(terminal_entry(value, *channel->second));
    }
    std::vector<std::unique_ptr<cudf::column>> channels;
    channels.push_back(std::move(rows));
    auto rep = reconstruct_decode_representation(node, {"row_offsets"}, std::move(channels), frame);
    store(key, decode_leaf(*rep, nid));
    return;
  }
  if (op_id_from_name(node.op) == OpId::OnPair) {
    // A plan read with deferred preparation that reached a bare request (the table facades
    // prepare such plans before submitting): prepare a request-local rep now, and feed its
    // decoded row byte offsets straight to this decode.
    std::unique_ptr<cudf::column> rows;
    auto prepared = prepare_onpair_node(nid, &rows);
    if (!prepared) throw std::invalid_argument("onpair: deferred preparation failed");
    if (!rows) {
      // Stored row_offsets are a terminal identity channel, which preparation only borrows; take
      // an owned copy the way a prepared plan's decode does.
      auto const port = output_port(tree, nid, "row_offsets");
      if (!port) throw std::invalid_argument("onpair: missing row_offsets output");
      auto const channel = node.channels.find(node.output_paths[*port]);
      if (channel == node.channels.end() || !channel->second)
        throw std::invalid_argument("onpair: missing row_offsets channel");
      rows = std::move(terminal_entry(ValueId{nid, *port}, *channel->second));
    }
    onpair_compressed_representation const rep(*prepared, std::move(rows));
    store(key, decode_leaf(rep, nid));
    return;
  }
#endif
  std::vector<std::string> names;
  std::vector<std::unique_ptr<cudf::column>> channels;
  names.reserve(node.output_names.size());
  channels.reserve(node.output_names.size());
  for (std::size_t i = 0; i < node.output_names.size(); ++i) {
    auto const& name = node.output_names[i];
    ValueId const value{nid, static_cast<ChannelId>(i)};
    auto child = std::find_if(node.children.begin(),
                              node.children.end(),
                              [&](PlanEdge const& edge) { return edge.channel == name; });
    if (child != node.children.end()) {
      if (!memo.contains(value_id_key(value))) materialize(child->child);
      names.push_back(name);
      channels.push_back(consume(value));
    } else {
      auto channel = node.channels.find(node.output_paths[i]);
      if (channel == node.channels.end()) continue;
      if (!channel->second) throw std::runtime_error("decode: missing terminal channel");
      names.push_back(name);
      channels.push_back(std::move(terminal_entry(value, *channel->second)));
    }
  }
  auto const rep = reconstruct_decode_representation(node, names, std::move(channels), frame);
  if (auto* strings = dynamic_cast<str_split_compressed_representation*>(rep.get())) {
    store(key, strings->decompress_consuming(frame));
    return;
  }
  store(key, decode_leaf(*rep, nid));
}

DecodeWalk::DecodeWalk(PlanTree const& tree,
                       decode_frame& frame,
                       decode_predicate const* pred,
                       decode_selection const* sel,
                       PlanTree* consumable)
  : tree(tree),
    consumable(consumable),
    frame(frame),
    pred(pred && pred->active() ? pred : nullptr),
    sel(sel && sel->active() ? sel : nullptr),
    sel_target(static_cast<NodeId>(tree.nodes.size()))
{
  for (auto const& node : tree.nodes)
    for (auto const& source : node.input_sources)
      ++remaining_consumers[value_id_key(source)];
  if (this->sel && this->sel->compacted()) {
    for (NodeId nid = 1; nid < tree.nodes.size(); ++nid) {
      auto const& sources = tree.nodes[nid].input_sources;
      if (sources.empty() || !(sources.front() == ValueId{0, 0})) continue;
      if (this->sel->route != sirius::codegen::decode_route::dict_codes)
        sel_target = nid;
      else {
        for (auto const& edge : tree.nodes[nid].children) {
          if (edge.channel == "indices") {
            sel_target = edge.child;
            break;
          }
        }
      }
      break;
    }
  }
}

bool DecodeWalk::predicate_applies_to(NodeId nid) const
{
  if (pred == nullptr || predicate_resolved) { return false; }
  // Only the node that produces the column's final value (node 0, port 0) may
  // answer the predicate; an inner node's output is an intermediate channel.
  auto const& sources = tree.nodes[nid].input_sources;
  return !sources.empty() && sources.front() == ValueId{0, 0};
}

bool DecodeWalk::selection_applies_to(NodeId nid) const
{
  return sel != nullptr && nid == sel_target;
}

#ifdef SIMPATICO_ONPAIR
namespace {
std::atomic<std::uint64_t> onpair_preparations{0};
}  // namespace

std::unique_ptr<onpair_compressed_representation> DecodeWalk::prepare_onpair_node(
  NodeId nid, std::unique_ptr<cudf::column>* keep_rows)
{
  auto const& node = tree.nodes.at(nid);
  if (node.rep || node.prepared_rep || op_id_from_name(node.op) != OpId::OnPair) return nullptr;
  auto const& names = op_info(OpId::OnPair).channels;
  if (node.output_names != names)
    throw std::invalid_argument("onpair: invalid output channels during preparation");
  auto const* metadata = std::get_if<leaf_meta::onpair>(&node.meta);
  if (!metadata) throw std::invalid_argument("onpair: missing leaf metadata");
  std::array<cudf::column_view, 5> views;
  std::vector<std::unique_ptr<cudf::column>> owners(5);
  for (std::size_t i = 0; i < 5; ++i) {
    if (auto const* terminal = terminal_identity_channel(node, names[i])) {
      views[i] = terminal->view();
      continue;
    }
    ValueId const value{nid, static_cast<ChannelId>(i)};
    auto child = std::find_if(node.children.begin(),
                              node.children.end(),
                              [&](PlanEdge const& edge) { return edge.channel == names[i]; });
    if (child == node.children.end())
      throw std::invalid_argument("onpair: missing channel during preparation");
    if (!memo.contains(value_id_key(value))) materialize(child->child);
    owners[i] = consume(value);
    views[i]  = owners[i]->view();
  }
  auto rep = std::make_unique<onpair_compressed_representation>(
    *metadata, std::move(views), std::move(owners), frame.stream(), frame.mr(), frame.scratch_mr());
  // consume() already accounted for this channel's consumer. Hand its owner directly to the
  // selected decode before finish_preparation releases it; no copy or second materialization.
  // Pin-time preparation has no selection and continues to release the decoded offsets.
  if (keep_rows)
    *keep_rows = std::move(rep->channels_[1]);
  else if (selection_applies_to(nid) && sel->route == sirius::codegen::decode_route::onpair)
    selected_onpair_rows = std::move(rep->channels_[1]);
  rep->finish_preparation(node);
  onpair_preparations.fetch_add(1, std::memory_order_relaxed);
  return rep;
}

cudf::column_view DecodeWalk::onpair_row_offsets(NodeId nid)
{
  if (selected_onpair_rows) return selected_onpair_rows->view();
  auto const& node = tree.nodes.at(nid);
  if (auto const* stored = terminal_identity_channel(node, "row_offsets")) return stored->view();
  auto const port  = output_port(tree, nid, "row_offsets");
  auto const child = std::find_if(node.children.begin(), node.children.end(), [](auto const& e) {
    return e.channel == "row_offsets";
  });
  if (!port || child == node.children.end())
    throw std::invalid_argument("onpair: missing selected row byte offsets");
  ValueId const value{nid, *port};
  if (!memo.contains(value_id_key(value))) materialize(child->child);
  // A pinned prepared rep has never consumed this walk's memo value. The selected decode is
  // its sole consumer (shared values still follow consume()'s accounting).
  selected_onpair_rows = consume(value);
  return selected_onpair_rows->view();
}
#endif

std::unique_ptr<cudf::column> DecodeWalk::run()
{
  auto const key = value_id_key(ValueId{0, 0});
  for (auto const& edge : tree.nodes[0].children) {
    if (find_memo(key)) break;
    materialize(edge.child);
  }
  if (!find_memo(key)) {
    for (NodeId nid = 1; nid < tree.nodes.size() && !find_memo(key); ++nid) {
      for (auto const& source : tree.nodes[nid].input_sources) {
        if (source == ValueId{0, 0}) {
          materialize(nid);
          break;
        }
      }
    }
  }
  auto const* value = find_memo(key);
  if (!value) throw std::runtime_error("decode: input column was not reconstructed");
  std::unique_ptr<cudf::column> result;
  if (pred && !predicate_resolved) {
    if (!pred->likes.empty())
      throw std::invalid_argument("onpair: LIKE was not answered by its root");
    auto const bool_type = cudf::data_type{cudf::type_id::BOOL8};
    for (auto const& text : pred->equals_any) {
      // Each needle waits for the stream while cuDF allocates its pinned validity buffer. Its
      // bytes upload from `text`, which the retained request copy owns.
      cudf::string_scalar const needle(text, true, frame.stream(), frame.mr());
      auto hit = cudf::binary_operation(
        value->view(), needle, cudf::binary_operator::EQUAL, bool_type, frame.stream(), frame.mr());
      if (!result) {
        result = std::move(hit);
      } else {
        result = cudf::binary_operation(result->view(),
                                        hit->view(),
                                        cudf::binary_operator::LOGICAL_OR,
                                        bool_type,
                                        frame.stream(),
                                        frame.mr());
      }
    }
    if (!result) throw std::runtime_error("decode: predicate has no values");
  } else {
    result = std::move(memo.at(key));
  }
  return result;
}

}  // namespace

#ifdef SIMPATICO_ONPAIR
std::uint64_t onpair_preparation_count() noexcept
{ return onpair_preparations.load(std::memory_order_relaxed); }

bool prepare_deferred_onpair(PlanTree& tree,
                             ::cuda::stream_ref stream,
                             rmm::device_async_resource_ref mr,
                             std::optional<rmm::device_async_resource_ref> scratch_mr)
{
  if (std::none_of(tree.nodes.begin(), tree.nodes.end(), [](auto const& node) {
        return !node.rep && !node.prepared_rep && op_id_from_name(node.op) == OpId::OnPair;
      }))
    return false;
  prepare_onpair_nodes(tree, stream, mr, scratch_mr);
  return true;
}

void prepare_onpair_nodes(PlanTree& tree,
                          ::cuda::stream_ref stream,
                          rmm::device_async_resource_ref mr,
                          std::optional<rmm::device_async_resource_ref> scratch_mr)
{
  // Only OnPair plans: leave unrelated codecs' preparation unchanged. Both fresh compression and
  // persisted-plan loading call this before publication, so concurrent scans never write a cache.
  bool const onpair_plan = std::any_of(tree.nodes.begin(), tree.nodes.end(), [](auto const& node) {
    return op_id_from_name(node.op) == OpId::OnPair;
  });
  if (onpair_plan && sirius::codegen::decompression_pushdown_batched_wave2_enabled()) {
    validate_plan(tree);
    struct frame_preparation : standalone_compressed_representation {
      PlanTree& tree;
      explicit frame_preparation(PlanTree& value) : tree(value) {}
      static bool needs_layout(PlanNode const& node)
      {
        auto const* meta = std::get_if<leaf_meta::ans>(&node.meta);
        return meta && !meta->frame_layout && meta->uncompressed_size != 0;
      }
      std::unique_ptr<cudf::column> decompress(decode_frame& frame) const override
      {
        for (auto& node : tree.nodes) {
          if (!needs_layout(node)) continue;
          auto& meta = std::get<leaf_meta::ans>(node.meta);
          // Only a terminal immutable frame can be cached independently of its decode walk.
          void const* payload = nullptr;
          std::size_t bytes   = 0;
          auto* rep           = dynamic_cast<ans_compressed_representation*>(node.rep.get());
          if (rep) {
            payload = rep->payload_data();
            bytes   = rep->payload_size();
          } else if (auto const* terminal = terminal_identity_channel(node, "output")) {
            payload = terminal->view().head<void>();
            bytes   = terminal->size();
          }
          if (!payload) continue;
          meta.frame_layout = std::make_shared<detail::nvcomp_frame_layout const>(
            detail::read_batched_frame_layout(payload, bytes, meta.uncompressed_size, frame));
          if (rep) rep->frame_layout = meta.frame_layout;
        }
        return cudf::make_empty_column(cudf::data_type{cudf::type_id::INT32});
      }
    } request(tree);
    if (std::any_of(tree.nodes.begin(), tree.nodes.end(), frame_preparation::needs_layout))
      (void)decode_one(column_decode_request{std::cref(request)}, stream, mr);
  }
  if (std::none_of(tree.nodes.begin(), tree.nodes.end(), [](auto const& node) {
        return !node.rep && !node.prepared_rep && op_id_from_name(node.op) == OpId::OnPair;
      }))
    return;
  // Preparation uses a standalone request, so it must validate the whole graph before the
  // decode walk can traverse an edge (including edges outside the prepared subtree).
  validate_plan(tree);
  // Reuse the decode walk and session lifetime rules to observe sub-plan channels
  // exactly once before publication. Published plans skip preparation in the scan path.
  struct preparation : standalone_compressed_representation {
    PlanTree& tree;
    explicit preparation(PlanTree& value) : tree(value) {}
    std::unique_ptr<cudf::column> decompress(decode_frame& frame) const override
    {
      DecodeWalk walk(tree, frame, nullptr, nullptr);
      for (auto it = tree.nodes.rbegin(); it != tree.nodes.rend(); ++it) {
        auto rep = walk.prepare_onpair_node(static_cast<NodeId>(&*it - tree.nodes.data()));
        if (rep) it->prepared_rep = std::move(rep);
      }
      return cudf::make_empty_column(cudf::data_type{cudf::type_id::INT32});
    }
  } request(tree);
  decode_session session{{&stream, 1}, mr, scratch_mr.value_or(mr)};
  session.append(column_decode_request{std::cref(request)});
  (void)session.finish();
}
#endif

namespace {

void validate_plan(PlanTree const& tree)
{
  if (tree.nodes.empty() || tree.nodes.front().op != "input") {
    throw std::invalid_argument("decode: tree missing input root");
  }
  for (auto const& node : tree.nodes) {
    if (node.output_names.size() != node.output_paths.size()) {
      throw std::invalid_argument("decode: output names and paths differ");
    }
    for (auto const& edge : node.children)
      if (edge.child >= tree.nodes.size())
        throw std::invalid_argument("decode: child out of range");
    for (auto const& source : node.input_sources) {
      if (source.node >= tree.nodes.size())
        throw std::invalid_argument("decode: source out of range");
      auto const channels =
        source.node == 0 ? std::size_t{1} : tree.nodes[source.node].output_names.size();
      if (source.channel >= channels)
        throw std::invalid_argument("decode: source channel out of range");
    }
  }
  std::vector<std::uint8_t> visited(tree.nodes.size());
  std::function<void(NodeId)> visit = [&](NodeId nid) {
    if (visited[nid] == 1) throw std::invalid_argument("decode: cyclic plan");
    if (visited[nid] == 2) return;
    visited[nid] = 1;
    for (auto const& edge : tree.nodes[nid].children)
      visit(edge.child);
    visited[nid] = 2;
  };
  for (NodeId nid = 0; nid < tree.nodes.size(); ++nid)
    visit(nid);
}

}  // namespace

namespace {
// Defined alongside probe_column below; used by decompress_column's route
// checks and the dictionary fast path.
NodeId root_value_producer(PlanTree const& tree);

// A compacted bitpack decode enumerates survivors from the index list rather than the mask bits.
bool uses_index_walk(PlanTree const& tree, decode_selection const& sel)
{
  auto const root = root_value_producer(tree);
  return sel.enumerate_by_index && sel.route == sirius::codegen::decode_route::bitpack_mask &&
         sel.survivor_count > 0 && root < tree.nodes.size() && tree.nodes[root].op == "bitpack" &&
         sel.survivor_indices.size() == sel.survivor_count;
}

// Chars held by a terminal nvCOMP child as one batched frame, which the selected route
// decodes chunk by chunk.
struct framed_chars {
  char const* codec;
  detail::batched_codec_ops const* ops = nullptr;
  void const* payload                  = nullptr;
  std::size_t payload_bytes            = 0;
  std::size_t chars_bytes              = 0;
  std::shared_ptr<detail::nvcomp_frame_layout const> layout;

  std::size_t block_count(std::size_t chunk) const
  {
    if (!chars_bytes) return 0;
    if (!chunk) throw std::runtime_error("selected string frame has zero block size");
    return 1 + (chars_bytes - 1) / chunk;
  }
};

struct str_split_shape {
  // Raw chars, or null when `framed` holds them.
  compressed_representation const* chars_rep = nullptr;
  std::optional<framed_chars> framed;
  NodeId offsets_nid = 0;
};
std::optional<str_split_shape> locate_str_split_shape(PlanTree const& tree);

// The specialized dictionary char-emit (launch_decode_fused_tree_dict_gather): null-free keys whose
// uniform width the plan node publishes, whose keys_chars is an identity-stored channel bound by
// view without a copy, and whose identity-stored keys_offsets count confirms the chars extent that
// width implies. An unknown or variable width, compressed key channels, and a width that
// contradicts the extent take the general route, which measures or rejects. The caller owns the
// analytic offsets (j * width) and the strings assembly; the kernel itself emits only the compacted
// chars. Returns nullptr only when this shape is unsupported. Execution failures throw; nothing
// shared is mutated before a semantic decline.
std::unique_ptr<cudf::column> try_dict_gather_fast_path(PlanTree const& tree,
                                                        decode_selection const& sel,
                                                        DecodeWalk& walk,
                                                        decode_frame& frame)
{
  auto const dict_nid = root_value_producer(tree);
  if (dict_nid >= tree.nodes.size()) return nullptr;
  auto const& node = tree.nodes[dict_nid];
  auto const width = node.dictionary_key_width_hint;
  if (width <= 0) return nullptr;
  auto const* const chars_column   = terminal_identity_channel(node, "keys_chars");
  auto const* const offsets_column = terminal_identity_channel(node, "keys_offsets");
  if (!chars_column || !offsets_column) return nullptr;
  auto const chars = chars_column->view();
  if (chars.type().id() != cudf::type_id::UINT8 || chars.null_count() != 0) return nullptr;
  // The gather addresses the chars with the width, so the extent it implies is checked against host
  // metadata first, the property reconstruct_decode_representation enforces on the general route.
  auto const keys = static_cast<std::int64_t>(offsets_column->size()) - 1;
  if (keys < 0 || static_cast<std::int64_t>(chars.size()) != keys * width) return nullptr;
  // The analytic offsets are INT32; a larger output takes the general route.
  if (sel.survivor_count * width > std::numeric_limits<cudf::size_type>::max()) return nullptr;
  auto codes_nid = static_cast<NodeId>(tree.nodes.size());
  for (auto const& edge : node.children)
    if (edge.channel == "indices") codes_nid = edge.child;
  if (codes_nid >= tree.nodes.size()) return nullptr;
  auto const region    = bind_fused_region(tree, codes_nid, walk, frame.stream());
  auto const survivors = static_cast<cudf::size_type>(sel.survivor_count);
  std::vector<std::unique_ptr<cudf::column>> children;
  auto const key_width = static_cast<std::int32_t>(width);
  // Built on the device: each cuDF scalar would wait for the stream and upload from the host.
  children.push_back(make_constant_width_offsets(survivors, key_width, frame.stream(), frame.mr()));
  rmm::device_buffer out_chars(
    static_cast<std::size_t>(survivors) * static_cast<std::size_t>(key_width),
    frame.stream(),
    frame.mr());
  auto* const char_data = out_chars.data();
  auto output           = std::make_unique<cudf::column>(
    cudf::data_type{cudf::type_id::STRING},
    survivors,
    std::move(out_chars),
    cudf::create_null_mask(0, cudf::mask_state::UNALLOCATED, frame.stream(), frame.mr()),
    0,
    std::move(children));
  if (survivors != 0) {
    launch_decode_fused_tree_dict_gather(*region.built.tree,
                                         region.labeled,
                                         region.dtype,
                                         region.num_rows,
                                         *sel.mask,
                                         row_enumeration{},
                                         chars.data<std::uint8_t>(),
                                         key_width,
                                         char_data,
                                         frame);
  }
  return output;
}

// Reconstruct the selected rows' byte ranges from the offsets subtree. For framed chars,
// remap the ranges into a buffer containing just the marked blocks before copying the strings.
std::unique_ptr<cudf::column> decode_str_split_selected(PlanTree const& tree,
                                                        decode_selection const& sel,
                                                        DecodeWalk& walk,
                                                        decode_frame& frame)
{
  auto const shape = locate_str_split_shape(tree);
  if (!shape) throw std::invalid_argument("decode: unsupported selected str_split shape");
  if (shape->framed && sel.survivor_count == 0) {
    auto const* offsets = node_rep(shape->offsets_nid, tree);
    if (!offsets || offsets->num_rows != sel.mask->num_rows + 1)
      throw std::invalid_argument("decode: selected string mask does not match the row domain");
    auto const& framed = *shape->framed;
    auto chunk         = framed.layout ? framed.layout->chunk_bytes : tree.framed_chars_chunk_bytes;
    if (!chunk && framed.chars_bytes)
      chunk = detail::read_batched_frame_chunk_size(
        framed.payload, framed.payload_bytes, framed.chars_bytes, frame.stream());
    auto const blocks = framed.block_count(chunk);
    log_chunk_selected({sel.mask->num_rows,
                        0,
                        0,
                        static_cast<std::uint32_t>(blocks),
                        0,
                        sel.column_index,
                        framed.codec});
    return cudf::make_empty_column(cudf::data_type{cudf::type_id::STRING});
  }
  void const* chars = nullptr;
  if (!shape->framed) {
    if (!tree.validity.empty())
      throw unsupported_nullable_selection(
        "decode: raw str_split selection requires all-valid rows");
    auto const channels = shape->chars_rep->named_channels(frame.stream());
    if (channels.empty() || channels.front().view.type().id() != cudf::type_id::UINT8) {
      throw std::invalid_argument("decode: selected str_split requires raw UINT8 chars");
    }
    chars = channels.front().view.head<void>();
  }
  auto const region = bind_fused_region(tree, shape->offsets_nid, walk, frame.stream());
  if (region.num_rows != sel.mask->num_rows + 1) {
    throw std::invalid_argument("decode: selected string mask does not match the row domain");
  }
  auto const survivors = static_cast<cudf::size_type>(sel.survivor_count);
  auto lengths         = cudf::make_fixed_width_column(cudf::data_type{cudf::type_id::INT32},
                                                       survivors + 1,
                                                       cudf::mask_state::UNALLOCATED,
                                                       frame.stream(),
                                                       frame.scratch_mr());
  rmm::device_buffer source_offsets(
    static_cast<std::size_t>(survivors) * sizeof(std::int64_t), frame.stream(), frame.scratch_mr());
  throw_if_cuda_error(cudaMemsetAsync(lengths->mutable_view().data<std::int32_t>() + survivors,
                                      0,
                                      sizeof(std::int32_t),
                                      frame.stream().get()),
                      "selected str_split: clear length tail");
  if (survivors > 0) {
    launch_decode_fused_tree_str_split_meta(*region.built.tree,
                                            region.labeled,
                                            region.dtype,
                                            sel.mask->num_rows,
                                            *sel.mask,
                                            row_enumeration{},
                                            static_cast<std::int64_t*>(source_offsets.data()),
                                            lengths->mutable_view().data<std::int32_t>(),
                                            frame);
  }
  cudf::size_type null_count = 0;
  auto null_mask =
    tree.validity.empty()
      ? cudf::create_null_mask(0, cudf::mask_state::UNALLOCATED, frame.stream(), frame.mr())
      : detail::select_string_validity(tree.validity,
                                       *sel.mask,
                                       lengths->mutable_view().data<std::int32_t>(),
                                       null_count,
                                       frame);
  auto offsets             = cudf::scan(lengths->view(),
                                        *cudf::make_sum_aggregation<cudf::scan_aggregation>(),
                                        cudf::scan_type::EXCLUSIVE,
                                        cudf::null_policy::EXCLUDE,
                                        frame.stream(),
                                        frame.mr());
  auto const* offset_data  = offsets->view().data<std::int32_t>();
  std::int64_t total_chars = 0;
  // Holds only the decoded chunks; released in stream order after the copy below reads it.
  rmm::device_buffer chunk_chars;
  if (shape->framed) {
    auto const& framed = *shape->framed;
    auto layout        = framed.layout;
    if (!layout && framed.chars_bytes > 0) {
      layout =
        std::make_shared<detail::nvcomp_frame_layout const>(detail::read_batched_frame_layout(
          framed.payload, framed.payload_bytes, framed.chars_bytes, frame));
    }
    auto const chunk_bytes  = layout ? layout->chunk_bytes : 0;
    auto const total_chunks = layout ? layout->compressed_bytes.size() : std::size_t{0};
    auto selected = detail::select_frame_chunks(static_cast<std::int64_t*>(source_offsets.data()),
                                                lengths->view().data<std::int32_t>(),
                                                offset_data,
                                                survivors,
                                                framed.chars_bytes,
                                                chunk_bytes,
                                                frame);
    total_chars   = selected.output_bytes;
    if (total_chars > 0 && selected.ids.empty())
      throw std::runtime_error("decode: selected string bytes without a marked chars chunk");
    std::size_t decoded_bytes = 0;
    for (auto const id : selected.ids)
      decoded_bytes += std::min(chunk_bytes, framed.chars_bytes - id * chunk_bytes);
    std::size_t decoded_blocks = 0;
    if (!selected.ids.empty()) {
      // The output resource, as for the full route's chars, keeps the decompression destination
      // eligible for the hardware decompression engine.
      chunk_chars    = rmm::device_buffer(decoded_bytes, frame.stream(), frame.mr());
      decoded_blocks = detail::batched_decompress_chunks(*framed.ops,
                                                         framed.payload,
                                                         framed.payload_bytes,
                                                         framed.chars_bytes,
                                                         *layout,
                                                         selected.ids,
                                                         chunk_chars.data(),
                                                         frame);
    }
    chars                     = chunk_chars.data();
    log_chunk_selected({sel.mask->num_rows,
                        survivors,
                        static_cast<std::uint32_t>(decoded_blocks),
                        static_cast<std::uint32_t>(total_chunks),
                        decoded_bytes,
                        sel.column_index,
                        framed.codec});
  } else {
    total_chars = frame.read_scalar(offset_data + survivors);
    if (total_chars < 0) throw std::runtime_error("decode: selected string size overflow");
  }
  bool const wide_offsets = shape->framed && total_chars >= cudf::strings::get_offset64_threshold();
  if (wide_offsets && !cudf::strings::is_large_strings_enabled())
    throw std::overflow_error("Size of output exceeds the column size limit");
  rmm::device_buffer output_chars(
    static_cast<std::size_t>(total_chars), frame.stream(), frame.mr());
  auto* destination = output_chars.data();
  if (survivors > 0 && total_chars > 0) {
    launch_masked_char_copy(chars,
                            static_cast<std::int64_t const*>(source_offsets.data()),
                            offset_data,
                            survivors,
                            destination,
                            frame);
  }
  if (wide_offsets) {
    offsets = cudf::cast(
      offsets->view(), cudf::data_type{cudf::type_id::INT64}, frame.stream(), frame.mr());
  }
  return cudf::make_strings_column(
    survivors, std::move(offsets), std::move(output_chars), null_count, std::move(null_mask));
}

#ifdef SIMPATICO_ONPAIR
onpair_compressed_representation const* selected_onpair_rep(PlanTree const& tree)
{
  auto const nid = root_value_producer(tree);
  if (nid >= tree.nodes.size() || tree.nodes[nid].op != "onpair_16") return nullptr;
  auto const& node = tree.nodes[nid];
  return dynamic_cast<onpair_compressed_representation const*>(node.rep ? node.rep.get()
                                                                        : node.prepared_rep.get());
}

bool onpair_selection_root(PlanTree const& tree)
{
  auto const nid = root_value_producer(tree);
  return nid < tree.nodes.size() && tree.nodes[nid].op == "onpair_16" &&
         (selected_onpair_rep(tree) ||
          std::holds_alternative<leaf_meta::onpair>(tree.nodes[nid].meta));
}

std::unique_ptr<cudf::column> decode_onpair_selected(PlanTree const& tree,
                                                     decode_selection const& sel,
                                                     DecodeWalk& walk,
                                                     decode_frame& frame)
{
  auto const nid       = root_value_producer(tree);
  auto const& node     = tree.nodes.at(nid);
  auto const* rep      = selected_onpair_rep(tree);
  auto const* metadata = std::get_if<leaf_meta::onpair>(&node.meta);
  if (!rep && !metadata) throw std::invalid_argument("onpair: missing leaf metadata");
  auto const num_rows = rep ? static_cast<std::uint64_t>(rep->num_rows) : metadata->num_rows;
  if (static_cast<std::uint64_t>(sel.mask->num_rows) != num_rows)
    throw std::invalid_argument("onpair: selection mask does not match the row domain");
  auto const survivors = static_cast<cudf::size_type>(sel.survivor_count);
  if (survivors == 0) {
    // Inspect only host metadata: even temporary preparation submits device work.
    // Diagnostic denominator only: raw prefixes give batches+1 entries; raw codes give the
    // token count; a compressed pair leaves it at 0.
    auto const* prefixes  = terminal_identity_channel(node, "batch_prefixes");
    auto const* codes     = terminal_identity_channel(node, "codes");
    std::uint32_t batches = 0;
    if (rep) {
      batches = rep->batch_count();
    } else if (prefixes && prefixes->size() > 0) {
      batches = static_cast<std::uint32_t>(prefixes->size() - 1);
    } else if (codes) {
      batches = metadata->batch_count(codes->size());
    }
    log_onpair_selected({sel.mask->num_rows,
                         0,
                         0,
                         batches,
                         0,
                         rep ? rep->tokens_per_batch() : metadata->tokens_per_batch});
    return cudf::make_empty_column(cudf::data_type{cudf::type_id::STRING});
  }
  // Unpublished/unprepared plans use a request-local rep. Already pinned plans skip preparation.
  std::unique_ptr<onpair_compressed_representation> local_rep;
  if (!rep) {
    local_rep = walk.prepare_onpair_node(nid);
    rep       = local_rep.get();
  }
  if (!rep) throw std::invalid_argument("onpair: selected root is not prepared");
  auto rows = rep->row_byte_offsets();
  if (rows.type().id() == cudf::type_id::EMPTY)
    rows = walk.onpair_row_offsets(root_value_producer(tree));
  auto batches     = rep->select_batches(rows, sel.survivor_indices, frame);
  auto const bytes = batches.output_bytes;
  // cuDF's owning strings factory accepts INT64 offsets. Keep the wide scan for large
  // selections; the existing INT32 char-copy ABI (also used by str_split) is unchanged.
  auto offsets = bytes > std::numeric_limits<std::int32_t>::max()
                   ? std::move(batches.output_offsets)
                   : cudf::cast(batches.output_offsets->view(),
                                cudf::data_type{cudf::type_id::INT32},
                                frame.stream(),
                                frame.mr());
  rmm::device_buffer scratch(batches.scratch_bytes, frame.stream(), frame.scratch_mr());
  rep->decode_batches(batches, scratch.data(), frame);
  rmm::device_buffer chars(static_cast<std::size_t>(bytes), frame.stream(), frame.mr());
  if (bytes > 0) {
    if (offsets->type().id() == cudf::type_id::INT64) {
      copy_onpair_selected_chars64(scratch.data(),
                                   static_cast<std::int64_t const*>(batches.source_offsets.data()),
                                   offsets->view().data<std::int64_t>(),
                                   survivors,
                                   chars.data(),
                                   frame);
    } else {
      launch_masked_char_copy(scratch.data(),
                              static_cast<std::int64_t const*>(batches.source_offsets.data()),
                              offsets->view().data<std::int32_t>(),
                              survivors,
                              chars.data(),
                              frame);
    }
  }
  log_onpair_selected({rep->num_rows,
                       survivors,
                       batches.count,
                       rep->batch_count(),
                       batches.scratch_bytes,
                       rep->tokens_per_batch()});
  return cudf::make_strings_column(
    survivors,
    std::move(offsets),
    std::move(chars),
    0,
    cudf::create_null_mask(0, cudf::mask_state::UNALLOCATED, frame.stream(), frame.mr()));
}
#endif

}  // namespace

validated_selection::validated_selection(PlanTree const& plan, decode_selection const& selection)
  : selection_(selection)
{
  validate_plan(plan);
  namespace sc = sirius::codegen;
  if (!selection.active() || (selection.mask != nullptr) == (selection.rows != nullptr)) {
    throw std::invalid_argument("decode: selection requires exactly one active row source");
  }
  if (selection.route != sc::decode_route::full &&
      selection.route != probe_column(plan).compact_route) {
    throw std::invalid_argument("decode: requested route does not match the plan");
  }
#ifdef SIMPATICO_ONPAIR
  if (selection.route == sc::decode_route::onpair &&
      (!selection.mask || selection.rows || selection.enumerate_by_index)) {
    throw std::invalid_argument(
      "onpair: selection requires a mask and survivor indices; chunk_row_set and "
      "enumerate_by_index refused, never silently decoded full width");
  }
#endif
  if (selection.rows) {
    if (selection.route != sc::decode_route::bitpack_mask || selection.rows->num_rows < 0 ||
        selection.rows->num_rows > std::numeric_limits<cudf::size_type>::max() ||
        !selection.rows->valid() || selection.rows->num_survivors != selection.survivor_count) {
      throw std::invalid_argument("decode: invalid compacted chunk row set");
    }
  } else {
    if (selection.mask->num_rows < 0 ||
        selection.mask->num_rows > std::numeric_limits<cudf::size_type>::max() ||
        (selection.mask->num_rows > 0 && !selection.mask->words) ||
        selection.mask->survivor_count != selection.survivor_count ||
        selection.survivor_count > selection.mask->num_rows ||
        (selection.compacted() && !selection.mask->chunk_offsets)) {
      throw std::invalid_argument("decode: inconsistent selection count or offsets");
    }
    bool uses_indices = !selection.compacted() || uses_index_walk(plan, selection);
#ifdef SIMPATICO_ONPAIR
    uses_indices = uses_indices || selection.route == sc::decode_route::onpair;
#endif
    if (uses_indices &&
        (selection.survivor_indices.type().id() != cudf::type_id::INT32 ||
         selection.survivor_indices.size() != selection.survivor_count ||
         selection.survivor_indices.null_count() != 0 ||
         (selection.survivor_count > 0 && !selection.survivor_indices.data<std::int32_t>()))) {
      throw std::invalid_argument("decode: survivor indices violate type/shape/null policy");
    }
  }
}

namespace {

PlanTree const* request_plan(column_decode_request const& request)
{
  if (auto const* tree = std::get_if<std::reference_wrapper<PlanTree const>>(&request.source))
    return &tree->get();
  if (auto const* tree = std::get_if<std::reference_wrapper<PlanTree>>(&request.source))
    return &tree->get();
  return nullptr;
}

// The host checks of a column request, before any device work.
void validate_request(column_decode_request const& request)
{
  auto const* predicate = std::get_if<predicate_result>(&request.result);
  auto const* tree      = request_plan(request);
  if (auto const* like = std::get_if<decoded_like_result>(&request.result)) {
    if (!tree || request.selection || like->filters.empty())
      throw std::invalid_argument("decoded LIKE requires a full value decode");
    for (auto const& [pattern, destination, observation] : like->filters)
      if (!pattern.valid(/*require_ascii=*/false) || !destination.words ||
          destination.num_rows <= 0)
        throw std::invalid_argument("invalid decoded LIKE mask request");
  }
  if (!tree) {
    if (predicate || request.selection)
      throw std::invalid_argument("standalone request cannot substitute or select");
    return;
  }
  validate_plan(*tree);
  // Only the framed str_split route gathers the validity sidecar in survivor order.
  if (request.selection && !tree->validity.empty() &&
      request.selection->get().route != sirius::codegen::decode_route::str_split)
    throw unsupported_nullable_selection("decode: selection on a null-masked column unsupported");
  if (predicate && !predicate->predicate.valid())
    throw std::invalid_argument("decode: predicate requires exactly one valid form");
  if (predicate && !predicate->predicate.likes.empty() &&
      (!probe_column(*tree).can_answer_like || request.selection))
    throw std::invalid_argument("onpair: LIKE requires a non-null OnPair root, no selection");
  if (predicate && request.selection &&
      request.selection->get().route != sirius::codegen::decode_route::dict_codes &&
      request.selection->get().route != sirius::codegen::decode_route::full) {
    throw std::invalid_argument("decode: predicate selection requires dict_codes or full route");
  }
}

// The host checks of a mask request, before any device work.
void validate_request(mask_decode_request const& request)
{
  validate_plan(request.plan);
  auto const destination = request.destination;
  if (destination.num_rows < 0 || (destination.num_rows > 0 && !destination.words)) {
    throw std::invalid_argument("decode: invalid mask destination");
  }
  if (std::holds_alternative<sirius::codegen::range_predicate>(request.source)) {
    if (!probe_column(request.plan).can_produce_mask()) {
      throw std::invalid_argument("decode: plan cannot produce a range ballot");
    }
  } else if (!std::get<membership_source>(request.source).probe) {
    throw std::invalid_argument("decode: empty membership probe");
  }
}

void match_decoded_strings(cudf::column_view const& strings,
                           decoded_like_result const& like,
                           PlanTree const& tree,
                           decode_frame& frame)
{
  auto const stream = frame.stream();
  auto const mr     = frame.scratch_mr();
  for (auto const& [pattern, dest, observation] : like.filters) {
    if (strings.size() != dest.num_rows) throw std::invalid_argument("decoded LIKE row mismatch");
    std::span<std::uint32_t> words{
      dest.words,
      static_cast<std::size_t>(sirius::codegen::selection_mask::AllocWordsFor(dest.num_rows))};
    auto const matcher = decoded_like_mask(
      cudf::strings_column_view(strings), pattern, like.try_swar, words, stream, mr);
    if (observation) {
      auto const nid   = root_value_producer(tree);
      auto const& root = tree.nodes.at(nid < tree.nodes.size() ? nid : 0);
      auto codec       = root.op;
      if (codec == "str_split") {
        auto const chars = std::ranges::find(root.children, "chars", &PlanEdge::channel);
        codec += "/" + (chars == root.children.end() ? std::string{"raw"}
                                                     : tree.nodes.at(chars->child).op);
      }
      observation->codec = std::move(codec);
      observation->matcher = matcher;
    }
  }
}

}  // namespace

std::unique_ptr<cudf::column> decode_request(column_decode_request const& request,
                                             decode_frame& frame)
{
  validate_request(request);
  auto const* predicate = std::get_if<predicate_result>(&request.result);
  auto const* selection = request.selection ? &request.selection->get() : nullptr;
  auto const* pred      = predicate ? &predicate->predicate : nullptr;
  if (auto const* standalone =
        std::get_if<std::reference_wrapper<standalone_compressed_representation const>>(
          &request.source)) {
    return standalone->get().decompress(frame);
  }
  auto const& tree  = *request_plan(request);
  auto const* owned = std::get_if<std::reference_wrapper<PlanTree>>(&request.source);
  namespace sc      = sirius::codegen;
  DecodeWalk walk{tree, frame, pred, selection, owned ? &owned->get() : nullptr};
  std::unique_ptr<cudf::column> output;
  if (selection && selection->route == sc::decode_route::str_split) {
    output = decode_str_split_selected(tree, *selection, walk, frame);
#ifdef SIMPATICO_ONPAIR
  } else if (selection && selection->route == sc::decode_route::onpair) {
    output = decode_onpair_selected(tree, *selection, walk, frame);
#endif
  } else {
    if (selection && selection->route == sc::decode_route::dict_codes && !predicate) {
      output = try_dict_gather_fast_path(tree, *selection, walk, frame);
    }
    if (!output && selection && !selection->compacted()) {
      auto const full = walk.run();
      if (full->size() != selection->mask->num_rows)
        throw std::invalid_argument("decode: full selection mask does not match the row domain");
      auto gathered = cudf::gather(cudf::table_view{{full->view()}},
                                   selection->survivor_indices,
                                   cudf::out_of_bounds_policy::DONT_CHECK,
                                   frame.stream(),
                                   frame.gather_mr());
      output        = std::move(gathered->release().front());
    } else if (!output) {
      output = walk.run();
    }
  }
  if (!selection) {
    attach_validity(*output, tree.validity, frame.stream(), frame.mr());
  } else if (output->size() != selection->survivor_count ||
             (tree.validity.empty() && output->null_count() != 0)) {
    throw std::runtime_error("decode: selected output has wrong size or unexpected nulls");
  }
  if (auto const* like = std::get_if<decoded_like_result>(&request.result))
    match_decoded_strings(output->view(), *like, tree, frame);
  if (predicate) {
    if (output->type().id() != cudf::type_id::BOOL8)
      throw std::runtime_error("decode: predicate is not BOOL8");
    if (predicate->ballot) {
      auto const& destination = *predicate->ballot;
      if (output->size() != destination.num_rows ||
          (destination.num_rows > 0 && !destination.words)) {
        throw std::invalid_argument("decode: predicate ballot shape mismatch");
      }
      if (output->null_count() != 0)
        throw unsupported_nullable_selection("decode: nullable predicate ballot unsupported");
      sc::mask_from_bool8(output->view().data<std::uint8_t>(),
                          destination.num_rows,
                          destination.words,
                          frame.stream());
    }
  }
  return output;
}

std::unique_ptr<cudf::column> decompress_column(PlanTree const& tree,
                                                ::cuda::stream_ref stream,
                                                rmm::device_async_resource_ref mr,
                                                std::string* error_out,
                                                decode_predicate const* pred,
                                                decode_selection const* sel)
{
  nvtx_scoped_range range{"simpatico::decompress_column"};
  column_decode_request request{std::cref(tree)};
  // Translate only documented host validation failures; submitted execution failures propagate.
  try {
    if (pred && pred->active()) request.result = predicate_result{*pred, std::nullopt};
    if (sel && sel->active()) request.selection.emplace(tree, *sel);
    validate_request(request);
  } catch (std::invalid_argument const& e) {
    if (error_out) *error_out = e.what();
    return nullptr;
  }
  auto column = decode_one(request, stream, mr);
  if (error_out) error_out->clear();
  return column;
}

namespace {

bool dictionary_value_root(PlanTree const& tree)
{
  if (tree.nodes.empty() || tree.nodes[0].op != "input") { return false; }
  // The producer of the column's final value is whichever node consumes (0,0).
  // Only `dictionary` can answer a predicate off its compressed form.
  for (NodeId nid = 1; nid < tree.nodes.size(); ++nid) {
    auto const& sources = tree.nodes[nid].input_sources;
    if (!sources.empty() && sources.front() == ValueId{0, 0}) {
      return tree.nodes[nid].op == "dictionary";
    }
  }
  return false;
}

// The node producing the column's final value: whichever consumes (0,0).
NodeId root_value_producer(PlanTree const& tree)
{
  if (tree.nodes.empty() || tree.nodes[0].op != "input") {
    return static_cast<NodeId>(tree.nodes.size());
  }
  for (NodeId nid = 1; nid < tree.nodes.size(); ++nid) {
    auto const& sources = tree.nodes[nid].input_sources;
    if (!sources.empty() && sources.front() == ValueId{0, 0}) { return nid; }
  }
  return static_cast<NodeId>(tree.nodes.size());
}

// The column's final value is produced by a bitpack region, so the masked
// ballot / mask-walk render variants apply to the root region directly.
bool bitpack_selection_root(PlanTree const& tree)
{
  NodeId const nid = root_value_producer(tree);
  return nid < tree.nodes.size() && tree.nodes[nid].op == "bitpack";
}

// A delta root whose `differences` child is bitpack. The mask_consume
// launcher renders this shape too — the per-chunk prefix-sum reconstruction
// still runs, only the stores are masked/compacted.
bool delta_selection_root(PlanTree const& tree)
{
  NodeId const nid = root_value_producer(tree);
  if (nid >= tree.nodes.size() || tree.nodes[nid].op != "delta") { return false; }
  for (auto const& e : tree.nodes[nid].children) {
    if (e.channel == "differences") {
      return e.child < tree.nodes.size() && tree.nodes[e.child].op == "bitpack";
    }
  }
  return false;  // raw-passthrough differences: not a rendered mask_consume shape
}

// A chars child the selected route can decode by chunk: a batched nvCOMP node consuming nothing
// but the chars, with nothing below it, whose UINT8 payload is its own representation or a terminal
// identity `output`.
std::optional<framed_chars> locate_framed_chars(PlanTree const& tree, NodeId nid)
{
  if (nid >= tree.nodes.size()) return std::nullopt;
  auto const& node = tree.nodes[nid];
  auto const op    = op_id_from_name(node.op);
  if (!op || !node.children.empty() || node.input_sources.size() != 1) return std::nullopt;
  detail::batched_codec_ops const& (*get_ops)();
  switch (*op) {
    case OpId::Snappy: get_ops = detail::snappy_batched_ops; break;
    case OpId::Ans: get_ops = detail::ans_batched_ops; break;
    case OpId::Lz4: get_ops = detail::lz4_batched_ops; break;
    case OpId::Deflate: get_ops = detail::deflate_batched_ops; break;
    default: return std::nullopt;
  }
  framed_chars framed;
  std::int32_t type_id = 0;
  std::uint8_t format  = detail::kNvcompPayloadFormat;
  if (node.rep) {
    auto const* rep = dynamic_cast<nvcomp_payload_rep const*>(node.rep.get());
    if (!node.output_names.empty() || !rep || rep->kind() != *op) return std::nullopt;
    if (auto const* ans = dynamic_cast<ans_compressed_representation const*>(rep))
      format = ans->payload_format;
    type_id              = static_cast<std::int32_t>(rep->original_type.id());
    framed.payload       = rep->payload_data();
    framed.payload_bytes = rep->payload_size();
    framed.chars_bytes   = rep->uncompressed_size;
    framed.layout        = rep->frame_layout;
  } else {
    if (node.output_names.size() != 1 || node.output_names.front() != "output") return std::nullopt;
    auto const* stored = terminal_identity_channel(node, "output");
    auto const* meta   = batched_meta(*op, node.meta);
    if (!stored || !meta || stored->type().id() != cudf::type_id::UINT8) return std::nullopt;
    if (*op == OpId::Ans) format = std::get<leaf_meta::ans>(node.meta).payload_format;
    type_id              = meta->original_type_id;
    framed.payload       = stored->view().head<void>();
    framed.payload_bytes = static_cast<std::size_t>(stored->size());
    framed.chars_bytes   = meta->uncompressed_size;
    framed.layout        = meta->frame_layout;
  }
  if (type_id != static_cast<std::int32_t>(cudf::type_id::UINT8) ||
      framed.chars_bytes > static_cast<std::size_t>(std::numeric_limits<cudf::size_type>::max()) ||
      (framed.chars_bytes > 0 && !framed.payload) ||
      (*op == OpId::Ans && !detail::nvcomp_payload_compatibility_error("ans", format).empty())) {
    return std::nullopt;
  }
  framed.codec = op_info(*op).name.data();
  framed.ops   = &get_ops();
  return framed;
}

std::optional<str_split_shape> locate_str_split_shape(PlanTree const& tree)
{
  NodeId const nid = root_value_producer(tree);
  if (nid >= tree.nodes.size() || tree.nodes[nid].op != "str_split") { return std::nullopt; }
  PlanNode const& node = tree.nodes[nid];
  str_split_shape shape;
  shape.offsets_nid = static_cast<NodeId>(tree.nodes.size());
  bool chars_ok     = false;
  bool offsets_ok   = false;
  for (std::size_t i = 0; i < node.output_names.size(); ++i) {
    std::string const& name = node.output_names[i];
    if (name == "chars") {
      auto const edge = std::find_if(node.children.begin(),
                                     node.children.end(),
                                     [&](PlanEdge const& e) { return e.channel == name; });
      if (edge != node.children.end()) {
        if (!chunk_selective_enabled()) return std::nullopt;
        shape.framed = locate_framed_chars(tree, edge->child);
        if (!shape.framed) { return std::nullopt; }
        chars_ok = true;
        continue;
      }
      auto it = node.channels.find(node.output_paths[i]);
      if (it != node.channels.end() && it->second && it->second->kind() == OpId::Identity &&
          it->second->decoded_type().id() == cudf::type_id::UINT8) {
        chars_ok        = true;
        shape.chars_rep = it->second.get();
      }
    } else if (name == "offsets") {
      for (auto const& e : node.children) {
        if (e.channel == name) {
          offsets_ok = e.child < tree.nodes.size() &&
                       (tree.nodes[e.child].op == "bitpack" || tree.nodes[e.child].op == "delta");
          shape.offsets_nid = e.child;
          break;
        }
      }
    }
  }
  if (!chars_ok || !offsets_ok) { return std::nullopt; }
  return shape;
}

bool dict_codes_selection_root(PlanTree const& tree)
{
  if (tree.nodes.empty() || tree.nodes[0].op != "input") { return false; }
  for (NodeId nid = 1; nid < tree.nodes.size(); ++nid) {
    auto const& sources = tree.nodes[nid].input_sources;
    if (sources.empty() || !(sources.front() == ValueId{0, 0})) { continue; }
    PlanNode const& node = tree.nodes[nid];
    if (node.op != "dictionary") { return false; }
    // The mask consumer is the codes region: the `indices` channel must be
    // routed to a bitpack child so it can decode compacted.
    for (auto const& e : node.children) {
      if (e.channel == "indices") {
        return e.child < tree.nodes.size() && tree.nodes[e.child].op == "bitpack";
      }
    }
    return false;  // indices stored inline (identity) — no fused region to mask
  }
  return false;
}

// A full decoded-LIKE source can use framed chars even with raw offsets or selective blocks off.
std::optional<framed_chars> frame_for_admission(PlanTree const& tree)
{
  if (!decoded_like_enabled()) {
    auto const shape = locate_str_split_shape(tree);
    return shape ? shape->framed : std::nullopt;
  }
  auto const nid = root_value_producer(tree);
  if (nid >= tree.nodes.size() || tree.nodes[nid].op != "str_split") return std::nullopt;
  for (auto const& edge : tree.nodes[nid].children)
    if (edge.channel == "chars") return locate_framed_chars(tree, edge.child);
  return std::nullopt;
}

}  // namespace

void prepare_selected_string_geometry(PlanTree& tree, ::cuda::stream_ref stream)
{
  if (!chunk_selective_enabled() && !decoded_like_enabled()) return;
  auto const framed = frame_for_admission(tree);
  if (!framed) return;
  auto const& frame = *framed;
  tree.framed_chars_chunk_bytes =
    frame.layout ? frame.layout->chunk_bytes
                 : detail::read_batched_frame_chunk_size(
                     frame.payload, frame.payload_bytes, frame.chars_bytes, stream);
}

std::size_t selected_string_scratch_bytes(PlanTree const& tree, std::size_t rows)
{
  auto const framed = frame_for_admission(tree);
  if (!framed || rows == 0) return 0;
  auto const& frame = *framed;
  auto const chunk  = frame.layout ? frame.layout->chunk_bytes : tree.framed_chars_chunk_bytes;
  auto const blocks = frame.block_count(chunk);
  // A workspace query for admission does not submit any blocks.
  snappy_backend_log_scope const quiet{nullptr};
  std::size_t workspace = 0;
  if (blocks && frame.ops->decompress_get_temp_size(
                  blocks, chunk, &workspace, frame.chars_bytes, cudf::get_default_stream().get()) !=
                  nvcompSuccess)
    throw std::runtime_error("cannot size selected string decompression workspace");

  // The scan already reserves the final chars and INT32 offsets. Allow all rows to survive.
  // Lengths and source starts coexist with decoded blocks. Validity gathering, the prefix
  // scan and a possible INT64 offsets cast also need space beyond that final column.
  auto const lengths_and_wide_offsets = (rows + 1) * (sizeof(std::int32_t) + sizeof(std::int64_t));
  auto const starts_and_validity_indices = rows * (sizeof(std::int64_t) + sizeof(std::int32_t));
  auto const masks = cudf::bitmask_allocation_size_bytes(static_cast<cudf::size_type>(rows));
  auto const block_tables =
    blocks * (5 * sizeof(std::size_t) + sizeof(nvcompStatus_t) + 2 * sizeof(std::uint32_t));
  return frame.chars_bytes + lengths_and_wide_offsets + starts_and_validity_indices + masks +
         block_tables + workspace;
}

column_decode_caps probe_column(PlanTree const& tree)
{
  namespace sc = sirius::codegen;
  column_decode_caps caps;
  bool const nullable = !tree.validity.empty();
  auto const strings  = locate_str_split_shape(tree);
  if (strings && (!nullable || strings->framed)) {
    caps.compact_route = sc::decode_route::str_split;
  } else if (nullable) {
    // Other routes cannot gather the validity sidecar.
  } else if (bitpack_selection_root(tree)) {
    caps.compact_route = sc::decode_route::bitpack_mask;
    // A root without a representation is refused later, when the ballot binds it.
    if (auto const* rep = node_rep(root_value_producer(tree), tree)) {
      caps.range_comparable_lane = cudf::size_of(rep->decoded_type()) <= sizeof(std::int64_t);
    }
  } else if (delta_selection_root(tree)) {
    caps.compact_route = sc::decode_route::delta_mask;
  } else if (dict_codes_selection_root(tree)) {
    caps.compact_route = sc::decode_route::dict_codes;
#ifdef SIMPATICO_ONPAIR
  } else if (onpair_selection_root(tree) && onpair_selective_enabled()) {
    caps.compact_route = sc::decode_route::onpair;
#endif
  }
  caps.can_answer_equality = dictionary_value_root(tree);
#ifdef SIMPATICO_ONPAIR
  caps.can_answer_like = !nullable && onpair_like_enabled() && onpair_selection_root(tree);
#endif
  return caps;
}

mask_source_status decode_request(mask_decode_request const& request, decode_frame& frame)
{
  validate_request(request);
  auto const destination = request.destination;
  DecodeWalk walk{request.plan, frame, nullptr, nullptr};
  if (auto const* range = std::get_if<sirius::codegen::range_predicate>(&request.source)) {
    auto const region =
      bind_fused_region(request.plan, root_value_producer(request.plan), walk, frame.stream());
    if (destination.num_rows != region.num_rows)
      throw std::invalid_argument("decode: ballot row count mismatch");
    sirius::codegen::selection_mask mask{destination.words, destination.num_rows, -1, nullptr};
    if (destination.num_rows > 0) {
      launch_decode_fused_tree_mask_out(
        *region.built.tree, region.labeled, region.dtype, region.num_rows, *range, mask, frame);
    }
    return mask_source_status::ACCEPTED;
  }
  auto const& source = std::get<membership_source>(request.source);
  auto const keys    = source.decoded_key ? nullptr : walk.run();
  auto const* key    = source.decoded_key ? source.decoded_key : keys.get();
  if (key->size() != destination.num_rows) {
    throw std::invalid_argument("decode: membership key row count mismatch");
  }
  if (keys && source.selected_string_key && !request.plan.validity.empty() &&
      chunk_selective_enabled())
    attach_validity(*keys, request.plan.validity, frame.stream(), frame.scratch_mr());
  auto key_view = key->view();
  if (key_view.type() != source.stored_type &&
      cudf::is_bit_castable(key_view.type(), source.stored_type)) {
    key_view = cudf::column_view{source.stored_type,
                                 key_view.size(),
                                 key_view.head<void>(),
                                 key_view.null_mask(),
                                 key_view.null_count(),
                                 key_view.offset()};
  }
  auto const flags =
    source.probe(key_view, source.prior_mask_words, frame.stream(), frame.scratch_mr());
  if (!flags) {
    auto const bytes = static_cast<std::size_t>(
                         sirius::codegen::selection_mask::AllocWordsFor(destination.num_rows)) *
                       sizeof(std::uint32_t);
    throw_if_cuda_error(cudaMemsetAsync(destination.words, 0xff, bytes, frame.stream().get()),
                        "membership decline: fill mask");
    return mask_source_status::DECLINED;
  }
  if (flags->type().id() != cudf::type_id::BOOL8 || flags->size() != destination.num_rows ||
      flags->null_count() != 0) {
    throw std::invalid_argument("decode: membership flags violate shape/type/null policy");
  }
  sirius::codegen::mask_from_bool8(
    flags->view().data<std::uint8_t>(), destination.num_rows, destination.words, frame.stream());
  return mask_source_status::ACCEPTED;
}

bool decompress_column_selection_mask(PlanTree const& tree,
                                      sirius::codegen::range_predicate pred,
                                      std::uint32_t* mask_words,
                                      ::cuda::stream_ref stream,
                                      rmm::device_async_resource_ref mr,
                                      std::string* error_out)
{
  nvtx_scoped_range nvtx_range{"simpatico::decompress_column_selection_mask"};

  std::optional<mask_decode_request> request;
  try {
    validate_plan(tree);
    if (!mask_words || !probe_column(tree).can_produce_mask()) {
      throw std::invalid_argument("decode: plan or destination cannot serve range ballot");
    }
    auto const* rep = node_rep(root_value_producer(tree), tree);
    if (!rep) throw std::invalid_argument("decode: range ballot root has no representation");
    request.emplace(tree, pred, mask_destination{mask_words, rep->num_rows});
    validate_request(*request);
  } catch (std::invalid_argument const& e) {
    if (error_out) *error_out = e.what();
    return false;
  }
  (void)decode_one(*request, stream, mr);
  if (error_out) error_out->clear();
  return true;
}

}  // namespace simpatico
