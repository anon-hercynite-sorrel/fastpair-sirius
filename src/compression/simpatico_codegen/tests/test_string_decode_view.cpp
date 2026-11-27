// SPDX-License-Identifier: Apache-2.0
#include "api/compressed_table_io.hpp"
#include "api/simpatico_codegen.hpp"
#include "codegen/selection/decompression_pushdown_policy.hpp"
#include "compression/compressed_representation.hpp"
#include "compression/compression_converters.hpp"
#include "compression/device_compressed_blob.hpp"
#include "data/data_batch_utils.hpp"
#include "data/sirius_converter_registry.hpp"
#include "decode/decode_scratch_resource.hpp"
#include "decode/decode_session.hpp"
#include "expression_evaluator/like_multiliteral.hpp"
#include "frame_budget_test_utils.hpp"
#include "late_mat/defer_directive.hpp"
#include "memory/sirius_memory_reservation_manager.hpp"
#include "op/scan/decoded_batch_representation.hpp"
#include "op/scan/detail/read_retirement.hpp"
#include "op/scan/duckdb_native_gpu_ingestible.hpp"
#include "op/scan/sirius_gpu_scan_operator.hpp"
#include "op/scan/sirius_gpu_scan_operator_data.hpp"
#include "op/scan/sirius_physical_dynamic_filter.hpp"
#include "pipeline/gpu_pipeline_executor.hpp"
#include "pipeline/gpu_pipeline_task.hpp"
#include "pipeline/repository_wiring.hpp"
#include "planner/late_mat_plan_pass.hpp"
#include "scan_manager/load_balancing_scan_batch_coalescer.hpp"
#include "scan_manager/memory_prefetcher.hpp"
#include "scan_manager/sirius_scan_manager.hpp"
#include "scan_manager/split_connector.hpp"
#include "test_utils.hpp"
#include "util/decoded_like.hpp"
#include "util/string_decode.hpp"
#include "utils/table_filter_test_utils.hpp"
#include "utils/telemetry_utils.hpp"
#include "utils/test_validation_utility.hpp"

#include <cudf/binaryop.hpp>
#include <cudf/replace.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/search.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/strings/contains.hpp>
#include <cudf/strings/find.hpp>
#include <cudf/unary.hpp>

#include <rmm/cuda_stream.hpp>

#include <cucascade/data/representation_converter.hpp>
#include <cucascade/memory/memory_space.hpp>
#include <cucascade/memory/reservation_aware_resource_adaptor.hpp>
#include <cucascade/memory/reservation_manager_configurator.hpp>
#include <duckdb/catalog/catalog.hpp>
#include <duckdb/catalog/catalog_entry/duck_table_entry.hpp>
#include <duckdb/catalog/catalog_entry/schema_catalog_entry.hpp>
#include <duckdb/main/connection.hpp>
#include <duckdb/main/database.hpp>
#include <duckdb/transaction/transaction_context.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <format>
#include <future>
#include <limits>
#include <memory>
#include <numeric>
#include <string>
#include <thread>
#include <vector>

// Linker observation covers both waves and the ordinary decode after a policy refusal.
namespace {
thread_local simpatico::PlanTree const* watched_plan = nullptr;
thread_local std::size_t watched_decodes             = 0;
thread_local cudaStream_t watched_decode_lane        = nullptr;
}  // namespace
extern "C" std::unique_ptr<cudf::column>
real_decode_column(simpatico::column_decode_request const&, simpatico::decode_frame&) asm(
  "__real__ZN9simpatico14decode_requestERKNS_21column_decode_requestERNS_12decode_frameE");
extern "C" std::unique_ptr<cudf::column>
count_decode_column(simpatico::column_decode_request const&, simpatico::decode_frame&) asm(
  "__wrap__ZN9simpatico14decode_requestERKNS_21column_decode_requestERNS_12decode_frameE");
std::unique_ptr<cudf::column> count_decode_column(simpatico::column_decode_request const& request,
                                                  simpatico::decode_frame& frame)
{
  if (watched_plan) {
    auto const* plan =
      std::get_if<std::reference_wrapper<simpatico::PlanTree const>>(&request.source);
    if (plan && &plan->get() == watched_plan) {
      ++watched_decodes;
      watched_decode_lane = frame.stream().get();
    }
  }
  return real_decode_column(request, frame);
}

namespace {

using gpu_rep = cucascade::gpu_table_representation;

// Reproduce device pin storage: the cached table owns slab slices; only the blob owns the payload.
std::shared_ptr<sirius::compressed_device_blob> pin(cudf::table_view input,
                                                    ::cuda::stream_ref stream,
                                                    rmm::device_async_resource_ref mr,
                                                    std::string plans = {})
{
  if (plans.empty()) {
    for (int i = 0; i < input.num_columns(); ++i) {
      if (i) plans += "---\n";
      plans += "input -> identity\n";
    }
  }
  auto compressed = simpatico::compress_with_plan(input, plans, stream, mr);
  std::vector<std::uint8_t> header;
  std::vector<simpatico::payload_buffer_ref> buffers;
  std::uint64_t wire_bytes = 0;
  auto error =
    simpatico::build_compressed_table_header(compressed, header, buffers, wire_bytes, stream);
  expect(error.empty(), error.c_str());
  auto blob = std::make_shared<sirius::compressed_device_blob>();
  for (auto const& b : buffers)
    blob->leaves.add_buffer(std::max(b.size_bytes, b.alloc_bytes));
  blob->leaves.allocate(stream, mr, simpatico::compressed_leaf_slab::alignment);
  cuda_check(cudaMemsetAsync(blob->leaves.data(), 0, blob->leaves.size(), stream.get()));
  std::size_t slot = 0;
  for (auto const& b : buffers) {
    if (std::max(b.size_bytes, b.alloc_bytes) == 0) continue;
    auto* dst = blob->leaves.slice(slot++);
    if (b.size_bytes > 0)
      cuda_check(
        cudaMemcpyAsync(dst, b.device_ptr, b.size_bytes, cudaMemcpyDeviceToDevice, stream.get()));
  }
  blob->table = simpatico::read_compressed_table_from_memory(
    header,
    [](std::uint64_t, std::size_t, void*, ::cuda::stream_ref) {},
    stream,
    mr,
    &error,
    blob->leaves.resource(),
    simpatico::collect_nvcomp_pin_metadata(compressed).get());
  expect(error.empty(), error.c_str());
  stream.sync();
  return blob;
}

simpatico::str_split_compressed_representation const& stored_strings(
  sirius::compressed_device_blob const& blob)
{
  auto const& tree    = *blob.table.columns[0].plan_tree;
  auto const* strings = dynamic_cast<simpatico::str_split_compressed_representation const*>(
    tree.nodes.at(tree.nodes[0].children.at(0).child).rep.get());
  expect(strings != nullptr, "identity STRING must store str_split");
  return *strings;
}

std::unique_ptr<sirius::compressed_device_representation> source(
  std::shared_ptr<sirius::compressed_device_blob> blob,
  cucascade::memory::memory_space& space,
  std::optional<std::size_t> uncompressed = std::nullopt)
{
  auto const bytes = blob->leaves.size();
  auto const rows  = blob->table.num_rows();
  std::vector<std::string> names(blob->table.num_columns(), "value");
  auto rep = std::make_unique<sirius::compressed_device_representation>(
    space, std::move(blob), std::move(names), bytes, uncompressed.value_or(bytes), rows);
  // Direct converter tests stand in for an unfiltered, non-deferring engine scan.
  rep->set_string_view_eligible(true);
  return rep;
}

void roundtrip(cudf::table_view input,
               bool copy,
               cucascade::memory::memory_space& space,
               cucascade::representation_converter_registry& registry,
               ::cuda::stream_ref stream)
{
  auto blob = pin(input, stream, rmm::mr::get_current_device_resource_ref());
  std::weak_ptr<sirius::compressed_device_blob> weak = blob;
  auto const& stored                                 = stored_strings(*blob);
  auto const* chars                                  = stored.channels_[1]->view().head<void>();
  auto const* offsets                                = stored.channels_[0]->view().head<void>();
  auto rep                                           = source(blob, space);
  auto output = registry.convert<gpu_rep>(*rep, &space, stream);
  expect(
    (dynamic_cast<sirius::borrowed_string_batch_representation*>(output.get()) != nullptr) == !copy,
    "engine borrowed-batch marker (including empty strings)");
  expect(output->get_writer_event() != nullptr, "borrowed batch must have known writer");
  auto view = output->get_table_view();
  if (chars) expect((view.column(0).head<void>() == chars) == !copy, "pin chars route pointer");
  if (input.num_rows() > 0)
    expect((view.column(0).child(0).head<void>() == offsets) == !copy, "pin offsets route pointer");
  // Standalone callers receive owned columns even when the same pin table is available.
  auto standalone = simpatico::decompress(blob->table, stream, space.get_default_allocator());
  if (chars) expect(standalone->view().column(0).head<void>() != chars, "standalone must copy");
  standalone.reset();
  rep.reset();
  blob.reset();
  expect(weak.expired() == copy, "batch must retain the pin payload after projection/unpin");
  expect(view.num_columns() == input.num_columns(), "mixed borrowed/owned output width");
  for (int i = 0; i < input.num_columns(); ++i)
    expect(columns_equal_any(input.column(i), view.column(i), stream),
           "engine identity values/nulls");
  // cuCascade can materialize a surviving view when an owning consumer explicitly requests it.
  auto owned = output->release_table(stream);
  expect(weak.expired(), "explicit owned release must drop the borrowed owner after copying");
  for (int i = 0; i < input.num_columns(); ++i)
    expect(columns_equal_any(input.column(i), owned->view().column(i), stream),
           "owned release values");
}

void guards(cucascade::memory::memory_space& space,
            cucascade::representation_converter_registry& registry,
            ::cuda::stream_ref stream)
{
  auto input        = make_strings_table({"alpha", "beta", "", "alpha"}, {}, stream);
  auto blob         = pin(input->view(), stream, rmm::mr::get_current_device_resource_ref());
  auto const* chars = stored_strings(*blob).channels_[1]->view().head<void>();
  auto rep          = source(blob, space);
  sirius::pushdown_request request;
  request.columns.resize(1);
  request.columns[0].equals_any = {"alpha"};
  rep->set_pushdown_scan(std::make_shared<const sirius::decompression_pushdown_scan>(request));
  auto output = registry.convert<gpu_rep>(*rep, &space, stream);
  expect(dynamic_cast<sirius::borrowed_string_batch_representation*>(output.get()) == nullptr,
         "predicate must disable pin view");
  auto expected = cudf::binary_operation(input->view().column(0),
                                         cudf::string_scalar{"alpha", true, stream},
                                         cudf::binary_operator::EQUAL,
                                         cudf::data_type{cudf::type_id::BOOL8},
                                         stream);
  expect(columns_equal_any(expected->view(), output->get_table_view().column(0), stream),
         "predicate returns the correct row answers");
  rep->set_pushdown_scan(nullptr);
  auto words = std::shared_ptr<std::uint32_t[]>(new std::uint32_t[1]{0b0101});
  rep->set_visibility_mask({words, 4});
  output = registry.convert<gpu_rep>(*rep, &space, stream);
  expect(dynamic_cast<sirius::borrowed_string_batch_representation*>(output.get()) == nullptr,
         "MVCC mask must disable pin view");
  expect(output->get_table_view().column(0).head<void>() != chars, "masked decode must copy");
  // Unsupported mask-only pushdown stays full width for the scan's positional visibility filter.
  expect(columns_equal_any(input->view().column(0), output->get_table_view().column(0), stream),
         "mask-only fallback preserves positional rows");
}

// Exercise native projection and the actual scan output without reading table storage.
struct native_scan_fixture {
  std::filesystem::path directory;
  std::unique_ptr<duckdb::DuckDB> db;
  std::unique_ptr<duckdb::Connection> con;
  std::shared_ptr<sirius::op::scan::duckdb_native_gpu_ingestible> ingestible;
  std::unique_ptr<sirius::op::scan::sirius_gpu_scan_operator> scan;

  std::shared_ptr<sirius::op::sirius_dynamic_filter_set> channel;
  std::shared_ptr<sirius::op::scan::dynamic_filter_consumer> consumer;

  native_scan_fixture(cudf::table_view input,
                      std::size_t output_width,
                      bool residual_filter     = false,
                      bool membership_endpoint = false)
  {
    char pattern[] = "/tmp/sirius-string-view-XXXXXX";
    auto* root     = mkdtemp(pattern);
    expect(root != nullptr, "create native scan fixture directory");
    directory       = root;
    db              = std::make_unique<duckdb::DuckDB>((directory / "scan.duckdb").c_str());
    con             = std::make_unique<duckdb::Connection>(*db);
    auto info       = std::make_unique<sirius::op::scan::duckdb_native_ingestible_table_info>();
    std::string ddl = "CREATE TABLE t(";
    for (int i = 0; i < input.num_columns(); ++i) {
      bool const string = input.column(i).type().id() == cudf::type_id::STRING;
      auto type =
        sirius::logical_type::make(string ? sirius::type_id::VARCHAR : sirius::type_id::INTEGER);
      if (i) ddl += ", ";
      ddl += "c" + std::to_string(i) + (string ? " VARCHAR" : " INTEGER");
      info->returned_types.push_back(type);
      info->projected_types.push_back(type);
      if (static_cast<std::size_t>(i) < output_width) info->output_types.push_back(type);
      info->column_ids.emplace_back(i);
      sirius::op::scan::projected_column col;
      col.storage_idx = duckdb::StorageIndex(i);
      info->projected_cols.push_back(col);
      info->names.push_back("c" + std::to_string(i));
    }
    auto result = con->Query(ddl + ")");
    expect(!result->HasError(), "create native scan fixture table");
    result = con->Query("BEGIN TRANSACTION");
    expect(!result->HasError(), "begin native scan fixture transaction");
    auto& ctx     = *con->context;
    auto& catalog = duckdb::Catalog::GetCatalog(ctx, "");
    duckdb::CatalogTransaction txn(catalog, ctx);
    auto& schema = catalog.GetSchema(txn, "main");
    auto entry   = schema.GetEntry(txn, duckdb::CatalogType::TABLE_ENTRY, duckdb::Identifier("t"));
    expect(bool(entry), "native scan fixture catalog entry");
    info->storage                = &entry->Cast<duckdb::DuckTableEntry>().GetStorage();
    info->context                = &ctx;
    info->projected_output_count = output_width;
    if (residual_filter) {
      info->table_filters = duckdb::make_uniq<duckdb::TableFilterSet>();
      info->table_filters->PushFilter(
        duckdb::ProjectionIndex(0),
        sirius::test::constant_filter(duckdb::ExpressionType::COMPARE_EQUAL,
                                      duckdb::Value("alpha")));
    }
    auto types = info->output_types;
    ingestible = sirius::op::scan::make_ingestible(std::move(info));
    if (membership_endpoint) {
      channel  = std::make_shared<sirius::op::sirius_dynamic_filter_set>();
      consumer = sirius::op::scan::make_dynamic_filter_consumer(
        channel,
        sirius::op::scan::dynamic_filter_apply_mode::MEMBERSHIP_MASKS_ONLY,
        sirius::op::scan::dynamic_filter_gate::k_default_keep_threshold,
        sirius::op::scan::scan_decode_bindings(*ingestible, types.size()),
        nullptr);
    }
    scan = std::make_unique<sirius::op::scan::sirius_gpu_scan_operator>(
      std::move(types), input.num_rows(), ingestible, 1, nullptr, nullptr, consumer);
  }

  ~native_scan_fixture()
  {
    scan.reset();
    ingestible.reset();
    con.reset();
    db.reset();
    std::filesystem::remove_all(directory);
  }
};

void admit(sirius::op::scan::scan_operator_input& split)
{
  sirius::op::scan::pin_validation validation;
  validation.identity = validation.layout = validation.structure = validation.iteration =
    validation.visibility                                        = {true, true};
  split.resident_contract_id                                     = 1;
  split.resident_validation                                      = validation;
}

void expect_same_buffers(cudf::column_view before, cudf::column_view after)
{
  expect(before.head<void>() == after.head<void>(), "post-projection column data pointer");
  expect(before.null_mask() == after.null_mask(), "post-projection column validity pointer");
  expect(before.num_children() == after.num_children(), "post-projection children");
  for (int i = 0; i < before.num_children(); ++i)
    expect_same_buffers(before.child(i), after.child(i));
}

std::shared_ptr<sirius::scan_manager::pinned_entry> provider_entry(
  std::shared_ptr<sirius::compressed_device_blob> blob, cucascade::memory::memory_space& space)
{
  auto entry      = std::make_shared<sirius::scan_manager::pinned_entry>();
  entry->tier     = cucascade::memory::Tier::GPU;
  entry->num_rows = blob->table.num_rows();
  for (int i = 0; i < blob->table.num_columns(); ++i)
    entry->cache_info.names.push_back("c" + std::to_string(i));
  entry->device_chunks.push_back({source(std::move(blob), space), {}, &space});
  entry->chunk_memory_spaces.push_back(&space);
  entry->late_mat_handle = std::make_shared<sirius::late_mat::pin_entry_handle>("string-view", 1);
  entry->late_mat_handle->set_entry(entry);
  return entry;
}

sirius::scan_manager::databatch_provider::batch serve(
  std::shared_ptr<sirius::scan_manager::pinned_entry> const& entry,
  sirius::op::scan::sirius_gpu_scan_operator const& scan,
  sirius::pushdown_request request = {})
{
  auto const selected = scan.get_ingestible().materialized_column_order();
  auto provider       = sirius::scan_manager::make_provider_for_pinned_entry(entry,
                                                                       selected,
                                                                             {{0}, 0},
                                                                             {},
                                                                             {},
                                                                             {},
                                                                             {},
                                                                       false,
                                                                       std::move(request),
                                                                       scan.decode_consumer(),
                                                                       scan.string_view_eligible());
  return provider->get_next_batch();
}

void scan_handoff(cudf::table_view input,
                  std::size_t output_width,
                  bool copy,
                  cucascade::memory::memory_space& space,
                  ::cuda::stream_ref stream,
                  std::string plans = {})
{
  native_scan_fixture fixture{input, output_width};
  std::unique_ptr<sirius::op::operator_data> output;
  std::weak_ptr<sirius::compressed_device_blob> weak;
  void const* chars;
  void const* offsets;
  cudf::table_view prepared;
  {
    auto blob   = pin(input, stream, rmm::mr::get_current_device_resource_ref(), std::move(plans));
    weak        = blob;
    chars       = stored_strings(*blob).channels_[1]->view().head<void>();
    offsets     = stored_strings(*blob).channels_[0]->view().head<void>();
    auto entry  = provider_entry(blob, space);
    auto served = serve(entry, *fixture.scan);
    auto batch  = served.data;
    sirius::op::scan::scan_operator_input split{batch};
    admit(split);
    sirius::converter_registry::initialize();
    split.prepare_for_processing(&space, stream);
    if (copy) {
      expect(split.stolen_table != nullptr, "copy switch keeps owned-table transfer");
      prepared = split.stolen_table->view();
    } else {
      expect(split.stolen_table == nullptr, "borrowed strings stay view-backed");
      auto read = batch->to_read_only();
      prepared  = sirius::get_cudf_table_view(read);
    }
    output = fixture.scan->execute(split, stream);
  }
  // Assert on the GPU representation in the batch execute() actually hands the pipeline,
  // after native projection, normalization and source split/pin teardown.
  auto* pipeline = dynamic_cast<sirius::op::pipelineable_operator_data*>(output.get());
  expect(pipeline && pipeline->get_data_batches().size() == 1, "scan emits one pipeline batch");
  auto emitted = pipeline->get_data_batches()[0]->to_read_only();
  auto view    = sirius::get_cudf_table_view(emitted);
  expect(view.num_columns() == static_cast<int>(output_width), "native projected output width");
  if (chars) expect((view.column(0).head<void>() == chars) == !copy, "emitted pin chars pointer");
  if (input.num_rows() > 0)
    expect((view.column(0).child(0).head<void>() == offsets) == !copy,
           "emitted pin offsets pointer");
  expect(weak.expired() == copy, "emitted view retains pin after source split teardown");
  for (int i = 0; i < view.num_columns(); ++i) {
    expect_same_buffers(prepared.column(i), view.column(i));
    expect(columns_equal_any(input.column(i), view.column(i), stream),
           "post-projection values/nulls");
  }
}

void scan_eligibility_guards(cucascade::memory::memory_space& space,
                             cucascade::representation_converter_registry& registry,
                             ::cuda::stream_ref stream,
                             bool copy)
{
  // A rejected wide row makes a full-width chars allocation much larger than the predicate,
  // compacted survivors and scan bookkeeping together, so the allocator peak detects a copy.
  auto columns =
    make_strings_table({"alpha", std::string(1U << 20, 'x'), "", "alpha"}, {}, stream)->release();
  auto numeric = make_int32_table(1, 4, 7)->release();
  columns.push_back(std::move(numeric[0]));
  cudf::table input{std::move(columns)};
  auto blob              = pin(input.view(), stream, rmm::mr::get_current_device_resource_ref());
  auto entry             = provider_entry(blob, space);
  auto const* chars      = stored_strings(*blob).channels_[1]->view().head<void>();
  auto const* offsets    = stored_strings(*blob).channels_[0]->view().head<void>();
  auto const chars_bytes = stored_strings(*blob).channels_[1]->alloc_size();
  native_scan_fixture plain{input.view(), 2};
  expect(plain.scan->string_view_eligible(), "unfiltered scan grants view eligibility");

  // Isolate the original-request guard: the scan itself permits views, but for_chunk removes
  // equality on raw str_split and returns no request. That must not grant view eligibility.
  sirius::pushdown_request equality;
  equality.columns.resize(2);
  equality.columns[0].equals_any = {"alpha"};
  auto narrowed                  = serve(entry, *plain.scan, equality);
  {
    auto read       = narrowed.data->to_read_only();
    auto const& rep = read.get_data()->cast<sirius::compressed_device_representation>();
    expect(!rep.pushdown_scan(), "raw equality is removed by for_chunk");
    expect(!rep.string_view_eligible(), "removed equality still vetoes view decode");
    auto clone = rep.clone(stream);
    expect(!clone->cast<sirius::compressed_device_representation>().string_view_eligible(),
           "clone retains the scan eligibility veto");
    auto decoded = registry.convert<gpu_rep>(rep, &space, stream);
    expect(dynamic_cast<sirius::borrowed_string_batch_representation*>(decoded.get()) == nullptr,
           "narrowed equality takes owned decode");
    expect(decoded->get_table_view().column(0).head<void>() != chars,
           "narrowed equality does not borrow pin chars");
    expect(columns_equal_any(input.view().column(0), decoded->get_table_view().column(0), stream),
           "narrowed equality leaves values for the residual filter");
  }

  sirius::pushdown_request companion_range;
  companion_range.columns.resize(2);
  companion_range.columns[1].range = sirius::decode_range{7, 58};
  auto ranged                      = serve(entry, *plain.scan, companion_range);
  {
    auto read       = ranged.data->to_read_only();
    auto const& rep = read.get_data()->cast<sirius::compressed_device_representation>();
    expect(!rep.string_view_eligible(), "range on a companion column vetoes all string views");
    auto decoded = registry.convert<gpu_rep>(rep, &space, stream);
    expect(dynamic_cast<sirius::borrowed_string_batch_representation*>(decoded.get()) == nullptr,
           "companion range takes owned decode");
  }

  // Static residuals consume a pin view directly and gather only matching rows. Cover both the
  // all-column filter and the folded projection that drops a trailing pure-filter column.
  for (std::size_t output_width : {std::size_t{1}, std::size_t{2}}) {
    native_scan_fixture residual{input.view(), output_width, true};
    expect(residual.scan->string_view_eligible(), "static residual grants view eligibility");
    auto filtered = serve(entry, *residual.scan);
    {
      auto read       = filtered.data->to_read_only();
      auto const& rep = read.get_data()->cast<sirius::compressed_device_representation>();
      expect(!rep.pushdown_scan() && rep.string_view_eligible(),
             "static residual has an eligible empty decode request");
    }
    sirius::op::scan::scan_operator_input split{filtered.data};
    admit(split);
    split.row_filter_pending = true;
    auto* allocator =
      space.get_memory_resource_as<cucascade::memory::reservation_aware_resource_adaptor>();
    expect(allocator != nullptr, "residual allocation tracker");
    auto const baseline = allocator->get_allocated_bytes(stream);
    allocator->reset_peak_allocated_bytes(stream);
    split.prepare_for_processing(&space, stream);
    expect(!split.stolen_table && !split.converted_table_steal_pending,
           "residual input stays available for filtering and retry");
    {
      auto read = filtered.data->to_read_only();
      expect((dynamic_cast<sirius::borrowed_string_batch_representation const*>(read.get_data()) !=
              nullptr) == !copy,
             "static residual takes the view route unless the copy switch is set");
      auto prepared = sirius::get_cudf_table_view(read);
      expect((prepared.column(0).head<void>() == chars) == !copy,
             "residual input chars route pointer");
      expect((prepared.column(0).child(0).head<void>() == offsets) == !copy,
             "residual input offsets route pointer");
    }
    // Fail after the filter has produced its output, then re-enter on the retained split.
    // Neither the failed assembly nor a subsequent execution may consume the source view.
    {
      using namespace sirius::op::scan::detail;
      read_retirement_test_hooks fault{read_checkpoint::NORMALIZATION};
      bool failed = false;
      try {
        (void)residual.scan->execute(split, stream);
      } catch (rmm::out_of_memory const&) {
        failed = true;
      }
      expect(failed && fault.checkpoint_reached, "inject OOM after residual compaction");
    }
    for (int attempt = 0; attempt < 2; ++attempt) {
      split.prepare_for_processing(&space, stream);
      auto output    = residual.scan->execute(split, stream);
      auto& pipeline = dynamic_cast<sirius::op::pipelineable_operator_data&>(*output);
      auto read      = pipeline.get_data_batches()[0]->to_read_only();
      auto view      = sirius::get_cudf_table_view(read);
      expect(view.num_rows() == 2 && view.num_columns() == static_cast<int>(output_width),
             "residual emits the projected survivors");
      expect(view.column(0).head<void>() != chars, "residual output chars are compacted");
      expect(view.column(0).child(0).head<void>() != offsets,
             "residual output offsets are compacted");
      auto expected = make_strings_table({"alpha", "alpha"}, {}, stream);
      expect(columns_equal_any(expected->view().column(0), view.column(0), stream),
             "residual survivor values");
      if (output_width == 2) {
        auto expected_numeric = make_int32_table(1, 2, 7);
        std::array<std::int32_t, 2> values{7, 58};  // Input rows 0 and 3.
        cuda_check(cudaMemcpyAsync(expected_numeric->mutable_view().column(0).head<void>(),
                                   values.data(),
                                   sizeof(values),
                                   cudaMemcpyHostToDevice,
                                   stream.get()));
        expect(columns_equal_any(expected_numeric->view().column(0), view.column(1), stream),
               "residual companion values");
      }
    }
    auto const peak = allocator->get_peak_allocated_bytes(stream);
    expect(peak >= baseline, "residual peak includes its baseline");
    expect((peak - baseline >= chars_bytes) == copy,
           "only the copy switch may allocate full-width string channels before filtering");
    auto read = filtered.data->to_read_only();
    expect(sirius::get_cudf_table_view(read).num_rows() == input.num_rows(),
           "residual retry retains the full source rows");
  }

  native_scan_fixture residual{input.view(), 2, true};
  auto residual_equality = serve(entry, *residual.scan, equality);
  sirius::op::scan::scan_operator_input residual_split{residual_equality.data};
  admit(residual_split);
  residual_split.row_filter_pending = true;
  residual_split.prepare_for_processing(&space, stream);
  auto residual_output    = residual.scan->execute(residual_split, stream);
  auto& residual_pipeline = dynamic_cast<sirius::op::pipelineable_operator_data&>(*residual_output);
  auto residual_read      = residual_pipeline.get_data_batches()[0]->to_read_only();
  auto residual_view      = sirius::get_cudf_table_view(residual_read);
  auto expected_strings   = make_strings_table({"alpha", "alpha"}, {}, stream);
  expect(columns_equal_any(expected_strings->view().column(0), residual_view.column(0), stream),
         "narrowed equality still applies the residual and emits the matching rows");

  // A direct converter without an eligibility grant also keeps the standalone owned contract.
  auto standalone = source(blob, space);
  standalone->set_string_view_eligible(false);
  auto decoded = registry.convert<gpu_rep>(*standalone, &space, stream);
  expect(dynamic_cast<sirius::borrowed_string_batch_representation*>(decoded.get()) == nullptr,
         "converter without scan eligibility takes owned decode");

  native_scan_fixture dynamic{input.view(), 2, false, true};
  expect(dynamic.scan->decode_consumer() && !dynamic.scan->string_view_eligible(),
         "a wired dynamic filter vetoes views before publication");
  auto dynamic_batch = serve(entry, *dynamic.scan);
  {
    auto read       = dynamic_batch.data->to_read_only();
    auto const& rep = read.get_data()->cast<sirius::compressed_device_representation>();
    expect(!rep.string_view_eligible(), "unpublished membership keeps the scan-wide veto");
    auto dynamic_decoded = registry.convert<gpu_rep>(rep, &space, stream);
    expect(
      dynamic_cast<sirius::borrowed_string_batch_representation*>(dynamic_decoded.get()) == nullptr,
      "wired dynamic filter takes owned decode without an attached probe");
  }

  if (!sirius::late_mat::late_mat_enabled()) return;
  native_scan_fixture deferring{input.view(), 2, false, true};
  auto keys   = make_int32_table(1, 1, 7);
  auto filter = std::make_shared<sirius::op::sirius_dynamic_small_in_list_filter>(
    keys->view().column(0), stream, rmm::mr::get_current_device_resource_ref());
  auto producer = deferring.channel->register_producer({1});
  expect(producer.push_filter(1, filter), "publish a membership probe before deferral");
  producer.finish(sirius::op::sirius_dynamic_filter_set::completion::PUBLISHED);
  expect(bool(deferring.scan->decode_consumer()), "non-deferring scan exposes decode consumer");
  auto const probes = deferring.consumer->select_for_decode(deferring.consumer->decode_bindings());
  expect(probes.steps.size() == 1 && probes.steps[0].column_index == 1,
         "non-deferring scan offers the published membership probe for the companion column");
  sirius::op::scan::sirius_physical_dynamic_filter port{
    deferring.scan->types, input.num_rows(), deferring.consumer};
  sirius::late_mat::column_origin origin;
  origin.handle     = entry->late_mat_handle;
  origin.column_pos = 0;
  origin.generation = entry->late_mat_handle->generation();
  auto const schema = deferring.scan->normalization_targets();
  // Isolate the deferral veto from the wired-consumer veto checked above.
  sirius::op::scan::sirius_physical_dynamic_filter plain_port{
    plain.scan->types, input.num_rows(), nullptr};
  expect(sirius::planner::install_deferral(
           *plain.scan,
           plain_port,
           sirius::late_mat::make_defer_pair(schema, {0}, schema, {0}, {origin})),
         "install a deferral without a dynamic consumer");
  expect(!plain.scan->consumer() && !plain.scan->string_view_eligible(),
         "deferral alone vetoes pin views");
  auto deferred_plain = serve(entry, *plain.scan);
  {
    auto read       = deferred_plain.data->to_read_only();
    auto const& rep = read.get_data()->cast<sirius::compressed_device_representation>();
    auto decoded    = registry.convert<gpu_rep>(rep, &space, stream);
    expect(dynamic_cast<sirius::borrowed_string_batch_representation*>(decoded.get()) == nullptr,
           "deferral without a consumer takes owned decode");
  }
  expect(
    sirius::planner::install_deferral(
      *deferring.scan, port, sirius::late_mat::make_defer_pair(schema, {0}, schema, {0}, {origin})),
    "install the late-materialization deferral");
  expect(!deferring.scan->decode_consumer(), "deferral withholds the available membership probe");
  expect(!deferring.scan->string_view_eligible(), "deferral vetoes pin views");
  auto deferred = serve(entry, *deferring.scan);
  expect(deferred.origin != nullptr, "late-mat env enables pin-order origin annotation");
  {
    auto read       = deferred.data->to_read_only();
    auto const& rep = read.get_data()->cast<sirius::compressed_device_representation>();
    expect((!rep.pushdown_scan() || rep.pushdown_scan()->request().empty()) &&
             !rep.string_view_eligible(),
           "withheld membership leaves an ineligible empty request");
    auto decoded_deferred = registry.convert<gpu_rep>(rep, &space, stream);
    expect(dynamic_cast<sirius::borrowed_string_batch_representation*>(decoded_deferred.get()) ==
             nullptr,
           "deferral takes owned decode even with no attached request");
    expect(decoded_deferred->get_table_view().column(0).head<void>() != chars,
           "deferral does not borrow pin chars");
    expect(columns_equal_any(
             input.view().column(0), decoded_deferred->get_table_view().column(0), stream),
           "withheld membership preserves all pin-order rows");
  }
  sirius::op::scan::scan_operator_input deferred_split{deferred.data};
  admit(deferred_split);
  deferred_split.origin   = deferred.origin;
  deferred_split.consumer = deferring.scan->decode_consumer();
  deferred_split.prepare_for_processing(&space, stream);
  auto deferred_output    = deferring.scan->execute(deferred_split, stream);
  auto& deferred_pipeline = dynamic_cast<sirius::op::pipelineable_operator_data&>(*deferred_output);
  auto deferred_read      = deferred_pipeline.get_data_batches()[0]->to_read_only();
  auto deferred_view      = sirius::get_cudf_table_view(deferred_read);
  expect(deferred_view.num_rows() == input.num_rows(), "deferral emits all pin-order rows");
  expect(deferred_view.column(0).type().id() == cudf::type_id::UINT64,
         "deferral emits its rowid instead of string values");
  expect(columns_equal_any(input.view().column(1), deferred_view.column(1), stream),
         "withheld membership leaves the companion values in pin order");
  std::array<std::uint64_t, 4> rowids{};
  cuda_check(cudaMemcpyAsync(rowids.data(),
                             deferred_view.column(0).head<void>(),
                             sizeof(rowids),
                             cudaMemcpyDeviceToHost,
                             stream.get()));
  stream.sync();
  expect(rowids == std::array<std::uint64_t, 4>{0, 1, 2, 3}, "deferred rowid values");
}

std::string framed_plan(std::string const& codec = "snappy")
{
  return "input -> str_split -> offsets, chars\nstr_split.chars -> " + codec +
         "\nstr_split.offsets -> delta -> differences\n"
         "str_split.offsets.differences -> bitpack -> chunk_min, chunk_count, chunk_bits, packed\n";
}

void nullable_membership(cucascade::memory::memory_space& space,
                         ::cuda::stream_ref stream,
                         std::string const& codec)
{
  auto const mr = rmm::mr::get_current_device_resource_ref();
  auto build    = make_strings_table({"", "x"}, {false, true}, stream);
  auto input    = make_strings_table({"", "miss"}, {false, true}, stream);
  auto blob     = pin(input->view(), stream, mr, framed_plan(codec));
  auto probe    = [&](cudf::column_view keys,
                      std::uint32_t const*,
                      ::cuda::stream_ref lane,
                      rmm::device_async_resource_ref resource) {
    auto const build_keys = build->view().column(0);
    auto hits             = cudf::contains(build_keys, keys, lane, resource);
    return cudf::replace_nulls(
      hits->view(),
      cudf::numeric_scalar<bool>{build_keys.has_nulls(), true, lane, resource},
      lane,
      resource);
  };
  using sirius::codegen::selection_mask;
  rmm::device_buffer mask(selection_mask::AllocWordsFor(2) * sizeof(std::uint32_t), stream, mr);
  auto* words = static_cast<std::uint32_t*>(mask.data());
  for (bool selected_key : {false, true}) {
    auto status = simpatico::decode_one(
      simpatico::mask_decode_request{
        *blob->table.columns[0].plan_tree,
        simpatico::membership_source{probe, input->view().column(0).type(), nullptr, selected_key},
        {words, 2}},
      stream,
      mr);
    std::uint32_t hits = 0;
    cuda_check(cudaMemcpyAsync(&hits, words, sizeof(hits), cudaMemcpyDeviceToHost, stream.get()));
    stream.sync();
    expect(status == simpatico::mask_source_status::ACCEPTED &&
             (hits & 3) == (selected_key && simpatico::chunk_selective_enabled()),
           "direct membership mask must restore validity only for a selected key with the gate on");
  }
  sirius::pushdown_request request;
  request.columns.resize(1);
  request.columns[0].membership.push_back({probe});
  request.membership_measurement = {.measure = true, .complete = true};
  sirius::decompression_pushdown_scan scan{std::move(request)};
  std::array<std::size_t, 1> projection{0};
  auto output = sirius::decompress_chunk(
    blob->table,
    projection,
    &scan,
    {},
    space,
    cucascade::memory::reservation_aware_resource_adaptor::AllocationTrackingScope::PER_THREAD,
    stream,
    space.get_default_allocator(),
    false,
    blob);
  if (simpatico::chunk_selective_enabled()) {
    auto const& counts = output.outcome.membership_keep;
    expect(counts && counts->complete && counts->rows_before == 2 && counts->rows_after == 1,
           "pinned scan must apply the NULL-safe membership probe");
  } else {
    expect(output.outcome.membership_skipped && output.table->num_rows() == 2,
           "full route leaves both rows for the join");
    auto hits    = probe(output.table->view().column(0), nullptr, stream, mr);
    output.table = cudf::apply_boolean_mask(output.table->view(), hits->view(), stream, mr);
  }
  auto expected = make_strings_table({""}, {false}, stream);
  stream.sync();
  expect(strings_equal(output.table->view().column(0), expected->view().column(0), stream),
         "NULL-safe string join must return exactly the NULL row");
}

void use_bounded_reservation(cucascade::memory::memory_space& space,
                             ::cuda::stream_ref stream,
                             std::size_t bytes)
{
  auto* tracker =
    space.get_memory_resource_as<cucascade::memory::reservation_aware_resource_adaptor>();
  tracker->reset_stream_reservation(stream);
  expect(tracker->attach_reservation_to_tracker(
           stream,
           space.make_reservation(bytes),
           std::make_unique<cucascade::memory::fail_reservation_limit_policy>(),
           std::make_unique<cucascade::memory::throw_on_oom_policy>()),
         "attach bounded decode reservation");
}

std::shared_ptr<sirius::pinned_compressed_blob> make_host_blob(
  simpatico::compressed_table const& table,
  cucascade::memory::memory_space& host_space,
  ::cuda::stream_ref stream,
  std::vector<std::size_t>* column_bytes = nullptr)
{
  std::vector<std::uint8_t> header;
  std::uint64_t payload_bytes = 0;
  std::vector<simpatico::payload_buffer_ref> buffers;
  auto error =
    simpatico::build_compressed_table_header(table, header, buffers, payload_bytes, stream);
  expect(error.empty(), error.c_str());
  if (column_bytes) {
    column_bytes->assign(table.num_columns(), 0);
    for (auto const& buffer : buffers)
      (*column_bytes)[buffer.column_index] += buffer.size_bytes;
  }
  auto host_blob             = std::make_shared<sirius::pinned_compressed_blob>();
  host_blob->header          = std::move(header);
  host_blob->payload_bytes   = payload_bytes;
  host_blob->nvcomp_metadata = simpatico::collect_nvcomp_pin_metadata(table);
  auto* host_mr              = host_space.get_memory_resource_of<cucascade::memory::Tier::HOST>();
  host_blob->payload         = host_mr->allocate_multiple_blocks(host_blob->payload_bytes);
  for (auto const& buffer : buffers)
    if (buffer.size_bytes)
      sirius::copy_device_to_pinned_blocks(
        buffer.device_ptr, *host_blob->payload, buffer.offset, buffer.size_bytes, stream);
  stream.sync();
  return host_blob;
}

// Independent byte accounting for mask slots, small per-column allocations and the optional arena.
std::size_t admission_overhead(std::size_t rows, std::size_t columns)
{
  auto const mask_bytes = ((rows + 2047) / 2048) * 256;
  return 8 * mask_bytes + columns * 65536 +
         (sirius::codegen::decode_scratch_arena_enabled() ? 1048576 : 0);
}

template <typename T>
std::string describe(T const& value)
{
  return std::format("{}", value);
}

template <typename T>
std::string describe(std::vector<T> const& values)
{
  std::string result = "[";
  for (T const& value : values) {
    if (result.size() > 1) result += ", ";
    result += describe(value);
  }
  return result + "]";
}

void expect_equal(std::string const& field, auto const& actual, auto const& expected)
{
  if (actual != expected)
    throw std::runtime_error(
      std::format("{}: expected {}, got {}", field, describe(expected), describe(actual)));
}

void decoded_like_cases(cucascade::memory::memory_space& space,
                        ::cuda::stream_ref stream,
                        std::string const& mode)
{
  bool const enabled = mode != "0" && mode != "unset";
  bool const frames  = mode != "no_frames";
  expect_equal("decoded LIKE gate", simpatico::decoded_like_enabled(), enabled);
  expect_equal("OnPair LIKE gate", simpatico::onpair_like_enabled(), false);
  expect_equal("frame selection gate", simpatico::chunk_selective_enabled(), frames);
  auto const mr      = rmm::mr::get_current_device_resource_ref();
  constexpr int rows = 259;  // Partial mask word and five 64 KiB payload blocks.
  std::vector<std::string> payload(rows);
  for (int i = 0; i < rows; ++i) {
    payload[i] = std::to_string(i);
    payload[i].resize(1024, 'x');
  }
  auto projected = make_strings_table(payload, {}, stream);
  std::vector<std::string> plans{
    framed_plan("snappy"),
    framed_plan("lz4"),
    framed_plan("ans"),
    framed_plan("deflate"),
    "input -> dictionary -> keys_offsets, keys_chars, indices\ndictionary.indices -> bitpack\n",
    "input -> str_split -> offsets, chars\nstr_split.offsets -> bitpack\n"};
#ifdef SIMPATICO_ONPAIR
  plans.push_back("input -> onpair_16\n");
#endif
  std::array<std::size_t, 2> const selected{0, 1};
  for (auto const& plan : plans) {
    for (bool nullable : {false, true}) {
      std::vector<std::string> text(rows, "common");
      text[0]   = "common google red blue";
      text[4]   = "common google blue red";
      text[64]  = "common red blue";
      text[130] = "common google red-blue";
      text[258] = "common Google red red";
      std::vector<bool> valid(rows, true);
      if (nullable) valid[0] = false;
      auto strings = make_strings_table(text, valid, stream);
      auto blob    = pin(cudf::table_view{{strings->view().column(0), projected->view().column(0)}},
                      stream,
                      mr,
                      plan + "---\n" + framed_plan());
      auto full    = simpatico::decompress(blob->table, stream, mr);
      auto logical = strings->alloc_size() + projected->alloc_size();
      for (auto const& pattern : std::vector<simpatico::like_predicate>{{{"google"}, false},
                                                                        {{"google"}, true},
                                                                        {{"red", "blue"}, false},
                                                                        {{"missing"}, false},
                                                                        {{"common"}, false}}) {
        auto check = [&](std::string_view field, auto const& actual, auto const& expected) {
          expect_equal(std::format("{} (mode={}, plan={}, literals={}, invert={}, nullable={})",
                                   field,
                                   mode,
                                   plan,
                                   describe(pattern.literals),
                                   pattern.invert,
                                   nullable),
                       actual,
                       expected);
        };
        sirius::pushdown_request request;
        if (enabled)
          request.decoded_like_cache = std::make_shared<sirius::like_multiliteral_cache>();
        request.columns.resize(2);
        request.columns[0].likes = {pattern};
        // A second conjunct exercises mask conjunction and sharing one source decode.
        if (pattern.literals.size() == 2) request.columns[0].likes.push_back({{"google"}, false});
        request.ranges_cover_whole_filter = true;
        auto scan = sirius::decompression_pushdown_scan{request}.for_chunk(blob->table, selected);
        check("chunk narrowed", bool(scan), enabled);
        auto rep = source(blob, space, logical);
        rep->set_pushdown_scan(scan);
        auto scratch = rep->selected_decode_scratch_bytes();
        if (enabled)
          expect(scratch >= logical,
                 std::format("decoded LIKE scratch={} < logical={}", scratch, logical));
        else
          check("selection scratch bytes", scratch, 0u);
        // Off: b4b1ac63 narrows away this sole source and needs no selection scratch.
        if (enabled) {
          simpatico::chunk_selective_plain_scope plain{scratch};
          check("plain grant scratch bytes", rep->selected_decode_scratch_bytes(), 0u);
          check("plain grant narrowed",
                bool(sirius::decompression_pushdown_scan{request}.for_chunk(blob->table, selected)),
                false);
          if (nullable && pattern.literals.front() == "missing") {
            auto fallback =
              sirius::decompress_chunk(blob->table,
                                       selected,
                                       scan.get(),
                                       {},
                                       space,
                                       cucascade::memory::reservation_aware_resource_adaptor::
                                         AllocationTrackingScope::PER_THREAD,
                                       stream,
                                       space.get_default_allocator());
            check("plain grant rows", fallback.table->num_rows(), rows);
            check("plain grant columns", fallback.table->num_columns(), 2);
            check("plain grant row_filtered", fallback.outcome.row_filtered, false);
            check("plain grant enforced_like_columns",
                  fallback.outcome.enforced_like_columns,
                  std::vector<std::size_t>{});
            check("plain grant selected blocks", fallback.outcome.selected_frame_blocks, 0u);
          }
        }
        auto expected_mask = [&](cudf::column_view values) {
          cudf::strings_column_view strings{values};
          if (pattern.literals.size() == 2) {
            auto ordered  = cudf::strings::like(strings, "%red%blue%", "", stream, mr);
            auto contains = cudf::strings::contains(
              strings, cudf::string_scalar("google", true, stream, mr), stream, mr);
            return cudf::binary_operation(ordered->view(),
                                          contains->view(),
                                          cudf::binary_operator::LOGICAL_AND,
                                          cudf::data_type{cudf::type_id::BOOL8},
                                          stream,
                                          mr);
          }
          auto contains = cudf::strings::contains(
            strings, cudf::string_scalar(pattern.literals.front(), true, stream, mr), stream, mr);
          return pattern.invert
                   ? cudf::unary_operation(contains->view(), cudf::unary_operator::NOT, stream, mr)
                   : std::move(contains);
        };
        auto mask            = expected_mask(full->view().column(0));
        auto expected        = cudf::apply_boolean_mask(full->view(), mask->view(), stream, mr);
        auto const survivors = expected->num_rows();
        auto const capped    = mode == "capped" && survivors > rows * 0.10;
        if (enabled) use_bounded_reservation(space, stream, logical + logical / 4 + scratch);
        watched_plan    = blob->table.columns[0].plan_tree.get();
        watched_decodes = 0;
        auto result =
          sirius::decompress_chunk(blob->table,
                                   selected,
                                   scan.get(),
                                   {},
                                   space,
                                   cucascade::memory::reservation_aware_resource_adaptor::
                                     AllocationTrackingScope::PER_THREAD,
                                   stream,
                                   space.get_default_allocator());
        watched_plan = nullptr;
        check("projected LIKE source decodes", watched_decodes, 1u);
        bool const applied = enabled && !capped;
        check("row_filtered", result.outcome.row_filtered, applied);
        check("enforced_like_columns",
              result.outcome.enforced_like_columns,
              applied ? std::vector<std::size_t>{0} : std::vector<std::size_t>{});
        check("output columns", result.table->num_columns(), 2);
        check("scan output rows", result.table->num_rows(), applied ? survivors : rows);
        if (!applied) {
          check("residual has full string values",
                strings_equal(result.table->view().column(0), full->view().column(0), stream),
                true);
          auto residual = expected_mask(result.table->view().column(0));
          result.table =
            cudf::apply_boolean_mask(result.table->view(), residual->view(), stream, mr);
        }
        check("survivors after residual", result.table->num_rows(), survivors);
        for (int i : {0, 1})
          check(std::format("column {} matches full decode plus residual", i),
                strings_equal(result.table->view().column(i), expected->view().column(i), stream),
                true);
        // Every payload row is 1024 bytes. Count the distinct source blocks of the reference rows.
        auto hits = host_validity_bits(mask->view());
        std::vector<uint8_t> bits(rows);
        cuda_check(
          cudaMemcpy(bits.data(), mask->view().head<void>(), rows, cudaMemcpyDeviceToHost));
        std::array<bool, 5> touched{};
        for (int i = 0; i < rows; ++i)
          if (hits[i] && bits[i]) touched[i / 64] = true;
        auto blocks = static_cast<std::size_t>(std::count(touched.begin(), touched.end(), true));
        check("wave 2 survivor payload blocks",
              result.outcome.selected_frame_blocks,
              applied && frames ? blocks : 0);
      }
    }
  }
}

void all_survivor_like_offsets(cucascade::memory::memory_space& space, ::cuda::stream_ref stream)
{
  auto const mr = space.get_default_allocator();
  auto input    = make_strings_column({"match", "match twice", "rematch"}, {}, stream);
  auto contents = input->release();
  auto offsets  = cudf::cast(
    contents.children.front()->view(), cudf::data_type{cudf::type_id::INT64}, stream, mr);
  input = cudf::make_strings_column(
    3, std::move(offsets), std::move(*contents.data), 0, rmm::device_buffer{});
  auto blob = pin(cudf::table_view{{input->view()}}, stream, mr, "input -> str_split\n");
  std::array<std::size_t, 1> const selected{0};
  sirius::codegen::scan_filter_request request;
  request.routes = {sirius::codegen::decode_route::full};
  request.like_filters.push_back({0, {{"match"}, false}});
  void const* probe_chars = nullptr;
  if (simpatico::decoded_like_enabled()) {
    request.membership_filters.push_back(
      {0,
       [&](cudf::column_view keys,
           std::uint32_t const* prior,
           ::cuda::stream_ref lane,
           rmm::device_async_resource_ref resource) {
         expect_equal("one-slot probe prior supplied", prior != nullptr, true);
         probe_chars = keys.head<void>();
         return cudf::strings::contains(cudf::strings_column_view{keys},
                                        cudf::string_scalar{"match", true, lane, resource},
                                        lane,
                                        resource);
       }});
  }
  std::array<::cuda::stream_ref, 1> const lanes{stream};
  sirius::codegen::scan_filter_result result;
  auto output =
    simpatico::decompress_scan_filter(blob->table, selected, request, result, lanes, stream, mr);
  expect_equal("one-slot applied", result.applied, simpatico::decoded_like_enabled());
  if (result.applied) expect_equal("one-slot survivors", result.survivor_count, 3);
  expect_equal("one-slot survivor index bytes", result.row_indices.size(), 0u);
  expect_equal("one-slot output rows", output->num_rows(), 3);
  expect_equal("one-slot output columns", output->num_columns(), 1);
  auto actual = output->view().column(0);
  if (result.applied) expect_equal("one-slot retained key chars", actual.head<void>(), probe_chars);
  expect_equal("all-survivor offsets type",
               sirius::test::type_id_to_string(actual.child(0).type().id()),
               "INT64");
  expect_equal("all-survivor offsets",
               sirius::test::detail::copy_device_values<std::int64_t>(
                 actual.child(0).head<void>(), actual.child(0).size(), stream),
               std::vector<std::int64_t>{0, 5, 16, 23});
  expect_equal("all-survivor strings", strings_equal(actual, input->view(), stream), true);
}

void decoded_like_membership(::cuda::stream_ref stream, bool nullable)
{
  using sirius::test::detail::copy_device_values;
  auto const mr = rmm::mr::get_current_device_resource_ref();
  std::vector<std::string> text{"match A", "match B", "match C", "match D", "match E", "match F"};
  std::vector<std::string> dict(text.size(), "keep");
  std::vector<bool> valid(text.size(), true);
  if (nullable) {
    dict[1]  = "skip";
    text[2]  = "other";
    valid[4] = false;
  }
  auto dict_input = make_strings_column(dict, {}, stream);
  auto key        = make_strings_column(text, nullable ? valid : std::vector<bool>{}, stream);
  auto blob       = pin(cudf::table_view{{dict_input->view(), key->view()}},
                  stream,
                  mr,
                  "input -> dictionary\n---\ninput -> str_split\n");
  rmm::cuda_stream key_stream;
  std::array<::cuda::stream_ref, 2> const lanes{stream, key_stream};
  std::array<std::size_t, 2> const selected{0, 1};
  sirius::codegen::scan_filter_request request;
  request.routes.assign(2, sirius::codegen::decode_route::full);
  request.bool8_filters.push_back({0, {"keep"}});
  request.like_filters.push_back({1, {{"match"}, false}});
  auto check = [&](std::string_view field, auto const& actual, auto const& expected) {
    expect_equal(std::format("membership {} (nullable={})", field, nullable), actual, expected);
  };
  void const* probe_chars = nullptr;
  request.membership_filters.push_back(
    {1,
     [&](cudf::column_view keys,
         std::uint32_t const* prior,
         ::cuda::stream_ref lane,
         rmm::device_async_resource_ref resource) {
       check("prior supplied", prior != nullptr, true);
       check("probe stream", static_cast<void*>(lane.get()), static_cast<void*>(stream.get()));
       check("key type", sirius::test::type_id_to_string(keys.type().id()), "STRING");
       check("key rows", keys.size(), static_cast<int>(text.size()));
       check("key null count", keys.null_count(), nullable ? 1 : 0);
       auto const bits = sirius::test::detail::column_row_validity(keys, lane);
       check("key validity", std::vector<bool>(bits.begin(), bits.end()), valid);
       probe_chars = keys.head<void>();
       if (!nullable)
         return cudf::strings::contains(cudf::strings_column_view{keys},
                                        cudf::string_scalar{"match", true, lane, resource},
                                        lane,
                                        resource);
       // Of the live non-null LIKE matches A, D and F, reject D.
       auto reject = cudf::strings::contains(cudf::strings_column_view{keys},
                                             cudf::string_scalar{"D", true, lane, resource},
                                             lane,
                                             resource);
       auto keep = cudf::unary_operation(reject->view(), cudf::unary_operator::NOT, lane, resource);
       return cudf::replace_nulls(
         keep->view(), cudf::numeric_scalar<bool>{false, true, lane, resource}, lane, resource);
     }});

  sirius::codegen::scan_filter_result result;
  watched_plan        = blob->table.columns[1].plan_tree.get();
  watched_decode_lane = nullptr;
  // Count value walks inside mask requests too. Only the key is str_split.
  static thread_local std::size_t str_split_decodes;
  str_split_decodes = 0;
  simpatico::string_decode_log_scope count{[](char const*) { ++str_split_decodes; }};
  auto output =
    simpatico::decompress_scan_filter(blob->table, selected, request, result, lanes, stream, mr);
  watched_plan = nullptr;
  check("applied", result.applied, true);
  check("key decode stream",
        static_cast<void*>(watched_decode_lane),
        static_cast<void*>(key_stream.value()));
  check("str_split decodes", str_split_decodes, 1u);
  check("probes applied", result.membership_probes_applied, 1u);
  std::vector<std::int32_t> survivors =
    nullable ? std::vector<std::int32_t>{0, 5} : std::vector<std::int32_t>{0, 1, 2, 3, 4, 5};
  check("survivor count", result.survivor_count, static_cast<std::int64_t>(survivors.size()));
  check("output rows", output->num_rows(), static_cast<int>(survivors.size()));
  check("output columns", output->num_columns(), 2);
  check("survivor index bytes", result.row_indices.size(), survivors.size() * sizeof(std::int32_t));
  check("sibling survivor indices",
        copy_device_values<std::int32_t>(result.row_indices.data(), survivors.size(), stream),
        survivors);
  auto sibling = output->view().column(0);
  check("sibling type", sirius::test::type_id_to_string(sibling.type().id()), "BOOL8");
  check("sibling values",
        copy_device_values<std::uint8_t>(sibling.head<void>(), sibling.size(), stream),
        std::vector<std::uint8_t>(survivors.size(), 1));
  std::vector<std::string> expected;
  for (auto row : survivors)
    expected.push_back(text[row]);
  auto projected_key = output->view().column(1);
  check("projected key values",
        strings_equal(projected_key, make_strings_column(expected, {}, stream)->view(), stream),
        true);
  if (!nullable) check("retained key chars", projected_key.head<void>(), probe_chars);
}

void framed_reservation(cucascade::memory::memory_space& space,
                        ::cuda::stream_ref stream,
                        std::string const& codec)
{
  constexpr std::size_t rows = 1 << 20;
  auto const mr              = rmm::mr::get_current_device_resource_ref();
  auto strings               = make_strings_table(std::vector<std::string>(rows, "x"), {}, stream);
  auto keys                  = make_int32_table(1, rows, 0);
  cudf::table_view input{
    {strings->view().column(0), strings->view().column(0), keys->view().column(0)}};
  auto blob = pin(input,
                  stream,
                  mr,
                  framed_plan(codec) + "---\n" + framed_plan(codec) + "---\ninput -> bitpack\n");
  auto const batch_bytes = 2 * strings->alloc_size() + keys->alloc_size();
  sirius::pushdown_request request;
  request.columns.resize(3);
  request.columns[2].range          = sirius::decode_range{0, 999};
  request.ranges_cover_whole_filter = true;
  auto scan = std::make_shared<const sirius::decompression_pushdown_scan>(std::move(request));
  auto rep  = source(blob, space, batch_bytes);
  rep->set_pushdown_scan(scan);
  sirius::op::scan::scan_operator_input split{cucascade::data_batch::make(1, std::move(rep))};
  split.row_filter_pending = true;
  auto estimate            = split.get_estimated_working_set_size_in_bytes();

  std::array<std::size_t, 3> projection{0, 1, 2};
  auto const scratch =
    2 * frame_budget(rows, rows, stream, codec) + batch_bytes + admission_overhead(rows, 3);
  expect(estimate == batch_bytes / 4 + batch_bytes + scratch,
         "scan must reserve the full-width outputs and both frames' scratch");
  sirius::scan_manager::mvcc_chunk_mask const visibility{
    std::make_shared<std::uint32_t[]>((rows + 31) / 32), rows};
  auto visible_rep = source(blob, space, batch_bytes);
  visible_rep->set_visibility_mask({visibility.words, rows});
  sirius::op::scan::scan_operator_input visible{
    cucascade::data_batch::make(2, std::move(visible_rep))};
  visible.mvcc_keep_mask = visibility;
  sirius::op::scan::scan_operator_input::selective_decode_memory memory;
  auto const visible_peak = visible.get_estimated_working_set_size_in_bytes(std::nullopt, &memory);
  expect(memory.scratch == scratch && visible_peak == std::max(2 * batch_bytes + rows, estimate) +
                                                        visible.mvcc_keep_mask.view().size_bytes(),
         "MVCC is a possible source at admission");
  auto* tracker =
    space.get_memory_resource_as<cucascade::memory::reservation_aware_resource_adaptor>();
  auto const baseline = tracker->get_allocated_bytes(stream);
  tracker->reset_peak_allocated_bytes(stream);
  auto output = sirius::decompress_chunk(
    blob->table,
    projection,
    scan.get(),
    {},
    space,
    cucascade::memory::reservation_aware_resource_adaptor::AllocationTrackingScope::PER_THREAD,
    stream,
    space.get_default_allocator(),
    false,
    blob);
  stream.sync();
  expect(output.outcome.row_filtered && output.table->num_rows() == rows,
         "both framed columns must decode all one-byte strings");
  auto const peak = tracker->get_peak_allocated_bytes(stream) - baseline;
  expect(peak >= (2 * sizeof(std::int32_t) + sizeof(std::int64_t) + 2) * rows,
         "measured peak must include lengths, starts, offsets, decoded blocks and survivor chars");
  expect(estimate >= peak, "scan reservation falls below the measured multi-lane decode peak");

  cucascade::memory::host_memory_space_config config;
  config.numa_id              = 0;
  config.block_size           = 1 << 20;
  config.pool_size            = (blob->leaves.size() + config.block_size - 1) / config.block_size;
  config.memory_capacity      = config.pool_size * config.block_size;
  config.initial_number_pools = 1;
  cucascade::memory::memory_space host_space{config};
  auto host_blob = make_host_blob(blob->table, host_space, stream);
  sirius::compressed_host_representation pinned{
    host_space, host_blob, {"s1", "s2", "k"}, host_blob->payload_bytes, batch_bytes, rows};
  expect(sirius::prefetch_materialization_bytes(&pinned) == host_blob->payload_bytes + batch_bytes,
         "unfiltered host prefetch must use the plain reservation");
  pinned.set_pushdown_scan(scan);
  auto const prefetch = sirius::prefetch_materialization_bytes(&pinned);
  expect(prefetch == host_blob->payload_bytes + estimate,
         "host prefetch must cover the upload and the scan's selected-string estimate");
  output.table.reset();
  stream.sync();
  use_bounded_reservation(space, stream, prefetch);
  auto const prefetch_baseline = tracker->get_allocated_bytes(stream);
  tracker->reset_peak_allocated_bytes(stream);
  cucascade::representation_converter_registry registry;
  sirius::register_compression_converters(registry);
  auto uploaded = registry.convert<gpu_rep>(pinned, &space, stream);
  stream.sync();
  auto* decoded =
    dynamic_cast<sirius::decompression_pushdown_batch_representation*>(uploaded.get());
  expect(decoded && decoded->outcome().row_filtered &&
           uploaded->get_table_view().num_rows() == rows &&
           prefetch >= tracker->get_peak_allocated_bytes(stream) - prefetch_baseline,
         "prefetch reservation must cover the host-to-GPU converter's measured peak");
}

void warmed_framed_reservation(cucascade::memory::memory_space& space,
                               ::cuda::stream_ref stream,
                               std::string const& codec)
{
  constexpr int rows   = 65536;
  constexpr int blocks = 64;
  auto const mr        = rmm::mr::get_current_device_resource_ref();
  auto strings =
    make_strings_table(std::vector<std::string>(rows, std::string(64, 'x')), {}, stream);
  sirius::pushdown_request request;
  request.columns.resize(2);
  request.columns[1].range          = sirius::decode_range{1, 1};
  request.ranges_cover_whole_filter = true;
  auto scan = std::make_shared<const sirius::decompression_pushdown_scan>(std::move(request));
  namespace pipeline = sirius::pipeline;
  auto global        = std::make_shared<pipeline::gpu_pipeline_task_global_state>(
    nullptr, sirius::test::make_test_telemetry_context());
  auto* tracker =
    space.get_memory_resource_as<cucascade::memory::reservation_aware_resource_adaptor>();
  sirius::converter_registry::initialize();
  std::size_t clustered_peak = 0;
  for (bool scattered : {false, true}) {
    std::vector<std::int32_t> host(rows);
    for (int i = 0; i < blocks; ++i)
      host[scattered ? i * (rows / blocks) : i] = 1;
    auto keys = cudf::make_numeric_column(
      cudf::data_type{cudf::type_id::INT32}, rows, cudf::mask_state::UNALLOCATED, stream, mr);
    cuda_check(cudaMemcpyAsync(keys->mutable_view().head<void>(),
                               host.data(),
                               rows * sizeof(host[0]),
                               cudaMemcpyHostToDevice,
                               stream.get()));
    auto blob = pin(cudf::table_view{{strings->view().column(0), keys->view()}},
                    stream,
                    mr,
                    framed_plan(codec) + "---\ninput -> bitpack\n");
    auto rep  = source(blob, space, strings->alloc_size() + keys->alloc_size());
    rep->set_pushdown_scan(scan);
    auto split = std::make_unique<sirius::op::scan::scan_operator_input>(
      cucascade::data_batch::make(scattered ? 2 : 1, std::move(rep)));
    split->row_filter_pending = true;
    auto& chunk               = *split;
    admit(chunk);
    auto const estimate = chunk.get_estimated_working_set_size_in_bytes();
    pipeline::gpu_pipeline_task task(
      scattered ? 2 : 1,
      std::vector<cucascade::shared_data_repository*>{},
      std::make_unique<pipeline::gpu_pipeline_task_local_state>(std::move(split)),
      global);
    auto const admission = task.get_estimated_reservation_size_info(&space);
    expect(admission.had_history == scattered, "second resident chunk must use scheduler history");
    use_bounded_reservation(space, stream, admission.reservation_size);
    auto const baseline = tracker->get_allocated_bytes(stream);
    tracker->reset_peak_allocated_bytes(stream);
    chunk.prepare_for_processing(&space, stream);
    stream.sync();
    auto const peak = tracker->get_peak_allocated_bytes(stream) - baseline;
    expect(
      chunk.pushdown_row_filtered && chunk.stolen_table && chunk.stolen_table->num_rows() == blocks,
      "both chunks must compact to the same number of rows");
    if (scattered) {
      expect(peak > clustered_peak && admission.peak_memory_estimate >= estimate &&
               admission.reservation_size >= peak,
             "warmed reservation must cover the scattered block decode");
    } else {
      clustered_peak = peak;
      global->get_memory_history().record(
        {admission.input_basis, peak, chunk.stolen_table->alloc_size()});
    }
  }
}

void check_scan_reservation(std::shared_ptr<cucascade::data_batch> batch,
                            cucascade::memory::memory_space& gpu,
                            sirius::scan_manager::mvcc_chunk_mask mask,
                            std::size_t logical,
                            std::size_t materialization,
                            std::size_t cold_peak,
                            std::size_t warm_peak,
                            bool reserve_warm = false)
{
  namespace pipe = sirius::pipeline;
  auto pipeline =
    std::make_shared<pipe::sirius_pipeline>(pipe::pipeline_build_context{nullptr, true});
  sirius::op::scan::sirius_gpu_scan_operator op{
    duckdb::vector<sirius::logical_type>{}, 0, nullptr, 1};
  pipe::sirius_pipeline_build_state build;
  build.set_pipeline_source(*pipeline, op);
  build.add_pipeline_operator(*pipeline, op);
  auto global = std::make_shared<pipe::gpu_pipeline_task_global_state>(
    pipeline, sirius::test::make_test_telemetry_context());
  auto split = std::make_unique<sirius::op::scan::scan_operator_input>(std::move(batch));
  split->row_filter_pending = !mask.has_mask();
  split->mvcc_keep_mask     = std::move(mask);
  pipe::gpu_pipeline_task task{
    1,
    std::vector<cucascade::shared_data_repository*>{},
    std::make_unique<pipe::gpu_pipeline_task_local_state>(std::move(split)),
    global};
  for (bool warm : {false, true}) {
    if (warm) global->get_memory_history().record({logical, logical, logical});
    auto const info     = task.get_estimated_reservation_size_info(&gpu);
    auto const expected = warm ? warm_peak : cold_peak;
    expect(info.had_history == warm && info.bytes_to_materialize_input == materialization &&
             info.input_basis == logical && info.peak_memory_estimate == expected &&
             info.reservation_size == materialization + expected,
           "incorrect scan reservation bytes");
    if (warm && reserve_warm)
      expect(bool(gpu.make_reservation_or_null(info.reservation_size)),
             "remote GPU must fit its warmed reservation");
  }
}

std::unique_ptr<gpu_rep> bounded_host_decode(sirius::compressed_host_representation const& rep,
                                             cucascade::memory::memory_space& gpu,
                                             ::cuda::stream_ref stream)
{
  auto const budget = sirius::prefetch_materialization_bytes(&rep);
  use_bounded_reservation(gpu, stream, budget);
  auto* tracker =
    gpu.get_memory_resource_as<cucascade::memory::reservation_aware_resource_adaptor>();
  cucascade::representation_converter_registry registry;
  sirius::register_compression_converters(registry);
  auto result = registry.convert<gpu_rep>(rep, &gpu, stream);
  stream.sync();
  auto const peak = tracker->get_peak_allocated_bytes(stream);
  expect(peak <= budget,
         std::format("host conversion peak={} exceeds reservation={}", peak, budget));
  tracker->reset_stream_reservation(stream);
  return result;
}

// A host pin without nvCOMP frame metadata still needs a grant for full decoded LIKE sources.
void decoded_like_host_admission(cucascade::memory::memory_space& gpu, ::cuda::stream_ref stream)
{
  constexpr int rows = 32;
  std::vector<std::string> titles(rows, "other"), urls(rows, "home");
  titles[1] = titles[2] = titles[3] = "news";
  urls[2]                           = "read";  // Contains "ad", so row 2 fails NOT LIKE.
  std::vector<bool> valid(rows, true);
  valid[3]     = false;
  auto title   = make_strings_table(titles, {}, stream);
  auto url     = make_strings_table(urls, valid, stream);
  auto id      = make_int32_table(1, rows, 0);
  auto logical = id->alloc_size() + title->alloc_size() + url->alloc_size();
  auto blob =
    pin(cudf::table_view{{id->view().column(0), title->view().column(0), url->view().column(0)}},
        stream,
        rmm::mr::get_current_device_resource_ref(),
        "input -> bitpack\n---\ninput -> identity\n---\ninput -> dictionary\n");
  cucascade::memory::host_memory_space_config config;
  config.numa_id              = 0;
  config.block_size           = 1 << 20;
  config.pool_size            = 1;
  config.memory_capacity      = config.block_size;
  config.initial_number_pools = 1;
  cucascade::memory::memory_space host{config};
  auto bytes = make_host_blob(blob->table, host, stream);
  expect_equal("host pin has frame metadata", bool(bytes->nvcomp_metadata), false);
  expect_equal("host declined selective bytes before conversion",
               simpatico::declined_chunk_selective_bytes,
               0u);
  sirius::compressed_host_representation rep{
    host, bytes, {"id", "title", "url"}, bytes->payload_bytes, logical, rows};
  sirius::pushdown_request request;
  request.columns.resize(3);
  request.columns[1].likes          = {{{"news"}, false}};
  request.columns[2].likes          = {{{"ad"}, true}};
  request.ranges_cover_whole_filter = true;
  request.output_prefix_width       = 1;
  rep.set_pushdown_scan(std::make_shared<const sirius::decompression_pushdown_scan>(request));
  auto plain  = sirius::peak_materialization_bytes(&rep);
  auto budget = sirius::prefetch_materialization_bytes(&rep);
  bool const enabled = simpatico::decoded_like_enabled();
  if (enabled)
    expect(budget > plain,
           std::format("decoded LIKE host grant={} must exceed plain={}", budget, plain));
  else
    expect_equal("legacy host grant", budget, plain);

  auto result = [&] {
    if (enabled) return bounded_host_decode(rep, gpu, stream);
    cucascade::representation_converter_registry registry;
    sirius::register_compression_converters(registry);
    return registry.convert<gpu_rep>(rep, &gpu, stream);
  }();
  auto view = result->get_table_view();
  auto const* decoded =
    dynamic_cast<sirius::decompression_pushdown_batch_representation const*>(result.get());
  sirius::pushdown_outcome const plain_outcome{};
  auto const& outcome = decoded ? decoded->outcome() : plain_outcome;
  auto check          = [&](std::string_view field, auto const& actual, auto const& expected) {
    expect_equal(
      std::format("host {} (rows={}, columns={}, row_filtered={}, enforced_like_columns={})",
                  field,
                  view.num_rows(),
                  view.num_columns(),
                  outcome.row_filtered,
                  describe(outcome.enforced_like_columns)),
      actual,
      expected);
  };
  check("row_filtered", outcome.row_filtered, enabled);
  check("enforced_like_columns",
        outcome.enforced_like_columns,
        enabled ? std::vector<std::size_t>{1, 2} : std::vector<std::size_t>{});
  check("output columns", view.num_columns(), enabled ? 1 : 3);
  check("output rows", view.num_rows(), enabled ? 1 : rows);
  check("id type", sirius::test::type_id_to_string(view.column(0).type().id()), "INT32");
  std::unique_ptr<cudf::table> residual;
  if (!enabled) {
    auto const mr = rmm::mr::get_current_device_resource_ref();
    auto news =
      cudf::strings::like(cudf::strings_column_view{view.column(1)}, "%news%", "", stream, mr);
    auto ad =
      cudf::strings::like(cudf::strings_column_view{view.column(2)}, "%ad%", "", stream, mr);
    auto keep_url = cudf::unary_operation(ad->view(), cudf::unary_operator::NOT, stream, mr);
    auto keep     = cudf::binary_operation(news->view(),
                                       keep_url->view(),
                                       cudf::binary_operator::LOGICAL_AND,
                                       cudf::data_type{cudf::type_id::BOOL8},
                                       stream,
                                       mr);
    residual      = cudf::apply_boolean_mask(view, keep->view(), stream, mr);
    view          = residual->view();
  }
  check("survivor ids",
        sirius::test::detail::copy_device_values<std::int32_t>(
          view.column(0).head<void>(), view.num_rows(), stream),
        std::vector<std::int32_t>{17});
  if (enabled) {
    simpatico::chunk_selective_plain_scope declined{budget};
    expect_equal(
      "host plain grant after decline", sirius::prefetch_materialization_bytes(&rep), plain);
  }
}

void tiny_mask_admission(cucascade::memory::memory_space& gpu,
                         cucascade::memory::memory_space& host,
                         ::cuda::stream_ref stream)
{
  auto strings = make_strings_table({""}, {}, stream);
  auto keys    = make_int32_table(8, 1, 0);
  std::vector<cudf::column_view> views{strings->view().column(0)};
  std::string plans = framed_plan();
  sirius::pushdown_request request;
  request.columns.resize(9);
  request.ranges_cover_whole_filter = true;
  for (int i = 0; i < 8; ++i) {
    views.push_back(keys->view().column(i));
    plans += "---\ninput -> bitpack\n";
    request.columns[i + 1].range = sirius::decode_range{0, 999};
  }
  auto blob      = pin(cudf::table_view{views}, stream, gpu.get_default_allocator(), plans);
  auto host_blob = make_host_blob(blob->table, host, stream);
  constexpr std::size_t logical = 40;  // two offsets and eight integers
  sirius::compressed_host_representation rep{
    host, host_blob, std::vector<std::string>(9), host_blob->payload_bytes, logical, 1};
  rep.set_pushdown_scan(
    std::make_shared<const sirius::decompression_pushdown_scan>(std::move(request)));
  auto const budget = sirius::prefetch_materialization_bytes(&rep);
  auto const base   = logical + host_blob->payload_bytes;
  if (!simpatico::chunk_selective_enabled()) {
    expect(budget == base, "legacy one-row prefetch reservation");
    return;
  }
  auto const non_mask = base + logical / 4 + logical + frame_budget(1, 0, stream) + 9 * 65536 +
                        (sirius::codegen::decode_scratch_arena_enabled() ? 1048576 : 0);
  expect(budget == non_mask + 8 * 256, "one separately aligned allocation per selection mask");
  auto result = bounded_host_decode(rep, gpu, stream);
  auto const* decoded =
    dynamic_cast<sirius::decompression_pushdown_batch_representation const*>(result.get());
  expect(decoded && decoded->outcome().row_filtered && decoded->get_table_view().num_rows() == 1,
         "eight-source mask must select the single row");
}

void latched_mask_admission(cucascade::memory::memory_space& gpu, ::cuda::stream_ref stream)
{
  using namespace sirius;
  constexpr int rows = 4096;
  auto strings =
    make_strings_table(std::vector<std::string>(rows, std::string(64, 'x')), {}, stream);
  auto dict = make_strings_table(std::vector<std::string>(rows, "match"), {}, stream);
  auto keys = make_int32_table(7, rows, 0);
  std::vector<cudf::column_view> views{strings->view().column(0), dict->view().column(0)};
  std::string plans = framed_plan() +
                      "---\ninput -> dictionary -> keys_offsets, keys_chars, "
                      "indices\ndictionary.indices -> bitpack\n";
  pushdown_request request;
  request.columns.resize(9);
  request.columns[1].equals_any = {"match"};
  for (int i = 0; i < 7; ++i) {
    views.push_back(keys->view().column(i));
    plans += "---\ninput -> bitpack\n";
    request.columns[i + 2].range = decode_range{0, 999};
  }
  auto blob          = pin(cudf::table_view{views}, stream, gpu.get_default_allocator(), plans);
  auto const logical = strings->alloc_size() + dict->alloc_size() + keys->alloc_size();
  auto rep           = source(blob, gpu, logical);
  auto words         = std::make_shared<std::uint32_t[]>(rows / 32);
  for (int i = 0; i < rows; i += 1024)
    words[i / 32] = 1;
  rep->set_visibility_mask({words, rows});
  rep->set_pushdown_scan(std::make_shared<const decompression_pushdown_scan>(std::move(request)));
  auto batch                             = cucascade::data_batch::make(3, std::move(rep));
  auto split                             = std::make_unique<op::scan::scan_operator_input>(batch);
  split->row_filter_pending              = true;
  split->mvcc_keep_mask                  = {words, rows};
  auto latch                             = std::make_shared<std::atomic<bool>>(false);
  split->pushdown_selection_unprofitable = latch;
  auto& chunk                            = *split;
  admit(chunk);
  auto global = std::make_shared<pipeline::gpu_pipeline_task_global_state>(
    nullptr, test::make_test_telemetry_context());
  global->get_memory_history().record({logical, logical, logical});
  pipeline::gpu_pipeline_task task{
    1,
    std::vector<cucascade::shared_data_repository*>{},
    std::make_unique<pipeline::gpu_pipeline_task_local_state>(std::move(split)),
    global};
  auto const budget  = task.get_estimated_reservation_size_info(&gpu).reservation_size;
  bool const enabled = simpatico::chunk_selective_enabled();
  if (!enabled) {
    expect(budget == logical, "legacy warmed reservation before latch transition");
    return;
  }
  expect(budget > 2 * logical, "the over-cap request must reserve before the latch changes");
  latch->store(true);
  use_bounded_reservation(gpu, stream, budget);
  converter_registry::initialize();
  chunk.prepare_for_processing(&gpu, stream);
  stream.sync();
  auto read = batch->to_read_only();
  auto const* decoded =
    dynamic_cast<decompression_pushdown_batch_representation const*>(read.get_data());
  expect(decoded && decoded->outcome().visibility_mask_applied &&
           decoded->get_table_view().num_rows() == 4 &&
           decoded->outcome().selected_frame_blocks == 4,
         "latched dictionary equality and visibility must decode all four scattered blocks");
  gpu.get_memory_resource_as<cucascade::memory::reservation_aware_resource_adaptor>()
    ->reset_stream_reservation(stream);
}

#ifdef SIMPATICO_ONPAIR
void predicate_admission(cucascade::memory::memory_space& gpu,
                         cucascade::memory::memory_space& host,
                         ::cuda::stream_ref stream)
{
  constexpr std::size_t rows = 4096;
  auto const literal         = std::string(32, 'a');
  std::vector<std::string> comments(rows, "miss"), keys(rows, "hit");
  comments[0] = comments[1] = literal;
  keys[1]                   = "miss";
  auto strings              = make_strings_table(std::vector<std::string>(rows, "x"), {}, stream);
  auto dict                 = make_strings_table(keys, {}, stream);
  auto onpair               = make_strings_table(comments, {}, stream);
  auto blob                 = pin(
    cudf::table_view{{strings->view().column(0), dict->view().column(0), onpair->view().column(0)}},
    stream,
    gpu.get_default_allocator(),
    framed_plan() +
      "---\ninput -> dictionary -> keys_offsets, keys_chars, indices\n"
      "dictionary.indices -> bitpack\n---\ninput -> onpair_16\n");
  auto host_blob = make_host_blob(blob->table, host, stream);
  auto logical   = strings->alloc_size() + dict->alloc_size() + onpair->alloc_size();
  sirius::compressed_host_representation rep{
    host, host_blob, {"frame", "key", "comment"}, host_blob->payload_bytes, logical, rows};
  sirius::pushdown_request request;
  request.columns.resize(3);
  request.columns[1].equals_any     = {"hit", std::string(16 << 20, 'z')};
  request.columns[2].likes          = {{{literal}, false}};
  request.ranges_cover_whole_filter = true;
  request.output_prefix_width       = 1;
  rep.set_pushdown_scan(
    std::make_shared<const sirius::decompression_pushdown_scan>(std::move(request)));
  auto const base   = logical + host_blob->payload_bytes;
  auto const budget = sirius::prefetch_materialization_bytes(&rep);
  if (!simpatico::chunk_selective_enabled()) {
    expect(budget == base, "legacy prefetch does not separately charge predicate buffers");
    return;
  }
  expect(budget >= base + (16 << 20) + 3 + 3 * sizeof(int32_t) + 33 * 65536 * sizeof(uint16_t),
         "reserve the IN literal storage and LIKE transitions");
  auto result = bounded_host_decode(rep, gpu, stream);
  auto const* decoded =
    dynamic_cast<sirius::decompression_pushdown_batch_representation const*>(result.get());
  expect(decoded && decoded->outcome().row_filtered &&
           decoded->outcome().selected_frame_blocks == 1 &&
           result->get_table_view().num_rows() == 1,
         "the strict reservation must cover both predicates and the surviving string");
}
#endif

void scan_capacity_fallback(::cuda::stream_ref stream)
{
  using namespace sirius;
  namespace pipe     = sirius::pipeline;
  constexpr int rows = 65536;
  cucascade::memory::gpu_memory_space_config config;
  config.device_id                  = 0;
  config.per_stream_reservation     = false;
  config.memory_capacity            = 12 << 20;
  config.reservation_limit_fraction = 1;
  cucascade::memory::memory_space gpu{config};
  auto ballast = gpu.make_reservation_or_null(1 << 20);
  expect(bool(ballast), "could not hold the concurrent reservation");
  // The executor and what it references are destroyed after every batch holder: the decoded
  // table frees its device memory on an executor stream.
  exec::query_lifecycle_registry lifecycle;
  exec::channel<std::unique_ptr<pipe::task_request>> requests;
  auto telemetry = test::make_test_telemetry_context();
  std::unique_ptr<pipe::gpu_pipeline_executor> executor_owner;

  std::vector<std::string> values(rows), expected;
  for (int i = 0; i < rows; ++i) {
    values[i] = std::to_string(i);
    values[i].resize(64, 'x');
    if (i % 1000 == 0) expected.push_back(values[i]);
  }
  auto strings       = make_strings_table(values, {}, stream);
  auto keys          = make_int32_table(1, rows, 0);
  auto blob          = pin(cudf::table_view{{strings->view().column(0), keys->view().column(0)}},
                           stream,
                           rmm::mr::get_current_device_resource_ref(),
                           framed_plan() + "---\ninput -> bitpack\n");
  auto const logical = strings->alloc_size() + keys->alloc_size();
  auto const legacy  = 2 * logical + rows * sizeof(std::int64_t);
  auto rep           = source(blob, gpu, logical);
  pushdown_request request;
  request.columns.resize(2);
  request.columns[1].range          = decode_range{0, 0};
  request.ranges_cover_whole_filter = true;
  rep->set_pushdown_scan(std::make_shared<const decompression_pushdown_scan>(std::move(request)));
  auto batch                = cucascade::data_batch::make(4, std::move(rep));
  auto split                = std::make_unique<op::scan::scan_operator_input>(batch);
  split->row_filter_pending = true;
  // A scan that widens its integer key keeps the decoded table until the cast.
  split->needs_carrier_conversion     = true;
  split->conversion_destination_bytes = rows * sizeof(std::int64_t);
  admit(*split);
  auto pipeline  = std::make_shared<pipe::sirius_pipeline>(pipe::pipeline_build_context{telemetry});
  pipeline->set_query_id(make_query_id(0));
  op::scan::sirius_gpu_scan_operator scan{duckdb::vector<logical_type>{}, rows, nullptr, 1};
  pipe::sirius_pipeline_build_state build;
  build.set_pipeline_source(*pipeline, scan);
  build.add_pipeline_operator(*pipeline, scan);
  pipe::assign_operator_ids({pipeline});  // task execution reads operator ids
  auto global = std::make_shared<pipe::gpu_pipeline_task_global_state>(pipeline, telemetry);
  global->get_memory_history().record({logical, logical, logical});
  struct observed_scan : pipe::gpu_pipeline_task {
    using gpu_pipeline_task::gpu_pipeline_task;
    std::promise<std::size_t> granted;
    std::unique_ptr<op::operator_data> compute_task(::cuda::stream_ref) override
    {
      granted.set_value(
        _local_state->cast<pipe::gpu_pipeline_task_local_state>().get_reservation_bytes());
      return nullptr;
    }
  };
  auto task = std::make_unique<observed_scan>(
    1,
    std::vector<cucascade::shared_data_repository*>{},
    std::make_unique<pipe::gpu_pipeline_task_local_state>(std::move(split)),
    global);
  auto done           = task->granted.get_future();
  auto const estimate = task->get_estimated_reservation_size_info(&gpu).reservation_size;
  if (simpatico::chunk_selective_enabled())
    expect(estimate > gpu.get_max_memory(), "selective scan must exceed the space capacity");
  else
    expect(estimate == legacy, "gate-off scan must request the legacy history estimate");
  lifecycle.open_query(pipeline->get_query_id());
  executor_owner.reset(new pipe::gpu_pipeline_executor{
    lifecycle, {1, "scan-capacity"}, &gpu, requests.make_publisher(), nullptr, telemetry});
  auto& executor = *executor_owner;
  converter_registry::initialize();
  executor.start();
  expect(executor.schedule(std::move(task)), "scan submission failed");
  bool const ready = done.wait_for(std::chrono::seconds(90)) == std::future_status::ready;
  ballast.reset();
  executor.stop();
  expect(ready && done.get() == legacy,
         "scan must finish with the legacy grant while ballast is held");
  auto read = batch->to_read_only();
  auto const* result =
    dynamic_cast<decompression_pushdown_batch_representation const*>(read.get_data());
  expect(result && result->outcome().row_filtered && result->outcome().selected_frame_blocks == 0,
         "capacity fallback must gather the full decoded strings");
  result->wait_for_writer(stream);
  auto wanted = make_strings_table(expected, {}, stream);
  expect(strings_equal(result->get_table_view().column(0), wanted->view().column(0), stream),
         "capacity fallback returned different survivor strings");
}

void mixed_host_admission(::cuda::stream_ref stream, bool report_bytes, bool cross_gpu)
{
  namespace sm = sirius::scan_manager;
  using namespace cucascade::memory;
  constexpr std::size_t rows        = 32768;
  constexpr std::size_t large_bytes = 32 << 20;
  gpu_memory_space_config gpu_config;
  gpu_config.per_stream_reservation     = false;
  gpu_config.device_id                  = 0;
  gpu_config.reservation_limit_fraction = 1.0;
  gpu_config.memory_capacity            = 512ull << 20;
  host_memory_space_config host_config;
  host_config.numa_id              = 0;
  host_config.memory_capacity      = 32 << 20;
  host_config.block_size           = 1 << 20;
  host_config.pool_size            = 4;
  host_config.initial_number_pools = 1;
  sirius::memory::sirius_memory_reservation_manager manager{{gpu_config, host_config}};
  auto& gpu  = *manager.get_memory_space(Tier::GPU, 0);
  auto& host = *manager.get_memory_space(Tier::HOST, 0);
  if (!report_bytes && !cross_gpu) {
    scan_capacity_fallback(stream);
#ifdef SIMPATICO_ONPAIR
    predicate_admission(gpu, host, stream);
#endif
    tiny_mask_admission(gpu, host, stream);
    latched_mask_admission(gpu, stream);
  }
  std::vector<std::string> framed(rows, "x");
  framed[1].assign(3 * 65536, 'z');  // Only row 1 touches the two middle blocks.
  auto small = make_strings_table(framed, {}, stream);
  std::vector<std::string> skewed(rows);
  skewed.front().assign(large_bytes, 'z');
  auto large     = make_strings_table(skewed, {}, stream);
  auto keys      = make_int32_table(1, rows, 0);
  // Raw offsets keep the skewed column on the full-decode route.
  std::string const full_plan = "input -> str_split -> offsets, chars\nstr_split.chars -> lz4\n";
  auto blob =
    pin(cudf::table_view{{small->view().column(0),
                          large->view().column(0),
                          keys->view().column(0),
                          keys->view().column(0),
                          small->view().column(0)}},
        stream,
        gpu.get_default_allocator(),
        framed_plan("lz4") + "---\n" + full_plan +
          "---\ninput -> bitpack\n---\ninput -> identity\n---\n" + framed_plan("deflate"));
  auto sizes     = std::make_shared<sirius::per_column_byte_sizes>();
  auto host_blob = make_host_blob(blob->table, host, stream, &sizes->compressed);
  std::vector<std::string> names{"s1", "large", "range", "plain", "s2"};
  sizes->uncompressed = {small->alloc_size(),
                         large->alloc_size(),
                         keys->alloc_size(),
                         keys->alloc_size(),
                         small->alloc_size()};
  auto const logical =
    std::accumulate(sizes->uncompressed.begin(), sizes->uncompressed.end(), std::size_t{0});
  auto const upload  = host_blob->payload_bytes + logical;
  auto const* gate   = std::getenv("SIRIUS_EXP_CHUNK_SELECTIVE");
  bool const enabled = gate && std::strcmp(gate, "1") == 0;
  expect(enabled == simpatico::chunk_selective_enabled(),
         "admission gate disagrees with the environment");
  auto const chars        = rows - 1 + 3 * 65536;
  auto const frame        = enabled ? frame_budget(rows, chars, stream, "lz4") : 0;
  auto const second_frame = enabled ? frame_budget(rows, chars, stream, "deflate") : 0;
  auto const scratch      = frame + second_frame + logical + admission_overhead(rows, names.size());
  auto const decode_peak = logical + logical / 4 + scratch;
  std::shared_ptr<const sirius::decompression_pushdown_scan> prefetch_scan;
  enum class filter { bitpack_range, identity_range, visibility };
  for (auto scenario : {filter::bitpack_range, filter::identity_range, filter::visibility}) {
    bool const viable = scenario == filter::bitpack_range;
    sm::mvcc_chunk_mask mask;
    sirius::pushdown_request request;
    if (scenario == filter::visibility) {
      mask = {std::make_shared<std::uint32_t[]>((rows + 31) / 32), rows};
    } else {
      request.columns.resize(names.size());
      request.columns[viable ? 2 : 3].range = sirius::decode_range{0, 0};
      request.ranges_cover_whole_filter     = true;
    }
    auto scan   = std::make_shared<const sirius::decompression_pushdown_scan>(std::move(request));
    auto pinned = std::make_unique<sirius::compressed_host_representation>(
      host, host_blob, names, host_blob->payload_bytes, logical, rows, sizes);
    auto resident = std::make_unique<sirius::compressed_device_representation>(
      gpu, blob, names, blob->leaves.size(), logical, rows, sizes);
    pinned->set_pushdown_scan(scan);
    resident->set_pushdown_scan(scan);
    pinned->set_visibility_mask({mask.words, mask.row_count});
    resident->set_visibility_mask({mask.words, mask.row_count});
    if (report_bytes) {
      std::printf("ADMISSION_BYTES=%zu\n", sirius::prefetch_materialization_bytes(pinned.get()));
      return;
    }
    auto const plain_peak = 2 * logical + mask.row_count + mask.view().size_bytes();
    expect(sirius::prefetch_materialization_bytes(pinned.get()) ==
             (enabled ? host_blob->payload_bytes + decode_peak : upload),
           "host prefetch must charge only eligible frames and every full-width gather");
    if (viable) {
      std::array<std::size_t, 3> projection{1, 2, 0};
      sirius::pushdown_request projected_request;
      projected_request.columns.resize(3);
      projected_request.columns[1].range = sirius::decode_range{0, 0};
      projected_request.ranges_cover_whole_filter = true;
      auto projected                     = pinned->select_columns(projection);
      projected->set_pushdown_scan(
        std::make_shared<const sirius::decompression_pushdown_scan>(std::move(projected_request)));
      auto const projected_logical = small->alloc_size() + large->alloc_size() + keys->alloc_size();
      auto const base              = projected->get_uncompressed_data_size_in_bytes();
      auto const expected          = frame + admission_overhead(rows, projection.size()) +
                                     projected_logical +
                                     (projected_logical - std::min(base, projected_logical));
      expect(projected->selected_decode_scratch_bytes() == (enabled ? expected : 0),
             "a reordered projection must include the projected logical widths");
      sirius::compressed_host_representation legacy{
        host, host_blob, names, host_blob->payload_bytes, logical, rows};
      auto narrow = legacy.select_columns(projection);
      narrow->set_pushdown_scan(projected->pushdown_scan());
      auto const apportioned = logical * projection.size() / names.size();
      auto const payload     = host_blob->payload_bytes * projection.size() / names.size();
      expect(sirius::prefetch_materialization_bytes(narrow.get()) ==
               (enabled ? host_blob->payload_bytes + apportioned / 4 + 2 * logical + frame +
                            admission_overhead(rows, projection.size())
                        : payload + apportioned),
             "a skewed projection without sizes must cover the whole pin for input and output");
      if (enabled && !cross_gpu) {
        auto result = bounded_host_decode(*narrow, gpu, stream);
        auto const* decoded =
          dynamic_cast<sirius::decompression_pushdown_batch_representation const*>(result.get());
        auto view = result->get_table_view();
        expect(decoded && decoded->outcome().selected_frame_blocks == 2 &&
                 view.num_rows() == 1 + (rows - 1) / 1000 &&
                 cudf::strings_column_view(view.column(0)).chars_size(stream) == large_bytes,
               "projected legacy pin must gather the skewed string within its reservation");
      }
    }
    auto host_batch  = cucascade::data_batch::make(1, std::move(pinned));
    auto gpu_batch   = cucascade::data_batch::make(2, std::move(resident));
    auto const floor = std::max(plain_peak, decode_peak + mask.view().size_bytes());
    check_scan_reservation(host_batch,
                           gpu,
                           mask,
                           logical,
                           upload,
                           enabled ? floor - logical : plain_peak,
                           enabled ? floor - logical : logical);
    check_scan_reservation(
      gpu_batch, gpu, mask, logical, 0, enabled ? floor : plain_peak, enabled ? floor : logical);
    if (cross_gpu && viable) {
      gpu_config.device_id = 1;
      memory_space remote{gpu_config};
      auto const materialize = blob->leaves.size() + logical;
      auto const peak        = enabled ? floor - logical : logical;
      auto const available   = remote.get_available_memory();
      auto ballast           = remote.make_reservation_or_null(available - materialize - peak);
      expect(bool(ballast), "could not bound remote admission");
      check_scan_reservation(gpu_batch,
                             remote,
                             mask,
                             logical,
                             materialize,
                             enabled ? floor - logical : plain_peak,
                             peak,
                             true);
      return;
    }
    if (viable) prefetch_scan = std::move(scan);
  }
  // Model a spill between the prefetcher's two locks. The same payload costs an
  // additional header under the exclusive lock. A second reader distinguishes the lock modes.
  struct sampled_pin : sirius::compressed_host_representation {
    using compressed_host_representation::compressed_host_representation;
    std::weak_ptr<cucascade::data_batch> owner;
    std::size_t* grant;
    std::size_t get_size_in_bytes() const override
    {
      auto bytes = compressed_host_representation::get_size_in_bytes();
      if (auto batch = owner.lock()) {
        bool shared =
          std::async(std::launch::async, [&] { return bool(batch->try_to_read_only()); }).get();
        if (!shared) bytes += header().size();
      }
      return bytes;
    }
  };
  sirius::converter_registry::initialize();
  sirius::converter_registry::get().register_converter<sampled_pin, gpu_rep>(
    [](auto const& input, auto* space, ::cuda::stream_ref lane, auto*) {
      auto const& pin = static_cast<sampled_pin const&>(input);
      *pin.grant      = space->get_available_memory(lane) - space->get_available_memory();
      return sirius::converter_registry::get().convert<gpu_rep>(*pin.clone(lane), space, lane);
    });
  for (bool tight : {false, true}) {
    if (tight && !enabled) continue;
    struct provider : sm::databatch_provider {
      std::shared_ptr<cucascade::data_batch> pending;
      batch get_next_batch() override { return {std::move(pending)}; }
    } source;
    std::size_t grant{};
    auto pinned = std::make_unique<sampled_pin>(
      host, host_blob, names, host_blob->payload_bytes, logical, rows, sizes);
    pinned->set_pushdown_scan(prefetch_scan);
    pinned->grant  = &grant;
    auto& observed = *pinned;
    auto batch     = cucascade::data_batch::make(3, std::move(pinned));
    if (!enabled) observed.owner = batch;
    source.pending = batch;
    auto connector = std::make_shared<sm::split_connector>();
    sm::load_balancing_scan_batch_coalescer::drain_cached_provider(source, *connector, {}, true);
    auto const budget = host_blob->payload_bytes + decode_peak;
    expect(gpu.get_available_memory() > budget, "test space must fit one mixed-route conversion");
    auto ballast =
      enabled ? gpu.make_reservation_or_null(gpu.get_available_memory() - budget + (tight ? 1 : 0))
              : nullptr;
    expect(!enabled || bool(ballast), "reserve all but the mixed-route prefetch budget");
    sm::memory_prefetcher_config config;
    config.enable            = true;
    config.num_threads       = 1;
    config.min_free_fraction = 0;
    config.poll_interval_ms  = 5;
    sirius::exec::query_lifecycle_registry lifecycle;
    sm::memory_prefetcher worker{config, {connector}, &gpu, lifecycle};
    auto const deadline = std::chrono::steady_clock::now() + std::chrono::seconds(90);
    while (!worker.batches_prefetched() && !lifecycle.runtime_failed() &&
           std::chrono::steady_clock::now() < deadline)
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    worker.stop();
    expect(worker.batches_prefetched() == 1, "mixed-route sweep must fit its hard reservation");
    expect(grant == (enabled && !tight ? budget : upload),
           "the worker must request the exact selected or legacy payload-plus-output bytes");
    auto read = batch->to_read_only();
    auto const* result =
      dynamic_cast<sirius::decompression_pushdown_batch_representation const*>(read.get_data());
    expect(result && result->outcome().row_filtered,
           "prefetch must actually compact the mixed-route pin");
    expect(result->outcome().selected_frame_blocks == (enabled && !tight ? 4 : 0),
           "each framed output must submit blocks 0 and 3");
    result->wait_for_writer(stream);
    auto view = result->get_table_view();
    expect(view.num_rows() == 1 + (rows - 1) / 1000 &&
             cudf::strings_column_view(view.column(1)).chars_size(stream) == large_bytes,
           "the sparse range must gather all bytes of the full-route string");
  }
}

void large_strings(bool copy,
                   cucascade::memory::memory_space& space,
                   cucascade::representation_converter_registry& registry,
                   ::cuda::stream_ref stream,
                   std::string const& codec)
{
  auto const mr                = rmm::mr::get_current_device_resource_ref();
  constexpr std::int64_t bytes = (std::int64_t{1} << 31) + 5;  // UINT32 carrier with a padded tail.
  std::array<std::int64_t, 3> offsets{0, std::int64_t{1} << 30, bytes};
  auto off = cudf::make_fixed_width_column(
    cudf::data_type{cudf::type_id::INT64}, 3, cudf::mask_state::UNALLOCATED, stream, mr);
  cuda_check(cudaMemcpyAsync(off->mutable_view().head<void>(),
                             offsets.data(),
                             sizeof(offsets),
                             cudaMemcpyHostToDevice,
                             stream.get()));
  rmm::device_buffer chars(bytes, stream, mr);
  cuda_check(cudaMemsetAsync(chars.data(), 'Q', chars.size(), stream.get()));
  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(cudf::make_strings_column(
    2,
    std::move(off),
    std::move(chars),
    0,
    cudf::create_null_mask(0, cudf::mask_state::UNALLOCATED, stream, mr)));
  cudf::table input{std::move(columns)};
  auto blob       = pin(input.view(), stream, mr, codec.empty() ? "" : framed_plan(codec));
  auto rep        = source(blob, space);
  auto output     = registry.convert<gpu_rep>(*rep, &space, stream);
  auto const view = output->get_table_view().column(0);
  if (!codec.empty()) {
    auto const& plan  = *blob->table.columns[0].plan_tree;
    auto const node   = std::find_if(
      plan.nodes.begin(), plan.nodes.end(), [&](auto const& n) { return n.op == codec; });
    expect(node != plan.nodes.end() && node->rep &&
             node->rep->original_type.id() == cudf::type_id::UINT32,
           "oversized framed chars must have a widened carrier");
    expect(simpatico::probe_column(plan).compact_route == sirius::codegen::decode_route::full,
           "oversized framed chars must use full decode");
  } else {
    auto const& stored = stored_strings(*blob);
    expect(stored.channels_[0]->type().id() == cudf::type_id::INT64, "large INT64 offsets");
    expect(stored.channels_[1]->type().id() == cudf::type_id::UINT32, "large UINT32 chars carrier");
    expect((view.head<void>() == stored.channels_[1]->view().head<void>()) == !copy,
           "large chars pointer");
  }
  expect(view.child(0).type().id() == cudf::type_id::INT64, "large served INT64 offsets");
  expect(cudf::strings_column_view(view).chars_size(stream) == bytes, "large exact logical bytes");
  std::array<std::int64_t, 3> actual_offsets;
  cuda_check(cudaMemcpyAsync(actual_offsets.data(),
                             view.child(0).head<void>(),
                             sizeof(actual_offsets),
                             cudaMemcpyDeviceToHost,
                             stream.get()));
  stream.sync();
  expect(actual_offsets == offsets, "large offsets values");
  std::vector<char> actual(8U << 20);
  for (std::int64_t pos = 0; pos < bytes; pos += actual.size()) {
    auto const n = std::min<std::size_t>(actual.size(), bytes - pos);
    cuda_check(cudaMemcpyAsync(actual.data(),
                               static_cast<char const*>(view.head<void>()) + pos,
                               n,
                               cudaMemcpyDeviceToHost,
                               stream.get()));
    stream.sync();
    expect(std::all_of(actual.begin(), actual.begin() + n, [](char c) { return c == 'Q'; }),
           "large strings byte identity");
  }
}

}  // namespace

int main(int argc, char** argv)
{
  try {
    bool const framed       = argc > 1 && std::strcmp(argv[1], "--framed") == 0;
    bool const framed_large = argc > 1 && std::strcmp(argv[1], "--large-framed") == 0;
    std::string const codec = argc > 2 ? argv[2] : "snappy";
    bool const large        = framed_large || (argc > 1 && std::strcmp(argv[1], "--large") == 0);
    bool const copy  = (argc > 1 && std::strcmp(argv[argc - 1], "--copy") == 0);
    expect(simpatico::string_decode_copy_enabled() == copy, "copy switch/process mismatch");
    bool const report_bytes = argc == 2 && std::strcmp(argv[1], "--admission-bytes") == 0;
    bool const cross_gpu    = argc == 2 && std::strcmp(argv[1], "--cross-gpu-admission") == 0;
    if (cross_gpu) {
      int devices       = 0;
      auto const status = cudaGetDeviceCount(&devices);
      if (status == cudaErrorNoDevice) return 77;
      cuda_check(status);
      if (devices < 2) return 77;
    }
    cuda_check(cudaSetDevice(0));
    rmm::cuda_stream owned_stream;
    ::cuda::stream_ref const stream{owned_stream};
    if (report_bytes || cross_gpu || (argc == 2 && std::strcmp(argv[1], "--admission") == 0)) {
      mixed_host_admission(stream, report_bytes, cross_gpu);
      return 0;
    }
    cucascade::memory::gpu_memory_space_config config;
    bool const decoded_like       = argc == 3 && std::strcmp(argv[1], "--decoded-like") == 0;
    config.per_stream_reservation = !framed && !decoded_like;
    config.device_id       = 0;
    config.memory_capacity = std::size_t{12} << 30;
    cucascade::memory::memory_space space{config};
    auto* allocator =
      space.get_memory_resource_as<cucascade::memory::reservation_aware_resource_adaptor>();
    auto reservation = space.make_reservation(large ? std::size_t{4} << 30 : std::size_t{64} << 20);
    expect_equal("decode allocator present", allocator != nullptr, true);
    expect_equal("decode reservation granted", bool(reservation), true);
    expect(allocator->attach_reservation_to_tracker(
             stream,
             std::move(reservation),
             std::make_unique<cucascade::memory::ignore_reservation_limit_policy>(),
             std::make_unique<cucascade::memory::throw_on_oom_policy>()),
           "attach decode reservation");
    struct reset_reservation {
      cucascade::memory::reservation_aware_resource_adaptor& allocator;
      ::cuda::stream_ref stream;
      ~reset_reservation()
      {
        stream.sync();
        allocator.reset_stream_reservation(stream);
      }
    } reset{*allocator, stream};
    cucascade::representation_converter_registry registry;
    sirius::register_compression_converters(registry);
    if (decoded_like) {
      if (std::string_view{argv[2]} != "capped") {
        all_survivor_like_offsets(space, stream);
        if (simpatico::decoded_like_enabled()) {
          decoded_like_membership(stream, false);
          decoded_like_membership(stream, true);
        }
      }
      decoded_like_cases(space, stream, argv[2]);
      decoded_like_host_admission(space, stream);
      return 0;
    }
    if (framed) {
      nullable_membership(space, stream, codec);
      if (simpatico::chunk_selective_enabled()) {
        framed_reservation(space, stream, codec);
        warmed_framed_reservation(space, stream, codec);
      }
      return 0;
    }
    if (large) {
      large_strings(copy, space, registry, stream, framed_large ? codec : "");
    } else {
      auto empty = make_strings_table({}, {}, stream);
      roundtrip(empty->view(), copy, space, registry, stream);
      auto nulls =
        make_strings_table({"alpha", "ignored", "", "beta"}, {true, false, true, true}, stream);
      roundtrip(nulls->view(), copy, space, registry, stream);
      auto all_null = make_strings_table({"ignored", "ignored"}, {false, false}, stream);
      roundtrip(all_null->view(), copy, space, registry, stream);
      scan_handoff(empty->view(), 1, copy, space, stream);
      scan_handoff(nulls->view(), 1, copy, space, stream);
      scan_handoff(all_null->view(), 1, copy, space, stream);
      auto mixed   = nulls->release();
      auto numbers = make_int32_table(1, 4, 42)->release();
      mixed.push_back(std::move(numbers[0]));
      cudf::table mixed_input{std::move(mixed)};
      roundtrip(mixed_input.view(), copy, space, registry, stream);
      guards(space, registry, stream);
      scan_eligibility_guards(space, registry, stream, copy);
      scan_handoff(mixed_input.view(), 2, copy, space, stream);
      // A decoded dictionary and numeric column travel with the raw strings; the last
      // numeric column is dropped by native projection. All retained buffers must survive.
      auto companions = make_strings_table({"a", "b", "a", "b"}, {}, stream)->release();
      auto projected  = mixed_input.release();
      auto numeric    = make_int32_table(2, 4, 7)->release();
      projected.push_back(std::move(companions[0]));
      projected.push_back(std::move(numeric[1]));
      cudf::table projection_input{std::move(projected)};
      scan_handoff(projection_input.view(),
                   3,
                   copy,
                   space,
                   stream,
                   "input -> identity\n---\ninput -> bitpack\n---\n"
                   "input -> dictionary\n---\ninput -> identity\n");
    }
    std::puts("test_string_decode_view: PASS");
    return 0;
  } catch (std::exception const& e) {
    std::fprintf(stderr, "test_string_decode_view: FAIL: %s\n", e.what());
    return 1;
  }
}
