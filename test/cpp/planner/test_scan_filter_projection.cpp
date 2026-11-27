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

#include "planner/sirius_physical_plan_generator.hpp"

#include <catch.hpp>
#include <duckdb.hpp>  // IWYU pragma: keep
#include <duckdb/common/multi_file/multi_file_reader.hpp>
#include <duckdb/common/multi_file/multi_file_states.hpp>
#include <duckdb/execution/expression_executor.hpp>
#include <duckdb/execution/column_binding_resolver.hpp>
#include <duckdb/planner/expression/bound_columnref_expression.hpp>
#include <duckdb/planner/expression/bound_comparison_expression.hpp>
#include <duckdb/planner/expression/bound_constant_expression.hpp>
#include <duckdb/planner/operator/logical_aggregate.hpp>
#include <duckdb/planner/operator/logical_filter.hpp>
#include <duckdb/planner/operator/logical_get.hpp>
#include <op/scan/scan_filter_analysis.hpp>
#include <op/scan/scan_plan.hpp>
#include <op/sirius_physical_table_scan.hpp>
#include <planner/connector_registry.hpp>
#include <utils/parquet_fixture_utils.hpp>
#include <utils/sirius_test_env.hpp>
#include <utils/table_filter_test_utils.hpp>

#include <cstddef>
#include <filesystem>
#include <string>
#include <unordered_set>
#include <vector>

namespace {
duckdb::LogicalGet* find_logical_get(duckdb::LogicalOperator& root)
{
  if (root.type == duckdb::LogicalOperatorType::LOGICAL_GET)
    return &root.Cast<duckdb::LogicalGet>();
  for (auto& child : root.children)
    if (auto* get = find_logical_get(*child)) return get;
  return nullptr;
}
}  // namespace

TEST_CASE("unpushed scan filters follow the emitted projection",
          "[planner][scan_filter_projection]")
{
  sirius::test::scoped_sirius_disable disable;
  duckdb::DuckDB db(nullptr);
  duckdb::Connection con(db);
  duckdb::TableFunction function;
  function.supports_pushdown_type = [](duckdb::FunctionData const&, duckdb::idx_t primary) {
    return primary != 2;
  };
  duckdb::LogicalGet scan(
    duckdb::TableIndex(0),
    duckdb::BoundTableFunction(function),
    duckdb::make_uniq<duckdb::TableFunctionData>(),
    {duckdb::LogicalType::INTEGER, duckdb::LogicalType::INTEGER, duckdb::LogicalType::INTEGER},
    {"a", "b", "c"});
  // Scan slots [c, a, b]; explicit projection emits [b, c].
  scan.AddColumnId(2);
  scan.AddColumnId(0);
  scan.AddColumnId(1);
  auto const reordered = GENERATE(false, true);
  if (reordered) { scan.projection_ids = {duckdb::ProjectionIndex(2), duckdb::ProjectionIndex(0)}; }

  duckdb::TableFilterSet filters;
  filters.PushFilter(duckdb::ProjectionIndex(0),
                     sirius::test::constant_filter(duckdb::ExpressionType::COMPARE_EQUAL,
                                                   duckdb::Value::INTEGER(10)));
  filters.PushFilter(duckdb::ProjectionIndex(2),
                     sirius::test::constant_filter(duckdb::ExpressionType::COMPARE_GREATERTHAN,
                                                   duckdb::Value::INTEGER(0)));
  auto predicates = sirius::planner::extract_unpushed_scan_filters(scan, filters);
  REQUIRE(predicates.size() == 1);
  REQUIRE(filters.FilterCount() == 1);
  REQUIRE(filters.HasFilter(duckdb::ProjectionIndex(2)));

  duckdb::idx_t const width = reordered ? 2 : 3;
  duckdb::idx_t const c     = reordered ? 1 : 0;
  duckdb::DataChunk input;
  input.Initialize(duckdb::Allocator::DefaultAllocator(),
                   duckdb::vector<duckdb::LogicalType>(width, duckdb::LogicalType::INTEGER));
  input.SetChildCardinality(3);
  for (duckdb::idx_t col = 0; col < width; ++col) {
    input.data[col].SetValue(0, duckdb::Value::INTEGER(11));
    input.data[col].SetValue(1, duckdb::Value::INTEGER(10));
    input.data[col].SetValue(2, duckdb::Value::INTEGER(99));
  }
  input.data[c].SetValue(0, duckdb::Value::INTEGER(10));
  input.data[c].SetValue(1, duckdb::Value::INTEGER(20));
  input.data[c].SetValue(2, duckdb::Value(duckdb::LogicalType::INTEGER));
  duckdb::DataChunk result;
  result.Initialize(duckdb::Allocator::DefaultAllocator(), {duckdb::LogicalType::BOOLEAN});
  duckdb::ExpressionExecutor executor(*con.context, predicates);
  executor.Execute(input, result);
  REQUIRE(result.size() == 3);
  CHECK(result.GetValue(0, 0) == duckdb::Value::BOOLEAN(true));
  CHECK(result.GetValue(0, 1) == duckdb::Value::BOOLEAN(false));
  CHECK(result.GetValue(0, 2).IsNull());
}

TEST_CASE("DuckDB count-only and projected LIKE retain distinct projection fields",
          "[planner][scan][onpair_like][scan_plan][roles]")
{
  // CPU-only optimizer regression: inspect scan columns, projections, types and bindings
  // separately.
  sirius::test::scoped_sirius_disable disable;
  sirius::test::scratch_dir scratch{"count_like_projection"};
  duckdb::DuckDB db(nullptr);
  duckdb::Connection con(db);
  auto written = con.Query(
    "COPY (SELECT * FROM (VALUES ('google'), ('other'), ('google-x')) "
    "AS t(URL)) TO " +
    scratch.file_literal("hits.parquet") + " (FORMAT PARQUET)");
  REQUIRE(written);
  REQUIRE_FALSE(written->HasError());
  auto const counted = GENERATE(false, true);
  auto logical = con.ExtractPlan(std::string("SELECT ") + (counted ? "COUNT(*)" : "URL") +
                                 " FROM read_parquet(" + scratch.file_literal("hits.parquet") +
                                 ") WHERE URL LIKE '%google%'");
  REQUIRE(logical);
  logical->ResolveOperatorTypes();
  auto* get = find_logical_get(*logical);
  REQUIRE(get);
  REQUIRE(get->function.projection_pushdown);
  REQUIRE(get->function.filter_prune);
  REQUIRE(get->GetColumnIds().size() == 1);
  CHECK(get->GetColumnIds()[0].GetPrimaryIndex() == 0);
  if (counted) {
    CHECK(get->projection_ids.empty());
  } else {
    REQUIRE(get->projection_ids.size() == 1);
    CHECK(get->projection_ids[0].GetIndex() == 0);
  }
  REQUIRE(get->types.size() == 1);
  CHECK(get->types[0] == duckdb::LogicalType::VARCHAR);
  CHECK(get->GetColumnBindings().size() == 1);
  REQUIRE(get->table_filters.FilterCount() == 1);
  duckdb::vector<sirius::logical_type> types{sirius::logical_type::make(sirius::type_id::VARCHAR)};
  auto conjuncts =
    sirius::op::decompose_table_filters(get->table_filters, get->GetColumnIds(), types, {0});
  REQUIRE(conjuncts.size() == 1);
  auto like = sirius::op::classify_scan_like(*conjuncts[0].expr);
  REQUIRE(like);
  CHECK(like->literals == std::vector<std::string>{"google"});
  CHECK_FALSE(like->invert);

  namespace scan = sirius::op::scan;
  // The scan-local contract still interprets empty projection_ids as all columns. Only an
  // aggregate-aware Sirius planner can prove the count-only child needs no value outputs.
  duckdb::vector<duckdb::idx_t> projections;
  for (auto index : get->projection_ids)
    projections.push_back(index.GetIndex());
  auto plan =
    scan::build_scan_plan(get->GetColumnIds(), projections, {"URL"}, types, get->types.size(), {});
  REQUIRE(plan.data_columns.size() == 1);
  CHECK(plan.columns.data_roles == std::vector<scan::scan_column_layout::data_role>{
                                     scan::scan_column_layout::data_role::OUTPUT});
  REQUIRE(plan.output_layout.size() == 1);
  CHECK(plan.output_layout[0].source == scan::scan_plan::output_entry::DATA);
  CHECK(plan.output_layout[0].idx == 0);
  CHECK_FALSE(scan::allows_row_count_carrier(plan));

  // The carrier already works when a caller can prove zero outputs; this is the information
  // DuckDB's scan-local empty-projection sentinel does not express.
  auto filter_only = scan::build_scan_plan(get->GetColumnIds(), {0}, {"URL"}, types, 0, {});
  CHECK(filter_only.output_layout.empty());
  CHECK(filter_only.filter_only_batch_positions() == std::unordered_set<std::size_t>{0});
  CHECK(scan::allows_row_count_carrier(filter_only));
}

namespace {
class count_scan_planner : public sirius::planner::sirius_physical_plan_generator {
 public:
  using sirius_physical_plan_generator::sirius_physical_plan_generator;
  auto physical_tree(duckdb::unique_ptr<duckdb::LogicalOperator> logical)
  {
    logical->ResolveOperatorTypes();
    duckdb::ColumnBindingResolver resolver;
    resolver.VisitOperator(*logical);
    return create_plan(*logical);
  }
};
// Sirius physical planning binds catalog state through ClientContext::ActiveTransaction, which
// con.ExtractPlan leaves closed. Hold one for the planner and discard it afterwards.
struct scoped_transaction {
  duckdb::Connection& con;
  explicit scoped_transaction(duckdb::Connection& connection) : con(connection)
  {
    con.BeginTransaction();
  }
  ~scoped_transaction() { con.Rollback(); }
};
sirius::op::sirius_physical_table_scan* find_physical_scan(
  sirius::op::sirius_physical_operator& root)
{
  if (root.type == sirius::op::SiriusPhysicalOperatorType::TABLE_SCAN)
    return &root.Cast<sirius::op::sirius_physical_table_scan>();
  for (auto& child : root.children)
    if (auto* scan = find_physical_scan(*child)) return scan;
  return nullptr;
}
}  // namespace

TEST_CASE("Sirius plain COUNT over pushed LIKE plans a filter-only Parquet scan",
          "[planner][scan][onpair_like][count_carrier]")
{
  sirius::test::scoped_sirius_disable disable;
  sirius::test::scratch_dir scratch{"sirius_count_like"};
  duckdb::DuckDB db(nullptr);
  duckdb::Connection con(db);
  auto written = con.Query(
    "COPY (SELECT * FROM (VALUES ('google'), ('other'), ('google-x')) "
    "AS t(URL)) TO " +
    scratch.file_literal("hits.parquet") + " (FORMAT PARQUET)");
  REQUIRE(written);
  REQUIRE_FALSE(written->HasError());
  std::string suffix;
  bool unpushed_filter     = false;
  std::string projection   = "COUNT(*)";
  bool carrier             = true;
  bool explicit_projection = false;
  SECTION("Q20 count has no value consumer") {}
  SECTION("HAVING consumes the COUNT result") { suffix = " HAVING COUNT(*) > 0"; }
  SECTION("projection above aggregate consumes only the COUNT result")
  { projection = "COUNT(*) + 1"; }
  SECTION("DISTINCT consumes the inner COUNT result") { projection = "DISTINCT COUNT(*)"; }
  SECTION("COUNT DISTINCT reads values")
  {
    projection = "COUNT(DISTINCT URL)";
    carrier    = false;
  }
  SECTION("unpushed filter between aggregate and GET reads values")
  {
    unpushed_filter = true;
    carrier         = false;
  }
  SECTION("projected URL retains its output")
  {
    projection = "URL";
    carrier    = false;
  }
  SECTION("another aggregate consumes the URL")
  {
    projection = "COUNT(*), MAX(URL)";
    carrier    = false;
  }
  SECTION("COUNT with an explicit projected column is unchanged")
  {
    explicit_projection = true;
    carrier             = false;
  }
  auto logical =
    con.ExtractPlan("SELECT " + projection + " FROM read_parquet(" +
                    scratch.file_literal("hits.parquet") + ") WHERE URL LIKE '%google%'" + suffix);
  REQUIRE(logical);
  if (explicit_projection) {
    auto* get = find_logical_get(*logical);
    REQUIRE(get);
    get->projection_ids = {duckdb::ProjectionIndex(0)};
  }
  if (unpushed_filter) {
    // Insert after optimization, so this exercises the direct-GET exclusion with a retained
    // value-reading residual even though the original LIKE remains pushed into the scan.
    auto insert = [&](auto&& self, duckdb::LogicalOperator& root) -> bool {
      if (root.type == duckdb::LogicalOperatorType::LOGICAL_AGGREGATE_AND_GROUP_BY &&
          root.children.size() == 1 &&
          root.children[0]->type == duckdb::LogicalOperatorType::LOGICAL_GET) {
        auto& get = root.children[0]->Cast<duckdb::LogicalGet>();
        REQUIRE(get.projection_ids.empty());
        auto predicate = duckdb::BoundComparisonExpression::Create(
          duckdb::ExpressionType::COMPARE_NOTEQUAL,
          duckdb::make_uniq<duckdb::BoundColumnRefExpression>(
            duckdb::LogicalType::VARCHAR,
            duckdb::ColumnBinding(get.table_index, duckdb::ProjectionIndex(0))),
          duckdb::make_uniq<duckdb::BoundConstantExpression>(duckdb::Value("google-x")));
        auto filter = duckdb::make_uniq<duckdb::LogicalFilter>(std::move(predicate));
        filter->children.push_back(std::move(root.children[0]));
        root.children[0] = std::move(filter);
        return true;
      }
      for (auto& child : root.children)
        if (self(self, *child)) return true;
      return false;
    };
    REQUIRE(insert(insert, *logical));
  }
  scoped_transaction transaction{con};
  count_scan_planner planner(*con.context);
  auto physical = planner.physical_tree(std::move(logical));
  REQUIRE(physical);
  auto* scan = find_physical_scan(*physical);
  REQUIRE(scan);
  namespace ss = sirius::op::scan;
  auto layout  = ss::build_scan_plan(scan->column_ids,
                                     scan->projection_ids,
                                     scan->names,
                                     scan->returned_types,
                                     scan->projected_output_count,
                                     {});
  REQUIRE(layout.data_columns.size() == 1);
  CHECK(layout.columns.data_roles == std::vector<ss::scan_column_layout::data_role>{
                                       carrier ? ss::scan_column_layout::data_role::FILTER_ONLY
                                               : ss::scan_column_layout::data_role::OUTPUT});
  CHECK(scan->projected_output_count == (carrier ? 0 : 1));
  CHECK(scan->types.size() == (carrier ? 0 : 1));
  CHECK(layout.output_layout.size() == (carrier ? 0 : 1));
  CHECK(ss::allows_row_count_carrier(layout) == carrier);
  if (!carrier) {
    CHECK(layout.output_layout[0].source == ss::scan_plan::output_entry::DATA);
    CHECK(layout.output_layout[0].idx == 0);
  }
}

TEST_CASE("Sirius COUNT excludes legacy virtual bindings, requested virtuals and Hive partitions",
          "[planner][scan][onpair_like][count_carrier]")
{
  sirius::test::scoped_sirius_disable disable;
  sirius::test::scratch_dir scratch{"count_virtuals"};
  duckdb::DuckDB db(nullptr);
  duckdb::Connection con(db);
  std::filesystem::create_directories(scratch.path() / "part=one");
  auto written =
    con.Query("COPY (SELECT * FROM (VALUES ('google'), ('other'), ('google-x')) AS t(URL)) TO " +
              scratch.file_literal("part=one/hits.parquet") + " (FORMAT PARQUET)");
  REQUIRE(written);
  REQUIRE_FALSE(written->HasError());
  std::string options = ", hive_partitioning=false";
  enum class binding { filename, row_number, modern, hive };
  binding kind = binding::filename;
  SECTION("legacy filename binding and retained predicate")
  { options += ", filename='source_path'"; }
  SECTION("legacy row-number binding and retained predicate")
  {
    options += ", file_row_number=true";
    kind = binding::row_number;
  }
  SECTION("requested modern virtual column") { kind = binding::modern; }
  SECTION("Hive partitions even when only URL is requested")
  {
    kind    = binding::hive;
    options = ", hive_partitioning=true";
  }
  auto logical = con.ExtractPlan("SELECT COUNT(*) FROM read_parquet(" +
                                 scratch.file_literal("part=one/hits.parquet") + options +
                                 ") WHERE URL LIKE '%google%'");
  REQUIRE(logical);
  auto* get = find_logical_get(*logical);
  REQUIRE(get);
  REQUIRE(get->projection_ids.empty());
  REQUIRE(get->table_filters.FilterCount() > 0);
  auto const* bind = dynamic_cast<duckdb::MultiFileBindData const*>(get->bind_data.get());
  REQUIRE(bind);
  if (kind != binding::hive) {
    duckdb::column_t primary;
    if (kind == binding::filename) {
      REQUIRE(bind->reader_bind.filename_idx.IsValid());
      primary = bind->reader_bind.filename_idx.GetIndex();
      REQUIRE(primary < get->returned_types.size());
      CHECK(get->names[primary] == "source_path");
    } else if (kind == binding::row_number) {
      REQUIRE(bind->reader_bind.file_row_number_idx.IsValid());
      primary = bind->reader_bind.file_row_number_idx.GetIndex();
      REQUIRE(primary < get->returned_types.size());
      CHECK(get->names[primary] == "file_row_number");
    } else {
      primary = duckdb::MultiFileReader::COLUMN_IDENTIFIER_FILE_INDEX;
      REQUIRE(get->virtual_columns.find(primary) != get->virtual_columns.end());
    }
    // Keep a real bound virtual predicate after optimizer pruning. Legacy indices are ordinary
    // ColumnIndex values: neither IsVirtualColumn nor the virtual_columns map can catch them.
    auto const slot = get->GetColumnIds().size();
    get->AddColumnId(primary);
    if (kind != binding::modern) {
      CHECK_FALSE(get->GetColumnIds().back().IsVirtualColumn());
      CHECK(get->virtual_columns.find(primary) == get->virtual_columns.end());
    }
    auto value = kind == binding::filename ? duckdb::Value(scratch.file("part=one/hits.parquet"))
                 : kind == binding::row_number ? duckdb::Value::BIGINT(0)
                                               : duckdb::Value::UBIGINT(0);
    get->table_filters.PushFilter(
      duckdb::ProjectionIndex(slot),
      sirius::test::constant_filter(duckdb::ExpressionType::COMPARE_EQUAL, std::move(value)));
    get->ResolveOperatorTypes();
  } else {
    REQUIRE_FALSE(bind->reader_bind.hive_partitioning_indexes.empty());
  }
  scoped_transaction transaction{con};
  count_scan_planner planner(*con.context);
  auto physical = planner.physical_tree(std::move(logical));
  REQUIRE(physical);
  auto* scan = find_physical_scan(*physical);
  REQUIRE(scan);
  CHECK(scan->projected_output_count > 0);
  CHECK_FALSE(scan->types.empty());
  // No count rewrite may turn the DuckDB all-columns sentinel into a zero-output projection.
  CHECK(scan->projection_ids.empty());
}

TEST_CASE("Sirius COUNT leaves native provider value outputs intact",
          "[planner][scan][count_carrier]")
{
  sirius::test::scoped_sirius_disable disable;
  sirius::test::scratch_dir scratch{"count_native"};
  auto path = scratch.file("native.duckdb");
  duckdb::DuckDB db(path.c_str());
  duckdb::Connection con(db);
  REQUIRE_FALSE(
    con
      .Query("CREATE TABLE count_native AS SELECT * FROM (VALUES ('google'), ('other')) AS t(URL)")
      ->HasError());
  REQUIRE_FALSE(con.Query("CHECKPOINT")->HasError());
  auto logical = con.ExtractPlan("SELECT COUNT(*) FROM count_native WHERE URL LIKE '%google%'");
  REQUIRE(logical);
  auto* get = find_logical_get(*logical);
  REQUIRE(get);
  REQUIRE(get->function.GetName() == "seq_scan");
  REQUIRE(get->projection_ids.empty());
  REQUIRE(get->table_filters.FilterCount() > 0);
  auto const* source = sirius::planner::lookup_connector(*get, *con.context);
  REQUIRE(source);
  REQUIRE(source->lower == sirius::planner::lower_native_scan);
  scoped_transaction transaction{con};
  count_scan_planner planner(*con.context);
  auto physical = planner.physical_tree(std::move(logical));
  REQUIRE(physical);
  auto* scan = find_physical_scan(*physical);
  REQUIRE(scan);
  CHECK(scan->projected_output_count > 0);
  CHECK_FALSE(scan->types.empty());
}

TEST_CASE("Sirius COUNT leaves Iceberg provider value outputs intact",
          "[planner][scan][count_carrier][shared_context]")
{
  REQUIRE(sirius::test::g_shared_env);
  auto con = sirius::test::g_shared_env->make_connection();
  REQUIRE_FALSE(con.Query("SET gpu_execution=false")->HasError());
  REQUIRE_FALSE(con.Query("LOAD iceberg")->HasError());
  REQUIRE(std::filesystem::exists("test/cpp/integration/data/iceberg_v1/metadata"));
  auto const path = std::string(SIRIUS_PROJECT_ROOT) + "/test/cpp/integration/data/iceberg_v1";
  auto logical =
    con.ExtractPlan("SELECT COUNT(*) FROM iceberg_scan(" + sirius::test::sql_literal(path) +
                    ", snapshot_from_id=1000000000000000001) WHERE fruit LIKE '%a%'");
  REQUIRE(logical);
  auto* get = find_logical_get(*logical);
  REQUIRE(get);
  REQUIRE(get->function.GetName() == "iceberg_scan");
  REQUIRE(get->projection_ids.empty());
  REQUIRE(get->table_filters.FilterCount() > 0);
  auto const* source = sirius::planner::lookup_connector(*get, *con.context);
  REQUIRE(source);
  REQUIRE(source->lower == sirius::planner::lower_iceberg_scan);
  scoped_transaction transaction{con};
  count_scan_planner planner(*con.context);
  auto physical = planner.physical_tree(std::move(logical));
  REQUIRE(physical);
  auto* scan = find_physical_scan(*physical);
  REQUIRE(scan);
  CHECK(scan->projected_output_count > 0);
  CHECK_FALSE(scan->types.empty());
}
