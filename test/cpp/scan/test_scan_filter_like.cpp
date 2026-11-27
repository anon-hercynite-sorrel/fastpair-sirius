// SPDX-License-Identifier: Apache-2.0
#include <cuda_runtime_api.h>

#include <catch.hpp>
#include <compression/decompression_pushdown_policy.hpp>
#include <duckdb/planner/expression/bound_comparison_expression.hpp>
#include <duckdb/planner/expression/bound_conjunction_expression.hpp>
#include <duckdb/planner/expression/bound_constant_expression.hpp>
#include <duckdb/planner/expression/bound_function_expression.hpp>
#include <duckdb/planner/expression/bound_operator_expression.hpp>
#include <duckdb/planner/expression/bound_reference_expression.hpp>
#include <duckdb/planner/filter/expression_filter.hpp>
#include <op/scan/gpu_ingestible_types.hpp>
#include <op/scan/scan_execution_record.hpp>
#include <op/scan/scan_filter_analysis.hpp>
#include <util/decoded_like.hpp>
#include <utils/gpu_execution_fixture.hpp>
#include <utils/like_matcher_trace.hpp>
#include <utils/log_test_utils.hpp>
#include <utils/parquet_fixture_utils.hpp>
#include <utils/pinned_entry_census.hpp>

#include <array>
#include <fstream>

namespace {
using duckdb::Expression;
using sirius::op::classify_scan_like;

// Use the same bound function/reference shape DuckDB generic expression pushdown
// stores in an ExpressionFilter. No database or GPU is needed for this analysis.
duckdb::unique_ptr<Expression> function(char const* name, std::string pattern, bool escape = false)
{
  duckdb::vector<duckdb::unique_ptr<Expression>> arguments;
  arguments.push_back(
    duckdb::make_uniq<duckdb::BoundReferenceExpression>(duckdb::LogicalType::VARCHAR, 0));
  arguments.push_back(duckdb::make_uniq<duckdb::BoundConstantExpression>(duckdb::Value(pattern)));
  duckdb::vector<duckdb::LogicalType> types{duckdb::LogicalType::VARCHAR,
                                            duckdb::LogicalType::VARCHAR};
  if (escape) {
    arguments.push_back(duckdb::make_uniq<duckdb::BoundConstantExpression>(duckdb::Value("\\")));
    types.push_back(duckdb::LogicalType::VARCHAR);
  }
  return duckdb::make_uniq<duckdb::BoundFunctionExpression>(
    duckdb::BoundScalarFunction(
      duckdb::ScalarFunction(name, types, duckdb::LogicalType::BOOLEAN, nullptr)),
    std::move(arguments),
    nullptr);
}

duckdb::unique_ptr<Expression> negated(duckdb::unique_ptr<Expression> child)
{
  auto expression = duckdb::make_uniq<duckdb::BoundOperatorExpression>(
    duckdb::ExpressionType::OPERATOR_NOT, duckdb::LogicalType::BOOLEAN);
  expression->GetChildrenMutable().push_back(std::move(child));
  return expression;
}

struct fixture {
  duckdb::TableFilterSet filters;
  duckdb::vector<duckdb::ColumnIndex> ids{duckdb::ColumnIndex(0)};
  duckdb::vector<sirius::logical_type> types{sirius::logical_type::make(sirius::type_id::VARCHAR)};

  void add(duckdb::unique_ptr<Expression> expression)
  {
    filters.PushFilter(duckdb::ProjectionIndex(0),
                       duckdb::make_uniq<duckdb::ExpressionFilter>(std::move(expression)));
  }
  auto analyze() const { return sirius::op::analyze_scan_filters(filters, ids, types); }
};
}  // namespace

TEST_CASE("Scan LIKE classifier preserves exact pattern shape and inversion",
          "[scan][onpair_like][scan_like_analysis]")
{
  for (auto name : {"~~", "!~~", "contains"}) {
    auto expression = function(name, std::string(name) == "contains" ? "ab" : "%ab%cd%");
    auto pattern    = classify_scan_like(*expression);
    REQUIRE(pattern);
    REQUIRE(pattern->is_contains == (std::string_view{name} == "contains"));
    REQUIRE(pattern->invert == (std::string(name) == "!~~"));
    REQUIRE(pattern->literals == (std::string(name) == "contains"
                                    ? std::vector<std::string>{"ab"}
                                    : std::vector<std::string>{"ab", "cd"}));
    fixture f;
    f.add(std::move(expression));
    auto result = f.analyze();
    REQUIRE(result.like_sets.size() == (simpatico::scan_like_enabled() ? 1 : 0));
    REQUIRE(result.ranges_cover_whole_filter == simpatico::scan_like_enabled());
    REQUIRE(result.equality_sets.empty());  // projected columns qualify for LIKE
  }
  auto optimized_not_like = classify_scan_like(*negated(function("contains", "google")));
  REQUIRE(optimized_not_like);
  REQUIRE(optimized_not_like->is_contains);
  REQUIRE(optimized_not_like->invert);
  REQUIRE(optimized_not_like->literals == std::vector<std::string>{"google"});
  auto four = classify_scan_like(*function("~~", "%a%b%c%d%"));
  REQUIRE(four);
  REQUIRE(four->literals.size() == 4);
  REQUIRE(classify_scan_like(*function("~~", "%" + std::string(32, 'x') + "%")));
  for (auto name : {"like", "not_like"}) {
    auto parsed = classify_scan_like(*negated(negated(function(name, "%%a%%%b%%c%%"))));
    REQUIRE(parsed);
    CHECK(parsed->literals == std::vector<std::string>{"a", "b", "c"});
    CHECK(parsed->invert == (std::string_view{name} == "not_like"));
  }
}

TEST_CASE("Equivalent bound contains and LIKE conjuncts share one selection source",
          "[scan][scan_like_analysis]")
{
  fixture f;
  f.add(function("contains", "red"));
  f.add(function("~~", "%red%"));
  auto analysis = f.analyze();
  if (!simpatico::scan_like_enabled()) {
    REQUIRE(analysis.like_sets.empty());
    return;
  }
  REQUIRE(analysis.like_sets.at(0).size() == 1);
  CHECK(analysis.like_sets.at(0).front().is_contains);
  sirius::op::residual_filter residual{
    sirius::op::decompose_table_filters(f.filters, f.ids, f.types, {0}), {}, analysis.like_sets};
  std::array<std::size_t, 1> const enforced{0};
  auto progress = residual.progress_from_enforced_likes(enforced);
  REQUIRE(progress.size() == 2);
  CHECK(residual.against_progress(progress) == nullptr);
}

TEST_CASE("Scan LIKE refuses wildcards escapes and unsupported patterns",
          "[scan][onpair_like][scan_like_analysis]")
{
  for (auto text :
       {"%a_b%", "abc%", "%abc", "%", "%a%b%c%d%e%", "%abcdefghijklmnopqrstuvwxyz0123456%"}) {
    fixture f;
    auto expression = function("~~", text);
    REQUIRE_FALSE(classify_scan_like(*expression));
    f.add(std::move(expression));
    auto analysis = f.analyze();
    REQUIRE(analysis.like_sets.empty());
    REQUIRE_FALSE(analysis.ranges_cover_whole_filter);
  }
  REQUIRE_FALSE(classify_scan_like(*function("like_escape", "%a_b%", true)));
  fixture escaped;
  escaped.add(function("like_escape", "%a_b%", true));
  REQUIRE(escaped.analyze().like_sets.empty());
  REQUIRE_FALSE(escaped.analyze().ranges_cover_whole_filter);
  REQUIRE_FALSE(classify_scan_like(*function("contains", "a%b")));
  REQUIRE_FALSE(classify_scan_like(*function("contains", "a_b")));
  REQUIRE_FALSE(classify_scan_like(*function("prefix", "red")));
  REQUIRE_FALSE(classify_scan_like(*function("suffix", "red")));
  fixture ilike;
  ilike.add(function("~~*", "%Google%"));
  REQUIRE(ilike.analyze().like_sets.empty());
  REQUIRE_FALSE(ilike.analyze().ranges_cover_whole_filter);
}

TEST_CASE("LIKE under OR stays residual alongside an eligible AND sibling",
          "[scan][scan_like_analysis]")
{
  fixture f;
  auto either =
    duckdb::make_uniq<duckdb::BoundConjunctionExpression>(duckdb::ExpressionType::CONJUNCTION_OR);
  either->GetChildrenMutable().push_back(function("~~", "%blue%"));
  either->GetChildrenMutable().push_back(function("~~", "%green%"));
  f.add(std::move(either));
  f.add(function("~~", "%red%"));
  auto analysis = f.analyze();
  REQUIRE_FALSE(analysis.ranges_cover_whole_filter);
  if (simpatico::scan_like_enabled()) {
    REQUIRE(analysis.like_sets.at(0) == std::vector<simpatico::like_predicate>{{{"red"}, false}});
  } else {
    REQUIRE(analysis.like_sets.empty());
  }
  sirius::op::residual_filter residual{
    sirius::op::decompose_table_filters(f.filters, f.ids, f.types, {0}), {}, analysis.like_sets};
  std::array<std::size_t, 1> enforced{0};
  auto progress = simpatico::scan_like_enabled() ? residual.progress_from_enforced_likes(enforced)
                                                 : std::vector<sirius::op::predicate_progress>{};
  REQUIRE(progress.size() == (simpatico::scan_like_enabled() ? 1 : 0));
  REQUIRE(residual.against_progress(progress));
}

TEST_CASE_METHOD(sirius::test::GpuExecutionFixture,
                 "Pinned LIKE results agree with DuckDB across predicate shapes",
                 "[scan][integration][decoded_like_sql][decoded_like_trace]")
{
  sirius::test::scratch_dir files("decoded_like");
  auto const file = files.file_literal("strings.parquet");
  run_ok("SET gpu_execution=false");
  run_ok(
    "COPY (SELECT * FROM (VALUES (0, NULL::VARCHAR), (1,''), (2,'red blue'), "
    "(3,'r_e_d'), (4,'RED'), (5,'red'), (6,'red green'), (7,'red blue gold'), "
    "(8,'red blue gold sun'), (9,'blue red'), (10,'abcdefghijklmnopqrstuvwxyz123456'), "
    "(11,'blue')) "
    "t(id, s)) TO " +
    file + " (FORMAT PARQUET)");
  {
    std::ofstream plan(files.path() / "like_shapes.txt");
    plan << "input -> bitpack\n---\ninput -> str_split\n";
    REQUIRE(plan.good());
  }
  run_ok("SET pin_table_input_compression_plan_dir=" +
         sirius::test::sql_literal(files.path().string()));
  run_ok("SET pin_table_compression_min_batch_size_bytes=0");
  run_ok("SET pin_table_compression_max_compressed_fraction=1000");
  run_ok("CALL pin_table(" + file + ", tier='gpu', name='like_shapes', compression=true)");
  auto census = sirius::test::census_entry(*con, "like_shapes");
  REQUIRE(census.chunks > 0);
  REQUIRE(census.compressed_chunks == census.chunks);
  auto const select = "SELECT id, s FROM read_parquet(" + file + ") WHERE ";
  for (bool swar : {false, true}) {
    CAPTURE(swar);
    run_ok(swar ? "SET like_swar_fastpath=true" : "SET like_swar_fastpath=false");
    struct matcher_case {
      char const* predicate;
      char const* matcher;
      bool pushable;
    };
    for (auto const& tc :
         {matcher_case{"s LIKE '%red%'", "contains", true},
          matcher_case{"NOT contains(s, 'red')", "contains", true},
          matcher_case{"s LIKE '%red%blue%'", swar ? "swar" : "cudf_like", true},
          matcher_case{"s NOT LIKE '%red%blue%'", swar ? "swar" : "cudf_like", true},
          // NOT keeps DuckDB from replacing prefix with a lexicographic range.
          matcher_case{"NOT prefix(s, 'red')", "prefix", false},
          matcher_case{"suffix(s, 'red')", "suffix", false}}) {
      CAPTURE(tc.predicate, tc.matcher);
      sirius::test::scoped_recording_log_sink logs{"info"};
      {
        sirius::test::like_matcher_trace trace{tc.matcher};
        compare_gpu_vs_cpu(select + tc.predicate);
      }
      std::vector<std::string> matchers;
      for (auto const& record : logs.records()) {
        if (!record.message.starts_with("[decoded-like]")) continue;
        auto const start = record.message.find("matcher=");
        REQUIRE(start != std::string::npos);
        matchers.push_back(
          record.message.substr(start + 8, record.message.find(' ', start) - start - 8));
      }
      if (tc.pushable && simpatico::decoded_like_enabled())
        CHECK(matchers == std::vector<std::string>{tc.matcher});
      else
        CHECK(matchers.empty());
    }
    sirius::test::scoped_recording_log_sink logs{"info"};
    compare_gpu_vs_cpu(select + "s LIKE '%red%' AND (s LIKE '%blue%' OR s LIKE '%green%')");
    // Six rows contain red. The residual OR removes red alone, and the blue-only
    // row would leak through if the scan reported red as enforced without applying it.
    bool evaluated_red = false;
    bool applied_red   = false;
    std::vector<std::string> decisions;
    for (auto const& record : logs.records()) {
      auto const& msg = record.message;
      if (msg.starts_with("[decoded-like]")) {
        decisions.push_back(msg);
        evaluated_red |= msg.ends_with("matcher=contains rows=12 matches=6");
      }
      if (msg.starts_with("[decompression-pushdown] decode ")) {
        decisions.push_back(msg);
        applied_red |= msg.starts_with("[decompression-pushdown] decode APPLIED") &&
                       msg.find("likes=1 join_filters=0 survivors=6/12 ") != std::string::npos &&
                       msg.find("like_delivery=compact") != std::string::npos;
      }
    }
    CAPTURE(decisions);
    CHECK(evaluated_red == simpatico::decoded_like_enabled());
    CHECK(applied_red ==
          (simpatico::decoded_like_enabled() && sirius::decompression_pushdown_diag_enabled()));
  }
  run_ok("RESET like_swar_fastpath");
  for (auto where : {"s LIKE '%red%'",
                     "s NOT LIKE '%red%'",
                     "contains(s,'red')",
                     "NOT contains(s,'red')",
                     "s LIKE '%%red%%%blue%%'",
                     "s NOT LIKE '%%red%%%blue%%'",
                     "s LIKE '%red%blue%gold%'",
                     "s NOT LIKE '%red%blue%gold%'",
                     "s LIKE '%red%blue%gold%sun%'",
                     "s NOT LIKE '%red%blue%gold%sun%'",
                     "s LIKE '%abcdefghijklmnopqrstuvwxyz123456%'",
                     "s LIKE '%r_d%'"}) {
    CAPTURE(where);
    compare_gpu_vs_cpu(select + where);
  }
  // ESCAPE and ILIKE may fall back to DuckDB after scan classification refuses them.
  run_ok("SET enable_duckdb_fallback=true");
  for (auto where : {"s LIKE '%r!_e%' ESCAPE '!'", "s ILIKE '%red%'"}) {
    CAPTURE(where);
    run_ok("SET gpu_execution=true");
    auto result = con->Query(select + where);
    REQUIRE_FALSE(result->HasError());
    run_ok("SET gpu_execution=false");
    auto cpu = con->Query(select + where);
    REQUIRE_FALSE(cpu->HasError());
    REQUIRE(collect_rows(*result) == collect_rows(*cpu));
  }
  run_ok("CALL unpin_table('like_shapes')");
  REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
}

TEST_CASE("LIKE request maps projected slots and retains unrelated conjuncts",
          "[scan][onpair_like][scan_like_analysis]")
{
  fixture f;
  f.add(function("~~", "%ab%cd%"));
  f.add(function("contains", "other"));  // a second distinct conjunct on the same column
  auto analysis = f.analyze();
  if (!simpatico::scan_like_enabled()) {
    REQUIRE(analysis.like_sets.empty());
    return;
  }
  REQUIRE(analysis.like_sets.size() == 1);
  REQUIRE(analysis.ranges_cover_whole_filter);
  REQUIRE(analysis.like_sets.at(0).size() == 2);
  std::array<std::size_t, 1> slots{0};
  auto request = sirius::op::build_pushdown_request(analysis, slots, std::nullopt);
  REQUIRE(request.columns[0].likes == analysis.like_sets.at(0));
  REQUIRE(request.selects_rows());
  REQUIRE_FALSE(request.empty());
  REQUIRE(request.ranges_cover_whole_filter);

  auto catalogue = sirius::op::decompose_table_filters(f.filters, f.ids, f.types, {0});
  REQUIRE(catalogue.size() == 2);
  {
    sirius::op::residual_filter all{
      sirius::op::decompose_table_filters(f.filters, f.ids, f.types, {0}),
      {},
      {{0, analysis.like_sets.at(0)}}};
    std::array<std::size_t, 1> positions{0};
    auto progress = all.progress_from_enforced_likes(positions);
    REQUIRE(progress.size() == 2);
    REQUIRE(all.against_progress(progress) == nullptr);
    REQUIRE(sirius::op::build_pushdown_request(analysis, slots, {}, true).row_count_only);
  }
  // Partial offerings must still leave the other literal/inversion residual.
  sirius::op::residual_filter residual{
    std::move(catalogue), {}, {{0, {analysis.like_sets.at(0).front()}}}};
  std::array<std::size_t, 1> answered{0};
  auto progress = residual.progress_from_answers(answered, true);
  REQUIRE(progress.size() == 1);
  REQUIRE(progress[0].stage == sirius::op::predicate_stage::enforced);
  REQUIRE_FALSE(progress[0].answer_slot);  // string values remain valid for the OTHER conjunct
  REQUIRE(residual.against_progress(progress) != nullptr);
  REQUIRE_THROWS_AS(residual.progress_from_answers(answered, false), std::invalid_argument);

  auto without = sirius::decompression_pushdown_scan{request}.without_row_selection();
  REQUIRE(without == nullptr);  // mask-only LIKE cannot survive disabled row selection
}

TEST_CASE("All LIKE conjuncts cover the filter and drop precisely their residual",
          "[scan][onpair_like][scan_like_analysis]")
{
  if (!simpatico::scan_like_enabled()) return;
  fixture f;
  f.add(negated(function("contains", "google")));
  auto analysis = f.analyze();
  REQUIRE(analysis.ranges_cover_whole_filter);
  std::array<std::size_t, 1> slots{0};
  auto request = sirius::op::build_pushdown_request(analysis, slots, std::nullopt);
  REQUIRE(request.ranges_cover_whole_filter);
  auto catalogue = sirius::op::decompose_table_filters(f.filters, f.ids, f.types, {0});
  sirius::op::residual_filter residual{std::move(catalogue), {}, {{0, analysis.like_sets.at(0)}}};
  REQUIRE(residual.against({0}, true) == nullptr);
  std::array<std::size_t, 1> enforced{0};
  auto record = sirius::op::scan::translate_legacy_execution(
    {.state = sirius::op::scan::filter_state::UNFILTERED, .enforced_like_columns = enforced},
    residual,
    1,
    1);
  REQUIRE(record.predicates.size() == 1);
  REQUIRE(record.predicates[0].stage == sirius::op::predicate_stage::enforced);
  REQUIRE_FALSE(record.predicates[0].answer_slot);
  REQUIRE(residual.against_progress(record.predicates) == nullptr);
  auto count = sirius::op::build_pushdown_request(analysis, slots, {}, true);
  REQUIRE(count.row_count_only);
  REQUIRE_FALSE(request.row_count_only);
  REQUIRE(residual.against({}, false) != nullptr);
  slots[0] = 99;
  REQUIRE(sirius::op::build_pushdown_request(analysis, slots, {}).empty());
}

TEST_CASE("Ranges and LIKE jointly cover the whole scan filter",
          "[scan][onpair_like][scan_like_analysis]")
{
  if (!simpatico::scan_like_enabled()) return;
  fixture f;
  f.add(function("~~", "%ab%cd%"));
  f.ids.emplace_back(1);
  f.types.push_back(sirius::logical_type::make(sirius::type_id::INTEGER));
  auto reference =
    duckdb::make_uniq<duckdb::BoundReferenceExpression>(duckdb::LogicalType::INTEGER, 0);
  auto constant = duckdb::make_uniq<duckdb::BoundConstantExpression>(duckdb::Value::INTEGER(5));
  f.filters.PushFilter(
    duckdb::ProjectionIndex(1),
    duckdb::make_uniq<duckdb::ExpressionFilter>(duckdb::BoundComparisonExpression::Create(
      duckdb::ExpressionType::COMPARE_GREATERTHANOREQUALTO,
      std::move(reference),
      std::move(constant))));
  auto analysis = f.analyze();
  REQUIRE(analysis.like_sets.size() == 1);
  REQUIRE(analysis.ranges.size() == 1);
  REQUIRE(analysis.ranges_cover_whole_filter);
  std::array<std::size_t, 2> slots{0, 1};
  auto request = sirius::op::build_pushdown_request(analysis, slots, 1);
  REQUIRE_FALSE(request.columns[0].likes.empty());
  REQUIRE(request.columns[1].range);
  REQUIRE(request.ranges_cover_whole_filter);
  REQUIRE(request.output_prefix_width == 1);
}

TEST_CASE("Filter-only LIKE BOOL8 answers remain residual until their mask is applied",
          "[scan][onpair_like][fused_scan_filter][delivery]")
{
  if (!simpatico::scan_like_enabled()) return;
  fixture f;
  f.add(negated(function("~~", "%special%requests%")));
  f.add(function("contains", "ordinary"));
  auto analysis = f.analyze();
  REQUIRE(analysis.ranges_cover_whole_filter);
  sirius::op::residual_filter filter_only{
    sirius::op::decompose_table_filters(f.filters, f.ids, f.types, {0}),
    {0},  // independently proved filter-only; both conjuncts may read the same combined BOOL8
    {{0, analysis.like_sets.at(0)}}};
  std::array<std::size_t, 1> answered{0};
  auto record = sirius::op::scan::translate_legacy_execution(
    {.state            = sirius::op::scan::filter_state::UNFILTERED,
     .answer_columns   = answered,
     .answers_enforced = false},
    filter_only,
    1,
    1);
  REQUIRE_FALSE(record.whole_static_complete);
  REQUIRE(record.predicates.size() == 2);
  for (auto const& progress : record.predicates) {
    REQUIRE(progress.stage == sirius::op::predicate_stage::answer);
    REQUIRE(progress.answer_slot == 0);
  }
  REQUIRE(filter_only.against_progress(record.predicates) != nullptr);
  REQUIRE(filter_only.against({0}, true) == nullptr);
  sirius::op::residual_filter projected{
    sirius::op::decompose_table_filters(f.filters, f.ids, f.types, {0}),
    {},
    {{0, analysis.like_sets.at(0)}}};
  REQUIRE_THROWS_AS(projected.progress_from_answers(answered, false), std::invalid_argument);
  REQUIRE_THROWS_AS(projected.against_progress(record.predicates), std::invalid_argument);
  REQUIRE(projected.against({0}, true) == nullptr);
}
