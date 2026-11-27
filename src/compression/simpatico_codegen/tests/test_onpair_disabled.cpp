// SPDX-License-Identifier: Apache-2.0
#include "api/compressed_table_io.hpp"
#include "api/simpatico_codegen.hpp"
#include "codegen/plan/plan_tree.hpp"
#include "decode_session_test_access.hpp"
#include "test_utils.hpp"

#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

int main()
{
  try {
    auto const stream = cudf::get_default_stream();
    auto const mr     = rmm::mr::get_current_device_resource_ref();
    for (std::string const op : {"onpair", "onpair_9", "onpair_16"}) {
      expect(simpatico::make_compressor(op) == nullptr, "onpair OFF: no encoder");
      std::vector<std::unique_ptr<cudf::column>> columns;
      columns.push_back(cudf::make_empty_column(cudf::data_type{cudf::type_id::STRING}));
      cudf::table input(std::move(columns));
      std::string error;
      try {
        (void)simpatico::compress_with_plan(input.view(), "input -> " + op + "\n", stream, mr);
      } catch (std::exception const& failure) {
        error = failure.what();
      }
      expect(error.find("SIMPATICO_ONPAIR is OFF") != std::string::npos,
             ("onpair OFF: clear plan rejection; got: " + error).c_str());

      // A structural v17 fixture requires no encoder and carries no device payload.
      // The reader must reject the node before attempting any fetch/allocation.
      simpatico::compressed_table fixture;
      fixture.columns.resize(1);
      auto& column = fixture.columns[0];
      column.dtype = cudf::data_type{cudf::type_id::STRING};
      auto tree    = simpatico::plan_tree_from_dsl("input -> " + op + "\n", &error);
      expect(tree.has_value(), "onpair OFF: fixture plan structure");
      column.plan_tree                = std::make_unique<simpatico::PlanTree>(std::move(*tree));
      column.plan_tree->nodes[1].meta = simpatico::leaf_meta::onpair{0, 0, 16};
      std::vector<std::uint8_t> header;
      std::vector<simpatico::payload_buffer_ref> buffers;
      std::uint64_t bytes = 0;
      error = simpatico::build_compressed_table_header(fixture, header, buffers, bytes, stream);
      expect(error.empty(), error.c_str());
      auto fetch = [](std::uint64_t, std::size_t, void*, ::cuda::stream_ref) {
        throw std::runtime_error("onpair OFF: must reject before payload fetch");
      };
      auto rejected =
        simpatico::read_compressed_table_from_memory(header, fetch, stream, mr, &error);
      expect(rejected.columns.empty() && error.find("SIMPATICO_ONPAIR is OFF") != std::string::npos,
             "onpair OFF: clear file rejection");
      expect(throws<std::invalid_argument>([&] {
               simpatico::PlanNode node;
               node.op = op;
               simpatico::decode_session session{{&stream, 1}, mr};
               auto& frame = simpatico::decode_session_test_access::frame(session);
               (void)simpatico::reconstruct_decode_representation(node, {}, {}, frame);
             }),
             "onpair OFF: no decode reconstruction");
    }
    std::puts("test_onpair_disabled: PASS");
    return 0;
  } catch (std::exception const& error) {
    std::fprintf(stderr, "test_onpair_disabled: FAIL: %s\n", error.what());
    return 1;
  }
}
