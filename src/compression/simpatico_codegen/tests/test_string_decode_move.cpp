// SPDX-License-Identifier: Apache-2.0
#include "api/compressed_leaf_slab.hpp"
#include "api/compressed_table_io.hpp"
#include "api/simpatico_codegen.hpp"
#include "decode_session_test_access.hpp"
#include "test_utils.hpp"
#include "util/string_decode.hpp"

#include <rmm/cuda_stream.hpp>
#include <rmm/mr/cuda_async_memory_resource.hpp>

#include <array>
#include <cstdio>
#include <cstring>
#include <string>

namespace {

void snappy_strings(bool copy, ::cuda::stream_ref stream, rmm::device_async_resource_ref mr)
{
  auto input = make_strings_table(
    {"alpha", "", "beta", "alpha", "ignored"}, {true, true, true, true, false}, stream);
  auto compressed = simpatico::compress_with_plan(input->view(),
                                                  "input -> str_split -> offsets, chars\n"
                                                  "str_split.offsets -> delta -> differences\n"
                                                  "str_split.chars -> snappy\n",
                                                  stream,
                                                  mr);
  // A sole-consumer Snappy channel must reach the STRING parent without another allocation.
  // Repeat against the same stored plan to ensure consuming the rebuilt rep preserves the pin.
  for (int i = 0; i < 2; ++i) {
    std::array streams{stream};
    simpatico::decode_session session{streams, mr};
    auto& frame = simpatico::decode_session_test_access::frame(session);
    auto output = simpatico::decode_request(
      simpatico::column_decode_request{std::cref(*compressed.columns[0].plan_tree)}, frame);
    auto const* unsnap = simpatico::decode_session_test_access::snappy_output(frame);
    expect(unsnap != nullptr, "Snappy did not expose its output buffer");
    expect((output->view().head<void>() == unsnap) == !copy, "Snappy chars route pointer");
    (void)session.finish();
    expect(columns_equal_any(input->view().column(0), output->view(), stream),
           "Snappy strings values/nulls");
  }
}

void uploaded_strings(bool copy, ::cuda::stream_ref stream, rmm::device_async_resource_ref mr)
{
  auto input = make_strings_table({"alpha", "ignored", "beta"}, {true, false, true}, stream);
  stream.sync();
  rmm::mr::cuda_async_memory_resource upload_resource;
  auto const upload_mr = rmm::device_async_resource_ref{upload_resource};
  std::unique_ptr<cudf::column> output;
  {
    rmm::cuda_stream upload_stream(rmm::cuda_stream::flags::non_blocking);
    auto encoded =
      simpatico::compress_with_plan(input->view(), "input -> identity\n", upload_stream, upload_mr);
    std::vector<std::uint8_t> header;
    std::vector<simpatico::payload_buffer_ref> buffers;
    std::uint64_t bytes = 0;
    auto error =
      simpatico::build_compressed_table_header(encoded, header, buffers, bytes, upload_stream);
    expect(error.empty(), error.c_str());
    std::vector<std::uint8_t> host(bytes);
    for (auto const& b : buffers)
      cuda_check(cudaMemcpyAsync(host.data() + b.offset,
                                 b.device_ptr,
                                 b.size_bytes,
                                 cudaMemcpyDeviceToHost,
                                 upload_stream.value()));
    upload_stream.synchronize();
    simpatico::compressed_leaf_slab slab;
    auto compressed = simpatico::read_uploaded_compressed_table(
      header,
      [&](std::uint64_t offset, std::size_t size, void* dst, cuda::stream_ref lane) {
        cuda_check(
          cudaMemcpyAsync(dst, host.data() + offset, size, cudaMemcpyHostToDevice, lane.get()));
      },
      std::nullopt,
      upload_stream,
      upload_mr,
      slab,
      &error);
    expect(error.empty(), error.c_str());
    expect(compressed.columns[0].plan_tree->validity.mask.data() == slab.data(),
           "mixed validity sidecar uses upload slab");
    auto ready = ::cuda::stream_ref{upload_stream}.record_event();
    stream.wait(ready);
    auto& tree    = *compressed.columns[0].plan_tree;
    auto& root    = tree.nodes.at(tree.nodes[0].children.at(0).child);
    auto* strings = dynamic_cast<simpatico::str_split_compressed_representation*>(root.rep.get());
    expect(strings != nullptr, "identity STRING storage");
    auto const* chars   = strings->channels_[1]->view().head<void>();
    auto const* offsets = strings->channels_[0]->view().head<void>();
    output = simpatico::decode_one(simpatico::column_decode_request{std::ref(tree)}, stream, mr);
    expect((output->view().head<void>() == chars) == !copy, "uploaded chars route pointer");
    expect((output->view().child(0).head<void>() == offsets) == !copy,
           "uploaded offsets route pointer");
  }
  // The output must remain readable and freeable after its upload stream and plan are gone.
  // Its moved data keeps the upload resource; copies and regenerated validity use the decode MR.
  expect(columns_equal_any(input->view().column(0), output->view(), stream),
         "uploaded strings values/nulls");
  {
    auto contents = output->release();
    auto offsets  = contents.children.at(0)->release();
    expect(contents.data->stream().get() == stream.get(), "uploaded chars deallocation stream");
    expect(offsets.data->stream().get() == stream.get(), "uploaded offsets deallocation stream");
    auto const data_mr = copy ? mr : upload_mr;
    expect(contents.data->memory_resource() == data_mr, "uploaded chars allocating resource");
    expect(offsets.data->memory_resource() == data_mr, "uploaded offsets allocating resource");
    expect(contents.null_mask->memory_resource() == mr, "uploaded validity allocating resource");
  }
  stream.sync();  // Observe the moved buffers' frees before their resource is destroyed.
}

void padded_carrier(bool copy, ::cuda::stream_ref stream, rmm::device_async_resource_ref mr)
{
  // Exercise the widened carrier's tail without requiring a multi-GiB allocation.
  std::array<std::int64_t, 3> host_offsets{0, 2, 5};
  std::array<char, 8> host_chars{'a', 'b', 'c', 'd', 'e', 0, 0, 0};
  auto offsets = cudf::make_fixed_width_column(
    cudf::data_type{cudf::type_id::INT64}, 3, cudf::mask_state::UNALLOCATED, stream, mr);
  auto chars = cudf::make_fixed_width_column(
    cudf::data_type{cudf::type_id::UINT32}, 2, cudf::mask_state::UNALLOCATED, stream, mr);
  cuda_check(cudaMemcpyAsync(offsets->mutable_view().head<void>(),
                             host_offsets.data(),
                             sizeof(host_offsets),
                             cudaMemcpyHostToDevice,
                             stream.get()));
  cuda_check(cudaMemcpyAsync(chars->mutable_view().head<void>(),
                             host_chars.data(),
                             sizeof(host_chars),
                             cudaMemcpyHostToDevice,
                             stream.get()));
  auto const* pointer = chars->view().head<void>();
  simpatico::str_split_compressed_representation rep{2, std::move(offsets), std::move(chars)};
  std::array streams{stream};
  simpatico::decode_session session{streams, mr};
  auto& frame = simpatico::decode_session_test_access::frame(session);
  auto output = rep.decompress_consuming(frame);
  (void)session.finish();
  expect((output->view().head<void>() == pointer) == !copy, "padded carrier pointer");
  expect(cudf::strings_column_view(output->view()).chars_size(stream) == 5,
         "padded allocation is not the logical chars size");
  auto expected = make_strings_column({"ab", "cde"}, {}, stream);
  expect(columns_equal_any(expected->view(), output->view(), stream), "padded carrier values");
}

}  // namespace

int main(int argc, char** argv)
{
  try {
    bool const copy = argc == 2 && std::strcmp(argv[1], "--copy") == 0;
    expect(simpatico::string_decode_copy_enabled() == copy, "copy switch/process mismatch");
    rmm::mr::cuda_async_memory_resource resource;
    rmm::cuda_stream stream;
    auto const mr = rmm::device_async_resource_ref{resource};
    snappy_strings(copy, stream.view(), mr);
    uploaded_strings(copy, stream.view(), mr);
    padded_carrier(copy, stream.view(), mr);
    std::puts("test_string_decode_move: PASS");
    return 0;
  } catch (std::exception const& e) {
    std::fprintf(stderr, "test_string_decode_move: FAIL: %s\n", e.what());
    return 1;
  }
}
