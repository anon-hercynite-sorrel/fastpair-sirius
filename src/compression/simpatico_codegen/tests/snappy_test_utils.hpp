// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "api/simpatico_codegen.hpp"
#include "test_utils.hpp"
#include "util/host_observation.hpp"

#include <cuda.h>

#include <cstdio>

inline bool snappy_test_device_has_de()
{
  expect(cuInit(0) == CUDA_SUCCESS, "cuInit");
  int device = 0;
  cuda_check(cudaGetDevice(&device));
  int algorithms = 0;
  expect(cuDeviceGetAttribute(
           &algorithms, CU_DEVICE_ATTRIBUTE_MEM_DECOMPRESS_ALGORITHM_MASK, device) == CUDA_SUCCESS,
         "query DE algorithm mask");
  return algorithms != 0;
}

inline void snappy_test_roundtrip(std::size_t bytes,
                                  int pattern,
                                  ::cuda::stream_ref stream,
                                  rmm::device_async_resource_ref mr)
{
  std::vector<std::uint8_t> host(bytes);
  std::uint32_t random = 42;
  for (std::size_t i = 0; i < bytes; ++i) {
    random  = random * 1664525u + 1013904223u;
    host[i] = pattern == 0 ? 0 : pattern == 1 ? (i % 251) : (random >> 24);
  }
  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(cudf::make_fixed_width_column(cudf::data_type{cudf::type_id::UINT8},
                                                  static_cast<cudf::size_type>(bytes),
                                                  cudf::mask_state::UNALLOCATED,
                                                  stream,
                                                  mr));
  if (bytes) {
    cuda_check(cudaMemcpyAsync(columns[0]->mutable_view().head<void>(),
                               host.data(),
                               bytes,
                               cudaMemcpyHostToDevice,
                               stream.get()));
  }
  cudf::table input{std::move(columns)};
  auto compressed = simpatico::compress_with_plan(input.view(), "input -> snappy\n", stream, mr);
  for (int repeat = 0; repeat < 2; ++repeat) {
    auto const before = simpatico::host_observations.readbacks;
    auto output = compressed.decompress(stream, mr);
    expect(simpatico::host_observations.readbacks == before, "Snappy decode read frame metadata");
    std::vector<std::uint8_t> decoded(bytes);
    if (bytes) {
      cuda_check(cudaMemcpyAsync(decoded.data(),
                                 output->view().column(0).head<void>(),
                                 bytes,
                                 cudaMemcpyDeviceToHost,
                                 stream.get()));
    }
    stream.sync();
    expect(decoded == host, "Snappy decoded bytes differ from original bytes");
  }
}
