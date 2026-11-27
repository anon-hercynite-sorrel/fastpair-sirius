// SPDX-License-Identifier: Apache-2.0
#include "memory/hw_decompression_resource.hpp"

#include <rmm/detail/runtime_capabilities.hpp>

#include <cuda_runtime_api.h>

#include <algorithm>
#include <stdexcept>
#include <string>

namespace sirius::memory {
namespace {

[[noreturn]] void fail(std::string const& reason)
{
  throw std::runtime_error("operator_params.use_hw_decompression=true: " + reason);
}

}  // namespace

bool hw_decompression_pool_supported()
{
  // Match both guards in RMM 26.10's cuda_async_memory_resource implementation.
#if CUDART_VERSION >= RMM_MIN_HWDECOMPRESS_CUDA_VERSION
  return rmm::detail::hwdecompress::is_supported();
#else
  return false;
#endif
}

bool resolve_hw_decompression(bool requested,
                              cucascade::memory::system_topology_info const& topology)
{
  if (!requested) { return false; }
  if (topology.num_gpus == 0 || topology.gpus.empty() ||
      !std::all_of(topology.gpus.begin(), topology.gpus.end(), [](auto const& gpu) {
        return gpu.runtime_attributes.has_value() && gpu.runtime_attributes->hw_decomp;
      })) {
    fail("hardware decompression is not supported on every GPU in the topology");
  }
  bool pool_supported = false;
  try {
    pool_supported = hw_decompression_pool_supported();
  } catch (std::exception const& e) {
    fail(std::string("could not query RMM hardware decompression support: ") + e.what());
  }
  if (!pool_supported) {
    fail("stock RMM GPU pools require CUDA driver and build-time cudart >= 12.8");
  }
  return true;
}

}  // namespace sirius::memory
