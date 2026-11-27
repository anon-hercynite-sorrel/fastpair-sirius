// SPDX-License-Identifier: Apache-2.0
#include "memory/hw_decompression_resource.hpp"
#include "sirius_config.hpp"
#include "test_utils.hpp"

#include <cucascade/memory/config.hpp>
#include <yaml-cpp/yaml.h>

namespace {
int driver_probes = 0;

template <typename F>
void expect_startup_error(F&& operation)
{
  try {
    operation();
  } catch (std::exception const& e) {
    expect(std::strstr(e.what(), "operator_params.use_hw_decompression=true") != nullptr,
           "startup error did not name hardware decompression knob");
    std::printf("expected startup error: %s\n", e.what());
    return;
  }
  throw std::runtime_error("hardware decompression startup request unexpectedly succeeded");
}

sirius::sirius_config resolve(bool enabled,
                              bool low_level,
                              cucascade::memory::system_topology_info const& topology)
{
  auto const* text = low_level ? R"(
sirius:
  space:
    gpu: [{device_id: 0, memory_capacity: 8 MiB}]
)"
                               : R"(
sirius:
  memory:
    gpu: {usage_limit_bytes: 8 MiB}
    host: {capacity_bytes: 8 MiB}
)";

  auto node                                                 = YAML::Load(text);
  node["sirius"]["operator_params"]["use_hw_decompression"] = enabled;
  return sirius::parsed_sirius_config::from_node(node).resolve(topology);
}

void expect_stock_factory(sirius::sirius_config const& config, bool low_level)
{
  auto const& gpu =
    std::get<cucascade::memory::gpu_memory_space_config>(config.get_memory_space_configs().front());
  if (low_level) {
    expect(!gpu.mr_factory_fn, "changed low-level default factory selection");
  } else {
    auto const* factory =
      gpu.mr_factory_fn.target<decltype(&cucascade::memory::make_default_gpu_memory_resource)>();
    expect(factory && *factory == cucascade::memory::make_default_gpu_memory_resource,
           "changed configurator's stock RMM factory");
  }
}
}  // namespace

extern "C" cudaError_t __real_cudaDriverGetVersion(int* version);
extern "C" cudaError_t __wrap_cudaDriverGetVersion(int* version)
{
  ++driver_probes;
  return __real_cudaDriverGetVersion(version);
}

int main()
{
  try {
    using namespace sirius::memory;
    cucascade::memory::system_topology_info topology{};
    topology.num_gpus = 1;
    topology.gpus.emplace_back();
    topology.gpus[0].runtime_attributes = cucascade::memory::gpu_runtime_attributes{};

    // Run before anything can populate RMM's cached driver probe. A zero DE attribute
    // rejects an explicit request; disabling the knob never touches the driver probe.
    expect(!resolve_hw_decompression(false, {}), "knob off with no runtime attributes");
    expect(!resolve_hw_decompression(false, topology), "knob off with no DE");
    expect_startup_error([&] { (void)resolve_hw_decompression(true, {}); });
    expect_startup_error([&] { (void)resolve_hw_decompression(true, topology); });
    topology.gpus[0].runtime_attributes->hw_decomp  = true;
    topology.num_gpus                               = 2;
    topology.gpus.emplace_back().runtime_attributes = cucascade::memory::gpu_runtime_attributes{};
    expect_startup_error([&] { (void)resolve_hw_decompression(true, topology); });
    topology.gpus.pop_back();
    topology.num_gpus                              = 1;
    topology.gpus[0].runtime_attributes->hw_decomp = false;
    expect(driver_probes == 0, "knob off or missing DE probed the driver");

    for (bool low_level : {false, true}) {
      auto off = resolve(false, low_level, topology);
      expect(!off.hw_decompression_enabled(), "knob off enabled hardware decompression");
      expect_stock_factory(off, low_level);
      expect_startup_error([&] { (void)resolve(true, low_level, topology); });
      topology.gpus[0].runtime_attributes->hw_decomp = true;
      if (hw_decompression_pool_supported()) {
        auto on = resolve(true, low_level, topology);
        expect(on.hw_decompression_enabled(), "supported request did not enable cuDF's reader");
        expect_stock_factory(on, low_level);
      } else {
        expect_startup_error([&] { (void)resolve(true, low_level, topology); });
      }
      topology.gpus[0].runtime_attributes.reset();
      expect_startup_error([&] { (void)resolve(true, low_level, topology); });
      topology.gpus[0].runtime_attributes = cucascade::memory::gpu_runtime_attributes{};
    }
    std::puts("test_hw_decompression_gate: PASS");
    return 0;
  } catch (std::exception const& e) {
    std::fprintf(stderr, "test_hw_decompression_gate: FAIL: %s\n", e.what());
    return 1;
  }
}
