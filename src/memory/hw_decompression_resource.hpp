// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cucascade/memory/topology_discovery.hpp>

namespace sirius::memory {

// The condition RMM uses to set cudaMemPoolCreateUsageHwDecompress on its stock async pools.
// Pool capability is independent of both Sirius's knob and the device's DE attribute.
bool hw_decompression_pool_supported();

// Resolves before memory spaces are constructed. An explicit request requires stock RMM
// pool support and a DE on every GPU, then enables cuDF's reader; it never changes the factory.
bool resolve_hw_decompression(bool requested,
                              cucascade::memory::system_topology_info const& topology);

}  // namespace sirius::memory
