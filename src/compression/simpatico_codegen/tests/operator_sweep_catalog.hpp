// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "codegen/plan/operator_registry.hpp"

namespace simpatico::test {

// Exploration deliberately excludes opt-in codecs. Correctness sweeps should
// cover every enabled codec, without changing the production search catalog.
inline std::vector<std::string> const& operator_sweep_catalog()
{
  static auto const names = [] {
    auto result = all_compressor_names();
#ifdef SIMPATICO_ONPAIR
    result.push_back("onpair_9");
    result.push_back("onpair");
    result.push_back("onpair_16");
#endif
    return result;
  }();
  return names;
}

}  // namespace simpatico::test
