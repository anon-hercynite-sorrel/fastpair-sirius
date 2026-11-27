// SPDX-License-Identifier: Apache-2.0
#include "util/onpair_k.hpp"

#include <string_view>

int main(int argc, char** argv)
{
  if (argc != 2) return 1;
  std::string_view const expected{argv[1]};
  try {
    auto const k = simpatico::onpair_compression_k();
    if (std::to_string(k) != expected) return 1;
    setenv("SIRIUS_EXP_ONPAIR_K", "changed-after-first-read", 1);
    // Standalone use warmed the cache. Each later pin must log once, across its chunks.
    static int calls    = 0;
    static int logged_k = 0;
    for (int pin = 1; pin <= 2; ++pin) {
      simpatico::onpair_k_log_scope scope{[](int value) {
        ++calls;
        logged_k = value;
      }};
      for (int chunk = 0; chunk < 2; ++chunk)
        if (simpatico::onpair_compression_k() != k) return 1;
      if (calls != pin || logged_k != k) return 1;
    }
    return 0;
  } catch (simpatico::onpair_k_configuration_error const& e) {
    return expected == "invalid" && std::strstr(e.what(), "SIRIUS_EXP_ONPAIR_K") ? 0 : 1;
  }
}
