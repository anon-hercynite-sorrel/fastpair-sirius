// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "util/chunk_selective.hpp"
#include "util/onpair_like.hpp"

namespace simpatico {

inline bool decoded_like_configured()
{
  static bool const enabled = [] {
    auto const* value = std::getenv("SIRIUS_EXP_DECODED_LIKE");
    if (!value || std::strcmp(value, "0") == 0) return false;
    if (std::strcmp(value, "1") == 0) return true;
    throw std::invalid_argument("SIRIUS_EXP_DECODED_LIKE: expected 0 or 1");
  }();
  return enabled;
}

inline bool decoded_like_enabled()
{
  return decoded_like_configured() && !declined_chunk_selective_bytes;
}

inline bool scan_like_enabled() { return onpair_like_enabled() || decoded_like_configured(); }

}  // namespace simpatico
