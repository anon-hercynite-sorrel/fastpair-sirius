// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>

namespace simpatico::onpair_detail {

template <int K>
__device__ inline std::uint32_t token_length(std::uint8_t const* lengths, std::uint32_t code)
{
  static_assert(K == 4 || K == 6);
  if constexpr (K == 4) return lengths[code];
  // Even codes use the low nibble. Encoding length - 1 represents all lengths 1..16.
  return ((lengths[code / 2] >> ((code & 1) * 4)) & 15u) + 1u;
}

}  // namespace simpatico::onpair_detail
