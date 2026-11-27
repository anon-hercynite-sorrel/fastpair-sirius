// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <algorithm>
#include <string>
#include <vector>

namespace simpatico {

// Exact ordered, non-overlapping ASCII occurrences: %lit1%...%litN%.
struct like_predicate {
  std::vector<std::string> literals;
  bool invert                                  = false;
  // Only decoded scans distinguish the bound contains function from LIKE.
  bool is_contains                             = false;
  bool operator==(like_predicate const&) const = default;

  // Both routes bound the request size. Code-level matching requires ASCII,
  // while decoded matching can pass a non-ASCII literal to the cuDF fallback.
  [[nodiscard]] bool valid(bool require_ascii = true) const noexcept
  {
    return (!is_contains || literals.size() == 1) && !literals.empty() && literals.size() <= 4 &&
           std::all_of(literals.begin(), literals.end(), [require_ascii](auto const& literal) {
             return !literal.empty() && literal.size() <= 32 &&
                    std::all_of(literal.begin(), literal.end(), [require_ascii](unsigned char c) {
                      return c > 0 && (!require_ascii || c < 128) && c != '%' && c != '_';
                    });
           });
  }
};

}  // namespace simpatico
