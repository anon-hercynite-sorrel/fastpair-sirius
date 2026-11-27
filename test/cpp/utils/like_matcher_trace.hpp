// SPDX-License-Identifier: Apache-2.0
#pragma once

namespace sirius::test {

// The GDB CTests observe the unmodified, optimized matcher entry points between
// these markers. The label is stored in memory so the observer needs no DWARF.
extern "C" {
inline char const* sirius_test_expected_matcher = nullptr;
[[gnu::noinline, gnu::visibility("default")]] inline void sirius_test_matcher_checkpoint()
{
  asm volatile("" ::: "memory");
}
}

struct like_matcher_trace {
  explicit like_matcher_trace(char const* expected)
  {
    sirius_test_expected_matcher = expected;
    sirius_test_matcher_checkpoint();
  }
  ~like_matcher_trace()
  {
    sirius_test_expected_matcher = nullptr;
    sirius_test_matcher_checkpoint();
  }
};

}  // namespace sirius::test
