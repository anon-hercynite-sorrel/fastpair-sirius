// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "util/like_predicate.hpp"

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <stdexcept>

namespace simpatico {

struct onpair_like_stats {
  like_predicate pattern;
  std::int64_t rows              = 0;
  std::uint32_t tokens           = 0;  // row_token_bounds uses UINT32 indices
  std::uint16_t tokens_per_batch = 128;
  float table_ms                 = 0;
  float row_ms                   = 0;
  std::uint64_t matches          = 0;  // after inversion
};
using onpair_like_log_fn = void (*)(bool, bool, onpair_like_stats const&);
inline thread_local onpair_like_log_fn onpair_like_logger = nullptr;
inline thread_local bool onpair_like_collect_stats        = false;
class onpair_like_log_scope {
 public:
  explicit onpair_like_log_scope(onpair_like_log_fn logger, bool collect_stats = true)
    : previous_(onpair_like_logger), previous_stats_(onpair_like_collect_stats)
  {
    onpair_like_logger = logger;
    onpair_like_collect_stats = logger && collect_stats;
  }
  ~onpair_like_log_scope()
  {
    onpair_like_logger        = previous_;
    onpair_like_collect_stats = previous_stats_;
  }
  onpair_like_log_scope(onpair_like_log_scope const&)            = delete;
  onpair_like_log_scope& operator=(onpair_like_log_scope const&) = delete;

 private:
  onpair_like_log_fn previous_;
  bool previous_stats_;
};

inline bool onpair_like_enabled()
{
#ifdef SIMPATICO_ONPAIR
  static bool const enabled = [] {
    auto const* value = std::getenv("SIRIUS_EXP_ONPAIR_LIKE");
    if (!value || std::strcmp(value, "1") == 0) return true;
    if (std::strcmp(value, "0") == 0) return false;
    throw std::invalid_argument("SIRIUS_EXP_ONPAIR_LIKE: expected 0 or 1");
  }();
#else
  constexpr bool enabled = false;
#endif
  if (onpair_like_logger) {
    static std::once_flag logged;
    std::call_once(logged, [&] { onpair_like_logger(true, enabled, {}); });
  }
  return enabled;
}

}  // namespace simpatico
