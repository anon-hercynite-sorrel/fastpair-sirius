// SPDX-License-Identifier: Apache-2.0
#pragma once

#ifdef SIMPATICO_ONPAIR
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <string>

namespace simpatico {

struct onpair_selective_stats {
  std::int64_t rows            = 0;
  std::int64_t survivors       = 0;
  std::uint32_t marked_batches = 0;
  std::uint32_t total_batches  = 0;
  std::size_t scratch_bytes    = 0;
  std::uint16_t tokens_per_batch = 128;
};

// As with string_decode_log_scope, the engine supplies its sinks on the submitting thread.
using onpair_selective_log_fn = void (*)(bool, bool, onpair_selective_stats const&);
inline thread_local onpair_selective_log_fn onpair_selective_logger = nullptr;

class onpair_selective_log_scope {
 public:
  explicit onpair_selective_log_scope(onpair_selective_log_fn logger)
    : previous_(onpair_selective_logger)
  { onpair_selective_logger = logger; }
  ~onpair_selective_log_scope() { onpair_selective_logger = previous_; }
  onpair_selective_log_scope(onpair_selective_log_scope const&)            = delete;
  onpair_selective_log_scope& operator=(onpair_selective_log_scope const&) = delete;

 private:
  onpair_selective_log_fn previous_;
};

inline bool onpair_selective_enabled()
{
  static bool const enabled = [] {
    auto const* value = std::getenv("SIRIUS_EXP_ONPAIR_SELECTIVE");
    if (!value || std::strcmp(value, "1") == 0) return true;
    if (std::strcmp(value, "0") == 0) return false;
    throw std::invalid_argument(std::string("SIRIUS_EXP_ONPAIR_SELECTIVE: invalid value '") +
                                value + "' (expected 0 or 1)");
  }();
  if (onpair_selective_logger) {
    static std::once_flag logged;
    std::call_once(logged, [&] { onpair_selective_logger(true, enabled, {}); });
  }
  return enabled;
}

inline void log_onpair_selected(onpair_selective_stats const& stats)
{
  if (onpair_selective_logger) onpair_selective_logger(false, true, stats);
}

}  // namespace simpatico
#endif
