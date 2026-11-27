// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>

namespace simpatico {

// One selected str_split decode whose chars are a batched nvCOMP frame.
struct chunk_selective_stats {
  std::int64_t rows           = 0;
  std::int64_t survivors      = 0;
  std::uint32_t marked_chunks = 0;
  std::uint32_t total_chunks  = 0;
  std::size_t decoded_bytes   = 0;
  std::size_t column_index    = 0;
  char const* codec           = nullptr;
};

// As with onpair_selective_log_scope, the engine supplies its sinks on the submitting thread.
using chunk_selective_log_fn = std::function<void(bool, bool, chunk_selective_stats const&)>;
inline thread_local chunk_selective_log_fn const* chunk_selective_logger = nullptr;

class chunk_selective_log_scope {
 public:
  explicit chunk_selective_log_scope(chunk_selective_log_fn logger)
    : logger_(std::move(logger)), previous_(chunk_selective_logger)
  { chunk_selective_logger = logger_ ? &logger_ : nullptr; }
  ~chunk_selective_log_scope() { chunk_selective_logger = previous_; }
  chunk_selective_log_scope(chunk_selective_log_scope const&)            = delete;
  chunk_selective_log_scope& operator=(chunk_selective_log_scope const&) = delete;

 private:
  chunk_selective_log_fn logger_;
  chunk_selective_log_fn const* previous_;
};

// Admission can decline the optional frame reservation for one conversion.
inline thread_local std::size_t declined_chunk_selective_bytes = 0;
class chunk_selective_plain_scope {
 public:
  explicit chunk_selective_plain_scope(std::size_t declined_bytes)
    : previous_(declined_chunk_selective_bytes)
  {
    if (declined_bytes) declined_chunk_selective_bytes = declined_bytes;
  }
  ~chunk_selective_plain_scope() { declined_chunk_selective_bytes = previous_; }
  chunk_selective_plain_scope(chunk_selective_plain_scope const&)            = delete;
  chunk_selective_plain_scope& operator=(chunk_selective_plain_scope const&) = delete;

 private:
  std::size_t previous_;
};

// Pinned SQL scans require SIRIUS_EXP_FUSED_SCAN_FILTER=1 and an admitted row selection.
// With SIRIUS_EXP_CHUNK_SELECTIVE=1, selected str_split strings decode only touched Snappy or
// ANS blocks. With the switch unset or 0 they use the full chars decode.
inline bool chunk_selective_enabled()
{
  static bool const enabled = [] {
    auto const* value = std::getenv("SIRIUS_EXP_CHUNK_SELECTIVE");
    if (!value || std::strcmp(value, "0") == 0) return false;
    if (std::strcmp(value, "1") == 0) return true;
    throw std::invalid_argument(std::string("SIRIUS_EXP_CHUNK_SELECTIVE: invalid value '") + value +
                                "' (expected 0 or 1)");
  }();
  if (declined_chunk_selective_bytes) return false;
  if (enabled && chunk_selective_logger) {
    static std::once_flag logged;
    std::call_once(logged, [&] { (*chunk_selective_logger)(true, enabled, {}); });
  }
  return enabled;
}

inline void log_chunk_selected(chunk_selective_stats const& stats)
{
  if (chunk_selective_logger) (*chunk_selective_logger)(false, true, stats);
}

}  // namespace simpatico
