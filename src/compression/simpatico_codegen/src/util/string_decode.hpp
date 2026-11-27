// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdlib>
#include <cstring>

namespace simpatico {

inline bool string_decode_copy_enabled()
{
  static bool const enabled = [] {
    auto const* value = std::getenv("SIRIUS_EXP_STRING_DECODE_COPY");
    return value && std::strcmp(value, "1") == 0;
  }();
  return enabled;
}

// The engine supplies its DEBUG sink without making standalone Simpatico depend on it.
// Decode sessions submit work synchronously on the calling CPU thread.
inline thread_local void (*string_decode_logger)(char const*) = nullptr;

class string_decode_log_scope {
 public:
  explicit string_decode_log_scope(void (*logger)(char const*)) : previous_(string_decode_logger)
  {
    string_decode_logger = logger;
  }
  ~string_decode_log_scope() { string_decode_logger = previous_; }
  string_decode_log_scope(string_decode_log_scope const&)            = delete;
  string_decode_log_scope& operator=(string_decode_log_scope const&) = delete;

 private:
  void (*previous_)(char const*);
};

inline void log_string_decode(char const* route)
{
  if (string_decode_logger) string_decode_logger(route);
}

}  // namespace simpatico
