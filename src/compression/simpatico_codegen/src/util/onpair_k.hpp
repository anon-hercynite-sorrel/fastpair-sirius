// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>

namespace simpatico {

class onpair_k_configuration_error : public std::invalid_argument {
 public:
  using std::invalid_argument::invalid_argument;
};

// A pin installs its INFO sink until the first OnPair encode on the submitting thread.
using onpair_k_log_fn                               = void (*)(int);
inline thread_local onpair_k_log_fn onpair_k_logger = nullptr;

class onpair_k_log_scope {
 public:
  explicit onpair_k_log_scope(onpair_k_log_fn logger) : previous_(onpair_k_logger)
  { onpair_k_logger = logger; }
  ~onpair_k_log_scope() { onpair_k_logger = previous_; }
  onpair_k_log_scope(onpair_k_log_scope const&)            = delete;
  onpair_k_log_scope& operator=(onpair_k_log_scope const&) = delete;

 private:
  onpair_k_log_fn previous_;
};

// Only the encoder consults this gate. Prepared and persisted columns carry their own geometry.
inline int onpair_compression_k()
{
  static int const k = [] {
    auto const* value = std::getenv("SIRIUS_EXP_ONPAIR_K");
    int result;
    if (!value || std::strcmp(value, "4") == 0)
      result = 4;
    else if (std::strcmp(value, "6") == 0)
      result = 6;
    else
      throw onpair_k_configuration_error(std::string("SIRIUS_EXP_ONPAIR_K: invalid value '") +
                                         value + "' (expected 4 or 6)");
    if (!onpair_k_logger)
      std::fprintf(stderr, "[onpair-k] k=%d tokens_per_batch=%d\n", result, 32 * result);
    return result;
  }();
  if (onpair_k_logger) {
    onpair_k_logger(k);
    onpair_k_logger = nullptr;
  }
  return k;
}

}  // namespace simpatico
