// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdlib>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <string>

namespace simpatico {

using experimental_policy_log_fn = void (*)(char const*, bool);
inline thread_local experimental_policy_log_fn experimental_policy_logger = nullptr;

class experimental_policy_log_scope {
 public:
  explicit experimental_policy_log_scope(experimental_policy_log_fn logger)
    : previous_(experimental_policy_logger)
  { experimental_policy_logger = logger; }
  ~experimental_policy_log_scope() { experimental_policy_logger = previous_; }
  experimental_policy_log_scope(experimental_policy_log_scope const&)            = delete;
  experimental_policy_log_scope& operator=(experimental_policy_log_scope const&) = delete;

 private:
  experimental_policy_log_fn previous_;
};

// Read once; announce once when the engine installs its INFO logger. Standalone callers need no
// logging dependency, and pin-time reads before a scan cannot consume the announcement.
class experimental_policy {
 public:
  experimental_policy(char const* name, char const* tag) : tag_(tag)
  {
    auto const* value = std::getenv(name);
    if (!value || std::strcmp(value, "1") == 0)
      enabled_ = true;
    else if (std::strcmp(value, "0") == 0)
      enabled_ = false;
    else
      throw std::invalid_argument(std::string(name) + ": expected 0 or 1");
  }
  bool enabled()
  {
    if (experimental_policy_logger)
      std::call_once(logged_, [&] { experimental_policy_logger(tag_, enabled_); });
    return enabled_;
  }

 private:
  char const* tag_;
  bool enabled_;
  std::once_flag logged_;
};

}  // namespace simpatico
