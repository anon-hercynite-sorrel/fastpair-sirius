// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <nvcomp/snappy.h>

#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <string>

namespace simpatico {

struct snappy_backend_options {
  nvcompDecompressBackend_t value;
  char const* name;
};

inline snappy_backend_options const& snappy_backend()
{
  static snappy_backend_options const backend = [] {
    auto const* value = std::getenv("SIRIUS_EXP_SNAPPY_BACKEND");
    if (!value || std::strcmp(value, "default") == 0) {
      return snappy_backend_options{NVCOMP_DECOMPRESS_BACKEND_DEFAULT, "default"};
    }
    if (std::strcmp(value, "hardware") == 0) {
      return snappy_backend_options{NVCOMP_DECOMPRESS_BACKEND_HARDWARE, "hardware"};
    }
    if (std::strcmp(value, "cuda") == 0) {
      return snappy_backend_options{NVCOMP_DECOMPRESS_BACKEND_CUDA, "cuda"};
    }
    throw std::invalid_argument(std::string("SIRIUS_EXP_SNAPPY_BACKEND: invalid value '") + value +
                                "' (expected default, hardware, or cuda)");
  }();
  return backend;
}

// The engine supplies INFO/DEBUG sinks without a standalone Simpatico logging dependency.
// Like string_decode_logger, the scope follows synchronous submission on the calling thread.
using snappy_log_fn = void (*)(bool, char const*, std::size_t, std::size_t);
inline thread_local snappy_log_fn snappy_backend_logger = nullptr;

class snappy_backend_log_scope {
 public:
  explicit snappy_backend_log_scope(snappy_log_fn logger) : previous_logger_(snappy_backend_logger)
  {
    snappy_backend_logger = logger;
  }
  ~snappy_backend_log_scope() { snappy_backend_logger = previous_logger_; }
  snappy_backend_log_scope(snappy_backend_log_scope const&)            = delete;
  snappy_backend_log_scope& operator=(snappy_backend_log_scope const&) = delete;

 private:
  snappy_log_fn previous_logger_;
};

inline void log_snappy_decompress(std::size_t chunks, std::size_t bytes)
{
  if (!snappy_backend_logger) { return; }
  auto const& backend = snappy_backend();
  static std::once_flag logged;
  std::call_once(logged, [&] { snappy_backend_logger(true, backend.name, chunks, bytes); });
  snappy_backend_logger(false, backend.name, chunks, bytes);
}

inline nvcompStatus_t check_snappy_decompress(nvcompStatus_t status, char const* operation)
{
  if (status != nvcompSuccess) {
    throw std::runtime_error(std::string("nvcomp Snappy backend=") + snappy_backend().name + ": " +
                             operation + " failed (status " +
                             std::to_string(static_cast<int>(status)) + ")");
  }
  return status;
}

}  // namespace simpatico
