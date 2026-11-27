/*
 * Copyright 2026, Sirius Contributors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <cuda_runtime.h>

#include <cucascade/error.hpp>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <mutex>
#include <string_view>

namespace sirius::cuda {

inline constexpr int fatal_cuda_exit_code = 86;
inline std::atomic<cudaError_t> first_cuda_failure{cudaSuccess};
inline std::atomic<bool> cuda_context_poisoned{false};
inline std::mutex cuda_failure_record_mutex;
// Zero means the submitter has no engine query identity (e.g. an external reader).
inline thread_local std::uint32_t active_cuda_query_id{0};

inline bool poisons_context(cudaError_t status) noexcept
{
  return status == cudaErrorIllegalAddress || status == cudaErrorLaunchFailure ||
         status == cudaErrorAssert || status == cudaErrorLaunchTimeout ||
         status == cudaErrorECCUncorrectable || status == cudaErrorHardwareStackError ||
         status == cudaErrorIllegalInstruction || status == cudaErrorMisalignedAddress ||
         status == cudaErrorInvalidAddressSpace || status == cudaErrorInvalidPc ||
         status == cudaErrorUnknown || status == cudaErrorContextIsDestroyed;
}

inline void record_cuda_failure(cudaError_t status,
                                char const* operation,
                                cudaStream_t stream,
                                int device             = -1,
                                std::size_t offset     = 0,
                                std::size_t bytes      = 0,
                                std::uint32_t query_id = active_cuda_query_id,
                                bool poison            = false) noexcept
{
  if (status == cudaSuccess) return;
  std::lock_guard lock(cuda_failure_record_mutex);
  auto expected    = cudaSuccess;
  bool const first = first_cuda_failure.compare_exchange_strong(expected, status);
  // Publish a non-success first status before other threads can reject admission.
  if (poison || poisons_context(status))
    cuda_context_poisoned.store(true, std::memory_order_release);
  if (first) {
    // Avoid the asynchronous logger: this record must survive immediate process exit.
    std::fprintf(stderr,
                 "Sirius first CUDA failure: operation=%s status=%d device=%d stream=%p "
                 "offset=%zu bytes=%zu query=%u\n",
                 operation,
                 static_cast<int>(status),
                 device,
                 static_cast<void*>(stream),
                 offset,
                 bytes,
                 query_id);
  }
}

// Failed synchronization cannot detach the complete source/destination/allocator
// ownership graph. Quarantine it in place and exit without running destructors.
// CUDA error 700 requires a fresh process; no scheduler recovery is attempted.
[[noreturn]] inline void fatal_cuda_failure(cudaError_t status,
                                            char const* operation,
                                            cudaStream_t stream,
                                            int device             = -1,
                                            std::size_t offset     = 0,
                                            std::size_t bytes      = 0,
                                            std::uint32_t query_id = active_cuda_query_id) noexcept
{
  record_cuda_failure(status, operation, stream, device, offset, bytes, query_id, true);
  std::fprintf(
    stderr,
    "Sirius CUDA context poisoned: operation=%s status=%d device=%d stream=%p "
    "offset=%zu bytes=%zu query=%u; unresolved owners retained; process restart required\n",
    operation,
    static_cast<int>(status),
    device,
    static_cast<void*>(stream),
    offset,
    bytes,
    query_id);
  std::fflush(stderr);
  std::_Exit(fatal_cuda_exit_code);
}

inline void reject_poisoned_cuda_context(char const* operation, cudaStream_t stream) noexcept
{
  if (cuda_context_poisoned.load(std::memory_order_acquire)) {
    fatal_cuda_failure(first_cuda_failure.load(), operation, stream);
  }
}

inline void check_cuda_failure(cudaError_t status,
                               char const* operation,
                               cudaStream_t stream,
                               int device             = -1,
                               std::size_t offset     = 0,
                               std::size_t bytes      = 0,
                               std::uint32_t query_id = active_cuda_query_id) noexcept
{
  record_cuda_failure(status, operation, stream, device, offset, bytes, query_id);
  if (poisons_context(status)) {
    fatal_cuda_failure(status, operation, stream, device, offset, bytes, query_id);
  }
}

namespace detail {
// The CUDA_TRY macros of RMM and cuCascade clear CUDA's last-error slot and keep the
// status only as the complete status token that ends their message
// ("...: <cudaErrorName> <description>"). Match that token, never occurrences in
// the file path or description, and only against the poisoning statuses.
inline cudaError_t poisoning_cuda_status_in_message(std::string_view message) noexcept
{
  // The status follows the "<file>:<line>: " field. Locate that field from the left
  // (the first ": " preceded by an all-digit line number) rather than taking the
  // last ": " in the message: the description after the status is unescaped text
  // and may itself contain ": " or a status name.
  auto status_start = message.find(": ");
  while (status_start != std::string_view::npos) {
    auto const line_start =
      status_start == 0 ? std::string_view::npos : message.rfind(':', status_start - 1);
    if (line_start != std::string_view::npos) {
      auto const line = message.substr(line_start + 1, status_start - line_start - 1);
      if (!line.empty() && line.find_first_not_of("0123456789") == std::string_view::npos) {
        break;
      }
    }
    status_start = message.find(": ", status_start + 2);
  }
  if (status_start == std::string_view::npos) return cudaSuccess;
  message.remove_prefix(status_start + 2);
  auto const name = message.substr(0, message.find(' '));
  for (auto status : {cudaErrorIllegalInstruction,
                      cudaErrorMisalignedAddress,
                      cudaErrorInvalidAddressSpace,
                      cudaErrorInvalidPc,
                      cudaErrorHardwareStackError,
                      cudaErrorIllegalAddress,
                      cudaErrorAssert,
                      cudaErrorLaunchFailure,
                      cudaErrorLaunchTimeout,
                      cudaErrorECCUncorrectable,
                      cudaErrorContextIsDestroyed,
                      cudaErrorUnknown}) {
    if (name == cudaGetErrorName(status)) return status;
  }
  return cudaSuccess;
}
}  // namespace detail

// Classify a cuCascade CUDA exception caught on an I/O error path the way a raw
// status is classified by check_cuda_failure: a poisoning status is recorded and
// exits the process; ordinary setup errors and other exceptions stay recoverable.
inline void check_cucascade_cuda_failure(std::exception_ptr error,
                                         char const* operation,
                                         cudaStream_t stream,
                                         int device             = -1,
                                         std::size_t offset     = 0,
                                         std::size_t bytes      = 0,
                                         std::uint32_t query_id = active_cuda_query_id) noexcept
{
  try {
    if (error) std::rethrow_exception(error);
  } catch (cucascade::cuda_error const& e) {
    std::string_view const message{e.what()};
    if (!message.starts_with("CUDA error at: ")) return;
    auto const status = detail::poisoning_cuda_status_in_message(message);
    if (status != cudaSuccess) {
      check_cuda_failure(status, operation, stream, device, offset, bytes, query_id);
    }
  } catch (...) {
  }
}

}  // namespace sirius::cuda
