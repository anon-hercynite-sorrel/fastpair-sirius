// SPDX-License-Identifier: Apache-2.0
#include "snappy_test_utils.hpp"
#include "util/snappy_backend.hpp"

#include <rmm/cuda_stream.hpp>
#include <rmm/mr/cuda_async_memory_resource.hpp>

#include <array>
#include <string>

namespace {
int info_messages  = 0;
int debug_messages = 0;

void log_decode(bool info, char const* backend, std::size_t chunks, std::size_t bytes)
{
  if (info) {
    ++info_messages;
    std::printf("[snappy-backend] backend=%s\n", backend);
  } else {
    ++debug_messages;
    expect(chunks == (bytes + 65535) / 65536, "Snappy log chunk/byte counts");
  }
}
}  // namespace

int main(int argc, char** argv)
{
  try {
    expect(argc == 2, "expected backend or invalid argument");
    std::string const mode = argv[1];
    if (mode == "invalid") {
      try {
        (void)simpatico::snappy_backend();
      } catch (std::invalid_argument const& e) {
        expect(std::strstr(e.what(), "SIRIUS_EXP_SNAPPY_BACKEND") != nullptr, "invalid env error");
        std::puts("test_snappy_backend invalid: PASS");
        return 0;
      }
      throw std::runtime_error("invalid backend was accepted");
    }
    bool const hardware = mode == "hardware";
    expect(simpatico::snappy_backend().name == mode, "backend/process mismatch");
    // The cached startup selection must survive later environment changes.
    setenv("SIRIUS_EXP_SNAPPY_BACKEND", "invalid-after-startup", 1);
    expect(simpatico::snappy_backend().name == mode, "backend was read more than once");
    if (hardware && snappy_test_device_has_de()) {
      std::puts("SKIP: device has a DE; negative hardware test requires a device without one");
      return 77;
    }
    rmm::mr::cuda_async_memory_resource resource;
    rmm::cuda_stream stream;
    auto const mr = rmm::device_async_resource_ref{resource};
    simpatico::snappy_backend_log_scope const logging{log_decode};
    if (hardware) {
      try {
        snappy_test_roundtrip(65537, 1, stream, mr);
      } catch (std::runtime_error const& e) {
        std::string const error = e.what();
        expect(error.find("nvcomp Snappy backend=hardware") != std::string::npos,
               "hardware error omitted backend");
        expect(error.find("decompress_async failed (status ") != std::string::npos,
               "hardware error did not surface nvCOMP decompress_async status");
        stream.synchronize();
        std::printf("test_snappy_backend hardware: PASS: %s\n", e.what());
        return 0;
      }
      throw std::runtime_error("hardware backend unexpectedly succeeded without a DE");
    }
    for (auto bytes : std::array<std::size_t, 7>{0, 1, 65535, 65536, 65537, 131072, 196625}) {
      for (int pattern = 0; pattern < 3; ++pattern) {
        snappy_test_roundtrip(bytes, pattern, stream, mr);
      }
    }
    expect(info_messages == 1, "Snappy INFO was not logged exactly once");
    expect(debug_messages == 36, "Snappy DEBUG was not logged for every nonempty decode");
    std::puts("test_snappy_backend: PASS");
    return 0;
  } catch (std::exception const& e) {
    std::fprintf(stderr, "test_snappy_backend: FAIL: %s\n", e.what());
    return 1;
  }
}
