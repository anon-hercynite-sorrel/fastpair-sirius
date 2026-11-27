// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <rmm/device_buffer.hpp>
#include <rmm/resource_ref.hpp>

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <memory>
#include <optional>
#include <span>
#include <utility>
#include <vector>

namespace simpatico {

// One decode call, one host thread. Returned columns and published representations must use
// upstream directly. Retained slack, alignment padding and dead scratch together are <= max_bytes.
class decode_scratch_resource {
 public:
  static constexpr std::size_t max_bytes = 1024 * 1024;
  static constexpr std::size_t small_workspace_bytes = 64 * 1024;
  static constexpr std::size_t alignment = 256;

  decode_scratch_resource(rmm::device_async_resource_ref upstream,
                          std::span<const ::cuda::stream_ref> streams,
                          std::size_t rows)
    : state_(std::make_shared<state>(upstream))
  {
    for (auto stream : streams) {
      if (std::none_of(state_->lanes_.begin(), state_->lanes_.end(), [&](auto const& lane) {
            return lane.stream.get() == stream.get();
          }))
        state_->lanes_.emplace_back(stream);
    }
    // Row byte offsets/token bounds plus small status/scan workspaces. Large LIKE tables and
    // selected char buffers go upstream. Divide the cap across distinct lanes, not columns.
    auto const estimate = std::min(rows, max_bytes / 8) * 8 + small_workspace_bytes;
    state_->capacity_ =
      state_->lanes_.empty() ? 0 : std::min(estimate, max_bytes / state_->lanes_.size());
    state_->capacity_ = state_->capacity_ / alignment * alignment;
  }

  // Per-host-thread diagnostics: no atomics or driver calls on allocation/reuse. Tests observe
  // live backing while outputs/preparation are still alive, as well as the rare retirement path.
  struct counters {
    std::size_t backing_allocations      = 0;
    std::size_t live_backing_bytes       = 0;
    std::size_t cross_stream_retirements = 0;
  };
  static counters statistics() noexcept { return diagnostics_; }

  [[nodiscard]] std::size_t backed_lane_count() const noexcept
  {
    return std::count_if(state_->lanes_.begin(), state_->lanes_.end(), [](auto const& lane) {
      return lane.storage.has_value();
    });
  }

  void* allocate(::cuda::stream_ref stream, std::size_t bytes, std::size_t align)
  {
    // Limit slices to capacity/4, rounded down to the 256-byte alignment, so large live
    // buffers leave room for frequent small status/count words, CUB workspaces and prefixes.
    auto const max_slice = state_->capacity_ / 4 / alignment * alignment;
    if (bytes == 0 || bytes > max_slice || align > alignment)
      return state_->upstream_.allocate(stream, bytes, align);
    auto lane = std::find_if(state_->lanes_.begin(), state_->lanes_.end(), [&](auto const& item) {
      return item.stream.get() == stream.get();
    });
    if (lane == state_->lanes_.end() || lane->disabled)
      return state_->upstream_.allocate(stream, bytes, align);
    auto const rounded = (bytes + alignment - 1) / alignment * alignment;
    if (!lane->storage) {
      lane->segments.reserve(state_->capacity_ / alignment + 1);
      try {
        lane->storage.emplace(state_->capacity_, stream, state_->upstream_);
      } catch (rmm::out_of_memory const&) {
        // Optional capacity must not turn a satisfiable exact-size request into an OOM.
        lane->disabled = true;
        return state_->upstream_.allocate(stream, bytes, align);
      }
      ++diagnostics_.backing_allocations;
      diagnostics_.live_backing_bytes += state_->capacity_;
      lane->segments.push_back({0, state_->capacity_, true});
    }
    auto& segments = lane->segments;
    for (auto it = segments.begin(); it != segments.end(); ++it) {
      if (!it->free || it->bytes < rounded) continue;
      auto const offset = it->offset;
      auto const tail   = it->bytes - rounded;
      it->bytes         = rounded;
      it->free          = false;
      if (tail) segments.insert(it + 1, {offset + rounded, tail, true});
      return static_cast<std::byte*>(lane->storage->data()) + offset;
    }
    return state_->upstream_.allocate(stream, bytes, align);
  }

  void deallocate(::cuda::stream_ref stream,
                  void* pointer,
                  std::size_t bytes,
                  std::size_t align) noexcept
  {
    auto const address = reinterpret_cast<std::uintptr_t>(pointer);
    for (auto& lane : state_->lanes_) {
      if (!lane.storage) continue;
      auto const base = reinterpret_cast<std::uintptr_t>(lane.storage->data());
      if (address < base || address - base >= lane.storage->size()) continue;
      // Known cross-stream owners stay upstream. Retain this defensive path for an unexpected
      // consumer: order it before reuse and backing release on the allocation lane.
      if (stream.get() != lane.stream.get()) {
        ++diagnostics_.cross_stream_retirements;
        auto check = [](cudaError_t status) {
          if (status == cudaSuccess) return;
          std::fprintf(
            stderr, "decode scratch retirement failed: %s\n", cudaGetErrorString(status));
          std::terminate();
        };
        if (!lane.retired) check(cudaEventCreateWithFlags(&lane.retired, cudaEventDisableTiming));
        check(cudaEventRecord(lane.retired, stream.get()));
        check(cudaStreamWaitEvent(lane.stream.get(), lane.retired, 0));
      }
      auto& segments = lane.segments;
      auto it        = std::find_if(segments.begin(), segments.end(), [&](auto const& segment) {
        return segment.offset == address - base;
      });
      assert(it != segments.end() && !it->free);
      it->free = true;
      if (it + 1 != segments.end() && (it + 1)->free) {
        it->bytes += (it + 1)->bytes;
        segments.erase(it + 1);
      }
      if (it != segments.begin() && (it - 1)->free) {
        (it - 1)->bytes += it->bytes;
        segments.erase(it);
      }
      return;
    }
    state_->upstream_.deallocate(stream, pointer, bytes, align);
  }

  void* allocate_sync(std::size_t bytes, std::size_t align)
  { return state_->upstream_.allocate_sync(bytes, align); }
  void deallocate_sync(void* pointer, std::size_t bytes, std::size_t align) noexcept
  {
    auto const stream = ::cuda::stream_ref{cudaStream_t{nullptr}};
    deallocate(stream, pointer, bytes, align);
    if (cudaStreamSynchronize(stream.get()) != cudaSuccess) std::terminate();
  }
  bool operator==(decode_scratch_resource const& other) const noexcept
  { return state_ == other.state_; }
  friend void get_property(decode_scratch_resource const&, ::cuda::mr::device_accessible) noexcept
  {
  }

 private:
  struct segment {
    std::size_t offset, bytes;
    bool free;
  };
  struct lane_storage {
    explicit lane_storage(::cuda::stream_ref value) : stream(value) {}
    lane_storage(lane_storage&& other) noexcept
      : stream(other.stream),
        storage(std::move(other.storage)),
        segments(std::move(other.segments)),
        retired(std::exchange(other.retired, nullptr)),
        disabled(other.disabled)
    {
    }
    ~lane_storage()
    {
      assert(!storage || storage->is_empty() || (segments.size() == 1 && segments.front().free));
      if (storage) diagnostics_.live_backing_bytes -= storage->size();
      // cudaEventDestroy does not wait; queued waits keep the recorded event alive.
      if (retired) (void)cudaEventDestroy(retired);
      // storage frees on stream, after every use and every cross-stream retirement wait.
    }
    ::cuda::stream_ref stream;
    std::optional<rmm::device_buffer> storage;
    std::vector<segment> segments;
    cudaEvent_t retired = nullptr;
    bool disabled       = false;
  };
  struct state {
    explicit state(rmm::device_async_resource_ref upstream) : upstream_(upstream) {}
    rmm::device_async_resource_ref upstream_;
    std::vector<lane_storage> lanes_;
    std::size_t capacity_ = 0;
  };
  std::shared_ptr<state> state_;
  static thread_local counters diagnostics_;
};

inline thread_local decode_scratch_resource::counters decode_scratch_resource::diagnostics_{};

}  // namespace simpatico
