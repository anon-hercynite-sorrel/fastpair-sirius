// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <rmm/aligned.hpp>
#include <rmm/device_buffer.hpp>

#include <cuda/memory_resource>
#include <cuda/stream>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace simpatico {

// One slice per nonempty reconstructed payload buffer, in reader allocation order.
// Keep the slab at a stable address until the last slice and all its GPU reads are done.
class compressed_leaf_slab {
 public:
  compressed_leaf_slab()                                       = default;
  compressed_leaf_slab(compressed_leaf_slab const&)            = delete;
  compressed_leaf_slab& operator=(compressed_leaf_slab const&) = delete;

  static constexpr std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT;

  void add_buffer(std::uint64_t bytes)
  {
    if (payload_.size()) throw std::logic_error("compressed leaf slab already allocated");
    if (!bytes) return;  // Empty cuDF columns make no allocation.
    constexpr auto max = std::numeric_limits<std::size_t>::max();
    if (bytes > max - (alignment - 1)) throw std::overflow_error("compressed leaf size overflow");
    auto const padded = (static_cast<std::size_t>(bytes) + alignment - 1) & ~(alignment - 1);
    auto const offset = ends_.empty() ? 0 : ends_.back();
    if (padded > max - offset) throw std::overflow_error("compressed leaf slab size overflow");
    ends_.push_back(offset + padded);
  }

  void allocate(::cuda::stream_ref stream, rmm::device_async_resource_ref mr, std::size_t tail = 0)
  {
    if (payload_.size()) throw std::logic_error("compressed leaf slab already allocated");
    auto const bytes = ends_.empty() ? 0 : ends_.back();
    if (tail > std::numeric_limits<std::size_t>::max() - bytes)
      throw std::overflow_error("compressed leaf slab tail overflow");
    payload_ = rmm::device_buffer(bytes + tail, stream, mr);
  }

  std::size_t size() const { return payload_.size(); }
  void* data() { return payload_.data(); }
  void* slice(std::size_t i)
  {
    if (!data() || i >= ends_.size()) throw std::logic_error("invalid compressed leaf slice");
    return static_cast<std::byte*>(data()) + (i == 0 ? 0 : ends_[i - 1]);
  }
  rmm::device_async_resource_ref resource() { return rmm::device_async_resource_ref{resource_}; }

 private:
  void* next_slice(std::size_t bytes, std::size_t align)
  {
    if (bytes == 0) return nullptr;
    auto const i = cursor_;
    if (i >= ends_.size() || align == 0 || alignment % align != 0 ||
        bytes > ends_[i] - (i ? ends_[i - 1] : 0))
      throw std::logic_error("compressed leaf allocation does not fit its slab slice");
    ++cursor_;
    return slice(i);
  }

  struct slab_resource {
    compressed_leaf_slab* slab;

    void* allocate_sync(std::size_t bytes, std::size_t align = alignment)
    {
      ::cuda::stream_ref{slab->payload_.stream()}.sync();
      return slab->next_slice(bytes, align);
    }
    void* allocate(::cuda::stream_ref stream, std::size_t bytes, std::size_t align = alignment)
    {
      // Reconstruction allocates on the backing stream. Order a different allocation stream
      // after the backing allocation if a column factory requests one.
      ::cuda::stream_ref const backing{slab->payload_.stream()};
      if (stream.get() != backing.get()) {
        auto ready = backing.record_event();
        stream.wait(ready);
      }
      return slab->next_slice(bytes, align);
    }
    void deallocate_sync(void*, std::size_t, std::size_t = alignment) noexcept {}
    void deallocate(::cuda::stream_ref, void*, std::size_t, std::size_t = alignment) noexcept {}
    bool operator==(slab_resource const&) const noexcept = default;
    friend void get_property(slab_resource const&, ::cuda::mr::device_accessible) noexcept {}
  };

  std::vector<std::size_t> ends_;
  std::size_t cursor_ = 0;
  rmm::device_buffer payload_;
  slab_resource resource_{this};
};

}  // namespace simpatico
