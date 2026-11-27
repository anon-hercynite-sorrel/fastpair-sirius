// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "util/like_predicate.hpp"

#include <cudf/column/column.hpp>
#include <cudf/strings/strings_column_view.hpp>

#include <rmm/resource_ref.hpp>

#include <cstdint>
#include <functional>
#include <span>
#include <string_view>

namespace simpatico {
// Optional engine SWAR attempt for LIKE. A declined pattern or layout returns nullptr.
using decoded_like_swar_fn =
  std::function<std::unique_ptr<cudf::column>(cudf::strings_column_view const&,
                                              std::string_view,
                                              bool,
                                              ::cuda::stream_ref,
                                              rmm::device_async_resource_ref)>;

// Apply the bound function and pack its nullable BOOL8 result into WHERE bits.
// Returns the name of the matcher that produced the result.
std::string_view decoded_like_mask(cudf::strings_column_view const& input,
                                   like_predicate const& pattern,
                                   decoded_like_swar_fn const& try_swar,
                                   std::span<std::uint32_t> destination,
                                   ::cuda::stream_ref stream,
                                   rmm::device_async_resource_ref mr);
}  // namespace simpatico
