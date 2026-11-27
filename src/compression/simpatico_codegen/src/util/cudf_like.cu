// SPDX-License-Identifier: Apache-2.0
#include "util/cudf_like.hpp"

#include <cudf/scalar/scalar.hpp>
#include <cudf/strings/contains.hpp>
#include <cudf/strings/find.hpp>
#include <cudf/unary.hpp>
#include <cudf/utilities/bit.hpp>
#include <cudf/utilities/error.hpp>

#include <stdexcept>

namespace simpatico {
namespace {

// Matching and optional NOT have finished. WHERE rejects UNKNOWN and padding.
__global__ void pack_like(bool const* flags,
                          cudf::bitmask_type const* validity,
                          cudf::size_type rows,
                          std::uint32_t* words,
                          std::size_t word_count)
{
  auto const row  = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  bool const keep = row < rows &&
                    (!validity || cudf::bit_is_set(validity, static_cast<cudf::size_type>(row))) &&
                    flags[row];
  auto const bits = __ballot_sync(0xffffffffu, keep);
  if ((threadIdx.x & 31) == 0 && row / 32 < word_count) words[row / 32] = bits;
}

void pack_where(cudf::column_view const& view,
                std::span<std::uint32_t> destination,
                ::cuda::stream_ref stream)
{
  if (destination.size() < (static_cast<std::size_t>(view.size()) + 31) / 32)
    throw std::invalid_argument("LIKE mask destination is too small");
  if (destination.empty()) return;
  constexpr int threads = 256;
  auto const blocks     = (destination.size() + threads / 32 - 1) / (threads / 32);
  pack_like<<<blocks, threads, 0, stream.get()>>>(
    view.data<bool>(), view.null_mask(), view.size(), destination.data(), destination.size());
  CUDF_CUDA_TRY(cudaGetLastError());
}
}  // namespace

std::string_view decoded_like_mask(cudf::strings_column_view const& input,
                                   like_predicate const& pattern,
                                   decoded_like_swar_fn const& try_swar,
                                   std::span<std::uint32_t> destination,
                                   ::cuda::stream_ref stream,
                                   rmm::device_async_resource_ref mr)
{
  std::unique_ptr<cudf::column> flags;
  if (pattern.is_contains) {
    flags = cudf::strings::contains(
      input, cudf::string_scalar(pattern.literals.front(), true, stream, mr), stream, mr);
  } else {
    std::string sql{"%"};
    for (auto const& literal : pattern.literals)
      sql.append(literal).push_back('%');
    if (try_swar) {
      if (auto matched = try_swar(input, sql, pattern.invert, stream, mr)) {
        pack_where(matched->view(), destination, stream);
        return "swar";  // SWAR fuses NOT into its BOOL8 output.
      }
    }
    flags = cudf::strings::like(input, sql, std::string_view{}, stream, mr);
  }
  if (pattern.invert)
    flags = cudf::unary_operation(flags->view(), cudf::unary_operator::NOT, stream, mr);
  pack_where(flags->view(), destination, stream);
  return pattern.is_contains ? "contains" : "cudf_like";
}
}  // namespace simpatico
