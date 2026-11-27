// SPDX-License-Identifier: Apache-2.0
#include "codegen/util/cuda_check.hpp"
#include "decode/decode_session.hpp"
#include "operators/onpair_compressor.hpp"
#include "operators/onpair_layout.cuh"
#include "util/host_observation.hpp"
#include "util/onpair_like.hpp"

#include <cub/cub.cuh>

#include <array>
#include <limits>

namespace simpatico {
namespace {

struct literal_automaton {
  std::uint16_t const* delta;
  std::uint8_t text[32];
  std::uint8_t failure[32];
  int length;
};
struct pattern_automaton {
  literal_automaton literals[4];
  int count;
  bool invert;
};

__device__ int step(literal_automaton const& literal, int state, std::uint8_t byte)
{
  if (state == literal.length) state = literal.failure[state - 1];
  while (state && byte != literal.text[state])
    state = literal.failure[state - 1];
  return state + (byte == literal.text[state]);
}

__global__ void build_delta(literal_automaton literal,
                            std::uint8_t const* bytes,
                            std::uint32_t const* offsets,
                            std::uint32_t dict_size,
                            std::uint16_t* delta)
{
  auto const entry = blockIdx.x * blockDim.x + threadIdx.x;
  if (entry >= (literal.length + 1) * dict_size) return;
  auto const token = entry % dict_size;
  int state        = entry / dict_size;
  int first_end    = 0;
  for (auto pos = offsets[token]; pos < offsets[token + 1]; ++pos) {
    state = step(literal, state, bytes[pos]);
    if (!first_end && state == literal.length) first_end = pos - offsets[token] + 1;
  }
  // [5:0] state, [6] hit, [10:7] FIRST end minus one. 16 is representable.
  delta[entry] = static_cast<std::uint16_t>(state | (first_end ? 64 : 0) |
                                            (first_end ? (first_end - 1) << 7 : 0));
}

// One warp per batch. Each scan recovers another 32 token byte starts.
// lower_bound + equal-run writes handle any number of adjacent empty rows.
template <int K, typename Offset>
__device__ __forceinline__ void row_token_bounds_impl(std::uint16_t const* codes,
                                                      std::uint32_t tokens,
                                                      std::uint8_t const* lengths,
                                                      std::uint64_t const* prefixes,
                                                      Offset const* rows,
                                                      std::uint32_t num_rows,
                                                      std::uint32_t* starts)
{
  auto const batch = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
  if (static_cast<std::uint64_t>(batch) * (32 * K) >= tokens) return;
  auto const lane     = threadIdx.x % 32;
  auto base           = prefixes[batch];
  std::uint32_t first = 0, last = 0;
  if (lane == 0) {
    // Restrict token-to-row searches to this batch's byte interval. Two global
    // searches per warp avoid a global row_offsets search for every token.
    auto lower_row = [&](std::uint64_t byte) {
      std::uint32_t lo = 0, hi = num_rows + 1;
      while (lo < hi) {
        auto const mid = lo + (hi - lo) / 2;
        if (static_cast<std::uint64_t>(rows[mid]) < byte)
          lo = mid + 1;
        else
          hi = mid;
      }
      return lo;
    };
    first = lower_row(base);
    last  = lower_row(prefixes[batch + 1]);
  }
  first = __shfl_sync(0xffffffff, first, 0);
  last  = __shfl_sync(0xffffffff, last, 0);
  for (int group = 0; group < K; ++group) {
    auto const token   = batch * (32 * K) + group * 32 + lane;
    unsigned const len = token < tokens ? onpair_detail::token_length<K>(lengths, codes[token]) : 0;
    unsigned sum       = len;
    for (int shift = 1; shift < 32; shift *= 2) {
      auto const previous = __shfl_up_sync(0xffffffff, sum, shift);
      if (lane >= shift) sum += previous;
    }
    auto const byte = base + sum - len;
    if (token < tokens) {
      std::uint32_t lo = first, hi = last;
      while (lo < hi) {
        auto const mid = lo + (hi - lo) / 2;
        if (static_cast<std::uint64_t>(rows[mid]) < byte)
          lo = mid + 1;
        else
          hi = mid;
      }
      while (lo < last && static_cast<std::uint64_t>(rows[lo]) == byte)
        starts[lo++] = token;
    }
    base += __shfl_sync(0xffffffff, sum, 31);
  }
}

template <typename Offset>
__global__ void row_token_bounds(std::uint16_t const* codes,
                                 std::uint32_t tokens,
                                 std::uint8_t const* lengths,
                                 std::uint64_t const* prefixes,
                                 Offset const* rows,
                                 std::uint32_t num_rows,
                                 std::uint32_t* starts)
{ row_token_bounds_impl<4>(codes, tokens, lengths, prefixes, rows, num_rows, starts); }

template <typename Offset>
__global__ void row_token_bounds_k6(std::uint16_t const* codes,
                                    std::uint32_t tokens,
                                    std::uint8_t const* lengths,
                                    std::uint64_t const* prefixes,
                                    Offset const* rows,
                                    std::uint32_t num_rows,
                                    std::uint32_t* starts)
{ row_token_bounds_impl<6>(codes, tokens, lengths, prefixes, rows, num_rows, starts); }

template <typename Offset>
__global__ void validate_bounds(Offset const* rows,
                                std::uint32_t num_rows,
                                std::uint64_t bytes,
                                std::uint32_t tokens,
                                std::uint32_t* starts,
                                std::uint32_t* status)
{
  auto const row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row > num_rows) return;
  auto const byte = static_cast<std::uint64_t>(rows[row]);
  if (byte == bytes) starts[row] = tokens;  // sentinel + trailing empty rows
  if (byte > bytes || (row == 0 && byte != 0) || (row == num_rows && byte != bytes) ||
      (row && rows[row] < rows[row - 1]) || starts[row] == 0xffffffffu)
    atomicExch(status, 1u);
}

__global__ void answer_rows(pattern_automaton pattern,
                            std::uint16_t const* codes,
                            std::uint32_t dict_size,
                            std::uint8_t const* bytes,
                            std::uint32_t const* offsets,
                            std::uint32_t const* starts,
                            std::uint32_t rows,
                            std::uint32_t const* status,
                            std::uint8_t* answers,
                            bool intersect,
                            unsigned long long* matches)
{
  auto const row = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned hit   = 0;
  if (row < rows) {
    if (*status != 0) {
      answers[row] = 0;  // no unchecked spans before completion reports corruption
    } else {
      int literal = 0, state = 0;
      for (auto token = starts[row]; token < starts[row + 1] && literal < pattern.count; ++token) {
        auto const code   = codes[token];
        auto const packed = pattern.literals[literal].delta[state * dict_size + code];
        if (!(packed & 64)) {
          state = packed & 63;
          continue;
        }
        ++literal;
        state = 0;
        if (literal == pattern.count) break;
        // Continue AFTER the first completion: literals cannot overlap. A one-literal
        // predicate never reads dictionary bytes here. Multiple hits can share one token.
        auto pos = offsets[code] + ((packed >> 7) & 15) + 1;
        while (literal < pattern.count && pos < offsets[code + 1]) {
          state = step(pattern.literals[literal], state, bytes[pos++]);
          if (state == pattern.literals[literal].length) {
            ++literal;
            state = 0;
          }
        }
      }
      hit          = (literal == pattern.count) != pattern.invert;
      answers[row] = static_cast<std::uint8_t>(hit && (!intersect || answers[row]));
    }
  }
  if (matches) {
    using reduction = cub::BlockReduce<unsigned, 256>;
    __shared__ reduction::TempStorage scratch;
    auto const total = reduction(scratch).Sum(hit);
    if (threadIdx.x == 0) atomicAdd(matches, static_cast<unsigned long long>(total));
  }
}

// Retained by the completion callback, including on exceptional cleanup.
struct diagnostics {
  cudaEvent_t begin = nullptr, tables = nullptr, done = nullptr;
  rmm::device_buffer matches;
  onpair_like_stats stats;
  onpair_like_log_fn logger;
  diagnostics(::cuda::stream_ref stream, rmm::device_async_resource_ref mr, onpair_like_stats value)
    : matches(sizeof(unsigned long long), stream, mr),
      stats(std::move(value)),
      logger(onpair_like_logger)
  {
    try {
      throw_if_cuda_error(cudaEventCreate(&begin), "onpair: LIKE timing start");
      throw_if_cuda_error(cudaEventCreate(&tables), "onpair: LIKE timing tables");
      throw_if_cuda_error(cudaEventCreate(&done), "onpair: LIKE timing rows");
      throw_if_cuda_error(cudaMemsetAsync(matches.data(), 0, matches.size(), stream.get()),
                          "onpair: LIKE match counter");
    } catch (...) {
      if (begin) cudaEventDestroy(begin);
      if (tables) cudaEventDestroy(tables);
      if (done) cudaEventDestroy(done);
      throw;
    }
  }
  ~diagnostics()
  {
    if (begin) cudaEventDestroy(begin);
    if (tables) cudaEventDestroy(tables);
    if (done) cudaEventDestroy(done);
  }
};

}  // namespace

std::unique_ptr<cudf::column> onpair_compressed_representation::decompress_predicate(
  decode_predicate const& pred, decode_frame& frame) const
{
  if (!onpair_like_enabled() || !pred.valid() || pred.likes.empty() || preparation_only_)
    throw std::invalid_argument("onpair: LIKE requires a valid pattern and row offsets");
  auto const stream      = frame.stream();
  auto const mr          = frame.scratch_mr();
  auto const token_count = static_cast<std::uint64_t>(views_[0].size());
  if (token_count >= (std::uint64_t{1} << 32))
    throw std::invalid_argument("onpair: LIKE token count must be below 2^32");
  auto const tokens      = static_cast<std::uint32_t>(token_count);
  auto const rows        = static_cast<std::uint32_t>(num_rows);
  auto const dict_size   = static_cast<std::uint32_t>(views_[3].size() - 1);
  auto const& dictionary = prepared_ ? *prepared_ : *this;
  rmm::device_buffer starts(
    (static_cast<std::size_t>(rows) + 1) * sizeof(std::uint32_t), stream, mr);
  rmm::device_buffer status(sizeof(std::uint32_t), stream, mr);
  auto* bounds = static_cast<std::uint32_t*>(starts.data());
  auto* flag   = static_cast<std::uint32_t*>(status.data());
  throw_if_cuda_error(cudaMemsetAsync(bounds, 0xff, starts.size(), stream.get()),
                      "onpair: LIKE initialize row token bounds");
  throw_if_cuda_error(cudaMemsetAsync(flag, 0, status.size(), stream.get()),
                      "onpair: LIKE initialize validation");
  auto launch_bounds = [&]<typename Offset>() {
    if (tokens) {
      auto const kernel =
        tokens_per_batch() == 192 ? row_token_bounds_k6<Offset> : row_token_bounds<Offset>;
      kernel<<<(batch_count() + 7) / 8, 256, 0, stream.get()>>>(
        views_[0].data<std::uint16_t>(),
        tokens,
        static_cast<std::uint8_t const*>(dictionary.lengths_.data()),
        views_[4].data<std::uint64_t>(),
        views_[1].data<Offset>(),
        rows,
        bounds);
    }
    validate_bounds<<<(static_cast<std::uint64_t>(rows) + 1 + 255) / 256, 256, 0, stream.get()>>>(
      views_[1].data<Offset>(), rows, metadata_.decoded_bytes, tokens, bounds, flag);
  };
  if (views_[1].type().id() == cudf::type_id::INT64)
    launch_bounds.template operator()<std::int64_t>();
  else
    launch_bounds.template operator()<std::int32_t>();
  throw_if_cuda_error(cudaGetLastError(), "onpair: LIKE row token bounds");

  auto output    = cudf::make_fixed_width_column(cudf::data_type{cudf::type_id::BOOL8},
                                                 num_rows,
                                                 cudf::mask_state::UNALLOCATED,
                                                 stream,
                                                 frame.mr());
  bool intersect = false;
  for (auto const& like : pred.likes) {
    std::shared_ptr<diagnostics> timing;
    if (onpair_like_collect_stats) {
      timing = std::make_shared<diagnostics>(
        stream, mr, onpair_like_stats{like, num_rows, tokens, tokens_per_batch()});
      // Install before submitting timed work so exceptional cleanup retains the event/counter
      // owners.
      frame.on_completion([timing, stream] {
        throw_if_cuda_error(
          cudaEventElapsedTime(&timing->stats.table_ms, timing->begin, timing->tables),
          "onpair: LIKE table elapsed time");
        throw_if_cuda_error(
          cudaEventElapsedTime(&timing->stats.row_ms, timing->tables, timing->done),
          "onpair: LIKE row elapsed time");
        read_device_bytes_completed(
          &timing->stats.matches, timing->matches.data(), sizeof(timing->stats.matches), stream);
        timing->logger(false, true, timing->stats);
      });
      throw_if_cuda_error(cudaEventRecord(timing->begin, stream.get()),
                          "onpair: LIKE timing begin");
    }
    pattern_automaton pattern{};
    pattern.count  = like.literals.size();
    pattern.invert = like.invert;
    std::array<rmm::device_buffer, 4> tables;
    for (int i = 0; i < pattern.count; ++i) {
      auto const& text = like.literals[i];
      auto& literal    = pattern.literals[i];
      literal.length   = text.size();
      std::copy(text.begin(), text.end(), literal.text);
      for (int j = 1, state = 0; j < literal.length; ++j) {
        while (state && literal.text[j] != literal.text[state])
          state = literal.failure[state - 1];
        if (literal.text[j] == literal.text[state]) ++state;
        literal.failure[j] = state;
      }
      auto const entries = static_cast<std::size_t>(literal.length + 1) * dict_size;
      tables[i]          = rmm::device_buffer(entries * sizeof(std::uint16_t), stream, mr);
      auto* delta        = static_cast<std::uint16_t*>(tables[i].data());
      literal.delta      = delta;
      if (entries)
        build_delta<<<(entries + 255) / 256, 256, 0, stream.get()>>>(
          literal,
          views_[2].data<std::uint8_t>(),
          views_[3].data<std::uint32_t>(),
          dict_size,
          delta);
    }
    throw_if_cuda_error(cudaGetLastError(), "onpair: LIKE build transition tables");
    if (timing)
      throw_if_cuda_error(cudaEventRecord(timing->tables, stream.get()),
                          "onpair: LIKE tables done");
    if (rows)
      answer_rows<<<(static_cast<std::uint64_t>(rows) + 255) / 256, 256, 0, stream.get()>>>(
        pattern,
        views_[0].data<std::uint16_t>(),
        dict_size,
        views_[2].data<std::uint8_t>(),
        views_[3].data<std::uint32_t>(),
        bounds,
        rows,
        flag,
        output->mutable_view().data<std::uint8_t>(),
        intersect,
        timing ? static_cast<unsigned long long*>(timing->matches.data()) : nullptr);
    throw_if_cuda_error(cudaGetLastError(), "onpair: LIKE answer rows");
    if (timing)
      throw_if_cuda_error(cudaEventRecord(timing->done, stream.get()), "onpair: LIKE rows done");
    intersect = true;
  }
  frame.check_device_status(std::move(status),
                            "onpair: corrupt row offsets (row starts inside a token)");
  return output;
}

}  // namespace simpatico
