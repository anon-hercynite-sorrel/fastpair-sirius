// SPDX-License-Identifier: Apache-2.0

#include "api/compressed_table_io.hpp"
#include "api/simpatico_codegen.hpp"
#include "codegen/plan/plan_tree.hpp"
#include "decode/decode_session.hpp"
#include "operators/onpair_compressor.hpp"
#include "test_utils.hpp"
#include "util/host_observation.hpp"
#include "util/onpair_k.hpp"

#include <cudf/copying.hpp>
#include <cudf/strings/strings_column_view.hpp>

#include <rmm/cuda_stream.hpp>
#include <rmm/mr/cuda_async_memory_resource.hpp>

#include <unistd.h>

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <string_view>
#include <vector>

// Catch accidental host observations in the statically linked decoder during append().
namespace decode_observation {
thread_local bool active       = false;
thread_local std::size_t calls = 0;
thread_local std::size_t readbacks = 0;
}  // namespace decode_observation
extern "C" cudaError_t __real_cudaStreamQuery(cudaStream_t);
extern "C" cudaError_t __real_cudaStreamSynchronize(cudaStream_t);
extern "C" cudaError_t __real_cudaMemcpyAsync(
  void*, void const*, std::size_t, cudaMemcpyKind, cudaStream_t);
extern "C" cudaError_t __wrap_cudaMemcpyAsync(
  void* dst, void const* src, std::size_t bytes, cudaMemcpyKind kind, cudaStream_t stream)
{
  if (decode_observation::active && kind == cudaMemcpyDeviceToHost) ++decode_observation::readbacks;
  return __real_cudaMemcpyAsync(dst, src, bytes, kind, stream);
}
extern "C" cudaError_t __wrap_cudaStreamQuery(cudaStream_t stream)
{
  if (decode_observation::active) ++decode_observation::calls;
  return __real_cudaStreamQuery(stream);
}
extern "C" cudaError_t __wrap_cudaStreamSynchronize(cudaStream_t stream)
{
  if (decode_observation::active) ++decode_observation::calls;
  return __real_cudaStreamSynchronize(stream);
}

namespace {

struct temp_file {
  std::string path;
  temp_file()
  {
    char name[] = "/tmp/simpatico_onpair_XXXXXX";
    auto fd     = mkstemp(name);
    if (fd < 0) throw std::runtime_error("onpair: mkstemp failed");
    ::close(fd);
    path = name;
  }
  ~temp_file() { std::remove(path.c_str()); }
};

template <typename T>
std::unique_ptr<cudf::column> upload_channel(std::vector<T> const& host,
                                             cudf::type_id type,
                                             ::cuda::stream_ref stream,
                                             rmm::device_async_resource_ref mr)
{
  auto col = cudf::make_fixed_width_column(
    cudf::data_type{type}, host.size(), cudf::mask_state::UNALLOCATED, stream, mr);
  if (!host.empty())
    cuda_check(cudaMemcpyAsync(col->mutable_view().template head<void>(),
                               host.data(),
                               host.size() * sizeof(T),
                               cudaMemcpyHostToDevice,
                               stream.get()));
  // Allow callers to pass a temporary vector; construction is outside decode.
  stream.sync();
  return col;
}

// Compare the entire offsets child, including null/empty rows. Slices rebase to
// zero at compression; a childless empty STRING has the implicit boundary [0].
std::vector<std::int64_t> byte_offsets(cudf::column_view column, ::cuda::stream_ref stream)
{
  if (column.size() == 0) return {0};
  auto const offsets = cudf::strings_column_view(column).offsets();
  std::vector<std::int64_t> result(static_cast<std::size_t>(column.size()) + 1);
  if (offsets.type().id() == cudf::type_id::INT64) {
    simpatico::read_device_bytes_completed(result.data(),
                                           offsets.data<std::int64_t>() + column.offset(),
                                           result.size() * sizeof(std::int64_t),
                                           stream);
  } else {
    std::vector<std::int32_t> narrow(result.size());
    simpatico::read_device_bytes_completed(narrow.data(),
                                           offsets.data<std::int32_t>() + column.offset(),
                                           narrow.size() * sizeof(std::int32_t),
                                           stream);
    std::copy(narrow.begin(), narrow.end(), result.begin());
  }
  auto const origin = result.front();
  for (auto& offset : result)
    offset -= origin;
  return result;
}

void compare_offsets(cudf::column_view input, cudf::column_view decoded, ::cuda::stream_ref stream)
{
  expect(byte_offsets(input, stream) == byte_offsets(decoded, stream),
         "onpair: decoded byte offsets differ from input");
}

void check_named_channels(simpatico::compressed_representation const& rep,
                          cudf::type_id offset_type,
                          ::cuda::stream_ref stream)
{
  auto const channels = rep.named_channels(stream);
  auto const& names   = simpatico::op_info(simpatico::OpId::OnPair).channels;
  std::array<cudf::type_id, 5> const types{cudf::type_id::UINT16,
                                           offset_type,
                                           cudf::type_id::UINT8,
                                           cudf::type_id::UINT32,
                                           cudf::type_id::UINT64};
  expect(channels.size() == 5, "onpair: five named compact channels");
  for (std::size_t i = 0; i < channels.size(); ++i)
    expect(channels[i].name == names[i] && channels[i].view.type().id() == types[i],
           "onpair: named channel order and types (row_offsets in bytes)");
  auto const width = std::get<simpatico::leaf_meta::onpair>(rep.describe_meta()).tokens_per_batch;
  expect(channels[4].view.size() ==
           (channels[0].view.size() + static_cast<int64_t>(width) - 1) / width + 1,
         "onpair: batch_prefixes has a byte offset for every batch and final total");
}

// Controlled fixtures supply the same decoded byte prefixes as the device sweep.
std::vector<std::uint64_t> batch_prefixes(std::vector<std::uint16_t> const& codes,
                                          std::vector<std::uint32_t> const& dictionary,
                                          int width = 128)
{
  std::vector<std::uint64_t> result{0};
  std::uint64_t bytes = 0;
  for (std::size_t i = 0; i < codes.size(); ++i) {
    bytes += dictionary[codes[i] + 1] - dictionary[codes[i]];
    if ((i + 1) % width == 0 || i + 1 == codes.size()) result.push_back(bytes);
  }
  return result;
}

void check_resident_bytes(simpatico::compressed_table const& table, ::cuda::stream_ref stream)
{
  auto const descs = table.describe(stream);
  for (std::size_t c = 0; c < table.columns.size(); ++c) {
    std::size_t wire_bytes     = table.columns[c].plan_tree->validity.mask.size();
    std::size_t prepared_bytes = 0;
    for (auto const& leaf : descs[c])
      for (auto const& buffer : leaf.buffers)
        wire_bytes += buffer.size_bytes;
    for (auto const& node : table.columns[c].plan_tree->nodes) {
      auto dictionary_bytes = [&](std::size_t entries) {
        auto const width = std::get<simpatico::leaf_meta::onpair>(node.meta).tokens_per_batch;
        return width == 192 ? 16 * entries + (entries + 1) / 2 : 25 * entries;
      };
      if (node.rep && node.rep->kind() == simpatico::OpId::OnPair) {
        auto const channels = node.rep->named_channels(stream);
        prepared_bytes += dictionary_bytes(channels[3].view.size() - 1);
      } else if (node.prepared_rep) {
        auto const* offsets = simpatico::terminal_identity_channel(node, "dict_offsets");
        expect(offsets != nullptr, "onpair: fixture dictionary is an identity channel");
        prepared_bytes += dictionary_bytes(offsets->size() - 1);
        expect(node.prepared_rep->named_channels(stream).empty(),
               "onpair: prepared state must not be serialized twice");
      }
    }
    expect(table.columns[c].resident_size_bytes(stream) == wire_bytes + prepared_bytes,
           "onpair: resident footprint must include all three dictionary planes exactly once");
  }
}

void compare(cudf::table_view input,
             simpatico::compressed_table const& compressed,
             ::cuda::stream_ref stream,
             rmm::device_async_resource_ref mr)
{
  check_resident_bytes(compressed, stream);
  // Decode twice: a scan must reconstruct its output every time, including
  // when channels were consumed by reconstruction during the previous scan.
  for (int scan = 0; scan < 2; ++scan) {
    auto decoded = simpatico::decompress(compressed, stream, mr);
    expect(decoded && decoded->num_columns() == input.num_columns(), "onpair: decoded shape");
    for (int i = 0; i < input.num_columns(); ++i) {
      expect(columns_equal_any(input.column(i), decoded->view().column(i), stream),
             "onpair: data or validity mismatch");
      compare_offsets(input.column(i), decoded->view().column(i), stream);
      if (input.column(i).size() != 0)
        expect(cudf::strings_column_view(input.column(i)).offsets().type() ==
                 cudf::strings_column_view(decoded->view().column(i)).offsets().type(),
               "onpair: decode preserves input byte-offset width");
    }
  }
}

void roundtrip(cudf::table_view input,
               std::string const& plan,
               ::cuda::stream_ref stream,
               rmm::device_async_resource_ref mr)
{
  auto compressed = simpatico::compress_with_plan(input, plan, stream, mr);
  compare(input, compressed, stream, mr);

  temp_file file;
  auto error = simpatico::write_compressed_table(compressed, file.path, stream);
  expect(error.empty(), error.c_str());
  auto restored = simpatico::read_compressed_table(file.path, stream, mr, &error);
  expect(error.empty(), error.c_str());
  expect(restored.columns.size() == compressed.columns.size(), "onpair: restored shape");
  compare(input, restored, stream, mr);

  auto const before = compressed.describe(stream);
  auto const after  = restored.describe(stream);
  expect(before.size() == after.size(), "onpair: descriptor count");
  for (std::size_t c = 0; c < before.size(); ++c) {
    expect(before[c].size() == after[c].size(), "onpair: leaf count");
    for (std::size_t i = 0; i < before[c].size(); ++i) {
      expect(before[c][i].kind == after[c][i].kind, "onpair: persisted leaf kind");
      if (auto const* meta = std::get_if<simpatico::leaf_meta::onpair>(&before[c][i].meta)) {
        auto const* read = std::get_if<simpatico::leaf_meta::onpair>(&after[c][i].meta);
        expect(read && read->num_rows == meta->num_rows &&
                 read->decoded_bytes == meta->decoded_bytes && read->dict_bits == meta->dict_bits &&
                 read->original_type_id == meta->original_type_id &&
                 read->tokens_per_batch == meta->tokens_per_batch,
               "onpair: persisted leaf metadata");
        expect(before[c][i].buffers.size() == 5 && after[c][i].buffers.size() == 5,
               "onpair: five compact channels on disk");
        auto const& names = simpatico::op_info(simpatico::OpId::OnPair).channels;
        for (std::size_t channel = 0; channel < names.size(); ++channel)
          expect(after[c][i].buffers[channel].name == names[channel],
                 "onpair: persisted compact channel names (row_offsets in bytes)");
        expect(after[c][i].buffers[4].type_tag ==
                 simpatico::dtype_to_tag(cudf::data_type{cudf::type_id::UINT64}),
               "onpair: persisted batch byte prefixes are UINT64");
      }
    }
    auto const& a  = compressed.columns[c].plan_tree->nodes[1].meta;
    auto const& b  = restored.columns[c].plan_tree->nodes[1].meta;
    auto const* ma = std::get_if<simpatico::leaf_meta::onpair>(&a);
    auto const* mb = std::get_if<simpatico::leaf_meta::onpair>(&b);
    expect(ma && mb && ma->decoded_bytes == mb->decoded_bytes && ma->dict_bits == mb->dict_bits &&
             ma->tokens_per_batch == mb->tokens_per_batch,
           "onpair: node metadata survives channel sub-plans");
  }

  // Exercise the pin path's split header/payload API as well as the file API.
  std::vector<std::uint8_t> header;
  std::vector<simpatico::payload_buffer_ref> buffers;
  std::uint64_t bytes = 0;
  error = simpatico::build_compressed_table_header(compressed, header, buffers, bytes, stream);
  expect(error.empty(), error.c_str());
  std::vector<std::uint8_t> payload(bytes);
  for (auto const& b : buffers) {
    if (b.size_bytes != 0)
      cuda_check(cudaMemcpyAsync(payload.data() + b.offset,
                                 b.device_ptr,
                                 b.size_bytes,
                                 cudaMemcpyDeviceToHost,
                                 stream.get()));
  }
  stream.sync();
  auto fetch = [&](std::uint64_t offset, std::size_t size, void* dst, ::cuda::stream_ref s) {
    cuda_check(
      cudaMemcpyAsync(dst, payload.data() + offset, size, cudaMemcpyHostToDevice, s.get()));
  };
  auto pinned = simpatico::read_compressed_table_from_memory(header, fetch, stream, mr, &error);
  expect(error.empty(), error.c_str());
  compare(input, pinned, stream, mr);
}

void test_registry()
{
  using namespace simpatico;
  expect(op_id_from_name("onpair") == OpId::OnPair, "onpair: bare name");
  for (int bits = 9; bits <= 16; ++bits)
    expect(make_compressor("onpair_" + std::to_string(bits)) != nullptr,
           "onpair: supported preset");
  for (char const* name : {"onpair_", "onpair_8", "onpair_17", "onpair_012", "onpair_16x"})
    expect(!op_id_from_name(name) && !make_compressor(name), "onpair: reject invalid preset");
  expect(is_terminal_compressor("onpair_16"), "onpair: terminal");
  expect(op_info(OpId::OnPair).channels ==
           std::vector<std::string>{
             "codes", "row_offsets", "dict_bytes", "dict_offsets", "batch_prefixes"},
         "onpair: channel order");
  auto const& defaults = all_compressor_names();
  expect(std::find(defaults.begin(), defaults.end(), "onpair") == defaults.end(),
         "onpair: explore remains opt-in");
  auto numeric = make_int32_table(1, 8, 1);
  expect(throws([&] { compress_with_plan(numeric->view(), "input -> onpair\n"); }),
         "onpair: reject non-STRING");
}

void test_inputs(::cuda::stream_ref stream, rmm::device_async_resource_ref mr)
{
  std::vector<std::string> mixed;
  std::vector<bool> valid;
  for (int i = 0; i < 10000; ++i) {
    switch (i % 7) {
      case 0: mixed.emplace_back(""); break;
      case 1: mixed.emplace_back("https://example.com/path/" + std::to_string(i)); break;
      case 2: mixed.emplace_back("a repeated dictionary token a repeated dictionary token"); break;
      case 3: mixed.emplace_back("UTF-8: \xc3\xa9 \xe6\x97\xa5 \xf0\x9f\x8c\x8d"); break;
      case 4: mixed.emplace_back("zero\0byte\xff", 10); break;
      case 5: mixed.emplace_back(i % 257, 'x'); break;
      default: mixed.emplace_back("short"); break;
    }
    valid.push_back(i % 11 != 0);
  }
  std::string long_row;
  for (int i = 0; i < 20480; ++i)
    long_row.push_back(static_cast<char>((i * 17 + i / 97) % 256));

  for (std::string const op : {"onpair_9", "onpair", "onpair_16"}) {
    auto const plan = "input -> " + op + "\n";
    for (auto const& values : std::vector<std::vector<std::string>>{
           {}, {"", "", ""}, {"one row"}, mixed, {"before", long_row, "", long_row, "after"}}) {
      auto input = make_strings_table(values, {}, stream);
      roundtrip(input->view(), plan, stream, mr);
    }
    std::vector<std::unique_ptr<cudf::column>> columns;
    columns.push_back(cudf::make_empty_column(cudf::data_type{cudf::type_id::STRING}));
    cudf::table childless(std::move(columns));
    roundtrip(childless.view(), plan, stream, mr);
    auto empty_rep =
      simpatico::make_compressor(op)->compress(childless.view().column(0), stream, mr);
    auto const empty_channels = empty_rep->named_channels(stream);
    check_named_channels(*empty_rep, cudf::type_id::INT32, stream);
    expect(empty_channels.size() == 5 && empty_channels[0].view.size() == 0 &&
             empty_channels[1].view.size() == 1 && empty_channels[2].view.size() == 0 &&
             empty_channels[3].view.size() == 1 && empty_channels[4].view.size() == 1,
           "onpair: canonical empty representation");
    std::uint64_t empty_prefix = 1;
    simpatico::read_device_bytes_completed(
      &empty_prefix, empty_channels[4].view.data<std::uint64_t>(), sizeof(empty_prefix), stream);
    expect(empty_prefix == 0, "onpair: canonical empty batch byte prefix is zero");

    auto nullable = make_strings_table(mixed, valid, stream);
    roundtrip(nullable->view(), plan, stream, mr);
    auto all_null = make_strings_table({"a", "b", "c"}, {false, false, false}, stream);
    roundtrip(all_null->view(), plan, stream, mr);

    auto full =
      make_strings_table({"excluded", "", "abc", long_row, "tail", "excluded"}, {}, stream);
    for (auto const& slice : cudf::slice(full->view().column(0), {0, 4, 1, 5, 2, 2}))
      roundtrip(cudf::table_view{{slice}}, plan, stream, mr);

    // Metadata belongs to the parent even when all storage is in its channels.
    auto const cascade =
      "input -> " + op + " -> codes, row_offsets, dict_bytes, dict_offsets, batch_prefixes\n" + op +
      ".row_offsets -> delta -> differences\n" + op + ".row_offsets.differences -> bitpack\n";
    auto input         = make_strings_table(mixed, valid, stream);
    roundtrip(input->view(), cascade, stream, mr);
    auto rep = simpatico::make_compressor(op)->compress(input->view().column(0), stream, mr);
    check_named_channels(*rep, cudf::type_id::INT32, stream);
  }
}

// Force the large-string offset path with small storage. This tests encode64
// and INT64 byte offsets without requiring a >2 GiB allocation in CTest.
void test_int64_offsets(::cuda::stream_ref stream, rmm::device_async_resource_ref mr)
{
  std::vector<std::int64_t> host{0, 1, 1, 4, 9};
  auto offsets = cudf::make_fixed_width_column(
    cudf::data_type{cudf::type_id::INT64}, host.size(), cudf::mask_state::UNALLOCATED, stream, mr);
  cuda_check(cudaMemcpyAsync(offsets->mutable_view().head<void>(),
                             host.data(),
                             host.size() * 8,
                             cudaMemcpyHostToDevice,
                             stream.get()));
  std::string chars = "aBCD12345";
  rmm::device_buffer bytes(chars.data(), chars.size(), stream, mr);
  auto input =
    cudf::make_strings_column(4,
                              std::move(offsets),
                              std::move(bytes),
                              0,
                              cudf::create_null_mask(0, cudf::mask_state::UNALLOCATED, stream, mr));
  stream.sync();
  for (std::string const op : {"onpair_9", "onpair", "onpair_16"}) {
    roundtrip(cudf::table_view{{input->view()}}, "input -> " + op + "\n", stream, mr);
    auto codec    = simpatico::make_compressor(op);
    auto rep      = codec->compress(input->view(), stream, mr);
    auto channels = rep->named_channels(stream);
    check_named_channels(*rep, cudf::type_id::INT64, stream);
    expect(channels[0].view.type().id() == cudf::type_id::UINT16, "onpair: UINT16 codes");
    expect(channels[1].view.type().id() == cudf::type_id::INT64, "onpair: INT64 byte offsets");
    for (auto const& slice : cudf::slice(input->view(), {0, 2, 1, 4})) {
      roundtrip(cudf::table_view{{slice}}, "input -> " + op + "\n", stream, mr);
      auto sliced = codec->compress(slice, stream, mr);
      expect(sliced->named_channels(stream)[1].view.type().id() == cudf::type_id::INT64,
             "onpair: sliced INT64 byte offsets");
    }
  }
}

void test_nonzero_first_offset(::cuda::stream_ref stream, rmm::device_async_resource_ref mr)
{
  auto check = [&]<typename Offset>(cudf::type_id type) {
    auto offsets            = upload_channel<Offset>({2, 3, 5}, type, stream, mr);
    std::string const chars = "__abc";
    rmm::device_buffer bytes(chars.data(), chars.size(), stream, mr);
    auto input = cudf::make_strings_column(
      2,
      std::move(offsets),
      std::move(bytes),
      0,
      cudf::create_null_mask(0, cudf::mask_state::UNALLOCATED, stream, mr));
    expect(
      input->view().offset() == 0 && cudf::strings_column_view(input->view()).offsets().size() == 3,
      "onpair: nonzero-origin fixture must bypass slice normalization");
    auto rep = simpatico::onpair_compressor{}.compress(input->view(), stream, mr);
    check_named_channels(*rep, type, stream);
    auto const channels = rep->named_channels(stream);
    std::vector<Offset> actual(3);
    simpatico::read_device_bytes_completed(actual.data(),
                                           channels[1].view.template data<Offset>(),
                                           actual.size() * sizeof(Offset),
                                           stream);
    expect(actual == std::vector<Offset>{0, 1, 3},
           "onpair: store normalized byte offsets for a nonzero first boundary");
    expect(std::get<simpatico::leaf_meta::onpair>(rep->describe_meta()).decoded_bytes == 3,
           "onpair: decoded_bytes excludes bytes before the first boundary");
    std::string error;
    auto decoded  = simpatico::decompress_standalone_representation(rep.get(), stream, mr, &error);
    auto expected = make_strings_column({"a", "bc"}, {}, stream);
    expect(decoded && error.empty() && strings_equal(expected->view(), decoded->view(), stream),
           "onpair: nonzero-origin round trip must be byte-exact");
    compare_offsets(expected->view(), decoded->view(), stream);
  };
  check.template operator()<std::int32_t>(cudf::type_id::INT32);
  check.template operator()<std::int64_t>(cudf::type_id::INT64);
}

// A controlled compact dictionary guarantees the rare 9..16-byte gather and
// exact batch/tile boundaries are exercised, independent of encoder training.
void test_kernel_boundaries(::cuda::stream_ref stream, rmm::device_async_resource_ref mr)
{
  std::vector<std::string> const dict{"a", "12345678", "123456789", "0123456789abcdef"};
  std::vector<std::uint8_t> bytes;
  std::vector<std::uint32_t> dict_offsets{0};
  for (auto const& token : dict) {
    bytes.insert(bytes.end(), token.begin(), token.end());
    dict_offsets.push_back(bytes.size());
  }
  auto upload = [&]<typename T>(std::vector<T> const& host, cudf::type_id type) {
    auto col = cudf::make_fixed_width_column(
      cudf::data_type{type}, host.size(), cudf::mask_state::UNALLOCATED, stream, mr);
    if (!host.empty())
      cuda_check(cudaMemcpyAsync(col->mutable_view().template head<void>(),
                                 host.data(),
                                 host.size() * sizeof(T),
                                 cudaMemcpyHostToDevice,
                                 stream.get()));
    return col;
  };
  for (int count : {1, 127, 128, 129, 2047, 2048, 2049, 8193}) {
    std::vector<std::uint16_t> codes(count);
    std::vector<std::string> rows(3);
    for (int i = 0; i < count; ++i) {
      codes[i] = i % dict.size();
      rows[i < count / 2 ? 0 : 2] += dict[codes[i]];
    }
    std::vector<std::int32_t> row_offsets{
      0,
      static_cast<std::int32_t>(rows[0].size()),
      static_cast<std::int32_t>(rows[0].size()),
      static_cast<std::int32_t>(rows[0].size() + rows[2].size())};  // byte offsets
    auto const prefixes = batch_prefixes(codes, dict_offsets);
    std::vector<std::unique_ptr<cudf::column>> channels;
    channels.push_back(upload(codes, cudf::type_id::UINT16));
    channels.push_back(upload(row_offsets, cudf::type_id::INT32));
    channels.push_back(upload(bytes, cudf::type_id::UINT8));
    channels.push_back(upload(dict_offsets, cudf::type_id::UINT32));
    channels.push_back(upload(prefixes, cudf::type_id::UINT64));
    simpatico::onpair_compressed_representation rep(
      simpatico::leaf_meta::onpair{3, rows[0].size() + rows[2].size(), 12},
      std::move(channels),
      stream,
      mr);
    stream.sync();
    auto expected = make_strings_column(rows, {}, stream);
    for (int scan = 0; scan < 2; ++scan) {
      std::string error;
      auto decoded = simpatico::decompress_standalone_representation(&rep, stream, mr, &error);
      expect(decoded && strings_equal(expected->view(), decoded->view(), stream),
             "onpair: batch boundary or split dictionary gather mismatch");
      compare_offsets(expected->view(), decoded->view(), stream);
    }
  }
}

// The controlled stream gives exact token counts independent of dictionary training.
void test_k6_batches(::cuda::stream_ref stream, rmm::device_async_resource_ref mr)
{
  std::vector<std::string> const dictionary{"x", "12345678", "123456789", "0123456789abcdef", "z"};
  std::vector<std::uint8_t> bytes;
  std::vector<std::uint32_t> offsets{0};
  for (auto const& token : dictionary) {
    bytes.insert(bytes.end(), token.begin(), token.end());
    offsets.push_back(bytes.size());
  }
  for (int n : {1, 31, 32, 191, 192, 193, 383, 384, 385, 2047, 2048, 8193}) {
    std::vector<std::uint16_t> codes(n);
    std::vector<std::string> rows(4);
    for (int i = 0; i < n; ++i) {
      // A full batch of long tokens exhausts all six rounds of high requests.
      codes[i] = i < 192 ? 3 : i % dictionary.size();
      rows[i < 31 ? 0 : 2] += dictionary[codes[i]];
    }
    auto oracle                                 = make_strings_column(rows, {}, stream);
    std::vector<std::int64_t> const row_offsets = byte_offsets(oracle->view(), stream);
    for (bool wide : {false, true}) {
      std::unique_ptr<cudf::column> baseline;
      for (std::uint16_t width : {128, 192}) {
        auto const expected = batch_prefixes(codes, offsets, width);
        std::vector<std::unique_ptr<cudf::column>> channels;
        channels.push_back(upload_channel(codes, cudf::type_id::UINT16, stream, mr));
        channels.push_back(
          wide ? upload_channel(row_offsets, cudf::type_id::INT64, stream, mr)
               : upload_channel(std::vector<std::int32_t>(row_offsets.begin(), row_offsets.end()),
                                cudf::type_id::INT32,
                                stream,
                                mr));
        channels.push_back(upload_channel(bytes, cudf::type_id::UINT8, stream, mr));
        channels.push_back(upload_channel(offsets, cudf::type_id::UINT32, stream, mr));
        channels.push_back(upload_channel(std::vector<std::uint64_t>((n + width - 1) / width + 1),
                                          cudf::type_id::UINT64,
                                          stream,
                                          mr));
        simpatico::leaf_meta::onpair meta{
          rows.size(), static_cast<std::uint64_t>(row_offsets.back()), 16};
        meta.tokens_per_batch = width;
        simpatico::onpair_compressed_representation rep(
          meta, std::move(channels), stream, mr, true);
        auto const stored = rep.named_channels(stream)[4].view;
        std::vector<std::uint64_t> actual(stored.size());
        simpatico::read_device_bytes_completed(
          actual.data(), stored.data<std::uint64_t>(), actual.size() * sizeof(actual[0]), stream);
        expect(actual == expected, "onpair: every batch prefix equals the host length sum");
        std::string error;
        auto decoded = simpatico::decompress_standalone_representation(&rep, stream, mr, &error);
        expect(decoded && error.empty() && strings_equal(oracle->view(), decoded->view(), stream),
               "onpair: exact token count round trip");
        compare_offsets(oracle->view(), decoded->view(), stream);
        if (width == 128)
          baseline = std::move(decoded);
        else
          expect(strings_equal(baseline->view(), decoded->view(), stream),
                 "onpair: K=6 and K=4 output bytes agree");
      }
    }
  }
}

void cross_gate(bool write,
                char const* path,
                ::cuda::stream_ref stream,
                rmm::device_async_resource_ref mr)
{
  std::vector<std::string> rows(1025);
  for (std::size_t i = 0; i < rows.size(); ++i)
    rows[i] = "id/" + std::to_string(i);
  auto input = make_strings_table(rows, {}, stream);
  // Exercise both leaf and parent metadata with a prepared row-offset subplan.
  cudf::table_view inputs{{input->view().column(0), input->view().column(0)}};
  std::string error;
  auto table = write ? simpatico::compress_with_plan(
                         inputs,
                         "input -> onpair_16\n---\ninput -> onpair_16 -> codes, row_offsets, "
                         "dict_bytes, dict_offsets, batch_prefixes\n"
                         "onpair_16.row_offsets -> delta -> "
                         "differences\nonpair_16.row_offsets.differences -> bitpack\n",
                         stream,
                         mr)
                     : simpatico::read_compressed_table(path, stream, mr, &error);
  expect(error.empty() && table.num_columns() == 2, "onpair: cross-gate table");
  int const k = simpatico::onpair_compression_k();
  for (auto const& column : table.columns) {
    auto const& meta = std::get<simpatico::leaf_meta::onpair>(column.plan_tree->nodes[1].meta);
    expect(meta.tokens_per_batch == 32 * (write ? k : 10 - k),
           "onpair: stored width must follow the writer");
  }
  if (write) {
    error = simpatico::write_compressed_table(table, path, stream);
    expect(error.empty(), error.c_str());
  } else {
    compare(inputs, table, stream, mr);
  }
}

// Check stored byte offsets directly, including empty rows around long rows.
// A controlled dictionary makes single-token and multi-batch rows deterministic.
void test_row_byte_offsets(::cuda::stream_ref stream, rmm::device_async_resource_ref mr)
{
  // Two token lengths with period 3 (128 is not a multiple of 3), so every
  // 128-token batch has a different byte total from its neighbours and a row's
  // bytes depend on exactly which codes it spans. Rows cross batch boundaries at
  // unaligned positions; the 123-token pad aligns the stream so the 128- and
  // 256-token rows start and end on batch boundaries, the 130-token row starts
  // aligned and ends unaligned, the 129-token row starts aligned, and the final
  // 255-token row starts unaligned and ends on an aligned total (2944 = 23*128).
  std::string const long_token  = "0123456789abcdef";
  std::string const short_token = "xyz";
  std::vector<std::uint8_t> dict(long_token.begin(), long_token.end());
  dict.insert(dict.end(), short_token.begin(), short_token.end());
  std::vector<std::uint16_t> codes;
  std::vector<std::int64_t> byte_offsets{0};
  std::vector<std::string> rows;
  for (int tokens :
       {0, 1, 0, 129, 0, 385, 1, 0, 1280, 0, 1, 0, 123, 128, 256, 130, 126, 129, 0, 255}) {
    std::string row;
    for (int i = 0; i < tokens; ++i) {
      auto const code = static_cast<std::uint16_t>(codes.size() % 3 == 0 ? 1 : 0);
      codes.push_back(code);
      row += code == 0 ? long_token : short_token;
    }
    byte_offsets.push_back(byte_offsets.back() + row.size());
    rows.push_back(std::move(row));
  }
  auto expected = make_strings_column(rows, {}, stream);
  auto check    = [&]<typename Offset>(cudf::type_id type) {
    std::vector<Offset> const host_bytes(byte_offsets.begin(), byte_offsets.end());
    std::vector<std::unique_ptr<cudf::column>> channels;
    channels.push_back(upload_channel(codes, cudf::type_id::UINT16, stream, mr));
    channels.push_back(upload_channel(host_bytes, type, stream, mr));
    channels.push_back(upload_channel(dict, cudf::type_id::UINT8, stream, mr));
    channels.push_back(
      upload_channel<std::uint32_t>({0, 16, 19}, cudf::type_id::UINT32, stream, mr));
    channels.push_back(
      upload_channel(batch_prefixes(codes, {0, 16, 19}), cudf::type_id::UINT64, stream, mr));
    simpatico::onpair_compressed_representation rep(
      simpatico::leaf_meta::onpair{
        rows.size(), static_cast<std::uint64_t>(byte_offsets.back()), 12},
      std::move(channels),
      stream,
      mr);
    for (int scan = 0; scan < 2; ++scan) {
      std::string error;
      auto decoded = simpatico::decompress_standalone_representation(&rep, stream, mr, &error);
      expect(decoded && error.empty(), "onpair: row-byte-offset fixture decode");
      expect(strings_equal(expected->view(), decoded->view(), stream),
             "onpair: mixed row bytes mismatch");
      compare_offsets(expected->view(), decoded->view(), stream);
      auto const offsets = cudf::strings_column_view(decoded->view()).offsets();
      expect(offsets.type().id() == type && offsets.size() == host_bytes.size(),
             "onpair: decoded byte-offset width or count");
      std::vector<Offset> actual(host_bytes.size());
      simpatico::read_device_bytes_completed(
        actual.data(), offsets.data<Offset>(), actual.size() * sizeof(Offset), stream);
      expect(actual == host_bytes, "onpair: decoded byte offsets differ from host prefix sum");
    }
  };
  check.template operator()<std::int32_t>(cudf::type_id::INT32);
  check.template operator()<std::int64_t>(cudf::type_id::INT64);
}

// Every corrupt channel is rejected as an ordinary exception; a subsequent
// valid decode in this same CUDA context must still work.
void test_corruption(::cuda::stream_ref stream, rmm::device_async_resource_ref mr)
{
  auto valid      = make_strings_table({"valid after rejection", "", "again"}, {}, stream);
  auto compressed = simpatico::compress_with_plan(valid->view(), "input -> onpair\n", stream, mr);
  auto reject     = [&](std::vector<std::uint16_t> codes,
                    std::vector<std::int32_t> rows,
                    std::vector<std::uint8_t> bytes,
                    std::vector<std::uint32_t> dictionary,
                    std::uint64_t decoded_bytes,
                    std::vector<std::uint64_t> prefixes = {},
                    std::string_view diagnostic         = {}) {
    if (prefixes.empty()) {
      prefixes.resize((codes.size() + 127) / 128 + 1);
      if (!codes.empty()) prefixes.back() = decoded_bytes;
    }
    std::vector<std::unique_ptr<cudf::column>> channels;
    channels.push_back(upload_channel(codes, cudf::type_id::UINT16, stream, mr));
    channels.push_back(upload_channel(rows, cudf::type_id::INT32, stream, mr));
    channels.push_back(upload_channel(bytes, cudf::type_id::UINT8, stream, mr));
    channels.push_back(upload_channel(dictionary, cudf::type_id::UINT32, stream, mr));
    channels.push_back(upload_channel(prefixes, cudf::type_id::UINT64, stream, mr));
    expect(throws<std::invalid_argument>([&] {
             try {
               simpatico::onpair_compressed_representation rep(
                 simpatico::leaf_meta::onpair{rows.size() - 1, decoded_bytes, 12},
                 std::move(channels),
                 stream,
                 mr);
             } catch (std::invalid_argument const& e) {
               expect(diagnostic.empty() || std::string_view{e.what()} == diagnostic, e.what());
               throw;
             }
           }),
           "onpair: corrupt channels must throw an ordinary host exception");
    cuda_check(cudaGetLastError());
    compare(valid->view(), compressed, stream, mr);
  };
  reject({0}, {0, 1}, {'x'}, {0, 0}, 1);  // empty dictionary entry
  reject({0}, {0, 1}, {'x'}, {0, 2}, 1);  // dictionary gather out of bounds
  reject({0}, {0, 1}, {'x'}, {1, 1}, 1);  // nonzero dictionary origin
  // unreferenced trailing bytes beyond the bridge's 16-byte read padding
  reject({0}, {0, 1}, std::vector<std::uint8_t>(1 + 17, 'x'), {0, 1}, 1);
  reject({0}, {0, 1}, std::vector<std::uint8_t>(17, 'x'), {0, 17}, 1);
  reject({1}, {0, 1}, {'x'}, {0, 1}, 1);           // code outside dictionary
  reject({0}, {0, 1}, {'x'}, {0, 1}, 2);           // decoded size mismatch
  reject({0}, {0, 2}, {'x'}, {0, 1}, 1);           // row beyond decoded bytes
  reject({0}, {0, -1, 1}, {'x'}, {0, 1}, 1);       // negative row boundary
  reject({0, 0}, {0, 2, 1, 2}, {'x'}, {0, 1}, 2);  // decreasing boundaries
  reject({}, {0, 1}, {}, {0}, 0);
  reject({}, {0, -1}, {}, {0}, 0);
  reject({}, {0, 1, 0}, {}, {0}, 0);
  reject({}, {1, 0}, {}, {0}, 0);
  reject({}, {0, 0}, {}, {0}, 1);
  reject({}, {0, 0}, {}, {1}, 0);
  // Correct byte rows and total, but an interior prefix is off by one while
  // remaining monotone. Merely checking order would miss this unsafe table.
  reject(std::vector<std::uint16_t>(129, 0), {0, 129}, {'x'}, {0, 1}, 129, {0, 127, 129});
  reject({0}, {0, 1}, {'x'}, {0, 1}, 1, {1, 1});  // nonzero batch byte origin
  reject({0}, {0, 1}, {'x'}, {0, 1}, 1, {0, 2});  // wrong batch byte total
  // This must hit the width/total guard before the unrelated token-count guard,
  // without allocating multiple GiB for otherwise consistent codes and chars.
  reject({0},
         {0, std::numeric_limits<std::int32_t>::max()},
         {'x'},
         {0, 1},
         std::uint64_t{1} << 31,
         {},
         "onpair: INT32 row byte offsets cannot represent decoded_bytes above INT32_MAX");

  for (auto const& rows :
       std::vector<std::vector<std::int64_t>>{{0, 1}, {0, -1}, {0, 1, 0}, {1, 0}}) {
    std::vector<std::unique_ptr<cudf::column>> channels;
    channels.push_back(upload_channel<std::uint16_t>({}, cudf::type_id::UINT16, stream, mr));
    channels.push_back(upload_channel(rows, cudf::type_id::INT64, stream, mr));
    channels.push_back(upload_channel<std::uint8_t>({}, cudf::type_id::UINT8, stream, mr));
    channels.push_back(upload_channel<std::uint32_t>({0}, cudf::type_id::UINT32, stream, mr));
    channels.push_back(upload_channel<std::uint64_t>({0}, cudf::type_id::UINT64, stream, mr));
    expect(
      throws<std::invalid_argument>([&] {
        simpatico::onpair_compressed_representation rep(
          simpatico::leaf_meta::onpair{rows.size() - 1, 0, 9}, std::move(channels), stream, mr);
      }),
      "onpair: reject malformed INT64 empty-channel boundaries");
    cuda_check(cudaGetLastError());
    compare(valid->view(), compressed, stream, mr);
  }

  // The same validation is applied to persisted compact channels, both with
  // a self representation and with a row-byte-offset sub-plan.
  for (std::string const plan :
       {"input -> onpair\n",
        "input -> onpair -> codes, row_offsets, dict_bytes, dict_offsets, batch_prefixes\n"
        "onpair.row_offsets -> delta -> differences\n"
        "onpair.row_offsets.differences -> bitpack\n"}) {
    for (bool const corrupt_prefix : {false, true}) {
      auto stored = simpatico::compress_with_plan(valid->view(), plan, stream, mr);
      std::vector<std::uint8_t> header;
      std::vector<simpatico::payload_buffer_ref> buffers;
      std::uint64_t size = 0;
      auto error = simpatico::build_compressed_table_header(stored, header, buffers, size, stream);
      expect(error.empty(), error.c_str());
      std::vector<std::uint8_t> payload(size);
      auto const& node             = stored.columns[0].plan_tree->nodes[1];
      auto const channel_name      = corrupt_prefix ? "batch_prefixes" : "codes";
      auto const channel_index     = corrupt_prefix ? 4 : 0;
      auto const channel           = node.rep
                                       ? node.rep->named_channels(stream)[channel_index].view
                                       : simpatico::terminal_identity_channel(node, channel_name)->view();
      void const* target           = channel.head<void>();
      std::uint64_t corrupt_offset = size;
      for (auto const& buffer : buffers) {
        if (buffer.size_bytes)
          cuda_check(cudaMemcpyAsync(payload.data() + buffer.offset,
                                     buffer.device_ptr,
                                     buffer.size_bytes,
                                     cudaMemcpyDeviceToHost,
                                     stream.get()));
        if (buffer.device_ptr == target) corrupt_offset = buffer.offset;
      }
      stream.sync();
      expect(corrupt_offset + 1 < size, "onpair: find persisted compact channel");
      if (corrupt_prefix) {
        // An off-by-one final decoded byte prefix remains monotone.
        auto const last = corrupt_offset + (channel.size() - 1) * sizeof(std::uint64_t);
        std::uint64_t total;
        std::memcpy(&total, payload.data() + last, sizeof(total));
        ++total;
        std::memcpy(payload.data() + last, &total, sizeof(total));
      } else {
        payload[corrupt_offset] = payload[corrupt_offset + 1] = 0xff;
      }
      auto fetch = [&](std::uint64_t off, std::size_t bytes, void* dst, ::cuda::stream_ref lane) {
        cuda_check(
          cudaMemcpyAsync(dst, payload.data() + off, bytes, cudaMemcpyHostToDevice, lane.get()));
      };
      auto rejected =
        simpatico::read_compressed_table_from_memory(header, fetch, stream, mr, &error);
      stream.sync();
      expect(!error.empty() && rejected.columns.empty(),
             "onpair: reject corrupt persisted codes or batch byte prefixes");
      cuda_check(cudaGetLastError());
      compare(valid->view(), compressed, stream, mr);
    }
  }
}

void test_graph_rejection(::cuda::stream_ref stream, rmm::device_async_resource_ref mr)
{
  auto input = make_strings_table({"a", "bc"}, {}, stream);
  for (bool const cyclic : {true, false}) {
    auto stored = simpatico::compress_with_plan(
      input->view(),
      "input -> onpair -> codes, row_offsets, dict_bytes, dict_offsets, batch_prefixes\n"
      "onpair.row_offsets -> delta -> differences\n"
      "onpair.row_offsets.differences -> bitpack\n",
      stream,
      mr);
    auto& nodes = stored.columns[0].plan_tree->nodes;
    auto delta =
      std::find_if(nodes.begin(), nodes.end(), [](auto const& node) { return node.op == "delta"; });
    expect(delta != nodes.end() && delta->children.size() == 1,
           "onpair: graph rejection fixture must have a delta child");
    delta->children.front().child = cyclic ? static_cast<simpatico::NodeId>(delta - nodes.begin())
                                           : static_cast<simpatico::NodeId>(nodes.size());
    std::vector<std::uint8_t> header;
    std::vector<simpatico::payload_buffer_ref> buffers;
    std::uint64_t bytes = 0;
    auto error = simpatico::build_compressed_table_header(stored, header, buffers, bytes, stream);
    expect(error.empty(), error.c_str());
    std::vector<std::uint8_t> payload(bytes);
    for (auto const& buffer : buffers)
      if (buffer.size_bytes)
        cuda_check(cudaMemcpyAsync(payload.data() + buffer.offset,
                                   buffer.device_ptr,
                                   buffer.size_bytes,
                                   cudaMemcpyDeviceToHost,
                                   stream.get()));
    stream.sync();
    auto fetch = [&](std::uint64_t off, std::size_t size, void* dst, ::cuda::stream_ref lane) {
      cuda_check(
        cudaMemcpyAsync(dst, payload.data() + off, size, cudaMemcpyHostToDevice, lane.get()));
    };
    auto rejected = simpatico::read_compressed_table_from_memory(header, fetch, stream, mr, &error);
    stream.sync();
    expect(rejected.columns.empty(), "onpair: malformed graph header must be rejected");
    expect(error.find(cyclic ? "cyclic plan" : "child out of range") != std::string::npos,
           error.c_str());
    cuda_check(cudaGetLastError());
  }
}

void test_prepared_reconstruction(::cuda::stream_ref stream, rmm::device_async_resource_ref mr)
{
  for (bool const empty : {false, true}) {
    auto input = make_strings_table(
      empty ? std::vector<std::string>{"", "", ""} : std::vector<std::string>{"a", "", "bc"},
      {},
      stream);
    auto stored = simpatico::compress_with_plan(
      input->view(),
      "input -> onpair -> codes, row_offsets, dict_bytes, dict_offsets, batch_prefixes\n",
      stream,
      mr);
    auto& node = stored.columns[0].plan_tree->nodes[1];
    auto const* prepared =
      dynamic_cast<simpatico::onpair_compressed_representation const*>(node.prepared_rep.get());
    expect(prepared != nullptr, "onpair: dictionary must be prepared before scan");
    auto const total = empty ? 0 : 3;
    auto check_rows  = [&]<typename Offset>(cudf::type_id type) {
      // These byte offsets stand in for a malformed channel decoded by a child
      // plan: negative/decreasing, nonzero origin, wrong total, and non-monotone
      // with correct endpoints and (for nonempty input) entries within the byte bound.
      for (auto const& boundaries : std::vector<std::vector<Offset>>{
             {0, -1, 0, total}, {1, 1, 1, total}, {0, 0, 0, total + 1}, {0, 2, 1, total}}) {
        auto rows = upload_channel(boundaries, type, stream, mr);
        simpatico::onpair_compressed_representation rebuilt(*prepared, std::move(rows));
        expect(rebuilt.channels_.size() == 5 && !rebuilt.channels_[0] && rebuilt.channels_[1] &&
                 !rebuilt.channels_[2] && !rebuilt.channels_[3] && !rebuilt.channels_[4],
               "onpair: reconstruction owns only row byte offsets");
        expect(rebuilt.resident_size_bytes(stream) == rebuilt.channels_[1]->alloc_size(),
               "onpair: reconstruction must not count borrowed channels or dictionary planes");
        check_named_channels(rebuilt, type, stream);
        auto const views = rebuilt.named_channels(stream);
        for (std::size_t i : {0u, 2u, 3u, 4u})
          expect(views[i].view.head<void>() ==
                   simpatico::terminal_identity_channel(node, views[i].name)->view().head<void>(),
                 "onpair: reconstruction must borrow unchanged channels");
        simpatico::decode_session session{{&stream, 1}, mr};
        decode_observation::calls  = 0;
        decode_observation::active = true;
        session.append(simpatico::column_decode_request{std::cref(rebuilt)});
        decode_observation::active = false;
        expect(decode_observation::calls == 0,
               "onpair: append must not wait for device validation");
        expect(throws<std::invalid_argument>([&] { (void)session.finish(); }),
               "onpair: row-byte-offset status must surface at session completion");
        cuda_check(cudaGetLastError());

        // Feed the same malformed offsets through the walk, where equals_any
        // consumes strings before finish() observes the retained status word.
        // An empty-string predicate must see only zero-length, in-bounds spans.
        auto& row_channel  = node.channels.at(node.output_paths[1]);
        auto original_rows = std::move(row_channel);
        row_channel        = std::make_unique<simpatico::identity_compressed_representation>(
          std::make_unique<cudf::column>(rebuilt.channels_[1]->view(), stream, mr));
        auto const words = sirius::codegen::selection_mask::AllocWordsFor(3);
        rmm::device_buffer ballot(words * sizeof(std::uint32_t), stream, mr);
        cuda_check(cudaMemsetAsync(ballot.data(), 0, ballot.size(), stream.get()));
        simpatico::decode_session consuming_session{{&stream, 1}, mr};
        consuming_session.append(simpatico::column_decode_request{
          .source = std::cref(*stored.columns[0].plan_tree),
          .result = simpatico::predicate_result{
            simpatico::decode_predicate{{""}},
            simpatico::mask_destination{static_cast<std::uint32_t*>(ballot.data()), 3}}});
        expect(throws<std::invalid_argument>([&] { (void)consuming_session.finish(); }),
               "onpair: same-request predicate must retain the deferred offset exception");
        std::uint32_t matches = 0;
        simpatico::read_device_bytes_completed(&matches, ballot.data(), sizeof(matches), stream);
        expect(matches == 0b111u,
               "onpair: same-request predicate must see three empty spans after offset rejection");
        row_channel = std::move(original_rows);
        cuda_check(cudaGetLastError());
      }
    };
    check_rows.template operator()<std::int32_t>(cudf::type_id::INT32);
    check_rows.template operator()<std::int64_t>(cudf::type_id::INT64);
    compare(input->view(), stored, stream, mr);
  }

  // Zero rows have exactly one boundary. Both signed widths must validate it on the device.
  auto input  = std::make_unique<cudf::table>([] {
    std::vector<std::unique_ptr<cudf::column>> columns;
    columns.push_back(cudf::make_empty_column(cudf::data_type{cudf::type_id::STRING}));
    return columns;
  }());
  auto stored = simpatico::compress_with_plan(
    input->view(),
    "input -> onpair -> codes, row_offsets, dict_bytes, dict_offsets, batch_prefixes\n",
    stream,
    mr);
  auto const* prepared = dynamic_cast<simpatico::onpair_compressed_representation const*>(
    stored.columns[0].plan_tree->nodes[1].prepared_rep.get());
  expect(prepared != nullptr, "onpair: zero-row dictionary must be prepared");
  auto check_empty = [&]<typename Offset>(cudf::type_id type) {
    for (Offset const boundary : {Offset{0}, Offset{1}, Offset{-1}}) {
      auto rows = upload_channel<Offset>({boundary}, type, stream, mr);
      simpatico::onpair_compressed_representation rebuilt(*prepared, std::move(rows));
      simpatico::decode_session session{{&stream, 1}, mr};
      decode_observation::calls  = 0;
      decode_observation::active = true;
      session.append(simpatico::column_decode_request{std::cref(rebuilt)});
      decode_observation::active = false;
      expect(decode_observation::calls == 0,
             "onpair: zero-row append must not wait for validation");
      if (boundary == 0) {
        auto results = session.finish();
        expect(results.size() == 1 && results[0]->size() == 0,
               "onpair: canonical zero-row reconstruction must succeed");
      } else {
        expect(throws<std::invalid_argument>([&] { (void)session.finish(); }),
               "onpair: nonzero sole boundary must fail at finish");
      }
      cuda_check(cudaGetLastError());
    }
  };
  check_empty.template operator()<std::int32_t>(cudf::type_id::INT32);
  check_empty.template operator()<std::int64_t>(cudf::type_id::INT64);
  compare(input->view(), stored, stream, mr);
}

void test_code_65535(::cuda::stream_ref stream,
                     rmm::device_async_resource_ref mr,
                     std::uint16_t width = 128)
{
  // At K=6, code 65535 needs both halves. 33 long codes exercise the held first
  // round of high requests and the subsequent queue drain in the same warp.
  std::string const last = width == 192 ? "0123456789abcdef" : "Z";
  std::vector<std::uint8_t> bytes(65535, 'x');
  bytes.insert(bytes.end(), last.begin(), last.end());
  std::vector<std::uint32_t> offsets(65537);
  for (std::uint32_t i = 0; i < offsets.size(); ++i)
    offsets[i] = i;
  offsets.back() = bytes.size();
  std::vector<std::uint16_t> codes(width == 192 ? 34 : 3, 65535);
  codes[1]               = 0;
  std::string second_row = "x";
  for (std::size_t i = 2; i < codes.size(); ++i)
    second_row += last;
  auto const total = last.size() + second_row.size();
  std::vector<std::unique_ptr<cudf::column>> channels;
  channels.push_back(upload_channel(codes, cudf::type_id::UINT16, stream, mr));
  channels.push_back(upload_channel<std::int32_t>(
    {0, static_cast<std::int32_t>(last.size()), static_cast<std::int32_t>(total)},
    cudf::type_id::INT32,
    stream,
    mr));
  channels.push_back(upload_channel(bytes, cudf::type_id::UINT8, stream, mr));
  channels.push_back(upload_channel(offsets, cudf::type_id::UINT32, stream, mr));
  channels.push_back(upload_channel<std::uint64_t>({0, total}, cudf::type_id::UINT64, stream, mr));
  std::string error;
  simpatico::leaf_meta::onpair metadata{2, total, 16};
  metadata.tokens_per_batch = width;
  auto rep                  = simpatico::reconstruct_representation(
    "onpair_16",
    {"codes", "row_offsets", "dict_bytes", "dict_offsets", "batch_prefixes"},
    std::move(channels),
    stream,
    mr,
    &error,
    metadata);
  expect(rep != nullptr && error.empty(), "onpair_16: full dictionary construction");
  auto decoded  = simpatico::decompress_standalone_representation(rep.get(), stream, mr, &error);
  auto expected = make_strings_column({last, second_row}, {}, stream);
  expect(decoded && strings_equal(expected->view(), decoded->view(), stream),
         "onpair_16: code 65535 must decode without narrowing");
  compare_offsets(expected->view(), decoded->view(), stream);
}

// Opt-in round trip with >2 GiB of chars: byte offsets are INT64 at encode
// time, and decode must preserve every byte, including the interior of each row.
void test_int64_large_offsets(::cuda::stream_ref stream, rmm::device_async_resource_ref mr)
{
  constexpr std::int64_t decoded_bytes = (std::int64_t{1} << 31) + 16;
  auto offsets                         = upload_channel<std::int64_t>(
    {0, std::int64_t{1} << 30, decoded_bytes}, cudf::type_id::INT64, stream, mr);
  rmm::device_buffer chars(decoded_bytes, stream, mr);
  cuda_check(cudaMemsetAsync(chars.data(), 'Q', chars.size(), stream.get()));
  auto input =
    cudf::make_strings_column(2,
                              std::move(offsets),
                              std::move(chars),
                              0,
                              cudf::create_null_mask(0, cudf::mask_state::UNALLOCATED, stream, mr));
  auto rep = simpatico::onpair_compressor{9}.compress(input->view(), stream, mr);
  check_named_channels(*rep, cudf::type_id::INT64, stream);
  auto const stored_offsets = rep->named_channels(stream)[1].view;
  std::int64_t actual[3];
  simpatico::read_device_bytes_completed(
    actual, stored_offsets.data<std::int64_t>(), sizeof(actual), stream);
  expect(std::vector<std::int64_t>(actual, actual + 3) == byte_offsets(input->view(), stream),
         "onpair: stored byte offsets are INT64 input offsets before decode");
  // Isolate the byte-bound check with otherwise consistent large channels.
  {
    std::vector<std::unique_ptr<cudf::column>> narrow;
    for (auto const& channel : rep->named_channels(stream))
      narrow.push_back(std::make_unique<cudf::column>(channel.view, stream, mr));
    narrow[1] = upload_channel<std::int32_t>(
      {0, std::int32_t{1} << 30, std::numeric_limits<std::int32_t>::max()},
      cudf::type_id::INT32,
      stream,
      mr);
    auto const metadata = std::get<simpatico::leaf_meta::onpair>(rep->describe_meta());
    expect(throws<std::invalid_argument>([&] {
             simpatico::onpair_compressed_representation rejected(
               metadata, std::move(narrow), stream, mr);
           }),
           "onpair: INT32 byte offsets cannot represent a 64-bit decoded total");
    cuda_check(cudaGetLastError());
  }
  // Bound host verification storage while checking the whole decoded chars buffer.
  std::vector<std::uint8_t> source(8 * 1024 * 1024), output(source.size());
  std::string error;
  for (int scan = 0; scan < 2; ++scan) {
    auto decoded = simpatico::decompress_standalone_representation(rep.get(), stream, mr, &error);
    expect(decoded != nullptr && error.empty(), "onpair: large decode must succeed");
    auto const strings = cudf::strings_column_view(decoded->view());
    expect(strings.offsets().type().id() == cudf::type_id::INT64,
           "onpair: large decode preserves stored INT64 byte-offset type");
    compare_offsets(input->view(), decoded->view(), stream);
    for (std::uint64_t pos = 0; pos < decoded_bytes; pos += source.size()) {
      auto const count = std::min<std::uint64_t>(source.size(), decoded_bytes - pos);
      std::array<simpatico::device_read, 2> reads{
        {{source.data(), cudf::strings_column_view(input->view()).chars_begin(stream) + pos, count},
         {output.data(), strings.chars_begin(stream) + pos, count}}};
      simpatico::read_device_bytes_completed(reads, stream);
      expect(std::memcmp(source.data(), output.data(), count) == 0,
             "onpair: >2 GiB INT64 round trip must be byte-exact");
    }
  }
  // Select BOTH rows: each row fits INT32, but their aggregate requires INT64 output offsets.
  // CTest registers this fixture only with SIMPATICO_LARGE_MEMORY_TESTS enabled.
  auto parsed = simpatico::plan_tree_from_dsl("input -> onpair_16\n");
  expect(parsed.has_value(), "onpair: large selected fixture plan must parse");
  auto& plan         = *parsed;
  plan.nodes[1].meta = rep->describe_meta();
  plan.nodes[1].rep  = std::move(rep);
  namespace sc       = sirius::codegen;
  expect(simpatico::probe_column(plan).compact_route == sc::decode_route::onpair,
         "onpair: large selected fixture must exercise the selected route");
  auto indices = upload_channel<std::int32_t>({0, 1}, cudf::type_id::INT32, stream, mr);
  std::vector<std::uint32_t> words(sc::selection_mask::AllocWordsFor(2), 0);
  words[0]        = 3;
  auto mask_words = upload_channel(words, cudf::type_id::UINT32, stream, mr);
  auto counts     = upload_channel<std::uint32_t>({0, 2}, cudf::type_id::UINT32, stream, mr);
  sc::selection_mask mask{mask_words->mutable_view().data<std::uint32_t>(),
                          2,
                          2,
                          counts->mutable_view().data<std::uint32_t>()};
  simpatico::decode_selection selection;
  selection.mask             = &mask;
  selection.survivor_count   = 2;
  selection.survivor_indices = indices->view();
  selection.route            = sc::decode_route::onpair;
  auto full = simpatico::decode_one(simpatico::column_decode_request{std::cref(plan)}, stream, mr);
  auto expected = cudf::gather(cudf::table_view{{full->view()}},
                               indices->view(),
                               cudf::out_of_bounds_policy::DONT_CHECK,
                               stream,
                               mr);
  full.reset();
  simpatico::decode_session session{{&stream, 1}, mr};
  decode_observation::calls     = 0;
  decode_observation::readbacks = 0;
  decode_observation::active    = true;
  session.append(simpatico::column_decode_request{
    .source = std::cref(plan), .selection = simpatico::validated_selection(plan, selection)});
  decode_observation::active = false;
  expect(decode_observation::calls == 1 && decode_observation::readbacks == 1,
         "onpair: large selected append must have exactly one sizing readback/wait");
  decode_observation::active = true;
  auto selected              = session.finish();
  decode_observation::active = false;
  expect(decode_observation::readbacks == 1,
         "onpair: selected completion must not read a second status word");
  auto const selected_strings = cudf::strings_column_view(selected.front()->view());
  expect(selected_strings.offsets().type().id() == cudf::type_id::INT64,
         "onpair: >2 GiB selected output must have INT64 offsets");
  compare_offsets(expected->view().column(0), selected.front()->view(), stream);
  for (std::uint64_t pos = 0; pos < decoded_bytes; pos += source.size()) {
    auto const count = std::min<std::uint64_t>(source.size(), decoded_bytes - pos);
    std::array<simpatico::device_read, 2> reads{
      {{source.data(),
        cudf::strings_column_view(expected->view().column(0)).chars_begin(stream) + pos,
        count},
       {output.data(), selected_strings.chars_begin(stream) + pos, count}}};
    simpatico::read_device_bytes_completed(reads, stream);
    expect(std::memcmp(source.data(), output.data(), count) == 0,
           "onpair: >2 GiB selected output must match full decode plus gather byte-for-byte");
  }
}

}  // namespace

int main(int argc, char** argv)
{
  try {
    cuda_check(cudaSetDevice(0));
    test_registry();
    // Use a non-default stream and stream-ordered frees for all GPU cases.
    rmm::cuda_stream stream;
    rmm::mr::cuda_async_memory_resource resource;
    auto const mr = rmm::device_async_resource_ref{resource};
    if (argc == 2 && std::string_view{argv[1]} == "--k6-code-65535") {
      test_code_65535(stream, mr, 192);
      return 0;
    }
    if (argc == 2 && std::string_view{argv[1]} == "--k6") {
      expect(simpatico::onpair_compression_k() == 6, "onpair: K=6 test gate");
      test_k6_batches(stream, mr);
      test_inputs(stream, mr);
      test_int64_offsets(stream, mr);
      return 0;
    }
    if (argc == 3) {
      std::string_view const mode{argv[1]};
      if (mode == "--write-layout" || mode == "--read-other-layout") {
        cross_gate(mode == "--write-layout", argv[2], stream, mr);
        return 0;
      }
    }
    if (argc == 2 && std::string_view{argv[1]} == "--promoted-offsets") {
      std::puts("test_onpair_roundtrip: test_int64_large_offsets");
      test_int64_large_offsets(stream, mr);
      stream.synchronize();
      std::puts("test_onpair_roundtrip: PASS (large INT64 byte offsets)");
      return 0;
    }
    if (argc != 1) throw std::invalid_argument("expected no arguments or --promoted-offsets");
    std::puts("test_onpair_roundtrip: test_inputs");
    test_inputs(stream, mr);
    std::puts("test_onpair_roundtrip: test_int64_offsets");
    test_int64_offsets(stream, mr);
    std::puts("test_onpair_roundtrip: test_nonzero_first_offset");
    test_nonzero_first_offset(stream, mr);
    std::puts("test_onpair_roundtrip: test_kernel_boundaries");
    test_kernel_boundaries(stream, mr);
    std::puts("test_onpair_roundtrip: test_row_byte_offsets");
    test_row_byte_offsets(stream, mr);
    std::puts("test_onpair_roundtrip: test_corruption");
    test_corruption(stream, mr);
    std::puts("test_onpair_roundtrip: test_graph_rejection");
    test_graph_rejection(stream, mr);
    std::puts("test_onpair_roundtrip: test_prepared_reconstruction");
    test_prepared_reconstruction(stream, mr);
    std::puts("test_onpair_roundtrip: test_code_65535");
    test_code_65535(stream, mr);
    stream.synchronize();
    std::puts("test_onpair_roundtrip: PASS");
    return 0;
  } catch (std::exception const& error) {
    std::fprintf(stderr, "test_onpair_roundtrip: FAIL: %s\n", error.what());
    return 1;
  }
}
