// SPDX-License-Identifier: Apache-2.0
// C ABI over the onpair crate (src/lib.rs is the authoritative description).
#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

enum onpair_bridge_status {
  ONPAIR_BRIDGE_OK                   = 0,
  ONPAIR_BRIDGE_ERR_NULL_ARG         = 1,
  ONPAIR_BRIDGE_ERR_BAD_CONFIG       = 2,
  ONPAIR_BRIDGE_ERR_BAD_OFFSETS      = 3,
  ONPAIR_BRIDGE_ERR_ENCODE           = 4,
  ONPAIR_BRIDGE_ERR_BAD_DICT         = 5,
  ONPAIR_BRIDGE_ERR_OUTPUT_TOO_SMALL = 6,
  ONPAIR_BRIDGE_ERR_PANIC            = 7,
};

typedef struct onpair_bridge_config {
  uint8_t max_dict_bits;  // 9..16; 12 = OnPair-12, 16 = OnPair-16
  double threshold;       // (0, 1]: dynamic training-sample fraction (paper: 0.2)
  uint64_t seed;
  uint8_t seed_is_set;    // 0 = non-deterministic sampling
  uint32_t threads;       // parsing cap: 0 = available parallelism (default), 1 = serial
} onpair_bridge_config;

typedef struct onpair_bridge_column {
  const uint8_t* dict_bytes;    // read-padded compact dictionary bytes
  uint64_t dict_bytes_len;
  const uint32_t* dict_offsets; // num_tokens + 1 entries
  uint64_t dict_offsets_len;
  const uint16_t* codes;        // rows concatenated; every code < num_tokens
  uint64_t codes_len;
  const uint8_t* row_offsets;   // num_rows + 1 elements of row_offset_width bytes, offsets into codes
  uint64_t row_offsets_len;
  uint8_t row_offset_width;     // 4 (int32 input) or 8 (int64 input)
  void* owner_;                 // private
} onpair_bridge_column;

// The paper's configuration: 12-bit dictionary, 20% sample, seed 42.
onpair_bridge_config onpair_bridge_config_default(void);

// Train and encode a cuDF strings column (chars + int32 offsets, num_rows + 1 entries;
// offsets[0] may be non-zero). On success *out is an owned column; free with
// onpair_bridge_column_free.
int32_t onpair_bridge_encode32(const uint8_t* bytes, uint64_t bytes_len, const int32_t* offsets,
                               uint64_t num_rows, onpair_bridge_config cfg,
                               onpair_bridge_column** out);
// Same for a large-strings column (int64 offsets).
int32_t onpair_bridge_encode64(const uint8_t* bytes, uint64_t bytes_len, const int64_t* offsets,
                               uint64_t num_rows, onpair_bridge_config cfg,
                               onpair_bridge_column** out);
void onpair_bridge_column_free(onpair_bridge_column* col);

// CPU reference decode (tests and --verify). decoded_len validates the dictionary and
// returns the exact output size; decode fills out (capacity out_cap) and reports the
// bytes written, or ONPAIR_BRIDGE_ERR_OUTPUT_TOO_SMALL.
int32_t onpair_bridge_decoded_len(const uint8_t* dict_bytes, uint64_t dict_bytes_len,
                                  const uint32_t* dict_offsets, uint64_t dict_offsets_len,
                                  const uint16_t* codes, uint64_t codes_len, uint64_t* out_len);
int32_t onpair_bridge_decode(const uint8_t* dict_bytes, uint64_t dict_bytes_len,
                             const uint32_t* dict_offsets, uint64_t dict_offsets_len,
                             const uint16_t* codes, uint64_t codes_len, uint8_t* out,
                             uint64_t out_cap, uint64_t* written);

// e.g. "onpair 0.3.0"; static storage.
const char* onpair_bridge_version(void);

#ifdef __cplusplus
}
#endif
