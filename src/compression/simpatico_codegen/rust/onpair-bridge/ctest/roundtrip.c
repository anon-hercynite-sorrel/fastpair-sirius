// SPDX-License-Identifier: Apache-2.0
// Round-trip a newline-delimited text file through the bridge from C: encode, CPU decode,
// byte-compare, report ratio and timings. Usage: roundtrip <file> [bits] [threshold] [threads]
#define _POSIX_C_SOURCE 200809L
#include "onpair_bridge.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

static double now(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main(int argc, char** argv) {
  if (argc < 2) {
    fprintf(stderr, "usage: %s <file> [bits] [threshold] [threads]\n", argv[0]);
    return 2;
  }
  FILE* f = fopen(argv[1], "rb");
  if (!f) { perror("open"); return 2; }
  fseek(f, 0, SEEK_END);
  long n = ftell(f);
  fseek(f, 0, SEEK_SET);
  uint8_t* data = malloc(n);
  if (fread(data, 1, n, f) != (size_t)n) { perror("read"); return 2; }
  fclose(f);

  // rows = lines without their '\n'; chars = concatenation. Offsets are built in 64 bits;
  // the int32 (cuDF strings) path is used when they fit, the int64 (large strings) path
  // otherwise, so files past 2 GiB exercise encode64.
  uint8_t* chars = malloc(n ? n : 1);
  int64_t* offsets64 = malloc(sizeof(int64_t) * (n + 2));
  uint64_t rows = 0;
  int64_t pos = 0;
  offsets64[0] = 0;
  for (long i = 0; i < n;) {
    long j = i;
    while (j < n && data[j] != '\n') j++;
    memcpy(chars + pos, data + i, j - i);
    pos += j - i;
    offsets64[++rows] = pos;
    i = j + 1;
  }
  free(data);
  const int wide = pos > INT32_MAX;
  int32_t* offsets32 = NULL;
  if (!wide) {
    offsets32 = malloc(sizeof(int32_t) * (rows + 1));
    for (uint64_t k = 0; k <= rows; ++k) offsets32[k] = (int32_t)offsets64[k];
  }

  onpair_bridge_config cfg = onpair_bridge_config_default();
  if (argc > 2) cfg.max_dict_bits = (uint8_t)atoi(argv[2]);
  if (argc > 3) cfg.threshold = atof(argv[3]);
  if (argc > 4) cfg.threads = (uint32_t)strtoul(argv[4], NULL, 10);
  // Resolve the benchmark's auto cap explicitly so we can report actual workers.
  // The bridge itself uses Rust's available_parallelism() for threads == 0.
  if (cfg.threads == 0) {
    long cpus   = sysconf(_SC_NPROCESSORS_ONLN);
    cfg.threads = cpus > 0 ? (uint32_t)cpus : 1;
  }
  uint64_t chunks = rows / 65536;
  if (chunks == 0) chunks = 1;
  if (chunks < cfg.threads) cfg.threads = (uint32_t)chunks;

  onpair_bridge_column* col = NULL;
  double t0 = now();
  int32_t st = wide ? onpair_bridge_encode64(chars, pos, offsets64, rows, cfg, &col)
                    : onpair_bridge_encode32(chars, pos, offsets32, rows, cfg, &col);
  double t1 = now();
  if (st != ONPAIR_BRIDGE_OK) { fprintf(stderr, "encode failed: %d\n", st); return 1; }

  uint64_t dlen = 0;
  st = onpair_bridge_decoded_len(col->dict_bytes, col->dict_bytes_len, col->dict_offsets,
                                 col->dict_offsets_len, col->codes, col->codes_len, &dlen);
  if (st != ONPAIR_BRIDGE_OK || dlen != (uint64_t)pos) {
    fprintf(stderr, "decoded_len mismatch: st=%d len=%llu expected %lld\n", st,
            (unsigned long long)dlen, (long long)pos);
    return 1;
  }
  uint8_t* dec = malloc(pos ? pos : 1);
  uint64_t written = 0;
  double t2 = now();
  st = onpair_bridge_decode(col->dict_bytes, col->dict_bytes_len, col->dict_offsets,
                            col->dict_offsets_len, col->codes, col->codes_len, dec, pos, &written);
  double t3 = now();
  if (st != ONPAIR_BRIDGE_OK || written != (uint64_t)pos || memcmp(dec, chars, pos) != 0) {
    fprintf(stderr, "decode mismatch: st=%d written=%llu\n", st, (unsigned long long)written);
    return 1;
  }
  // row boundaries: code offsets -> byte lengths
  if (col->row_offsets_len != rows + 1 || col->row_offset_width != (wide ? 8 : 4)) {
    fprintf(stderr, "row_offsets shape wrong\n");
    return 1;
  }
  const uint32_t* ro32 = (const uint32_t*)col->row_offsets;
  const uint64_t* ro64 = (const uint64_t*)col->row_offsets;
  for (uint64_t k = 0; k < rows; ++k) {
    uint64_t len = 0;
    uint64_t c0 = wide ? ro64[k] : ro32[k], c1 = wide ? ro64[k + 1] : ro32[k + 1];
    for (uint64_t c = c0; c < c1; ++c) {
      uint16_t t = col->codes[c];
      len += col->dict_offsets[t + 1] - col->dict_offsets[t];
    }
    if (len != (uint64_t)(offsets64[k + 1] - offsets64[k])) {
      fprintf(stderr, "row %llu length mismatch\n", (unsigned long long)k);
      return 1;
    }
  }
  uint64_t comp = col->codes_len * 2 + col->dict_bytes_len + col->dict_offsets_len * 4 +
                  col->row_offsets_len * col->row_offset_width;
  printf(
    "%s: %s rows=%llu bytes=%lld (%s) bits=%u thr=%.2f threads=%u | encode %.2fs (%.0f MB/s) "
    "decode %.3fs "
    "(%.0f MB/s) ratio %.2f tokens=%llu dict=%llu B OK\n",
    argv[1],
    onpair_bridge_version(),
    (unsigned long long)rows,
    (long long)pos,
    wide ? "int64" : "int32",
    cfg.max_dict_bits,
    cfg.threshold,
    cfg.threads,
    t1 - t0,
    pos / 1e6 / (t1 - t0),
    t3 - t2,
    pos / 1e6 / (t3 - t2),
    (double)pos / (double)comp,
    (unsigned long long)(col->dict_offsets_len - 1),
    (unsigned long long)col->dict_bytes_len);
  onpair_bridge_column_free(col);
  free(dec); free(chars); free(offsets64); free(offsets32);
  return 0;
}
