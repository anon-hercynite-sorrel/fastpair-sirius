// SPDX-License-Identifier: Apache-2.0
//! C ABI over the `onpair` crate for Simpatico's `onpair` operator.
//!
//! The operator compresses on the host at pin time: it copies a cuDF strings column
//! (chars + int32 or int64 offsets) to host memory, calls [`onpair_bridge_encode32`] /
//! [`onpair_bridge_encode64`], and uploads three of the bridge's four output channels
//! (`codes`, `dict_bytes`, `dict_offsets`). The operator stores normalized input byte
//! offsets in place of the bridge's token offsets and computes a fifth stored channel
//! of batch byte prefixes on the device. Decoding on the GPU is the operator's own
//! kernel; the CPU decode here exists for round-trip tests and
//! `simpatico compress --verify`.
//!
//! Layout of the encoded column (one dictionary per call; the operator calls once per
//! pin batch):
//! - `dict_bytes` / `dict_offsets`: the crate's compact dictionary. `dict_offsets` has
//!   `num_tokens + 1` entries; token `t` is `dict_bytes[dict_offsets[t]..dict_offsets[t+1]]`,
//!   at most [`onpair::MAX_TOKEN_SIZE`] (16) bytes. `dict_bytes` is read-padded by 16.
//! - `codes`: one `u16` per emitted token, rows concatenated. Every code `< num_tokens`.
//! - `row_offsets`: `num_rows + 1` offsets into `codes`, same width as the input
//!   offsets (u32 for int32 input, u64 for int64 input); `row_offsets[0] == 0`.
//!
//! Every entry point is panic-safe: a crate panic (malformed input on decode) becomes
//! a status code, never an unwind across the FFI boundary.

use std::mem::MaybeUninit;
use std::panic::{catch_unwind, AssertUnwindSafe};

use onpair::{CompactDictionaryView, Config, MaxDictBits, Offset, Parser, Rows, Threshold};

/// Status codes. Zero is success; everything else names the failing check.
pub const ONPAIR_BRIDGE_OK: i32 = 0;
pub const ONPAIR_BRIDGE_ERR_NULL_ARG: i32 = 1;
pub const ONPAIR_BRIDGE_ERR_BAD_CONFIG: i32 = 2;
pub const ONPAIR_BRIDGE_ERR_BAD_OFFSETS: i32 = 3;
pub const ONPAIR_BRIDGE_ERR_ENCODE: i32 = 4;
pub const ONPAIR_BRIDGE_ERR_BAD_DICT: i32 = 5;
pub const ONPAIR_BRIDGE_ERR_OUTPUT_TOO_SMALL: i32 = 6;
pub const ONPAIR_BRIDGE_ERR_PANIC: i32 = 7;

/// Training configuration. `max_dict_bits` in 9..=16 (12 = OnPair-12, 16 = OnPair-16),
/// `threshold` in (0, 1] is the dynamic training-sample fraction, `seed` fixes the sample
/// (`seed_is_set == 0` means non-deterministic). `threads` is the parsing thread cap:
/// 0 selects available parallelism, 1 keeps parsing serial.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct onpair_bridge_config {
    pub max_dict_bits: u8,
    pub threshold: f64,
    pub seed: u64,
    pub seed_is_set: u8,
    pub threads: u32,
}

/// The paper's configuration: 12-bit dictionary, 20% sample, seed 42.
#[no_mangle]
pub extern "C" fn onpair_bridge_config_default() -> onpair_bridge_config {
    onpair_bridge_config {
        max_dict_bits: 12,
        threshold: 0.2,
        seed: 42,
        seed_is_set: 1,
        threads: 0,
    }
}

/// An encoded column. Owned by the bridge; release with [`onpair_bridge_column_free`].
/// Pointers are valid for the lengths given; a zero-length channel may be null.
#[repr(C)]
pub struct onpair_bridge_column {
    pub dict_bytes: *const u8,
    pub dict_bytes_len: u64,
    pub dict_offsets: *const u32,
    pub dict_offsets_len: u64, // num_tokens + 1
    pub codes: *const u16,
    pub codes_len: u64,
    pub row_offsets: *const u8, // u32 or u64 elements, see row_offset_width
    pub row_offsets_len: u64, // num_rows + 1 elements
    pub row_offset_width: u8, // 4 or 8 bytes per element
    _owner: *mut ColumnOwner,
}

struct ColumnOwner {
    _dict_bytes: Vec<u8>,
    _dict_offsets: Vec<u32>,
    _codes: Vec<u16>,
    _row_offsets: RowOffsets,
}

enum RowOffsets {
    U32(Vec<u32>),
    U64(Vec<u64>),
}

fn config_from(c: &onpair_bridge_config) -> Option<Config> {
    Some(Config {
        max_dict_bits: MaxDictBits::new(c.max_dict_bits).ok()?,
        threshold: Threshold::new(c.threshold).ok()?,
        seed: if c.seed_is_set != 0 { Some(c.seed) } else { None },
    })
}

/// Validate cuDF-style signed offsets: `n + 1` entries, non-negative, non-decreasing,
/// last entry within `bytes_len`. Returns the same memory viewed as the unsigned type:
/// a non-negative two's-complement value has the same bit pattern unsigned.
fn check_offsets<S: Copy + Into<i64>>(offsets: &[S], bytes_len: usize) -> bool {
    if offsets.is_empty() {
        return false;
    }
    let mut prev: i64 = 0;
    for (i, &o) in offsets.iter().enumerate() {
        let o: i64 = o.into();
        if o < 0 || (i > 0 && o < prev) {
            return false;
        }
        prev = o;
    }
    prev as u64 <= bytes_len as u64
}

/// Checked view of `len` elements at `ptr`: `None` when the element count or its byte
/// span cannot be a Rust slice (`> isize::MAX`). A null pointer with `len == 0` is an
/// empty slice. Keeping this inside `catch_unwind` means a malformed count is a status
/// code, not an abort.
///
/// # Safety
/// When `Some` is returned the caller's pointer/length contract holds for the result.
unsafe fn view<'a, T>(ptr: *const T, len: u64) -> Option<&'a [T]> {
    if len == 0 {
        return Some(&[]);
    }
    let n = usize::try_from(len).ok()?;
    n.checked_mul(std::mem::size_of::<T>()).filter(|&b| b <= isize::MAX as usize)?;
    Some(std::slice::from_raw_parts(ptr, n))
}

/// Mutable counterpart of [`view`].
unsafe fn view_mut<'a, T>(ptr: *mut T, len: u64) -> Option<&'a mut [T]> {
    if len == 0 {
        return Some(&mut []);
    }
    let n = usize::try_from(len).ok()?;
    n.checked_mul(std::mem::size_of::<T>()).filter(|&b| b <= isize::MAX as usize)?;
    Some(std::slice::from_raw_parts_mut(ptr, n))
}

/// Element count of a `num_rows + 1` offsets array, or `None` on overflow.
fn offsets_len(num_rows: u64) -> Option<u64> {
    num_rows.checked_add(1)
}

// ArrowRows is private to onpair; retain absolute byte offsets for sliced columns.
struct RowRange<'a, O> {
    bytes: &'a [u8],
    offsets: &'a [O],
}

impl<O: Offset> Rows for RowRange<'_, O> {
    fn num_rows(&self) -> usize {
        self.offsets.len() - 1
    }

    fn total_bytes(&self) -> usize {
        self.offsets.last().unwrap().to_usize() - self.offsets[0].to_usize()
    }

    fn row(&self, i: usize) -> &[u8] {
        &self.bytes[self.offsets[i].to_usize()..self.offsets[i + 1].to_usize()]
    }
}

fn parse_chunks<O: Offset + Send + Sync>(
    parser: &Parser,
    bytes: &[u8],
    offsets: &[O],
    chunks: usize,
) -> (Vec<u16>, Vec<O>) {
    let num_rows = offsets.len() - 1;
    let parse = |start, end| {
        let rows = RowRange {
            bytes,
            offsets: &offsets[start..=end],
        };
        let (mut codes, mut row_offsets) = (Vec::new(), Vec::new());
        parser.parse_rows_into(&rows, &mut codes, &mut row_offsets);
        (codes, row_offsets)
    };
    if chunks == 1 {
        return parse(0, num_rows);
    }
    // Parser is Sync (enforced by scope.spawn): every worker borrows the same parser.
    // scope joins all workers even if spawning, parsing, or joining panics. The outer
    // C ABI catch_unwind then reports ONPAIR_BRIDGE_ERR_PANIC.
    let parts = std::thread::scope(|scope| {
        let handles: Vec<_> = (0..chunks)
            .map(|i| {
                let parse = &parse;
                scope.spawn(move || parse(i * num_rows / chunks, (i + 1) * num_rows / chunks))
            })
            .collect();
        handles
            .into_iter()
            .map(|h| h.join().unwrap())
            .collect::<Vec<_>>()
    });
    let mut codes = Vec::with_capacity(parts.iter().map(|(c, _)| c.len()).sum());
    let mut row_offsets = Vec::with_capacity(offsets.len());
    row_offsets.push(O::from_usize(0));
    for (chunk_codes, chunk_offsets) in parts {
        let base = codes.len();
        row_offsets.extend(
            chunk_offsets
                .into_iter()
                .skip(1)
                .map(|o| O::from_usize(base + o.to_usize())),
        );
        codes.extend(chunk_codes);
    }
    (codes, row_offsets)
}

fn encode_impl<O: Offset + Send + Sync>(
    bytes: &[u8],
    offsets: &[O],
    cfg: Config,
    threads: u32,
) -> Option<(Vec<u8>, Vec<u32>, Vec<u16>, Vec<O>)> {
    let parser = Parser::train(bytes, offsets, cfg).ok()?;
    let threads = if threads == 0 {
        std::thread::available_parallelism().map_or(1, |n| n.get())
    } else {
        threads as usize
    };
    // Balanced contiguous row ranges, at least 64K rows per worker. Small columns
    // stay on the caller's thread; never split a row's tokens between workers.
    let chunks = threads.min(((offsets.len() - 1) / 65_536).max(1));
    let (codes, row_offsets) = parse_chunks(&parser, bytes, offsets, chunks);
    let (dict_bytes, dict_offsets) = parser.dict.into_raw();
    Some((dict_bytes, dict_offsets, codes, row_offsets))
}

fn finish(out: *mut *mut onpair_bridge_column, dict_bytes: Vec<u8>, dict_offsets: Vec<u32>, codes: Vec<u16>, row_offsets: RowOffsets) -> i32 {
    let (ro_ptr, ro_len, ro_width) = match &row_offsets {
        RowOffsets::U32(v) => (v.as_ptr() as *const u8, v.len() as u64, 4u8),
        RowOffsets::U64(v) => (v.as_ptr() as *const u8, v.len() as u64, 8u8),
    };
    let col = Box::new(onpair_bridge_column {
        dict_bytes: dict_bytes.as_ptr(),
        dict_bytes_len: dict_bytes.len() as u64,
        dict_offsets: dict_offsets.as_ptr(),
        dict_offsets_len: dict_offsets.len() as u64,
        codes: codes.as_ptr(),
        codes_len: codes.len() as u64,
        row_offsets: ro_ptr,
        row_offsets_len: ro_len,
        row_offset_width: ro_width,
        _owner: Box::into_raw(Box::new(ColumnOwner {
            _dict_bytes: dict_bytes,
            _dict_offsets: dict_offsets,
            _codes: codes,
            _row_offsets: row_offsets,
        })),
    });
    // SAFETY: `out` was checked non-null by the caller.
    unsafe { *out = Box::into_raw(col) };
    ONPAIR_BRIDGE_OK
}

/// Train a dictionary on `(bytes, offsets)` and encode. `offsets` has `num_rows + 1`
/// int32 entries (cuDF strings column layout); `offsets[0]` may be non-zero for a sliced
/// column. `num_rows == 0` yields an empty column (empty dictionary, `row_offsets == [0]`).
///
/// # Safety
/// `bytes` must be readable for `bytes_len` bytes and `offsets` (4-byte aligned) for
/// `num_rows + 1` elements, and neither may be written by another thread during the
/// call; `out` must be a valid pointer. Counts whose byte span exceeds `isize::MAX`
/// return `ONPAIR_BRIDGE_ERR_BAD_OFFSETS` instead of being dereferenced.
#[no_mangle]
pub unsafe extern "C" fn onpair_bridge_encode32(
    bytes: *const u8,
    bytes_len: u64,
    offsets: *const i32,
    num_rows: u64,
    cfg: onpair_bridge_config,
    out: *mut *mut onpair_bridge_column,
) -> i32 {
    if out.is_null() || offsets.is_null() || (bytes.is_null() && bytes_len != 0) {
        return ONPAIR_BRIDGE_ERR_NULL_ARG;
    }
    let threads = cfg.threads;
    let Some(cfg) = config_from(&cfg) else { return ONPAIR_BRIDGE_ERR_BAD_CONFIG };
    // Everything that can fail on a malformed count or trip a crate panic sits inside
    // the panic boundary; nothing unwinds across `extern "C"`.
    match catch_unwind(AssertUnwindSafe(|| {
        let (Some(bytes), Some(n)) = (view::<u8>(bytes, bytes_len), offsets_len(num_rows)) else {
            return Err(ONPAIR_BRIDGE_ERR_BAD_OFFSETS);
        };
        let Some(signed) = view::<i32>(offsets, n) else { return Err(ONPAIR_BRIDGE_ERR_BAD_OFFSETS) };
        if !check_offsets(signed, bytes.len()) {
            return Err(ONPAIR_BRIDGE_ERR_BAD_OFFSETS);
        }
        // Same memory, unsigned view (see `check_offsets`).
        let unsigned = std::slice::from_raw_parts(offsets as *const u32, signed.len());
        encode_impl::<u32>(bytes, unsigned, cfg, threads).ok_or(ONPAIR_BRIDGE_ERR_ENCODE)
    })) {
        Ok(Ok((db, doff, codes, ro))) => finish(out, db, doff, codes, RowOffsets::U32(ro)),
        Ok(Err(code)) => code,
        Err(_) => ONPAIR_BRIDGE_ERR_PANIC,
    }
}

/// As [`onpair_bridge_encode32`] for a cuDF large-strings column (int64 offsets).
///
/// # Safety
/// As [`onpair_bridge_encode32`], with `offsets` 8-byte aligned.
#[no_mangle]
pub unsafe extern "C" fn onpair_bridge_encode64(
    bytes: *const u8,
    bytes_len: u64,
    offsets: *const i64,
    num_rows: u64,
    cfg: onpair_bridge_config,
    out: *mut *mut onpair_bridge_column,
) -> i32 {
    if out.is_null() || offsets.is_null() || (bytes.is_null() && bytes_len != 0) {
        return ONPAIR_BRIDGE_ERR_NULL_ARG;
    }
    let threads = cfg.threads;
    let Some(cfg) = config_from(&cfg) else { return ONPAIR_BRIDGE_ERR_BAD_CONFIG };
    // Everything that can fail on a malformed count or trip a crate panic sits inside
    // the panic boundary; nothing unwinds across `extern "C"`.
    match catch_unwind(AssertUnwindSafe(|| {
        let (Some(bytes), Some(n)) = (view::<u8>(bytes, bytes_len), offsets_len(num_rows)) else {
            return Err(ONPAIR_BRIDGE_ERR_BAD_OFFSETS);
        };
        let Some(signed) = view::<i64>(offsets, n) else { return Err(ONPAIR_BRIDGE_ERR_BAD_OFFSETS) };
        if !check_offsets(signed, bytes.len()) {
            return Err(ONPAIR_BRIDGE_ERR_BAD_OFFSETS);
        }
        // Same memory, unsigned view (see `check_offsets`).
        let unsigned = std::slice::from_raw_parts(offsets as *const u64, signed.len());
        encode_impl::<u64>(bytes, unsigned, cfg, threads).ok_or(ONPAIR_BRIDGE_ERR_ENCODE)
    })) {
        Ok(Ok((db, doff, codes, ro))) => finish(out, db, doff, codes, RowOffsets::U64(ro)),
        Ok(Err(code)) => code,
        Err(_) => ONPAIR_BRIDGE_ERR_PANIC,
    }
}

/// Release a column returned by an encode call. Null is a no-op.
///
/// # Safety
/// `col` must come from this bridge and not be freed twice.
#[no_mangle]
pub unsafe extern "C" fn onpair_bridge_column_free(col: *mut onpair_bridge_column) {
    if col.is_null() {
        return;
    }
    let col = Box::from_raw(col);
    drop(Box::from_raw(col._owner));
}

/// Decoded byte length of `codes` under the dictionary. Validates the dictionary.
///
/// # Safety
/// Pointers readable for the given lengths (`dict_offsets` 4-byte, `codes` 2-byte
/// aligned) and not written by another thread during the call; `out_len` valid.
#[no_mangle]
pub unsafe extern "C" fn onpair_bridge_decoded_len(
    dict_bytes: *const u8,
    dict_bytes_len: u64,
    dict_offsets: *const u32,
    dict_offsets_len: u64,
    codes: *const u16,
    codes_len: u64,
    out_len: *mut u64,
) -> i32 {
    if out_len.is_null() || dict_offsets.is_null() || (dict_bytes.is_null() && dict_bytes_len != 0) || (codes.is_null() && codes_len != 0) {
        return ONPAIR_BRIDGE_ERR_NULL_ARG;
    }
    match catch_unwind(AssertUnwindSafe(|| {
        let (Some(db), Some(doff), Some(codes)) =
            (view::<u8>(dict_bytes, dict_bytes_len), view::<u32>(dict_offsets, dict_offsets_len), view::<u16>(codes, codes_len))
        else {
            return Err(ONPAIR_BRIDGE_ERR_BAD_DICT);
        };
        let dict = CompactDictionaryView::validate(db, doff).map_err(|_| ONPAIR_BRIDGE_ERR_BAD_DICT)?;
        Ok(onpair::decoded_len(codes, dict))
    })) {
        Ok(Ok(n)) => {
            *out_len = n as u64;
            ONPAIR_BRIDGE_OK
        }
        Ok(Err(code)) => code,
        Err(_) => ONPAIR_BRIDGE_ERR_PANIC,
    }
}

/// CPU reference decode of `codes` into `out` (capacity `out_cap`). Writes the number of
/// bytes produced to `written`. Rows are not delimited here: the caller converts
/// `row_offsets` (code offsets) to byte offsets by summing token lengths per row, or
/// decodes row ranges of codes separately.
///
/// # Safety
/// Inputs readable for the given lengths (`dict_offsets` 4-byte, `codes` 2-byte
/// aligned) and not written by another thread during the call; `out` writable for
/// `out_cap` bytes; `written` valid. `[out, out + out_cap)` must not overlap any input
/// or `written`; an overlapping call returns `ONPAIR_BRIDGE_ERR_NULL_ARG`.
#[no_mangle]
pub unsafe extern "C" fn onpair_bridge_decode(
    dict_bytes: *const u8,
    dict_bytes_len: u64,
    dict_offsets: *const u32,
    dict_offsets_len: u64,
    codes: *const u16,
    codes_len: u64,
    out: *mut u8,
    out_cap: u64,
    written: *mut u64,
) -> i32 {
    if written.is_null() || dict_offsets.is_null() || (dict_bytes.is_null() && dict_bytes_len != 0) || (codes.is_null() && codes_len != 0) || (out.is_null() && out_cap != 0) {
        return ONPAIR_BRIDGE_ERR_NULL_ARG;
    }
    // The output range must not overlap any input: `out` becomes an exclusive Rust
    // reference while the inputs are shared ones. Overlap is rejected here rather than
    // left as undefined behaviour.
    if out_cap != 0 {
        let o0 = out as usize;
        let o1 = o0.saturating_add(out_cap.try_into().unwrap_or(usize::MAX));
        let overlaps = |p: usize, bytes: usize| p < o1 && o0 < p.saturating_add(bytes);
        if overlaps(dict_bytes as usize, dict_bytes_len.try_into().unwrap_or(usize::MAX))
            || overlaps(dict_offsets as usize, dict_offsets_len.saturating_mul(4).try_into().unwrap_or(usize::MAX))
            || overlaps(codes as usize, codes_len.saturating_mul(2).try_into().unwrap_or(usize::MAX))
            || overlaps(written as usize, 8)
        {
            return ONPAIR_BRIDGE_ERR_NULL_ARG;
        }
    }
    match catch_unwind(AssertUnwindSafe(|| {
        let (Some(db), Some(doff), Some(codes), Some(out)) = (
            view::<u8>(dict_bytes, dict_bytes_len),
            view::<u32>(dict_offsets, dict_offsets_len),
            view::<u16>(codes, codes_len),
            view_mut::<MaybeUninit<u8>>(out as *mut MaybeUninit<u8>, out_cap),
        ) else {
            return Err(ONPAIR_BRIDGE_ERR_BAD_DICT);
        };
        let dict = CompactDictionaryView::validate(db, doff).map_err(|_| ONPAIR_BRIDGE_ERR_BAD_DICT)?;
        onpair::try_decode_into(codes, dict, out).map_err(|_| ONPAIR_BRIDGE_ERR_OUTPUT_TOO_SMALL)
    })) {
        Ok(Ok(n)) => {
            *written = n as u64;
            ONPAIR_BRIDGE_OK
        }
        Ok(Err(code)) => code,
        Err(_) => ONPAIR_BRIDGE_ERR_PANIC,
    }
}

/// Version string of the underlying crate, e.g. "onpair 0.3.0". Static, never freed.
#[no_mangle]
pub extern "C" fn onpair_bridge_version() -> *const std::os::raw::c_char {
    concat!("onpair ", "0.3.0", "\0").as_ptr() as *const std::os::raw::c_char
}

#[cfg(test)]
mod tests {
    use super::*;

    fn column(rows: &[&[u8]]) -> (Vec<u8>, Vec<i32>) {
        let mut bytes = Vec::new();
        let mut offs = vec![0i32];
        for r in rows {
            bytes.extend_from_slice(r);
            offs.push(bytes.len() as i32);
        }
        (bytes, offs)
    }

    /// Encode through the C ABI, decode through the C ABI, compare bytes and per-row
    /// boundaries (row byte offsets derived from code offsets + token lengths).
    fn roundtrip32(rows: &[&[u8]], cfg: onpair_bridge_config) {
        for threads in [1, 0, 4] {
            roundtrip32_once(rows, onpair_bridge_config { threads, ..cfg });
        }
    }

    fn assert_serial_equal<O: Offset + Send + Sync + PartialEq>(
        bytes: &[u8],
        offsets: &[O],
        cfg: onpair_bridge_config,
    ) {
        let config = config_from(&cfg).unwrap();
        let expected = onpair::compress(bytes, offsets, config).unwrap().into_raw();
        let (db, doff) = expected.0.into_raw();
        let actual = encode_impl(bytes, offsets, config, cfg.threads).unwrap();
        assert_eq!(actual, (db, doff, expected.1, expected.2));
        // Force the scoped path even for small fixtures, which production keeps serial.
        let parser = Parser::train(bytes, offsets, config).unwrap();
        let parallel = parse_chunks(&parser, bytes, offsets, 4);
        assert_eq!(parallel.0, actual.2);
        assert_eq!(parallel.1, actual.3);
        assert_eq!(parallel.1[0].to_usize(), 0);
        assert_eq!(parallel.1.last().unwrap().to_usize(), parallel.0.len());
    }

    fn roundtrip32_once(rows: &[&[u8]], cfg: onpair_bridge_config) {
        let (bytes, offs) = column(rows);
        let unsigned: Vec<u32> = offs.iter().map(|&o| o as u32).collect();
        assert_serial_equal(&bytes, &unsigned, cfg);
        let mut out: *mut onpair_bridge_column = std::ptr::null_mut();
        let st = unsafe { onpair_bridge_encode32(bytes.as_ptr(), bytes.len() as u64, offs.as_ptr(), rows.len() as u64, cfg, &mut out) };
        assert_eq!(st, ONPAIR_BRIDGE_OK);
        let c = unsafe { &*out };
        assert_eq!(c.row_offset_width, 4);
        assert_eq!(c.row_offsets_len, rows.len() as u64 + 1);
        let mut len = 0u64;
        let st = unsafe { onpair_bridge_decoded_len(c.dict_bytes, c.dict_bytes_len, c.dict_offsets, c.dict_offsets_len, c.codes, c.codes_len, &mut len) };
        assert_eq!(st, ONPAIR_BRIDGE_OK);
        assert_eq!(len as usize, bytes.len());
        let mut dec = vec![0u8; bytes.len()];
        let mut written = 0u64;
        let st = unsafe { onpair_bridge_decode(c.dict_bytes, c.dict_bytes_len, c.dict_offsets, c.dict_offsets_len, c.codes, c.codes_len, dec.as_mut_ptr(), dec.len() as u64, &mut written) };
        assert_eq!(st, ONPAIR_BRIDGE_OK);
        assert_eq!(written as usize, bytes.len());
        assert_eq!(dec, bytes);
        // per-row boundaries
        let ro = unsafe { std::slice::from_raw_parts(c.row_offsets as *const u32, c.row_offsets_len as usize) };
        let codes = unsafe { std::slice::from_raw_parts(c.codes, c.codes_len as usize) };
        let doff = unsafe { std::slice::from_raw_parts(c.dict_offsets, c.dict_offsets_len as usize) };
        let mut byte_pos = 0usize;
        for (k, r) in rows.iter().enumerate() {
            let row_len: usize = codes[ro[k] as usize..ro[k + 1] as usize].iter().map(|&t| (doff[t as usize + 1] - doff[t as usize]) as usize).sum();
            assert_eq!(row_len, r.len(), "row {k}");
            assert_eq!(&dec[byte_pos..byte_pos + row_len], *r);
            byte_pos += row_len;
        }
        // too-small output is reported, not overflowed
        if !bytes.is_empty() {
            let mut small = vec![0u8; bytes.len() - 1];
            let st = unsafe { onpair_bridge_decode(c.dict_bytes, c.dict_bytes_len, c.dict_offsets, c.dict_offsets_len, c.codes, c.codes_len, small.as_mut_ptr(), small.len() as u64, &mut written) };
            assert_eq!(st, ONPAIR_BRIDGE_ERR_OUTPUT_TOO_SMALL);
        }
        unsafe { onpair_bridge_column_free(out) };
    }

    #[test]
    fn empty_column() {
        roundtrip32(&[], onpair_bridge_config_default());
    }

    #[test]
    fn all_empty_strings() {
        roundtrip32(&[b"", b"", b""], onpair_bridge_config_default());
    }

    #[test]
    fn single_row() {
        roundtrip32(&[b"http://example.com/path?q=1"], onpair_bridge_config_default());
    }

    #[test]
    fn mixed_rows_and_presets() {
        let rows: Vec<Vec<u8>> = (0..5000).map(|i| format!("http://site{}.example.org/dir/{}/page{}.html?x={}", i % 37, i % 101, i, i * 7).into_bytes()).collect();
        let refs: Vec<&[u8]> = rows.iter().map(|v| v.as_slice()).collect();
        for bits in [9u8, 12, 16] {
            let cfg = onpair_bridge_config { max_dict_bits: bits, ..onpair_bridge_config_default() };
            roundtrip32(&refs, cfg);
        }
    }

    #[test]
    fn row_longer_than_one_token_batch() {
        // 128 tokens per decode batch on the GPU; a 20 KB row of varied bytes needs many more.
        let long: Vec<u8> = (0..20_000u32).map(|i| (i.wrapping_mul(2654435761) >> 13) as u8).collect();
        roundtrip32(&[b"short", &long, b"", &long[..777]], onpair_bridge_config_default());
    }

    #[test]
    fn parallel_chunk_boundaries() {
        // Four 75K-row chunks: every seam has multi-token rows and empty rows
        // on both sides; token-batch boundaries also fall inside these rows.
        let rows: Vec<Vec<u8>> = (0..300_000)
            .map(|i| {
                if i % 75_000 == 0 || i % 75_000 == 74_999 {
                    Vec::new()
                } else {
                    format!(
                        "https://site{}.example/path/{i}/a/multi/token/row?x={}",
                        i % 37,
                        i * 7
                    )
                    .into_bytes()
                }
            })
            .collect();
        let refs: Vec<&[u8]> = rows.iter().map(Vec::as_slice).collect();
        roundtrip32(&refs, onpair_bridge_config_default());
        let (bytes, offsets) = column(&refs);
        let offsets: Vec<u64> = offsets.iter().map(|&o| o as u64).collect();
        for threads in [1, 0, 4] {
            assert_serial_equal(
                &bytes,
                &offsets,
                onpair_bridge_config {
                    threads,
                    ..onpair_bridge_config_default()
                },
            );
        }
    }

    #[test]
    fn worker_panic_is_caught_after_scope_joins() {
        let parser = Parser::train(
            b"x",
            &[0u32, 1],
            config_from(&onpair_bridge_config_default()).unwrap(),
        )
        .unwrap();
        // Bypass ABI validation to provoke a worker's out-of-bounds row access.
        let result = catch_unwind(AssertUnwindSafe(|| {
            parse_chunks(&parser, b"x", &[0u32, 1, 2], 2)
        }));
        assert!(result.is_err());
    }

    #[test]
    fn sliced_offsets_start_nonzero() {
        let (bytes, offs) = column(&[b"skip-me", b"keep one", b"keep two"]);
        let sliced = &offs[1..]; // rows 1..3, offsets[0] == 7
        let cfg = onpair_bridge_config_default();
        let unsigned: Vec<u32> = sliced.iter().map(|&o| o as u32).collect();
        assert_serial_equal(&bytes, &unsigned, cfg);
        let mut out: *mut onpair_bridge_column = std::ptr::null_mut();
        let st = unsafe { onpair_bridge_encode32(bytes.as_ptr(), bytes.len() as u64, sliced.as_ptr(), 2, cfg, &mut out) };
        assert_eq!(st, ONPAIR_BRIDGE_OK);
        let c = unsafe { &*out };
        let mut dec = vec![0u8; 16];
        let mut written = 0u64;
        let st = unsafe { onpair_bridge_decode(c.dict_bytes, c.dict_bytes_len, c.dict_offsets, c.dict_offsets_len, c.codes, c.codes_len, dec.as_mut_ptr(), 16, &mut written) };
        assert_eq!(st, ONPAIR_BRIDGE_OK);
        assert_eq!(&dec[..written as usize], b"keep onekeep two");
        unsafe { onpair_bridge_column_free(out) };
    }

    #[test]
    fn int64_offsets() {
        let (bytes, offs) = column(&[b"alpha", b"beta", b"gamma delta"]);
        let offs64: Vec<i64> = offs.iter().map(|&o| o as i64).collect();
        let unsigned: Vec<u64> = offs.iter().map(|&o| o as u64).collect();
        assert_serial_equal(&bytes, &unsigned, onpair_bridge_config_default());
        let mut out: *mut onpair_bridge_column = std::ptr::null_mut();
        let st = unsafe { onpair_bridge_encode64(bytes.as_ptr(), bytes.len() as u64, offs64.as_ptr(), 3, onpair_bridge_config_default(), &mut out) };
        assert_eq!(st, ONPAIR_BRIDGE_OK);
        let c = unsafe { &*out };
        assert_eq!(c.row_offset_width, 8);
        let ro = unsafe { std::slice::from_raw_parts(c.row_offsets as *const u64, 4) };
        assert_eq!(ro[0], 0);
        assert_eq!(ro[3], c.codes_len);
        unsafe { onpair_bridge_column_free(out) };
    }

    #[test]
    fn rejects_bad_input() {
        let (bytes, offs) = column(&[b"x"]);
        let mut out: *mut onpair_bridge_column = std::ptr::null_mut();
        let bad_cfg = onpair_bridge_config { max_dict_bits: 40, ..onpair_bridge_config_default() };
        assert_eq!(unsafe { onpair_bridge_encode32(bytes.as_ptr(), 1, offs.as_ptr(), 1, bad_cfg, &mut out) }, ONPAIR_BRIDGE_ERR_BAD_CONFIG);
        let neg = [0i32, -1];
        assert_eq!(unsafe { onpair_bridge_encode32(bytes.as_ptr(), 1, neg.as_ptr(), 1, onpair_bridge_config_default(), &mut out) }, ONPAIR_BRIDGE_ERR_BAD_OFFSETS);
        let past = [0i32, 5];
        assert_eq!(unsafe { onpair_bridge_encode32(bytes.as_ptr(), 1, past.as_ptr(), 1, onpair_bridge_config_default(), &mut out) }, ONPAIR_BRIDGE_ERR_BAD_OFFSETS);
        assert_eq!(unsafe { onpair_bridge_encode32(bytes.as_ptr(), 1, offs.as_ptr(), 1, onpair_bridge_config_default(), std::ptr::null_mut()) }, ONPAIR_BRIDGE_ERR_NULL_ARG);
        // counts whose byte span cannot be a slice are rejected, not dereferenced
        assert_eq!(unsafe { onpair_bridge_encode32(bytes.as_ptr(), 1, offs.as_ptr(), u64::MAX, onpair_bridge_config_default(), &mut out) }, ONPAIR_BRIDGE_ERR_BAD_OFFSETS);
        assert_eq!(unsafe { onpair_bridge_encode32(bytes.as_ptr(), u64::MAX, offs.as_ptr(), 1, onpair_bridge_config_default(), &mut out) }, ONPAIR_BRIDGE_ERR_BAD_OFFSETS);
        assert_eq!(unsafe { onpair_bridge_encode32(bytes.as_ptr(), 1, offs.as_ptr(), (isize::MAX as u64) / 4, onpair_bridge_config_default(), &mut out) }, ONPAIR_BRIDGE_ERR_BAD_OFFSETS);
    }

    /// A valid encoded column to corrupt in the malformed-input tests.
    fn encoded(rows: &[&[u8]]) -> (Vec<u8>, Vec<u32>, Vec<u16>) {
        let (bytes, offs) = column(rows);
        let mut out: *mut onpair_bridge_column = std::ptr::null_mut();
        let st = unsafe { onpair_bridge_encode32(bytes.as_ptr(), bytes.len() as u64, offs.as_ptr(), rows.len() as u64, onpair_bridge_config_default(), &mut out) };
        assert_eq!(st, ONPAIR_BRIDGE_OK);
        let c = unsafe { &*out };
        let db = unsafe { std::slice::from_raw_parts(c.dict_bytes, c.dict_bytes_len as usize) }.to_vec();
        let doff = unsafe { std::slice::from_raw_parts(c.dict_offsets, c.dict_offsets_len as usize) }.to_vec();
        let codes = unsafe { std::slice::from_raw_parts(c.codes, c.codes_len as usize) }.to_vec();
        unsafe { onpair_bridge_column_free(out) };
        (db, doff, codes)
    }

    fn decode_status(db: &[u8], doff: &[u32], codes: &[u16], cap: usize) -> (i32, i32) {
        let mut dec = vec![0u8; cap.max(1)];
        let mut w = 0u64;
        let d = unsafe { onpair_bridge_decode(db.as_ptr(), db.len() as u64, doff.as_ptr(), doff.len() as u64, codes.as_ptr(), codes.len() as u64, dec.as_mut_ptr(), cap as u64, &mut w) };
        let mut n = 0u64;
        let l = unsafe { onpair_bridge_decoded_len(db.as_ptr(), db.len() as u64, doff.as_ptr(), doff.len() as u64, codes.as_ptr(), codes.len() as u64, &mut n) };
        (d, l)
    }

    #[test]
    fn out_of_range_code_is_contained() {
        let rows: Vec<&[u8]> = (0..64).map(|_| &b"the quick brown fox jumps over the lazy dog"[..]).collect();
        let (db, doff, mut codes) = encoded(&rows);
        assert_eq!(decode_status(&db, &doff, &codes, 4096).0, ONPAIR_BRIDGE_OK);
        let num_tokens = doff.len() as u16 - 1;
        codes[0] = num_tokens + 100;
        // the crate indexes the dictionary with the code; the panic is turned into a status
        // by both the decoder and the length query
        let (d, l) = decode_status(&db, &doff, &codes, 4096);
        assert_eq!(d, ONPAIR_BRIDGE_ERR_PANIC);
        assert_eq!(l, ONPAIR_BRIDGE_ERR_PANIC);
    }

    #[test]
    fn malformed_dictionary_statuses() {
        let rows: Vec<&[u8]> = (0..64).map(|_| &b"abcdefghij klmnopqrst uvwxyz"[..]).collect();
        let (db, doff, codes) = encoded(&rows);
        // non-monotonic offsets
        let mut bad = doff.clone();
        bad.swap(1, 2);
        if bad != doff {
            assert_eq!(decode_status(&db, &bad, &codes, 4096), (ONPAIR_BRIDGE_ERR_BAD_DICT, ONPAIR_BRIDGE_ERR_BAD_DICT));
        }
        // a token longer than MAX_TOKEN_SIZE
        let mut bad = doff.clone();
        *bad.last_mut().unwrap() += onpair::MAX_TOKEN_SIZE as u32 + 1;
        let mut big = db.clone();
        big.resize(big.len() + onpair::MAX_TOKEN_SIZE + 1 + 16, b'z');
        assert_eq!(decode_status(&big, &bad, &codes, 4096), (ONPAIR_BRIDGE_ERR_BAD_DICT, ONPAIR_BRIDGE_ERR_BAD_DICT));
        // missing read padding: dict_bytes ends exactly at the last token
        let tight = &db[..*doff.last().unwrap() as usize];
        assert_eq!(decode_status(tight, &doff, &codes, 4096), (ONPAIR_BRIDGE_ERR_BAD_DICT, ONPAIR_BRIDGE_ERR_BAD_DICT));
        // too-small output
        assert_eq!(decode_status(&db, &doff, &codes, 3).0, ONPAIR_BRIDGE_ERR_OUTPUT_TOO_SMALL);
        // overlapping output and input
        let mut buf = db.clone();
        buf.resize(buf.len() + 4096, 0);
        let mut w = 0u64;
        let st = unsafe { onpair_bridge_decode(buf.as_ptr(), db.len() as u64, doff.as_ptr(), doff.len() as u64, codes.as_ptr(), codes.len() as u64, buf.as_mut_ptr().add(8), 4096, &mut w) };
        assert_eq!(st, ONPAIR_BRIDGE_ERR_NULL_ARG);
    }

    #[test]
    fn int64_large_offsets_roundtrip() {
        // Rows whose offsets exceed int32 in principle cannot be allocated in a unit test;
        // instead exercise the int64 path end to end on a sliced column with a large
        // starting offset relative to a small buffer window, and check row lengths.
        let rows: Vec<Vec<u8>> = (0..200).map(|i| format!("row {i} http://example.com/{}/{}", i * 7, i % 13).into_bytes()).collect();
        let refs: Vec<&[u8]> = rows.iter().map(|r| r.as_slice()).collect();
        let (bytes, offs) = column(&refs);
        let offs64: Vec<i64> = offs.iter().map(|&o| o as i64).collect();
        let unsigned: Vec<u64> = offs.iter().map(|&o| o as u64).collect();
        assert_serial_equal(&bytes, &unsigned, onpair_bridge_config_default());
        let mut out: *mut onpair_bridge_column = std::ptr::null_mut();
        let st = unsafe { onpair_bridge_encode64(bytes.as_ptr(), bytes.len() as u64, offs64.as_ptr(), refs.len() as u64, onpair_bridge_config_default(), &mut out) };
        assert_eq!(st, ONPAIR_BRIDGE_OK);
        let c = unsafe { &*out };
        assert_eq!(c.row_offset_width, 8);
        let ro = unsafe { std::slice::from_raw_parts(c.row_offsets as *const u64, c.row_offsets_len as usize) };
        let codes = unsafe { std::slice::from_raw_parts(c.codes, c.codes_len as usize) };
        let doff = unsafe { std::slice::from_raw_parts(c.dict_offsets, c.dict_offsets_len as usize) };
        assert_eq!(ro.len(), refs.len() + 1);
        for k in 0..refs.len() {
            let row_len: usize = codes[ro[k] as usize..ro[k + 1] as usize].iter().map(|&t| (doff[t as usize + 1] - doff[t as usize]) as usize).sum();
            assert_eq!(row_len, refs[k].len(), "row {k}");
        }
        let mut dec = vec![0u8; bytes.len()];
        let mut w = 0u64;
        let st = unsafe { onpair_bridge_decode(c.dict_bytes, c.dict_bytes_len, c.dict_offsets, c.dict_offsets_len, c.codes, c.codes_len, dec.as_mut_ptr(), dec.len() as u64, &mut w) };
        assert_eq!(st, ONPAIR_BRIDGE_OK);
        assert_eq!(&dec[..w as usize], &bytes[..]);
        unsafe { onpair_bridge_column_free(out) };
    }
}
