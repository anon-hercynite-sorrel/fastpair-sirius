#!/usr/bin/env bash
# Build the bridge (release) and the C round-trip test, then run it on the given files.
# Usage: ctest/run.sh <file>... ; CARGO_TARGET_DIR is honoured.
set -euo pipefail
here=$(cd "$(dirname "$0")/.." && pwd)
target=${CARGO_TARGET_DIR:-$here/target}
cargo build --release --manifest-path "$here/Cargo.toml"
cc -O2 -std=c11 -I"$here/include" "$here/ctest/roundtrip.c" "$target/release/libonpair_bridge.a" \
   -lpthread -ldl -lm -o "$target/roundtrip"
for f in "$@"; do "$target/roundtrip" "$f" 12 0.2; "$target/roundtrip" "$f" 16 0.2; done
