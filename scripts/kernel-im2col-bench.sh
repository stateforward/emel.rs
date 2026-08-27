#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EMEL_CPP_SOURCE="${EMEL_CPP_SOURCE_DIR:-$ROOT_DIR/../emel.cpp-worktrees/pinned-843a117386ef17dc5a50549bbfc821074c2141d6}"
SML_SOURCE="${EMEL_STATEFORWARD_SML_SOURCE:-$ROOT_DIR/../../sml/sml.cpp-worktrees/pinned-49207123cd3f39767764bae774932cb48623f92f}"
BUILD_DIR="${EMEL_KERNEL_IM2COL_BENCH_ARTIFACT_DIR:-$ROOT_DIR/.artifacts/kernel-goal/im2col-root/bench}"
SNAPSHOT_OVERRIDE="${EMEL_KERNEL_IM2COL_BENCH_SNAPSHOT:-}"
SOURCE_COMMIT=843a117386ef17dc5a50549bbfc821074c2141d6
SML_COMMIT=49207123cd3f39767764bae774932cb48623f92f
EVENTS_BLOB=4b3f02fa5ff9c5071d1fc83938cef1b22b4658b9
DETAIL_BLOB=c8a82643eabfe8f2d7883e655955f455794511b0
X86_SM_BLOB=0b4d635ebbd0fbd52dbca8a2345547fb571205c8
ITERATIONS=10000
WARMUP=1000
UPDATE=true
COMPARE=true

for argument in "$@"; do
  case "$argument" in
    --snapshot) COMPARE=true ;;
    --no-snapshot) COMPARE=false ;;
    --update) UPDATE=true ;;
    --no-update) UPDATE=false ;;
    --update-only) UPDATE=true; COMPARE=false ;;
    --help|-h)
      printf '%s\n' 'usage: scripts/kernel-im2col-bench.sh [--snapshot|--no-snapshot] [--update|--no-update|--update-only]'
      exit 0
      ;;
    *) echo "error: unknown argument: $argument" >&2; exit 2 ;;
  esac
done

for command in cargo cmake diff git rustc shasum; do
  command -v "$command" >/dev/null 2>&1 || { echo "error: missing $command" >&2; exit 2; }
done

assert_clean_reference_tree() {
  local name="$1"
  local path="$2"
  local status
  status="$(git -C "$path" status --porcelain=v1 --untracked-files=all)"
  if [[ -n "$status" ]]; then
    printf 'error: %s reference tree is dirty or contains untracked files: %s\n' "$name" "$path" >&2
    printf '%s\n' "$status" >&2
    exit 1
  fi
}

assert_clean_reference_tree "emel.cpp" "$EMEL_CPP_SOURCE"
assert_clean_reference_tree "stateforward-sml" "$SML_SOURCE"
[[ "$(git -C "$EMEL_CPP_SOURCE" rev-parse HEAD)" == "$SOURCE_COMMIT" ]] || {
  echo "error: emel.cpp is not at pinned commit $SOURCE_COMMIT" >&2; exit 1;
}
[[ "$(git -C "$SML_SOURCE" rev-parse HEAD)" == "$SML_COMMIT" ]] || {
  echo "error: stateforward-sml is not at pinned commit $SML_COMMIT" >&2; exit 1;
}
for entry in \
  "src/emel/kernel/events.hpp:$EVENTS_BLOB" \
  "src/emel/kernel/detail.hpp:$DETAIL_BLOB" \
  "src/emel/kernel/x86_64/sm.hpp:$X86_SM_BLOB"; do
  path="${entry%%:*}"
  expected="${entry##*:}"
  actual="$(git -C "$EMEL_CPP_SOURCE" rev-parse "$SOURCE_COMMIT:$path")"
  [[ "$actual" == "$expected" ]] || {
    echo "error: pinned blob mismatch for $path: $actual" >&2; exit 1;
  }
done

mkdir -p "$BUILD_DIR/tmp" "$BUILD_DIR/target/cargo" "$BUILD_DIR/logs"
export TMPDIR="$BUILD_DIR/tmp"
export CARGO_TARGET_DIR="$BUILD_DIR/target/cargo"
cmake -S "$ROOT_DIR/tools/emel-kernel-im2col-reference" \
  -B "$BUILD_DIR/target/cpp-build" -G Ninja \
  -DEMEL_CPP_SOURCE_DIR="$EMEL_CPP_SOURCE" \
  -DFETCHCONTENT_SOURCE_DIR_STATEFORWARD_SML="$SML_SOURCE" \
  -DCMAKE_BUILD_TYPE=Release -DEMEL_ENABLE_TESTS=OFF -DEMEL_ENABLE_FUZZ=OFF \
  >"$BUILD_DIR/logs/cmake-configure.log" 2>&1
cmake --build "$BUILD_DIR/target/cpp-build" --target emel-kernel-im2col-reference \
  >"$BUILD_DIR/logs/cmake-build.log" 2>&1
cargo build --quiet --offline --locked --release --manifest-path "$ROOT_DIR/Cargo.toml" \
  -p emel-kernel-im2col-parity >"$BUILD_DIR/logs/cargo-build.log" 2>&1

"$BUILD_DIR/target/cpp-build/emel-kernel-im2col-reference" --benchmark >"$BUILD_DIR/bench-cpp.out"
"$CARGO_TARGET_DIR/release/emel-kernel-im2col-parity" --benchmark >"$BUILD_DIR/bench-rust.out"
cpp_ns="$(awk -F'[ =]' '/case=op_im2col/{print $4}' "$BUILD_DIR/bench-cpp.out")"
rust_ns="$(awk -F'[ =]' '/case=op_im2col/{print $4}' "$BUILD_DIR/bench-rust.out")"
cpp_output="$(awk -F'[ =]' '/case=op_im2col/{print $6}' "$BUILD_DIR/bench-cpp.out")"
rust_output="$(awk -F'[ =]' '/case=op_im2col/{print $6}' "$BUILD_DIR/bench-rust.out")"
cpp_probe="$(awk -F'[ =]' '/case=op_im2col/{print $8}' "$BUILD_DIR/bench-cpp.out")"
rust_probe="$(awk -F'[ =]' '/case=op_im2col/{print $8}' "$BUILD_DIR/bench-rust.out")"
[[ -n "$cpp_ns" && -n "$rust_ns" && "$cpp_output" == 192 && "$rust_output" == 192 \
  && "$cpp_probe" == 00000000,3f800000,3f800000 \
  && "$rust_probe" == 00000000,3f800000,3f800000 ]] || {
  echo 'error: malformed im2col benchmark output' >&2; exit 1;
}
max_ratio="${EMEL_KERNEL_BENCH_MAX_RUST_VS_CPP_RATIO:-2.0}"
awk -v rust="$rust_ns" -v cpp="$cpp_ns" -v maximum="$max_ratio" '
  BEGIN {
    numeric = "^[0-9]+([.][0-9]+)?([eE][+-]?[0-9]+)?$"
    if (rust !~ numeric || cpp !~ numeric || maximum !~ numeric ||
        (rust + 0) <= 0 || (cpp + 0) <= 0 || (rust + 0) / (cpp + 0) > (maximum + 0)) {
      exit 1
    }
  }
' || {
  echo "error: im2col Rust/C++ ratio exceeds ${max_ratio}x" >&2; exit 1;
}

TARGET_ARCH="$(rustc --print cfg | awk -F'"' '/^target_arch=/{print $2; exit}')"
if [[ -n "$SNAPSHOT_OVERRIDE" ]]; then
  SNAPSHOT="$SNAPSHOT_OVERRIDE"
else
  SNAPSHOT="$ROOT_DIR/snapshots/bench/kernel-im2col-${TARGET_ARCH}.txt"
fi
current="$BUILD_DIR/current.txt"
{
  printf '# bench_host_arch: %s\n' "$TARGET_ARCH"
  printf '# source_repository: stateforward/emel.cpp\n'
  printf '# source_commit: %s\n' "$SOURCE_COMMIT"
  printf '# source_sml_commit: %s\n' "$SML_COMMIT"
  printf '# source_kernel_events_blob: %s\n' "$EVENTS_BLOB"
  printf '# source_kernel_detail_blob: %s\n' "$DETAIL_BLOB"
  printf '# source_kernel_x86_sm_blob: %s\n' "$X86_SM_BLOB"
  printf '# benchmark_config: input=64 kernel=3 padding=1 stride=1 dilation=1 iterations=%s warmup=%s\n' "$ITERATIONS" "$WARMUP"
  printf '# benchmark_operand: dense contiguous F32 input=1.0, public Rust Kernel -> owned Im2ColKernel, reference public Kernel route\n'
  printf '# benchmark_timing: storage, actor construction, and event construction are outside measurement; actor is reused\n'
  printf '# benchmark_validation: both lanes report output=192 and probe_bits=00000000,3f800000,3f800000\n'
  printf 'kernel/im2col/op_im2col rust_ns_per_dispatch=%s cpp_ns_per_dispatch=%s rust_vs_cpp_ratio=%.6f output=192 probe_bits=00000000,3f800000,3f800000\n' \
    "$rust_ns" "$cpp_ns" "$(awk -v rust="$rust_ns" -v cpp="$cpp_ns" 'BEGIN { print (rust + 0) / (cpp + 0) }')"
} >"$current"

normalize() {
  sed -E 's/(rust_ns_per_dispatch|cpp_ns_per_dispatch|rust_vs_cpp_ratio)=[^ ]+/\1=VALUE/g' "$1"
}
if $COMPARE && [[ -f "$SNAPSHOT" ]]; then
  if ! diff -u <(normalize "$SNAPSHOT") <(normalize "$current"); then
    if ! $UPDATE; then
      echo "error: benchmark snapshot differs; rerun with --update to accept the new baseline" >&2
      exit 1
    fi
    echo "im2col benchmark snapshot differs; updating because --update is enabled" >&2
  fi
elif $COMPARE; then
  if ! $UPDATE; then
    echo "error: benchmark snapshot is missing: $SNAPSHOT" >&2
    exit 1
  fi
  echo "benchmark snapshot missing; it will be created because --update is enabled" >&2
fi
if $UPDATE; then
  mkdir -p "$(dirname "$SNAPSHOT")"
  install -m 0644 "$current" "$SNAPSHOT"
fi
if $COMPARE; then
  [[ -f "$SNAPSHOT" ]] || { echo "error: missing benchmark snapshot $SNAPSHOT" >&2; exit 1; }
  diff -u <(normalize "$SNAPSHOT") <(normalize "$current")
fi
cat "$current"
