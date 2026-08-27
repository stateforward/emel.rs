#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EMEL_CPP_SOURCE="${EMEL_CPP_SOURCE_DIR:-$ROOT_DIR/../emel.cpp-worktrees/pinned-843a117386ef17dc5a50549bbfc821074c2141d6}"
SML_SOURCE="${EMEL_STATEFORWARD_SML_SOURCE:-$ROOT_DIR/../../sml/sml.cpp-worktrees/pinned-49207123cd3f39767764bae774932cb48623f92f}"
BUILD_DIR="${EMEL_KERNEL_UNARY_BENCH_ARTIFACT_DIR:-$ROOT_DIR/.artifacts/kernel-goal/unary-bench}"
SNAPSHOT="${EMEL_KERNEL_UNARY_BENCH_SNAPSHOT:-$ROOT_DIR/snapshots/bench/kernel-unary-aarch64.txt}"
SOURCE_COMMIT=843a117386ef17dc5a50549bbfc821074c2141d6
SML_COMMIT=49207123cd3f39767764bae774932cb48623f92f
ITERATIONS=10000
WARMUP=1000
UPDATE=true
COMPARE=true

for argument in "$@"; do
  case "$argument" in
    --snapshot) COMPARE=true ;;
    --no-snapshot) COMPARE=false; UPDATE=false ;;
    --update) UPDATE=true ;;
    --no-update) UPDATE=false ;;
    --help|-h)
      printf '%s\n' 'usage: scripts/kernel-unary-bench.sh [--snapshot|--no-snapshot] [--update|--no-update]'
      exit 0
      ;;
    *) echo "error: unknown argument: $argument" >&2; exit 2 ;;
  esac
done
for command in cargo cmake diff git rustc; do
  command -v "$command" >/dev/null 2>&1 || { echo "error: missing $command" >&2; exit 2; }
done
[[ "$ITERATIONS" =~ ^[1-9][0-9]*$ && "$WARMUP" =~ ^[1-9][0-9]*$ ]] || {
  echo 'error: benchmark iterations and warmup must be positive integers' >&2; exit 2;
}
[[ "$(git -C "$EMEL_CPP_SOURCE" rev-parse HEAD)" == "$SOURCE_COMMIT" ]] || {
  echo "error: emel.cpp is not at pinned commit $SOURCE_COMMIT" >&2; exit 1;
}
[[ "$(git -C "$SML_SOURCE" rev-parse HEAD)" == "$SML_COMMIT" ]] || {
  echo "error: stateforward-sml is not at pinned commit $SML_COMMIT" >&2; exit 1;
}
mkdir -p "$BUILD_DIR/tmp" "$BUILD_DIR/target/cargo" "$BUILD_DIR/logs"
export TMPDIR="$BUILD_DIR/tmp"
export CARGO_TARGET_DIR="$BUILD_DIR/target/cargo"
cmake -S "$ROOT_DIR/tools/emel-kernel-unary-reference" \
  -B "$BUILD_DIR/target/cpp-build" -G Ninja \
  -DEMEL_CPP_SOURCE_DIR="$EMEL_CPP_SOURCE" \
  -DFETCHCONTENT_SOURCE_DIR_STATEFORWARD_SML="$SML_SOURCE" \
  -DCMAKE_BUILD_TYPE=Release -DEMEL_ENABLE_TESTS=OFF -DEMEL_ENABLE_FUZZ=OFF \
  >"$BUILD_DIR/logs/cmake-configure.log" 2>&1
cmake --build "$BUILD_DIR/target/cpp-build" --target emel-kernel-unary-reference \
  >"$BUILD_DIR/logs/cmake-build.log" 2>&1
cargo build --quiet --offline --release --manifest-path "$ROOT_DIR/Cargo.toml" \
  -p emel-kernel-unary-parity >"$BUILD_DIR/logs/cargo-build.log" 2>&1
"$BUILD_DIR/target/cpp-build/emel-kernel-unary-reference" --benchmark \
  >"$BUILD_DIR/cpp.out"
"$CARGO_TARGET_DIR/release/emel-kernel-unary-parity" --benchmark >"$BUILD_DIR/rust.out"
cpp_ns="$(awk -F'[ =]' '/case=op_unary_abs/{print $4}' "$BUILD_DIR/cpp.out")"
rust_ns="$(awk -F'[ =]' '/case=op_unary_abs/{print $4}' "$BUILD_DIR/rust.out")"
cpp_output="$(awk -F'[ =]' '/case=op_unary_abs/{print $6}' "$BUILD_DIR/cpp.out")"
rust_output="$(awk -F'[ =]' '/case=op_unary_abs/{print $6}' "$BUILD_DIR/rust.out")"
[[ -n "$cpp_ns" && -n "$rust_ns" && "$cpp_output" == 1.25 && "$rust_output" == 1.25 ]] || {
  echo 'error: malformed unary benchmark output' >&2; exit 1;
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
  echo "error: unary Rust/C++ ratio exceeds ${max_ratio}x" >&2; exit 1;
}
arch="$(rustc --print cfg | awk -F'"' '/^target_arch=/{print $2; exit}')"
current="$BUILD_DIR/current.txt"
{
  printf '# bench_host_arch: %s\n' "$arch"
  printf '# source_repository: stateforward/emel.cpp\n'
  printf '# source_commit: %s\n' "$SOURCE_COMMIT"
  printf '# source_sml_commit: %s\n' "$SML_COMMIT"
  printf '# benchmark_config: count=1024 iterations=%s warmup=%s\n' "$ITERATIONS" "$WARMUP"
  printf '# benchmark_operand: dense contiguous F32 OpUnary::Abs input=1.25\n'
  printf '# benchmark_validation: both public observer lanes report output=1.25\n'
  printf '# rust_execution: public_Kernel_process_event_to_owned_UnaryKernel\n'
  printf 'kernel/unary/op_abs rust_ns_per_dispatch=%s cpp_ns_per_dispatch=%s rust_vs_cpp_ratio=%.6f output=1.25\n' \
    "$rust_ns" "$cpp_ns" "$(awk -v rust="$rust_ns" -v cpp="$cpp_ns" 'BEGIN { print (rust + 0) / (cpp + 0) }')"
} >"$current"
if $UPDATE; then
  mkdir -p "$(dirname "$SNAPSHOT")"
  install -m 0644 "$current" "$SNAPSHOT"
fi
if $COMPARE; then
  [[ -f "$SNAPSHOT" ]] || { echo "error: missing benchmark snapshot $SNAPSHOT" >&2; exit 1; }
  diff -u <(sed -E 's/(rust_ns_per_dispatch|cpp_ns_per_dispatch|rust_vs_cpp_ratio)=[^ ]+/'"\1=VALUE"'/g' "$SNAPSHOT") \
    <(sed -E 's/(rust_ns_per_dispatch|cpp_ns_per_dispatch|rust_vs_cpp_ratio)=[^ ]+/'"\1=VALUE"'/g' "$current")
fi
cat "$current"
