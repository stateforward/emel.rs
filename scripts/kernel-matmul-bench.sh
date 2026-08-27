#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EMEL_CPP_SOURCE="${EMEL_CPP_SOURCE_DIR:-$ROOT_DIR/../emel.cpp-worktrees/pinned-843a117386ef17dc5a50549bbfc821074c2141d6}"
SML_SOURCE="${EMEL_STATEFORWARD_SML_SOURCE:-$ROOT_DIR/../../sml/sml.cpp-worktrees/pinned-49207123cd3f39767764bae774932cb48623f92f}"
BUILD_DIR="${EMEL_KERNEL_MATMUL_BENCH_ARTIFACT_DIR:-$ROOT_DIR/.artifacts/kernel-goal/matmul-bench}"
SNAPSHOT_OVERRIDE="${EMEL_KERNEL_MATMUL_BENCH_SNAPSHOT:-}"
SOURCE_COMMIT=843a117386ef17dc5a50549bbfc821074c2141d6
SML_COMMIT=49207123cd3f39767764bae774932cb48623f92f
EVENTS_BLOB=4b3f02fa5ff9c5071d1fc83938cef1b22b4658b9
DETAIL_BLOB=c8a82643eabfe8f2d7883e655955f455794511b0
X86_SM_BLOB=0b4d635ebbd0fbd52dbca8a2345547fb571205c8
X86_GUARDS_BLOB=cb3dac8253f8417c9b44acff1de414f6d0a3a3cf
X86_ACTIONS_BLOB=d45558f5eb96950f43c16a09d768cb4f382d6d61
AARCH64_SM_BLOB=865a9cc6ba6115382ed043c464f3d62bcd851357
AARCH64_GUARDS_BLOB=c25714566ec9a02679daef85089544575123408e
AARCH64_ACTIONS_BLOB=267d4f74e6e7498155c8535920322ffef2c02fb6
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
      printf '%s\n' 'usage: scripts/kernel-matmul-bench.sh [--snapshot|--no-snapshot] [--update|--no-update|--update-only]'
      exit 0
      ;;
    *) printf 'error: unknown argument: %s\n' "$argument" >&2; exit 2 ;;
  esac
done

for command in cargo cmake git rustc diff shasum awk; do
  command -v "$command" >/dev/null 2>&1 || { printf 'error: missing %s\n' "$command" >&2; exit 2; }
done

[[ "$(git -C "$EMEL_CPP_SOURCE" rev-parse HEAD)" == "$SOURCE_COMMIT" ]] || {
  printf 'error: emel.cpp is not at pinned commit %s\n' "$SOURCE_COMMIT" >&2; exit 1;
}
[[ -z "$(git -C "$EMEL_CPP_SOURCE" status --porcelain=v1 --untracked-files=all)" ]] || {
  printf 'error: emel.cpp reference tree is dirty\n' >&2; exit 1;
}
[[ "$(git -C "$SML_SOURCE" rev-parse HEAD)" == "$SML_COMMIT" ]] || {
  printf 'error: stateforward-sml is not at pinned commit %s\n' "$SML_COMMIT" >&2; exit 1;
}
[[ -z "$(git -C "$SML_SOURCE" status --porcelain=v1 --untracked-files=all)" ]] || {
  printf 'error: stateforward-sml reference tree is dirty\n' >&2; exit 1;
}

mkdir -p "$BUILD_DIR/tmp" "$BUILD_DIR/target/cargo" "$BUILD_DIR/logs"
export TMPDIR="$BUILD_DIR/tmp"
export CARGO_TARGET_DIR="$BUILD_DIR/target/cargo"

cmake -S "$ROOT_DIR/tools/emel-kernel-matmul-reference" \
  -B "$BUILD_DIR/target/cpp-bench" -G Ninja \
  -DEMEL_CPP_SOURCE_DIR="$EMEL_CPP_SOURCE" \
  -DFETCHCONTENT_SOURCE_DIR_STATEFORWARD_SML="$SML_SOURCE" \
  -DCMAKE_BUILD_TYPE=Release -DEMEL_ENABLE_TESTS=OFF -DEMEL_ENABLE_FUZZ=OFF \
  >"$BUILD_DIR/logs/cmake-configure.log" 2>&1
cmake --build "$BUILD_DIR/target/cpp-bench" --target emel-kernel-matmul-reference \
  >"$BUILD_DIR/logs/cmake-build.log" 2>&1
cargo build --quiet --offline --locked --release --manifest-path "$ROOT_DIR/Cargo.toml" \
  -p emel-kernel-matmul-parity >"$BUILD_DIR/logs/cargo-build.log" 2>&1

"$BUILD_DIR/target/cpp-bench/emel-kernel-matmul-reference" --benchmark >"$BUILD_DIR/bench-cpp.out"
"$CARGO_TARGET_DIR/release/emel-kernel-matmul-parity" --benchmark >"$BUILD_DIR/bench-rust.out"
cpp_ns="$(awk -F'[ =]' '/case=op_mul_mat_f32/{print $4}' "$BUILD_DIR/bench-cpp.out")"
rust_ns="$(awk -F'[ =]' '/case=op_mul_mat_f32/{print $4}' "$BUILD_DIR/bench-rust.out")"
cpp_output="$(awk -F'[ =]' '/case=op_mul_mat_f32/{print $6}' "$BUILD_DIR/bench-cpp.out")"
rust_output="$(awk -F'[ =]' '/case=op_mul_mat_f32/{print $6}' "$BUILD_DIR/bench-rust.out")"
[[ -n "$cpp_ns" && -n "$rust_ns" && "$cpp_output" == 16 && "$rust_output" == 16 ]] || {
  printf 'error: malformed matmul benchmark output\n' >&2; exit 1;
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
  printf 'error: matmul Rust/C++ ratio exceeds %sx\n' "$max_ratio" >&2; exit 1;
}

TARGET_ARCH="$(rustc --print cfg | awk -F'"' '/^target_arch=/{print $2; exit}')"
if [[ -n "$SNAPSHOT_OVERRIDE" ]]; then
  SNAPSHOT="$SNAPSHOT_OVERRIDE"
else
  SNAPSHOT="$ROOT_DIR/snapshots/bench/kernel-matmul-${TARGET_ARCH}.txt"
fi
case "$TARGET_ARCH" in
  aarch64) ACTIVE_SM="$AARCH64_SM_BLOB"; ACTIVE_GUARDS="$AARCH64_GUARDS_BLOB"; ACTIVE_ACTIONS="$AARCH64_ACTIONS_BLOB" ;;
  x86_64) ACTIVE_SM="$X86_SM_BLOB"; ACTIVE_GUARDS="$X86_GUARDS_BLOB"; ACTIVE_ACTIONS="$X86_ACTIONS_BLOB" ;;
  *) printf 'error: unsupported target architecture: %s\n' "$TARGET_ARCH" >&2; exit 1 ;;
esac
current="$BUILD_DIR/bench-current.txt"
{
  printf '# bench_host_arch: %s\n' "$TARGET_ARCH"
  printf '# source_repository: stateforward/emel.cpp\n'
  printf '# source_commit: %s\n' "$SOURCE_COMMIT"
  printf '# source_sml_commit: %s\n' "$SML_COMMIT"
  printf '# source_kernel_events_blob: %s\n' "$EVENTS_BLOB"
  printf '# source_kernel_detail_blob: %s\n' "$DETAIL_BLOB"
  printf '# source_kernel_x86_sm_blob: %s\n' "$X86_SM_BLOB"
  printf '# source_kernel_x86_guards_blob: %s\n' "$X86_GUARDS_BLOB"
  printf '# source_kernel_x86_actions_blob: %s\n' "$X86_ACTIONS_BLOB"
  printf '# source_kernel_aarch64_sm_blob: %s\n' "$AARCH64_SM_BLOB"
  printf '# source_kernel_aarch64_guards_blob: %s\n' "$AARCH64_GUARDS_BLOB"
  printf '# source_kernel_aarch64_actions_blob: %s\n' "$AARCH64_ACTIONS_BLOB"
  printf '# active_kernel_sm_blob: %s\n' "$ACTIVE_SM"
  printf '# active_kernel_guards_blob: %s\n' "$ACTIVE_GUARDS"
  printf '# active_kernel_actions_blob: %s\n' "$ACTIVE_ACTIONS"
  printf '# benchmark_config: k=16,m=1,n=1 iterations=%s warmup=%s\n' "$ITERATIONS" "$WARMUP"
  printf '# benchmark_operand: dense explicit nonzero-stride F32-by-F32-to-F32 root matmul\n'
  printf '# benchmark_validation: both public observer lanes report output=16\n'
  printf 'kernel/matmul/op_mul_mat_f32 rust_ns_per_dispatch=%s cpp_ns_per_dispatch=%s rust_vs_cpp_ratio=%.6f output=16\n' \
    "$rust_ns" "$cpp_ns" "$(awk -v rust="$rust_ns" -v cpp="$cpp_ns" 'BEGIN { print (rust + 0) / (cpp + 0) }')"
} >"$current"
if $COMPARE; then
  [[ -f "$SNAPSHOT" ]] || { printf 'error: missing benchmark snapshot %s\n' "$SNAPSHOT" >&2; exit 1; }
  diff -u <(sed -E 's/(rust_ns_per_dispatch|cpp_ns_per_dispatch|rust_vs_cpp_ratio)=[^ ]+/\1=VALUE/g' "$SNAPSHOT") \
    <(sed -E 's/(rust_ns_per_dispatch|cpp_ns_per_dispatch|rust_vs_cpp_ratio)=[^ ]+/\1=VALUE/g' "$current")
fi
if $UPDATE; then
  mkdir -p "$(dirname "$SNAPSHOT")"
  install -m 0644 "$current" "$SNAPSHOT"
fi
cat "$current"
