#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ARTIFACT_DIR="${EMEL_KERNEL_FLASH_ATTN_BENCH_ARTIFACT_DIR:-$ROOT_DIR/.artifacts/kernel-goal/flash-root-bench}"
SNAPSHOT="${EMEL_KERNEL_FLASH_ATTN_BENCH_SNAPSHOT:-$ROOT_DIR/snapshots/bench/kernel-flash-attn-aarch64.txt}"
EMEL_CPP_SOURCE="${EMEL_CPP_SOURCE_DIR:-$ROOT_DIR/../emel.cpp-worktrees/pinned-843a117386ef17dc5a50549bbfc821074c2141d6}"
SML_SOURCE="${EMEL_STATEFORWARD_SML_SOURCE:-$ROOT_DIR/../../sml/sml.cpp-worktrees/pinned-49207123cd3f39767764bae774932cb48623f92f}"
SOURCE_COMMIT=843a117386ef17dc5a50549bbfc821074c2141d6
SML_COMMIT=49207123cd3f39767764bae774932cb48623f92f
UPDATE=true
COMPARE=true

for argument in "$@"; do
  case "$argument" in
    --update) UPDATE=true ;;
    --no-update) UPDATE=false ;;
    --snapshot) COMPARE=true ;;
    --no-snapshot) COMPARE=false ;;
    --update-only) UPDATE=true; COMPARE=false ;;
    --help|-h) printf '%s\n' 'usage: scripts/kernel-flash-attn-bench.sh [--snapshot|--no-snapshot] [--update|--no-update|--update-only]'; exit 0 ;;
    *) printf 'error: unknown argument: %s\n' "$argument" >&2; exit 2 ;;
  esac
done

for command in cargo cmake git rustc diff awk; do
  command -v "$command" >/dev/null 2>&1 || { printf 'error: missing %s\n' "$command" >&2; exit 2; }
done
[[ "$(git -C "$EMEL_CPP_SOURCE" rev-parse HEAD)" == "$SOURCE_COMMIT" ]] || { printf 'error: emel.cpp is not pinned\n' >&2; exit 1; }
[[ "$(git -C "$SML_SOURCE" rev-parse HEAD)" == "$SML_COMMIT" ]] || { printf 'error: stateforward-sml is not pinned\n' >&2; exit 1; }

mkdir -p "$ARTIFACT_DIR/tmp" "$ARTIFACT_DIR/target/cargo" "$ARTIFACT_DIR/logs"
export TMPDIR="$ARTIFACT_DIR/tmp"
export CARGO_TARGET_DIR="$ARTIFACT_DIR/target/cargo"
cmake -S "$ROOT_DIR/tools/emel-kernel-flash-attn-reference" \
  -B "$ARTIFACT_DIR/target/cpp-bench" -G Ninja \
  -DEMEL_CPP_SOURCE_DIR="$EMEL_CPP_SOURCE" \
  -DFETCHCONTENT_SOURCE_DIR_STATEFORWARD_SML="$SML_SOURCE" \
  -DCMAKE_BUILD_TYPE=Release -DEMEL_ENABLE_TESTS=OFF -DEMEL_ENABLE_FUZZ=OFF \
  >"$ARTIFACT_DIR/logs/cmake-configure.log" 2>&1
cmake --build "$ARTIFACT_DIR/target/cpp-bench" \
  --target emel-kernel-flash-attn-reference >"$ARTIFACT_DIR/logs/cmake-build.log" 2>&1
cargo build --quiet --offline --locked --release --manifest-path "$ROOT_DIR/Cargo.toml" \
  -p emel-kernel-flash-attn-parity >"$ARTIFACT_DIR/logs/cargo-build.log" 2>&1

"$ARTIFACT_DIR/target/cpp-bench/emel-kernel-flash-attn-reference" --benchmark \
  >"$ARTIFACT_DIR/bench-cpp.out"
"$CARGO_TARGET_DIR/release/emel-kernel-flash-attn-parity" --benchmark \
  >"$ARTIFACT_DIR/bench-rust.out"
cpp_ns="$(awk -F'[ =]' '/case=op_flash_attn_ext_canonical/{print $4}' "$ARTIFACT_DIR/bench-cpp.out")"
rust_ns="$(awk -F'[ =]' '/case=op_flash_attn_ext_canonical/{print $4}' "$ARTIFACT_DIR/bench-rust.out")"
[[ -n "$cpp_ns" && -n "$rust_ns" ]] || { printf 'error: malformed FlashAttention benchmark output\n' >&2; exit 1; }
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
  printf 'error: FlashAttention Rust/C++ ratio exceeds %sx\n' "$max_ratio" >&2; exit 1;
}

current="$ARTIFACT_DIR/bench-current.txt"
{
  printf '# bench_host_arch: %s\n' "$(rustc --print cfg | awk -F'"' '/^target_arch=/{print $2; exit}')"
  printf '# source_repository: stateforward/emel.cpp\n'
  printf '# source_commit: %s\n' "$SOURCE_COMMIT"
  printf '# source_sml_commit: %s\n' "$SML_COMMIT"
  printf '# benchmark_config: flash_attn_ext_canonical iterations=10000 warmup=1000\n'
  printf '# benchmark_operand: f32_q_f32_dst_f16_k_f16_v query_count=1\n'
  printf '# benchmark_validation: both public observer lanes report output=2\n'
  printf 'kernel/flash_attn_ext/canonical rust_ns_per_dispatch=%s cpp_ns_per_dispatch=%s rust_vs_cpp_ratio=%.6f output=2\n' \
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
