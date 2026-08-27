#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PARITY_ARTIFACT_DIR="${EMEL_KERNEL_GET_ROWS_PARITY_ARTIFACT_DIR:-$ROOT_DIR/.artifacts/kernel-goal/get-rows-parity}"
SNAPSHOT="${EMEL_KERNEL_GET_ROWS_BENCH_SNAPSHOT:-$ROOT_DIR/snapshots/bench/kernel-get-rows-aarch64.txt}"
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
    --help|-h) printf '%s\n' 'usage: scripts/kernel-get-rows-bench.sh [--snapshot|--no-snapshot] [--update|--no-update|--update-only]'; exit 0 ;;
    *) printf 'error: unknown argument: %s\n' "$argument" >&2; exit 2 ;;
  esac
done

for command in cargo cmake git rustc diff awk; do
  command -v "$command" >/dev/null 2>&1 || { printf 'error: missing %s\n' "$command" >&2; exit 2; }
done
[[ "$(git -C "$EMEL_CPP_SOURCE" rev-parse HEAD)" == "$SOURCE_COMMIT" ]] || { printf 'error: emel.cpp is not pinned\n' >&2; exit 1; }
[[ -z "$(git -C "$EMEL_CPP_SOURCE" status --porcelain=v1 --untracked-files=all)" ]] || { printf 'error: emel.cpp tree is dirty\n' >&2; exit 1; }
[[ "$(git -C "$SML_SOURCE" rev-parse HEAD)" == "$SML_COMMIT" ]] || { printf 'error: stateforward-sml is not pinned\n' >&2; exit 1; }
[[ -z "$(git -C "$SML_SOURCE" status --porcelain=v1 --untracked-files=all)" ]] || { printf 'error: stateforward-sml tree is dirty\n' >&2; exit 1; }

mkdir -p "$PARITY_ARTIFACT_DIR/tmp" "$PARITY_ARTIFACT_DIR/target/cargo" "$PARITY_ARTIFACT_DIR/logs"
export TMPDIR="$PARITY_ARTIFACT_DIR/tmp"
export CARGO_TARGET_DIR="$PARITY_ARTIFACT_DIR/target/cargo"
cmake -S "$ROOT_DIR/tools/emel-kernel-get-rows-reference" -B "$PARITY_ARTIFACT_DIR/target/cpp-bench" -G Ninja \
  -DEMEL_CPP_SOURCE_DIR="$EMEL_CPP_SOURCE" -DFETCHCONTENT_SOURCE_DIR_STATEFORWARD_SML="$SML_SOURCE" \
  -DCMAKE_BUILD_TYPE=Release -DEMEL_ENABLE_TESTS=OFF -DEMEL_ENABLE_FUZZ=OFF \
  >"$PARITY_ARTIFACT_DIR/logs/bench-cmake-configure.log" 2>&1
cmake --build "$PARITY_ARTIFACT_DIR/target/cpp-bench" --target emel-kernel-get-rows-reference \
  >"$PARITY_ARTIFACT_DIR/logs/bench-cmake-build.log" 2>&1
cargo build --quiet --offline --locked --release --manifest-path "$ROOT_DIR/Cargo.toml" \
  -p emel-kernel-get-rows-parity >"$PARITY_ARTIFACT_DIR/logs/bench-cargo-build.log" 2>&1

"$PARITY_ARTIFACT_DIR/target/cpp-bench/emel-kernel-get-rows-reference" --benchmark >"$PARITY_ARTIFACT_DIR/bench-cpp.out"
"$CARGO_TARGET_DIR/release/emel-kernel-get-rows-parity" --benchmark >"$PARITY_ARTIFACT_DIR/bench-rust.out"
cpp_ns="$(awk -F'[ =]' '/case=op_get_rows_f32/{print $4}' "$PARITY_ARTIFACT_DIR/bench-cpp.out")"
rust_ns="$(awk -F'[ =]' '/case=op_get_rows_f32/{print $4}' "$PARITY_ARTIFACT_DIR/bench-rust.out")"
[[ -n "$cpp_ns" && -n "$rust_ns" ]] || { printf 'error: malformed get_rows benchmark output\n' >&2; exit 1; }
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
  printf 'error: get_rows Rust/C++ ratio exceeds %sx\n' "$max_ratio" >&2; exit 1;
}

current="$PARITY_ARTIFACT_DIR/bench-current.txt"
{
  printf '# bench_host_arch: %s\n' "$(rustc --print cfg | awk -F'"' '/^target_arch=/{print $2; exit}')"
  printf '# source_repository: stateforward/emel.cpp\n'
  printf '# source_commit: %s\n' "$SOURCE_COMMIT"
  printf '# source_sml_commit: %s\n' "$SML_COMMIT"
  printf '# benchmark_config: k=rows2xcols2 iterations=10000 warmup=1000\n'
  printf '# benchmark_operand: dense explicit-stride F32 get_rows root route\n'
  printf '# benchmark_validation: both public observer lanes report output=4\n'
  printf 'kernel/get_rows/op_get_rows_f32 rust_ns_per_dispatch=%s cpp_ns_per_dispatch=%s rust_vs_cpp_ratio=%.6f output=4\n' \
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
