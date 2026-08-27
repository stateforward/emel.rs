#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EMEL_CPP_SOURCE="${EMEL_CPP_SOURCE_DIR:-$ROOT_DIR/../emel.cpp-worktrees/pinned-843a117386ef17dc5a50549bbfc821074c2141d6}"
SML_SOURCE="${EMEL_STATEFORWARD_SML_SOURCE:-$ROOT_DIR/../../sml/sml.cpp-worktrees/pinned-49207123cd3f39767764bae774932cb48623f92f}"
BUILD_DIR="${EMEL_KERNEL_ELEMENTWISE_BENCH_BUILD_DIR:-$ROOT_DIR/.artifacts/kernel-elementwise-bench}"
BASELINE_DIR="${EMEL_BENCH_BASELINE_DIR:-$ROOT_DIR/snapshots/bench}"
SOURCE_COMMIT=843a117386ef17dc5a50549bbfc821074c2141d6
SML_COMMIT=49207123cd3f39767764bae774932cb48623f92f
EVENTS_BLOB=4b3f02fa5ff9c5071d1fc83938cef1b22b4658b9
DETAIL_BLOB=c8a82643eabfe8f2d7883e655955f455794511b0
ANY_BLOB=a85d0942c81ef8f4d2dc293c5ceeaa63ac0e2c49
ITERATIONS=10000
RUNS=5
WARMUP=1000
SNAPSHOT=true
UPDATE=true

if [[ ! -d "$SML_SOURCE" && -d "$ROOT_DIR/../../sml/sml.cpp-worktrees/pinned-$SML_COMMIT" ]]; then
  SML_SOURCE="$ROOT_DIR/../../sml/sml.cpp-worktrees/pinned-$SML_COMMIT"
fi

usage() {
  cat <<'USAGE'
usage: scripts/kernel-elementwise-bench.sh [OPTIONS]

Options:
  --snapshot                 compare with snapshots/bench/kernel-elementwise-<arch>.txt
  --update                   update the architecture-scoped snapshot (requires --snapshot)
  --no-snapshot              run live benchmark without snapshot comparison
  --no-update                preserve the existing snapshot after live comparison
  --iterations=N             timed dispatches per sample (default: 10000)
  --runs=N                   samples (default: 5)
  --warmup-iterations=N      untimed warmup dispatches (default: 1000)
USAGE
}

for argument in "$@"; do
  case "$argument" in
    --snapshot) SNAPSHOT=true ;;
    --update) UPDATE=true ;;
    --no-snapshot) SNAPSHOT=false; UPDATE=false ;;
    --no-update) UPDATE=false ;;
    --iterations=*) ITERATIONS="${argument#*=}" ;;
    --runs=*) RUNS="${argument#*=}" ;;
    --warmup-iterations=*) WARMUP="${argument#*=}" ;;
    --help|-h) usage; exit 0 ;;
    *) echo "error: unknown argument: $argument" >&2; usage >&2; exit 2 ;;
  esac
done

if $UPDATE && ! $SNAPSHOT; then
  echo "error: --update requires --snapshot" >&2
  exit 2
fi
for value in "$ITERATIONS" "$RUNS" "$WARMUP"; do
  [[ "$value" =~ ^[1-9][0-9]*$ ]] || {
    echo "error: benchmark values must be positive integers" >&2
    exit 2
  }
done
for command in cargo cmake c++ diff git rustc getconf; do
  command -v "$command" >/dev/null 2>&1 || {
    echo "error: required command is missing: $command" >&2
    exit 2
  }
done

sha256_file() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{print $1}'
  else
    shasum -a 256 "$1" | awk '{print $1}'
  fi
}

[[ "$(git -C "$EMEL_CPP_SOURCE" rev-parse HEAD)" == "$SOURCE_COMMIT" ]] || {
  echo "error: emel.cpp is not at pinned commit $SOURCE_COMMIT" >&2
  exit 1
}
[[ "$(git -C "$SML_SOURCE" rev-parse HEAD)" == "$SML_COMMIT" ]] || {
  echo "error: stateforward-sml is not at pinned commit $SML_COMMIT" >&2
  exit 1
}
[[ "$(git -C "$EMEL_CPP_SOURCE" rev-parse "$SOURCE_COMMIT:src/emel/kernel/events.hpp")" == "$EVENTS_BLOB" ]]
[[ "$(git -C "$EMEL_CPP_SOURCE" rev-parse "$SOURCE_COMMIT:src/emel/kernel/detail.hpp")" == "$DETAIL_BLOB" ]]
[[ "$(git -C "$EMEL_CPP_SOURCE" rev-parse "$SOURCE_COMMIT:src/emel/kernel/any.hpp")" == "$ANY_BLOB" ]]

mkdir -p "$BUILD_DIR" "$BUILD_DIR/tmp" "$BUILD_DIR/cargo-target"
export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-$BUILD_DIR/cargo-target}"
export TMPDIR="${TMPDIR:-$BUILD_DIR/tmp}"
mkdir -p "$CARGO_TARGET_DIR" "$TMPDIR"

cmake -S "$ROOT_DIR/tools/emel-kernel-elementwise-reference" \
  -B "$BUILD_DIR/cpp-build" -G Ninja \
  -DEMEL_CPP_SOURCE_DIR="$EMEL_CPP_SOURCE" \
  -DFETCHCONTENT_SOURCE_DIR_STATEFORWARD_SML="$SML_SOURCE" \
  -DCMAKE_BUILD_TYPE=Release \
  -DEMEL_ENABLE_TESTS=OFF -DEMEL_ENABLE_FUZZ=OFF >/dev/null
cmake --build "$BUILD_DIR/cpp-build" --target emel-kernel-elementwise-reference >/dev/null
cargo build --quiet --locked --release --manifest-path "$ROOT_DIR/Cargo.toml" \
  -p emel-kernel-elementwise-parity

CXX_OBSERVER="$BUILD_DIR/cpp-build/emel-kernel-elementwise-reference"
RUST_OBSERVER="$CARGO_TARGET_DIR/release/emel-kernel-elementwise-parity"
[[ -x "$CXX_OBSERVER" && -x "$RUST_OBSERVER" ]]
"$CXX_OBSERVER" --benchmark "$ITERATIONS" "$RUNS" "$WARMUP" >"$BUILD_DIR/cpp.out"
"$RUST_OBSERVER" --benchmark "$ITERATIONS" "$RUNS" "$WARMUP" >"$BUILD_DIR/rust.out"

arch="$(rustc --print cfg | awk -F'"' '/^target_arch=/{print $2; exit}')"
pointer_width="$(getconf LONG_BIT)"
BASELINE="$BASELINE_DIR/kernel-elementwise-$arch.txt"
CURRENT="$BUILD_DIR/current.txt"

awk -v expected_iterations="$ITERATIONS" -v expected_runs="$RUNS" -v expected_warmup="$WARMUP" '
  function field(name,    i, parts) {
    for (i = 1; i <= NF; ++i) {
      split($i, parts, "=")
      if (parts[1] == name) return parts[2]
    }
    return ""
  }
  FNR == NR {
    if ($1 == "case=op_add") { cpp_add = field("cpp_ns_per_op"); cpp_add_output = field("output") }
    if ($1 == "case=op_mul") { cpp_mul = field("cpp_ns_per_op"); cpp_mul_output = field("output") }
    next
  }
  {
    if ($1 == "case=op_add") { rust_add = field("rust_ns_per_op"); rust_add_output = field("output") }
    if ($1 == "case=op_mul") { rust_mul = field("rust_ns_per_op"); rust_mul_output = field("output") }
  }
  END {
    if (cpp_add <= 0 || cpp_mul <= 0 || rust_add <= 0 || rust_mul <= 0 ||
        cpp_add_output != rust_add_output || cpp_mul_output != rust_mul_output) exit 1
    printf "kernel/elementwise/op_add rust_ns_per_op=%.3f cpp_ns_per_op=%.3f rust_vs_cpp_ratio=%.6f output=%s iter=%s runs=%s warmup=%s\n", rust_add, cpp_add, rust_add / cpp_add, rust_add_output, expected_iterations, expected_runs, expected_warmup
    printf "kernel/elementwise/op_mul rust_ns_per_op=%.3f cpp_ns_per_op=%.3f rust_vs_cpp_ratio=%.6f output=%s iter=%s runs=%s warmup=%s\n", rust_mul, cpp_mul, rust_mul / cpp_mul, rust_mul_output, expected_iterations, expected_runs, expected_warmup
  }
' "$BUILD_DIR/cpp.out" "$BUILD_DIR/rust.out" >"$BUILD_DIR/results.txt"

max_ratio="${EMEL_KERNEL_ELEMENTWISE_BENCH_MAX_RUST_VS_CPP_RATIO:-2.0}"
awk -v maximum="$max_ratio" '
  /^#/ { next }
  {
    ratio = 0
    for (i = 1; i <= NF; ++i) {
      split($i, parts, "=")
      if (parts[1] == "rust_vs_cpp_ratio") ratio = parts[2]
    }
    if (ratio <= 0 || ratio > maximum) exit 1
  }
' "$BUILD_DIR/results.txt" || {
  echo "error: Rust/C++ cross-lane ratio exceeds ${max_ratio}x" >&2
  exit 1
}

{
  printf '# bench_host_arch: %s\n' "$arch"
  printf '# bench_pointer_width: %s\n' "$pointer_width"
  printf '# source_repository: stateforward/emel.cpp\n'
  printf '# source_commit: %s\n' "$SOURCE_COMMIT"
  printf '# source_kernel_events_blob: %s\n' "$EVENTS_BLOB"
  printf '# source_kernel_detail_blob: %s\n' "$DETAIL_BLOB"
  printf '# source_kernel_any_blob: %s\n' "$ANY_BLOB"
  printf '# benchmark_config: count=4096 iterations=%s runs=%s sample_policy=median warmup_iterations=%s\n' "$ITERATIONS" "$RUNS" "$WARMUP"
  printf '# benchmark_operand: same dense contiguous F32 lhs=1.25 rhs=0.75 count=4096 for public Rust Kernel and pinned C++ Kernel\n'
  printf '# benchmark_timing: timed public actor dispatch only; storage and machine construction are outside measurement\n'
  printf '# benchmark_validation: both lanes report expected terminal output values and equal iteration/run configuration\n'
  printf '# reference_tool_sha256: %s\n' "$(sha256_file "$ROOT_DIR/tools/emel-kernel-elementwise-reference/main.cpp")"
  printf '# rust_observer_sha256: %s\n' "$(sha256_file "$ROOT_DIR/tools/emel-kernel-elementwise-parity/src/main.rs")"
  cat "$BUILD_DIR/results.txt"
} >"$CURRENT"

if $UPDATE; then
  mkdir -p "$BASELINE_DIR"
  install -m 0644 "$CURRENT" "$BASELINE"
fi
if $SNAPSHOT; then
  [[ -f "$BASELINE" ]] || { echo "error: missing benchmark baseline: $BASELINE" >&2; exit 1; }
  diff -u <(sed '/ns_per_op=/s/ rust_ns_per_op=[^ ]* cpp_ns_per_op=[^ ]* rust_vs_cpp_ratio=[^ ]*/ rust_ns_per_op=VALUE cpp_ns_per_op=VALUE rust_vs_cpp_ratio=VALUE/' "$BASELINE") \
    <(sed '/ns_per_op=/s/ rust_ns_per_op=[^ ]* cpp_ns_per_op=[^ ]* rust_vs_cpp_ratio=[^ ]*/ rust_ns_per_op=VALUE cpp_ns_per_op=VALUE rust_vs_cpp_ratio=VALUE/' "$CURRENT") >/dev/null || {
    echo "error: benchmark baseline metadata/configuration differs" >&2
    exit 1
  }
fi
cat "$CURRENT"
