#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${EMEL_MODEL_GENERATION_BENCH_BUILD_DIR:-$ROOT_DIR/target/model-generation-bench}"
SOURCE_DIR="${EMEL_CPP_SOURCE_DIR:-$ROOT_DIR/../emel.cpp}"
ITERATIONS=1000000
RUNS=11
WARMUP=100000
SNAPSHOT=false
UPDATE=false
MAX_RATIO="${EMEL_BENCH_MAX_REGRESSION_RATIO:-2.0}"
for argument in "$@"; do
  case "$argument" in
    --iterations=*) ITERATIONS="${argument#*=}" ;;
    --runs=*) RUNS="${argument#*=}" ;;
    --warmup-iterations=*) WARMUP="${argument#*=}" ;;
    --snapshot|--compare) SNAPSHOT=true ;;
    --update) UPDATE=true ;;
    *) echo "error: unknown argument: $argument" >&2; exit 2 ;;
  esac
done
$UPDATE && ! $SNAPSHOT && { echo "error: --update requires --snapshot" >&2; exit 2; }

mkdir -p "$BUILD_DIR"
EMEL_CPP_SOURCE_DIR="$SOURCE_DIR" \
  EMEL_MODEL_GENERATION_PARITY_BUILD_DIR="$BUILD_DIR/reference" \
  "$ROOT_DIR/scripts/model-generation-parity.sh" --live >/dev/null
CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-$ROOT_DIR/target}" cargo build --quiet --locked --release --manifest-path "$ROOT_DIR/Cargo.toml" \
  -p emel-model --example generation_observer
RUST_OBSERVER="${CARGO_TARGET_DIR:-$ROOT_DIR/target}/release/examples/generation_observer"
[[ -x "$RUST_OBSERVER" ]] || { echo "error: missing Rust observer: $RUST_OBSERVER" >&2; exit 1; }
"$BUILD_DIR/reference/cpp-observer" --generation-benchmark "$ITERATIONS" "$RUNS" "$WARMUP" >"$BUILD_DIR/cpp.out"
"$RUST_OBSERVER" --benchmark "$ITERATIONS" "$RUNS" "$WARMUP" >"$BUILD_DIR/rust.out"

cpp="$(awk -F'[ =]' '{for(i=1;i<=NF;i++) if($i=="cpp_ns_per_visit") print $(i+1)}' "$BUILD_DIR/cpp.out")"
rust="$(awk -F'[ =]' '{for(i=1;i<=NF;i++) if($i=="rust_ns_per_visit") print $(i+1)}' "$BUILD_DIR/rust.out")"
ratio="$(awk -v rust="$rust" -v cpp="$cpp" 'BEGIN { printf "%.6f", rust / cpp }')"
awk -v ratio="$ratio" -v maximum="$MAX_RATIO" 'BEGIN { exit !(ratio <= maximum) }' || {
  echo "error: model generation Rust/C++ ratio $ratio exceeds $MAX_RATIO" >&2
  exit 1
}
arch="$(rustc --print cfg | awk -F'"' '/^target_arch=/{print $2; exit}')"
CURRENT="$BUILD_DIR/current.txt"
BASELINE="$ROOT_DIR/snapshots/bench/model-generation-$arch.txt"
{
  printf '# source_commit: 843a117386ef17dc5a50549bbfc821074c2141d6\n'
  printf '# source_generation_header_blob: d521cf68e1bf52a2a193bbdb460741772199b318\n'
  printf '# source_generation_implementation_blob: 099058ccd441d1dc6bebbb0c4994070d2f533c47\n'
  printf '# benchmark_operand: same prebound attention+QK block index 0; Rust public BlockVisit vs C++ public lookup_block_view; full returned view black-boxed and index checksum observed\n'
  printf '# benchmark_config: iterations=%s runs=%s warmup_iterations=%s sample_policy=median max_ratio=%s\n' "$ITERATIONS" "$RUNS" "$WARMUP" "$MAX_RATIO"
  printf 'model/generation/block_visit rust_ns_per_op=%s cpp_ns_per_op=%s rust_vs_cpp_ratio=%s outcome=found checksum=0\n' "$rust" "$cpp" "$ratio"
} >"$CURRENT"
if $UPDATE; then
  mkdir -p "$(dirname "$BASELINE")"
  install -m 0644 "$CURRENT" "$BASELINE"
fi
if $SNAPSHOT; then
  [[ -f "$BASELINE" ]] || {
    echo "error: missing model generation benchmark baseline: $BASELINE" >&2
    exit 1
  }
  awk -v max_ratio="$MAX_RATIO" '
    function field(name,    i, parts) {
      for (i = 1; i <= NF; ++i) {
        split($i, parts, "=")
        if (parts[1] == name) return parts[2]
      }
      return ""
    }
    FNR == NR {
      if ($0 ~ /^# source_commit:/) baseline_source = $0
      if ($0 ~ /^# source_generation_header_blob:/) baseline_header = $0
      if ($0 ~ /^# source_generation_implementation_blob:/) baseline_impl = $0
      if ($0 ~ /^# benchmark_operand:/) baseline_operand = $0
      if ($0 ~ /^# benchmark_config:/) baseline_config = $0
      if ($1 == "model/generation/block_visit") {
        baseline_rust = field("rust_ns_per_op")
        baseline_outcome = field("outcome")
        baseline_checksum = field("checksum")
      }
      next
    }
    {
      if ($0 ~ /^# source_commit:/) current_source = $0
      if ($0 ~ /^# source_generation_header_blob:/) current_header = $0
      if ($0 ~ /^# source_generation_implementation_blob:/) current_impl = $0
      if ($0 ~ /^# benchmark_operand:/) current_operand = $0
      if ($0 ~ /^# benchmark_config:/) current_config = $0
      if ($1 == "model/generation/block_visit") {
        current_rust = field("rust_ns_per_op")
        current_outcome = field("outcome")
        current_checksum = field("checksum")
      }
    }
    END {
      comparable = baseline_source != "" && baseline_source == current_source &&
        baseline_header != "" && baseline_header == current_header &&
        baseline_impl != "" && baseline_impl == current_impl &&
        baseline_operand != "" && baseline_operand == current_operand &&
        baseline_config != "" && baseline_config == current_config &&
        baseline_outcome == current_outcome &&
        baseline_checksum == current_checksum
      if (!comparable) {
        print "error: model generation benchmark baseline is not comparable" > "/dev/stderr"
        exit 1
      }
      if (baseline_rust <= 0 || current_rust <= 0 ||
          current_rust > baseline_rust * max_ratio) exit 1
    }
  ' "$BASELINE" "$CURRENT"
fi
cat "$CURRENT"
