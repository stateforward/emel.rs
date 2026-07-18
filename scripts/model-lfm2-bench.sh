#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${EMEL_MODEL_LFM2_BENCH_BUILD_DIR:-$ROOT_DIR/target/model-lfm2-bench}"
SOURCE_DIR="${EMEL_CPP_SOURCE_DIR:-$ROOT_DIR/../emel.cpp}"
ITERATIONS=1000000
RUNS=11
WARMUP=100000
PROCESS_RUNS=5
SNAPSHOT=false
UPDATE=false
MAX_RATIO="${EMEL_LFM2_BENCH_MAX_REGRESSION_RATIO:-2.0}"

for argument in "$@"; do
  case "$argument" in
    --iterations=*) ITERATIONS="${argument#*=}" ;;
    --runs=*) RUNS="${argument#*=}" ;;
    --warmup-iterations=*) WARMUP="${argument#*=}" ;;
    --process-runs=*) PROCESS_RUNS="${argument#*=}" ;;
    --snapshot|--compare) SNAPSHOT=true ;;
    --update) UPDATE=true ;;
    *) echo "error: unknown argument: $argument" >&2; exit 2 ;;
  esac
done

$UPDATE && ! $SNAPSHOT && { echo "error: --update requires --snapshot" >&2; exit 2; }
[[ "$ITERATIONS" =~ ^[1-9][0-9]*$ && "$RUNS" =~ ^[1-9][0-9]*$ &&
   "$WARMUP" =~ ^[0-9]+$ && "$PROCESS_RUNS" =~ ^[1-9][0-9]*$ &&
   $((PROCESS_RUNS % 2)) -eq 1 ]] || {
  echo "error: benchmark counts must be positive integers and process runs must be odd" >&2
  exit 2
}

mkdir -p "$BUILD_DIR"
EMEL_CPP_SOURCE_DIR="$SOURCE_DIR" \
  EMEL_MODEL_LFM2_PARITY_BUILD_DIR="$BUILD_DIR/reference" \
  "$ROOT_DIR/scripts/model-lfm2-parity.sh" --live >/dev/null
cargo build --quiet --locked --release --manifest-path "$ROOT_DIR/Cargo.toml" \
  -p emel-model --example lfm2_observer
RUNNER=()
SCHEDULING_POLICY=default
SCHEDULING_KEY=portable-default
if [[ "$(uname -s)" == "Darwin" ]] && command -v taskpolicy >/dev/null 2>&1; then
  RUNNER=(taskpolicy -t 0 -l 0)
  SCHEDULING_POLICY=darwin-throughput-0-latency-0
  SCHEDULING_KEY=darwin-tier0
fi
: >"$BUILD_DIR/cpp.out"
: >"$BUILD_DIR/rust.out"
process_run=0
while [[ "$process_run" -lt "$PROCESS_RUNS" ]]; do
  "${RUNNER[@]}" "$BUILD_DIR/reference/cpp-observer" --benchmark "$ITERATIONS" "$RUNS" "$WARMUP" >>"$BUILD_DIR/cpp.out"
  "${RUNNER[@]}" "$ROOT_DIR/target/release/examples/lfm2_observer" --benchmark "$ITERATIONS" "$RUNS" "$WARMUP" >>"$BUILD_DIR/rust.out"
  process_run=$((process_run + 1))
done

cpp_checksum="$(awk -F'[ =]' '{for(i=1;i<=NF;i++) if($i=="checksum") print $(i+1)}' "$BUILD_DIR/cpp.out" | sort -u)"
rust_checksum="$(awk -F'[ =]' '{for(i=1;i<=NF;i++) if($i=="checksum") print $(i+1)}' "$BUILD_DIR/rust.out" | sort -u)"
[[ "$cpp_checksum" =~ ^[1-9][0-9]*$ && "$cpp_checksum" == "$rust_checksum" ]] || {
  echo "error: benchmark checksums are zero, unstable, or unequal" >&2
  exit 1
}

SAMPLES="$BUILD_DIR/process-samples.txt"
paste "$BUILD_DIR/cpp.out" "$BUILD_DIR/rust.out" | awk -F'[ =\t]+' '
  {
    cpp = ""; rust = ""
    for (i = 1; i <= NF; ++i) {
      if ($i == "cpp_ns_per_visit") cpp = $(i + 1)
      if ($i == "rust_ns_per_visit") rust = $(i + 1)
    }
    if (cpp <= 0 || rust <= 0) exit 1
    printf "model/lfm2/block_visit/process run=%d cpp_ns_per_op=%s rust_ns_per_op=%s paired_ratio=%.6f\n", NR, cpp, rust, rust / cpp
  }
' >"$SAMPLES"
middle=$((PROCESS_RUNS / 2 + 1))
cpp="$(awk -F'[ =]' '{for(i=1;i<=NF;i++) if($i=="cpp_ns_per_op") print $(i+1)}' "$SAMPLES" | sort -n | sed -n "${middle}p")"
rust="$(awk -F'[ =]' '{for(i=1;i<=NF;i++) if($i=="rust_ns_per_op") print $(i+1)}' "$SAMPLES" | sort -n | sed -n "${middle}p")"
ratio="$(awk -F'[ =]' '{for(i=1;i<=NF;i++) if($i=="paired_ratio") print $(i+1)}' "$SAMPLES" | sort -n | sed -n "${middle}p")"
awk -v ratio="$ratio" -v maximum="$MAX_RATIO" 'BEGIN { exit !(ratio <= maximum) }' || {
  echo "error: model LFM2 Rust/C++ ratio $ratio exceeds $MAX_RATIO" >&2
  exit 1
}

arch="$(rustc --print cfg | awk -F'"' '/^target_arch=/{print $2; exit}')"
target_os="$(rustc --print cfg | awk -F'"' '/^target_os=/{print $2; exit}')"
CURRENT="$BUILD_DIR/current.txt"
BASELINE="$ROOT_DIR/snapshots/bench/model-lfm2-$target_os-$arch-$SCHEDULING_KEY.txt"
{
  printf '# source_commit: 843a117386ef17dc5a50549bbfc821074c2141d6\n'
  printf '# source_lfm2_detail_header_blob: af0e7d509a3854a847bdde2dad831c5ed60de379\n'
  printf '# source_lfm2_detail_implementation_blob: a6f215e74ed793ea8f606048c00faa80c37056be\n'
  printf '# benchmark_operand: same alternating completed LFM2.5-230M hybrid block indices 0 (shortconv) and 2 (attention); Rust public family BlockVisit vs C++ public generation lookup_block_view over the source-built LFM2 contract; full descriptor black-boxed and nonzero index checksum observed\n'
  printf '# benchmark_config: iterations=%s runs=%s warmup_iterations=%s process_runs=%s sample_policy=median_paired_process_ratio scheduling=%s max_ratio=%s\n' "$ITERATIONS" "$RUNS" "$WARMUP" "$PROCESS_RUNS" "$SCHEDULING_POLICY" "$MAX_RATIO"
  cat "$SAMPLES"
  printf 'model/lfm2/block_visit rust_median_ns_per_op=%s cpp_median_ns_per_op=%s paired_ratio_median=%s outcome=found checksum=%s\n' "$rust" "$cpp" "$ratio" "$rust_checksum"
} >"$CURRENT"

if $UPDATE; then
  mkdir -p "$(dirname "$BASELINE")"
  install -m 0644 "$CURRENT" "$BASELINE"
fi
if $SNAPSHOT; then
  [[ -f "$BASELINE" ]] || { echo "error: missing model LFM2 benchmark baseline: $BASELINE" >&2; exit 1; }
  awk -v max_ratio="$MAX_RATIO" '
    function field(name,    i, parts) {
      for (i = 1; i <= NF; ++i) {
        split($i, parts, "=")
        if (parts[1] == name) return parts[2]
      }
      return ""
    }
    FNR == NR {
      if ($0 ~ /^# source_/) baseline_source = baseline_source $0
      if ($0 ~ /^# benchmark_operand:/) baseline_operand = $0
      if ($0 ~ /^# benchmark_config:/) baseline_config = $0
      if ($1 == "model/lfm2/block_visit") baseline_rust = field("rust_median_ns_per_op")
      next
    }
    {
      if ($0 ~ /^# source_/) current_source = current_source $0
      if ($0 ~ /^# benchmark_operand:/) current_operand = $0
      if ($0 ~ /^# benchmark_config:/) current_config = $0
      if ($1 == "model/lfm2/block_visit") {
        current_rust = field("rust_median_ns_per_op")
        current_ratio = field("paired_ratio_median")
      }
    }
    END {
      if (baseline_source != current_source || baseline_operand != current_operand ||
          baseline_config != current_config || baseline_rust <= 0 || current_rust <= 0 ||
          current_ratio > max_ratio || current_rust > baseline_rust * max_ratio) exit 1
    }
  ' "$BASELINE" "$CURRENT"
fi
cat "$CURRENT"
