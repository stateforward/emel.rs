#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SOURCE_DIR="${EMEL_CPP_SOURCE_DIR:-$ROOT_DIR/../emel.cpp}"
BUILD_DIR="${EMEL_MODEL_SORTFORMER_BENCH_BUILD_DIR:-$ROOT_DIR/target/model-sortformer-bench}"
FIXTURE="$SOURCE_DIR/tests/models/diar_streaming_sortformer_4spk-v2.1.gguf"
ITERATIONS=1 RUNS=3 WARMUP=0 PROCESS_RUNS=3 UPDATE=false SNAPSHOT=false
MAX_RATIO="${EMEL_SORTFORMER_BENCH_MAX_REGRESSION_RATIO:-2.0}"
for argument in "$@"; do case "$argument" in
  --iterations=*) ITERATIONS="${argument#*=}";; --runs=*) RUNS="${argument#*=}";;
  --warmup-iterations=*) WARMUP="${argument#*=}";; --process-runs=*) PROCESS_RUNS="${argument#*=}";;
  --snapshot|--compare) SNAPSHOT=true;; --update) UPDATE=true; SNAPSHOT=true;; *) echo "unknown argument: $argument" >&2; exit 2;;
esac; done
[[ "$ITERATIONS" =~ ^[1-9][0-9]*$ && "$RUNS" =~ ^[1-9][0-9]*$ &&
   "$WARMUP" =~ ^[0-9]+$ && "$PROCESS_RUNS" =~ ^[1-9][0-9]*$ &&
   $((PROCESS_RUNS % 2)) -eq 1 ]] || {
  echo "error: benchmark counts must be positive integers and process runs must be odd" >&2
  exit 2
}
mkdir -p "$BUILD_DIR"
EMEL_CPP_SOURCE_DIR="$SOURCE_DIR" EMEL_MODEL_SORTFORMER_PARITY_BUILD_DIR="$BUILD_DIR/reference" "$ROOT_DIR/scripts/model-sortformer-parity.sh" --live
cargo build --quiet --locked --release --manifest-path "$ROOT_DIR/Cargo.toml" -p emel-model --example sortformer_observer
: > "$BUILD_DIR/cpp.out"; : > "$BUILD_DIR/rust.out"
process_run=0
while [[ "$process_run" -lt "$PROCESS_RUNS" ]]; do
  env -u EMEL_MODEL_LOAD_IO_STRATEGY "$BUILD_DIR/reference/cpp-observer" --benchmark "$ITERATIONS" "$RUNS" "$WARMUP" >> "$BUILD_DIR/cpp.out"
  "$ROOT_DIR/target/release/examples/sortformer_observer" --benchmark "$FIXTURE" "$ITERATIONS" "$RUNS" "$WARMUP" >> "$BUILD_DIR/rust.out"
  process_run=$((process_run + 1))
done
paste "$BUILD_DIR/cpp.out" "$BUILD_DIR/rust.out" | awk -F'[ =\t]+' '
  {cpp="";rust="";csum="";rsum="";for(i=1;i<=NF;i++){if($i=="cpp_ns_per_visit")cpp=$(i+1);if($i=="rust_ns_per_visit")rust=$(i+1);if($i=="checksum") {if(csum=="")csum=$(i+1); else rsum=$(i+1)}}
   if(cpp<=0||rust<=0||csum!=rsum)exit 1; printf "model/sortformer/contract_visit/process run=%d cpp_ns_per_op=%s rust_ns_per_op=%s paired_ratio=%.6f checksum=%s\n",NR,cpp,rust,rust/cpp,csum}' > "$BUILD_DIR/samples.txt"
middle=$((PROCESS_RUNS / 2 + 1))
ratio="$(awk -F'[ =]' '{for(i=1;i<=NF;i++)if($i=="paired_ratio")print $(i+1)}' "$BUILD_DIR/samples.txt" | sort -n | sed -n "${middle}p")"
cpp="$(awk -F'[ =]' '{for(i=1;i<=NF;i++)if($i=="cpp_ns_per_op")print $(i+1)}' "$BUILD_DIR/samples.txt" | sort -n | sed -n "${middle}p")"
rust="$(awk -F'[ =]' '{for(i=1;i<=NF;i++)if($i=="rust_ns_per_op")print $(i+1)}' "$BUILD_DIR/samples.txt" | sort -n | sed -n "${middle}p")"
checksum="$(awk -F'[ =]' '{for(i=1;i<=NF;i++)if($i=="checksum")print $(i+1)}' "$BUILD_DIR/samples.txt" | sort -u)"
[[ "$checksum" =~ ^[0-9]+$ ]] || { echo "error: inconsistent Sortformer benchmark checksums" >&2; exit 1; }
awk -v ratio="$ratio" -v maximum="$MAX_RATIO" 'BEGIN{exit !(ratio<=maximum)}'
arch="$(rustc --print cfg | awk -F'"' '/^target_arch=/{print $2;exit}')"; os="$(rustc --print cfg | awk -F'"' '/^target_os=/{print $2;exit}')"
BASELINE="$ROOT_DIR/snapshots/bench/model-sortformer-$os-$arch-default.txt"; CURRENT="$BUILD_DIR/current.txt"
{
  echo '# source_commit: 843a117386ef17dc5a50549bbfc821074c2141d6'
  echo '# fixture_sha256: 1b85d7bf641350d0d355e7494c4b7d92a1ff2fb2d886cd6dcc43f358a6266ff0'
  echo '# operand: same real GGUF bytes through each public loader, model binding, Sortformer family contract construction, and final descriptor observation'
  printf '# config: iterations=%s runs=%s warmup=%s process_runs=%s max_ratio=%s\n' "$ITERATIONS" "$RUNS" "$WARMUP" "$PROCESS_RUNS" "$MAX_RATIO"
  cat "$BUILD_DIR/samples.txt"
  printf 'model/sortformer/contract_visit rust_median_ns_per_op=%s cpp_median_ns_per_op=%s paired_ratio_median=%s checksum=%s\n' "$rust" "$cpp" "$ratio" "$checksum"
} > "$CURRENT"
if $UPDATE; then mkdir -p "$(dirname "$BASELINE")"; install -m 0644 "$CURRENT" "$BASELINE"; fi
if $SNAPSHOT; then
  [[ -f "$BASELINE" ]] || { echo "error: missing Sortformer benchmark baseline: $BASELINE" >&2; exit 1; }
  awk -v max_ratio="$MAX_RATIO" '
    function field(name,    i, parts) {
      for (i = 1; i <= NF; ++i) {
        split($i, parts, "=")
        if (parts[1] == name) return parts[2]
      }
      return ""
    }
    FNR == NR {
      if ($0 ~ /^# source_commit:/ || $0 ~ /^# fixture_sha256:/) baseline_source = baseline_source $0
      if ($0 ~ /^# operand:/) baseline_operand = $0
      if ($0 ~ /^# config:/) baseline_config = $0
      if ($1 == "model/sortformer/contract_visit") baseline_rust = field("rust_median_ns_per_op")
      if ($1 == "model/sortformer/contract_visit") baseline_checksum = field("checksum")
      next
    }
    {
      if ($0 ~ /^# source_commit:/ || $0 ~ /^# fixture_sha256:/) current_source = current_source $0
      if ($0 ~ /^# operand:/) current_operand = $0
      if ($0 ~ /^# config:/) current_config = $0
      if ($1 == "model/sortformer/contract_visit") {
        current_rust = field("rust_median_ns_per_op")
        current_ratio = field("paired_ratio_median")
        current_checksum = field("checksum")
      }
    }
    END {
      if (baseline_source != current_source || baseline_operand != current_operand ||
          baseline_config != current_config || baseline_rust <= 0 || current_rust <= 0 ||
          baseline_checksum == "" || baseline_checksum != current_checksum ||
          current_ratio > max_ratio || current_rust > baseline_rust * max_ratio) exit 1
    }
  ' "$BASELINE" "$CURRENT"
fi
cat "$CURRENT"
