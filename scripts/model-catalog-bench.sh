#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EMEL_CPP_SOURCE="${EMEL_CPP_SOURCE_DIR:-$ROOT_DIR/../emel.cpp}"
BUILD_DIR="${EMEL_MODEL_CATALOG_BENCH_BUILD_DIR:-$ROOT_DIR/target/bench/model-catalog-reference}"
BASELINE_DIR="${EMEL_BENCH_BASELINE_DIR:-$ROOT_DIR/snapshots/bench}"
SOURCE_COMMIT=843a117386ef17dc5a50549bbfc821074c2141d6
OPERAND_SHA256=4c0c663efcac31d2c927c729d2f88aeb3fc52dc07a55d8f47c6fce9a42fda4b7
iterations=200000
runs=7
warmup=20000
snapshot=false
update=false
max_regression_ratio="${EMEL_BENCH_MAX_REGRESSION_RATIO:-2.0}"

for argument in "$@"; do
  case "$argument" in
    --iterations=*) iterations="${argument#*=}" ;;
    --runs=*) runs="${argument#*=}" ;;
    --warmup-iterations=*) warmup="${argument#*=}" ;;
    --snapshot|--compare) snapshot=true ;;
    --update) update=true ;;
    --help|-h)
      echo "usage: scripts/model-catalog-bench.sh [--snapshot] [--update] [--iterations=N --runs=N --warmup-iterations=N]"
      exit 0
      ;;
    *) echo "error: unknown model catalog benchmark argument: $argument" >&2; exit 2 ;;
  esac
done
if $update && ! $snapshot; then
  echo "error: --update requires --snapshot" >&2
  exit 2
fi
for value in "$iterations" "$runs" "$warmup"; do
  [[ "$value" =~ ^[0-9]+$ ]] || { echo "error: benchmark counts must be integers" >&2; exit 2; }
done
(( iterations > 0 && runs > 0 )) || {
  echo "error: benchmark iterations and runs must be positive" >&2
  exit 2
}
if ! awk -v value="$max_regression_ratio" '
  BEGIN {
    valid = value ~ /^([0-9]+([.][0-9]*)?|[.][0-9]+)$/ && value + 0 > 0
    exit !valid
  }
'; then
  echo "error: EMEL_BENCH_MAX_REGRESSION_RATIO must be a positive number" >&2
  exit 2
fi

mkdir -p "$BUILD_DIR"
EMEL_CPP_SOURCE_DIR="$EMEL_CPP_SOURCE" \
  EMEL_MODEL_CATALOG_PARITY_BUILD_DIR="$BUILD_DIR" \
  "$ROOT_DIR/scripts/model-catalog-parity.sh" --live >/dev/null
cargo build --quiet --locked --release --manifest-path "$ROOT_DIR/Cargo.toml" \
  -p emel-model --example catalog_observer
"$BUILD_DIR/cpp-observer" --benchmark "$iterations" "$runs" "$warmup" \
  >"$BUILD_DIR/cpp.out"
"$ROOT_DIR/target/release/examples/catalog_observer" --benchmark \
  "$iterations" "$runs" "$warmup" >"$BUILD_DIR/rust.out"

host_arch="$(rustc --print cfg | awk -F'"' '/^target_arch=/{print $2; exit}')"
current="$BUILD_DIR/current.txt"
awk -v arch="$host_arch" -v commit="$SOURCE_COMMIT" \
  -v operand="$OPERAND_SHA256" -v warmup="$warmup" '
  function field(name,    i, parts) {
    for (i = 1; i <= NF; ++i) {
      split($i, parts, "=")
      if (parts[1] == name) return parts[2]
    }
    return ""
  }
  FNR == NR {
    cpp = field("cpp_ns_per_lookup")
    cpp_outcome = field("outcome")
    cpp_checksum = field("checksum")
    cpp_iter = field("iter")
    cpp_runs = field("runs")
    next
  }
  {
    rust = field("rust_ns_per_lookup")
    setup = field("setup_ns")
    rust_outcome = field("outcome")
    rust_checksum = field("checksum")
    rust_iter = field("iter")
    rust_runs = field("runs")
  }
  END {
    # This live cross-implementation ceiling is intentionally fixed; the
    # configurable regression ratio applies to the checked-in Rust baseline.
    ratio = rust / cpp
    if (cpp <= 0 || rust <= 0 || setup <= 0 || ratio > 2.0 ||
        cpp_outcome != "found" || rust_outcome != cpp_outcome ||
        cpp_checksum != rust_checksum || cpp_iter != rust_iter ||
        cpp_runs != rust_runs) exit 1
    print "# bench_host_arch: " arch
    print "# source_repository: stateforward/emel.cpp"
    print "# source_commit: " commit
    print "# operand_sha256: " operand
    print "# benchmark_config: records=256 target=tensor.255 iterations=" cpp_iter " runs=" cpp_runs " sample_policy=median warmup_iterations=" warmup
    print "# benchmark_timing: setup is separate; steady Rust timing includes public Catalog FindTensor dispatch; C++ uses public generation::bind_tensor_view"
    printf "model/catalog/find_last rust_ns_per_lookup=%.3f cpp_ns_per_lookup=%.3f rust_vs_cpp_ratio=%.6f setup_ns=%s checksum=%s iter=%s runs=%s\n", rust, cpp, ratio, setup, rust_checksum, rust_iter, rust_runs
  }
' "$BUILD_DIR/cpp.out" "$BUILD_DIR/rust.out" >"$current"

baseline="$BASELINE_DIR/model-catalog-$host_arch.txt"
if $snapshot; then
  if $update; then
    mkdir -p "$BASELINE_DIR"
    install -m 0644 "$current" "$baseline"
  fi
  [[ -f "$baseline" ]] || { echo "error: missing model catalog benchmark baseline: $baseline" >&2; exit 1; }
  awk -v max_ratio="$max_regression_ratio" '
    function field(name,    i, parts) {
      for (i = 1; i <= NF; ++i) {
        split($i, parts, "=")
        if (parts[1] == name) return parts[2]
      }
      return ""
    }
    FNR == NR {
      if ($0 ~ /^# bench_host_arch:/) baseline_arch = $0
      if ($0 ~ /^# source_commit:/) baseline_source = $0
      if ($0 ~ /^# operand_sha256:/) baseline_operand = $0
      if ($0 ~ /^# benchmark_config:/) baseline_config = $0
      if ($1 == "model/catalog/find_last") {
        baseline = field("rust_ns_per_lookup")
        baseline_checksum = field("checksum")
        baseline_iter = field("iter")
        baseline_runs = field("runs")
      }
      next
    }
    {
      if ($0 ~ /^# bench_host_arch:/) current_arch = $0
      if ($0 ~ /^# source_commit:/) current_source = $0
      if ($0 ~ /^# operand_sha256:/) current_operand = $0
      if ($0 ~ /^# benchmark_config:/) current_config = $0
      if ($1 == "model/catalog/find_last") {
        current = field("rust_ns_per_lookup")
        current_checksum = field("checksum")
        current_iter = field("iter")
        current_runs = field("runs")
      }
    }
    END {
      comparable = baseline_arch != "" && baseline_arch == current_arch &&
        baseline_source != "" && baseline_source == current_source &&
        baseline_operand != "" && baseline_operand == current_operand &&
        baseline_config != "" && baseline_config == current_config &&
        baseline_checksum != "" && baseline_checksum == current_checksum &&
        baseline_iter != "" && baseline_iter == current_iter &&
        baseline_runs != "" && baseline_runs == current_runs
      if (!comparable) {
        print "error: model catalog benchmark baseline is not comparable" > "/dev/stderr"
        exit 1
      }
      if (baseline <= 0 || current <= 0 || current > baseline * max_ratio) exit 1
    }
  ' "$baseline" "$current"
fi
cat "$current"
