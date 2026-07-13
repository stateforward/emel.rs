#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${EMEL_BENCH_BUILD_DIR:-$ROOT_DIR/target/bench}"
SNAPSHOT_MODE=false
UPDATE=false
runner_args=(gguf)

usage() {
  cat <<'USAGE'
usage: scripts/bench.sh [--snapshot|--compare] [--update] [runner options]

  --snapshot  compare against the architecture-scoped benchmark baseline
  --compare   alias for --snapshot
  --update    replace the baseline after a successful benchmark run

Runner options: --iterations=N --runs=N --warmup-iterations=N
Set EMEL_BENCH_MAX_REGRESSION_RATIO to change the default 2.0x gate.
USAGE
}

for argument in "$@"; do
  case "$argument" in
    --snapshot|--compare) SNAPSHOT_MODE=true ;;
    --update) UPDATE=true ;;
    --iterations=*|--runs=*|--warmup-iterations=*) runner_args+=("$argument") ;;
    --suite=gguf) ;;
    --help|-h) usage; exit 0 ;;
    *) echo "error: unknown argument: $argument" >&2; usage >&2; exit 2 ;;
  esac
done

if $UPDATE && ! $SNAPSHOT_MODE; then
  echo "error: --update requires --snapshot or --compare" >&2
  exit 2
fi

cargo build --manifest-path "$ROOT_DIR/Cargo.toml" --release -p emel-bench
RUNNER="$ROOT_DIR/target/release/emel-bench"
mkdir -p "$BUILD_DIR"
CURRENT="$BUILD_DIR/gguf-current.txt"
"$RUNNER" "${runner_args[@]}" >"$CURRENT"

if ! $SNAPSHOT_MODE; then
  cat "$CURRENT"
  exit 0
fi

host_arch="$(awk -F': ' '/^# bench_host_arch: / { print $2; exit }' "$CURRENT")"
BASELINE="$ROOT_DIR/snapshots/bench/gguf-$host_arch.txt"

if $UPDATE; then
  mkdir -p "$(dirname "$BASELINE")"
  install -m 0644 "$CURRENT" "$BASELINE"
  echo "Updated benchmark snapshot: $BASELINE"
  exit 0
fi

if [[ ! -f "$BASELINE" ]]; then
  echo "error: missing benchmark baseline: $BASELINE" >&2
  echo "run scripts/bench.sh --snapshot --update" >&2
  exit 1
fi

baseline_config="$(grep '^# benchmark_config: ' "$BASELINE" || true)"
current_config="$(grep '^# benchmark_config: ' "$CURRENT" || true)"
if [[ -z "$baseline_config" || "$baseline_config" != "$current_config" ]]; then
  echo "error: benchmark configuration differs from the baseline" >&2
  echo "baseline: $baseline_config" >&2
  echo "current:  $current_config" >&2
  echo "run scripts/bench.sh --snapshot --update after an intentional configuration change" >&2
  exit 1
fi

max_ratio="${EMEL_BENCH_MAX_REGRESSION_RATIO:-2.0}"
awk -v max_ratio="$max_ratio" '
  function timing_value(    field_index, parts) {
    for (field_index = 1; field_index <= NF; ++field_index) {
      if ($field_index ~ /^ns_per_op=/) {
        split($field_index, parts, "=")
        return parts[2] + 0
      }
    }
    return 0
  }
  FNR == NR && $0 !~ /^#/ && NF > 1 {
    baseline[$1] = timing_value()
    next
  }
  FNR != NR && $0 !~ /^#/ && NF > 1 {
    current[$1] = timing_value()
  }
  END {
    failed = 0
    for (name in baseline) {
      if (!(name in current)) {
        printf "missing benchmark case: %s\n", name > "/dev/stderr"
        failed = 1
        continue
      }
      ratio = current[name] / baseline[name]
      printf "%s baseline=%.3f current=%.3f ratio=%.3fx\n", \
        name, baseline[name], current[name], ratio
      if (ratio > max_ratio) {
        printf "benchmark regression: %s exceeds %.3fx\n", name, max_ratio > "/dev/stderr"
        failed = 1
      }
    }
    for (name in current) {
      if (!(name in baseline)) {
        printf "new benchmark case missing from baseline: %s\n", name > "/dev/stderr"
        failed = 1
      }
    }
    exit failed
  }
' "$BASELINE" "$CURRENT"
