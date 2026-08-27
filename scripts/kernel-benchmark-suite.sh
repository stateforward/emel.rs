#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODE=all

for argument in "$@"; do
  case "$argument" in
    --snapshot-only) MODE=verify ;;
    --live-only) MODE=live ;;
    --update-only) MODE=update ;;
    --update) MODE=all ;;
    --no-update) MODE=verify ;;
    --help|-h)
      cat <<'USAGE'
usage: scripts/kernel-benchmark-suite.sh [--snapshot-only|--live-only|--update-only]
                                         [--update|--no-update]

Every benchmark is a split pinned-reference/public-Rust run. Snapshot-only
still executes both timing lanes because timing snapshots validate the fixed
configuration and output contract, while ignoring timing-value drift.

The shared EMEL_KERNEL_BENCH_MAX_RUST_VS_CPP_RATIO limit defaults to 2.0 and
is enforced by every registered benchmark before its snapshot is written.
USAGE
      exit 0
      ;;
    *) printf 'error: unknown argument: %s\n' "$argument" >&2; exit 2 ;;
  esac
done

BENCHMARKS=(
  kernel-elementwise-bench.sh
  kernel-unary-bench.sh
  kernel-f16-matmul-bench.sh
  kernel-flash-attn-bench.sh
  kernel-get-rows-bench.sh
  kernel-im2col-bench.sh
  kernel-matmul-bench.sh
  kernel-rope-bench.sh
)

for relative in "${BENCHMARKS[@]}"; do
  [[ -x "$ROOT_DIR/scripts/$relative" ]] || {
    printf 'error: registered kernel benchmark is missing: scripts/%s\n' "$relative" >&2
    exit 1
  }
done

run_benchmark() {
  local script="$1"
  case "$MODE" in
    all) "$script" --snapshot --update ;;
    verify) "$script" --snapshot --no-update ;;
    live) "$script" --no-snapshot --no-update ;;
    update) "$script" --no-snapshot --update ;;
  esac
}

for relative in "${BENCHMARKS[@]}"; do
  run_benchmark "$ROOT_DIR/scripts/$relative"
done

case "$MODE" in
  all) "$ROOT_DIR/scripts/bench.sh" --suite=kernel-capability --snapshot --update ;;
  verify) "$ROOT_DIR/scripts/bench.sh" --suite=kernel-capability --snapshot ;;
  live) "$ROOT_DIR/scripts/bench.sh" --suite=kernel-capability ;;
  update) "$ROOT_DIR/scripts/bench.sh" --suite=kernel-capability --snapshot --update ;;
esac

printf 'Kernel benchmark suite passed: %s operator benchmarks plus kernel capability\n' \
  "${#BENCHMARKS[@]}"
