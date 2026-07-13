#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COVERAGE_TOOLCHAIN="${EMEL_COVERAGE_TOOLCHAIN:-nightly-2026-07-12}"
CARGO_LLVM_COV_VERSION="${CARGO_LLVM_COV_VERSION:-0.8.6}"
LINE_COVERAGE_MIN="${LINE_COVERAGE_MIN:-90}"
BRANCH_COVERAGE_MIN="${BRANCH_COVERAGE_MIN:-50}"
REPORT="${EMEL_COVERAGE_REPORT:-$ROOT_DIR/target/coverage/emel-gguf.json}"

if [[ $# -ne 0 ]]; then
  echo "usage: scripts/coverage.sh" >&2
  echo "configure thresholds with LINE_COVERAGE_MIN and BRANCH_COVERAGE_MIN" >&2
  exit 2
fi

if ! rustup run "$COVERAGE_TOOLCHAIN" rustc --version >/dev/null 2>&1; then
  echo "error: coverage requires $COVERAGE_TOOLCHAIN" >&2
  echo "install it with: rustup toolchain install $COVERAGE_TOOLCHAIN --profile minimal --component llvm-tools-preview" >&2
  exit 2
fi

if ! rustup component list --installed --toolchain "$COVERAGE_TOOLCHAIN" \
  | grep -q '^llvm-tools-'; then
  echo "error: coverage requires llvm-tools-preview for $COVERAGE_TOOLCHAIN" >&2
  echo "install it with: rustup component add llvm-tools-preview --toolchain $COVERAGE_TOOLCHAIN" >&2
  exit 2
fi

if ! llvm_cov_version="$(cargo +"$COVERAGE_TOOLCHAIN" llvm-cov --version 2>/dev/null)"; then
  echo "error: cargo-llvm-cov $CARGO_LLVM_COV_VERSION is required" >&2
  echo "install it with: cargo +$COVERAGE_TOOLCHAIN install cargo-llvm-cov --version $CARGO_LLVM_COV_VERSION --locked" >&2
  exit 2
fi

installed_version="${llvm_cov_version##* }"
if [[ "$installed_version" != "$CARGO_LLVM_COV_VERSION" ]]; then
  echo "error: cargo-llvm-cov $CARGO_LLVM_COV_VERSION is required; found $installed_version" >&2
  exit 2
fi

mkdir -p "$(dirname "$REPORT")"
echo "Coverage scope: emel-gguf production sources (generated sm.rs excluded)"
echo "Coverage thresholds: lines >= ${LINE_COVERAGE_MIN}%, branches >= ${BRANCH_COVERAGE_MIN}%"

cargo +"$COVERAGE_TOOLCHAIN" llvm-cov \
  --manifest-path "$ROOT_DIR/Cargo.toml" \
  --package emel-gguf \
  --all-features \
  --locked \
  --branch \
  --ignore-filename-regex '(^|/)sm\.rs$' \
  --summary-only \
  --json \
  --output-path "$REPORT"

python3 - "$REPORT" "$LINE_COVERAGE_MIN" "$BRANCH_COVERAGE_MIN" <<'PY'
import json
import math
import pathlib
import sys

report_path = pathlib.Path(sys.argv[1])

try:
    line_min = float(sys.argv[2])
    branch_min = float(sys.argv[3])
except ValueError as error:
    raise SystemExit(f"error: coverage thresholds must be numbers: {error}") from error

if not all(math.isfinite(value) and 0 <= value <= 100 for value in (line_min, branch_min)):
    raise SystemExit("error: coverage thresholds must be between 0 and 100")

with report_path.open(encoding="utf-8") as report_file:
    report = json.load(report_file)

try:
    totals = report["data"][0]["totals"]
    lines = totals["lines"]
    branches = totals["branches"]
except (KeyError, IndexError, TypeError) as error:
    raise SystemExit(f"error: malformed llvm-cov report: {error}") from error

line_percent = float(lines["percent"])
branch_percent = float(branches["percent"])
print(
    f"Line coverage: {lines['covered']}/{lines['count']} "
    f"({line_percent:.2f}%, required {line_min:g}%)"
)
print(
    f"Branch coverage: {branches['covered']}/{branches['count']} "
    f"({branch_percent:.2f}%, required {branch_min:g}%)"
)

failures = []
if line_percent < line_min:
    failures.append(f"line coverage {line_percent:.2f}% is below {line_min:g}%")
if branch_percent < branch_min:
    failures.append(f"branch coverage {branch_percent:.2f}% is below {branch_min:g}%")
if failures:
    raise SystemExit("coverage gate failed: " + "; ".join(failures))
PY
