#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COVERAGE_TOOLCHAIN="${EMEL_COVERAGE_TOOLCHAIN:-nightly-2026-07-12}"
CARGO_LLVM_COV_VERSION="${CARGO_LLVM_COV_VERSION:-0.8.6}"
LINE_COVERAGE_MIN="${LINE_COVERAGE_MIN:-90}"
BRANCH_COVERAGE_MIN="${BRANCH_COVERAGE_MIN:-50}"
GGUF_REPORT="${EMEL_COVERAGE_REPORT:-$ROOT_DIR/target/coverage/emel-gguf.json}"
IO_REPORT="${EMEL_IO_COVERAGE_REPORT:-$ROOT_DIR/target/coverage/emel-io.json}"
IO_MMAP_REPORT="${EMEL_IO_MMAP_COVERAGE_REPORT:-$ROOT_DIR/target/coverage/emel-io-mmap.json}"
IO_STAGED_READ_REPORT="${EMEL_IO_STAGED_READ_COVERAGE_REPORT:-$ROOT_DIR/target/coverage/emel-io-staged-read.json}"
IO_LOADER_REPORT="${EMEL_IO_LOADER_COVERAGE_REPORT:-$ROOT_DIR/target/coverage/emel-io-loader.json}"
MODEL_TENSOR_REPORT="${EMEL_MODEL_TENSOR_COVERAGE_REPORT:-$ROOT_DIR/target/coverage/emel-model-tensor.json}"
GGUF_COVERAGE_TARGET_DIR="${EMEL_GGUF_COVERAGE_TARGET_DIR:-$ROOT_DIR/target/llvm-cov-target/emel-gguf}"
IO_COVERAGE_TARGET_DIR="${EMEL_IO_COVERAGE_TARGET_DIR:-$ROOT_DIR/target/llvm-cov-target/emel-io}"
IO_MMAP_COVERAGE_TARGET_DIR="${EMEL_IO_MMAP_COVERAGE_TARGET_DIR:-$ROOT_DIR/target/llvm-cov-target/emel-io-mmap}"
IO_STAGED_READ_COVERAGE_TARGET_DIR="${EMEL_IO_STAGED_READ_COVERAGE_TARGET_DIR:-$ROOT_DIR/target/llvm-cov-target/emel-io-staged-read}"
IO_LOADER_COVERAGE_TARGET_DIR="${EMEL_IO_LOADER_COVERAGE_TARGET_DIR:-$ROOT_DIR/target/llvm-cov-target/emel-io-loader}"
MODEL_TENSOR_COVERAGE_TARGET_DIR="${EMEL_MODEL_TENSOR_COVERAGE_TARGET_DIR:-$ROOT_DIR/target/llvm-cov-target/emel-model-tensor}"

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

mkdir -p \
  "$(dirname "$GGUF_REPORT")" \
  "$(dirname "$IO_REPORT")" \
  "$(dirname "$IO_MMAP_REPORT")" \
  "$(dirname "$IO_STAGED_READ_REPORT")" \
  "$(dirname "$IO_LOADER_REPORT")" \
  "$(dirname "$MODEL_TENSOR_REPORT")"
echo "Coverage scope: emel-gguf, maintained emel-io, and emel-model tensor-core sources"
echo "Generated GGUF sm.rs is excluded; maintained emel-io and tensor-core SML sources are fully enforced"
echo "Coverage thresholds: lines >= ${LINE_COVERAGE_MIN}%, branches >= ${BRANCH_COVERAGE_MIN}%"

echo "Running emel-gguf coverage"
CARGO_LLVM_COV_TARGET_DIR="$GGUF_COVERAGE_TARGET_DIR" cargo +"$COVERAGE_TOOLCHAIN" llvm-cov \
  --manifest-path "$ROOT_DIR/Cargo.toml" \
  --package emel-gguf \
  --all-features \
  --locked \
  --branch \
  --ignore-filename-regex '(^|/)sm\.rs$' \
  --summary-only \
  --json \
  --output-path "$GGUF_REPORT"

echo "Running emel-io coverage"
CARGO_LLVM_COV_TARGET_DIR="$IO_COVERAGE_TARGET_DIR" cargo +"$COVERAGE_TOOLCHAIN" llvm-cov \
  --manifest-path "$ROOT_DIR/Cargo.toml" \
  --package emel-io \
  --all-features \
  --locked \
  --branch \
  --ignore-filename-regex 'crates/emel-io/src/(loader|mmap|staged_read)/' \
  --summary-only \
  --json \
  --output-path "$IO_REPORT"

echo "Running maintained emel-io mmap coverage"
CARGO_LLVM_COV_TARGET_DIR="$IO_MMAP_COVERAGE_TARGET_DIR" cargo +"$COVERAGE_TOOLCHAIN" llvm-cov \
  --manifest-path "$ROOT_DIR/Cargo.toml" \
  --package emel-io \
  --all-features \
  --locked \
  --branch \
  --ignore-filename-regex 'crates/emel-io/(src/(lib\.rs|loader/|read/|staged_read/|mmap/tests\.rs)|tests/|examples/)' \
  --summary-only \
  --json \
  --output-path "$IO_MMAP_REPORT"

echo "Running maintained emel-io staged-read coverage"
CARGO_LLVM_COV_TARGET_DIR="$IO_STAGED_READ_COVERAGE_TARGET_DIR" cargo +"$COVERAGE_TOOLCHAIN" llvm-cov \
  --manifest-path "$ROOT_DIR/Cargo.toml" \
  --package emel-io \
  --all-features \
  --locked \
  --branch \
  --ignore-filename-regex 'crates/emel-io/(src/(lib\.rs|loader/|read/|mmap/|staged_read/tests\.rs)|tests/|examples/)' \
  --summary-only \
  --json \
  --output-path "$IO_STAGED_READ_REPORT"

echo "Running maintained emel-io loader coverage"
CARGO_LLVM_COV_TARGET_DIR="$IO_LOADER_COVERAGE_TARGET_DIR" cargo +"$COVERAGE_TOOLCHAIN" llvm-cov \
  --manifest-path "$ROOT_DIR/Cargo.toml" --package emel-io --all-features --locked --branch \
  --ignore-filename-regex 'crates/emel-io/(src/(lib\.rs|mmap/|read/|staged_read/|loader/tests\.rs)|tests/|examples/)' \
  --summary-only --json --output-path "$IO_LOADER_REPORT"

echo "Running maintained emel-model tensor coverage with caller-owned mapped capability proof"
CARGO_LLVM_COV_TARGET_DIR="$MODEL_TENSOR_COVERAGE_TARGET_DIR" cargo +"$COVERAGE_TOOLCHAIN" llvm-cov \
  --manifest-path "$ROOT_DIR/Cargo.toml" --package emel-model --package emel-bench \
  --all-features --locked --branch \
  --ignore-filename-regex '(^|/)(tools/|crates/emel-io/|crates/emel-gguf/|crates/emel-model/(src/(lib\.rs|architecture/|gemma4/|generation/|lfm2/|llama/|loader/|moshi/|omniembed/|port_inventory_tests\.rs|qwen3/|sortformer/|tensor/(tests\.rs|window/)|whisper/)|tests/|examples/))' \
  --summary-only --json --output-path "$MODEL_TENSOR_REPORT"

python3 - "$LINE_COVERAGE_MIN" "$BRANCH_COVERAGE_MIN" \
  "emel-gguf=$GGUF_REPORT" "emel-io=$IO_REPORT" \
  "emel-io-mmap=$IO_MMAP_REPORT" \
  "emel-io-staged-read=$IO_STAGED_READ_REPORT" \
  "emel-io-loader=$IO_LOADER_REPORT" \
  "emel-model-tensor=$MODEL_TENSOR_REPORT" <<'PY'
import json
import math
import pathlib
import sys

try:
    line_min = float(sys.argv[1])
    branch_min = float(sys.argv[2])
except ValueError as error:
    raise SystemExit(f"error: coverage thresholds must be numbers: {error}") from error

if not all(math.isfinite(value) and 0 <= value <= 100 for value in (line_min, branch_min)):
    raise SystemExit("error: coverage thresholds must be between 0 and 100")

failures = []
for report_arg in sys.argv[3:]:
    label, report_name = report_arg.split("=", 1)
    line_report_name, separator, branch_report_name = report_name.partition(",")
    line_report_path = pathlib.Path(line_report_name)
    branch_report_path = pathlib.Path(branch_report_name) if separator else line_report_path
    with line_report_path.open(encoding="utf-8") as report_file:
        line_report = json.load(report_file)
    with branch_report_path.open(encoding="utf-8") as report_file:
        branch_report = json.load(report_file)

    try:
        lines = line_report["data"][0]["totals"]["lines"]
        branches = branch_report["data"][0]["totals"]["branches"]
    except (KeyError, IndexError, TypeError) as error:
        raise SystemExit(f"error: malformed {label} llvm-cov report: {error}") from error

    line_percent = float(lines["percent"])
    branch_percent = float(branches["percent"])
    print(
        f"{label} line coverage: {lines['covered']}/{lines['count']} "
        f"({line_percent:.2f}%, required {line_min:g}%)"
    )
    print(
        f"{label} branch coverage: {branches['covered']}/{branches['count']} "
        f"({branch_percent:.2f}%, required {branch_min:g}%)"
    )

    if line_percent < line_min:
        failures.append(f"{label} line coverage {line_percent:.2f}% is below {line_min:g}%")
    if branch_percent < branch_min:
        failures.append(f"{label} branch coverage {branch_percent:.2f}% is below {branch_min:g}%")
if failures:
    raise SystemExit("coverage gate failed: " + "; ".join(failures))
PY
