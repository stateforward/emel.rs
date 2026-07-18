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
MODEL_VOCABULARY_REPORT="${EMEL_MODEL_VOCABULARY_COVERAGE_REPORT:-$ROOT_DIR/target/coverage/emel-model-vocabulary.json}"
TOKEN_PROFILE_REPORT="${EMEL_TOKEN_PROFILE_COVERAGE_REPORT:-$ROOT_DIR/target/coverage/emel-token-profile.json}"
GGUF_COVERAGE_TARGET_DIR="${EMEL_GGUF_COVERAGE_TARGET_DIR:-$ROOT_DIR/target/llvm-cov-target/emel-gguf}"
IO_COVERAGE_TARGET_DIR="${EMEL_IO_COVERAGE_TARGET_DIR:-$ROOT_DIR/target/llvm-cov-target/emel-io}"
IO_MMAP_COVERAGE_TARGET_DIR="${EMEL_IO_MMAP_COVERAGE_TARGET_DIR:-$ROOT_DIR/target/llvm-cov-target/emel-io-mmap}"
IO_STAGED_READ_COVERAGE_TARGET_DIR="${EMEL_IO_STAGED_READ_COVERAGE_TARGET_DIR:-$ROOT_DIR/target/llvm-cov-target/emel-io-staged-read}"
IO_LOADER_COVERAGE_TARGET_DIR="${EMEL_IO_LOADER_COVERAGE_TARGET_DIR:-$ROOT_DIR/target/llvm-cov-target/emel-io-loader}"
MODEL_TENSOR_COVERAGE_TARGET_DIR="${EMEL_MODEL_TENSOR_COVERAGE_TARGET_DIR:-$ROOT_DIR/target/llvm-cov-target/emel-model-tensor}"
MODEL_VOCABULARY_COVERAGE_TARGET_DIR="${EMEL_MODEL_VOCABULARY_COVERAGE_TARGET_DIR:-$ROOT_DIR/target/llvm-cov-target/emel-model-vocabulary}"
TOKEN_PROFILE_COVERAGE_TARGET_DIR="${EMEL_TOKEN_PROFILE_COVERAGE_TARGET_DIR:-$ROOT_DIR/target/llvm-cov-target/emel-token-profile}"

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
  "$(dirname "$MODEL_TENSOR_REPORT")" \
  "$(dirname "$MODEL_VOCABULARY_REPORT")" \
  "$(dirname "$TOKEN_PROFILE_REPORT")"
echo "Coverage scope: emel-gguf, maintained emel-io, maintained emel-model data/tensor/vocabulary, and emel-token profile sources"
echo "Generated sm.rs files are excluded; maintained emel-io, model, and token profile sources are enforced"
echo "Coverage thresholds: lines >= ${LINE_COVERAGE_MIN}%, branches >= ${BRANCH_COVERAGE_MIN}%"

run_coverage() {
  local target_dir="$1"
  local report="$2"
  local ignore_filename_regex="$3"
  shift 3

  # Keep execution and reporting in one scoped cargo-llvm-cov command. A separate
  # `llvm-cov report` invocation loses Cargo's package selection and can fold
  # duplicate cfg(test) instantiations from the target directory into the line
  # denominator even though the executed campaign was package-scoped.
  CARGO_LLVM_COV_TARGET_DIR="$target_dir" cargo +"$COVERAGE_TOOLCHAIN" llvm-cov \
    --manifest-path "$ROOT_DIR/Cargo.toml" \
    --all-features \
    --locked \
    --branch \
    --ignore-filename-regex "$ignore_filename_regex" \
    --summary-only \
    --json \
    --output-path "$report" \
    "$@"
}

echo "Running emel-gguf coverage"
run_coverage "$GGUF_COVERAGE_TARGET_DIR" "$GGUF_REPORT" '(^|/)sm\.rs$' \
  --package emel-gguf

echo "Running emel-io coverage"
run_coverage "$IO_COVERAGE_TARGET_DIR" "$IO_REPORT" \
  'crates/emel-io/src/(loader|mmap|staged_read)/' \
  --package emel-io

echo "Running maintained emel-io mmap coverage"
run_coverage "$IO_MMAP_COVERAGE_TARGET_DIR" "$IO_MMAP_REPORT" \
  'crates/emel-io/(src/(lib\.rs|loader/|read/|staged_read/|mmap/tests\.rs)|tests/|examples/)' \
  --package emel-io

echo "Running maintained emel-io staged-read coverage"
run_coverage "$IO_STAGED_READ_COVERAGE_TARGET_DIR" "$IO_STAGED_READ_REPORT" \
  'crates/emel-io/(src/(lib\.rs|loader/|read/|mmap/|staged_read/tests\.rs)|tests/|examples/)' \
  --package emel-io

echo "Running maintained emel-io loader coverage"
run_coverage "$IO_LOADER_COVERAGE_TARGET_DIR" "$IO_LOADER_REPORT" \
  'crates/emel-io/(src/(lib\.rs|mmap/|read/|staged_read/|loader/tests\.rs)|tests/|examples/)' \
  --package emel-io

echo "Running maintained emel-model data/tensor coverage with caller-owned mapped capability proof"
run_coverage "$MODEL_TENSOR_COVERAGE_TARGET_DIR" "$MODEL_TENSOR_REPORT" \
  '(^|/)(tools/|crates/emel-io/|crates/emel-gguf/|crates/emel-model/(src/(lib\.rs|architecture/|gemma4/|generation/|lfm2/|llama/|loader/|moshi/|omniembed/|port_inventory_tests\.rs|qwen3/|sortformer/|tensor/(tests\.rs|window/)|vocabulary/|whisper/)|tests/|examples/))' \
  --package emel-model --package emel-bench

echo "Running maintained emel-model vocabulary and typed hyperparameter coverage"
run_coverage "$MODEL_VOCABULARY_COVERAGE_TARGET_DIR" "$MODEL_VOCABULARY_REPORT" \
  '(^|/)(tools/|crates/emel-io/|crates/emel-gguf/|crates/emel-token/|crates/emel-model/(src/(lib\.rs|architecture/|data\.rs|gemma4/|generation/|lfm2/|llama/|loader/(mod\.rs|test_gguf\.rs)|moshi/|omniembed/|port_inventory_tests\.rs|qwen3/|sortformer/|tensor/|vocabulary/tests\.rs|whisper/)|tests/|examples/)|sm\.rs$)' \
  --package emel-model

echo "Running maintained emel-token profile coverage"
run_coverage "$TOKEN_PROFILE_COVERAGE_TARGET_DIR" "$TOKEN_PROFILE_REPORT" \
  'crates/emel-token/(src/(lib\.rs|batcher/)|tests/)' \
  --package emel-token

python3 - "$LINE_COVERAGE_MIN" "$BRANCH_COVERAGE_MIN" \
  "emel-gguf=$GGUF_REPORT" "emel-io=$IO_REPORT" \
  "emel-io-mmap=$IO_MMAP_REPORT" \
  "emel-io-staged-read=$IO_STAGED_READ_REPORT" \
  "emel-io-loader=$IO_LOADER_REPORT" \
  "emel-model-tensor=$MODEL_TENSOR_REPORT" \
  "emel-model-vocabulary=$MODEL_VOCABULARY_REPORT" \
  "emel-token-profile=$TOKEN_PROFILE_REPORT" <<'PY'
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
