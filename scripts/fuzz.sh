#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FUZZ_DIR="$ROOT_DIR/fuzz"
BUILD_DIR="${EMEL_FUZZ_BUILD_DIR:-$ROOT_DIR/target/fuzz}"
GENERATED_FIXTURES="$BUILD_DIR/generated-fixtures"
CORPUS_DIR="$BUILD_DIR/gguf-corpus"
VOCAB_CORPUS_DIR="$BUILD_DIR/model-vocabulary-corpus"
KERNEL_CORPUS_DIR="$BUILD_DIR/emel-kernel-elementwise-corpus"
FUZZ_TMPDIR="${EMEL_FUZZ_TMPDIR:-$BUILD_DIR/tmp}"
FUZZ_TARGET_DIR="${EMEL_FUZZ_TARGET_DIR:-$BUILD_DIR/target}"
FUZZ_TOOLCHAIN="${EMEL_FUZZ_TOOLCHAIN:-nightly-2026-08-25}"
DURATION_SECONDS="${EMEL_FUZZ_SECONDS:-10}"
MAX_LEN="${EMEL_FUZZ_MAX_LEN:-65536}"
MODE="run"
TARGETS=(gguf_loader gguf_load gguf_lifecycle gguf_metadata emel_io_read emel_io_mmap emel_io_staged_read emel_io_loader emel_model_tensor emel_model_catalog emel_model_generation emel_token_profile emel_model_vocabulary emel_tensor_dtype emel_kernel_elementwise)

usage() {
  cat <<'USAGE'
usage: scripts/fuzz.sh [OPTIONS]

Seed and run the isolated cargo-fuzz targets for emel-gguf, emel-io, and emel-model.

  --target NAME   run one configured fuzz target (see fuzz/Cargo.toml)
  --seconds N     libFuzzer time budget per target (default: 10)
  --max-len N     maximum generated input size (default: 65536)
  --build-only    compile selected targets without running them
  --seed-only     regenerate the shared parity-fixture corpus without fuzzing
  --help          show this help

Requires the pinned coverage-compatible nightly toolchain
(`nightly-2026-08-25` by default, override with `EMEL_FUZZ_TOOLCHAIN`) and
cargo-fuzz 0.13.2 or newer.
USAGE
}

is_target() {
  local requested="$1"
  local target
  for target in "${TARGETS[@]}"; do
    if [[ "$requested" == "$target" ]]; then
      return 0
    fi
  done
  return 1
}

target_feature() {
  case "$1" in
    gguf_loader|gguf_load|gguf_lifecycle|gguf_metadata) echo gguf_fuzz ;;
    emel_io_read|emel_io_mmap|emel_io_staged_read|emel_io_loader) echo io_fuzz ;;
    emel_model_tensor) echo model_tensor_fuzz ;;
    emel_token_profile) echo token_fuzz ;;
    emel_model_vocabulary) echo model_vocabulary_fuzz ;;
    emel_model_catalog) echo model_fuzz ;;
    emel_model_generation) echo model_generation_fuzz ;;
    emel_tensor_dtype) echo tensor_fuzz ;;
    emel_kernel_elementwise) echo kernel_fuzz ;;
    *) return 1 ;;
  esac
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --target)
      [[ $# -ge 2 ]] || { echo "error: --target requires a name" >&2; exit 2; }
      is_target "$2" || { echo "error: unknown fuzz target: $2" >&2; exit 2; }
      TARGETS=("$2")
      shift
      ;;
    --seconds)
      [[ $# -ge 2 ]] || { echo "error: --seconds requires a value" >&2; exit 2; }
      DURATION_SECONDS="$2"
      shift
      ;;
    --max-len)
      [[ $# -ge 2 ]] || { echo "error: --max-len requires a value" >&2; exit 2; }
      MAX_LEN="$2"
      shift
      ;;
    --build-only) MODE="build" ;;
    --seed-only) MODE="seed" ;;
    --help|-h) usage; exit 0 ;;
    *) echo "error: unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
  shift
done

if [[ ! "$DURATION_SECONDS" =~ ^[1-9][0-9]*$ ]]; then
  echo "error: --seconds must be a positive integer" >&2
  exit 2
fi
if [[ ! "$MAX_LEN" =~ ^[1-9][0-9]*$ ]] || ((MAX_LEN > 65536)); then
  echo "error: --max-len must be an integer between 1 and 65536" >&2
  exit 2
fi

mkdir -p "$FUZZ_TMPDIR" "$FUZZ_TARGET_DIR"
export TMPDIR="$FUZZ_TMPDIR"
export CARGO_TARGET_DIR="$FUZZ_TARGET_DIR"

seed_corpus() {
  mkdir -p "$GENERATED_FIXTURES" "$CORPUS_DIR"
  cargo +"$FUZZ_TOOLCHAIN" run --quiet --locked --manifest-path "$ROOT_DIR/Cargo.toml" \
    -p emel-gguf-parity -- --write-fixtures "$GENERATED_FIXTURES"

  local fixture relative destination
  while IFS= read -r -d '' fixture; do
    relative="${fixture#"$GENERATED_FIXTURES"/}"
    destination="${relative//\//-}"
    cp "$fixture" "$CORPUS_DIR/$destination"
  done < <(find "$GENERATED_FIXTURES/valid" "$GENERATED_FIXTURES/invalid" \
    -type f -name '*.gguf' -print0)

  local seed_count corpus_count
  seed_count="$(find "$GENERATED_FIXTURES/valid" "$GENERATED_FIXTURES/invalid" \
    -type f -name '*.gguf' | wc -l | tr -d '[:space:]')"
  corpus_count="$(find "$CORPUS_DIR" -type f | wc -l | tr -d '[:space:]')"
  echo "Prepared GGUF fuzz corpus with $seed_count parity seeds and $corpus_count total inputs"
}

seed_vocabulary_corpus() {
  mkdir -p "$VOCAB_CORPUS_DIR"
  local index model pre wire scenario octal_model octal_pre octal_wire octal_scenario
  for index in $(seq 0 60); do
    model=$((index % 10))
    pre=$index
    wire=$((index % 13))
    scenario=$((index % 4))
    printf -v octal_model '%03o' "$model"
    printf -v octal_pre '%03o' "$pre"
    printf -v octal_wire '%03o' "$wire"
    printf -v octal_scenario '%03o' "$scenario"
    printf '%b' "\\$octal_model\\$octal_pre\\$octal_wire\\$octal_scenario" \
      >"$VOCAB_CORPUS_DIR/alias-$index"
  done
  echo "Prepared model vocabulary fuzz corpus with 61 alias/wire/scenario seeds"
}

seed_kernel_corpus() {
  mkdir -p "$KERNEL_CORPUS_DIR"
  local index byte octal
  for index in 0 1 2 3 4; do
    : >"$KERNEL_CORPUS_DIR/seed-$index"
    for byte in $(seq 0 $((15 + index * 8))); do
      octal=$(( (byte * 37 + index * 53) % 256 ))
      printf -v octal '%03o' "$octal"
      printf '\\%s' "$octal" >>"$KERNEL_CORPUS_DIR/seed-$index"
    done
  done
  echo "Prepared emel-kernels elementwise fuzz corpus with 5 deterministic seeds"
}

needs_gguf_corpus=false
needs_kernel_corpus=false
for target in "${TARGETS[@]}"; do
  if [[ "$target" == "emel_kernel_elementwise" ]]; then
    needs_kernel_corpus=true
  elif [[ "$target" != "emel_io_read" && "$target" != "emel_io_mmap" && "$target" != "emel_io_staged_read" && "$target" != "emel_io_loader" && "$target" != "emel_model_tensor" && "$target" != "emel_model_catalog" && "$target" != "emel_model_generation" && "$target" != "emel_token_profile" && "$target" != "emel_model_vocabulary" && "$target" != "emel_tensor_dtype" ]]; then
    needs_gguf_corpus=true
  fi
done

if [[ "$MODE" != "build" ]] && $needs_gguf_corpus; then
  seed_corpus
fi
if [[ "$MODE" != "build" ]] && [[ " ${TARGETS[*]} " == *" emel_model_vocabulary "* ]]; then
  seed_vocabulary_corpus
fi
if [[ "$MODE" != "build" ]] && $needs_kernel_corpus; then
  seed_kernel_corpus
fi
if [[ "$MODE" == "seed" ]]; then
  exit 0
fi

if ! command -v cargo-fuzz >/dev/null 2>&1; then
  echo "error: cargo-fuzz is required; install it with cargo +$FUZZ_TOOLCHAIN install cargo-fuzz --version 0.13.2 --locked" >&2
  exit 2
fi
if ! rustup run "$FUZZ_TOOLCHAIN" rustc --version >/dev/null 2>&1; then
  echo "error: the fuzz toolchain $FUZZ_TOOLCHAIN is required; install it with rustup toolchain install $FUZZ_TOOLCHAIN --profile minimal" >&2
  exit 2
fi

for target in "${TARGETS[@]}"; do
  feature="$(target_feature "$target")"
  if [[ "$MODE" == "build" ]]; then
    (cd "$FUZZ_DIR" && cargo +"$FUZZ_TOOLCHAIN" fuzz build "$target" --features "$feature")
  else
    echo "Fuzzing $target for ${DURATION_SECONDS}s with max_len=$MAX_LEN"
    if [[ "$target" == "emel_model_vocabulary" ]]; then
      (cd "$FUZZ_DIR" && cargo +"$FUZZ_TOOLCHAIN" fuzz run "$target" --features "$feature" "$VOCAB_CORPUS_DIR" -- \
        -seed=1 -max_total_time="$DURATION_SECONDS" -max_len="$MAX_LEN")
    elif [[ "$target" == "emel_kernel_elementwise" ]]; then
      (cd "$FUZZ_DIR" && cargo +"$FUZZ_TOOLCHAIN" fuzz run "$target" --features "$feature" "$KERNEL_CORPUS_DIR" -- \
        -seed=1 -max_total_time="$DURATION_SECONDS" -max_len="$MAX_LEN")
    elif [[ "$target" == "emel_io_read" || "$target" == "emel_io_mmap" || "$target" == "emel_io_staged_read" || "$target" == "emel_io_loader" || "$target" == "emel_model_tensor" || "$target" == "emel_model_catalog" || "$target" == "emel_model_generation" || "$target" == "emel_token_profile" || "$target" == "emel_tensor_dtype" ]]; then
      (cd "$FUZZ_DIR" && cargo +"$FUZZ_TOOLCHAIN" fuzz run "$target" --features "$feature" -- \
        -seed=1 -max_total_time="$DURATION_SECONDS" -max_len="$MAX_LEN")
    else
      (cd "$FUZZ_DIR" && cargo +"$FUZZ_TOOLCHAIN" fuzz run "$target" --features "$feature" "$CORPUS_DIR" -- \
        -seed=1 -max_total_time="$DURATION_SECONDS" -max_len="$MAX_LEN")
    fi
  fi
done
