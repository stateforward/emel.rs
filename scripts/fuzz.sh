#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Some macOS hosts have Command Line Tools selected even though the complete
# Xcode toolchain is installed. libfuzzer-sys needs the matching compiler and
# SDK; bind both explicitly when the caller has not selected another C++
# toolchain.
if [[ "$(uname -s)" == "Darwin" && -z "${CXX:-}" ]]; then
  XCODE_CXX=/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/clang++
  if [[ -x "$XCODE_CXX" ]]; then
    XCODE_SDK="$(DEVELOPER_DIR=/Applications/Xcode.app xcrun --sdk macosx --show-sdk-path)"
    export CXX="$XCODE_CXX"
    export SDKROOT="$XCODE_SDK"
    export CXXFLAGS="${CXXFLAGS:+$CXXFLAGS }-isysroot $XCODE_SDK"
  fi
fi

FUZZ_DIR="$ROOT_DIR/fuzz"
BUILD_DIR="${EMEL_FUZZ_BUILD_DIR:-$ROOT_DIR/target/fuzz}"
GENERATED_FIXTURES="$BUILD_DIR/generated-fixtures"
CORPUS_DIR="$BUILD_DIR/gguf-corpus"
VOCAB_CORPUS_DIR="$BUILD_DIR/model-vocabulary-corpus"
DURATION_SECONDS="${EMEL_FUZZ_SECONDS:-10}"
MAX_LEN="${EMEL_FUZZ_MAX_LEN:-65536}"
MODE="run"
TARGETS=(gguf_loader gguf_load gguf_lifecycle gguf_metadata emel_io_read emel_io_mmap emel_io_staged_read emel_io_loader emel_model_tensor emel_model_catalog emel_model_generation emel_model_sortformer emel_token_profile emel_model_vocabulary emel_tensor_dtype)

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

Requires a nightly Rust toolchain and cargo-fuzz 0.13.2 or newer.
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

seed_corpus() {
  mkdir -p "$GENERATED_FIXTURES" "$CORPUS_DIR"
  cargo run --quiet --manifest-path "$ROOT_DIR/Cargo.toml" \
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

needs_gguf_corpus=false
for target in "${TARGETS[@]}"; do
  if [[ "$target" != "emel_io_read" && "$target" != "emel_io_mmap" && "$target" != "emel_io_staged_read" && "$target" != "emel_io_loader" && "$target" != "emel_model_tensor" && "$target" != "emel_model_catalog" && "$target" != "emel_model_generation" && "$target" != "emel_model_sortformer" && "$target" != "emel_token_profile" && "$target" != "emel_model_vocabulary" && "$target" != "emel_tensor_dtype" ]]; then
    needs_gguf_corpus=true
  fi
done

if [[ "$MODE" != "build" ]] && $needs_gguf_corpus; then
  seed_corpus
fi
if [[ "$MODE" != "build" ]] && [[ " ${TARGETS[*]} " == *" emel_model_vocabulary "* ]]; then
  seed_vocabulary_corpus
fi
if [[ "$MODE" == "seed" ]]; then
  exit 0
fi

if ! command -v cargo-fuzz >/dev/null 2>&1; then
  echo "error: cargo-fuzz is required; install it with cargo +nightly install cargo-fuzz --version 0.13.2 --locked" >&2
  exit 2
fi
if ! rustup run nightly rustc --version >/dev/null 2>&1; then
  echo "error: the nightly Rust toolchain is required; install it with rustup toolchain install nightly --profile minimal" >&2
  exit 2
fi

for target in "${TARGETS[@]}"; do
  if [[ "$MODE" == "build" ]]; then
    (cd "$FUZZ_DIR" && cargo fuzz build "$target")
  else
    echo "Fuzzing $target for ${DURATION_SECONDS}s with max_len=$MAX_LEN"
    if [[ "$target" == "emel_model_vocabulary" ]]; then
      (cd "$FUZZ_DIR" && cargo fuzz run "$target" "$VOCAB_CORPUS_DIR" -- \
        -seed=1 -max_total_time="$DURATION_SECONDS" -max_len="$MAX_LEN")
    elif [[ "$target" == "emel_io_read" || "$target" == "emel_io_mmap" || "$target" == "emel_io_staged_read" || "$target" == "emel_io_loader" || "$target" == "emel_model_tensor" || "$target" == "emel_model_catalog" || "$target" == "emel_model_generation" || "$target" == "emel_model_sortformer" || "$target" == "emel_token_profile" || "$target" == "emel_tensor_dtype" ]]; then
      (cd "$FUZZ_DIR" && cargo fuzz run "$target" -- \
        -seed=1 -max_total_time="$DURATION_SECONDS" -max_len="$MAX_LEN")
    else
      (cd "$FUZZ_DIR" && cargo fuzz run "$target" "$CORPUS_DIR" -- \
        -seed=1 -max_total_time="$DURATION_SECONDS" -max_len="$MAX_LEN")
    fi
  fi
done
