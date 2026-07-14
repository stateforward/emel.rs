#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FUZZ_DIR="$ROOT_DIR/fuzz"
BUILD_DIR="${EMEL_FUZZ_BUILD_DIR:-$ROOT_DIR/target/fuzz}"
GENERATED_FIXTURES="$BUILD_DIR/generated-fixtures"
CORPUS_DIR="$BUILD_DIR/gguf-corpus"
DURATION_SECONDS="${EMEL_FUZZ_SECONDS:-10}"
MAX_LEN="${EMEL_FUZZ_MAX_LEN:-65536}"
MODE="run"
TARGETS=(gguf_loader gguf_load gguf_lifecycle emel_io_read)

usage() {
  cat <<'USAGE'
usage: scripts/fuzz.sh [OPTIONS]

Seed and run the isolated cargo-fuzz targets for emel-gguf and emel-io.

  --target NAME   run one of: gguf_loader, gguf_load, gguf_lifecycle, emel_io_read
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
  for target in gguf_loader gguf_load gguf_lifecycle emel_io_read; do
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

needs_gguf_corpus=false
for target in "${TARGETS[@]}"; do
  if [[ "$target" != "emel_io_read" ]]; then
    needs_gguf_corpus=true
  fi
done

if [[ "$MODE" != "build" ]] && $needs_gguf_corpus; then
  seed_corpus
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
    if [[ "$target" == "emel_io_read" ]]; then
      (cd "$FUZZ_DIR" && cargo fuzz run "$target" -- \
        -seed=1 -max_total_time="$DURATION_SECONDS" -max_len="$MAX_LEN")
    else
      (cd "$FUZZ_DIR" && cargo fuzz run "$target" "$CORPUS_DIR" -- \
        -seed=1 -max_total_time="$DURATION_SECONDS" -max_len="$MAX_LEN")
    fi
  fi
done
