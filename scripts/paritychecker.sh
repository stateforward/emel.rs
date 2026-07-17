#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${EMEL_GGUF_PARITY_BUILD_DIR:-$ROOT_DIR/target/gguf-parity}"
FIXTURE_DIR="$BUILD_DIR/fixtures"
REFERENCE_BUILD_DIR="$BUILD_DIR/llama-reference"
SNAPSHOT="${EMEL_GGUF_PARITY_SNAPSHOT:-$ROOT_DIR/snapshots/parity/gguf/manifest.txt}"
IO_READ_SNAPSHOT="${EMEL_IO_READ_PARITY_SNAPSHOT:-$ROOT_DIR/snapshots/parity/io-read/manifest.txt}"
IO_READ_BUILD_DIR="${EMEL_IO_READ_PARITY_BUILD_DIR:-$ROOT_DIR/target/io-read-parity}"
IO_MMAP_SNAPSHOT="${EMEL_IO_MMAP_PARITY_SNAPSHOT:-$ROOT_DIR/snapshots/parity/io-mmap/manifest.txt}"
IO_MMAP_BUILD_DIR="${EMEL_IO_MMAP_PARITY_BUILD_DIR:-$ROOT_DIR/target/io-mmap-parity}"
IO_STAGED_READ_SNAPSHOT="${EMEL_IO_STAGED_READ_PARITY_SNAPSHOT:-$ROOT_DIR/snapshots/parity/io-staged-read/manifest.txt}"
IO_STAGED_READ_BUILD_DIR="${EMEL_IO_STAGED_READ_PARITY_BUILD_DIR:-$ROOT_DIR/target/io-staged-read-parity}"
IO_LOADER_SNAPSHOT="${EMEL_IO_LOADER_PARITY_SNAPSHOT:-$ROOT_DIR/snapshots/parity/io-loader/manifest.txt}"
IO_LOADER_BUILD_DIR="${EMEL_IO_LOADER_PARITY_BUILD_DIR:-$ROOT_DIR/target/io-loader-parity}"
MODEL_TENSOR_SNAPSHOT="${EMEL_MODEL_TENSOR_PARITY_SNAPSHOT:-$ROOT_DIR/snapshots/parity/model-tensor/manifest.txt}"
MODEL_TENSOR_BUILD_DIR="${EMEL_MODEL_TENSOR_PARITY_BUILD_DIR:-$ROOT_DIR/target/model-tensor-parity}"
MODEL_DATA_SNAPSHOT="${EMEL_MODEL_DATA_PARITY_SNAPSHOT:-$ROOT_DIR/snapshots/parity/model-data/manifest.txt}"
MODEL_DATA_BUILD_DIR="${EMEL_MODEL_DATA_PARITY_BUILD_DIR:-$ROOT_DIR/target/model-data-parity}"
EMEL_CPP_SOURCE="${EMEL_CPP_SOURCE_DIR:-$ROOT_DIR/../emel.cpp}"
EMEL_CPP_COMMIT=843a117386ef17dc5a50549bbfc821074c2141d6
EMEL_CPP_IO_TREE=ff00a9978b00b4ea1e5268d2ecd483d4855b6eaa
EMEL_CPP_MODEL_TENSOR_TREE=06306d4ffad3455fcf5df71dc692df52514b9865
EMEL_CPP_MODEL_TREE=278b7b20545b630be33bee8bed0cb4c8db8990c3
EMEL_CPP_MODEL_DATA_HEADER_BLOB=78a25b987423d8cbef17965a8ca92596ffc0ecef
EMEL_CPP_MODEL_DATA_IMPLEMENTATION_BLOB=b33ace170b569d076844a36146c7ca86d4ffa7fc
EMEL_CPP_MODEL_DATA_HEADER_SHA256=4b604d57fef1c22c8a9ebee36779a0c9cd4e7fe811856455d5cafd645498a151
EMEL_CPP_MODEL_DATA_IMPLEMENTATION_SHA256=c8231f4feb2bc395a642bf3d66e74f5ee6ad243d706f277245b2b28c620c8816
EXPECTED_REF="$(tr -d '[:space:]' <"$ROOT_DIR/tools/llama-gguf-reference/reference_ref.txt")"
RUN_SNAPSHOT=true
RUN_LIVE=true
RUN_UPDATE=true
REFERENCE_SOURCE="${LLAMA_CPP_SOURCE_DIR:-}"
REFERENCE_CONFIGURED=false
SUITE=all
models=()

usage() {
  cat <<'USAGE'
usage: scripts/paritychecker.sh [OPTIONS] [model.gguf ...]

By default all three phases run: refresh from pinned llama.cpp, live parity,
then checked-in snapshot verification.

  --snapshot       enable checked-in snapshot verification (default)
  --live           enable direct pinned llama.cpp parity (default)
  --update         enable snapshot refresh after exact parity (default)
  --no-snapshot    disable checked-in snapshot verification
  --no-live        disable the separate live phase
  --no-update      disable snapshot refresh
  --snapshot-only  run only checked-in snapshot gates and required reference comparisons
  --live-only      run only direct pinned llama.cpp parity
  --update-only    run only snapshot refresh and its parity validation
  --suite=NAME     run all, gguf, io-read, io-mmap, io-staged-read, io-loader, model-tensor, or model-data (default: all)

Model paths are checked during the live phase. Without paths, the deterministic
fixture corpus is used.
USAGE
}

for argument in "$@"; do
  case "$argument" in
    --snapshot) RUN_SNAPSHOT=true ;;
    --live) RUN_LIVE=true ;;
    --update) RUN_UPDATE=true ;;
    --no-snapshot) RUN_SNAPSHOT=false ;;
    --no-live) RUN_LIVE=false ;;
    --no-update) RUN_UPDATE=false ;;
    --snapshot-only)
      RUN_SNAPSHOT=true
      RUN_LIVE=false
      RUN_UPDATE=false
      ;;
    --live-only)
      RUN_SNAPSHOT=false
      RUN_LIVE=true
      RUN_UPDATE=false
      ;;
    --update-only)
      RUN_SNAPSHOT=false
      RUN_LIVE=false
      RUN_UPDATE=true
      ;;
    --suite=all) SUITE=all ;;
    --suite=gguf) SUITE=gguf ;;
    --suite=io-read) SUITE=io-read ;;
    --suite=io-mmap) SUITE=io-mmap ;;
    --suite=io-staged-read) SUITE=io-staged-read ;;
    --suite=io-loader) SUITE=io-loader ;;
    --suite=model-tensor) SUITE=model-tensor ;;
    --suite=model-data) SUITE=model-data ;;
    --help|-h) usage; exit 0 ;;
    --*) echo "error: unknown argument: $argument" >&2; usage >&2; exit 2 ;;
    *) models+=("$argument") ;;
  esac
done

if ! $RUN_SNAPSHOT && ! $RUN_LIVE && ! $RUN_UPDATE; then
  echo "error: at least one parity phase must be enabled" >&2
  exit 2
fi
if [[ ${#models[@]} -gt 0 ]] && ! $RUN_LIVE; then
  echo "error: model paths require the live phase" >&2
  exit 2
fi
if [[ ${#models[@]} -gt 0 && "$SUITE" != "all" && "$SUITE" != "gguf" ]]; then
  echo "error: model paths are only valid for the GGUF suite" >&2
  exit 2
fi

RUN_GGUF=false
RUN_IO_READ=false
RUN_IO_MMAP=false
RUN_IO_STAGED_READ=false
RUN_IO_LOADER=false
RUN_MODEL_TENSOR=false
RUN_MODEL_DATA=false
case "$SUITE" in
  all)
    RUN_GGUF=true
    RUN_IO_READ=true
    RUN_IO_MMAP=true
    RUN_IO_STAGED_READ=true
    RUN_IO_LOADER=true
    RUN_MODEL_TENSOR=true
    RUN_MODEL_DATA=true
    ;;
  gguf) RUN_GGUF=true ;;
  io-read) RUN_IO_READ=true ;;
  io-mmap) RUN_IO_MMAP=true ;;
  io-staged-read) RUN_IO_STAGED_READ=true ;;
  io-loader) RUN_IO_LOADER=true ;;
  model-tensor) RUN_MODEL_TENSOR=true ;;
  model-data) RUN_MODEL_DATA=true ;;
esac

fixture_models=()
if $RUN_GGUF; then
  cargo build --manifest-path "$ROOT_DIR/Cargo.toml" -p emel-gguf-parity
  RUST_RUNNER="$ROOT_DIR/target/debug/emel-gguf-parity"
  "$RUST_RUNNER" --write-fixtures "$FIXTURE_DIR"
  while IFS= read -r path; do
    fixture_models+=("$path")
  done < <(find "$FIXTURE_DIR/valid" "$FIXTURE_DIR/invalid" -type f -name '*.gguf' -print | sort)

  if [[ ${#fixture_models[@]} -eq 0 ]]; then
    echo "error: no GGUF parity fixtures found" >&2
    exit 2
  fi
fi

sha256_file() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{print $1}'
  else
    shasum -a 256 "$1" | awk '{print $1}'
  fi
}

configure_reference() {
  if $REFERENCE_CONFIGURED; then
    return
  fi

  if [[ -z "$REFERENCE_SOURCE" ]]; then
    local sibling_source="$ROOT_DIR/../emel.cpp/build/paritychecker_zig/_deps/reference_impl-src"
    local sibling_ref
    sibling_ref="$(git -C "$sibling_source" rev-parse HEAD 2>/dev/null || true)"
    if [[ "$sibling_ref" == "$EXPECTED_REF" ]]; then
      REFERENCE_SOURCE="$sibling_source"
    fi
  fi

  local cmake_args=(
    -S "$ROOT_DIR/tools/llama-gguf-reference"
    -B "$REFERENCE_BUILD_DIR"
    -DCMAKE_BUILD_TYPE=Release
  )
  if command -v ninja >/dev/null 2>&1; then
    cmake_args+=(-G Ninja)
  fi
  if [[ -n "$REFERENCE_SOURCE" ]]; then
    cmake_args+=("-DLLAMA_CPP_SOURCE_DIR=$REFERENCE_SOURCE")
  else
    cmake_args+=(-ULLAMA_CPP_SOURCE_DIR)
  fi
  cmake "${cmake_args[@]}"
  cmake --build "$REFERENCE_BUILD_DIR" --parallel --target llama-gguf-reference
  REFERENCE_RUNNER="$REFERENCE_BUILD_DIR/llama-gguf-reference"
  REFERENCE_CONFIGURED=true
}

run_reference() {
  local model="$1"
  local output="$2"
  local model_directory model_absolute
  model_directory="$(cd "$(dirname "$model")" && pwd -P)"
  model_absolute="$model_directory/$(basename "$model")"

  if ! { "$REFERENCE_RUNNER" "$model" >"$output"; } 2>"$BUILD_DIR/reference.err"; then
    if [[ "$model_absolute" == "$FIXTURE_DIR/invalid/"* ]]; then
      printf 'gguf-parity/v1\nstatus=error\n' >"$output"
      return
    fi
    echo "llama.cpp reference runner failed: $model" >&2
    cat "$BUILD_DIR/reference.err" >&2
    exit 1
  fi
}

write_manifest() {
  local runner="$1"
  local destination="$2"
  local output="$BUILD_DIR/$runner.out"
  {
    echo "gguf-parity-snapshot/v1"
    echo "reference_ref=$EXPECTED_REF"
    echo "canonical_format=gguf-parity/v1"
    echo "fixture_count=${#fixture_models[@]}"
    local model relative digest bytes
    for model in "${fixture_models[@]}"; do
      relative="${model#"$FIXTURE_DIR/"}"
      if [[ "$runner" == "rust" ]]; then
        "$RUST_RUNNER" "$model" >"$output"
      else
        run_reference "$model" "$output"
      fi
      digest="$(sha256_file "$output")"
      bytes="$(wc -c <"$output" | tr -d '[:space:]')"
      echo "fixture=$relative sha256=$digest bytes=$bytes"
    done
  } >"$destination"
}

check_snapshot() {
  local actual="$BUILD_DIR/manifest.actual.txt"
  if [[ ! -f "$SNAPSHOT" ]]; then
    echo "error: missing parity snapshot: $SNAPSHOT" >&2
    echo "run scripts/paritychecker.sh --update" >&2
    exit 1
  fi
  write_manifest rust "$actual"
  diff -u "$SNAPSHOT" "$actual"
  echo "GGUF parity snapshot passed (${#fixture_models[@]} fixtures, llama.cpp $EXPECTED_REF)"
}

live_parity() {
  configure_reference
  local selected=()
  if [[ ${#models[@]} -eq 0 ]]; then
    selected=("${fixture_models[@]}")
  else
    selected=("${models[@]}")
  fi
  local model rust_output="$BUILD_DIR/rust.out" reference_output="$BUILD_DIR/reference.out"
  for model in "${selected[@]}"; do
    "$RUST_RUNNER" "$model" >"$rust_output"
    run_reference "$model" "$reference_output"
    if ! diff -u "$reference_output" "$rust_output"; then
      echo "GGUF live parity failed: $model" >&2
      exit 1
    fi
    echo "GGUF live parity passed: $model"
  done
}

update_snapshot() {
  configure_reference
  local candidate="$BUILD_DIR/manifest.reference.txt"
  local rust_actual="$BUILD_DIR/manifest.rust.txt"
  write_manifest reference "$candidate"
  write_manifest rust "$rust_actual"
  if ! diff -u "$candidate" "$rust_actual"; then
    echo "error: Rust does not match the refreshed llama.cpp snapshot" >&2
    exit 1
  fi
  mkdir -p "$(dirname "$SNAPSHOT")"
  install -m 0644 "$candidate" "$SNAPSHOT"
  echo "Updated GGUF parity snapshot from llama.cpp $EXPECTED_REF"
}

run_io_read_parity() {
  local source_commit source_tree rust_output reference_output materialized_source
  source_commit="$(git -C "$EMEL_CPP_SOURCE" rev-parse HEAD)"
  source_tree="$(git -C "$EMEL_CPP_SOURCE" rev-parse HEAD:src/emel/io)"
  if [[ "$source_commit" != "$EMEL_CPP_COMMIT" || "$source_tree" != "$EMEL_CPP_IO_TREE" ]]; then
    echo "error: emel.cpp I/O reference identity drifted" >&2
    echo "commit: $source_commit" >&2
    echo "tree:   $source_tree" >&2
    exit 1
  fi
  if ! git -C "$EMEL_CPP_SOURCE" diff --quiet -- src/emel/io tests/io; then
    echo "error: emel.cpp I/O reference files are dirty" >&2
    exit 1
  fi

  materialized_source="$IO_READ_BUILD_DIR/emel-cpp-source"
  cmake -E remove_directory "$materialized_source"
  cmake -E make_directory "$materialized_source"
  git -C "$EMEL_CPP_SOURCE" archive "$EMEL_CPP_COMMIT" | \
    tar -x -C "$materialized_source"

  local cmake_args=(
    -S "$ROOT_DIR/tools/emel-io-read-reference"
    -B "$IO_READ_BUILD_DIR/reference-build"
    -DCMAKE_BUILD_TYPE=Release
    "-DEMEL_CPP_SOURCE_DIR=$materialized_source"
  )
  if command -v ninja >/dev/null 2>&1; then
    cmake_args+=(-G Ninja)
  fi
  cmake "${cmake_args[@]}"
  cmake --build "$IO_READ_BUILD_DIR/reference-build" --parallel \
    --target emel-io-read-reference

  mkdir -p "$IO_READ_BUILD_DIR"
  rust_output="$IO_READ_BUILD_DIR/rust.out"
  reference_output="$IO_READ_BUILD_DIR/reference.out"
  cargo run --quiet --manifest-path "$ROOT_DIR/Cargo.toml" \
    -p emel-io --example read_parity >"$rust_output"
  "$IO_READ_BUILD_DIR/reference-build/emel-io-read-reference" >"$reference_output"
  diff -u "$reference_output" "$rust_output"

  if $RUN_UPDATE; then
    mkdir -p "$(dirname "$IO_READ_SNAPSHOT")"
    install -m 0644 "$reference_output" "$IO_READ_SNAPSHOT"
    echo "Updated I/O read parity snapshot from emel.cpp $EMEL_CPP_COMMIT"
  fi
  if $RUN_SNAPSHOT; then
    diff -u "$IO_READ_SNAPSHOT" "$rust_output"
  fi
  echo "I/O read parity passed (24 cases, emel.cpp $EMEL_CPP_COMMIT)"
}

run_io_mmap_parity() {
  local source_commit source_tree rust_output reference_output materialized_source
  source_commit="$(git -C "$EMEL_CPP_SOURCE" rev-parse HEAD)"
  source_tree="$(git -C "$EMEL_CPP_SOURCE" rev-parse HEAD:src/emel/io)"
  if [[ "$source_commit" != "$EMEL_CPP_COMMIT" || "$source_tree" != "$EMEL_CPP_IO_TREE" ]]; then
    echo "error: emel.cpp I/O reference identity drifted" >&2
    echo "commit: $source_commit" >&2
    echo "tree:   $source_tree" >&2
    exit 1
  fi
  if ! git -C "$EMEL_CPP_SOURCE" diff --quiet -- src/emel/io tests/io; then
    echo "error: emel.cpp I/O reference files are dirty" >&2
    exit 1
  fi

  materialized_source="$IO_MMAP_BUILD_DIR/emel-cpp-source"
  cmake -E remove_directory "$materialized_source"
  cmake -E make_directory "$materialized_source"
  git -C "$EMEL_CPP_SOURCE" archive "$EMEL_CPP_COMMIT" | \
    tar -x -C "$materialized_source"

  local cmake_args=(
    -S "$ROOT_DIR/tools/emel-io-mmap-reference"
    -B "$IO_MMAP_BUILD_DIR/reference-build"
    -DCMAKE_BUILD_TYPE=Release
    "-DEMEL_CPP_SOURCE_DIR=$materialized_source"
  )
  if command -v ninja >/dev/null 2>&1; then
    cmake_args+=(-G Ninja)
  fi
  cmake "${cmake_args[@]}"
  cmake --build "$IO_MMAP_BUILD_DIR/reference-build" --parallel \
    --target emel-io-mmap-reference

  cargo build --manifest-path "$ROOT_DIR/Cargo.toml" -p emel-io \
    --example mmap_parity
  local rust_runner="$ROOT_DIR/target/debug/examples/mmap_parity"
  local fixture="$IO_MMAP_BUILD_DIR/fixture.bin"
  "$rust_runner" --write-fixture "$fixture"
  rust_output="$IO_MMAP_BUILD_DIR/rust.out"
  reference_output="$IO_MMAP_BUILD_DIR/reference.out"
  "$rust_runner" "$fixture" >"$rust_output"
  "$IO_MMAP_BUILD_DIR/reference-build/emel-io-mmap-reference" \
    "$fixture" >"$reference_output"
  diff -u "$reference_output" "$rust_output"
  echo "I/O mmap deterministic lifecycle matches emel.cpp $EMEL_CPP_COMMIT"

  if $RUN_UPDATE; then
    mkdir -p "$(dirname "$IO_MMAP_SNAPSHOT")"
    install -m 0644 "$reference_output" "$IO_MMAP_SNAPSHOT"
    echo "Updated I/O mmap parity snapshot"
  fi
  if $RUN_SNAPSHOT; then
    diff -u "$IO_MMAP_SNAPSHOT" "$rust_output"
  fi

  if ! grep -qx 'native_semantics_complete=true' "$rust_output"; then
    echo "error: I/O mmap parity does not prove complete native semantics" >&2
    return 1
  fi
  echo "I/O mmap parity passed (emel.cpp $EMEL_CPP_COMMIT)"
}

run_io_staged_read_parity() {
  local source_commit source_tree rust_output reference_output materialized_source
  source_commit="$(git -C "$EMEL_CPP_SOURCE" rev-parse HEAD)"
  source_tree="$(git -C "$EMEL_CPP_SOURCE" rev-parse HEAD:src/emel/io)"
  if [[ "$source_commit" != "$EMEL_CPP_COMMIT" || "$source_tree" != "$EMEL_CPP_IO_TREE" ]]; then
    echo "error: emel.cpp I/O reference identity drifted" >&2
    echo "commit: $source_commit" >&2
    echo "tree:   $source_tree" >&2
    exit 1
  fi
  if ! git -C "$EMEL_CPP_SOURCE" diff --quiet -- src/emel/io tests/io; then
    echo "error: emel.cpp I/O reference files are dirty" >&2
    exit 1
  fi

  materialized_source="$IO_STAGED_READ_BUILD_DIR/emel-cpp-source"
  cmake -E remove_directory "$materialized_source"
  cmake -E make_directory "$materialized_source"
  git -C "$EMEL_CPP_SOURCE" archive "$EMEL_CPP_COMMIT" | \
    tar -x -C "$materialized_source"

  local cmake_args=(
    -S "$ROOT_DIR/tools/emel-io-staged-read-reference"
    -B "$IO_STAGED_READ_BUILD_DIR/reference-build"
    -DCMAKE_BUILD_TYPE=Release
    "-DEMEL_CPP_SOURCE_DIR=$materialized_source"
  )
  if command -v ninja >/dev/null 2>&1; then
    cmake_args+=(-G Ninja)
  fi
  cmake "${cmake_args[@]}"
  cmake --build "$IO_STAGED_READ_BUILD_DIR/reference-build" --parallel \
    --target emel-io-staged-read-reference

  mkdir -p "$IO_STAGED_READ_BUILD_DIR"
  rust_output="$IO_STAGED_READ_BUILD_DIR/rust.out"
  reference_output="$IO_STAGED_READ_BUILD_DIR/reference.out"
  cargo run --quiet --manifest-path "$ROOT_DIR/Cargo.toml" \
    -p emel-io --example staged_read_parity >"$rust_output"
  "$IO_STAGED_READ_BUILD_DIR/reference-build/emel-io-staged-read-reference" \
    >"$reference_output"
  diff -u "$reference_output" "$rust_output"

  if $RUN_UPDATE; then
    mkdir -p "$(dirname "$IO_STAGED_READ_SNAPSHOT")"
    install -m 0644 "$reference_output" "$IO_STAGED_READ_SNAPSHOT"
    echo "Updated I/O staged-read parity snapshot from emel.cpp $EMEL_CPP_COMMIT"
  fi
  if $RUN_SNAPSHOT; then
    diff -u "$IO_STAGED_READ_SNAPSHOT" "$rust_output"
  fi
  echo "I/O staged-read parity passed (12 cases, emel.cpp $EMEL_CPP_COMMIT)"
}

run_io_loader_parity() {
  local source_commit source_tree rust_output reference_output materialized_source
  source_commit="$(git -C "$EMEL_CPP_SOURCE" rev-parse HEAD)"
  source_tree="$(git -C "$EMEL_CPP_SOURCE" rev-parse HEAD:src/emel/io)"
  if [[ "$source_commit" != "$EMEL_CPP_COMMIT" || "$source_tree" != "$EMEL_CPP_IO_TREE" ]]; then
    echo "error: emel.cpp I/O reference identity drifted" >&2
    exit 1
  fi
  if ! git -C "$EMEL_CPP_SOURCE" diff --quiet -- src/emel/io tests/io; then
    echo "error: emel.cpp I/O reference files are dirty" >&2
    exit 1
  fi
  materialized_source="$IO_LOADER_BUILD_DIR/emel-cpp-source"
  cmake -E remove_directory "$materialized_source"
  cmake -E make_directory "$materialized_source"
  git -C "$EMEL_CPP_SOURCE" archive "$EMEL_CPP_COMMIT" | tar -x -C "$materialized_source"
  local cmake_args=(-S "$ROOT_DIR/tools/emel-io-loader-reference"
    -B "$IO_LOADER_BUILD_DIR/reference-build" -DCMAKE_BUILD_TYPE=Release
    "-DEMEL_CPP_SOURCE_DIR=$materialized_source")
  if command -v ninja >/dev/null 2>&1; then cmake_args+=(-G Ninja); fi
  cmake "${cmake_args[@]}"
  cmake --build "$IO_LOADER_BUILD_DIR/reference-build" --parallel --target emel-io-loader-reference
  mkdir -p "$IO_LOADER_BUILD_DIR"
  rust_output="$IO_LOADER_BUILD_DIR/rust.out"
  reference_output="$IO_LOADER_BUILD_DIR/reference.out"
  cargo run --quiet --manifest-path "$ROOT_DIR/Cargo.toml" -p emel-io --example loader_parity >"$rust_output"
  "$IO_LOADER_BUILD_DIR/reference-build/emel-io-loader-reference" >"$reference_output"
  diff -u "$reference_output" "$rust_output"
  if $RUN_UPDATE; then
    mkdir -p "$(dirname "$IO_LOADER_SNAPSHOT")"
    install -m 0644 "$reference_output" "$IO_LOADER_SNAPSHOT"
  fi
  if $RUN_SNAPSHOT; then diff -u "$IO_LOADER_SNAPSHOT" "$rust_output"; fi
  echo "I/O loader parity passed (16 cases, emel.cpp $EMEL_CPP_COMMIT)"
}

run_model_tensor_parity() {
  local source_commit source_tree materialized_source rust_output reference_output
  source_commit="$(git -C "$EMEL_CPP_SOURCE" rev-parse HEAD)"
  source_tree="$(git -C "$EMEL_CPP_SOURCE" rev-parse HEAD:src/emel/model/tensor)"
  if [[ "$source_commit" != "$EMEL_CPP_COMMIT" || "$source_tree" != "$EMEL_CPP_MODEL_TENSOR_TREE" ]]; then
    echo "error: emel.cpp model tensor reference identity drifted" >&2
    echo "commit: $source_commit" >&2
    echo "tree:   $source_tree" >&2
    exit 1
  fi
  if ! git -C "$EMEL_CPP_SOURCE" diff --quiet -- src/emel/model/tensor tests/model/tensor; then
    echo "error: emel.cpp model tensor reference files are dirty" >&2
    exit 1
  fi

  materialized_source="$MODEL_TENSOR_BUILD_DIR/emel-cpp-source"
  cmake -E remove_directory "$materialized_source"
  cmake -E make_directory "$materialized_source"
  git -C "$EMEL_CPP_SOURCE" archive "$EMEL_CPP_COMMIT" | tar -x -C "$materialized_source"
  local cmake_args=(
    -S "$ROOT_DIR/tools/emel-model-tensor-reference"
    -B "$MODEL_TENSOR_BUILD_DIR/reference-build"
    -DCMAKE_BUILD_TYPE=Release
    "-DEMEL_CPP_SOURCE_DIR=$materialized_source"
  )
  if command -v ninja >/dev/null 2>&1; then
    cmake_args+=(-G Ninja)
  fi
  cmake "${cmake_args[@]}"
  cmake --build "$MODEL_TENSOR_BUILD_DIR/reference-build" --parallel \
    --target emel-model-tensor-reference

  mkdir -p "$MODEL_TENSOR_BUILD_DIR"
  rust_output="$MODEL_TENSOR_BUILD_DIR/rust.out"
  reference_output="$MODEL_TENSOR_BUILD_DIR/reference.out"
  cargo run --quiet --locked --manifest-path "$ROOT_DIR/Cargo.toml" \
    -p emel-model --example tensor_parity >"$rust_output"
  local mapped_fixture="$MODEL_TENSOR_BUILD_DIR/mapped-fixture.bin"
  cargo run --quiet --locked --manifest-path "$ROOT_DIR/Cargo.toml" \
    -p emel-bench -- --model-tensor-mapped-parity "$mapped_fixture" \
    >>"$rust_output"
  "$MODEL_TENSOR_BUILD_DIR/reference-build/emel-model-tensor-reference" >"$reference_output"
  "$MODEL_TENSOR_BUILD_DIR/reference-build/emel-model-tensor-reference" \
    --mapped-parity "$mapped_fixture" >>"$reference_output"
  rm -f "$mapped_fixture"

  grep -v '^extension=' "$rust_output" >"$MODEL_TENSOR_BUILD_DIR/rust.shared"
  grep -v '^reference_observation=' "$reference_output" >"$MODEL_TENSOR_BUILD_DIR/reference.shared"
  diff -u "$MODEL_TENSOR_BUILD_DIR/reference.shared" "$MODEL_TENSOR_BUILD_DIR/rust.shared"
  grep -qx 'reference_observation=second_plan behavior=unexpected_recovery_to_ready' "$reference_output"
  grep -qx 'reference_observation=unknown_strategy behavior=planned_as_io_load' "$reference_output"
  grep -qx 'extension=typed_busy reference_behavior=unexpected_recovery_to_ready rust_error=busy phase=awaiting_bound' "$rust_output"
  grep -qx 'extension=typed_unknown reference_behavior=planned_as_io_load rust_error=unsupported_strategy' "$rust_output"

  if $RUN_UPDATE; then
    mkdir -p "$(dirname "$MODEL_TENSOR_SNAPSHOT")"
    install -m 0644 "$rust_output" "$MODEL_TENSOR_SNAPSHOT"
    echo "Updated model tensor parity snapshot from emel.cpp $EMEL_CPP_COMMIT"
  fi
  if $RUN_SNAPSHOT; then
    diff -u "$MODEL_TENSOR_SNAPSHOT" "$rust_output"
  fi
  echo "Model tensor shared parity passed with explicit Rust extensions (emel.cpp $EMEL_CPP_COMMIT)"
}

run_model_data_parity() {
  local source_commit source_tree header_blob implementation_blob
  local header_sha256 implementation_sha256 candidate
  source_commit="$(git -C "$EMEL_CPP_SOURCE" rev-parse HEAD)"
  source_tree="$(git -C "$EMEL_CPP_SOURCE" rev-parse HEAD:src/emel/model)"
  header_blob="$(git -C "$EMEL_CPP_SOURCE" rev-parse HEAD:src/emel/model/data.hpp)"
  implementation_blob="$(git -C "$EMEL_CPP_SOURCE" rev-parse HEAD:src/emel/model/data.cpp)"
  if [[ "$source_commit" != "$EMEL_CPP_COMMIT" || \
        "$source_tree" != "$EMEL_CPP_MODEL_TREE" || \
        "$header_blob" != "$EMEL_CPP_MODEL_DATA_HEADER_BLOB" || \
        "$implementation_blob" != "$EMEL_CPP_MODEL_DATA_IMPLEMENTATION_BLOB" ]]; then
    echo "error: emel.cpp model data reference identity drifted" >&2
    exit 1
  fi
  if ! git -C "$EMEL_CPP_SOURCE" diff --quiet -- \
    src/emel/model/data.hpp src/emel/model/data.cpp; then
    echo "error: emel.cpp model data reference files are dirty" >&2
    exit 1
  fi
  if ! command -v shasum >/dev/null 2>&1; then
    echo "error: shasum is required for model data parity" >&2
    exit 1
  fi
  header_sha256="$(git -C "$EMEL_CPP_SOURCE" cat-file blob "$header_blob" | shasum -a 256 | awk '{print $1}')"
  implementation_sha256="$(git -C "$EMEL_CPP_SOURCE" cat-file blob "$implementation_blob" | shasum -a 256 | awk '{print $1}')"
  if [[ "$header_sha256" != "$EMEL_CPP_MODEL_DATA_HEADER_SHA256" || \
        "$implementation_sha256" != "$EMEL_CPP_MODEL_DATA_IMPLEMENTATION_SHA256" ]]; then
    echo "error: emel.cpp model data content digest drifted" >&2
    exit 1
  fi

  mkdir -p "$MODEL_DATA_BUILD_DIR"
  candidate="$MODEL_DATA_BUILD_DIR/manifest.rust.txt"
  rm -f "$candidate"
  if $RUN_UPDATE; then
    EMEL_MODEL_DATA_PARITY_OUTPUT="$candidate" \
      EMEL_MODEL_DATA_PARITY_UPDATE=1 \
      cargo test --quiet --locked --manifest-path "$ROOT_DIR/Cargo.toml" \
        -p emel-model --lib data::tests::parity_snapshot_cases_are_derived_from_behavior \
        -- --exact
    mkdir -p "$(dirname "$MODEL_DATA_SNAPSHOT")"
    install -m 0644 "$candidate" "$MODEL_DATA_SNAPSHOT"
  else
    EMEL_MODEL_DATA_PARITY_OUTPUT="$candidate" \
      cargo test --quiet --locked --manifest-path "$ROOT_DIR/Cargo.toml" \
        -p emel-model --lib data::tests::parity_snapshot_cases_are_derived_from_behavior \
        -- --exact
  fi
  diff -u "$MODEL_DATA_SNAPSHOT" "$candidate"
  echo "Model data parity passed with exact source identity (emel.cpp $EMEL_CPP_COMMIT)"
}

if $RUN_GGUF; then
  if $RUN_UPDATE; then
    update_snapshot
  fi
  if $RUN_LIVE; then
    live_parity
  fi
  if $RUN_SNAPSHOT; then
    check_snapshot
  fi
fi

if $RUN_IO_READ; then
  run_io_read_parity
fi
if $RUN_IO_MMAP; then
  run_io_mmap_parity
fi
if $RUN_IO_STAGED_READ; then
  run_io_staged_read_parity
fi
if $RUN_IO_LOADER; then
  run_io_loader_parity
fi
if $RUN_MODEL_TENSOR; then
  run_model_tensor_parity
fi
if $RUN_MODEL_DATA; then
  run_model_data_parity
fi
