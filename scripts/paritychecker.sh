#!/usr/bin/env bash
set -euo pipefail

if [[ -d /opt/homebrew/bin ]]; then
  export PATH="/opt/homebrew/bin:$PATH"
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${EMEL_GGUF_PARITY_BUILD_DIR:-$ROOT_DIR/target/gguf-parity}"
FIXTURE_DIR="$BUILD_DIR/fixtures"
REFERENCE_BUILD_DIR="$BUILD_DIR/llama-reference"
PACKED_REFERENCE_BUILD_DIR="$BUILD_DIR/emel-packed-reference"
SNAPSHOT="${EMEL_GGUF_PARITY_SNAPSHOT:-$ROOT_DIR/snapshots/parity/gguf/manifest.txt}"
LIVE_MODEL_SNAPSHOT="${EMEL_GGUF_LIVE_MODEL_SNAPSHOT:-$ROOT_DIR/snapshots/parity/gguf/live-model.txt}"
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
MODEL_CATALOG_SNAPSHOT="${EMEL_MODEL_CATALOG_PARITY_SNAPSHOT:-$ROOT_DIR/snapshots/parity/model-catalog/manifest.txt}"
MODEL_CATALOG_BUILD_DIR="${EMEL_MODEL_CATALOG_PARITY_BUILD_DIR:-$ROOT_DIR/target/model-catalog-parity}"
TOKEN_PROFILE_SNAPSHOT="${EMEL_TOKEN_PROFILE_PARITY_SNAPSHOT:-$ROOT_DIR/snapshots/parity/token-profile/manifest.txt}"
TOKEN_PROFILE_BUILD_DIR="${EMEL_TOKEN_PROFILE_PARITY_BUILD_DIR:-$ROOT_DIR/target/token-profile-parity}"
MODEL_VOCAB_SNAPSHOT="${EMEL_MODEL_VOCAB_PARITY_SNAPSHOT:-$ROOT_DIR/snapshots/parity/model-vocabulary/manifest.txt}"
MODEL_VOCAB_BUILD_DIR="${EMEL_MODEL_VOCAB_PARITY_BUILD_DIR:-$ROOT_DIR/target/model-vocabulary-parity}"
KERNEL_CAPABILITY_SNAPSHOT="${EMEL_KERNEL_CAPABILITY_PARITY_SNAPSHOT:-$ROOT_DIR/snapshots/parity/kernel-capability/manifest.txt}"
KERNEL_CAPABILITY_BUILD_DIR="${EMEL_KERNEL_CAPABILITY_PARITY_BUILD_DIR:-$ROOT_DIR/target/kernel-capability-parity}"
EMEL_CPP_SOURCE="${EMEL_CPP_SOURCE_DIR:-$ROOT_DIR/../emel.cpp}"
EMEL_CPP_COMMIT=843a117386ef17dc5a50549bbfc821074c2141d6
EMEL_CPP_IO_TREE=ff00a9978b00b4ea1e5268d2ecd483d4855b6eaa
EMEL_CPP_MODEL_TENSOR_TREE=06306d4ffad3455fcf5df71dc692df52514b9865
EMEL_CPP_MODEL_TREE=278b7b20545b630be33bee8bed0cb4c8db8990c3
EMEL_CPP_MODEL_DATA_HEADER_BLOB=78a25b987423d8cbef17965a8ca92596ffc0ecef
EMEL_CPP_MODEL_DATA_IMPLEMENTATION_BLOB=b33ace170b569d076844a36146c7ca86d4ffa7fc
EMEL_CPP_MODEL_DATA_HEADER_SHA256=4b604d57fef1c22c8a9ebee36779a0c9cd4e7fe811856455d5cafd645498a151
EMEL_CPP_MODEL_DATA_IMPLEMENTATION_SHA256=c8231f4feb2bc395a642bf3d66e74f5ee6ad243d706f277245b2b28c620c8816
EMEL_CPP_TOKEN_MODEL_BLOB=ef7ff8da51f1f281082901bf4919a4b9a63f2671
EMEL_CPP_TOKEN_PRE_BLOB=16b2982ca16dfdfbee016d50d0eb924a7cdc18c4
EMEL_CPP_TOKEN_MODEL_SHA256=60e10d81dda5c2c3a47a4d22c1b93b20c5d2077a4869f5a815b9c1e6881829b0
EMEL_CPP_TOKEN_PRE_SHA256=b9c8cf6a0680802f94c3b8b2247af0306c5233adaada0b00fe14df178e5a9721
EMEL_CPP_KERNEL_EVENTS_BLOB=4b3f02fa5ff9c5071d1fc83938cef1b22b4658b9
EMEL_CPP_KERNEL_DETAIL_BLOB=c8a82643eabfe8f2d7883e655955f455794511b0
EMEL_CPP_GENERATION_HEADER_BLOB=d521cf68e1bf52a2a193bbdb460741772199b318
EMEL_CPP_GENERATION_BLOB=099058ccd441d1dc6bebbb0c4994070d2f533c47
EMEL_KERNEL_CAPABILITY_SML_COMMIT=49207123cd3f39767764bae774932cb48623f92f
EMEL_KERNEL_CAPABILITY_SML_SOURCE="${EMEL_STATEFORWARD_SML_SOURCE:-$EMEL_CPP_SOURCE/build/zig/_deps/stateforward_sml-src}"
GGUF_LIVE_MODEL_RELATIVE=tests/models/Llama-68M-Chat-v1-Q2_K.gguf
GGUF_LIVE_MODEL_LFS_BLOB=49aa0ba91b29a4015339757cc67544f614e0fa21
GGUF_LIVE_MODEL_SHA256=8ed06dc5bd84bce3154a2b7e751c45a56562691933ee25b5823393f909329a67
GGUF_LIVE_MODEL_BYTES=35877760
EXPECTED_REF="$(tr -d '[:space:]' <"$ROOT_DIR/tools/llama-gguf-reference/reference_ref.txt")"
RUN_SNAPSHOT=true
RUN_LIVE=true
RUN_UPDATE=true
REFERENCE_SOURCE="${LLAMA_CPP_SOURCE_DIR:-}"
REFERENCE_CONFIGURED=false
PACKED_REFERENCE_CONFIGURED=false
SUITE=all
models=()
PINNED_LIVE_MODEL=

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
  --suite=NAME     run all, gguf, io-read, io-mmap, io-staged-read, io-loader, model-tensor, model-data, model-catalog, token-profile, model-vocab, or kernel-capability (default: all)

Model paths are checked during the live phase. Without paths, both the
deterministic fixture corpus and the pinned independently sourced model run.
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
    --suite=model-catalog) SUITE=model-catalog ;;
    --suite=token-profile) SUITE=token-profile ;;
    --suite=model-vocab) SUITE=model-vocab ;;
    --suite=kernel-capability) SUITE=kernel-capability ;;
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
RUN_MODEL_CATALOG=false
RUN_TOKEN_PROFILE=false
RUN_MODEL_VOCAB=false
RUN_KERNEL_CAPABILITY=false
case "$SUITE" in
  all)
    RUN_GGUF=true
    RUN_IO_READ=true
    RUN_IO_MMAP=true
    RUN_IO_STAGED_READ=true
    RUN_IO_LOADER=true
    RUN_MODEL_TENSOR=true
    RUN_MODEL_DATA=true
    RUN_MODEL_CATALOG=true
    RUN_TOKEN_PROFILE=true
    RUN_MODEL_VOCAB=true
    RUN_KERNEL_CAPABILITY=true
    ;;
  gguf) RUN_GGUF=true ;;
  io-read) RUN_IO_READ=true ;;
  io-mmap) RUN_IO_MMAP=true ;;
  io-staged-read) RUN_IO_STAGED_READ=true ;;
  io-loader) RUN_IO_LOADER=true ;;
  model-tensor) RUN_MODEL_TENSOR=true ;;
  model-data) RUN_MODEL_DATA=true ;;
  model-catalog) RUN_MODEL_CATALOG=true ;;
  token-profile) RUN_TOKEN_PROFILE=true ;;
  model-vocab) RUN_MODEL_VOCAB=true ;;
  kernel-capability) RUN_KERNEL_CAPABILITY=true ;;
esac

if $RUN_GGUF || $RUN_IO_READ || $RUN_IO_MMAP || $RUN_IO_STAGED_READ || \
  $RUN_IO_LOADER || $RUN_MODEL_TENSOR || $RUN_TOKEN_PROFILE || \
  $RUN_MODEL_VOCAB || $RUN_KERNEL_CAPABILITY; then
  if ! command -v cmake >/dev/null 2>&1; then
    echo "error: cmake is required for the selected parity suite" >&2
    exit 2
  fi
fi
if $RUN_TOKEN_PROFILE && ! command -v rg >/dev/null 2>&1; then
  echo "error: rg is required for tokenizer profile dependency checks" >&2
  exit 2
fi

fixture_models=()
if $RUN_GGUF; then
  if ! command -v rg >/dev/null 2>&1; then
    echo "error: rg is required for GGUF dependency-boundary checks" >&2
    exit 2
  fi
  if rg -n 'WithMetadataDescriptor|MetadataDescriptor|MetadataKind' \
      "$ROOT_DIR/crates/emel-model" >/dev/null; then
    echo "error: emel-model must not consume the GGUF metadata descriptor event" >&2
    exit 2
  fi
  if rg -n 'emel_gguf::loader|emel_gguf::[^;]*(KvEntry|TensorInfo)' \
      "$ROOT_DIR/crates/emel-model" >/dev/null; then
    echo "error: emel-model must not consume private or raw GGUF loader records" >&2
    exit 2
  fi
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

prepare_pinned_live_model() {
  local pointer expected_pointer digest bytes
  if ! git -C "$EMEL_CPP_SOURCE" cat-file -e \
      "$EMEL_CPP_COMMIT:$GGUF_LIVE_MODEL_RELATIVE" 2>/dev/null; then
    echo "error: pinned emel.cpp GGUF source identity is unavailable" >&2
    exit 1
  fi
  pointer="$(git -C "$EMEL_CPP_SOURCE" show \
    "$EMEL_CPP_COMMIT:$GGUF_LIVE_MODEL_RELATIVE")"
  expected_pointer="$(printf '%s\n%s\n%s' \
    'version https://git-lfs.github.com/spec/v1' \
    "oid sha256:$GGUF_LIVE_MODEL_SHA256" \
    "size $GGUF_LIVE_MODEL_BYTES")"
  if [[ "$pointer" != "$expected_pointer" ]]; then
    echo "error: pinned real-model Git LFS identity drifted" >&2
    exit 1
  fi
  if [[ "$(git -C "$EMEL_CPP_SOURCE" rev-parse \
      "$EMEL_CPP_COMMIT:$GGUF_LIVE_MODEL_RELATIVE")" != \
      "$GGUF_LIVE_MODEL_LFS_BLOB" ]]; then
    echo "error: pinned real-model source blob drifted" >&2
    exit 1
  fi

  PINNED_LIVE_MODEL="$EMEL_CPP_SOURCE/$GGUF_LIVE_MODEL_RELATIVE"
  if [[ ! -f "$PINNED_LIVE_MODEL" ]]; then
    echo "error: pinned real GGUF is not materialized; fetch the emel.cpp LFS object" >&2
    exit 1
  fi
  digest="$(sha256_file "$PINNED_LIVE_MODEL")"
  bytes="$(wc -c <"$PINNED_LIVE_MODEL" | tr -d '[:space:]')"
  if [[ "$digest" != "$GGUF_LIVE_MODEL_SHA256" || \
        "$bytes" != "$GGUF_LIVE_MODEL_BYTES" ]]; then
    echo "error: materialized pinned real GGUF identity drifted" >&2
    exit 1
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

configure_packed_reference() {
  if $PACKED_REFERENCE_CONFIGURED; then
    return
  fi
  local source_archive="$PACKED_REFERENCE_BUILD_DIR/emel-cpp-source"
  [[ "$(git -C "$EMEL_CPP_SOURCE" rev-parse "$EMEL_CPP_COMMIT:src/emel/kernel/events.hpp")" == "4b3f02fa5ff9c5071d1fc83938cef1b22b4658b9" ]] || {
    echo "error: pinned emel.cpp kernel events identity drifted" >&2
    exit 1
  }
  [[ "$(git -C "$EMEL_CPP_SOURCE" rev-parse "$EMEL_CPP_COMMIT:src/emel/kernel/detail.hpp")" == "c8a82643eabfe8f2d7883e655955f455794511b0" ]] || {
    echo "error: pinned emel.cpp kernel detail identity drifted" >&2
    exit 1
  }
  [[ "$(git -C "$EMEL_CPP_SOURCE" rev-parse "$EMEL_CPP_COMMIT:src/emel/gguf/loader/detail.hpp")" == "4ec9829c6d640cfa65a6d546c5891cb048b29f90" ]] || {
    echo "error: pinned emel.cpp GGUF loader detail identity drifted" >&2
    exit 1
  }
  cmake -E remove_directory "$source_archive"
  cmake -E make_directory "$source_archive"
  git -C "$EMEL_CPP_SOURCE" archive "$EMEL_CPP_COMMIT" | tar -x -C "$source_archive"
  local cmake_args=(
    -S "$ROOT_DIR/tools/emel-gguf-packed-reference"
    -B "$PACKED_REFERENCE_BUILD_DIR/build"
    -DCMAKE_BUILD_TYPE=Release
    "-DEMEL_CPP_SOURCE_DIR=$source_archive"
  )
  if command -v ninja >/dev/null 2>&1; then
    cmake_args+=(-G Ninja)
  fi
  cmake "${cmake_args[@]}"
  cmake --build "$PACKED_REFERENCE_BUILD_DIR/build" --parallel \
    --target emel-gguf-packed-reference
  PACKED_REFERENCE_RUNNER="$PACKED_REFERENCE_BUILD_DIR/build/emel-gguf-packed-reference"
  PACKED_REFERENCE_CONFIGURED=true
}

run_reference() {
  local model="$1"
  local output="$2"
  local model_directory model_absolute
  model_directory="$(cd "$(dirname "$model")" && pwd -P)"
  model_absolute="$model_directory/$(basename "$model")"

  local runner="$REFERENCE_RUNNER"
  if [[ "$model_absolute" == "$FIXTURE_DIR/valid/tensors.gguf" ]]; then
    configure_packed_reference >&2
    runner="$PACKED_REFERENCE_RUNNER"
  fi
  if ! { "$runner" "$model" >"$output"; } 2>"$BUILD_DIR/reference.err"; then
    if [[ "$model_absolute" == "$FIXTURE_DIR/invalid/"* ]]; then
      printf 'gguf-parity/v1\nstatus=error\n' >"$output"
      return
    fi
    echo "llama.cpp reference runner failed: $model" >&2
    cat "$BUILD_DIR/reference.err" >&2
    exit 1
  fi
}

compare_live_model() {
  local model="$1"
  local label="$2"
  local rust_output="$BUILD_DIR/rust.out"
  local reference_output="$BUILD_DIR/reference.out"
  configure_reference
  "$RUST_RUNNER" "$model" >"$rust_output"
  run_reference "$model" "$reference_output"
  if ! diff -u "$reference_output" "$rust_output"; then
    echo "GGUF live parity failed: $label" >&2
    exit 1
  fi
  echo "GGUF live parity passed: $label"
}

write_live_model_evidence() {
  local destination="$1"
  {
    echo "gguf-live-model-parity/v1"
    echo "source_repository=https://github.com/stateforward/emel.cpp"
    echo "source_commit=$EMEL_CPP_COMMIT"
    echo "model=$GGUF_LIVE_MODEL_RELATIVE"
    echo "model_lfs_blob=$GGUF_LIVE_MODEL_LFS_BLOB"
    echo "model_sha256=$GGUF_LIVE_MODEL_SHA256"
    echo "model_bytes=$GGUF_LIVE_MODEL_BYTES"
    echo "reference_ref=$EXPECTED_REF"
    echo "emel_lane=workspace-emel-gguf-public-loader-events"
    echo "reference_lane=out-of-process-llama-gguf-reference"
    echo "comparison=canonical-gguf-parity-v1-byte-exact"
    echo "scope=all-metadata-descriptors-typed-values-and-tensor-descriptor-payload-digests"
    echo "command=scripts/paritychecker.sh --suite=gguf"
    echo "default_phases=update,live,snapshot"
    echo "result=pass"
  } >"$destination"
}

update_live_model_snapshot() {
  local candidate="$BUILD_DIR/live-model.reference.txt"
  prepare_pinned_live_model
  compare_live_model "$PINNED_LIVE_MODEL" "$GGUF_LIVE_MODEL_RELATIVE"
  write_live_model_evidence "$candidate"
  mkdir -p "$(dirname "$LIVE_MODEL_SNAPSHOT")"
  install -m 0644 "$candidate" "$LIVE_MODEL_SNAPSHOT"
  echo "Updated pinned real-model parity evidence"
}

check_live_model_snapshot() {
  local candidate="$BUILD_DIR/live-model.actual.txt"
  if [[ ! -f "$LIVE_MODEL_SNAPSHOT" ]]; then
    echo "error: missing live-model parity evidence: $LIVE_MODEL_SNAPSHOT" >&2
    exit 1
  fi
  prepare_pinned_live_model
  compare_live_model "$PINNED_LIVE_MODEL" "$GGUF_LIVE_MODEL_RELATIVE"
  write_live_model_evidence "$candidate"
  diff -u "$LIVE_MODEL_SNAPSHOT" "$candidate"
  echo "GGUF pinned real-model snapshot passed"
}

write_manifest() {
  local runner="$1"
  local destination="$2"
  local output="$BUILD_DIR/$runner.out"
  {
    echo "gguf-parity-snapshot/v1"
    echo "reference_ref=$EXPECTED_REF"
    echo "packed_reference_repository=https://github.com/stateforward/emel.cpp"
    echo "packed_reference_commit=$EMEL_CPP_COMMIT"
    echo "packed_reference_loader_detail_blob=4ec9829c6d640cfa65a6d546c5891cb048b29f90"
    echo "packed_reference_lane=out-of-process-public-loader-events"
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
  check_live_model_snapshot
}

live_parity() {
  configure_reference
  local selected=()
  if [[ ${#models[@]} -eq 0 ]]; then
    prepare_pinned_live_model
    selected=("${fixture_models[@]}" "$PINNED_LIVE_MODEL")
  else
    selected=("${models[@]}")
  fi
  local model label
  for model in "${selected[@]}"; do
    label="$model"
    if [[ "$model" == "$PINNED_LIVE_MODEL" ]]; then
      label="$GGUF_LIVE_MODEL_RELATIVE"
    fi
    compare_live_model "$model" "$label"
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
  update_live_model_snapshot
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

run_model_catalog_parity() {
  local mode=()
  if $RUN_UPDATE; then
    mode=(--update)
  elif $RUN_LIVE && ! $RUN_SNAPSHOT; then
    mode=(--live)
  fi
  if [[ ${#mode[@]} -eq 0 ]]; then
    EMEL_CPP_SOURCE_DIR="$EMEL_CPP_SOURCE" \
      EMEL_MODEL_CATALOG_PARITY_BUILD_DIR="$MODEL_CATALOG_BUILD_DIR" \
      EMEL_MODEL_CATALOG_PARITY_SNAPSHOT="$MODEL_CATALOG_SNAPSHOT" \
      "$ROOT_DIR/scripts/model-catalog-parity.sh"
  else
    EMEL_CPP_SOURCE_DIR="$EMEL_CPP_SOURCE" \
      EMEL_MODEL_CATALOG_PARITY_BUILD_DIR="$MODEL_CATALOG_BUILD_DIR" \
      EMEL_MODEL_CATALOG_PARITY_SNAPSHOT="$MODEL_CATALOG_SNAPSHOT" \
      "$ROOT_DIR/scripts/model-catalog-parity.sh" "${mode[0]}"
  fi
  if [[ ${#mode[@]} -eq 0 ]]; then
    EMEL_CPP_SOURCE_DIR="$EMEL_CPP_SOURCE" \
      "$ROOT_DIR/scripts/model-generation-parity.sh"
  else
    EMEL_CPP_SOURCE_DIR="$EMEL_CPP_SOURCE" \
      "$ROOT_DIR/scripts/model-generation-parity.sh" "${mode[0]}"
  fi
}

run_token_profile_parity() {
  local model_blob pre_blob model_sha256 pre_sha256 candidate dependency_tree
  local materialized_source reference_build reference_output reference_runner
  model_blob="$(git -C "$EMEL_CPP_SOURCE" rev-parse \
    "$EMEL_CPP_COMMIT:src/emel/text/tokenizer/detail.hpp")"
  pre_blob="$(git -C "$EMEL_CPP_SOURCE" rev-parse \
    "$EMEL_CPP_COMMIT:src/emel/text/tokenizer/preprocessor/detail.hpp")"
  if [[ "$model_blob" != "$EMEL_CPP_TOKEN_MODEL_BLOB" || \
        "$pre_blob" != "$EMEL_CPP_TOKEN_PRE_BLOB" ]]; then
    echo "error: emel.cpp tokenizer profile source identity drifted" >&2
    exit 1
  fi
  model_sha256="$(git -C "$EMEL_CPP_SOURCE" cat-file blob "$model_blob" | shasum -a 256 | awk '{print $1}')"
  pre_sha256="$(git -C "$EMEL_CPP_SOURCE" cat-file blob "$pre_blob" | shasum -a 256 | awk '{print $1}')"
  if [[ "$model_sha256" != "$EMEL_CPP_TOKEN_MODEL_SHA256" || \
        "$pre_sha256" != "$EMEL_CPP_TOKEN_PRE_SHA256" ]]; then
    echo "error: emel.cpp tokenizer profile source digest drifted" >&2
    exit 1
  fi
  if ! git -C "$EMEL_CPP_SOURCE" diff --quiet -- \
    src/emel/text/tokenizer/detail.hpp \
    src/emel/text/tokenizer/preprocessor/detail.hpp; then
    echo "error: emel.cpp tokenizer profile source files are dirty" >&2
    exit 1
  fi
  dependency_tree="$(cargo tree --locked --manifest-path "$ROOT_DIR/Cargo.toml" -p emel-token --prefix none)"
  if grep -Eq '^emel-(model|text|gguf) ' <<<"$dependency_tree"; then
    echo "error: emel-token profile owner depends on a downstream model/text/GGUF crate" >&2
    exit 1
  fi
  if rg -n '^pub use .*profile::event' "$ROOT_DIR/crates/emel-token/src/lib.rs" >/dev/null; then
    echo "error: tokenizer profile events must remain in the coherent owner namespace" >&2
    exit 1
  fi

  mkdir -p "$TOKEN_PROFILE_BUILD_DIR"
  materialized_source="$TOKEN_PROFILE_BUILD_DIR/emel-cpp-source"
  reference_build="$TOKEN_PROFILE_BUILD_DIR/reference-build"
  reference_output="$TOKEN_PROFILE_BUILD_DIR/manifest.cpp.txt"
  cmake -E remove_directory "$materialized_source"
  cmake -E remove_directory "$reference_build"
  cmake -E make_directory "$materialized_source"
  git -C "$EMEL_CPP_SOURCE" archive "$EMEL_CPP_COMMIT" | \
    tar -x -C "$materialized_source"
  cmake -S "$ROOT_DIR/tools/emel-token-profile-reference" \
    -B "$reference_build" \
    -DCMAKE_BUILD_TYPE=Release \
    "-DEMEL_CPP_SOURCE_DIR=$materialized_source"
  cmake --build "$reference_build"
  reference_runner="$reference_build/emel-token-profile-reference"
  "$reference_runner" >"$reference_output"

  candidate="$TOKEN_PROFILE_BUILD_DIR/manifest.rust.txt"
  EMEL_TOKEN_PROFILE_PARITY_OUTPUT="$candidate" \
    cargo test --quiet --locked --manifest-path "$ROOT_DIR/Cargo.toml" \
      -p emel-token --lib profile::tests::parity_snapshot_is_source_backed -- --exact
  diff -u "$reference_output" "$candidate"
  if $RUN_UPDATE; then
    mkdir -p "$(dirname "$TOKEN_PROFILE_SNAPSHOT")"
    install -m 0644 "$candidate" "$TOKEN_PROFILE_SNAPSHOT"
    echo "Updated tokenizer profile parity snapshot from emel.cpp $EMEL_CPP_COMMIT"
  fi
  if $RUN_LIVE; then
    echo "Tokenizer profile independent C++/Rust live comparison passed"
  fi
  if $RUN_SNAPSHOT; then
    diff -u "$TOKEN_PROFILE_SNAPSHOT" "$candidate"
    echo "Tokenizer profile checked snapshot passed"
  fi
  echo "Tokenizer profile parity passed (10 model inputs including aliases/unknown, 61 pre inputs, unknown success)"
}

run_model_vocab_parity() {
  local candidate="$MODEL_VOCAB_BUILD_DIR/manifest.observed.txt"
  mkdir -p "$MODEL_VOCAB_BUILD_DIR"
  EMEL_CPP_SOURCE_REPO="$EMEL_CPP_SOURCE" \
    "$ROOT_DIR/scripts/model-vocab-parity.sh" >"$candidate"
  grep -qx 'result=match' "$candidate"

  if $RUN_UPDATE; then
    mkdir -p "$(dirname "$MODEL_VOCAB_SNAPSHOT")"
    install -m 0644 "$candidate" "$MODEL_VOCAB_SNAPSHOT"
    echo "Updated model vocabulary parity snapshot from emel.cpp $EMEL_CPP_COMMIT"
  fi
  if $RUN_LIVE; then
    echo "Model vocabulary independent C++/Rust live comparison passed"
  fi
  if $RUN_SNAPSHOT; then
    if [[ ! -f "$MODEL_VOCAB_SNAPSHOT" ]]; then
      echo "error: missing model vocabulary parity snapshot: $MODEL_VOCAB_SNAPSHOT" >&2
      exit 1
    fi
    diff -u "$MODEL_VOCAB_SNAPSHOT" "$candidate"
    echo "Model vocabulary checked snapshot passed"
  fi
}

run_kernel_capability_parity() {
  local events_blob detail_blob generation_header_blob generation_blob
  local materialized_source reference_build
  local reference_output rust_output reference_runner
  events_blob="$(git -C "$EMEL_CPP_SOURCE" rev-parse \
    "$EMEL_CPP_COMMIT:src/emel/kernel/events.hpp")"
  detail_blob="$(git -C "$EMEL_CPP_SOURCE" rev-parse \
    "$EMEL_CPP_COMMIT:src/emel/kernel/detail.hpp")"
  generation_header_blob="$(git -C "$EMEL_CPP_SOURCE" rev-parse \
    "$EMEL_CPP_COMMIT:src/emel/model/generation/any.hpp")"
  generation_blob="$(git -C "$EMEL_CPP_SOURCE" rev-parse \
    "$EMEL_CPP_COMMIT:src/emel/model/generation/any.cpp")"
  if [[ "$events_blob" != "$EMEL_CPP_KERNEL_EVENTS_BLOB" || \
        "$detail_blob" != "$EMEL_CPP_KERNEL_DETAIL_BLOB" || \
        "$generation_header_blob" != "$EMEL_CPP_GENERATION_HEADER_BLOB" || \
        "$generation_blob" != "$EMEL_CPP_GENERATION_BLOB" ]]; then
    echo "error: emel.cpp kernel capability source identity drifted" >&2
    exit 1
  fi

  materialized_source="$KERNEL_CAPABILITY_BUILD_DIR/emel-cpp-source"
  reference_build="$KERNEL_CAPABILITY_BUILD_DIR/reference-build"
  reference_output="$KERNEL_CAPABILITY_BUILD_DIR/manifest.cpp.txt"
  rust_output="$KERNEL_CAPABILITY_BUILD_DIR/manifest.rust.txt"
  cmake -E remove_directory "$materialized_source"
  cmake -E remove_directory "$reference_build"
  cmake -E make_directory "$materialized_source"
  if [[ "$(git -C "$EMEL_KERNEL_CAPABILITY_SML_SOURCE" rev-parse HEAD)" != \
        "$EMEL_KERNEL_CAPABILITY_SML_COMMIT" ]]; then
    echo "error: stateforward-sml source is not pinned at $EMEL_KERNEL_CAPABILITY_SML_COMMIT" >&2
    exit 1
  fi
  git -C "$EMEL_CPP_SOURCE" archive "$EMEL_CPP_COMMIT" \
    CMakeLists.txt cmake include src | \
    tar -x -C "$materialized_source"
  cmake -S "$ROOT_DIR/tools/emel-kernel-capability-reference" \
    -B "$reference_build" \
    -DCMAKE_BUILD_TYPE=Release \
    "-DFETCHCONTENT_SOURCE_DIR_STATEFORWARD_SML=$EMEL_KERNEL_CAPABILITY_SML_SOURCE" \
    "-DEMEL_CPP_SOURCE_DIR=$materialized_source"
  cmake --build "$reference_build" --parallel \
    --target emel-kernel-capability-reference
  reference_runner="$reference_build/emel-kernel-capability-reference"
  "$reference_runner" --parity >"$reference_output"
  cargo run --quiet --locked --manifest-path "$ROOT_DIR/Cargo.toml" \
    -p emel-kernel-capability-parity -- --parity >"$rust_output"

  diff -u "$reference_output" "$rust_output"
  if [[ "$(grep -c '^scope=' "$rust_output")" -ne 68 || \
        "$(grep -c '^label=' "$rust_output")" -ne 7 ]]; then
    echo "error: kernel capability parity must contain 68 scope rows and 7 labels" >&2
    exit 1
  fi
  if $RUN_UPDATE; then
    mkdir -p "$(dirname "$KERNEL_CAPABILITY_SNAPSHOT")"
    install -m 0644 "$rust_output" "$KERNEL_CAPABILITY_SNAPSHOT"
    echo "Updated kernel capability parity snapshot from emel.cpp $EMEL_CPP_COMMIT"
  fi
  if $RUN_LIVE; then
    echo "Kernel capability independent C++/Rust live comparison passed"
  fi
  if $RUN_SNAPSHOT; then
    diff -u "$KERNEL_CAPABILITY_SNAPSHOT" "$rust_output"
    echo "Kernel capability checked snapshot passed"
  fi
  echo "Kernel capability parity passed (34 serialized types x 2 contract scopes, 7 labels)"
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
if $RUN_MODEL_CATALOG; then
  run_model_catalog_parity
fi
if $RUN_TOKEN_PROFILE; then
  run_token_profile_parity
fi
if $RUN_MODEL_VOCAB; then
  run_model_vocab_parity
fi
if $RUN_KERNEL_CAPABILITY; then
  run_kernel_capability_parity
fi
