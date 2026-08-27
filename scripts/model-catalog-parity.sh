#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EMEL_CPP_SOURCE="${EMEL_CPP_SOURCE_DIR:-$ROOT_DIR/../emel.cpp-worktrees/pinned-843a117386ef17dc5a50549bbfc821074c2141d6}"
SML_SOURCE="${EMEL_STATEFORWARD_SML_SOURCE:-$ROOT_DIR/../../sml/sml.cpp-worktrees/pinned-49207123cd3f39767764bae774932cb48623f92f}"
BUILD_DIR="${EMEL_MODEL_CATALOG_PARITY_BUILD_DIR:-$ROOT_DIR/target/model-catalog-parity}"
SNAPSHOT="${EMEL_MODEL_CATALOG_PARITY_SNAPSHOT:-$ROOT_DIR/snapshots/parity/model-catalog/manifest.txt}"
SOURCE_COMMIT=843a117386ef17dc5a50549bbfc821074c2141d6
SML_COMMIT=49207123cd3f39767764bae774932cb48623f92f
DATA_HEADER_BLOB=78a25b987423d8cbef17965a8ca92596ffc0ecef
DATA_IMPLEMENTATION_BLOB=b33ace170b569d076844a36146c7ca86d4ffa7fc
GENERATION_HEADER_BLOB=d521cf68e1bf52a2a193bbdb460741772199b318
GENERATION_IMPLEMENTATION_BLOB=099058ccd441d1dc6bebbb0c4994070d2f533c47
LLAMA_RELATIVE=tests/models/Llama-68M-Chat-v1-Q2_K.gguf
LLAMA_LFS_BLOB=49aa0ba91b29a4015339757cc67544f614e0fa21
LLAMA_SHA256=8ed06dc5bd84bce3154a2b7e751c45a56562691933ee25b5823393f909329a67
LLAMA_BYTES=35877760
LFM_RELATIVE=tests/models/LFM2.5-230M-Q8_0.gguf
LFM_LFS_BLOB=d85b80d116e1e76ea5153585d3d3ab80457f7ea9
LFM_SHA256=855be85429300602eda72958547614703541b7d6dd965a8f8f6052b85a7aa935
LFM_BYTES=246598496
OPERAND_SPEC='model-catalog-v1|synthetic=same,same,ff0078,empty,unbound,zero-size,five-good|benchmark=tensor.000..tensor.255,target=tensor.255|digest=fnv1a64,name,type,n_dims,active_dims,data_size,binding_status'
OPERAND_SHA256=4c0c663efcac31d2c927c729d2f88aeb3fc52dc07a55d8f47c6fce9a42fda4b7
UPDATE=false
COMPARE_SNAPSHOT=true

if [[ $# -gt 1 || (${1:-} != "" && ${1:-} != "--update" && ${1:-} != "--live") ]]; then
  echo "usage: scripts/model-catalog-parity.sh [--update|--live]" >&2
  exit 2
fi
if [[ ${1:-} == "--update" ]]; then
  UPDATE=true
elif [[ ${1:-} == "--live" ]]; then
  COMPARE_SNAPSHOT=false
fi

sha256_file() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{print $1}'
  else
    shasum -a 256 "$1" | awk '{print $1}'
  fi
}

sha256_text() {
  if command -v sha256sum >/dev/null 2>&1; then
    printf '%s' "$1" | sha256sum | awk '{print $1}'
  else
    printf '%s' "$1" | shasum -a 256 | awk '{print $1}'
  fi
}

verify_blob() {
  local path="$1" expected="$2"
  [[ "$(git -C "$EMEL_CPP_SOURCE" rev-parse "$SOURCE_COMMIT:$path")" == "$expected" ]] || {
    echo "error: pinned model catalog source blob drifted: $path" >&2
    exit 1
  }
}

verify_fixture() {
  local relative="$1" blob="$2" digest="$3" bytes="$4" file="$EMEL_CPP_SOURCE/$1"
  local pointer expected_pointer
  [[ "$(git -C "$EMEL_CPP_SOURCE" rev-parse "$SOURCE_COMMIT:$relative")" == "$blob" ]] || {
    echo "error: pinned model catalog fixture LFS blob drifted: $relative" >&2
    exit 1
  }
  pointer="$(git -C "$EMEL_CPP_SOURCE" show "$SOURCE_COMMIT:$relative")"
  expected_pointer="$(printf '%s\n%s\n%s' \
    'version https://git-lfs.github.com/spec/v1' \
    "oid sha256:$digest" "size $bytes")"
  [[ "$pointer" == "$expected_pointer" ]] || {
    echo "error: pinned model catalog fixture pointer drifted: $relative" >&2
    exit 1
  }
  [[ -f "$file" && "$(sha256_file "$file")" == "$digest" && \
    "$(wc -c <"$file" | tr -d '[:space:]')" == "$bytes" ]] || {
    echo "error: materialized model catalog fixture identity drifted: $relative" >&2
    exit 1
  }
}

[[ "$(git -C "$EMEL_CPP_SOURCE" rev-parse HEAD)" == "$SOURCE_COMMIT" ]] || {
  echo "error: emel.cpp is not at the pinned model catalog commit" >&2
  exit 1
}
[[ "$(git -C "$SML_SOURCE" rev-parse HEAD)" == "$SML_COMMIT" ]] || {
  echo "error: stateforward-sml is not at the pinned model catalog commit" >&2
  exit 1
}
verify_blob src/emel/model/data.hpp "$DATA_HEADER_BLOB"
verify_blob src/emel/model/data.cpp "$DATA_IMPLEMENTATION_BLOB"
verify_blob src/emel/model/generation/any.hpp "$GENERATION_HEADER_BLOB"
verify_blob src/emel/model/generation/any.cpp "$GENERATION_IMPLEMENTATION_BLOB"
verify_fixture "$LLAMA_RELATIVE" "$LLAMA_LFS_BLOB" "$LLAMA_SHA256" "$LLAMA_BYTES"
verify_fixture "$LFM_RELATIVE" "$LFM_LFS_BLOB" "$LFM_SHA256" "$LFM_BYTES"
[[ "$(sha256_text "$OPERAND_SPEC")" == "$OPERAND_SHA256" ]] || {
  echo "error: model catalog operand identity drifted" >&2
  exit 1
}

rm -rf "$BUILD_DIR/source"
mkdir -p "$BUILD_DIR/source"
git -C "$EMEL_CPP_SOURCE" archive "$SOURCE_COMMIT" include src | tar -x -C "$BUILD_DIR/source"
linker_gc=(-Wl,--gc-sections)
if [[ "$(uname -s)" == Darwin ]]; then
  linker_gc=(-Wl,-dead_strip)
fi
"${CXX:-c++}" -std=c++20 -O2 -ffunction-sections -fdata-sections \
  -I"$BUILD_DIR/source/include" -I"$BUILD_DIR/source/src" -I"$SML_SOURCE/include" \
  "$ROOT_DIR/tools/emel-model-catalog-reference/main.cpp" \
  "$BUILD_DIR/source/src/emel/model/data.cpp" \
  "$BUILD_DIR/source/src/emel/model/generation/any.cpp" \
  "${linker_gc[@]}" -o "$BUILD_DIR/cpp-observer"
CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-$ROOT_DIR/target}" cargo build --quiet --locked --manifest-path "$ROOT_DIR/Cargo.toml" \
  -p emel-model --example catalog_observer
RUST_OBSERVER="${CARGO_TARGET_DIR:-$ROOT_DIR/target}/debug/examples/catalog_observer"
[[ -x "$RUST_OBSERVER" ]] || { echo "error: missing Rust catalog observer: $RUST_OBSERVER" >&2; exit 1; }

"$BUILD_DIR/cpp-observer" >"$BUILD_DIR/cpp.synthetic"
"$RUST_OBSERVER" >"$BUILD_DIR/rust.synthetic"
grep '^source_case=' "$BUILD_DIR/cpp.synthetic" >"$BUILD_DIR/cpp.shared"
grep '^source_case=' "$BUILD_DIR/rust.synthetic" >"$BUILD_DIR/rust.shared"
diff -u "$BUILD_DIR/cpp.shared" "$BUILD_DIR/rust.shared"

"$BUILD_DIR/cpp-observer" --fixtures \
  "$EMEL_CPP_SOURCE/$LLAMA_RELATIVE" "$EMEL_CPP_SOURCE/$LFM_RELATIVE" \
  >"$BUILD_DIR/cpp.fixtures"
"$RUST_OBSERVER" --fixtures \
  "$EMEL_CPP_SOURCE/$LLAMA_RELATIVE" "$EMEL_CPP_SOURCE/$LFM_RELATIVE" \
  >"$BUILD_DIR/rust.fixtures"
diff -u "$BUILD_DIR/cpp.fixtures" "$BUILD_DIR/rust.fixtures"

tool_sha256="$(sha256_file "$ROOT_DIR/tools/emel-model-catalog-reference/main.cpp")"
{
  cat "$BUILD_DIR/rust.synthetic"
  grep '^reference_case=' "$BUILD_DIR/cpp.synthetic"
  printf 'reference_tool_sha256=%s\n' "$tool_sha256"
  printf 'fixture_llama_lfs_blob=%s\n' "$LLAMA_LFS_BLOB"
  printf 'fixture_llama_bytes=%s\n' "$LLAMA_BYTES"
  printf 'fixture_lfm_lfs_blob=%s\n' "$LFM_LFS_BLOB"
  printf 'fixture_lfm_bytes=%s\n' "$LFM_BYTES"
  printf 'operand_spec=%s\n' "$OPERAND_SPEC"
  cat "$BUILD_DIR/rust.fixtures"
} >"$BUILD_DIR/manifest.txt"

if $UPDATE; then
  mkdir -p "$(dirname "$SNAPSHOT")"
  install -m 0644 "$BUILD_DIR/manifest.txt" "$SNAPSHOT"
fi
if $COMPARE_SNAPSHOT; then
  diff -u "$SNAPSHOT" "$BUILD_DIR/manifest.txt"
fi
echo "Model catalog source and real-fixture parity passed (emel.cpp $SOURCE_COMMIT)"
