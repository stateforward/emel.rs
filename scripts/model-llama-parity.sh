#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EMEL_CPP_SOURCE="${EMEL_CPP_SOURCE_DIR:-$ROOT_DIR/../emel.cpp}"
SML_SOURCE="${EMEL_STATEFORWARD_SML_SOURCE:-$EMEL_CPP_SOURCE/build/zig/_deps/stateforward_sml-src}"
BUILD_DIR="${EMEL_MODEL_LLAMA_PARITY_BUILD_DIR:-$ROOT_DIR/target/model-llama-parity}"
SNAPSHOT="$ROOT_DIR/snapshots/parity/model-llama/manifest.txt"
INVENTORY="$ROOT_DIR/snapshots/parity/model-llama/source-inventory.txt"
CANONICAL_AST="$ROOT_DIR/tools/emel-model-parity-inventory/snapshots/pinned-ast-coverage.tsv"
SOURCE_COMMIT=843a117386ef17dc5a50549bbfc821074c2141d6
ANY_BLOB=430ba6a8897854b5f58660d3f4d51d006727559e
DETAIL_HEADER_BLOB=91f96245d29a22aa83c570cb4fe8bc8cf9e1d081
DETAIL_IMPLEMENTATION_BLOB=c78e3656d41772dcc4dba5c46623d9f2fec784ed
LLAMA_RELATIVE=tests/models/Llama-68M-Chat-v1-Q2_K.gguf
LLAMA_SHA256=8ed06dc5bd84bce3154a2b7e751c45a56562691933ee25b5823393f909329a67
UPDATE=false
COMPARE=true

[[ ${1:-} == "--update" ]] && UPDATE=true
[[ ${1:-} == "--live" ]] && COMPARE=false
[[ $# -le 1 && (${1:-} == "" || ${1:-} == "--update" || ${1:-} == "--live") ]] || {
  echo "usage: scripts/model-llama-parity.sh [--update|--live]" >&2
  exit 2
}

sha256_file() {
  if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | awk '{print $1}'
  else shasum -a 256 "$1" | awk '{print $1}'; fi
}

[[ "$(git -C "$EMEL_CPP_SOURCE" rev-parse HEAD)" == "$SOURCE_COMMIT" ]]
[[ "$(git -C "$EMEL_CPP_SOURCE" rev-parse "$SOURCE_COMMIT:src/emel/model/llama/any.hpp")" == "$ANY_BLOB" ]]
[[ "$(git -C "$EMEL_CPP_SOURCE" rev-parse "$SOURCE_COMMIT:src/emel/model/llama/detail.hpp")" == "$DETAIL_HEADER_BLOB" ]]
[[ "$(git -C "$EMEL_CPP_SOURCE" rev-parse "$SOURCE_COMMIT:src/emel/model/llama/detail.cpp")" == "$DETAIL_IMPLEMENTATION_BLOB" ]]
[[ "$(sha256_file "$EMEL_CPP_SOURCE/$LLAMA_RELATIVE")" == "$LLAMA_SHA256" ]]
[[ -f "$INVENTORY" && -f "$CANONICAL_AST" ]]

awk -F '\t' '
  NR > 1 && $1 == "src/emel/model/llama/any.hpp" { any[$8]++; any_total++ }
  NR > 1 && $1 == "src/emel/model/llama/detail.hpp" { header[$8]++; header_total++ }
  NR > 1 && $1 == "src/emel/model/llama/detail.cpp" { implementation[$8]++; implementation_total++ }
  END {
    exit !(any_total == 15 && any["NamespaceDecl"] == 3 &&
      any["TypeAliasDecl"] == 6 && any["FunctionDecl"] == 6 &&
      header_total == 42 && header["NamespaceDecl"] == 4 &&
      header["VarDecl"] == 3 && header["TypeAliasDecl"] == 17 &&
      header["UsingDecl"] == 12 && header["FunctionDecl"] == 6 &&
      implementation_total == 13 && implementation["NamespaceDecl"] == 5 &&
      implementation["VarDecl"] == 2 && implementation["FunctionDecl"] == 6)
  }
' "$CANONICAL_AST"

rm -rf "$BUILD_DIR/source"
mkdir -p "$BUILD_DIR/source"
git -C "$EMEL_CPP_SOURCE" archive "$SOURCE_COMMIT" include src tests/model/loader/lifecycle_tests.cpp | tar -x -C "$BUILD_DIR/source"

physical_lines="$(wc -l \
  "$BUILD_DIR/source/src/emel/model/llama/any.hpp" \
  "$BUILD_DIR/source/src/emel/model/llama/detail.hpp" \
  "$BUILD_DIR/source/src/emel/model/llama/detail.cpp" | awk 'END {print $1}')"
[[ "$physical_lines" == 298 ]]

key_count=0
while IFS= read -r key; do
  grep -RFq "\"$key\"" "$ROOT_DIR/crates/emel-model/src/llama"
  key_count=$((key_count + 1))
done < <(grep -o '"llama\.[^"]*"' "$BUILD_DIR/source/src/emel/model/llama/detail.cpp" | tr -d '"')
[[ "$key_count" == 18 ]]
[[ "$(awk 'index($0, "reject_block_tensor(") { count++ } END { print count + 0 }' "$BUILD_DIR/source/src/emel/model/llama/detail.cpp")" == 0 ]]
grep -Fq 'using emel::model::generation::reject_block_tensor;' "$BUILD_DIR/source/src/emel/model/llama/detail.hpp"
grep -Fqx 'future_family_guard_boundary=reject_block_tensor llama_disposition=imported_not_executed status=remains_open_for_executing_family proof=zero_calls_in_llama_detail_cpp' "$INVENTORY"

for test_name in \
  model_llama_detail_builds_execution_view_for_canonical_tensor_set \
  model_llama_detail_builds_execution_view_without_contiguous_weights_blob \
  model_llama_detail_rejects_missing_required_tensor \
  model_execution_contract_accepts_canonical_llama_contract \
  model_detail_loads_llama_hparams_from_gguf_binding \
  model_llama_detail_quantized_audit_name_helpers_publish_supported_labels \
  model_detail_loads_gguf_vocab_metadata_without_llama_bootstrap; do
  source_pattern="$test_name"
  [[ "$test_name" == model_llama_detail_builds_execution_view_without_contiguous_weights_blob ]] && \
    source_pattern=model_llama_detail_builds_execution_view_without_contiguous_weights_
  [[ "$test_name" == model_llama_detail_quantized_audit_name_helpers_publish_supported_labels ]] && \
    source_pattern=model_llama_detail_quantized_audit_name_helpers_publish_supported_
  grep -Fq "$source_pattern" "$BUILD_DIR/source/tests/model/loader/lifecycle_tests.cpp"
  grep -Fq "test=$test_name " "$INVENTORY"
done

linker_gc=(-Wl,--gc-sections)
[[ "$(uname -s)" == Darwin ]] && linker_gc=(-Wl,-dead_strip)
"${CXX:-c++}" -std=c++20 -O2 -ffunction-sections -fdata-sections \
  -I"$BUILD_DIR/source/include" -I"$BUILD_DIR/source/src" -I"$SML_SOURCE/include" \
  "$ROOT_DIR/tools/emel-model-llama-reference/main.cpp" \
  "$BUILD_DIR/source/src/emel/model/data.cpp" \
  "$BUILD_DIR/source/src/emel/model/generation/any.cpp" \
  "$BUILD_DIR/source/src/emel/model/llama/detail.cpp" \
  "${linker_gc[@]}" -o "$BUILD_DIR/cpp-observer"
cargo build --quiet --locked --manifest-path "$ROOT_DIR/Cargo.toml" \
  -p emel-model --example llama_observer
cargo test --quiet --locked --manifest-path "$ROOT_DIR/Cargo.toml" \
  -p emel-model llama::tests::loads_source_exact_hparam_values_from_typed_gguf_queries

"$BUILD_DIR/cpp-observer" --fixture "$EMEL_CPP_SOURCE/$LLAMA_RELATIVE" >"$BUILD_DIR/cpp.fixture"
"$ROOT_DIR/target/debug/examples/llama_observer" --fixture "$EMEL_CPP_SOURCE/$LLAMA_RELATIVE" >"$BUILD_DIR/rust.fixture"
diff -u "$BUILD_DIR/cpp.fixture" "$BUILD_DIR/rust.fixture"

{
  cat "$BUILD_DIR/rust.fixture"
  printf 'fixture_llama_sha256=%s\n' "$LLAMA_SHA256"
  printf 'source_inventory_sha256=%s\n' "$(sha256_file "$INVENTORY")"
  printf 'canonical_ast_inventory_sha256=%s\n' "$(sha256_file "$CANONICAL_AST")"
  printf 'reference_tool_sha256=%s\n' "$(sha256_file "$ROOT_DIR/tools/emel-model-llama-reference/main.cpp")"
} >"$BUILD_DIR/manifest.txt"

if $UPDATE; then
  mkdir -p "$(dirname "$SNAPSHOT")"
  install -m 0644 "$BUILD_DIR/manifest.txt" "$SNAPSHOT"
fi
$COMPARE && diff -u "$SNAPSHOT" "$BUILD_DIR/manifest.txt"
echo "Model Llama source and real-fixture parity passed (emel.cpp $SOURCE_COMMIT)"
