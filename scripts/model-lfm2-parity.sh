#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EMEL_CPP_SOURCE="${EMEL_CPP_SOURCE_DIR:-$ROOT_DIR/../emel.cpp}"
SML_SOURCE="${EMEL_STATEFORWARD_SML_SOURCE:-$EMEL_CPP_SOURCE/build/zig/_deps/stateforward_sml-src}"
BUILD_DIR="${EMEL_MODEL_LFM2_PARITY_BUILD_DIR:-$ROOT_DIR/target/model-lfm2-parity}"
SNAPSHOT="$ROOT_DIR/snapshots/parity/model-lfm2/manifest.txt"
INVENTORY="$ROOT_DIR/snapshots/parity/model-lfm2/source-inventory.txt"
OWNER_MAP="$ROOT_DIR/snapshots/parity/model-lfm2/source-owner-map.tsv"
CANONICAL_AST="$ROOT_DIR/tools/emel-model-parity-inventory/snapshots/pinned-ast-coverage.tsv"
SOURCE_COMMIT=843a117386ef17dc5a50549bbfc821074c2141d6
DETAIL_HEADER_BLOB=af0e7d509a3854a847bdde2dad831c5ed60de379
DETAIL_IMPLEMENTATION_BLOB=a6f215e74ed793ea8f606048c00faa80c37056be
LIFECYCLE_TEST_BLOB=27f51f427cb300709369aa7bf63774c93f739b71
INVENTORY_SHA256=248d0b9f13566ba463bc69390704b86aceddd050d2e1eb1088c27e41574aa1f4
OWNER_MAP_SHA256=f77237495442e5e613d23f11c1ee5f6d698ff53e99feb68eab51eff52e126369
LFM2_RELATIVE=tests/models/LFM2.5-230M-Q8_0.gguf
LFM2_SHA256=855be85429300602eda72958547614703541b7d6dd965a8f8f6052b85a7aa935
LFM2_1_2B_RELATIVE=tests/models/LFM2.5-1.2B-Thinking-Q4_K_M.gguf
LFM2_1_2B_SHA256=7223a2202405b02e8e1e6c5baa543c43dc98c1d9741a5c2a0ee1583212e1231b
UPDATE=false
COMPARE=true

[[ ${1:-} == "--update" ]] && UPDATE=true
[[ ${1:-} == "--live" ]] && COMPARE=false
[[ $# -le 1 && (${1:-} == "" || ${1:-} == "--update" || ${1:-} == "--live") ]] || {
  echo "usage: scripts/model-lfm2-parity.sh [--update|--live]" >&2
  exit 2
}

sha256_file() {
  if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | awk '{print $1}'
  else shasum -a 256 "$1" | awk '{print $1}'; fi
}

sha256_text() {
  if command -v sha256sum >/dev/null 2>&1; then sha256sum | awk '{print $1}'
  else shasum -a 256 | awk '{print $1}'; fi
}

[[ "$(git -C "$EMEL_CPP_SOURCE" rev-parse HEAD)" == "$SOURCE_COMMIT" ]]
[[ "$(git -C "$EMEL_CPP_SOURCE" rev-parse "$SOURCE_COMMIT:src/emel/model/lfm2/detail.hpp")" == "$DETAIL_HEADER_BLOB" ]]
[[ "$(git -C "$EMEL_CPP_SOURCE" rev-parse "$SOURCE_COMMIT:src/emel/model/lfm2/detail.cpp")" == "$DETAIL_IMPLEMENTATION_BLOB" ]]
[[ "$(git -C "$EMEL_CPP_SOURCE" rev-parse "$SOURCE_COMMIT:tests/model/loader/lifecycle_tests.cpp")" == "$LIFECYCLE_TEST_BLOB" ]]
[[ "$(sha256_file "$EMEL_CPP_SOURCE/$LFM2_RELATIVE")" == "$LFM2_SHA256" ]]
[[ "$(sha256_file "$EMEL_CPP_SOURCE/$LFM2_1_2B_RELATIVE")" == "$LFM2_1_2B_SHA256" ]]
[[ -f "$INVENTORY" && -f "$OWNER_MAP" && -f "$CANONICAL_AST" ]]
[[ "$(sha256_file "$INVENTORY")" == "$INVENTORY_SHA256" ]]
[[ "$(sha256_file "$OWNER_MAP")" == "$OWNER_MAP_SHA256" ]]
mkdir -p "$BUILD_DIR"

awk -F '\t' '
  NR > 1 && $1 == "src/emel/model/lfm2/detail.hpp" { header[$8]++; header_total++ }
  NR > 1 && $1 == "src/emel/model/lfm2/detail.cpp" { implementation[$8]++; implementation_total++ }
  END {
    exit !(header_total == 10 && header["NamespaceDecl"] == 4 &&
      header["FunctionDecl"] == 6 && implementation_total == 32 &&
      implementation["NamespaceDecl"] == 6 && implementation["VarDecl"] == 18 &&
      implementation["FunctionDecl"] == 8)
  }
' "$CANONICAL_AST"

awk -F '\t' '
  NR == 1 { next }
  ($1 == "src/emel/model/lfm2/detail.hpp" ||
   $1 == "src/emel/model/lfm2/detail.cpp") {
    print $1 "\t" $8 "\t" $9 "\t" $10 "\t" $11
  }
' "$CANONICAL_AST" | LC_ALL=C sort >"$BUILD_DIR/canonical-owner-nodes.tsv"
awk -F '\t' '
  NR == 1 {
    if (!($1 == "path" && $2 == "kind" && $3 == "qualified_name" &&
      $4 == "canonical_signature" && $5 == "node_id" &&
      $6 == "disposition" && $7 == "rust_counterpart" && NF == 7)) exit 1
    next
  }
  $2 != "File" {
    if (NF != 7 || $6 == "" || $7 == "") exit 1
    print $1 "\t" $2 "\t" $3 "\t" $4 "\t" $5
    nodes++
  }
  END { if (nodes != 42) exit 1 }
' "$OWNER_MAP" | LC_ALL=C sort >"$BUILD_DIR/mapped-owner-nodes.tsv"
diff -u "$BUILD_DIR/canonical-owner-nodes.tsv" "$BUILD_DIR/mapped-owner-nodes.tsv"
grep -Fqx $'src/emel/model/lfm2/detail.hpp\tFile\tsrc/emel/model/lfm2/detail.hpp\tblob af0e7d509a3854a847bdde2dad831c5ed60de379\tfile:af0e7d509a3854a847bdde2dad831c5ed60de379\towned_file_complete\tcrates/emel-model/src/lfm2' "$OWNER_MAP"
grep -Fqx $'src/emel/model/lfm2/detail.cpp\tFile\tsrc/emel/model/lfm2/detail.cpp\tblob a6f215e74ed793ea8f606048c00faa80c37056be\tfile:a6f215e74ed793ea8f606048c00faa80c37056be\towned_file_complete\tcrates/emel-model/src/lfm2' "$OWNER_MAP"
[[ "$(awk -F '\t' 'NR > 1 && $2 == "File" { count++ } END { print count + 0 }' "$OWNER_MAP")" == 2 ]]
while IFS= read -r counterpart; do
  [[ -e "$ROOT_DIR/$counterpart" ]]
done < <(awk -F '\t' 'NR > 1 { split($7, path, "::"); print path[1] }' "$OWNER_MAP" | LC_ALL=C sort -u)

rm -rf "$BUILD_DIR/source"
mkdir -p "$BUILD_DIR/source"
git -C "$EMEL_CPP_SOURCE" archive "$SOURCE_COMMIT" include src tests/model/loader/lifecycle_tests.cpp | tar -x -C "$BUILD_DIR/source"

physical_lines="$(wc -l \
  "$BUILD_DIR/source/src/emel/model/lfm2/detail.hpp" \
  "$BUILD_DIR/source/src/emel/model/lfm2/detail.cpp" | awk 'END {print $1}')"
[[ "$physical_lines" == 314 ]]

key_count=0
while IFS= read -r key; do
  grep -RFq "\"$key\"" "$ROOT_DIR/crates/emel-model/src/lfm2"
  key_count=$((key_count + 1))
done < <(grep -o '"lfm2\.[^"]*"' "$BUILD_DIR/source/src/emel/model/lfm2/detail.cpp" | tr -d '"' | sort -u)
[[ "$key_count" == 10 ]]
[[ "$(awk 'index($0, "reject_block_tensor(") { count++ } END { print count + 0 }' "$BUILD_DIR/source/src/emel/model/lfm2/detail.cpp")" == 9 ]]
grep -RFq 'RejectBlockTensors' "$ROOT_DIR/crates/emel-model/src/generation" "$ROOT_DIR/crates/emel-model/src/lfm2"
grep -Fqx 'tensor_policy=reject_opposite_block_family owner=lfm2 status=ported proof=nine_source_calls_explicit_guard_selected_builder_event_and_rollback' "$INVENTORY"

for test_name in \
  model_execution_contract_accepts_canonical_lfm2_hybrid_contract \
  model_lfm2_detail_builds_topology_with_hybrid_tensor_count \
  model_lfm2_detail_describes_hybrid_generation_layers \
  model_execution_contract_rejects_lfm2_without_token_embedding_norm \
  model_execution_contract_rejects_lfm2_with_noncanonical_hybrid_block_tensors \
  model_execution_contract_accepts_lfm2_230m_pattern_layout \
  model_execution_contract_rejects_lfm2_230m_when_pattern_contradicts_block_tensors \
  model_execution_contract_rejects_lfm2_attention_block_with_shortconv_weights \
  model_detail_loads_lfm2_hparams_from_gguf_binding; do
  source_pattern="$test_name"
  [[ "$test_name" == model_execution_contract_rejects_lfm2_with_noncanonical_hybrid_block_tensors ]] && \
    source_pattern=model_execution_contract_rejects_lfm2_with_noncanonical_hybrid_
  [[ "$test_name" == model_execution_contract_rejects_lfm2_230m_when_pattern_contradicts_block_tensors ]] && \
    source_pattern=model_execution_contract_rejects_lfm2_230m_when_pattern_
  [[ "$test_name" == model_execution_contract_rejects_lfm2_attention_block_with_shortconv_weights ]] && \
    source_pattern=model_execution_contract_rejects_lfm2_attention_block_with_
  grep -Fq "$source_pattern" "$BUILD_DIR/source/tests/model/loader/lifecycle_tests.cpp"
  grep -Fq "test=$test_name " "$INVENTORY"
done

proof_token="$(printf '%s\n' "$SOURCE_COMMIT" "$DETAIL_HEADER_BLOB" \
  "$DETAIL_IMPLEMENTATION_BLOB" "$LIFECYCLE_TEST_BLOB" "$LFM2_SHA256" \
  "$LFM2_1_2B_SHA256" | sha256_text)"
grep -Fqx "proof_token_sha256=$proof_token" "$INVENTORY"

linker_gc=(-Wl,--gc-sections)
[[ "$(uname -s)" == Darwin ]] && linker_gc=(-Wl,-dead_strip)
"${CXX:-c++}" -std=c++20 -O2 -ffunction-sections -fdata-sections \
  -I"$BUILD_DIR/source/include" -I"$BUILD_DIR/source/src" -I"$SML_SOURCE/include" \
  "$ROOT_DIR/tools/emel-model-lfm2-reference/main.cpp" \
  "$BUILD_DIR/source/src/emel/model/data.cpp" \
  "$BUILD_DIR/source/src/emel/model/generation/any.cpp" \
  "$BUILD_DIR/source/src/emel/model/lfm2/detail.cpp" \
  "${linker_gc[@]}" -o "$BUILD_DIR/cpp-observer"
cargo build --quiet --locked --manifest-path "$ROOT_DIR/Cargo.toml" \
  -p emel-model --example lfm2_observer
cargo test --quiet --locked --manifest-path "$ROOT_DIR/Cargo.toml" \
  -p emel-model lfm2::tests::loads_source_exact_hparams_and_per_layer_flags_from_typed_gguf_queries

"$BUILD_DIR/cpp-observer" --fixture "$EMEL_CPP_SOURCE/$LFM2_RELATIVE" >"$BUILD_DIR/cpp.230m.fixture"
"$ROOT_DIR/target/debug/examples/lfm2_observer" --fixture "$EMEL_CPP_SOURCE/$LFM2_RELATIVE" >"$BUILD_DIR/rust.230m.fixture"
diff -u "$BUILD_DIR/cpp.230m.fixture" "$BUILD_DIR/rust.230m.fixture"
"$BUILD_DIR/cpp-observer" --fixture "$EMEL_CPP_SOURCE/$LFM2_1_2B_RELATIVE" >"$BUILD_DIR/cpp.1_2b.fixture"
"$ROOT_DIR/target/debug/examples/lfm2_observer" --fixture "$EMEL_CPP_SOURCE/$LFM2_1_2B_RELATIVE" >"$BUILD_DIR/rust.1_2b.fixture"
diff -u "$BUILD_DIR/cpp.1_2b.fixture" "$BUILD_DIR/rust.1_2b.fixture"
"$BUILD_DIR/cpp-observer" --validation >"$BUILD_DIR/cpp.validation"
"$ROOT_DIR/target/debug/examples/lfm2_observer" --validation >"$BUILD_DIR/rust.validation"
diff -u "$BUILD_DIR/cpp.validation" "$BUILD_DIR/rust.validation"

{
  cat "$BUILD_DIR/rust.230m.fixture"
  sed -n '5,$p' "$BUILD_DIR/rust.1_2b.fixture"
  cat "$BUILD_DIR/rust.validation"
  printf 'fixture_lfm2_230m_sha256=%s\n' "$LFM2_SHA256"
  printf 'fixture_lfm2_1_2b_sha256=%s\n' "$LFM2_1_2B_SHA256"
  printf 'proof_token_sha256=%s\n' "$proof_token"
  printf 'source_inventory_sha256=%s\n' "$(sha256_file "$INVENTORY")"
  printf 'source_owner_map_sha256=%s\n' "$(sha256_file "$OWNER_MAP")"
  printf 'canonical_ast_inventory_sha256=%s\n' "$(sha256_file "$CANONICAL_AST")"
  printf 'reference_tool_sha256=%s\n' "$(sha256_file "$ROOT_DIR/tools/emel-model-lfm2-reference/main.cpp")"
} >"$BUILD_DIR/manifest.txt"

if $UPDATE; then
  mkdir -p "$(dirname "$SNAPSHOT")"
  install -m 0644 "$BUILD_DIR/manifest.txt" "$SNAPSHOT"
fi
$COMPARE && diff -u "$SNAPSHOT" "$BUILD_DIR/manifest.txt"
echo "Model LFM2 source and real-fixture parity passed (emel.cpp $SOURCE_COMMIT)"
