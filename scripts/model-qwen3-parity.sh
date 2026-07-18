#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EMEL_CPP_SOURCE="${EMEL_CPP_SOURCE_DIR:-$ROOT_DIR/../emel.cpp}"
SML_SOURCE="${EMEL_STATEFORWARD_SML_SOURCE:-$EMEL_CPP_SOURCE/build/zig/_deps/stateforward_sml-src}"
BUILD_DIR="${EMEL_MODEL_QWEN3_PARITY_BUILD_DIR:-$ROOT_DIR/target/model-qwen3-parity}"
SNAPSHOT="$ROOT_DIR/snapshots/parity/model-qwen3/manifest.txt"
INVENTORY="$ROOT_DIR/snapshots/parity/model-qwen3/source-inventory.txt"
OWNER_MAP="$ROOT_DIR/snapshots/parity/model-qwen3/source-owner-map.tsv"
CANONICAL_AST="$ROOT_DIR/tools/emel-model-parity-inventory/snapshots/pinned-ast-coverage.tsv"
SOURCE_COMMIT=843a117386ef17dc5a50549bbfc821074c2141d6
DETAIL_HEADER_BLOB=6e31b097184042bfe1ec9486af060214f1d01b2b
DETAIL_IMPLEMENTATION_BLOB=a43d8fee3ed962d84447360e7601c999468bf6f8
LIFECYCLE_TEST_BLOB=27f51f427cb300709369aa7bf63774c93f739b71
INVENTORY_SHA256=73d6c0145cc1491ce2eb2066559dc8d43f0b057f5c7b1743c3dee8630f1f5046
OWNER_MAP_SHA256=928fdf228683410ab924d8df09dc9cea7ec34ffa4c05d28490e0dc1d1583dce5
CATALOG_REFERENCE_TOOL_SHA256=7a647756db55e6279a3869e4a87ff2f582e8f59db02bb5f4666c35edd5a11aaf
QWEN3_RELATIVE=tests/models/Qwen3-0.6B-Q8_0.gguf
QWEN3_SHA256=9465e63a22add5354d9bb4b99e90117043c7124007664907259bd16d043bb031
UPDATE=false
COMPARE=true

[[ ${1:-} == "--update" ]] && UPDATE=true
[[ ${1:-} == "--live" ]] && COMPARE=false
[[ $# -le 1 && (${1:-} == "" || ${1:-} == "--update" || ${1:-} == "--live") ]] || {
  echo "usage: scripts/model-qwen3-parity.sh [--update|--live]" >&2
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

rust_counterpart_exists() {
  local counterpart="$1"
  local path="${counterpart%%::*}"
  [[ -e "$ROOT_DIR/$path" ]] || return 1
  [[ "$counterpart" == *::* ]] || return 0
  local symbol="${counterpart#*::}"
  [[ "$symbol" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]] || return 1
  grep -Eq "^[[:space:]]*((pub|pub\\([^)]*\\))[[:space:]]+)?(const|fn|struct|enum|trait|type)[[:space:]]+${symbol}([^A-Za-z0-9_]|$)|^[[:space:]]*${symbol}([<{]|$)" \
    "$ROOT_DIR/$path"
}

[[ "$(git -C "$EMEL_CPP_SOURCE" rev-parse HEAD)" == "$SOURCE_COMMIT" ]]
[[ "$(git -C "$EMEL_CPP_SOURCE" rev-parse "$SOURCE_COMMIT:src/emel/model/qwen3/detail.hpp")" == "$DETAIL_HEADER_BLOB" ]]
[[ "$(git -C "$EMEL_CPP_SOURCE" rev-parse "$SOURCE_COMMIT:src/emel/model/qwen3/detail.cpp")" == "$DETAIL_IMPLEMENTATION_BLOB" ]]
[[ "$(git -C "$EMEL_CPP_SOURCE" rev-parse "$SOURCE_COMMIT:tests/model/loader/lifecycle_tests.cpp")" == "$LIFECYCLE_TEST_BLOB" ]]
[[ "$(sha256_file "$EMEL_CPP_SOURCE/$QWEN3_RELATIVE")" == "$QWEN3_SHA256" ]]
[[ -f "$INVENTORY" && -f "$OWNER_MAP" && -f "$CANONICAL_AST" ]]
[[ "$(sha256_file "$INVENTORY")" == "$INVENTORY_SHA256" ]]
[[ "$(sha256_file "$OWNER_MAP")" == "$OWNER_MAP_SHA256" ]]
[[ "$(sha256_file "$ROOT_DIR/tools/emel-model-catalog-reference/main.cpp")" == \
  "$CATALOG_REFERENCE_TOOL_SHA256" ]]
mkdir -p "$BUILD_DIR"

awk -F '\t' '
  NR > 1 && $1 == "src/emel/model/qwen3/detail.hpp" { header[$8]++; header_total++ }
  NR > 1 && $1 == "src/emel/model/qwen3/detail.cpp" { implementation[$8]++; implementation_total++ }
  END {
    exit !(header_total == 7 && header["NamespaceDecl"] == 4 &&
      header["FunctionDecl"] == 3 && implementation_total == 12 &&
      implementation["NamespaceDecl"] == 5 && implementation["VarDecl"] == 4 &&
      implementation["FunctionDecl"] == 3)
  }
' "$CANONICAL_AST"

awk -F '\t' '
  NR == 1 { next }
  ($1 == "src/emel/model/qwen3/detail.hpp" ||
   $1 == "src/emel/model/qwen3/detail.cpp") {
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
  $2 == "NamespaceDecl" || $2 == "VarDecl" || $2 == "FunctionDecl" {
    if (NF != 7 || $6 == "" || $7 == "") exit 1
    print $1 "\t" $2 "\t" $3 "\t" $4 "\t" $5
    nodes++
  }
  END { if (nodes != 19) exit 1 }
' "$OWNER_MAP" | LC_ALL=C sort >"$BUILD_DIR/mapped-owner-nodes.tsv"
diff -u "$BUILD_DIR/canonical-owner-nodes.tsv" "$BUILD_DIR/mapped-owner-nodes.tsv"
grep -Fqx $'src/emel/model/qwen3/detail.hpp\tFile\tsrc/emel/model/qwen3/detail.hpp\tblob 6e31b097184042bfe1ec9486af060214f1d01b2b\tfile:6e31b097184042bfe1ec9486af060214f1d01b2b\towned_file_complete\tcrates/emel-model/src/qwen3' "$OWNER_MAP"
grep -Fqx $'src/emel/model/qwen3/detail.cpp\tFile\tsrc/emel/model/qwen3/detail.cpp\tblob a43d8fee3ed962d84447360e7601c999468bf6f8\tfile:a43d8fee3ed962d84447360e7601c999468bf6f8\towned_file_complete\tcrates/emel-model/src/qwen3' "$OWNER_MAP"
[[ "$(awk -F '\t' 'NR > 1 && $2 == "File" { count++ } END { print count + 0 }' "$OWNER_MAP")" == 2 ]]
[[ "$(awk -F '\t' 'NR > 1 && $2 == "TestCase" { count++ } END { print count + 0 }' "$OWNER_MAP")" == 7 ]]
[[ "$(awk -F '\t' 'NR > 1 && $2 == "TestHelper" { count++ } END { print count + 0 }' "$OWNER_MAP")" == 1 ]]
[[ "$(awk 'END { print NR - 1 }' "$OWNER_MAP")" == 29 ]]
while IFS= read -r counterpart; do
  rust_counterpart_exists "$counterpart"
done < <(awk -F '\t' 'NR > 1 { print $7 }' "$OWNER_MAP" | LC_ALL=C sort -u)
! rust_counterpart_exists \
  "crates/emel-model/src/qwen3/tests.rs::__emel_missing_counterpart_probe"

rm -rf "$BUILD_DIR/source"
mkdir -p "$BUILD_DIR/source"
git -C "$EMEL_CPP_SOURCE" archive "$SOURCE_COMMIT" include src tests/model/loader/lifecycle_tests.cpp | tar -x -C "$BUILD_DIR/source"

physical_lines="$(wc -l \
  "$BUILD_DIR/source/src/emel/model/qwen3/detail.hpp" \
  "$BUILD_DIR/source/src/emel/model/qwen3/detail.cpp" | awk 'END {print $1}')"
[[ "$physical_lines" == 146 ]]

key_count=0
while IFS= read -r key; do
  grep -RFq "\"$key\"" "$ROOT_DIR/crates/emel-model/src/qwen3"
  key_count=$((key_count + 1))
done < <(grep -o '"qwen3\.[^"]*"' "$BUILD_DIR/source/src/emel/model/qwen3/detail.cpp" | tr -d '"')
[[ "$key_count" == 10 ]]

perl -0777 -ne '
  while (/TEST_CASE\s*\((.*?)\)\s*\{/sg) {
    $arguments = $1;
    $name = "";
    $name .= $1 while $arguments =~ /"([^"]*)"/g;
    print "$name\n" if $name =~ /qwen3/ ||
      $name eq "model_generation_build_contract_resolves_current_architecture";
  }
' "$BUILD_DIR/source/tests/model/loader/lifecycle_tests.cpp" | LC_ALL=C sort >"$BUILD_DIR/source-tests.txt"
awk -F '\t' 'NR > 1 && $2 == "TestCase" { print $3 }' "$OWNER_MAP" |
  LC_ALL=C sort >"$BUILD_DIR/mapped-tests.txt"
diff -u "$BUILD_DIR/source-tests.txt" "$BUILD_DIR/mapped-tests.txt"
grep -Fq 'void build_qwen3_model(emel::model::data &model' \
  "$BUILD_DIR/source/tests/model/loader/lifecycle_tests.cpp"
grep -Fqx 'build_qwen3_model' <(awk -F '\t' 'NR > 1 && $2 == "TestHelper" { print $3 }' "$OWNER_MAP")
while IFS=$'\t' read -r name node_id; do
  expected="$(printf 'tests/model/loader/lifecycle_tests.cpp::%s' "$name" | sha256_text)"
  [[ "$node_id" == "$expected" ]]
done < <(awk -F '\t' 'NR > 1 && ($2 == "TestCase" || $2 == "TestHelper") { print $3 "\t" $5 }' "$OWNER_MAP")
while IFS= read -r test_name; do
  grep -Fq "test=$test_name " "$INVENTORY"
done <"$BUILD_DIR/source-tests.txt"

proof_token="$(printf '%s\n' "$SOURCE_COMMIT" "$DETAIL_HEADER_BLOB" \
  "$DETAIL_IMPLEMENTATION_BLOB" "$LIFECYCLE_TEST_BLOB" "$QWEN3_SHA256" \
  "$CATALOG_REFERENCE_TOOL_SHA256" | sha256_text)"
grep -Fqx "proof_token_sha256=$proof_token" "$INVENTORY"

linker_gc=(-Wl,--gc-sections)
[[ "$(uname -s)" == Darwin ]] && linker_gc=(-Wl,-dead_strip)
"${CXX:-c++}" -std=c++20 -O2 -ffunction-sections -fdata-sections \
  -I"$BUILD_DIR/source/include" -I"$BUILD_DIR/source/src" -I"$SML_SOURCE/include" \
  "$ROOT_DIR/tools/emel-model-qwen3-reference/main.cpp" \
  "$BUILD_DIR/source/src/emel/model/data.cpp" \
  "$BUILD_DIR/source/src/emel/model/generation/any.cpp" \
  "$BUILD_DIR/source/src/emel/model/qwen3/detail.cpp" \
  "${linker_gc[@]}" -o "$BUILD_DIR/cpp-observer"
cargo build --quiet --locked --manifest-path "$ROOT_DIR/Cargo.toml" \
  -p emel-model --example qwen3_observer
cargo test --quiet --locked --manifest-path "$ROOT_DIR/Cargo.toml" \
  -p emel-model qwen3::tests::loads_source_exact_hparams_and_derived_layout_from_typed_gguf_queries

"$BUILD_DIR/cpp-observer" --fixture "$EMEL_CPP_SOURCE/$QWEN3_RELATIVE" >"$BUILD_DIR/cpp.fixture"
"$ROOT_DIR/target/debug/examples/qwen3_observer" --fixture "$EMEL_CPP_SOURCE/$QWEN3_RELATIVE" >"$BUILD_DIR/rust.fixture"
diff -u "$BUILD_DIR/cpp.fixture" "$BUILD_DIR/rust.fixture"

{
  cat "$BUILD_DIR/rust.fixture"
  printf 'fixture_qwen3_sha256=%s\n' "$QWEN3_SHA256"
  printf 'proof_token_sha256=%s\n' "$proof_token"
  printf 'source_inventory_sha256=%s\n' "$(sha256_file "$INVENTORY")"
  printf 'source_owner_map_sha256=%s\n' "$(sha256_file "$OWNER_MAP")"
  printf 'canonical_ast_inventory_sha256=%s\n' "$(sha256_file "$CANONICAL_AST")"
  printf 'reference_tool_sha256=%s\n' "$(sha256_file "$ROOT_DIR/tools/emel-model-qwen3-reference/main.cpp")"
  printf 'reference_catalog_helper_sha256=%s\n' "$CATALOG_REFERENCE_TOOL_SHA256"
} >"$BUILD_DIR/manifest.txt"

if $UPDATE; then
  mkdir -p "$(dirname "$SNAPSHOT")"
  install -m 0644 "$BUILD_DIR/manifest.txt" "$SNAPSHOT"
fi
$COMPARE && diff -u "$SNAPSHOT" "$BUILD_DIR/manifest.txt"
echo "Model Qwen3 source and real-fixture parity passed (emel.cpp $SOURCE_COMMIT)"
