#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EMEL_CPP_SOURCE="${EMEL_CPP_SOURCE_DIR:-$ROOT_DIR/../emel.cpp}"
SML_SOURCE="${EMEL_STATEFORWARD_SML_SOURCE:-$EMEL_CPP_SOURCE/build/zig/_deps/stateforward_sml-src}"
BUILD_DIR="${EMEL_MODEL_GEMMA4_PARITY_BUILD_DIR:-$ROOT_DIR/target/model-gemma4-parity}"
SNAPSHOT="$ROOT_DIR/snapshots/parity/model-gemma4/manifest.txt"
INVENTORY="$ROOT_DIR/snapshots/parity/model-gemma4/source-inventory.txt"
OWNER_MAP="$ROOT_DIR/snapshots/parity/model-gemma4/source-owner-map.tsv"
CANONICAL_AST="$ROOT_DIR/tools/emel-model-parity-inventory/snapshots/pinned-ast-coverage.tsv"
SOURCE_COMMIT=843a117386ef17dc5a50549bbfc821074c2141d6
DETAIL_HEADER_BLOB=5ee94dcc8554ef8e9de0a51007acae6bb58565c9
DETAIL_IMPLEMENTATION_BLOB=ac9cc9757a5d6e97ebfbd1ef905d01f4af34dee4
LIFECYCLE_TEST_BLOB=27f51f427cb300709369aa7bf63774c93f739b71
INVENTORY_SHA256=e0da02de426d5330ef57eb22d3dee137ce1c7827861a1b862a1b51fb5bdfc9e1
OWNER_MAP_SHA256=78ea286a1b80a55d735963700c0c8c6437cfa955345a13990045724ea1aab2ab
REFERENCE_TOOL_SHA256=0b45c86bc9c2758197320b255e2c8ff0de959e48fb101ef21f08345049e16710
CATALOG_REFERENCE_TOOL_SHA256=7a647756db55e6279a3869e4a87ff2f582e8f59db02bb5f4666c35edd5a11aaf
UPDATE=false

[[ ${1:-} == "--update" ]] && UPDATE=true
[[ $# -le 1 && (${1:-} == "" || ${1:-} == "--update" || ${1:-} == "--live") ]] || {
  echo "usage: scripts/model-gemma4-parity.sh [--update|--live]" >&2
  exit 2
}

echo "Validating the pinned Gemma4 source inventory"

sha256_file() {
  if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | awk '{print $1}'
  else shasum -a 256 "$1" | awk '{print $1}'; fi
}
sha256_text() {
  if command -v sha256sum >/dev/null 2>&1; then sha256sum | awk '{print $1}'
  else shasum -a 256 | awk '{print $1}'; fi
}
fail() {
  echo "model-gemma4 parity: $1" >&2
  exit 1
}
require_equal() {
  local actual="$1"
  local expected="$2"
  local label="$3"
  if [[ "$actual" != "$expected" ]]; then
    fail "$label mismatch: expected $expected, got $actual"
  fi
}
rust_counterpart_exists() {
  local counterpart="$1"
  local path="${counterpart%%::*}"
  [[ -e "$ROOT_DIR/$path" ]] || return 1
  [[ "$counterpart" == *::* ]] || return 0
  local symbol="${counterpart#*::}"
  [[ "$symbol" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]] || return 1
  grep -Eq "^[[:space:]]*((pub|pub\\([^)]*\\))[[:space:]]+)?(((const|async|unsafe)[[:space:]]+)*fn|const|struct|enum|trait|type)[[:space:]]+${symbol}([^A-Za-z0-9_]|$)|^[[:space:]]*${symbol}([<{]|$)" "$ROOT_DIR/$path"
}
require_function_mapping() {
  local symbol="$1"
  local expected="$2"
  local actual
  actual="$(awk -F '\t' -v needle="::$symbol(" '
    NR > 1 && $2 == "FunctionDecl" && index($3, needle) { print $7 }
  ' "$OWNER_MAP" | LC_ALL=C sort -u)"
  require_equal "$actual" "$expected" "owner mapping for $symbol"
}
require_transition() {
  local row="$1"
  local file="$2"
  grep -Fq "$row" "$ROOT_DIR/$file" || fail "missing semantic transition: $row"
}

require_equal "$(git -C "$EMEL_CPP_SOURCE" rev-parse HEAD)" "$SOURCE_COMMIT" "source commit"
require_equal "$(git -C "$EMEL_CPP_SOURCE" rev-parse "$SOURCE_COMMIT:src/emel/model/gemma4/detail.hpp")" "$DETAIL_HEADER_BLOB" "detail.hpp blob"
require_equal "$(git -C "$EMEL_CPP_SOURCE" rev-parse "$SOURCE_COMMIT:src/emel/model/gemma4/detail.cpp")" "$DETAIL_IMPLEMENTATION_BLOB" "detail.cpp blob"
require_equal "$(git -C "$EMEL_CPP_SOURCE" rev-parse "$SOURCE_COMMIT:tests/model/loader/lifecycle_tests.cpp")" "$LIFECYCLE_TEST_BLOB" "lifecycle tests blob"
require_equal "$(sha256_file "$INVENTORY")" "$INVENTORY_SHA256" "source inventory hash"
require_equal "$(sha256_file "$OWNER_MAP")" "$OWNER_MAP_SHA256" "source owner-map hash"
require_equal "$(sha256_file "$ROOT_DIR/tools/emel-model-gemma4-reference/main.cpp")" "$REFERENCE_TOOL_SHA256" "reference tool hash"
require_equal "$(sha256_file "$ROOT_DIR/tools/emel-model-catalog-reference/main.cpp")" "$CATALOG_REFERENCE_TOOL_SHA256" "catalog helper hash"
mkdir -p "$BUILD_DIR"

awk -F '\t' '
  NR > 1 && ($1 == "src/emel/model/gemma4/detail.hpp" ||
             $1 == "src/emel/model/gemma4/detail.cpp") {
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
  END { if (nodes != 52) exit 1 }
' "$OWNER_MAP" | LC_ALL=C sort >"$BUILD_DIR/mapped-owner-nodes.tsv"
diff -u "$BUILD_DIR/canonical-owner-nodes.tsv" "$BUILD_DIR/mapped-owner-nodes.tsv"
require_equal "$(awk -F '\t' 'NR > 1 && $2 == "File" { count++ } END { print count + 0 }' "$OWNER_MAP")" 2 "owned file count"
require_equal "$(awk -F '\t' 'NR > 1 && $2 == "TestCase" { count++ } END { print count + 0 }' "$OWNER_MAP")" 9 "test case count"
require_equal "$(awk -F '\t' 'NR > 1 && $2 == "TestHelper" { count++ } END { print count + 0 }' "$OWNER_MAP")" 1 "test helper count"
require_equal "$(awk 'END { print NR - 1 }' "$OWNER_MAP")" 64 "owner-map row count"
if ! awk -F '\t' '
  NR > 1 && $2 == "FunctionDecl" && $7 !~ /^crates\/emel-model\/src\/gemma4\// { exit 1 }
' "$OWNER_MAP"; then
  fail "every Gemma4 function must map to its owning component"
fi
require_function_mapping is_shared_kv_layer crates/emel-model/src/gemma4/actor.rs::is_fixed_shared_kv_layer
require_function_mapping is_bound_shared_kv_layer crates/emel-model/src/gemma4/actor.rs::block_is_bound_shared
require_function_mapping is_execution_architecture crates/emel-model/src/gemma4/actor.rs::is_execution_architecture
require_function_mapping load_hparams crates/emel-model/src/gemma4/hparams.rs::load_hparams
VALIDATE_CONTRACT_OWNERS='crates/emel-model/src/gemma4/actor.rs::block_requires_dedicated_value;crates/emel-model/src/gemma4/actor.rs::guard_block_shared_full_required_value;crates/emel-model/src/generation/actor.rs::guard_attention_qk_shared_required_value'
BUILD_CONTRACT_OWNERS='crates/emel-model/src/gemma4/actor.rs::block_is_bound_shared;crates/emel-model/src/gemma4/actor.rs::guard_block_dedicated_full;crates/emel-model/src/gemma4/actor.rs::guard_block_shared_full;crates/emel-model/src/gemma4/actor.rs::guard_block_sliding_key_swa;crates/emel-model/src/gemma4/actor.rs::effect_block_dedicated_full_child;crates/emel-model/src/gemma4/actor.rs::effect_block_shared_full_child;crates/emel-model/src/gemma4/actor.rs::effect_block_selected_dedicated_child;crates/emel-model/src/gemma4/actor.rs::effect_block_selected_shared_child;crates/emel-model/src/gemma4/actor.rs::effect_topology_shared_child;crates/emel-model/src/attention_family/context.rs::effect_global_child'
require_function_mapping validate_contract "$VALIDATE_CONTRACT_OWNERS"
require_function_mapping validate_builder_contract crates/emel-model/src/gemma4/actor.rs::guard_begin_profile_validation_valid
require_function_mapping validate_data crates/emel-model/src/gemma4/actor.rs::guard_begin_profile_parameters_invalid
require_function_mapping validate_execution_contract crates/emel-model/src/gemma4/actor.rs::guard_begin_profile_execution_valid
require_function_mapping build_generation_contract "$BUILD_CONTRACT_OWNERS"
require_transition '"state_begin_child"_s <= *"state_bound"_s + Begin(BeginRuntime<' crates/emel-model/src/gemma4/sm.rs
require_transition '[guard_begin_profile_execution_valid] / effect_begin_profile_child,' crates/emel-model/src/gemma4/sm.rs
require_transition '[guard_begin_profile_validation_valid] / effect_begin_profile_child,' crates/emel-model/src/gemma4/sm.rs
require_transition '[guard_begin_profile_parameters_invalid] / effect_begin_model_invalid,' crates/emel-model/src/gemma4/sm.rs
require_transition '[guard_block_shared_full_required_value] / effect_block_shared_full_required_value_child,' crates/emel-model/src/gemma4/sm.rs
require_transition '[guard_block_selected_shared_required_value] / effect_block_selected_shared_required_value_child,' crates/emel-model/src/gemma4/sm.rs
require_transition '[guard_attention_qk_shared_required_value] / effect_query_attention_qk_shared_required_value,' crates/emel-model/src/generation/sm.rs
while IFS= read -r mapping; do
  IFS=';' read -r -a counterparts <<<"$mapping"
  for counterpart in "${counterparts[@]}"; do
    if ! rust_counterpart_exists "$counterpart"; then
      fail "missing exact Rust counterpart: $counterpart"
    fi
  done
done < <(awk -F '\t' 'NR > 1 { print $7 }' "$OWNER_MAP" | LC_ALL=C sort -u)
echo "Validated all Gemma4 source-owner counterparts"
if rust_counterpart_exists "crates/emel-model/src/gemma4/tests.rs::__missing_counterpart_probe"; then
  fail "missing-counterpart negative probe unexpectedly resolved"
fi

rm -rf "$BUILD_DIR/source"
mkdir -p "$BUILD_DIR/source"
git -C "$EMEL_CPP_SOURCE" archive "$SOURCE_COMMIT" include src tests/model/loader/lifecycle_tests.cpp | tar -x -C "$BUILD_DIR/source"
physical_lines="$(wc -l "$BUILD_DIR/source/src/emel/model/gemma4/detail.hpp" "$BUILD_DIR/source/src/emel/model/gemma4/detail.cpp" | awk 'END {print $1}')"
require_equal "$physical_lines" 357 "pinned physical line count"
key_count="$(grep -o '"gemma4\.[^"]*"' "$BUILD_DIR/source/src/emel/model/gemma4/detail.cpp" | sort -u | wc -l | tr -d ' ')"
require_equal "$key_count" 21 "Gemma4 metadata key count"

perl -0777 -ne '
  while (/TEST_CASE\s*\((.*?)\)\s*\{/sg) {
    $arguments = $1; $name = "";
    $name .= $1 while $arguments =~ /"([^"]*)"/g;
    print "$name\n" if $name =~ /gemma4/;
  }
' "$BUILD_DIR/source/tests/model/loader/lifecycle_tests.cpp" | LC_ALL=C sort >"$BUILD_DIR/source-tests.txt"
awk -F '\t' 'NR > 1 && $2 == "TestCase" { print $3 }' "$OWNER_MAP" | LC_ALL=C sort >"$BUILD_DIR/mapped-tests.txt"
diff -u "$BUILD_DIR/source-tests.txt" "$BUILD_DIR/mapped-tests.txt"
grep -Fq 'void build_gemma4_model(emel::model::data &model' "$BUILD_DIR/source/tests/model/loader/lifecycle_tests.cpp"
grep -Fqx 'build_gemma4_model' <(awk -F '\t' 'NR > 1 && $2 == "TestHelper" { print $3 }' "$OWNER_MAP")
while IFS=$'\t' read -r name node_id; do
  expected="$(printf 'tests/model/loader/lifecycle_tests.cpp::%s' "$name" | sha256_text)"
  require_equal "$node_id" "$expected" "test owner node for $name"
done < <(awk -F '\t' 'NR > 1 && ($2 == "TestCase" || $2 == "TestHelper") { print $3 "\t" $5 }' "$OWNER_MAP")
while IFS= read -r test_name; do
  grep -Fq "test=$test_name" "$INVENTORY"
done <"$BUILD_DIR/source-tests.txt"

proof_token="$(printf '%s\n' "$SOURCE_COMMIT" "$DETAIL_HEADER_BLOB" "$DETAIL_IMPLEMENTATION_BLOB" "$LIFECYCLE_TEST_BLOB" "$CATALOG_REFERENCE_TOOL_SHA256" "$REFERENCE_TOOL_SHA256" | sha256_text)"
grep -Fqx "proof_token_sha256=$proof_token" "$INVENTORY"

compiler=("${CXX:-c++}")
compiler_flags=(-Wno-unused-command-line-argument)
linker_gc=(-Wl,--gc-sections)
if [[ "$(uname -s)" == Darwin ]]; then
  linker_gc=(-Wl,-dead_strip)
  xcode_clang=/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/clang++
  xcode_sdk=/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX15.2.sdk
  if [[ -z ${CXX:-} && -x "$xcode_clang" && -d "$xcode_sdk" ]]; then
    compiler=("$xcode_clang")
    compiler_flags+=(-isysroot "$xcode_sdk")
  fi
fi
compile_flags=(
  "${compiler_flags[@]}"
  -std=c++20 -O2 -ffunction-sections -fdata-sections
  -I"$BUILD_DIR/source/include" -I"$BUILD_DIR/source/src" -I"$SML_SOURCE/include"
)
echo "Compiling the pinned Gemma4 reference observer"
"${compiler[@]}" "${compile_flags[@]}" -c \
  "$ROOT_DIR/tools/emel-model-gemma4-reference/main.cpp" \
  -o "$BUILD_DIR/reference-main.o"
"${compiler[@]}" "${compile_flags[@]}" -c \
  "$BUILD_DIR/source/src/emel/model/data.cpp" \
  -o "$BUILD_DIR/reference-data.o"
"${compiler[@]}" "${compile_flags[@]}" -c \
  "$BUILD_DIR/source/src/emel/model/generation/any.cpp" \
  -o "$BUILD_DIR/reference-generation.o"
"${compiler[@]}" "${compile_flags[@]}" -c \
  "$BUILD_DIR/source/src/emel/model/gemma4/detail.cpp" \
  -o "$BUILD_DIR/reference-gemma4.o"
"${compiler[@]}" \
  "${compiler_flags[@]}" \
  "$BUILD_DIR/reference-main.o" \
  "$BUILD_DIR/reference-data.o" \
  "$BUILD_DIR/reference-generation.o" \
  "$BUILD_DIR/reference-gemma4.o" \
  "${linker_gc[@]}" -o "$BUILD_DIR/cpp-observer"
cargo build --quiet --locked --manifest-path "$ROOT_DIR/Cargo.toml"   -p emel-model --example gemma4_observer
cargo test --quiet --locked --manifest-path "$ROOT_DIR/Cargo.toml"   -p emel-model gemma4::tests::loads_all_source_hparams_array_feed_forward_and_derived_facts

"$BUILD_DIR/cpp-observer" --source-built >"$BUILD_DIR/cpp.out"
"$ROOT_DIR/target/debug/examples/gemma4_observer" --source-built >"$BUILD_DIR/rust.out"
diff -u "$BUILD_DIR/cpp.out" "$BUILD_DIR/rust.out"

{
  cat "$BUILD_DIR/rust.out"
  printf 'proof_token_sha256=%s\n' "$proof_token"
  printf 'source_inventory_sha256=%s\n' "$(sha256_file "$INVENTORY")"
  printf 'source_owner_map_sha256=%s\n' "$(sha256_file "$OWNER_MAP")"
  printf 'canonical_ast_inventory_sha256=%s\n' "$(sha256_file "$CANONICAL_AST")"
  printf 'reference_tool_sha256=%s\n' "$REFERENCE_TOOL_SHA256"
  printf 'reference_catalog_helper_sha256=%s\n' "$CATALOG_REFERENCE_TOOL_SHA256"
} >"$BUILD_DIR/manifest.txt"

if $UPDATE; then
  mkdir -p "$(dirname "$SNAPSHOT")"
  install -m 0644 "$BUILD_DIR/manifest.txt" "$SNAPSHOT"
fi
diff -u "$SNAPSHOT" "$BUILD_DIR/manifest.txt"
echo "Model Gemma4 pinned source-built parity passed (emel.cpp $SOURCE_COMMIT)"
