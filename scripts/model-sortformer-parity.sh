#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SOURCE_DIR="${EMEL_CPP_SOURCE_DIR:-$ROOT_DIR/../emel.cpp}"
SML_DIR="${EMEL_STATEFORWARD_SML_SOURCE:-$SOURCE_DIR/build/zig/_deps/stateforward_sml-src}"
BUILD_DIR="${EMEL_MODEL_SORTFORMER_PARITY_BUILD_DIR:-$ROOT_DIR/target/model-sortformer-parity}"
FIXTURE="$SOURCE_DIR/tests/models/diar_streaming_sortformer_4spk-v2.1.gguf"
SNAPSHOT="$ROOT_DIR/snapshots/parity/model-sortformer/manifest.txt"
OWNER_MAP="$ROOT_DIR/snapshots/parity/model-sortformer/source-owner-map.tsv"
INVENTORY="$ROOT_DIR/snapshots/parity/model-sortformer/source-inventory.txt"
CANONICAL_AST="$ROOT_DIR/tools/emel-model-parity-inventory/snapshots/pinned-ast-coverage.tsv"
SOURCE_COMMIT=843a117386ef17dc5a50549bbfc821074c2141d6
SML_COMMIT=49207123cd3f39767764bae774932cb48623f92f
SML_INCLUDE_TREE=a4a817e1a1cdb4e4b6beeec79c2b56a7903ed914
INVENTORY_SHA256=5060986a587f454bec1165afd7fd5e2b122180f4f92968996e562c3669dd424e
OWNER_MAP_SHA256=661bb39af6a284fee6b9ce15f37032e99a5835d3fb831c57309b93eb6af0cc62
UPDATE=false
[[ ${1:-} == --update ]] && UPDATE=true
[[ $# -le 1 && (${1:-} == "" || ${1:-} == --update || ${1:-} == --live) ]] || { echo "usage: scripts/model-sortformer-parity.sh [--update|--live]" >&2; exit 2; }

sha256_file() { if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | awk '{print $1}'; else shasum -a 256 "$1" | awk '{print $1}'; fi; }
require_equal() { [[ "$1" == "$2" ]] || { echo "model-sortformer parity: $3 mismatch: expected $2 got $1" >&2; exit 1; }; }
fail() { echo "model-sortformer parity: $1" >&2; exit 1; }
rust_counterpart_exists() {
  local counterpart="$1"
  local path="${counterpart%%::*}"
  [[ -e "$ROOT_DIR/$path" ]] || return 1
  [[ "$counterpart" == *::* ]] || return 0
  local symbol="${counterpart##*::}"
  [[ "$symbol" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]] || return 1
  grep -Eq "^[[:space:]]*((pub|pub\([^)]*\))[[:space:]]+)?(((const|async|unsafe)[[:space:]]+)*fn|const|struct|enum|trait|type)[[:space:]]+${symbol}([^A-Za-z0-9_]|$)|^[[:space:]]*${symbol}([<{]|$)" "$ROOT_DIR/$path"
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
  grep -Fq "$1" "$ROOT_DIR/$2" || fail "missing semantic transition: $1"
}

require_equal "$(git -C "$SOURCE_DIR" rev-parse HEAD)" "$SOURCE_COMMIT" "source commit"
git -C "$SML_DIR" cat-file -e "$SML_COMMIT^{commit}" 2>/dev/null \
  || fail "pinned stateforward-sml commit is unavailable"
require_equal "$(git -C "$SML_DIR" rev-parse "$SML_COMMIT:include")" "$SML_INCLUDE_TREE" "stateforward-sml include tree"
require_equal "$(git -C "$SOURCE_DIR" rev-parse "$SOURCE_COMMIT:src/emel/model/sortformer/any.hpp")" d79cb231e0dfc75f5aa4d8da7db2ecf36c92c9b6 "any.hpp blob"
require_equal "$(git -C "$SOURCE_DIR" rev-parse "$SOURCE_COMMIT:src/emel/model/sortformer/detail.hpp")" 72766c7042707a8037ac49779b8e3f8e2b34c555 "detail.hpp blob"
require_equal "$(git -C "$SOURCE_DIR" rev-parse "$SOURCE_COMMIT:src/emel/model/sortformer/detail.cpp")" 011a3ec4dfaab102457c3f6ca6e20a5adf9934b1 "detail.cpp blob"
require_equal "$(git -C "$SOURCE_DIR" rev-parse "$SOURCE_COMMIT:tests/model/loader/lifecycle_tests.cpp")" 27f51f427cb300709369aa7bf63774c93f739b71 "lifecycle blob"
require_equal "$(git -C "$SOURCE_DIR" rev-parse "$SOURCE_COMMIT:tests/models/diar_streaming_sortformer_4spk-v2.1.gguf")" a1d4f3b3528c47693a7a9137689aa4032a62603a "fixture blob"
require_equal "$(sha256_file "$FIXTURE")" 1b85d7bf641350d0d355e7494c4b7d92a1ff2fb2d886cd6dcc43f358a6266ff0 "fixture sha256"
require_equal "$(wc -c < "$FIXTURE" | tr -d ' ')" 471107712 "fixture bytes"
require_equal "$(sha256_file "$INVENTORY")" "$INVENTORY_SHA256" "source inventory"
require_equal "$(sha256_file "$OWNER_MAP")" "$OWNER_MAP_SHA256" "owner map"
require_equal "$(sha256_file "$ROOT_DIR/tools/emel-model-sortformer-reference/main.cpp")" c714ef05b3461221f00179098e9aa5151e2d7e63b796992d1ce2a722e01d4db5 "reference observer"
require_equal "$(sha256_file "$SOURCE_DIR/tools/bench/diarization/sortformer_fixture.hpp")" 88c1b3bd9643c3020654200b47656368c1429d143ef8aa15b3448bc37cfe26b7 "reference fixture façade"
mkdir -p "$BUILD_DIR"
awk -F '\t' '
  $1 == "src/emel/model/sortformer/any.hpp" ||
  $1 == "src/emel/model/sortformer/detail.hpp" ||
  $1 == "src/emel/model/sortformer/detail.cpp" {
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
  $2 != "File" && $2 != "TestCase" && $2 != "TestHelper" && $2 != "TestAssertion" {
    if (NF != 7 || $6 == "" || $7 == "") exit 1
    print $1 "\t" $2 "\t" $3 "\t" $4 "\t" $5
    nodes++
  }
  END { if (nodes != 73) exit 1 }
' "$OWNER_MAP" | LC_ALL=C sort >"$BUILD_DIR/mapped-owner-nodes.tsv"
diff -u "$BUILD_DIR/canonical-owner-nodes.tsv" "$BUILD_DIR/mapped-owner-nodes.tsv"
require_equal "$(awk -F '\t' 'NR > 1 && $2 == "File" { count++ } END { print count + 0 }' "$OWNER_MAP")" 3 "owned file count"
require_equal "$(awk -F '\t' 'NR > 1 && $2 == "TestCase" { count++ } END { print count + 0 }' "$OWNER_MAP")" 5 "test case count"
require_equal "$(awk -F '\t' 'NR > 1 && $2 == "TestHelper" { count++ } END { print count + 0 }' "$OWNER_MAP")" 1 "test helper count"
require_equal "$(awk -F '\t' 'NR > 1 && $2 == "TestAssertion" { count++ } END { print count + 0 }' "$OWNER_MAP")" 25 "test assertion count"
require_equal "$(awk -F '\t' 'NR > 1 && $2 == "TestAssertion" && $6 == "ported_assertion" { count++ } END { print count + 0 }' "$OWNER_MAP")" 21 "ported assertion count"
require_equal "$(awk -F '\t' 'NR > 1 && $2 == "TestAssertion" && $6 == "open_architecture_router" { count++ } END { print count + 0 }' "$OWNER_MAP")" 4 "open architecture-router assertion count"
require_equal "$(awk -F '\t' 'NR > 1 && $2 == "TestCase" && $6 == "assertion_split_complete" { count++ } END { print count + 0 }' "$OWNER_MAP")" 2 "whole ported test count"
require_equal "$(awk -F '\t' 'NR > 1 && $2 == "TestCase" && $6 == "assertion_split_partial" { count++ } END { print count + 0 }' "$OWNER_MAP")" 3 "partial test count"
require_equal "$(awk 'END { print NR - 1 }' "$OWNER_MAP")" 107 "owner-map row count"

awk -F '\t' '
  NR == 1 { next }
  $2 == "TestCase" { tests[$3] = 1; next }
  $2 == "TestAssertion" {
    parent = $3
    sub(/::.*/, "", parent)
    assertions[parent]++
  }
  END {
    for (parent in assertions) if (!(parent in tests)) exit 1
    for (test in tests) if (!(test in assertions)) exit 1
  }
' "$OWNER_MAP" || fail "test assertion parent coverage is incomplete"
require_equal "$(awk -F '\t' '
  $2 == "TestAssertion" && $3 == "model_sortformer_detail_rejects_noncanonical_stream_contract::speaker_count_drift_rejects" { print $7 }
' "$OWNER_MAP")" 'crates/emel-model/src/sortformer/tests.rs::loads_source_exact_hparams_aliases_defaults_and_rejects_contract_drift' "noncanonical stream assertion owner"

while IFS=$'\t' read -r _ kind _ _ _ disposition counterparts; do
  [[ "$counterparts" == "rust_counterpart" ]] && continue
  [[ "$kind" == "TestCase" ]] && continue
  if [[ "$disposition" == "open_architecture_router" ]]; then
    require_equal "$counterparts" 'future:crates/emel-model/src/architecture' "future architecture-router owner"
    continue
  fi
  IFS=';' read -r -a owners <<<"$counterparts"
  for owner in "${owners[@]}"; do
    rust_counterpart_exists "$owner" || fail "missing or malformed Rust counterpart: $owner"
  done
done <"$OWNER_MAP"

git -C "$SOURCE_DIR" show "$SOURCE_COMMIT:tests/model/loader/lifecycle_tests.cpp" \
  | perl -0777 -ne 'while (/TEST_CASE\s*\(\s*"([^"]*sortformer[^"]*)"\s*\)/g) { print "$1\n" }' \
  | LC_ALL=C sort >"$BUILD_DIR/source-tests.txt"
awk -F '\t' 'NR > 1 && $2 == "TestCase" { print $3 }' "$OWNER_MAP" \
  | LC_ALL=C sort >"$BUILD_DIR/mapped-tests.txt"
diff -u "$BUILD_DIR/source-tests.txt" "$BUILD_DIR/mapped-tests.txt"
git -C "$SOURCE_DIR" show "$SOURCE_COMMIT:tests/model/loader/lifecycle_tests.cpp" \
  | awk '/^void build_sortformer_model\(/ { found = 1 } END { exit !found }' \
  || fail "missing pinned build_sortformer_model helper"

require_function_mapping require_string 'crates/emel-model/src/sortformer/hparams.rs::query_string;crates/emel-model/src/sortformer/hparams.rs::guard_string_present'
require_function_mapping assign_i32_any 'crates/emel-model/src/sortformer/hparams.rs::query_unsigned;crates/emel-model/src/sortformer/hparams.rs::guard_unsigned_i32'
require_function_mapping require_positive_i32 crates/emel-model/src/sortformer/hparams.rs::guard_unsigned_positive
require_function_mapping tensor_has_storage crates/emel-model/src/sortformer/actor.rs::observation_valid
require_function_mapping assign_family_view 'crates/emel-model/src/sortformer/actor.rs::contract_begin;crates/emel-model/src/sortformer/actor.rs::store_first;crates/emel-model/src/sortformer/actor.rs::increment;crates/emel-model/src/sortformer/actor.rs::effect_observation_ignored;crates/emel-model/src/sortformer/actor.rs::guard_families_complete'
require_function_mapping validate_contract 'crates/emel-model/src/sortformer/hparams.rs::guard_architecture_valid;crates/emel-model/src/sortformer/actor.rs::guard_begin_valid;crates/emel-model/src/sortformer/actor.rs::guard_families_complete'
require_function_mapping is_execution_architecture 'crates/emel-model/src/sortformer/hparams.rs::effect_query_architecture;crates/emel-model/src/sortformer/hparams.rs::guard_architecture_valid'
require_function_mapping load_hparams crates/emel-model/src/sortformer/hparams.rs::load_hparams
require_function_mapping build_execution_contract crates/emel-model/src/sortformer/actor.rs::effect_visit
require_function_mapping validate_data 'crates/emel-model/src/sortformer/hparams.rs::guard_architecture_valid;crates/emel-model/src/sortformer/actor.rs::guard_begin_valid;crates/emel-model/src/sortformer/actor.rs::guard_families_complete'
require_function_mapping validate_execution_contract 'crates/emel-model/src/sortformer/hparams.rs::guard_architecture_valid;crates/emel-model/src/sortformer/actor.rs::guard_begin_valid;crates/emel-model/src/sortformer/actor.rs::guard_families_complete'
require_transition '"state_scanning"_s <= *"state_empty"_s + Begin(BeginRuntime' crates/emel-model/src/sortformer/sm.rs
require_transition '"state_ready"_s <= "state_scanning"_s + Finish(FinishRuntime' crates/emel-model/src/sortformer/sm.rs
require_transition '"state_ready"_s <= "state_ready"_s + Visit(VisitRuntime' crates/emel-model/src/sortformer/sm.rs
require_transition '"state_architecture_decision"_s <= *"state_idle"_s + Load(HparamLoadRuntime' crates/emel-model/src/sortformer/sm.rs
require_transition '[guard_unsigned_missing] / effect_default_skipped_query_sample_primary' crates/emel-model/src/sortformer/sm.rs
env CARGO_BUILD_JOBS=1 cargo test --quiet --locked \
  --package emel-model-parity-inventory --test sortformer_callback_policy

rm -rf "$BUILD_DIR/source" "$BUILD_DIR/sml" "$BUILD_DIR/objects"
mkdir -p "$BUILD_DIR/source"
git -C "$SOURCE_DIR" archive "$SOURCE_COMMIT" include src tools/bench/diarization/sortformer_fixture.hpp tools/bench/model_load_strategy.hpp | tar -x -C "$BUILD_DIR/source"
mkdir -p "$BUILD_DIR/sml"
git -C "$SML_DIR" archive "$SML_COMMIT" include | tar -x -C "$BUILD_DIR/sml"
compiler="${CXX:-c++}"
flags=()
if [[ "$(uname -s)" == Darwin && -x /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/clang++ ]]; then
  compiler=/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/clang++
  flags=(-isysroot /Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX15.2.sdk)
fi
mkdir -p "$BUILD_DIR/objects"
sources=(
  "$ROOT_DIR/tools/emel-model-sortformer-reference/main.cpp"
  "$BUILD_DIR/source/src/emel/model/data.cpp"
  "$BUILD_DIR/source/src/emel/model/detail.cpp"
  "$BUILD_DIR/source/src/emel/model/architecture/detail.cpp"
  "$BUILD_DIR/source/src/emel/model/generation/any.cpp"
  "$BUILD_DIR/source/src/emel/model/gemma4/detail.cpp"
  "$BUILD_DIR/source/src/emel/model/lfm2/detail.cpp"
  "$BUILD_DIR/source/src/emel/model/llama/detail.cpp"
  "$BUILD_DIR/source/src/emel/model/moshi/detail.cpp"
  "$BUILD_DIR/source/src/emel/model/omniembed/detail.cpp"
  "$BUILD_DIR/source/src/emel/model/qwen3/detail.cpp"
  "$BUILD_DIR/source/src/emel/model/sortformer/detail.cpp"
  "$BUILD_DIR/source/src/emel/model/whisper/detail.cpp"
)
objects=()
index=0
for source in "${sources[@]}"; do
  object="$BUILD_DIR/objects/$index.o"
  echo "model-sortformer parity: compiling $(basename "$source") [$index]"
  "$compiler" "${flags[@]}" -std=c++20 -O2 -Wno-unused-command-line-argument \
    -DEMEL_TEST_REPO_ROOT=\"$SOURCE_DIR\" -I"$BUILD_DIR/source/include" \
    -I"$BUILD_DIR/source/src" -I"$BUILD_DIR/source/tools/bench" \
    -I"$BUILD_DIR/sml/include" -c "$source" -o "$object"
  objects+=("$object")
  index=$((index + 1))
done
echo "model-sortformer parity: linking reference observer"
"$compiler" "${flags[@]}" "${objects[@]}" -o "$BUILD_DIR/cpp-observer"
env CARGO_BUILD_JOBS=1 cargo test --quiet --locked -p emel-model \
  sortformer::tests::reports_public_hparam_missing_wrong_kind_range_and_query_failures -- --exact
env CARGO_BUILD_JOBS=1 cargo test --quiet --locked -p emel-model \
  sortformer::tests::loads_all_hparam_aliases_and_optional_defaults -- --exact
env CARGO_BUILD_JOBS=1 cargo test --quiet --locked -p emel-model \
  sortformer::tests::rejects_missing_family_and_source_contract_drift_during_binding -- --exact
env -u EMEL_MODEL_LOAD_IO_STRATEGY "$BUILD_DIR/cpp-observer" --negative > "$BUILD_DIR/cpp-negative.out"
printf '%s\n' \
  'negative architecture=rejected' \
  'negative missing_modules=rejected' \
  'negative missing_skipped=accepted_default_zero' > "$BUILD_DIR/expected-negative.out"
diff -u "$BUILD_DIR/expected-negative.out" "$BUILD_DIR/cpp-negative.out"
env CARGO_BUILD_JOBS=1 cargo build --quiet --locked --manifest-path "$ROOT_DIR/Cargo.toml" -p emel-model --example sortformer_observer
env -u EMEL_MODEL_LOAD_IO_STRATEGY "$BUILD_DIR/cpp-observer" --fixture > "$BUILD_DIR/cpp.out"
"$ROOT_DIR/target/debug/examples/sortformer_observer" --fixture "$FIXTURE" > "$BUILD_DIR/rust.out"
diff -u "$BUILD_DIR/cpp.out" "$BUILD_DIR/rust.out"
{
  cat "$BUILD_DIR/rust.out"
  printf 'fixture_sha256=%s\n' 1b85d7bf641350d0d355e7494c4b7d92a1ff2fb2d886cd6dcc43f358a6266ff0
  printf 'source_owner_map_sha256=%s\n' "$OWNER_MAP_SHA256"
  printf 'reference_tool_sha256=%s\n' c714ef05b3461221f00179098e9aa5151e2d7e63b796992d1ce2a722e01d4db5
} > "$BUILD_DIR/manifest.txt"
if $UPDATE; then install -m 0644 "$BUILD_DIR/manifest.txt" "$SNAPSHOT"; fi
diff -u "$SNAPSHOT" "$BUILD_DIR/manifest.txt"
echo "Model Sortformer real-fixture parity passed"
