#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EMEL_CPP_SOURCE="${EMEL_CPP_SOURCE_DIR:-$ROOT_DIR/../emel.cpp}"
SML_SOURCE="${EMEL_STATEFORWARD_SML_SOURCE:-$EMEL_CPP_SOURCE/build/zig/_deps/stateforward_sml-src}"
BUILD_DIR="${EMEL_MODEL_GENERATION_PARITY_BUILD_DIR:-$ROOT_DIR/target/model-generation-parity}"
SNAPSHOT="${EMEL_MODEL_GENERATION_PARITY_SNAPSHOT:-$ROOT_DIR/snapshots/parity/model-generation/manifest.txt}"
INVENTORY="${EMEL_MODEL_GENERATION_INVENTORY:-$ROOT_DIR/snapshots/parity/model-generation/source-inventory.txt}"
CANONICAL_AST="${EMEL_MODEL_GENERATION_CANONICAL_AST:-$ROOT_DIR/tools/emel-model-parity-inventory/snapshots/pinned-ast-coverage.tsv}"
SOURCE_COMMIT=843a117386ef17dc5a50549bbfc821074c2141d6
HEADER_BLOB=d521cf68e1bf52a2a193bbdb460741772199b318
IMPLEMENTATION_BLOB=099058ccd441d1dc6bebbb0c4994070d2f533c47
LLAMA_RELATIVE=tests/models/Llama-68M-Chat-v1-Q2_K.gguf
LLAMA_SHA256=8ed06dc5bd84bce3154a2b7e751c45a56562691933ee25b5823393f909329a67
UPDATE=false
COMPARE=true

[[ ${1:-} == "--update" ]] && UPDATE=true
[[ ${1:-} == "--live" ]] && COMPARE=false
[[ $# -le 1 && (${1:-} == "" || ${1:-} == "--update" || ${1:-} == "--live") ]] || {
  echo "usage: scripts/model-generation-parity.sh [--update|--live]" >&2
  exit 2
}

sha256_file() {
  if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | awk '{print $1}'
  else shasum -a 256 "$1" | awk '{print $1}'; fi
}

[[ "$(git -C "$EMEL_CPP_SOURCE" rev-parse HEAD)" == "$SOURCE_COMMIT" ]]
[[ "$(git -C "$EMEL_CPP_SOURCE" rev-parse "$SOURCE_COMMIT:src/emel/model/generation/any.hpp")" == "$HEADER_BLOB" ]]
[[ "$(git -C "$EMEL_CPP_SOURCE" rev-parse "$SOURCE_COMMIT:src/emel/model/generation/any.cpp")" == "$IMPLEMENTATION_BLOB" ]]
[[ "$(sha256_file "$EMEL_CPP_SOURCE/$LLAMA_RELATIVE")" == "$LLAMA_SHA256" ]]
[[ -f "$INVENTORY" && -f "$CANONICAL_AST" ]]
grep -qx "source_commit=$SOURCE_COMMIT" "$INVENTORY"
grep -qx "source_header_blob=$HEADER_BLOB" "$INVENTORY"
grep -qx "source_implementation_blob=$IMPLEMENTATION_BLOB" "$INVENTORY"
grep -qx \
  "algorithm=reject_block_tensor owner=future_family_guard_boundary status=deferred_family_specific proof=pinned_source_deferred_mapping" \
  "$INVENTORY"

validate_proof_reference() {
  local proof=$1
  local proof_file
  local proof_pattern
  case "$proof" in
    14_stage_parity | block_validation_tests | block_visit_tests | contract_visit_tests)
      proof_file="$ROOT_DIR/crates/emel-model/src/generation/tests.rs"
      proof_pattern=public_actor_builds_source_exact_attention_contract_and_fourteen_stage_audit
      ;;
    14_stage_synthetic_and_real_parity | real_fixture_parity)
      proof_file="$ROOT_DIR/scripts/model-generation-parity.sh"
      proof_pattern=--generation-fixture
      ;;
    all_four_attention_variant_tests)
      proof_file="$ROOT_DIR/crates/emel-model/src/generation/tests.rs"
      proof_pattern=remaining_attention_variants_use_explicit_transition_rows
      ;;
    all_labels_test)
      proof_file="$ROOT_DIR/crates/emel-model/src/generation/tests.rs"
      proof_pattern=exact_labels_cover_source_enums
      ;;
    all_supported_and_unknown_labels_test | plan_accessors)
      proof_file="$ROOT_DIR/crates/emel-model/src/generation/tests.rs"
      proof_pattern=every_source_label_and_descriptor_accessor_is_observed
      ;;
    attention)
      proof_file="$ROOT_DIR/crates/emel-model/src/generation/tests.rs"
      proof_pattern=public_actor_builds_source_exact_attention_contract_and_fourteen_stage_audit
      ;;
    catalog_unit | missing_and_unbound_tests)
      proof_file="$ROOT_DIR/crates/emel-model/src/catalog/tests.rs"
      proof_pattern=bound_predicate_matches_pinned_generation_semantics
      ;;
    direct_and_tied_tests)
      proof_file="$ROOT_DIR/crates/emel-model/src/generation/tests.rs"
      proof_pattern=shared_value_and_tied_output_preserve_source_bindings
      ;;
    equivalent_operand_benchmark)
      proof_file="$ROOT_DIR/scripts/model-generation-bench.sh"
      proof_pattern=benchmark_operand
      ;;
    lifecycle_and_completion_tests)
      proof_file="$ROOT_DIR/crates/emel-model/src/generation/tests.rs"
      proof_pattern=lifecycle_and_validation_failures_are_typed_and_recoverable
      ;;
    missing_required_tensor_test | missing_tests)
      proof_file="$ROOT_DIR/crates/emel-model/src/generation/tests.rs"
      proof_pattern=missing_required_tensor_is_model_invalid
      ;;
    multi_digit_name_test)
      proof_file="$ROOT_DIR/crates/emel-model/src/generation/tests.rs"
      proof_pattern='block_name(123'
      ;;
    pinned_source_deferred_mapping)
      proof_file="$CANONICAL_AST"
      proof_pattern='::reject_block_tensor('
      ;;
    reset_release_reuse_test)
      proof_file="$ROOT_DIR/crates/emel-model/src/generation/tests.rs"
      proof_pattern=reset_and_release_preserve_preallocated_storage_ownership
      ;;
    shortconv | shortconv_not_applicable_test | shortconv_test)
      proof_file="$ROOT_DIR/crates/emel-model/src/generation/tests.rs"
      proof_pattern=shortconv_marks_attention_stages_not_applicable
      ;;
    source_parity)
      proof_file="$ROOT_DIR/scripts/model-generation-parity.sh"
      proof_pattern="SOURCE_COMMIT=$SOURCE_COMMIT"
      ;;
    stage_audit_parity)
      proof_file="$ROOT_DIR/crates/emel-model/examples/generation_observer.rs"
      proof_pattern='for family in QuantizedStageFamily::ALL'
      ;;
    synthetic_parity)
      proof_file="$ROOT_DIR/scripts/model-generation-parity.sh"
      proof_pattern=cpp.synthetic
      ;;
    two_block_consistency_test)
      proof_file="$ROOT_DIR/crates/emel-model/src/generation/tests.rs"
      proof_pattern=block_stage_consistency_is_an_explicit_transition_outcome
      ;;
    *)
      echo "error: unresolved model generation inventory proof: $proof" >&2
      return 1
      ;;
  esac
  if [[ ! -f "$proof_file" ]] || ! grep -Fq -- "$proof_pattern" "$proof_file"; then
    echo "error: unresolved model generation inventory proof target: $proof" >&2
    return 1
  fi
}

while IFS= read -r proof; do
  validate_proof_reference "$proof"
done < <(
  awk '
    match($0, /proof=[^ ]+/) {
      proofs = substr($0, RSTART + 6, RLENGTH - 6)
      count = split(proofs, items, ",")
      for (item_index = 1; item_index <= count; item_index++) print items[item_index]
    }
  ' "$INVENTORY" | sort -u
)

while IFS=$'\t' read -r source_kind source_name; do
  source_pattern="::$source_name("
  [[ "$source_name" == contract_reset ]] && source_pattern='::contract::reset('
  if ! grep -Fq -- "$source_pattern" "$CANONICAL_AST"; then
    echo "error: inventory $source_kind mapping is absent from pinned AST: $source_name" >&2
    exit 1
  fi
done < <(
  awk '
    /^(algorithm|guard|public_api)=/ {
      separator = index($0, "=")
      kind = substr($0, 1, separator - 1)
      rest = substr($0, separator + 1)
      split(rest, fields, " ")
      print kind "\t" fields[1]
    }
  ' "$INVENTORY"
)
awk -F '\t' '
  $1 == "src/emel/model/generation/any.cpp" ||
  $1 == "src/emel/model/generation/any.hpp" { rows++ }
  END { exit !(rows == 164) }
' "$CANONICAL_AST"
awk -F '=' '
  /^(algorithm|guard|invariant|public_api|type|field|constant|special_member)=/ {
    count[$1]++
    total++
  }
  END {
    exit !(total == 104 && count["algorithm"] == 26 && count["guard"] == 1 &&
      count["invariant"] == 1 && count["public_api"] == 4 &&
      count["type"] == 18 && count["field"] == 48 &&
      count["constant"] == 5 && count["special_member"] == 1)
  }
' "$INVENTORY"

rm -rf "$BUILD_DIR/source"
mkdir -p "$BUILD_DIR/source"
git -C "$EMEL_CPP_SOURCE" archive "$SOURCE_COMMIT" include src | tar -x -C "$BUILD_DIR/source"
linker_gc=(-Wl,--gc-sections)
[[ "$(uname -s)" == Darwin ]] && linker_gc=(-Wl,-dead_strip)
"${CXX:-c++}" -std=c++20 -O2 -ffunction-sections -fdata-sections \
  -I"$BUILD_DIR/source/include" -I"$BUILD_DIR/source/src" -I"$SML_SOURCE/include" \
  "$ROOT_DIR/tools/emel-model-catalog-reference/main.cpp" \
  "$BUILD_DIR/source/src/emel/model/data.cpp" \
  "$BUILD_DIR/source/src/emel/model/generation/any.cpp" \
  "${linker_gc[@]}" -o "$BUILD_DIR/cpp-observer"
cargo build --quiet --locked --manifest-path "$ROOT_DIR/Cargo.toml" \
  -p emel-model --example generation_observer

"$BUILD_DIR/cpp-observer" --generation >"$BUILD_DIR/cpp.synthetic"
"$ROOT_DIR/target/debug/examples/generation_observer" >"$BUILD_DIR/rust.synthetic"
diff -u "$BUILD_DIR/cpp.synthetic" "$BUILD_DIR/rust.synthetic"
"$BUILD_DIR/cpp-observer" --generation-fixture "$EMEL_CPP_SOURCE/$LLAMA_RELATIVE" >"$BUILD_DIR/cpp.fixture"
"$ROOT_DIR/target/debug/examples/generation_observer" --fixture "$EMEL_CPP_SOURCE/$LLAMA_RELATIVE" >"$BUILD_DIR/rust.fixture"
diff -u "$BUILD_DIR/cpp.fixture" "$BUILD_DIR/rust.fixture"

{
  cat "$BUILD_DIR/rust.synthetic"
  printf 'reference_tool_sha256=%s\n' "$(sha256_file "$ROOT_DIR/tools/emel-model-catalog-reference/main.cpp")"
  printf 'fixture_llama_sha256=%s\n' "$LLAMA_SHA256"
  printf 'source_inventory_sha256=%s\n' "$(sha256_file "$INVENTORY")"
  printf 'canonical_ast_inventory_sha256=%s\n' "$(sha256_file "$CANONICAL_AST")"
  cat "$BUILD_DIR/rust.fixture"
} >"$BUILD_DIR/manifest.txt"
if $UPDATE; then
  mkdir -p "$(dirname "$SNAPSHOT")"
  install -m 0644 "$BUILD_DIR/manifest.txt" "$SNAPSHOT"
fi
$COMPARE && diff -u "$SNAPSHOT" "$BUILD_DIR/manifest.txt"
echo "Model generation source and real-fixture parity passed (emel.cpp $SOURCE_COMMIT)"
