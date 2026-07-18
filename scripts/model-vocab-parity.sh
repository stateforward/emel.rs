#!/usr/bin/env bash
set -euo pipefail

if [[ -d /opt/homebrew/bin ]]; then
  export PATH="/opt/homebrew/bin:$PATH"
fi

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
SOURCE_REPO=${EMEL_CPP_SOURCE_REPO:-"$ROOT/../emel.cpp"}
SOURCE_COMMIT=843a117386ef17dc5a50549bbfc821074c2141d6
SOURCE_TREE=d2fd66887fbdef6e0894de9391155099839e625c
DETAIL_BLOB=7c964f7449640fd7fe9eef21f4c14f3a65bf7b6a
DATA_BLOB=78a25b987423d8cbef17965a8ca92596ffc0ecef
MODEL_PROFILE_BLOB=ef7ff8da51f1f281082901bf4919a4b9a63f2671
PRE_PROFILE_BLOB=16b2982ca16dfdfbee016d50d0eb924a7cdc18c4
SML_COMMIT=49207123cd3f39767764bae774932cb48623f92f
SML_SOURCE=${EMEL_STATEFORWARD_SML_SOURCE:-"$SOURCE_REPO/build/zig/_deps/stateforward_sml-src"}
FIXTURE_ROOT=${EMEL_CPP_FIXTURE_ROOT:-"$SOURCE_REPO/tests/models"}
TOKEN_PROFILE_MANIFEST=${EMEL_TOKEN_PROFILE_PARITY_SNAPSHOT:-"$ROOT/snapshots/parity/token-profile/manifest.txt"}

for command in git tar cmake "${CXX:-c++}" cargo diff; do
  command -v "$command" >/dev/null || {
    printf 'required command is missing: %s\n' "$command" >&2
    exit 2
  }
done

sha256() {
  if command -v sha256sum >/dev/null; then
    sha256sum "$1" | awk '{print $1}'
  else
    shasum -a 256 "$1" | awk '{print $1}'
  fi
}

check_equal() {
  local label=$1
  local actual=$2
  local expected=$3
  if [[ "$actual" != "$expected" ]]; then
    printf '%s identity mismatch: expected %s, got %s\n' \
      "$label" "$expected" "$actual" >&2
    exit 2
  fi
}

check_equal source_tree \
  "$(git -C "$SOURCE_REPO" rev-parse "$SOURCE_COMMIT^{tree}")" "$SOURCE_TREE"
check_equal detail_blob \
  "$(git -C "$SOURCE_REPO" rev-parse "$SOURCE_COMMIT:src/emel/model/detail.cpp")" \
  "$DETAIL_BLOB"
check_equal data_blob \
  "$(git -C "$SOURCE_REPO" rev-parse "$SOURCE_COMMIT:src/emel/model/data.hpp")" \
  "$DATA_BLOB"
check_equal model_profile_blob \
  "$(git -C "$SOURCE_REPO" rev-parse "$SOURCE_COMMIT:src/emel/text/tokenizer/detail.hpp")" \
  "$MODEL_PROFILE_BLOB"
check_equal pre_profile_blob \
  "$(git -C "$SOURCE_REPO" rev-parse "$SOURCE_COMMIT:src/emel/text/tokenizer/preprocessor/detail.hpp")" \
  "$PRE_PROFILE_BLOB"
check_equal sml_commit "$(git -C "$SML_SOURCE" rev-parse HEAD)" "$SML_COMMIT"

fixture_names=(
  Llama-68M-Chat-v1-Q2_K.gguf
  distilgpt2.Q2_K.gguf
  flan-t5-small.Q2_K.gguf
  bert-base-uncased-q4_k_m.gguf
  rwkv7-0.1B-g1-F16.gguf
  model-tiny-q80.gguf
)
fixture_hashes=(
  8ed06dc5bd84bce3154a2b7e751c45a56562691933ee25b5823393f909329a67
  b046ac09ba24a848e2140676fba58c1dcf2f19617e45b03524043eabdb556a31
  a67f632d17d2bdb819071c9b4d51e26a191f75c7f725861ee22e23e1d903dc57
  48c02c00843964c2e1675e6d6aebfbdb03d4ca330d65a6b9695eee6f160109b0
  fea5c54f3fd2370ac90ae58f2ecd6cbe57c31df023598aed4c95b0966170f9c8
  52deb0fdcbb9c36b4d570e35f5a65a5ad4275ccdb85e7a06e81a8b05b3743c9d
)
fixture_paths=()
for index in "${!fixture_names[@]}"; do
  path="$FIXTURE_ROOT/${fixture_names[$index]}"
  [[ -f "$path" ]] || {
    printf 'required pinned fixture is missing: %s\n' "$path" >&2
    exit 2
  }
  check_equal "fixture_${fixture_names[$index]}" "$(sha256 "$path")" \
    "${fixture_hashes[$index]}"
  fixture_paths+=("$path")
done

WORK=$(mktemp -d "${TMPDIR:-/tmp}/emel-model-vocab-parity.XXXXXX")
trap 'rm -rf "$WORK"' EXIT

EMEL_CPP_SOURCE_DIR="$SOURCE_REPO" \
  "$ROOT/scripts/paritychecker.sh" --suite=token-profile --no-update \
  >"$WORK/token-profile-parity.log"
grep -qx 'Tokenizer profile independent C++/Rust live comparison passed' \
  "$WORK/token-profile-parity.log"
grep -qx 'Tokenizer profile checked snapshot passed' \
  "$WORK/token-profile-parity.log"

mkdir -p "$WORK/source"
git -C "$SOURCE_REPO" archive "$SOURCE_COMMIT" CMakeLists.txt cmake include src \
  | tar -x -C "$WORK/source"

check_equal archived_detail_sha256 \
  "$(sha256 "$WORK/source/src/emel/model/detail.cpp")" \
  8a817e8ca007fd8230e5e71d79f7b59cf37e514ef34408c50813cd41910c9d6f

cmake -S "$WORK/source" -B "$WORK/build" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DEMEL_ENABLE_TESTS=OFF \
  -DFETCHCONTENT_SOURCE_DIR_STATEFORWARD_SML="$SML_SOURCE" >/dev/null
cmake --build "$WORK/build" --target emel >/dev/null

"${CXX:-c++}" -std=c++20 -O2 \
  -I"$WORK/source/include" \
  -I"$WORK/source/src" \
  -I"$SML_SOURCE/include" \
  "$ROOT/tools/emel-model-vocab-reference/main.cpp" \
  "$WORK/build/libemel.a" \
  -o "$WORK/cpp-observer"

"$WORK/cpp-observer" "${fixture_paths[@]}" >"$WORK/cpp.out"
cargo run --quiet --locked --manifest-path "$ROOT/Cargo.toml" \
  -p emel-model --example vocabulary_observer -- "${fixture_paths[@]}" \
  >"$WORK/rust.out"
diff -u "$WORK/cpp.out" "$WORK/rust.out"
cargo test --quiet --locked --manifest-path "$ROOT/Cargo.toml" \
  -p emel-model --lib \
  vocabulary::tests::dependency_retains_substituted_pre_identity_and_rejects_other_identity \
  -- --exact >"$WORK/pre-retention-test.log"

printf 'model-vocab-parity/v1\n'
printf 'source_commit=%s\n' "$SOURCE_COMMIT"
printf 'source_tree=%s\n' "$SOURCE_TREE"
printf 'source_detail_blob=%s\n' "$DETAIL_BLOB"
printf 'source_data_blob=%s\n' "$DATA_BLOB"
printf 'source_model_profile_blob=%s\n' "$MODEL_PROFILE_BLOB"
printf 'source_pre_profile_blob=%s\n' "$PRE_PROFILE_BLOB"
printf 'source_sml_commit=%s\n' "$SML_COMMIT"
printf 'reference_tool_sha256=%s\n' \
  "$(sha256 "$ROOT/tools/emel-model-vocab-reference/main.cpp")"
printf 'rust_observer_sha256=%s\n' \
  "$(sha256 "$ROOT/crates/emel-model/examples/vocabulary_observer.rs")"
printf 'token_profile_manifest_sha256=%s\n' "$(sha256 "$TOKEN_PROFILE_MANIFEST")"
printf 'token_profile_reference_tool_sha256=%s\n' \
  "$(sha256 "$ROOT/tools/emel-token-profile-reference/main.cpp")"
printf 'token_profile_rust_machine_sha256=%s\n' \
  "$(sha256 "$ROOT/crates/emel-token/src/profile/sm.rs")"
printf 'token_profile_rust_parity_test_sha256=%s\n' \
  "$(sha256 "$ROOT/crates/emel-token/src/profile/tests.rs")"
printf 'config=Release,tests-off,cxx20,public-cpp-model-facade,public-rust-actor,semantic-fnv1a64\n'
for index in "${!fixture_names[@]}"; do
  printf 'fixture=%s sha256=%s\n' \
    "${fixture_names[$index]}" "${fixture_hashes[$index]}"
done
printf 'result=match\n'
printf 'pre_identity_proof=token-profile-live+opaque-eq-retention+observable-model-vocab\n'
printf 'pre_identity_mutation=alternate-public-resolver-id-rejected\n'
cat "$WORK/cpp.out"
