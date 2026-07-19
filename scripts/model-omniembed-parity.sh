#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SOURCE_DIR="${EMEL_CPP_SOURCE_DIR:-$ROOT_DIR/../emel.cpp}"
SML_DIR="${EMEL_STATEFORWARD_SML_SOURCE:-$SOURCE_DIR/build/zig/_deps/stateforward_sml-src}"
BUILD_DIR="${EMEL_MODEL_OMNIEMBED_PARITY_BUILD_DIR:-$ROOT_DIR/target/model-omniembed-parity}"
FIXTURE="$SOURCE_DIR/tests/models/TE-75M-q8_0.gguf"
SNAPSHOT="$ROOT_DIR/snapshots/parity/model-omniembed/manifest.txt"
OWNER_MAP="$ROOT_DIR/snapshots/parity/model-omniembed/source-owner-map.tsv"
INVENTORY="$ROOT_DIR/snapshots/parity/model-omniembed/source-inventory.txt"
CANONICAL_AST="$ROOT_DIR/tools/emel-model-parity-inventory/snapshots/pinned-ast-coverage.tsv"
SOURCE_COMMIT=843a117386ef17dc5a50549bbfc821074c2141d6
SML_COMMIT=49207123cd3f39767764bae774932cb48623f92f
SML_INCLUDE_TREE=a4a817e1a1cdb4e4b6beeec79c2b56a7903ed914
INVENTORY_SHA256=ef68ef1dcd85003ad1c4be3c828df2a7e7d25832c8345a2ff3bb47861db6e901
OWNER_MAP_SHA256=ac3a4db5101cbb5c561c09a9624fb9cc5fbb0c10948aa76395ad402e6350775c
REFERENCE_SHA256=dbdd4ee1577e288836b552c223e24279e9c7d3ed62993ce6e02afddb0067cbbc
UPDATE=false
[[ ${1:-} == --update ]] && UPDATE=true
[[ $# -le 1 && (${1:-} == "" || ${1:-} == --update || ${1:-} == --live) ]] || {
  echo "usage: scripts/model-omniembed-parity.sh [--update|--live]" >&2
  exit 2
}

sha256_file() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{print $1}'
  else
    shasum -a 256 "$1" | awk '{print $1}'
  fi
}
require_equal() {
  [[ "$1" == "$2" ]] || {
    echo "model-omniembed parity: $3 mismatch: expected $2 got $1" >&2
    exit 1
  }
}
fail() { echo "model-omniembed parity: $1" >&2; exit 1; }
rust_counterpart_exists() {
  local counterpart="$1" path symbol
  path="${counterpart%%::*}"
  [[ -e "$ROOT_DIR/$path" ]] || return 1
  [[ "$counterpart" == *::* ]] || return 0
  symbol="${counterpart##*::}"
  [[ "$symbol" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]] || return 1
  grep -Eq "^[[:space:]]*((pub|pub\([^)]*\))[[:space:]]+)?(((const|async|unsafe)[[:space:]]+)*fn|const|struct|enum|trait|type)[[:space:]]+${symbol}([^A-Za-z0-9_]|$)|^[[:space:]]*${symbol}([<{]|$)" "$ROOT_DIR/$path"
}

require_equal "$(git -C "$SOURCE_DIR" rev-parse HEAD)" "$SOURCE_COMMIT" "source commit"
git -C "$SML_DIR" cat-file -e "$SML_COMMIT^{commit}" 2>/dev/null || fail "pinned stateforward-sml commit is unavailable"
require_equal "$(git -C "$SML_DIR" rev-parse "$SML_COMMIT:include")" "$SML_INCLUDE_TREE" "stateforward-sml include tree"
require_equal "$(git -C "$SOURCE_DIR" rev-parse "$SOURCE_COMMIT:src/emel/model/omniembed/detail.hpp")" 88669d66bc799b05dcd1a2b99c88516f951a48e7 "detail.hpp blob"
require_equal "$(git -C "$SOURCE_DIR" rev-parse "$SOURCE_COMMIT:src/emel/model/omniembed/detail.cpp")" 03e558a90e8df7824a0880ec2aa5500d9cc6f76b "detail.cpp blob"
require_equal "$(git -C "$SOURCE_DIR" rev-parse "$SOURCE_COMMIT:tests/model/loader/lifecycle_tests.cpp")" 27f51f427cb300709369aa7bf63774c93f739b71 "lifecycle blob"
require_equal "$(git -C "$SOURCE_DIR" rev-parse "$SOURCE_COMMIT:tests/models/TE-75M-q8_0.gguf")" 346f96b045939347eda71538547a2b932cf85a63 "fixture blob"
require_equal "$(sha256_file "$FIXTURE")" 955b5c847cc95c94ff14a27667d9aca039983448fd8cefe4f2804d3bfae621ae "fixture sha256"
require_equal "$(wc -c < "$FIXTURE" | tr -d ' ')" 119710336 "fixture bytes"
require_equal "$(sha256_file "$INVENTORY")" "$INVENTORY_SHA256" "source inventory"
require_equal "$(sha256_file "$OWNER_MAP")" "$OWNER_MAP_SHA256" "source owner map"
require_equal "$(sha256_file "$ROOT_DIR/tools/emel-model-omniembed-reference/main.cpp")" "$REFERENCE_SHA256" "reference observer"

mkdir -p "$BUILD_DIR"
awk -F '\t' '
  $1 == "src/emel/model/omniembed/detail.hpp" ||
  $1 == "src/emel/model/omniembed/detail.cpp" {
    print $1 "\t" $8 "\t" $9 "\t" $10 "\t" $11
  }
' "$CANONICAL_AST" | LC_ALL=C sort >"$BUILD_DIR/canonical-owner-nodes.tsv"
awk -F '\t' '
  NR == 1 { next }
  $2 != "File" && $2 != "TestCase" && $2 != "TestHelper" && $2 != "TestAssertion" {
    if (NF != 7 || $6 == "" || $7 == "") exit 1
    print $1 "\t" $2 "\t" $3 "\t" $4 "\t" $5
    nodes++
  }
  END { if (nodes != 73) exit 1 }
' "$OWNER_MAP" | LC_ALL=C sort >"$BUILD_DIR/mapped-owner-nodes.tsv"
diff -u "$BUILD_DIR/canonical-owner-nodes.tsv" "$BUILD_DIR/mapped-owner-nodes.tsv"
require_equal "$(awk -F '\t' 'NR>1 && $2=="File"{n++} END{print n+0}' "$OWNER_MAP")" 2 "owned file count"
require_equal "$(awk -F '\t' 'NR>1 && $2=="TestCase"{n++} END{print n+0}' "$OWNER_MAP")" 8 "test case count"
require_equal "$(awk -F '\t' 'NR>1 && $2=="TestHelper"{n++} END{print n+0}' "$OWNER_MAP")" 1 "test helper count"
require_equal "$(awk -F '\t' 'NR>1 && $2=="TestAssertion"{n++} END{print n+0}' "$OWNER_MAP")" 67 "test assertion count"
require_equal "$(awk -F '\t' 'NR>1 && $2=="TestAssertion" && $6=="ported_assertion"{n++} END{print n+0}' "$OWNER_MAP")" 62 "ported assertion count"
require_equal "$(awk -F '\t' 'NR>1 && $2=="TestAssertion" && $6=="open_architecture_router"{n++} END{print n+0}' "$OWNER_MAP")" 4 "open router assertion count"
require_equal "$(awk -F '\t' 'NR>1 && $2=="TestAssertion" && $6=="open_generic_model_loader"{n++} END{print n+0}' "$OWNER_MAP")" 1 "open generic model-loader assertion count"
require_equal "$(awk 'END{print NR-1}' "$OWNER_MAP")" 151 "owner map row count"

while IFS=$'\t' read -r _ kind _ _ _ disposition counterparts; do
  [[ "$counterparts" == rust_counterpart || "$kind" == TestCase ]] && continue
  if [[ "$disposition" == open_architecture_router ]]; then
    require_equal "$counterparts" future:crates/emel-model/src/architecture "future router owner"
    continue
  fi
  if [[ "$disposition" == open_generic_model_loader ]]; then
    require_equal "$counterparts" future:crates/emel-model/src/loader "future generic model-loader owner"
    continue
  fi
  IFS=';' read -r -a owners <<<"$counterparts"
  for owner in "${owners[@]}"; do
    rust_counterpart_exists "$owner" || fail "missing or malformed Rust counterpart: $owner"
  done
done <"$OWNER_MAP"

git -C "$SOURCE_DIR" show "$SOURCE_COMMIT:tests/model/loader/lifecycle_tests.cpp" |
  perl -0777 -ne 'while (/TEST_CASE\s*\(((?:"[^"]*"\s*)+)\)/g) { $x=$1; $x=~s/"\s*"//g; $x=~s/^\s*"//; $x=~s/"\s*$//; print "$x\n" if $x=~/omniembed/ }' |
  LC_ALL=C sort >"$BUILD_DIR/source-tests.txt"
awk -F '\t' 'NR>1 && $2=="TestCase"{print $3}' "$OWNER_MAP" | LC_ALL=C sort >"$BUILD_DIR/mapped-tests.txt"
diff -u "$BUILD_DIR/source-tests.txt" "$BUILD_DIR/mapped-tests.txt"
git -C "$SOURCE_DIR" show "$SOURCE_COMMIT:tests/model/loader/lifecycle_tests.cpp" |
  awk '/^void build_omniembed_model\(/ { found=1 } END { exit !found }' ||
  fail "missing pinned build_omniembed_model helper"

mkdir -p "$BUILD_DIR/source" "$BUILD_DIR/sml" "$BUILD_DIR/objects"
git -C "$SOURCE_DIR" archive "$SOURCE_COMMIT" include src | tar -x -C "$BUILD_DIR/source"
git -C "$SML_DIR" archive "$SML_COMMIT" include | tar -x -C "$BUILD_DIR/sml"
compiler="${CXX:-c++}"
flags=()
if [[ "$(uname -s)" == Darwin && -x /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/clang++ ]]; then
  compiler=/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/clang++
  flags=(-isysroot /Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX15.2.sdk)
fi
sources=(
  "$ROOT_DIR/tools/emel-model-omniembed-reference/main.cpp"
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
  "$compiler" "${flags[@]}" -std=c++20 -O2 -Wno-unused-command-line-argument \
    -I"$BUILD_DIR/source/include" -I"$BUILD_DIR/source/src" -I"$BUILD_DIR/sml/include" \
    -c "$source" -o "$object"
  objects+=("$object")
  index=$((index + 1))
done
"$compiler" "${flags[@]}" "${objects[@]}" -o "$BUILD_DIR/cpp-observer"

env CARGO_BUILD_JOBS=1 cargo test --quiet --locked -p emel-model omniembed::tests --lib
env CARGO_BUILD_JOBS=1 cargo test --quiet --locked -p emel-model --test omniembed_privacy
"$BUILD_DIR/cpp-observer" --negative "$FIXTURE" >"$BUILD_DIR/cpp-negative.out"
printf '%s\n' 'negative missing_audio_projection=rejected' 'negative invalid_matryoshka=rejected' >"$BUILD_DIR/expected-negative.out"
diff -u "$BUILD_DIR/expected-negative.out" "$BUILD_DIR/cpp-negative.out"
env CARGO_BUILD_JOBS=1 cargo build --quiet --locked -p emel-model --example omniembed_observer
"$BUILD_DIR/cpp-observer" --fixture "$FIXTURE" >"$BUILD_DIR/cpp.out"
"$ROOT_DIR/target/debug/examples/omniembed_observer" --fixture "$FIXTURE" >"$BUILD_DIR/rust.out"
diff -u "$BUILD_DIR/cpp.out" "$BUILD_DIR/rust.out"
{
  cat "$BUILD_DIR/rust.out"
  printf 'fixture_sha256=%s\n' 955b5c847cc95c94ff14a27667d9aca039983448fd8cefe4f2804d3bfae621ae
  printf 'source_owner_map_sha256=%s\n' "$OWNER_MAP_SHA256"
  printf 'reference_tool_sha256=%s\n' "$REFERENCE_SHA256"
} >"$BUILD_DIR/manifest.txt"
if $UPDATE; then
  mkdir -p "$(dirname "$SNAPSHOT")"
  install -m 0644 "$BUILD_DIR/manifest.txt" "$SNAPSHOT"
fi
diff -u "$SNAPSHOT" "$BUILD_DIR/manifest.txt"
echo "Model OmniEmbed real-fixture parity passed"
