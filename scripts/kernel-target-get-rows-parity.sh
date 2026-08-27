#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EMEL_CPP_SOURCE="${EMEL_CPP_SOURCE_DIR:-$ROOT_DIR/../emel.cpp-worktrees/pinned-843a117386ef17dc5a50549bbfc821074c2141d6}"
SML_SOURCE="${EMEL_STATEFORWARD_SML_SOURCE:-$ROOT_DIR/../../sml/sml.cpp-worktrees/pinned-49207123cd3f39767764bae774932cb48623f92f}"
BUILD_DIR="${EMEL_TARGET_GET_ROWS_LIVE_ARTIFACT_DIR:-$ROOT_DIR/.artifacts/kernel-goal/target-get-rows-live}"
SNAPSHOT="${EMEL_TARGET_GET_ROWS_PARITY_SNAPSHOT:-$ROOT_DIR/snapshots/parity/kernel-target-get-rows/manifest.txt}"
SOURCE_COMMIT=843a117386ef17dc5a50549bbfc821074c2141d6
SML_COMMIT=49207123cd3f39767764bae774932cb48623f92f
EVENTS_BLOB=4b3f02fa5ff9c5071d1fc83938cef1b22b4658b9
DETAIL_BLOB=c8a82643eabfe8f2d7883e655955f455794511b0
X86_SM_BLOB=0b4d635ebbd0fbd52dbca8a2345547fb571205c8
AARCH64_SM_BLOB=865a9cc6ba6115382ed043c464f3d62bcd851357
for argument in "$@"; do
  case "$argument" in
    --snapshot-only) LIVE=false; UPDATE=false; COMPARE=true ;;
    --live-only) LIVE=true; UPDATE=false; COMPARE=false ;;
    --update-only) LIVE=true; UPDATE=true; COMPARE=false ;;
    --update) UPDATE=true ;;
    --no-update) UPDATE=false ;;
    --help|-h)
      printf '%s\n' 'usage: scripts/kernel-target-get-rows-parity.sh [--snapshot-only|--live-only|--update-only] [--update|--no-update]'
      exit 0
      ;;
    *) printf 'error: unknown argument: %s\n' "$argument" >&2; exit 2 ;;
  esac
done

LIVE="${LIVE:-true}"
UPDATE="${UPDATE:-true}"
COMPARE="${COMPARE:-true}"

for command in cargo cmake diff git shasum; do
  command -v "$command" >/dev/null 2>&1 || {
    printf 'error: required command is missing: %s\n' "$command" >&2
    exit 2
  }
done

EMEL_CPP_SOURCE="$(cd "$EMEL_CPP_SOURCE" && pwd)"
SML_SOURCE="$(cd "$SML_SOURCE" && pwd)"
[[ "$(git -C "$EMEL_CPP_SOURCE" rev-parse HEAD)" == "$SOURCE_COMMIT" ]] || {
  printf 'error: emel.cpp is not pinned to %s\n' "$SOURCE_COMMIT" >&2
  exit 1
}
[[ "$(git -C "$SML_SOURCE" rev-parse HEAD)" == "$SML_COMMIT" ]] || {
  printf 'error: stateforward-sml is not pinned to %s\n' "$SML_COMMIT" >&2
  exit 1
}
for entry in \
  "src/emel/kernel/events.hpp:$EVENTS_BLOB" \
  "src/emel/kernel/detail.hpp:$DETAIL_BLOB" \
  "src/emel/kernel/x86_64/sm.hpp:$X86_SM_BLOB" \
  "src/emel/kernel/aarch64/sm.hpp:$AARCH64_SM_BLOB"; do
  path="${entry%%:*}"
  expected="${entry##*:}"
  [[ "$(git -C "$EMEL_CPP_SOURCE" rev-parse "$SOURCE_COMMIT:$path")" == "$expected" ]] || {
    printf 'error: pinned blob mismatch for %s\n' "$path" >&2
    exit 1
  }
done

mkdir -p "$BUILD_DIR/tmp" "$BUILD_DIR/target" "$BUILD_DIR/logs"
export TMPDIR="$BUILD_DIR/tmp"
export CARGO_TARGET_DIR="$BUILD_DIR/target"

if $LIVE; then
cmake -S "$ROOT_DIR/tools/emel-kernel-target-get-rows-reference" \
  -B "$BUILD_DIR/cpp-build" -G Ninja \
  -DEMEL_CPP_SOURCE_DIR="$EMEL_CPP_SOURCE" \
  -DFETCHCONTENT_SOURCE_DIR_STATEFORWARD_SML="$SML_SOURCE" \
  -DCMAKE_BUILD_TYPE=Release -DEMEL_ENABLE_TESTS=OFF -DEMEL_ENABLE_FUZZ=OFF \
  >"$BUILD_DIR/logs/cmake-configure.log" 2>&1
cmake --build "$BUILD_DIR/cpp-build" --target emel-kernel-target-get-rows-reference \
  >"$BUILD_DIR/logs/cmake-build.log" 2>&1
cargo build --locked --offline -p emel-kernel-target-get-rows-parity \
  >"$BUILD_DIR/logs/cargo-build.log" 2>&1

"$BUILD_DIR/cpp-build/emel-kernel-target-get-rows-reference" >"$BUILD_DIR/cpp.out"
"$CARGO_TARGET_DIR/debug/emel-kernel-target-get-rows-parity" >"$BUILD_DIR/rust.out"
diff -u "$BUILD_DIR/cpp.out" "$BUILD_DIR/rust.out" >"$BUILD_DIR/logs/observer-diff.log"
else
  [[ -f "$BUILD_DIR/manifest.txt" ]] || {
    printf 'error: --snapshot-only requires existing manifest %s\n' "$BUILD_DIR/manifest.txt" >&2
    exit 1
  }
fi

if $LIVE; then
  {
    cat "$BUILD_DIR/rust.out"
    printf 'source_detail_spans=can_run_get_rows:4460-4501,run_get_rows_as:4503-4525\n'
    printf 'source_x86_transition_span=src/emel/kernel/x86_64/sm.hpp:535-568\n'
    printf 'source_aarch64_transition_span=src/emel/kernel/aarch64/sm.hpp:656-689\n'
    printf 'reference_execution=detail_can_run_get_rows_plus_run_get_rows_as\n'
    printf 'rust_execution=public_target_router_process_get_rows\n'
    printf 'output_sha256=%s\n' "$(shasum -a 256 "$BUILD_DIR/rust.out" | awk '{print $1}')"
    printf 'result=match\n'
  } >"$BUILD_DIR/manifest.txt"
fi

if $UPDATE; then
  mkdir -p "$(dirname "$SNAPSHOT")"
  install -m 0644 "$BUILD_DIR/manifest.txt" "$SNAPSHOT"
fi
if $COMPARE; then
  [[ -f "$SNAPSHOT" ]] || { printf 'error: missing parity snapshot %s\n' "$SNAPSHOT" >&2; exit 1; }
  diff -u "$SNAPSHOT" "$BUILD_DIR/manifest.txt" >"$BUILD_DIR/logs/snapshot-diff.log"
fi

printf 'Target get_rows live parity passed (emel.cpp %s)\n' "$SOURCE_COMMIT"
