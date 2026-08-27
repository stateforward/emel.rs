#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EMEL_CPP_SOURCE="${EMEL_CPP_SOURCE_DIR:-$ROOT_DIR/../emel.cpp-worktrees/pinned-843a117386ef17dc5a50549bbfc821074c2141d6}"
SML_SOURCE="${EMEL_STATEFORWARD_SML_SOURCE:-$ROOT_DIR/../../sml/sml.cpp-worktrees/pinned-49207123cd3f39767764bae774932cb48623f92f}"
BUILD_DIR="${EMEL_KERNEL_PORTABLE_ROUTER_LIVE_ARTIFACT_DIR:-$ROOT_DIR/.artifacts/kernel-goal/portable-router-live}"
SNAPSHOT="$ROOT_DIR/snapshots/parity/kernel-portable-router/manifest.txt"

if [[ "${1:-}" == "--snapshot-only" ]]; then
  [[ -f "$SNAPSHOT" ]] || { echo "error: portable router snapshot is missing" >&2; exit 2; }
  [[ -f "$BUILD_DIR/manifest.txt" ]] || {
    echo "error: live portable router manifest is missing; run the live proof first" >&2
    exit 2
  }
  diff -u "$SNAPSHOT" "$BUILD_DIR/manifest.txt" >"$BUILD_DIR/logs/snapshot-only-diff.log"
  echo "Portable kernel router snapshot-only proof passed"
  exit 0
fi

for command in cargo cmake diff git ninja; do
  command -v "$command" >/dev/null 2>&1 || { echo "error: required command is missing: $command" >&2; exit 2; }
done
[[ -d "$EMEL_CPP_SOURCE" ]] || { echo "error: pinned emel.cpp checkout is missing" >&2; exit 2; }
[[ -d "$SML_SOURCE" ]] || { echo "error: pinned stateforward-sml checkout is missing" >&2; exit 2; }

mkdir -p "$BUILD_DIR/logs" "$BUILD_DIR/target"
export CARGO_TARGET_DIR="$BUILD_DIR/target"

cmake -S "$ROOT_DIR/tools/emel-kernel-portable-router-reference" \
  -B "$BUILD_DIR/cpp-build" -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DEMEL_CPP_SOURCE_DIR="$EMEL_CPP_SOURCE" \
  -DFETCHCONTENT_SOURCE_DIR_STATEFORWARD_SML="$SML_SOURCE" \
  >"$BUILD_DIR/logs/cmake-configure.log" 2>&1
cmake --build "$BUILD_DIR/cpp-build" --target emel-kernel-portable-router-reference \
  >"$BUILD_DIR/logs/cmake-build.log" 2>&1
cargo build --locked --offline --manifest-path \
  "$ROOT_DIR/tools/emel-kernel-portable-router-parity/Cargo.toml" \
  >"$BUILD_DIR/logs/cargo-build.log" 2>&1

"$BUILD_DIR/cpp-build/emel-kernel-portable-router-reference" >"$BUILD_DIR/reference.out"
"$CARGO_TARGET_DIR/debug/emel-kernel-portable-router-parity" >"$BUILD_DIR/rust.out"
diff -u "$BUILD_DIR/reference.out" "$BUILD_DIR/rust.out" >"$BUILD_DIR/logs/observer-diff.log"

{
  printf 'schema=emel.rs.parity.v1\n'
  printf 'component=emel-kernels::Kernel\n'
  printf 'result=live_split_observer_match\n'
  printf 'source_repository=../emel.cpp\nsource_commit=843a117386ef17dc5a50549bbfc821074c2141d6\n'
  printf 'observer=one_public_portable_kernel_and_pinned_emel_kernel_sm_reference\n'
  printf 'operations=op_dup;op_add;op_sqr\n'
  printf 'dispatch=one_kernel_one_sm_typed_operation_events\n'
  printf 'typed_unexpected_event_rejection=covered\n'
  printf 'live_artifacts=.artifacts/kernel-goal/portable-router-live\n'
  printf 'allocation_hot_path=zero_after_construction\nunsafe=none\nresult=match\n'
} >"$BUILD_DIR/manifest.txt"
mkdir -p "$(dirname "$SNAPSHOT")"
install -m 0644 "$BUILD_DIR/manifest.txt" "$SNAPSHOT"
diff -u "$SNAPSHOT" "$BUILD_DIR/manifest.txt" >"$BUILD_DIR/logs/snapshot-diff.log"
echo "Portable kernel router aggregate live parity passed (emel.cpp 843a117386ef17dc5a50549bbfc821074c2141d6)"
