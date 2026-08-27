#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EMEL_CPP_SOURCE="${EMEL_CPP_SOURCE_DIR:-$ROOT_DIR/../emel.cpp-worktrees/pinned-843a117386ef17dc5a50549bbfc821074c2141d6}"
SML_SOURCE="${EMEL_STATEFORWARD_SML_SOURCE:-$ROOT_DIR/../../sml/sml.cpp-worktrees/pinned-49207123cd3f39767764bae774932cb48623f92f}"
BUILD_DIR="${EMEL_TARGET_SCALAR_TRIG_LIVE_ARTIFACT_DIR:-$ROOT_DIR/.artifacts/kernel-goal/target-scalar-trig-live}"
SNAPSHOT="${EMEL_TARGET_SCALAR_TRIG_SNAPSHOT:-$ROOT_DIR/snapshots/parity/kernel-target-scalar-trig/manifest.txt}"
SOURCE_COMMIT=843a117386ef17dc5a50549bbfc821074c2141d6
SML_COMMIT=49207123cd3f39767764bae774932cb48623f92f
LIVE=true; UPDATE=false; COMPARE=true

for argument in "$@"; do
  case "$argument" in
    --snapshot) COMPARE=true ;; --no-snapshot) COMPARE=false ;; --update) UPDATE=true ;; --no-update) UPDATE=false ;;
    --snapshot-only) LIVE=false; UPDATE=false; COMPARE=true ;; --live-only) LIVE=true; UPDATE=false; COMPARE=false ;; --update-only) LIVE=true; UPDATE=true; COMPARE=false ;;
    --help|-h) printf '%s\n' 'usage: scripts/kernel-target-scalar-trig-parity.sh [--snapshot-only|--live-only|--update-only]'; exit 0 ;;
    *) printf 'error: unknown argument: %s\n' "$argument" >&2; exit 2 ;;
  esac
done
for command in cargo cmake diff git ninja rustc; do command -v "$command" >/dev/null || { echo "error: required command is missing: $command" >&2; exit 2; }; done
TARGET_ARCH="$(rustc --print cfg | awk -F'"' '/^target_arch=/{print $2; exit}')"
[[ "$TARGET_ARCH" == aarch64 ]] || { echo 'error: this observer requires a native AArch64 Rust target' >&2; exit 2; }
EMEL_CPP_SOURCE="$(cd "$EMEL_CPP_SOURCE" && pwd)"; SML_SOURCE="$(cd "$SML_SOURCE" && pwd)"
[[ "$(git -C "$EMEL_CPP_SOURCE" rev-parse HEAD)" == "$SOURCE_COMMIT" ]] || { echo 'error: emel.cpp is not pinned' >&2; exit 1; }
[[ "$(git -C "$SML_SOURCE" rev-parse HEAD)" == "$SML_COMMIT" ]] || { echo 'error: stateforward-sml is not pinned' >&2; exit 1; }
for entry in "src/emel/kernel/events.hpp:4b3f02fa5ff9c5071d1fc83938cef1b22b4658b9" "src/emel/kernel/detail.hpp:c8a82643eabfe8f2d7883e655955f455794511b0" "src/emel/kernel/aarch64/guards.hpp:c25714566ec9a02679daef85089544575123408e" "src/emel/kernel/aarch64/actions.hpp:267d4f74e6e7498155c8535920322ffef2c02fb6" "src/emel/kernel/aarch64/sm.hpp:865a9cc6ba6115382ed043c464f3d62bcd851357"; do
  path="${entry%%:*}"; expected="${entry##*:}"; [[ "$(git -C "$EMEL_CPP_SOURCE" rev-parse "$SOURCE_COMMIT:$path")" == "$expected" ]] || { echo "error: pinned blob mismatch for $path" >&2; exit 1; }
done
mkdir -p "$BUILD_DIR/tmp" "$BUILD_DIR/target" "$BUILD_DIR/logs"; export TMPDIR="$BUILD_DIR/tmp" CARGO_TARGET_DIR="$BUILD_DIR/target"
if $LIVE; then
  cmake -S "$ROOT_DIR/tools/emel-kernel-target-scalar-trig-reference" -B "$BUILD_DIR/cpp-build" -G Ninja -DEMEL_CPP_SOURCE_DIR="$EMEL_CPP_SOURCE" -DFETCHCONTENT_SOURCE_DIR_STATEFORWARD_SML="$SML_SOURCE" -DCMAKE_BUILD_TYPE=Release -DEMEL_ENABLE_TESTS=OFF -DEMEL_ENABLE_FUZZ=OFF >"$BUILD_DIR/logs/cmake-configure.log" 2>&1
  cmake --build "$BUILD_DIR/cpp-build" --target emel-kernel-target-scalar-trig-reference >"$BUILD_DIR/logs/cmake-build.log" 2>&1
  cargo build --locked --offline -p emel-kernel-target-scalar-trig-parity >"$BUILD_DIR/logs/cargo-build.log" 2>&1
  "$BUILD_DIR/cpp-build/emel-kernel-target-scalar-trig-reference" >"$BUILD_DIR/cpp.out"
  "$BUILD_DIR/target/debug/emel-kernel-target-scalar-trig-parity" >"$BUILD_DIR/rust.out"
fi
[[ -s "$BUILD_DIR/cpp.out" && -s "$BUILD_DIR/rust.out" ]] || { echo 'error: live observer outputs are required' >&2; exit 1; }
diff -u "$BUILD_DIR/cpp.out" "$BUILD_DIR/rust.out" >"$BUILD_DIR/logs/observer-diff.log"
{ cat "$BUILD_DIR/rust.out"; printf '%s\n' 'result=live_split_observer_match' 'reference_execution=pinned_aarch64_sm_process_event' 'rust_execution=public_target_aarch64_kernel_process_scalar_trig' 'source_aarch64_transition_span=src/emel/kernel/aarch64/sm.hpp:176-203' 'source_detail_span=src/emel/kernel/detail.hpp:5357-5362' 'rust_router=crates/emel-kernels/src/aarch64/sm.rs' 'rust_child=crates/emel-kernels/src/any/reductions.rs' 'allocation=zero_after_construction' 'unsafe=none' 'live_artifacts=.artifacts/kernel-goal/target-scalar-trig-live'; } >"$BUILD_DIR/manifest.txt"
if $UPDATE; then install -m 0644 "$BUILD_DIR/manifest.txt" "$SNAPSHOT"; fi
$COMPARE && diff -u "$SNAPSHOT" "$BUILD_DIR/manifest.txt"
printf '%s\n' 'Kernel target AArch64 scalar trig parity passed'
