#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EMEL_CPP_SOURCE="${EMEL_CPP_SOURCE_DIR:-$ROOT_DIR/../emel.cpp-worktrees/pinned-843a117386ef17dc5a50549bbfc821074c2141d6}"
SML_SOURCE="${EMEL_STATEFORWARD_SML_SOURCE:-$ROOT_DIR/../../sml/sml.cpp-worktrees/pinned-49207123cd3f39767764bae774932cb48623f92f}"
BUILD_DIR="${EMEL_TARGET_IM2COL_LIVE_ARTIFACT_DIR:-$ROOT_DIR/.artifacts/kernel-goal/target-im2col-live}"
SNAPSHOT="${EMEL_TARGET_IM2COL_SNAPSHOT:-$ROOT_DIR/snapshots/parity/kernel-target-im2col/manifest.txt}"
SOURCE_COMMIT=843a117386ef17dc5a50549bbfc821074c2141d6
SML_COMMIT=49207123cd3f39767764bae774932cb48623f92f
EVENTS_BLOB=4b3f02fa5ff9c5071d1fc83938cef1b22b4658b9
DETAIL_BLOB=c8a82643eabfe8f2d7883e655955f455794511b0
GUARDS_BLOB=c25714566ec9a02679daef85089544575123408e
ACTIONS_BLOB=267d4f74e6e7498155c8535920322ffef2c02fb6
SM_BLOB=865a9cc6ba6115382ed043c464f3d62bcd851357
LIVE=true; UPDATE=false; COMPARE=true

for argument in "$@"; do
  case "$argument" in
    --snapshot) COMPARE=true ;;
    --no-snapshot) COMPARE=false ;;
    --update) UPDATE=true ;;
    --no-update) UPDATE=false ;;
    --snapshot-only) LIVE=false; UPDATE=false; COMPARE=true ;;
    --live-only) LIVE=true; UPDATE=false; COMPARE=false ;;
    --update-only) LIVE=true; UPDATE=true; COMPARE=false ;;
    --help|-h) printf '%s\n' 'usage: scripts/kernel-target-im2col-parity.sh [--snapshot-only|--live-only|--update-only]'; exit 0 ;;
    *) printf 'error: unknown argument: %s\n' "$argument" >&2; exit 2 ;;
  esac
done
for command in cargo cmake diff git ninja rustc rustfmt shasum; do
  command -v "$command" >/dev/null 2>&1 || { printf 'error: required command is missing: %s\n' "$command" >&2; exit 2; }
done
TARGET_ARCH="$(rustc --print cfg | awk -F'"' '/^target_arch=/{print $2; exit}')"
[[ "$TARGET_ARCH" == aarch64 ]] || { printf '%s\n' 'error: this observer requires a native AArch64 Rust target' >&2; exit 2; }
EMEL_CPP_SOURCE="$(cd "$EMEL_CPP_SOURCE" && pwd)"; SML_SOURCE="$(cd "$SML_SOURCE" && pwd)"
[[ "$(git -C "$EMEL_CPP_SOURCE" rev-parse HEAD)" == "$SOURCE_COMMIT" ]] || { printf '%s\n' 'error: emel.cpp is not pinned' >&2; exit 1; }
[[ "$(git -C "$SML_SOURCE" rev-parse HEAD)" == "$SML_COMMIT" ]] || { printf '%s\n' 'error: stateforward-sml is not pinned' >&2; exit 1; }
for entry in "src/emel/kernel/events.hpp:$EVENTS_BLOB" "src/emel/kernel/detail.hpp:$DETAIL_BLOB" "src/emel/kernel/aarch64/guards.hpp:$GUARDS_BLOB" "src/emel/kernel/aarch64/actions.hpp:$ACTIONS_BLOB" "src/emel/kernel/aarch64/sm.hpp:$SM_BLOB"; do
  path="${entry%%:*}"; expected="${entry##*:}"
  [[ "$(git -C "$EMEL_CPP_SOURCE" rev-parse "$SOURCE_COMMIT:$path")" == "$expected" ]] || { printf 'error: pinned blob mismatch for %s\n' "$path" >&2; exit 1; }
done
mkdir -p "$BUILD_DIR/tmp" "$BUILD_DIR/target" "$BUILD_DIR/logs"
export TMPDIR="$BUILD_DIR/tmp" CARGO_TARGET_DIR="$BUILD_DIR/target"
if $LIVE; then
  cmake -S "$ROOT_DIR/tools/emel-kernel-target-im2col-reference" -B "$BUILD_DIR/cpp-build" -G Ninja -DEMEL_CPP_SOURCE_DIR="$EMEL_CPP_SOURCE" -DFETCHCONTENT_SOURCE_DIR_STATEFORWARD_SML="$SML_SOURCE" -DCMAKE_BUILD_TYPE=Release -DEMEL_ENABLE_TESTS=OFF -DEMEL_ENABLE_FUZZ=OFF >"$BUILD_DIR/logs/cmake-configure.log" 2>&1
  cmake --build "$BUILD_DIR/cpp-build" --target emel-kernel-target-im2col-reference >"$BUILD_DIR/logs/cmake-build.log" 2>&1
  cargo build --locked --offline --manifest-path "$ROOT_DIR/tools/emel-kernel-target-im2col-parity/Cargo.toml" >"$BUILD_DIR/logs/cargo-build.log" 2>&1
  "$BUILD_DIR/cpp-build/emel-kernel-target-im2col-reference" >"$BUILD_DIR/cpp.out" 2>"$BUILD_DIR/logs/cpp.stderr"
  "$BUILD_DIR/target/debug/emel-kernel-target-im2col-parity" >"$BUILD_DIR/rust.out" 2>"$BUILD_DIR/logs/rust.stderr"
else
  [[ -f "$BUILD_DIR/cpp.out" && -f "$BUILD_DIR/rust.out" ]] || { printf '%s\n' 'error: --snapshot-only requires existing observer outputs' >&2; exit 1; }
fi
diff -u "$BUILD_DIR/cpp.out" "$BUILD_DIR/rust.out" >"$BUILD_DIR/logs/observer-diff.log"
{
  printf 'kind=live_split_observer\nresult=live_split_observer_match\nobserver=split_pinned_cpp_and_public_rust_observers\n'
  cat "$BUILD_DIR/rust.out"
  printf 'events_blob=%s\ndetail_blob=%s\nguards_blob=%s\nactions_blob=%s\nstate_machine_blob=%s\n' "$EVENTS_BLOB" "$DETAIL_BLOB" "$GUARDS_BLOB" "$ACTIONS_BLOB" "$SM_BLOB"
  printf 'source_aarch64_transition_span=src/emel/kernel/aarch64/sm.hpp:821-833\nsource_detail_span=src/emel/kernel/detail.hpp:4858-4960\n'
  printf 'reference_execution=pinned_aarch64_sm_process_event\nrust_execution=public_target_aarch64_kernel_process_im2col\noutput_sha256=%s\n' "$(shasum -a 256 "$BUILD_DIR/rust.out" | awk '{print $1}')"
  printf 'rust_router=crates/emel-kernels/src/aarch64/sm.rs\nrust_child=crates/emel-kernels/src/any/im2col.rs\n'
  printf 'scope=dense native AArch64 F32 1-D op_im2col\npositive_cases=zero_padding\nrejection_cases=invalid_parameters/shape_mismatch\nallocation=zero_after_construction\nunsafe=none\nresiduals=im2col_back;im2col_3d\nlive_artifacts=.artifacts/kernel-goal/target-im2col-live\nresult=match\n'
} >"$BUILD_DIR/manifest.txt"
if $UPDATE; then mkdir -p "$(dirname "$SNAPSHOT")"; install -m 0644 "$BUILD_DIR/manifest.txt" "$SNAPSHOT"; fi
$COMPARE && diff -u "$SNAPSHOT" "$BUILD_DIR/manifest.txt" >"$BUILD_DIR/logs/snapshot-diff.log"
printf 'AArch64 target im2col live parity passed (emel.cpp %s)\n' "$SOURCE_COMMIT"
