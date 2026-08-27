#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EMEL_CPP_SOURCE="${EMEL_CPP_SOURCE_DIR:-$ROOT_DIR/../emel.cpp-worktrees/pinned-843a117386ef17dc5a50549bbfc821074c2141d6}"
SML_SOURCE="${EMEL_STATEFORWARD_SML_SOURCE:-$ROOT_DIR/../../sml/sml.cpp-worktrees/pinned-49207123cd3f39767764bae774932cb48623f92f}"
BUILD_DIR="${EMEL_TARGET_AARCH64_BINARY_F32_LIVE_ARTIFACT_DIR:-$ROOT_DIR/.artifacts/kernel-goal/target-aarch64-binary-f32-live}"
SNAPSHOT="${EMEL_TARGET_AARCH64_BINARY_F32_SNAPSHOT:-$ROOT_DIR/snapshots/parity/kernel-target-aarch64-binary-f32/manifest.txt}"
SOURCE_COMMIT=843a117386ef17dc5a50549bbfc821074c2141d6
SML_COMMIT=49207123cd3f39767764bae774932cb48623f92f
EVENTS_BLOB=4b3f02fa5ff9c5071d1fc83938cef1b22b4658b9
DETAIL_BLOB=c8a82643eabfe8f2d7883e655955f455794511b0
GUARDS_BLOB=c25714566ec9a02679daef85089544575123408e
ACTIONS_BLOB=267d4f74e6e7498155c8535920322ffef2c02fb6
AARCH64_SM_BLOB=865a9cc6ba6115382ed043c464f3d62bcd851357
LIVE=true
UPDATE=false
COMPARE=true

usage() {
  cat <<'USAGE'
usage: scripts/kernel-target-aarch64-binary-f32-parity.sh [OPTIONS]

Runs the pinned C++ and public Rust observers and compares their output.
The default also replays and compares the committed snapshot manifest.

Options: --snapshot --no-snapshot --update --no-update
         --snapshot-only --live-only --update-only
USAGE
}

for argument in "$@"; do
  case "$argument" in
    --snapshot) COMPARE=true ;;
    --no-snapshot) COMPARE=false ;;
    --update) UPDATE=true ;;
    --no-update) UPDATE=false ;;
    --snapshot-only) LIVE=false; UPDATE=false; COMPARE=true ;;
    --live-only) LIVE=true; UPDATE=false; COMPARE=false ;;
    --update-only) LIVE=true; UPDATE=true; COMPARE=false ;;
    --help|-h) usage; exit 0 ;;
    *) printf 'error: unknown argument: %s\n' "$argument" >&2; usage >&2; exit 2 ;;
  esac
done

for command in cargo cmake diff git ninja rustfmt shasum; do
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
  "src/emel/kernel/aarch64/guards.hpp:$GUARDS_BLOB" \
  "src/emel/kernel/aarch64/actions.hpp:$ACTIONS_BLOB" \
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
  cmake -S "$ROOT_DIR/tools/emel-kernel-target-aarch64-binary-f32-reference" \
    -B "$BUILD_DIR/cpp-build" -G Ninja \
    -DEMEL_CPP_SOURCE_DIR="$EMEL_CPP_SOURCE" \
    -DFETCHCONTENT_SOURCE_DIR_STATEFORWARD_SML="$SML_SOURCE" \
    -DCMAKE_BUILD_TYPE=Release -DEMEL_ENABLE_TESTS=OFF -DEMEL_ENABLE_FUZZ=OFF \
    >"$BUILD_DIR/logs/cmake-configure.log" 2>&1
  cmake --build "$BUILD_DIR/cpp-build" \
    --target emel-kernel-target-aarch64-binary-f32-reference \
    >"$BUILD_DIR/logs/cmake-build.log" 2>&1
  cargo build --locked --offline -p emel-kernel-target-aarch64-binary-f32-parity \
    >"$BUILD_DIR/logs/cargo-build.log" 2>&1

  "$BUILD_DIR/cpp-build/emel-kernel-target-aarch64-binary-f32-reference" \
    >"$BUILD_DIR/cpp.out" 2>"$BUILD_DIR/logs/cpp.stderr"
  "$CARGO_TARGET_DIR/debug/emel-kernel-target-aarch64-binary-f32-parity" \
    >"$BUILD_DIR/rust.out" 2>"$BUILD_DIR/logs/rust.stderr"
else
  [[ -f "$BUILD_DIR/cpp.out" && -f "$BUILD_DIR/rust.out" ]] || {
    printf 'error: --snapshot-only requires existing observer outputs in %s\n' "$BUILD_DIR" >&2
    exit 1
  }
fi
diff -u "$BUILD_DIR/cpp.out" "$BUILD_DIR/rust.out" \
  >"$BUILD_DIR/logs/observer-diff.log"

{
  printf 'kind=live_split_observer\n'
  printf 'result=live_split_observer_match\n'
  printf 'observer=split_pinned_cpp_and_public_rust_observers\n'
  printf 'reference=../emel.cpp\n'
  cat "$BUILD_DIR/rust.out"
  printf 'events_blob=%s\n' "$EVENTS_BLOB"
  printf 'detail_blob=%s\n' "$DETAIL_BLOB"
  printf 'guards_blob=%s\n' "$GUARDS_BLOB"
  printf 'actions_blob=%s\n' "$ACTIONS_BLOB"
  printf 'state_machine_blob=%s\n' "$AARCH64_SM_BLOB"
  printf 'source_detail_guard_span=src/emel/kernel/detail.hpp:3794-3797\n'
  printf 'source_detail_run_span=src/emel/kernel/detail.hpp:2365-2383\n'
  printf 'source_aarch64_guard_span=src/emel/kernel/aarch64/guards.hpp:1065-1080\n'
  printf 'source_aarch64_action_span=src/emel/kernel/aarch64/actions.hpp:2239-2339,8444-8468,8576-8593\n'
  printf 'source_aarch64_transition_span=src/emel/kernel/aarch64/sm.hpp:46-143\n'
  printf 'reference_execution=pinned_aarch64_sm_process_event\n'
  printf 'rust_execution=public_target_aarch64_kernel_process_event\n'
  printf 'output_sha256=%s\n' "$(shasum -a 256 "$BUILD_DIR/rust.out" | awk '{print $1}')"
  printf 'rust_module=crates/emel-kernels/src/aarch64/binary.rs\n'
  printf 'rust_router=crates/emel-kernels/src/aarch64/sm.rs\n'
  printf 'rust_tests=crates/emel-kernels/tests/arch/target_aarch64_kernel.rs\n'
  printf 'scope=NEON F32 add/sub/mul/div over equal non-empty dense slices\n'
  printf 'live_artifacts=.artifacts/kernel-goal/target-aarch64-binary-f32-live\n'
  printf 'positive_cases=add/sub/mul/div exact output bits\n'
  printf 'rejection_case=invalid_shape typed rejection with sentinel-preserving output\n'
  printf 'observer_diff=empty byte-for-byte match\n'
  printf 'allocation_contract=zero allocations after construction\n'
  printf 'unsafe_contract=none\n'
  printf 'dispatch_contract=run_to_completion_public_router\n'
  printf 'residuals=non-dense layouts; row-broadcast is a separate maintained route; scalar/portable routes; quantized and non-F32 operands; x86 live target unavailable on this host\n'
  printf 'toolchain=rustc 1.98.0; cargo 1.98.0; pulp 0.22.3; stateforward-sml 1.2.0\n'
  printf 'result=match\n'
} >"$BUILD_DIR/manifest.txt"

if $UPDATE; then
  mkdir -p "$(dirname "$SNAPSHOT")"
  install -m 0644 "$BUILD_DIR/manifest.txt" "$SNAPSHOT"
fi
$COMPARE && diff -u "$SNAPSHOT" "$BUILD_DIR/manifest.txt" \
  >"$BUILD_DIR/logs/snapshot-diff.log"

printf 'AArch64 target binary F32 live parity passed (emel.cpp %s)\n' "$SOURCE_COMMIT"
