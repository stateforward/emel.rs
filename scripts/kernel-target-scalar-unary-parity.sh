#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -n "${EMEL_CPP_SOURCE_DIR:-}" ]]; then
  EMEL_CPP_SOURCE="$EMEL_CPP_SOURCE_DIR"
else
  EMEL_CPP_SOURCE="$ROOT_DIR/../emel.cpp-worktrees/pinned-843a117386ef17dc5a50549bbfc821074c2141d6"
fi
SML_SOURCE="${EMEL_STATEFORWARD_SML_SOURCE:-$ROOT_DIR/../../sml/sml.cpp-worktrees/pinned-49207123cd3f39767764bae774932cb48623f92f}"
BUILD_DIR="${EMEL_TARGET_SCALAR_UNARY_LIVE_ARTIFACT_DIR:-$ROOT_DIR/.artifacts/kernel-goal/target-scalar-unary-live}"
SNAPSHOT="${EMEL_TARGET_SCALAR_UNARY_SNAPSHOT:-$ROOT_DIR/snapshots/parity/kernel-target-scalar-unary/manifest.txt}"
SOURCE_COMMIT=843a117386ef17dc5a50549bbfc821074c2141d6
SML_COMMIT=49207123cd3f39767764bae774932cb48623f92f
EVENTS_BLOB=4b3f02fa5ff9c5071d1fc83938cef1b22b4658b9
DETAIL_BLOB=c8a82643eabfe8f2d7883e655955f455794511b0
GUARDS_BLOB=c25714566ec9a02679daef85089544575123408e
ACTIONS_BLOB=267d4f74e6e7498155c8535920322ffef2c02fb6
AARCH64_SM_BLOB=865a9cc6ba6115382ed043c464f3d62bcd851357
LIVE=true
COMPARE=true
UPDATE=true

usage() {
  cat <<'USAGE'
usage: scripts/kernel-target-scalar-unary-parity.sh [OPTIONS]

Runs pinned C++ and public Rust AArch64 scalar unary observers. Snapshot,
update, and live execution are enabled by default.

Options: --snapshot --no-snapshot --update --no-update
         --snapshot-only --live-only --update-only --help
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

for command in cargo cmake diff git ninja rustc shasum; do
  command -v "$command" >/dev/null 2>&1 || {
    printf 'error: required command is missing: %s\n' "$command" >&2
    exit 2
  }
done

TARGET_ARCH="$(rustc --print cfg | awk -F'"' '/^target_arch=/{print $2; exit}')"
[[ "$TARGET_ARCH" == aarch64 ]] || {
  printf 'error: this observer requires a native AArch64 Rust target\n' >&2
  exit 2
}
[[ -n "$EMEL_CPP_SOURCE" && -d "$EMEL_CPP_SOURCE" ]] || {
  printf 'error: pinned emel.cpp checkout is missing\n' >&2
  exit 2
}

EMEL_CPP_SOURCE="$(cd "$EMEL_CPP_SOURCE" && pwd)"
SML_SOURCE="$(cd "$SML_SOURCE" && pwd)"
for tree in "$EMEL_CPP_SOURCE" "$SML_SOURCE"; do
  [[ -z "$(git -C "$tree" status --porcelain=v1 --untracked-files=all)" ]] || {
    printf 'error: reference tree is dirty: %s\n' "$tree" >&2
    exit 1
  }
done
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
  cmake -S "$ROOT_DIR/tools/emel-kernel-target-scalar-unary-reference" \
    -B "$BUILD_DIR/cpp-build" -G Ninja \
    -DEMEL_CPP_SOURCE_DIR="$EMEL_CPP_SOURCE" \
    -DFETCHCONTENT_SOURCE_DIR_STATEFORWARD_SML="$SML_SOURCE" \
    -DCMAKE_BUILD_TYPE=Release -DEMEL_ENABLE_TESTS=OFF -DEMEL_ENABLE_FUZZ=OFF \
    >"$BUILD_DIR/logs/cmake-configure.log" 2>&1
  cmake --build "$BUILD_DIR/cpp-build" \
    --target emel-kernel-target-scalar-unary-reference \
    >"$BUILD_DIR/logs/cmake-build.log" 2>&1
  cargo build --locked --offline -p emel-kernel-target-scalar-unary-parity \
    >"$BUILD_DIR/logs/cargo-build.log" 2>&1
  "$BUILD_DIR/cpp-build/emel-kernel-target-scalar-unary-reference" \
    >"$BUILD_DIR/cpp.out" 2>"$BUILD_DIR/cpp.stderr"
  "$CARGO_TARGET_DIR/debug/emel-kernel-target-scalar-unary-parity" \
    >"$BUILD_DIR/rust.out" 2>"$BUILD_DIR/logs/rust.stderr"
else
  [[ -f "$BUILD_DIR/cpp.out" && -f "$BUILD_DIR/rust.out" ]] || {
    printf 'error: --snapshot-only requires existing observer outputs in %s\n' "$BUILD_DIR" >&2
    exit 1
  }
  [[ -f "$BUILD_DIR/cpp.stderr" ]] || {
    printf 'error: --snapshot-only requires C++ metadata probe evidence\n' >&2
    exit 1
  }
fi

grep -E '^malformed_metadata status=reject output_preserved=true output_bits=[0-9a-f]+(,[0-9a-f]+){8}$' \
  "$BUILD_DIR/cpp.stderr" >"$BUILD_DIR/logs/malformed-metadata-probe.log"
if [[ -s "$BUILD_DIR/logs/rust.stderr" ]]; then
  printf 'error: Rust public observer wrote unexpected stderr\n' >&2
  exit 1
fi
diff -u "$BUILD_DIR/cpp.out" "$BUILD_DIR/rust.out" \
  >"$BUILD_DIR/logs/observer-diff.log"

sha256_file() { shasum -a 256 "$1" | awk '{print $1}'; }
{
  printf 'kind=live_split_observer\n'
  printf 'result=live_split_observer_match\n'
  printf 'observer=split_pinned_cpp_aarch64_sm_and_public_scalar_target_router\n'
  printf 'reference=../emel.cpp\n'
  cat "$BUILD_DIR/rust.out"
  printf 'source_detail_span=src/emel/kernel/detail.hpp:2392-2411,3271-3325\n'
  printf 'source_aarch64_guard_span=src/emel/kernel/aarch64/guards.hpp:826-834,848-860,877-886\n'
  printf 'source_aarch64_action_span=src/emel/kernel/aarch64/actions.hpp:9194-9211,9376-9380\n'
  printf 'source_aarch64_transition_span=src/emel/kernel/aarch64/sm.hpp:1195-1223\n'
  printf 'events_blob=%s\n' "$EVENTS_BLOB"
  printf 'detail_blob=%s\n' "$DETAIL_BLOB"
  printf 'guards_blob=%s\n' "$GUARDS_BLOB"
  printf 'actions_blob=%s\n' "$ACTIONS_BLOB"
  printf 'state_machine_blob=%s\n' "$AARCH64_SM_BLOB"
  printf 'reference_execution=pinned_aarch64_sm_process_event\n'
  printf 'rust_execution=public_target_aarch64_kernel_process_event\n'
  printf 'output_sha256=%s\n' "$(sha256_file "$BUILD_DIR/rust.out")"
  printf 'reference_observer_sha256=%s\n' "$(sha256_file "$ROOT_DIR/tools/emel-kernel-target-scalar-unary-reference/main.cpp")"
  printf 'rust_observer_sha256=%s\n' "$(sha256_file "$ROOT_DIR/tools/emel-kernel-target-scalar-unary-parity/src/main.rs")"
  printf 'rust_module=crates/emel-kernels/src/detail/scalar_unary.rs\n'
  printf 'rust_router=crates/emel-kernels/src/aarch64/sm.rs\n'
  printf 'rust_tests=crates/emel-kernels/tests/arch/target_aarch64_kernel.rs::scalar_unary_*\n'
  printf 'scope=scalar F32 exp/tanh/elu/gelu/silu over equal non-empty dense slices\n'
  printf 'body_tail_case=count=5 (scalar path; SiLU misses SIMD alignment predicate)\n'
  printf 'positive_cases=exp/tanh/elu/gelu/silu scalar residual exact output bits\n'
  printf 'rejection_cases=zero_exp/zero_tanh/zero_silu and unequal invalid_shape typed rejection\n'
  printf 'rejection_sentinel=unequal_count output remains 7fc00001; C++ zero-count storage remains sentinel\n'
  printf 'cpp_only_probe=malformed stride plus unknown-dtype metadata rejected with sentinel-preserving output\n'
  printf 'observer_diff=empty byte-for-byte match\n'
  printf 'allocation_contract=zero allocations after construction (focused actor test)\n'
  printf 'unsafe_contract=none\n'
  printf 'dispatch_contract=run_to_completion_public_target_router\n'
  printf 'live_artifacts=.artifacts/kernel-goal/target-scalar-unary-live\n'
  printf 'residuals=SIMD SiLU is excluded; no portable/strided/quantized/non-F32 variants; no x86 live target claim\n'
  printf 'toolchain=rustc 1.98.0; cargo 1.98.0; pulp 0.22.3; stateforward-sml C++ commit %s\n' "$SML_COMMIT"
  printf 'result=match\n'
} >"$BUILD_DIR/manifest.txt"

if $UPDATE; then
  mkdir -p "$(dirname "$SNAPSHOT")"
  install -m 0644 "$BUILD_DIR/manifest.txt" "$SNAPSHOT"
fi
if $COMPARE; then
  [[ -f "$SNAPSHOT" ]] || {
    printf 'error: parity snapshot is missing: %s\n' "$SNAPSHOT" >&2
    exit 1
  }
  diff -u "$SNAPSHOT" "$BUILD_DIR/manifest.txt" \
    >"$BUILD_DIR/logs/snapshot-diff.log"
fi

printf 'AArch64 target scalar unary F32 live parity passed (emel.cpp %s)\n' "$SOURCE_COMMIT"
