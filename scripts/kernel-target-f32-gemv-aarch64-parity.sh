#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EMEL_CPP_SOURCE="${EMEL_CPP_SOURCE_DIR:-$ROOT_DIR/../emel.cpp-worktrees/pinned-843a117386ef17dc5a50549bbfc821074c2141d6}"
SML_SOURCE="${EMEL_STATEFORWARD_SML_SOURCE:-$ROOT_DIR/../../sml/sml.cpp-worktrees/pinned-49207123cd3f39767764bae774932cb48623f92f}"
BUILD_DIR="${EMEL_TARGET_F32_GEMV_AARCH64_LIVE_ARTIFACT_DIR:-$ROOT_DIR/.artifacts/kernel-goal/target-f32-gemv-aarch64-live}"
SNAPSHOT="${EMEL_TARGET_F32_GEMV_AARCH64_SNAPSHOT:-$ROOT_DIR/snapshots/parity/kernel-target-f32-gemv-aarch64/manifest.txt}"
SOURCE_COMMIT=843a117386ef17dc5a50549bbfc821074c2141d6
SML_COMMIT=49207123cd3f39767764bae774932cb48623f92f
GUARDS_BLOB=c25714566ec9a02679daef85089544575123408e
ACTIONS_BLOB=267d4f74e6e7498155c8535920322ffef2c02fb6
SM_BLOB=865a9cc6ba6115382ed043c464f3d62bcd851357
LIVE=true
COMPARE=true
UPDATE=false

for argument in "$@"; do
  case "$argument" in
    --snapshot) COMPARE=true ;;
    --no-snapshot) COMPARE=false ;;
    --update) UPDATE=true ;;
    --no-update) UPDATE=false ;;
    --snapshot-only) LIVE=false; UPDATE=false; COMPARE=true ;;
    --live-only) LIVE=true; UPDATE=false; COMPARE=false ;;
    --update-only) LIVE=true; UPDATE=true; COMPARE=false ;;
    --help|-h) printf '%s\n' 'usage: scripts/kernel-target-f32-gemv-aarch64-parity.sh [--snapshot|--no-snapshot] [--update|--no-update] [--snapshot-only|--live-only|--update-only]'; exit 0 ;;
    *) printf 'error: unknown argument: %s\n' "$argument" >&2; exit 2 ;;
  esac
done

for command in cargo cmake diff git ninja rustfmt shasum; do
  command -v "$command" >/dev/null 2>&1 || { printf 'error: required command is missing: %s\n' "$command" >&2; exit 2; }
done
EMEL_CPP_SOURCE="$(cd "$EMEL_CPP_SOURCE" && pwd)"
SML_SOURCE="$(cd "$SML_SOURCE" && pwd)"
[[ "$(git -C "$EMEL_CPP_SOURCE" rev-parse HEAD)" == "$SOURCE_COMMIT" ]]
[[ "$(git -C "$SML_SOURCE" rev-parse HEAD)" == "$SML_COMMIT" ]]
for entry in "src/emel/kernel/aarch64/guards.hpp:$GUARDS_BLOB" "src/emel/kernel/aarch64/actions.hpp:$ACTIONS_BLOB" "src/emel/kernel/aarch64/sm.hpp:$SM_BLOB"; do
  path="${entry%%:*}"; expected="${entry##*:}"
  [[ "$(git -C "$EMEL_CPP_SOURCE" rev-parse "$SOURCE_COMMIT:$path")" == "$expected" ]]
done

mkdir -p "$BUILD_DIR/tmp" "$BUILD_DIR/target" "$BUILD_DIR/logs"
export TMPDIR="$BUILD_DIR/tmp"
export CARGO_TARGET_DIR="$BUILD_DIR/target"
if $LIVE; then
  cmake -S "$ROOT_DIR/tools/emel-kernel-target-f32-gemv-aarch64-reference" -B "$BUILD_DIR/cpp-build" -G Ninja \
    -DEMEL_CPP_SOURCE_DIR="$EMEL_CPP_SOURCE" -DFETCHCONTENT_SOURCE_DIR_STATEFORWARD_SML="$SML_SOURCE" \
    -DCMAKE_BUILD_TYPE=Release -DEMEL_ENABLE_TESTS=OFF -DEMEL_ENABLE_FUZZ=OFF >"$BUILD_DIR/logs/cmake-configure.log" 2>&1
  cmake --build "$BUILD_DIR/cpp-build" --target emel-kernel-target-f32-gemv-aarch64-reference >"$BUILD_DIR/logs/cmake-build.log" 2>&1
  cargo run --locked --offline --manifest-path "$ROOT_DIR/tools/emel-kernel-target-f32-gemv-aarch64-parity/Cargo.toml" --quiet \
    >"$BUILD_DIR/rust.out" 2>"$BUILD_DIR/logs/rust.stderr"
  "$BUILD_DIR/cpp-build/emel-kernel-target-f32-gemv-aarch64-reference" >"$BUILD_DIR/cpp.out" 2>"$BUILD_DIR/logs/cpp.stderr"
else
  [[ -f "$BUILD_DIR/cpp.out" && -f "$BUILD_DIR/rust.out" ]]
fi
[[ ! -s "$BUILD_DIR/logs/rust.stderr" ]]
diff -u "$BUILD_DIR/cpp.out" "$BUILD_DIR/rust.out" >"$BUILD_DIR/logs/observer-diff.log"
grep -F 'case=invalid_shape status=reject error=InvalidShape output_bits=7fc00001,7fc00001,7fc00001,7fc00001,7fc00001' "$BUILD_DIR/rust.out" >/dev/null

{
  printf 'kind=live_split_observer\nresult=live_split_observer_match\nobserver=split_pinned_cpp_aarch64_sm_and_public_target_router\nreference=../emel.cpp\n'
  cat "$BUILD_DIR/rust.out"
  printf 'source_aarch64_guard_span=src/emel/kernel/aarch64/guards.hpp:397-405\nsource_aarch64_action_span=src/emel/kernel/aarch64/actions.hpp:1210-1231\nsource_aarch64_execution_span=src/emel/kernel/aarch64/actions.hpp:7019-7042\nsource_aarch64_transition_span=src/emel/kernel/aarch64/sm.hpp:515-518\nreference_execution=pinned_aarch64_sm_process_event\nrust_execution=public_target_aarch64_kernel_process_event\n'
  printf 'guards_blob=%s\nactions_blob=%s\nstate_machine_blob=%s\n' "$GUARDS_BLOB" "$ACTIONS_BLOB" "$SM_BLOB"
  printf 'output_sha256=%s\n' "$(shasum -a 256 "$BUILD_DIR/rust.out" | awk '{print $1}')"
  printf 'rust_module=crates/emel-kernels/src/aarch64/gemv.rs\nrust_router=crates/emel-kernels/src/aarch64/sm.rs\nrust_tests=crates/emel-kernels/tests/arch/target_aarch64_kernel.rs::router_dispatches_gemv\n'
  printf 'scope=dense_contiguous_f32_gemv_m5_k17_body_tail_and_typed_rejection\npositive_cases=gemv_m5_k17_exact_output_bits\nrejection_cases=zero_count_and_dimension_mismatch_sentinel_preserving\nobserver_diff=empty byte-for-byte match\nallocation_contract=zero allocations after construction (existing focused actor test)\nunsafe_contract=none\ndispatch_contract=run_to_completion_public_target_router\nlive_artifacts=.artifacts/kernel-goal/target-f32-gemv-aarch64-live\nresiduals=n_gt_1 dense matrix, noncontiguous strides, other dtypes, and x86 live target\ntoolchain=rustc 1.98.0; cargo 1.98.0; pulp 0.22.3; stateforward-sml %s\nresult=match\n' "$SML_COMMIT"
} >"$BUILD_DIR/manifest.txt"

if $UPDATE; then mkdir -p "$(dirname "$SNAPSHOT")"; install -m 0644 "$BUILD_DIR/manifest.txt" "$SNAPSHOT"; fi
if $COMPARE; then diff -u "$SNAPSHOT" "$BUILD_DIR/manifest.txt" >"$BUILD_DIR/logs/snapshot-diff.log"; fi
printf 'AArch64 target F32 GEMV live parity passed (emel.cpp %s)\n' "$SOURCE_COMMIT"
