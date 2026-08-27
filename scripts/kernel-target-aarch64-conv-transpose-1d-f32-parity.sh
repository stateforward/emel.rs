#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EMEL_CPP_SOURCE="${EMEL_CPP_SOURCE_DIR:-$ROOT_DIR/../emel.cpp-worktrees/pinned-843a117386ef17dc5a50549bbfc821074c2141d6}"
SML_SOURCE="${EMEL_STATEFORWARD_SML_SOURCE:-$ROOT_DIR/../../sml/sml.cpp-worktrees/pinned-49207123cd3f39767764bae774932cb48623f92f}"
BUILD_DIR="${EMEL_TARGET_AARCH64_CONV_TRANSPOSE_LIVE_ARTIFACT_DIR:-$ROOT_DIR/.artifacts/kernel-goal/target-aarch64-conv-transpose-1d-f32-live}"
SNAPSHOT="$ROOT_DIR/snapshots/parity/kernel-target-aarch64-conv-transpose-1d-f32/manifest.txt"
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
    *) printf 'error: unknown argument: %s\n' "$argument" >&2; exit 2 ;;
  esac
done

for command in cargo cmake diff git grep rustc shasum; do
  command -v "$command" >/dev/null 2>&1 || { printf 'error: missing command: %s\n' "$command" >&2; exit 2; }
done
TARGET_ARCH="$(rustc --print cfg | awk -F'"' '/^target_arch=/{print $2; exit}')"
[[ "$TARGET_ARCH" == aarch64 ]] || { printf 'error: native AArch64 is required\n' >&2; exit 2; }
[[ -d "$EMEL_CPP_SOURCE" && -d "$SML_SOURCE" ]] || { printf 'error: pinned source tree missing\n' >&2; exit 2; }
EMEL_CPP_SOURCE="$(cd "$EMEL_CPP_SOURCE" && pwd)"
SML_SOURCE="$(cd "$SML_SOURCE" && pwd)"
[[ -z "$(git -C "$EMEL_CPP_SOURCE" status --porcelain=v1 --untracked-files=all)" ]] || { printf 'error: emel.cpp tree is dirty\n' >&2; exit 1; }
[[ -z "$(git -C "$SML_SOURCE" status --porcelain=v1 --untracked-files=all)" ]] || { printf 'error: SML tree is dirty\n' >&2; exit 1; }
[[ "$(git -C "$EMEL_CPP_SOURCE" rev-parse HEAD)" == "$SOURCE_COMMIT" ]] || { printf 'error: emel.cpp pin mismatch\n' >&2; exit 1; }
[[ "$(git -C "$SML_SOURCE" rev-parse HEAD)" == "$SML_COMMIT" ]] || { printf 'error: SML pin mismatch\n' >&2; exit 1; }
for entry in "src/emel/kernel/aarch64/guards.hpp:$GUARDS_BLOB" "src/emel/kernel/aarch64/actions.hpp:$ACTIONS_BLOB" "src/emel/kernel/aarch64/sm.hpp:$SM_BLOB"; do
  path="${entry%%:*}"; expected="${entry##*:}"
  [[ "$(git -C "$EMEL_CPP_SOURCE" rev-parse "$SOURCE_COMMIT:$path")" == "$expected" ]] || { printf 'error: source blob mismatch for %s\n' "$path" >&2; exit 1; }
done

mkdir -p "$BUILD_DIR/tmp" "$BUILD_DIR/target" "$BUILD_DIR/logs"
export TMPDIR="$BUILD_DIR/tmp"
export CARGO_TARGET_DIR="$BUILD_DIR/target"
if $LIVE; then
  cmake -S "$ROOT_DIR/tools/emel-kernel-target-aarch64-conv-transpose-1d-f32-reference" -B "$BUILD_DIR/cpp-build" -G Ninja \
    -DEMEL_CPP_SOURCE_DIR="$EMEL_CPP_SOURCE" -DFETCHCONTENT_SOURCE_DIR_STATEFORWARD_SML="$SML_SOURCE" \
    -DCMAKE_BUILD_TYPE=Release -DEMEL_ENABLE_TESTS=OFF -DEMEL_ENABLE_FUZZ=OFF >"$BUILD_DIR/logs/cmake-configure.log" 2>&1
  cmake --build "$BUILD_DIR/cpp-build" --target emel-kernel-target-aarch64-conv-transpose-1d-f32-reference >"$BUILD_DIR/logs/cmake-build.log" 2>&1
  cargo build --locked --offline -p emel-kernel-target-aarch64-conv-transpose-1d-f32-parity >"$BUILD_DIR/logs/cargo-build.log" 2>&1
  "$BUILD_DIR/cpp-build/emel-kernel-target-aarch64-conv-transpose-1d-f32-reference" >"$BUILD_DIR/cpp.out" 2>"$BUILD_DIR/cpp.stderr"
  "$CARGO_TARGET_DIR/debug/emel-kernel-target-aarch64-conv-transpose-1d-f32-parity" >"$BUILD_DIR/rust.out" 2>"$BUILD_DIR/logs/rust.stderr"
else
  [[ -f "$BUILD_DIR/cpp.out" && -f "$BUILD_DIR/rust.out" ]] || { printf 'error: observer outputs missing\n' >&2; exit 1; }
fi
[[ ! -s "$BUILD_DIR/cpp.stderr" ]] || { printf 'error: C++ observer wrote stderr\n' >&2; exit 1; }
[[ ! -s "$BUILD_DIR/logs/rust.stderr" ]] || { printf 'error: Rust observer wrote stderr\n' >&2; exit 1; }
diff -u "$BUILD_DIR/cpp.out" "$BUILD_DIR/rust.out" >"$BUILD_DIR/logs/observer-diff.log"
sha256_file() { shasum -a 256 "$1" | awk '{print $1}'; }
{
  printf 'kind=live_split_observer\nresult=live_split_observer_match\n'
  printf 'observer=split_pinned_cpp_aarch64_sm_and_public_target_router\nreference=../emel.cpp\n'
  cat "$BUILD_DIR/rust.out"
  printf 'source_guard_span=src/emel/kernel/aarch64/guards.hpp:287-297,408-417,1026-1049\n'
  printf 'source_action_span=src/emel/kernel/aarch64/actions.hpp:1252-1258,7074-7121,8770-8779\n'
  printf 'source_transition_span=src/emel/kernel/aarch64/sm.hpp:800-818\n'
  printf 'reference_execution=pinned_aarch64_sm_process_event\nrust_execution=public_target_aarch64_kernel_process_event\n'
  printf 'output_sha256=%s\n' "$(sha256_file "$BUILD_DIR/rust.out")"
  printf 'reference_observer_sha256=%s\n' "$(sha256_file "$ROOT_DIR/tools/emel-kernel-target-aarch64-conv-transpose-1d-f32-reference/main.cpp")"
  printf 'rust_observer_sha256=%s\n' "$(sha256_file "$ROOT_DIR/tools/emel-kernel-target-aarch64-conv-transpose-1d-f32-parity/src/main.rs")"
  printf 'rust_router=crates/emel-kernels/src/aarch64/sm.rs\n'
  printf 'rust_actor=crates/emel-kernels/src/aarch64/conv_transpose_1d.rs\n'
  printf 'positive_case=dense F32 NEON op_conv_transpose_1d kernel5 out2 in1 input_length2 stride2\n'
  printf 'rejection_cases=invalid input length and nonzero padding preserve sentinel\n'
  printf 'residuals=F16 weights,strided tensor metadata,shared scalar transition,portable and x86 routes\n'
  printf 'observer_diff=empty byte-for-byte match\nallocation_contract=zero allocations covered by maintained actor test\nunsafe_contract=none\ndispatch_contract=run_to_completion_public_target_router\n'
  printf 'live_artifacts=%s\nsource_commit=%s\nsml_commit=%s\nguards_blob=%s\nactions_blob=%s\nstate_machine_blob=%s\nresult=match\n' "$BUILD_DIR" "$SOURCE_COMMIT" "$SML_COMMIT" "$GUARDS_BLOB" "$ACTIONS_BLOB" "$SM_BLOB"
} >"$BUILD_DIR/manifest.txt"
if $UPDATE; then mkdir -p "$(dirname "$SNAPSHOT")"; install -m 0644 "$BUILD_DIR/manifest.txt" "$SNAPSHOT"; fi
if $COMPARE; then [[ -f "$SNAPSHOT" ]] || { printf 'error: snapshot missing\n' >&2; exit 1; }; diff -u "$SNAPSHOT" "$BUILD_DIR/manifest.txt" >"$BUILD_DIR/logs/snapshot-diff.log"; fi
printf 'AArch64 target conv-transpose-1d F32 live parity passed (emel.cpp %s)\n' "$SOURCE_COMMIT"
