#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EMEL_CPP_SOURCE="${EMEL_CPP_SOURCE_DIR:-$ROOT_DIR/../emel.cpp-worktrees/pinned-843a117386ef17dc5a50549bbfc821074c2141d6}"
SML_SOURCE="${EMEL_STATEFORWARD_SML_SOURCE:-$ROOT_DIR/../../sml/sml.cpp-worktrees/pinned-49207123cd3f39767764bae774932cb48623f92f}"
BUILD_DIR="${EMEL_TARGET_ROPE_LIVE_ARTIFACT_DIR:-$ROOT_DIR/.artifacts/kernel-goal/target-rope-live}"
SNAPSHOT="${EMEL_TARGET_ROPE_SNAPSHOT:-$ROOT_DIR/snapshots/parity/kernel-target-rope/manifest.txt}"
SOURCE_COMMIT=843a117386ef17dc5a50549bbfc821074c2141d6
SML_COMMIT=49207123cd3f39767764bae774932cb48623f92f
EVENTS_BLOB=4b3f02fa5ff9c5071d1fc83938cef1b22b4658b9
DETAIL_BLOB=c8a82643eabfe8f2d7883e655955f455794511b0
GUARDS_BLOB=c25714566ec9a02679daef85089544575123408e
ACTIONS_BLOB=267d4f74e6e7498155c8535920322ffef2c02fb6
SM_BLOB=865a9cc6ba6115382ed043c464f3d62bcd851357
LIVE=true; COMPARE=true; UPDATE=true

usage() { printf '%s\n' 'usage: scripts/kernel-target-rope-parity.sh [--snapshot-only|--live-only|--update-only] [--no-update]'; }
for argument in "$@"; do case "$argument" in
  --snapshot-only) LIVE=false; UPDATE=false; COMPARE=true;;
  --live-only) LIVE=true; UPDATE=false; COMPARE=false;;
  --update-only) LIVE=true; UPDATE=true; COMPARE=false;;
  --no-update) UPDATE=false;; --help|-h) usage; exit 0;; *) usage >&2; exit 2;;
esac; done
for command in cargo cmake diff git ninja shasum; do command -v "$command" >/dev/null || { echo "missing command: $command" >&2; exit 2; }; done
EMEL_CPP_SOURCE="$(cd "$EMEL_CPP_SOURCE" && pwd)"; SML_SOURCE="$(cd "$SML_SOURCE" && pwd)"
[[ "$(git -C "$EMEL_CPP_SOURCE" rev-parse HEAD)" == "$SOURCE_COMMIT" ]] || { echo 'emel.cpp pin mismatch' >&2; exit 1; }
[[ "$(git -C "$SML_SOURCE" rev-parse HEAD)" == "$SML_COMMIT" ]] || { echo 'stateforward-sml pin mismatch' >&2; exit 1; }
for entry in "src/emel/kernel/events.hpp:$EVENTS_BLOB" "src/emel/kernel/detail.hpp:$DETAIL_BLOB" "src/emel/kernel/aarch64/guards.hpp:$GUARDS_BLOB" "src/emel/kernel/aarch64/actions.hpp:$ACTIONS_BLOB" "src/emel/kernel/aarch64/sm.hpp:$SM_BLOB"; do
  path="${entry%%:*}"; expected="${entry##*:}"; [[ "$(git -C "$EMEL_CPP_SOURCE" rev-parse "$SOURCE_COMMIT:$path")" == "$expected" ]] || { echo "blob mismatch: $path" >&2; exit 1; }
done
mkdir -p "$BUILD_DIR/logs" "$BUILD_DIR/target"
export CARGO_TARGET_DIR="$BUILD_DIR/target"
if $LIVE; then
  cmake -S "$ROOT_DIR/tools/emel-kernel-target-rope-reference" -B "$BUILD_DIR/cpp-build" -G Ninja -DEMEL_CPP_SOURCE_DIR="$EMEL_CPP_SOURCE" -DFETCHCONTENT_SOURCE_DIR_STATEFORWARD_SML="$SML_SOURCE" -DCMAKE_BUILD_TYPE=Release >"$BUILD_DIR/logs/cmake.log" 2>&1
  cmake --build "$BUILD_DIR/cpp-build" --target emel-kernel-target-rope-reference >"$BUILD_DIR/logs/cmake-build.log" 2>&1
  cargo build --manifest-path "$ROOT_DIR/tools/emel-kernel-target-rope-parity/Cargo.toml" --locked --offline >"$BUILD_DIR/logs/cargo.log" 2>&1
  "$BUILD_DIR/cpp-build/emel-kernel-target-rope-reference" >"$BUILD_DIR/cpp.out" 2>"$BUILD_DIR/logs/cpp.stderr"
  "$CARGO_TARGET_DIR/debug/emel-kernel-target-rope-parity" >"$BUILD_DIR/rust.out" 2>"$BUILD_DIR/logs/rust.stderr"
else
  [[ -f "$BUILD_DIR/cpp.out" && -f "$BUILD_DIR/rust.out" ]] || { echo '--snapshot-only requires existing live outputs' >&2; exit 1; }
fi
diff -u "$BUILD_DIR/cpp.out" "$BUILD_DIR/rust.out" >"$BUILD_DIR/logs/live-diff.log"
{
  printf 'kind=live_split_observer\nresult=live_split_observer_match\nobserver=split_pinned_cpp_and_public_rust_observers\nsource_commit=%s\n' "$SOURCE_COMMIT"
  cat "$BUILD_DIR/rust.out"
  printf 'source_aarch64_sm_span=src/emel/kernel/aarch64/sm.hpp:761-778\nsource_detail_span=src/emel/kernel/detail.hpp:4650-4770\n'
  printf 'reference_execution=pinned_aarch64_sm_process_event\nrust_execution=public_target_aarch64_kernel_process_rope\n'
  printf 'output_sha256=%s\n' "$(shasum -a 256 "$BUILD_DIR/rust.out" | awk '{print $1}')"
  printf 'variants=norm,neox,timestep\nscope=dense_contiguous_f32_rope_and_typed_rejection\nobserver_diff=empty byte-for-byte match\nresiduals=rope_back; non-F32; non-contiguous; full fixture differential\nunsafe=none\ndispatch=run_to_completion\nlive_artifacts=.artifacts/kernel-goal/target-rope-live\n'
} >"$BUILD_DIR/manifest.txt"
if $UPDATE; then install -m 0644 "$BUILD_DIR/manifest.txt" "$SNAPSHOT"; fi
$COMPARE && diff -u "$SNAPSHOT" "$BUILD_DIR/manifest.txt" >"$BUILD_DIR/logs/snapshot-diff.log"
printf 'AArch64 target RoPE live parity passed (emel.cpp %s)\n' "$SOURCE_COMMIT"
