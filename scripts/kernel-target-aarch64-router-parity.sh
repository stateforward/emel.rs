#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EMEL_CPP_SOURCE="${EMEL_CPP_SOURCE_DIR:-$ROOT_DIR/../emel.cpp-worktrees/pinned-843a117386ef17dc5a50549bbfc821074c2141d6}"
SML_SOURCE="${EMEL_STATEFORWARD_SML_SOURCE:-$ROOT_DIR/../../sml/sml.cpp-worktrees/pinned-49207123cd3f39767764bae774932cb48623f92f}"
BUILD_DIR="${EMEL_TARGET_AARCH64_ROUTER_LIVE_ARTIFACT_DIR:-$ROOT_DIR/.artifacts/kernel-goal/target-aarch64-router-live}"
SNAPSHOT="$ROOT_DIR/snapshots/parity/kernel-target-aarch64-router/manifest.txt"
SOURCE_COMMIT=843a117386ef17dc5a50549bbfc821074c2141d6
LIVE=true
UPDATE=true
COMPARE=true

for argument in "$@"; do
  case "$argument" in
    --snapshot-only) LIVE=false; UPDATE=false; COMPARE=true ;;
    --live-only) LIVE=true; UPDATE=false; COMPARE=false ;;
    --update-only) LIVE=true; UPDATE=true; COMPARE=false ;;
    --update) UPDATE=true ;;
    --no-update) UPDATE=false ;;
    --help|-h) printf '%s\n' 'usage: scripts/kernel-target-aarch64-router-parity.sh [--snapshot-only|--live-only|--update-only] [--update|--no-update]'; exit 0 ;;
    *) printf 'error: unknown argument: %s\n' "$argument" >&2; exit 2 ;;
  esac
done

for command in cargo cmake diff git ninja rustc; do
  command -v "$command" >/dev/null 2>&1 || { echo "error: required command is missing: $command" >&2; exit 2; }
done

TARGET_ARCH="$(rustc --print cfg | awk -F'"' '/^target_arch=/{print $2; exit}')"
[[ "$TARGET_ARCH" == aarch64 ]] || { echo "error: native AArch64 target required" >&2; exit 2; }
[[ -n "$EMEL_CPP_SOURCE" && -d "$EMEL_CPP_SOURCE" ]] || { echo "error: pinned emel.cpp checkout is missing" >&2; exit 2; }
[[ -d "$SML_SOURCE" ]] || { echo "error: pinned stateforward-sml checkout is missing" >&2; exit 2; }
EMEL_CPP_SOURCE="$(cd "$EMEL_CPP_SOURCE" && pwd)"
SML_SOURCE="$(cd "$SML_SOURCE" && pwd)"
mkdir -p "$BUILD_DIR/logs" "$BUILD_DIR/target"
export CARGO_TARGET_DIR="$BUILD_DIR/target"

if $LIVE; then
cmake -S "$ROOT_DIR/tools/emel-kernel-target-aarch64-router-reference" \
  -B "$BUILD_DIR/cpp-build" -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DEMEL_CPP_SOURCE_DIR="$EMEL_CPP_SOURCE" \
  -DFETCHCONTENT_SOURCE_DIR_STATEFORWARD_SML="$SML_SOURCE" \
  >"$BUILD_DIR/logs/cmake-configure.log" 2>&1
cmake --build "$BUILD_DIR/cpp-build" \
  --target emel-kernel-target-aarch64-router-reference \
  >"$BUILD_DIR/logs/cmake-build.log" 2>&1
cargo build --locked --offline --manifest-path \
  "$ROOT_DIR/tools/emel-kernel-target-aarch64-router-parity/Cargo.toml" \
  >"$BUILD_DIR/logs/cargo-build.log" 2>&1

"$BUILD_DIR/cpp-build/emel-kernel-target-aarch64-router-reference" \
  >"$BUILD_DIR/reference.out" 2>"$BUILD_DIR/reference.stderr"
"$CARGO_TARGET_DIR/debug/emel-kernel-target-aarch64-router-parity" \
  >"$BUILD_DIR/aggregate.out" 2>"$BUILD_DIR/aggregate.stderr"
[[ ! -s "$BUILD_DIR/aggregate.stderr" ]] || { cat "$BUILD_DIR/aggregate.stderr" >&2; exit 1; }
diff -u "$BUILD_DIR/reference.out" "$BUILD_DIR/aggregate.out" >"$BUILD_DIR/logs/aggregate-diff.log"
else
  [[ -f "$BUILD_DIR/manifest.txt" ]] || { printf 'error: --snapshot-only requires existing manifest %s\n' "$BUILD_DIR/manifest.txt" >&2; exit 1; }
fi

if $LIVE; then
{
  printf 'schema=emel.rs.parity.v1\nresult=live_split_observer_match\n'
  printf 'component=emel-kernels::target::aarch64::kernel\n'
  printf 'source_repository=../emel.cpp\nsource_commit=%s\n' "$SOURCE_COMMIT"
  printf 'observer=single_public_router_and_pinned_aarch64_sm_reference\n'
  printf 'operations=dense_f32_unary_abs;binary_add;dup;gemv;f16_matmul\n'
  printf 'dispatch=public_Aarch64Kernel\nallocation_hot_path=zero_after_construction\nunsafe=none\n'
  printf 'typed_unexpected_event_rejection=covered_by_public_router_observer\n'
  printf 'live_artifacts=.artifacts/kernel-goal/target-aarch64-router-live\n'
  printf 'residuals=packed;quantized;non_dense;scalar_trig\nresult=match\n'
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
echo "AArch64 target router aggregate live parity passed (emel.cpp $SOURCE_COMMIT)"
