#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EMEL_CPP_SOURCE="${EMEL_CPP_SOURCE_DIR:-$ROOT_DIR/../emel.cpp-worktrees/pinned-843a117386ef17dc5a50549bbfc821074c2141d6}"
SML_SOURCE="${EMEL_STATEFORWARD_SML_SOURCE:-$ROOT_DIR/../../sml/sml.cpp-worktrees/pinned-49207123cd3f39767764bae774932cb48623f92f}"
BUILD_DIR="${EMEL_KERNEL_IM2COL_PARITY_BUILD_DIR:-$ROOT_DIR/.artifacts/kernel-goal/im2col}"
SNAPSHOT="${EMEL_KERNEL_IM2COL_PARITY_SNAPSHOT:-$ROOT_DIR/snapshots/parity/kernel-im2col/manifest.txt}"
SOURCE_COMMIT=843a117386ef17dc5a50549bbfc821074c2141d6
SML_COMMIT=49207123cd3f39767764bae774932cb48623f92f
EVENTS_BLOB=4b3f02fa5ff9c5071d1fc83938cef1b22b4658b9
DETAIL_BLOB=c8a82643eabfe8f2d7883e655955f455794511b0
X86_SM_BLOB=0b4d635ebbd0fbd52dbca8a2345547fb571205c8
UPDATE=true
COMPARE=true
LIVE=true

usage() {
  cat <<'USAGE'
usage: scripts/kernel-im2col-parity.sh [OPTIONS]

Runs the pinned C++ and public Rust im2col observers, compares F32 output bits,
updates the snapshot, and verifies it by default.
Options: --snapshot --no-snapshot --live --no-live --update --no-update
         --snapshot-only --live-only --update-only
USAGE
}

for argument in "$@"; do
  case "$argument" in
    --snapshot) COMPARE=true ;;
    --no-snapshot) COMPARE=false ;;
    --live) LIVE=true ;;
    --no-live) LIVE=false ;;
    --update) UPDATE=true ;;
    --no-update) UPDATE=false ;;
    --snapshot-only) LIVE=false; COMPARE=true; UPDATE=false ;;
    --live-only) LIVE=true; COMPARE=false; UPDATE=false ;;
    --update-only) LIVE=true; COMPARE=false; UPDATE=true ;;
    --help|-h) usage; exit 0 ;;
    *) echo "error: unknown argument: $argument" >&2; usage >&2; exit 2 ;;
  esac
done

for command in cargo cmake diff git; do
  command -v "$command" >/dev/null 2>&1 || {
    echo "error: required command is missing: $command" >&2
    exit 2
  }
done

[[ "$(git -C "$EMEL_CPP_SOURCE" rev-parse HEAD)" == "$SOURCE_COMMIT" ]] || {
  echo "error: emel.cpp is not at pinned commit $SOURCE_COMMIT" >&2
  exit 1
}
[[ -z "$(git -C "$EMEL_CPP_SOURCE" status --porcelain=v1 --untracked-files=all)" ]] || {
  echo "error: emel.cpp source checkout is dirty" >&2
  exit 1
}
[[ "$(git -C "$SML_SOURCE" rev-parse HEAD)" == "$SML_COMMIT" ]] || {
  echo "error: stateforward-sml is not at pinned commit $SML_COMMIT" >&2
  exit 1
}
[[ -z "$(git -C "$SML_SOURCE" status --porcelain=v1 --untracked-files=all)" ]] || {
  echo "error: stateforward-sml source checkout is dirty" >&2
  exit 1
}
[[ "$(git -C "$EMEL_CPP_SOURCE" rev-parse "$SOURCE_COMMIT:src/emel/kernel/events.hpp")" == "$EVENTS_BLOB" ]] || {
  echo "error: pinned events.hpp blob does not match" >&2
  exit 1
}
[[ "$(git -C "$EMEL_CPP_SOURCE" rev-parse "$SOURCE_COMMIT:src/emel/kernel/detail.hpp")" == "$DETAIL_BLOB" ]] || {
  echo "error: pinned detail.hpp blob does not match" >&2
  exit 1
}
[[ "$(git -C "$EMEL_CPP_SOURCE" rev-parse "$SOURCE_COMMIT:src/emel/kernel/x86_64/sm.hpp")" == "$X86_SM_BLOB" ]] || {
  echo "error: pinned x86 sm.hpp blob does not match" >&2
  exit 1
}

mkdir -p "$BUILD_DIR/tmp" "$BUILD_DIR/target" "$BUILD_DIR/logs"
export TMPDIR="$BUILD_DIR/tmp"
export CARGO_TARGET_DIR="$BUILD_DIR/target/cargo"
mkdir -p "$CARGO_TARGET_DIR"

if $LIVE; then
  cmake -S "$ROOT_DIR/tools/emel-kernel-im2col-reference" \
    -B "$BUILD_DIR/target/cpp-build" -G Ninja \
    -DEMEL_CPP_SOURCE_DIR="$EMEL_CPP_SOURCE" \
    -DFETCHCONTENT_SOURCE_DIR_STATEFORWARD_SML="$SML_SOURCE" \
    -DCMAKE_BUILD_TYPE=Release -DEMEL_ENABLE_TESTS=OFF -DEMEL_ENABLE_FUZZ=OFF \
    >"$BUILD_DIR/logs/cmake.log" 2>&1
  cmake --build "$BUILD_DIR/target/cpp-build" \
    --target emel-kernel-im2col-reference >"$BUILD_DIR/logs/cmake-build.log" 2>&1
  cargo build --quiet --offline --locked --manifest-path "$ROOT_DIR/Cargo.toml" \
    -p emel-kernel-im2col-parity >"$BUILD_DIR/logs/cargo-build.log" 2>&1

  CPP_OBSERVER="$BUILD_DIR/target/cpp-build/emel-kernel-im2col-reference"
  RUST_OBSERVER="$CARGO_TARGET_DIR/debug/emel-kernel-im2col-parity"
  [[ -x "$CPP_OBSERVER" && -x "$RUST_OBSERVER" ]]
  "$CPP_OBSERVER" >"$BUILD_DIR/logs/cpp.out"
  "$RUST_OBSERVER" >"$BUILD_DIR/logs/rust.out"
  diff -u "$BUILD_DIR/logs/cpp.out" "$BUILD_DIR/logs/rust.out"
else
  [[ -f "$BUILD_DIR/logs/cpp.out" && -f "$BUILD_DIR/logs/rust.out" ]] || {
    echo "error: --snapshot-only requires prior live observer outputs in $BUILD_DIR/logs" >&2
    exit 2
  }
fi

{
  cat "$BUILD_DIR/logs/rust.out"
  printf 'source_sml_commit=%s\n' "$SML_COMMIT"
  printf 'source_kernel_events_blob=%s\n' "$EVENTS_BLOB"
  printf 'source_kernel_detail_blob=%s\n' "$DETAIL_BLOB"
  printf 'source_kernel_x86_sm_blob=%s\n' "$X86_SM_BLOB"
  printf 'rust_observer_sha256=%s\n' "$(shasum -a 256 "$ROOT_DIR/tools/emel-kernel-im2col-parity/src/main.rs" | awk '{print $1}')"
  printf 'reference_observer_sha256=%s\n' "$(shasum -a 256 "$ROOT_DIR/tools/emel-kernel-im2col-reference/main.cpp" | awk '{print $1}')"
  printf 'config=Release,cxx20,portable-scalar-f32-1d-dense-output,public-rust-root-and-child-im2col-actors,reference-detail\n'
  printf 'rust_execution=public_Kernel_process_event_to_owned_Im2ColKernel\n'
  printf 'reference_execution=direct_detail_run_im2col_as_reference_lane\n'
  printf 'result=source_contract_match_and_public_root_dispatch_equivalence\n'
} >"$BUILD_DIR/target/manifest.txt"

if $COMPARE && [[ -f "$SNAPSHOT" ]]; then
  if ! diff -u "$SNAPSHOT" "$BUILD_DIR/target/manifest.txt"; then
    if ! $UPDATE; then
      echo "error: im2col parity snapshot differs; rerun with --update to accept the new baseline" >&2
      exit 1
    fi
    echo "im2col parity snapshot differs; updating because --update is enabled" >&2
  fi
elif $COMPARE; then
  echo "error: im2col parity snapshot is missing: $SNAPSHOT" >&2
  exit 1
fi

if $UPDATE; then
  mkdir -p "$(dirname "$SNAPSHOT")"
  install -m 0644 "$BUILD_DIR/target/manifest.txt" "$SNAPSHOT"
fi
$COMPARE && diff -u "$SNAPSHOT" "$BUILD_DIR/target/manifest.txt"
echo "Kernel im2col F32 parity passed (emel.cpp $SOURCE_COMMIT)"
