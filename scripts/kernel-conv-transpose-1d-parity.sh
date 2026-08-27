#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EMEL_CPP_SOURCE="${EMEL_CPP_SOURCE_DIR:-$ROOT_DIR/../emel.cpp-worktrees/pinned-843a117386ef17dc5a50549bbfc821074c2141d6}"
SML_SOURCE="${EMEL_STATEFORWARD_SML_SOURCE:-$ROOT_DIR/../../sml/sml.cpp-worktrees/pinned-49207123cd3f39767764bae774932cb48623f92f}"
BUILD_DIR="${EMEL_KERNEL_CONV_TRANSPOSE_1D_PARITY_ARTIFACT_DIR:-$ROOT_DIR/.artifacts/kernel-goal/conv-transpose-1d}"
SNAPSHOT="${EMEL_KERNEL_CONV_TRANSPOSE_1D_PARITY_SNAPSHOT:-$ROOT_DIR/snapshots/parity/kernel-conv-transpose-1d/manifest.txt}"
SOURCE_COMMIT=843a117386ef17dc5a50549bbfc821074c2141d6
SML_COMMIT=49207123cd3f39767764bae774932cb48623f92f
LIVE=true
UPDATE=true
COMPARE=true

usage() {
  cat <<'USAGE'
usage: scripts/kernel-conv-transpose-1d-parity.sh [OPTIONS]

Default mode runs the live C++/Rust comparison, updates the snapshot, and
verifies it. --snapshot-only reruns both observers and verifies the checked-in
snapshot without updating it. --live-only runs the comparison without snapshot
I/O. Any explicit --no-live mode fails closed before reading observer outputs.

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
    --snapshot-only) LIVE=true; UPDATE=false; COMPARE=true ;;
    --live-only) LIVE=true; UPDATE=false; COMPARE=false ;;
    --update-only) LIVE=true; UPDATE=true; COMPARE=false ;;
    --help|-h) usage; exit 0 ;;
    *) echo "error: unknown argument: $argument" >&2; usage >&2; exit 2 ;;
  esac
done

if ! $LIVE; then
  echo "error: observer execution is required; --no-live cannot read or reuse observer outputs" >&2
  exit 2
fi

if [[ ! -d "$SML_SOURCE" && -d "$ROOT_DIR/../../sml/sml.cpp-worktrees/pinned-$SML_COMMIT" ]]; then
  SML_SOURCE="$ROOT_DIR/../../sml/sml.cpp-worktrees/pinned-$SML_COMMIT"
fi

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
[[ "$(git -C "$SML_SOURCE" rev-parse HEAD)" == "$SML_COMMIT" ]] || {
  echo "error: stateforward-sml is not at pinned commit $SML_COMMIT" >&2
  exit 1
}
[[ "$(git -C "$EMEL_CPP_SOURCE" rev-parse "$SOURCE_COMMIT:src/emel/kernel/events.hpp")" == "4b3f02fa5ff9c5071d1fc83938cef1b22b4658b9" ]]
[[ "$(git -C "$EMEL_CPP_SOURCE" rev-parse "$SOURCE_COMMIT:src/emel/kernel/detail.hpp")" == "c8a82643eabfe8f2d7883e655955f455794511b0" ]]
[[ "$(git -C "$EMEL_CPP_SOURCE" rev-parse "$SOURCE_COMMIT:src/emel/kernel/x86_64/sm.hpp")" == "0b4d635ebbd0fbd52dbca8a2345547fb571205c8" ]]
[[ "$(git -C "$EMEL_CPP_SOURCE" rev-parse "$SOURCE_COMMIT:src/emel/kernel/aarch64/sm.hpp")" == "865a9cc6ba6115382ed043c464f3d62bcd851357" ]]

mkdir -p "$BUILD_DIR/tmp" "$BUILD_DIR/target" "$BUILD_DIR/logs"
export TMPDIR="${TMPDIR:-$BUILD_DIR/tmp}"
export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-$BUILD_DIR/target}"
mkdir -p "$TMPDIR" "$CARGO_TARGET_DIR"

if $LIVE; then
  cmake -S "$ROOT_DIR/tools/emel-kernel-conv-transpose-1d-reference" \
    -B "$BUILD_DIR/cpp-build" -G Ninja \
    -DEMEL_CPP_SOURCE_DIR="$EMEL_CPP_SOURCE" \
    -DFETCHCONTENT_SOURCE_DIR_STATEFORWARD_SML="$SML_SOURCE" \
    -DCMAKE_BUILD_TYPE=Release \
    -DEMEL_ENABLE_TESTS=OFF -DEMEL_ENABLE_FUZZ=OFF \
    >"$BUILD_DIR/logs/cmake-configure.log"
  cmake --build "$BUILD_DIR/cpp-build" \
    --target emel-kernel-conv-transpose-1d-reference \
    >"$BUILD_DIR/logs/cmake-build.log"
  cargo build --quiet --locked --manifest-path "$ROOT_DIR/Cargo.toml" \
    -p emel-kernel-conv-transpose-1d-parity \
    >"$BUILD_DIR/logs/cargo-build.log"
  CXX_OBSERVER="$BUILD_DIR/cpp-build/emel-kernel-conv-transpose-1d-reference"
  RUST_OBSERVER="$CARGO_TARGET_DIR/debug/emel-kernel-conv-transpose-1d-parity"
  [[ -x "$CXX_OBSERVER" && -x "$RUST_OBSERVER" ]]
  "$CXX_OBSERVER" >"$BUILD_DIR/cpp.out"
  "$RUST_OBSERVER" >"$BUILD_DIR/rust.out"
  diff -u "$BUILD_DIR/cpp.out" "$BUILD_DIR/rust.out" \
    >"$BUILD_DIR/logs/observer-diff.log"
fi

sha256_file() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{print $1}'
  else
    shasum -a 256 "$1" | awk '{print $1}'
  fi
}

{
  cat "$BUILD_DIR/rust.out"
  printf 'reference_tool_sha256=%s\n' \
    "$(sha256_file "$ROOT_DIR/tools/emel-kernel-conv-transpose-1d-reference/main.cpp")"
  printf 'rust_observer_sha256=%s\n' \
    "$(sha256_file "$ROOT_DIR/tools/emel-kernel-conv-transpose-1d-parity/src/main.rs")"
  printf 'config=Release,cxx20,f32-weights,aligned-nonzero-strides-explicit-dense-output,public-rust-root-Kernel,public-cpp-kernel\n'
  printf 'rust_execution=public_Kernel_process_event\n'
  printf 'reference_execution=public_emel_kernel_Kernel_process_event\n'
  printf 'result=match\n'
} >"$BUILD_DIR/manifest.txt"
if $UPDATE; then
  mkdir -p "$(dirname "$SNAPSHOT")"
  install -m 0644 "$BUILD_DIR/manifest.txt" "$SNAPSHOT"
fi
$COMPARE && diff -u "$SNAPSHOT" "$BUILD_DIR/manifest.txt" \
  >"$BUILD_DIR/logs/snapshot-diff.log"
echo "Kernel F32 conv-transpose-1d parity passed (emel.cpp $SOURCE_COMMIT)"
