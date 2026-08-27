#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EMEL_CPP_SOURCE="${EMEL_CPP_SOURCE_DIR:-$ROOT_DIR/../emel.cpp-worktrees/pinned-843a117386ef17dc5a50549bbfc821074c2141d6}"
SML_SOURCE="${EMEL_STATEFORWARD_SML_SOURCE:-$ROOT_DIR/../../sml/sml.cpp-worktrees/pinned-49207123cd3f39767764bae774932cb48623f92f}"
BUILD_DIR="${EMEL_KERNEL_BROADCAST_PARITY_BUILD_DIR:-$ROOT_DIR/.artifacts/kernel-goal/broadcast-parity}"
SNAPSHOT="${EMEL_KERNEL_BROADCAST_PARITY_SNAPSHOT:-$ROOT_DIR/snapshots/parity/kernel-broadcast/manifest.txt}"
SOURCE_COMMIT=843a117386ef17dc5a50549bbfc821074c2141d6
SML_COMMIT=49207123cd3f39767764bae774932cb48623f92f
UPDATE=true
COMPARE=true
NO_LIVE=false

for argument in "$@"; do
  case "$argument" in
    --snapshot) COMPARE=true ;;
    --no-snapshot) COMPARE=false ;;
    --update) UPDATE=true ;;
    --no-update) UPDATE=false ;;
    --snapshot-only) COMPARE=true; UPDATE=false ;;
    --live-only) COMPARE=false; UPDATE=false ;;
    --update-only) COMPARE=false; UPDATE=true ;;
    --no-live) NO_LIVE=true ;;
    --live) : ;;
    *) echo "error: unknown argument: $argument" >&2; exit 2 ;;
  esac
done

for command in cargo cmake diff git c++; do
  command -v "$command" >/dev/null 2>&1 || { echo "error: required command is missing: $command" >&2; exit 2; }
done

mkdir -p "$BUILD_DIR/tmp" "$BUILD_DIR/cargo-target" "$BUILD_DIR/logs"
export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-$BUILD_DIR/cargo-target}"
export TMPDIR="${TMPDIR:-$BUILD_DIR/tmp}"
mkdir -p "$CARGO_TARGET_DIR" "$TMPDIR"

if $NO_LIVE; then
  message="error: explicit --no-live is rejected; broadcast parity requires fresh split observer execution"
  printf '%s\n' "$message" >"$BUILD_DIR/logs/no-live-rejected.log"
  printf '%s\n' "$message" >&2
  exit 2
fi

[[ "$(git -C "$EMEL_CPP_SOURCE" rev-parse HEAD)" == "$SOURCE_COMMIT" ]] || { echo "error: emel.cpp is not pinned" >&2; exit 1; }
[[ -z "$(git -C "$EMEL_CPP_SOURCE" status --porcelain=v1 --untracked-files=all)" ]] || { echo "error: emel.cpp is dirty" >&2; exit 1; }
[[ "$(git -C "$SML_SOURCE" rev-parse HEAD)" == "$SML_COMMIT" ]] || { echo "error: stateforward-sml is not pinned" >&2; exit 1; }
[[ -z "$(git -C "$SML_SOURCE" status --porcelain=v1 --untracked-files=all)" ]] || { echo "error: stateforward-sml is dirty" >&2; exit 1; }

cmake -S "$ROOT_DIR/tools/emel-kernel-broadcast-reference" -B "$BUILD_DIR/cpp-build" -G Ninja \
  -DEMEL_CPP_SOURCE_DIR="$EMEL_CPP_SOURCE" \
  -DFETCHCONTENT_SOURCE_DIR_STATEFORWARD_SML="$SML_SOURCE" \
  -DCMAKE_BUILD_TYPE=Release -DEMEL_ENABLE_TESTS=OFF -DEMEL_ENABLE_FUZZ=OFF \
  >"$BUILD_DIR/logs/cmake-configure.log" 2>&1
cmake --build "$BUILD_DIR/cpp-build" --target emel-kernel-broadcast-reference \
  >"$BUILD_DIR/logs/cmake-build.log" 2>&1
cargo build --quiet --locked --manifest-path "$ROOT_DIR/Cargo.toml" \
  -p emel-kernel-broadcast-parity >"$BUILD_DIR/logs/cargo-build.log" 2>&1

CPP_OBSERVER="$BUILD_DIR/cpp-build/emel-kernel-broadcast-reference"
RUST_OBSERVER="$CARGO_TARGET_DIR/debug/emel-kernel-broadcast-parity"
[[ -x "$CPP_OBSERVER" && -x "$RUST_OBSERVER" ]]
"$CPP_OBSERVER" >"$BUILD_DIR/cpp.out"
"$RUST_OBSERVER" >"$BUILD_DIR/rust.out"
diff -u "$BUILD_DIR/cpp.out" "$BUILD_DIR/rust.out" >"$BUILD_DIR/logs/observer-diff.log"

{
  cat "$BUILD_DIR/rust.out"
  printf 'config=Release,cxx20,public-rust-broadcast-actor,public-cpp-kernel\n'
  printf 'rust_execution=public_BroadcastKernel_process_event\n'
  printf 'reference_execution=public_emel_kernel_Kernel_process_event\n'
  printf 'result=match\n'
} >"$BUILD_DIR/manifest.txt"

if $UPDATE; then
  mkdir -p "$(dirname "$SNAPSHOT")"
  install -m 0644 "$BUILD_DIR/manifest.txt" "$SNAPSHOT"
fi
$COMPARE && diff -u "$SNAPSHOT" "$BUILD_DIR/manifest.txt"
echo "Kernel broadcast parity passed (emel.cpp $SOURCE_COMMIT)"
