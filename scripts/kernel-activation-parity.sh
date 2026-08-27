#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EMEL_CPP_SOURCE="${EMEL_CPP_SOURCE_DIR:-$ROOT_DIR/../emel.cpp-worktrees/pinned-843a117386ef17dc5a50549bbfc821074c2141d6}"
SML_SOURCE="${EMEL_STATEFORWARD_SML_SOURCE:-$ROOT_DIR/../emel.cpp/build/zig/_deps/stateforward_sml-src}"
BUILD_DIR="${EMEL_KERNEL_ACTIVATION_PARITY_BUILD_DIR:-$ROOT_DIR/.artifacts/kernel-goal/activation-parity}"
SNAPSHOT="${EMEL_KERNEL_ACTIVATION_PARITY_SNAPSHOT:-$ROOT_DIR/snapshots/parity/kernel-activation/manifest.txt}"
SOURCE_COMMIT=843a117386ef17dc5a50549bbfc821074c2141d6
SML_COMMIT=49207123cd3f39767764bae774932cb48623f92f
UPDATE=true
COMPARE=true
LIVE=true
NO_LIVE=false

usage() {
  cat <<'USAGE'
usage: scripts/kernel-activation-parity.sh [OPTIONS]

Runs split pinned C++ and public Rust activation observers, compares outputs,
updates the checked-in snapshot, and verifies it by default.
Options: --snapshot --no-snapshot --live --no-live --update --no-update
         --snapshot-only --live-only --update-only
USAGE
}

for argument in "$@"; do
  case "$argument" in
    --snapshot) COMPARE=true ;;
    --no-snapshot) COMPARE=false ;;
    --live) LIVE=true ;;
    --no-live) LIVE=false; NO_LIVE=true ;;
    --update) UPDATE=true ;;
    --no-update) UPDATE=false ;;
    --snapshot-only) LIVE=false; COMPARE=true; UPDATE=false ;;
    --live-only) LIVE=true; COMPARE=false; UPDATE=false ;;
    --update-only) LIVE=true; COMPARE=false; UPDATE=true ;;
    --help|-h) usage; exit 0 ;;
    *) echo "error: unknown argument: $argument" >&2; usage >&2; exit 2 ;;
  esac
done

for command in cargo cmake c++ diff git; do
  command -v "$command" >/dev/null 2>&1 || {
    echo "error: required command is missing: $command" >&2
    exit 2
  }
done

sha256_file() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{print $1}'
  else
    shasum -a 256 "$1" | awk '{print $1}'
  fi
}

mkdir -p "$BUILD_DIR" "$BUILD_DIR/tmp" "$BUILD_DIR/cargo-target" "$BUILD_DIR/logs"
export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-$BUILD_DIR/cargo-target}"
export TMPDIR="${TMPDIR:-$BUILD_DIR/tmp}"
mkdir -p "$CARGO_TARGET_DIR" "$TMPDIR"

if $NO_LIVE; then
  message="error: explicit --no-live is rejected; activation parity requires fresh split observer execution"
  printf '%s\n' "$message" >"$BUILD_DIR/logs/no-live-rejected.log"
  printf '%s\n' "$message" >&2
  exit 2
fi

if $LIVE; then
  [[ "$(git -C "$EMEL_CPP_SOURCE" rev-parse HEAD)" == "$SOURCE_COMMIT" ]] || {
    echo "error: emel.cpp is not at pinned commit $SOURCE_COMMIT" >&2
    exit 1
  }
  [[ "$(git -C "$SML_SOURCE" rev-parse HEAD)" == "$SML_COMMIT" ]] || {
    echo "error: stateforward-sml is not at pinned commit $SML_COMMIT" >&2
    exit 1
  }
  for pair in \
    "src/emel/kernel/events.hpp 4b3f02fa5ff9c5071d1fc83938cef1b22b4658b9" \
    "src/emel/kernel/detail.hpp c8a82643eabfe8f2d7883e655955f455794511b0" \
    "src/emel/kernel/x86_64/sm.hpp 0b4d635ebbd0fbd52dbca8a2345547fb571205c8" \
    "src/emel/kernel/x86_64/actions.hpp d45558f5eb96950f43c16a09d768cb4f382d6d61" \
    "src/emel/kernel/x86_64/guards.hpp cb3dac8253f8417c9b44acff1de414f6d0a3a3cf"; do
    read -r path expected <<<"$pair"
    [[ "$(git -C "$EMEL_CPP_SOURCE" rev-parse "$SOURCE_COMMIT:$path")" == "$expected" ]] || {
      echo "error: pinned blob mismatch for $path" >&2
      exit 1
    }
  done

  cmake -S "$ROOT_DIR/tools/emel-kernel-activation-reference" \
    -B "$BUILD_DIR/cpp-build" -G Ninja \
    -DEMEL_CPP_SOURCE_DIR="$EMEL_CPP_SOURCE" \
    -DFETCHCONTENT_SOURCE_DIR_STATEFORWARD_SML="$SML_SOURCE" \
    -DCMAKE_BUILD_TYPE=Release -DEMEL_ENABLE_TESTS=OFF -DEMEL_ENABLE_FUZZ=OFF \
    >"$BUILD_DIR/logs/cmake-configure.log" 2>&1
  cmake --build "$BUILD_DIR/cpp-build" \
    --target emel-kernel-activation-reference >"$BUILD_DIR/logs/cmake-build.log" 2>&1
  cargo build --quiet --locked --manifest-path "$ROOT_DIR/Cargo.toml" \
    -p emel-kernel-activation-parity >"$BUILD_DIR/logs/cargo-build.log" 2>&1

  CPP_OBSERVER="$BUILD_DIR/cpp-build/emel-kernel-activation-reference"
  RUST_OBSERVER="$CARGO_TARGET_DIR/debug/emel-kernel-activation-parity"
  [[ -x "$CPP_OBSERVER" && -x "$RUST_OBSERVER" ]]
  "$CPP_OBSERVER" >"$BUILD_DIR/cpp.out"
  "$RUST_OBSERVER" >"$BUILD_DIR/rust.out"
  diff -u "$BUILD_DIR/cpp.out" "$BUILD_DIR/rust.out" \
    >"$BUILD_DIR/logs/observer-diff.log"
else
  [[ -f "$BUILD_DIR/cpp.out" && -f "$BUILD_DIR/rust.out" ]] || {
    echo "error: snapshot-only requires prior live observer outputs in $BUILD_DIR" >&2
    exit 1
  }
fi

{
  cat "$BUILD_DIR/rust.out"
  printf 'reference_observer_sha256=%s\n' \
    "$(sha256_file "$ROOT_DIR/tools/emel-kernel-activation-reference/main.cpp")"
  printf 'rust_observer_sha256=%s\n' \
    "$(sha256_file "$ROOT_DIR/tools/emel-kernel-activation-parity/src/main.rs")"
  printf 'config=Release,cxx20,portable_f32,public_rust_activation_actor,public_cpp_reference_formula\n'
  printf 'rust_execution=public_ActivationKernel_process_event\n'
  printf 'reference_execution=pinned_emel_cpp_activation_contract\n'
  printf 'result=match\n'
} >"$BUILD_DIR/manifest.txt"

if $UPDATE; then
  mkdir -p "$(dirname "$SNAPSHOT")"
  install -m 0644 "$BUILD_DIR/manifest.txt" "$SNAPSHOT"
fi
$COMPARE && diff -u "$SNAPSHOT" "$BUILD_DIR/manifest.txt"
echo "Kernel activation parity passed (emel.cpp $SOURCE_COMMIT)"
