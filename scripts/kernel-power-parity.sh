#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EMEL_CPP_SOURCE="${EMEL_CPP_SOURCE_DIR:-$ROOT_DIR/../emel.cpp-worktrees/pinned-843a117386ef17dc5a50549bbfc821074c2141d6}"
SML_SOURCE="${EMEL_STATEFORWARD_SML_SOURCE:-$ROOT_DIR/../../sml/sml.cpp-worktrees/pinned-49207123cd3f39767764bae774932cb48623f92f}"
BUILD_DIR="${EMEL_KERNEL_POWER_PARITY_BUILD_DIR:-$ROOT_DIR/.artifacts/kernel-goal/power-parity}"
SNAPSHOT="${EMEL_KERNEL_POWER_PARITY_SNAPSHOT:-$ROOT_DIR/snapshots/parity/kernel-power/manifest.txt}"
SOURCE_COMMIT=843a117386ef17dc5a50549bbfc821074c2141d6
SML_COMMIT=49207123cd3f39767764bae774932cb48623f92f
EVENTS_BLOB=4b3f02fa5ff9c5071d1fc83938cef1b22b4658b9
DETAIL_BLOB=c8a82643eabfe8f2d7883e655955f455794511b0
X86_SM_BLOB=0b4d635ebbd0fbd52dbca8a2345547fb571205c8
X86_ACTIONS_BLOB=d45558f5eb96950f43c16a09d768cb4f382d6d61
X86_GUARDS_BLOB=cb3dac8253f8417c9b44acff1de414f6d0a3a3cf
UPDATE=true
COMPARE=true
LIVE=true
NO_LIVE=false

usage() {
  cat <<'USAGE'
usage: scripts/kernel-power-parity.sh [OPTIONS]

Runs the pinned C++ and public Rust power observers, compares their fresh
outputs, updates the checked-in snapshot, and verifies it by default.
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
    --snapshot-only) LIVE=true; COMPARE=true; UPDATE=false ;;
    --live-only) LIVE=true; COMPARE=false; UPDATE=false ;;
    --update-only) LIVE=true; COMPARE=false; UPDATE=true ;;
    --help|-h) usage; exit 0 ;;
    *) echo "error: unknown argument: $argument" >&2; usage >&2; exit 2 ;;
  esac
done

for command in cargo cmake diff git c++; do
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
  message="error: explicit --no-live is rejected; power parity requires fresh split observer execution"
  printf '%s\n' "$message" >"$BUILD_DIR/logs/no-live-rejected.log"
  printf '%s\n' "$message" >&2
  exit 2
fi

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
[[ "$(git -C "$EMEL_CPP_SOURCE" rev-parse "$SOURCE_COMMIT:src/emel/kernel/x86_64/actions.hpp")" == "$X86_ACTIONS_BLOB" ]] || {
  echo "error: pinned x86 actions.hpp blob does not match" >&2
  exit 1
}
[[ "$(git -C "$EMEL_CPP_SOURCE" rev-parse "$SOURCE_COMMIT:src/emel/kernel/x86_64/guards.hpp")" == "$X86_GUARDS_BLOB" ]] || {
  echo "error: pinned x86 guards.hpp blob does not match" >&2
  exit 1
}

cmake -S "$ROOT_DIR/tools/emel-kernel-power-reference" \
  -B "$BUILD_DIR/cpp-build" -G Ninja \
  -DEMEL_CPP_SOURCE_DIR="$EMEL_CPP_SOURCE" \
  -DFETCHCONTENT_SOURCE_DIR_STATEFORWARD_SML="$SML_SOURCE" \
  -DCMAKE_BUILD_TYPE=Release -DEMEL_ENABLE_TESTS=OFF -DEMEL_ENABLE_FUZZ=OFF \
  >"$BUILD_DIR/logs/cmake-configure.log" 2>&1
cmake --build "$BUILD_DIR/cpp-build" \
  --target emel-kernel-power-reference >"$BUILD_DIR/logs/cmake-build.log" 2>&1
cargo build --quiet --locked --manifest-path "$ROOT_DIR/Cargo.toml" \
  -p emel-kernel-power-parity >"$BUILD_DIR/logs/cargo-build.log" 2>&1

CPP_OBSERVER="$BUILD_DIR/cpp-build/emel-kernel-power-reference"
RUST_OBSERVER="$CARGO_TARGET_DIR/debug/emel-kernel-power-parity"
[[ -x "$CPP_OBSERVER" && -x "$RUST_OBSERVER" ]]
"$CPP_OBSERVER" >"$BUILD_DIR/cpp.out"
"$RUST_OBSERVER" >"$BUILD_DIR/rust.out"
diff -u "$BUILD_DIR/cpp.out" "$BUILD_DIR/rust.out" \
  >"$BUILD_DIR/logs/observer-diff.log"

{
  cat "$BUILD_DIR/rust.out"
  printf 'reference_observer_sha256=%s\n' \
    "$(sha256_file "$ROOT_DIR/tools/emel-kernel-power-reference/main.cpp")"
  printf 'rust_observer_sha256=%s\n' \
    "$(sha256_file "$ROOT_DIR/tools/emel-kernel-power-parity/src/main.rs")"
  printf 'config=Release,cxx20,public-rust-root-and-child-power-actors,public-cpp-kernel\n'
  printf 'rust_execution=public_Kernel_process_event_to_owned_PowerKernel_for_root_cases;public_PowerKernel_process_event_for_view_cases\n'
  printf 'rust_root_execution_cases=dense_square,count_equal_different_dimensions,dense_square_root,shape_rejection\n'
  printf 'rust_direct_child_cases=strided_square,implicit_contiguous_square\n'
  printf 'reference_execution=public_emel_kernel_Kernel_process_event\n'
  printf 'result=match\n'
} >"$BUILD_DIR/manifest.txt"

if $COMPARE; then
  [[ -f "$SNAPSHOT" ]] || {
    echo "error: required snapshot is missing: $SNAPSHOT" >&2
    exit 1
  }
  diff -u "$SNAPSHOT" "$BUILD_DIR/manifest.txt" \
    >"$BUILD_DIR/logs/snapshot-diff.log"
fi
if $UPDATE; then
  mkdir -p "$(dirname "$SNAPSHOT")"
  install -m 0644 "$BUILD_DIR/manifest.txt" "$SNAPSHOT"
fi
echo "Kernel power parity passed (emel.cpp $SOURCE_COMMIT)"
