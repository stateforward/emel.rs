#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EMEL_CPP_SOURCE="${EMEL_CPP_SOURCE_DIR:-$ROOT_DIR/../emel.cpp-worktrees/pinned-843a117386ef17dc5a50549bbfc821074c2141d6}"
SML_SOURCE="${EMEL_STATEFORWARD_SML_SOURCE:-$ROOT_DIR/../../sml/sml.cpp-worktrees/pinned-49207123cd3f39767764bae774932cb48623f92f}"
BUILD_DIR="${EMEL_KERNEL_FLASH_ATTN_PARITY_ARTIFACT_DIR:-$ROOT_DIR/.artifacts/kernel-goal/flash-root}"
SNAPSHOT="${EMEL_KERNEL_FLASH_ATTN_PARITY_SNAPSHOT:-$ROOT_DIR/snapshots/parity/kernel-flash-attn/root-manifest.txt}"
SOURCE_COMMIT=843a117386ef17dc5a50549bbfc821074c2141d6
SML_COMMIT=49207123cd3f39767764bae774932cb48623f92f
EVENTS_BLOB=4b3f02fa5ff9c5071d1fc83938cef1b22b4658b9
DETAIL_BLOB=c8a82643eabfe8f2d7883e655955f455794511b0
X86_SM_BLOB=0b4d635ebbd0fbd52dbca8a2345547fb571205c8
X86_GUARDS_BLOB=cb3dac8253f8417c9b44acff1de414f6d0a3a3cf
X86_ACTIONS_BLOB=d45558f5eb96950f43c16a09d768cb4f382d6d61
X86_CONTEXT_BLOB=aa41d7d942997a9ac5070bd3eba267b5680d6b84
LIVE=true
UPDATE=true
COMPARE=true

usage() {
  cat <<'USAGE'
usage: scripts/kernel-flash-attn-parity.sh [OPTIONS]

Runs pinned C++ and public Rust FlashAttention observers, compares output,
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
for entry in \
  "src/emel/kernel/events.hpp:$EVENTS_BLOB" \
  "src/emel/kernel/detail.hpp:$DETAIL_BLOB" \
  "src/emel/kernel/x86_64/sm.hpp:$X86_SM_BLOB" \
  "src/emel/kernel/x86_64/guards.hpp:$X86_GUARDS_BLOB" \
  "src/emel/kernel/x86_64/actions.hpp:$X86_ACTIONS_BLOB" \
  "src/emel/kernel/x86_64/context.hpp:$X86_CONTEXT_BLOB"; do
  path="${entry%%:*}"
  expected="${entry##*:}"
  [[ "$(git -C "$EMEL_CPP_SOURCE" rev-parse "$SOURCE_COMMIT:$path")" == "$expected" ]] || {
    echo "error: pinned blob does not match: $path" >&2
    exit 1
  }
done

mkdir -p "$BUILD_DIR/tmp" "$BUILD_DIR/logs"
export TMPDIR="$BUILD_DIR/tmp"
export CARGO_TARGET_DIR="$BUILD_DIR/target/cargo"
mkdir -p "$CARGO_TARGET_DIR"

cmake -S "$ROOT_DIR/tools/emel-kernel-flash-attn-reference" \
  -B "$BUILD_DIR/target/cpp-build" -G Ninja \
  -DEMEL_CPP_SOURCE_DIR="$EMEL_CPP_SOURCE" \
  -DFETCHCONTENT_SOURCE_DIR_STATEFORWARD_SML="$SML_SOURCE" \
  -DCMAKE_BUILD_TYPE=Release -DEMEL_ENABLE_TESTS=OFF -DEMEL_ENABLE_FUZZ=OFF \
  >"$BUILD_DIR/logs/cmake-configure.log" 2>&1
cmake --build "$BUILD_DIR/target/cpp-build" \
  --target emel-kernel-flash-attn-reference >"$BUILD_DIR/logs/cmake-build.log" 2>&1
cargo build --quiet --locked --manifest-path "$ROOT_DIR/Cargo.toml" \
  -p emel-kernel-flash-attn-parity >"$BUILD_DIR/logs/cargo-build.log" 2>&1

CPP_OBSERVER="$BUILD_DIR/target/cpp-build/emel-kernel-flash-attn-reference"
RUST_OBSERVER="$CARGO_TARGET_DIR/debug/emel-kernel-flash-attn-parity"
[[ -x "$CPP_OBSERVER" && -x "$RUST_OBSERVER" ]]
"$CPP_OBSERVER" >"$BUILD_DIR/cpp.out"
"$RUST_OBSERVER" >"$BUILD_DIR/rust.out"
diff -u "$BUILD_DIR/cpp.out" "$BUILD_DIR/rust.out" >"$BUILD_DIR/logs/observer-diff.log"

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
    "$(sha256_file "$ROOT_DIR/tools/emel-kernel-flash-attn-reference/main.cpp")"
  printf 'rust_observer_sha256=%s\n' \
    "$(sha256_file "$ROOT_DIR/tools/emel-kernel-flash-attn-parity/src/main.rs")"
  printf 'config=Release,cxx20,positive-contiguous-f32-f16,public-rust-root-kernel,reference-x86-kernel\n'
  printf 'rust_execution=public_Kernel_process_event_to_owned_FlashAttnKernel\n'
  printf 'reference_execution=public_Kernel_process_event\n'
  printf 'result=match\n'
} >"$BUILD_DIR/manifest.txt"
if $UPDATE; then
  mkdir -p "$(dirname "$SNAPSHOT")"
  install -m 0644 "$BUILD_DIR/manifest.txt" "$SNAPSHOT"
fi
$COMPARE && diff -u "$SNAPSHOT" "$BUILD_DIR/manifest.txt" >"$BUILD_DIR/logs/snapshot-diff.log"
echo "Kernel FlashAttention parity passed (emel.cpp $SOURCE_COMMIT)"
