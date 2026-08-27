#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EMEL_CPP_SOURCE="${EMEL_CPP_SOURCE_DIR:-$ROOT_DIR/../emel.cpp-worktrees/pinned-843a117386ef17dc5a50549bbfc821074c2141d6}"
SML_SOURCE="${EMEL_STATEFORWARD_SML_SOURCE:-$ROOT_DIR/../../sml/sml.cpp-worktrees/pinned-49207123cd3f39767764bae774932cb48623f92f}"
BUILD_DIR="${EMEL_KERNEL_QUANTIZED_PARITY_BUILD_DIR:-$ROOT_DIR/.artifacts/kernel-goal/quant-k-parity}"
SNAPSHOT="${EMEL_KERNEL_QUANTIZED_PARITY_SNAPSHOT:-$ROOT_DIR/snapshots/parity/kernel-quantized/manifest.txt}"
SOURCE_COMMIT=843a117386ef17dc5a50549bbfc821074c2141d6
SML_COMMIT=49207123cd3f39767764bae774932cb48623f92f
DETAIL_BLOB=c8a82643eabfe8f2d7883e655955f455794511b0
UPDATE=true
COMPARE=true
LIVE=true

usage() {
  cat <<'USAGE'
usage: scripts/kernel-quantized-parity.sh [OPTIONS]

Runs the pinned C++ scalar q4_0/q8_0 and q2_k/q3_k/q4_k/q5_k/q6_k observers and the public Rust quant API,
compares IEEE-754 output bits, updates the snapshot, and verifies it by default.
Options: --snapshot --no-snapshot --live --no-live --update --no-update
         --snapshot-only --live-only --update-only

`--snapshot-only` verifies the checked-in snapshot against prior split observer
outputs without launching the reference build.
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

if $UPDATE && ! $LIVE; then
  echo "error: snapshot updates require fresh observer execution; use --update-only" >&2
  exit 2
fi
for command in cargo cmake diff git; do
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

[[ "$(git -C "$EMEL_CPP_SOURCE" rev-parse HEAD)" == "$SOURCE_COMMIT" ]] || {
  echo "error: emel.cpp is not at pinned commit $SOURCE_COMMIT" >&2
  exit 1
}
[[ -z "$(git -C "$EMEL_CPP_SOURCE" status --porcelain=v1 --untracked-files=all)" ]] || {
  echo "error: emel.cpp source checkout is dirty; refusing to compile a mutable reference" >&2
  exit 1
}
[[ -z "$(git -C "$SML_SOURCE" status --porcelain=v1 --untracked-files=all)" ]] || {
  echo "error: stateforward-sml source checkout is dirty; refusing to configure a mutable reference" >&2
  exit 1
}
[[ "$(git -C "$SML_SOURCE" rev-parse HEAD)" == "$SML_COMMIT" ]] || {
  echo "error: stateforward-sml is not at pinned commit $SML_COMMIT" >&2
  exit 1
}
[[ "$(git -C "$EMEL_CPP_SOURCE" rev-parse "$SOURCE_COMMIT:src/emel/kernel/detail.hpp")" == "$DETAIL_BLOB" ]] || {
  echo "error: pinned detail.hpp blob does not match" >&2
  exit 1
}

mkdir -p "$BUILD_DIR/tmp" "$BUILD_DIR/target" "$BUILD_DIR/logs"
export CARGO_TARGET_DIR="$BUILD_DIR/target/cargo"
export TMPDIR="$BUILD_DIR/tmp"
mkdir -p "$CARGO_TARGET_DIR" "$TMPDIR"

if $LIVE; then
  cmake -S "$ROOT_DIR/tools/emel-kernel-quantized-reference" \
    -B "$BUILD_DIR/target/cpp-build" -G Ninja \
    -DEMEL_CPP_SOURCE_DIR="$EMEL_CPP_SOURCE" \
    -DFETCHCONTENT_SOURCE_DIR_STATEFORWARD_SML="$SML_SOURCE" \
    -DCMAKE_BUILD_TYPE=Release -DEMEL_ENABLE_TESTS=OFF -DEMEL_ENABLE_FUZZ=OFF >/dev/null
  cmake --build "$BUILD_DIR/target/cpp-build" --target emel-kernel-quantized-reference >/dev/null
  cargo build --quiet --locked --manifest-path "$ROOT_DIR/Cargo.toml" \
    -p emel-kernel-quantized-parity
  CXX_OBSERVER="$BUILD_DIR/target/cpp-build/emel-kernel-quantized-reference"
  RUST_OBSERVER="$CARGO_TARGET_DIR/debug/emel-kernel-quantized-parity"
  [[ -x "$CXX_OBSERVER" && -x "$RUST_OBSERVER" ]]
  "$CXX_OBSERVER" >"$BUILD_DIR/logs/cpp.out"
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
  printf 'source_kernel_detail_blob=%s\n' "$DETAIL_BLOB"
  printf 'source_sml_commit=%s\n' "$SML_COMMIT"
  printf 'reference_tool_sha256=%s\n' \
    "$(sha256_file "$ROOT_DIR/tools/emel-kernel-quantized-reference/main.cpp")"
  printf 'rust_observer_sha256=%s\n' \
    "$(sha256_file "$ROOT_DIR/tools/emel-kernel-quantized-parity/src/main.rs")"
  printf 'config=Release,cxx20,scalar-packed-q4_0-q8_0-q5_k-q8_k-q2_k-row-aggregation-block-scalar-reference-q3_k-q4_k-q6_k,public-rust-quant-api,reference-detail\n'
  printf 'result=match\n'
} >"$BUILD_DIR/target/manifest.txt"

if $UPDATE; then
  mkdir -p "$(dirname "$SNAPSHOT")"
  install -m 0644 "$BUILD_DIR/target/manifest.txt" "$SNAPSHOT"
fi
$COMPARE && diff -u "$SNAPSHOT" "$BUILD_DIR/target/manifest.txt"
echo "Kernel scalar quantized parity passed (emel.cpp $SOURCE_COMMIT)"
