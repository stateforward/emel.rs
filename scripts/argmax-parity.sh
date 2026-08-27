#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ARTIFACT_DIR="${EMEL_ARGMAX_PARITY_ARTIFACT_DIR:-$ROOT_DIR/.artifacts/kernel-goal/argmax-parity}"
TMP_DIR="$ARTIFACT_DIR/tmp"
TARGET_DIR="$ARTIFACT_DIR/target"
SNAPSHOT="$ROOT_DIR/snapshots/parity/kernel-argmax-parity/manifest.txt"
SOURCE_DIR="${EMEL_CPP_SOURCE_DIR:-$ROOT_DIR/../emel.cpp-worktrees/pinned-843a117386ef17dc5a50549bbfc821074c2141d6}"
PINNED_COMMIT=843a117386ef17dc5a50549bbfc821074c2141d6
RUN_SNAPSHOT=false
RUN_LIVE=false
RUN_UPDATE=false

usage() {
  cat <<'USAGE'
usage: scripts/argmax-parity.sh [--snapshot|--snapshot-only] [--live] [--update] [--no-live]

--snapshot verifies the checked-in Rust output without the C++ observer.
--snapshot-only is an explicit alias for the checked-in snapshot-only mode.
--live runs independent Rust and pinned C++ observers and compares outputs.
--update refreshes the checked-in snapshot only after a live match.
--no-live disables observer access and may be combined with --snapshot.
USAGE
}

for argument in "$@"; do
  case "$argument" in
    --snapshot) RUN_SNAPSHOT=true ;;
    --snapshot-only) RUN_SNAPSHOT=true; RUN_LIVE=false; RUN_UPDATE=false ;;
    --live) RUN_LIVE=true ;;
    --update) RUN_UPDATE=true ;;
    --no-live) RUN_LIVE=false ;;
    --help|-h) usage; exit 0 ;;
    *) echo "error: unknown argument: $argument" >&2; usage >&2; exit 2 ;;
  esac
done

if ! $RUN_SNAPSHOT && ! $RUN_LIVE && ! $RUN_UPDATE; then
  RUN_SNAPSHOT=true
fi
if $RUN_UPDATE && ! $RUN_LIVE; then
  echo "error: --update requires --live; no observer access occurred" >&2
  exit 2
fi

mkdir -p "$TMP_DIR" "$TARGET_DIR" "$ARTIFACT_DIR/logs"

run_rust() {
  TMPDIR="$TMP_DIR" CARGO_TARGET_DIR="$TARGET_DIR" \
    cargo run -p emel-kernel-argmax-parity --locked \
    >"$ARTIFACT_DIR/logs/rust-observer.log" 2>&1
  sed -n '/^kernel-argmax-parity\//,$p' \
    "$ARTIFACT_DIR/logs/rust-observer.log" >"$TMP_DIR/rust-output.txt"
}

if $RUN_SNAPSHOT; then
  run_rust
  diff -u "$SNAPSHOT" "$TMP_DIR/rust-output.txt" \
    >"$ARTIFACT_DIR/logs/snapshot-diff.log"
fi

if $RUN_LIVE; then
  command -v cmake >/dev/null 2>&1 || {
    echo "error: required command is missing: cmake" >&2
    exit 2
  }
  git -C "$SOURCE_DIR" rev-parse --git-dir >/dev/null 2>&1 || {
    echo "error: reference source is not a git checkout: $SOURCE_DIR" >&2
    exit 1
  }
  [[ "$(git -C "$SOURCE_DIR" rev-parse HEAD)" == "$PINNED_COMMIT" ]] || {
    echo "error: reference checkout is not at pinned commit $PINNED_COMMIT" >&2
    exit 1
  }
  [[ -z "$(git -C "$SOURCE_DIR" status --porcelain=v1 --untracked-files=all)" ]] || {
    echo "error: reference checkout is dirty; refusing to compile mutable source" >&2
    exit 1
  }
  if ! git -C "$SOURCE_DIR" cat-file -e \
      "$PINNED_COMMIT:src/emel/kernel/detail.hpp"; then
    echo "error: pinned emel.cpp detail source is unavailable" >&2
    exit 1
  fi
  if [[ "$(git -C "$SOURCE_DIR" rev-parse \
      "$PINNED_COMMIT:src/emel/kernel/detail.hpp")" != \
      c8a82643eabfe8f2d7883e655955f455794511b0 || \
      "$(git -C "$SOURCE_DIR" rev-parse \
      "$PINNED_COMMIT:src/emel/kernel/events.hpp")" != \
      4b3f02fa5ff9c5071d1fc83938cef1b22b4658b9 ]]; then
    echo "error: pinned emel.cpp source identity drifted" >&2
    exit 1
  fi
  run_rust
  cmake -S "$ROOT_DIR/tools/emel-kernel-argmax-reference" \
    -B "$TARGET_DIR/reference-build" -DEMEL_CPP_SOURCE_DIR="$SOURCE_DIR" \
    >"$ARTIFACT_DIR/logs/cmake-configure.log" 2>&1
  cmake --build "$TARGET_DIR/reference-build" --parallel 2 \
    >"$ARTIFACT_DIR/logs/cmake-build.log" 2>&1
  "$TARGET_DIR/reference-build/emel-kernel-argmax-reference" \
    >"$ARTIFACT_DIR/logs/cpp-observer.log" 2>&1
  cp "$ARTIFACT_DIR/logs/cpp-observer.log" "$TMP_DIR/cpp-output.txt"
  diff -u "$TMP_DIR/cpp-output.txt" "$TMP_DIR/rust-output.txt" \
    >"$ARTIFACT_DIR/logs/live-diff.log"
  if $RUN_UPDATE; then
    cp "$TMP_DIR/rust-output.txt" "$SNAPSHOT"
  fi
fi

echo "argmax parity passed snapshot=$RUN_SNAPSHOT live=$RUN_LIVE update=$RUN_UPDATE"
