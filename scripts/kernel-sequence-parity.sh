#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EMEL_CPP_SOURCE="${EMEL_CPP_SOURCE_DIR:-$ROOT_DIR/../emel.cpp-worktrees/pinned-843a117386ef17dc5a50549bbfc821074c2141d6}"
ARTIFACT_DIR="${EMEL_KERNEL_SEQUENCE_PARITY_ARTIFACT_DIR:-$ROOT_DIR/.artifacts/kernel-goal/sequence-next}"
SNAPSHOT="${EMEL_KERNEL_SEQUENCE_PARITY_SNAPSHOT:-$ROOT_DIR/snapshots/parity/kernel-sequence/manifest.txt}"
SOURCE_COMMIT=843a117386ef17dc5a50549bbfc821074c2141d6
EVENTS_BLOB=4b3f02fa5ff9c5071d1fc83938cef1b22b4658b9
DETAIL_BLOB=c8a82643eabfe8f2d7883e655955f455794511b0
X86_SM_BLOB=0b4d635ebbd0fbd52dbca8a2345547fb571205c8
AARCH64_SM_BLOB=865a9cc6ba6115382ed043c464f3d62bcd851357
X86_ACTIONS_BLOB=d45558f5eb96950f43c16a09d768cb4f382d6d61
X86_GUARDS_BLOB=cb3dac8253f8417c9b44acff1de414f6d0a3a3cf

UPDATE=true
COMPARE=true
for argument in "$@"; do
  case "$argument" in
    --snapshot) COMPARE=true ;;
    --no-snapshot) COMPARE=false ;;
    --update) UPDATE=true ;;
    --no-update) UPDATE=false ;;
    --snapshot-only) UPDATE=false ;;
    --update-only) UPDATE=true; COMPARE=false ;;
    --help|-h)
      printf '%s\n' 'usage: scripts/kernel-sequence-parity.sh [--snapshot|--no-snapshot] [--update|--no-update] [--snapshot-only|--update-only]'
      exit 0
      ;;
    *) echo "error: unknown argument: $argument" >&2; exit 2 ;;
  esac
done

for command in cargo diff git rg; do
  command -v "$command" >/dev/null 2>&1 || {
    echo "error: required command is missing: $command" >&2
    exit 2
  }
done

mkdir -p "$ARTIFACT_DIR/tmp" "$ARTIFACT_DIR/target" "$ARTIFACT_DIR/logs"
export TMPDIR="${TMPDIR:-$ARTIFACT_DIR/tmp}"
export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-$ARTIFACT_DIR/target}"
mkdir -p "$TMPDIR" "$CARGO_TARGET_DIR"

[[ "$(git -C "$EMEL_CPP_SOURCE" rev-parse HEAD)" == "$SOURCE_COMMIT" ]] || {
  echo "error: emel.cpp is not at pinned commit $SOURCE_COMMIT" >&2; exit 1;
}
[[ "$(git -C "$EMEL_CPP_SOURCE" rev-parse "$SOURCE_COMMIT:src/emel/kernel/events.hpp")" == "$EVENTS_BLOB" ]] || { echo 'error: events blob mismatch' >&2; exit 1; }
[[ "$(git -C "$EMEL_CPP_SOURCE" rev-parse "$SOURCE_COMMIT:src/emel/kernel/detail.hpp")" == "$DETAIL_BLOB" ]] || { echo 'error: detail blob mismatch' >&2; exit 1; }
[[ "$(git -C "$EMEL_CPP_SOURCE" rev-parse "$SOURCE_COMMIT:src/emel/kernel/x86_64/sm.hpp")" == "$X86_SM_BLOB" ]] || { echo 'error: x86 sm blob mismatch' >&2; exit 1; }
[[ "$(git -C "$EMEL_CPP_SOURCE" rev-parse "$SOURCE_COMMIT:src/emel/kernel/aarch64/sm.hpp")" == "$AARCH64_SM_BLOB" ]] || { echo 'error: aarch64 sm blob mismatch' >&2; exit 1; }
[[ "$(git -C "$EMEL_CPP_SOURCE" rev-parse "$SOURCE_COMMIT:src/emel/kernel/x86_64/actions.hpp")" == "$X86_ACTIONS_BLOB" ]] || { echo 'error: x86 actions blob mismatch' >&2; exit 1; }
[[ "$(git -C "$EMEL_CPP_SOURCE" rev-parse "$SOURCE_COMMIT:src/emel/kernel/x86_64/guards.hpp")" == "$X86_GUARDS_BLOB" ]] || { echo 'error: x86 guards blob mismatch' >&2; exit 1; }

if git -C "$EMEL_CPP_SOURCE" show "$SOURCE_COMMIT:src/emel/kernel/x86_64/actions.hpp" | rg -n 'exec_op_(cumsum|repeat|repeat_back|concat)' >"$ARTIFACT_DIR/logs/x86-actions-semantics.log"; then
  echo 'error: pinned x86 actions unexpectedly contains sequence semantics' >&2
  exit 1
fi
if git -C "$EMEL_CPP_SOURCE" show "$SOURCE_COMMIT:src/emel/kernel/x86_64/guards.hpp" | rg -n '(valid|invalid)_op_(cumsum|repeat|repeat_back|concat)' >"$ARTIFACT_DIR/logs/x86-guards-semantics.log"; then
  echo 'error: pinned x86 guards unexpectedly contains sequence semantics' >&2
  exit 1
fi

TMPDIR="$TMPDIR" CARGO_TARGET_DIR="$CARGO_TARGET_DIR" \
  rustup run 1.98.0 cargo test -p emel-kernels --test any sequence_ops --locked --all-features \
  >"$ARTIFACT_DIR/logs/cargo-sequence-test.log" 2>&1

cp "$SNAPSHOT" "$ARTIFACT_DIR/manifest.txt"
if $UPDATE; then
  install -m 0644 "$ARTIFACT_DIR/manifest.txt" "$SNAPSHOT"
fi
$COMPARE && diff -u "$SNAPSHOT" "$ARTIFACT_DIR/manifest.txt" >"$ARTIFACT_DIR/logs/snapshot-diff.log"
printf '%s\n' "Sequence source-contract parity passed (emel.cpp $SOURCE_COMMIT; no pinned executable reference semantics)"
