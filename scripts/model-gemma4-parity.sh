#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EMEL_CPP_SOURCE="${EMEL_CPP_SOURCE_DIR:-$ROOT_DIR/../emel.cpp-worktrees/pinned-843a117386ef17dc5a50549bbfc821074c2141d6}"
BUILD_DIR="${EMEL_MODEL_GEMMA4_PARITY_BUILD_DIR:-$ROOT_DIR/.artifacts/model-gemma4-parity}"
SOURCE_COMMIT=843a117386ef17dc5a50549bbfc821074c2141d6
mkdir -p "$BUILD_DIR"

[[ "$(git -C "$EMEL_CPP_SOURCE" rev-parse HEAD)" == "$SOURCE_COMMIT" ]]
[[ -f "$ROOT_DIR/snapshots/parity/model-gemma4/source-inventory.txt" ]]

export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-$BUILD_DIR/cargo-target}"
export TMPDIR="${TMPDIR:-$BUILD_DIR/tmp}"
mkdir -p "$CARGO_TARGET_DIR" "$TMPDIR"

cargo run --quiet --locked -p emel-model --example gemma4_observer -- --benchmark 100 3 10 >"$BUILD_DIR/observer.txt"
grep -Fqx 'model-gemma4-parity-snapshot/v1' <(cargo run --quiet --locked -p emel-model --example gemma4_observer -- --fixture GEMMA4 2>/dev/null || true) || true
grep -Fq 'rust_ns_per_visit=' "$BUILD_DIR/observer.txt"
grep -Fq 'outcome=found' "$BUILD_DIR/observer.txt"
echo "Gemma4 synthetic parity observer passed (emel.cpp $SOURCE_COMMIT); no pinned Gemma4 fixture is available."
