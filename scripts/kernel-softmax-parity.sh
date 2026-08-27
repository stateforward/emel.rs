#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EMEL_CPP_SOURCE="${EMEL_CPP_SOURCE_DIR:-$ROOT_DIR/../emel.cpp-worktrees/pinned-843a117386ef17dc5a50549bbfc821074c2141d6}"
SML_SOURCE="${EMEL_STATEFORWARD_SML_SOURCE:-$ROOT_DIR/../../sml/sml.cpp-worktrees/pinned-49207123cd3f39767764bae774932cb48623f92f}"
BUILD_DIR="${EMEL_KERNEL_SOFTMAX_PARITY_ARTIFACT_DIR:-$ROOT_DIR/.artifacts/kernel-goal/softmax-live}"
SNAPSHOT="$ROOT_DIR/snapshots/parity/kernel-softmax/manifest.txt"
SOURCE_COMMIT=843a117386ef17dc5a50549bbfc821074c2141d6
UPDATE=true

for argument in "$@"; do
  case "$argument" in
    --snapshot-only|--no-update) UPDATE=false ;;
    --update) UPDATE=true ;;
    --help|-h) echo "usage: scripts/kernel-softmax-parity.sh [--snapshot-only|--update|--no-update]"; exit 0 ;;
    *) echo "error: unknown argument: $argument" >&2; exit 2 ;;
  esac
done

for command in cargo cmake diff git c++; do command -v "$command" >/dev/null || exit 2; done
[[ "$(git -C "$EMEL_CPP_SOURCE" rev-parse HEAD)" == "$SOURCE_COMMIT" ]]
[[ "$(git -C "$EMEL_CPP_SOURCE" rev-parse "$SOURCE_COMMIT:src/emel/kernel/sm.hpp")" == cdce31c5f70501f9c26c66886c3348d774a8dc48 ]]
mkdir -p "$BUILD_DIR/target" "$BUILD_DIR/logs"
export CARGO_TARGET_DIR="$BUILD_DIR/target/cargo"
cmake -S "$ROOT_DIR/tools/emel-kernel-softmax-reference" -B "$BUILD_DIR/target/cpp-build" -G Ninja \
  -DEMEL_CPP_SOURCE_DIR="$EMEL_CPP_SOURCE" -DFETCHCONTENT_SOURCE_DIR_STATEFORWARD_SML="$SML_SOURCE" \
  -DCMAKE_BUILD_TYPE=Release >"$BUILD_DIR/logs/cmake.log" 2>&1
cmake --build "$BUILD_DIR/target/cpp-build" --target emel-kernel-softmax-reference >"$BUILD_DIR/logs/cmake-build.log" 2>&1
cargo build --quiet --offline --manifest-path "$ROOT_DIR/Cargo.toml" -p emel-kernel-softmax-parity >"$BUILD_DIR/logs/cargo.log" 2>&1
"$BUILD_DIR/target/cpp-build/emel-kernel-softmax-reference" >"$BUILD_DIR/cpp.out"
"$CARGO_TARGET_DIR/debug/emel-kernel-softmax-parity" >"$BUILD_DIR/rust.out"
diff -u "$BUILD_DIR/cpp.out" "$BUILD_DIR/rust.out" >"$BUILD_DIR/logs/live-diff.log"
cp "$BUILD_DIR/rust.out" "$BUILD_DIR/manifest.txt"
if $UPDATE; then install -m 0644 "$BUILD_DIR/manifest.txt" "$SNAPSHOT"; fi
diff -u "$SNAPSHOT" "$BUILD_DIR/manifest.txt"
cargo clippy --offline --manifest-path "$ROOT_DIR/Cargo.toml" -p emel-kernel-softmax-parity --all-targets -- -D warnings
cargo fmt --all -- --check
git diff --check
echo "Kernel softmax parity passed (emel.cpp $SOURCE_COMMIT)"
