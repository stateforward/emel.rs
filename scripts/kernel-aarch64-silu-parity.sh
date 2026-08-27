#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EMEL_CPP_SOURCE="${EMEL_CPP_SOURCE_DIR:-$ROOT_DIR/../emel.cpp-worktrees/pinned-843a117386ef17dc5a50549bbfc821074c2141d6}"
SML_SOURCE="${EMEL_STATEFORWARD_SML_SOURCE:-$ROOT_DIR/../../sml/sml.cpp-worktrees/pinned-49207123cd3f39767764bae774932cb48623f92f}"
BUILD_DIR="${EMEL_AARCH64_SILU_PARITY_ARTIFACT_DIR:-$ROOT_DIR/.artifacts/kernel-goal/aarch64-silu-live-parity}"
SNAPSHOT="${EMEL_AARCH64_SILU_PARITY_SNAPSHOT:-$ROOT_DIR/snapshots/parity/kernel-aarch64-silu-live/manifest.txt}"
SOURCE_COMMIT=843a117386ef17dc5a50549bbfc821074c2141d6
SML_COMMIT=49207123cd3f39767764bae774932cb48623f92f
EVENTS_BLOB=4b3f02fa5ff9c5071d1fc83938cef1b22b4658b9
DETAIL_BLOB=c8a82643eabfe8f2d7883e655955f455794511b0
AARCH64_GUARDS_BLOB=c25714566ec9a02679daef85089544575123408e
AARCH64_ACTIONS_BLOB=267d4f74e6e7498155c8535920322ffef2c02fb6
AARCH64_SM_BLOB=865a9cc6ba6115382ed043c464f3d62bcd851357
UPDATE=false
COMPARE=false

usage() {
  cat <<'USAGE'
usage: scripts/kernel-aarch64-silu-parity.sh [OPTIONS]

Runs split pinned C++ and public Rust AArch64 SiLU observers, compares exact
output bits for vector, scalar-tail, and invalid/no-mutation cases, and can
write the resulting evidence snapshot. Options:
  --snapshot --no-snapshot --update --no-update --snapshot-only --live-only
USAGE
}

for argument in "$@"; do
  case "$argument" in
    --snapshot) COMPARE=true ;;
    --no-snapshot) COMPARE=false ;;
    --update) UPDATE=true ;;
    --no-update) UPDATE=false ;;
    --snapshot-only) COMPARE=true; UPDATE=false ;;
    --live-only) COMPARE=false; UPDATE=false ;;
    --update-only) COMPARE=false; UPDATE=true ;;
    --help|-h) usage; exit 0 ;;
    *) echo "error: unknown argument: $argument" >&2; usage >&2; exit 2 ;;
  esac
done

for command in cargo cmake diff git shasum; do
  command -v "$command" >/dev/null 2>&1 || {
    echo "error: required command is missing: $command" >&2
    exit 2
  }
done

case "$(uname -m)" in
  arm64|aarch64) ;;
  *) echo "error: live AArch64 execution requires an AArch64 host" >&2; exit 2 ;;
esac

[[ "$(git -C "$EMEL_CPP_SOURCE" rev-parse HEAD)" == "$SOURCE_COMMIT" ]] || {
  echo "error: emel.cpp is not at pinned commit $SOURCE_COMMIT" >&2; exit 1;
}
[[ -z "$(git -C "$EMEL_CPP_SOURCE" status --porcelain=v1 --untracked-files=all)" ]] || {
  echo "error: emel.cpp source checkout is dirty" >&2; exit 1;
}
[[ "$(git -C "$SML_SOURCE" rev-parse HEAD)" == "$SML_COMMIT" ]] || {
  echo "error: stateforward-sml is not at pinned commit $SML_COMMIT" >&2; exit 1;
}
[[ -z "$(git -C "$SML_SOURCE" status --porcelain=v1 --untracked-files=all)" ]] || {
  echo "error: stateforward-sml source checkout is dirty" >&2; exit 1;
}
for entry in \
  "src/emel/kernel/events.hpp:$EVENTS_BLOB" \
  "src/emel/kernel/detail.hpp:$DETAIL_BLOB" \
  "src/emel/kernel/aarch64/guards.hpp:$AARCH64_GUARDS_BLOB" \
  "src/emel/kernel/aarch64/actions.hpp:$AARCH64_ACTIONS_BLOB" \
  "src/emel/kernel/aarch64/sm.hpp:$AARCH64_SM_BLOB"; do
  path="${entry%%:*}"
  expected="${entry##*:}"
  [[ "$(git -C "$EMEL_CPP_SOURCE" rev-parse "$SOURCE_COMMIT:$path")" == "$expected" ]] || {
    echo "error: pinned blob mismatch for $path" >&2
    exit 1
  }
done

mkdir -p "$BUILD_DIR/tmp" "$BUILD_DIR/target" "$BUILD_DIR/logs"
export TMPDIR="$BUILD_DIR/tmp"
export CARGO_TARGET_DIR="$BUILD_DIR/target/cargo"
mkdir -p "$CARGO_TARGET_DIR"

cmake -S "$ROOT_DIR/tools/emel-kernel-aarch64-silu-reference" \
  -B "$BUILD_DIR/target/cpp-build" -G Ninja \
  -DEMEL_CPP_SOURCE_DIR="$EMEL_CPP_SOURCE" \
  -DFETCHCONTENT_SOURCE_DIR_STATEFORWARD_SML="$SML_SOURCE" \
  -DCMAKE_BUILD_TYPE=Release -DEMEL_ENABLE_TESTS=OFF -DEMEL_ENABLE_FUZZ=OFF \
  >"$BUILD_DIR/logs/cmake-configure.log" 2>&1
cmake --build "$BUILD_DIR/target/cpp-build" \
  --target emel-kernel-aarch64-silu-reference \
  >"$BUILD_DIR/logs/cmake-build.log" 2>&1
cargo build --quiet --offline --locked --manifest-path "$ROOT_DIR/Cargo.toml" \
  -p emel-kernel-aarch64-silu-parity >"$BUILD_DIR/logs/cargo-build.log" 2>&1

CPP_OBSERVER="$BUILD_DIR/target/cpp-build/emel-kernel-aarch64-silu-reference"
RUST_OBSERVER="$CARGO_TARGET_DIR/debug/emel-kernel-aarch64-silu-parity"
[[ -x "$CPP_OBSERVER" && -x "$RUST_OBSERVER" ]]
"$CPP_OBSERVER" >"$BUILD_DIR/reference.out"
"$RUST_OBSERVER" >"$BUILD_DIR/rust.out"
diff -u "$BUILD_DIR/reference.out" "$BUILD_DIR/rust.out" \
  >"$BUILD_DIR/logs/observer-diff.log"

sha256_file() { shasum -a 256 "$1" | awk '{print $1}'; }
{
  cat "$BUILD_DIR/rust.out"
  printf 'reference_observer_sha256=%s\n' "$(sha256_file "$ROOT_DIR/tools/emel-kernel-aarch64-silu-reference/main.cpp")"
  printf 'rust_observer_sha256=%s\n' "$(sha256_file "$ROOT_DIR/tools/emel-kernel-aarch64-silu-parity/src/main.rs")"
  printf 'config=Release,cxx20,native_aarch64_neon,public_Aarch64Kernel,public_emel_kernel_Kernel\n'
  printf 'comparison=exact_output_bits_and_invalid_no_mutation\n'
  printf 'rust_execution=public_Aarch64Kernel_process_event_to_target_unary_child\n'
  printf 'reference_execution=public_emel_kernel_Kernel_process_event_to_aarch64_unary_route\n'
  printf 'allocation_hot_path=covered_by_existing_target_router_allocation_test\n'
  printf 'unsafe=none\n'
  printf 'result=exact_bits_match_vector_scalar_tail_invalid_no_mutation\n'
} >"$BUILD_DIR/manifest.txt"

if $COMPARE; then
  [[ -f "$SNAPSHOT" ]] || {
    echo "error: parity snapshot is missing: $SNAPSHOT" >&2
    exit 1
  }
  if ! diff -u "$SNAPSHOT" "$BUILD_DIR/manifest.txt" >"$BUILD_DIR/logs/snapshot-diff.log"; then
    if ! $UPDATE; then
      echo "error: parity snapshot differs; rerun with --update after reviewing live evidence" >&2
      exit 1
    fi
  fi
fi

if $UPDATE; then
  mkdir -p "$(dirname "$SNAPSHOT")"
  install -m 0644 "$BUILD_DIR/manifest.txt" "$SNAPSHOT"
fi

if $COMPARE; then
  diff -u "$SNAPSHOT" "$BUILD_DIR/manifest.txt"
fi
echo "AArch64 SiLU live parity passed (emel.cpp $SOURCE_COMMIT)"
