#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EMEL_CPP_SOURCE="${EMEL_CPP_SOURCE_DIR:-$ROOT_DIR/../emel.cpp-worktrees/pinned-843a117386ef17dc5a50549bbfc821074c2141d6}"
BUILD_DIR="${EMEL_KERNEL_QUANT_MORE_PARITY_BUILD_DIR:-$ROOT_DIR/.artifacts/kernel-goal/quant-more-live}"
SNAPSHOT="$ROOT_DIR/snapshots/parity/kernel-quant-more/manifest.txt"
SOURCE_COMMIT=843a117386ef17dc5a50549bbfc821074c2141d6
DETAIL_BLOB=c8a82643eabfe8f2d7883e655955f455794511b0
LIVE=true
UPDATE=true
COMPARE=false

for argument in "$@"; do
  case "$argument" in
    --snapshot-only) LIVE=false; UPDATE=false; COMPARE=true ;;
    --live-only) LIVE=true; UPDATE=false; COMPARE=false ;;
    --update-only) LIVE=true; UPDATE=true; COMPARE=false ;;
    --update) UPDATE=true ;;
    --no-update) UPDATE=false ;;
    --snapshot) COMPARE=true ;;
    --no-snapshot) COMPARE=false ;;
    --help|-h) printf '%s\n' 'usage: scripts/kernel-quant-more-parity.sh [--snapshot-only|--live-only|--update-only] [--update|--no-update]'; exit 0 ;;
    *) printf 'error: unknown argument: %s\n' "$argument" >&2; exit 2 ;;
  esac
done

sha256_file() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{print $1}'
  else
    shasum -a 256 "$1" | awk '{print $1}'
  fi
}

for command in cargo cmake diff git; do
  command -v "$command" >/dev/null 2>&1 || { echo "error: required command is missing: $command" >&2; exit 2; }
done

[[ "$(git -C "$EMEL_CPP_SOURCE" rev-parse HEAD)" == "$SOURCE_COMMIT" ]] || {
  echo "error: emel.cpp is not at pinned commit $SOURCE_COMMIT" >&2; exit 1;
}
[[ -z "$(git -C "$EMEL_CPP_SOURCE" status --porcelain=v1 --untracked-files=all)" ]] || {
  echo "error: emel.cpp source checkout is dirty" >&2; exit 1;
}
[[ "$(git -C "$EMEL_CPP_SOURCE" rev-parse "$SOURCE_COMMIT:src/emel/kernel/detail.hpp")" == "$DETAIL_BLOB" ]] || {
  echo "error: pinned detail.hpp blob does not match" >&2; exit 1;
}

mkdir -p "$BUILD_DIR/tmp" "$BUILD_DIR/target/cargo" "$BUILD_DIR/target/cpp-build" "$BUILD_DIR/logs"
export CARGO_TARGET_DIR="$BUILD_DIR/target/cargo"
export TMPDIR="$BUILD_DIR/tmp"

if $LIVE; then
cmake -S "$ROOT_DIR/tools/emel-kernel-quant-more-reference" \
  -B "$BUILD_DIR/target/cpp-build" -G Ninja \
  -DEMEL_CPP_SOURCE_DIR="$EMEL_CPP_SOURCE" \
  -DCMAKE_BUILD_TYPE=Release -DEMEL_ENABLE_TESTS=OFF -DEMEL_ENABLE_FUZZ=OFF >/dev/null
cmake --build "$BUILD_DIR/target/cpp-build" --target emel-kernel-quant-more-reference >/dev/null
cargo build --quiet --locked --manifest-path "$ROOT_DIR/Cargo.toml" -p emel-kernel-quant-more-parity

CXX_OBSERVER="$BUILD_DIR/target/cpp-build/emel-kernel-quant-more-reference"
RUST_OBSERVER="$CARGO_TARGET_DIR/debug/emel-kernel-quant-more-parity"
"$CXX_OBSERVER" >"$BUILD_DIR/logs/cpp.out"
"$RUST_OBSERVER" >"$BUILD_DIR/logs/rust.out"
diff -u "$BUILD_DIR/logs/cpp.out" "$BUILD_DIR/logs/rust.out"

cp "$BUILD_DIR/logs/rust.out" "$BUILD_DIR/live-manifest.txt"
{
  cat "$BUILD_DIR/logs/rust.out"
  printf 'source_detail_blob=%s\n' "$DETAIL_BLOB"
  printf 'rust_surface=crates/emel-kernels/src/any/quant_more.rs\n'
  printf 'rust_tests=crates/emel-kernels/tests/portable/quant_more.rs\n'
  printf 'dispatch=direct_safe_scalar_no_allocation\n'
  printf 'observer=split_live_cpp_and_public_rust\n'
  printf 'result=live_match\n'
  printf 'typed_rejection=Rust observer asserted InvalidBlockLength(Q5K,175) and MismatchedBlockCount(lhs=1,rhs=2)\n'
  printf 'observer_sha256=cpp:%s,rust:%s\n' \
    "$(sha256_file "$ROOT_DIR/tools/emel-kernel-quant-more-reference/main.cpp")" \
    "$(sha256_file "$ROOT_DIR/tools/emel-kernel-quant-more-parity/src/main.rs")"
  printf 'isolated_paths=BUILD_DIR=%s,CARGO_TARGET_DIR=%s,TMPDIR=%s,CMAKE_BUILD=%s\n' \
    "$BUILD_DIR" "$CARGO_TARGET_DIR" "$TMPDIR" "$BUILD_DIR/target/cpp-build"
  printf 'residual=other_quantized_families,target_specialized_simd\n'
} >"$BUILD_DIR/manifest.txt"
else
  [[ -f "$BUILD_DIR/manifest.txt" ]] || { printf 'error: --snapshot-only requires existing manifest %s\n' "$BUILD_DIR/manifest.txt" >&2; exit 1; }
fi
if $UPDATE; then
  mkdir -p "$(dirname "$SNAPSHOT")"
  cp "$BUILD_DIR/manifest.txt" "$SNAPSHOT"
fi
if $COMPARE; then
  [[ -f "$SNAPSHOT" ]] || { printf 'error: missing parity snapshot %s\n' "$SNAPSHOT" >&2; exit 1; }
  diff -u "$SNAPSHOT" "$BUILD_DIR/manifest.txt"
fi
cat "$BUILD_DIR/manifest.txt" >"$BUILD_DIR/report.md"
printf 'proof=live C++ and public Rust observers matched under isolated build paths\n' >>"$BUILD_DIR/report.md"
printf 'source=%s\n' "$EMEL_CPP_SOURCE" >>"$BUILD_DIR/report.md"
printf 'source_commit=%s\n' "$SOURCE_COMMIT" >>"$BUILD_DIR/report.md"
echo "Kernel quant-more live parity passed (emel.cpp $SOURCE_COMMIT)"
