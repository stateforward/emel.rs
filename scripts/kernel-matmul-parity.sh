#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EMEL_CPP_SOURCE="${EMEL_CPP_SOURCE_DIR:-$ROOT_DIR/../emel.cpp-worktrees/pinned-843a117386ef17dc5a50549bbfc821074c2141d6}"
SML_SOURCE="${EMEL_STATEFORWARD_SML_SOURCE:-$ROOT_DIR/../../sml/sml.cpp-worktrees/pinned-49207123cd3f39767764bae774932cb48623f92f}"
BUILD_DIR="${EMEL_KERNEL_MATMUL_PARITY_BUILD_DIR:-$ROOT_DIR/.artifacts/kernel-goal/matmul-parity}"
SNAPSHOT="${EMEL_KERNEL_MATMUL_PARITY_SNAPSHOT:-$ROOT_DIR/snapshots/parity/kernel-matmul/root-manifest.txt}"
SOURCE_COMMIT=843a117386ef17dc5a50549bbfc821074c2141d6
SML_COMMIT=49207123cd3f39767764bae774932cb48623f92f
DETAIL_BLOB=c8a82643eabfe8f2d7883e655955f455794511b0
UPDATE=true
COMPARE=true

for argument in "$@"; do
  case "$argument" in
    --update) UPDATE=true ;;
    --no-update) UPDATE=false ;;
    --snapshot-only) UPDATE=false ;;
    --help|-h) printf '%s\n' 'usage: scripts/kernel-matmul-parity.sh [--no-update|--snapshot-only]'; exit 0 ;;
    *) printf 'error: unknown argument: %s\n' "$argument" >&2; exit 2 ;;
  esac
done

for command in cargo cmake diff git python3 shasum; do
  command -v "$command" >/dev/null 2>&1 || {
    printf 'error: required command is missing: %s\n' "$command" >&2
    exit 2
  }
done

[[ "$(git -C "$EMEL_CPP_SOURCE" rev-parse HEAD)" == "$SOURCE_COMMIT" ]] || {
  printf 'error: emel.cpp is not at pinned commit %s\n' "$SOURCE_COMMIT" >&2; exit 1;
}
[[ -z "$(git -C "$EMEL_CPP_SOURCE" status --porcelain=v1 --untracked-files=all)" ]] || {
  printf 'error: emel.cpp reference tree is dirty\n' >&2; exit 1;
}
[[ "$(git -C "$SML_SOURCE" rev-parse HEAD)" == "$SML_COMMIT" ]] || {
  printf 'error: stateforward-sml is not at pinned commit %s\n' "$SML_COMMIT" >&2; exit 1;
}
[[ -z "$(git -C "$SML_SOURCE" status --porcelain=v1 --untracked-files=all)" ]] || {
  printf 'error: stateforward-sml reference tree is dirty\n' >&2; exit 1;
}
[[ "$(git -C "$EMEL_CPP_SOURCE" rev-parse "$SOURCE_COMMIT:src/emel/kernel/detail.hpp")" == "$DETAIL_BLOB" ]] || {
  printf 'error: pinned detail.hpp blob mismatch\n' >&2; exit 1;
}

mkdir -p "$BUILD_DIR/tmp" "$BUILD_DIR/target/cargo" "$BUILD_DIR/logs"
export TMPDIR="$BUILD_DIR/tmp"
export CARGO_TARGET_DIR="$BUILD_DIR/target/cargo"

cmake -S "$ROOT_DIR/tools/emel-kernel-matmul-reference" \
  -B "$BUILD_DIR/target/cpp-build" -G Ninja \
  -DEMEL_CPP_SOURCE_DIR="$EMEL_CPP_SOURCE" \
  -DFETCHCONTENT_SOURCE_DIR_STATEFORWARD_SML="$SML_SOURCE" \
  -DCMAKE_BUILD_TYPE=Release -DEMEL_ENABLE_TESTS=OFF -DEMEL_ENABLE_FUZZ=OFF \
  >"$BUILD_DIR/logs/cmake-configure.log" 2>&1
cmake --build "$BUILD_DIR/target/cpp-build" --target emel-kernel-matmul-reference \
  >"$BUILD_DIR/logs/cmake-build.log" 2>&1
cargo build --quiet --offline --locked --manifest-path "$ROOT_DIR/Cargo.toml" \
  -p emel-kernel-matmul-parity >"$BUILD_DIR/logs/cargo-build.log" 2>&1

"$BUILD_DIR/target/cpp-build/emel-kernel-matmul-reference" >"$BUILD_DIR/cpp.out"
"$CARGO_TARGET_DIR/debug/emel-kernel-matmul-parity" >"$BUILD_DIR/rust.out"
diff -u "$BUILD_DIR/cpp.out" "$BUILD_DIR/rust.out" >"$BUILD_DIR/logs/observer-diff.log"

python3 - "$BUILD_DIR/rust.out" <<'PY'
import sys

expected = {
    "kernel-matmul-parity/v1": None,
    "source_repository": "stateforward/emel.cpp",
    "source_commit": "843a117386ef17dc5a50549bbfc821074c2141d6",
    "source_sml_commit": "49207123cd3f39767764bae774932cb48623f92f",
    "source_kernel_events_blob": "4b3f02fa5ff9c5071d1fc83938cef1b22b4658b9",
    "source_kernel_detail_blob": "c8a82643eabfe8f2d7883e655955f455794511b0",
    "source_kernel_x86_sm_blob": "0b4d635ebbd0fbd52dbca8a2345547fb571205c8",
    "source_kernel_x86_guards_blob": "cb3dac8253f8417c9b44acff1de414f6d0a3a3cf",
    "source_kernel_x86_actions_blob": "d45558f5eb96950f43c16a09d768cb4f382d6d61",
    "source_kernel_aarch64_sm_blob": "865a9cc6ba6115382ed043c464f3d62bcd851357",
    "source_kernel_aarch64_guards_blob": "c25714566ec9a02679daef85089544575123408e",
    "source_kernel_aarch64_actions_blob": "267d4f74e6e7498155c8535920322ffef2c02fb6",
    "scope": "f32_public_root_matmul_and_argmax_dense_rank4",
}
cases = {}
metadata = {}
for raw in open(sys.argv[1], encoding="utf-8"):
    line = raw.strip()
    if not line:
        continue
    if line.startswith("case="):
        fields = dict(item.split("=", 1) for item in line.split() if "=" in item)
        name = fields.pop("case")
        cases[name] = fields
    elif "=" in line:
        key, value = line.split("=", 1)
        metadata[key] = None if value == "" else value
    else:
        metadata[line] = None
if metadata != expected:
    raise SystemExit(f"metadata mismatch: {metadata!r}")
if cases != {
    "matrix": {"status": "ok", "output_bits": "40800000,40a00000,41200000,41300000"},
    "argmax": {"status": "ok", "index": "1", "output_bits": "40800000"},
}:
    raise SystemExit(f"case mismatch: {cases!r}")
PY

manifest="$BUILD_DIR/target/manifest.txt"
{
  cat "$BUILD_DIR/rust.out"
  printf 'reference_tool_sha256=%s\n' "$(shasum -a 256 "$ROOT_DIR/tools/emel-kernel-matmul-reference/main.cpp" | awk '{print $1}')"
  printf 'rust_observer_sha256=%s\n' "$(shasum -a 256 "$ROOT_DIR/tools/emel-kernel-matmul-parity/src/main.rs" | awk '{print $1}')"
  printf 'config=Release,cxx20,public-rust-root-matmul-and-argmax,reference-kernel\n'
  printf 'result=match\n'
} >"$manifest"
if $UPDATE; then
  mkdir -p "$(dirname "$SNAPSHOT")"
  install -m 0644 "$manifest" "$SNAPSHOT"
fi
$COMPARE && diff -u "$SNAPSHOT" "$manifest"
printf 'Kernel public matmul parity passed (emel.cpp %s)\n' "$SOURCE_COMMIT"
