#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EMEL_CPP_SOURCE="${EMEL_CPP_SOURCE_DIR:-$ROOT_DIR/../emel.cpp-worktrees/pinned-843a117386ef17dc5a50549bbfc821074c2141d6}"
SML_SOURCE="${EMEL_STATEFORWARD_SML_SOURCE:-$ROOT_DIR/../../sml/sml.cpp-worktrees/pinned-49207123cd3f39767764bae774932cb48623f92f}"
BUILD_DIR="${EMEL_KERNEL_GET_ROWS_PARITY_ARTIFACT_DIR:-$ROOT_DIR/.artifacts/kernel-goal/get-rows-parity}"
SNAPSHOT="${EMEL_KERNEL_GET_ROWS_PARITY_SNAPSHOT:-$ROOT_DIR/snapshots/parity/kernel-get-rows/root-manifest.txt}"
SOURCE_COMMIT=843a117386ef17dc5a50549bbfc821074c2141d6
SML_COMMIT=49207123cd3f39767764bae774932cb48623f92f
EVENTS_BLOB=4b3f02fa5ff9c5071d1fc83938cef1b22b4658b9
DETAIL_BLOB=c8a82643eabfe8f2d7883e655955f455794511b0
X86_SM_BLOB=0b4d635ebbd0fbd52dbca8a2345547fb571205c8
UPDATE=true
COMPARE=true
LIVE=true

for argument in "$@"; do
  case "$argument" in
    --update) UPDATE=true ;;
    --no-update) UPDATE=false ;;
    --snapshot-only) LIVE=false; UPDATE=false ;;
    --help|-h)
      printf '%s\n' 'usage: scripts/kernel-get-rows-parity.sh [--no-update|--snapshot-only]'
      exit 0
      ;;
    *) printf 'error: unknown argument: %s\n' "$argument" >&2; exit 2 ;;
  esac
done

for command in cargo cmake diff git python3 shasum; do
  command -v "$command" >/dev/null 2>&1 || {
    printf 'error: required command is missing: %s\n' "$command" >&2
    exit 2
  }
done

mkdir -p "$BUILD_DIR/tmp" "$BUILD_DIR/target/cargo" "$BUILD_DIR/logs"
export TMPDIR="$BUILD_DIR/tmp"
export CARGO_TARGET_DIR="$BUILD_DIR/target/cargo"

if $LIVE; then
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
  for entry in \
    "src/emel/kernel/events.hpp:$EVENTS_BLOB" \
    "src/emel/kernel/detail.hpp:$DETAIL_BLOB" \
    "src/emel/kernel/x86_64/sm.hpp:$X86_SM_BLOB"; do
    path="${entry%%:*}"; expected="${entry##*:}"
    [[ "$(git -C "$EMEL_CPP_SOURCE" rev-parse "$SOURCE_COMMIT:$path")" == "$expected" ]] || {
      printf 'error: pinned blob mismatch for %s\n' "$path" >&2; exit 1;
    }
  done

  cmake -S "$ROOT_DIR/tools/emel-kernel-get-rows-reference" \
    -B "$BUILD_DIR/target/cpp-build" -G Ninja \
    -DEMEL_CPP_SOURCE_DIR="$EMEL_CPP_SOURCE" \
    -DFETCHCONTENT_SOURCE_DIR_STATEFORWARD_SML="$SML_SOURCE" \
    -DCMAKE_BUILD_TYPE=Release -DEMEL_ENABLE_TESTS=OFF -DEMEL_ENABLE_FUZZ=OFF \
    >"$BUILD_DIR/logs/cmake-configure.log" 2>&1
  cmake --build "$BUILD_DIR/target/cpp-build" --target emel-kernel-get-rows-reference \
    >"$BUILD_DIR/logs/cmake-build.log" 2>&1
  cargo build --quiet --offline --locked --manifest-path "$ROOT_DIR/Cargo.toml" \
    -p emel-kernel-get-rows-parity >"$BUILD_DIR/logs/cargo-build.log" 2>&1

  "$BUILD_DIR/target/cpp-build/emel-kernel-get-rows-reference" >"$BUILD_DIR/cpp.out"
  "$CARGO_TARGET_DIR/debug/emel-kernel-get-rows-parity" >"$BUILD_DIR/rust.out"
  diff -u "$BUILD_DIR/cpp.out" "$BUILD_DIR/rust.out" >"$BUILD_DIR/logs/observer-diff.log"
else
  [[ -f "$BUILD_DIR/cpp.out" && -f "$BUILD_DIR/rust.out" ]] || {
    printf 'error: --snapshot-only requires prior live observer outputs in %s\n' "$BUILD_DIR" >&2
    exit 1
  }
fi

python3 - "$BUILD_DIR/rust.out" <<'PY'
import sys

expected = {
    "kernel-get-rows-parity/v1": None,
    "source_repository": "stateforward/emel.cpp",
    "source_commit": "843a117386ef17dc5a50549bbfc821074c2141d6",
    "source_kernel_events_blob": "4b3f02fa5ff9c5071d1fc83938cef1b22b4658b9",
    "source_kernel_detail_blob": "c8a82643eabfe8f2d7883e655955f455794511b0",
    "source_kernel_x86_sm_blob": "0b4d635ebbd0fbd52dbca8a2345547fb571205c8",
    "scope": "f32_public_root_get_rows_dense_rank4",
}
metadata = {}
cases = {}
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
if cases != {"rows": {"status": "ok", "output_bits": "40400000,40800000,3f800000,40000000"}}:
    raise SystemExit(f"case mismatch: {cases!r}")
PY

manifest="$BUILD_DIR/target/manifest.txt"
{
  cat "$BUILD_DIR/rust.out"
  printf 'source_sml_commit=%s\n' "$SML_COMMIT"
  printf 'rust_observer_sha256=%s\n' "$(shasum -a 256 "$ROOT_DIR/tools/emel-kernel-get-rows-parity/src/main.rs" | awk '{print $1}')"
  printf 'reference_observer_sha256=%s\n' "$(shasum -a 256 "$ROOT_DIR/tools/emel-kernel-get-rows-reference/main.cpp" | awk '{print $1}')"
  printf 'config=Release,cxx20,public-rust-root-get-rows,reference-detail-run_get_rows_as_f32\n'
  printf 'rust_execution=public_Kernel_process_event_to_owned_GetRowsKernel\n'
  printf 'reference_execution=direct_detail_run_get_rows_as_dtype_f32\n'
  printf 'result=match\n'
} >"$manifest"
if $UPDATE; then
  mkdir -p "$(dirname "$SNAPSHOT")"
  install -m 0644 "$manifest" "$SNAPSHOT"
fi
$COMPARE && diff -u "$SNAPSHOT" "$manifest"
printf 'Kernel public get_rows parity passed (emel.cpp %s)\n' "$SOURCE_COMMIT"
