#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EMEL_CPP_SOURCE="${EMEL_CPP_SOURCE_DIR:-$ROOT_DIR/../emel.cpp-worktrees/pinned-843a117386ef17dc5a50549bbfc821074c2141d6}"
SML_SOURCE="${EMEL_STATEFORWARD_SML_SOURCE:-$ROOT_DIR/../../sml/sml.cpp-worktrees/pinned-49207123cd3f39767764bae774932cb48623f92f}"
BUILD_DIR="${EMEL_AARCH64_F16_MATMUL_PARITY_ARTIFACT_DIR:-$ROOT_DIR/.artifacts/kernel-goal/target-f16-matmul-live}"
SNAPSHOT="${EMEL_AARCH64_F16_MATMUL_PARITY_SNAPSHOT:-$ROOT_DIR/snapshots/parity/kernel-aarch64-f16-matmul-live/manifest.txt}"
SOURCE_COMMIT=843a117386ef17dc5a50549bbfc821074c2141d6
SML_COMMIT=49207123cd3f39767764bae774932cb48623f92f
EVENTS_BLOB=4b3f02fa5ff9c5071d1fc83938cef1b22b4658b9
DETAIL_BLOB=c8a82643eabfe8f2d7883e655955f455794511b0
AARCH64_GUARDS_BLOB=c25714566ec9a02679daef85089544575123408e
AARCH64_ACTIONS_BLOB=267d4f74e6e7498155c8535920322ffef2c02fb6
AARCH64_SM_BLOB=865a9cc6ba6115382ed043c464f3d62bcd851357
UPDATE=false
COMPARE=false
LIVE=true

usage() {
  printf '%s\n' 'usage: scripts/kernel-aarch64-f16-matmul-parity.sh [--snapshot|--no-snapshot] [--update|--no-update] [--snapshot-only|--live-only|--update-only]'
}

for argument in "$@"; do
  case "$argument" in
    --snapshot) COMPARE=true ;;
    --no-snapshot) COMPARE=false ;;
    --update) UPDATE=true ;;
    --no-update) UPDATE=false ;;
    --snapshot-only) LIVE=false; COMPARE=true; UPDATE=false ;;
    --live-only) LIVE=true; COMPARE=false; UPDATE=false ;;
    --update-only) LIVE=true; COMPARE=false; UPDATE=true ;;
    --help|-h) usage; exit 0 ;;
    *) echo "error: unknown argument: $argument" >&2; usage >&2; exit 2 ;;
  esac
done

for command in cargo cmake diff git python3 rustc shasum; do
  command -v "$command" >/dev/null 2>&1 || {
    echo "error: required command is missing: $command" >&2
    exit 2
  }
done

TARGET_ARCH="$(rustc --print cfg | awk -F'"' '/^target_arch=/{print $2; exit}')"
[[ "$TARGET_ARCH" == aarch64 ]] || {
  echo "error: this observer requires a native AArch64 Rust target" >&2
  exit 2
}

for tree in "$EMEL_CPP_SOURCE" "$SML_SOURCE"; do
  [[ -z "$(git -C "$tree" status --porcelain=v1 --untracked-files=all)" ]] || {
    echo "error: reference tree is dirty: $tree" >&2
    exit 1
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

if $LIVE; then
  cmake -S "$ROOT_DIR/tools/emel-kernel-aarch64-f16-matmul-reference" \
    -B "$BUILD_DIR/target/cpp-build" -G Ninja \
    -DEMEL_CPP_SOURCE_DIR="$EMEL_CPP_SOURCE" \
    -DFETCHCONTENT_SOURCE_DIR_STATEFORWARD_SML="$SML_SOURCE" \
    -DCMAKE_BUILD_TYPE=Release -DEMEL_ENABLE_TESTS=OFF -DEMEL_ENABLE_FUZZ=OFF \
    >"$BUILD_DIR/logs/cmake-configure.log" 2>&1
  cmake --build "$BUILD_DIR/target/cpp-build" \
    --target emel-kernel-aarch64-f16-matmul-reference \
    >"$BUILD_DIR/logs/cmake-build.log" 2>&1
  cargo build --quiet --offline --locked --manifest-path "$ROOT_DIR/Cargo.toml" \
    -p emel-kernel-aarch64-f16-matmul-parity >"$BUILD_DIR/logs/cargo-build.log" 2>&1
  CPP_OBSERVER="$BUILD_DIR/target/cpp-build/emel-kernel-aarch64-f16-matmul-reference"
  RUST_OBSERVER="$CARGO_TARGET_DIR/debug/emel-kernel-aarch64-f16-matmul-parity"
  [[ -x "$CPP_OBSERVER" && -x "$RUST_OBSERVER" ]]
  "$CPP_OBSERVER" >"$BUILD_DIR/reference.out"
  "$RUST_OBSERVER" >"$BUILD_DIR/rust.out"
else
  [[ -f "$BUILD_DIR/reference.out" && -f "$BUILD_DIR/rust.out" ]] || {
    echo "error: --snapshot-only requires existing observer output files" >&2
    exit 1
  }
fi

python3 - "$BUILD_DIR/reference.out" "$BUILD_DIR/rust.out" <<'PY' \
  >"$BUILD_DIR/logs/observer-diff.log"
import sys

metadata_expected = {
    "source_repository": "stateforward/emel.cpp",
    "source_commit": "843a117386ef17dc5a50549bbfc821074c2141d6",
    "source_kernel_events_blob": "4b3f02fa5ff9c5071d1fc83938cef1b22b4658b9",
    "source_kernel_detail_blob": "c8a82643eabfe8f2d7883e655955f455794511b0",
    "source_kernel_aarch64_guards_blob": "c25714566ec9a02679daef85089544575123408e",
    "source_kernel_aarch64_actions_blob": "267d4f74e6e7498155c8535920322ffef2c02fb6",
    "source_kernel_aarch64_sm_blob": "865a9cc6ba6115382ed043c464f3d62bcd851357",
    "source_detail_span": "src/emel/kernel/detail.hpp:3874-3894,4198-4214,5089-5096",
    "source_guard_span": "src/emel/kernel/aarch64/guards.hpp:275-285,973-994",
    "source_action_span": "src/emel/kernel/aarch64/actions.hpp:7044-7064,9213-9216",
    "source_transition_span": "src/emel/kernel/aarch64/sm.hpp:525-543",
    "scope": "dense_f16_matmul_target_scalar_positive_shape_dtype_rejection",
}
expected_cases = {
    "matrix": ("ok", "3f800000,40800000,40c00000,41700000"),
    "invalid_shape": ("error", "7fc01234,7fc01234,7fc01234,7fc01234"),
    "invalid_dtype": ("error", "7fc01234,7fc01234,7fc01234,7fc01234"),
}

def parse(path):
    metadata = {}
    cases = {}
    for raw in open(path, encoding="utf-8"):
        line = raw.strip()
        if not line:
            continue
        if line == "kernel-aarch64-f16-matmul-live/v1":
            metadata["header"] = line
            continue
        if line.startswith("case="):
            fields = dict(item.split("=", 1) for item in line.split() if "=" in item)
            if set(fields) != {"case", "status", "output_bits"}:
                raise SystemExit(f"{path}: malformed case line: {line}")
            cases[fields["case"]] = (fields["status"], fields["output_bits"])
            continue
        key, value = line.split("=", 1)
        metadata[key] = value
    if metadata.pop("header", None) != "kernel-aarch64-f16-matmul-live/v1":
        raise SystemExit(f"{path}: missing or malformed header")
    if metadata != metadata_expected:
        raise SystemExit(f"{path}: metadata mismatch: {metadata!r}")
    if cases != expected_cases:
        raise SystemExit(f"{path}: cases mismatch: {cases!r}")
    return metadata, cases

left = parse(sys.argv[1])
right = parse(sys.argv[2])
if left != right:
    raise SystemExit("reference/Rust observer records differ")
print("C++/Rust AArch64 target F16 matmul observers match exact metadata, statuses, bits")
PY

sha256_file() { shasum -a 256 "$1" | awk '{print $1}'; }
{
  cat "$BUILD_DIR/rust.out"
  printf 'reference_observer_sha256=%s\n' "$(sha256_file "$ROOT_DIR/tools/emel-kernel-aarch64-f16-matmul-reference/main.cpp")"
  printf 'rust_observer_sha256=%s\n' "$(sha256_file "$ROOT_DIR/tools/emel-kernel-aarch64-f16-matmul-parity/src/main.rs")"
  printf 'config=Release,cxx20,native_aarch64,target_public_root_actor,direct_pinned_detail_reference\n'
  printf 'comparison=exact_case_status_and_output_bits\n'
  printf 'rust_execution=public_Aarch64Kernel_process_f16_matmul\n'
  printf 'reference_execution=direct_pinned_detail_run_mul_mat_f16_positive_public_emel_kernel_rejections\n'
  printf 'allocation_hot_path=covered_by_existing_target_f16_matmul_allocation_test\n'
  printf 'unsafe=none\n'
  printf 'result=exact_bits_and_rejection_no_mutation\n'
} >"$BUILD_DIR/manifest.txt"

if $COMPARE; then
  [[ -f "$SNAPSHOT" ]] || {
    echo "error: parity snapshot is missing: $SNAPSHOT" >&2
    exit 1
  }
fi
if $UPDATE; then
  mkdir -p "$(dirname "$SNAPSHOT")"
  install -m 0644 "$BUILD_DIR/manifest.txt" "$SNAPSHOT"
fi
if $COMPARE; then
  diff -u "$SNAPSHOT" "$BUILD_DIR/manifest.txt"
fi
echo "AArch64 target F16 matmul live parity passed (emel.cpp $SOURCE_COMMIT)"
