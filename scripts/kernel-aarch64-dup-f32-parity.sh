#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EMEL_CPP_SOURCE="${EMEL_CPP_SOURCE_DIR:-$ROOT_DIR/../emel.cpp-worktrees/pinned-843a117386ef17dc5a50549bbfc821074c2141d6}"
SML_SOURCE="${EMEL_STATEFORWARD_SML_SOURCE:-$ROOT_DIR/../../sml/sml.cpp-worktrees/pinned-49207123cd3f39767764bae774932cb48623f92f}"
BUILD_DIR="${EMEL_AARCH64_DUP_F32_PARITY_ARTIFACT_DIR:-$ROOT_DIR/.artifacts/kernel-goal/aarch64-dup-f32-live}"
SNAPSHOT="${EMEL_AARCH64_DUP_F32_PARITY_SNAPSHOT:-$ROOT_DIR/snapshots/parity/kernel-aarch64-dup-f32-live/manifest.txt}"
SOURCE_COMMIT=843a117386ef17dc5a50549bbfc821074c2141d6
SML_COMMIT=49207123cd3f39767764bae774932cb48623f92f
EVENTS_BLOB=4b3f02fa5ff9c5071d1fc83938cef1b22b4658b9
DETAIL_BLOB=c8a82643eabfe8f2d7883e655955f455794511b0
AARCH64_GUARDS_BLOB=c25714566ec9a02679daef85089544575123408e
AARCH64_ACTIONS_BLOB=267d4f74e6e7498155c8535920322ffef2c02fb6
AARCH64_SM_BLOB=865a9cc6ba6115382ed043c464f3d62bcd851357
LIVE=true
UPDATE=false
COMPARE=false

usage() {
  printf '%s\n' 'usage: scripts/kernel-aarch64-dup-f32-parity.sh [--snapshot|--no-snapshot] [--update|--no-update] [--snapshot-only|--live-only|--update-only]'
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
  cmake -S "$ROOT_DIR/tools/emel-kernel-aarch64-dup-f32-reference" \
    -B "$BUILD_DIR/target/cpp-build" -G Ninja \
    -DEMEL_CPP_SOURCE_DIR="$EMEL_CPP_SOURCE" \
    -DFETCHCONTENT_SOURCE_DIR_STATEFORWARD_SML="$SML_SOURCE" \
    -DCMAKE_BUILD_TYPE=Release -DEMEL_ENABLE_TESTS=OFF -DEMEL_ENABLE_FUZZ=OFF \
    >"$BUILD_DIR/logs/cmake-configure.log" 2>&1
  cmake --build "$BUILD_DIR/target/cpp-build" \
    --target emel-kernel-aarch64-dup-f32-reference \
    >"$BUILD_DIR/logs/cmake-build.log" 2>&1
  cargo build --quiet --offline --locked --manifest-path "$ROOT_DIR/Cargo.toml" \
    -p emel-kernel-aarch64-dup-f32-parity >"$BUILD_DIR/logs/cargo-build.log" 2>&1
  CPP_OBSERVER="$BUILD_DIR/target/cpp-build/emel-kernel-aarch64-dup-f32-reference"
  RUST_OBSERVER="$CARGO_TARGET_DIR/debug/emel-kernel-aarch64-dup-f32-parity"
  [[ -x "$CPP_OBSERVER" && -x "$RUST_OBSERVER" ]]
  "$CPP_OBSERVER" >"$BUILD_DIR/reference.out"
  "$RUST_OBSERVER" >"$BUILD_DIR/rust.out"
else
  [[ -f "$BUILD_DIR/reference.out" && -f "$BUILD_DIR/rust.out" ]] || {
    echo "error: --snapshot-only requires existing observer outputs" >&2
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
    "source_detail_span": "src/emel/kernel/detail.hpp:1535-1555,1700-1748,1920-1926,3788-3791",
    "source_guard_span": "src/emel/kernel/aarch64/guards.hpp:320-329,805-824",
    "source_action_span": "src/emel/kernel/aarch64/actions.hpp:887-930,2217-2237,8447-8479",
    "source_transition_span": "src/emel/kernel/aarch64/sm.hpp:30-43",
    "scope": "dense_f32_dup_vector_tail_zero_count_metadata_rejection",
    "metadata_semantics": "cpp_tensor_view_rejects_unknown_dtype;rust_slice_api_unrepresentable",
}
common_cases = {
    "vector_tail": ("ok", "00000000,bfc00000,40100000,40800000,c1000000,41800000,42000000,c2800000,80000000"),
    "zero_count": ("error", "7fc01234"),
}
cpp_metadata_case = ("error", "7fc01234,7fc01234,7fc01234,7fc01234")

def parse(path):
    metadata = {}
    cases = {}
    extras = []
    for raw in open(path, encoding="utf-8"):
        line = raw.strip()
        if not line:
            continue
        if line == "kernel-aarch64-dup-f32-live/v1":
            metadata["header"] = line
        elif line.startswith("case="):
            fields = dict(item.split("=", 1) for item in line.split() if "=" in item)
            cases[fields["case"]] = (fields["status"], fields["output_bits"])
        elif "=" in line:
            key, value = line.split("=", 1)
            if key == "rust_metadata_case":
                extras.append((key, value))
            else:
                metadata[key] = value
        else:
            raise SystemExit(f"{path}: malformed line: {line}")
    if metadata.pop("header", None) != "kernel-aarch64-dup-f32-live/v1":
        raise SystemExit(f"{path}: malformed header")
    if metadata != metadata_expected:
        raise SystemExit(f"{path}: metadata mismatch: {metadata!r}")
    return cases, extras

cpp_cases, cpp_extras = parse(sys.argv[1])
rust_cases, rust_extras = parse(sys.argv[2])
if {name: cpp_cases.get(name) for name in common_cases} != common_cases:
    raise SystemExit(f"C++ common cases mismatch: {cpp_cases!r}")
if {name: rust_cases.get(name) for name in common_cases} != common_cases:
    raise SystemExit(f"Rust common cases mismatch: {rust_cases!r}")
if cpp_cases.get("invalid_metadata_unknown_dtype") != cpp_metadata_case:
    raise SystemExit(f"C++ metadata rejection mismatch: {cpp_cases!r}")
if rust_extras != [("rust_metadata_case", "unrepresentable_through_typed_dense_f32_slice_api")]:
    raise SystemExit(f"Rust metadata boundary mismatch: {rust_extras!r}")
print("C++/Rust common cases match exact bits; C++ metadata rejection and Rust typed-slice boundary are explicit")
PY

sha256_file() { shasum -a 256 "$1" | awk '{print $1}'; }
{
  cat "$BUILD_DIR/rust.out"
  printf 'reference_metadata_case=invalid_metadata_unknown_dtype status=error output_bits=7fc01234,7fc01234,7fc01234,7fc01234\n'
  printf 'reference_observer_sha256=%s\n' "$(sha256_file "$ROOT_DIR/tools/emel-kernel-aarch64-dup-f32-reference/main.cpp")"
  printf 'rust_observer_sha256=%s\n' "$(sha256_file "$ROOT_DIR/tools/emel-kernel-aarch64-dup-f32-parity/src/main.rs")"
  printf 'config=Release,cxx20,native_aarch64,public_Aarch64Kernel,public_emel_kernel_reference\n'
  printf 'comparison=exact_common_bits_zero_count_and_explicit_metadata_boundary\n'
  printf 'rust_execution=public_Aarch64Kernel_process_event_to_dup_child\n'
  printf 'reference_execution=public_emel_kernel_Kernel_process_event_aarch64_dup_route\n'
  printf 'allocation_hot_path=covered_by_existing_target_dup_allocation_test\n'
  printf 'unsafe=none\n'
  printf 'result=exact_vector_tail_zero_count_and_metadata_boundary\n'
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
echo "AArch64 target F32 dup live parity passed (emel.cpp $SOURCE_COMMIT)"
