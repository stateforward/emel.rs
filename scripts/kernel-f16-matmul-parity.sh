#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EMEL_CPP_SOURCE="${EMEL_CPP_SOURCE_DIR:-$ROOT_DIR/../emel.cpp-worktrees/pinned-843a117386ef17dc5a50549bbfc821074c2141d6}"
SML_SOURCE="${EMEL_STATEFORWARD_SML_SOURCE:-$ROOT_DIR/../../sml/sml.cpp-worktrees/pinned-49207123cd3f39767764bae774932cb48623f92f}"
BUILD_DIR="${EMEL_KERNEL_F16_MATMUL_PARITY_ARTIFACT_DIR:-$ROOT_DIR/.artifacts/kernel-goal/f16-matmul}"
SNAPSHOT="${EMEL_KERNEL_F16_MATMUL_PARITY_SNAPSHOT:-$ROOT_DIR/snapshots/parity/kernel-f16-matmul/manifest.txt}"
SOURCE_COMMIT=843a117386ef17dc5a50549bbfc821074c2141d6
SML_COMMIT=49207123cd3f39767764bae774932cb48623f92f
EVENTS_BLOB=4b3f02fa5ff9c5071d1fc83938cef1b22b4658b9
DETAIL_BLOB=c8a82643eabfe8f2d7883e655955f455794511b0
X86_SM_BLOB=0b4d635ebbd0fbd52dbca8a2345547fb571205c8
X86_GUARDS_BLOB=cb3dac8253f8417c9b44acff1de414f6d0a3a3cf
X86_ACTIONS_BLOB=d45558f5eb96950f43c16a09d768cb4f382d6d61
AARCH64_SM_BLOB=865a9cc6ba6115382ed043c464f3d62bcd851357
AARCH64_GUARDS_BLOB=c25714566ec9a02679daef85089544575123408e
AARCH64_ACTIONS_BLOB=267d4f74e6e7498155c8535920322ffef2c02fb6
UPDATE=true
COMPARE=true
LIVE=true
SNAPSHOT_ONLY=false

for argument in "$@"; do
  case "$argument" in
    --snapshot) COMPARE=true ;;
    --no-snapshot) COMPARE=false ;;
    --update) UPDATE=true ;;
    --no-update) UPDATE=false ;;
    --live) LIVE=true ;;
    --no-live) LIVE=false ;;
    --snapshot-only) LIVE=false; SNAPSHOT_ONLY=true; UPDATE=false; COMPARE=true ;;
    --live-only) LIVE=true; UPDATE=false; COMPARE=false ;;
    --update-only) LIVE=true; UPDATE=true; COMPARE=false ;;
    --help|-h)
      printf '%s\n' 'usage: scripts/kernel-f16-matmul-parity.sh [--snapshot|--no-snapshot] [--live|--no-live] [--snapshot-only|--live-only|--update-only]'
      exit 0
      ;;
    *) echo "error: unknown argument: $argument" >&2; exit 2 ;;
  esac
done

for command in cargo cmake diff git python3 rustc shasum; do
  command -v "$command" >/dev/null 2>&1 || {
    echo "error: required command is missing: $command" >&2
    exit 2
  }
done

assert_clean_reference_tree() {
  local name="$1"
  local path="$2"
  local status
  status="$(git -C "$path" status --porcelain=v1 --untracked-files=all)"
  if [[ -n "$status" ]]; then
    printf 'error: %s reference tree is dirty or contains untracked files: %s\n' \
      "$name" "$path" >&2
    printf '%s\n' "$status" >&2
    exit 1
  fi
}

assert_clean_reference_tree "emel.cpp" "$EMEL_CPP_SOURCE"
assert_clean_reference_tree "stateforward-sml" "$SML_SOURCE"

mkdir -p "$BUILD_DIR/tmp" "$BUILD_DIR/target/cargo" "$BUILD_DIR/logs"
export TMPDIR="$BUILD_DIR/tmp"
export CARGO_TARGET_DIR="$BUILD_DIR/target/cargo"

if ! $LIVE && ! $SNAPSHOT_ONLY; then
  message='error: observer execution is required; --no-live is rejected for F16 matmul parity'
  printf '%s\n' "$message" >"$BUILD_DIR/logs/no-live-rejected.log"
  printf '%s\n' "$message" >&2
  exit 2
fi

[[ "$(git -C "$EMEL_CPP_SOURCE" rev-parse HEAD)" == "$SOURCE_COMMIT" ]] || {
  echo "error: emel.cpp is not at pinned commit $SOURCE_COMMIT" >&2; exit 1;
}
[[ "$(git -C "$SML_SOURCE" rev-parse HEAD)" == "$SML_COMMIT" ]] || {
  echo "error: stateforward-sml is not at pinned commit $SML_COMMIT" >&2; exit 1;
}
for entry in \
  "src/emel/kernel/events.hpp:$EVENTS_BLOB" \
  "src/emel/kernel/detail.hpp:$DETAIL_BLOB" \
  "src/emel/kernel/x86_64/sm.hpp:$X86_SM_BLOB" \
  "src/emel/kernel/x86_64/guards.hpp:$X86_GUARDS_BLOB" \
  "src/emel/kernel/x86_64/actions.hpp:$X86_ACTIONS_BLOB" \
  "src/emel/kernel/aarch64/sm.hpp:$AARCH64_SM_BLOB" \
  "src/emel/kernel/aarch64/guards.hpp:$AARCH64_GUARDS_BLOB" \
  "src/emel/kernel/aarch64/actions.hpp:$AARCH64_ACTIONS_BLOB"; do
  path="${entry%%:*}"; expected="${entry##*:}"
  [[ "$(git -C "$EMEL_CPP_SOURCE" rev-parse "$SOURCE_COMMIT:$path")" == "$expected" ]] || {
    echo "error: pinned blob mismatch for $path" >&2; exit 1;
  }
done

TARGET_ARCH="$(rustc --print cfg | awk -F'"' '/^target_arch=/{print $2; exit}')"
case "$TARGET_ARCH" in
  aarch64)
    ACTIVE_SM_BLOB="$AARCH64_SM_BLOB"
    ACTIVE_GUARDS_BLOB="$AARCH64_GUARDS_BLOB"
    ACTIVE_ACTIONS_BLOB="$AARCH64_ACTIONS_BLOB"
    ;;
  x86_64)
    ACTIVE_SM_BLOB="$X86_SM_BLOB"
    ACTIVE_GUARDS_BLOB="$X86_GUARDS_BLOB"
    ACTIVE_ACTIONS_BLOB="$X86_ACTIONS_BLOB"
    ;;
  *)
    echo "error: unsupported observer target architecture: $TARGET_ARCH" >&2
    exit 1
    ;;
esac

if $LIVE; then
  cmake -S "$ROOT_DIR/tools/emel-kernel-f16-matmul-reference" \
    -B "$BUILD_DIR/target/cpp-build" -G Ninja \
    -DEMEL_CPP_SOURCE_DIR="$EMEL_CPP_SOURCE" \
    -DFETCHCONTENT_SOURCE_DIR_STATEFORWARD_SML="$SML_SOURCE" \
    -DCMAKE_BUILD_TYPE=Release -DEMEL_ENABLE_TESTS=OFF -DEMEL_ENABLE_FUZZ=OFF \
    >"$BUILD_DIR/logs/cmake-configure.log" 2>&1
  cmake --build "$BUILD_DIR/target/cpp-build" \
    --target emel-kernel-f16-matmul-reference >"$BUILD_DIR/logs/cmake-build.log" 2>&1
  cargo build --quiet --offline --locked --manifest-path "$ROOT_DIR/Cargo.toml" \
    -p emel-kernel-f16-matmul-parity >"$BUILD_DIR/logs/cargo-build.log" 2>&1

  CPP_OBSERVER="$BUILD_DIR/target/cpp-build/emel-kernel-f16-matmul-reference"
  RUST_OBSERVER="$CARGO_TARGET_DIR/debug/emel-kernel-f16-matmul-parity"
  [[ -x "$CPP_OBSERVER" && -x "$RUST_OBSERVER" ]]
  "$CPP_OBSERVER" >"$BUILD_DIR/cpp.out"
  "$RUST_OBSERVER" >"$BUILD_DIR/rust.out"
else
  [[ -f "$BUILD_DIR/cpp.out" && -f "$BUILD_DIR/rust.out" ]] || {
    echo "error: snapshot-only requires existing observer outputs in $BUILD_DIR" >&2
    exit 1
  }
fi

python3 - "$BUILD_DIR/cpp.out" "$BUILD_DIR/rust.out" <<'PY' >"$BUILD_DIR/logs/observer-diff.log"
import sys

EXPECTED_METADATA = {
    "kernel-f16-matmul-parity/v1": None,
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
    "scope": "dense_explicit_nonzero_stride_f16_by_f16_to_f32_scalar_double_accumulation",
}
EXPECTED_CASES = {
    "matrix": ("ok", "3f800000,40800000,40c00000,41700000"),
    "double_accumulation": ("ok", "47800e98"),
}

def parse(path):
    metadata = {}
    cases = {}
    case_lines = 0
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        if line == "kernel-f16-matmul-parity/v1":
            metadata[line] = None
            continue
        if line.startswith("case="):
            fields = dict(item.split("=", 1) for item in line.split() if "=" in item)
            if set(fields) != {"case", "status", "output_bits"}:
                raise SystemExit(f"{path}: malformed case line: {line}")
            case_lines += 1
            name = fields["case"]
            if name in cases:
                raise SystemExit(f"{path}: duplicate case: {name}")
            cases[name] = (fields["status"], fields["output_bits"])
            continue
        if "=" not in line:
            raise SystemExit(f"{path}: malformed metadata line: {line}")
        key, value = line.split("=", 1)
        if key in metadata:
            raise SystemExit(f"{path}: duplicate metadata: {key}")
        metadata[key] = value
    if case_lines != len(EXPECTED_CASES):
        raise SystemExit(f"{path}: expected exactly {len(EXPECTED_CASES)} cases, found {case_lines}")
    if (
        "kernel-f16-matmul-parity/v1" not in metadata
        or metadata["kernel-f16-matmul-parity/v1"] is not None
    ):
        raise SystemExit(f"{path}: malformed observer header")
    expected_keys = set(EXPECTED_METADATA)
    if set(metadata) != expected_keys:
        raise SystemExit(
            f"{path}: metadata keys differ: {sorted(metadata)} != {sorted(expected_keys)}"
        )
    metadata.pop("kernel-f16-matmul-parity/v1", None)
    for key, expected in EXPECTED_METADATA.items():
        if key == "kernel-f16-matmul-parity/v1":
            continue
        if metadata.get(key) != expected:
            raise SystemExit(f"{path}: metadata {key} != {expected!r}: {metadata.get(key)!r}")
    return metadata, cases

cpp_metadata, cpp_cases = parse(sys.argv[1])
rust_metadata, rust_cases = parse(sys.argv[2])
if cpp_metadata != rust_metadata:
    raise SystemExit(f"observer metadata mismatch: {cpp_metadata!r} != {rust_metadata!r}")
if cpp_cases != EXPECTED_CASES or rust_cases != EXPECTED_CASES:
    raise SystemExit(f"observer cases mismatch: {cpp_cases!r} / {rust_cases!r}")
print("C++/Rust F16 matmul observers match exact metadata, cases, statuses, and output bits")
PY

sha256_file() { shasum -a 256 "$1" | awk '{print $1}'; }
{
  cat "$BUILD_DIR/rust.out"
  printf 'reference_observer_sha256=%s\n' "$(sha256_file "$ROOT_DIR/tools/emel-kernel-f16-matmul-reference/main.cpp")"
  printf 'rust_observer_sha256=%s\n' "$(sha256_file "$ROOT_DIR/tools/emel-kernel-f16-matmul-parity/src/main.rs")"
  printf 'reference_target_arch=%s\n' "$TARGET_ARCH"
  printf 'active_kernel_sm_blob=%s\n' "$ACTIVE_SM_BLOB"
  printf 'active_kernel_guards_blob=%s\n' "$ACTIVE_GUARDS_BLOB"
  printf 'active_kernel_actions_blob=%s\n' "$ACTIVE_ACTIONS_BLOB"
  printf 'config=Release,cxx20,dense_explicit_nonzero_stride_f16_by_f16_to_f32_scalar_double_accumulation,public_rust_root_kernel_f16_matmul_child,public_cpp_reference\n'
  printf 'comparison=exact_case_status_and_output_bits\n'
  printf 'rust_execution=public_Kernel_process_event_to_owned_F16MatmulKernel\n'
  printf 'reference_execution=public_emel_kernel_Kernel_process_event\n'
  printf 'result=exact_bits\n'
} >"$BUILD_DIR/manifest.txt"

if $COMPARE; then
  [[ -f "$SNAPSHOT" ]] || {
    echo "error: missing parity snapshot $SNAPSHOT (use --update-only for an intentional first baseline)" >&2
    exit 1
  }
  diff -u "$SNAPSHOT" "$BUILD_DIR/manifest.txt"
fi
if $UPDATE; then
  mkdir -p "$(dirname "$SNAPSHOT")"
  install -m 0644 "$BUILD_DIR/manifest.txt" "$SNAPSHOT"
fi
echo "F16 matmul parity passed (emel.cpp $SOURCE_COMMIT)"
