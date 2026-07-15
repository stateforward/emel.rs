#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TEMP_DIR="$(mktemp -d "${TMPDIR:-/tmp}/emel-bench-io-mmap.XXXXXX")"
trap 'rm -rf "$TEMP_DIR"' EXIT

ARTIFACT="$TEMP_DIR/artifact.txt"
VALIDATOR="$ROOT_DIR/scripts/validate_io_mmap_bench.sh"

write_artifact() {
  local first="$1"
  local second="$2"
  local extra="${3:-}"
  {
    printf '%s\n' '# source_repository: stateforward/emel.cpp'
    printf '%s\n' '# source_commit: 843a117386ef17dc5a50549bbfc821074c2141d6'
    printf '%s\n' '# source_tree: ff00a9978b00b4ea1e5268d2ecd483d4855b6eaa'
    printf '%s\n' '# benchmark_fixture: public file-backed mmap lifecycle, file_bytes=1048576 pattern=incrementing_u8 offset=0 cases=16384,1048576'
    printf '%s\n' '# benchmark_operations: source open, map, full immutable access checksum, sequential advice, will-need advice, don'\''t-need advice, release'
    printf '%s\n' '# benchmark_validation: typed outcomes and handle length checked each iteration, full mapped FNV-1a equals precomputed fixture checksum'
    printf '%s\n' '# native_semantics_complete: true'
    printf '%s\n' '# missing_native_semantics: none'
    printf '%s\n' '# contract_delta: rust_mmap_source_capability'
    printf '%s\n' "$first" "$second"
    if [[ -n "$extra" ]]; then
      printf '%s\n' "$extra"
    fi
  } >"$ARTIFACT"
}

expect_failure() {
  local label="$1"
  if "$VALIDATOR" "$ARTIFACT" "$label" 1000 5 >/dev/null 2>&1; then
    echo "error: expected mmap artifact rejection: $label" >&2
    exit 1
  fi
}

FIRST='io/mmap/lifecycle_16kib rust_ns_per_op=10 cpp_ns_per_op=20 rust_vs_cpp_ratio=0.5 iter=1000 runs=5'
SECOND='io/mmap/lifecycle_1mib rust_ns_per_op=30 cpp_ns_per_op=40 rust_vs_cpp_ratio=0.75 iter=1000 runs=5'

write_artifact "$FIRST" "$SECOND"
"$VALIDATOR" "$ARTIFACT" valid 1000 5

invalid_seconds=(
  'io/mmap/lifecycle_1mib rust_ns_per_op=30 cpp_ns_per_op=40 rust_vs_cpp_ratio=0.75 iter=1000'
  'io/mmap/lifecycle_1mib rust_ns_per_op=30 cpp_ns_per_op=40 cpp_ns_per_op=40 rust_vs_cpp_ratio=0.75 iter=1000 runs=5'
  'io/mmap/lifecycle_1mib rust_ns_per_op=bad cpp_ns_per_op=40 rust_vs_cpp_ratio=0.75 iter=1000 runs=5'
  'io/mmap/lifecycle_1mib rust_ns_per_op=30 cpp_ns_per_op=40 rust_vs_cpp_ratio=0.50 iter=1000 runs=5'
  'io/mmap/lifecycle_1mib rust_ns_per_op=30 cpp_ns_per_op=40 rust_vs_cpp_ratio=0.75 iter=999 runs=5'
  'io/mmap/lifecycle_1mib rust_ns_per_op=30 cpp_ns_per_op=40 rust_vs_cpp_ratio=0.75 iter=1000 runs=6'
  'io/mmap/lifecycle_16kib rust_ns_per_op=30 cpp_ns_per_op=40 rust_vs_cpp_ratio=0.75 iter=1000 runs=5'
  'io/mmap/lifecycle_1mib rust_ns_per_op=30 cpp_ns_per_op=40 rust_vs_cpp_ratio=0.75 iter=1000 runs=5 unknown=1'
)

case_number=0
for second in "${invalid_seconds[@]}"; do
  case_number=$((case_number + 1))
  write_artifact "$FIRST" "$second"
  expect_failure "row-$case_number"
done

write_artifact "$FIRST" "$SECOND" '# contract_delta: rust_mmap_source_capability'
expect_failure duplicate-contract

write_artifact "$FIRST" "$SECOND" '# source_commit: attacker'
expect_failure conflicting-source-commit

printf '%s\n' 'I/O mmap benchmark artifact fail-closed tests passed (10 negative paths, 1 valid path)'
