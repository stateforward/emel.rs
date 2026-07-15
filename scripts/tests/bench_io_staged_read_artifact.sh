#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TEMP_DIR="$(mktemp -d "${TMPDIR:-/tmp}/emel-bench-io-staged-read.XXXXXX")"
trap 'rm -rf "$TEMP_DIR"' EXIT

ARTIFACT="$TEMP_DIR/artifact.txt"
VALIDATOR="$ROOT_DIR/scripts/validate_io_staged_read_bench.sh"

write_artifact() {
  local first="$1" second="$2" extra="${3:-}"
  {
    printf '%s\n' '# source_repository: stateforward/emel.cpp'
    printf '%s\n' '# source_commit: 843a117386ef17dc5a50549bbfc821074c2141d6'
    printf '%s\n' '# source_tree: ff00a9978b00b4ea1e5268d2ecd483d4855b6eaa'
    printf '%s\n' '# benchmark_fixture: public Stager/StageWindow, immutable source fill=0xa5, caller target cases=16384,1048576, chunk_bytes=4096'
    printf '%s\n' '# benchmark_validation: typed done and synchronous callback checked each iteration, target equals source after measurement'
    printf '%s\n' "$first" "$second"
    [[ -z "$extra" ]] || printf '%s\n' "$extra"
  } >"$ARTIFACT"
}

expect_failure() {
  if bash "$VALIDATOR" "$ARTIFACT" "$1" 1000 5 >/dev/null 2>&1; then
    echo "error: expected staged-read artifact rejection: $1" >&2
    exit 1
  fi
}

FIRST='io/staged-read/copy_16kib rust_ns_per_op=10 cpp_ns_per_op=20 rust_vs_cpp_ratio=0.500000 iter=1000 runs=5'
SECOND='io/staged-read/copy_1mib rust_ns_per_op=30 cpp_ns_per_op=40 rust_vs_cpp_ratio=0.750000 iter=1000 runs=5'
write_artifact "$FIRST" "$SECOND"
bash "$VALIDATOR" "$ARTIFACT" valid 1000 5

invalid_rows=(
  'io/staged-read/copy_1mib rust_ns_per_op=30 cpp_ns_per_op=40 rust_vs_cpp_ratio=0.500000 iter=1000 runs=5'
  'io/staged-read/copy_1mib rust_ns_per_op=bad cpp_ns_per_op=40 rust_vs_cpp_ratio=0.750000 iter=1000 runs=5'
  'io/staged-read/copy_1mib rust_ns_per_op=30 cpp_ns_per_op=40 rust_vs_cpp_ratio=0.750000 iter=999 runs=5'
  'io/staged-read/copy_16kib rust_ns_per_op=30 cpp_ns_per_op=40 rust_vs_cpp_ratio=0.750000 iter=1000 runs=5'
  'io/staged-read/copy_1mib rust_ns_per_op=30 cpp_ns_per_op=40 rust_vs_cpp_ratio=0.750000 iter=1000 runs=5 extra=1'
)
index=0
for second in "${invalid_rows[@]}"; do
  index=$((index + 1))
  write_artifact "$FIRST" "$second"
  expect_failure "row-$index"
done
write_artifact "$FIRST" "$SECOND" '# source_commit: attacker'
expect_failure conflicting-source
printf '%s\n' 'I/O staged-read benchmark artifact fail-closed tests passed (6 negative paths, 1 valid path)'
