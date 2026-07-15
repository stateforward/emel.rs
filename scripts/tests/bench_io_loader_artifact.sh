#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TEMP_DIR="$(mktemp -d "${TMPDIR:-/tmp}/emel-bench-io-loader.XXXXXX")"
trap 'rm -rf "$TEMP_DIR"' EXIT
ARTIFACT="$TEMP_DIR/artifact.txt"
VALIDATOR="$ROOT_DIR/scripts/validate_io_loader_bench.sh"

write_artifact() {
  local row="$1" extra="${2:-}"
  {
    printf '%s\n' '# source_repository: stateforward/emel.cpp'
    printf '%s\n' '# source_commit: 843a117386ef17dc5a50549bbfc821074c2141d6'
    printf '%s\n' '# source_tree: ff00a9978b00b4ea1e5268d2ecd483d4855b6eaa'
    printf '%s\n' '# benchmark_fixture: public Loader/LoadTensor read_copy, immutable source bytes=1048576 fill=0xa5, caller target bytes=1048576'
    printf '%s\n' '# benchmark_validation: typed loader done checked each iteration, target equals source after measurement'
    printf '%s\n' "$row"
    [[ -z "$extra" ]] || printf '%s\n' "$extra"
  } >"$ARTIFACT"
}

expect_failure() {
  if bash "$VALIDATOR" "$ARTIFACT" "$1" 1000 5 >/dev/null 2>&1; then
    echo "error: expected loader artifact rejection: $1" >&2
    exit 1
  fi
}

VALID='io/loader/read_copy_1mib rust_ns_per_op=10 cpp_ns_per_op=20 rust_vs_cpp_ratio=0.500000 iter=1000 runs=5'
write_artifact "$VALID"
bash "$VALIDATOR" "$ARTIFACT" valid 1000 5

invalid_rows=(
  'io/loader/read_copy_1mib rust_ns_per_op=10 cpp_ns_per_op=20 rust_vs_cpp_ratio=0.750000 iter=1000 runs=5'
  'io/loader/read_copy_1mib rust_ns_per_op=10garbage cpp_ns_per_op=20 rust_vs_cpp_ratio=0.500000 iter=1000 runs=5'
  'io/loader/read_copy_1mib rust_ns_per_op=10 cpp_ns_per_op=20 rust_vs_cpp_ratio=0.500000 iter=999 runs=5'
  'io/loader/read_copy_1mib rust_ns_per_op=10 rust_ns_per_op=10 cpp_ns_per_op=20 rust_vs_cpp_ratio=0.500000 iter=1000 runs=5'
  'io/loader/read_copy_1mib rust_ns_per_op=10 cpp_ns_per_op=20 rust_vs_cpp_ratio=0.500000 iter=1000 runs=5 extra=1'
)
index=0
for row in "${invalid_rows[@]}"; do
  index=$((index + 1))
  write_artifact "$row"
  expect_failure "row-$index"
done
write_artifact "$VALID" '# source_commit: attacker'
expect_failure conflicting-source
write_artifact "$VALID" "$VALID"
expect_failure duplicate-case
printf '%s\n' 'I/O loader benchmark artifact fail-closed tests passed (7 negative paths, 1 valid path)'
