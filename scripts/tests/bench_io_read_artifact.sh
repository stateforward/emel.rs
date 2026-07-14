#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TEMP_DIR="$(mktemp -d "${TMPDIR:-/tmp}/emel-bench-io-read.XXXXXX")"
trap 'rm -rf "$TEMP_DIR"' EXIT

RUNNER="$TEMP_DIR/runner"
BUILD_DIR="$TEMP_DIR/build"
BASELINE_DIR="$TEMP_DIR/baselines"
ARTIFACT="$TEMP_DIR/runner-artifact.txt"
BASELINE="$BASELINE_DIR/io-read-aarch64.txt"

cat >"$RUNNER" <<'RUNNER'
#!/usr/bin/env bash
set -euo pipefail
cat "$EMEL_TEST_BENCH_ARTIFACT"
RUNNER
chmod +x "$RUNNER"

write_artifact() {
  local destination="$1"
  local config="$2"
  local result="$3"
  local arch="${4:-aarch64}"
  local extra_line="${5:-}"
  local pointer_width="${6:-64}"
  mkdir -p "$(dirname "$destination")"
  {
    printf '# bench_host_arch: %s\n' "$arch"
    printf '# bench_pointer_width: %s\n' "$pointer_width"
    printf '%s\n' "$config"
    printf '%s\n' '# source_repository: stateforward/emel.cpp'
    printf '%s\n' '# source_commit: 843a117386ef17dc5a50549bbfc821074c2141d6'
    printf '%s\n' '# source_tree: ff00a9978b00b4ea1e5268d2ecd483d4855b6eaa'
    printf '%s\n' '# benchmark_fixture: public Reader/ReadTensor, immutable source bytes=1048576 fill=0xa5, caller target bytes=1048576'
    printf '%s\n' '# benchmark_validation: typed done checked each iteration, target equals source after measurement'
    printf '%s\n' "$result"
    if [[ -n "$extra_line" ]]; then
      printf '%s\n' "$extra_line"
    fi
  } >"$destination"
}

run_bench() {
  EMEL_BENCH_RUNNER="$RUNNER" \
  EMEL_BENCH_BUILD_DIR="$BUILD_DIR" \
  EMEL_BENCH_BASELINE_DIR="$BASELINE_DIR" \
  EMEL_TEST_BENCH_ARTIFACT="$ARTIFACT" \
    "$ROOT_DIR/scripts/bench.sh" "$@"
}

expect_failure() {
  local label="$1"
  shift
  if "$@" >"$TEMP_DIR/$label.stdout" 2>"$TEMP_DIR/$label.stderr"; then
    echo "error: expected failure: $label" >&2
    exit 1
  fi
}

prove_rejected_artifact() {
  local label="$1"
  local invalid_artifact="$2"
  local before_checksum

  cp "$invalid_artifact" "$ARTIFACT"
  before_checksum="$(cksum "$BASELINE")"
  expect_failure "update-$label" run_bench --snapshot --update --suite=io-read
  if [[ "$(cksum "$BASELINE")" != "$before_checksum" ]]; then
    echo "error: rejected update mutated the isolated baseline: $label" >&2
    exit 1
  fi
  expect_failure "compare-current-$label" run_bench --snapshot --suite=io-read
  if [[ "$(cksum "$BASELINE")" != "$before_checksum" ]]; then
    echo "error: rejected current comparison mutated the isolated baseline: $label" >&2
    exit 1
  fi

  cp "$VALID_ARTIFACT" "$ARTIFACT"
  cp "$invalid_artifact" "$BASELINE"
  before_checksum="$(cksum "$BASELINE")"
  expect_failure "compare-baseline-$label" run_bench --snapshot --suite=io-read
  if [[ "$(cksum "$BASELINE")" != "$before_checksum" ]]; then
    echo "error: rejected baseline comparison mutated the isolated baseline: $label" >&2
    exit 1
  fi
  cp "$VALID_ARTIFACT" "$BASELINE"
}

CONFIG='# benchmark_config: iterations=1000 runs=5 sample_policy=median warmup_iterations=100'
VALID_CASE='io/read/copy_1mib ns_per_op=1.25 iter=1000 runs=5'
VALID_ARTIFACT="$TEMP_DIR/valid-artifact.txt"

write_artifact "$ARTIFACT" "$CONFIG" "$VALID_CASE"
cp "$ARTIFACT" "$VALID_ARTIFACT"
run_bench --snapshot --update --suite=io-read >/dev/null
run_bench --snapshot --suite=io-read >/dev/null
baseline_checksum="$(cksum "$BASELINE")"

invalid_cases=(
  'io/read/copy_1mib ns_per_op=1.25 iter=1000'
  'io/read/copy_1mib ns_per_op=1.25 ns_per_op=2 iter=1000 runs=5'
  'io/read/copy_1mib ns_per_op=bad iter=1000 runs=5'
  'io/read/copy_1mib ns_per_op=1.25 iter=-1 runs=5'
  'io/read/copy_1mib ns_per_op=1.25 iter=1000 runs=5 unknown=value'
  'io/read/copy_1mib ns_per_op=NaN iter=1000 runs=5'
  'io/read/copy_1mib ns_per_op=1e999 iter=1000 runs=5'
  'io/read/copy_1mib ns_per_op=0 iter=1000 runs=5'
  'io/read/copy_1mib ns_per_op=-1 iter=1000 runs=5'
  'io/read/copy_1mib ns_per_op=1.25 iter=999 runs=5'
  'io/read/copy_1mib ns_per_op=1.25 iter=1000 runs=6'
)

case_number=0
for invalid_case in "${invalid_cases[@]}"; do
  case_number=$((case_number + 1))

  write_artifact "$ARTIFACT" "$CONFIG" "$invalid_case"
  expect_failure "update-$case_number" run_bench --snapshot --update --suite=io-read
  if [[ "$(cksum "$BASELINE")" != "$baseline_checksum" ]]; then
    echo "error: failed update mutated the isolated baseline: case $case_number" >&2
    exit 1
  fi
  expect_failure "compare-current-$case_number" run_bench --snapshot --suite=io-read
  if [[ "$(cksum "$BASELINE")" != "$baseline_checksum" ]]; then
    echo "error: failed comparison mutated the isolated baseline: case $case_number" >&2
    exit 1
  fi

  write_artifact "$ARTIFACT" "$CONFIG" "$VALID_CASE"
  write_artifact "$BASELINE" "$CONFIG" "$invalid_case"
  invalid_baseline_checksum="$(cksum "$BASELINE")"
  expect_failure "compare-$case_number" run_bench --snapshot --suite=io-read
  if [[ "$(cksum "$BASELINE")" != "$invalid_baseline_checksum" ]]; then
    echo "error: rejected baseline comparison mutated the isolated baseline: case $case_number" >&2
    exit 1
  fi
  write_artifact "$BASELINE" "$CONFIG" "$VALID_CASE"
done

LARGE_CONFIG='# benchmark_config: iterations=1000 runs=9007199254740993 sample_policy=median warmup_iterations=100'
LARGE_VALID_CASE='io/read/copy_1mib ns_per_op=1.25 iter=1000 runs=9007199254740993'
large_invalid_cases=(
  'io/read/copy_1mib ns_per_op=1.25 iter=1000 runs=9007199254740992'
)

write_artifact "$ARTIFACT" "$LARGE_CONFIG" "$LARGE_VALID_CASE"
run_bench --snapshot --update --suite=io-read >/dev/null
large_baseline_checksum="$(cksum "$BASELINE")"

large_case_number=0
for invalid_case in "${large_invalid_cases[@]}"; do
  large_case_number=$((large_case_number + 1))

  write_artifact "$ARTIFACT" "$LARGE_CONFIG" "$invalid_case"
  expect_failure "update-large-$large_case_number" run_bench --snapshot --update --suite=io-read
  if [[ "$(cksum "$BASELINE")" != "$large_baseline_checksum" ]]; then
    echo "error: failed large-count update mutated the isolated baseline: case $large_case_number" >&2
    exit 1
  fi
  expect_failure "compare-current-large-$large_case_number" run_bench --snapshot --suite=io-read
  if [[ "$(cksum "$BASELINE")" != "$large_baseline_checksum" ]]; then
    echo "error: failed large-count comparison mutated the isolated baseline: case $large_case_number" >&2
    exit 1
  fi

  write_artifact "$ARTIFACT" "$LARGE_CONFIG" "$LARGE_VALID_CASE"
  write_artifact "$BASELINE" "$LARGE_CONFIG" "$invalid_case"
  invalid_large_baseline_checksum="$(cksum "$BASELINE")"
  expect_failure "compare-baseline-large-$large_case_number" run_bench --snapshot --suite=io-read
  if [[ "$(cksum "$BASELINE")" != "$invalid_large_baseline_checksum" ]]; then
    echo "error: rejected large-count baseline comparison mutated the isolated baseline: case $large_case_number" >&2
    exit 1
  fi
  write_artifact "$BASELINE" "$LARGE_CONFIG" "$LARGE_VALID_CASE"
done

write_artifact "$BASELINE" "$CONFIG" "$VALID_CASE"

producer_invalid_artifacts=(
  'iterations-overflow|# benchmark_config: iterations=4294967296 runs=5 sample_policy=median warmup_iterations=100|io/read/copy_1mib ns_per_op=1.25 iter=4294967296 runs=5'
  'runs-overflow|# benchmark_config: iterations=1000 runs=18446744073709551616 sample_policy=median warmup_iterations=100|io/read/copy_1mib ns_per_op=1.25 iter=1000 runs=18446744073709551616'
  'warmup-overflow|# benchmark_config: iterations=1000 runs=5 sample_policy=median warmup_iterations=18446744073709551616|io/read/copy_1mib ns_per_op=1.25 iter=1000 runs=5'
)

for invalid_spec in "${producer_invalid_artifacts[@]}"; do
  label="${invalid_spec%%|*}"
  remainder="${invalid_spec#*|}"
  invalid_config="${remainder%%|*}"
  invalid_case="${remainder#*|}"
  invalid_artifact="$TEMP_DIR/$label.txt"
  write_artifact "$invalid_artifact" "$invalid_config" "$invalid_case"
  prove_rejected_artifact "$label" "$invalid_artifact"
done

provenance_extras=(
  'duplicate-source-repository|# source_repository: stateforward/emel.cpp'
  'conflicting-source-repository|# source_repository: contradictory/repository'
  'conflicting-source-commit|# source_commit: 0000000000000000000000000000000000000000'
  'conflicting-source-tree|# source_tree: 0000000000000000000000000000000000000000'
  'conflicting-fixture|# benchmark_fixture: contradictory fixture'
  'conflicting-validation|# benchmark_validation: contradictory validation'
  'duplicate-architecture|# bench_host_arch: aarch64'
  'duplicate-pointer-width|# bench_pointer_width: 64'
)

for invalid_spec in "${provenance_extras[@]}"; do
  label="${invalid_spec%%|*}"
  extra_line="${invalid_spec#*|}"
  invalid_artifact="$TEMP_DIR/$label.txt"
  write_artifact "$invalid_artifact" "$CONFIG" "$VALID_CASE" aarch64 "$extra_line"
  prove_rejected_artifact "$label" "$invalid_artifact"
done

invalid_artifact="$TEMP_DIR/invalid-architecture.txt"
write_artifact "$invalid_artifact" "$CONFIG" "$VALID_CASE" '../../escape'
prove_rejected_artifact 'invalid-architecture' "$invalid_artifact"

invalid_artifact="$TEMP_DIR/invalid-pointer-width.txt"
write_artifact "$invalid_artifact" "$CONFIG" "$VALID_CASE" aarch64 '' 128
prove_rejected_artifact 'invalid-pointer-width' "$invalid_artifact"

ILP32_OVERFLOW_CONFIG='# benchmark_config: iterations=1000 runs=4294967296 sample_policy=median warmup_iterations=100'
ILP32_OVERFLOW_CASE='io/read/copy_1mib ns_per_op=1.25 iter=1000 runs=4294967296'
invalid_artifact="$TEMP_DIR/ilp32-runs-overflow.txt"
write_artifact "$invalid_artifact" "$ILP32_OVERFLOW_CONFIG" "$ILP32_OVERFLOW_CASE" aarch64 '' 32
prove_rejected_artifact 'ilp32-runs-overflow' "$invalid_artifact"

INVALID_CONFIG='# benchmark_config: iterations=1000 iterations=1000 runs=5 sample_policy=median warmup_iterations=100'
write_artifact "$BASELINE" "$CONFIG" "$VALID_CASE"
write_artifact "$ARTIFACT" "$INVALID_CONFIG" "$VALID_CASE"
expect_failure 'update-invalid-config' run_bench --snapshot --update --suite=io-read
if [[ "$(cksum "$BASELINE")" != "$baseline_checksum" ]]; then
  echo 'error: invalid configuration update mutated the isolated baseline' >&2
  exit 1
fi
expect_failure 'compare-current-invalid-config' run_bench --snapshot --suite=io-read
if [[ "$(cksum "$BASELINE")" != "$baseline_checksum" ]]; then
  echo 'error: invalid configuration comparison mutated the isolated baseline' >&2
  exit 1
fi

write_artifact "$ARTIFACT" "$CONFIG" "$VALID_CASE"
write_artifact "$BASELINE" "$INVALID_CONFIG" "$VALID_CASE"
invalid_config_baseline_checksum="$(cksum "$BASELINE")"
expect_failure 'compare-invalid-config' run_bench --snapshot --suite=io-read
if [[ "$(cksum "$BASELINE")" != "$invalid_config_baseline_checksum" ]]; then
  echo 'error: rejected invalid configuration baseline comparison mutated the isolated baseline' >&2
  exit 1
fi

write_artifact "$BASELINE" "$CONFIG" "$VALID_CASE"
run_bench --snapshot --suite=io-read >/dev/null
printf '%s\n' 'I/O benchmark artifact fail-closed tests passed (81 negative paths, 4 valid paths)'
