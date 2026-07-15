#!/usr/bin/env bash
set -euo pipefail

artifact="$1"
label="$2"
expected_iterations="$3"
expected_runs="$4"

for expected in \
  '# source_repository: stateforward/emel.cpp' \
  '# source_commit: 843a117386ef17dc5a50549bbfc821074c2141d6' \
  '# source_tree: ff00a9978b00b4ea1e5268d2ecd483d4855b6eaa' \
  '# benchmark_fixture: public file-backed mmap lifecycle, file_bytes=1048576 pattern=incrementing_u8 offset=0 cases=16384,1048576' \
  '# benchmark_operations: source open, map, full immutable access checksum, sequential advice, will-need advice, don'\''t-need advice, release' \
  '# benchmark_validation: typed outcomes and handle length checked each iteration, full mapped FNV-1a equals precomputed fixture checksum' \
  '# native_semantics_complete: true' \
  '# missing_native_semantics: none' \
  '# contract_delta: rust_mmap_source_capability'; do
  field_prefix="${expected%%:*}: "
  if ! awk -v expected="$expected" -v field_prefix="$field_prefix" '
    index($0, field_prefix) == 1 {
      fields += 1
      if ($0 == expected) { exact += 1 }
    }
    END { exit !(fields == 1 && exact == 1) }
  ' "$artifact"; then
    echo "error: $label must contain exactly one mmap benchmark field: $expected" >&2
    exit 1
  fi
done

max_cross_lane_ratio="${EMEL_MMAP_BENCH_MAX_RUST_VS_CPP_RATIO:-2.0}"
if ! awk -v expected_iterations="$expected_iterations" -v expected_runs="$expected_runs" \
    -v max_cross_lane_ratio="$max_cross_lane_ratio" '
  function unsigned(value) { return value ~ /^[0-9]+$/ }
  function finite_positive(value) {
    return length(value) <= 64 && value ~ /^([0-9]+([.][0-9]*)?|[.][0-9]+)$/ && value + 0 > 0
  }
  /^[[:space:]]*$/ || /^#/ { next }
  {
    rows += 1
    if ($1 != "io/mmap/lifecycle_16kib" &&
        $1 != "io/mmap/lifecycle_1mib") {
      invalid = 1
    }
    if (seen_case[$1]++) { invalid = 1 }
    delete fields
    delete values
    for (field_index = 2; field_index <= NF; ++field_index) {
      if (split($field_index, part, "=") != 2) { invalid = 1 }
      if (part[1] != "rust_ns_per_op" && part[1] != "cpp_ns_per_op" &&
          part[1] != "rust_vs_cpp_ratio" && part[1] != "iter" && part[1] != "runs") {
        invalid = 1
      }
      if (fields[part[1]]++) { invalid = 1 }
      values[part[1]] = part[2]
    }
    if (NF != 6 ||
        fields["rust_ns_per_op"] != 1 ||
        fields["cpp_ns_per_op"] != 1 ||
        fields["rust_vs_cpp_ratio"] != 1 ||
        fields["iter"] != 1 ||
        fields["runs"] != 1) {
      invalid = 1
    }
    expected_ratio = values["rust_ns_per_op"] / values["cpp_ns_per_op"]
    ratio_delta = values["rust_vs_cpp_ratio"] - expected_ratio
    if (ratio_delta < 0) { ratio_delta = -ratio_delta }
    if (!finite_positive(values["rust_ns_per_op"]) ||
        !finite_positive(values["cpp_ns_per_op"]) ||
        !finite_positive(values["rust_vs_cpp_ratio"]) ||
        values["rust_vs_cpp_ratio"] > max_cross_lane_ratio ||
        ratio_delta > 0.00001 ||
        !unsigned(values["iter"]) || !unsigned(values["runs"]) ||
        ("value:" values["iter"]) != ("value:" expected_iterations) ||
        ("value:" values["runs"]) != ("value:" expected_runs)) {
      invalid = 1
    }
  }
  END { exit !(rows == 2 && !invalid) }
' "$artifact"; then
  echo "error: $label has invalid or non-comparable mmap benchmark rows" >&2
  exit 1
fi
