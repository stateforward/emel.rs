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
  '# benchmark_fixture: public Stager/StageWindow, immutable source fill=0xa5, caller target cases=16384,1048576, chunk_bytes=4096' \
  '# benchmark_validation: typed done and synchronous callback checked each iteration, target equals source after measurement'; do
  field_prefix="${expected%%:*}: "
  if ! awk -v expected="$expected" -v field_prefix="$field_prefix" '
    index($0, field_prefix) == 1 { fields += 1; exact += ($0 == expected) }
    END { exit !(fields == 1 && exact == 1) }
  ' "$artifact"; then
    echo "error: $label must contain exactly one staged-read benchmark field: $expected" >&2
    exit 1
  fi
done

max_cross_lane_ratio="${EMEL_STAGED_READ_BENCH_MAX_RUST_VS_CPP_RATIO:-2.0}"
if ! awk -v expected_iterations="$expected_iterations" -v expected_runs="$expected_runs" \
    -v max_cross_lane_ratio="$max_cross_lane_ratio" '
  function unsigned(value) { return value ~ /^[0-9]+$/ }
  function finite_positive(value) {
    return length(value) <= 64 && value ~ /^([0-9]+([.][0-9]*)?|[.][0-9]+)$/ && value + 0 > 0
  }
  /^[[:space:]]*$/ || /^#/ { next }
  {
    rows += 1
    if ($1 != "io/staged-read/copy_16kib" &&
        $1 != "io/staged-read/copy_1mib") {
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
    expected_ratio = values["rust_ns_per_op"] / values["cpp_ns_per_op"]
    ratio_delta = values["rust_vs_cpp_ratio"] - expected_ratio
    if (ratio_delta < 0) ratio_delta = -ratio_delta
    if (NF != 6 ||
        fields["rust_ns_per_op"] != 1 ||
        fields["cpp_ns_per_op"] != 1 ||
        fields["rust_vs_cpp_ratio"] != 1 ||
        fields["iter"] != 1 ||
        fields["runs"] != 1) {
      invalid = 1
    }
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
  END {
    valid = rows == 2 && !invalid
    valid = valid && seen_case["io/staged-read/copy_16kib"] == 1
    valid = valid && seen_case["io/staged-read/copy_1mib"] == 1
    exit !valid
  }
' "$artifact"; then
  echo "error: $label has an invalid staged-read benchmark result schema" >&2
  exit 1
fi
