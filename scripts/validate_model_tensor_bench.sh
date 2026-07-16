#!/usr/bin/env bash
set -euo pipefail

artifact="$1"
label="$2"
expected_iterations="$3"
expected_runs="$4"

for expected in \
  '# source_repository: stateforward/emel.cpp' \
  '# source_commit: 843a117386ef17dc5a50549bbfc821074c2141d6' \
  '# source_tree: 06306d4ffad3455fcf5df71dc692df52514b9865' \
  '# benchmark_fixture: production Store/process_event, direct read/copy bytes=4096 with preallocated actor-owned targets, plus 64 bound metadata records and a preallocated mapped effect buffer' \
  '# benchmark_validation: direct read times public child dispatch and copy on both lanes with typed completion and byte checks outside timing; mapped planning checks typed count and lane-native recovery with returned allocation reset and reused' \
  '# contract_delta: direct-read target ownership differs but both timed lanes dispatch the public tensor actor and copy 4096 bytes into preallocated target storage; each timed plan closes with its lane-native backend-error event'; do
  field_prefix="${expected%%:*}: "
  if ! awk -v expected="$expected" -v field_prefix="$field_prefix" '
    index($0, field_prefix) == 1 { fields += 1; exact += ($0 == expected) }
    END { exit !(fields == 1 && exact == 1) }
  ' "$artifact"; then
    echo "error: $label must contain exactly one model tensor benchmark field: $expected" >&2
    exit 1
  fi
done

max_cross_lane_ratio="${EMEL_MODEL_TENSOR_BENCH_MAX_RUST_VS_CPP_RATIO:-2.0}"
if ! awk -v expected_iterations="$expected_iterations" -v expected_runs="$expected_runs" \
    -v max_cross_lane_ratio="$max_cross_lane_ratio" '
  function unsigned(value) { return value ~ /^[0-9]+$/ }
  function finite_positive(value) {
    return length(value) <= 64 && value ~ /^([0-9]+([.][0-9]*)?|[.][0-9]+)$/ && value + 0 > 0
  }
  /^[[:space:]]*$/ || /^#/ { next }
  {
    rows += 1
    if (($1 != "model/tensor/plan_mapped_64" && $1 != "model/tensor/direct_read_4k") ||
        seen_case[$1]++) { invalid = 1 }
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
    if (NF != 6 || fields["rust_ns_per_op"] != 1 || fields["cpp_ns_per_op"] != 1 ||
        fields["rust_vs_cpp_ratio"] != 1 || fields["iter"] != 1 || fields["runs"] != 1 ||
        !finite_positive(values["rust_ns_per_op"]) || !finite_positive(values["cpp_ns_per_op"]) ||
        !finite_positive(values["rust_vs_cpp_ratio"]) ||
        values["rust_vs_cpp_ratio"] > max_cross_lane_ratio || ratio_delta > 0.00001 ||
        !unsigned(values["iter"]) || !unsigned(values["runs"]) ||
        ("value:" values["iter"]) != ("value:" expected_iterations) ||
        ("value:" values["runs"]) != ("value:" expected_runs)) { invalid = 1 }
  }
  END { exit !(rows == 2 && !invalid && seen_case["model/tensor/plan_mapped_64"] == 1 &&
               seen_case["model/tensor/direct_read_4k"] == 1) }
' "$artifact"; then
  echo "error: $label has an invalid model tensor benchmark result schema" >&2
  exit 1
fi
