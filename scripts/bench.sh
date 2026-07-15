#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${EMEL_BENCH_BUILD_DIR:-$ROOT_DIR/target/bench}"
BASELINE_DIR="${EMEL_BENCH_BASELINE_DIR:-$ROOT_DIR/snapshots/bench}"
EMEL_CPP_SOURCE="${EMEL_CPP_SOURCE_DIR:-$ROOT_DIR/../emel.cpp}"
EMEL_CPP_COMMIT=843a117386ef17dc5a50549bbfc821074c2141d6
EMEL_CPP_IO_TREE=ff00a9978b00b4ea1e5268d2ecd483d4855b6eaa
EMEL_CPP_MODEL_TENSOR_TREE=06306d4ffad3455fcf5df71dc692df52514b9865
SNAPSHOT_MODE=false
UPDATE=false
SUITE=gguf
runner_args=(gguf)

usage() {
  cat <<'USAGE'
usage: scripts/bench.sh [--snapshot|--compare] [--update] [runner options]

  --snapshot  compare against the architecture-scoped benchmark baseline
  --compare   alias for --snapshot
  --update    replace the baseline after a successful benchmark run

Suites: --suite=gguf, --suite=io-read, --suite=io-mmap, --suite=io-staged-read, --suite=io-loader, --suite=model-tensor
Runner options: --iterations=N --runs=N --warmup-iterations=N
Set EMEL_BENCH_MAX_REGRESSION_RATIO to change the default 2.0x gate.
USAGE
}

for argument in "$@"; do
  case "$argument" in
    --snapshot|--compare) SNAPSHOT_MODE=true ;;
    --update) UPDATE=true ;;
    --iterations=*|--runs=*|--warmup-iterations=*) runner_args+=("$argument") ;;
    --suite=gguf) SUITE=gguf; runner_args[0]=gguf ;;
    --suite=io-read) SUITE=io-read; runner_args[0]=io-read ;;
    --suite=io-mmap) SUITE=io-mmap; runner_args[0]=io-mmap ;;
    --suite=io-staged-read) SUITE=io-staged-read; runner_args[0]=io-staged-read ;;
    --suite=io-loader) SUITE=io-loader; runner_args[0]=io-loader ;;
    --suite=model-tensor) SUITE=model-tensor; runner_args[0]=model-tensor ;;
    --help|-h) usage; exit 0 ;;
    *) echo "error: unknown argument: $argument" >&2; usage >&2; exit 2 ;;
  esac
done

if $UPDATE && ! $SNAPSHOT_MODE; then
  echo "error: --update requires --snapshot or --compare" >&2
  exit 2
fi

if [[ -n "${EMEL_BENCH_RUNNER:-}" ]]; then
  RUNNER="$EMEL_BENCH_RUNNER"
  if [[ ! -x "$RUNNER" ]]; then
    echo "error: EMEL_BENCH_RUNNER is not executable: $RUNNER" >&2
    exit 1
  fi
else
  cargo build --locked --manifest-path "$ROOT_DIR/Cargo.toml" --release -p emel-bench
  RUNNER="$ROOT_DIR/target/release/emel-bench"
fi
mkdir -p "$BUILD_DIR"
CURRENT="$BUILD_DIR/$SUITE-current.txt"
if [[ "$SUITE" == "io-mmap" ]]; then
  fixture="$BUILD_DIR/io-mmap-fixture.bin"
  EMEL_IO_MMAP_BENCH_FIXTURE="$fixture" \
    "$RUNNER" "${runner_args[@]}" >"$CURRENT"

  source_commit="$(git -C "$EMEL_CPP_SOURCE" rev-parse HEAD)"
  source_tree="$(git -C "$EMEL_CPP_SOURCE" rev-parse HEAD:src/emel/io)"
  if [[ "$source_commit" != "$EMEL_CPP_COMMIT" || "$source_tree" != "$EMEL_CPP_IO_TREE" ]]; then
    echo "error: emel.cpp I/O reference identity drifted" >&2
    exit 1
  fi
  if ! git -C "$EMEL_CPP_SOURCE" diff --quiet -- src/emel/io tests/io; then
    echo "error: emel.cpp I/O reference files are dirty" >&2
    exit 1
  fi

  reference_root="$BUILD_DIR/io-mmap-reference"
  materialized_source="$reference_root/emel-cpp-source"
  cmake -E remove_directory "$materialized_source"
  cmake -E make_directory "$materialized_source"
  git -C "$EMEL_CPP_SOURCE" archive "$EMEL_CPP_COMMIT" | \
    tar -x -C "$materialized_source"
  cmake_args=(
    -S "$ROOT_DIR/tools/emel-io-mmap-reference"
    -B "$reference_root/build"
    -DCMAKE_BUILD_TYPE=Release
    "-DEMEL_CPP_SOURCE_DIR=$materialized_source"
  )
  if command -v ninja >/dev/null 2>&1; then
    cmake_args+=(-G Ninja)
  fi
  cmake "${cmake_args[@]}"
  cmake --build "$reference_root/build" --parallel \
    --target emel-io-mmap-reference

  config_values="$(awk '
    /^# benchmark_config: / {
      for (field_index = 3; field_index <= NF; ++field_index) {
        split($field_index, part, "=")
        value[part[1]] = part[2]
      }
    }
    END { print value["iterations"], value["runs"], value["warmup_iterations"] }
  ' "$CURRENT")"
  read -r iterations runs warmup_iterations <<<"$config_values"
  reference_current="$CURRENT.reference"
  "$reference_root/build/emel-io-mmap-reference" --benchmark \
    "$fixture" "$iterations" "$runs" "$warmup_iterations" >"$reference_current"
  rm -f "$fixture"
  rust_current="$CURRENT.rust"
  mv "$CURRENT" "$rust_current"
  awk '
    function field_value(name,    field_index, part) {
      for (field_index = 2; field_index <= NF; ++field_index) {
        split($field_index, part, "=")
        if (part[1] == name) { return part[2] }
      }
      return ""
    }
    function emit(case_name,    ratio) {
      ratio = rust[case_name] / cpp[case_name]
      printf "io/mmap/lifecycle_%s rust_ns_per_op=%.3f cpp_ns_per_op=%.3f rust_vs_cpp_ratio=%.6f iter=%s runs=%s\n", \
        case_name, rust[case_name], cpp[case_name], ratio, iterations[case_name], runs[case_name]
    }
    FNR == NR {
      if ($0 ~ /^#/) { print; next }
      if ($1 ~ /^io\/mmap\/rust\/lifecycle_/) {
        case_name = $1
        sub(/^io\/mmap\/rust\/lifecycle_/, "", case_name)
        rust[case_name] = field_value("ns_per_op")
        iterations[case_name] = field_value("iter")
        runs[case_name] = field_value("runs")
      }
      next
    }
    $1 ~ /^io\/mmap\/reference\/lifecycle_/ {
      case_name = $1
      sub(/^io\/mmap\/reference\/lifecycle_/, "", case_name)
      cpp[case_name] = field_value("ns_per_op")
    }
    END {
      emit("16kib")
      emit("1mib")
    }
  ' "$rust_current" "$reference_current" >"$CURRENT"
elif [[ "$SUITE" == "io-staged-read" ]]; then
  "$RUNNER" "${runner_args[@]}" >"$CURRENT"
  source_commit="$(git -C "$EMEL_CPP_SOURCE" rev-parse HEAD)"
  source_tree="$(git -C "$EMEL_CPP_SOURCE" rev-parse HEAD:src/emel/io)"
  if [[ "$source_commit" != "$EMEL_CPP_COMMIT" || "$source_tree" != "$EMEL_CPP_IO_TREE" ]]; then
    echo "error: emel.cpp I/O reference identity drifted" >&2
    exit 1
  fi
  if ! git -C "$EMEL_CPP_SOURCE" diff --quiet -- src/emel/io tests/io; then
    echo "error: emel.cpp I/O reference files are dirty" >&2
    exit 1
  fi
  reference_root="$BUILD_DIR/io-staged-read-reference"
  materialized_source="$reference_root/emel-cpp-source"
  cmake -E remove_directory "$materialized_source"
  cmake -E make_directory "$materialized_source"
  git -C "$EMEL_CPP_SOURCE" archive "$EMEL_CPP_COMMIT" | \
    tar -x -C "$materialized_source"
  cmake_args=(
    -S "$ROOT_DIR/tools/emel-io-staged-read-reference"
    -B "$reference_root/build"
    -DCMAKE_BUILD_TYPE=Release
    "-DEMEL_CPP_SOURCE_DIR=$materialized_source"
  )
  if command -v ninja >/dev/null 2>&1; then
    cmake_args+=(-G Ninja)
  fi
  cmake "${cmake_args[@]}"
  cmake --build "$reference_root/build" --parallel \
    --target emel-io-staged-read-reference
  config_values="$(awk '
    /^# benchmark_config: / {
      for (field_index = 3; field_index <= NF; ++field_index) {
        split($field_index, part, "=")
        value[part[1]] = part[2]
      }
    }
    END { print value["iterations"], value["runs"], value["warmup_iterations"] }
  ' "$CURRENT")"
  read -r iterations runs warmup_iterations <<<"$config_values"
  reference_current="$CURRENT.reference"
  "$reference_root/build/emel-io-staged-read-reference" --benchmark \
    "$iterations" "$runs" "$warmup_iterations" >"$reference_current"
  rust_current="$CURRENT.rust"
  mv "$CURRENT" "$rust_current"
  awk '
    function field_value(name,    field_index, part) {
      for (field_index = 2; field_index <= NF; ++field_index) {
        split($field_index, part, "=")
        if (part[1] == name) { return part[2] }
      }
      return ""
    }
    function emit(case_name,    ratio) {
      ratio = rust[case_name] / cpp[case_name]
      printf "io/staged-read/copy_%s rust_ns_per_op=%.3f cpp_ns_per_op=%.3f rust_vs_cpp_ratio=%.6f iter=%s runs=%s\n", \
        case_name, rust[case_name], cpp[case_name], ratio, iterations[case_name], runs[case_name]
    }
    FNR == NR {
      if ($0 ~ /^#/) { print; next }
      if ($1 ~ /^io\/staged-read\/rust\/copy_/) {
        case_name = $1
        sub(/^io\/staged-read\/rust\/copy_/, "", case_name)
        rust[case_name] = field_value("ns_per_op")
        iterations[case_name] = field_value("iter")
        runs[case_name] = field_value("runs")
      }
      next
    }
    $1 ~ /^io\/staged-read\/reference\/copy_/ {
      case_name = $1
      sub(/^io\/staged-read\/reference\/copy_/, "", case_name)
      cpp[case_name] = field_value("ns_per_op")
    }
    END { emit("16kib"); emit("1mib") }
  ' "$rust_current" "$reference_current" >"$CURRENT"
elif [[ "$SUITE" == "io-loader" ]]; then
  "$RUNNER" "${runner_args[@]}" >"$CURRENT"
  source_commit="$(git -C "$EMEL_CPP_SOURCE" rev-parse HEAD)"
  source_tree="$(git -C "$EMEL_CPP_SOURCE" rev-parse HEAD:src/emel/io)"
  if [[ "$source_commit" != "$EMEL_CPP_COMMIT" || "$source_tree" != "$EMEL_CPP_IO_TREE" ]]; then
    echo "error: emel.cpp I/O reference identity drifted" >&2; exit 1
  fi
  reference_root="$BUILD_DIR/io-loader-reference"
  materialized_source="$reference_root/emel-cpp-source"
  cmake -E remove_directory "$materialized_source"; cmake -E make_directory "$materialized_source"
  git -C "$EMEL_CPP_SOURCE" archive "$EMEL_CPP_COMMIT" | tar -x -C "$materialized_source"
  cmake_args=(-S "$ROOT_DIR/tools/emel-io-loader-reference" -B "$reference_root/build"
    -DCMAKE_BUILD_TYPE=Release "-DEMEL_CPP_SOURCE_DIR=$materialized_source")
  if command -v ninja >/dev/null 2>&1; then cmake_args+=(-G Ninja); fi
  cmake "${cmake_args[@]}"; cmake --build "$reference_root/build" --parallel --target emel-io-loader-reference
  config_values="$(awk '/^# benchmark_config: / { for(i=3;i<=NF;i++){split($i,p,"=");v[p[1]]=p[2]} } END {print v["iterations"],v["runs"],v["warmup_iterations"]}' "$CURRENT")"
  read -r iterations runs warmup_iterations <<<"$config_values"
  reference_current="$CURRENT.reference"
  "$reference_root/build/emel-io-loader-reference" --benchmark "$iterations" "$runs" "$warmup_iterations" >"$reference_current"
  rust_current="$CURRENT.rust"; mv "$CURRENT" "$rust_current"
  awk '
    function field(name, i,p){for(i=2;i<=NF;i++){split($i,p,"=");if(p[1]==name)return p[2]}return ""}
    FNR==NR {if($0~/^#/){print;next} if($1=="io/loader/rust/read_copy_1mib"){r=field("ns_per_op");it=field("iter");runs=field("runs")} next}
    $1=="io/loader/reference/read_copy_1mib" {c=field("ns_per_op")}
    END {printf "io/loader/read_copy_1mib rust_ns_per_op=%.3f cpp_ns_per_op=%.3f rust_vs_cpp_ratio=%.6f iter=%s runs=%s\n",r,c,r/c,it,runs}
  ' "$rust_current" "$reference_current" >"$CURRENT"
elif [[ "$SUITE" == "model-tensor" ]]; then
  "$RUNNER" "${runner_args[@]}" >"$CURRENT"
  source_commit="$(git -C "$EMEL_CPP_SOURCE" rev-parse HEAD)"
  source_tree="$(git -C "$EMEL_CPP_SOURCE" rev-parse HEAD:src/emel/model/tensor)"
  if [[ "$source_commit" != "$EMEL_CPP_COMMIT" || "$source_tree" != "$EMEL_CPP_MODEL_TENSOR_TREE" ]]; then
    echo "error: emel.cpp model tensor reference identity drifted" >&2
    exit 1
  fi
  if ! git -C "$EMEL_CPP_SOURCE" diff --quiet -- src/emel/model/tensor tests/model/tensor; then
    echo "error: emel.cpp model tensor reference files are dirty" >&2
    exit 1
  fi
  reference_root="$BUILD_DIR/model-tensor-reference"
  materialized_source="$reference_root/emel-cpp-source"
  cmake -E remove_directory "$materialized_source"
  cmake -E make_directory "$materialized_source"
  git -C "$EMEL_CPP_SOURCE" archive "$EMEL_CPP_COMMIT" | tar -x -C "$materialized_source"
  cmake_args=(-S "$ROOT_DIR/tools/emel-model-tensor-reference" -B "$reference_root/build"
    -DCMAKE_BUILD_TYPE=Release "-DEMEL_CPP_SOURCE_DIR=$materialized_source")
  if command -v ninja >/dev/null 2>&1; then cmake_args+=(-G Ninja); fi
  cmake "${cmake_args[@]}"
  cmake --build "$reference_root/build" --parallel --target emel-model-tensor-reference
  config_values="$(awk '/^# benchmark_config: / { for(i=3;i<=NF;i++){split($i,p,"=");v[p[1]]=p[2]} } END {print v["iterations"],v["runs"],v["warmup_iterations"]}' "$CURRENT")"
  read -r iterations runs warmup_iterations <<<"$config_values"
  reference_current="$CURRENT.reference"
  "$reference_root/build/emel-model-tensor-reference" --benchmark \
    "$iterations" "$runs" "$warmup_iterations" >"$reference_current"
  rust_current="$CURRENT.rust"
  mv "$CURRENT" "$rust_current"
  awk '
    function field(name, i,p){for(i=2;i<=NF;i++){split($i,p,"=");if(p[1]==name)return p[2]}return ""}
    FNR==NR {if($0~/^#/){print;next} if($1=="model/tensor/rust/plan_mapped_64"){r=field("ns_per_op");it=field("iter");runs=field("runs")} next}
    $1=="model/tensor/reference/plan_mapped_64" {c=field("ns_per_op")}
    END {printf "model/tensor/plan_mapped_64 rust_ns_per_op=%.3f cpp_ns_per_op=%.3f rust_vs_cpp_ratio=%.6f iter=%s runs=%s\n",r,c,r/c,it,runs}
  ' "$rust_current" "$reference_current" >"$CURRENT"
else
  "$RUNNER" "${runner_args[@]}" >"$CURRENT"
fi

if ! $SNAPSHOT_MODE; then
  cat "$CURRENT"
  exit 0
fi

validate_io_read_arch() {
  local artifact="$1"
  local label="$2"
  if ! awk '
    /^# bench_host_arch: / {
      arch_rows += 1
      arch = substr($0, length("# bench_host_arch: ") + 1)
    }
    END {
      supported = arch == "aarch64" || arch == "x86_64" || arch == "x86" || arch == "arm" ||
        arch == "powerpc" || arch == "powerpc64" || arch == "s390x" || arch == "riscv32" ||
        arch == "riscv64" || arch == "loongarch64" || arch == "mips" || arch == "mips64" ||
        arch == "mips32r6" || arch == "mips64r6" || arch == "m68k" || arch == "csky" ||
        arch == "sparc" || arch == "sparc64" || arch == "hexagon"
      if (arch_rows != 1 || !supported) {
        exit 1
      }
      print arch
    }
  ' "$artifact"; then
    echo "error: $label has an invalid or unsupported bench_host_arch field" >&2
    exit 1
  fi
}

validate_io_read_pointer_width() {
  local artifact="$1"
  local label="$2"
  if ! awk '
    /^# bench_pointer_width: / {
      width_rows += 1
      width = substr($0, length("# bench_pointer_width: ") + 1)
    }
    END {
      if (width_rows != 1 || (width != "32" && width != "64")) {
        exit 1
      }
      print width
    }
  ' "$artifact"; then
    echo "error: $label has an invalid bench_pointer_width field" >&2
    exit 1
  fi
}

if [[ "$SUITE" == "io-read" || "$SUITE" == "io-mmap" || "$SUITE" == "io-staged-read" || "$SUITE" == "io-loader" || "$SUITE" == "model-tensor" ]]; then
  host_arch="$(validate_io_read_arch "$CURRENT" "current benchmark artifact")"
  pointer_width="$(validate_io_read_pointer_width "$CURRENT" "current benchmark artifact")"
else
  host_arch="$(awk -F': ' '/^# bench_host_arch: / { print $2; exit }' "$CURRENT")"
fi
BASELINE="$BASELINE_DIR/$SUITE-$host_arch.txt"

current_config="$(grep '^# benchmark_config: ' "$CURRENT" || true)"
if [[ -z "$host_arch" || -z "$current_config" ]]; then
  echo "error: benchmark runner emitted an incomplete architecture/configuration header" >&2
  exit 1
fi

validate_io_read_artifact() {
  local artifact="$1"
  local label="$2"
  local expected_iterations="$3"
  local expected_runs="$4"
  local expected
  local field_prefix
  for expected in \
    '# source_repository: stateforward/emel.cpp' \
    '# source_commit: 843a117386ef17dc5a50549bbfc821074c2141d6' \
    '# source_tree: ff00a9978b00b4ea1e5268d2ecd483d4855b6eaa' \
    '# benchmark_fixture: public Reader/ReadTensor, immutable source bytes=1048576 fill=0xa5, caller target bytes=1048576' \
    '# benchmark_validation: typed done checked each iteration, target equals source after measurement'; do
    field_prefix="${expected%%:*}: "
    if ! awk -v expected="$expected" -v field_prefix="$field_prefix" '
      index($0, field_prefix) == 1 {
        fields += 1
        if ($0 == expected) {
          exact += 1
        }
      }
      END { exit !(fields == 1 && exact == 1) }
    ' "$artifact"; then
      echo "error: $label must contain exactly one required I/O benchmark field: $expected" >&2
      exit 1
    fi
  done
  if ! awk -v expected_iterations="$expected_iterations" -v expected_runs="$expected_runs" '
    function unsigned(value) {
      return value ~ /^[0-9]+$/
    }
    function finite_positive(value) {
      return length(value) <= 64 && value ~ /^([0-9]+([.][0-9]*)?|[.][0-9]+)$/ && value + 0 > 0
    }
    /^[[:space:]]*$/ || /^#/ { next }
    {
      case_rows += 1
      if ($1 != "io/read/copy_1mib") {
        invalid = 1
        next
      }
      for (field_index = 2; field_index <= NF; ++field_index) {
        if (split($field_index, parts, "=") != 2) {
          invalid = 1
          continue
        }
        key = parts[1]
        value = parts[2]
        if (key != "ns_per_op" && key != "iter" && key != "runs") {
          invalid = 1
          continue
        }
        if (seen[key]++) {
          invalid = 1
          continue
        }
        values[key] = value
      }
    }
    END {
      valid = case_rows == 1 && !invalid
      valid = valid && seen["ns_per_op"] == 1 && finite_positive(values["ns_per_op"])
      valid = valid && seen["iter"] == 1 && unsigned(values["iter"])
      valid = valid && seen["runs"] == 1 && unsigned(values["runs"])
      valid = valid && ("value:" values["iter"]) == ("value:" expected_iterations)
      valid = valid && ("value:" values["runs"]) == ("value:" expected_runs)
      exit !valid
    }
  ' "$artifact"; then
    echo "error: $label has an invalid or configuration-incoherent io/read/copy_1mib result" >&2
    exit 1
  fi
}

validate_io_read_config() {
  local artifact="$1"
  local label="$2"
  local pointer_width="$3"
  if ! awk -v pointer_width="$pointer_width" '
    function canonical_decimal(value) {
      return value ~ /^[0-9]+$/ && (value == "0" || value !~ /^0/)
    }
    function decimal_at_most(value, maximum) {
      if (!canonical_decimal(value)) {
        return 0
      }
      if (length(value) != length(maximum)) {
        return length(value) < length(maximum)
      }
      return ("value:" value) <= ("value:" maximum)
    }
    function usize_maximum() {
      if (pointer_width == "64") {
        return "18446744073709551615"
      }
      return "4294967295"
    }
    /^# benchmark_config: / {
      config_rows += 1
      for (field_index = 3; field_index <= NF; ++field_index) {
        if (split($field_index, parts, "=") != 2) {
          invalid = 1
          continue
        }
        key = parts[1]
        value = parts[2]
        if (key != "iterations" && key != "runs" && key != "sample_policy" && key != "warmup_iterations") {
          invalid = 1
          continue
        }
        if (seen[key]++) {
          invalid = 1
          continue
        }
        values[key] = value
      }
    }
    END {
      valid = config_rows == 1 && !invalid
      valid = valid && seen["iterations"] == 1 && values["iterations"] != "0"
      valid = valid && decimal_at_most(values["iterations"], "4294967295")
      valid = valid && seen["runs"] == 1 && values["runs"] != "0"
      valid = valid && decimal_at_most(values["runs"], usize_maximum())
      valid = valid && seen["sample_policy"] == 1 && values["sample_policy"] == "median"
      valid = valid && seen["warmup_iterations"] == 1
      valid = valid && decimal_at_most(values["warmup_iterations"], "18446744073709551615")
      if (!valid) {
        exit 1
      }
      print values["iterations"], values["runs"]
    }
  ' "$artifact"; then
    echo "error: $label has an invalid benchmark_config schema" >&2
    exit 1
  fi
}

validate_io_mmap_artifact() {
  "$ROOT_DIR/scripts/validate_io_mmap_bench.sh" "$@"
}

validate_io_staged_read_artifact() {
  bash "$ROOT_DIR/scripts/validate_io_staged_read_bench.sh" "$@"
}

validate_io_loader_artifact() {
  bash "$ROOT_DIR/scripts/validate_io_loader_bench.sh" "$@"
}

validate_model_tensor_artifact() {
  bash "$ROOT_DIR/scripts/validate_model_tensor_bench.sh" "$@"
}

if [[ "$SUITE" == "io-read" ]]; then
  current_config_values="$(validate_io_read_config "$CURRENT" "current benchmark artifact" "$pointer_width")"
  current_iterations="${current_config_values%% *}"
  current_runs="${current_config_values#* }"
  validate_io_read_artifact \
    "$CURRENT" "current benchmark artifact" "$current_iterations" "$current_runs"
elif [[ "$SUITE" == "io-mmap" ]]; then
  current_config_values="$(validate_io_read_config "$CURRENT" "current benchmark artifact" "$pointer_width")"
  current_iterations="${current_config_values%% *}"
  current_runs="${current_config_values#* }"
  validate_io_mmap_artifact \
    "$CURRENT" "current benchmark artifact" "$current_iterations" "$current_runs"
elif [[ "$SUITE" == "io-staged-read" ]]; then
  current_config_values="$(validate_io_read_config "$CURRENT" "current benchmark artifact" "$pointer_width")"
  current_iterations="${current_config_values%% *}"
  current_runs="${current_config_values#* }"
  validate_io_staged_read_artifact \
    "$CURRENT" "current benchmark artifact" "$current_iterations" "$current_runs"
elif [[ "$SUITE" == "io-loader" ]]; then
  current_config_values="$(validate_io_read_config "$CURRENT" "current benchmark artifact" "$pointer_width")"
  current_iterations="${current_config_values%% *}"
  current_runs="${current_config_values#* }"
  validate_io_loader_artifact "$CURRENT" "current benchmark artifact" "$current_iterations" "$current_runs"
elif [[ "$SUITE" == "model-tensor" ]]; then
  current_config_values="$(validate_io_read_config "$CURRENT" "current benchmark artifact" "$pointer_width")"
  current_iterations="${current_config_values%% *}"
  current_runs="${current_config_values#* }"
  validate_model_tensor_artifact "$CURRENT" "current benchmark artifact" "$current_iterations" "$current_runs"
fi

if $UPDATE; then
  mkdir -p "$(dirname "$BASELINE")"
  install -m 0644 "$CURRENT" "$BASELINE"
  echo "Updated benchmark snapshot: $BASELINE"
  exit 0
fi

if [[ ! -f "$BASELINE" ]]; then
  echo "error: missing benchmark baseline: $BASELINE" >&2
  echo "run scripts/bench.sh --snapshot --update" >&2
  exit 1
fi

baseline_config="$(grep '^# benchmark_config: ' "$BASELINE" || true)"
if [[ -z "$baseline_config" || "$baseline_config" != "$current_config" ]]; then
  echo "error: benchmark configuration differs from the baseline" >&2
  echo "baseline: $baseline_config" >&2
  echo "current:  $current_config" >&2
  echo "run scripts/bench.sh --snapshot --update after an intentional configuration change" >&2
  exit 1
fi

if [[ "$SUITE" == "io-read" ]]; then
  baseline_arch="$(validate_io_read_arch "$BASELINE" "benchmark baseline")"
  if [[ "$baseline_arch" != "$host_arch" ]]; then
    echo "error: benchmark baseline architecture differs from the current artifact" >&2
    exit 1
  fi
  baseline_pointer_width="$(validate_io_read_pointer_width "$BASELINE" "benchmark baseline")"
  if [[ "$baseline_pointer_width" != "$pointer_width" ]]; then
    echo "error: benchmark baseline pointer width differs from the current artifact" >&2
    exit 1
  fi
  baseline_config_values="$(validate_io_read_config "$BASELINE" "benchmark baseline" "$baseline_pointer_width")"
  baseline_iterations="${baseline_config_values%% *}"
  baseline_runs="${baseline_config_values#* }"
  validate_io_read_artifact \
    "$BASELINE" "benchmark baseline" "$baseline_iterations" "$baseline_runs"
  for field in source_repository source_commit source_tree benchmark_fixture benchmark_validation; do
    baseline_value="$(grep "^# $field: " "$BASELINE" || true)"
    current_value="$(grep "^# $field: " "$CURRENT" || true)"
    if [[ -z "$baseline_value" || "$baseline_value" != "$current_value" ]]; then
      echo "error: io-read benchmark $field provenance differs from the baseline" >&2
      echo "baseline: $baseline_value" >&2
      echo "current:  $current_value" >&2
      exit 1
    fi
  done
elif [[ "$SUITE" == "io-mmap" ]]; then
  baseline_arch="$(validate_io_read_arch "$BASELINE" "benchmark baseline")"
  baseline_pointer_width="$(validate_io_read_pointer_width "$BASELINE" "benchmark baseline")"
  baseline_config_values="$(validate_io_read_config "$BASELINE" "benchmark baseline" "$baseline_pointer_width")"
  baseline_iterations="${baseline_config_values%% *}"
  baseline_runs="${baseline_config_values#* }"
  validate_io_mmap_artifact \
    "$BASELINE" "benchmark baseline" "$baseline_iterations" "$baseline_runs"
  if [[ "$baseline_arch" != "$host_arch" || "$baseline_pointer_width" != "$pointer_width" ]]; then
    echo "error: mmap benchmark baseline architecture differs from current" >&2
    exit 1
  fi
  for field in source_repository source_commit source_tree benchmark_fixture benchmark_operations benchmark_validation native_semantics_complete missing_native_semantics contract_delta; do
    baseline_value="$(grep "^# $field: " "$BASELINE" || true)"
    current_value="$(grep "^# $field: " "$CURRENT" || true)"
    if [[ -z "$baseline_value" || "$baseline_value" != "$current_value" ]]; then
      echo "error: io-mmap benchmark $field provenance differs from baseline" >&2
      exit 1
    fi
  done
elif [[ "$SUITE" == "io-staged-read" ]]; then
  baseline_arch="$(validate_io_read_arch "$BASELINE" "benchmark baseline")"
  baseline_pointer_width="$(validate_io_read_pointer_width "$BASELINE" "benchmark baseline")"
  baseline_config_values="$(validate_io_read_config "$BASELINE" "benchmark baseline" "$baseline_pointer_width")"
  baseline_iterations="${baseline_config_values%% *}"
  baseline_runs="${baseline_config_values#* }"
  validate_io_staged_read_artifact \
    "$BASELINE" "benchmark baseline" "$baseline_iterations" "$baseline_runs"
  if [[ "$baseline_arch" != "$host_arch" || "$baseline_pointer_width" != "$pointer_width" ]]; then
    echo "error: staged-read benchmark baseline architecture differs from current" >&2
    exit 1
  fi
  for field in source_repository source_commit source_tree benchmark_fixture benchmark_validation; do
    baseline_value="$(grep "^# $field: " "$BASELINE" || true)"
    current_value="$(grep "^# $field: " "$CURRENT" || true)"
    if [[ -z "$baseline_value" || "$baseline_value" != "$current_value" ]]; then
      echo "error: io-staged-read benchmark $field provenance differs from baseline" >&2
      exit 1
    fi
  done
elif [[ "$SUITE" == "io-loader" ]]; then
  baseline_arch="$(validate_io_read_arch "$BASELINE" "benchmark baseline")"
  baseline_pointer_width="$(validate_io_read_pointer_width "$BASELINE" "benchmark baseline")"
  baseline_config_values="$(validate_io_read_config "$BASELINE" "benchmark baseline" "$baseline_pointer_width")"
  baseline_iterations="${baseline_config_values%% *}"
  baseline_runs="${baseline_config_values#* }"
  validate_io_loader_artifact "$BASELINE" "benchmark baseline" "$baseline_iterations" "$baseline_runs"
  if [[ "$baseline_arch" != "$host_arch" || "$baseline_pointer_width" != "$pointer_width" ]]; then
    echo "error: loader benchmark architecture differs from baseline" >&2; exit 1
  fi
  for field in source_repository source_commit source_tree benchmark_fixture benchmark_validation; do
    [[ "$(grep "^# $field: " "$BASELINE")" == "$(grep "^# $field: " "$CURRENT")" ]] || {
      echo "error: io-loader benchmark $field provenance differs from baseline" >&2; exit 1; }
  done
elif [[ "$SUITE" == "model-tensor" ]]; then
  baseline_arch="$(validate_io_read_arch "$BASELINE" "benchmark baseline")"
  baseline_pointer_width="$(validate_io_read_pointer_width "$BASELINE" "benchmark baseline")"
  baseline_config_values="$(validate_io_read_config "$BASELINE" "benchmark baseline" "$baseline_pointer_width")"
  baseline_iterations="${baseline_config_values%% *}"
  baseline_runs="${baseline_config_values#* }"
  validate_model_tensor_artifact "$BASELINE" "benchmark baseline" "$baseline_iterations" "$baseline_runs"
  if [[ "$baseline_arch" != "$host_arch" || "$baseline_pointer_width" != "$pointer_width" ]]; then
    echo "error: model tensor benchmark architecture differs from baseline" >&2
    exit 1
  fi
  for field in source_repository source_commit source_tree benchmark_fixture benchmark_validation contract_delta; do
    [[ "$(grep "^# $field: " "$BASELINE")" == "$(grep "^# $field: " "$CURRENT")" ]] || {
      echo "error: model tensor benchmark $field differs from baseline" >&2
      exit 1
    }
  done
fi

max_ratio="${EMEL_BENCH_MAX_REGRESSION_RATIO:-2.0}"
if [[ "$SUITE" == "io-mmap" || "$SUITE" == "io-staged-read" || "$SUITE" == "io-loader" || "$SUITE" == "model-tensor" ]]; then
  awk -v max_ratio="$max_ratio" -v suite="$SUITE" '
    function value(name,    field_index, part) {
      for (field_index = 2; field_index <= NF; ++field_index) {
        split($field_index, part, "=")
        if (part[1] == name) { return part[2] + 0 }
      }
      return 0
    }
    FNR == NR && $0 !~ /^#/ && NF > 1 {
      baseline_rust[$1] = value("rust_ns_per_op")
      baseline_cpp[$1] = value("cpp_ns_per_op")
      next
    }
    FNR != NR && $0 !~ /^#/ && NF > 1 {
      current_rust[$1] = value("rust_ns_per_op")
      current_cpp[$1] = value("cpp_ns_per_op")
    }
    END {
      failed = 0
      for (name in baseline_rust) {
        if (!(name in current_rust)) {
          printf "missing %s benchmark case: %s\n", suite, name > "/dev/stderr"
          failed = 1
          continue
        }
        rust_ratio = current_rust[name] / baseline_rust[name]
        cpp_ratio = current_cpp[name] / baseline_cpp[name]
        printf "%s rust_baseline=%.3f rust_current=%.3f rust_ratio=%.3fx cpp_baseline=%.3f cpp_current=%.3f cpp_ratio=%.3fx\n", \
          name, baseline_rust[name], current_rust[name], rust_ratio, \
          baseline_cpp[name], current_cpp[name], cpp_ratio
        if (rust_ratio > max_ratio || cpp_ratio > max_ratio) {
          printf "%s benchmark regression: %s exceeds %.3fx\n", suite, name, max_ratio > "/dev/stderr"
          failed = 1
        }
      }
      for (name in current_rust) {
        if (!(name in baseline_rust)) {
          printf "new %s benchmark case missing from baseline: %s\n", suite, name > "/dev/stderr"
          failed = 1
        }
      }
      exit failed
    }
  ' "$BASELINE" "$CURRENT"
else
  awk -v max_ratio="$max_ratio" '
  function timing_value(    field_index, parts) {
    for (field_index = 1; field_index <= NF; ++field_index) {
      if ($field_index ~ /^ns_per_op=/) {
        split($field_index, parts, "=")
        return parts[2] + 0
      }
    }
    return 0
  }
  FNR == NR && $0 !~ /^#/ && NF > 1 {
    baseline[$1] = timing_value()
    next
  }
  FNR != NR && $0 !~ /^#/ && NF > 1 {
    current[$1] = timing_value()
  }
  END {
    failed = 0
    for (name in baseline) {
      if (!(name in current)) {
        printf "missing benchmark case: %s\n", name > "/dev/stderr"
        failed = 1
        continue
      }
      ratio = current[name] / baseline[name]
      printf "%s baseline=%.3f current=%.3f ratio=%.3fx\n", \
        name, baseline[name], current[name], ratio
      if (ratio > max_ratio) {
        printf "benchmark regression: %s exceeds %.3fx\n", name, max_ratio > "/dev/stderr"
        failed = 1
      }
    }
    for (name in current) {
      if (!(name in baseline)) {
        printf "new benchmark case missing from baseline: %s\n", name > "/dev/stderr"
        failed = 1
      }
    }
    exit failed
  }
  ' "$BASELINE" "$CURRENT"
fi
