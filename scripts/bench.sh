#!/usr/bin/env bash
set -euo pipefail

if [[ -d /opt/homebrew/bin ]]; then
  export PATH="/opt/homebrew/bin:$PATH"
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${EMEL_BENCH_BUILD_DIR:-$ROOT_DIR/target/bench}"
BASELINE_DIR="${EMEL_BENCH_BASELINE_DIR:-$ROOT_DIR/snapshots/bench}"
EMEL_CPP_SOURCE="${EMEL_CPP_SOURCE_DIR:-$ROOT_DIR/../emel.cpp}"
EMEL_CPP_COMMIT=843a117386ef17dc5a50549bbfc821074c2141d6
EMEL_CPP_IO_TREE=ff00a9978b00b4ea1e5268d2ecd483d4855b6eaa
EMEL_CPP_MODEL_TENSOR_TREE=06306d4ffad3455fcf5df71dc692df52514b9865
EMEL_CPP_MODEL_VOCAB_TREE=d2fd66887fbdef6e0894de9391155099839e625c
EMEL_CPP_MODEL_VOCAB_DETAIL_BLOB=7c964f7449640fd7fe9eef21f4c14f3a65bf7b6a
EMEL_MODEL_VOCAB_SML_COMMIT=49207123cd3f39767764bae774932cb48623f92f
EMEL_MODEL_VOCAB_SML_SOURCE="${EMEL_STATEFORWARD_SML_SOURCE:-$EMEL_CPP_SOURCE/build/zig/_deps/stateforward_sml-src}"
EMEL_KERNEL_CAPABILITY_SML_COMMIT=49207123cd3f39767764bae774932cb48623f92f
EMEL_KERNEL_CAPABILITY_SML_SOURCE="${EMEL_STATEFORWARD_SML_SOURCE:-$EMEL_CPP_SOURCE/build/zig/_deps/stateforward_sml-src}"
SNAPSHOT_MODE=false
UPDATE=false
SUITE=gguf
runner_args=(gguf)
model_data_iterations=5
model_data_runs=5
model_data_warmup_iterations=1
model_vocabulary_iterations=5
model_vocabulary_runs=5
model_vocabulary_warmup_iterations=1
token_profile_iterations=10000000
token_profile_runs=7
token_profile_warmup_iterations=1000000
kernel_capability_iterations=1000
kernel_capability_runs=5
kernel_capability_warmup_iterations=100
model_catalog_iterations=200000
model_catalog_runs=7
model_catalog_warmup_iterations=20000
model_generation_iterations=1000000
model_generation_runs=11
model_generation_warmup_iterations=100000
model_qwen3_iterations=1000000
model_qwen3_runs=11
model_qwen3_warmup_iterations=100000

sha256_file() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{print $1}'
  else
    shasum -a 256 "$1" | awk '{print $1}'
  fi
}

usage() {
  cat <<'USAGE'
usage: scripts/bench.sh [--snapshot|--compare] [--update] [runner options]

  --snapshot  compare against the architecture-scoped benchmark baseline
  --compare   alias for --snapshot
  --update    replace the baseline after a successful benchmark run

Suites: --suite=gguf, --suite=io-read, --suite=io-mmap, --suite=io-staged-read, --suite=io-loader, --suite=model-tensor, --suite=model-data, --suite=model-catalog, --suite=model-generation, --suite=model-qwen3, --suite=model-vocabulary, --suite=token-profile, --suite=kernel-capability
Runner options: --iterations=N --runs=N --warmup-iterations=N
Set EMEL_BENCH_MAX_REGRESSION_RATIO to change the default 2.0x gate.
USAGE
}

for argument in "$@"; do
  case "$argument" in
    --snapshot|--compare) SNAPSHOT_MODE=true ;;
    --update) UPDATE=true ;;
    --iterations=*)
      runner_args+=("$argument")
      model_data_iterations="${argument#*=}"
      model_vocabulary_iterations="${argument#*=}"
      token_profile_iterations="${argument#*=}"
      kernel_capability_iterations="${argument#*=}"
      model_catalog_iterations="${argument#*=}"
      model_generation_iterations="${argument#*=}"
      model_qwen3_iterations="${argument#*=}"
      ;;
    --runs=*)
      runner_args+=("$argument")
      model_data_runs="${argument#*=}"
      model_vocabulary_runs="${argument#*=}"
      token_profile_runs="${argument#*=}"
      kernel_capability_runs="${argument#*=}"
      model_catalog_runs="${argument#*=}"
      model_generation_runs="${argument#*=}"
      model_qwen3_runs="${argument#*=}"
      ;;
    --warmup-iterations=*)
      runner_args+=("$argument")
      model_data_warmup_iterations="${argument#*=}"
      model_vocabulary_warmup_iterations="${argument#*=}"
      token_profile_warmup_iterations="${argument#*=}"
      kernel_capability_warmup_iterations="${argument#*=}"
      model_catalog_warmup_iterations="${argument#*=}"
      model_generation_warmup_iterations="${argument#*=}"
      model_qwen3_warmup_iterations="${argument#*=}"
      ;;
    --suite=gguf) SUITE=gguf; runner_args[0]=gguf ;;
    --suite=io-read) SUITE=io-read; runner_args[0]=io-read ;;
    --suite=io-mmap) SUITE=io-mmap; runner_args[0]=io-mmap ;;
    --suite=io-staged-read) SUITE=io-staged-read; runner_args[0]=io-staged-read ;;
    --suite=io-loader) SUITE=io-loader; runner_args[0]=io-loader ;;
    --suite=model-tensor) SUITE=model-tensor; runner_args[0]=model-tensor ;;
    --suite=model-data) SUITE=model-data ;;
    --suite=model-catalog) SUITE=model-catalog ;;
    --suite=model-generation) SUITE=model-generation ;;
    --suite=model-qwen3) SUITE=model-qwen3 ;;
    --suite=model-vocabulary) SUITE=model-vocabulary ;;
    --suite=token-profile) SUITE=token-profile ;;
    --suite=kernel-capability) SUITE=kernel-capability ;;
    --help|-h) usage; exit 0 ;;
    *) echo "error: unknown argument: $argument" >&2; usage >&2; exit 2 ;;
  esac
done

if $UPDATE && ! $SNAPSHOT_MODE; then
  echo "error: --update requires --snapshot or --compare" >&2
  exit 2
fi

if [[ "$SUITE" == "model-catalog" ]]; then
  catalog_args=(
    "--iterations=$model_catalog_iterations"
    "--runs=$model_catalog_runs"
    "--warmup-iterations=$model_catalog_warmup_iterations"
  )
  if $SNAPSHOT_MODE; then
    catalog_args+=(--snapshot)
  fi
  if $UPDATE; then
    catalog_args+=(--update)
  fi
  exec "$ROOT_DIR/scripts/model-catalog-bench.sh" "${catalog_args[@]}"
fi

if [[ "$SUITE" == "model-generation" ]]; then
  generation_args=("--iterations=$model_generation_iterations" "--runs=$model_generation_runs" "--warmup-iterations=$model_generation_warmup_iterations")
  $SNAPSHOT_MODE && generation_args+=(--snapshot)
  $UPDATE && generation_args+=(--update)
  exec "$ROOT_DIR/scripts/model-generation-bench.sh" "${generation_args[@]}"
fi

if [[ "$SUITE" == "model-qwen3" ]]; then
  qwen3_args=("--iterations=$model_qwen3_iterations" "--runs=$model_qwen3_runs" "--warmup-iterations=$model_qwen3_warmup_iterations")
  $SNAPSHOT_MODE && qwen3_args+=(--snapshot)
  $UPDATE && qwen3_args+=(--update)
  exec "$ROOT_DIR/scripts/model-qwen3-bench.sh" "${qwen3_args[@]}"
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
if [[ "$SUITE" == "kernel-capability" ]]; then
  events_blob="$(git -C "$EMEL_CPP_SOURCE" rev-parse \
    "$EMEL_CPP_COMMIT:src/emel/kernel/events.hpp")"
  detail_blob="$(git -C "$EMEL_CPP_SOURCE" rev-parse \
    "$EMEL_CPP_COMMIT:src/emel/kernel/detail.hpp")"
  generation_header_blob="$(git -C "$EMEL_CPP_SOURCE" rev-parse \
    "$EMEL_CPP_COMMIT:src/emel/model/generation/any.hpp")"
  generation_blob="$(git -C "$EMEL_CPP_SOURCE" rev-parse \
    "$EMEL_CPP_COMMIT:src/emel/model/generation/any.cpp")"
  if [[ "$events_blob" != "4b3f02fa5ff9c5071d1fc83938cef1b22b4658b9" || \
        "$detail_blob" != "c8a82643eabfe8f2d7883e655955f455794511b0" || \
        "$generation_header_blob" != "d521cf68e1bf52a2a193bbdb460741772199b318" || \
        "$generation_blob" != "099058ccd441d1dc6bebbb0c4994070d2f533c47" ]]; then
    echo "error: emel.cpp kernel capability source identity drifted" >&2
    exit 1
  fi
  capability_work="$BUILD_DIR/kernel-capability-reference"
  materialized_source="$capability_work/emel-cpp-source"
  reference_build="$capability_work/build"
  cmake -E remove_directory "$materialized_source"
  cmake -E remove_directory "$reference_build"
  cmake -E make_directory "$materialized_source"
  if [[ "$(git -C "$EMEL_KERNEL_CAPABILITY_SML_SOURCE" rev-parse HEAD)" != \
        "$EMEL_KERNEL_CAPABILITY_SML_COMMIT" ]]; then
    echo "error: stateforward-sml source is not pinned at $EMEL_KERNEL_CAPABILITY_SML_COMMIT" >&2
    exit 1
  fi
  git -C "$EMEL_CPP_SOURCE" archive "$EMEL_CPP_COMMIT" \
    CMakeLists.txt cmake include src | \
    tar -x -C "$materialized_source"
  cmake -S "$ROOT_DIR/tools/emel-kernel-capability-reference" \
    -B "$reference_build" -DCMAKE_BUILD_TYPE=Release \
    "-DFETCHCONTENT_SOURCE_DIR_STATEFORWARD_SML=$EMEL_KERNEL_CAPABILITY_SML_SOURCE" \
    "-DEMEL_CPP_SOURCE_DIR=$materialized_source"
  cmake --build "$reference_build" --parallel \
    --target emel-kernel-capability-reference
  "$reference_build/emel-kernel-capability-reference" --benchmark \
    "$kernel_capability_iterations" "$kernel_capability_runs" \
    "$kernel_capability_warmup_iterations" >"$capability_work/cpp.out"
  cargo run --quiet --locked --release --manifest-path "$ROOT_DIR/Cargo.toml" \
    -p emel-kernel-capability-parity -- --benchmark \
    "$kernel_capability_iterations" "$kernel_capability_runs" \
    "$kernel_capability_warmup_iterations" >"$capability_work/rust.out"
  {
    printf '# bench_host_arch: %s\n' \
      "$(rustc --print cfg | awk -F'"' '/^target_arch=/{print $2; exit}')"
    printf '# bench_pointer_width: %s\n' "$(( $(getconf LONG_BIT) ))"
    printf '# benchmark_config: iterations=%s runs=%s sample_policy=median warmup_iterations=%s\n' \
      "$kernel_capability_iterations" "$kernel_capability_runs" \
      "$kernel_capability_warmup_iterations"
    printf '# source_repository: stateforward/emel.cpp\n'
    printf '# source_commit: %s\n' "$EMEL_CPP_COMMIT"
    printf '# source_kernel_events_blob: %s\n' "$events_blob"
    printf '# source_kernel_detail_blob: %s\n' "$detail_blob"
    printf '# source_generation_header_blob: %s\n' "$generation_header_blob"
    printf '# source_generation_blob: %s\n' "$generation_blob"
    printf '# benchmark_fixture: same MatrixWeightContract tensor_type=Q4_K expected=NativeQuantized in both public owner APIs\n'
    printf '# benchmark_timing: Rust public Resolver dispatch versus pinned C++ public 14-stage generation audit; 100 warmups then median 5x1000 by default\n'
    printf '# benchmark_validation: every observation is native_quantized; Rust dispatch allocation separately proven zero\n'
    printf '# benchmark_interpretation: conservative source-owner comparison; C++ timing includes construction and traversal of its complete audit result\n'
    awk '
      function value(name,    i, part) {
        for (i = 1; i <= NF; ++i) {
          split($i, part, "=")
          if (part[1] == name) return part[2]
        }
        return ""
      }
      FNR == NR {
        cpp = value("cpp_ns_per_op")
        cpp_outcome = value("outcome")
        cpp_iter = value("iter")
        cpp_runs = value("runs")
        next
      }
      {
        rust = value("rust_ns_per_op")
        rust_outcome = value("outcome")
        rust_iter = value("iter")
        rust_runs = value("runs")
      }
      END {
        ratio = rust / cpp
        if (cpp <= 0 || rust <= 0 || cpp_outcome != "native_quantized" ||
            rust_outcome != cpp_outcome || rust_iter != cpp_iter ||
            rust_runs != cpp_runs || ratio > 2.0) exit 1
        printf "kernel/capability/resolve_matrix_q4_k rust_ns_per_op=%.3f cpp_ns_per_op=%.3f rust_vs_cpp_ratio=%.6f outcome=native_quantized dispatch_allocations=0 iter=%s runs=%s\n", rust, cpp, ratio, rust_iter, rust_runs
      }
    ' "$capability_work/cpp.out" "$capability_work/rust.out"
  } >"$CURRENT"
elif [[ "$SUITE" == "token-profile" ]]; then
  raw_output="$CURRENT.raw"
  EMEL_TOKEN_PROFILE_BENCH_ITERATIONS="$token_profile_iterations" \
    EMEL_TOKEN_PROFILE_BENCH_RUNS="$token_profile_runs" \
    EMEL_TOKEN_PROFILE_BENCH_WARMUP_ITERATIONS="$token_profile_warmup_iterations" \
    cargo test --locked --manifest-path "$ROOT_DIR/Cargo.toml" --release \
      -p emel-token --lib profile::tests::benchmark_profile_resolver_dispatch -- \
      --ignored --exact --nocapture >"$raw_output"
  grep -E '^(# bench_|# benchmark_|# source_|token/profile/)' "$raw_output" >"$CURRENT"
elif [[ "$SUITE" == "model-data" ]]; then
  raw_output="$CURRENT.raw"
  EMEL_MODEL_DATA_BENCH_ITERATIONS="$model_data_iterations" \
    EMEL_MODEL_DATA_BENCH_RUNS="$model_data_runs" \
    EMEL_MODEL_DATA_BENCH_WARMUP_ITERATIONS="$model_data_warmup_iterations" \
    cargo test --locked --manifest-path "$ROOT_DIR/Cargo.toml" --release \
      -p emel-model --lib data::tests::benchmark_model_data_foundation -- \
      --ignored --exact --nocapture >"$raw_output"
  grep -E '^(# bench_|# benchmark_|# source_|model/data/)' "$raw_output" >"$CURRENT"
elif [[ "$SUITE" == "model-vocabulary" ]]; then
  real_fixture="${EMEL_MODEL_VOCAB_BENCH_FIXTURE:-$EMEL_CPP_SOURCE/tests/models/distilgpt2.Q2_K.gguf}"
  [[ -f "$real_fixture" ]] || {
    echo "error: required model vocabulary benchmark fixture is missing: $real_fixture" >&2
    exit 2
  }
  [[ "$(sha256_file "$real_fixture")" == "b046ac09ba24a848e2140676fba58c1dcf2f19617e45b03524043eabdb556a31" ]] || {
    echo "error: model vocabulary real benchmark fixture identity drifted" >&2
    exit 2
  }
  [[ "$(git -C "$EMEL_CPP_SOURCE" rev-parse "$EMEL_CPP_COMMIT^{tree}")" == "$EMEL_CPP_MODEL_VOCAB_TREE" ]] || {
    echo "error: emel.cpp model vocabulary source tree identity drifted" >&2
    exit 2
  }
  [[ "$(git -C "$EMEL_CPP_SOURCE" rev-parse "$EMEL_CPP_COMMIT:src/emel/model/detail.cpp")" == "$EMEL_CPP_MODEL_VOCAB_DETAIL_BLOB" ]] || {
    echo "error: emel.cpp model vocabulary detail identity drifted" >&2
    exit 2
  }
  [[ "$(git -C "$EMEL_MODEL_VOCAB_SML_SOURCE" rev-parse HEAD)" == "$EMEL_MODEL_VOCAB_SML_COMMIT" ]] || {
    echo "error: model vocabulary stateforward-sml identity drifted" >&2
    exit 2
  }

  vocabulary_work="$(mktemp -d "${TMPDIR:-/tmp}/emel-model-vocabulary-bench.XXXXXX")"
  mkdir -p "$vocabulary_work/source" "$vocabulary_work/fixtures"
  cargo run --quiet --locked --release --manifest-path "$ROOT_DIR/Cargo.toml" \
    -p emel-model --example vocabulary_observer -- \
    --write-benchmark-fixtures "$vocabulary_work/fixtures"
  small_fixture="$vocabulary_work/fixtures/small.gguf"
  boundary_fixture="$vocabulary_work/fixtures/gemma4-400001.gguf"
  small_fixture_sha256="$(sha256_file "$small_fixture")"
  boundary_fixture_sha256="$(sha256_file "$boundary_fixture")"

  git -C "$EMEL_CPP_SOURCE" archive "$EMEL_CPP_COMMIT" CMakeLists.txt cmake include src \
    | tar -x -C "$vocabulary_work/source"
  cmake -S "$vocabulary_work/source" -B "$vocabulary_work/build" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DEMEL_ENABLE_TESTS=OFF \
    -DFETCHCONTENT_SOURCE_DIR_STATEFORWARD_SML="$EMEL_MODEL_VOCAB_SML_SOURCE" >/dev/null
  cmake --build "$vocabulary_work/build" --target emel >/dev/null
  "${CXX:-c++}" -std=c++20 -O2 \
    -I"$vocabulary_work/source/include" \
    -I"$vocabulary_work/source/src" \
    -I"$EMEL_MODEL_VOCAB_SML_SOURCE/include" \
    "$ROOT_DIR/tools/emel-model-vocab-reference/main.cpp" \
    "$vocabulary_work/build/libemel.a" \
    -o "$vocabulary_work/cpp-observer"

  "$vocabulary_work/cpp-observer" --benchmark \
    "$model_vocabulary_iterations" "$model_vocabulary_runs" \
    "$model_vocabulary_warmup_iterations" \
    "$small_fixture" "$real_fixture" "$boundary_fixture" \
    >"$vocabulary_work/cpp.out"
  cargo run --quiet --locked --release --manifest-path "$ROOT_DIR/Cargo.toml" \
    -p emel-model --example vocabulary_observer -- --benchmark \
    "$model_vocabulary_iterations" "$model_vocabulary_runs" \
    "$model_vocabulary_warmup_iterations" \
    "$small_fixture" "$real_fixture" "$boundary_fixture" \
    >"$vocabulary_work/rust.out"

  {
    printf '# bench_host_arch: %s\n' \
      "$(rustc --print cfg | awk -F'"' '/^target_arch=/{print $2; exit}')"
    printf '# bench_pointer_width: %s\n' "$(( $(getconf LONG_BIT) ))"
    printf '# benchmark_config: iterations=%s runs=%s sample_policy=median warmup_iterations=%s\n' \
      "$model_vocabulary_iterations" "$model_vocabulary_runs" \
      "$model_vocabulary_warmup_iterations"
    printf '# source_repository: stateforward/emel.cpp\n'
    printf '# source_commit: %s\n' "$EMEL_CPP_COMMIT"
    printf '# source_tree: %s\n' "$EMEL_CPP_MODEL_VOCAB_TREE"
    printf '# source_detail_blob: %s\n' "$EMEL_CPP_MODEL_VOCAB_DETAIL_BLOB"
    printf '# source_sml_commit: %s\n' "$EMEL_MODEL_VOCAB_SML_COMMIT"
    printf '# reference_tool_sha256: %s\n' \
      "$(sha256_file "$ROOT_DIR/tools/emel-model-vocab-reference/main.cpp")"
    printf '# rust_tool_sha256: %s\n' \
      "$(sha256_file "$ROOT_DIR/crates/emel-model/examples/vocabulary_observer.rs")"
    printf '# fixture_small_sha256: %s\n' "$small_fixture_sha256"
    printf '# fixture_real_sha256: %s\n' \
      b046ac09ba24a848e2140676fba58c1dcf2f19617e45b03524043eabdb556a31
    printf '# fixture_boundary_sha256: %s\n' "$boundary_fixture_sha256"
    printf '# benchmark_fixture: identical GGUF bytes in both lanes; small=standard_2_tokens_1_merge real=distilgpt2.Q2_K.gguf boundary=gemma4_2_tokens_400001_merges\n'
    printf '# benchmark_timing: fixture IO, GGUF parse, archive/build, actor construction, and bounded storage allocation excluded; repeated vocabulary materialization only\n'
    printf '# benchmark_validation: full semantic digest and typed counts match across public C++ model facade and public Rust Load actor lanes; Rust dispatch allocations=0\n'
    awk -v iterations="$model_vocabulary_iterations" \
      -v runs="$model_vocabulary_runs" '
      function field(name,    field_index, part) {
        for (field_index = 1; field_index <= NF; ++field_index) {
          split($field_index, part, "=")
          if (part[1] == name) return part[2]
        }
        return ""
      }
      FNR == NR {
        name = field("case")
        cpp[name] = field("cpp_ns_per_op")
        cpp_digest[name] = field("digest")
        cpp_tokens[name] = field("token_count")
        cpp_merges[name] = field("merge_count")
        next
      }
      {
        name = field("case")
        rust = field("rust_ns_per_op")
        digest = field("digest")
        tokens = field("token_count")
        merges = field("merge_count")
        allocations = field("dispatch_allocations")
        if (!(name in cpp) || cpp[name] + 0 <= 0 || rust + 0 <= 0 ||
            digest != cpp_digest[name] || tokens != cpp_tokens[name] ||
            merges != cpp_merges[name] || allocations != "0") {
          print "invalid model vocabulary benchmark evidence for " name > "/dev/stderr"
          failed = 1
          next
        }
        ratio = rust / cpp[name]
        if (ratio > 2.0) {
          printf "model vocabulary source ratio %.6fx exceeds 2.0x for %s\n", ratio, name > "/dev/stderr"
          failed = 1
        }
        printf "model/vocabulary/materialize_%s rust_ns_per_op=%.3f cpp_ns_per_op=%.3f rust_vs_cpp_ratio=%.6f digest=%s token_count=%s merge_count=%s dispatch_allocations=0 iter=%s runs=%s\n", \
          name, rust, cpp[name], ratio, digest, tokens, merges, iterations, runs
        seen[name] = 1
      }
      END {
        if (length(cpp) != 3 || length(seen) != 3) failed = 1
        exit failed
      }
    ' "$vocabulary_work/cpp.out" "$vocabulary_work/rust.out"
  } >"$CURRENT"
  cmake -E remove_directory "$vocabulary_work"
elif [[ "$SUITE" == "io-mmap" ]]; then
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
    FNR==NR {
      if($0~/^#/){print;next}
      if($1=="model/tensor/rust/plan_mapped_64"){rp=field("ns_per_op");it=field("iter");runs=field("runs")}
      if($1=="model/tensor/rust/direct_read_4k"){rd=field("ns_per_op")}
      next
    }
    $1=="model/tensor/reference/plan_mapped_64" {cp=field("ns_per_op")}
    $1=="model/tensor/reference/direct_read_4k" {cd=field("ns_per_op")}
    END {
      printf "model/tensor/plan_mapped_64 rust_ns_per_op=%.3f cpp_ns_per_op=%.3f rust_vs_cpp_ratio=%.6f iter=%s runs=%s\n",rp,cp,rp/cp,it,runs
      printf "model/tensor/direct_read_4k rust_ns_per_op=%.3f cpp_ns_per_op=%.3f rust_vs_cpp_ratio=%.6f iter=%s runs=%s\n",rd,cd,rd/cd,it,runs
    }
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

if [[ "$SUITE" == "gguf" || "$SUITE" == "io-read" || "$SUITE" == "io-mmap" || "$SUITE" == "io-staged-read" || "$SUITE" == "io-loader" || "$SUITE" == "model-tensor" || "$SUITE" == "model-data" || "$SUITE" == "model-vocabulary" || "$SUITE" == "token-profile" || "$SUITE" == "kernel-capability" ]]; then
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

validate_gguf_artifact() {
  local artifact="$1"
  local label="$2"
  local expected_iterations="$3"
  local expected_runs="$4"
  local expected
  for expected in \
    '# source_repository: ggml-org/llama.cpp' \
    '# source_commit: ecbcb7ea9d3303097519723b264a8b5f1e977028' \
    '# contract_source_repository: stateforward/emel.cpp' \
    '# contract_source_commit: 843a117386ef17dc5a50549bbfc821074c2141d6' \
    '# contract_source_blobs: kernel_events=4b3f02fa5ff9c5071d1fc83938cef1b22b4658b9 kernel_detail=c8a82643eabfe8f2d7883e655955f455794511b0 gguf_loader_detail=4ec9829c6d640cfa65a6d546c5891cb048b29f90' \
    '# benchmark_fixture: metadata_entries=64 tensor_entries=64 tensor_shape=256x4 tensor_type=f32 packed_tensor_entries=64 packed_wire_codes=41,42 packed_shapes=256x9,512x8 packed_tensor_bytes=2304 string_array_elements=4096 string_element_encoding=le_u64_length+le_u64_index' \
    '# benchmark_validation: public_typed_outcomes every_timed_observation serialized_q4_k_bytes=9437184 serialized_packed_41_bytes=2304 serialized_packed_42_bytes=2304 exact_string_visit_count=4096 exact_string_visit_order=0..4095 expected_string_fnv1a=0x743e126dc9e1e125' \
    '# benchmark_allocation: probe times Arc capability clone; load includes caller preallocation; parse reuses bound storage; string query times allocation-free RTC dispatch over a preloaded actor'; do
    if [[ "$(grep -Fxc "$expected" "$artifact")" -ne 1 ]]; then
      echo "error: $label must contain exactly one GGUF benchmark field: $expected" >&2
      exit 1
    fi
  done
  if ! awk -v expected_iterations="$expected_iterations" -v expected_runs="$expected_runs" '
    /^#/ { next }
    $1 == "gguf/probe/metadata_64" || $1 == "gguf/load/metadata_64" ||
    $1 == "gguf/probe/tensors_64" || $1 == "gguf/parse/tensors_64" ||
    $1 == "gguf/parse/packed_41_42_tensors_64" ||
    $1 == "gguf/load/tensors_64" || $1 == "gguf/query/string_array_4096" ||
    $1 == "tensor/dtype/serialized_size" {
      cases[$1] += 1
      case_name = $1
      for (i = 2; i <= NF; ++i) {
        if (split($i, part, "=") != 2) {
          invalid = 1
          continue
        }
        key = part[1]
        if (key != "ns_per_op" && key != "iter" && key != "runs") {
          invalid = 1
          continue
        }
        compound = case_name SUBSEP key
        if (seen[compound]++) {
          invalid = 1
          continue
        }
        values[compound] = part[2]
      }
      valid = seen[case_name SUBSEP "ns_per_op"] == 1 &&
        values[case_name SUBSEP "ns_per_op"] + 0 > 0
      valid = valid && seen[case_name SUBSEP "iter"] == 1 &&
        ("value:" values[case_name SUBSEP "iter"]) == ("value:" expected_iterations)
      valid = valid && seen[case_name SUBSEP "runs"] == 1 &&
        ("value:" values[case_name SUBSEP "runs"]) == ("value:" expected_runs)
      if (!valid) invalid = 1
      next
    }
    NF { invalid = 1 }
    END {
      exit invalid || length(cases) != 8 ||
        cases["gguf/probe/metadata_64"] != 1 || cases["gguf/load/metadata_64"] != 1 ||
        cases["gguf/probe/tensors_64"] != 1 || cases["gguf/parse/tensors_64"] != 1 ||
        cases["gguf/parse/packed_41_42_tensors_64"] != 1 ||
        cases["gguf/load/tensors_64"] != 1 || cases["gguf/query/string_array_4096"] != 1 ||
        cases["tensor/dtype/serialized_size"] != 1
    }
  ' "$artifact"; then
    echo "error: $label has invalid or configuration-incoherent GGUF benchmark cases" >&2
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

validate_model_data_artifact() {
  local artifact="$1"
  local label="$2"
  local expected
  for expected in \
    '# source_repository: stateforward/emel.cpp' \
    '# source_commit: 843a117386ef17dc5a50549bbfc821074c2141d6' \
    '# source_header_blob: 78a25b987423d8cbef17965a8ca92596ffc0ecef' \
    '# benchmark_fixture: complete fixed-capacity model data schema' \
    '# benchmark_validation: safe construction; allocation-free reset and accessors'; do
    if [[ "$(grep -Fxc "$expected" "$artifact")" -ne 1 ]]; then
      echo "error: $label must contain exactly one model-data field: $expected" >&2
      exit 1
    fi
  done
  if ! awk '
    /^#/ { next }
    $1 == "model/data/construction" || $1 == "model/data/reset" || $1 == "model/data/accessors" {
      cases[$1] += 1
      for (i = 2; i <= NF; ++i) {
        split($i, part, "=")
        if (part[1] == "ns_per_op" && part[2] + 0 > 0) timing[$1] += 1
      }
      next
    }
    NF { invalid = 1 }
    END {
      exit invalid || cases["model/data/construction"] != 1 ||
        cases["model/data/reset"] != 1 || cases["model/data/accessors"] != 1 ||
        timing["model/data/construction"] != 1 || timing["model/data/reset"] != 1 ||
        timing["model/data/accessors"] != 1
    }
  ' "$artifact"; then
    echo "error: $label has invalid model-data benchmark cases" >&2
    exit 1
  fi
}

validate_model_vocabulary_artifact() {
  local artifact="$1"
  local label="$2"
  local expected
  for expected in \
    '# source_repository: stateforward/emel.cpp' \
    '# source_commit: 843a117386ef17dc5a50549bbfc821074c2141d6' \
    '# source_tree: d2fd66887fbdef6e0894de9391155099839e625c' \
    '# source_detail_blob: 7c964f7449640fd7fe9eef21f4c14f3a65bf7b6a' \
    '# source_sml_commit: 49207123cd3f39767764bae774932cb48623f92f' \
    '# fixture_real_sha256: b046ac09ba24a848e2140676fba58c1dcf2f19617e45b03524043eabdb556a31' \
    '# benchmark_fixture: identical GGUF bytes in both lanes; small=standard_2_tokens_1_merge real=distilgpt2.Q2_K.gguf boundary=gemma4_2_tokens_400001_merges' \
    '# benchmark_timing: fixture IO, GGUF parse, archive/build, actor construction, and bounded storage allocation excluded; repeated vocabulary materialization only' \
    '# benchmark_validation: full semantic digest and typed counts match across public C++ model facade and public Rust Load actor lanes; Rust dispatch allocations=0'; do
    if [[ "$(grep -Fxc "$expected" "$artifact")" -ne 1 ]]; then
      echo "error: $label must contain exactly one model-vocabulary field: $expected" >&2
      exit 1
    fi
  done
  if ! awk '
    function value(name,    field_index, part) {
      for (field_index = 2; field_index <= NF; ++field_index) {
        split($field_index, part, "=")
        if (part[1] == name) return part[2]
      }
      return ""
    }
    /^# benchmark_config: / {
      expected_iterations = value("iterations")
      expected_runs = value("runs")
      next
    }
    /^# (reference_tool_sha256|rust_tool_sha256|fixture_small_sha256|fixture_boundary_sha256): / {
      hashes += 1
      if (length($3) != 64 || $3 !~ /^[0-9a-f]+$/) invalid = 1
      next
    }
    /^#/ { next }
    $1 == "model/vocabulary/materialize_small" ||
    $1 == "model/vocabulary/materialize_real_distilgpt2" ||
    $1 == "model/vocabulary/materialize_gemma4_400001" {
      cases[$1] += 1
      rust = value("rust_ns_per_op")
      cpp = value("cpp_ns_per_op")
      ratio = value("rust_vs_cpp_ratio")
      digest = value("digest")
      valid = rust + 0 > 0 && cpp + 0 > 0 && ratio + 0 > 0 && ratio + 0 <= 2.0
      valid = valid && ((rust / cpp) - ratio < 0.00001) && (ratio - (rust / cpp) < 0.00001)
      valid = valid && length(digest) == 16 && digest ~ /^[0-9a-f]+$/
      valid = valid && value("dispatch_allocations") == "0"
      valid = valid && value("token_count") ~ /^[0-9]+$/ && value("merge_count") ~ /^[0-9]+$/
      valid = valid && value("iter") == expected_iterations && value("runs") == expected_runs
      if (!valid) invalid = 1
      next
    }
    NF { invalid = 1 }
    END {
      exit invalid || hashes != 4 || cases["model/vocabulary/materialize_small"] != 1 ||
        cases["model/vocabulary/materialize_real_distilgpt2"] != 1 ||
        cases["model/vocabulary/materialize_gemma4_400001"] != 1
    }
  ' "$artifact"; then
    echo "error: $label has invalid model-vocabulary benchmark cases" >&2
    exit 1
  fi
}

validate_token_profile_artifact() {
  local artifact="$1"
  local label="$2"
  local expected
  for expected in \
    '# source_repository: stateforward/emel.cpp' \
    '# source_commit: 843a117386ef17dc5a50549bbfc821074c2141d6' \
    '# source_model_blob: ef7ff8da51f1f281082901bf4919a4b9a63f2671' \
    '# source_pre_blob: 16b2982ca16dfdfbee016d50d0eb924a7cdc18c4' \
    '# benchmark_fixture: public Resolver/Resolve model=gpt2 pre=solar-open' \
    '# benchmark_validation: typed success checked each iteration; allocation separately proven'; do
    if [[ "$(grep -Fxc "$expected" "$artifact")" -ne 1 ]]; then
      echo "error: $label must contain exactly one token-profile field: $expected" >&2
      exit 1
    fi
  done
  if ! awk '
    /^#/ { next }
    $1 == "token/profile/resolve_worst_known" {
      cases += 1
      for (i = 2; i <= NF; ++i) {
        split($i, part, "=")
        if (part[1] == "ns_per_op" && part[2] + 0 > 0) timing += 1
      }
      next
    }
    NF { invalid = 1 }
    END { exit invalid || cases != 1 || timing != 1 }
  ' "$artifact"; then
    echo "error: $label has an invalid token-profile benchmark case" >&2
    exit 1
  fi
}

validate_kernel_capability_artifact() {
  local artifact="$1"
  local label="$2"
  local expected
  for expected in \
    '# source_repository: stateforward/emel.cpp' \
    '# source_commit: 843a117386ef17dc5a50549bbfc821074c2141d6' \
    '# source_kernel_events_blob: 4b3f02fa5ff9c5071d1fc83938cef1b22b4658b9' \
    '# source_kernel_detail_blob: c8a82643eabfe8f2d7883e655955f455794511b0' \
    '# source_generation_header_blob: d521cf68e1bf52a2a193bbdb460741772199b318' \
    '# source_generation_blob: 099058ccd441d1dc6bebbb0c4994070d2f533c47' \
    '# benchmark_fixture: same MatrixWeightContract tensor_type=Q4_K expected=NativeQuantized in both public owner APIs' \
    '# benchmark_timing: Rust public Resolver dispatch versus pinned C++ public 14-stage generation audit; 100 warmups then median 5x1000 by default' \
    '# benchmark_validation: every observation is native_quantized; Rust dispatch allocation separately proven zero' \
    '# benchmark_interpretation: conservative source-owner comparison; C++ timing includes construction and traversal of its complete audit result'; do
    if [[ "$(grep -Fxc "$expected" "$artifact")" -ne 1 ]]; then
      echo "error: $label must contain exactly one kernel-capability field: $expected" >&2
      exit 1
    fi
  done
  if ! awk '
    function value(name,    i, part) {
      for (i = 2; i <= NF; ++i) {
        split($i, part, "=")
        if (part[1] == name) return part[2]
      }
      return ""
    }
    /^# benchmark_config: / {
      expected_iterations = value("iterations")
      expected_runs = value("runs")
      next
    }
    /^#/ { next }
    $1 == "kernel/capability/resolve_matrix_q4_k" {
      cases += 1
      rust = value("rust_ns_per_op")
      cpp = value("cpp_ns_per_op")
      ratio = value("rust_vs_cpp_ratio")
      valid = rust + 0 > 0 && cpp + 0 > 0 && ratio + 0 > 0 && ratio + 0 <= 2.0
      valid = valid && ((rust / cpp) - ratio < 0.00001) && (ratio - (rust / cpp) < 0.00001)
      valid = valid && value("outcome") == "native_quantized"
      valid = valid && value("dispatch_allocations") == "0"
      valid = valid && value("iter") == expected_iterations && value("runs") == expected_runs
      if (!valid) invalid = 1
      next
    }
    NF { invalid = 1 }
    END { exit invalid || cases != 1 }
  ' "$artifact"; then
    echo "error: $label has an invalid kernel-capability benchmark case" >&2
    exit 1
  fi
}

if [[ "$SUITE" == "gguf" ]]; then
  current_config_values="$(validate_io_read_config "$CURRENT" "current benchmark artifact" "$pointer_width")"
  current_iterations="${current_config_values%% *}"
  current_runs="${current_config_values#* }"
  validate_gguf_artifact \
    "$CURRENT" "current benchmark artifact" "$current_iterations" "$current_runs"
elif [[ "$SUITE" == "io-read" ]]; then
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
elif [[ "$SUITE" == "model-data" ]]; then
  current_config_values="$(validate_io_read_config "$CURRENT" "current benchmark artifact" "$pointer_width")"
  validate_model_data_artifact "$CURRENT" "current benchmark artifact"
elif [[ "$SUITE" == "model-vocabulary" ]]; then
  current_config_values="$(validate_io_read_config "$CURRENT" "current benchmark artifact" "$pointer_width")"
  validate_model_vocabulary_artifact "$CURRENT" "current benchmark artifact"
elif [[ "$SUITE" == "token-profile" ]]; then
  current_config_values="$(validate_io_read_config "$CURRENT" "current benchmark artifact" "$pointer_width")"
  validate_token_profile_artifact "$CURRENT" "current benchmark artifact"
elif [[ "$SUITE" == "kernel-capability" ]]; then
  current_config_values="$(validate_io_read_config "$CURRENT" "current benchmark artifact" "$pointer_width")"
  validate_kernel_capability_artifact "$CURRENT" "current benchmark artifact"
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

if [[ "$SUITE" == "gguf" ]]; then
  baseline_arch="$(validate_io_read_arch "$BASELINE" "benchmark baseline")"
  baseline_pointer_width="$(validate_io_read_pointer_width "$BASELINE" "benchmark baseline")"
  baseline_config_values="$(validate_io_read_config "$BASELINE" "benchmark baseline" "$baseline_pointer_width")"
  baseline_iterations="${baseline_config_values%% *}"
  baseline_runs="${baseline_config_values#* }"
  validate_gguf_artifact \
    "$BASELINE" "benchmark baseline" "$baseline_iterations" "$baseline_runs"
  if [[ "$baseline_arch" != "$host_arch" || "$baseline_pointer_width" != "$pointer_width" ]]; then
    echo "error: GGUF benchmark baseline architecture differs from current" >&2
    exit 1
  fi
  for field in source_repository source_commit contract_source_repository contract_source_commit contract_source_blobs benchmark_fixture benchmark_validation benchmark_allocation; do
    [[ "$(grep "^# $field: " "$BASELINE")" == "$(grep "^# $field: " "$CURRENT")" ]] || {
      echo "error: GGUF benchmark $field differs from baseline" >&2; exit 1; }
  done
elif [[ "$SUITE" == "io-read" ]]; then
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
elif [[ "$SUITE" == "model-data" ]]; then
  baseline_arch="$(validate_io_read_arch "$BASELINE" "benchmark baseline")"
  baseline_pointer_width="$(validate_io_read_pointer_width "$BASELINE" "benchmark baseline")"
  validate_io_read_config "$BASELINE" "benchmark baseline" "$baseline_pointer_width" >/dev/null
  validate_model_data_artifact "$BASELINE" "benchmark baseline"
  if [[ "$baseline_arch" != "$host_arch" || "$baseline_pointer_width" != "$pointer_width" ]]; then
    echo "error: model data benchmark architecture differs from baseline" >&2
    exit 1
  fi
  for field in source_repository source_commit source_header_blob benchmark_fixture benchmark_validation; do
    [[ "$(grep "^# $field: " "$BASELINE")" == "$(grep "^# $field: " "$CURRENT")" ]] || {
      echo "error: model data benchmark $field differs from baseline" >&2
      exit 1
    }
  done
elif [[ "$SUITE" == "model-vocabulary" ]]; then
  baseline_arch="$(validate_io_read_arch "$BASELINE" "benchmark baseline")"
  baseline_pointer_width="$(validate_io_read_pointer_width "$BASELINE" "benchmark baseline")"
  validate_io_read_config "$BASELINE" "benchmark baseline" "$baseline_pointer_width" >/dev/null
  validate_model_vocabulary_artifact "$BASELINE" "benchmark baseline"
  if [[ "$baseline_arch" != "$host_arch" || "$baseline_pointer_width" != "$pointer_width" ]]; then
    echo "error: model vocabulary benchmark architecture differs from baseline" >&2
    exit 1
  fi
  for field in source_repository source_commit source_tree source_detail_blob source_sml_commit reference_tool_sha256 rust_tool_sha256 fixture_small_sha256 fixture_real_sha256 fixture_boundary_sha256 benchmark_fixture benchmark_timing benchmark_validation; do
    [[ "$(grep "^# $field: " "$BASELINE")" == "$(grep "^# $field: " "$CURRENT")" ]] || {
      echo "error: model vocabulary benchmark $field differs from baseline" >&2
      exit 1
    }
  done
elif [[ "$SUITE" == "token-profile" ]]; then
  baseline_arch="$(validate_io_read_arch "$BASELINE" "benchmark baseline")"
  baseline_pointer_width="$(validate_io_read_pointer_width "$BASELINE" "benchmark baseline")"
  validate_io_read_config "$BASELINE" "benchmark baseline" "$baseline_pointer_width" >/dev/null
  validate_token_profile_artifact "$BASELINE" "benchmark baseline"
  if [[ "$baseline_arch" != "$host_arch" || "$baseline_pointer_width" != "$pointer_width" ]]; then
    echo "error: token profile benchmark architecture differs from baseline" >&2
    exit 1
  fi
  for field in source_repository source_commit source_model_blob source_pre_blob benchmark_fixture benchmark_validation; do
    [[ "$(grep "^# $field: " "$BASELINE")" == "$(grep "^# $field: " "$CURRENT")" ]] || {
      echo "error: token profile benchmark $field differs from baseline" >&2
      exit 1
    }
  done
elif [[ "$SUITE" == "kernel-capability" ]]; then
  baseline_arch="$(validate_io_read_arch "$BASELINE" "benchmark baseline")"
  baseline_pointer_width="$(validate_io_read_pointer_width "$BASELINE" "benchmark baseline")"
  validate_io_read_config "$BASELINE" "benchmark baseline" "$baseline_pointer_width" >/dev/null
  validate_kernel_capability_artifact "$BASELINE" "benchmark baseline"
  if [[ "$baseline_arch" != "$host_arch" || "$baseline_pointer_width" != "$pointer_width" ]]; then
    echo "error: kernel capability benchmark architecture differs from baseline" >&2
    exit 1
  fi
  for field in source_repository source_commit source_kernel_events_blob source_kernel_detail_blob source_generation_header_blob source_generation_blob benchmark_fixture benchmark_timing benchmark_validation benchmark_interpretation; do
    [[ "$(grep "^# $field: " "$BASELINE")" == "$(grep "^# $field: " "$CURRENT")" ]] || {
      echo "error: kernel capability benchmark $field differs from baseline" >&2
      exit 1
    }
  done
fi

max_ratio="${EMEL_BENCH_MAX_REGRESSION_RATIO:-2.0}"
if [[ "$SUITE" == "io-mmap" || "$SUITE" == "io-staged-read" || "$SUITE" == "io-loader" || "$SUITE" == "model-tensor" || "$SUITE" == "model-vocabulary" || "$SUITE" == "kernel-capability" ]]; then
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
