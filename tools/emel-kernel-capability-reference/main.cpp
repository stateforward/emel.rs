#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string_view>
#include <vector>

#include "emel/model/generation/any.hpp"

namespace {

constexpr std::array<std::uint8_t, 34> k_types = {
    0,  1,  2,  3,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17,
    18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 34, 35, 39, 41, 42,
};

enum class scope { vector, matrix };

using outcome = emel::model::generation::quantized_contract_kind;

template <typename value_type> value_type opaque(value_type value) {
#if defined(__clang__) || defined(__GNUC__)
  __asm__ __volatile__("" : "+r"(value) : : "memory");
  return value;
#else
  volatile value_type observed = value;
  return observed;
#endif
}

outcome classify(const scope contract_scope, const std::uint8_t code) {
  emel::model::data::tensor_record tensor{};
  tensor.type = static_cast<std::int32_t>(code);
  emel::model::generation::execution_view execution{};
  if (contract_scope == scope::vector) {
    execution.token_embedding.tensor = &tensor;
  } else {
    execution.output.tensor = &tensor;
  }
  const auto audit =
      emel::model::generation::build_quantized_path_audit(execution);
  const auto family = contract_scope == scope::vector
                          ? emel::model::generation::quantized_stage_family::
                                token_embedding
                          : emel::model::generation::quantized_stage_family::
                                output;
  return audit.stages[static_cast<std::size_t>(family)].contract;
}

std::string_view outcome_name(const outcome value) {
  return emel::model::generation::quantized_contract_kind_name(value);
}

void write_parity() {
  std::cout << "kernel-capability-parity/v1\n"
            << "source_repository=stateforward/emel.cpp\n"
            << "source_commit=843a117386ef17dc5a50549bbfc821074c2141d6\n"
            << "source_kernel_events_blob=4b3f02fa5ff9c5071d1fc83938cef1b22b4658b9\n"
            << "source_kernel_detail_blob=c8a82643eabfe8f2d7883e655955f455794511b0\n"
            << "source_generation_header_blob=d521cf68e1bf52a2a193bbdb460741772199b318\n"
            << "source_generation_blob=099058ccd441d1dc6bebbb0c4994070d2f533c47\n";
  for (const std::uint8_t code : {std::uint8_t{0}, std::uint8_t{10},
                                  std::uint8_t{11}, std::uint8_t{12},
                                  std::uint8_t{14}, std::uint8_t{2},
                                  std::uint8_t{1}}) {
    const auto label = emel::model::generation::tensor_type_name(code);
    std::cout << "label=" << label << " value=" << label << '\n';
  }
  for (const auto [scope_name, contract_scope] :
       {std::pair{"vector", scope::vector}, std::pair{"matrix", scope::matrix}}) {
    for (const std::uint8_t code : k_types) {
      std::cout << "scope=" << scope_name << " code="
                << static_cast<unsigned>(code) << " outcome="
                << outcome_name(classify(contract_scope, code)) << '\n';
    }
  }
}

std::uint64_t parse_positive(const char *text) {
  const auto value = std::stoull(text);
  if (value == 0) {
    throw std::invalid_argument("benchmark values must be positive");
  }
  return value;
}

void run_queries(const std::uint64_t iterations, volatile std::uint64_t &digest) {
  for (std::uint64_t index = 0; index < iterations; ++index) {
    const auto value =
        opaque(classify(opaque(scope::matrix), opaque(std::uint8_t{12})));
    digest = digest + static_cast<std::uint64_t>(
                          value == outcome::native_quantized);
  }
}

void write_benchmark(const std::uint64_t iterations, const std::size_t runs,
                     const std::uint64_t warmup_iterations) {
  volatile std::uint64_t digest = 0;
  run_queries(warmup_iterations, digest);
  std::vector<double> samples;
  samples.reserve(runs);
  for (std::size_t run = 0; run < runs; ++run) {
    const auto started = std::chrono::steady_clock::now();
    run_queries(iterations, digest);
    const auto elapsed = std::chrono::duration<double, std::nano>(
        std::chrono::steady_clock::now() - started);
    samples.push_back(elapsed.count() / static_cast<double>(iterations));
  }
  std::sort(samples.begin(), samples.end());
  const double median = runs % 2 == 0
                            ? (samples[runs / 2 - 1] + samples[runs / 2]) / 2.0
                            : samples[runs / 2];
  if (digest == 0) {
    std::abort();
  }
  std::cout << std::fixed << std::setprecision(3)
            << "case=matrix_q4_k cpp_ns_per_op=" << median
            << " outcome=native_quantized iter=" << iterations
            << " runs=" << runs << '\n';
}

} // namespace

int main(const int argc, char **argv) {
  try {
    if (argc == 1 || (argc == 2 && std::string_view{argv[1]} == "--parity")) {
      write_parity();
      return 0;
    }
    if (argc == 5 && std::string_view{argv[1]} == "--benchmark") {
      write_benchmark(parse_positive(argv[2]),
                      static_cast<std::size_t>(parse_positive(argv[3])),
                      parse_positive(argv[4]));
      return 0;
    }
  } catch (const std::exception &error) {
    std::cerr << "error: " << error.what() << '\n';
    return 1;
  }
  std::cerr << "usage: emel-kernel-capability-reference "
               "[--parity|--benchmark ITERATIONS RUNS WARMUP]\n";
  return 2;
}
