#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <string_view>
#include <stdexcept>
#include <vector>

#include "emel/kernel/sm.hpp"

namespace {

constexpr char k_source_commit[] =
    "843a117386ef17dc5a50549bbfc821074c2141d6";

emel::kernel::event::tensor_view view(const float *data, const std::size_t count) {
  const auto bytes = static_cast<std::uint64_t>(count * sizeof(float));
  return {.data = data,
          .type = emel::kernel::event::dtype::f32,
          .ne = {count, 1, 1, 1},
          .nb = {sizeof(float), bytes, bytes, bytes}};
}

emel::kernel::event::tensor_view_mut view_mut(float *data,
                                               const std::size_t count) {
  const auto bytes = static_cast<std::uint64_t>(count * sizeof(float));
  return {.data = data,
          .type = emel::kernel::event::dtype::f32,
          .ne = {count, 1, 1, 1},
          .nb = {sizeof(float), bytes, bytes, bytes}};
}

std::string bits(const float *values, const std::size_t count) {
  std::ostringstream output;
  for (std::size_t index = 0; index < count; ++index) {
    if (index != 0) {
      output << ',';
    }
    output << std::hex << std::setw(8) << std::setfill('0')
           << std::bit_cast<std::uint32_t>(values[index]);
  }
  return output.str();
}

template <typename operation>
bool execute(emel::kernel::Kernel &kernel, operation &request) {
  return kernel.process_event(request);
}

std::uint64_t parse_positive(const char *value) {
  const auto parsed = std::stoull(value);
  if (parsed == 0) {
    throw std::invalid_argument("benchmark values must be positive");
  }
  return parsed;
}

double median(std::vector<double> samples) {
  std::sort(samples.begin(), samples.end());
  const auto middle = samples.size() / 2;
  if (samples.size() % 2 == 0) {
    return (samples[middle - 1] + samples[middle]) / 2.0;
  }
  return samples[middle];
}

double benchmark_add(const std::uint64_t iterations, const std::uint64_t warmup,
                     const std::size_t runs) {
  constexpr std::size_t count = 4096;
  const std::array<float, count> lhs = [] {
    std::array<float, count> values{};
    values.fill(1.25F);
    return values;
  }();
  const std::array<float, count> rhs = [] {
    std::array<float, count> values{};
    values.fill(0.75F);
    return values;
  }();
  std::array<float, count> output{};
  emel::kernel::Kernel kernel;
  auto request = emel::kernel::event::op_add{};
  request.src0 = view(lhs.data(), lhs.size());
  request.src1 = view(rhs.data(), rhs.size());
  request.dst = view_mut(output.data(), output.size());
  for (std::uint64_t index = 0; index < warmup; ++index) {
    if (!execute(kernel, request)) {
      throw std::runtime_error("reference op_add rejected benchmark request");
    }
  }
  volatile float checksum = output[count - 1];
  std::vector<double> samples;
  samples.reserve(runs);
  for (std::size_t run = 0; run < runs; ++run) {
    const auto started = std::chrono::steady_clock::now();
    for (std::uint64_t index = 0; index < iterations; ++index) {
      if (!execute(kernel, request)) {
        throw std::runtime_error("reference op_add rejected benchmark request");
      }
    }
    checksum += output[count - 1];
    const auto elapsed = std::chrono::duration<double, std::nano>(
        std::chrono::steady_clock::now() - started);
    samples.push_back(elapsed.count() / static_cast<double>(iterations));
  }
  if (checksum == 0.0F) {
    throw std::runtime_error("reference op_add benchmark checksum is zero");
  }
  return median(std::move(samples));
}

double benchmark_mul(const std::uint64_t iterations, const std::uint64_t warmup,
                     const std::size_t runs) {
  constexpr std::size_t count = 4096;
  const std::array<float, count> lhs = [] {
    std::array<float, count> values{};
    values.fill(1.25F);
    return values;
  }();
  const std::array<float, count> rhs = [] {
    std::array<float, count> values{};
    values.fill(0.75F);
    return values;
  }();
  std::array<float, count> output{};
  emel::kernel::Kernel kernel;
  auto request = emel::kernel::event::op_mul{};
  request.src0 = view(lhs.data(), lhs.size());
  request.src1 = view(rhs.data(), rhs.size());
  request.dst = view_mut(output.data(), output.size());
  for (std::uint64_t index = 0; index < warmup; ++index) {
    if (!execute(kernel, request)) {
      throw std::runtime_error("reference op_mul rejected benchmark request");
    }
  }
  volatile float checksum = output[count - 1];
  std::vector<double> samples;
  samples.reserve(runs);
  for (std::size_t run = 0; run < runs; ++run) {
    const auto started = std::chrono::steady_clock::now();
    for (std::uint64_t index = 0; index < iterations; ++index) {
      if (!execute(kernel, request)) {
        throw std::runtime_error("reference op_mul rejected benchmark request");
      }
    }
    checksum += output[count - 1];
    const auto elapsed = std::chrono::duration<double, std::nano>(
        std::chrono::steady_clock::now() - started);
    samples.push_back(elapsed.count() / static_cast<double>(iterations));
  }
  if (checksum == 0.0F) {
    throw std::runtime_error("reference op_mul benchmark checksum is zero");
  }
  return median(std::move(samples));
}

void write_benchmark(const std::uint64_t iterations, const std::size_t runs,
                     const std::uint64_t warmup) {
  const auto add = benchmark_add(iterations, warmup, runs);
  const auto mul = benchmark_mul(iterations, warmup, runs);
  std::cout << "kernel-elementwise-bench/v1\n"
            << "source_repository=stateforward/emel.cpp\n"
            << "source_commit=" << k_source_commit << '\n'
            << "source_kernel_events_blob=4b3f02fa5ff9c5071d1fc83938cef1b22b4658b9\n"
            << "source_kernel_detail_blob=c8a82643eabfe8f2d7883e655955f455794511b0\n"
            << "source_kernel_any_blob=a85d0942c81ef8f4d2dc293c5ceeaa63ac0e2c49\n"
            << "benchmark_operand=dense_contiguous_f32,count=4096,lhs=1.25,rhs=0.75\n"
            << std::fixed << std::setprecision(3)
            << "case=op_add cpp_ns_per_op=" << add
            << " output=1.875 iter=" << iterations << " runs=" << runs
            << " warmup=" << warmup << '\n'
            << "case=op_mul cpp_ns_per_op=" << mul
            << " output=0.9375 iter=" << iterations << " runs=" << runs
            << " warmup=" << warmup << '\n';
}

}  // namespace

int main(const int argc, char **argv) {
  if (argc == 5 && std::string_view{argv[1]} == "--benchmark") {
    try {
      write_benchmark(parse_positive(argv[2]),
                      static_cast<std::size_t>(parse_positive(argv[3])),
                      parse_positive(argv[4]));
      return 0;
    } catch (const std::exception &error) {
      std::cerr << "error: " << error.what() << '\n';
      return 1;
    }
  }
  if (argc != 1) {
    std::cerr << "usage: emel-kernel-elementwise-reference [--benchmark ITERATIONS RUNS WARMUP]\n";
    return 2;
  }
  std::cout << "kernel-elementwise-parity/v1\n"
            << "source_repository=stateforward/emel.cpp\n"
            << "source_commit=" << k_source_commit << '\n'
            << "source_kernel_events_blob=4b3f02fa5ff9c5071d1fc83938cef1b22b4658b9\n"
            << "source_kernel_detail_blob=c8a82643eabfe8f2d7883e655955f455794511b0\n"
            << "source_kernel_any_blob=a85d0942c81ef8f4d2dc293c5ceeaa63ac0e2c49\n"
            << "scope=dense_contiguous_f32\n";

  emel::kernel::Kernel kernel;
  const std::array<float, 3> input = {1.5F, -2.0F, 0.25F};
  const std::array<float, 3> other = {-0.5F, 4.0F, 2.0F};
  std::array<float, 3> output = {};

  auto duplicate = emel::kernel::event::op_dup{};
  duplicate.src0 = view(input.data(), input.size());
  duplicate.dst = view_mut(output.data(), output.size());
  std::cout << "case=op_dup status=" << (execute(kernel, duplicate) ? "ok" : "invalid_shape")
            << " output_bits=" << bits(output.data(), output.size()) << '\n';

  auto add = emel::kernel::event::op_add{};
  add.src0 = view(input.data(), input.size());
  add.src1 = view(other.data(), other.size());
  add.dst = view_mut(output.data(), output.size());
  std::cout << "case=op_add status=" << (execute(kernel, add) ? "ok" : "invalid_shape")
            << " output_bits=" << bits(output.data(), output.size()) << '\n';

  auto sub = emel::kernel::event::op_sub{};
  sub.src0 = view(input.data(), input.size());
  sub.src1 = view(other.data(), other.size());
  sub.dst = view_mut(output.data(), output.size());
  std::cout << "case=op_sub status=" << (execute(kernel, sub) ? "ok" : "invalid_shape")
            << " output_bits=" << bits(output.data(), output.size()) << '\n';

  auto mul = emel::kernel::event::op_mul{};
  mul.src0 = view(input.data(), input.size());
  mul.src1 = view(other.data(), other.size());
  mul.dst = view_mut(output.data(), output.size());
  std::cout << "case=op_mul status=" << (execute(kernel, mul) ? "ok" : "invalid_shape")
            << " output_bits=" << bits(output.data(), output.size()) << '\n';

  auto div = emel::kernel::event::op_div{};
  div.src0 = view(input.data(), input.size());
  div.src1 = view(other.data(), other.size());
  div.dst = view_mut(output.data(), output.size());
  std::cout << "case=op_div status=" << (execute(kernel, div) ? "ok" : "invalid_shape")
            << " output_bits=" << bits(output.data(), output.size()) << '\n';

  auto sqr = emel::kernel::event::op_sqr{};
  sqr.src0 = view(input.data(), input.size());
  sqr.dst = view_mut(output.data(), output.size());
  std::cout << "case=op_sqr status=" << (execute(kernel, sqr) ? "ok" : "invalid_shape")
            << " output_bits=" << bits(output.data(), output.size()) << '\n';

  const std::array<float, 3> sqrt_input = {0.0F, 4.0F, 9.0F};
  auto sqrt = emel::kernel::event::op_sqrt{};
  sqrt.src0 = view(sqrt_input.data(), sqrt_input.size());
  sqrt.dst = view_mut(output.data(), output.size());
  std::cout << "case=op_sqrt status=" << (execute(kernel, sqrt) ? "ok" : "invalid_shape")
            << " output_bits=" << bits(output.data(), output.size()) << '\n';

  std::array<float, 2> short_output = {};
  auto invalid = emel::kernel::event::op_dup{};
  invalid.src0 = view(input.data(), input.size());
  invalid.dst = view_mut(short_output.data(), short_output.size());
  std::cout << "case=invalid_shape status="
            << (execute(kernel, invalid) ? "ok" : "invalid_shape") << '\n';

}
