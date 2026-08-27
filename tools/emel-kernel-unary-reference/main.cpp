#include <array>
#include <bit>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <chrono>
#include <string>

#include "emel/kernel/sm.hpp"

namespace {

constexpr char k_source_commit[] =
    "843a117386ef17dc5a50549bbfc821074c2141d6";

emel::kernel::event::tensor_view view(
    const float *data, const std::array<std::uint64_t, 4> ne,
    const std::array<std::uint64_t, 4> nb) {
  return {.data = data,
          .type = emel::kernel::event::dtype::f32,
          .ne = ne,
          .nb = nb};
}

emel::kernel::event::tensor_view_mut view_mut(
    float *data, const std::array<std::uint64_t, 4> ne,
    const std::array<std::uint64_t, 4> nb) {
  return {.data = data,
          .type = emel::kernel::event::dtype::f32,
          .ne = ne,
          .nb = nb};
}

template <std::size_t N>
void print_case(const char *name, const bool status,
                const std::array<float, N> &output) {
  std::cout << "case=" << name << " status=" << (status ? "ok" : "error")
            << " output_bits=";
  for (std::size_t index = 0; index < N; ++index) {
    if (index != 0) {
      std::cout << ',';
    }
    std::cout << std::hex << std::setw(8) << std::setfill('0')
              << std::bit_cast<std::uint32_t>(output[index]);
  }
  std::cout << std::dec << '\n';
}

template <std::size_t N>
void run_case(const char *name, const emel::kernel::event::unary_subop subop,
              const std::array<float, N> &input) {
  constexpr std::array<std::uint64_t, 4> ne = {N, 1, 1, 1};
  constexpr std::array<std::uint64_t, 4> nb = {4, 4 * N, 4 * N, 4 * N};
  std::array<float, N> output{};
  emel::kernel::Kernel kernel;
  emel::kernel::event::op_unary request{};
  request.src0 = view(input.data(), ne, nb);
  request.dst = view_mut(output.data(), ne, nb);
  request.subop = subop;
  print_case(name, kernel.process_event(request), output);
}

}  // namespace

int main(int argc, char **argv) {
  if (argc > 1 && std::string(argv[1]) == "--benchmark") {
    constexpr std::size_t kCount = 1024;
    constexpr std::size_t kIterations = 10000;
    constexpr std::size_t kWarmup = 1000;
    std::array<float, kCount> input{};
    std::array<float, kCount> output{};
    input.fill(1.25F);
    constexpr std::array<std::uint64_t, 4> ne = {kCount, 1, 1, 1};
    constexpr std::array<std::uint64_t, 4> nb = {4, 4 * kCount, 4 * kCount, 4 * kCount};
    emel::kernel::Kernel kernel;
    emel::kernel::event::op_unary request{};
    request.src0 = view(input.data(), ne, nb);
    request.dst = view_mut(output.data(), ne, nb);
    request.subop = emel::kernel::event::unary_subop::abs;
    for (std::size_t i = 0; i < kWarmup; ++i) {
      (void)kernel.process_event(request);
    }
    const auto start = std::chrono::steady_clock::now();
    for (std::size_t i = 0; i < kIterations; ++i) {
      (void)kernel.process_event(request);
    }
    const auto elapsed = std::chrono::duration<double, std::nano>(
        std::chrono::steady_clock::now() - start).count();
    std::cout << "case=op_unary_abs cpp_ns_per_dispatch="
              << (elapsed / static_cast<double>(kIterations))
              << " output=" << output[0] << '\n';
    return 0;
  }
  constexpr std::array<float, 8> input = {
      -10.1F, -10.0F, -1.2345F, -0.125F,
      0.0F,   0.12345F, 10.0F, 10.1F,
  };
  std::cout << "kernel-unary-parity/v1\n"
            << "source_repository=stateforward/emel.cpp\n"
            << "source_commit=" << k_source_commit << '\n'
            << "source_sml_commit=49207123cd3f39767764bae774932cb48623f92f\n"
            << "source_kernel_events_blob=4b3f02fa5ff9c5071d1fc83938cef1b22b4658b9\n"
            << "source_kernel_detail_blob=c8a82643eabfe8f2d7883e655955f455794511b0\n"
            << "source_kernel_x86_sm_blob=0b4d635ebbd0fbd52dbca8a2345547fb571205c8\n"
            << "source_supported_scalar_subops=abs,neg,relu,exp,tanh,elu,gelu,silu\n"
            << "scope=portable_f32_generic_op_unary_equal_count_dense\n";
  run_case("abs", emel::kernel::event::unary_subop::abs, input);
  run_case("neg", emel::kernel::event::unary_subop::neg, input);
  run_case("relu", emel::kernel::event::unary_subop::relu, input);
  run_case("exp", emel::kernel::event::unary_subop::exp, input);
  run_case("tanh", emel::kernel::event::unary_subop::tanh, input);
  run_case("elu", emel::kernel::event::unary_subop::elu, input);
  run_case("gelu", emel::kernel::event::unary_subop::gelu, input);
  run_case("silu", emel::kernel::event::unary_subop::silu, input);
  return 0;
}
