#include <array>
#include <bit>
#include <cstdint>
#include <iomanip>
#include <iostream>

#include "emel/kernel/sm.hpp"

namespace {

constexpr char k_source_commit[] =
    "843a117386ef17dc5a50549bbfc821074c2141d6";

template <std::size_t N>
emel::kernel::event::tensor_view view(const std::array<float, N> &data) {
  return {.data = data.data(),
          .type = emel::kernel::event::dtype::f32,
          .ne = {N, 1, 1, 1},
          .nb = {4, 4 * N, 4 * N, 4 * N}};
}

template <std::size_t N>
emel::kernel::event::tensor_view_mut view_mut(std::array<float, N> &data) {
  return {.data = data.data(),
          .type = emel::kernel::event::dtype::f32,
          .ne = {N, 1, 1, 1},
          .nb = {4, 4 * N, 4 * N, 4 * N}};
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
void run_valid(const char *name, const std::array<float, N> &input) {
  std::array<float, N> output{};
  emel::kernel::Kernel kernel;
  emel::kernel::event::op_unary request{};
  request.src0 = view(input);
  request.dst = view_mut(output);
  request.subop = emel::kernel::event::unary_subop::silu;
  print_case(name, kernel.process_event(request), output);
}

void run_invalid() {
  constexpr std::array<float, 3> input = {1.0F, 2.0F, 3.0F};
  constexpr std::uint32_t k_sentinel = 0x7fc01234U;
  std::array<float, 2> output = {std::bit_cast<float>(k_sentinel),
                                 std::bit_cast<float>(k_sentinel)};
  emel::kernel::Kernel kernel;
  emel::kernel::event::op_unary request{};
  request.src0 = view(input);
  request.dst = view_mut(output);
  request.subop = emel::kernel::event::unary_subop::silu;
  print_case("invalid_shape", kernel.process_event(request), output);
}

}  // namespace

int main() {
  constexpr std::array<float, 4> vector_input = {-4.0F, -1.0F, 0.0F, 4.0F};
  constexpr std::array<float, 5> scalar_tail_input = {
      1.0F, 2.0F, 3.0F, -4.0F, 0.5F};

  std::cout << "kernel-aarch64-silu-live-parity/v1\n"
            << "source_repository=stateforward/emel.cpp\n"
            << "source_commit=" << k_source_commit << '\n'
            << "source_kernel_events_blob=4b3f02fa5ff9c5071d1fc83938cef1b22b4658b9\n"
            << "source_kernel_detail_blob=c8a82643eabfe8f2d7883e655955f455794511b0\n"
            << "source_kernel_aarch64_guards_blob=c25714566ec9a02679daef85089544575123408e\n"
            << "source_kernel_aarch64_actions_blob=267d4f74e6e7498155c8535920322ffef2c02fb6\n"
            << "source_kernel_aarch64_sm_blob=865a9cc6ba6115382ed043c464f3d62bcd851357\n"
            << "source_action_formula_span=src/emel/kernel/aarch64/actions.hpp:89-123,180-196\n"
            << "source_guard_span=src/emel/kernel/aarch64/guards.hpp:826-886\n"
            << "source_transition_span=src/emel/kernel/aarch64/sm.hpp:1161-1178\n"
            << "scope=dense_f32_neon_silu_vector_and_scalar_tail_invalid_no_mutation\n";
  run_valid("vector", vector_input);
  run_valid("scalar_tail", scalar_tail_input);
  run_invalid();
  return 0;
}
