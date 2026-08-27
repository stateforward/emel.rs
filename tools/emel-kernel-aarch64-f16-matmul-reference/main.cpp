#include <array>
#include <bit>
#include <cstdint>
#include <iomanip>
#include <iostream>

#include "emel/kernel/sm.hpp"

namespace {

constexpr char k_source_commit[] =
    "843a117386ef17dc5a50549bbfc821074c2141d6";
constexpr std::array<std::uint16_t, 6> k_lhs = {
    0x3c00, 0x4000, 0x4200, 0x4400, 0x4500, 0x4600};
constexpr std::array<std::uint16_t, 6> k_rhs = {
    0x3c00, 0x0000, 0x0000, 0x3c00, 0x3c00, 0x3c00};
constexpr std::uint32_t k_sentinel = 0x7fc01234U;

emel::kernel::event::tensor_view f16_view(
    const std::uint16_t *data, const std::array<std::uint64_t, 4> ne,
    const std::array<std::uint64_t, 4> nb) {
  return {.data = data,
          .type = emel::kernel::event::dtype::f16,
          .ne = ne,
          .nb = nb};
}

emel::kernel::event::tensor_view_mut output_view(
    float *data, const emel::kernel::event::dtype type,
    const std::array<std::uint64_t, 4> ne,
    const std::array<std::uint64_t, 4> nb) {
  return {.data = data, .type = type, .ne = ne, .nb = nb};
}

template <std::size_t N>
void print_case(const char *name, const bool status,
                const std::array<float, N> &output) {
  std::cout << "case=" << name << " status=" << (status ? "ok" : "error")
            << " output_bits=";
  for (std::size_t index = 0; index < N; ++index) {
    if (index != 0) std::cout << ',';
    std::cout << std::hex << std::setw(8) << std::setfill('0')
              << std::bit_cast<std::uint32_t>(output[index]);
  }
  std::cout << std::dec << '\n';
}

void run_matrix() {
  std::array<float, 4> output{};
  emel::kernel::event::op_mul_mat request{};
  request.src0 = f16_view(k_lhs.data(), {3, 2, 1, 1}, {2, 6, 12, 12});
  request.src1 = f16_view(k_rhs.data(), {3, 2, 1, 1}, {2, 6, 12, 12});
  request.dst = output_view(output.data(), emel::kernel::event::dtype::f32,
                            {2, 2, 1, 1}, {4, 8, 16, 16});
  print_case("matrix", emel::kernel::detail::run_mul_mat_f16(request), output);
}

void run_invalid_shape() {
  std::array<float, 4> output = {std::bit_cast<float>(k_sentinel),
                                 std::bit_cast<float>(k_sentinel),
                                 std::bit_cast<float>(k_sentinel),
                                 std::bit_cast<float>(k_sentinel)};
  emel::kernel::Kernel kernel;
  emel::kernel::event::op_mul_mat request{};
  request.src0 = f16_view(k_lhs.data(), {3, 2, 1, 1}, {2, 6, 12, 12});
  request.src1 = f16_view(k_rhs.data(), {2, 2, 1, 1}, {2, 4, 8, 8});
  request.dst = output_view(output.data(), emel::kernel::event::dtype::f32,
                            {2, 2, 1, 1}, {4, 8, 16, 16});
  print_case("invalid_shape", kernel.process_event(request), output);
}

void run_invalid_dtype() {
  std::array<float, 4> output = {std::bit_cast<float>(k_sentinel),
                                 std::bit_cast<float>(k_sentinel),
                                 std::bit_cast<float>(k_sentinel),
                                 std::bit_cast<float>(k_sentinel)};
  emel::kernel::Kernel kernel;
  emel::kernel::event::op_mul_mat request{};
  request.src0 = f16_view(k_lhs.data(), {3, 2, 1, 1}, {2, 6, 12, 12});
  request.src1 = f16_view(k_rhs.data(), {3, 2, 1, 1}, {2, 6, 12, 12});
  request.dst = output_view(output.data(), emel::kernel::event::dtype::f16,
                            {2, 2, 1, 1}, {2, 4, 8, 8});
  print_case("invalid_dtype", kernel.process_event(request), output);
}

}  // namespace

int main() {
  std::cout << "kernel-aarch64-f16-matmul-live/v1\n"
            << "source_repository=stateforward/emel.cpp\n"
            << "source_commit=" << k_source_commit << '\n'
            << "source_kernel_events_blob=4b3f02fa5ff9c5071d1fc83938cef1b22b4658b9\n"
            << "source_kernel_detail_blob=c8a82643eabfe8f2d7883e655955f455794511b0\n"
            << "source_kernel_aarch64_guards_blob=c25714566ec9a02679daef85089544575123408e\n"
            << "source_kernel_aarch64_actions_blob=267d4f74e6e7498155c8535920322ffef2c02fb6\n"
            << "source_kernel_aarch64_sm_blob=865a9cc6ba6115382ed043c464f3d62bcd851357\n"
            << "source_detail_span=src/emel/kernel/detail.hpp:3874-3894,4198-4214,5089-5096\n"
            << "source_guard_span=src/emel/kernel/aarch64/guards.hpp:275-285,973-994\n"
            << "source_action_span=src/emel/kernel/aarch64/actions.hpp:7044-7064,9213-9216\n"
            << "source_transition_span=src/emel/kernel/aarch64/sm.hpp:525-543\n"
            << "scope=dense_f16_matmul_target_scalar_positive_shape_dtype_rejection\n";
  run_matrix();
  run_invalid_shape();
  run_invalid_dtype();
  return 0;
}
