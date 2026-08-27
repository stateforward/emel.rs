#include <array>
#include <bit>
#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <string>

#include "emel/kernel/sm.hpp"

namespace {

constexpr char k_source_commit[] = "843a117386ef17dc5a50549bbfc821074c2141d6";

emel::kernel::event::tensor_view f32_view(
    const float *data, const std::array<std::uint64_t, 4> ne,
    const std::array<std::uint64_t, 4> nb) {
  return {.data = data,
          .type = emel::kernel::event::dtype::f32,
          .ne = ne,
          .nb = nb};
}

emel::kernel::event::tensor_view_mut f32_view_mut(
    float *data, const std::array<std::uint64_t, 4> ne,
    const std::array<std::uint64_t, 4> nb) {
  return {.data = data,
          .type = emel::kernel::event::dtype::f32,
          .ne = ne,
          .nb = nb};
}

template <std::size_t N>
void print_matrix(const char *name, const bool status,
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

}  // namespace

int main(int argc, char **argv) {
  if (argc > 1 && std::string(argv[1]) == "--benchmark") {
    constexpr std::size_t k = 16;
    constexpr std::size_t iterations = 10000;
    constexpr std::size_t warmup = 1000;
    std::array<float, k> lhs{};
    std::array<float, k> rhs{};
    lhs.fill(1.0F);
    rhs.fill(1.0F);
    constexpr std::array<std::uint64_t, 4> lhs_ne = {k, 1, 1, 1};
    constexpr std::array<std::uint64_t, 4> rhs_ne = {1, k, 1, 1};
    constexpr std::array<std::uint64_t, 4> lhs_nb = {4, 4 * k, 4 * k, 4 * k};
    constexpr std::array<std::uint64_t, 4> rhs_nb = {4, 4, 4 * k, 4 * k};
    constexpr std::array<std::uint64_t, 4> dst_ne = {1, 1, 1, 1};
    constexpr std::array<std::uint64_t, 4> dst_nb = {4, 4, 4, 4};
    std::array<float, 1> output{};
    emel::kernel::Kernel kernel;
    emel::kernel::event::op_mul_mat request{};
    request.src0 = f32_view(lhs.data(), lhs_ne, lhs_nb);
    request.src1 = f32_view(rhs.data(), rhs_ne, rhs_nb);
    request.dst = f32_view_mut(output.data(), dst_ne, dst_nb);
    for (std::size_t i = 0; i < warmup; ++i) (void)kernel.process_event(request);
    const auto start = std::chrono::steady_clock::now();
    for (std::size_t i = 0; i < iterations; ++i) (void)kernel.process_event(request);
    const auto elapsed = std::chrono::duration<double, std::nano>(
        std::chrono::steady_clock::now() - start).count();
    std::cout << "case=op_mul_mat_f32 cpp_ns_per_dispatch="
              << (elapsed / static_cast<double>(iterations))
              << " output=" << output[0] << '\n';
    return 0;
  }

  std::cout << "kernel-matmul-parity/v1\n"
            << "source_repository=stateforward/emel.cpp\n"
            << "source_commit=" << k_source_commit << '\n'
            << "source_sml_commit=49207123cd3f39767764bae774932cb48623f92f\n"
            << "source_kernel_events_blob=4b3f02fa5ff9c5071d1fc83938cef1b22b4658b9\n"
            << "source_kernel_detail_blob=c8a82643eabfe8f2d7883e655955f455794511b0\n"
            << "source_kernel_x86_sm_blob=0b4d635ebbd0fbd52dbca8a2345547fb571205c8\n"
            << "source_kernel_x86_guards_blob=cb3dac8253f8417c9b44acff1de414f6d0a3a3cf\n"
            << "source_kernel_x86_actions_blob=d45558f5eb96950f43c16a09d768cb4f382d6d61\n"
            << "source_kernel_aarch64_sm_blob=865a9cc6ba6115382ed043c464f3d62bcd851357\n"
            << "source_kernel_aarch64_guards_blob=c25714566ec9a02679daef85089544575123408e\n"
            << "source_kernel_aarch64_actions_blob=267d4f74e6e7498155c8535920322ffef2c02fb6\n"
            << "scope=f32_public_root_matmul_and_argmax_dense_rank4\n";

  constexpr std::array<float, 6> lhs = {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F};
  constexpr std::array<float, 6> rhs = {1.0F, 0.0F, 0.0F, 1.0F, 1.0F, 1.0F};
  constexpr std::array<std::uint64_t, 4> lhs_ne = {3, 2, 1, 1};
  constexpr std::array<std::uint64_t, 4> rhs_ne = {2, 3, 1, 1};
  constexpr std::array<std::uint64_t, 4> dst_ne = {2, 2, 1, 1};
  constexpr std::array<std::uint64_t, 4> lhs_nb = {4, 12, 24, 24};
  constexpr std::array<std::uint64_t, 4> rhs_nb = {4, 8, 24, 24};
  constexpr std::array<std::uint64_t, 4> dst_nb = {4, 8, 16, 16};

  std::array<float, 4> output{};
  emel::kernel::Kernel kernel;
  emel::kernel::event::op_mul_mat request{};
  request.src0 = f32_view(lhs.data(), lhs_ne, lhs_nb);
  request.src1 = f32_view(rhs.data(), rhs_ne, rhs_nb);
  request.dst = f32_view_mut(output.data(), dst_ne, dst_nb);
  const bool status = kernel.process_event(request);

  print_matrix("matrix", status, output);

  std::array<float, 1> best_value{};
  std::int32_t best_index = -1;
  emel::kernel::event::op_mul_mat_argmax argmax{};
  constexpr std::array<std::uint64_t, 4> vector_ne = {1, 3, 1, 1};
  constexpr std::array<std::uint64_t, 4> vector_nb = {4, 4, 12, 12};
  constexpr std::array<std::uint64_t, 4> scalar_ne = {1, 1, 1, 1};
  constexpr std::array<std::uint64_t, 4> scalar_nb = {4, 4, 4, 4};
  argmax.src0 = f32_view(lhs.data(), lhs_ne, lhs_nb);
  argmax.src1 = f32_view(rhs.data(), vector_ne, vector_nb);
  argmax.dst = f32_view_mut(best_value.data(), scalar_ne, scalar_nb);
  argmax.index_out = &best_index;
  const bool argmax_status = kernel.process_event(argmax);
  std::cout << "case=argmax status=" << (argmax_status ? "ok" : "error")
            << " index=" << best_index << " output_bits=" << std::hex
            << std::setw(8) << std::setfill('0')
            << std::bit_cast<std::uint32_t>(best_value[0]) << std::dec << '\n';
}
