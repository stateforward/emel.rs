#include <array>
#include <bit>
#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <string>

#include "emel/kernel/sm.hpp"

namespace {

constexpr char k_source_commit[] =
    "843a117386ef17dc5a50549bbfc821074c2141d6";

emel::kernel::event::tensor_view f16_view(
    const std::uint16_t *data, const std::array<std::uint64_t, 4> ne,
    const std::array<std::uint64_t, 4> nb) {
  return {.data = data,
          .type = emel::kernel::event::dtype::f16,
          .ne = ne,
          .nb = nb};
}

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

template <std::size_t K, std::size_t M, std::size_t N>
void run_case(const char *name, const std::array<std::uint16_t, K * M> &lhs,
              const std::array<std::uint16_t, K * N> &rhs) {
  constexpr std::array<std::uint64_t, 4> lhs_ne = {K, M, 1, 1};
  constexpr std::array<std::uint64_t, 4> rhs_ne = {K, N, 1, 1};
  constexpr std::array<std::uint64_t, 4> dst_ne = {M, N, 1, 1};
  constexpr std::array<std::uint64_t, 4> lhs_nb = {2, 2 * K, 2 * K * M,
                                                    2 * K * M};
  constexpr std::array<std::uint64_t, 4> rhs_nb = {2, 2 * K, 2 * K * N,
                                                    2 * K * N};
  constexpr std::array<std::uint64_t, 4> dst_nb = {4, 4 * M, 4 * M * N,
                                                    4 * M * N};
  std::array<float, M * N> output{};
  emel::kernel::Kernel kernel;
  emel::kernel::event::op_mul_mat request{};
  request.src0 = f16_view(lhs.data(), lhs_ne, lhs_nb);
  request.src1 = f16_view(rhs.data(), rhs_ne, rhs_nb);
  request.dst = f32_view_mut(output.data(), dst_ne, dst_nb);
  print_case(name, kernel.process_event(request), output);
}

}  // namespace

int main(int argc, char **argv) {
  if (argc > 1 && std::string(argv[1]) == "--benchmark") {
    constexpr std::size_t k = 16;
    constexpr std::size_t iterations = 10000;
    constexpr std::size_t warmup = 1000;
    std::array<std::uint16_t, k> lhs{};
    std::array<std::uint16_t, k> rhs{};
    lhs.fill(0x3c00);
    rhs.fill(0x3c00);
    constexpr std::array<std::uint64_t, 4> ne = {k, 1, 1, 1};
    constexpr std::array<std::uint64_t, 4> nb = {2, 2 * k, 2 * k, 2 * k};
    constexpr std::array<std::uint64_t, 4> dst_ne = {1, 1, 1, 1};
    constexpr std::array<std::uint64_t, 4> dst_nb = {4, 4, 4, 4};
    std::array<float, 1> output{};
    emel::kernel::Kernel kernel;
    emel::kernel::event::op_mul_mat request{};
    request.src0 = f16_view(lhs.data(), ne, nb);
    request.src1 = f16_view(rhs.data(), ne, nb);
    request.dst = f32_view_mut(output.data(), dst_ne, dst_nb);
    for (std::size_t i = 0; i < warmup; ++i) (void)kernel.process_event(request);
    const auto start = std::chrono::steady_clock::now();
    for (std::size_t i = 0; i < iterations; ++i) (void)kernel.process_event(request);
    const auto elapsed = std::chrono::duration<double, std::nano>(
        std::chrono::steady_clock::now() - start).count();
    std::cout << "case=op_mul_mat_f16 cpp_ns_per_dispatch="
              << (elapsed / static_cast<double>(iterations))
              << " output=" << output[0] << '\n';
    return 0;
  }

  const std::array<std::uint16_t, 6> lhs = {
      0x3c00, 0x4000, 0x4200, 0x4400, 0x4500, 0x4600};
  const std::array<std::uint16_t, 6> rhs = {
      0x3c00, 0x0000, 0x0000, 0x3c00, 0x3c00, 0x3c00};
  const std::array<std::uint16_t, 16> acc_lhs = {
      22892, 382, 28000, 56722, 59028, 37094, 30984, 61306,
      6332, 3406, 63920, 15970, 36836, 5814, 36696, 27210};
  const std::array<std::uint16_t, 16> acc_rhs = {
      34267, 15045, 49471, 24521, 41187, 62221, 18631, 47249,
      7403, 46165, 49487, 10841, 6643, 40605, 19159, 54561};
  std::cout << "kernel-f16-matmul-parity/v1\n"
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
            << "scope=dense_explicit_nonzero_stride_f16_by_f16_to_f32_scalar_double_accumulation\n";
  run_case<3, 2, 2>("matrix", lhs, rhs);
  run_case<16, 1, 1>("double_accumulation", acc_lhs, acc_rhs);
  return 0;
}
