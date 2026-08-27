#include <array>
#include <bit>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>

#include "emel/kernel/aarch64/sm.hpp"

namespace {
constexpr char k_commit[] = "843a117386ef17dc5a50549bbfc821074c2141d6";
constexpr char k_events_blob[] = "4b3f02fa5ff9c5071d1fc83938cef1b22b4658b9";
constexpr char k_detail_blob[] = "c8a82643eabfe8f2d7883e655955f455794511b0";
constexpr char k_guards_blob[] = "c25714566ec9a02679daef85089544575123408e";
constexpr char k_actions_blob[] = "267d4f74e6e7498155c8535920322ffef2c02fb6";
constexpr char k_sm_blob[] = "865a9cc6ba6115382ed043c464f3d62bcd851357";
constexpr float k_sentinel = std::bit_cast<float>(std::uint32_t{0x7fc00001});

template <std::size_t N> void write_bits(const std::array<float, N> &values) {
  for (std::size_t index = 0; index < N; ++index) {
    if (index != 0) std::cout << ',';
    std::cout << std::hex << std::setw(8) << std::setfill('0')
              << std::bit_cast<std::uint32_t>(values[index]) << std::dec;
  }
}

emel::kernel::event::tensor_view view(const float *data, std::uint64_t count) {
  return {.data = data, .type = emel::kernel::event::dtype::f32,
          .ne = {count, 1, 1, 1}, .nb = {4, count * 4, count * 4, count * 4}};
}
emel::kernel::event::tensor_view_mut view_mut(float *data, std::uint64_t count) {
  return {.data = data, .type = emel::kernel::event::dtype::f32,
          .ne = {count, 1, 1, 1}, .nb = {4, count * 4, count * 4, count * 4}};
}

template <typename Request, std::size_t N>
void emit_positive(const char *name, emel::kernel::aarch64::sm &kernel,
                   Request request, std::array<float, N> &output) {
  if (!kernel.process_event(request)) std::exit(1);
  std::cout << "case=" << name << " status=ok output_bits=";
  write_bits(output);
  std::cout << '\n';
}
}  // namespace

int main() {
  const std::array<float, 9> unary_input = {-3.5F, -1.0F, 0.0F, 0.5F, 2.25F,
                                            -4.0F, 5.5F, -8.0F, 9.75F};
  const std::array<float, 9> binary_lhs = {8.0F, -9.0F, 6.0F, 4.0F, -2.0F,
                                           3.0F, 7.0F, 11.0F, 5.0F};
  const std::array<float, 9> binary_rhs = {2.0F, 3.0F, -2.0F, 4.0F, 2.0F,
                                           -3.0F, 7.0F, 11.0F, 5.0F};
  const std::array<float, 9> dup_input = {0.0F, -1.5F, 2.25F, 4.0F, -8.0F,
                                          16.0F, 32.0F, -64.0F, 128.0F};
  const std::array<float, 85> gemv_lhs = {
      0.25F, -1.5F, 2.0F, 3.25F, -4.5F, 5.0F, 6.75F, -7.0F, 8.5F, 9.0F, -10.25F, 11.5F, 12.0F, -13.75F, 14.25F, 15.0F, -16.5F,
      1.0F, 2.25F, -3.5F, 4.0F, 5.75F, -6.0F, 7.5F, 8.0F, -9.25F, 10.5F, 11.0F, -12.75F, 13.25F, 14.0F, -15.5F, 16.0F, 17.75F,
      -1.0F, 2.0F, 3.5F, -4.0F, 5.25F, 6.0F, -7.5F, 8.75F, 9.0F, -10.0F, 11.25F, 12.5F, -13.0F, 14.75F, 15.0F, -16.25F, 17.0F,
      2.0F, -2.5F, 3.0F, 4.25F, -5.0F, 6.5F, 7.0F, -8.75F, 9.5F, 10.0F, -11.5F, 12.0F, 13.25F, -14.0F, 15.5F, 16.0F, -17.75F,
      -0.5F, 1.5F, 2.75F, -3.0F, 4.5F, 5.25F, -6.0F, 7.75F, 8.0F, -9.5F, 10.25F, 11.0F, -12.5F, 13.0F, 14.25F, -15.0F, 16.5F};
  const std::array<float, 17> gemv_rhs = {0.5F, -1.0F, 1.5F, 2.0F, -2.5F, 3.0F, 3.5F, -4.0F, 4.5F, 5.0F, -5.5F, 6.0F, 6.5F, -7.0F, 7.5F, 8.0F, -8.5F};

  std::cout << "kernel-target-aarch64-router-live/v1\nsource_repository=stateforward/emel.cpp\nsource_commit=" << k_commit
            << "\nsource_kernel_events_blob=" << k_events_blob << "\nsource_kernel_detail_blob=" << k_detail_blob
            << "\nsource_kernel_aarch64_guards_blob=" << k_guards_blob << "\nsource_kernel_aarch64_actions_blob=" << k_actions_blob
            << "\nsource_kernel_aarch64_sm_blob=" << k_sm_blob << "\ntarget_arch=aarch64\n"
            << "scope=target_router_unary_abs_binary_add_dup_gemv_and_unexpected_rejection\n"
            << "execution=one_public_target_router_dispatching_four_live_events\n";

  emel::kernel::aarch64::sm kernel;
  std::array<float, 9> output{};
  emel::kernel::event::op_unary unary{};
  unary.src0 = view(unary_input.data(), unary_input.size());
  unary.dst = view_mut(output.data(), output.size());
  unary.subop = emel::kernel::event::unary_subop::abs;
  emit_positive("unary_abs", kernel, unary, output);

  emel::kernel::event::op_add add{};
  add.src0 = view(binary_lhs.data(), binary_lhs.size());
  add.src1 = view(binary_rhs.data(), binary_rhs.size());
  add.dst = view_mut(output.data(), output.size());
  emit_positive("binary_add", kernel, add, output);

  emel::kernel::event::op_dup dup{};
  dup.src0 = view(dup_input.data(), dup_input.size());
  dup.dst = view_mut(output.data(), output.size());
  emit_positive("dup", kernel, dup, output);

  constexpr std::uint64_t k = 17, m = 5;
  constexpr std::array<std::uint64_t, 4> lhs_ne = {k, m, 1, 1};
  constexpr std::array<std::uint64_t, 4> lhs_nb = {4, 4 * k, 4 * k * m, 4 * k * m};
  constexpr std::array<std::uint64_t, 4> rhs_ne = {1, k, 1, 1};
  constexpr std::array<std::uint64_t, 4> rhs_nb = {4, 4, 4 * k, 4 * k};
  constexpr std::array<std::uint64_t, 4> dst_ne = {1, m, 1, 1};
  constexpr std::array<std::uint64_t, 4> dst_nb = {4, 4, 4 * m, 4 * m};
  std::array<float, 5> gemv_output{};
  emel::kernel::event::op_mul_mat gemv{};
  gemv.src0 = {gemv_lhs.data(), emel::kernel::event::dtype::f32, lhs_ne, lhs_nb};
  gemv.src1 = {gemv_rhs.data(), emel::kernel::event::dtype::f32, rhs_ne, rhs_nb};
  gemv.dst = {gemv_output.data(), emel::kernel::event::dtype::f32, dst_ne, dst_nb};
  emit_positive("gemv_m5_k17", kernel, gemv, gemv_output);

  std::array<float, 9> rejected;
  rejected.fill(k_sentinel);
  std::cout << "case=unexpected status=reject error=UnexpectedEvent output_bits=";
  write_bits(rejected);
  std::cout << '\n';
}
