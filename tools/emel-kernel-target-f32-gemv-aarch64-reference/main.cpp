#include <array>
#include <bit>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>

#include "emel/kernel/aarch64/sm.hpp"

namespace {

constexpr char k_commit[] = "843a117386ef17dc5a50549bbfc821074c2141d6";
constexpr char k_guards_blob[] = "c25714566ec9a02679daef85089544575123408e";
constexpr char k_actions_blob[] = "267d4f74e6e7498155c8535920322ffef2c02fb6";
constexpr char k_sm_blob[] = "865a9cc6ba6115382ed043c464f3d62bcd851357";
constexpr float k_sentinel = std::bit_cast<float>(std::uint32_t{0x7fc00001});

template <std::size_t N>
void write_bits(const std::array<float, N> &values) {
  for (std::size_t index = 0; index < N; ++index) {
    if (index != 0) std::cout << ',';
    std::cout << std::hex << std::setw(8) << std::setfill('0')
              << std::bit_cast<std::uint32_t>(values[index]) << std::dec;
  }
}

emel::kernel::event::tensor_view view(const float *data,
                                      std::array<std::uint64_t, 4> ne,
                                      std::array<std::uint64_t, 4> nb) {
  return {.data = data, .type = emel::kernel::event::dtype::f32, .ne = ne, .nb = nb};
}

emel::kernel::event::tensor_view_mut view_mut(float *data,
                                              std::array<std::uint64_t, 4> ne,
                                              std::array<std::uint64_t, 4> nb) {
  return {.data = data, .type = emel::kernel::event::dtype::f32, .ne = ne, .nb = nb};
}

}  // namespace

int main() {
  constexpr std::uint64_t k = 17;
  constexpr std::uint64_t m = 5;
  constexpr std::array<std::uint64_t, 4> lhs_ne = {k, m, 1, 1};
  constexpr std::array<std::uint64_t, 4> lhs_nb = {4, 4 * k, 4 * k * m, 4 * k * m};
  constexpr std::array<std::uint64_t, 4> rhs_ne = {1, k, 1, 1};
  constexpr std::array<std::uint64_t, 4> rhs_nb = {4, 4, 4 * k, 4 * k};
  constexpr std::array<std::uint64_t, 4> dst_ne = {1, m, 1, 1};
  constexpr std::array<std::uint64_t, 4> dst_nb = {4, 4, 4 * m, 4 * m};
  const std::array<float, 85> lhs = {
      0.25F, -1.5F, 2.0F, 3.25F, -4.5F, 5.0F, 6.75F, -7.0F, 8.5F, 9.0F, -10.25F, 11.5F, 12.0F, -13.75F, 14.25F, 15.0F, -16.5F,
      1.0F, 2.25F, -3.5F, 4.0F, 5.75F, -6.0F, 7.5F, 8.0F, -9.25F, 10.5F, 11.0F, -12.75F, 13.25F, 14.0F, -15.5F, 16.0F, 17.75F,
      -1.0F, 2.0F, 3.5F, -4.0F, 5.25F, 6.0F, -7.5F, 8.75F, 9.0F, -10.0F, 11.25F, 12.5F, -13.0F, 14.75F, 15.0F, -16.25F, 17.0F,
      2.0F, -2.5F, 3.0F, 4.25F, -5.0F, 6.5F, 7.0F, -8.75F, 9.5F, 10.0F, -11.5F, 12.0F, 13.25F, -14.0F, 15.5F, 16.0F, -17.75F,
      -0.5F, 1.5F, 2.75F, -3.0F, 4.5F, 5.25F, -6.0F, 7.75F, 8.0F, -9.5F, 10.25F, 11.0F, -12.5F, 13.0F, 14.25F, -15.0F, 16.5F};
  const std::array<float, 17> rhs = {0.5F, -1.0F, 1.5F, 2.0F, -2.5F, 3.0F, 3.5F, -4.0F, 4.5F, 5.0F, -5.5F, 6.0F, 6.5F, -7.0F, 7.5F, 8.0F, -8.5F};
  std::cout << "kernel-target-f32-gemv-aarch64-live/v1\nsource_repository=stateforward/emel.cpp\nsource_commit=" << k_commit
            << "\nsource_kernel_aarch64_guards_blob=" << k_guards_blob
            << "\nsource_kernel_aarch64_actions_blob=" << k_actions_blob
            << "\nsource_kernel_aarch64_sm_blob=" << k_sm_blob
            << "\ntarget_arch=aarch64\nscope=target_router_f32_gemv_dense_contiguous_body_and_tail_rejection\nexecution=split_pinned_aarch64_sm_and_public_target_router\n";
  emel::kernel::aarch64::sm kernel;
  std::array<float, 5> output{};
  emel::kernel::event::op_mul_mat request{};
  request.src0 = view(lhs.data(), lhs_ne, lhs_nb);
  request.src1 = view(rhs.data(), rhs_ne, rhs_nb);
  request.dst = view_mut(output.data(), dst_ne, dst_nb);
  if (!kernel.process_event(request)) return 1;
  std::cout << "case=gemv_m5_k17 status=ok output_bits="; write_bits(output); std::cout << '\n';

  std::array<float, 1> zero = {k_sentinel};
  emel::kernel::event::op_mul_mat invalid{};
  invalid.src0 = view(zero.data(), {0, 0, 1, 1}, {4, 0, 0, 0});
  invalid.src1 = view(zero.data(), {1, 0, 1, 1}, {4, 4, 0, 0});
  invalid.dst = view_mut(zero.data(), {1, 0, 1, 1}, {4, 4, 0, 0});
  if (kernel.process_event(invalid)) return 1;
  std::cout << "case=zero_count status=reject error=InvalidShape output_bits="; write_bits(zero); std::cout << '\n';

  std::array<float, 5> rejected; rejected.fill(k_sentinel);
  invalid.src0 = view(lhs.data(), {k - 1, m, 1, 1}, {4, 4 * (k - 1), 4 * (k - 1) * m, 4 * (k - 1) * m});
  invalid.src1 = view(rhs.data(), rhs_ne, rhs_nb);
  invalid.dst = view_mut(rejected.data(), dst_ne, dst_nb);
  if (kernel.process_event(invalid)) return 1;
  std::cout << "case=invalid_shape status=reject error=InvalidShape output_bits="; write_bits(rejected); std::cout << '\n';
}
