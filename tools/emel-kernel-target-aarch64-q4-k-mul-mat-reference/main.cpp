#include <array>
#include <bit>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include "emel/kernel/aarch64/sm.hpp"
#include "emel/kernel/detail.hpp"

namespace {
constexpr char k_commit[] = "843a117386ef17dc5a50549bbfc821074c2141d6";
constexpr char k_guards_blob[] = "c25714566ec9a02679daef85089544575123408e";
constexpr char k_actions_blob[] = "267d4f74e6e7498155c8535920322ffef2c02fb6";
constexpr char k_sm_blob[] = "865a9cc6ba6115382ed043c464f3d62bcd851357";
constexpr float k_sentinel = std::bit_cast<float>(std::uint32_t{0x7fc00001});
constexpr std::uint64_t m = 2;
constexpr std::uint64_t k = 256;
constexpr std::uint64_t n = 2;
constexpr std::size_t q4_bytes = 144;

template <std::size_t N>
void write_bits(const std::array<float, N> &values) {
  for (std::size_t index = 0; index < N; ++index) {
    if (index != 0) std::cout << ',';
    std::cout << std::hex << std::setw(8) << std::setfill('0')
              << std::bit_cast<std::uint32_t>(values[index]) << std::dec;
  }
}

void pack_q4(::emel::kernel::detail::quant::block_q4_k &block, std::uint16_t d,
             std::uint16_t dmin, std::uint8_t scale, std::uint8_t qs) {
  block.d = d;
  block.dmin = dmin;
  block.scales.fill(scale);
  block.qs.fill(qs);
}

emel::kernel::event::tensor_view view(const void *data, emel::kernel::event::dtype type,
                                      std::array<std::uint64_t, 4> ne,
                                      std::array<std::uint64_t, 4> nb) {
  return {.data = data, .type = type, .ne = ne, .nb = nb};
}
emel::kernel::event::tensor_view_mut view_mut(float *data, std::array<std::uint64_t, 4> ne,
                                              std::array<std::uint64_t, 4> nb) {
  return {.data = data, .type = emel::kernel::event::dtype::f32, .ne = ne, .nb = nb};
}
}  // namespace

int main() {
  std::array<::emel::kernel::detail::quant::block_q4_k, m> lhs_blocks{};
  pack_q4(lhs_blocks[0], 0x3c00, 0x3400, 2, 0x13);
  pack_q4(lhs_blocks[1], 0x4000, 0x3800, 3, 0x24);
  std::array<float, k * n> rhs{};
  for (std::uint64_t index = 0; index < k; ++index) {
    rhs[index * n] = (static_cast<float>(index) - 128.0f) * 0.03125f;
    rhs[index * n + 1] = (64.0f - static_cast<float>(index)) * 0.015625f;
  }
  constexpr std::array<std::uint64_t, 4> lhs_ne = {k, m, 1, 1};
  constexpr std::array<std::uint64_t, 4> lhs_nb = {1, q4_bytes, q4_bytes * m, q4_bytes * m};
  constexpr std::array<std::uint64_t, 4> rhs_ne = {n, k, 1, 1};
  constexpr std::array<std::uint64_t, 4> rhs_nb = {4, 4 * n, 4 * n * k, 4 * n * k};
  constexpr std::array<std::uint64_t, 4> dst_ne = {n, m, 1, 1};
  constexpr std::array<std::uint64_t, 4> dst_nb = {4, 4 * n, 4 * n * m, 4 * n * m};

  std::cout << "kernel-target-aarch64-q4-k-mul-mat-live/v1\nsource_repository=stateforward/emel.cpp\nsource_commit="
            << k_commit << "\nsource_kernel_aarch64_guards_blob=" << k_guards_blob
            << "\nsource_kernel_aarch64_actions_blob=" << k_actions_blob
            << "\nsource_kernel_aarch64_sm_blob=" << k_sm_blob
            << "\ntarget_arch=aarch64\nscope=target_router_q4_k_mul_mat_m2_k256_n2_and_typed_rejection\nexecution=split_pinned_aarch64_sm_and_public_target_router\n";

  emel::kernel::aarch64::sm kernel;
  std::array<float, m * n> output{};
  emel::kernel::event::op_mul_mat request{};
  request.src0 = view(lhs_blocks.data(), emel::kernel::event::dtype::q4_k, lhs_ne, lhs_nb);
  request.src1 = view(rhs.data(), emel::kernel::event::dtype::f32, rhs_ne, rhs_nb);
  request.dst = view_mut(output.data(), dst_ne, dst_nb);
  if (!kernel.process_event(request)) return 1;
  std::cout << "case=q4_k_mul_mat_m2_k256_n2 status=ok output_bits=";
  write_bits(output);
  std::cout << '\n';

  std::array<float, 4> zero = {k_sentinel, k_sentinel, k_sentinel, k_sentinel};
  emel::kernel::event::op_mul_mat invalid{};
  invalid.src0 = view(zero.data(), emel::kernel::event::dtype::q4_k, {0, 0, 1, 1}, {1, 0, 0, 0});
  invalid.src1 = view(zero.data(), emel::kernel::event::dtype::f32, {0, 0, 1, 1}, {4, 0, 0, 0});
  invalid.dst = view_mut(zero.data(), {0, 0, 1, 1}, {4, 0, 0, 0});
  if (kernel.process_event(invalid)) return 1;
  std::cout << "case=zero_count status=reject error=InvalidShape output_bits=";
  write_bits(zero);
  std::cout << '\n';

  std::array<float, 4> rejected = {k_sentinel, k_sentinel, k_sentinel, k_sentinel};
  invalid.src0 = view(lhs_blocks.data(), emel::kernel::event::dtype::q4_k, {0, m, 1, 1},
                      {1, q4_bytes, q4_bytes * m, q4_bytes * m});
  invalid.src1 = view(rhs.data(), emel::kernel::event::dtype::f32, {n, 0, 1, 1}, {4, 4 * n, 0, 0});
  invalid.dst = view_mut(rejected.data(), dst_ne, dst_nb);
  if (kernel.process_event(invalid)) return 1;
  std::cout << "case=invalid_shape status=reject error=InvalidShape output_bits=";
  write_bits(rejected);
  std::cout << '\n';
}
