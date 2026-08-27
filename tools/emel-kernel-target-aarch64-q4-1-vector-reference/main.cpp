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
constexpr std::uint64_t k = 32;
constexpr std::uint64_t m = 5;
constexpr std::size_t q4_block_bytes = 20;

template <std::size_t N>
void write_bits(const std::array<float, N> &values) {
  for (std::size_t index = 0; index < N; ++index) {
    if (index != 0) std::cout << ',';
    std::cout << std::hex << std::setw(8) << std::setfill('0')
              << std::bit_cast<std::uint32_t>(values[index]) << std::dec;
  }
}

emel::kernel::event::tensor_view view(const void *data,
                                      emel::kernel::event::dtype type,
                                      std::array<std::uint64_t, 4> ne,
                                      std::array<std::uint64_t, 4> nb) {
  return {.data = data, .type = type, .ne = ne, .nb = nb};
}

emel::kernel::event::tensor_view_mut view_mut(float *data,
                                              std::array<std::uint64_t, 4> ne,
                                              std::array<std::uint64_t, 4> nb) {
  return {.data = data, .type = emel::kernel::event::dtype::f32, .ne = ne, .nb = nb};
}

}  // namespace

int main() {
  constexpr std::array<std::uint64_t, 4> lhs_ne = {k, m, 1, 1};
  constexpr std::array<std::uint64_t, 4> lhs_nb = {1, q4_block_bytes, q4_block_bytes * m,
                                                   q4_block_bytes * m};
  constexpr std::array<std::uint64_t, 4> rhs_ne = {1, k, 1, 1};
  constexpr std::array<std::uint64_t, 4> rhs_nb = {4, 4, 4 * k, 4 * k};
  constexpr std::array<std::uint64_t, 4> dst_ne = {1, m, 1, 1};
  constexpr std::array<std::uint64_t, 4> dst_nb = {4, 4, 4 * m, 4 * m};

  std::array<std::uint8_t, m * q4_block_bytes> lhs{};
  for (std::uint64_t row = 0; row < m; ++row) {
    const std::size_t offset = static_cast<std::size_t>(row) * q4_block_bytes;
    lhs[offset] = 0x00;
    lhs[offset + 1] = 0x3c;
    lhs[offset + 2] = 0x00;
    lhs[offset + 3] = 0x38;
    for (std::size_t index = 0; index < 16; ++index) {
      const std::uint8_t low =
          static_cast<std::uint8_t>((row + index) % 16u);
      const std::uint8_t high =
          static_cast<std::uint8_t>((row * 3u + index) % 16u);
      lhs[offset + 4 + index] = static_cast<std::uint8_t>(low | (high << 4));
    }
  }
  std::array<float, k> rhs{};
  for (std::uint64_t index = 0; index < k; ++index) {
    rhs[index] = static_cast<float>(static_cast<int>(index) - 16) * 0.25F;
  }

  std::cout << "kernel-target-aarch64-q4-1-vector-live/v1\nsource_repository=stateforward/emel.cpp\nsource_commit="
            << k_commit << "\nsource_kernel_aarch64_guards_blob=" << k_guards_blob
            << "\nsource_kernel_aarch64_actions_blob=" << k_actions_blob
            << "\nsource_kernel_aarch64_sm_blob=" << k_sm_blob
            << "\ntarget_arch=aarch64\nscope=target_router_q4_1_vector_m5_k32_and_typed_rejection\nexecution=split_pinned_aarch64_sm_and_public_target_router\n";

  emel::kernel::aarch64::sm kernel;
  std::array<float, m> output{};
  emel::kernel::event::op_mul_mat request{};
  request.src0 = view(lhs.data(), emel::kernel::event::dtype::q4_1, lhs_ne, lhs_nb);
  request.src1 = view(rhs.data(), emel::kernel::event::dtype::f32, rhs_ne, rhs_nb);
  request.dst = view_mut(output.data(), dst_ne, dst_nb);
  if (!kernel.process_event(request)) return 1;
  std::cout << "case=q4_1_vector_m5_k32 status=ok output_bits=";
  write_bits(output);
  std::cout << '\n';

  std::array<float, 1> zero = {k_sentinel};
  emel::kernel::event::op_mul_mat invalid{};
  invalid.src0 = view(zero.data(), emel::kernel::event::dtype::q4_1, {0, 0, 1, 1}, {1, 0, 0, 0});
  invalid.src1 = view(zero.data(), emel::kernel::event::dtype::f32, {1, 0, 1, 1}, {4, 4, 0, 0});
  invalid.dst = view_mut(zero.data(), {1, 0, 1, 1}, {4, 4, 0, 0});
  if (kernel.process_event(invalid)) return 1;
  std::cout << "case=zero_count status=reject error=InvalidShape output_bits=";
  write_bits(zero);
  std::cout << '\n';

  std::array<float, m> rejected;
  rejected.fill(k_sentinel);
  invalid.src0 = view(lhs.data(), emel::kernel::event::dtype::q4_1, {31, m, 1, 1},
                      {1, q4_block_bytes, q4_block_bytes * m, q4_block_bytes * m});
  invalid.src1 = view(rhs.data(), emel::kernel::event::dtype::f32, {1, 31, 1, 1},
                      {4, 4, 4 * 31, 4 * 31});
  invalid.dst = view_mut(rejected.data(), dst_ne, dst_nb);
  if (kernel.process_event(invalid)) return 1;
  std::cout << "case=invalid_shape status=reject error=InvalidShape output_bits=";
  write_bits(rejected);
  std::cout << '\n';
}
