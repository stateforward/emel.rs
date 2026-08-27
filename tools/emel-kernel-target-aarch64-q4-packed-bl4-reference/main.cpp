#include <array>
#include <bit>
#include <cstdint>
#include <cstdlib>
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
constexpr std::uint64_t k = 256;
constexpr std::uint64_t m = 5;
constexpr std::size_t q4_x8_block_bytes =
    sizeof(::emel::kernel::detail::quant::block_q4_kx8);
constexpr std::size_t q8_k_block_bytes =
    sizeof(::emel::kernel::detail::quant::block_q8_k);
constexpr std::uint64_t group_count =
    ::emel::kernel::detail::quant::packed_q4_k_x8_group_count(m);
constexpr std::size_t group_bytes = q4_x8_block_bytes;

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
  constexpr std::array<std::uint64_t, 4> lhs_nb = {
      1, group_bytes, group_bytes * group_count, group_bytes * group_count};
  constexpr std::array<std::uint64_t, 4> rhs_ne = {1, k, 1, 1};
  constexpr std::array<std::uint64_t, 4> rhs_nb = {
      1, q8_k_block_bytes, q8_k_block_bytes, q8_k_block_bytes};
  constexpr std::array<std::uint64_t, 4> dst_ne = {1, m, 1, 1};
  constexpr std::array<std::uint64_t, 4> dst_nb = {4, 4, 4 * m, 4 * m};

  std::array<::emel::kernel::detail::quant::block_q4_k, m> src_rows{};
  for (std::uint64_t row = 0; row < m; ++row) {
    src_rows[row].d = static_cast<std::uint16_t>(0x3c00u + row * 17u);
    src_rows[row].dmin = static_cast<std::uint16_t>(0x3400u + row * 11u);
    for (std::size_t index = 0; index < src_rows[row].scales.size(); ++index) {
      src_rows[row].scales[index] = static_cast<std::uint8_t>(
          (row * 37u + index * 13u + 9u) & 0xffu);
    }
    for (std::size_t index = 0; index < src_rows[row].qs.size(); ++index) {
      src_rows[row].qs[index] = static_cast<std::uint8_t>(
          (row * 29u + index * 5u + 3u) & 0xffu);
    }
  }
  std::array<std::uint8_t, group_count * q4_x8_block_bytes> lhs{};
  if (!::emel::kernel::detail::quant::pack_q4_k_rows_x8_bl4(src_rows.data(), m, k,
                                                           lhs.data())) {
    return 1;
  }

  ::emel::kernel::detail::quant::block_q8_k rhs{};
  rhs.d = 0.0625f;
  for (std::size_t index = 0; index < rhs.qs.size(); ++index) {
    rhs.qs[index] = static_cast<std::int8_t>(
        static_cast<int>((static_cast<int>(index) * 7) % 31) - 15);
  }
  for (std::size_t group = 0; group < rhs.bsums.size(); ++group) {
    int sum = 0;
    for (std::size_t lane = 0; lane < 16; ++lane) {
      sum += static_cast<int>(rhs.qs[group * 16 + lane]);
    }
    rhs.bsums[group] = static_cast<std::int16_t>(sum);
  }

  std::cout << "kernel-target-aarch64-q4-packed-bl4-live/v1\nsource_repository=stateforward/emel.cpp\nsource_commit="
            << k_commit << "\nsource_kernel_aarch64_guards_blob=" << k_guards_blob
            << "\nsource_kernel_aarch64_actions_blob=" << k_actions_blob
            << "\nsource_kernel_aarch64_sm_blob=" << k_sm_blob
            << "\ntarget_arch=aarch64\nscope=target_router_q4_packed_bl4_m5_k256_and_typed_rejection\nexecution=split_pinned_aarch64_sm_and_public_target_router\n";

  emel::kernel::aarch64::sm kernel;
  std::array<float, m> output{};
  emel::kernel::event::op_mul_mat request{};
  request.src0 =
      view(lhs.data(), emel::kernel::event::dtype::q4_k_x8_bl4, lhs_ne, lhs_nb);
  request.src1 = view(&rhs, emel::kernel::event::dtype::q8_k, rhs_ne, rhs_nb);
  request.dst = view_mut(output.data(), dst_ne, dst_nb);
  if (!kernel.process_event(request)) return 1;
  std::cout << "case=q4_packed_bl4_m5_k256 status=ok output_bits=";
  write_bits(output);
  std::cout << '\n';

  std::array<float, 1> zero = {k_sentinel};
  emel::kernel::event::op_mul_mat invalid{};
  invalid.src0 = view(zero.data(), emel::kernel::event::dtype::q4_k_x8_bl4,
                      {0, 0, 1, 1}, {1, 0, 0, 0});
  invalid.src1 =
      view(zero.data(), emel::kernel::event::dtype::q8_k, {1, 0, 1, 1}, {1, 0, 0, 0});
  invalid.dst = view_mut(zero.data(), {1, 0, 1, 1}, {4, 4, 0, 0});
  if (kernel.process_event(invalid)) return 1;
  std::cout << "case=zero_count status=reject error=InvalidShape output_bits=";
  write_bits(zero);
  std::cout << '\n';

  std::array<float, m> rejected;
  rejected.fill(k_sentinel);
  invalid.src0 = view(lhs.data(), emel::kernel::event::dtype::q4_k_x8_bl4,
                      {255, m, 1, 1},
                      {1, group_bytes, group_bytes * group_count,
                       group_bytes * group_count});
  invalid.src1 = view(&rhs, emel::kernel::event::dtype::q8_k, {1, 255, 1, 1},
                      {1, q8_k_block_bytes, q8_k_block_bytes, q8_k_block_bytes});
  invalid.dst = view_mut(rejected.data(), dst_ne, dst_nb);
  if (kernel.process_event(invalid)) return 1;
  std::cout << "case=invalid_shape status=reject error=InvalidShape output_bits=";
  write_bits(rejected);
  std::cout << '\n';
}
