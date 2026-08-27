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
constexpr std::uint64_t k = 32;
constexpr std::uint64_t m = 5;
constexpr std::size_t q8_block_bytes = 34;
constexpr std::size_t x4_block_bytes =
    sizeof(::emel::kernel::detail::quant::block_q8_0x4);
constexpr std::uint64_t group_count =
    ::emel::kernel::detail::quant::packed_q8_0_x4_group_count(m);
constexpr std::size_t group_bytes = x4_block_bytes;

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
  constexpr std::array<std::uint64_t, 4> rhs_nb = {1, q8_block_bytes, q8_block_bytes,
                                                   q8_block_bytes};
  constexpr std::array<std::uint64_t, 4> dst_ne = {1, m, 1, 1};
  constexpr std::array<std::uint64_t, 4> dst_nb = {4, 4, 4 * m, 4 * m};

  std::array<::emel::kernel::detail::quant::block_q8_0, m> src_rows{};
  for (std::uint64_t row = 0; row < m; ++row) {
    src_rows[row].d = 0x3c00;
    for (std::size_t index = 0; index < 32; ++index) {
      src_rows[row].qs[index] =
          static_cast<std::int8_t>(static_cast<int>(index) - 16 + static_cast<int>(row));
    }
  }
  std::array<std::uint8_t, group_count * x4_block_bytes> lhs{};
  if (!::emel::kernel::detail::quant::pack_q8_0_rows_x4_bl8(src_rows.data(), m, k,
                                                            lhs.data())) {
    return 1;
  }

  std::array<std::uint8_t, q8_block_bytes> rhs{};
  rhs[0] = 0x00;
  rhs[1] = 0x38;
  for (std::size_t index = 0; index < 32; ++index) {
    rhs[2 + index] = static_cast<std::uint8_t>(
        static_cast<std::int8_t>(static_cast<int>(index) - 16));
  }

  std::cout << "kernel-target-aarch64-q8-0-x4-bl8-live/v1\nsource_repository=stateforward/emel.cpp\nsource_commit="
            << k_commit << "\nsource_kernel_aarch64_guards_blob=" << k_guards_blob
            << "\nsource_kernel_aarch64_actions_blob=" << k_actions_blob
            << "\nsource_kernel_aarch64_sm_blob=" << k_sm_blob
            << "\ntarget_arch=aarch64\nscope=target_router_q8_0_packed_bl8_m5_k32_and_typed_rejection\nexecution=split_pinned_aarch64_sm_and_public_target_router\n";

  emel::kernel::aarch64::sm kernel;
  std::array<float, m> output{};
  emel::kernel::event::op_mul_mat request{};
  request.src0 = view(lhs.data(), emel::kernel::event::dtype::q8_0_x4_bl8, lhs_ne, lhs_nb);
  request.src1 = view(rhs.data(), emel::kernel::event::dtype::q8_0, rhs_ne, rhs_nb);
  request.dst = view_mut(output.data(), dst_ne, dst_nb);
  if (!kernel.process_event(request)) return 1;
  std::cout << "case=q8_0_packed_bl8_m5_k32 status=ok output_bits=";
  write_bits(output);
  std::cout << '\n';

  std::array<float, 1> zero = {k_sentinel};
  emel::kernel::event::op_mul_mat invalid{};
  invalid.src0 =
      view(zero.data(), emel::kernel::event::dtype::q8_0_x4_bl8, {0, 0, 1, 1}, {1, 0, 0, 0});
  invalid.src1 = view(zero.data(), emel::kernel::event::dtype::q8_0, {1, 0, 1, 1}, {1, 0, 0, 0});
  invalid.dst = view_mut(zero.data(), {1, 0, 1, 1}, {4, 4, 0, 0});
  if (kernel.process_event(invalid)) return 1;
  std::cout << "case=zero_count status=reject error=InvalidShape output_bits=";
  write_bits(zero);
  std::cout << '\n';

  std::array<float, m> rejected;
  rejected.fill(k_sentinel);
  invalid.src0 = view(lhs.data(), emel::kernel::event::dtype::q8_0_x4_bl8, {31, m, 1, 1},
                      {1, group_bytes, group_bytes * group_count, group_bytes * group_count});
  invalid.src1 = view(rhs.data(), emel::kernel::event::dtype::q8_0, {1, 31, 1, 1},
                      {1, q8_block_bytes, q8_block_bytes, q8_block_bytes});
  invalid.dst = view_mut(rejected.data(), dst_ne, dst_nb);
  if (kernel.process_event(invalid)) return 1;
  std::cout << "case=invalid_shape status=reject error=InvalidShape output_bits=";
  write_bits(rejected);
  std::cout << '\n';
}
