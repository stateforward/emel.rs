#include <bit>
#include <cstdint>
#include <iomanip>
#include <iostream>

#include "emel/kernel/detail.hpp"

namespace {

constexpr char k_source_commit[] = "843a117386ef17dc5a50549bbfc821074c2141d6";

emel::kernel::detail::quant::block_q5_k q5_k_block() {
  emel::kernel::detail::quant::block_q5_k block{};
  block.d = 0x3c00u;
  block.dmin = 0x3800u;
  block.scales.fill(1u);
  for (std::size_t index = 4; index < 8; ++index) block.scales[index] = 2u;
  for (std::size_t lane = 0; lane < block.qh.size(); ++lane) {
    block.qh[lane] = static_cast<std::uint8_t>(lane * 37u + 0x5au);
  }
  for (std::size_t chunk = 0; chunk < 4; ++chunk) {
    for (std::size_t lane = 0; lane < 32; ++lane) {
      const auto low = static_cast<std::uint8_t>((lane + chunk * 3u) % 16u);
      const auto high = static_cast<std::uint8_t>((15u + chunk + 16u - lane) % 16u);
      block.qs[chunk * 32u + lane] = low | static_cast<std::uint8_t>(high << 4u);
    }
  }
  return block;
}

emel::kernel::detail::quant::block_q8_k q8_k_block() {
  emel::kernel::detail::quant::block_q8_k block{};
  block.d = 1.0f;
  for (std::size_t lane = 0; lane < block.qs.size(); ++lane) {
    block.qs[lane] = static_cast<std::int8_t>(static_cast<int>(lane % 23u) - 11);
  }
  for (std::size_t group = 0; group < block.bsums.size(); ++group) {
    for (std::size_t lane = 0; lane < 16; ++lane) {
      block.bsums[group] += block.qs[group * 16 + lane];
    }
  }
  return block;
}

void write_result(const char *name, const float value) {
  std::cout << "case=" << name << " status=ok output_bits=" << std::hex
            << std::setw(8) << std::setfill('0') << std::bit_cast<std::uint32_t>(value)
            << std::dec << '\n';
}

}  // namespace

int main() {
  using emel::kernel::detail::dot_q5_k_q8_k_row_scalar;
  using emel::kernel::detail::quant::block_q5_k;
  using emel::kernel::detail::quant::block_q8_k;

  std::cout << "kernel-quant-more-parity/v1\n"
            << "source_repository=stateforward/emel.cpp\n"
            << "source_commit=" << k_source_commit << '\n'
            << "source_layout=q5_k_176_bytes,q8_k_292_bytes,qk_k_256_values\n"
            << "source_scalar_dot_span=2944-3000\n"
            << "scope=borrowed_scalar_q5_k_q8_k_rows\n";

  const block_q5_k lhs = q5_k_block();
  const block_q8_k rhs = q8_k_block();
  write_result("dot_q5_k_q8_k_varied_qs", dot_q5_k_q8_k_row_scalar(&lhs, &rhs, 1));
  std::cout << "case=invalid_block_length status=error error=InvalidBlockLength { format: Q5K, bytes: 175 }\n"
            << "case=mismatched_block_count status=error error=MismatchedBlockCount { lhs: 1, rhs: 2 }\n";
}
