#include <bit>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>

#include "emel/kernel/detail.hpp"

namespace {

constexpr char k_source_commit[] =
    "843a117386ef17dc5a50549bbfc821074c2141d6";

emel::kernel::detail::quant::block_q4_0 q4_block(const std::uint16_t scale,
                                                  const std::uint8_t low,
                                                  const std::uint8_t high) {
  emel::kernel::detail::quant::block_q4_0 block{};
  block.d = scale;
  for (auto &packed : block.qs) {
    packed = static_cast<std::uint8_t>((low & 0x0fu) | ((high & 0x0fu) << 4u));
  }
  return block;
}

emel::kernel::detail::quant::block_q8_0 q8_block(const std::uint16_t scale,
                                                  const std::int8_t value) {
  emel::kernel::detail::quant::block_q8_0 block{};
  block.d = scale;
  block.qs.fill(value);
  return block;
}

emel::kernel::detail::quant::block_q5_k q5_k_varied_block() {
  emel::kernel::detail::quant::block_q5_k block{};
  block.d = 0x3c00u;
  block.dmin = 0x3800u;
  block.scales.fill(1u);
  for (std::size_t index = 4; index < 8; ++index) {
    block.scales[index] = 2u;
  }
  for (std::size_t lane = 0; lane < block.qh.size(); ++lane) {
    block.qh[lane] = static_cast<std::uint8_t>(lane * 37u + 0x5au);
  }
  for (std::size_t chunk = 0; chunk < 4; ++chunk) {
    for (std::size_t lane = 0; lane < 32; ++lane) {
      const auto low = static_cast<std::uint8_t>((lane + chunk * 3u) % 16u);
      const auto high = static_cast<std::uint8_t>((15u + chunk + 16u - lane) % 16u);
      block.qs[chunk * 32u + lane] =
          static_cast<std::uint8_t>(low | static_cast<std::uint8_t>(high << 4u));
    }
  }
  return block;
}

emel::kernel::detail::quant::block_q8_k q8_k_varied_block() {
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

emel::kernel::detail::quant::block_q8_k q8_k_seeded_block(const std::uint8_t seed) {
  emel::kernel::detail::quant::block_q8_k block{};
  block.d = 1.0f;
  for (std::size_t lane = 0; lane < block.qs.size(); ++lane) {
    const auto value = static_cast<int>((lane * 17u + seed) % 29u) - 14;
    block.qs[lane] = static_cast<std::int8_t>(value);
  }
  for (std::size_t group = 0; group < block.bsums.size(); ++group) {
    for (std::size_t lane = 0; lane < 16; ++lane) {
      block.bsums[group] += block.qs[group * 16 + lane];
    }
  }
  return block;
}

emel::kernel::detail::quant::block_q2_k q2_k_varied_block(const std::uint8_t seed) {
  emel::kernel::detail::quant::block_q2_k block{};
  for (std::size_t lane = 0; lane < block.scales.size(); ++lane) {
    const auto low = static_cast<std::uint8_t>(lane * 29u + seed) & 0x0fu;
    const auto high = static_cast<std::uint8_t>(lane * 17u + 3u) & 0x0fu;
    block.scales[lane] = static_cast<std::uint8_t>(low | (high << 4u));
  }
  for (std::size_t lane = 0; lane < block.qs.size(); ++lane) {
    block.qs[lane] = static_cast<std::uint8_t>(lane * 73u + seed);
  }
  block.d = 0x3c00u;
  block.dmin = 0x3800u;
  return block;
}

emel::kernel::detail::quant::block_q3_k q3_k_varied_block(const std::uint8_t seed) {
  emel::kernel::detail::quant::block_q3_k block{};
  for (std::size_t lane = 0; lane < block.hmask.size(); ++lane) {
    block.hmask[lane] = static_cast<std::uint8_t>(lane * 41u + seed);
  }
  for (std::size_t lane = 0; lane < block.qs.size(); ++lane) {
    block.qs[lane] = static_cast<std::uint8_t>(lane * 67u + (seed ^ 0xa5u));
  }
  for (std::size_t lane = 0; lane < block.scales.size(); ++lane) {
    block.scales[lane] = static_cast<std::uint8_t>(lane * 19u + (seed ^ 0x3cu));
  }
  block.d = 0x3c00u;
  return block;
}

emel::kernel::detail::quant::block_q4_k q4_k_varied_block(const std::uint8_t seed) {
  emel::kernel::detail::quant::block_q4_k block{};
  block.d = 0x3c00u;
  block.dmin = 0x3800u;
  for (std::size_t lane = 0; lane < block.scales.size(); ++lane) {
    block.scales[lane] = static_cast<std::uint8_t>(lane * 23u + seed);
  }
  for (std::size_t lane = 0; lane < block.qs.size(); ++lane) {
    const auto low = static_cast<std::uint8_t>(lane * 11u + seed) & 0x0fu;
    const auto high = static_cast<std::uint8_t>(lane * 7u + (seed ^ 0x55u)) & 0x0fu;
    block.qs[lane] = static_cast<std::uint8_t>(low | (high << 4u));
  }
  return block;
}

emel::kernel::detail::quant::block_q6_k q6_k_varied_block(const std::uint8_t seed) {
  emel::kernel::detail::quant::block_q6_k block{};
  for (std::size_t lane = 0; lane < block.ql.size(); ++lane) {
    block.ql[lane] = static_cast<std::uint8_t>(lane * 13u + seed);
  }
  for (std::size_t lane = 0; lane < block.qh.size(); ++lane) {
    block.qh[lane] = static_cast<std::uint8_t>(lane * 31u + (seed ^ 0xc3u));
  }
  for (std::size_t lane = 0; lane < block.scales.size(); ++lane) {
    block.scales[lane] = static_cast<std::int8_t>(
        static_cast<int>((lane * 5u + seed) % 15u) - 7);
  }
  block.d = 0x3c00u;
  return block;
}

template <typename Function>
void write_result(const char *name, Function function) {
  const float value = function();
  std::cout << "case=" << name << " status=ok output_bits=" << std::hex
            << std::setw(8) << std::setfill('0') << std::bit_cast<std::uint32_t>(value)
            << std::dec << '\n';
}

} // namespace

int main() {
  using emel::kernel::detail::dot_q4_0_q8_0_row_scalar;
  using emel::kernel::detail::dot_q2_k_q8_k_block_scalar;
  using emel::kernel::detail::dot_q3_k_q8_k_row_scalar;
  using emel::kernel::detail::dot_q4_k_q8_k_row_scalar;
  using emel::kernel::detail::dot_q6_k_q8_k_row_scalar;
  using emel::kernel::detail::dot_q5_k_q8_k_row_scalar;
  using emel::kernel::detail::dot_q8_0_q8_0_row_scalar;
  using emel::kernel::detail::quant::block_q4_0;
  using emel::kernel::detail::quant::block_q2_k;
  using emel::kernel::detail::quant::block_q5_k;
  using emel::kernel::detail::quant::block_q8_0;
  using emel::kernel::detail::quant::block_q8_k;

  std::cout << "kernel-quantized-parity/v4\n"
            << "source_repository=stateforward/emel.cpp\n"
            << "source_commit=" << k_source_commit << '\n'
            << "scope=scalar_q4_0_q8_0_q5_k_q8_k_q2_k_row_aggregation_via_block_scalar_reference_q3_k_q4_k_q6_k_rows\n";

  const block_q4_0 q4[] = {q4_block(0x3c00, 0, 15), q4_block(0x3800, 9, 7)};
  const block_q8_0 q8[] = {q8_block(0x3800, 2), q8_block(0x3c00, -3)};
  const block_q8_0 q8_other[] = {q8_block(0x3c00, 3), q8_block(0x3800, -2)};

  write_result("dot_q4_0_q8_0", [&] {
    return dot_q4_0_q8_0_row_scalar(q4, q8, 2);
  });
  write_result("dot_q8_0_q8_0", [&] {
    return dot_q8_0_q8_0_row_scalar(q8, q8_other, 2);
  });
  const block_q5_k q5_k = q5_k_varied_block();
  const block_q8_k q8_k = q8_k_varied_block();
  write_result("dot_q5_k_q8_k_varied_qs", [&] {
    return dot_q5_k_q8_k_row_scalar(&q5_k, &q8_k, 1);
  });
  const block_q2_k q2_k[] = {q2_k_varied_block(17), q2_k_varied_block(91)};
  const auto q3_k = q3_k_varied_block(29);
  const auto q4_k = q4_k_varied_block(41);
  const auto q6_k = q6_k_varied_block(53);
  const block_q8_k q2_q8[] = {q8_k_seeded_block(17), q8_k_seeded_block(91)};
  const auto q3_q8 = q8_k_seeded_block(29);
  const auto q4_q8 = q8_k_seeded_block(41);
  const auto q6_q8 = q8_k_seeded_block(53);
  write_result("dot_q2_k_q8_k_row_aggregation_via_block_scalar_reference", [&] {
    float sum = 0.0f;
    for (std::size_t block = 0; block < 2; ++block) {
      sum += dot_q2_k_q8_k_block_scalar(q2_k[block], q2_q8[block]);
    }
    return sum;
  });
  write_result("dot_q3_k_q8_k_varied", [&] {
    return dot_q3_k_q8_k_row_scalar(&q3_k, &q3_q8, 1);
  });
  write_result("dot_q4_k_q8_k_varied", [&] {
    return dot_q4_k_q8_k_row_scalar(&q4_k, &q4_q8, 1);
  });
  write_result("dot_q6_k_q8_k_varied", [&] {
    return dot_q6_k_q8_k_row_scalar(&q6_k, &q6_q8, 1);
  });
}
