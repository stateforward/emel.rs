#include <array>
#include <bit>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>

#include "emel/kernel/sm.hpp"

namespace {
constexpr char k_commit[] = "843a117386ef17dc5a50549bbfc821074c2141d6";
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
}  // namespace

int main() {
  const std::array<float, 4> input = {0.0F, -1.5F, 2.25F, 4.0F};
  const std::array<float, 4> lhs = {8.0F, -9.0F, 6.0F, 4.0F};
  const std::array<float, 4> rhs = {2.0F, 3.0F, -2.0F, 4.0F};
  const std::array<float, 4> square_input = {-3.5F, -1.0F, 0.5F, 2.25F};

  std::cout << "kernel-portable-router-live/v1\n"
            << "source_repository=stateforward/emel.cpp\n"
            << "source_commit=" << k_commit << "\n"
            << "target_arch=portable\n"
            << "scope=portable_router_dup_add_sqr_and_typed_unexpected_rejection\n"
            << "execution=one_public_kernel_dispatching_three_live_events\n";

  emel::kernel::sm kernel;
  std::array<float, 4> output{};

  emel::kernel::event::op_dup dup{};
  dup.src0 = view(input.data(), input.size());
  dup.dst = view_mut(output.data(), output.size());
  if (!kernel.process_event(dup)) return 1;
  std::cout << "case=op_dup status=ok output_bits=";
  write_bits(output);
  std::cout << '\n';

  emel::kernel::event::op_add add{};
  add.src0 = view(lhs.data(), lhs.size());
  add.src1 = view(rhs.data(), rhs.size());
  add.dst = view_mut(output.data(), output.size());
  if (!kernel.process_event(add)) return 1;
  std::cout << "case=op_add status=ok output_bits=";
  write_bits(output);
  std::cout << '\n';

  emel::kernel::event::op_sqr sqr{};
  sqr.src0 = view(square_input.data(), square_input.size());
  sqr.dst = view_mut(output.data(), output.size());
  if (!kernel.process_event(sqr)) return 1;
  std::cout << "case=op_sqr status=ok output_bits=";
  write_bits(output);
  std::cout << '\n';

  std::array<float, 4> rejected;
  rejected.fill(k_sentinel);
  emel::kernel::event::op_dup malformed{};
  malformed.src0 = view(input.data(), input.size());
  malformed.dst = view_mut(rejected.data(), 3);
  if (kernel.process_event(malformed)) return 1;
  std::cout << "case=typed_unexpected_event status=reject error=UnexpectedEvent output_bits=";
  write_bits(rejected);
  std::cout << '\n';
}
