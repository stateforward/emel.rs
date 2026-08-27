#include <array>
#include <bit>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include "emel/kernel/sm.hpp"

namespace {
constexpr float k_sentinel = std::bit_cast<float>(std::uint32_t{0x7fc00001});
emel::kernel::event::tensor_view view(const float *data,
    const std::array<std::uint64_t, 4> ne, const std::array<std::uint64_t, 4> nb,
    const emel::kernel::event::dtype type = emel::kernel::event::dtype::f32) {
  return {.data = data, .type = type, .ne = ne, .nb = nb};
}
emel::kernel::event::tensor_view_mut view_mut(float *data,
    const std::array<std::uint64_t, 4> ne, const std::array<std::uint64_t, 4> nb,
    const emel::kernel::event::dtype type = emel::kernel::event::dtype::f32) {
  return {.data = data, .type = type, .ne = ne, .nb = nb};
}
void bits(const float *values, std::size_t count) {
  for (std::size_t i = 0; i < count; ++i) {
    if (i != 0) std::cout << ',';
    std::cout << std::hex << std::setw(8) << std::setfill('0')
              << std::bit_cast<std::uint32_t>(values[i]) << std::dec;
  }
}
}

int main() {
  std::cout << "kernel-softmax-live/v1\nsource_repository=stateforward/emel.cpp\nsource_commit=843a117386ef17dc5a50549bbfc821074c2141d6\nsource_sml_commit=49207123cd3f39767764bae774932cb48623f92f\nsource_kernel_events_blob=4b3f02fa5ff9c5071d1fc83938cef1b22b4658b9\nsource_kernel_detail_blob=c8a82643eabfe8f2d7883e655955f455794511b0\nsource_kernel_sm_blob=cdce31c5f70501f9c26c66886c3348d774a8dc48\nscope=portable_f32_op_soft_max_dense_and_contract_rejection\nexecution=public_emel_kernel_sm_process_event\n";
  const std::array<float, 6> input = {1.0F, 2.0F, 4.0F, 2.0F, 5.0F, 8.0F};
  emel::kernel::Kernel kernel;
  auto request = emel::kernel::event::op_soft_max{};
  std::array<float, 6> output{};
  request.src0 = view(input.data(), {3, 2, 1, 1}, {4, 12, 24, 24});
  request.dst = view_mut(output.data(), {3, 2, 1, 1}, {4, 12, 24, 24});
  if (!kernel.process_event(request)) return 1;
  std::cout << "case=dense_rows status=ok output_bits="; bits(output.data(), 6); std::cout << '\n';

  output.fill(0.0F);
  if (!kernel.process_event(request)) return 1;
  std::cout << "case=implicit_contiguous status=ok output_bits="; bits(output.data(), 6); std::cout << '\n';

  std::array<float, 4> mismatch_output;
  mismatch_output.fill(k_sentinel);
  request.src0 = view(input.data(), {3, 2, 1, 1}, {4, 12, 24, 24});
  request.dst = view_mut(mismatch_output.data(), {2, 2, 1, 1}, {4, 8, 16, 16});
  if (kernel.process_event(request)) return 1;
  std::cout << "case=shape_mismatch status=reject error=ShapeMismatch output_bits="; bits(mismatch_output.data(), 4); std::cout << '\n';
}
