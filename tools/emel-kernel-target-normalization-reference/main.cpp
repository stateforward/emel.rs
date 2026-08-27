#include <array>
#include <bit>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>

#include "emel/kernel/aarch64/sm.hpp"

namespace {
constexpr char k_commit[] = "843a117386ef17dc5a50549bbfc821074c2141d6";
constexpr float k_sentinel = std::bit_cast<float>(std::uint32_t{0x7fc00001});

emel::kernel::event::tensor_view view(const float *data,
                                     const std::array<std::uint64_t, 4> ne,
                                     const std::array<std::uint64_t, 4> nb) {
  return {.data = data, .type = emel::kernel::event::dtype::f32,
          .ne = ne, .nb = nb};
}
emel::kernel::event::tensor_view_mut view_mut(
    float *data, const std::array<std::uint64_t, 4> ne,
    const std::array<std::uint64_t, 4> nb) {
  return {.data = data, .type = emel::kernel::event::dtype::f32,
          .ne = ne, .nb = nb};
}
void epsilon(auto &request, const float value) {
  std::memcpy(request.op_params.data(), &value, sizeof(value));
  request.op_params_size = sizeof(value);
}
void bits(const float *values, std::size_t count) {
  for (std::size_t i = 0; i < count; ++i) {
    if (i != 0) std::cout << ',';
    std::cout << std::hex << std::setw(8) << std::setfill('0')
              << std::bit_cast<std::uint32_t>(values[i]) << std::dec;
  }
}
void header() {
  std::cout << "kernel-target-normalization-live/v1\n"
            << "source_repository=stateforward/emel.cpp\n"
            << "source_commit=" << k_commit << '\n'
            << "source_kernel_events_blob=4b3f02fa5ff9c5071d1fc83938cef1b22b4658b9\n"
            << "source_kernel_detail_blob=c8a82643eabfe8f2d7883e655955f455794511b0\n"
            << "source_kernel_aarch64_guards_blob=c25714566ec9a02679daef85089544575123408e\n"
            << "source_kernel_aarch64_actions_blob=267d4f74e6e7498155c8535920322ffef2c02fb6\n"
            << "source_kernel_aarch64_sm_blob=865a9cc6ba6115382ed043c464f3d62bcd851357\n"
            << "target_arch=aarch64\n"
            << "scope=target_router_norm_rms_norm_f32_dense\n"
            << "execution=split_pinned_aarch64_sm_and_public_target_router\n";
}
}

int main() {
  header();
  const std::array<float, 6> input = {1.0F, 2.0F, 4.0F, 2.0F, 5.0F, 8.0F};
  emel::kernel::aarch64::sm actor;
  std::array<float, 6> output{};
  auto norm = emel::kernel::event::op_norm{};
  norm.src0 = view(input.data(), {3, 2, 1, 1}, {4, 12, 24, 24});
  norm.dst = view_mut(output.data(), {3, 2, 1, 1}, {4, 12, 24, 24});
  epsilon(norm, 1.0F);
  if (!actor.process_event(norm)) return 1;
  std::cout << "case=norm status=ok output_bits="; bits(output.data(), output.size()); std::cout << '\n';
  auto rms = emel::kernel::event::op_rms_norm{};
  rms.src0 = view(input.data(), {3, 2, 1, 1}, {4, 12, 24, 24});
  rms.dst = view_mut(output.data(), {3, 2, 1, 1}, {4, 12, 24, 24});
  epsilon(rms, 1.0F);
  if (!actor.process_event(rms)) return 1;
  std::cout << "case=rms_norm status=ok output_bits="; bits(output.data(), output.size()); std::cout << '\n';
  output.fill(k_sentinel);
  auto invalid = emel::kernel::event::op_norm{};
  invalid.src0 = view(input.data(), {3, 2, 1, 1}, {4, 12, 24, 24});
  invalid.dst = view_mut(output.data(), {3, 2, 1, 1}, {4, 12, 24, 24});
  epsilon(invalid, std::numeric_limits<float>::quiet_NaN());
  if (actor.process_event(invalid)) return 1;
  std::cout << "case=invalid_epsilon status=reject error=InvalidParameters output_bits="; bits(output.data(), output.size()); std::cout << '\n';
  auto zero = emel::kernel::event::op_rms_norm{};
  zero.src0 = view(input.data(), {0, 2, 1, 1}, {4, 0, 0, 0});
  zero.dst = view_mut(output.data(), {0, 2, 1, 1}, {4, 0, 0, 0});
  epsilon(zero, 1.0F);
  if (actor.process_event(zero)) return 1;
  std::cout << "case=zero_columns status=reject error=InvalidView output_bits=";
  bits(output.data(), 1); std::cout << '\n';
  auto mismatch = emel::kernel::event::op_norm{};
  mismatch.src0 = view(input.data(), {3, 2, 1, 1}, {4, 12, 24, 24});
  mismatch.dst = view_mut(output.data(), {2, 2, 1, 1}, {4, 8, 16, 16});
  epsilon(mismatch, 1.0F);
  if (actor.process_event(mismatch)) return 1;
  std::cout << "case=shape_mismatch status=reject error=ShapeMismatch output_bits=";
  bits(output.data(), 4); std::cout << '\n';
}
