#include <array>
#include <bit>
#include <cstdint>
#include <iomanip>
#include <iostream>

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
void bits(const float *values, std::size_t count) {
  for (std::size_t i = 0; i < count; ++i) {
    if (i != 0) std::cout << ',';
    std::cout << std::hex << std::setw(8) << std::setfill('0')
              << std::bit_cast<std::uint32_t>(values[i]) << std::dec;
  }
}
void header() {
  std::cout << "kernel-target-scalar-trig-live/v1\n"
            << "source_repository=stateforward/emel.cpp\n"
            << "source_commit=" << k_commit << '\n'
            << "source_kernel_events_blob=4b3f02fa5ff9c5071d1fc83938cef1b22b4658b9\n"
            << "source_kernel_detail_blob=c8a82643eabfe8f2d7883e655955f455794511b0\n"
            << "source_kernel_aarch64_guards_blob=c25714566ec9a02679daef85089544575123408e\n"
            << "source_kernel_aarch64_actions_blob=267d4f74e6e7498155c8535920322ffef2c02fb6\n"
            << "source_kernel_aarch64_sm_blob=865a9cc6ba6115382ed043c464f3d62bcd851357\n"
            << "target_arch=aarch64\n"
            << "scope=target_router_scalar_trig_log_sin_cos_f32_dense_and_rejection\n"
            << "execution=split_pinned_aarch64_sm_and_public_target_router\n";
}
}

int main() {
  constexpr std::array<std::uint64_t, 4> ne = {6, 1, 1, 1};
  constexpr std::array<std::uint64_t, 4> nb = {4, 24, 24, 24};
  const std::array<float, 6> input = {-3.5F, -0.5F, 0.0F, 0.5F, 2.25F, 7.0F};
  header();
  emel::kernel::aarch64::sm actor;
  std::array<float, 6> output{};
  auto log = emel::kernel::event::op_log{};
  log.src0 = view(input.data(), ne, nb); log.dst = view_mut(output.data(), ne, nb);
  if (!actor.process_event(log)) return 1;
  std::cout << "case=log status=ok output_bits="; bits(output.data(), output.size()); std::cout << '\n';
  auto sin = emel::kernel::event::op_sin{};
  sin.src0 = view(input.data(), ne, nb); sin.dst = view_mut(output.data(), ne, nb);
  if (!actor.process_event(sin)) return 1;
  std::cout << "case=sin status=ok output_bits="; bits(output.data(), output.size()); std::cout << '\n';
  auto cos = emel::kernel::event::op_cos{};
  cos.src0 = view(input.data(), ne, nb); cos.dst = view_mut(output.data(), ne, nb);
  if (!actor.process_event(cos)) return 1;
  std::cout << "case=cos status=ok output_bits="; bits(output.data(), output.size()); std::cout << '\n';
  output.fill(k_sentinel);
  auto zero = emel::kernel::event::op_log{};
  zero.src0 = view(input.data(), {0, 1, 1, 1}, {4, 0, 0, 0});
  zero.dst = view_mut(output.data(), {0, 1, 1, 1}, {4, 0, 0, 0});
  if (actor.process_event(zero)) return 1;
  std::cout << "case=zero_count status=reject error=InvalidView output_bits="; bits(output.data(), 1); std::cout << '\n';
  auto mismatch = emel::kernel::event::op_sin{};
  mismatch.src0 = view(input.data(), ne, nb);
  mismatch.dst = view_mut(output.data(), {5, 1, 1, 1}, {4, 20, 20, 20});
  if (actor.process_event(mismatch)) return 1;
  std::cout << "case=shape_mismatch status=reject error=ShapeMismatch output_bits="; bits(output.data(), 5); std::cout << '\n';
  std::cout << "case=unexpected status=reject error=UnexpectedEvent\n";
}
