#include <array>
#include <bit>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>

#include "emel/kernel/aarch64/sm.hpp"

namespace {
constexpr char k_commit[] = "843a117386ef17dc5a50549bbfc821074c2141d6";
constexpr float k_sentinel = std::bit_cast<float>(std::uint32_t{0x7fc00001});

emel::kernel::event::tensor_view view(const float *data, std::array<std::uint64_t, 4> ne,
                                      std::array<std::uint64_t, 4> nb) {
  return {.data = data, .type = emel::kernel::event::dtype::f32, .ne = ne, .nb = nb};
}
emel::kernel::event::tensor_view_mut view_mut(float *data, std::array<std::uint64_t, 4> ne,
                                              std::array<std::uint64_t, 4> nb) {
  return {.data = data, .type = emel::kernel::event::dtype::f32, .ne = ne, .nb = nb};
}
void params(emel::kernel::event::op_im2col &request, std::int32_t is_2d) {
  const std::int32_t stride = 1, padding = 1, dilation = 1;
  request.op_params_size = 28;
  std::memcpy(request.op_params.data() + 0, &stride, 4);
  std::memcpy(request.op_params.data() + 8, &padding, 4);
  std::memcpy(request.op_params.data() + 16, &dilation, 4);
  std::memcpy(request.op_params.data() + 24, &is_2d, 4);
}
void bits(const float *values, std::size_t count) {
  for (std::size_t i = 0; i < count; ++i) {
    if (i) std::cout << ',';
    std::cout << std::hex << std::setw(8) << std::setfill('0')
              << std::bit_cast<std::uint32_t>(values[i]) << std::dec;
  }
}
void header() {
  std::cout << "kernel-target-im2col-live/v1\n"
            << "source_repository=stateforward/emel.cpp\n"
            << "source_commit=" << k_commit << '\n'
            << "source_kernel_events_blob=4b3f02fa5ff9c5071d1fc83938cef1b22b4658b9\n"
            << "source_kernel_detail_blob=c8a82643eabfe8f2d7883e655955f455794511b0\n"
            << "source_kernel_aarch64_guards_blob=c25714566ec9a02679daef85089544575123408e\n"
            << "source_kernel_aarch64_actions_blob=267d4f74e6e7498155c8535920322ffef2c02fb6\n"
            << "source_kernel_aarch64_sm_blob=865a9cc6ba6115382ed043c464f3d62bcd851357\n"
            << "target_arch=aarch64\n"
            << "scope=target_router_im2col_f32_1d_dense\n"
            << "execution=split_pinned_aarch64_sm_and_public_target_router\n";
}
}  // namespace

int main() {
  header();
  const std::array<float, 4> input = {1.0F, 2.0F, 3.0F, 4.0F};
  const auto kernel_ne = std::array<std::uint64_t, 4>{3, 1, 1, 1};
  const auto input_ne = std::array<std::uint64_t, 4>{4, 1, 1, 1};
  const auto output_ne = std::array<std::uint64_t, 4>{3, 4, 1, 1};
  const auto input_dense = std::array<std::uint64_t, 4>{4, 16, 64, 64};
  const auto output_dense = std::array<std::uint64_t, 4>{4, 12, 48, 48};
  const std::array<float, 3> kernel_data = {0.0F, 0.0F, 0.0F};
  emel::kernel::aarch64::sm actor;
  std::array<float, 12> output{};
  emel::kernel::event::op_im2col request{};
  request.src0 = view(kernel_data.data(), kernel_ne, {4, 12, 12, 12});
  request.src1 = view(input.data(), input_ne, input_dense);
  request.dst = view_mut(output.data(), output_ne, output_dense);
  params(request, 0);
  if (!actor.process_event(request)) return 1;
  std::cout << "case=zero_padding status=ok output_bits="; bits(output.data(), output.size()); std::cout << '\n';

  output.fill(k_sentinel);
  request = {};
  request.src0 = view(kernel_data.data(), kernel_ne, {4, 12, 12, 12});
  request.src1 = view(input.data(), input_ne, input_dense);
  request.dst = view_mut(output.data(), output_ne, output_dense);
  params(request, 1);
  if (actor.process_event(request)) return 1;
  std::cout << "case=invalid_parameters status=reject error=InvalidParameters output_bits="; bits(output.data(), output.size()); std::cout << '\n';

  std::array<float, 9> mismatch;
  mismatch.fill(k_sentinel);
  request = {};
  request.src0 = view(kernel_data.data(), kernel_ne, {4, 12, 12, 12});
  request.src1 = view(input.data(), input_ne, input_dense);
  request.dst = view_mut(mismatch.data(), {3, 3, 1, 1}, {4, 12, 36, 36});
  params(request, 0);
  if (actor.process_event(request)) return 1;
  std::cout << "case=shape_mismatch status=reject error=ShapeMismatch output_bits="; bits(mismatch.data(), mismatch.size()); std::cout << '\n';
}
