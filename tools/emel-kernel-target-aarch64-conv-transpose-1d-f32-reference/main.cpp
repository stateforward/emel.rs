#include <array>
#include <bit>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>

#include "emel/kernel/aarch64/sm.hpp"

namespace {
constexpr char k_commit[] = "843a117386ef17dc5a50549bbfc821074c2141d6";
constexpr char k_guards[] = "c25714566ec9a02679daef85089544575123408e";
constexpr char k_actions[] = "267d4f74e6e7498155c8535920322ffef2c02fb6";
constexpr char k_sm[] = "865a9cc6ba6115382ed043c464f3d62bcd851357";
constexpr float k_sentinel = std::bit_cast<float>(std::uint32_t{0x7fc00001});

emel::kernel::event::tensor_view view(const float *data,
                                      std::array<std::uint64_t, 4> ne,
                                      std::array<std::uint64_t, 4> nb) {
  return {.data = data, .type = emel::kernel::event::dtype::f32, .ne = ne, .nb = nb};
}

emel::kernel::event::tensor_view_mut view_mut(float *data,
                                             std::array<std::uint64_t, 4> ne,
                                             std::array<std::uint64_t, 4> nb) {
  return {.data = data, .type = emel::kernel::event::dtype::f32, .ne = ne, .nb = nb};
}

void bits(const float *values, std::size_t count) {
  for (std::size_t index = 0; index < count; ++index) {
    if (index != 0) std::cout << ',';
    std::cout << std::hex << std::setw(8) << std::setfill('0')
              << std::bit_cast<std::uint32_t>(values[index]) << std::dec;
  }
}

bool execute(emel::kernel::aarch64::sm &kernel, const float *weights,
             const float *input, float *output, std::array<std::uint64_t, 4> dst_ne,
             std::array<std::uint64_t, 4> dst_nb, std::array<std::int32_t, 3> params) {
  emel::kernel::event::op_conv_transpose_1d request{};
  request.src0 = view(weights, {5, 2, 1, 1}, {4, 20, 40, 40});
  request.src1 = view(input, {2, 1, 1, 1}, {4, 8, 8, 8});
  request.dst = view_mut(output, dst_ne, dst_nb);
  for (std::size_t index = 0; index < params.size(); ++index) {
    std::memcpy(request.op_params.data() + index * sizeof(std::int32_t), &params[index],
                sizeof(std::int32_t));
  }
  request.op_params_size = static_cast<std::uint32_t>(params.size() * sizeof(std::int32_t));
  return kernel.process_event(request);
}
}  // namespace

int main() {
  constexpr std::array<float, 10> weights = {1.0F, 2.0F, 3.0F, 4.0F, 5.0F,
                                             0.5F, 1.5F, 2.5F, 3.5F, 4.5F};
  constexpr std::array<float, 2> input = {1.0F, 2.0F};
  std::array<float, 14> output{};
  emel::kernel::aarch64::sm kernel;
  std::cout << "kernel-target-aarch64-conv-transpose-1d-f32-live/v1\n"
            << "source_repository=stateforward/emel.cpp\n"
            << "source_commit=" << k_commit << '\n'
            << "source_kernel_aarch64_guards_blob=" << k_guards << '\n'
            << "source_kernel_aarch64_actions_blob=" << k_actions << '\n'
            << "source_kernel_aarch64_sm_blob=" << k_sm << '\n'
            << "target_arch=aarch64\n"
            << "event=op_conv_transpose_1d shape=kernel5_out2_in1_length2_stride2_padding0_dilation1\n"
            << "scope=dense_f32_weights_input_output\n"
            << "execution=split_pinned_aarch64_sm_and_public_target_router\n";
  const bool accepted = execute(kernel, weights.data(), input.data(), output.data(),
                                {7, 2, 1, 1}, {4, 28, 56, 56}, {2, 0, 1});
  if (!accepted) return 1;
  std::cout << "case=valid_dense_f32_neon status=ok output_bits=";
  bits(output.data(), output.size());
  std::cout << '\n';

  output.fill(k_sentinel);
  const bool invalid_length = execute(kernel, weights.data(), input.data(), output.data(),
                                      {9, 2, 1, 1}, {4, 36, 72, 72}, {2, 0, 1});
  if (invalid_length) return 1;
  std::cout << "case=invalid_input_length status=reject error=InvalidShape output_bits=";
  bits(output.data(), output.size());
  std::cout << '\n';

  output.fill(k_sentinel);
  const bool invalid_padding = execute(kernel, weights.data(), input.data(), output.data(),
                                       {7, 2, 1, 1}, {4, 28, 56, 56}, {2, 1, 1});
  if (invalid_padding) return 1;
  std::cout << "case=invalid_padding status=reject error=InvalidShape output_bits=";
  bits(output.data(), output.size());
  std::cout << '\n';
}
