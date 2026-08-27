#include <array>
#include <bit>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <sstream>

#include "emel/kernel/sm.hpp"

namespace {

constexpr char k_source_commit[] =
    "843a117386ef17dc5a50549bbfc821074c2141d6";

emel::kernel::event::tensor_view view(const float *data,
                                      const std::array<std::uint64_t, 4> ne,
                                      const std::array<std::uint64_t, 4> nb) {
  return {.data = data,
          .type = emel::kernel::event::dtype::f32,
          .ne = ne,
          .nb = nb};
}

emel::kernel::event::tensor_view_mut view_mut(
    float *data, const std::array<std::uint64_t, 4> ne,
    const std::array<std::uint64_t, 4> nb) {
  return {.data = data,
          .type = emel::kernel::event::dtype::f32,
          .ne = ne,
          .nb = nb};
}

std::string bits(const float *values, const std::size_t count) {
  std::ostringstream output;
  for (std::size_t index = 0; index < count; ++index) {
    if (index != 0) {
      output << ',';
    }
    output << std::hex << std::setw(8) << std::setfill('0')
           << std::bit_cast<std::uint32_t>(values[index]);
  }
  return output.str();
}

template <std::size_t N>
bool execute(emel::kernel::Kernel &kernel,
             const std::array<float, N> &weights,
             const std::array<std::uint64_t, 4> weights_ne,
             const std::array<std::uint64_t, 4> weights_nb,
             const std::array<float, 11> &input,
             const std::array<std::uint64_t, 4> input_ne,
             const std::array<std::uint64_t, 4> input_nb,
             std::array<float, 15> &output,
             const std::array<std::uint64_t, 4> output_ne,
             const std::array<std::uint64_t, 4> output_nb,
             const std::array<std::int32_t, 3> params) {
  auto request = emel::kernel::event::op_conv_transpose_1d{};
  request.src0 = view(weights.data(), weights_ne, weights_nb);
  request.src1 = view(input.data(), input_ne, input_nb);
  request.dst = view_mut(output.data(), output_ne, output_nb);
  for (std::size_t index = 0; index < params.size(); ++index) {
    std::memcpy(request.op_params.data() + index * sizeof(std::int32_t),
                &params[index], sizeof(std::int32_t));
  }
  request.op_params_size = static_cast<std::uint32_t>(
      params.size() * sizeof(std::int32_t));
  return kernel.process_event(request);
}

void write_parity() {
  std::cout << "kernel-conv-transpose-1d-parity/v1\n"
            << "source_repository=stateforward/emel.cpp\n"
            << "source_commit=" << k_source_commit << '\n'
            << "source_kernel_events_blob=4b3f02fa5ff9c5071d1fc83938cef1b22b4658b9\n"
            << "source_kernel_detail_blob=c8a82643eabfe8f2d7883e655955f455794511b0\n"
            << "source_kernel_x86_sm_blob=0b4d635ebbd0fbd52dbca8a2345547fb571205c8\n"
            << "source_kernel_aarch64_sm_blob=865a9cc6ba6115382ed043c464f3d62bcd851357\n"
            << "source_detail_guard_span=4955-5009\n"
            << "source_detail_run_span=5011-5064\n"
            << "scope=f32_weights_aligned_nonzero_strides_explicit_dense_output\n";

  constexpr std::array<float, 6> weights = {1.0F, 2.0F, 3.0F,
                                             4.0F, 5.0F, 6.0F};
  constexpr std::array<float, 11> input = {1.0F, 0.0F, 2.0F, 0.0F, 3.0F, 0.0F,
                                            4.0F, 0.0F, 5.0F, 0.0F, 6.0F};
  std::array<float, 15> output{};
  emel::kernel::Kernel kernel;
  const auto status = execute(
      kernel, weights, {3, 2, 1, 1}, {4, 12, 24, 24}, input,
      {3, 1, 1, 1}, {8, 24, 24, 24}, output, {7, 2, 1, 1},
      {4, 28, 56, 56}, {2, 0, 1});
  std::cout << "case=valid_strided_f32 status=" << (status ? "ok" : "rejected")
            << " output_bits=" << bits(output.data(), 14) << '\n';

  std::array<float, 15> invalid_output;
  invalid_output.fill(9.0F);
  const auto invalid_status = execute(
      kernel, weights, {3, 2, 1, 1}, {4, 12, 24, 24}, input,
      {3, 1, 1, 1}, {8, 24, 24, 24}, invalid_output, {6, 2, 1, 1},
      {4, 24, 48, 48}, {2, 0, 1});
  std::cout << "case=invalid_shape status="
            << (invalid_status ? "ok" : "rejected")
            << " output_bits=" << bits(invalid_output.data(), 8) << '\n';
}

} // namespace

int main() {
  write_parity();
  return 0;
}
