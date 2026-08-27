#include <bit>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "emel/kernel/events.hpp"
#include "emel/kernel/sm.hpp"

namespace {

constexpr char k_source_commit[] =
    "843a117386ef17dc5a50549bbfc821074c2141d6";

void write_case(const char *name, const std::vector<float> &values) {
  std::cout << "case=" << name << " status=ok output_bits=";
  for (std::size_t index = 0; index < values.size(); ++index) {
    if (index != 0) {
      std::cout << ',';
    }
    std::cout << std::hex << std::setw(8) << std::setfill('0')
              << std::bit_cast<std::uint32_t>(values[index]) << std::dec;
  }
  std::cout << '\n';
}

void run_case(const char *name, const std::array<std::uint64_t, 4> kernel_shape,
              const std::array<std::uint64_t, 4> input_shape,
              const std::array<std::uint64_t, 4> output_shape,
              const std::vector<float> &input, const std::int32_t stride,
              const std::int32_t padding, const std::int32_t dilation,
              const bool implicit_layouts) {
  using emel::kernel::event::dtype;
  using emel::kernel::event::op_im2col;
  using emel::kernel::event::tensor_view;
  using emel::kernel::event::tensor_view_mut;

  op_im2col request{};
  const std::vector<float> kernel_data(kernel_shape[0] * kernel_shape[1], 0.0F);
  request.src0 = tensor_view{kernel_data.data(), dtype::f32, kernel_shape, {4, 4 * kernel_shape[0], 4 * kernel_shape[0] * kernel_shape[1], 4 * kernel_shape[0] * kernel_shape[1]}};
  const auto src1_nb = implicit_layouts
                           ? std::array<std::uint64_t, 4>{0, 0, 0, 0}
                           : std::array<std::uint64_t, 4>{
                                 4, 4 * input_shape[0],
                                 4 * input_shape[0] * input_shape[1],
                                 4 * input_shape[0] * input_shape[1] * input_shape[2]};
  request.src1 = tensor_view{input.data(), dtype::f32, input_shape, src1_nb};
  std::vector<float> output(output_shape[0] * output_shape[1] * output_shape[2]);
  const auto dst_nb = implicit_layouts
                          ? std::array<std::uint64_t, 4>{0, 0, 0, 0}
                          : std::array<std::uint64_t, 4>{
                                4, 4 * output_shape[0],
                                4 * output_shape[0] * output_shape[1],
                                4 * output_shape[0] * output_shape[1] * output_shape[2]};
  request.dst = tensor_view_mut{output.data(), dtype::f32, output_shape, dst_nb};
  request.op_params_size = 28;
  std::memcpy(request.op_params.data() + 0, &stride, sizeof(stride));
  std::memcpy(request.op_params.data() + 8, &padding, sizeof(padding));
  std::memcpy(request.op_params.data() + 16, &dilation, sizeof(dilation));
  const std::int32_t is_2d = 0;
  std::memcpy(request.op_params.data() + 24, &is_2d, sizeof(is_2d));
  emel::kernel::Kernel kernel;
  if (!kernel.process_event(request)) {
    std::cout << "case=" << name << " status=error\n";
    return;
  }
  write_case(name, output);
}

}  // namespace

void benchmark() {
  constexpr std::size_t kIterations = 10000;
  constexpr std::size_t kWarmup = 1000;
  constexpr std::size_t kInputLength = 64;
  constexpr std::size_t kKernelLength = 3;
  constexpr std::size_t kOutputLength = kInputLength;
  constexpr std::size_t kOutputElements = kKernelLength * kOutputLength;
  const std::array<float, kKernelLength> kernel_data{};
  const std::vector<float> input(kInputLength, 1.0F);
  std::vector<float> output(kOutputElements, 0.0F);
  using emel::kernel::event::dtype;
  using emel::kernel::event::op_im2col;
  using emel::kernel::event::tensor_view;
  using emel::kernel::event::tensor_view_mut;
  op_im2col request{};
  request.src0 = tensor_view{
      kernel_data.data(), dtype::f32, {kKernelLength, 1, 1, 1},
      {4, 4 * kKernelLength, 4 * kKernelLength, 4 * kKernelLength}};
  request.src1 = tensor_view{input.data(), dtype::f32,
                             {kInputLength, 1, 1, 1},
                             {4, 4 * kInputLength, 4 * kInputLength,
                              4 * kInputLength}};
  request.dst = tensor_view_mut{output.data(), dtype::f32,
                                {kKernelLength, kOutputLength, 1, 1},
                                {4, 4 * kKernelLength,
                                 4 * kKernelLength * kOutputLength,
                                 4 * kKernelLength * kOutputLength}};
  const std::int32_t stride = 1;
  const std::int32_t padding = 1;
  const std::int32_t dilation = 1;
  const std::int32_t is_2d = 0;
  request.op_params_size = 28;
  std::memcpy(request.op_params.data() + 0, &stride, sizeof(stride));
  std::memcpy(request.op_params.data() + 8, &padding, sizeof(padding));
  std::memcpy(request.op_params.data() + 16, &dilation, sizeof(dilation));
  std::memcpy(request.op_params.data() + 24, &is_2d, sizeof(is_2d));
  emel::kernel::Kernel kernel;
  for (std::size_t index = 0; index < kWarmup; ++index) {
    if (!kernel.process_event(request)) {
      std::cerr << "im2col benchmark warmup failed\n";
      std::exit(1);
    }
  }
  const auto start = std::chrono::steady_clock::now();
  for (std::size_t index = 0; index < kIterations; ++index) {
    if (!kernel.process_event(request)) {
      std::cerr << "im2col benchmark failed\n";
      std::exit(1);
    }
  }
  const auto elapsed = std::chrono::duration<double, std::nano>(
      std::chrono::steady_clock::now() - start).count();
  if (output[0] != 0.0F || output[1] != 1.0F || output[2] != 1.0F) {
    std::cerr << "im2col benchmark output validation failed\n";
    std::exit(1);
  }
  std::cout << "case=op_im2col cpp_ns_per_dispatch="
            << (elapsed / static_cast<double>(kIterations))
            << " output=" << kOutputElements
            << " probe_bits=00000000,3f800000,3f800000\n";
}

int main(int argc, char **argv) {
  if (argc > 1 && std::string(argv[1]) == "--benchmark") {
    benchmark();
    return 0;
  }
  std::cout << "kernel-im2col-parity/v2\n"
            << "source_repository=stateforward/emel.cpp\n"
            << "source_commit=" << k_source_commit << '\n'
            << "source_kernel_events_blob=4b3f02fa5ff9c5071d1fc83938cef1b22b4658b9\n"
            << "source_kernel_detail_blob=c8a82643eabfe8f2d7883e655955f455794511b0\n"
            << "source_kernel_x86_sm_blob=0b4d635ebbd0fbd52dbca8a2345547fb571205c8\n"
            << "scope=portable_scalar_f32_1d_dense_output\n";

  run_case("zero_padding", {3, 1, 1, 1}, {4, 1, 1, 1}, {3, 4, 1, 1},
           {1.0F, 2.0F, 3.0F, 4.0F}, 1, 1, 1, false);
  run_case("channels_batches_stride_dilation", {2, 2, 1, 1}, {5, 2, 2, 1},
           {4, 2, 2, 1},
           {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 10.0F, 20.0F, 30.0F, 40.0F,
           50.0F, 101.0F, 102.0F, 103.0F, 104.0F, 105.0F, 110.0F, 120.0F,
           130.0F, 140.0F, 150.0F},
           2, 0, 2, false);
  run_case("implicit_contiguous", {2, 1, 1, 1}, {3, 1, 1, 1}, {2, 2, 1, 1},
           {7.0F, 8.0F, 9.0F}, 1, 0, 1, true);
  run_case("zero_batch_implicit_noop", {2, 1, 1, 1}, {3, 1, 0, 1},
           {2, 2, 0, 1}, {0.0F}, 1, 0, 1, true);
}
