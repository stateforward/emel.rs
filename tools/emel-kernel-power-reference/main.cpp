#include <array>
#include <bit>
#include <cstdint>
#include <iomanip>
#include <iostream>

#include "emel/kernel/sm.hpp"

namespace {

constexpr char k_source_commit[] =
    "843a117386ef17dc5a50549bbfc821074c2141d6";

emel::kernel::event::tensor_view view(
    const float *data, const std::array<std::uint64_t, 4> ne,
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

template <std::size_t N>
std::string bits(const std::array<float, N> &data,
                const std::array<std::uint64_t, 4> ne,
                const std::array<std::uint64_t, 4> nb) {
  auto strides = nb;
  if (strides[0] == 0) {
    strides[0] = sizeof(float);
    for (std::size_t dimension = 1; dimension < 4; ++dimension) {
      strides[dimension] = strides[dimension - 1] * ne[dimension - 1];
    }
  }
  const std::size_t count = static_cast<std::size_t>(
      ne[0] * ne[1] * ne[2] * ne[3]);
  std::string result;
  for (std::size_t ordinal = 0; ordinal < count; ++ordinal) {
    std::uint64_t remaining = ordinal;
    std::uint64_t byte_offset = 0;
    for (std::size_t dimension = 0; dimension < 4; ++dimension) {
      byte_offset += (remaining % ne[dimension]) * strides[dimension];
      remaining /= ne[dimension];
    }
    if (ordinal != 0) {
      result.push_back(',');
    }
    const auto value = data[byte_offset / sizeof(float)];
    std::uint32_t word = std::bit_cast<std::uint32_t>(value);
    for (int shift = 28; shift >= 0; shift -= 4) {
      result.push_back("0123456789abcdef"[(word >> shift) & 0x0f]);
    }
  }
  return result;
}

void print_header() {
  std::cout << "kernel-power-parity/v1\n"
            << "source_repository=stateforward/emel.cpp\n"
            << "source_commit=" << k_source_commit << '\n'
            << "source_sml_commit=49207123cd3f39767764bae774932cb48623f92f\n"
            << "source_kernel_events_blob=4b3f02fa5ff9c5071d1fc83938cef1b22b4658b9\n"
            << "source_kernel_detail_blob=c8a82643eabfe8f2d7883e655955f455794511b0\n"
            << "source_kernel_x86_sm_blob=0b4d635ebbd0fbd52dbca8a2345547fb571205c8\n"
            << "source_kernel_x86_actions_blob=d45558f5eb96950f43c16a09d768cb4f382d6d61\n"
            << "source_kernel_x86_guards_blob=cb3dac8253f8417c9b44acff1de414f6d0a3a3cf\n"
            << "source_detail_guard_span=3838-3841\n"
            << "source_detail_run_span=5353-5356\n"
            << "scope=portable_f32_dense_and_validated_strided_square_and_square_root\n"
            << "composition_root_cases=dense_square,count_equal_different_dimensions,dense_square_root,shape_rejection\n"
            << "composition_direct_child_cases=strided_square,implicit_contiguous_square\n";
}

void run_square() {
  const std::array<float, 4> input = {-2.0F, -0.5F, 0.5F, 2.0F};
  std::array<float, 4> output{};
  emel::kernel::Kernel kernel;
  emel::kernel::event::op_sqr request{};
  request.src0 = view(input.data(), {4, 1, 1, 1}, {4, 16, 64, 64});
  request.dst = view_mut(output.data(), {4, 1, 1, 1}, {4, 16, 64, 64});
  const bool status = kernel.process_event(request);
  std::cout << "case=dense_square status=" << (status ? "ok" : "error")
            << " output_bits=" << bits(output, {4, 1, 1, 1}, {4, 16, 64, 64})
            << '\n';
}

void run_strided_square() {
  const std::array<float, 5> input = {3.0F, 99.0F, -4.0F, 99.0F, 0.5F};
  std::array<float, 5> output{};
  emel::kernel::Kernel kernel;
  emel::kernel::event::op_sqr request{};
  request.src0 = view(input.data(), {3, 1, 1, 1}, {8, 24, 72, 72});
  request.dst = view_mut(output.data(), {3, 1, 1, 1}, {8, 24, 72, 72});
  const bool status = kernel.process_event(request);
  std::cout << "case=strided_square status=" << (status ? "ok" : "error")
            << " output_bits=" << bits(output, {3, 1, 1, 1}, {8, 24, 72, 72})
            << '\n';
}

void run_count_equal_different_dimensions() {
  const std::array<float, 4> input = {1.0F, 2.0F, 3.0F, 4.0F};
  std::array<float, 4> output{};
  emel::kernel::Kernel kernel;
  emel::kernel::event::op_sqr request{};
  request.src0 = view(input.data(), {2, 2, 1, 1}, {4, 8, 16, 16});
  request.dst = view_mut(output.data(), {4, 1, 1, 1}, {4, 16, 64, 64});
  const bool status = kernel.process_event(request);
  std::cout << "case=count_equal_different_dimensions status="
            << (status ? "ok" : "error")
            << " output_bits=" << bits(output, {4, 1, 1, 1}, {4, 16, 64, 64})
            << '\n';
}

void run_implicit_contiguous_square() {
  const std::array<float, 4> input = {1.0F, 2.0F, 3.0F, 4.0F};
  std::array<float, 4> output{};
  emel::kernel::Kernel kernel;
  emel::kernel::event::op_sqr request{};
  request.src0 = view(input.data(), {4, 1, 1, 1}, {0, 0, 0, 0});
  request.dst = view_mut(output.data(), {4, 1, 1, 1}, {0, 0, 0, 0});
  const bool status = kernel.process_event(request);
  std::cout << "case=implicit_contiguous_square status="
            << (status ? "ok" : "error")
            << " output_bits=" << bits(output, {4, 1, 1, 1}, {0, 0, 0, 0})
            << '\n';
}

void run_square_root() {
  const std::array<float, 4> input = {0.0F, 1.0F, 4.0F, 9.0F};
  std::array<float, 4> output{};
  emel::kernel::Kernel kernel;
  emel::kernel::event::op_sqrt request{};
  request.src0 = view(input.data(), {4, 1, 1, 1}, {4, 16, 64, 64});
  request.dst = view_mut(output.data(), {4, 1, 1, 1}, {4, 16, 64, 64});
  const bool status = kernel.process_event(request);
  std::cout << "case=dense_square_root status=" << (status ? "ok" : "error")
            << " output_bits=" << bits(output, {4, 1, 1, 1}, {4, 16, 64, 64})
            << '\n';
}

void run_rejection() {
  const std::array<float, 4> input = {1.0F, 2.0F, 3.0F, 4.0F};
  std::array<float, 3> output = {9.0F, 9.0F, 9.0F};
  emel::kernel::Kernel kernel;
  emel::kernel::event::op_sqr request{};
  request.src0 = view(input.data(), {4, 1, 1, 1}, {4, 16, 64, 64});
  request.dst = view_mut(output.data(), {3, 1, 1, 1}, {4, 12, 36, 36});
  const bool status = kernel.process_event(request);
  std::cout << "case=shape_rejection status=" << (status ? "ok" : "error")
            << " output_bits=" << bits(output, {3, 1, 1, 1}, {4, 12, 36, 36})
            << '\n';
}

}  // namespace

int main() {
  print_header();
  run_square();
  run_strided_square();
  run_count_equal_different_dimensions();
  run_implicit_contiguous_square();
  run_square_root();
  run_rejection();
  return 0;
}
