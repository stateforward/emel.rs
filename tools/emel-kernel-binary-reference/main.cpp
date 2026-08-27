#include <array>
#include <bit>
#include <cstdint>
#include <iostream>
#include <string>

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
    const std::uint32_t word = std::bit_cast<std::uint32_t>(value);
    for (int shift = 28; shift >= 0; shift -= 4) {
      result.push_back("0123456789abcdef"[(word >> shift) & 0x0f]);
    }
  }
  return result;
}

void print_header() {
  std::cout << "kernel-binary-parity/v1\n"
            << "source_repository=stateforward/emel.cpp\n"
            << "source_commit=" << k_source_commit << '\n'
            << "source_sml_commit=49207123cd3f39767764bae774932cb48623f92f\n"
            << "source_kernel_events_blob=4b3f02fa5ff9c5071d1fc83938cef1b22b4658b9\n"
            << "source_kernel_detail_blob=c8a82643eabfe8f2d7883e655955f455794511b0\n"
            << "source_kernel_x86_sm_blob=0b4d635ebbd0fbd52dbca8a2345547fb571205c8\n"
            << "source_kernel_x86_actions_blob=d45558f5eb96950f43c16a09d768cb4f382d6d61\n"
            << "source_kernel_x86_guards_blob=cb3dac8253f8417c9b44acff1de414f6d0a3a3cf\n"
            << "source_detail_guard_span=3794-3797\n"
            << "source_detail_run_span=2366-2383\n"
            << "scope=portable_f32_equal_count_dense_or_validated_strided_implicit\n";
}

template <typename Request, std::size_t N>
void print_case(const char *name, const Request &request,
                const bool status, const std::array<float, N> &output,
                const std::array<std::uint64_t, 4> output_ne,
                const std::array<std::uint64_t, 4> output_nb) {
  std::cout << "case=" << name << " status=" << (status ? "ok" : "error")
            << " output_bits=" << bits(output, output_ne, output_nb) << '\n';
  (void)request;
}

void run_dense() {
  const std::array<float, 4> lhs = {1.0F, 2.0F, 3.0F, 4.0F};
  const std::array<float, 4> rhs = {10.0F, 20.0F, 30.0F, 40.0F};
  constexpr std::array<std::uint64_t, 4> ne = {4, 1, 1, 1};
  constexpr std::array<std::uint64_t, 4> nb = {4, 16, 64, 64};

  {
    std::array<float, 4> output{};
    emel::kernel::Kernel kernel;
    emel::kernel::event::op_add request{};
    request.src0 = view(lhs.data(), ne, nb);
    request.src1 = view(rhs.data(), ne, nb);
    request.dst = view_mut(output.data(), ne, nb);
    print_case("dense_add", request, kernel.process_event(request), output, ne,
               nb);
  }
  {
    std::array<float, 4> output{};
    emel::kernel::Kernel kernel;
    emel::kernel::event::op_sub request{};
    request.src0 = view(lhs.data(), ne, nb);
    request.src1 = view(rhs.data(), ne, nb);
    request.dst = view_mut(output.data(), ne, nb);
    print_case("dense_sub", request, kernel.process_event(request), output, ne,
               nb);
  }
  {
    std::array<float, 4> output{};
    emel::kernel::Kernel kernel;
    emel::kernel::event::op_mul request{};
    request.src0 = view(lhs.data(), ne, nb);
    request.src1 = view(rhs.data(), ne, nb);
    request.dst = view_mut(output.data(), ne, nb);
    print_case("dense_mul", request, kernel.process_event(request), output, ne,
               nb);
  }
  {
    std::array<float, 4> output{};
    emel::kernel::Kernel kernel;
    emel::kernel::event::op_div request{};
    request.src0 = view(lhs.data(), ne, nb);
    request.src1 = view(rhs.data(), ne, nb);
    request.dst = view_mut(output.data(), ne, nb);
    print_case("dense_div", request, kernel.process_event(request), output, ne,
               nb);
  }
}

void run_strided() {
  const std::array<float, 5> lhs = {1.0F, 99.0F, 2.0F, 99.0F, 3.0F};
  const std::array<float, 5> rhs = {10.0F, 99.0F, 20.0F, 99.0F, 30.0F};
  constexpr std::array<std::uint64_t, 4> ne = {3, 1, 1, 1};
  constexpr std::array<std::uint64_t, 4> nb = {8, 24, 72, 72};
  std::array<float, 5> output{};
  emel::kernel::Kernel kernel;
  emel::kernel::event::op_add request{};
  request.src0 = view(lhs.data(), ne, nb);
  request.src1 = view(rhs.data(), ne, nb);
  request.dst = view_mut(output.data(), ne, nb);
  print_case("strided_add", request, kernel.process_event(request), output, ne,
             nb);
}

void run_reshaped() {
  const std::array<float, 4> lhs = {1.0F, 2.0F, 3.0F, 4.0F};
  const std::array<float, 4> rhs = {10.0F, 20.0F, 30.0F, 40.0F};
  std::array<float, 4> output{};
  constexpr std::array<std::uint64_t, 4> lhs_ne = {2, 2, 1, 1};
  constexpr std::array<std::uint64_t, 4> lhs_nb = {4, 8, 16, 16};
  constexpr std::array<std::uint64_t, 4> output_ne = {4, 1, 1, 1};
  constexpr std::array<std::uint64_t, 4> output_nb = {4, 16, 64, 64};
  emel::kernel::Kernel kernel;
  emel::kernel::event::op_add request{};
  request.src0 = view(lhs.data(), lhs_ne, lhs_nb);
  request.src1 = view(rhs.data(), output_ne, output_nb);
  request.dst = view_mut(output.data(), output_ne, output_nb);
  print_case("count_equal_different_dimensions", request,
             kernel.process_event(request), output, output_ne, output_nb);
}

void run_implicit() {
  const std::array<float, 4> lhs = {1.0F, 2.0F, 3.0F, 4.0F};
  const std::array<float, 4> rhs = {10.0F, 20.0F, 30.0F, 40.0F};
  std::array<float, 4> output{};
  constexpr std::array<std::uint64_t, 4> ne = {4, 1, 1, 1};
  constexpr std::array<std::uint64_t, 4> nb = {0, 0, 0, 0};
  emel::kernel::Kernel kernel;
  emel::kernel::event::op_mul request{};
  request.src0 = view(lhs.data(), ne, nb);
  request.src1 = view(rhs.data(), ne, nb);
  request.dst = view_mut(output.data(), ne, nb);
  print_case("implicit_contiguous_mul", request, kernel.process_event(request),
             output, ne, nb);
}

void run_rejection() {
  const std::array<float, 4> lhs = {1.0F, 2.0F, 3.0F, 4.0F};
  const std::array<float, 4> rhs = {10.0F, 20.0F, 30.0F, 40.0F};
  std::array<float, 3> output = {9.0F, 9.0F, 9.0F};
  constexpr std::array<std::uint64_t, 4> input_ne = {4, 1, 1, 1};
  constexpr std::array<std::uint64_t, 4> input_nb = {4, 16, 64, 64};
  constexpr std::array<std::uint64_t, 4> output_ne = {3, 1, 1, 1};
  constexpr std::array<std::uint64_t, 4> output_nb = {4, 12, 36, 36};
  emel::kernel::Kernel kernel;
  emel::kernel::event::op_div request{};
  request.src0 = view(lhs.data(), input_ne, input_nb);
  request.src1 = view(rhs.data(), input_ne, input_nb);
  request.dst = view_mut(output.data(), output_ne, output_nb);
  print_case("count_mismatch_rejection", request, kernel.process_event(request),
             output, output_ne, output_nb);
}

}  // namespace

int main() {
  print_header();
  run_dense();
  run_strided();
  run_reshaped();
  run_implicit();
  run_rejection();
  return 0;
}
