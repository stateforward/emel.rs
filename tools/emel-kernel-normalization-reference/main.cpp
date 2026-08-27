#include <array>
#include <bit>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>

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

template <class Request>
void set_epsilon(Request &request, const float epsilon) {
  std::memcpy(request.op_params.data(), &epsilon, sizeof(epsilon));
  request.op_params_size = sizeof(epsilon);
}

void write_header() {
  std::cout << "kernel-normalization-parity/v1\n"
            << "source_repository=stateforward/emel.cpp\n"
            << "source_commit=" << k_source_commit << '\n'
            << "source_kernel_events_blob=4b3f02fa5ff9c5071d1fc83938cef1b22b4658b9\n"
            << "source_kernel_detail_blob=c8a82643eabfe8f2d7883e655955f455794511b0\n"
            << "source_kernel_x86_sm_blob=0b4d635ebbd0fbd52dbca8a2345547fb571205c8\n"
            << "source_kernel_aarch64_sm_blob=865a9cc6ba6115382ed043c464f3d62bcd851357\n"
            << "source_detail_guard_span=4546-4564\n"
            << "source_detail_rms_run_span=4568-4603\n"
            << "source_detail_norm_run_span=4605-4648\n"
            << "scope=f32_norm_and_rms_norm_aligned_explicit_or_implicit_contiguous\n";
}

void write_cases() {
  constexpr std::array<float, 6> input = {1.0F, 2.0F, 4.0F,
                                           2.0F, 5.0F, 8.0F};
  emel::kernel::Kernel kernel;

  std::array<float, 6> rms_output{};
  auto rms = emel::kernel::event::op_rms_norm{};
  rms.src0 = view(input.data(), {3, 2, 1, 1}, {0, 0, 0, 0});
  rms.dst = view_mut(rms_output.data(), {3, 2, 1, 1}, {0, 0, 0, 0});
  set_epsilon(rms, 1.0F);
  const bool rms_status = kernel.process_event(rms);
  std::cout << "case=implicit_rms_norm status="
            << (rms_status ? "ok" : "rejected")
            << " output_bits=" << bits(rms_output.data(), rms_output.size())
            << '\n';

  constexpr std::array<float, 7> strided_input = {1.0F, 2.0F, 99.0F, 99.0F,
                                                   2.0F, 5.0F, 8.0F};
  std::array<float, 7> norm_output;
  norm_output.fill(9.0F);
  auto norm = emel::kernel::event::op_norm{};
  norm.src0 = view(strided_input.data(), {3, 2, 1, 1}, {4, 16, 32, 32});
  norm.dst = view_mut(norm_output.data(), {3, 2, 1, 1}, {4, 16, 32, 32});
  set_epsilon(norm, 1.0F);
  const bool norm_status = kernel.process_event(norm);
  std::cout << "case=strided_norm status="
            << (norm_status ? "ok" : "rejected")
            << " output_bits=" << bits(norm_output.data(), norm_output.size())
            << '\n';

  std::array<float, 1> empty_output = {9.0F};
  auto empty = emel::kernel::event::op_norm{};
  empty.src0 = view(input.data(), {3, 0, 1, 1}, {4, 12, 0, 0});
  empty.dst = view_mut(empty_output.data(), {3, 0, 1, 1}, {4, 12, 0, 0});
  set_epsilon(empty, 1.0F);
  const bool empty_status = kernel.process_event(empty);
  std::cout << "case=empty_outer_norm status="
            << (empty_status ? "ok" : "rejected")
            << " output_bits=" << bits(empty_output.data(), empty_output.size())
            << '\n';

  std::array<float, 6> invalid_output;
  invalid_output.fill(9.0F);
  auto invalid = emel::kernel::event::op_norm{};
  invalid.src0 = view(input.data(), {3, 2, 1, 1}, {0, 0, 0, 0});
  invalid.dst = view_mut(invalid_output.data(), {3, 2, 1, 1}, {0, 0, 0, 0});
  set_epsilon(invalid, std::numeric_limits<float>::quiet_NaN());
  const bool invalid_status = kernel.process_event(invalid);
  std::cout << "case=invalid_epsilon status="
            << (invalid_status ? "ok" : "rejected")
            << " output_bits=" << bits(invalid_output.data(), invalid_output.size())
            << '\n';
}

} // namespace

int main() {
  write_header();
  write_cases();
  return 0;
}
