#include <array>
#include <bit>
#include <chrono>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <sstream>

#include "emel/kernel/sm.hpp"

namespace {

constexpr char k_source_commit[] =
    "843a117386ef17dc5a50549bbfc821074c2141d6";

emel::kernel::event::tensor_view f32_view(
    const float *data, const std::array<std::uint64_t, 4> ne) {
  return {.data = data,
          .type = emel::kernel::event::dtype::f32,
          .ne = ne,
          .nb = {4, 4 * ne[0], 4 * ne[0] * ne[1],
                 4 * ne[0] * ne[1] * ne[2]}};
}

emel::kernel::event::tensor_view f16_view(
    const std::uint16_t *data, const std::array<std::uint64_t, 4> ne) {
  return {.data = data,
          .type = emel::kernel::event::dtype::f16,
          .ne = ne,
          .nb = {2, 2 * ne[0], 2 * ne[0] * ne[1],
                 2 * ne[0] * ne[1] * ne[2]}};
}

emel::kernel::event::tensor_view_mut f32_view_mut(
    float *data, const std::array<std::uint64_t, 4> ne) {
  return {.data = data,
          .type = emel::kernel::event::dtype::f32,
          .ne = ne,
          .nb = {4, 4 * ne[0], 4 * ne[0] * ne[1],
                 4 * ne[0] * ne[1] * ne[2]}};
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
void set_scale(Request &request, const float scale) {
  std::memcpy(request.op_params.data(), &scale, sizeof(scale));
  request.op_params_size = sizeof(scale);
}

void write_case(emel::kernel::Kernel &kernel, const char *name,
                emel::kernel::event::op_flash_attn_ext &request,
                const float *output, const std::size_t count) {
  const bool accepted = kernel.process_event(request);
  std::cout << "case=" << name << " status="
            << (accepted ? "ok" : "rejected") << " output_bits="
            << bits(output, count) << '\n';
}

void benchmark() {
  constexpr std::uint32_t k_iterations = 10'000;
  constexpr std::uint32_t k_warmup = 1'000;
  constexpr std::array<float, 2> query = {1.0F, 0.0F};
  constexpr std::array<std::uint16_t, 2> key = {0x3c00U, 0U};
  constexpr std::array<std::uint16_t, 2> value = {0x4000U, 0U};
  std::array<float, 2> output{};
  emel::kernel::Kernel kernel;
  volatile float sink = 0.0F;
  auto dispatch = [&] {
    auto request = emel::kernel::event::op_flash_attn_ext{};
    request.src0 = f32_view(query.data(), {2, 1, 1, 1});
    request.src1 = f16_view(key.data(), {2, 1, 1, 1});
    request.src2 = f16_view(value.data(), {2, 1, 1, 1});
    request.dst = f32_view_mut(output.data(), {2, 1, 1, 1});
    set_scale(request, 1.0F);
    if (!kernel.process_event(request)) {
      std::abort();
    }
    sink = output[0];
  };
  for (std::uint32_t index = 0; index < k_warmup; ++index) {
    dispatch();
  }
  const auto start = std::chrono::steady_clock::now();
  for (std::uint32_t index = 0; index < k_iterations; ++index) {
    dispatch();
  }
  const auto elapsed = std::chrono::duration<double, std::nano>(
      std::chrono::steady_clock::now() - start);
  if (sink == 0.0F) {
    std::abort();
  }
  std::cout << "case=op_flash_attn_ext_canonical cpp_ns_per_dispatch="
            << elapsed.count() / static_cast<double>(k_iterations)
            << " output=2\n";
}

} // namespace

int main(int argc, char **argv) {
  if (argc > 1 && std::string(argv[1]) == "--benchmark") {
    benchmark();
    return 0;
  }

  std::cout << "kernel-flash-attn-parity/v1\n"
            << "source_repository=stateforward/emel.cpp\n"
            << "source_commit=" << k_source_commit << '\n'
            << "source_sml_commit=49207123cd3f39767764bae774932cb48623f92f\n"
            << "source_kernel_events_blob=4b3f02fa5ff9c5071d1fc83938cef1b22b4658b9\n"
            << "source_kernel_detail_blob=c8a82643eabfe8f2d7883e655955f455794511b0\n"
            << "source_kernel_x86_sm_blob=0b4d635ebbd0fbd52dbca8a2345547fb571205c8\n"
            << "source_kernel_x86_guards_blob=cb3dac8253f8417c9b44acff1de414f6d0a3a3cf\n"
            << "source_kernel_x86_actions_blob=d45558f5eb96950f43c16a09d768cb4f382d6d61\n"
            << "source_detail_spans=170-185,2154-2214,4225-4248,4254-4327,4339-4389,5239-5311\n"
            << "scope=op_flash_attn_ext_f32_q_f32_dst_f16_k_f16_v_query_count_1_head_replication\n";

  constexpr std::array<float, 4> query = {1.0F, 0.0F, 1.0F, 0.0F};
  constexpr std::array<std::uint16_t, 8> key = {
      0x3c00U, 0U, 0U, 0U, 0U, 0x3c00U, 0U, 0U};
  constexpr std::array<std::uint16_t, 8> value = {
      0x4000U, 0x4400U, 0x4600U, 0x4800U,
      0x4000U, 0x4400U, 0x4600U, 0x4800U};
  std::array<float, 4> output{};
  emel::kernel::Kernel kernel;
  auto request = emel::kernel::event::op_flash_attn_ext{};
  request.src0 = f32_view(query.data(), {2, 1, 2, 1});
  request.src1 = f16_view(key.data(), {2, 2, 1, 1});
  request.src2 = f16_view(value.data(), {2, 2, 1, 1});
  request.dst = f32_view_mut(output.data(), {2, 1, 2, 1});
  set_scale(request, 1.0F);
  write_case(kernel, "canonical", request, output.data(), output.size());

  std::array<float, 4> invalid_output = {9.0F, 9.0F, 9.0F, 9.0F};
  auto invalid = emel::kernel::event::op_flash_attn_ext{};
  invalid.src0 = f32_view(query.data(), {2, 1, 2, 1});
  invalid.src1 = f16_view(key.data(), {2, 2, 1, 1});
  invalid.src2 = f16_view(value.data(), {2, 2, 1, 1});
  invalid.dst = f32_view_mut(invalid_output.data(), {2, 1, 2, 1});
  set_scale(invalid, 0.0F);
  write_case(kernel, "invalid_scale", invalid, invalid_output.data(),
             invalid_output.size());
}
