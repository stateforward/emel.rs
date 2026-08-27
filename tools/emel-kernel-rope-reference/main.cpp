#include <array>
#include <bit>
#include <chrono>
#include <cstdlib>
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

emel::kernel::event::tensor_view positions(
    const std::int32_t *data, const std::array<std::uint64_t, 4> ne) {
  return {.data = data,
          .type = emel::kernel::event::dtype::i32,
          .ne = ne,
          .nb = {4, 4 * ne[0], 4 * ne[0] * ne[1],
                 4 * ne[0] * ne[1] * ne[2]}};
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
void set_params(Request &request, const std::int32_t mode) {
  const std::int32_t n_dims = 4;
  const float freq_base = 10'000.0F;
  const float freq_scale = 1.0F;
  const float ext_factor = 0.0F;
  const float attn_factor = 1.0F;
  std::memcpy(request.op_params.data() + 4, &n_dims, sizeof(n_dims));
  std::memcpy(request.op_params.data() + 8, &mode, sizeof(mode));
  std::memcpy(request.op_params.data() + 20, &freq_base, sizeof(freq_base));
  std::memcpy(request.op_params.data() + 24, &freq_scale, sizeof(freq_scale));
  std::memcpy(request.op_params.data() + 28, &ext_factor, sizeof(ext_factor));
  std::memcpy(request.op_params.data() + 32, &attn_factor, sizeof(attn_factor));
  request.op_params_size = 36;
}

template <class Request>
void write_case(emel::kernel::Kernel &kernel, const char *name, Request &request,
                float *output, const std::size_t count) {
  const bool accepted = kernel.process_event(request);
  std::cout << "case=" << name << " status="
            << (accepted ? "ok" : "rejected")
            << " output_bits=" << bits(output, count) << '\n';
}

void benchmark() {
  constexpr std::uint32_t k_iterations = 10'000;
  constexpr std::uint32_t k_warmup = 1'000;
  constexpr std::array<float, 4> input = {1.0F, 2.0F, 3.0F, 4.0F};
  constexpr std::array<std::int32_t, 1> position = {1};
  std::array<float, 4> output{};
  emel::kernel::Kernel kernel;
  volatile float sink = 0.0F;
  auto dispatch = [&] {
    auto request = emel::kernel::event::op_rope{};
    request.src0 = view(input.data(), {4, 1, 1, 1}, {4, 16, 16, 16});
    request.src1 = positions(position.data(), {1, 1, 1, 1});
    request.dst = view_mut(output.data(), {4, 1, 1, 1}, {4, 16, 16, 16});
    set_params(request, 0);
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
  std::cout << "case=op_rope_norm cpp_ns_per_dispatch="
            << elapsed.count() / static_cast<double>(k_iterations)
            << " output=4\n";
}

} // namespace

int main() {
  // The benchmark is a separate observer mode so normal parity output remains
  // deterministic and byte-comparable.
  if (std::getenv("EMEL_KERNEL_ROPE_BENCHMARK") != nullptr) {
    benchmark();
    return 0;
  }
  std::cout << "kernel-rope-parity/v2\n"
            << "source_repository=stateforward/emel.cpp\n"
            << "source_commit=" << k_source_commit << '\n'
            << "source_kernel_events_blob=4b3f02fa5ff9c5071d1fc83938cef1b22b4658b9\n"
            << "source_kernel_detail_blob=c8a82643eabfe8f2d7883e655955f455794511b0\n"
            << "source_kernel_x86_sm_blob=0b4d635ebbd0fbd52dbca8a2345547fb571205c8\n"
            << "source_kernel_x86_guards_blob=cb3dac8253f8417c9b44acff1de414f6d0a3a3cf\n"
            << "source_kernel_x86_actions_blob=d45558f5eb96950f43c16a09d768cb4f382d6d61\n"
            << "source_detail_guard_span=4650-4702\n"
            << "source_detail_rotation_span=4704-4761\n"
            << "source_detail_timestep_span=4763-4830\n"
            << "scope=op_rope_f32_norm_neox_timestep_positive_contiguous\n";

  constexpr std::array<float, 12> input = {
      1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F,
      7.0F, 8.0F, 9.0F, 10.0F, 11.0F, 12.0F};
  constexpr std::array<std::int32_t, 2> position_values = {1, 2};
  emel::kernel::Kernel kernel;

  std::array<float, 12> norm_output{};
  auto norm = emel::kernel::event::op_rope{};
  norm.src0 = view(input.data(), {6, 1, 2, 1}, {4, 24, 24, 48});
  norm.src1 = positions(position_values.data(), {2, 1, 1, 1});
  norm.dst = view_mut(norm_output.data(), {6, 1, 2, 1}, {4, 24, 24, 48});
  set_params(norm, 0);
  write_case(kernel, "norm", norm, norm_output.data(), norm_output.size());

  constexpr std::array<float, 4> short_input = {1.0F, 2.0F, 3.0F, 4.0F};
  constexpr std::array<std::int32_t, 1> one_position = {1};
  std::array<float, 4> neox_output{};
  auto neox = emel::kernel::event::op_rope{};
  neox.src0 = view(short_input.data(), {4, 1, 1, 1}, {4, 16, 16, 16});
  neox.src1 = positions(one_position.data(), {1, 1, 1, 1});
  neox.dst = view_mut(neox_output.data(), {4, 1, 1, 1}, {4, 16, 16, 16});
  set_params(neox, 2);
  write_case(kernel, "neox", neox, neox_output.data(), neox_output.size());

  std::array<float, 4> timestep_output{};
  auto timestep = emel::kernel::event::op_rope{};
  timestep.src0 = view(short_input.data(), {4, 1, 1, 1}, {4, 16, 16, 16});
  timestep.src1 = positions(one_position.data(), {1, 1, 1, 1});
  timestep.dst = view_mut(timestep_output.data(), {4, 1, 1, 1}, {4, 16, 16, 16});
  set_params(timestep, 4);
  write_case(kernel, "timestep", timestep, timestep_output.data(), timestep_output.size());

  std::array<float, 4> invalid_output = {9.0F, 9.0F, 9.0F, 9.0F};
  auto invalid = emel::kernel::event::op_rope{};
  invalid.src0 = view(short_input.data(), {4, 1, 1, 1}, {4, 16, 16, 16});
  invalid.src1 = positions(one_position.data(), {1, 1, 1, 1});
  invalid.dst = view_mut(invalid_output.data(), {4, 1, 1, 1}, {4, 16, 16, 16});
  set_params(invalid, 99);
  write_case(kernel, "invalid_mode", invalid, invalid_output.data(), invalid_output.size());
}
