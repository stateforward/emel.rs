#include <array>
#include <bit>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>

#include "emel/kernel/aarch64/sm.hpp"

namespace {

constexpr char k_source_commit[] =
    "843a117386ef17dc5a50549bbfc821074c2141d6";
constexpr char k_events_blob[] =
    "4b3f02fa5ff9c5071d1fc83938cef1b22b4658b9";
constexpr char k_detail_blob[] =
    "c8a82643eabfe8f2d7883e655955f455794511b0";
constexpr char k_guards_blob[] =
    "c25714566ec9a02679daef85089544575123408e";
constexpr char k_actions_blob[] =
    "267d4f74e6e7498155c8535920322ffef2c02fb6";
constexpr char k_sm_blob[] =
    "865a9cc6ba6115382ed043c464f3d62bcd851357";
constexpr float k_sentinel = std::bit_cast<float>(std::uint32_t{0x7fc00001});

template <std::size_t N>
void write_bits(const std::array<float, N> &values) {
  for (std::size_t index = 0; index < N; ++index) {
    if (index != 0) {
      std::cout << ',';
    }
    std::cout << std::hex << std::setw(8) << std::setfill('0')
              << std::bit_cast<std::uint32_t>(values[index]) << std::dec;
  }
}

emel::kernel::event::tensor_view view(const float *data, const std::uint64_t count,
                                      const emel::kernel::event::dtype type) {
  return {.data = data,
          .type = type,
          .ne = {count, 1, 1, 1},
          .nb = {4, count * 4, count * 4, count * 4}};
}

emel::kernel::event::tensor_view_mut view_mut(float *data,
                                              const std::uint64_t count) {
  return {.data = data,
          .type = emel::kernel::event::dtype::f32,
          .ne = {count, 1, 1, 1},
          .nb = {4, count * 4, count * 4, count * 4}};
}

template <typename Request, std::size_t N>
void emit_positive(const char *name, emel::kernel::aarch64::sm &kernel,
                   Request request, std::array<float, N> &output) {
  const bool accepted = kernel.process_event(request);
  if (!accepted) {
    std::cerr << "target binary positive request rejected\n";
    std::exit(1);
  }
  std::cout << "case=" << name << " status=ok output_bits=";
  write_bits(output);
  std::cout << '\n';
}

template <typename Request, std::size_t N>
void emit_rejection(const char *name, emel::kernel::aarch64::sm &kernel,
                    Request request, const std::array<float, N> &output) {
  if (kernel.process_event(request)) {
    std::cerr << "target binary invalid request accepted\n";
    std::exit(1);
  }
  std::cout << "case=" << name << " status=reject error=InvalidShape output_bits=";
  write_bits(output);
  std::cout << '\n';
}

void print_header() {
  std::cout << "kernel-target-aarch64-binary-f32-live/v1\n"
            << "source_repository=stateforward/emel.cpp\n"
            << "source_commit=" << k_source_commit << '\n'
            << "source_kernel_events_blob=" << k_events_blob << '\n'
            << "source_kernel_detail_blob=" << k_detail_blob << '\n'
            << "source_kernel_aarch64_guards_blob=" << k_guards_blob << '\n'
            << "source_kernel_aarch64_actions_blob=" << k_actions_blob << '\n'
            << "source_kernel_aarch64_sm_blob=" << k_sm_blob << '\n'
            << "target_arch=aarch64\n"
            << "scope=target_router_binary_f32_add_sub_mul_div_dense_equal_length_and_rejection\n"
            << "execution=split_pinned_aarch64_sm_and_public_target_router\n";
}

}  // namespace

int main() {
  const std::array<float, 9> lhs = {8.0F, -9.0F, 6.0F, 4.0F, -2.0F,
                                    3.0F, 7.0F, 11.0F, 5.0F};
  const std::array<float, 9> rhs = {2.0F, 3.0F, -2.0F, 4.0F, 2.0F,
                                    -3.0F, 7.0F, 11.0F, 5.0F};
  print_header();
  emel::kernel::aarch64::sm kernel;
  std::array<float, 9> output{};

  {
    emel::kernel::event::op_add request{};
    request.src0 = view(lhs.data(), lhs.size(), emel::kernel::event::dtype::f32);
    request.src1 = view(rhs.data(), rhs.size(), emel::kernel::event::dtype::f32);
    request.dst = view_mut(output.data(), output.size());
    emit_positive("add", kernel, request, output);
  }
  {
    emel::kernel::event::op_sub request{};
    request.src0 = view(lhs.data(), lhs.size(), emel::kernel::event::dtype::f32);
    request.src1 = view(rhs.data(), rhs.size(), emel::kernel::event::dtype::f32);
    request.dst = view_mut(output.data(), output.size());
    emit_positive("sub", kernel, request, output);
  }
  {
    emel::kernel::event::op_mul request{};
    request.src0 = view(lhs.data(), lhs.size(), emel::kernel::event::dtype::f32);
    request.src1 = view(rhs.data(), rhs.size(), emel::kernel::event::dtype::f32);
    request.dst = view_mut(output.data(), output.size());
    emit_positive("mul", kernel, request, output);
  }
  {
    emel::kernel::event::op_div request{};
    request.src0 = view(lhs.data(), lhs.size(), emel::kernel::event::dtype::f32);
    request.src1 = view(rhs.data(), rhs.size(), emel::kernel::event::dtype::f32);
    request.dst = view_mut(output.data(), output.size());
    emit_positive("div", kernel, request, output);
  }

  std::array<float, 9> sentinel_output{};
  sentinel_output.fill(k_sentinel);
  const std::array<float, 1> short_lhs = {1.0F};
  const std::array<float, 2> short_rhs = {2.0F, 3.0F};
  emel::kernel::event::op_add invalid{};
  invalid.src0 = view(short_lhs.data(), short_lhs.size(),
                      emel::kernel::event::dtype::f32);
  invalid.src1 = view(short_rhs.data(), short_rhs.size(),
                      emel::kernel::event::dtype::f32);
  invalid.dst = view_mut(sentinel_output.data(), sentinel_output.size());
  emit_rejection("invalid_shape", kernel, invalid, sentinel_output);

}
