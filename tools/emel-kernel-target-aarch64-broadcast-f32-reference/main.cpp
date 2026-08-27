#include <array>
#include <bit>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>

#include "emel/kernel/aarch64/sm.hpp"

namespace {

constexpr char k_commit[] = "843a117386ef17dc5a50549bbfc821074c2141d6";
constexpr char k_events_blob[] = "4b3f02fa5ff9c5071d1fc83938cef1b22b4658b9";
constexpr char k_detail_blob[] = "c8a82643eabfe8f2d7883e655955f455794511b0";
constexpr char k_guards_blob[] = "c25714566ec9a02679daef85089544575123408e";
constexpr char k_actions_blob[] = "267d4f74e6e7498155c8535920322ffef2c02fb6";
constexpr char k_sm_blob[] = "865a9cc6ba6115382ed043c464f3d62bcd851357";
constexpr float k_sentinel = std::bit_cast<float>(std::uint32_t{0x7fc00001});

template <std::size_t N>
void write_bits(const std::array<float, N> &values) {
  for (std::size_t index = 0; index < N; ++index) {
    if (index != 0) std::cout << ',';
    std::cout << std::hex << std::setw(8) << std::setfill('0')
              << std::bit_cast<std::uint32_t>(values[index]) << std::dec;
  }
}

emel::kernel::event::tensor_view view(const float *data,
                                      std::array<std::uint64_t, 4> ne,
                                      std::array<std::uint64_t, 4> nb) {
  return {.data = data,
          .type = emel::kernel::event::dtype::f32,
          .ne = ne,
          .nb = nb};
}

emel::kernel::event::tensor_view_mut view_mut(float *data,
                                              std::array<std::uint64_t, 4> ne,
                                              std::array<std::uint64_t, 4> nb) {
  return {.data = data,
          .type = emel::kernel::event::dtype::f32,
          .ne = ne,
          .nb = nb};
}

template <std::size_t N>
void emit_positive(const char *name, emel::kernel::aarch64::sm &kernel,
                   auto request, const std::array<float, N> &output) {
  if (!kernel.process_event(request)) {
    std::cerr << "target broadcast positive request rejected\n";
    std::exit(1);
  }
  std::cout << "case=" << name << " status=ok output_bits=";
  write_bits(output);
  std::cout << '\n';
}

template <std::size_t N>
void emit_rejection(const char *name, emel::kernel::aarch64::sm &kernel,
                    auto request, const std::array<float, N> &output) {
  if (kernel.process_event(request)) {
    std::cerr << "target broadcast invalid request accepted\n";
    std::exit(1);
  }
  std::cout << "case=" << name << " status=reject error=InvalidShape output_bits=";
  write_bits(output);
  std::cout << '\n';
}

void print_header() {
  std::cout << "kernel-target-aarch64-broadcast-f32-live/v1\n"
            << "source_repository=stateforward/emel.cpp\n"
            << "source_commit=" << k_commit << '\n'
            << "source_kernel_events_blob=" << k_events_blob << '\n'
            << "source_kernel_detail_blob=" << k_detail_blob << '\n'
            << "source_kernel_aarch64_guards_blob=" << k_guards_blob << '\n'
            << "source_kernel_aarch64_actions_blob=" << k_actions_blob << '\n'
            << "source_kernel_aarch64_sm_blob=" << k_sm_blob << '\n'
            << "target_arch=aarch64\n"
            << "scope=target_router_binary_f32_row_broadcast_add_mul_dense\n"
            << "execution=split_pinned_aarch64_sm_and_public_target_router\n";
}

}  // namespace

int main() {
  constexpr std::array<std::uint64_t, 4> source_ne = {3, 2, 1, 1};
  constexpr std::array<std::uint64_t, 4> row_ne = {3, 1, 1, 1};
  constexpr std::array<std::uint64_t, 4> dense_source_nb = {4, 12, 24, 24};
  constexpr std::array<std::uint64_t, 4> dense_row_nb = {4, 12, 12, 12};
  const std::array<float, 6> source = {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F};
  const std::array<float, 3> row = {10.0F, 20.0F, 30.0F};
  print_header();
  emel::kernel::aarch64::sm kernel;

  std::array<float, 6> output{};
  {
    emel::kernel::event::op_add request{};
    request.src0 = view(source.data(), source_ne, dense_source_nb);
    request.src1 = view(row.data(), row_ne, dense_row_nb);
    request.dst = view_mut(output.data(), source_ne, dense_source_nb);
    emit_positive("row_add", kernel, request, output);
  }
  {
    emel::kernel::event::op_mul request{};
    request.src0 = view(source.data(), source_ne, dense_source_nb);
    request.src1 = view(row.data(), row_ne, dense_row_nb);
    request.dst = view_mut(output.data(), source_ne, dense_source_nb);
    emit_positive("row_mul", kernel, request, output);
  }

  std::array<float, 6> sentinel_output{};
  sentinel_output.fill(k_sentinel);
  constexpr std::array<std::uint64_t, 4> invalid_row_ne = {0, 1, 1, 1};
  constexpr std::array<std::uint64_t, 4> invalid_row_nb = {4, 0, 0, 0};
  emel::kernel::event::op_add invalid{};
  invalid.src0 = view(source.data(), source_ne, dense_source_nb);
  invalid.src1 = view(row.data(), invalid_row_ne, invalid_row_nb);
  invalid.dst = view_mut(sentinel_output.data(), source_ne, dense_source_nb);
  emit_rejection("invalid_shape", kernel, invalid, sentinel_output);
}
