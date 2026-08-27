#include <algorithm>
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
    std::cerr << "target dup positive request rejected\n";
    std::exit(1);
  }
  std::cout << "case=" << name << " status=ok output_bits=";
  write_bits(output);
  std::cout << '\n';
}

void emit_zero_rejection(const char *name, emel::kernel::aarch64::sm &kernel,
                         auto request) {
  if (kernel.process_event(request)) {
    std::cerr << "target dup zero-count request accepted\n";
    std::exit(1);
  }
  std::cout << "case=" << name
            << " status=reject error=InvalidShape output_bits=\n";
}

template <std::size_t N>
void emit_rejection(const char *name, emel::kernel::aarch64::sm &kernel,
                    auto request, const std::array<float, N> &output) {
  if (kernel.process_event(request)) {
    std::cerr << "target dup invalid request accepted\n";
    std::exit(1);
  }
  std::cout << "case=" << name << " status=reject error=InvalidShape output_bits=";
  write_bits(output);
  std::cout << '\n';
}

void audit_malformed_metadata(emel::kernel::aarch64::sm &kernel,
                              const float *input, float *output) {
  constexpr std::array<std::uint64_t, 4> ne = {9, 1, 1, 1};
  constexpr std::array<std::uint64_t, 4> malformed_nb = {8, 36, 36, 36};
  constexpr std::array<std::uint64_t, 4> dense_nb = {4, 36, 36, 36};
  emel::kernel::event::op_dup request{};
  request.src0 = view(input, ne, malformed_nb);
  request.dst = view_mut(output, ne, dense_nb);
  const bool accepted = kernel.process_event(request);
  const auto sentinel = std::bit_cast<std::uint32_t>(k_sentinel);
  bool preserved = true;
  for (std::size_t index = 0; index < 9; ++index) {
    if (std::bit_cast<std::uint32_t>(output[index]) != sentinel) {
      preserved = false;
    }
  }
  std::cerr << "malformed_metadata status=" << (accepted ? "accepted" : "reject")
            << " output_preserved=" << (preserved ? "true" : "false") << '\n';
}

void print_header() {
  std::cout << "kernel-target-aarch64-dup-f32-live/v1\n"
            << "source_repository=stateforward/emel.cpp\n"
            << "source_commit=" << k_commit << '\n'
            << "source_kernel_events_blob=" << k_events_blob << '\n'
            << "source_kernel_detail_blob=" << k_detail_blob << '\n'
            << "source_kernel_aarch64_guards_blob=" << k_guards_blob << '\n'
            << "source_kernel_aarch64_actions_blob=" << k_actions_blob << '\n'
            << "source_kernel_aarch64_sm_blob=" << k_sm_blob << '\n'
            << "target_arch=aarch64\n"
            << "scope=target_router_dup_f32_dense_body_tail_and_rejection\n"
            << "execution=split_pinned_aarch64_sm_and_public_target_router\n";
}

}  // namespace

int main() {
  constexpr std::array<std::uint64_t, 4> ne = {9, 1, 1, 1};
  constexpr std::array<std::uint64_t, 4> dense_nb = {4, 36, 36, 36};
  const std::array<float, 9> input = {0.0F, -1.5F, 2.25F, 4.0F, -8.0F,
                                      16.0F, 32.0F, -64.0F, 128.0F};
  print_header();
  emel::kernel::aarch64::sm kernel;
  std::array<float, 9> output{};
  emel::kernel::event::op_dup request{};
  request.src0 = view(input.data(), ne, dense_nb);
  request.dst = view_mut(output.data(), ne, dense_nb);
  emit_positive("dup", kernel, request, output);

  std::array<float, 18> malformed_input{};
  std::copy(input.begin(), input.end(), malformed_input.begin());
  std::array<float, 9> sentinel_output{};
  sentinel_output.fill(k_sentinel);
  audit_malformed_metadata(kernel, malformed_input.data(), sentinel_output.data());

  std::array<float, 1> zero_storage = {k_sentinel};
  emel::kernel::event::op_dup zero{};
  zero.src0 = view(zero_storage.data(), {0, 1, 1, 1}, {4, 0, 0, 0});
  zero.dst = view_mut(zero_storage.data(), {0, 1, 1, 1}, {4, 0, 0, 0});
  emit_zero_rejection("zero_count", kernel, zero);

  const std::array<float, 3> short_input = {1.0F, 2.0F, 3.0F};
  std::array<float, 2> invalid_output = {k_sentinel, k_sentinel};
  emel::kernel::event::op_dup invalid{};
  invalid.src0 = view(short_input.data(), {3, 1, 1, 1}, {4, 12, 12, 12});
  invalid.dst = view_mut(invalid_output.data(), {2, 1, 1, 1}, {4, 8, 8, 8});
  emit_rejection("invalid_shape", kernel, invalid, invalid_output);
}
