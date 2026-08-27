#include <array>
#include <bit>
#include <cstdint>
#include <iomanip>
#include <iostream>

#include "emel/kernel/sm.hpp"

namespace {

constexpr char k_source_commit[] =
    "843a117386ef17dc5a50549bbfc821074c2141d6";
constexpr std::uint32_t k_sentinel = 0x7fc01234U;

emel::kernel::event::tensor_view view(
    const float *data, const emel::kernel::event::dtype type,
    const std::array<std::uint64_t, 4> ne,
    const std::array<std::uint64_t, 4> nb) {
  return {.data = data, .type = type, .ne = ne, .nb = nb};
}

emel::kernel::event::tensor_view_mut view_mut(
    float *data, const emel::kernel::event::dtype type,
    const std::array<std::uint64_t, 4> ne,
    const std::array<std::uint64_t, 4> nb) {
  return {.data = data, .type = type, .ne = ne, .nb = nb};
}

template <std::size_t N>
void print_case(const char *name, const bool status,
                const std::array<float, N> &output) {
  std::cout << "case=" << name << " status=" << (status ? "ok" : "error")
            << " output_bits=";
  for (std::size_t index = 0; index < N; ++index) {
    if (index != 0) std::cout << ',';
    std::cout << std::hex << std::setw(8) << std::setfill('0')
              << std::bit_cast<std::uint32_t>(output[index]);
  }
  std::cout << std::dec << '\n';
}

void run_vector() {
  constexpr std::array<float, 9> input = {
      0.0F, -1.5F, 2.25F, 4.0F, -8.0F, 16.0F, 32.0F, -64.0F,
      std::bit_cast<float>(0x80000000U)};
  std::array<float, 9> output{};
  emel::kernel::Kernel kernel;
  emel::kernel::event::op_dup request{};
  request.src0 = view(input.data(), emel::kernel::event::dtype::f32,
                      {9, 1, 1, 1}, {4, 36, 36, 36});
  request.dst = view_mut(output.data(), emel::kernel::event::dtype::f32,
                         {9, 1, 1, 1}, {4, 36, 36, 36});
  print_case("vector_tail", kernel.process_event(request), output);
}

void run_zero_count() {
  const std::array<float, 1> input = {std::bit_cast<float>(k_sentinel)};
  std::array<float, 1> output = input;
  emel::kernel::Kernel kernel;
  emel::kernel::event::op_dup request{};
  request.src0 = view(input.data(), emel::kernel::event::dtype::f32,
                      {0, 1, 1, 1}, {4, 0, 0, 0});
  request.dst = view_mut(output.data(), emel::kernel::event::dtype::f32,
                         {0, 1, 1, 1}, {4, 0, 0, 0});
  print_case("zero_count", kernel.process_event(request), output);
}

void run_invalid_metadata() {
  const std::array<float, 4> input = {1.0F, 2.0F, 3.0F, 4.0F};
  std::array<float, 4> output = {std::bit_cast<float>(k_sentinel),
                                 std::bit_cast<float>(k_sentinel),
                                 std::bit_cast<float>(k_sentinel),
                                 std::bit_cast<float>(k_sentinel)};
  emel::kernel::Kernel kernel;
  emel::kernel::event::op_dup request{};
  request.src0 = view(input.data(), emel::kernel::event::dtype::f32,
                      {4, 1, 1, 1}, {4, 16, 16, 16});
  request.dst = view_mut(output.data(), emel::kernel::event::dtype::unknown,
                         {4, 1, 1, 1}, {4, 16, 16, 16});
  print_case("invalid_metadata_unknown_dtype", kernel.process_event(request),
             output);
}

}  // namespace

int main() {
  std::cout << "kernel-aarch64-dup-f32-live/v1\n"
            << "source_repository=stateforward/emel.cpp\n"
            << "source_commit=" << k_source_commit << '\n'
            << "source_kernel_events_blob=4b3f02fa5ff9c5071d1fc83938cef1b22b4658b9\n"
            << "source_kernel_detail_blob=c8a82643eabfe8f2d7883e655955f455794511b0\n"
            << "source_kernel_aarch64_guards_blob=c25714566ec9a02679daef85089544575123408e\n"
            << "source_kernel_aarch64_actions_blob=267d4f74e6e7498155c8535920322ffef2c02fb6\n"
            << "source_kernel_aarch64_sm_blob=865a9cc6ba6115382ed043c464f3d62bcd851357\n"
            << "source_detail_span=src/emel/kernel/detail.hpp:1535-1555,1700-1748,1920-1926,3788-3791\n"
            << "source_guard_span=src/emel/kernel/aarch64/guards.hpp:320-329,805-824\n"
            << "source_action_span=src/emel/kernel/aarch64/actions.hpp:887-930,2217-2237,8447-8479\n"
            << "source_transition_span=src/emel/kernel/aarch64/sm.hpp:30-43\n"
            << "scope=dense_f32_dup_vector_tail_zero_count_metadata_rejection\n"
            << "metadata_semantics=cpp_tensor_view_rejects_unknown_dtype;rust_slice_api_unrepresentable\n";
  run_vector();
  run_zero_count();
  run_invalid_metadata();
  return 0;
}
