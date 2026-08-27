#include <array>
#include <chrono>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <string>

#include "emel/kernel/events.hpp"
#include "emel/kernel/sm.hpp"

namespace {

constexpr char k_source_commit[] =
    "843a117386ef17dc5a50549bbfc821074c2141d6";

void write_bits(const float *values, std::size_t count) {
  for (std::size_t index = 0; index < count; ++index) {
    if (index != 0) {
      std::cout << ',';
    }
    std::uint32_t bits = 0;
    std::memcpy(&bits, values + index, sizeof(bits));
    std::cout << std::hex << std::setw(8) << std::setfill('0') << bits
              << std::dec;
  }
}

#if defined(__clang__) || defined(__GNUC__)
__attribute__((noinline))
#endif
bool run_rows(emel::kernel::Kernel &kernel, float *destination) {
  using emel::kernel::event::dtype;
  using emel::kernel::event::op_get_rows;
  using emel::kernel::event::tensor_view;
  using emel::kernel::event::tensor_view_mut;

  constexpr float source[] = {1.0F, 2.0F, 3.0F, 4.0F};
  constexpr std::int32_t indices[] = {1, 0};
  op_get_rows request{};
  request.src0 = tensor_view{
      source, dtype::f32, {2, 2, 1, 1}, {4, 8, 16, 16}};
  request.src1 = tensor_view{
      indices, dtype::i32, {2, 1, 1, 1}, {4, 8, 8, 8}};
  request.dst = tensor_view_mut{
      destination, dtype::f32, {2, 2, 1, 1}, {4, 8, 16, 16}};
  return kernel.process_event(request);
}

}  // namespace

void benchmark() {
  constexpr std::size_t kIterations = 10000;
  constexpr std::size_t kWarmup = 1000;
  float destination[4] = {};
  emel::kernel::Kernel kernel;
  volatile float benchmark_sink = 0.0F;
  for (std::size_t index = 0; index < kWarmup; ++index) {
    if (!run_rows(kernel, destination)) {
      std::cerr << "get_rows benchmark warmup failed\n";
      std::exit(1);
    }
  }
  const auto start = std::chrono::steady_clock::now();
  for (std::size_t index = 0; index < kIterations; ++index) {
    if (!run_rows(kernel, destination)) {
      std::cerr << "get_rows benchmark failed\n";
      std::exit(1);
    }
    benchmark_sink += destination[index & 3U];
  }
  const auto elapsed = std::chrono::duration<double, std::nano>(
      std::chrono::steady_clock::now() - start).count();
  if (destination[0] != 3.0F || destination[1] != 4.0F ||
      destination[2] != 1.0F || destination[3] != 2.0F) {
    std::cerr << "get_rows benchmark output validation failed\n";
    std::exit(1);
  }
  if (benchmark_sink == 0.0F) {
    std::cerr << "get_rows benchmark sink validation failed\n";
    std::exit(1);
  }
  std::cout << "case=op_get_rows_f32 cpp_ns_per_dispatch="
            << (elapsed / static_cast<double>(kIterations))
            << " output=4\n";
}

int main(int argc, char **argv) {
  if (argc > 1 && std::string(argv[1]) == "--benchmark") {
    benchmark();
    return 0;
  }
  float destination[4] = {};
  emel::kernel::Kernel kernel;
  if (!run_rows(kernel, destination)) {
    std::cerr << "get_rows reference execution failed\n";
    return 1;
  }
  std::cout << "kernel-get-rows-parity/v1\n"
            << "source_repository=stateforward/emel.cpp\n"
            << "source_commit=" << k_source_commit << '\n'
            << "source_kernel_events_blob=4b3f02fa5ff9c5071d1fc83938cef1b22b4658b9\n"
            << "source_kernel_detail_blob=c8a82643eabfe8f2d7883e655955f455794511b0\n"
            << "source_kernel_x86_sm_blob=0b4d635ebbd0fbd52dbca8a2345547fb571205c8\n"
            << "scope=f32_public_root_get_rows_dense_rank4\n"
            << "case=rows status=ok output_bits=";
  write_bits(destination, 4);
  std::cout << '\n';
}
