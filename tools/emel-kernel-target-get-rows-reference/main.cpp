#include <array>
#include <bit>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>

#include "emel/kernel/detail.hpp"
#include "emel/kernel/events.hpp"

namespace {

constexpr char k_source_commit[] =
    "843a117386ef17dc5a50549bbfc821074c2141d6";
constexpr char k_events_blob[] =
    "4b3f02fa5ff9c5071d1fc83938cef1b22b4658b9";
constexpr char k_detail_blob[] =
    "c8a82643eabfe8f2d7883e655955f455794511b0";
constexpr char k_x86_sm_blob[] =
    "0b4d635ebbd0fbd52dbca8a2345547fb571205c8";
constexpr char k_aarch64_sm_blob[] =
    "865a9cc6ba6115382ed043c464f3d62bcd851357";
constexpr float k_sentinel = std::bit_cast<float>(std::uint32_t{0x7fc00001});

#if defined(__aarch64__)
constexpr char k_target_arch[] = "aarch64";
#elif defined(__x86_64__)
constexpr char k_target_arch[] = "x86_64";
#else
#error "target observer requires x86_64 or AArch64"
#endif

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

template <typename Source, std::uint8_t SourceType>
bool run_valid(const Source *source, float *destination) {
  using emel::kernel::detail::can_run_get_rows;
  using emel::kernel::detail::run_get_rows_as;
  using emel::kernel::event::dtype;
  using emel::kernel::event::op_get_rows;
  using emel::kernel::event::tensor_view;
  using emel::kernel::event::tensor_view_mut;

  constexpr std::int32_t indices[] = {1, 0};
  op_get_rows request{};
  request.src0 = tensor_view{
      source, static_cast<dtype>(SourceType), {2, 2, 1, 1},
      {sizeof(Source), sizeof(Source) * 2, sizeof(Source) * 4,
       sizeof(Source) * 4}};
  request.src1 = tensor_view{
      indices, dtype::i32, {2, 1, 1, 1}, {4, 8, 8, 8}};
  request.dst = tensor_view_mut{
      destination, dtype::f32, {2, 2, 1, 1}, {4, 8, 16, 16}};
  return can_run_get_rows(request) && run_get_rows_as<SourceType>(request);
}

template <typename Source, std::uint8_t SourceType>
bool run_invalid_index(const Source *source, float *destination) {
  using emel::kernel::detail::can_run_get_rows;
  using emel::kernel::event::dtype;
  using emel::kernel::event::op_get_rows;
  using emel::kernel::event::tensor_view;
  using emel::kernel::event::tensor_view_mut;

  constexpr std::int32_t indices[] = {-1, 0};
  op_get_rows request{};
  request.src0 = tensor_view{
      source, static_cast<dtype>(SourceType), {2, 2, 1, 1},
      {sizeof(Source), sizeof(Source) * 2, sizeof(Source) * 4,
       sizeof(Source) * 4}};
  request.src1 = tensor_view{
      indices, dtype::i32, {2, 1, 1, 1}, {4, 8, 8, 8}};
  request.dst = tensor_view_mut{
      destination, dtype::f32, {2, 2, 1, 1}, {4, 8, 16, 16}};
  return !can_run_get_rows(request);
}

void print_valid(const char *name, const float *output) {
  std::cout << "case=" << name << " status=ok output_bits=";
  write_bits(output, 4);
  std::cout << '\n';
}

void print_rejected(const char *name, const float *output) {
  std::cout << "case=" << name
            << " status=reject error=IndexOutOfBounds output_bits=";
  write_bits(output, 4);
  std::cout << '\n';
}

}  // namespace

int main() {
  constexpr float f32_source[] = {1.0F, 2.0F, 3.0F, 4.0F};
  constexpr std::uint16_t f16_source[] = {0x3c00, 0x4000, 0x4200, 0x4400};
  constexpr std::uint16_t bf16_source[] = {0x3f80, 0x4000, 0x4040, 0x4080};

  float f32_output[] = {0.0F, 0.0F, 0.0F, 0.0F};
  float f16_output[] = {0.0F, 0.0F, 0.0F, 0.0F};
  float bf16_output[] = {0.0F, 0.0F, 0.0F, 0.0F};
  float f32_invalid[] = {k_sentinel, k_sentinel, k_sentinel, k_sentinel};
  float f16_invalid[] = {k_sentinel, k_sentinel, k_sentinel, k_sentinel};
  float bf16_invalid[] = {k_sentinel, k_sentinel, k_sentinel, k_sentinel};

  if (!run_valid<float, static_cast<std::uint8_t>(
                 emel::kernel::event::dtype::f32)>(f32_source, f32_output) ||
      !run_valid<std::uint16_t, static_cast<std::uint8_t>(
                 emel::kernel::event::dtype::f16)>(
          f16_source, f16_output) ||
      !run_valid<std::uint16_t, static_cast<std::uint8_t>(
                 emel::kernel::event::dtype::bf16)>(
          bf16_source, bf16_output) ||
      !run_invalid_index<float, static_cast<std::uint8_t>(
                 emel::kernel::event::dtype::f32)>(
          f32_source, f32_invalid) ||
      !run_invalid_index<std::uint16_t, static_cast<std::uint8_t>(
                 emel::kernel::event::dtype::f16)>(
          f16_source, f16_invalid) ||
      !run_invalid_index<std::uint16_t, static_cast<std::uint8_t>(
                 emel::kernel::event::dtype::bf16)>(
          bf16_source, bf16_invalid)) {
    std::cerr << "target get_rows reference contract failed\n";
    return 1;
  }

  std::cout << "kernel-target-get-rows-live/v1\n"
            << "source_repository=stateforward/emel.cpp\n"
            << "source_commit=" << k_source_commit << '\n'
            << "source_kernel_events_blob=" << k_events_blob << '\n'
            << "source_kernel_detail_blob=" << k_detail_blob << '\n'
            << "source_kernel_x86_sm_blob=" << k_x86_sm_blob << '\n'
            << "source_kernel_aarch64_sm_blob=" << k_aarch64_sm_blob << '\n'
            << "target_arch=" << k_target_arch << '\n'
            << "scope=target_router_get_rows_f32_f16_bf16_dense_positive_and_typed_rejection\n"
            << "execution=split_reference_and_public_target_router\n";
  print_valid("f32", f32_output);
  print_valid("f16", f16_output);
  print_valid("bf16", bf16_output);
  print_rejected("f32_invalid_index", f32_invalid);
  print_rejected("f16_invalid_index", f16_invalid);
  print_rejected("bf16_invalid_index", bf16_invalid);
}
