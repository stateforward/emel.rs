#include <array>
#include <bit>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <string>

#include "emel/kernel/detail.hpp"
#include "emel/kernel/events.hpp"

namespace {

constexpr char k_source_commit[] =
    "843a117386ef17dc5a50549bbfc821074c2141d6";
constexpr char k_events_blob[] =
    "4b3f02fa5ff9c5071d1fc83938cef1b22b4658b9";
constexpr char k_detail_blob[] =
    "c8a82643eabfe8f2d7883e655955f455794511b0";
constexpr char k_aarch64_sm_blob[] =
    "865a9cc6ba6115382ed043c464f3d62bcd851357";
constexpr float k_sentinel = std::bit_cast<float>(std::uint32_t{0x7fc00001});

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

template <std::uint8_t DType, std::size_t Bytes, std::size_t Columns>
emel::kernel::event::op_get_rows make_request(
    const std::array<std::uint8_t, Bytes> &source,
    const std::int32_t *indices, float *destination,
    const std::uint64_t source_stride_0,
    const std::uint64_t source_stride_1) {
  using emel::kernel::event::dtype;
  using emel::kernel::event::op_get_rows;
  using emel::kernel::event::tensor_view;
  using emel::kernel::event::tensor_view_mut;

  op_get_rows request{};
  request.src0 = tensor_view{
      source.data(), static_cast<dtype>(DType),
      {Columns, 1, 1, 1},
      {source_stride_0, source_stride_1, source_stride_1, source_stride_1}};
  request.src1 = tensor_view{
      indices, dtype::i32, {1, 1, 1, 1}, {4, 4, 4, 4}};
  request.dst = tensor_view_mut{
      destination, dtype::f32, {Columns, 1, 1, 1},
      {4, Columns * 4, Columns * 4, Columns * 4}};
  return request;
}

template <std::uint8_t DType, std::size_t Bytes, std::size_t Columns>
bool run_valid(const std::array<std::uint8_t, Bytes> &source,
               float *destination) {
  constexpr std::int32_t indices[] = {0};
  auto request = make_request<DType, Bytes, Columns>(
      source, indices, destination, 1, Bytes);
  return emel::kernel::detail::can_run_get_rows(request) &&
         emel::kernel::detail::run_get_rows_as<DType>(request);
}

template <std::uint8_t DType, std::size_t Bytes, std::size_t Columns>
bool run_invalid_index(const std::array<std::uint8_t, Bytes> &source,
                       float *destination) {
  constexpr std::int32_t indices[] = {-1};
  auto request = make_request<DType, Bytes, Columns>(
      source, indices, destination, 1, Bytes);
  return !emel::kernel::detail::can_run_get_rows(request);
}

template <std::uint8_t DType, std::size_t Bytes, std::size_t Columns>
bool run_invalid_layout(const std::array<std::uint8_t, Bytes> &source,
                        float *destination) {
  constexpr std::int32_t indices[] = {0};
  auto request = make_request<DType, Bytes, Columns>(
      source, indices, destination, 2, Bytes);
  return !emel::kernel::detail::can_run_get_rows(request);
}

template <std::size_t Count>
void print_ok(const char *name, const std::array<float, Count> &output) {
  std::cout << "case=" << name << " status=ok output_bits=";
  write_bits(output.data(), output.size());
  std::cout << '\n';
}

template <std::size_t Count>
void print_reject(const char *name, const char *error,
                 const std::array<float, Count> &output) {
  std::cout << "case=" << name << " status=reject error=" << error
            << " output_bits=";
  write_bits(output.data(), output.size());
  std::cout << '\n';
}

template <std::size_t Bytes>
std::array<std::uint8_t, Bytes> q4_source() {
  std::array<std::uint8_t, Bytes> source{};
  source[0] = 0x00;
  source[1] = 0x3c;
  for (std::size_t index = 2; index < Bytes; ++index) {
    source[index] = 0x11;
  }
  return source;
}

std::array<std::uint8_t, 34> q8_source() {
  std::array<std::uint8_t, 34> source{};
  source[0] = 0x00;
  source[1] = 0x3c;
  for (std::size_t index = 0; index < 32; ++index) {
    source[index + 2] = static_cast<std::uint8_t>(index + 1);
  }
  return source;
}

std::array<std::uint8_t, 144> q4_k_source() {
  std::array<std::uint8_t, 144> source{};
  source[0] = 0x00;
  source[1] = 0x3c;
  source[2] = 0x00;
  source[3] = 0x38;
  for (std::size_t index = 4; index < 8; ++index) {
    source[index] = 1;
  }
  for (std::size_t index = 8; index < 12; ++index) {
    source[index] = 1;
  }
  for (std::size_t index = 12; index < 16; ++index) {
    source[index] = 0x11;
  }
  for (std::size_t index = 16; index < source.size(); ++index) {
    source[index] = 0x21;
  }
  return source;
}

template <std::uint8_t DType, std::size_t Bytes, std::size_t Columns>
bool emit_case(const char *name,
               const std::array<std::uint8_t, Bytes> &source) {
  std::array<float, Columns> output{};
  std::array<float, Columns> invalid_index{};
  std::array<float, Columns> invalid_layout{};
  invalid_index.fill(k_sentinel);
  invalid_layout.fill(k_sentinel);
  if (!run_valid<DType, Bytes, Columns>(source, output.data()) ||
      !run_invalid_index<DType, Bytes, Columns>(source, invalid_index.data()) ||
      !run_invalid_layout<DType, Bytes, Columns>(source, invalid_layout.data())) {
    return false;
  }
  print_ok(name, output);
  print_reject((std::string(name) + "_invalid_index").c_str(),
               "IndexOutOfBounds", invalid_index);
  print_reject((std::string(name) + "_invalid_layout").c_str(), "InvalidView",
               invalid_layout);
  return true;
}

}  // namespace

int main() {
  const auto q4 = q4_source<18>();
  const auto q8 = q8_source();
  const auto q4k = q4_k_source();
  if (!emit_case<2, 18, 32>("q4_0", q4) ||
      !emit_case<8, 34, 32>("q8_0", q8) ||
      !emit_case<12, 144, 256>("q4_k", q4k)) {
    std::cerr << "packed target get_rows reference contract failed\n";
    return 1;
  }

  std::cout << "kernel-target-get-rows-live-packed/v1\n"
            << "source_repository=stateforward/emel.cpp\n"
            << "source_commit=" << k_source_commit << '\n'
            << "source_kernel_events_blob=" << k_events_blob << '\n'
            << "source_kernel_detail_blob=" << k_detail_blob << '\n'
            << "source_kernel_aarch64_sm_blob=" << k_aarch64_sm_blob << '\n'
            << "target_arch=aarch64\n"
            << "scope=target_router_get_rows_q4_0_q8_0_q4_k_positive_and_typed_rejection\n"
            << "execution=split_reference_and_public_target_router\n";
}
