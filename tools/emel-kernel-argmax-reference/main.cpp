#include <array>
#include <bit>
#include <cstdint>
#include <iomanip>
#include <iostream>

#include "emel/kernel/sm.hpp"

namespace {

constexpr char k_commit[] = "843a117386ef17dc5a50549bbfc821074c2141d6";

emel::kernel::event::tensor_view f32_view(
    const float *data, const std::array<std::uint64_t, 4> ne,
    const std::array<std::uint64_t, 4> nb) {
  return {.data = data,
          .type = emel::kernel::event::dtype::f32,
          .ne = ne,
          .nb = nb};
}

emel::kernel::event::tensor_view packed_view(
    const std::uint8_t *data, const emel::kernel::event::dtype type,
    const std::uint64_t k, const std::uint64_t rows,
    const std::uint64_t row_bytes) {
  return {.data = data,
          .type = type,
          .ne = {k, rows, 1, 1},
          .nb = {1, row_bytes, row_bytes * rows, row_bytes * rows}};
}

emel::kernel::event::tensor_view_mut f32_view_mut(
    float *data, const std::array<std::uint64_t, 4> ne,
    const std::array<std::uint64_t, 4> nb) {
  return {.data = data,
          .type = emel::kernel::event::dtype::f32,
          .ne = ne,
          .nb = nb};
}

void print_result(const char *name, const bool status, const std::int32_t index,
                  const float output) {
  std::cout << "case=" << name << " status="
            << (status ? "ok" : "error") << ' ';
  if (status) {
    std::cout << "index=" << index;
  } else {
    std::cout << "error=InvalidOrShape index=-77";
  }
  std::cout << " output_bits=" << std::hex << std::setw(8)
            << std::setfill('0') << std::bit_cast<std::uint32_t>(output)
            << std::dec << '\n';
}

void print_error(const char *name, const char *error, const float output) {
  std::cout << "case=" << name << " status=error error=" << error
            << " index=-77 output_bits=" << std::hex << std::setw(8)
            << std::setfill('0') << std::bit_cast<std::uint32_t>(output)
            << std::dec << '\n';
}

template <std::size_t N>
std::array<std::uint8_t, N> q5_fixture() {
  std::array<std::uint8_t, N> bytes{};
  for (std::size_t block = 0; block < 16; ++block) {
    const std::size_t offset = block * 22;
    bytes[offset] = 0x00;
    bytes[offset + 1] = 0x3c;
    if (block >= 8) {
      for (std::size_t i = offset + 2; i < offset + 6; ++i) bytes[i] = 0xff;
      for (std::size_t i = offset + 6; i < offset + 22; ++i) bytes[i] = 0x0f;
    }
  }
  return bytes;
}

template <std::size_t N>
std::array<std::uint8_t, N> q8_fixture() {
  std::array<std::uint8_t, N> bytes{};
  for (std::size_t block = 0; block < 16; ++block) {
    const std::size_t offset = block * 34;
    bytes[offset] = 0x00;
    bytes[offset + 1] = 0x3c;
    if (block >= 8) {
      for (std::size_t i = offset + 2; i < offset + 34; ++i) bytes[i] = 1;
    }
  }
  return bytes;
}

template <std::size_t N>
std::array<std::uint8_t, N> q2_fixture() {
  std::array<std::uint8_t, N> bytes{};
  bytes[84 + 80] = 0x00;
  bytes[84 + 81] = 0x3c;
  for (std::size_t i = 84; i < 84 + 16; ++i) bytes[i] = 0xff;
  for (std::size_t i = 84 + 16; i < 84 + 80; ++i) bytes[i] = 0xff;
  return bytes;
}

template <std::size_t N>
std::array<std::uint8_t, N> q3_fixture() {
  std::array<std::uint8_t, N> bytes{};
  for (std::size_t i = 110; i < 110 + 32; ++i) bytes[i] = 0xff;
  for (std::size_t i = 110 + 32; i < 110 + 96; ++i) bytes[i] = 0xff;
  for (std::size_t i = 110 + 96; i < 110 + 108; ++i) bytes[i] = 0x21;
  bytes[110 + 108] = 0x00;
  bytes[110 + 109] = 0x3c;
  return bytes;
}

template <std::size_t N>
std::array<std::uint8_t, N> q4_k_fixture() {
  std::array<std::uint8_t, N> bytes{};
  bytes[144] = 0x00;
  bytes[145] = 0x3c;
  for (std::size_t i = 144 + 4; i < 144 + 16; ++i) bytes[i] = 1;
  for (std::size_t i = 144 + 16; i < N; ++i) bytes[i] = 0xff;
  return bytes;
}

template <std::size_t N>
std::array<std::uint8_t, N> q6_k_fixture() {
  std::array<std::uint8_t, N> bytes{};
  for (std::size_t i = 210; i < 210 + 128; ++i) bytes[i] = 0xff;
  for (std::size_t i = 210 + 128; i < 210 + 192; ++i) bytes[i] = 0xff;
  for (std::size_t i = 210 + 192; i < 210 + 208; ++i) bytes[i] = 1;
  bytes[210 + 208] = 0x00;
  bytes[210 + 209] = 0x3c;
  return bytes;
}

template <std::size_t N>
void run_quant(const char *name, const emel::kernel::event::dtype type,
               const std::uint64_t row_bytes,
               const std::array<std::uint8_t, N> &lhs, const float *rhs) {
  std::array<float, 1> output{std::bit_cast<float>(0x7fc01234U)};
  std::int32_t index = -77;
  emel::kernel::event::op_mul_mat_argmax request{};
  request.src0 = packed_view(lhs.data(), type, 256, 2, row_bytes);
  request.src1 = f32_view(rhs, {1, 256, 1, 1}, {4, 4, 1024, 1024});
  request.dst = f32_view_mut(output.data(), {1, 1, 1, 1}, {4, 4, 4, 4});
  request.index_out = &index;
  emel::kernel::Kernel kernel;
  print_result(name, kernel.process_event(request), index, output[0]);
}

}  // namespace

int main() {
  std::cout << "kernel-argmax-parity/v1\n"
            << "source_repository=stateforward/emel.cpp\n"
            << "source_commit=" << k_commit << '\n'
            << "source_detail_blob=c8a82643eabfe8f2d7883e655955f455794511b0\n"
            << "source_events_blob=4b3f02fa5ff9c5071d1fc83938cef1b22b4658b9\n"
            << "scope=f32_plus_aligned_native_quantized_argmax\n"
            << "families=q5_0,q8_0,q2_k,q3_k,q4_k,q6_k\n"
            << "excluded=q4_0,q4_1_reference_run_has_no_branch\n";

  const std::array<float, 6> lhs = {1, 2, 3, 4, 5, 6};
  const std::array<float, 3> rhs = {1, 1, 1};
  std::array<float, 1> output{std::bit_cast<float>(0x7fc01234U)};
  std::int32_t index = -77;
  emel::kernel::event::op_mul_mat_argmax f32{};
  f32.src0 = f32_view(lhs.data(), {3, 2, 1, 1}, {4, 12, 24, 24});
  f32.src1 = f32_view(rhs.data(), {1, 3, 1, 1}, {4, 4, 12, 12});
  f32.dst = f32_view_mut(output.data(), {1, 1, 1, 1}, {4, 4, 4, 4});
  f32.index_out = &index;
  emel::kernel::Kernel kernel;
  print_result("f32_success", kernel.process_event(f32), index, output[0]);

  output[0] = std::bit_cast<float>(0x7fc05678U);
  index = -77;
  f32.src1 = f32_view(rhs.data(), {1, 2, 1, 1}, {4, 4, 8, 8});
  (void)kernel.process_event(f32);
  print_error("f32_shape_error", "ShapeMismatch", output[0]);

  output[0] = std::bit_cast<float>(0x7fc09abcU);
  index = -77;
  f32.src1.type = emel::kernel::event::dtype::f16;
  f32.src1.nb = {2, 2, 6, 6};
  (void)kernel.process_event(f32);
  print_error("f32_invalid_view", "InvalidView", output[0]);

  const std::array<float, 256> quant_rhs = [] {
    std::array<float, 256> values{};
    values.fill(1.0F);
    return values;
  }();
  run_quant("q5_0_success", emel::kernel::event::dtype::q5_0, 176,
            q5_fixture<352>(), quant_rhs.data());
  run_quant("q8_0_success", emel::kernel::event::dtype::q8_0, 272,
            q8_fixture<544>(), quant_rhs.data());
  run_quant("q2_k_success", emel::kernel::event::dtype::q2_k, 84,
            q2_fixture<168>(), quant_rhs.data());
  run_quant("q3_k_success", emel::kernel::event::dtype::q3_k, 110,
            q3_fixture<220>(), quant_rhs.data());
  run_quant("q4_k_success", emel::kernel::event::dtype::q4_k, 144,
            q4_k_fixture<288>(), quant_rhs.data());
  run_quant("q6_k_success", emel::kernel::event::dtype::q6_k, 210,
            q6_k_fixture<420>(), quant_rhs.data());
}
