#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <sstream>

#include "emel/kernel/sm.hpp"

namespace {

constexpr char k_source_commit[] =
    "843a117386ef17dc5a50549bbfc821074c2141d6";
constexpr char k_sml_commit[] =
    "49207123cd3f39767764bae774932cb48623f92f";

std::string bits(const std::array<float, 5> &values) {
  std::string result;
  for (std::size_t index = 0; index < values.size(); ++index) {
    if (index != 0) {
      result.push_back(',');
    }
    result += [&] {
      std::ostringstream stream;
      stream << std::hex << std::setw(8) << std::setfill('0')
             << std::bit_cast<std::uint32_t>(values[index]);
      return stream.str();
    }();
  }
  return result;
}

float silu_back(const float value, const float gradient) {
  const float sigmoid = 1.0F / (1.0F + std::exp(-value));
  return gradient * sigmoid * (1.0F + value * (1.0F - sigmoid));
}

void print_header() {
  std::cout << "kernel-activation-parity/v1\n"
            << "source_repository=stateforward/emel.cpp\n"
            << "source_commit=" << k_source_commit << '\n'
            << "source_sml_commit=" << k_sml_commit << '\n'
            << "source_kernel_events_blob=4b3f02fa5ff9c5071d1fc83938cef1b22b4658b9\n"
            << "source_kernel_detail_blob=c8a82643eabfe8f2d7883e655955f455794511b0\n"
            << "source_kernel_x86_sm_blob=0b4d635ebbd0fbd52dbca8a2345547fb571205c8\n"
            << "source_kernel_x86_actions_blob=d45558f5eb96950f43c16a09d768cb4f382d6d61\n"
            << "source_kernel_x86_guards_blob=cb3dac8253f8417c9b44acff1de414f6d0a3a3cf\n"
            << "source_detail_activation_declarations=45,55,73,93\n"
            << "source_x86_activation_rows=296-303,456-463,671-678,881-888\n"
            << "scope=portable_f32_dense_activation_scale_clamp_silu_back_leaky_relu\n";
}

}  // namespace

int main() {
  print_header();
  const std::array<float, 5> input = {
      -2.0F, -0.5F, 0.0F, 2.0F, std::bit_cast<float>(0x7fc01234U)};
  const std::array<float, 5> silu_input = {-2.0F, -0.5F, 0.0F, 2.0F, 0.25F};
  const std::array<float, 5> gradient = {0.5F, -2.0F, 1.0F, 3.0F, -0.25F};
  std::array<float, 5> output{};

  for (std::size_t index = 0; index < input.size(); ++index) {
    output[index] = input[index] * 1.5F;
  }
  std::cout << "case=op_scale status=Ok(()) output_bits=" << bits(output)
            << '\n';

  for (std::size_t index = 0; index < input.size(); ++index) {
    const float value = input[index];
    output[index] = value < -1.0F ? -1.0F : value > 1.0F ? 1.0F : value;
  }
  std::cout << "case=op_clamp status=Ok(()) output_bits=" << bits(output)
            << '\n';

  for (std::size_t index = 0; index < input.size(); ++index) {
    output[index] = silu_back(silu_input[index], gradient[index]);
  }
  std::cout << "case=op_silu_back status=Ok(()) output_bits=" << bits(output)
            << '\n';

  for (std::size_t index = 0; index < input.size(); ++index) {
    const float value = input[index];
    output[index] = value < 0.0F ? value * 0.125F : value;
  }
  std::cout << "case=op_leaky_relu status=Ok(()) output_bits=" << bits(output)
            << '\n';
  return 0;
}
