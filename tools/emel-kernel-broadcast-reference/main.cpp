#include <array>
#include <bit>
#include <cstdint>
#include <iostream>

#include "emel/kernel/sm.hpp"

namespace {
constexpr char k_commit[] = "843a117386ef17dc5a50549bbfc821074c2141d6";

emel::kernel::event::tensor_view view(const float *data,
                                     std::array<std::uint64_t, 4> ne,
                                     std::array<std::uint64_t, 4> nb) {
  return {.data = data, .type = emel::kernel::event::dtype::f32, .ne = ne, .nb = nb};
}
emel::kernel::event::tensor_view_mut view_mut(float *data,
                                              std::array<std::uint64_t, 4> ne,
                                              std::array<std::uint64_t, 4> nb) {
  return {.data = data, .type = emel::kernel::event::dtype::f32, .ne = ne, .nb = nb};
}

template <std::size_t N>
void print_case(const char *name, bool ok, const std::array<float, N> &data,
                std::array<std::uint64_t, 4> ne,
                std::array<std::uint64_t, 4> nb) {
  auto strides = nb;
  if (strides[0] == 0) {
    strides[0] = 4;
    for (std::size_t d = 1; d < 4; ++d) strides[d] = strides[d - 1] * ne[d - 1];
  }
  const auto count = static_cast<std::size_t>(ne[0] * ne[1] * ne[2] * ne[3]);
  std::cout << "case=" << name << " status=" << (ok ? "ok" : "error") << " output_bits=";
  for (std::size_t ordinal = 0; ordinal < count; ++ordinal) {
    std::uint64_t remaining = ordinal;
    std::uint64_t offset = 0;
    for (std::size_t d = 0; d < 4; ++d) {
      offset += (remaining % ne[d]) * strides[d];
      remaining /= ne[d];
    }
    if (ordinal != 0) std::cout << ',';
    const auto word = std::bit_cast<std::uint32_t>(data[offset / 4]);
    for (int shift = 28; shift >= 0; shift -= 4) std::cout << "0123456789abcdef"[(word >> shift) & 15];
  }
  std::cout << '\n';
}

template <class Op, std::size_t N>
void run(const char *name, const float *src, const float *row,
         std::array<std::uint64_t, 4> src_ne, std::array<std::uint64_t, 4> src_nb,
         std::array<std::uint64_t, 4> row_ne, std::array<std::uint64_t, 4> row_nb,
         std::array<std::uint64_t, 4> dst_ne, std::array<std::uint64_t, 4> dst_nb) {
  std::array<float, N> output{};
  Op request{};
  request.src0 = view(src, src_ne, src_nb);
  request.src1 = view(row, row_ne, row_nb);
  request.dst = view_mut(output.data(), dst_ne, dst_nb);
  emel::kernel::Kernel kernel;
  print_case(name, kernel.process_event(request), output, dst_ne, dst_nb);
}
}  // namespace

int main() {
  std::cout << "kernel-broadcast-parity/v1\n"
            << "source_repository=stateforward/emel.cpp\n"
            << "source_commit=" << k_commit << '\n'
            << "source_sml_commit=49207123cd3f39767764bae774932cb48623f92f\n"
            << "source_detail_guard_span=3788-3836\n"
            << "source_detail_run_span=3823-3836\n"
            << "source_x86_add_sm_span=46-63\n"
            << "source_x86_mul_sm_span=111-128\n"
            << "scope=portable_f32_row_broadcast_add_mul\n";
  constexpr std::array<std::uint64_t, 4> src_ne = {3, 2, 1, 1};
  constexpr std::array<std::uint64_t, 4> src_nb = {4, 12, 24, 24};
  constexpr std::array<std::uint64_t, 4> row_ne = {3, 1, 1, 1};
  constexpr std::array<std::uint64_t, 4> row_nb = {4, 12, 12, 12};
  const std::array<float, 6> source = {1, 2, 3, 4, 5, 6};
  const std::array<float, 3> row = {10, 20, 30};
  run<emel::kernel::event::op_add, 6>("row_add", source.data(), row.data(), src_ne, src_nb,
                                      row_ne, row_nb, src_ne, src_nb);
  run<emel::kernel::event::op_mul, 6>("row_mul", source.data(), row.data(), src_ne, src_nb,
                                      row_ne, row_nb, src_ne, src_nb);
  constexpr std::array<std::uint64_t, 4> implicit = {0, 0, 0, 0};
  run<emel::kernel::event::op_add, 6>("row_add_implicit", source.data(), row.data(), src_ne, implicit,
                                      row_ne, implicit, src_ne, implicit);
  return 0;
}
