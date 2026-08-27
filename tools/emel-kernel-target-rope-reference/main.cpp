#include <array>
#include <bit>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <sstream>
#include "emel/kernel/aarch64/sm.hpp"

namespace {
constexpr char k_commit[] = "843a117386ef17dc5a50549bbfc821074c2141d6";
constexpr char k_events[] = "4b3f02fa5ff9c5071d1fc83938cef1b22b4658b9";
constexpr char k_detail[] = "c8a82643eabfe8f2d7883e655955f455794511b0";
constexpr char k_guards[] = "c25714566ec9a02679daef85089544575123408e";
constexpr char k_actions[] = "267d4f74e6e7498155c8535920322ffef2c02fb6";
constexpr char k_sm[] = "865a9cc6ba6115382ed043c464f3d62bcd851357";
constexpr float k_sentinel = std::bit_cast<float>(std::uint32_t{0x7fc00001});

emel::kernel::event::tensor_view view(const float *data, std::array<std::uint64_t, 4> ne,
                                      std::array<std::uint64_t, 4> nb) {
  return {.data=data, .type=emel::kernel::event::dtype::f32, .ne=ne, .nb=nb};
}
emel::kernel::event::tensor_view positions(const std::int32_t *data, std::uint64_t count) {
  return {.data=data, .type=emel::kernel::event::dtype::i32,
          .ne={count,1,1,1}, .nb={4,4*count,4*count,4*count}};
}
emel::kernel::event::tensor_view_mut output(float *data, std::array<std::uint64_t, 4> ne) {
  return {.data=data, .type=emel::kernel::event::dtype::f32, .ne=ne,
          .nb={4,4*ne[0],4*ne[0]*ne[1],4*ne[0]*ne[1]*ne[2]}};
}
std::string bits(const float *values, std::size_t count) {
  std::ostringstream out;
  for (std::size_t i=0; i<count; ++i) { if (i) out << ','; out << std::hex << std::setw(8) << std::setfill('0') << std::bit_cast<std::uint32_t>(values[i]); }
  return out.str();
}
void set_params(auto &request, std::int32_t mode) {
  const std::int32_t n_dims=4; const float base=10'000.0F, scale=1.0F, ext=0.0F, attn=1.0F;
  std::memcpy(request.op_params.data()+4, &n_dims, 4); std::memcpy(request.op_params.data()+8, &mode, 4);
  std::memcpy(request.op_params.data()+20, &base, 4); std::memcpy(request.op_params.data()+24, &scale, 4);
  std::memcpy(request.op_params.data()+28, &ext, 4); std::memcpy(request.op_params.data()+32, &attn, 4);
  request.op_params_size=36;
}
void header() {
  std::cout << "kernel-target-rope-live/v1\nsource_repository=stateforward/emel.cpp\nsource_commit=" << k_commit
            << "\nsource_kernel_events_blob=" << k_events << "\nsource_kernel_detail_blob=" << k_detail
            << "\nsource_kernel_aarch64_guards_blob=" << k_guards << "\nsource_kernel_aarch64_actions_blob=" << k_actions
            << "\nsource_kernel_aarch64_sm_blob=" << k_sm << "\ntarget_arch=aarch64\n"
            << "scope=dense_contiguous_f32_target_router_rope_norm_neox_timestep_and_typed_rejection\n"
            << "execution=split_pinned_aarch64_sm_and_public_target_aarch64_kernel\n";
}
void emit(auto &kernel, const char *name, auto &request, float *data, std::size_t count) {
  if (!kernel.process_event(request)) std::abort();
  std::cout << "case=" << name << " status=ok output_bits=" << bits(data,count) << '\n';
}
}

int main() {
  using namespace emel::kernel;
  header(); event::op_rope norm{}; aarch64::sm kernel;
  constexpr std::array<float,12> source={1,2,3,4,5,6,7,8,9,10,11,12}; constexpr std::array<std::int32_t,2> pos={1,2}; std::array<float,12> out{};
  norm.src0=view(source.data(),{6,1,2,1},{4,24,24,48}); norm.src1=positions(pos.data(),2); norm.dst=output(out.data(),{6,1,2,1}); set_params(norm,0); emit(kernel,"norm",norm,out.data(),out.size());
  constexpr std::array<float,4> short_source={1,2,3,4}; constexpr std::array<std::int32_t,1> one={1}; std::array<float,4> short_out{};
  auto make = [&](std::int32_t mode) { event::op_rope request{}; request.src0=view(short_source.data(),{4,1,1,1},{4,16,16,16}); request.src1=positions(one.data(),1); request.dst=output(short_out.data(),{4,1,1,1}); set_params(request,mode); return request; };
  auto neox=make(2); emit(kernel,"neox",neox,short_out.data(),short_out.size()); auto timestep=make(4); emit(kernel,"timestep",timestep,short_out.data(),short_out.size());
  auto invalid=make(99); short_out.fill(k_sentinel); if (kernel.process_event(invalid)) std::abort(); std::cout << "case=invalid_mode status=reject error=InvalidMode output_bits=" << bits(short_out.data(),short_out.size()) << '\n';
}
