#include <array>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "emel/io/loader/events.hpp"
#include "emel/io/loader/sm.hpp"
#include "emel/io/read/errors.hpp"
#include "emel/io/read/sm.hpp"
#include "emel/io/staged_read/errors.hpp"
#include "emel/io/staged_read/sm.hpp"

namespace {
namespace loader = emel::io::loader;

struct owner {
  bool done{};
  emel::error::type error{};
  emel::error::type strategy_error{};
  loader::event::strategy_kind strategy{};
  std::uint32_t count{};
  std::uint64_t bytes{};
  std::uint32_t failed_index{};
};

void single_done(void *raw, const loader::events::load_tensor_done &event) noexcept {
  auto &value = *static_cast<owner *>(raw);
  value.done = true; value.strategy = event.strategy; value.bytes = event.buffer_bytes;
}
void single_error(void *raw, const loader::events::load_tensor_error &event) noexcept {
  auto &value = *static_cast<owner *>(raw);
  value.error = event.err; value.strategy_error = event.strategy_err;
}
void batch_done(void *raw, const loader::events::load_tensor_batch_done &event) noexcept {
  auto &value = *static_cast<owner *>(raw);
  value.done = true; value.strategy = event.strategy; value.count = event.done_count;
  value.bytes = event.bytes_done;
}
void batch_error(void *raw, const loader::events::load_tensor_batch_error &event) noexcept {
  auto &value = *static_cast<owner *>(raw);
  value.error = event.err; value.strategy_error = event.strategy_err;
  value.failed_index = event.failed_index;
}

std::string hex(const char *bytes, std::size_t size) {
  std::ostringstream output; output << std::hex << std::setfill('0');
  for (std::size_t index = 0; index < size; ++index) {
    output << std::setw(2) << static_cast<unsigned>(static_cast<unsigned char>(bytes[index]));
  }
  return output.str();
}

const char *strategy_name(loader::event::strategy_kind value) {
  if (value == loader::event::strategy_kind::read_copy) return "read_copy";
  if (value == loader::event::strategy_kind::staged_read) return "staged_read";
  return "other";
}
const char *error_name(emel::error::type value) {
  if (value == emel::error::cast(loader::error::invalid_request)) return "invalid_request";
  if (value == emel::error::cast(loader::error::unsupported_strategy)) return "unsupported_strategy";
  if (value == emel::error::cast(loader::error::unavailable)) return "unavailable";
  return "internal_error";
}
const char *strategy_error_name(emel::error::type value) {
  if (value == emel::error::cast(emel::io::read::error::file_read_failed)) return "file_read_failed";
  if (value == emel::error::cast(loader::error::none)) return "none";
  if (value == emel::error::cast(emel::io::staged_read::error::invalid_stage_contract)) return "invalid_stage_contract";
  return "read_error";
}

template <class Machine>
void render_single_error(const char *name, Machine &machine,
                         const loader::event::tensor_load_span &span,
                         loader::event::strategy_kind strategy,
                         std::uint64_t chunk = loader::event::k_default_staged_read_chunk_bytes) {
  owner state{}; const loader::event::strategy_policy policy{strategy, chunk};
  loader::event::load_tensor request{span, policy};
  request.on_done = {&state, single_done}; request.on_error = {&state, single_error};
  (void)machine.process_event(request);
  std::cout << "case=" << name << " outcome=error error=" << error_name(state.error)
            << " strategy_error=" << strategy_error_name(state.strategy_error) << '\n';
}

template <class Machine>
void render_batch_error(const char *name, Machine &machine,
                        std::span<const loader::event::tensor_load_span> spans,
                        loader::event::strategy_kind strategy,
                        std::uint64_t chunk = loader::event::k_default_staged_read_chunk_bytes) {
  owner state{}; const loader::event::strategy_policy policy{strategy, chunk};
  loader::event::load_tensor_batch request{spans, policy};
  request.on_done = {&state, batch_done}; request.on_error = {&state, batch_error};
  (void)machine.process_event(request);
  std::cout << "case=" << name << " outcome=error error=" << error_name(state.error)
            << " strategy_error=" << strategy_error_name(state.strategy_error)
            << " failed_index=" << state.failed_index << '\n';
}

double benchmark_loader(std::uint64_t iterations, std::size_t runs,
                        std::uint64_t warmup) {
  constexpr std::size_t size = 1024 * 1024;
  std::vector<char> source(size, static_cast<char>(0xa5));
  std::vector<char> target(size);
  emel::io::read::sm reader{}; loader::sm machine{{.io_read=&reader}};
  const loader::event::tensor_load_span span{.tensor_id=1,.file_offset=0,.byte_size=size,
    .file_path="benchmark.bin",.source_buffer=source.data(),.source_buffer_bytes=size,
    .target=target.data(),.target_bytes=size};
  const loader::event::strategy_policy policy{loader::event::strategy_kind::read_copy};
  auto execute = [&](std::uint64_t count) {
    for (std::uint64_t index=0; index<count; ++index) {
      loader::event::load_tensor request{span, policy};
      if (!machine.process_event(request)) throw std::runtime_error("loader benchmark failed");
    }
  };
  execute(warmup);
  std::vector<double> samples;
  for (std::size_t run=0; run<runs; ++run) {
    const auto begin=std::chrono::steady_clock::now(); execute(iterations);
    const auto elapsed=std::chrono::duration<double,std::nano>(std::chrono::steady_clock::now()-begin).count();
    samples.push_back(elapsed/static_cast<double>(iterations));
  }
  if (target != source) throw std::runtime_error("loader benchmark target mismatch");
  std::sort(samples.begin(),samples.end()); return samples[samples.size()/2];
}
}

int main(int argc, char **argv) {
  if (argc == 5 && std::string_view{argv[1]} == "--benchmark") {
    const auto iterations=std::stoull(argv[2]); const auto runs=std::stoull(argv[3]);
    const auto warmup=std::stoull(argv[4]);
    std::cout << "io/loader/reference/read_copy_1mib ns_per_op=" << std::fixed << std::setprecision(3)
              << benchmark_loader(iterations,runs,warmup) << " iter=" << iterations << " runs=" << runs << '\n';
    return 0;
  }
  std::cout << "io-loader-parity-snapshot/v1\n"
               "source_repository=stateforward/emel.cpp\n"
               "source_commit=843a117386ef17dc5a50549bbfc821074c2141d6\n"
               "source_tree=ff00a9978b00b4ea1e5268d2ecd483d4855b6eaa\n"
               "source_files=src/emel/io/loader/events.hpp,errors.hpp,context.hpp,detail.hpp,guards.hpp,actions.hpp,sm.hpp\n"
               "source_tests=tests/io/loader/lifecycle_tests.cpp\n"
               "fixture_config=public_loader,static_read_and_staged_dependencies,single_and_batch,typed_failures\n"
               "contract_delta=rust_target_capability_replaces_raw_buffer_result;rust_loader_batch_max=65536_for_bounded_rtc_cpp_reference_unbounded\n";
  constexpr char source[] = "abcdefghij";
  std::array<char, 4> target{};
  const loader::event::tensor_load_span valid{.tensor_id=3,.file_index=1,.file_offset=1,.byte_size=3,
      .file_path="fixture.bin",.source_buffer=source,.source_buffer_bytes=10,.target=target.data(),.target_bytes=3};
  auto invalid = valid; invalid.byte_size = 0;
  loader::sm absent{};
  render_single_error("invalid", absent, invalid, loader::event::strategy_kind::read_copy);
  render_single_error("none", absent, valid, loader::event::strategy_kind::none);
  render_single_error("mapped", absent, valid, loader::event::strategy_kind::mapped_file);
  render_single_error("external", absent, valid, loader::event::strategy_kind::external_buffer);
  render_single_error("unknown", absent, valid, static_cast<loader::event::strategy_kind>(255));

  for (const auto [name, strategy] : std::array{
      std::pair{"read_success", loader::event::strategy_kind::read_copy},
      std::pair{"staged_success", loader::event::strategy_kind::staged_read}}) {
    std::array<char, 4> bytes{}; auto span = valid; span.tensor_id=17; span.file_offset=2;
    span.byte_size=4; span.target=bytes.data(); span.target_bytes=4;
    emel::io::read::sm reader{}; emel::io::staged_read::sm stager{};
    loader::sm machine{{.io_read=&reader,.io_staged_read=&stager}};
    owner state{}; const loader::event::strategy_policy policy{strategy, 2};
    loader::event::load_tensor request{span, policy};
    request.on_done={&state,single_done}; request.on_error={&state,single_error};
    (void)machine.process_event(request);
    std::cout << "case=" << name << " outcome=done strategy=" << strategy_name(state.strategy)
              << " bytes_loaded=" << state.bytes << " callback=" << std::boolalpha << state.done
              << " target=" << hex(bytes.data(), bytes.size()) << '\n';
  }
  render_single_error("read_absent", absent, valid, loader::event::strategy_kind::read_copy);
  auto failed=valid; failed.source_error=emel::error::cast(emel::io::read::error::file_read_failed);
  emel::io::read::sm reader{}; loader::sm read_loader{{.io_read=&reader}};
  render_single_error("read_failure", read_loader, failed, loader::event::strategy_kind::read_copy);
  emel::io::staged_read::sm failed_stager{};
  loader::sm staged_loader{{.io_staged_read=&failed_stager}};
  render_single_error("staged_failure", staged_loader, valid,
                      loader::event::strategy_kind::staged_read, 0);

  for (const auto [name, strategy] : std::array{
      std::pair{"batch_read", loader::event::strategy_kind::read_copy},
      std::pair{"batch_staged", loader::event::strategy_kind::staged_read}}) {
    std::array<char,3> first{}; std::array<char,4> second{};
    std::array spans{valid,valid}; spans[0].tensor_id=10; spans[0].file_offset=1;
    spans[0].target=first.data(); spans[0].target_bytes=3;
    spans[1].tensor_id=11; spans[1].file_offset=5; spans[1].byte_size=4;
    spans[1].target=second.data(); spans[1].target_bytes=4;
    emel::io::read::sm batch_reader{}; emel::io::staged_read::sm batch_stager{};
    loader::sm machine{{.io_read=&batch_reader,.io_staged_read=&batch_stager}};
    owner state{}; const loader::event::strategy_policy policy{strategy};
    loader::event::load_tensor_batch request{spans,policy};
    request.on_done={&state,batch_done}; request.on_error={&state,batch_error};
    (void)machine.process_event(request);
    std::cout << "case=" << name << " outcome=done strategy=" << strategy_name(state.strategy)
              << " done_count=" << state.count << " bytes_loaded=" << state.bytes
              << " callback=" << std::boolalpha << state.done << " first=" << hex(first.data(),first.size())
              << " second=" << hex(second.data(),second.size()) << '\n';
  }
  std::array invalid_batch{invalid};
  render_batch_error("batch_invalid", read_loader, invalid_batch, loader::event::strategy_kind::read_copy);
  std::array valid_batch{valid};
  render_batch_error("batch_absent", absent, valid_batch, loader::event::strategy_kind::read_copy);
  std::array read_failed_batch{valid, failed};
  render_batch_error("batch_read_failure", read_loader, read_failed_batch,
                     loader::event::strategy_kind::read_copy);
  render_batch_error("batch_staged_failure", staged_loader, valid_batch,
                     loader::event::strategy_kind::staged_read, 0);
}
