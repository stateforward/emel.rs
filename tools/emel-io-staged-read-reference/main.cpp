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

#include "emel/error/error.hpp"
#include "emel/io/events.hpp"
#include "emel/io/staged_read/errors.hpp"
#include "emel/io/staged_read/events.hpp"
#include "emel/io/staged_read/sm.hpp"

namespace {

using staged_error = emel::io::staged_read::error;

struct single_owner {
  bool done = false;
  bool failed = false;
  std::uint64_t bytes_committed = 0;
  emel::error::type error = emel::error::cast(staged_error::none);
};

struct batch_owner {
  bool done = false;
  bool failed = false;
  std::uint32_t done_count = 0;
  std::uint64_t bytes_committed = 0;
  std::uint32_t failed_index = 0;
  emel::error::type error = emel::error::cast(staged_error::none);
};

void on_single_done(
    void *object,
    const emel::io::staged_read::events::staged_window_done &event) noexcept {
  auto &owner = *static_cast<single_owner *>(object);
  owner.done = true;
  owner.bytes_committed = event.bytes_committed;
}

void on_single_error(
    void *object,
    const emel::io::staged_read::events::staged_window_error &event) noexcept {
  auto &owner = *static_cast<single_owner *>(object);
  owner.failed = true;
  owner.error = event.err;
}

void on_batch_done(
    void *object,
    const emel::io::staged_read::events::staged_window_batch_done &event) noexcept {
  auto &owner = *static_cast<batch_owner *>(object);
  owner.done = true;
  owner.done_count = event.done_count;
  owner.bytes_committed = event.bytes_committed;
}

void on_batch_error(
    void *object,
    const emel::io::staged_read::events::staged_window_batch_error &event) noexcept {
  auto &owner = *static_cast<batch_owner *>(object);
  owner.failed = true;
  owner.error = event.err;
  owner.failed_index = event.failed_index;
}

std::string_view error_name(const emel::error::type error) {
  if (error == emel::error::cast(staged_error::invalid_callbacks)) {
    return "invalid_callbacks";
  }
  if (error == emel::error::cast(staged_error::invalid_stage_contract)) {
    return "invalid_stage_contract";
  }
  if (error == emel::error::cast(staged_error::invalid_target_window)) {
    return "invalid_target_window";
  }
  if (error == emel::error::cast(staged_error::unsupported_platform)) {
    return "unsupported_platform";
  }
  if (error == emel::error::cast(staged_error::null_source_span)) {
    return "null_source_span";
  }
  if (error == emel::error::cast(staged_error::source_span_size_mismatch)) {
    return "source_span_size_mismatch";
  }
  if (error == emel::error::cast(staged_error::insufficient_source_span)) {
    return "insufficient_source_span";
  }
  return "internal_error";
}

std::string hex(const std::uint8_t *bytes, const std::size_t length) {
  std::ostringstream output;
  output << std::hex << std::setfill('0');
  for (std::size_t index = 0; index < length; ++index) {
    output << std::setw(2) << static_cast<unsigned>(bytes[index]);
  }
  return output.str();
}

void render_single_success(std::ostream &output, const std::string_view name,
                           const std::string_view source,
                           const std::uint64_t chunk) {
  emel::io::staged_read::sm actor{};
  single_owner owner{};
  std::vector<std::uint8_t> target(source.size());
  const emel::io::staged_read::event::staged_window_request request{
      .file_offset = 17,
      .logical_byte_length = source.size(),
      .stage_chunk_bytes = chunk,
      .source_span = source.data(),
      .source_span_bytes = source.size(),
      .target_buffer = target.data(),
      .target_window_bytes = target.size(),
  };
  emel::io::staged_read::event::staged_window event{request};
  event.on_done = {&owner, on_single_done};
  event.on_error = {&owner, on_single_error};
  if (!actor.process_event(event) || !owner.done || owner.failed) {
    throw std::runtime_error("single success did not complete");
  }
  output << "case=single_" << name
         << " outcome=done bytes_committed=" << owner.bytes_committed
         << " callback_bytes_committed=" << owner.bytes_committed
         << " target=" << hex(target.data(), target.size()) << '\n';
}

void render_single_error(std::ostream &output, const std::string_view name,
                         const std::uint64_t offset,
                         const std::uint64_t logical,
                         const std::uint64_t chunk, const void *source,
                         const std::uint64_t source_bytes,
                         const std::size_t target_length,
                         const bool callbacks) {
  emel::io::staged_read::sm actor{};
  single_owner owner{};
  std::vector<std::uint8_t> target(target_length);
  const emel::io::staged_read::event::staged_window_request request{
      .file_offset = offset,
      .logical_byte_length = logical,
      .stage_chunk_bytes = chunk,
      .source_span = source,
      .source_span_bytes = source_bytes,
      .target_buffer = target.data(),
      .target_window_bytes = target.size(),
  };
  emel::io::staged_read::event::staged_window event{request};
  if (callbacks) {
    event.on_done = {&owner, on_single_done};
    event.on_error = {&owner, on_single_error};
  }
  if (actor.process_event(event) || owner.done) {
    throw std::runtime_error("single error did not fail");
  }
  const auto error = callbacks ? owner.error
                               : emel::error::cast(staged_error::invalid_callbacks);
  output << "case=single_" << name << " outcome=error error="
         << error_name(error) << " callback=" << std::boolalpha << owner.failed
         << " target=" << hex(target.data(), target.size()) << '\n';
}

void render_single_errors(std::ostream &output) {
  constexpr std::string_view source = "abcde";
  render_single_error(output, "missing_callbacks", 0, 4, 2, source.data(), 4,
                      4, false);
  render_single_error(output, "invalid_contract", 0, 0, 0, source.data(), 0,
                      0, true);
  render_single_error(output, "offset_overflow", UINT64_MAX, 2, 1,
                      source.data(), 2, 2, true);
  render_single_error(output, "invalid_target", 0, 4, 2, source.data(), 4, 3,
                      true);
  render_single_error(output, "null_source", 0, 4, 2, nullptr, 0, 4, true);
  render_single_error(output, "insufficient_source", 0, 4, 2, source.data(),
                      3, 4, true);
  render_single_error(output, "source_mismatch", 0, 4, 2, source.data(), 5, 4,
                      true);
}

void render_batch_success(std::ostream &output) {
  emel::io::staged_read::sm actor{};
  batch_owner owner{};
  constexpr std::string_view source = "0123456789abcdef";
  std::array<std::uint8_t, 4> first{};
  std::array<std::uint8_t, 5> second{};
  const std::array spans{
      emel::io::event::tensor_load_span{
          .file_offset = 2,
          .byte_size = 4,
          .source_buffer = source.data(),
          .source_buffer_bytes = source.size(),
          .target = first.data(),
          .target_bytes = first.size(),
      },
      emel::io::event::tensor_load_span{
          .file_offset = 8,
          .byte_size = 5,
          .source_buffer = source.data(),
          .source_buffer_bytes = source.size(),
          .target = second.data(),
          .target_bytes = second.size(),
      },
  };
  emel::io::staged_read::event::staged_window_batch event{spans, 3};
  event.on_done = {&owner, on_batch_done};
  event.on_error = {&owner, on_batch_error};
  if (!actor.process_event(event) || !owner.done || owner.failed) {
    throw std::runtime_error("batch success did not complete");
  }
  output << "case=batch_success outcome=done done_count=" << owner.done_count
         << " bytes_committed=" << owner.bytes_committed
         << " callback_done_count=" << owner.done_count
         << " callback_bytes_committed=" << owner.bytes_committed
         << " first_target=" << hex(first.data(), first.size())
         << " second_target=" << hex(second.data(), second.size()) << '\n';
}

void render_batch_errors(std::ostream &output) {
  emel::io::staged_read::sm actor{};
  batch_owner owner{};
  constexpr std::string_view source = "0123456789abcdef";
  std::array<std::uint8_t, 4> first{};
  std::array<std::uint8_t, 1> second{};
  const std::array spans{
      emel::io::event::tensor_load_span{
          .file_offset = 0,
          .byte_size = 4,
          .source_buffer = source.data(),
          .source_buffer_bytes = source.size(),
          .target = first.data(),
          .target_bytes = first.size(),
      },
      emel::io::event::tensor_load_span{
          .file_offset = 15,
          .byte_size = 2,
          .source_buffer = source.data(),
          .source_buffer_bytes = source.size(),
          .target = second.data(),
          .target_bytes = second.size(),
      },
  };
  emel::io::staged_read::event::staged_window_batch event{spans, 2};
  event.on_done = {&owner, on_batch_done};
  event.on_error = {&owner, on_batch_error};
  if (actor.process_event(event) || owner.done || !owner.failed) {
    throw std::runtime_error("batch error did not fail");
  }
  output << "case=batch_invalid outcome=error error=" << error_name(owner.error)
         << " failed_index=" << owner.failed_index
         << " callback=" << std::boolalpha << owner.failed
         << " first_target=" << hex(first.data(), first.size())
         << " second_target=" << hex(second.data(), second.size()) << '\n';

  const std::array<emel::io::event::tensor_load_span, 0> empty{};
  emel::io::staged_read::event::staged_window_batch missing{empty, 1};
  if (actor.process_event(missing)) {
    throw std::runtime_error("missing callbacks did not fail");
  }
  output << "case=batch_missing_callbacks outcome=error error=invalid_callbacks"
            " failed_index=0 callback=false\n";
}

double median(std::vector<double> samples) {
  std::sort(samples.begin(), samples.end());
  const auto middle = samples.size() / 2;
  if ((samples.size() % 2) == 0) {
    return (samples[middle - 1] + samples[middle]) / 2.0;
  }
  return samples[middle];
}

double benchmark_case(const std::size_t copy_bytes,
                      const std::uint64_t iterations,
                      const std::size_t runs,
                      const std::uint64_t warmup_iterations) {
  emel::io::staged_read::sm actor{};
  single_owner owner{};
  std::vector<std::uint8_t> source(copy_bytes, 0xa5);
  std::vector<std::uint8_t> target(copy_bytes);
  const emel::io::staged_read::event::staged_window_request request{
      .file_offset = 0,
      .logical_byte_length = copy_bytes,
      .stage_chunk_bytes = 4096,
      .source_span = source.data(),
      .source_span_bytes = source.size(),
      .target_buffer = target.data(),
      .target_window_bytes = target.size(),
  };
  emel::io::staged_read::event::staged_window event{request};
  event.on_done = {&owner, on_single_done};
  event.on_error = {&owner, on_single_error};
  const auto operation = [&] {
    owner = {};
    if (!actor.process_event(event) || !owner.done || owner.failed ||
        owner.bytes_committed != copy_bytes) {
      throw std::runtime_error{"staged-read benchmark validation failed"};
    }
  };
  for (std::uint64_t index = 0; index < warmup_iterations; ++index) {
    operation();
  }
  std::vector<double> samples;
  samples.reserve(runs);
  for (std::size_t run = 0; run < runs; ++run) {
    const auto started = std::chrono::steady_clock::now();
    for (std::uint64_t index = 0; index < iterations; ++index) {
      operation();
    }
    const auto elapsed = std::chrono::duration<double, std::nano>(
                             std::chrono::steady_clock::now() - started)
                             .count();
    samples.push_back(elapsed / static_cast<double>(iterations));
  }
  if (target != source) {
    throw std::runtime_error{"staged-read benchmark target mismatch"};
  }
  return median(std::move(samples));
}

void benchmark(const std::uint64_t iterations, const std::size_t runs,
               const std::uint64_t warmup_iterations) {
  if (iterations == 0 || runs == 0) {
    throw std::runtime_error{"benchmark iterations and runs must be nonzero"};
  }
  std::cout << std::fixed << std::setprecision(3)
            << "io/staged-read/reference/copy_16kib ns_per_op="
            << benchmark_case(16'384, iterations, runs, warmup_iterations)
            << " iter=" << iterations << " runs=" << runs << '\n'
            << "io/staged-read/reference/copy_1mib ns_per_op="
            << benchmark_case(1'048'576, iterations, runs, warmup_iterations)
            << " iter=" << iterations << " runs=" << runs << '\n';
}

} // namespace

int main(const int argc, char **argv) {
  try {
    if (argc == 5 && std::string_view{argv[1]} == "--benchmark") {
      benchmark(std::stoull(argv[2]), std::stoull(argv[3]),
                std::stoull(argv[4]));
      return 0;
    }
    if (argc != 1) {
      std::cerr << "usage: emel-io-staged-read-reference\n"
                << "       emel-io-staged-read-reference --benchmark ITERATIONS RUNS WARMUP\n";
      return 2;
    }
    std::cout
        << "io-staged-read-parity-snapshot/v1\n"
        << "source_repository=stateforward/emel.cpp\n"
        << "source_commit=843a117386ef17dc5a50549bbfc821074c2141d6\n"
        << "source_tree=ff00a9978b00b4ea1e5268d2ecd483d4855b6eaa\n"
        << "source_files=src/emel/io/staged_read/events.hpp,errors.hpp,context.hpp,detail.hpp,guards.hpp,actions.hpp,sm.hpp\n"
        << "source_tests=tests/io/staged_read/lifecycle_tests.cpp\n"
        << "fixture_config=caller_owned_memory,platform_supported,synchronous_callbacks,single_aligned_and_remainder,batch_offsets,validation_precedence\n"
        << "contract_delta=rust_borrows_prove_non_null_targets_and_source_lengths\n";
    render_single_success(std::cout, "aligned", "abcdefgh", 4);
    render_single_success(std::cout, "remainder", "abcdefghij", 4);
    render_single_errors(std::cout);
    render_batch_success(std::cout);
    render_batch_errors(std::cout);
  } catch (const std::exception &error) {
    std::cerr << "error: " << error.what() << '\n';
    return 1;
  }
  return 0;
}
