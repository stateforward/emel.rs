#include <cstdint>
#include <cstring>

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#include "emel/io/mmap/errors.hpp"
#include "emel/io/mmap/events.hpp"
#include "emel/io/mmap/sm.hpp"
#include "emel/machines.hpp"

namespace {

using error_type = emel::error::type;
using mmap_error = emel::io::mmap::error;

struct map_outcome {
  bool done = false;
  bool failed = false;
  std::uint32_t handle = emel::io::mmap::k_invalid_mapping_handle;
  const void *buffer = nullptr;
  std::uint64_t bytes = 0;
  error_type error = emel::error::cast(mmap_error::none);
};

struct operation_outcome {
  bool done = false;
  bool failed = false;
  error_type error = emel::error::cast(mmap_error::none);
};

void on_map_done(void *object,
                 const emel::io::mmap::events::map_tensor_done &event) noexcept {
  auto &outcome = *static_cast<map_outcome *>(object);
  outcome.done = true;
  outcome.handle = event.handle;
  outcome.buffer = event.buffer;
  outcome.bytes = event.buffer_bytes;
}

void on_map_error(
    void *object,
    const emel::io::mmap::events::map_tensor_error &event) noexcept {
  auto &outcome = *static_cast<map_outcome *>(object);
  outcome.failed = true;
  outcome.error = event.err;
}

void on_release_done(
    void *object,
    const emel::io::mmap::events::release_mapping_done &) noexcept {
  static_cast<operation_outcome *>(object)->done = true;
}

void on_release_error(
    void *object,
    const emel::io::mmap::events::release_mapping_error &event) noexcept {
  auto &outcome = *static_cast<operation_outcome *>(object);
  outcome.failed = true;
  outcome.error = event.err;
}

void on_advise_done(
    void *object,
    const emel::io::mmap::events::advise_mapping_done &) noexcept {
  static_cast<operation_outcome *>(object)->done = true;
}

void on_advise_error(
    void *object,
    const emel::io::mmap::events::advise_mapping_error &event) noexcept {
  auto &outcome = *static_cast<operation_outcome *>(object);
  outcome.failed = true;
  outcome.error = event.err;
}

std::string error_name(error_type error) {
  if (error == emel::error::cast(mmap_error::invalid_request)) {
    return "invalid_request";
  }
  if (error == emel::error::cast(mmap_error::unsupported_platform)) {
    return "unsupported_platform";
  }
  if (error == emel::error::cast(mmap_error::unsupported_resource)) {
    return "unsupported_resource";
  }
  if (error == emel::error::cast(mmap_error::resource_exhausted)) {
    return "resource_exhausted";
  }
  if (error == emel::error::cast(mmap_error::file_open_failed)) {
    return "file_open_failed";
  }
  if (error == emel::error::cast(mmap_error::mapping_failed)) {
    return "mapping_failed";
  }
  if (error == emel::error::cast(mmap_error::unmap_failed)) {
    return "unmap_failed";
  }
  if (error == emel::error::cast(mmap_error::invalid_advise_range)) {
    return "invalid_advise_range";
  }
  if (error == emel::error::cast(mmap_error::advise_failed)) {
    return "advise_failed";
  }
  if (error == emel::error::cast(mmap_error::internal_error)) {
    return "internal_error";
  }
  return "unknown";
}

map_outcome map(emel::io::mmap::sm &actor, std::int32_t tensor_id,
                std::string_view path, std::uint16_t file_index,
                std::uint64_t offset, std::uint64_t bytes) {
  const emel::io::mmap::event::map_tensor_request request{
      .tensor_id = tensor_id,
      .file_index = file_index,
      .file_offset = offset,
      .byte_size = bytes,
      .file_path = path,
  };
  map_outcome outcome{};
  emel::io::mmap::event::map_tensor event{request};
  event.on_done = {&outcome, on_map_done};
  event.on_error = {&outcome, on_map_error};
  (void)actor.process_event(event);
  return outcome;
}

operation_outcome release(emel::io::mmap::sm &actor, std::int32_t tensor_id,
                          std::uint32_t handle) {
  operation_outcome outcome{};
  emel::io::mmap::event::release_mapping event{tensor_id, handle};
  event.on_done = {&outcome, on_release_done};
  event.on_error = {&outcome, on_release_error};
  (void)actor.process_event(event);
  return outcome;
}

operation_outcome advise(emel::io::mmap::sm &actor, std::int32_t tensor_id,
                         std::uint32_t handle, std::uint64_t offset,
                         std::uint64_t length,
                         emel::io::mmap::event::advice kind) {
  operation_outcome outcome{};
  emel::io::mmap::event::advise_mapping event{tensor_id, handle, offset,
                                               length, kind};
  event.on_done = {&outcome, on_advise_done};
  event.on_error = {&outcome, on_advise_error};
  (void)actor.process_event(event);
  return outcome;
}

std::uint64_t fnv1a64(const void *address, std::uint64_t bytes) {
  const auto *data = static_cast<const std::uint8_t *>(address);
  std::uint64_t hash = 0xcbf29ce484222325ULL;
  for (std::uint64_t index = 0; index < bytes; ++index) {
    hash = (hash ^ data[index]) * 0x00000100000001b3ULL;
  }
  return hash;
}

void print_map_error(std::string_view name, const map_outcome &outcome) {
  if (!outcome.failed) {
    throw std::runtime_error{"expected map failure"};
  }
  std::cout << "case=map_" << name << " outcome=error error="
            << error_name(outcome.error) << '\n';
}

void print_operation_error(std::string_view name,
                           const operation_outcome &outcome) {
  if (!outcome.failed) {
    throw std::runtime_error{"expected operation failure"};
  }
  std::cout << "case=" << name << " outcome=error error="
            << error_name(outcome.error) << '\n';
}

void render(std::string_view path) {
  constexpr std::uint64_t mapping_bytes = 16'384;
  std::cout
      << "io-mmap-parity-snapshot/v1\n"
      << "source_repository=stateforward/emel.cpp\n"
      << "source_commit=843a117386ef17dc5a50549bbfc821074c2141d6\n"
      << "source_tree=ff00a9978b00b4ea1e5268d2ecd483d4855b6eaa\n"
      << "source_files=src/emel/io/mmap/events.hpp,errors.hpp,context.hpp,detail.hpp,guards.hpp,actions.hpp,actions.cpp,sm.hpp\n"
      << "source_tests=tests/io/mmap/lifecycle_tests.cpp,tests/io/mmap/advise_tests.cpp\n"
      << "fixture_config=file_bytes=32768,pattern=incrementing_u8,map_bytes=16384,offset=0\n"
      << "native_semantics_complete=true\n"
      << "missing_native_semantics=none\n"
      << "contract_delta=rust_safe_access_event,rust_mmap_source_capability\n";

  emel::io::mmap::sm validation_actor{};
  print_map_error("invalid_zero",
                  map(validation_actor, 1, path, 0, 0, 0));
  print_map_error("invalid_empty_path",
                  map(validation_actor, 2, {}, 0, 0, mapping_bytes));
  print_map_error("unsupported_index",
                  map(validation_actor, 3, path, UINT16_MAX, 0,
                      mapping_bytes));
  print_map_error("unsupported_offset",
                  map(validation_actor, 4, path, 0, 1, mapping_bytes));
  print_map_error("unsupported_length",
                  map(validation_actor, 5, path, 0, 0, (1ULL << 40) + 1));
  print_map_error("unsupported_layout",
                  map(validation_actor, 6, path, 0, UINT64_MAX - 16'383,
                      mapping_bytes));
  print_map_error("file_open",
                  map(validation_actor, 7, std::string{path} + ".missing", 0,
                      0, mapping_bytes));
  print_map_error("past_eof",
                  map(validation_actor, 8, path, 0, 16'384, 32'768));

  emel::io::mmap::sm actor{};
  const map_outcome mapped = map(actor, 10, path, 0, 0, mapping_bytes);
  if (!mapped.done || mapped.buffer == nullptr || mapped.bytes != mapping_bytes) {
    throw std::runtime_error{"expected successful map"};
  }
  std::cout << "case=map_success outcome=done tensor_id=10 handle="
            << mapped.handle << " bytes=" << mapped.bytes << " checksum="
            << std::hex << std::setw(16) << std::setfill('0')
            << fnv1a64(mapped.buffer, mapped.bytes) << std::dec << '\n';

  for (const auto [name, kind] : {
           std::pair{"sequential",
                     emel::io::mmap::event::advice::k_sequential},
           std::pair{"will_need", emel::io::mmap::event::advice::k_willneed},
           std::pair{"dont_need", emel::io::mmap::event::advice::k_dontneed},
       }) {
    const auto outcome = advise(actor, 10, mapped.handle, 24, 1'000, kind);
    if (!outcome.done) {
      throw std::runtime_error{"expected successful advice"};
    }
    std::cout << "case=advise_" << name << " outcome=done\n";
  }
  print_operation_error(
      "advise_wrong_owner",
      advise(actor, 11, mapped.handle, 0, 64,
             emel::io::mmap::event::advice::k_willneed));
  print_operation_error(
      "advise_invalid_range",
      advise(actor, 10, mapped.handle, mapping_bytes - 1, 2,
             emel::io::mmap::event::advice::k_willneed));
  print_operation_error("release_wrong_owner",
                        release(actor, 11, mapped.handle));
  const auto released = release(actor, 10, mapped.handle);
  if (!released.done) {
    throw std::runtime_error{"expected successful release"};
  }
  std::cout << "case=release_success outcome=done\n";
  print_operation_error("release_again", release(actor, 10, mapped.handle));
}

bool run_benchmark_lifecycle(emel::io::mmap::sm &actor,
                             std::string_view path,
                             std::uint64_t mapping_bytes,
                             std::uint64_t expected_checksum) {
  const auto mapped = map(actor, 71, path, 0, 0, mapping_bytes);
  if (!mapped.done || mapped.bytes != mapping_bytes || mapped.buffer == nullptr ||
      fnv1a64(mapped.buffer, mapped.bytes) != expected_checksum) {
    return false;
  }
  for (const auto kind : {
           emel::io::mmap::event::advice::k_sequential,
           emel::io::mmap::event::advice::k_willneed,
           emel::io::mmap::event::advice::k_dontneed,
       }) {
    const std::uint64_t length =
        kind == emel::io::mmap::event::advice::k_sequential ? mapping_bytes
                                                            : 4'096;
    if (!advise(actor, 71, mapped.handle, 0, length, kind).done) {
      return false;
    }
  }
  return release(actor, 71, mapped.handle).done;
}

double measure_benchmark_case(emel::io::mmap::sm &actor,
                              std::string_view path,
                              std::uint64_t mapping_bytes,
                              std::uint64_t expected_checksum,
                              std::uint64_t iterations, std::size_t runs,
                              std::uint64_t warmup_iterations) {
  for (std::uint64_t iteration = 0; iteration < warmup_iterations;
       ++iteration) {
    if (!run_benchmark_lifecycle(actor, path, mapping_bytes,
                                 expected_checksum)) {
      throw std::runtime_error{"mmap benchmark warmup validation failed"};
    }
  }
  std::vector<double> samples;
  samples.reserve(runs);
  for (std::size_t run = 0; run < runs; ++run) {
    const auto started = std::chrono::steady_clock::now();
    for (std::uint64_t iteration = 0; iteration < iterations; ++iteration) {
      if (!run_benchmark_lifecycle(actor, path, mapping_bytes,
                                   expected_checksum)) {
        throw std::runtime_error{"mmap benchmark validation failed"};
      }
    }
    const auto elapsed = std::chrono::duration<double, std::nano>(
        std::chrono::steady_clock::now() - started);
    samples.push_back(elapsed.count() / static_cast<double>(iterations));
  }
  std::sort(samples.begin(), samples.end());
  const std::size_t middle = samples.size() / 2;
  const double median = samples.size() % 2 == 0
                            ? (samples[middle - 1] + samples[middle]) / 2.0
                            : samples[middle];
  return median;
}

void benchmark(std::string_view path, std::uint64_t iterations,
               std::size_t runs, std::uint64_t warmup_iterations) {
  constexpr std::uint64_t bytes_16kib = 16'384;
  constexpr std::uint64_t bytes_1mib = 1'048'576;
  if (iterations == 0 || runs == 0) {
    throw std::runtime_error{"benchmark iterations and runs must be nonzero"};
  }
  std::ifstream input{std::string{path}, std::ios::binary};
  const std::vector<std::uint8_t> fixture{
      std::istreambuf_iterator<char>{input}, std::istreambuf_iterator<char>{}};
  if (fixture.size() != bytes_1mib) {
    throw std::runtime_error{"mmap benchmark fixture size mismatch"};
  }
  const auto checksum_16kib = fnv1a64(fixture.data(), bytes_16kib);
  const auto checksum_1mib = fnv1a64(fixture.data(), bytes_1mib);
  emel::io::mmap::sm actor{};
  const double timing_16kib = measure_benchmark_case(
      actor, path, bytes_16kib, checksum_16kib, iterations, runs,
      warmup_iterations);
  const double timing_1mib = measure_benchmark_case(
      actor, path, bytes_1mib, checksum_1mib, iterations, runs,
      warmup_iterations);
  std::cout << std::fixed << std::setprecision(3)
            << "io/mmap/reference/lifecycle_16kib ns_per_op="
            << timing_16kib << " iter=" << iterations << " runs=" << runs
            << '\n'
            << "io/mmap/reference/lifecycle_1mib ns_per_op=" << timing_1mib
            << " iter=" << iterations << " runs=" << runs << '\n';
}

} // namespace

int main(int argc, char **argv) {
  try {
    if (argc == 2) {
      render(argv[1]);
      return 0;
    }
    if (argc == 6 && std::string_view{argv[1]} == "--benchmark") {
      benchmark(argv[2], std::stoull(argv[3]),
                static_cast<std::size_t>(std::stoull(argv[4])),
                std::stoull(argv[5]));
      return 0;
    }
    std::cerr << "usage: emel-io-mmap-reference FILE\n"
              << "       emel-io-mmap-reference --benchmark FILE ITERATIONS RUNS WARMUP\n";
    return 2;
  } catch (const std::exception &error) {
    std::cerr << "emel-io-mmap-reference: " << error.what() << '\n';
    return 1;
  }
  return 1;
}
