#include <array>
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
#include "emel/io/read/errors.hpp"
#include "emel/io/read/events.hpp"
#include "emel/io/read/sm.hpp"

namespace {

using read_error = emel::io::read::error;

struct read_owner {
  bool done = false;
  bool failed = false;
  std::int32_t tensor_id = 0;
  std::uint64_t bytes_copied = 0;
  emel::error::type error = emel::error::cast(read_error::none);
};

struct batch_owner {
  bool done = false;
  bool failed = false;
  std::uint32_t done_count = 0;
  std::uint64_t bytes_copied = 0;
  std::uint32_t failed_index = 0;
  emel::error::type error = emel::error::cast(read_error::none);
};

void on_read_done(
    void *object,
    const emel::io::read::events::read_tensor_done &event) noexcept {
  auto &owner = *static_cast<read_owner *>(object);
  owner.done = true;
  owner.tensor_id = event.request.request.tensor_id;
  owner.bytes_copied = event.bytes_copied;
}

void on_read_error(
    void *object,
    const emel::io::read::events::read_tensor_error &event) noexcept {
  auto &owner = *static_cast<read_owner *>(object);
  owner.failed = true;
  owner.error = event.err;
}

void on_batch_done(
    void *object,
    const emel::io::read::events::read_tensor_batch_done &event) noexcept {
  auto &owner = *static_cast<batch_owner *>(object);
  owner.done = true;
  owner.done_count = event.done_count;
  owner.bytes_copied = event.bytes_copied;
}

void on_batch_error(
    void *object,
    const emel::io::read::events::read_tensor_batch_error &event) noexcept {
  auto &owner = *static_cast<batch_owner *>(object);
  owner.failed = true;
  owner.error = event.err;
  owner.failed_index = event.failed_index;
}

std::string error_name(const emel::error::type error) {
  if (error == emel::error::cast(read_error::invalid_request)) {
    return "invalid_request";
  }
  if (error == emel::error::cast(read_error::unsupported_platform)) {
    return "unsupported_platform";
  }
  if (error == emel::error::cast(read_error::unsupported_resource)) {
    return "unsupported_resource";
  }
  if (error == emel::error::cast(read_error::file_open_failed)) {
    return "file_open_failed";
  }
  if (error == emel::error::cast(read_error::file_seek_failed)) {
    return "file_seek_failed";
  }
  if (error == emel::error::cast(read_error::file_read_failed)) {
    return "file_read_failed";
  }
  if (error == emel::error::cast(read_error::short_read)) {
    return "short_read";
  }
  if (error == emel::error::cast(read_error::internal_error)) {
    return "internal_error";
  }
  return "unknown";
}

template <std::size_t Size>
std::string hex(const std::array<std::uint8_t, Size> &bytes) {
  std::ostringstream output;
  output << std::hex << std::setfill('0');
  for (const auto byte : bytes) {
    output << std::setw(2) << static_cast<unsigned>(byte);
  }
  return output.str();
}

void render_single_success(std::ostream &output) {
  emel::io::read::sm reader{};
  read_owner owner{};
  constexpr std::string_view source = "abcdef";
  std::array<std::uint8_t, 4> target{};
  emel::io::read::event::read_tensor_request request{
      .tensor_id = 7,
      .file_index = 0,
      .file_offset = 1,
      .byte_size = 4,
      .file_path = "tensor.bin",
      .source_buffer = source.data(),
      .source_buffer_bytes = source.size(),
      .source_error = emel::error::cast(read_error::none),
      .target_buffer = target.data(),
      .target_buffer_bytes = target.size(),
  };
  emel::io::read::event::read_tensor event{request};
  event.on_done = {&owner, on_read_done};
  event.on_error = {&owner, on_read_error};
  if (!reader.process_event(event) || !owner.done || owner.failed) {
    throw std::runtime_error("single_success did not complete");
  }
  output << "case=single_success outcome=done tensor_id=" << owner.tensor_id
         << " bytes_copied=" << owner.bytes_copied << " target=" << hex(target)
         << " callback_tensor_id=" << owner.tensor_id
         << " callback_bytes_copied=" << owner.bytes_copied << '\n';
}

struct single_error_case {
  std::string_view name;
  std::string_view path = "tensor.bin";
  const void *source = "abcd";
  std::uint64_t source_bytes = 4;
  emel::error::type source_error = emel::error::cast(read_error::none);
  std::uint16_t file_index = 0;
  std::uint64_t offset = 0;
  std::uint64_t byte_size = 4;
};

void render_single_error(std::ostream &output, const single_error_case &spec) {
  emel::io::read::sm reader{};
  read_owner owner{};
  std::array<std::uint8_t, 4> target{};
  emel::io::read::event::read_tensor_request request{
      .tensor_id = 1,
      .file_index = spec.file_index,
      .file_offset = spec.offset,
      .byte_size = spec.byte_size,
      .file_path = spec.path,
      .source_buffer = spec.source,
      .source_buffer_bytes = spec.source_bytes,
      .source_error = spec.source_error,
      .target_buffer = target.data(),
      .target_buffer_bytes = target.size(),
  };
  emel::io::read::event::read_tensor event{request};
  event.on_done = {&owner, on_read_done};
  event.on_error = {&owner, on_read_error};
  if (reader.process_event(event) || owner.done || !owner.failed) {
    throw std::runtime_error("single_" + std::string{spec.name} +
                             " did not fail");
  }
  output << "case=single_" << spec.name << " outcome=error error="
         << error_name(owner.error) << " callback_error=" << error_name(owner.error)
         << " target=" << hex(target) << '\n';
}

void render_single_errors(std::ostream &output) {
  render_single_error(output, {.name = "invalid_zero", .byte_size = 0});
  render_single_error(output,
                      {.name = "invalid_path",
                       .path = std::string_view{"bad\0path", 8}});
  render_single_error(
      output, {.name = "unsupported_index", .file_index = UINT16_MAX});
  render_single_error(output,
                      {.name = "unsupported_length",
                       .byte_size = (std::uint64_t{1} << 40) + 1});
  render_single_error(output,
                      {.name = "unsupported_layout",
                       .offset = UINT64_MAX - 1,
                       .byte_size = 4});
  render_single_error(output, {.name = "invalid_target", .byte_size = 5});
  render_single_error(output, {.name = "file_open", .source = nullptr});
  render_single_error(output, {.name = "file_seek", .offset = 5});
  render_single_error(
      output,
      {.name = "file_read", .source_error = emel::error::type{1u << 15}});
  render_single_error(output,
                      {.name = "short_read", .source = "ab", .source_bytes = 2});
}

void render_batch_success(std::ostream &output) {
  emel::io::read::sm reader{};
  batch_owner owner{};
  constexpr std::string_view first_source = "abcdef";
  constexpr std::string_view second_source = "wxyz";
  std::array<std::uint8_t, 3> first_target{};
  std::array<std::uint8_t, 4> second_target{};
  const std::array tensors{
      emel::io::event::tensor_load_span{
          .tensor_id = 1,
          .file_index = 0,
          .file_offset = 2,
          .byte_size = 3,
          .file_path = "first.bin",
          .source_buffer = first_source.data(),
          .source_buffer_bytes = first_source.size(),
          .source_error = emel::error::cast(read_error::none),
          .target = first_target.data(),
          .target_bytes = first_target.size(),
      },
      emel::io::event::tensor_load_span{
          .tensor_id = 2,
          .file_index = 0,
          .file_offset = 0,
          .byte_size = 4,
          .file_path = "second.bin",
          .source_buffer = second_source.data(),
          .source_buffer_bytes = second_source.size(),
          .source_error = emel::error::cast(read_error::none),
          .target = second_target.data(),
          .target_bytes = second_target.size(),
      },
  };
  emel::io::read::event::read_tensor_batch event{tensors};
  event.on_done = {&owner, on_batch_done};
  event.on_error = {&owner, on_batch_error};
  if (!reader.process_event(event) || !owner.done || owner.failed) {
    throw std::runtime_error("batch_success did not complete");
  }
  output << "case=batch_success outcome=done done_count=" << owner.done_count
         << " bytes_copied=" << owner.bytes_copied
         << " first_target=" << hex(first_target)
         << " second_target=" << hex(second_target)
         << " callback_done_count=" << owner.done_count
         << " callback_bytes_copied=" << owner.bytes_copied << '\n';
}

enum class batch_case {
  invalid_request,
  unsupported_resource,
  file_open,
  file_seek,
  file_read,
  short_read,
  mixed_short_then_file_read,
  mixed_seek_then_open,
  mixed_resource_then_invalid,
  same_phase_two_file_reads,
};

std::string_view batch_name(const batch_case value) {
  switch (value) {
  case batch_case::invalid_request:
    return "invalid_request";
  case batch_case::unsupported_resource:
    return "unsupported_resource";
  case batch_case::file_open:
    return "file_open";
  case batch_case::file_seek:
    return "file_seek";
  case batch_case::file_read:
    return "file_read";
  case batch_case::short_read:
    return "short_read";
  case batch_case::mixed_short_then_file_read:
    return "mixed_short_then_file_read";
  case batch_case::mixed_seek_then_open:
    return "mixed_seek_then_open";
  case batch_case::mixed_resource_then_invalid:
    return "mixed_resource_then_invalid";
  case batch_case::same_phase_two_file_reads:
    return "same_phase_two_file_reads";
  }
  return "unknown";
}

void render_batch_error(std::ostream &output, const batch_case selected_case) {
  emel::io::read::sm reader{};
  batch_owner owner{};
  constexpr std::string_view source = "abcd";
  constexpr std::string_view short_source = "ab";
  std::array<std::uint8_t, 4> first_target{};
  std::array<std::uint8_t, 4> second_target{};
  std::array tensors{
      emel::io::event::tensor_load_span{
          .tensor_id = 1,
          .file_index = 0,
          .file_offset = 0,
          .byte_size = 4,
          .file_path = "first.bin",
          .source_buffer = source.data(),
          .source_buffer_bytes = source.size(),
          .source_error = emel::error::cast(read_error::none),
          .target = first_target.data(),
          .target_bytes = first_target.size(),
      },
      emel::io::event::tensor_load_span{
          .tensor_id = 2,
          .file_index = 0,
          .file_offset = 0,
          .byte_size = 4,
          .file_path = "second.bin",
          .source_buffer = source.data(),
          .source_buffer_bytes = source.size(),
          .source_error = emel::error::cast(read_error::none),
          .target = second_target.data(),
          .target_bytes = second_target.size(),
      },
  };
  switch (selected_case) {
  case batch_case::invalid_request:
    tensors[1].byte_size = 5;
    break;
  case batch_case::unsupported_resource:
    tensors[1].file_path = std::string_view{"bad\0path", 8};
    break;
  case batch_case::file_open:
    tensors[1].source_buffer = nullptr;
    break;
  case batch_case::file_seek:
    tensors[1].file_offset = 5;
    break;
  case batch_case::file_read:
    tensors[1].source_error = emel::error::type{1u << 15};
    break;
  case batch_case::short_read:
    tensors[1].source_buffer = short_source.data();
    tensors[1].source_buffer_bytes = short_source.size();
    break;
  case batch_case::mixed_short_then_file_read:
    tensors[0].source_buffer = short_source.data();
    tensors[0].source_buffer_bytes = short_source.size();
    tensors[1].source_error = emel::error::type{1u << 15};
    break;
  case batch_case::mixed_seek_then_open:
    tensors[0].file_offset = 5;
    tensors[1].source_buffer = nullptr;
    break;
  case batch_case::mixed_resource_then_invalid:
    tensors[0].file_path = std::string_view{"bad\0path", 8};
    tensors[1].byte_size = 5;
    break;
  case batch_case::same_phase_two_file_reads:
    tensors[0].source_error = emel::error::type{1u << 15};
    tensors[1].source_error = emel::error::type{1u << 15};
    break;
  }
  emel::io::read::event::read_tensor_batch event{tensors};
  event.on_done = {&owner, on_batch_done};
  event.on_error = {&owner, on_batch_error};
  if (reader.process_event(event) || owner.done || !owner.failed) {
    throw std::runtime_error("batch error case did not fail");
  }
  output << "case=batch_" << batch_name(selected_case)
         << " outcome=error error=" << error_name(owner.error)
         << " failed_index=" << owner.failed_index
         << " callback_error=" << error_name(owner.error)
         << " callback_failed_index=" << owner.failed_index
         << " first_target=" << hex(first_target)
         << " second_target=" << hex(second_target) << '\n';
}

void render_batch_count_boundaries(std::ostream &output) {
  emel::io::read::sm reader{};
  batch_owner owner{};
  constexpr std::string_view source = "x";
  std::array<std::uint8_t, 1> target{};
  std::vector<emel::io::event::tensor_load_span> tensors(
      emel::io::read::k_max_read_batch_tensors);
  for (auto &tensor : tensors) {
    tensor = {
        .tensor_id = 21,
        .file_index = 0,
        .file_offset = 0,
        .byte_size = 1,
        .file_path = "batch-cap.bin",
        .source_buffer = source.data(),
        .source_buffer_bytes = source.size(),
        .source_error = emel::error::cast(read_error::none),
        .target = target.data(),
        .target_bytes = target.size(),
    };
  }
  emel::io::read::event::read_tensor_batch exact_cap{tensors};
  exact_cap.on_done = {&owner, on_batch_done};
  exact_cap.on_error = {&owner, on_batch_error};
  if (!reader.process_event(exact_cap) || !owner.done || owner.failed) {
    throw std::runtime_error("batch exact-cap case did not complete");
  }
  output << "case=batch_count_exact_cap outcome=done done_count="
         << owner.done_count << " bytes_copied=" << owner.bytes_copied
         << " target=" << hex(target)
         << " callback_done_count=" << owner.done_count
         << " callback_bytes_copied=" << owner.bytes_copied << '\n';

  owner = {};
  std::array<std::uint8_t, 1> over_cap_target{};
  std::vector<emel::io::event::tensor_load_span> over_cap(
      static_cast<std::size_t>(emel::io::read::k_max_read_batch_tensors) + 1u);
  for (auto &tensor : over_cap) {
    tensor = {
        .tensor_id = 22,
        .file_index = 0,
        .file_offset = 0,
        .byte_size = 1,
        .file_path = "batch-over-cap.bin",
        .source_buffer = source.data(),
        .source_buffer_bytes = source.size(),
        .source_error = emel::error::cast(read_error::none),
        .target = over_cap_target.data(),
        .target_bytes = over_cap_target.size(),
    };
  }
  emel::io::read::event::read_tensor_batch over_cap_event{over_cap};
  over_cap_event.on_done = {&owner, on_batch_done};
  over_cap_event.on_error = {&owner, on_batch_error};
  if (reader.process_event(over_cap_event) || owner.done || !owner.failed) {
    throw std::runtime_error("batch over-cap case did not fail");
  }
  output << "case=batch_count_over_cap outcome=error error="
         << error_name(owner.error) << " failed_index=" << owner.failed_index
         << " callback_error=" << error_name(owner.error)
         << " callback_failed_index=" << owner.failed_index << '\n';
}

void render_manifest(std::ostream &output) {
  output
      << "io-read-parity-snapshot/v1\n"
      << "source_repository=stateforward/emel.cpp\n"
      << "source_commit=843a117386ef17dc5a50549bbfc821074c2141d6\n"
      << "source_tree=ff00a9978b00b4ea1e5268d2ecd483d4855b6eaa\n"
      << "source_files=src/emel/io/read/events.hpp,errors.hpp,context.hpp,detail.hpp,guards.hpp,actions.hpp,sm.hpp\n"
      << "source_tests=tests/io/read/lifecycle_tests.cpp\n"
      << "fixture_config=caller_owned_memory,platform_supported,synchronous_callbacks,distinct_batch_targets,mixed_batch_phase_precedence,batch_count_boundaries\n"
      << "contract_delta=rust_typed_results_make_callbacks_optional\n";
  render_single_success(output);
  render_single_errors(output);
  render_batch_success(output);
  for (const auto selected_case : {batch_case::invalid_request,
                                   batch_case::unsupported_resource,
                                   batch_case::file_open,
                                   batch_case::file_seek,
                                   batch_case::file_read,
                                   batch_case::short_read,
                                   batch_case::mixed_short_then_file_read,
                                   batch_case::mixed_seek_then_open,
                                   batch_case::mixed_resource_then_invalid,
                                   batch_case::same_phase_two_file_reads}) {
    render_batch_error(output, selected_case);
  }
  render_batch_count_boundaries(output);
}

} // namespace

int main() {
  try {
    render_manifest(std::cout);
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "emel-io-read-reference: " << error.what() << '\n';
    return 1;
  }
}
