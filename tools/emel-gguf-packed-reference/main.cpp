#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <span>
#include <string>
#include <vector>

#include "emel/gguf/loader/events.hpp"
#include "emel/gguf/loader/sm.hpp"
#include "emel/model/data.hpp"

namespace {

struct callback_state {
  bool probe_done = false;
  bool probe_error = false;
  bool bind_done = false;
  bool bind_error = false;
  bool parse_done = false;
  bool parse_error = false;
};

callback_state *g_state = nullptr;

void on_probe_done(const emel::gguf::loader::events::probe_done &) {
  g_state->probe_done = true;
}

void on_probe_error(const emel::gguf::loader::events::probe_error &) {
  g_state->probe_error = true;
}

void on_bind_done(const emel::gguf::loader::events::bind_done &) {
  g_state->bind_done = true;
}

void on_bind_error(const emel::gguf::loader::events::bind_error &) {
  g_state->bind_error = true;
}

void on_parse_done(const emel::gguf::loader::events::parse_done &) {
  g_state->parse_done = true;
}

void on_parse_error(const emel::gguf::loader::events::parse_error &) {
  g_state->parse_error = true;
}

std::vector<uint8_t> read_file(const char *path) {
  std::ifstream input{path, std::ios::binary | std::ios::ate};
  if (!input) {
    return {};
  }
  const auto end = input.tellg();
  if (end <= 0) {
    return {};
  }
  std::vector<uint8_t> bytes(static_cast<size_t>(end));
  input.seekg(0);
  input.read(reinterpret_cast<char *>(bytes.data()), end);
  return input ? bytes : std::vector<uint8_t>{};
}

uint64_t fnv1a64(const std::span<const uint8_t> bytes) {
  uint64_t hash = 0xcbf29ce484222325ULL;
  for (const uint8_t byte : bytes) {
    hash = (hash ^ byte) * 0x00000100000001b3ULL;
  }
  return hash;
}

void write_hex(const std::span<const uint8_t> bytes) {
  for (const uint8_t byte : bytes) {
    std::cout << std::hex << std::setfill('0') << std::setw(2)
              << static_cast<unsigned>(byte);
  }
  std::cout << std::dec;
}

bool load(const std::span<const uint8_t> file,
          emel::gguf::loader::requirements &requirements,
          std::vector<emel::model::data::tensor_record> &tensors) {
  emel::gguf::loader::sm loader{};
  callback_state state{};
  g_state = &state;
  const auto probe_done =
      emel::gguf::loader::event::probe_done_fn::from<&on_probe_done>();
  const auto probe_error =
      emel::gguf::loader::event::probe_error_fn::from<&on_probe_error>();
  const auto bind_done =
      emel::gguf::loader::event::bind_done_fn::from<&on_bind_done>();
  const auto bind_error =
      emel::gguf::loader::event::bind_error_fn::from<&on_bind_error>();
  const auto parse_done =
      emel::gguf::loader::event::parse_done_fn::from<&on_parse_done>();
  const auto parse_error =
      emel::gguf::loader::event::parse_error_fn::from<&on_parse_error>();

  const emel::gguf::loader::event::probe probe{file, requirements,
                                                probe_done, probe_error};
  if (!loader.process_event(probe) || !state.probe_done || state.probe_error ||
      requirements.kv_count != 0) {
    std::cerr << "packed reference probe failed done=" << state.probe_done
              << " error=" << state.probe_error
              << " kv_count=" << requirements.kv_count << '\n';
    return false;
  }

  tensors.resize(requirements.tensor_count);
  // The pinned public loader requires non-null bound spans even when the
  // probed metadata count is zero; one inert slot satisfies that public API.
  std::vector<uint8_t> arena(1);
  std::vector<emel::gguf::loader::kv_entry> entries(1);
  const emel::gguf::loader::event::bind_storage bind{
      std::span<uint8_t>{arena}, std::span<emel::gguf::loader::kv_entry>{entries},
      std::span<emel::model::data::tensor_record>{tensors}, bind_done, bind_error};
  if (!loader.process_event(bind) || !state.bind_done || state.bind_error) {
    std::cerr << "packed reference bind failed done=" << state.bind_done
              << " error=" << state.bind_error << '\n';
    return false;
  }

  const emel::gguf::loader::event::parse parse{file, parse_done, parse_error};
  const bool result = loader.process_event(parse);
  if (!result || !state.parse_done || state.parse_error) {
    std::cerr << "packed reference parse failed result=" << result
              << " done=" << state.parse_done << " error=" << state.parse_error
              << '\n';
    return false;
  }
  return true;
}

} // namespace

int main(int argc, char **argv) {
  if (argc != 2) {
    return 2;
  }
  const auto file = read_file(argv[1]);
  emel::gguf::loader::requirements requirements{};
  std::vector<emel::model::data::tensor_record> tensors;
  if (file.empty() || !load(std::span<const uint8_t>{file}, requirements, tensors)) {
    std::cout << "gguf-parity/v1\nstatus=error\n";
    return 0;
  }

  std::cout << "gguf-parity/v1\nstatus=ok\nversion=3\nalignment=32\n"
            << "kv_count=0\ntensor_count=" << requirements.tensor_count << '\n';
  for (size_t index = 0; index < tensors.size(); ++index) {
    const auto &tensor = tensors[index];
    std::cout << "tensor." << index << ".name=";
    write_hex(std::span<const uint8_t>{file}.subspan(tensor.name_offset,
                                                     tensor.name_length));
    std::cout << '\n'
              << "tensor." << index << ".type=" << tensor.type << '\n'
              << "tensor." << index << ".dims=" << tensor.dims[0] << ','
              << tensor.dims[1] << ',' << tensor.dims[2] << ',' << tensor.dims[3]
              << '\n'
              << "tensor." << index << ".offset=" << tensor.data_offset << '\n'
              << "tensor." << index << ".size=" << tensor.data_size << '\n'
              << "tensor." << index << ".hash=" << std::hex << std::setfill('0')
              << std::setw(16)
              << fnv1a64(std::span<const uint8_t>{
                     static_cast<const uint8_t *>(tensor.data),
                     static_cast<size_t>(tensor.data_size)})
              << std::dec << '\n';
  }
  return 0;
}
