#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "emel/gguf/loader/events.hpp"
#include "emel/gguf/loader/sm.hpp"
#include "emel/model/data.hpp"
#include "emel/model/generation/any.hpp"

namespace {

using model_data = emel::model::data;
using tensor_record = model_data::tensor_record;

struct fixture {
  std::unique_ptr<model_data> model = std::make_unique<model_data>();
};

void require(bool condition, std::string_view message) {
  if (!condition) {
    throw std::runtime_error(std::string{message});
  }
}

void append(fixture &value, std::span<const std::uint8_t> name,
            std::int32_t type, std::int32_t dimensions,
            std::array<std::int64_t, 4> shape, std::uint64_t data_size,
            bool bound) {
  auto &model = *value.model;
  require(model.n_tensors < model.tensors.size(), "tensor capacity");
  require(model.name_bytes_used + name.size() <= model.name_storage.size(),
          "name capacity");
  auto &record = model.tensors[model.n_tensors++];
  record.name_offset = model.name_bytes_used;
  record.name_length = static_cast<std::uint32_t>(name.size());
  std::copy(name.begin(), name.end(),
            model.name_storage.begin() + model.name_bytes_used);
  model.name_bytes_used += static_cast<std::uint32_t>(name.size());
  record.type = type;
  record.n_dims = dimensions;
  record.dims = shape;
  record.data_size = data_size;
  record.data = bound ? value.model.get() : nullptr;
}

void append(fixture &value, std::string_view name, std::int32_t type = 0,
            std::int32_t dimensions = 2,
            std::array<std::int64_t, 4> shape = {4, 8, 1, 1},
            std::uint64_t data_size = 128, bool bound = true) {
  append(value,
         std::span{reinterpret_cast<const std::uint8_t *>(name.data()),
                   name.size()},
         type, dimensions, shape, data_size, bound);
}

const tensor_record *find_first(const model_data &model,
                                std::span<const std::uint8_t> name) {
  const std::string_view target{
      reinterpret_cast<const char *>(name.data()), name.size()};
  for (std::uint32_t index = 0; index < model.n_tensors; ++index) {
    if (emel::model::tensor_name_view(model, model.tensors[index]) == target) {
      return &model.tensors[index];
    }
  }
  return nullptr;
}

bool bindable(const model_data &model, std::span<const std::uint8_t> name) {
  emel::model::generation::tensor_view view{};
  return emel::model::generation::bind_tensor_view(
      model,
      std::string_view{reinterpret_cast<const char *>(name.data()), name.size()},
      view);
}

std::string hex(std::span<const std::uint8_t> bytes) {
  static constexpr char digits[] = "0123456789abcdef";
  std::string result(bytes.size() * 2, '0');
  for (std::size_t index = 0; index < bytes.size(); ++index) {
    result[index * 2] = digits[bytes[index] >> 4];
    result[index * 2 + 1] = digits[bytes[index] & 0x0f];
  }
  return result;
}

void print_case(std::string_view label, const fixture &value,
                std::span<const std::uint8_t> name) {
  const auto *record = find_first(*value.model, name);
  std::cout << "source_case=" << label << " present=" << (record != nullptr)
            << " bindable=" << bindable(*value.model, name);
  if (record != nullptr) {
    const auto canonical = emel::model::tensor_name_view(*value.model, *record);
    std::cout << " type=" << record->type << " dims=" << record->n_dims
              << " data_size=" << record->data_size << " name_hex="
              << hex(std::span{
                     reinterpret_cast<const std::uint8_t *>(canonical.data()),
                     canonical.size()});
  }
  std::cout << '\n';
}

fixture parity_fixture() {
  fixture value{};
  append(value, "same");
  append(value, "same", 1, 1, {99, 1, 1, 1}, 42, true);
  constexpr std::array<std::uint8_t, 3> arbitrary{0xff, 0, 'x'};
  append(value, arbitrary, 0, 2, {4, 8, 1, 1}, 128, true);
  append(value, "");
  append(value, "unbound", 0, 1, {1, 1, 1, 1}, 1, false);
  append(value, "zero-size", 0, 1, {1, 1, 1, 1}, 0, true);
  append(value, "five-good", 0, 5, {1, 2, 3, 4}, 1, true);
  return value;
}

void parity() {
  const auto value = parity_fixture();
  constexpr std::array<std::uint8_t, 3> arbitrary{0xff, 0, 'x'};
  print_case("first_duplicate", value,
             std::span<const std::uint8_t>{
                 reinterpret_cast<const std::uint8_t *>("same"), 4});
  print_case("arbitrary_nul", value, arbitrary);
  print_case("empty", value, std::span<const std::uint8_t>{});
  print_case("unbound", value,
             std::span<const std::uint8_t>{
                 reinterpret_cast<const std::uint8_t *>("unbound"), 7});
  print_case("zero_size", value,
             std::span<const std::uint8_t>{
                 reinterpret_cast<const std::uint8_t *>("zero-size"), 9});
  print_case("five_dims", value,
             std::span<const std::uint8_t>{
                 reinterpret_cast<const std::uint8_t *>("five-good"), 9});
  print_case("missing", value,
             std::span<const std::uint8_t>{
                 reinterpret_cast<const std::uint8_t *>("missing"), 7});

  tensor_record malformed{};
  malformed.name_offset = UINT32_MAX;
  malformed.name_length = UINT32_MAX;
  std::cout << "reference_case=malformed_range canonical_bytes="
            << emel::model::tensor_name_view(*value.model, malformed).size()
            << '\n';
}

double median(std::vector<double> samples) {
  std::sort(samples.begin(), samples.end());
  return samples[samples.size() / 2];
}

void probe_done(const emel::gguf::loader::events::probe_done &) {}
void probe_error(const emel::gguf::loader::events::probe_error &event) {
  throw std::runtime_error{"GGUF probe error " + std::to_string(event.err)};
}
void bind_done(const emel::gguf::loader::events::bind_done &) {}
void bind_error(const emel::gguf::loader::events::bind_error &event) {
  throw std::runtime_error{"GGUF bind error " + std::to_string(event.err)};
}
void parse_done(const emel::gguf::loader::events::parse_done &) {}
void parse_error(const emel::gguf::loader::events::parse_error &event) {
  throw std::runtime_error{"GGUF parse error " + std::to_string(event.err)};
}

struct parsed_fixture {
  std::vector<std::uint8_t> bytes{};
  std::vector<std::uint8_t> arena{};
  std::vector<emel::gguf::loader::kv_entry> entries{};
  std::unique_ptr<model_data> model = std::make_unique<model_data>();
};

parsed_fixture parse_fixture(const std::filesystem::path &path) {
  parsed_fixture output{};
  std::ifstream stream(path, std::ios::binary | std::ios::ate);
  require(stream.good(), "open real fixture");
  const auto end = stream.tellg();
  require(end > 0, "nonempty real fixture");
  output.bytes.resize(static_cast<std::size_t>(end));
  stream.seekg(0, std::ios::beg);
  stream.read(reinterpret_cast<char *>(output.bytes.data()), end);
  require(stream.good(), "read real fixture");

  emel::gguf::loader::sm loader{};
  emel::gguf::loader::requirements requirements{};
  const auto on_probe_done =
      emel::gguf::loader::event::probe_done_fn::from<&probe_done>();
  const auto on_probe_error =
      emel::gguf::loader::event::probe_error_fn::from<&probe_error>();
  const auto on_bind_done =
      emel::gguf::loader::event::bind_done_fn::from<&bind_done>();
  const auto on_bind_error =
      emel::gguf::loader::event::bind_error_fn::from<&bind_error>();
  const auto on_parse_done =
      emel::gguf::loader::event::parse_done_fn::from<&parse_done>();
  const auto on_parse_error =
      emel::gguf::loader::event::parse_error_fn::from<&parse_error>();
  require(loader.process_event(emel::gguf::loader::event::probe{
              output.bytes, requirements, on_probe_done, on_probe_error}),
          "probe real fixture");
  const std::uint64_t per_entry =
      static_cast<std::uint64_t>(requirements.max_key_bytes) +
      static_cast<std::uint64_t>(requirements.max_value_bytes);
  require(requirements.kv_count == 0 ||
              per_entry <= UINT64_MAX / requirements.kv_count,
          "real fixture arena overflow");
  output.arena.resize(static_cast<std::size_t>(
      per_entry * static_cast<std::uint64_t>(requirements.kv_count)));
  output.entries.resize(requirements.kv_count);
  require(requirements.tensor_count <= output.model->tensors.size(),
          "real fixture tensor capacity");
  require(loader.process_event(emel::gguf::loader::event::bind_storage{
              output.arena, output.entries,
              std::span<tensor_record>{output.model->tensors.data(),
                                       requirements.tensor_count},
              on_bind_done, on_bind_error}),
          "bind real fixture");
  require(loader.process_event(emel::gguf::loader::event::parse{
              output.bytes, on_parse_done, on_parse_error}),
          "parse real fixture");
  output.model->n_tensors = requirements.tensor_count;
  for (std::uint32_t index = 0; index < output.model->n_tensors; ++index) {
    auto &record = output.model->tensors[index];
    require(output.model->name_bytes_used + record.name_length <=
                output.model->name_storage.size(),
            "real fixture name capacity");
    require(static_cast<std::uint64_t>(record.name_offset) +
                record.name_length <=
                output.bytes.size(),
            "real fixture name range");
    std::copy_n(output.bytes.begin() + record.name_offset, record.name_length,
                output.model->name_storage.begin() +
                    output.model->name_bytes_used);
    record.name_offset = output.model->name_bytes_used;
    output.model->name_bytes_used += record.name_length;
    record.data = output.bytes.data();
  }
  return output;
}

struct semantic_digest {
  std::uint64_t value = 14695981039346656037ULL;
  void byte(std::uint8_t input) {
    value ^= input;
    value *= 1099511628211ULL;
  }
  void u32(std::uint32_t input) {
    for (unsigned shift = 0; shift < 32; shift += 8) {
      byte(static_cast<std::uint8_t>(input >> shift));
    }
  }
  void u64(std::uint64_t input) {
    for (unsigned shift = 0; shift < 64; shift += 8) {
      byte(static_cast<std::uint8_t>(input >> shift));
    }
  }
  void bytes(std::span<const std::uint8_t> input) {
    u64(input.size());
    for (const auto value : input) {
      byte(value);
    }
  }
};

void digest_record(semantic_digest &digest, const model_data &model,
                   const tensor_record &record) {
  const auto name = emel::model::tensor_name_view(model, record);
  digest.bytes({reinterpret_cast<const std::uint8_t *>(name.data()),
                name.size()});
  digest.u32(static_cast<std::uint32_t>(record.type));
  digest.u32(static_cast<std::uint32_t>(record.n_dims));
  const auto dimension_count = std::min<std::int32_t>(
      std::max<std::int32_t>(record.n_dims, 0),
      static_cast<std::int32_t>(record.dims.size()));
  for (std::int32_t index = 0; index < dimension_count; ++index) {
    digest.u64(static_cast<std::uint64_t>(record.dims[index]));
  }
  digest.u64(record.data_size);
  digest.byte(static_cast<std::uint8_t>(
      bindable(model, {reinterpret_cast<const std::uint8_t *>(name.data()),
                       name.size()})));
}

void observe_fixture(std::string_view label, const std::filesystem::path &path,
                     std::string_view required_name) {
  const auto value = parse_fixture(path);
  semantic_digest all{};
  for (std::uint32_t index = 0; index < value.model->n_tensors; ++index) {
    digest_record(all, *value.model, value.model->tensors[index]);
  }
  const auto *required = find_first(
      *value.model,
      {reinterpret_cast<const std::uint8_t *>(required_name.data()),
       required_name.size()});
  require(required != nullptr, "required real fixture tensor");
  semantic_digest selected{};
  digest_record(selected, *value.model, *required);
  std::cout << "fixture_case=" << label
            << " tensor_count=" << value.model->n_tensors << " digest="
            << std::hex << std::setw(16) << std::setfill('0') << all.value
            << " required_name=" << required_name << " required_digest="
            << std::setw(16) << selected.value << std::dec << std::setfill(' ')
            << '\n';
}

void benchmark(std::uint64_t iterations, std::size_t runs,
               std::uint64_t warmup) {
  fixture value{};
  std::array<char, 32> name{};
  for (std::size_t index = 0; index < 256; ++index) {
    const int length = std::snprintf(name.data(), name.size(), "tensor.%03zu", index);
    require(length > 0, "benchmark name");
    append(value, std::string_view{name.data(), static_cast<std::size_t>(length)});
  }
  constexpr std::string_view target = "tensor.255";
  emel::model::generation::tensor_view view{};
  for (std::uint64_t index = 0; index < warmup; ++index) {
    require(emel::model::generation::bind_tensor_view(*value.model, target, view),
            "warmup lookup");
  }
  std::vector<double> samples;
  samples.reserve(runs);
  std::uint64_t checksum = 0;
  for (std::size_t run = 0; run < runs; ++run) {
    const auto start = std::chrono::steady_clock::now();
    for (std::uint64_t index = 0; index < iterations; ++index) {
      require(emel::model::generation::bind_tensor_view(*value.model, target,
                                                        view),
              "benchmark lookup");
      checksum += view.tensor->data_size;
    }
    const auto elapsed = std::chrono::duration<double, std::nano>(
        std::chrono::steady_clock::now() - start);
    samples.push_back(elapsed.count() / static_cast<double>(iterations));
  }
  std::cout << std::fixed << std::setprecision(3)
            << "cpp_ns_per_lookup=" << median(std::move(samples))
            << " outcome=found checksum=" << checksum << " iter=" << iterations
            << " runs=" << runs << '\n';
}

} // namespace

int main(int argc, char **argv) try {
  if (argc == 1) {
    parity();
    return 0;
  }
  if (argc == 5 && std::string_view{argv[1]} == "--benchmark") {
    benchmark(std::strtoull(argv[2], nullptr, 10),
              static_cast<std::size_t>(std::strtoull(argv[3], nullptr, 10)),
              std::strtoull(argv[4], nullptr, 10));
    return 0;
  }
  if (argc == 4 && std::string_view{argv[1]} == "--fixtures") {
    observe_fixture("llama", argv[2], "output_norm.weight");
    observe_fixture("lfm", argv[3], "token_embd.weight");
    return 0;
  }
  std::cerr << "usage: emel-model-catalog-reference [--benchmark ITER RUNS WARMUP|--fixtures LLAMA LFM]\n";
  return 2;
} catch (const std::exception &error) {
  std::cerr << "error: " << error.what() << '\n';
  return 1;
}
