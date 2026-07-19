#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <span>
#include <string_view>
#include <vector>

#include "emel/gguf/loader/detail.hpp"
#include "emel/gguf/loader/events.hpp"
#include "emel/gguf/loader/sm.hpp"
#include "emel/model/loader/errors.hpp"
#include "emel/model/omniembed/detail.hpp"

namespace {

constexpr std::string_view k_source_commit =
    "843a117386ef17dc5a50549bbfc821074c2141d6";
constexpr std::string_view k_detail_header_blob =
    "88669d66bc799b05dcd1a2b99c88516f951a48e7";
constexpr std::string_view k_detail_implementation_blob =
    "03e558a90e8df7824a0880ec2aa5500d9cc6f76b";

struct model_fixture {
  std::vector<std::uint8_t> file_bytes = {};
  std::vector<std::uint8_t> kv_arena = {};
  std::vector<emel::gguf::loader::kv_entry> kv_entries = {};
  std::unique_ptr<emel::model::data> model =
      std::make_unique<emel::model::data>();
  emel::model::omniembed::detail::execution_contract contract = {};
};

volatile std::uint64_t g_checksum = 0;

void noop_probe_done(const emel::gguf::loader::events::probe_done &) {}
void noop_probe_error(const emel::gguf::loader::events::probe_error &) {}
void noop_bind_done(const emel::gguf::loader::events::bind_done &) {}
void noop_bind_error(const emel::gguf::loader::events::bind_error &) {}
void noop_parse_done(const emel::gguf::loader::events::parse_done &) {}
void noop_parse_error(const emel::gguf::loader::events::parse_error &) {}

bool read_file(const char *path, std::vector<std::uint8_t> &bytes) {
  std::ifstream stream(path, std::ios::binary | std::ios::ate);
  if (!stream.good()) return false;
  const auto length = stream.tellg();
  if (length <= 0) return false;
  bytes.resize(static_cast<std::size_t>(length));
  stream.seekg(0, std::ios::beg);
  return static_cast<bool>(stream.read(reinterpret_cast<char *>(bytes.data()),
                                       length));
}

bool prepare(const char *path, model_fixture &fixture) {
  if (!read_file(path, fixture.file_bytes)) return false;
  emel::gguf::loader::sm loader{};
  emel::gguf::loader::requirements requirements = {};
  const auto on_probe_done =
      emel::gguf::loader::event::probe_done_fn::from<&noop_probe_done>();
  const auto on_probe_error =
      emel::gguf::loader::event::probe_error_fn::from<&noop_probe_error>();
  const auto on_bind_done =
      emel::gguf::loader::event::bind_done_fn::from<&noop_bind_done>();
  const auto on_bind_error =
      emel::gguf::loader::event::bind_error_fn::from<&noop_bind_error>();
  const auto on_parse_done =
      emel::gguf::loader::event::parse_done_fn::from<&noop_parse_done>();
  const auto on_parse_error =
      emel::gguf::loader::event::parse_error_fn::from<&noop_parse_error>();
  const emel::gguf::loader::event::probe probe{
      std::span<const std::uint8_t>{fixture.file_bytes}, requirements,
      on_probe_done, on_probe_error};
  if (!loader.process_event(probe) || requirements.tensor_count == 0u ||
      requirements.tensor_count > fixture.model->tensors.size()) {
    return false;
  }

  fixture.kv_arena.resize(static_cast<std::size_t>(
      emel::gguf::loader::detail::required_kv_arena_bytes(requirements)));
  fixture.kv_entries.resize(requirements.kv_count);
  fixture.model->n_tensors = requirements.tensor_count;
  const emel::gguf::loader::event::bind_storage bind{
      std::span<std::uint8_t>{fixture.kv_arena},
      std::span<emel::gguf::loader::kv_entry>{fixture.kv_entries},
      std::span<emel::model::data::tensor_record>{fixture.model->tensors.data(),
                                                  fixture.model->n_tensors},
      on_bind_done, on_bind_error};
  if (!loader.process_event(bind)) return false;
  const emel::gguf::loader::event::parse parse{
      std::span<const std::uint8_t>{fixture.file_bytes},
      on_parse_done, on_parse_error};
  if (!loader.process_event(parse)) return false;

  const emel::model::detail::kv_binding binding{
      .arena = std::span<const std::uint8_t>{fixture.kv_arena},
      .entries = std::span<const emel::gguf::loader::kv_entry>{fixture.kv_entries}};
  const auto *architecture_entry =
      emel::model::detail::find_kv_entry(binding, "general.architecture");
  std::string_view architecture = {};
  if (architecture_entry == nullptr ||
      !emel::model::detail::decode_string_value(binding, *architecture_entry,
                                                architecture) ||
      !emel::model::omniembed::detail::is_execution_architecture(architecture)) {
    return false;
  }
  emel::model::detail::copy_name(fixture.model->architecture_name, architecture);
  const emel::model::detail::hparam_loader hparams{binding};
  if (!emel::model::omniembed::detail::load_hparams(hparams, *fixture.model)) {
    return false;
  }
  fixture.model->weights_data = fixture.file_bytes.data();
  fixture.model->weights_size = fixture.file_bytes.size();
  fixture.model->name_bytes_used = 0u;
  for (std::uint32_t index = 0; index < fixture.model->n_tensors; ++index) {
    auto &tensor = fixture.model->tensors[index];
    const auto source_offset = static_cast<std::size_t>(tensor.name_offset);
    const auto length = static_cast<std::size_t>(tensor.name_length);
    if (source_offset + length > fixture.file_bytes.size() ||
        fixture.model->name_bytes_used + length >
            fixture.model->name_storage.size()) {
      return false;
    }
    std::memcpy(fixture.model->name_storage.data() +
                    fixture.model->name_bytes_used,
                fixture.file_bytes.data() + source_offset, length);
    tensor.name_offset = fixture.model->name_bytes_used;
    fixture.model->name_bytes_used += static_cast<std::uint32_t>(length);
  }
  return emel::model::omniembed::detail::build_execution_contract(
             *fixture.model, fixture.contract) ==
      emel::error::cast(emel::model::loader::error::none);
}

void print_family(const emel::model::omniembed::detail::family_view &family) {
  std::printf("family=%.*s count=%u first=%.*s\n",
              static_cast<int>(family.prefix.size()), family.prefix.data(),
              family.tensor_count, static_cast<int>(family.first.name.size()),
              family.first.name.data());
}

void observe(model_fixture &fixture) {
  const auto &contract = fixture.contract;
  const auto &metadata = fixture.model->meta;
  std::printf("model-omniembed-parity-snapshot/v1\n");
  std::printf("source_commit=%.*s\n", static_cast<int>(k_source_commit.size()),
              k_source_commit.data());
  std::printf("source_omniembed_detail_header_blob=%.*s\n",
              static_cast<int>(k_detail_header_blob.size()),
              k_detail_header_blob.data());
  std::printf("source_omniembed_detail_implementation_blob=%.*s\n",
              static_cast<int>(k_detail_implementation_blob.size()),
              k_detail_implementation_blob.data());
  std::printf(
      "contract embedding=%d image_encoder=%d audio_encoder=%d "
      "matryoshka_count=%u matryoshka_0=%d matryoshka_3=%d image_size=%d "
      "audio_rate=%d audio_fft=%d audio_window=%d audio_hop=%d audio_mels=%d\n",
      contract.embedding_length, contract.image_encoder_length,
      contract.audio_encoder_length, contract.matryoshka_dimension_count,
      contract.matryoshka_dimensions[0], contract.matryoshka_dimensions[3],
      metadata.clip_vision_data.image_size,
      metadata.clip_audio_data.sample_rate, metadata.clip_audio_data.n_fft,
      metadata.clip_audio_data.win_length, metadata.clip_audio_data.hop_size,
      metadata.clip_audio_data.num_mel_bins);
  print_family(contract.text_encoder);
  print_family(contract.text_projection);
  print_family(contract.image_encoder);
  print_family(contract.image_projection);
  print_family(contract.audio_encoder);
  print_family(contract.audio_projection);
}

void benchmark(const char *path, const std::uint64_t iterations,
               const std::size_t runs, const std::uint64_t warmup) {
  for (std::uint64_t index = 0; index < warmup; ++index) {
    auto fixture = std::make_unique<model_fixture>();
    if (!prepare(path, *fixture)) std::abort();
    g_checksum += fixture->contract.text_encoder.tensor_count;
  }
  std::vector<std::uint64_t> samples;
  samples.reserve(runs);
  g_checksum = 0;
  for (std::size_t run = 0; run < runs; ++run) {
    const auto begin = std::chrono::steady_clock::now();
    for (std::uint64_t index = 0; index < iterations; ++index) {
      auto fixture = std::make_unique<model_fixture>();
      if (!prepare(path, *fixture)) std::abort();
      g_checksum += fixture->contract.text_encoder.tensor_count;
    }
    const auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
                             std::chrono::steady_clock::now() - begin)
                             .count();
    samples.push_back(static_cast<std::uint64_t>(elapsed) / iterations);
  }
  std::sort(samples.begin(), samples.end());
  std::printf("cpp_ns_per_visit=%llu outcome=found checksum=%llu iter=%llu runs=%zu\n",
              static_cast<unsigned long long>(samples[samples.size() / 2]),
              static_cast<unsigned long long>(g_checksum),
              static_cast<unsigned long long>(iterations), runs);
}

bool rejected(const emel::error::type result) {
  return result != emel::error::cast(emel::model::loader::error::none);
}

bool negative_cases(const char *path) {
  auto missing = std::make_unique<model_fixture>();
  if (!prepare(path, *missing)) return false;
  for (std::uint32_t index = 0; index < missing->model->n_tensors; ++index) {
    auto &tensor = missing->model->tensors[index];
    if (emel::model::tensor_name_view(*missing->model, tensor)
            .starts_with("audio_projection.")) {
      tensor.data = nullptr;
      tensor.data_size = 0;
    }
  }
  if (!rejected(emel::model::omniembed::detail::build_execution_contract(
          *missing->model, missing->contract))) {
    return false;
  }
  std::printf("negative missing_audio_projection=rejected\n");

  auto dimensions = std::make_unique<model_fixture>();
  if (!prepare(path, *dimensions)) return false;
  dimensions->model->params.matryoshka_dimensions[1] = 1536;
  if (!rejected(emel::model::omniembed::detail::validate_data(
          *dimensions->model))) {
    return false;
  }
  std::printf("negative invalid_matryoshka=rejected\n");
  return true;
}

}  // namespace

int main(int argc, char **argv) {
  if (argc == 3 && std::string_view{argv[1]} == "--fixture") {
    auto fixture = std::make_unique<model_fixture>();
    if (!prepare(argv[2], *fixture)) return 1;
    observe(*fixture);
    return 0;
  }
  if (argc == 3 && std::string_view{argv[1]} == "--negative") {
    return negative_cases(argv[2]) ? 0 : 1;
  }
  if (argc == 6 && std::string_view{argv[1]} == "--benchmark") {
    benchmark(argv[2], std::strtoull(argv[3], nullptr, 10),
              static_cast<std::size_t>(std::strtoull(argv[4], nullptr, 10)),
              std::strtoull(argv[5], nullptr, 10));
    return 0;
  }
  std::fprintf(stderr,
               "usage: omniembed-reference --fixture GGUF|--negative GGUF|--benchmark GGUF ITER RUNS WARMUP\n");
  return 2;
}
