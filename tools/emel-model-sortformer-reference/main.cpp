#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string_view>
#include <memory>
#include <vector>

#include "diarization/sortformer_fixture.hpp"
#include "emel/model/sortformer/detail.hpp"

namespace fixture = emel::bench::diarization::sortformer_fixture;

namespace {

constexpr std::string_view k_source_commit =
    "843a117386ef17dc5a50549bbfc821074c2141d6";
constexpr std::string_view k_any_blob =
    "d79cb231e0dfc75f5aa4d8da7db2ecf36c92c9b6";
constexpr std::string_view k_detail_header_blob =
    "72766c7042707a8037ac49779b8e3f8e2b34c555";
constexpr std::string_view k_detail_implementation_blob =
    "011a3ec4dfaab102457c3f6ca6e20a5adf9934b1";

volatile std::uint64_t g_checksum = 0;

void print_family(const emel::model::sortformer::family_view &family) {
  std::printf("family=%.*s count=%u first=%.*s\n",
              static_cast<int>(family.prefix.size()), family.prefix.data(),
              family.tensor_count, static_cast<int>(family.first.name.size()),
              family.first.name.data());
}

void observe(fixture::model_fixture &model) {
  const auto &contract = model.contract;
  std::printf("model-sortformer-parity-snapshot/v1\n");
  std::printf("source_commit=%.*s\n", static_cast<int>(k_source_commit.size()),
              k_source_commit.data());
  std::printf("source_sortformer_any_blob=%.*s\n",
              static_cast<int>(k_any_blob.size()), k_any_blob.data());
  std::printf("source_sortformer_detail_header_blob=%.*s\n",
              static_cast<int>(k_detail_header_blob.size()),
              k_detail_header_blob.data());
  std::printf("source_sortformer_detail_implementation_blob=%.*s\n",
              static_cast<int>(k_detail_implementation_blob.size()),
              k_detail_implementation_blob.data());
  std::printf(
      "contract sample_rate=%d speakers=%d frame_shift_ms=%d chunk_len=%d "
      "right_context=%d fifo_len=%d cache_period=%d cache_len=%d\n",
      contract.sample_rate, contract.speaker_count, contract.frame_shift_ms,
      contract.chunk_len, contract.chunk_right_context, contract.fifo_len,
      contract.spkcache_update_period, contract.spkcache_len);
  print_family(contract.feature_extractor);
  print_family(contract.encoder);
  print_family(contract.modules);
  print_family(contract.transformer_encoder);
}

void benchmark(const std::uint64_t iterations, const std::size_t runs,
               const std::uint64_t warmup) {
  for (std::uint64_t i = 0; i < warmup; ++i) {
    auto model = std::make_unique<fixture::model_fixture>();
    if (!fixture::prepare(*model)) std::abort();
    g_checksum += model->contract.encoder.tensor_count;
  }
  std::vector<std::uint64_t> samples;
  samples.reserve(runs);
  g_checksum = 0;
  for (std::size_t run = 0; run < runs; ++run) {
    const auto begin = std::chrono::steady_clock::now();
    for (std::uint64_t i = 0; i < iterations; ++i) {
      auto model = std::make_unique<fixture::model_fixture>();
      if (!fixture::prepare(*model)) std::abort();
      g_checksum += model->contract.encoder.tensor_count;
    }
    const auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
                             std::chrono::steady_clock::now() - begin)
                             .count();
    samples.push_back(static_cast<std::uint64_t>(elapsed) / iterations);
  }
  std::sort(samples.begin(), samples.end());
  const std::uint64_t checksum = g_checksum;
  std::printf("cpp_ns_per_visit=%llu outcome=found checksum=%llu iter=%llu runs=%zu\n",
              static_cast<unsigned long long>(samples[samples.size() / 2]),
              static_cast<unsigned long long>(checksum),
              static_cast<unsigned long long>(iterations), runs);
}

bool is_rejected(const emel::error::type result) {
  return result !=
         emel::error::cast(emel::model::loader::error::none);
}

bool negative_cases() {
  auto model = std::make_unique<fixture::model_fixture>();
  if (!fixture::prepare(*model)) return false;
  model->model.architecture_name.fill('\0');
  constexpr std::string_view wrong_architecture = "llama";
  std::copy(wrong_architecture.begin(), wrong_architecture.end(),
            model->model.architecture_name.begin());
  emel::model::sortformer::execution_contract contract = {};
  if (!is_rejected(emel::model::sortformer::build_execution_contract(
          model->model, contract))) {
    return false;
  }
  std::printf("negative architecture=rejected\n");

  model->model.architecture_name.fill('\0');
  constexpr std::string_view sortformer_architecture = "sortformer";
  std::copy(sortformer_architecture.begin(), sortformer_architecture.end(),
            model->model.architecture_name.begin());
  for (std::uint32_t index = 0; index < model->model.n_tensors; ++index) {
    auto &tensor = model->model.tensors[index];
    if (emel::model::tensor_name_view(model->model, tensor)
            .starts_with("mods.")) {
      tensor.data = nullptr;
      tensor.data_size = 0;
    }
  }
  if (!is_rejected(emel::model::sortformer::build_execution_contract(
          model->model, contract))) {
    return false;
  }
  std::printf("negative missing_modules=rejected\n");

  constexpr std::string_view skipped_key = "sortformer.skipped_tensor_count";
  std::erase_if(model->kv_entries, [&](const auto &entry) {
    return std::string_view{
               reinterpret_cast<const char *>(model->kv_arena.data() +
                                                entry.key_offset),
               entry.key_length} == skipped_key;
  });
  const emel::model::detail::hparam_loader loader{
      fixture::kv_binding_from_fixture(*model)};
  if (!emel::model::sortformer::detail::load_hparams(loader, model->model) ||
      model->model.params.n_features != 4) {
    return false;
  }
  std::printf("negative missing_skipped=accepted_default_zero\n");
  return true;
}

}  // namespace

int main(int argc, char **argv) {
  if (argc != 2 && argc != 5) {
    std::fprintf(stderr, "usage: sortformer-reference --fixture|--benchmark ITER RUNS WARMUP\n");
    return 2;
  }
  if (std::string_view{argv[1]} == "--fixture" && argc == 2) {
    auto model = std::make_unique<fixture::model_fixture>();
    if (!fixture::prepare(*model)) {
      std::fprintf(stderr, "failed to prepare pinned Sortformer model\n");
      return 1;
    }
    observe(*model);
    return 0;
  }
  if (std::string_view{argv[1]} == "--negative" && argc == 2) {
    return negative_cases() ? 0 : 1;
  }
  if (std::string_view{argv[1]} == "--benchmark" && argc == 5) {
    benchmark(std::strtoull(argv[2], nullptr, 10),
              static_cast<std::size_t>(std::strtoull(argv[3], nullptr, 10)),
              std::strtoull(argv[4], nullptr, 10));
    return 0;
  }
  return 2;
}
