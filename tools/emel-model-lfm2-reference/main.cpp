#define main emel_model_common_reference_main
#include "../emel-model-catalog-reference/main.cpp"
#undef main

#include "emel/model/lfm2/detail.hpp"

namespace {

constexpr std::string_view k_source_commit =
    "843a117386ef17dc5a50549bbfc821074c2141d6";

bool lfm2_attention_block(std::int32_t index) {
  return index >= 2 && index % 2 == 0;
}

void configure_lfm2_230m(model_data &model) {
  model.n_layers = 14;
  model.params.n_layer = 14;
  model.params.n_ctx = 128000;
  model.params.n_embd = 1024;
  model.params.n_embd_out = 1024;
  model.params.n_ff = 4096;
  model.params.n_head = 16;
  model.params.n_head_kv = 8;
  model.params.n_vocab = 65536;
  model.params.shortconv_l_cache = 3;
  model.params.attention_layer_norm_rms_epsilon = 1.0e-6F;
  model.params.rope_freq_base = 1000000.0F;
  model.params.attention_layer_pattern_count = 14;
  for (std::int32_t index = 0; index < model.n_layers; ++index) {
    model.params.attention_layer_pattern_flags[static_cast<std::size_t>(index)] =
        lfm2_attention_block(index) ? 1u : 0u;
  }
}

fixture lfm2_230m_fixture() {
  fixture value{};
  append(value, "token_embd.weight", 0, 1, {1, 1, 1, 1}, 4, true);
  append(value, "token_embd_norm.weight", 0, 1, {1, 1, 1, 1}, 4, true);
  for (std::int32_t index = 0; index < 14; ++index) {
    for (const std::string_view suffix : {
             "attn_norm.weight", "ffn_norm.weight", "ffn_gate.weight",
             "ffn_down.weight", "ffn_up.weight"}) {
      append(value, std::string{"blk."} + std::to_string(index) + "." +
                        std::string{suffix},
             0, 1, {1, 1, 1, 1}, 4, true);
    }
    if (lfm2_attention_block(index)) {
      for (const std::string_view suffix : {
               "attn_q.weight", "attn_k.weight", "attn_v.weight",
               "attn_q_norm.weight", "attn_k_norm.weight",
               "attn_output.weight"}) {
        append(value, std::string{"blk."} + std::to_string(index) + "." +
                          std::string{suffix},
               0, 1, {1, 1, 1, 1}, 4, true);
      }
    } else {
      for (const std::string_view suffix : {
               "shortconv.conv.weight", "shortconv.in_proj.weight",
               "shortconv.out_proj.weight"}) {
        append(value, std::string{"blk."} + std::to_string(index) + "." +
                          std::string{suffix},
               0, 1, {1, 1, 1, 1}, 4, true);
      }
    }
  }
  configure_lfm2_230m(*value.model);
  return value;
}

std::string_view lfm2_variant(const model_data &model) {
  if (model.params.n_layer == 14 && model.params.n_embd == 1024 &&
      model.params.n_head == 16) {
    return "230m";
  }
  if (model.params.n_layer == 16 && model.params.n_embd == 2048 &&
      model.params.n_head == 32) {
    return "1.2b";
  }
  return "unknown";
}

void print_lfm2(std::string_view label, model_data &model) {
  require(emel::model::lfm2::detail::validate_execution_contract(model) == 0,
          "LFM2 public validation");
  emel::model::generation::contract contract{};
  require(emel::model::lfm2::detail::build_generation_contract(model,
                                                                contract) == 0,
          "LFM2 public generation contract");
  emel::model::generation::block_view block0{};
  emel::model::generation::block_view block2{};
  require(emel::model::generation::lookup_block_view(contract.execution, 0,
                                                      block0) == 0,
          "LFM2 public block 0 view");
  require(emel::model::generation::lookup_block_view(contract.execution, 2,
                                                      block2) == 0,
          "LFM2 public block 2 view");
  std::uint32_t attention_blocks = 0;
  for (std::int32_t index = 0; index < contract.execution.block_count; ++index) {
    emel::model::generation::block_view block{};
    require(emel::model::generation::lookup_block_view(contract.execution, index,
                                                        block) == 0,
            "LFM2 public block view");
    attention_blocks += block.uses_attention ? 1u : 0u;
  }
  const auto audit =
      emel::model::generation::build_quantized_path_audit(contract.execution);
  std::cout << "lfm2_case=" << label << " variant=" << lfm2_variant(model)
            << " block_count=" << contract.execution.block_count
            << " tensor_count=" << contract.topology.tensor_count
            << " workspace=" << contract.topology.workspace_capacity_bytes
            << " prefill=" << contract.prefill_plan.max_step_tokens
            << " decode=" << contract.decode_plan.max_step_tokens
            << " attention_blocks=" << attention_blocks
            << " shortconv_blocks="
            << static_cast<std::uint32_t>(contract.execution.block_count) -
                   attention_blocks
            << " block0_attention=" << (block0.uses_attention ? 1 : 0)
            << " block2_attention=" << (block2.uses_attention ? 1 : 0) << '\n';
  for (const auto &stage : audit.stages) {
    std::cout << "stage="
              << emel::model::generation::quantized_stage_family_name(stage.family)
              << " type="
              << emel::model::generation::tensor_type_name(stage.tensor_type)
              << " contract="
              << emel::model::generation::quantized_contract_kind_name(
                     stage.contract)
              << " consistent=" << stage.consistent_across_layers << '\n';
  }
}

void parity_fixture(const std::filesystem::path &path) {
  auto value = parse_fixture(path);
  const emel::model::detail::kv_binding binding{
      .arena = std::span<const std::uint8_t>{value.arena},
      .entries = std::span<const emel::gguf::loader::kv_entry>{value.entries},
  };
  const emel::model::detail::hparam_loader loader{binding};
  require(emel::model::lfm2::detail::load_hparams(loader, *value.model),
          "LFM2 source hparams");
  value.model->n_layers = value.model->params.n_layer;
  std::cout << "model-lfm2-parity-snapshot/v1\n";
  std::cout << "source_commit=" << k_source_commit << '\n';
  std::cout << "source_lfm2_detail_header_blob="
               "af0e7d509a3854a847bdde2dad831c5ed60de379\n";
  std::cout << "source_lfm2_detail_implementation_blob="
               "a6f215e74ed793ea8f606048c00faa80c37056be\n";
  const auto variant = lfm2_variant(*value.model);
  const auto label = variant == "230m" ? "real_fixture_230m"
                                      : "real_fixture_1_2b";
  print_lfm2(label, *value.model);
}

void validation_parity() {
  auto value = lfm2_230m_fixture();
  auto &model = *value.model;
  model.architecture_name[0] = 'l';
  model.architecture_name[1] = 'f';
  model.architecture_name[2] = 'm';
  model.architecture_name[3] = '2';
  model.n_layers = 3;
  model.params.n_layer = 3;
  model.params.n_ctx = 17;
  model.params.n_embd = 64;
  model.params.n_embd_out = 64;
  model.params.n_ff = 128;
  model.params.n_head = 4;
  model.params.n_head_kv = 2;
  model.params.n_vocab = 99;
  model.params.shortconv_l_cache = 1;
  model.params.attention_layer_norm_rms_epsilon = 1.0e-5F;
  model.params.rope_freq_base = 10.0F;
  model.params.attention_layer_pattern_count = 3;
  for (std::size_t index = 0;
       index < model.params.attention_layer_pattern_flags.size(); ++index) {
    model.params.attention_layer_pattern_flags[index] = index == 2 ? 1u : 0u;
  }
  const bool builder_ok =
      emel::model::lfm2::detail::validate_builder_contract(model) == 0;
  const bool data_ok = emel::model::lfm2::detail::validate_data(model) == 0;
  const bool execution_invalid =
      emel::model::lfm2::detail::validate_execution_contract(model) != 0;
  require(builder_ok && data_ok && execution_invalid,
          "LFM2 non-strict validation distinction");
  std::cout << "lfm2_validation=non_strict builder=ok data=ok "
               "execution=model_invalid block_count=3\n";
}

void lfm2_benchmark(std::uint64_t iterations, std::size_t runs,
                    std::uint64_t warmup) {
  auto value = lfm2_230m_fixture();
  emel::model::generation::contract contract{};
  require(emel::model::lfm2::detail::build_generation_contract(*value.model,
                                                                contract) == 0,
          "benchmark LFM2 contract");
  emel::model::generation::block_view block{};
  for (std::uint64_t iteration = 0; iteration < warmup; ++iteration) {
    const std::int32_t block_index = iteration % 2u == 0u ? 0 : 2;
    require(emel::model::generation::lookup_block_view(contract.execution,
                                                        block_index, block) == 0,
            "warmup lookup");
    do_not_optimize(block);
  }
  std::vector<double> samples;
  samples.reserve(runs);
  std::int64_t checksum = 0;
  for (std::size_t run = 0; run < runs; ++run) {
    const auto start = std::chrono::steady_clock::now();
    for (std::uint64_t iteration = 0; iteration < iterations; ++iteration) {
      const std::int32_t block_index = iteration % 2u == 0u ? 0 : 2;
      require(emel::model::generation::lookup_block_view(contract.execution,
                                                          block_index, block) == 0,
              "block lookup");
      do_not_optimize(block);
      checksum += block.index;
    }
    const auto elapsed = std::chrono::duration<double, std::nano>(
        std::chrono::steady_clock::now() - start);
    samples.push_back(elapsed.count() / static_cast<double>(iterations));
  }
  std::cout << std::fixed << std::setprecision(3)
            << "cpp_ns_per_visit=" << median(std::move(samples))
            << " outcome=found checksum=" << checksum << " iter=" << iterations
            << " runs=" << runs << '\n';
}

}  // namespace

int main(int argc, char **argv) {
  if (argc == 3 && std::string_view{argv[1]} == "--fixture") {
    parity_fixture(argv[2]);
    return 0;
  }
  if (argc == 5 && std::string_view{argv[1]} == "--benchmark") {
    lfm2_benchmark(std::strtoull(argv[2], nullptr, 10),
                   static_cast<std::size_t>(
                       std::strtoull(argv[3], nullptr, 10)),
                   std::strtoull(argv[4], nullptr, 10));
    return 0;
  }
  if (argc == 2 && std::string_view{argv[1]} == "--validation") {
    validation_parity();
    return 0;
  }
  std::cerr << "usage: emel-model-lfm2-reference --fixture LFM2|--validation|--benchmark ITER RUNS WARMUP\n";
  return 2;
}
