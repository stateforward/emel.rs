#define main emel_model_common_reference_main
#include "../emel-model-catalog-reference/main.cpp"
#undef main

#include "emel/model/qwen3/detail.hpp"

namespace {

constexpr std::string_view k_source_commit =
    "843a117386ef17dc5a50549bbfc821074c2141d6";

void configure_qwen3(model_data &model, std::int32_t block_count) {
  model.n_layers = block_count;
  model.params.n_layer = block_count;
  model.params.n_ctx = 4096;
  model.params.n_embd = 128;
  model.params.n_embd_out = 128;
  model.params.n_ff = 512;
  model.params.n_head = 8;
  model.params.n_head_kv = 4;
  model.params.attention_key_length = 32;
  model.params.attention_value_length = 40;
  model.params.n_rot = 32;
  model.params.attention_layer_norm_rms_epsilon = 1e-6F;
  model.params.rope_freq_base = 10000.0F;
  model.params.tie_word_embeddings = true;
}

void print_qwen3(std::string_view label, model_data &model) {
  require(emel::model::qwen3::detail::validate_data(model) == 0,
          "Qwen3 public validate_data");
  emel::model::generation::contract contract{};
  require(emel::model::qwen3::detail::build_generation_contract(model,
                                                                 contract) == 0,
          "Qwen3 public build_generation_contract");
  emel::model::generation::block_view block{};
  require(emel::model::generation::lookup_block_view(contract.execution, 0,
                                                       block) == 0,
          "Qwen3 common public lookup_block_view");
  std::cout << "qwen3_case=" << label
            << " block_count=" << contract.execution.block_count
            << " tensor_count=" << contract.topology.tensor_count
            << " workspace=" << contract.topology.workspace_capacity_bytes
            << " prefill=" << contract.prefill_plan.max_step_tokens
            << " decode=" << contract.decode_plan.max_step_tokens
            << " uses_attention=" << (block.uses_attention ? 1 : 0)
            << " qk_norm_headwise_rms="
            << (contract.generation_execution.layers[0].qk_norm_route ==
                emel::model::generation::generation_attention_qk_norm_route::headwise_rms)
            << " key_length="
            << contract.generation_execution.layers[0].attention_key_length
            << " value_length="
            << contract.generation_execution.layers[0].attention_value_length
            << " rope_dim="
            << contract.generation_execution.layers[0].attention_rope_dim << '\n';
  for (const auto &stage : contract.quantized_audit.stages) {
    std::cout << "stage="
              << emel::model::generation::quantized_stage_family_name(stage.family)
              << " type="
              << emel::model::generation::tensor_type_name(stage.tensor_type)
              << " contract="
              << emel::model::generation::quantized_contract_kind_name(stage.contract)
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
  require(emel::model::qwen3::detail::load_hparams(loader, *value.model),
          "Qwen3 public load_hparams");
  value.model->n_layers = value.model->params.n_layer;
  std::cout << "model-qwen3-parity-snapshot/v1\n";
  std::cout << "source_commit=" << k_source_commit << '\n';
  std::cout << "source_qwen3_detail_header_blob="
               "6e31b097184042bfe1ec9486af060214f1d01b2b\n";
  std::cout << "source_qwen3_detail_implementation_blob="
               "a43d8fee3ed962d84447360e7601c999468bf6f8\n";
  print_qwen3("real_fixture_0_6b", *value.model);
}

void qwen3_benchmark(std::uint64_t iterations, std::size_t runs,
                     std::uint64_t warmup) {
  auto value = generation_fixture();
  for (const std::string_view suffix : {
           "attn_norm.weight", "attn_q.weight", "attn_k.weight",
           "attn_v.weight", "attn_q_norm.weight", "attn_k_norm.weight",
           "attn_output.weight", "ffn_norm.weight", "ffn_gate.weight",
           "ffn_down.weight", "ffn_up.weight"}) {
    append(value, std::string{"blk.1."} + std::string{suffix}, 0, 1,
           {1, 1, 1, 1}, 4, true);
  }
  configure_qwen3(*value.model, 2);
  emel::model::generation::contract contract{};
  require(emel::model::qwen3::detail::build_generation_contract(*value.model,
                                                                 contract) == 0,
          "benchmark Qwen3 contract");
  emel::model::generation::block_view block{};
  for (std::uint64_t index = 0; index < warmup; ++index) {
    const auto block_index = static_cast<std::int32_t>(index & 1u);
    require(emel::model::generation::lookup_block_view(
                contract.execution, block_index, block) == 0,
            "warmup lookup");
  }
  std::vector<double> samples;
  samples.reserve(runs);
  std::int64_t checksum = 0;
  for (std::size_t run = 0; run < runs; ++run) {
    const auto start = std::chrono::steady_clock::now();
    for (std::uint64_t index = 0; index < iterations; ++index) {
      const auto block_index = static_cast<std::int32_t>(index & 1u);
      require(emel::model::generation::lookup_block_view(
                  contract.execution, block_index, block) == 0,
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
    qwen3_benchmark(std::strtoull(argv[2], nullptr, 10),
                    static_cast<std::size_t>(std::strtoull(argv[3], nullptr, 10)),
                    std::strtoull(argv[4], nullptr, 10));
    return 0;
  }
  std::cerr << "usage: emel-model-qwen3-reference --fixture QWEN3|--benchmark ITER RUNS WARMUP\n";
  return 2;
}
