#define main emel_model_common_reference_main
#include "../emel-model-catalog-reference/main.cpp"
#undef main

#include "emel/model/llama/any.hpp"

namespace {

constexpr std::string_view k_source_commit =
    "843a117386ef17dc5a50549bbfc821074c2141d6";

void configure_llama(model_data &model, std::int32_t block_count) {
  model.n_layers = block_count;
  model.params.n_layer = block_count;
  model.params.n_ctx = 4096;
  model.params.n_embd = 128;
  model.params.n_embd_out = 128;
  model.params.n_head = 8;
  model.params.n_head_kv = 4;
  model.params.n_rot = 32;
  model.params.attention_key_length = 32;
  model.params.attention_value_length = 32;
  model.params.rope_freq_base = 10000.0F;
}

void print_llama(std::string_view label, model_data &model) {
  emel::model::llama::execution_view facade{};
  require(emel::model::llama::build_execution_view(model, facade) == 0,
          "Llama public build_execution_view");
  emel::model::generation::contract contract{};
  require(emel::model::llama::detail::build_generation_contract(model,
                                                                 contract) == 0,
          "Llama generation contract");
  emel::model::llama::block_view block{};
  require(emel::model::llama::lookup_block_view(facade, 0, block) == 0,
          "Llama public lookup_block_view");
  const auto audit = emel::model::llama::build_quantized_path_audit(facade);
  require(contract.execution.block_count == facade.block_count,
          "facade contract equivalence");
  std::cout << "llama_case=" << label
            << " block_count=" << facade.block_count
            << " tensor_count=" << contract.topology.tensor_count
            << " workspace=" << contract.topology.workspace_capacity_bytes
            << " prefill=" << contract.prefill_plan.max_step_tokens
            << " decode=" << contract.decode_plan.max_step_tokens
            << " uses_attention=" << (block.uses_attention ? 1 : 0) << '\n';
  for (const auto &stage : audit.stages) {
    std::cout << "stage="
              << emel::model::llama::quantized_stage_family_name(stage.family)
              << " type=" << emel::model::llama::tensor_type_name(stage.tensor_type)
              << " contract="
              << emel::model::llama::quantized_contract_kind_name(stage.contract)
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
  require(emel::model::llama::detail::load_hparams(loader, *value.model),
          "Llama source hparams");
  value.model->n_layers = value.model->params.n_layer;
  std::cout << "model-llama-parity-snapshot/v1\n";
  std::cout << "source_commit=" << k_source_commit << '\n';
  std::cout << "source_llama_any_blob=430ba6a8897854b5f58660d3f4d51d006727559e\n";
  std::cout << "source_llama_detail_header_blob=91f96245d29a22aa83c570cb4fe8bc8cf9e1d081\n";
  std::cout << "source_llama_detail_implementation_blob=c78e3656d41772dcc4dba5c46623d9f2fec784ed\n";
  print_llama("real_fixture", *value.model);
}

void llama_benchmark(std::uint64_t iterations, std::size_t runs,
                     std::uint64_t warmup) {
  auto value = generation_fixture();
  for (const std::string_view suffix : {
           "attn_norm.weight", "attn_q.weight", "attn_k.weight",
           "attn_v.weight", "attn_output.weight", "ffn_norm.weight",
           "ffn_gate.weight", "ffn_down.weight", "ffn_up.weight"}) {
    append(value, std::string{"blk.1."} + std::string{suffix});
  }
  configure_llama(*value.model, 2);
  emel::model::llama::execution_view execution{};
  require(emel::model::llama::build_execution_view(*value.model, execution) == 0,
          "benchmark execution");
  emel::model::llama::block_view block{};
  for (std::uint64_t index = 0; index < warmup; ++index) {
    const auto block_index = static_cast<std::int32_t>(index & 1u);
    require(emel::model::llama::lookup_block_view(execution, block_index, block) == 0,
            "warmup lookup");
  }
  std::vector<double> samples;
  samples.reserve(runs);
  std::int64_t checksum = 0;
  for (std::size_t run = 0; run < runs; ++run) {
    const auto start = std::chrono::steady_clock::now();
    for (std::uint64_t index = 0; index < iterations; ++index) {
      const auto block_index = static_cast<std::int32_t>(index & 1u);
      require(emel::model::llama::lookup_block_view(execution, block_index, block) == 0,
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
    llama_benchmark(std::strtoull(argv[2], nullptr, 10),
                    static_cast<std::size_t>(std::strtoull(argv[3], nullptr, 10)),
                    std::strtoull(argv[4], nullptr, 10));
    return 0;
  }
  std::cerr << "usage: emel-model-llama-reference --fixture LLAMA|--benchmark ITER RUNS WARMUP\n";
  return 2;
}
