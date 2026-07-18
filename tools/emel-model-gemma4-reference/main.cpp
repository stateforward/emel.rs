#define main emel_model_common_reference_main
#include "../emel-model-catalog-reference/main.cpp"
#undef main

#include "emel/model/gemma4/detail.hpp"

namespace {

constexpr std::string_view k_source_commit =
    "843a117386ef17dc5a50549bbfc821074c2141d6";

void append_block(fixture &value, std::int32_t index, bool shared) {
  for (const std::string_view suffix : {
           "attn_norm.weight", "attn_q.weight", "attn_k.weight",
           "attn_v.weight", "attn_q_norm.weight", "attn_k_norm.weight",
           "attn_output.weight", "ffn_norm.weight", "ffn_gate.weight",
           "ffn_down.weight", "ffn_up.weight"}) {
    if (shared && suffix == "attn_v.weight") {
      continue;
    }
    const bool quantized =
        suffix == "attn_k.weight" || suffix == "attn_v.weight";
    append(value, std::string{"blk."} + std::to_string(index) + "." +
                      std::string{suffix},
           quantized ? 14 : 0, 1, quantized ? std::array<std::int64_t, 4>{256, 1, 1, 1}
                                             : std::array<std::int64_t, 4>{1, 1, 1, 1},
           quantized ? 210 : 4, true);
  }
}

fixture gemma4_fixture() {
  fixture value{};
  append(value, "token_embd.weight", 10, 1, {256, 1, 1, 1}, 84, true);
  append(value, "output_norm.weight", 0, 1, {1, 1, 1, 1}, 4, true);
  for (std::int32_t index = 0; index < 35; ++index) {
    append_block(value, index, index >= 15);
  }
  auto &model = *value.model;
  std::copy_n("gemma4", 6, model.architecture_name.begin());
  model.n_layers = 35;
  model.params.n_layer = 35;
  model.params.n_ctx = 131072;
  model.params.n_embd = 1536;
  model.params.n_embd_out = 1536;
  model.params.embd_length_per_layer_input = 256;
  model.params.n_ff = 6144;
  model.params.n_head = 8;
  model.params.n_head_kv = 1;
  model.params.attention_key_length = 512;
  model.params.attention_key_length_swa = 256;
  model.params.attention_value_length = 512;
  model.params.attention_value_length_swa = 256;
  model.params.n_vocab = 262144;
  model.params.attention_sliding_window = 512;
  model.params.attention_shared_kv_layers = 20;
  model.params.n_rot = 512;
  model.params.n_rot_swa = 256;
  model.params.attention_layer_norm_rms_epsilon = 1e-6F;
  model.params.final_logit_softcapping = 30.0F;
  model.params.rope_freq_base = 1000000.0F;
  model.params.rope_freq_base_swa = 10000.0F;
  model.params.full_attention_interval = 5;
  model.params.tie_word_embeddings = true;
  model.params.attention_sliding_window_pattern_count = 35;
  for (std::int32_t index = 0; index < 35; ++index) {
    model.params.attention_sliding_window_pattern_flags[index] =
        ((index + 1) % 5 == 0) ? 0 : 1;
  }
  return value;
}

void print_layer(const emel::model::generation::contract &contract,
                 std::int32_t index) {
  const auto &layer = contract.generation_execution.layers[index];
  emel::model::generation::block_view block{};
  require(emel::model::generation::lookup_block_view(
              contract.execution, index, block) == 0,
          "Gemma4 public block view");
  std::cout << "block=" << index
            << " window=" << static_cast<int>(layer.window_route)
            << " value=" << static_cast<int>(layer.value_route)
            << " v_norm=" << static_cast<int>(layer.v_norm_route)
            << " qk_norm=" << static_cast<int>(layer.qk_norm_route)
            << " key=" << layer.attention_key_length
            << " val=" << layer.attention_value_length
            << " rope=" << layer.attention_rope_dim
            << " freq=" << static_cast<std::int64_t>(layer.attention_rope_freq_base)
            << " shared_alias=" << (block.attention_v.tensor == block.attention_k.tensor)
            << '\n';
}

void observe() {
  auto value = gemma4_fixture();
  require(emel::model::gemma4::detail::validate_execution_contract(*value.model) == 0,
          "Gemma4 public execution validation");
  emel::model::generation::contract contract{};
  require(emel::model::gemma4::detail::build_generation_contract(
              *value.model, contract) == 0,
          "Gemma4 public generation contract");
  std::cout << "model-gemma4-parity-snapshot/v1\n";
  std::cout << "source_commit=" << k_source_commit << '\n';
  std::cout << "source_gemma4_detail_header_blob="
               "5ee94dcc8554ef8e9de0a51007acae6bb58565c9\n";
  std::cout << "source_gemma4_detail_implementation_blob="
               "ac9cc9757a5d6e97ebfbd1ef905d01f4af34dee4\n";
  std::cout << "scope=pinned_source_built_model_no_real_fixture\n";
  std::cout << "contract blocks=" << contract.execution.block_count
            << " tensors=" << contract.topology.tensor_count
            << " nodes=" << contract.topology.node_count
            << " workspace=" << contract.topology.workspace_capacity_bytes
            << " prefill=" << contract.prefill_plan.max_step_tokens
            << " decode=" << contract.decode_plan.max_step_tokens
            << " tied=" << (contract.execution.output.tensor ==
                             contract.execution.token_embedding.tensor)
            << '\n';
  for (const std::int32_t index : {0, 4, 14, 15}) {
    print_layer(contract, index);
  }
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

void gemma4_benchmark(std::uint64_t iterations, std::size_t runs,
                      std::uint64_t warmup) {
  auto value = gemma4_fixture();
  emel::model::generation::contract contract{};
  require(emel::model::gemma4::detail::build_generation_contract(
              *value.model, contract) == 0,
          "Gemma4 benchmark contract");
  emel::model::generation::block_view block{};
  for (std::uint64_t iteration = 0; iteration < warmup; ++iteration) {
    require(emel::model::generation::lookup_block_view(
                contract.execution, 14 + static_cast<std::int32_t>(iteration & 1),
                block) == 0,
            "warmup");
  }
  std::vector<double> samples;
  samples.reserve(runs);
  std::int64_t checksum = 0;
  for (std::size_t run = 0; run < runs; ++run) {
    const auto start = std::chrono::steady_clock::now();
    for (std::uint64_t iteration = 0; iteration < iterations; ++iteration) {
      require(emel::model::generation::lookup_block_view(
                  contract.execution,
                  14 + static_cast<std::int32_t>(iteration & 1), block) == 0,
              "visit");
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

int main(int argc, char **argv) try {
  if (argc == 2 && std::string_view{argv[1]} == "--source-built") {
    observe();
    return 0;
  }
  if (argc == 5 && std::string_view{argv[1]} == "--benchmark") {
    gemma4_benchmark(
        std::strtoull(argv[2], nullptr, 10),
        static_cast<std::size_t>(std::strtoull(argv[3], nullptr, 10)),
        std::strtoull(argv[4], nullptr, 10));
    return 0;
  }
  std::cerr << "usage: emel-model-gemma4-reference --source-built|--benchmark ITER RUNS WARMUP\n";
  return 2;
} catch (const std::exception &error) {
  std::cerr << "error: " << error.what() << '\n';
  return 1;
}
