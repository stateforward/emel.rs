#include <array>
#include <cstdint>
#include <iostream>
#include <memory>
#include <string_view>

#include "emel/model/data.hpp"
#include "emel/text/tokenizer/detail.hpp"
#include "emel/text/tokenizer/preprocessor/detail.hpp"

namespace {

using vocab = emel::model::data::vocab;
using tokenizer_model = emel::model::data::tokenizer_model;

void reset_profile(vocab &value) {
  value.tokenizer_model_id = tokenizer_model::UNKNOWN;
  value.tokenizer_pre_id = emel::model::data::tokenizer_pre::DEFAULT;
  value.bos_id = -1;
  value.eos_id = -1;
  value.eot_id = -1;
  value.eom_id = -1;
  value.unk_id = -1;
  value.sep_id = -1;
  value.pad_id = -1;
  value.cls_id = -1;
  value.mask_id = -1;
  value.prefix_id = -1;
  value.suffix_id = -1;
  value.middle_id = -1;
  value.fim_pre_id = -1;
  value.fim_suf_id = -1;
  value.fim_mid_id = -1;
  value.fim_pad_id = -1;
  value.fim_rep_id = -1;
  value.fim_sep_id = -1;
  value.add_bos = false;
  value.add_eos = false;
  value.add_sep = false;
  value.add_space_prefix = false;
  value.remove_extra_whitespaces = false;
  value.escape_whitespaces = true;
  value.treat_whitespace_as_suffix = false;
  value.ignore_merges = false;
}

std::string_view model_name(const tokenizer_model model) {
  switch (model) {
  case tokenizer_model::NONE:
    return "None";
  case tokenizer_model::SPM:
    return "SentencePiece";
  case tokenizer_model::BPE:
    return "Bpe";
  case tokenizer_model::WPM:
    return "WordPiece";
  case tokenizer_model::UGM:
    return "Unigram";
  case tokenizer_model::RWKV:
    return "Rwkv";
  case tokenizer_model::PLAMO2:
    return "Plamo2";
  case tokenizer_model::UNKNOWN:
    return "Unknown";
  }
  return "Unknown";
}

std::uint8_t profile_flags(const vocab &value) {
  return static_cast<std::uint8_t>(
      (value.add_bos ? 1U << 0U : 0U) |
      (value.add_eos ? 1U << 1U : 0U) |
      (value.add_sep ? 1U << 2U : 0U) |
      (value.add_space_prefix ? 1U << 3U : 0U) |
      (value.remove_extra_whitespaces ? 1U << 4U : 0U) |
      (value.escape_whitespaces ? 1U << 5U : 0U) |
      (value.treat_whitespace_as_suffix ? 1U << 6U : 0U) |
      (value.ignore_merges ? 1U << 7U : 0U));
}

void write_defaults(std::ostream &output, const vocab &value) {
  output << "Defaults { bos_id: " << value.bos_id
         << ", eos_id: " << value.eos_id << ", eot_id: " << value.eot_id
         << ", eom_id: " << value.eom_id << ", unk_id: " << value.unk_id
         << ", sep_id: " << value.sep_id << ", pad_id: " << value.pad_id
         << ", cls_id: " << value.cls_id << ", mask_id: " << value.mask_id
         << ", prefix_id: " << value.prefix_id
         << ", suffix_id: " << value.suffix_id
         << ", middle_id: " << value.middle_id
         << ", fim_pre_id: " << value.fim_pre_id
         << ", fim_suf_id: " << value.fim_suf_id
         << ", fim_mid_id: " << value.fim_mid_id
         << ", fim_pad_id: " << value.fim_pad_id
         << ", fim_rep_id: " << value.fim_rep_id
         << ", fim_sep_id: " << value.fim_sep_id
         << ", flags: " << static_cast<unsigned>(profile_flags(value)) << " }";
}

void write_model(std::ostream &output, vocab &value, const std::string_view name) {
  reset_profile(value);
  value.tokenizer_model_id =
      emel::text::tokenizer::detail::tokenizer_model_from_name(name);
  emel::text::tokenizer::detail::apply_tokenizer_model_defaults(name, value);
  output << "model=" << name << " class=" << model_name(value.tokenizer_model_id)
         << " defaults=";
  write_defaults(output, value);
  output << '\n';
}

void write_pre(std::ostream &output, vocab &value, const std::string_view name) {
  reset_profile(value);
  value.tokenizer_pre_id = emel::text::tokenizer::preprocessor::detail::
      tokenizer_pre_profile_from_name(name);
  emel::text::tokenizer::preprocessor::detail::apply_tokenizer_pre_defaults(
      name, value);
  output << "pre=" << (name.empty() ? "<empty>" : name)
         << " profile=" << static_cast<unsigned>(value.tokenizer_pre_id)
         << " defaults=";
  write_defaults(output, value);
  output << '\n';
}

} // namespace

int main() {
  auto value = std::make_unique<vocab>();
  std::cout << "token-profile-parity/v1\n"
            << "source_commit=843a117386ef17dc5a50549bbfc821074c2141d6\n"
            << "model_source_blob=ef7ff8da51f1f281082901bf4919a4b9a63f2671\n"
            << "pre_source_blob=16b2982ca16dfdfbee016d50d0eb924a7cdc18c4\n";

  constexpr auto models = std::to_array<std::string_view>({
      "none", "no_vocab", "llama", "gemma4", "gpt2", "bert", "t5",
      "rwkv", "plamo2", "future",
  });
  for (const auto model : models) {
    write_model(std::cout, *value, model);
  }

  constexpr auto pre_profiles = std::to_array<std::string_view>({
      "",              "default",         "llama3",      "llama-v3",
      "llama-bpe",     "falcon3",         "falcon-h1",   "pixtral",
      "midm-2.0",      "lfm2",            "jina-v5-nano", "jais2",
      "dbrx",          "smaug",           "deepseek-llm", "deepseek-coder",
      "deepseek-v3",   "youtu",           "falcon",       "mpt",
      "starcoder",     "gpt2",            "gpt-2",        "jais",
      "refact",        "command-r",       "qwen2",        "qwen2.5",
      "qwen35",        "stablelm2",       "olmo",         "poro",
      "chatglm4",      "viking",          "tekken",       "smollm",
      "codeshell",     "bloom",           "gpt3-finnish", "exaone",
      "exaone4",       "exaone-moe",      "chameleon",    "minerva",
      "megrez",        "gpt4o",           "gpt-4o",       "tiny-aya",
      "superbpe",      "trillion",        "granite-docling",
      "bailingmoe",    "seed-coder",      "hunyuan",      "hunyuan-dense",
      "joyai-llm",     "kimi-k2",         "grok-2",       "afmoe",
      "minimax-m2",    "solar-open",
  });
  for (const auto pre : pre_profiles) {
    write_pre(std::cout, *value, pre);
  }
  write_pre(std::cout, *value, "future");
}
