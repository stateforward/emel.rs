#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <span>
#include <stdexcept>
#include <string_view>
#include <type_traits>
#include <vector>

#include "emel/gguf/loader/events.hpp"
#include "emel/gguf/loader/sm.hpp"
#include "emel/model/any.hpp"

namespace {

namespace constants {

inline constexpr std::uint32_t gguf_type_uint8 = 0U;
inline constexpr std::uint32_t gguf_type_uint32 = 4U;
inline constexpr std::uint32_t gguf_type_int32 = 5U;
inline constexpr std::uint32_t gguf_type_float32 = 6U;
inline constexpr std::uint32_t gguf_type_bool = 7U;
inline constexpr std::uint32_t gguf_type_string = 8U;
inline constexpr std::uint32_t gguf_type_array = 9U;

} // namespace constants

template <class T>
void append_scalar(std::vector<std::uint8_t> &bytes, const T value) {
  using unsigned_type = std::make_unsigned_t<T>;
  const auto raw = static_cast<unsigned_type>(value);
  for (std::size_t index = 0; index < sizeof(T); ++index) {
    bytes.push_back(static_cast<std::uint8_t>((raw >> (index * 8U)) & 0xffU));
  }
}

void append_string(std::vector<std::uint8_t> &bytes,
                   const std::string_view value) {
  append_scalar<std::uint64_t>(bytes, value.size());
  bytes.insert(bytes.end(), value.begin(), value.end());
}

struct fixture {
  std::vector<std::uint8_t> arena = {};
  std::vector<emel::gguf::loader::kv_entry> entries = {};

  void entry(const std::string_view key, const std::uint32_t kind,
             const std::span<const std::uint8_t> payload) {
    emel::gguf::loader::kv_entry item = {};
    item.key_offset = static_cast<std::uint32_t>(arena.size());
    item.key_length = static_cast<std::uint32_t>(key.size());
    arena.insert(arena.end(), key.begin(), key.end());
    item.value_offset = static_cast<std::uint32_t>(arena.size());
    item.value_length = static_cast<std::uint32_t>(payload.size());
    item.value_type = kind;
    arena.insert(arena.end(), payload.begin(), payload.end());
    entries.push_back(item);
  }

  void string(const std::string_view key, const std::string_view value) {
    std::vector<std::uint8_t> payload = {};
    append_string(payload, value);
    entry(key, constants::gguf_type_string, payload);
  }

  template <class T>
  void scalar(const std::string_view key, const std::uint32_t kind,
              const T value) {
    std::vector<std::uint8_t> payload = {};
    append_scalar<T>(payload, value);
    entry(key, kind, payload);
  }

  void boolean(const std::string_view key, const bool value) {
    const std::array<std::uint8_t, 1> payload = {
        static_cast<std::uint8_t>(value)};
    entry(key, constants::gguf_type_bool, payload);
  }

  void strings(const std::string_view key,
               const std::span<const std::string_view> values) {
    std::vector<std::uint8_t> payload = {};
    append_scalar<std::uint32_t>(payload, constants::gguf_type_string);
    append_scalar<std::uint64_t>(payload, values.size());
    for (const auto value : values) {
      append_string(payload, value);
    }
    entry(key, constants::gguf_type_array, payload);
  }

  template <class T>
  void array(const std::string_view key, const std::uint32_t element_kind,
             const std::span<const T> values) {
    std::vector<std::uint8_t> payload = {};
    append_scalar<std::uint32_t>(payload, element_kind);
    append_scalar<std::uint64_t>(payload, values.size());
    for (const auto value : values) {
      if constexpr (std::is_floating_point_v<T>) {
        using bits = std::conditional_t<sizeof(T) == 4, std::uint32_t,
                                        std::uint64_t>;
        append_scalar<bits>(payload, std::bit_cast<bits>(value));
      } else {
        append_scalar<T>(payload, value);
      }
    }
    entry(key, constants::gguf_type_array, payload);
  }

  void repeated_strings(const std::string_view key,
                        const std::string_view value,
                        const std::uint64_t count) {
    std::vector<std::uint8_t> payload = {};
    append_scalar<std::uint32_t>(payload, constants::gguf_type_string);
    append_scalar<std::uint64_t>(payload, count);
    for (std::uint64_t index = 0; index < count; ++index) {
      append_string(payload, value);
    }
    entry(key, constants::gguf_type_array, payload);
  }

  [[nodiscard]] emel::model::kv_binding binding() const {
    return {
        .arena = arena,
        .entries = entries,
    };
  }
};

std::string_view model_name(const emel::model::data::tokenizer_model value) {
  using model = emel::model::data::tokenizer_model;
  switch (value) {
  case model::NONE:
    return "None";
  case model::SPM:
    return "SentencePiece";
  case model::BPE:
    return "Bpe";
  case model::WPM:
    return "WordPiece";
  case model::UGM:
    return "Unigram";
  case model::RWKV:
    return "Rwkv";
  case model::PLAMO2:
    return "Plamo2";
  case model::UNKNOWN:
    return "Unknown";
  }
  return "Unknown";
}

std::uint8_t flags(const emel::model::data::vocab &value) {
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

struct semantic_digest {
  std::uint64_t value = 14695981039346656037ULL;

  void byte(const std::uint8_t input) {
    value ^= input;
    value *= 1099511628211ULL;
  }

  void bytes(const std::span<const std::uint8_t> input) {
    u64(input.size());
    for (const auto input_byte : input) {
      byte(input_byte);
    }
  }

  void u32(const std::uint32_t input) {
    for (unsigned shift = 0; shift < 32U; shift += 8U) {
      byte(static_cast<std::uint8_t>((input >> shift) & 0xffU));
    }
  }

  void i32(const std::int32_t input) { u32(static_cast<std::uint32_t>(input)); }

  void u64(const std::uint64_t input) {
    for (unsigned shift = 0; shift < 64U; shift += 8U) {
      byte(static_cast<std::uint8_t>((input >> shift) & 0xffU));
    }
  }
};

std::span<const std::uint8_t> c_bytes(const char *value) {
  const auto text = std::string_view{value};
  return {reinterpret_cast<const std::uint8_t *>(text.data()), text.size()};
}

std::uint64_t digest_vocab(const emel::model::data::vocab &vocab) {
  semantic_digest digest = {};
  const auto model = model_name(vocab.tokenizer_model_id);
  digest.bytes({reinterpret_cast<const std::uint8_t *>(model.data()),
                model.size()});
  digest.bytes(c_bytes(vocab.tokenizer_model_name.data()));
  digest.bytes(c_bytes(vocab.tokenizer_pre_name.data()));
  digest.u32(vocab.n_tokens);
  digest.u32(vocab.n_token_types);
  digest.u32(vocab.token_bytes_used);
  digest.u32(vocab.n_merges);
  digest.u32(vocab.merge_bytes_used);
  digest.u32(vocab.precompiled_charsmap_size);
  for (const auto id : {vocab.bos_id, vocab.eos_id, vocab.eot_id,
                        vocab.eom_id, vocab.unk_id, vocab.sep_id,
                        vocab.pad_id, vocab.cls_id, vocab.mask_id,
                        vocab.prefix_id, vocab.suffix_id, vocab.middle_id,
                        vocab.fim_pre_id, vocab.fim_suf_id, vocab.fim_mid_id,
                        vocab.fim_pad_id, vocab.fim_rep_id, vocab.fim_sep_id}) {
    digest.i32(id);
  }
  digest.byte(flags(vocab));
  for (std::uint32_t index = 0; index < vocab.n_tokens; ++index) {
    const auto &entry = vocab.entries[index];
    digest.bytes({reinterpret_cast<const std::uint8_t *>(
                      vocab.token_storage.data()) +
                      entry.text_offset,
                  entry.text_length});
    digest.u32(std::bit_cast<std::uint32_t>(entry.score));
    digest.i32(entry.type);
    digest.byte(static_cast<std::uint8_t>(
        (vocab.lstrip_flags[index / 8U] >> (index % 8U)) & 1U));
    digest.byte(static_cast<std::uint8_t>(
        (vocab.rstrip_flags[index / 8U] >> (index % 8U)) & 1U));
  }
  for (std::uint32_t index = 0; index < vocab.n_merges; ++index) {
    digest.bytes({reinterpret_cast<const std::uint8_t *>(
                      vocab.merge_storage.data()) +
                      vocab.merge_offsets[index],
                  vocab.merge_lengths[index]});
  }
  digest.bytes({vocab.precompiled_charsmap.data(),
                vocab.precompiled_charsmap_size});
  return digest.value;
}

void hex_bytes(std::ostream &output, const std::span<const std::uint8_t> bytes) {
  const auto prior_flags = output.flags();
  const auto prior_fill = output.fill();
  output << std::hex << std::setfill('0');
  for (const auto byte : bytes) {
    output << std::setw(2) << static_cast<unsigned>(byte);
  }
  output.flags(prior_flags);
  output.fill(prior_fill);
}

void observe_binding(const std::string_view case_name,
                     const emel::model::kv_binding binding) {
  auto vocab = std::make_unique<emel::model::data::vocab>();
  const bool ok = emel::model::load_vocab_from_gguf(binding, *vocab);
  std::cout << "case=" << case_name << " ok=" << static_cast<unsigned>(ok);
  if (!ok) {
    std::cout << '\n';
    return;
  }
  std::cout << " model=" << model_name(vocab->tokenizer_model_id)
            << " model_name=" << vocab->tokenizer_model_name.data()
            << " pre_name=" << vocab->tokenizer_pre_name.data()
            << " counts=" << vocab->n_tokens << ',' << vocab->n_token_types
            << ',' << vocab->token_bytes_used << ',' << vocab->n_merges << ','
            << vocab->merge_bytes_used << ',' << vocab->precompiled_charsmap_size
            << " ids=" << vocab->bos_id << ',' << vocab->eos_id << ','
            << vocab->eot_id << ',' << vocab->eom_id << ',' << vocab->unk_id
            << ',' << vocab->sep_id << ',' << vocab->pad_id << ','
            << vocab->cls_id << ',' << vocab->mask_id << ',' << vocab->prefix_id
            << ',' << vocab->suffix_id << ',' << vocab->middle_id << ','
            << vocab->fim_pre_id << ',' << vocab->fim_suf_id << ','
            << vocab->fim_mid_id << ',' << vocab->fim_pad_id << ','
            << vocab->fim_rep_id << ',' << vocab->fim_sep_id
            << " flags=" << static_cast<unsigned>(flags(*vocab)) << " digest="
            << std::hex << std::setw(16) << std::setfill('0')
            << digest_vocab(*vocab) << std::dec << std::setfill(' ')
            << " tokens=";

  const auto token_limit = std::min<std::uint32_t>(vocab->n_tokens, 3U);
  for (std::uint32_t index = 0; index < token_limit; ++index) {
    if (index != 0) {
      std::cout << ';';
    }
    const auto &entry = vocab->entries[index];
    const auto text = std::span<const std::uint8_t>{
        reinterpret_cast<const std::uint8_t *>(vocab->token_storage.data()) +
            entry.text_offset,
        entry.text_length};
    hex_bytes(std::cout, text);
    const auto lstrip =
        (vocab->lstrip_flags[index / 8U] >> (index % 8U)) & 1U;
    const auto rstrip =
        (vocab->rstrip_flags[index / 8U] >> (index % 8U)) & 1U;
    std::cout << ':' << std::bit_cast<std::uint32_t>(entry.score) << ':'
              << entry.type << ':' << static_cast<unsigned>(lstrip) << ':'
              << static_cast<unsigned>(rstrip);
  }
  std::cout << " merges=";
  const auto merge_limit = std::min<std::uint32_t>(vocab->n_merges, 3U);
  for (std::uint32_t index = 0; index < merge_limit; ++index) {
    if (index != 0) {
      std::cout << ';';
    }
    hex_bytes(std::cout,
              std::span<const std::uint8_t>{
                  reinterpret_cast<const std::uint8_t *>(
                      vocab->merge_storage.data()) +
                      vocab->merge_offsets[index],
                  vocab->merge_lengths[index]});
  }
  std::cout << " charmap=";
  const auto charmap_limit =
      std::min<std::uint32_t>(vocab->precompiled_charsmap_size, 16U);
  hex_bytes(std::cout,
            std::span<const std::uint8_t>{vocab->precompiled_charsmap.data(),
                                          charmap_limit});
  std::cout << '\n';
}

void observe(const std::string_view case_name, const fixture &input) {
  observe_binding(case_name, input.binding());
}

fixture standard_case() {
  fixture value = {};
  constexpr auto tokens = std::to_array<std::string_view>({"<|pad|>", "hello"});
  constexpr auto merges = std::to_array<std::string_view>({"h ello"});
  constexpr auto types = std::to_array<std::uint32_t>({3U, 1U});
  constexpr auto scores = std::to_array<float>({0.0F, 1.5F});
  constexpr auto charmap = std::to_array<std::uint8_t>({7U, 8U, 9U});
  value.string("tokenizer.model", "gpt2");
  value.string("tokenizer.pre", "lfm2");
  value.strings("tokenizer.tokens", tokens);
  value.array("tokenizer.token_type", constants::gguf_type_uint32,
              std::span<const std::uint32_t>{types});
  value.scalar("tokenizer.token_type_count", constants::gguf_type_uint32, 4U);
  value.array("tokenizer.scores", constants::gguf_type_float32,
              std::span<const float>{scores});
  value.strings("tokenizer.merges", merges);
  value.array("tokenizer.precompiled_charsmap", constants::gguf_type_uint8,
              std::span<const std::uint8_t>{charmap});
  value.scalar("tokenizer.bos_token_id", constants::gguf_type_uint32, 0U);
  value.scalar("tokenizer.eos_token_id", constants::gguf_type_uint32, 0U);
  value.scalar("tokenizer.padding_token_id", constants::gguf_type_uint32, 0U);
  value.boolean("tokenizer.add_eos_token", false);
  return value;
}

fixture legacy_t5_case() {
  fixture value = {};
  constexpr auto tokens =
      std::to_array<std::string_view>({"<pad>", "</s>", "<unk>", "\xE2\x96\x81"});
  constexpr auto types = std::to_array<std::uint32_t>({3U, 3U, 2U, 1U});
  value.string("tokenizer.ggml.model", "t5");
  value.string("tokenizer.ggml.pre", "default");
  value.strings("tokenizer.ggml.tokens", tokens);
  value.array("tokenizer.ggml.token_type", constants::gguf_type_uint32,
              std::span<const std::uint32_t>{types});
  value.boolean("tokenizer.ggml.add_space_prefix", true);
  value.boolean("tokenizer.ggml.remove_extra_whitespaces", true);
  return value;
}

fixture signed_case() {
  fixture value = {};
  constexpr auto tokens = std::to_array<std::string_view>({"<s>", "hello"});
  constexpr auto types = std::to_array<std::uint32_t>({3U, 1U});
  value.string("tokenizer.ggml.model", "rwkv");
  value.strings("tokenizer.ggml.tokens", tokens);
  value.array("tokenizer.ggml.token_type", constants::gguf_type_uint32,
              std::span<const std::uint32_t>{types});
  value.scalar("tokenizer.ggml.bos_token_id", constants::gguf_type_uint32, 0U);
  value.scalar("tokenizer.ggml.eos_token_id", constants::gguf_type_uint32, 0U);
  value.scalar("tokenizer.ggml.padding_token_id", constants::gguf_type_int32, -1);
  value.scalar("tokenizer.ggml.prefix_token_id", constants::gguf_type_int32, -1);
  value.scalar("tokenizer.ggml.fim_pre_token_id", constants::gguf_type_int32, -1);
  return value;
}

fixture gemma4_case() {
  fixture value = {};
  constexpr auto tokens = std::to_array<std::string_view>({"<bos>", "<eos>"});
  constexpr auto types = std::to_array<std::uint32_t>({3U, 3U});
  constexpr auto scores = std::to_array<float>({0.0F, 0.0F});
  value.string("tokenizer.ggml.model", "gemma4");
  value.strings("tokenizer.ggml.tokens", tokens);
  value.array("tokenizer.ggml.token_type", constants::gguf_type_uint32,
              std::span<const std::uint32_t>{types});
  value.scalar("tokenizer.ggml.token_type_count", constants::gguf_type_uint32, 4U);
  value.array("tokenizer.ggml.scores", constants::gguf_type_float32,
              std::span<const float>{scores});
  value.repeated_strings("tokenizer.ggml.merges", "", 400001U);
  value.scalar("tokenizer.ggml.bos_token_id", constants::gguf_type_uint32, 0U);
  value.scalar("tokenizer.ggml.eos_token_id", constants::gguf_type_uint32, 1U);
  value.boolean("tokenizer.ggml.add_bos_token", true);
  value.boolean("tokenizer.ggml.add_space_prefix", true);
  return value;
}

void noop_probe_done(const emel::gguf::loader::events::probe_done &) {}
void noop_probe_error(const emel::gguf::loader::events::probe_error &event) {
  std::cerr << "GGUF probe error=" << event.err << '\n';
}
void noop_bind_done(const emel::gguf::loader::events::bind_done &) {}
void noop_bind_error(const emel::gguf::loader::events::bind_error &event) {
  std::cerr << "GGUF bind error=" << event.err << '\n';
}
void noop_parse_done(const emel::gguf::loader::events::parse_done &) {}
void noop_parse_error(const emel::gguf::loader::events::parse_error &event) {
  std::cerr << "GGUF parse error=" << event.err << '\n';
}

struct parsed_fixture {
  std::vector<std::uint8_t> bytes = {};
  std::vector<std::uint8_t> arena = {};
  std::vector<emel::gguf::loader::kv_entry> entries = {};
  std::unique_ptr<emel::model::data> model = {};

  [[nodiscard]] emel::model::kv_binding binding() const {
    return {.arena = arena, .entries = entries};
  }
};

bool parse_file(const std::filesystem::path &path, parsed_fixture &output) {
  std::ifstream stream(path, std::ios::binary | std::ios::ate);
  if (!stream.good()) {
    return false;
  }
  const auto end = stream.tellg();
  if (end <= 0) {
    return false;
  }
  output.bytes.resize(static_cast<std::size_t>(end));
  stream.seekg(0, std::ios::beg);
  stream.read(reinterpret_cast<char *>(output.bytes.data()), end);
  if (!stream.good()) {
    return false;
  }

  emel::gguf::loader::sm loader = {};
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
      output.bytes, requirements, on_probe_done, on_probe_error};
  if (!loader.process_event(probe)) {
    std::cerr << "benchmark fixture probe failed: " << path << '\n';
    return false;
  }
  output.model = std::make_unique<emel::model::data>();
  if (requirements.tensor_count > output.model->tensors.size()) {
    std::cerr << "benchmark fixture tensor capacity failed: " << path << '\n';
    return false;
  }
  const auto entry_bytes =
      static_cast<std::uint64_t>(requirements.max_key_bytes) +
      static_cast<std::uint64_t>(requirements.max_value_bytes);
  if (requirements.kv_count != 0U &&
      entry_bytes > std::numeric_limits<std::uint64_t>::max() /
                        requirements.kv_count) {
    std::cerr << "benchmark fixture arena overflow: " << path << '\n';
    return false;
  }
  const auto arena_bytes =
      entry_bytes * static_cast<std::uint64_t>(requirements.kv_count);
  if (arena_bytes > std::numeric_limits<std::size_t>::max()) {
    std::cerr << "benchmark fixture arena capacity failed: " << path << '\n';
    return false;
  }
  output.arena.resize(
      std::max<std::size_t>(1U, static_cast<std::size_t>(arena_bytes)));
  output.entries.resize(std::max<std::uint32_t>(1U, requirements.kv_count));
  const emel::gguf::loader::event::bind_storage bind{
      output.arena, output.entries,
      std::span<emel::model::data::tensor_record>{
          output.model->tensors.data(), requirements.tensor_count},
      on_bind_done, on_bind_error};
  if (!loader.process_event(bind)) {
    std::cerr << "benchmark fixture bind failed: " << path << '\n';
    return false;
  }
  const emel::gguf::loader::event::parse parse{output.bytes, on_parse_done,
                                                on_parse_error};
  if (!loader.process_event(parse)) {
    std::cerr << "benchmark fixture parse failed: " << path << '\n';
    return false;
  }
  return true;
}

struct benchmark_result {
  double nanoseconds_per_operation = 0.0;
  std::uint64_t digest = 0U;
  std::uint32_t token_count = 0U;
  std::uint32_t merge_count = 0U;
};

benchmark_result benchmark_binding(
    const emel::model::kv_binding binding,
    const std::uint32_t expected_tokens, const std::uint32_t expected_merges,
    const std::size_t iterations, const std::size_t runs,
    const std::size_t warmup_iterations) {
  auto vocab = std::make_unique<emel::model::data::vocab>();
  const auto load_and_validate = [&] {
    if (!emel::model::load_vocab_from_gguf(binding, *vocab) ||
        vocab->n_tokens != expected_tokens ||
        vocab->n_merges != expected_merges) {
      throw std::runtime_error{"benchmark vocabulary validation failed"};
    }
  };
  load_and_validate();
  for (std::size_t index = 0; index < warmup_iterations; ++index) {
    load_and_validate();
  }
  std::vector<double> samples = {};
  samples.reserve(runs);
  for (std::size_t run = 0; run < runs; ++run) {
    const auto start = std::chrono::steady_clock::now();
    for (std::size_t iteration = 0; iteration < iterations; ++iteration) {
      load_and_validate();
    }
    const auto elapsed = std::chrono::steady_clock::now() - start;
    samples.push_back(
        std::chrono::duration<double, std::nano>{elapsed}.count() /
        static_cast<double>(iterations));
  }
  std::sort(samples.begin(), samples.end());
  return {
      .nanoseconds_per_operation = samples[samples.size() / 2U],
      .digest = digest_vocab(*vocab),
      .token_count = vocab->n_tokens,
      .merge_count = vocab->n_merges,
  };
}

void benchmark_case(const std::string_view name,
                    const std::filesystem::path &path,
                    const std::uint32_t expected_tokens,
                    const std::uint32_t expected_merges,
                    const std::size_t iterations, const std::size_t runs,
                    const std::size_t warmup_iterations) {
  parsed_fixture fixture = {};
  if (!parse_file(path, fixture)) {
    throw std::runtime_error{"failed to parse benchmark fixture"};
  }
  const auto result = benchmark_binding(
      fixture.binding(), expected_tokens, expected_merges, iterations, runs,
      warmup_iterations);
  std::cout << "case=" << name << " cpp_ns_per_op=" << std::fixed
            << std::setprecision(3) << result.nanoseconds_per_operation
            << " digest=" << std::hex << std::setw(16) << std::setfill('0')
            << result.digest << std::dec << std::setfill(' ')
            << " token_count=" << result.token_count
            << " merge_count=" << result.merge_count << '\n';
}

int benchmark_main(const int argc, const char *const *argv) {
  if (argc != 8) {
    std::cerr << "usage: observer --benchmark ITER RUNS WARMUP SMALL REAL "
                 "BOUNDARY\n";
    return 2;
  }
  const auto iterations = static_cast<std::size_t>(std::stoull(argv[2]));
  const auto runs = static_cast<std::size_t>(std::stoull(argv[3]));
  const auto warmup_iterations =
      static_cast<std::size_t>(std::stoull(argv[4]));
  if (iterations == 0U || runs == 0U) {
    throw std::runtime_error{"iterations and runs must be nonzero"};
  }
  benchmark_case("small", argv[5], 2U, 1U, iterations, runs,
                 warmup_iterations);
  benchmark_case("real_distilgpt2", argv[6], 50257U, 50000U, iterations,
                 runs, warmup_iterations);
  benchmark_case("gemma4_400001", argv[7], 2U, 400001U, iterations, runs,
                 warmup_iterations);
  return 0;
}

} // namespace

int main(const int argc, const char *const *argv) {
  if (argc > 1 && std::string_view{argv[1]} == "--benchmark") {
    return benchmark_main(argc, argv);
  }
  std::cout << "vocabulary-observer/v1\n";
  observe("standard", standard_case());
  observe("legacy_t5", legacy_t5_case());
  observe("signed", signed_case());
  observe("gemma4_400001", gemma4_case());
  for (int index = 1; index < argc; ++index) {
    const std::filesystem::path path = argv[index];
    parsed_fixture fixture = {};
    if (!parse_file(path, fixture)) {
      std::cerr << "failed to parse real fixture: " << path << '\n';
      return 2;
    }
    const auto case_name = "real/" + path.filename().string();
    observe_binding(case_name, fixture.binding());
  }
}
