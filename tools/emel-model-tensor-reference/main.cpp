#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "emel/model/tensor/errors.hpp"
#include "emel/model/tensor/events.hpp"
#include "emel/model/tensor/sm.hpp"

namespace {
namespace tensor = emel::model::tensor;

struct outcome {
  bool done{};
  emel::error::type error{};
  std::uint32_t effect_count{};
};

void bind_done(void *raw, const tensor::events::bind_storage_done &) noexcept {
  static_cast<outcome *>(raw)->done = true;
}
void bind_error(void *raw,
                const tensor::events::bind_storage_error &event) noexcept {
  static_cast<outcome *>(raw)->error = event.err;
}
void plan_done(void *raw,
               const tensor::events::plan_load_done &event) noexcept {
  auto &value = *static_cast<outcome *>(raw);
  value.done = true;
  value.effect_count = event.effect_count;
}
void plan_error(void *raw,
                const tensor::events::plan_load_error &event) noexcept {
  static_cast<outcome *>(raw)->error = event.err;
}
void apply_done(void *raw,
                const tensor::events::apply_effect_results_done &) noexcept {
  static_cast<outcome *>(raw)->done = true;
}
void apply_error(
    void *raw,
    const tensor::events::apply_effect_results_error &event) noexcept {
  static_cast<outcome *>(raw)->error = event.err;
}

void *fake(std::uintptr_t address) {
  return reinterpret_cast<void *>(address);
}

void require(bool condition, std::string_view message) {
  if (!condition) {
    throw std::runtime_error(std::string{message});
  }
}

outcome run_bind(tensor::sm &machine,
                 std::span<emel::model::data::tensor_record> records) {
  outcome value{};
  tensor::event::bind_storage request{records};
  request.on_done = {&value, bind_done};
  request.on_error = {&value, bind_error};
  (void)machine.process_event(request);
  return value;
}

outcome plan(tensor::sm &machine,
             std::span<tensor::event::effect_request> effects,
             emel::io::loader::event::strategy_kind strategy) {
  outcome value{};
  tensor::event::plan_load request{effects};
  request.strategy = strategy;
  request.on_done = {&value, plan_done};
  request.on_error = {&value, plan_error};
  (void)machine.process_event(request);
  return value;
}

outcome run_apply(tensor::sm &machine,
                  std::span<const tensor::event::effect_result> results,
                  std::span<emel::model::data::tensor_record> records = {}) {
  outcome value{};
  tensor::event::apply_effect_results request{results, records};
  request.on_done = {&value, apply_done};
  request.on_error = {&value, apply_error};
  (void)machine.process_event(request);
  return value;
}

tensor::event::tensor_state capture(tensor::sm &machine,
                                    const std::int32_t tensor_id) {
  tensor::event::tensor_state state{};
  std::int32_t error = emel::error::cast(tensor::error::none);
  tensor::event::capture_tensor_state request{};
  request.tensor_id = tensor_id;
  request.state_out = &state;
  request.error_out = &error;
  require(machine.process_event(request), "capture accepted");
  require(error == emel::error::cast(tensor::error::none), "capture done");
  return state;
}

std::string effect_text(const tensor::event::effect_request &effect) {
  std::string_view strategy = "none";
  if (effect.kind == tensor::event::effect_kind::k_io_load) {
    switch (effect.strategy) {
    case emel::io::loader::event::strategy_kind::mapped_file:
      strategy = "mapped_file";
      break;
    case emel::io::loader::event::strategy_kind::read_copy:
      strategy = "read_copy";
      break;
    case emel::io::loader::event::strategy_kind::external_buffer:
      strategy = "external_buffer";
      break;
    case emel::io::loader::event::strategy_kind::staged_read:
      strategy = "staged_read";
      break;
    default:
      strategy = "unknown";
      break;
    }
  }
  std::ostringstream text;
  text << strategy << ':' << effect.tensor_id << ':' << effect.file_index << ':'
       << effect.offset << ':' << effect.size;
  return text.str();
}

std::string_view error_text(const emel::error::type error) {
  if (error == emel::error::cast(tensor::error::invalid_request)) {
    return "invalid_request";
  }
  if (error == emel::error::cast(tensor::error::backend_error)) {
    return "backend_error";
  }
  return "other";
}

std::string_view lifecycle_text(const tensor::event::lifecycle lifecycle) {
  return lifecycle == tensor::event::lifecycle::resident ? "resident" : "other";
}

double benchmark(std::uint64_t iterations, std::size_t runs,
                 std::uint64_t warmup) {
  std::array<emel::model::data::tensor_record, 64> records{};
  for (std::size_t index = 0; index < records.size(); ++index) {
    records[index].file_offset = 4096u * (index + 1u);
    records[index].data_size = 32u;
    records[index].file_index = static_cast<std::uint16_t>(index % 4u);
  }
  tensor::sm machine{};
  require(run_bind(machine, records).done, "benchmark bind");
  std::array<tensor::event::effect_request, 64> effects{};
  std::array<tensor::event::effect_result, 64> results{};
  for (auto &result : results) {
    result.kind = tensor::event::effect_kind::k_io_load;
    result.err = emel::error::cast(tensor::error::out_of_memory);
  }
  auto execute = [&](std::uint64_t count) {
    double planned_nanoseconds = 0.0;
    for (std::uint64_t index = 0; index < count; ++index) {
      for (auto &effect : effects) {
        effect = {};
      }
      const auto begin = std::chrono::steady_clock::now();
      require(plan(machine, effects,
                   emel::io::loader::event::strategy_kind::mapped_file)
                  .done,
              "benchmark plan");
      planned_nanoseconds += std::chrono::duration<double, std::nano>(
                                 std::chrono::steady_clock::now() - begin)
                                 .count();
      require(run_apply(machine, results).error ==
                  emel::error::cast(tensor::error::backend_error),
              "benchmark apply error");
    }
    return planned_nanoseconds;
  };
  (void)execute(warmup);
  std::vector<double> samples;
  samples.reserve(runs);
  for (std::size_t run = 0; run < runs; ++run) {
    samples.push_back(execute(iterations) / static_cast<double>(iterations));
  }
  std::sort(samples.begin(), samples.end());
  return samples[samples.size() / 2u];
}

void parity() {
  tensor::sm machine{};
  std::array<emel::model::data::tensor_record, 2> records{};
  records[0].file_offset = 4096u;
  records[0].data_size = 2u;
  records[0].file_index = 1u;
  records[0].type = 7;
  records[0].data = fake(0x1000u);
  records[1].file_offset = 8192u;
  records[1].data_size = 3u;
  records[1].file_index = 2u;
  records[1].type = 7;
  records[1].data = fake(0x2000u);
  const auto bind = run_bind(machine, records);
  require(bind.done, "bind");
  const auto bound_first = capture(machine, 0);
  const auto bound_second = capture(machine, 1);
  require(bound_first.file_offset == records[0].file_offset &&
              bound_second.file_offset == records[1].file_offset,
          "bind metadata captured");

  std::array<tensor::event::effect_request, 2> effects{};
  const auto none =
      plan(machine, effects, emel::io::loader::event::strategy_kind::none);
  require(none.done && none.effect_count == 2u, "none plan");
  require(effects[0].tensor_id == 0 && effects[0].file_index == 1u &&
              effects[0].offset == 4096u && effects[0].size == 2u,
          "first none effect");
  require(effects[1].tensor_id == 1 && effects[1].file_index == 2u &&
              effects[1].offset == 8192u && effects[1].size == 3u,
          "second none effect");
  tensor::sm busy_machine{};
  require(run_bind(busy_machine, records).done, "busy bind");
  std::array<tensor::event::effect_request, 2> busy_effects{};
  require(plan(busy_machine, busy_effects,
               emel::io::loader::event::strategy_kind::none)
              .done,
          "busy initial plan");
  const auto busy =
      plan(busy_machine, busy_effects,
           emel::io::loader::event::strategy_kind::read_copy);
  require(!busy.done &&
              busy.error == emel::error::cast(tensor::error::none),
          "reference leaves a second plan unhandled");
  const bool busy_recovered_ready =
      busy_machine.is(stateforward::sml::state<tensor::ready>);
  require(busy_recovered_ready, "unexpected event recovers reference ready");
  std::array<tensor::event::effect_result, 2> bound_results{};
  bound_results[0].handle = fake(0x3000u);
  bound_results[1].handle = fake(0x4000u);
  const auto bound_apply = run_apply(machine, bound_results, records);
  require(bound_apply.done, "bound apply");
  const auto resident_first = capture(machine, 0);
  const auto resident_second = capture(machine, 1);
  require(resident_first.lifecycle_state == tensor::event::lifecycle::resident &&
              resident_first.buffer_bytes == 2u &&
              resident_second.lifecycle_state == tensor::event::lifecycle::resident &&
              resident_second.buffer_bytes == 3u,
          "bound residency captured");

  std::array<emel::model::data::tensor_record, 1> read_record{};
  read_record[0].file_offset = 16384u;
  read_record[0].data_size = 4u;
  read_record[0].file_index = 5u;
  read_record[0].type = 7;
  read_record[0].data = fake(0x5000u);
  require(run_bind(machine, read_record).done, "read bind");
  std::array<tensor::event::effect_request, 1> read_effect{};
  const auto read_plan =
      plan(machine, read_effect,
           emel::io::loader::event::strategy_kind::read_copy);
  require(read_plan.done && read_plan.effect_count == 1u, "read plan");
  require(read_effect[0].tensor_id == 0 && read_effect[0].file_index == 5u &&
              read_effect[0].offset == 16384u && read_effect[0].size == 4u,
          "read effect");
  std::array<tensor::event::effect_result, 1> owned_result{};
  owned_result[0].kind = tensor::event::effect_kind::k_io_load;
  owned_result[0].handle = fake(0x6000u);
  const auto owned_apply = run_apply(machine, owned_result, read_record);
  require(owned_apply.done, "owned apply");
  const auto owned_state = capture(machine, 0);
  require(owned_state.lifecycle_state == tensor::event::lifecycle::resident &&
              owned_state.buffer_bytes == 4u,
          "owned residency captured");

  std::array<tensor::event::effect_result, 1> failed{};
  failed[0].kind = tensor::event::effect_kind::k_io_load;
  failed[0].err = emel::error::cast(tensor::error::out_of_memory);
  std::array<tensor::event::effect_request, 1> external_effect{};
  const auto external_plan =
      plan(machine, external_effect,
           emel::io::loader::event::strategy_kind::external_buffer);
  require(external_plan.done && external_plan.effect_count == 1u &&
              external_effect[0].strategy ==
                  emel::io::loader::event::strategy_kind::external_buffer &&
              external_effect[0].tensor_id == 0 &&
              external_effect[0].file_index == 5u &&
              external_effect[0].offset == 16384u &&
              external_effect[0].size == 4u,
          "external effect");
  const auto external_error = run_apply(machine, failed);
  const bool external_ready =
      machine.is(stateforward::sml::state<tensor::ready>);
  require(external_error.error ==
              emel::error::cast(tensor::error::backend_error) &&
              external_ready,
          "external backend recovery");

  std::array<tensor::event::effect_request, 1> staged_effect{};
  const auto staged_plan =
      plan(machine, staged_effect,
           emel::io::loader::event::strategy_kind::staged_read);
  require(staged_plan.done && staged_plan.effect_count == 1u &&
              staged_effect[0].strategy ==
                  emel::io::loader::event::strategy_kind::staged_read &&
              staged_effect[0].tensor_id == 0 &&
              staged_effect[0].file_index == 5u &&
              staged_effect[0].offset == 16384u &&
              staged_effect[0].size == 4u,
          "staged effect");
  const auto staged_error = run_apply(machine, failed);
  const bool staged_ready =
      machine.is(stateforward::sml::state<tensor::ready>);
  require(staged_error.error ==
              emel::error::cast(tensor::error::backend_error) && staged_ready,
          "staged backend recovery");

  std::span<emel::model::data::tensor_record> empty{};
  const auto invalid_bind = run_bind(machine, empty);
  require(invalid_bind.error ==
              emel::error::cast(tensor::error::invalid_request),
          "invalid bind");
  const auto preserved_state = capture(machine, 0);
  require(preserved_state.file_offset == 16384u,
          "invalid bind preserves metadata");

  tensor::sm unknown_machine{};
  require(run_bind(unknown_machine, read_record).done, "unknown bind");
  std::array<tensor::event::effect_request, 1> unknown_effect{};
  const auto unknown_strategy =
      static_cast<emel::io::loader::event::strategy_kind>(255u);
  require(plan(unknown_machine, unknown_effect, unknown_strategy).done,
          "reference unknown plan");
  require(unknown_effect[0].strategy == unknown_strategy,
          "reference unknown strategy propagation");
  require(run_apply(unknown_machine, failed).error ==
              emel::error::cast(tensor::error::backend_error),
          "reference unknown recovery");

  std::array<tensor::event::effect_request, 1> mapped_effect{};
  const auto mapped_plan =
      plan(machine, mapped_effect,
           emel::io::loader::event::strategy_kind::mapped_file);
  require(mapped_plan.done && mapped_plan.effect_count == 1u &&
              mapped_effect[0].strategy ==
                  emel::io::loader::event::strategy_kind::mapped_file &&
              mapped_effect[0].tensor_id == 0 &&
              mapped_effect[0].file_index == 5u &&
              mapped_effect[0].offset == 16384u &&
              mapped_effect[0].size == 4u,
          "mapped effect");
  const auto mapped_error = run_apply(machine, failed);
  const bool mapped_ready =
      machine.is(stateforward::sml::state<tensor::ready>);
  require(mapped_error.error ==
              emel::error::cast(tensor::error::backend_error) && mapped_ready,
          "mapped backend recovery");
  const auto no_plan = run_apply(machine, failed);
  require(no_plan.error ==
              emel::error::cast(tensor::error::invalid_request),
          "ready result rejection");

  std::cout << "model-tensor-parity-snapshot/v1\n"
         "source_repository=stateforward/emel.cpp\n"
         "source_commit=843a117386ef17dc5a50549bbfc821074c2141d6\n"
         "source_tree=06306d4ffad3455fcf5df71dc692df52514b9865\n"
         "source_files=src/emel/model/tensor/events.hpp,errors.hpp,context.hpp,detail.hpp,guards.hpp,actions.hpp,sm.hpp\n"
         "source_tests=tests/model/tensor/lifecycle_tests.cpp\n"
         "fixture_config=public_actor_typed_events,owned_batches,two_tensors,supported_strategy_families_plus_rust_unknown_extension\n"
         "contract_delta=rust_owned_batches_replace_raw_spans_and_pointers;rust_busy_is_typed;mapped_success_deferred\n"
      << "reference_observation=second_plan behavior="
      << (busy_recovered_ready ? "unexpected_recovery_to_ready" : "other")
      << '\n'
      << "reference_observation=unknown_strategy behavior="
      << (unknown_effect[0].strategy == unknown_strategy ? "planned_as_io_load"
                                                         : "other")
      << '\n'
      << "case=bind outcome=" << (bind.done ? "done" : "error")
      << " active_extent=" << records.size() << '\n'
      << "case=plan_none outcome=" << (none.done ? "done" : "error")
      << " effect_count=" << none.effect_count
      << " first=" << effect_text(effects[0])
      << " second=" << effect_text(effects[1]) << '\n'
      << "case=apply_bound outcome=" << (bound_apply.done ? "done" : "error")
      << " first=resident:" << resident_first.buffer_bytes
      << " second=resident:" << resident_second.buffer_bytes << '\n'
      << "case=plan_read outcome=" << (read_plan.done ? "done" : "error")
      << " effect_count=" << read_plan.effect_count << " first="
      << effect_text(read_effect[0]) << '\n'
      << "case=apply_owned outcome=" << (owned_apply.done ? "done" : "error")
      << " lifecycle=" << lifecycle_text(owned_state.lifecycle_state)
      << " buffer_bytes="
      << owned_state.buffer_bytes << '\n'
      << "case=plan_external outcome=" << (external_plan.done ? "done" : "error")
      << " effect_count=" << external_plan.effect_count
      << " first=" << effect_text(external_effect[0]) << '\n'
      << "case=external_error outcome="
      << (external_error.error == emel::error::cast(tensor::error::none) ? "done" : "error")
      << " error=" << error_text(external_error.error)
      << " phase=" << (external_ready ? "ready" : "other") << '\n'
      << "case=plan_staged outcome=" << (staged_plan.done ? "done" : "error")
      << " effect_count=" << staged_plan.effect_count
      << " first=" << effect_text(staged_effect[0]) << '\n'
      << "case=staged_error outcome="
      << (staged_error.error == emel::error::cast(tensor::error::none) ? "done" : "error")
      << " error=" << error_text(staged_error.error)
      << " phase=" << (staged_ready ? "ready" : "other") << '\n'
      << "case=invalid_bind outcome="
      << (invalid_bind.error == emel::error::cast(tensor::error::none) ? "done" : "error")
      << " error=" << error_text(invalid_bind.error)
      << " preserved_offset=" << preserved_state.file_offset << '\n'
      << "case=plan_mapped outcome=" << (mapped_plan.done ? "done" : "error")
      << " effect_count=" << mapped_plan.effect_count
      << " first=" << effect_text(mapped_effect[0]) << '\n'
      << "case=mapped_error outcome="
      << (mapped_error.error == emel::error::cast(tensor::error::none) ? "done" : "error")
      << " error=" << error_text(mapped_error.error)
      << " phase=" << (mapped_ready ? "ready" : "other") << '\n'
      << "case=result_without_plan outcome="
      << (no_plan.error == emel::error::cast(tensor::error::none) ? "done" : "error")
      << " error=" << error_text(no_plan.error)
      << '\n';
}
} // namespace

int main(int argc, char **argv) {
  try {
    if (argc == 5 && std::string_view{argv[1]} == "--benchmark") {
      const auto iterations = std::stoull(argv[2]);
      const auto runs = std::stoull(argv[3]);
      const auto warmup = std::stoull(argv[4]);
      std::cout << "model/tensor/reference/plan_mapped_64 ns_per_op="
                << std::fixed << std::setprecision(3)
                << benchmark(iterations, runs, warmup)
                << " iter=" << iterations << " runs=" << runs << '\n';
      return 0;
    }
    parity();
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "emel-model-tensor-reference: " << error.what() << '\n';
    return 2;
  }
}
