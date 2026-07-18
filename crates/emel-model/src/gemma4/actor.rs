//! Public Gemma4 actor and local generated-machine dispatch wrapper.

#![allow(
    clippy::missing_const_for_fn,
    clippy::needless_pass_by_ref_mut,
    clippy::trivially_copy_pass_by_ref,
    clippy::unnecessary_wraps,
    clippy::unused_self
)]

use core::cell::{Cell, RefCell};
use core::fmt;

use emel_kernels::capability;

use crate::attention_family::context::{self, Context, Policy};
use crate::catalog::Catalog;
use crate::generation::{
    self, AttentionQkNormRoute, AttentionVNormRoute, AttentionValueRoute, AttentionWindowRoute,
    ContractDescriptor, LayerExecution, ResidualRoute,
};

pub(super) use crate::attention_family::context::{
    AuditRuntime, BlockRuntime, BlockVisitRuntime, PlanRuntime, ReleaseRuntime, ResetRuntime,
    StageRuntime, TopologyRuntime, UnexpectedRuntime, ValidateRuntime, VisitRuntime,
};

use super::event;
use super::sm::{Gemma4MachineEvents, Gemma4MachineStateMachine, Gemma4MachineStateMachineContext};
use super::{
    ARCHITECTURE_NAME, BLOCK_COUNT, DEDICATED_KV_BLOCK_TENSOR_COUNT, GLOBAL_TENSOR_COUNT,
    OUTPUT_NORM_NAME, TOKEN_EMBEDDING_NAME,
};

type Error = generation::event::Error;
type ChildResult = Result<(), Error>;

#[derive(Clone, Copy)]
pub(super) struct BeginRuntime<'a> {
    common: context::BeginRuntime<'a>,
    parameters: &'a event::Parameters,
    layer_count: i32,
    execution_contract: bool,
}

pub(super) struct Gemma4Policy;
impl Policy for Gemma4Policy {
    const ARCHITECTURE_NAME: &'static [u8] = ARCHITECTURE_NAME;
    const TOKEN_EMBEDDING_NAME: &'static [u8] = TOKEN_EMBEDDING_NAME;
    const OUTPUT_NORM_NAME: &'static [u8] = OUTPUT_NORM_NAME;
    const GLOBAL_TENSOR_COUNT: u32 = GLOBAL_TENSOR_COUNT;
    const BLOCK_TENSOR_COUNT: u32 = DEDICATED_KV_BLOCK_TENSOR_COUNT;
    const QK_NORM_ROUTE: AttentionQkNormRoute = AttentionQkNormRoute::HeadwiseRms;
    const TIED_OUTPUT: bool = true;
}

struct Gemma4Context {
    family: Context<Gemma4Policy>,
    parameters: event::Parameters,
}

impl Gemma4Context {
    #[allow(clippy::result_large_err)]
    fn new(
        catalog: Catalog,
        capability: capability::Resolver,
        storage: event::Storage,
    ) -> Result<Self, event::StorageBindError> {
        Ok(Self {
            family: Context::new(catalog, capability, storage)?,
            parameters: event::Parameters::default(),
        })
    }

    const fn family_facts(&self) -> context::Parameters {
        self.family.facts()
    }

    const fn parameters(&self) -> event::Parameters {
        self.parameters
    }

    const fn builder_mut(&mut self) -> &mut generation::Builder {
        self.family.builder_mut()
    }
}

macro_rules! forward_family_guards {
    ($($name:ident: $event:ty),+ $(,)?) => {
        $(
            fn $name(&self, event: &$event) -> Result<bool, ()> {
                self.family.$name(event)
            }
        )+
    };
}

macro_rules! forward_family_effects {
    ($($name:ident: $event:ty),+ $(,)?) => {
        $(
            fn $name(&mut self, event: $event) -> Result<(), ()> {
                self.family.$name(event)
            }
        )+
    };
}

fn is_execution_architecture(architecture: &[u8]) -> bool {
    architecture == ARCHITECTURE_NAME
}

const fn is_fixed_shared_kv_layer(block_index: i32) -> bool {
    block_index >= BLOCK_COUNT - super::SHARED_KV_LAYERS && block_index < BLOCK_COUNT
}

impl Gemma4Context {
    fn block_index(&self, event: &BlockRuntime<'_>) -> Option<usize> {
        let facts = self.family_facts();
        let index = usize::try_from(event.event.index).ok()?;
        (index < usize::try_from(facts.block_count).ok()?).then_some(index)
    }

    fn block_is_bound_shared(&self, event: &BlockRuntime<'_>) -> bool {
        let facts = self.family_facts();
        let parameters = self.parameters();
        let Some(index) = self.block_index(event) else {
            return false;
        };
        let Ok(shared) = usize::try_from(parameters.attention_shared_kv_layers) else {
            return false;
        };
        let Ok(blocks) = usize::try_from(facts.block_count) else {
            return false;
        };
        shared > 0 && shared <= blocks && index >= blocks - shared
    }

    fn block_requires_dedicated_value(&self, event: &BlockRuntime<'_>) -> bool {
        !is_fixed_shared_kv_layer(event.event.index)
    }

    fn block_is_sliding(&self, event: &BlockRuntime<'_>) -> bool {
        let parameters = self.parameters();
        self.block_index(event).is_some_and(|index| {
            index < parameters.sliding_window_pattern_count as usize
                && parameters
                    .sliding_window_pattern_flags
                    .get(index)
                    .is_some_and(|flag| *flag != 0)
        })
    }

    fn shared_topology_values(&self) -> Option<(u32, u64)> {
        let facts = self.family_facts();
        let parameters = self.parameters();
        let blocks = u32::try_from(facts.block_count).ok()?;
        let shared = u32::try_from(parameters.attention_shared_kv_layers).ok()?;
        let dedicated = blocks.checked_sub(shared)?;
        let embedding = u64::try_from(facts.embedding_length).ok()?;
        let tensors = shared
            .checked_mul(super::SHARED_KV_BLOCK_TENSOR_COUNT)?
            .checked_add(dedicated.checked_mul(DEDICATED_KV_BLOCK_TENSOR_COUNT)?)?
            .checked_add(GLOBAL_TENSOR_COUNT)?;
        let workspace = u64::from(tensors).checked_mul(embedding)?.checked_mul(4)?;
        Some((tensors, workspace))
    }

    fn dedicated_topology_values(&self) -> Option<(u32, u64)> {
        let facts = self.family_facts();
        let blocks = u32::try_from(facts.block_count).ok()?;
        let embedding = u64::try_from(facts.embedding_length).ok()?;
        let tensors = blocks
            .checked_mul(DEDICATED_KV_BLOCK_TENSOR_COUNT)?
            .checked_add(GLOBAL_TENSOR_COUNT)?;
        let workspace = u64::from(tensors).checked_mul(embedding)?.checked_mul(4)?;
        Some((tensors, workspace))
    }

    fn guard_begin_profile_execution_valid(&self, event: &BeginRuntime<'_>) -> Result<bool, ()> {
        let parameters = event.parameters;
        Ok(is_execution_architecture(event.common.event.architecture)
            && event.execution_contract
            && event.layer_count == BLOCK_COUNT
            && parameters.block_count == BLOCK_COUNT
            && parameters.context_length == super::CONTEXT_LENGTH
            && parameters.embedding_length == super::EMBEDDING_LENGTH
            && parameters.embedding_length_out == super::EMBEDDING_LENGTH
            && parameters.embedding_length_per_layer_input
                == super::EMBEDDING_LENGTH_PER_LAYER_INPUT
            && parameters.feed_forward_length == super::FEED_FORWARD_LENGTH
            && parameters.attention_head_count == super::HEAD_COUNT
            && parameters.attention_head_count_kv == super::HEAD_COUNT_KV
            && parameters.attention_key_length == super::KEY_LENGTH
            && parameters.attention_key_length_swa == super::KEY_LENGTH_SWA
            && parameters.attention_value_length == super::VALUE_LENGTH
            && parameters.attention_value_length_swa == super::VALUE_LENGTH_SWA
            && parameters.vocab_size == super::VOCAB_SIZE
            && parameters.attention_sliding_window == super::SLIDING_WINDOW
            && parameters.attention_shared_kv_layers == super::SHARED_KV_LAYERS
            && parameters.full_attention_interval == super::FULL_ATTENTION_INTERVAL
            && parameters.attention_layer_norm_rms_epsilon == super::LAYER_NORM_RMS_EPSILON
            && parameters.rope_dimension_count == super::ROPE_DIMENSION_COUNT
            && parameters.rope_dimension_count_swa == super::ROPE_DIMENSION_COUNT_SWA
            && parameters.rope_freq_base == super::ROPE_FREQ_BASE
            && parameters.rope_freq_base_swa == super::ROPE_FREQ_BASE_SWA
            && parameters.tie_word_embeddings
            && parameters.sliding_window_pattern_count == BLOCK_COUNT as u32
            && parameters.sliding_window_pattern_flags[..BLOCK_COUNT as usize]
                == super::SLIDING_WINDOW_PATTERN)
    }

    fn guard_begin_profile_validation_valid(&self, event: &BeginRuntime<'_>) -> Result<bool, ()> {
        let parameters = event.parameters;
        Ok(is_execution_architecture(event.common.event.architecture)
            && !event.execution_contract
            && event.layer_count > 0
            && i32::try_from(super::MAX_BLOCKS).is_ok_and(|maximum| event.layer_count <= maximum)
            && event.layer_count == parameters.block_count
            && parameters.context_length > 0
            && parameters.embedding_length > 0
            && parameters.embedding_length_out > 0
            && parameters.feed_forward_length > 0
            && parameters.attention_head_count > 0
            && parameters.attention_head_count_kv > 0
            && parameters.attention_key_length > 0
            && parameters.attention_value_length > 0
            && parameters.vocab_size > 0
            && parameters.attention_layer_norm_rms_epsilon > 0.0
            && parameters.rope_freq_base > 0.0
            && parameters.tie_word_embeddings)
    }

    fn guard_begin_profile_parameters_invalid(&self, event: &BeginRuntime<'_>) -> Result<bool, ()> {
        Ok(is_execution_architecture(event.common.event.architecture)
            && !self.guard_begin_profile_execution_valid(event)?
            && !self.guard_begin_profile_validation_valid(event)?)
    }

    fn effect_begin_profile_child(&mut self, event: BeginRuntime<'_>) -> Result<(), ()> {
        self.family.effect_begin_child(event.common)
    }

    fn guard_block_dedicated_full(&self, event: &BlockRuntime<'_>) -> Result<bool, ()> {
        Ok(self.block_index(event).is_some()
            && !self.block_is_bound_shared(event)
            && !self.block_is_sliding(event))
    }

    fn guard_block_shared_full(&self, event: &BlockRuntime<'_>) -> Result<bool, ()> {
        Ok(self.block_index(event).is_some()
            && self.block_is_bound_shared(event)
            && !self.block_requires_dedicated_value(event)
            && !self.block_is_sliding(event))
    }

    fn guard_block_shared_full_required_value(&self, event: &BlockRuntime<'_>) -> Result<bool, ()> {
        Ok(self.block_index(event).is_some()
            && self.block_is_bound_shared(event)
            && self.block_requires_dedicated_value(event)
            && !self.block_is_sliding(event))
    }

    fn effect_block_dedicated_full_child(&mut self, event: BlockRuntime<'_>) -> Result<(), ()> {
        let facts = self.family_facts();
        let layer = LayerExecution::new(
            ResidualRoute::Attention,
            AttentionQkNormRoute::HeadwiseRms,
            AttentionValueRoute::DedicatedValue,
            AttentionVNormRoute::None,
            AttentionWindowRoute::FullContext,
            facts.attention_key_length,
            facts.attention_value_length,
            facts.rope_dimension_count,
            facts.rope_freq_base,
        );
        event.child_result.set(self.builder_mut().process_event(
            generation::event::AttentionBlock::new(event.event.index, layer),
        ));
        Ok(())
    }

    fn effect_block_shared_full_child(&mut self, event: BlockRuntime<'_>) -> Result<(), ()> {
        let facts = self.family_facts();
        let layer = LayerExecution::new(
            ResidualRoute::Attention,
            AttentionQkNormRoute::HeadwiseRms,
            AttentionValueRoute::SharedKeyValue,
            AttentionVNormRoute::Rms,
            AttentionWindowRoute::FullContext,
            facts.attention_key_length,
            facts.attention_value_length,
            facts.rope_dimension_count,
            facts.rope_freq_base,
        );
        event.child_result.set(self.builder_mut().process_event(
            generation::event::AttentionBlock::new(event.event.index, layer),
        ));
        Ok(())
    }

    fn effect_block_shared_full_required_value_child(
        &mut self,
        event: BlockRuntime<'_>,
    ) -> Result<(), ()> {
        let facts = self.family_facts();
        let layer = LayerExecution::new(
            ResidualRoute::Attention,
            AttentionQkNormRoute::HeadwiseRms,
            AttentionValueRoute::SharedKeyValue,
            AttentionVNormRoute::Rms,
            AttentionWindowRoute::FullContext,
            facts.attention_key_length,
            facts.attention_value_length,
            facts.rope_dimension_count,
            facts.rope_freq_base,
        );
        event.child_result.set(self.builder_mut().process_event(
            generation::event::AttentionBlock::requiring_dedicated_value(event.event.index, layer),
        ));
        Ok(())
    }

    fn guard_block_sliding_key_swa(&self, event: &BlockRuntime<'_>) -> Result<bool, ()> {
        Ok(self.block_index(event).is_some()
            && self.block_is_sliding(event)
            && self.parameters().attention_key_length_swa > 0)
    }

    fn guard_block_sliding_key_fallback(&self, event: &BlockRuntime<'_>) -> Result<bool, ()> {
        Ok(self.block_index(event).is_some()
            && self.block_is_sliding(event)
            && self.parameters().attention_key_length_swa <= 0)
    }

    fn effect_select_key_swa(&mut self, event: BlockRuntime<'_>) -> Result<(), ()> {
        event
            .key_length
            .set(self.parameters().attention_key_length_swa);
        Ok(())
    }

    fn effect_select_key_fallback(&mut self, event: BlockRuntime<'_>) -> Result<(), ()> {
        event
            .key_length
            .set(self.family_facts().attention_key_length);
        Ok(())
    }

    fn guard_value_swa(&self, _event: &BlockRuntime<'_>) -> Result<bool, ()> {
        Ok(self.parameters().attention_value_length_swa > 0)
    }

    fn guard_value_fallback(&self, _event: &BlockRuntime<'_>) -> Result<bool, ()> {
        Ok(self.parameters().attention_value_length_swa <= 0)
    }

    fn effect_select_value_swa(&mut self, event: BlockRuntime<'_>) -> Result<(), ()> {
        event
            .value_length
            .set(self.parameters().attention_value_length_swa);
        Ok(())
    }

    fn effect_select_value_fallback(&mut self, event: BlockRuntime<'_>) -> Result<(), ()> {
        event
            .value_length
            .set(self.family_facts().attention_value_length);
        Ok(())
    }

    fn guard_rope_dimension_swa(&self, _event: &BlockRuntime<'_>) -> Result<bool, ()> {
        Ok(self.parameters().rope_dimension_count_swa > 0)
    }

    fn guard_rope_dimension_fallback(&self, _event: &BlockRuntime<'_>) -> Result<bool, ()> {
        Ok(self.parameters().rope_dimension_count_swa <= 0)
    }

    fn effect_select_rope_dimension_swa(&mut self, event: BlockRuntime<'_>) -> Result<(), ()> {
        event
            .rope_dimension
            .set(self.parameters().rope_dimension_count_swa);
        Ok(())
    }

    fn effect_select_rope_dimension_fallback(&mut self, event: BlockRuntime<'_>) -> Result<(), ()> {
        event
            .rope_dimension
            .set(self.family_facts().rope_dimension_count);
        Ok(())
    }

    fn guard_rope_frequency_swa(&self, _event: &BlockRuntime<'_>) -> Result<bool, ()> {
        Ok(self.parameters().rope_freq_base_swa > 0.0)
    }

    fn guard_rope_frequency_fallback(&self, _event: &BlockRuntime<'_>) -> Result<bool, ()> {
        Ok(self.parameters().rope_freq_base_swa <= 0.0)
    }

    fn effect_select_rope_frequency_swa(&mut self, event: BlockRuntime<'_>) -> Result<(), ()> {
        event
            .rope_frequency
            .set(self.parameters().rope_freq_base_swa);
        Ok(())
    }

    fn effect_select_rope_frequency_fallback(&mut self, event: BlockRuntime<'_>) -> Result<(), ()> {
        event.rope_frequency.set(self.family_facts().rope_freq_base);
        Ok(())
    }

    fn guard_block_selected_dedicated(&self, event: &BlockRuntime<'_>) -> Result<bool, ()> {
        Ok(!self.block_is_bound_shared(event))
    }

    fn guard_block_selected_shared(&self, event: &BlockRuntime<'_>) -> Result<bool, ()> {
        Ok(self.block_is_bound_shared(event) && !self.block_requires_dedicated_value(event))
    }

    fn guard_block_selected_shared_required_value(
        &self,
        event: &BlockRuntime<'_>,
    ) -> Result<bool, ()> {
        Ok(self.block_is_bound_shared(event) && self.block_requires_dedicated_value(event))
    }

    fn effect_block_selected_dedicated_child(&mut self, event: BlockRuntime<'_>) -> Result<(), ()> {
        let layer = LayerExecution::new(
            ResidualRoute::Attention,
            AttentionQkNormRoute::HeadwiseRms,
            AttentionValueRoute::DedicatedValue,
            AttentionVNormRoute::None,
            AttentionWindowRoute::SlidingWindow,
            event.key_length.get(),
            event.value_length.get(),
            event.rope_dimension.get(),
            event.rope_frequency.get(),
        );
        event.child_result.set(self.builder_mut().process_event(
            generation::event::AttentionBlock::new(event.event.index, layer),
        ));
        Ok(())
    }

    fn effect_block_selected_shared_child(&mut self, event: BlockRuntime<'_>) -> Result<(), ()> {
        let layer = LayerExecution::new(
            ResidualRoute::Attention,
            AttentionQkNormRoute::HeadwiseRms,
            AttentionValueRoute::SharedKeyValue,
            AttentionVNormRoute::Rms,
            AttentionWindowRoute::SlidingWindow,
            event.key_length.get(),
            event.value_length.get(),
            event.rope_dimension.get(),
            event.rope_frequency.get(),
        );
        event.child_result.set(self.builder_mut().process_event(
            generation::event::AttentionBlock::new(event.event.index, layer),
        ));
        Ok(())
    }

    fn effect_block_selected_shared_required_value_child(
        &mut self,
        event: BlockRuntime<'_>,
    ) -> Result<(), ()> {
        let layer = LayerExecution::new(
            ResidualRoute::Attention,
            AttentionQkNormRoute::HeadwiseRms,
            AttentionValueRoute::SharedKeyValue,
            AttentionVNormRoute::Rms,
            AttentionWindowRoute::SlidingWindow,
            event.key_length.get(),
            event.value_length.get(),
            event.rope_dimension.get(),
            event.rope_frequency.get(),
        );
        event.child_result.set(self.builder_mut().process_event(
            generation::event::AttentionBlock::requiring_dedicated_value(event.event.index, layer),
        ));
        Ok(())
    }

    fn guard_topology_shared_valid(&self, _event: &TopologyRuntime<'_>) -> Result<bool, ()> {
        let facts = self.family_facts();
        let parameters = self.parameters();
        Ok(parameters.attention_shared_kv_layers > 0
            && parameters.attention_shared_kv_layers <= facts.block_count
            && self.shared_topology_values().is_some())
    }

    fn guard_topology_dedicated_valid(&self, _event: &TopologyRuntime<'_>) -> Result<bool, ()> {
        let facts = self.family_facts();
        let parameters = self.parameters();
        Ok((parameters.attention_shared_kv_layers <= 0
            || parameters.attention_shared_kv_layers > facts.block_count)
            && self.dedicated_topology_values().is_some())
    }

    fn guard_topology_profile_invalid(&self, event: &TopologyRuntime<'_>) -> Result<bool, ()> {
        Ok(!self.guard_topology_shared_valid(event)?
            && !self.guard_topology_dedicated_valid(event)?)
    }

    fn effect_topology_shared_child(&mut self, event: TopologyRuntime<'_>) -> Result<(), ()> {
        let (tensor_count, workspace) = self
            .shared_topology_values()
            .expect("guarded shared topology");
        event.child_result.set(
            self.builder_mut()
                .process_event(generation::event::Topology::new(
                    tensor_count,
                    tensor_count,
                    4,
                    workspace,
                )),
        );
        Ok(())
    }

    fn effect_topology_dedicated_child(&mut self, event: TopologyRuntime<'_>) -> Result<(), ()> {
        let (tensor_count, workspace) = self
            .dedicated_topology_values()
            .expect("guarded dedicated topology");
        event.child_result.set(
            self.builder_mut()
                .process_event(generation::event::Topology::new(
                    tensor_count,
                    tensor_count,
                    4,
                    workspace,
                )),
        );
        Ok(())
    }
}

impl Gemma4MachineStateMachineContext for Gemma4Context {
    forward_family_guards!(
        guard_block_invalid: BlockRuntime<'_>,
        guard_block_child_ok: BlockRuntime<'_>,
        guard_block_child_error: BlockRuntime<'_>,
        guard_topology_child_ok: TopologyRuntime<'_>,
        guard_topology_child_error: TopologyRuntime<'_>,
        guard_plan_child_ok: PlanRuntime<'_>,
        guard_plan_child_error: PlanRuntime<'_>,
        guard_validate_child_ok: ValidateRuntime<'_>,
        guard_validate_child_error: ValidateRuntime<'_>,
        guard_validate_visit_ok: ValidateRuntime<'_>,
        guard_validate_visit_error: ValidateRuntime<'_>,
        guard_audit_child_ok: AuditRuntime<'_>,
        guard_audit_child_error: AuditRuntime<'_>,
        guard_stage_child_ok: StageRuntime<'_>,
        guard_stage_child_error: StageRuntime<'_>,
        guard_visit_child_ok: VisitRuntime<'_>,
        guard_visit_child_error: VisitRuntime<'_>,
        guard_block_visit_valid: BlockVisitRuntime<'_>,
        guard_block_visit_invalid: BlockVisitRuntime<'_>,
        guard_reset_child_ok: ResetRuntime<'_>,
        guard_reset_child_error: ResetRuntime<'_>,
        guard_release_child_ok: ReleaseRuntime<'_>,
        guard_release_child_error: ReleaseRuntime<'_>,
        guard_never: UnexpectedRuntime,
    );
    crate::attention_family::forward_guards!(Gemma4Context;
        guard_begin_profile_execution_valid: BeginRuntime<'_>,
        guard_begin_profile_validation_valid: BeginRuntime<'_>,
        guard_begin_profile_parameters_invalid: BeginRuntime<'_>,
        guard_block_dedicated_full: BlockRuntime<'_>,
        guard_block_shared_full: BlockRuntime<'_>,
        guard_block_shared_full_required_value: BlockRuntime<'_>,
        guard_block_sliding_key_swa: BlockRuntime<'_>,
        guard_block_sliding_key_fallback: BlockRuntime<'_>,
        guard_value_swa: BlockRuntime<'_>,
        guard_value_fallback: BlockRuntime<'_>,
        guard_rope_dimension_swa: BlockRuntime<'_>,
        guard_rope_dimension_fallback: BlockRuntime<'_>,
        guard_rope_frequency_swa: BlockRuntime<'_>,
        guard_rope_frequency_fallback: BlockRuntime<'_>,
        guard_block_selected_dedicated: BlockRuntime<'_>,
        guard_block_selected_shared: BlockRuntime<'_>,
        guard_block_selected_shared_required_value: BlockRuntime<'_>,
        guard_topology_shared_valid: TopologyRuntime<'_>,
        guard_topology_dedicated_valid: TopologyRuntime<'_>,
        guard_topology_profile_invalid: TopologyRuntime<'_>,
    );
    forward_family_effects!(
        effect_block_invalid: BlockRuntime<'_>,
        effect_child_ok_block: BlockRuntime<'_>,
        effect_child_error_block: BlockRuntime<'_>,
        effect_topology_invalid: TopologyRuntime<'_>,
        effect_child_ok_topology: TopologyRuntime<'_>,
        effect_child_error_topology: TopologyRuntime<'_>,
        effect_plan_child: PlanRuntime<'_>,
        effect_child_ok_plan: PlanRuntime<'_>,
        effect_child_error_plan: PlanRuntime<'_>,
        effect_validate_child: ValidateRuntime<'_>,
        effect_validate_visit: ValidateRuntime<'_>,
        effect_cache_validated_block: ValidateRuntime<'_>,
        effect_validate_visit_error: ValidateRuntime<'_>,
        effect_child_error_validate: ValidateRuntime<'_>,
        effect_audit_child: AuditRuntime<'_>,
        effect_child_ok_audit: AuditRuntime<'_>,
        effect_child_error_audit: AuditRuntime<'_>,
        effect_stage_child: StageRuntime<'_>,
        effect_child_ok_stage: StageRuntime<'_>,
        effect_child_error_stage: StageRuntime<'_>,
        effect_visit_child: VisitRuntime<'_>,
        effect_visit_ok: VisitRuntime<'_>,
        effect_visit_error: VisitRuntime<'_>,
        effect_block_visit: BlockVisitRuntime<'_>,
        effect_invalid_block_visit: BlockVisitRuntime<'_>,
        effect_reset_child: ResetRuntime<'_>,
        effect_reset_ok: ResetRuntime<'_>,
        effect_reset_error: ResetRuntime<'_>,
        effect_release_child: ReleaseRuntime<'_>,
        effect_release_ok: ReleaseRuntime<'_>,
        effect_release_error: ReleaseRuntime<'_>,
        effect_busy_release: ReleaseRuntime<'_>,
    );
    crate::attention_family::forward_effects!(Gemma4Context;
        effect_begin_profile_child: BeginRuntime<'_>,
        effect_block_dedicated_full_child: BlockRuntime<'_>,
        effect_block_shared_full_child: BlockRuntime<'_>,
        effect_block_shared_full_required_value_child: BlockRuntime<'_>,
        effect_select_key_swa: BlockRuntime<'_>,
        effect_select_key_fallback: BlockRuntime<'_>,
        effect_select_value_swa: BlockRuntime<'_>,
        effect_select_value_fallback: BlockRuntime<'_>,
        effect_select_rope_dimension_swa: BlockRuntime<'_>,
        effect_select_rope_dimension_fallback: BlockRuntime<'_>,
        effect_select_rope_frequency_swa: BlockRuntime<'_>,
        effect_select_rope_frequency_fallback: BlockRuntime<'_>,
        effect_block_selected_dedicated_child: BlockRuntime<'_>,
        effect_block_selected_shared_child: BlockRuntime<'_>,
        effect_block_selected_shared_required_value_child: BlockRuntime<'_>,
        effect_topology_shared_child: TopologyRuntime<'_>,
        effect_topology_dedicated_child: TopologyRuntime<'_>,
    );
    fn guard_begin_architecture_invalid(&self, event: &BeginRuntime<'_>) -> Result<bool, ()> {
        self.family.guard_begin_architecture_invalid(&event.common)
    }
    fn guard_begin_child_ok(&self, event: &BeginRuntime<'_>) -> Result<bool, ()> {
        self.family.guard_begin_child_ok(&event.common)
    }
    fn guard_begin_child_error(&self, event: &BeginRuntime<'_>) -> Result<bool, ()> {
        self.family.guard_begin_child_error(&event.common)
    }
    fn guard_global_child_ok(&self, event: &BeginRuntime<'_>) -> Result<bool, ()> {
        self.family.guard_global_child_ok(&event.common)
    }
    fn guard_global_child_error(&self, event: &BeginRuntime<'_>) -> Result<bool, ()> {
        self.family.guard_global_child_error(&event.common)
    }
    fn guard_begin_reset_ok(&self, event: &BeginRuntime<'_>) -> Result<bool, ()> {
        self.family.guard_begin_reset_ok(&event.common)
    }
    fn guard_begin_reset_error(&self, event: &BeginRuntime<'_>) -> Result<bool, ()> {
        self.family.guard_begin_reset_error(&event.common)
    }
    fn effect_global_child(&mut self, event: BeginRuntime<'_>) -> Result<(), ()> {
        self.family.effect_global_child(event.common)
    }
    fn effect_begin_child_error(&mut self, event: BeginRuntime<'_>) -> Result<(), ()> {
        self.family.effect_begin_child_error(event.common)
    }
    fn effect_begin_complete(&mut self, event: BeginRuntime<'_>) -> Result<(), ()> {
        self.parameters = *event.parameters;
        self.family.effect_begin_complete(event.common)
    }
    fn effect_begin_reset(&mut self, event: BeginRuntime<'_>) -> Result<(), ()> {
        self.family.effect_begin_reset(event.common)
    }
    fn effect_global_child_error(&mut self, event: BeginRuntime<'_>) -> Result<(), ()> {
        self.family.effect_global_child_error(event.common)
    }
    fn effect_begin_reset_error(&mut self, event: BeginRuntime<'_>) -> Result<(), ()> {
        self.family.effect_begin_reset_error(event.common)
    }
    fn effect_begin_invalid_request(&mut self, event: BeginRuntime<'_>) -> Result<(), ()> {
        self.family.effect_begin_invalid_request(event.common)
    }
    fn effect_begin_model_invalid(&mut self, event: BeginRuntime<'_>) -> Result<(), ()> {
        self.family.effect_begin_model_invalid(event.common)
    }
    fn effect_busy_begin(&mut self, event: BeginRuntime<'_>) -> Result<(), ()> {
        self.family.effect_busy_begin(event.common)
    }
    fn effect_unexpected(&mut self) -> Result<(), ()> {
        self.family.effect_unexpected()
    }
}

/// Single-writer, run-to-completion Gemma4 family actor.
pub struct Gemma4 {
    machine: Gemma4MachineStateMachine<Gemma4Context>,
}

impl Gemma4 {
    /// Constructs a Gemma4 actor from public child actors and caller-preallocated storage.
    ///
    /// # Errors
    ///
    /// A rejected bind returns every family storage region.
    #[allow(clippy::result_large_err)]
    pub fn new(
        catalog: Catalog,
        capability: capability::Resolver,
        storage: event::Storage,
    ) -> Result<Self, event::StorageBindError> {
        Ok(Self {
            machine: Gemma4MachineStateMachine::new(Gemma4Context::new(
                catalog, capability, storage,
            )?),
        })
    }

    /// Dispatches one public event synchronously.
    #[inline]
    pub fn process_event<E: event::Event>(&mut self, event: E) -> E::Output {
        event.dispatch(self)
    }

    pub(crate) fn contract_begin(&mut self, event: event::ContractBegin<'_>) -> Result<(), Error> {
        let begin_result = Cell::new(Err(Error::Internal));
        let global_result = Cell::new(Err(Error::Internal));
        let reset_result = Cell::new(Err(Error::Internal));
        let result = Cell::new(Err(Error::UnexpectedEvent));
        let parameters = context::Parameters {
            context_length: event.parameters.context_length,
            embedding_length: event.parameters.embedding_length,
            attention_key_length: event.parameters.attention_key_length,
            attention_value_length: event.parameters.attention_value_length,
            rope_dimension_count: event.parameters.rope_dimension_count,
            block_count: event.parameters.block_count,
            rope_freq_base: event.parameters.rope_freq_base,
        };
        self.machine
            .process_event(Gemma4MachineEvents::Begin(BeginRuntime {
                common: context::BeginRuntime {
                    event: crate::attention_family::event::ContractBegin::new(
                        event.architecture,
                        event.model,
                        &parameters,
                    ),
                    begin_result: &begin_result,
                    global_result: &global_result,
                    reset_result: &reset_result,
                    result: &result,
                },
                parameters: event.parameters,
                layer_count: event.layer_count,
                execution_contract: event.contract == event::BeginContract::Execution,
            }))
            .map_err(|_| Error::Internal)?;
        result.get()
    }

    pub(crate) fn block_build(&mut self, event: event::BlockBuild) -> Result<(), Error> {
        let child_result = Cell::new(Err(Error::Internal));
        let key_length = Cell::new(0);
        let value_length = Cell::new(0);
        let rope_dimension = Cell::new(0);
        let rope_frequency = Cell::new(0.0);
        let result = Cell::new(Err(Error::UnexpectedEvent));
        self.machine
            .process_event(Gemma4MachineEvents::Block(BlockRuntime {
                event,
                child_result: &child_result,
                key_length: &key_length,
                value_length: &value_length,
                rope_dimension: &rope_dimension,
                rope_frequency: &rope_frequency,
                result: &result,
            }))
            .map_err(|_| Error::Internal)?;
        result.get()
    }

    pub(crate) fn topology_build(&mut self, _event: event::TopologyBuild) -> Result<(), Error> {
        let child_result = Cell::new(Err(Error::Internal));
        let result = Cell::new(Err(Error::UnexpectedEvent));
        self.machine
            .process_event(Gemma4MachineEvents::Topology(TopologyRuntime {
                child_result: &child_result,
                result: &result,
            }))
            .map_err(|_| Error::Internal)?;
        result.get()
    }

    pub(crate) fn plan_build(&mut self, _event: event::PlanBuild) -> Result<(), Error> {
        let child_result = Cell::new(Err(Error::Internal));
        let result = Cell::new(Err(Error::UnexpectedEvent));
        self.machine
            .process_event(Gemma4MachineEvents::Plan(PlanRuntime {
                child_result: &child_result,
                result: &result,
            }))
            .map_err(|_| Error::Internal)?;
        result.get()
    }

    pub(crate) fn block_validation(&mut self, event: event::BlockValidation) -> Result<(), Error> {
        let child_result = Cell::new(Err(Error::Internal));
        let descriptor = Cell::new(Err(Error::Internal));
        let result = Cell::new(Err(Error::UnexpectedEvent));
        self.machine
            .process_event(Gemma4MachineEvents::Validate(ValidateRuntime {
                event,
                child_result: &child_result,
                descriptor: &descriptor,
                result: &result,
            }))
            .map_err(|_| Error::Internal)?;
        result.get()
    }

    pub(crate) fn block_audit(&mut self, event: event::BlockAudit) -> Result<(), Error> {
        let child_result = Cell::new(Err(Error::Internal));
        let result = Cell::new(Err(Error::UnexpectedEvent));
        self.machine
            .process_event(Gemma4MachineEvents::Audit(AuditRuntime {
                event,
                child_result: &child_result,
                result: &result,
            }))
            .map_err(|_| Error::Internal)?;
        result.get()
    }

    pub(crate) fn stage_audit(&mut self, event: event::StageAudit) -> Result<(), Error> {
        let child_result = Cell::new(Err(Error::Internal));
        let result = Cell::new(Err(Error::UnexpectedEvent));
        self.machine
            .process_event(Gemma4MachineEvents::Stage(StageRuntime {
                event,
                child_result: &child_result,
                result: &result,
            }))
            .map_err(|_| Error::Internal)?;
        result.get()
    }

    pub(crate) fn contract_visit(
        &mut self,
        _event: event::ContractVisit,
    ) -> Result<ContractDescriptor, Error> {
        let child_result = Cell::new(Err(Error::Internal));
        let result = Cell::new(Err(Error::UnexpectedEvent));
        self.machine
            .process_event(Gemma4MachineEvents::Visit(VisitRuntime {
                child_result: &child_result,
                result: &result,
            }))
            .map_err(|_| Error::Internal)?;
        result.get()
    }

    #[inline]
    pub(crate) fn block_visit(
        &mut self,
        event: event::BlockVisit,
    ) -> Result<generation::event::BlockDescriptor, Error> {
        let result = Cell::new(Err(Error::UnexpectedEvent));
        self.machine
            .process_event(Gemma4MachineEvents::BlockVisit(BlockVisitRuntime {
                event,
                result: &result,
            }))
            .map_err(|_| Error::Internal)?;
        result.get()
    }

    pub(crate) fn contract_reset(&mut self, _event: event::ContractReset) -> Result<(), Error> {
        let child_result = Cell::new(Err(Error::Internal));
        let result = Cell::new(Err(Error::UnexpectedEvent));
        self.machine
            .process_event(Gemma4MachineEvents::Reset(ResetRuntime {
                child_result: &child_result,
                result: &result,
            }))
            .map_err(|_| Error::Internal)?;
        result.get()
    }

    pub(crate) fn storage_release(
        &mut self,
        _event: event::StorageRelease,
    ) -> Result<event::Storage, Error> {
        let child_result = RefCell::new(Err(Error::Internal));
        let result = RefCell::new(Err(Error::UnexpectedEvent));
        self.machine
            .process_event(Gemma4MachineEvents::Release(ReleaseRuntime {
                child_result: &child_result,
                result: &result,
            }))
            .map_err(|_| Error::Internal)?;
        result.into_inner()
    }
}

impl fmt::Debug for Gemma4 {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.debug_struct("Gemma4").finish_non_exhaustive()
    }
}
