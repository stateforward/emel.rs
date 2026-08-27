//! Public Lfm2 actor and private same-RTC dispatch runtimes.

use core::cell::{Cell, RefCell};
use core::fmt;

use crate::generation::quantized_path as capability;

use crate::catalog::Catalog;
use crate::generation::{
    self, AttentionQkNormRoute, AttentionVNormRoute, AttentionValueRoute, AttentionWindowRoute,
    ContractDescriptor, LayerExecution, ResidualRoute,
};

use super::event::{self, Parameters};
use super::sm::{Lfm2MachineEvents, Lfm2MachineStateMachine, Lfm2MachineStateMachineContext};
use super::{
    ARCHITECTURE_NAME, ATTENTION_BLOCK_TENSOR_COUNT, GLOBAL_TENSOR_COUNT, OUTPUT_NORM_NAME,
    SHORTCONV_BLOCK_TENSOR_COUNT, TOKEN_EMBEDDING_NAME,
};

const FALLBACK_ATTENTION_LAYERS: [i32; 6] = [2, 5, 8, 10, 12, 14];

type Error = generation::event::Error;
type ChildResult = Result<(), Error>;

#[derive(Clone, Copy)]
pub(super) struct BeginRuntime<'a> {
    pub(super) event: event::ContractBegin<'a>,
    begin_result: &'a Cell<ChildResult>,
    global_result: &'a Cell<ChildResult>,
    reset_result: &'a Cell<ChildResult>,
    result: &'a Cell<ChildResult>,
}

#[derive(Clone, Copy)]
pub(super) struct BlockRuntime<'a> {
    event: event::BlockBuild,
    child_result: &'a Cell<ChildResult>,
    exclusion_result: &'a Cell<ChildResult>,
    reset_result: &'a Cell<ChildResult>,
    result: &'a Cell<ChildResult>,
}

#[derive(Clone, Copy)]
pub(super) struct TopologyRuntime<'a> {
    child_result: &'a Cell<ChildResult>,
    result: &'a Cell<ChildResult>,
}

#[derive(Clone, Copy)]
pub(super) struct PlanRuntime<'a> {
    child_result: &'a Cell<ChildResult>,
    result: &'a Cell<ChildResult>,
}

#[derive(Clone, Copy)]
pub(super) struct ValidateRuntime<'a> {
    event: event::BlockValidation,
    child_result: &'a Cell<ChildResult>,
    descriptor: &'a Cell<Result<generation::event::BlockDescriptor, Error>>,
    result: &'a Cell<ChildResult>,
}

#[derive(Clone, Copy)]
pub(super) struct AuditRuntime<'a> {
    event: event::BlockAudit,
    child_result: &'a Cell<ChildResult>,
    result: &'a Cell<ChildResult>,
}

#[derive(Clone, Copy)]
pub(super) struct StageRuntime<'a> {
    event: event::StageAudit,
    child_result: &'a Cell<ChildResult>,
    result: &'a Cell<ChildResult>,
}

#[derive(Clone, Copy)]
pub(super) struct VisitRuntime<'a> {
    child_result: &'a Cell<Result<ContractDescriptor, Error>>,
    result: &'a Cell<Result<ContractDescriptor, Error>>,
}

#[derive(Clone, Copy)]
pub(super) struct BlockVisitRuntime<'a> {
    event: event::BlockVisit,
    result: &'a Cell<Result<generation::event::BlockDescriptor, Error>>,
}

#[derive(Clone, Copy)]
pub(super) struct ResetRuntime<'a> {
    child_result: &'a Cell<ChildResult>,
    result: &'a Cell<ChildResult>,
}

#[derive(Clone, Copy)]
pub(super) struct ReleaseRuntime<'a> {
    child_result: &'a RefCell<Result<generation::event::Storage, Error>>,
    result: &'a RefCell<Result<event::Storage, Error>>,
}

#[derive(Clone, Copy, Debug)]
pub(super) struct UnexpectedRuntime;

pub(super) struct Context {
    builder: generation::Builder,
    parameters: Parameters,
    blocks: Vec<Option<generation::event::BlockDescriptor>>,
}

/// Single-writer, run-to-completion Lfm2 family actor.
pub struct Lfm2 {
    machine: Lfm2MachineStateMachine<Context>,
}

impl Lfm2 {
    /// Constructs a Lfm2 actor from public child actors and caller-preallocated storage.
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
        let event::Storage { common, blocks } = storage;
        let mut builder = generation::Builder::new(catalog, capability);
        if let Err(error) = builder.process_event(generation::event::StorageBind::new(common)) {
            return Err(event::StorageBindError::new(
                error.error(),
                event::Storage {
                    common: error.into_storage(),
                    blocks,
                },
            ));
        }
        Ok(Self {
            machine: Lfm2MachineStateMachine::new(Context {
                builder,
                parameters: Parameters::default(),
                blocks,
            }),
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
        self.machine
            .process_event(Lfm2MachineEvents::Begin(BeginRuntime {
                event,
                begin_result: &begin_result,
                global_result: &global_result,
                reset_result: &reset_result,
                result: &result,
            }))
            .map_err(|_| Error::Internal)?;
        result.get()
    }

    pub(crate) fn block_build(&mut self, event: event::BlockBuild) -> Result<(), Error> {
        let child_result = Cell::new(Err(Error::Internal));
        let exclusion_result = Cell::new(Err(Error::Internal));
        let reset_result = Cell::new(Err(Error::Internal));
        let result = Cell::new(Err(Error::UnexpectedEvent));
        self.machine
            .process_event(Lfm2MachineEvents::Block(BlockRuntime {
                event,
                child_result: &child_result,
                exclusion_result: &exclusion_result,
                reset_result: &reset_result,
                result: &result,
            }))
            .map_err(|_| Error::Internal)?;
        result.get()
    }

    pub(crate) fn topology_build(&mut self, _event: event::TopologyBuild) -> Result<(), Error> {
        let child_result = Cell::new(Err(Error::Internal));
        let result = Cell::new(Err(Error::UnexpectedEvent));
        self.machine
            .process_event(Lfm2MachineEvents::Topology(TopologyRuntime {
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
            .process_event(Lfm2MachineEvents::Plan(PlanRuntime {
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
            .process_event(Lfm2MachineEvents::Validate(ValidateRuntime {
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
            .process_event(Lfm2MachineEvents::Audit(AuditRuntime {
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
            .process_event(Lfm2MachineEvents::Stage(StageRuntime {
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
            .process_event(Lfm2MachineEvents::Visit(VisitRuntime {
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
            .process_event(Lfm2MachineEvents::BlockVisit(BlockVisitRuntime {
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
            .process_event(Lfm2MachineEvents::Reset(ResetRuntime {
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
            .process_event(Lfm2MachineEvents::Release(ReleaseRuntime {
                child_result: &child_result,
                result: &result,
            }))
            .map_err(|_| Error::Internal)?;
        result.into_inner()
    }
}

impl fmt::Debug for Lfm2 {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.debug_struct("Lfm2").finish_non_exhaustive()
    }
}

impl Context {
    const fn block_index_valid(&self, event: &BlockRuntime<'_>) -> bool {
        event.event.index >= 0 && event.event.index < self.parameters.block_count
    }

    const fn attention_layer(&self) -> LayerExecution {
        LayerExecution::new(
            ResidualRoute::Attention,
            AttentionQkNormRoute::HeadwiseRms,
            AttentionValueRoute::DedicatedValue,
            AttentionVNormRoute::None,
            AttentionWindowRoute::FullContext,
            self.parameters.attention_key_length,
            self.parameters.attention_value_length,
            self.parameters.rope_dimension_count,
            self.parameters.rope_freq_base,
        )
    }

    const fn shortconv_layer(&self) -> LayerExecution {
        LayerExecution::new(
            ResidualRoute::Shortconv,
            AttentionQkNormRoute::None,
            AttentionValueRoute::DedicatedValue,
            AttentionVNormRoute::None,
            AttentionWindowRoute::FullContext,
            self.parameters.attention_key_length,
            self.parameters.attention_value_length,
            self.parameters.rope_dimension_count,
            self.parameters.rope_freq_base,
        )
    }

    fn topology_values_pattern(&self) -> Option<(u32, u64)> {
        let blocks = usize::try_from(self.parameters.block_count).ok()?;
        let pattern_count = usize::try_from(self.parameters.attention_layer_pattern_count).ok()?;
        let count = core::cmp::min(blocks, pattern_count);
        let mut attention_blocks = 0u32;
        for flag in &self.parameters.attention_layer_pattern_flags[..count] {
            attention_blocks = attention_blocks.checked_add(u32::from(*flag != 0))?;
        }
        self.topology_values_for_attention_count(attention_blocks)
    }

    fn topology_values_fallback(&self) -> Option<(u32, u64)> {
        let blocks = self.parameters.block_count;
        let mut attention_blocks = 0u32;
        for index in FALLBACK_ATTENTION_LAYERS {
            attention_blocks = attention_blocks.checked_add(u32::from(index < blocks))?;
        }
        self.topology_values_for_attention_count(attention_blocks)
    }

    fn topology_values_for_attention_count(&self, attention_blocks: u32) -> Option<(u32, u64)> {
        let blocks = u32::try_from(self.parameters.block_count).ok()?;
        let shortconv_blocks = blocks.checked_sub(attention_blocks)?;
        let embedding = u64::try_from(self.parameters.embedding_length).ok()?;
        let tensors = attention_blocks
            .checked_mul(ATTENTION_BLOCK_TENSOR_COUNT)?
            .checked_add(shortconv_blocks.checked_mul(SHORTCONV_BLOCK_TENSOR_COUNT)?)?
            .checked_add(GLOBAL_TENSOR_COUNT)?;
        let workspace = u64::from(tensors).checked_mul(embedding)?.checked_mul(4)?;
        Some((tensors, workspace))
    }
}

const fn child_ok(result: &Cell<ChildResult>) -> bool {
    result.get().is_ok()
}

impl Lfm2MachineStateMachineContext for Context {
    fn guard_begin_1_2b(&self, event: &BeginRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.contract == event::BeginContract::Execution
            && event.event.architecture == ARCHITECTURE_NAME
            && event.event.layer_count == event.event.parameters.block_count
            && event.event.parameters.variant() == Some(event::Variant::OnePointTwoB)
            && source_fixed_metadata_valid(event.event.parameters))
    }
    fn guard_begin_230m(&self, event: &BeginRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.contract == event::BeginContract::Execution
            && event.event.architecture == ARCHITECTURE_NAME
            && event.event.layer_count == event.event.parameters.block_count
            && event.event.parameters.variant() == Some(event::Variant::TwoHundredThirtyM)
            && source_fixed_metadata_valid(event.event.parameters))
    }
    fn guard_begin_validation(&self, event: &BeginRuntime<'_>) -> Result<bool, ()> {
        let parameters = event.event.parameters;
        Ok(event.event.contract == event::BeginContract::Validation
            && event.event.architecture == ARCHITECTURE_NAME
            && event.event.layer_count > 0
            && event.event.layer_count == parameters.block_count
            && parameters.context_length > 0
            && parameters.embedding_length > 0
            && parameters.attention_head_count > 0
            && parameters.attention_head_count_kv > 0
            && parameters.vocab_size > 0
            && parameters.shortconv_l_cache > 0
            && parameters.rope_freq_base > 0.0)
    }
    fn guard_begin_architecture_invalid(&self, event: &BeginRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.architecture != ARCHITECTURE_NAME)
    }
    fn guard_begin_parameters_invalid(&self, event: &BeginRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.architecture == ARCHITECTURE_NAME
            && !self.guard_begin_1_2b(event)?
            && !self.guard_begin_230m(event)?
            && !self.guard_begin_validation(event)?)
    }
    fn effect_begin_child(&mut self, event: BeginRuntime<'_>) -> Result<(), ()> {
        event.begin_result.set(
            self.builder
                .process_event(generation::event::ContractBegin::new(
                    event.event.model,
                    event.event.parameters.block_count,
                    event.event.parameters.context_length,
                )),
        );
        Ok(())
    }
    fn guard_begin_child_ok(&self, event: &BeginRuntime<'_>) -> Result<bool, ()> {
        Ok(child_ok(event.begin_result))
    }
    fn guard_begin_child_error(&self, event: &BeginRuntime<'_>) -> Result<bool, ()> {
        Ok(!child_ok(event.begin_result))
    }
    fn effect_global_child(&mut self, event: BeginRuntime<'_>) -> Result<(), ()> {
        event.global_result.set(self.builder.process_event(
            generation::event::GlobalBindings::new(TOKEN_EMBEDDING_NAME, OUTPUT_NORM_NAME, true),
        ));
        Ok(())
    }
    fn effect_begin_child_error(&mut self, event: BeginRuntime<'_>) -> Result<(), ()> {
        event.result.set(event.begin_result.get());
        Ok(())
    }
    fn guard_global_child_ok(&self, event: &BeginRuntime<'_>) -> Result<bool, ()> {
        Ok(child_ok(event.global_result))
    }
    fn guard_global_child_error(&self, event: &BeginRuntime<'_>) -> Result<bool, ()> {
        Ok(!child_ok(event.global_result))
    }
    fn effect_begin_complete(&mut self, event: BeginRuntime<'_>) -> Result<(), ()> {
        self.parameters = *event.event.parameters;
        event.result.set(Ok(()));
        Ok(())
    }
    fn effect_begin_reset(&mut self, event: BeginRuntime<'_>) -> Result<(), ()> {
        event.reset_result.set(
            self.builder
                .process_event(generation::event::ContractReset::new()),
        );
        Ok(())
    }
    fn guard_begin_reset_ok(&self, event: &BeginRuntime<'_>) -> Result<bool, ()> {
        Ok(child_ok(event.reset_result))
    }
    fn guard_begin_reset_error(&self, event: &BeginRuntime<'_>) -> Result<bool, ()> {
        Ok(!child_ok(event.reset_result))
    }
    fn effect_global_child_error(&mut self, event: BeginRuntime<'_>) -> Result<(), ()> {
        event.result.set(event.global_result.get());
        Ok(())
    }
    fn effect_begin_reset_error(&mut self, event: BeginRuntime<'_>) -> Result<(), ()> {
        event.result.set(Err(Error::Internal));
        Ok(())
    }
    fn effect_begin_invalid_request(&mut self, event: BeginRuntime<'_>) -> Result<(), ()> {
        event.result.set(Err(Error::InvalidRequest));
        Ok(())
    }
    fn effect_begin_model_invalid(&mut self, event: BeginRuntime<'_>) -> Result<(), ()> {
        event.result.set(Err(Error::ModelInvalid));
        Ok(())
    }
    fn effect_busy_begin(&mut self, event: BeginRuntime<'_>) -> Result<(), ()> {
        event.result.set(Err(Error::Busy));
        Ok(())
    }

    fn guard_block_pattern_attention(&self, event: &BlockRuntime<'_>) -> Result<bool, ()> {
        let index = usize::try_from(event.event.index).ok();
        Ok(self.block_index_valid(event)
            && self.parameters.attention_layer_pattern_count > 0
            && index.is_some_and(|value| {
                u32::try_from(value).is_ok_and(|value| {
                    value < self.parameters.attention_layer_pattern_count
                        && self.parameters.attention_layer_pattern_flags
                            [usize::try_from(value).expect("bounded pattern index")]
                            != 0
                })
            }))
    }
    fn guard_block_pattern_shortconv(&self, event: &BlockRuntime<'_>) -> Result<bool, ()> {
        Ok(self.block_index_valid(event)
            && self.parameters.attention_layer_pattern_count > 0
            && !self.guard_block_pattern_attention(event)?)
    }
    fn guard_block_fallback_attention(&self, event: &BlockRuntime<'_>) -> Result<bool, ()> {
        Ok(self.block_index_valid(event)
            && self.parameters.attention_layer_pattern_count == 0
            && FALLBACK_ATTENTION_LAYERS.contains(&event.event.index))
    }
    fn guard_block_fallback_shortconv(&self, event: &BlockRuntime<'_>) -> Result<bool, ()> {
        Ok(self.block_index_valid(event)
            && self.parameters.attention_layer_pattern_count == 0
            && !FALLBACK_ATTENTION_LAYERS.contains(&event.event.index))
    }
    fn guard_block_invalid(&self, event: &BlockRuntime<'_>) -> Result<bool, ()> {
        Ok(!self.block_index_valid(event))
    }
    fn effect_attention_child(&mut self, event: BlockRuntime<'_>) -> Result<(), ()> {
        event.child_result.set(
            self.builder
                .process_event(generation::event::AttentionBlock::new(
                    event.event.index,
                    self.attention_layer(),
                )),
        );
        Ok(())
    }
    fn effect_shortconv_child(&mut self, event: BlockRuntime<'_>) -> Result<(), ()> {
        event.child_result.set(
            self.builder
                .process_event(generation::event::ShortconvBlock::new(
                    event.event.index,
                    self.shortconv_layer(),
                )),
        );
        Ok(())
    }
    fn effect_block_invalid(&mut self, event: BlockRuntime<'_>) -> Result<(), ()> {
        event.result.set(Err(Error::InvalidRequest));
        Ok(())
    }
    fn guard_block_child_ok(&self, event: &BlockRuntime<'_>) -> Result<bool, ()> {
        Ok(child_ok(event.child_result))
    }
    fn guard_block_child_error(&self, event: &BlockRuntime<'_>) -> Result<bool, ()> {
        Ok(!child_ok(event.child_result))
    }
    fn effect_reject_shortconv_child(&mut self, event: BlockRuntime<'_>) -> Result<(), ()> {
        event.exclusion_result.set(self.builder.process_event(
            generation::event::RejectBlockTensors::new(
                event.event.index,
                generation::event::BlockFamily::Shortconv,
            ),
        ));
        Ok(())
    }
    fn effect_reject_attention_child(&mut self, event: BlockRuntime<'_>) -> Result<(), ()> {
        event.exclusion_result.set(self.builder.process_event(
            generation::event::RejectBlockTensors::new(
                event.event.index,
                generation::event::BlockFamily::Attention,
            ),
        ));
        Ok(())
    }
    fn guard_exclusion_child_ok(&self, event: &BlockRuntime<'_>) -> Result<bool, ()> {
        Ok(child_ok(event.exclusion_result))
    }
    fn guard_exclusion_child_error(&self, event: &BlockRuntime<'_>) -> Result<bool, ()> {
        Ok(!child_ok(event.exclusion_result))
    }
    fn effect_child_ok_block(&mut self, event: BlockRuntime<'_>) -> Result<(), ()> {
        event.result.set(Ok(()));
        Ok(())
    }
    fn effect_child_error_block(&mut self, event: BlockRuntime<'_>) -> Result<(), ()> {
        event.result.set(event.child_result.get());
        Ok(())
    }
    fn effect_block_reset_child(&mut self, event: BlockRuntime<'_>) -> Result<(), ()> {
        event.reset_result.set(
            self.builder
                .process_event(generation::event::ContractReset::new()),
        );
        Ok(())
    }
    fn guard_block_reset_ok(&self, event: &BlockRuntime<'_>) -> Result<bool, ()> {
        Ok(child_ok(event.reset_result))
    }
    fn guard_block_reset_error(&self, event: &BlockRuntime<'_>) -> Result<bool, ()> {
        Ok(!child_ok(event.reset_result))
    }
    fn effect_block_reset_error(&mut self, event: BlockRuntime<'_>) -> Result<(), ()> {
        self.parameters = Parameters::default();
        self.blocks.fill(None);
        event.result.set(Err(Error::Internal));
        Ok(())
    }
    fn effect_block_reset_complete(&mut self, event: BlockRuntime<'_>) -> Result<(), ()> {
        self.parameters = Parameters::default();
        self.blocks.fill(None);
        event.result.set(event.exclusion_result.get());
        Ok(())
    }

    fn guard_topology_pattern_valid(&self, _event: &TopologyRuntime<'_>) -> Result<bool, ()> {
        Ok(self.parameters.attention_layer_pattern_count > 0
            && self.topology_values_pattern().is_some())
    }
    fn guard_topology_fallback_valid(&self, _event: &TopologyRuntime<'_>) -> Result<bool, ()> {
        Ok(self.parameters.attention_layer_pattern_count == 0
            && self.topology_values_fallback().is_some())
    }
    fn guard_topology_invalid(&self, event: &TopologyRuntime<'_>) -> Result<bool, ()> {
        Ok(!self.guard_topology_pattern_valid(event)?
            && !self.guard_topology_fallback_valid(event)?)
    }
    fn effect_topology_pattern_child(&mut self, event: TopologyRuntime<'_>) -> Result<(), ()> {
        let (tensor_count, workspace) = self
            .topology_values_pattern()
            .expect("guarded pattern topology");
        event
            .child_result
            .set(self.builder.process_event(generation::event::Topology::new(
                tensor_count,
                tensor_count,
                4,
                workspace,
            )));
        Ok(())
    }
    fn effect_topology_fallback_child(&mut self, event: TopologyRuntime<'_>) -> Result<(), ()> {
        let (tensor_count, workspace) = self
            .topology_values_fallback()
            .expect("guarded fallback topology");
        event
            .child_result
            .set(self.builder.process_event(generation::event::Topology::new(
                tensor_count,
                tensor_count,
                4,
                workspace,
            )));
        Ok(())
    }
    fn effect_topology_invalid(&mut self, event: TopologyRuntime<'_>) -> Result<(), ()> {
        event.result.set(Err(Error::InvalidRequest));
        Ok(())
    }
    fn guard_topology_child_ok(&self, event: &TopologyRuntime<'_>) -> Result<bool, ()> {
        Ok(child_ok(event.child_result))
    }
    fn guard_topology_child_error(&self, event: &TopologyRuntime<'_>) -> Result<bool, ()> {
        Ok(!child_ok(event.child_result))
    }
    fn effect_child_ok_topology(&mut self, event: TopologyRuntime<'_>) -> Result<(), ()> {
        event.result.set(Ok(()));
        Ok(())
    }
    fn effect_child_error_topology(&mut self, event: TopologyRuntime<'_>) -> Result<(), ()> {
        event.result.set(event.child_result.get());
        Ok(())
    }

    fn effect_plan_child(&mut self, event: PlanRuntime<'_>) -> Result<(), ()> {
        event
            .child_result
            .set(self.builder.process_event(generation::event::Plan::new()));
        Ok(())
    }
    fn guard_plan_child_ok(&self, event: &PlanRuntime<'_>) -> Result<bool, ()> {
        Ok(child_ok(event.child_result))
    }
    fn guard_plan_child_error(&self, event: &PlanRuntime<'_>) -> Result<bool, ()> {
        Ok(!child_ok(event.child_result))
    }
    fn effect_child_ok_plan(&mut self, event: PlanRuntime<'_>) -> Result<(), ()> {
        event.result.set(Ok(()));
        Ok(())
    }
    fn effect_child_error_plan(&mut self, event: PlanRuntime<'_>) -> Result<(), ()> {
        event.result.set(event.child_result.get());
        Ok(())
    }

    fn effect_validate_child(&mut self, event: ValidateRuntime<'_>) -> Result<(), ()> {
        event.child_result.set(
            self.builder
                .process_event(generation::event::BlockValidation::new(event.event.index)),
        );
        Ok(())
    }
    fn guard_validate_child_ok(&self, event: &ValidateRuntime<'_>) -> Result<bool, ()> {
        Ok(child_ok(event.child_result))
    }
    fn guard_validate_child_error(&self, event: &ValidateRuntime<'_>) -> Result<bool, ()> {
        Ok(!child_ok(event.child_result))
    }
    fn effect_validate_visit(&mut self, event: ValidateRuntime<'_>) -> Result<(), ()> {
        event.descriptor.set(
            self.builder
                .process_event(generation::event::BlockVisit::new(event.event.index)),
        );
        Ok(())
    }
    fn guard_validate_visit_ok(&self, event: &ValidateRuntime<'_>) -> Result<bool, ()> {
        Ok(event.descriptor.get().is_ok())
    }
    fn guard_validate_visit_error(&self, event: &ValidateRuntime<'_>) -> Result<bool, ()> {
        Ok(event.descriptor.get().is_err())
    }
    fn effect_cache_validated_block(&mut self, event: ValidateRuntime<'_>) -> Result<(), ()> {
        let index = usize::try_from(event.event.index).expect("validated index");
        self.blocks[index] = event.descriptor.get().ok();
        event.result.set(Ok(()));
        Ok(())
    }
    fn effect_validate_visit_error(&mut self, event: ValidateRuntime<'_>) -> Result<(), ()> {
        event.result.set(Err(Error::Internal));
        Ok(())
    }
    fn effect_child_error_validate(&mut self, event: ValidateRuntime<'_>) -> Result<(), ()> {
        event.result.set(event.child_result.get());
        Ok(())
    }

    fn effect_audit_child(&mut self, event: AuditRuntime<'_>) -> Result<(), ()> {
        event.child_result.set(
            self.builder
                .process_event(generation::event::BlockAudit::new(event.event.index)),
        );
        Ok(())
    }
    fn guard_audit_child_ok(&self, event: &AuditRuntime<'_>) -> Result<bool, ()> {
        Ok(child_ok(event.child_result))
    }
    fn guard_audit_child_error(&self, event: &AuditRuntime<'_>) -> Result<bool, ()> {
        Ok(!child_ok(event.child_result))
    }
    fn effect_child_ok_audit(&mut self, event: AuditRuntime<'_>) -> Result<(), ()> {
        event.result.set(Ok(()));
        Ok(())
    }
    fn effect_child_error_audit(&mut self, event: AuditRuntime<'_>) -> Result<(), ()> {
        event.result.set(event.child_result.get());
        Ok(())
    }

    fn effect_stage_child(&mut self, event: StageRuntime<'_>) -> Result<(), ()> {
        event.child_result.set(
            self.builder
                .process_event(generation::event::StageAudit::new(event.event.family)),
        );
        Ok(())
    }
    fn guard_stage_child_ok(&self, event: &StageRuntime<'_>) -> Result<bool, ()> {
        Ok(child_ok(event.child_result))
    }
    fn guard_stage_child_error(&self, event: &StageRuntime<'_>) -> Result<bool, ()> {
        Ok(!child_ok(event.child_result))
    }
    fn effect_child_ok_stage(&mut self, event: StageRuntime<'_>) -> Result<(), ()> {
        event.result.set(Ok(()));
        Ok(())
    }
    fn effect_child_error_stage(&mut self, event: StageRuntime<'_>) -> Result<(), ()> {
        event.result.set(event.child_result.get());
        Ok(())
    }

    fn effect_visit_child(&mut self, event: VisitRuntime<'_>) -> Result<(), ()> {
        event.child_result.set(
            self.builder
                .process_event(generation::event::ContractVisit::new()),
        );
        Ok(())
    }
    fn guard_visit_child_ok(&self, event: &VisitRuntime<'_>) -> Result<bool, ()> {
        Ok(event.child_result.get().is_ok())
    }
    fn guard_visit_child_error(&self, event: &VisitRuntime<'_>) -> Result<bool, ()> {
        Ok(event.child_result.get().is_err())
    }
    fn effect_visit_ok(&mut self, event: VisitRuntime<'_>) -> Result<(), ()> {
        event.result.set(event.child_result.get());
        Ok(())
    }
    fn effect_visit_error(&mut self, event: VisitRuntime<'_>) -> Result<(), ()> {
        event.result.set(event.child_result.get());
        Ok(())
    }

    fn guard_block_visit_valid(&self, event: &BlockVisitRuntime<'_>) -> Result<bool, ()> {
        Ok(usize::try_from(event.event.index)
            .ok()
            .and_then(|index| self.blocks.get(index))
            .is_some_and(Option::is_some))
    }
    fn guard_block_visit_invalid(&self, event: &BlockVisitRuntime<'_>) -> Result<bool, ()> {
        Ok(!self.guard_block_visit_valid(event)?)
    }
    fn effect_block_visit(&mut self, event: BlockVisitRuntime<'_>) -> Result<(), ()> {
        let index = usize::try_from(event.event.index).expect("guarded index");
        event
            .result
            .set(Ok(self.blocks[index].expect("validated block")));
        Ok(())
    }
    fn effect_invalid_block_visit(&mut self, event: BlockVisitRuntime<'_>) -> Result<(), ()> {
        event.result.set(Err(Error::InvalidRequest));
        Ok(())
    }

    fn effect_reset_child(&mut self, event: ResetRuntime<'_>) -> Result<(), ()> {
        event.child_result.set(
            self.builder
                .process_event(generation::event::ContractReset::new()),
        );
        Ok(())
    }
    fn guard_reset_child_ok(&self, event: &ResetRuntime<'_>) -> Result<bool, ()> {
        Ok(child_ok(event.child_result))
    }
    fn guard_reset_child_error(&self, event: &ResetRuntime<'_>) -> Result<bool, ()> {
        Ok(!child_ok(event.child_result))
    }
    fn effect_reset_ok(&mut self, event: ResetRuntime<'_>) -> Result<(), ()> {
        self.parameters = Parameters::default();
        self.blocks.fill(None);
        event.result.set(Ok(()));
        Ok(())
    }
    fn effect_reset_error(&mut self, event: ResetRuntime<'_>) -> Result<(), ()> {
        event.result.set(event.child_result.get());
        Ok(())
    }

    fn effect_release_child(&mut self, event: ReleaseRuntime<'_>) -> Result<(), ()> {
        *event.child_result.borrow_mut() = self
            .builder
            .process_event(generation::event::StorageRelease::new());
        Ok(())
    }
    fn guard_release_child_ok(&self, event: &ReleaseRuntime<'_>) -> Result<bool, ()> {
        Ok(event.child_result.borrow().is_ok())
    }
    fn guard_release_child_error(&self, event: &ReleaseRuntime<'_>) -> Result<bool, ()> {
        Ok(event.child_result.borrow().is_err())
    }
    fn effect_release_ok(&mut self, event: ReleaseRuntime<'_>) -> Result<(), ()> {
        let common =
            core::mem::replace(&mut *event.child_result.borrow_mut(), Err(Error::Internal))
                .expect("guarded release");
        *event.result.borrow_mut() = Ok(event::Storage {
            common,
            blocks: core::mem::take(&mut self.blocks),
        });
        Ok(())
    }
    fn effect_release_error(&mut self, event: ReleaseRuntime<'_>) -> Result<(), ()> {
        *event.result.borrow_mut() = Err(event
            .child_result
            .borrow()
            .as_ref()
            .expect_err("guarded release error")
            .to_owned());
        Ok(())
    }
    fn effect_busy_release(&mut self, event: ReleaseRuntime<'_>) -> Result<(), ()> {
        *event.result.borrow_mut() = Err(Error::Busy);
        Ok(())
    }

    fn guard_never(&self, _event: &UnexpectedRuntime) -> Result<bool, ()> {
        Ok(false)
    }
    fn effect_unexpected(&mut self) -> Result<(), ()> {
        Ok(())
    }
}

const fn source_fixed_metadata_valid(parameters: &Parameters) -> bool {
    parameters.context_length == 128_000
        && parameters.attention_head_count_kv == 8
        && parameters.vocab_size == 65_536
        && parameters.shortconv_l_cache == 3
        && parameters.rope_freq_base.to_bits() == 1_000_000.0f32.to_bits()
}
