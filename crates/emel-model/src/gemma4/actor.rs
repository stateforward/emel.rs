//! Public Gemma4 actor and private same-RTC dispatch runtimes.

use core::cell::{Cell, RefCell};
use core::fmt;

use crate::generation::quantized_path as capability;

use crate::catalog::Catalog;
use crate::generation::{
    self, AttentionQkNormRoute, AttentionVNormRoute, AttentionValueRoute, AttentionWindowRoute,
    ContractDescriptor, LayerExecution, ResidualRoute,
};

use super::event::{self, Parameters};
use super::sm::{Gemma4MachineEvents, Gemma4MachineStateMachine, Gemma4MachineStateMachineContext};
use super::{
    ARCHITECTURE_NAME, DEDICATED_KV_BLOCK_TENSOR_COUNT, GLOBAL_TENSOR_COUNT, OUTPUT_NORM_NAME,
    SHARED_KV_BLOCK_TENSOR_COUNT, TOKEN_EMBEDDING_NAME,
};

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

/// Single-writer, run-to-completion Gemma4 family actor.
pub struct Gemma4 {
    machine: Gemma4MachineStateMachine<Context>,
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
            machine: Gemma4MachineStateMachine::new(Context {
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
            .process_event(Gemma4MachineEvents::Begin(BeginRuntime {
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
        let result = Cell::new(Err(Error::UnexpectedEvent));
        self.machine
            .process_event(Gemma4MachineEvents::Block(BlockRuntime {
                event,
                child_result: &child_result,
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

impl Context {
    fn layer(&self, index: i32) -> LayerExecution {
        let shared_start = self
            .parameters
            .block_count
            .saturating_sub(self.parameters.attention_shared_kv_layers);
        let shared = index >= shared_start;
        let sliding = usize::try_from(index).ok().is_some_and(|index| {
            index < self.parameters.sliding_window_pattern_count as usize
                && self
                    .parameters
                    .sliding_window_pattern
                    .get(index)
                    .is_some_and(|flag| *flag != 0)
        });
        LayerExecution::new(
            ResidualRoute::Attention,
            AttentionQkNormRoute::HeadwiseRms,
            if shared {
                AttentionValueRoute::SharedKeyValue
            } else {
                AttentionValueRoute::DedicatedValue
            },
            if shared {
                AttentionVNormRoute::Rms
            } else {
                AttentionVNormRoute::None
            },
            if sliding {
                AttentionWindowRoute::SlidingWindow
            } else {
                AttentionWindowRoute::FullContext
            },
            if sliding {
                self.parameters.attention_key_length_swa
            } else {
                self.parameters.attention_key_length
            },
            if sliding {
                self.parameters.attention_value_length_swa
            } else {
                self.parameters.attention_value_length
            },
            if sliding {
                self.parameters.rope_dimension_count_swa
            } else {
                self.parameters.rope_dimension_count
            },
            if sliding {
                self.parameters.rope_freq_base_swa
            } else {
                self.parameters.rope_freq_base
            },
        )
    }

    fn topology_values(&self) -> Option<(u32, u64)> {
        let blocks = u32::try_from(self.parameters.block_count).ok()?;
        let embedding = u64::try_from(self.parameters.embedding_length).ok()?;
        let shared_start =
            blocks.saturating_sub(u32::try_from(self.parameters.attention_shared_kv_layers).ok()?);
        let tensors = (0..blocks).try_fold(GLOBAL_TENSOR_COUNT, |total, index| {
            let per_block = if index >= shared_start {
                SHARED_KV_BLOCK_TENSOR_COUNT
            } else {
                DEDICATED_KV_BLOCK_TENSOR_COUNT
            };
            total.checked_add(per_block)
        })?;
        let workspace = u64::from(tensors).checked_mul(embedding)?.checked_mul(4)?;
        Some((tensors, workspace))
    }
}

const fn child_ok(result: &Cell<ChildResult>) -> bool {
    result.get().is_ok()
}

impl Gemma4MachineStateMachineContext for Context {
    fn guard_begin_valid(&self, event: &BeginRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.architecture == ARCHITECTURE_NAME
            && event.event.parameters.block_count > 0
            && event.event.parameters.embedding_length > 0
            && event.event.parameters.context_length > 0)
    }
    fn guard_begin_architecture_invalid(&self, event: &BeginRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.architecture != ARCHITECTURE_NAME)
    }
    fn guard_begin_parameters_invalid(&self, event: &BeginRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.architecture == ARCHITECTURE_NAME && !self.guard_begin_valid(event)?)
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
        self.parameters = event.event.parameters;
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

    fn guard_block_valid(&self, event: &BlockRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.index >= 0 && event.event.index < self.parameters.block_count)
    }
    fn guard_block_invalid(&self, event: &BlockRuntime<'_>) -> Result<bool, ()> {
        Ok(!self.guard_block_valid(event)?)
    }
    fn effect_block_child(&mut self, event: BlockRuntime<'_>) -> Result<(), ()> {
        event.child_result.set(
            self.builder
                .process_event(generation::event::AttentionBlock::new(
                    event.event.index,
                    self.layer(event.event.index),
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
    fn effect_child_ok_block(&mut self, event: BlockRuntime<'_>) -> Result<(), ()> {
        event.result.set(Ok(()));
        Ok(())
    }
    fn effect_child_error_block(&mut self, event: BlockRuntime<'_>) -> Result<(), ()> {
        event.result.set(event.child_result.get());
        Ok(())
    }

    fn guard_topology_valid(&self, _event: &TopologyRuntime<'_>) -> Result<bool, ()> {
        Ok(self.topology_values().is_some())
    }
    fn guard_topology_invalid(&self, event: &TopologyRuntime<'_>) -> Result<bool, ()> {
        Ok(!self.guard_topology_valid(event)?)
    }
    fn effect_topology_child(&mut self, event: TopologyRuntime<'_>) -> Result<(), ()> {
        let (tensor_count, workspace) = self.topology_values().expect("guarded topology");
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
