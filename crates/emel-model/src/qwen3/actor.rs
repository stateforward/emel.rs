//! Public Qwen3 actor and local generated-machine dispatch wrapper.

use core::cell::{Cell, RefCell};
use core::fmt;

use emel_kernels::capability;

use crate::attention_family::context::{self, Context, Policy};
use crate::catalog::Catalog;
use crate::generation::{self, AttentionQkNormRoute, ContractDescriptor};

pub(super) use crate::attention_family::context::{
    AuditRuntime, BeginRuntime, BlockRuntime, BlockVisitRuntime, PlanRuntime, ReleaseRuntime,
    ResetRuntime, StageRuntime, TopologyRuntime, UnexpectedRuntime, ValidateRuntime, VisitRuntime,
};

use super::event;
use super::sm::{Qwen3MachineEvents, Qwen3MachineStateMachine, Qwen3MachineStateMachineContext};
use super::{
    ARCHITECTURE_NAME, BLOCK_TENSOR_COUNT, GLOBAL_TENSOR_COUNT, OUTPUT_NORM_NAME,
    TOKEN_EMBEDDING_NAME,
};

type Error = generation::event::Error;
type ChildResult = Result<(), Error>;

pub(super) struct Qwen3Policy;
impl Policy for Qwen3Policy {
    const ARCHITECTURE_NAME: &'static [u8] = ARCHITECTURE_NAME;
    const TOKEN_EMBEDDING_NAME: &'static [u8] = TOKEN_EMBEDDING_NAME;
    const OUTPUT_NORM_NAME: &'static [u8] = OUTPUT_NORM_NAME;
    const GLOBAL_TENSOR_COUNT: u32 = GLOBAL_TENSOR_COUNT;
    const BLOCK_TENSOR_COUNT: u32 = BLOCK_TENSOR_COUNT;
    const QK_NORM_ROUTE: AttentionQkNormRoute = AttentionQkNormRoute::HeadwiseRms;
    const TIED_OUTPUT: bool = true;
}

crate::attention_family::impl_machine_context!(Qwen3MachineStateMachineContext, Qwen3Policy);

/// Single-writer, run-to-completion Qwen3 family actor.
pub struct Qwen3 {
    machine: Qwen3MachineStateMachine<Context<Qwen3Policy>>,
}

impl Qwen3 {
    /// Constructs a Qwen3 actor from public child actors and caller-preallocated storage.
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
            machine: Qwen3MachineStateMachine::new(Context::new(catalog, capability, storage)?),
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
            .process_event(Qwen3MachineEvents::Begin(BeginRuntime {
                event: crate::attention_family::event::ContractBegin::new(
                    event.architecture,
                    event.model,
                    &parameters,
                ),
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
        let key_length = Cell::new(0);
        let value_length = Cell::new(0);
        let rope_dimension = Cell::new(0);
        let rope_frequency = Cell::new(0.0);
        let result = Cell::new(Err(Error::UnexpectedEvent));
        self.machine
            .process_event(Qwen3MachineEvents::Block(BlockRuntime {
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
            .process_event(Qwen3MachineEvents::Topology(TopologyRuntime {
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
            .process_event(Qwen3MachineEvents::Plan(PlanRuntime {
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
            .process_event(Qwen3MachineEvents::Validate(ValidateRuntime {
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
            .process_event(Qwen3MachineEvents::Audit(AuditRuntime {
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
            .process_event(Qwen3MachineEvents::Stage(StageRuntime {
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
            .process_event(Qwen3MachineEvents::Visit(VisitRuntime {
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
            .process_event(Qwen3MachineEvents::BlockVisit(BlockVisitRuntime {
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
            .process_event(Qwen3MachineEvents::Reset(ResetRuntime {
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
            .process_event(Qwen3MachineEvents::Release(ReleaseRuntime {
                child_result: &child_result,
                result: &result,
            }))
            .map_err(|_| Error::Internal)?;
        result.into_inner()
    }
}

impl fmt::Debug for Qwen3 {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.debug_struct("Qwen3").finish_non_exhaustive()
    }
}
