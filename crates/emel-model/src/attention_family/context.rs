//! Shared compile-time attention-family context algorithms and dispatch payloads.

// These callbacks intentionally retain the exact generated SML trait signatures;
// the family-local adapters forward them without adding control logic.
#![allow(
    clippy::missing_const_for_fn,
    clippy::needless_pass_by_ref_mut,
    clippy::trivially_copy_pass_by_ref,
    clippy::unnecessary_wraps,
    clippy::unused_self
)]

use core::cell::{Cell, RefCell};
use core::marker::PhantomData;

use emel_kernels::capability;

use crate::catalog::Catalog;
use crate::generation::{
    self, AttentionQkNormRoute, AttentionVNormRoute, AttentionValueRoute, AttentionWindowRoute,
    ContractDescriptor, LayerExecution, ResidualRoute,
};

use super::event::{self, ContractBegin};

pub type Error = generation::event::Error;
pub type ChildResult = Result<(), Error>;

/// Family-owned facts consumed by the shared compile-time mechanics.
#[derive(Clone, Copy, Debug, Default)]
pub struct Parameters {
    pub context_length: i32,
    pub embedding_length: i32,
    pub attention_key_length: i32,
    pub attention_value_length: i32,
    pub rope_dimension_count: i32,
    pub block_count: i32,
    pub rope_freq_base: f32,
}

/// Compile-time family policy; runtime behavior selection remains in local SML tables.
pub trait Policy {
    const ARCHITECTURE_NAME: &'static [u8];
    const TOKEN_EMBEDDING_NAME: &'static [u8];
    const OUTPUT_NORM_NAME: &'static [u8];
    const GLOBAL_TENSOR_COUNT: u32;
    const BLOCK_TENSOR_COUNT: u32;
    const QK_NORM_ROUTE: AttentionQkNormRoute;
    const TIED_OUTPUT: bool;
}

#[derive(Clone, Copy)]
pub struct BeginRuntime<'a> {
    pub event: ContractBegin<'a, Parameters>,
    pub begin_result: &'a Cell<ChildResult>,
    pub global_result: &'a Cell<ChildResult>,
    pub reset_result: &'a Cell<ChildResult>,
    pub result: &'a Cell<ChildResult>,
}

#[derive(Clone, Copy)]
pub struct BlockRuntime<'a> {
    pub event: event::BlockBuild,
    pub child_result: &'a Cell<ChildResult>,
    pub result: &'a Cell<ChildResult>,
}

#[derive(Clone, Copy)]
pub struct TopologyRuntime<'a> {
    pub child_result: &'a Cell<ChildResult>,
    pub result: &'a Cell<ChildResult>,
}

#[derive(Clone, Copy)]
pub struct PlanRuntime<'a> {
    pub child_result: &'a Cell<ChildResult>,
    pub result: &'a Cell<ChildResult>,
}

#[derive(Clone, Copy)]
pub struct ValidateRuntime<'a> {
    pub event: event::BlockValidation,
    pub child_result: &'a Cell<ChildResult>,
    pub descriptor: &'a Cell<Result<generation::event::BlockDescriptor, Error>>,
    pub result: &'a Cell<ChildResult>,
}

#[derive(Clone, Copy)]
pub struct AuditRuntime<'a> {
    pub event: event::BlockAudit,
    pub child_result: &'a Cell<ChildResult>,
    pub result: &'a Cell<ChildResult>,
}

#[derive(Clone, Copy)]
pub struct StageRuntime<'a> {
    pub event: event::StageAudit,
    pub child_result: &'a Cell<ChildResult>,
    pub result: &'a Cell<ChildResult>,
}

#[derive(Clone, Copy)]
pub struct VisitRuntime<'a> {
    pub child_result: &'a Cell<Result<ContractDescriptor, Error>>,
    pub result: &'a Cell<Result<ContractDescriptor, Error>>,
}

#[derive(Clone, Copy)]
pub struct BlockVisitRuntime<'a> {
    pub event: event::BlockVisit,
    pub result: &'a Cell<Result<generation::event::BlockDescriptor, Error>>,
}

#[derive(Clone, Copy)]
pub struct ResetRuntime<'a> {
    pub child_result: &'a Cell<ChildResult>,
    pub result: &'a Cell<ChildResult>,
}

#[derive(Clone, Copy)]
pub struct ReleaseRuntime<'a> {
    pub child_result: &'a RefCell<Result<generation::event::Storage, Error>>,
    pub result: &'a RefCell<Result<event::Storage, Error>>,
}

#[derive(Clone, Copy, Debug)]
pub struct UnexpectedRuntime;

pub struct Context<P> {
    builder: generation::Builder,
    parameters: Parameters,
    blocks: Vec<Option<generation::event::BlockDescriptor>>,
    policy: PhantomData<P>,
}

impl<P: Policy> Context<P> {
    /// Constructs shared family context from child actors and caller-preallocated storage.
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
            builder,
            parameters: Parameters::default(),
            blocks,
            policy: PhantomData,
        })
    }
    const fn layer(&self) -> LayerExecution {
        LayerExecution::new(
            ResidualRoute::Attention,
            P::QK_NORM_ROUTE,
            AttentionValueRoute::DedicatedValue,
            AttentionVNormRoute::None,
            AttentionWindowRoute::FullContext,
            self.parameters.attention_key_length,
            self.parameters.attention_value_length,
            self.parameters.rope_dimension_count,
            self.parameters.rope_freq_base,
        )
    }

    fn topology_values(&self) -> Option<(u32, u64)> {
        let blocks = u32::try_from(self.parameters.block_count).ok()?;
        let embedding = u64::try_from(self.parameters.embedding_length).ok()?;
        let tensors = blocks
            .checked_mul(P::BLOCK_TENSOR_COUNT)?
            .checked_add(P::GLOBAL_TENSOR_COUNT)?;
        let workspace = u64::from(tensors).checked_mul(embedding)?.checked_mul(4)?;
        Some((tensors, workspace))
    }
}

const fn child_ok(result: &Cell<ChildResult>) -> bool {
    result.get().is_ok()
}

impl<P: Policy> Context<P> {
    pub fn guard_begin_valid(&self, event: &BeginRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.architecture == P::ARCHITECTURE_NAME
            && event.event.parameters.block_count > 0
            && event.event.parameters.embedding_length > 0
            && event.event.parameters.context_length > 0)
    }
    pub fn guard_begin_architecture_invalid(&self, event: &BeginRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.architecture != P::ARCHITECTURE_NAME)
    }
    pub fn guard_begin_parameters_invalid(&self, event: &BeginRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.architecture == P::ARCHITECTURE_NAME && !self.guard_begin_valid(event)?)
    }
    pub fn effect_begin_child(&mut self, event: BeginRuntime<'_>) -> Result<(), ()> {
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
    pub fn guard_begin_child_ok(&self, event: &BeginRuntime<'_>) -> Result<bool, ()> {
        Ok(child_ok(event.begin_result))
    }
    pub fn guard_begin_child_error(&self, event: &BeginRuntime<'_>) -> Result<bool, ()> {
        Ok(!child_ok(event.begin_result))
    }
    pub fn effect_global_child(&mut self, event: BeginRuntime<'_>) -> Result<(), ()> {
        event.global_result.set(self.builder.process_event(
            generation::event::GlobalBindings::new(
                P::TOKEN_EMBEDDING_NAME,
                P::OUTPUT_NORM_NAME,
                P::TIED_OUTPUT,
            ),
        ));
        Ok(())
    }
    pub fn effect_begin_child_error(&mut self, event: BeginRuntime<'_>) -> Result<(), ()> {
        event.result.set(event.begin_result.get());
        Ok(())
    }
    pub fn guard_global_child_ok(&self, event: &BeginRuntime<'_>) -> Result<bool, ()> {
        Ok(child_ok(event.global_result))
    }
    pub fn guard_global_child_error(&self, event: &BeginRuntime<'_>) -> Result<bool, ()> {
        Ok(!child_ok(event.global_result))
    }
    pub fn effect_begin_complete(&mut self, event: BeginRuntime<'_>) -> Result<(), ()> {
        self.parameters = event.event.parameters;
        event.result.set(Ok(()));
        Ok(())
    }
    pub fn effect_begin_reset(&mut self, event: BeginRuntime<'_>) -> Result<(), ()> {
        event.reset_result.set(
            self.builder
                .process_event(generation::event::ContractReset::new()),
        );
        Ok(())
    }
    pub fn guard_begin_reset_ok(&self, event: &BeginRuntime<'_>) -> Result<bool, ()> {
        Ok(child_ok(event.reset_result))
    }
    pub fn guard_begin_reset_error(&self, event: &BeginRuntime<'_>) -> Result<bool, ()> {
        Ok(!child_ok(event.reset_result))
    }
    pub fn effect_global_child_error(&mut self, event: BeginRuntime<'_>) -> Result<(), ()> {
        event.result.set(event.global_result.get());
        Ok(())
    }
    pub fn effect_begin_reset_error(&mut self, event: BeginRuntime<'_>) -> Result<(), ()> {
        event.result.set(Err(Error::Internal));
        Ok(())
    }
    pub fn effect_begin_invalid_request(&mut self, event: BeginRuntime<'_>) -> Result<(), ()> {
        event.result.set(Err(Error::InvalidRequest));
        Ok(())
    }
    pub fn effect_begin_model_invalid(&mut self, event: BeginRuntime<'_>) -> Result<(), ()> {
        event.result.set(Err(Error::ModelInvalid));
        Ok(())
    }
    pub fn effect_busy_begin(&mut self, event: BeginRuntime<'_>) -> Result<(), ()> {
        event.result.set(Err(Error::Busy));
        Ok(())
    }

    pub fn guard_block_valid(&self, event: &BlockRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.index >= 0 && event.event.index < self.parameters.block_count)
    }
    pub fn guard_block_invalid(&self, event: &BlockRuntime<'_>) -> Result<bool, ()> {
        Ok(!self.guard_block_valid(event)?)
    }
    pub fn effect_block_child(&mut self, event: BlockRuntime<'_>) -> Result<(), ()> {
        event.child_result.set(
            self.builder
                .process_event(generation::event::AttentionBlock::new(
                    event.event.index,
                    self.layer(),
                )),
        );
        Ok(())
    }
    pub fn effect_block_invalid(&mut self, event: BlockRuntime<'_>) -> Result<(), ()> {
        event.result.set(Err(Error::InvalidRequest));
        Ok(())
    }
    pub fn guard_block_child_ok(&self, event: &BlockRuntime<'_>) -> Result<bool, ()> {
        Ok(child_ok(event.child_result))
    }
    pub fn guard_block_child_error(&self, event: &BlockRuntime<'_>) -> Result<bool, ()> {
        Ok(!child_ok(event.child_result))
    }
    pub fn effect_child_ok_block(&mut self, event: BlockRuntime<'_>) -> Result<(), ()> {
        event.result.set(Ok(()));
        Ok(())
    }
    pub fn effect_child_error_block(&mut self, event: BlockRuntime<'_>) -> Result<(), ()> {
        event.result.set(event.child_result.get());
        Ok(())
    }

    pub fn guard_topology_valid(&self, _event: &TopologyRuntime<'_>) -> Result<bool, ()> {
        Ok(self.topology_values().is_some())
    }
    pub fn guard_topology_invalid(&self, event: &TopologyRuntime<'_>) -> Result<bool, ()> {
        Ok(!self.guard_topology_valid(event)?)
    }
    pub fn effect_topology_child(&mut self, event: TopologyRuntime<'_>) -> Result<(), ()> {
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
    pub fn effect_topology_invalid(&mut self, event: TopologyRuntime<'_>) -> Result<(), ()> {
        event.result.set(Err(Error::InvalidRequest));
        Ok(())
    }
    pub fn guard_topology_child_ok(&self, event: &TopologyRuntime<'_>) -> Result<bool, ()> {
        Ok(child_ok(event.child_result))
    }
    pub fn guard_topology_child_error(&self, event: &TopologyRuntime<'_>) -> Result<bool, ()> {
        Ok(!child_ok(event.child_result))
    }
    pub fn effect_child_ok_topology(&mut self, event: TopologyRuntime<'_>) -> Result<(), ()> {
        event.result.set(Ok(()));
        Ok(())
    }
    pub fn effect_child_error_topology(&mut self, event: TopologyRuntime<'_>) -> Result<(), ()> {
        event.result.set(event.child_result.get());
        Ok(())
    }

    pub fn effect_plan_child(&mut self, event: PlanRuntime<'_>) -> Result<(), ()> {
        event
            .child_result
            .set(self.builder.process_event(generation::event::Plan::new()));
        Ok(())
    }
    pub fn guard_plan_child_ok(&self, event: &PlanRuntime<'_>) -> Result<bool, ()> {
        Ok(child_ok(event.child_result))
    }
    pub fn guard_plan_child_error(&self, event: &PlanRuntime<'_>) -> Result<bool, ()> {
        Ok(!child_ok(event.child_result))
    }
    pub fn effect_child_ok_plan(&mut self, event: PlanRuntime<'_>) -> Result<(), ()> {
        event.result.set(Ok(()));
        Ok(())
    }
    pub fn effect_child_error_plan(&mut self, event: PlanRuntime<'_>) -> Result<(), ()> {
        event.result.set(event.child_result.get());
        Ok(())
    }

    pub fn effect_validate_child(&mut self, event: ValidateRuntime<'_>) -> Result<(), ()> {
        event.child_result.set(
            self.builder
                .process_event(generation::event::BlockValidation::new(event.event.index)),
        );
        Ok(())
    }
    pub fn guard_validate_child_ok(&self, event: &ValidateRuntime<'_>) -> Result<bool, ()> {
        Ok(child_ok(event.child_result))
    }
    pub fn guard_validate_child_error(&self, event: &ValidateRuntime<'_>) -> Result<bool, ()> {
        Ok(!child_ok(event.child_result))
    }
    pub fn effect_validate_visit(&mut self, event: ValidateRuntime<'_>) -> Result<(), ()> {
        event.descriptor.set(
            self.builder
                .process_event(generation::event::BlockVisit::new(event.event.index)),
        );
        Ok(())
    }
    pub fn guard_validate_visit_ok(&self, event: &ValidateRuntime<'_>) -> Result<bool, ()> {
        Ok(event.descriptor.get().is_ok())
    }
    pub fn guard_validate_visit_error(&self, event: &ValidateRuntime<'_>) -> Result<bool, ()> {
        Ok(event.descriptor.get().is_err())
    }
    pub fn effect_cache_validated_block(&mut self, event: ValidateRuntime<'_>) -> Result<(), ()> {
        let index = usize::try_from(event.event.index).expect("validated index");
        self.blocks[index] = event.descriptor.get().ok();
        event.result.set(Ok(()));
        Ok(())
    }
    pub fn effect_validate_visit_error(&mut self, event: ValidateRuntime<'_>) -> Result<(), ()> {
        event.result.set(Err(Error::Internal));
        Ok(())
    }
    pub fn effect_child_error_validate(&mut self, event: ValidateRuntime<'_>) -> Result<(), ()> {
        event.result.set(event.child_result.get());
        Ok(())
    }

    pub fn effect_audit_child(&mut self, event: AuditRuntime<'_>) -> Result<(), ()> {
        event.child_result.set(
            self.builder
                .process_event(generation::event::BlockAudit::new(event.event.index)),
        );
        Ok(())
    }
    pub fn guard_audit_child_ok(&self, event: &AuditRuntime<'_>) -> Result<bool, ()> {
        Ok(child_ok(event.child_result))
    }
    pub fn guard_audit_child_error(&self, event: &AuditRuntime<'_>) -> Result<bool, ()> {
        Ok(!child_ok(event.child_result))
    }
    pub fn effect_child_ok_audit(&mut self, event: AuditRuntime<'_>) -> Result<(), ()> {
        event.result.set(Ok(()));
        Ok(())
    }
    pub fn effect_child_error_audit(&mut self, event: AuditRuntime<'_>) -> Result<(), ()> {
        event.result.set(event.child_result.get());
        Ok(())
    }

    pub fn effect_stage_child(&mut self, event: StageRuntime<'_>) -> Result<(), ()> {
        event.child_result.set(
            self.builder
                .process_event(generation::event::StageAudit::new(event.event.family)),
        );
        Ok(())
    }
    pub fn guard_stage_child_ok(&self, event: &StageRuntime<'_>) -> Result<bool, ()> {
        Ok(child_ok(event.child_result))
    }
    pub fn guard_stage_child_error(&self, event: &StageRuntime<'_>) -> Result<bool, ()> {
        Ok(!child_ok(event.child_result))
    }
    pub fn effect_child_ok_stage(&mut self, event: StageRuntime<'_>) -> Result<(), ()> {
        event.result.set(Ok(()));
        Ok(())
    }
    pub fn effect_child_error_stage(&mut self, event: StageRuntime<'_>) -> Result<(), ()> {
        event.result.set(event.child_result.get());
        Ok(())
    }

    pub fn effect_visit_child(&mut self, event: VisitRuntime<'_>) -> Result<(), ()> {
        event.child_result.set(
            self.builder
                .process_event(generation::event::ContractVisit::new()),
        );
        Ok(())
    }
    pub fn guard_visit_child_ok(&self, event: &VisitRuntime<'_>) -> Result<bool, ()> {
        Ok(event.child_result.get().is_ok())
    }
    pub fn guard_visit_child_error(&self, event: &VisitRuntime<'_>) -> Result<bool, ()> {
        Ok(event.child_result.get().is_err())
    }
    pub fn effect_visit_ok(&mut self, event: VisitRuntime<'_>) -> Result<(), ()> {
        event.result.set(event.child_result.get());
        Ok(())
    }
    pub fn effect_visit_error(&mut self, event: VisitRuntime<'_>) -> Result<(), ()> {
        event.result.set(event.child_result.get());
        Ok(())
    }

    pub fn guard_block_visit_valid(&self, event: &BlockVisitRuntime<'_>) -> Result<bool, ()> {
        Ok(usize::try_from(event.event.index)
            .ok()
            .and_then(|index| self.blocks.get(index))
            .is_some_and(Option::is_some))
    }
    pub fn guard_block_visit_invalid(&self, event: &BlockVisitRuntime<'_>) -> Result<bool, ()> {
        Ok(!self.guard_block_visit_valid(event)?)
    }
    pub fn effect_block_visit(&mut self, event: BlockVisitRuntime<'_>) -> Result<(), ()> {
        let index = usize::try_from(event.event.index).expect("guarded index");
        event
            .result
            .set(Ok(self.blocks[index].expect("validated block")));
        Ok(())
    }
    pub fn effect_invalid_block_visit(&mut self, event: BlockVisitRuntime<'_>) -> Result<(), ()> {
        event.result.set(Err(Error::InvalidRequest));
        Ok(())
    }

    pub fn effect_reset_child(&mut self, event: ResetRuntime<'_>) -> Result<(), ()> {
        event.child_result.set(
            self.builder
                .process_event(generation::event::ContractReset::new()),
        );
        Ok(())
    }
    pub fn guard_reset_child_ok(&self, event: &ResetRuntime<'_>) -> Result<bool, ()> {
        Ok(child_ok(event.child_result))
    }
    pub fn guard_reset_child_error(&self, event: &ResetRuntime<'_>) -> Result<bool, ()> {
        Ok(!child_ok(event.child_result))
    }
    pub fn effect_reset_ok(&mut self, event: ResetRuntime<'_>) -> Result<(), ()> {
        self.parameters = Parameters::default();
        self.blocks.fill(None);
        event.result.set(Ok(()));
        Ok(())
    }
    pub fn effect_reset_error(&mut self, event: ResetRuntime<'_>) -> Result<(), ()> {
        event.result.set(event.child_result.get());
        Ok(())
    }

    pub fn effect_release_child(&mut self, event: ReleaseRuntime<'_>) -> Result<(), ()> {
        *event.child_result.borrow_mut() = self
            .builder
            .process_event(generation::event::StorageRelease::new());
        Ok(())
    }
    pub fn guard_release_child_ok(&self, event: &ReleaseRuntime<'_>) -> Result<bool, ()> {
        Ok(event.child_result.borrow().is_ok())
    }
    pub fn guard_release_child_error(&self, event: &ReleaseRuntime<'_>) -> Result<bool, ()> {
        Ok(event.child_result.borrow().is_err())
    }
    pub fn effect_release_ok(&mut self, event: ReleaseRuntime<'_>) -> Result<(), ()> {
        let common =
            core::mem::replace(&mut *event.child_result.borrow_mut(), Err(Error::Internal))
                .expect("guarded release");
        *event.result.borrow_mut() = Ok(event::Storage {
            common,
            blocks: core::mem::take(&mut self.blocks),
        });
        Ok(())
    }
    pub fn effect_release_error(&mut self, event: ReleaseRuntime<'_>) -> Result<(), ()> {
        *event.result.borrow_mut() = Err(event
            .child_result
            .borrow()
            .as_ref()
            .expect_err("guarded release error")
            .to_owned());
        Ok(())
    }
    pub fn effect_busy_release(&mut self, event: ReleaseRuntime<'_>) -> Result<(), ()> {
        *event.result.borrow_mut() = Err(Error::Busy);
        Ok(())
    }

    pub fn guard_never(&self, _event: &UnexpectedRuntime) -> Result<bool, ()> {
        Ok(false)
    }
    pub fn effect_unexpected(&mut self) -> Result<(), ()> {
        Ok(())
    }
}
