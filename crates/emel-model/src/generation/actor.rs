//! Public generation builder and private same-RTC dispatch runtimes.

use core::cell::{Cell, RefCell};
use core::fmt;

use crate::generation::quantized_path::{self as capability, Outcome, Scope};
use emel_tensor::dtype::SerializedType;

use crate::catalog::event::{ModelDescriptor, ModelIdentity, TensorDescriptor};
use crate::catalog::{self, Catalog};

use super::event::{self, BlockDescriptor, Error, Storage, StorageBindError};
use super::sm::{
    GenerationBuilderEvents, GenerationBuilderStateMachine, GenerationBuilderStateMachineContext,
};
use super::storage::{AuditObservation, BlockSlot, TensorView};
use super::{
    AttentionQkNormRoute, ContractDescriptor, QUANTIZED_STAGE_FAMILY_COUNT, QuantizedContractKind,
    QuantizedStageFamily, StageAudit, StepKind, StepPlan, TopologyDescriptor,
};

const OUTPUT_NAME: &[u8] = b"output.weight";
const SLOT_ATTENTION_NORM: usize = 0;
const SLOT_ATTENTION_Q: usize = 1;
const SLOT_ATTENTION_K: usize = 2;
const SLOT_ATTENTION_V: usize = 3;
const SLOT_ATTENTION_Q_NORM: usize = 4;
const SLOT_ATTENTION_K_NORM: usize = 5;
const SLOT_ATTENTION_OUTPUT: usize = 6;
const SLOT_SHORTCONV_CONV: usize = 7;
const SLOT_SHORTCONV_IN: usize = 8;
const SLOT_SHORTCONV_OUT: usize = 9;
const SLOT_FFN_NORM: usize = 10;
const SLOT_FFN_GATE: usize = 11;
const SLOT_FFN_DOWN: usize = 12;
const SLOT_FFN_UP: usize = 13;
const BLOCK_QUERY_COUNT: usize = 14;
const REJECT_QUERY_COUNT: usize = 6;

type CatalogQuery = Result<Option<TensorDescriptor>, catalog::event::Error>;
type CapabilityQuery = Result<Outcome, capability::Error>;

#[derive(Clone, Copy)]
pub(super) struct StorageBindRuntime<'a> {
    pub(super) storage: &'a RefCell<Option<Storage>>,
    pub(super) result: &'a Cell<Result<(), Error>>,
}

#[derive(Clone, Copy)]
pub(super) struct BeginRuntime<'a> {
    pub(super) event: event::ContractBegin,
    pub(super) descriptor: &'a Cell<Result<ModelDescriptor, catalog::event::Error>>,
    pub(super) result: &'a Cell<Result<(), Error>>,
}

#[derive(Clone, Copy)]
pub(super) struct GlobalRuntime<'a> {
    pub(super) event: event::GlobalBindings<'a>,
    pub(super) queries: &'a RefCell<[CatalogQuery; 3]>,
    pub(super) result: &'a Cell<Result<(), Error>>,
}

#[derive(Clone, Copy)]
pub(super) struct AttentionRuntime<'a> {
    pub(super) event: event::AttentionBlock,
    pub(super) queries: &'a RefCell<[CatalogQuery; BLOCK_QUERY_COUNT]>,
    pub(super) result: &'a Cell<Result<(), Error>>,
}

#[derive(Clone, Copy)]
pub(super) struct ShortconvRuntime<'a> {
    pub(super) event: event::ShortconvBlock,
    pub(super) queries: &'a RefCell<[CatalogQuery; BLOCK_QUERY_COUNT]>,
    pub(super) result: &'a Cell<Result<(), Error>>,
}

#[derive(Clone, Copy)]
pub(super) struct RejectRuntime<'a> {
    pub(super) event: event::RejectBlockTensors,
    pub(super) queries: &'a RefCell<[CatalogQuery; REJECT_QUERY_COUNT]>,
    pub(super) result: &'a Cell<Result<(), Error>>,
}

#[derive(Clone, Copy)]
pub(super) struct TopologyRuntime<'a> {
    pub(super) event: event::Topology,
    pub(super) result: &'a Cell<Result<(), Error>>,
}

#[derive(Clone, Copy)]
pub(super) struct PlanRuntime<'a> {
    pub(super) result: &'a Cell<Result<(), Error>>,
}

#[derive(Clone, Copy)]
pub(super) struct ValidateRuntime<'a> {
    pub(super) event: event::BlockValidation,
    pub(super) result: &'a Cell<Result<(), Error>>,
}

#[derive(Clone, Copy)]
pub(super) struct AuditRuntime<'a> {
    pub(super) event: event::BlockAudit,
    observations: &'a RefCell<[Option<AuditObservation>; QUANTIZED_STAGE_FAMILY_COUNT]>,
    pub(super) result: &'a Cell<Result<(), Error>>,
}

#[derive(Clone, Copy)]
pub(super) struct StageRuntime<'a> {
    pub(super) event: event::StageAudit,
    outcome: &'a Cell<Option<CapabilityQuery>>,
    pub(super) result: &'a Cell<Result<(), Error>>,
}

#[derive(Clone, Copy)]
pub(super) struct VisitRuntime<'a> {
    pub(super) result: &'a Cell<Result<ContractDescriptor, Error>>,
}

#[derive(Clone, Copy)]
pub(super) struct BlockVisitRuntime<'a> {
    pub(super) event: event::BlockVisit,
    pub(super) result: &'a RefCell<Result<BlockDescriptor, Error>>,
}

#[derive(Clone, Copy)]
pub(super) struct ResetRuntime<'a> {
    pub(super) result: &'a Cell<Result<(), Error>>,
}

#[derive(Clone, Copy)]
pub(super) struct StorageReleaseRuntime<'a> {
    pub(super) storage: &'a RefCell<Option<Storage>>,
    pub(super) result: &'a Cell<Result<(), Error>>,
}

#[derive(Clone, Copy, Debug)]
pub(super) struct UnexpectedRuntime;

#[allow(clippy::struct_field_names)]
pub(super) struct Context {
    catalog: Catalog,
    capability: capability::Resolver,
    storage: Option<Storage>,
    model: Option<ModelIdentity>,
    block_count: u32,
    context_length: i32,
    token_embedding: Option<TensorDescriptor>,
    output_norm: Option<TensorDescriptor>,
    output: Option<TensorDescriptor>,
    topology: Option<TopologyDescriptor>,
    prefill: Option<StepPlan>,
    decode: Option<StepPlan>,
    audits: [StageAudit; QUANTIZED_STAGE_FAMILY_COUNT],
    finalized: [bool; QUANTIZED_STAGE_FAMILY_COUNT],
    bound_blocks: u32,
    validated_blocks: u32,
    audited_blocks: u32,
    finalized_stages: u32,
}

impl Context {
    fn reset_contract(&mut self) {
        self.storage_mut().regions.clear();
        self.model = None;
        self.block_count = 0;
        self.context_length = 0;
        self.token_embedding = None;
        self.output_norm = None;
        self.output = None;
        self.topology = None;
        self.prefill = None;
        self.decode = None;
        self.audits =
            core::array::from_fn(|index| StageAudit::empty(QuantizedStageFamily::ALL[index]));
        self.finalized = [false; QUANTIZED_STAGE_FAMILY_COUNT];
        self.bound_blocks = 0;
        self.validated_blocks = 0;
        self.audited_blocks = 0;
        self.finalized_stages = 0;
    }

    const fn storage(&self) -> &Storage {
        self.storage.as_ref().expect("bound state owns storage")
    }
    const fn storage_mut(&mut self) -> &mut Storage {
        self.storage.as_mut().expect("bound state owns storage")
    }

    fn index_valid(&self, index: i32) -> bool {
        index >= 0 && u32::try_from(index).is_ok_and(|value| value < self.block_count)
    }

    fn block(&self, index: i32) -> &BlockSlot {
        &self.storage().regions.blocks[usize::try_from(index).expect("validated index")]
    }

    fn query_name(&mut self, name: &[u8]) -> CatalogQuery {
        let model = self.model.expect("building state has model identity");
        self.catalog
            .process_event(catalog::event::FindTensor::new(model, name))
    }

    fn query_block(&mut self, index: i32, suffix: &[u8]) -> CatalogQuery {
        let (name, length) = block_name(index, suffix);
        self.query_name(&name[..length])
    }

    fn query_capability(&mut self, scope: Scope, tensor: TensorDescriptor) -> CapabilityQuery {
        self.capability
            .process_event(capability::event::Query::new(scope, tensor.tensor_type()))
    }
}

/// Single-writer, run-to-completion common generation builder.
pub struct Builder<CatalogActor = Catalog, CapabilityActor = capability::Resolver> {
    machine: GenerationBuilderStateMachine<Context>,
    marker: core::marker::PhantomData<fn() -> (CatalogActor, CapabilityActor)>,
}

impl Builder<Catalog, capability::Resolver> {
    /// Constructs a builder from its public child actors without allocation.
    #[must_use]
    pub fn new(catalog: Catalog, capability: capability::Resolver) -> Self {
        let audits =
            core::array::from_fn(|index| StageAudit::empty(QuantizedStageFamily::ALL[index]));
        Self {
            machine: GenerationBuilderStateMachine::new(Context {
                catalog,
                capability,
                storage: None,
                model: None,
                block_count: 0,
                context_length: 0,
                token_embedding: None,
                output_norm: None,
                output: None,
                topology: None,
                prefill: None,
                decode: None,
                audits,
                finalized: [false; QUANTIZED_STAGE_FAMILY_COUNT],
                bound_blocks: 0,
                validated_blocks: 0,
                audited_blocks: 0,
                finalized_stages: 0,
            }),
            marker: core::marker::PhantomData,
        }
    }

    /// Dispatches one public event synchronously.
    pub fn process_event<E>(&mut self, event: E) -> E::Output
    where
        E: event::Event<Catalog, capability::Resolver>,
    {
        event.dispatch(self)
    }

    // Rejected binds return all caller-preallocated regions without allocating a box.
    #[allow(clippy::result_large_err)]
    pub(crate) fn storage_bind(
        &mut self,
        event: event::StorageBind,
    ) -> Result<(), StorageBindError> {
        let storage = RefCell::new(Some(event.storage));
        let result = Cell::new(Err(Error::Internal));
        let runtime = StorageBindRuntime {
            storage: &storage,
            result: &result,
        };
        if self
            .machine
            .process_event(GenerationBuilderEvents::StorageBind(runtime))
            .is_err()
        {
            result.set(Err(Error::Internal));
        }
        result.get().map_err(|error| {
            StorageBindError::new(
                error,
                storage.into_inner().expect("rejected bind retains storage"),
            )
        })
    }

    pub(crate) fn contract_begin(&mut self, event: event::ContractBegin) -> Result<(), Error> {
        let descriptor = Cell::new(Err(catalog::event::Error::Internal));
        let result = Cell::new(Err(Error::UnexpectedEvent));
        self.machine
            .process_event(GenerationBuilderEvents::Begin(BeginRuntime {
                event,
                descriptor: &descriptor,
                result: &result,
            }))
            .map_err(|_| Error::Internal)?;
        result.get()
    }

    pub(crate) fn global_bindings(
        &mut self,
        event: event::GlobalBindings<'_>,
    ) -> Result<(), Error> {
        let queries = RefCell::new([Ok(None); 3]);
        let result = Cell::new(Err(Error::UnexpectedEvent));
        let runtime = GlobalRuntime {
            event,
            queries: &queries,
            result: &result,
        };
        self.machine
            .process_event(GenerationBuilderEvents::Global(runtime))
            .map_err(|_| Error::Internal)?;
        result.get()
    }

    pub(crate) fn attention_block(&mut self, event: event::AttentionBlock) -> Result<(), Error> {
        let queries = RefCell::new([Ok(None); BLOCK_QUERY_COUNT]);
        let result = Cell::new(Err(Error::UnexpectedEvent));
        let runtime = AttentionRuntime {
            event,
            queries: &queries,
            result: &result,
        };
        self.machine
            .process_event(GenerationBuilderEvents::Attention(runtime))
            .map_err(|_| Error::Internal)?;
        result.get()
    }

    pub(crate) fn shortconv_block(&mut self, event: event::ShortconvBlock) -> Result<(), Error> {
        let queries = RefCell::new([Ok(None); BLOCK_QUERY_COUNT]);
        let result = Cell::new(Err(Error::UnexpectedEvent));
        let runtime = ShortconvRuntime {
            event,
            queries: &queries,
            result: &result,
        };
        self.machine
            .process_event(GenerationBuilderEvents::Shortconv(runtime))
            .map_err(|_| Error::Internal)?;
        result.get()
    }

    pub(crate) fn reject_block_tensors(
        &mut self,
        event: event::RejectBlockTensors,
    ) -> Result<(), Error> {
        let queries = RefCell::new([Ok(None); REJECT_QUERY_COUNT]);
        let result = Cell::new(Err(Error::UnexpectedEvent));
        self.machine
            .process_event(GenerationBuilderEvents::Reject(RejectRuntime {
                event,
                queries: &queries,
                result: &result,
            }))
            .map_err(|_| Error::Internal)?;
        result.get()
    }

    pub(crate) fn topology(&mut self, event: event::Topology) -> Result<(), Error> {
        let result = Cell::new(Err(Error::UnexpectedEvent));
        self.machine
            .process_event(GenerationBuilderEvents::Topology(TopologyRuntime {
                event,
                result: &result,
            }))
            .map_err(|_| Error::Internal)?;
        result.get()
    }
    pub(crate) fn plan(&mut self, _event: event::Plan) -> Result<(), Error> {
        let result = Cell::new(Err(Error::UnexpectedEvent));
        self.machine
            .process_event(GenerationBuilderEvents::Plan(PlanRuntime {
                result: &result,
            }))
            .map_err(|_| Error::Internal)?;
        result.get()
    }
    pub(crate) fn block_validation(&mut self, event: event::BlockValidation) -> Result<(), Error> {
        let result = Cell::new(Err(Error::UnexpectedEvent));
        self.machine
            .process_event(GenerationBuilderEvents::Validate(ValidateRuntime {
                event,
                result: &result,
            }))
            .map_err(|_| Error::Internal)?;
        result.get()
    }
    pub(crate) fn block_audit(&mut self, event: event::BlockAudit) -> Result<(), Error> {
        let observations = RefCell::new([None; QUANTIZED_STAGE_FAMILY_COUNT]);
        let result = Cell::new(Err(Error::UnexpectedEvent));
        self.machine
            .process_event(GenerationBuilderEvents::Audit(AuditRuntime {
                event,
                observations: &observations,
                result: &result,
            }))
            .map_err(|_| Error::Internal)?;
        result.get()
    }
    pub(crate) fn stage_audit(&mut self, event: event::StageAudit) -> Result<(), Error> {
        let outcome = Cell::new(None);
        let result = Cell::new(Err(Error::UnexpectedEvent));
        self.machine
            .process_event(GenerationBuilderEvents::Stage(StageRuntime {
                event,
                outcome: &outcome,
                result: &result,
            }))
            .map_err(|_| Error::Internal)?;
        result.get()
    }
    pub(crate) fn contract_visit(
        &mut self,
        _event: event::ContractVisit,
    ) -> Result<ContractDescriptor, Error> {
        let result = Cell::new(Err(Error::UnexpectedEvent));
        self.machine
            .process_event(GenerationBuilderEvents::Visit(VisitRuntime {
                result: &result,
            }))
            .map_err(|_| Error::Internal)?;
        result.get()
    }
    pub(crate) fn block_visit(
        &mut self,
        event: event::BlockVisit,
    ) -> Result<BlockDescriptor, Error> {
        let result = RefCell::new(Err(Error::UnexpectedEvent));
        self.machine
            .process_event(GenerationBuilderEvents::BlockVisit(BlockVisitRuntime {
                event,
                result: &result,
            }))
            .map_err(|_| Error::Internal)?;
        result.into_inner()
    }
    pub(crate) fn contract_reset(&mut self, _event: event::ContractReset) -> Result<(), Error> {
        let result = Cell::new(Err(Error::UnexpectedEvent));
        self.machine
            .process_event(GenerationBuilderEvents::Reset(ResetRuntime {
                result: &result,
            }))
            .map_err(|_| Error::Internal)?;
        result.get()
    }
    pub(crate) fn storage_release(
        &mut self,
        _event: event::StorageRelease,
    ) -> Result<Storage, Error> {
        let storage = RefCell::new(None);
        let result = Cell::new(Err(Error::UnexpectedEvent));
        self.machine
            .process_event(GenerationBuilderEvents::Release(StorageReleaseRuntime {
                storage: &storage,
                result: &result,
            }))
            .map_err(|_| Error::Internal)?;
        result.get()?;
        storage.into_inner().ok_or(Error::Internal)
    }
}

impl fmt::Debug for Builder<Catalog, capability::Resolver> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Builder").finish_non_exhaustive()
    }
}

pub(super) fn block_name(index: i32, suffix: &[u8]) -> ([u8; 64], usize) {
    let mut output = [0u8; 64];
    output[..4].copy_from_slice(b"blk.");
    let value = u32::try_from(index).expect("validated nonnegative block index");
    let mut digits = [0u8; 10];
    let mut count = 0usize;
    let mut remaining = value;
    loop {
        digits[count] = b'0' + u8::try_from(remaining % 10).expect("digit");
        count += 1;
        remaining /= 10;
        if remaining == 0 {
            break;
        }
    }
    for offset in 0..count {
        output[4 + offset] = digits[count - offset - 1];
    }
    let separator = 4 + count;
    output[separator] = b'.';
    let start = separator + 1;
    output[start..start + suffix.len()].copy_from_slice(suffix);
    (output, start + suffix.len())
}

fn required_queries_valid(queries: &[CatalogQuery], required: &[usize]) -> bool {
    required
        .iter()
        .all(|index| matches!(queries[*index], Ok(Some(_))))
}
fn required_queries_missing(queries: &[CatalogQuery], required: &[usize]) -> bool {
    queries.iter().all(Result::is_ok) && !required_queries_valid(queries, required)
}
fn queries_dependency_error(queries: &[CatalogQuery]) -> bool {
    queries.iter().any(Result::is_err)
}

const COMMON: [usize; 5] = [
    SLOT_ATTENTION_NORM,
    SLOT_FFN_NORM,
    SLOT_FFN_GATE,
    SLOT_FFN_DOWN,
    SLOT_FFN_UP,
];
const ATTN_PLAIN: [usize; 9] = [
    SLOT_ATTENTION_NORM,
    SLOT_ATTENTION_Q,
    SLOT_ATTENTION_K,
    SLOT_ATTENTION_V,
    SLOT_ATTENTION_OUTPUT,
    SLOT_FFN_NORM,
    SLOT_FFN_GATE,
    SLOT_FFN_DOWN,
    SLOT_FFN_UP,
];
const ATTN_QK: [usize; 11] = [
    SLOT_ATTENTION_NORM,
    SLOT_ATTENTION_Q,
    SLOT_ATTENTION_K,
    SLOT_ATTENTION_V,
    SLOT_ATTENTION_Q_NORM,
    SLOT_ATTENTION_K_NORM,
    SLOT_ATTENTION_OUTPUT,
    SLOT_FFN_NORM,
    SLOT_FFN_GATE,
    SLOT_FFN_DOWN,
    SLOT_FFN_UP,
];
const ATTN_SHARED: [usize; 8] = [
    SLOT_ATTENTION_NORM,
    SLOT_ATTENTION_Q,
    SLOT_ATTENTION_K,
    SLOT_ATTENTION_OUTPUT,
    SLOT_FFN_NORM,
    SLOT_FFN_GATE,
    SLOT_FFN_DOWN,
    SLOT_FFN_UP,
];
const ATTN_QK_SHARED: [usize; 10] = [
    SLOT_ATTENTION_NORM,
    SLOT_ATTENTION_Q,
    SLOT_ATTENTION_K,
    SLOT_ATTENTION_Q_NORM,
    SLOT_ATTENTION_K_NORM,
    SLOT_ATTENTION_OUTPUT,
    SLOT_FFN_NORM,
    SLOT_FFN_GATE,
    SLOT_FFN_DOWN,
    SLOT_FFN_UP,
];
const SHORTCONV: [usize; 8] = [
    SLOT_ATTENTION_NORM,
    SLOT_SHORTCONV_CONV,
    SLOT_SHORTCONV_IN,
    SLOT_SHORTCONV_OUT,
    SLOT_FFN_NORM,
    SLOT_FFN_GATE,
    SLOT_FFN_DOWN,
    SLOT_FFN_UP,
];

impl GenerationBuilderStateMachineContext for Context {
    fn guard_begin_request_valid(&self, event: &BeginRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.block_count > 0
            && event.event.context_length > 0
            && usize::try_from(event.event.block_count)
                .is_ok_and(|count| count <= self.storage().regions.blocks.len()))
    }
    fn guard_begin_request_invalid(&self, event: &BeginRuntime<'_>) -> Result<bool, ()> {
        Ok(!self.guard_begin_request_valid(event)?)
    }
    fn effect_describe_model(&mut self, event: BeginRuntime<'_>) -> Result<(), ()> {
        event.descriptor.set(
            self.catalog
                .process_event(catalog::event::DescribeModel::new(event.event.model)),
        );
        Ok(())
    }
    fn guard_begin_catalog_valid(&self, event: &BeginRuntime<'_>) -> Result<bool, ()> {
        Ok(event
            .descriptor
            .get()
            .is_ok_and(|descriptor| descriptor.tensor_count() > 0))
    }
    fn guard_begin_wrong_identity(&self, event: &BeginRuntime<'_>) -> Result<bool, ()> {
        Ok(matches!(
            event.descriptor.get(),
            Err(catalog::event::Error::WrongModelIdentity)
        ))
    }
    fn guard_begin_stale_identity(&self, event: &BeginRuntime<'_>) -> Result<bool, ()> {
        Ok(matches!(
            event.descriptor.get(),
            Err(catalog::event::Error::StaleModelIdentity)
        ))
    }
    fn guard_begin_storage_unavailable(&self, event: &BeginRuntime<'_>) -> Result<bool, ()> {
        Ok(matches!(
            event.descriptor.get(),
            Err(catalog::event::Error::StorageUnavailable)
        ))
    }
    fn guard_begin_model_invalid(&self, event: &BeginRuntime<'_>) -> Result<bool, ()> {
        Ok(matches!(
            event.descriptor.get(),
            Ok(_) | Err(catalog::event::Error::ModelInvalid)
        ) && !self.guard_begin_catalog_valid(event)?)
    }
    fn guard_begin_dependency_error(&self, event: &BeginRuntime<'_>) -> Result<bool, ()> {
        Ok(!self.guard_begin_catalog_valid(event)?
            && !self.guard_begin_wrong_identity(event)?
            && !self.guard_begin_stale_identity(event)?
            && !self.guard_begin_storage_unavailable(event)?
            && !self.guard_begin_model_invalid(event)?)
    }
    fn effect_begin(&mut self, event: BeginRuntime<'_>) -> Result<(), ()> {
        self.reset_contract();
        self.model = Some(event.event.model);
        self.block_count = u32::try_from(event.event.block_count).expect("guarded");
        self.context_length = event.event.context_length;
        event.result.set(Ok(()));
        Ok(())
    }
    fn effect_begin_wrong_identity(&mut self, event: BeginRuntime<'_>) -> Result<(), ()> {
        event.result.set(Err(Error::WrongModelIdentity));
        Ok(())
    }
    fn effect_begin_stale_identity(&mut self, event: BeginRuntime<'_>) -> Result<(), ()> {
        event.result.set(Err(Error::StaleModelIdentity));
        Ok(())
    }
    fn effect_begin_storage_unavailable(&mut self, event: BeginRuntime<'_>) -> Result<(), ()> {
        event.result.set(Err(Error::StorageUnavailable));
        Ok(())
    }
    fn effect_begin_model_invalid(&mut self, event: BeginRuntime<'_>) -> Result<(), ()> {
        event.result.set(Err(Error::ModelInvalid));
        Ok(())
    }
    fn effect_begin_dependency_error(&mut self, event: BeginRuntime<'_>) -> Result<(), ()> {
        event.result.set(Err(Error::Dependency));
        Ok(())
    }

    fn guard_global_request_direct(&self, event: &GlobalRuntime<'_>) -> Result<bool, ()> {
        Ok(self.token_embedding.is_none()
            && !event.event.allow_tied_output
            && !event.event.token_embedding.is_empty()
            && !event.event.output_norm.is_empty())
    }
    fn guard_global_request_tied(&self, event: &GlobalRuntime<'_>) -> Result<bool, ()> {
        Ok(self.token_embedding.is_none()
            && event.event.allow_tied_output
            && !event.event.token_embedding.is_empty()
            && !event.event.output_norm.is_empty())
    }
    fn guard_global_request_invalid(&self, event: &GlobalRuntime<'_>) -> Result<bool, ()> {
        Ok(!self.guard_global_request_direct(event)? && !self.guard_global_request_tied(event)?)
    }
    fn effect_query_globals(&mut self, event: GlobalRuntime<'_>) -> Result<(), ()> {
        let mut queries = event.queries.borrow_mut();
        queries[0] = self.query_name(event.event.token_embedding);
        queries[1] = self.query_name(event.event.output_norm);
        queries[2] = self.query_name(OUTPUT_NAME);
        Ok(())
    }
    fn guard_globals_direct(&self, event: &GlobalRuntime<'_>) -> Result<bool, ()> {
        Ok(required_queries_valid(&*event.queries.borrow(), &[0, 1, 2]))
    }
    fn guard_globals_tied(&self, event: &GlobalRuntime<'_>) -> Result<bool, ()> {
        let q = event.queries.borrow();
        Ok(matches!(q[0], Ok(Some(_))) && matches!(q[1], Ok(Some(_))) && matches!(q[2], Ok(None)))
    }
    fn guard_globals_missing(&self, event: &GlobalRuntime<'_>) -> Result<bool, ()> {
        Ok(event.queries.borrow().iter().all(Result::is_ok)
            && !self.guard_globals_direct(event)?
            && !self.guard_globals_tied(event)?)
    }
    fn guard_globals_dependency_error(&self, event: &GlobalRuntime<'_>) -> Result<bool, ()> {
        Ok(queries_dependency_error(&*event.queries.borrow()))
    }
    fn effect_bind_globals_direct(&mut self, event: GlobalRuntime<'_>) -> Result<(), ()> {
        let q = event.queries.borrow();
        self.token_embedding = q[0].ok().flatten();
        self.output_norm = q[1].ok().flatten();
        self.output = q[2].ok().flatten();
        event.result.set(Ok(()));
        Ok(())
    }
    fn effect_bind_globals_tied(&mut self, event: GlobalRuntime<'_>) -> Result<(), ()> {
        let q = event.queries.borrow();
        self.token_embedding = q[0].ok().flatten();
        self.output_norm = q[1].ok().flatten();
        self.output = self.token_embedding;
        event.result.set(Ok(()));
        Ok(())
    }
    fn effect_global_dependency_error(&mut self, event: GlobalRuntime<'_>) -> Result<(), ()> {
        event.result.set(Err(Error::Dependency));
        Ok(())
    }

    fn guard_attention_plain(&self, event: &AttentionRuntime<'_>) -> Result<bool, ()> {
        Ok(self.attention_request_base(event)
            && !event.event.requires_qk_norm()
            && !event.event.uses_shared_value())
    }
    fn guard_attention_qk(&self, event: &AttentionRuntime<'_>) -> Result<bool, ()> {
        Ok(self.attention_request_base(event)
            && event.event.requires_qk_norm()
            && !event.event.uses_shared_value())
    }
    fn guard_attention_shared(&self, event: &AttentionRuntime<'_>) -> Result<bool, ()> {
        Ok(self.attention_request_base(event)
            && !event.event.requires_qk_norm()
            && event.event.uses_shared_value())
    }
    fn guard_attention_qk_shared(&self, event: &AttentionRuntime<'_>) -> Result<bool, ()> {
        Ok(self.attention_request_base(event)
            && event.event.requires_qk_norm()
            && event.event.uses_shared_value())
    }
    fn guard_attention_invalid(&self, event: &AttentionRuntime<'_>) -> Result<bool, ()> {
        Ok(!self.guard_attention_plain(event)?
            && !self.guard_attention_qk(event)?
            && !self.guard_attention_shared(event)?
            && !self.guard_attention_qk_shared(event)?)
    }
    fn effect_query_attention_plain(&mut self, event: AttentionRuntime<'_>) -> Result<(), ()> {
        self.query_attention_dedicated(event);
        Ok(())
    }
    fn effect_query_attention_qk(&mut self, event: AttentionRuntime<'_>) -> Result<(), ()> {
        self.query_attention_dedicated(event);
        self.query_attention_qk(event);
        Ok(())
    }
    fn effect_query_attention_shared(&mut self, event: AttentionRuntime<'_>) -> Result<(), ()> {
        self.query_attention_shared(event);
        Ok(())
    }
    fn effect_query_attention_qk_shared(&mut self, event: AttentionRuntime<'_>) -> Result<(), ()> {
        self.query_attention_shared(event);
        self.query_attention_qk(event);
        Ok(())
    }
    fn guard_block_queries_valid(&self, event: &AttentionRuntime<'_>) -> Result<bool, ()> {
        let required = if event.event.requires_qk_norm() {
            if event.event.uses_shared_value() {
                &ATTN_QK_SHARED[..]
            } else {
                &ATTN_QK[..]
            }
        } else if event.event.uses_shared_value() {
            &ATTN_SHARED[..]
        } else {
            &ATTN_PLAIN[..]
        };
        Ok(required_queries_valid(&*event.queries.borrow(), required))
    }
    fn guard_block_queries_missing(&self, event: &AttentionRuntime<'_>) -> Result<bool, ()> {
        let required = if event.event.requires_qk_norm() {
            if event.event.uses_shared_value() {
                &ATTN_QK_SHARED[..]
            } else {
                &ATTN_QK[..]
            }
        } else if event.event.uses_shared_value() {
            &ATTN_SHARED[..]
        } else {
            &ATTN_PLAIN[..]
        };
        Ok(required_queries_missing(&*event.queries.borrow(), required))
    }
    fn guard_block_queries_dependency_error(
        &self,
        event: &AttentionRuntime<'_>,
    ) -> Result<bool, ()> {
        Ok(queries_dependency_error(&*event.queries.borrow()))
    }
    fn effect_bind_attention_plain(&mut self, event: AttentionRuntime<'_>) -> Result<(), ()> {
        let q = event.queries.borrow();
        self.bind_attention_selected(
            event,
            view(q[SLOT_ATTENTION_V]),
            TensorView::default(),
            TensorView::default(),
        );
        Ok(())
    }
    fn effect_bind_attention_qk(&mut self, event: AttentionRuntime<'_>) -> Result<(), ()> {
        let q = event.queries.borrow();
        self.bind_attention_selected(
            event,
            view(q[SLOT_ATTENTION_V]),
            view(q[SLOT_ATTENTION_Q_NORM]),
            view(q[SLOT_ATTENTION_K_NORM]),
        );
        Ok(())
    }
    fn effect_bind_attention_shared(&mut self, event: AttentionRuntime<'_>) -> Result<(), ()> {
        let q = event.queries.borrow();
        self.bind_attention_selected(
            event,
            view(q[SLOT_ATTENTION_K]),
            TensorView::default(),
            TensorView::default(),
        );
        Ok(())
    }
    fn effect_bind_attention_qk_shared(&mut self, event: AttentionRuntime<'_>) -> Result<(), ()> {
        let q = event.queries.borrow();
        self.bind_attention_selected(
            event,
            view(q[SLOT_ATTENTION_K]),
            view(q[SLOT_ATTENTION_Q_NORM]),
            view(q[SLOT_ATTENTION_K_NORM]),
        );
        Ok(())
    }
    fn effect_attention_dependency_error(&mut self, event: AttentionRuntime<'_>) -> Result<(), ()> {
        event.result.set(Err(Error::Dependency));
        Ok(())
    }

    fn guard_shortconv_valid(&self, event: &ShortconvRuntime<'_>) -> Result<bool, ()> {
        Ok(self.index_valid(event.event.index)
            && super::event::shortconv_request_valid(event.event)
            && !self.block(event.event.index).bound)
    }
    fn guard_shortconv_invalid(&self, event: &ShortconvRuntime<'_>) -> Result<bool, ()> {
        Ok(!self.guard_shortconv_valid(event)?)
    }
    fn effect_query_shortconv(&mut self, event: ShortconvRuntime<'_>) -> Result<(), ()> {
        let mut q = event.queries.borrow_mut();
        let i = event.event.index;
        query_slot(self, &mut q, SLOT_ATTENTION_NORM, i, b"attn_norm.weight");
        query_slot(
            self,
            &mut q,
            SLOT_SHORTCONV_CONV,
            i,
            b"shortconv.conv.weight",
        );
        query_slot(
            self,
            &mut q,
            SLOT_SHORTCONV_IN,
            i,
            b"shortconv.in_proj.weight",
        );
        query_slot(
            self,
            &mut q,
            SLOT_SHORTCONV_OUT,
            i,
            b"shortconv.out_proj.weight",
        );
        query_slot(self, &mut q, SLOT_FFN_NORM, i, b"ffn_norm.weight");
        query_slot(self, &mut q, SLOT_FFN_GATE, i, b"ffn_gate.weight");
        query_slot(self, &mut q, SLOT_FFN_DOWN, i, b"ffn_down.weight");
        query_slot(self, &mut q, SLOT_FFN_UP, i, b"ffn_up.weight");
        Ok(())
    }
    fn guard_shortconv_queries_valid(&self, event: &ShortconvRuntime<'_>) -> Result<bool, ()> {
        Ok(required_queries_valid(&*event.queries.borrow(), &SHORTCONV))
    }
    fn guard_shortconv_queries_missing(&self, event: &ShortconvRuntime<'_>) -> Result<bool, ()> {
        Ok(required_queries_missing(
            &*event.queries.borrow(),
            &SHORTCONV,
        ))
    }
    fn guard_shortconv_queries_dependency_error(
        &self,
        event: &ShortconvRuntime<'_>,
    ) -> Result<bool, ()> {
        Ok(queries_dependency_error(&*event.queries.borrow()))
    }
    fn effect_bind_shortconv(&mut self, event: ShortconvRuntime<'_>) -> Result<(), ()> {
        let q = event.queries.borrow();
        let index = usize::try_from(event.event.index).expect("guarded");
        let slot = &mut self.storage_mut().regions.blocks[index];
        *slot = BlockSlot {
            index: event.event.index,
            uses_attention: false,
            attention_norm: view(q[SLOT_ATTENTION_NORM]),
            shortconv_conv: view(q[SLOT_SHORTCONV_CONV]),
            shortconv_in_proj: view(q[SLOT_SHORTCONV_IN]),
            shortconv_out_proj: view(q[SLOT_SHORTCONV_OUT]),
            feed_forward_norm: view(q[SLOT_FFN_NORM]),
            feed_forward_gate: view(q[SLOT_FFN_GATE]),
            feed_forward_down: view(q[SLOT_FFN_DOWN]),
            feed_forward_up: view(q[SLOT_FFN_UP]),
            bound: true,
            ..BlockSlot::default()
        };
        self.storage_mut().regions.layers[index] = event.event.layer;
        self.store_block_view(index, event.event.layer);
        self.bound_blocks += 1;
        event.result.set(Ok(()));
        Ok(())
    }

    fn guard_reject_shortconv_valid(&self, event: &RejectRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.family == event::BlockFamily::Shortconv
            && self.index_valid(event.event.index)
            && self.block(event.event.index).bound
            && self.block(event.event.index).uses_attention)
    }
    fn guard_reject_attention_valid(&self, event: &RejectRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.family == event::BlockFamily::Attention
            && self.index_valid(event.event.index)
            && self.block(event.event.index).bound
            && !self.block(event.event.index).uses_attention)
    }
    fn guard_reject_invalid(&self, event: &RejectRuntime<'_>) -> Result<bool, ()> {
        Ok(!self.guard_reject_shortconv_valid(event)?
            && !self.guard_reject_attention_valid(event)?)
    }
    fn effect_query_reject_shortconv(&mut self, event: RejectRuntime<'_>) -> Result<(), ()> {
        let mut queries = event.queries.borrow_mut();
        let index = event.event.index;
        query_reject_slot(self, &mut queries, 0, index, b"shortconv.conv.weight");
        query_reject_slot(self, &mut queries, 1, index, b"shortconv.in_proj.weight");
        query_reject_slot(self, &mut queries, 2, index, b"shortconv.out_proj.weight");
        Ok(())
    }
    fn effect_query_reject_attention(&mut self, event: RejectRuntime<'_>) -> Result<(), ()> {
        let mut queries = event.queries.borrow_mut();
        let index = event.event.index;
        query_reject_slot(self, &mut queries, 0, index, b"attn_q.weight");
        query_reject_slot(self, &mut queries, 1, index, b"attn_k.weight");
        query_reject_slot(self, &mut queries, 2, index, b"attn_v.weight");
        query_reject_slot(self, &mut queries, 3, index, b"attn_q_norm.weight");
        query_reject_slot(self, &mut queries, 4, index, b"attn_k_norm.weight");
        query_reject_slot(self, &mut queries, 5, index, b"attn_output.weight");
        Ok(())
    }
    fn guard_reject_queries_absent(&self, event: &RejectRuntime<'_>) -> Result<bool, ()> {
        Ok(event
            .queries
            .borrow()
            .iter()
            .all(|query| matches!(query, Ok(None))))
    }
    fn guard_reject_queries_present(&self, event: &RejectRuntime<'_>) -> Result<bool, ()> {
        Ok(event
            .queries
            .borrow()
            .iter()
            .any(|query| matches!(query, Ok(Some(_)))))
    }
    fn guard_reject_queries_dependency_error(&self, event: &RejectRuntime<'_>) -> Result<bool, ()> {
        Ok(event.queries.borrow().iter().any(Result::is_err))
    }
    fn effect_reject_ok(&mut self, event: RejectRuntime<'_>) -> Result<(), ()> {
        event.result.set(Ok(()));
        Ok(())
    }
    fn effect_reject_present(&mut self, event: RejectRuntime<'_>) -> Result<(), ()> {
        event.result.set(Err(Error::ModelInvalid));
        Ok(())
    }
    fn effect_reject_dependency_error(&mut self, event: RejectRuntime<'_>) -> Result<(), ()> {
        event.result.set(Err(Error::Dependency));
        Ok(())
    }
    fn effect_invalid_reject(&mut self, event: RejectRuntime<'_>) -> Result<(), ()> {
        event.result.set(Err(Error::InvalidRequest));
        Ok(())
    }
    fn effect_shortconv_dependency_error(&mut self, event: ShortconvRuntime<'_>) -> Result<(), ()> {
        event.result.set(Err(Error::Dependency));
        Ok(())
    }

    fn guard_topology_valid(&self, event: &TopologyRuntime<'_>) -> Result<bool, ()> {
        Ok(self.topology.is_none()
            && self.bound_blocks == self.block_count
            && event.event.node_count > 0
            && event.event.tensor_count > 0)
    }
    fn guard_topology_invalid(&self, event: &TopologyRuntime<'_>) -> Result<bool, ()> {
        Ok(!self.guard_topology_valid(event)?)
    }
    fn effect_topology(&mut self, event: TopologyRuntime<'_>) -> Result<(), ()> {
        self.topology = Some(TopologyDescriptor {
            node_count: event.event.node_count,
            tensor_count: event.event.tensor_count,
            bytes_per_tensor: event.event.bytes_per_tensor,
            workspace_capacity_bytes: event.event.workspace_capacity_bytes,
        });
        event.result.set(Ok(()));
        Ok(())
    }
    fn guard_plan_valid(&self, _event: &PlanRuntime<'_>) -> Result<bool, ()> {
        Ok(self.topology.is_some() && self.prefill.is_none())
    }
    fn guard_plan_invalid(&self, event: &PlanRuntime<'_>) -> Result<bool, ()> {
        Ok(!self.guard_plan_valid(event)?)
    }
    fn effect_plan(&mut self, event: PlanRuntime<'_>) -> Result<(), ()> {
        let graph = self.topology.expect("guarded");
        self.prefill = Some(StepPlan {
            kind: StepKind::Prefill,
            node_count: graph.node_count(),
            tensor_count: graph.tensor_count(),
            expected_outputs: 1,
            max_step_tokens: self.context_length,
        });
        self.decode = Some(StepPlan {
            kind: StepKind::Decode,
            node_count: graph.node_count(),
            tensor_count: graph.tensor_count(),
            expected_outputs: 1,
            max_step_tokens: 1,
        });
        event.result.set(Ok(()));
        Ok(())
    }

    fn guard_validate_attention_valid(&self, event: &ValidateRuntime<'_>) -> Result<bool, ()> {
        Ok(self.validate_base(event)
            && self.block(event.event.index).uses_attention
            && attention_required(
                self.block(event.event.index),
                self.storage().regions.layers[usize::try_from(event.event.index).expect("guarded")],
            ))
    }
    fn guard_validate_shortconv_valid(&self, event: &ValidateRuntime<'_>) -> Result<bool, ()> {
        Ok(self.validate_base(event)
            && !self.block(event.event.index).uses_attention
            && shortconv_required(self.block(event.event.index)))
    }
    fn guard_validate_invalid(&self, event: &ValidateRuntime<'_>) -> Result<bool, ()> {
        Ok(!self.guard_validate_attention_valid(event)?
            && !self.guard_validate_shortconv_valid(event)?)
    }
    fn effect_validate(&mut self, event: ValidateRuntime<'_>) -> Result<(), ()> {
        let index = usize::try_from(event.event.index).expect("guarded");
        self.storage_mut().regions.blocks[index].validated = true;
        self.validated_blocks += 1;
        event.result.set(Ok(()));
        Ok(())
    }

    fn guard_audit_attention_plain(&self, event: &AuditRuntime<'_>) -> Result<bool, ()> {
        Ok(self.audit_base(event)
            && self.block(event.event.index).uses_attention
            && self.storage().regions.layers[usize::try_from(event.event.index).expect("guarded")]
                .qk_norm_route()
                == AttentionQkNormRoute::None)
    }
    fn guard_audit_attention_qk(&self, event: &AuditRuntime<'_>) -> Result<bool, ()> {
        Ok(self.audit_base(event)
            && self.block(event.event.index).uses_attention
            && self.storage().regions.layers[usize::try_from(event.event.index).expect("guarded")]
                .qk_norm_route()
                != AttentionQkNormRoute::None)
    }
    fn guard_audit_shortconv(&self, event: &AuditRuntime<'_>) -> Result<bool, ()> {
        Ok(self.audit_base(event) && !self.block(event.event.index).uses_attention)
    }
    fn guard_audit_invalid(&self, event: &AuditRuntime<'_>) -> Result<bool, ()> {
        Ok(!self.guard_audit_attention_plain(event)?
            && !self.guard_audit_attention_qk(event)?
            && !self.guard_audit_shortconv(event)?)
    }
    fn effect_query_audit_attention_plain(&mut self, event: AuditRuntime<'_>) -> Result<(), ()> {
        self.query_audit_common(event);
        self.query_audit_attention(event);
        Ok(())
    }
    fn effect_query_audit_attention_qk(&mut self, event: AuditRuntime<'_>) -> Result<(), ()> {
        self.query_audit_common(event);
        self.query_audit_attention(event);
        self.query_audit_qk(event);
        Ok(())
    }
    fn effect_query_audit_shortconv(&mut self, event: AuditRuntime<'_>) -> Result<(), ()> {
        self.query_audit_common(event);
        Ok(())
    }
    fn guard_audit_dependency_valid(&self, event: &AuditRuntime<'_>) -> Result<bool, ()> {
        Ok(event
            .observations
            .borrow()
            .iter()
            .flatten()
            .all(|fact| fact.outcome.is_ok()))
    }
    fn guard_audit_dependency_error(&self, event: &AuditRuntime<'_>) -> Result<bool, ()> {
        Ok(!self.guard_audit_dependency_valid(event)?)
    }
    fn effect_record_audit(&mut self, event: AuditRuntime<'_>) -> Result<(), ()> {
        let index = usize::try_from(event.event.index).expect("guarded");
        self.storage_mut().regions.block_audits[index] = super::storage::BlockAuditSlot {
            audited: true,
            observations: *event.observations.borrow(),
        };
        self.audited_blocks += 1;
        event.result.set(Ok(()));
        Ok(())
    }
    fn effect_audit_dependency_error(&mut self, event: AuditRuntime<'_>) -> Result<(), ()> {
        event.result.set(Err(Error::Dependency));
        Ok(())
    }

    fn guard_stage_global_vector(&self, event: &StageRuntime<'_>) -> Result<bool, ()> {
        Ok(self.stage_base(event)
            && matches!(
                event.event.family,
                QuantizedStageFamily::TokenEmbedding | QuantizedStageFamily::OutputNorm
            ))
    }
    fn guard_stage_global_matrix(&self, event: &StageRuntime<'_>) -> Result<bool, ()> {
        Ok(self.stage_base(event) && event.event.family == QuantizedStageFamily::Output)
    }
    fn guard_stage_block(&self, event: &StageRuntime<'_>) -> Result<bool, ()> {
        Ok(self.stage_base(event)
            && !matches!(
                event.event.family,
                QuantizedStageFamily::TokenEmbedding
                    | QuantizedStageFamily::OutputNorm
                    | QuantizedStageFamily::Output
            ))
    }
    fn guard_stage_request_invalid(&self, event: &StageRuntime<'_>) -> Result<bool, ()> {
        Ok(!self.guard_stage_global_vector(event)?
            && !self.guard_stage_global_matrix(event)?
            && !self.guard_stage_block(event)?)
    }
    fn effect_query_global_stage_vector(&mut self, event: StageRuntime<'_>) -> Result<(), ()> {
        let tensor = self.global_tensor(event.event.family).expect("guarded");
        event.outcome.set(Some(
            self.query_capability(Scope::VectorDequantContract, tensor),
        ));
        Ok(())
    }
    fn effect_query_global_stage_matrix(&mut self, event: StageRuntime<'_>) -> Result<(), ()> {
        let tensor = self.global_tensor(event.event.family).expect("guarded");
        event.outcome.set(Some(
            self.query_capability(Scope::MatrixWeightContract, tensor),
        ));
        Ok(())
    }
    fn effect_no_stage_query(&mut self, _event: StageRuntime<'_>) -> Result<(), ()> {
        Ok(())
    }
    fn guard_stage_global_vector_dense(&self, event: &StageRuntime<'_>) -> Result<bool, ()> {
        Ok(self.stage_is_global_vector(event)
            && event.outcome.get().is_some_and(|outcome| {
                matches!(
                    outcome,
                    Ok(Outcome::NativeQuantized | Outcome::ApprovedDenseF32ByContract)
                )
            }))
    }
    fn guard_stage_global_vector_no_claim(&self, event: &StageRuntime<'_>) -> Result<bool, ()> {
        Ok(self.stage_is_global_vector(event)
            && event.outcome.get() == Some(Ok(Outcome::ExplicitNoClaim)))
    }
    fn guard_stage_global_matrix_native(&self, event: &StageRuntime<'_>) -> Result<bool, ()> {
        Ok(self.stage_is_global_matrix(event)
            && event.outcome.get() == Some(Ok(Outcome::NativeQuantized)))
    }
    fn guard_stage_global_matrix_disallowed(&self, event: &StageRuntime<'_>) -> Result<bool, ()> {
        Ok(self.stage_is_global_matrix(event)
            && event.outcome.get() == Some(Ok(Outcome::DisallowedFallback)))
    }
    fn guard_stage_global_matrix_no_claim(&self, event: &StageRuntime<'_>) -> Result<bool, ()> {
        Ok(self.stage_is_global_matrix(event)
            && event.outcome.get() == Some(Ok(Outcome::ExplicitNoClaim)))
    }
    fn guard_stage_block_native(&self, event: &StageRuntime<'_>) -> Result<bool, ()> {
        Ok(self.stage_is_block(event)
            && !is_vector_stage(event.event.family)
            && self.stage_has_observations(event.event.family)
            && self.stage_all_native(event.event.family))
    }
    fn guard_stage_block_dense(&self, event: &StageRuntime<'_>) -> Result<bool, ()> {
        Ok(self.stage_is_block(event)
            && is_vector_stage(event.event.family)
            && self.stage_has_observations(event.event.family)
            && self.stage_all_vector_approved(event.event.family))
    }
    fn guard_stage_block_disallowed(&self, event: &StageRuntime<'_>) -> Result<bool, ()> {
        Ok(self.stage_is_block(event)
            && !is_vector_stage(event.event.family)
            && self.stage_has_observations(event.event.family)
            && !self.stage_all_native(event.event.family)
            && self.stage_any_disallowed(event.event.family))
    }
    fn guard_stage_block_no_claim(&self, event: &StageRuntime<'_>) -> Result<bool, ()> {
        Ok(self.stage_is_block(event)
            && self.stage_has_observations(event.event.family)
            && ((is_vector_stage(event.event.family)
                && !self.stage_all_vector_approved(event.event.family))
                || (!is_vector_stage(event.event.family)
                    && !self.stage_all_native(event.event.family)
                    && !self.stage_any_disallowed(event.event.family))))
    }
    fn guard_stage_block_not_applicable(&self, event: &StageRuntime<'_>) -> Result<bool, ()> {
        Ok(self.stage_is_block(event) && !self.stage_has_observations(event.event.family))
    }
    fn guard_stage_block_consistent(&self, event: &StageRuntime<'_>) -> Result<bool, ()> {
        Ok(self.stage_consistent(event.event.family))
    }
    fn guard_stage_block_inconsistent(&self, event: &StageRuntime<'_>) -> Result<bool, ()> {
        Ok(!self.stage_consistent(event.event.family))
    }
    fn guard_stage_dependency_error(&self, event: &StageRuntime<'_>) -> Result<bool, ()> {
        Ok(event.outcome.get().is_some_and(|outcome| outcome.is_err()))
    }
    fn effect_stage_global_dense(&mut self, event: StageRuntime<'_>) -> Result<(), ()> {
        self.finalize_global_stage(event, QuantizedContractKind::ApprovedDenseF32ByContract);
        Ok(())
    }
    fn effect_stage_global_no_claim(&mut self, event: StageRuntime<'_>) -> Result<(), ()> {
        self.finalize_global_stage(event, QuantizedContractKind::ExplicitNoClaim);
        Ok(())
    }
    fn effect_stage_global_native(&mut self, event: StageRuntime<'_>) -> Result<(), ()> {
        self.finalize_global_stage(event, QuantizedContractKind::NativeQuantized);
        Ok(())
    }
    fn effect_stage_global_disallowed(&mut self, event: StageRuntime<'_>) -> Result<(), ()> {
        self.finalize_global_stage(event, QuantizedContractKind::DisallowedFallback);
        Ok(())
    }
    fn effect_stage_block_native_consistent(&mut self, event: StageRuntime<'_>) -> Result<(), ()> {
        self.finalize_block_stage(event, QuantizedContractKind::NativeQuantized, true);
        Ok(())
    }
    fn effect_stage_block_native_inconsistent(
        &mut self,
        event: StageRuntime<'_>,
    ) -> Result<(), ()> {
        self.finalize_block_stage(event, QuantizedContractKind::NativeQuantized, false);
        Ok(())
    }
    fn effect_stage_block_dense_consistent(&mut self, event: StageRuntime<'_>) -> Result<(), ()> {
        self.finalize_block_stage(
            event,
            QuantizedContractKind::ApprovedDenseF32ByContract,
            true,
        );
        Ok(())
    }
    fn effect_stage_block_dense_inconsistent(&mut self, event: StageRuntime<'_>) -> Result<(), ()> {
        self.finalize_block_stage(
            event,
            QuantizedContractKind::ApprovedDenseF32ByContract,
            false,
        );
        Ok(())
    }
    fn effect_stage_block_disallowed_consistent(
        &mut self,
        event: StageRuntime<'_>,
    ) -> Result<(), ()> {
        self.finalize_block_stage(event, QuantizedContractKind::DisallowedFallback, true);
        Ok(())
    }
    fn effect_stage_block_disallowed_inconsistent(
        &mut self,
        event: StageRuntime<'_>,
    ) -> Result<(), ()> {
        self.finalize_block_stage(event, QuantizedContractKind::DisallowedFallback, false);
        Ok(())
    }
    fn effect_stage_block_no_claim_consistent(
        &mut self,
        event: StageRuntime<'_>,
    ) -> Result<(), ()> {
        self.finalize_block_stage(event, QuantizedContractKind::ExplicitNoClaim, true);
        Ok(())
    }
    fn effect_stage_block_no_claim_inconsistent(
        &mut self,
        event: StageRuntime<'_>,
    ) -> Result<(), ()> {
        self.finalize_block_stage(event, QuantizedContractKind::ExplicitNoClaim, false);
        Ok(())
    }
    fn effect_stage_block_not_applicable(&mut self, event: StageRuntime<'_>) -> Result<(), ()> {
        self.publish_stage(event, QuantizedContractKind::NotApplicable, None, false);
        Ok(())
    }
    fn effect_stage_dependency_error(&mut self, event: StageRuntime<'_>) -> Result<(), ()> {
        event.result.set(Err(Error::Dependency));
        Ok(())
    }

    fn guard_contract_complete(&self, _event: &VisitRuntime<'_>) -> Result<bool, ()> {
        Ok(self.prefill.is_some()
            && self.decode.is_some()
            && self.validated_blocks == self.block_count
            && self.audited_blocks == self.block_count
            && self.finalized_stages == 14)
    }
    fn guard_contract_incomplete(&self, event: &VisitRuntime<'_>) -> Result<bool, ()> {
        Ok(!self.guard_contract_complete(event)?)
    }
    fn effect_visit(&mut self, event: VisitRuntime<'_>) -> Result<(), ()> {
        event.result.set(Ok(ContractDescriptor {
            model: self.model.expect("complete"),
            token_embedding: self.token_embedding.expect("complete").tensor_id(),
            output_norm: self.output_norm.expect("complete").tensor_id(),
            output: self.output.expect("complete").tensor_id(),
            block_count: self.block_count,
            topology: self.topology.expect("complete"),
            prefill: self.prefill.expect("complete"),
            decode: self.decode.expect("complete"),
            audit: self.audits,
        }));
        Ok(())
    }
    fn guard_block_visit_valid(&self, event: &BlockVisitRuntime<'_>) -> Result<bool, ()> {
        Ok(self.index_valid(event.event.index) && self.block(event.event.index).validated)
    }
    fn guard_block_visit_invalid(&self, event: &BlockVisitRuntime<'_>) -> Result<bool, ()> {
        Ok(!self.guard_block_visit_valid(event)?)
    }
    fn effect_block_visit(&mut self, event: BlockVisitRuntime<'_>) -> Result<(), ()> {
        let index = usize::try_from(event.event.index).expect("guarded");
        *event.result.borrow_mut() =
            Ok(self.storage().regions.views[index].expect("bound block view"));
        Ok(())
    }

    fn effect_storage_bind(&mut self, event: StorageBindRuntime<'_>) -> Result<(), ()> {
        self.storage = event.storage.take();
        event.result.set(Ok(()));
        Ok(())
    }
    fn effect_storage_bind_busy(&mut self, event: StorageBindRuntime<'_>) -> Result<(), ()> {
        event.result.set(Err(Error::Busy));
        Ok(())
    }
    fn effect_reset(&mut self, event: ResetRuntime<'_>) -> Result<(), ()> {
        self.reset_contract();
        event.result.set(Ok(()));
        Ok(())
    }
    fn effect_release(&mut self, event: StorageReleaseRuntime<'_>) -> Result<(), ()> {
        *event.storage.borrow_mut() = self.storage.take();
        event.result.set(Ok(()));
        Ok(())
    }
    fn effect_invalid_begin(&mut self, event: BeginRuntime<'_>) -> Result<(), ()> {
        set_unit(event, Error::InvalidRequest)
    }
    fn effect_invalid_global(&mut self, event: GlobalRuntime<'_>) -> Result<(), ()> {
        set_unit(event, Error::InvalidRequest)
    }
    fn effect_invalid_attention(&mut self, event: AttentionRuntime<'_>) -> Result<(), ()> {
        set_unit(event, Error::InvalidRequest)
    }
    fn effect_invalid_shortconv(&mut self, event: ShortconvRuntime<'_>) -> Result<(), ()> {
        set_unit(event, Error::InvalidRequest)
    }
    fn effect_invalid_topology(&mut self, event: TopologyRuntime<'_>) -> Result<(), ()> {
        set_unit(event, Error::InvalidRequest)
    }
    fn effect_invalid_plan(&mut self, event: PlanRuntime<'_>) -> Result<(), ()> {
        set_unit(event, Error::InvalidRequest)
    }
    fn effect_invalid_audit(&mut self, event: AuditRuntime<'_>) -> Result<(), ()> {
        set_unit(event, Error::InvalidRequest)
    }
    fn effect_invalid_stage(&mut self, event: StageRuntime<'_>) -> Result<(), ()> {
        set_unit(event, Error::InvalidRequest)
    }
    fn effect_invalid_visit(&mut self, event: VisitRuntime<'_>) -> Result<(), ()> {
        event.result.set(Err(Error::InvalidRequest));
        Ok(())
    }
    fn effect_invalid_block_visit(&mut self, event: BlockVisitRuntime<'_>) -> Result<(), ()> {
        *event.result.borrow_mut() = Err(Error::InvalidRequest);
        Ok(())
    }
    fn effect_model_invalid_global(&mut self, event: GlobalRuntime<'_>) -> Result<(), ()> {
        set_unit(event, Error::ModelInvalid)
    }
    fn effect_model_invalid_attention(&mut self, event: AttentionRuntime<'_>) -> Result<(), ()> {
        set_unit(event, Error::ModelInvalid)
    }
    fn effect_model_invalid_shortconv(&mut self, event: ShortconvRuntime<'_>) -> Result<(), ()> {
        set_unit(event, Error::ModelInvalid)
    }
    fn effect_model_invalid_validate(&mut self, event: ValidateRuntime<'_>) -> Result<(), ()> {
        set_unit(event, Error::ModelInvalid)
    }
    fn effect_busy_begin(&mut self, event: BeginRuntime<'_>) -> Result<(), ()> {
        set_unit(event, Error::Busy)
    }
    fn effect_busy_release(&mut self, event: StorageReleaseRuntime<'_>) -> Result<(), ()> {
        set_unit(event, Error::Busy)
    }
    fn effect_storage_unavailable_begin(&mut self, event: BeginRuntime<'_>) -> Result<(), ()> {
        set_unit(event, Error::StorageUnavailable)
    }
    fn effect_storage_unavailable_reset(&mut self, event: ResetRuntime<'_>) -> Result<(), ()> {
        set_unit(event, Error::StorageUnavailable)
    }
    fn effect_storage_unavailable_release(
        &mut self,
        event: StorageReleaseRuntime<'_>,
    ) -> Result<(), ()> {
        set_unit(event, Error::StorageUnavailable)
    }
    fn guard_never(&self, _event: &UnexpectedRuntime) -> Result<bool, ()> {
        Ok(false)
    }
    fn effect_unexpected(&mut self) -> Result<(), ()> {
        Ok(())
    }
}

trait UnitResult {
    fn result(&self) -> &Cell<Result<(), Error>>;
}
macro_rules! unit_result { ($($ty:ident),+ $(,)?) => { $(impl UnitResult for $ty<'_> { fn result(&self) -> &Cell<Result<(), Error>> { self.result } })+ }; }
unit_result!(
    BeginRuntime,
    GlobalRuntime,
    AttentionRuntime,
    ShortconvRuntime,
    TopologyRuntime,
    PlanRuntime,
    ValidateRuntime,
    AuditRuntime,
    StageRuntime,
    ResetRuntime,
    StorageReleaseRuntime
);

#[allow(clippy::needless_pass_by_value, clippy::unnecessary_wraps)]
fn set_unit<T: UnitResult>(event: T, error: Error) -> Result<(), ()> {
    event.result().set(Err(error));
    Ok(())
}

impl Context {
    fn attention_request_base(&self, event: &AttentionRuntime<'_>) -> bool {
        self.index_valid(event.event.index)
            && super::event::attention_request_valid(event.event)
            && !self.block(event.event.index).bound
    }
    fn query_attention_common(&mut self, event: AttentionRuntime<'_>) {
        let mut q = event.queries.borrow_mut();
        let i = event.event.index;
        query_slot(self, &mut q, SLOT_ATTENTION_NORM, i, b"attn_norm.weight");
        query_slot(self, &mut q, SLOT_ATTENTION_Q, i, b"attn_q.weight");
        query_slot(self, &mut q, SLOT_ATTENTION_K, i, b"attn_k.weight");
        query_slot(
            self,
            &mut q,
            SLOT_ATTENTION_OUTPUT,
            i,
            b"attn_output.weight",
        );
        query_slot(self, &mut q, SLOT_FFN_NORM, i, b"ffn_norm.weight");
        query_slot(self, &mut q, SLOT_FFN_GATE, i, b"ffn_gate.weight");
        query_slot(self, &mut q, SLOT_FFN_DOWN, i, b"ffn_down.weight");
        query_slot(self, &mut q, SLOT_FFN_UP, i, b"ffn_up.weight");
    }
    fn query_attention_dedicated(&mut self, event: AttentionRuntime<'_>) {
        self.query_attention_common(event);
        let mut q = event.queries.borrow_mut();
        query_slot(
            self,
            &mut q,
            SLOT_ATTENTION_V,
            event.event.index,
            b"attn_v.weight",
        );
    }
    fn query_attention_shared(&mut self, event: AttentionRuntime<'_>) {
        self.query_attention_common(event);
    }
    fn query_attention_qk(&mut self, event: AttentionRuntime<'_>) {
        let mut q = event.queries.borrow_mut();
        let i = event.event.index;
        query_slot(
            self,
            &mut q,
            SLOT_ATTENTION_Q_NORM,
            i,
            b"attn_q_norm.weight",
        );
        query_slot(
            self,
            &mut q,
            SLOT_ATTENTION_K_NORM,
            i,
            b"attn_k_norm.weight",
        );
    }
    fn bind_attention_selected(
        &mut self,
        event: AttentionRuntime<'_>,
        value: TensorView,
        q_norm: TensorView,
        k_norm: TensorView,
    ) {
        let q = event.queries.borrow();
        let index = usize::try_from(event.event.index).expect("guarded");
        let key = view(q[SLOT_ATTENTION_K]);
        let slot = &mut self.storage_mut().regions.blocks[index];
        *slot = BlockSlot {
            index: event.event.index,
            uses_attention: true,
            attention_norm: view(q[SLOT_ATTENTION_NORM]),
            attention_q: view(q[SLOT_ATTENTION_Q]),
            attention_k: key,
            attention_v: value,
            attention_q_norm: q_norm,
            attention_k_norm: k_norm,
            attention_output: view(q[SLOT_ATTENTION_OUTPUT]),
            feed_forward_norm: view(q[SLOT_FFN_NORM]),
            feed_forward_gate: view(q[SLOT_FFN_GATE]),
            feed_forward_down: view(q[SLOT_FFN_DOWN]),
            feed_forward_up: view(q[SLOT_FFN_UP]),
            bound: true,
            ..BlockSlot::default()
        };
        self.storage_mut().regions.layers[index] = event.event.layer;
        self.bound_blocks += 1;
        event.result.set(Ok(()));
        self.store_block_view(index, event.event.layer);
    }
    fn validate_base(&self, event: &ValidateRuntime<'_>) -> bool {
        self.prefill.is_some()
            && self.index_valid(event.event.index)
            && self.block(event.event.index).bound
            && !self.block(event.event.index).validated
    }
    fn audit_base(&self, event: &AuditRuntime<'_>) -> bool {
        self.validated_blocks == self.block_count
            && self.index_valid(event.event.index)
            && self.block(event.event.index).validated
            && !self.storage().regions.block_audits
                [usize::try_from(event.event.index).expect("guarded")]
            .audited
    }
    fn query_audit_common(&mut self, event: AuditRuntime<'_>) {
        let block = *self.block(event.event.index);
        let mut observations = event.observations.borrow_mut();
        audit_fact(
            self,
            &mut observations,
            QuantizedStageFamily::AttentionNorm,
            block.attention_norm.tensor,
            Scope::VectorDequantContract,
        );
        audit_fact(
            self,
            &mut observations,
            QuantizedStageFamily::FeedForwardNorm,
            block.feed_forward_norm.tensor,
            Scope::VectorDequantContract,
        );
        audit_fact(
            self,
            &mut observations,
            QuantizedStageFamily::FeedForwardGate,
            block.feed_forward_gate.tensor,
            Scope::MatrixWeightContract,
        );
        audit_fact(
            self,
            &mut observations,
            QuantizedStageFamily::FeedForwardDown,
            block.feed_forward_down.tensor,
            Scope::MatrixWeightContract,
        );
        audit_fact(
            self,
            &mut observations,
            QuantizedStageFamily::FeedForwardUp,
            block.feed_forward_up.tensor,
            Scope::MatrixWeightContract,
        );
    }
    fn query_audit_attention(&mut self, event: AuditRuntime<'_>) {
        let block = *self.block(event.event.index);
        let mut observations = event.observations.borrow_mut();
        audit_fact(
            self,
            &mut observations,
            QuantizedStageFamily::AttentionQ,
            block.attention_q.tensor,
            Scope::MatrixWeightContract,
        );
        audit_fact(
            self,
            &mut observations,
            QuantizedStageFamily::AttentionK,
            block.attention_k.tensor,
            Scope::MatrixWeightContract,
        );
        audit_fact(
            self,
            &mut observations,
            QuantizedStageFamily::AttentionV,
            block.attention_v.tensor,
            Scope::MatrixWeightContract,
        );
        audit_fact(
            self,
            &mut observations,
            QuantizedStageFamily::AttentionOutput,
            block.attention_output.tensor,
            Scope::MatrixWeightContract,
        );
    }
    fn query_audit_qk(&mut self, event: AuditRuntime<'_>) {
        let block = *self.block(event.event.index);
        let mut observations = event.observations.borrow_mut();
        audit_fact(
            self,
            &mut observations,
            QuantizedStageFamily::AttentionQNorm,
            block.attention_q_norm.tensor,
            Scope::VectorDequantContract,
        );
        audit_fact(
            self,
            &mut observations,
            QuantizedStageFamily::AttentionKNorm,
            block.attention_k_norm.tensor,
            Scope::VectorDequantContract,
        );
    }
    const fn stage_base(&self, event: &StageRuntime<'_>) -> bool {
        self.audited_blocks == self.block_count
            && !self.finalized[event.event.family.index()]
            && self.global_tensor_exists_or_block(event.event.family)
    }
    const fn global_tensor_exists_or_block(&self, family: QuantizedStageFamily) -> bool {
        match family {
            QuantizedStageFamily::TokenEmbedding => self.token_embedding.is_some(),
            QuantizedStageFamily::OutputNorm => self.output_norm.is_some(),
            QuantizedStageFamily::Output => self.output.is_some(),
            _ => true,
        }
    }
    const fn global_tensor(&self, family: QuantizedStageFamily) -> Option<TensorDescriptor> {
        match family {
            QuantizedStageFamily::TokenEmbedding => self.token_embedding,
            QuantizedStageFamily::OutputNorm => self.output_norm,
            QuantizedStageFamily::Output => self.output,
            _ => None,
        }
    }
    #[allow(clippy::missing_const_for_fn, clippy::unused_self)]
    fn stage_is_global_vector(&self, event: &StageRuntime<'_>) -> bool {
        matches!(
            event.event.family,
            QuantizedStageFamily::TokenEmbedding | QuantizedStageFamily::OutputNorm
        )
    }
    #[allow(clippy::missing_const_for_fn, clippy::unused_self)]
    fn stage_is_global_matrix(&self, event: &StageRuntime<'_>) -> bool {
        event.event.family == QuantizedStageFamily::Output
    }
    fn stage_is_block(&self, event: &StageRuntime<'_>) -> bool {
        !self.stage_is_global_vector(event) && !self.stage_is_global_matrix(event)
    }
    fn stage_has_observations(&self, family: QuantizedStageFamily) -> bool {
        self.block_observations(family).next().is_some()
    }
    fn stage_all_native(&self, family: QuantizedStageFamily) -> bool {
        self.block_observations(family)
            .all(|observation| observation.outcome == Ok(Outcome::NativeQuantized))
    }
    fn stage_all_vector_approved(&self, family: QuantizedStageFamily) -> bool {
        self.block_observations(family).all(|observation| {
            matches!(
                observation.outcome,
                Ok(Outcome::NativeQuantized | Outcome::ApprovedDenseF32ByContract)
            )
        })
    }
    fn stage_any_disallowed(&self, family: QuantizedStageFamily) -> bool {
        self.block_observations(family)
            .any(|observation| observation.outcome == Ok(Outcome::DisallowedFallback))
    }
    fn stage_consistent(&self, family: QuantizedStageFamily) -> bool {
        let Some(first) = self.block_observations(family).next() else {
            return false;
        };
        self.block_observations(family)
            .all(|observation| observation.tensor_type == first.tensor_type)
    }
    fn block_observations(
        &self,
        family: QuantizedStageFamily,
    ) -> impl Iterator<Item = AuditObservation> + '_ {
        self.storage().regions.block_audits
            [..usize::try_from(self.block_count).expect("bounded block count")]
            .iter()
            .filter_map(move |block| block.observations[family.index()])
    }
    fn finalize_global_stage(&mut self, event: StageRuntime<'_>, kind: QuantizedContractKind) {
        let tensor = self
            .global_tensor(event.event.family)
            .expect("global stage");
        self.publish_stage(event, kind, Some(tensor.tensor_type()), true);
    }
    fn finalize_block_stage(
        &mut self,
        event: StageRuntime<'_>,
        kind: QuantizedContractKind,
        consistent: bool,
    ) {
        let tensor_type = self
            .block_observations(event.event.family)
            .next()
            .map(|observation| observation.tensor_type);
        self.publish_stage(event, kind, tensor_type, consistent);
    }
    fn publish_stage(
        &mut self,
        event: StageRuntime<'_>,
        kind: QuantizedContractKind,
        tensor_type: Option<SerializedType>,
        consistent: bool,
    ) {
        let index = event.event.family.index();
        self.audits[index] = StageAudit {
            family: event.event.family,
            tensor_type,
            contract: kind,
            consistent_across_layers: consistent,
        };
        self.finalized[index] = true;
        self.finalized_stages += 1;
        event.result.set(Ok(()));
    }
    fn store_block_view(&mut self, index: usize, layer: super::LayerExecution) {
        let descriptor = block_descriptor(&self.storage().regions.blocks[index], layer);
        self.storage_mut().regions.views[index] = Some(descriptor);
    }
}

fn query_slot(
    context: &mut Context,
    queries: &mut [CatalogQuery; BLOCK_QUERY_COUNT],
    slot: usize,
    index: i32,
    suffix: &[u8],
) {
    queries[slot] = context.query_block(index, suffix);
}
fn query_reject_slot(
    context: &mut Context,
    queries: &mut [CatalogQuery; REJECT_QUERY_COUNT],
    slot: usize,
    index: i32,
    suffix: &[u8],
) {
    queries[slot] = context.query_block(index, suffix);
}
fn view(query: CatalogQuery) -> TensorView {
    TensorView {
        tensor: query.ok().flatten(),
    }
}
fn attention_required(block: &BlockSlot, layer: super::LayerExecution) -> bool {
    COMMON
        .iter()
        .all(|slot| block_tensor(block, *slot).is_some())
        && block.attention_q.tensor.is_some()
        && block.attention_k.tensor.is_some()
        && (layer.value_route() == super::AttentionValueRoute::SharedKeyValue
            || block.attention_v.tensor.is_some())
        && block.attention_output.tensor.is_some()
        && (layer.qk_norm_route() == AttentionQkNormRoute::None
            || (block.attention_q_norm.tensor.is_some() && block.attention_k_norm.tensor.is_some()))
}
fn shortconv_required(block: &BlockSlot) -> bool {
    COMMON
        .iter()
        .all(|slot| block_tensor(block, *slot).is_some())
        && block.shortconv_conv.tensor.is_some()
        && block.shortconv_in_proj.tensor.is_some()
        && block.shortconv_out_proj.tensor.is_some()
}
const fn block_tensor(block: &BlockSlot, slot: usize) -> Option<TensorDescriptor> {
    match slot {
        SLOT_ATTENTION_NORM => block.attention_norm.tensor,
        SLOT_FFN_NORM => block.feed_forward_norm.tensor,
        SLOT_FFN_GATE => block.feed_forward_gate.tensor,
        SLOT_FFN_DOWN => block.feed_forward_down.tensor,
        SLOT_FFN_UP => block.feed_forward_up.tensor,
        _ => None,
    }
}
fn audit_fact(
    context: &mut Context,
    observations: &mut [Option<AuditObservation>; QUANTIZED_STAGE_FAMILY_COUNT],
    family: QuantizedStageFamily,
    tensor: Option<TensorDescriptor>,
    scope: Scope,
) {
    let tensor = tensor.expect("validated block tensor");
    observations[family.index()] = Some(AuditObservation {
        tensor_type: tensor.tensor_type(),
        outcome: context.query_capability(scope, tensor),
    });
}
const fn is_vector_stage(family: QuantizedStageFamily) -> bool {
    matches!(
        family,
        QuantizedStageFamily::TokenEmbedding
            | QuantizedStageFamily::OutputNorm
            | QuantizedStageFamily::AttentionNorm
            | QuantizedStageFamily::AttentionQNorm
            | QuantizedStageFamily::AttentionKNorm
            | QuantizedStageFamily::FeedForwardNorm
    )
}
fn block_descriptor(block: &BlockSlot, layer: super::LayerExecution) -> BlockDescriptor {
    let tensors = [
        block.attention_norm.tensor,
        block.attention_q.tensor,
        block.attention_k.tensor,
        block.attention_v.tensor,
        block.attention_q_norm.tensor,
        block.attention_k_norm.tensor,
        block.attention_output.tensor,
        block.shortconv_conv.tensor,
        block.shortconv_in_proj.tensor,
        block.shortconv_out_proj.tensor,
        block.feed_forward_norm.tensor,
        block.feed_forward_gate.tensor,
        block.feed_forward_down.tensor,
        block.feed_forward_up.tensor,
    ];
    let fallback = block
        .attention_norm
        .tensor
        .expect("validated block has attention norm")
        .tensor_id();
    let mut ids = [fallback; 14];
    let mut mask = 0u16;
    for (index, tensor) in tensors.into_iter().enumerate() {
        if let Some(tensor) = tensor {
            ids[index] = tensor.tensor_id();
            mask |= 1u16 << index;
        }
    }
    BlockDescriptor::new(block.index, block.uses_attention, layer, ids, mask)
}
