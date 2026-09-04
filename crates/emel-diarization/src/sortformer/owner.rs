//! Maintained public owner boundary for the Sortformer family.
//!
//! The owner validates the source-fixed metadata descriptor and delegates
//! caller-owned bounded requests to the existing request, pipeline, and
//! executor state machines. Numeric model work remains explicit through the
//! synchronous callbacks on those machines; this facade makes no numerical
//! parity claim and never schedules deferred work.

use emel_model::catalog::event::ModelIdentity;
use emel_model::sortformer::{self, ContractDescriptor, Parameters};

use super::{executor, pipeline, request};

/// Lifecycle of the single-writer Sortformer owner actor.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum State {
    /// No validated metadata contract is installed.
    Uninitialized,
    /// Metadata validation completed successfully.
    Validated,
    /// The last owner event failed.
    Errored,
}

/// Typed failures from the maintained Sortformer owner boundary.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum Error {
    /// No error has been recorded since construction or reset.
    #[default]
    None,
    /// The event is not valid for the current owner lifecycle state.
    InvalidRequest,
    /// A second metadata validation was attempted without resetting first.
    Busy,
    /// The architecture or source-fixed parameters do not match Sortformer.
    ModelInvalid,
    /// The emel-model metadata contract failed validation.
    Metadata(sortformer::ContractError),
    /// The delegated request machine rejected its input or callback.
    Request(request::Error),
    /// The delegated pipeline machine rejected its input or callback.
    Pipeline(pipeline::PipelineError),
    /// The delegated executor machine rejected its input or callback.
    Executor(executor::Error),
    /// An event was delivered outside its generated/owned run-to-completion path.
    UnexpectedEvent,
}

/// Validated source-fixed Sortformer metadata exposed by the owner.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MetadataContract {
    parameters: Parameters,
    descriptor: ContractDescriptor,
}

impl MetadataContract {
    #[must_use]
    pub const fn parameters(self) -> Parameters {
        self.parameters
    }

    #[must_use]
    pub const fn model(self) -> ModelIdentity {
        self.descriptor.model()
    }

    #[must_use]
    pub const fn prep_tensor_count(self) -> u32 {
        self.descriptor.prep().count()
    }

    #[must_use]
    pub const fn encoder_tensor_count(self) -> u32 {
        self.descriptor.encoder().count()
    }

    #[must_use]
    pub const fn modules_tensor_count(self) -> u32 {
        self.descriptor.modules().count()
    }

    #[must_use]
    pub const fn output_tensor_count(self) -> u32 {
        self.descriptor.output().count()
    }

    #[must_use]
    pub const fn descriptor(self) -> ContractDescriptor {
        self.descriptor
    }
}

/// Typed successful request completion.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PrepareDone {
    pub frame_count: i32,
    pub feature_bin_count: i32,
}

/// Typed successful pipeline completion.
#[allow(clippy::struct_field_names)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RunDone {
    pub frame_count: i32,
    pub probability_count: i32,
    pub segment_count: i32,
}

/// Typed successful executor completion.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ExecuteDone {
    pub frame_count: i32,
    pub hidden_dim: i32,
}

/// Typed owner error event payload.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ErrorEvent {
    pub error: Error,
}

/// Typed events accepted by [`Sortformer`].
pub mod event {
    use super::{Error, MetadataContract, Sortformer, State};
    use crate::sortformer::{executor, pipeline, request};
    use emel_model::sortformer::{ContractDescriptor, Parameters};

    mod sealed {
        pub trait Sealed {}
    }

    /// Public owner event with a run-to-completion typed outcome.
    pub trait Event: sealed::Sealed {
        type Output;
        #[doc(hidden)]
        fn dispatch(self, actor: &mut Sortformer) -> Self::Output;
    }

    /// Installs a validated metadata contract produced by `emel-model`.
    #[derive(Clone, Copy, Debug)]
    pub struct ValidateContract<'a> {
        pub architecture: &'a [u8],
        pub parameters: Parameters,
        pub descriptor: ContractDescriptor,
    }

    impl<'a> ValidateContract<'a> {
        #[must_use]
        pub const fn new(
            architecture: &'a [u8],
            parameters: Parameters,
            descriptor: ContractDescriptor,
        ) -> Self {
            Self {
                architecture,
                parameters,
                descriptor,
            }
        }
    }

    /// Visits the installed metadata contract without changing ownership.
    #[derive(Clone, Copy, Debug, Default)]
    pub struct Visit;

    impl Visit {
        #[must_use]
        pub const fn new() -> Self {
            Self
        }
    }

    /// Restores the owner to its initial lifecycle state.
    #[derive(Clone, Copy, Debug, Default)]
    pub struct Reset;

    impl Reset {
        #[must_use]
        pub const fn new() -> Self {
            Self
        }
    }

    /// Delegates one request preparation event to the maintained request machine.
    #[allow(missing_debug_implementations)]
    pub struct Prepare<'a>(pub request::EventPrepareRun<'a>);

    impl<'a> Prepare<'a> {
        #[must_use]
        pub const fn new(event: request::EventPrepareRun<'a>) -> Self {
            Self(event)
        }
    }

    /// Delegates one complete pipeline event to the maintained pipeline machine.
    #[allow(missing_debug_implementations)]
    pub struct Run<'a>(pub pipeline::EventRunFlow<'a>);

    impl<'a> Run<'a> {
        #[must_use]
        pub const fn new(event: pipeline::EventRunFlow<'a>) -> Self {
            Self(event)
        }
    }

    /// Delegates one executor event to the maintained executor machine.
    #[allow(missing_debug_implementations)]
    pub struct Execute<'a>(pub executor::EventExecuteRun<'a>);

    impl<'a> Execute<'a> {
        #[must_use]
        pub const fn new(event: executor::EventExecuteRun<'a>) -> Self {
            Self(event)
        }
    }

    /// Explicit unexpected-event input used to exercise owner recovery.
    #[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
    pub struct Unexpected;

    impl Unexpected {
        #[must_use]
        pub const fn new() -> Self {
            Self
        }
    }

    pub type ContractResult = Result<MetadataContract, Error>;
    pub type VisitResult = Result<MetadataContract, Error>;
    pub type ResetResult = Result<(), Error>;
    pub type PrepareResult = Result<(), Error>;
    pub type RunResult = Result<(), Error>;
    pub type ExecuteResult = Result<(), Error>;
    pub type UnexpectedResult = Result<(), Error>;

    impl sealed::Sealed for ValidateContract<'_> {}
    impl Event for ValidateContract<'_> {
        type Output = ContractResult;
        fn dispatch(self, actor: &mut Sortformer) -> Self::Output {
            actor.validate_contract(self)
        }
    }
    impl sealed::Sealed for Visit {}
    impl Event for Visit {
        type Output = VisitResult;
        fn dispatch(self, actor: &mut Sortformer) -> Self::Output {
            actor.visit()
        }
    }
    impl sealed::Sealed for Reset {}
    impl Event for Reset {
        type Output = ResetResult;
        fn dispatch(self, actor: &mut Sortformer) -> Self::Output {
            actor.reset();
            Ok(())
        }
    }
    impl sealed::Sealed for Prepare<'_> {}
    impl Event for Prepare<'_> {
        type Output = PrepareResult;
        fn dispatch(self, actor: &mut Sortformer) -> Self::Output {
            actor.prepare(self.0)
        }
    }
    impl sealed::Sealed for Run<'_> {}
    impl Event for Run<'_> {
        type Output = RunResult;
        fn dispatch(self, actor: &mut Sortformer) -> Self::Output {
            actor.run(self.0)
        }
    }
    impl sealed::Sealed for Execute<'_> {}
    impl Event for Execute<'_> {
        type Output = ExecuteResult;
        fn dispatch(self, actor: &mut Sortformer) -> Self::Output {
            actor.execute(self.0)
        }
    }
    impl sealed::Sealed for Unexpected {}
    impl Event for Unexpected {
        type Output = UnexpectedResult;
        fn dispatch(self, actor: &mut Sortformer) -> Self::Output {
            actor.process_unexpected_event()
        }
    }

    #[allow(dead_code)]
    const fn _state_type_is_public(_: State) {}
}

/// Public single-writer Sortformer owner actor.
#[allow(missing_debug_implementations)]
pub struct Sortformer {
    contract: Option<MetadataContract>,
    state: State,
    error: Error,
    unexpected_count: u64,
    request: Box<request::Request>,
    pipeline: Box<pipeline::DiarizationSortformerPipeline>,
    executor: Box<executor::Executor>,
}

impl Default for Sortformer {
    fn default() -> Self {
        Self::new()
    }
}

impl Sortformer {
    #[must_use]
    pub fn new() -> Self {
        Self {
            contract: None,
            state: State::Uninitialized,
            error: Error::None,
            unexpected_count: 0,
            request: Box::new(request::Request::new(
                request::DiarizationSortformerRequestContext::default(),
            )),
            pipeline: Box::new(pipeline::DiarizationSortformerPipeline::new()),
            executor: Box::new(executor::Executor::new()),
        }
    }

    /// Dispatches one typed public event synchronously and to completion.
    pub fn process_event<E: event::Event>(&mut self, event: E) -> E::Output {
        event.dispatch(self)
    }

    #[allow(clippy::large_types_passed_by_value)]
    fn validate_contract(&mut self, event: event::ValidateContract<'_>) -> event::ContractResult {
        if self.state == State::Validated {
            return self.fail(Error::Busy);
        }
        if !sortformer::is_execution_architecture(event.architecture)
            || !event.parameters.counts_valid()
            || event.parameters.sample_rate != pipeline::SAMPLE_RATE
            || event.parameters.speaker_count != pipeline::SPEAKER_COUNT
            || event.parameters.chunk_len != pipeline::CHUNK_LEN
            || event.parameters.chunk_right_context != request::CHUNK_RIGHT_CONTEXT
            || event.parameters.fifo_len != 0
            || event.parameters.spkcache_update_period != pipeline::CHUNK_LEN
            || event.parameters.spkcache_len != pipeline::CHUNK_LEN
        {
            return self.fail(Error::ModelInvalid);
        }
        let summaries = [
            event.descriptor.prep(),
            event.descriptor.encoder(),
            event.descriptor.modules(),
            event.descriptor.output(),
        ];
        if summaries.iter().any(|summary| {
            summary.count() == 0 || !summary.all_bound() || !summary.all_geometry_valid()
        }) {
            return self.fail(Error::Metadata(sortformer::ContractError::ModelInvalid));
        }
        let contract = MetadataContract {
            parameters: event.parameters,
            descriptor: event.descriptor,
        };
        self.contract = Some(contract);
        self.state = State::Validated;
        self.error = Error::None;
        Ok(contract)
    }

    fn visit(&self) -> event::VisitResult {
        if self.state != State::Validated {
            return Err(Error::UnexpectedEvent);
        }
        self.contract.ok_or(Error::UnexpectedEvent)
    }

    #[allow(clippy::needless_pass_by_value)]
    fn prepare(&mut self, event: request::EventPrepareRun<'_>) -> event::PrepareResult {
        if self.state != State::Validated {
            return self.fail(Error::InvalidRequest);
        }
        if !self.request.prepare(&event) {
            return self.fail(Error::Request(self.request.error()));
        }
        self.error = Error::None;
        Ok(())
    }

    fn run(&mut self, mut event: pipeline::EventRunFlow<'_>) -> event::RunResult {
        if let Err(error) = self.pipeline.run(&mut event) {
            return self.fail(Error::Pipeline(error));
        }
        self.error = Error::None;
        Ok(())
    }

    #[allow(clippy::needless_pass_by_value)]
    fn execute(&mut self, event: executor::EventExecuteRun<'_>) -> event::ExecuteResult {
        if self.state != State::Validated {
            return self.fail(Error::InvalidRequest);
        }
        if !self.executor.execute(&event) {
            return self.fail(Error::Executor(self.executor.error()));
        }
        self.error = Error::None;
        Ok(())
    }

    const fn fail<T>(&mut self, error: Error) -> Result<T, Error> {
        self.state = State::Errored;
        self.error = error;
        Err(error)
    }

    /// Records an explicit unexpected event and returns its typed failure.
    ///
    /// # Errors
    ///
    /// Always returns [`Error::UnexpectedEvent`] and transitions the owner to
    /// [`State::Errored`].
    pub const fn process_unexpected_event(&mut self) -> event::UnexpectedResult {
        self.unexpected_count = self.unexpected_count.saturating_add(1);
        self.fail(Error::UnexpectedEvent)
    }

    /// Returns the owner lifecycle state.
    #[must_use]
    pub const fn state(&self) -> State {
        self.state
    }

    /// Returns the most recent owner error.
    #[must_use]
    pub const fn error(&self) -> Error {
        self.error
    }

    /// Returns how many explicit unexpected events have been observed.
    #[must_use]
    pub const fn unexpected_count(&self) -> u64 {
        self.unexpected_count
    }

    /// Returns the validated metadata contract, if installed.
    #[must_use]
    pub const fn contract(&self) -> Option<MetadataContract> {
        self.contract
    }

    /// Resets owner metadata and error state without allocating.
    pub const fn reset(&mut self) {
        self.contract = None;
        self.state = State::Uninitialized;
        self.error = Error::None;
    }
}

impl core::fmt::Display for Error {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::None => formatter.write_str("no sortformer owner error"),
            Self::InvalidRequest => formatter.write_str("invalid sortformer owner request"),
            Self::Busy => formatter.write_str("sortformer owner is already validated"),
            Self::ModelInvalid => formatter.write_str("sortformer owner metadata is invalid"),
            Self::Metadata(error) => write!(formatter, "sortformer metadata error: {error:?}"),
            Self::Request(error) => write!(formatter, "sortformer request error: {error:?}"),
            Self::Pipeline(error) => write!(formatter, "sortformer pipeline error: {error:?}"),
            Self::Executor(error) => write!(formatter, "sortformer executor error: {error:?}"),
            Self::UnexpectedEvent => formatter.write_str("unexpected sortformer owner event"),
        }
    }
}

impl std::error::Error for Error {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sortformer::request::{self, EventPrepareRun};

    #[test]
    fn owner_reports_unexpected_visit_until_metadata_is_validated() {
        let mut owner = Sortformer::new();
        assert_eq!(owner.state(), State::Uninitialized);
        assert_eq!(
            owner.process_event(event::Visit::new()),
            Err(Error::UnexpectedEvent)
        );
        assert_eq!(owner.state(), State::Uninitialized);
        assert_eq!(owner.error(), Error::None);
    }

    #[test]
    fn owner_rejects_request_before_validation_and_recovers_after_reset() {
        let mut owner = Sortformer::new();
        let contract = request::ExecutionContract::pinned();
        let pcm = vec![0.0_f32; pipeline::sm::REQUIRED_SAMPLE_COUNT];
        let mut features = vec![0.0_f32; pipeline::sm::REQUIRED_FEATURE_COUNT];
        let mut frame_count = 0;
        let mut feature_bins = 0;
        let request = EventPrepareRun::new(
            &contract,
            &pcm,
            request::SAMPLE_RATE,
            pipeline::sm::CHANNEL_COUNT,
            &mut features,
            &mut frame_count,
            &mut feature_bins,
        );
        assert_eq!(
            owner.process_event(event::Prepare::new(request)),
            Err(Error::InvalidRequest)
        );
        assert_eq!(owner.state(), State::Errored);
        assert_eq!(owner.process_event(event::Reset::new()), Ok(()));
        assert_eq!(owner.state(), State::Uninitialized);
        assert_eq!(owner.error(), Error::None);
        assert_eq!(
            owner.process_event(event::Visit::new()),
            Err(Error::UnexpectedEvent)
        );
    }

    #[test]
    fn unexpected_event_is_typed_counted_and_recoverable() {
        let mut owner = Sortformer::new();
        assert_eq!(
            owner.process_event(event::Unexpected::new()),
            Err(Error::UnexpectedEvent)
        );
        assert_eq!(owner.state(), State::Errored);
        assert_eq!(owner.error(), Error::UnexpectedEvent);
        assert_eq!(owner.unexpected_count(), 1);
        owner.reset();
        assert_eq!(owner.state(), State::Uninitialized);
        assert_eq!(owner.error(), Error::None);
        assert_eq!(owner.unexpected_count(), 1);
    }
    #[test]
    fn owner_rejects_pipeline_and_executor_before_validation() {
        let mut owner = Sortformer::new();
        let pipeline_contract = pipeline::sm::PipelineContract::pinned();
        let pcm = vec![0.0_f32; pipeline::sm::REQUIRED_SAMPLE_COUNT];
        let mut features = vec![0.0_f32; pipeline::sm::REQUIRED_FEATURE_COUNT];
        let mut encoder_frames = vec![0.0_f32; pipeline::sm::REQUIRED_ENCODER_VALUE_COUNT];
        let mut hidden = vec![0.0_f32; pipeline::sm::REQUIRED_HIDDEN_VALUE_COUNT];
        let mut probabilities = vec![0.0_f32; pipeline::sm::REQUIRED_PROBABILITY_VALUE_COUNT];
        let mut segments =
            vec![pipeline::sm::SegmentRecord::default(); pipeline::sm::MAX_SEGMENT_COUNT];
        let mut frame_count = 0;
        let mut probability_count = 0;
        let mut segment_count = 0;
        let mut pipeline_error = pipeline::PipelineError::None;
        let run = pipeline::EventRunFlow::new(
            &pipeline_contract,
            &pcm,
            pipeline::SAMPLE_RATE,
            pipeline::sm::CHANNEL_COUNT,
            &mut features,
            &mut encoder_frames,
            &mut hidden,
            &mut probabilities,
            &mut segments,
            &mut frame_count,
            &mut probability_count,
            &mut segment_count,
            &mut pipeline_error,
        );
        assert_eq!(
            owner.process_event(event::Run::new(run)),
            Err(Error::InvalidRequest)
        );
        assert_eq!(owner.state(), State::Errored);
        assert_eq!(owner.error(), Error::InvalidRequest);

        owner.reset();
        let executor_contract = executor::sm::ExecutionContract::pinned();
        let encoder_frames = vec![0.0_f32; executor::sm::REQUIRED_ENCODER_VALUE_COUNT];
        let mut hidden = vec![0.0_f32; executor::sm::REQUIRED_HIDDEN_VALUE_COUNT];
        let mut frame_count = 0;
        let mut hidden_dim = 0;
        let execute = executor::EventExecuteRun::new(
            &executor_contract,
            &encoder_frames,
            &mut hidden,
            &mut frame_count,
            &mut hidden_dim,
        );
        assert_eq!(
            owner.process_event(event::Execute::new(execute)),
            Err(Error::InvalidRequest)
        );
        assert_eq!(owner.state(), State::Errored);
        assert_eq!(owner.error(), Error::InvalidRequest);
    }

    #[test]
    fn owner_reset_is_allocation_free_and_idempotent() {
        let mut owner = Sortformer::new();
        owner.reset();
        owner.reset();
        assert_eq!(owner.state(), State::Uninitialized);
        assert!(owner.contract().is_none());
        assert_eq!(owner.error(), Error::None);
    }
}
