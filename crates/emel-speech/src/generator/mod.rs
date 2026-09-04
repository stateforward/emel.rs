//! Module for `generator` state machines.
pub(crate) mod sm;
/// Public actor and typed request/result contracts; generated machine details remain private.
pub use sm::{
    ConditionPhase, ConditionResult, ConditionRun, EventReset, FlushRun, FramePhase, FrameResult,
    GenerateRun, InitPhase, InitRun, SpeechGeneratorDuplexModelActor,
    SpeechGeneratorDuplexModelContext, SpeechGeneratorDuplexModelStates, SpeechGeneratorError,
    SpeechGeneratorSynthesisModelActor, SpeechGeneratorSynthesisModelContext,
    SpeechGeneratorSynthesisModelStates, StreamRun,
};
