//! Module for `codec/mimi` state machines.
pub mod binding;
pub(crate) mod decoder;
pub(crate) mod encoder;
pub(crate) mod quantizer;
pub(crate) mod sm;
/// Public decoder-stage contracts; implementation state-machine details remain private.
pub use decoder::sm::{CodecStreamingState, DecoderBackendStage, DecoderTransformerStage};
/// Public encoder runtime and caller-owned streaming/callback contracts.
pub use encoder::sm::{
    EncoderRuntime as MimiEncoderRuntime, EncoderStage as MimiEncoderStage,
    EncoderStreamingState as MimiEncoderStreamingState,
};
/// Public quantizer runtime and native model-owned binding.
pub use quantizer::sm::{NativeF32Binding, QuantizerRuntime};
/// Public top-level Mimi actor and typed request/event contracts.
pub use sm::{
    DecodeRun, EncodeRun, EventCaptureDiagnostics, EventDecodeRun, EventEncodeRun, EventInitRun,
    EventResetStreamRun, InitRun, Mimi, MimiDecoderBackendStage, MimiDecoderStreamingState,
    MimiDecoderTransformerStage, MimiDiagnostics, MimiError, MimiEvent, MimiQuantizerRuntime,
    SpeechCodecMimi,
};
