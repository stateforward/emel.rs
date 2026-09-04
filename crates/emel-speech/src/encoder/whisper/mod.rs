//! Module for `encoder/whisper` state machines.
mod detail;
pub(crate) mod sm;

pub use sm::{
    CHANNEL_COUNT, EncodeDone, EncodeError, EncoderError, EventEncodeRun, ExecutionContract,
    ModelAssets, SAMPLE_RATE, SpeechEncoderWhisperActor, SpeechEncoderWhisperStates, WeightVariant,
    encoder_frame_count, mel_frame_count, required_encoder_output_floats,
    required_workspace_floats,
};
