//! Module for `tokenizer/preprocessor` state machines.
pub mod bpe;
pub mod fallback;
pub mod plamo2;
pub mod rwkv;
pub mod spm;
pub mod ugm;
pub mod wpm;
/// Maintained UGM actor, re-exported at the preprocessor boundary.
pub use ugm::sm::TextTokenizerPreprocessorUgmActor;
/// Concise UGM actor alias matching the maintained child naming convention.
pub type UgmPreprocessor = TextTokenizerPreprocessorUgmActor;
/// Concise WPM actor alias at the preprocessor boundary.
pub type WpmPreprocessor<'a> = wpm::sm::TextTokenizerPreprocessorWpmActor<'a>;
