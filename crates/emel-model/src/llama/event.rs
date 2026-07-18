//! Public Llama-family events and source-exact parameter contract.

use core::fmt;

use crate::attention_family;
use crate::generation::{self, ContractDescriptor};

use super::Llama;

pub use attention_family::event::{
    BlockAudit, BlockBuild, BlockValidation, BlockVisit, ContractReset, ContractVisit, PlanBuild,
    StageAudit, Storage, StorageBindError, StorageRelease, TopologyBuild,
};
pub use generation::event::Error;

mod sealed {
    pub trait Sealed {}
}

/// Event accepted by [`Llama`].
pub trait Event: sealed::Sealed {
    type Output;
    #[doc(hidden)]
    fn dispatch(self, actor: &mut Llama) -> Self::Output;
}

/// The exact eighteen metadata keys consumed by pinned Llama `load_hparams`.
pub const HPARAM_KEYS: [&str; 18] = [
    "llama.context_length",
    "llama.embedding_length",
    "llama.embedding_length_out",
    "llama.feed_forward_length",
    "llama.attention.head_count",
    "llama.attention.head_count_kv",
    "llama.rope.dimension_count",
    "llama.block_count",
    "llama.vocab_size",
    "llama.attention.layer_norm_epsilon",
    "llama.attention.layer_norm_rms_epsilon",
    "llama.attention.clamp_kqv",
    "llama.attn_logit_softcapping",
    "llama.final_logit_softcapping",
    "llama.residual_scale",
    "llama.embedding_scale",
    "llama.rope.freq_base",
    "llama.rope.freq_base_swa",
];

/// Source-exact Llama hyperparameters plus the two model-derived attention lengths.
///
/// Allocation and metadata decoding happen before dispatch. The actor retains this
/// owned `Copy` value because its layer facts are used across later block events.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Parameters {
    pub context_length: i32,
    pub embedding_length: i32,
    pub embedding_length_out: i32,
    pub feed_forward_length: i32,
    pub attention_head_count: i32,
    pub attention_head_count_kv: i32,
    pub rope_dimension_count: i32,
    pub block_count: i32,
    pub vocab_size: i32,
    pub attention_layer_norm_epsilon: f32,
    pub attention_layer_norm_rms_epsilon: f32,
    pub attention_clamp_kqv: f32,
    pub attn_logit_softcapping: f32,
    pub final_logit_softcapping: f32,
    pub residual_scale: f32,
    pub embedding_scale: f32,
    pub rope_freq_base: f32,
    pub rope_freq_base_swa: f32,
    pub attention_key_length: i32,
    pub attention_value_length: i32,
}

/// Begins the source-fixed Llama contract and binds its three global tensors.
pub type ContractBegin<'a> = attention_family::event::ContractBegin<'a, Parameters>;

macro_rules! impl_event {
    ($event:ty, $output:ty, $method:ident) => {
        impl sealed::Sealed for $event {}
        impl Event for $event {
            type Output = $output;
            #[inline]
            fn dispatch(self, actor: &mut Llama) -> Self::Output {
                actor.$method(self)
            }
        }
    };
}

impl_event!(ContractBegin<'_>, Result<(), generation::event::Error>, contract_begin);
impl_event!(BlockBuild, Result<(), generation::event::Error>, block_build);
impl_event!(TopologyBuild, Result<(), generation::event::Error>, topology_build);
impl_event!(PlanBuild, Result<(), generation::event::Error>, plan_build);
impl_event!(BlockValidation, Result<(), generation::event::Error>, block_validation);
impl_event!(BlockAudit, Result<(), generation::event::Error>, block_audit);
impl_event!(StageAudit, Result<(), generation::event::Error>, stage_audit);
impl_event!(ContractVisit, Result<ContractDescriptor, generation::event::Error>, contract_visit);
impl_event!(BlockVisit, Result<generation::event::BlockDescriptor, generation::event::Error>, block_visit);
impl_event!(ContractReset, Result<(), generation::event::Error>, contract_reset);
impl_event!(StorageRelease, Result<Storage, generation::event::Error>, storage_release);

impl fmt::Display for Parameters {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "Llama {} layers, embedding {}, context {}",
            self.block_count, self.embedding_length, self.context_length
        )
    }
}
