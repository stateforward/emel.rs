//! Public Qwen3-family events and source-exact parameter contract.

use core::fmt;

use crate::attention_family;
use crate::generation::{self, ContractDescriptor};

use super::Qwen3;

pub use attention_family::event::{
    BlockAudit, BlockBuild, BlockValidation, BlockVisit, ContractReset, ContractVisit, PlanBuild,
    StageAudit, Storage, StorageBindError, StorageRelease, TopologyBuild,
};
pub use generation::event::Error;

mod sealed {
    pub trait Sealed {}
}

/// Event accepted by [`Qwen3`].
pub trait Event: sealed::Sealed {
    type Output;
    #[doc(hidden)]
    fn dispatch(self, actor: &mut Qwen3) -> Self::Output;
}

/// The ten metadata keys inspected by pinned Qwen3 `load_hparams`.
pub const HPARAM_KEYS: [&str; 10] = [
    "qwen3.context_length",
    "qwen3.embedding_length",
    "qwen3.feed_forward_length",
    "qwen3.attention.head_count",
    "qwen3.attention.head_count_kv",
    "qwen3.attention.key_length",
    "qwen3.attention.value_length",
    "qwen3.block_count",
    "qwen3.attention.layer_norm_rms_epsilon",
    "qwen3.rope.freq_base",
];

/// Source-exact Qwen3 hyperparameters and derived fixed-layout facts.
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
    pub attention_key_length: i32,
    pub attention_value_length: i32,
    pub rope_dimension_count: i32,
    pub block_count: i32,
    pub attention_layer_norm_rms_epsilon: f32,
    pub rope_freq_base: f32,
    pub tie_word_embeddings: bool,
    pub rope_pair_x0_stride: i32,
    pub rope_pair_x1_stride: i32,
    pub rope_pair_x1_offset: i32,
    pub rope_pair_x1_half_rot_offset: i32,
}

/// Begins the source-fixed Qwen3 contract and binds its three global tensors.
pub type ContractBegin<'a> = attention_family::event::ContractBegin<'a, Parameters>;

macro_rules! impl_event {
    ($event:ty, $output:ty, $method:ident) => {
        impl sealed::Sealed for $event {}
        impl Event for $event {
            type Output = $output;
            #[inline]
            fn dispatch(self, actor: &mut Qwen3) -> Self::Output {
                actor.$method(self)
            }
        }
    };
}

impl_event!(ContractBegin<'_>, Result<(), Error>, contract_begin);
impl_event!(BlockBuild, Result<(), Error>, block_build);
impl_event!(TopologyBuild, Result<(), Error>, topology_build);
impl_event!(PlanBuild, Result<(), Error>, plan_build);
impl_event!(BlockValidation, Result<(), Error>, block_validation);
impl_event!(BlockAudit, Result<(), Error>, block_audit);
impl_event!(StageAudit, Result<(), Error>, stage_audit);
impl_event!(ContractVisit, Result<ContractDescriptor, Error>, contract_visit);
impl_event!(BlockVisit, Result<generation::event::BlockDescriptor, Error>, block_visit);
impl_event!(ContractReset, Result<(), Error>, contract_reset);
impl_event!(StorageRelease, Result<Storage, Error>, storage_release);

impl fmt::Display for Parameters {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "Qwen3 {} layers, embedding {}, context {}",
            self.block_count, self.embedding_length, self.context_length
        )
    }
}
