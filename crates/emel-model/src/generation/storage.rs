use crate::generation::quantized_path::Outcome;
use emel_tensor::dtype::SerializedType;

use crate::catalog::event::TensorDescriptor;

use super::LayerExecution;
use super::event::BlockDescriptor;

#[derive(Clone, Copy, Debug, Default)]
pub(super) struct TensorView {
    pub(super) tensor: Option<TensorDescriptor>,
}

#[derive(Clone, Copy, Debug, Default)]
pub(super) struct BlockSlot {
    pub(super) index: i32,
    pub(super) uses_attention: bool,
    pub(super) attention_norm: TensorView,
    pub(super) attention_q: TensorView,
    pub(super) attention_k: TensorView,
    pub(super) attention_v: TensorView,
    pub(super) attention_q_norm: TensorView,
    pub(super) attention_k_norm: TensorView,
    pub(super) attention_output: TensorView,
    pub(super) shortconv_conv: TensorView,
    pub(super) shortconv_in_proj: TensorView,
    pub(super) shortconv_out_proj: TensorView,
    pub(super) feed_forward_norm: TensorView,
    pub(super) feed_forward_gate: TensorView,
    pub(super) feed_forward_down: TensorView,
    pub(super) feed_forward_up: TensorView,
    pub(super) bound: bool,
    pub(super) validated: bool,
}

#[derive(Clone, Copy, Debug)]
pub(super) struct AuditObservation {
    pub(super) tensor_type: SerializedType,
    pub(super) outcome: Result<Outcome, crate::generation::quantized_path::Error>,
}

#[derive(Clone, Copy, Debug, Default)]
pub(super) struct BlockAuditSlot {
    pub(super) audited: bool,
    pub(super) observations: [Option<AuditObservation>; super::QUANTIZED_STAGE_FAMILY_COUNT],
}

#[derive(Debug)]
pub(super) struct StorageRegions {
    pub(super) blocks: Vec<BlockSlot>,
    pub(super) views: Vec<Option<BlockDescriptor>>,
    pub(super) layers: Vec<LayerExecution>,
    pub(super) block_audits: Vec<BlockAuditSlot>,
}

impl StorageRegions {
    pub(super) fn clear(&mut self) {
        self.blocks.fill(BlockSlot::default());
        self.views.fill(None);
        self.layers.fill(LayerExecution::default());
        self.block_audits.fill(BlockAuditSlot::default());
    }
}
