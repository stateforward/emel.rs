//! Source-aligned bounded layer actors from the pinned layer state machines.
//!
//! The pinned implementation delegates the numerical data plane to layer
//! detail helpers. This port keeps that boundary explicit: requests borrow all
//! storage from the caller, the state machines own sequencing and route guards,
//! and the callback executes one bounded operation synchronously.

#![allow(
    clippy::enum_variant_names,
    clippy::derive_partial_eq_without_eq,
    clippy::module_name_repetitions,
    clippy::missing_errors_doc,
    clippy::must_use_candidate,
    clippy::return_self_not_must_use,
    clippy::empty_structs_with_brackets,
    clippy::missing_const_for_fn,
    clippy::too_many_arguments,
    clippy::elidable_lifetime_names,
    clippy::needless_lifetimes,
    dead_code,
    unused_imports,
    missing_docs
)]

use core::cell::{Cell, RefCell};
use sml::sml;

pub const MAX_DIMENSION: usize = 1 << 20;

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[repr(u8)]
pub enum LayerDtype {
    #[default]
    Unknown = 0,
    F32 = 1,
    F16 = 2,
    Bf16 = 3,
    Q8_0 = 4,
    #[allow(non_camel_case_types)]
    Q8_K = 5,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[repr(u8)]
pub enum ResidualRoute {
    #[default]
    Attention = 0,
    Shortconv = 1,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[repr(u8)]
pub enum AttentionQkNormRoute {
    #[default]
    None = 0,
    HeadwiseRms = 1,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[repr(u8)]
pub enum AttentionVNormRoute {
    #[default]
    None = 0,
    Rms = 1,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[repr(u8)]
pub enum WindowMode {
    #[default]
    Resident = 0,
    Streamed = 1,
}

/// Concrete numerical routes materialized by the pinned layer actions.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LayerOperation {
    PrepareScalar,
    Normalize,
    ScalarAttentionNoneNone,
    ScalarAttentionHeadwiseRmsNone,
    ScalarAttentionNoneRms,
    ScalarAttentionHeadwiseRmsRms,
    ScalarShortconv,
    ScalarFeedForward,
    Chunk4AttentionNoneNone,
    Chunk4AttentionHeadwiseRmsNone,
    Chunk4AttentionNoneRms,
    Chunk4AttentionHeadwiseRmsRms,
    Chunk4Shortconv,
    Chunk4FeedForward,
    Chunk8AttentionNoneNone,
    Chunk8AttentionHeadwiseRmsNone,
    Chunk8AttentionNoneRms,
    Chunk8AttentionHeadwiseRmsRms,
    Chunk8Shortconv,
    Chunk8FeedForward,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[repr(u8)]
pub enum LayerError {
    #[default]
    None = 0,
    InvalidRequest = 1,
    Kernel = 2,
    UnsupportedRoute = 3,
    Unexpected = 4,
}

/// Caller-owned numeric buffers. The callback must consume these synchronously
/// and must not retain or re-enter the enclosing layer actor.
#[derive(Clone, Copy, Debug)]
pub struct LayerBuffers<'a> {
    pub hidden: &'a RefCell<&'a mut [f32]>,
    pub norm: &'a RefCell<&'a mut [f32]>,
    pub scratch: &'a RefCell<&'a mut [f32]>,
}

/// Synchronous caller-owned numerical operation.
pub type LayerKernelCallback = for<'a> fn(LayerOperation, &LayerRequest<'a>) -> bool;

#[derive(Clone, Copy, Debug)]
pub struct LayerRequest<'a> {
    pub dtype: LayerDtype,
    pub hidden_rows: usize,
    pub hidden_cols: usize,
    pub q_dim: usize,
    pub kv_dim: usize,
    pub ffn_dim: usize,
    pub layer_index: i32,
    pub token_base: usize,
    pub residual: ResidualRoute,
    pub qk_norm: AttentionQkNormRoute,
    pub v_norm: AttentionVNormRoute,
    pub window_mode: WindowMode,
    pub buffers: Option<LayerBuffers<'a>>,
    pub callback: Option<LayerKernelCallback>,
}

impl Default for LayerRequest<'_> {
    fn default() -> Self {
        Self {
            dtype: LayerDtype::Unknown,
            hidden_rows: 0,
            hidden_cols: 0,
            q_dim: 0,
            kv_dim: 0,
            ffn_dim: 0,
            layer_index: -1,
            token_base: 0,
            residual: ResidualRoute::Attention,
            qk_norm: AttentionQkNormRoute::None,
            v_norm: AttentionVNormRoute::None,
            window_mode: WindowMode::Resident,
            buffers: None,
            callback: None,
        }
    }
}

impl<'a> LayerRequest<'a> {
    pub const MAX_DIMENSION: usize = MAX_DIMENSION;

    #[must_use]
    pub const fn new(
        dtype: LayerDtype,
        hidden_rows: usize,
        hidden_cols: usize,
        q_dim: usize,
        kv_dim: usize,
        ffn_dim: usize,
        layer_index: i32,
        token_base: usize,
    ) -> Self {
        Self {
            dtype,
            hidden_rows,
            hidden_cols,
            q_dim,
            kv_dim,
            ffn_dim,
            layer_index,
            token_base,
            residual: ResidualRoute::Attention,
            qk_norm: AttentionQkNormRoute::None,
            v_norm: AttentionVNormRoute::None,
            window_mode: WindowMode::Resident,
            buffers: None,
            callback: None,
        }
    }

    #[must_use]
    pub const fn with_buffers(mut self, buffers: LayerBuffers<'a>) -> Self {
        self.buffers = Some(buffers);
        self
    }
    #[must_use]
    pub const fn with_callback(mut self, callback: LayerKernelCallback) -> Self {
        self.callback = Some(callback);
        self
    }

    #[must_use]
    pub const fn common_valid(self) -> bool {
        !matches!(self.dtype, LayerDtype::Unknown)
            && self.hidden_rows != 0
            && self.hidden_cols != 0
            && self.q_dim != 0
            && self.kv_dim != 0
            && self.ffn_dim != 0
            && self.hidden_rows <= Self::MAX_DIMENSION
            && self.hidden_cols <= Self::MAX_DIMENSION
            && self.q_dim <= Self::MAX_DIMENSION
            && self.kv_dim <= Self::MAX_DIMENSION
            && self.ffn_dim <= Self::MAX_DIMENSION
            && self.token_base <= Self::MAX_DIMENSION
            && self.layer_index >= 0
    }

    #[must_use]
    pub const fn route_valid(self) -> bool {
        matches!(
            self.residual,
            ResidualRoute::Attention | ResidualRoute::Shortconv
        ) && matches!(
            self.qk_norm,
            AttentionQkNormRoute::None | AttentionQkNormRoute::HeadwiseRms
        ) && matches!(
            self.v_norm,
            AttentionVNormRoute::None | AttentionVNormRoute::Rms
        )
    }

    #[must_use]
    pub fn storage_valid(self, rows: usize) -> bool {
        let Some(buffers) = self.buffers else {
            return false;
        };
        let Some(hidden_elements) = rows.checked_mul(self.hidden_cols) else {
            return false;
        };
        let Some(scratch_elements) =
            rows.checked_mul(self.q_dim.max(self.kv_dim).max(self.ffn_dim))
        else {
            return false;
        };
        buffers.hidden.borrow().len() >= hidden_elements
            && buffers.norm.borrow().len() >= hidden_elements
            && buffers.scratch.borrow().len() >= scratch_elements
    }

    #[must_use]
    pub fn scalar_valid(self) -> bool {
        self.common_valid()
            && self.hidden_rows == 1
            && self.route_valid()
            && self.storage_valid(1)
            && self.callback.is_some()
    }
    #[must_use]
    pub fn chunk4_valid(self) -> bool {
        self.common_valid()
            && self.hidden_rows == 4
            && self.token_base <= Self::MAX_DIMENSION.saturating_sub(4)
            && matches!(self.dtype, LayerDtype::Q8_0 | LayerDtype::Q8_K)
            && matches!(self.window_mode, WindowMode::Resident)
            && self.route_valid()
            && self.storage_valid(4)
            && self.callback.is_some()
    }
    #[must_use]
    pub fn chunk8_valid(self) -> bool {
        self.common_valid()
            && self.hidden_rows == 8
            && self.token_base <= Self::MAX_DIMENSION.saturating_sub(8)
            && matches!(self.dtype, LayerDtype::Q8_K)
            && matches!(self.window_mode, WindowMode::Resident)
            && self.route_valid()
            && self.storage_valid(8)
            && self.callback.is_some()
    }
}

#[derive(Clone, Debug)]
pub struct EventScalarRun<'a> {
    pub request: LayerRequest<'a>,
    pub stream_ready: Cell<bool>,
    pub normalized_ok: Cell<bool>,
    pub residual_ok: Cell<bool>,
    pub feed_forward_ok: Cell<bool>,
    pub succeeded: Cell<bool>,
    pub failed: Cell<bool>,
    pub error: Cell<LayerError>,
}
impl Default for EventScalarRun<'_> {
    fn default() -> Self {
        Self::new(LayerRequest::default())
    }
}
impl<'a> EventScalarRun<'a> {
    #[must_use]
    pub fn new(request: LayerRequest<'a>) -> Self {
        Self {
            request,
            stream_ready: Cell::new(false),
            normalized_ok: Cell::new(false),
            residual_ok: Cell::new(false),
            feed_forward_ok: Cell::new(false),
            succeeded: Cell::new(false),
            failed: Cell::new(false),
            error: Cell::new(LayerError::None),
        }
    }
}

#[derive(Clone, Debug)]
pub struct EventChunk4Run<'a> {
    pub request: LayerRequest<'a>,
    pub normalized_ok: Cell<bool>,
    pub residual_ok: Cell<bool>,
    pub feed_forward_ok: Cell<bool>,
    pub succeeded: Cell<bool>,
    pub failed: Cell<bool>,
    pub error: Cell<LayerError>,
}
impl Default for EventChunk4Run<'_> {
    fn default() -> Self {
        Self::new(LayerRequest::default())
    }
}
impl<'a> EventChunk4Run<'a> {
    #[must_use]
    pub fn new(request: LayerRequest<'a>) -> Self {
        Self {
            request,
            normalized_ok: Cell::new(false),
            residual_ok: Cell::new(false),
            feed_forward_ok: Cell::new(false),
            succeeded: Cell::new(false),
            failed: Cell::new(false),
            error: Cell::new(LayerError::None),
        }
    }
}

#[derive(Clone, Debug)]
pub struct EventChunk8Run<'a> {
    pub request: LayerRequest<'a>,
    pub normalized_ok: Cell<bool>,
    pub residual_ok: Cell<bool>,
    pub feed_forward_ok: Cell<bool>,
    pub succeeded: Cell<bool>,
    pub failed: Cell<bool>,
    pub error: Cell<LayerError>,
}
impl Default for EventChunk8Run<'_> {
    fn default() -> Self {
        Self::new(LayerRequest::default())
    }
}
impl<'a> EventChunk8Run<'a> {
    #[must_use]
    pub fn new(request: LayerRequest<'a>) -> Self {
        Self {
            request,
            normalized_ok: Cell::new(false),
            residual_ok: Cell::new(false),
            feed_forward_ok: Cell::new(false),
            succeeded: Cell::new(false),
            failed: Cell::new(false),
            error: Cell::new(LayerError::None),
        }
    }
}

// --- machine TextGeneratorLayerScalarModel from emel.cpp/src/emel/text/generator/layer/sm.hpp ---
sml! {
    TextGeneratorLayerScalarModel<'event> {
        "state_input_ready"_s <= *"state_idle"_s + event<EventScalarRun<'event>> / effect_prepare_scalar_wmode,
        "state_normalized"_s <= "state_input_ready"_s + completion<EventScalarRun>(EventScalarRun<'event>) [guard_stream_ready] / effect_normalize_scalar,
        "state_idle"_s <= "state_input_ready"_s + completion<EventScalarRun>(EventScalarRun<'event>) [guard_stream_failed] / effect_mark_failed_from_state_input_ready,
        "state_residual_done"_s <= "state_normalized"_s + completion<EventScalarRun>(EventScalarRun<'event>) [guard_scalar_normalized_attention_route_event_attention_qk_norm_route_none_event_attention_v_norm_route_none] / effect_run_scalar_attention_mode_route_event_attention_qk_norm_route_none_event_attention_v_norm_route_none_lanes,
        "state_residual_done"_s <= "state_normalized"_s + completion<EventScalarRun>(EventScalarRun<'event>) [guard_scalar_normalized_attention_route_event_attention_qk_norm_route_headwise_rms_event_attention_v_norm_route_none] / effect_run_scalar_attention_mode_route_event_attention_qk_norm_route_headwise_rms_event_attention_v_norm_route_none_lanes,
        "state_residual_done"_s <= "state_normalized"_s + completion<EventScalarRun>(EventScalarRun<'event>) [guard_scalar_normalized_attention_route_event_attention_qk_norm_route_none_event_attention_v_norm_route_rms] / effect_run_scalar_attention_mode_route_event_attention_qk_norm_route_none_event_attention_v_norm_route_rms_lanes,
        "state_residual_done"_s <= "state_normalized"_s + completion<EventScalarRun>(EventScalarRun<'event>) [guard_scalar_normalized_attention_route_event_attention_qk_norm_route_headwise_rms_event_attention_v_norm_route_rms] / effect_run_scalar_attention_mode_route_event_attention_qk_norm_route_headwise_rms_event_attention_v_norm_route_rms_lanes,
        "state_residual_done"_s <= "state_normalized"_s + completion<EventScalarRun>(EventScalarRun<'event>) [guard_scalar_normalized_shortconv_route] / effect_run_scalar_shortconv_route_lanes,
        "state_idle"_s <= "state_normalized"_s + completion<EventScalarRun>(EventScalarRun<'event>) [guard_normalized_ok] / effect_reject_unsupported_route,
        "state_idle"_s <= "state_normalized"_s + completion<EventScalarRun>(EventScalarRun<'event>) [guard_normalized_failed] / effect_mark_failed_from_state_normalized,
        "state_feed_forward_done"_s <= "state_residual_done"_s + completion<EventScalarRun>(EventScalarRun<'event>) [guard_residual_ok] / effect_run_scalar_feed_forward_route_lanes,
        "state_idle"_s <= "state_residual_done"_s + completion<EventScalarRun>(EventScalarRun<'event>) [guard_residual_failed] / effect_mark_failed_from_state_residual_done,
        "state_idle"_s <= "state_feed_forward_done"_s + completion<EventScalarRun>(EventScalarRun<'event>) [guard_feed_forward_ok] / effect_mark_succeeded,
        "state_idle"_s <= "state_feed_forward_done"_s + completion<EventScalarRun>(EventScalarRun<'event>) [guard_feed_forward_failed] / effect_mark_failed_from_state_feed_forward_done,
        "state_idle"_s <= "state_idle"_s + unexpected_event<_> / effect_on_unexpected,
    }
}

fn callback<'a>(request: &LayerRequest<'a>, operation: LayerOperation) -> bool {
    request
        .callback
        .is_some_and(|kernel| kernel(operation, request))
}
fn reset_scalar(event: &EventScalarRun<'_>) {
    event.stream_ready.set(false);
    event.normalized_ok.set(false);
    event.residual_ok.set(false);
    event.feed_forward_ok.set(false);
    event.succeeded.set(false);
    event.failed.set(false);
    event.error.set(LayerError::None);
}
fn reset_chunk4(event: &EventChunk4Run<'_>) {
    event.normalized_ok.set(false);
    event.residual_ok.set(false);
    event.feed_forward_ok.set(false);
    event.succeeded.set(false);
    event.failed.set(false);
    event.error.set(LayerError::None);
}
fn reset_chunk8(event: &EventChunk8Run<'_>) {
    event.normalized_ok.set(false);
    event.residual_ok.set(false);
    event.feed_forward_ok.set(false);
    event.succeeded.set(false);
    event.failed.set(false);
    event.error.set(LayerError::None);
}
fn prepare_scalar(event: &EventScalarRun<'_>) {
    reset_scalar(event);
    if !event.request.common_valid()
        || !event.request.storage_valid(1)
        || event.request.callback.is_none()
    {
        event.error.set(LayerError::InvalidRequest);
        event.failed.set(true);
        return;
    }
    event.stream_ready.set(match event.request.window_mode {
        WindowMode::Resident => true,
        WindowMode::Streamed => callback(&event.request, LayerOperation::PrepareScalar),
    });
    if !event.stream_ready.get() {
        event.error.set(LayerError::Kernel);
        event.failed.set(true);
    }
}
fn normalize_scalar(event: &EventScalarRun<'_>) {
    event.normalized_ok.set(false);
    event.residual_ok.set(false);
    event.feed_forward_ok.set(false);
    event.succeeded.set(false);
    event.failed.set(false);
    event.error.set(LayerError::None);
    if !event.request.scalar_valid() {
        event.error.set(LayerError::InvalidRequest);
    } else if callback(&event.request, LayerOperation::Normalize) {
        event.normalized_ok.set(true);
    } else {
        event.error.set(LayerError::Kernel);
    }
    if !event.normalized_ok.get() {
        event.failed.set(true);
    }
}
fn normalize_chunk4(event: &EventChunk4Run<'_>) {
    reset_chunk4(event);
    if !event.request.chunk4_valid() {
        event.error.set(LayerError::InvalidRequest);
    } else if callback(&event.request, LayerOperation::Normalize) {
        event.normalized_ok.set(true);
    } else {
        event.error.set(LayerError::Kernel);
    }
    if !event.normalized_ok.get() {
        event.failed.set(true);
    }
}
fn normalize_chunk8(event: &EventChunk8Run<'_>) {
    reset_chunk8(event);
    if !event.request.chunk8_valid() {
        event.error.set(LayerError::InvalidRequest);
    } else if callback(&event.request, LayerOperation::Normalize) {
        event.normalized_ok.set(true);
    } else {
        event.normalized_ok.set(false);
        event.error.set(LayerError::Kernel);
    }
    if !event.normalized_ok.get() {
        event.failed.set(true);
    }
}
fn residual_scalar(event: &EventScalarRun<'_>, operation: LayerOperation) {
    event.residual_ok.set(callback(&event.request, operation));
    if !event.residual_ok.get() {
        event.error.set(LayerError::Kernel);
    }
}
fn residual_chunk4(event: &EventChunk4Run<'_>, operation: LayerOperation) {
    event.residual_ok.set(callback(&event.request, operation));
    if !event.residual_ok.get() {
        event.error.set(LayerError::Kernel);
    }
}
fn residual_chunk8(event: &EventChunk8Run<'_>, operation: LayerOperation) {
    event.residual_ok.set(callback(&event.request, operation));
    if !event.residual_ok.get() {
        event.error.set(LayerError::Kernel);
    }
}
fn ff_scalar(event: &EventScalarRun<'_>) {
    event
        .feed_forward_ok
        .set(callback(&event.request, LayerOperation::ScalarFeedForward));
    if !event.feed_forward_ok.get() {
        event.error.set(LayerError::Kernel);
    }
}
fn ff_chunk4(event: &EventChunk4Run<'_>) {
    event
        .feed_forward_ok
        .set(callback(&event.request, LayerOperation::Chunk4FeedForward));
    if !event.feed_forward_ok.get() {
        event.error.set(LayerError::Kernel);
    }
}
fn ff_chunk8(event: &EventChunk8Run<'_>) {
    event
        .feed_forward_ok
        .set(callback(&event.request, LayerOperation::Chunk8FeedForward));
    if !event.feed_forward_ok.get() {
        event.error.set(LayerError::Kernel);
    }
}
fn mark_failed(failed: &Cell<bool>, succeeded: &Cell<bool>) {
    succeeded.set(false);
    failed.set(true);
}

#[derive(Debug, Default)]
pub struct TextGeneratorLayerScalarModelContext {
    pub unexpected: bool,
    pub error: LayerError,
}
impl TextGeneratorLayerScalarModelStateMachineContext for TextGeneratorLayerScalarModelContext {
    fn effect_mark_failed_from_state_feed_forward_done(
        &mut self,
        event: &EventScalarRun<'_>,
    ) -> Result<(), ()> {
        mark_failed(&event.failed, &event.succeeded);
        self.error = event.error.get();
        Ok(())
    }
    fn effect_mark_failed_from_state_input_ready(
        &mut self,
        event: &EventScalarRun<'_>,
    ) -> Result<(), ()> {
        mark_failed(&event.failed, &event.succeeded);
        self.error = event.error.get();
        Ok(())
    }
    fn effect_mark_failed_from_state_normalized(
        &mut self,
        event: &EventScalarRun<'_>,
    ) -> Result<(), ()> {
        mark_failed(&event.failed, &event.succeeded);
        self.error = event.error.get();
        Ok(())
    }
    fn effect_mark_failed_from_state_residual_done(
        &mut self,
        event: &EventScalarRun<'_>,
    ) -> Result<(), ()> {
        mark_failed(&event.failed, &event.succeeded);
        self.error = event.error.get();
        Ok(())
    }
    fn effect_mark_succeeded(&mut self, event: &EventScalarRun<'_>) -> Result<(), ()> {
        event.succeeded.set(true);
        event.failed.set(false);
        event.error.set(LayerError::None);
        self.error = LayerError::None;
        Ok(())
    }
    fn effect_normalize_scalar(&mut self, event: &EventScalarRun<'_>) -> Result<(), ()> {
        normalize_scalar(event);
        self.error = event.error.get();
        Ok(())
    }
    fn effect_on_unexpected(&mut self) -> Result<(), ()> {
        self.unexpected = true;
        self.error = LayerError::Unexpected;
        Ok(())
    }
    fn effect_prepare_scalar_wmode(&mut self, event: &EventScalarRun<'_>) -> Result<(), ()> {
        prepare_scalar(event);
        self.error = event.error.get();
        Ok(())
    }
    fn effect_reject_unsupported_route(&mut self, event: &EventScalarRun<'_>) -> Result<(), ()> {
        event.succeeded.set(false);
        event.failed.set(true);
        event.error.set(LayerError::UnsupportedRoute);
        self.error = LayerError::UnsupportedRoute;
        Ok(())
    }
    fn effect_run_scalar_attention_mode_route_event_attention_qk_norm_route_headwise_rms_event_attention_v_norm_route_none_lanes(
        &mut self,
        event: &EventScalarRun<'_>,
    ) -> Result<(), ()> {
        residual_scalar(event, LayerOperation::ScalarAttentionHeadwiseRmsNone);
        self.error = event.error.get();
        Ok(())
    }
    fn effect_run_scalar_attention_mode_route_event_attention_qk_norm_route_headwise_rms_event_attention_v_norm_route_rms_lanes(
        &mut self,
        event: &EventScalarRun<'_>,
    ) -> Result<(), ()> {
        residual_scalar(event, LayerOperation::ScalarAttentionHeadwiseRmsRms);
        self.error = event.error.get();
        Ok(())
    }
    fn effect_run_scalar_attention_mode_route_event_attention_qk_norm_route_none_event_attention_v_norm_route_none_lanes(
        &mut self,
        event: &EventScalarRun<'_>,
    ) -> Result<(), ()> {
        residual_scalar(event, LayerOperation::ScalarAttentionNoneNone);
        self.error = event.error.get();
        Ok(())
    }
    fn effect_run_scalar_attention_mode_route_event_attention_qk_norm_route_none_event_attention_v_norm_route_rms_lanes(
        &mut self,
        event: &EventScalarRun<'_>,
    ) -> Result<(), ()> {
        residual_scalar(event, LayerOperation::ScalarAttentionNoneRms);
        self.error = event.error.get();
        Ok(())
    }
    fn effect_run_scalar_feed_forward_route_lanes(
        &mut self,
        event: &EventScalarRun<'_>,
    ) -> Result<(), ()> {
        ff_scalar(event);
        self.error = event.error.get();
        Ok(())
    }
    fn effect_run_scalar_shortconv_route_lanes(
        &mut self,
        event: &EventScalarRun<'_>,
    ) -> Result<(), ()> {
        residual_scalar(event, LayerOperation::ScalarShortconv);
        self.error = event.error.get();
        Ok(())
    }
    fn guard_feed_forward_failed(&self, event: &EventScalarRun<'_>) -> Result<bool, ()> {
        Ok(!event.feed_forward_ok.get())
    }
    fn guard_feed_forward_ok(&self, event: &EventScalarRun<'_>) -> Result<bool, ()> {
        Ok(event.feed_forward_ok.get())
    }
    fn guard_normalized_failed(&self, event: &EventScalarRun<'_>) -> Result<bool, ()> {
        Ok(!event.normalized_ok.get())
    }
    fn guard_normalized_ok(&self, event: &EventScalarRun<'_>) -> Result<bool, ()> {
        Ok(event.normalized_ok.get())
    }
    fn guard_residual_failed(&self, event: &EventScalarRun<'_>) -> Result<bool, ()> {
        Ok(!event.residual_ok.get())
    }
    fn guard_residual_ok(&self, event: &EventScalarRun<'_>) -> Result<bool, ()> {
        Ok(event.residual_ok.get())
    }
    fn guard_scalar_normalized_attention_route_event_attention_qk_norm_route_headwise_rms_event_attention_v_norm_route_none(
        &self,
        event: &EventScalarRun<'_>,
    ) -> Result<bool, ()> {
        Ok(event.normalized_ok.get()
            && event.request.residual == ResidualRoute::Attention
            && event.request.qk_norm == AttentionQkNormRoute::HeadwiseRms
            && event.request.v_norm == AttentionVNormRoute::None)
    }
    fn guard_scalar_normalized_attention_route_event_attention_qk_norm_route_headwise_rms_event_attention_v_norm_route_rms(
        &self,
        event: &EventScalarRun<'_>,
    ) -> Result<bool, ()> {
        Ok(event.normalized_ok.get()
            && event.request.residual == ResidualRoute::Attention
            && event.request.qk_norm == AttentionQkNormRoute::HeadwiseRms
            && event.request.v_norm == AttentionVNormRoute::Rms)
    }
    fn guard_scalar_normalized_attention_route_event_attention_qk_norm_route_none_event_attention_v_norm_route_none(
        &self,
        event: &EventScalarRun<'_>,
    ) -> Result<bool, ()> {
        Ok(event.normalized_ok.get()
            && event.request.residual == ResidualRoute::Attention
            && event.request.qk_norm == AttentionQkNormRoute::None
            && event.request.v_norm == AttentionVNormRoute::None)
    }
    fn guard_scalar_normalized_attention_route_event_attention_qk_norm_route_none_event_attention_v_norm_route_rms(
        &self,
        event: &EventScalarRun<'_>,
    ) -> Result<bool, ()> {
        Ok(event.normalized_ok.get()
            && event.request.residual == ResidualRoute::Attention
            && event.request.qk_norm == AttentionQkNormRoute::None
            && event.request.v_norm == AttentionVNormRoute::Rms)
    }
    fn guard_scalar_normalized_shortconv_route(
        &self,
        event: &EventScalarRun<'_>,
    ) -> Result<bool, ()> {
        Ok(event.normalized_ok.get() && event.request.residual == ResidualRoute::Shortconv)
    }
    fn guard_stream_failed(&self, event: &EventScalarRun<'_>) -> Result<bool, ()> {
        Ok(!event.stream_ready.get())
    }
    fn guard_stream_ready(&self, event: &EventScalarRun<'_>) -> Result<bool, ()> {
        Ok(event.stream_ready.get())
    }
}

// --- machine TextGeneratorLayerChunk4Model from emel.cpp/src/emel/text/generator/layer/sm.hpp ---
sml! {
    TextGeneratorLayerChunk4Model<'event> {
        "state_normalized"_s <= *"state_idle"_s + event<EventChunk4Run<'event>> / effect_normalize_chunk4,
        "state_residual_done"_s <= "state_normalized"_s + completion<EventChunk4Run>(EventChunk4Run<'event>) [guard_chunk4_normalized_attention_route_event_attention_qk_norm_route_none_event_attention_v_norm_route_none] / effect_run_chunk4_attention_mode_route_event_attention_qk_norm_route_none_event_attention_v_norm_route_none_lanes,
        "state_residual_done"_s <= "state_normalized"_s + completion<EventChunk4Run>(EventChunk4Run<'event>) [guard_chunk4_normalized_attention_route_event_attention_qk_norm_route_headwise_rms_event_attention_v_norm_route_none] / effect_run_chunk4_attention_mode_route_event_attention_qk_norm_route_headwise_rms_event_attention_v_norm_route_none_lanes,
        "state_residual_done"_s <= "state_normalized"_s + completion<EventChunk4Run>(EventChunk4Run<'event>) [guard_chunk4_normalized_attention_route_event_attention_qk_norm_route_none_event_attention_v_norm_route_rms] / effect_run_chunk4_attention_mode_route_event_attention_qk_norm_route_none_event_attention_v_norm_route_rms_lanes,
        "state_residual_done"_s <= "state_normalized"_s + completion<EventChunk4Run>(EventChunk4Run<'event>) [guard_chunk4_normalized_attention_route_event_attention_qk_norm_route_headwise_rms_event_attention_v_norm_route_rms] / effect_run_chunk4_attention_mode_route_event_attention_qk_norm_route_headwise_rms_event_attention_v_norm_route_rms_lanes,
        "state_residual_done"_s <= "state_normalized"_s + completion<EventChunk4Run>(EventChunk4Run<'event>) [guard_chunk4_normalized_shortconv_route] / effect_run_chunk4_shortconv_route_lanes,
        "state_idle"_s <= "state_normalized"_s + completion<EventChunk4Run>(EventChunk4Run<'event>) [guard_normalized_ok] / effect_reject_unsupported_route,
        "state_idle"_s <= "state_normalized"_s + completion<EventChunk4Run>(EventChunk4Run<'event>) [guard_normalized_failed] / effect_mark_failed_from_state_normalized,
        "state_feed_forward_done"_s <= "state_residual_done"_s + completion<EventChunk4Run>(EventChunk4Run<'event>) [guard_residual_ok] / effect_run_chunk4_feed_forward_route_lanes,
        "state_idle"_s <= "state_residual_done"_s + completion<EventChunk4Run>(EventChunk4Run<'event>) [guard_residual_failed] / effect_mark_failed_from_state_residual_done,
        "state_idle"_s <= "state_feed_forward_done"_s + completion<EventChunk4Run>(EventChunk4Run<'event>) [guard_feed_forward_ok] / effect_mark_succeeded,
        "state_idle"_s <= "state_feed_forward_done"_s + completion<EventChunk4Run>(EventChunk4Run<'event>) [guard_feed_forward_failed] / effect_mark_failed_from_state_feed_forward_done,
        "state_idle"_s <= "state_idle"_s + unexpected_event<_> / effect_on_unexpected,
    }
}
#[derive(Debug, Default)]
pub struct TextGeneratorLayerChunk4ModelContext {
    pub unexpected: bool,
    pub error: LayerError,
}
impl TextGeneratorLayerChunk4ModelStateMachineContext for TextGeneratorLayerChunk4ModelContext {
    fn effect_mark_failed_from_state_feed_forward_done(
        &mut self,
        event: &EventChunk4Run<'_>,
    ) -> Result<(), ()> {
        mark_failed(&event.failed, &event.succeeded);
        self.error = event.error.get();
        Ok(())
    }
    fn effect_mark_failed_from_state_normalized(
        &mut self,
        event: &EventChunk4Run<'_>,
    ) -> Result<(), ()> {
        mark_failed(&event.failed, &event.succeeded);
        self.error = event.error.get();
        Ok(())
    }
    fn effect_mark_failed_from_state_residual_done(
        &mut self,
        event: &EventChunk4Run<'_>,
    ) -> Result<(), ()> {
        mark_failed(&event.failed, &event.succeeded);
        self.error = event.error.get();
        Ok(())
    }
    fn effect_mark_succeeded(&mut self, event: &EventChunk4Run<'_>) -> Result<(), ()> {
        event.succeeded.set(true);
        event.failed.set(false);
        event.error.set(LayerError::None);
        self.error = LayerError::None;
        Ok(())
    }
    fn effect_normalize_chunk4(&mut self, event: &EventChunk4Run<'_>) -> Result<(), ()> {
        normalize_chunk4(event);
        self.error = event.error.get();
        Ok(())
    }
    fn effect_on_unexpected(&mut self) -> Result<(), ()> {
        self.unexpected = true;
        self.error = LayerError::Unexpected;
        Ok(())
    }
    fn effect_reject_unsupported_route(&mut self, event: &EventChunk4Run<'_>) -> Result<(), ()> {
        event.succeeded.set(false);
        event.failed.set(true);
        event.error.set(LayerError::UnsupportedRoute);
        self.error = LayerError::UnsupportedRoute;
        Ok(())
    }
    fn effect_run_chunk4_attention_mode_route_event_attention_qk_norm_route_headwise_rms_event_attention_v_norm_route_none_lanes(
        &mut self,
        event: &EventChunk4Run<'_>,
    ) -> Result<(), ()> {
        residual_chunk4(event, LayerOperation::Chunk4AttentionHeadwiseRmsNone);
        self.error = event.error.get();
        Ok(())
    }
    fn effect_run_chunk4_attention_mode_route_event_attention_qk_norm_route_headwise_rms_event_attention_v_norm_route_rms_lanes(
        &mut self,
        event: &EventChunk4Run<'_>,
    ) -> Result<(), ()> {
        residual_chunk4(event, LayerOperation::Chunk4AttentionHeadwiseRmsRms);
        self.error = event.error.get();
        Ok(())
    }
    fn effect_run_chunk4_attention_mode_route_event_attention_qk_norm_route_none_event_attention_v_norm_route_none_lanes(
        &mut self,
        event: &EventChunk4Run<'_>,
    ) -> Result<(), ()> {
        residual_chunk4(event, LayerOperation::Chunk4AttentionNoneNone);
        self.error = event.error.get();
        Ok(())
    }
    fn effect_run_chunk4_attention_mode_route_event_attention_qk_norm_route_none_event_attention_v_norm_route_rms_lanes(
        &mut self,
        event: &EventChunk4Run<'_>,
    ) -> Result<(), ()> {
        residual_chunk4(event, LayerOperation::Chunk4AttentionNoneRms);
        self.error = event.error.get();
        Ok(())
    }
    fn effect_run_chunk4_feed_forward_route_lanes(
        &mut self,
        event: &EventChunk4Run<'_>,
    ) -> Result<(), ()> {
        ff_chunk4(event);
        self.error = event.error.get();
        Ok(())
    }
    fn effect_run_chunk4_shortconv_route_lanes(
        &mut self,
        event: &EventChunk4Run<'_>,
    ) -> Result<(), ()> {
        residual_chunk4(event, LayerOperation::Chunk4Shortconv);
        self.error = event.error.get();
        Ok(())
    }
    fn guard_chunk4_normalized_attention_route_event_attention_qk_norm_route_headwise_rms_event_attention_v_norm_route_none(
        &self,
        event: &EventChunk4Run<'_>,
    ) -> Result<bool, ()> {
        Ok(event.normalized_ok.get()
            && event.request.residual == ResidualRoute::Attention
            && event.request.qk_norm == AttentionQkNormRoute::HeadwiseRms
            && event.request.v_norm == AttentionVNormRoute::None)
    }
    fn guard_chunk4_normalized_attention_route_event_attention_qk_norm_route_headwise_rms_event_attention_v_norm_route_rms(
        &self,
        event: &EventChunk4Run<'_>,
    ) -> Result<bool, ()> {
        Ok(event.normalized_ok.get()
            && event.request.residual == ResidualRoute::Attention
            && event.request.qk_norm == AttentionQkNormRoute::HeadwiseRms
            && event.request.v_norm == AttentionVNormRoute::Rms)
    }
    fn guard_chunk4_normalized_attention_route_event_attention_qk_norm_route_none_event_attention_v_norm_route_none(
        &self,
        event: &EventChunk4Run<'_>,
    ) -> Result<bool, ()> {
        Ok(event.normalized_ok.get()
            && event.request.residual == ResidualRoute::Attention
            && event.request.qk_norm == AttentionQkNormRoute::None
            && event.request.v_norm == AttentionVNormRoute::None)
    }
    fn guard_chunk4_normalized_attention_route_event_attention_qk_norm_route_none_event_attention_v_norm_route_rms(
        &self,
        event: &EventChunk4Run<'_>,
    ) -> Result<bool, ()> {
        Ok(event.normalized_ok.get()
            && event.request.residual == ResidualRoute::Attention
            && event.request.qk_norm == AttentionQkNormRoute::None
            && event.request.v_norm == AttentionVNormRoute::Rms)
    }
    fn guard_chunk4_normalized_shortconv_route(
        &self,
        event: &EventChunk4Run<'_>,
    ) -> Result<bool, ()> {
        Ok(event.normalized_ok.get() && event.request.residual == ResidualRoute::Shortconv)
    }
    fn guard_feed_forward_failed(&self, event: &EventChunk4Run<'_>) -> Result<bool, ()> {
        Ok(!event.feed_forward_ok.get())
    }
    fn guard_feed_forward_ok(&self, event: &EventChunk4Run<'_>) -> Result<bool, ()> {
        Ok(event.feed_forward_ok.get())
    }
    fn guard_normalized_failed(&self, event: &EventChunk4Run<'_>) -> Result<bool, ()> {
        Ok(!event.normalized_ok.get())
    }
    fn guard_normalized_ok(&self, event: &EventChunk4Run<'_>) -> Result<bool, ()> {
        Ok(event.normalized_ok.get())
    }
    fn guard_residual_failed(&self, event: &EventChunk4Run<'_>) -> Result<bool, ()> {
        Ok(!event.residual_ok.get())
    }
    fn guard_residual_ok(&self, event: &EventChunk4Run<'_>) -> Result<bool, ()> {
        Ok(event.residual_ok.get())
    }
}

// --- machine TextGeneratorLayerChunk8Model from emel.cpp/src/emel/text/generator/layer/sm.hpp ---
sml! {
    TextGeneratorLayerChunk8Model<'event> {
        "state_normalized"_s <= *"state_idle"_s + event<EventChunk8Run<'event>> / effect_normalize_chunk8,
        "state_residual_done"_s <= "state_normalized"_s + completion<EventChunk8Run>(EventChunk8Run<'event>) [guard_chunk8_normalized_attention_route_event_attention_qk_norm_route_none_event_attention_v_norm_route_none] / effect_run_chunk8_attention_mode_event_attention_qk_norm_route_none_event_attention_v_norm_route_none_lanes,
        "state_residual_done"_s <= "state_normalized"_s + completion<EventChunk8Run>(EventChunk8Run<'event>) [guard_chunk8_normalized_attention_route_event_attention_qk_norm_route_headwise_rms_event_attention_v_norm_route_none] / effect_run_chunk8_attention_mode_event_attention_qk_norm_route_headwise_rms_event_attention_v_norm_route_none_lanes,
        "state_residual_done"_s <= "state_normalized"_s + completion<EventChunk8Run>(EventChunk8Run<'event>) [guard_chunk8_normalized_attention_route_event_attention_qk_norm_route_none_event_attention_v_norm_route_rms] / effect_run_chunk8_attention_mode_event_attention_qk_norm_route_none_event_attention_v_norm_route_rms_lanes,
        "state_residual_done"_s <= "state_normalized"_s + completion<EventChunk8Run>(EventChunk8Run<'event>) [guard_chunk8_normalized_attention_route_event_attention_qk_norm_route_headwise_rms_event_attention_v_norm_route_rms] / effect_run_chunk8_attention_mode_event_attention_qk_norm_route_headwise_rms_event_attention_v_norm_route_rms_lanes,
        "state_residual_done"_s <= "state_normalized"_s + completion<EventChunk8Run>(EventChunk8Run<'event>) [guard_chunk8_normalized_shortconv_route] / effect_run_chunk8_shortconv_lanes,
        "state_idle"_s <= "state_normalized"_s + completion<EventChunk8Run>(EventChunk8Run<'event>) [guard_normalized_ok] / effect_reject_unsupported_route,
        "state_idle"_s <= "state_normalized"_s + completion<EventChunk8Run>(EventChunk8Run<'event>) [guard_normalized_failed] / effect_mark_failed_from_state_normalized,
        "state_feed_forward_done"_s <= "state_residual_done"_s + completion<EventChunk8Run>(EventChunk8Run<'event>) [guard_residual_ok] / effect_run_chunk8_feed_forward_lanes,
        "state_idle"_s <= "state_residual_done"_s + completion<EventChunk8Run>(EventChunk8Run<'event>) [guard_residual_failed] / effect_mark_failed_from_state_residual_done,
        "state_idle"_s <= "state_feed_forward_done"_s + completion<EventChunk8Run>(EventChunk8Run<'event>) [guard_feed_forward_ok] / effect_mark_succeeded,
        "state_idle"_s <= "state_feed_forward_done"_s + completion<EventChunk8Run>(EventChunk8Run<'event>) [guard_feed_forward_failed] / effect_mark_failed_from_state_feed_forward_done,
        "state_idle"_s <= "state_idle"_s + unexpected_event<_> / effect_on_unexpected,
    }
}
#[derive(Debug, Default)]
pub struct TextGeneratorLayerChunk8ModelContext {
    pub unexpected: bool,
    pub error: LayerError,
}
impl TextGeneratorLayerChunk8ModelStateMachineContext for TextGeneratorLayerChunk8ModelContext {
    fn effect_mark_failed_from_state_feed_forward_done(
        &mut self,
        event: &EventChunk8Run<'_>,
    ) -> Result<(), ()> {
        mark_failed(&event.failed, &event.succeeded);
        self.error = event.error.get();
        Ok(())
    }
    fn effect_mark_failed_from_state_normalized(
        &mut self,
        event: &EventChunk8Run<'_>,
    ) -> Result<(), ()> {
        mark_failed(&event.failed, &event.succeeded);
        self.error = event.error.get();
        Ok(())
    }
    fn effect_mark_failed_from_state_residual_done(
        &mut self,
        event: &EventChunk8Run<'_>,
    ) -> Result<(), ()> {
        mark_failed(&event.failed, &event.succeeded);
        self.error = event.error.get();
        Ok(())
    }
    fn effect_mark_succeeded(&mut self, event: &EventChunk8Run<'_>) -> Result<(), ()> {
        event.succeeded.set(true);
        event.failed.set(false);
        event.error.set(LayerError::None);
        self.error = LayerError::None;
        Ok(())
    }
    fn effect_normalize_chunk8(&mut self, event: &EventChunk8Run<'_>) -> Result<(), ()> {
        normalize_chunk8(event);
        self.error = event.error.get();
        Ok(())
    }
    fn effect_on_unexpected(&mut self) -> Result<(), ()> {
        self.unexpected = true;
        self.error = LayerError::Unexpected;
        Ok(())
    }
    fn effect_reject_unsupported_route(&mut self, event: &EventChunk8Run<'_>) -> Result<(), ()> {
        event.succeeded.set(false);
        event.failed.set(true);
        event.error.set(LayerError::UnsupportedRoute);
        self.error = LayerError::UnsupportedRoute;
        Ok(())
    }
    fn effect_run_chunk8_attention_mode_event_attention_qk_norm_route_headwise_rms_event_attention_v_norm_route_none_lanes(
        &mut self,
        event: &EventChunk8Run<'_>,
    ) -> Result<(), ()> {
        residual_chunk8(event, LayerOperation::Chunk8AttentionHeadwiseRmsNone);
        self.error = event.error.get();
        Ok(())
    }
    fn effect_run_chunk8_attention_mode_event_attention_qk_norm_route_headwise_rms_event_attention_v_norm_route_rms_lanes(
        &mut self,
        event: &EventChunk8Run<'_>,
    ) -> Result<(), ()> {
        residual_chunk8(event, LayerOperation::Chunk8AttentionHeadwiseRmsRms);
        self.error = event.error.get();
        Ok(())
    }
    fn effect_run_chunk8_attention_mode_event_attention_qk_norm_route_none_event_attention_v_norm_route_none_lanes(
        &mut self,
        event: &EventChunk8Run<'_>,
    ) -> Result<(), ()> {
        residual_chunk8(event, LayerOperation::Chunk8AttentionNoneNone);
        self.error = event.error.get();
        Ok(())
    }
    fn effect_run_chunk8_attention_mode_event_attention_qk_norm_route_none_event_attention_v_norm_route_rms_lanes(
        &mut self,
        event: &EventChunk8Run<'_>,
    ) -> Result<(), ()> {
        residual_chunk8(event, LayerOperation::Chunk8AttentionNoneRms);
        self.error = event.error.get();
        Ok(())
    }
    fn effect_run_chunk8_feed_forward_lanes(
        &mut self,
        event: &EventChunk8Run<'_>,
    ) -> Result<(), ()> {
        ff_chunk8(event);
        self.error = event.error.get();
        Ok(())
    }
    fn effect_run_chunk8_shortconv_lanes(&mut self, event: &EventChunk8Run<'_>) -> Result<(), ()> {
        residual_chunk8(event, LayerOperation::Chunk8Shortconv);
        self.error = event.error.get();
        Ok(())
    }
    fn guard_chunk8_normalized_attention_route_event_attention_qk_norm_route_headwise_rms_event_attention_v_norm_route_none(
        &self,
        event: &EventChunk8Run<'_>,
    ) -> Result<bool, ()> {
        Ok(event.normalized_ok.get()
            && event.request.residual == ResidualRoute::Attention
            && event.request.qk_norm == AttentionQkNormRoute::HeadwiseRms
            && event.request.v_norm == AttentionVNormRoute::None)
    }
    fn guard_chunk8_normalized_attention_route_event_attention_qk_norm_route_headwise_rms_event_attention_v_norm_route_rms(
        &self,
        event: &EventChunk8Run<'_>,
    ) -> Result<bool, ()> {
        Ok(event.normalized_ok.get()
            && event.request.residual == ResidualRoute::Attention
            && event.request.qk_norm == AttentionQkNormRoute::HeadwiseRms
            && event.request.v_norm == AttentionVNormRoute::Rms)
    }
    fn guard_chunk8_normalized_attention_route_event_attention_qk_norm_route_none_event_attention_v_norm_route_none(
        &self,
        event: &EventChunk8Run<'_>,
    ) -> Result<bool, ()> {
        Ok(event.normalized_ok.get()
            && event.request.residual == ResidualRoute::Attention
            && event.request.qk_norm == AttentionQkNormRoute::None
            && event.request.v_norm == AttentionVNormRoute::None)
    }
    fn guard_chunk8_normalized_attention_route_event_attention_qk_norm_route_none_event_attention_v_norm_route_rms(
        &self,
        event: &EventChunk8Run<'_>,
    ) -> Result<bool, ()> {
        Ok(event.normalized_ok.get()
            && event.request.residual == ResidualRoute::Attention
            && event.request.qk_norm == AttentionQkNormRoute::None
            && event.request.v_norm == AttentionVNormRoute::Rms)
    }
    fn guard_chunk8_normalized_shortconv_route(
        &self,
        event: &EventChunk8Run<'_>,
    ) -> Result<bool, ()> {
        Ok(event.normalized_ok.get() && event.request.residual == ResidualRoute::Shortconv)
    }
    fn guard_feed_forward_failed(&self, event: &EventChunk8Run<'_>) -> Result<bool, ()> {
        Ok(!event.feed_forward_ok.get())
    }
    fn guard_feed_forward_ok(&self, event: &EventChunk8Run<'_>) -> Result<bool, ()> {
        Ok(event.feed_forward_ok.get())
    }
    fn guard_normalized_failed(&self, event: &EventChunk8Run<'_>) -> Result<bool, ()> {
        Ok(!event.normalized_ok.get())
    }
    fn guard_normalized_ok(&self, event: &EventChunk8Run<'_>) -> Result<bool, ()> {
        Ok(event.normalized_ok.get())
    }
    fn guard_residual_failed(&self, event: &EventChunk8Run<'_>) -> Result<bool, ()> {
        Ok(!event.residual_ok.get())
    }
    fn guard_residual_ok(&self, event: &EventChunk8Run<'_>) -> Result<bool, ()> {
        Ok(event.residual_ok.get())
    }
}
/// Public single-writer synchronous wrapper for the scalar layer machine.
pub struct TextGeneratorLayerScalarActor {
    machine: TextGeneratorLayerScalarModelStateMachine<TextGeneratorLayerScalarModelContext>,
}

impl Default for TextGeneratorLayerScalarActor {
    fn default() -> Self {
        Self::new()
    }
}

impl TextGeneratorLayerScalarActor {
    #[must_use]
    pub fn new() -> Self {
        Self {
            machine: TextGeneratorLayerScalarModelStateMachine::new(
                TextGeneratorLayerScalarModelContext::default(),
            ),
        }
    }

    /// Processes one scalar layer event synchronously.
    pub fn process_event(&mut self, event: EventScalarRun<'_>) -> Result<(), LayerError> {
        if self
            .machine
            .process_event(TextGeneratorLayerScalarModelEvents::EventScalarRun(event))
            .is_err()
        {
            self.machine.context_mut().error = LayerError::Unexpected;
            return Err(LayerError::Unexpected);
        }
        match self.machine.context().error {
            LayerError::None => Ok(()),
            error => Err(error),
        }
    }

    /// Source-compatible scalar dispatch spelling.
    pub fn run(&mut self, event: EventScalarRun<'_>) -> Result<(), LayerError> {
        self.process_event(event)
    }

    /// Drives the explicit unexpected-event path and returns to idle.
    pub fn process_unexpected_event(&mut self) -> Result<(), LayerError> {
        self.machine.context_mut().unexpected = true;
        self.machine.context_mut().error = LayerError::Unexpected;
        self.machine
            .set_state(TextGeneratorLayerScalarModelStates::StateIdle);
        Err(LayerError::Unexpected)
    }

    #[must_use]
    pub fn state(&self) -> &TextGeneratorLayerScalarModelStates {
        self.machine.state()
    }
    #[must_use]
    pub fn is(&self, state: &TextGeneratorLayerScalarModelStates) -> bool {
        self.machine.is(state)
    }
    #[must_use]
    pub fn context(&self) -> &TextGeneratorLayerScalarModelContext {
        self.machine.context()
    }
}

/// Public single-writer synchronous wrapper for the chunk4 layer machine.
pub struct TextGeneratorLayerChunk4Actor {
    machine: TextGeneratorLayerChunk4ModelStateMachine<TextGeneratorLayerChunk4ModelContext>,
}

impl Default for TextGeneratorLayerChunk4Actor {
    fn default() -> Self {
        Self::new()
    }
}

impl TextGeneratorLayerChunk4Actor {
    #[must_use]
    pub fn new() -> Self {
        Self {
            machine: TextGeneratorLayerChunk4ModelStateMachine::new(
                TextGeneratorLayerChunk4ModelContext::default(),
            ),
        }
    }

    /// Processes one chunk4 layer event synchronously.
    pub fn process_event(&mut self, event: EventChunk4Run<'_>) -> Result<(), LayerError> {
        if self
            .machine
            .process_event(TextGeneratorLayerChunk4ModelEvents::EventChunk4Run(event))
            .is_err()
        {
            self.machine.context_mut().error = LayerError::Unexpected;
            return Err(LayerError::Unexpected);
        }
        match self.machine.context().error {
            LayerError::None => Ok(()),
            error => Err(error),
        }
    }

    /// Source-compatible chunk4 dispatch spelling.
    pub fn run(&mut self, event: EventChunk4Run<'_>) -> Result<(), LayerError> {
        self.process_event(event)
    }

    /// Drives the explicit unexpected-event path and returns to idle.
    pub fn process_unexpected_event(&mut self) -> Result<(), LayerError> {
        self.machine.context_mut().unexpected = true;
        self.machine.context_mut().error = LayerError::Unexpected;
        self.machine
            .set_state(TextGeneratorLayerChunk4ModelStates::StateIdle);
        Err(LayerError::Unexpected)
    }

    #[must_use]
    pub fn state(&self) -> &TextGeneratorLayerChunk4ModelStates {
        self.machine.state()
    }
    #[must_use]
    pub fn is(&self, state: &TextGeneratorLayerChunk4ModelStates) -> bool {
        self.machine.is(state)
    }
    #[must_use]
    pub fn context(&self) -> &TextGeneratorLayerChunk4ModelContext {
        self.machine.context()
    }
}

/// Public single-writer synchronous wrapper for the chunk8 layer machine.
pub struct TextGeneratorLayerChunk8Actor {
    machine: TextGeneratorLayerChunk8ModelStateMachine<TextGeneratorLayerChunk8ModelContext>,
}

impl Default for TextGeneratorLayerChunk8Actor {
    fn default() -> Self {
        Self::new()
    }
}

impl TextGeneratorLayerChunk8Actor {
    #[must_use]
    pub fn new() -> Self {
        Self {
            machine: TextGeneratorLayerChunk8ModelStateMachine::new(
                TextGeneratorLayerChunk8ModelContext::default(),
            ),
        }
    }

    /// Processes one chunk8 layer event synchronously.
    pub fn process_event(&mut self, event: EventChunk8Run<'_>) -> Result<(), LayerError> {
        if self
            .machine
            .process_event(TextGeneratorLayerChunk8ModelEvents::EventChunk8Run(event))
            .is_err()
        {
            self.machine.context_mut().error = LayerError::Unexpected;
            return Err(LayerError::Unexpected);
        }
        match self.machine.context().error {
            LayerError::None => Ok(()),
            error => Err(error),
        }
    }

    /// Source-compatible chunk8 dispatch spelling.
    pub fn run(&mut self, event: EventChunk8Run<'_>) -> Result<(), LayerError> {
        self.process_event(event)
    }

    /// Drives the explicit unexpected-event path and returns to idle.
    pub fn process_unexpected_event(&mut self) -> Result<(), LayerError> {
        self.machine.context_mut().unexpected = true;
        self.machine.context_mut().error = LayerError::Unexpected;
        self.machine
            .set_state(TextGeneratorLayerChunk8ModelStates::StateIdle);
        Err(LayerError::Unexpected)
    }

    #[must_use]
    pub fn state(&self) -> &TextGeneratorLayerChunk8ModelStates {
        self.machine.state()
    }
    #[must_use]
    pub fn is(&self, state: &TextGeneratorLayerChunk8ModelStates) -> bool {
        self.machine.is(state)
    }
    #[must_use]
    pub fn context(&self) -> &TextGeneratorLayerChunk8ModelContext {
        self.machine.context()
    }
}

pub type LayerScalarActor = TextGeneratorLayerScalarActor;
pub type LayerChunk4Actor = TextGeneratorLayerChunk4Actor;
pub type LayerChunk8Actor = TextGeneratorLayerChunk8Actor;
