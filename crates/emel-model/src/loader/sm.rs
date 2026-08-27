//! Run-to-completion model-loader orchestration.
//!
//! Source contract: `emel.cpp/src/emel/model/loader/sm.hpp`.
//!
//! The pinned C++ loader has a deliberately explicit phase graph. The Rust
//! actor keeps that graph in SML and delegates tensor residency to an injected
//! actor. The injected actor is the ownership boundary for storage and I/O;
//! this machine only validates the request, selects the phase, and publishes
//! the typed outcome.

#![allow(
    clippy::derive_partial_eq_without_eq,
    clippy::module_name_repetitions,
    clippy::missing_errors_doc,
    clippy::must_use_candidate,
    clippy::return_self_not_must_use,
    clippy::empty_structs_with_brackets,
    clippy::missing_const_for_fn,
    dead_code,
    unused_imports,
    missing_docs
)]

use std::cell::{Cell, RefCell};

use sml::sml;

use super::event::{Error, LoadRequest, LoadStats, LoadStatus};

/// Runtime payload carried across the bounded phase transitions.
#[derive(Clone, Copy)]
pub(crate) struct EventLoadRuntime<'dispatch> {
    pub(crate) request: &'dispatch RefCell<LoadRequest<'dispatch>>,
    pub(crate) status: &'dispatch Cell<LoadStatus>,
    pub(crate) outcome: &'dispatch Cell<Result<LoadStats, Error>>,
}

// Explicit state graph for the model loader.
sml! {
    ModelLoader<'dispatch> {
        "request_decision"_s <= *"ready"_s + Load(EventLoadRuntime<'dispatch>) / effect_begin_load,

        "parsing"_s <= "request_decision"_s + completion<Load>(EventLoadRuntime<'dispatch>) [guard_valid_request],
        "errored"_s <= "request_decision"_s + completion<Load>(EventLoadRuntime<'dispatch>) [guard_invalid_request] / effect_invalid_request,

        "parse_decision"_s <= "parsing"_s + completion<Load>(EventLoadRuntime<'dispatch>) / effect_parse,
        "parse_phase_decision"_s <= "parse_decision"_s + completion<Load>(EventLoadRuntime<'dispatch>),
        "parse_load_tensors_policy_decision"_s <= "parse_phase_decision"_s + completion<Load>(EventLoadRuntime<'dispatch>) [guard_no_error],
        "errored"_s <= "parse_phase_decision"_s + completion<Load>(EventLoadRuntime<'dispatch>) [guard_has_error] / effect_preserve_error,

        "parse_load_tensors_handler_decision"_s <= "parse_load_tensors_policy_decision"_s + completion<Load>(EventLoadRuntime<'dispatch>) [guard_should_load_tensors],
        "structure_decision"_s <= "parse_load_tensors_policy_decision"_s + completion<Load>(EventLoadRuntime<'dispatch>) [guard_skip_load_tensors],
        "errored"_s <= "parse_load_tensors_policy_decision"_s + completion<Load>(EventLoadRuntime<'dispatch>) / effect_internal_error,

        "loading_tensors"_s <= "parse_load_tensors_handler_decision"_s + completion<Load>(EventLoadRuntime<'dispatch>) [guard_can_load_tensors],
        "errored"_s <= "parse_load_tensors_handler_decision"_s + completion<Load>(EventLoadRuntime<'dispatch>) [guard_cannot_load_tensors] / effect_invalid_request,

        // Tensor planning, I/O, and apply are owned by the injected tensor actor.
        // These phase states remain visible so lifecycle inspection matches the
        // reference machine without allowing reach-through to another actor.
        "state_tensor_bind_decision"_s <= "loading_tensors"_s + completion<Load>(EventLoadRuntime<'dispatch>) / effect_load_tensors,
        "state_tensor_plan_dispatch"_s <= "state_tensor_bind_decision"_s + completion<Load>(EventLoadRuntime<'dispatch>) [guard_tensor_load_done],
        "errored"_s <= "state_tensor_bind_decision"_s + completion<Load>(EventLoadRuntime<'dispatch>) [guard_tensor_load_error] / effect_preserve_error,
        "state_tensor_plan_decision"_s <= "state_tensor_plan_dispatch"_s + completion<Load>(EventLoadRuntime<'dispatch>),
        "state_tensor_apply_dispatch"_s <= "state_tensor_plan_decision"_s + completion<Load>(EventLoadRuntime<'dispatch>) [guard_tensor_load_done],
        "errored"_s <= "state_tensor_plan_decision"_s + completion<Load>(EventLoadRuntime<'dispatch>) [guard_tensor_load_error] / effect_preserve_error,
        "state_tensor_apply_decision"_s <= "state_tensor_apply_dispatch"_s + completion<Load>(EventLoadRuntime<'dispatch>),
        "load_map_policy_decision"_s <= "state_tensor_apply_decision"_s + completion<Load>(EventLoadRuntime<'dispatch>) [guard_tensor_load_done],
        "errored"_s <= "state_tensor_apply_decision"_s + completion<Load>(EventLoadRuntime<'dispatch>) [guard_tensor_load_error] / effect_preserve_error,

        "mapping_layers"_s <= "load_map_policy_decision"_s + completion<Load>(EventLoadRuntime<'dispatch>) [guard_can_map_layers],
        "errored"_s <= "load_map_policy_decision"_s + completion<Load>(EventLoadRuntime<'dispatch>) / effect_invalid_request,
        "map_layers_decision"_s <= "mapping_layers"_s + completion<Load>(EventLoadRuntime<'dispatch>) / effect_map_layers,
        "structure_decision"_s <= "map_layers_decision"_s + completion<Load>(EventLoadRuntime<'dispatch>) [guard_no_error],
        "errored"_s <= "map_layers_decision"_s + completion<Load>(EventLoadRuntime<'dispatch>) [guard_has_error] / effect_preserve_error,

        "structure_policy_decision"_s <= "structure_decision"_s + completion<Load>(EventLoadRuntime<'dispatch>),
        "architecture_decision"_s <= "structure_policy_decision"_s + completion<Load>(EventLoadRuntime<'dispatch>) [guard_skip_structure_validation],
        "validating_structure"_s <= "structure_policy_decision"_s + completion<Load>(EventLoadRuntime<'dispatch>) [guard_can_validate_structure],
        "errored"_s <= "structure_policy_decision"_s + completion<Load>(EventLoadRuntime<'dispatch>) / effect_invalid_request,
        "structure_validation_decision"_s <= "validating_structure"_s + completion<Load>(EventLoadRuntime<'dispatch>) / effect_validate_structure,
        "architecture_decision"_s <= "structure_validation_decision"_s + completion<Load>(EventLoadRuntime<'dispatch>) [guard_no_error],
        "errored"_s <= "structure_validation_decision"_s + completion<Load>(EventLoadRuntime<'dispatch>) [guard_has_error] / effect_preserve_error,

        "architecture_policy_decision"_s <= "architecture_decision"_s + completion<Load>(EventLoadRuntime<'dispatch>),
        "done"_s <= "architecture_policy_decision"_s + completion<Load>(EventLoadRuntime<'dispatch>) [guard_skip_architecture_validation],
        "validating_architecture"_s <= "architecture_policy_decision"_s + completion<Load>(EventLoadRuntime<'dispatch>) [guard_can_validate_architecture],
        "errored"_s <= "architecture_policy_decision"_s + completion<Load>(EventLoadRuntime<'dispatch>) / effect_invalid_request,
        "architecture_validation_decision"_s <= "validating_architecture"_s + completion<Load>(EventLoadRuntime<'dispatch>) / effect_validate_architecture,
        "done"_s <= "architecture_validation_decision"_s + completion<Load>(EventLoadRuntime<'dispatch>) [guard_no_error],
        "errored"_s <= "architecture_validation_decision"_s + completion<Load>(EventLoadRuntime<'dispatch>) [guard_has_error] / effect_preserve_error,

        "ready"_s <= "done"_s + completion<Load>(EventLoadRuntime<'dispatch>) [guard_done_callback] / effect_publish_done,
        "ready"_s <= "done"_s + completion<Load>(EventLoadRuntime<'dispatch>) / effect_publish_done_noop,
        "ready"_s <= "errored"_s + completion<Load>(EventLoadRuntime<'dispatch>) [guard_error_callback] / effect_publish_error,
        "ready"_s <= "errored"_s + completion<Load>(EventLoadRuntime<'dispatch>) / effect_publish_error_noop,

        "ready"_s <= "ready"_s + unexpected_event<_> / effect_unexpected,
        "ready"_s <= "request_decision"_s + unexpected_event<_> / effect_unexpected,
        "ready"_s <= "parsing"_s + unexpected_event<_> / effect_unexpected,
        "ready"_s <= "parse_decision"_s + unexpected_event<_> / effect_unexpected,
        "ready"_s <= "parse_phase_decision"_s + unexpected_event<_> / effect_unexpected,
        "ready"_s <= "parse_load_tensors_policy_decision"_s + unexpected_event<_> / effect_unexpected,
        "ready"_s <= "parse_load_tensors_handler_decision"_s + unexpected_event<_> / effect_unexpected,
        "ready"_s <= "loading_tensors"_s + unexpected_event<_> / effect_unexpected,
        "ready"_s <= "state_tensor_bind_decision"_s + unexpected_event<_> / effect_unexpected,
        "ready"_s <= "state_tensor_plan_dispatch"_s + unexpected_event<_> / effect_unexpected,
        "ready"_s <= "state_tensor_plan_decision"_s + unexpected_event<_> / effect_unexpected,
        "ready"_s <= "state_tensor_apply_dispatch"_s + unexpected_event<_> / effect_unexpected,
        "ready"_s <= "state_tensor_apply_decision"_s + unexpected_event<_> / effect_unexpected,
        "ready"_s <= "load_map_policy_decision"_s + unexpected_event<_> / effect_unexpected,
        "ready"_s <= "mapping_layers"_s + unexpected_event<_> / effect_unexpected,
        "ready"_s <= "map_layers_decision"_s + unexpected_event<_> / effect_unexpected,
        "ready"_s <= "structure_decision"_s + unexpected_event<_> / effect_unexpected,
        "ready"_s <= "structure_policy_decision"_s + unexpected_event<_> / effect_unexpected,
        "ready"_s <= "validating_structure"_s + unexpected_event<_> / effect_unexpected,
        "ready"_s <= "structure_validation_decision"_s + unexpected_event<_> / effect_unexpected,
        "ready"_s <= "architecture_decision"_s + unexpected_event<_> / effect_unexpected,
        "ready"_s <= "architecture_policy_decision"_s + unexpected_event<_> / effect_unexpected,
        "ready"_s <= "validating_architecture"_s + unexpected_event<_> / effect_unexpected,
        "ready"_s <= "architecture_validation_decision"_s + unexpected_event<_> / effect_unexpected,
        "ready"_s <= "done"_s + unexpected_event<_> / effect_unexpected,
        "ready"_s <= "errored"_s + unexpected_event<_> / effect_unexpected,
    }
}

/// Actor-local context. Request data and outcome cells are dispatch-local.
pub(crate) struct ModelLoaderContext<T> {
    pub(crate) tensor_loader: T,
}

impl<T> ModelLoaderContext<T> {
    pub(crate) const fn new(tensor_loader: T) -> Self {
        Self { tensor_loader }
    }
}

impl Default for ModelLoaderContext<super::actor::NoTensorLoader> {
    fn default() -> Self {
        Self::new(super::actor::NoTensorLoader)
    }
}

impl<T: super::actor::TensorLoader> ModelLoaderStateMachineContext for ModelLoaderContext<T> {
    fn effect_begin_load(&mut self, event: EventLoadRuntime<'_>) -> Result<(), ()> {
        event.status.set(LoadStatus::default());
        event.outcome.set(Err(Error::InternalError));
        Ok(())
    }

    fn guard_valid_request(&self, event: &EventLoadRuntime<'_>) -> Result<bool, ()> {
        Ok(event.request.borrow().is_valid())
    }

    fn guard_invalid_request(&self, event: &EventLoadRuntime<'_>) -> Result<bool, ()> {
        Ok(!event.request.borrow().is_valid())
    }

    fn effect_invalid_request(&mut self, event: EventLoadRuntime<'_>) -> Result<(), ()> {
        event.status.set(LoadStatus::error(Error::InvalidRequest));
        Ok(())
    }

    fn effect_parse(&mut self, event: EventLoadRuntime<'_>) -> Result<(), ()> {
        let mut request = event.request.borrow_mut();
        let parser = request.parse_model.expect("valid request has parser");
        let source = request.source;
        let model = &mut *request.model;
        let error = parser(model, source);
        event.status.set(LoadStatus::error(error));
        Ok(())
    }

    fn guard_no_error(&self, event: &EventLoadRuntime<'_>) -> Result<bool, ()> {
        Ok(event.status.get().error == Error::None)
    }

    fn guard_has_error(&self, event: &EventLoadRuntime<'_>) -> Result<bool, ()> {
        Ok(event.status.get().error != Error::None)
    }

    fn effect_preserve_error(&mut self, _event: EventLoadRuntime<'_>) -> Result<(), ()> {
        Ok(())
    }

    fn guard_should_load_tensors(&self, event: &EventLoadRuntime<'_>) -> Result<bool, ()> {
        Ok(!event.request.borrow().vocab_only)
    }

    fn guard_skip_load_tensors(&self, event: &EventLoadRuntime<'_>) -> Result<bool, ()> {
        Ok(event.request.borrow().vocab_only)
    }

    fn guard_can_load_tensors(&self, event: &EventLoadRuntime<'_>) -> Result<bool, ()> {
        let request = event.request.borrow();
        Ok(request.model.n_tensors > 0 && request.tensor_capacity_valid())
    }

    fn guard_cannot_load_tensors(&self, event: &EventLoadRuntime<'_>) -> Result<bool, ()> {
        let request = event.request.borrow();
        Ok(request.model.n_tensors == 0 || !request.tensor_capacity_valid())
    }

    fn effect_load_tensors(&mut self, event: EventLoadRuntime<'_>) -> Result<(), ()> {
        let mut request = event.request.borrow_mut();
        let source = request.source;
        let strategy = request.io_strategy;
        let model = &mut *request.model;
        let result = self.tensor_loader.load(model, source, strategy);
        event.outcome.set(result);
        let status = result.map_or_else(LoadStatus::error, LoadStatus::success);
        event.status.set(status);
        Ok(())
    }

    fn guard_tensor_load_done(&self, event: &EventLoadRuntime<'_>) -> Result<bool, ()> {
        Ok(event.outcome.get().is_ok())
    }

    fn guard_tensor_load_error(&self, event: &EventLoadRuntime<'_>) -> Result<bool, ()> {
        Ok(event.outcome.get().is_err())
    }

    fn guard_can_map_layers(&self, event: &EventLoadRuntime<'_>) -> Result<bool, ()> {
        Ok(event.request.borrow().map_layers.is_some())
    }

    fn effect_map_layers(&mut self, event: EventLoadRuntime<'_>) -> Result<(), ()> {
        let mut request = event.request.borrow_mut();
        let error = request.map_layers.expect("guard selected map callback")(request.model);
        event.status.set(LoadStatus {
            error,
            stats: event.status.get().stats,
        });
        Ok(())
    }

    fn guard_skip_structure_validation(&self, event: &EventLoadRuntime<'_>) -> Result<bool, ()> {
        Ok(!event.request.borrow().check_tensors)
    }

    fn guard_can_validate_structure(&self, event: &EventLoadRuntime<'_>) -> Result<bool, ()> {
        let request = event.request.borrow();
        Ok(request.check_tensors && request.validate_structure.is_some())
    }

    fn effect_validate_structure(&mut self, event: EventLoadRuntime<'_>) -> Result<(), ()> {
        let mut request = event.request.borrow_mut();
        let error = request
            .validate_structure
            .expect("guard selected structure callback")(request.model);
        event.status.set(LoadStatus {
            error,
            stats: event.status.get().stats,
        });
        Ok(())
    }

    fn guard_skip_architecture_validation(&self, event: &EventLoadRuntime<'_>) -> Result<bool, ()> {
        Ok(!event.request.borrow().validate_architecture)
    }

    fn guard_can_validate_architecture(&self, event: &EventLoadRuntime<'_>) -> Result<bool, ()> {
        let request = event.request.borrow();
        Ok(request.validate_architecture && request.validate_architecture_impl.is_some())
    }

    fn effect_validate_architecture(&mut self, event: EventLoadRuntime<'_>) -> Result<(), ()> {
        let mut request = event.request.borrow_mut();
        let error = request
            .validate_architecture_impl
            .expect("guard selected architecture callback")(request.model);
        event.status.set(LoadStatus {
            error,
            stats: event.status.get().stats,
        });
        Ok(())
    }

    fn guard_done_callback(&self, event: &EventLoadRuntime<'_>) -> Result<bool, ()> {
        Ok(event.request.borrow().on_done.is_some())
    }

    fn guard_error_callback(&self, event: &EventLoadRuntime<'_>) -> Result<bool, ()> {
        Ok(event.request.borrow().on_error.is_some())
    }

    fn effect_publish_done(&mut self, event: EventLoadRuntime<'_>) -> Result<(), ()> {
        let request = event.request.borrow();
        let stats = event.status.get().stats;
        request.on_done.expect("guard selected done callback")(&request, stats);
        event.outcome.set(Ok(stats));
        Ok(())
    }

    fn effect_publish_done_noop(&mut self, event: EventLoadRuntime<'_>) -> Result<(), ()> {
        event.outcome.set(Ok(event.status.get().stats));
        Ok(())
    }

    fn effect_publish_error(&mut self, event: EventLoadRuntime<'_>) -> Result<(), ()> {
        let request = event.request.borrow();
        let status = event.status.get();
        let error = super::event::LoadError::new(
            status.error,
            request.io_strategy,
            status.stats.used_strategy,
        );
        request.on_error.expect("guard selected error callback")(&request, error);
        event.outcome.set(Err(status.error));
        Ok(())
    }

    fn effect_publish_error_noop(&mut self, event: EventLoadRuntime<'_>) -> Result<(), ()> {
        event.outcome.set(Err(event.status.get().error));
        Ok(())
    }

    fn effect_internal_error(&mut self, event: EventLoadRuntime<'_>) -> Result<(), ()> {
        event.status.set(LoadStatus::error(Error::InternalError));
        Ok(())
    }

    fn effect_unexpected(&mut self) -> Result<(), ()> {
        Err(())
    }
}
