//! Explicit SML lifecycle for the GGUF loader.

#![allow(
    clippy::derive_partial_eq_without_eq,
    clippy::needless_lifetimes,
    reason = "SML-generated action and guard signatures mirror structural event lifetimes"
)]

use sml::sml;

use super::{Error, Gguf, KvEntry, Requirements, TensorInfo, detail};

#[derive(Clone, Copy, Debug)]
pub(super) struct EventBindRuntime {
    pub(super) kv_arena_bytes: usize,
    pub(super) kv_entry_capacity: usize,
    pub(super) tensor_capacity: usize,
}

sml! {
    GgufLoader {
        "probe_execution_pending"_s <= *"uninitialized"_s + ProbeRequest(&'a [u8]) [probe_request_valid] / begin_probe,
        "probe_execution_pending"_s <= "probed"_s + ProbeRequest(&'a [u8]) [probe_request_valid] / begin_probe,
        "probe_execution_pending"_s <= "bound"_s + ProbeRequest(&'a [u8]) [probe_request_valid] / begin_probe,
        "probe_execution_pending"_s <= "parsed"_s + ProbeRequest(&'a [u8]) [probe_request_valid] / begin_probe,
        "probe_execution_pending"_s <= "errored"_s + ProbeRequest(&'a [u8]) [probe_request_valid] / begin_probe,
        "errored"_s <= "uninitialized"_s + ProbeRequest(&'a [u8]) [probe_request_invalid] / mark_probe_invalid,
        "errored"_s <= "probed"_s + ProbeRequest(&'a [u8]) [probe_request_invalid] / mark_probe_invalid,
        "errored"_s <= "bound"_s + ProbeRequest(&'a [u8]) [probe_request_invalid] / mark_probe_invalid,
        "errored"_s <= "parsed"_s + ProbeRequest(&'a [u8]) [probe_request_invalid] / mark_probe_invalid,
        "errored"_s <= "errored"_s + ProbeRequest(&'a [u8]) [probe_request_invalid] / mark_probe_invalid,

        "probed"_s <= "probe_execution_pending"_s + ProbeResult(Result<Requirements, Error>) [probe_result_ok] / commit_probe,
        "errored"_s <= "probe_execution_pending"_s + ProbeResult(Result<Requirements, Error>) [probe_result_invalid_request] / publish_probe_error,
        "errored"_s <= "probe_execution_pending"_s + ProbeResult(Result<Requirements, Error>) [probe_result_model_invalid] / publish_probe_error,
        "errored"_s <= "probe_execution_pending"_s + ProbeResult(Result<Requirements, Error>) [probe_result_capacity] / publish_probe_error,
        "errored"_s <= "probe_execution_pending"_s + ProbeResult(Result<Requirements, Error>) [probe_result_parse_failed] / publish_probe_error,
        "errored"_s <= "probe_execution_pending"_s + ProbeResult(Result<Requirements, Error>) [probe_result_internal] / publish_probe_error,
        "errored"_s <= "probe_execution_pending"_s + ProbeResult(Result<Requirements, Error>) [probe_result_untracked] / publish_probe_error,

        "bind_capacity_decision"_s <= "probed"_s + BindRequest(EventBindRuntime) / begin_bind,
        "bind_capacity_decision"_s <= "bound"_s + BindRequest(EventBindRuntime) / begin_bind,
        "bind_capacity_decision"_s <= "parsed"_s + BindRequest(EventBindRuntime) / begin_bind,
        "errored"_s <= "uninitialized"_s + BindRequest(EventBindRuntime) / mark_bind_invalid,
        "errored"_s <= "errored"_s + BindRequest(EventBindRuntime) / mark_bind_invalid,
        "bind_allocation_pending"_s <= "bind_capacity_decision"_s + completion<BindRequest>(EventBindRuntime) [bind_capacity_sufficient],
        "errored"_s <= "bind_capacity_decision"_s + completion<BindRequest>(EventBindRuntime) [bind_capacity_insufficient] / mark_bind_capacity,

        "bound"_s <= "bind_allocation_pending"_s + BindResult(Result<BoundStorage, Error>) [bind_result_ok] / commit_bind,
        "errored"_s <= "bind_allocation_pending"_s + BindResult(Result<BoundStorage, Error>) [bind_result_invalid_request] / publish_bind_error,
        "errored"_s <= "bind_allocation_pending"_s + BindResult(Result<BoundStorage, Error>) [bind_result_model_invalid] / publish_bind_error,
        "errored"_s <= "bind_allocation_pending"_s + BindResult(Result<BoundStorage, Error>) [bind_result_capacity] / publish_bind_error,
        "errored"_s <= "bind_allocation_pending"_s + BindResult(Result<BoundStorage, Error>) [bind_result_parse_failed] / publish_bind_error,
        "errored"_s <= "bind_allocation_pending"_s + BindResult(Result<BoundStorage, Error>) [bind_result_internal] / publish_bind_error,
        "errored"_s <= "bind_allocation_pending"_s + BindResult(Result<BoundStorage, Error>) [bind_result_untracked] / publish_bind_error,

        "parse_request_decision"_s <= "bound"_s + ParseRequest(&'a [u8]) / begin_parse,
        "parse_request_decision"_s <= "parsed"_s + ParseRequest(&'a [u8]) / begin_parse,
        "errored"_s <= "uninitialized"_s + ParseRequest(&'a [u8]) / mark_parse_invalid,
        "errored"_s <= "probed"_s + ParseRequest(&'a [u8]) / mark_parse_invalid,
        "errored"_s <= "errored"_s + ParseRequest(&'a [u8]) / mark_parse_invalid,
        "parse_bound_storage_decision"_s <= "parse_request_decision"_s + completion<ParseRequest>(&'a [u8]) [parse_has_file_image],
        "errored"_s <= "parse_request_decision"_s + completion<ParseRequest>(&'a [u8]) [parse_missing_file_image] / mark_parse_invalid,
        "parse_capacity_decision"_s <= "parse_bound_storage_decision"_s + completion<ParseRequest>(&'a [u8]) [parse_has_bound_storage],
        "errored"_s <= "parse_bound_storage_decision"_s + completion<ParseRequest>(&'a [u8]) [parse_missing_bound_storage] / mark_parse_invalid,
        "parse_execution_pending"_s <= "parse_capacity_decision"_s + completion<ParseRequest>(&'a [u8]) [parse_bound_capacity_sufficient],
        "errored"_s <= "parse_capacity_decision"_s + completion<ParseRequest>(&'a [u8]) [parse_bound_capacity_insufficient] / mark_parse_capacity,

        "parsed"_s <= "parse_execution_pending"_s + ParseResult(Result<(), Error>) [parse_result_ok],
        "errored"_s <= "parse_execution_pending"_s + ParseResult(Result<(), Error>) [parse_result_invalid_request] / publish_parse_error,
        "errored"_s <= "parse_execution_pending"_s + ParseResult(Result<(), Error>) [parse_result_model_invalid] / publish_parse_error,
        "errored"_s <= "parse_execution_pending"_s + ParseResult(Result<(), Error>) [parse_result_capacity] / publish_parse_error,
        "errored"_s <= "parse_execution_pending"_s + ParseResult(Result<(), Error>) [parse_result_parse_failed] / publish_parse_error,
        "errored"_s <= "parse_execution_pending"_s + ParseResult(Result<(), Error>) [parse_result_internal] / publish_parse_error,
        "errored"_s <= "parse_execution_pending"_s + ParseResult(Result<(), Error>) [parse_result_untracked] / publish_parse_error,

        "errored"_s <= "uninitialized"_s + unexpected_event<_> / on_unexpected,
        "errored"_s <= "probed"_s + unexpected_event<_> / on_unexpected,
        "errored"_s <= "bound"_s + unexpected_event<_> / on_unexpected,
        "errored"_s <= "parsed"_s + unexpected_event<_> / on_unexpected,
        "errored"_s <= "errored"_s + unexpected_event<_> / on_unexpected,
        "errored"_s <= "probe_execution_pending"_s + unexpected_event<_> / on_unexpected,
        "errored"_s <= "bind_capacity_decision"_s + unexpected_event<_> / on_unexpected,
        "errored"_s <= "bind_allocation_pending"_s + unexpected_event<_> / on_unexpected,
        "errored"_s <= "parse_request_decision"_s + unexpected_event<_> / on_unexpected,
        "errored"_s <= "parse_bound_storage_decision"_s + unexpected_event<_> / on_unexpected,
        "errored"_s <= "parse_capacity_decision"_s + unexpected_event<_> / on_unexpected,
        "errored"_s <= "parse_execution_pending"_s + unexpected_event<_> / on_unexpected,
    }
}

#[derive(Debug)]
pub(super) struct BoundStorage {
    kv_arena: Vec<u8>,
    kv_entries: Vec<KvEntry>,
    tensors: Vec<TensorInfo>,
}

impl BoundStorage {
    pub(super) fn allocate(event: EventBindRuntime) -> Result<Self, Error> {
        Ok(Self {
            kv_arena: zeroed_vec(event.kv_arena_bytes)?,
            kv_entries: zeroed_vec(event.kv_entry_capacity)?,
            tensors: zeroed_vec(event.tensor_capacity)?,
        })
    }
}

#[derive(Debug)]
pub(super) struct GgufLoaderContext {
    pub(super) probed: Requirements,
    error: Option<Error>,
    bound: Option<BoundStorage>,
}

impl GgufLoaderContext {
    pub(super) const fn new() -> Self {
        Self {
            probed: Requirements {
                tensor_count: 0,
                kv_count: 0,
                max_key_bytes: 0,
                max_value_bytes: 0,
                tensor_data_bytes: 0,
            },
            error: None,
            bound: None,
        }
    }

    pub(super) fn result(&self) -> Result<(), Error> {
        self.error.map_or(Ok(()), Err)
    }

    pub(super) fn execute_parse(&mut self, file_image: &[u8]) -> Result<(), Error> {
        let bound = self
            .bound
            .as_mut()
            .expect("parse execution state guarantees bound storage");
        detail::parse(
            file_image,
            self.probed,
            &mut bound.kv_arena,
            &mut bound.kv_entries,
            &mut bound.tensors,
        )
    }

    pub(super) fn parsed<'a>(&self, file_image: &'a [u8]) -> Result<Gguf<'a>, Error> {
        let bound = self.bound.as_ref().ok_or(Error::Internal)?;
        Ok(Gguf {
            file_image,
            requirements: self.probed,
            kv_arena: bound.kv_arena.clone(),
            kv_entries: bound.kv_entries
                [..usize::try_from(self.probed.kv_count).map_err(|_| Error::Capacity)?]
                .to_vec(),
            tensors: bound.tensors
                [..usize::try_from(self.probed.tensor_count).map_err(|_| Error::Capacity)?]
                .to_vec(),
        })
    }

    const fn begin(&mut self) {
        self.error = None;
    }

    const fn mark(&mut self, error: Error) {
        self.error = Some(error);
    }

    fn capacity_sufficient(&self, event: EventBindRuntime) -> bool {
        event.tensor_capacity >= self.probed.tensor_count as usize
            && event.kv_entry_capacity >= self.probed.kv_count as usize
            && self
                .probed
                .required_kv_arena_bytes()
                .is_ok_and(|required| event.kv_arena_bytes >= required)
    }

    fn bound_capacity_sufficient(&self) -> bool {
        self.bound.as_ref().is_some_and(|bound| {
            bound.tensors.len() >= self.probed.tensor_count as usize
                && bound.kv_entries.len() >= self.probed.kv_count as usize
                && self
                    .probed
                    .required_kv_arena_bytes()
                    .is_ok_and(|required| bound.kv_arena.len() >= required)
        })
    }
}

fn zeroed_vec<T: Clone + Default>(length: usize) -> Result<Vec<T>, Error> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(length)
        .map_err(|_| Error::Capacity)?;
    values.resize(length, T::default());
    Ok(values)
}

fn result_is<T>(result: &Result<T, Error>, expected: Error) -> bool {
    result.as_ref().err() == Some(&expected)
}

impl GgufLoaderStateMachineContext for GgufLoaderContext {
    fn begin_probe<'a>(&mut self, _event: &'a [u8]) -> Result<(), ()> {
        self.begin();
        Ok(())
    }

    fn probe_request_valid<'a>(&self, event: &'a [u8]) -> Result<bool, ()> {
        Ok(!event.is_empty())
    }

    fn probe_request_invalid<'a>(&self, event: &'a [u8]) -> Result<bool, ()> {
        Ok(event.is_empty())
    }

    fn mark_probe_invalid<'a>(&mut self, _event: &'a [u8]) -> Result<(), ()> {
        self.mark(Error::InvalidRequest);
        Ok(())
    }

    fn probe_result_ok(&self, event: &Result<Requirements, Error>) -> Result<bool, ()> {
        Ok(event.is_ok())
    }

    fn probe_result_invalid_request(
        &self,
        event: &Result<Requirements, Error>,
    ) -> Result<bool, ()> {
        Ok(result_is(event, Error::InvalidRequest))
    }

    fn probe_result_model_invalid(&self, event: &Result<Requirements, Error>) -> Result<bool, ()> {
        Ok(result_is(event, Error::ModelInvalid))
    }

    fn probe_result_capacity(&self, event: &Result<Requirements, Error>) -> Result<bool, ()> {
        Ok(result_is(event, Error::Capacity))
    }

    fn probe_result_parse_failed(&self, event: &Result<Requirements, Error>) -> Result<bool, ()> {
        Ok(result_is(event, Error::ParseFailed))
    }

    fn probe_result_internal(&self, event: &Result<Requirements, Error>) -> Result<bool, ()> {
        Ok(result_is(event, Error::Internal))
    }

    fn probe_result_untracked(&self, event: &Result<Requirements, Error>) -> Result<bool, ()> {
        Ok(result_is(event, Error::Untracked))
    }

    fn commit_probe(&mut self, event: Result<Requirements, Error>) -> Result<(), ()> {
        self.probed = event.expect("probe_result_ok guard guarantees requirements");
        self.bound = None;
        Ok(())
    }

    fn publish_probe_error(&mut self, event: Result<Requirements, Error>) -> Result<(), ()> {
        self.mark(event.expect_err("probe error guard guarantees an error"));
        Ok(())
    }

    fn begin_bind(&mut self, _event: EventBindRuntime) -> Result<(), ()> {
        self.begin();
        Ok(())
    }

    fn mark_bind_invalid(&mut self, _event: EventBindRuntime) -> Result<(), ()> {
        self.mark(Error::InvalidRequest);
        Ok(())
    }

    fn bind_capacity_sufficient(&self, event: &EventBindRuntime) -> Result<bool, ()> {
        Ok(self.capacity_sufficient(*event))
    }

    fn bind_capacity_insufficient(&self, event: &EventBindRuntime) -> Result<bool, ()> {
        Ok(!self.capacity_sufficient(*event))
    }

    fn mark_bind_capacity(&mut self, _event: EventBindRuntime) -> Result<(), ()> {
        self.mark(Error::Capacity);
        Ok(())
    }

    fn bind_result_ok(&self, event: &Result<BoundStorage, Error>) -> Result<bool, ()> {
        Ok(event.is_ok())
    }

    fn bind_result_invalid_request(&self, event: &Result<BoundStorage, Error>) -> Result<bool, ()> {
        Ok(result_is(event, Error::InvalidRequest))
    }

    fn bind_result_model_invalid(&self, event: &Result<BoundStorage, Error>) -> Result<bool, ()> {
        Ok(result_is(event, Error::ModelInvalid))
    }

    fn bind_result_capacity(&self, event: &Result<BoundStorage, Error>) -> Result<bool, ()> {
        Ok(result_is(event, Error::Capacity))
    }

    fn bind_result_parse_failed(&self, event: &Result<BoundStorage, Error>) -> Result<bool, ()> {
        Ok(result_is(event, Error::ParseFailed))
    }

    fn bind_result_internal(&self, event: &Result<BoundStorage, Error>) -> Result<bool, ()> {
        Ok(result_is(event, Error::Internal))
    }

    fn bind_result_untracked(&self, event: &Result<BoundStorage, Error>) -> Result<bool, ()> {
        Ok(result_is(event, Error::Untracked))
    }

    fn commit_bind(&mut self, event: Result<BoundStorage, Error>) -> Result<(), ()> {
        self.bound = Some(event.expect("bind_result_ok guard guarantees storage"));
        Ok(())
    }

    fn publish_bind_error(&mut self, event: Result<BoundStorage, Error>) -> Result<(), ()> {
        self.mark(event.expect_err("bind error guard guarantees an error"));
        Ok(())
    }

    fn begin_parse<'a>(&mut self, _event: &'a [u8]) -> Result<(), ()> {
        self.begin();
        Ok(())
    }

    fn mark_parse_invalid<'a>(&mut self, _event: &'a [u8]) -> Result<(), ()> {
        self.mark(Error::InvalidRequest);
        Ok(())
    }

    fn parse_has_file_image<'a>(&self, event: &'a [u8]) -> Result<bool, ()> {
        Ok(!event.is_empty())
    }

    fn parse_missing_file_image<'a>(&self, event: &'a [u8]) -> Result<bool, ()> {
        Ok(event.is_empty())
    }

    fn parse_has_bound_storage<'a>(&self, _event: &'a [u8]) -> Result<bool, ()> {
        Ok(self.bound.is_some())
    }

    fn parse_missing_bound_storage<'a>(&self, _event: &'a [u8]) -> Result<bool, ()> {
        Ok(self.bound.is_none())
    }

    fn parse_bound_capacity_sufficient<'a>(&self, _event: &'a [u8]) -> Result<bool, ()> {
        Ok(self.bound_capacity_sufficient())
    }

    fn parse_bound_capacity_insufficient<'a>(&self, _event: &'a [u8]) -> Result<bool, ()> {
        Ok(!self.bound_capacity_sufficient())
    }

    fn mark_parse_capacity<'a>(&mut self, _event: &'a [u8]) -> Result<(), ()> {
        self.mark(Error::Capacity);
        Ok(())
    }

    fn parse_result_ok(&self, event: &Result<(), Error>) -> Result<bool, ()> {
        Ok(event.is_ok())
    }

    fn parse_result_invalid_request(&self, event: &Result<(), Error>) -> Result<bool, ()> {
        Ok(result_is(event, Error::InvalidRequest))
    }

    fn parse_result_model_invalid(&self, event: &Result<(), Error>) -> Result<bool, ()> {
        Ok(result_is(event, Error::ModelInvalid))
    }

    fn parse_result_capacity(&self, event: &Result<(), Error>) -> Result<bool, ()> {
        Ok(result_is(event, Error::Capacity))
    }

    fn parse_result_parse_failed(&self, event: &Result<(), Error>) -> Result<bool, ()> {
        Ok(result_is(event, Error::ParseFailed))
    }

    fn parse_result_internal(&self, event: &Result<(), Error>) -> Result<bool, ()> {
        Ok(result_is(event, Error::Internal))
    }

    fn parse_result_untracked(&self, event: &Result<(), Error>) -> Result<bool, ()> {
        Ok(result_is(event, Error::Untracked))
    }

    fn publish_parse_error(&mut self, event: Result<(), Error>) -> Result<(), ()> {
        self.mark(event.expect_err("parse error guard guarantees an error"));
        Ok(())
    }

    fn on_unexpected(&mut self) -> Result<(), ()> {
        self.mark(Error::Internal);
        Ok(())
    }
}
