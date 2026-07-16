//! Private explicit mapper orchestration.

#![allow(
    clippy::derive_partial_eq_without_eq,
    reason = "stateforward-sml generated state tokens intentionally derive PartialEq"
)]

use core::cell::{Cell, RefCell};

use sml::sml;

use super::event::{
    AdviceRequest, Error, MapDone, MapTensor, MappingOperation, ReleaseMapping, WithMapping,
};
use super::platform::{Platform, PlatformError, Region, SetupError};

const MAX_FILE_INDEX: u16 = 65_534;
const MAX_MAPPING_BYTES: u64 = 1_u64 << 40;
const MAX_MAPPINGS: usize = 256;

struct Slot {
    tensor_id: i32,
    region: Option<Region>,
}

pub(super) struct Context<P: Platform> {
    platform: P,
    slots: [Slot; MAX_MAPPINGS],
    free_stack: [u32; MAX_MAPPINGS],
    free_count: usize,
}

impl<P: Platform> Context<P> {
    fn new(platform: P) -> Self {
        Self {
            platform,
            slots: core::array::from_fn(|_| Slot {
                tensor_id: -1,
                region: None,
            }),
            free_stack: core::array::from_fn(|index| {
                u32::try_from(MAX_MAPPINGS - 1 - index).expect("mapping capacity fits u32")
            }),
            free_count: MAX_MAPPINGS,
        }
    }
}

#[derive(Clone, Copy)]
pub(super) struct MapRuntime<'dispatch, 'data> {
    request: &'data MapTensor,
    handle: &'dispatch Cell<u32>,
    setup: &'dispatch RefCell<Option<Result<Region, SetupError>>>,
    cleanup: &'dispatch Cell<Option<Result<(), PlatformError>>>,
    result: &'dispatch Cell<Result<MapDone, Error>>,
}

#[derive(Clone, Copy)]
pub(super) struct ReleaseRuntime<'dispatch> {
    request: ReleaseMapping,
    region: &'dispatch RefCell<Option<Region>>,
    native: &'dispatch Cell<Option<Result<(), PlatformError>>>,
    result: &'dispatch Cell<Result<(), Error>>,
}

#[derive(Clone, Copy)]
pub(super) struct AdviceRuntime<'dispatch> {
    request: AdviceRequest,
    region: &'dispatch RefCell<Option<Region>>,
    native: &'dispatch Cell<Option<Result<(), PlatformError>>>,
    result: &'dispatch Cell<Result<(), Error>>,
}

#[derive(Clone, Copy)]
pub(super) struct AccessRequest {
    tensor_id: i32,
    handle: u32,
}

sml! {
    IoMmap {
        // Pure request validation; setup is structurally unreachable here.
        "state_request_decision"_s <= *"state_ready"_s + Map(MapRuntime<'dispatch, 'data>),
        "state_resource_decision"_s <= "state_request_decision"_s + completion<Map>(MapRuntime<'dispatch, 'data>) [guard_request_valid],
        "state_ready"_s <= "state_request_decision"_s + completion<Map>(MapRuntime<'dispatch, 'data>) [guard_request_invalid] / effect_invalid_request,
        "state_platform_decision"_s <= "state_resource_decision"_s + completion<Map>(MapRuntime<'dispatch, 'data>) [guard_resource_valid],
        "state_ready"_s <= "state_resource_decision"_s + completion<Map>(MapRuntime<'dispatch, 'data>) [guard_resource_invalid] / effect_unsupported_resource,
        "state_capacity_decision"_s <= "state_platform_decision"_s + completion<Map>(MapRuntime<'dispatch, 'data>) [guard_platform_supported],
        "state_ready"_s <= "state_platform_decision"_s + completion<Map>(MapRuntime<'dispatch, 'data>) [guard_platform_unsupported] / effect_unsupported_platform,

        // Capacity-selected native setup and explicit result classification.
        "state_setup_decision"_s <= "state_capacity_decision"_s + completion<Map>(MapRuntime<'dispatch, 'data>) [guard_capacity_available] / effect_reserve_and_prepare_mapping,
        "state_ready"_s <= "state_capacity_decision"_s + completion<Map>(MapRuntime<'dispatch, 'data>) [guard_capacity_exhausted] / effect_resource_exhausted_before_map,
        "state_setup_view_decision"_s <= "state_setup_decision"_s + completion<Map>(MapRuntime<'dispatch, 'data>) [guard_setup_native_succeeded],
        "state_ready"_s <= "state_setup_decision"_s + completion<Map>(MapRuntime<'dispatch, 'data>) [guard_setup_mapping_failed] / effect_mapping_failed,
        "state_ready"_s <= "state_setup_decision"_s + completion<Map>(MapRuntime<'dispatch, 'data>) [guard_setup_platform_failed] / effect_unsupported_platform_setup,
        "state_ready"_s <= "state_setup_view_decision"_s + completion<Map>(MapRuntime<'dispatch, 'data>) [guard_setup_view_representable] / effect_commit_mapping,
        "state_setup_cleanup_decision"_s <= "state_setup_view_decision"_s + completion<Map>(MapRuntime<'dispatch, 'data>) [guard_setup_view_unrepresentable] / effect_release_unrepresentable_mapping,
        "state_ready"_s <= "state_setup_cleanup_decision"_s + completion<Map>(MapRuntime<'dispatch, 'data>) [guard_setup_cleanup_succeeded] / effect_finish_unrepresentable_mapping,
        "state_ready"_s <= "state_setup_cleanup_decision"_s + completion<Map>(MapRuntime<'dispatch, 'data>) [guard_setup_cleanup_failed] / effect_abort_unrepresentable_cleanup,

        // Release ownership transfer and native result classification.
        "state_release_owner_decision"_s <= "state_ready"_s + Release(ReleaseRuntime<'dispatch>),
        "state_release_native_decision"_s <= "state_release_owner_decision"_s + completion<Release>(ReleaseRuntime<'dispatch>) [guard_release_owned] / effect_release_native,
        "state_ready"_s <= "state_release_owner_decision"_s + completion<Release>(ReleaseRuntime<'dispatch>) [guard_release_invalid] / effect_release_invalid,
        "state_ready"_s <= "state_release_native_decision"_s + completion<Release>(ReleaseRuntime<'dispatch>) [guard_release_succeeded] / effect_finish_release,
        "state_ready"_s <= "state_release_native_decision"_s + completion<Release>(ReleaseRuntime<'dispatch>) [guard_release_failed] / effect_restore_release_failure,

        // Sequential advice validation and ownership transfer.
        "state_sequential_owner_decision"_s <= "state_ready"_s + Sequential(AdviceRuntime<'dispatch>),
        "state_sequential_range_decision"_s <= "state_sequential_owner_decision"_s + completion<Sequential>(AdviceRuntime<'dispatch>) [guard_sequential_owned],
        "state_ready"_s <= "state_sequential_owner_decision"_s + completion<Sequential>(AdviceRuntime<'dispatch>) [guard_sequential_invalid_owner] / effect_sequential_invalid_owner,
        "state_sequential_native_decision"_s <= "state_sequential_range_decision"_s + completion<Sequential>(AdviceRuntime<'dispatch>) [guard_sequential_range_valid] / effect_advise_sequential_native,
        "state_ready"_s <= "state_sequential_range_decision"_s + completion<Sequential>(AdviceRuntime<'dispatch>) [guard_sequential_range_invalid] / effect_sequential_invalid_range,
        "state_ready"_s <= "state_sequential_native_decision"_s + completion<Sequential>(AdviceRuntime<'dispatch>) [guard_advice_succeeded] / effect_restore_advice_success,
        "state_ready"_s <= "state_sequential_native_decision"_s + completion<Sequential>(AdviceRuntime<'dispatch>) [guard_advice_failed] / effect_restore_advice_failure,

        // Will-need advice validation and ownership transfer.
        "state_will_need_owner_decision"_s <= "state_ready"_s + WillNeed(AdviceRuntime<'dispatch>),
        "state_will_need_range_decision"_s <= "state_will_need_owner_decision"_s + completion<WillNeed>(AdviceRuntime<'dispatch>) [guard_will_need_owned],
        "state_ready"_s <= "state_will_need_owner_decision"_s + completion<WillNeed>(AdviceRuntime<'dispatch>) [guard_will_need_invalid_owner] / effect_will_need_invalid_owner,
        "state_will_need_native_decision"_s <= "state_will_need_range_decision"_s + completion<WillNeed>(AdviceRuntime<'dispatch>) [guard_will_need_range_valid] / effect_advise_will_need_native,
        "state_ready"_s <= "state_will_need_range_decision"_s + completion<WillNeed>(AdviceRuntime<'dispatch>) [guard_will_need_range_invalid] / effect_will_need_invalid_range,
        "state_ready"_s <= "state_will_need_native_decision"_s + completion<WillNeed>(AdviceRuntime<'dispatch>) [guard_advice_succeeded] / effect_restore_advice_success,
        "state_ready"_s <= "state_will_need_native_decision"_s + completion<WillNeed>(AdviceRuntime<'dispatch>) [guard_advice_failed] / effect_restore_advice_failure,

        // Don't-need advice validation and ownership transfer.
        "state_dont_need_owner_decision"_s <= "state_ready"_s + DontNeed(AdviceRuntime<'dispatch>),
        "state_dont_need_range_decision"_s <= "state_dont_need_owner_decision"_s + completion<DontNeed>(AdviceRuntime<'dispatch>) [guard_dont_need_owned],
        "state_ready"_s <= "state_dont_need_owner_decision"_s + completion<DontNeed>(AdviceRuntime<'dispatch>) [guard_dont_need_invalid_owner] / effect_dont_need_invalid_owner,
        "state_dont_need_native_decision"_s <= "state_dont_need_range_decision"_s + completion<DontNeed>(AdviceRuntime<'dispatch>) [guard_dont_need_range_valid] / effect_advise_dont_need_native,
        "state_ready"_s <= "state_dont_need_range_decision"_s + completion<DontNeed>(AdviceRuntime<'dispatch>) [guard_dont_need_range_invalid] / effect_dont_need_invalid_range,
        "state_ready"_s <= "state_dont_need_native_decision"_s + completion<DontNeed>(AdviceRuntime<'dispatch>) [guard_advice_succeeded] / effect_restore_advice_success,
        "state_ready"_s <= "state_dont_need_native_decision"_s + completion<DontNeed>(AdviceRuntime<'dispatch>) [guard_advice_failed] / effect_restore_advice_failure,

        // Explicit unexpected events preserve every decision state.
        "state_ready"_s <= "state_ready"_s + unexpected_event<_> / effect_unexpected,
        "state_ready"_s <= "state_request_decision"_s + unexpected_event<_> / effect_unexpected,
        "state_ready"_s <= "state_resource_decision"_s + unexpected_event<_> / effect_unexpected,
        "state_ready"_s <= "state_platform_decision"_s + unexpected_event<_> / effect_unexpected,
        "state_ready"_s <= "state_setup_decision"_s + unexpected_event<_> / effect_unexpected,
        "state_ready"_s <= "state_setup_view_decision"_s + unexpected_event<_> / effect_unexpected,
        "state_ready"_s <= "state_setup_cleanup_decision"_s + unexpected_event<_> / effect_unexpected,
        "state_ready"_s <= "state_capacity_decision"_s + unexpected_event<_> / effect_unexpected,
        "state_ready"_s <= "state_release_owner_decision"_s + unexpected_event<_> / effect_unexpected,
        "state_ready"_s <= "state_release_native_decision"_s + unexpected_event<_> / effect_unexpected,
        "state_ready"_s <= "state_sequential_owner_decision"_s + unexpected_event<_> / effect_unexpected,
        "state_ready"_s <= "state_sequential_range_decision"_s + unexpected_event<_> / effect_unexpected,
        "state_ready"_s <= "state_sequential_native_decision"_s + unexpected_event<_> / effect_unexpected,
        "state_ready"_s <= "state_will_need_owner_decision"_s + unexpected_event<_> / effect_unexpected,
        "state_ready"_s <= "state_will_need_range_decision"_s + unexpected_event<_> / effect_unexpected,
        "state_ready"_s <= "state_will_need_native_decision"_s + unexpected_event<_> / effect_unexpected,
        "state_ready"_s <= "state_dont_need_owner_decision"_s + unexpected_event<_> / effect_unexpected,
        "state_ready"_s <= "state_dont_need_range_decision"_s + unexpected_event<_> / effect_unexpected,
        "state_ready"_s <= "state_dont_need_native_decision"_s + unexpected_event<_> / effect_unexpected,
    }
}

pub(super) struct AccessRuntime<'dispatch, Operation>
where
    Operation: MappingOperation,
{
    request: AccessRequest,
    operation: &'dispatch RefCell<&'dispatch mut Operation>,
    result: &'dispatch RefCell<Option<Result<Operation::Output, Error>>>,
}

impl<Operation> Clone for AccessRuntime<'_, Operation>
where
    Operation: MappingOperation,
{
    fn clone(&self) -> Self {
        *self
    }
}

impl<Operation> Copy for AccessRuntime<'_, Operation> where Operation: MappingOperation {}

struct AccessContext<'access, P: Platform> {
    mappings: &'access Context<P>,
}

sml! {
    IoMmapAccess<'dispatch, Operation>
    where
        Operation: MappingOperation + 'dispatch,
    {
        X <= *"state_owner_decision"_s + Access(AccessRuntime<'dispatch, Operation>) [guard_access_owned] / effect_invoke_access,
        X <= "state_owner_decision"_s + Access(AccessRuntime<'dispatch, Operation>) [guard_access_invalid] / effect_access_invalid,
        X <= "state_owner_decision"_s + unexpected_event<_> / effect_access_unexpected,
    }
}

impl<P> IoMmapAccessStateMachineContext for AccessContext<'_, P>
where
    P: Platform,
{
    fn guard_access_owned<'dispatch, Operation>(
        &self,
        event: &AccessRuntime<'dispatch, Operation>,
    ) -> Result<bool, ()>
    where
        Operation: MappingOperation + 'dispatch,
    {
        Ok(self
            .mappings
            .owned(event.request.tensor_id, event.request.handle))
    }

    fn guard_access_invalid<'dispatch, Operation>(
        &self,
        event: &AccessRuntime<'dispatch, Operation>,
    ) -> Result<bool, ()>
    where
        Operation: MappingOperation + 'dispatch,
    {
        Ok(!self
            .mappings
            .owned(event.request.tensor_id, event.request.handle))
    }

    fn effect_invoke_access<'dispatch, Operation>(
        &mut self,
        event: AccessRuntime<'dispatch, Operation>,
    ) -> Result<(), ()>
    where
        Operation: MappingOperation + 'dispatch,
    {
        let bytes = self.mappings.slots[event.request.handle as usize]
            .region
            .as_ref()
            .expect("access guard selected an owned live mapping")
            .bytes();
        let output = event.operation.borrow_mut().apply(bytes);
        event.result.replace(Some(Ok(output)));
        Ok(())
    }

    fn effect_access_invalid<'dispatch, Operation>(
        &mut self,
        event: AccessRuntime<'dispatch, Operation>,
    ) -> Result<(), ()>
    where
        Operation: MappingOperation + 'dispatch,
    {
        event.result.replace(Some(Err(Error::InvalidRequest)));
        Ok(())
    }

    fn effect_access_unexpected(&mut self) -> Result<(), ()> {
        Err(())
    }
}

pub(super) struct MapperCore<P: Platform> {
    machine: IoMmapStateMachine<Context<P>>,
}

impl<P: Platform> MapperCore<P> {
    pub(super) fn new(platform: P) -> Self {
        Self {
            machine: IoMmapStateMachine::new(Context::new(platform)),
        }
    }

    pub(super) fn map(&mut self, request: &MapTensor) -> Result<MapDone, Error> {
        let handle = Cell::new(0);
        let setup = RefCell::new(None);
        let cleanup = Cell::new(None);
        let result = Cell::new(Err(Error::InternalError));
        self.machine
            .process_event(IoMmapEvents::Map(MapRuntime {
                request,
                handle: &handle,
                setup: &setup,
                cleanup: &cleanup,
                result: &result,
            }))
            .expect("mmap SML callbacks are infallible");
        result.get()
    }

    pub(super) fn release(&mut self, request: ReleaseMapping) -> Result<(), Error> {
        let region = RefCell::new(None);
        let native = Cell::new(None);
        let result = Cell::new(Err(Error::InternalError));
        self.machine
            .process_event(IoMmapEvents::Release(ReleaseRuntime {
                request,
                region: &region,
                native: &native,
                result: &result,
            }))
            .expect("mmap SML callbacks are infallible");
        result.get()
    }

    pub(super) fn advise_sequential(&mut self, request: AdviceRequest) -> Result<(), Error> {
        let region = RefCell::new(None);
        let native = Cell::new(None);
        let result = Cell::new(Err(Error::InternalError));
        self.machine
            .process_event(IoMmapEvents::Sequential(AdviceRuntime {
                request,
                region: &region,
                native: &native,
                result: &result,
            }))
            .expect("mmap SML callbacks are infallible");
        result.get()
    }

    pub(super) fn advise_will_need(&mut self, request: AdviceRequest) -> Result<(), Error> {
        let region = RefCell::new(None);
        let native = Cell::new(None);
        let result = Cell::new(Err(Error::InternalError));
        self.machine
            .process_event(IoMmapEvents::WillNeed(AdviceRuntime {
                request,
                region: &region,
                native: &native,
                result: &result,
            }))
            .expect("mmap SML callbacks are infallible");
        result.get()
    }

    pub(super) fn advise_dont_need(&mut self, request: AdviceRequest) -> Result<(), Error> {
        let region = RefCell::new(None);
        let native = Cell::new(None);
        let result = Cell::new(Err(Error::InternalError));
        self.machine
            .process_event(IoMmapEvents::DontNeed(AdviceRuntime {
                request,
                region: &region,
                native: &native,
                result: &result,
            }))
            .expect("mmap SML callbacks are infallible");
        result.get()
    }

    #[allow(
        clippy::needless_pass_by_ref_mut,
        clippy::needless_pass_by_value,
        reason = "the event is consumed at the single-writer actor boundary even though immutable mapped bytes satisfy this operation"
    )]
    pub(super) fn with_mapping<Operation>(
        &mut self,
        request: WithMapping<'_, Operation>,
    ) -> Result<Operation::Output, Error>
    where
        Operation: MappingOperation,
    {
        let operation = RefCell::new(request.operation);
        let result = RefCell::new(None);
        let access = AccessContext {
            mappings: self.machine.context(),
        };
        let mut machine = IoMmapAccessStateMachine::new(access);
        machine
            .process_event(IoMmapAccessEvents::Access(AccessRuntime {
                request: AccessRequest {
                    tensor_id: request.tensor_id,
                    handle: request.handle,
                },
                operation: &operation,
                result: &result,
            }))
            .expect("mmap access SML callbacks are infallible");
        result
            .into_inner()
            .expect("mapped access SML stores one typed outcome")
    }
}

impl<P: Platform> IoMmapStateMachineContext for Context<P> {
    fn guard_request_valid(&self, event: &MapRuntime<'_, '_>) -> Result<bool, ()> {
        Ok(event.request.len > 0)
    }
    fn guard_request_invalid(&self, event: &MapRuntime<'_, '_>) -> Result<bool, ()> {
        Ok(event.request.len == 0)
    }
    fn guard_resource_valid(&self, event: &MapRuntime<'_, '_>) -> Result<bool, ()> {
        Ok(event.request.file_index <= MAX_FILE_INDEX
            && event.request.len <= MAX_MAPPING_BYTES
            && mapping_view_len_supported(event.request.len)
            && usize::try_from(event.request.len).is_ok()
            && event
                .request
                .offset
                .checked_add(event.request.len)
                .is_some_and(|end| end <= event.request.file.len())
            && self.platform.offset_supported(event.request.offset)
            && event
                .request
                .offset
                .is_multiple_of(self.platform.required_alignment()))
    }
    fn guard_resource_invalid(&self, event: &MapRuntime<'_, '_>) -> Result<bool, ()> {
        Ok(!self.guard_resource_valid(event)?)
    }
    fn guard_platform_supported(&self, _: &MapRuntime<'_, '_>) -> Result<bool, ()> {
        Ok(self.platform.supported())
    }
    fn guard_platform_unsupported(&self, _: &MapRuntime<'_, '_>) -> Result<bool, ()> {
        Ok(!self.platform.supported())
    }
    fn effect_invalid_request(&mut self, event: MapRuntime<'_, '_>) -> Result<(), ()> {
        event.result.set(Err(Error::InvalidRequest));
        Ok(())
    }
    fn effect_unsupported_resource(&mut self, event: MapRuntime<'_, '_>) -> Result<(), ()> {
        event.result.set(Err(Error::UnsupportedResource));
        Ok(())
    }
    fn effect_unsupported_platform(&mut self, event: MapRuntime<'_, '_>) -> Result<(), ()> {
        event.result.set(Err(Error::UnsupportedPlatform));
        Ok(())
    }
    fn guard_capacity_available(&self, _: &MapRuntime<'_, '_>) -> Result<bool, ()> {
        Ok(self.free_count > 0)
    }
    fn guard_capacity_exhausted(&self, _: &MapRuntime<'_, '_>) -> Result<bool, ()> {
        Ok(self.free_count == 0)
    }
    fn effect_reserve_and_prepare_mapping(&mut self, event: MapRuntime<'_, '_>) -> Result<(), ()> {
        self.free_count -= 1;
        let handle = self.free_stack[self.free_count];
        self.slots[handle as usize] = Slot {
            tensor_id: event.request.tensor_id,
            region: None,
        };
        event.handle.set(handle);
        let len = usize::try_from(event.request.len)
            .expect("resource guard selected a mapping length that fits usize");
        event.setup.replace(Some(self.platform.prepare(
            &event.request.file,
            event.request.offset,
            len,
        )));
        Ok(())
    }
    fn effect_resource_exhausted_before_map(
        &mut self,
        event: MapRuntime<'_, '_>,
    ) -> Result<(), ()> {
        event.result.set(Err(Error::ResourceExhausted));
        Ok(())
    }
    fn guard_setup_native_succeeded(&self, event: &MapRuntime<'_, '_>) -> Result<bool, ()> {
        Ok(matches!(event.setup.borrow().as_ref(), Some(Ok(_))))
    }
    fn guard_setup_mapping_failed(&self, event: &MapRuntime<'_, '_>) -> Result<bool, ()> {
        Ok(matches!(
            event.setup.borrow().as_ref(),
            Some(Err(SetupError::MappingFailed))
        ))
    }
    fn guard_setup_platform_failed(&self, event: &MapRuntime<'_, '_>) -> Result<bool, ()> {
        Ok(matches!(
            event.setup.borrow().as_ref(),
            Some(Err(SetupError::UnsupportedPlatform))
        ))
    }
    fn effect_mapping_failed(&mut self, event: MapRuntime<'_, '_>) -> Result<(), ()> {
        self.free_reservation(event.handle.get());
        event.result.set(Err(Error::MappingFailed));
        Ok(())
    }
    fn effect_unsupported_platform_setup(&mut self, event: MapRuntime<'_, '_>) -> Result<(), ()> {
        self.free_reservation(event.handle.get());
        event.result.set(Err(Error::UnsupportedPlatform));
        Ok(())
    }
    fn guard_setup_view_representable(&self, event: &MapRuntime<'_, '_>) -> Result<bool, ()> {
        Ok(matches!(
            event.setup.borrow().as_ref(),
            Some(Ok(region)) if region.representable()
        ))
    }
    fn guard_setup_view_unrepresentable(&self, event: &MapRuntime<'_, '_>) -> Result<bool, ()> {
        Ok(matches!(
            event.setup.borrow().as_ref(),
            Some(Ok(region)) if !region.representable()
        ))
    }
    fn effect_release_unrepresentable_mapping(
        &mut self,
        event: MapRuntime<'_, '_>,
    ) -> Result<(), ()> {
        let mut setup = event.setup.borrow_mut();
        let region = setup
            .as_mut()
            .expect("native setup outcome remains present during classification")
            .as_mut()
            .expect("native-success guard selected a region");
        event.cleanup.set(Some(self.platform.release(region)));
        Ok(())
    }
    fn guard_setup_cleanup_succeeded(&self, event: &MapRuntime<'_, '_>) -> Result<bool, ()> {
        Ok(matches!(event.cleanup.get(), Some(Ok(()))))
    }
    fn guard_setup_cleanup_failed(&self, event: &MapRuntime<'_, '_>) -> Result<bool, ()> {
        Ok(matches!(event.cleanup.get(), Some(Err(_))))
    }
    fn effect_finish_unrepresentable_mapping(
        &mut self,
        event: MapRuntime<'_, '_>,
    ) -> Result<(), ()> {
        self.free_reservation(event.handle.get());
        event.setup.borrow_mut().take();
        event.result.set(Err(Error::MappingFailed));
        Ok(())
    }
    fn effect_abort_unrepresentable_cleanup(&mut self, _: MapRuntime<'_, '_>) -> Result<(), ()> {
        std::process::abort()
    }
    fn effect_commit_mapping(&mut self, event: MapRuntime<'_, '_>) -> Result<(), ()> {
        let region = event
            .setup
            .borrow_mut()
            .take()
            .expect("setup outcome is present during synchronous completion")
            .expect("success guard selected the successful setup outcome");
        let handle = event.handle.get();
        let index = handle as usize;
        self.slots[index].region = Some(region);
        event.result.set(Ok(MapDone::new(
            handle,
            event.request.tensor_id,
            event.request.len,
        )));
        Ok(())
    }
    fn guard_release_owned(&self, event: &ReleaseRuntime<'_>) -> Result<bool, ()> {
        Ok(self.owned(event.request.tensor_id, event.request.handle))
    }
    fn guard_release_invalid(&self, event: &ReleaseRuntime<'_>) -> Result<bool, ()> {
        Ok(!self.owned(event.request.tensor_id, event.request.handle))
    }
    fn effect_release_native(&mut self, event: ReleaseRuntime<'_>) -> Result<(), ()> {
        let mut region = self.take_region(event.request.handle);
        event.native.set(Some(self.platform.release(&mut region)));
        event.region.replace(Some(region));
        Ok(())
    }
    fn effect_release_invalid(&mut self, event: ReleaseRuntime<'_>) -> Result<(), ()> {
        event.result.set(Err(Error::InvalidRequest));
        Ok(())
    }
    fn guard_release_succeeded(&self, event: &ReleaseRuntime<'_>) -> Result<bool, ()> {
        Ok(matches!(event.native.get(), Some(Ok(()))))
    }
    fn guard_release_failed(&self, event: &ReleaseRuntime<'_>) -> Result<bool, ()> {
        Ok(matches!(event.native.get(), Some(Err(_))))
    }
    fn effect_finish_release(&mut self, event: ReleaseRuntime<'_>) -> Result<(), ()> {
        let index = event.request.handle as usize;
        self.slots[index].tensor_id = -1;
        self.free_stack[self.free_count] = event.request.handle;
        self.free_count += 1;
        event.region.borrow_mut().take();
        event.result.set(Ok(()));
        Ok(())
    }
    fn effect_restore_release_failure(&mut self, event: ReleaseRuntime<'_>) -> Result<(), ()> {
        self.restore_region(event.request.handle, event.region);
        event.result.set(Err(Error::UnmapFailed));
        Ok(())
    }
    fn guard_sequential_owned(&self, event: &AdviceRuntime<'_>) -> Result<bool, ()> {
        Ok(self.owned(event.request.tensor_id, event.request.handle))
    }
    fn guard_sequential_invalid_owner(&self, event: &AdviceRuntime<'_>) -> Result<bool, ()> {
        Ok(!self.owned(event.request.tensor_id, event.request.handle))
    }
    fn guard_sequential_range_valid(&self, event: &AdviceRuntime<'_>) -> Result<bool, ()> {
        Ok(self.advice_range_valid(event.request))
    }
    fn guard_sequential_range_invalid(&self, event: &AdviceRuntime<'_>) -> Result<bool, ()> {
        Ok(!self.advice_range_valid(event.request))
    }
    fn effect_sequential_invalid_owner(&mut self, event: AdviceRuntime<'_>) -> Result<(), ()> {
        event.result.set(Err(Error::InvalidRequest));
        Ok(())
    }
    fn effect_sequential_invalid_range(&mut self, event: AdviceRuntime<'_>) -> Result<(), ()> {
        event.result.set(Err(Error::InvalidAdviceRange));
        Ok(())
    }
    fn effect_advise_sequential_native(&mut self, event: AdviceRuntime<'_>) -> Result<(), ()> {
        let region = self.take_region(event.request.handle);
        let offset = usize::try_from(event.request.offset)
            .expect("range guard selected an advice offset that fits usize");
        let len = usize::try_from(event.request.len)
            .expect("range guard selected an advice length that fits usize");
        event
            .native
            .set(Some(self.platform.advise_sequential(&region, offset, len)));
        event.region.replace(Some(region));
        Ok(())
    }

    fn guard_will_need_owned(&self, event: &AdviceRuntime<'_>) -> Result<bool, ()> {
        Ok(self.owned(event.request.tensor_id, event.request.handle))
    }
    fn guard_will_need_invalid_owner(&self, event: &AdviceRuntime<'_>) -> Result<bool, ()> {
        Ok(!self.owned(event.request.tensor_id, event.request.handle))
    }
    fn guard_will_need_range_valid(&self, event: &AdviceRuntime<'_>) -> Result<bool, ()> {
        Ok(self.advice_range_valid(event.request))
    }
    fn guard_will_need_range_invalid(&self, event: &AdviceRuntime<'_>) -> Result<bool, ()> {
        Ok(!self.advice_range_valid(event.request))
    }
    fn effect_will_need_invalid_owner(&mut self, event: AdviceRuntime<'_>) -> Result<(), ()> {
        event.result.set(Err(Error::InvalidRequest));
        Ok(())
    }
    fn effect_will_need_invalid_range(&mut self, event: AdviceRuntime<'_>) -> Result<(), ()> {
        event.result.set(Err(Error::InvalidAdviceRange));
        Ok(())
    }
    fn effect_advise_will_need_native(&mut self, event: AdviceRuntime<'_>) -> Result<(), ()> {
        let region = self.take_region(event.request.handle);
        let offset = usize::try_from(event.request.offset)
            .expect("range guard selected an advice offset that fits usize");
        let len = usize::try_from(event.request.len)
            .expect("range guard selected an advice length that fits usize");
        event
            .native
            .set(Some(self.platform.advise_will_need(&region, offset, len)));
        event.region.replace(Some(region));
        Ok(())
    }

    fn guard_dont_need_owned(&self, event: &AdviceRuntime<'_>) -> Result<bool, ()> {
        Ok(self.owned(event.request.tensor_id, event.request.handle))
    }
    fn guard_dont_need_invalid_owner(&self, event: &AdviceRuntime<'_>) -> Result<bool, ()> {
        Ok(!self.owned(event.request.tensor_id, event.request.handle))
    }
    fn guard_dont_need_range_valid(&self, event: &AdviceRuntime<'_>) -> Result<bool, ()> {
        Ok(self.advice_range_valid(event.request))
    }
    fn guard_dont_need_range_invalid(&self, event: &AdviceRuntime<'_>) -> Result<bool, ()> {
        Ok(!self.advice_range_valid(event.request))
    }
    fn effect_dont_need_invalid_owner(&mut self, event: AdviceRuntime<'_>) -> Result<(), ()> {
        event.result.set(Err(Error::InvalidRequest));
        Ok(())
    }
    fn effect_dont_need_invalid_range(&mut self, event: AdviceRuntime<'_>) -> Result<(), ()> {
        event.result.set(Err(Error::InvalidAdviceRange));
        Ok(())
    }
    fn effect_advise_dont_need_native(&mut self, event: AdviceRuntime<'_>) -> Result<(), ()> {
        let region = self.take_region(event.request.handle);
        let offset = usize::try_from(event.request.offset)
            .expect("range guard selected an advice offset that fits usize");
        let len = usize::try_from(event.request.len)
            .expect("range guard selected an advice length that fits usize");
        event
            .native
            .set(Some(self.platform.advise_dont_need(&region, offset, len)));
        event.region.replace(Some(region));
        Ok(())
    }
    fn guard_advice_succeeded(&self, event: &AdviceRuntime<'_>) -> Result<bool, ()> {
        Ok(matches!(event.native.get(), Some(Ok(()))))
    }
    fn guard_advice_failed(&self, event: &AdviceRuntime<'_>) -> Result<bool, ()> {
        Ok(matches!(event.native.get(), Some(Err(_))))
    }
    fn effect_restore_advice_success(&mut self, event: AdviceRuntime<'_>) -> Result<(), ()> {
        self.restore_region(event.request.handle, event.region);
        event.result.set(Ok(()));
        Ok(())
    }
    fn effect_restore_advice_failure(&mut self, event: AdviceRuntime<'_>) -> Result<(), ()> {
        self.restore_region(event.request.handle, event.region);
        event.result.set(Err(Error::AdviceFailed));
        Ok(())
    }
    fn effect_unexpected(&mut self) -> Result<(), ()> {
        std::process::abort()
    }
}

pub(super) const fn mapping_view_len_supported(len: u64) -> bool {
    len <= isize::MAX as u64
}

impl<P: Platform> Context<P> {
    const fn owned(&self, tensor_id: i32, handle: u32) -> bool {
        let index = handle as usize;
        index < MAX_MAPPINGS
            && self.slots[index].tensor_id == tensor_id
            && self.slots[index].region.is_some()
    }

    const fn take_region(&mut self, handle: u32) -> Region {
        self.slots[handle as usize]
            .region
            .take()
            .expect("ownership guard selected a live mapping")
    }

    const fn free_reservation(&mut self, handle: u32) {
        let index = handle as usize;
        self.slots[index].tensor_id = -1;
        self.free_stack[self.free_count] = handle;
        self.free_count += 1;
    }

    fn advice_range_valid(&self, request: AdviceRequest) -> bool {
        let mapped_len = self.slots[request.handle as usize]
            .region
            .as_ref()
            .map_or(0, Region::len) as u64;
        request.len > 0
            && usize::try_from(request.offset).is_ok()
            && usize::try_from(request.len).is_ok()
            && request.offset <= mapped_len
            && request.len <= mapped_len - request.offset
    }

    fn restore_region(&mut self, handle: u32, region: &RefCell<Option<Region>>) {
        let index = handle as usize;
        self.slots[index].region = Some(
            region
                .borrow_mut()
                .take()
                .expect("finish event retains exclusive region ownership"),
        );
    }
}
