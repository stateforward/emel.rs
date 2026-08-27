#![allow(missing_docs)]

//! Compile-time coverage for the target routers' explicit target-only surface.
//! Portable events are intentionally not accepted by this API: a target actor
//! must never disguise a portable child dispatch as target execution.

#[cfg(target_arch = "aarch64")]
use emel_kernels::aarch64::{KernelEvent, UnexpectedAarch64Kernel};
#[cfg(target_arch = "x86_64")]
use emel_kernels::x86_64::{UnexpectedX86Kernel, event::KernelEvent};

#[cfg(target_arch = "aarch64")]
use emel_kernels::aarch64::{Dup, Kernel as TargetKernel};
#[cfg(target_arch = "x86_64")]
use emel_kernels::x86_64::{Kernel as TargetKernel, X86Dup};

const fn assert_target_event<E: KernelEvent>() {}

#[test]
fn target_surface_has_explicit_event_and_unexpected_route() {
    #[cfg(target_arch = "aarch64")]
    assert_target_event::<Dup<'static>>();
    #[cfg(target_arch = "x86_64")]
    assert_target_event::<X86Dup<'static>>();
}

#[test]
fn target_unexpected_event_is_explicit() {
    let mut actor = TargetKernel::try_new().expect("target router");
    #[cfg(target_arch = "aarch64")]
    assert!(
        actor
            .process_event(UnexpectedAarch64Kernel, &mut [])
            .is_err()
    );
    #[cfg(target_arch = "x86_64")]
    assert!(actor.process_event(UnexpectedX86Kernel, &mut []).is_err());
}
