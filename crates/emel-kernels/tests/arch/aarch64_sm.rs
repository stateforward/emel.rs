#![allow(missing_docs)]
#![cfg(target_arch = "aarch64")]

use emel_kernels::aarch64::{Kernel, KernelError, UnexpectedAarch64Kernel};

const MODULE_SOURCE: &str = include_str!("../../src/aarch64/mod.rs");

#[test]
fn aarch64_sm_is_not_a_generated_scaffold() {
    assert!(!MODULE_SOURCE.contains("TODO"));
    assert!(!MODULE_SOURCE.contains("todo!"));
    assert!(!MODULE_SOURCE.contains("unsafe"));
    assert!(!MODULE_SOURCE.contains("scaffold"));
}

#[test]
fn aarch64_sm_exposes_the_maintained_router() {
    let Some(mut kernel) = Kernel::try_new() else {
        return;
    };

    assert!(kernel.is_ready());
    assert_eq!(
        kernel.process_event(UnexpectedAarch64Kernel, &mut []),
        Err(KernelError::UnexpectedEvent)
    );
    assert!(kernel.is_ready());
}
