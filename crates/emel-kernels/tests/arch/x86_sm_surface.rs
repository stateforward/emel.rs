#![allow(missing_docs)]

const SURFACE: &str = include_str!("../../src/x86_64/mod.rs");

#[test]
fn x86_sm_surface_contains_no_generated_scaffold() {
    assert!(!SURFACE.contains("todo!"));
    assert!(!SURFACE.contains("TODO"));
    assert!(!SURFACE.contains("scaffold"));
    assert!(SURFACE.contains("X86Kernel"));
}

#[test]
#[cfg(target_arch = "x86_64")]
fn x86_sm_surface_reuses_maintained_router() {
    use emel_kernels::x86_64::X86Kernel;

    let Some(router) = X86Kernel::try_new() else {
        return;
    };
    assert!(router.is_ready());
}
