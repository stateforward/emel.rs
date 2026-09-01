//! Generic and architecture-specialized numerical kernels.

#![forbid(unsafe_code)]
#![allow(missing_docs, dead_code)]

use emel_tensor as _;
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
use pulp as _;

#[cfg(test)]
use allocation_counter as _;

pub mod any;

#[cfg(target_arch = "aarch64")]
pub mod aarch64;

#[cfg(target_arch = "x86_64")]
pub mod x86_64;

mod detail;

pub use any::{Any, Error, Kernel, RuntimeKernel};

/// A kernel kind available to the kernel package.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum KernelKind {
    /// `x86_64` implementation family.
    X86_64,
    /// aarch64 implementation family.
    Aarch64,
}

/// Returns the kernel kind selected for this build.
#[must_use]
pub const fn kernel_kind() -> KernelKind {
    if cfg!(target_arch = "x86_64") {
        KernelKind::X86_64
    } else if cfg!(target_arch = "aarch64") {
        KernelKind::Aarch64
    } else {
        // The maintained C++ surface has no portable kernel kind. Keep this
        // fallback aligned with its non-aarch64 host selection.
        KernelKind::X86_64
    }
}

/// Returns the maintained operation implementation family for this build.
#[must_use]
pub const fn dispatch_name() -> &'static str {
    if cfg!(target_arch = "x86_64") {
        "x86_64"
    } else if cfg!(target_arch = "aarch64") {
        "aarch64"
    } else {
        "any"
    }
}

#[cfg(test)]
mod tests {
    use super::KernelKind;

    #[test]
    fn kernel_kind_variants_are_named() {
        assert_eq!(format!("{:?}", KernelKind::X86_64), "X86_64");
        assert_eq!(format!("{:?}", KernelKind::Aarch64), "Aarch64");
        assert_eq!(KernelKind::X86_64, KernelKind::X86_64);
        let _ = crate::kernel_kind();
        assert_eq!(
            crate::dispatch_name(),
            if cfg!(any(target_arch = "x86_64", target_arch = "aarch64")) {
                if cfg!(target_arch = "x86_64") {
                    "x86_64"
                } else {
                    "aarch64"
                }
            } else {
                "any"
            }
        );
    }
}
