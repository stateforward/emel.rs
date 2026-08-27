//! Shared target-router events for the scalar unary operation slice.
//!
//! The pinned x86-64 and `AArch64` machines both expose scalar action aliases
//! for `exp`, `tanh`, `elu`, `gelu`, and `silu`.  The request types here keep
//! those operations distinct at the target-router boundary.  The routers
//! delegate the selected operation to the maintained [`crate::any::unary::UnaryKernel`]
//! actor, so formulas remain owned by one SML machine.

#![allow(clippy::derive_partial_eq_without_eq)]

use crate::any::unary::UnaryResult;

/// Result returned by a scalar target-unary request.
pub type ScalarUnaryResult = UnaryResult;

/// Pinned reference revision for both target scalar-unary lanes.
pub const PINNED_EMEL_CPP_COMMIT: &str = "843a117386ef17dc5a50549bbfc821074c2141d6";

/// Source identity for the generic scalar formula and dense loop.
pub const PINNED_DETAIL_SPAN: &str = "src/emel/kernel/detail.hpp:2392-2411,3271-3325";

/// Source identity for the x86 scalar-unary guard aliases.
pub const PINNED_X86_GUARD_SPAN: &str = "src/emel/kernel/x86_64/guards.hpp:237-241,255-264";
/// Pinned blob identity for the x86 guard source.
pub const PINNED_X86_GUARD_BLOB: &str = "cb3dac8253f8417c9b44acff1de414f6d0a3a3cf";
/// Source identity for the x86 scalar-unary action aliases.
pub const PINNED_X86_ACTION_SPAN: &str = "src/emel/kernel/x86_64/actions.hpp:2583-2600,2716-2720";
/// Pinned blob identity for the x86 action source.
pub const PINNED_X86_ACTION_BLOB: &str = "d45558f5eb96950f43c16a09d768cb4f382d6d61";
/// Source identity for the x86 scalar-unary transitions.
pub const PINNED_X86_TRANSITION_SPAN: &str = "src/emel/kernel/x86_64/sm.hpp:1065-1093";
/// Pinned blob identity for the x86 transition source.
pub const PINNED_X86_TRANSITION_BLOB: &str = "0b4d635ebbd0fbd52dbca8a2345547fb571205c8";

/// Source identity for the `AArch64` scalar-unary guard aliases and the
/// source scalar-vs-SIMD predicate used by the `silu` alias.
pub const PINNED_AARCH64_GUARD_SPAN: &str =
    "src/emel/kernel/aarch64/guards.hpp:826-834,848-860,877-886";
/// Pinned blob identity for the `AArch64` guard source.
pub const PINNED_AARCH64_GUARD_BLOB: &str = "c25714566ec9a02679daef85089544575123408e";
/// Source identity for the `AArch64` scalar-unary action aliases.
pub const PINNED_AARCH64_ACTION_SPAN: &str =
    "src/emel/kernel/aarch64/actions.hpp:9194-9211,9376-9380";
/// Pinned blob identity for the `AArch64` action source.
pub const PINNED_AARCH64_ACTION_BLOB: &str = "267d4f74e6e7498155c8535920322ffef2c02fb6";
/// Source identity for the `AArch64` scalar-unary transitions.
pub const PINNED_AARCH64_TRANSITION_SPAN: &str = "src/emel/kernel/aarch64/sm.hpp:1195-1223";
/// Pinned blob identity for the `AArch64` transition source.
pub const PINNED_AARCH64_TRANSITION_BLOB: &str = "865a9cc6ba6115382ed043c464f3d62bcd851357";

macro_rules! define_scalar_unary_event {
    ($name:ident, $operation:literal) => {
        #[doc = concat!("A scalar target request for `", $operation, "` over a dense F32 slice.")]
        #[derive(Debug)]
        pub struct $name<'a> {
            input: &'a [f32],
        }

        impl<'a> $name<'a> {
            /// Creates a request; dense shape validation remains in the
            /// target router's explicit guard rows.
            #[must_use]
            pub const fn new(input: &'a [f32]) -> Self {
                Self { input }
            }

            /// Returns the borrowed input for same-RTC router handoff.
            #[must_use]
            pub(crate) const fn input(&self) -> &'a [f32] {
                self.input
            }
        }
    };
}

define_scalar_unary_event!(OpScalarUnaryExp, "exp");
define_scalar_unary_event!(OpScalarUnaryTanh, "tanh");
define_scalar_unary_event!(OpScalarUnaryElu, "elu");
define_scalar_unary_event!(OpScalarUnaryGelu, "gelu");
define_scalar_unary_event!(OpScalarUnarySilu, "silu");
