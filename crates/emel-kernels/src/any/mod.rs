//! Generic kernel actors and data-plane operations.

pub mod activation;
mod actor;
pub mod backward;
pub mod binary;
pub mod broadcast;
pub mod conv;
pub mod conv_transpose_1d;
pub mod count_equal;
pub mod custom;
pub mod diag;
pub mod elementwise;
pub mod event;
pub mod f16_matmul;
pub mod fill_arange;
pub mod flash_attn;
pub mod generic;
pub mod get_rows;
pub mod gla;
pub mod glu;
pub mod group_norm;
pub mod im2col;
pub mod indexed;
pub mod linalg;
pub mod matmul;
pub mod normalization;
pub mod pad;
pub mod pool;
pub mod positional;
pub mod power;
pub mod quant;
pub mod quant_more;
pub mod reductions;
pub mod rel;
pub mod reorder;
pub mod rope;
pub mod rwkv;
pub mod sequence;
pub mod shape;
pub mod sm;
pub mod sortformer;
pub mod ssm;
pub mod tensor_view;
pub mod train;
pub mod unary;
pub mod window;

pub use actor::{Error, Kernel};

/// Runtime-owned kernel facade selecting the host architecture without a
/// trait-object dispatch path.
///
/// The portable actor remains available for every generic event. Target
/// requests are accepted only by the architecture selected with [`set_kind`]
/// and are dispatched synchronously to that architecture's public actor.
pub struct Any {
    kind: crate::KernelKind,
    core: Core,
}

#[cfg(target_arch = "x86_64")]
enum Core {
    X86 {
        portable: Kernel,
        target: Option<crate::x86_64::X86Kernel>,
    },
}

#[cfg(target_arch = "aarch64")]
enum Core {
    Aarch64 {
        portable: Kernel,
        target: Option<crate::aarch64::Kernel>,
    },
}

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
enum Core {
    Portable { portable: Kernel },
}

impl core::fmt::Debug for Any {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter
            .debug_struct("Any")
            .field("kind", &self.kind)
            .finish_non_exhaustive()
    }
}

impl Default for Any {
    fn default() -> Self {
        Self::new()
    }
}

impl Any {
    /// Constructs a runtime facade selecting the host architecture.
    #[must_use]
    pub fn new() -> Self {
        Self::with_kind(crate::kernel_kind())
    }

    /// Constructs a runtime facade selecting `kind`.
    #[must_use]
    pub fn with_kind(kind: crate::KernelKind) -> Self {
        let core = Self::new_core();
        Self { kind, core }
    }

    #[cfg(target_arch = "x86_64")]
    fn new_core() -> Core {
        Core::X86 {
            portable: Kernel::new(),
            target: crate::x86_64::X86Kernel::try_new(),
        }
    }

    #[cfg(target_arch = "aarch64")]
    fn new_core() -> Core {
        Core::Aarch64 {
            portable: Kernel::new(),
            target: crate::aarch64::Kernel::try_new(),
        }
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    fn new_core() -> Core {
        Core::Portable {
            portable: Kernel::new(),
        }
    }

    /// Returns the currently selected architecture family.
    #[must_use]
    pub const fn kind(&self) -> crate::KernelKind {
        self.kind
    }

    /// Selects an architecture for subsequent target-event dispatch.
    ///
    /// Generic portable events continue to work regardless of the selected
    /// target. A target not compiled for this binary is rejected explicitly;
    /// a compiled target whose capability probe failed is reported as
    /// unavailable rather than silently falling back to another actor.
    ///
    /// # Errors
    ///
    /// Returns [`Error::KernelUnavailable`] when the requested compiled target
    /// failed its capability probe, or [`Error::UnsupportedKernelKind`] when
    /// the target is not compiled for this binary.
    pub fn set_kind(&mut self, kind: crate::KernelKind) -> Result<(), Error> {
        if kind == self.kind {
            return self.target_available(kind);
        }

        #[cfg(target_arch = "x86_64")]
        if kind == crate::KernelKind::X86_64 {
            let Core::X86 { target, .. } = &self.core;
            if target.is_none() {
                return Err(Error::KernelUnavailable(kind));
            }
            self.kind = kind;
            return Ok(());
        }

        #[cfg(target_arch = "aarch64")]
        if kind == crate::KernelKind::Aarch64 {
            let Core::Aarch64 { target, .. } = &self.core;
            if target.is_none() {
                return Err(Error::KernelUnavailable(kind));
            }
            self.kind = kind;
            return Ok(());
        }

        Err(Error::UnsupportedKernelKind(kind))
    }

    fn target_available(&self, kind: crate::KernelKind) -> Result<(), Error> {
        #[cfg(target_arch = "x86_64")]
        if kind == crate::KernelKind::X86_64 {
            let Core::X86 { target, .. } = &self.core;
            return target
                .as_ref()
                .map(|_| ())
                .ok_or(Error::KernelUnavailable(kind));
        }

        #[cfg(target_arch = "aarch64")]
        if kind == crate::KernelKind::Aarch64 {
            let Core::Aarch64 { target, .. } = &self.core;
            return target
                .as_ref()
                .map(|_| ())
                .ok_or(Error::KernelUnavailable(kind));
        }

        Err(Error::UnsupportedKernelKind(kind))
    }

    /// Dispatches one generic portable event synchronously.
    pub fn process_event<E: event::Event>(&mut self, event: E) -> E::Output {
        match &mut self.core {
            #[cfg(target_arch = "x86_64")]
            Core::X86 { portable, .. } => portable.process_event(event),
            #[cfg(target_arch = "aarch64")]
            Core::Aarch64 { portable, .. } => portable.process_event(event),
            #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
            Core::Portable { portable } => portable.process_event(event),
        }
    }

    /// Returns whether the portable actor is ready for another event.
    #[must_use]
    pub fn is_ready(&self) -> bool {
        match &self.core {
            #[cfg(target_arch = "x86_64")]
            Core::X86 { portable, .. } => portable.is_ready(),
            #[cfg(target_arch = "aarch64")]
            Core::Aarch64 { portable, .. } => portable.is_ready(),
            #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
            Core::Portable { portable } => portable.is_ready(),
        }
    }

    /// Dispatches a target-owned event through the selected architecture.
    ///
    /// The event's native result remains intact inside `Ok`; a selection or
    /// capability error is represented by the facade's typed [`Error`].
    ///
    /// # Errors
    ///
    /// Returns [`Error::KernelUnavailable`] when the selected target failed
    /// its capability probe, or [`Error::UnsupportedKernelKind`] when the
    /// selected target is not compiled for this binary.
    pub fn process_target_event<E: TargetEvent>(
        &mut self,
        event: E,
        output: &mut [f32],
    ) -> Result<E::Output, Error> {
        self.target_available(self.kind)?;
        match &mut self.core {
            #[cfg(target_arch = "x86_64")]
            Core::X86 { target, .. } => target
                .as_mut()
                .map(|target| event.dispatch(target, output))
                .ok_or(Error::KernelUnavailable(self.kind)),
            #[cfg(target_arch = "aarch64")]
            Core::Aarch64 { target, .. } => target
                .as_mut()
                .map(|target| event.dispatch(target, output))
                .ok_or(Error::KernelUnavailable(self.kind)),
            #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
            Core::Portable { .. } => Err(Error::UnsupportedKernelKind(self.kind)),
        }
    }
}

/// A target actor event accepted by [`Any::process_target_event`].
pub trait TargetEvent: Sized {
    /// Native target actor result.
    type Output;

    #[doc(hidden)]
    fn dispatch(self, actor: &mut TargetKernel, output: &mut [f32]) -> Self::Output;
}

#[cfg(target_arch = "x86_64")]
type TargetKernel = crate::x86_64::X86Kernel;

#[cfg(target_arch = "aarch64")]
type TargetKernel = crate::aarch64::Kernel;

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
type TargetKernel = Kernel;

#[cfg(target_arch = "x86_64")]
impl<E> TargetEvent for E
where
    E: crate::x86_64::X86KernelEvent,
{
    type Output = E::Output;

    fn dispatch(self, actor: &mut TargetKernel, output: &mut [f32]) -> Self::Output {
        actor.process_event(self, output)
    }
}

#[cfg(target_arch = "aarch64")]
impl<E> TargetEvent for E
where
    E: crate::aarch64::KernelEvent,
{
    type Output = E::Output;

    fn dispatch(self, actor: &mut TargetKernel, output: &mut [f32]) -> Self::Output {
        actor.process_event(self, output)
    }
}

/// Runtime facade alias retained for call sites that name the owner by role.
pub type RuntimeKernel = Any;
#[cfg(test)]
#[allow(clippy::float_cmp)]
mod tests {
    use super::Error;
    use crate::KernelKind;

    #[test]
    fn portable_event_routes_independently_of_selected_target() {
        let mut kernel = super::Any::with_kind(KernelKind::Aarch64);
        let input = [1.0_f32, -2.0, 3.5];
        let mut output = [0.0_f32; 3];

        let result = kernel.process_event(super::event::OpDup::new(&input, &mut output));

        assert_eq!(result, Ok(()));
        assert_eq!(output, input);
        assert!(kernel.is_ready());
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn target_event_preserves_native_output_and_errors() {
        let mut kernel = super::Any::new();
        let input = [2.0_f32, 4.0, 8.0];
        let mut output = [0.0_f32; 3];

        let result = kernel.process_target_event(crate::x86_64::X86Dup::new(&input), &mut output);

        assert!(matches!(result, Ok(Ok(()))));
        assert_eq!(output, input);

        let mut invalid_output = [9.0_f32; 2];
        let invalid =
            kernel.process_target_event(crate::x86_64::X86Dup::new(&input), &mut invalid_output);

        assert!(matches!(
            invalid,
            Ok(Err(crate::x86_64::X86DupF32Error::InvalidShape))
        ));
        assert_eq!(invalid_output, [9.0_f32; 2]);
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn target_event_preserves_native_output_and_errors() {
        let mut kernel = super::Any::new();
        let input = [2.0_f32, 4.0, 8.0];
        let mut output = [0.0_f32; 3];

        let result = kernel.process_target_event(crate::aarch64::Dup::new(&input), &mut output);

        assert!(matches!(result, Ok(Ok(()))));
        assert_eq!(output, input);

        let mut invalid_output = [9.0_f32; 2];
        let invalid =
            kernel.process_target_event(crate::aarch64::Dup::new(&input), &mut invalid_output);

        assert!(matches!(
            invalid,
            Ok(Err(crate::aarch64::DupF32Error::InvalidShape))
        ));
        assert_eq!(invalid_output, [9.0_f32; 2]);
    }

    #[test]
    fn unsupported_target_is_reported_without_fallback() {
        let mut kernel = super::Any::new();

        assert_eq!(
            kernel.set_kind(KernelKind::Aarch64),
            if cfg!(target_arch = "aarch64") {
                Ok(())
            } else {
                Err(Error::UnsupportedKernelKind(KernelKind::Aarch64))
            }
        );

        #[cfg(not(target_arch = "aarch64"))]
        assert_eq!(kernel.kind(), KernelKind::X86_64);
    }
}
