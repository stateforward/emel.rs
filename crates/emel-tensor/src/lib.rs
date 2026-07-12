//! Tensor contracts shared by model implementations and kernels.

#![forbid(unsafe_code)]
#![allow(missing_docs, dead_code)]

/// The element representation used by a tensor.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum ElementType {
    /// IEEE 754 binary32.
    F32,
    /// IEEE 754 binary16.
    F16,
    /// Brain floating point 16-bit representation.
    Bf16,
    /// A quantized representation to be defined by the model format.
    Quantized,
}

/// A tensor's immutable shape and element metadata.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TensorLayout {
    dimensions: Box<[usize]>,
    element_type: ElementType,
}

impl TensorLayout {
    /// Creates tensor metadata from its dimensions and element type.
    #[must_use]
    pub fn new(dimensions: impl Into<Box<[usize]>>, element_type: ElementType) -> Self {
        Self {
            dimensions: dimensions.into(),
            element_type,
        }
    }

    /// Returns the dimensions in row-major order.
    #[must_use]
    pub fn dimensions(&self) -> &[usize] {
        &self.dimensions
    }

    /// Returns the element representation.
    #[must_use]
    pub const fn element_type(&self) -> ElementType {
        self.element_type
    }
}
pub(crate) mod view;
