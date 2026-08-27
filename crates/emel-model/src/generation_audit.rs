//! Source-exact label values for generation quantization audit output.

/// A generation-audit tensor-type label chosen by an owning state machine.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TensorTypeLabel(&'static str);

impl TensorTypeLabel {
    /// Dense 32-bit floating point.
    pub const F32: Self = Self("f32");
    /// Two-bit K-block quantization.
    pub const Q2_K: Self = Self("q2_k");
    /// Three-bit K-block quantization.
    pub const Q3_K: Self = Self("q3_k");
    /// Four-bit K-block quantization.
    pub const Q4_K: Self = Self("q4_k");
    /// Six-bit K-block quantization.
    pub const Q6_K: Self = Self("q6_k");
    /// Four-bit legacy block quantization.
    pub const Q4_0: Self = Self("q4_0");
    /// The pinned source label for every unclassified tensor type.
    pub const UNKNOWN: Self = Self("unknown");

    /// Returns the immutable source-exact label text.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        self.0
    }
}

#[cfg(test)]
mod tests {
    use super::TensorTypeLabel;

    #[test]
    fn labels_are_source_exact_data_without_runtime_dtype_classification() {
        assert_eq!(TensorTypeLabel::F32.as_str(), "f32");
        assert_eq!(TensorTypeLabel::Q2_K.as_str(), "q2_k");
        assert_eq!(TensorTypeLabel::Q3_K.as_str(), "q3_k");
        assert_eq!(TensorTypeLabel::Q4_K.as_str(), "q4_k");
        assert_eq!(TensorTypeLabel::Q6_K.as_str(), "q6_k");
        assert_eq!(TensorTypeLabel::Q4_0.as_str(), "q4_0");
        assert_eq!(TensorTypeLabel::UNKNOWN.as_str(), "unknown");
    }
}
