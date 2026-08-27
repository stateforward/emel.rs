//! Safe F32 row-gather kernel.
//!
//! The pinned reference has separate transition rows for F32, F16, BF16 and
//! quantized source tensors. This module owns the F32, F16, BF16, `Q4_0`,
//! `Q8_0`, and `Q4_K` row-gather rows.
//! Index tensors are represented by immutable, caller-owned I32 views with
//! checked contiguous or byte-strided layouts. Reverse (`get_rows_back`) rows
//! remain explicit parity residuals rather than implicit fallbacks.
//!
//! The pinned validity guard permits an empty index tensor to take the no-op
//! route even when an outer source extent is zero. The empty-index guard below
//! therefore validates only metadata needed by the no-op: F32 source dtype,
//! an implicit `nb[0] == 0` or explicit `nb[0] == 4` representation, positive
//! row/column extents, and a dense destination. It does not call the non-empty
//! source span validator because the action performs no source or destination
//! access on that route.

#![allow(clippy::derive_partial_eq_without_eq)]
#![allow(private_interfaces)]

use core::cell::Cell;
use core::fmt;

use sml::sml;

use super::f16_matmul::F16View;
use super::matmul::QuantizedView;
use super::quant::{BLOCK_VALUES, Q4_0_BLOCK_BYTES, Q4_0Row, Q8_0_BLOCK_BYTES, Q8_0Row};
use super::quant_more::{Q4_K_BLOCK_BYTES, Q4KRow, QK_K_VALUES};
use super::tensor_view::{ByteTensorView, DType, Layout, TensorView, TensorViewMut};

const Q4_0_CODE: u8 = 2;
const Q8_0_CODE: u8 = 8;
const Q4_K_CODE: u8 = 12;

/// An immutable I32 index tensor for [`OpGetRows`].
///
/// The three extents correspond to the reference index tensor's `ne[0..3]`;
/// `ne[3]` is fixed to one for this operation.  The actor validates the
/// complete logical span before any index is read.  [`Self::new`] retains the
/// row-major contiguous form; [`Self::with_strides`] represents a caller-owned
/// non-contiguous view using checked byte strides.
#[derive(Clone, Copy, Debug)]
pub struct IndexView<'a> {
    data: &'a [i32],
    ne: [u64; 3],
    strides: [usize; 3],
    count: usize,
    shape_valid: bool,
    strides_valid: bool,
    span_valid: bool,
}

impl<'a> IndexView<'a> {
    /// Creates an immutable row-major index view.
    #[must_use]
    pub fn new(data: &'a [i32], ne: [u64; 3]) -> Self {
        let (byte_strides, inferred_valid) = contiguous_byte_strides(ne);
        Self::from_byte_strides(data, ne, byte_strides, inferred_valid, false)
    }

    /// Creates an immutable I32 index view with explicit byte strides.
    ///
    /// Each stride is measured from the first element of one logical
    /// dimension, matching the pinned tensor `nb[0..3]` contract.  As in the
    /// pinned layout, `byte_strides[0] == 0` selects implicit contiguous
    /// strides and the other supplied strides are ignored.  Explicit strides
    /// must be representable as whole I32 elements.  A zero stride is valid
    /// for a singleton (or empty) dimension and rejected for a dimension with
    /// more than one element.  The complete reachable byte span must fit
    /// `data`; invalid metadata and short storage are reported by the owning
    /// actor's guard as [`GetRowsError::InvalidView`].
    #[must_use]
    pub fn with_strides(data: &'a [i32], ne: [u64; 3], byte_strides: [u64; 3]) -> Self {
        if byte_strides[0] == 0 {
            let (contiguous, inferred_valid) = contiguous_byte_strides(ne);
            Self::from_byte_strides(data, ne, contiguous, inferred_valid, true)
        } else {
            Self::from_byte_strides(data, ne, byte_strides, true, false)
        }
    }

    fn from_byte_strides(
        data: &'a [i32],
        ne: [u64; 3],
        byte_strides: [u64; 3],
        inferred_valid: bool,
        allow_implicit_zero_strides: bool,
    ) -> Self {
        let mut count = 1usize;
        let mut zero_extent = false;
        let mut shape_valid = true;
        let mut strides_valid = inferred_valid;
        let mut element_strides = [0usize; 3];
        let mut dimension = 0;
        while dimension < 3 {
            let extent = ne[dimension];
            let stride = byte_strides[dimension];
            if extent > 1 && stride == 0 && !allow_implicit_zero_strides {
                strides_valid = false;
            }
            if stride != 0 && (stride < 4 || !stride.is_multiple_of(4)) {
                strides_valid = false;
            }
            if let Ok(element_stride) = usize::try_from(stride / 4) {
                element_strides[dimension] = element_stride;
            } else {
                strides_valid = false;
            }
            if extent == 0 {
                zero_extent = true;
            } else if !zero_extent {
                if let Some(value) = usize::try_from(extent)
                    .ok()
                    .and_then(|value| count.checked_mul(value))
                {
                    count = value;
                } else {
                    shape_valid = false;
                    count = 0;
                }
            }
            dimension += 1;
        }
        if zero_extent {
            count = 0;
        }
        let span_valid = strides_valid
            && shape_valid
            && (zero_extent || span_fits(data.len(), ne, byte_strides));
        Self {
            data,
            ne,
            strides: element_strides,
            count,
            shape_valid,
            strides_valid,
            span_valid,
        }
    }

    /// Returns the three logical extents.
    #[must_use]
    pub const fn ne(&self) -> [u64; 3] {
        self.ne
    }

    /// Returns whether the I32 shape and byte strides are representable. This
    /// does not inspect backing-storage length.
    #[must_use]
    pub const fn metadata_valid(&self) -> bool {
        self.shape_valid && self.strides_valid
    }

    /// Returns whether the index view has a complete span. A zero extent is a
    /// valid no-op index tensor; positive-shape overflow remains invalid.
    const fn validate(&self) -> bool {
        self.metadata_valid() && self.span_valid
    }

    /// Returns the number of logical indices after the owning guard proves
    /// the view valid.
    const fn count(&self) -> usize {
        self.count
    }

    /// Reads one logical index after the owning guard proves the view valid.
    #[inline]
    #[allow(clippy::cast_possible_truncation)]
    const fn read(&self, ordinal: usize) -> i32 {
        let extent_0 = self.ne[0] as usize;
        let i0 = ordinal % extent_0;
        let remaining = ordinal / extent_0;
        let extent_1 = self.ne[1] as usize;
        let i1 = remaining % extent_1;
        let i2 = remaining / extent_1;
        let element = i0 * self.strides[0] + i1 * self.strides[1] + i2 * self.strides[2];
        self.data[element]
    }
}

fn contiguous_byte_strides(ne: [u64; 3]) -> ([u64; 3], bool) {
    let mut byte_strides = [0_u64; 3];
    byte_strides[0] = 4;
    let mut inferred_valid = true;
    let mut dimension = 1;
    while dimension < 3 {
        let checked_stride = byte_strides[dimension - 1].checked_mul(ne[dimension - 1]);
        if checked_stride.is_none() {
            inferred_valid = false;
        }
        byte_strides[dimension] = checked_stride.unwrap_or(0);
        dimension += 1;
    }
    (byte_strides, inferred_valid)
}

const fn span_fits(data_len: usize, ne: [u64; 3], byte_strides: [u64; 3]) -> bool {
    let mut max_offset = 0_u128;
    let mut dimension = 0;
    while dimension < 3 {
        let Some(contribution) =
            ((ne[dimension] as u128) - 1).checked_mul(byte_strides[dimension] as u128)
        else {
            return false;
        };
        let Some(next_offset) = max_offset.checked_add(contribution) else {
            return false;
        };
        max_offset = next_offset;
        dimension += 1;
    }
    let Some(end) = max_offset.checked_add(4) else {
        return false;
    };
    end <= (data_len as u128).saturating_mul(4)
}

/// F32 row-gather request matching the reference `op_get_rows` operation.
#[derive(Debug)]
pub struct OpGetRows<'a> {
    src: TensorView<'a>,
    indices: IndexView<'a>,
    dst: TensorViewMut<'a>,
}

/// F32 row-gather request backed by bytes.
///
/// This event preserves the pinned `memcpy(cols * sizeof(float))` behavior
/// when an explicit outer source stride is not four-byte aligned.  The
/// byte-backed source is decoded safely and the destination remains a dense
/// mutable F32 view.
#[derive(Debug)]
pub struct OpGetRowsF32Bytes<'a> {
    source: ByteTensorView<'a>,
    indices: IndexView<'a>,
    destination: TensorViewMut<'a>,
}

impl<'a> OpGetRowsF32Bytes<'a> {
    /// Creates a byte-backed F32 row-gather event. Validation occurs in actor
    /// guards before any bytes are read or written.
    #[must_use]
    pub const fn new(
        source: ByteTensorView<'a>,
        indices: IndexView<'a>,
        destination: TensorViewMut<'a>,
    ) -> Self {
        Self {
            source,
            indices,
            destination,
        }
    }
}

/// F16 row-gather request matching the pinned `get_rows_as<f16>` operation.
///
/// The destination remains dense F32, while the source is decoded through the
/// maintained safe [`F16View`] contract. F16, BF16, and quantized source rows
/// are separate reference transition variants.
#[derive(Debug)]
pub struct OpGetRowsF16<'a> {
    source: F16View<'a>,
    indices: IndexView<'a>,
    destination: TensorViewMut<'a>,
}

/// An immutable BF16 tensor view for the pinned `get_rows_as<bf16>` source
/// variant.
///
/// Values are represented by their IEEE BF16 bit patterns in a `u16` slice and
/// converted directly to F32 during the selected action.
#[derive(Debug)]
pub struct Bf16View<'a> {
    data: &'a [u16],
    layout: super::tensor_view::Layout,
    strides: [u64; 4],
}

impl<'a> Bf16View<'a> {
    /// Creates a BF16 view. Validation is classified by the row-gather guards.
    #[must_use]
    pub const fn new(data: &'a [u16], layout: super::tensor_view::Layout) -> Self {
        let strides = match effective_bf16_strides(layout) {
            Some(strides) => strides,
            None => layout.nb(),
        };
        Self {
            data,
            layout,
            strides,
        }
    }

    /// Returns the BF16 view layout.
    #[must_use]
    pub const fn layout(&self) -> super::tensor_view::Layout {
        self.layout
    }

    fn validate(&self) -> bool {
        if self.layout.dtype() != super::tensor_view::DType::Bf16
            || self.layout.element_count().is_none()
        {
            return false;
        }
        let ne = self.layout.ne();
        let nb = self.layout.nb();
        let Some(strides) = effective_bf16_strides(self.layout) else {
            return false;
        };
        if nb[0] != 0 {
            let mut dimension = 0;
            while dimension < 4 {
                if (ne[dimension] > 1 && nb[dimension] == 0)
                    || (nb[dimension] != 0
                        && (nb[dimension] < 2 || !nb[dimension].is_multiple_of(2)))
                {
                    return false;
                }
                dimension += 1;
            }
        }
        let mut max_offset = 0u128;
        let mut dimension = 0;
        while dimension < 4 {
            let extent = u128::from(ne[dimension]);
            let stride = u128::from(strides[dimension]);
            let Some(contribution) = (extent - 1).checked_mul(stride) else {
                return false;
            };
            max_offset = match max_offset.checked_add(contribution) {
                Some(value) => value,
                None => return false,
            };
            dimension += 1;
        }
        let Some(end) = max_offset.checked_add(2) else {
            return false;
        };
        end <= (self.data.len() as u128).saturating_mul(2)
    }

    #[inline]
    fn read(&self, ordinal: usize) -> f32 {
        bf16_to_f32(self.data[self.element_index(ordinal)])
    }

    #[inline]
    #[allow(clippy::cast_possible_truncation)]
    const fn element_index(&self, ordinal: usize) -> usize {
        let mut remaining = ordinal as u64;
        let mut byte_offset = 0u64;
        let ne = self.layout.ne();
        let mut dimension = 0;
        while dimension < 4 {
            byte_offset += (remaining % ne[dimension]) * self.strides[dimension];
            remaining /= ne[dimension];
            dimension += 1;
        }
        (byte_offset / 2) as usize
    }
}

const fn effective_bf16_strides(layout: super::tensor_view::Layout) -> Option<[u64; 4]> {
    let nb = layout.nb();
    if nb[0] != 0 {
        return Some(nb);
    }
    let ne = layout.ne();
    let mut strides = [0_u64; 4];
    strides[0] = 2;
    let mut dimension = 1;
    while dimension < 4 {
        let Some(stride) = strides[dimension - 1].checked_mul(ne[dimension - 1]) else {
            return None;
        };
        strides[dimension] = stride;
        dimension += 1;
    }
    Some(strides)
}

/// BF16-source row-gather request matching the pinned
/// `run_get_rows_as<dtype_bf16>` operation. The destination is dense F32.
#[derive(Debug)]
pub struct OpGetRowsBf16<'a> {
    source: Bf16View<'a>,
    indices: IndexView<'a>,
    destination: TensorViewMut<'a>,
}

impl<'a> OpGetRowsBf16<'a> {
    /// Creates a BF16 row-gather event. Validation occurs in actor guards.
    #[must_use]
    pub const fn new(
        source: Bf16View<'a>,
        indices: IndexView<'a>,
        destination: TensorViewMut<'a>,
    ) -> Self {
        Self {
            source,
            indices,
            destination,
        }
    }
}

impl<'a> OpGetRowsF16<'a> {
    /// Creates an F16 row-gather event. Validation occurs in actor guards.
    #[must_use]
    pub const fn new(
        source: F16View<'a>,
        indices: IndexView<'a>,
        destination: TensorViewMut<'a>,
    ) -> Self {
        Self {
            source,
            indices,
            destination,
        }
    }
}

/// Q4_0-source row-gather request matching the pinned quantized
/// `run_get_rows_as` route. The destination is dense F32.
#[derive(Debug)]
pub struct OpGetRowsQ4_0<'a> {
    source: QuantizedView<'a>,
    indices: IndexView<'a>,
    destination: TensorViewMut<'a>,
}

impl<'a> OpGetRowsQ4_0<'a> {
    /// Creates a `Q4_0` row-gather event. Validation occurs in actor guards.
    #[must_use]
    pub const fn new(
        source: QuantizedView<'a>,
        indices: IndexView<'a>,
        destination: TensorViewMut<'a>,
    ) -> Self {
        Self {
            source,
            indices,
            destination,
        }
    }
}

/// Q8_0-source row-gather request matching the pinned quantized
/// `run_get_rows_as` route. The destination is dense F32.
#[derive(Debug)]
pub struct OpGetRowsQ8_0<'a> {
    source: QuantizedView<'a>,
    indices: IndexView<'a>,
    destination: TensorViewMut<'a>,
}

impl<'a> OpGetRowsQ8_0<'a> {
    /// Creates a `Q8_0` row-gather event. Validation occurs in actor guards.
    #[must_use]
    pub const fn new(
        source: QuantizedView<'a>,
        indices: IndexView<'a>,
        destination: TensorViewMut<'a>,
    ) -> Self {
        Self {
            source,
            indices,
            destination,
        }
    }
}

/// Q4_K-source row-gather request matching the pinned quantized
/// `run_get_rows_as` route. The destination is dense F32.
#[derive(Debug)]
pub struct OpGetRowsQ4K<'a> {
    source: QuantizedView<'a>,
    indices: IndexView<'a>,
    destination: TensorViewMut<'a>,
}

impl<'a> OpGetRowsQ4K<'a> {
    /// Creates a `Q4_K` row-gather event. Validation occurs in actor guards.
    #[must_use]
    pub const fn new(
        source: QuantizedView<'a>,
        indices: IndexView<'a>,
        destination: TensorViewMut<'a>,
    ) -> Self {
        Self {
            source,
            indices,
            destination,
        }
    }
}

impl<'a> OpGetRows<'a> {
    /// Creates a row-gather event. Validation occurs in an actor guard.
    #[must_use]
    pub const fn new(src: TensorView<'a>, indices: IndexView<'a>, dst: TensorViewMut<'a>) -> Self {
        Self { src, indices, dst }
    }
}

/// Explicitly reports an event outside the maintained row-gather API.
///
/// The generated machine also retains its generic unexpected-event row for
/// internal SML completeness; public callers receive this typed error through
/// this event instead of observing a silent discard.
#[derive(Clone, Copy, Debug, Default)]
pub struct UnexpectedGetRows;

/// Result returned by the public row-gather events after RTC dispatch.
pub type GetRowsOutcome = Result<(), GetRowsError>;

/// Errors returned by the maintained F32 row-gather actor.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GetRowsError {
    /// One or more views failed dtype, shape, stride, or bounds validation.
    InvalidView,
    /// Valid views do not have the reference row-gather extents.
    ShapeMismatch,
    /// An index is negative or outside the source row extent.
    IndexOutOfBounds,
    /// The F32 source uses the implicit stride representation, which is not
    /// included in this bounded source-backed slice.
    UnsupportedSourceStride,
    /// The generated machine rejected an unexpected event.
    UnexpectedEvent,
    /// The generated machine failed to dispatch an event.
    Internal,
}

impl fmt::Display for GetRowsError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidView => formatter.write_str("invalid get_rows tensor view"),
            Self::ShapeMismatch => formatter.write_str("get_rows tensor shapes differ"),
            Self::IndexOutOfBounds => formatter.write_str("get_rows index is out of bounds"),
            Self::UnsupportedSourceStride => {
                formatter.write_str("get_rows source element stride is unsupported")
            }
            Self::UnexpectedEvent => formatter.write_str("unexpected get_rows event"),
            Self::Internal => formatter.write_str("internal get_rows dispatch error"),
        }
    }
}

impl std::error::Error for GetRowsError {}

struct GetRowsRuntime<'a> {
    event: OpGetRows<'a>,
    result: &'a Cell<GetRowsOutcome>,
}

struct GetRowsF32BytesRuntime<'a> {
    event: OpGetRowsF32Bytes<'a>,
    result: &'a Cell<GetRowsOutcome>,
}

struct GetRowsF16Runtime<'a> {
    event: OpGetRowsF16<'a>,
    result: &'a Cell<GetRowsOutcome>,
}

struct GetRowsBf16Runtime<'a> {
    event: OpGetRowsBf16<'a>,
    result: &'a Cell<GetRowsOutcome>,
}

struct QuantizedOutputSlot<'data> {
    data: Cell<Option<&'data mut [f32]>>,
    layout: Layout,
    data_len: usize,
}

#[derive(Clone, Copy)]
struct QuantizedOutput<'slot, 'data> {
    data: &'slot Cell<Option<&'data mut [f32]>>,
    layout: Layout,
    data_len: usize,
}

impl<'data> QuantizedOutputSlot<'data> {
    const fn new(view: TensorViewMut<'data>) -> Self {
        let (data, layout) = view.into_parts();
        let data_len = data.len();
        Self {
            data: Cell::new(Some(data)),
            layout,
            data_len,
        }
    }

    const fn view(&self) -> QuantizedOutput<'_, 'data> {
        QuantizedOutput {
            data: &self.data,
            layout: self.layout,
            data_len: self.data_len,
        }
    }
}

impl QuantizedOutput<'_, '_> {
    const fn layout(self) -> Layout {
        self.layout
    }

    fn validate(self) -> bool {
        self.layout.validate(self.data_len).is_ok()
    }

    fn write(self, ordinal: usize, value: f32) {
        let data = self.data.take().expect("output slot is available");
        data[ordinal] = value;
        self.data.set(Some(data));
    }
}

#[derive(Clone, Copy)]
struct GetRowsQ4_0Runtime<'dispatch, 'data, 'slot> {
    source: QuantizedView<'data>,
    indices: IndexView<'data>,
    destination: QuantizedOutput<'slot, 'data>,
    result: &'dispatch Cell<GetRowsOutcome>,
}

#[derive(Clone, Copy)]
struct GetRowsQ8_0Runtime<'dispatch, 'data, 'slot> {
    source: QuantizedView<'data>,
    indices: IndexView<'data>,
    destination: QuantizedOutput<'slot, 'data>,
    result: &'dispatch Cell<GetRowsOutcome>,
}

#[derive(Clone, Copy)]
struct GetRowsQ4KRuntime<'dispatch, 'data, 'slot> {
    source: QuantizedView<'data>,
    indices: IndexView<'data>,
    destination: QuantizedOutput<'slot, 'data>,
    result: &'dispatch Cell<GetRowsOutcome>,
}

struct UnexpectedRuntime<'a> {
    result: &'a Cell<GetRowsOutcome>,
}

#[derive(Default)]
struct Context;

#[derive(Default)]
struct BytesContext;

sml! {
    GetRowsMachine<'dispatch, 'data, 'slot>
    where
        'data: 'slot,
    {
        // F32 source rows.
        "ready"_s <= *"ready"_s + GetRows(GetRowsRuntime<'dispatch>) [guard_get_rows_valid] / effect_get_rows_execute,
        "ready"_s <= "ready"_s + GetRows(GetRowsRuntime<'dispatch>) [guard_get_rows_shape_invalid] / effect_get_rows_shape_reject,
        "ready"_s <= "ready"_s + GetRows(GetRowsRuntime<'dispatch>) [guard_get_rows_stride_invalid] / effect_get_rows_stride_reject,
        "ready"_s <= "ready"_s + GetRows(GetRowsRuntime<'dispatch>) [guard_get_rows_index_invalid] / effect_get_rows_index_reject,
        "ready"_s <= "ready"_s + GetRows(GetRowsRuntime<'dispatch>) [guard_get_rows_invalid_view] / effect_get_rows_view_reject,

        // F16 source rows.
        "ready"_s <= "ready"_s + GetRowsF16(GetRowsF16Runtime<'dispatch>) [guard_get_rows_f16_valid] / effect_get_rows_f16_execute,
        "ready"_s <= "ready"_s + GetRowsF16(GetRowsF16Runtime<'dispatch>) [guard_get_rows_f16_shape_invalid] / effect_get_rows_shape_reject_f16,
        "ready"_s <= "ready"_s + GetRowsF16(GetRowsF16Runtime<'dispatch>) [guard_get_rows_f16_index_invalid] / effect_get_rows_index_reject_f16,
        "ready"_s <= "ready"_s + GetRowsF16(GetRowsF16Runtime<'dispatch>) [guard_get_rows_f16_invalid_view] / effect_get_rows_view_reject_f16,

        // BF16 source rows.
        "ready"_s <= "ready"_s + GetRowsBf16(GetRowsBf16Runtime<'dispatch>) [guard_get_rows_bf16_valid] / effect_get_rows_bf16_execute,
        "ready"_s <= "ready"_s + GetRowsBf16(GetRowsBf16Runtime<'dispatch>) [guard_get_rows_bf16_shape_invalid] / effect_get_rows_shape_reject_bf16,
        "ready"_s <= "ready"_s + GetRowsBf16(GetRowsBf16Runtime<'dispatch>) [guard_get_rows_bf16_index_invalid] / effect_get_rows_index_reject_bf16,
        "ready"_s <= "ready"_s + GetRowsBf16(GetRowsBf16Runtime<'dispatch>) [guard_get_rows_bf16_invalid_view] / effect_get_rows_view_reject_bf16,

        // Q4_0 source rows.
        "q4_0_decision"_s <= "ready"_s + GetRowsQ4_0(GetRowsQ4_0Runtime<'dispatch, 'data, 'slot>),
        "ready"_s <= "q4_0_decision"_s + completion<GetRowsQ4_0>(GetRowsQ4_0Runtime<'dispatch, 'data, 'slot>) [guard_get_rows_q4_0_valid] / effect_get_rows_q4_0_execute,
        "ready"_s <= "q4_0_decision"_s + completion<GetRowsQ4_0>(GetRowsQ4_0Runtime<'dispatch, 'data, 'slot>) [guard_get_rows_q4_0_shape_invalid] / effect_get_rows_shape_reject_q4_0,
        "ready"_s <= "q4_0_decision"_s + completion<GetRowsQ4_0>(GetRowsQ4_0Runtime<'dispatch, 'data, 'slot>) [guard_get_rows_q4_0_index_invalid] / effect_get_rows_index_reject_q4_0,
        "ready"_s <= "q4_0_decision"_s + completion<GetRowsQ4_0>(GetRowsQ4_0Runtime<'dispatch, 'data, 'slot>) [guard_get_rows_q4_0_invalid_view] / effect_get_rows_view_reject_q4_0,

        // Q8_0 source rows.
        "q8_0_decision"_s <= "ready"_s + GetRowsQ8_0(GetRowsQ8_0Runtime<'dispatch, 'data, 'slot>),
        "ready"_s <= "q8_0_decision"_s + completion<GetRowsQ8_0>(GetRowsQ8_0Runtime<'dispatch, 'data, 'slot>) [guard_get_rows_q8_0_valid] / effect_get_rows_q8_0_execute,
        "ready"_s <= "q8_0_decision"_s + completion<GetRowsQ8_0>(GetRowsQ8_0Runtime<'dispatch, 'data, 'slot>) [guard_get_rows_q8_0_shape_invalid] / effect_get_rows_shape_reject_q8_0,
        "ready"_s <= "q8_0_decision"_s + completion<GetRowsQ8_0>(GetRowsQ8_0Runtime<'dispatch, 'data, 'slot>) [guard_get_rows_q8_0_index_invalid] / effect_get_rows_index_reject_q8_0,
        "ready"_s <= "q8_0_decision"_s + completion<GetRowsQ8_0>(GetRowsQ8_0Runtime<'dispatch, 'data, 'slot>) [guard_get_rows_q8_0_invalid_view] / effect_get_rows_view_reject_q8_0,

        // Q4_K source rows.
        "q4_k_decision"_s <= "ready"_s + GetRowsQ4K(GetRowsQ4KRuntime<'dispatch, 'data, 'slot>),
        "ready"_s <= "q4_k_decision"_s + completion<GetRowsQ4K>(GetRowsQ4KRuntime<'dispatch, 'data, 'slot>) [guard_get_rows_q4_k_valid] / effect_get_rows_q4_k_execute,
        "ready"_s <= "q4_k_decision"_s + completion<GetRowsQ4K>(GetRowsQ4KRuntime<'dispatch, 'data, 'slot>) [guard_get_rows_q4_k_shape_invalid] / effect_get_rows_shape_reject_q4_k,
        "ready"_s <= "q4_k_decision"_s + completion<GetRowsQ4K>(GetRowsQ4KRuntime<'dispatch, 'data, 'slot>) [guard_get_rows_q4_k_index_invalid] / effect_get_rows_index_reject_q4_k,
        "ready"_s <= "q4_k_decision"_s + completion<GetRowsQ4K>(GetRowsQ4KRuntime<'dispatch, 'data, 'slot>) [guard_get_rows_q4_k_invalid_view] / effect_get_rows_view_reject_q4_k,

        "ready"_s <= "q4_0_decision"_s + unexpected_event<_> / effect_unexpected,
        "ready"_s <= "q8_0_decision"_s + unexpected_event<_> / effect_unexpected,
        "ready"_s <= "q4_k_decision"_s + unexpected_event<_> / effect_unexpected,

        // Explicit unexpected-event recovery.
        "ready"_s <= "ready"_s + Unexpected(UnexpectedRuntime<'dispatch>) / effect_get_rows_unexpected,
        "ready"_s <= "ready"_s + unexpected_event<_> / effect_unexpected,
    }
}

sml! {
    GetRowsBytesMachine<'dispatch> {
        "ready"_s <= *"ready"_s + GetRowsF32Bytes(GetRowsF32BytesRuntime<'dispatch>) [guard_get_rows_f32_bytes_valid] / effect_get_rows_f32_bytes_execute,
        "ready"_s <= "ready"_s + GetRowsF32Bytes(GetRowsF32BytesRuntime<'dispatch>) [guard_get_rows_f32_bytes_shape_invalid] / effect_get_rows_f32_bytes_shape_reject,
        "ready"_s <= "ready"_s + GetRowsF32Bytes(GetRowsF32BytesRuntime<'dispatch>) [guard_get_rows_f32_bytes_index_invalid] / effect_get_rows_f32_bytes_index_reject,
        "ready"_s <= "ready"_s + GetRowsF32Bytes(GetRowsF32BytesRuntime<'dispatch>) [guard_get_rows_f32_bytes_invalid_view] / effect_get_rows_f32_bytes_view_reject,
        "ready"_s <= "ready"_s + unexpected_event<_> / effect_unexpected,
    }
}

/// Single-writer, run-to-completion actor for the maintained F32 row gather.
pub struct GetRowsKernel {
    machine: GetRowsMachineStateMachine<Context>,
    bytes_machine: GetRowsBytesMachineStateMachine<BytesContext>,
}

impl fmt::Debug for GetRowsKernel {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("GetRowsKernel")
            .finish_non_exhaustive()
    }
}

impl Default for GetRowsKernel {
    fn default() -> Self {
        Self::new()
    }
}

impl GetRowsKernel {
    /// Constructs an independent row-gather actor.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            machine: GetRowsMachineStateMachine::new(Context),
            bytes_machine: GetRowsBytesMachineStateMachine::new(BytesContext),
        }
    }

    /// Dispatches one typed event synchronously to completion.
    pub fn process_event<E: GetRowsEvent>(&mut self, event: E) -> E::Output {
        let output = event.dispatch(self);
        let _ = self.machine.is(&GetRowsMachineStates::Ready);
        let _ = self.bytes_machine.is(&GetRowsBytesMachineStates::Ready);
        output
    }

    fn get_rows(&mut self, event: OpGetRows<'_>) -> GetRowsOutcome {
        let result = Cell::new(Err(GetRowsError::UnexpectedEvent));
        self.machine
            .process_event(GetRowsMachineEvents::GetRows(GetRowsRuntime {
                event,
                result: &result,
            }))
            .map_err(|_| GetRowsError::Internal)?;
        result.get()
    }

    fn get_rows_f32_bytes(&mut self, event: OpGetRowsF32Bytes<'_>) -> GetRowsOutcome {
        let result = Cell::new(Err(GetRowsError::UnexpectedEvent));
        self.bytes_machine
            .process_event(GetRowsBytesMachineEvents::GetRowsF32Bytes(
                GetRowsF32BytesRuntime {
                    event,
                    result: &result,
                },
            ))
            .map_err(|_| GetRowsError::Internal)?;
        result.get()
    }

    fn get_rows_f16(&mut self, event: OpGetRowsF16<'_>) -> GetRowsOutcome {
        let result = Cell::new(Err(GetRowsError::UnexpectedEvent));
        self.machine
            .process_event(GetRowsMachineEvents::GetRowsF16(GetRowsF16Runtime {
                event,
                result: &result,
            }))
            .map_err(|_| GetRowsError::Internal)?;
        result.get()
    }

    fn get_rows_bf16(&mut self, event: OpGetRowsBf16<'_>) -> GetRowsOutcome {
        let result = Cell::new(Err(GetRowsError::UnexpectedEvent));
        self.machine
            .process_event(GetRowsMachineEvents::GetRowsBf16(GetRowsBf16Runtime {
                event,
                result: &result,
            }))
            .map_err(|_| GetRowsError::Internal)?;
        result.get()
    }

    fn get_rows_q4_0(&mut self, event: OpGetRowsQ4_0<'_>) -> GetRowsOutcome {
        let result = Cell::new(Err(GetRowsError::UnexpectedEvent));
        let destination = QuantizedOutputSlot::new(event.destination);
        self.machine
            .process_event(GetRowsMachineEvents::GetRowsQ4_0(GetRowsQ4_0Runtime {
                source: event.source,
                indices: event.indices,
                destination: destination.view(),
                result: &result,
            }))
            .map_err(|_| GetRowsError::Internal)?;
        result.get()
    }

    fn get_rows_q8_0(&mut self, event: OpGetRowsQ8_0<'_>) -> GetRowsOutcome {
        let result = Cell::new(Err(GetRowsError::UnexpectedEvent));
        let destination = QuantizedOutputSlot::new(event.destination);
        self.machine
            .process_event(GetRowsMachineEvents::GetRowsQ8_0(GetRowsQ8_0Runtime {
                source: event.source,
                indices: event.indices,
                destination: destination.view(),
                result: &result,
            }))
            .map_err(|_| GetRowsError::Internal)?;
        result.get()
    }

    fn get_rows_q4_k(&mut self, event: OpGetRowsQ4K<'_>) -> GetRowsOutcome {
        let result = Cell::new(Err(GetRowsError::UnexpectedEvent));
        let destination = QuantizedOutputSlot::new(event.destination);
        self.machine
            .process_event(GetRowsMachineEvents::GetRowsQ4K(GetRowsQ4KRuntime {
                source: event.source,
                indices: event.indices,
                destination: destination.view(),
                result: &result,
            }))
            .map_err(|_| GetRowsError::Internal)?;
        result.get()
    }

    fn unexpected(&mut self, _event: UnexpectedGetRows) -> GetRowsOutcome {
        let result = Cell::new(Err(GetRowsError::Internal));
        self.machine
            .process_event(GetRowsMachineEvents::Unexpected(UnexpectedRuntime {
                result: &result,
            }))
            .map_err(|_| GetRowsError::Internal)?;
        result.get()
    }
}

/// Trait implemented by the public row-gather event.
pub trait GetRowsEvent: Sized {
    /// Result returned after RTC dispatch.
    type Output;

    #[doc(hidden)]
    fn dispatch(self, actor: &mut GetRowsKernel) -> Self::Output;
}

impl GetRowsEvent for OpGetRows<'_> {
    type Output = GetRowsOutcome;

    fn dispatch(self, actor: &mut GetRowsKernel) -> Self::Output {
        actor.get_rows(self)
    }
}

impl GetRowsEvent for OpGetRowsF32Bytes<'_> {
    type Output = GetRowsOutcome;

    fn dispatch(self, actor: &mut GetRowsKernel) -> Self::Output {
        actor.get_rows_f32_bytes(self)
    }
}

impl GetRowsEvent for UnexpectedGetRows {
    type Output = GetRowsOutcome;

    fn dispatch(self, actor: &mut GetRowsKernel) -> Self::Output {
        actor.unexpected(self)
    }
}

impl GetRowsEvent for OpGetRowsF16<'_> {
    type Output = GetRowsOutcome;

    fn dispatch(self, actor: &mut GetRowsKernel) -> Self::Output {
        actor.get_rows_f16(self)
    }
}

impl GetRowsEvent for OpGetRowsBf16<'_> {
    type Output = GetRowsOutcome;

    fn dispatch(self, actor: &mut GetRowsKernel) -> Self::Output {
        actor.get_rows_bf16(self)
    }
}

impl GetRowsEvent for OpGetRowsQ4_0<'_> {
    type Output = GetRowsOutcome;

    fn dispatch(self, actor: &mut GetRowsKernel) -> Self::Output {
        actor.get_rows_q4_0(self)
    }
}

impl GetRowsEvent for OpGetRowsQ8_0<'_> {
    type Output = GetRowsOutcome;

    fn dispatch(self, actor: &mut GetRowsKernel) -> Self::Output {
        actor.get_rows_q8_0(self)
    }
}

impl GetRowsEvent for OpGetRowsQ4K<'_> {
    type Output = GetRowsOutcome;

    fn dispatch(self, actor: &mut GetRowsKernel) -> Self::Output {
        actor.get_rows_q4_k(self)
    }
}

fn views_valid(event: &GetRowsRuntime<'_>) -> bool {
    event.event.indices.validate()
        && if event.event.indices.count() == 0 {
            empty_index_source_valid(event) && destination_valid(event)
        } else {
            event.event.src.validate_binary().is_ok() && destination_valid(event)
        }
}

fn views_valid_f32_bytes(event: &GetRowsF32BytesRuntime<'_>) -> bool {
    event.event.indices.validate()
        && if event.event.indices.count() == 0 {
            empty_f32_bytes_source_valid(event) && destination_valid_f32_bytes(event)
        } else {
            event.event.source.validate().is_ok() && destination_valid_f32_bytes(event)
        }
}

fn empty_f32_bytes_source_valid(event: &GetRowsF32BytesRuntime<'_>) -> bool {
    let layout = event.event.source.layout();
    if layout.dtype() != DType::F32 || layout.ne()[0] == 0 || layout.ne()[1] == 0 {
        return false;
    }
    let nb = layout.nb();
    if nb[0] != 0 && (nb[0] < 4 || !nb[0].is_multiple_of(4)) {
        return false;
    }
    if nb[0] != 0 {
        let ne = layout.ne();
        let mut dimension = 0;
        while dimension < 4 {
            if ne[dimension] > 1 && nb[dimension] == 0 {
                return false;
            }
            dimension += 1;
        }
    }
    true
}

fn destination_valid_f32_bytes(event: &GetRowsF32BytesRuntime<'_>) -> bool {
    if event.event.indices.count() == 0 {
        event.event.destination.layout().dtype() == DType::F32
            && event.event.destination.layout().is_dense_contiguous()
    } else {
        event.event.destination.validate().is_ok()
            && event.event.destination.layout().is_dense_contiguous()
    }
}

fn shape_valid_f32_bytes(event: &GetRowsF32BytesRuntime<'_>) -> bool {
    let src = event.event.source.layout().ne();
    let indices = event.event.indices.ne();
    let dst = event.event.destination.layout().ne();
    src[0] > 0
        && src[1] > 0
        && dst == [src[0], indices[0], indices[1], indices[2]]
        && src[2] == indices[1]
        && src[3] == indices[2]
}

const fn indices_valid_f32_bytes(event: &GetRowsF32BytesRuntime<'_>) -> bool {
    let rows = event.event.source.layout().ne()[1];
    let mut ordinal = 0;
    while ordinal < event.event.indices.count() {
        let row = event.event.indices.read(ordinal);
        if row < 0 || (row.unsigned_abs() as u64) >= rows {
            return false;
        }
        ordinal += 1;
    }
    true
}

fn empty_index_source_valid(event: &GetRowsRuntime<'_>) -> bool {
    let layout = event.event.src.layout();
    if layout.dtype() != super::tensor_view::DType::F32
        || layout.ne()[0] == 0
        || layout.ne()[1] == 0
        || (layout.nb()[0] != 0 && layout.nb()[0] != 4)
    {
        return false;
    }

    if layout.nb()[0] != 0 {
        let ne = layout.ne();
        let nb = layout.nb();
        let mut dimension = 0;
        while dimension < 4 {
            if ne[dimension] > 1 && nb[dimension] == 0 {
                return false;
            }
            dimension += 1;
        }
    }
    true
}

fn destination_valid(event: &GetRowsRuntime<'_>) -> bool {
    if event.event.indices.count() == 0 {
        event.event.dst.layout().dtype() == super::tensor_view::DType::F32
            && event.event.dst.layout().is_dense_contiguous()
    } else {
        event.event.dst.validate().is_ok() && event.event.dst.layout().is_dense_contiguous()
    }
}

const fn source_stride_valid(event: &GetRowsRuntime<'_>) -> bool {
    let stride = event.event.src.layout().nb()[0];
    stride == 0 || (stride >= 4 && stride.is_multiple_of(4))
}

fn shape_valid(event: &GetRowsRuntime<'_>) -> bool {
    let src = event.event.src.layout().ne();
    let indices = event.event.indices.ne();
    let dst = event.event.dst.layout().ne();
    src[0] > 0
        && src[1] > 0
        && dst == [src[0], indices[0], indices[1], indices[2]]
        && src[2] == indices[1]
        && src[3] == indices[2]
}

fn views_valid_f16(event: &GetRowsF16Runtime<'_>) -> bool {
    event.event.indices.validate()
        && event.event.source.validate()
        && event.event.destination.validate().is_ok()
        && event.event.destination.layout().is_dense_contiguous()
}

fn shape_valid_f16(event: &GetRowsF16Runtime<'_>) -> bool {
    let source = event.event.source.layout().ne();
    let indices = event.event.indices.ne();
    let destination = event.event.destination.layout().ne();
    source[0] > 0
        && source[1] > 0
        && destination == [source[0], indices[0], indices[1], indices[2]]
        && source[2] == indices[1]
        && source[3] == indices[2]
}

fn views_valid_bf16(event: &GetRowsBf16Runtime<'_>) -> bool {
    if event.event.indices.count() == 0 {
        event.event.indices.validate()
            && empty_bf16_source_valid(event)
            && empty_f32_destination_valid(event.event.destination.layout())
    } else {
        event.event.indices.validate()
            && event.event.source.validate()
            && event.event.destination.validate().is_ok()
            && event.event.destination.layout().is_dense_contiguous()
    }
}

fn empty_bf16_source_valid(event: &GetRowsBf16Runtime<'_>) -> bool {
    let layout = event.event.source.layout();
    if layout.dtype() != DType::Bf16
        || layout.ne()[0] == 0
        || layout.ne()[1] == 0
        || (layout.nb()[0] != 0 && layout.nb()[0] != 2)
    {
        return false;
    }
    if layout.nb()[0] != 0 {
        let ne = layout.ne();
        let nb = layout.nb();
        let mut dimension = 0;
        while dimension < 4 {
            if ne[dimension] > 1 && nb[dimension] == 0 {
                return false;
            }
            dimension += 1;
        }
    }
    true
}

fn empty_f32_destination_valid(layout: Layout) -> bool {
    layout.dtype() == DType::F32 && layout.is_dense_contiguous()
}

fn shape_valid_bf16(event: &GetRowsBf16Runtime<'_>) -> bool {
    let source = event.event.source.layout().ne();
    let indices = event.event.indices.ne();
    let destination = event.event.destination.layout().ne();
    source[0] > 0
        && source[1] > 0
        && destination == [source[0], indices[0], indices[1], indices[2]]
        && source[2] == indices[1]
        && source[3] == indices[2]
}

fn quantized_views_valid(
    source: QuantizedView<'_>,
    indices: &IndexView<'_>,
    destination: QuantizedOutput<'_, '_>,
    code: u8,
    block_values: usize,
    block_bytes: usize,
) -> bool {
    indices.validate()
        && source.validate(code, block_values, block_bytes)
        && if indices.count() == 0 {
            empty_f32_destination_valid(destination.layout())
        } else {
            destination.validate() && destination.layout().is_dense_contiguous()
        }
        && destination.layout().dtype() == DType::F32
}

fn quantized_shape_valid(
    source: QuantizedView<'_>,
    indices: &IndexView<'_>,
    destination: QuantizedOutput<'_, '_>,
) -> bool {
    let source_ne = source.layout().ne();
    let indices_ne = indices.ne();
    let destination_ne = destination.layout().ne();
    source_ne[0] > 0
        && source_ne[1] > 0
        && destination_ne == [source_ne[0], indices_ne[0], indices_ne[1], indices_ne[2]]
        && source_ne[2] == indices_ne[1]
        && source_ne[3] == indices_ne[2]
}

const fn quantized_indices_valid(source: QuantizedView<'_>, indices: &IndexView<'_>) -> bool {
    let rows = source.layout().ne()[1];
    let mut ordinal = 0;
    while ordinal < indices.count() {
        let row = indices.read(ordinal);
        if row < 0 || (row.unsigned_abs() as u64) >= rows {
            return false;
        }
        ordinal += 1;
    }
    true
}

fn views_valid_q4_0(event: &GetRowsQ4_0Runtime<'_, '_, '_>) -> bool {
    quantized_views_valid(
        event.source,
        &event.indices,
        event.destination,
        Q4_0_CODE,
        BLOCK_VALUES,
        Q4_0_BLOCK_BYTES,
    )
}

fn views_valid_q8_0(event: &GetRowsQ8_0Runtime<'_, '_, '_>) -> bool {
    quantized_views_valid(
        event.source,
        &event.indices,
        event.destination,
        Q8_0_CODE,
        BLOCK_VALUES,
        Q8_0_BLOCK_BYTES,
    )
}

fn views_valid_q4_k(event: &GetRowsQ4KRuntime<'_, '_, '_>) -> bool {
    quantized_views_valid(
        event.source,
        &event.indices,
        event.destination,
        Q4_K_CODE,
        QK_K_VALUES,
        Q4_K_BLOCK_BYTES,
    )
}

fn shape_valid_q4_0(event: &GetRowsQ4_0Runtime<'_, '_, '_>) -> bool {
    quantized_shape_valid(event.source, &event.indices, event.destination)
}

fn shape_valid_q8_0(event: &GetRowsQ8_0Runtime<'_, '_, '_>) -> bool {
    quantized_shape_valid(event.source, &event.indices, event.destination)
}

fn shape_valid_q4_k(event: &GetRowsQ4KRuntime<'_, '_, '_>) -> bool {
    quantized_shape_valid(event.source, &event.indices, event.destination)
}

const fn indices_valid_q4_0(event: &GetRowsQ4_0Runtime<'_, '_, '_>) -> bool {
    quantized_indices_valid(event.source, &event.indices)
}

const fn indices_valid_q8_0(event: &GetRowsQ8_0Runtime<'_, '_, '_>) -> bool {
    quantized_indices_valid(event.source, &event.indices)
}

const fn indices_valid_q4_k(event: &GetRowsQ4KRuntime<'_, '_, '_>) -> bool {
    quantized_indices_valid(event.source, &event.indices)
}

const fn indices_valid_f16(event: &GetRowsF16Runtime<'_>) -> bool {
    let rows = event.event.source.layout().ne()[1];
    let mut ordinal = 0;
    while ordinal < event.event.indices.count() {
        let row = event.event.indices.read(ordinal);
        if row < 0 || (row.unsigned_abs() as u64) >= rows {
            return false;
        }
        ordinal += 1;
    }
    true
}

const fn indices_valid_bf16(event: &GetRowsBf16Runtime<'_>) -> bool {
    let rows = event.event.source.layout().ne()[1];
    let mut ordinal = 0;
    while ordinal < event.event.indices.count() {
        let row = event.event.indices.read(ordinal);
        if row < 0 || (row.unsigned_abs() as u64) >= rows {
            return false;
        }
        ordinal += 1;
    }
    true
}

const fn indices_valid(event: &GetRowsRuntime<'_>) -> bool {
    let rows = event.event.src.layout().ne()[1];
    let mut ordinal = 0;
    while ordinal < event.event.indices.count() {
        let row = event.event.indices.read(ordinal);
        if row < 0 || (row.unsigned_abs() as u64) >= rows {
            return false;
        }
        ordinal += 1;
    }
    true
}

impl GetRowsMachineStateMachineContext for Context {
    fn guard_get_rows_valid(&self, event: &GetRowsRuntime<'_>) -> Result<bool, ()> {
        Ok(views_valid(event)
            && shape_valid(event)
            && source_stride_valid(event)
            && indices_valid(event))
    }

    fn guard_get_rows_shape_invalid(&self, event: &GetRowsRuntime<'_>) -> Result<bool, ()> {
        Ok(views_valid(event) && !shape_valid(event))
    }

    fn guard_get_rows_stride_invalid(&self, event: &GetRowsRuntime<'_>) -> Result<bool, ()> {
        Ok(views_valid(event) && shape_valid(event) && !source_stride_valid(event))
    }

    fn guard_get_rows_index_invalid(&self, event: &GetRowsRuntime<'_>) -> Result<bool, ()> {
        Ok(views_valid(event)
            && shape_valid(event)
            && source_stride_valid(event)
            && !indices_valid(event))
    }

    fn guard_get_rows_invalid_view(&self, event: &GetRowsRuntime<'_>) -> Result<bool, ()> {
        Ok(!views_valid(event))
    }

    fn guard_get_rows_f16_valid(&self, event: &GetRowsF16Runtime<'_>) -> Result<bool, ()> {
        Ok(views_valid_f16(event) && shape_valid_f16(event) && indices_valid_f16(event))
    }

    fn guard_get_rows_f16_shape_invalid(&self, event: &GetRowsF16Runtime<'_>) -> Result<bool, ()> {
        Ok(views_valid_f16(event) && !shape_valid_f16(event))
    }

    fn guard_get_rows_f16_index_invalid(&self, event: &GetRowsF16Runtime<'_>) -> Result<bool, ()> {
        Ok(views_valid_f16(event) && shape_valid_f16(event) && !indices_valid_f16(event))
    }

    fn guard_get_rows_f16_invalid_view(&self, event: &GetRowsF16Runtime<'_>) -> Result<bool, ()> {
        Ok(!views_valid_f16(event))
    }

    fn guard_get_rows_bf16_valid(&self, event: &GetRowsBf16Runtime<'_>) -> Result<bool, ()> {
        Ok(views_valid_bf16(event) && shape_valid_bf16(event) && indices_valid_bf16(event))
    }

    fn guard_get_rows_bf16_shape_invalid(
        &self,
        event: &GetRowsBf16Runtime<'_>,
    ) -> Result<bool, ()> {
        Ok(views_valid_bf16(event) && !shape_valid_bf16(event))
    }

    fn guard_get_rows_bf16_index_invalid(
        &self,
        event: &GetRowsBf16Runtime<'_>,
    ) -> Result<bool, ()> {
        Ok(views_valid_bf16(event) && shape_valid_bf16(event) && !indices_valid_bf16(event))
    }

    fn guard_get_rows_bf16_invalid_view(&self, event: &GetRowsBf16Runtime<'_>) -> Result<bool, ()> {
        Ok(!views_valid_bf16(event))
    }

    fn guard_get_rows_q4_0_valid<'dispatch, 'data, 'slot>(
        &self,
        event: &GetRowsQ4_0Runtime<'dispatch, 'data, 'slot>,
    ) -> Result<bool, ()>
    where
        'data: 'slot,
    {
        Ok(views_valid_q4_0(event) && shape_valid_q4_0(event) && indices_valid_q4_0(event))
    }

    fn guard_get_rows_q4_0_shape_invalid<'dispatch, 'data, 'slot>(
        &self,
        event: &GetRowsQ4_0Runtime<'dispatch, 'data, 'slot>,
    ) -> Result<bool, ()>
    where
        'data: 'slot,
    {
        Ok(views_valid_q4_0(event) && !shape_valid_q4_0(event))
    }

    fn guard_get_rows_q4_0_index_invalid<'dispatch, 'data, 'slot>(
        &self,
        event: &GetRowsQ4_0Runtime<'dispatch, 'data, 'slot>,
    ) -> Result<bool, ()>
    where
        'data: 'slot,
    {
        Ok(views_valid_q4_0(event) && shape_valid_q4_0(event) && !indices_valid_q4_0(event))
    }

    fn guard_get_rows_q4_0_invalid_view<'dispatch, 'data, 'slot>(
        &self,
        event: &GetRowsQ4_0Runtime<'dispatch, 'data, 'slot>,
    ) -> Result<bool, ()>
    where
        'data: 'slot,
    {
        Ok(!views_valid_q4_0(event))
    }

    fn guard_get_rows_q8_0_valid<'dispatch, 'data, 'slot>(
        &self,
        event: &GetRowsQ8_0Runtime<'dispatch, 'data, 'slot>,
    ) -> Result<bool, ()>
    where
        'data: 'slot,
    {
        Ok(views_valid_q8_0(event) && shape_valid_q8_0(event) && indices_valid_q8_0(event))
    }

    fn guard_get_rows_q8_0_shape_invalid<'dispatch, 'data, 'slot>(
        &self,
        event: &GetRowsQ8_0Runtime<'dispatch, 'data, 'slot>,
    ) -> Result<bool, ()>
    where
        'data: 'slot,
    {
        Ok(views_valid_q8_0(event) && !shape_valid_q8_0(event))
    }

    fn guard_get_rows_q8_0_index_invalid<'dispatch, 'data, 'slot>(
        &self,
        event: &GetRowsQ8_0Runtime<'dispatch, 'data, 'slot>,
    ) -> Result<bool, ()>
    where
        'data: 'slot,
    {
        Ok(views_valid_q8_0(event) && shape_valid_q8_0(event) && !indices_valid_q8_0(event))
    }

    fn guard_get_rows_q8_0_invalid_view<'dispatch, 'data, 'slot>(
        &self,
        event: &GetRowsQ8_0Runtime<'dispatch, 'data, 'slot>,
    ) -> Result<bool, ()>
    where
        'data: 'slot,
    {
        Ok(!views_valid_q8_0(event))
    }

    fn guard_get_rows_q4_k_valid<'dispatch, 'data, 'slot>(
        &self,
        event: &GetRowsQ4KRuntime<'dispatch, 'data, 'slot>,
    ) -> Result<bool, ()>
    where
        'data: 'slot,
    {
        Ok(views_valid_q4_k(event) && shape_valid_q4_k(event) && indices_valid_q4_k(event))
    }

    fn guard_get_rows_q4_k_shape_invalid<'dispatch, 'data, 'slot>(
        &self,
        event: &GetRowsQ4KRuntime<'dispatch, 'data, 'slot>,
    ) -> Result<bool, ()>
    where
        'data: 'slot,
    {
        Ok(views_valid_q4_k(event) && !shape_valid_q4_k(event))
    }

    fn guard_get_rows_q4_k_index_invalid<'dispatch, 'data, 'slot>(
        &self,
        event: &GetRowsQ4KRuntime<'dispatch, 'data, 'slot>,
    ) -> Result<bool, ()>
    where
        'data: 'slot,
    {
        Ok(views_valid_q4_k(event) && shape_valid_q4_k(event) && !indices_valid_q4_k(event))
    }

    fn guard_get_rows_q4_k_invalid_view<'dispatch, 'data, 'slot>(
        &self,
        event: &GetRowsQ4KRuntime<'dispatch, 'data, 'slot>,
    ) -> Result<bool, ()>
    where
        'data: 'slot,
    {
        Ok(!views_valid_q4_k(event))
    }

    #[allow(clippy::cast_possible_truncation)]
    fn effect_get_rows_execute(&mut self, mut event: GetRowsRuntime<'_>) -> Result<(), ()> {
        let source_ne = event.event.src.layout().ne();
        let indices_ne = event.event.indices.ne();
        let cols = source_ne[0] as usize;
        let mut dst_row = 0usize;
        let mut i2 = 0usize;
        while i2 < indices_ne[2] as usize {
            let mut i1 = 0usize;
            while i1 < indices_ne[1] as usize {
                let mut i0 = 0usize;
                while i0 < indices_ne[0] as usize {
                    let index = event.event.indices.read(dst_row).unsigned_abs() as usize;
                    let mut column = 0usize;
                    while column < cols {
                        let value = event.event.src.read_row_contiguous(index, i1, i2, column);
                        event.event.dst.write(dst_row * cols + column, value);
                        column += 1;
                    }
                    dst_row += 1;
                    i0 += 1;
                }
                i1 += 1;
            }
            i2 += 1;
        }
        event.result.set(Ok(()));
        Ok(())
    }

    fn effect_get_rows_shape_reject(&mut self, event: GetRowsRuntime<'_>) -> Result<(), ()> {
        event.result.set(Err(GetRowsError::ShapeMismatch));
        Ok(())
    }

    fn effect_get_rows_index_reject(&mut self, event: GetRowsRuntime<'_>) -> Result<(), ()> {
        event.result.set(Err(GetRowsError::IndexOutOfBounds));
        Ok(())
    }

    fn effect_get_rows_stride_reject(&mut self, event: GetRowsRuntime<'_>) -> Result<(), ()> {
        event.result.set(Err(GetRowsError::UnsupportedSourceStride));
        Ok(())
    }

    fn effect_get_rows_view_reject(&mut self, event: GetRowsRuntime<'_>) -> Result<(), ()> {
        event.result.set(Err(GetRowsError::InvalidView));
        Ok(())
    }

    #[allow(clippy::cast_possible_truncation)]
    fn effect_get_rows_f16_execute(&mut self, mut event: GetRowsF16Runtime<'_>) -> Result<(), ()> {
        let source_ne = event.event.source.layout().ne();
        let indices_ne = event.event.indices.ne();
        let cols = usize::try_from(source_ne[0]).expect("guard-proven cols fits usize");
        let rows = usize::try_from(source_ne[1]).expect("guard-proven rows fits usize");
        let source_batches =
            usize::try_from(source_ne[2]).expect("guard-proven source batch fits usize");
        let mut destination_row = 0usize;
        let mut i2 = 0usize;
        while i2 < indices_ne[2] as usize {
            let mut i1 = 0usize;
            while i1 < indices_ne[1] as usize {
                let mut i0 = 0usize;
                while i0 < indices_ne[0] as usize {
                    let index = event.event.indices.read(destination_row).unsigned_abs() as usize;
                    let source_row =
                        i2 * source_batches * rows * cols + i1 * rows * cols + index * cols;
                    let mut column = 0usize;
                    while column < cols {
                        let value = event.event.source.read(source_row + column);
                        event
                            .event
                            .destination
                            .write(destination_row * cols + column, value);
                        column += 1;
                    }
                    destination_row += 1;
                    i0 += 1;
                }
                i1 += 1;
            }
            i2 += 1;
        }
        event.result.set(Ok(()));
        Ok(())
    }

    fn effect_get_rows_shape_reject_f16(&mut self, event: GetRowsF16Runtime<'_>) -> Result<(), ()> {
        event.result.set(Err(GetRowsError::ShapeMismatch));
        Ok(())
    }

    fn effect_get_rows_index_reject_f16(&mut self, event: GetRowsF16Runtime<'_>) -> Result<(), ()> {
        event.result.set(Err(GetRowsError::IndexOutOfBounds));
        Ok(())
    }

    fn effect_get_rows_view_reject_f16(&mut self, event: GetRowsF16Runtime<'_>) -> Result<(), ()> {
        event.result.set(Err(GetRowsError::InvalidView));
        Ok(())
    }

    #[allow(clippy::cast_possible_truncation)]
    fn effect_get_rows_bf16_execute(
        &mut self,
        mut event: GetRowsBf16Runtime<'_>,
    ) -> Result<(), ()> {
        let source_ne = event.event.source.layout().ne();
        let indices_ne = event.event.indices.ne();
        let cols = usize::try_from(source_ne[0]).expect("guard-proven cols fits usize");
        let rows = usize::try_from(source_ne[1]).expect("guard-proven rows fits usize");
        let source_batches =
            usize::try_from(source_ne[2]).expect("guard-proven source batch fits usize");
        let mut destination_row = 0usize;
        let mut i2 = 0usize;
        while i2 < indices_ne[2] as usize {
            let mut i1 = 0usize;
            while i1 < indices_ne[1] as usize {
                let mut i0 = 0usize;
                while i0 < indices_ne[0] as usize {
                    let index = event.event.indices.read(destination_row).unsigned_abs() as usize;
                    let source_row =
                        i2 * source_batches * rows * cols + i1 * rows * cols + index * cols;
                    let mut column = 0usize;
                    while column < cols {
                        let value = event.event.source.read(source_row + column);
                        event
                            .event
                            .destination
                            .write(destination_row * cols + column, value);
                        column += 1;
                    }
                    destination_row += 1;
                    i0 += 1;
                }
                i1 += 1;
            }
            i2 += 1;
        }
        event.result.set(Ok(()));
        Ok(())
    }

    fn effect_get_rows_shape_reject_bf16(
        &mut self,
        event: GetRowsBf16Runtime<'_>,
    ) -> Result<(), ()> {
        event.result.set(Err(GetRowsError::ShapeMismatch));
        Ok(())
    }

    fn effect_get_rows_index_reject_bf16(
        &mut self,
        event: GetRowsBf16Runtime<'_>,
    ) -> Result<(), ()> {
        event.result.set(Err(GetRowsError::IndexOutOfBounds));
        Ok(())
    }

    fn effect_get_rows_view_reject_bf16(
        &mut self,
        event: GetRowsBf16Runtime<'_>,
    ) -> Result<(), ()> {
        event.result.set(Err(GetRowsError::InvalidView));
        Ok(())
    }

    fn effect_get_rows_q4_0_execute<'dispatch, 'data, 'slot>(
        &mut self,
        event: GetRowsQ4_0Runtime<'dispatch, 'data, 'slot>,
    ) -> Result<(), ()>
    where
        'data: 'slot,
    {
        let source_ne = event.source.layout().ne();
        let indices_ne = event.indices.ne();
        let cols = usize::try_from(source_ne[0]).expect("guard-proven Q4_0 cols fit usize");
        let rows = usize::try_from(source_ne[1]).expect("guard-proven Q4_0 rows fit usize");
        let source_batches =
            usize::try_from(source_ne[2]).expect("guard-proven Q4_0 batches fit usize");
        let blocks = cols / BLOCK_VALUES;
        let row_bytes = blocks
            .checked_mul(Q4_0_BLOCK_BYTES)
            .expect("guard-proven Q4_0 row bytes fit usize");
        let mut decoded = [0.0_f32; BLOCK_VALUES];
        let mut destination_row = 0usize;
        let mut i2 = 0usize;
        while i2 < usize::try_from(indices_ne[2]).expect("guard-proven indices fit usize") {
            let mut i1 = 0usize;
            while i1 < usize::try_from(indices_ne[1]).expect("guard-proven indices fit usize") {
                let mut i0 = 0usize;
                while i0 < usize::try_from(indices_ne[0]).expect("guard-proven indices fit usize") {
                    let index = event.indices.read(destination_row).unsigned_abs() as usize;
                    let source_row = i2
                        .checked_mul(source_batches)
                        .and_then(|value| value.checked_add(i1))
                        .and_then(|value| value.checked_mul(rows))
                        .and_then(|value| value.checked_add(index))
                        .expect("guard-proven Q4_0 source row fits usize");
                    let packed_row = event.source.row(source_row, row_bytes);
                    let row = Q4_0Row::from_bytes(packed_row)
                        .expect("guard-proven Q4_0 row has complete blocks");
                    let mut block_index = 0usize;
                    while block_index < blocks {
                        row.decode_block(block_index, &mut decoded)
                            .expect("guard-proven Q4_0 block is present");
                        let mut column = 0usize;
                        while column < BLOCK_VALUES {
                            event.destination.write(
                                destination_row * cols + block_index * BLOCK_VALUES + column,
                                decoded[column],
                            );
                            column += 1;
                        }
                        block_index += 1;
                    }
                    destination_row += 1;
                    i0 += 1;
                }
                i1 += 1;
            }
            i2 += 1;
        }
        event.result.set(Ok(()));
        Ok(())
    }

    fn effect_get_rows_q8_0_execute<'dispatch, 'data, 'slot>(
        &mut self,
        event: GetRowsQ8_0Runtime<'dispatch, 'data, 'slot>,
    ) -> Result<(), ()>
    where
        'data: 'slot,
    {
        let source_ne = event.source.layout().ne();
        let indices_ne = event.indices.ne();
        let cols = usize::try_from(source_ne[0]).expect("guard-proven Q8_0 cols fit usize");
        let rows = usize::try_from(source_ne[1]).expect("guard-proven Q8_0 rows fit usize");
        let source_batches =
            usize::try_from(source_ne[2]).expect("guard-proven Q8_0 batches fit usize");
        let blocks = cols / BLOCK_VALUES;
        let row_bytes = blocks
            .checked_mul(Q8_0_BLOCK_BYTES)
            .expect("guard-proven Q8_0 row bytes fit usize");
        let mut decoded = [0.0_f32; BLOCK_VALUES];
        let mut destination_row = 0usize;
        let mut i2 = 0usize;
        while i2 < usize::try_from(indices_ne[2]).expect("guard-proven indices fit usize") {
            let mut i1 = 0usize;
            while i1 < usize::try_from(indices_ne[1]).expect("guard-proven indices fit usize") {
                let mut i0 = 0usize;
                while i0 < usize::try_from(indices_ne[0]).expect("guard-proven indices fit usize") {
                    let index = event.indices.read(destination_row).unsigned_abs() as usize;
                    let source_row = i2
                        .checked_mul(source_batches)
                        .and_then(|value| value.checked_add(i1))
                        .and_then(|value| value.checked_mul(rows))
                        .and_then(|value| value.checked_add(index))
                        .expect("guard-proven Q8_0 source row fits usize");
                    let packed_row = event.source.row(source_row, row_bytes);
                    let row = Q8_0Row::from_bytes(packed_row)
                        .expect("guard-proven Q8_0 row has complete blocks");
                    let mut block_index = 0usize;
                    while block_index < blocks {
                        row.decode_block(block_index, &mut decoded)
                            .expect("guard-proven Q8_0 block is present");
                        let mut column = 0usize;
                        while column < BLOCK_VALUES {
                            event.destination.write(
                                destination_row * cols + block_index * BLOCK_VALUES + column,
                                decoded[column],
                            );
                            column += 1;
                        }
                        block_index += 1;
                    }
                    destination_row += 1;
                    i0 += 1;
                }
                i1 += 1;
            }
            i2 += 1;
        }
        event.result.set(Ok(()));
        Ok(())
    }

    fn effect_get_rows_q4_k_execute<'dispatch, 'data, 'slot>(
        &mut self,
        event: GetRowsQ4KRuntime<'dispatch, 'data, 'slot>,
    ) -> Result<(), ()>
    where
        'data: 'slot,
    {
        let source_ne = event.source.layout().ne();
        let indices_ne = event.indices.ne();
        let cols = usize::try_from(source_ne[0]).expect("guard-proven Q4_K cols fit usize");
        let rows = usize::try_from(source_ne[1]).expect("guard-proven Q4_K rows fit usize");
        let source_batches =
            usize::try_from(source_ne[2]).expect("guard-proven Q4_K batches fit usize");
        let blocks = cols / QK_K_VALUES;
        let row_bytes = blocks
            .checked_mul(Q4_K_BLOCK_BYTES)
            .expect("guard-proven Q4_K row bytes fit usize");
        let mut decoded = [0.0_f32; QK_K_VALUES];
        let mut destination_row = 0usize;
        let mut i2 = 0usize;
        while i2 < usize::try_from(indices_ne[2]).expect("guard-proven indices fit usize") {
            let mut i1 = 0usize;
            while i1 < usize::try_from(indices_ne[1]).expect("guard-proven indices fit usize") {
                let mut i0 = 0usize;
                while i0 < usize::try_from(indices_ne[0]).expect("guard-proven indices fit usize") {
                    let index = event.indices.read(destination_row).unsigned_abs() as usize;
                    let source_row = i2
                        .checked_mul(source_batches)
                        .and_then(|value| value.checked_add(i1))
                        .and_then(|value| value.checked_mul(rows))
                        .and_then(|value| value.checked_add(index))
                        .expect("guard-proven Q4_K source row fits usize");
                    let packed_row = event.source.row(source_row, row_bytes);
                    let row = Q4KRow::from_bytes(packed_row)
                        .expect("guard-proven Q4_K row has complete blocks");
                    let mut block_index = 0usize;
                    while block_index < blocks {
                        row.decode_block(block_index, &mut decoded)
                            .expect("guard-proven Q4_K block is present");
                        let mut column = 0usize;
                        while column < QK_K_VALUES {
                            event.destination.write(
                                destination_row * cols + block_index * QK_K_VALUES + column,
                                decoded[column],
                            );
                            column += 1;
                        }
                        block_index += 1;
                    }
                    destination_row += 1;
                    i0 += 1;
                }
                i1 += 1;
            }
            i2 += 1;
        }
        event.result.set(Ok(()));
        Ok(())
    }

    fn effect_get_rows_shape_reject_q4_0<'dispatch, 'data, 'slot>(
        &mut self,
        event: GetRowsQ4_0Runtime<'dispatch, 'data, 'slot>,
    ) -> Result<(), ()>
    where
        'data: 'slot,
    {
        event.result.set(Err(GetRowsError::ShapeMismatch));
        Ok(())
    }

    fn effect_get_rows_index_reject_q4_0<'dispatch, 'data, 'slot>(
        &mut self,
        event: GetRowsQ4_0Runtime<'dispatch, 'data, 'slot>,
    ) -> Result<(), ()>
    where
        'data: 'slot,
    {
        event.result.set(Err(GetRowsError::IndexOutOfBounds));
        Ok(())
    }

    fn effect_get_rows_view_reject_q4_0<'dispatch, 'data, 'slot>(
        &mut self,
        event: GetRowsQ4_0Runtime<'dispatch, 'data, 'slot>,
    ) -> Result<(), ()>
    where
        'data: 'slot,
    {
        event.result.set(Err(GetRowsError::InvalidView));
        Ok(())
    }

    fn effect_get_rows_shape_reject_q8_0<'dispatch, 'data, 'slot>(
        &mut self,
        event: GetRowsQ8_0Runtime<'dispatch, 'data, 'slot>,
    ) -> Result<(), ()>
    where
        'data: 'slot,
    {
        event.result.set(Err(GetRowsError::ShapeMismatch));
        Ok(())
    }

    fn effect_get_rows_index_reject_q8_0<'dispatch, 'data, 'slot>(
        &mut self,
        event: GetRowsQ8_0Runtime<'dispatch, 'data, 'slot>,
    ) -> Result<(), ()>
    where
        'data: 'slot,
    {
        event.result.set(Err(GetRowsError::IndexOutOfBounds));
        Ok(())
    }

    fn effect_get_rows_view_reject_q8_0<'dispatch, 'data, 'slot>(
        &mut self,
        event: GetRowsQ8_0Runtime<'dispatch, 'data, 'slot>,
    ) -> Result<(), ()>
    where
        'data: 'slot,
    {
        event.result.set(Err(GetRowsError::InvalidView));
        Ok(())
    }

    fn effect_get_rows_shape_reject_q4_k<'dispatch, 'data, 'slot>(
        &mut self,
        event: GetRowsQ4KRuntime<'dispatch, 'data, 'slot>,
    ) -> Result<(), ()>
    where
        'data: 'slot,
    {
        event.result.set(Err(GetRowsError::ShapeMismatch));
        Ok(())
    }

    fn effect_get_rows_index_reject_q4_k<'dispatch, 'data, 'slot>(
        &mut self,
        event: GetRowsQ4KRuntime<'dispatch, 'data, 'slot>,
    ) -> Result<(), ()>
    where
        'data: 'slot,
    {
        event.result.set(Err(GetRowsError::IndexOutOfBounds));
        Ok(())
    }

    fn effect_get_rows_view_reject_q4_k<'dispatch, 'data, 'slot>(
        &mut self,
        event: GetRowsQ4KRuntime<'dispatch, 'data, 'slot>,
    ) -> Result<(), ()>
    where
        'data: 'slot,
    {
        event.result.set(Err(GetRowsError::InvalidView));
        Ok(())
    }

    fn effect_get_rows_unexpected(&mut self, event: UnexpectedRuntime<'_>) -> Result<(), ()> {
        event.result.set(Err(GetRowsError::UnexpectedEvent));
        Ok(())
    }

    fn effect_unexpected(&mut self) -> Result<(), ()> {
        Err(())
    }
}

impl GetRowsBytesMachineStateMachineContext for BytesContext {
    fn guard_get_rows_f32_bytes_valid(
        &self,
        event: &GetRowsF32BytesRuntime<'_>,
    ) -> Result<bool, ()> {
        Ok(views_valid_f32_bytes(event)
            && shape_valid_f32_bytes(event)
            && indices_valid_f32_bytes(event))
    }

    fn guard_get_rows_f32_bytes_shape_invalid(
        &self,
        event: &GetRowsF32BytesRuntime<'_>,
    ) -> Result<bool, ()> {
        Ok(views_valid_f32_bytes(event) && !shape_valid_f32_bytes(event))
    }

    fn guard_get_rows_f32_bytes_index_invalid(
        &self,
        event: &GetRowsF32BytesRuntime<'_>,
    ) -> Result<bool, ()> {
        Ok(views_valid_f32_bytes(event)
            && shape_valid_f32_bytes(event)
            && !indices_valid_f32_bytes(event))
    }

    fn guard_get_rows_f32_bytes_invalid_view(
        &self,
        event: &GetRowsF32BytesRuntime<'_>,
    ) -> Result<bool, ()> {
        Ok(!views_valid_f32_bytes(event))
    }

    #[allow(clippy::cast_possible_truncation)]
    fn effect_get_rows_f32_bytes_execute(
        &mut self,
        mut event: GetRowsF32BytesRuntime<'_>,
    ) -> Result<(), ()> {
        let source_ne = event.event.source.layout().ne();
        let indices_ne = event.event.indices.ne();
        let cols = usize::try_from(source_ne[0]).expect("guard-proven cols fits usize");
        let mut destination_row = 0usize;
        let mut i2 = 0usize;
        while i2 < indices_ne[2] as usize {
            let mut i1 = 0usize;
            while i1 < indices_ne[1] as usize {
                let mut i0 = 0usize;
                while i0 < indices_ne[0] as usize {
                    let index = event.event.indices.read(destination_row).unsigned_abs() as usize;
                    let mut column = 0usize;
                    while column < cols {
                        let value = event
                            .event
                            .source
                            .read_row_contiguous(index, i1, i2, column);
                        event
                            .event
                            .destination
                            .write(destination_row * cols + column, value);
                        column += 1;
                    }
                    destination_row += 1;
                    i0 += 1;
                }
                i1 += 1;
            }
            i2 += 1;
        }
        event.result.set(Ok(()));
        Ok(())
    }

    fn effect_get_rows_f32_bytes_shape_reject(
        &mut self,
        event: GetRowsF32BytesRuntime<'_>,
    ) -> Result<(), ()> {
        event.result.set(Err(GetRowsError::ShapeMismatch));
        Ok(())
    }

    fn effect_get_rows_f32_bytes_index_reject(
        &mut self,
        event: GetRowsF32BytesRuntime<'_>,
    ) -> Result<(), ()> {
        event.result.set(Err(GetRowsError::IndexOutOfBounds));
        Ok(())
    }

    fn effect_get_rows_f32_bytes_view_reject(
        &mut self,
        event: GetRowsF32BytesRuntime<'_>,
    ) -> Result<(), ()> {
        event.result.set(Err(GetRowsError::InvalidView));
        Ok(())
    }

    fn effect_unexpected(&mut self) -> Result<(), ()> {
        Err(())
    }
}

#[inline]
fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits(u32::from(bits) << 16)
}
