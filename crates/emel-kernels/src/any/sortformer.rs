//! Sortformer-owned fixed-width numeric kernels.

/// Reusable metadata for a prepared dense weight matrix.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct DenseWeightCache {
    source: usize,
    input_dim: usize,
    output_dim: usize,
}

/// Computes the fixed Sortformer encoder projection (512 inputs to 192 outputs).
pub fn encoder_projection(
    encoder_frame: &[f32],
    weights: &[f32],
    bias: &[f32],
    hidden_out: &mut [f32],
) -> bool {
    if encoder_frame.len() != 512
        || weights.len() != 192 * 512
        || bias.len() != 192
        || hidden_out.len() != 192
    {
        return false;
    }
    dense(encoder_frame, weights, bias, hidden_out)
}

/// Computes speaker logits from current and cached hidden vectors.
// Preserve the pinned scalar dot-product order; fusion changes result bits.
#[allow(clippy::suboptimal_flops)]
pub fn speaker_logits(
    hidden: &[f32],
    cached_hidden: &[f32],
    weights: &[f32],
    bias: &[f32],
    logits_out: &mut [f32],
) -> bool {
    if hidden.len() != 192
        || cached_hidden.len() != 192
        || weights.len() != 4 * 384
        || bias.len() != 4
        || logits_out.len() != 4
    {
        return false;
    }
    for speaker in 0..4 {
        let row = &weights[speaker * 384..(speaker + 1) * 384];
        let mut value = bias[speaker];
        for index in 0..192 {
            value += row[index] * hidden[index];
            value += row[192 + index] * cached_hidden[index];
        }
        logits_out[speaker] = value;
    }
    true
}

/// Applies the pinned Sortformer layer normalization equation to one 192-wide row.
// Preserve the pinned unfused scalar normalization order; fusion changes result bits.
#[allow(clippy::suboptimal_flops)]
pub fn layer_norm_192(input: &[f32], scale: &[f32], bias: &[f32], output: &mut [f32]) -> bool {
    if input.len() != 192 || scale.len() != 192 || bias.len() != 192 || output.len() != 192 {
        return false;
    }
    let mean = input.iter().copied().sum::<f32>() / 192.0;
    let variance = input
        .iter()
        .map(|value| {
            let centered = *value - mean;
            centered * centered
        })
        .sum::<f32>()
        / 192.0;
    let inverse_std = (variance + 1.0e-5).sqrt().recip();
    for index in 0..192 {
        let normalized = (input[index] - mean) * inverse_std;
        let scaled = normalized * scale[index];
        output[index] = scaled + bias[index];
    }
    true
}

/// Computes one 24-wide attention head for up to 188 frames.
#[allow(clippy::too_many_arguments)]
// Preserve the pinned scalar attention accumulation order; fusion changes bits.
#[allow(clippy::suboptimal_flops)]
pub fn attention_head_24(
    query: &[f32],
    key: &[f32],
    value: &[f32],
    frame_count: usize,
    query_frame: usize,
    head_offset: usize,
    scores: &mut [f32],
    attended: &mut [f32],
) -> bool {
    if frame_count == 0
        || frame_count > 188
        || query_frame >= frame_count
        || head_offset.checked_add(24).is_none_or(|end| end > 192)
        || query.len() != frame_count * 192
        || key.len() != frame_count * 192
        || value.len() != frame_count * 192
        || scores.len() < frame_count
        || attended.len() < head_offset + 24
    {
        return false;
    }
    let query_base = query_frame * 192 + head_offset;
    let mut maximum = f32::NEG_INFINITY;
    for (frame, score_slot) in scores.iter_mut().enumerate().take(frame_count) {
        let key_base = frame * 192 + head_offset;
        let mut score = 0.0;
        for column in 0..24 {
            score += query[query_base + column] * key[key_base + column];
        }
        score /= 24.0_f32.sqrt();
        *score_slot = score;
        maximum = maximum.max(score);
    }
    let mut normalizer = 0.0;
    for score in &mut scores[..frame_count] {
        *score = (*score - maximum).exp();
        normalizer += *score;
    }
    if !normalizer.is_finite() || normalizer == 0.0 {
        return false;
    }
    for score in &mut scores[..frame_count] {
        *score /= normalizer;
    }
    attended[head_offset..head_offset + 24].fill(0.0);
    for (frame, score) in scores.iter().copied().enumerate().take(frame_count) {
        let value_base = frame * 192 + head_offset;
        for column in 0..24 {
            attended[head_offset + column] += score * value[value_base + column];
        }
    }
    true
}

/// Computes all eight pinned attention heads for one query frame.
pub fn multi_head_attention_192(
    query: &[f32],
    key: &[f32],
    value: &[f32],
    frame_count: usize,
    query_frame: usize,
    scores: &mut [f32],
    attended: &mut [f32],
) -> bool {
    if attended.len() < 192 || scores.len() < frame_count {
        return false;
    }
    attended[..192].fill(0.0);
    for head in 0..8 {
        if !attention_head_24(
            query,
            key,
            value,
            frame_count,
            query_frame,
            head * 24,
            scores,
            attended,
        ) {
            return false;
        }
    }
    true
}

/// Computes the pinned transformer feed-forward phase with a residual add.
#[allow(clippy::too_many_arguments)]
// Preserve the pinned scalar feed-forward accumulation order; fusion changes bits.
#[allow(clippy::suboptimal_flops)]
pub fn feed_forward_residual_192_768(
    input: &[f32],
    input_weight: &[f32],
    input_bias: &[f32],
    output_weight: &[f32],
    output_bias: &[f32],
    residual_scale: f32,
    transposed_input: &mut [f32],
    transposed_output: &mut [f32],
    hidden: &mut [f32],
    output: &mut [f32],
) -> bool {
    if input.len() != 192
        || input_weight.len() != 768 * 192
        || input_bias.len() != 768
        || output_weight.len() != 192 * 768
        || output_bias.len() != 192
        || transposed_input.len() < 192
        || transposed_output.len() < 768
        || hidden.len() != 768
        || output.len() != 192
    {
        return false;
    }
    for index in 0..768 {
        let row = &input_weight[index * 192..(index + 1) * 192];
        let mut value = input_bias[index];
        for column in 0..192 {
            value += row[column] * input[column];
        }
        hidden[index] = value.max(0.0);
    }
    for index in 0..192 {
        let row = &output_weight[index * 768..(index + 1) * 768];
        let mut value = output_bias[index];
        for column in 0..768 {
            value += row[column] * hidden[column];
        }
        output[index] = value.mul_add(residual_scale, input[index]);
    }
    transposed_input[..192].copy_from_slice(input);
    transposed_output[..768].copy_from_slice(hidden);
    true
}

/// Runs one bounded transformer frame through attention, residual normalization,
/// feed-forward residual, and final normalization.
#[allow(clippy::too_many_arguments)]
pub fn transformer_frame_192(
    input_frames: &[f32],
    query: &[f32],
    key: &[f32],
    value: &[f32],
    output_weight: &[f32],
    output_bias: &[f32],
    norm1_scale: &[f32],
    norm1_bias: &[f32],
    ff_input_weight: &[f32],
    ff_input_bias: &[f32],
    ff_output_weight: &[f32],
    ff_output_bias: &[f32],
    norm2_scale: &[f32],
    norm2_bias: &[f32],
    frame_count: usize,
    frame_index: usize,
    scores: &mut [f32],
    attended: &mut [f32],
    residual: &mut [f32],
    normalized: &mut [f32],
    feed_hidden: &mut [f32],
    feed_output: &mut [f32],
    ff_transposed_input: &mut [f32],
    ff_transposed_output: &mut [f32],
    output: &mut [f32],
) -> bool {
    if input_frames.len() != frame_count * 192
        || frame_index >= frame_count
        || residual.len() != 192
        || normalized.len() != 192
        || feed_hidden.len() != 768
        || feed_output.len() != 192
        || ff_transposed_input.len() < 192
        || ff_transposed_output.len() < 768
        || output.len() != 192
    {
        return false;
    }
    if !multi_head_attention_192(
        query,
        key,
        value,
        frame_count,
        frame_index,
        scores,
        attended,
    ) {
        return false;
    }
    let frame = &input_frames[frame_index * 192..(frame_index + 1) * 192];
    if !dense(attended, output_weight, output_bias, residual) {
        return false;
    }
    for index in 0..192 {
        residual[index] += frame[index];
    }
    if !layer_norm_192(residual, norm1_scale, norm1_bias, normalized) {
        return false;
    }
    if !feed_forward_residual_192_768(
        normalized,
        ff_input_weight,
        ff_input_bias,
        ff_output_weight,
        ff_output_bias,
        1.0,
        ff_transposed_input,
        ff_transposed_output,
        feed_hidden,
        feed_output,
    ) {
        return false;
    }
    layer_norm_192(feed_output, norm2_scale, norm2_bias, output)
}

/// Runs a bounded frame batch through one transformer layer using reusable scratch.
#[allow(clippy::too_many_arguments)]
pub fn transformer_batch_192(
    input_frames: &[f32],
    query: &[f32],
    key: &[f32],
    value: &[f32],
    output_weight: &[f32],
    output_bias: &[f32],
    norm1_scale: &[f32],
    norm1_bias: &[f32],
    ff_input_weight: &[f32],
    ff_input_bias: &[f32],
    ff_output_weight: &[f32],
    ff_output_bias: &[f32],
    norm2_scale: &[f32],
    norm2_bias: &[f32],
    frame_count: usize,
    scores: &mut [f32],
    attended: &mut [f32],
    residual: &mut [f32],
    normalized: &mut [f32],
    feed_hidden: &mut [f32],
    feed_output: &mut [f32],
    ff_transposed_input: &mut [f32],
    ff_transposed_output: &mut [f32],
    output_frames: &mut [f32],
) -> bool {
    if frame_count == 0
        || frame_count > 188
        || input_frames.len() != frame_count * 192
        || output_frames.len() != input_frames.len()
        || scores.len() < frame_count
    {
        return false;
    }
    for frame_index in 0..frame_count {
        if !transformer_frame_192(
            input_frames,
            query,
            key,
            value,
            output_weight,
            output_bias,
            norm1_scale,
            norm1_bias,
            ff_input_weight,
            ff_input_bias,
            ff_output_weight,
            ff_output_bias,
            norm2_scale,
            norm2_bias,
            frame_count,
            frame_index,
            scores,
            attended,
            residual,
            normalized,
            feed_hidden,
            feed_output,
            ff_transposed_input,
            ff_transposed_output,
            &mut output_frames[frame_index * 192..(frame_index + 1) * 192],
        ) {
            return false;
        }
    }
    true
}

impl DenseWeightCache {
    /// Creates an empty cache before model initialization.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            source: 0,
            input_dim: 0,
            output_dim: 0,
        }
    }

    /// Validates and records a matrix for reusable dispatch.
    pub fn prepare(&mut self, weights: &[f32], input_dim: usize, output_dim: usize) -> bool {
        let Some(expected) = input_dim.checked_mul(output_dim) else {
            return false;
        };
        if input_dim == 0 || output_dim == 0 || weights.len() != expected {
            return false;
        }
        let source = weights.as_ptr() as usize;
        if self.source == source && self.input_dim == input_dim && self.output_dim == output_dim {
            return true;
        }
        self.source = source;
        self.input_dim = input_dim;
        self.output_dim = output_dim;
        true
    }

    /// Reports whether this cache describes the supplied matrix.
    #[must_use]
    pub fn is_prepared(self, weights: &[f32], input_dim: usize, output_dim: usize) -> bool {
        self.source == weights.as_ptr() as usize
            && self.input_dim == input_dim
            && self.output_dim == output_dim
    }
}

/// Fixed caller-owned scratch required by one Sortformer transformer layer.
#[derive(Debug)]
pub struct TransformerLayerWorkspace {
    pub query: Box<[f32]>,
    pub key: Box<[f32]>,
    pub value: Box<[f32]>,
    pub first_norm: Box<[f32]>,
    pub feed_forward_rows: Box<[f32]>,
    pub dense_transposed_input: Box<[f32]>,
    pub dense_transposed_output: Box<[f32]>,
    pub feed_forward: Box<[f32]>,
    pub scores: Box<[f32]>,
    pub attended: Box<[f32]>,
    pub residual: Box<[f32]>,
    pub normalized: Box<[f32]>,
    pub output: Box<[f32]>,
}

impl TransformerLayerWorkspace {
    /// Allocates the complete bounded workspace before hot-path dispatch.
    #[must_use]
    pub fn new() -> Self {
        Self {
            query: vec![0.0; 188 * 192].into_boxed_slice(),
            key: vec![0.0; 188 * 192].into_boxed_slice(),
            value: vec![0.0; 188 * 192].into_boxed_slice(),
            first_norm: vec![0.0; 188 * 192].into_boxed_slice(),
            feed_forward_rows: vec![0.0; 188 * 768].into_boxed_slice(),
            dense_transposed_input: vec![0.0; 188 * 768].into_boxed_slice(),
            dense_transposed_output: vec![0.0; 188 * 768].into_boxed_slice(),
            feed_forward: vec![0.0; 768].into_boxed_slice(),
            scores: vec![0.0; 188].into_boxed_slice(),
            attended: vec![0.0; 192].into_boxed_slice(),
            residual: vec![0.0; 192].into_boxed_slice(),
            normalized: vec![0.0; 192].into_boxed_slice(),
            output: vec![0.0; 192].into_boxed_slice(),
        }
    }

    /// Clears reusable scratch without allocating.
    pub fn reset(&mut self) {
        self.query.fill(0.0);
        self.key.fill(0.0);
        self.value.fill(0.0);
        self.first_norm.fill(0.0);
        self.feed_forward_rows.fill(0.0);
        self.dense_transposed_input.fill(0.0);
        self.dense_transposed_output.fill(0.0);
        self.feed_forward.fill(0.0);
        self.scores.fill(0.0);
        self.attended.fill(0.0);
        self.residual.fill(0.0);
        self.normalized.fill(0.0);
        self.output.fill(0.0);
    }
}

impl Default for TransformerLayerWorkspace {
    fn default() -> Self {
        Self::new()
    }
}

/// Computes the source-compatible 64-element dot product.
#[must_use]
pub fn dot_64(lhs: &[f32], rhs: &[f32]) -> Option<f32> {
    dot_fixed(lhs, rhs, 64)
}

/// Computes the source-compatible 24-element dot product.
#[must_use]
pub fn dot_24(lhs: &[f32], rhs: &[f32]) -> Option<f32> {
    dot_fixed(lhs, rhs, 24)
}

/// Computes a weighted sum over strided value rows.
pub fn weighted_sum_64(
    weights: &[f32],
    values: &[f32],
    value_stride: usize,
    value_count: usize,
    output: &mut [f32],
) -> bool {
    weighted_sum(weights, values, value_stride, value_count, 64, output)
}

/// Computes a weighted sum over strided value rows.
pub fn weighted_sum_24(
    weights: &[f32],
    values: &[f32],
    value_stride: usize,
    value_count: usize,
    output: &mut [f32],
) -> bool {
    weighted_sum(weights, values, value_stride, value_count, 24, output)
}

/// Transposes contiguous row-major input into contiguous column-major rows.
pub fn transpose_dense_input(
    input_rows: &[f32],
    row_count: usize,
    input_dim: usize,
    transposed: &mut [f32],
) -> bool {
    let Some(input_len) = row_count.checked_mul(input_dim) else {
        return false;
    };
    if row_count == 0
        || input_dim == 0
        || input_rows.len() != input_len
        || transposed.len() < input_len
    {
        return false;
    }
    for input_index in 0..input_dim {
        for row in 0..row_count {
            transposed[input_index * row_count + row] = input_rows[row * input_dim + input_index];
        }
    }
    true
}

/// Computes one dense row and adds a bias vector.
// Preserve the pinned scalar weighted-sum order; fusion changes result bits.
#[allow(clippy::suboptimal_flops)]
pub fn dense(input: &[f32], weights: &[f32], bias: &[f32], output: &mut [f32]) -> bool {
    if input.is_empty() || output.is_empty() || bias.len() != output.len() {
        return false;
    }
    let Some(weight_len) = input.len().checked_mul(output.len()) else {
        return false;
    };
    if weights.len() != weight_len {
        return false;
    }
    for (output_index, destination) in output.iter_mut().enumerate() {
        let row = &weights[output_index * input.len()..(output_index + 1) * input.len()];
        let mut acc = bias[output_index];
        for (weight, value) in row.iter().zip(input) {
            acc += *weight * *value;
        }
        *destination = acc;
    }
    true
}

/// Computes one dense row without a bias vector.
pub fn dense_without_bias(input: &[f32], weights: &[f32], output: &mut [f32]) -> bool {
    if input.is_empty() || output.is_empty() {
        return false;
    }
    let Some(weight_len) = input.len().checked_mul(output.len()) else {
        return false;
    };
    if weights.len() != weight_len {
        return false;
    }
    for (output_index, destination) in output.iter_mut().enumerate() {
        let row = &weights[output_index * input.len()..(output_index + 1) * input.len()];
        *destination = row
            .iter()
            .zip(input)
            .map(|(weight, value)| weight * value)
            .sum();
    }
    true
}

/// Caller-owned buffers and dimensions for one dense batch operation.
#[derive(Debug)]
pub struct DenseBatch<'a> {
    pub input_rows: &'a [f32],
    pub row_count: usize,
    pub input_dim: usize,
    pub weights: &'a [f32],
    pub bias: &'a [f32],
    pub output_dim: usize,
    pub transposed_input: &'a mut [f32],
    pub transposed_output: &'a mut [f32],
    pub output_rows: &'a mut [f32],
}

/// Computes a batch of dense rows using caller-owned transposition scratch.
// Preserve the pinned scalar weighted-sum order; fusion changes result bits.
#[allow(clippy::suboptimal_flops)]
pub fn dense_batch(mut batch: DenseBatch<'_>) -> bool {
    let input_rows = batch.input_rows;
    let row_count = batch.row_count;
    let input_dim = batch.input_dim;
    let weights = batch.weights;
    let bias = batch.bias;
    let output_dim = batch.output_dim;
    let transposed_input = &mut batch.transposed_input;
    let transposed_output = &mut batch.transposed_output;
    let output_rows = &mut batch.output_rows;
    let (
        Some(input_len),
        Some(weight_len),
        Some(transposed_input_len),
        Some(transposed_output_len),
        Some(output_len),
    ) = (
        row_count.checked_mul(input_dim),
        input_dim.checked_mul(output_dim),
        input_dim.checked_mul(row_count),
        output_dim.checked_mul(row_count),
        row_count.checked_mul(output_dim),
    )
    else {
        return false;
    };
    if row_count == 0
        || input_dim == 0
        || output_dim == 0
        || input_rows.len() != input_len
        || weights.len() != weight_len
        || bias.len() != output_dim
        || transposed_input.len() < transposed_input_len
        || transposed_output.len() < transposed_output_len
        || output_rows.len() != output_len
    {
        return false;
    }
    if !transpose_dense_input(input_rows, row_count, input_dim, transposed_input) {
        return false;
    }
    for output_index in 0..output_dim {
        let row = &weights[output_index * input_dim..(output_index + 1) * input_dim];
        for batch_row in 0..row_count {
            let mut acc = bias[output_index];
            for input_index in 0..input_dim {
                acc += row[input_index] * transposed_input[input_index * row_count + batch_row];
            }
            transposed_output[output_index * row_count + batch_row] = acc;
        }
    }
    for row in 0..row_count {
        for output_index in 0..output_dim {
            output_rows[row * output_dim + output_index] =
                transposed_output[output_index * row_count + row];
        }
    }
    true
}

/// Computes a batch of dense rows without adding a bias vector.
pub fn dense_batch_without_bias(mut batch: DenseBatch<'_>) -> bool {
    let input_rows = batch.input_rows;
    let row_count = batch.row_count;
    let input_dim = batch.input_dim;
    let weights = batch.weights;
    let output_dim = batch.output_dim;
    let transposed_input = &mut batch.transposed_input;
    let transposed_output = &mut batch.transposed_output;
    let output_rows = &mut batch.output_rows;
    let (
        Some(input_len),
        Some(weight_len),
        Some(transposed_input_len),
        Some(transposed_output_len),
        Some(output_len),
    ) = (
        row_count.checked_mul(input_dim),
        input_dim.checked_mul(output_dim),
        input_dim.checked_mul(row_count),
        output_dim.checked_mul(row_count),
        row_count.checked_mul(output_dim),
    )
    else {
        return false;
    };
    if row_count == 0
        || input_dim == 0
        || output_dim == 0
        || input_rows.len() != input_len
        || weights.len() != weight_len
        || transposed_input.len() < transposed_input_len
        || transposed_output.len() < transposed_output_len
        || output_rows.len() != output_len
    {
        return false;
    }
    if !transpose_dense_input(input_rows, row_count, input_dim, transposed_input) {
        return false;
    }
    for output_index in 0..output_dim {
        let row = &weights[output_index * input_dim..(output_index + 1) * input_dim];
        for batch_row in 0..row_count {
            transposed_output[output_index * row_count + batch_row] = row
                .iter()
                .enumerate()
                .map(|(input_index, weight)| {
                    weight * transposed_input[input_index * row_count + batch_row]
                })
                .sum();
        }
    }
    for row in 0..row_count {
        for output_index in 0..output_dim {
            output_rows[row * output_dim + output_index] =
                transposed_output[output_index * row_count + row];
        }
    }
    true
}

/// Computes a prepared no-bias dense batch from row-major input.
pub fn dense_batch_without_bias_prepared(
    batch: &mut DenseBatch<'_>,
    cache: DenseWeightCache,
) -> bool {
    if !cache.is_prepared(batch.weights, batch.input_dim, batch.output_dim) {
        return false;
    }
    dense_batch_without_bias(DenseBatch {
        input_rows: batch.input_rows,
        row_count: batch.row_count,
        input_dim: batch.input_dim,
        weights: batch.weights,
        bias: &[],
        output_dim: batch.output_dim,
        transposed_input: batch.transposed_input,
        transposed_output: batch.transposed_output,
        output_rows: batch.output_rows,
    })
}

/// Caller-owned request for dense computation from transposed input.
#[derive(Debug)]
pub struct DenseBatchFromTransposed<'a> {
    pub transposed_input: &'a [f32],
    pub row_count: usize,
    pub input_dim: usize,
    pub weights: &'a [f32],
    pub bias: &'a [f32],
    pub output_dim: usize,
    pub transposed_output: &'a mut [f32],
    pub output_rows: &'a mut [f32],
}

/// Computes a dense batch from already-transposed input.
pub fn dense_batch_from_transposed(batch: &mut DenseBatchFromTransposed<'_>) -> bool {
    let Some(input_len) = batch.input_dim.checked_mul(batch.row_count) else {
        return false;
    };
    let Some(weight_len) = batch.input_dim.checked_mul(batch.output_dim) else {
        return false;
    };
    let Some(transposed_output_len) = batch.output_dim.checked_mul(batch.row_count) else {
        return false;
    };
    let Some(output_len) = batch.row_count.checked_mul(batch.output_dim) else {
        return false;
    };
    if batch.row_count == 0
        || batch.input_dim == 0
        || batch.output_dim == 0
        || batch.transposed_input.len() < input_len
        || batch.weights.len() != weight_len
        || batch.bias.len() != batch.output_dim
        || batch.transposed_output.len() < transposed_output_len
        || batch.output_rows.len() != output_len
    {
        return false;
    }
    for output_index in 0..batch.output_dim {
        let row =
            &batch.weights[output_index * batch.input_dim..(output_index + 1) * batch.input_dim];
        for batch_row in 0..batch.row_count {
            batch.transposed_output[output_index * batch.row_count + batch_row] = row
                .iter()
                .enumerate()
                .map(|(input_index, weight)| {
                    weight * batch.transposed_input[input_index * batch.row_count + batch_row]
                })
                .sum();
        }
    }
    for row in 0..batch.row_count {
        for output_index in 0..batch.output_dim {
            batch.output_rows[row * batch.output_dim + output_index] = batch.transposed_output
                [output_index * batch.row_count + row]
                + batch.bias[output_index];
        }
    }
    true
}

/// Computes a dense batch from transposed input using a prepared cache.
pub fn dense_batch_from_transposed_prepared(
    batch: &mut DenseBatchFromTransposed<'_>,
    cache: DenseWeightCache,
) -> bool {
    if !cache.is_prepared(batch.weights, batch.input_dim, batch.output_dim) {
        return false;
    }
    dense_batch_from_transposed(batch)
}

/// Caller-owned request for a dense batch with a scaled residual connection.
#[derive(Debug)]
pub struct DenseBatchResidual<'a> {
    pub transposed_input: &'a [f32],
    pub row_count: usize,
    pub input_dim: usize,
    pub weights: &'a [f32],
    pub cache: DenseWeightCache,
    pub bias: &'a [f32],
    pub output_dim: usize,
    pub dense_scale: f32,
    pub residual_rows: &'a [f32],
    pub transposed_output: &'a mut [f32],
    pub output_rows: &'a mut [f32],
}

/// Computes a prepared dense batch and adds a scaled residual row-wise.
pub fn dense_batch_scaled_residual(batch: &mut DenseBatchResidual<'_>) -> bool {
    let Some(transposed_input_len) = batch.input_dim.checked_mul(batch.row_count) else {
        return false;
    };
    let Some(weight_len) = batch.input_dim.checked_mul(batch.output_dim) else {
        return false;
    };
    let Some(output_len) = batch.row_count.checked_mul(batch.output_dim) else {
        return false;
    };
    let Some(transposed_output_len) = batch.output_dim.checked_mul(batch.row_count) else {
        return false;
    };
    if batch.row_count == 0
        || batch.input_dim == 0
        || batch.output_dim == 0
        || batch.transposed_input.len() < transposed_input_len
        || batch.weights.len() != weight_len
        || !batch
            .cache
            .is_prepared(batch.weights, batch.input_dim, batch.output_dim)
        || batch.bias.len() != batch.output_dim
        || batch.residual_rows.len() != output_len
        || batch.transposed_output.len() < transposed_output_len
        || batch.output_rows.len() != output_len
    {
        return false;
    }
    for output_index in 0..batch.output_dim {
        let row =
            &batch.weights[output_index * batch.input_dim..(output_index + 1) * batch.input_dim];
        for batch_row in 0..batch.row_count {
            batch.transposed_output[output_index * batch.row_count + batch_row] = row
                .iter()
                .enumerate()
                .map(|(input_index, weight)| {
                    weight * batch.transposed_input[input_index * batch.row_count + batch_row]
                })
                .sum();
        }
    }
    for row in 0..batch.row_count {
        for output_index in 0..batch.output_dim {
            let offset = row * batch.output_dim + output_index;
            let dense_value = batch.transposed_output[output_index * batch.row_count + row]
                + batch.bias[output_index];
            batch.output_rows[offset] =
                dense_value.mul_add(batch.dense_scale, batch.residual_rows[offset]);
        }
    }
    true
}

/// Computes a prepared dense batch from transposed input and adds a residual.
pub fn dense_batch_from_transposed_scaled_residual(batch: &mut DenseBatchResidual<'_>) -> bool {
    if !batch
        .cache
        .is_prepared(batch.weights, batch.input_dim, batch.output_dim)
    {
        return false;
    }
    let Some(input_len) = batch.input_dim.checked_mul(batch.row_count) else {
        return false;
    };
    let Some(weight_len) = batch.input_dim.checked_mul(batch.output_dim) else {
        return false;
    };
    let Some(output_len) = batch.row_count.checked_mul(batch.output_dim) else {
        return false;
    };
    let Some(transposed_output_len) = batch.output_dim.checked_mul(batch.row_count) else {
        return false;
    };
    if batch.row_count == 0
        || batch.input_dim == 0
        || batch.output_dim == 0
        || batch.transposed_input.len() < input_len
        || batch.weights.len() != weight_len
        || batch.bias.len() != batch.output_dim
        || batch.residual_rows.len() != output_len
        || batch.transposed_output.len() < transposed_output_len
        || batch.output_rows.len() != output_len
    {
        return false;
    }
    for output_index in 0..batch.output_dim {
        let row =
            &batch.weights[output_index * batch.input_dim..(output_index + 1) * batch.input_dim];
        for batch_row in 0..batch.row_count {
            batch.transposed_output[output_index * batch.row_count + batch_row] = row
                .iter()
                .enumerate()
                .map(|(input_index, weight)| {
                    weight * batch.transposed_input[input_index * batch.row_count + batch_row]
                })
                .sum();
        }
    }
    for row in 0..batch.row_count {
        for output_index in 0..batch.output_dim {
            let offset = row * batch.output_dim + output_index;
            let dense_value = batch.transposed_output[output_index * batch.row_count + row]
                + batch.bias[output_index];
            batch.output_rows[offset] =
                dense_value.mul_add(batch.dense_scale, batch.residual_rows[offset]);
        }
    }
    true
}

/// Computes a dense batch and leaves the biased result in transposed output.
pub fn dense_batch_to_transposed(batch: &mut DenseBatch<'_>) -> bool {
    let Some(input_len) = batch.input_dim.checked_mul(batch.row_count) else {
        return false;
    };
    let Some(weight_len) = batch.input_dim.checked_mul(batch.output_dim) else {
        return false;
    };
    let Some(output_len) = batch.output_dim.checked_mul(batch.row_count) else {
        return false;
    };
    if batch.row_count == 0
        || batch.input_dim == 0
        || batch.output_dim == 0
        || batch.input_rows.len() != input_len
        || batch.weights.len() != weight_len
        || batch.bias.len() != batch.output_dim
        || batch.transposed_input.len() < input_len
        || batch.transposed_output.len() < output_len
    {
        return false;
    }
    if !transpose_dense_input(
        batch.input_rows,
        batch.row_count,
        batch.input_dim,
        batch.transposed_input,
    ) {
        return false;
    }
    for output_index in 0..batch.output_dim {
        let row =
            &batch.weights[output_index * batch.input_dim..(output_index + 1) * batch.input_dim];
        for batch_row in 0..batch.row_count {
            batch.transposed_output[output_index * batch.row_count + batch_row] = row
                .iter()
                .enumerate()
                .map(|(input_index, weight)| {
                    weight * batch.transposed_input[input_index * batch.row_count + batch_row]
                })
                .sum::<f32>()
                + batch.bias[output_index];
        }
    }
    true
}

/// Computes a prepared dense batch and leaves its biased result transposed.
pub fn dense_batch_to_transposed_prepared(
    batch: &mut DenseBatch<'_>,
    cache: DenseWeightCache,
) -> bool {
    if !cache.is_prepared(batch.weights, batch.input_dim, batch.output_dim) {
        return false;
    }
    dense_batch_to_transposed(batch)
}

fn dot_fixed(lhs: &[f32], rhs: &[f32], width: usize) -> Option<f32> {
    let lhs = lhs.get(..width)?;
    let rhs = rhs.get(..width)?;
    Some(lhs.iter().zip(rhs).map(|(left, right)| left * right).sum())
}

// Preserve the pinned scalar weighted-sum order; fusion changes result bits.
#[allow(clippy::suboptimal_flops)]
fn weighted_sum(
    weights: &[f32],
    values: &[f32],
    value_stride: usize,
    value_count: usize,
    width: usize,
    output: &mut [f32],
) -> bool {
    if weights.len() < value_count
        || output.len() < width
        || value_stride < width
        || value_count
            .checked_sub(1)
            .and_then(|last| last.checked_mul(value_stride))
            .and_then(|offset| offset.checked_add(width))
            .is_none_or(|end| end > values.len())
    {
        return false;
    }
    output[..width].fill(0.0);
    for (index, weight) in weights.iter().copied().take(value_count).enumerate() {
        let start = index * value_stride;
        for lane in 0..width {
            output[lane] += weight * values[start + lane];
        }
    }
    true
}

#[cfg(test)]
mod tests {
    #![allow(clippy::cast_precision_loss)]
    #![allow(clippy::float_cmp)]
    #![allow(clippy::large_stack_arrays)]
    #![allow(clippy::large_stack_frames)]
    use super::*;

    #[test]
    fn fixed_width_dots_match_scalar_reference() {
        let lhs: Vec<_> = (0..64).map(|value| value as f32).collect();
        let rhs: Vec<_> = (0..64).map(|value| (64 - value) as f32).collect();
        let expected: f32 = lhs.iter().zip(&rhs).map(|(a, b)| a * b).sum();
        assert_eq!(dot_64(&lhs, &rhs), Some(expected));
        assert_eq!(
            dot_24(&lhs, &rhs),
            Some(lhs[..24].iter().zip(&rhs[..24]).map(|(a, b)| a * b).sum())
        );
        assert_eq!(dot_64(&lhs[..63], &rhs), None);
    }

    #[test]
    fn weighted_sum_handles_stride_and_rejects_bad_shapes() {
        let mut values = [0.0; 48];
        values[..24].fill(1.0);
        values[24..].fill(2.0);
        let mut output = [0.0; 24];
        assert!(weighted_sum_24(&[2.0, 3.0], &values, 24, 2, &mut output));
        assert_eq!(&output[..2], &[8.0, 8.0]);
        assert!(!weighted_sum_24(&[1.0], &values, 1, 1, &mut output));
    }

    #[test]
    fn dense_cache_reuses_exact_matrix_identity_without_allocating() {
        let weights = [1.0_f32; 6];
        let mut cache = DenseWeightCache::new();
        assert!(cache.prepare(&weights, 2, 3));
        assert!(cache.is_prepared(&weights, 2, 3));
        assert!(cache.prepare(&weights, 2, 3));
        assert!(!cache.prepare(&weights, 3, 3));
        assert!(cache.is_prepared(&weights, 2, 3));
    }

    #[test]
    fn fixed_sortformer_module_kernels_match_reference_shapes() {
        let frame = [1.0_f32; 512];
        let weights = [0.0_f32; 192 * 512];
        let bias = [2.0_f32; 192];
        let mut hidden = [0.0_f32; 192];
        assert!(encoder_projection(&frame, &weights, &bias, &mut hidden));
        assert!(hidden.iter().all(|value| *value == 2.0));

        let cached = [1.0_f32; 192];
        let pair_weights = [0.0_f32; 4 * 384];
        let pair_bias = [3.0_f32; 4];
        let mut logits = [0.0_f32; 4];
        assert!(speaker_logits(
            &hidden,
            &cached,
            &pair_weights,
            &pair_bias,
            &mut logits
        ));
        assert_eq!(logits, pair_bias);
        assert!(!encoder_projection(
            &frame[..511],
            &weights,
            &bias,
            &mut hidden
        ));
        assert!(!speaker_logits(
            &hidden,
            &cached,
            &pair_weights[..1535],
            &pair_bias,
            &mut logits
        ));
    }

    #[test]
    fn layer_norm_uses_pinned_epsilon_and_rejects_wrong_shapes() {
        let input = [2.0_f32; 192];
        let scale = [3.0_f32; 192];
        let bias = [4.0_f32; 192];
        let mut output = [0.0_f32; 192];
        assert!(layer_norm_192(&input, &scale, &bias, &mut output));
        assert!(
            output
                .iter()
                .all(|value| (*value - 4.0).abs() < f32::EPSILON)
        );
        assert!(!layer_norm_192(&input[..191], &scale, &bias, &mut output));
    }

    #[test]
    fn layer_norm_matches_unfused_reference() {
        let mut input = [0.0_f32; 192];
        for (index, value) in input.iter_mut().enumerate() {
            *value = if index % 2 == 0 { 1.0 } else { -1.0 };
        }
        let scale = [1_234_567.0_f32; 192];
        let bias = [0.1_f32; 192];
        let mut output = [0.0_f32; 192];
        let mut expected = [0.0_f32; 192];

        assert!(layer_norm_192(&input, &scale, &bias, &mut output));

        let mean = input.iter().copied().sum::<f32>() / 192.0;
        let variance = input
            .iter()
            .map(|value| {
                let centered = *value - mean;
                centered * centered
            })
            .sum::<f32>()
            / 192.0;
        let inverse_std = (variance + 1.0e-5).sqrt().recip();
        for index in 0..192 {
            let normalized = (input[index] - mean) * inverse_std;
            expected[index] = normalized * scale[index] + bias[index];
        }

        for (index, (actual, reference)) in output.iter().zip(expected.iter()).enumerate() {
            assert_eq!(
                actual.to_bits(),
                reference.to_bits(),
                "layer norm output mismatch at index {index}"
            );
        }
    }

    #[test]
    fn attention_head_matches_scaled_softmax_weighted_sum() {
        let query = [1.0_f32; 2 * 192];
        let key = [1.0_f32; 2 * 192];
        let value = [2.0_f32; 2 * 192];
        let mut scores = [0.0_f32; 188];
        let mut attended = [0.0_f32; 192];
        assert!(attention_head_24(
            &query,
            &key,
            &value,
            2,
            0,
            0,
            &mut scores,
            &mut attended
        ));
        assert!((scores[0] - 0.5).abs() < 1e-6);
        assert!(
            attended[..24]
                .iter()
                .all(|value| (*value - 2.0).abs() < 1e-6)
        );
        assert!(!attention_head_24(
            &query[..191],
            &key,
            &value,
            2,
            0,
            0,
            &mut scores,
            &mut attended
        ));
        assert!(multi_head_attention_192(
            &query,
            &key,
            &value,
            2,
            0,
            &mut scores,
            &mut attended
        ));
        assert!(attended.iter().all(|value| (*value - 2.0).abs() < 1e-6));
    }

    #[test]
    fn feed_forward_residual_preserves_relu_and_residual_equation() {
        let input = [1.0_f32; 192];
        let input_weight = [0.0_f32; 768 * 192];
        let input_bias = [2.0_f32; 768];
        let output_weight = [0.0_f32; 192 * 768];
        let output_bias = [3.0_f32; 192];
        let mut transposed_input = [0.0_f32; 192];
        let mut transposed_output = [0.0_f32; 768];
        let mut hidden = [0.0_f32; 768];
        let mut output = [0.0_f32; 192];
        assert!(feed_forward_residual_192_768(
            &input,
            &input_weight,
            &input_bias,
            &output_weight,
            &output_bias,
            1.0,
            &mut transposed_input,
            &mut transposed_output,
            &mut hidden,
            &mut output,
        ));
        assert!(hidden.iter().all(|value| (*value - 2.0).abs() < 1e-6));
        assert!(output.iter().all(|value| (*value - 4.0).abs() < 1e-6));
        assert!(!feed_forward_residual_192_768(
            &input[..191],
            &input_weight,
            &input_bias,
            &output_weight,
            &output_bias,
            1.0,
            &mut transposed_input,
            &mut transposed_output,
            &mut hidden,
            &mut output,
        ));
    }

    #[test]
    fn transformer_frame_composes_pinned_phase_order() {
        let frames = [1.0_f32; 192];
        let qkv = [0.0_f32; 192];
        let output_weight = [0.0_f32; 192 * 192];
        let output_bias = [0.0_f32; 192];
        let norm_scale = [1.0_f32; 192];
        let norm_bias = [0.0_f32; 192];
        let ff_in_weight = [0.0_f32; 768 * 192];
        let ff_in_bias = [0.0_f32; 768];
        let ff_out_weight = [0.0_f32; 192 * 768];
        let ff_out_bias = [0.0_f32; 192];
        let mut scores = [0.0_f32; 188];
        let mut attended = [0.0_f32; 192];
        let mut residual = [0.0_f32; 192];
        let mut normalized = [0.0_f32; 192];
        let mut feed_hidden = [0.0_f32; 768];
        let mut feed_output = [0.0_f32; 192];
        let mut output = [0.0_f32; 192];
        assert!(transformer_frame_192(
            &frames,
            &qkv,
            &qkv,
            &qkv,
            &output_weight,
            &output_bias,
            &norm_scale,
            &norm_bias,
            &ff_in_weight,
            &ff_in_bias,
            &ff_out_weight,
            &ff_out_bias,
            &norm_scale,
            &norm_bias,
            1,
            0,
            &mut scores,
            &mut attended,
            &mut residual,
            &mut normalized,
            &mut feed_hidden,
            &mut feed_output,
            &mut [0.0_f32; 192][..],
            &mut [0.0_f32; 768][..],
            &mut output,
        ));
        assert!(output.iter().all(|value| value.abs() < 1e-6));
    }

    #[test]
    fn transformer_workspace_is_bounded_and_resettable() {
        let mut workspace = Box::new(TransformerLayerWorkspace::new());
        workspace.query[0] = 3.0;
        workspace.feed_forward[0] = 4.0;
        workspace.reset();
        assert_eq!(workspace.query[0], 0.0);
        assert_eq!(workspace.feed_forward[0], 0.0);
    }

    #[test]
    fn transformer_batch_reuses_scratch_for_each_frame() {
        let frames = [1.0_f32; 2 * 192];
        let qkv = [0.0_f32; 2 * 192];
        let output_weight = [0.0_f32; 192 * 192];
        let output_bias = [0.0_f32; 192];
        let norm_scale = [1.0_f32; 192];
        let norm_bias = [0.0_f32; 192];
        let ff_in_weight = [0.0_f32; 768 * 192];
        let ff_in_bias = [0.0_f32; 768];
        let ff_out_weight = [0.0_f32; 192 * 768];
        let ff_out_bias = [0.0_f32; 192];
        let mut scores = [0.0_f32; 188];
        let mut attended = [0.0_f32; 192];
        let mut residual = [0.0_f32; 192];
        let mut normalized = [0.0_f32; 192];
        let mut feed_hidden = [0.0_f32; 768];
        let mut feed_output = [0.0_f32; 192];
        let mut ff_input = [0.0_f32; 192];
        let mut ff_output = [0.0_f32; 768];
        let mut output = [0.0_f32; 2 * 192];
        assert!(transformer_batch_192(
            &frames,
            &qkv,
            &qkv,
            &qkv,
            &output_weight,
            &output_bias,
            &norm_scale,
            &norm_bias,
            &ff_in_weight,
            &ff_in_bias,
            &ff_out_weight,
            &ff_out_bias,
            &norm_scale,
            &norm_bias,
            2,
            &mut scores,
            &mut attended,
            &mut residual,
            &mut normalized,
            &mut feed_hidden,
            &mut feed_output,
            &mut ff_input,
            &mut ff_output,
            &mut output,
        ));
        assert!(output.iter().all(|value| value.abs() < 1e-6));
    }

    #[test]
    fn dense_transpose_matches_pinned_column_major_layout() {
        let input = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut output = [0.0; 6];
        assert!(transpose_dense_input(&input, 2, 3, &mut output));
        assert_eq!(output, [1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
        assert!(!transpose_dense_input(&input, 0, 3, &mut output));
        assert!(!transpose_dense_input(&input[..5], 2, 3, &mut output));
        assert!(!transpose_dense_input(&input, 2, 3, &mut output[..5]));
    }

    #[test]
    fn dense_variants_match_reference_row_major_weights() {
        let input = [2.0, 3.0];
        let weights = [1.0, 4.0, -2.0, 5.0];
        let mut output = [0.0; 2];
        assert!(dense(&input, &weights, &[0.5, 1.0], &mut output));
        assert_eq!(output, [14.5, 12.0]);
        assert!(dense_without_bias(&input, &weights, &mut output));
        assert_eq!(output, [14.0, 11.0]);
        assert!(!dense(&input, &weights[..3], &[0.0, 0.0], &mut output));
        assert!(!dense_without_bias(&input, &weights, &mut output[..1]));
    }

    #[test]
    fn dense_batch_matches_reference_transposed_scratch_contract() {
        let input = [1.0, 2.0, 3.0, 4.0];
        let weights = [1.0, 2.0, -1.0, 3.0];
        let bias = [0.5, -1.0];
        let mut transposed_input = [0.0; 4];
        let mut transposed_output = [0.0; 4];
        let mut output = [0.0; 4];
        assert!(dense_batch(DenseBatch {
            input_rows: &input,
            row_count: 2,
            input_dim: 2,
            weights: &weights,
            bias: &bias,
            output_dim: 2,
            transposed_input: &mut transposed_input,
            transposed_output: &mut transposed_output,
            output_rows: &mut output,
        }));
        assert_eq!(transposed_input, [1.0, 3.0, 2.0, 4.0]);
        assert_eq!(output, [5.5, 4.0, 11.5, 8.0]);
        assert!(!dense_batch(DenseBatch {
            input_rows: &input[..3],
            row_count: 2,
            input_dim: 2,
            weights: &weights,
            bias: &bias,
            output_dim: 2,
            transposed_input: &mut transposed_input,
            transposed_output: &mut transposed_output,
            output_rows: &mut output,
        }));
    }

    #[test]
    fn dense_batch_scaled_residual_matches_reference() {
        let transposed_input = [1.0, 3.0, 2.0, 4.0];
        let weights = [1.0, 2.0, -1.0, 3.0];
        let bias = [0.5, -1.0];
        let residual = [10.0, 20.0, 30.0, 40.0];
        let mut cache = DenseWeightCache::new();
        assert!(cache.prepare(&weights, 2, 2));
        let mut transposed_output = [0.0; 4];
        let mut output = [0.0; 4];
        assert!(dense_batch_scaled_residual(&mut DenseBatchResidual {
            transposed_input: &transposed_input,
            row_count: 2,
            input_dim: 2,
            weights: &weights,
            cache,
            bias: &bias,
            output_dim: 2,
            dense_scale: 0.5,
            residual_rows: &residual,
            transposed_output: &mut transposed_output,
            output_rows: &mut output,
        }));
        assert_eq!(output, [12.75, 22.0, 35.75, 44.0]);
        assert!(!dense_batch_scaled_residual(&mut DenseBatchResidual {
            transposed_input: &transposed_input,
            row_count: 2,
            input_dim: 2,
            weights: &weights,
            cache: DenseWeightCache::new(),
            bias: &bias,
            output_dim: 2,
            dense_scale: 1.0,
            residual_rows: &residual,
            transposed_output: &mut transposed_output,
            output_rows: &mut output,
        }));
    }

    #[test]
    fn dense_batch_without_bias_preserves_the_batch_layout() {
        let input = [1.0, 2.0, 3.0, 4.0];
        let weights = [1.0, 2.0, -1.0, 3.0];
        let mut transposed_input = [0.0; 4];
        let mut transposed_output = [0.0; 4];
        let mut output = [0.0; 4];
        assert!(dense_batch_without_bias(DenseBatch {
            input_rows: &input,
            row_count: 2,
            input_dim: 2,
            weights: &weights,
            bias: &[],
            output_dim: 2,
            transposed_input: &mut transposed_input,
            transposed_output: &mut transposed_output,
            output_rows: &mut output,
        }));
        assert_eq!(output, [5.0, 5.0, 11.0, 9.0]);
        assert!(!dense_batch_without_bias(DenseBatch {
            input_rows: &input[..3],
            row_count: 2,
            input_dim: 2,
            weights: &weights,
            bias: &[],
            output_dim: 2,
            transposed_input: &mut transposed_input,
            transposed_output: &mut transposed_output,
            output_rows: &mut output,
        }));
    }

    #[test]
    fn prepared_no_bias_batch_requires_the_exact_cache() {
        let input = [1.0, 2.0, 3.0, 4.0];
        let weights = [1.0, 2.0, -1.0, 3.0];
        let mut cache = DenseWeightCache::new();
        assert!(cache.prepare(&weights, 2, 2));
        let mut transposed_input = [0.0; 4];
        let mut transposed_output = [0.0; 4];
        let mut output = [0.0; 4];
        {
            let mut batch = DenseBatch {
                input_rows: &input,
                row_count: 2,
                input_dim: 2,
                weights: &weights,
                bias: &[],
                output_dim: 2,
                transposed_input: &mut transposed_input,
                transposed_output: &mut transposed_output,
                output_rows: &mut output,
            };
            assert!(dense_batch_without_bias_prepared(&mut batch, cache));
        }
        assert_eq!(output, [5.0, 5.0, 11.0, 9.0]);
        let mut batch = DenseBatch {
            input_rows: &input,
            row_count: 2,
            input_dim: 2,
            weights: &weights,
            bias: &[],
            output_dim: 2,
            transposed_input: &mut transposed_input,
            transposed_output: &mut transposed_output,
            output_rows: &mut output,
        };
        assert!(!dense_batch_without_bias_prepared(
            &mut batch,
            DenseWeightCache::new()
        ));
    }

    #[test]
    fn dense_batch_from_transposed_matches_pinned_layout() {
        let transposed_input = [1.0, 3.0, 2.0, 4.0];
        let weights = [1.0, 2.0, -1.0, 3.0];
        let bias = [0.5, -1.0];
        let mut transposed_output = [0.0; 4];
        let mut output = [0.0; 4];
        {
            let mut batch = DenseBatchFromTransposed {
                transposed_input: &transposed_input,
                row_count: 2,
                input_dim: 2,
                weights: &weights,
                bias: &bias,
                output_dim: 2,
                transposed_output: &mut transposed_output,
                output_rows: &mut output,
            };
            assert!(dense_batch_from_transposed(&mut batch));
        }
        assert_eq!(output, [5.5, 4.0, 11.5, 8.0]);
        let mut batch = DenseBatchFromTransposed {
            transposed_input: &transposed_input[..3],
            row_count: 2,
            input_dim: 2,
            weights: &weights,
            bias: &bias,
            output_dim: 2,
            transposed_output: &mut transposed_output,
            output_rows: &mut output,
        };
        assert!(!dense_batch_from_transposed(&mut batch));
    }

    #[test]
    fn prepared_from_transposed_requires_the_exact_weight_identity() {
        let transposed_input = [1.0, 3.0, 2.0, 4.0];
        let weights = [1.0, 2.0, -1.0, 3.0];
        let bias = [0.5, -1.0];
        let mut cache = DenseWeightCache::new();
        assert!(cache.prepare(&weights, 2, 2));
        let mut transposed_output = [0.0; 4];
        let mut output = [0.0; 4];
        {
            let mut batch = DenseBatchFromTransposed {
                transposed_input: &transposed_input,
                row_count: 2,
                input_dim: 2,
                weights: &weights,
                bias: &bias,
                output_dim: 2,
                transposed_output: &mut transposed_output,
                output_rows: &mut output,
            };
            assert!(dense_batch_from_transposed_prepared(&mut batch, cache));
        }
        assert_eq!(output, [5.5, 4.0, 11.5, 8.0]);
        let mut batch = DenseBatchFromTransposed {
            transposed_input: &transposed_input,
            row_count: 2,
            input_dim: 2,
            weights: &weights,
            bias: &bias,
            output_dim: 2,
            transposed_output: &mut transposed_output,
            output_rows: &mut output,
        };
        assert!(!dense_batch_from_transposed_prepared(
            &mut batch,
            DenseWeightCache::new()
        ));
    }

    #[test]
    fn from_transposed_scaled_residual_preserves_prepared_contract() {
        let transposed_input = [1.0, 3.0, 2.0, 4.0];
        let weights = [1.0, 2.0, -1.0, 3.0];
        let bias = [0.5, -1.0];
        let residual = [10.0, 20.0, 30.0, 40.0];
        let mut cache = DenseWeightCache::new();
        assert!(cache.prepare(&weights, 2, 2));
        let mut transposed_output = [0.0; 4];
        let mut output = [0.0; 4];
        {
            let mut batch = DenseBatchResidual {
                transposed_input: &transposed_input,
                row_count: 2,
                input_dim: 2,
                weights: &weights,
                cache,
                bias: &bias,
                output_dim: 2,
                dense_scale: 0.5,
                residual_rows: &residual,
                transposed_output: &mut transposed_output,
                output_rows: &mut output,
            };
            assert!(dense_batch_from_transposed_scaled_residual(&mut batch));
        }
        assert_eq!(output, [12.75, 22.0, 35.75, 44.0]);

        let mut batch = DenseBatchResidual {
            transposed_input: &transposed_input,
            row_count: 2,
            input_dim: 2,
            weights: &weights,
            cache,
            bias: &bias,
            output_dim: 2,
            dense_scale: f32::NAN,
            residual_rows: &residual,
            transposed_output: &mut transposed_output,
            output_rows: &mut output,
        };
        assert!(dense_batch_from_transposed_scaled_residual(&mut batch));
        assert!(output.iter().any(|value| value.is_nan()));
    }

    #[test]
    fn dense_batch_to_transposed_matches_reference_output_orientation() {
        let input = [1.0, 2.0, 3.0, 4.0];
        let weights = [1.0, 2.0, -1.0, 3.0];
        let bias = [0.5, -1.0];
        let mut transposed_input = [0.0; 4];
        let mut transposed_output = [0.0; 4];
        let mut ignored_rows = [];
        {
            let mut batch = DenseBatch {
                input_rows: &input,
                row_count: 2,
                input_dim: 2,
                weights: &weights,
                bias: &bias,
                output_dim: 2,
                transposed_input: &mut transposed_input,
                transposed_output: &mut transposed_output,
                output_rows: &mut ignored_rows,
            };
            assert!(dense_batch_to_transposed(&mut batch));
        }
        assert_eq!(transposed_output, [5.5, 11.5, 4.0, 8.0]);
        let mut batch = DenseBatch {
            input_rows: &input,
            row_count: 2,
            input_dim: 2,
            weights: &weights,
            bias: &bias[..1],
            output_dim: 2,
            transposed_input: &mut transposed_input,
            transposed_output: &mut transposed_output,
            output_rows: &mut ignored_rows,
        };
        assert!(!dense_batch_to_transposed(&mut batch));
    }

    #[test]
    fn prepared_to_transposed_requires_exact_cache_identity() {
        let input = [1.0, 2.0, 3.0, 4.0];
        let weights = [1.0, 2.0, -1.0, 3.0];
        let bias = [0.5, -1.0];
        let mut cache = DenseWeightCache::new();
        assert!(cache.prepare(&weights, 2, 2));
        let mut transposed_input = [0.0; 4];
        let mut transposed_output = [0.0; 4];
        let mut ignored_rows = [];
        {
            let mut batch = DenseBatch {
                input_rows: &input,
                row_count: 2,
                input_dim: 2,
                weights: &weights,
                bias: &bias,
                output_dim: 2,
                transposed_input: &mut transposed_input,
                transposed_output: &mut transposed_output,
                output_rows: &mut ignored_rows,
            };
            assert!(dense_batch_to_transposed_prepared(&mut batch, cache));
        }
        assert_eq!(transposed_output, [5.5, 11.5, 4.0, 8.0]);
    }
}
