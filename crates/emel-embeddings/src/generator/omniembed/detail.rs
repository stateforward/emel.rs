//! Native, allocation-free `OmniEmbed` text execution details.
//!
//! The route mirrors the pinned source graph for resident F32 tensors. It
//! validates the complete bounded transformer inventory before execution and
//! reports malformed or unsupported tensors as typed failures.

use super::NativeTensorBinding;

const WORD: &[u8] = b"text_encoder.0.auto_model.embeddings.word_embeddings.weight";
const POSITION: &[u8] = b"text_encoder.0.auto_model.embeddings.position_embeddings.weight";
const TOKEN_TYPE: &[u8] = b"text_encoder.0.auto_model.embeddings.token_type_embeddings.weight";
const NORM_WEIGHT: &[u8] = b"text_encoder.0.auto_model.embeddings.LayerNorm.weight";
const NORM_BIAS: &[u8] = b"text_encoder.0.auto_model.embeddings.LayerNorm.bias";
const DENSE: &[u8] = b"text_encoder.2.linear.weight";
const DENSE_BIAS: &[u8] = b"text_encoder.2.linear.bias";
const EXPAND: &[u8] = b"text_projection.expand.0.weight";
const EXPAND_BIAS: &[u8] = b"text_projection.expand.0.bias";
const EXPAND_NORM_WEIGHT: &[u8] = b"text_projection.expand.2.weight";
const EXPAND_NORM_BIAS: &[u8] = b"text_projection.expand.2.bias";
const RESIDUAL: &[u8] = b"text_projection.residual_blocks.0.0.weight";
const RESIDUAL_BIAS: &[u8] = b"text_projection.residual_blocks.0.0.bias";
const RESIDUAL_NORM_WEIGHT: &[u8] = b"text_projection.residual_blocks.0.2.weight";
const RESIDUAL_NORM_BIAS: &[u8] = b"text_projection.residual_blocks.0.2.bias";
const PROJECT: &[u8] = b"text_projection.project.weight";
const PROJECT_BIAS: &[u8] = b"text_projection.project.bias";
const LAYER_PREFIX: &[u8] = b"text_encoder.0.auto_model.encoder.layer.";
pub(crate) const MAX_TEXT_LAYERS: usize = 16;
const MAX_WORK: usize = 4096;
const MAX_TOKEN_POSITIONS: usize = 4096;

const LAYER_SUFFIXES: [&[u8]; 16] = [
    b"attention.self.query.weight",
    b"attention.self.query.bias",
    b"attention.self.key.weight",
    b"attention.self.key.bias",
    b"attention.self.value.weight",
    b"attention.self.value.bias",
    b"attention.output.dense.weight",
    b"attention.output.dense.bias",
    b"attention.output.LayerNorm.weight",
    b"attention.output.LayerNorm.bias",
    b"intermediate.dense.weight",
    b"intermediate.dense.bias",
    b"output.dense.weight",
    b"output.dense.bias",
    b"output.LayerNorm.weight",
    b"output.LayerNorm.bias",
];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Error {
    InvalidRequest,
    UnsupportedTensor,
    ModelInvalid,
    Capacity,
}

fn f32_count(bytes: &[u8], count: usize) -> bool {
    count.checked_mul(4) == Some(bytes.len())
}

fn read_f32(bytes: &[u8], index: usize) -> Option<f32> {
    let start = index.checked_mul(4)?;
    Some(f32::from_le_bytes(
        bytes.get(start..start.checked_add(4)?)?.try_into().ok()?,
    ))
}

fn view<'a>(
    binding: &'a NativeTensorBinding<'_>,
    name: &[u8],
) -> Result<emel_model::bridge::TensorView<'a>, Error> {
    binding.data().tensor_named(name).ok_or(Error::ModelInvalid)
}

fn vector<'a>(
    binding: &'a NativeTensorBinding<'_>,
    name: &[u8],
    length: usize,
) -> Result<&'a [u8], Error> {
    let tensor = view(binding, name)?;
    let metadata = tensor.metadata().ok_or(Error::ModelInvalid)?;
    let dimensions = metadata.dimensions();
    if metadata.tensor_type().wire_code() != 0
        || metadata.dimension_count() != 1
        || dimensions[0] != u64::try_from(length).map_err(|_| Error::UnsupportedTensor)?
    {
        return Err(Error::UnsupportedTensor);
    }
    let bytes = tensor.bytes().ok_or(Error::ModelInvalid)?;
    f32_count(bytes, length)
        .then_some(bytes)
        .ok_or(Error::ModelInvalid)
}

fn matrix<'a>(
    binding: &'a NativeTensorBinding<'_>,
    name: &[u8],
    rows: usize,
    columns: usize,
) -> Result<&'a [u8], Error> {
    let tensor = view(binding, name)?;
    let metadata = tensor.metadata().ok_or(Error::ModelInvalid)?;
    let dimensions = metadata.dimensions();
    if metadata.tensor_type().wire_code() != 0
        || metadata.dimension_count() != 2
        || dimensions[0] != u64::try_from(columns).map_err(|_| Error::UnsupportedTensor)?
        || dimensions[1] != u64::try_from(rows).map_err(|_| Error::UnsupportedTensor)?
    {
        return Err(Error::UnsupportedTensor);
    }
    let bytes = tensor.bytes().ok_or(Error::ModelInvalid)?;
    let count = rows.checked_mul(columns).ok_or(Error::Capacity)?;
    f32_count(bytes, count)
        .then_some(bytes)
        .ok_or(Error::ModelInvalid)
}

#[allow(clippy::suboptimal_flops)]
fn matvec(matrix: &[u8], rows: usize, columns: usize, input: &[f32], output: &mut [f32]) -> bool {
    if input.len() < columns || output.len() < rows {
        return false;
    }
    for (row, output_value) in output.iter_mut().take(rows).enumerate() {
        let mut value = 0.0;
        let Some(start) = row.checked_mul(columns) else {
            return false;
        };
        for (column, input_value) in input.iter().take(columns).enumerate() {
            let Some(weight) = read_f32(matrix, start + column) else {
                return false;
            };
            value += weight * *input_value;
        }
        *output_value = value;
    }
    true
}

#[allow(clippy::cast_precision_loss, clippy::suboptimal_flops)]
fn layer_norm(values: &mut [f32], weight: &[u8], bias: &[u8], epsilon: f32) -> bool {
    if values.is_empty() || weight.len() != values.len() * 4 || bias.len() != values.len() * 4 {
        return false;
    }
    let mut mean = 0.0_f64;
    for value in values.iter().copied() {
        mean += f64::from(value);
    }
    mean /= values.len() as f64;
    let mut variance = 0.0_f64;
    for value in values.iter().copied() {
        let delta = f64::from(value) - mean;
        variance += delta * delta;
    }
    variance /= values.len() as f64;
    let scale = ((variance as f32) + epsilon).sqrt().recip();
    for (index, value) in values.iter_mut().enumerate() {
        let (Some(gain), Some(offset)) = (read_f32(weight, index), read_f32(bias, index)) else {
            return false;
        };
        *value = (*value - mean as f32) * scale * gain + offset;
    }
    true
}

#[allow(clippy::suboptimal_flops)]
fn gelu(value: f32) -> f32 {
    let sign = if value < 0.0 { -1.0 } else { 1.0 };
    let x = value.abs() / core::f32::consts::SQRT_2;
    let t = 1.0 / (1.0 + 0.327_591_1 * x);
    let polynomial = (((((1.061_405_4 * t - 1.453_152_1) * t) + 1.421_413_8) * t - 0.284_496_72)
        * t
        + 0.254_829_6)
        * t;
    let erf = sign * (1.0 - polynomial * (-x * x).exp());
    0.5 * value * (1.0 + erf)
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct LayerTensors<'a> {
    query: &'a [u8],
    query_bias: &'a [u8],
    key: &'a [u8],
    key_bias: &'a [u8],
    value: &'a [u8],
    value_bias: &'a [u8],
    attention_output: &'a [u8],
    attention_output_bias: &'a [u8],
    attention_norm_weight: &'a [u8],
    attention_norm_bias: &'a [u8],
    intermediate: &'a [u8],
    intermediate_bias: &'a [u8],
    output: &'a [u8],
    output_bias: &'a [u8],
    output_norm_weight: &'a [u8],
    output_norm_bias: &'a [u8],
}

fn layer_name<'a>(index: usize, suffix: &[u8], output: &'a mut [u8; 128]) -> Option<&'a [u8]> {
    let mut length = LAYER_PREFIX.len();
    output.get_mut(..length)?.copy_from_slice(LAYER_PREFIX);
    let mut digits = [0_u8; 20];
    let mut digit_count = 0;
    let mut value = index;
    loop {
        digits[digit_count] = b'0' + (value % 10) as u8;
        digit_count += 1;
        value /= 10;
        if value == 0 {
            break;
        }
    }
    for digit in digits[..digit_count].iter().rev() {
        *output.get_mut(length)? = *digit;
        length += 1;
    }
    *output.get_mut(length)? = b'.';
    length += 1;
    let end = length.checked_add(suffix.len())?;
    output.get_mut(length..end)?.copy_from_slice(suffix);
    Some(&output[..end])
}

fn tensor_named<'a>(
    data: &'a emel_model::bridge::Data,
    name: &[u8],
) -> Result<emel_model::bridge::TensorView<'a>, Error> {
    data.tensor_named(name).ok_or(Error::ModelInvalid)
}

fn layer_tensor<'a>(
    data: &'a emel_model::bridge::Data,
    index: usize,
    suffix: &[u8],
) -> Result<emel_model::bridge::TensorView<'a>, Error> {
    let mut name = [0_u8; 128];
    let name = layer_name(index, suffix, &mut name).ok_or(Error::Capacity)?;
    tensor_named(data, name)
}

fn bind_layer<'a>(
    data: &'a emel_model::bridge::Data,
    index: usize,
    hidden: usize,
) -> Result<LayerTensors<'a>, Error> {
    let intermediate = hidden.checked_mul(4).ok_or(Error::Capacity)?;
    let q = layer_tensor(data, index, LAYER_SUFFIXES[0])?;
    let qb = layer_tensor(data, index, LAYER_SUFFIXES[1])?;
    let k = layer_tensor(data, index, LAYER_SUFFIXES[2])?;
    let kb = layer_tensor(data, index, LAYER_SUFFIXES[3])?;
    let v = layer_tensor(data, index, LAYER_SUFFIXES[4])?;
    let vb = layer_tensor(data, index, LAYER_SUFFIXES[5])?;
    let ao = layer_tensor(data, index, LAYER_SUFFIXES[6])?;
    let aob = layer_tensor(data, index, LAYER_SUFFIXES[7])?;
    let anw = layer_tensor(data, index, LAYER_SUFFIXES[8])?;
    let anb = layer_tensor(data, index, LAYER_SUFFIXES[9])?;
    let ff = layer_tensor(data, index, LAYER_SUFFIXES[10])?;
    let ffb = layer_tensor(data, index, LAYER_SUFFIXES[11])?;
    let out = layer_tensor(data, index, LAYER_SUFFIXES[12])?;
    let outb = layer_tensor(data, index, LAYER_SUFFIXES[13])?;
    let onw = layer_tensor(data, index, LAYER_SUFFIXES[14])?;
    let onb = layer_tensor(data, index, LAYER_SUFFIXES[15])?;
    let matrix = |tensor: emel_model::bridge::TensorView<'a>, rows: usize, columns: usize| {
        let metadata = tensor.metadata().ok_or(Error::ModelInvalid)?;
        let dimensions = metadata.dimensions();
        if metadata.tensor_type().wire_code() != 0
            || metadata.dimension_count() != 2
            || dimensions[0] != u64::try_from(columns).map_err(|_| Error::UnsupportedTensor)?
            || dimensions[1] != u64::try_from(rows).map_err(|_| Error::UnsupportedTensor)?
        {
            return Err(Error::UnsupportedTensor);
        }
        let bytes = tensor.bytes().ok_or(Error::ModelInvalid)?;
        f32_count(bytes, rows.checked_mul(columns).ok_or(Error::Capacity)?)
            .then_some(bytes)
            .ok_or(Error::ModelInvalid)
    };
    let vector = |tensor: emel_model::bridge::TensorView<'a>, length: usize| {
        let metadata = tensor.metadata().ok_or(Error::ModelInvalid)?;
        let dimensions = metadata.dimensions();
        if metadata.tensor_type().wire_code() != 0
            || metadata.dimension_count() != 1
            || dimensions[0] != u64::try_from(length).map_err(|_| Error::UnsupportedTensor)?
        {
            return Err(Error::UnsupportedTensor);
        }
        let bytes = tensor.bytes().ok_or(Error::ModelInvalid)?;
        f32_count(bytes, length).then_some(bytes).ok_or(Error::ModelInvalid)
    };
    Ok(LayerTensors {
        query: matrix(q, hidden, hidden)?,
        query_bias: vector(qb, hidden)?,
        key: matrix(k, hidden, hidden)?,
        key_bias: vector(kb, hidden)?,
        value: matrix(v, hidden, hidden)?,
        value_bias: vector(vb, hidden)?,
        attention_output: matrix(ao, hidden, hidden)?,
        attention_output_bias: vector(aob, hidden)?,
        attention_norm_weight: vector(anw, hidden)?,
        attention_norm_bias: vector(anb, hidden)?,
        intermediate: matrix(ff, intermediate, hidden)?,
        intermediate_bias: vector(ffb, intermediate)?,
        output: matrix(out, hidden, intermediate)?,
        output_bias: vector(outb, hidden)?,
        output_norm_weight: vector(onw, hidden)?,
        output_norm_bias: vector(onb, hidden)?,
    })
}

/// Validates and binds the contiguous, bounded transformer-layer inventory.
pub(crate) fn bind_layer_inventory<'a>(
    data: &'a emel_model::bridge::Data,
) -> Result<([Option<LayerTensors<'a>>; MAX_TEXT_LAYERS], usize), Error> {
    let word = tensor_named(data, WORD)?;
    let metadata = word.metadata().ok_or(Error::ModelInvalid)?;
    let dimensions = metadata.dimensions();
    if metadata.tensor_type().wire_code() != 0 || metadata.dimension_count() != 2 {
        return Err(Error::UnsupportedTensor);
    }
    let hidden = usize::try_from(dimensions[0]).map_err(|_| Error::UnsupportedTensor)?;
    if hidden == 0 || hidden > MAX_WORK || hidden.checked_mul(4).ok_or(Error::Capacity)? > MAX_WORK {
        return Err(Error::Capacity);
    }
    let mut layers = [None; MAX_TEXT_LAYERS];
    let mut count = 0;
    for index in 0..MAX_TEXT_LAYERS {
        let mut present = false;
        for suffix in LAYER_SUFFIXES {
            let mut name = [0_u8; 128];
            let name = layer_name(index, suffix, &mut name).ok_or(Error::Capacity)?;
            if data.tensor_named(name).is_some() {
                present = true;
                break;
            }
        }
        if !present {
            break;
        }
        layers[index] = Some(bind_layer(data, index, hidden)?);
        count += 1;
    }
    for index in 0..data.tensor_count() {
        let tensor = data.tensor(index).ok_or(Error::ModelInvalid)?;
        if tensor.name().starts_with(LAYER_PREFIX) {
            let mut found = false;
            for layer in 0..count {
                let mut name = [0_u8; 128];
                for suffix in LAYER_SUFFIXES {
                    if layer_name(layer, suffix, &mut name) == Some(tensor.name()) {
                        found = true;
                    }
                }
            }
            if !found {
                return Err(if count == MAX_TEXT_LAYERS { Error::Capacity } else { Error::ModelInvalid });
            }
        }
    }
    Ok((layers, count))
}

fn run_layered(
    binding: &NativeTensorBinding<'_>, token_ids: &[i32], hidden: usize, vocabulary: usize,
    max_positions: usize, position: &[u8], token_type: &[u8], norm_weight: &[u8], norm_bias: &[u8],
    pooled: &mut [f32],
) -> Result<(), Error> {
    let heads = 12;
    if hidden == 0 || hidden % heads != 0 || token_ids.is_empty() || token_ids.len() > max_positions || max_positions > MAX_TOKEN_POSITIONS { return Err(Error::UnsupportedTensor); }
    let elements = token_ids.len().checked_mul(hidden).ok_or(Error::Capacity)?;
    if elements > MAX_WORK || binding.text_layer_count > MAX_TEXT_LAYERS { return Err(Error::Capacity); }
    let head_dim = hidden / heads;
    let mut sequence_a = [0.0_f32; MAX_WORK]; let mut sequence_b = [0.0_f32; MAX_WORK];
    let mut query = [0.0_f32; MAX_WORK]; let mut key = [0.0_f32; MAX_WORK]; let mut value = [0.0_f32; MAX_WORK];
    let mut context = [0.0_f32; MAX_WORK]; let mut scores = [0.0_f32; MAX_TOKEN_POSITIONS];
    let mut hidden_values = [0.0_f32; MAX_WORK]; let mut feed_forward = [0.0_f32; MAX_WORK];
    let word = view(binding, WORD)?.bytes().ok_or(Error::ModelInvalid)?;
    for (token_index, token_id) in token_ids.iter().copied().enumerate() {
        let token = usize::try_from(token_id).map_err(|_| Error::InvalidRequest)?;
        if token >= vocabulary { return Err(Error::InvalidRequest); }
        let start = token_index.checked_mul(hidden).ok_or(Error::Capacity)?;
        let word_start = token.checked_mul(hidden).ok_or(Error::Capacity)?;
        for feature in 0..hidden {
            sequence_a[start + feature] = read_f32(word, word_start + feature).ok_or(Error::ModelInvalid)?
                + read_f32(position, start + feature).ok_or(Error::ModelInvalid)?
                + read_f32(token_type, feature).ok_or(Error::ModelInvalid)?;
        }
        if !layer_norm(&mut sequence_a[start..start + hidden], norm_weight, norm_bias, 1.0e-12) { return Err(Error::ModelInvalid); }
    }
    for layer_index in 0..binding.text_layer_count {
        let layer = binding.text_layers[layer_index].ok_or(Error::ModelInvalid)?;
        for token_index in 0..token_ids.len() {
            let start = token_index * hidden; let input = &sequence_a[start..start + hidden];
            let q = &mut query[start..start + hidden]; let k = &mut key[start..start + hidden]; let v = &mut value[start..start + hidden];
            if !matvec(layer.query, hidden, hidden, input, q) || !matvec(layer.key, hidden, hidden, input, k) || !matvec(layer.value, hidden, hidden, input, v) { return Err(Error::ModelInvalid); }
            for feature in 0..hidden {
                q[feature] += read_f32(layer.query_bias, feature).ok_or(Error::ModelInvalid)?;
                k[feature] += read_f32(layer.key_bias, feature).ok_or(Error::ModelInvalid)?;
                v[feature] += read_f32(layer.value_bias, feature).ok_or(Error::ModelInvalid)?;
            }
        }
        for token_index in 0..token_ids.len() {
            let start = token_index * hidden; context[start..start + hidden].fill(0.0);
            for head in 0..heads {
                let qstart = start + head * head_dim;
                for other in 0..token_ids.len() {
                    let ostart = other * hidden + head * head_dim; let mut score = 0.0;
                    for feature in 0..head_dim { score += query[qstart + feature] * key[ostart + feature]; }
                    scores[other] = score * 0.176_776_7;
                }
                let max_score = scores[..token_ids.len()].iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let mut sum = 0.0_f32;
                for score in &mut scores[..token_ids.len()] { *score = (*score - max_score).exp(); sum += *score; }
                if !sum.is_finite() || sum <= 0.0 { return Err(Error::ModelInvalid); }
                for score in &mut scores[..token_ids.len()] { *score /= sum; }
                for other in 0..token_ids.len() {
                    let vstart = other * hidden + head * head_dim;
                    for feature in 0..head_dim { context[start + head * head_dim + feature] += value[vstart + feature] * scores[other]; }
                }
            }
            if !matvec(layer.attention_output, hidden, hidden, &context[start..start + hidden], &mut hidden_values[..hidden]) { return Err(Error::ModelInvalid); }
            for feature in 0..hidden { hidden_values[feature] += read_f32(layer.attention_output_bias, feature).ok_or(Error::ModelInvalid)? + sequence_a[start + feature]; }
            if !layer_norm(&mut hidden_values[..hidden], layer.attention_norm_weight, layer.attention_norm_bias, 1.0e-12) { return Err(Error::ModelInvalid); }
            if !matvec(layer.intermediate, hidden * 4, hidden, &hidden_values[..hidden], &mut feed_forward[..hidden * 4]) { return Err(Error::ModelInvalid); }
            for feature in 0..hidden * 4 { feed_forward[feature] = gelu(feed_forward[feature] + read_f32(layer.intermediate_bias, feature).ok_or(Error::ModelInvalid)?); }
            let output_row = &mut sequence_b[start..start + hidden];
            if !matvec(layer.output, hidden, hidden * 4, &feed_forward[..hidden * 4], output_row) { return Err(Error::ModelInvalid); }
            for feature in 0..hidden { output_row[feature] += read_f32(layer.output_bias, feature).ok_or(Error::ModelInvalid)? + hidden_values[feature]; }
            if !layer_norm(output_row, layer.output_norm_weight, layer.output_norm_bias, 1.0e-12) { return Err(Error::ModelInvalid); }
        }
        core::mem::swap(&mut sequence_a, &mut sequence_b);
    }
    pooled[..hidden].fill(0.0); let scale = 1.0 / token_ids.len() as f32;
    for token_index in 0..token_ids.len() { for feature in 0..hidden { pooled[feature] += sequence_a[token_index * hidden + feature] * scale; } }
    Ok(())
}

/// Execute the source text embedding/projection path.
///
/// This consumes model-owned resident bytes by immutable borrow and writes
/// only to caller-provided output. It never allocates. Quantized tensors,
/// malformed geometry, and incomplete transformer layers are typed failures.
#[allow(clippy::cast_precision_loss, clippy::too_many_lines)]
pub fn execute_text(
    binding: &NativeTensorBinding<'_>,
    token_ids: &[i32],
    output: &mut [f32],
) -> Result<usize, Error> {
    if token_ids.is_empty() || output.is_empty() { return Err(Error::InvalidRequest); }
    if token_ids.len() > MAX_TOKEN_POSITIONS { return Err(Error::Capacity); }
    let layer_count = binding.text_layer_count;
    if layer_count > MAX_TEXT_LAYERS { return Err(Error::Capacity); }
    let word_tensor = view(binding, WORD)?;
    let word_meta = word_tensor.metadata().ok_or(Error::ModelInvalid)?;
    let word_dims = word_meta.dimensions();
    if word_meta.tensor_type().wire_code() != 0 || word_meta.dimension_count() != 2 { return Err(Error::UnsupportedTensor); }
    let hidden = usize::try_from(word_dims[0]).map_err(|_| Error::UnsupportedTensor)?;
    let vocabulary = usize::try_from(word_dims[1]).map_err(|_| Error::UnsupportedTensor)?;
    if hidden == 0 || hidden > MAX_WORK || hidden % 12 != 0 || vocabulary == 0 { return Err(Error::Capacity); }
    let word = word_tensor.bytes().ok_or(Error::ModelInvalid)?;
    if !f32_count(word, hidden.checked_mul(vocabulary).ok_or(Error::Capacity)?) { return Err(Error::ModelInvalid); }
    let position_tensor = view(binding, POSITION)?;
    let position_meta = position_tensor.metadata().ok_or(Error::ModelInvalid)?;
    let position_dims = position_meta.dimensions();
    if position_meta.tensor_type().wire_code() != 0 || position_meta.dimension_count() != 2 { return Err(Error::UnsupportedTensor); }
    let max_positions = usize::try_from(position_dims[1]).map_err(|_| Error::UnsupportedTensor)?;
    if position_dims[0] != u64::try_from(hidden).map_err(|_| Error::UnsupportedTensor)? || max_positions == 0 || max_positions < token_ids.len() { return Err(Error::UnsupportedTensor); }
    let position = matrix(binding, POSITION, max_positions, hidden)?;
    let token_type_tensor = view(binding, TOKEN_TYPE)?;
    let token_type_meta = token_type_tensor.metadata().ok_or(Error::ModelInvalid)?;
    let token_type_dims = token_type_meta.dimensions();
    if token_type_meta.tensor_type().wire_code() != 0 || token_type_meta.dimension_count() != 2 { return Err(Error::UnsupportedTensor); }
    let token_type_rows = usize::try_from(token_type_dims[1]).map_err(|_| Error::UnsupportedTensor)?;
    if token_type_rows == 0 || token_type_dims[0] != u64::try_from(hidden).map_err(|_| Error::UnsupportedTensor)? { return Err(Error::UnsupportedTensor); }
    let token_type = matrix(binding, TOKEN_TYPE, token_type_rows, hidden)?;
    let norm_weight = vector(binding, NORM_WEIGHT, hidden)?;
    let norm_bias = vector(binding, NORM_BIAS, hidden)?;
    let dense_meta = view(binding, DENSE)?.metadata().ok_or(Error::ModelInvalid)?;
    let dense_dims = dense_meta.dimensions();
    let dense_output = usize::try_from(dense_dims[1]).map_err(|_| Error::UnsupportedTensor)?;
    if dense_output > MAX_WORK { return Err(Error::Capacity); }
    let dense = matrix(binding, DENSE, dense_output, hidden)?;
    let dense_bias = vector(binding, DENSE_BIAS, dense_output)?;
    let expand_meta = view(binding, EXPAND)?.metadata().ok_or(Error::ModelInvalid)?;
    let expand_dims = expand_meta.dimensions();
    let projection_hidden = usize::try_from(expand_dims[1]).map_err(|_| Error::UnsupportedTensor)?;
    if projection_hidden > MAX_WORK { return Err(Error::Capacity); }
    let expand = matrix(binding, EXPAND, projection_hidden, dense_output)?;
    let expand_bias = vector(binding, EXPAND_BIAS, projection_hidden)?;
    let expand_norm_weight = vector(binding, EXPAND_NORM_WEIGHT, projection_hidden)?;
    let expand_norm_bias = vector(binding, EXPAND_NORM_BIAS, projection_hidden)?;
    let residual = matrix(binding, RESIDUAL, projection_hidden, projection_hidden)?;
    let residual_bias = vector(binding, RESIDUAL_BIAS, projection_hidden)?;
    let residual_norm_weight = vector(binding, RESIDUAL_NORM_WEIGHT, projection_hidden)?;
    let residual_norm_bias = vector(binding, RESIDUAL_NORM_BIAS, projection_hidden)?;
    let project_meta = view(binding, PROJECT)?.metadata().ok_or(Error::ModelInvalid)?;
    let project_dims = project_meta.dimensions();
    let embedding = usize::try_from(project_dims[1]).map_err(|_| Error::UnsupportedTensor)?;
    if embedding == 0 || embedding > output.len() || embedding > MAX_WORK { return Err(Error::Capacity); }
    let project = matrix(binding, PROJECT, embedding, projection_hidden)?;
    let project_bias = vector(binding, PROJECT_BIAS, embedding)?;
    let mut pooled = [0.0_f32; MAX_WORK];
    let mut dense_values = [0.0_f32; MAX_WORK];
    let mut expanded = [0.0_f32; MAX_WORK];
    if layer_count == 0 {
        for (position_index, token_id) in token_ids.iter().copied().enumerate() {
            let token = usize::try_from(token_id).map_err(|_| Error::InvalidRequest)?;
            if token >= vocabulary { return Err(Error::InvalidRequest); }
            let position_start = position_index.checked_mul(hidden).ok_or(Error::Capacity)?;
            let word_start = token.checked_mul(hidden).ok_or(Error::Capacity)?;
            for feature in 0..hidden {
                expanded[position_start + feature] = read_f32(word, word_start + feature).ok_or(Error::ModelInvalid)? + read_f32(position, position_start + feature).ok_or(Error::ModelInvalid)? + read_f32(token_type, feature).ok_or(Error::ModelInvalid)?;
            }
            if !layer_norm(&mut expanded[position_start..position_start + hidden], norm_weight, norm_bias, 1.0e-12) { return Err(Error::ModelInvalid); }
        }
        pooled[..hidden].fill(0.0);
        let scale = 1.0 / token_ids.len() as f32;
        for token_index in 0..token_ids.len() { let start = token_index.checked_mul(hidden).ok_or(Error::Capacity)?; for feature in 0..hidden { pooled[feature] += expanded[start + feature] * scale; } }
    } else {
        run_layered(binding, token_ids, hidden, vocabulary, max_positions, position, token_type, norm_weight, norm_bias, &mut pooled)?;
    }
    if !matvec(dense, dense_output, hidden, &pooled[..hidden], &mut dense_values[..dense_output]) { return Err(Error::ModelInvalid); }
    for (index, value) in dense_values[..dense_output].iter_mut().enumerate() { *value += read_f32(dense_bias, index).ok_or(Error::ModelInvalid)?; }
    if !matvec(expand, projection_hidden, dense_output, &dense_values[..dense_output], &mut expanded[..projection_hidden]) { return Err(Error::ModelInvalid); }
    for (index, value) in expanded[..projection_hidden].iter_mut().enumerate() { *value = gelu(*value + read_f32(expand_bias, index).ok_or(Error::ModelInvalid)?); }
    if !layer_norm(&mut expanded[..projection_hidden], expand_norm_weight, expand_norm_bias, 1.0e-5) { return Err(Error::ModelInvalid); }
    let mut residual_values = [0.0_f32; MAX_WORK];
    if !matvec(residual, projection_hidden, projection_hidden, &expanded[..projection_hidden], &mut residual_values[..projection_hidden]) { return Err(Error::ModelInvalid); }
    for (index, expanded_value) in expanded[..projection_hidden].iter_mut().enumerate() { *expanded_value += residual_values[index] + read_f32(residual_bias, index).ok_or(Error::ModelInvalid)?; }
    if !layer_norm(&mut expanded[..projection_hidden], residual_norm_weight, residual_norm_bias, 1.0e-5) { return Err(Error::ModelInvalid); }
    if !matvec(project, embedding, projection_hidden, &expanded[..projection_hidden], &mut output[..embedding]) { return Err(Error::ModelInvalid); }
    for (index, value) in output[..embedding].iter_mut().enumerate() { *value += read_f32(project_bias, index).ok_or(Error::ModelInvalid)?; }
    let norm = output[..embedding].iter().map(|value| value * value).sum::<f32>().sqrt();
    if !norm.is_finite() || norm <= 0.0 { return Err(Error::ModelInvalid); }
    for value in &mut output[..embedding] { *value /= norm; }
    Ok(embedding)
}
