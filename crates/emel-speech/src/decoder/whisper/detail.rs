#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::needless_lifetimes,
    clippy::needless_pass_by_ref_mut,
    clippy::needless_pass_by_value,
    clippy::needless_range_loop,
    clippy::option_if_let_else,
    clippy::suboptimal_flops,
    clippy::too_many_arguments,
    clippy::too_many_lines,
    clippy::unreadable_literal,
    reason = "native decoder kernel preserves source arithmetic and fixed workspace semantics"
)]

//! Callback-free native `Q8_0` Whisper decoder execution.
//!
//! The implementation deliberately keeps all persistent storage in the caller's
//! workspace.  The fixed-size local token history is bounded to the decoder
//! sequence and is dispatch-local, never retained in the actor context.

use super::sm::{DecodeKernelRequest, DecodePolicy, WhisperExecutionContract};
use emel_model::bridge::{Data, TensorView};
use emel_tensor::dtype::SerializedType;

const E: usize = 384;
const FF: usize = 1536;
const BLOCKS: usize = 4;
const HEADS: usize = 6;
const HEAD_DIM: usize = 64;
const VOCAB: usize = 51_865;
const SEQ: usize = 448;
const EPS: f32 = 1.0e-5;
const E_F32: f32 = 384.0;
const HEAD_DIM_F32: f32 = 64.0;
const FNV_OFFSET: u64 = 1_469_598_103_934_665_603;
const FNV_PRIME: u64 = 1_099_511_628_211;
const ENCODER_ATTN_K_SUFFIX: &[u8] = b"encoder_attn.k_proj.weight";
const ENCODER_ATTN_V_SUFFIX: &[u8] = b"encoder_attn.v_proj.weight";
const ENCODER_ATTN_V_BIAS_SUFFIX: &[u8] = b"encoder_attn.v_proj.bias";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NativeError {
    MissingTensor,
    InvalidTensor,
    UnsupportedDType,
}

#[derive(Clone, Copy)]
struct Bytes<'a> {
    data: &'a [u8],
    kind: SerializedType,
}

trait LinearRoute {
    const KIND: SerializedType;
    fn value(bytes: Bytes<'_>, index: usize, cols: usize) -> Option<f32>;
}

trait AuxRoute {
    const KIND: SerializedType;
    fn value(bytes: Bytes<'_>, index: usize) -> Option<f32>;
}

struct Q8Linear;
struct Q40Linear;
struct Q41Linear;
struct Q8Aux;
struct F32Aux;

fn tensor<'a>(model: &'a Data, name: &[u8]) -> Result<TensorView<'a>, NativeError> {
    model.tensor_named(name).ok_or(NativeError::MissingTensor)
}

fn checked<'a>(
    view: TensorView<'a>,
    dims: &[u64],
    kind: SerializedType,
) -> Result<Bytes<'a>, NativeError> {
    let metadata = view.metadata().ok_or(NativeError::InvalidTensor)?;
    if metadata.tensor_type() != kind {
        return Err(NativeError::UnsupportedDType);
    }
    if usize::try_from(metadata.dimension_count()).ok() != Some(dims.len())
        || metadata.dimensions().get(..dims.len()) != Some(dims)
    {
        return Err(NativeError::InvalidTensor);
    }
    Ok(Bytes {
        data: view.bytes().ok_or(NativeError::InvalidTensor)?,
        kind,
    })
}

fn checked_any<'a>(view: TensorView<'a>, dims: &[u64]) -> Result<Bytes<'a>, NativeError> {
    let metadata = view.metadata().ok_or(NativeError::InvalidTensor)?;
    if usize::try_from(metadata.dimension_count()).ok() != Some(dims.len())
        || metadata.dimensions().get(..dims.len()) != Some(dims)
    {
        return Err(NativeError::InvalidTensor);
    }
    let kind = metadata.tensor_type();
    if !matches!(kind, SerializedType::Q8_0 | SerializedType::F32) {
        return Err(NativeError::UnsupportedDType);
    }
    Ok(Bytes {
        data: view.bytes().ok_or(NativeError::InvalidTensor)?,
        kind,
    })
}

fn f16(bits: u16) -> f32 {
    let word = u32::from(bits) << 16;
    let sign = word & 0x8000_0000;
    let doubled = word.wrapping_add(word);
    let normalized =
        f32::from_bits((doubled >> 4).wrapping_add(0xe0_u32 << 23)) * f32::from_bits(0x0780_0000);
    let denormalized = f32::from_bits((doubled >> 17) | (126_u32 << 23)) - 0.5;
    f32::from_bits(if doubled < 1_u32 << 27 {
        sign | denormalized.to_bits()
    } else {
        sign | normalized.to_bits()
    })
}

fn f32_at(data: &[u8], index: usize) -> Option<f32> {
    let start = index.checked_mul(4)?;
    Some(f32::from_ne_bytes(
        data.get(start..start + 4)?.try_into().ok()?,
    ))
}

fn q8_at(data: &[u8], index: usize, cols: usize) -> Option<f32> {
    let row = index / cols;
    let col = index % cols;
    let row_bytes = 34usize.checked_mul(cols.checked_div(32)?)?;
    let start = row.checked_mul(row_bytes)?.checked_add(col / 32 * 34)?;
    let block = data.get(start..start + 34)?;
    Some(
        f16(u16::from_ne_bytes(block[..2].try_into().ok()?))
            * f32::from(i8::from_ne_bytes([block[2 + col % 32]])),
    )
}

fn q4_at<const ONE: bool>(data: &[u8], index: usize, cols: usize) -> Option<f32> {
    let row = index / cols;
    let col = index % cols;
    let block_bytes: usize = if ONE { 20 } else { 18 };
    let row_bytes = block_bytes.checked_mul(cols.checked_div(32)?)?;
    let start = row
        .checked_mul(row_bytes)?
        .checked_add((col / 32).checked_mul(block_bytes)?)?;
    let block = data.get(start..start + block_bytes)?;
    let scale = f16(u16::from_ne_bytes(block[..2].try_into().ok()?));
    let minimum = if ONE {
        f16(u16::from_ne_bytes(block[2..4].try_into().ok()?))
    } else {
        0.0
    };
    let packed = block[if ONE { 4 } else { 2 } + col % 16];
    let nibble = if col % 32 < 16 {
        packed & 0x0f
    } else {
        packed >> 4
    };
    Some(
        scale
            * if ONE {
                f32::from(nibble)
            } else {
                f32::from(nibble) - 8.0
            }
            + minimum,
    )
}

impl LinearRoute for Q8Linear {
    const KIND: SerializedType = SerializedType::Q8_0;
    fn value(bytes: Bytes<'_>, index: usize, cols: usize) -> Option<f32> {
        q8_at(bytes.data, index, cols)
    }
}

impl LinearRoute for Q40Linear {
    const KIND: SerializedType = SerializedType::Q4_0;
    fn value(bytes: Bytes<'_>, index: usize, cols: usize) -> Option<f32> {
        q4_at::<false>(bytes.data, index, cols)
    }
}

impl LinearRoute for Q41Linear {
    const KIND: SerializedType = SerializedType::Q4_1;
    fn value(bytes: Bytes<'_>, index: usize, cols: usize) -> Option<f32> {
        q4_at::<true>(bytes.data, index, cols)
    }
}

impl AuxRoute for Q8Aux {
    const KIND: SerializedType = SerializedType::Q8_0;
    fn value(bytes: Bytes<'_>, index: usize) -> Option<f32> {
        q8_at(bytes.data, index, E)
    }
}

impl AuxRoute for F32Aux {
    const KIND: SerializedType = SerializedType::F32;
    fn value(bytes: Bytes<'_>, index: usize) -> Option<f32> {
        f32_at(bytes.data, index)
    }
}

fn name(block: usize, suffix: &[u8], out: &mut [u8; 128]) -> Option<usize> {
    let prefix = b"model.decoder.layers.";
    let mut used = prefix.len();
    out[..used].copy_from_slice(prefix);
    let mut digits = [0_u8; 20];
    let mut n = block;
    let mut count = 0;
    loop {
        digits[count] = b'0' + u8::try_from(n % 10).ok()?;
        count += 1;
        n /= 10;
        if n == 0 {
            break;
        }
    }
    while count != 0 {
        count -= 1;
        out[used] = digits[count];
        used += 1;
    }
    out[used] = b'.';
    used += 1;
    let end = used.checked_add(suffix.len())?;
    if end > out.len() {
        return None;
    }
    out[used..end].copy_from_slice(suffix);
    Some(end)
}

fn linear<L: LinearRoute, A: AuxRoute>(
    weight: Bytes<'_>,
    bias: Option<Bytes<'_>>,
    input: &[f32],
    output: &mut [f32],
    cols: usize,
    rows: usize,
) -> Option<()> {
    for row in 0..rows {
        let mut sum = 0.0;
        for col in 0..cols {
            sum += L::value(weight, row * cols + col, cols)? * input[col];
        }
        output[row] = sum + bias.and_then(|b| A::value(b, row)).unwrap_or(0.0);
    }
    Some(())
}

fn norm<A: AuxRoute>(
    input: &[f32],
    weight: Bytes<'_>,
    bias: Bytes<'_>,
    output: &mut [f32],
) -> Option<()> {
    let mean = input.iter().sum::<f32>() / E_F32;
    let variance = input
        .iter()
        .map(|x| {
            let d = *x - mean;
            d * d
        })
        .sum::<f32>()
        / E_F32;
    let inv = 1.0 / (variance + EPS).sqrt();
    for i in 0..E {
        output[i] = (input[i] - mean) * inv * A::value(weight, i)? + A::value(bias, i)?;
    }
    Some(())
}

fn softmax(values: &mut [f32]) {
    let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0;
    for value in &mut *values {
        *value = (*value - max).exp();
        sum += *value;
    }
    if sum != 0.0 {
        for value in &mut *values {
            *value /= sum;
        }
    }
}

struct Layer<'a> {
    ln1w: Bytes<'a>,
    ln1b: Bytes<'a>,
    sq: Bytes<'a>,
    sqb: Bytes<'a>,
    sk: Bytes<'a>,
    sv: Bytes<'a>,
    svb: Bytes<'a>,
    so: Bytes<'a>,
    sob: Bytes<'a>,
    lncw: Bytes<'a>,
    lncb: Bytes<'a>,
    cq: Bytes<'a>,
    cqb: Bytes<'a>,
    ck: Bytes<'a>,
    cv: Bytes<'a>,
    cvb: Bytes<'a>,
    co: Bytes<'a>,
    cob: Bytes<'a>,
    lnf_w: Bytes<'a>,
    lnf_b: Bytes<'a>,
    fc1: Bytes<'a>,
    fc1b: Bytes<'a>,
    fc2: Bytes<'a>,
    fc2b: Bytes<'a>,
}

fn block_bytes<'a>(
    model: &'a Data,
    block: usize,
    suffix: &[u8],
    dims: &[u64],
    kind: Option<SerializedType>,
) -> Result<Bytes<'a>, NativeError> {
    let mut n = [0_u8; 128];
    let used = name(block, suffix, &mut n).ok_or(NativeError::InvalidTensor)?;
    let view = tensor(model, &n[..used])?;
    match kind {
        Some(kind) => checked(view, dims, kind),
        None => checked_any(view, dims),
    }
}

fn layer<L: LinearRoute, A: AuxRoute>(
    model: &Data,
    block: usize,
) -> Result<Layer<'_>, NativeError> {
    let linear = Some(L::KIND);
    let aux = Some(A::KIND);
    Ok(Layer {
        ln1w: block_bytes(model, block, b"self_attn_layer_norm.weight", &[384], aux)?,
        ln1b: block_bytes(model, block, b"self_attn_layer_norm.bias", &[384], aux)?,
        sq: block_bytes(
            model,
            block,
            b"self_attn.q_proj.weight",
            &[384, 384],
            linear,
        )?,
        sqb: block_bytes(model, block, b"self_attn.q_proj.bias", &[384], aux)?,
        sk: block_bytes(
            model,
            block,
            b"self_attn.k_proj.weight",
            &[384, 384],
            linear,
        )?,
        sv: block_bytes(
            model,
            block,
            b"self_attn.v_proj.weight",
            &[384, 384],
            linear,
        )?,
        svb: block_bytes(model, block, b"self_attn.v_proj.bias", &[384], aux)?,
        so: block_bytes(
            model,
            block,
            b"self_attn.out_proj.weight",
            &[384, 384],
            linear,
        )?,
        sob: block_bytes(model, block, b"self_attn.out_proj.bias", &[384], aux)?,
        lncw: block_bytes(model, block, b"encoder_attn_layer_norm.weight", &[384], aux)?,
        lncb: block_bytes(model, block, b"encoder_attn_layer_norm.bias", &[384], aux)?,
        cq: block_bytes(
            model,
            block,
            b"encoder_attn.q_proj.weight",
            &[384, 384],
            linear,
        )?,
        cqb: block_bytes(model, block, b"encoder_attn.q_proj.bias", &[384], aux)?,
        ck: block_bytes(model, block, ENCODER_ATTN_K_SUFFIX, &[384, 384], linear)?,
        cv: block_bytes(model, block, ENCODER_ATTN_V_SUFFIX, &[384, 384], linear)?,
        cvb: block_bytes(model, block, ENCODER_ATTN_V_BIAS_SUFFIX, &[384], aux)?,
        co: block_bytes(
            model,
            block,
            b"encoder_attn.out_proj.weight",
            &[384, 384],
            linear,
        )?,
        cob: block_bytes(model, block, b"encoder_attn.out_proj.bias", &[384], aux)?,
        lnf_w: block_bytes(model, block, b"final_layer_norm.weight", &[384], aux)?,
        lnf_b: block_bytes(model, block, b"final_layer_norm.bias", &[384], aux)?,
        fc1: block_bytes(model, block, b"fc1.weight", &[384, 1536], linear)?,
        fc1b: block_bytes(model, block, b"fc1.bias", &[1536], aux)?,
        fc2: block_bytes(model, block, b"fc2.weight", &[1536, 384], linear)?,
        fc2b: block_bytes(model, block, b"fc2.bias", &[384], aux)?,
    })
}

fn layer_step<L: LinearRoute, A: AuxRoute>(
    layer: &Layer<'_>,
    pos: usize,
    frames: usize,
    hidden: &mut [f32],
    next: &mut [f32],
    q: &mut [f32],
    k: &mut [f32],
    v: &mut [f32],
    attn: &mut [f32],
    ff: &mut [f32],
    scores: &mut [f32],
    cache_k: &[f32],
    cache_v: &[f32],
) -> Option<()> {
    let current = &hidden[pos * E..pos * E + E];
    let current_next = &mut next[pos * E..pos * E + E];
    norm::<A>(current, layer.ln1w, layer.ln1b, &mut attn[..E])?;
    linear::<L, A>(layer.sq, Some(layer.sqb), &attn[..E], &mut q[..E], E, E)?;
    linear::<L, A>(
        layer.sk,
        None,
        &attn[..E],
        &mut k[pos * E..pos * E + E],
        E,
        E,
    )?;
    linear::<L, A>(
        layer.sv,
        Some(layer.svb),
        &attn[..E],
        &mut v[pos * E..pos * E + E],
        E,
        E,
    )?;
    let scale = 1.0 / HEAD_DIM_F32.sqrt();
    for h in 0..HEADS {
        for key in 0..=pos {
            let mut sum = 0.0;
            for d in 0..HEAD_DIM {
                sum += q[h * HEAD_DIM + d] * k[key * E + h * HEAD_DIM + d];
            }
            scores[key] = sum * scale;
        }
        softmax(&mut scores[..=pos]);
        for d in 0..HEAD_DIM {
            let mut sum = 0.0;
            for key in 0..=pos {
                sum += scores[key] * v[key * E + h * HEAD_DIM + d];
            }
            attn[h * HEAD_DIM + d] = sum;
        }
    }
    linear::<L, A>(layer.so, Some(layer.sob), &attn[..E], &mut q[..E], E, E)?;
    for d in 0..E {
        current_next[d] = current[d] + q[d];
    }
    norm::<A>(current_next, layer.lncw, layer.lncb, &mut attn[..E])?;
    linear::<L, A>(layer.cq, Some(layer.cqb), &attn[..E], &mut q[..E], E, E)?;
    for h in 0..HEADS {
        for key in 0..frames {
            let mut sum = 0.0;
            for d in 0..HEAD_DIM {
                sum += q[h * HEAD_DIM + d] * cache_k[key * E + h * HEAD_DIM + d];
            }
            scores[key] = sum * scale;
        }
        softmax(&mut scores[..frames]);
        for d in 0..HEAD_DIM {
            let mut sum = 0.0;
            for key in 0..frames {
                sum += scores[key] * cache_v[key * E + h * HEAD_DIM + d];
            }
            attn[h * HEAD_DIM + d] = sum;
        }
    }
    linear::<L, A>(layer.co, Some(layer.cob), &attn[..E], &mut q[..E], E, E)?;
    for d in 0..E {
        current_next[d] += q[d];
    }
    norm::<A>(current_next, layer.lnf_w, layer.lnf_b, &mut attn[..E])?;
    linear::<L, A>(
        layer.fc1,
        Some(layer.fc1b),
        &attn[..E],
        &mut ff[..FF],
        E,
        FF,
    )?;
    for x in &mut ff[..FF] {
        *x = 0.5 * *x * (1.0 + (0.797_884_6 * (*x + 0.044_715 * *x * *x * *x)).tanh());
    }
    linear::<L, A>(layer.fc2, Some(layer.fc2b), &ff[..FF], &mut q[..E], FF, E)?;
    for d in 0..E {
        current_next[d] += q[d];
    }
    Some(())
}

fn compute_decoder_logits<L: LinearRoute, A: AuxRoute>(
    embed: Bytes<'_>,
    positions: Bytes<'_>,
    final_w: Bytes<'_>,
    final_b: Bytes<'_>,
    layers: &[Layer<'_>; BLOCKS],
    frames: usize,
    cross: &[f32],
    tokens: &[i32],
    token_count: usize,
    hidden: &mut [f32],
    next: &mut [f32],
    q: &mut [f32],
    k: &mut [f32],
    v: &mut [f32],
    attn: &mut [f32],
    ff: &mut [f32],
    scores: &mut [f32],
    logits: &mut [f32],
    digest_out: &mut u64,
) -> Result<(), NativeError> {
    if token_count == 0 || token_count > SEQ || tokens.len() < token_count {
        return Err(NativeError::InvalidTensor);
    }
    for pos in 0..token_count {
        let token_id = usize::try_from(tokens[pos]).map_err(|_| NativeError::InvalidTensor)?;
        let dst = &mut hidden[pos * E..pos * E + E];
        for d in 0..E {
            dst[d] = Q8Linear::value(embed, token_id * E + d, E)
                .ok_or(NativeError::InvalidTensor)?
                + A::value(positions, pos * E + d).ok_or(NativeError::InvalidTensor)?;
        }
    }
    let cross_len = frames * E * 2;
    for (block, layer) in layers.iter().enumerate() {
        let cache = &cross[block * cross_len..(block + 1) * cross_len];
        let (cache_k, cache_v) = cache.split_at(frames * E);
        for pos in 0..token_count {
            layer_step::<L, A>(
                layer, pos, frames, hidden, next, q, k, v, attn, ff, scores, cache_k, cache_v,
            )
            .ok_or(NativeError::InvalidTensor)?;
            hidden[pos * E..pos * E + E].copy_from_slice(&next[pos * E..pos * E + E]);
        }
    }
    let last = &hidden[(token_count - 1) * E..token_count * E];
    norm::<A>(last, final_w, final_b, &mut attn[..E]).ok_or(NativeError::InvalidTensor)?;
    for id in 0..VOCAB {
        logits[id] = (0..E)
            .map(|d| L::value(embed, id * E + d, E).unwrap_or(0.0) * attn[d])
            .sum::<f32>();
    }
    let mut digest = FNV_OFFSET;
    for value in attn[..E].iter().copied() {
        digest ^= u64::from(value.to_bits());
        digest = digest.wrapping_mul(FNV_PRIME);
    }
    *digest_out = digest;
    Ok(())
}

fn select_greedy_timestamp_aware_token(
    policy: DecodePolicy,
    logits: &[f32],
    generated_tokens: &[i32],
    generated_token_count: usize,
    initial_token: bool,
    confidence_out: &mut f32,
) -> i32 {
    let last_was_timestamp = generated_token_count > 0
        && generated_tokens[generated_token_count - 1] >= policy.tokens.timestamp_begin;
    let penultimate_was_timestamp = generated_token_count < 2
        || generated_tokens[generated_token_count - 2] >= policy.tokens.timestamp_begin;
    let timestamp_begin = policy.tokens.timestamp_begin;

    let mut timestamp_max = f32::NEG_INFINITY;
    for token in timestamp_begin..i32::try_from(VOCAB).unwrap_or(i32::MAX) {
        let blocked_by_pair_rule = last_was_timestamp && penultimate_was_timestamp;
        let blocked_by_initial_limit = initial_token && token > timestamp_begin + 50;
        let score = if blocked_by_pair_rule || blocked_by_initial_limit {
            f32::NEG_INFINITY
        } else {
            logits[token as usize]
        };
        timestamp_max = timestamp_max.max(score);
    }

    let mut text_max = f32::NEG_INFINITY;
    for token in 0..timestamp_begin {
        let initial_suppressed =
            initial_token && (token == policy.tokens.eot || token == policy.tokens.space);
        let control_suppressed = token == policy.tokens.sot
            || token == policy.tokens.no_speech
            || token == policy.tokens.notimestamps
            || token == policy.tokens.translate
            || token == policy.tokens.transcribe;
        let blocked_by_pair_rule =
            last_was_timestamp && !penultimate_was_timestamp && token < policy.tokens.eot;
        let score = if initial_suppressed || control_suppressed || blocked_by_pair_rule {
            f32::NEG_INFINITY
        } else {
            logits[token as usize]
        };
        text_max = text_max.max(score);
    }

    let timestamp_sum = if timestamp_max > f32::NEG_INFINITY {
        let mut sum = 0.0;
        for token in timestamp_begin..i32::try_from(VOCAB).unwrap_or(i32::MAX) {
            let blocked_by_pair_rule = last_was_timestamp && penultimate_was_timestamp;
            let blocked_by_initial_limit = initial_token && token > timestamp_begin + 50;
            if !blocked_by_pair_rule && !blocked_by_initial_limit {
                sum += (logits[token as usize] - timestamp_max).exp();
            }
        }
        sum.ln() + timestamp_max
    } else {
        f32::NEG_INFINITY
    };
    let force_timestamp = timestamp_sum > text_max;

    let mut best_token = 0;
    let mut best_score = f32::NEG_INFINITY;
    for token in 0..i32::try_from(VOCAB).unwrap_or(i32::MAX) {
        let initial_suppressed =
            initial_token && (token == policy.tokens.eot || token == policy.tokens.space);
        let control_suppressed = token == policy.tokens.sot
            || token == policy.tokens.no_speech
            || token == policy.tokens.notimestamps
            || token == policy.tokens.translate
            || token == policy.tokens.transcribe;
        let timestamp_token = token >= timestamp_begin;
        let blocked_by_pair_rule = last_was_timestamp
            && ((penultimate_was_timestamp && timestamp_token)
                || (!penultimate_was_timestamp && token < policy.tokens.eot));
        let blocked_by_initial_limit =
            initial_token && timestamp_token && token > timestamp_begin + 50;
        let blocked_by_timestamp_mass = force_timestamp && !timestamp_token;
        let score = if initial_suppressed
            || control_suppressed
            || blocked_by_pair_rule
            || blocked_by_initial_limit
            || blocked_by_timestamp_mass
        {
            f32::NEG_INFINITY
        } else {
            logits[token as usize]
        };
        if score > best_score {
            best_score = score;
            best_token = token;
        }
    }
    *confidence_out = best_score;
    best_token
}

fn run_variant<L: LinearRoute, A: AuxRoute>(
    request: DecodeKernelRequest<'_>,
) -> Result<(), NativeError> {
    if usize::try_from(request.contract.vocab_size).ok() != Some(VOCAB)
        || usize::try_from(request.contract.embedding_length).ok() != Some(E)
        || usize::try_from(request.contract.decoder_block_count).ok() != Some(BLOCKS)
    {
        return Err(NativeError::InvalidTensor);
    }
    let frames =
        usize::try_from(request.encoder_frame_count).map_err(|_| NativeError::InvalidTensor)?;
    if request.encoder_state.len() < frames * E
        || request.generated_tokens.is_empty()
        || request.logits.len() < VOCAB
    {
        return Err(NativeError::InvalidTensor);
    }
    let embed = checked(
        tensor(request.model, b"model.decoder.embed_tokens.weight")?,
        &[384, 51865],
        SerializedType::Q8_0,
    )?;
    let positions = checked(
        tensor(request.model, b"model.decoder.embed_positions.weight")?,
        &[384, 448],
        A::KIND,
    )?;
    let final_w = checked(
        tensor(request.model, b"model.decoder.layer_norm.weight")?,
        &[384],
        A::KIND,
    )?;
    let final_b = checked(
        tensor(request.model, b"model.decoder.layer_norm.bias")?,
        &[384],
        A::KIND,
    )?;
    let layers = [
        layer::<L, A>(request.model, 0)?,
        layer::<L, A>(request.model, 1)?,
        layer::<L, A>(request.model, 2)?,
        layer::<L, A>(request.model, 3)?,
    ];
    let cross_len = frames * E * 2;
    let cross_total = cross_len * BLOCKS;
    let token_total = SEQ * E;
    let need = cross_total + token_total * 6 + FF + frames.max(SEQ);
    if request.workspace.len() < need {
        return Err(NativeError::InvalidTensor);
    }
    let (cross, rest) = request.workspace.split_at_mut(cross_total);
    let (token, rest) = rest.split_at_mut(token_total * 6);
    let (ff, rest) = rest.split_at_mut(FF);
    let (scores, _) = rest.split_at_mut(frames.max(SEQ));
    let (hidden, rest) = token.split_at_mut(token_total);
    let (next, rest) = rest.split_at_mut(token_total);
    let (q, rest) = rest.split_at_mut(token_total);
    let (k, rest) = rest.split_at_mut(token_total);
    let (v, attn) = rest.split_at_mut(token_total);
    for (block, layer) in layers.iter().enumerate() {
        let cache = &mut cross[block * cross_len..(block + 1) * cross_len];
        let (cache_k, cache_v) = cache.split_at_mut(frames * E);
        linear::<L, A>(layer.ck, None, request.encoder_state, cache_k, E, frames)
            .ok_or(NativeError::InvalidTensor)?;
        linear::<L, A>(
            layer.cv,
            Some(layer.cvb),
            request.encoder_state,
            cache_v,
            E,
            frames,
        )
        .ok_or(NativeError::InvalidTensor)?;
    }

    let p = request.policy;
    let prompt = [
        p.tokens.sot,
        p.tokens.language_en,
        p.tokens.transcribe,
        p.tokens.notimestamps,
    ];
    let mut tokens = [0_i32; SEQ];
    tokens[..prompt.len()].copy_from_slice(&prompt);
    let generation_limit = request.generated_tokens.len().min(SEQ - prompt.len());
    let mut token_count = prompt.len();
    let mut digest = FNV_OFFSET;
    for step in 0..generation_limit {
        compute_decoder_logits::<L, A>(
            embed,
            positions,
            final_w,
            final_b,
            &layers,
            frames,
            cross,
            &tokens,
            token_count,
            hidden,
            next,
            q,
            k,
            v,
            &mut attn[..E],
            ff,
            scores,
            request.logits,
            &mut digest,
        )?;
        let next_token = select_greedy_timestamp_aware_token(
            p,
            request.logits,
            request.generated_tokens,
            step,
            step == 0,
            request.confidence_out,
        );
        *request.token_out = next_token;
        request.generated_tokens[step] = next_token;
        tokens[token_count] = next_token;
        token_count += 1;
        if next_token == p.tokens.eot || (next_token >= p.tokens.timestamp_begin && step > 0) {
            break;
        }
    }
    *request.generated_token_count_out =
        i32::try_from(token_count - prompt.len()).map_err(|_| NativeError::InvalidTensor)?;
    *request.digest_out = digest;
    Ok(())
}

pub fn run_q8(request: DecodeKernelRequest<'_>) -> bool {
    request.variant == super::sm::DecodeVariant::Q8_0
        && run_variant::<Q8Linear, Q8Aux>(request).is_ok()
}

pub fn run_q8_f32_aux(request: DecodeKernelRequest<'_>) -> bool {
    request.variant == super::sm::DecodeVariant::Q8_0F32Aux
        && run_variant::<Q8Linear, F32Aux>(request).is_ok()
}

pub fn run_q4_0(request: DecodeKernelRequest<'_>) -> bool {
    request.variant == super::sm::DecodeVariant::Q4_0
        && run_variant::<Q40Linear, Q8Aux>(request).is_ok()
}

pub fn run_q4_1(request: DecodeKernelRequest<'_>) -> bool {
    request.variant == super::sm::DecodeVariant::Q4_1
        && run_variant::<Q41Linear, Q8Aux>(request).is_ok()
}

#[allow(dead_code)]
const fn _contract_type(_: &WhisperExecutionContract, _: DecodePolicy) {}
#[cfg(test)]
mod tests {
    use super::*;

    struct F32Linear;
    struct F32Bias;

    impl LinearRoute for F32Linear {
        const KIND: SerializedType = SerializedType::F32;

        fn value(bytes: Bytes<'_>, index: usize, _cols: usize) -> Option<f32> {
            f32_at(bytes.data, index)
        }
    }

    impl AuxRoute for F32Bias {
        const KIND: SerializedType = SerializedType::F32;

        fn value(bytes: Bytes<'_>, index: usize) -> Option<f32> {
            f32_at(bytes.data, index)
        }
    }

    fn raw_f32(values: &[f32]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|value| value.to_ne_bytes())
            .collect()
    }

    #[test]
    fn cross_cache_lookup_uses_encoder_k_weight_and_v_weight_bias() {
        let mut name_buf = [0_u8; 128];
        let used = name(2, ENCODER_ATTN_K_SUFFIX, &mut name_buf).unwrap();
        assert_eq!(
            &name_buf[..used],
            b"model.decoder.layers.2.encoder_attn.k_proj.weight"
        );
        let used = name(2, ENCODER_ATTN_V_SUFFIX, &mut name_buf).unwrap();
        assert_eq!(
            &name_buf[..used],
            b"model.decoder.layers.2.encoder_attn.v_proj.weight"
        );
        let used = name(2, ENCODER_ATTN_V_BIAS_SUFFIX, &mut name_buf).unwrap();
        assert_eq!(
            &name_buf[..used],
            b"model.decoder.layers.2.encoder_attn.v_proj.bias"
        );
    }

    #[test]
    fn self_attention_kv_receive_self_layer_normalized_input() {
        let zero_matrix_data = raw_f32(&vec![0.0; E * E]);
        let zero_fc1_data = raw_f32(&vec![0.0; E * FF]);
        let zero_fc2_data = raw_f32(&vec![0.0; E * FF]);
        let identity_matrix_data = {
            let mut values = vec![0.0; E * E];
            for i in 0..E {
                values[i * E + i] = 1.0;
            }
            raw_f32(&values)
        };
        let zero_vector_data = raw_f32(&vec![0.0; E]);
        let one_vector_data = raw_f32(&vec![1.0; E]);
        let zero_ff_data = raw_f32(&vec![0.0; FF]);
        let zero_matrix = Bytes {
            data: &zero_matrix_data,
            kind: SerializedType::F32,
        };
        let zero_fc1 = Bytes {
            data: &zero_fc1_data,
            kind: SerializedType::F32,
        };
        let zero_fc2 = Bytes {
            data: &zero_fc2_data,
            kind: SerializedType::F32,
        };
        let identity_matrix = Bytes {
            data: &identity_matrix_data,
            kind: SerializedType::F32,
        };
        let zero_vector = Bytes {
            data: &zero_vector_data,
            kind: SerializedType::F32,
        };
        let one_vector = Bytes {
            data: &one_vector_data,
            kind: SerializedType::F32,
        };
        let zero_ff = Bytes {
            data: &zero_ff_data,
            kind: SerializedType::F32,
        };
        let layer = Layer {
            ln1w: one_vector,
            ln1b: zero_vector,
            sq: zero_matrix,
            sqb: zero_vector,
            sk: identity_matrix,
            sv: identity_matrix,
            svb: zero_vector,
            so: zero_matrix,
            sob: zero_vector,
            lncw: one_vector,
            lncb: zero_vector,
            cq: zero_matrix,
            cqb: zero_vector,
            ck: zero_matrix,
            cv: zero_matrix,
            cvb: zero_vector,
            co: zero_matrix,
            cob: zero_vector,
            lnf_w: one_vector,
            lnf_b: zero_vector,
            fc1: zero_fc1,
            fc1b: zero_ff,
            fc2: zero_fc2,
            fc2b: zero_vector,
        };
        let mut hidden = vec![0.0; E];
        hidden[0] = 1.0;
        hidden[1] = 3.0;
        let mut next = vec![0.0; E];
        let mut q = vec![0.0; E];
        let mut k = vec![0.0; E];
        let mut v = vec![0.0; E];
        let mut attn = vec![0.0; E];
        let mut ff = vec![0.0; FF];
        let mut scores = vec![0.0; 1];
        let cache = vec![0.0; E];

        layer_step::<F32Linear, F32Bias>(
            &layer,
            0,
            1,
            &mut hidden,
            &mut next,
            &mut q,
            &mut k,
            &mut v,
            &mut attn,
            &mut ff,
            &mut scores,
            &cache,
            &cache,
        )
        .unwrap();

        let mean = hidden.iter().sum::<f32>() / E_F32;
        let variance = hidden
            .iter()
            .map(|x| {
                let d = *x - mean;
                d * d
            })
            .sum::<f32>()
            / E_F32;
        let inv = 1.0 / (variance + EPS).sqrt();
        assert!((k[0] - (1.0 - mean) * inv).abs() < 1.0e-6);
        assert!((k[1] - (3.0 - mean) * inv).abs() < 1.0e-6);
        assert!((v[0] - k[0]).abs() < 1.0e-6);
        assert!((v[1] - k[1]).abs() < 1.0e-6);
        assert!((k[0] - hidden[0]).abs() > 0.1);
    }

    fn assert_float_eq(actual: f32, expected: f32) {
        assert!((actual - expected).abs() < f32::EPSILON);
    }

    fn selector_logits(entries: &[(usize, f32)]) -> Vec<f32> {
        let mut logits = vec![f32::NEG_INFINITY; VOCAB];
        for &(token, value) in entries {
            logits[token] = value;
        }
        logits
    }

    #[test]
    fn selector_suppresses_timestamp_pairs_and_low_text_after_timestamp() {
        let policy = DecodePolicy::tiny_asr();
        let timestamp = policy.tokens.timestamp_begin as usize;
        let mut confidence = 0.0;
        let logits = selector_logits(&[(timestamp, 10.0), (timestamp + 1, 9.0), (40, 8.0)]);
        let generated = [timestamp as i32, timestamp as i32 + 1];
        let selected = select_greedy_timestamp_aware_token(
            policy,
            &logits,
            &generated,
            generated.len(),
            false,
            &mut confidence,
        );
        assert_eq!(selected, 40);
        assert_float_eq(confidence, 8.0);

        let logits = selector_logits(&[
            (timestamp, 10.0),
            (40, 9.0),
            (policy.tokens.eot as usize, 11.0),
        ]);
        let generated = [100_i32, timestamp as i32];
        let selected = select_greedy_timestamp_aware_token(
            policy,
            &logits,
            &generated,
            generated.len(),
            false,
            &mut confidence,
        );
        assert_eq!(selected, policy.tokens.eot);
        assert_float_eq(confidence, 11.0);
    }

    #[test]
    fn selector_limits_initial_timestamp_and_suppresses_controls() {
        let policy = DecodePolicy::tiny_asr();
        let timestamp = policy.tokens.timestamp_begin as usize;
        let mut confidence = 0.0;
        let logits = selector_logits(&[
            (timestamp + 51, 100.0),
            (timestamp + 50, 5.0),
            (policy.tokens.sot as usize, 90.0),
            (policy.tokens.translate as usize, 89.0),
            (policy.tokens.no_speech as usize, 88.0),
            (policy.tokens.transcribe as usize, 87.0),
            (policy.tokens.notimestamps as usize, 86.0),
            (100, 4.0),
        ]);
        let selected =
            select_greedy_timestamp_aware_token(policy, &logits, &[], 0, true, &mut confidence);
        assert_eq!(selected, (timestamp + 50) as i32);
        assert_float_eq(confidence, 5.0);
    }

    #[test]
    fn selector_forces_timestamp_when_timestamp_mass_exceeds_text_mass() {
        let policy = DecodePolicy::tiny_asr();
        let timestamp = policy.tokens.timestamp_begin as usize;
        let mut confidence = 0.0;
        let logits = selector_logits(&[(100, 2.0), (timestamp, 1.5), (timestamp + 1, 1.5)]);
        let selected =
            select_greedy_timestamp_aware_token(policy, &logits, &[], 0, false, &mut confidence);
        assert_eq!(selected, timestamp as i32);
        assert_float_eq(confidence, 1.5);
    }

    #[test]
    fn selector_suppresses_initial_eot_and_space_but_reports_raw_logit() {
        let policy = DecodePolicy::tiny_asr();
        let mut confidence = 0.0;
        let logits = selector_logits(&[
            (policy.tokens.eot as usize, 20.0),
            (policy.tokens.space as usize, 19.0),
            (42, 3.25),
        ]);
        let selected =
            select_greedy_timestamp_aware_token(policy, &logits, &[], 0, true, &mut confidence);
        assert_eq!(selected, 42);
        assert_float_eq(confidence, 3.25);
    }
}
