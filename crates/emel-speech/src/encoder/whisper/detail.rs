#![allow(
    clippy::cast_precision_loss,
    clippy::excessive_precision,
    clippy::needless_range_loop,
    clippy::similar_names,
    clippy::suboptimal_flops,
    clippy::too_many_arguments,
    clippy::too_many_lines,
    clippy::unreadable_literal,
    reason = "native kernels preserve source operation order and fixed workspace indexing"
)]

use emel_model::bridge::{Data, TensorView};
use emel_tensor::dtype::SerializedType;

use super::sm::{
    BLUESTEIN_FFT_SIZE, FFT_SIZE, WeightVariant, encoder_frame_count, mel_frame_count,
};

const FFT_BINS: usize = 201;
const HEADS: usize = 6;
const HEAD_DIM: usize = 64;
const EPSILON: f32 = 1.0e-5;
const LOG_ZERO: f32 = 1.0e-10;
const PI: f32 = core::f32::consts::PI;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NativeError {
    MissingTensor,
    InvalidTensor,
    UnsupportedDType,
}

fn tensor<'a>(model: &'a Data, name: &[u8]) -> Result<TensorView<'a>, NativeError> {
    model.tensor_named(name).ok_or(NativeError::MissingTensor)
}

fn bytes<'a>(
    tensor: TensorView<'a>,
    dims: &[u64],
    kind: SerializedType,
) -> Result<&'a [u8], NativeError> {
    let metadata = tensor.metadata().ok_or(NativeError::InvalidTensor)?;
    if metadata.tensor_type() != kind
        || usize::try_from(metadata.dimension_count()).ok() != Some(dims.len())
        || metadata.dimensions().get(..dims.len()) != Some(dims)
    {
        return Err(if metadata.tensor_type() == kind {
            NativeError::InvalidTensor
        } else {
            NativeError::UnsupportedDType
        });
    }
    tensor.bytes().ok_or(NativeError::InvalidTensor)
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

fn read_f32(data: &[u8], index: usize) -> Option<f32> {
    let start = index.checked_mul(4)?;
    let bytes = data.get(start..start + 4)?;
    Some(f32::from_ne_bytes(bytes.try_into().ok()?))
}

fn read_f16(data: &[u8], index: usize) -> Option<f32> {
    let start = index.checked_mul(2)?;
    let bytes = data.get(start..start + 2)?;
    Some(f16(u16::from_ne_bytes(bytes.try_into().ok()?)))
}

fn q8_value(data: &[u8], index: usize, cols: usize) -> Option<f32> {
    let row = index / cols;
    let col = index % cols;
    let row_bytes = 34usize.checked_mul(cols / 32)?;
    let start = row.checked_mul(row_bytes)?.checked_add(col / 32 * 34)?;
    let block = data.get(start..start + 34)?;
    let scale = f16(u16::from_ne_bytes(block[..2].try_into().ok()?));
    Some(scale * f32::from(i8::from_ne_bytes([block[2 + col % 32]])))
}

fn q4_value(data: &[u8], index: usize, cols: usize, one: bool) -> Option<f32> {
    let row = index / cols;
    let col = index % cols;
    let row_bytes = (if one { 20usize } else { 18usize }).checked_mul(cols / 32)?;
    let block_bytes = if one { 20usize } else { 18usize };
    let block_index = col / 32;
    let start = row
        .checked_mul(row_bytes)?
        .checked_add(block_index.checked_mul(block_bytes)?)?;
    let block = data.get(start..start + block_bytes)?;
    let scale = f16(u16::from_ne_bytes(block[..2].try_into().ok()?));
    let minimum = if one {
        f16(u16::from_ne_bytes(block[2..4].try_into().ok()?))
    } else {
        0.0
    };
    let packed = block[if one { 4 } else { 2 } + col % 16];
    let nibble = if col % 32 < 16 {
        packed & 0x0f
    } else {
        packed >> 4
    };
    Some(
        scale
            * if one {
                f32::from(nibble)
            } else {
                f32::from(nibble) - 8.0
            }
            + minimum,
    )
}

fn aux(data: &[u8], kind: SerializedType, index: usize) -> Option<f32> {
    match kind {
        SerializedType::F32 => read_f32(data, index),
        SerializedType::Q8_0 => q8_value(data, index, 384),
        _ => None,
    }
}

fn linear_value(variant: WeightVariant, data: &[u8], index: usize, cols: usize) -> Option<f32> {
    match variant {
        WeightVariant::Q8_0F32Aux | WeightVariant::Q8_0 => q8_value(data, index, cols),
        WeightVariant::Q4_0 => q4_value(data, index, cols, false),
        WeightVariant::Q4_1 => q4_value(data, index, cols, true),
        WeightVariant::Unsupported => None,
    }
}

fn gelu(x: f32) -> f32 {
    0.5 * x * (1.0 + (0.797_884_560_802_865_4 * (x + 0.044_715 * x * x * x)).tanh())
}

fn layer_norm(
    input: &[f32],
    weight: &[u8],
    bias: &[u8],
    aux_kind: SerializedType,
    output: &mut [f32],
) -> Option<()> {
    let mean = input.iter().sum::<f32>() / 384.0;
    let variance = input
        .iter()
        .map(|x| {
            let d = *x - mean;
            d * d
        })
        .sum::<f32>()
        / 384.0;
    let inv = 1.0 / (variance + EPSILON).sqrt();
    for i in 0..384 {
        output[i] = (input[i] - mean) * inv * aux(weight, aux_kind, i)? + aux(bias, aux_kind, i)?;
    }
    Some(())
}

fn linear(
    variant: WeightVariant,
    aux_kind: SerializedType,
    weight: &[u8],
    bias: &[u8],
    input: &[f32],
    output: &mut [f32],
    cols: usize,
    rows: usize,
) -> Option<()> {
    for row in 0..rows {
        let mut sum = 0.0;
        for col in 0..cols {
            sum += linear_value(variant, weight, row * cols + col, cols)? * input[col];
        }
        output[row] = sum + aux(bias, aux_kind, row)?;
    }
    Some(())
}

fn linear_no_bias(
    variant: WeightVariant,
    weight: &[u8],
    input: &[f32],
    output: &mut [f32],
    cols: usize,
    rows: usize,
) -> Option<()> {
    for row in 0..rows {
        let mut sum = 0.0;
        for col in 0..cols {
            sum += linear_value(variant, weight, row * cols + col, cols)? * input[col];
        }
        output[row] = sum;
    }
    Some(())
}

fn fft(real: &mut [f32], imag: &mut [f32], inverse: bool) {
    let n = real.len();
    let mut j = 0;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if i < j {
            real.swap(i, j);
            imag.swap(i, j);
        }
    }
    let mut len = 2;
    while len <= n {
        let sign = if inverse { 1.0 } else { -1.0 };
        let angle = sign * 2.0 * PI / len as f32;
        let wr = angle.cos();
        let wi = angle.sin();
        for offset in (0..n).step_by(len) {
            let mut r = 1.0;
            let mut im = 0.0;
            for i in 0..len / 2 {
                let even = offset + i;
                let odd = even + len / 2;
                let or = real[odd] * r - imag[odd] * im;
                let oi = real[odd] * im + imag[odd] * r;
                real[odd] = real[even] - or;
                imag[odd] = imag[even] - oi;
                real[even] += or;
                imag[even] += oi;
                let nr = r * wr - im * wi;
                im = r * wi + im * wr;
                r = nr;
            }
        }
        len <<= 1;
    }
    if inverse {
        let scale = 1.0 / n as f32;
        for i in 0..n {
            real[i] *= scale;
            imag[i] *= scale;
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn mel_features(
    pcm: &[f32],
    mel_filters: &[u8],
    mel_frames: usize,
    mel: &mut [f32],
    real: &mut [f32],
    imag: &mut [f32],
    kernel_real: &mut [f32],
    kernel_imag: &mut [f32],
    window: &mut [f32],
    chirp_real: &mut [f32],
    chirp_imag: &mut [f32],
) -> Option<()> {
    for n in 0..FFT_SIZE {
        window[n] = 0.5 - 0.5 * (2.0 * PI * n as f32 / FFT_SIZE as f32).cos();
        let angle = -PI * n as f32 * n as f32 / FFT_SIZE as f32;
        chirp_real[n] = angle.cos();
        chirp_imag[n] = angle.sin();
    }
    kernel_real.fill(0.0);
    kernel_imag.fill(0.0);
    for n in 0..FFT_SIZE {
        let angle = PI * n as f32 * n as f32 / FFT_SIZE as f32;
        let c = angle.cos();
        let s = angle.sin();
        kernel_real[n] = c;
        kernel_imag[n] = s;
        if n != 0 {
            kernel_real[BLUESTEIN_FFT_SIZE - n] += c;
            kernel_imag[BLUESTEIN_FFT_SIZE - n] += s;
        }
    }
    fft(kernel_real, kernel_imag, false);
    let mut frame = [0.0_f32; FFT_SIZE];
    let mut power = [0.0_f32; FFT_BINS];
    for fi in 0..mel_frames {
        let offset = fi * 160;
        let active = pcm.len() + FFT_SIZE / 2;
        for n in 0..FFT_SIZE {
            let pi = offset + n;
            let value = if pi < active {
                let source = pi.abs_diff(FFT_SIZE / 2);
                pcm.get(source).copied().unwrap_or(0.0)
            } else {
                0.0
            };
            frame[n] = value * window[n];
        }
        real.fill(0.0);
        imag.fill(0.0);
        for n in 0..FFT_SIZE {
            real[n] = frame[n] * chirp_real[n];
            imag[n] = frame[n] * chirp_imag[n];
        }
        fft(real, imag, false);
        for i in 0..BLUESTEIN_FFT_SIZE {
            let r = real[i] * kernel_real[i] - imag[i] * kernel_imag[i];
            let im = real[i] * kernel_imag[i] + imag[i] * kernel_real[i];
            real[i] = r;
            imag[i] = im;
        }
        fft(real, imag, true);
        for k in 0..FFT_BINS {
            let r = real[k] * chirp_real[k] - imag[k] * chirp_imag[k];
            let im = real[k] * chirp_imag[k] + imag[k] * chirp_real[k];
            power[k] = r * r + im * im;
        }
        for mb in 0..80 {
            let mut energy = 0.0;
            for (bin, power_value) in power.iter().enumerate().take(FFT_BINS) {
                energy += read_f32(mel_filters, bin + FFT_BINS * mb)? * *power_value;
            }
            mel[mb * mel_frames + fi] = energy.max(LOG_ZERO).log10();
        }
    }
    let max = mel.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let floor = max - 8.0;
    for x in mel {
        *x = (x.max(floor) + 4.0) * 0.25;
    }
    Some(())
}
fn conv1(
    mel: &[f32],
    frames: usize,
    weight: &[u8],
    bias: &[u8],
    aux_kind: SerializedType,
    output: &mut [f32],
) -> Option<()> {
    for out in 0..384 {
        let b = aux(bias, aux_kind, out)?;
        for f in 0..frames {
            let mut s = b;
            for input in 0..80 {
                let base = 3 * (input + 80 * out);
                if f > 0 {
                    s += mel[input * frames + f - 1] * read_f16(weight, base)?;
                }
                s += mel[input * frames + f] * read_f16(weight, base + 1)?;
                if f + 1 < frames {
                    s += mel[input * frames + f + 1] * read_f16(weight, base + 2)?;
                }
            }
            output[f * 384 + out] = gelu(s);
        }
    }
    Some(())
}
fn conv2(
    input: &[f32],
    mel_frames: usize,
    weight: &[u8],
    bias: &[u8],
    aux_kind: SerializedType,
    output: &mut [f32],
) -> Option<()> {
    let frames = mel_frames.div_ceil(2);
    for out in 0..384 {
        let b = aux(bias, aux_kind, out)?;
        for f in 0..frames {
            let center = f * 2;
            let mut s = b;
            for i in 0..384 {
                let base = 3 * (i + 384 * out);
                if center > 0 {
                    s += input[(center - 1) * 384 + i] * read_f16(weight, base)?;
                }
                if center < mel_frames {
                    s += input[center * 384 + i] * read_f16(weight, base + 1)?;
                }
                if center + 1 < mel_frames {
                    s += input[(center + 1) * 384 + i] * read_f16(weight, base + 2)?;
                }
            }
            output[f * 384 + out] = gelu(s);
        }
    }
    Some(())
}

#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
fn layer(
    model: &Data,
    variant: WeightVariant,
    aux_kind: SerializedType,
    layer: usize,
    frames: usize,
    hidden: &mut [f32],
    next: &mut [f32],
    q: &mut [f32],
    k: &mut [f32],
    v: &mut [f32],
    attn: &mut [f32],
    norm: &mut [f32],
    ff: &mut [f32],
    scores: &mut [f32],
) -> Result<(), NativeError> {
    let mut name = [0_u8; 96];
    let prefix = b"model.encoder.layers.";
    let mut make = |suffix: &[u8]| {
        let mut n = 0;
        name[..prefix.len()].copy_from_slice(prefix);
        n += prefix.len();
        let digit = u8::try_from(layer).ok()?;
        name[n] = b'0' + digit;
        n += 1;
        name[n] = b'.';
        n += 1;
        name[n..n + suffix.len()].copy_from_slice(suffix);
        n += suffix.len();
        tensor(model, &name[..n]).ok()
    };
    let ln1_weight = bytes(
        make(b"self_attn_layer_norm.weight").ok_or(NativeError::MissingTensor)?,
        &[384],
        aux_kind,
    )?;
    let ln1_bias = bytes(
        make(b"self_attn_layer_norm.bias").ok_or(NativeError::MissingTensor)?,
        &[384],
        aux_kind,
    )?;
    let qw = bytes(
        make(b"self_attn.q_proj.weight").ok_or(NativeError::MissingTensor)?,
        &[384, 384],
        variant.serialized(),
    )?;
    let qb = bytes(
        make(b"self_attn.q_proj.bias").ok_or(NativeError::MissingTensor)?,
        &[384],
        aux_kind,
    )?;
    let kw = bytes(
        make(b"self_attn.k_proj.weight").ok_or(NativeError::MissingTensor)?,
        &[384, 384],
        variant.serialized(),
    )?;
    let vw = bytes(
        make(b"self_attn.v_proj.weight").ok_or(NativeError::MissingTensor)?,
        &[384, 384],
        variant.serialized(),
    )?;
    let vb = bytes(
        make(b"self_attn.v_proj.bias").ok_or(NativeError::MissingTensor)?,
        &[384],
        aux_kind,
    )?;
    let ow = bytes(
        make(b"self_attn.out_proj.weight").ok_or(NativeError::MissingTensor)?,
        &[384, 384],
        variant.serialized(),
    )?;
    let ob = bytes(
        make(b"self_attn.out_proj.bias").ok_or(NativeError::MissingTensor)?,
        &[384],
        aux_kind,
    )?;
    let ln2_weight = bytes(
        make(b"final_layer_norm.weight").ok_or(NativeError::MissingTensor)?,
        &[384],
        aux_kind,
    )?;
    let ln2_bias = bytes(
        make(b"final_layer_norm.bias").ok_or(NativeError::MissingTensor)?,
        &[384],
        aux_kind,
    )?;
    let fc1_weight = bytes(
        make(b"fc1.weight").ok_or(NativeError::MissingTensor)?,
        &[384, 1536],
        variant.serialized(),
    )?;
    let fc1_bias = bytes(
        make(b"fc1.bias").ok_or(NativeError::MissingTensor)?,
        &[1536],
        aux_kind,
    )?;
    let fc2_weight = bytes(
        make(b"fc2.weight").ok_or(NativeError::MissingTensor)?,
        &[1536, 384],
        variant.serialized(),
    )?;
    let fc2_bias = bytes(
        make(b"fc2.bias").ok_or(NativeError::MissingTensor)?,
        &[384],
        aux_kind,
    )?;
    for f in 0..frames {
        layer_norm(
            &hidden[f * 384..f * 384 + 384],
            ln1_weight,
            ln1_bias,
            aux_kind,
            norm,
        )
        .ok_or(NativeError::InvalidTensor)?;
        linear(
            variant,
            aux_kind,
            qw,
            qb,
            norm,
            &mut q[f * 384..f * 384 + 384],
            384,
            384,
        )
        .ok_or(NativeError::InvalidTensor)?;
        linear_no_bias(variant, kw, norm, &mut k[f * 384..f * 384 + 384], 384, 384)
            .ok_or(NativeError::InvalidTensor)?;
        linear(
            variant,
            aux_kind,
            vw,
            vb,
            norm,
            &mut v[f * 384..f * 384 + 384],
            384,
            384,
        )
        .ok_or(NativeError::InvalidTensor)?;
    }
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();
    for f in 0..frames {
        attn[f * 384..f * 384 + 384].fill(0.0);
        for h in 0..HEADS {
            for key in 0..frames {
                let mut score = 0.0;
                for d in 0..HEAD_DIM {
                    score += q[f * 384 + h * 64 + d] * k[key * 384 + h * 64 + d];
                }
                scores[key] = score * scale;
            }
            let max = scores[..frames]
                .iter()
                .copied()
                .fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0;
            for x in &mut scores[..frames] {
                *x = (*x - max).exp();
                sum += *x;
            }
            for x in &mut scores[..frames] {
                *x /= sum;
            }
            for key in 0..frames {
                for d in 0..HEAD_DIM {
                    attn[f * 384 + h * 64 + d] += scores[key] * v[key * 384 + h * 64 + d];
                }
            }
        }
        linear(
            variant,
            aux_kind,
            ow,
            ob,
            &attn[f * 384..f * 384 + 384],
            norm,
            384,
            384,
        )
        .ok_or(NativeError::InvalidTensor)?;
        for d in 0..384 {
            next[f * 384 + d] = hidden[f * 384 + d] + norm[d];
        }
    }
    for f in 0..frames {
        layer_norm(
            &next[f * 384..f * 384 + 384],
            ln2_weight,
            ln2_bias,
            aux_kind,
            norm,
        )
        .ok_or(NativeError::InvalidTensor)?;
        linear(variant, aux_kind, fc1_weight, fc1_bias, norm, ff, 384, 1536)
            .ok_or(NativeError::InvalidTensor)?;
        for x in ff.iter_mut().take(1536) {
            *x = gelu(*x);
        }
        linear(variant, aux_kind, fc2_weight, fc2_bias, ff, norm, 1536, 384)
            .ok_or(NativeError::InvalidTensor)?;
        for d in 0..384 {
            hidden[f * 384 + d] = next[f * 384 + d] + norm[d];
        }
    }
    Ok(())
}

impl WeightVariant {
    const fn serialized(self) -> SerializedType {
        match self {
            Self::Q8_0F32Aux | Self::Q8_0 => SerializedType::Q8_0,
            Self::Q4_0 => SerializedType::Q4_0,
            Self::Q4_1 => SerializedType::Q4_1,
            Self::Unsupported => SerializedType::F32,
        }
    }
}
#[allow(clippy::too_many_lines)]
pub fn run(
    model: &Data,
    variant: WeightVariant,
    pcm: &[f32],
    workspace: &mut [f32],
    output: &mut [f32],
) -> Result<(i32, u64), NativeError> {
    let mel_frames = mel_frame_count(pcm.len());
    let frames = encoder_frame_count(pcm.len());
    let mel_bytes = bytes(
        tensor(model, b"mel_filters")?,
        &[201, 80],
        SerializedType::F32,
    )?;
    let aux_kind = if variant == WeightVariant::Q8_0F32Aux {
        SerializedType::F32
    } else {
        SerializedType::Q8_0
    };
    let conv1_weight = bytes(
        tensor(model, b"model.encoder.conv1.weight")?,
        &[3, 80, 384],
        SerializedType::F16,
    )?;
    let conv1_bias = bytes(
        tensor(model, b"model.encoder.conv1.bias")?,
        &[384],
        aux_kind,
    )?;
    let conv2_weight = bytes(
        tensor(model, b"model.encoder.conv2.weight")?,
        &[3, 384, 384],
        SerializedType::F16,
    )?;
    let conv2_bias = bytes(
        tensor(model, b"model.encoder.conv2.bias")?,
        &[384],
        aux_kind,
    )?;
    let pos = bytes(
        tensor(model, b"model.encoder.embed_positions.weight")?,
        &[384, 1500],
        aux_kind,
    )?;
    let final_weight = bytes(
        tensor(model, b"model.encoder.layer_norm.weight")?,
        &[384],
        aux_kind,
    )?;
    let final_bias = bytes(
        tensor(model, b"model.encoder.layer_norm.bias")?,
        &[384],
        aux_kind,
    )?;
    let (mel, rest) = workspace.split_at_mut(80 * mel_frames);
    let (conv1_out, rest) = rest.split_at_mut(384 * mel_frames);
    let (hidden, rest) = rest.split_at_mut(384 * frames);
    let (next, rest) = rest.split_at_mut(384 * frames);
    let (q, rest) = rest.split_at_mut(384 * frames);
    let (k, rest) = rest.split_at_mut(384 * frames);
    let (v, rest) = rest.split_at_mut(384 * frames);
    let (attn, rest) = rest.split_at_mut(384 * frames);
    let (norm, rest) = rest.split_at_mut(384);
    let (ff, rest) = rest.split_at_mut(1536);
    let (scores, rest) = rest.split_at_mut(frames);
    let (real, rest) = rest.split_at_mut(BLUESTEIN_FFT_SIZE);
    let (imag, rest) = rest.split_at_mut(BLUESTEIN_FFT_SIZE);
    let (kr, rest) = rest.split_at_mut(BLUESTEIN_FFT_SIZE);
    let (ki, rest) = rest.split_at_mut(BLUESTEIN_FFT_SIZE);
    let (window, rest) = rest.split_at_mut(FFT_SIZE);
    let (cr, ci) = rest.split_at_mut(FFT_SIZE);
    mel_features(
        pcm, mel_bytes, mel_frames, mel, real, imag, kr, ki, window, cr, ci,
    )
    .ok_or(NativeError::InvalidTensor)?;
    conv1(
        mel,
        mel_frames,
        conv1_weight,
        conv1_bias,
        aux_kind,
        conv1_out,
    )
    .ok_or(NativeError::InvalidTensor)?;
    conv2(
        conv1_out,
        mel_frames,
        conv2_weight,
        conv2_bias,
        aux_kind,
        hidden,
    )
    .ok_or(NativeError::InvalidTensor)?;
    for f in 0..frames {
        for d in 0..384 {
            hidden[f * 384 + d] +=
                aux(pos, aux_kind, f * 384 + d).ok_or(NativeError::InvalidTensor)?;
        }
    }
    for l in 0..4 {
        layer(
            model, variant, aux_kind, l, frames, hidden, next, q, k, v, attn, norm, ff, scores,
        )?;
    }
    for f in 0..frames {
        layer_norm(
            &hidden[f * 384..f * 384 + 384],
            final_weight,
            final_bias,
            aux_kind,
            &mut output[f * 384..f * 384 + 384],
        )
        .ok_or(NativeError::InvalidTensor)?;
    }
    let mut digest = 1_469_598_103_934_665_603_u64;
    for x in output.iter().take(frames * 384) {
        digest ^= u64::from(x.to_bits());
        digest = digest.wrapping_mul(1_099_511_628_211);
    }
    Ok((
        i32::try_from(frames).map_err(|_| NativeError::InvalidTensor)?,
        digest,
    ))
}
