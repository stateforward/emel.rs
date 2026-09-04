//! Model-owned immutable Whisper metadata and tensor contract.

use crate::data::{Data, TensorInput, TensorView};
use emel_tensor::dtype::SerializedType;

pub const ARCHITECTURE_NAME: &[u8] = b"whisper";
pub const SAMPLE_RATE: i32 = 16_000;
pub const N_MELS: u32 = 80;
pub const N_VOCAB: u32 = 51_865;
pub const N_EMBD: u32 = 384;
pub const N_FF: u32 = 1_536;
pub const N_HEAD: u32 = 6;
pub const N_HEAD_KV: u32 = 6;
pub const ENCODER_CONTEXT_LENGTH: u32 = 1_500;
pub const N_CTX: u32 = 448;
pub const ENCODER_BLOCK_COUNT: u32 = 4;
pub const DECODER_BLOCK_COUNT: u32 = 4;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum WhisperHParamsError {
    InvalidNMels,
    InvalidNVocab,
    InvalidEmbedding,
    InvalidFeedForward,
    InvalidHeads,
    InvalidContext,
    InvalidEncoderBlocks,
    InvalidDecoderBlocks,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct WhisperHParamsInput {
    pub n_mels: u32,
    pub n_vocab: u32,
    pub n_embd: u32,
    pub n_ff: u32,
    pub n_head: u32,
    pub n_head_kv: u32,
    pub n_ctx: u32,
    pub encoder_block_count: u32,
    pub decoder_block_count: u32,
}

impl Default for WhisperHParamsInput {
    fn default() -> Self {
        Self {
            n_mels: N_MELS,
            n_vocab: N_VOCAB,
            n_embd: N_EMBD,
            n_ff: N_FF,
            n_head: N_HEAD,
            n_head_kv: N_HEAD_KV,
            n_ctx: N_CTX,
            encoder_block_count: ENCODER_BLOCK_COUNT,
            decoder_block_count: DECODER_BLOCK_COUNT,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct WhisperHParams(WhisperHParamsInput);

impl WhisperHParams {
    /// Validates fixed Whisper model dimensions.
    ///
    /// # Errors
    ///
    /// Returns the specific dimension error for any mismatched hparam.
    pub const fn try_new(input: WhisperHParamsInput) -> Result<Self, WhisperHParamsError> {
        if input.n_mels != N_MELS {
            return Err(WhisperHParamsError::InvalidNMels);
        }
        if input.n_vocab != N_VOCAB {
            return Err(WhisperHParamsError::InvalidNVocab);
        }
        if input.n_embd != N_EMBD {
            return Err(WhisperHParamsError::InvalidEmbedding);
        }
        if input.n_ff != N_FF {
            return Err(WhisperHParamsError::InvalidFeedForward);
        }
        if input.n_head != N_HEAD || input.n_head_kv != N_HEAD_KV {
            return Err(WhisperHParamsError::InvalidHeads);
        }
        if input.n_ctx != N_CTX {
            return Err(WhisperHParamsError::InvalidContext);
        }
        if input.encoder_block_count != ENCODER_BLOCK_COUNT {
            return Err(WhisperHParamsError::InvalidEncoderBlocks);
        }
        if input.decoder_block_count != DECODER_BLOCK_COUNT {
            return Err(WhisperHParamsError::InvalidDecoderBlocks);
        }
        Ok(Self(input))
    }
    pub const fn n_mels(self) -> u32 {
        self.0.n_mels
    }
    pub const fn n_vocab(self) -> u32 {
        self.0.n_vocab
    }
    pub const fn n_embd(self) -> u32 {
        self.0.n_embd
    }
    pub const fn n_ff(self) -> u32 {
        self.0.n_ff
    }
    pub const fn n_head(self) -> u32 {
        self.0.n_head
    }
    pub const fn n_head_kv(self) -> u32 {
        self.0.n_head_kv
    }
    pub const fn n_ctx(self) -> u32 {
        self.0.n_ctx
    }
    pub const fn encoder_block_count(self) -> u32 {
        self.0.encoder_block_count
    }
    pub const fn decoder_block_count(self) -> u32 {
        self.0.decoder_block_count
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum LegacyError {
    NotLegacy,
    Truncated,
    InvalidHParams,
    InvalidMel,
    InvalidVocabulary,
    InvalidTensorHeader,
    UnknownTensorName,
    InvalidDimensions,
    UnsupportedDType,
    InvalidTensorSize,
    Capacity,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum WhisperError {
    WrongArchitecture,
    MissingHParams,
    InvalidHParams(WhisperHParamsError),
    MissingTensor,
    WrongShape,
    WrongDType,
    NonResident,
    InvalidTensor,
    MissingFamily(WhisperFamily),
    MissingEncoderBlock(u32),
    MissingDecoderBlock(u32),
    TooManyTensors,
    NameCapacity,
    Capacity,
    Legacy(LegacyError),
}
pub type WhisperDataError = WhisperError;

#[derive(Clone, Copy, Debug)]
pub struct WhisperDataInput<'a> {
    pub architecture: &'a [u8],
    pub hparams: WhisperHParamsInput,
    pub tensors: &'a [TensorInput<'a>],
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum WhisperFamily {
    Encoder,
    Decoder,
}

#[derive(Clone, Copy, Debug)]
pub struct WhisperFamilyView<'a> {
    data: &'a Data,
    family: WhisperFamily,
}
impl<'a> WhisperFamilyView<'a> {
    pub const fn family(self) -> WhisperFamily {
        self.family
    }
    pub fn tensor_count(self) -> u32 {
        self.data.whisper_family_count(self.family)
    }
    pub fn first_tensor(self) -> Option<TensorView<'a>> {
        self.data.whisper_family_tensor(self.family, 0)
    }
    pub fn tensor(self, index: u32) -> Option<TensorView<'a>> {
        self.data.whisper_family_tensor(self.family, index)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct WhisperBindingInput<'a> {
    pub(crate) data: &'a Data,
}
impl<'a> WhisperBindingInput<'a> {
    pub fn architecture(self) -> &'a [u8] {
        self.data.architecture_name()
    }
    pub const fn hparams(self) -> Option<&'a WhisperHParams> {
        self.data.whisper_hparams()
    }
    pub fn tensor_count(self) -> u32 {
        self.data.whisper_tensor_count()
    }
    pub fn tensor_named(self, name: &[u8]) -> Option<TensorView<'a>> {
        self.data.whisper_tensor_named(name)
    }
    pub fn tensor(self, index: u32) -> Option<TensorView<'a>> {
        self.data.whisper_tensor(index)
    }
    pub const fn encoder(self) -> WhisperFamilyView<'a> {
        self.family(WhisperFamily::Encoder)
    }
    pub const fn decoder(self) -> WhisperFamilyView<'a> {
        self.family(WhisperFamily::Decoder)
    }
    pub const fn family(self, family: WhisperFamily) -> WhisperFamilyView<'a> {
        WhisperFamilyView {
            data: self.data,
            family,
        }
    }
    pub fn block_count(self, family: WhisperFamily) -> u32 {
        self.data.whisper_block_count(family)
    }
}

/// Immutable execution data exposed to speech-owned model callers.
#[derive(Clone, Copy, Debug)]
pub struct ExecutionContract<'a> {
    model: WhisperBindingInput<'a>,
    sample_rate: i32,
    mel_bin_count: i32,
    vocab_size: i32,
    embedding_length: i32,
    feed_forward_length: i32,
    attention_head_count: i32,
    encoder_context_length: i32,
    decoder_context_length: i32,
    encoder_block_count: i32,
    decoder_block_count: i32,
    mel_filters: TensorView<'a>,
    encoder: WhisperFamilyView<'a>,
    decoder: WhisperFamilyView<'a>,
}

impl<'a> ExecutionContract<'a> {
    pub const fn model(self) -> WhisperBindingInput<'a> {
        self.model
    }
    pub const fn sample_rate(self) -> i32 {
        self.sample_rate
    }
    pub const fn mel_bin_count(self) -> i32 {
        self.mel_bin_count
    }
    pub const fn vocab_size(self) -> i32 {
        self.vocab_size
    }
    pub const fn embedding_length(self) -> i32 {
        self.embedding_length
    }
    pub const fn feed_forward_length(self) -> i32 {
        self.feed_forward_length
    }
    pub const fn attention_head_count(self) -> i32 {
        self.attention_head_count
    }
    pub const fn encoder_context_length(self) -> i32 {
        self.encoder_context_length
    }
    pub const fn decoder_context_length(self) -> i32 {
        self.decoder_context_length
    }
    pub const fn encoder_block_count(self) -> i32 {
        self.encoder_block_count
    }
    pub const fn decoder_block_count(self) -> i32 {
        self.decoder_block_count
    }
    pub const fn mel_filters(self) -> TensorView<'a> {
        self.mel_filters
    }
    pub const fn encoder(self) -> WhisperFamilyView<'a> {
        self.encoder
    }
    pub const fn decoder(self) -> WhisperFamilyView<'a> {
        self.decoder
    }
}

#[allow(clippy::too_many_lines)]
pub(crate) fn validate_input(
    input: WhisperDataInput<'_>,
    hparams: WhisperHParams,
) -> Result<(), WhisperError> {
    if input.architecture != ARCHITECTURE_NAME {
        return Err(WhisperError::WrongArchitecture);
    }
    if input.tensors.len() > crate::data::MAX_TENSORS {
        return Err(WhisperError::TooManyTensors);
    }
    for tensor in input.tensors {
        let metadata = tensor.metadata();
        if metadata.dimension_count() == 0 || metadata.data_size() == 0 {
            return Err(WhisperError::InvalidTensor);
        }
        if metadata.storage().is_some() && tensor.bytes().is_none() {
            return Err(WhisperError::NonResident);
        }
    }
    let require =
        |name: &[u8], dims: &[u64], kinds: &[SerializedType]| -> Result<(), WhisperError> {
            let tensor = input
                .tensors
                .iter()
                .copied()
                .find(|tensor| tensor.name() == name)
                .ok_or(WhisperError::MissingTensor)?;
            let metadata = tensor.metadata();
            let bytes = tensor.bytes().ok_or(WhisperError::NonResident)?;
            if metadata.dimension_count() as usize != dims.len()
                || metadata.dimensions()[..dims.len()] != *dims
            {
                return Err(WhisperError::WrongShape);
            }
            if !kinds.contains(&metadata.tensor_type()) {
                return Err(WhisperError::WrongDType);
            }
            if bytes.len() as u64 != metadata.data_size() {
                return Err(WhisperError::NonResident);
            }
            Ok(())
        };
    let f32_only = [SerializedType::F32];
    let f16_only = [SerializedType::F16];
    let q8 = [SerializedType::Q8_0];
    let q8_or_f32 = [SerializedType::Q8_0, SerializedType::F32];
    require(b"mel_filters", &[201, 80], &f32_only)?;
    require(b"model.encoder.conv1.weight", &[3, 80, 384], &f16_only)?;
    require(b"model.encoder.conv1.bias", &[384], &q8_or_f32)?;
    require(b"model.encoder.conv2.weight", &[3, 384, 384], &f16_only)?;
    require(b"model.encoder.conv2.bias", &[384], &q8_or_f32)?;
    require(
        b"model.encoder.embed_positions.weight",
        &[384, 1500],
        &q8_or_f32,
    )?;
    require(b"model.encoder.layer_norm.weight", &[384], &q8_or_f32)?;
    require(b"model.encoder.layer_norm.bias", &[384], &q8_or_f32)?;
    require(b"model.decoder.embed_tokens.weight", &[384, 51865], &q8)?;
    require(
        b"model.decoder.embed_positions.weight",
        &[384, 448],
        &q8_or_f32,
    )?;
    require(b"model.decoder.layer_norm.weight", &[384], &q8_or_f32)?;
    require(b"model.decoder.layer_norm.bias", &[384], &q8_or_f32)?;
    let linear = [
        SerializedType::Q8_0,
        SerializedType::Q4_0,
        SerializedType::Q4_1,
    ];
    for block in 0..hparams.encoder_block_count() {
        let mut name = [0_u8; 96];
        let used = decimal_name(
            &mut name,
            b"model.encoder.layers.",
            block as usize,
            b".self_attn.q_proj.weight",
        )?;
        require(&name[..used], &[384, 384], &linear)?;
        let used = decimal_name(
            &mut name,
            b"model.encoder.layers.",
            block as usize,
            b".self_attn.q_proj.bias",
        )?;
        require(&name[..used], &[384], &q8_or_f32)?;
    }
    for block in 0..hparams.decoder_block_count() {
        let mut name = [0_u8; 96];
        let used = decimal_name(
            &mut name,
            b"model.decoder.layers.",
            block as usize,
            b".self_attn.q_proj.weight",
        )?;
        require(&name[..used], &[384, 384], &linear)?;
        let used = decimal_name(
            &mut name,
            b"model.decoder.layers.",
            block as usize,
            b".self_attn.q_proj.bias",
        )?;
        require(&name[..used], &[384], &q8_or_f32)?;
        let used = decimal_name(
            &mut name,
            b"model.decoder.layers.",
            block as usize,
            b".encoder_attn.q_proj.weight",
        )?;
        require(&name[..used], &[384, 384], &linear)?;
        let used = decimal_name(
            &mut name,
            b"model.decoder.layers.",
            block as usize,
            b".encoder_attn.q_proj.bias",
        )?;
        require(&name[..used], &[384], &q8_or_f32)?;
    }
    Ok(())
}

fn decimal_name(
    buffer: &mut [u8],
    prefix: &[u8],
    number: usize,
    suffix: &[u8],
) -> Result<usize, WhisperError> {
    let mut used = prefix.len();
    if used > buffer.len() {
        return Err(WhisperError::Capacity);
    }
    buffer[..used].copy_from_slice(prefix);
    let mut digits = [0_u8; 20];
    let mut count = 0;
    let mut value = number;
    loop {
        if count == digits.len() {
            return Err(WhisperError::Capacity);
        }
        digits[count] = b'0' + u8::try_from(value % 10).unwrap_or(0);
        count += 1;
        value /= 10;
        if value == 0 {
            break;
        }
    }
    while count != 0 {
        count -= 1;
        buffer[used] = digits[count];
        used += 1;
    }
    let end = used
        .checked_add(suffix.len())
        .ok_or(WhisperError::Capacity)?;
    if end > buffer.len() {
        return Err(WhisperError::Capacity);
    }
    buffer[used..end].copy_from_slice(suffix);
    Ok(end)
}

#[must_use]
pub fn is_execution_architecture(name: &[u8]) -> bool {
    name == ARCHITECTURE_NAME
}

#[must_use]
pub fn is_legacy_lmgg_whisper(source: &[u8]) -> bool {
    source.get(..4) == Some(b"lmgg")
}

struct LegacyReader<'a> {
    bytes: &'a [u8],
    offset: usize,
}
impl<'a> LegacyReader<'a> {
    fn read(&mut self, size: usize) -> Result<&'a [u8], LegacyError> {
        let end = self.offset.checked_add(size).ok_or(LegacyError::Capacity)?;
        let bytes = self
            .bytes
            .get(self.offset..end)
            .ok_or(LegacyError::Truncated)?;
        self.offset = end;
        Ok(bytes)
    }
    fn i32(&mut self) -> Result<i32, LegacyError> {
        self.read(4)?
            .try_into()
            .map(i32::from_le_bytes)
            .map_err(|_| LegacyError::Truncated)
    }
    fn u32(&mut self) -> Result<u32, LegacyError> {
        self.read(4)?
            .try_into()
            .map(u32::from_le_bytes)
            .map_err(|_| LegacyError::Truncated)
    }
}

struct LegacyTensor<'a> {
    name: Vec<u8>,
    dims: [u64; 4],
    n_dims: u32,
    dtype: u32,
    data: &'a [u8],
}
fn checked_product(values: &[u64]) -> Result<u64, LegacyError> {
    values.iter().try_fold(1_u64, |total, value| {
        total.checked_mul(*value).ok_or(LegacyError::Capacity)
    })
}
fn legacy_data_size(dtype: u32, dims: &[u64]) -> Result<usize, LegacyError> {
    let elements = checked_product(dims)?;
    let bytes = match dtype {
        0 => elements.checked_mul(4),
        1 => elements.checked_mul(2),
        8 if dims.first().is_some_and(|value| value % 32 == 0) => elements
            .checked_div(32)
            .and_then(|value| value.checked_mul(34)),
        _ => return Err(LegacyError::UnsupportedDType),
    }
    .ok_or(LegacyError::InvalidTensorSize)?;
    usize::try_from(bytes).map_err(|_| LegacyError::Capacity)
}
const fn canonical_attention_suffix(name: &[u8]) -> Option<&'static [u8]> {
    match name {
        b"query.weight" => Some(b"q_proj.weight"),
        b"query.bias" => Some(b"q_proj.bias"),
        b"key.weight" => Some(b"k_proj.weight"),
        b"value.weight" => Some(b"v_proj.weight"),
        b"value.bias" => Some(b"v_proj.bias"),
        b"out.weight" => Some(b"out_proj.weight"),
        b"out.bias" => Some(b"out_proj.bias"),
        _ => None,
    }
}
const fn canonical_encoder_suffix(name: &[u8]) -> Option<&'static [u8]> {
    match name {
        b"attn_ln.weight" => Some(b"self_attn_layer_norm.weight"),
        b"attn_ln.bias" => Some(b"self_attn_layer_norm.bias"),
        b"attn.query.weight" => Some(b"self_attn.q_proj.weight"),
        b"attn.query.bias" => Some(b"self_attn.q_proj.bias"),
        b"attn.key.weight" => Some(b"self_attn.k_proj.weight"),
        b"attn.value.weight" => Some(b"self_attn.v_proj.weight"),
        b"attn.value.bias" => Some(b"self_attn.v_proj.bias"),
        b"attn.out.weight" => Some(b"self_attn.out_proj.weight"),
        b"attn.out.bias" => Some(b"self_attn.out_proj.bias"),
        b"mlp_ln.weight" => Some(b"final_layer_norm.weight"),
        b"mlp_ln.bias" => Some(b"final_layer_norm.bias"),
        b"mlp.0.weight" => Some(b"fc1.weight"),
        b"mlp.0.bias" => Some(b"fc1.bias"),
        b"mlp.2.weight" => Some(b"fc2.weight"),
        b"mlp.2.bias" => Some(b"fc2.bias"),
        _ => None,
    }
}
fn canonical_decoder_suffix(name: &[u8]) -> Option<Vec<u8>> {
    if let Some(suffix) = name.strip_prefix(b"cross_attn_ln.") {
        return match suffix {
            b"weight" => Some(b"encoder_attn_layer_norm.weight".to_vec()),
            b"bias" => Some(b"encoder_attn_layer_norm.bias".to_vec()),
            _ => None,
        };
    }
    canonical_encoder_suffix(name).map(<[u8]>::to_vec)
}
fn canonical_name(name: &[u8]) -> Option<Vec<u8>> {
    let fixed = match name {
        b"encoder.positional_embedding" => Some(b"model.encoder.embed_positions.weight".as_slice()),
        b"encoder.conv1.weight"
        | b"encoder.conv1.bias"
        | b"encoder.conv2.weight"
        | b"encoder.conv2.bias" => {
            let mut output = b"model.".to_vec();
            output.extend_from_slice(name);
            return Some(output);
        }
        b"encoder.ln_post.weight" => Some(b"model.encoder.layer_norm.weight".as_slice()),
        b"encoder.ln_post.bias" => Some(b"model.encoder.layer_norm.bias".as_slice()),
        b"decoder.positional_embedding" => Some(b"model.decoder.embed_positions.weight".as_slice()),
        b"decoder.token_embedding.weight" => Some(b"model.decoder.embed_tokens.weight".as_slice()),
        b"decoder.ln.weight" => Some(b"model.decoder.layer_norm.weight".as_slice()),
        b"decoder.ln.bias" => Some(b"model.decoder.layer_norm.bias".as_slice()),
        _ => None,
    };
    if let Some(value) = fixed {
        return Some(value.to_vec());
    }
    for block in 0..ENCODER_BLOCK_COUNT {
        let encoder = format!("encoder.blocks.{block}.");
        let decoder = format!("decoder.blocks.{block}.");
        let cross = format!("decoder.blocks.{block}.cross_attn.");
        if let Some(suffix) = name
            .strip_prefix(encoder.as_bytes())
            .and_then(canonical_encoder_suffix)
        {
            return Some(
                format!(
                    "model.encoder.layers.{block}.{}",
                    String::from_utf8_lossy(suffix)
                )
                .into_bytes(),
            );
        }
        if let Some(suffix) = name
            .strip_prefix(cross.as_bytes())
            .and_then(canonical_attention_suffix)
        {
            return Some(
                format!(
                    "model.decoder.layers.{block}.encoder_attn.{}",
                    String::from_utf8_lossy(suffix)
                )
                .into_bytes(),
            );
        }
        if let Some(suffix) = name
            .strip_prefix(decoder.as_bytes())
            .and_then(canonical_decoder_suffix)
        {
            return Some(
                format!(
                    "model.decoder.layers.{block}.{}",
                    String::from_utf8_lossy(&suffix)
                )
                .into_bytes(),
            );
        }
    }
    None
}

#[allow(clippy::too_many_lines)]
fn parse_legacy(source: &[u8]) -> Result<(Vec<LegacyTensor<'_>>, i32, i32), LegacyError> {
    let mut reader = LegacyReader {
        bytes: source,
        offset: 0,
    };
    if reader.read(4)? != b"lmgg" {
        return Err(LegacyError::NotLegacy);
    }
    let mut hparams = [0_i32; 11];
    for value in &mut hparams {
        *value = reader.i32()?;
    }
    let expected = [
        N_VOCAB,
        ENCODER_CONTEXT_LENGTH,
        N_EMBD,
        N_HEAD,
        ENCODER_BLOCK_COUNT,
        N_CTX,
        N_EMBD,
        N_HEAD,
        DECODER_BLOCK_COUNT,
        N_MELS,
    ];
    if hparams[..10]
        .iter()
        .zip(expected)
        .any(|(&actual, expected)| u32::try_from(actual).ok() != Some(expected))
    {
        return Err(LegacyError::InvalidHParams);
    }
    let n_mel = reader.i32()?;
    let n_fft = reader.i32()?;
    if n_mel != i32::try_from(N_MELS).unwrap_or(i32::MAX) || n_fft <= 0 {
        return Err(LegacyError::InvalidMel);
    }
    let mel_size = i64::from(n_mel)
        .checked_mul(i64::from(n_fft))
        .and_then(|value| value.checked_mul(4))
        .and_then(|value| usize::try_from(value).ok())
        .ok_or(LegacyError::Capacity)?;
    let mel_data = reader.read(mel_size)?;
    let mut tensors = vec![LegacyTensor {
        name: b"mel_filters".to_vec(),
        dims: [
            u64::try_from(n_fft).map_err(|_| LegacyError::InvalidMel)?,
            u64::try_from(n_mel).map_err(|_| LegacyError::InvalidMel)?,
            1,
            1,
        ],
        n_dims: 2,
        dtype: 0,
        data: mel_data,
    }];
    let vocab = reader.i32()?;
    if vocab <= 0 {
        return Err(LegacyError::InvalidVocabulary);
    }
    for _ in 0..vocab {
        let size = usize::try_from(reader.u32()?).map_err(|_| LegacyError::Capacity)?;
        reader.read(size)?;
    }
    while reader.offset < source.len() {
        let n_dims = reader.i32()?;
        let name_size = reader.i32()?;
        let dtype = reader.i32()?;
        let active = usize::try_from(n_dims)
            .ok()
            .filter(|count| (1..=4).contains(count))
            .ok_or(LegacyError::InvalidTensorHeader)?;
        let name_size = usize::try_from(name_size)
            .ok()
            .filter(|size| *size > 0)
            .ok_or(LegacyError::InvalidTensorHeader)?;
        let dtype = u32::try_from(dtype).map_err(|_| LegacyError::UnsupportedDType)?;
        let mut dims = [1_u64; 4];
        for dimension in dims.iter_mut().take(active) {
            let value = reader.i32()?;
            if value <= 0 {
                return Err(LegacyError::InvalidDimensions);
            }
            *dimension = u64::try_from(value).map_err(|_| LegacyError::InvalidDimensions)?;
        }
        let source_name = reader.read(name_size)?;
        let name = canonical_name(source_name).ok_or(LegacyError::UnknownTensorName)?;
        let (n_dims, dims) = if active == 2
            && dims[0] == 1
            && (name.ends_with(b".weight") || name.ends_with(b".bias"))
        {
            (1, [dims[1], 1, 1, 1])
        } else {
            (
                u32::try_from(n_dims).map_err(|_| LegacyError::InvalidTensorHeader)?,
                dims,
            )
        };
        let size = legacy_data_size(
            dtype,
            &dims[..usize::try_from(n_dims).map_err(|_| LegacyError::Capacity)?],
        )?;
        let data = reader.read(size)?;
        tensors.push(LegacyTensor {
            name,
            dims,
            n_dims,
            dtype,
            data,
        });
    }
    Ok((tensors, n_mel, hparams[0]))
}

fn append_u32(output: &mut Vec<u8>, value: u32) {
    output.extend_from_slice(&value.to_le_bytes());
}
fn append_u64(output: &mut Vec<u8>, value: u64) {
    output.extend_from_slice(&value.to_le_bytes());
}
fn append_string(output: &mut Vec<u8>, value: &[u8]) {
    append_u64(output, u64::try_from(value.len()).unwrap_or(u64::MAX));
    output.extend_from_slice(value);
}
fn append_padding(output: &mut Vec<u8>) {
    let padding = (32 - output.len() % 32) % 32;
    output.resize(output.len() + padding, 0);
}

/// Converts a validated legacy `lmgg` image into canonical GGUF v3 bytes.
///
/// # Errors
///
/// Returns an error when the legacy image is malformed, unsupported, or exceeds bounded capacities.
pub fn normalize_legacy_lmgg_to_gguf(
    source: &[u8],
    output: &mut Vec<u8>,
) -> Result<(), LegacyError> {
    let (tensors, n_mels, n_vocab) = parse_legacy(source)?;
    output.clear();
    output.extend_from_slice(b"GGUF");
    append_u32(output, 3);
    append_u64(
        output,
        u64::try_from(tensors.len()).map_err(|_| LegacyError::Capacity)?,
    );
    append_u64(output, 4);
    append_string(output, b"general.architecture");
    append_u32(output, 8);
    append_string(output, ARCHITECTURE_NAME);
    append_string(output, b"general.alignment");
    append_u32(output, 4);
    append_u32(output, 32);
    append_string(output, b"whisper.n_mels");
    append_u32(output, 4);
    append_u32(
        output,
        u32::try_from(n_mels).map_err(|_| LegacyError::Capacity)?,
    );
    append_string(output, b"whisper.n_vocab");
    append_u32(output, 4);
    append_u32(
        output,
        u32::try_from(n_vocab).map_err(|_| LegacyError::Capacity)?,
    );
    let mut offset = 0_u64;
    for tensor in &tensors {
        append_string(output, &tensor.name);
        append_u32(output, tensor.n_dims);
        for dimension in tensor
            .dims
            .iter()
            .take(usize::try_from(tensor.n_dims).map_err(|_| LegacyError::Capacity)?)
        {
            append_u64(output, *dimension);
        }
        append_u32(output, tensor.dtype);
        append_u64(output, offset);
        offset = offset
            .checked_add(u64::try_from(tensor.data.len()).map_err(|_| LegacyError::Capacity)?)
            .ok_or(LegacyError::Capacity)?;
        offset = offset
            .checked_add((32 - offset % 32) % 32)
            .ok_or(LegacyError::Capacity)?;
    }
    append_padding(output);
    for tensor in tensors {
        output.extend_from_slice(tensor.data);
        append_padding(output);
    }
    Ok(())
}

fn required_tensor<'a>(
    model: WhisperBindingInput<'a>,
    name: &[u8],
    dims: &[u64],
    kinds: &[SerializedType],
) -> Result<TensorView<'a>, WhisperError> {
    let tensor = model
        .tensor_named(name)
        .ok_or(WhisperError::MissingTensor)?;
    let metadata = tensor.metadata().ok_or(WhisperError::InvalidTensor)?;
    if metadata.dimension_count() as usize != dims.len()
        || metadata.dimensions()[..dims.len()] != *dims
    {
        return Err(WhisperError::WrongShape);
    }
    if !kinds.contains(&metadata.tensor_type()) {
        return Err(WhisperError::WrongDType);
    }
    if tensor.bytes().is_none() {
        return Err(WhisperError::NonResident);
    }
    if tensor
        .bytes()
        .is_some_and(|bytes| u64::try_from(bytes.len()).ok() != Some(metadata.data_size()))
    {
        return Err(WhisperError::InvalidTensor);
    }
    Ok(tensor)
}

fn validate_contract(
    model: WhisperBindingInput<'_>,
) -> Result<ExecutionContract<'_>, WhisperError> {
    if !is_execution_architecture(model.architecture()) {
        return Err(WhisperError::WrongArchitecture);
    }
    let hparams = *model.hparams().ok_or(WhisperError::MissingHParams)?;
    WhisperHParams::try_new(WhisperHParamsInput {
        n_mels: hparams.n_mels(),
        n_vocab: hparams.n_vocab(),
        n_embd: hparams.n_embd(),
        n_ff: hparams.n_ff(),
        n_head: hparams.n_head(),
        n_head_kv: hparams.n_head_kv(),
        n_ctx: hparams.n_ctx(),
        encoder_block_count: hparams.encoder_block_count(),
        decoder_block_count: hparams.decoder_block_count(),
    })
    .map_err(WhisperError::InvalidHParams)?;
    let q8_or_f32 = [SerializedType::Q8_0, SerializedType::F32];
    let mel_filters = required_tensor(model, b"mel_filters", &[201, 80], &[SerializedType::F32])?;
    required_tensor(
        model,
        b"model.encoder.conv1.weight",
        &[3, 80, 384],
        &[SerializedType::F16],
    )?;
    required_tensor(
        model,
        b"model.encoder.embed_positions.weight",
        &[384, 1500],
        &q8_or_f32,
    )?;
    required_tensor(
        model,
        b"model.decoder.embed_tokens.weight",
        &[384, 51865],
        &[SerializedType::Q8_0],
    )?;
    required_tensor(
        model,
        b"model.decoder.embed_positions.weight",
        &[384, 448],
        &q8_or_f32,
    )?;
    let encoder = model.encoder();
    let decoder = model.decoder();
    if encoder.tensor_count() == 0 {
        return Err(WhisperError::MissingFamily(WhisperFamily::Encoder));
    }
    if decoder.tensor_count() == 0 {
        return Err(WhisperError::MissingFamily(WhisperFamily::Decoder));
    }
    Ok(ExecutionContract {
        model,
        sample_rate: SAMPLE_RATE,
        mel_bin_count: i32::try_from(N_MELS).unwrap_or(i32::MAX),
        vocab_size: i32::try_from(N_VOCAB).unwrap_or(i32::MAX),
        embedding_length: i32::try_from(N_EMBD).unwrap_or(i32::MAX),
        feed_forward_length: i32::try_from(N_FF).unwrap_or(i32::MAX),
        attention_head_count: i32::try_from(N_HEAD).unwrap_or(i32::MAX),
        encoder_context_length: i32::try_from(ENCODER_CONTEXT_LENGTH).unwrap_or(i32::MAX),
        decoder_context_length: i32::try_from(N_CTX).unwrap_or(i32::MAX),
        encoder_block_count: i32::try_from(ENCODER_BLOCK_COUNT).unwrap_or(i32::MAX),
        decoder_block_count: i32::try_from(DECODER_BLOCK_COUNT).unwrap_or(i32::MAX),
        mel_filters,
        encoder,
        decoder,
    })
}

#[derive(Debug, Default)]
pub struct Detail;
impl Detail {
    /// Binds validated Whisper layers.
    ///
    /// # Errors
    ///
    /// Returns an error when the model contract is invalid.
    pub fn bind_layers<'a>(
        &self,
        model: WhisperBindingInput<'a>,
    ) -> Result<ExecutionContract<'a>, WhisperError> {
        validate_contract(model)
    }
    /// Loads validated Whisper hparams.
    ///
    /// # Errors
    ///
    /// Returns an error when the model contract is invalid or hparams are absent.
    pub fn load_hparams<'a>(
        &self,
        model: WhisperBindingInput<'a>,
    ) -> Result<&'a WhisperHParams, WhisperError> {
        self.bind_layers(model)?
            .model()
            .hparams()
            .ok_or(WhisperError::MissingHParams)
    }
}

#[derive(Debug, Default, PartialEq, Eq)]
pub struct Any;
impl Any {
    /// Binds a validated Whisper model.
    ///
    /// # Errors
    ///
    /// Returns an error when the model contract is invalid.
    pub fn bind(model: WhisperBindingInput<'_>) -> Result<ExecutionContract<'_>, WhisperError> {
        Detail.bind_layers(model)
    }
}

#[cfg(test)]
#[allow(clippy::too_many_lines)]
mod tests {
    use super::*;
    use crate::data::TensorRecord;

    fn append_i32(output: &mut Vec<u8>, value: i32) {
        output.extend_from_slice(&value.to_le_bytes());
    }

    fn append_u32(output: &mut Vec<u8>, value: u32) {
        output.extend_from_slice(&value.to_le_bytes());
    }

    fn legacy_header(n_mel: i32, n_fft: i32, vocab: i32, mel: bool) -> Vec<u8> {
        let mut output = b"lmgg".to_vec();
        for value in [
            N_VOCAB,
            ENCODER_CONTEXT_LENGTH,
            N_EMBD,
            N_HEAD,
            ENCODER_BLOCK_COUNT,
            N_CTX,
            N_EMBD,
            N_HEAD,
            DECODER_BLOCK_COUNT,
            N_MELS,
            0,
        ] {
            append_i32(
                &mut output,
                i32::try_from(value).expect("Whisper fixture hparam fits i32"),
            );
        }
        append_i32(&mut output, n_mel);
        append_i32(&mut output, n_fft);
        if mel {
            let size = usize::try_from(n_mel)
                .unwrap()
                .checked_mul(usize::try_from(n_fft).unwrap())
                .unwrap()
                .checked_mul(4)
                .unwrap();
            output.resize(output.len() + size, 0);
        }
        append_i32(&mut output, vocab);
        output
    }

    fn valid_prefix() -> Vec<u8> {
        let mut output = legacy_header(80, 1, 1, true);
        append_u32(&mut output, 0);
        output
    }

    fn append_tensor(
        output: &mut Vec<u8>,
        name: &[u8],
        n_dims: i32,
        dtype: i32,
        dimensions: &[i32],
        data: &[u8],
    ) {
        append_i32(output, n_dims);
        append_i32(
            output,
            i32::try_from(name.len()).expect("Whisper fixture name length fits i32"),
        );
        append_i32(output, dtype);
        for &dimension in dimensions {
            append_i32(output, dimension);
        }
        output.extend_from_slice(name);
        output.extend_from_slice(data);
    }

    fn valid_legacy() -> Vec<u8> {
        let mut output = valid_prefix();
        append_tensor(
            &mut output,
            b"encoder.conv1.weight",
            2,
            0,
            &[1, 3],
            &[0; 12],
        );
        output
    }

    fn assert_legacy_error(source: &[u8], expected: LegacyError) {
        match parse_legacy(source) {
            Err(actual) => assert_eq!(actual, expected),
            Ok(_) => panic!("expected legacy error"),
        }
    }

    #[test]
    fn hparams_default_validation_errors_and_accessors() {
        let base = WhisperHParamsInput::default();
        assert_eq!(base.n_mels, N_MELS);
        assert_eq!(base.n_vocab, N_VOCAB);
        assert_eq!(base.n_embd, N_EMBD);
        assert_eq!(base.n_ff, N_FF);
        assert_eq!(base.n_head, N_HEAD);
        assert_eq!(base.n_head_kv, N_HEAD_KV);
        assert_eq!(base.n_ctx, N_CTX);
        assert_eq!(base.encoder_block_count, ENCODER_BLOCK_COUNT);
        assert_eq!(base.decoder_block_count, DECODER_BLOCK_COUNT);

        assert_eq!(
            WhisperHParams::try_new(WhisperHParamsInput { n_mels: 0, ..base }),
            Err(WhisperHParamsError::InvalidNMels)
        );
        assert_eq!(
            WhisperHParams::try_new(WhisperHParamsInput { n_vocab: 0, ..base }),
            Err(WhisperHParamsError::InvalidNVocab)
        );
        assert_eq!(
            WhisperHParams::try_new(WhisperHParamsInput { n_embd: 0, ..base }),
            Err(WhisperHParamsError::InvalidEmbedding)
        );
        assert_eq!(
            WhisperHParams::try_new(WhisperHParamsInput { n_ff: 0, ..base }),
            Err(WhisperHParamsError::InvalidFeedForward)
        );
        assert_eq!(
            WhisperHParams::try_new(WhisperHParamsInput { n_head: 0, ..base }),
            Err(WhisperHParamsError::InvalidHeads)
        );
        assert_eq!(
            WhisperHParams::try_new(WhisperHParamsInput {
                n_head_kv: 0,
                ..base
            }),
            Err(WhisperHParamsError::InvalidHeads)
        );
        assert_eq!(
            WhisperHParams::try_new(WhisperHParamsInput { n_ctx: 0, ..base }),
            Err(WhisperHParamsError::InvalidContext)
        );
        assert_eq!(
            WhisperHParams::try_new(WhisperHParamsInput {
                encoder_block_count: 0,
                ..base
            }),
            Err(WhisperHParamsError::InvalidEncoderBlocks)
        );
        assert_eq!(
            WhisperHParams::try_new(WhisperHParamsInput {
                decoder_block_count: 0,
                ..base
            }),
            Err(WhisperHParamsError::InvalidDecoderBlocks)
        );

        let hparams = WhisperHParams::try_new(base).unwrap();
        assert_eq!(hparams.n_mels(), N_MELS);
        assert_eq!(hparams.n_vocab(), N_VOCAB);
        assert_eq!(hparams.n_embd(), N_EMBD);
        assert_eq!(hparams.n_ff(), N_FF);
        assert_eq!(hparams.n_head(), N_HEAD);
        assert_eq!(hparams.n_head_kv(), N_HEAD_KV);
        assert_eq!(hparams.n_ctx(), N_CTX);
        assert_eq!(hparams.encoder_block_count(), ENCODER_BLOCK_COUNT);
        assert_eq!(hparams.decoder_block_count(), DECODER_BLOCK_COUNT);
    }

    #[test]
    fn canonical_suffixes_and_names_cover_all_routes() {
        let attention = [
            (b"query.weight".as_slice(), b"q_proj.weight".as_slice()),
            (b"query.bias", b"q_proj.bias"),
            (b"key.weight", b"k_proj.weight"),
            (b"value.weight", b"v_proj.weight"),
            (b"value.bias", b"v_proj.bias"),
            (b"out.weight", b"out_proj.weight"),
            (b"out.bias", b"out_proj.bias"),
        ];
        for (source, expected) in attention {
            assert_eq!(canonical_attention_suffix(source), Some(expected));
        }
        assert_eq!(canonical_attention_suffix(b"unknown"), None);

        let encoder = [
            (
                b"attn_ln.weight".as_slice(),
                b"self_attn_layer_norm.weight".as_slice(),
            ),
            (b"attn_ln.bias", b"self_attn_layer_norm.bias"),
            (b"attn.query.weight", b"self_attn.q_proj.weight"),
            (b"attn.query.bias", b"self_attn.q_proj.bias"),
            (b"attn.key.weight", b"self_attn.k_proj.weight"),
            (b"attn.value.weight", b"self_attn.v_proj.weight"),
            (b"attn.value.bias", b"self_attn.v_proj.bias"),
            (b"attn.out.weight", b"self_attn.out_proj.weight"),
            (b"attn.out.bias", b"self_attn.out_proj.bias"),
            (b"mlp_ln.weight", b"final_layer_norm.weight"),
            (b"mlp_ln.bias", b"final_layer_norm.bias"),
            (b"mlp.0.weight", b"fc1.weight"),
            (b"mlp.0.bias", b"fc1.bias"),
            (b"mlp.2.weight", b"fc2.weight"),
            (b"mlp.2.bias", b"fc2.bias"),
        ];
        for (source, expected) in encoder {
            assert_eq!(canonical_encoder_suffix(source), Some(expected));
        }
        assert_eq!(canonical_encoder_suffix(b"unknown"), None);
        assert_eq!(
            canonical_decoder_suffix(b"cross_attn_ln.weight"),
            Some(b"encoder_attn_layer_norm.weight".to_vec())
        );
        assert_eq!(
            canonical_decoder_suffix(b"cross_attn_ln.bias"),
            Some(b"encoder_attn_layer_norm.bias".to_vec())
        );
        assert_eq!(canonical_decoder_suffix(b"cross_attn_ln.unknown"), None);
        assert_eq!(
            canonical_decoder_suffix(b"attn.query.weight"),
            Some(b"self_attn.q_proj.weight".to_vec())
        );
        assert_eq!(canonical_decoder_suffix(b"unknown"), None);

        let fixed = [
            (
                b"encoder.positional_embedding".as_slice(),
                b"model.encoder.embed_positions.weight".as_slice(),
            ),
            (b"encoder.conv1.weight", b"model.encoder.conv1.weight"),
            (b"encoder.conv1.bias", b"model.encoder.conv1.bias"),
            (b"encoder.conv2.weight", b"model.encoder.conv2.weight"),
            (b"encoder.conv2.bias", b"model.encoder.conv2.bias"),
            (
                b"encoder.ln_post.weight",
                b"model.encoder.layer_norm.weight",
            ),
            (b"encoder.ln_post.bias", b"model.encoder.layer_norm.bias"),
            (
                b"decoder.positional_embedding",
                b"model.decoder.embed_positions.weight",
            ),
            (
                b"decoder.token_embedding.weight",
                b"model.decoder.embed_tokens.weight",
            ),
            (b"decoder.ln.weight", b"model.decoder.layer_norm.weight"),
            (b"decoder.ln.bias", b"model.decoder.layer_norm.bias"),
        ];
        for (source, expected) in fixed {
            assert_eq!(canonical_name(source), Some(expected.to_vec()));
        }
        assert_eq!(
            canonical_name(b"encoder.blocks.0.attn_ln.weight"),
            Some(b"model.encoder.layers.0.self_attn_layer_norm.weight".to_vec())
        );
        assert_eq!(
            canonical_name(b"encoder.blocks.3.mlp.2.bias"),
            Some(b"model.encoder.layers.3.fc2.bias".to_vec())
        );
        assert_eq!(
            canonical_name(b"decoder.blocks.1.attn.query.weight"),
            Some(b"model.decoder.layers.1.self_attn.q_proj.weight".to_vec())
        );
        assert_eq!(
            canonical_name(b"decoder.blocks.2.cross_attn.query.weight"),
            Some(b"model.decoder.layers.2.encoder_attn.q_proj.weight".to_vec())
        );
        assert_eq!(
            canonical_name(b"decoder.blocks.3.cross_attn_ln.bias"),
            Some(b"model.decoder.layers.3.encoder_attn_layer_norm.bias".to_vec())
        );
        assert_eq!(canonical_name(b"encoder.blocks.0.unknown"), None);
        assert_eq!(canonical_name(b"decoder.blocks.4.attn.query.weight"), None);
        assert_eq!(canonical_name(b"unknown"), None);
    }

    #[test]
    fn scalar_helpers_cover_capacity_and_dtype_paths() {
        let mut buffer = [0_u8; 64];
        let used = decimal_name(&mut buffer, b"model.", 1203, b".weight").unwrap();
        assert_eq!(&buffer[..used], b"model.1203.weight");
        assert_eq!(
            decimal_name(&mut [0_u8; 2], b"prefix", 0, b"suffix"),
            Err(WhisperError::Capacity)
        );
        assert_eq!(
            decimal_name(&mut [0_u8; 8], b"prefix", 0, b"suffix"),
            Err(WhisperError::Capacity)
        );

        let mut reader = LegacyReader {
            bytes: &[1, 0, 0, 0, 2, 0, 0, 0],
            offset: 0,
        };
        assert_eq!(reader.i32(), Ok(1));
        assert_eq!(reader.u32(), Ok(2));
        assert_eq!(reader.read(1), Err(LegacyError::Truncated));
        reader.offset = usize::MAX;
        assert_eq!(reader.read(1), Err(LegacyError::Capacity));

        assert_eq!(checked_product(&[]), Ok(1));
        assert_eq!(checked_product(&[2, 3, 4]), Ok(24));
        assert_eq!(checked_product(&[u64::MAX, 2]), Err(LegacyError::Capacity));
        assert_eq!(legacy_data_size(0, &[2, 3]), Ok(24));
        assert_eq!(legacy_data_size(1, &[2, 3]), Ok(12));
        assert_eq!(legacy_data_size(8, &[32]), Ok(34));
        assert_eq!(legacy_data_size(8, &[]), Err(LegacyError::UnsupportedDType));
        assert_eq!(
            legacy_data_size(8, &[31]),
            Err(LegacyError::UnsupportedDType)
        );
        assert_eq!(
            legacy_data_size(7, &[1]),
            Err(LegacyError::UnsupportedDType)
        );
        assert_eq!(
            legacy_data_size(0, &[u64::MAX / 4 + 1]),
            Err(LegacyError::InvalidTensorSize)
        );
    }

    #[test]
    fn legacy_parser_accepts_minimal_rank_normalized_tensor_and_normalizes() {
        let source = valid_legacy();
        let (tensors, n_mels, n_vocab) = parse_legacy(&source).unwrap();
        assert_eq!(n_mels, 80);
        assert_eq!(
            n_vocab,
            i32::try_from(N_VOCAB).expect("Whisper vocab fits i32")
        );
        assert_eq!(tensors.len(), 2);
        assert_eq!(tensors[0].name, b"mel_filters");
        assert_eq!(tensors[0].dims, [1, 80, 1, 1]);
        assert_eq!(tensors[0].n_dims, 2);
        assert_eq!(tensors[0].dtype, 0);
        assert_eq!(tensors[0].data.len(), 320);
        assert_eq!(tensors[1].name, b"model.encoder.conv1.weight");
        assert_eq!(tensors[1].dims, [3, 1, 1, 1]);
        assert_eq!(tensors[1].n_dims, 1);
        assert_eq!(tensors[1].data.len(), 12);

        let mut output = vec![9_u8, 8];
        normalize_legacy_lmgg_to_gguf(&source, &mut output).unwrap();
        assert_eq!(&output[..4], b"GGUF");
        assert_eq!(u32::from_le_bytes(output[4..8].try_into().unwrap()), 3);
        assert!(
            output
                .windows(b"general.architecture".len())
                .any(|window| window == b"general.architecture")
        );
        assert!(
            output
                .windows(b"model.encoder.conv1.weight".len())
                .any(|window| window == b"model.encoder.conv1.weight")
        );
        let mut unchanged = vec![1_u8, 2, 3];
        assert_eq!(
            normalize_legacy_lmgg_to_gguf(b"nope", &mut unchanged),
            Err(LegacyError::NotLegacy)
        );
        assert_eq!(unchanged, vec![1, 2, 3]);
    }

    #[test]
    fn legacy_parser_rejects_malformed_headers_metadata_and_tensor_payloads() {
        assert_legacy_error(b"nope", LegacyError::NotLegacy);
        assert_legacy_error(b"lmgg", LegacyError::Truncated);

        let mut invalid_hparams = legacy_header(80, 1, 1, true);
        invalid_hparams[4..8].copy_from_slice(&0_i32.to_le_bytes());
        assert_legacy_error(&invalid_hparams, LegacyError::InvalidHParams);
        assert_legacy_error(&legacy_header(79, 1, 1, false), LegacyError::InvalidMel);
        assert_legacy_error(&legacy_header(80, 0, 1, false), LegacyError::InvalidMel);
        assert_legacy_error(&legacy_header(80, 1, 1, false), LegacyError::Truncated);
        assert_legacy_error(
            &legacy_header(80, 1, 0, true),
            LegacyError::InvalidVocabulary,
        );

        let truncated_vocab = legacy_header(80, 1, 1, true);
        assert_legacy_error(&truncated_vocab, LegacyError::Truncated);

        let mut invalid_dimensions = valid_prefix();
        append_tensor(&mut invalid_dimensions, b"x", 1, 0, &[0], &[]);
        assert_legacy_error(&invalid_dimensions, LegacyError::InvalidDimensions);

        let mut invalid_tensor_header = valid_prefix();
        append_tensor(&mut invalid_tensor_header, b"x", 0, 0, &[], &[]);
        assert_legacy_error(&invalid_tensor_header, LegacyError::InvalidTensorHeader);

        let mut invalid_name_header = valid_prefix();
        append_i32(&mut invalid_name_header, 1);
        append_i32(&mut invalid_name_header, 0);
        append_i32(&mut invalid_name_header, 0);
        assert_legacy_error(&invalid_name_header, LegacyError::InvalidTensorHeader);

        let mut unsupported_dtype = valid_prefix();
        append_tensor(&mut unsupported_dtype, b"x", 1, -1, &[], &[]);
        assert_legacy_error(&unsupported_dtype, LegacyError::UnsupportedDType);

        let mut unknown_name = valid_prefix();
        append_tensor(&mut unknown_name, b"unknown", 1, 0, &[1], &[]);
        assert_legacy_error(&unknown_name, LegacyError::UnknownTensorName);

        let mut truncated_data = valid_prefix();
        append_tensor(
            &mut truncated_data,
            b"encoder.conv1.weight",
            1,
            0,
            &[1],
            &[],
        );
        assert_legacy_error(&truncated_data, LegacyError::Truncated);
    }

    #[test]
    fn architecture_predicates_cover_exact_and_short_inputs() {
        assert!(is_execution_architecture(b"whisper"));
        assert!(!is_execution_architecture(b"whisper.extra"));
        assert!(!is_execution_architecture(b""));
        assert!(is_legacy_lmgg_whisper(b"lmgg"));
        assert!(is_legacy_lmgg_whisper(b"lmgg payload"));
        assert!(!is_legacy_lmgg_whisper(b"lmg"));
        assert!(!is_legacy_lmgg_whisper(b"GGUF"));
    }
    fn compact_fixture(with_encoder: bool, with_decoder: bool) -> Data {
        let mut data = Data::try_new().unwrap();
        data.architecture_name[..ARCHITECTURE_NAME.len()].copy_from_slice(ARCHITECTURE_NAME);
        data.whisper = Some(WhisperHParams::try_new(WhisperHParamsInput::default()).unwrap());
        let mut specs = vec![
            (
                b"mel_filters".as_slice(),
                SerializedType::F32,
                [201, 80, 1, 1],
                2,
            ),
            (
                b"model.encoder.conv1.weight".as_slice(),
                SerializedType::F16,
                [3, 80, 384, 1],
                3,
            ),
            (
                b"model.encoder.embed_positions.weight".as_slice(),
                SerializedType::F32,
                [384, 1500, 1, 1],
                2,
            ),
            (
                b"model.decoder.embed_tokens.weight".as_slice(),
                SerializedType::Q8_0,
                [384, 51865, 1, 1],
                2,
            ),
            (
                b"model.decoder.embed_positions.weight".as_slice(),
                SerializedType::F32,
                [384, 448, 1, 1],
                2,
            ),
        ];
        if with_encoder {
            specs.push((
                b"model.encoder.layers.0.self_attn.q_proj.weight",
                SerializedType::Q4_0,
                [384, 384, 1, 1],
                2,
            ));
        }
        if with_decoder {
            specs.push((
                b"model.decoder.layers.0.self_attn.q_proj.weight",
                SerializedType::Q4_0,
                [384, 384, 1, 1],
                2,
            ));
        }
        for (index, (name, ty, dims, rank)) in specs.iter().enumerate() {
            let offset = data.name_bytes_used as usize;
            data.name_storage[offset..offset + name.len()].copy_from_slice(name);
            data.name_bytes_used += u32::try_from(name.len()).unwrap();
            data.tensors[index] = TensorRecord {
                name_offset: u32::try_from(offset).unwrap(),
                name_length: u32::try_from(name.len()).unwrap(),
                r#type: i32::try_from(ty.wire_code()).unwrap(),
                n_dims: *rank,
                dims: dims.map(i64::from),
                data_offset: 0,
                file_offset: 0,
                data_size: 1,
                data: None,
                bytes: Some(vec![0_u8].into_boxed_slice()),
                file_index: 0,
            };
        }
        data.n_tensors = u32::try_from(specs.len()).unwrap();
        data
    }

    #[test]
    fn binding_contract_exposes_all_accessors() {
        let data = compact_fixture(true, true);
        let model = data.whisper_binding_input();
        let contract = Any::bind(model).unwrap();
        assert_eq!(
            (
                contract.sample_rate(),
                contract.mel_bin_count(),
                contract.vocab_size()
            ),
            (
                SAMPLE_RATE,
                i32::try_from(N_MELS).expect("Whisper mel count fits i32"),
                i32::try_from(N_VOCAB).expect("Whisper vocab fits i32")
            )
        );
        assert_eq!(
            (
                contract.embedding_length(),
                contract.feed_forward_length(),
                contract.attention_head_count()
            ),
            (
                i32::try_from(N_EMBD).expect("Whisper embedding size fits i32"),
                i32::try_from(N_FF).expect("Whisper feed-forward size fits i32"),
                i32::try_from(N_HEAD).expect("Whisper head count fits i32")
            )
        );
        assert_eq!(
            (
                contract.encoder_context_length(),
                contract.decoder_context_length()
            ),
            (
                i32::try_from(ENCODER_CONTEXT_LENGTH).expect("Whisper encoder context fits i32"),
                i32::try_from(N_CTX).expect("Whisper decoder context fits i32")
            )
        );
        assert_eq!(
            (
                contract.encoder_block_count(),
                contract.decoder_block_count()
            ),
            (
                i32::try_from(ENCODER_BLOCK_COUNT).expect("Whisper encoder block count fits i32"),
                i32::try_from(DECODER_BLOCK_COUNT).expect("Whisper decoder block count fits i32")
            )
        );
        let mel = contract.mel_filters();
        assert_eq!(mel.name(), b"mel_filters");
        assert_eq!(mel.metadata().unwrap().dimensions(), [201, 80, 1, 1]);
        assert_eq!(mel.bytes(), Some(&[0_u8][..]));
        let binding = contract.model();
        assert_eq!(binding.architecture(), ARCHITECTURE_NAME);
        assert_eq!(binding.hparams().unwrap().n_ctx(), N_CTX);
        assert_eq!(binding.tensor_count(), 7);
        assert_eq!(
            binding.tensor_named(b"mel_filters").unwrap().name(),
            b"mel_filters"
        );
        assert_eq!(binding.tensor(0).unwrap().name(), b"mel_filters");
        assert!(binding.tensor_named(b"missing").is_none() && binding.tensor(99).is_none());
        assert_eq!(
            binding.block_count(WhisperFamily::Encoder),
            ENCODER_BLOCK_COUNT
        );
        assert_eq!(
            binding.block_count(WhisperFamily::Decoder),
            DECODER_BLOCK_COUNT
        );
        for (family, view, name) in [
            (
                WhisperFamily::Encoder,
                contract.encoder(),
                b"model.encoder.layers.0.self_attn.q_proj.weight".as_slice(),
            ),
            (
                WhisperFamily::Decoder,
                contract.decoder(),
                b"model.decoder.layers.0.self_attn.q_proj.weight".as_slice(),
            ),
        ] {
            assert_eq!(view.family(), family);
            assert_eq!(view.tensor_count(), 1);
            assert_eq!(view.first_tensor().unwrap().name(), name);
            assert_eq!(view.tensor(0).unwrap().name(), name);
            assert!(view.tensor(1).is_none());
        }
        assert_eq!(Detail.load_hparams(binding).unwrap().n_vocab(), N_VOCAB);
    }

    #[test]
    fn binding_contract_reports_public_failure_boundaries() {
        assert!(matches!(
            Any::bind(compact_fixture(false, true).whisper_binding_input()),
            Err(WhisperError::MissingFamily(WhisperFamily::Encoder))
        ));
        assert!(matches!(
            Any::bind(compact_fixture(true, false).whisper_binding_input()),
            Err(WhisperError::MissingFamily(WhisperFamily::Decoder))
        ));
        let mut wrong_arch = compact_fixture(true, true);
        wrong_arch.architecture_name[0] = b'x';
        assert!(matches!(
            Any::bind(wrong_arch.whisper_binding_input()),
            Err(WhisperError::WrongArchitecture)
        ));
        let mut missing = compact_fixture(true, true);
        missing.whisper = None;
        assert!(matches!(
            Any::bind(missing.whisper_binding_input()),
            Err(WhisperError::MissingHParams)
        ));
        let owned = compact_fixture(true, true);
        let model = owned.whisper_binding_input();
        assert!(matches!(
            required_tensor(model, b"missing", &[1], &[SerializedType::F32]),
            Err(WhisperError::MissingTensor)
        ));
        assert!(matches!(
            required_tensor(model, b"mel_filters", &[1, 80], &[SerializedType::F32]),
            Err(WhisperError::WrongShape)
        ));
        assert!(matches!(
            required_tensor(model, b"mel_filters", &[201, 80], &[SerializedType::F16]),
            Err(WhisperError::WrongDType)
        ));
        let mut nonresident = compact_fixture(true, true);
        nonresident.tensors[0].bytes = None;
        assert!(matches!(
            required_tensor(
                nonresident.whisper_binding_input(),
                b"mel_filters",
                &[201, 80],
                &[SerializedType::F32]
            ),
            Err(WhisperError::NonResident)
        ));
        let mut invalid = compact_fixture(true, true);
        invalid.tensors[0].r#type = -1;
        assert!(matches!(
            required_tensor(
                invalid.whisper_binding_input(),
                b"mel_filters",
                &[201, 80],
                &[SerializedType::F32]
            ),
            Err(WhisperError::InvalidTensor)
        ));
    }
}
