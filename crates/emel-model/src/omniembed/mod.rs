//! Model-owned immutable `OmniEmbed` contract boundary.

use core::fmt;

/// Exact architecture name accepted by the pinned model detail.
pub const ARCHITECTURE_NAME: &[u8] = b"omniembed";
/// Maximum number of Matryoshka dimensions accepted by the source schema.
pub const MAX_MATRYOSHKA_DIMS: usize = 16;

/// Encoder identities accepted by the pinned `OmniEmbed` metadata contract.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Encoder {
    MobileNetV4Medium,
    EfficientAtMn20As,
}

impl Encoder {
    /// Returns the exact source metadata value for this encoder.
    #[must_use]
    pub const fn source_name(self) -> &'static [u8] {
        match self {
            Self::MobileNetV4Medium => b"mobilenetv4_conv_medium.e180_r384_in12k",
            Self::EfficientAtMn20As => b"efficientat_mn20_as",
        }
    }
}

/// Tensor namespace identities used by the model/embeddings bridge.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Family {
    TextEncoder,
    TextProjection,
    ImageEncoder,
    ImageProjection,
    AudioEncoder,
    AudioProjection,
}

impl Family {
    /// Returns the exact source tensor prefix for this family.
    #[must_use]
    pub const fn prefix(self) -> &'static [u8] {
        match self {
            Self::TextEncoder => b"text_encoder.",
            Self::TextProjection => b"text_projection.",
            Self::ImageEncoder => b"image_encoder.",
            Self::ImageProjection => b"image_projection.",
            Self::AudioEncoder => b"audio_encoder.",
            Self::AudioProjection => b"audio_projection.",
        }
    }
}

/// Counts of tensors present in each source family namespace.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct TensorFamilies {
    pub text_encoder: u32,
    pub text_projection: u32,
    pub image_encoder: u32,
    pub image_projection: u32,
    pub audio_encoder: u32,
    pub audio_projection: u32,
}

impl TensorFamilies {
    /// Returns the count associated with one family identity.
    #[must_use]
    pub const fn count(self, family: Family) -> u32 {
        match family {
            Family::TextEncoder => self.text_encoder,
            Family::TextProjection => self.text_projection,
            Family::ImageEncoder => self.image_encoder,
            Family::ImageProjection => self.image_projection,
            Family::AudioEncoder => self.audio_encoder,
            Family::AudioProjection => self.audio_projection,
        }
    }
}

/// Vision preprocessing fixed by the pinned `MobileNetV4` contract.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VisionPreprocessing {
    pub image_size: i32,
    pub mean: [f32; 3],
    pub std: [f32; 3],
}

/// Audio preprocessing fixed by the pinned `EfficientAT` contract.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AudioPreprocessing {
    pub sample_rate: i32,
    pub n_fft: i32,
    pub win_length: i32,
    pub hop_size: i32,
    pub num_mel_bins: i32,
    pub low_frequency: f32,
    pub high_frequency: f32,
    pub preemphasis: f32,
    pub log_offset: f32,
    pub normalize_bias: f32,
    pub normalize_scale: f32,
}

/// Caller-owned metadata values validated before actor dispatch.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct HParamsInput<'a> {
    pub embedding_length: i32,
    pub image_encoder_length: i32,
    pub audio_encoder_length: i32,
    pub image_encoder: Encoder,
    pub audio_encoder: Encoder,
    pub matryoshka_dimensions: &'a [i32],
}

/// Validated immutable `OmniEmbed` metadata.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HParams {
    embedding_length: i32,
    image_encoder_length: i32,
    audio_encoder_length: i32,
    image_encoder: Encoder,
    audio_encoder: Encoder,
    matryoshka_dimensions: [i32; MAX_MATRYOSHKA_DIMS],
    matryoshka_dimension_count: u32,
    vision: VisionPreprocessing,
    audio: AudioPreprocessing,
}

impl HParams {
    /// Validates dimensions and the source-fixed encoder pairing.
    ///
    /// # Errors
    ///
    /// Returns an error when dimensions, Matryoshka values, or encoder pairing violate the pinned contract.
    pub fn try_new(input: HParamsInput<'_>) -> Result<Self, Error> {
        if input.embedding_length <= 0
            || input.image_encoder_length <= 0
            || input.audio_encoder_length <= 0
        {
            return Err(Error::InvalidDimensions);
        }
        if input.matryoshka_dimensions.is_empty()
            || input.matryoshka_dimensions.len() > MAX_MATRYOSHKA_DIMS
        {
            return Err(Error::InvalidMatryoshkaDimensions);
        }

        let mut previous = input
            .embedding_length
            .checked_add(1)
            .ok_or(Error::InvalidDimensions)?;
        let mut dimensions = [0; MAX_MATRYOSHKA_DIMS];
        for (index, &dimension) in input.matryoshka_dimensions.iter().enumerate() {
            if dimension <= 0 || dimension > input.embedding_length || dimension >= previous {
                return Err(Error::InvalidMatryoshkaDimensions);
            }
            dimensions[index] = dimension;
            previous = dimension;
        }

        let vision = match input.image_encoder {
            Encoder::MobileNetV4Medium => VisionPreprocessing {
                image_size: 384,
                mean: [0.485, 0.456, 0.406],
                std: [0.229, 0.224, 0.225],
            },
            Encoder::EfficientAtMn20As => {
                return Err(Error::UnsupportedEncoder(Family::ImageEncoder));
            }
        };
        let audio = match input.audio_encoder {
            Encoder::EfficientAtMn20As => AudioPreprocessing {
                sample_rate: 32_000,
                n_fft: 1024,
                win_length: 800,
                hop_size: 320,
                num_mel_bins: 128,
                low_frequency: 0.0,
                high_frequency: 15_000.0,
                preemphasis: 0.97,
                log_offset: 1.0e-5,
                normalize_bias: 4.5,
                normalize_scale: 5.0,
            },
            Encoder::MobileNetV4Medium => {
                return Err(Error::UnsupportedEncoder(Family::AudioEncoder));
            }
        };

        Ok(Self {
            embedding_length: input.embedding_length,
            image_encoder_length: input.image_encoder_length,
            audio_encoder_length: input.audio_encoder_length,
            image_encoder: input.image_encoder,
            audio_encoder: input.audio_encoder,
            matryoshka_dimensions: dimensions,
            matryoshka_dimension_count: u32::try_from(input.matryoshka_dimensions.len())
                .unwrap_or(0),
            vision,
            audio,
        })
    }

    #[must_use]
    pub const fn embedding_length(self) -> i32 {
        self.embedding_length
    }
    #[must_use]
    pub const fn image_encoder_length(self) -> i32 {
        self.image_encoder_length
    }
    #[must_use]
    pub const fn audio_encoder_length(self) -> i32 {
        self.audio_encoder_length
    }
    #[must_use]
    pub const fn image_encoder(self) -> Encoder {
        self.image_encoder
    }
    #[must_use]
    pub const fn audio_encoder(self) -> Encoder {
        self.audio_encoder
    }
    #[must_use]
    pub const fn matryoshka_dimension_count(self) -> u32 {
        self.matryoshka_dimension_count
    }
    #[must_use]
    pub const fn matryoshka_dimensions(self) -> [i32; MAX_MATRYOSHKA_DIMS] {
        self.matryoshka_dimensions
    }
    #[must_use]
    pub const fn vision(self) -> VisionPreprocessing {
        self.vision
    }
    #[must_use]
    pub const fn audio(self) -> AudioPreprocessing {
        self.audio
    }
}

/// Immutable bridge input retained only as borrowed model facts.
#[derive(Clone, Copy, Debug)]
pub struct BindingInput<'a> {
    architecture: &'a [u8],
    hparams: &'a HParams,
    tensor_families: TensorFamilies,
}

impl<'a> BindingInput<'a> {
    /// Creates a bridge from caller-owned immutable metadata.
    #[must_use]
    pub const fn new(
        architecture: &'a [u8],
        hparams: &'a HParams,
        tensor_families: TensorFamilies,
    ) -> Self {
        Self {
            architecture,
            hparams,
            tensor_families,
        }
    }
    #[must_use]
    pub const fn architecture(self) -> &'a [u8] {
        self.architecture
    }
    #[must_use]
    pub const fn hparams(self) -> &'a HParams {
        self.hparams
    }
    #[must_use]
    pub const fn tensor_families(self) -> TensorFamilies {
        self.tensor_families
    }
}

/// Explicit malformed, unsupported, and boundary-validation failures.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum Error {
    WrongArchitecture,
    InvalidDimensions,
    InvalidMatryoshkaDimensions,
    UnsupportedEncoder(Family),
    EmptyTensorFamily(Family),
}

impl fmt::Display for Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::WrongArchitecture => {
                formatter.write_str("OmniEmbed architecture metadata is missing or mismatched")
            }
            Self::InvalidDimensions => formatter.write_str("OmniEmbed dimensions are invalid"),
            Self::InvalidMatryoshkaDimensions => {
                formatter.write_str("OmniEmbed Matryoshka dimensions are malformed")
            }
            Self::UnsupportedEncoder(family) => {
                write!(formatter, "unsupported OmniEmbed encoder for {family:?}")
            }
            Self::EmptyTensorFamily(family) => {
                write!(formatter, "OmniEmbed tensor family {family:?} is empty")
            }
        }
    }
}

impl std::error::Error for Error {}

/// Returns true only for the exact byte-oriented source architecture name.
#[must_use]
pub fn is_execution_architecture(name: &[u8]) -> bool {
    name == ARCHITECTURE_NAME
}

/// Validated model-side contract for the public embeddings actor.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Contract<'a> {
    hparams: &'a HParams,
    tensor_families: TensorFamilies,
}

impl<'a> Contract<'a> {
    /// Returns the validated immutable model metadata.
    #[must_use]
    pub const fn hparams(self) -> &'a HParams {
        self.hparams
    }
    /// Returns validated source family counts.
    #[must_use]
    pub const fn tensor_families(self) -> TensorFamilies {
        self.tensor_families
    }
}

/// Model detail facade used before dispatching the public embeddings actor.
#[derive(Debug, Default)]
pub struct Detail;

impl Detail {
    /// Validates architecture, immutable metadata, and all family counts.
    ///
    /// # Errors
    ///
    /// Returns an error when architecture, dimensions, encoder pairing, or family counts violate the pinned contract.
    pub fn validate<'a>(&self, input: BindingInput<'a>) -> Result<Contract<'a>, Error> {
        if !is_execution_architecture(input.architecture) {
            return Err(Error::WrongArchitecture);
        }
        if input.hparams.embedding_length <= 0
            || input.hparams.image_encoder_length <= 0
            || input.hparams.audio_encoder_length <= 0
            || input.hparams.matryoshka_dimension_count == 0
        {
            return Err(Error::InvalidDimensions);
        }
        let count = usize::try_from(input.hparams.matryoshka_dimension_count)
            .map_err(|_| Error::InvalidMatryoshkaDimensions)?;
        if count > MAX_MATRYOSHKA_DIMS {
            return Err(Error::InvalidMatryoshkaDimensions);
        }
        let mut previous = input
            .hparams
            .embedding_length
            .checked_add(1)
            .ok_or(Error::InvalidDimensions)?;
        for &dimension in &input.hparams.matryoshka_dimensions[..count] {
            if dimension <= 0 || dimension > input.hparams.embedding_length || dimension >= previous
            {
                return Err(Error::InvalidMatryoshkaDimensions);
            }
            previous = dimension;
        }
        if input.hparams.image_encoder != Encoder::MobileNetV4Medium {
            return Err(Error::UnsupportedEncoder(Family::ImageEncoder));
        }
        if input.hparams.audio_encoder != Encoder::EfficientAtMn20As {
            return Err(Error::UnsupportedEncoder(Family::AudioEncoder));
        }
        for family in [
            Family::TextEncoder,
            Family::TextProjection,
            Family::ImageEncoder,
            Family::ImageProjection,
            Family::AudioEncoder,
            Family::AudioProjection,
        ] {
            if input.tensor_families.count(family) == 0 {
                return Err(Error::EmptyTensorFamily(family));
            }
        }
        Ok(Contract {
            hparams: input.hparams,
            tensor_families: input.tensor_families,
        })
    }

    /// Validates and returns the ownership-safe actor bridge.
    ///
    /// # Errors
    ///
    /// Returns an error when metadata validation fails.
    pub fn bind_layers<'a>(&self, input: BindingInput<'a>) -> Result<Contract<'a>, Error> {
        self.validate(input)
    }

    /// Validates and returns immutable hparams for actor dispatch.
    ///
    /// # Errors
    ///
    /// Returns an error when metadata validation fails.
    pub fn load_hparams<'a>(&self, input: BindingInput<'a>) -> Result<&'a HParams, Error> {
        Ok(self.validate(input)?.hparams())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hparams() -> HParams {
        HParams::try_new(HParamsInput {
            embedding_length: 512,
            image_encoder_length: 768,
            audio_encoder_length: 1024,
            image_encoder: Encoder::MobileNetV4Medium,
            audio_encoder: Encoder::EfficientAtMn20As,
            matryoshka_dimensions: &[512, 256, 128],
        })
        .unwrap()
    }

    fn families() -> TensorFamilies {
        TensorFamilies {
            text_encoder: 1,
            text_projection: 1,
            image_encoder: 1,
            image_projection: 1,
            audio_encoder: 1,
            audio_projection: 1,
        }
    }

    #[test]
    fn exact_architecture_and_source_preprocessing_are_exposed() {
        let metadata = hparams();
        assert!(is_execution_architecture(b"omniembed"));
        assert!(!is_execution_architecture(b"omniembed.extra"));
        assert!(!is_execution_architecture(&[0xff]));
        assert_eq!(metadata.vision().image_size, 384);
        assert_eq!(metadata.vision().mean[0].to_bits(), 0.485_f32.to_bits());
        assert_eq!(metadata.vision().mean[1].to_bits(), 0.456_f32.to_bits());
        assert_eq!(metadata.vision().mean[2].to_bits(), 0.406_f32.to_bits());
        assert_eq!(metadata.audio().sample_rate, 32_000);
        assert_eq!(metadata.audio().num_mel_bins, 128);
        assert_eq!(
            metadata.image_encoder().source_name(),
            b"mobilenetv4_conv_medium.e180_r384_in12k"
        );
        assert_eq!(
            metadata.audio_encoder().source_name(),
            b"efficientat_mn20_as"
        );
    }

    #[test]
    fn rejects_malformed_dimensions_and_encoder_pairing() {
        let invalid = HParamsInput {
            embedding_length: 0,
            image_encoder_length: 768,
            audio_encoder_length: 1024,
            image_encoder: Encoder::MobileNetV4Medium,
            audio_encoder: Encoder::EfficientAtMn20As,
            matryoshka_dimensions: &[1],
        };
        assert_eq!(HParams::try_new(invalid), Err(Error::InvalidDimensions));
        assert_eq!(
            HParams::try_new(HParamsInput {
                embedding_length: 512,
                matryoshka_dimensions: &[256, 256],
                ..invalid
            }),
            Err(Error::InvalidMatryoshkaDimensions)
        );
        assert_eq!(
            HParams::try_new(HParamsInput {
                image_encoder: Encoder::EfficientAtMn20As,
                ..invalid
            }),
            Err(Error::InvalidDimensions)
        );
        let valid = HParamsInput {
            embedding_length: 512,
            ..invalid
        };
        assert_eq!(
            HParams::try_new(HParamsInput {
                image_encoder: Encoder::EfficientAtMn20As,
                ..valid
            }),
            Err(Error::UnsupportedEncoder(Family::ImageEncoder))
        );
        assert_eq!(
            HParams::try_new(HParamsInput {
                audio_encoder: Encoder::MobileNetV4Medium,
                ..valid
            }),
            Err(Error::UnsupportedEncoder(Family::AudioEncoder))
        );
    }

    #[test]
    fn validates_family_counts_and_borrowed_bridge() {
        let metadata = hparams();
        let input = BindingInput::new(b"omniembed", &metadata, families());
        let contract = Detail.validate(input).unwrap();
        assert_eq!(Detail.load_hparams(input).unwrap(), &metadata);
        assert_eq!(contract.tensor_families().count(Family::AudioProjection), 1);
        assert_eq!(
            Detail.validate(BindingInput::new(
                b"omniembed",
                &metadata,
                TensorFamilies {
                    audio_projection: 0,
                    ..families()
                }
            )),
            Err(Error::EmptyTensorFamily(Family::AudioProjection))
        );
        assert_eq!(
            Detail.validate(BindingInput::new(b"wrong", &metadata, families())),
            Err(Error::WrongArchitecture)
        );
    }
}
