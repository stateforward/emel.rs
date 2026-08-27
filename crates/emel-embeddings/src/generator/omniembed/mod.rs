//! `OmniEmbed` metadata contract owned by the embeddings crate.

#![allow(clippy::too_many_arguments)]
#![allow(clippy::unused_self)]
#![allow(clippy::needless_pass_by_ref_mut)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::enum_variant_names)]

use core::fmt;

use emel_model::catalog::{self, Catalog};

pub const MAX_MATRYOSHKA_DIMS: usize = 8;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Error {
    InvalidArchitecture,
    InvalidDimensions,
    InvalidEncoder,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Encoder {
    MobileNetV4Medium,
    EfficientAtMn20As,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VisionPreprocessing {
    pub image_size: i32,
    pub mean: [f32; 3],
    pub std: [f32; 3],
}

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

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Contract {
    pub embedding_length: i32,
    pub image_encoder_length: i32,
    pub audio_encoder_length: i32,
    pub image_encoder: Encoder,
    pub audio_encoder: Encoder,
    pub matryoshka_dimensions: [i32; MAX_MATRYOSHKA_DIMS],
    pub matryoshka_dimension_count: u32,
    pub vision: VisionPreprocessing,
    pub audio: AudioPreprocessing,
    pub tensor_families: TensorFamilies,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TensorFamilies {
    pub text_encoder: u32,
    pub text_projection: u32,
    pub image_encoder: u32,
    pub image_projection: u32,
    pub audio_encoder: u32,
    pub audio_projection: u32,
}

/// Public owner actor for `OmniEmbed` metadata and tensor-family validation.
#[derive(Debug, Default)]
pub struct OmniEmbed {
    contract: Option<Contract>,
}

/// Typed `OmniEmbed` actor events.
pub mod event {
    use super::{Contract, Encoder, Error, TensorFamilies};

    #[derive(Clone, Copy, Debug)]
    pub struct ValidateContract<'a> {
        pub architecture: &'a [u8],
        pub embedding_length: i32,
        pub image_encoder_length: i32,
        pub audio_encoder_length: i32,
        pub image_encoder: Encoder,
        pub audio_encoder: Encoder,
        pub dimensions: &'a [i32],
        pub tensor_families: TensorFamilies,
    }

    impl<'a> ValidateContract<'a> {
        pub const fn new(
            architecture: &'a [u8],
            embedding_length: i32,
            image_encoder_length: i32,
            audio_encoder_length: i32,
            image_encoder: Encoder,
            audio_encoder: Encoder,
            dimensions: &'a [i32],
            tensor_families: TensorFamilies,
        ) -> Self {
            Self {
                architecture,
                embedding_length,
                image_encoder_length,
                audio_encoder_length,
                image_encoder,
                audio_encoder,
                dimensions,
                tensor_families,
            }
        }
    }

    pub type ContractResult = Result<Contract, Error>;
}

impl OmniEmbed {
    pub fn process_event(&mut self, event: event::ValidateContract<'_>) -> event::ContractResult {
        let contract = Contract::validate(
            event.architecture,
            event.embedding_length,
            event.image_encoder_length,
            event.audio_encoder_length,
            event.image_encoder,
            event.audio_encoder,
            event.dimensions,
            event.tensor_families,
        )?;
        self.contract = Some(contract);
        Ok(contract)
    }

    pub fn validate_families(
        &mut self,
        catalog: &mut Catalog,
        model: catalog::event::ModelIdentity,
    ) -> Result<(), Error> {
        self.contract
            .ok_or(Error::InvalidDimensions)?
            .validate_families(catalog, model)
    }
}

impl Contract {
    pub fn validate_families(
        &self,
        catalog: &mut Catalog,
        model: catalog::event::ModelIdentity,
    ) -> Result<(), Error> {
        for prefix in [
            b"text_encoder.".as_slice(),
            b"text_projection.",
            b"image_encoder.",
            b"image_projection.",
            b"audio_encoder.",
            b"audio_projection.",
        ] {
            let summary = catalog
                .process_event(catalog::event::ScanPrefix::new(model, prefix))
                .map_err(|_| Error::InvalidDimensions)?;
            if summary.count() == 0 || !summary.all_bound() || !summary.all_geometry_valid() {
                return Err(Error::InvalidDimensions);
            }
        }
        Ok(())
    }

    pub fn validate(
        architecture: &[u8],
        embedding_length: i32,
        image_encoder_length: i32,
        audio_encoder_length: i32,
        image_encoder: Encoder,
        audio_encoder: Encoder,
        dimensions: &[i32],
        tensor_families: TensorFamilies,
    ) -> Result<Self, Error> {
        if architecture != b"omniembed" {
            return Err(Error::InvalidArchitecture);
        }
        if embedding_length <= 0 || image_encoder_length <= 0 || audio_encoder_length <= 0 {
            return Err(Error::InvalidDimensions);
        }
        if dimensions.is_empty() || dimensions.len() > MAX_MATRYOSHKA_DIMS {
            return Err(Error::InvalidDimensions);
        }
        if tensor_families.text_encoder == 0
            || tensor_families.text_projection == 0
            || tensor_families.image_encoder == 0
            || tensor_families.image_projection == 0
            || tensor_families.audio_encoder == 0
            || tensor_families.audio_projection == 0
        {
            return Err(Error::InvalidDimensions);
        }
        let mut values = [0; MAX_MATRYOSHKA_DIMS];
        let mut previous = embedding_length + 1;
        for (index, &value) in dimensions.iter().enumerate() {
            if value <= 0 || value > embedding_length || value >= previous {
                return Err(Error::InvalidDimensions);
            }
            values[index] = value;
            previous = value;
        }
        let vision = match image_encoder {
            Encoder::MobileNetV4Medium => VisionPreprocessing {
                image_size: 384,
                mean: [0.485, 0.456, 0.406],
                std: [0.229, 0.224, 0.225],
            },
            Encoder::EfficientAtMn20As => return Err(Error::InvalidEncoder),
        };
        let audio = match audio_encoder {
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
            Encoder::MobileNetV4Medium => return Err(Error::InvalidEncoder),
        };
        Ok(Self {
            embedding_length,
            image_encoder_length,
            audio_encoder_length,
            image_encoder,
            audio_encoder,
            matryoshka_dimensions: values,
            matryoshka_dimension_count: dimensions.len() as u32,
            vision,
            audio,
            tensor_families,
        })
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::InvalidArchitecture => "invalid architecture",
            Self::InvalidDimensions => "invalid dimensions",
            Self::InvalidEncoder => "invalid encoder",
        })
    }
}
impl std::error::Error for Error {}

#[cfg(test)]
mod tests {
    use super::*;
    use emel_model::catalog::Catalog;
    use emel_model::catalog::event::{BindStorage, SealModel, Storage, TensorInput};

    fn catalog_with(names: &[&[u8]]) -> (Catalog, emel_model::catalog::event::ModelIdentity) {
        let bytes = names.iter().map(|name| name.len()).sum();
        let mut storage = Storage::with_capacity(names.len(), bytes, names.len()).unwrap();
        for name in names {
            storage
                .push_tensor(TensorInput::new(name, 0, 2, [4, 8, 1, 1], 128, true))
                .unwrap();
        }
        let mut catalog = Catalog::try_new().unwrap();
        catalog.process_event(BindStorage::new(storage)).unwrap();
        let model = catalog.process_event(SealModel::new()).unwrap();
        (catalog, model)
    }

    fn catalog() -> (Catalog, emel_model::catalog::event::ModelIdentity) {
        catalog_with(&[
            b"text_encoder.layer",
            b"text_projection.weight",
            b"image_encoder.layer",
            b"image_projection.weight",
            b"audio_encoder.layer",
            b"audio_projection.weight",
        ])
    }
    #[test]
    fn validates_source_contract_and_rejects_non_monotonic_dimensions() {
        let contract = Contract::validate(
            b"omniembed",
            512,
            768,
            1024,
            Encoder::MobileNetV4Medium,
            Encoder::EfficientAtMn20As,
            &[512, 256, 128],
            TensorFamilies {
                text_encoder: 1,
                text_projection: 1,
                image_encoder: 1,
                image_projection: 1,
                audio_encoder: 1,
                audio_projection: 1,
            },
        )
        .unwrap();
        assert_eq!(contract.matryoshka_dimension_count, 3);
        assert_eq!(contract.vision.image_size, 384);
        assert_eq!(contract.audio.sample_rate, 32_000);
        assert_eq!(
            Contract::validate(
                b"omniembed",
                512,
                1,
                1,
                Encoder::MobileNetV4Medium,
                Encoder::EfficientAtMn20As,
                &[256, 256],
                TensorFamilies {
                    text_encoder: 1,
                    text_projection: 1,
                    image_encoder: 1,
                    image_projection: 1,
                    audio_encoder: 1,
                    audio_projection: 1
                }
            ),
            Err(Error::InvalidDimensions)
        );
    }

    #[test]
    fn validates_all_tensor_families_through_catalog_actor() {
        let (mut catalog, model) = catalog();
        let contract = Contract::validate(
            b"omniembed",
            512,
            768,
            1024,
            Encoder::MobileNetV4Medium,
            Encoder::EfficientAtMn20As,
            &[512, 256, 128],
            TensorFamilies {
                text_encoder: 1,
                text_projection: 1,
                image_encoder: 1,
                image_projection: 1,
                audio_encoder: 1,
                audio_projection: 1,
            },
        )
        .unwrap();
        assert_eq!(contract.validate_families(&mut catalog, model), Ok(()));
    }

    #[test]
    fn rejects_missing_tensor_family_through_catalog_actor() {
        let (mut catalog, model) = catalog_with(&[
            b"text_encoder.layer",
            b"text_projection.weight",
            b"image_projection.weight",
            b"audio_encoder.layer",
            b"audio_projection.weight",
        ]);
        let contract = Contract::validate(
            b"omniembed",
            512,
            768,
            1024,
            Encoder::MobileNetV4Medium,
            Encoder::EfficientAtMn20As,
            &[512, 256, 128],
            TensorFamilies {
                text_encoder: 1,
                text_projection: 1,
                image_encoder: 1,
                image_projection: 1,
                audio_encoder: 1,
                audio_projection: 1,
            },
        )
        .unwrap();
        assert_eq!(
            contract.validate_families(&mut catalog, model),
            Err(Error::InvalidDimensions)
        );
    }
}
