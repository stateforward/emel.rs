//! `OmniEmbed` metadata contract owned by the `embeddings` crate.

#![allow(clippy::too_many_arguments)]
#![allow(clippy::unused_self)]
#![allow(clippy::needless_pass_by_ref_mut)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::enum_variant_names)]

use core::fmt;

use emel_model::catalog::{self, Catalog};

pub(crate) mod detail;

pub const MAX_MATRYOSHKA_DIMS: usize = 16;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Error {
    InvalidArchitecture,
    InvalidDimensions,
    InvalidEncoder,
    InvalidRequest,
    ModelInvalid,
    Busy,
    UnexpectedEvent,
    Catalog(catalog::event::Error),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Encoder {
    MobileNetV4Medium,
    EfficientAtMn20As,
}

impl Encoder {
    #[must_use]
    pub const fn source_name(self) -> &'static [u8] {
        match self {
            Self::MobileNetV4Medium => b"mobilenetv4_conv_medium.e180_r384_in12k",
            Self::EfficientAtMn20As => b"efficientat_mn20_as",
        }
    }
}

/// Lifecycle state of the single-writer `OmniEmbed` contract actor.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum State {
    Uninitialized,
    Validated,
    Errored,
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

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct TensorFamilies {
    pub text_encoder: u32,
    pub text_projection: u32,
    pub image_encoder: u32,
    pub image_projection: u32,
    pub audio_encoder: u32,
    pub audio_projection: u32,
}

/// Borrowed native tensor binding retained by the caller across execution.
///
/// The binding never copies model bytes. Its lifetime is tied to the immutable
/// [`emel_model::bridge::Data`] owner, and construction rejects any family with
/// missing resident payload bytes. Because the generated state-machine event
/// payloads are owned values, this borrow is consumed through an explicit
/// synchronous execution method rather than retained in generator context.
/// The method writes only to caller-owned output.
#[derive(Clone, Copy, Debug)]
pub struct NativeTensorBinding<'a> {
    data: &'a emel_model::bridge::Data,
    tensor_families: TensorFamilies,
    pub(crate) text_layers: [Option<detail::LayerTensors<'a>>; detail::MAX_TEXT_LAYERS],
    pub(crate) text_layer_count: usize,
}

impl<'a> NativeTensorBinding<'a> {
    #[must_use]
    pub const fn data(&self) -> &'a emel_model::bridge::Data {
        self.data
    }

    #[must_use]
    pub const fn tensor_families(&self) -> TensorFamilies {
        self.tensor_families
    }
}

/// Public owner actor for `OmniEmbed` metadata, lifecycle, and tensor-family validation.
#[derive(Debug)]
pub struct OmniEmbed {
    contract: Option<Contract>,
    state: State,
    families_validated: bool,
}

impl Default for OmniEmbed {
    fn default() -> Self {
        Self {
            contract: None,
            state: State::Uninitialized,
            families_validated: false,
        }
    }
}

/// Typed `OmniEmbed` actor events.
pub mod event {
    use super::{Catalog, Contract, Encoder, Error, OmniEmbed, TensorFamilies};

    mod sealed {
        pub trait Sealed {}
    }

    /// Event accepted by [`OmniEmbed`].
    pub trait Event: sealed::Sealed {
        type Output;

        #[doc(hidden)]
        fn dispatch(self, actor: &mut OmniEmbed) -> Self::Output;
    }

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

    /// Validates declared tensor families through the public catalog actor.
    #[derive(Debug)]
    pub struct ValidateFamilies<'a> {
        pub catalog: &'a mut Catalog,
        pub model: emel_model::catalog::event::ModelIdentity,
    }

    impl<'a> ValidateFamilies<'a> {
        #[must_use]
        pub const fn new(
            catalog: &'a mut Catalog,
            model: emel_model::catalog::event::ModelIdentity,
        ) -> Self {
            Self { catalog, model }
        }
    }

    #[derive(Clone, Copy, Debug, Default)]
    pub struct Visit;

    impl Visit {
        #[must_use]
        pub const fn new() -> Self {
            Self
        }
    }

    #[derive(Clone, Copy, Debug, Default)]
    pub struct Reset;

    impl Reset {
        #[must_use]
        pub const fn new() -> Self {
            Self
        }
    }

    pub type ContractResult = Result<Contract, Error>;
    pub type FamiliesResult = Result<(), Error>;

    impl sealed::Sealed for ValidateContract<'_> {}
    impl Event for ValidateContract<'_> {
        type Output = ContractResult;

        fn dispatch(self, actor: &mut OmniEmbed) -> Self::Output {
            actor.validate_contract(self)
        }
    }

    impl sealed::Sealed for ValidateFamilies<'_> {}
    impl Event for ValidateFamilies<'_> {
        type Output = FamiliesResult;

        fn dispatch(self, actor: &mut OmniEmbed) -> Self::Output {
            actor.validate_families(self.catalog, self.model)
        }
    }

    impl sealed::Sealed for Visit {}
    impl Event for Visit {
        type Output = ContractResult;

        fn dispatch(self, actor: &mut OmniEmbed) -> Self::Output {
            actor.visit()
        }
    }

    impl sealed::Sealed for Reset {}
    impl Event for Reset {
        type Output = FamiliesResult;

        fn dispatch(self, actor: &mut OmniEmbed) -> Self::Output {
            actor.reset();
            Ok(())
        }
    }
}

impl OmniEmbed {
    /// Dispatches a typed public event synchronously.
    pub fn process_event<E: event::Event>(&mut self, event: E) -> E::Output {
        event.dispatch(self)
    }

    fn validate_contract(&mut self, event: event::ValidateContract<'_>) -> event::ContractResult {
        if self.state == State::Validated {
            self.state = State::Errored;
            return Err(Error::Busy);
        }
        let contract = match Contract::validate(
            event.architecture,
            event.embedding_length,
            event.image_encoder_length,
            event.audio_encoder_length,
            event.image_encoder,
            event.audio_encoder,
            event.dimensions,
            event.tensor_families,
        ) {
            Ok(contract) => contract,
            Err(error) => {
                self.state = State::Errored;
                return Err(error);
            }
        };
        self.contract = Some(contract);
        self.families_validated = false;
        self.state = State::Validated;
        Ok(contract)
    }

    /// # Errors
    ///
    /// Returns [`Error::InvalidRequest`] unless the actor is validated, or a
    /// catalog/contract validation error when a tensor family is invalid.
    pub fn validate_families(
        &mut self,
        catalog: &mut Catalog,
        model: catalog::event::ModelIdentity,
    ) -> Result<(), Error> {
        if self.state != State::Validated {
            return Err(Error::InvalidRequest);
        }
        let result = self
            .contract
            .ok_or(Error::InvalidRequest)
            .and_then(|contract| contract.validate_families(catalog, model));
        if result.is_err() {
            self.state = State::Errored;
        } else {
            self.families_validated = true;
        }
        result
    }

    /// # Errors
    ///
    /// Returns [`Error::UnexpectedEvent`] unless the actor is validated and a
    /// contract is present.
    pub fn visit(&self) -> Result<Contract, Error> {
        if self.state != State::Validated {
            return Err(Error::UnexpectedEvent);
        }
        self.contract.ok_or(Error::UnexpectedEvent)
    }

    pub const fn reset(&mut self) {
        self.contract = None;
        self.families_validated = false;
        self.state = State::Uninitialized;
    }

    #[must_use]
    pub const fn state(&self) -> State {
        self.state
    }
    /// Binds validated metadata into the fixed generator context.
    ///
    /// The generated event payloads own their request data and cannot retain a
    /// borrow into model [`Data`]. Native text therefore uses the explicit
    /// [`Self::execute_native_text`] synchronous handoff below; callbacks never
    /// stand in for native execution. Image and audio remain backend routes.
    /// # Errors
    ///
    /// Returns [`Error::InvalidRequest`] when validation or family validation
    /// has not completed, or [`Error::InvalidDimensions`] when the contract
    /// cannot be represented by the supplied generator context.
    pub fn bind_generator(
        &self,
        context: &mut super::sm::EmbeddingsGeneratorContext,
        max_positions: usize,
    ) -> Result<(), Error> {
        if self.state != State::Validated || !self.families_validated {
            return Err(Error::InvalidRequest);
        }
        let contract = self.contract.ok_or(Error::InvalidRequest)?;
        let embedding_length =
            usize::try_from(contract.embedding_length).map_err(|_| Error::InvalidDimensions)?;
        context.configure(embedding_length, max_positions);
        if !context.model_ready || !context.scratch_ready {
            return Err(Error::InvalidDimensions);
        }
        context.image_encoder_length =
            usize::try_from(contract.image_encoder_length).map_err(|_| Error::InvalidDimensions)?;
        context.audio_encoder_length =
            usize::try_from(contract.audio_encoder_length).map_err(|_| Error::InvalidDimensions)?;
        let count = usize::try_from(contract.matryoshka_dimension_count)
            .map_err(|_| Error::InvalidDimensions)?;
        if count == 0 || count > context.matryoshka_dimensions.len() {
            return Err(Error::InvalidDimensions);
        }
        context.matryoshka_dimensions.fill(0);
        for (destination, source) in context.matryoshka_dimensions[..count]
            .iter_mut()
            .zip(contract.matryoshka_dimensions[..count].iter().copied())
        {
            *destination = usize::try_from(source).map_err(|_| Error::InvalidDimensions)?;
        }
        context.matryoshka_dimension_count = count;
        context.set_native_execution_unavailable();
        context.set_routes(
            super::sm::TextRouteKind::Encoder,
            super::sm::ImageRouteKind::Encoder,
            super::sm::AudioRouteKind::Encoder,
        );
        Ok(())
    }

    /// # Errors
    ///
    /// Returns [`Error::InvalidRequest`] when validation has not completed,
    /// [`Error::InvalidArchitecture`] for non-OmniEmbed data,
    /// [`Error::ModelInvalid`] when resident tensor bytes are missing, or
    /// [`Error::InvalidDimensions`] when tensor-family counts do not match.
    pub fn bind_native_tensors<'a>(
        &self,
        data: &'a emel_model::bridge::Data,
    ) -> Result<NativeTensorBinding<'a>, Error> {
        if self.state != State::Validated || !self.families_validated {
            return Err(Error::InvalidRequest);
        }
        if data.architecture_name() != b"omniembed" {
            return Err(Error::InvalidArchitecture);
        }
        let contract = self.contract.ok_or(Error::InvalidRequest)?;
        let expected = [
            (
                b"text_encoder.".as_slice(),
                contract.tensor_families.text_encoder,
            ),
            (
                b"text_projection.",
                contract.tensor_families.text_projection,
            ),
            (b"image_encoder.", contract.tensor_families.image_encoder),
            (
                b"image_projection.",
                contract.tensor_families.image_projection,
            ),
            (b"audio_encoder.", contract.tensor_families.audio_encoder),
            (
                b"audio_projection.",
                contract.tensor_families.audio_projection,
            ),
        ];
        let mut counts = TensorFamilies::default();
        for index in 0..data.tensor_count() {
            let tensor = data.tensor(index).ok_or(Error::ModelInvalid)?;
            if tensor.bytes().is_none() {
                return Err(Error::ModelInvalid);
            }
            let name = tensor.name();
            if let Some((prefix, _)) = expected.iter().find(|(prefix, _)| name.starts_with(prefix))
            {
                let slot = if *prefix == b"text_encoder." {
                    &mut counts.text_encoder
                } else if *prefix == b"text_projection." {
                    &mut counts.text_projection
                } else if *prefix == b"image_encoder." {
                    &mut counts.image_encoder
                } else if *prefix == b"image_projection." {
                    &mut counts.image_projection
                } else if *prefix == b"audio_encoder." {
                    &mut counts.audio_encoder
                } else {
                    &mut counts.audio_projection
                };
                *slot = slot.checked_add(1).ok_or(Error::InvalidDimensions)?;
            }
        }
        if counts != contract.tensor_families {
            return Err(Error::InvalidDimensions);
        }
        let (text_layers, text_layer_count) =
            detail::bind_layer_inventory(data).map_err(|error| match error {
                detail::Error::UnsupportedTensor | detail::Error::Capacity => {
                    Error::InvalidDimensions
                }
                detail::Error::InvalidRequest | detail::Error::ModelInvalid => Error::ModelInvalid,
            })?;
        Ok(NativeTensorBinding {
            data,
            tensor_families: counts,
            text_layers,
            text_layer_count,
        })
    }
    /// Executes the validated native `OmniEmbed` text path for one synchronous call.
    ///
    /// The model data is borrowed only while this method binds and dispatches;
    /// the generated event machine does not retain that borrow. This is the
    /// public owner/generator handoff for native text. Image and audio remain
    /// explicit backend routes.
    ///
    /// # Errors
    ///
    /// Returns [`sm::EmbeddingsGeneratorStatus::InvalidRequest`] when the owner
    /// has not completed validation, [`sm::EmbeddingsGeneratorStatus::ModelInvalid`]
    /// when model tensors cannot be bound, or the generator's typed execution
    /// error for invalid tokens or insufficient output capacity.
    pub fn execute_native_text(
        &self,
        generator: &mut super::sm::EmbeddingsGeneratorActor,
        data: &'_ emel_model::bridge::Data,
        token_ids: &[i32],
        output: &mut [f32],
    ) -> Result<usize, super::sm::EmbeddingsGeneratorStatus> {
        let binding = self
            .bind_native_tensors(data)
            .map_err(|error| match error {
                Error::InvalidArchitecture | Error::InvalidDimensions | Error::ModelInvalid => {
                    super::sm::EmbeddingsGeneratorStatus::ModelInvalid
                }
                Error::InvalidRequest
                | Error::InvalidEncoder
                | Error::Busy
                | Error::UnexpectedEvent
                | Error::Catalog(_) => super::sm::EmbeddingsGeneratorStatus::InvalidRequest,
            })?;
        generator.process_native_text(&binding, token_ids, output)
    }
}

impl Contract {
    /// # Errors
    ///
    /// Returns [`Error::Catalog`] when catalog scanning fails, or
    /// [`Error::InvalidDimensions`] when a scanned family does not match the
    /// contract.
    pub fn validate_families(
        &self,
        catalog: &mut Catalog,
        model: catalog::event::ModelIdentity,
    ) -> Result<(), Error> {
        let families = [
            (
                b"text_encoder.".as_slice(),
                self.tensor_families.text_encoder,
            ),
            (b"text_projection.", self.tensor_families.text_projection),
            (b"image_encoder.", self.tensor_families.image_encoder),
            (b"image_projection.", self.tensor_families.image_projection),
            (b"audio_encoder.", self.tensor_families.audio_encoder),
            (b"audio_projection.", self.tensor_families.audio_projection),
        ];
        for (prefix, expected_count) in families {
            let summary = catalog
                .process_event(catalog::event::ScanPrefix::new(model, prefix))
                .map_err(Error::Catalog)?;
            if summary.count() != expected_count
                || !summary.all_bound()
                || !summary.all_geometry_valid()
            {
                return Err(Error::InvalidDimensions);
            }
        }
        Ok(())
    }

    /// # Errors
    ///
    /// Returns [`Error::InvalidArchitecture`] for a non-OmniEmbed architecture,
    /// [`Error::InvalidDimensions`] for invalid dimensions or tensor families,
    /// or [`Error::InvalidEncoder`] for an unsupported encoder combination.
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
        let mut previous = embedding_length
            .checked_add(1)
            .ok_or(Error::InvalidDimensions)?;
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
        match self {
            Self::InvalidArchitecture => f.write_str("invalid architecture"),
            Self::InvalidDimensions => f.write_str("invalid dimensions"),
            Self::InvalidEncoder => f.write_str("invalid encoder"),
            Self::InvalidRequest => f.write_str("invalid request"),
            Self::ModelInvalid => f.write_str("model invalid"),
            Self::Busy => f.write_str("omniembed actor busy"),
            Self::UnexpectedEvent => f.write_str("unexpected omniembed event"),
            Self::Catalog(error) => error.fmt(f),
        }
    }
}
impl std::error::Error for Error {}

#[cfg(test)]
mod tests {
    use super::*;
    use emel_model::bridge::Data;
    use emel_model::bridge::{
        OmniEmbedDataInput, OmniEmbedEncoder, OmniEmbedHParams, OmniEmbedHParamsInput,
        OmniEmbedTensorFamilies, TensorBinding, TensorInput, TensorMetadata, TensorMetadataInput,
    };
    use emel_model::catalog::Catalog;
    use emel_model::catalog::event::{
        BindStorage, SealModel, Storage, TensorInput as CatalogTensorInput,
    };
    use emel_tensor::dtype::SerializedType;

    fn f32_bytes(count: usize, value: f32) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(count * 4);
        for _ in 0..count {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        bytes
    }

    fn identity_bytes(rows: usize, columns: usize) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(rows * columns * 4);
        for row in 0..rows {
            for column in 0..columns {
                let value = f32::from(u8::from(row == column));
                bytes.extend_from_slice(&value.to_le_bytes());
            }
        }
        bytes
    }

    type TextTensorSpec = (&'static [u8], u32, [u64; 4]);

    const RESIDENT_TEXT_SPECS: &[TextTensorSpec] = &[
        (
            b"text_encoder.0.auto_model.embeddings.word_embeddings.weight",
            2,
            [12, 2, 1, 1],
        ),
        (
            b"text_encoder.0.auto_model.embeddings.position_embeddings.weight",
            2,
            [12, 1, 1, 1],
        ),
        (
            b"text_encoder.0.auto_model.embeddings.token_type_embeddings.weight",
            2,
            [12, 1, 1, 1],
        ),
        (
            b"text_encoder.0.auto_model.embeddings.LayerNorm.weight",
            1,
            [12, 1, 1, 1],
        ),
        (
            b"text_encoder.0.auto_model.embeddings.LayerNorm.bias",
            1,
            [12, 1, 1, 1],
        ),
        (b"text_encoder.2.linear.weight", 2, [12, 12, 1, 1]),
        (b"text_encoder.2.linear.bias", 1, [12, 1, 1, 1]),
        (b"text_projection.expand.0.weight", 2, [12, 12, 1, 1]),
        (b"text_projection.expand.0.bias", 1, [12, 1, 1, 1]),
        (b"text_projection.expand.2.weight", 1, [12, 1, 1, 1]),
        (b"text_projection.expand.2.bias", 1, [12, 1, 1, 1]),
        (
            b"text_projection.residual_blocks.0.0.weight",
            2,
            [12, 12, 1, 1],
        ),
        (
            b"text_projection.residual_blocks.0.0.bias",
            1,
            [12, 1, 1, 1],
        ),
        (
            b"text_projection.residual_blocks.0.2.weight",
            1,
            [12, 1, 1, 1],
        ),
        (
            b"text_projection.residual_blocks.0.2.bias",
            1,
            [12, 1, 1, 1],
        ),
        (b"text_projection.project.weight", 2, [12, 4, 1, 1]),
        (b"text_projection.project.bias", 1, [4, 1, 1, 1]),
        (b"image_encoder.weight", 1, [1, 1, 1, 1]),
        (b"image_projection.weight", 1, [1, 1, 1, 1]),
        (b"audio_encoder.weight", 1, [1, 1, 1, 1]),
        (b"audio_projection.weight", 1, [1, 1, 1, 1]),
        (
            b"text_encoder.0.auto_model.encoder.layer.0.attention.self.query.weight",
            2,
            [12, 12, 1, 1],
        ),
        (
            b"text_encoder.0.auto_model.encoder.layer.0.attention.self.query.bias",
            1,
            [12, 1, 1, 1],
        ),
        (
            b"text_encoder.0.auto_model.encoder.layer.0.attention.self.key.weight",
            2,
            [12, 12, 1, 1],
        ),
        (
            b"text_encoder.0.auto_model.encoder.layer.0.attention.self.key.bias",
            1,
            [12, 1, 1, 1],
        ),
        (
            b"text_encoder.0.auto_model.encoder.layer.0.attention.self.value.weight",
            2,
            [12, 12, 1, 1],
        ),
        (
            b"text_encoder.0.auto_model.encoder.layer.0.attention.self.value.bias",
            1,
            [12, 1, 1, 1],
        ),
        (
            b"text_encoder.0.auto_model.encoder.layer.0.attention.output.dense.weight",
            2,
            [12, 12, 1, 1],
        ),
        (
            b"text_encoder.0.auto_model.encoder.layer.0.attention.output.dense.bias",
            1,
            [12, 1, 1, 1],
        ),
        (
            b"text_encoder.0.auto_model.encoder.layer.0.attention.output.LayerNorm.weight",
            1,
            [12, 1, 1, 1],
        ),
        (
            b"text_encoder.0.auto_model.encoder.layer.0.attention.output.LayerNorm.bias",
            1,
            [12, 1, 1, 1],
        ),
        (
            b"text_encoder.0.auto_model.encoder.layer.0.intermediate.dense.weight",
            2,
            [12, 48, 1, 1],
        ),
        (
            b"text_encoder.0.auto_model.encoder.layer.0.intermediate.dense.bias",
            1,
            [48, 1, 1, 1],
        ),
        (
            b"text_encoder.0.auto_model.encoder.layer.0.output.dense.weight",
            2,
            [48, 12, 1, 1],
        ),
        (
            b"text_encoder.0.auto_model.encoder.layer.0.output.dense.bias",
            1,
            [12, 1, 1, 1],
        ),
        (
            b"text_encoder.0.auto_model.encoder.layer.0.output.LayerNorm.weight",
            1,
            [12, 1, 1, 1],
        ),
        (
            b"text_encoder.0.auto_model.encoder.layer.0.output.LayerNorm.bias",
            1,
            [12, 1, 1, 1],
        ),
    ];

    fn resident_text_payload((name, rank, dimensions): TextTensorSpec) -> Vec<u8> {
        let count = dimensions[..usize::try_from(rank).unwrap()]
            .iter()
            .copied()
            .map(|value| usize::try_from(value).unwrap())
            .product();
        if name == RESIDENT_TEXT_SPECS[0].0 {
            (0..count)
                .map(|index| {
                    let bounded_index = u16::try_from(index + 1).expect("tiny fixture index");
                    0.1 * (f32::from(bounded_index) + 1.0)
                })
                .flat_map(f32::to_le_bytes)
                .collect()
        } else if matches!(
            name,
            b"text_encoder.2.linear.weight"
                | b"text_projection.expand.0.weight"
                | b"text_projection.residual_blocks.0.0.weight"
                | b"text_projection.project.weight"
        ) {
            identity_bytes(
                usize::try_from(dimensions[1]).unwrap(),
                usize::try_from(dimensions[0]).unwrap(),
            )
        } else if name.ends_with(b"LayerNorm.weight")
            || matches!(
                name,
                b"text_encoder.0.auto_model.embeddings.LayerNorm.weight"
                    | b"text_projection.expand.2.weight"
                    | b"text_projection.residual_blocks.0.2.weight"
            )
        {
            f32_bytes(count, 1.0)
        } else {
            f32_bytes(count, 0.0)
        }
    }

    fn resident_text_tensors(payloads: &[Vec<u8>]) -> Vec<TensorInput<'_>> {
        RESIDENT_TEXT_SPECS
            .iter()
            .copied()
            .zip(payloads)
            .map(|((name, rank, dimensions), bytes)| {
                TensorInput::with_bytes(
                    name,
                    TensorMetadata::new(TensorMetadataInput {
                        tensor_type: SerializedType::F32,
                        dimension_count: rank,
                        dimensions,
                        data_offset: 0,
                        file_offset: 0,
                        data_size: u64::try_from(bytes.len()).unwrap(),
                        file_index: 0,
                        storage: Some(TensorBinding::new(
                            0,
                            0,
                            u64::try_from(bytes.len()).unwrap(),
                        )),
                    }),
                    bytes,
                )
            })
            .collect()
    }

    fn resident_text_data() -> (Data, Vec<&'static [u8]>) {
        let payloads: Vec<_> = RESIDENT_TEXT_SPECS
            .iter()
            .copied()
            .map(resident_text_payload)
            .collect();
        let tensors = resident_text_tensors(&payloads);
        let data = Data::try_from_omniembed(OmniEmbedDataInput {
            architecture: b"omniembed",
            hparams: OmniEmbedHParams::try_new(OmniEmbedHParamsInput {
                embedding_length: 4,
                image_encoder_length: 2,
                audio_encoder_length: 2,
                image_encoder: OmniEmbedEncoder::MobileNetV4Medium,
                audio_encoder: OmniEmbedEncoder::EfficientAtMn20As,
                matryoshka_dimensions: &[2, 1],
            })
            .unwrap(),
            tensor_families: OmniEmbedTensorFamilies {
                text_encoder: 23,
                text_projection: 10,
                image_encoder: 1,
                image_projection: 1,
                audio_encoder: 1,
                audio_projection: 1,
            },
            tensors: &tensors,
        })
        .unwrap();
        let names = RESIDENT_TEXT_SPECS
            .iter()
            .map(|(name, _, _)| *name)
            .collect();
        (data, names)
    }

    fn catalog_with(names: &[&[u8]]) -> (Catalog, emel_model::catalog::event::ModelIdentity) {
        let bytes = names.iter().map(|name| name.len()).sum();
        let mut storage = Storage::with_capacity(names.len(), bytes, names.len()).unwrap();
        for name in names {
            storage
                .push_tensor(CatalogTensorInput::new(name, 0, 2, [4, 8, 1, 1], 128, true))
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
    #[test]
    fn rejects_embedding_length_overflow_through_public_validation_boundary() {
        assert_eq!(
            Contract::validate(
                b"omniembed",
                i32::MAX,
                768,
                1024,
                Encoder::MobileNetV4Medium,
                Encoder::EfficientAtMn20As,
                &[i32::MAX],
                TensorFamilies {
                    text_encoder: 1,
                    text_projection: 1,
                    image_encoder: 1,
                    image_projection: 1,
                    audio_encoder: 1,
                    audio_projection: 1,
                },
            ),
            Err(Error::InvalidDimensions)
        );
    }

    #[test]
    fn actor_events_are_synchronous_and_state_is_inspectable() {
        let mut actor = OmniEmbed::default();
        assert_eq!(actor.state(), State::Uninitialized);
        assert_eq!(
            actor.process_event(event::Visit::new()),
            Err(Error::UnexpectedEvent)
        );
        assert_eq!(actor.state(), State::Uninitialized);

        let contract = actor
            .process_event(event::ValidateContract::new(
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
            ))
            .unwrap();
        assert_eq!(actor.state(), State::Validated);
        assert_eq!(actor.process_event(event::Visit::new()), Ok(contract));

        assert_eq!(
            actor.process_event(event::ValidateContract::new(
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
            )),
            Err(Error::Busy)
        );
        assert_eq!(actor.state(), State::Errored);
        assert_eq!(
            actor.process_event(event::Visit::new()),
            Err(Error::UnexpectedEvent)
        );
        assert_eq!(actor.process_event(event::Reset::new()), Ok(()));
        assert_eq!(actor.state(), State::Uninitialized);
    }

    #[test]
    fn binds_validated_contract_into_fixed_generator_context() {
        let (mut catalog, model) = catalog();
        let mut actor = OmniEmbed::default();
        actor
            .process_event(event::ValidateContract::new(
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
            ))
            .unwrap();
        actor
            .process_event(event::ValidateFamilies::new(&mut catalog, model))
            .unwrap();
        let mut context = super::super::sm::EmbeddingsGeneratorContext::default();
        actor.bind_generator(&mut context, 2048).unwrap();
        assert!(context.model_ready);
        assert!(context.scratch_ready);
        assert_eq!(context.embedding_length, 512);
        assert_eq!(context.image_encoder_length, 768);
        assert_eq!(context.audio_encoder_length, 1024);
        assert_eq!(context.matryoshka_dimension_count, 3);
        assert_eq!(context.matryoshka_dimensions[..3], [512, 256, 128]);
        assert!(!context.native_execution_ready());
    }

    #[test]
    fn binding_rejects_unvalidated_families_and_capacity_overflow() {
        let (_catalog, _model) = catalog();
        let mut actor = OmniEmbed::default();
        let mut context = super::super::sm::EmbeddingsGeneratorContext::default();
        assert_eq!(
            actor.bind_generator(&mut context, 2048),
            Err(Error::InvalidRequest)
        );
        let (mut catalog, model) = catalog();
        actor
            .process_event(event::ValidateContract::new(
                b"omniembed",
                512,
                768,
                1024,
                Encoder::MobileNetV4Medium,
                Encoder::EfficientAtMn20As,
                &[512],
                TensorFamilies {
                    text_encoder: 1,
                    text_projection: 1,
                    image_encoder: 1,
                    image_projection: 1,
                    audio_encoder: 1,
                    audio_projection: 1,
                },
            ))
            .unwrap();
        actor
            .process_event(event::ValidateFamilies::new(&mut catalog, model))
            .unwrap();
        assert_eq!(
            actor.bind_generator(&mut context, super::super::sm::MAX_TOKEN_POSITIONS + 1),
            Err(Error::InvalidDimensions)
        );
    }

    #[test]
    fn family_binding_event_rejects_uninitialized_actor() {
        let (mut catalog, model) = catalog();
        assert_eq!(
            OmniEmbed::default().process_event(event::ValidateFamilies::new(&mut catalog, model)),
            Err(Error::InvalidRequest)
        );
    }
    #[test]
    fn encoder_names_match_pinned_source() {
        assert_eq!(
            Encoder::MobileNetV4Medium.source_name(),
            b"mobilenetv4_conv_medium.e180_r384_in12k"
        );
        assert_eq!(
            Encoder::EfficientAtMn20As.source_name(),
            b"efficientat_mn20_as"
        );
    }

    #[test]
    fn native_text_boundary_rejects_invalid_request_and_unbound_model() {
        let data = Data::try_new().unwrap();
        let binding = NativeTensorBinding {
            data: &data,
            tensor_families: TensorFamilies::default(),
            text_layers: [None; detail::MAX_TEXT_LAYERS],
            text_layer_count: 0,
        };
        let mut actor = super::super::sm::EmbeddingsGeneratorActor::new();
        let mut output = [0.0_f32; 1];

        assert_eq!(
            actor.process_native_text(&binding, &[], &mut output),
            Err(super::super::sm::EmbeddingsGeneratorStatus::InvalidRequest)
        );
        assert_eq!(
            actor.process_native_text(&binding, &[0], &mut output),
            Err(super::super::sm::EmbeddingsGeneratorStatus::ModelInvalid)
        );
        assert!(matches!(
            detail::bind_layer_inventory(&data),
            Err(detail::Error::ModelInvalid)
        ));
    }
    #[test]
    fn native_text_binds_and_executes_complete_one_layer_model() {
        let (data, names) = resident_text_data();
        let (mut catalog, model) = catalog_with(&names);
        let mut actor = OmniEmbed::default();
        actor
            .process_event(event::ValidateContract::new(
                b"omniembed",
                4,
                2,
                2,
                Encoder::MobileNetV4Medium,
                Encoder::EfficientAtMn20As,
                &[2, 1],
                TensorFamilies {
                    text_encoder: 23,
                    text_projection: 10,
                    image_encoder: 1,
                    image_projection: 1,
                    audio_encoder: 1,
                    audio_projection: 1,
                },
            ))
            .unwrap();
        actor
            .process_event(event::ValidateFamilies::new(&mut catalog, model))
            .unwrap();
        let mut generator = super::super::sm::EmbeddingsGeneratorActor::new();
        let mut output = [0.0_f32; 4];
        let binding = actor.bind_native_tensors(&data).unwrap();
        let direct = detail::execute_text(&binding, &[0], &mut output);
        assert_eq!(direct, Ok(4));
        assert_eq!(
            actor.execute_native_text(&mut generator, &data, &[0], &mut output),
            Ok(4)
        );
        assert!(output.iter().all(|value| value.is_finite()));
        let norm = output.iter().map(|value| value * value).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1.0e-5);
    }
}
