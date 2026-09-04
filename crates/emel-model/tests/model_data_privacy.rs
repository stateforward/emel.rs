//! External model-data ownership boundary checks.

use allocation_counter as _;
use emel_io as _;
use emel_model as _;
use emel_tensor as _;
use sml as _;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

fn check_project(case: &str, source: &str) -> Output {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let project = root
        .join("../../.artifacts/model-data-privacy")
        .join(format!("{case}-{}", std::process::id()));
    let _ = fs::remove_dir_all(&project);
    fs::create_dir_all(project.join("src")).unwrap();
    fs::write(
        project.join("Cargo.toml"),
        format!(
            "[package]\nname = \"model-data-privacy-{case}\"\nversion = \"0.0.0\"\nedition = \"2024\"\n\
             [dependencies]\nemel-model = {{ path = \"{}\", default-features = false }}\n\
             [workspace]\n",
            cargo_path(&root)
        ),
    )
    .unwrap();
    fs::write(project.join("src/main.rs"), source).unwrap();
    let output = Command::new(env!("CARGO"))
        .args(["check", "--offline", "--quiet"])
        .env(
            "CARGO_TARGET_DIR",
            root.join("../../target/model-data-privacy"),
        )
        .current_dir(&project)
        .output()
        .unwrap();
    let _ = fs::remove_dir_all(project);
    output
}

fn cargo_path(path: &Path) -> String {
    path.display().to_string().replace('\\', "\\\\")
}

#[test]
fn model_data_bridge_is_public_but_raw_storage_remains_private() {
    let valid = check_project(
        "public",
        "use emel_model::bridge::Data;\n\
         fn main() {\n\
             let data = Data::try_new().unwrap();\n\
             let _ = data.mimi_binding_input();\n\
             let _ = data.architecture_name();\n\
         }",
    );
    assert!(
        valid.status.success(),
        "{}",
        String::from_utf8_lossy(&valid.stderr)
    );

    let invalid = check_project(
        "private-fields",
        "use emel_model::bridge::Data;\n\
         fn main() {\n\
             let data = Data::try_new().unwrap();\n\
             let _ = data.tensors;\n\
         }",
    );
    assert!(!invalid.status.success());
    let stderr = String::from_utf8_lossy(&invalid.stderr);
    assert!(stderr.contains("private"), "{stderr}");
    assert!(stderr.contains("tensors"), "{stderr}");
}

use emel_model::bridge::{
    Data, DataError, MimiDataInput, MimiHParams, MimiHParamsInput, MoshiComponent, TensorBinding,
    TensorInput, TensorMetadata, TensorMetadataInput, TensorView,
};
use emel_tensor::dtype::SerializedType;

fn valid_mimi_hparams() -> MimiHParams {
    MimiHParams::try_new(MimiHParamsInput {
        sample_rate: 24_000,
        frame_rate: 12.5,
        n_q: 2,
        card: 32,
        dim: 16,
        semantic_n_q: 1,
        codebook_dim: 8,
        transformer_num_layers: 2,
        transformer_num_heads: 2,
        transformer_context: 8,
        transformer_max_period: 1_000,
    })
    .unwrap()
}

const fn resident_metadata(data_size: u64) -> TensorMetadata {
    TensorMetadata::new(TensorMetadataInput {
        tensor_type: SerializedType::F32,
        dimension_count: 1,
        dimensions: [4, 1, 1, 1],
        data_offset: 8,
        file_offset: 16,
        data_size,
        file_index: 2,
        storage: Some(TensorBinding::new(2, 16, data_size)),
    })
}

#[test]
fn tensor_metadata_and_input_constructors_preserve_ownership_contract() {
    let binding = TensorBinding::new(7, 123, 456);
    assert_eq!(binding.split_index(), 7);
    assert_eq!(binding.offset(), 123);
    assert_eq!(binding.length(), 456);

    let metadata = TensorMetadata::new(TensorMetadataInput {
        tensor_type: SerializedType::F16,
        dimension_count: 2,
        dimensions: [3, 5, 1, 1],
        data_offset: 32,
        file_offset: 64,
        data_size: 30,
        file_index: 4,
        storage: Some(binding),
    });
    assert_eq!(metadata.tensor_type(), SerializedType::F16);
    assert_eq!(metadata.dimension_count(), 2);
    assert_eq!(metadata.dimensions(), [3, 5, 1, 1]);
    assert_eq!(metadata.data_offset(), 32);
    assert_eq!(metadata.file_offset(), 64);
    assert_eq!(metadata.data_size(), 30);
    assert_eq!(metadata.file_index(), 4);
    assert_eq!(metadata.storage(), Some(binding));

    let nonresident = TensorInput::new(b"nonresident", metadata);
    assert_eq!(nonresident.name(), b"nonresident");
    assert_eq!(nonresident.metadata().storage(), None);
    assert_eq!(nonresident.bytes(), None);

    let payload = [9_u8; 30];
    let resident = TensorInput::with_bytes(b"resident", metadata, &payload);
    assert_eq!(resident.name(), b"resident");
    assert_eq!(resident.metadata(), metadata);
    assert_eq!(resident.bytes(), Some(&payload[..]));
}

#[test]
fn data_construction_copies_resident_storage_and_exposes_checked_views() {
    let mut payload = vec![9_u8; 16];
    let metadata = resident_metadata(payload.len() as u64);
    let tensors = [TensorInput::with_bytes(b"mimi.weight", metadata, &payload)];
    let data = Data::try_from_mimi(MimiDataInput {
        hparams: valid_mimi_hparams(),
        tensors: &tensors,
    })
    .unwrap();
    payload.fill(3);

    assert_eq!(data.architecture_name(), b"moshi");
    assert_eq!(data.component(), MoshiComponent::Mimi);
    assert_eq!(data.tensor_count(), 1);
    assert_eq!(data.weights_size(), 32);
    assert_eq!(data.weights_split_count(), 3);
    assert_eq!(data.mimi_hparams().unwrap().frame_samples(), Some(1_920));
    assert!(data.tensor(1).is_none());
    let view = data.tensor_named(b"mimi.weight").unwrap();
    assert_eq!(view.name(), b"mimi.weight");
    assert_eq!(view.metadata(), Some(metadata));
    assert_eq!(view.byte_view(), Some(&[9_u8; 16][..]));
    assert_eq!(view.bytes(), Some(&[9_u8; 16][..]));
    assert!(data.tensor_named(b"missing").is_none());

    let binding = data.mimi_binding_input();
    assert_eq!(binding.tensor_count(), 1);
    assert_eq!(binding.tensor(0).unwrap().name(), b"mimi.weight");
    assert!(binding.tensor(1).is_none());
}

#[test]
fn data_defaults_are_stable_through_public_accessors() {
    let data = Data::try_new().unwrap();
    assert_eq!(data.architecture_name(), b"");
    assert_eq!(data.component(), MoshiComponent::None);
    assert_eq!(data.tensor_count(), 0);
    assert_eq!(data.weights_size(), 0);
    assert_eq!(data.weights_split_count(), 1);
    assert!(data.mimi_hparams().is_none());
    assert!(data.tensor(0).is_none());
    assert!(data.tensor_named(b"anything").is_none());
    assert_eq!(data.mimi_binding_input().architecture_name(), b"");
}

#[test]
#[allow(clippy::too_many_lines)]
fn public_construction_classifies_tensor_and_fixed_capacity_boundaries() {
    let valid_hparams = valid_mimi_hparams();
    let payload = [0_u8; 16];

    let mut zero_dimension = resident_metadata(16);
    zero_dimension = TensorMetadata::new(TensorMetadataInput {
        dimensions: [0, 1, 1, 1],
        ..TensorMetadataInput {
            tensor_type: zero_dimension.tensor_type(),
            dimension_count: zero_dimension.dimension_count(),
            dimensions: zero_dimension.dimensions(),
            data_offset: zero_dimension.data_offset(),
            file_offset: zero_dimension.file_offset(),
            data_size: zero_dimension.data_size(),
            file_index: zero_dimension.file_index(),
            storage: zero_dimension.storage(),
        }
    });
    assert_eq!(
        Data::try_from_mimi(MimiDataInput {
            hparams: valid_hparams,
            tensors: &[TensorInput::with_bytes(b"zero", zero_dimension, &payload)],
        }),
        Err(DataError::InvalidTensor)
    );

    let inactive_dimension = TensorMetadata::new(TensorMetadataInput {
        tensor_type: SerializedType::F32,
        dimension_count: 1,
        dimensions: [4, 2, 1, 1],
        data_offset: 0,
        file_offset: 0,
        data_size: 16,
        file_index: 0,
        storage: None,
    });
    assert_eq!(
        Data::try_from_mimi(MimiDataInput {
            hparams: valid_mimi_hparams(),
            tensors: &[TensorInput::with_bytes(
                b"inactive",
                inactive_dimension,
                &payload
            )],
        }),
        Err(DataError::InvalidTensor)
    );

    let wrong_storage = TensorMetadata::new(TensorMetadataInput {
        tensor_type: SerializedType::F32,
        dimension_count: 1,
        dimensions: [4, 1, 1, 1],
        data_offset: 0,
        file_offset: 0,
        data_size: 16,
        file_index: 0,
        storage: Some(TensorBinding::new(1, 0, 16)),
    });
    assert_eq!(
        Data::try_from_mimi(MimiDataInput {
            hparams: valid_mimi_hparams(),
            tensors: &[TensorInput::with_bytes(b"storage", wrong_storage, &payload)],
        }),
        Err(DataError::InvalidTensor)
    );

    let too_many = vec![
        TensorInput::with_bytes(
            b"tensor",
            TensorMetadata::new(TensorMetadataInput {
                tensor_type: SerializedType::F32,
                dimension_count: 1,
                dimensions: [4, 1, 1, 1],
                data_offset: 0,
                file_offset: 0,
                data_size: 16,
                file_index: 0,
                storage: None,
            }),
            &payload,
        );
        65_537
    ];
    assert_eq!(
        Data::try_from_mimi(MimiDataInput {
            hparams: valid_mimi_hparams(),
            tensors: &too_many,
        }),
        Err(DataError::TooManyTensors)
    );

    let oversized_name = vec![b'n'; 4_194_305];
    let metadata = TensorMetadata::new(TensorMetadataInput {
        tensor_type: SerializedType::F32,
        dimension_count: 1,
        dimensions: [4, 1, 1, 1],
        data_offset: 0,
        file_offset: 0,
        data_size: 16,
        file_index: 0,
        storage: None,
    });
    assert_eq!(
        Data::try_from_mimi(MimiDataInput {
            hparams: valid_mimi_hparams(),
            tensors: &[TensorInput::with_bytes(&oversized_name, metadata, &payload)],
        }),
        Err(DataError::NameCapacity)
    );
}

#[test]
fn mimi_hparams_public_validation_classifies_boundary_failures() {
    let base = MimiHParamsInput {
        sample_rate: 24_000,
        frame_rate: 12.5,
        n_q: 2,
        card: 32,
        dim: 16,
        semantic_n_q: 1,
        codebook_dim: 8,
        transformer_num_layers: 2,
        transformer_num_heads: 2,
        transformer_context: 8,
        transformer_max_period: 1_000,
    };
    assert_eq!(
        MimiHParams::try_new(MimiHParamsInput {
            sample_rate: 0,
            ..base
        }),
        Err(emel_model::bridge::MimiHParamsError::NonPositive)
    );
    assert_eq!(
        MimiHParams::try_new(MimiHParamsInput {
            frame_rate: f32::NAN,
            ..base
        }),
        Err(emel_model::bridge::MimiHParamsError::InvalidFrameRate)
    );
    assert_eq!(
        MimiHParams::try_new(MimiHParamsInput {
            semantic_n_q: 2,
            ..base
        }),
        Err(emel_model::bridge::MimiHParamsError::InvalidCodebookPartition)
    );
    assert_eq!(
        MimiHParams::try_new(MimiHParamsInput {
            transformer_num_heads: 3,
            ..base
        }),
        Err(emel_model::bridge::MimiHParamsError::InvalidHeadGeometry)
    );
    assert_eq!(
        MimiHParams::try_new(MimiHParamsInput {
            dim: 6,
            transformer_num_heads: 2,
            ..base
        }),
        Err(emel_model::bridge::MimiHParamsError::OddHeadDimension)
    );

    let hparams = MimiHParams::try_new(base).unwrap();
    assert_eq!(hparams.sample_rate(), 24_000);
    assert_eq!(hparams.frame_rate().to_bits(), 12.5_f32.to_bits());
    assert_eq!(hparams.n_q(), 2);
    assert_eq!(hparams.card(), 32);
    assert_eq!(hparams.dim(), 16);
    assert_eq!(hparams.semantic_n_q(), 1);
    assert_eq!(hparams.codebook_dim(), 8);
    assert_eq!(hparams.transformer_num_layers(), 2);
    assert_eq!(hparams.transformer_num_heads(), 2);
    assert_eq!(hparams.transformer_context(), 8);
    assert_eq!(hparams.transformer_max_period(), 1_000);
    assert_eq!(hparams.frame_samples(), Some(1_920));
    assert!(hparams.validate().is_ok());
}

const fn seeded_moshi_array() -> [i32; 64] {
    let mut values = [0; 64];
    values[1] = 1;
    values[2] = 2;
    values
}

const fn valid_moshi_lm_input() -> emel_model::bridge::MoshiLmHParamsInput {
    emel_model::bridge::MoshiLmHParamsInput {
        card: 32,
        n_q: 2,
        dep_q: 2,
        inference_dep_q: 2,
        text_card: 32,
        text_padding_id: 0,
        dim: 16,
        num_layers: 2,
        num_heads: 2,
        context: 8,
        max_period: 1_000,
        dim_feedforward: 32,
        depformer_dim: 16,
        depformer_num_heads: 2,
        depformer_num_layers: 2,
        depformer_dim_feedforward: 32,
        depformer_context: 8,
        depformer_max_period: 1_000,
        depformer_low_rank_embeddings: 1,
        extra_heads_num_heads: 2,
        inference_pre_text_silence_frames: 0,
        inference_post_text_silence_frames: 0,
        delay_count: 3,
        inference_prompt_token_count: 3,
        depformer_weight_schedule_count: 3,
        delays: seeded_moshi_array(),
        inference_prompt_tokens: seeded_moshi_array(),
        depformer_weight_schedule: seeded_moshi_array(),
        causal: true,
        cross_attention: false,
        demux_second_stream: false,
        depformer_multi_linear: false,
        depformer_weights_per_step: false,
    }
}

const fn tensor_with_shape<'a>(
    name: &'static [u8],
    tensor_type: SerializedType,
    dimensions: [u64; 4],
    payload: &'a [u8],
) -> TensorInput<'a> {
    let dimension_count = if dimensions[3] != 1 {
        4
    } else if dimensions[2] != 1 {
        3
    } else if dimensions[1] != 1 {
        2
    } else {
        1
    };
    tensor_with_rank(name, tensor_type, dimensions, dimension_count, payload)
}

const fn tensor_with_rank<'a>(
    name: &'static [u8],
    tensor_type: SerializedType,
    dimensions: [u64; 4],
    dimension_count: u32,
    payload: &'a [u8],
) -> TensorInput<'a> {
    TensorInput::with_bytes(
        name,
        TensorMetadata::new(TensorMetadataInput {
            tensor_type,
            dimension_count,
            dimensions,
            data_offset: 0,
            file_offset: 0,
            data_size: payload.len() as u64,
            file_index: 0,
            storage: Some(TensorBinding::new(0, 0, payload.len() as u64)),
        }),
        payload,
    )
}

#[test]
fn public_moshi_lm_and_voice_bindings_preserve_source_contracts() {
    let input = valid_moshi_lm_input();
    let hparams = emel_model::bridge::MoshiLmHParams::try_new(&input).unwrap();
    assert_eq!(hparams.delay_count(), 3);
    assert_eq!(hparams.delays()[..3], [0, 1, 2]);
    assert!(hparams.validate().is_ok());

    let payloads = [
        vec![0_u8; 16 * 33 * 4],
        vec![0_u8; 16 * 32 * 4],
        vec![0_u8; 16 * 33 * 4],
        vec![0_u8; 16 * 33 * 4],
        vec![0_u8; 16 * 4],
    ];
    let lm_tensors = [
        tensor_with_shape(
            b"lm.text_emb.weight",
            SerializedType::F32,
            [16, 33, 1, 1],
            &payloads[0],
        ),
        tensor_with_shape(
            b"lm.text_linear.weight",
            SerializedType::F32,
            [16, 32, 1, 1],
            &payloads[1],
        ),
        tensor_with_shape(
            b"lm.emb.0.weight",
            SerializedType::F32,
            [16, 33, 1, 1],
            &payloads[2],
        ),
        tensor_with_shape(
            b"lm.emb.1.weight",
            SerializedType::F32,
            [16, 33, 1, 1],
            &payloads[3],
        ),
        tensor_with_shape(
            b"lm.out_norm.alpha",
            SerializedType::F32,
            [16, 1, 1, 1],
            &payloads[4],
        ),
    ];
    let lm_data = Data::try_from_moshi_lm(emel_model::bridge::MoshiLmDataInput {
        hparams,
        tensors: &lm_tensors,
    })
    .unwrap();
    let lm = lm_data.moshi_lm_binding_input();
    assert_eq!(lm.hparams().n_q().saturating_add(1), 3);
    assert!(!lm.hparams().depformer_weights_per_step());
    assert_eq!(
        lm.tensor_named(b"lm.emb.1.weight").map(TensorView::name),
        Some(&b"lm.emb.1.weight"[..])
    );

    let voice_payloads = [vec![0_u8; 16 * 4], vec![0_u8; 5 * 3 * 4]];
    let voice_tensors = [
        tensor_with_rank(
            b"voice.embeddings",
            SerializedType::F32,
            [16, 1, 1, 1],
            4,
            &voice_payloads[0],
        ),
        tensor_with_rank(
            b"voice.cache",
            SerializedType::I32,
            [5, 3, 1, 1],
            2,
            &voice_payloads[1],
        ),
    ];
    let voice_data = Data::try_from_moshi_voice(emel_model::bridge::MoshiVoiceDataInput {
        format: b"personaplex_prompt_v1",
        tensors: &voice_tensors,
    })
    .unwrap();
    let voice = voice_data.moshi_voice_binding_input();
    assert_eq!(voice.format(), b"personaplex_prompt_v1");
    assert_eq!(
        voice.embeddings().map(TensorView::name),
        Some(&b"voice.embeddings"[..])
    );
    assert_eq!(
        voice.cache().map(TensorView::name),
        Some(&b"voice.cache"[..])
    );
}

#[test]
fn public_moshi_lm_validation_classifies_scalar_delay_and_array_failures() {
    let base = valid_moshi_lm_input();
    assert_eq!(
        emel_model::bridge::MoshiLmHParams::try_new(&emel_model::bridge::MoshiLmHParamsInput {
            dim: 0,
            ..base
        }),
        Err(emel_model::bridge::MoshiLmHParamsError::InvalidValue)
    );
    assert_eq!(
        emel_model::bridge::MoshiLmHParams::try_new(&emel_model::bridge::MoshiLmHParamsInput {
            delay_count: 2,
            ..base
        }),
        Err(emel_model::bridge::MoshiLmHParamsError::InvalidDelays)
    );
    let mut invalid = base;
    invalid.inference_prompt_tokens[0] = -1;
    assert_eq!(
        emel_model::bridge::MoshiLmHParams::try_new(&invalid),
        Err(emel_model::bridge::MoshiLmHParamsError::InvalidArray)
    );
}

use emel_gguf as _;
use emel_kernels as _;
use emel_token as _;
