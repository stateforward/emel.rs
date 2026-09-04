//! Public `Mimi` `GGUF` fixture ownership and initialization proof.
//!
//! Successful encode/decode uses deterministic callback probes here; native
//! RVQ and upsample still execute through their maintained paths.
//!
//! The lifecycle proof also verifies the public decoder's pre-initialization
//! error route.

use std::sync::Arc;
use std::sync::atomic::{AtomicU32, AtomicUsize, Ordering};

use emel_gguf::Loader;
use emel_gguf::event::{Bind, Parse, Probe, ReadF32, ReadSigned, ReadStringInto, Storage};
use emel_model::bridge::{Data, MimiHParams, MimiHParamsInput, TensorView};

use crate::mimi::{self, ArenaCapacities, RuntimeVariant};
use crate::{
    MimiDecodeRun, MimiDecoderStreamingState, MimiDiagnostics, MimiDiagnosticsEvent, MimiEncodeRun,
    MimiEncoderRuntime, MimiEncoderStreamingState, MimiError, MimiEvent, MimiInitRun,
    MimiQuantizerRuntime, SpeechCodecMimi,
};

static INIT_DONE_CALLS: AtomicUsize = AtomicUsize::new(0);
static INIT_DONE_FRAME_SAMPLES: AtomicU32 = AtomicU32::new(0);
static INIT_DONE_N_Q: AtomicU32 = AtomicU32::new(0);

fn record_initialize_done(done: crate::MimiInitializeDone) -> bool {
    INIT_DONE_FRAME_SAMPLES.store(done.frame_samples, Ordering::SeqCst);
    INIT_DONE_N_Q.store(done.n_q, Ordering::SeqCst);
    INIT_DONE_CALLS.fetch_add(1, Ordering::SeqCst);
    true
}

fn no_op_encoder_stage(
    _runtime: &MimiEncoderRuntime,
    _streaming: &mut MimiEncoderStreamingState<'_>,
    _input: &mut [f32],
    _workspace: &mut [f32],
    _variant: bool,
) {
}

fn deterministic_decoder_transformer(
    _runtime: &mimi::CodecRuntime<'_>,
    _streaming: &mut MimiDecoderStreamingState<'_>,
    frame: &mut [f32],
    _workspace: &mut [f32],
    _variant: bool,
) -> bool {
    frame.fill(0.0);
    true
}

fn deterministic_decoder_backend(
    _runtime: &mimi::CodecRuntime<'_>,
    _streaming: &mut MimiDecoderStreamingState<'_>,
    _frame: &[f32],
    pcm_out: &mut [f32],
    _workspace: &mut [f32],
    _variant: bool,
) -> bool {
    pcm_out.fill(0.25);
    true
}
// This proves public synchronous callback routing, not full numeric parity.
fn exercise_public_encode_decode(
    codec: &mut SpeechCodecMimi,
    runtime: &mimi::CodecRuntime<'_>,
    capacities: &ArenaCapacities,
    hparams: &MimiHParams,
) {
    let mut encoder_arena = vec![0.0; capacities.state_floats()];
    let mut encoder_stream = MimiEncoderStreamingState::new(&mut encoder_arena);
    let encoder_runtime = MimiEncoderRuntime::new(
        1_920,
        required_hparam_usize(hparams.dim()),
        capacities.frame_floats(),
        capacities.workspace_floats(),
        false,
        false,
    )
    .with_stages(
        no_op_encoder_stage,
        no_op_encoder_stage,
        no_op_encoder_stage,
    );
    let quantizer = MimiQuantizerRuntime::new(
        required_hparam_usize(hparams.n_q()),
        required_hparam_usize(hparams.dim()),
        required_hparam_usize(hparams.semantic_n_q()),
        required_hparam_usize(hparams.codebook_dim()),
        required_hparam_usize(hparams.card()),
        false,
        false,
    )
    .with_native_f32(
        runtime
            .native_f32()
            .expect("prepared F32 binding exposes native RVQ"),
    );
    let pcm = vec![0.0; 1_920];
    let mut encoder_frame = vec![0.0; capacities.frame_floats()];
    let mut encoder_workspace = vec![0.0; capacities.workspace_floats()];
    let mut codes_out = vec![0_i32; required_hparam_usize(hparams.n_q())];
    let mut encode_error = MimiError::None;
    let encode = MimiEncodeRun::new(
        &encoder_runtime,
        &quantizer,
        &mut encoder_stream,
        &pcm,
        &mut encoder_frame,
        &mut encoder_workspace,
        &mut codes_out,
    )
    .with_error_out(&mut encode_error);
    assert_eq!(codec.process_event(MimiEvent::Encode(encode)), Ok(()));
    assert_eq!(encode_error, MimiError::None);
    assert_eq!(codes_out.len(), required_hparam_usize(hparams.n_q()));
    let mut decoder_arena = vec![0.0; capacities.state_floats()];
    let mut decoder_stream = MimiDecoderStreamingState::new(&mut decoder_arena);
    let mut decode_frame = vec![0.0; capacities.frame_floats()];
    let mut decode_workspace = vec![0.0; capacities.workspace_floats()];
    let quantizer_indices = vec![0_i32; required_hparam_usize(hparams.n_q())];
    let mut pcm_out = vec![0.0; 1_920];
    let mut decode_error = MimiError::None;
    let decode = MimiDecodeRun::new(
        *runtime,
        &quantizer,
        &mut decoder_stream,
        &quantizer_indices,
        &mut decode_frame,
        &mut decode_workspace,
        &mut pcm_out,
    )
    .with_stages(
        deterministic_decoder_transformer,
        deterministic_decoder_backend,
    )
    .with_error_out(&mut decode_error);
    assert_eq!(codec.process_event(MimiEvent::Decode(decode)), Ok(()));
    assert_eq!(decode_error, MimiError::None);
    assert!(
        pcm_out
            .iter()
            .all(|sample| sample.to_bits() == 0.25_f32.to_bits())
    );
}

fn read_required_signed(loader: &mut Loader, key: &'static [u8]) -> i32 {
    i32::try_from(
        loader
            .process_event(ReadSigned::new(key))
            .expect("Mimi metadata query succeeds")
            .expect("pinned Mimi metadata key is present"),
    )
    .expect("pinned Mimi metadata fits i32")
}

fn read_required_f32(loader: &mut Loader, key: &'static [u8]) -> f32 {
    loader
        .process_event(ReadF32::new(key))
        .expect("Mimi metadata query succeeds")
        .expect("pinned Mimi metadata key is present")
}

fn read_required_string(loader: &mut Loader, key: &'static [u8]) -> Vec<u8> {
    let mut destination = [0_u8; 32];
    let length = loader
        .process_event(ReadStringInto::new(key, &mut destination))
        .expect("Mimi metadata query succeeds")
        .expect("pinned Mimi metadata key is present");
    destination[..length].to_vec()
}

fn read_mimi_hparams(loader: &mut Loader) -> MimiHParams {
    assert_eq!(
        read_required_string(loader, b"general.architecture"),
        b"moshi"
    );
    assert_eq!(read_required_string(loader, b"moshi.component"), b"mimi");
    MimiHParams::try_new(MimiHParamsInput {
        sample_rate: read_required_signed(loader, b"moshi.mimi.sample_rate"),
        frame_rate: read_required_f32(loader, b"moshi.mimi.frame_rate"),
        n_q: read_required_signed(loader, b"moshi.mimi.n_q"),
        card: read_required_signed(loader, b"moshi.mimi.card"),
        dim: read_required_signed(loader, b"moshi.mimi.dim"),
        semantic_n_q: read_required_signed(loader, b"moshi.mimi.semantic_n_q"),
        codebook_dim: read_required_signed(loader, b"moshi.mimi.codebook_dim"),
        transformer_num_layers: read_required_signed(loader, b"moshi.mimi.transformer.num_layers"),
        transformer_num_heads: read_required_signed(loader, b"moshi.mimi.transformer.num_heads"),
        transformer_context: read_required_signed(loader, b"moshi.mimi.transformer.context"),
        transformer_max_period: read_required_signed(loader, b"moshi.mimi.transformer.max_period"),
    })
    .expect("pinned Mimi metadata satisfies model geometry")
}

fn required_hparam_u32(value: i32) -> u32 {
    u32::try_from(value).expect("MimiHParams::try_new validates positive fixture hparams")
}

fn required_hparam_usize(value: i32) -> usize {
    usize::try_from(value).expect("MimiHParams::try_new validates positive fixture hparams")
}

fn assert_runtime_owns_model_bytes(runtime: &mimi::CodecRuntime<'_>) {
    assert!(
        runtime
            .model()
            .tensor_named(b"mimi.upsample.convtr.convtr.convtr.weight")
            .and_then(TensorView::byte_view)
            .is_some_and(|bytes| !bytes.is_empty())
    );
    assert!(runtime.upsample().weight(0, 0).is_some());
}

fn uninitialized_codec_after_decode(
    runtime: &mimi::CodecRuntime<'_>,
    capacities: &ArenaCapacities,
    hparams: &MimiHParams,
) -> (SpeechCodecMimi, std::cell::Cell<MimiDiagnostics>) {
    let mut codec = SpeechCodecMimi::new();
    let diagnostics = std::cell::Cell::new(MimiDiagnostics::default());
    codec
        .capture_diagnostics(MimiDiagnosticsEvent::new(&diagnostics))
        .expect("diagnostics capture succeeds before initialization");
    assert!(!diagnostics.get().initialized);
    assert!(codec.is_uninitialized());

    let mut decoder_arena = vec![0.0; capacities.state_floats()];
    let mut decoder_stream = MimiDecoderStreamingState::new(&mut decoder_arena);
    let quantizer = MimiQuantizerRuntime::new(
        required_hparam_usize(hparams.n_q()),
        required_hparam_usize(hparams.dim()),
        required_hparam_usize(hparams.semantic_n_q()),
        required_hparam_usize(hparams.codebook_dim()),
        required_hparam_usize(hparams.card()),
        false,
        false,
    )
    .with_native_f32(
        runtime
            .native_f32()
            .expect("prepared F32 binding exposes native RVQ"),
    );
    let mut decode_frame = vec![0.0; capacities.frame_floats()];
    let mut decode_workspace = vec![0.0; capacities.workspace_floats()];
    let mut pcm_out = vec![0.0; 1_920];
    let decode = MimiDecodeRun::new(
        *runtime,
        &quantizer,
        &mut decoder_stream,
        &[],
        &mut decode_frame,
        &mut decode_workspace,
        &mut pcm_out,
    );
    assert_eq!(
        codec.process_event(MimiEvent::Decode(decode)),
        Err(MimiError::NotInitialized)
    );
    assert!(!diagnostics.get().initialized);
    (codec, diagnostics)
}

fn initialize_and_assert(
    codec: &mut SpeechCodecMimi,
    diagnostics: &std::cell::Cell<MimiDiagnostics>,
    prepared: &mimi::PreparedMimiBinding<'_>,
    capacities: &ArenaCapacities,
    hparams: &MimiHParams,
) {
    INIT_DONE_CALLS.store(0, Ordering::SeqCst);
    INIT_DONE_FRAME_SAMPLES.store(0, Ordering::SeqCst);
    INIT_DONE_N_Q.store(0, Ordering::SeqCst);
    let mut prepared_arena = vec![0.0; capacities.prepared_floats()];
    let mut state_arena = vec![0.0; capacities.state_floats()];
    let mut workspace = vec![0.0; capacities.workspace_floats()];
    let mut frame = vec![0.0; capacities.frame_floats()];
    let mut error = MimiError::None;
    let init = MimiInitRun::new(
        *prepared,
        &mut prepared_arena,
        &mut state_arena,
        &mut workspace,
        &mut frame,
    )
    .with_error_out(&mut error)
    .with_callbacks(Some(record_initialize_done), None);
    assert_eq!(codec.process_event(MimiEvent::Init(init)), Ok(()));
    assert_eq!(error, MimiError::None);
    assert_eq!(INIT_DONE_CALLS.load(Ordering::SeqCst), 1);
    assert_eq!(INIT_DONE_FRAME_SAMPLES.load(Ordering::SeqCst), 1_920);
    assert_eq!(
        INIT_DONE_N_Q.load(Ordering::SeqCst),
        required_hparam_u32(hparams.n_q())
    );
    codec
        .capture_diagnostics(MimiDiagnosticsEvent::new(diagnostics))
        .expect("diagnostics capture succeeds after initialization");
    assert!(diagnostics.get().initialized);
    assert!(!codec.is_uninitialized());
}

fn assert_undersized_initialization_rejected(
    prepared: &mimi::PreparedMimiBinding<'_>,
    capacities: &ArenaCapacities,
) {
    let mut undersized_codec = SpeechCodecMimi::new();
    let mut undersized_prepared = vec![0.0; capacities.prepared_floats().saturating_sub(1)];
    let mut undersized_state = vec![0.0; capacities.state_floats()];
    let mut undersized_workspace = vec![0.0; capacities.workspace_floats()];
    let mut undersized_frame = vec![0.0; capacities.frame_floats()];
    let mut undersized_error = MimiError::None;
    let undersized = MimiInitRun::new(
        *prepared,
        &mut undersized_prepared,
        &mut undersized_state,
        &mut undersized_workspace,
        &mut undersized_frame,
    )
    .with_error_out(&mut undersized_error);
    assert_eq!(
        undersized_codec.process_event(MimiEvent::Init(undersized)),
        Err(MimiError::ArenaCapacity)
    );
    assert_eq!(undersized_error, MimiError::ArenaCapacity);
    let undersized_diagnostics = std::cell::Cell::new(MimiDiagnostics::default());
    undersized_codec
        .capture_diagnostics(MimiDiagnosticsEvent::new(&undersized_diagnostics))
        .expect("diagnostics capture succeeds after rejected initialization");
    assert!(!undersized_diagnostics.get().initialized);
}

#[test]
fn public_mimi_fixture_initializes_after_source_and_loader_drop() {
    let Ok(source_dir) = std::env::var("EMEL_CPP_SOURCE_DIR") else {
        return;
    };
    let path = std::path::Path::new(&source_dir)
        .join("tests")
        .join("models")
        .join("mimi-tiny.gguf");
    let Ok(bytes) = std::fs::read(path) else {
        return;
    };

    let source: Arc<[u8]> = Arc::from(bytes);
    let mut loader = Loader::new();
    let probe = loader
        .process_event(Probe::new(Arc::clone(&source)))
        .expect("pinned Mimi fixture probes through the public actor");
    let storage = Storage::exact(probe).expect("probe capacities are sufficient");
    loader
        .process_event(Bind::new(storage))
        .expect("pinned Mimi fixture binds through the public actor");
    let parsed = loader
        .process_event(Parse::new())
        .expect("pinned Mimi fixture parses through the public actor");
    let hparams = read_mimi_hparams(&mut loader);

    let data = Data::try_from_gguf_mimi(&mut loader, parsed, hparams)
        .expect("pinned Mimi fixture converts into model-owned data");
    let input = data.mimi_binding_input();
    let required = mimi::required_arena_capacities(input, 1_920, RuntimeVariant::F32)
        .expect("pinned Mimi fixture has calculable public arena requirements");
    let capacities: ArenaCapacities = required.capacities();
    let prepared = mimi::prepare_mimi(&mut loader, parsed, input, capacities, RuntimeVariant::F32)
        .expect("pinned Mimi fixture prepares through the public binding API");
    let runtime = prepared.codec_runtime();
    assert_eq!(runtime.frame_samples(), 1_920);
    assert_eq!(runtime.n_q(), required_hparam_u32(hparams.n_q()));

    // Data owns the resident tensor bytes; neither loader storage nor source
    // bytes may be required after the prepared binding is created.
    drop(loader);
    drop(source);
    assert_runtime_owns_model_bytes(&runtime);

    let (mut codec, diagnostics) =
        uninitialized_codec_after_decode(&runtime, &capacities, &hparams);
    initialize_and_assert(&mut codec, &diagnostics, &prepared, &capacities, &hparams);
    assert_undersized_initialization_rejected(&prepared, &capacities);
    exercise_public_encode_decode(&mut codec, &runtime, &capacities, &hparams);
}
