//! Structural verification that model SMs are present and constructible.

#![allow(clippy::single_element_loop)]

#[test]
fn loader_sm_table_is_constructible() {
    use crate::loader::sm::{ModelLoaderContext, ModelLoaderStateMachine, ModelLoaderStates};
    let sm = ModelLoaderStateMachine::new(ModelLoaderContext::default());
    assert!(sm.is(&ModelLoaderStates::Ready));
}

#[test]
fn tensor_sm_table_is_constructible() {
    let store = crate::tensor::Store::new(1).unwrap();
    assert!(store.is_ready());
}

#[test]
fn window_actor_boundary_is_constructible() {
    let _window = crate::tensor::window::actor::Window::new();
}

#[test]
fn model_sm_sources_use_sml_and_cite_cpp() {
    for src in [include_str!("loader/sm.rs")] {
        assert!(src.contains("sml!"));
        assert!(src.contains("emel.cpp/src/emel/model"));
    }

    let tensor = include_str!("tensor/sm.rs");
    assert!(tensor.contains("sml!"));
    assert!(tensor.contains("emel.cpp/src/emel/model"));
    assert!(!tensor.contains("TODO"));
}

fn parse_ok(
    _model: &mut crate::data::Data,
    _source: crate::loader::event::Source<'_>,
) -> crate::loader::event::Error {
    crate::loader::event::Error::None
}

fn parse_failed(
    _model: &mut crate::data::Data,
    _source: crate::loader::event::Source<'_>,
) -> crate::loader::event::Error {
    crate::loader::event::Error::ParseFailed
}

fn check_ok(_model: &mut crate::data::Data) -> crate::loader::event::Error {
    crate::loader::event::Error::None
}

fn check_failed(_model: &mut crate::data::Data) -> crate::loader::event::Error {
    crate::loader::event::Error::ModelInvalid
}

fn load_done(
    _request: &crate::loader::event::LoadRequest<'_>,
    _stats: crate::loader::event::LoadStats,
) {
}

fn load_error(
    _request: &crate::loader::event::LoadRequest<'_>,
    _error: crate::loader::event::LoadError,
) {
}

#[derive(Clone, Copy)]
struct SuccessfulTensorLoader;

impl crate::loader::TensorLoader for SuccessfulTensorLoader {
    fn load(
        &mut self,
        _model: &mut crate::data::Data,
        _source: crate::loader::event::Source<'_>,
        strategy: emel_io::loader::event::StrategyKind,
    ) -> Result<crate::loader::event::LoadStats, crate::loader::event::Error> {
        Ok(crate::loader::event::LoadStats {
            bytes_total: 12,
            bytes_done: 12,
            used_mmap: matches!(strategy, emel_io::loader::event::StrategyKind::MappedFile),
            used_strategy: strategy,
        })
    }
}

#[derive(Clone, Copy)]
struct FailedTensorLoader;

impl crate::loader::TensorLoader for FailedTensorLoader {
    fn load(
        &mut self,
        _model: &mut crate::data::Data,
        _source: crate::loader::event::Source<'_>,
        _strategy: emel_io::loader::event::StrategyKind,
    ) -> Result<crate::loader::event::LoadStats, crate::loader::event::Error> {
        Err(crate::loader::event::Error::BackendError)
    }
}

fn valid_request(model: &mut crate::data::Data) -> crate::loader::event::LoadRequest<'_> {
    crate::loader::event::LoadRequest::new(
        model,
        crate::loader::event::Source {
            model_path: "fixture.gguf",
            file_image: None,
            mapped_files: None,
        },
    )
}

#[test]
fn loader_event_contract_classifies_request_and_status_values() {
    use emel_io::loader::event::StrategyKind;

    let mut data = crate::data::Data::try_new().unwrap();
    let mut request = valid_request(&mut data);
    assert!(!request.is_valid());
    request.parse_model = Some(parse_ok);
    assert!(request.is_valid());
    request.source = crate::loader::event::Source {
        model_path: "fixture.gguf",
        file_image: None,
        mapped_files: None,
    };
    assert!(request.is_valid());
    request.source.file_image = Some(&[]);
    assert!(!request.is_valid());
    request.source.model_path = "fixture.gguf";
    assert!(request.tensor_capacity_valid());
    request.model.n_tensors = u32::try_from(crate::data::MAX_TENSORS)
        .unwrap()
        .saturating_add(1);
    assert!(!request.tensor_capacity_valid());

    let load_error = crate::loader::event::LoadError::new(
        crate::loader::event::Error::BackendError,
        StrategyKind::MappedFile,
        StrategyKind::ReadCopy,
    );
    assert_eq!(load_error.error, crate::loader::event::Error::BackendError);
    assert_eq!(load_error.requested_strategy, StrategyKind::MappedFile);
    assert_eq!(load_error.used_strategy, StrategyKind::ReadCopy);

    let error_status =
        crate::loader::event::LoadStatus::error(crate::loader::event::Error::InvalidRequest);
    assert_eq!(
        error_status.error,
        crate::loader::event::Error::InvalidRequest
    );
    assert_eq!(
        error_status.stats,
        crate::loader::event::LoadStats::default()
    );
    let stats = crate::loader::event::LoadStats {
        bytes_total: 7,
        bytes_done: 5,
        used_mmap: true,
        used_strategy: StrategyKind::MappedFile,
    };
    assert_eq!(
        crate::loader::event::LoadStatus::success(stats),
        crate::loader::event::LoadStatus {
            error: crate::loader::event::Error::None,
            stats,
        }
    );
}

#[test]
fn loader_rejects_invalid_request_and_returns_ready() {
    let mut model = crate::data::Data::try_new().unwrap();
    let request = valid_request(&mut model);
    let mut loader = crate::loader::ModelLoader::new();
    assert_eq!(
        loader.process_event(request),
        Err(crate::loader::event::Error::InvalidRequest)
    );
    assert!(loader.is_ready());
}

#[test]
fn loader_publishes_parse_error_to_error_callback() {
    let mut model = crate::data::Data::try_new().unwrap();
    let mut request = valid_request(&mut model);
    request.parse_model = Some(parse_failed);
    request.vocab_only = true;
    request.on_error = Some(load_error);
    let mut loader = crate::loader::ModelLoader::new();
    assert_eq!(
        loader.process_event(request),
        Err(crate::loader::event::Error::ParseFailed)
    );
    assert!(loader.is_ready());
}

#[test]
fn loader_runs_tensor_and_validation_phases_to_done_callback() {
    use emel_io::loader::event::StrategyKind;

    let mut model = crate::data::Data::try_new().unwrap();
    model.n_tensors = 1;
    let mut request = valid_request(&mut model);
    request.parse_model = Some(parse_ok);
    request.io_strategy = StrategyKind::MappedFile;
    request.map_layers = Some(check_ok);
    request.validate_structure = Some(check_ok);
    request.validate_architecture_impl = Some(check_ok);
    request.on_done = Some(load_done);
    let mut loader = crate::loader::ModelLoader::with_tensor_loader(SuccessfulTensorLoader);
    let stats = loader.process_event(request).unwrap();
    assert_eq!(stats.bytes_total, 12);
    assert_eq!(stats.bytes_done, 12);
    assert!(stats.used_mmap);
    assert_eq!(stats.used_strategy, StrategyKind::MappedFile);
    assert!(loader.is_ready());
}

#[test]
fn loader_preserves_tensor_error_and_returns_ready() {
    let mut model = crate::data::Data::try_new().unwrap();
    model.n_tensors = 1;
    let mut request = valid_request(&mut model);
    request.parse_model = Some(parse_ok);
    request.on_error = Some(load_error);
    let mut loader = crate::loader::ModelLoader::with_tensor_loader(FailedTensorLoader);
    assert_eq!(
        loader.process_event(request),
        Err(crate::loader::event::Error::BackendError)
    );
    assert!(loader.is_ready());
}

#[test]
fn loader_classifies_missing_tensor_and_validation_policies() {
    let mut model = crate::data::Data::try_new().unwrap();
    let mut request = valid_request(&mut model);
    request.parse_model = Some(parse_ok);
    request.vocab_only = false;
    assert_eq!(
        crate::loader::ModelLoader::new().process_event(request),
        Err(crate::loader::event::Error::InvalidRequest)
    );

    let mut model = crate::data::Data::try_new().unwrap();
    model.n_tensors = 1;
    let mut request = valid_request(&mut model);
    request.parse_model = Some(parse_ok);
    request.vocab_only = true;
    request.check_tensors = true;
    assert_eq!(
        crate::loader::ModelLoader::new().process_event(request),
        Err(crate::loader::event::Error::InvalidRequest)
    );

    let mut model = crate::data::Data::try_new().unwrap();
    model.n_tensors = 1;
    let mut request = valid_request(&mut model);
    request.parse_model = Some(parse_ok);
    request.vocab_only = true;
    request.check_tensors = true;
    request.validate_structure = Some(check_failed);
    assert_eq!(
        crate::loader::ModelLoader::new().process_event(request),
        Err(crate::loader::event::Error::ModelInvalid)
    );

    let mut model = crate::data::Data::try_new().unwrap();
    model.n_tensors = 1;
    let mut request = valid_request(&mut model);
    request.parse_model = Some(parse_ok);
    request.vocab_only = true;
    request.check_tensors = false;
    request.validate_architecture = true;
    assert_eq!(
        crate::loader::ModelLoader::new().process_event(request),
        Err(crate::loader::event::Error::InvalidRequest)
    );
}
