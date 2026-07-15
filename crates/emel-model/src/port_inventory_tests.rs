//! Structural verification that model SMs are present and constructible.

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
fn window_sm_table_is_constructible() {
    use crate::tensor::window::sm::{
        ModelTensorWindowContext, ModelTensorWindowStateMachine, ModelTensorWindowStates,
    };
    let sm = ModelTensorWindowStateMachine::new(ModelTensorWindowContext::default());
    assert!(sm.is(&ModelTensorWindowStates::StateUnbound));
}

#[test]
fn model_sm_sources_use_sml_and_cite_cpp() {
    for src in [
        include_str!("loader/sm.rs"),
        include_str!("tensor/window/sm.rs"),
    ] {
        assert!(src.contains("sml!"));
        assert!(src.contains("emel.cpp/src/emel/model"));
        assert!(src.contains("TODO"));
    }

    let tensor = include_str!("tensor/sm.rs");
    assert!(tensor.contains("sml!"));
    assert!(tensor.contains("emel.cpp/src/emel/model"));
    assert!(!tensor.contains("TODO"));
}
