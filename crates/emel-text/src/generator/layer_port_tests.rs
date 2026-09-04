//! Multi-table port checks for text generator layer SMs.

#[test]
fn layer_ports_all_three_cpp_models() {
    let src = include_str!("layer/sm.rs");
    assert_eq!(
        src.matches("sml!").count(),
        3,
        "scalar/chunk4/chunk8 models"
    );
    assert!(src.contains("TextGeneratorLayerScalarModel"));
    assert!(src.contains("TextGeneratorLayerChunk4Model"));
    assert!(src.contains("TextGeneratorLayerChunk8Model"));
    assert!(src.contains("EventChunk4Run") && src.contains("EventChunk8Run"));
    assert!(src.contains("emel.cpp/src/emel/text/generator/layer/sm.hpp"));
    assert!(src.contains("LayerOperation::ScalarAttentionNoneNone"));
    assert!(src.contains("LayerOperation::Chunk4AttentionNoneNone"));
    assert!(src.contains("LayerOperation::Chunk8AttentionNoneNone"));
    assert!(!src.contains("todo!("));
    assert!(!src.contains("TODO"));
}

#[test]
fn layer_request_validation_is_bounded_and_route_explicit() {
    use super::generator::layer::sm::{
        AttentionQkNormRoute, AttentionVNormRoute, LayerDtype, LayerRequest, ResidualRoute,
        WindowMode,
    };

    let request = LayerRequest::new(LayerDtype::F32, 1, 8, 8, 8, 16, 0, 0);
    assert!(!request.scalar_valid());
    assert!(request.route_valid());
    assert_eq!(request.residual, ResidualRoute::Attention);
    assert_eq!(request.qk_norm, AttentionQkNormRoute::None);
    assert_eq!(request.v_norm, AttentionVNormRoute::None);
    assert_eq!(request.window_mode, WindowMode::Resident);
}

#[test]
fn layer_machine_surface_exposes_source_shaped_actors() {
    use super::generator::layer::sm::{LayerChunk4Actor, LayerChunk8Actor, LayerScalarActor};

    let mut scalar = LayerScalarActor::new();
    let mut chunk4 = LayerChunk4Actor::new();
    let mut chunk8 = LayerChunk8Actor::new();
    assert!(scalar.process_unexpected_event().is_err());
    assert!(chunk4.process_unexpected_event().is_err());
    assert!(chunk8.process_unexpected_event().is_err());
}
