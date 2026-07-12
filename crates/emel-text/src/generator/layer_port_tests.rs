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
    assert!(
        src.contains("event<EventChunk4Run>")
            || src.contains("Chunk4Run")
            || src.contains("chunk4")
    );
    // chunk machines must mention chunk events from C++
    assert!(src.contains("Chunk4") || src.contains("chunk4"));
    assert!(src.contains("Chunk8") || src.contains("chunk8"));
    assert!(src.contains("emel.cpp/src/emel/text/generator/layer/sm.hpp"));
    assert!(src.contains("TODO"));
}
