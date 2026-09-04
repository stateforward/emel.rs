//! Multi-table port checks for speech generator SMs.

#[test]
fn speech_generator_ports_duplex_and_synthesis() {
    let src = include_str!("generator/sm.rs");
    assert_eq!(src.matches("sml!").count(), 2);
    assert!(src.contains("SpeechGeneratorDuplexModel"));
    assert!(src.contains("SpeechGeneratorSynthesisModel"));
    assert!(src.contains("state_initialize_synthesis") || src.contains("initialize_synthesis"));
    assert!(src.contains("emel.cpp/src/emel/speech/generator/sm.hpp"));
    assert!(src.contains("state_flush_pending_result"));
    assert!(src.contains("state_flush_produced_result"));
    assert!(src.contains("state_flush_done_channel_decision"));
    assert!(src.contains("effect_emit_flush_done_dependencies_type"));
}
