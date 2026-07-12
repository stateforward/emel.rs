//! llama family bindings from `emel.cpp/src/emel/model/llama/`.

/// Family detail helpers (C++ `emel::model::llama::detail`).
#[derive(Debug, Default)]
pub struct Detail;

impl Detail {
    /// Bind family weight map / layers (TODO).
    pub fn bind_layers() {
        // TODO: convert from emel.cpp/src/emel/model/llama/detail.cpp and detail.hpp
        todo!("TODO: port llama::detail::bind_layers from emel.cpp/src/emel/model/llama/detail.cpp")
    }

    /// Load family-specific hparams (TODO).
    pub fn load_hparams() {
        // TODO: convert from emel.cpp/src/emel/model/llama/detail.cpp and detail.hpp
        todo!("TODO: port llama::detail::load_hparams from emel.cpp/src/emel/model/llama/detail.hpp")
    }
}

/// Family façade (C++ `emel::model::llama` any surface).
#[derive(Debug, Default)]
pub struct Any;

impl Any {
    /// Construct family binding (TODO).
    pub fn bind() -> Self {
        // TODO: convert from emel.cpp/src/emel/model/llama/any.hpp
        todo!("TODO: port llama::Any from emel.cpp/src/emel/model/llama/any.hpp")
    }
}
