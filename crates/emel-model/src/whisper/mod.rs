//! whisper family bindings from `emel.cpp/src/emel/model/whisper/`.

/// Family detail helpers (C++ `emel::model::whisper::detail`).
#[derive(Debug, Default)]
pub struct Detail;

impl Detail {
    /// Bind family weight map / layers (TODO).
    pub fn bind_layers() {
        // TODO: convert from emel.cpp/src/emel/model/whisper/detail.cpp and detail.hpp
        todo!("TODO: port whisper::detail::bind_layers from emel.cpp/src/emel/model/whisper/detail.cpp")
    }

    /// Load family-specific hparams (TODO).
    pub fn load_hparams() {
        // TODO: convert from emel.cpp/src/emel/model/whisper/detail.cpp and detail.hpp
        todo!("TODO: port whisper::detail::load_hparams from emel.cpp/src/emel/model/whisper/detail.hpp")
    }
}

/// Family façade (C++ `emel::model::whisper` any surface).
#[derive(Debug, Default)]
pub struct Any;

impl Any {
    /// Construct family binding (TODO).
    pub fn bind() -> Self {
        // TODO: convert from emel.cpp/src/emel/model/whisper/any.hpp
        todo!("TODO: port whisper::Any from emel.cpp/src/emel/model/whisper/any.hpp")
    }
}
