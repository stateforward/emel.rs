//! sortformer family bindings from `emel.cpp/src/emel/model/sortformer/`.

/// Family detail helpers (C++ `emel::model::sortformer::detail`).
#[derive(Debug, Default)]
pub struct Detail;

impl Detail {
    /// Bind family weight map / layers (TODO).
    pub fn bind_layers() {
        // TODO: convert from emel.cpp/src/emel/model/sortformer/detail.cpp and detail.hpp
        todo!("TODO: port sortformer::detail::bind_layers from emel.cpp/src/emel/model/sortformer/detail.cpp")
    }

    /// Load family-specific hparams (TODO).
    pub fn load_hparams() {
        // TODO: convert from emel.cpp/src/emel/model/sortformer/detail.cpp and detail.hpp
        todo!("TODO: port sortformer::detail::load_hparams from emel.cpp/src/emel/model/sortformer/detail.hpp")
    }
}

/// Family façade (C++ `emel::model::sortformer` any surface).
#[derive(Debug, Default)]
pub struct Any;

impl Any {
    /// Construct family binding (TODO).
    pub fn bind() -> Self {
        // TODO: convert from emel.cpp/src/emel/model/sortformer/any.hpp
        todo!("TODO: port sortformer::Any from emel.cpp/src/emel/model/sortformer/any.hpp")
    }
}
