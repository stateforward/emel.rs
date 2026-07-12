//! Architecture registry from `emel.cpp/src/emel/model/architecture/detail.hpp`.

use crate::data::ModelData;
use crate::detail::HparamLoader;
use crate::generation::Contract;

/// Architecture plugin entry (C++ `emel::model::architecture`).
#[derive(Clone, Debug)]
pub struct Architecture {
    /// Architecture name key (e.g. `"llama"`).
    pub name: &'static str,
    /// Optional hparam loader callback (TODO: function pointers from C++).
    pub load_hparams: Option<fn(&HparamLoader, &mut ModelData) -> bool>,
    /// Optional data validator.
    pub validate_data: Option<fn(&ModelData) -> Result<(), i32>>,
    /// Optional generation-contract builder.
    pub build_generation_contract: Option<fn(&ModelData, &mut Contract) -> Result<(), i32>>,
}

/// Span of registered architectures (C++ `emel::model::architectures`).
pub type Architectures = &'static [Architecture];

/// Default architecture table.
///
/// TODO: convert from emel.cpp/src/emel/model/architecture/detail.cpp
/// (`default_architecture_span`).
#[must_use]
pub fn default_architecture_span() -> Architectures {
    // TODO: convert from emel.cpp/src/emel/model/architecture/detail.cpp /
    // emel.cpp/src/emel/model/architecture/detail.hpp::default_architecture_span
    todo!(
        "TODO: port default_architecture_span from \
         emel.cpp/src/emel/model/architecture/detail.cpp"
    )
}

/// Resolve an architecture by name against an available table.
///
/// TODO: convert from emel.cpp/src/emel/model/architecture/detail.cpp
/// (`resolve_architecture`).
#[must_use]
pub fn resolve_architecture(
    _name: &str,
    _available: Architectures,
) -> Option<&'static Architecture> {
    // TODO: convert from emel.cpp/src/emel/model/architecture/detail.cpp /
    // emel.cpp/src/emel/model/architecture/detail.hpp::resolve_architecture
    todo!(
        "TODO: port resolve_architecture from \
         emel.cpp/src/emel/model/architecture/detail.cpp"
    )
}
