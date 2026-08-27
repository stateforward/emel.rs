//! Architecture registry from `emel.cpp/src/emel/model/architecture/detail.hpp`.

#![allow(clippy::cognitive_complexity)]

/// Architecture plugin entry (C++ `emel::model::architecture`).
#[derive(Clone, Debug)]
pub struct Architecture {
    /// Architecture name key (e.g. `"llama"`).
    pub name: &'static str,
}

/// Span of registered architectures (C++ `emel::model::architectures`).
pub type Architectures = &'static [Architecture];

/// Result of the common architecture routing boundary.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RouteOutcome {
    /// The architecture name is registered and can be routed to its family
    /// actor for validation.
    Supported,
    /// The architecture name is not present in the pinned registry.
    Unsupported,
}

/// Public family identity selected by the pinned architecture registry.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Family {
    Llama,
    Qwen3,
    Lfm2,
    Gemma4,
    OmniEmbed,
    Sortformer,
    Whisper,
    Moshi,
}

/// Whether a resolved family currently exposes a maintained actor boundary.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ActorAvailability {
    Maintained,
    PendingPort,
}

/// Exact architecture routing result used by model-domain orchestrators.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Route {
    pub family: Family,
    pub availability: ActorAvailability,
}

/// Resolves an exact architecture name and reports its owner boundary.
#[must_use]
pub const fn resolve_route(name: &[u8]) -> Option<Route> {
    let Some(family) = resolve_family(name) else {
        return None;
    };
    Some(Route {
        family,
        availability: actor_availability(family),
    })
}

/// Reports actor availability without constructing or exposing actor internals.
#[must_use]
pub const fn actor_availability(family: Family) -> ActorAvailability {
    match family {
        Family::Llama | Family::Qwen3 | Family::Lfm2 | Family::Gemma4 => {
            ActorAvailability::Maintained
        }
        Family::OmniEmbed | Family::Sortformer | Family::Whisper | Family::Moshi => {
            ActorAvailability::PendingPort
        }
    }
}

/// Resolves an exact architecture name to its family identity.
#[must_use]
pub const fn resolve_family(name: &[u8]) -> Option<Family> {
    match name {
        b"llama" => Some(Family::Llama),
        b"qwen3" => Some(Family::Qwen3),
        b"lfm2" => Some(Family::Lfm2),
        b"gemma4" => Some(Family::Gemma4),
        b"omniembed" => Some(Family::OmniEmbed),
        b"sortformer" => Some(Family::Sortformer),
        b"whisper" => Some(Family::Whisper),
        b"moshi" => Some(Family::Moshi),
        _ => None,
    }
}

/// Classifies an architecture before family-specific actor validation.
///
/// This is deliberately a pure, allocation-free boundary. Family actors keep
/// ownership of model data and perform the contract validation itself; callers
/// use this result to route an explicit typed validation event.
#[must_use]
pub fn classify_route(name: &[u8]) -> RouteOutcome {
    if is_supported_execution_architecture(name) {
        RouteOutcome::Supported
    } else {
        RouteOutcome::Unsupported
    }
}

/// Default architecture table ported from the pinned C++ registry.
#[must_use]
pub fn default_architecture_span() -> Architectures {
    static ARCHITECTURES: [Architecture; 8] = [
        Architecture { name: "llama" },
        Architecture { name: "qwen3" },
        Architecture { name: "lfm2" },
        Architecture { name: "gemma4" },
        Architecture { name: "omniembed" },
        Architecture { name: "sortformer" },
        Architecture { name: "whisper" },
        Architecture { name: "moshi" },
    ];
    &ARCHITECTURES
}

/// Resolve an architecture by exact name against an available table.
#[must_use]
pub fn resolve_architecture(name: &str, available: Architectures) -> Option<&'static Architecture> {
    available.iter().find(|candidate| candidate.name == name)
}

#[must_use]
pub fn is_supported_execution_architecture(name: &[u8]) -> bool {
    core::str::from_utf8(name)
        .ok()
        .and_then(|name| resolve_architecture(name, default_architecture_span()))
        .is_some()
}

/// Returns whether the architecture is the pinned LFM2 execution family.
#[must_use]
pub fn is_lfm2_execution_architecture(name: &[u8]) -> bool {
    name == b"lfm2"
}

/// Returns whether the architecture is the pinned Gemma4 execution family.
#[must_use]
pub fn is_gemma4_execution_architecture(name: &[u8]) -> bool {
    name == b"gemma4"
}

#[must_use]
pub fn is_moshi_execution_architecture(name: &[u8]) -> bool {
    name == b"moshi"
}

#[must_use]
pub fn is_whisper_execution_architecture(name: &[u8]) -> bool {
    name == b"whisper"
}

#[must_use]
pub fn is_omniembed_execution_architecture(name: &[u8]) -> bool {
    name == b"omniembed"
}

#[must_use]
pub fn is_sortformer_execution_architecture(name: &[u8]) -> bool {
    name == b"sortformer"
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_registry_resolves_source_fixed_names() {
        let architectures = default_architecture_span();
        assert_eq!(architectures.len(), 8);
        assert_eq!(
            resolve_architecture("llama", architectures).map(|a| a.name),
            Some("llama")
        );
        assert!(resolve_architecture("missing", architectures).is_none());
        for name in [
            b"llama".as_slice(),
            b"qwen3",
            b"lfm2",
            b"gemma4",
            b"omniembed",
            b"sortformer",
            b"whisper",
            b"moshi",
        ] {
            assert!(is_supported_execution_architecture(name));
        }
        assert!(!is_supported_execution_architecture(b"unknown"));
        assert!(is_lfm2_execution_architecture(b"lfm2"));
        assert!(!is_lfm2_execution_architecture(b"llama"));
        assert!(is_gemma4_execution_architecture(b"gemma4"));
        assert!(!is_gemma4_execution_architecture(b"llama"));
        assert_eq!(classify_route(b"llama"), RouteOutcome::Supported);
        assert_eq!(classify_route(b"unknown"), RouteOutcome::Unsupported);
        assert_eq!(resolve_family(b"llama"), Some(Family::Llama));
        assert_eq!(resolve_family(b"qwen3"), Some(Family::Qwen3));
        assert_eq!(resolve_family(b"lfm2"), Some(Family::Lfm2));
        assert_eq!(resolve_family(b"gemma4"), Some(Family::Gemma4));
        assert_eq!(resolve_family(b"omniembed"), Some(Family::OmniEmbed));
        assert_eq!(resolve_family(b"sortformer"), Some(Family::Sortformer));
        assert_eq!(resolve_family(b"whisper"), Some(Family::Whisper));
        assert_eq!(resolve_family(b"moshi"), Some(Family::Moshi));
        assert_eq!(resolve_family(b"gemma4.extra"), None);
        assert_eq!(
            resolve_route(b"omniembed"),
            Some(Route {
                family: Family::OmniEmbed,
                availability: ActorAvailability::PendingPort,
            })
        );
        assert_eq!(
            resolve_route(b"llama"),
            Some(Route {
                family: Family::Llama,
                availability: ActorAvailability::Maintained,
            })
        );
        assert_eq!(resolve_route(b"omniembed.extra"), None);
        assert_eq!(resolve_route(&[0xff, b'l', b'l', b'a', b'm', b'a']), None);
        assert_eq!(
            actor_availability(Family::Gemma4),
            ActorAvailability::Maintained
        );
        assert_eq!(
            actor_availability(Family::Moshi),
            ActorAvailability::PendingPort
        );
        assert!(!is_supported_execution_architecture(&[
            0xff, b'l', b'a', b'm', b'a'
        ]));
        assert!(!is_lfm2_execution_architecture(b"lfm2.extra"));
        assert!(!is_gemma4_execution_architecture(b"gemma4.extra"));
        assert!(!is_omniembed_execution_architecture(b"omniembed.extra"));
        assert!(!is_sortformer_execution_architecture(b"sortformer.extra"));
        assert!(!is_whisper_execution_architecture(b"whisper.extra"));
        assert!(!is_moshi_execution_architecture(b"moshi.extra"));
    }

    #[test]
    fn every_family_has_explicit_route_and_availability() {
        let cases = [
            (
                b"llama".as_slice(),
                Family::Llama,
                ActorAvailability::Maintained,
            ),
            (b"qwen3", Family::Qwen3, ActorAvailability::Maintained),
            (b"lfm2", Family::Lfm2, ActorAvailability::Maintained),
            (b"gemma4", Family::Gemma4, ActorAvailability::Maintained),
            (
                b"omniembed",
                Family::OmniEmbed,
                ActorAvailability::PendingPort,
            ),
            (
                b"sortformer",
                Family::Sortformer,
                ActorAvailability::PendingPort,
            ),
            (b"whisper", Family::Whisper, ActorAvailability::PendingPort),
            (b"moshi", Family::Moshi, ActorAvailability::PendingPort),
        ];
        for (name, family, availability) in cases {
            assert_eq!(
                resolve_route(name),
                Some(Route {
                    family,
                    availability
                })
            );
            assert_eq!(actor_availability(family), availability);
        }
    }
}
