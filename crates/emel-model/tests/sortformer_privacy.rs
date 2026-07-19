//! Downstream compile checks for the Sortformer actor boundary.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use allocation_counter as _;
use emel_gguf as _;
use emel_io as _;
use emel_kernels as _;
use emel_model as _;
use emel_tensor as _;
use emel_token as _;
use sml as _;

fn cargo_path(path: &Path) -> String {
    path.display().to_string().replace('\\', "\\\\")
}

fn check_project(case: &str, source: &str) -> Output {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let project = std::env::temp_dir().join(format!(
        "emel-model-sortformer-privacy-{case}-{}",
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&project);
    fs::create_dir_all(project.join("src")).unwrap();
    fs::write(
        project.join("Cargo.toml"),
        format!(
            "[package]\nname = \"sortformer-privacy-{case}\"\nversion = \"0.0.0\"\nedition = \"2024\"\n\
             [dependencies]\nemel-model = {{ path = \"{}\", default-features = false }}\n\
             emel-gguf = {{ path = \"{}\" }}\n[workspace]\n",
            cargo_path(&root),
            cargo_path(&root.join("../emel-gguf"))
        ),
    )
    .unwrap();
    fs::write(project.join("src/main.rs"), source).unwrap();
    let output = Command::new(env!("CARGO"))
        .args(["check", "--offline", "--quiet"])
        .env(
            "CARGO_TARGET_DIR",
            root.join("../../target/model-sortformer-privacy"),
        )
        .current_dir(&project)
        .output()
        .unwrap();
    let _ = fs::remove_dir_all(project);
    output
}

#[test]
fn public_actor_events_descriptors_and_typed_load_errors_compile() {
    let output = check_project(
        "public",
        "use emel_gguf::{Loader, event::ParseDone};\n\
         use emel_model::sortformer::{Sortformer, Storage, HparamError, HparamErrorKind, HparamField, event::LoadError};\n\
         fn main() {\n\
           let _load: fn(Loader, ParseDone, Storage) -> Result<Sortformer, LoadError> = Sortformer::load;\n\
           let _ = core::mem::size_of::<(HparamError, HparamErrorKind, HparamField)>();\n\
         }",
    );
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn generated_machine_context_state_and_observation_storage_are_inaccessible() {
    for (case, source) in [
        (
            "machine",
            "use emel_model::sortformer::sm::SortformerMachineStateMachine; fn main() {}",
        ),
        (
            "context",
            "use emel_model::sortformer::actor::Context; fn main() {}",
        ),
        (
            "state",
            "fn inspect(value: emel_model::sortformer::Sortformer) { let _ = value.state(); } fn main() {}",
        ),
        (
            "storage",
            "fn inspect(value: emel_model::sortformer::Storage) { let _ = value.observations; } fn main() {}",
        ),
        (
            "observation-event",
            "use emel_model::sortformer::event::TensorObservation; fn main() {}",
        ),
        (
            "finish-event",
            "use emel_model::sortformer::event::ContractFinish; fn main() {}",
        ),
    ] {
        let output = check_project(case, source);
        assert!(!output.status.success(), "{case} unexpectedly compiled");
    }
}
