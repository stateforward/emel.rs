//! External Llama actor composition and generated-state privacy checks.

use allocation_counter as _;
use emel_gguf as _;
use emel_io as _;
use emel_kernels as _;
use emel_model as _;
use emel_tensor as _;
use emel_token as _;
use sml as _;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

fn check_project(case: &str, source: &str) -> Output {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let project = std::env::temp_dir().join(format!(
        "emel-model-llama-privacy-{case}-{}",
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&project);
    fs::create_dir_all(project.join("src")).unwrap();
    fs::write(
        project.join("Cargo.toml"),
        format!(
            "[package]\nname = \"llama-privacy-{case}\"\nversion = \"0.0.0\"\nedition = \"2024\"\n\
             [dependencies]\nemel-model = {{ path = \"{}\", default-features = false }}\n\
             emel-kernels = {{ path = \"{}\" }}\n[workspace]\n",
            cargo_path(&root),
            cargo_path(&root.join("../emel-kernels"))
        ),
    )
    .unwrap();
    fs::write(project.join("src/main.rs"), source).unwrap();
    let output = Command::new(env!("CARGO"))
        .args(["check", "--offline", "--quiet"])
        .env(
            "CARGO_TARGET_DIR",
            root.join("../../target/model-llama-privacy"),
        )
        .current_dir(&project)
        .output()
        .unwrap();
    let _ = fs::remove_dir_all(project);
    output
}

fn cargo_path(path: &Path) -> String {
    path.display().to_string().replace('\\', "\\\\")
}

#[test]
fn public_parameters_and_events_compile() {
    let output = check_project(
        "public",
        "use emel_model::llama::{Parameters, event::{ContractBegin, HPARAM_KEYS}};\n\
         fn main() { let _ = Parameters::default(); let _ = HPARAM_KEYS; let _ = core::mem::size_of::<ContractBegin<'static>>(); }",
    );
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn generated_machine_context_and_state_are_inaccessible() {
    for (case, source) in [
        (
            "machine",
            "use emel_model::llama::sm::LlamaMachineStateMachine; fn main() {}",
        ),
        (
            "context",
            "use emel_model::llama::actor::Context; fn main() {}",
        ),
        (
            "state",
            "fn inspect(value: emel_model::llama::Llama) { let _ = value.state(); } fn main() {}",
        ),
    ] {
        let output = check_project(case, source);
        assert!(!output.status.success(), "{case} unexpectedly compiled");
    }
}
