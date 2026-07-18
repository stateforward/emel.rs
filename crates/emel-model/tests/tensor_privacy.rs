//! External tensor actor API boundary checks.

use allocation_counter as _;
use emel_io as _;
use emel_model as _;
use emel_tensor as _;
use sml as _;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

fn check_project(case: &str, source: &str) -> Output {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let project =
        std::env::temp_dir().join(format!("emel-model-privacy-{case}-{}", std::process::id()));
    let _ = fs::remove_dir_all(&project);
    fs::create_dir_all(project.join("src")).unwrap();
    fs::write(
        project.join("Cargo.toml"),
        format!(
            "[package]\nname = \"privacy-{case}\"\nversion = \"0.0.0\"\nedition = \"2024\"\n\
             [dependencies]\nemel-model = {{ path = \"{}\", default-features = false }}\n\
             [workspace]\n",
            cargo_path(&root)
        ),
    )
    .unwrap();
    fs::write(project.join("src/main.rs"), source).unwrap();
    let output = Command::new(env!("CARGO"))
        .args(["check", "--offline", "--quiet"])
        .env(
            "CARGO_TARGET_DIR",
            root.join("../../target/model-tensor-privacy"),
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
fn default_features_export_the_actor_events_and_dependency_contracts() {
    let output = check_project(
        "default-public",
        "use emel_model::tensor::{dependency::Actors, event, Store};\n\
         fn main() {\n\
             let mut actor = Store::with_dependencies(1, Actors::new((), (), ())).unwrap();\n\
             let _ = actor.process_event(event::CaptureTensorState::new(0));\n\
         }",
    );
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn machine_context_and_state_inspection_remain_private() {
    let invalid = check_project(
        "actor-private",
        "use emel_model::tensor::{Context, sm, window};\n\
         fn main() {\n\
             let actor = emel_model::tensor::Store::new(1).unwrap();\n\
             let _ = actor.is_ready();\n\
         }",
    );
    assert!(!invalid.status.success());
    let stderr = String::from_utf8_lossy(&invalid.stderr);
    assert!(
        stderr.contains("unresolved import") || stderr.contains("private"),
        "{stderr}"
    );
    assert!(stderr.contains("Context"), "{stderr}");
    assert!(stderr.contains("sm"), "{stderr}");
    assert!(stderr.contains("window"), "{stderr}");
    assert!(stderr.contains("is_ready"), "{stderr}");
}
use emel_gguf as _;
use emel_kernels as _;
use emel_token as _;
