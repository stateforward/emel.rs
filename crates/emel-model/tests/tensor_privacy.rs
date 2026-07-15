//! External default-feature and internal-proof-feature API boundary checks.

use allocation_counter as _;
use emel_model as _;
use sml as _;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

fn check_project(case: &str, features: Option<&str>, source: &str) -> Output {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let project =
        std::env::temp_dir().join(format!("emel-model-privacy-{case}-{}", std::process::id()));
    let _ = fs::remove_dir_all(&project);
    fs::create_dir_all(project.join("src")).unwrap();
    let feature_clause =
        features.map_or_else(String::new, |value| format!(", features = [\"{value}\"]"));
    fs::write(
        project.join("Cargo.toml"),
        format!(
            "[package]\nname = \"privacy-{case}\"\nversion = \"0.0.0\"\nedition = \"2024\"\n\
             [dependencies]\nemel-model = {{ path = \"{}\", default-features = false{} }}\n\
             [workspace]\n",
            cargo_path(&root),
            feature_clause
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
fn default_features_keep_the_tensor_actor_private() {
    let output = check_project(
        "default-private",
        None,
        "use emel_model::tensor::Store; fn main() { let _ = Store::new(1); }",
    );
    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("module `tensor` is private"), "{stderr}");
}

#[test]
fn proof_feature_exposes_only_store_and_events() {
    let valid = check_project(
        "proof-valid",
        Some("model-tensor-proof"),
        "use emel_model::model_tensor_proof::{event, Store};\n\
         fn main() {\n\
             let mut actor = Store::new(1).unwrap();\n\
             let _ = actor.process_event(event::CaptureTensorState::new(0));\n\
         }",
    );
    assert!(
        valid.status.success(),
        "{}",
        String::from_utf8_lossy(&valid.stderr)
    );

    let invalid = check_project(
        "proof-private",
        Some("model-tensor-proof"),
        "use emel_model::model_tensor_proof::{Context, sm}; fn main() {}",
    );
    assert!(!invalid.status.success());
    let stderr = String::from_utf8_lossy(&invalid.stderr);
    assert!(stderr.contains("unresolved imports"), "{stderr}");
    assert!(stderr.contains("Context"), "{stderr}");
    assert!(stderr.contains("sm"), "{stderr}");
}
