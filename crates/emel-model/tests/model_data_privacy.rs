//! External model-data ownership boundary checks.

use allocation_counter as _;
use emel_io as _;
use emel_model as _;
use sml as _;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

fn check_project(case: &str, source: &str) -> Output {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let project = std::env::temp_dir().join(format!(
        "emel-model-data-privacy-{case}-{}",
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&project);
    fs::create_dir_all(project.join("src")).unwrap();
    fs::write(
        project.join("Cargo.toml"),
        format!(
            "[package]\nname = \"model-data-privacy-{case}\"\nversion = \"0.0.0\"\nedition = \"2024\"\n\
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
            root.join("../../target/model-data-privacy"),
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
fn model_data_schema_and_storage_are_not_public_api() {
    let invalid = check_project(
        "private",
        "use emel_model::data::{Data, Metadata, TensorRecord};\n\
         fn main() { let _ = Data::try_new(); }",
    );
    assert!(!invalid.status.success());
    let stderr = String::from_utf8_lossy(&invalid.stderr);
    assert!(stderr.contains("private"), "{stderr}");
    assert!(stderr.contains("data"), "{stderr}");
}
use emel_gguf as _;
use emel_token as _;
