//! External model-data ownership boundary checks.

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
    let project = root
        .join("../../.artifacts/model-data-privacy")
        .join(format!("{case}-{}", std::process::id()));
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
fn model_data_bridge_is_public_but_raw_storage_remains_private() {
    let valid = check_project(
        "public",
        "use emel_model::bridge::Data;\n\
         fn main() {\n\
             let data = Data::try_new().unwrap();\n\
             let _ = data.mimi_binding_input();\n\
             let _ = data.architecture_name();\n\
         }",
    );
    assert!(
        valid.status.success(),
        "{}",
        String::from_utf8_lossy(&valid.stderr)
    );

    let invalid = check_project(
        "private-fields",
        "use emel_model::bridge::Data;\n\
         fn main() {\n\
             let data = Data::try_new().unwrap();\n\
             let _ = data.tensors;\n\
         }",
    );
    assert!(!invalid.status.success());
    let stderr = String::from_utf8_lossy(&invalid.stderr);
    assert!(stderr.contains("private"), "{stderr}");
    assert!(stderr.contains("tensors"), "{stderr}");
}
use emel_gguf as _;
use emel_kernels as _;
use emel_token as _;
