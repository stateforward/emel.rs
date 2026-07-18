//! External catalog actor ownership and identity boundary checks.

use allocation_counter as _;
use emel_gguf as _;
use emel_io as _;
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
        "emel-model-catalog-privacy-{case}-{}",
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&project);
    fs::create_dir_all(project.join("src")).unwrap();
    fs::write(
        project.join("Cargo.toml"),
        format!(
            "[package]\nname = \"catalog-privacy-{case}\"\nversion = \"0.0.0\"\nedition = \"2024\"\n\
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
            root.join("../../target/model-catalog-privacy"),
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
fn public_actor_and_events_compile() {
    let output = check_project(
        "public",
        "use emel_model::catalog::{Catalog, event::{BindStorage, SealModel, Storage, TensorInput}};\n\
         fn main() {\n\
           let mut storage = Storage::with_capacity(1, 1, 1).unwrap();\n\
           storage.push_tensor(TensorInput::new(b\"x\", 0, 1, [1; 4], 4, true)).unwrap();\n\
           let mut actor = Catalog::try_new().unwrap();\n\
           actor.process_event(BindStorage::new(storage)).unwrap();\n\
           let _ = actor.process_event(SealModel::new()).unwrap();\n\
         }",
    );
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn private_records_storage_fields_identities_and_state_are_inaccessible() {
    for (case, source) in [
        (
            "record",
            "use emel_model::catalog::storage::Record; fn main() {}",
        ),
        (
            "identity",
            "use emel_model::catalog::event::ModelIdentity; fn main() { let _ = ModelIdentity { owner: 1, generation: 1 }; }",
        ),
        (
            "storage",
            "use emel_model::catalog::event::Storage; fn main() { let value = Storage::with_capacity(1, 1, 1).unwrap(); let _ = value.records; }",
        ),
        (
            "state",
            "fn main() { let value = emel_model::catalog::Catalog::try_new().unwrap(); let _ = value.state(); }",
        ),
    ] {
        let output = check_project(case, source);
        assert!(!output.status.success(), "{case} unexpectedly compiled");
    }
}
