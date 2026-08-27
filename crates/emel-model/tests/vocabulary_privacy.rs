//! External vocabulary actor API boundary checks.

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
        "emel-model-vocabulary-privacy-{case}-{}",
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&project);
    fs::create_dir_all(project.join("src")).unwrap();
    fs::write(
        project.join("Cargo.toml"),
        format!(
            "[package]\nname = \"vocabulary-privacy-{case}\"\nversion = \"0.0.0\"\nedition = \"2024\"\n\
             [dependencies]\nemel-model = {{ path = \"{}\", default-features = false }}\n[workspace]\n",
            cargo_path(&root),
        ),
    )
    .unwrap();
    fs::write(project.join("src/main.rs"), source).unwrap();
    let output = Command::new(env!("CARGO"))
        .args(["check", "--offline", "--quiet"])
        .env("CARGO_TARGET_DIR", root.join("../../target/privacy"))
        .current_dir(&project)
        .output()
        .unwrap();
    let _ = fs::remove_dir_all(project);
    output
}

fn cargo_path(path: &Path) -> String {
    path.to_string_lossy().replace('\\', "\\\\")
}

#[test]
fn generated_machine_and_owned_vocabulary_are_private() {
    for (case, source) in [
        (
            "machine",
            "fn main() { let _ = core::mem::size_of::<emel_model::vocabulary::sm::VocabularyLoaderStates>(); }",
        ),
        (
            "data",
            "fn main() { let _ = core::mem::size_of::<emel_model::data::Vocab>(); }",
        ),
    ] {
        let output = check_project(case, source);
        assert!(!output.status.success(), "private surface compiled: {case}");
    }
}

#[test]
fn callback_borrows_cannot_escape_the_dispatch() {
    let output = check_project(
        "borrow-escape",
        "use emel_model::vocabulary::{event::WithInfo, Loader};\n\
         fn main() {\n\
           let mut loader = Loader::try_new().unwrap();\n\
           let _: &[u8] = loader.process_event(WithInfo::new(|info| info.model_name)).unwrap();\n\
         }",
    );
    assert!(
        !output.status.success(),
        "callback borrow escaped actor dispatch"
    );
}
use emel_kernels as _;
