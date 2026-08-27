//! External tokenizer profile ownership and injection boundary checks.

use allocation_counter as _;
use emel_token as _;
use sml as _;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

fn check_project(case: &str, source: &str) -> Output {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let project = std::env::temp_dir().join(format!(
        "emel-token-profile-privacy-{case}-{}",
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&project);
    fs::create_dir_all(project.join("src")).unwrap();
    fs::write(
        project.join("Cargo.toml"),
        format!(
            "[package]\nname = \"token-profile-privacy-{case}\"\nversion = \"0.0.0\"\nedition = \"2024\"\n\
             [dependencies]\nemel-token = {{ path = \"{}\" }}\n[workspace]\n",
            cargo_path(&root)
        ),
    )
    .unwrap();
    fs::write(project.join("src/main.rs"), source).unwrap();
    let output = Command::new(env!("CARGO"))
        .args(["check", "--offline", "--quiet"])
        .env(
            "CARGO_TARGET_DIR",
            root.join("../../target/token-profile-privacy"),
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
fn actor_events_and_replacement_dependency_are_public() {
    let output = check_project(
        "public",
        "use emel_token::profile::{Dependency, Resolver, event::{Error, Resolve, Resolved}};\n\
         struct Replacement;\n\
         impl Dependency for Replacement {\n\
             fn process_event(&mut self, _: Resolve<'_>) -> Result<Resolved, Error> {\n\
                 Resolver::new().process_event(Resolve::new(\"none\", \"default\"))\n\
             }\n\
         }\n\
         fn main() { let mut dependency = Replacement; let _ = Dependency::process_event(&mut dependency, Resolve::new(\"x\", \"y\")); }",
    );
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn pre_identity_machine_and_actor_storage_remain_opaque() {
    let output = check_project(
        "private",
        "use emel_token::profile::{Resolver, event::PreId, sm};\n\
         fn main() { let resolver = Resolver::new(); let _ = PreId(1); let _ = resolver.machine; }",
    );
    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("private"), "{stderr}");
    assert!(stderr.contains("PreId"), "{stderr}");
    assert!(
        stderr.contains("machine") || stderr.contains("sm"),
        "{stderr}"
    );
}
