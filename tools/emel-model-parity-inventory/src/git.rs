use std::fs;
use std::io::Write as _;
use std::path::Path;
use std::process::{Command, Stdio};
use std::thread;
use std::time::{Duration, Instant};

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Blob {
    pub path: String,
    pub bytes: Vec<u8>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BlobIdentity {
    pub path: String,
    pub oid: String,
}

pub fn output(repo: &Path, arguments: &[&str], deadline: Instant) -> Result<Vec<u8>, String> {
    let label = format!("git {}", arguments.join(" "));
    let mut command = Command::new("git");
    command.arg("-C").arg(repo).args(arguments);
    run_command_with_deadline(&mut command, &label, deadline)
}

fn run_command_with_deadline(
    command: &mut Command,
    label: &str,
    deadline: Instant,
) -> Result<Vec<u8>, String> {
    run_command_with_optional_input(command, label, deadline, None)
}

fn run_command_with_optional_input(
    command: &mut Command,
    label: &str,
    deadline: Instant,
    input: Option<&[u8]>,
) -> Result<Vec<u8>, String> {
    if Instant::now() >= deadline {
        return Err(format!("{label} timed out before process launch"));
    }
    let stdout = tempfile::NamedTempFile::new()
        .map_err(|error| format!("cannot create {label} stdout capture: {error}"))?;
    let stderr = tempfile::NamedTempFile::new()
        .map_err(|error| format!("cannot create {label} stderr capture: {error}"))?;
    let stdin = input
        .map(|bytes| {
            let mut file = tempfile::NamedTempFile::new()
                .map_err(|error| format!("cannot create {label} stdin capture: {error}"))?;
            file.write_all(bytes)
                .map_err(|error| format!("cannot write {label} stdin capture: {error}"))?;
            file.as_file_mut()
                .sync_all()
                .map_err(|error| format!("cannot sync {label} stdin capture: {error}"))?;
            Ok::<_, String>(file)
        })
        .transpose()?;
    let child_stdin = stdin.as_ref().map_or_else(
        || Ok(Stdio::null()),
        |file| {
            file.reopen()
                .map(Stdio::from)
                .map_err(|error| format!("cannot open {label} stdin capture: {error}"))
        },
    )?;
    command
        .stdin(child_stdin)
        .stdout(
            stdout
                .reopen()
                .map_err(|error| format!("cannot open {label} stdout capture: {error}"))?,
        )
        .stderr(
            stderr
                .reopen()
                .map_err(|error| format!("cannot open {label} stderr capture: {error}"))?,
        );
    let mut child = command
        .spawn()
        .map_err(|error| format!("failed to execute {label}: {error}"))?;
    let status = loop {
        if let Some(status) = child
            .try_wait()
            .map_err(|error| format!("cannot poll {label}: {error}"))?
        {
            break status;
        }
        let now = Instant::now();
        if now >= deadline {
            let kill_error = child.kill().err();
            let status = child
                .wait()
                .map_err(|error| format!("cannot reap timed-out {label}: {error}"))?;
            let captured_stdout = fs::read(stdout.path())
                .map_err(|error| format!("cannot read {label} stdout capture: {error}"))?;
            let captured_stderr = fs::read(stderr.path())
                .map_err(|error| format!("cannot read {label} stderr capture: {error}"))?;
            return Err(format!(
                "{label} timed out; kill_error={}; status={status}; stdout={}; stderr={}",
                kill_error
                    .as_ref()
                    .map_or_else(|| "none".to_owned(), ToString::to_string),
                String::from_utf8_lossy(&captured_stdout).trim(),
                String::from_utf8_lossy(&captured_stderr).trim(),
            ));
        }
        thread::sleep(Duration::from_millis(10).min(deadline.saturating_duration_since(now)));
    };
    let captured_stdout = fs::read(stdout.path())
        .map_err(|error| format!("cannot read {label} stdout capture: {error}"))?;
    let captured_stderr = fs::read(stderr.path())
        .map_err(|error| format!("cannot read {label} stderr capture: {error}"))?;
    if !status.success() {
        return Err(format!(
            "{label} failed with {status}: stdout={}; stderr={}",
            String::from_utf8_lossy(&captured_stdout).trim(),
            String::from_utf8_lossy(&captured_stderr).trim(),
        ));
    }
    Ok(captured_stdout)
}

pub fn text(repo: &Path, arguments: &[&str], deadline: Instant) -> Result<String, String> {
    String::from_utf8(output(repo, arguments, deadline)?)
        .map_err(|error| format!("git emitted non-UTF-8 text: {error}"))
}

pub fn tree(repo: &Path, commit: &str, path: &str, deadline: Instant) -> Result<String, String> {
    Ok(
        text(repo, &["rev-parse", &format!("{commit}:{path}")], deadline)?
            .trim()
            .to_owned(),
    )
}

pub fn resolved_commit(repo: &Path, commit: &str, deadline: Instant) -> Result<String, String> {
    let mut commit_object = commit.to_owned();
    commit_object.push('^');
    commit_object.push('{');
    commit_object.push_str("commit");
    commit_object.push('}');
    let resolved = text(
        repo,
        &["rev-parse", "--verify", "--end-of-options", &commit_object],
        deadline,
    )?;
    let resolved = resolved.trim();
    validate_oid(resolved)?;
    Ok(resolved.to_owned())
}

pub fn blob_identities(
    repo: &Path,
    commit: &str,
    roots: &[&str],
    deadline: Instant,
) -> Result<Vec<BlobIdentity>, String> {
    let mut result = Vec::new();
    for root in roots {
        let listing = output(repo, &["ls-tree", "-r", "-z", commit, root], deadline)?;
        result.extend(parse_blob_identities(&listing, None)?);
    }
    finish_blob_identities(result)
}

/// Lists an immutable tree object and prefixes every returned path with its bound owner path.
pub fn blob_identities_from_tree(
    repo: &Path,
    tree_oid: &str,
    root: &str,
    deadline: Instant,
) -> Result<Vec<BlobIdentity>, String> {
    validate_oid(tree_oid)?;
    validate_path(root)?;
    let listing = output(repo, &["ls-tree", "-r", "-z", tree_oid], deadline)?;
    finish_blob_identities(parse_blob_identities(&listing, Some(root))?)
}

fn parse_blob_identities(
    listing: &[u8],
    root_prefix: Option<&str>,
) -> Result<Vec<BlobIdentity>, String> {
    let mut result = Vec::new();
    for entry in listing
        .split(|byte| *byte == 0)
        .filter(|entry| !entry.is_empty())
    {
        let entry = std::str::from_utf8(entry)
            .map_err(|error| format!("non-UTF-8 Git tree entry: {error}"))?;
        let (metadata, relative_path) = entry
            .split_once('\t')
            .ok_or_else(|| format!("malformed Git tree entry: {entry:?}"))?;
        let mut fields = metadata.split(' ');
        let mode = fields.next().ok_or("Git tree entry is missing mode")?;
        let object_type = fields.next().ok_or("Git tree entry is missing type")?;
        let oid = fields.next().ok_or("Git tree entry is missing object ID")?;
        if fields.next().is_some() {
            return Err(format!("malformed Git tree metadata: {metadata:?}"));
        }
        let path = root_prefix.map_or_else(
            || relative_path.to_owned(),
            |root| format!("{root}/{relative_path}"),
        );
        if !matches!(mode, "100644" | "100755") || object_type != "blob" {
            return Err(format!(
                "unsupported Git entry for Rust preflight: {mode} {object_type} {path}"
            ));
        }
        validate_path(&path)?;
        validate_oid(oid)?;
        result.push(BlobIdentity {
            path,
            oid: oid.to_owned(),
        });
    }
    Ok(result)
}

fn finish_blob_identities(mut result: Vec<BlobIdentity>) -> Result<Vec<BlobIdentity>, String> {
    result.sort_by(|left, right| left.path.cmp(&right.path));
    if result.windows(2).any(|pair| pair[0].path == pair[1].path) {
        return Err("duplicate Git path in blob identity manifest".to_owned());
    }
    if result.is_empty() {
        return Err("Git blob identity manifest is empty".to_owned());
    }
    Ok(result)
}

pub fn blob_bytes(repo: &Path, oid: &str, deadline: Instant) -> Result<Vec<u8>, String> {
    validate_oid(oid)?;
    output(repo, &["cat-file", "blob", oid], deadline)
}

fn batch_blob_bytes(
    repo: &Path,
    identities: &[BlobIdentity],
    deadline: Instant,
) -> Result<Vec<Vec<u8>>, String> {
    let mut queries = Vec::new();
    for identity in identities {
        validate_oid(&identity.oid)?;
        queries.extend_from_slice(identity.oid.as_bytes());
        queries.push(b'\n');
    }
    let mut command = Command::new("git");
    command.arg("-C").arg(repo).args(["cat-file", "--batch"]);
    let output = run_command_with_optional_input(
        &mut command,
        "git cat-file --batch",
        deadline,
        Some(&queries),
    )?;
    parse_batch_blob_output(identities, &output)
}

fn parse_batch_blob_output(
    identities: &[BlobIdentity],
    output: &[u8],
) -> Result<Vec<Vec<u8>>, String> {
    let mut cursor = 0usize;
    let mut blobs = Vec::with_capacity(identities.len());
    for identity in identities {
        let header_end = output[cursor..]
            .iter()
            .position(|byte| *byte == b'\n')
            .and_then(|offset| cursor.checked_add(offset))
            .ok_or_else(|| format!("truncated Git batch header for {}", identity.path))?;
        let header = std::str::from_utf8(&output[cursor..header_end])
            .map_err(|error| format!("non-UTF-8 Git batch header: {error}"))?;
        let mut fields = header.split(' ');
        let oid = fields
            .next()
            .ok_or("Git batch header is missing object ID")?;
        let object_type = fields
            .next()
            .ok_or("Git batch header is missing object type")?;
        let size = fields
            .next()
            .ok_or("Git batch header is missing object size")?
            .parse::<usize>()
            .map_err(|error| format!("invalid Git batch object size: {error}"))?;
        if fields.next().is_some() {
            return Err(format!("malformed Git batch header: {header:?}"));
        }
        if oid != identity.oid || object_type != "blob" {
            return Err(format!(
                "Git batch identity mismatch for {}: expected {} blob, got {oid} {object_type}",
                identity.path, identity.oid
            ));
        }
        let body_start = header_end
            .checked_add(1)
            .ok_or("Git batch body offset overflow")?;
        let body_end = body_start
            .checked_add(size)
            .ok_or("Git batch body size overflow")?;
        if output.get(body_end) != Some(&b'\n') {
            return Err(format!("truncated Git batch body for {}", identity.path));
        }
        blobs.push(output[body_start..body_end].to_vec());
        cursor = body_end.checked_add(1).ok_or("Git batch cursor overflow")?;
    }
    if cursor != output.len() {
        return Err("Git batch output contains trailing bytes".to_owned());
    }
    Ok(blobs)
}

pub fn blobs_from_identities(
    repo: &Path,
    identities: &[BlobIdentity],
    deadline: Instant,
) -> Result<Vec<Blob>, String> {
    let bytes = batch_blob_bytes(repo, identities, deadline)?;
    Ok(identities
        .iter()
        .cloned()
        .zip(bytes)
        .map(|(identity, bytes)| Blob {
            path: identity.path,
            bytes,
        })
        .collect())
}

pub fn blobs(
    repo: &Path,
    commit: &str,
    roots: &[&str],
    deadline: Instant,
) -> Result<Vec<Blob>, String> {
    let identities = blob_identities(repo, commit, roots, deadline)?;
    blobs_from_identities(repo, &identities, deadline)
}

pub fn validate_path(path: &str) -> Result<(), String> {
    if path.is_empty()
        || path.starts_with('/')
        || path.contains('\\')
        || path
            .split('/')
            .any(|part| part.is_empty() || part == "." || part == "..")
    {
        return Err(format!("non-canonical Git path: {path:?}"));
    }
    Ok(())
}

pub fn validate_oid(oid: &str) -> Result<(), String> {
    if !matches!(oid.len(), 40 | 64)
        || !oid
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(format!("invalid Git object ID: {oid:?}"));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    #[cfg(unix)]
    use std::fs;
    #[cfg(unix)]
    use std::os::unix::fs::PermissionsExt as _;
    #[cfg(unix)]
    use std::path::Path;
    use std::process::Command;
    use std::time::{Duration, Instant};

    use super::*;

    #[test]
    fn child_deadline_kills_reaps_and_preserves_diagnostics() {
        let mut command = Command::new("sh");
        command.args([
            "-c",
            "trap '' TERM; printf stdout-proof; printf stderr-proof >&2; exec sleep 30",
        ]);
        let started = Instant::now();
        let error = run_command_with_deadline(
            &mut command,
            "deadline-fixture",
            started + Duration::from_millis(100),
        )
        .unwrap_err();
        assert!(started.elapsed() < Duration::from_secs(5));
        assert!(error.contains("timed out"), "{error}");
        assert!(error.contains("stdout-proof"), "{error}");
        assert!(error.contains("stderr-proof"), "{error}");
    }

    #[cfg(unix)]
    #[test]
    fn exact_blob_batch_never_invokes_git_filters() {
        let temp = tempfile::tempdir().unwrap();
        let repo = temp.path().join("repo");
        fs::create_dir(&repo).unwrap();
        git(&repo, &["init", "-q"]);
        git(
            &repo,
            &["config", "user.email", "inventory@example.invalid"],
        );
        git(&repo, &["config", "user.name", "Inventory Test"]);
        fs::create_dir(repo.join("src")).unwrap();
        fs::write(repo.join(".gitattributes"), "*.hpp filter=must-not-run\n").unwrap();
        fs::write(repo.join("src/model.hpp"), b"pinned-object-bytes\n").unwrap();
        git(&repo, &["add", ".gitattributes", "src/model.hpp"]);
        git(&repo, &["commit", "-qm", "fixture"]);

        let marker = temp.path().join("filter-invoked");
        let filter = temp.path().join("filter.sh");
        fs::write(
            &filter,
            format!(
                "#!/bin/sh\nprintf invoked > '{}'\nexit 99\n",
                marker.display()
            ),
        )
        .unwrap();
        let mut permissions = fs::metadata(&filter).unwrap().permissions();
        permissions.set_mode(0o755);
        fs::set_permissions(&filter, permissions).unwrap();
        git(
            &repo,
            &[
                "config",
                "filter.must-not-run.process",
                filter.to_str().unwrap(),
            ],
        );
        git(&repo, &["config", "filter.must-not-run.required", "true"]);

        let blobs = blobs(
            &repo,
            "HEAD",
            &["src"],
            Instant::now() + Duration::from_secs(10),
        )
        .unwrap();
        assert_eq!(
            blobs,
            [Blob {
                path: "src/model.hpp".into(),
                bytes: b"pinned-object-bytes\n".to_vec(),
            }]
        );
        assert!(!marker.exists(), "Git filter process was invoked");
    }

    #[cfg(unix)]
    #[test]
    fn blob_manifest_rejects_nonregular_entries_in_selected_root() {
        let temp = tempfile::tempdir().unwrap();
        let repo = temp.path().join("repo");
        fs::create_dir(&repo).unwrap();
        git(&repo, &["init", "-q"]);
        git(
            &repo,
            &["config", "user.email", "inventory@example.invalid"],
        );
        git(&repo, &["config", "user.name", "Inventory Test"]);
        fs::create_dir(repo.join("tests")).unwrap();
        std::os::unix::fs::symlink("missing-target", repo.join("tests/nonregular")).unwrap();
        git(&repo, &["add", "tests/nonregular"]);
        git(&repo, &["commit", "-qm", "nonregular fixture"]);

        let error = blobs(
            &repo,
            "HEAD",
            &["tests"],
            Instant::now() + Duration::from_secs(10),
        )
        .unwrap_err();
        assert!(error.contains("120000 blob tests/nonregular"), "{error}");
    }

    #[cfg(unix)]
    fn git(repo: &Path, arguments: &[&str]) {
        let status = Command::new("git")
            .arg("-C")
            .arg(repo)
            .args(arguments)
            .status()
            .unwrap();
        assert!(status.success(), "git {arguments:?} failed with {status}");
    }
}
