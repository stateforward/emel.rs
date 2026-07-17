use std::collections::{BTreeMap, BTreeSet};
use std::ffi::OsString;
use std::fmt::Write as _;
use std::fs;
#[cfg(not(unix))]
use std::fs::OpenOptions;
use std::fs::{File, TryLockError};
use std::io::Write as _;
use std::io::{ErrorKind, Read as _};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

#[cfg(unix)]
use std::os::unix::fs::MetadataExt as _;

#[cfg(unix)]
use cap_std::ambient_authority;
#[cfg(unix)]
use cap_std::fs::{Dir as CapabilityDir, MetadataExt as _, OpenOptions as CapabilityOpenOptions};
#[cfg(unix)]
use cap_tempfile::TempFile as CapabilityTempFile;

#[cfg(windows)]
use std::os::windows::fs::OpenOptionsExt as _;
#[cfg(windows)]
use winapi_util::{Handle as WindowsHandle, file as windows_file};
#[cfg(windows)]
use windows_sys::Win32::Storage::FileSystem::{
    FILE_FLAG_BACKUP_SEMANTICS, FILE_SHARE_READ, FILE_SHARE_WRITE,
};

use crate::clang;
use crate::git::{self, Blob, BlobIdentity};
use crate::hash::sha256;
use crate::scan::{self, Extracted};
use crate::schema::{self, COVERAGE_HEADER, GAP_HEADER, INVENTORY_HEADER, InventoryRow};
use serde::{Deserialize, Serialize};
use sha2::{Digest as _, Sha256};

const SOURCE_MODEL_PATH: &str = "src/emel/model";
const SOURCE_TEST_PATH: &str = "tests/model";
const RUST_PATH: &str = "crates/emel-model";
const EXPECTED_SOURCE_COMMIT: &str = "843a117386ef17dc5a50549bbfc821074c2141d6";
const EXPECTED_MODEL_TREE: &str = "278b7b20545b630be33bee8bed0cb4c8db8990c3";
const EXPECTED_TEST_TREE: &str = "5e79d49dd6dd26411cfa480aea2d7acbd863ba73";
const PINNED_AST_COVERAGE_SHA256: &str =
    "240554f572f10c2469807e2e8f09325494d7a878e1eb608a052ef297b3a81ecc";
const GENERATION_JOURNAL: &str = ".emel-model-inventory-generation";
const INITIALIZING_GENERATION_JOURNAL: &str = ".emel-model-inventory-generation-initializing";
const GENERATION_LOCK: &str = ".emel-model-inventory.lock";
const GENERATION_TOKEN: &str = ".emel-model-inventory.generation";
const GENERATION_TOKEN_SCHEMA: &str = "emel-model-inventory-generation/v1";
const LOCK_ACQUIRE_TIMEOUT: Duration = Duration::from_secs(5);
const PUBLISHED_ARTIFACT_NAMES: &[&str] = &[
    "open-gaps.tsv",
    "primary-rust-inventory.tsv",
    "primary-source-inventory.tsv",
    "proof-coverage-matrix.tsv",
    "reconciled-inventory.tsv",
    "terminal-rust-inventory.tsv",
    "terminal-source-inventory.tsv",
];
const DEFAULT_GIT_TIMEOUT: Duration = Duration::from_secs(300);
const COMPONENT_ROLE_FILES: &[&str] = &[
    "actions", "context", "detail", "errors", "events", "guards", "mod", "sm", "tests",
];

/// Immutable extraction inputs.
#[derive(Clone, Debug)]
pub struct ExtractOptions {
    /// Repository containing the pinned C++ reference commit.
    pub source_repo: PathBuf,
    /// Exact C++ reference commit.
    pub source_commit: String,
    /// Repository containing the Rust workspace.
    pub rust_repo: PathBuf,
    /// Exact committed Rust input tree.
    pub rust_commit: String,
    /// Pinned compilation database used for C++ parsing.
    pub compile_commands: PathBuf,
    /// Destination for the seven root parity artifacts.
    pub artifact_dir: PathBuf,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct Artifacts {
    files: BTreeMap<String, Vec<u8>>,
    coverage: Vec<u8>,
    external_includes: Vec<u8>,
}

#[cfg(unix)]
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct FileIdentity {
    device: u64,
    inode: u64,
}

#[cfg(windows)]
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct FileIdentity {
    volume_serial_number: u64,
    file_index: u64,
}

#[cfg(all(not(unix), not(windows)))]
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct FileIdentity;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct FileObservation {
    identity: FileIdentity,
    link_count: u64,
}

#[cfg(unix)]
fn observe_open_file(file: &File) -> Result<FileObservation, String> {
    let metadata = file
        .metadata()
        .map_err(|error| format!("cannot inspect opened file: {error}"))?;
    Ok(FileObservation {
        identity: FileIdentity {
            device: metadata.dev(),
            inode: metadata.ino(),
        },
        link_count: metadata.nlink(),
    })
}

#[cfg(windows)]
fn observe_open_file(file: &File) -> Result<FileObservation, String> {
    let information = windows_file::information(file)
        .map_err(|error| format!("cannot inspect opened Windows file: {error}"))?;
    Ok(FileObservation {
        identity: FileIdentity {
            volume_serial_number: information.volume_serial_number(),
            file_index: information.file_index(),
        },
        link_count: information.number_of_links(),
    })
}

#[cfg(all(not(unix), not(windows)))]
fn observe_open_file(_file: &File) -> Result<FileObservation, String> {
    Err("stable file identity is unsupported on this target".to_owned())
}

#[cfg(unix)]
type OwnerIdentityHandle = CapabilityDir;

#[cfg(windows)]
type OwnerIdentityHandle = WindowsHandle;

#[cfg(all(not(unix), not(windows)))]
type OwnerIdentityHandle = ();

#[cfg(unix)]
fn bind_owner_identity(
    path: &Path,
    metadata: &fs::Metadata,
) -> Result<(FileIdentity, OwnerIdentityHandle), String> {
    let capability =
        CapabilityDir::open_ambient_dir(path, ambient_authority()).map_err(|error| {
            format!(
                "cannot retain Unix generation owner capability {}: {error}",
                path.display()
            )
        })?;
    let opened = capability.dir_metadata().map_err(|error| {
        format!(
            "cannot inspect retained Unix generation owner {}: {error}",
            path.display()
        )
    })?;
    let path_identity = FileIdentity {
        device: metadata.dev(),
        inode: metadata.ino(),
    };
    let opened_identity = FileIdentity {
        device: opened.dev(),
        inode: opened.ino(),
    };
    if path_identity != opened_identity {
        return Err(format!(
            "generation owner changed while retaining Unix capability: {}",
            path.display()
        ));
    }
    Ok((opened_identity, capability))
}

#[cfg(windows)]
fn open_windows_owner_identity(path: &Path) -> Result<WindowsHandle, String> {
    OpenOptions::new()
        .read(true)
        .share_mode(FILE_SHARE_READ | FILE_SHARE_WRITE)
        .custom_flags(FILE_FLAG_BACKUP_SEMANTICS)
        .open(path)
        .map(WindowsHandle::from_file)
        .map_err(|error| {
            format!(
                "cannot open stable Windows generation owner {} without delete sharing: {error}",
                path.display()
            )
        })
}

#[cfg(windows)]
fn bind_owner_identity(
    path: &Path,
    _metadata: &fs::Metadata,
) -> Result<(FileIdentity, OwnerIdentityHandle), String> {
    let handle = open_windows_owner_identity(path)?;
    let information = windows_file::information(&handle).map_err(|error| {
        format!(
            "cannot inspect stable Windows generation owner {}: {error}",
            path.display()
        )
    })?;
    Ok((
        FileIdentity {
            volume_serial_number: information.volume_serial_number(),
            file_index: information.file_index(),
        },
        handle,
    ))
}

#[cfg(all(not(unix), not(windows)))]
fn bind_owner_identity(
    _path: &Path,
    _metadata: &fs::Metadata,
) -> Result<(FileIdentity, OwnerIdentityHandle), String> {
    Err("stable generation owner identity is unsupported on this target".to_owned())
}

#[cfg(unix)]
#[allow(clippy::unnecessary_wraps)]
fn revalidate_owner_identity(path: &Path, metadata: &fs::Metadata) -> Result<FileIdentity, String> {
    let _ = path;
    Ok(FileIdentity {
        device: metadata.dev(),
        inode: metadata.ino(),
    })
}

#[cfg(windows)]
fn revalidate_owner_identity(
    path: &Path,
    _metadata: &fs::Metadata,
) -> Result<FileIdentity, String> {
    let handle = open_windows_owner_identity(path)?;
    let information = windows_file::information(&handle).map_err(|error| {
        format!(
            "cannot revalidate stable Windows generation owner {}: {error}",
            path.display()
        )
    })?;
    Ok(FileIdentity {
        volume_serial_number: information.volume_serial_number(),
        file_index: information.file_index(),
    })
}

#[cfg(all(not(unix), not(windows)))]
fn revalidate_owner_identity(
    _path: &Path,
    _metadata: &fs::Metadata,
) -> Result<FileIdentity, String> {
    Err("stable generation owner identity is unsupported on this target".to_owned())
}

fn identities_alias(left: FileIdentity, right: FileIdentity) -> bool {
    left == right
}

fn reject_hardlinks(observation: FileObservation, path: &Path) -> Result<(), String> {
    if observation.link_count == 1 {
        return Ok(());
    }
    Err(format!(
        "stable generation file has multiple hard links: {}",
        path.display()
    ))
}

#[derive(Debug)]
struct BoundOwner {
    path: PathBuf,
    identity: FileIdentity,
    // Unix mutation and read paths remain relative to this retained directory capability even if
    // the ambient pathname is renamed or replaced.
    #[cfg(unix)]
    capability: OwnerIdentityHandle,
    // Windows file indexes are stable only while the identifying handle remains open.
    #[cfg(windows)]
    _identity_handle: OwnerIdentityHandle,
    lock_path: PathBuf,
    lock_identity: FileIdentity,
}

#[derive(Debug)]
struct GenerationLayout {
    artifact_owner: PathBuf,
    snapshot_owner: PathBuf,
    owners: Vec<BoundOwner>,
}

#[derive(Debug)]
struct UnlockedGenerationLock {
    path: PathBuf,
    file: File,
}

/// Opens every Unix owner ancestor, including the filesystem root, in canonical path order.
///
/// These advisory directory locks deliberately serialize all cooperative writers that share an
/// ancestor (ultimately `/`). That broad serialization is the cost of keeping one rename-stable
/// cooperative lock domain when an owner pathname is renamed and recreated. Mutations still use
/// the separately retained owner capabilities, so a non-cooperative rename cannot redirect I/O.
#[cfg(unix)]
fn open_ancestor_lock_capabilities(
    owner_paths: &[PathBuf],
) -> Result<Vec<UnlockedGenerationLock>, String> {
    let paths = owner_paths
        .iter()
        .flat_map(|owner| owner.ancestors().map(Path::to_path_buf))
        .collect::<BTreeSet<_>>();
    paths
        .into_iter()
        .map(|path| {
            let before = fs::symlink_metadata(&path).map_err(|error| {
                format!(
                    "cannot inspect generation owner ancestor {}: {error}",
                    path.display()
                )
            })?;
            if before.file_type().is_symlink() || !before.file_type().is_dir() {
                return Err(format!(
                    "generation owner ancestor is not a directory: {}",
                    path.display()
                ));
            }
            let file = File::open(&path).map_err(|error| {
                format!(
                    "cannot retain generation owner ancestor {}: {error}",
                    path.display()
                )
            })?;
            let opened = file.metadata().map_err(|error| {
                format!(
                    "cannot inspect retained generation owner ancestor {}: {error}",
                    path.display()
                )
            })?;
            let after = fs::symlink_metadata(&path).map_err(|error| {
                format!(
                    "cannot reinspect generation owner ancestor {}: {error}",
                    path.display()
                )
            })?;
            let before_identity = (before.dev(), before.ino());
            let opened_identity = (opened.dev(), opened.ino());
            let after_identity = (after.dev(), after.ino());
            if before_identity != opened_identity || after_identity != opened_identity {
                return Err(format!(
                    "generation owner ancestor changed while opening: {}",
                    path.display()
                ));
            }
            Ok(UnlockedGenerationLock { path, file })
        })
        .collect()
}

#[cfg(not(unix))]
#[allow(clippy::missing_const_for_fn, clippy::unnecessary_wraps)]
fn open_ancestor_lock_capabilities(
    _owner_paths: &[PathBuf],
) -> Result<Vec<UnlockedGenerationLock>, String> {
    Ok(Vec::new())
}

#[cfg_attr(not(unix), allow(clippy::unused_self))]
impl GenerationLayout {
    fn bind(
        artifact_dir: &Path,
        rust_repo: &Path,
    ) -> Result<(Self, Vec<UnlockedGenerationLock>), String> {
        let snapshot = snapshot_path(rust_repo);
        let snapshot_owner = snapshot
            .parent()
            .ok_or("AST coverage snapshot lacks an owner directory")?;
        Self::bind_owners(artifact_dir, snapshot_owner, &[])
    }

    fn bind_owners(
        artifact_owner: &Path,
        snapshot_owner: &Path,
        additional_owners: &[PathBuf],
    ) -> Result<(Self, Vec<UnlockedGenerationLock>), String> {
        let artifact_owner = canonical_owner(artifact_owner)?;
        let snapshot_owner = canonical_owner(snapshot_owner)?;
        let mut owner_paths = vec![artifact_owner.clone(), snapshot_owner.clone()];
        for owner in additional_owners {
            owner_paths.push(canonical_owner(owner)?);
        }
        owner_paths.sort();
        owner_paths.dedup();

        let mut owners_and_files = owner_paths
            .into_iter()
            .map(open_bound_owner)
            .collect::<Result<Vec<_>, _>>()?;
        owners_and_files.sort_by(|(left, _), (right, _)| left.lock_path.cmp(&right.lock_path));
        if owners_and_files
            .windows(2)
            .any(|pair| identities_alias(pair[0].0.lock_identity, pair[1].0.lock_identity))
        {
            return Err("generation lock aliases share one filesystem identity".to_owned());
        }
        let bound_owner_paths = owners_and_files
            .iter()
            .map(|(owner, _)| owner.path.clone())
            .collect::<Vec<_>>();
        let ancestor_locks = open_ancestor_lock_capabilities(&bound_owner_paths)?;
        let (owners, files): (Vec<_>, Vec<_>) = owners_and_files.into_iter().unzip();
        let mut lock_files = ancestor_locks;
        lock_files.extend(
            owners
                .iter()
                .zip(files)
                .map(|(owner, file)| UnlockedGenerationLock {
                    path: owner.lock_path.clone(),
                    file,
                }),
        );
        Ok((
            Self {
                artifact_owner,
                snapshot_owner,
                owners,
            },
            lock_files,
        ))
    }

    fn artifact_path(&self, name: &str) -> PathBuf {
        self.artifact_owner.join(name)
    }

    fn snapshot_path(&self) -> PathBuf {
        self.snapshot_owner.join("pinned-ast-coverage.tsv")
    }

    fn external_snapshot_path(&self) -> PathBuf {
        self.snapshot_owner.join("pinned-external-includes.tsv")
    }

    #[cfg(unix)]
    fn bound_path(&self, path: &Path) -> Result<(&BoundOwner, PathBuf), String> {
        self.owners
            .iter()
            .filter_map(|owner| {
                path.strip_prefix(&owner.path)
                    .ok()
                    .map(|relative| (owner, relative.to_path_buf()))
            })
            .max_by_key(|(owner, _)| owner.path.components().count())
            .ok_or_else(|| {
                format!(
                    "path is outside every retained generation owner capability: {}",
                    path.display()
                )
            })
    }

    #[cfg(unix)]
    fn capability_relative(relative: &Path) -> &Path {
        if relative.as_os_str().is_empty() {
            Path::new(".")
        } else {
            relative
        }
    }

    fn try_exists(&self, path: &Path) -> Result<bool, String> {
        #[cfg(unix)]
        {
            let (owner, relative) = self.bound_path(path)?;
            owner
                .capability
                .try_exists(Self::capability_relative(&relative))
                .map_err(|error| format!("cannot inspect {}: {error}", path.display()))
        }
        #[cfg(not(unix))]
        {
            path.try_exists()
                .map_err(|error| format!("cannot inspect {}: {error}", path.display()))
        }
    }

    fn read(&self, path: &Path) -> Result<Vec<u8>, String> {
        #[cfg(unix)]
        {
            let (owner, relative) = self.bound_path(path)?;
            owner
                .capability
                .read(Self::capability_relative(&relative))
                .map_err(|error| format!("cannot read {}: {error}", path.display()))
        }
        #[cfg(not(unix))]
        {
            fs::read(path).map_err(|error| format!("cannot read {}: {error}", path.display()))
        }
    }

    fn read_dir_names(&self, path: &Path) -> Result<Vec<OsString>, String> {
        #[cfg(unix)]
        {
            let (owner, relative) = self.bound_path(path)?;
            owner
                .capability
                .read_dir(Self::capability_relative(&relative))
                .map_err(|error| format!("cannot list {}: {error}", path.display()))?
                .map(|entry| {
                    entry
                        .map(|entry| entry.file_name())
                        .map_err(|error| format!("cannot read {} entry: {error}", path.display()))
                })
                .collect()
        }
        #[cfg(not(unix))]
        {
            fs::read_dir(path)
                .map_err(|error| format!("cannot list {}: {error}", path.display()))?
                .map(|entry| {
                    entry
                        .map(|entry| entry.file_name())
                        .map_err(|error| format!("cannot read {} entry: {error}", path.display()))
                })
                .collect()
        }
    }

    fn create_file(&self, path: &Path, create_new: bool) -> Result<File, String> {
        #[cfg(unix)]
        {
            let (owner, relative) = self.bound_path(path)?;
            let mut options = CapabilityOpenOptions::new();
            options.write(true).create(true).truncate(!create_new);
            if create_new {
                options.create_new(true);
            }
            owner
                .capability
                .open_with(Self::capability_relative(&relative), &options)
                .map(cap_std::fs::File::into_std)
                .map_err(|error| format!("cannot create {}: {error}", path.display()))
        }
        #[cfg(not(unix))]
        {
            OpenOptions::new()
                .write(true)
                .create(true)
                .truncate(!create_new)
                .create_new(create_new)
                .open(path)
                .map_err(|error| format!("cannot create {}: {error}", path.display()))
        }
    }

    fn open_file(&self, path: &Path) -> Result<File, String> {
        #[cfg(unix)]
        {
            let (owner, relative) = self.bound_path(path)?;
            owner
                .capability
                .open(Self::capability_relative(&relative))
                .map(cap_std::fs::File::into_std)
                .map_err(|error| format!("cannot open {}: {error}", path.display()))
        }
        #[cfg(not(unix))]
        {
            File::open(path).map_err(|error| format!("cannot open {}: {error}", path.display()))
        }
    }

    fn create_dir(&self, path: &Path) -> Result<(), String> {
        #[cfg(unix)]
        {
            let (owner, relative) = self.bound_path(path)?;
            owner
                .capability
                .create_dir(Self::capability_relative(&relative))
                .map_err(|error| format!("cannot create directory {}: {error}", path.display()))
        }
        #[cfg(not(unix))]
        {
            fs::create_dir(path)
                .map_err(|error| format!("cannot create directory {}: {error}", path.display()))
        }
    }

    fn remove_dir_all(&self, path: &Path) -> Result<(), String> {
        #[cfg(unix)]
        {
            let (owner, relative) = self.bound_path(path)?;
            owner
                .capability
                .remove_dir_all(Self::capability_relative(&relative))
                .map_err(|error| format!("cannot remove directory {}: {error}", path.display()))
        }
        #[cfg(not(unix))]
        {
            fs::remove_dir_all(path)
                .map_err(|error| format!("cannot remove directory {}: {error}", path.display()))
        }
    }

    fn remove_file(&self, path: &Path) -> Result<(), String> {
        #[cfg(unix)]
        {
            let (owner, relative) = self.bound_path(path)?;
            owner
                .capability
                .remove_file(Self::capability_relative(&relative))
                .map_err(|error| format!("cannot remove {}: {error}", path.display()))
        }
        #[cfg(not(unix))]
        {
            fs::remove_file(path)
                .map_err(|error| format!("cannot remove {}: {error}", path.display()))
        }
    }

    fn rename(&self, from: &Path, to: &Path) -> Result<(), String> {
        #[cfg(unix)]
        {
            let (from_owner, from_relative) = self.bound_path(from)?;
            let (to_owner, to_relative) = self.bound_path(to)?;
            from_owner
                .capability
                .rename(
                    Self::capability_relative(&from_relative),
                    &to_owner.capability,
                    Self::capability_relative(&to_relative),
                )
                .map_err(|error| {
                    format!(
                        "cannot rename {} to {}: {error}",
                        from.display(),
                        to.display()
                    )
                })
        }
        #[cfg(not(unix))]
        {
            fs::rename(from, to).map_err(|error| {
                format!(
                    "cannot rename {} to {}: {error}",
                    from.display(),
                    to.display()
                )
            })
        }
    }

    fn copy(&self, from: &Path, to: &Path) -> Result<(), String> {
        #[cfg(unix)]
        {
            let (from_owner, from_relative) = self.bound_path(from)?;
            let (to_owner, to_relative) = self.bound_path(to)?;
            from_owner
                .capability
                .copy(
                    Self::capability_relative(&from_relative),
                    &to_owner.capability,
                    Self::capability_relative(&to_relative),
                )
                .map(|_| ())
                .map_err(|error| {
                    format!(
                        "cannot copy {} to {}: {error}",
                        from.display(),
                        to.display()
                    )
                })
        }
        #[cfg(not(unix))]
        {
            fs::copy(from, to).map(|_| ()).map_err(|error| {
                format!(
                    "cannot copy {} to {}: {error}",
                    from.display(),
                    to.display()
                )
            })
        }
    }

    #[cfg_attr(
        not(unix),
        allow(clippy::missing_const_for_fn, clippy::unnecessary_wraps)
    )]
    fn sync_directory(&self, path: &Path) -> Result<(), String> {
        #[cfg(unix)]
        {
            let (owner, relative) = self.bound_path(path)?;
            let directory = if relative.as_os_str().is_empty() {
                owner.capability.try_clone()
            } else {
                owner.capability.open_dir(&relative)
            }
            .map_err(|error| {
                format!(
                    "cannot open retained directory {} for sync: {error}",
                    path.display()
                )
            })?;
            directory
                .into_std_file()
                .sync_all()
                .map_err(|error| format!("cannot sync directory {}: {error}", path.display()))
        }
        #[cfg(not(unix))]
        {
            let _ = path;
            Ok(())
        }
    }

    fn sync_file(&self, path: &Path) -> Result<(), String> {
        self.open_file(path)?
            .sync_all()
            .map_err(|error| format!("cannot sync {}: {error}", path.display()))
    }

    fn verify(&self) -> Result<(), String> {
        for owner in &self.owners {
            let canonical = owner.path.canonicalize().map_err(|error| {
                format!(
                    "cannot revalidate generation owner {}: {error}",
                    owner.path.display()
                )
            })?;
            let metadata = fs::symlink_metadata(&owner.path).map_err(|error| {
                format!(
                    "cannot inspect generation owner {}: {error}",
                    owner.path.display()
                )
            })?;
            if canonical != owner.path
                || !metadata.file_type().is_dir()
                || revalidate_owner_identity(&owner.path, &metadata)? != owner.identity
            {
                return Err(format!(
                    "generation owner identity changed: {}",
                    owner.path.display()
                ));
            }
            #[cfg(unix)]
            let (_lock_file, lock_observation) = open_observed_capability_file(
                &owner.capability,
                Path::new(GENERATION_LOCK),
                &owner.lock_path,
            )?;
            #[cfg(not(unix))]
            let (_lock_file, lock_observation) = open_observed_regular_file(&owner.lock_path)?;
            reject_hardlinks(lock_observation, &owner.lock_path)?;
            if lock_observation.identity != owner.lock_identity {
                return Err(format!(
                    "generation lock identity changed: {}",
                    owner.lock_path.display()
                ));
            }
        }
        Ok(())
    }
}

fn canonical_owner(owner: &Path) -> Result<PathBuf, String> {
    fs::create_dir_all(owner).map_err(|error| {
        format!(
            "cannot create generation owner {}: {error}",
            owner.display()
        )
    })?;
    let canonical = owner.canonicalize().map_err(|error| {
        format!(
            "cannot canonicalize generation owner {}: {error}",
            owner.display()
        )
    })?;
    let metadata = fs::symlink_metadata(&canonical).map_err(|error| {
        format!(
            "cannot inspect generation owner {}: {error}",
            canonical.display()
        )
    })?;
    if !metadata.file_type().is_dir() {
        return Err(format!(
            "generation owner is not a directory: {}",
            canonical.display()
        ));
    }
    Ok(canonical)
}

#[cfg(not(unix))]
fn open_generation_lock(path: &Path, create_new: bool) -> std::io::Result<File> {
    let mut options = OpenOptions::new();
    options.read(true).write(true).create_new(create_new);
    #[cfg(windows)]
    options.share_mode(FILE_SHARE_READ | FILE_SHARE_WRITE);
    options.open(path)
}

#[cfg(unix)]
fn open_unix_generation_lock(owner: &CapabilityDir, create_new: bool) -> std::io::Result<File> {
    let mut options = CapabilityOpenOptions::new();
    options.read(true).write(true).create_new(create_new);
    owner
        .open_with(GENERATION_LOCK, &options)
        .map(cap_std::fs::File::into_std)
}

fn open_bound_owner(owner: PathBuf) -> Result<(BoundOwner, File), String> {
    let owner_metadata = fs::symlink_metadata(&owner).map_err(|error| {
        format!(
            "cannot inspect generation owner {}: {error}",
            owner.display()
        )
    })?;
    let (owner_identity, identity_handle) = bind_owner_identity(&owner, &owner_metadata)?;
    let lock_path = owner.join(GENERATION_LOCK);
    #[cfg(unix)]
    let lock_metadata = identity_handle.symlink_metadata(GENERATION_LOCK);
    #[cfg(not(unix))]
    let lock_metadata = fs::symlink_metadata(&lock_path);
    let file = match lock_metadata {
        Ok(metadata) => {
            if !metadata.file_type().is_file() || metadata.file_type().is_symlink() {
                return Err(format!(
                    "generation lock is not a regular file: {}",
                    lock_path.display()
                ));
            }
            #[cfg(unix)]
            {
                open_unix_generation_lock(&identity_handle, false)
            }
            #[cfg(not(unix))]
            {
                open_generation_lock(&lock_path, false)
            }
        }
        Err(error) if error.kind() == ErrorKind::NotFound => {
            #[cfg(unix)]
            {
                open_unix_generation_lock(&identity_handle, true)
            }
            #[cfg(not(unix))]
            {
                open_generation_lock(&lock_path, true)
            }
        }
        Err(error) => Err(error),
    }
    .map_err(|error| {
        format!(
            "cannot open stable generation lock {}: {error}",
            lock_path.display()
        )
    })?;
    let opened_observation = observe_open_file(&file)?;
    reject_hardlinks(opened_observation, &lock_path)?;
    #[cfg(unix)]
    let (_path_file, path_observation) =
        open_observed_capability_file(&identity_handle, Path::new(GENERATION_LOCK), &lock_path)?;
    #[cfg(not(unix))]
    let (_path_file, path_observation) = open_observed_regular_file(&lock_path)?;
    reject_hardlinks(path_observation, &lock_path)?;
    if path_observation.identity != opened_observation.identity {
        return Err(format!(
            "generation lock changed while opening: {}",
            lock_path.display()
        ));
    }
    Ok((
        BoundOwner {
            path: owner,
            identity: owner_identity,
            #[cfg(unix)]
            capability: identity_handle,
            #[cfg(windows)]
            _identity_handle: identity_handle,
            lock_path,
            lock_identity: opened_observation.identity,
        },
        file,
    ))
}

#[cfg(not(unix))]
fn regular_file_metadata(path: &Path) -> Result<fs::Metadata, String> {
    let metadata = fs::symlink_metadata(path)
        .map_err(|error| format!("cannot inspect {}: {error}", path.display()))?;
    if metadata.file_type().is_symlink() || !metadata.file_type().is_file() {
        return Err(format!("path is not a regular file: {}", path.display()));
    }
    Ok(metadata)
}

#[cfg(unix)]
fn open_observed_capability_file(
    owner: &CapabilityDir,
    relative: &Path,
    display_path: &Path,
) -> Result<(File, FileObservation), String> {
    let before = owner.symlink_metadata(relative).map_err(|error| {
        format!(
            "cannot inspect capability-relative file {}: {error}",
            display_path.display()
        )
    })?;
    if before.file_type().is_symlink() || !before.file_type().is_file() {
        return Err(format!(
            "path is not a regular capability-relative file: {}",
            display_path.display()
        ));
    }
    let file = owner
        .open(relative)
        .map(cap_std::fs::File::into_std)
        .map_err(|error| {
            format!(
                "cannot open stable capability-relative file {}: {error}",
                display_path.display()
            )
        })?;
    let after = owner.symlink_metadata(relative).map_err(|error| {
        format!(
            "cannot reinspect capability-relative file {}: {error}",
            display_path.display()
        )
    })?;
    let observation = observe_open_file(&file)?;
    let before_identity = FileIdentity {
        device: before.dev(),
        inode: before.ino(),
    };
    let after_identity = FileIdentity {
        device: after.dev(),
        inode: after.ino(),
    };
    if before_identity != observation.identity || after_identity != observation.identity {
        return Err(format!(
            "stable capability-relative file changed while opening: {}",
            display_path.display()
        ));
    }
    Ok((file, observation))
}

#[cfg(windows)]
fn open_windows_observed_file(path: &Path) -> Result<File, String> {
    OpenOptions::new()
        .read(true)
        .share_mode(FILE_SHARE_READ | FILE_SHARE_WRITE)
        .open(path)
        .map_err(|error| format!("cannot open stable file {}: {error}", path.display()))
}

#[cfg(not(unix))]
fn open_observed_regular_file(path: &Path) -> Result<(File, FileObservation), String> {
    #[cfg(unix)]
    let before = regular_file_metadata(path)?;
    #[cfg(not(unix))]
    regular_file_metadata(path)?;
    #[cfg(not(windows))]
    let file = File::open(path)
        .map_err(|error| format!("cannot open stable file {}: {error}", path.display()))?;
    #[cfg(windows)]
    let file = open_windows_observed_file(path)?;
    #[cfg(unix)]
    let after = regular_file_metadata(path)?;
    #[cfg(not(unix))]
    regular_file_metadata(path)?;
    let observation = observe_open_file(&file)?;
    #[cfg(unix)]
    if (FileIdentity {
        device: before.dev(),
        inode: before.ino(),
    }) != observation.identity
        || (FileIdentity {
            device: after.dev(),
            inode: after.ino(),
        }) != observation.identity
    {
        return Err(format!(
            "stable file changed while opening: {}",
            path.display()
        ));
    }
    #[cfg(windows)]
    {
        let current = open_windows_observed_file(path)?;
        let current_observation = observe_open_file(&current)?;
        if current_observation.identity != observation.identity {
            return Err(format!(
                "stable Windows file changed while opening: {}",
                path.display()
            ));
        }
    }
    Ok((file, observation))
}

/// One canonical stable resource lock retained for a complete top-level operation.
#[derive(Debug)]
struct HeldGenerationLock {
    _path: PathBuf,
    _file: File,
}

/// Process-scoped shared ownership of every resource containing a published target.
///
/// Lock paths are canonicalized, globally sorted, deduplicated, and acquired exactly once in that
/// order. The files are never deleted or renamed. Dropping all handles releases the OS-backed lock
/// set, and process exit closes every handle even when a transaction crash aborts the process.
#[derive(Debug)]
struct SharedGenerationLocks {
    // Retaining these handles is the ownership contract; they are intentionally never accessed.
    #[allow(dead_code)]
    locks: Vec<HeldGenerationLock>,
    layout: GenerationLayout,
    entry_generation: String,
}

/// Process-scoped exclusive ownership of every resource containing a published target.
#[derive(Debug)]
struct ExclusiveGenerationLocks {
    // Retaining these handles is the ownership contract; they are intentionally never accessed.
    #[allow(dead_code)]
    locks: Vec<HeldGenerationLock>,
    layout: GenerationLayout,
}

impl SharedGenerationLocks {
    fn acquire(artifact_dir: &Path, rust_repo: &Path) -> Result<Self, String> {
        let (layout, files) = GenerationLayout::bind(artifact_dir, rust_repo)?;
        maybe_mark_lock_attempt("shared");
        let locks = acquire_generation_locks(&layout, files, true)?;
        layout.verify()?;
        let entry_generation = read_complete_generation(&layout)?;
        maybe_pause_after_lock("shared");
        Ok(Self {
            locks,
            layout,
            entry_generation,
        })
    }

    fn finish(&self) -> Result<(), String> {
        self.layout.verify()?;
        let current = read_complete_generation(&self.layout)?;
        if current != self.entry_generation {
            return Err("published generation changed during reader operation".to_owned());
        }
        Ok(())
    }

    #[cfg(test)]
    const fn len(&self) -> usize {
        self.locks.len()
    }
}

impl ExclusiveGenerationLocks {
    fn acquire(artifact_dir: &Path, rust_repo: &Path) -> Result<Self, String> {
        let (layout, files) = GenerationLayout::bind(artifact_dir, rust_repo)?;
        Self::acquire_layout(layout, files)
    }

    #[cfg(test)]
    fn acquire_for_targets(
        journal_parent: &Path,
        targets: &[(PathBuf, Vec<u8>)],
    ) -> Result<Self, String> {
        let owners = target_lock_owners(journal_parent, targets)?;
        let snapshot_owner = owners
            .iter()
            .find(|owner| owner.as_path() != journal_parent)
            .map_or(journal_parent, PathBuf::as_path);
        let (layout, files) =
            GenerationLayout::bind_owners(journal_parent, snapshot_owner, &owners)?;
        Self::acquire_layout(layout, files)
    }

    fn acquire_layout(
        layout: GenerationLayout,
        files: Vec<UnlockedGenerationLock>,
    ) -> Result<Self, String> {
        maybe_mark_lock_attempt("exclusive");
        let locks = acquire_generation_locks(&layout, files, false)?;
        layout.verify()?;
        maybe_pause_after_lock("exclusive");
        Ok(Self { locks, layout })
    }

    #[cfg(test)]
    fn try_acquire(artifact_dir: &Path, rust_repo: &Path) -> Result<Self, String> {
        let (layout, files) = GenerationLayout::bind(artifact_dir, rust_repo)?;
        let locks = try_generation_locks_once(&layout, files, false)?;
        layout.verify()?;
        Ok(Self { locks, layout })
    }

    #[cfg(test)]
    const fn len(&self) -> usize {
        self.locks.len()
    }
}

#[cfg(test)]
fn target_lock_owners(
    journal_parent: &Path,
    targets: &[(PathBuf, Vec<u8>)],
) -> Result<Vec<PathBuf>, String> {
    let mut owners = vec![journal_parent.to_path_buf()];
    for (target, _) in targets {
        owners.push(
            target
                .parent()
                .ok_or_else(|| format!("target {} lacks an owner directory", target.display()))?
                .to_path_buf(),
        );
    }
    Ok(owners)
}

fn acquire_generation_locks(
    _layout: &GenerationLayout,
    files: Vec<UnlockedGenerationLock>,
    shared: bool,
) -> Result<Vec<HeldGenerationLock>, String> {
    let deadline = Instant::now()
        .checked_add(LOCK_ACQUIRE_TIMEOUT)
        .ok_or("generation lock deadline overflow")?;
    let mut held = Vec::with_capacity(files.len());
    for unlocked in files {
        let path = unlocked.path;
        let file = unlocked.file;
        loop {
            let result = if shared {
                file.try_lock_shared()
            } else {
                file.try_lock()
            };
            match result {
                Ok(()) => break,
                Err(TryLockError::WouldBlock) if Instant::now() < deadline => {
                    std::thread::sleep(Duration::from_millis(5));
                }
                Err(TryLockError::WouldBlock) => {
                    return Err(format!(
                        "timed out acquiring generation lock set at {}",
                        path.display()
                    ));
                }
                Err(TryLockError::Error(error)) => {
                    return Err(format!(
                        "cannot acquire generation lock {}: {error}",
                        path.display()
                    ));
                }
            }
        }
        held.push(HeldGenerationLock {
            _path: path,
            _file: file,
        });
    }
    Ok(held)
}

#[cfg(test)]
fn try_generation_locks_once(
    _layout: &GenerationLayout,
    files: Vec<UnlockedGenerationLock>,
    shared: bool,
) -> Result<Vec<HeldGenerationLock>, String> {
    let mut held = Vec::with_capacity(files.len());
    for unlocked in files {
        let path = unlocked.path;
        let file = unlocked.file;
        let result = if shared {
            file.try_lock_shared()
        } else {
            file.try_lock()
        };
        result
            .map_err(|error| format!("cannot try generation lock {}: {error}", path.display()))?;
        held.push(HeldGenerationLock {
            _path: path,
            _file: file,
        });
    }
    Ok(held)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum GenerationStatus {
    InProgress,
    Complete,
}

impl GenerationStatus {
    const fn as_str(self) -> &'static str {
        match self {
            Self::InProgress => "in-progress",
            Self::Complete => "complete",
        }
    }
}

#[cfg_attr(unix, allow(clippy::unnecessary_wraps))]
fn fresh_generation_id(layout: &GenerationLayout) -> Result<String, String> {
    #[cfg(unix)]
    {
        let _ = layout;
        Ok(format!(
            "{}-{}",
            std::process::id(),
            uuid::Uuid::new_v4().simple()
        ))
    }
    #[cfg(not(unix))]
    {
        let unique = tempfile::Builder::new()
            .prefix(".emel-generation-id-")
            .tempfile_in(&layout.artifact_owner)
            .map_err(|error| format!("cannot create unique generation identity: {error}"))?;
        let name = unique
            .path()
            .file_name()
            .and_then(|name| name.to_str())
            .ok_or("unique generation identity is not UTF-8")?
            .bytes()
            .filter(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_'))
            .map(char::from)
            .collect::<String>();
        drop(unique);
        Ok(format!("{}-{name}", std::process::id()))
    }
}

fn generation_token_path(owner: &BoundOwner) -> PathBuf {
    owner.path.join(GENERATION_TOKEN)
}

fn read_generation_token(owner: &BoundOwner) -> Result<(GenerationStatus, String), String> {
    let path = generation_token_path(owner);
    #[cfg(unix)]
    let (mut file, observation) =
        open_observed_capability_file(&owner.capability, Path::new(GENERATION_TOKEN), &path)?;
    #[cfg(not(unix))]
    let (mut file, observation) = open_observed_regular_file(&path)?;
    reject_hardlinks(observation, &path)?;
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes).map_err(|error| {
        format!(
            "cannot read generation identity {}: {error}",
            path.display()
        )
    })?;
    let text = std::str::from_utf8(&bytes)
        .map_err(|error| format!("generation identity is not UTF-8: {error}"))?;
    let line = text
        .strip_suffix('\n')
        .ok_or("generation identity lacks terminal newline")?;
    if line.contains('\n') {
        return Err("generation identity contains extra records".to_owned());
    }
    let mut fields = line.split('\t');
    if fields.next() != Some(GENERATION_TOKEN_SCHEMA) {
        return Err("generation identity schema drift".to_owned());
    }
    let status = match fields.next() {
        Some("in-progress") => GenerationStatus::InProgress,
        Some("complete") => GenerationStatus::Complete,
        _ => return Err("generation identity status drift".to_owned()),
    };
    let id = fields
        .next()
        .filter(|id| {
            !id.is_empty()
                && id
                    .bytes()
                    .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_'))
        })
        .ok_or("generation identity value is invalid")?;
    if fields.next().is_some() {
        return Err("generation identity has extra fields".to_owned());
    }
    Ok((status, id.to_owned()))
}

fn read_complete_generation(layout: &GenerationLayout) -> Result<String, String> {
    layout.verify()?;
    let mut expected = None;
    for owner in &layout.owners {
        let (status, id) = read_generation_token(owner)?;
        if status != GenerationStatus::Complete {
            return Err(format!(
                "published generation is not complete at {}",
                owner.path.display()
            ));
        }
        if expected.as_ref().is_some_and(|expected| expected != &id) {
            return Err("published generation identities disagree across owners".to_owned());
        }
        expected = Some(id);
    }
    expected.ok_or_else(|| "published generation has no bound owners".to_owned())
}

fn write_generation_status(
    layout: &GenerationLayout,
    status: GenerationStatus,
    id: &str,
) -> Result<(), String> {
    layout.verify()?;
    let bytes = format!("{GENERATION_TOKEN_SCHEMA}\t{}\t{id}\n", status.as_str());
    for (index, owner) in layout.owners.iter().enumerate() {
        let path = generation_token_path(owner);
        #[cfg(unix)]
        let mut temporary = CapabilityTempFile::new(&owner.capability)
            .map_err(|error| format!("cannot stage generation identity: {error}"))?;
        #[cfg(not(unix))]
        let mut temporary = tempfile::Builder::new()
            .prefix(".emel-generation-write-")
            .tempfile_in(&owner.path)
            .map_err(|error| format!("cannot stage generation identity: {error}"))?;
        temporary
            .write_all(bytes.as_bytes())
            .map_err(|error| format!("cannot write staged generation identity: {error}"))?;
        temporary
            .as_file_mut()
            .sync_all()
            .map_err(|error| format!("cannot sync staged generation identity: {error}"))?;
        #[cfg(unix)]
        let token_metadata = owner.capability.symlink_metadata(GENERATION_TOKEN);
        #[cfg(not(unix))]
        let token_metadata = fs::symlink_metadata(&path);
        match token_metadata {
            Ok(metadata) => {
                if metadata.file_type().is_symlink() || !metadata.file_type().is_file() {
                    return Err(format!(
                        "generation identity is not a regular file: {}",
                        path.display()
                    ));
                }
                #[cfg(unix)]
                let (file, observation) = open_observed_capability_file(
                    &owner.capability,
                    Path::new(GENERATION_TOKEN),
                    &path,
                )?;
                #[cfg(not(unix))]
                let (file, observation) = open_observed_regular_file(&path)?;
                reject_hardlinks(observation, &path)?;
                drop(file);
            }
            Err(error) if error.kind() == ErrorKind::NotFound => {}
            Err(error) => {
                return Err(format!(
                    "cannot inspect generation identity {}: {error}",
                    path.display()
                ));
            }
        }
        #[cfg(unix)]
        temporary.replace(GENERATION_TOKEN).map_err(|error| {
            format!(
                "cannot publish generation identity {}: {}",
                path.display(),
                error
            )
        })?;
        #[cfg(not(unix))]
        temporary.persist(&path).map_err(|error| {
            format!(
                "cannot publish generation identity {}: {}",
                path.display(),
                error.error
            )
        })?;
        layout.sync_directory(&owner.path)?;
        maybe_crash_transaction(&format!("generation-{}-{index}", status.as_str()));
    }
    layout.verify()
}

#[cfg(test)]
fn maybe_mark_lock_attempt(mode: &str) {
    if let Some(path) = std::env::var_os("EMEL_MODEL_INVENTORY_LOCK_ATTEMPT") {
        fs::write(path, mode).expect("cannot mark generation lock attempt");
    }
}

#[cfg(not(test))]
const fn maybe_mark_lock_attempt(mode: &str) {
    let _ = mode;
}

#[cfg(test)]
fn maybe_pause_after_lock(mode: &str) {
    let Some(entered) = std::env::var_os("EMEL_MODEL_INVENTORY_LOCK_ENTERED") else {
        return;
    };
    let release = PathBuf::from(
        std::env::var_os("EMEL_MODEL_INVENTORY_LOCK_RELEASE")
            .expect("lock release path is required with lock entered path"),
    );
    fs::write(entered, mode).expect("cannot mark acquired generation lock");
    let deadline = Instant::now() + Duration::from_secs(30);
    while !release.exists() {
        assert!(
            Instant::now() < deadline,
            "timed out waiting to release generation lock"
        );
        std::thread::sleep(Duration::from_millis(10));
    }
}

#[cfg(not(test))]
const fn maybe_pause_after_lock(mode: &str) {
    let _ = mode;
}

/// Deterministic result of parsing the committed Rust model crate without running Clang.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RustPreflightReport {
    /// Canonical local path of the repository whose Git objects were read.
    pub rust_repo: PathBuf,
    /// Revision expression supplied by the caller.
    pub requested_commit: String,
    /// Exact commit object resolved from the requested revision.
    pub resolved_commit: String,
    /// Exact Git subtree object for `crates/emel-model`.
    pub rust_tree: String,
    /// Number of ordered blob identities in that subtree.
    pub blob_count: usize,
    /// SHA-256 of length-delimited ordered path and blob-OID pairs.
    pub blob_manifest_sha256: String,
    /// Number of non-test declaration and member rows produced by `syn`.
    pub declaration_count: usize,
    /// Number of Rust test declaration rows produced by `syn`.
    pub test_count: usize,
    /// Number of assertion invocation rows produced by the Rust AST visitor.
    pub assertion_count: usize,
    /// Number of transition rows extracted from `sml!` invocations.
    pub sml_transition_count: usize,
}

impl RustPreflightReport {
    /// Renders the stable, line-oriented proof record emitted by the CLI.
    #[must_use]
    pub fn render(&self) -> String {
        format!(
            "rust_repo={}\nrequested_commit={}\nresolved_commit={}\nrust_tree={}\nblob_count={}\nblob_manifest_sha256={}\ndeclaration_count={}\ntest_count={}\nassertion_count={}\nsml_transition_count={}\n",
            self.rust_repo.display(),
            self.requested_commit,
            self.resolved_commit,
            self.rust_tree,
            self.blob_count,
            self.blob_manifest_sha256,
            self.declaration_count,
            self.test_count,
            self.assertion_count,
            self.sml_transition_count,
        )
    }
}

pub fn cli(arguments: impl IntoIterator<Item = OsString>) -> Result<(), String> {
    let mut arguments = arguments.into_iter();
    let command = arguments
        .next()
        .ok_or("expected extract, validate, check-deterministic, or preflight-rust")?;
    let command = command.to_str().ok_or("command is not UTF-8")?;
    match command {
        "extract" => {
            let options = parse_extract_options(arguments)?;
            extract(&options)
        }
        "check-deterministic" => {
            let options = parse_extract_options(arguments)?;
            check_deterministic(&options)
        }
        "validate" => {
            let values = parse_flags(arguments)?;
            reject_unknown(
                &values,
                &[
                    "artifact-dir",
                    "source-commit",
                    "source-model-tree",
                    "source-tests-tree",
                    "rust-repo",
                    "rust-commit",
                ],
            )?;
            let artifact_dir = required_path(&values, "artifact-dir")?;
            let source_commit = required(&values, "source-commit")?;
            let model_tree = required(&values, "source-model-tree")?;
            let tests_tree = required(&values, "source-tests-tree")?;
            let rust_repo = required_path(&values, "rust-repo")?
                .canonicalize()
                .map_err(|error| format!("cannot canonicalize Rust repository: {error}"))?;
            let rust_commit = required(&values, "rust-commit")?;
            let generation_lock = SharedGenerationLocks::acquire(&artifact_dir, &rust_repo)?;
            let deadline = deadline_after(DEFAULT_GIT_TIMEOUT, "Git validation")?;
            let rust_snapshot =
                load_rust_git_snapshot_with(&rust_repo, &rust_commit, deadline, || {})?;
            require_rust_manifest(&rust_snapshot.blobs)?;
            verify_blob_identities(
                &rust_repo,
                &rust_snapshot.blobs,
                &rust_snapshot.identities,
                deadline,
            )?;
            validate_artifacts_locked(
                &generation_lock,
                &artifact_dir,
                &rust_repo,
                &source_commit,
                &model_tree,
                &tests_tree,
                &rust_snapshot.tree,
            )?;
            require_requested_rust_revision(
                &rust_repo,
                &rust_commit,
                &rust_snapshot.resolved_commit,
                deadline,
            )
        }
        "preflight-rust" => {
            let values = parse_flags(arguments)?;
            reject_unknown(&values, &["rust-repo", "rust-commit", "deadline-seconds"])?;
            let rust_repo = required_path(&values, "rust-repo")?;
            let rust_commit = required(&values, "rust-commit")?;
            let deadline_seconds = required(&values, "deadline-seconds")?
                .parse::<u64>()
                .map_err(|error| format!("deadline-seconds must be a positive integer: {error}"))?;
            if deadline_seconds == 0 {
                return Err("deadline-seconds must be a positive integer".to_owned());
            }
            print!(
                "{}",
                rust_preflight(
                    &rust_repo,
                    &rust_commit,
                    Duration::from_secs(deadline_seconds),
                )?
                .render()
            );
            Ok(())
        }
        _ => Err(format!("unknown command: {command}")),
    }
}

/// Parses and counts the exact committed `crates/emel-model` Git tree without reading worktree data.
///
/// # Errors
///
/// Returns a diagnostic on Git identity drift, malformed or unsupported Git entries, Rust parser or
/// scanner rejection, duplicate extracted identities, or count overflow.
pub fn rust_preflight(
    rust_repo: &Path,
    rust_commit: &str,
    timeout: Duration,
) -> Result<RustPreflightReport, String> {
    let deadline = deadline_after(timeout, "Rust preflight")?;
    let canonical_repo = rust_repo
        .canonicalize()
        .map_err(|error| format!("cannot canonicalize Rust repository: {error}"))?;
    let snapshot = load_rust_git_snapshot_with(&canonical_repo, rust_commit, deadline, || {})?;
    require_rust_manifest(&snapshot.blobs)?;
    verify_blob_identities(
        &canonical_repo,
        &snapshot.blobs,
        &snapshot.identities,
        deadline,
    )?;

    let extracted = scan::rust(&snapshot.blobs)?;
    let counts = rust_counts(&extracted)?;

    require_requested_rust_revision(
        &canonical_repo,
        rust_commit,
        &snapshot.resolved_commit,
        deadline,
    )?;

    Ok(RustPreflightReport {
        rust_repo: canonical_repo,
        requested_commit: rust_commit.to_owned(),
        resolved_commit: snapshot.resolved_commit,
        rust_tree: snapshot.tree,
        blob_count: checked_count(snapshot.identities.iter(), "Rust blob count")?,
        blob_manifest_sha256: blob_manifest_sha256(&snapshot.identities)?,
        declaration_count: counts.declarations,
        test_count: counts.tests,
        assertion_count: counts.assertions,
        sml_transition_count: counts.transitions,
    })
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct RustGitSnapshot {
    resolved_commit: String,
    tree: String,
    identities: Vec<BlobIdentity>,
    blobs: Vec<Blob>,
}

fn load_rust_git_snapshot_with(
    repo: &Path,
    requested_commit: &str,
    deadline: Instant,
    after_resolve: impl FnOnce(),
) -> Result<RustGitSnapshot, String> {
    let resolved_commit = git::resolved_commit(repo, requested_commit, deadline)?;
    after_resolve();
    let tree = git::tree(repo, &resolved_commit, RUST_PATH, deadline)?;
    git::validate_oid(&tree)?;
    let identities = git::blob_identities_from_tree(repo, &tree, RUST_PATH, deadline)?;
    let blobs = git::blobs_from_identities(repo, &identities, deadline)?;
    Ok(RustGitSnapshot {
        resolved_commit,
        tree,
        identities,
        blobs,
    })
}

fn require_requested_rust_revision(
    repo: &Path,
    requested_commit: &str,
    resolved_commit: &str,
    deadline: Instant,
) -> Result<(), String> {
    let resolved_after = git::resolved_commit(repo, requested_commit, deadline)?;
    if resolved_after != resolved_commit {
        return Err("requested Rust commit identity drifted during operation".to_owned());
    }
    Ok(())
}

fn deadline_after(timeout: Duration, label: &str) -> Result<Instant, String> {
    if timeout.is_zero() {
        return Err(format!("{label} timeout must be positive"));
    }
    Instant::now()
        .checked_add(timeout)
        .ok_or_else(|| format!("{label} deadline overflow"))
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct RustCounts {
    declarations: usize,
    tests: usize,
    assertions: usize,
    transitions: usize,
}

fn rust_counts(extracted: &[Extracted]) -> Result<RustCounts, String> {
    let mut counts = RustCounts::default();
    for item in extracted {
        match item.note.as_str() {
            "syn-item"
            | "syn-enum-variant"
            | "syn-field"
            | "syn-foreign-mod"
            | "syn-impl"
            | "syn-associated-macro"
            | "syn-trait-item"
            | "syn-macro-item" => {
                if item.kind == "test" {
                    checked_increment(&mut counts.tests, "Rust test count")?;
                } else {
                    checked_increment(&mut counts.declarations, "Rust declaration count")?;
                }
            }
            "syn-assertion-macro" if item.kind == "assertion" => {
                checked_increment(&mut counts.assertions, "Rust assertion count")?;
            }
            "syn-sml-row" if item.kind == "transition" => {
                checked_increment(&mut counts.transitions, "Rust SML transition count")?;
            }
            "deterministic-scanner"
                if matches!(item.kind.as_str(), "fixture" | "runtime-entrypoint") => {}
            note => {
                return Err(format!(
                    "unclassified Rust extraction proof: {note}/{}",
                    item.kind
                ));
            }
        }
    }
    Ok(counts)
}

fn checked_increment(value: &mut usize, label: &str) -> Result<(), String> {
    *value = value
        .checked_add(1)
        .ok_or_else(|| format!("{label} overflow"))?;
    Ok(())
}

fn checked_count<'a>(
    values: impl IntoIterator<Item = &'a BlobIdentity>,
    label: &str,
) -> Result<usize, String> {
    let mut count = 0usize;
    for _ in values {
        checked_increment(&mut count, label)?;
    }
    Ok(count)
}

fn verify_blob_identities(
    repo: &Path,
    blobs: &[Blob],
    identities: &[BlobIdentity],
    deadline: Instant,
) -> Result<(), String> {
    if blobs.len() != identities.len() {
        return Err("Rust blob count differs from Git identity manifest".to_owned());
    }
    for (blob, identity) in blobs.iter().zip(identities) {
        if blob.path != identity.path {
            return Err(format!(
                "Rust blob path differs from Git identity manifest: {} != {}",
                blob.path, identity.path
            ));
        }
        if git::blob_bytes(repo, &identity.oid, deadline)? != blob.bytes {
            return Err(format!(
                "Rust blob bytes differ from listed Git object ID: {}",
                blob.path
            ));
        }
    }
    Ok(())
}

fn blob_manifest_sha256(identities: &[BlobIdentity]) -> Result<String, String> {
    let mut hash = Sha256::new();
    for identity in identities {
        for field in [&identity.path, &identity.oid] {
            let length = u64::try_from(field.len())
                .map_err(|_| "Rust blob identity field length overflow".to_owned())?;
            hash.update(length.to_be_bytes());
            hash.update(field.as_bytes());
        }
    }
    Ok(format!("{hash:x}", hash = hash.finalize()))
}

fn parse_extract_options(
    arguments: impl IntoIterator<Item = OsString>,
) -> Result<ExtractOptions, String> {
    let values = parse_flags(arguments)?;
    reject_unknown(
        &values,
        &[
            "source-repo",
            "source-commit",
            "rust-repo",
            "rust-commit",
            "compile-commands",
            "artifact-dir",
        ],
    )?;
    Ok(ExtractOptions {
        source_repo: required_path(&values, "source-repo")?,
        source_commit: required(&values, "source-commit")?,
        rust_repo: required_path(&values, "rust-repo")?,
        rust_commit: required(&values, "rust-commit")?,
        compile_commands: required_path(&values, "compile-commands")?,
        artifact_dir: required_path(&values, "artifact-dir")?,
    })
}

fn parse_flags(
    arguments: impl IntoIterator<Item = OsString>,
) -> Result<BTreeMap<String, OsString>, String> {
    let mut arguments = arguments.into_iter();
    let mut values = BTreeMap::new();
    while let Some(flag) = arguments.next() {
        let flag = flag.to_str().ok_or("flag is not UTF-8")?;
        let name = flag
            .strip_prefix("--")
            .ok_or_else(|| format!("expected flag, got {flag}"))?;
        let value = arguments
            .next()
            .ok_or_else(|| format!("{flag} requires a value"))?;
        if values.insert(name.to_owned(), value).is_some() {
            return Err(format!("duplicate flag: {flag}"));
        }
    }
    Ok(values)
}

fn required(values: &BTreeMap<String, OsString>, name: &str) -> Result<String, String> {
    values
        .get(name)
        .ok_or_else(|| format!("--{name} is required"))?
        .to_str()
        .map(ToOwned::to_owned)
        .ok_or_else(|| format!("--{name} is not UTF-8"))
}

fn required_path(values: &BTreeMap<String, OsString>, name: &str) -> Result<PathBuf, String> {
    values
        .get(name)
        .map(PathBuf::from)
        .ok_or_else(|| format!("--{name} is required"))
}

fn reject_unknown(values: &BTreeMap<String, OsString>, expected: &[&str]) -> Result<(), String> {
    if let Some(name) = values
        .keys()
        .find(|name| !expected.contains(&name.as_str()))
    {
        return Err(format!("unknown flag: --{name}"));
    }
    Ok(())
}

/// Extracts and transactionally publishes all primary artifacts.
///
/// # Errors
///
/// Returns a diagnostic without publishing partial output when any immutable input or extractor fails.
pub fn extract(options: &ExtractOptions) -> Result<(), String> {
    let generation_lock =
        ExclusiveGenerationLocks::acquire(&options.artifact_dir, &options.rust_repo)?;
    let artifacts = build(options)?;
    publish_locked(&generation_lock, options, &artifacts)
}

fn check_deterministic(options: &ExtractOptions) -> Result<(), String> {
    check_deterministic_with(options, build)
}

fn check_deterministic_with(
    options: &ExtractOptions,
    build_artifacts: impl Fn(&ExtractOptions) -> Result<Artifacts, String>,
) -> Result<(), String> {
    let generation_lock =
        SharedGenerationLocks::acquire(&options.artifact_dir, &options.rust_repo)?;
    check_deterministic_locked(&generation_lock, options, build_artifacts)
}

fn check_deterministic_locked(
    generation_lock: &SharedGenerationLocks,
    options: &ExtractOptions,
    build_artifacts: impl Fn(&ExtractOptions) -> Result<Artifacts, String>,
) -> Result<(), String> {
    let layout = &generation_lock.layout;
    layout.verify()?;
    reject_generation_transaction(layout)?;
    let first = build_artifacts(options)?;
    let second = build_artifacts(options)?;
    if let Some(difference) = artifact_difference(&first, &second) {
        return Err(format!(
            "repeated extraction was not byte-identical: {difference}"
        ));
    }
    reject_generation_transaction(layout)?;
    for (name, expected) in &first.files {
        let actual = layout.read(&layout.artifact_path(name))?;
        if actual != *expected {
            return Err(format!("published artifact drift: {name}"));
        }
    }
    let snapshot = layout.read(&layout.snapshot_path())?;
    if snapshot != first.coverage {
        return Err("AST coverage snapshot drift".to_owned());
    }
    let external = layout.read(&layout.external_snapshot_path())?;
    if external != first.external_includes {
        return Err("external include snapshot drift".to_owned());
    }
    reject_generation_transaction(layout)?;
    generation_lock.finish()
}

fn artifact_difference(first: &Artifacts, second: &Artifacts) -> Option<String> {
    let names = first
        .files
        .keys()
        .chain(second.files.keys())
        .collect::<BTreeSet<_>>();
    for name in names {
        let first_bytes = first.files.get(name).map_or(&[][..], Vec::as_slice);
        let second_bytes = second.files.get(name).map_or(&[][..], Vec::as_slice);
        if let Some(difference) = byte_difference("artifact", name, first_bytes, second_bytes) {
            return Some(difference);
        }
    }
    byte_difference(
        "snapshot",
        "pinned-ast-coverage.tsv",
        &first.coverage,
        &second.coverage,
    )
    .or_else(|| {
        byte_difference(
            "snapshot",
            "pinned-external-includes.tsv",
            &first.external_includes,
            &second.external_includes,
        )
    })
}

fn byte_difference(component: &str, name: &str, first: &[u8], second: &[u8]) -> Option<String> {
    if first == second {
        return None;
    }
    let offset = first
        .iter()
        .zip(second)
        .position(|(left, right)| left != right)
        .unwrap_or_else(|| first.len().min(second.len()));
    Some(format!(
        "component={component} file={name} first_len={} second_len={} first_sha256={} \
         second_sha256={} first_diff_offset={offset} first_context={} second_context={}",
        first.len(),
        second.len(),
        sha256(first),
        sha256(second),
        escaped_context(first, offset),
        escaped_context(second, offset),
    ))
}

fn escaped_context(bytes: &[u8], offset: usize) -> String {
    const RADIUS: usize = 16;
    let start = offset.saturating_sub(RADIUS);
    let end = bytes.len().min(offset.saturating_add(RADIUS));
    let mut escaped = String::new();
    escaped.push('"');
    for byte in &bytes[start..end] {
        for character in std::ascii::escape_default(*byte) {
            escaped.push(char::from(character));
        }
    }
    escaped.push('"');
    escaped
}

#[allow(clippy::too_many_lines)]
fn build(options: &ExtractOptions) -> Result<Artifacts, String> {
    let git_deadline = deadline_after(DEFAULT_GIT_TIMEOUT, "Git extraction")?;
    let source_repo = options
        .source_repo
        .canonicalize()
        .map_err(|error| format!("cannot canonicalize source repository: {error}"))?;
    let rust_repo = options
        .rust_repo
        .canonicalize()
        .map_err(|error| format!("cannot canonicalize Rust repository: {error}"))?;
    let resolved_source = git::resolved_commit(&source_repo, &options.source_commit, git_deadline)?;
    if resolved_source != EXPECTED_SOURCE_COMMIT {
        return Err(format!(
            "pinned source commit identity drift: expected {EXPECTED_SOURCE_COMMIT}, got {resolved_source}"
        ));
    }
    let source_model_tree = git::tree(
        &source_repo,
        &resolved_source,
        SOURCE_MODEL_PATH,
        git_deadline,
    )?;
    let source_tests_tree = git::tree(
        &source_repo,
        &resolved_source,
        SOURCE_TEST_PATH,
        git_deadline,
    )?;
    if source_model_tree != EXPECTED_MODEL_TREE || source_tests_tree != EXPECTED_TEST_TREE {
        return Err("pinned source tree identity drift".to_owned());
    }
    let rust_snapshot =
        load_rust_git_snapshot_with(&rust_repo, &options.rust_commit, git_deadline, || {})?;
    let RustGitSnapshot {
        resolved_commit: resolved_rust,
        tree: rust_tree,
        identities: _,
        blobs: rust_blobs,
    } = rust_snapshot;
    let source_blobs = git::blobs(
        &source_repo,
        &resolved_source,
        &[SOURCE_MODEL_PATH, SOURCE_TEST_PATH],
        git_deadline,
    )?;
    require_source_manifest(&source_blobs)?;
    require_rust_manifest(&rust_blobs)?;
    let cpp = clang::extract(
        &source_repo,
        &resolved_source,
        &options.compile_commands,
        &source_blobs,
    )?;
    let rust = scan::rust(&rust_blobs)?;
    let resolved_source_after =
        git::resolved_commit(&source_repo, &options.source_commit, git_deadline)?;
    if resolved_source_after != resolved_source {
        return Err("requested source commit identity drifted during extraction".to_owned());
    }
    let model_tree_after = git::tree(
        &source_repo,
        &resolved_source,
        SOURCE_MODEL_PATH,
        git_deadline,
    )?;
    let tests_tree_after = git::tree(
        &source_repo,
        &resolved_source,
        SOURCE_TEST_PATH,
        git_deadline,
    )?;
    if model_tree_after != source_model_tree || tests_tree_after != source_tests_tree {
        return Err("pinned source trees drifted during extraction".to_owned());
    }
    require_requested_rust_revision(
        &rust_repo,
        &options.rust_commit,
        &resolved_rust,
        git_deadline,
    )?;
    let source_tree =
        format!("commit={resolved_source};model={source_model_tree};tests={source_tests_tree}");
    let mut source_rows = rows("cpp", &source_blobs, cpp.items, &source_tree, &rust_tree)?;
    let mut rust_rows = rows("rust", &rust_blobs, rust, &source_tree, &rust_tree)?;
    reconcile(&mut source_rows, &mut rust_rows);
    let mut reconciled = source_rows.clone();
    reconciled.extend(rust_rows.clone());
    reconciled.sort();

    let gaps = gaps(&reconciled);
    let coverage = proof_coverage(&reconciled);
    let terminal = format!("{INVENTORY_HEADER}\n").into_bytes();
    let files = BTreeMap::from([
        (
            "primary-source-inventory.tsv".into(),
            schema::inventory_tsv(&source_rows)?,
        ),
        (
            "primary-rust-inventory.tsv".into(),
            schema::inventory_tsv(&rust_rows)?,
        ),
        (
            "reconciled-inventory.tsv".into(),
            schema::inventory_tsv(&reconciled)?,
        ),
        ("terminal-source-inventory.tsv".into(), terminal.clone()),
        ("terminal-rust-inventory.tsv".into(), terminal),
        ("proof-coverage-matrix.tsv".into(), coverage),
        ("open-gaps.tsv".into(), gaps),
    ]);
    let coverage = clang::coverage_tsv(&cpp.coverage);
    validate_built(&files, &source_blobs, &rust_blobs)?;
    Ok(Artifacts {
        files,
        coverage,
        external_includes: cpp.external_includes,
    })
}

fn require_source_manifest(blobs: &[Blob]) -> Result<(), String> {
    let production = blobs
        .iter()
        .filter(|blob| blob.path.starts_with("src/emel/model/"))
        .count();
    let tests = blobs
        .iter()
        .filter(|blob| blob.path.starts_with("tests/model/"))
        .count();
    if production != 49 || tests != 7 {
        return Err(format!(
            "source manifest drift: {production} production and {tests} test files; expected 49 and 7"
        ));
    }
    Ok(())
}

fn require_rust_manifest(blobs: &[Blob]) -> Result<(), String> {
    if blobs.iter().any(|blob| blob.path.contains("placeholder")) {
        return Err("placeholder Rust path is unsupported".to_owned());
    }
    if !blobs
        .iter()
        .any(|blob| blob.path == "crates/emel-model/Cargo.toml")
    {
        return Err("Rust package manifest missing".to_owned());
    }
    Ok(())
}

fn rows(
    side: &str,
    blobs: &[Blob],
    extracted: Vec<Extracted>,
    source_tree: &str,
    rust_tree: &str,
) -> Result<Vec<InventoryRow>, String> {
    let mut by_id = BTreeMap::new();
    for blob in blobs {
        let item = Extracted {
            path: blob.path.clone(),
            kind: "file".into(),
            name: blob.path.clone(),
            payload: sha256(&blob.bytes),
            note: "git-blob".into(),
        };
        insert_row(&mut by_id, scan::row(side, &item, source_tree, rust_tree))?;
    }
    for item in extracted {
        insert_row(&mut by_id, scan::row(side, &item, source_tree, rust_tree))?;
    }
    Ok(by_id.into_values().collect())
}

fn insert_row(rows: &mut BTreeMap<String, InventoryRow>, row: InventoryRow) -> Result<(), String> {
    if let Some(existing) = rows.get(&row.item_id) {
        if existing.semantic_hash != row.semantic_hash {
            return Err(format!(
                "duplicate ID has conflicting semantics: {}",
                row.item_id
            ));
        }
    } else {
        rows.insert(row.item_id.clone(), row);
    }
    Ok(())
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct StructuralKey {
    component: String,
    kind: String,
    leaf: String,
}

fn reconcile(source: &mut [InventoryRow], rust: &mut [InventoryRow]) {
    let mut source_by_key = BTreeMap::<StructuralKey, Vec<usize>>::new();
    let mut rust_by_key = BTreeMap::<StructuralKey, Vec<usize>>::new();
    for (index, row) in source.iter().enumerate() {
        if let Some(key) = structural_key(row) {
            source_by_key.entry(key).or_default().push(index);
        }
    }
    for (index, row) in rust.iter().enumerate() {
        if let Some(key) = structural_key(row) {
            rust_by_key.entry(key).or_default().push(index);
        }
    }
    for (key, source_indexes) in source_by_key {
        let Some(rust_indexes) = rust_by_key.get(&key) else {
            continue;
        };
        if source_indexes.len() != 1 || rust_indexes.len() != 1 {
            continue;
        }
        let source_index = source_indexes[0];
        let rust_index = rust_indexes[0];
        source[source_index]
            .counterpart_id
            .clone_from(&rust[rust_index].item_id);
        rust[rust_index]
            .counterpart_id
            .clone_from(&source[source_index].item_id);
        source[source_index].disposition = "partial".into();
        rust[rust_index].disposition = "partial".into();
    }
}

fn structural_key(row: &InventoryRow) -> Option<StructuralKey> {
    if row.kind == "file" {
        return None;
    }
    Some(StructuralKey {
        component: normalized_component(&row.owner_path, &row.side)?,
        kind: row.kind.clone(),
        leaf: normalized_leaf(&row.qualified_name, &row.kind)?,
    })
}

fn normalized_component(path: &str, side: &str) -> Option<String> {
    let relative = match side {
        "cpp" => path
            .strip_prefix("src/emel/model/")
            .or_else(|| path.strip_prefix("tests/model/"))?,
        "rust" => path
            .strip_prefix("crates/emel-model/src/")
            .or_else(|| path.strip_prefix("crates/emel-model/tests/"))
            .or_else(|| path.strip_prefix("crates/emel-model/examples/"))?,
        _ => return None,
    };
    let mut components = relative.split('/').collect::<Vec<_>>();
    let file = components.pop()?;
    let stem = file.rsplit_once('.').map_or(file, |(stem, _)| stem);
    if !COMPONENT_ROLE_FILES.contains(&stem) {
        components.push(stem);
    }
    (!components.is_empty()).then(|| components.join("/"))
}

fn normalized_leaf(name: &str, kind: &str) -> Option<String> {
    let without_signature = name.split('(').next().unwrap_or(name);
    let leaf = without_signature
        .rsplit("::")
        .next()
        .unwrap_or(without_signature);
    let leaf = leaf.split('@').next().unwrap_or(leaf);
    let prefixes: &[&str] = match kind {
        "event" | "outcome" => &["event_"],
        "guard" => &["guard_"],
        "action" => &["effect_", "action_"],
        "transition" => &["transition_"],
        _ => &[],
    };
    let leaf = prefixes
        .iter()
        .find_map(|prefix| leaf.strip_prefix(prefix))
        .unwrap_or(leaf);
    let normalized = leaf
        .bytes()
        .filter(u8::is_ascii_alphanumeric)
        .map(|byte| char::from(byte.to_ascii_lowercase()))
        .collect::<String>();
    (!normalized.is_empty()).then_some(normalized)
}

fn gaps(rows: &[InventoryRow]) -> Vec<u8> {
    let mut output = String::from(GAP_HEADER);
    output.push('\n');
    for row in rows {
        let kind = if row.counterpart_id.is_empty() {
            "unmatched"
        } else {
            "unproven"
        };
        writeln!(
            output,
            "{}\t{}\tunassigned\t{}\tbind implementation and proof\topen",
            row.item_id, kind, row.disposition
        )
        .expect("writing to String cannot fail");
    }
    output.into_bytes()
}

fn proof_coverage(rows: &[InventoryRow]) -> Vec<u8> {
    let mut output = String::from(COVERAGE_HEADER);
    output.push('\n');
    for row in rows {
        writeln!(
            output,
            "{}\t\t\t\t\t\t\t\t\t{}\topen",
            row.item_id, row.rust_tree
        )
        .expect("writing to String cannot fail");
    }
    output.into_bytes()
}

fn validate_built(
    files: &BTreeMap<String, Vec<u8>>,
    source_blobs: &[Blob],
    rust_blobs: &[Blob],
) -> Result<(), String> {
    let source = schema::parse_inventory_tsv(files.get("primary-source-inventory.tsv").unwrap())?;
    let rust = schema::parse_inventory_tsv(files.get("primary-rust-inventory.tsv").unwrap())?;
    let reconciled = schema::parse_inventory_tsv(files.get("reconciled-inventory.tsv").unwrap())?;
    let terminal_source =
        schema::parse_inventory_tsv(files.get("terminal-source-inventory.tsv").unwrap())?;
    let terminal_rust =
        schema::parse_inventory_tsv(files.get("terminal-rust-inventory.tsv").unwrap())?;
    let coverage = schema::parse_coverage_tsv(files.get("proof-coverage-matrix.tsv").unwrap())?;
    let gaps = schema::parse_gap_tsv(files.get("open-gaps.tsv").unwrap())?;
    if !terminal_source.is_empty() || !terminal_rust.is_empty() {
        return Err("terminal inventories contain unproven rows".into());
    }
    let source_paths = source
        .iter()
        .filter(|row| row.kind == "file")
        .map(|row| &row.owner_path)
        .collect::<BTreeSet<_>>();
    let rust_paths = rust
        .iter()
        .filter(|row| row.kind == "file")
        .map(|row| &row.owner_path)
        .collect::<BTreeSet<_>>();
    if source_paths.len() != source_blobs.len() || rust_paths.len() != rust_blobs.len() {
        return Err("file-row completeness failure".to_owned());
    }
    let expected_source = source
        .first()
        .ok_or("source inventory is empty")?
        .source_tree
        .clone();
    let expected_rust = rust
        .first()
        .ok_or("Rust inventory is empty")?
        .rust_tree
        .clone();
    validate_relations(
        &source,
        &rust,
        &reconciled,
        &coverage,
        &gaps,
        &expected_source,
        &expected_rust,
    )
}

/// Validates published schemas, tree bindings, and explicit gaps.
///
/// # Errors
///
/// Returns a diagnostic for any missing, malformed, incomplete, or drifted artifact.
pub fn validate_artifacts(
    artifact_dir: &Path,
    rust_repo: &Path,
    source_commit: &str,
    source_model_tree: &str,
    source_tests_tree: &str,
    rust_tree: &str,
) -> Result<(), String> {
    let generation_lock = SharedGenerationLocks::acquire(artifact_dir, rust_repo)?;
    validate_artifacts_locked(
        &generation_lock,
        artifact_dir,
        rust_repo,
        source_commit,
        source_model_tree,
        source_tests_tree,
        rust_tree,
    )
}

#[allow(clippy::too_many_arguments)]
fn validate_artifacts_locked(
    generation_lock: &SharedGenerationLocks,
    _artifact_dir: &Path,
    _rust_repo: &Path,
    source_commit: &str,
    source_model_tree: &str,
    source_tests_tree: &str,
    rust_tree: &str,
) -> Result<(), String> {
    let layout = &generation_lock.layout;
    layout.verify()?;
    reject_generation_transaction(layout)?;
    if source_commit != EXPECTED_SOURCE_COMMIT
        || source_model_tree != EXPECTED_MODEL_TREE
        || source_tests_tree != EXPECTED_TEST_TREE
    {
        return Err("validation tree identity drift".to_owned());
    }
    for target in published_target_paths(layout) {
        if !layout.try_exists(&target)? {
            return Err(format!(
                "published nine-target generation is missing {}",
                target.display()
            ));
        }
    }
    let expected_source =
        format!("commit={source_commit};model={source_model_tree};tests={source_tests_tree}");
    let source = schema::parse_inventory_tsv(
        &layout.read(&layout.artifact_path("primary-source-inventory.tsv"))?,
    )?;
    let rust = schema::parse_inventory_tsv(
        &layout.read(&layout.artifact_path("primary-rust-inventory.tsv"))?,
    )?;
    let reconciled = schema::parse_inventory_tsv(
        &layout.read(&layout.artifact_path("reconciled-inventory.tsv"))?,
    )?;
    let coverage = schema::parse_coverage_tsv(
        &layout.read(&layout.artifact_path("proof-coverage-matrix.tsv"))?,
    )?;
    let gaps = schema::parse_gap_tsv(&layout.read(&layout.artifact_path("open-gaps.tsv"))?)?;
    let terminal_source = schema::parse_inventory_tsv(
        &layout.read(&layout.artifact_path("terminal-source-inventory.tsv"))?,
    )?;
    let terminal_rust = schema::parse_inventory_tsv(
        &layout.read(&layout.artifact_path("terminal-rust-inventory.tsv"))?,
    )?;
    if !terminal_source.is_empty() || !terminal_rust.is_empty() {
        return Err("terminal inventories contain open or unproven rows".into());
    }
    let coverage_snapshot = layout.read(&layout.snapshot_path())?;
    if sha256(&coverage_snapshot) != PINNED_AST_COVERAGE_SHA256 {
        return Err("pinned AST coverage snapshot drift".into());
    }
    let external_snapshot = layout.read(&layout.external_snapshot_path())?;
    if sha256(&external_snapshot) != clang::EXTERNAL_INCLUDES_SHA256 {
        return Err("pinned external include snapshot drift".into());
    }
    reject_generation_transaction(layout)?;
    validate_relations(
        &source,
        &rust,
        &reconciled,
        &coverage,
        &gaps,
        &expected_source,
        rust_tree,
    )?;
    reject_generation_transaction(layout)?;
    generation_lock.finish()
}

#[allow(clippy::too_many_arguments)]
fn validate_relations(
    source: &[InventoryRow],
    rust: &[InventoryRow],
    reconciled: &[InventoryRow],
    coverage: &[schema::CoverageRow],
    gaps: &[schema::GapRow],
    expected_source: &str,
    expected_rust: &str,
) -> Result<(), String> {
    if source.is_empty() || rust.is_empty() {
        return Err("primary inventory is empty".into());
    }
    if source.iter().any(|row| row.side != "cpp") || rust.iter().any(|row| row.side != "rust") {
        return Err("primary inventory side mismatch".into());
    }
    let mut expected = source.to_vec();
    expected.extend_from_slice(rust);
    expected.sort();
    let mut actual = reconciled.to_vec();
    actual.sort();
    if actual != expected {
        return Err("reconciled inventory is not the exact primary-row union".into());
    }
    if reconciled
        .iter()
        .any(|row| row.source_tree != expected_source || row.rust_tree != expected_rust)
    {
        return Err("inventory generation tree binding mismatch".into());
    }
    let by_id = reconciled
        .iter()
        .map(|row| (row.item_id.as_str(), row))
        .collect::<BTreeMap<_, _>>();
    for row in reconciled {
        if row.counterpart_id.is_empty() {
            continue;
        }
        let counterpart = by_id
            .get(row.counterpart_id.as_str())
            .ok_or_else(|| format!("dangling counterpart: {}", row.item_id))?;
        let symmetric = counterpart.counterpart_id == row.item_id;
        let opposite_side = counterpart.side != row.side;
        let compatible = counterpart.disposition == row.disposition;
        let structurally_compatible = row.disposition != "partial"
            || structural_key(row).is_some_and(|key| Some(key) == structural_key(counterpart));
        if !(symmetric && opposite_side && compatible && structurally_compatible) {
            return Err(format!(
                "asymmetric or incompatible counterpart: {}",
                row.item_id
            ));
        }
    }
    let expected_ids = by_id.keys().copied().collect::<BTreeSet<_>>();
    let coverage_ids = coverage
        .iter()
        .map(|row| row.item_id.as_str())
        .collect::<BTreeSet<_>>();
    let gap_ids = gaps
        .iter()
        .map(|row| row.item_id.as_str())
        .collect::<BTreeSet<_>>();
    if coverage_ids != expected_ids || gap_ids != expected_ids {
        return Err("coverage or gap IDs are not the exact reconciled ID set".into());
    }
    for row in coverage {
        let inventory = by_id[row.item_id.as_str()];
        if row.status != inventory.proof_status || row.terminal_tree != expected_rust {
            return Err(format!("coverage binding mismatch: {}", row.item_id));
        }
    }
    for row in gaps {
        let inventory = by_id[row.item_id.as_str()];
        let expected_kind = if inventory.counterpart_id.is_empty() {
            "unmatched"
        } else {
            "unproven"
        };
        if row.gap_kind != expected_kind || row.status != "open" {
            return Err(format!("gap binding mismatch: {}", row.item_id));
        }
    }
    Ok(())
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct TransactionEntry {
    target: PathBuf,
    staged: PathBuf,
    backup: PathBuf,
    existed: bool,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct TransactionManifest {
    revision: usize,
    phase: String,
    committed: usize,
    entries: Vec<TransactionEntry>,
}

fn publish_locked(
    generation_lock: &ExclusiveGenerationLocks,
    _options: &ExtractOptions,
    artifacts: &Artifacts,
) -> Result<(), String> {
    if artifacts.files.len() != PUBLISHED_ARTIFACT_NAMES.len()
        || PUBLISHED_ARTIFACT_NAMES
            .iter()
            .any(|name| !artifacts.files.contains_key(*name))
    {
        return Err("published artifact target set drift".into());
    }
    let mut targets = PUBLISHED_ARTIFACT_NAMES
        .iter()
        .map(|name| {
            (
                generation_lock.layout.artifact_path(name),
                artifacts.files[*name].clone(),
            )
        })
        .collect::<Vec<_>>();
    targets.push((
        generation_lock.layout.snapshot_path(),
        artifacts.coverage.clone(),
    ));
    targets.push((
        generation_lock.layout.external_snapshot_path(),
        artifacts.external_includes.clone(),
    ));
    publish_generation_locked(generation_lock, &targets, None)
}

fn published_target_paths(layout: &GenerationLayout) -> Vec<PathBuf> {
    PUBLISHED_ARTIFACT_NAMES
        .iter()
        .map(|name| layout.artifact_path(name))
        .chain([layout.snapshot_path(), layout.external_snapshot_path()])
        .collect()
}

#[cfg(test)]
fn published_recovery_targets(layout: &GenerationLayout) -> Vec<(PathBuf, Vec<u8>)> {
    published_target_paths(layout)
        .into_iter()
        .map(|path| (path, Vec::new()))
        .collect()
}

fn reject_generation_transaction(layout: &GenerationLayout) -> Result<(), String> {
    let journal = layout.artifact_path(GENERATION_JOURNAL);
    let initializing = layout.artifact_path(INITIALIZING_GENERATION_JOURNAL);
    let journal_exists = layout.try_exists(&journal)?;
    let initializing_exists = layout.try_exists(&initializing)?;
    if journal_exists || initializing_exists {
        return Err(format!(
            "published generation is not quiescent: journal={journal_exists} \
             initializing={initializing_exists}"
        ));
    }
    Ok(())
}

#[cfg(test)]
fn recover_published_generation(artifact_dir: &Path, rust_repo: &Path) -> Result<(), String> {
    let generation_lock = ExclusiveGenerationLocks::acquire(artifact_dir, rust_repo)?;
    recover_published_generation_locked(&generation_lock)
}

#[cfg(test)]
fn recover_published_generation_locked(
    generation_lock: &ExclusiveGenerationLocks,
) -> Result<(), String> {
    let targets = published_recovery_targets(&generation_lock.layout);
    reconcile_generation(generation_lock, &targets)
}

#[allow(clippy::too_many_lines)]
#[cfg(test)]
fn publish_paths(
    journal_parent: &Path,
    targets: &[(PathBuf, Vec<u8>)],
    injected_failure_after: Option<usize>,
) -> Result<(), String> {
    let generation_lock = ExclusiveGenerationLocks::acquire_for_targets(journal_parent, targets)?;
    let bound_targets = bind_target_paths(&generation_lock.layout, targets)?;
    publish_generation_locked(&generation_lock, &bound_targets, injected_failure_after)
}

#[cfg(test)]
fn bind_target_paths(
    layout: &GenerationLayout,
    targets: &[(PathBuf, Vec<u8>)],
) -> Result<Vec<(PathBuf, Vec<u8>)>, String> {
    targets
        .iter()
        .map(|(target, bytes)| {
            let parent = target
                .parent()
                .ok_or_else(|| format!("target {} lacks parent", target.display()))?;
            let canonical_parent = parent.canonicalize().map_err(|error| {
                format!(
                    "cannot canonicalize target owner {}: {error}",
                    parent.display()
                )
            })?;
            let owner = layout
                .owners
                .iter()
                .find(|owner| owner.path == canonical_parent)
                .ok_or_else(|| {
                    format!(
                        "target owner is not in bound generation layout: {}",
                        canonical_parent.display()
                    )
                })?;
            let name = target
                .file_name()
                .ok_or_else(|| format!("target {} lacks file name", target.display()))?;
            Ok((owner.path.join(name), bytes.clone()))
        })
        .collect()
}

fn publish_generation_locked(
    generation_lock: &ExclusiveGenerationLocks,
    targets: &[(PathBuf, Vec<u8>)],
    injected_failure_after: Option<usize>,
) -> Result<(), String> {
    reconcile_generation(generation_lock, targets)?;
    let generation_id = fresh_generation_id(&generation_lock.layout)?;
    write_generation_status(
        &generation_lock.layout,
        GenerationStatus::InProgress,
        &generation_id,
    )?;
    let result = publish_paths_locked(generation_lock, targets, injected_failure_after);
    match result {
        Ok(()) => write_generation_status(
            &generation_lock.layout,
            GenerationStatus::Complete,
            &generation_id,
        ),
        Err(error) => {
            if reject_generation_transaction(&generation_lock.layout).is_ok() {
                write_generation_status(
                    &generation_lock.layout,
                    GenerationStatus::Complete,
                    &generation_id,
                )?;
            }
            Err(error)
        }
    }
}

fn reconcile_generation(
    generation_lock: &ExclusiveGenerationLocks,
    targets: &[(PathBuf, Vec<u8>)],
) -> Result<(), String> {
    let layout = &generation_lock.layout;
    layout.verify()?;
    let journal = layout.artifact_path(GENERATION_JOURNAL);
    let initializing = layout.artifact_path(INITIALIZING_GENERATION_JOURNAL);
    if layout.try_exists(&initializing)? {
        if layout.try_exists(&journal)? {
            return Err("generation journal and initializing journal both exist".into());
        }
        layout
            .remove_dir_all(&initializing)
            .map_err(|error| format!("cannot remove abandoned initializing journal: {error}"))?;
        layout.sync_directory(&layout.artifact_owner)?;
    }
    if layout.try_exists(&journal)? {
        recover_journal(layout, &journal, targets)?;
    }
    reject_generation_transaction(layout)?;
    let complete = layout
        .owners
        .iter()
        .map(read_generation_token_if_present)
        .collect::<Result<Vec<_>, _>>()?;
    let already_complete = complete
        .first()
        .and_then(Option::as_ref)
        .is_some_and(|first| {
            first.0 == GenerationStatus::Complete
                && complete.iter().all(|token| token.as_ref() == Some(first))
        });
    if !already_complete {
        let id = fresh_generation_id(layout)?;
        write_generation_status(layout, GenerationStatus::Complete, &id)?;
    }
    layout.verify()
}

fn read_generation_token_if_present(
    owner: &BoundOwner,
) -> Result<Option<(GenerationStatus, String)>, String> {
    let path = generation_token_path(owner);
    #[cfg(unix)]
    let metadata = owner.capability.symlink_metadata(GENERATION_TOKEN);
    #[cfg(not(unix))]
    let metadata = fs::symlink_metadata(&path);
    match metadata {
        Ok(_) => read_generation_token(owner).map(Some),
        Err(error) if error.kind() == ErrorKind::NotFound => Ok(None),
        Err(error) => Err(format!(
            "cannot inspect generation identity {}: {error}",
            path.display()
        )),
    }
}

#[allow(clippy::too_many_lines)]
fn publish_paths_locked(
    generation_lock: &ExclusiveGenerationLocks,
    targets: &[(PathBuf, Vec<u8>)],
    injected_failure_after: Option<usize>,
) -> Result<(), String> {
    let layout = &generation_lock.layout;
    let journal_parent = &layout.artifact_owner;
    layout.verify()?;
    let journal = journal_parent.join(GENERATION_JOURNAL);
    let initializing = journal_parent.join(INITIALIZING_GENERATION_JOURNAL);
    if layout.try_exists(&initializing)? {
        if layout.try_exists(&journal)? {
            return Err("generation journal and initializing journal both exist".into());
        }
        layout
            .remove_dir_all(&initializing)
            .map_err(|error| format!("cannot remove abandoned initializing journal: {error}"))?;
        layout.sync_directory(journal_parent)?;
    }
    if layout.try_exists(&journal)? {
        recover_journal(layout, &journal, targets)?;
    }

    let entries = targets
        .iter()
        .enumerate()
        .map(|(index, (target, _))| {
            Ok(TransactionEntry {
                target: target.clone(),
                staged: journal.join("stage").join(index.to_string()),
                backup: journal.join("backup").join(index.to_string()),
                existed: layout.try_exists(target)?,
            })
        })
        .collect::<Result<Vec<_>, String>>()?;
    let initial_manifest = TransactionManifest {
        revision: 0,
        phase: "preparing".into(),
        committed: 0,
        entries,
    };

    layout
        .create_dir(&initializing)
        .map_err(|error| format!("cannot create initializing journal: {error}"))?;
    maybe_crash_transaction("init-dir-created");
    if let Err(error) = write_manifest_with_crash_points(
        layout,
        &initializing,
        &initial_manifest,
        Some("init-manifest"),
    ) {
        return cleanup_initializing_failure(layout, journal_parent, &initializing, &error);
    }
    if let Err(error) = layout.rename(&initializing, &journal) {
        return cleanup_initializing_failure(
            layout,
            journal_parent,
            &initializing,
            &format!("cannot publish initialized generation journal: {error}"),
        );
    }
    maybe_crash_transaction("journal-renamed");
    layout.sync_directory(journal_parent)?;
    maybe_crash_transaction("journal-published");

    layout
        .create_dir(&journal.join("stage"))
        .map_err(|error| format!("cannot create generation stage: {error}"))?;
    maybe_crash_transaction("stage-dir-created");
    layout.sync_directory(&journal)?;
    maybe_crash_transaction("stage-dir-synced");
    layout
        .create_dir(&journal.join("backup"))
        .map_err(|error| format!("cannot create generation backup: {error}"))?;
    maybe_crash_transaction("backup-dir-created");
    layout.sync_directory(&journal)?;
    maybe_crash_transaction("backup-dir-synced");

    let prepare = (|| {
        for (index, (entry, (_, bytes))) in initial_manifest.entries.iter().zip(targets).enumerate()
        {
            let target = &entry.target;
            let parent = target
                .parent()
                .ok_or_else(|| format!("target {} lacks parent", target.display()))?;
            if !layout.try_exists(parent)? {
                return Err(format!("target parent {} is missing", parent.display()));
            }
            if layout.try_exists(target)? != entry.existed {
                return Err(format!(
                    "target existence changed during preparation: {}",
                    target.display()
                ));
            }
            write_synced_with_crash_points(
                layout,
                &entry.staged,
                bytes,
                &format!("stage-{index}"),
            )?;
            if entry.existed {
                layout
                    .copy(target, &entry.backup)
                    .map_err(|error| format!("cannot back up {}: {error}", target.display()))?;
                maybe_crash_transaction(&format!("backup-{index}-copied"));
                layout.sync_file(&entry.backup)?;
                maybe_crash_transaction(&format!("backup-{index}-synced"));
            }
            maybe_crash_transaction(&format!("entry-{index}-prepared"));
        }
        let manifest = TransactionManifest {
            revision: 1,
            phase: "prepared".into(),
            committed: 0,
            entries: initial_manifest.entries.clone(),
        };
        write_manifest(layout, &journal, &manifest)?;
        maybe_crash_transaction("prepared");
        Ok::<_, String>(manifest)
    })();

    let mut manifest = match prepare {
        Ok(manifest) => manifest,
        Err(error) => {
            return cleanup_initializing_failure(layout, journal_parent, &journal, &error);
        }
    };
    manifest.phase = "committing".into();
    manifest.revision += 1;
    if let Err(error) = write_manifest(layout, &journal, &manifest) {
        return rollback_and_report(layout, &journal, &manifest, &error);
    }
    for index in 0..manifest.entries.len() {
        if injected_failure_after == Some(index) {
            let error = format!("injected generation failure after {index} commits");
            return rollback_and_report(layout, &journal, &manifest, &error);
        }
        let entry = &manifest.entries[index];
        if layout.try_exists(&entry.target)?
            && let Err(error) = layout.remove_file(&entry.target)
        {
            return rollback_and_report(
                layout,
                &journal,
                &manifest,
                &format!("cannot replace {}: {error}", entry.target.display()),
            );
        }
        if let Err(error) = layout.rename(&entry.staged, &entry.target) {
            return rollback_and_report(
                layout,
                &journal,
                &manifest,
                &format!("cannot commit {}: {error}", entry.target.display()),
            );
        }
        if let Err(error) = sync_parent(layout, &entry.target) {
            return rollback_and_report(layout, &journal, &manifest, &error);
        }
        maybe_crash_transaction(&format!("commit-{index}"));
        manifest.committed = index + 1;
        manifest.revision += 1;
        if let Err(error) = write_manifest(layout, &journal, &manifest) {
            return rollback_and_report(layout, &journal, &manifest, &error);
        }
    }
    manifest.phase = "complete".into();
    manifest.revision += 1;
    if let Err(error) = write_manifest(layout, &journal, &manifest) {
        return rollback_and_report(layout, &journal, &manifest, &error);
    }
    maybe_crash_transaction("complete");
    layout
        .remove_dir_all(&journal)
        .map_err(|error| format!("generation committed but journal cleanup failed: {error}"))?;
    layout.sync_directory(journal_parent)
}

#[cfg(test)]
fn maybe_crash_transaction(point: &str) {
    if std::env::var("EMEL_MODEL_INVENTORY_CRASH_POINT").as_deref() == Ok(point) {
        std::process::abort();
    }
}

#[cfg(not(test))]
const fn maybe_crash_transaction(point: &str) {
    let _ = point;
}

fn recover_journal(
    layout: &GenerationLayout,
    journal: &Path,
    targets: &[(PathBuf, Vec<u8>)],
) -> Result<(), String> {
    let manifest = latest_manifest(layout, journal)?;
    validate_manifest(journal, &manifest, targets)?;
    if manifest.phase == "preparing" {
        let parent = journal
            .parent()
            .ok_or("generation journal lacks parent directory")?;
        layout
            .remove_dir_all(journal)
            .map_err(|error| format!("cannot remove preparing generation journal: {error}"))?;
        return layout.sync_directory(parent);
    }
    if manifest.phase == "complete" {
        let parent = journal
            .parent()
            .ok_or("generation journal lacks parent directory")?;
        layout
            .remove_dir_all(journal)
            .map_err(|error| format!("cannot remove completed generation journal: {error}"))?;
        return layout.sync_directory(parent);
    }
    rollback(layout, journal, &manifest)
}

fn latest_manifest(
    layout: &GenerationLayout,
    journal: &Path,
) -> Result<TransactionManifest, String> {
    let mut manifests = Vec::new();
    for name in layout.read_dir_names(journal)? {
        let name = name.to_string_lossy();
        let Some(revision) = name
            .strip_prefix("manifest-")
            .and_then(|value| value.strip_suffix(".json"))
            .and_then(|value| value.parse::<usize>().ok())
        else {
            continue;
        };
        let bytes = layout.read(&journal.join(name.as_ref())).map_err(|error| {
            format!("cannot read generation journal revision {revision}: {error}")
        })?;
        let manifest: TransactionManifest = serde_json::from_slice(&bytes).map_err(|error| {
            format!("cannot parse generation journal revision {revision}: {error}")
        })?;
        if manifest.revision != revision {
            return Err(format!(
                "generation journal revision mismatch: file {revision}, manifest {}",
                manifest.revision
            ));
        }
        manifests.push(manifest);
    }
    manifests
        .into_iter()
        .max_by_key(|manifest| manifest.revision)
        .ok_or_else(|| "existing generation journal has no complete manifest revision".to_owned())
}

fn validate_manifest(
    journal: &Path,
    manifest: &TransactionManifest,
    targets: &[(PathBuf, Vec<u8>)],
) -> Result<(), String> {
    if !matches!(
        manifest.phase.as_str(),
        "preparing" | "prepared" | "committing" | "complete"
    ) || manifest.committed > manifest.entries.len()
        || (manifest.phase == "preparing" && manifest.committed != 0)
        || (manifest.phase == "prepared" && manifest.committed != 0)
        || (manifest.phase == "complete" && manifest.committed != manifest.entries.len())
    {
        return Err("generation journal phase or commit count is invalid".to_owned());
    }
    if manifest.entries.len() != targets.len() {
        return Err("generation journal target count drift".to_owned());
    }
    let mut unique = BTreeSet::new();
    for (index, (entry, (target, _))) in manifest.entries.iter().zip(targets).enumerate() {
        if entry.target != *target
            || entry.staged != journal.join("stage").join(index.to_string())
            || entry.backup != journal.join("backup").join(index.to_string())
            || !unique.insert(&entry.target)
        {
            return Err(format!(
                "generation journal target set drift at index {index}"
            ));
        }
    }
    Ok(())
}

fn rollback_and_report(
    layout: &GenerationLayout,
    journal: &Path,
    manifest: &TransactionManifest,
    original: &str,
) -> Result<(), String> {
    match rollback(layout, journal, manifest) {
        Ok(()) => Err(original.into()),
        Err(rollback) => Err(format!("{original}; rollback failed: {rollback}")),
    }
}

fn rollback(
    layout: &GenerationLayout,
    journal: &Path,
    manifest: &TransactionManifest,
) -> Result<(), String> {
    let mut failures = Vec::new();
    for entry in manifest.entries.iter().rev() {
        let result = if entry.existed {
            if layout.try_exists(&entry.backup)? {
                if layout.try_exists(&entry.target)?
                    && let Err(error) = layout.remove_file(&entry.target)
                {
                    failures.push(format!("cannot remove {}: {error}", entry.target.display()));
                    continue;
                }
                layout
                    .copy(&entry.backup, &entry.target)
                    .map_err(|error| format!("cannot restore {}: {error}", entry.target.display()))
                    .and_then(|()| layout.sync_file(&entry.target))
            } else {
                Err(format!("missing backup for {}", entry.target.display()))
            }
        } else if layout.try_exists(&entry.target)? {
            layout
                .remove_file(&entry.target)
                .map_err(|error| format!("cannot remove new {}: {error}", entry.target.display()))
        } else {
            Ok(())
        };
        if let Err(error) = result {
            failures.push(error);
        }
    }
    if !failures.is_empty() {
        return Err(failures.join("; "));
    }
    let parent = journal
        .parent()
        .ok_or("generation journal lacks parent directory")?;
    layout
        .remove_dir_all(journal)
        .map_err(|error| format!("cannot remove rolled-back generation journal: {error}"))?;
    layout.sync_directory(parent)
}

fn write_manifest(
    layout: &GenerationLayout,
    journal: &Path,
    manifest: &TransactionManifest,
) -> Result<(), String> {
    write_manifest_with_crash_points(layout, journal, manifest, None)
}

fn write_manifest_with_crash_points(
    layout: &GenerationLayout,
    journal: &Path,
    manifest: &TransactionManifest,
    crash_prefix: Option<&str>,
) -> Result<(), String> {
    let bytes = serde_json::to_vec(manifest)
        .map_err(|error| format!("cannot serialize generation journal: {error}"))?;
    let temporary = journal.join(format!("manifest-{}.json.tmp", manifest.revision));
    let committed = journal.join(format!("manifest-{}.json", manifest.revision));
    if layout.try_exists(&temporary)? || layout.try_exists(&committed)? {
        return Err(format!(
            "generation journal revision {} already exists",
            manifest.revision
        ));
    }
    let mut file = layout.create_file(&temporary, true)?;
    maybe_crash_with_suffix(crash_prefix, "temp-created");
    file.write_all(&bytes)
        .map_err(|error| format!("cannot write {}: {error}", temporary.display()))?;
    maybe_crash_with_suffix(crash_prefix, "written");
    file.sync_all()
        .map_err(|error| format!("cannot sync {}: {error}", temporary.display()))?;
    maybe_crash_with_suffix(crash_prefix, "synced");
    drop(file);
    layout.rename(&temporary, &committed).map_err(|error| {
        format!(
            "cannot atomically commit {} to {}: {error}",
            temporary.display(),
            committed.display()
        )
    })?;
    maybe_crash_with_suffix(crash_prefix, "renamed");
    layout.sync_directory(journal)?;
    maybe_crash_with_suffix(crash_prefix, "directory-synced");
    Ok(())
}

fn write_synced_with_crash_points(
    layout: &GenerationLayout,
    path: &Path,
    bytes: &[u8],
    crash_prefix: &str,
) -> Result<(), String> {
    let mut file = layout.create_file(path, false)?;
    maybe_crash_with_suffix(
        (!crash_prefix.is_empty()).then_some(crash_prefix),
        "created",
    );
    file.write_all(bytes)
        .map_err(|error| format!("cannot write {}: {error}", path.display()))?;
    maybe_crash_with_suffix(
        (!crash_prefix.is_empty()).then_some(crash_prefix),
        "written",
    );
    file.sync_all()
        .map_err(|error| format!("cannot sync {}: {error}", path.display()))?;
    maybe_crash_with_suffix((!crash_prefix.is_empty()).then_some(crash_prefix), "synced");
    Ok(())
}

fn maybe_crash_with_suffix(prefix: Option<&str>, suffix: &str) {
    if let Some(prefix) = prefix {
        maybe_crash_transaction(&format!("{prefix}-{suffix}"));
    }
}

fn cleanup_initializing_failure(
    layout: &GenerationLayout,
    journal_parent: &Path,
    initializing: &Path,
    original: &str,
) -> Result<(), String> {
    match layout.remove_dir_all(initializing) {
        Ok(()) => match layout.sync_directory(journal_parent) {
            Ok(()) => Err(original.into()),
            Err(cleanup) => Err(format!("{original}; cleanup failed: {cleanup}")),
        },
        Err(cleanup) => Err(format!("{original}; cleanup failed: {cleanup}")),
    }
}

fn sync_parent(layout: &GenerationLayout, path: &Path) -> Result<(), String> {
    let parent = path
        .parent()
        .ok_or_else(|| format!("{} lacks a parent directory", path.display()))?;
    layout.sync_directory(parent)
}

fn snapshot_path(repo: &Path) -> PathBuf {
    repo.join("tools/emel-model-parity-inventory/snapshots/pinned-ast-coverage.tsv")
}

#[cfg(test)]
fn external_includes_snapshot_path(repo: &Path) -> PathBuf {
    repo.join("tools/emel-model-parity-inventory/snapshots/pinned-external-includes.tsv")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::process::{Command, Stdio};

    static GENERATION_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    fn generation_test_guard() -> std::sync::MutexGuard<'static, ()> {
        GENERATION_TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    fn git_test(repo: &Path, arguments: &[&str]) {
        let status = Command::new("git")
            .arg("-C")
            .arg(repo)
            .args(arguments)
            .status()
            .unwrap();
        assert!(status.success(), "git {arguments:?} failed with {status}");
    }

    fn git_test_output(repo: &Path, arguments: &[&str]) -> String {
        let output = Command::new("git")
            .arg("-C")
            .arg(repo)
            .args(arguments)
            .output()
            .unwrap();
        assert!(output.status.success(), "git {arguments:?} failed");
        String::from_utf8(output.stdout).unwrap().trim().to_owned()
    }

    #[test]
    fn preflight_count_overflow_is_rejected() {
        let mut count = usize::MAX;
        assert_eq!(
            checked_increment(&mut count, "Rust declaration count"),
            Err("Rust declaration count overflow".to_owned())
        );
    }

    #[test]
    fn flags_fail_closed_on_duplicates_and_unknowns() {
        assert!(parse_flags(["--x".into(), "a".into(), "--x".into(), "b".into()]).is_err());
        assert!(reject_unknown(&BTreeMap::from([("x".into(), "a".into())]), &["y"]).is_err());
    }

    #[test]
    fn preflight_cli_requires_a_positive_explicit_deadline() {
        let missing = cli([
            "preflight-rust".into(),
            "--rust-repo".into(),
            ".".into(),
            "--rust-commit".into(),
            "HEAD".into(),
        ])
        .unwrap_err();
        assert_eq!(missing, "--deadline-seconds is required");

        let zero = cli([
            "preflight-rust".into(),
            "--rust-repo".into(),
            ".".into(),
            "--rust-commit".into(),
            "HEAD".into(),
            "--deadline-seconds".into(),
            "0".into(),
        ])
        .unwrap_err();
        assert_eq!(zero, "deadline-seconds must be a positive integer");
    }

    #[test]
    fn rust_snapshot_is_bound_to_one_commit_and_rejects_moving_ref_drift() {
        let temp = tempfile::tempdir().unwrap();
        let repo = temp.path().join("repo");
        fs::create_dir(&repo).unwrap();
        git_test(&repo, &["init", "-q"]);
        git_test(
            &repo,
            &["config", "user.email", "inventory@example.invalid"],
        );
        git_test(&repo, &["config", "user.name", "Inventory Test"]);
        let model = repo.join(RUST_PATH);
        fs::create_dir_all(&model).unwrap();
        fs::write(model.join("Cargo.toml"), b"[package]\nname='first'\n").unwrap();
        git_test(&repo, &["add", "."]);
        git_test(&repo, &["commit", "-qm", "first"]);
        let first = git_test_output(&repo, &["rev-parse", "HEAD"]);
        fs::write(model.join("Cargo.toml"), b"[package]\nname='second'\n").unwrap();
        git_test(&repo, &["add", "."]);
        git_test(&repo, &["commit", "-qm", "second"]);
        let second = git_test_output(&repo, &["rev-parse", "HEAD"]);
        git_test(&repo, &["update-ref", "refs/heads/moving", &first]);

        let deadline = Instant::now() + Duration::from_secs(10);
        let snapshot = load_rust_git_snapshot_with(&repo, "moving", deadline, || {
            git_test(&repo, &["update-ref", "refs/heads/moving", &second]);
        })
        .unwrap();

        assert_eq!(snapshot.resolved_commit, first);
        assert_eq!(snapshot.blobs.len(), 1);
        assert_eq!(snapshot.blobs[0].path, format!("{RUST_PATH}/Cargo.toml"));
        assert_eq!(snapshot.blobs[0].bytes, b"[package]\nname='first'\n");
        assert!(
            require_requested_rust_revision(&repo, "moving", &snapshot.resolved_commit, deadline)
                .unwrap_err()
                .contains("drifted")
        );
    }

    #[test]
    fn artifact_validation_rejects_tree_equivalent_wrong_source_commit() {
        let _generation_test = generation_test_guard();
        let temp = tempfile::tempdir().unwrap();
        let (layout, files) = GenerationLayout::bind(temp.path(), temp.path()).unwrap();
        drop(files);
        let generation_id = fresh_generation_id(&layout).unwrap();
        write_generation_status(&layout, GenerationStatus::Complete, &generation_id).unwrap();
        assert_eq!(
            validate_artifacts(
                temp.path(),
                temp.path(),
                &"0".repeat(40),
                EXPECTED_MODEL_TREE,
                EXPECTED_TEST_TREE,
                &"1".repeat(40),
            ),
            Err("validation tree identity drift".into())
        );
    }

    #[test]
    fn semantic_hash_is_version_bound() {
        assert_ne!(
            crate::hash::semantic_hash(&[schema::VERSION, "x"]),
            crate::hash::semantic_hash(&["old", "x"])
        );
    }

    #[test]
    fn conflicting_definitions_with_one_canonical_identity_fail_closed() {
        let definition = |payload: &str| Extracted {
            path: "src/emel/model/tensor/events.hpp".into(),
            kind: "type".into(),
            name: "emel::model::tensor::event::bind_storage".into(),
            payload: payload.into(),
            note: "clang-ast".into(),
        };
        let error = rows(
            "cpp",
            &[],
            vec![definition("first"), definition("conflicting")],
            &"a".repeat(40),
            &"b".repeat(40),
        )
        .unwrap_err();
        assert!(error.contains("duplicate ID has conflicting semantics"));
    }

    #[test]
    fn deterministic_difference_diagnostic_accepts_equal_bytes() {
        assert_eq!(byte_difference("artifact", "a.tsv", b"same", b"same"), None);
    }

    #[test]
    fn deterministic_difference_diagnostic_reports_length_mismatch() {
        let difference = byte_difference("artifact", "a.tsv", b"abc", b"abcd").unwrap();
        assert!(difference.contains("component=artifact file=a.tsv"));
        assert!(difference.contains("first_len=3 second_len=4"));
        assert!(difference.contains("first_diff_offset=3"));
        assert!(difference.contains("first_sha256="));
        assert!(difference.contains("second_sha256="));
        assert!(difference.contains("first_context=\"abc\""));
        assert!(difference.contains("second_context=\"abcd\""));
    }

    #[test]
    fn deterministic_difference_diagnostic_reports_escaped_content_mismatch() {
        let difference = byte_difference("snapshot", "coverage.tsv", b"a\nc", b"a\tc").unwrap();
        assert!(difference.contains("component=snapshot file=coverage.tsv"));
        assert!(difference.contains("first_len=3 second_len=3"));
        assert!(difference.contains("first_diff_offset=1"));
        assert!(difference.contains("first_context=\"a\\nc\""));
        assert!(difference.contains("second_context=\"a\\tc\""));
    }

    #[test]
    fn reconciliation_never_invents_location_or_leaf_name_links() {
        let make = |side: &str, path: &str, kind: &str, name: &str| {
            scan::row(
                side,
                &Extracted {
                    path: path.into(),
                    kind: kind.into(),
                    name: name.into(),
                    payload: "semantic".into(),
                    note: "test".into(),
                },
                "commit=source;model=model;tests=tests",
                "rust-tree",
            )
        };
        let mut source = vec![
            make(
                "cpp",
                "src/emel/model/loader/sm.hpp",
                "transition",
                "transition-row@49:9",
            ),
            make(
                "cpp",
                "tests/model/moshi/binding_tests.cpp",
                "assertion",
                "assertion@73:5",
            ),
            make(
                "cpp",
                "src/emel/model/detail.hpp",
                "algorithm",
                "load(std::span<const std::pair<int,int>>)",
            ),
        ];
        let mut rust = vec![
            make(
                "rust",
                "crates/emel-model/src/tensor/window/sm.rs",
                "transition",
                "transition-row@49:9",
            ),
            make(
                "rust",
                "crates/emel-model/src/tensor/tests.rs",
                "assertion",
                "assertion@73:5",
            ),
            make(
                "rust",
                "crates/emel-model/src/other.rs",
                "algorithm",
                "pair<int,int>>)",
            ),
        ];
        reconcile(&mut source, &mut rust);
        assert!(source.iter().all(|row| row.counterpart_id.is_empty()));
        assert!(rust.iter().all(|row| row.counterpart_id.is_empty()));
        assert!(source.iter().all(|row| row.disposition == "missing"));
        assert!(rust.iter().all(|row| row.disposition == "rust-only"));
    }

    #[test]
    fn generation_transaction_rolls_back_mid_commit() {
        let _generation_test = generation_test_guard();
        let temp = tempfile::tempdir().unwrap();
        let artifacts = temp.path().join("artifacts");
        let snapshots = temp.path().join("snapshots");
        fs::create_dir_all(&artifacts).unwrap();
        fs::create_dir_all(&snapshots).unwrap();
        let first = artifacts.join("first.tsv");
        let second = artifacts.join("second.tsv");
        let snapshot = snapshots.join("coverage.tsv");
        fs::write(&first, b"old-first").unwrap();
        fs::write(&second, b"old-second").unwrap();
        fs::write(&snapshot, b"old-snapshot").unwrap();
        let new_target = snapshots.join("new.tsv");
        let targets = vec![
            (first.clone(), b"new-first".to_vec()),
            (second.clone(), b"new-second".to_vec()),
            (snapshot.clone(), b"new-snapshot".to_vec()),
            (new_target.clone(), b"new-target".to_vec()),
        ];
        assert!(publish_paths(&artifacts, &targets, Some(1)).is_err());
        assert_eq!(fs::read(first).unwrap(), b"old-first");
        assert_eq!(fs::read(second).unwrap(), b"old-second");
        assert_eq!(fs::read(snapshot).unwrap(), b"old-snapshot");
        assert!(!new_target.exists());
        assert!(!artifacts.join(".emel-model-inventory-generation").exists());
    }

    #[test]
    fn generation_transaction_rolls_back_after_tsvs_before_snapshot() {
        let _generation_test = generation_test_guard();
        let temp = tempfile::tempdir().unwrap();
        let artifacts = temp.path().join("artifacts");
        let snapshots = temp.path().join("snapshots");
        fs::create_dir_all(&artifacts).unwrap();
        fs::create_dir_all(&snapshots).unwrap();
        let first = artifacts.join("first.tsv");
        let second = artifacts.join("second.tsv");
        let snapshot = snapshots.join("coverage.tsv");
        for (path, bytes) in [
            (&first, b"old-first".as_slice()),
            (&second, b"old-second".as_slice()),
            (&snapshot, b"old-snapshot".as_slice()),
        ] {
            fs::write(path, bytes).unwrap();
        }
        let targets = vec![
            (first.clone(), b"new-first".to_vec()),
            (second.clone(), b"new-second".to_vec()),
            (snapshot.clone(), b"new-snapshot".to_vec()),
        ];
        assert!(publish_paths(&artifacts, &targets, Some(2)).is_err());
        assert_eq!(fs::read(first).unwrap(), b"old-first");
        assert_eq!(fs::read(second).unwrap(), b"old-second");
        assert_eq!(fs::read(snapshot).unwrap(), b"old-snapshot");
    }

    fn valid_nine_target_fixture(root: &Path) -> (PathBuf, PathBuf, String, Artifacts) {
        let artifact_dir = root.join("artifacts");
        let rust_repo = root.join("rust-repo");
        valid_nine_target_fixture_at(artifact_dir, rust_repo)
    }

    fn valid_nine_target_fixture_at(
        artifact_dir: PathBuf,
        rust_repo: PathBuf,
    ) -> (PathBuf, PathBuf, String, Artifacts) {
        fs::create_dir_all(&artifact_dir).unwrap();
        fs::create_dir_all(snapshot_path(&rust_repo).parent().unwrap()).unwrap();
        let source_tree = format!(
            "commit={EXPECTED_SOURCE_COMMIT};model={EXPECTED_MODEL_TREE};tests={EXPECTED_TEST_TREE}"
        );
        let rust_tree = "c".repeat(40);
        let mut source = vec![scan::row(
            "cpp",
            &Extracted {
                path: "src/emel/model/a.hpp".into(),
                kind: "type".into(),
                name: "emel::model::A".into(),
                payload: "A".into(),
                note: "test-fixture".into(),
            },
            &source_tree,
            &rust_tree,
        )];
        let mut rust = vec![scan::row(
            "rust",
            &Extracted {
                path: "crates/emel-model/src/a.rs".into(),
                kind: "type".into(),
                name: "A".into(),
                payload: "A".into(),
                note: "test-fixture".into(),
            },
            &source_tree,
            &rust_tree,
        )];
        reconcile(&mut source, &mut rust);
        let mut reconciled = source.clone();
        reconciled.extend(rust.clone());
        reconciled.sort();
        let terminal = format!("{INVENTORY_HEADER}\n").into_bytes();
        let files = BTreeMap::from([
            ("open-gaps.tsv".into(), gaps(&reconciled)),
            (
                "primary-rust-inventory.tsv".into(),
                schema::inventory_tsv(&rust).unwrap(),
            ),
            (
                "primary-source-inventory.tsv".into(),
                schema::inventory_tsv(&source).unwrap(),
            ),
            (
                "proof-coverage-matrix.tsv".into(),
                proof_coverage(&reconciled),
            ),
            (
                "reconciled-inventory.tsv".into(),
                schema::inventory_tsv(&reconciled).unwrap(),
            ),
            ("terminal-rust-inventory.tsv".into(), terminal.clone()),
            ("terminal-source-inventory.tsv".into(), terminal),
        ]);
        let artifacts = Artifacts {
            files,
            coverage: include_bytes!("../snapshots/pinned-ast-coverage.tsv").to_vec(),
            external_includes: include_bytes!("../snapshots/pinned-external-includes.tsv").to_vec(),
        };
        for (name, bytes) in &artifacts.files {
            fs::write(artifact_dir.join(name), bytes).unwrap();
        }
        fs::write(snapshot_path(&rust_repo), &artifacts.coverage).unwrap();
        fs::write(
            external_includes_snapshot_path(&rust_repo),
            &artifacts.external_includes,
        )
        .unwrap();
        let (layout, files) = GenerationLayout::bind(&artifact_dir, &rust_repo).unwrap();
        drop(files);
        let generation_id = fresh_generation_id(&layout).unwrap();
        write_generation_status(&layout, GenerationStatus::Complete, &generation_id).unwrap();
        (artifact_dir, rust_repo, rust_tree, artifacts)
    }

    fn published_target_paths_for(artifact_dir: &Path, rust_repo: &Path) -> Vec<PathBuf> {
        let (layout, files) = GenerationLayout::bind(artifact_dir, rust_repo).unwrap();
        drop(files);
        published_target_paths(&layout)
    }

    fn deterministic_fixture_options(artifact_dir: &Path, rust_repo: &Path) -> ExtractOptions {
        ExtractOptions {
            source_repo: PathBuf::new(),
            source_commit: EXPECTED_SOURCE_COMMIT.into(),
            rust_repo: rust_repo.into(),
            rust_commit: "HEAD".into(),
            compile_commands: PathBuf::new(),
            artifact_dir: artifact_dir.into(),
        }
    }

    fn validate_nine_target_fixture(
        artifact_dir: &Path,
        rust_repo: &Path,
        rust_tree: &str,
    ) -> Result<(), String> {
        validate_artifacts(
            artifact_dir,
            rust_repo,
            EXPECTED_SOURCE_COMMIT,
            EXPECTED_MODEL_TREE,
            EXPECTED_TEST_TREE,
            rust_tree,
        )
    }

    fn wait_for_path(path: &Path) {
        let deadline = Instant::now() + Duration::from_secs(10);
        while fs::metadata(path).map_or(true, |metadata| metadata.len() == 0) {
            assert!(
                Instant::now() < deadline,
                "timed out waiting for {}",
                path.display()
            );
            std::thread::sleep(Duration::from_millis(10));
        }
    }

    #[test]
    #[ignore = "subprocess helper for concurrent generation readers"]
    fn concurrent_generation_reader_helper() {
        let artifact_dir = PathBuf::from(
            std::env::var_os("EMEL_MODEL_INVENTORY_CONCURRENT_ARTIFACT_DIR").unwrap(),
        );
        let rust_repo =
            PathBuf::from(std::env::var_os("EMEL_MODEL_INVENTORY_CONCURRENT_RUST_REPO").unwrap());
        let mode = std::env::var("EMEL_MODEL_INVENTORY_CONCURRENT_READER").unwrap();
        let (artifact_dir, rust_repo, rust_tree, artifacts) =
            valid_nine_target_fixture_at(artifact_dir, rust_repo);
        match mode.as_str() {
            "validate" => {
                validate_nine_target_fixture(&artifact_dir, &rust_repo, &rust_tree).unwrap();
            }
            "deterministic" => {
                let options = deterministic_fixture_options(&artifact_dir, &rust_repo);
                check_deterministic_with(&options, |_| Ok(artifacts.clone())).unwrap();
            }
            _ => panic!("unknown concurrent reader mode: {mode}"),
        }
    }

    #[test]
    #[ignore = "subprocess helper for nine-target publication and crash recovery"]
    fn nine_target_writer_helper() {
        let artifact_dir =
            PathBuf::from(std::env::var_os("EMEL_MODEL_INVENTORY_CRASH_ARTIFACT_DIR").unwrap());
        let rust_repo =
            PathBuf::from(std::env::var_os("EMEL_MODEL_INVENTORY_CRASH_RUST_REPO").unwrap());
        let targets = published_target_paths_for(&artifact_dir, &rust_repo)
            .into_iter()
            .enumerate()
            .map(|(index, path)| (path, format!("new-{index}").into_bytes()))
            .collect::<Vec<_>>();
        publish_paths(&artifact_dir, &targets, None).unwrap();
    }

    fn spawn_concurrent_reader(
        artifact_dir: &Path,
        rust_repo: &Path,
        mode: &str,
        entered: &Path,
        release: &Path,
    ) -> std::process::Child {
        Command::new(std::env::current_exe().unwrap())
            .args([
                "--ignored",
                "--exact",
                "engine::tests::concurrent_generation_reader_helper",
                "--nocapture",
            ])
            .env("EMEL_MODEL_INVENTORY_CONCURRENT_ARTIFACT_DIR", artifact_dir)
            .env("EMEL_MODEL_INVENTORY_CONCURRENT_RUST_REPO", rust_repo)
            .env("EMEL_MODEL_INVENTORY_CONCURRENT_READER", mode)
            .env("EMEL_MODEL_INVENTORY_LOCK_ENTERED", entered)
            .env("EMEL_MODEL_INVENTORY_LOCK_RELEASE", release)
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .unwrap()
    }

    fn spawn_nine_target_writer(
        artifact_dir: &Path,
        rust_repo: &Path,
        attempt: &Path,
    ) -> std::process::Child {
        Command::new(std::env::current_exe().unwrap())
            .args([
                "--ignored",
                "--exact",
                "engine::tests::nine_target_writer_helper",
                "--nocapture",
            ])
            .env("EMEL_MODEL_INVENTORY_CRASH_ARTIFACT_DIR", artifact_dir)
            .env("EMEL_MODEL_INVENTORY_CRASH_RUST_REPO", rust_repo)
            .env("EMEL_MODEL_INVENTORY_LOCK_ATTEMPT", attempt)
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .unwrap()
    }

    fn wait_for_child_success(child: &mut std::process::Child, label: &str) {
        let deadline = Instant::now() + Duration::from_secs(5);
        loop {
            if let Some(status) = child.try_wait().unwrap() {
                assert!(status.success(), "{label} failed");
                return;
            }
            if Instant::now() >= deadline {
                child.kill().unwrap();
                child.wait().unwrap();
                panic!("timed out waiting for {label}");
            }
            std::thread::sleep(Duration::from_millis(10));
        }
    }

    #[test]
    fn canonical_resource_lock_paths_are_sorted_and_deduplicated() {
        let _generation_test = generation_test_guard();
        let temp = tempfile::tempdir().unwrap();
        let first = temp.path().join("z-owner");
        let second = temp.path().join("a-owner");
        let (layout, files) =
            GenerationLayout::bind_owners(&first, &first, &[second, first.clone()]).unwrap();
        assert_eq!(layout.owners.len(), 2);
        assert!(layout.owners[0].lock_path < layout.owners[1].lock_path);
        drop(files);

        let rust_repo = temp.path().join("rust-repo");
        let snapshot_owner = snapshot_path(&rust_repo).parent().unwrap().to_path_buf();
        let (layout, files) = GenerationLayout::bind(&snapshot_owner, &rust_repo).unwrap();
        drop(files);
        write_generation_status(&layout, GenerationStatus::Complete, "test-generation").unwrap();
        let locks = SharedGenerationLocks::acquire(&snapshot_owner, &rust_repo).unwrap();
        #[cfg(unix)]
        assert!(locks.len() > 1);
        #[cfg(not(unix))]
        assert_eq!(locks.len(), 1);
    }

    #[cfg(unix)]
    #[test]
    fn owner_aliases_bind_once_and_retargeting_the_alias_cannot_redirect_io() {
        use std::os::unix::fs::symlink;

        let _generation_test = generation_test_guard();
        let temp = tempfile::tempdir().unwrap();
        let rust_repo = temp.path().join("rust-repo");
        let first = snapshot_path(&rust_repo).parent().unwrap().to_path_buf();
        let second = temp.path().join("second");
        fs::create_dir_all(&first).unwrap();
        fs::create_dir_all(&second).unwrap();
        let alias = temp.path().join("owner-alias");
        symlink(&first, &alias).unwrap();
        let (layout, files) = GenerationLayout::bind(&alias, &rust_repo).unwrap();
        assert_eq!(layout.owners.len(), 1);
        assert_eq!(layout.artifact_owner, first.canonicalize().unwrap());
        drop(files);
        write_generation_status(&layout, GenerationStatus::Complete, "before-retarget").unwrap();
        let reader = SharedGenerationLocks::acquire(&alias, &rust_repo).unwrap();

        fs::remove_file(&alias).unwrap();
        symlink(&second, &alias).unwrap();

        reader.finish().unwrap();
        assert!(!second.join(GENERATION_TOKEN).exists());
    }

    #[cfg(unix)]
    #[test]
    fn renamed_and_recreated_owners_keep_one_lock_domain_and_capability_relative_io() {
        let _generation_test = generation_test_guard();
        let temp = tempfile::tempdir().unwrap();
        let (artifact_dir, rust_repo, _, _) = valid_nine_target_fixture(temp.path());
        let writer = ExclusiveGenerationLocks::acquire(&artifact_dir, &rust_repo).unwrap();
        let owner_paths = writer
            .layout
            .owners
            .iter()
            .map(|owner| owner.path.clone())
            .collect::<Vec<_>>();
        let mut displaced = Vec::new();

        for (index, owner) in owner_paths.iter().enumerate() {
            let moved = owner.with_extension(format!("displaced-{index}"));
            fs::rename(owner, &moved).unwrap();
            fs::create_dir(owner).unwrap();
            fs::write(owner.join("replacement-sentinel"), b"replacement").unwrap();
            displaced.push(moved);
        }

        assert!(writer.layout.verify().is_err());
        for (index, owner) in owner_paths.iter().enumerate() {
            let probe = owner.join(format!("capability-probe-{index}"));
            write_synced_with_crash_points(&writer.layout, &probe, b"retained", "").unwrap();
            assert!(!probe.exists(), "ambient replacement owner was mutated");
            assert_eq!(
                fs::read(displaced[index].join(format!("capability-probe-{index}"))).unwrap(),
                b"retained"
            );
            assert_eq!(
                fs::read(owner.join("replacement-sentinel")).unwrap(),
                b"replacement"
            );
        }

        let (contender_layout, contender_files) =
            GenerationLayout::bind(&artifact_dir, &rust_repo).unwrap();
        assert!(try_generation_locks_once(&contender_layout, contender_files, false).is_err());

        drop(writer);
        let recovered = ExclusiveGenerationLocks::acquire(&artifact_dir, &rust_repo).unwrap();
        write_generation_status(
            &recovered.layout,
            GenerationStatus::Complete,
            "replacement-generation",
        )
        .unwrap();
        assert_eq!(
            read_complete_generation(&recovered.layout).unwrap(),
            "replacement-generation"
        );
    }

    #[cfg(unix)]
    #[test]
    fn lock_symlinks_hardlinks_and_replacements_fail_closed_without_deadlock() {
        use std::os::unix::fs::symlink;

        let temp = tempfile::tempdir().unwrap();
        let first = temp.path().join("first");
        let second = temp.path().join("second");
        let (layout, files) = GenerationLayout::bind_owners(&first, &second, &[]).unwrap();
        drop(files);
        let first_lock = layout.owners[0].lock_path.clone();
        let second_lock = layout.owners[1].lock_path.clone();

        fs::remove_file(&second_lock).unwrap();
        symlink(&first_lock, &second_lock).unwrap();
        assert!(GenerationLayout::bind_owners(&first, &second, &[]).is_err());
        fs::remove_file(&second_lock).unwrap();
        fs::hard_link(&first_lock, &second_lock).unwrap();
        assert!(GenerationLayout::bind_owners(&first, &second, &[]).is_err());
        fs::remove_file(&second_lock).unwrap();
        fs::write(&second_lock, b"").unwrap();

        let (bound, held) = GenerationLayout::bind_owners(&first, &second, &[]).unwrap();
        let replaced = bound.owners[0].lock_path.clone();
        fs::rename(&replaced, replaced.with_extension("old")).unwrap();
        assert!(bound.verify().is_err());
        fs::write(&replaced, b"").unwrap();
        assert!(bound.verify().is_err());
        drop(held);
    }

    #[cfg(windows)]
    #[test]
    fn windows_lock_hardlink_aliases_and_replacements_fail_closed() {
        let temp = tempfile::tempdir().unwrap();
        let first = temp.path().join("first");
        let second = temp.path().join("second");
        let (layout, files) = GenerationLayout::bind_owners(&first, &second, &[]).unwrap();
        drop(files);
        let first_lock = layout.owners[0].lock_path.clone();
        let second_lock = layout.owners[1].lock_path.clone();

        fs::remove_file(&second_lock).unwrap();
        fs::hard_link(&first_lock, &second_lock).unwrap();
        assert!(GenerationLayout::bind_owners(&first, &second, &[]).is_err());
        fs::remove_file(&second_lock).unwrap();
        fs::write(&second_lock, b"").unwrap();

        let (bound, held) = GenerationLayout::bind_owners(&first, &second, &[]).unwrap();
        let replaced = bound.owners[0].lock_path.clone();
        assert!(fs::rename(&replaced, replaced.with_extension("old")).is_err());
        assert!(fs::remove_file(&replaced).is_err());
        bound.verify().unwrap();
        drop(held);
    }

    #[cfg(windows)]
    #[test]
    fn windows_owner_replacement_and_second_writer_are_rejected_while_writer_is_live() {
        let temp = tempfile::tempdir().unwrap();
        let owner = temp.path().join("owner");
        let (old_layout, old_files) = GenerationLayout::bind_owners(&owner, &owner, &[]).unwrap();
        let old_writer = ExclusiveGenerationLocks::acquire_layout(old_layout, old_files).unwrap();

        let displaced = temp.path().join("displaced-owner");
        assert!(fs::rename(&owner, &displaced).is_err());
        assert!(fs::remove_dir_all(&owner).is_err());
        old_writer.layout.verify().unwrap();

        let (contender_layout, contender_files) =
            GenerationLayout::bind_owners(&owner, &owner, &[]).unwrap();
        assert!(try_generation_locks_once(&contender_layout, contender_files, false).is_err());
        old_writer.layout.verify().unwrap();
    }

    #[cfg(windows)]
    #[test]
    fn windows_lock_symlink_is_rejected_when_target_supports_symlinks() {
        use std::os::windows::fs::symlink_file;

        let temp = tempfile::tempdir().unwrap();
        let first = temp.path().join("first");
        let second = temp.path().join("second");
        let (layout, files) = GenerationLayout::bind_owners(&first, &second, &[]).unwrap();
        drop(files);
        let first_lock = layout.owners[0].lock_path.clone();
        let second_lock = layout.owners[1].lock_path.clone();
        fs::remove_file(&second_lock).unwrap();
        match symlink_file(&first_lock, &second_lock) {
            Ok(()) => {
                assert!(GenerationLayout::bind_owners(&first, &second, &[]).is_err());
            }
            Err(error) if error.kind() == ErrorKind::PermissionDenied => {}
            Err(error) => panic!("cannot create Windows symlink fixture: {error}"),
        }
    }

    #[cfg(windows)]
    #[test]
    fn windows_generation_token_replacement_is_valid_under_long_lived_locks() {
        let temp = tempfile::tempdir().unwrap();
        let (artifact_dir, rust_repo, _, _) = valid_nine_target_fixture(temp.path());
        let writer = ExclusiveGenerationLocks::acquire(&artifact_dir, &rust_repo).unwrap();
        let first = read_complete_generation(&writer.layout).unwrap();
        let token = generation_token_path(&writer.layout.owners[0]);
        let alias = token.with_extension("hard-link");

        fs::hard_link(&token, &alias).unwrap();
        assert!(
            write_generation_status(&writer.layout, GenerationStatus::InProgress, "next")
                .unwrap_err()
                .contains("multiple hard links")
        );
        fs::remove_file(alias).unwrap();
        assert_eq!(read_complete_generation(&writer.layout).unwrap(), first);

        write_generation_status(&writer.layout, GenerationStatus::InProgress, "next").unwrap();
        for owner in &writer.layout.owners {
            assert_eq!(
                read_generation_token(owner).unwrap(),
                (GenerationStatus::InProgress, "next".to_owned())
            );
        }
        write_generation_status(&writer.layout, GenerationStatus::Complete, "next").unwrap();
        assert_eq!(read_complete_generation(&writer.layout).unwrap(), "next");

        let owner = &writer.layout.owners[0];
        assert!(fs::rename(&owner.path, owner.path.with_extension("renamed")).is_err());
        assert!(fs::remove_file(&owner.lock_path).is_err());
        writer.layout.verify().unwrap();
    }

    #[test]
    fn reader_generation_token_detects_a_complete_transaction_between_samples() {
        let _generation_test = generation_test_guard();
        let temp = tempfile::tempdir().unwrap();
        let (artifact_dir, rust_repo, _, _) = valid_nine_target_fixture(temp.path());
        let reader = SharedGenerationLocks::acquire(&artifact_dir, &rust_repo).unwrap();
        let next = fresh_generation_id(&reader.layout).unwrap();
        write_generation_status(&reader.layout, GenerationStatus::InProgress, &next).unwrap();
        write_generation_status(&reader.layout, GenerationStatus::Complete, &next).unwrap();
        assert!(reader.finish().unwrap_err().contains("changed"));
    }

    #[test]
    fn recovery_reconciles_mismatched_generation_tokens_to_one_complete_identity() {
        let _generation_test = generation_test_guard();
        let temp = tempfile::tempdir().unwrap();
        let (artifact_dir, rust_repo, _, _) = valid_nine_target_fixture(temp.path());
        let writer = ExclusiveGenerationLocks::acquire(&artifact_dir, &rust_repo).unwrap();
        fs::write(
            generation_token_path(&writer.layout.owners[0]),
            format!("{GENERATION_TOKEN_SCHEMA}\tin-progress\tinterrupted\n"),
        )
        .unwrap();
        fs::write(
            generation_token_path(&writer.layout.owners[1]),
            format!("{GENERATION_TOKEN_SCHEMA}\tcomplete\tdifferent\n"),
        )
        .unwrap();

        reconcile_generation(&writer, &published_recovery_targets(&writer.layout)).unwrap();
        assert!(!read_complete_generation(&writer.layout).unwrap().is_empty());
    }

    #[test]
    fn shared_readers_exclude_both_partial_overlap_patterns() {
        let _generation_test = generation_test_guard();
        for mode in ["validate", "deterministic"] {
            for overlap in ["snapshot-owner", "artifact-owner"] {
                let temp = tempfile::tempdir().unwrap();
                let reader_artifact = temp.path().join("reader-artifacts");
                let reader_rust = temp.path().join("reader-rust");
                let (reader_artifact, reader_rust, _, _) =
                    valid_nine_target_fixture_at(reader_artifact, reader_rust);
                let (writer_artifact, writer_rust) = if overlap == "snapshot-owner" {
                    (temp.path().join("writer-artifacts"), reader_rust.clone())
                } else {
                    (reader_artifact.clone(), temp.path().join("writer-rust"))
                };
                let old_reader_targets = published_target_paths_for(&reader_artifact, &reader_rust)
                    .into_iter()
                    .map(|path| {
                        let bytes = fs::read(&path).unwrap();
                        (path, bytes)
                    })
                    .collect::<Vec<_>>();
                let writer_targets = published_target_paths_for(&writer_artifact, &writer_rust);
                let entered = temp.path().join("reader-entered");
                let release = temp.path().join("reader-release");
                let writer_attempt = temp.path().join("writer-attempt");
                let mut reader = spawn_concurrent_reader(
                    &reader_artifact,
                    &reader_rust,
                    mode,
                    &entered,
                    &release,
                );
                wait_for_path(&entered);
                assert_eq!(fs::read_to_string(&entered).unwrap(), "shared");

                let mut writer =
                    spawn_nine_target_writer(&writer_artifact, &writer_rust, &writer_attempt);
                wait_for_path(&writer_attempt);
                assert_eq!(fs::read_to_string(&writer_attempt).unwrap(), "exclusive");
                std::thread::sleep(Duration::from_millis(100));
                assert!(reader.try_wait().unwrap().is_none());
                assert!(writer.try_wait().unwrap().is_none());
                assert!(!writer_artifact.join(GENERATION_JOURNAL).exists());
                assert!(
                    !writer_artifact
                        .join(INITIALIZING_GENERATION_JOURNAL)
                        .exists()
                );
                for (path, bytes) in &old_reader_targets {
                    assert_eq!(
                        fs::read(path).unwrap(),
                        *bytes,
                        "{mode}/{overlap} reader saw mixed state"
                    );
                }

                fs::write(&release, b"release").unwrap();
                assert!(
                    reader.wait().unwrap().success(),
                    "{mode}/{overlap} reader failed"
                );
                assert!(
                    writer.wait().unwrap().success(),
                    "writer failed after {mode}/{overlap}"
                );
                for (index, path) in writer_targets.iter().enumerate() {
                    assert_eq!(fs::read(path).unwrap(), format!("new-{index}").as_bytes());
                }
                assert!(!writer_artifact.join(GENERATION_JOURNAL).exists());
                assert!(
                    !writer_artifact
                        .join(INITIALIZING_GENERATION_JOURNAL)
                        .exists()
                );
            }
        }
    }

    #[test]
    fn ancestor_lock_serializes_disjoint_resource_sets_without_deadlock() {
        let _generation_test = generation_test_guard();
        let temp = tempfile::tempdir().unwrap();
        let reader_artifact = temp.path().join("reader-artifacts");
        let reader_rust = temp.path().join("reader-rust");
        let (reader_artifact, reader_rust, _, _) =
            valid_nine_target_fixture_at(reader_artifact, reader_rust);
        let writer_artifact = temp.path().join("writer-artifacts");
        let writer_rust = temp.path().join("writer-rust");
        let writer_targets = published_target_paths_for(&writer_artifact, &writer_rust);
        let entered = temp.path().join("reader-entered");
        let release = temp.path().join("reader-release");
        let writer_attempt = temp.path().join("writer-attempt");
        let mut reader = spawn_concurrent_reader(
            &reader_artifact,
            &reader_rust,
            "validate",
            &entered,
            &release,
        );
        wait_for_path(&entered);
        let mut writer = spawn_nine_target_writer(&writer_artifact, &writer_rust, &writer_attempt);
        wait_for_path(&writer_attempt);
        assert!(reader.try_wait().unwrap().is_none());
        assert!(writer.try_wait().unwrap().is_none());
        fs::write(&release, b"release").unwrap();
        assert!(reader.wait().unwrap().success());
        wait_for_child_success(&mut writer, "serialized writer");
        for (index, path) in writer_targets.iter().enumerate() {
            assert_eq!(fs::read(path).unwrap(), format!("new-{index}").as_bytes());
        }
    }

    #[test]
    fn nine_target_crashes_are_rejected_until_exact_recovery() {
        let _generation_test = generation_test_guard();
        for point in [
            "generation-in-progress-0",
            "generation-in-progress-1",
            "commit-6",
            "commit-8",
        ] {
            let temp = tempfile::tempdir().unwrap();
            let (artifact_dir, rust_repo, rust_tree, artifacts) =
                valid_nine_target_fixture(temp.path());
            let old_targets = published_target_paths_for(&artifact_dir, &rust_repo)
                .into_iter()
                .map(|path| {
                    let bytes = fs::read(&path).unwrap();
                    (path, bytes)
                })
                .collect::<Vec<_>>();
            let status = Command::new(std::env::current_exe().unwrap())
                .args([
                    "--ignored",
                    "--exact",
                    "engine::tests::nine_target_writer_helper",
                    "--nocapture",
                ])
                .env("EMEL_MODEL_INVENTORY_CRASH_ARTIFACT_DIR", &artifact_dir)
                .env("EMEL_MODEL_INVENTORY_CRASH_RUST_REPO", &rust_repo)
                .env("EMEL_MODEL_INVENTORY_CRASH_POINT", point)
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .status()
                .unwrap();
            assert!(!status.success(), "helper did not crash at {point}");

            let released_lock = ExclusiveGenerationLocks::try_acquire(&artifact_dir, &rust_repo)
                .unwrap_or_else(|error| panic!("writer crash retained generation lock: {error}"));
            #[cfg(unix)]
            assert!(released_lock.len() > 2);
            #[cfg(not(unix))]
            assert_eq!(released_lock.len(), 2);
            drop(released_lock);

            let validation = validate_nine_target_fixture(&artifact_dir, &rust_repo, &rust_tree);
            assert!(
                validation
                    .unwrap_err()
                    .contains("published generation is not complete")
            );
            let options = deterministic_fixture_options(&artifact_dir, &rust_repo);
            let deterministic =
                check_deterministic_with(&options, |_| Ok(artifacts.clone())).unwrap_err();
            assert!(deterministic.contains("published generation is not complete"));

            recover_published_generation(&artifact_dir, &rust_repo).unwrap();
            for (path, bytes) in &old_targets {
                assert_eq!(fs::read(path).unwrap(), *bytes, "recovery drift at {point}");
            }
            validate_nine_target_fixture(&artifact_dir, &rust_repo, &rust_tree).unwrap();
            check_deterministic_with(&options, |_| Ok(artifacts.clone())).unwrap();
        }
    }

    #[test]
    fn crash_during_complete_token_publication_recovers_the_committed_generation() {
        let _generation_test = generation_test_guard();
        let temp = tempfile::tempdir().unwrap();
        let (artifact_dir, rust_repo, _, _) = valid_nine_target_fixture(temp.path());
        let targets = published_target_paths_for(&artifact_dir, &rust_repo);
        let status = Command::new(std::env::current_exe().unwrap())
            .args([
                "--ignored",
                "--exact",
                "engine::tests::nine_target_writer_helper",
                "--nocapture",
            ])
            .env("EMEL_MODEL_INVENTORY_CRASH_ARTIFACT_DIR", &artifact_dir)
            .env("EMEL_MODEL_INVENTORY_CRASH_RUST_REPO", &rust_repo)
            .env("EMEL_MODEL_INVENTORY_CRASH_POINT", "generation-complete-0")
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .unwrap();
        assert!(!status.success());
        assert!(SharedGenerationLocks::acquire(&artifact_dir, &rust_repo).is_err());

        recover_published_generation(&artifact_dir, &rust_repo).unwrap();
        for (index, path) in targets.iter().enumerate() {
            assert_eq!(fs::read(path).unwrap(), format!("new-{index}").as_bytes());
        }
        SharedGenerationLocks::acquire(&artifact_dir, &rust_repo)
            .unwrap()
            .finish()
            .unwrap();
    }

    #[test]
    fn initializing_generation_is_rejected_until_exact_recovery() {
        let _generation_test = generation_test_guard();
        let temp = tempfile::tempdir().unwrap();
        let (artifact_dir, rust_repo, rust_tree, artifacts) =
            valid_nine_target_fixture(temp.path());
        fs::create_dir(artifact_dir.join(INITIALIZING_GENERATION_JOURNAL)).unwrap();
        assert!(
            validate_nine_target_fixture(&artifact_dir, &rust_repo, &rust_tree)
                .unwrap_err()
                .contains("published generation is not quiescent")
        );
        let options = deterministic_fixture_options(&artifact_dir, &rust_repo);
        assert!(
            check_deterministic_with(&options, |_| Ok(artifacts.clone()))
                .unwrap_err()
                .contains("published generation is not quiescent")
        );
        recover_published_generation(&artifact_dir, &rust_repo).unwrap();
        validate_nine_target_fixture(&artifact_dir, &rust_repo, &rust_tree).unwrap();
        check_deterministic_with(&options, |_| Ok(artifacts.clone())).unwrap();
    }

    #[test]
    fn validation_binds_both_pinned_snapshots() {
        let _generation_test = generation_test_guard();
        let temp = tempfile::tempdir().unwrap();
        let (artifact_dir, rust_repo, rust_tree, artifacts) =
            valid_nine_target_fixture(temp.path());
        fs::write(snapshot_path(&rust_repo), b"drifted coverage\n").unwrap();
        assert_eq!(
            validate_nine_target_fixture(&artifact_dir, &rust_repo, &rust_tree),
            Err("pinned AST coverage snapshot drift".into())
        );
        fs::write(snapshot_path(&rust_repo), &artifacts.coverage).unwrap();
        fs::write(
            external_includes_snapshot_path(&rust_repo),
            b"drifted external includes\n",
        )
        .unwrap();
        assert_eq!(
            validate_nine_target_fixture(&artifact_dir, &rust_repo, &rust_tree),
            Err("pinned external include snapshot drift".into())
        );
    }

    #[test]
    #[ignore = "subprocess helper for transaction crash recovery"]
    fn transaction_crash_helper() {
        let root = PathBuf::from(std::env::var_os("EMEL_MODEL_INVENTORY_CRASH_ROOT").unwrap());
        let targets = (0..3)
            .map(|index| {
                (
                    root.join(format!("target-{index}.tsv")),
                    format!("new-{index}").into_bytes(),
                )
            })
            .collect::<Vec<_>>();
        publish_paths(&root, &targets, None).unwrap();
    }

    #[test]
    fn generation_journal_recovers_subprocess_crashes_at_every_boundary() {
        let _generation_test = generation_test_guard();
        let mut points = vec![
            "init-dir-created".to_owned(),
            "init-manifest-temp-created".to_owned(),
            "init-manifest-written".to_owned(),
            "init-manifest-synced".to_owned(),
            "init-manifest-renamed".to_owned(),
            "init-manifest-directory-synced".to_owned(),
            "journal-renamed".to_owned(),
            "journal-published".to_owned(),
            "stage-dir-created".to_owned(),
            "stage-dir-synced".to_owned(),
            "backup-dir-created".to_owned(),
            "backup-dir-synced".to_owned(),
        ];
        for index in 0..3 {
            points.extend([
                format!("stage-{index}-created"),
                format!("stage-{index}-written"),
                format!("stage-{index}-synced"),
                format!("backup-{index}-copied"),
                format!("backup-{index}-synced"),
                format!("entry-{index}-prepared"),
            ]);
        }
        points.extend([
            "prepared".to_owned(),
            "commit-0".to_owned(),
            "commit-1".to_owned(),
            "commit-2".to_owned(),
            "complete".to_owned(),
        ]);
        for point in points {
            let temp = tempfile::tempdir().unwrap();
            let targets = (0..3)
                .map(|index| {
                    let path = temp.path().join(format!("target-{index}.tsv"));
                    fs::write(&path, format!("old-{index}")).unwrap();
                    (path, format!("new-{index}").into_bytes())
                })
                .collect::<Vec<_>>();
            let status = Command::new(std::env::current_exe().unwrap())
                .args([
                    "--ignored",
                    "--exact",
                    "engine::tests::transaction_crash_helper",
                    "--nocapture",
                ])
                .env("EMEL_MODEL_INVENTORY_CRASH_ROOT", temp.path())
                .env("EMEL_MODEL_INVENTORY_CRASH_POINT", &point)
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .status()
                .unwrap();
            assert!(!status.success(), "helper did not crash at {point}");

            publish_paths(temp.path(), &targets, None).unwrap();
            for (index, (target, _)) in targets.iter().enumerate() {
                assert_eq!(fs::read(target).unwrap(), format!("new-{index}").as_bytes());
            }
            assert!(
                !temp
                    .path()
                    .join(".emel-model-inventory-generation")
                    .exists()
            );
        }
    }

    #[test]
    fn generation_journal_rejects_any_target_set_drift() {
        let temp = tempfile::tempdir().unwrap();
        let (layout, _) = GenerationLayout::bind_owners(temp.path(), temp.path(), &[]).unwrap();
        let journal = layout.artifact_path(GENERATION_JOURNAL);
        fs::create_dir_all(journal.join("stage")).unwrap();
        fs::create_dir_all(journal.join("backup")).unwrap();
        let expected = layout.artifact_path("expected.tsv");
        let unexpected = layout.artifact_path("unexpected.tsv");
        let manifest = TransactionManifest {
            revision: 0,
            phase: "prepared".into(),
            committed: 0,
            entries: vec![TransactionEntry {
                target: unexpected,
                staged: journal.join("stage/0"),
                backup: journal.join("backup/0"),
                existed: false,
            }],
        };
        write_manifest(&layout, &journal, &manifest).unwrap();
        let targets = vec![(expected, b"new".to_vec())];
        assert_eq!(
            recover_journal(&layout, &journal, &targets),
            Err("generation journal target set drift at index 0".into())
        );
    }

    #[test]
    #[allow(clippy::too_many_lines)]
    fn relational_validation_rejects_id_set_tree_and_counterpart_corruption() {
        let source_tree = format!(
            "commit={EXPECTED_SOURCE_COMMIT};model={};tests={}",
            "a".repeat(40),
            "b".repeat(40)
        );
        let rust_tree = "c".repeat(40);
        let make = |side: &str, path: &str, name: &str| {
            scan::row(
                side,
                &Extracted {
                    path: path.into(),
                    kind: "type".into(),
                    name: name.into(),
                    payload: "semantic".into(),
                    note: "test".into(),
                },
                &source_tree,
                &rust_tree,
            )
        };
        let source = vec![make("cpp", "src/emel/model/a.hpp", "A")];
        let rust = vec![make("rust", "crates/emel-model/src/a.rs", "A")];
        let reconciled = [source.clone(), rust.clone()].concat();
        let coverage = reconciled
            .iter()
            .map(|row| schema::CoverageRow {
                item_id: row.item_id.clone(),
                terminal_tree: rust_tree.clone(),
                status: "open".into(),
            })
            .collect::<Vec<_>>();
        let gaps = reconciled
            .iter()
            .map(|row| schema::GapRow {
                item_id: row.item_id.clone(),
                gap_kind: "unmatched".into(),
                owner_lane: "lane".into(),
                blocker: "missing".into(),
                next_proof: "prove".into(),
                status: "open".into(),
            })
            .collect::<Vec<_>>();
        assert!(
            validate_relations(
                &source,
                &rust,
                &reconciled,
                &coverage,
                &gaps,
                &source_tree,
                &rust_tree,
            )
            .is_ok()
        );

        let mut wrong_coverage = coverage.clone();
        wrong_coverage[0].item_id = "wrong".into();
        assert!(
            validate_relations(
                &source,
                &rust,
                &reconciled,
                &wrong_coverage,
                &gaps,
                &source_tree,
                &rust_tree,
            )
            .is_err()
        );
        let mut missing_gaps = gaps.clone();
        missing_gaps.pop();
        assert!(
            validate_relations(
                &source,
                &rust,
                &reconciled,
                &coverage,
                &missing_gaps,
                &source_tree,
                &rust_tree,
            )
            .is_err()
        );
        let mut mixed = reconciled;
        mixed[1].rust_tree = "d".repeat(40);
        assert!(
            validate_relations(
                &source,
                &rust,
                &mixed,
                &coverage,
                &gaps,
                &source_tree,
                &rust_tree,
            )
            .is_err()
        );
        let mut linked_source = source;
        linked_source[0].counterpart_id = rust[0].item_id.clone();
        linked_source[0].disposition = "partial".into();
        let asymmetrical = [linked_source.clone(), rust.clone()].concat();
        assert!(
            validate_relations(
                &linked_source,
                &rust,
                &asymmetrical,
                &coverage,
                &gaps,
                &source_tree,
                &rust_tree,
            )
            .is_err()
        );
    }

    #[test]
    fn reconciliation_links_only_unique_same_component_structural_matches() {
        let source_tree = format!(
            "commit={EXPECTED_SOURCE_COMMIT};model={};tests={}",
            "a".repeat(40),
            "b".repeat(40)
        );
        let rust_tree = "c".repeat(40);
        let make = |side: &str, path: &str, kind: &str, name: &str| {
            scan::row(
                side,
                &Extracted {
                    path: path.into(),
                    kind: kind.into(),
                    name: name.into(),
                    payload: name.into(),
                    note: "test".into(),
                },
                &source_tree,
                &rust_tree,
            )
        };

        let mut source = vec![make(
            "cpp",
            "src/emel/model/tensor/sm.hpp",
            "action",
            "emel::model::tensor::load(void ())",
        )];
        let mut rust = vec![make(
            "rust",
            "crates/emel-model/src/tensor/sm.rs",
            "action",
            "effect_load",
        )];
        reconcile(&mut source, &mut rust);
        assert_eq!(source[0].counterpart_id, rust[0].item_id);
        assert_eq!(rust[0].counterpart_id, source[0].item_id);
        assert_eq!(source[0].disposition, "partial");
        assert_eq!(rust[0].disposition, "partial");

        let mut ambiguous_source = vec![
            make(
                "cpp",
                "src/emel/model/tensor/sm.hpp",
                "action",
                "emel::model::tensor::load(int)",
            ),
            make(
                "cpp",
                "src/emel/model/tensor/sm.hpp",
                "action",
                "emel::model::tensor::load(float)",
            ),
        ];
        let mut one_rust = vec![make(
            "rust",
            "crates/emel-model/src/tensor/sm.rs",
            "action",
            "effect_load",
        )];
        reconcile(&mut ambiguous_source, &mut one_rust);
        assert!(
            ambiguous_source
                .iter()
                .all(|row| row.counterpart_id.is_empty())
        );
        assert!(one_rust[0].counterpart_id.is_empty());

        let mut cross_source = vec![make(
            "cpp",
            "src/emel/model/loader/sm.hpp",
            "action",
            "load",
        )];
        let mut cross_rust = vec![make(
            "rust",
            "crates/emel-model/src/tensor/sm.rs",
            "action",
            "effect_load",
        )];
        reconcile(&mut cross_source, &mut cross_rust);
        assert!(cross_source[0].counterpart_id.is_empty());
        assert!(cross_rust[0].counterpart_id.is_empty());
    }
}
