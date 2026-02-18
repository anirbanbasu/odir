//! Utility functions for the Ollama Downloader in Rust (ODIR),
//! including model presence checks, downloading blobs, saving manifests,
//! and cleaning up temporary files.
use crate::downloader::model_downloader::{DownloaderError, Result};
use indicatif::{ProgressBar, ProgressStyle};
use log::{debug, error, info, warn};
use reqwest::blocking::Client;
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::HashSet;
use std::env;
use std::fs;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::Command;
use tempfile::NamedTempFile;

/// Check if a model is present in the Ollama server.
///
/// # Arguments
/// * `client` - HTTP client for making requests
/// * `server_url` - Base URL of the Ollama server
/// * `model_names` - Model names to check (exact match)
///
/// # Returns
/// * `Result<bool>` - True if model is present, false if not found, or error
pub fn is_model_present_in_ollama(
    client: &Client,
    server_url: &str,
    model_names: &[String],
) -> Result<bool> {
    let tags_url = format!("{}/api/tags", server_url.trim_end_matches('/'));

    debug!(
        "Checking Ollama server for model(s) {:?} at {}",
        model_names, tags_url
    );

    let response = client.get(&tags_url).send()?;

    if !response.status().is_success() {
        return Err(DownloaderError::HttpError(
            response.error_for_status().unwrap_err(),
        ));
    }

    let tags_response: Value = response.json()?;

    // Parse the JSON response to check for the model
    // Response format: {"models": [{"name": "model:tag", ...}]}
    if let Some(models) = tags_response.get("models").and_then(|m| m.as_array()) {
        for model_obj in models {
            if let Some(name) = model_obj.get("name").and_then(|n| n.as_str())
                && model_names.iter().any(|target| name == target)
            {
                debug!("Model {} found in Ollama server", name);
                return Ok(true);
            }
        }
        debug!("Model(s) {:?} not found in Ollama server", model_names);
        return Ok(false);
    }

    error!("Failed to parse Ollama tags response");
    Err(DownloaderError::Other(
        "Failed to parse Ollama tags response".to_string(),
    ))
}

pub fn expand_models_path(models_path: &str) -> Result<PathBuf> {
    if models_path.starts_with('~') {
        let home = env::var("HOME")
            .map_err(|_| DownloaderError::Other("HOME environment variable not set".to_string()))?;
        Ok(PathBuf::from(models_path.replacen('~', &home, 1)))
    } else {
        Ok(PathBuf::from(models_path))
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Ownership {
    pub uid: u32,
    pub gid: u32,
}

pub fn infer_models_dir_ownership(models_path: &str) -> Result<Option<Ownership>> {
    if !is_running_as_root() {
        return Ok(None);
    }

    let models_path = expand_models_path(models_path)?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        match fs::metadata(&models_path) {
            Ok(metadata) => Ok(Some(Ownership {
                uid: metadata.uid(),
                gid: metadata.gid(),
            })),
            Err(e) => {
                warn!(
                    "Failed to infer models directory ownership for {:?}: {}",
                    models_path, e
                );
                Ok(None)
            }
        }
    }
    #[cfg(not(unix))]
    {
        let _ = models_path;
        Ok(None)
    }
}

pub fn warn_if_models_path_requires_root(models_path: &str, is_download: bool) {
    if is_running_as_root() || !is_download {
        return;
    }

    let models_path = match expand_models_path(models_path) {
        Ok(path) => path,
        Err(e) => {
            warn!("Failed to expand models path {:?}: {}", models_path, e);
            return;
        }
    };

    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        let current_uid = unsafe { libc::geteuid() } as u32;
        match fs::metadata(&models_path) {
            Ok(metadata) => {
                if metadata.uid() != current_uid {
                    warn!(
                        "Models path {:?} is not owned by the current user. Run this command with superuser rights.",
                        models_path
                    );
                }
            }
            Err(e) => {
                warn!(
                    "Cannot verify ownership of models path {:?}: {}. Run this command with superuser rights.",
                    models_path, e
                );
            }
        }
    }
}

fn is_running_as_root() -> bool {
    #[cfg(unix)]
    unsafe {
        libc::geteuid() == 0
    }
    #[cfg(not(unix))]
    {
        false
    }
}

pub fn download_model_blob(
    client: &Client,
    url: &str,
    named_digest: &str,
    unnecessary_files: &mut HashSet<PathBuf>,
) -> Result<(PathBuf, String)> {
    // Check for interruption before starting download
    if crate::signal_handler::is_interrupted() {
        warn!("Download interrupted by user");
        return Err(DownloaderError::Other(
            "Download interrupted by user".to_string(),
        ));
    }
    if crate::signal_handler::confirm_pending_interrupt() {
        warn!("Download interrupted by user");
        return Err(DownloaderError::Other(
            "Download interrupted by user".to_string(),
        ));
    }

    let mut hasher = Sha256::new();
    let mut temp_file = NamedTempFile::new().map_err(DownloaderError::IoError)?;

    let temp_path = temp_file.path().to_path_buf();
    unnecessary_files.insert(temp_path.clone());

    let response = client.get(url).send()?;

    if !response.status().is_success() {
        return Err(DownloaderError::HttpError(
            response.error_for_status().unwrap_err(),
        ));
    }

    let total_size = response.content_length().unwrap_or(0);

    let pb = ProgressBar::new(total_size);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{msg} [{bar:40.cyan/blue}] {bytes}/{total_bytes} ({eta})")
            .unwrap()
            .progress_chars("#>-"),
    );
    pb.set_message(format!("Downloading BLOB {}", &named_digest));

    struct ProgressGuard;
    impl Drop for ProgressGuard {
        fn drop(&mut self) {
            crate::signal_handler::set_progress_active(false);
        }
    }

    crate::signal_handler::set_progress_active(true);
    let _progress_guard = ProgressGuard;

    // Stream chunks from the response
    let mut response_reader = response;
    let mut buffer = [0u8; 8192];

    loop {
        // Check for interruption signal during download
        if crate::signal_handler::is_interrupted() {
            warn!("Download interrupted by user while downloading BLOB");
            pb.abandon();
            return Err(DownloaderError::Other(
                "Download interrupted by user".to_string(),
            ));
        }

        if crate::signal_handler::interrupt_requested() {
            let should_exit = pb.suspend(crate::signal_handler::confirm_pending_interrupt);
            if should_exit {
                warn!("Download interrupted by user while downloading BLOB");
                pb.abandon();
                return Err(DownloaderError::Other(
                    "Download interrupted by user".to_string(),
                ));
            }
        }

        let bytes_read = response_reader.read(&mut buffer)?;
        if bytes_read == 0 {
            break;
        }

        let chunk = &buffer[..bytes_read];
        hasher.update(chunk);
        temp_file.write_all(chunk)?;
        pb.inc(bytes_read as u64);
    }

    pb.finish_with_message("Downloaded");

    let computed_digest = format!("{:x}", hasher.finalize());
    debug!("Downloaded {} to {:?}", url, temp_path);
    debug!("Computed SHA256 digest: {}", computed_digest);

    // Persist the temp file
    let persisted_path = temp_file.into_temp_path();
    let final_path = persisted_path
        .keep()
        .map_err(|e| DownloaderError::Other(format!("Failed to persist temp file: {}", e)))?;

    Ok((final_path, computed_digest))
}

pub fn save_blob(
    models_path: &str,
    source: &Path,
    named_digest: &str,
    computed_digest: &str,
    models_dir_ownership: Option<Ownership>,
    unnecessary_files: &mut HashSet<PathBuf>,
) -> Result<PathBuf> {
    // Verify digest matches (skip "sha256:" prefix)
    let expected_digest = &named_digest[7..];
    if computed_digest != expected_digest {
        error!(
            "Digest mismatch: expected {}, got {}",
            expected_digest, computed_digest
        );
        return Err(DownloaderError::Other(format!(
            "Digest mismatch for {}",
            named_digest
        )));
    }

    info!("BLOB {} digest verified successfully.", named_digest);

    let models_path = expand_models_path(models_path)?;
    let blobs_dir = models_path.join("blobs");

    if !blobs_dir.exists() {
        return Err(DownloaderError::Other(format!(
            "BLOBS directory {:?} does not exist",
            blobs_dir
        )));
    }

    if !blobs_dir.is_dir() {
        return Err(DownloaderError::Other(format!(
            "BLOBS path {:?} is not a directory",
            blobs_dir
        )));
    }

    let target_file = blobs_dir.join(named_digest.replace(':', "-"));
    fs::copy(source, &target_file)?;

    if let Some(ownership) = models_dir_ownership {
        ensure_ownership(&target_file, ownership);
        ensure_ownership(&blobs_dir, ownership);
    }

    // Remove source from unnecessary files and add target
    unnecessary_files.remove(&source.to_path_buf());
    unnecessary_files.insert(target_file.clone());

    info!("Moved {:?} to {:?}", source, target_file);

    Ok(target_file)
}

pub fn save_manifest(
    data: &str,
    models_root: &Path,
    manifests_dir: &Path,
    tag: &str,
    models_dir_ownership: Option<Ownership>,
    chown_dirs: &[&Path],
    unnecessary_files: &mut HashSet<PathBuf>,
) -> Result<PathBuf> {
    if !manifests_dir.exists() {
        warn!(
            "Manifests path {:?} does not exist. Creating it.",
            manifests_dir
        );
        fs::create_dir_all(manifests_dir)?;
        unnecessary_files.insert(manifests_dir.to_path_buf());
    }

    let target_file = manifests_dir.join(tag);
    fs::write(&target_file, data)?;

    if let Some(ownership) = models_dir_ownership {
        ensure_ownership_for_dir_tree(models_root, manifests_dir, ownership);
        ensure_ownership(&target_file, ownership);
        for dir in chown_dirs {
            ensure_ownership(dir, ownership);
        }
    }
    info!("Saved manifest to {:?}", target_file);

    unnecessary_files.insert(target_file.clone());

    Ok(target_file)
}

pub fn cleanup_unnecessary_files(unnecessary_files: &mut HashSet<PathBuf>) {
    let files_to_remove: Vec<PathBuf> = unnecessary_files.iter().cloned().collect();

    for file_path in files_to_remove {
        if file_path.is_file() {
            if let Err(e) = fs::remove_file(&file_path) {
                warn!("Failed to remove unnecessary file {:?}: {}", file_path, e);
            } else {
                info!("Removed unnecessary file: {:?}", file_path);
                unnecessary_files.remove(&file_path);
            }
        } else if file_path.is_dir() {
            if let Err(e) = fs::remove_dir(&file_path) {
                debug!(
                    "Failed to remove unnecessary directory {:?}: {}",
                    file_path, e
                );
            } else {
                info!("Removed unnecessary directory: {:?}", file_path);
                unnecessary_files.remove(&file_path);
            }
        }
    }
}

fn ensure_ownership(path: &Path, ownership: Ownership) {
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        match fs::metadata(path) {
            Ok(metadata) => {
                if metadata.uid() != ownership.uid || metadata.gid() != ownership.gid {
                    apply_ownership(path, ownership);
                }
            }
            Err(e) => warn!("Failed to read ownership for {:?}: {}", path, e),
        }
    }
    #[cfg(not(unix))]
    {
        let _ = (path, ownership);
    }
}

fn ensure_ownership_for_dir_tree(models_root: &Path, dir: &Path, ownership: Ownership) {
    if !dir.starts_with(models_root) {
        return;
    }

    let mut current = dir;
    loop {
        ensure_ownership(current, ownership);
        if current == models_root {
            break;
        }

        match current.parent() {
            Some(parent) => current = parent,
            None => break,
        }
    }
}

fn apply_ownership(path: &Path, ownership: Ownership) {
    #[cfg(unix)]
    {
        let spec = format!("{}:{}", ownership.uid, ownership.gid);
        match Command::new("chown").arg(&spec).arg(path).status() {
            Ok(status) if status.success() => {}
            Ok(status) => warn!("Failed to chown {:?}: exit status {}", path, status),
            Err(e) => warn!("Failed to chown {:?}: {}", path, e),
        }
    }
    #[cfg(not(unix))]
    {
        let _ = (path, ownership);
    }
}
