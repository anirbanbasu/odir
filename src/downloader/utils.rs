//! Utility functions for the Ollama Downloader in Rust (ODIR),
//! including model presence checks, downloading blobs, saving manifests,
//! and cleaning up temporary files.
use crate::downloader::model_downloader::{DownloaderError, Result};
use indicatif::{ProgressBar, ProgressStyle};
use log::{debug, error, info, warn};
use reqwest::StatusCode;
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
    chunk_size_bytes: u64,
    blobs_dir: Option<&Path>,
    models_dir_ownership: Option<Ownership>,
) -> Result<(PathBuf, String)> {
    ensure_download_not_interrupted()?;

    if let Some(result) = try_download_model_blob_chunked(ChunkedBlobProbe {
        client,
        url,
        named_digest,
        unnecessary_files,
        chunk_size_bytes,
        blobs_dir,
        models_dir_ownership,
    }) {
        return result;
    }

    download_model_blob_single_stream(client, url, named_digest, unnecessary_files)
}

fn ensure_download_not_interrupted() -> Result<()> {
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

    Ok(())
}

struct ChunkedBlobProbe<'a> {
    client: &'a Client,
    url: &'a str,
    named_digest: &'a str,
    unnecessary_files: &'a mut HashSet<PathBuf>,
    chunk_size_bytes: u64,
    blobs_dir: Option<&'a Path>,
    models_dir_ownership: Option<Ownership>,
}

fn try_download_model_blob_chunked(
    request: ChunkedBlobProbe<'_>,
) -> Option<Result<(PathBuf, String)>> {
    let ChunkedBlobProbe {
        client,
        url,
        named_digest,
        unnecessary_files,
        chunk_size_bytes,
        blobs_dir,
        models_dir_ownership,
    } = request;

    if chunk_size_bytes == 0 {
        return None;
    }

    let blobs_dir = blobs_dir?;

    // Detect byte-range support robustly: use HEAD hints when available,
    // then fall back to an active range probe when needed.
    let mut total_size: u64 = 0;
    let mut range_supported = false;

    match client.head(url).send() {
        Ok(head_resp) if head_resp.status().is_success() => {
            total_size = head_resp.content_length().unwrap_or(0);
            range_supported = head_resp
                .headers()
                .get("accept-ranges")
                .and_then(|v| v.to_str().ok())
                .map(|v| v.eq_ignore_ascii_case("bytes"))
                .unwrap_or(false);

            if range_supported {
                debug!("HEAD indicates byte-range support for {}", named_digest);
            }
        }
        Ok(head_resp) => {
            debug!(
                "HEAD returned non-success ({}) for {}; trying active range probe",
                head_resp.status(),
                named_digest
            );
        }
        Err(e) => {
            debug!(
                "HEAD failed for {}: {}; trying active range probe",
                named_digest, e
            );
        }
    }

    // Run active probe when support is unknown, or when HEAD omitted size.
    if !range_supported || total_size == 0 {
        let (probe_supported, probe_total_size) =
            probe_byte_range_support(client, url, named_digest);
        if !range_supported {
            range_supported = probe_supported;
        }
        if total_size == 0 {
            total_size = probe_total_size.unwrap_or(0);
        }
    }

    if range_supported && total_size > chunk_size_bytes {
        debug!(
            "Using chunked download for {} ({} bytes, {} byte chunks)",
            named_digest, total_size, chunk_size_bytes
        );
        return Some(download_model_blob_chunked(ChunkedBlobDownload {
            client,
            url,
            named_digest,
            unnecessary_files,
            chunk_size: chunk_size_bytes,
            blobs_dir,
            total_size,
            models_dir_ownership,
        }));
    }

    if range_supported {
        if total_size == 0 {
            warn!(
                "Byte-range probe succeeded for {}, but total size is unknown; \
                         falling back to single-stream download",
                named_digest
            );
        }
    } else {
        warn!(
            "Server does not support byte-range probe for {}; \
                     falling back to single-stream download",
            named_digest
        );
    }

    None
}

fn download_model_blob_single_stream(
    client: &Client,
    url: &str,
    named_digest: &str,
    unnecessary_files: &mut HashSet<PathBuf>,
) -> Result<(PathBuf, String)> {
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
    pb.set_message(format!("Downloading BLOB {}", named_digest));

    struct ProgressGuard;
    impl Drop for ProgressGuard {
        fn drop(&mut self) {
            crate::signal_handler::set_progress_active(false);
        }
    }

    crate::signal_handler::set_progress_active(true);
    let _progress_guard = ProgressGuard;

    let mut response_reader = response;
    let mut buffer = [0u8; 8192];

    loop {
        check_single_stream_interrupt(&pb)?;

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

    let persisted_path = temp_file.into_temp_path();
    let final_path = persisted_path
        .keep()
        .map_err(|e| DownloaderError::Other(format!("Failed to persist temp file: {}", e)))?;

    Ok((final_path, computed_digest))
}

fn check_single_stream_interrupt(pb: &ProgressBar) -> Result<()> {
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

    Ok(())
}

/// Parse the complete length from a Content-Range value.
///
/// Supported forms:
/// - `bytes 0-0/12345` -> Some(12345)
/// - `bytes */12345`   -> Some(12345)
/// - `bytes 0-0/*`     -> None
fn parse_total_size_from_content_range(content_range: &str) -> Option<u64> {
    let (_, total) = content_range.split_once('/')?;
    let trimmed = total.trim();
    if trimmed == "*" {
        None
    } else {
        trimmed.parse::<u64>().ok()
    }
}

/// Actively probe byte-range support by requesting a one-byte range.
///
/// Returns `(supported, total_size_if_known)`.
fn probe_byte_range_support(client: &Client, url: &str, named_digest: &str) -> (bool, Option<u64>) {
    match client.get(url).header("Range", "bytes=0-0").send() {
        Ok(response) if response.status() == StatusCode::PARTIAL_CONTENT => {
            let total_size = response
                .headers()
                .get("content-range")
                .and_then(|v| v.to_str().ok())
                .and_then(parse_total_size_from_content_range);

            debug!(
                "Active byte-range probe succeeded for {} (total size: {:?})",
                named_digest, total_size
            );
            (true, total_size)
        }
        Ok(response) if response.status() == StatusCode::RANGE_NOT_SATISFIABLE => {
            let total_size = response
                .headers()
                .get("content-range")
                .and_then(|v| v.to_str().ok())
                .and_then(parse_total_size_from_content_range);

            debug!(
                "Active byte-range probe got 416 for {} (total size: {:?})",
                named_digest, total_size
            );
            (true, total_size)
        }
        Ok(response) => {
            debug!(
                "Active byte-range probe returned {} for {}; treating as unsupported",
                response.status(),
                named_digest
            );
            (false, None)
        }
        Err(e) => {
            debug!("Active byte-range probe failed for {}: {}", named_digest, e);
            (false, None)
        }
    }
}

// ─── Chunked / resumable download helpers ────────────────────────────────────

/// Returns the directory that holds in-progress parts for a specific blob.
/// Layout: `{blobs_dir}/.parts/{named_digest_as_fs_name}/`
fn get_parts_dir(blobs_dir: &Path, named_digest: &str) -> PathBuf {
    blobs_dir
        .join(".parts")
        .join(named_digest.replace(':', "-"))
}

/// Zero-padded part filename, e.g. `part_00000003`.
fn part_file_name(index: u64) -> String {
    format!("part_{:08}", index)
}

/// Expected byte count for part `part_idx` out of `num_parts` total.
fn expected_part_size(part_idx: u64, num_parts: u64, total_size: u64, chunk_size: u64) -> u64 {
    if part_idx == num_parts - 1 {
        let rem = total_size % chunk_size;
        if rem == 0 { chunk_size } else { rem }
    } else {
        chunk_size
    }
}

/// Returns `true` when the part file exists and has exactly the expected byte count.
fn is_part_complete(part_path: &Path, expected_size: u64) -> bool {
    fs::metadata(part_path)
        .map(|m| m.len() == expected_size)
        .unwrap_or(false)
}

struct ChunkedBlobDownload<'a> {
    client: &'a Client,
    url: &'a str,
    named_digest: &'a str,
    unnecessary_files: &'a mut HashSet<PathBuf>,
    chunk_size: u64,
    blobs_dir: &'a Path,
    total_size: u64,
    models_dir_ownership: Option<Ownership>,
}

/// Download a blob by fetching it in sequential byte-range parts and assembling
/// them into a single file.
///
/// * **User abort without chunk removal** – parts directory is retained; subsequent call can resume from where it left off.
/// * **User abort with chunk removal** – the entire parts directory is wiped; callers get a clean slate.
/// * **Network error** – only the incomplete current-part file is removed; already
///   downloaded parts are kept so a subsequent call can resume from where it left off.
fn download_model_blob_chunked(request: ChunkedBlobDownload<'_>) -> Result<(PathBuf, String)> {
    let ChunkedBlobDownload {
        client,
        url,
        named_digest,
        unnecessary_files,
        chunk_size,
        blobs_dir,
        total_size,
        models_dir_ownership,
    } = request;

    let num_parts = total_size.div_ceil(chunk_size);
    let parts_dir = get_parts_dir(blobs_dir, named_digest);

    if !parts_dir.exists() {
        fs::create_dir_all(&parts_dir)?;
        if let Some(ownership) = models_dir_ownership {
            ensure_ownership_for_dir_tree(blobs_dir, &parts_dir, ownership);
        }
        info!("Created parts directory: {:?}", parts_dir);
    }

    // Determine which parts are already complete.
    let mut already_downloaded: u64 = 0;
    let mut completed_parts: u64 = 0;
    let mut missing_parts: Vec<u64> = Vec::new();
    for part_idx in 0..num_parts {
        let part_path = parts_dir.join(part_file_name(part_idx));
        let expected = expected_part_size(part_idx, num_parts, total_size, chunk_size);
        if is_part_complete(&part_path, expected) {
            already_downloaded += expected;
            completed_parts += 1;
        } else {
            missing_parts.push(part_idx);
        }
    }

    if missing_parts.is_empty() {
        info!(
            "All {} parts of {} are already on disk; assembling",
            num_parts, named_digest
        );
    } else {
        info!(
            "Downloading {} missing parts for {} ({} of {} parts already complete, {} bytes already on disk)",
            missing_parts.len(),
            named_digest,
            completed_parts,
            num_parts,
            already_downloaded
        );
    }

    // Progress bar covering the full file, pre-advanced for already-downloaded bytes.
    let pb = ProgressBar::new(total_size);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{msg} [{bar:40.cyan/blue}] {bytes}/{total_bytes} ({eta})")
            .unwrap()
            .progress_chars("#>-"),
    );
    pb.set_message(format!(
        "Downloading BLOB {} part {}/{}",
        named_digest, completed_parts, num_parts
    ));
    pb.set_position(already_downloaded);

    struct ProgressGuard;
    impl Drop for ProgressGuard {
        fn drop(&mut self) {
            crate::signal_handler::set_progress_active(false);
        }
    }
    crate::signal_handler::set_progress_active(true);
    let _progress_guard = ProgressGuard;

    let mut buffer = [0u8; 8192];

    for &part_idx in &missing_parts {
        pb.set_message(format!(
            "Downloading BLOB {} part {}/{}",
            named_digest,
            completed_parts + 1,
            num_parts
        ));

        let part_path = parts_dir.join(part_file_name(part_idx));
        let expected_size = expected_part_size(part_idx, num_parts, total_size, chunk_size);
        let range_start = part_idx * chunk_size;
        let range_end = range_start + expected_size - 1;

        // Interrupt check before fetching this part.
        if crate::signal_handler::is_interrupted() {
            warn!("Download interrupted by user before part {}", part_idx);
            pb.abandon();
            if !crate::signal_handler::keep_partial_downloads() {
                let _ = fs::remove_dir_all(&parts_dir);
            }
            return Err(DownloaderError::Other(
                "Download interrupted by user".to_string(),
            ));
        }
        if crate::signal_handler::interrupt_requested() {
            let should_exit = pb.suspend(|| {
                crate::signal_handler::confirm_pending_interrupt_for_chunked(completed_parts)
            });
            if should_exit {
                warn!("Download interrupted by user before part {}", part_idx);
                pb.abandon();
                if !crate::signal_handler::keep_partial_downloads() {
                    let _ = fs::remove_dir_all(&parts_dir);
                }
                return Err(DownloaderError::Other(
                    "Download interrupted by user".to_string(),
                ));
            }
        }

        let range_header = format!("bytes={}-{}", range_start, range_end);
        let response = match client.get(url).header("Range", &range_header).send() {
            Ok(r) => r,
            Err(e) => {
                pb.abandon();
                return Err(DownloaderError::HttpError(e));
            }
        };

        if response.status() != StatusCode::PARTIAL_CONTENT {
            pb.abandon();
            let status = response.status();
            let _ = fs::remove_dir_all(&parts_dir);
            return Err(DownloaderError::Other(format!(
                "Expected 206 Partial Content for range request on {}, got {}",
                named_digest, status
            )));
        }

        let mut part_file = match fs::File::create(&part_path) {
            Ok(f) => f,
            Err(e) => {
                pb.abandon();
                return Err(DownloaderError::IoError(e));
            }
        };
        if let Some(ownership) = models_dir_ownership {
            ensure_ownership(&part_path, ownership);
        }

        let mut response_reader = response;
        let mut bytes_written: u64 = 0;

        loop {
            // Interrupt check inside the read loop.
            if crate::signal_handler::is_interrupted() {
                warn!(
                    "Download interrupted by user while downloading part {}",
                    part_idx
                );
                pb.abandon();
                drop(part_file);
                let _ = fs::remove_file(&part_path);
                if !crate::signal_handler::keep_partial_downloads() {
                    let _ = fs::remove_dir_all(&parts_dir);
                }
                return Err(DownloaderError::Other(
                    "Download interrupted by user".to_string(),
                ));
            }
            if crate::signal_handler::interrupt_requested() {
                let should_exit = pb.suspend(|| {
                    crate::signal_handler::confirm_pending_interrupt_for_chunked(completed_parts)
                });
                if should_exit {
                    warn!(
                        "Download interrupted by user while downloading part {}",
                        part_idx
                    );
                    pb.abandon();
                    drop(part_file);
                    let _ = fs::remove_file(&part_path);
                    if !crate::signal_handler::keep_partial_downloads() {
                        let _ = fs::remove_dir_all(&parts_dir);
                    }
                    return Err(DownloaderError::Other(
                        "Download interrupted by user".to_string(),
                    ));
                }
            }

            let bytes_read = match response_reader.read(&mut buffer) {
                Ok(0) => break,
                Ok(n) => n,
                Err(e) => {
                    // Network error: keep completed parts; remove the incomplete one.
                    pb.abandon();
                    drop(part_file);
                    let _ = fs::remove_file(&part_path);
                    return Err(DownloaderError::IoError(e));
                }
            };

            let chunk = &buffer[..bytes_read];
            if let Err(e) = part_file.write_all(chunk) {
                pb.abandon();
                drop(part_file);
                let _ = fs::remove_file(&part_path);
                return Err(DownloaderError::IoError(e));
            }
            bytes_written += bytes_read as u64;
            pb.inc(bytes_read as u64);
        }

        if bytes_written != expected_size {
            warn!(
                "Part {} incomplete: expected {} bytes, got {}",
                part_idx, expected_size, bytes_written
            );
            let _ = fs::remove_file(&part_path);
            pb.abandon();
            return Err(DownloaderError::Other(format!(
                "Part {} of {} incomplete ({}/{} bytes)",
                part_idx, named_digest, bytes_written, expected_size
            )));
        }

        debug!(
            "Downloaded part {}/{} for {}",
            part_idx + 1,
            num_parts,
            named_digest
        );

        completed_parts += 1;
        pb.set_message(format!(
            "Downloading BLOB {} part {}/{}",
            named_digest, completed_parts, num_parts
        ));
    }

    pb.finish_with_message("Downloaded (assembling parts)");

    // Assemble all parts into a NamedTempFile while computing SHA-256.
    let mut hasher = Sha256::new();
    let mut temp_file = NamedTempFile::new()?;
    let temp_path = temp_file.path().to_path_buf();
    unnecessary_files.insert(temp_path.clone());

    info!("Assembling {} parts for {}", num_parts, named_digest);
    for part_idx in 0..num_parts {
        let part_path = parts_dir.join(part_file_name(part_idx));
        let mut part_file = fs::File::open(&part_path)?;
        let mut read_buf = [0u8; 8192];
        loop {
            let n = part_file.read(&mut read_buf)?;
            if n == 0 {
                break;
            }
            let chunk = &read_buf[..n];
            hasher.update(chunk);
            temp_file.write_all(chunk)?;
        }
    }

    let computed_digest = format!("{:x}", hasher.finalize());
    debug!(
        "Assembled {} into {:?}, digest: {}",
        named_digest, temp_path, computed_digest
    );

    // Parts are no longer needed.
    if let Err(e) = fs::remove_dir_all(&parts_dir) {
        warn!("Failed to remove parts directory {:?}: {}", parts_dir, e);
    } else {
        debug!("Removed parts directory {:?}", parts_dir);
    }

    let persisted_path = temp_file.into_temp_path();
    let final_path = persisted_path
        .keep()
        .map_err(|e| DownloaderError::Other(format!("Failed to persist temp file: {}", e)))?;

    Ok((final_path, computed_digest))
}

// ─── Blob / manifest persistence ─────────────────────────────────────────────

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
