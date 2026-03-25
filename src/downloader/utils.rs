//! Utility functions for the Ollama Downloader in Rust (ODIR),
//! including model presence checks, downloading blobs, saving manifests,
//! and cleaning up temporary files.
use crate::downloader::model_downloader::{DownloaderError, Result};
use indicatif::{ProgressBar, ProgressStyle};
use log::{debug, error, info, warn};
use reqwest::StatusCode;
use reqwest::blocking::Client;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::{HashSet, VecDeque};
use std::env;
use std::fs;
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, mpsc};
use std::thread;
use std::time::Duration;
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

pub struct BlobDownloadRequest<'a> {
    pub client: &'a Client,
    pub url: &'a str,
    pub named_digest: &'a str,
    pub unnecessary_files: &'a mut HashSet<PathBuf>,
    pub chunk_size_bytes: u64,
    pub blobs_dir: Option<&'a Path>,
    pub models_dir_ownership: Option<Ownership>,
    pub download_chunks_in_parallel: bool,
}

pub fn download_model_blob(request: BlobDownloadRequest<'_>) -> Result<(PathBuf, String)> {
    let BlobDownloadRequest {
        client,
        url,
        named_digest,
        unnecessary_files,
        chunk_size_bytes,
        blobs_dir,
        models_dir_ownership,
        download_chunks_in_parallel,
    } = request;

    ensure_download_not_interrupted()?;

    if let Some(result) = try_download_model_blob_chunked(ChunkedBlobProbe {
        client,
        url,
        named_digest,
        unnecessary_files,
        chunk_size_bytes,
        blobs_dir,
        models_dir_ownership,
        download_chunks_in_parallel,
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
    download_chunks_in_parallel: bool,
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
        download_chunks_in_parallel,
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
            download_chunks_in_parallel,
        }));
    }

    log_single_stream_fallback(range_supported, total_size, named_digest);

    None
}

fn log_single_stream_fallback(range_supported: bool, total_size: u64, named_digest: &str) {
    if !range_supported {
        warn!(
            "Server does not support byte-range probe for {}; \
                     falling back to single-stream download",
            named_digest
        );
        return;
    }

    if total_size == 0 {
        warn!(
            "Byte-range probe succeeded for {}, but total size is unknown; \
                     falling back to single-stream download",
            named_digest
        );
    }
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
    let mut buffer = vec![0u8; STREAM_IO_BUFFER_SIZE];

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

const CHUNKED_STATE_VERSION: u32 = 1;
const MAX_PARALLEL_CHUNK_WORKERS: usize = 4;
const CHUNKED_REFRESH_IDLE_MS: u64 = 100;
const CHUNKED_REFRESH_ACTIVE_MS: u64 = 30;
const STREAM_IO_BUFFER_SIZE: usize = 4 * 1024 * 1024;
const CHUNK_IO_BUFFER_SIZE: usize = 4 * 1024 * 1024;
const HASH_IO_BUFFER_SIZE: usize = 4 * 1024 * 1024;

fn digest_fs_name(named_digest: &str) -> String {
    named_digest.replace(':', "-")
}

#[derive(Debug)]
struct ChunkedWorkspace {
    data_file: PathBuf,
    state_file: PathBuf,
}

#[derive(Debug, Serialize, Deserialize)]
struct ChunkedState {
    version: u32,
    total_size: u64,
    chunk_size: u64,
    completed: Vec<bool>,
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

struct ChunkedBlobDownload<'a> {
    client: &'a Client,
    url: &'a str,
    named_digest: &'a str,
    unnecessary_files: &'a mut HashSet<PathBuf>,
    chunk_size: u64,
    blobs_dir: &'a Path,
    total_size: u64,
    models_dir_ownership: Option<Ownership>,
    download_chunks_in_parallel: bool,
}

/// Download a blob by fetching it in sequential byte-range parts and assembling
/// them directly into a single preallocated file.
///
/// * **User abort without chunk removal** – parts directory is retained; subsequent call can resume from where it left off.
/// * **User abort with chunk removal** – the entire parts directory is wiped; callers get a clean slate.
/// * **Network error** – completed chunk map is retained and incomplete chunks are re-fetched on next run.
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
        download_chunks_in_parallel,
    } = request;

    let num_parts = total_size.div_ceil(chunk_size);
    let workspace = ensure_chunked_workspace(blobs_dir, named_digest, models_dir_ownership)?;

    let mut state = load_or_initialize_chunked_state(
        &workspace,
        total_size,
        chunk_size,
        num_parts,
        models_dir_ownership,
    )?;

    let scan = scan_chunk_state(&state, num_parts, total_size, chunk_size);
    log_scan_summary(named_digest, num_parts, &scan);

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
        named_digest,
        (scan.completed_parts + 1).min(num_parts),
        num_parts
    ));
    pb.set_position(scan.already_downloaded);

    struct ProgressGuard;
    impl Drop for ProgressGuard {
        fn drop(&mut self) {
            crate::signal_handler::set_progress_active(false);
        }
    }
    crate::signal_handler::set_progress_active(true);
    let _progress_guard = ProgressGuard;

    let user_aborted = download_missing_chunks_parallel(DownloadChunksParallelRequest {
        client,
        url,
        named_digest,
        workspace: &workspace,
        missing_parts: &scan.missing_parts,
        state: &mut state,
        num_parts,
        chunk_size,
        total_size,
        already_downloaded: scan.already_downloaded,
        completed_parts: scan.completed_parts,
        pb: &pb,
        models_dir_ownership,
        download_chunks_in_parallel,
    })?;

    if user_aborted {
        pb.abandon();
        if !crate::signal_handler::keep_partial_downloads() {
            cleanup_chunk_workspace(&workspace);
        }
        return Err(DownloaderError::Other(
            "Download interrupted by user".to_string(),
        ));
    }

    pb.finish_with_message("Downloaded");

    let computed_digest = compute_file_sha256(&workspace.data_file, named_digest)?;
    debug!(
        "Downloaded {} to {:?}, digest: {}",
        named_digest, workspace.data_file, computed_digest
    );

    // State file is no longer needed after success.
    if let Err(e) = fs::remove_file(&workspace.state_file)
        && e.kind() != std::io::ErrorKind::NotFound
    {
        warn!(
            "Failed to remove chunk state file {:?}: {}",
            workspace.state_file, e
        );
    }

    unnecessary_files.insert(workspace.data_file.clone());

    Ok((workspace.data_file, computed_digest))
}

struct ChunkedPartScan {
    already_downloaded: u64,
    completed_parts: u64,
    missing_parts: Vec<u64>,
}

fn ensure_chunked_workspace(
    blobs_dir: &Path,
    named_digest: &str,
    models_dir_ownership: Option<Ownership>,
) -> Result<ChunkedWorkspace> {
    let parts_root = blobs_dir.join(".parts");
    if !parts_root.exists() {
        fs::create_dir_all(&parts_root)?;
        info!("Created parts directory: {:?}", parts_root);
    }
    if let Some(ownership) = models_dir_ownership {
        ensure_ownership_for_dir_tree(blobs_dir, &parts_root, ownership);
    }

    let stem = digest_fs_name(named_digest);
    let data_file = parts_root.join(format!("{}.bin", stem));
    let state_file = parts_root.join(format!("{}.state.json", stem));

    Ok(ChunkedWorkspace {
        data_file,
        state_file,
    })
}

fn load_or_initialize_chunked_state(
    workspace: &ChunkedWorkspace,
    total_size: u64,
    chunk_size: u64,
    num_parts: u64,
    models_dir_ownership: Option<Ownership>,
) -> Result<ChunkedState> {
    let num_parts_usize = usize::try_from(num_parts)
        .map_err(|_| DownloaderError::Other("Blob has too many chunks".to_string()))?;

    let mut reusable_state: Option<ChunkedState> = None;
    if let Ok(raw) = fs::read_to_string(&workspace.state_file)
        && let Ok(state) = serde_json::from_str::<ChunkedState>(&raw)
        && state.version == CHUNKED_STATE_VERSION
        && state.total_size == total_size
        && state.chunk_size == chunk_size
        && state.completed.len() == num_parts_usize
        && fs::metadata(&workspace.data_file)
            .map(|m| m.len() == total_size)
            .unwrap_or(false)
    {
        reusable_state = Some(state);
    }

    if let Some(state) = reusable_state {
        return Ok(state);
    }

    let file = fs::OpenOptions::new()
        .create(true)
        .truncate(true)
        .read(true)
        .write(true)
        .open(&workspace.data_file)?;
    file.set_len(total_size)?;

    if let Some(ownership) = models_dir_ownership {
        ensure_ownership(&workspace.data_file, ownership);
    }

    let state = ChunkedState {
        version: CHUNKED_STATE_VERSION,
        total_size,
        chunk_size,
        completed: vec![false; num_parts_usize],
    };
    persist_chunked_state(&workspace.state_file, &state, models_dir_ownership)?;
    Ok(state)
}

fn persist_chunked_state(
    state_file: &Path,
    state: &ChunkedState,
    models_dir_ownership: Option<Ownership>,
) -> Result<()> {
    let temp_path = state_file.with_extension("state.tmp");
    let payload = serde_json::to_vec_pretty(state)
        .map_err(|e| DownloaderError::Other(format!("Failed to serialize chunk state: {}", e)))?;
    fs::write(&temp_path, payload)?;
    fs::rename(&temp_path, state_file)?;
    if let Some(ownership) = models_dir_ownership {
        ensure_ownership(state_file, ownership);
    }
    Ok(())
}

fn scan_chunk_state(
    state: &ChunkedState,
    num_parts: u64,
    total_size: u64,
    chunk_size: u64,
) -> ChunkedPartScan {
    let mut already_downloaded: u64 = 0;
    let mut completed_parts: u64 = 0;
    let mut missing_parts: Vec<u64> = Vec::new();

    for part_idx in 0..num_parts {
        let idx = part_idx as usize;
        if state.completed[idx] {
            already_downloaded += expected_part_size(part_idx, num_parts, total_size, chunk_size);
            completed_parts += 1;
        } else {
            missing_parts.push(part_idx);
        }
    }

    ChunkedPartScan {
        already_downloaded,
        completed_parts,
        missing_parts,
    }
}

fn log_scan_summary(named_digest: &str, num_parts: u64, scan: &ChunkedPartScan) {
    if scan.missing_parts.is_empty() {
        info!(
            "All {} parts of {} are already downloaded; verifying digest",
            num_parts, named_digest
        );
        return;
    }

    info!(
        "Downloading {} missing parts for {} ({} of {} parts already complete, {} bytes already on disk)",
        scan.missing_parts.len(),
        named_digest,
        scan.completed_parts,
        num_parts,
        scan.already_downloaded
    );
}

struct DownloadChunksParallelRequest<'a> {
    client: &'a Client,
    url: &'a str,
    named_digest: &'a str,
    workspace: &'a ChunkedWorkspace,
    missing_parts: &'a [u64],
    state: &'a mut ChunkedState,
    num_parts: u64,
    chunk_size: u64,
    total_size: u64,
    already_downloaded: u64,
    completed_parts: u64,
    pb: &'a ProgressBar,
    models_dir_ownership: Option<Ownership>,
    download_chunks_in_parallel: bool,
}

/// Checks signal-handler interrupt flags and, if warranted, prompts the user.
/// Stores `true` into `cancel` and returns `true` when the coordinator should abort.
fn check_chunk_interrupt(cancel: &AtomicBool, pb: &ProgressBar, parts_done: &AtomicU64) -> bool {
    if crate::signal_handler::is_interrupted() {
        cancel.store(true, Ordering::Release);
        return true;
    }
    if crate::signal_handler::interrupt_requested() {
        let should_exit = pb.suspend(|| {
            crate::signal_handler::confirm_pending_interrupt_for_chunked(
                parts_done.load(Ordering::Acquire),
            )
        });
        if should_exit {
            cancel.store(true, Ordering::Release);
            return true;
        }
    }
    false
}

/// Marks `part_idx` as completed, persists sidecar state, and increments `finished_missing`.
/// Returns `true` when the coordinator loop should break (state persistence failed).
fn on_chunk_completed(
    part_idx: u64,
    state: &mut ChunkedState,
    workspace: &ChunkedWorkspace,
    models_dir_ownership: Option<Ownership>,
    cancel: &AtomicBool,
    first_error: &Mutex<Option<DownloaderError>>,
    finished_missing: &mut usize,
) -> bool {
    *finished_missing += 1;
    let idx = part_idx as usize;
    if idx < state.completed.len() {
        state.completed[idx] = true;
        if let Err(e) = persist_chunked_state(&workspace.state_file, state, models_dir_ownership) {
            *first_error.lock().unwrap_or_else(|pe| pe.into_inner()) = Some(e);
            cancel.store(true, Ordering::Release);
            return true;
        }
    }
    false
}

/// Updates the adaptive refresh interval and repaints the progress bar.
fn update_chunk_progress(
    pb: &ProgressBar,
    bytes_done: &AtomicU64,
    parts_done: &AtomicU64,
    named_digest: &str,
    num_parts: u64,
    last_bytes_done: &mut u64,
    refresh_interval: &mut Duration,
) {
    let current = bytes_done.load(Ordering::Acquire);
    let delta = current.saturating_sub(*last_bytes_done);
    *refresh_interval = if delta > 0 {
        Duration::from_millis(CHUNKED_REFRESH_ACTIVE_MS)
    } else {
        Duration::from_millis(CHUNKED_REFRESH_IDLE_MS)
    };
    *last_bytes_done = current;
    pb.set_position(current);
    pb.set_message(format!(
        "Downloading BLOB {} part {}/{}",
        named_digest,
        (parts_done.load(Ordering::Acquire) + 1).min(num_parts),
        num_parts
    ));
}

struct CoordinatorLoopParams<'a> {
    done_rx: &'a mpsc::Receiver<u64>,
    cancel: &'a AtomicBool,
    first_error: &'a Mutex<Option<DownloaderError>>,
    bytes_done: &'a AtomicU64,
    parts_done: &'a AtomicU64,
    pb: &'a ProgressBar,
    named_digest: &'a str,
    num_parts: u64,
    models_dir_ownership: Option<Ownership>,
    state: &'a mut ChunkedState,
    workspace: &'a ChunkedWorkspace,
    missing_parts_count: usize,
    already_downloaded: u64,
}

/// Runs the coordinator loop, polling the done channel, persisting state,
/// and updating progress until all parts finish or the user aborts.
/// Returns `(user_aborted, finished_missing)`.
fn run_coordinator_loop(p: CoordinatorLoopParams<'_>) -> (bool, usize) {
    let mut user_aborted = false;
    let mut finished_missing = 0usize;
    let mut last_bytes_done = p.already_downloaded;
    let mut refresh_interval = Duration::from_millis(CHUNKED_REFRESH_IDLE_MS);

    while finished_missing < p.missing_parts_count {
        if check_chunk_interrupt(p.cancel, p.pb, p.parts_done) {
            user_aborted = true;
            break;
        }

        if p.first_error
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .is_some()
        {
            p.cancel.store(true, Ordering::Release);
            break;
        }

        match p.done_rx.recv_timeout(refresh_interval) {
            Ok(part_idx) => {
                if on_chunk_completed(
                    part_idx,
                    p.state,
                    p.workspace,
                    p.models_dir_ownership,
                    p.cancel,
                    p.first_error,
                    &mut finished_missing,
                ) {
                    break;
                }
            }
            Err(mpsc::RecvTimeoutError::Timeout) => {}
            Err(mpsc::RecvTimeoutError::Disconnected) => break,
        }

        update_chunk_progress(
            p.pb,
            p.bytes_done,
            p.parts_done,
            p.named_digest,
            p.num_parts,
            &mut last_bytes_done,
            &mut refresh_interval,
        );
    }

    (user_aborted, finished_missing)
}

fn download_missing_chunks_parallel(request: DownloadChunksParallelRequest<'_>) -> Result<bool> {
    let DownloadChunksParallelRequest {
        client,
        url,
        named_digest,
        workspace,
        missing_parts,
        state,
        num_parts,
        chunk_size,
        total_size,
        already_downloaded,
        completed_parts,
        pb,
        models_dir_ownership,
        download_chunks_in_parallel,
    } = request;

    if missing_parts.is_empty() {
        return Ok(false);
    }

    let max_workers = if download_chunks_in_parallel {
        MAX_PARALLEL_CHUNK_WORKERS
    } else {
        1
    };
    let workers = missing_parts.len().min(max_workers);
    let queue = Arc::new(Mutex::new(VecDeque::from(missing_parts.to_vec())));
    let bytes_done = Arc::new(AtomicU64::new(already_downloaded));
    let parts_done = Arc::new(AtomicU64::new(completed_parts));
    let cancel = Arc::new(AtomicBool::new(false));
    let first_error = Arc::new(Mutex::new(None::<DownloaderError>));
    let (done_tx, done_rx) = mpsc::channel::<u64>();

    let mut user_aborted = false;
    let mut finished_missing: usize = 0;

    thread::scope(|scope| {
        for _ in 0..workers {
            let worker = ChunkWorker {
                client: client.clone(),
                url,
                named_digest,
                data_file: workspace.data_file.clone(),
                queue: Arc::clone(&queue),
                bytes_done: Arc::clone(&bytes_done),
                parts_done: Arc::clone(&parts_done),
                cancel: Arc::clone(&cancel),
                first_error: Arc::clone(&first_error),
                done_tx: done_tx.clone(),
                num_parts,
                chunk_size,
                total_size,
            };
            scope.spawn(move || worker.run());
        }

        drop(done_tx);

        (user_aborted, finished_missing) = run_coordinator_loop(CoordinatorLoopParams {
            done_rx: &done_rx,
            cancel: &cancel,
            first_error: &first_error,
            bytes_done: &bytes_done,
            parts_done: &parts_done,
            pb,
            named_digest,
            num_parts,
            models_dir_ownership,
            state,
            workspace,
            missing_parts_count: missing_parts.len(),
            already_downloaded,
        });

        cancel.store(true, Ordering::Release);
    });

    pb.set_position(bytes_done.load(Ordering::Acquire));

    if let Some(err) = first_error.lock().unwrap_or_else(|e| e.into_inner()).take() {
        return Err(err);
    }

    if user_aborted {
        return Ok(true);
    }

    if finished_missing < missing_parts.len() {
        return Err(DownloaderError::Other(
            "Chunked download stopped before completion".to_string(),
        ));
    }

    Ok(false)
}

struct ChunkWorker<'a> {
    client: Client,
    url: &'a str,
    named_digest: &'a str,
    data_file: PathBuf,
    queue: Arc<Mutex<VecDeque<u64>>>,
    bytes_done: Arc<AtomicU64>,
    parts_done: Arc<AtomicU64>,
    cancel: Arc<AtomicBool>,
    first_error: Arc<Mutex<Option<DownloaderError>>>,
    done_tx: mpsc::Sender<u64>,
    num_parts: u64,
    chunk_size: u64,
    total_size: u64,
}

impl ChunkWorker<'_> {
    fn run(self) {
        loop {
            if self.cancel.load(Ordering::Acquire) || crate::signal_handler::is_interrupted() {
                return;
            }

            let next_part = {
                let mut guard = self.queue.lock().unwrap_or_else(|e| e.into_inner());
                guard.pop_front()
            };

            let Some(part_idx) = next_part else {
                return;
            };

            match self.download_one_part(part_idx) {
                Ok(()) => {
                    self.parts_done.fetch_add(1, Ordering::AcqRel);
                    let _ = self.done_tx.send(part_idx);
                }
                Err(e) => {
                    self.set_error_once(e);
                    self.cancel.store(true, Ordering::Release);
                    return;
                }
            }
        }
    }

    fn set_error_once(&self, err: DownloaderError) {
        let mut guard = self.first_error.lock().unwrap_or_else(|e| e.into_inner());
        if guard.is_none() {
            *guard = Some(err);
        }
    }

    fn download_one_part(&self, part_idx: u64) -> Result<()> {
        let expected_size =
            expected_part_size(part_idx, self.num_parts, self.total_size, self.chunk_size);
        let range_start = part_idx * self.chunk_size;
        let range_end = range_start + expected_size - 1;
        let range_header = format!("bytes={}-{}", range_start, range_end);

        let mut response = self
            .client
            .get(self.url)
            .header("Range", &range_header)
            .send()
            .map_err(DownloaderError::HttpError)?;

        if response.status() != StatusCode::PARTIAL_CONTENT {
            return Err(DownloaderError::Other(format!(
                "Expected 206 Partial Content for range request on {}, got {}",
                self.named_digest,
                response.status()
            )));
        }

        let mut out = fs::OpenOptions::new()
            .write(true)
            .open(&self.data_file)
            .map_err(DownloaderError::IoError)?;
        out.seek(SeekFrom::Start(range_start))
            .map_err(DownloaderError::IoError)?;

        let mut bytes_written: u64 = 0;
        let mut buffer = vec![0u8; CHUNK_IO_BUFFER_SIZE];

        loop {
            if self.cancel.load(Ordering::Acquire) || crate::signal_handler::is_interrupted() {
                return Ok(());
            }

            let bytes_read = match response.read(&mut buffer) {
                Ok(0) => break,
                Ok(n) => n,
                Err(e) => return Err(DownloaderError::IoError(e)),
            };

            out.write_all(&buffer[..bytes_read])
                .map_err(DownloaderError::IoError)?;
            bytes_written += bytes_read as u64;
            self.bytes_done
                .fetch_add(bytes_read as u64, Ordering::AcqRel);
        }

        if bytes_written != expected_size {
            return Err(DownloaderError::Other(format!(
                "Part {} of {} incomplete ({}/{} bytes)",
                part_idx, self.named_digest, bytes_written, expected_size
            )));
        }

        Ok(())
    }
}

fn compute_file_sha256(path: &Path, named_digest: &str) -> Result<String> {
    let mut hasher = Sha256::new();
    let mut file = fs::File::open(path)?;
    let mut buffer = vec![0u8; HASH_IO_BUFFER_SIZE];
    let total_size = file.metadata().map(|m| m.len()).unwrap_or(0);

    info!("Verifying downloaded BLOB {}", named_digest);

    let pb = ProgressBar::new(total_size);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{msg} [{bar:40.cyan/blue}] {bytes}/{total_bytes} ({eta})")
            .unwrap()
            .progress_chars("#>-"),
    );
    pb.set_message(format!("Verifying BLOB {}", named_digest));

    loop {
        let n = file.read(&mut buffer)?;
        if n == 0 {
            break;
        }
        hasher.update(&buffer[..n]);
        pb.inc(n as u64);
    }

    pb.finish_with_message("Verified");

    Ok(format!("{:x}", hasher.finalize()))
}

fn cleanup_chunk_workspace(workspace: &ChunkedWorkspace) {
    if let Err(e) = fs::remove_file(&workspace.state_file)
        && e.kind() != std::io::ErrorKind::NotFound
    {
        warn!(
            "Failed to remove chunk state file {:?}: {}",
            workspace.state_file, e
        );
    }
    if let Err(e) = fs::remove_file(&workspace.data_file)
        && e.kind() != std::io::ErrorKind::NotFound
    {
        warn!(
            "Failed to remove chunk data file {:?}: {}",
            workspace.data_file, e
        );
    }
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
