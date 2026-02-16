use crate::config::AppSettings;
use crate::downloader::manifest::ImageManifest;
use crate::downloader::model_downloader::{DownloaderError, ModelDownloader, Result};
use indicatif::{ProgressBar, ProgressStyle};
use log::{debug, error, info, warn};
use reqwest::blocking::Client;
use scraper::{Html, Selector};
use sha2::{Digest, Sha256};
use std::collections::HashSet;
use std::env;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use tempfile::NamedTempFile;

/// Downloader for Ollama library models
pub struct OllamaModelDownloader {
    settings: AppSettings,
    user_agent: String,
    client: Client,
    unnecessary_files: HashSet<PathBuf>,
}

impl OllamaModelDownloader {
    /// Create a new Ollama model downloader
    ///
    /// # Arguments
    /// * `settings` - Application settings
    ///
    /// # Returns
    /// * `Result<Self>` - New downloader instance or error
    pub fn new(settings: AppSettings) -> Result<Self> {
        let pkg_version = env!("CARGO_PKG_VERSION");
        let os_info = format!("{}-{}", std::env::consts::OS, std::env::consts::ARCH);
        let user_agent = format!("odir/{} ({})", pkg_version, os_info);

        let client = Client::builder()
            .user_agent(&user_agent)
            .danger_accept_invalid_certs(!settings.ollama_library.verify_ssl)
            .timeout(std::time::Duration::from_secs_f64(
                settings.ollama_library.timeout,
            ))
            .build()?;

        Ok(Self {
            settings,
            user_agent,
            client,
            unnecessary_files: HashSet::new(),
        })
    }

    /// Construct the manifest URL for a given model identifier
    fn make_manifest_url(&self, model: &str, tag: &str) -> String {
        format!(
            "{}{}/manifests/{}",
            self.settings.ollama_library.registry_base_url, model, tag
        )
    }

    /// Fetch the manifest JSON for a model
    fn fetch_manifest(&self, model: &str, tag: &str) -> Result<String> {
        let url = self.make_manifest_url(model, tag);
        info!("Downloading manifest from {}", url);

        let response = self.client.get(&url).send()?;

        if !response.status().is_success() {
            return Err(DownloaderError::HttpError(
                response.error_for_status().unwrap_err(),
            ));
        }

        Ok(response.text()?)
    }

    /// Construct the blob URL for a given model and digest
    fn make_blob_url(&self, model: &str, digest: &str) -> String {
        format!(
            "{}{}/blobs/{}",
            self.settings.ollama_library.registry_base_url,
            model,
            digest.replace(':', "-")
        )
    }

    /// Download a model blob with progress tracking
    fn download_model_blob(
        &mut self,
        model: &str,
        named_digest: &str,
    ) -> Result<(PathBuf, String)> {
        let url = self.make_blob_url(model, named_digest);

        let mut hasher = Sha256::new();
        let mut temp_file = NamedTempFile::new().map_err(DownloaderError::IoError)?;

        let temp_path = temp_file.path().to_path_buf();
        self.unnecessary_files.insert(temp_path.clone());

        let response = self.client.get(&url).send()?;

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
        pb.set_message(format!(
            "Downloading BLOB {}...{}",
            &named_digest[..11.min(named_digest.len())],
            &named_digest[named_digest.len().saturating_sub(4)..]
        ));

        // For blocking client, get all bytes at once
        let bytes = response.bytes()?;

        for chunk in bytes.chunks(8192) {
            hasher.update(chunk);
            temp_file.write_all(chunk)?;
            pb.inc(chunk.len() as u64);
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

    /// Save the blob to the models directory
    fn save_blob(
        &mut self,
        source: &Path,
        named_digest: &str,
        computed_digest: &str,
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

        let models_path = if self.settings.ollama_library.models_path.starts_with('~') {
            let home = std::env::var("HOME").map_err(|_| {
                DownloaderError::Other("HOME environment variable not set".to_string())
            })?;
            PathBuf::from(
                self.settings
                    .ollama_library
                    .models_path
                    .replacen("~", &home, 1),
            )
        } else {
            PathBuf::from(&self.settings.ollama_library.models_path)
        };

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

        // Remove source from unnecessary files and add target
        self.unnecessary_files.remove(&source.to_path_buf());
        self.unnecessary_files.insert(target_file.clone());

        info!("Moved {:?} to {:?}", source, target_file);

        Ok(target_file)
    }

    /// Save the manifest to the models directory
    fn save_manifest(&mut self, data: &str, model: &str, tag: &str) -> Result<PathBuf> {
        let models_path = if self.settings.ollama_library.models_path.starts_with('~') {
            let home = std::env::var("HOME").map_err(|_| {
                DownloaderError::Other("HOME environment variable not set".to_string())
            })?;
            PathBuf::from(
                self.settings
                    .ollama_library
                    .models_path
                    .replacen("~", &home, 1),
            )
        } else {
            PathBuf::from(&self.settings.ollama_library.models_path)
        };

        let manifests_toplevel_dir = models_path.join("manifests");

        // Parse registry hostname from URL
        let registry_url = &self.settings.ollama_library.registry_base_url;
        let registry_host = registry_url
            .split("//")
            .nth(1)
            .and_then(|s| s.split('/').next())
            .unwrap_or("registry.ollama.ai");

        let manifests_dir = manifests_toplevel_dir
            .join(registry_host)
            .join("library")
            .join(model);

        if !manifests_dir.exists() {
            warn!(
                "Manifests path {:?} does not exist. Creating it.",
                manifests_dir
            );
            fs::create_dir_all(&manifests_dir)?;
            self.unnecessary_files.insert(manifests_dir.clone());
        }

        let target_file = manifests_dir.join(tag);
        fs::write(&target_file, data)?;
        info!("Saved manifest to {:?}", target_file);

        self.unnecessary_files.insert(target_file.clone());

        Ok(target_file)
    }

    /// Cleanup unnecessary files on error
    fn cleanup_unnecessary_files(&mut self) {
        let files_to_remove: Vec<PathBuf> = self.unnecessary_files.iter().cloned().collect();

        for file_path in files_to_remove {
            if file_path.is_file() {
                if let Err(e) = fs::remove_file(&file_path) {
                    warn!("Failed to remove unnecessary file {:?}: {}", file_path, e);
                } else {
                    info!("Removed unnecessary file: {:?}", file_path);
                    self.unnecessary_files.remove(&file_path);
                }
            } else if file_path.is_dir() {
                if let Err(e) = fs::remove_dir(&file_path) {
                    debug!(
                        "Failed to remove unnecessary directory {:?}: {}",
                        file_path, e
                    );
                } else {
                    info!("Removed unnecessary directory: {:?}", file_path);
                    self.unnecessary_files.remove(&file_path);
                }
            }
        }
    }
}

impl ModelDownloader for OllamaModelDownloader {
    fn download_model(&self, model_identifier: &str) -> Result<bool> {
        let (model, tag) = if model_identifier.contains(':') {
            let parts: Vec<&str> = model_identifier.split(':').collect();
            (parts[0].to_string(), parts[1].to_string())
        } else {
            (model_identifier.to_string(), "latest".to_string())
        };

        println!("Downloading Ollama library model {}:{}", model, tag);

        // Make self mutable for this scope
        let mut self_mut = Self {
            settings: self.settings.clone(),
            user_agent: self.user_agent.clone(),
            client: self.client.clone(),
            unnecessary_files: HashSet::new(),
        };

        // Fetch and parse manifest
        let manifest_json = self_mut.fetch_manifest(&model, &tag)?;
        info!("Validating manifest for {}:{}", model, tag);

        let manifest: ImageManifest = serde_json::from_str(&manifest_json)
            .map_err(|e| DownloaderError::ParseError(format!("Failed to parse manifest: {}", e)))?;

        // Track files to be saved (source_path, named_digest, computed_digest)
        let mut files_to_be_copied: Vec<(PathBuf, String, String)> = Vec::new();

        // Download model configuration BLOB
        info!("Downloading model configuration {}", manifest.config.digest);
        let (file_model_config, digest_model_config) =
            self_mut.download_model_blob(&model, &manifest.config.digest)?;
        files_to_be_copied.push((
            file_model_config,
            manifest.config.digest.clone(),
            digest_model_config,
        ));

        // Download layers if present
        if let Some(layers) = &manifest.layers {
            for layer in layers {
                debug!(
                    "Layer: {}, Size: {} bytes, Digest: {}",
                    layer.media_type, layer.size, layer.digest
                );
                info!("Downloading {} layer {}", layer.media_type, layer.digest);
                let (file_layer, digest_layer) =
                    self_mut.download_model_blob(&model, &layer.digest)?;
                files_to_be_copied.push((file_layer, layer.digest.clone(), digest_layer));
            }
        }

        // All BLOBs downloaded, now save them
        for (source, named_digest, computed_digest) in files_to_be_copied {
            match self_mut.save_blob(&source, &named_digest, &computed_digest) {
                Ok(_) => {
                    // Cleanup source file
                    let _ = fs::remove_file(&source);
                }
                Err(e) => {
                    error!("Failed to save BLOB {}: {}", named_digest, e);
                    self_mut.cleanup_unnecessary_files();
                    return Err(e);
                }
            }
        }

        // Save the manifest
        match self_mut.save_manifest(&manifest_json, &model, &tag) {
            Ok(_) => {}
            Err(e) => {
                error!("Failed to save manifest: {}", e);
                if self.settings.ollama_server.remove_downloaded_on_error {
                    self_mut.cleanup_unnecessary_files();
                }
                return Err(e);
            }
        }

        // Clear unnecessary files list on success
        self_mut.unnecessary_files.clear();

        println!("Model {}:{} successfully downloaded", model, tag);
        Ok(true)
    }

    fn list_available_models(
        &self,
        page: Option<u32>,
        page_size: Option<u32>,
    ) -> Result<Vec<String>> {
        debug!(
            "Updating models list from Ollama library {}",
            self.settings.ollama_library.library_base_url
        );

        let response = self
            .client
            .get(&self.settings.ollama_library.library_base_url)
            .send()?;

        if !response.status().is_success() {
            return Err(DownloaderError::HttpError(
                response.error_for_status().unwrap_err(),
            ));
        }

        let html_content = response.text()?;
        let document = Html::parse_document(&html_content);

        // Select all anchor tags
        let link_selector = Selector::parse("a[href]")
            .map_err(|e| DownloaderError::ParseError(format!("Invalid selector: {:?}", e)))?;

        let library_prefix = "/library/";
        let mut available_models: Vec<String> = Vec::new();

        for element in document.select(&link_selector) {
            if let Some(href) = element.value().attr("href")
                && href.starts_with(library_prefix)
            {
                let model_name = href.trim_start_matches(library_prefix).to_string();
                // Only add if not empty and doesn't end with slash (avoid directory links)
                if !model_name.is_empty() && !model_name.ends_with('/') {
                    available_models.push(model_name);
                }
            }
        }

        debug!(
            "Found {} models in the Ollama library",
            available_models.len()
        );

        // Sort models case-insensitively
        available_models.sort_by_key(|a| a.to_lowercase());

        // Apply pagination if requested
        let paginated_result = if let (Some(page), Some(page_size)) = (page, page_size) {
            let start_index = ((page - 1) * page_size) as usize;
            let end_index = (start_index + page_size as usize).min(available_models.len());

            if start_index >= available_models.len() {
                warn!(
                    "No models found for page {} with page size {}. Returning all models instead.",
                    page, page_size
                );
                available_models
            } else {
                available_models[start_index..end_index].to_vec()
            }
        } else {
            available_models
        };

        Ok(paginated_result)
    }

    fn list_model_tags(&self, model_identifier: &str) -> Result<Vec<String>> {
        // Check if model exists first
        let available_models = self.list_available_models(None, None)?;
        if !available_models.contains(&model_identifier.to_string()) {
            return Err(DownloaderError::ModelNotFound(format!(
                "Model {} not found in the library models list",
                model_identifier
            )));
        }

        let tags_url = format!(
            "{}{}/tags",
            self.settings.ollama_library.library_base_url, model_identifier
        );

        debug!(
            "Fetching tags for model {} from the Ollama library.",
            model_identifier
        );

        let response = self.client.get(&tags_url).send()?;

        if !response.status().is_success() {
            return Err(DownloaderError::HttpError(
                response.error_for_status().unwrap_err(),
            ));
        }

        let html_content = response.text()?;
        let document = Html::parse_document(&html_content);

        debug!("Parsing tags for model {}.", model_identifier);

        let link_selector = Selector::parse("a[href]")
            .map_err(|e| DownloaderError::ParseError(format!("Invalid selector: {:?}", e)))?;

        let library_prefix = "/library/";
        let model_tag_prefix = format!("{}{}:", library_prefix, model_identifier);
        let mut named_model_unique_tags = std::collections::HashSet::new();

        for element in document.select(&link_selector) {
            if let Some(href) = element.value().attr("href")
                && href.starts_with(&model_tag_prefix)
            {
                let model_tag = href.trim_start_matches(library_prefix).to_string();
                named_model_unique_tags.insert(model_tag);
            }
        }

        let mut models_tags: Vec<String> = named_model_unique_tags.into_iter().collect();

        // Sort tags case-insensitively
        models_tags.sort_by_key(|a| a.to_lowercase());

        Ok(models_tags)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ollama_downloader_creation() {
        let settings = AppSettings::default();
        let downloader = OllamaModelDownloader::new(settings);
        assert!(downloader.is_ok());
    }
}
