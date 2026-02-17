//! Downloader implementation for Ollama library models.
use crate::config::AppSettings;
use crate::downloader::manifest::ImageManifest;
use crate::downloader::model_downloader::{DownloaderError, ModelDownloader, Result};
use crate::downloader::utils::{
    Ownership, cleanup_unnecessary_files, download_model_blob, expand_models_path,
    infer_models_dir_ownership, is_model_present_in_ollama, save_blob, save_manifest,
};
use log::{debug, error, info, warn};
use reqwest::blocking::Client;
use scraper::{Html, Selector};
use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};

/// Downloader for Ollama library models
pub struct OllamaModelDownloader {
    settings: AppSettings,
    user_agent: String,
    client: Client,
    unnecessary_files: HashSet<PathBuf>,
    models_dir_ownership: Option<Ownership>,
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

        let models_dir_ownership =
            infer_models_dir_ownership(&settings.ollama_library.models_path)?;

        Ok(Self {
            settings,
            user_agent,
            client,
            unnecessary_files: HashSet::new(),
            models_dir_ownership,
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
        download_model_blob(
            &self.client,
            &url,
            named_digest,
            &mut self.unnecessary_files,
        )
    }

    /// Save the blob to the models directory
    fn save_blob(
        &mut self,
        source: &Path,
        named_digest: &str,
        computed_digest: &str,
    ) -> Result<PathBuf> {
        save_blob(
            &self.settings.ollama_library.models_path,
            source,
            named_digest,
            computed_digest,
            self.models_dir_ownership,
            &mut self.unnecessary_files,
        )
    }

    /// Save the manifest to the models directory
    fn save_manifest(&mut self, data: &str, model: &str, tag: &str) -> Result<PathBuf> {
        let models_path = expand_models_path(&self.settings.ollama_library.models_path)?;
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

        save_manifest(
            data,
            &models_path,
            &manifests_dir,
            tag,
            self.models_dir_ownership,
            &[&manifests_dir, &manifests_toplevel_dir],
            &mut self.unnecessary_files,
        )
    }

    /// Cleanup unnecessary files on error
    fn cleanup_unnecessary_files(&mut self) {
        cleanup_unnecessary_files(&mut self.unnecessary_files);
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
            models_dir_ownership: self.models_dir_ownership,
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

        // Verify the model is present in the Ollama server if configured
        if self.settings.ollama_server.check_model_presence {
            info!(
                "Verifying model {}:{} is present in Ollama server",
                model, tag
            );
            let model_name = format!("{}:{}", model, tag);
            let registry_host = self
                .settings
                .ollama_library
                .registry_base_url
                .split("//")
                .nth(1)
                .and_then(|s| s.split('/').next())
                .unwrap_or("registry.ollama.ai");
            let model_names = vec![
                model_name.clone(),
                format!("library/{}", model_name),
                format!("{}/library/{}", registry_host, model_name),
            ];
            let model_present = match is_model_present_in_ollama(
                &self_mut.client,
                &self.settings.ollama_server.url,
                &model_names,
            ) {
                Ok(present) => present,
                Err(e) => {
                    error!("Failed to verify model with Ollama server: {}", e);
                    if self.settings.ollama_server.remove_downloaded_on_error {
                        info!("Removing downloaded files due to verification failure");
                        self_mut.cleanup_unnecessary_files();
                    }
                    return Err(e);
                }
            };

            if !model_present {
                let err_msg = format!(
                    "Model {}:{} not found in Ollama server after download",
                    model, tag
                );
                error!("{}", err_msg);
                if self.settings.ollama_server.remove_downloaded_on_error {
                    info!("Removing downloaded files because model not found in Ollama server");
                    self_mut.cleanup_unnecessary_files();
                }
                return Err(DownloaderError::Other(err_msg));
            }

            info!("Model {}:{} verified in Ollama server", model, tag);
        } else {
            debug!("Model presence check is disabled via settings");
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

    #[test]
    #[ignore] // Run manually with: cargo test -- --ignored
    fn test_ollama_model_download() {
        // Initialize logger for test output
        let _ = env_logger::builder().is_test(true).try_init();

        let settings = AppSettings::default();
        let downloader = OllamaModelDownloader::new(settings).expect("Failed to create downloader");

        // Download a small model for testing
        let model_identifier = "all-minilm:22m";
        println!("Testing download of {}", model_identifier);

        let result = downloader.download_model(model_identifier);

        match result {
            Ok(success) => {
                assert!(success, "Download should return true on success");
                println!("Successfully downloaded {}", model_identifier);
            }
            Err(e) => {
                panic!("Download failed: {:?}", e);
            }
        }
    }
}
