use crate::config::AppSettings;
use crate::downloader::manifest::ImageManifest;
use crate::downloader::model_downloader::{DownloaderError, ModelDownloader, Result};
use crate::downloader::utils::{
    Ownership, cleanup_unnecessary_files, download_model_blob, expand_models_path,
    infer_models_dir_ownership, is_model_present_in_ollama, save_blob, save_manifest,
};
use log::{debug, error, info, warn};
use reqwest::blocking::Client;
use serde::Deserialize;
use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};

const HF_BASE_URL: &str = "https://hf.co/v2/";

#[derive(Debug, Deserialize)]
struct HfModel {
    #[serde(rename = "modelId")]
    model_id: String,
}

#[derive(Debug, Deserialize)]
struct HfModelSibling {
    rfilename: String,
}

#[derive(Debug, Deserialize)]
struct HfModelInfo {
    siblings: Vec<HfModelSibling>,
}

/// Downloader for Hugging Face models compatible with Ollama
pub struct HuggingFaceModelDownloader {
    settings: AppSettings,
    user_agent: String,
    client: Client,
    unnecessary_files: HashSet<PathBuf>,
    models_dir_ownership: Option<Ownership>,
}

impl HuggingFaceModelDownloader {
    /// Create a new Hugging Face model downloader
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

    /// Construct the manifest URL for a HuggingFace model
    fn make_manifest_url(&self, model_identifier: &str) -> String {
        // model_identifier should be like "user/repo:tag"
        let url_part = model_identifier.replace(':', "/manifests/");
        format!("{}{}", HF_BASE_URL, url_part)
    }

    /// Fetch the manifest JSON for a HuggingFace model
    fn fetch_manifest(&self, model_identifier: &str) -> Result<String> {
        let url = self.make_manifest_url(model_identifier);
        info!("Downloading manifest from {}", url);

        let response = self.client.get(&url).send()?;

        if !response.status().is_success() {
            return Err(DownloaderError::HttpError(
                response.error_for_status().unwrap_err(),
            ));
        }

        Ok(response.text()?)
    }

    /// Construct the blob URL for a HuggingFace model
    fn make_blob_url(&self, model_repo: &str, digest: &str) -> String {
        format!("{}{}/blobs/{}", HF_BASE_URL, model_repo, digest)
    }

    /// Download a model blob with progress tracking
    fn download_model_blob(
        &mut self,
        model_repo: &str,
        named_digest: &str,
    ) -> Result<(PathBuf, String)> {
        let url = self.make_blob_url(model_repo, named_digest);
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
    fn save_manifest(&mut self, data: &str, model_identifier: &str) -> Result<PathBuf> {
        let models_path = expand_models_path(&self.settings.ollama_library.models_path)?;
        let manifests_toplevel_dir = models_path.join("manifests");

        // Parse HF hostname
        let hf_host = HF_BASE_URL
            .split("//")
            .nth(1)
            .and_then(|s| s.split('/').next())
            .unwrap_or("hf.co");

        let parts: Vec<&str> = model_identifier.split(':').collect();
        let model_repo = parts[0];
        let tag = parts.get(1).unwrap_or(&"latest");

        let manifests_dir = manifests_toplevel_dir.join(hf_host).join(model_repo);

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

impl ModelDownloader for HuggingFaceModelDownloader {
    fn download_model(&self, model_identifier: &str) -> Result<bool> {
        let (model_repo, quant) = if model_identifier.contains(':') {
            let parts: Vec<&str> = model_identifier.split(':').collect();
            (parts[0].to_string(), parts[1].to_string())
        } else {
            (model_identifier.to_string(), "latest".to_string())
        };

        let parts: Vec<&str> = model_repo.split('/').collect();
        if parts.len() != 2 {
            return Err(DownloaderError::InvalidIdentifier(
                "HuggingFace model identifier must be in format 'user/repository:quantization'"
                    .to_string(),
            ));
        }

        let user = parts[0];
        let repo = parts[1];

        println!(
            "Downloading Hugging Face model {} from {} with {} quantisation",
            repo, user, quant
        );

        // Make self mutable for this scope
        let mut self_mut = Self {
            settings: self.settings.clone(),
            user_agent: self.user_agent.clone(),
            client: self.client.clone(),
            unnecessary_files: HashSet::new(),
            models_dir_ownership: self.models_dir_ownership,
        };

        // Fetch and parse manifest
        let manifest_json = self_mut.fetch_manifest(model_identifier)?;
        info!("Validating manifest for {}", model_identifier);

        let manifest: ImageManifest = serde_json::from_str(&manifest_json)
            .map_err(|e| DownloaderError::ParseError(format!("Failed to parse manifest: {}", e)))?;

        // Track files to be saved (source_path, named_digest, computed_digest)
        let mut files_to_be_copied: Vec<(PathBuf, String, String)> = Vec::new();

        // Download model configuration BLOB
        info!("Downloading model configuration {}", manifest.config.digest);
        let (file_model_config, digest_model_config) =
            self_mut.download_model_blob(&model_repo, &manifest.config.digest)?;
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
                    self_mut.download_model_blob(&model_repo, &layer.digest)?;
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
        match self_mut.save_manifest(&manifest_json, model_identifier) {
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
            let hf_host = HF_BASE_URL
                .split("//")
                .nth(1)
                .and_then(|s| s.split('/').next())
                .unwrap_or("hf.co");
            let model_name = format!("{}/{}", hf_host, model_identifier);
            let model_names = vec![
                model_name.clone(),
                format!("huggingface.co/{}", model_identifier),
                model_identifier.to_string(),
            ];

            info!("Verifying model {} is present in Ollama server", model_name);
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
                    "Model {} not found in Ollama server after download",
                    model_name
                );
                error!("{}", err_msg);
                if self.settings.ollama_server.remove_downloaded_on_error {
                    info!("Removing downloaded files because model not found in Ollama server");
                    self_mut.cleanup_unnecessary_files();
                }
                return Err(DownloaderError::Other(err_msg));
            }

            info!("Model {} verified in Ollama server", model_name);
        } else {
            debug!("Model presence check is disabled via settings");
        }

        // Clear unnecessary files list on success
        self_mut.unnecessary_files.clear();

        println!(
            "HuggingFace model {} successfully downloaded",
            model_identifier
        );

        Ok(true)
    }

    fn list_available_models(
        &self,
        page: Option<u32>,
        page_size: Option<u32>,
    ) -> Result<Vec<String>> {
        let page = page.unwrap_or(1);
        let page_size = page_size.unwrap_or(25).min(100);

        // Check HuggingFace limitation
        if page_size * (page + 1) >= 1000 {
            warn!("Hugging Face currently does not allow paging beyond the first 999 models");
            return Err(DownloaderError::Other(format!(
                "Hugging Face currently does not allow obtaining information beyond the first 999 models. \
                Your requested page {} with page size {} exceeds this limit by {} model(s).",
                page,
                page_size,
                (page + 1) * page_size - 999
            )));
        }

        let api_url = format!(
            "https://huggingface.co/api/models?apps=ollama&gated=false&limit={}&sort=trendingScore",
            page_size
        );

        let mut next_page_url = Some(api_url.clone());
        let mut current_page = 1u32;

        // Navigate to the requested page
        while current_page < page && next_page_url.is_some() {
            let url = next_page_url.unwrap();
            debug!("Checking pagination for page {}", current_page);

            let response = self.client.head(&url).send()?;

            if !response.status().is_success() {
                return Err(DownloaderError::HttpError(
                    response.error_for_status().unwrap_err(),
                ));
            }

            // Extract next page URL from Link header
            next_page_url = response
                .headers()
                .get("link")
                .and_then(|link| link.to_str().ok())
                .and_then(|link_str| {
                    // Parse Link header to extract "next" URL
                    link_str.split(',').find_map(|part| {
                        if part.contains("rel=\"next\"") {
                            let url_part = part.split(';').next()?;
                            let url = url_part
                                .trim()
                                .trim_start_matches('<')
                                .trim_end_matches('>');
                            Some(url.to_string())
                        } else {
                            None
                        }
                    })
                });

            current_page += 1;
        }

        if next_page_url.is_none() {
            return Err(DownloaderError::Other(format!(
                "Requested page {} is beyond available data",
                page
            )));
        }

        let final_url = next_page_url.unwrap();

        if current_page > 1 {
            info!("Requesting page {} from {}", current_page, final_url);
        }

        let response = self.client.get(&final_url).send()?;

        if !response.status().is_success() {
            return Err(DownloaderError::HttpError(
                response.error_for_status().unwrap_err(),
            ));
        }

        let models: Vec<HfModel> = response.json()?;
        let mut model_identifiers: Vec<String> = models.into_iter().map(|m| m.model_id).collect();

        warn!("HuggingFace models are sorted in the context of the selected page only");

        // Sort case-insensitively
        model_identifiers.sort_by_key(|a| a.to_lowercase());

        Ok(model_identifiers)
    }

    fn list_model_tags(&self, model_identifier: &str) -> Result<Vec<String>> {
        let api_url = format!(
            "https://huggingface.co/api/models/{}?blobs=true",
            model_identifier
        );

        debug!(
            "Fetching tags for model {} from HuggingFace API",
            model_identifier
        );

        let response = self.client.get(&api_url).send()?;

        if !response.status().is_success() {
            return Err(DownloaderError::HttpError(
                response.error_for_status().unwrap_err(),
            ));
        }

        let model_info: HfModelInfo = response.json()?;
        let mut tags: Vec<String> = Vec::new();

        for sibling in model_info.siblings {
            if sibling.rfilename.ends_with(".gguf") {
                // Extract quantisation from filename
                // Typically filenames are like: model-Q4_K_M.gguf
                if let Some(tag_part) = sibling
                    .rfilename
                    .strip_suffix(".gguf")
                    .and_then(|s| s.split('-').next_back())
                {
                    tags.push(format!("{}:{}", model_identifier, tag_part));
                }
            }
        }

        if tags.is_empty() {
            return Err(DownloaderError::Other(format!(
                "The model {} has no support for Ollama (no .gguf files found)",
                model_identifier
            )));
        }

        // Sort case-insensitively
        tags.sort_by_key(|a| a.to_lowercase());

        Ok(tags)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hf_downloader_creation() {
        let settings = AppSettings::default();
        let downloader = HuggingFaceModelDownloader::new(settings);
        assert!(downloader.is_ok());
    }

    #[test]
    #[ignore] // Run manually with: cargo test -- --ignored
    fn test_hf_model_download() {
        // Initialize logger for test output
        let _ = env_logger::builder().is_test(true).try_init();

        let settings = AppSettings::default();
        let downloader =
            HuggingFaceModelDownloader::new(settings).expect("Failed to create downloader");

        // Download a small model for testing
        let model_identifier = "unsloth/SmolLM2-135M-Instruct-GGUF:Q4_K_M";
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
