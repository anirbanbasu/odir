use std::io;
use thiserror::Error;

/// Error types for model downloading operations
#[derive(Error, Debug)]
pub enum DownloaderError {
    #[error("HTTP request failed: {0}")]
    HttpError(#[from] reqwest::Error),

    #[error("Failed to parse HTML: {0}")]
    ParseError(String),

    #[error("Model not found: {0}")]
    ModelNotFound(String),

    #[error("IO error: {0}")]
    IoError(#[from] io::Error),

    #[error("Invalid model identifier: {0}")]
    InvalidIdentifier(String),

    #[error("{0}")]
    Other(String),
}

pub type Result<T> = std::result::Result<T, DownloaderError>;

/// Trait defining the common interface for model downloaders
pub trait ModelDownloader {
    /// Download a model from the model source.
    ///
    /// # Arguments
    /// * `model_identifier` - The model identifier (e.g., "llama2:latest" or "user/repo:tag")
    ///
    /// # Returns
    /// * `Result<bool>` - True if download successful
    fn download_model(&self, model_identifier: &str) -> Result<bool>;

    /// List available models from the model source.
    ///
    /// # Arguments
    /// * `page` - Optional page number (1-indexed) for pagination
    /// * `page_size` - Optional number of models per page
    ///
    /// # Returns
    /// * `Result<Vec<String>>` - List of model identifiers
    fn list_available_models(
        &self,
        page: Option<u32>,
        page_size: Option<u32>,
    ) -> Result<Vec<String>>;

    /// List available tags for a specific model.
    ///
    /// # Arguments
    /// * `model_identifier` - The name of the model (without tag)
    ///
    /// # Returns
    /// * `Result<Vec<String>>` - List of available tags for the model
    fn list_model_tags(&self, model_identifier: &str) -> Result<Vec<String>>;
}
