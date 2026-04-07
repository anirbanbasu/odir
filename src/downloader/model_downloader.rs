//! Model downloader trait and error definitions for the Ollama Downloader in Rust (ODIR).
use reqwest::StatusCode;
use reqwest::blocking::Response;
use std::io;
use thiserror::Error;

/// Error types for model downloading operations
#[derive(Error, Debug)]
pub enum DownloaderError {
    #[error("HTTP request failed: {0}")]
    HttpError(#[from] reqwest::Error),

    #[error("HTTP request failed: {0}")]
    HttpStatus(String),

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

pub fn http_status_error_from_response(response: Response) -> DownloaderError {
    let status = response.status();
    let url = response.url().to_string();
    let body = response.text().ok();
    let summarized_body = summarize_http_error_body(body.as_deref());

    DownloaderError::HttpStatus(format_http_status_error(
        status,
        &url,
        summarized_body.as_deref(),
    ))
}

fn format_http_status_error(status: StatusCode, url: &str, body: Option<&str>) -> String {
    let error_kind = if status.is_client_error() {
        "client error"
    } else if status.is_server_error() {
        "server error"
    } else {
        "error"
    };

    let message = format!("HTTP status {} ({}) for url ({})", error_kind, status, url);

    match body {
        Some(body) => format!("{}\n\n{}\n", message, body),
        None => message,
    }
}

fn summarize_http_error_body(body: Option<&str>) -> Option<String> {
    let trimmed = body?.trim();
    if trimmed.is_empty() {
        return None;
    }

    const MAX_BODY_CHARS: usize = 256;
    let mut summary = trimmed.chars().take(MAX_BODY_CHARS).collect::<String>();

    if trimmed.chars().count() > MAX_BODY_CHARS {
        summary.push_str("...");
    }

    Some(summary)
}

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_http_status_error_without_body() {
        let message = format_http_status_error(
            StatusCode::PRECONDITION_FAILED,
            "https://registry.ollama.ai/v2/library/qwen3.5/manifests/35b-a3b-coding-nvfp4",
            None,
        );

        assert_eq!(
            message,
            "HTTP status client error (412 Precondition Failed) for url (https://registry.ollama.ai/v2/library/qwen3.5/manifests/35b-a3b-coding-nvfp4)"
        );
    }

    #[test]
    fn test_format_http_status_error_with_body() {
        let message = format_http_status_error(
            StatusCode::PRECONDITION_FAILED,
            "https://registry.ollama.ai/v2/library/qwen3.5/manifests/35b-a3b-coding-nvfp4",
            Some("this model requires macOS"),
        );

        assert_eq!(
            message,
            "HTTP status client error (412 Precondition Failed) for url (https://registry.ollama.ai/v2/library/qwen3.5/manifests/35b-a3b-coding-nvfp4)\n\nthis model requires macOS\n"
        );
    }

    #[test]
    fn test_summarize_http_error_body_trims_and_truncates() {
        let long_body = format!("  {}  ", "x".repeat(300));
        let summary = summarize_http_error_body(Some(&long_body)).unwrap();

        assert_eq!(summary.len(), 259);
        assert!(summary.ends_with("..."));
        assert!(!summary.starts_with(' '));
    }

    #[test]
    fn test_summarize_http_error_body_drops_blank_body() {
        assert_eq!(summarize_http_error_body(None), None);
        assert_eq!(summarize_http_error_body(Some("   \n\t  ")), None);
    }
}
