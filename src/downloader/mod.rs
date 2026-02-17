//! Downloaders for the Ollama Downloader in Rust (ODIR),
//! including implementations for Hugging Face and Ollama library models,
//! as well as utility functions for downloading and managing model files.
pub mod hf_downloader;
pub mod manifest;
pub mod model_downloader;
pub mod ollama_downloader;
pub mod utils;

pub use hf_downloader::HuggingFaceModelDownloader;
pub use model_downloader::ModelDownloader;
pub use ollama_downloader::OllamaModelDownloader;
