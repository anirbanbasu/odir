//! Data models for the Ollama Downloader in Rust (ODIR),
//! including the image manifest structure based on the OCI Image Manifest specification.
use serde::{Deserialize, Serialize};

/// Configuration section of the image manifest
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ImageManifestConfig {
    /// The media type of the image manifest configuration
    pub media_type: String,

    /// The size of the image manifest configuration in bytes
    pub size: u64,

    /// The digest of the image manifest configuration, used for content addressing
    pub digest: String,
}

/// A single layer entry in the image manifest
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ImageManifestLayerEntry {
    /// The media type of the layer
    pub media_type: String,

    /// The size of the layer in bytes
    pub size: u64,

    /// The digest of the layer, used for content addressing
    pub digest: String,

    /// Optional list of URLs where the layer can be downloaded from
    #[serde(skip_serializing_if = "Option::is_none")]
    pub urls: Option<Vec<String>>,
}

/// Data model representing an Ollama image manifest
/// Based on: [Image Manifest specification](https://distribution.github.io/distribution/spec/manifest-v2-2/#image-manifest)
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ImageManifest {
    /// The schema version of the image manifest
    pub schema_version: u32,

    /// The media type of the image manifest
    pub media_type: String,

    /// Configuration for the image manifest
    pub config: ImageManifestConfig,

    /// List of layers in the image manifest
    #[serde(skip_serializing_if = "Option::is_none")]
    pub layers: Option<Vec<ImageManifestLayerEntry>>,
}

/// Supported source types for model downloads.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum DownloadSourceType {
    Ollama,
    Hf,
}

/// Per-item download state persisted in the advisory download journal.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum JournalItemState {
    Pending,
    Completed,
    Failed,
}

/// One digest entry tracked in the advisory download journal.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DownloadJournalItem {
    pub digest: String,
    pub media_type: String,
    pub size: u64,
    pub state: JournalItemState,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub last_error: Option<String>,
}

/// Advisory journal for a specific model download.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DownloadJournal {
    pub model_identifier: String,
    pub source_type: DownloadSourceType,
    pub tag_or_quant: String,
    pub started_at: u64,
    pub updated_at: u64,
    pub items: Vec<DownloadJournalItem>,
}

/// Compact journal metadata used by list output.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DownloadJournalListEntry {
    pub model_identifier: String,
    pub source_type: DownloadSourceType,
    pub tag_or_quant: String,
    pub updated_at: u64,
    pub item_count: usize,
}
