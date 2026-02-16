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
/// Based on: https://distribution.github.io/distribution/spec/manifest-v2-2/#image-manifest
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
