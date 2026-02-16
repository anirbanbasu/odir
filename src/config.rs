use directories::ProjectDirs;
use log::{LevelFilter, info};
use serde::{Deserialize, Serialize};
use std::env;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

/// Settings for connecting to the Ollama server.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OllamaServer {
    /// URL of the Ollama server.
    pub url: String,

    /// API key for the Ollama server, if required.
    pub api_key: Option<String>,

    /// Whether to remove downloaded files if the downloaded model cannot be found
    /// on the Ollama server, or the Ollama server cannot be accessed.
    pub remove_downloaded_on_error: bool,
}

impl Default for OllamaServer {
    fn default() -> Self {
        Self {
            url: "http://localhost:11434/".to_string(),
            api_key: None,
            remove_downloaded_on_error: true,
        }
    }
}

/// Settings for accessing the Ollama library and storing models locally.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OllamaLibrary {
    /// Path to the Ollama models on the filesystem.
    pub models_path: String,

    /// URL of the remote registry for Ollama models.
    pub registry_base_url: String,

    /// Base URL for the Ollama library.
    pub library_base_url: String,

    /// Whether to verify SSL certificates.
    pub verify_ssl: bool,

    /// Timeout for HTTP requests in seconds.
    pub timeout: f64,

    /// Username and group that should own the Ollama models path.
    pub user_group: Option<(String, String)>,
}

impl Default for OllamaLibrary {
    fn default() -> Self {
        Self {
            models_path: "~/.ollama/models".to_string(),
            registry_base_url: "https://registry.ollama.ai/v2/library/".to_string(),
            library_base_url: "https://ollama.com/library/".to_string(),
            verify_ssl: true,
            timeout: 120.0,
            user_group: None,
        }
    }
}

/// Application settings for the Ollama Downloader.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct AppSettings {
    /// Settings for the Ollama server connection.
    pub ollama_server: OllamaServer,

    /// Settings for accessing the Ollama library and storing locally.
    pub ollama_library: OllamaLibrary,
}

impl AppSettings {
    /// Load settings from the configuration file, or create default settings if the file does not exist.
    ///
    /// # Arguments
    /// * `settings_file` - Path to the settings file
    ///
    /// # Returns
    /// * `Result<Self, io::Error>` - The loaded or created settings
    pub fn load_or_create_default<P: AsRef<Path>>(settings_file: P) -> io::Result<Self> {
        match Self::load_settings(&settings_file) {
            Ok(settings) => Ok(settings),
            Err(e) if e.kind() == io::ErrorKind::NotFound => {
                info!(
                    "Settings file '{}' not found, creating with default values",
                    settings_file.as_ref().display()
                );
                let settings = Self::default();
                settings.save_settings(&settings_file)?;
                Ok(settings)
            }
            Err(e) => Err(e),
        }
    }

    /// Load settings from the configuration file.
    ///
    /// # Arguments
    /// * `settings_file` - Path to the settings file
    ///
    /// # Returns
    /// * `Result<Self, io::Error>` - The loaded settings or an error
    pub fn load_settings<P: AsRef<Path>>(settings_file: P) -> io::Result<Self> {
        let content = fs::read_to_string(settings_file)?;
        serde_json::from_str(&content).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
    }

    /// Save the application settings to the configuration file.
    ///
    /// # Arguments
    /// * `settings_file` - Path to the settings file
    ///
    /// # Returns
    /// * `Result<(), io::Error>` - Success or error
    pub fn save_settings<P: AsRef<Path>>(&self, settings_file: P) -> io::Result<()> {
        let settings_path = settings_file.as_ref();
        // Create parent directory if it doesn't exist
        if let Some(parent) = settings_path.parent()
            && !parent.as_os_str().is_empty()
            && !parent.exists()
        {
            fs::create_dir_all(parent)?;
        }

        let json = serde_json::to_string_pretty(self)?;
        fs::write(settings_path, json)?;
        Ok(())
    }
}

/// Configuration for the ODIR application loaded from environment variables.
///
/// Supports both ODIR_* and OD_* prefixes (for compatibility with Python version).
/// When both are present, ODIR_* takes precedence.
#[derive(Debug, Clone)]
pub struct Config {
    /// Log level for the application (default: INFO)
    pub log_level: LevelFilter,

    /// User agent string for HTTP requests (default: odir/<version>)
    pub user_agent: String,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            log_level: LevelFilter::Info,
            user_agent: format!("odir/{}", env!("CARGO_PKG_VERSION")),
        }
    }
}

impl Config {
    /// Load configuration from environment variables.
    ///
    /// Checks both ODIR_* and OD_* prefixes, with ODIR_* taking precedence.
    pub fn from_env() -> Self {
        let mut config = Self::default();

        // Load log level from ODIR_LOG_LEVEL or OD_LOG_LEVEL
        if let Ok(level) = Self::get_env_with_fallback("ODIR_LOG_LEVEL", "OD_LOG_LEVEL") {
            config.log_level = Self::parse_log_level(&level);
        }

        // Load user agent from ODIR_UA_NAME_VER or OD_UA_NAME_VER
        if let Ok(ua) = Self::get_env_with_fallback("ODIR_UA_NAME_VER", "OD_UA_NAME_VER") {
            config.user_agent = ua;
        }

        config
    }

    /// Get environment variable with fallback to alternative name.
    /// Primary takes precedence over fallback.
    fn get_env_with_fallback(primary: &str, fallback: &str) -> Result<String, env::VarError> {
        env::var(primary).or_else(|_| env::var(fallback))
    }

    /// Parse log level string to LevelFilter.
    ///
    /// Supports: TRACE, DEBUG, INFO, WARN, ERROR, OFF (case-insensitive)
    fn parse_log_level(level: &str) -> LevelFilter {
        match level.to_uppercase().as_str() {
            "TRACE" => LevelFilter::Trace,
            "DEBUG" => LevelFilter::Debug,
            "INFO" => LevelFilter::Info,
            "WARN" | "WARNING" => LevelFilter::Warn,
            "ERROR" => LevelFilter::Error,
            "OFF" => LevelFilter::Off,
            _ => {
                eprintln!("Warning: Invalid log level '{}', using INFO", level);
                LevelFilter::Info
            }
        }
    }
}

/// Get the path to the settings file using OS-standard user config directories.
///
/// Returns the path to `settings.json` in the user's config directory.
/// On Linux: `~/.config/odir/settings.json`
/// On macOS: `~/Library/Application Support/odir/settings.json`
/// On Windows: `C:\Users\<user>\AppData\Roaming\odir\settings.json`
///
/// Creates the config directory if it doesn't exist.
///
/// # Returns
/// * `PathBuf` - Path to the settings file
///
/// # Panics
/// Panics if the config directory cannot be determined or created.
pub fn get_settings_file_path() -> PathBuf {
    let proj_dirs =
        ProjectDirs::from("", "", "odir").expect("Failed to determine config directory");

    let config_dir = proj_dirs.config_dir();

    // Create the directory if it doesn't exist
    if let Err(e) = fs::create_dir_all(config_dir) {
        panic!("Failed to create config directory: {}", e);
    }

    config_dir.join("settings.json")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn test_default_config() {
        let config = Config::default();
        assert_eq!(config.log_level, LevelFilter::Info);
        assert!(config.user_agent.starts_with("odir/"));
    }

    #[test]
    fn test_parse_log_level() {
        assert_eq!(Config::parse_log_level("DEBUG"), LevelFilter::Debug);
        assert_eq!(Config::parse_log_level("debug"), LevelFilter::Debug);
        assert_eq!(Config::parse_log_level("INFO"), LevelFilter::Info);
        assert_eq!(Config::parse_log_level("WARN"), LevelFilter::Warn);
        assert_eq!(Config::parse_log_level("WARNING"), LevelFilter::Warn);
        assert_eq!(Config::parse_log_level("ERROR"), LevelFilter::Error);
        assert_eq!(Config::parse_log_level("TRACE"), LevelFilter::Trace);
        assert_eq!(Config::parse_log_level("OFF"), LevelFilter::Off);
        assert_eq!(Config::parse_log_level("invalid"), LevelFilter::Info);
    }

    #[test]
    fn test_default_ollama_server() {
        let server = OllamaServer::default();
        assert_eq!(server.url, "http://localhost:11434/");
        assert_eq!(server.api_key, None);
        assert_eq!(server.remove_downloaded_on_error, true);
    }

    #[test]
    fn test_default_ollama_library() {
        let library = OllamaLibrary::default();
        assert_eq!(library.models_path, "~/.ollama/models");
        assert_eq!(
            library.registry_base_url,
            "https://registry.ollama.ai/v2/library/"
        );
        assert_eq!(library.library_base_url, "https://ollama.com/library/");
        assert_eq!(library.verify_ssl, true);
        assert_eq!(library.timeout, 120.0);
        assert_eq!(library.user_group, None);
    }

    #[test]
    fn test_default_app_settings() {
        let settings = AppSettings::default();
        assert_eq!(settings.ollama_server.url, "http://localhost:11434/");
        assert_eq!(settings.ollama_library.models_path, "~/.ollama/models");
    }

    #[test]
    fn test_app_settings_serialization() {
        let settings = AppSettings::default();
        let json = serde_json::to_string_pretty(&settings).unwrap();
        assert!(json.contains("ollama_server"));
        assert!(json.contains("ollama_library"));
        assert!(json.contains("http://localhost:11434/"));
    }

    #[test]
    fn test_app_settings_deserialization() {
        let json = r#"{
            "ollama_server": {
                "url": "http://test:8080/",
                "api_key": null,
                "remove_downloaded_on_error": true
            },
            "ollama_library": {
                "models_path": "/test/path",
                "registry_base_url": "https://registry.test.com/",
                "library_base_url": "https://library.test.com/",
                "verify_ssl": false,
                "timeout": 60.0,
                "user_group": null
            }
        }"#;

        let settings: AppSettings = serde_json::from_str(json).unwrap();
        assert_eq!(settings.ollama_server.url, "http://test:8080/");
        assert_eq!(settings.ollama_library.models_path, "/test/path");
        assert_eq!(settings.ollama_library.verify_ssl, false);
        assert_eq!(settings.ollama_library.timeout, 60.0);
    }

    #[test]
    fn test_save_and_load_settings() {
        let test_file = "target/test_settings.json";

        // Clean up any existing file
        let _ = fs::remove_file(test_file);

        // Create and save settings
        let settings = AppSettings::default();
        settings.save_settings(test_file).unwrap();

        // Verify file exists
        assert!(Path::new(test_file).exists());

        // Load settings
        let loaded = AppSettings::load_settings(test_file).unwrap();
        assert_eq!(loaded.ollama_server.url, settings.ollama_server.url);
        assert_eq!(
            loaded.ollama_library.models_path,
            settings.ollama_library.models_path
        );

        // Clean up
        fs::remove_file(test_file).unwrap();
    }

    #[test]
    fn test_load_or_create_default() {
        let test_file = "target/test_load_or_create.json";

        // Clean up any existing file
        let _ = fs::remove_file(test_file);

        // Should create file with defaults when it doesn't exist
        let settings = AppSettings::load_or_create_default(test_file).unwrap();
        assert_eq!(settings.ollama_server.url, "http://localhost:11434/");
        assert!(Path::new(test_file).exists());

        // Modify and save
        let mut modified = settings.clone();
        modified.ollama_server.url = "http://modified:9999/".to_string();
        modified.save_settings(test_file).unwrap();

        // Should load existing file
        let loaded = AppSettings::load_or_create_default(test_file).unwrap();
        assert_eq!(loaded.ollama_server.url, "http://modified:9999/");

        // Clean up
        fs::remove_file(test_file).unwrap();
    }

    #[test]
    fn test_save_settings_creates_directory() {
        let test_file = "target/test_subdir/nested/settings.json";

        // Clean up any existing directory
        let _ = fs::remove_dir_all("target/test_subdir");

        // Save should create parent directories
        let settings = AppSettings::default();
        settings.save_settings(test_file).unwrap();

        assert!(Path::new(test_file).exists());

        // Clean up
        fs::remove_dir_all("target/test_subdir").unwrap();
    }

    #[test]
    fn test_load_settings_file_not_found() {
        let result = AppSettings::load_settings("nonexistent_file.json");
        assert!(result.is_err());
        assert_eq!(result.unwrap_err().kind(), io::ErrorKind::NotFound);
    }

    #[test]
    fn test_load_settings_invalid_json() {
        let test_file = "target/test_invalid.json";
        fs::write(test_file, "{ invalid json }").unwrap();

        let result = AppSettings::load_settings(test_file);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err().kind(), io::ErrorKind::InvalidData);

        fs::remove_file(test_file).unwrap();
    }
}
