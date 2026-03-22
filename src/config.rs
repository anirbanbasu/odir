//! Configuration management for the Ollama Downloader in Rust (ODIR).
use directories::ProjectDirs;
use log::{LevelFilter, info, warn};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::env;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use thiserror::Error;
use url::{ParseError, Url};

/// Error type for HTTP URL validation.
#[derive(PartialEq, Debug, Error)]
pub enum HttpUrlParseError {
    /// URL parsing failed.
    #[error("URL parsing failed: {0}")]
    ParseError(#[from] ParseError),

    /// URL scheme is not http or https.
    #[error("URL scheme should either be http or https, got: {0}")]
    InvalidScheme(String),
}

pub fn validate_string_as_http_url(url_str: &str) -> Result<Url, HttpUrlParseError> {
    let url = Url::parse(url_str)?;
    if url.scheme() != "http" && url.scheme() != "https" {
        return Err(HttpUrlParseError::InvalidScheme(url.scheme().to_string()));
    }
    Ok(url)
}

/// Settings for connecting to the Ollama server.
#[derive(PartialEq, Debug, Clone, Serialize, Deserialize)]
pub struct OllamaServer {
    /// URL of the Ollama server.
    pub url: String,

    /// API key for the Ollama server, if required.
    pub api_key: Option<String>,

    /// Whether to remove downloaded files if the downloaded model cannot be found
    /// on the Ollama server, or the Ollama server cannot be accessed.
    pub remove_downloaded_on_error: bool,

    /// Whether to check if the model is present in the Ollama server after downloading.
    pub check_model_presence: bool,
}

impl Default for OllamaServer {
    fn default() -> Self {
        Self {
            url: "http://localhost:11434/".to_string(),
            api_key: None,
            remove_downloaded_on_error: true,
            check_model_presence: true,
        }
    }
}

/// Settings for accessing the Ollama library and storing models locally.
#[derive(PartialEq, Debug, Clone, Serialize, Deserialize)]
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
}

impl Default for OllamaLibrary {
    fn default() -> Self {
        Self {
            models_path: "~/.ollama/models".to_string(),
            registry_base_url: "https://registry.ollama.ai/v2/library/".to_string(),
            library_base_url: "https://ollama.com/library/".to_string(),
            verify_ssl: true,
            timeout: 120.0,
        }
    }
}

/// Application settings for the Ollama Downloader.
#[derive(PartialEq, Debug, Clone, Serialize, Deserialize, Default)]
pub struct AppSettings {
    /// Settings for the Ollama server for which the models should be downloaded.
    pub ollama_server: OllamaServer,

    /// Settings for accessing the Ollama library and storing locally.
    pub ollama_library: OllamaLibrary,
}

impl AppSettings {
    /// Validate all HTTP URLs in the settings.
    ///
    /// # Returns
    /// * `Result<(), HttpUrlParseError>` - Success or validation error
    pub fn validate_urls(&self) -> Result<(), HttpUrlParseError> {
        validate_string_as_http_url(&self.ollama_server.url)?;
        validate_string_as_http_url(&self.ollama_library.registry_base_url)?;
        validate_string_as_http_url(&self.ollama_library.library_base_url)?;
        Ok(())
    }

    /// Load settings from the configuration file, or create default settings if the file does not exist.
    /// If the file exists but has validation errors, attempts to repair it with defaults.
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
                    "Settings file '{}' not found, creating one with default values",
                    settings_file.as_ref().display()
                );
                let settings = Self::default();
                settings
                    .validate_urls()
                    .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))?;
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
        match serde_json::from_str::<AppSettings>(&content) {
            Ok(settings) => {
                settings.validate_urls().map_err(|e: HttpUrlParseError| {
                    io::Error::new(io::ErrorKind::InvalidData, e.to_string())
                })?;
                Ok(settings)
            }
            Err(e) => {
                // If deserialization fails, try lenient loading with defaults
                warn!(
                    "Strict deserialization failed: {}. Attempting to load with defaults...",
                    e
                );
                Self::load_settings_lenient(&content).map_err(|lenient_err| {
                    io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!(
                            "Failed strict deserialization: {}. Failed lenient deserialization: {}",
                            e, lenient_err
                        ),
                    )
                })
            }
        }
    }

    /// Load settings from JSON content with lenient deserialization.
    /// Missing fields will be replaced with defaults, and warnings will be issued.
    ///
    /// # Arguments
    /// * `content` - The JSON content as a string
    ///
    /// # Returns
    /// * `Result<Self, String>` - The loaded settings with defaults, or error message
    fn load_settings_lenient(content: &str) -> Result<Self, String> {
        let mut parsed: Value =
            serde_json::from_str(content).map_err(|e| format!("Invalid JSON: {}", e))?;

        // Get or create the ollama_server object
        let mut ollama_server = parsed
            .get_mut("ollama_server")
            .and_then(|v| v.as_object_mut())
            .map(|obj| obj.clone())
            .unwrap_or_default();

        // Fill in missing ollama_server fields with defaults
        let defaults = OllamaServer::default();
        if !ollama_server.contains_key("url") {
            warn!(
                "Missing field 'ollama_server.url', using default: {}",
                defaults.url
            );
            ollama_server.insert("url".to_string(), Value::String(defaults.url));
        }
        if !ollama_server.contains_key("api_key") {
            warn!("Missing field 'ollama_server.api_key', using default: None");
            ollama_server.insert("api_key".to_string(), Value::Null);
        }
        if !ollama_server.contains_key("remove_downloaded_on_error") {
            warn!(
                "Missing field 'ollama_server.remove_downloaded_on_error', using default: {}",
                defaults.remove_downloaded_on_error
            );
            ollama_server.insert(
                "remove_downloaded_on_error".to_string(),
                Value::Bool(defaults.remove_downloaded_on_error),
            );
        }
        if !ollama_server.contains_key("check_model_presence") {
            warn!(
                "Missing field 'ollama_server.check_model_presence', using default: {}",
                defaults.check_model_presence
            );
            ollama_server.insert(
                "check_model_presence".to_string(),
                Value::Bool(defaults.check_model_presence),
            );
        }

        // Get or create the ollama_library object
        let mut ollama_library = parsed
            .get_mut("ollama_library")
            .and_then(|v| v.as_object_mut())
            .map(|obj| obj.clone())
            .unwrap_or_default();

        // Fill in missing ollama_library fields with defaults
        let defaults = OllamaLibrary::default();
        if !ollama_library.contains_key("models_path") {
            warn!(
                "Missing field 'ollama_library.models_path', using default: {}",
                defaults.models_path
            );
            ollama_library.insert(
                "models_path".to_string(),
                Value::String(defaults.models_path),
            );
        }
        if !ollama_library.contains_key("registry_base_url") {
            warn!(
                "Missing field 'ollama_library.registry_base_url', using default: {}",
                defaults.registry_base_url
            );
            ollama_library.insert(
                "registry_base_url".to_string(),
                Value::String(defaults.registry_base_url),
            );
        }
        if !ollama_library.contains_key("library_base_url") {
            warn!(
                "Missing field 'ollama_library.library_base_url', using default: {}",
                defaults.library_base_url
            );
            ollama_library.insert(
                "library_base_url".to_string(),
                Value::String(defaults.library_base_url),
            );
        }
        if !ollama_library.contains_key("verify_ssl") {
            warn!(
                "Missing field 'ollama_library.verify_ssl', using default: {}",
                defaults.verify_ssl
            );
            ollama_library.insert("verify_ssl".to_string(), Value::Bool(defaults.verify_ssl));
        }
        if !ollama_library.contains_key("timeout") {
            warn!(
                "Missing field 'ollama_library.timeout', using default: {}",
                defaults.timeout
            );
            ollama_library.insert(
                "timeout".to_string(),
                Value::Number(serde_json::Number::from_f64(defaults.timeout).unwrap()),
            );
        }

        // Reconstruct the settings object with filled-in values
        let settings_object = json!({
            "ollama_server": ollama_server,
            "ollama_library": ollama_library,
        });

        let settings: AppSettings = serde_json::from_value(settings_object)
            .map_err(|e| format!("Failed to deserialize settings with defaults: {}", e))?;

        // Validate all URLs
        settings
            .validate_urls()
            .map_err(|e| format!("URL validation failed: {}", e))?;

        Ok(settings)
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
}

impl Default for Config {
    fn default() -> Self {
        Self {
            log_level: LevelFilter::Info,
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
/// * `Result<PathBuf, io::Error>` - Path to the settings file or error
pub fn get_settings_file_path() -> Result<PathBuf, io::Error> {
    let proj_dirs =
        ProjectDirs::from("", "", "odir").expect("Failed to determine config directory");

    get_settings_file_path_for_dir(proj_dirs.config_dir())
}

/// Get the path to the settings file and panic on error.
///
/// # Panics
/// Panics if the config directory cannot be determined or created.
pub fn get_settings_file_path_or_panic() -> PathBuf {
    settings_path_or_panic(get_settings_file_path())
}

fn get_settings_file_path_for_dir(config_dir: &Path) -> Result<PathBuf, io::Error> {
    fs::create_dir_all(config_dir)?;
    Ok(config_dir.join("settings.json"))
}

fn settings_path_or_panic(result: Result<PathBuf, io::Error>) -> PathBuf {
    result.unwrap_or_else(|e| panic!("Failed to create config directory: {}", e))
}

/// Get the user agent string for HTTP requests.
///
/// Returns a string in the format "odir/{version}".
///
/// # Returns
/// * `String` - User agent string
pub fn get_user_agent() -> String {
    format!("odir/{}", env!("CARGO_PKG_VERSION"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::tempdir;

    /// Initialize logger for tests to enable log coverage
    fn init_test_logger() {
        let _ = env_logger::builder()
            .is_test(true)
            .filter_level(log::LevelFilter::Trace)
            .try_init();
    }

    #[test]
    fn test_validate_string_as_http_url() {
        assert!(validate_string_as_http_url("http://example.tld").is_ok());
        assert!(validate_string_as_http_url("https://example.tld").is_ok());
        assert!(validate_string_as_http_url("https://example.tld/path?query=string").is_ok());

        assert!(validate_string_as_http_url("ftp://example.tld").is_err());
        assert!(validate_string_as_http_url("example.tld").is_err());
        assert!(validate_string_as_http_url("http://").is_err());
        assert!(validate_string_as_http_url("hello world").is_err());
    }

    #[test]
    fn test_default_config() {
        let config = Config::default();
        assert_eq!(config.log_level, LevelFilter::Info);
    }

    #[test]
    fn test_get_user_agent() {
        let user_agent = get_user_agent();
        assert!(user_agent.starts_with("odir/"));
    }

    #[test]
    fn test_parse_log_level() {
        assert_eq!(Config::parse_log_level("DEBUG"), LevelFilter::Debug);
        assert_eq!(Config::parse_log_level("debug"), LevelFilter::Debug);
        assert_eq!(Config::parse_log_level("INFO"), LevelFilter::Info);
        assert_eq!(Config::parse_log_level("info"), LevelFilter::Info);
        assert_eq!(Config::parse_log_level("WARN"), LevelFilter::Warn);
        assert_eq!(Config::parse_log_level("warn"), LevelFilter::Warn);
        assert_eq!(Config::parse_log_level("WARNING"), LevelFilter::Warn);
        assert_eq!(Config::parse_log_level("warning"), LevelFilter::Warn);
        assert_eq!(Config::parse_log_level("ERROR"), LevelFilter::Error);
        assert_eq!(Config::parse_log_level("error"), LevelFilter::Error);
        assert_eq!(Config::parse_log_level("TRACE"), LevelFilter::Trace);
        assert_eq!(Config::parse_log_level("trace"), LevelFilter::Trace);
        assert_eq!(Config::parse_log_level("OFF"), LevelFilter::Off);
        assert_eq!(Config::parse_log_level("off"), LevelFilter::Off);
        assert_eq!(Config::parse_log_level("invalid"), LevelFilter::Info);
        assert_eq!(Config::parse_log_level("something else"), LevelFilter::Info);
    }

    #[test]
    fn test_from_env_cleanup_removes_vars() {
        unsafe {
            env::set_var("OD_LOG_LEVEL", "debug");
            env::set_var("ODIR_LOG_LEVEL", "warn");
        }

        let config = Config::from_env();
        assert_eq!(config.log_level, LevelFilter::Warn);

        unsafe {
            env::remove_var("ODIR_LOG_LEVEL");
            env::remove_var("OD_LOG_LEVEL");
        }
    }

    #[test]
    fn test_default_ollama_server() {
        let server = OllamaServer::default();
        assert_eq!(server.url, "http://localhost:11434/");
        assert_eq!(server.api_key, None);
        assert_eq!(server.remove_downloaded_on_error, true);
        assert_eq!(server.check_model_presence, true);
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
        init_test_logger();
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
    fn test_load_or_create_default_settings_invalid_json() {
        init_test_logger();
        let test_file = "target/";

        let result = AppSettings::load_or_create_default(test_file);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err().kind(), io::ErrorKind::IsADirectory);
    }

    #[test]
    fn test_load_settings_invalid_json() {
        init_test_logger();
        let test_file = "target/test_invalid.json";
        fs::write(test_file, "{ invalid json }").unwrap();

        let result = AppSettings::load_settings(test_file);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err().kind(), io::ErrorKind::InvalidData);

        fs::remove_file(test_file).unwrap();
    }

    #[test]
    fn test_load_settings_with_missing_fields() {
        init_test_logger();
        let test_file = "target/test_missing_fields.json";
        let json_with_missing_fields = r#"{
            "ollama_server": {
            },
            "ollama_library": {
            }
        }"#;
        fs::write(test_file, json_with_missing_fields).unwrap();

        // Should successfully load with lenient deserialization
        let result = AppSettings::load_settings(test_file);
        assert!(result.is_ok());

        let settings = result.unwrap();
        // Check that provided values are preserved
        assert_eq!(settings.ollama_server.url, "http://localhost:11434/");
        assert_eq!(settings.ollama_server.remove_downloaded_on_error, true);
        assert_eq!(settings.ollama_library.models_path, "~/.ollama/models");

        // Check that missing values use defaults
        assert_eq!(settings.ollama_server.api_key, None);
        assert_eq!(settings.ollama_server.check_model_presence, true); // default
        assert_eq!(
            settings.ollama_library.registry_base_url,
            "https://registry.ollama.ai/v2/library/"
        ); // default
        assert_eq!(
            settings.ollama_library.library_base_url,
            "https://ollama.com/library/"
        ); // default
        assert_eq!(settings.ollama_library.verify_ssl, true); // default
        assert_eq!(settings.ollama_library.timeout, 120.0); // default

        fs::remove_file(test_file).unwrap();
    }

    #[test]
    fn test_load_settings_lenient_with_extra_fields() {
        let test_file = "target/test_extra_fields.json";
        let json_with_extra_fields = r#"{
            "ollama_server": {
                "url": "http://test:9000/",
                "api_key": "test_key",
                "remove_downloaded_on_error": true,
                "check_model_presence": false,
                "extra_field": "should_be_ignored"
            },
            "ollama_library": {
                "models_path": "/test",
                "registry_base_url": "https://localhost/registry",
                "library_base_url": "https://test.lib",
                "timeout": 60.0,
                "unknown_field": 123
            }
        }"#;
        fs::write(test_file, json_with_extra_fields).unwrap();

        // Should successfully load even with extra unknown fields
        let result = AppSettings::load_settings(test_file);
        assert!(result.is_ok());

        let settings = result.unwrap();
        assert_eq!(settings.ollama_server.url, "http://test:9000/");
        assert_eq!(settings.ollama_server.api_key, Some("test_key".to_string()));
        assert_eq!(settings.ollama_server.check_model_presence, false);
        assert_eq!(settings.ollama_library.timeout, 60.0);
        assert_eq!(settings.ollama_library.verify_ssl, true); // default for missing field

        fs::remove_file(test_file).unwrap();
    }

    #[test]
    fn test_load_settings_strict_with_http_url_errors() {
        init_test_logger();
        let test_file = "target/test_http_url_errors.json";
        let json_with_http_url_errors = r#"{
            "ollama_server": {
                "url": "ftp://test:9000/",
                "api_key": null,
                "remove_downloaded_on_error": true,
                "check_model_presence": false
            },
            "ollama_library": {
                "models_path": "/test",
                "registry_base_url": "https://test",
                "library_base_url": "this-is-not-a-url",
                "verify_ssl": true,
                "timeout": 60.0
            }
        }"#;
        fs::write(test_file, json_with_http_url_errors).unwrap();

        // Should fail to load due to invalid URL scheme
        let result = AppSettings::load_settings(test_file);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err().kind(), io::ErrorKind::InvalidData);

        fs::remove_file(test_file).unwrap();
    }

    #[test]
    fn test_get_settings_file_path() {
        let path = get_settings_file_path().expect("Settings path should be created");
        assert_eq!(
            path.file_name().and_then(|name| name.to_str()),
            Some("settings.json")
        );

        let parent = path
            .parent()
            .expect("Settings path should have a parent directory");
        assert!(parent.exists());
    }

    #[test]
    fn test_get_settings_file_path_error() {
        let temp_dir = tempdir().expect("Temp dir should be created");
        let file_path = temp_dir.path().join("not_a_dir");
        fs::write(&file_path, "not a directory").unwrap();

        let result = get_settings_file_path_for_dir(&file_path);
        assert!(result.is_err());
    }

    #[test]
    fn test_get_settings_file_path_or_panic_success() {
        let path = get_settings_file_path_or_panic();
        assert_eq!(
            path.file_name().and_then(|name| name.to_str()),
            Some("settings.json")
        );
        let parent = path
            .parent()
            .expect("Settings path should have a parent directory");
        assert!(parent.exists());
    }

    #[test]
    #[should_panic(expected = "Failed to create config directory:")]
    fn test_get_settings_file_path_or_panic_panic() {
        let temp_dir = tempdir().expect("Temp dir should be created");
        let file_path = temp_dir.path().join("not_a_dir");
        fs::write(&file_path, "not a directory").unwrap();

        let _ = settings_path_or_panic(get_settings_file_path_for_dir(&file_path));
    }
}
