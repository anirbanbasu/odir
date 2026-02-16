use log::LevelFilter;
use std::env;

/// Configuration for the ODIR application loaded from environment variables.
///
/// Supports both ODIR_* and OD_* prefixes (for compatibility with Python version).
/// When both are present, ODIR_* takes precedence.
#[derive(Debug, Clone)]
pub struct Config {
    /// Log level for the application (default: INFO)
    pub log_level: LevelFilter,

    /// Path to the settings file (default: odir-settings.json)
    pub settings_file: String,

    /// User agent string for HTTP requests (default: odir/<version>)
    pub user_agent: String,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            log_level: LevelFilter::Info,
            settings_file: "odir-settings.json".to_string(),
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

        // Load settings file from ODIR_SETTINGS_FILE or OD_SETTINGS_FILE
        if let Ok(path) = Self::get_env_with_fallback("ODIR_SETTINGS_FILE", "OD_SETTINGS_FILE") {
            config.settings_file = path;
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = Config::default();
        assert_eq!(config.log_level, LevelFilter::Info);
        assert_eq!(config.settings_file, "odir-settings.json");
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
}
