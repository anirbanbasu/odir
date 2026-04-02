#![doc = include_str!("../README.md")]
//! ## Source code
//! The source code is available in the [GitHub repository](
//! https://github.com/anirbanbasu/odir).

use chrono::{Local, TimeZone};
use clap::builder::styling::{AnsiColor, Effects, Styles};
use clap::{Parser, Subcommand, ValueEnum};
use log::{debug, error, info, warn};
use std::io::{self, Write};
use std::path::PathBuf;

mod config;
use config::{AppSettings, Config};

mod downloader;
use downloader::manifest::{DownloadJournalListEntry, DownloadSourceType};
use downloader::utils::{clear_journal_for_model, list_available_journals, load_journal_for_model};
use downloader::{
    DownloaderError, HuggingFaceModelDownloader, ModelDownloader, OllamaModelDownloader,
};

mod signal_handler;

#[doc(hidden)]
const STYLES: Styles = Styles::styled()
    .header(AnsiColor::Green.on_default().effects(Effects::BOLD))
    .usage(AnsiColor::Cyan.on_default().effects(Effects::BOLD))
    .literal(AnsiColor::Blue.on_default().effects(Effects::BOLD))
    .placeholder(AnsiColor::Yellow.on_default())
    .error(AnsiColor::Red.on_default().effects(Effects::BOLD))
    .valid(AnsiColor::Green.on_default())
    .invalid(AnsiColor::Red.on_default());

/// A command-line interface for the Ollama Downloader in Rust (ODIR), which is a Rust port and successor of the Python-based [Ollama Downloader](https://github.com/anirbanbasu/ollama-downloader).
#[derive(Parser)]
#[command(name = "odir")]
#[command(version, about)]
// #[command(disable_help_subcommand = false)]
#[command(styles = STYLES)]
// #[command(override_usage = "odir <COMMAND> [OPTIONS]... [ARGS]...")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

/// The available commands for the Ollama Downloader in Rust (ODIR) command-line application.
#[derive(Subcommand)]
enum Commands {
    #[command(subcommand_help_heading = "Configuration")]
    /// Shows the application configuration as JSON.
    ShowConfig,

    #[command(subcommand_help_heading = "Configuration")]
    /// Interactively edits application settings through step-by-step questions.
    ///
    /// If a settings file already exists, the current values will be shown as defaults.
    /// Otherwise, the default configuration values will be used.
    EditConfig {
        /// Optional configuration file path to edit.
        /// If not provided, uses the default user settings location.
        #[arg(long, short)]
        config_file: Option<String>,
    },

    #[command(subcommand_help_heading = "Ollama Library")]
    /// Lists all available models in the Ollama library.
    ///
    /// If pagination options are not provided, all models will be listed.
    ListModels {
        /// The page number to retrieve (1-indexed).
        #[arg(long)]
        page: Option<u32>,

        /// The number of models to retrieve per page.
        #[arg(long)]
        page_size: Option<u32>,
    },

    #[command(subcommand_help_heading = "Ollama Library")]
    /// Lists all tags for a specific model.
    ListTags {
        /// The name of the model to list tags for, e.g., llama3.1.
        model_identifier: String,
    },

    #[command(subcommand_help_heading = "Ollama Library")]
    /// Downloads a specific Ollama model with the given tag.
    ModelDownload {
        /// The name of the model and a specific tag to download, specified as {model}:{tag},
        /// e.g., llama3.1:8b. If no tag is specified, 'latest' will be assumed.
        model_tag: String,
    },

    #[command(subcommand_help_heading = "Hugging Face Models")]
    /// Lists available models from Hugging Face that can be downloaded into Ollama.
    HfListModels {
        /// The page number to retrieve (1-indexed).
        #[arg(long, default_value_t = 1)]
        page: u32,

        /// The number of models to retrieve per page.
        #[arg(long, default_value_t = 25)]
        page_size: u32,
    },

    #[command(subcommand_help_heading = "Hugging Face Models")]
    /// Lists all available quantisations as tags for a Hugging Face model that can be downloaded into Ollama.
    ///
    /// Note that these are NOT the same as Hugging Face model tags.
    HfListTags {
        /// The name of the model to list tags for, e.g., bartowski/Llama-3.2-1B-Instruct-GGUF.
        model_identifier: String,
    },

    #[command(subcommand_help_heading = "Hugging Face Models")]
    /// Downloads a specified Hugging Face model.
    HfModelDownload {
        /// The name of the specific Hugging Face model to download, specified as
        /// {username}/{repository}:{quantisation}, e.g., bartowski/Llama-3.2-1B-Instruct-GGUF:Q4_K_M.
        user_repo_quant: String,
    },

    #[command(subcommand_help_heading = "Compatibility")]
    /// Copies a Ollama Downloader settings file to the ODIR settings location.
    OdCopySettings {
        /// Path to the existing Ollama Downloader settings file.
        od_settings_file: String,
    },

    #[command(subcommand_help_heading = "Diagnostics")]
    /// Displays the advisory download journal for a model.
    Journal {
        /// Model identifier used for the download, e.g. llama3.2:8b or user/repo:Q4_K_M.
        #[arg(required_unless_present = "list")]
        model_identifier: Option<String>,

        /// List all available journals.
        #[arg(long)]
        list: bool,

        /// Source type for the journal lookup.
        #[arg(long)]
        source: Option<JournalSourceArg>,

        /// Delete the journal for one specific model (with confirmation).
        #[arg(long, conflicts_with = "list", conflicts_with = "json")]
        clear: bool,

        /// Print machine-readable JSON output.
        #[arg(long)]
        json: bool,
    },
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, ValueEnum)]
enum JournalSourceArg {
    Ollama,
    Hf,
}

impl From<JournalSourceArg> for DownloadSourceType {
    fn from(value: JournalSourceArg) -> Self {
        match value {
            JournalSourceArg::Ollama => DownloadSourceType::Ollama,
            JournalSourceArg::Hf => DownloadSourceType::Hf,
        }
    }
}

fn format_epoch_local_rfc2822(epoch_seconds: u64) -> String {
    match Local.timestamp_opt(epoch_seconds as i64, 0).single() {
        Some(dt) => dt.to_rfc2822(),
        None => format!("{}", epoch_seconds),
    }
}

/// Prompts the user for a string input with a default value.
///
/// # Arguments
/// * `prompt` - The prompt message to display
/// * `default` - The default value if user presses Enter without input
///
/// # Returns
/// * `String` - The user's input or the default value
fn prompt_string(prompt: &str, default: &str) -> String {
    print!("{} [{}]: ", prompt, default);
    io::stdout().flush().unwrap();

    let mut input = String::new();
    io::stdin().read_line(&mut input).unwrap();
    let input = input.trim();

    if input.is_empty() {
        default.to_string()
    } else {
        input.to_string()
    }
}

/// Prompts the user for an optional string input.
///
/// # Arguments
/// * `prompt` - The prompt message to display
///
/// # Returns
/// * `Option<String>` - Some(input) if provided, None if empty
fn prompt_optional_string(prompt: &str) -> Option<String> {
    print!("{} (press Enter to skip): ", prompt);
    io::stdout().flush().unwrap();

    let mut input = String::new();
    io::stdin().read_line(&mut input).unwrap();
    let input = input.trim();

    if input.is_empty() {
        None
    } else {
        Some(input.to_string())
    }
}

/// Prompts the user for a boolean (yes/no) input with a default value.
///
/// # Arguments
/// * `prompt` - The prompt message to display
/// * `default` - The default value if user presses Enter without input
///
/// # Returns
/// * `bool` - The user's selection or the default value
fn prompt_bool(prompt: &str, default: bool) -> bool {
    let default_str = if default { "Y/n" } else { "y/N" };
    print!("{} [{}]: ", prompt, default_str);
    io::stdout().flush().unwrap();

    let mut input = String::new();
    io::stdin().read_line(&mut input).unwrap();
    let input = input.trim().to_lowercase();

    if input.is_empty() {
        default
    } else {
        matches!(input.as_str(), "y" | "yes")
    }
}

/// Prompts the user for a floating-point number with a default value.
///
/// # Arguments
/// * `prompt` - The prompt message to display
/// * `default` - The default value if user presses Enter without input
///
/// # Returns
/// * `f64` - The user's input or the default value
fn prompt_f64(prompt: &str, default: f64) -> f64 {
    loop {
        print!("{} [{}]: ", prompt, default);
        io::stdout().flush().unwrap();

        let mut input = String::new();
        io::stdin().read_line(&mut input).unwrap();
        let input = input.trim();

        if input.is_empty() {
            return default;
        }

        match input.parse::<f64>() {
            Ok(value) => return value,
            Err(_) => {
                println!("Invalid number. Please try again.");
            }
        }
    }
}

/// Prompts the user for a u64 value chosen from a fixed list of allowed values.
fn prompt_u64_choice(prompt: &str, default: u64, allowed: &[u64]) -> u64 {
    loop {
        let choices = allowed
            .iter()
            .map(u64::to_string)
            .collect::<Vec<_>>()
            .join(", ");

        print!("{} [{}] (allowed: {}): ", prompt, default, choices);
        io::stdout().flush().unwrap();

        let mut input = String::new();
        io::stdin().read_line(&mut input).unwrap();
        let input = input.trim();

        if input.is_empty() {
            return default;
        }

        match input.parse::<u64>() {
            Ok(value) if allowed.contains(&value) => return value,
            Ok(_) => println!("Invalid choice. Please select one of: {}", choices),
            Err(_) => println!("Invalid number. Please try again."),
        }
    }
}

/// Interactively configures application settings by prompting the user.
///
/// # Arguments
/// * `existing_settings` - Existing settings to use as defaults, or None for default values
///
/// # Returns
/// * `AppSettings` - The configured settings
fn interactive_config(existing_settings: Option<AppSettings>) -> AppSettings {
    println!("\n=== Interactive Configuration ===\n");

    let has_existing = existing_settings.is_some();
    let mut settings = existing_settings.unwrap_or_default();

    if has_existing {
        println!("Editing existing configuration. Press Enter to keep current values.\n");
    } else {
        println!("Creating new configuration. Press Enter to accept default values.\n");
    }

    // Ollama Server settings
    println!("--- Ollama Server Settings ---");
    let current_url = settings.ollama_server.url.clone();
    settings.ollama_server.url = prompt_string("Ollama server URL", &current_url);

    // For API key, show current value or indicate it's optional
    let current_api_key = settings.ollama_server.api_key.clone();
    if let Some(ref current_key) = current_api_key {
        println!("Ollama API key (current: {})", current_key);
        settings.ollama_server.api_key =
            prompt_optional_string("  Enter new API key or press Enter to keep current");
        if settings.ollama_server.api_key.is_none() {
            settings.ollama_server.api_key = Some(current_key.clone());
        }
    } else {
        settings.ollama_server.api_key = prompt_optional_string("Ollama API key");
    }

    settings.ollama_server.remove_downloaded_on_error = prompt_bool(
        "Remove downloaded files on error?",
        settings.ollama_server.remove_downloaded_on_error,
    );

    settings.ollama_server.check_model_presence = prompt_bool(
        "Check model presence in Ollama server after downloading?",
        settings.ollama_server.check_model_presence,
    );

    settings.ollama_server.keep_verified_blobs_on_error = prompt_bool(
        "Keep verified blobs on error?",
        settings.ollama_server.keep_verified_blobs_on_error,
    );

    // Ollama Library settings
    println!("\n--- Ollama Library Settings ---");
    settings.ollama_library.models_path =
        prompt_string("Ollama models path", &settings.ollama_library.models_path);

    settings.ollama_library.registry_base_url = prompt_string(
        "Ollama registry base URL",
        &settings.ollama_library.registry_base_url,
    );

    settings.ollama_library.library_base_url = prompt_string(
        "Ollama library base URL",
        &settings.ollama_library.library_base_url,
    );

    settings.ollama_library.verify_ssl = prompt_bool(
        "Verify SSL certificates?",
        settings.ollama_library.verify_ssl,
    );

    settings.ollama_library.timeout = prompt_f64(
        "HTTP request timeout (seconds)",
        settings.ollama_library.timeout,
    );

    settings.ollama_library.chunk_size_mib = prompt_u64_choice(
        "Chunk size for large blob downloads (MiB -- 0 for no chunking)",
        settings.ollama_library.chunk_size_mib,
        &[0, 32, 64, 128, 256, 512],
    );

    settings.ollama_library.download_chunks_in_parallel = prompt_bool(
        "Download chunks in parallel?",
        settings.ollama_library.download_chunks_in_parallel,
    );

    settings.ollama_library.transient_cleanup_enabled = prompt_bool(
        "Enable transient artefact cleanup?",
        settings.ollama_library.transient_cleanup_enabled,
    );

    settings.ollama_library.transient_ttl_hours = prompt_u64_choice(
        "Transient artefact TTL (hours)",
        settings.ollama_library.transient_ttl_hours,
        &[0, 24, 48, 72, 96, 168, 336],
    );

    settings.ollama_library.failed_journal_ttl_hours = prompt_u64_choice(
        "Failed journal TTL (hours)",
        settings.ollama_library.failed_journal_ttl_hours,
        &[0, 24, 48, 72, 96, 168, 336],
    );

    settings.ollama_library.completed_journal_ttl_hours = prompt_u64_choice(
        "Completed journal TTL (hours)",
        settings.ollama_library.completed_journal_ttl_hours,
        &[0, 12, 24, 48, 72, 168],
    );

    println!("\n=== Configuration Complete ===\n");
    settings
}

fn handle_show_config() {
    match AppSettings::load_or_create_default(config::get_settings_file_path_or_panic()) {
        Ok(settings) => match serde_json::to_string_pretty(&settings) {
            Ok(json) => {
                println!("{}", json);
                info!(
                    "Settings loaded from {:?}",
                    config::get_settings_file_path_or_panic()
                );
            }
            Err(e) => {
                error!("Failed to serialize settings: {}", e);
                std::process::exit(1);
            }
        },
        Err(e) => {
            error!(
                "Failed to load or create settings file '{:?}': {}",
                config::get_settings_file_path_or_panic(),
                e
            );
            // Provide helpful guidance to the user
            if e.kind() == io::ErrorKind::InvalidData {
                eprintln!("\n⚠ Settings file has validation errors that could not be recovered.");
                eprintln!("  Try running 'odir edit-config' to fix your settings.\n");
            }
            std::process::exit(1);
        }
    }
}

fn handle_edit_config(config_file: Option<String>) {
    // Determine config file path
    let config_path = config_file
        .as_ref()
        .map(PathBuf::from)
        .unwrap_or_else(config::get_settings_file_path_or_panic);

    // Try to load existing settings from the config file
    let existing_settings = if config_path.exists() {
        match AppSettings::load_settings(&config_path) {
            Ok(settings) => {
                info!("Loaded existing settings from: {}", config_path.display());
                Some(settings)
            }
            Err(e) => {
                warn!(
                    "Settings file exists but could not be loaded: {}. Using defaults.",
                    e
                );
                eprintln!(
                    "\n⚠ Settings file has validation errors but default values have been used."
                );
                eprintln!("  Error: {}\n", e);
                None
            }
        }
    } else {
        info!("No existing settings file found. Creating new configuration.");
        None
    };

    // Interactively configure settings
    let settings = interactive_config(existing_settings);

    // Save settings to file
    match settings.save_settings(&config_path) {
        Ok(_) => {
            println!(
                "\n✓ Settings saved successfully to: {}",
                config_path.display()
            );
            info!("Settings saved to: {}", config_path.display());

            // Display the saved settings
            match serde_json::to_string_pretty(&settings) {
                Ok(json) => {
                    println!("\nSaved configuration:");
                    println!("{}", json);
                }
                Err(e) => {
                    warn!("Failed to display saved settings: {}", e);
                }
            }
        }
        Err(e) => {
            error!(
                "Failed to save settings to '{}': {}",
                config_path.display(),
                e
            );
            std::process::exit(1);
        }
    }
}

fn handle_list_models(page: Option<u32>, page_size: Option<u32>) {
    match AppSettings::load_or_create_default(config::get_settings_file_path_or_panic()) {
        Ok(settings) => match OllamaModelDownloader::new(settings) {
            Ok(downloader) => match downloader.list_available_models(page, page_size) {
                Ok(models) => {
                    if let (Some(p), Some(_ps)) = (page, page_size) {
                        println!(
                            "Model identifiers: ({}, page {}): {:?}",
                            models.len(),
                            p,
                            models
                        );
                    } else {
                        println!("Model identifiers: ({}): {:?}", models.len(), models);
                    }
                }
                Err(e) => {
                    error!("Error listing models: {}", e);
                    std::process::exit(1);
                }
            },
            Err(e) => {
                error!("Failed to create Ollama downloader: {}", e);
                std::process::exit(1);
            }
        },
        Err(e) => {
            error!("Failed to load settings: {}", e);
            std::process::exit(1);
        }
    }
}

fn handle_list_tags(model_identifier: String) {
    match AppSettings::load_or_create_default(config::get_settings_file_path_or_panic()) {
        Ok(settings) => match OllamaModelDownloader::new(settings) {
            Ok(downloader) => match downloader.list_model_tags(&model_identifier) {
                Ok(tags) => {
                    println!("Model tags: ({} tags): {:?}", tags.len(), tags);
                }
                Err(e) => {
                    error!("Error listing tags for model '{}': {}", model_identifier, e);
                    std::process::exit(1);
                }
            },
            Err(e) => {
                error!("Failed to create Ollama downloader: {}", e);
                std::process::exit(1);
            }
        },
        Err(e) => {
            error!("Failed to load settings: {}", e);
            std::process::exit(1);
        }
    }
}

fn handle_model_download(model_tag: String) {
    match AppSettings::load_or_create_default(config::get_settings_file_path_or_panic()) {
        Ok(settings) => match OllamaModelDownloader::new(settings) {
            Ok(downloader) => match downloader.download_model(&model_tag) {
                Ok(_) => {
                    println!("Model {} download completed successfully", model_tag);
                    signal_handler::set_cleanup_done();
                }
                Err(e) => {
                    if should_log_download_error_in_main(&e) {
                        error!("Error downloading model '{}': {}", model_tag, e);
                    }
                    if !signal_handler::is_interrupted() {
                        std::process::exit(1);
                    }
                    signal_handler::set_cleanup_done();
                }
            },
            Err(e) => {
                error!("Failed to create Ollama downloader: {}", e);
                std::process::exit(1);
            }
        },
        Err(e) => {
            error!("Failed to load settings: {}", e);
            std::process::exit(1);
        }
    }
}

fn handle_hf_list_models(page: u32, page_size: u32) {
    match AppSettings::load_or_create_default(config::get_settings_file_path_or_panic()) {
        Ok(settings) => match HuggingFaceModelDownloader::new(settings) {
            Ok(downloader) => match downloader.list_available_models(Some(page), Some(page_size)) {
                Ok(models) => {
                    println!(
                        "Model identifiers: ({}, page {}): {:?}",
                        models.len(),
                        page,
                        models
                    );
                }
                Err(e) => {
                    error!("Error listing HuggingFace models: {}", e);
                    std::process::exit(1);
                }
            },
            Err(e) => {
                error!("Failed to create HuggingFace downloader: {}", e);
                std::process::exit(1);
            }
        },
        Err(e) => {
            error!("Failed to load settings: {}", e);
            std::process::exit(1);
        }
    }
}

fn handle_hf_list_tags(model_identifier: String) {
    match AppSettings::load_or_create_default(config::get_settings_file_path_or_panic()) {
        Ok(settings) => match HuggingFaceModelDownloader::new(settings) {
            Ok(downloader) => match downloader.list_model_tags(&model_identifier) {
                Ok(tags) => {
                    println!("Model tags: ({} tags): {:?}", tags.len(), tags);
                }
                Err(e) => {
                    error!(
                        "Error listing tags for HuggingFace model '{}': {}",
                        model_identifier, e
                    );
                    std::process::exit(1);
                }
            },
            Err(e) => {
                error!("Failed to create HuggingFace downloader: {}", e);
                std::process::exit(1);
            }
        },
        Err(e) => {
            error!("Failed to load settings: {}", e);
            std::process::exit(1);
        }
    }
}

fn handle_hf_model_download(user_repo_quant: String) {
    match AppSettings::load_or_create_default(config::get_settings_file_path_or_panic()) {
        Ok(settings) => match HuggingFaceModelDownloader::new(settings) {
            Ok(downloader) => match downloader.download_model(&user_repo_quant) {
                Ok(_) => {
                    println!(
                        "HuggingFace model {} download completed successfully",
                        user_repo_quant
                    );
                    signal_handler::set_cleanup_done();
                }
                Err(e) => {
                    if should_log_download_error_in_main(&e) {
                        error!(
                            "Error downloading HuggingFace model '{}': {}",
                            user_repo_quant, e
                        );
                    }
                    if !signal_handler::is_interrupted() {
                        std::process::exit(1);
                    }
                    signal_handler::set_cleanup_done();
                }
            },
            Err(e) => {
                error!("Failed to create HuggingFace downloader: {}", e);
                std::process::exit(1);
            }
        },
        Err(e) => {
            error!("Failed to load settings: {}", e);
            std::process::exit(1);
        }
    }
}

fn should_log_download_error_in_main(error: &DownloaderError) -> bool {
    !matches!(
        error,
        DownloaderError::HttpError(_) | DownloaderError::HttpStatus(_)
    )
}

fn handle_od_copy_settings(od_settings_file: String) {
    use std::fs;
    use std::path::Path;

    let source_path = Path::new(&od_settings_file);
    let dest_path = config::get_settings_file_path_or_panic();

    // Check if source file exists
    if !source_path.exists() {
        error!("Source settings file does not exist: {}", od_settings_file);
        std::process::exit(1);
    }

    // Check if source file is readable
    if let Err(e) = fs::metadata(source_path) {
        error!(
            "Cannot access source settings file '{}': {}",
            od_settings_file, e
        );
        std::process::exit(1);
    }

    // Check if destination file already exists
    if dest_path.exists() {
        println!("Settings file already exists at: {}", dest_path.display());
        print!("Overwrite existing settings file? [y/N]: ");
        io::stdout().flush().unwrap();

        let mut input = String::new();
        if let Err(e) = io::stdin().read_line(&mut input) {
            error!("Failed to read user input: {}", e);
            std::process::exit(1);
        }

        let input = input.trim().to_lowercase();
        if input != "y" && input != "yes" {
            info!("Operation cancelled by user.");
            return;
        }
    }

    // Copy the file
    match fs::copy(source_path, &dest_path) {
        Ok(_) => {
            info!(
                "Successfully copied settings from '{}' to '{}'",
                od_settings_file,
                dest_path.display()
            );
            println!(
                "Settings file copied successfully to: {}",
                dest_path.display()
            );
        }
        Err(e) => {
            error!(
                "Failed to copy settings from '{}' to '{}': {}",
                od_settings_file,
                dest_path.display(),
                e
            );
            std::process::exit(1);
        }
    }
}

fn handle_journal(
    model_identifier: Option<String>,
    list: bool,
    source: Option<JournalSourceArg>,
    clear: bool,
    as_json: bool,
) {
    let source_hint = source.map(DownloadSourceType::from);
    if list {
        match list_available_journals(source_hint) {
            Ok(journals) => {
                if as_json {
                    let list_entries: Vec<DownloadJournalListEntry> = journals
                        .iter()
                        .map(|journal| DownloadJournalListEntry {
                            model_identifier: journal.model_identifier.clone(),
                            source_type: journal.source_type.clone(),
                            tag_or_quant: journal.tag_or_quant.clone(),
                            updated_at: journal.updated_at,
                            item_count: journal.items.len(),
                        })
                        .collect();

                    match serde_json::to_string_pretty(&list_entries) {
                        Ok(serialized) => println!("{}", serialized),
                        Err(e) => {
                            error!("Failed to serialize journal list as JSON: {}", e);
                            std::process::exit(1);
                        }
                    }
                    return;
                }

                if journals.is_empty() {
                    println!("No journals found");
                    return;
                }

                println!("Available journals ({}):", journals.len());
                for journal in journals {
                    println!(
                        "- {} | {:?} | {} | updated={} | items={}",
                        journal.model_identifier,
                        journal.source_type,
                        journal.tag_or_quant,
                        journal.updated_at,
                        journal.items.len()
                    );
                }
                return;
            }
            Err(e) => {
                error!("Failed to list journals: {}", e);
                std::process::exit(1);
            }
        }
    }

    let model_identifier =
        model_identifier.expect("clap should require model_identifier unless --list");
    match load_journal_for_model(&model_identifier, source_hint.clone()) {
        Ok(journal) => {
            if clear {
                let journal_is_completed = journal.items.iter().all(|item| {
                    matches!(
                        item.state,
                        crate::downloader::manifest::JournalItemState::Completed
                    )
                });
                let source_label = match journal.source_type {
                    DownloadSourceType::Ollama => "ollama",
                    DownloadSourceType::Hf => "hf",
                };
                if journal_is_completed {
                    print!(
                        "Do you really want to delete the download journal {} from {}? This model download has completed, hence its journal can be safely cleared without deleting the downloaded model. [y/N]: ",
                        model_identifier, source_label
                    );
                } else {
                    print!(
                        "Do you really want to delete the download journal {} from {}? If you do, this model download cannot be resumed from its current state. [y/N]: ",
                        model_identifier, source_label
                    );
                }
                if io::stdout().flush().is_err() {
                    error!("Failed to flush confirmation prompt");
                    std::process::exit(1);
                }

                let mut input = String::new();
                if io::stdin().read_line(&mut input).is_err() {
                    error!("Failed to read confirmation response");
                    std::process::exit(1);
                }
                let confirmed = matches!(input.trim().to_lowercase().as_str(), "y" | "yes");
                if !confirmed {
                    println!("Journal clear cancelled");
                    return;
                }

                let settings = match AppSettings::load_or_create_default(
                    config::get_settings_file_path_or_panic(),
                ) {
                    Ok(s) => s,
                    Err(e) => {
                        error!("Failed to load settings for journal clear: {}", e);
                        std::process::exit(1);
                    }
                };

                match clear_journal_for_model(
                    &model_identifier,
                    journal.source_type.clone(),
                    &settings.ollama_library.models_path,
                ) {
                    Ok(summary) => {
                        if summary.had_completed_download {
                            println!(
                                "Deleted journal for {} (source: {}); completed download data was kept",
                                model_identifier, source_label
                            );
                        } else {
                            println!(
                                "Deleted journal for {} (source: {}); removed {} partial file(s)",
                                model_identifier, source_label, summary.removed_partial_files
                            );
                        }
                        return;
                    }
                    Err(e) => {
                        error!("Failed to clear journal for '{}': {}", model_identifier, e);
                        std::process::exit(1);
                    }
                }
            }

            if as_json {
                match serde_json::to_string_pretty(&journal) {
                    Ok(serialized) => {
                        println!("{}", serialized);
                    }
                    Err(e) => {
                        error!("Failed to serialize journal as JSON: {}", e);
                        std::process::exit(1);
                    }
                }
                return;
            }

            println!("Model: {}", journal.model_identifier);
            println!("Source: {:?}", journal.source_type);
            println!("Tag or Quantisation: {}", journal.tag_or_quant);
            println!(
                "Started At: {}",
                format_epoch_local_rfc2822(journal.started_at)
            );
            println!(
                "Updated At: {}",
                format_epoch_local_rfc2822(journal.updated_at)
            );
            let total_items = journal.items.len();
            let completed_items = journal
                .items
                .iter()
                .filter(|item| {
                    matches!(
                        item.state,
                        crate::downloader::manifest::JournalItemState::Completed
                    )
                })
                .count();
            let completed_pct = if total_items == 0 {
                0.0
            } else {
                (completed_items as f64) * 100.0 / (total_items as f64)
            };
            println!(
                "Completed: {:.2}% ({}/{} items)",
                completed_pct, completed_items, total_items
            );
            println!("Items:");
            for item in journal.items {
                let err = item.last_error.unwrap_or_else(|| "-".to_string());
                println!(
                    "  - {} | {} | {} bytes | {:?} | {}",
                    item.digest, item.media_type, item.size, item.state, err
                );
            }
        }
        Err(e) => {
            error!("Failed to load journal for '{}': {}", model_identifier, e);
            std::process::exit(1);
        }
    }
}

/// The main entry point for the Ollama Downloader in Rust (ODIR) command-line application.
fn main() {
    // Initialize configuration from environment variables
    let config = Config::from_env();

    // Initialize logger with the configured log level
    env_logger::Builder::new()
        .filter_level(config.log_level)
        .init();

    debug!(
        "Configuration loaded: log_level={:?}, user_agent={}, settings_file={:?}",
        config.log_level,
        config::get_user_agent(),
        config::get_settings_file_path_or_panic()
    );

    // Install signal handlers for graceful shutdown
    signal_handler::install_signal_handlers();

    let cli = Cli::parse();

    let requires_interrupt_confirmation = matches!(
        &cli.command,
        Commands::ListModels { .. }
            | Commands::ListTags { .. }
            | Commands::ModelDownload { .. }
            | Commands::HfListModels { .. }
            | Commands::HfListTags { .. }
            | Commands::HfModelDownload { .. }
            | Commands::Journal { .. }
    );
    signal_handler::set_confirmation_required(requires_interrupt_confirmation);

    match cli.command {
        Commands::ShowConfig => handle_show_config(),
        Commands::EditConfig { config_file } => handle_edit_config(config_file),
        Commands::ListModels { page, page_size } => handle_list_models(page, page_size),
        Commands::ListTags { model_identifier } => handle_list_tags(model_identifier),
        Commands::ModelDownload { model_tag } => handle_model_download(model_tag),
        Commands::HfListModels { page, page_size } => handle_hf_list_models(page, page_size),
        Commands::HfListTags { model_identifier } => handle_hf_list_tags(model_identifier),
        Commands::HfModelDownload { user_repo_quant } => handle_hf_model_download(user_repo_quant),
        Commands::OdCopySettings { od_settings_file } => handle_od_copy_settings(od_settings_file),
        Commands::Journal {
            model_identifier,
            list,
            source,
            clear,
            json,
        } => handle_journal(model_identifier, list, source, clear, json),
    }
}
