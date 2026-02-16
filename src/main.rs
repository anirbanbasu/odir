use clap::builder::styling::{AnsiColor, Effects, Styles};
use clap::{Parser, Subcommand};
use log::{debug, error, info, warn};
use std::io::{self, Write};

mod config;
use config::{AppSettings, Config};

mod downloader;
use downloader::{HuggingFaceModelDownloader, ModelDownloader, OllamaModelDownloader};

mod sysinfo;
use sysinfo::OllamaSystemInfo;

const STYLES: Styles = Styles::styled()
    .header(AnsiColor::Green.on_default().effects(Effects::BOLD))
    .usage(AnsiColor::Cyan.on_default().effects(Effects::BOLD))
    .literal(AnsiColor::Blue.on_default().effects(Effects::BOLD))
    .placeholder(AnsiColor::Yellow.on_default())
    .error(AnsiColor::Red.on_default().effects(Effects::BOLD))
    .valid(AnsiColor::Green.on_default())
    .invalid(AnsiColor::Red.on_default());

/// A command-line interface for the Ollama Downloader in Rust (ODIR), which is a Rust port of the Python-based Ollama Downloader (https://github.com/anirbanbasu/ollama-downloader).
#[derive(Parser)]
#[command(name = "odir")]
#[command(version, about)]
#[command(disable_help_subcommand = true)]
#[command(styles = STYLES)]
#[command(override_usage = "odir [OPTIONS] <COMMAND> [ARGS]...")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Shows the application configuration as JSON.
    ShowConfig,

    /// Displays an automatically inferred configuration.
    AutoConfig,

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

    /// Lists all tags for a specific model.
    ListTags {
        /// The name of the model to list tags for, e.g., llama3.1.
        model_identifier: String,
    },

    /// Downloads a specific Ollama model with the given tag.
    ModelDownload {
        /// The name of the model and a specific tag to download, specified as <model>:<tag>,
        /// e.g., llama3.1:8b. If no tag is specified, 'latest' will be assumed.
        model_tag: String,
    },

    /// Lists available models from Hugging Face that can be downloaded into Ollama.
    HfListModels {
        /// The page number to retrieve (1-indexed).
        #[arg(long, default_value_t = 1)]
        page: u32,

        /// The number of models to retrieve per page.
        #[arg(long, default_value_t = 25)]
        page_size: u32,
    },

    /// Lists all available quantisations as tags for a Hugging Face model that can be downloaded into Ollama.
    ///
    /// Note that these are NOT the same as Hugging Face model tags.
    HfListTags {
        /// The name of the model to list tags for, e.g., bartowski/Llama-3.2-1B-Instruct-GGUF.
        model_identifier: String,
    },

    /// Downloads a specified Hugging Face model.
    HfModelDownload {
        /// The name of the specific Hugging Face model to download, specified as
        /// <username>/<repository>:<quantisation>, e.g., bartowski/Llama-3.2-1B-Instruct-GGUF:Q4_K_M.
        user_repo_quant: String,
    },

    /// Copies a Ollama Downloader settings file to the ODIR settings location.
    OdCopySettings {
        /// Path to the existing Ollama Downloader settings file.
        od_settings_file: String,
    },
}

fn main() {
    // Initialize configuration from environment variables
    let config = Config::from_env();

    // Initialize logger with the configured log level
    env_logger::Builder::new()
        .filter_level(config.log_level)
        .init();

    debug!(
        "Configuration loaded: log_level={:?}, user_agent={}",
        config.log_level, config.user_agent
    );

    let cli = Cli::parse();

    match cli.command {
        Commands::ShowConfig => {
            match AppSettings::load_or_create_default(config::get_settings_file_path()) {
                Ok(settings) => match serde_json::to_string_pretty(&settings) {
                    Ok(json) => {
                        println!("{}", json);
                        info!(
                            "Settings loaded from {:?}",
                            config::get_settings_file_path()
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
                        config::get_settings_file_path(),
                        e
                    );
                    std::process::exit(1);
                }
            }
        }
        Commands::AutoConfig => {
            warn!(
                "Automatic configuration is an experimental feature. Its output maybe incorrect!"
            );

            let mut system_info = OllamaSystemInfo::new();

            if system_info.is_windows() {
                error!("Automatic configuration is not supported on Windows yet.");
                std::process::exit(1);
            }

            // Check if Ollama is running
            if !system_info.is_running() {
                error!("Ollama process not found. Make sure Ollama is running.");
                std::process::exit(1);
            }

            let listening_on = system_info.infer_listening_on();
            let models_dir_path = system_info.infer_models_dir_path();

            let mut super_user_maybe_needed = false;
            super_user_maybe_needed = super_user_maybe_needed
                || listening_on.is_none()
                || listening_on == Some("".to_string());
            super_user_maybe_needed = super_user_maybe_needed
                || models_dir_path.is_none()
                || models_dir_path == Some("".to_string());

            if super_user_maybe_needed {
                error!(
                    "Automatic configuration could not infer some settings. Maybe super-user permissions are necessary. Or, perhaps, Ollama has no models installed yet."
                );
                std::process::exit(1);
            } else {
                let mut inferred_settings = AppSettings::default();

                if let Some(url) = listening_on {
                    inferred_settings.ollama_server.url = url;
                }

                if let Some(path) = models_dir_path {
                    inferred_settings.ollama_library.models_path = path;
                }

                if system_info.is_likely_daemon() {
                    if system_info.is_macos() {
                        warn!(
                            "Automatic configuration on macOS maybe flawed if Ollama is configured to run as a system background service."
                        );
                    }

                    if let Some(owner) = system_info.get_process_owner() {
                        inferred_settings.ollama_library.user_group =
                            Some((owner.username.clone(), owner.groupname.clone()));
                    }
                }

                match serde_json::to_string_pretty(&inferred_settings) {
                    Ok(json) => {
                        println!("{}", json);
                    }
                    Err(e) => {
                        error!("Failed to serialize inferred settings: {}", e);
                        std::process::exit(1);
                    }
                }
            }
        }
        Commands::ListModels { page, page_size } => {
            match AppSettings::load_or_create_default(config::get_settings_file_path()) {
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
        Commands::ListTags { model_identifier } => {
            match AppSettings::load_or_create_default(config::get_settings_file_path()) {
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
        Commands::ModelDownload { model_tag } => {
            match AppSettings::load_or_create_default(config::get_settings_file_path()) {
                Ok(settings) => match OllamaModelDownloader::new(settings) {
                    Ok(downloader) => match downloader.download_model(&model_tag) {
                        Ok(_) => {
                            println!("Model {} download completed successfully", model_tag);
                        }
                        Err(e) => {
                            error!("Error downloading model '{}': {}", model_tag, e);
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
        Commands::HfListModels { page, page_size } => {
            match AppSettings::load_or_create_default(config::get_settings_file_path()) {
                Ok(settings) => match HuggingFaceModelDownloader::new(settings) {
                    Ok(downloader) => {
                        match downloader.list_available_models(Some(page), Some(page_size)) {
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
                        }
                    }
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
        Commands::HfListTags { model_identifier } => {
            match AppSettings::load_or_create_default(config::get_settings_file_path()) {
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
        Commands::HfModelDownload { user_repo_quant } => {
            match AppSettings::load_or_create_default(config::get_settings_file_path()) {
                Ok(settings) => match HuggingFaceModelDownloader::new(settings) {
                    Ok(downloader) => match downloader.download_model(&user_repo_quant) {
                        Ok(_) => {
                            println!(
                                "HuggingFace model {} download completed successfully",
                                user_repo_quant
                            );
                        }
                        Err(e) => {
                            error!(
                                "Error downloading HuggingFace model '{}': {}",
                                user_repo_quant, e
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
        Commands::OdCopySettings { od_settings_file } => {
            use std::fs;
            use std::path::Path;

            let source_path = Path::new(&od_settings_file);
            let dest_path = config::get_settings_file_path();

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
    }
}
