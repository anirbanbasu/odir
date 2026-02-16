use clap::builder::styling::{AnsiColor, Effects, Styles};
use clap::{Parser, Subcommand};
use log::{debug, error, info};

mod config;
use config::{AppSettings, Config};

mod downloader;
use downloader::{HuggingFaceModelDownloader, ModelDownloader, OllamaModelDownloader};

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
}

fn main() {
    // Initialize configuration from environment variables
    let config = Config::from_env();

    // Initialize logger with the configured log level
    env_logger::Builder::new()
        .filter_level(config.log_level)
        .init();

    debug!(
        "Configuration loaded: log_level={:?}, settings_file={}, user_agent={}",
        config.log_level, config.settings_file, config.user_agent
    );

    let cli = Cli::parse();

    match cli.command {
        Commands::ShowConfig => match AppSettings::load_or_create_default(&config.settings_file) {
            Ok(settings) => match serde_json::to_string_pretty(&settings) {
                Ok(json) => {
                    println!("{}", json);
                    info!("Settings loaded from {}", config.settings_file);
                }
                Err(e) => {
                    error!("Failed to serialize settings: {}", e);
                    std::process::exit(1);
                }
            },
            Err(e) => {
                error!(
                    "Failed to load or create settings file '{}': {}",
                    config.settings_file, e
                );
                std::process::exit(1);
            }
        },
        Commands::AutoConfig => {
            eprintln!("auto-config: Not yet implemented");
        }
        Commands::ListModels { page, page_size } => {
            match AppSettings::load_or_create_default(&config.settings_file) {
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
            match AppSettings::load_or_create_default(&config.settings_file) {
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
            match AppSettings::load_or_create_default(&config.settings_file) {
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
            match AppSettings::load_or_create_default(&config.settings_file) {
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
            match AppSettings::load_or_create_default(&config.settings_file) {
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
            match AppSettings::load_or_create_default(&config.settings_file) {
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
    }
}
