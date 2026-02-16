use clap::builder::styling::{AnsiColor, Effects, Styles};
use clap::{Parser, Subcommand};

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
    let cli = Cli::parse();

    match cli.command {
        Commands::ShowConfig => {
            eprintln!("show-config: Not yet implemented");
        }
        Commands::AutoConfig => {
            eprintln!("auto-config: Not yet implemented");
        }
        Commands::ListModels { page, page_size } => {
            eprintln!("list-models: Not yet implemented");
            if let Some(p) = page {
                eprintln!("  page: {}", p);
            }
            if let Some(ps) = page_size {
                eprintln!("  page_size: {}", ps);
            }
        }
        Commands::ListTags { model_identifier } => {
            eprintln!("list-tags: Not yet implemented");
            eprintln!("  model_identifier: {}", model_identifier);
        }
        Commands::ModelDownload { model_tag } => {
            eprintln!("model-download: Not yet implemented");
            eprintln!("  model_tag: {}", model_tag);
        }
        Commands::HfListModels { page, page_size } => {
            eprintln!("hf-list-models: Not yet implemented");
            eprintln!("  page: {}", page);
            eprintln!("  page_size: {}", page_size);
        }
        Commands::HfListTags { model_identifier } => {
            eprintln!("hf-list-tags: Not yet implemented");
            eprintln!("  model_identifier: {}", model_identifier);
        }
        Commands::HfModelDownload { user_repo_quant } => {
            eprintln!("hf-model-download: Not yet implemented");
            eprintln!("  user_repo_quant: {}", user_repo_quant);
        }
    }
}
