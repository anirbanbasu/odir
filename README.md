# Ollama Downloader in Rust (ODIR or _oh dear_)

[![Rust tests](https://github.com/anirbanbasu/odir/actions/workflows/rust.yml/badge.svg)](https://github.com/anirbanbasu/odir/actions/workflows/rust.yml) [![Markdown Lint](https://github.com/anirbanbasu/odir/actions/workflows/md-lint.yml/badge.svg)](https://github.com/anirbanbasu/odir/actions/workflows/md-lint.yml) [![CodeQL Advanced](https://github.com/anirbanbasu/odir/actions/workflows/codeql.yml/badge.svg)](https://github.com/anirbanbasu/odir/actions/workflows/codeql.yml) [![OpenSSF Scorecard](https://api.scorecard.dev/projects/github.com/anirbanbasu/odir/badge)](https://scorecard.dev/viewer/?uri=github.com/anirbanbasu/odir) [![OpenSSF Best Practices](https://www.bestpractices.dev/projects/11975/badge)](https://www.bestpractices.dev/projects/11975) ![crates.io](https://img.shields.io/crates/v/odir.svg)  


Ollama Downloader in Rust (ODIR), pronounced _oh dear_, is a command-line tool written in Rust for downloading models from Ollama. A successor of the original Python implementation called [Ollama Downloader](https://github.com/anirbanbasu/ollama-downloader), ODIR has been rewritten in Rust to leverage its performance and safety features.

## A bit of history

(and information from the Ollama Downloader README)

Rather evident from the name, this is a tool to help download models for [Ollama](https://ollama.com/) including [supported models from Hugging Face](https://huggingface.co/models?apps=ollama). However, doesn't Ollama already download models from its library using `ollama pull <model:tag>`?

Yes, but wait, not so fast...!

### How did we get here?

While `ollama pull <model:tag>` certainly works, not always will you get lucky. This is a documented problem, see [issue 941](https://github.com/ollama/ollama/issues/941). The crux of the problem is that Ollama fails to pull a model from its library spitting out an error message as follows.

> `Error: digest mismatch, file must be downloaded again: want sha256:1a640cd4d69a5260bcc807a531f82ddb3890ebf49bc2a323e60a9290547135c1, got sha256:5eef5d8ec5ce977b74f91524c0002f9a7adeb61606cdbdad6460e25d58d0f454`

People have been facing this for a variety of unrelated reasons and have found specific solutions that perhaps work for only when those specific reasons exist.

[Comment 2989194688](https://github.com/ollama/ollama/issues/941#issuecomment-2989194688) in the issue thread proposes a manual way to download the models from the library. This solution is likely to work more than others.

_Hence, this tool and its predecessor, the Python-based Ollama Downloader – an automation of that manual process_!

ODIR, like its predecessor, can also _download supported models from Hugging Face_!

### Apart from `ollama pull`

Ollama's issues with the `ollama pull` command can also implicitly bite you when using `ollama create`.

As shown in the official [example of customising a prompt using a Modelfile](https://github.com/ollama/ollama?tab=readme-ov-file#customize-a-prompt), if you omit the step `ollama pull llama3.2`, then Ollama will automatically pull that model when you run `ollama create mario -f ./Modelfile`. Thus, if Ollama had issues with pulling that model, then those issues will hinder the custom model creation.

Likewise, a more obvious command that will encounter the same issues as `ollama pull` is `ollama run`, which implicitly pulls the model if it does not exist.

Thus, the safer route is to pull the model, in advance, using this downloader so that Ollama does not try to pull it implicitly (and fail at it).

## Installation

The current preferred way is to download and compile the source from the HEAD of the main branch of ODIR using Cargo, Rust's package manager. You must have the [Rust toolchain installed](https://rust-lang.org/tools/install/). Run the following command in your terminal to install ODIR.

```bash
cargo install --git https://github.com/anirbanbasu/odir
```

If you want to install the latest released version, you can install it directly from crates.io by running the following command.

```bash
cargo install odir
```

There is also a Homebrew formula for ODIR, which may be sometimes less up-to-date than the Cargo installation. You can install it using Homebrew with the following command.

```bash
brew install anirbanbasu/tap/odir
```

## Configuration

There will exist, upon execution of the tool, a configuration file `settings.json` in the user-specific configuration directory for the operating system. This is, for instance, `/Users/username/Library/Application Support/odir` on macOS, or `/home/username/.config/odir` on Linux. It will be created upon the first run. However, you may need to modify it depending on your Ollama installation.

Let's explore the configuration in details. The default content is as follows.

```json
{
    "ollama_server": {
        "url": "http://localhost:11434",
        "api_key": null,
        "remove_downloaded_on_error": true,
        "check_model_presence": true,
        "keep_verified_blobs_on_error": true
    },
    "ollama_library": {
        "models_path": "/home/username/.ollama/models",
        "registry_base_url": "https://registry.ollama.ai/v2/library/",
        "library_base_url": "https://ollama.com/library",
        "verify_ssl": true,
        "timeout": 120.0,
        "chunk_size_mib": 128,
        "download_chunks_in_parallel": true,
        "transient_cleanup_enabled": true,
        "transient_ttl_hours": 72,
        "failed_journal_ttl_hours": 168,
        "completed_journal_ttl_hours": 24
    }
}
```

There are two main configuration groups: `ollama_server` and `ollama_library`. The former refers to the server for which you wish to download the model. The latter refers to the Ollama library where the model and related information ought to be downloaded from.

### `ollama_server`

- The `url` points to the HTTP endpoint of your Ollama server. While the default is [http://localhost:11434](http://localhost:11434), note that your Ollama server may actually be running on a different machine, in which case, the URL will have to point to that endpoint correctly.
- The `api_key` is only necessary if your Ollama server endpoint expects an API key to connect, which is typically not the case.
- The `remove_downloaded_on_error` is a boolean flag, typically set to `true`. This helps specify whether this downloader tool should remove downloaded files (including temporary files) if it fails to connect to the Ollama server or fails to find the downloaded model.
- The `check_model_presence` is a boolean flag, typically set to `true`. This helps specify whether this downloader tool should check for the presence of the model in the Ollama server after downloading it.
- The `keep_verified_blobs_on_error` is a boolean flag, typically set to `true`. When enabled, already verified blobs are preserved across failures so reruns only fetch missing/invalid blobs. Set this to `false` to restore strict rollback behavior.

### `ollama_library`

- The `models_path` points to the models directory of your Ollama installation. On Linux/UNIX systems, if it has been installed for your own user only then the path is the default `/home/username/.ollama/models`. If it has been installed as a service, however, it could be, for example on Ubuntu, `/usr/share/ollama/.ollama/models`. Also note that the path could be a network share, if Ollama is on a different machine. If the path is not in the current user directory, on a Linux/UNIX system, you may need to run ODIR using `sudo` to have the necessary permissions to write to that path.
- The `registry_base_url` is the URL to the Ollama registry. Unless you have a custom Ollama registry, use the default value as shown above.
- Likewise, the `library_base_url` is the URL to the Ollama library. Keep the default value unless you really need to point it to some mirror.
- The `verify_ssl` is a flag that tells the downloader tool to verify the authenticity of the HTTPS connections it makes to the Ollama registry or the library. Turn this off only if you have a man-in-the-middle proxy with self-signed certificates. Even in that case, typically environment variables `SSL_CERT_FILE` and `SSL_CERT_DIR` can be correctly configured to validate such certificates.
- The self-explanatory `timeout` specifies the number of seconds to wait before any HTTPS connection to the Ollama registry or library should be allowed to fail.
- The `chunk_size_mib` controls the size (in MiB) of each chunk when downloading large model blobs. Large blobs are split into sequential byte-range requests of this size, which makes downloads more robust over unreliable connections. If a download is interrupted by a network error, only the missing parts will be re-fetched on the next run. Set to `0` to disable chunked downloading and download each blob in a single stream. The value must be one of `0`, `32`, `64`, `128`, `256`, or `512`. The default is `128` (128 MiB). Note that if the remote server does not support chunked downloading, or if the blob is smaller than the configured chunk size, the download will automatically fall back to single-stream mode regardless of this setting.
- The `download_chunks_in_parallel` controls whether chunked downloads use multiple workers (`true`) or a single worker (`false`). The default is `true`. This setting has effect only when chunked downloading is active (`chunk_size_mib > 0`) and the server supports byte-range downloads.
- The `transient_cleanup_enabled` controls whether stale transient artefacts should be cleaned up automatically before and after downloads. Default is `true`.
- The `transient_ttl_hours` controls the age threshold (in hours) used to remove stale transient files such as chunk work files under `.parts`. Default is `72`.
- The `failed_journal_ttl_hours` controls how long failed/incomplete journal files are kept before cleanup. Default is `168`.
- The `completed_journal_ttl_hours` controls how long completed journal files are kept before cleanup. Default is `24`.

### Journal diagnostics

ODIR maintains an advisory download journal per model in its application data directory (not in the Ollama models directory). You can inspect it with:

```bash
odir journal <model_identifier>
```

To emit machine-readable output instead of the default human-friendly view:

```bash
odir journal <model_identifier> --json
```

The human-friendly per-model view also includes a computed `Completed` field
showing progress as a percentage of completed items in the journal.

Optional source selection can be provided with `--source ollama` or `--source hf`.

To list all available journals without providing a model identifier:

```bash
odir journal --list
```

You can combine list mode with JSON output:

```bash
odir journal --list --json
```

In list JSON mode, each entry contains `model_identifier`, `source_type`,
`tag_or_quant`, `updated_at`, and `item_count`.

To delete one specific journal and remove resumable partial data for incomplete
downloads:

```bash
odir journal <model_identifier> --clear
```

`--clear` asks for confirmation with a default answer of `N` (No). If the
journal indicates the download had already completed, only the journal file is
deleted and completed model data is kept.

## Environment variables

The environment variable(s), listed below, are _optional_. If not specified, their default values will be used.

| Variable  | Description and default value(s)                                     |
|-----------|----------------------------------------------------------------------|
| `ODIR_LOG_LEVEL` or `OD_LOG_LEVEL` | The level to be set for the logger. Default value is `INFO`. See all valid values in [Rust logging documentation](https://docs.rs/log/latest/log/enum.Level.html). The level specification can be set to `OFF`, which turns off logging completely.|
| `OD_UA` | Overrides the HTTP User-Agent string used by ODIR. By default, ODIR uses `odir/<version> (<arch> <os>)` with lower-case, Ollama-compatible platform names such as `amd64 linux` or `arm64 darwin`. |

_Note that the environment variable `ODIR_LOG_LEVEL` takes precedence over `OD_LOG_LEVEL` if both are set. Also note that in the original Ollama Downloader, it was possible to specify `OD_SETTINGS_FILE` as an [environment variable](https://github.com/anirbanbasu/ollama-downloader?tab=readme-ov-file#environment-variables), but that is no longer supported in ODIR. Instead, the default value is the user-specific settings file location for the operating system._

## Usage

The usage is straightforward. Run the following command in your terminal to show the available commands and options.

```bash
odir --help
```

The output will be as follows.

```bash
Ollama Downloader in Rust (ODIR), pronounced oh dear, is a command-line tool written in Rust for downloading models from Ollama.

Usage: odir <COMMAND>

Commands:
  show-config        Shows the application configuration as JSON
  edit-config        Interactively edits application settings through step-by-step questions
  list-models        Lists all available models in the Ollama library
  list-tags          Lists all tags for a specific model
  model-download     Downloads a specific Ollama model with the given tag
  hf-list-models     Lists available models from Hugging Face that can be downloaded into Ollama
  hf-list-tags       Lists all available quantisations as tags for a Hugging Face model that can be downloaded into Ollama
  hf-model-download  Downloads a specified Hugging Face model
  od-copy-settings   Copies a Ollama Downloader settings file to the ODIR settings location
  journal            Displays the advisory download journal for a model
  help               Print this message or the help of the given subcommand(s)

Options:
  -h, --help     Print help
  -V, --version  Print version
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on how to contribute to this project.

## License

This project is licensed under the [MIT License](https://choosealicense.com/licenses/mit/).
