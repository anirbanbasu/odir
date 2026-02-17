# Ollama Downloader in Rust (ODIR or _oh dear_!)

[![Rust tests](https://github.com/anirbanbasu/odir/actions/workflows/rust.yml/badge.svg)](https://github.com/anirbanbasu/odir/actions/workflows/rust.yml) [![Markdown Lint](https://github.com/anirbanbasu/odir/actions/workflows/md-lint.yml/badge.svg)](https://github.com/anirbanbasu/odir/actions/workflows/md-lint.yml) [![CodeQL Advanced](https://github.com/anirbanbasu/odir/actions/workflows/codeql.yml/badge.svg)](https://github.com/anirbanbasu/odir/actions/workflows/codeql.yml) [![OpenSSF Scorecard](https://api.scorecard.dev/projects/github.com/anirbanbasu/odir/badge)](https://scorecard.dev/viewer/?uri=github.com/anirbanbasu/odir)


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

_Note that other methods of installation will be available in the future_.

## Configuration

There will exist, upon execution of the tool, a configuration file `settings.json` in the user-specific configuration directory for the operating system. This is, for instance, `/Users/username/Library/Application Support/odir` on macOS, or `/home/username/.config/odir` on Linux. It will be created upon the first run. However, you may need to modify it depending on your Ollama installation.

Let's explore the configuration in details. The default content is as follows.

```json
{
    "ollama_server": {
        "url": "http://localhost:11434",
        "api_key": null,
        "remove_downloaded_on_error": true,
        "check_model_presence": true
    },
    "ollama_library": {
        "models_path": "~/.ollama/models",
        "registry_base_url": "https://registry.ollama.ai/v2/library/",
        "library_base_url": "https://ollama.com/library",
        "verify_ssl": true,
        "timeout": 120.0,
    }
}
```

There are two main configuration groups: `ollama_server` and `ollama_library`. The former refers to the server for which you wish to download the model. The latter refers to the Ollama library where the model and related information ought to be downloaded from.

### `ollama_server`

- The `url` points to the HTTP endpoint of your Ollama server. While the default is [http://localhost:11434](http://localhost:11434), note that your Ollama server may actually be running on a different machine, in which case, the URL will have to point to that endpoint correctly.
- The `api_key` is only necessary if your Ollama server endpoint expects an API key to connect, which is typically not the case.
- The `remove_downloaded_on_error` is a boolean flag, typically set to `true`. This helps specify whether this downloader tool should remove downloaded files (including temporary files) if it fails to connect to the Ollama server or fails to find the downloaded model.
- The `check_model_presence` is a boolean flag, typically set to `true`. This helps specify whether this downloader tool should check for the presence of the model in the Ollama server after downloading it.

### `ollama_library`

- The `models_path` points to the models directory of your Ollama installation. On Linux/UNIX systems, if it has been installed for your own user only then the path is the default `~/.ollama/models`. If it has been installed as a service, however, it could be, for example on Ubuntu, `/usr/share/ollama/.ollama/models`. Also note that the path could be a network share, if Ollama is on a different machine. If the path is not in the current user directory, on a Linux/UNIX system, you may need to run ODIR using `sudo` to have the necessary permissions to write to that path.
- The `registry_base_url` is the URL to the Ollama registry. Unless you have a custom Ollama registry, use the default value as shown above.
- Likewise, the `library_base_url` is the URL to the Ollama library. Keep the default value unless you really need to point it to some mirror.
- The `verify_ssl` is a flag that tells the downloader tool to verify the authenticity of the HTTPS connections it makes to the Ollama registry or the library. Turn this off only if you have a man-in-the-middle proxy with self-signed certificates. Even in that case, typically environment variables `SSL_CERT_FILE` and `SSL_CERT_DIR` can be correctly configured to validate such certificates.
- The self-explanatory `timeout` specifies the number of seconds to wait before any HTTPS connection to the Ollama registry or library should be allowed to fail.

## Environment variables

The environment variable(s), listed below, are _optional_. If not specified, their default values will be used.

| Variable  | Description and default value(s)                                     |
|-----------|----------------------------------------------------------------------|
| `ODIR_LOG_LEVEL` or `OD_LOG_LEVEL` | The level to be set for the logger. Default value is `INFO`. See all valid values in [Rust logging documentation](https://docs.rs/log/latest/log/enum.Level.html). The level specification can be set to `OFF`, which turns off logging completely.|

_Note that the environment variable `ODIR_LOG_LEVEL` takes precedence over `OD_LOG_LEVEL` if both are set. Also note that in the original Ollama Downloader, it was possible to specify `OD_SETTINGS_FILE` and `OD_UA_NAME_VER` as [environment variables](https://github.com/anirbanbasu/ollama-downloader?tab=readme-ov-file#environment-variables), but those are no longer supported in ODIR. Instead, the default values for these are the user-specific settings file location for the operating system; and `odir/<app-version>`_.

## Usage

The usage is straightforward. Run the following command in your terminal to show the available commands and options.

```bash
odir --help
```

The output will be as follows.

```bash
A command-line interface for the Ollama Downloader in Rust (ODIR), which is a Rust port and successor of the Python-based [Ollama Downloader](https://github.com/anirbanbasu/ollama-downloader)

Usage: odir [OPTIONS] <COMMAND> [ARGS]...

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
  help               Print this message or the help of the given subcommand(s)

Options:
  -h, --help     Print help
  -V, --version  Print version
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on how to contribute to this project.

## License

This project is licensed under the [MIT License](https://choosealicense.com/licenses/mit/).
