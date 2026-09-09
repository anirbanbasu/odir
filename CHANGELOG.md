# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/) and this project adheres to [Semantic Versioning](https://semver.org/).

## [unreleased]

### Added

- None documented yet.

### Changed

- None documented yet.

### Deprecated

- None documented yet.

### Removed

- None documented yet.

### Fixed

- None documented yet.

### Security

- None documented yet.

## [0.2.0] - 2026-09-09

### Added

- Added automatic retrying of failed downloads with exponential backoff, configurable via a new `download_retry` settings section (`enabled`, `max_retries`, `initial_backoff_ms`, `max_backoff_ms`). Only plausibly transient errors are retried (network/IO errors and HTTP 5xx/429/408 responses); the retry budget resets whenever a failure occurs at a different download stage than the previous one, so downloads making genuine progress aren't penalised for unrelated earlier setbacks.
- ODIR now offers to persist healed settings back to the configuration file when loading interactively, so a `settings.json` predating a newer field (e.g. the `download_retry` section) is no longer silently re-healed and re-warned about on every invocation. The prompt is skipped when stdin is non-interactive (scripts, cron jobs).

### Fixed

- `hf-list-tags` no longer advertises Hugging Face GGUF quantisation tags that the manifest registry doesn't recognise. Each candidate tag is now probed against the manifest endpoint and only returned if it resolves, falling back to `:latest` when none do, preventing a subsequent `hf-model-download` from failing with HTTP 400.

## [0.1.1] - 2026-04-08

### Added

- Improved coverage but this is still ongoing.
- Added support for chunked downloading of large model blobs, which makes downloads more robust over unreliable connections.
- Added support for resuming multi-layer models downloads, which can be interrupted (intentionally or not) and resumed without starting over.
- Added a new `journal` command to view the status of succeeded, pending, and failed downloads.

### Changed

- The default HTTP user-agent now includes normalized architecture and operating system names, for example `odir/0.1.1 (arm64 darwin)`.
- Added `OD_UA` environment variable support to override the HTTP user-agent used by ODIR.

## [0.1.0] - 2026-02-20

### Added

- Implemented all commands of the original [Ollama Downloader](https://github.com/anirbanbasu/ollama-downloader), as drop-in replacements, except `auto-config` and `version`, see the _Removed_ section below.
- Added support, through the `od-copy-settings` command for copying existing Ollama Downloader configuration files to the expected ODIR user-specific settings location for the operating system.

### Removed

- The `auto-config` command of the original Ollama Downloader has been removed. Instead, an interactive `edit-config` command has been implemented.
- The `version` command of the original Ollama Downloader has been removed in favour of the `-V` or the `--version` flag.

### Security

- Added a security policy.
- Added [OpenSSF scorecard badge](https://scorecard.dev/viewer/?uri=github.com/anirbanbasu/odir).
- Added [OpenSSF best practices badge](https://www.bestpractices.dev/projects/11975).
- Added CodeQL analysis.
- Added Open Source Vulnerability (OSV) analysis.


[unreleased]: https://github.com/anirbanbasu/odir/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/anirbanbasu/odir/compare/v0.1.1...v0.2.0
[0.1.1]: https://github.com/anirbanbasu/odir/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/anirbanbasu/odir/compare/v0.0.1...v0.1.0
