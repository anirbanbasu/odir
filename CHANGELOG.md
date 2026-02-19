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


[unreleased]: https://github.com/anirbanbasu/odir/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/anirbanbasu/odir/compare/v0.0.1...v0.1.0
