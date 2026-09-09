# AGENTS.md

ODIR (Ollama Downloader in Rust) is a CLI tool for downloading Ollama and
Hugging Face models, working around known bugs in `ollama pull`. Single Rust
binary crate, edition 2024.

## Architecture

- `src/main.rs` — clap-derived CLI; one `handle_*` function per subcommand.
- `src/config.rs` — `AppSettings`, JSON-backed, stored in the OS config dir;
  lenient deserialization fills in missing fields with defaults and
  revalidates URLs.
- `src/downloader/model_downloader.rs` — `ModelDownloader` trait and
  `DownloaderError`, implemented by:
  - `src/downloader/ollama_downloader.rs` — Ollama registry/library client.
  - `src/downloader/hf_downloader.rs` — Hugging Face client.
- `src/downloader/utils.rs` — bulk of the download logic: chunked/parallel
  blob downloads, resumable state, sha256 verification, the advisory
  download journal, and stale-artefact cleanup.
- `src/downloader/manifest.rs` — OCI-style image manifest and journal data
  structures.
- `src/signal_handler.rs` — SIGINT/SIGTERM handling with interactive
  confirmation so in-flight chunked downloads can be resumed instead of
  corrupted.

## Build, lint, test

- Build: `cargo build` (debug) / `cargo build --release`
- Lint (must be warning-free): `cargo clippy -- -D warnings`
- Format: `cargo fmt`
- Fast tests (what CI runs): `cargo test -- --test-threads=1`
- Full suite including real downloads: `cargo test -- --include-ignored
  --test-threads=1` (needs network access to Ollama/HF; slow)

`just -l` lists convenience recipes wrapping the above (build-debug,
type-check-and-lint, test, test-comprehensive, coverage-and-show-in-browser,
etc).

## Testing caveats

- Integration tests in `tests/` (`cli_*download*.rs`, `cli_journal.rs`)
  perform real downloads and are `#[ignore]`-gated; only run with
  `--include-ignored`.
- Always pass `--test-threads=1` — tests share state (settings file, journal
  directory, models path) and will conflict if run in parallel.
- Unit tests live inline in each module under `#[cfg(test)] mod tests`.

## Commit conventions

- DCO sign-off is required and CI-enforced (`.github/workflows/dco.yml`).
  Always commit with `git commit -s` so commits carry a `Signed-off-by`
  trailer.
