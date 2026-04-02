# Improved Resume Strategy: Concrete Task Breakdown

## Goal

Make model download resume robust across multi-layer failures by persisting verified blobs, adding a lightweight download journal, and introducing cleanup policies for transient artefacts.

## Out of Scope

- No broad CLI command surface changes in this plan.
- One exception: add a single read-only `journal` command with optional `--json` output.
- No README updates in this plan.

## Architecture Decisions

- Commit boundary is per blob, not per model.
- Manifest write remains final and atomic.
- Journal is advisory and reconciled against filesystem truth.
- Cleanup targets transient artefacts first, never verified blobs by default.

## Phase 1: Blob-Level Durability and Resume

### 1.1 Preserve verified blobs on error

- Update error cleanup behavior so verified blobs are retained.
- Keep removal for incomplete temp files and chunk parts only.

Files:

- src/downloader/ollama_downloader.rs
- src/downloader/hf_downloader.rs
- src/downloader/utils.rs
- src/downloader/model_downloader.rs

Tasks:

- Identify all paths where cleanup_unnecessary_files is invoked on download failure.
- Stop tracking committed blobs as unnecessary files.
- Ensure chunk-part temp files remain tracked and removable.

Acceptance:

- Failure after downloading layer N does not delete already verified layers 1..N.

### 1.2 Skip existing verified blobs on rerun

- Add blob existence and digest verification checks before attempting download.

Files:

- src/downloader/utils.rs
- src/downloader/ollama_downloader.rs
- src/downloader/hf_downloader.rs

Tasks:

- Add helper to resolve blob target path from digest and model storage root.
- Add helper to verify local blob digest matches expected digest.
- In per-item loop, short-circuit download if verified local blob exists.
- Print progress line indicating item skipped as already present.

Acceptance:

- Rerun after failure downloads only missing or invalid blobs.

### 1.3 Keep manifest atomic

- Save manifest only after all required blobs are present and verified.

Files:

- src/downloader/ollama_downloader.rs
- src/downloader/hf_downloader.rs

Tasks:

- Confirm manifest save remains after all blob operations.
- Ensure failure before manifest save leaves reusable committed blobs.

Acceptance:

- Partial model state can resume on next run without starting over.

## Phase 2: Lightweight Download Journal

### 2.1 Define journal schema

- Add per-model journal metadata and per-digest item states.

Files:

- src/downloader/manifest.rs
- src/downloader/utils.rs
- src/config.rs

Suggested schema:

- model_identifier
- source_type (ollama or hf)
- tag_or_quant
- started_at
- updated_at
- items: array of digest entries with
  - digest
  - media_type
  - size
  - state (pending, completed, failed)
  - last_error (optional)

Tasks:

- Define serde structs for journal documents.
- Add deterministic on-disk path for journals under ODIR app state (not Ollama models path).
- Use JSON as the journal format for consistency with existing serde_json usage.

Acceptance:

- Journal can be parsed and written atomically.
- Journal location is source-agnostic and shared by Ollama and HF flows.

### 2.2 Journal lifecycle and reconciliation

- Initialize journal after manifest parsing.
- Update per item state after skip, success, or failure.
- Reconcile journal with filesystem at start.

Files:

- src/downloader/ollama_downloader.rs
- src/downloader/hf_downloader.rs
- src/downloader/utils.rs

Tasks:

- Build manifest-derived item list and ensure journal alignment.
- On startup, mark completed where verified blobs exist.
- On successful blob commit, transition item state to completed.
- On error, write failed state with short error summary.

Acceptance:

- Crash or interruption still leaves useful progress metadata.

### 2.3 Atomic writes and corruption handling

- Write journal via temp file then rename.
- Handle malformed journals gracefully.

Files:

- src/downloader/utils.rs

Tasks:

- Implement write_journal_atomic helper.
- On parse failure, rename bad journal with .corrupt suffix and continue.

Acceptance:

- Journal cannot leave partial writes that break reruns.

### 2.4 Read-only journal CLI

- Add one read-only command to inspect journal state.

Files:

- src/main.rs
- src/downloader/utils.rs
- src/downloader/manifest.rs

Tasks:

- Add `journal` command with no subcommands (single-purpose view command).
- Support default user-friendly output and optional `--json` machine-readable output.
- Do not add any command to edit, mutate, or delete journal entries.
- On malformed journal, display a clear message and continue with filesystem-reconciled behavior.

Acceptance:

- Users can inspect journal progress and failure reasons without manual file inspection.
- Journal remains an internal advisory state; manual editing is unsupported and unnecessary.

## Phase 3: Cleanup Policy for Transient Artefacts

### 3.1 Add cleanup policy settings

- Introduce policy values with safe defaults.

Files:

- src/config.rs

Suggested settings:

- transient_cleanup_enabled: bool, default true
- transient_ttl_hours: u64, default 72
- failed_journal_ttl_hours: u64, default 168
- completed_journal_ttl_hours: u64, default 24

Tasks:

- Extend AppSettings model and default values.
- Validate new values in config validation path.

Acceptance:

- Existing config remains backward compatible via lenient loading.

### 3.2 Implement transient cleanup execution

- Remove stale chunk parts, temp files, and old journals.
- Do not remove verified blobs by default.

Files:

- src/downloader/utils.rs
- src/downloader/ollama_downloader.rs
- src/downloader/hf_downloader.rs

Tasks:

- Add cleanup function based on file mtime and configured TTL.
- Call cleanup at downloader start and optionally at downloader end.
- Log cleanup summary counts.

Acceptance:

- Storage does not grow unbounded from transient artefacts.

### 3.3 Optional strict rollback mode

- Preserve current all-or-nothing semantics as explicit opt-in only.

Files:

- src/config.rs
- src/downloader/ollama_downloader.rs
- src/downloader/hf_downloader.rs

Tasks:

- Add setting keep_verified_blobs_on_error: bool default true.
- If false, allow legacy full cleanup behavior.

Acceptance:

- Users can choose strict rollback, but resilient behavior is default.

## Test Plan

## Unit Tests

Files:

- src/downloader/utils.rs
- src/downloader/model_downloader.rs
- src/config.rs

Cases:

- Existing verified blob is detected and skipped.
- Invalid local blob digest triggers redownload.
- Journal serialization and deserialization roundtrip.
- Corrupt journal recovery path.
- Atomic journal writes produce valid files.
- Cleanup TTL removes only stale transient artefacts.

## Integration Tests

Files:

- tests/cli_ollama_download.rs
- tests/cli_hf_download.rs
- tests/common/mod.rs

Cases:

- Fail after multiple completed blobs, rerun resumes remaining items only.
- Interrupt mid-chunk and rerun resumes chunk and skips completed blobs.
- Failure before manifest save, rerun still reuses committed blobs.
- Cleanup does not remove verified blobs.

## Rollout Sequence

1. Phase 1 only, behind no feature flag.
2. Validate with integration tests and manual network interruption tests.
3. Phase 2 journal support with reconciliation logic.
4. Phase 3 cleanup policy settings and execution.
5. Final hardening pass on logs, telemetry-style counters, and failure messages.

## Implementation Notes

- Favor adding reusable helpers in src/downloader/utils.rs for shared behavior.
- Keep source-specific logic in src/downloader/ollama_downloader.rs and src/downloader/hf_downloader.rs limited to manifest parsing and naming differences.
- Ensure cleanup_unnecessary_files semantics are clearly split between transient and committed artefacts.
