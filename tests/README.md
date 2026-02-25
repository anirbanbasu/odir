# Integration Tests

This directory contains end-to-end integration tests for the ODIR application. These tests verify the complete functionality by spawning the actual CLI binary and testing real-world scenarios.

## Test Files

- **`cli_ollama_download.rs`** - Integration tests for Ollama library model downloads
- **`cli_hf_download.rs`** - Integration tests for HuggingFace model downloads  
- **`common/mod.rs`** - Shared test utilities and helper functions

## Running the Tests

### Prerequisites

1. **Build the project first:**

   ```bash
   cargo build
   ```

2. **Network access** - Most integration tests require internet connectivity to reach Ollama and HuggingFace servers.

### Basic Usage

**Run tests that don't require network access:**

```bash
cargo test --test cli_ollama_download test_ollama_invalid_model -- --nocapture
cargo test --test cli_hf_download test_hf_invalid_model -- --nocapture
cargo test --test cli_hf_download test_hf_malformed_identifier -- --nocapture
```

**Run network-dependent tests:**

```bash
RUN_INTEGRATION_TESTS=1 cargo test --test cli_ollama_download -- --nocapture
RUN_INTEGRATION_TESTS=1 cargo test --test cli_hf_download -- --nocapture
```

**Run all integration tests (including ignored ones):**

```bash
RUN_INTEGRATION_TESTS=1 cargo test --tests -- --ignored --nocapture
```

### Test Categories

#### Ollama Tests (`cli_ollama_download.rs`)

- `test_ollama_interrupt_handling` - Tests SIGINT (Ctrl+C) handling (requires network)
- `test_ollama_sigterm_handling` - Tests SIGTERM handling (requires network)
- `test_ollama_download_success` - Full download test (ignored by default, requires network)
- `test_ollama_invalid_model` - Tests error handling for non-existent models
- `test_ollama_list_models` - Tests listing available models (requires network)
- `test_ollama_list_tags` - Tests listing model tags (requires network)

#### HuggingFace Tests (`cli_hf_download.rs`)

- `test_hf_interrupt_handling` - Tests SIGINT handling (requires network)
- `test_hf_sigterm_handling` - Tests SIGTERM handling (requires network)
- `test_hf_download_success` - Full download test (ignored by default, requires network)
- `test_hf_invalid_model` - Tests error handling for non-existent models
- `test_hf_malformed_identifier` - Tests error handling for malformed identifiers
- `test_hf_list_models` - Tests listing available models (requires network)
- `test_hf_list_tags` - Tests listing model tags (requires network)
- `test_hf_list_models_pagination` - Tests pagination (requires network)

### Environment Variables

- `RUN_INTEGRATION_TESTS=1` - Enables network-dependent integration tests
  - Set this to run tests that interact with real Ollama and HuggingFace servers

### Examples

**Run a specific test:**

```bash
RUN_INTEGRATION_TESTS=1 cargo test --test cli_ollama_download test_ollama_interrupt_handling -- --nocapture
```

**Run only interrupt handling tests:**

```bash
RUN_INTEGRATION_TESTS=1 cargo test interrupt_handling -- --nocapture
```

**Run full download tests (takes several minutes):**

```bash
RUN_INTEGRATION_TESTS=1 cargo test download_success -- --ignored --nocapture
```

## Notes

- **Interrupt handling tests** send signals to the spawned process and verify graceful shutdown
- **Full download tests** are marked with `#[ignore]` because they:
  - Take significant time to complete
  - Download actual model files
  - Consume bandwidth
- Tests use small models where possible to minimize download times
- All tests verify correct exit codes and error messages
- Signal handling tests work on both Unix and Windows platforms

## CI/CD Integration

For continuous integration, you can run the fast tests without network:

```bash
cargo test --test cli_ollama_download test_ollama_invalid_model
cargo test --test cli_hf_download test_hf_invalid_model
cargo test --test cli_hf_download test_hf_malformed_identifier
```

For comprehensive testing (e.g., nightly builds), enable all tests:

```bash
RUN_INTEGRATION_TESTS=1 cargo test --tests -- --nocapture
```
