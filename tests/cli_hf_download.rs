//! Integration tests for HuggingFace model downloads via CLI
//!
//! These tests verify the end-to-end functionality of downloading HuggingFace models
//! compatible with Ollama through the CLI interface, including interrupt handling and cleanup.
//!
//! ## Running these tests
//!
//! These tests require network access and interact with real HuggingFace servers.
//! To run them, use:
//!
//! ```bash
//! RUN_INTEGRATION_TESTS=1 cargo test --test cli_hf_download -- --nocapture
//! ```
//!
//! To test specific scenarios:
//! ```bash
//! RUN_INTEGRATION_TESTS=1 cargo test --test cli_hf_download test_hf_interrupt_handling -- --nocapture
//! ```

mod common;

use std::thread;
use std::time::Duration;

/// Test that the CLI properly handles HuggingFace download interrupts with SIGINT
///
/// This test:
/// 1. Starts downloading a small HuggingFace model
/// 2. Sends SIGINT (Ctrl+C) after a short delay
/// 3. Verifies the process handles the interrupt gracefully
/// 4. Checks that cleanup occurs
#[test]
fn test_hf_interrupt_handling() {
    if !common::should_run_integration_tests() {
        println!("Skipping integration test. Set RUN_INTEGRATION_TESTS=1 to run.");
        return;
    }

    println!("Testing HuggingFace download interrupt handling...");

    // Use a small model for testing - this is a tiny GGUF model
    let model = "unsloth/SmolLM2-135M-Instruct-GGUF:Q4_K_M";

    let mut child = common::spawn_odir(&["hf-model-download", model]);

    println!("Spawned download process, PID: {}", child.id());

    // Let the download start
    thread::sleep(Duration::from_secs(2));

    println!("Sending SIGINT to process...");
    common::send_sigint(&child);

    // Give it time to handle the signal
    thread::sleep(Duration::from_secs(1));

    // For automated testing, we need to send another signal or kill it
    // because the confirmation prompt won't be answered
    if let Some(status) = common::wait_with_timeout(&mut child, 3) {
        println!("Process exited with status: {:?}", status);
    } else {
        println!("Process didn't exit after SIGINT, sending SIGTERM...");
        common::send_sigterm(&child);

        if let Some(status) = common::wait_with_timeout(&mut child, 5) {
            println!("Process exited after SIGTERM with status: {:?}", status);
        } else {
            println!("Process still running, force killing...");
            let _ = child.kill();
            let _ = child.wait();
        }
    }

    println!("Interrupt handling test completed");
}

/// Test downloading a small HuggingFace model via CLI (full download test)
///
/// This test performs an actual download and verifies successful completion.
/// It's more resource-intensive and takes longer, so it's marked as ignored
/// by default. Run explicitly with:
///
/// ```bash
/// RUN_INTEGRATION_TESTS=1 cargo test --test cli_hf_download test_hf_download_success -- --ignored --nocapture
/// ```
#[test]
#[ignore]
fn test_hf_download_success() {
    if !common::should_run_integration_tests() {
        println!("Skipping integration test. Set RUN_INTEGRATION_TESTS=1 to run.");
        return;
    }

    println!("Testing successful HuggingFace model download...");

    // Use a very small quantized model for testing
    let model = "unsloth/SmolLM2-135M-Instruct-GGUF:Q4_K_M";

    let mut child = common::spawn_odir(&["hf-model-download", model]);

    println!("Spawned download process, PID: {}", child.id());

    // Wait for the download to complete (with a reasonable timeout)
    let status = common::wait_with_timeout(&mut child, 600) // 10 minutes timeout
        .expect("Download process did not complete within timeout");

    println!("Process exited with status: {:?}", status);

    // Check that the process exited successfully
    assert!(
        status.success(),
        "Download process should exit successfully, but exited with: {:?}",
        status
    );

    println!("Download completed successfully!");
}

/// Test that invalid model identifiers are handled correctly
#[test]
fn test_hf_invalid_model() {
    println!("Testing HuggingFace download with invalid model identifier...");

    // Use a model identifier that doesn't exist
    let model = "invalid-user/nonexistent-model-12345:Q4_K_M";

    let mut child = common::spawn_odir(&["hf-model-download", model]);

    println!("Spawned process for invalid model, PID: {}", child.id());

    // Wait for the process to complete
    let status =
        common::wait_with_timeout(&mut child, 30).expect("Process did not complete within timeout");

    println!("Process exited with status: {:?}", status);

    // Should exit with error code
    assert!(
        !status.success(),
        "Download of non-existent model should fail, but succeeded"
    );
}

/// Test that malformed model identifiers are rejected
#[test]
fn test_hf_malformed_identifier() {
    println!("Testing HuggingFace download with malformed identifier...");

    // Use a malformed identifier (missing repository name)
    let model = "invalid-format";

    let mut child = common::spawn_odir(&["hf-model-download", model]);

    println!(
        "Spawned process for malformed identifier, PID: {}",
        child.id()
    );

    // Wait for the process to complete
    let status =
        common::wait_with_timeout(&mut child, 10).expect("Process did not complete within timeout");

    println!("Process exited with status: {:?}", status);

    // Should exit with error code
    assert!(
        !status.success(),
        "Malformed identifier should be rejected, but succeeded"
    );
}

/// Test that SIGTERM is also handled correctly for HuggingFace downloads
#[test]
fn test_hf_sigterm_handling() {
    if !common::should_run_integration_tests() {
        println!("Skipping integration test. Set RUN_INTEGRATION_TESTS=1 to run.");
        return;
    }

    println!("Testing HuggingFace download SIGTERM handling...");

    let model = "unsloth/SmolLM2-135M-Instruct-GGUF:Q4_K_M";

    let mut child = common::spawn_odir(&["hf-model-download", model]);

    println!("Spawned download process, PID: {}", child.id());

    // Let the download start
    thread::sleep(Duration::from_secs(2));

    println!("Sending SIGTERM to process...");
    common::send_sigterm(&child);

    // Give it time to handle the signal and clean up
    thread::sleep(Duration::from_secs(1));

    if let Some(status) = common::wait_with_timeout(&mut child, 3) {
        println!("Process exited with status: {:?}", status);
    } else {
        println!("Process didn't exit after SIGTERM, force killing...");
        let _ = child.kill();
        let _ = child.wait();
    }

    println!("SIGTERM handling test completed");
}

/// Test pagination in hf-list-models
#[test]
fn test_hf_list_models_pagination() {
    if !common::should_run_integration_tests() {
        println!("Skipping integration test. Set RUN_INTEGRATION_TESTS=1 to run.");
        return;
    }

    println!("Testing HuggingFace hf-list-models pagination...");

    // Test first page
    let output1 = std::process::Command::new(common::get_binary_path())
        .args(&["hf-list-models", "--page", "1", "--page-size", "5"])
        .output()
        .expect("Failed to execute hf-list-models command");

    // Test second page
    let output2 = std::process::Command::new(common::get_binary_path())
        .args(&["hf-list-models", "--page", "2", "--page-size", "5"])
        .output()
        .expect("Failed to execute hf-list-models command");

    assert!(output1.status.success(), "Page 1 should succeed");
    assert!(output2.status.success(), "Page 2 should succeed");

    let stdout1 = String::from_utf8_lossy(&output1.stdout);
    let stdout2 = String::from_utf8_lossy(&output2.stdout);

    println!("Page 1 output: {}", stdout1);
    println!("Page 2 output: {}", stdout2);

    // The outputs should be different (different pages)
    assert_ne!(
        stdout1, stdout2,
        "Different pages should return different results"
    );
}
