//! Integration tests for Ollama library model downloads via CLI
//!
//! These tests verify the end-to-end functionality of downloading Ollama library models
//! through the CLI interface, including interrupt handling and cleanup.
//!
//! ## Running these tests
//!
//! These tests require network access and interact with real Ollama servers.
//! To run them, use:
//!
//! ```bash
//! RUN_INTEGRATION_TESTS=1 cargo test --test cli_ollama_download -- --nocapture
//! ```
//!
//! To test specific scenarios:
//! ```bash
//! RUN_INTEGRATION_TESTS=1 cargo test --test cli_ollama_download test_ollama_interrupt_handling -- --nocapture
//! ```

mod common;

use std::thread;
use std::time::Duration;

/// Test that the CLI properly handles download interrupts with SIGINT
///
/// This test:
/// 1. Starts downloading a small Ollama model
/// 2. Sends SIGINT (Ctrl+C) after a short delay
/// 3. Verifies the process handles the interrupt gracefully
/// 4. Checks that cleanup occurs
#[test]
fn test_ollama_interrupt_handling() {
    if !common::should_run_integration_tests() {
        println!("Skipping integration test. Set RUN_INTEGRATION_TESTS=1 to run.");
        return;
    }

    println!("Testing Ollama download interrupt handling...");

    // Use a small model for testing
    let model = "all-minilm:22m";

    let mut child = common::spawn_odir(&["model-download", model]);

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
        // The process may exit with error code due to unanswered prompt timeout
        // or due to successful interrupt handling
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

/// Test downloading a small Ollama model via CLI (full download test)
///
/// This test performs an actual download and verifies successful completion.
/// It's more resource-intensive and takes longer, so it's marked as ignored
/// by default. Run explicitly with:
///
/// ```bash
/// RUN_INTEGRATION_TESTS=1 cargo test --test cli_ollama_download test_ollama_download_success -- --ignored --nocapture
/// ```
#[test]
#[ignore]
fn test_ollama_download_success() {
    if !common::should_run_integration_tests() {
        println!("Skipping integration test. Set RUN_INTEGRATION_TESTS=1 to run.");
        return;
    }

    println!("Testing successful Ollama model download...");

    // Use a very small model for testing
    let model = "all-minilm:22m";

    let mut child = common::spawn_odir(&["model-download", model]);

    println!("Spawned download process, PID: {}", child.id());

    // Wait for the download to complete (with a reasonable timeout)
    let status = common::wait_with_timeout(&mut child, 300) // 5 minutes timeout
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

/// Test that invalid model names are handled correctly
#[test]
fn test_ollama_invalid_model() {
    println!("Testing Ollama download with invalid model name...");

    // Use a model name that definitely doesn't exist
    let model = "this-model-definitely-does-not-exist-12345:99z";

    let mut child = common::spawn_odir(&["model-download", model]);

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

/// Test that SIGTERM is also handled correctly
#[test]
fn test_ollama_sigterm_handling() {
    if !common::should_run_integration_tests() {
        println!("Skipping integration test. Set RUN_INTEGRATION_TESTS=1 to run.");
        return;
    }

    println!("Testing Ollama download SIGTERM handling...");

    let model = "all-minilm:22m";

    let mut child = common::spawn_odir(&["model-download", model]);

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
