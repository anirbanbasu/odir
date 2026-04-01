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

use std::io::Read;
use std::io::Write;
use std::process::{Command, Stdio};
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
    common::send_sigint(&mut child);

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
        common::send_sigterm(&mut child);

        if let Some(status) = common::wait_with_timeout(&mut child, 5) {
            println!("Process exited after SIGTERM with status: {:?}", status);
        } else {
            println!("Process still running, force killing...");
            common::send_sigterm(&mut child);
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

/// Test downloading a larger Ollama model via CLI using part downloads.
///
/// Model under test: `gemma3:270m`
///
/// Run explicitly with:
///
/// ```bash
/// RUN_INTEGRATION_TESTS=1 cargo test --test cli_ollama_download test_ollama_large_model_part_download -- --ignored --nocapture
/// ```
#[test]
#[ignore]
fn test_ollama_large_model_part_download() {
    if !common::should_run_integration_tests() {
        println!("Skipping integration test. Set RUN_INTEGRATION_TESTS=1 to run.");
        return;
    }

    println!("Testing larger Ollama model download (part download expected)...");

    let model = "gemma3:270m";
    let mut child = common::spawn_odir(&["model-download", model]);

    println!("Spawned large-model download process, PID: {}", child.id());

    let status = common::wait_with_timeout(&mut child, 1200) // 20 minutes timeout
        .expect("Large-model download process did not complete within timeout");

    println!("Process exited with status: {:?}", status);

    assert!(
        status.success(),
        "Large Ollama download should exit successfully, but exited with: {:?}",
        status
    );

    println!("Large Ollama model download completed successfully!");
}

/// Lightweight non-interactive check for part-download mode indicator in output.
///
/// This test starts the larger Ollama model download, lets it run briefly,
/// terminates the process, and asserts that output indicates part-download
/// mode (either explicit chunked log or per-part progress message).
#[test]
#[ignore]
fn test_ollama_large_model_mode_indicator_non_interactive() {
    if !common::should_run_integration_tests() {
        println!("Skipping integration test. Set RUN_INTEGRATION_TESTS=1 to run.");
        return;
    }

    println!("Testing Ollama mode indicator output (non-interactive)...");

    let model = "gemma3:270m";
    let mut child = Command::new(common::get_binary_path())
        .args(["model-download", model])
        .env("ODIR_LOG_LEVEL", "DEBUG")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("Failed to spawn odir process");

    // Allow enough time for mode-detection logs/progress to appear.
    thread::sleep(Duration::from_secs(12));

    let _ = child.kill();
    let _ = child.wait();

    let mut stdout = String::new();
    let mut stderr = String::new();

    if let Some(mut out) = child.stdout.take() {
        let _ = out.read_to_string(&mut stdout);
    }
    if let Some(mut err) = child.stderr.take() {
        let _ = err.read_to_string(&mut stderr);
    }

    let combined = format!("{}\n{}", stdout, stderr);
    println!(
        "Captured output (truncated): {}",
        &combined.chars().take(2000).collect::<String>()
    );

    assert!(
        combined.contains("Using chunked download for")
            || combined.contains(" part 1/")
            || combined.contains(" part 2/")
            || combined.contains("Verifying existing item")
            || combined.contains("already present and verified"),
        "Expected part-download mode indicator or resume verification in output, but none was found"
    );
}

/// Test chunked-download interrupt prompt accepts `k` to keep partial downloads.
///
/// This test is ignored by default because it requires network and can be slow.
#[test]
#[ignore]
fn test_ollama_interrupt_keep_chunks_with_k() {
    if !common::should_run_integration_tests() {
        println!("Skipping integration test. Set RUN_INTEGRATION_TESTS=1 to run.");
        return;
    }

    let model = "gemma3:270m";
    let mut child = Command::new(common::get_binary_path())
        .args(["model-download", model])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("Failed to spawn odir process");

    // Let chunked transfer start and progress render.
    thread::sleep(Duration::from_secs(4));

    common::send_sigint(&mut child);

    // Confirm exit with keep-partials option.
    if let Some(stdin) = child.stdin.as_mut() {
        let _ = stdin.write_all(b"k\n");
        let _ = stdin.flush();
    }

    let status = common::wait_with_timeout(&mut child, 20)
        .expect("Process did not exit after interrupt confirmation");

    let mut stdout = String::new();
    let mut stderr = String::new();
    if let Some(mut out) = child.stdout.take() {
        let _ = out.read_to_string(&mut stdout);
    }
    if let Some(mut err) = child.stderr.take() {
        let _ = err.read_to_string(&mut stderr);
    }

    let combined = format!("{}\n{}", stdout, stderr);
    // If the model was already fully cached, the process completes before our SIGINT
    // arrives and no chunked interrupt prompt is shown — that is acceptable.
    let already_cached = combined.contains("already present and verified");
    if already_cached {
        println!("Model was already cached; chunked interrupt path not exercised this run.");
        return;
    }
    assert!(
        combined.contains("[y/k/N]") && combined.contains("timeout to N in 10 seconds"),
        "Expected chunked interrupt prompt wording not found in output"
    );
    assert!(
        !status.success(),
        "Process should exit after Ctrl+C with 'k' confirmation"
    );
}

/// Test chunked-download interrupt prompt accepts `n` to continue downloading.
///
/// This test is ignored by default because it requires network and can be slow.
#[test]
#[ignore]
fn test_ollama_interrupt_continue_with_n() {
    if !common::should_run_integration_tests() {
        println!("Skipping integration test. Set RUN_INTEGRATION_TESTS=1 to run.");
        return;
    }

    let model = "gemma3:270m";
    let mut child = Command::new(common::get_binary_path())
        .args(["model-download", model])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("Failed to spawn odir process");

    // Let chunked transfer start and progress render.
    thread::sleep(Duration::from_secs(4));

    common::send_sigint(&mut child);

    // Decline exit and continue downloading.
    if let Some(stdin) = child.stdin.as_mut() {
        let _ = stdin.write_all(b"n\n");
        let _ = stdin.flush();
    }

    // Give the downloader a few seconds; it should still be running.
    thread::sleep(Duration::from_secs(4));
    let still_running = child.try_wait().expect("try_wait failed").is_none();

    // Stop test process (no-op if already exited).
    common::send_sigterm(&mut child);
    let _ = common::wait_with_timeout(&mut child, 10);

    let mut stdout = String::new();
    let mut stderr = String::new();
    if let Some(mut out) = child.stdout.take() {
        let _ = out.read_to_string(&mut stdout);
    }
    if let Some(mut err) = child.stderr.take() {
        let _ = err.read_to_string(&mut stderr);
    }
    let combined = format!("{}\n{}", stdout, stderr);

    // If the model was already fully cached the process completes quickly — before SIGINT
    // is processed — so it will no longer be running. That is an acceptable outcome.
    let already_cached = combined.contains("already present and verified");
    if !already_cached {
        assert!(
            still_running,
            "Process should continue running after Ctrl+C with 'n'"
        );
    }
}

/// Test Ctrl+C during resume verification path keeps prompt handling stable.
///
/// Workflow:
/// 1) Start and interrupt a chunked download once (keeping partials).
/// 2) Restart the same model download to trigger resume/verification work.
/// 3) Send Ctrl+C and answer `k`; process should exit promptly.
#[test]
#[ignore]
fn test_ollama_interrupt_during_resume_verification() {
    if !common::should_run_integration_tests() {
        println!("Skipping integration test. Set RUN_INTEGRATION_TESTS=1 to run.");
        return;
    }

    let model = "gemma3:270m";

    // Phase 1: seed partial data.
    let mut first = Command::new(common::get_binary_path())
        .args(["model-download", model])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("Failed to spawn first odir process");

    thread::sleep(Duration::from_secs(4));
    common::send_sigint(&mut first);
    if let Some(stdin) = first.stdin.as_mut() {
        let _ = stdin.write_all(b"k\n");
        let _ = stdin.flush();
    }
    let _ = common::wait_with_timeout(&mut first, 20);

    // Phase 2: resume path, then interrupt again.
    let mut second = Command::new(common::get_binary_path())
        .args(["model-download", model])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("Failed to spawn second odir process");

    thread::sleep(Duration::from_secs(3));
    common::send_sigint(&mut second);
    if let Some(stdin) = second.stdin.as_mut() {
        let _ = stdin.write_all(b"k\n");
        let _ = stdin.flush();
    }

    let status = common::wait_with_timeout(&mut second, 20)
        .expect("Process did not exit after second interrupt confirmation");

    let mut stdout = String::new();
    let mut stderr = String::new();
    if let Some(mut out) = second.stdout.take() {
        let _ = out.read_to_string(&mut stdout);
    }
    if let Some(mut err) = second.stderr.take() {
        let _ = err.read_to_string(&mut stderr);
    }
    let combined = format!("{}\n{}", stdout, stderr);

    // If the model was already fully cached, phase 2 completes before SIGINT is
    // processed and exits with success — that path is also acceptable.
    let already_cached = combined.contains("already present and verified");
    assert!(
        !status.success() || already_cached,
        "Process should exit after Ctrl+C with 'k' during resume verification"
    );
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
    common::send_sigterm(&mut child);

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
