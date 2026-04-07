//! Common test utilities for integration tests
#![allow(dead_code)]

use std::path::PathBuf;
use std::process::{Child, Command};
use std::thread;
use std::time::Duration;

/// Get the path to the compiled odir binary
///
/// Prefer Cargo's `CARGO_BIN_EXE_odir` path, which is profile/target-dir aware.
/// Fall back to a local target/debug lookup for non-Cargo invocations.
pub fn get_binary_path() -> PathBuf {
    if let Some(binary) = std::env::var_os("CARGO_BIN_EXE_odir") {
        let binary_path = PathBuf::from(binary);
        assert!(
            binary_path.exists(),
            "Binary from CARGO_BIN_EXE_odir not found at {:?}",
            binary_path
        );
        return binary_path;
    }

    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let mut binary_path = PathBuf::from(manifest_dir);
    binary_path.push("target");
    binary_path.push("debug");

    #[cfg(target_os = "windows")]
    binary_path.push("odir.exe");

    #[cfg(not(target_os = "windows"))]
    binary_path.push("odir");

    assert!(
        binary_path.exists(),
        "Binary not found at {:?}. Run integration tests through Cargo so CARGO_BIN_EXE_odir is set, or build the binary first",
        binary_path
    );

    binary_path
}

/// Spawn the odir binary with the given arguments
///
/// # Arguments
/// * `args` - Command line arguments to pass to the binary
///
/// # Returns
/// * `Child` - The spawned process handle
pub fn spawn_odir(args: &[&str]) -> Child {
    let binary_path = get_binary_path();

    Command::new(binary_path)
        .args(args)
        .spawn()
        .expect("Failed to spawn odir process")
}

/// Send SIGINT (Ctrl+C) signal to a process
///
/// # Arguments
/// * `child` - The process to send the signal to
///
/// # Panics
/// Panics if unable to send the signal
#[cfg(unix)]
pub fn send_sigint(child: &mut Child) {
    use nix::sys::signal::{self, Signal};
    use nix::unistd::Pid;

    let pid = Pid::from_raw(child.id() as i32);
    signal::kill(pid, Signal::SIGINT).expect("Failed to send SIGINT");
}

/// Send SIGINT equivalent on Windows
#[cfg(windows)]
pub fn send_sigint(child: &mut Child) {
    // On Windows, we'll just kill the process since there's no SIGINT equivalent
    // in the same way. For a more sophisticated approach, we'd need to use
    // GenerateConsoleCtrlEvent, but that's complex for this use case.
    child.kill().expect("Failed to terminate process");
}

/// Send SIGTERM signal to a process
///
/// # Arguments
/// * `child` - The process to send the signal to
///
/// # Panics
/// Panics if unable to send the signal
#[cfg(unix)]
pub fn send_sigterm(child: &mut Child) {
    use nix::errno::Errno;
    use nix::sys::signal::{self, Signal};
    use nix::unistd::Pid;

    let pid = Pid::from_raw(child.id() as i32);
    match signal::kill(pid, Signal::SIGTERM) {
        Ok(()) => {}
        // Process already exited — nothing to do.
        Err(Errno::ESRCH) => {}
        Err(e) => panic!("Failed to send SIGTERM: {e}"),
    }
}

/// Send SIGTERM equivalent on Windows
#[cfg(windows)]
pub fn send_sigterm(child: &mut Child) {
    child.kill().expect("Failed to terminate process");
}

/// Wait for a process with a timeout
///
/// # Arguments
/// * `child` - The process to wait for
/// * `timeout_secs` - Maximum seconds to wait
///
/// # Returns
/// * `Option<std::process::ExitStatus>` - The exit status if the process exited within timeout, None otherwise
pub fn wait_with_timeout(child: &mut Child, timeout_secs: u64) -> Option<std::process::ExitStatus> {
    let start = std::time::Instant::now();
    let timeout = Duration::from_secs(timeout_secs);

    loop {
        match child.try_wait() {
            Ok(Some(status)) => return Some(status),
            Ok(None) => {
                if start.elapsed() >= timeout {
                    return None;
                }
                thread::sleep(Duration::from_millis(100));
            }
            Err(_) => return None,
        }
    }
}

/// Check if integration tests should run
///
/// Integration tests are controlled by the environment variable RUN_INTEGRATION_TESTS.
/// Set RUN_INTEGRATION_TESTS=1 to enable network-dependent integration tests. Note that
/// not all tests that require network access are necessarily marked as integration tests.
///
/// # Returns
/// * `bool` - true if integration tests should run, false otherwise
pub fn should_run_integration_tests() -> bool {
    std::env::var("RUN_INTEGRATION_TESTS")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false)
}
