//! Signal handler for graceful shutdown on CTRL+C
//!
//! This module provides functionality to cleanly handle interrupt signals,
//! ensuring that any temporary files or incomplete downloads are properly cleaned up.
//! When an interrupt signal is received, the user is prompted to confirm before proceeding.

use log::{debug, error, info};
use signal_hook::consts::signal::*;
use signal_hook::iterator::Signals;
use std::io::{self, Write};
#[cfg(unix)]
use std::os::unix::io::AsRawFd;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::thread;
#[cfg(not(unix))]
use std::time::Duration;

/// Flag that indicates if the application has been interrupted
pub static INTERRUPTED: AtomicBool = AtomicBool::new(false);
static INTERRUPT_REQUESTED: AtomicBool = AtomicBool::new(false);
static PROGRESS_ACTIVE: AtomicBool = AtomicBool::new(false);
static PENDING_SIGNAL: AtomicUsize = AtomicUsize::new(0);
static CONFIRMATION_REQUIRED: AtomicBool = AtomicBool::new(false);

/// Check if an interrupt signal has been received
pub fn is_interrupted() -> bool {
    INTERRUPTED.load(Ordering::Relaxed)
}

/// Set the interrupted flag
pub fn set_interrupted() {
    INTERRUPTED.store(true, Ordering::Relaxed);
}

/// Enable or disable confirmation prompts for interrupts
pub fn set_confirmation_required(required: bool) {
    CONFIRMATION_REQUIRED.store(required, Ordering::Relaxed);
}

/// Mark whether a progress bar is currently active
pub fn set_progress_active(active: bool) {
    PROGRESS_ACTIVE.store(active, Ordering::Relaxed);
}

/// Check if an interrupt has been requested but not yet confirmed
pub fn interrupt_requested() -> bool {
    INTERRUPT_REQUESTED.load(Ordering::Relaxed)
}

/// Prompt the user to confirm interrupt for a pending signal
/// Returns true if the user confirms the interrupt
pub fn confirm_pending_interrupt() -> bool {
    if !CONFIRMATION_REQUIRED.load(Ordering::Relaxed) {
        return false;
    }

    if !INTERRUPT_REQUESTED.swap(false, Ordering::Relaxed) {
        return false;
    }

    let signal_id = PENDING_SIGNAL.swap(0, Ordering::Relaxed);
    let label = match signal_id {
        x if x == SIGTERM as usize => "Termination",
        _ => "Interrupt",
    };

    if prompt_for_interrupt_confirmation(label) {
        set_interrupted();
        true
    } else {
        false
    }
}

/// Prompt the user to confirm interrupt
/// Returns true if user confirms (Yes/Y), false if user cancels (No/N)
fn prompt_for_interrupt_confirmation(signal_name: &str) -> bool {
    eprint!(
        "\n{}: All partially completed downloaded data will be removed. Do you really want to exit? [y/N] (timeout to N in 10 seconds): ",
        signal_name
    );
    let _ = io::stderr().flush();

    #[cfg(unix)]
    {
        let stdin = io::stdin();
        let fd = stdin.as_raw_fd();
        let mut fds = libc::pollfd {
            fd,
            events: libc::POLLIN,
            revents: 0,
        };

        let timeout_ms = 10_000;
        let poll_result = unsafe { libc::poll(&mut fds as *mut libc::pollfd, 1, timeout_ms) };
        if poll_result > 0 && (fds.revents & libc::POLLIN) != 0 {
            let mut input = String::new();
            match stdin.read_line(&mut input) {
                Ok(_) => {
                    let response = input.trim().to_lowercase();
                    return matches!(response.as_str(), "y" | "yes");
                }
                Err(_) => {
                    error!("Failed to read user input for interrupt confirmation");
                    return false;
                }
            }
        }

        if poll_result == 0 {
            info!("Interrupt confirmation timed out; continuing");
            return false;
        }

        error!("Failed to poll stdin for interrupt confirmation");
        false
    }

    #[cfg(not(unix))]
    {
        let (tx, rx) = std::sync::mpsc::channel();

        thread::spawn(move || {
            let stdin = io::stdin();
            let mut input = String::new();
            let read_result = stdin.read_line(&mut input).map(|_| input);
            let _ = tx.send(read_result);
        });

        match rx.recv_timeout(Duration::from_secs(10)) {
            Ok(Ok(input)) => {
                let response = input.trim().to_lowercase();
                matches!(response.as_str(), "y" | "yes")
            }
            Ok(Err(_)) => {
                error!("Failed to read user input for interrupt confirmation");
                false
            }
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                info!("Interrupt confirmation timed out; continuing");
                false
            }
            Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
                error!("Interrupt confirmation channel disconnected");
                false
            }
        }
    }
}

/// Install a signal handler for graceful shutdown on SIGINT (CTRL+C)
///
/// This function sets up a background thread that listens for SIGINT and SIGTERM signals.
/// When either signal is received, the user is prompted to confirm the interruption.
/// If confirmed, all temporary files are cleaned up and the application exits gracefully.
///
/// # Panics
/// Panics if signal handling setup fails.
pub fn install_signal_handlers() {
    let mut signals = match Signals::new([SIGINT, SIGTERM]) {
        Ok(s) => s,
        Err(e) => {
            error!("Failed to install signal handlers: {}", e);
            panic!("Failed to install signal handlers: {}", e);
        }
    };

    // Spawn a background thread to handle signals
    thread::spawn(move || {
        for sig in signals.forever() {
            match sig {
                SIGINT => {
                    info!("Received SIGINT (CTRL+C)");
                    if !CONFIRMATION_REQUIRED.load(Ordering::Relaxed) {
                        eprintln!("\nInterrupt received. Exiting...");
                        std::process::exit(130); // Standard exit code for SIGINT
                    }

                    if PROGRESS_ACTIVE.load(Ordering::Relaxed) {
                        PENDING_SIGNAL.store(SIGINT as usize, Ordering::Relaxed);
                        INTERRUPT_REQUESTED.store(true, Ordering::Relaxed);
                        continue;
                    }

                    if prompt_for_interrupt_confirmation("Interrupt") {
                        info!("User confirmed interrupt");
                        // eprintln!("Cleaning up downloads...");
                        set_interrupted();
                        // Give the main thread a moment to clean up
                        thread::sleep(std::time::Duration::from_millis(100));
                        std::process::exit(130); // Standard exit code for SIGINT
                    } else {
                        info!("User cancelled interrupt, continuing...");
                        // eprintln!("Continuing...");
                    }
                }
                SIGTERM => {
                    info!("Received SIGTERM");
                    if !CONFIRMATION_REQUIRED.load(Ordering::Relaxed) {
                        eprintln!("\nTermination signal received. Exiting...");
                        std::process::exit(143); // Standard exit code for SIGTERM
                    }

                    if PROGRESS_ACTIVE.load(Ordering::Relaxed) {
                        PENDING_SIGNAL.store(SIGTERM as usize, Ordering::Relaxed);
                        INTERRUPT_REQUESTED.store(true, Ordering::Relaxed);
                        continue;
                    }

                    if prompt_for_interrupt_confirmation("Termination") {
                        info!("User confirmed termination");
                        // eprintln!("Cleaning up downloads...");
                        set_interrupted();
                        // Give the main thread a moment to clean up
                        thread::sleep(std::time::Duration::from_millis(100));
                        std::process::exit(143); // Standard exit code for SIGTERM
                    } else {
                        info!("User cancelled termination, continuing...");
                        // eprintln!("Continuing...");
                    }
                }
                _ => {
                    // This shouldn't happen given our signal list
                }
            }
        }
    });

    debug!("Signal handlers installed successfully");
}
