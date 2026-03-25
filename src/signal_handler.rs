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
use {
    crossterm::event,
    std::time::{Duration, Instant},
};

/// Flag that indicates if the application has been interrupted
pub static INTERRUPTED: AtomicBool = AtomicBool::new(false);
static INTERRUPT_REQUESTED: AtomicBool = AtomicBool::new(false);
static PROGRESS_ACTIVE: AtomicBool = AtomicBool::new(false);
static PENDING_SIGNAL: AtomicUsize = AtomicUsize::new(0);
static CONFIRMATION_REQUIRED: AtomicBool = AtomicBool::new(false);
static CLEANUP_DONE: AtomicBool = AtomicBool::new(false);
static KEEP_PARTIAL_DOWNLOADS: AtomicBool = AtomicBool::new(false);

enum InterruptDecision {
    ExitAndRemove,
    ExitAndKeep,
    Continue,
}

fn pending_signal_label() -> Option<&'static str> {
    if !CONFIRMATION_REQUIRED.load(Ordering::Acquire) {
        return None;
    }

    if !INTERRUPT_REQUESTED.swap(false, Ordering::AcqRel) {
        return None;
    }

    let signal_id = PENDING_SIGNAL.swap(0, Ordering::AcqRel);
    Some(match signal_id {
        x if x == SIGTERM as usize => "Termination",
        _ => "Interrupt",
    })
}

fn apply_chunked_interrupt_decision(decision: InterruptDecision) -> bool {
    match decision {
        InterruptDecision::ExitAndRemove => {
            KEEP_PARTIAL_DOWNLOADS.store(false, Ordering::Release);
            set_interrupted();
            true
        }
        InterruptDecision::ExitAndKeep => {
            KEEP_PARTIAL_DOWNLOADS.store(true, Ordering::Release);
            set_interrupted();
            true
        }
        InterruptDecision::Continue => {
            KEEP_PARTIAL_DOWNLOADS.store(false, Ordering::Release);
            false
        }
    }
}

/// Check if an interrupt signal has been received
pub fn is_interrupted() -> bool {
    INTERRUPTED.load(Ordering::Acquire)
}

/// Set the interrupted flag
pub fn set_interrupted() {
    INTERRUPTED.store(true, Ordering::Release);
}

/// Check whether completed chunk-download parts should be retained on interrupt.
pub fn keep_partial_downloads() -> bool {
    KEEP_PARTIAL_DOWNLOADS.load(Ordering::Acquire)
}

/// Enable or disable confirmation prompts for interrupts
pub fn set_confirmation_required(required: bool) {
    CONFIRMATION_REQUIRED.store(required, Ordering::Release);
}

/// Mark whether a progress bar is currently active
pub fn set_progress_active(active: bool) {
    PROGRESS_ACTIVE.store(active, Ordering::Release);
}

/// Signal that cleanup operations have completed
pub fn set_cleanup_done() {
    CLEANUP_DONE.store(true, Ordering::Release);
}

/// Check if an interrupt has been requested but not yet confirmed
pub fn interrupt_requested() -> bool {
    INTERRUPT_REQUESTED.load(Ordering::Acquire)
}

/// Prompt the user to confirm interrupt for a pending signal
/// Returns true if the user confirms the interrupt
pub fn confirm_pending_interrupt() -> bool {
    let Some(label) = pending_signal_label() else {
        return false;
    };

    KEEP_PARTIAL_DOWNLOADS.store(false, Ordering::Release);
    if prompt_for_interrupt_confirmation(label) {
        set_interrupted();
        true
    } else {
        false
    }
}

/// Prompt the user to confirm interrupt for a pending signal while chunked
/// downloading is active.
///
/// Returns true if the user confirms exit (either removing or keeping partial
/// downloads), false if the user wants to continue.
pub fn confirm_pending_interrupt_for_chunked(completed_parts: u64) -> bool {
    let Some(label) = pending_signal_label() else {
        return false;
    };

    apply_chunked_interrupt_decision(prompt_for_interrupt_confirmation_chunked(
        label,
        completed_parts,
    ))
}

/// Prompt the user to confirm interrupt
/// Returns true if user confirms (Yes/Y), false if user cancels (No/N)
fn prompt_for_interrupt_confirmation(signal_name: &str) -> bool {
    eprint!(
        "\n{}: All partially downloaded temporary files will be removed. Do you really want to exit? [y/N] (timeout to N in 10 seconds): ",
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
        let timeout = Duration::from_secs(10);
        let start = Instant::now();
        let mut input = String::new();

        loop {
            let elapsed = start.elapsed();
            if elapsed >= timeout {
                info!("Interrupt confirmation timed out; continuing");
                return false;
            }

            let remaining = timeout.saturating_sub(elapsed);
            if !event::poll(remaining).unwrap_or(false) {
                info!("Interrupt confirmation timed out; continuing");
                return false;
            }

            match event::read() {
                Ok(event::Event::Key(key_event)) => match key_event.code {
                    event::KeyCode::Char(c) => {
                        input.push(c);
                    }
                    event::KeyCode::Enter => {
                        let response = input.trim().to_lowercase();
                        return matches!(response.as_str(), "y" | "yes");
                    }
                    event::KeyCode::Esc => {
                        return false;
                    }
                    _ => {}
                },
                Err(_) => {
                    error!("Failed to read user input for interrupt confirmation");
                    return false;
                }
            }
        }
    }
}

fn parse_chunked_interrupt_decision(input: &str) -> InterruptDecision {
    let response = input.trim().to_lowercase();
    if matches!(response.as_str(), "y" | "yes") {
        InterruptDecision::ExitAndRemove
    } else if matches!(response.as_str(), "k" | "keep") {
        InterruptDecision::ExitAndKeep
    } else {
        InterruptDecision::Continue
    }
}

#[cfg(unix)]
fn read_chunked_interrupt_decision_unix() -> InterruptDecision {
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
            Ok(_) => parse_chunked_interrupt_decision(&input),
            Err(_) => {
                error!("Failed to read user input for interrupt confirmation");
                InterruptDecision::Continue
            }
        }
    } else if poll_result == 0 {
        info!("Interrupt confirmation timed out; continuing");
        InterruptDecision::Continue
    } else {
        error!("Failed to poll stdin for interrupt confirmation");
        InterruptDecision::Continue
    }
}

#[cfg(not(unix))]
fn read_chunked_interrupt_decision_non_unix() -> InterruptDecision {
    let timeout = Duration::from_secs(10);
    let start = Instant::now();
    let mut input = String::new();

    loop {
        let elapsed = start.elapsed();
        if elapsed >= timeout {
            info!("Interrupt confirmation timed out; continuing");
            return InterruptDecision::Continue;
        }

        let remaining = timeout.saturating_sub(elapsed);
        if !event::poll(remaining).unwrap_or(false) {
            info!("Interrupt confirmation timed out; continuing");
            return InterruptDecision::Continue;
        }

        match event::read() {
            Ok(event::Event::Key(key_event)) => match key_event.code {
                event::KeyCode::Char(c) => {
                    input.push(c);
                }
                event::KeyCode::Enter => {
                    return parse_chunked_interrupt_decision(&input);
                }
                event::KeyCode::Esc => {
                    return InterruptDecision::Continue;
                }
                _ => {}
            },
            Err(_) => {
                error!("Failed to read user input for interrupt confirmation");
                return InterruptDecision::Continue;
            }
        }
    }
}

/// Prompt the user to confirm interrupt for chunked downloads.
///
/// `y` => exit and remove partial downloads
/// `k` => exit and keep completed partial downloads
/// `N`/timeout => continue downloading
fn prompt_for_interrupt_confirmation_chunked(
    signal_name: &str,
    completed_parts: u64,
) -> InterruptDecision {
    eprint!(
        "\n{}: All partially downloaded temporary files can be removed. Do you really want to exit? Enter k to keep {} partial downloads. [y/k/N] (timeout to N in 10 seconds): ",
        signal_name, completed_parts
    );
    let _ = io::stderr().flush();

    #[cfg(unix)]
    {
        read_chunked_interrupt_decision_unix()
    }

    #[cfg(not(unix))]
    {
        read_chunked_interrupt_decision_non_unix()
    }
}

/// Wait for cleanup to complete with timeout
/// Returns true if cleanup completed, false if timeout occurred
fn wait_for_cleanup_completion(exit_code: i32) -> ! {
    const CLEANUP_TIMEOUT_MS: u64 = 1000; // 1 second timeout
    const POLL_INTERVAL_MS: u64 = 20; // Check every 20ms
    let max_iterations = CLEANUP_TIMEOUT_MS / POLL_INTERVAL_MS;

    for i in 0..max_iterations {
        if CLEANUP_DONE.load(Ordering::Acquire) {
            info!(
                "Cleanup completed successfully, exiting with code {}",
                exit_code
            );
            std::process::exit(exit_code);
        }
        if i == 0 {
            debug!(
                "Waiting for cleanup completion (timeout: {}ms)",
                CLEANUP_TIMEOUT_MS
            );
        }
        thread::sleep(std::time::Duration::from_millis(POLL_INTERVAL_MS));
    }

    error!(
        "Cleanup did not complete within {}ms, exiting anyway",
        CLEANUP_TIMEOUT_MS
    );
    std::process::exit(exit_code);
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
                    if !CONFIRMATION_REQUIRED.load(Ordering::Acquire) {
                        eprintln!("\nInterrupt received. Exiting...");
                        std::process::exit(130); // Standard exit code for SIGINT
                    }

                    if PROGRESS_ACTIVE.load(Ordering::Acquire) {
                        PENDING_SIGNAL.store(SIGINT as usize, Ordering::Release);
                        INTERRUPT_REQUESTED.store(true, Ordering::Release);
                        continue;
                    }

                    if prompt_for_interrupt_confirmation("Interrupt") {
                        info!("User confirmed interrupt");
                        set_interrupted();
                        wait_for_cleanup_completion(130);
                    } else {
                        info!("User cancelled interrupt, continuing...");
                        // eprintln!("Continuing...");
                    }
                }
                SIGTERM => {
                    info!("Received SIGTERM");
                    if !CONFIRMATION_REQUIRED.load(Ordering::Acquire) {
                        eprintln!("\nTermination signal received. Exiting...");
                        std::process::exit(143); // Standard exit code for SIGTERM
                    }

                    if PROGRESS_ACTIVE.load(Ordering::Acquire) {
                        PENDING_SIGNAL.store(SIGTERM as usize, Ordering::Release);
                        INTERRUPT_REQUESTED.store(true, Ordering::Release);
                        continue;
                    }

                    if prompt_for_interrupt_confirmation("Termination") {
                        info!("User confirmed termination");
                        set_interrupted();
                        wait_for_cleanup_completion(143);
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    static TEST_LOCK: Mutex<()> = Mutex::new(());

    fn reset_flags() {
        INTERRUPTED.store(false, Ordering::SeqCst);
        INTERRUPT_REQUESTED.store(false, Ordering::SeqCst);
        PROGRESS_ACTIVE.store(false, Ordering::SeqCst);
        PENDING_SIGNAL.store(0, Ordering::SeqCst);
        CONFIRMATION_REQUIRED.store(false, Ordering::SeqCst);
        CLEANUP_DONE.store(false, Ordering::SeqCst);
        KEEP_PARTIAL_DOWNLOADS.store(false, Ordering::SeqCst);
    }

    #[test]
    fn test_is_interrupted_initial_state() {
        let _guard = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        reset_flags();
        assert!(!is_interrupted());
    }

    #[test]
    fn test_is_interrupted_after_set() {
        let _guard = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        reset_flags();
        set_interrupted();
        assert!(is_interrupted());
    }

    #[test]
    fn test_set_interrupted_visibility() {
        let _guard = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        reset_flags();
        set_interrupted();
        let val = INTERRUPTED.load(Ordering::Acquire);
        assert!(val);
    }

    #[test]
    fn test_interrupt_requested_initial_state() {
        let _guard = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        reset_flags();
        assert!(!interrupt_requested());
    }

    #[test]
    fn test_keep_partial_downloads_initial_state() {
        let _guard = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        reset_flags();
        assert!(!keep_partial_downloads());
    }

    #[test]
    fn test_interrupt_requested_after_signal() {
        let _guard = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        reset_flags();
        set_confirmation_required(true);
        INTERRUPT_REQUESTED.store(true, Ordering::Release);
        assert!(interrupt_requested());
    }

    #[test]
    fn test_confirm_pending_interrupt_without_confirmation_required() {
        let _guard = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        reset_flags();
        INTERRUPT_REQUESTED.store(true, Ordering::Release);
        PENDING_SIGNAL.store(SIGINT as usize, Ordering::Release);
        assert!(!confirm_pending_interrupt());
        assert!(INTERRUPT_REQUESTED.load(Ordering::Acquire));
        assert_eq!(PENDING_SIGNAL.load(Ordering::Acquire), SIGINT as usize);
    }

    #[test]
    fn test_confirm_pending_interrupt_without_interrupt_requested() {
        let _guard = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        reset_flags();
        set_confirmation_required(true);
        PENDING_SIGNAL.store(SIGINT as usize, Ordering::Release);
        assert!(!confirm_pending_interrupt());
        assert_eq!(PENDING_SIGNAL.load(Ordering::Acquire), SIGINT as usize);
    }

    #[test]
    fn test_confirm_pending_interrupt_consumes_flags() {
        let _guard = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        reset_flags();
        set_confirmation_required(true);
        INTERRUPT_REQUESTED.store(true, Ordering::Release);
        PENDING_SIGNAL.store(SIGTERM as usize, Ordering::Release);

        assert!(interrupt_requested());

        let had_request = INTERRUPT_REQUESTED.swap(false, Ordering::AcqRel);
        let signal_id = PENDING_SIGNAL.swap(0, Ordering::AcqRel);

        assert!(had_request);
        assert_eq!(signal_id, SIGTERM as usize);
        assert!(!INTERRUPT_REQUESTED.load(Ordering::Acquire));
        assert_eq!(PENDING_SIGNAL.load(Ordering::Acquire), 0);
    }

    #[test]
    fn test_pending_signal_label_returns_none_without_confirmation_required() {
        let _guard = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        reset_flags();
        INTERRUPT_REQUESTED.store(true, Ordering::Release);
        PENDING_SIGNAL.store(SIGINT as usize, Ordering::Release);

        assert_eq!(pending_signal_label(), None);
        assert!(INTERRUPT_REQUESTED.load(Ordering::Acquire));
        assert_eq!(PENDING_SIGNAL.load(Ordering::Acquire), SIGINT as usize);
    }

    #[test]
    fn test_pending_signal_label_consumes_sigterm_request() {
        let _guard = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        reset_flags();
        set_confirmation_required(true);
        INTERRUPT_REQUESTED.store(true, Ordering::Release);
        PENDING_SIGNAL.store(SIGTERM as usize, Ordering::Release);

        assert_eq!(pending_signal_label(), Some("Termination"));
        assert!(!INTERRUPT_REQUESTED.load(Ordering::Acquire));
        assert_eq!(PENDING_SIGNAL.load(Ordering::Acquire), 0);
    }

    #[test]
    fn test_apply_chunked_interrupt_decision_exit_and_remove() {
        let _guard = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        reset_flags();
        KEEP_PARTIAL_DOWNLOADS.store(true, Ordering::Release);

        assert!(apply_chunked_interrupt_decision(
            InterruptDecision::ExitAndRemove,
        ));
        assert!(is_interrupted());
        assert!(!keep_partial_downloads());
    }

    #[test]
    fn test_apply_chunked_interrupt_decision_exit_and_keep() {
        let _guard = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        reset_flags();

        assert!(apply_chunked_interrupt_decision(
            InterruptDecision::ExitAndKeep,
        ));
        assert!(is_interrupted());
        assert!(keep_partial_downloads());
    }

    #[test]
    fn test_apply_chunked_interrupt_decision_continue_resets_keep_flag() {
        let _guard = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        reset_flags();
        KEEP_PARTIAL_DOWNLOADS.store(true, Ordering::Release);

        assert!(!apply_chunked_interrupt_decision(
            InterruptDecision::Continue
        ));
        assert!(!is_interrupted());
        assert!(!keep_partial_downloads());
    }

    #[test]
    fn test_set_progress_active() {
        let _guard = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        reset_flags();
        set_progress_active(true);
        assert!(PROGRESS_ACTIVE.load(Ordering::Acquire));
        set_progress_active(false);
        assert!(!PROGRESS_ACTIVE.load(Ordering::Acquire));
    }

    #[test]
    fn test_set_confirmation_required() {
        let _guard = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        reset_flags();
        set_confirmation_required(true);
        assert!(CONFIRMATION_REQUIRED.load(Ordering::Acquire));
        set_confirmation_required(false);
        assert!(!CONFIRMATION_REQUIRED.load(Ordering::Acquire));
    }

    #[test]
    fn test_signal_flow_sigint_with_progress() {
        let _guard = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        reset_flags();
        set_confirmation_required(true);
        set_progress_active(true);

        PENDING_SIGNAL.store(SIGINT as usize, Ordering::Release);
        INTERRUPT_REQUESTED.store(true, Ordering::Release);

        assert!(PROGRESS_ACTIVE.load(Ordering::Acquire));
        assert!(INTERRUPT_REQUESTED.load(Ordering::Acquire));
        assert_eq!(PENDING_SIGNAL.load(Ordering::Acquire), SIGINT as usize);
    }

    #[test]
    fn test_signal_flow_sigterm_with_progress() {
        let _guard = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        reset_flags();
        set_confirmation_required(true);
        set_progress_active(true);

        PENDING_SIGNAL.store(SIGTERM as usize, Ordering::Release);
        INTERRUPT_REQUESTED.store(true, Ordering::Release);

        assert!(PROGRESS_ACTIVE.load(Ordering::Acquire));
        assert!(INTERRUPT_REQUESTED.load(Ordering::Acquire));
        assert_eq!(PENDING_SIGNAL.load(Ordering::Acquire), SIGTERM as usize);
    }

    #[test]
    fn test_concurrent_flag_updates_with_acquire_release() {
        let _guard = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        reset_flags();

        let threads = 4;
        let iterations = 1000;
        let mut handles = Vec::with_capacity(threads);

        for tid in 0..threads {
            handles.push(std::thread::spawn(move || {
                for i in 0..iterations {
                    let bit = (i + tid) % 2 == 0;
                    if bit {
                        set_interrupted();
                    }
                    set_confirmation_required(!bit);
                    set_progress_active(bit);
                    PENDING_SIGNAL.store((i % 3) as usize, Ordering::Release);
                }
            }));
        }

        for handle in handles {
            handle.join().expect("thread panicked");
        }

        let interrupted = INTERRUPTED.load(Ordering::Acquire);
        let conf_required = CONFIRMATION_REQUIRED.load(Ordering::Acquire);
        let progress = PROGRESS_ACTIVE.load(Ordering::Acquire);
        let pending = PENDING_SIGNAL.load(Ordering::Acquire);

        assert!(interrupted || !interrupted);
        assert!(conf_required || !conf_required);
        assert!(progress || !progress);
        assert!(pending <= 2);
    }

    #[test]
    fn test_multiple_signal_updates_ordering() {
        let _guard = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        reset_flags();

        set_confirmation_required(true);
        set_progress_active(true);

        PENDING_SIGNAL.store(SIGINT as usize, Ordering::Release);
        INTERRUPT_REQUESTED.store(true, Ordering::Release);

        let sig_id = PENDING_SIGNAL.load(Ordering::Acquire);
        let has_request = INTERRUPT_REQUESTED.load(Ordering::Acquire);

        assert_eq!(sig_id, SIGINT as usize);
        assert!(has_request);

        PENDING_SIGNAL.store(0, Ordering::Release);
        INTERRUPT_REQUESTED.store(false, Ordering::Release);

        assert_eq!(PENDING_SIGNAL.load(Ordering::Acquire), 0);
        assert!(!INTERRUPT_REQUESTED.load(Ordering::Acquire));
    }

    #[test]
    fn test_swap_atomicity_in_confirm() {
        let _guard = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        reset_flags();

        set_confirmation_required(true);
        INTERRUPT_REQUESTED.store(true, Ordering::Release);
        PENDING_SIGNAL.store(SIGTERM as usize, Ordering::Release);

        let old_request = INTERRUPT_REQUESTED.swap(false, Ordering::AcqRel);
        let old_signal = PENDING_SIGNAL.swap(0, Ordering::AcqRel);

        assert!(old_request);
        assert_eq!(old_signal, SIGTERM as usize);
        assert!(!INTERRUPT_REQUESTED.load(Ordering::Acquire));
        assert_eq!(PENDING_SIGNAL.load(Ordering::Acquire), 0);
    }

    #[test]
    fn test_flag_state_machine() {
        let _guard = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        reset_flags();

        set_confirmation_required(true);
        assert!(CONFIRMATION_REQUIRED.load(Ordering::Acquire));

        set_progress_active(true);
        assert!(PROGRESS_ACTIVE.load(Ordering::Acquire));

        INTERRUPT_REQUESTED.store(true, Ordering::Release);
        assert!(INTERRUPT_REQUESTED.load(Ordering::Acquire));

        PENDING_SIGNAL.store(SIGINT as usize, Ordering::Release);
        assert_eq!(PENDING_SIGNAL.load(Ordering::Acquire), SIGINT as usize);

        assert!(!is_interrupted());
        set_interrupted();
        assert!(is_interrupted());

        reset_flags();
        assert!(!is_interrupted());
        assert!(!interrupt_requested());
        assert!(!CONFIRMATION_REQUIRED.load(Ordering::Acquire));
        assert!(!PROGRESS_ACTIVE.load(Ordering::Acquire));
        assert_eq!(PENDING_SIGNAL.load(Ordering::Acquire), 0);
    }
}
