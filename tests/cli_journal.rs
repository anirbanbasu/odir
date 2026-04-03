//! Integration tests for journal CLI flows that do not require network access.

mod common;

use sha2::{Digest, Sha256};
use std::fs;
use std::io::{Read, Write};
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::time::{SystemTime, UNIX_EPOCH};
use tempfile::TempDir;

struct TestSandbox {
    _root: TempDir,
    home_dir: PathBuf,
    xdg_data_home: PathBuf,
}

impl TestSandbox {
    fn new() -> Self {
        let root = TempDir::new().expect("create temp test dir");
        let home_dir = root.path().join("home");
        let xdg_data_home = root.path().join("xdg-data");

        fs::create_dir_all(&home_dir).expect("create sandbox home");
        fs::create_dir_all(&xdg_data_home).expect("create sandbox XDG data dir");

        Self {
            _root: root,
            home_dir,
            xdg_data_home,
        }
    }

    fn apply_env(&self, command: &mut Command) {
        command.env("HOME", &self.home_dir);
        command.env("XDG_DATA_HOME", &self.xdg_data_home);

        #[cfg(target_os = "windows")]
        {
            command.env("USERPROFILE", &self.home_dir);
            command.env("LOCALAPPDATA", self.home_dir.join("AppData").join("Local"));
            command.env("APPDATA", self.home_dir.join("AppData").join("Roaming"));
        }
    }

    fn journal_dir(&self) -> PathBuf {
        let data_local_dir = self.platform_data_local_dir();
        let journal_dir = data_local_dir.join("journals");
        fs::create_dir_all(&journal_dir).expect("create journals dir");
        journal_dir
    }

    fn platform_data_local_dir(&self) -> PathBuf {
        #[cfg(any(
            target_os = "linux",
            target_os = "freebsd",
            target_os = "dragonfly",
            target_os = "netbsd",
            target_os = "openbsd",
            target_os = "android"
        ))]
        {
            return self.xdg_data_home.join("odir");
        }

        #[cfg(any(target_os = "macos", target_os = "ios"))]
        {
            return self
                .home_dir
                .join("Library")
                .join("Application Support")
                .join("odir");
        }

        #[cfg(target_os = "windows")]
        {
            return self
                .home_dir
                .join("AppData")
                .join("Local")
                .join("odir")
                .join("data");
        }
    }
}

fn unique_model_identifier() -> String {
    let ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("clock error")
        .as_nanos();
    format!("journal-test-{}:q4", ts)
}

fn journal_path_for_ollama(sandbox: &TestSandbox, model_identifier: &str) -> std::path::PathBuf {
    let journal_dir = sandbox.journal_dir();

    let mut hasher = Sha256::new();
    hasher.update(b"ollama");
    hasher.update(b"::");
    hasher.update(model_identifier.as_bytes());
    let digest = hex::encode(hasher.finalize());
    journal_dir.join(format!("ollama_{}.json", digest))
}

fn write_test_journal(path: &std::path::Path, model_identifier: &str) {
    let payload = serde_json::json!({
        "model_identifier": model_identifier,
        "source_type": "ollama",
        "tag_or_quant": "q4",
        "started_at": 100,
        "updated_at": 200,
        "items": [
            {
                "digest": "sha256:aaaa",
                "media_type": "application/test",
                "size": 111,
                "state": "completed"
            },
            {
                "digest": "sha256:bbbb",
                "media_type": "application/test",
                "size": 222,
                "state": "pending"
            }
        ]
    });
    fs::write(
        path,
        serde_json::to_vec_pretty(&payload).expect("serialize json"),
    )
    .expect("write journal");
}

fn write_completed_test_journal(path: &std::path::Path, model_identifier: &str) {
    let payload = serde_json::json!({
        "model_identifier": model_identifier,
        "source_type": "ollama",
        "tag_or_quant": "q4",
        "started_at": 100,
        "updated_at": 200,
        "items": [
            {
                "digest": "sha256:aaaa",
                "media_type": "application/test",
                "size": 111,
                "state": "completed"
            },
            {
                "digest": "sha256:bbbb",
                "media_type": "application/test",
                "size": 222,
                "state": "completed"
            }
        ]
    });
    fs::write(
        path,
        serde_json::to_vec_pretty(&payload).expect("serialize json"),
    )
    .expect("write journal");
}

#[test]
fn test_journal_display_includes_completed_percentage() {
    let sandbox = TestSandbox::new();
    let model_identifier = unique_model_identifier();
    let journal_path = journal_path_for_ollama(&sandbox, &model_identifier);
    write_test_journal(&journal_path, &model_identifier);

    let mut cmd = Command::new(common::get_binary_path());
    sandbox.apply_env(&mut cmd);
    let output = cmd
        .args(["journal", &model_identifier, "--source", "ollama"])
        .output()
        .expect("run odir journal");

    let combined = format!(
        "{}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );

    assert!(
        output.status.success(),
        "journal command should succeed, output: {}",
        combined
    );
    assert!(
        combined.contains("Completed: 50.00% (1/2 items)"),
        "Expected completed percentage line not found in output: {}",
        combined
    );
    assert!(
        combined.contains("Started At:") && combined.contains(","),
        "Expected human-readable Started At timestamp in output: {}",
        combined
    );
    assert!(
        combined.contains("Updated At:") && (combined.contains('+') || combined.contains('-')),
        "Expected human-readable Updated At timestamp with timezone offset in output: {}",
        combined
    );
}

#[test]
fn test_journal_clear_completed_uses_safe_clear_message() {
    let sandbox = TestSandbox::new();
    let model_identifier = unique_model_identifier();
    let journal_path = journal_path_for_ollama(&sandbox, &model_identifier);
    write_completed_test_journal(&journal_path, &model_identifier);

    let mut cmd = Command::new(common::get_binary_path());
    sandbox.apply_env(&mut cmd);
    let mut child = cmd
        .args([
            "journal",
            &model_identifier,
            "--source",
            "ollama",
            "--clear",
        ])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn journal clear command");

    if let Some(stdin) = child.stdin.as_mut() {
        let _ = stdin.write_all(b"\n");
        let _ = stdin.flush();
    }

    let status = child.wait().expect("wait for command");

    let mut stdout = String::new();
    let mut stderr = String::new();
    if let Some(mut out) = child.stdout.take() {
        let _ = out.read_to_string(&mut stdout);
    }
    if let Some(mut err) = child.stderr.take() {
        let _ = err.read_to_string(&mut stderr);
    }

    let combined = format!("{}\n{}", stdout, stderr);
    assert!(status.success(), "clear command failed: {}", combined);
    assert!(
        combined.contains("this model download has completed, hence its journal can be safely cleared without deleting the downloaded model")
            || combined.contains("This model download has completed, hence its journal can be safely cleared without deleting the downloaded model"),
        "Expected completed-journal confirmation text not found: {}",
        combined
    );
    assert!(
        combined.contains("Journal clear cancelled"),
        "Expected clear cancellation message not found: {}",
        combined
    );
    assert!(
        journal_path.exists(),
        "Journal file should remain when default answer is No"
    );
}

#[test]
fn test_journal_clear_default_no_keeps_journal() {
    let sandbox = TestSandbox::new();
    let model_identifier = unique_model_identifier();
    let journal_path = journal_path_for_ollama(&sandbox, &model_identifier);
    write_test_journal(&journal_path, &model_identifier);

    let mut cmd = Command::new(common::get_binary_path());
    sandbox.apply_env(&mut cmd);
    let mut child = cmd
        .args([
            "journal",
            &model_identifier,
            "--source",
            "ollama",
            "--clear",
        ])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn journal clear command");

    if let Some(stdin) = child.stdin.as_mut() {
        let _ = stdin.write_all(b"\n");
        let _ = stdin.flush();
    }

    let status = child.wait().expect("wait for command");

    let mut stdout = String::new();
    let mut stderr = String::new();
    if let Some(mut out) = child.stdout.take() {
        let _ = out.read_to_string(&mut stdout);
    }
    if let Some(mut err) = child.stderr.take() {
        let _ = err.read_to_string(&mut stderr);
    }

    let combined = format!("{}\n{}", stdout, stderr);
    assert!(status.success(), "clear command failed: {}", combined);
    assert!(
        combined.contains("Do you really want to delete the download journal")
            && combined.contains("[y/N]"),
        "Expected clear confirmation prompt was not shown: {}",
        combined
    );
    assert!(
        combined.contains("Journal clear cancelled"),
        "Expected clear cancellation message not found: {}",
        combined
    );
    assert!(
        journal_path.exists(),
        "Journal file should remain when default answer is No"
    );
}
