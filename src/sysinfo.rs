use log::{debug, warn};
use regex::Regex;
use std::env;
use std::path::PathBuf;
use sysinfo::{Pid, ProcessesToUpdate, System};
use uzers::{get_group_by_gid, get_user_by_uid};

const PROCESS_NAME: &str = "ollama";

#[derive(Debug, Default)]
pub struct ProcessOwner {
    pub username: String,
    #[allow(dead_code)]
    pub uid: u32,
    pub groupname: String,
    #[allow(dead_code)]
    pub gid: u32,
}

#[derive(Debug)]
pub struct OllamaSystemInfo {
    os_name: String,
    process_id: Option<Pid>,
    process_env_vars: std::collections::HashMap<String, String>,
    parent_process_id: Option<Pid>,
    process_owner: Option<ProcessOwner>,
    listening_on: Option<String>,
    models_dir_path: Option<String>,
    likely_daemon: Option<bool>,
}

impl OllamaSystemInfo {
    pub fn new() -> Self {
        let os_name = std::env::consts::OS.to_string();
        Self {
            os_name,
            process_id: None,
            process_env_vars: std::collections::HashMap::new(),
            parent_process_id: None,
            process_owner: None,
            listening_on: None,
            models_dir_path: None,
            likely_daemon: None,
        }
    }

    pub fn is_windows(&self) -> bool {
        self.os_name.to_lowercase() == "windows"
    }

    #[allow(dead_code)]
    pub fn is_linux(&self) -> bool {
        self.os_name.to_lowercase() == "linux"
    }

    pub fn is_macos(&self) -> bool {
        self.os_name.to_lowercase() == "darwin" || self.os_name.to_lowercase() == "macos"
    }

    pub fn is_running(&mut self) -> bool {
        if self.process_id.is_some() {
            return true;
        }

        let mut system = System::new();
        system.refresh_processes(ProcessesToUpdate::All, true);

        for (pid, process) in system.processes() {
            let proc_name = process.name().to_string_lossy().to_lowercase();
            // Check for exact match or with .exe extension (Windows)
            if proc_name == PROCESS_NAME || proc_name == format!("{}.exe", PROCESS_NAME) {
                self.process_id = Some(*pid);
                debug!("Ollama process found with PID {:?}.", pid);

                // Get environment variables
                self.process_env_vars = process
                    .environ()
                    .iter()
                    .filter_map(|s| {
                        let s = s.to_string_lossy();
                        s.split_once('=')
                            .map(|(k, v)| (k.to_string(), v.to_string()))
                    })
                    .collect();

                if !self.process_env_vars.is_empty() {
                    debug!(
                        "{} environment variables of process {} were obtained.",
                        self.process_env_vars.len(),
                        PROCESS_NAME
                    );
                } else {
                    warn!(
                        "Environment variables of process {} cannot be retrieved. Run auto-config as super-user.",
                        PROCESS_NAME
                    );
                }

                break;
            }
        }

        if self.process_id.is_none() {
            warn!("Ollama process not found. Maybe, it is not installed or it is not running.");
        }

        self.process_id.is_some()
    }

    pub fn get_parent_process_id(&mut self) -> Option<Pid> {
        if self.parent_process_id.is_some() {
            return self.parent_process_id;
        }

        if !self.is_running() {
            return None;
        }

        let mut system = System::new();
        system.refresh_processes(ProcessesToUpdate::All, true);

        if let Some(pid) = self.process_id
            && let Some(process) = system.process(pid)
        {
            self.parent_process_id = process.parent();
        }

        self.parent_process_id
    }

    pub fn get_process_owner(&mut self) -> Option<&ProcessOwner> {
        if self.process_owner.is_some() {
            return self.process_owner.as_ref();
        }

        if !self.is_running() {
            return None;
        }

        let mut system = System::new();
        system.refresh_processes(ProcessesToUpdate::All, true);

        if let Some(pid) = self.process_id
            && let Some(process) = system.process(pid)
        {
            #[cfg(unix)]
            {
                if let (Some(uid), Some(gid)) = (process.user_id(), process.group_id()) {
                    let uid_val = **uid;
                    let gid_val = *gid;

                    let username = get_user_by_uid(uid_val)
                        .map(|u| u.name().to_string_lossy().to_string())
                        .unwrap_or_default();

                    let groupname = get_group_by_gid(gid_val)
                        .map(|g| g.name().to_string_lossy().to_string())
                        .unwrap_or_default();

                    self.process_owner = Some(ProcessOwner {
                        username,
                        uid: uid_val,
                        groupname,
                        gid: gid_val,
                    });

                    debug!(
                        "Owner of process {} ({:?}): {:?}",
                        PROCESS_NAME, pid, self.process_owner
                    );
                }
            }

            #[cfg(not(unix))]
            {
                // Windows - only get username
                if let Some(user) = process.user_id() {
                    self.process_owner = Some(ProcessOwner {
                        username: user.to_string(),
                        uid: 0,
                        groupname: String::new(),
                        gid: 0,
                    });
                }
            }
        }

        self.process_owner.as_ref()
    }

    #[allow(dead_code)]
    pub fn is_model_dir_env_var_set(&self) -> bool {
        self.process_env_vars.contains_key("OLLAMA_MODELS")
    }

    pub fn infer_listening_on(&mut self) -> Option<String> {
        if self.listening_on.is_some() {
            return self.listening_on.clone();
        }

        if !self.is_running() {
            return None;
        }

        let mut system = System::new();
        system.refresh_processes(ProcessesToUpdate::All, true);

        if let Some(pid) = self.process_id {
            // Try to get from environment variable first
            if let Some(host) = self.process_env_vars.get("OLLAMA_HOST") {
                // Ensure it starts with http:// or https://
                let url = if host.starts_with("http://") || host.starts_with("https://") {
                    host.clone()
                } else {
                    format!("http://{}", host)
                };
                self.listening_on = Some(url);
                return self.listening_on.clone();
            }

            // Try to find listening connections (Unix only)
            #[cfg(unix)]
            {
                use std::process::Command;

                // Use lsof or netstat to find listening ports for this PID
                let output = Command::new("lsof")
                    .args(["-Pan", "-p", &pid.to_string(), "-i", "TCP"])
                    .output();

                if let Ok(output) = output
                    && output.status.success()
                {
                    let stdout = String::from_utf8_lossy(&output.stdout);
                    for line in stdout.lines() {
                        if line.contains("LISTEN") {
                            // Parse the line to extract the port
                            if let Some(addr_part) = line.split_whitespace().nth(8) {
                                // Format is typically *:port or address:port
                                if let Some(port) = addr_part.split(':').next_back() {
                                    let url = format!("http://127.0.0.1:{}", port);
                                    self.listening_on = Some(url);
                                    return self.listening_on.clone();
                                }
                            }
                        }
                    }
                }
            }

            // Default listening address
            self.listening_on = Some("http://127.0.0.1:11434".to_string());
        }

        self.listening_on.clone()
    }

    pub fn infer_models_dir_path(&mut self) -> Option<String> {
        if self.models_dir_path.is_some() {
            return self.models_dir_path.clone();
        }

        // First check environment variable
        if let Some(models_path) = self.process_env_vars.get("OLLAMA_MODELS") {
            self.models_dir_path = Some(models_path.clone());
            return self.models_dir_path.clone();
        }

        // Try to infer from Ollama API
        if let Some(url) = self.infer_listening_on() {
            // Try to query Ollama API for model list
            let client = reqwest::blocking::Client::new();
            let list_url = format!("{}/api/tags", url.trim_end_matches('/'));

            if let Ok(response) = client.get(&list_url).send()
                && let Ok(data) = response.json::<serde_json::Value>()
                && let Some(models) = data["models"].as_array()
            {
                if let Some(first_model) = models.first() {
                    if let Some(model_name) = first_model["name"].as_str() {
                        // Query the model details to get the modelfile
                        let show_url = format!("{}/api/show", url.trim_end_matches('/'));
                        let body = serde_json::json!({
                            "name": model_name
                        });

                        if let Ok(show_response) = client.post(&show_url).json(&body).send()
                            && let Ok(show_data) = show_response.json::<serde_json::Value>()
                            && let Some(modelfile) = show_data["modelfile"].as_str()
                        {
                            // Parse the FROM line to extract the blob path
                            let re = Regex::new(r"(?m)^\s*FROM\s+(.+?)(?:\s*#.*)?$").unwrap();
                            if let Some(captures) = re.captures(modelfile)
                                && let Some(blob_path) = captures.get(1)
                            {
                                let path = PathBuf::from(blob_path.as_str().trim());
                                // Get parent's parent directory
                                if let Some(parent) = path.parent()
                                    && let Some(grandparent) = parent.parent()
                                {
                                    let models_dir = grandparent.to_string_lossy();
                                    // Replace home directory with ~
                                    if let Ok(home) = env::var("HOME") {
                                        let result = models_dir.replace(&home, "~");
                                        self.models_dir_path = Some(result);
                                        return self.models_dir_path.clone();
                                    }
                                }
                            }
                        }
                    }
                } else {
                    warn!(
                        "No models are currently installed in Ollama. Cannot infer the models directory path."
                    );
                }
            }
        }

        self.models_dir_path.clone()
    }

    pub fn is_likely_daemon(&mut self) -> bool {
        if let Some(likely) = self.likely_daemon {
            return likely;
        }

        let parent_pid = self.get_parent_process_id();

        // Check if parent is PID 1 or doesn't exist
        if parent_pid.is_none() || parent_pid == Some(Pid::from(1)) {
            // Additional checks for daemon-like behavior
            if let Some(owner) = self.get_process_owner() {
                let username_lower = owner.username.to_lowercase();
                if username_lower == "ollama" || username_lower == "root" {
                    self.likely_daemon = Some(true);
                    return true;
                }
            }
        }

        self.likely_daemon = Some(false);
        false
    }

    #[allow(dead_code)]
    pub fn get_url(&self) -> Option<&String> {
        self.listening_on.as_ref()
    }

    #[allow(dead_code)]
    pub fn get_models_path(&self) -> Option<&String> {
        self.models_dir_path.as_ref()
    }
}

impl Default for OllamaSystemInfo {
    fn default() -> Self {
        Self::new()
    }
}
