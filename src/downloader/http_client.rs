use crate::config::OllamaLibrary;
use reqwest::blocking::Client;

use super::model_downloader::Result;

pub fn build_registry_client(user_agent: &str, settings: &OllamaLibrary) -> Result<Client> {
    // This is the only approved location for disabling certificate verification.
    // codeql[rust/disabled-certificate-check]
    let client = Client::builder()
        .user_agent(user_agent)
        .danger_accept_invalid_certs(!settings.verify_ssl)
        .timeout(std::time::Duration::from_secs_f64(settings.timeout))
        .build()?;

    Ok(client)
}
