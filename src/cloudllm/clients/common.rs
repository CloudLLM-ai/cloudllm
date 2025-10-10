use crate::client_wrapper::TokenUsage;
use lazy_static::lazy_static;
use openai_rust::chat;
use openai_rust2 as openai_rust;
use reqwest;
use std::error::Error;
use std::time::Duration;
use tokio::sync::Mutex;

lazy_static! {
    /// Shared HTTP client with persistent connection pooling.
    /// This maintains persistent connections, disables per-request DNS/TLS churn,
    /// and provides efficient connection reuse across all client instances.
    static ref SHARED_HTTP_CLIENT: reqwest::Client = {
        reqwest::ClientBuilder::new()
            .pool_idle_timeout(Some(Duration::from_secs(90)))
            .pool_max_idle_per_host(10)
            .tcp_keepalive(Some(Duration::from_secs(60)))
            .timeout(Duration::from_secs(300))
            .connect_timeout(Duration::from_secs(30))
            .build()
            .expect("Failed to build shared HTTP client")
    };
}

/// Get a reference to the shared HTTP client with persistent connections.
/// All clients should use this to benefit from connection pooling.
pub fn get_shared_http_client() -> &'static reqwest::Client {
    &SHARED_HTTP_CLIENT
}

/// Send a chat request, record its usage, and return the assistant’s content.
pub async fn send_and_track(
    api: &openai_rust::Client,
    model: &str,
    formatted_msgs: Vec<chat::Message>,
    url_path: Option<String>,
    usage_slot: &Mutex<Option<TokenUsage>>,
    optional_search_parameters: Option<openai_rust::chat::SearchParameters>,
) -> Result<String, Box<dyn Error>> {
    let mut chat_arguments = chat::ChatArguments::new(model, formatted_msgs);

    if let Some(search_params) = optional_search_parameters {
        chat_arguments = chat_arguments.with_search_parameters(search_params);
    }

    let response = api.create_chat(chat_arguments, url_path).await;

    match response {
        Ok(response) => {
            let usage = TokenUsage {
                input_tokens: response.usage.prompt_tokens as usize,
                output_tokens: response.usage.completion_tokens as usize,
                total_tokens: response.usage.total_tokens as usize,
            };

            // Store it for get_last_usage()
            *usage_slot.lock().await = Some(usage);

            // Return the assistant’s content
            Ok(response.choices[0].message.content.clone())
        }
        Err(err) => {
            if log::log_enabled!(log::Level::Error) {
                log::error!(
                    "cloudllm::clients::common::send_and_track(...): OpenAI API Error: {}",
                    err
                );
            }
            Err(err.into()) // Convert the error to Box<dyn Error>
        }
    }
}
