//! Shared utilities used across provider client implementations.
//!
//! The helpers in this module are useful when implementing additional providers that expose an
//! OpenAI-compatible HTTP surface.  They provide a tuned [`reqwest`] client, convenience
//! functions for sending chat requests, and adapters for streaming responses.
//!
//! # Example: building a custom wrapper
//!
//! ```rust,no_run
//! use std::sync::Arc;
//!
//! use async_trait::async_trait;
//! use cloudllm::client_wrapper::{ClientWrapper, Message, Role, TokenUsage};
//! use cloudllm::clients::common::{get_shared_http_client, send_and_track};
//! use openai_rust2 as openai_rust;
//! use openai_rust2::chat::GrokTool;
//! use tokio::sync::Mutex;
//!
//! struct MyHostedClient {
//!     client: openai_rust::Client,
//!     model: String,
//!     usage: Mutex<Option<TokenUsage>>,
//! }
//!
//! impl MyHostedClient {
//!     fn new(key: &str, base_url: &str, model: &str) -> Self {
//!         Self {
//!             client: openai_rust::Client::new_with_client_and_base_url(
//!                 key,
//!                 get_shared_http_client().clone(),
//!                 base_url,
//!             ),
//!             model: model.to_owned(),
//!             usage: Mutex::new(None),
//!         }
//!     }
//! }
//!
//! #[async_trait]
//! impl ClientWrapper for MyHostedClient {
//!     fn model_name(&self) -> &str {
//!         &self.model
//!     }
//!
//!     async fn send_message(
//!         &self,
//!         messages: &[Message],
//!         optional_grok_tools: Option<Vec<GrokTool>>,
//!     ) -> Result<Message, Box<dyn std::error::Error>> {
//!         let formatted = messages
//!             .iter()
//!             .map(|msg| openai_rust::chat::Message {
//!                 role: match msg.role {
//!                     Role::System => "system".into(),
//!                     Role::User => "user".into(),
//!                     Role::Assistant => "assistant".into(),
//!                 },
//!                 content: msg.content.as_ref().to_owned(),
//!             })
//!             .collect();
//!
//!         let reply = send_and_track(
//!             &self.client,
//!             &self.model,
//!             formatted,
//!             Some("/v1/chat/completions".to_string()),
//!             &self.usage,
//!             optional_grok_tools,
//!         )
//!         .await?;
//!
//!         Ok(Message {
//!             role: Role::Assistant,
//!             content: Arc::<str>::from(reply),
//!         })
//!     }
//! }
//! ```
//!
//! The same helpers can be combined with [`chunks_to_stream`] to wire streaming support into the
//! custom client.

use crate::client_wrapper::{MessageChunk, TokenUsage};
use lazy_static::lazy_static;
use openai_rust::chat;
use openai_rust::chat::{
    GrokTool, OpenAIResponsesArguments, OpenAITool, ResponsesArguments, ResponsesMessage,
};
use openai_rust2 as openai_rust;
use std::error::Error;
use std::time::Duration;
use tokio::sync::Mutex;

lazy_static! {
    /// Shared HTTP client with persistent connection pooling.
    ///
    /// The single client instance keeps TLS sessions and DNS lookups warm which significantly
    /// reduces latency when many concurrent requests are issued to upstream providers.
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

/// Borrow the lazily initialised shared [`reqwest::Client`].
///
/// The returned reference can be cloned and reused by individual client wrappers.
pub fn get_shared_http_client() -> &'static reqwest::Client {
    &SHARED_HTTP_CLIENT
}

/// Send a chat completion request, persist token usage, and surface the assistant content.
///
/// The helper captures the common logic shared by OpenAI-compatible endpoints (OpenAI, Anthropic
/// via the Claude proxy, Gemini, xAI Grok, etc.).
pub async fn send_and_track(
    api: &openai_rust::Client,
    model: &str,
    formatted_msgs: Vec<chat::Message>,
    url_path: Option<String>,
    usage_slot: &Mutex<Option<TokenUsage>>,
    optional_grok_tools: Option<Vec<GrokTool>>,
) -> Result<String, Box<dyn Error>> {
    let mut chat_arguments = chat::ChatArguments::new(model, formatted_msgs);

    if let Some(grok_tools) = optional_grok_tools {
        chat_arguments = chat_arguments.with_grok_tools(grok_tools);
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

/// Send a request to xAI's Responses API (/v1/responses) with agentic tool calling.
///
/// This function is used when grok_tools are provided, as the Responses API uses
/// a different endpoint and request/response format than Chat Completions.
pub async fn send_and_track_responses(
    api: &openai_rust::Client,
    model: &str,
    formatted_msgs: Vec<chat::Message>,
    url_path: Option<String>,
    usage_slot: &Mutex<Option<TokenUsage>>,
    grok_tools: Vec<GrokTool>,
) -> Result<String, Box<dyn Error>> {
    // Convert chat messages to ResponsesMessage format
    let input: Vec<ResponsesMessage> = formatted_msgs
        .into_iter()
        .map(|msg| ResponsesMessage {
            role: msg.role,
            content: msg.content,
        })
        .collect();

    let args = ResponsesArguments::new(model, input).with_tools(grok_tools);

    let response = api.create_responses(args, url_path).await;

    match response {
        Ok(response) => {
            let usage = TokenUsage {
                input_tokens: response.usage.input_tokens as usize,
                output_tokens: response.usage.output_tokens as usize,
                total_tokens: response.usage.total_tokens as usize,
            };

            // Store it for get_last_usage()
            *usage_slot.lock().await = Some(usage);

            // Return the assistant's content
            Ok(response.get_text_content())
        }
        Err(err) => {
            if log::log_enabled!(log::Level::Error) {
                log::error!(
                    "cloudllm::clients::common::send_and_track_responses(...): xAI Responses API Error: {}",
                    err
                );
            }
            Err(err.into())
        }
    }
}

/// Send a request to OpenAI's Responses API (/v1/responses) with agentic tool calling.
///
/// This function is used when openai_tools are provided (web_search, file_search, code_interpreter),
/// as the Responses API uses a different endpoint and request/response format than Chat Completions.
pub async fn send_and_track_openai_responses(
    api: &openai_rust::Client,
    model: &str,
    formatted_msgs: Vec<chat::Message>,
    url_path: Option<String>,
    usage_slot: &Mutex<Option<TokenUsage>>,
    openai_tools: Vec<OpenAITool>,
) -> Result<String, Box<dyn Error>> {
    // Convert chat messages to ResponsesMessage format
    let input: Vec<ResponsesMessage> = formatted_msgs
        .into_iter()
        .map(|msg| ResponsesMessage {
            role: msg.role,
            content: msg.content,
        })
        .collect();

    let args = OpenAIResponsesArguments::new(model, input).with_tools(openai_tools);

    let response = api.create_openai_responses(args, url_path).await;

    match response {
        Ok(response) => {
            let usage = TokenUsage {
                input_tokens: response.usage.input_tokens as usize,
                output_tokens: response.usage.output_tokens as usize,
                total_tokens: response.usage.total_tokens as usize,
            };

            // Store it for get_last_usage()
            *usage_slot.lock().await = Some(usage);

            // Return the assistant's content (with citations extracted)
            Ok(response.get_text_content())
        }
        Err(err) => {
            if log::log_enabled!(log::Level::Error) {
                log::error!(
                    "cloudllm::clients::common::send_and_track_openai_responses(...): OpenAI Responses API Error: {}",
                    err
                );
            }
            Err(err.into())
        }
    }
}

/// Thin error wrapper used when streaming responses fail mid-flight.
#[derive(Debug, Clone)]
pub struct StreamError(pub String);

impl std::fmt::Display for StreamError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl Error for StreamError {}

/// Convert eagerly collected message chunks into a boxed stream suitable for [`ClientWrapper`](crate::client_wrapper::ClientWrapper)
/// implementations.
pub fn chunks_to_stream(
    chunks: Vec<Result<MessageChunk, Box<dyn Error + Send>>>,
) -> crate::client_wrapper::MessageChunkStream {
    let stream = futures_util::stream::iter(
        chunks
            .into_iter()
            .map(|r| r.map_err(|e| e as Box<dyn Error>)),
    );
    Box::pin(stream)
}
