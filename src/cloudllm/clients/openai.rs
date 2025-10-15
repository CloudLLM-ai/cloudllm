//! The `OpenAIClient` struct implements `ClientWrapper` for OpenAI’s Chat API,
//! capturing both the assistant response and detailed token usage (input vs output)
//! for cost tracking.
//!
//! # Key Features
//!
//! - **send_message(...)**: unchanged signature; returns a `Message` as before.
//! - **Automatic Usage Capture**: stores the latest `TokenUsage` (input_tokens, output_tokens, total_tokens) internally.
//! - **Inspect Usage**: call `get_last_usage()` after `send_message()` to retrieve actual usage stats.
//!
//! # Example
//!
//! ```rust
//! use cloudllm::clients::openai::{OpenAIClient, Model};
//! use cloudllm::client_wrapper::{ClientWrapper, Message, Role};
//!
//! #[tokio::main]
//! async fn main() {
//!     // Initialize with your API key and model enum.
//!     let secret_key : String = std::env::var("OPEN_AI_SECRET").expect("OPEN_AI_SECRET not set");
//!     let client = OpenAIClient::new_with_model_enum(&secret_key, Model::GPT41Nano);
//!
//!     // Send system + user messages.
//!     let resp = client.send_message(&vec![
//!         Message { role: Role::System,    content: "You are an assistant.".into() },
//!         Message { role: Role::User,      content: "Hello!".into() },
//!     ], None).await.unwrap();
//!     println!("Assistant: {}", resp.content);
//!
//!     // Then pull the real token usage.
//!     if let Some(usage) = client.get_last_usage().await {
//!         println!(
//!             "Tokens — input: {}, output: {}, total: {}",
//!             usage.input_tokens, usage.output_tokens, usage.total_tokens
//!         );
//!     }
//! }
//! ```
//!
//! # Note
//!
//! Make sure `OPENAI_API_KEY` is set and pick a valid model name (e.g. `"gpt-4.1-nano"`).
use std::error::Error;
use std::sync::Arc;

use async_trait::async_trait;
use futures_util::stream::StreamExt;
use openai_rust::chat;
use openai_rust2 as openai_rust;

use crate::client_wrapper::{MessageChunk, TokenUsage};
use crate::clients::common::{chunks_to_stream, send_and_track, StreamError};
use crate::cloudllm::client_wrapper::{ClientWrapper, Message, Role};
use tokio::sync::Mutex;

/// Official model identifiers supported by OpenAI's Chat Completions API.
#[allow(non_camel_case_types)]
pub enum Model {
    /// `gpt-5` – high reasoning, medium latency, text or multimodal input.
    GPT5,
    /// `gpt-5-mini` – fast variant of GPT-5 with balanced cost and quality.
    GPT5Mini,
    /// `gpt-5-nano` – lowest latency GPT-5 configuration.
    GPT5Nano,
    /// `gpt-5-chat-latest` – ChatGPT's production deployment of GPT-5.
    GPT5ChatLatest,
    /// `gpt-4o` – Omni model with text + image inputs.
    GPT4o,
    /// `chatgpt-4o-latest` – the ChatGPT tuned interface to GPT-4o.
    ChatGPT4oLatest,
    /// `gpt-4o-mini` – cost effective GPT-4o derivative.
    GPt4oMini,
    /// `o1` – reasoning-focused O-series frontier model.
    O1,
    /// `o1-mini` – faster/cheaper O-series offering.
    O1Mini,
    /// `o1-preview` – preview build of the O1 family.
    O1Preview,
    /// `o3-mini` – compact successor in the O-series.
    O3Mini,
    /// `o4-mini` – newest O-series low-latency tier.
    O4Mini,
    /// `o4-mini-high` – higher accuracy variant of `o4-mini`.
    O4MiniHigh,
    /// `o3` – general availability O-series release.
    O3,
    /// `gpt-4o-realtime-preview` – realtime WebRTC capable GPT-4o.
    GPT4oRealtimePreview,
    /// `gpt-4o-mini-realtime-preview` – lightweight realtime GPT-4o.
    GPT4oMiniRealtimePreview,
    /// `gpt-4o-audio-preview` – GPT-4o tuned for audio conversations.
    GPT4oAudioPreview,
    /// `gpt-4.5-preview` – preview of the 4.5 Omni upgrade.
    GPT45Preview,
    /// `gpt-4.1` – general availability GPT-4.1.
    GPT41,
    /// `gpt-4.1-mini` – reduced cost GPT-4.1 tier.
    GPT41Mini,
    /// `gpt-4.1-nano` – ultra low cost GPT-4.1 derivative.
    GPT41Nano,
}

/// Convert a [`Model`] variant into the string identifier expected by the REST API.
pub fn model_to_string(model: Model) -> String {
    match model {
        Model::GPT5 => "gpt-5".to_string(),
        Model::GPT5Mini => "gpt-5-mini".to_string(),
        Model::GPT5Nano => "gpt-5-nano".to_string(),
        Model::GPT5ChatLatest => "gpt-5-chat-latest".to_string(),
        Model::GPT4o => "gpt-4o".to_string(),
        Model::ChatGPT4oLatest => "chatgpt-4o-latest".to_string(),
        Model::GPt4oMini => "gpt-4o-mini".to_string(),
        Model::O1 => "o1".to_string(),
        Model::O1Mini => "o1-mini".to_string(),
        Model::O1Preview => "o1-preview".to_string(),
        Model::O3Mini => "o3-mini".to_string(),
        Model::O4Mini => "o4-mini".to_string(),
        Model::O4MiniHigh => "o4-mini-high".to_string(),
        Model::O3 => "o3".to_string(),
        Model::GPT4oRealtimePreview => "gpt-4o-realtime-preview".to_string(),
        Model::GPT4oMiniRealtimePreview => "gpt-4o-mini-realtime-preview".to_string(),
        Model::GPT4oAudioPreview => "gpt-4o-audio-preview".to_string(),
        Model::GPT45Preview => "gpt-4.5-preview".to_string(),
        Model::GPT41 => "gpt-4.1".to_string(),
        Model::GPT41Mini => "gpt-4.1-mini".to_string(),
        Model::GPT41Nano => "gpt-4.1-nano".to_string(),
    }
}

/// Client wrapper for OpenAI's Chat Completions API.
///
/// The wrapper maintains the selected model identifier plus an internal [`TokenUsage`] slot so
/// callers can inspect how many tokens each request consumed.  It reuses the shared HTTP client
/// configured in [`crate::cloudllm::clients::common`].
pub struct OpenAIClient {
    /// Underlying SDK client pointing at the REST endpoint.
    client: openai_rust::Client,
    /// Model name that will be injected into each request.
    model: String,
    /// Storage for the token usage returned by the most recent request.
    token_usage: Mutex<Option<TokenUsage>>,
}

impl OpenAIClient {
    /// Construct a new client using the provided API key and [`Model`] variant.
    pub fn new_with_model_enum(secret_key: &str, model: Model) -> Self {
        Self::new_with_model_string(secret_key, &model_to_string(model))
    }

    /// Construct a new client using the provided API key and explicit model name.
    ///
    /// This is the most general constructor and can be used for unofficial model identifiers
    /// (e.g. OpenAI compatible self-hosted deployments).
    pub fn new_with_model_string(secret_key: &str, model_name: &str) -> Self {
        use crate::clients::common::get_shared_http_client;
        OpenAIClient {
            client: openai_rust::Client::new_with_client(
                secret_key,
                get_shared_http_client().clone(),
            ),
            model: model_name.to_string(),
            token_usage: Mutex::new(None),
        }
    }

    /// Construct a client targeting a custom OpenAI compatible base URL.
    pub fn new_with_base_url(secret_key: &str, model_name: &str, base_url: &str) -> Self {
        use crate::clients::common::get_shared_http_client;
        OpenAIClient {
            client: openai_rust::Client::new_with_client_and_base_url(
                secret_key,
                get_shared_http_client().clone(),
                base_url,
            ),
            model: model_name.to_string(),
            token_usage: Mutex::new(None),
        }
    }

    /// Convenience helper wrapping [`OpenAIClient::new_with_base_url`] for strongly typed models.
    pub fn new_with_base_url_and_model_enum(
        secret_key: &str,
        model: Model,
        base_url: &str,
    ) -> Self {
        Self::new_with_base_url(secret_key, &model_to_string(model), base_url)
    }
}

#[async_trait]
impl ClientWrapper for OpenAIClient {
    fn model_name(&self) -> &str {
        &self.model
    }

    async fn send_message(
        &self,
        messages: &[Message],
        optional_search_parameters: Option<openai_rust::chat::SearchParameters>,
    ) -> Result<Message, Box<dyn Error>> {
        // Convert the provided messages into the format expected by openai_rust
        let mut formatted_messages = Vec::with_capacity(messages.len());
        for msg in messages {
            formatted_messages.push(chat::Message {
                role: match msg.role {
                    Role::System => "system".to_owned(),
                    Role::User => "user".to_owned(),
                    Role::Assistant => "assistant".to_owned(),
                },
                content: msg.content.to_string(),
            });
        }

        let url_path_string = "/v1/chat/completions".to_string();

        let result = send_and_track(
            &self.client,
            &self.model,
            formatted_messages,
            Some(url_path_string),
            &self.token_usage,
            optional_search_parameters,
        )
        .await;

        match result {
            Ok(c) => Ok(Message {
                role: Role::Assistant,
                content: Arc::from(c.as_str()),
            }),
            Err(_) => {
                if log::log_enabled!(log::Level::Error) {
                    log::error!(
                        "OpenAIClient::send_message(...): OpenAI API Error: {}",
                        "Error occurred while sending message"
                    );
                }
                Err("Error occurred while sending message".into())
            }
        }
    }

    fn send_message_stream<'a>(
        &'a self,
        messages: &'a [Message],
        optional_search_parameters: Option<openai_rust::chat::SearchParameters>,
    ) -> crate::client_wrapper::MessageStreamFuture<'a> {
        Box::pin(async move {
            // Convert the provided messages into the format expected by openai_rust
            let mut formatted_messages = Vec::with_capacity(messages.len());
            for msg in messages {
                formatted_messages.push(chat::Message {
                    role: match msg.role {
                        Role::System => "system".to_owned(),
                        Role::User => "user".to_owned(),
                        Role::Assistant => "assistant".to_owned(),
                    },
                    content: msg.content.to_string(),
                });
            }

            let url_path_string = "/v1/chat/completions".to_string();

            // Build the chat arguments
            let mut chat_arguments = chat::ChatArguments::new(&self.model, formatted_messages);
            if let Some(search_params) = optional_search_parameters {
                chat_arguments = chat_arguments.with_search_parameters(search_params);
            }

            // Create the streaming request
            let stream_result = self
                .client
                .create_chat_stream(chat_arguments, Some(url_path_string))
                .await;

            match stream_result {
                Ok(mut chunk_stream) => {
                    // Collect all chunks into a Vec
                    let mut chunks: Vec<Result<MessageChunk, Box<dyn Error + Send>>> = Vec::new();

                    while let Some(chunk_result) = chunk_stream.next().await {
                        let message_chunk: Result<MessageChunk, Box<dyn Error + Send>> =
                            match chunk_result {
                                Ok(chunk) => {
                                    // Extract content and finish_reason from the chunk
                                    let content = chunk
                                        .choices
                                        .first()
                                        .and_then(|choice| choice.delta.content.clone())
                                        .unwrap_or_default();

                                    let finish_reason = chunk
                                        .choices
                                        .first()
                                        .and_then(|choice| choice.finish_reason.clone());

                                    Ok(MessageChunk {
                                        content,
                                        finish_reason,
                                    })
                                }
                                Err(err) => {
                                    if log::log_enabled!(log::Level::Error) {
                                        log::error!(
                                    "OpenAIClient::send_message_stream(...): Stream chunk error: {}",
                                    err
                                );
                                    }
                                    Err(Box::new(StreamError(format!(
                                        "Stream chunk error: {}",
                                        err
                                    )))
                                        as Box<dyn Error + Send>)
                                }
                            };

                        chunks.push(message_chunk);
                    }

                    // Convert the collected chunks into a stream
                    Ok(Some(chunks_to_stream(chunks)))
                }
                Err(err) => {
                    if log::log_enabled!(log::Level::Error) {
                        log::error!(
                            "OpenAIClient::send_message_stream(...): OpenAI API Error: {}",
                            err
                        );
                    }
                    Err(err.into())
                }
            }
        })
    }

    fn usage_slot(&self) -> Option<&Mutex<Option<TokenUsage>>> {
        Some(&self.token_usage)
    }
}
