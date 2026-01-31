//! OpenAI Chat Completions client that captures token usage statistics.
//!
//! # Key Features
//!
//! - **`send_message`**: returns a `Message` compatible with the higher level [`LLMSession`](crate::LLMSession) API.
//! - **Automatic usage capture**: the last token accounting is stored in a shared slot.
//! - **Streaming support**: `send_message_stream` converts streamed responses into [`MessageChunk`] values.
//! - **Image generation**: Uses DALL-E 3 for generating images from prompts via the [`ImageGenerationClient`] trait.
//!
//! # Example
//!
//! ```rust
//! use std::sync::Arc;
//!
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
//!         Message { role: Role::System,    content: Arc::<str>::from("You are an assistant.") },
//!         Message { role: Role::User,      content: Arc::<str>::from("Hello!") },
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
//! # Streaming usage
//!
//! ```rust,no_run
//! use std::sync::Arc;
//!
//! use cloudllm::client_wrapper::{ClientWrapper, Message, Role};
//! use cloudllm::clients::openai::{Model, OpenAIClient};
//! use futures_util::StreamExt;
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let key = std::env::var("OPEN_AI_SECRET")?;
//!     let client = OpenAIClient::new_with_model_enum(&key, Model::GPT41Mini);
//!     let request = [Message {
//!         role: Role::User,
//!         content: Arc::<str>::from("Stream a limerick about async Rust."),
//!     }];
//!
//!     if let Some(mut stream) = client.send_message_stream(&request, None).await? {
//!         while let Some(chunk) = stream.next().await {
//!             print!("{}", chunk?.content);
//!         }
//!     }
//!     Ok(())
//! }
//! ```
//!
//! # Image Generation with DALL-E 3
//!
//! ```rust,no_run
//! use std::sync::Arc;
//!
//! use cloudllm::clients::openai::{OpenAIClient, Model};
//! use cloudllm::image_generation::{ImageGenerationClient, ImageGenerationOptions};
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let key = std::env::var("OPEN_AI_SECRET")?;
//!     let client = OpenAIClient::new_with_model_enum(&key, Model::GPT41Nano);
//!
//!     let options = ImageGenerationOptions {
//!         aspect_ratio: Some("16:9".to_string()),
//!         num_images: Some(1),
//!         response_format: Some("url".to_string()),
//!     };
//!
//!     let response = client.generate_image(
//!         "A futuristic city at sunset",
//!         options,
//!     ).await?;
//!
//!     for image in response.images {
//!         if let Some(url) = image.url {
//!             println!("Generated: {}", url);
//!         }
//!     }
//!     Ok(())
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
use crate::clients::common::{
    chunks_to_stream, send_and_track, send_and_track_openai_responses, StreamError,
};
use crate::cloudllm::client_wrapper::{ClientWrapper, Message, Role};
use crate::cloudllm::image_generation::{
    ImageData, ImageGenerationClient, ImageGenerationOptions, ImageGenerationResponse,
};
use tokio::sync::Mutex;

/// Image generation model identifiers for OpenAI.
#[allow(non_camel_case_types)]
pub enum ImageModel {
    /// `gpt-image-1.5` – Current OpenAI image generation model
    GPTImage1_5,
}

/// Convert an [`ImageModel`] variant into the string identifier expected by the API.
pub fn image_model_to_string(model: ImageModel) -> String {
    match model {
        ImageModel::GPTImage1_5 => "gpt-image-1.5".to_string(),
    }
}

/// Official model identifiers supported by OpenAI's Chat Completions API.
#[allow(non_camel_case_types)]
pub enum Model {
    /// `gpt-5.2` – Complex reasoning, broad world knowledge, and code-heavy or multi-step agentic tasks
    GPT52,
    /// `gpt-5.2-chat-latest` – ChatGPT's production deployment of GPT-5.2.
    GPT52ChatLatest,
    /// `gpt-5.2-pro` – Tough problems that may take longer to solve but require harder thinking
    GPT52Pro,
    /// `gpt-5.1 - flagship for coding and agentic tasks with configurable reasoning and non-reasoning effort.
    GPT51,
    /// `gpt-5` – high-reasoning, medium latency, text or multimodal input.
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
        Model::GPT52 => "gpt-5.2".to_string(),
        Model::GPT52ChatLatest => "gpt-5.2-chat-latest".to_string(),
        Model::GPT52Pro => "gpt-5.2-pro".to_string(),
        Model::GPT51 => "gpt-5.1".to_string(),
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
    /// API key needed for image generation
    api_key: String,
    /// Base URL for API calls
    base_url: String,
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
        let base_url = "https://api.openai.com/v1";
        OpenAIClient {
            client: openai_rust::Client::new_with_client(
                secret_key,
                get_shared_http_client().clone(),
            ),
            model: model_name.to_string(),
            token_usage: Mutex::new(None),
            api_key: secret_key.to_string(),
            base_url: base_url.to_string(),
        }
    }

    /// Construct a client targeting a custom OpenAI compatible base URL.
    /// Note: base_url should not have a trailing slash (e.g., "https://api.openai.com/v1")
    pub fn new_with_base_url(secret_key: &str, model_name: &str, base_url: &str) -> Self {
        use crate::clients::common::get_shared_http_client;
        let base_url_normalized = base_url.trim_end_matches('/');
        OpenAIClient {
            client: openai_rust::Client::new_with_client_and_base_url(
                secret_key,
                get_shared_http_client().clone(),
                &format!("{}/", base_url_normalized),
            ),
            model: model_name.to_string(),
            token_usage: Mutex::new(None),
            api_key: secret_key.to_string(),
            base_url: base_url_normalized.to_string(),
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
    async fn send_message(
        &self,
        messages: &[Message],
        optional_grok_tools: Option<Vec<openai_rust::chat::GrokTool>>,
        optional_openai_tools: Option<Vec<openai_rust::chat::OpenAITool>>,
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

        // Use the Responses API when OpenAI tools are provided, otherwise use Chat Completions
        let result = if let Some(openai_tools) = optional_openai_tools {
            // Use the Responses API (/v1/responses) for agentic tool calling
            send_and_track_openai_responses(
                &self.client,
                &self.model,
                formatted_messages,
                Some("/v1/responses".to_string()),
                &self.token_usage,
                openai_tools,
            )
            .await
        } else {
            // Use the standard Chat Completions API (/v1/chat/completions)
            send_and_track(
                &self.client,
                &self.model,
                formatted_messages,
                Some("/v1/chat/completions".to_string()),
                &self.token_usage,
                optional_grok_tools,
            )
            .await
        };

        match result {
            Ok(c) => Ok(Message {
                role: Role::Assistant,
                content: Arc::from(c.as_str()),
            }),
            Err(e) => {
                if log::log_enabled!(log::Level::Error) {
                    log::error!("OpenAIClient::send_message(...): OpenAI API Error: {}", e);
                }
                Err(e)
            }
        }
    }

    fn send_message_stream<'a>(
        &'a self,
        messages: &'a [Message],
        optional_grok_tools: Option<Vec<openai_rust::chat::GrokTool>>,
        _optional_openai_tools: Option<Vec<openai_rust::chat::OpenAITool>>,
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
            if let Some(grok_tools) = optional_grok_tools {
                chat_arguments = chat_arguments.with_grok_tools(grok_tools);
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

    fn model_name(&self) -> &str {
        &self.model
    }

    fn usage_slot(&self) -> Option<&Mutex<Option<TokenUsage>>> {
        Some(&self.token_usage)
    }
}

#[async_trait]
impl ImageGenerationClient for OpenAIClient {
    async fn generate_image(
        &self,
        prompt: &str,
        options: ImageGenerationOptions,
    ) -> Result<ImageGenerationResponse, Box<dyn Error + Send + Sync>> {
        // Map aspect ratio to OpenAI's supported sizes
        // gpt-image-1.5 supports: 1024x1024, 1024x1536 (portrait), 1536x1024 (landscape), auto
        let size = match options.aspect_ratio.as_deref() {
            Some("16:9") | Some("4:3") | Some("3:2") => "1536x1024", // Landscape options
            Some("9:16") | Some("3:4") | Some("2:3") => "1024x1536", // Portrait options
            Some("1:1") | None => "1024x1024", // Square (default)
            Some(ratio) => {
                if log::log_enabled!(log::Level::Warn) {
                    log::warn!(
                        "OpenAI unsupported aspect ratio '{}', using 1024x1024",
                        ratio
                    );
                }
                "1024x1024"
            }
        };

        let n = options.num_images.unwrap_or(1).min(10); // OpenAI max is 10

        let model_name = image_model_to_string(ImageModel::GPTImage1_5);

        // Build request body for direct HTTP call (honors base_url)
        let request_body = serde_json::json!({
            "model": model_name,
            "prompt": prompt,
            "n": n,
            "size": size,
        });

        if log::log_enabled!(log::Level::Debug) {
            log::debug!(
                "OpenAI image generation: model={}, n={}, size={}",
                model_name,
                n,
                size
            );
        }

        // Make direct HTTP request to honor base_url
        let http_client = crate::clients::common::get_shared_http_client();
        let url = format!("{}/images/generations", self.base_url);

        let response = http_client
            .post(&url)
            .header("Authorization", format!("Bearer {}", &self.api_key))
            .json(&request_body)
            .send()
            .await
            .map_err(|e| {
                let err_msg = format!("OpenAI image generation error: {}", e);
                if log::log_enabled!(log::Level::Error) {
                    log::error!("{}", err_msg);
                }
                Box::<dyn Error + Send + Sync>::from(err_msg)
            })?;

        let response_text = response.text().await?;

        if log::log_enabled!(log::Level::Debug) {
            log::debug!("OpenAI image API response: {}", response_text);
        }

        let response_json: serde_json::Value = serde_json::from_str(&response_text)?;

        // Check for API errors
        if let Some(error) = response_json.get("error") {
            if let Some(message) = error.get("message").and_then(|m| m.as_str()) {
                return Err(format!("OpenAI API error: {}", message).into());
            }
            return Err("OpenAI API returned an error".into());
        }

        // Parse the response - OpenAI returns: { "data": [{ "url": "..." }, ...] }
        let mut images = Vec::new();

        if let Some(data_array) = response_json.get("data").and_then(|d| d.as_array()) {
            for item in data_array {
                if let Some(url) = item.get("url").and_then(|u| u.as_str()) {
                    if !url.is_empty() {
                        images.push(ImageData {
                            url: Some(url.to_string()),
                            b64_json: None,
                        });
                    }
                } else if let Some(b64) = item.get("b64_json").and_then(|b| b.as_str()) {
                    if !b64.is_empty() {
                        images.push(ImageData {
                            url: None,
                            b64_json: Some(b64.to_string()),
                        });
                    }
                }
            }
        }

        if images.is_empty() {
            return Err("No images in OpenAI response".into());
        }

        Ok(ImageGenerationResponse {
            images,
            revised_prompt: None,
        })
    }

    fn model_name(&self) -> &str {
        "gpt-image-1.5" // ImageModel::GPTImage1_5
    }
}
