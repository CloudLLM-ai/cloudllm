//! xAI Grok client wrapper with support for the Responses API.
//!
//! The `GrokClient` connects to xAI's Grok models. When tools are provided (e.g., `web_search`,
//! `x_search`), it automatically uses the Responses API (`/v1/responses`) for agentic tool calling.
//! Otherwise, it uses the standard Chat Completions API (`/v1/chat/completions`).
//!
//! It also supports image generation via Grok Imagine API for creating images from prompts.
//!
//! # Example: Chat Completions
//!
//! ```rust,no_run
//! use std::sync::Arc;
//!
//! use cloudllm::client_wrapper::{ClientWrapper, Message, Role};
//! use cloudllm::clients::grok::{GrokClient, Model};
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let key = std::env::var("XAI_KEY")?;
//!     let client = GrokClient::new_with_model_enum(&key, Model::Grok3Mini);
//!     let reply = client
//!         .send_message(
//!             &[Message {
//!                 role: Role::User,
//!                 content: Arc::<str>::from("Give me a witty coding tip."),
//!             }],
//!             None,
//!         )
//!         .await?;
//!     println!("{}", reply.content);
//!     Ok(())
//! }
//! ```
//!
//! # Example: Image Generation
//!
//! ```rust,no_run
//! use std::sync::Arc;
//!
//! use cloudllm::clients::grok::GrokClient;
//! use cloudllm::image_generation::{ImageGenerationClient, ImageGenerationOptions};
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let key = std::env::var("XAI_KEY")?;
//!     let client = GrokClient::new_with_model_str(&key, "grok-3-mini");
//!
//!     let response = client.generate_image(
//!         "A surreal landscape with floating islands",
//!         ImageGenerationOptions {
//!             aspect_ratio: None,
//!             num_images: Some(1),
//!             response_format: Some("url".to_string()),
//!         },
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

use crate::client_wrapper::{Role, TokenUsage};
use crate::clients::common::{get_shared_http_client, send_and_track, send_and_track_responses};
use crate::cloudllm::image_generation::{
    ImageData, ImageGenerationClient, ImageGenerationOptions, ImageGenerationResponse,
};
use crate::{ClientWrapper, Message};
use async_trait::async_trait;
use openai_rust2 as openai_rust;
use openai_rust2::chat;
use std::error::Error;
use std::sync::Arc;
use tokio::sync::Mutex;

/// Image generation model identifiers for Grok.
pub enum ImageModel {
    /// `grok-2-image` – Grok's image generation model
    Grok2Image,
}

/// Convert a [`ImageModel`] variant into the string identifier expected by the API.
fn image_model_to_string(model: ImageModel) -> String {
    match model {
        ImageModel::Grok2Image => "grok-2-image".to_string(),
    }
}

/// Client wrapper for xAI's Grok models with Responses API support.
pub struct GrokClient {
    /// Underlying SDK client pointing at the xAI REST endpoint.
    client: openai_rust::Client,
    /// Selected Grok model name.
    model: String,
    /// Storage for the token usage returned by the most recent request.
    token_usage: Mutex<Option<TokenUsage>>,
}

/// Grok model identifiers available as of April 2025.
pub enum Model {
    /// `grok-2` – production Grok 2 multi-modal model.
    Grok2,
    /// `grok-2-latest` – most recent Grok 2 drop.
    Grok2Latest,
    /// `grok-2-1212` – Grok 2 tuned for low latency, priced at $2/MMT input.
    Grok21212,
    /// `grok-3-mini-fast` – quick reasoning Grok 3 mini tier.
    Grok3MiniFast,
    /// `grok-3-mini` – economical Grok 3 mini.
    Grok3Mini,
    /// `grok-3-fast` – high throughput Grok 3.
    Grok3Fast,
    /// `grok-3` – general Grok 3 release.
    Grok3,
    /// `grok-4-0709` – midsummer 2024 Grok 4 release.
    Grok4_0709,
    /// `grok-4-fast-reasoning` – reasoning tuned fast Grok 4.
    Grok4FastReasoning,
    /// `grok-4-fast-nonreasoning` – non-reasoning Grok 4 fast tier.
    Grok4FastNonReasoning,
    /// `grok-code-fast-1` – code-focused Grok fast tier.
    GrokCodeFast1,
    /// `grok-4-1-fast-reasoning` - frontier multimodal model with reasoning, supports server_tools
    Grok41FastReasoning,
    /// `grok-4-1-fast-non-reasoning` - frontier multimodal model without reasoning, supports server_tools
    Grok41FastNonReasoning,
}

/// Convert a [`Model`] variant into the identifier expected by the xAI API.
fn model_to_string(model: Model) -> String {
    match model {
        Model::Grok2 => "grok-2".to_string(),
        Model::Grok2Latest => "grok-2-latest".to_string(),
        Model::Grok21212 => "grok-2-1212".to_string(),
        Model::Grok3MiniFast => "grok-3-mini-fast".to_string(),
        Model::Grok3Mini => "grok-3-mini".to_string(), // cheapest model
        Model::Grok3Fast => "grok-3-fast".to_string(),
        Model::Grok3 => "grok-3".to_string(),
        Model::Grok4_0709 => "grok-4-0709".to_string(),
        Model::Grok4FastReasoning => "grok-4-fast-reasoning".to_string(),
        Model::Grok4FastNonReasoning => "grok-4-fast-nonreasoning".to_string(),
        Model::GrokCodeFast1 => "grok-code-fast-1".to_string(),
        Model::Grok41FastReasoning => "grok-4-1-fast-reasoning".to_string(),
        Model::Grok41FastNonReasoning => "grok-4-1-fast-non-reasoning".to_string(),
    }
}

impl GrokClient {
    /// Construct a client from an API key and typed model variant.
    pub fn new_with_model_enum(secret_key: &str, model: Model) -> Self {
        Self::new_with_model_str(secret_key, &model_to_string(model))
    }

    /// Construct a client from an API key and explicit model name.
    pub fn new_with_model_str(secret_key: &str, model_name: &str) -> Self {
        Self::new_with_base_url(secret_key, model_name, "https://api.x.ai/v1")
    }

    /// Construct a client for Grok-compatible endpoints hosted at a custom base URL.
    pub fn new_with_base_url(secret_key: &str, model_name: &str, base_url: &str) -> Self {
        GrokClient {
            client: openai_rust::Client::new_with_client_and_base_url(
                secret_key,
                get_shared_http_client().clone(),
                base_url,
            ),
            model: model_name.to_string(),
            token_usage: Mutex::new(None),
        }
    }

    /// Convenience wrapper around [`GrokClient::new_with_base_url`].
    pub fn new_with_base_url_and_model_enum(
        secret_key: &str,
        model: Model,
        base_url: &str,
    ) -> Self {
        Self::new_with_base_url(secret_key, &model_to_string(model), base_url)
    }
}

#[async_trait]
impl ClientWrapper for GrokClient {
    fn model_name(&self) -> &str {
        &self.model
    }

    async fn send_message(
        &self,
        messages: &[Message],
        optional_grok_tools: Option<Vec<openai_rust::chat::GrokTool>>,
        _optional_openai_tools: Option<Vec<openai_rust::chat::OpenAITool>>,
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

        // Use the Responses API when tools are provided, otherwise use Chat Completions
        let result = if let Some(grok_tools) = optional_grok_tools {
            // Use the Responses API (/v1/responses) for agentic tool calling
            send_and_track_responses(
                &self.client,
                &self.model,
                formatted_messages,
                Some("/v1/responses".to_string()),
                &self.token_usage,
                grok_tools,
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
                None,
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
                    log::error!("GrokClient::send_message(...): API Error: {}", e);
                }
                Err(e)
            }
        }
    }

    fn send_message_stream<'a>(
        &'a self,
        _messages: &'a [Message],
        _optional_grok_tools: Option<Vec<openai_rust::chat::GrokTool>>,
        _optional_openai_tools: Option<Vec<openai_rust::chat::OpenAITool>>,
    ) -> crate::client_wrapper::MessageStreamFuture<'a> {
        // Note: Streaming is not yet supported for the Responses API
        // For now, return an error indicating streaming is not available
        Box::pin(
            async move { Err("Streaming is not yet supported for the xAI Responses API".into()) },
        )
    }

    fn usage_slot(&self) -> Option<&Mutex<Option<TokenUsage>>> {
        Some(&self.token_usage)
    }
}

#[async_trait]
impl ImageGenerationClient for GrokClient {
    async fn generate_image(
        &self,
        prompt: &str,
        options: ImageGenerationOptions,
    ) -> Result<ImageGenerationResponse, Box<dyn Error + Send + Sync>> {
        // Note: Grok Imagine doesn't support aspect_ratio parameter
        // aspect_ratio is ignored here
        if options.aspect_ratio.is_some() && log::log_enabled!(log::Level::Warn) {
            log::warn!("Grok Imagine does not support aspect_ratio, it will be ignored");
        }

        let n = options.num_images.unwrap_or(1).min(10); // Grok max is 10

        // Create ImageArguments for the API call
        let mut args = openai_rust::images::ImageArguments::new(prompt);
        let model_name = image_model_to_string(ImageModel::Grok2Image);
        args.model = Some(model_name);
        args.n = Some(n);

        // Make the request to the Grok Imagine endpoint
        let response_strings = self
            .client
            .create_image_old(args, Some("/v1/images/generations".to_string()))
            .await?;

        // Convert response strings to ImageData
        let mut images = Vec::new();
        for response_str in response_strings {
            let image_data = if response_str.starts_with("http") {
                // It's a URL
                ImageData {
                    url: Some(response_str),
                    b64_json: None,
                }
            } else {
                // It's likely base64 data
                ImageData {
                    url: None,
                    b64_json: Some(response_str),
                }
            };
            images.push(image_data);
        }

        Ok(ImageGenerationResponse {
            images,
            revised_prompt: None, // TODO: Extract from response if Grok provides it
        })
    }

    fn model_name(&self) -> &str {
        "grok-2-image"
    }
}
