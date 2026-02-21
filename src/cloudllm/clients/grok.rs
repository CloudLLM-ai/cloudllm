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

use crate::client_wrapper::{Role, TokenUsage, ToolDefinition};
use crate::clients::common::{get_shared_http_client, send_and_track, send_with_native_tools};
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
    /// `grok-2-image-1212` – legacy Grok image generation model
    Grok2Image,
    /// `grok-imagine-image` – current Grok Imagine image generation model
    GrokImagineImage,
}

/// Convert a [`ImageModel`] variant into the string identifier expected by the API.
fn image_model_to_string(model: ImageModel) -> String {
    match model {
        ImageModel::Grok2Image => "grok-2-image-1212".to_string(),
        ImageModel::GrokImagineImage => "grok-imagine-image".to_string(),
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
    /// API key needed for image generation
    api_key: String,
    /// Base URL for API calls
    base_url: String,
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
    /// Note: base_url should not have a trailing slash (e.g., "https://api.x.ai/v1")
    pub fn new_with_base_url(secret_key: &str, model_name: &str, base_url: &str) -> Self {
        let base_url_normalized = base_url.trim_end_matches('/');
        GrokClient {
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

    /// Send a chat completion, routing to native tool calling when `tools` is non-empty.
    ///
    /// When `tools` is `Some` and non-empty the request is forwarded to
    /// [`send_with_native_tools`](crate::clients::common::send_with_native_tools).
    /// Otherwise the standard Chat Completions endpoint is used.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use std::sync::Arc;
    /// use cloudllm::client_wrapper::{ClientWrapper, Message, Role};
    /// use cloudllm::clients::grok::{GrokClient, Model};
    ///
    /// # #[tokio::main]
    /// # async fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let client = GrokClient::new_with_model_enum(&std::env::var("XAI_KEY")?, Model::Grok3Mini);
    /// let resp = client.send_message(
    ///     &[Message { role: Role::User, content: Arc::from("Hello"), tool_calls: vec![] }],
    ///     None,
    /// ).await?;
    /// println!("{}", resp.content);
    /// # Ok(())
    /// # }
    /// ```
    async fn send_message(
        &self,
        messages: &[Message],
        tools: Option<Vec<ToolDefinition>>,
    ) -> Result<Message, Box<dyn Error>> {
        // Route to native tool calling when tools are provided
        if let Some(tool_defs) = tools.filter(|t| !t.is_empty()) {
            return send_with_native_tools(
                &self.base_url,
                &self.api_key,
                &self.model,
                messages,
                &tool_defs,
                get_shared_http_client(),
                &self.token_usage,
            )
            .await
            .map_err(|e| {
                if log::log_enabled!(log::Level::Error) {
                    log::error!("GrokClient::send_message (native tools): {}", e);
                }
                e
            });
        }

        // Standard Chat Completions path (no tools)
        let mut formatted_messages = Vec::with_capacity(messages.len());
        for msg in messages {
            formatted_messages.push(chat::Message {
                role: match &msg.role {
                    Role::System => "system".to_owned(),
                    Role::User => "user".to_owned(),
                    Role::Assistant => "assistant".to_owned(),
                    Role::Tool { .. } => "tool".to_owned(),
                },
                content: msg.content.to_string(),
            });
        }

        let result = send_and_track(
            &self.client,
            &self.model,
            formatted_messages,
            Some("/v1/chat/completions".to_string()),
            &self.token_usage,
            None,
        )
        .await;

        match result {
            Ok(c) => Ok(Message {
                role: Role::Assistant,
                content: Arc::from(c.as_str()),
                tool_calls: vec![],
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
        _tools: Option<Vec<ToolDefinition>>,
    ) -> crate::client_wrapper::MessageStreamFuture<'a> {
        Box::pin(async move {
            Err("Streaming is not yet supported for GrokClient".into())
        })
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
        let n = options.num_images.unwrap_or(1).min(10); // Grok max is 10

        // Build request body with all supported parameters
        let mut request_body = serde_json::json!({
            "model": image_model_to_string(ImageModel::GrokImagineImage),
            "prompt": prompt,
            "n": n,
        });

        // Add optional parameters if provided
        if let Some(ref ar) = options.aspect_ratio {
            request_body["aspect_ratio"] = serde_json::json!(ar);
        }
        if let Some(ref fmt) = options.response_format {
            request_body["response_format"] = serde_json::json!(fmt);
        }

        // Make direct HTTP request to Grok's image generation endpoint
        let http_client = get_shared_http_client();
        let url = format!("{}/images/generations", self.base_url);

        log::info!(
            "Grok Imagine API request: model={}, prompt_len={}, n={}",
            request_body["model"],
            prompt.len(),
            n
        );

        let response = http_client
            .post(&url)
            .header("Authorization", format!("Bearer {}", &self.api_key))
            .json(&request_body)
            .send()
            .await?;

        let status = response.status();
        let response_text = response.text().await?;

        if log::log_enabled!(log::Level::Debug) {
            log::debug!(
                "Grok image API response (HTTP {}): {}",
                status,
                response_text
            );
        }

        // Parse JSON response
        let response_json: serde_json::Value = serde_json::from_str(&response_text)?;

        // Check for API errors (non-2xx status or explicit error field)
        if !status.is_success() || response_json.get("error").is_some() {
            log::error!(
                "Grok Imagine API error (HTTP {}): {}",
                status,
                response_text
            );
            if let Some(error) = response_json.get("error") {
                return Err(
                    format!("Grok Imagine API error (HTTP {}): {}", status, error).into(),
                );
            }
            return Err(
                format!("Grok Imagine API error (HTTP {}): {}", status, response_text).into(),
            );
        }

        // Parse the response - Grok returns: { "data": [{ "url": "..." }, ...] }
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
            return Err("No images in Grok response".into());
        }

        Ok(ImageGenerationResponse {
            images,
            revised_prompt: response_json
                .get("revised_prompt")
                .and_then(|r| r.as_str())
                .map(|s| s.to_string()),
        })
    }

    fn model_name(&self) -> &str {
        "grok-imagine-image"
    }
}
