//! Google Gemini client wrapper exposing the `ClientWrapper` trait.
//!
//! The `GeminiClient` connects to Google's Generative Language (Gemini) API using the
//! same message structures and token accounting abstractions employed by the rest of CloudLLM.
//!
//! It also supports image generation via Gemini's image generation models with
//! configurable aspect ratios and quality tiers.
//!
//! # Selecting a model and sending a message
//!
//! ```rust,no_run
//! use std::sync::Arc;
//!
//! use cloudllm::client_wrapper::{ClientWrapper, Message, Role};
//! use cloudllm::clients::gemini::{GeminiClient, Model};
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let key = std::env::var("GEMINI_KEY")?;
//!     let client = GeminiClient::new_with_model_enum(&key, Model::Gemini20Flash);
//!     let reply = client
//!         .send_message(
//!             &[Message {
//!                 role: Role::User,
//!                 content: Arc::<str>::from("What industries benefit most from Gemini?"),
//!             }],
//!             None,
//!         )
//!         .await?;
//!     println!("{}", reply.content);
//!     Ok(())
//! }
//! ```
//!
//! # Image Generation with Gemini
//!
//! ```rust,no_run
//! use std::sync::Arc;
//!
//! use cloudllm::clients::gemini::GeminiClient;
//! use cloudllm::image_generation::{ImageGenerationClient, ImageGenerationOptions};
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let key = std::env::var("GEMINI_KEY")?;
//!     let client = GeminiClient::new_with_model_string(&key, "gemini-2.5-flash-image");
//!
//!     let response = client.generate_image(
//!         "A serene mountain landscape at sunrise",
//!         ImageGenerationOptions {
//!             aspect_ratio: Some("16:9".to_string()),
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

use crate::client_wrapper::TokenUsage;
use crate::clients::common::{get_shared_http_client, send_and_track};
use crate::cloudllm::image_generation::{
    ImageData, ImageGenerationClient, ImageGenerationOptions, ImageGenerationResponse,
};
use crate::{ClientWrapper, Message, Role};
use async_trait::async_trait;
use log::error;
use openai_rust::chat;
use openai_rust2 as openai_rust;
use serde_json::json;
use std::sync::Arc;
use tokio::sync::Mutex;

/// Image generation model identifiers for Gemini.
pub enum ImageModel {
    /// `gemini-2.5-flash-image` – Fast, efficient Gemini image generation
    Gemini25FlashImage,
}

/// Convert a [`ImageModel`] variant into the string identifier expected by the API.
fn image_model_to_string(model: ImageModel) -> String {
    match model {
        ImageModel::Gemini25FlashImage => "gemini-2.5-flash-image".to_string(),
    }
}

/// Client wrapper for Google Gemini (Generative Language) chat-style endpoints.
pub struct GeminiClient {
    /// Underlying OpenAI compatible client pointed at the Gemini base URL.
    client: openai_rust::Client,
    /// Model identifier used for subsequent requests.
    pub model: String,
    /// Storage for the most recent token usage report.
    token_usage: Mutex<Option<TokenUsage>>,
    /// API key needed for image generation (Gemini uses query parameters instead of bearer token)
    api_key: String,
    /// Base URL for API calls
    base_url: String,
}

/// Gemini model identifiers returned by the public API (February 2025 snapshot).
///
/// Every variant maps 1:1 to the hyphenated model name that the API expects.  Use
/// [`model_to_string`] when you need the string literal.
pub enum Model {
    Gemini20Flash,
    Gemini20FlashExp,
    Gemini20Flash001,
    Gemini20FlashLite001,
    Gemini20FlashThinking001,
    Gemini20FlashLitePreview,
    Gemini20FlashLitePreview0205,
    Gemini20ProExp,
    Gemini20ProExp0205,
    ChatBison001,
    TextBison001,
    TextBisonSafetyRecitationOff,
    TextBisonSafetyOff,
    TextBisonRecitationOff,
    EmbeddingGecko001,
    EvergreenCustom,
    Gemini10ProLatest,
    Gemini10Pro,
    GeminiPro,
    Gemini10Pro001,
    Gemini10ProVisionLatest,
    GeminiProVision,
    Gemini15ProLatest,
    Gemini15Pro001,
    Gemini15Pro002,
    Gemini15Pro,
    Gemini15FlashLatest,
    Gemini15Flash001,
    Gemini15Flash001Tuning,
    Gemini15Flash,
    Gemini15Flash002,
    Gemini15FlashDarkLaunch,
    GeminiTest23,
    Gemini15Flash8b,
    Gemini15Flash8b001,
    Gemini15Flash8bLatest,
    Gemini15Flash8bExp0827,
    Gemini15Flash8bExp0924,
    GeminiExp1206,
    GeminiToolTest,
    Gemini20FlashThinkingExp0121,
    Gemini20FlashThinkingExp,
    Gemini20FlashThinkingExp1219,
    Gemini20FlashThinkingExpNoThoughts,
    Gemini20Flash001BidiTest,
    Gemini20FlashAudiogenRev17,
    Gemini20FlashMmgenRev17,
    Gemini20FlashJarvis,
    Learnlm15ProExperimental,
    Embedding001,
    TextEmbedding004,
    Aqa,
    Imagen30Generate002,
    Imagen30Generate002Exp,
    ImageVerification001,
    Veo20Generate001,
    Gemini25Flash,
    Gemini25Pro,
    Gemini25FlashLitePreview0617,
}

/// Convert a strongly typed [`Model`] into the string literal expected by the Gemini endpoint.
pub fn model_to_string(model: Model) -> String {
    match model {
        Model::Gemini20Flash => "gemini-2.0-flash".to_string(),
        Model::Gemini20FlashExp => "gemini-2.0-flash-exp".to_string(),
        Model::Gemini20Flash001 => "gemini-2.0-flash-001".to_string(),
        Model::Gemini20FlashLite001 => "gemini-2.0-flash-lite-001".to_string(),
        Model::Gemini20FlashThinking001 => "gemini-2.0-flash-thinking-001".to_string(),
        Model::Gemini20FlashLitePreview => "gemini-2.0-flash-lite-preview".to_string(),
        Model::Gemini20FlashLitePreview0205 => "gemini-2.0-flash-lite-preview-02-05".to_string(),
        Model::Gemini20ProExp => "gemini-2.0-pro-exp".to_string(),
        Model::Gemini20ProExp0205 => "gemini-2.0-pro-exp-02-05".to_string(),
        Model::ChatBison001 => "chat-bison-001".to_string(),
        Model::TextBison001 => "text-bison-001".to_string(),
        Model::TextBisonSafetyRecitationOff => "text-bison-safety-recitation-off".to_string(),
        Model::TextBisonSafetyOff => "text-bison-safety-off".to_string(),
        Model::TextBisonRecitationOff => "text-bison-recitation-off".to_string(),
        Model::EmbeddingGecko001 => "embedding-gecko-001".to_string(),
        Model::EvergreenCustom => "evergreen-custom".to_string(),
        Model::Gemini10ProLatest => "gemini-1.0-pro-latest".to_string(),
        Model::Gemini10Pro => "gemini-1.0-pro".to_string(),
        Model::GeminiPro => "gemini-pro".to_string(),
        Model::Gemini10Pro001 => "gemini-1.0-pro-001".to_string(),
        Model::Gemini10ProVisionLatest => "gemini-1.0-pro-vision-latest".to_string(),
        Model::GeminiProVision => "gemini-pro-vision".to_string(),
        Model::Gemini15ProLatest => "gemini-1.5-pro-latest".to_string(),
        Model::Gemini15Pro001 => "gemini-1.5-pro-001".to_string(),
        Model::Gemini15Pro002 => "gemini-1.5-pro-002".to_string(),
        Model::Gemini15Pro => "gemini-1.5-pro".to_string(),
        Model::Gemini15FlashLatest => "gemini-1.5-flash-latest".to_string(),
        Model::Gemini15Flash001 => "gemini-1.5-flash-001".to_string(),
        Model::Gemini15Flash001Tuning => "gemini-1.5-flash-001-tuning".to_string(),
        Model::Gemini15Flash => "gemini-1.5-flash".to_string(),
        Model::Gemini15Flash002 => "gemini-1.5-flash-002".to_string(),
        Model::Gemini15FlashDarkLaunch => "gemini-1.5-flash-dark-launch".to_string(),
        Model::GeminiTest23 => "gemini-test-23".to_string(),
        Model::Gemini15Flash8b => "gemini-1.5-flash-8b".to_string(),
        Model::Gemini15Flash8b001 => "gemini-1.5-flash-8b-001".to_string(),
        Model::Gemini15Flash8bLatest => "gemini-1.5-flash-8b-latest".to_string(),
        Model::Gemini15Flash8bExp0827 => "gemini-1.5-flash-8b-exp-0827".to_string(),
        Model::Gemini15Flash8bExp0924 => "gemini-1.5-flash-8b-exp-0924".to_string(),
        Model::GeminiExp1206 => "gemini-exp-1206".to_string(),
        Model::GeminiToolTest => "gemini-tool-test".to_string(),
        Model::Gemini20FlashThinkingExp0121 => "gemini-2.0-flash-thinking-exp-01-21".to_string(),
        Model::Gemini20FlashThinkingExp => "gemini-2.0-flash-thinking-exp".to_string(),
        Model::Gemini20FlashThinkingExp1219 => "gemini-2.0-flash-thinking-exp-1219".to_string(),
        Model::Gemini20FlashThinkingExpNoThoughts => {
            "gemini-2.0-flash-thinking-exp-no-thoughts".to_string()
        }
        Model::Gemini20Flash001BidiTest => "gemini-2.0-flash-001-bidi-test".to_string(),
        Model::Gemini20FlashAudiogenRev17 => "gemini-2.0-flash-audiogen-rev17".to_string(),
        Model::Gemini20FlashMmgenRev17 => "gemini-2.0-flash-mmgen-rev17".to_string(),
        Model::Gemini20FlashJarvis => "gemini-2.0-flash-jarvis".to_string(),
        Model::Learnlm15ProExperimental => "learnlm-1.5-pro-experimental".to_string(),
        Model::Embedding001 => "embedding-001".to_string(),
        Model::TextEmbedding004 => "text-embedding-004".to_string(),
        Model::Aqa => "aqa".to_string(),
        Model::Imagen30Generate002 => "imagen-3.0-generate-002".to_string(),
        Model::Imagen30Generate002Exp => "imagen-3.0-generate-002-exp".to_string(),
        Model::ImageVerification001 => "image-verification-001".to_string(),
        Model::Veo20Generate001 => "veo-2.0-generate-001".to_string(),
        Model::Gemini25Flash => "gemini-2.5-flash".to_string(),
        Model::Gemini25Pro => "gemini-2.5-pro".to_string(),
        Model::Gemini25FlashLitePreview0617 => "gemini-2.5-flash-lite-preview-06-17".to_string(),
    }
}

impl GeminiClient {
    /// Construct a client using the default Gemini base URL and an explicit model name.
    pub fn new_with_model_string(secret_key: &str, model_name: &str) -> Self {
        use crate::clients::common::get_shared_http_client;
        let base_url = "https://generativelanguage.googleapis.com/v1beta";
        GeminiClient {
            client: openai_rust::Client::new_with_client_and_base_url(
                secret_key,
                get_shared_http_client().clone(),
                &format!("{}/", base_url),
            ),
            model: model_name.to_string(),
            token_usage: Mutex::new(None),
            api_key: secret_key.to_string(),
            base_url: base_url.to_string(),
        }
    }

    /// Construct a client from an API key and [`Model`] variant.
    pub fn new_with_model_enum(secret_key: &str, model: Model) -> Self {
        Self::new_with_model_string(secret_key, &model_to_string(model))
    }

    /// This function is used to create a GeminiClient with a custom base URL.
    /// Note: base_url should not have a trailing slash (e.g., "https://generativelanguage.googleapis.com/v1beta")
    pub fn new_with_base_url(secret_key: &str, model_name: &str, base_url: &str) -> Self {
        use crate::clients::common::get_shared_http_client;
        let base_url_normalized = base_url.trim_end_matches('/');
        GeminiClient {
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

    /// Variant of [`GeminiClient::new_with_base_url`] that accepts a strongly typed [`Model`].
    pub fn new_with_base_url_and_model_enum(
        secret_key: &str,
        model: Model,
        base_url: &str,
    ) -> Self {
        Self::new_with_base_url(secret_key, &model_to_string(model), base_url)
    }
}

#[async_trait]
impl ClientWrapper for GeminiClient {
    fn model_name(&self) -> &str {
        &self.model
    }

    /// Send a synchronous message to the Gemini endpoint.
    async fn send_message(
        &self,
        messages: &[Message],
        optional_grok_tools: Option<Vec<openai_rust::chat::GrokTool>>,
        _optional_openai_tools: Option<Vec<openai_rust::chat::OpenAITool>>,
    ) -> Result<Message, Box<dyn std::error::Error>> {
        // Convert to openai_rust chat::Message

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

        // Use the shared helper to send & track usage
        let url_path = Some("/v1beta/chat/completions".to_string());
        let result = send_and_track(
            &self.client,
            &self.model,
            formatted_messages,
            url_path,
            &self.token_usage,
            optional_grok_tools,
        )
        .await;

        match result {
            Ok(content) => Ok(Message {
                role: Role::Assistant,
                content: Arc::from(content.as_str()),
            }),
            Err(err) => {
                if log::log_enabled!(log::Level::Error) {
                    error!("GeminiClient::send_message error: {}", err);
                }
                Err(err)
            }
        }
    }

    /// Expose the storage slot used by [`ClientWrapper::get_last_usage`].
    ///
    /// Returning `Some(...)` enables downstream consumers to pull accurate Gemini billing data.
    fn usage_slot(&self) -> Option<&Mutex<Option<TokenUsage>>> {
        Some(&self.token_usage)
    }
}

#[async_trait]
impl ImageGenerationClient for GeminiClient {
    async fn generate_image(
        &self,
        prompt: &str,
        options: ImageGenerationOptions,
    ) -> Result<ImageGenerationResponse, Box<dyn std::error::Error + Send + Sync>> {
        // Use gemini-2.5-flash-image by default (faster model)
        let model_name = image_model_to_string(ImageModel::Gemini25FlashImage);

        // Map aspect ratio to Gemini's supported format (default to 1:1)
        let aspect_ratio = options.aspect_ratio.as_deref().unwrap_or("1:1");

        // Validate aspect ratio
        let valid_ratios = vec![
            "1:1", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4", "9:16", "16:9", "21:9",
        ];
        if !valid_ratios.contains(&aspect_ratio) && log::log_enabled!(log::Level::Warn) {
            log::warn!(
                "Gemini unsupported aspect ratio '{}', using 1:1",
                aspect_ratio
            );
        }

        // Build the Gemini image generation request
        let request_body = json!({
            "contents": [
                {
                    "parts": [
                        {
                            "text": prompt
                        }
                    ]
                }
            ],
            "generationConfig": {
                "responseModalities": ["image"],
                "imageConfig": {
                    "aspectRatio": aspect_ratio
                }
            }
        });

        // Build the URL for image generation - model field in URL path (uses base_url)
        let url = format!(
            "{}/models/{}:generateContent?key={}",
            self.base_url, model_name, self.api_key
        );

        // Make the request
        let http_client = get_shared_http_client();
        let response = http_client.post(&url).json(&request_body).send().await?;

        let response_text = response.text().await?;

        if log::log_enabled!(log::Level::Debug) {
            log::debug!("Gemini image API response: {}", response_text);
        }

        let response_json: serde_json::Value = serde_json::from_str(&response_text)?;

        // Check for API errors
        if let Some(error) = response_json.get("error") {
            if let Some(message) = error.get("message").and_then(|m| m.as_str()) {
                return Err(format!("Gemini API error: {}", message).into());
            }
            return Err("Gemini API returned an error".into());
        }

        // Parse the Gemini response structure
        let mut images = Vec::new();

        // Gemini returns images in candidates[].content.parts[].inlineData
        if let Some(candidates) = response_json.get("candidates").and_then(|c| c.as_array()) {
            for candidate in candidates {
                if let Some(content) = candidate.get("content") {
                    if let Some(parts) = content.get("parts").and_then(|p| p.as_array()) {
                        for part in parts {
                            // Check for image data
                            if let Some(inline_data) = part.get("inlineData") {
                                if let Some(mime_type) =
                                    inline_data.get("mimeType").and_then(|m| m.as_str())
                                {
                                    if mime_type.starts_with("image/") {
                                        if let Some(data) =
                                            inline_data.get("data").and_then(|d| d.as_str())
                                        {
                                            let image_data = ImageData {
                                                url: None,
                                                b64_json: Some(format!(
                                                    "data:{};base64,{}",
                                                    mime_type, data
                                                )),
                                            };
                                            images.push(image_data);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        // Handle response format preference
        // If "url" format was requested but we only have base64, that's what we return
        // Gemini doesn't support direct URL format for generated images in the standard API

        Ok(ImageGenerationResponse {
            images,
            revised_prompt: None,
        })
    }

    fn model_name(&self) -> &str {
        "gemini-2.5-flash-image"
    }
}
