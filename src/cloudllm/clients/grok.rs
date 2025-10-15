//! xAI Grok client wrapper routed through the OpenAI-compatible surface.
//!
//! The `GrokClient` connects to xAI's Grok models using the same transport as the OpenAI client.
//! It is therefore straightforward to reuse existing session or council code while targeting the
//! Grok family of models.
//!
//! # Example
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

use crate::client_wrapper::TokenUsage;
use crate::clients::openai::OpenAIClient;
use crate::{ClientWrapper, Message};
use async_trait::async_trait;
use openai_rust2 as openai_rust;
use std::error::Error;
use tokio::sync::Mutex;

/// Client wrapper for xAI's Grok models accessed via the OpenAI-style API surface.
pub struct GrokClient {
    /// Delegated OpenAI-compatible client.
    delegate_client: OpenAIClient,
    /// Selected Grok model name.
    model: String,
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
    }
}

impl GrokClient {
    /// Construct a client from an API key and typed model variant.
    pub fn new_with_model_enum(secret_key: &str, model: Model) -> Self {
        Self::new_with_model_str(secret_key, &model_to_string(model))
    }

    /// Construct a client from an API key and explicit model name.
    pub fn new_with_model_str(secret_key: &str, model_name: &str) -> Self {
        GrokClient {
            // we reuse the OpenAIClient for Grok and delegate the calls to it
            delegate_client: OpenAIClient::new_with_base_url(
                secret_key,
                model_name,
                "https://api.x.ai/v1",
            ),
            model: model_name.to_string(),
        }
    }

    /// Construct a client for Grok-compatible endpoints hosted at a custom base URL.
    pub fn new_with_base_url(secret_key: &str, model_name: &str, base_url: &str) -> Self {
        GrokClient {
            delegate_client: OpenAIClient::new_with_base_url(secret_key, model_name, base_url),
            model: model_name.to_string(),
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
        optional_search_parameters: Option<openai_rust::chat::SearchParameters>,
    ) -> Result<Message, Box<dyn Error>> {
        self.delegate_client
            .send_message(messages, optional_search_parameters)
            .await
    }

    fn send_message_stream<'a>(
        &'a self,
        messages: &'a [Message],
        optional_search_parameters: Option<openai_rust::chat::SearchParameters>,
    ) -> crate::client_wrapper::MessageStreamFuture<'a> {
        self.delegate_client
            .send_message_stream(messages, optional_search_parameters)
    }

    fn usage_slot(&self) -> Option<&Mutex<Option<TokenUsage>>> {
        self.delegate_client.usage_slot()
    }
}
