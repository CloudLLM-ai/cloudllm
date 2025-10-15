//! Anthropic Claude client wrapper built on the OpenAI-compatible transport.
//!
//! Use this module when you want to call Anthropic's Claude models through the same
//! [`ClientWrapper`] interface used by the rest of the
//! crate.  The wrapper delegates HTTP concerns to the shared OpenAI implementation, so swapping
//! from OpenAI to Claude only requires a different constructor.
//!
//! # Example
//!
//! ```rust,no_run
//! use std::sync::Arc;
//!
//! use cloudllm::client_wrapper::{ClientWrapper, Message, Role};
//! use cloudllm::clients::claude::{ClaudeClient, Model};
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let key = std::env::var("ANTHROPIC_KEY")?;
//!     let client = ClaudeClient::new_with_model_enum(&key, Model::ClaudeSonnet4);
//!     let reply = client
//!         .send_message(
//!             &[Message {
//!                 role: Role::User,
//!                 content: Arc::<str>::from("List three Claude capabilities."),
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

/// Client wrapper for Anthropic's Claude API routed through the OpenAI compatible surface.
pub struct ClaudeClient {
    /// Delegated client that handles the HTTP interactions.
    delegate_client: OpenAIClient,
    /// Exposed model name.
    model: String,
}

/// Anthropic Claude models available through the compatibility layer (Jan 2025 snapshot).
pub enum Model {
    /// `claude-opus-4-1` – flagship reasoning tier.
    ClaudeOpus41,
    /// `claude-opus-4-0` – previous Opus generation.
    ClaudeOpus4,
    /// `claude-sonnet-4-0` – balanced reasoning + throughput.
    ClaudeSonnet4,
    /// `claude-sonnet-3-7-sonnet-latest` – latest Sonnet iteration.
    ClaudeSonnet37,
    /// `claude-haiku-3-5-haiku-latest` – fastest Claude tier.
    ClaudeHaiku35,
}

/// Convert a [`Model`] variant into its public string identifier.
fn model_to_string(model: Model) -> String {
    match model {
        Model::ClaudeOpus41 => "claude-opus-4-1".to_string(),
        Model::ClaudeOpus4 => "claude-opus-4-0".to_string(),
        Model::ClaudeSonnet4 => "claude-sonnet-4-0".to_string(),
        Model::ClaudeSonnet37 => "claude-sonnet-3-7-sonnet-latest".to_string(),
        Model::ClaudeHaiku35 => "claude-haiku-3-5-haiku-latest".to_string(),
    }
}

impl ClaudeClient {
    /// Create a client from an API key and strongly typed model variant.
    pub fn new_with_model_enum(secret_key: &str, model: Model) -> Self {
        Self::new_with_model_str(secret_key, &model_to_string(model))
    }

    /// Create a client from an API key and explicit model string.
    pub fn new_with_model_str(secret_key: &str, model_name: &str) -> Self {
        ClaudeClient {
            // we reuse the OpenAIClient for Claude and delegate the calls to it
            delegate_client: OpenAIClient::new_with_base_url(
                secret_key,
                model_name,
                "https://api.anthropic.com/v1",
            ),
            model: model_name.to_string(),
        }
    }

    /// Create a client pointing at a custom Claude-compatible base URL.
    pub fn new_with_base_url(secret_key: &str, model_name: &str, base_url: &str) -> Self {
        ClaudeClient {
            delegate_client: OpenAIClient::new_with_base_url(secret_key, model_name, base_url),
            model: model_name.to_string(),
        }
    }

    /// Variant of [`ClaudeClient::new_with_base_url`] that accepts a [`Model`] variant.
    pub fn new_with_base_url_and_model_enum(
        secret_key: &str,
        model: Model,
        base_url: &str,
    ) -> Self {
        Self::new_with_base_url(secret_key, &model_to_string(model), base_url)
    }
}

#[async_trait]
impl ClientWrapper for ClaudeClient {
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

    fn usage_slot(&self) -> Option<&Mutex<Option<TokenUsage>>> {
        self.delegate_client.usage_slot()
    }
}
