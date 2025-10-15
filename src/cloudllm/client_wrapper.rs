//! Shared primitives for provider-agnostic LLM clients.
//!
//! Applications typically interact with CloudLLM through the [`ClientWrapper`] trait and the
//! lightweight data types defined in this module.  The trait abstracts over concrete vendor
//! implementations while the supporting structs describe chat messages, streaming chunks, and
//! token accounting.
//!
//! # Basic request/response
//!
//! ```rust,no_run
//! use std::sync::Arc;
//!
//! use cloudllm::client_wrapper::{ClientWrapper, Message, Role};
//! use cloudllm::clients::openai::{Model, OpenAIClient};
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let key = std::env::var("OPEN_AI_SECRET")?;
//!     let client = OpenAIClient::new_with_model_enum(&key, Model::GPT41Nano);
//!
//!     let response = client
//!         .send_message(
//!             &[Message {
//!                 role: Role::User,
//!                 content: Arc::from("Who are you?"),
//!             }],
//!             None,
//!         )
//!         .await?;
//!
//!     println!("Assistant: {}", response.content);
//!     Ok(())
//! }
//! ```
//!
//! # Streaming quick start
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
//!         content: Arc::from("Explain Rust lifetimes in a sentence."),
//!     }];
//!
//!     if let Some(mut chunks) = client.send_message_stream(&request, None).await? {
//!         while let Some(chunk) = chunks.next().await {
//!             print!("{}", chunk?.content);
//!         }
//!     }
//!     Ok(())
//! }
//! ```

use async_trait::async_trait;
use futures_util::stream::Stream;
use openai_rust2 as openai_rust;
use std::error::Error;
use std::pin::Pin;
use std::sync::Arc;
use tokio::sync::Mutex;

/// Trait-driven abstraction for a concrete cloud provider.
///
/// A [`ClientWrapper`] instance is responsible for translating CloudLLM requests into the
/// provider specific wire format and for returning provider responses in a uniform shape.  The
/// abstraction deliberately excludes any conversation bookkeeping: for that functionality see
/// [`crate::LLMSession`].
///
/// All implementations **must** be thread-safe (`Send + Sync`) so they can be shared between
/// async tasks.  Where a provider exposes token accounting information, wrappers should capture
/// it and make it visible via [`ClientWrapper::get_last_usage`].
/// Represents the possible roles for a message.
#[derive(Debug, Clone)]
pub enum Role {
    /// A system authored message that primes or constrains assistant behaviour.
    System,
    /// A user authored message (frequently mirror of a human end-user request).
    User,
    /// An assistant authored message (model responses or developer supplied exemplars).
    Assistant,
}

/// How many tokens were spent on prompt vs. completion?
#[derive(Clone, Debug)]
pub struct TokenUsage {
    /// Number of prompt/input tokens billed by the provider.
    pub input_tokens: usize,
    /// Number of generated/output tokens billed by the provider.
    pub output_tokens: usize,
    /// Convenience total equal to `input_tokens + output_tokens`.
    pub total_tokens: usize,
}

/// Represents a generic message to be sent to an LLM.
#[derive(Clone)]
pub struct Message {
    /// The role associated with the message.
    pub role: Role,
    /// The message body.  Stored as `Arc<str>` so that histories can be cheaply cloned by
    /// [`crate::LLMSession`] and downstream components.
    pub content: Arc<str>,
}

/// Represents a chunk of content in a streaming response.
/// Each chunk contains a delta (incremental piece) of the assistant's response.
#[derive(Clone, Debug)]
pub struct MessageChunk {
    /// The incremental content delta in this chunk.
    /// May be empty for chunks that don't contain content (e.g., finish_reason chunks).
    pub content: String,
    /// Optional finish reason mirroring the provider specific completion status (e.g. `"stop"`).
    pub finish_reason: Option<String>,
}

/// Type alias for a stream of message chunks compatible with `Send` executors.
pub type MessageChunkStream =
    Pin<Box<dyn Stream<Item = Result<MessageChunk, Box<dyn Error>>> + Send>>;

/// Type alias for the future returned by [`ClientWrapper::send_message_stream`].
pub type MessageStreamFuture<'a> = Pin<
    Box<dyn std::future::Future<Output = Result<Option<MessageChunkStream>, Box<dyn Error>>> + 'a>,
>;

/// Trait defining the interface to interact with various LLM services.
#[async_trait]
pub trait ClientWrapper: Send + Sync {
    /// Send a full request/response style chat completion.
    ///
    /// The `messages` slice must include any system priming messages the caller wishes to send.
    /// Providers that support additional retrieval or search arguments can inspect
    /// `optional_search_parameters`.  Implementations should surface provider specific errors via
    /// `Err(Box<dyn Error>)`.
    async fn send_message(
        &self,
        messages: &[Message],
        optional_search_parameters: Option<openai_rust::chat::SearchParameters>,
    ) -> Result<Message, Box<dyn Error>>;

    /// Request a streaming response from the provider.
    ///
    /// Implementors that sit in front of providers without streaming support can inherit the
    /// default implementation which simply resolves to `Ok(None)`.  A `Some(MessageChunkStream)`
    /// return value must yield [`MessageChunk`] instances that mirror the incremental tokens
    /// supplied by the upstream service.
    ///
    /// Returning a boxed future avoids imposing `Send` bounds on the internal async machinery,
    /// which lets implementations use provider SDKs that are not `Send` internally.
    fn send_message_stream<'a>(
        &'a self,
        _messages: &'a [Message],
        _optional_search_parameters: Option<openai_rust::chat::SearchParameters>,
    ) -> MessageStreamFuture<'a> {
        Box::pin(async { Ok(None) })
    }

    /// Return the identifier used to select the upstream model (e.g. `"gpt-4.1"`).
    fn model_name(&self) -> &str;

    /// Hook to retrieve usage from the most recent [`ClientWrapper::send_message`] call.
    ///
    /// Wrappers that propagate token accounting should override [`ClientWrapper::usage_slot`].
    async fn get_last_usage(&self) -> Option<TokenUsage> {
        if let Some(slot) = self.usage_slot() {
            slot.lock().await.clone()
        } else {
            None
        }
    }

    /// Expose a shared mutable slot where the implementation can persist token usage.
    ///
    /// By default wrappers report no usage data.  Providers that expose billing information
    /// should return `Some(&Mutex<Option<TokenUsage>>)` so that [`ClientWrapper::get_last_usage`]
    /// can surface the recorded values to callers.
    fn usage_slot(&self) -> Option<&Mutex<Option<TokenUsage>>> {
        None
    }
}
