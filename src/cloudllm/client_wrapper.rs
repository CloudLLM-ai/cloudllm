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
//!                 tool_calls: vec![],
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
//!         tool_calls: vec![],
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
use std::error::Error;
use std::pin::Pin;
use std::sync::Arc;
use tokio::sync::Mutex;

/// A single tool call returned by the LLM in a native function-calling response.
///
/// Providers assign an opaque [`id`](NativeToolCall::id) to each call so that the
/// tool result can be correlated back in a follow-up `Role::Tool` message.
///
/// # Example
///
/// ```rust
/// use cloudllm::client_wrapper::NativeToolCall;
///
/// let tc = NativeToolCall {
///     id: "call_abc123".to_string(),
///     name: "calculator".to_string(),
///     arguments: serde_json::json!({"expression": "2 + 2"}),
/// };
/// assert_eq!(tc.name, "calculator");
/// ```
#[derive(Debug, Clone)]
pub struct NativeToolCall {
    /// Provider-assigned call ID, e.g. `"call_abc123"`.
    pub id: String,
    /// Tool name matching one of the [`ToolDefinition`]s sent in the request.
    pub name: String,
    /// Parsed JSON arguments supplied by the LLM for this call.
    pub arguments: serde_json::Value,
}

/// Provider-agnostic tool schema passed to the LLM along with a chat request.
///
/// Derived from [`ToolMetadata`](crate::tool_protocol::ToolMetadata) via
/// [`ToolMetadata::to_tool_definition`](crate::tool_protocol::ToolMetadata::to_tool_definition).
/// Serialised as an OpenAI-compatible `tools` array entry before transmission.
///
/// # Example
///
/// ```rust
/// use cloudllm::client_wrapper::ToolDefinition;
///
/// let def = ToolDefinition {
///     name: "calculator".to_string(),
///     description: "Evaluates a mathematical expression.".to_string(),
///     parameters_schema: serde_json::json!({
///         "type": "object",
///         "properties": {
///             "expression": {"type": "string", "description": "The expression to evaluate"}
///         },
///         "required": ["expression"]
///     }),
/// };
/// assert_eq!(def.name, "calculator");
/// ```
#[derive(Debug, Clone)]
pub struct ToolDefinition {
    /// Tool name as it will appear in the API `tools` array.
    pub name: String,
    /// Human-readable description surfaced to the LLM to aid tool selection.
    pub description: String,
    /// JSON Schema object describing the accepted parameters.
    pub parameters_schema: serde_json::Value,
}

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
    /// A tool-result message correlating with a prior assistant [`NativeToolCall`].
    ///
    /// Serialises as `{"role": "tool", "tool_call_id": "<call_id>", "content": "..."}` in the
    /// OpenAI wire format.
    ///
    /// # Example
    ///
    /// ```rust
    /// use cloudllm::client_wrapper::Role;
    ///
    /// let role = Role::Tool { call_id: "call_abc123".to_string() };
    /// // When serialised by send_with_native_tools this becomes role="tool"
    /// ```
    Tool { call_id: String },
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
///
/// The `tool_calls` field is populated by [`ClientWrapper::send_message`] when the provider
/// returns native function-calling results.  It defaults to an empty `Vec` for all other
/// message kinds.
#[derive(Clone)]
pub struct Message {
    /// The role associated with the message.
    pub role: Role,
    /// The message body.  Stored as `Arc<str>` so that histories can be cheaply cloned by
    /// [`crate::LLMSession`] and downstream components.
    pub content: Arc<str>,
    /// Native tool calls requested by the assistant.  Non-empty only on assistant messages
    /// returned by [`ClientWrapper::send_message`] when the provider responds with
    /// function-calling results.
    pub tool_calls: Vec<NativeToolCall>,
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
    /// The `tools` parameter carries native [`ToolDefinition`]s that are forwarded to the
    /// provider's function-calling API.  When `Some` and non-empty, implementations route to
    /// [`send_with_native_tools`](crate::clients::common::send_with_native_tools); when `None`
    /// or empty, they fall through to the standard Chat Completions path.
    ///
    /// On success the returned [`Message`] may contain non-empty [`Message::tool_calls`] when
    /// the provider selected one or more tools.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use std::sync::Arc;
    /// use cloudllm::client_wrapper::{ClientWrapper, Message, Role};
    /// use cloudllm::clients::openai::{Model, OpenAIClient};
    ///
    /// # #[tokio::main]
    /// # async fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let client = OpenAIClient::new_with_model_enum(
    ///     &std::env::var("OPEN_AI_SECRET")?,
    ///     Model::GPT41Nano,
    /// );
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
    ) -> Result<Message, Box<dyn Error>>;

    /// Request a streaming response from the provider.
    ///
    /// Implementors that sit in front of providers without streaming support can inherit the
    /// default implementation which simply resolves to `Ok(None)`.  A `Some(MessageChunkStream)`
    /// return value must yield [`MessageChunk`] instances that mirror the incremental tokens
    /// supplied by the upstream service.
    ///
    /// The `tools` parameter mirrors [`send_message`](ClientWrapper::send_message); streaming
    /// with native tool calling is out of scope and implementors may ignore it (returning
    /// `Ok(None)` is acceptable).
    ///
    /// Returning a boxed future avoids imposing `Send` bounds on the internal async machinery,
    /// which lets implementations use provider SDKs that are not `Send` internally.
    fn send_message_stream<'a>(
        &'a self,
        _messages: &'a [Message],
        _tools: Option<Vec<ToolDefinition>>,
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
