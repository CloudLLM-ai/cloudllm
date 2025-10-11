use async_trait::async_trait;
use futures_util::stream::Stream;
use openai_rust2 as openai_rust;
/// A ClientWrapper is a wrapper around a specific cloud LLM service.
/// It provides a common interface to interact with the LLMs.
/// It does not keep track of the conversation/session, for that we use an LLMSession
/// which keeps track of the conversation history and other session-specific data
/// and uses a ClientWrapper to interact with the LLM.
// src/client_wrapper
use std::error::Error;
use std::pin::Pin;
use std::sync::Arc;
use tokio::sync::Mutex;

/// Represents the possible roles for a message.
#[derive(Clone)]
pub enum Role {
    System,
    // set by the developer to steer the model's responses
    User,
    // a message sent by a human user (or app user)
    Assistant, // lets the model know the content was generated as a response to a user message
               // Add other roles as needed
}

/// How many tokens were spent on prompt vs. completion?
#[derive(Clone, Debug)]
pub struct TokenUsage {
    pub input_tokens: usize,
    pub output_tokens: usize,
    pub total_tokens: usize,
}

/// Represents a generic message to be sent to an LLM.
#[derive(Clone)]
pub struct Message {
    /// The role associated with the message.
    pub role: Role,
    /// The actual content of the message stored as Arc<str> to avoid clones.
    pub content: Arc<str>,
}

/// Represents a chunk of content in a streaming response.
/// Each chunk contains a delta (incremental piece) of the assistant's response.
#[derive(Clone, Debug)]
pub struct MessageChunk {
    /// The incremental content delta in this chunk.
    /// May be empty for chunks that don't contain content (e.g., finish_reason chunks).
    pub content: String,
    /// Optional finish reason indicating why the stream ended (e.g., "stop", "length").
    pub finish_reason: Option<String>,
}

/// Type alias for a stream of message chunks.
pub type MessageChunkStream =
    Pin<Box<dyn Stream<Item = Result<MessageChunk, Box<dyn Error>>> + Send>>;

/// Type alias for the future returned by send_message_stream.
pub type MessageStreamFuture<'a> = Pin<
    Box<dyn std::future::Future<Output = Result<Option<MessageChunkStream>, Box<dyn Error>>> + 'a>,
>;

/// Trait defining the interface to interact with various LLM services.
#[async_trait]
pub trait ClientWrapper: Send + Sync {
    /// Send a message to the LLM and get a response.
    /// - `messages`: The messages to send in the request.
    async fn send_message(
        &self,
        messages: &[Message],
        optional_search_parameters: Option<openai_rust::chat::SearchParameters>,
    ) -> Result<Message, Box<dyn Error>>;

    /// Send a message to the LLM and get a streaming response.
    /// Returns a stream of MessageChunk items that arrive as the LLM generates them.
    /// - `messages`: The messages to send in the request.
    ///
    /// Default implementation returns None, indicating streaming is not supported.
    /// Providers that support streaming should override this method.
    ///
    /// Note: This method returns a boxed future instead of using async fn to avoid
    /// Send requirements on the internal stream processing.
    fn send_message_stream<'a>(
        &'a self,
        _messages: &'a [Message],
        _optional_search_parameters: Option<openai_rust::chat::SearchParameters>,
    ) -> MessageStreamFuture<'a> {
        Box::pin(async { Ok(None) })
    }

    /// Returns the model identifier configured for this client.
    fn model_name(&self) -> &str;

    /// Hook to retrieve usage from the *last* send_message() call.
    /// Default impl returns None, so existing wrappers don’t break.
    async fn get_last_usage(&self) -> Option<TokenUsage> {
        if let Some(slot) = self.usage_slot() {
            slot.lock().await.clone()
        } else {
            None
        }
    }

    fn usage_slot(&self) -> Option<&Mutex<Option<TokenUsage>>> {
        // ClientWrapper implementations supporting TokenUsage tracking should return a Mutex<Option<TokenUsage>> by overriding this method.
        None
    }
}
