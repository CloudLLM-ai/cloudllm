use async_trait::async_trait;
use futures_util::Stream;
use openai_rust2 as openai_rust;
use std::error::Error;
use std::pin::Pin;
use std::sync::Mutex;

/// A ClientWrapper is a wrapper around a specific cloud LLM service.
/// It provides a common interface to interact with the LLMs.
/// It does not keep track of the conversation/session, for that we use an LLMSession
/// which keeps track of the conversation history and other session-specific data
/// and uses a ClientWrapper to interact with the LLM.
// src/client_wrapper

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

/// How many tokens were spent on prompt vs. completion.
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
    /// The actual content of the message.
    pub content: String,
}

/// Represents a chunk of a streaming message response.
#[derive(Clone, Debug)]
pub struct MessageChunk {
    /// The incremental content in this chunk.
    pub content: String,
    /// Whether this is the final chunk in the stream.
    pub is_final: bool,
}

/// Type alias for a Send-able error box
pub type SendError = Box<dyn Error + Send>;

/// Trait defining the interface to interact with various LLM services.
#[async_trait]
pub trait ClientWrapper: Send + Sync {
    /// Send a message to the LLM and get a response.
    /// - `messages`: The messages to send in the request.
    async fn send_message(
        &self,
        messages: Vec<Message>,
        optional_search_parameters: Option<openai_rust::chat::SearchParameters>,
    ) -> Result<Message, Box<dyn Error>>;

    /// Send a message to the LLM and get a streaming response.
    /// - `messages`: The messages to send in the request.
    /// Returns a Stream of MessageChunk items, allowing tokens to be processed as they arrive.
    /// This method has a default implementation that returns an error, so existing
    /// implementations don't break. Clients that support streaming should override this.
    /// Note: The returned stream may not be Send-safe and must be consumed in the same task.
    async fn send_message_stream(
        &self,
        _messages: Vec<Message>,
        _optional_search_parameters: Option<openai_rust::chat::SearchParameters>,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<MessageChunk, SendError>>>>, Box<dyn Error>> {
        Err("Streaming not supported by this client".into())
    }

    /// Hook to retrieve usage from the *last* send_message() call.
    /// Default impl returns None so existing wrappers don’t break.
    fn get_last_usage(&self) -> Option<TokenUsage> {
        self.usage_slot()
            .and_then(|slot| slot.lock().ok().and_then(|u| u.clone()))
    }

    fn usage_slot(&self) -> Option<&Mutex<Option<TokenUsage>>> {
        // ClientWrapper implementations supporting TokenUsage tracking should return a Mutex<Option<TokenUsage>> by overriding this method.
        None
    }
}
