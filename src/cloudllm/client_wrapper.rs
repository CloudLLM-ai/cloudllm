/// A ClientWrapper is a wrapper around a specific cloud LLM service.
/// It provides a common interface to interact with the LLMs.
/// It does not keep track of the conversation/session, for that we use an LLMSession
/// which keeps track of the conversation history and other session-specific data
/// and uses a ClientWrapper to interact with the LLM.
// src/client_wrapper
use std::error::Error;
use std::sync::Mutex;
use async_trait::async_trait;

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

/// Trait defining the interface to interact with various LLM services.
#[async_trait]
pub trait ClientWrapper: Send + Sync {
    /// Send a message to the LLM and get a response.
    /// - `messages`: The messages to send in the request.
    async fn send_message(&self, messages: Vec<Message>) -> Result<Message, Box<dyn Error>>;

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
