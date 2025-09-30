//! The `llm_session` module manages a conversational session with an LLM,
//! handling not just message history and context pruning, but also
//! real token accounting (input vs. output) for cost estimates.
//!
//! **Key features:**
//! - **Automatic context trimming**: never exceed your `max_tokens` window.
//! - **Token tracking**: accumulates `input_tokens` & `output_tokens` per call.
//! - **Easy inspection**: call `session.token_usage()` to get a `TokenUsage` struct.
//!
//! ## Quickstart
//!
//! ```rust
//! use std::sync::Arc;
//! use tokio::runtime::Runtime;
//! use cloudllm::client_wrapper::Role;
//! use cloudllm::clients::openai::OpenAIClient;
//! use cloudllm::clients::openai::Model;
//! use cloudllm::LLMSession;
//!
//! // 1) Build the client & session
//! let secret_key : String = std::env::var("OPEN_AI_SECRET").expect("OPEN_AI_SECRET not set");
//! let client = OpenAIClient::new_with_model_enum(&secret_key, Model::GPT41Nano);
//! let mut session = LLMSession::new(
//!     Arc::new(client),
//!     "You are a bilingual crypto journalist.".into(),
//!     8_192  // max context window
//! );
//!
//! // 2) Send a message
//! let rt = Runtime::new().unwrap();
//!
//! let reply = rt.block_on(async {
//!    match session.send_message(Role::User, "Hola, ¿cómo estás?".into(), None).await {
//!        Ok(response) => response,
//!     Err(e) => {
//!         panic!("client error: {}", e);
//!     }
//!   } // await
//! });
//! println!("Assistant: {}", reply.content);
//!
//! // 3) Inspect token usage so far
//! let usage = session.token_usage();
//! println!(
//!     "Input: {} tokens, Output: {} tokens, Total: {} tokens",
//!     usage.input_tokens, usage.output_tokens, usage.total_tokens
//! );
//! ```
//!
//! The session automatically prunes oldest messages when cumulative tokens exceed the configured window.

use crate::client_wrapper;
use std::sync::Arc;
// src/llm_session.rs
use crate::cloudllm::client_wrapper::{ClientWrapper, Message, Role};
use openai_rust2 as openai_rust;

/// A conversation session with an LLM, including:
///
/// - `client`: your `ClientWrapper` (e.g. `OpenAIClient`).
/// - `system_prompt`: the context-steering system message.
/// - `conversation_history`: all user & assistant messages (excluding system prompt).
/// - `max_tokens`: your configured context window size.
/// - `total_input_tokens`: sum of all prompt tokens sent so far.
/// - `total_output_tokens`: sum of all completion tokens received so far.
/// - `total_context_tokens`: shortcut for input + output totals.
/// - `total_token_count`: total tokens used in the current session.
pub struct LLMSession {
    client: Arc<dyn ClientWrapper>,
    system_prompt: Message,
    conversation_history: Vec<Message>,
    max_tokens: usize,
    total_input_tokens: usize,
    total_output_tokens: usize,
    total_token_count: usize,
}

impl LLMSession {
    /// Creates a new `LLMSession` with the given client and system prompt.
    /// Initializes the conversation history and sets a default maximum token limit.
    pub fn new(client: Arc<dyn ClientWrapper>, system_prompt: String, max_tokens: usize) -> Self {
        // Create the system prompt message
        let system_prompt_message = Message {
            role: Role::System,
            content: system_prompt,
        };
        // Count tokens in the system prompt message
        LLMSession {
            client,
            system_prompt: system_prompt_message,
            conversation_history: Vec::new(),
            max_tokens,
            total_input_tokens: 0,
            total_output_tokens: 0,
            total_token_count: 0,
        }
    }

    /// Sends a user/system message, receives the assistant's reply, and
    /// automatically:
    /// 1. Adds the message to history
    /// 2. Trims oldest messages if estimated tokens exceed `max_tokens` (pre-transmission)
    /// 3. Calls into your client's `send_message(...)` with trimmed history
    /// 4. Pulls real token usage via `client.get_last_usage()`
    /// 5. Updates `total_input_tokens`, `total_output_tokens`
    /// 6. Trims again if actual usage exceeds `max_tokens` (post-response)
    ///
    /// Returns the assistant's `Message`; call `session.token_usage()`
    /// to see your cumulative usage.
    pub async fn send_message(
        &mut self,
        role: Role,
        content: String,
        optional_search_parameters: Option<openai_rust::chat::SearchParameters>,
    ) -> Result<Message, Box<dyn std::error::Error>> {
        let message = Message { role, content };

        // Add the new message to the conversation history
        self.conversation_history.push(message);

        // Estimate total tokens before sending:
        // system prompt + all conversation history
        let system_prompt_tokens = estimate_message_token_count(&self.system_prompt);
        let mut estimated_total: usize = system_prompt_tokens;
        for msg in &self.conversation_history {
            estimated_total += estimate_message_token_count(msg);
        }

        // Trim oldest messages if estimated total exceeds max_tokens
        while estimated_total > self.max_tokens && !self.conversation_history.is_empty() {
            let removed_msg = self.conversation_history.remove(0);
            let removed_tokens = estimate_message_token_count(&removed_msg);
            estimated_total = estimated_total.saturating_sub(removed_tokens);
        }

        // Temporarily add the system prompt to the start of the conversation history
        self.conversation_history
            .insert(0, self.system_prompt.clone());

        // Send the messages to the LLM
        let response = self
            .client
            .send_message(
                self.conversation_history.clone(),
                optional_search_parameters,
            )
            .await?;

        // Remove the system prompt from the conversation history
        self.conversation_history.remove(0);

        // Add the LLM's response to the conversation history
        self.conversation_history.push(response);

        // Update token counts from actual provider usage
        if let Some(usage) = self.client.get_last_usage() {
            self.total_input_tokens = usage.input_tokens;
            self.total_output_tokens = usage.output_tokens;
            self.total_token_count = usage.total_tokens;

            // Trim again if actual usage exceeded max_tokens
            // This can happen if our estimate was off or if the response was large
            if self.total_token_count > self.max_tokens {
                let mut excess = self.total_token_count - self.max_tokens;
                
                // Remove oldest messages until we're back under the limit
                while excess > 0 && !self.conversation_history.is_empty() {
                    let msg = self.conversation_history.remove(0);
                    let removed = estimate_message_token_count(&msg);
                    excess = excess.saturating_sub(removed);
                }
            }
        }

        // Return the last message, which is the assistant's response
        Ok(self.conversation_history.last().unwrap().clone())
    }

    /// Sets a new system prompt for the session.
    /// Updates the token count accordingly.
    pub fn set_system_prompt(&mut self, prompt: String) {
        self.system_prompt = Message {
            role: Role::System,
            content: prompt,
        };
    }

    /// Returns the current token usage statistics
    pub fn token_usage(&self) -> client_wrapper::TokenUsage {
        client_wrapper::TokenUsage {
            input_tokens: self.total_input_tokens,
            output_tokens: self.total_output_tokens,
            total_tokens: self.total_token_count,
        }
    }

    pub fn get_max_tokens(&self) -> usize {
        self.max_tokens
    }
}

/// Estimates the number of tokens in a string.
/// Uses an approximate formula: one token per 4 characters.
fn estimate_token_count(text: &str) -> usize {
    (text.len() / 4).max(1)
}

/// Estimates the number of tokens in a Message, including role annotations.
fn estimate_message_token_count(message: &Message) -> usize {
    // Assuming the role adds some fixed number of tokens, e.g., 1 token
    let role_token_count = 1;
    let content_token_count = estimate_token_count(&message.content);
    role_token_count + content_token_count
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cloudllm::client_wrapper::{ClientWrapper, Message, Role, TokenUsage};
    use async_trait::async_trait;
    use std::sync::Mutex;
    use std::error::Error;

    /// A mock client that returns a fixed response and tracks how many messages it receives
    struct MockClient {
        response_content: String,
        last_message_count: Mutex<usize>,
        usage: Mutex<Option<TokenUsage>>,
    }

    impl MockClient {
        fn new(response: &str) -> Self {
            MockClient {
                response_content: response.to_string(),
                last_message_count: Mutex::new(0),
                usage: Mutex::new(None),
            }
        }

        fn get_last_message_count(&self) -> usize {
            *self.last_message_count.lock().unwrap()
        }
    }

    #[async_trait]
    impl ClientWrapper for MockClient {
        async fn send_message(
            &self,
            messages: Vec<Message>,
            _optional_search_parameters: Option<openai_rust::chat::SearchParameters>,
        ) -> Result<Message, Box<dyn Error>> {
            // Record how many messages were sent
            *self.last_message_count.lock().unwrap() = messages.len();
            
            // Calculate token usage
            let mut input_tokens = 0;
            for msg in &messages {
                input_tokens += estimate_message_token_count(msg);
            }
            let output_tokens = estimate_message_token_count(&Message {
                role: Role::Assistant,
                content: self.response_content.clone(),
            });
            
            *self.usage.lock().unwrap() = Some(TokenUsage {
                input_tokens,
                output_tokens,
                total_tokens: input_tokens + output_tokens,
            });

            Ok(Message {
                role: Role::Assistant,
                content: self.response_content.clone(),
            })
        }

        fn usage_slot(&self) -> Option<&Mutex<Option<TokenUsage>>> {
            Some(&self.usage)
        }
    }

    #[tokio::test]
    async fn test_pre_transmission_trimming() {
        // Create a mock client
        let client = Arc::new(MockClient::new("Response"));
        
        // Create a session with a very small max_tokens limit
        // System prompt: "System" = (6/4).max(1) + 1 = 2 + 1 = 3 tokens
        let mut session = LLMSession::new(
            client.clone(),
            "System".to_string(),
            20, // Very small limit to trigger trimming
        );

        // Add several messages that exceed the limit
        // Each message with 4 chars = (4/4).max(1) + 1 = 1 + 1 = 2 tokens
        let _ = session.send_message(Role::User, "Msg1".to_string(), None).await;
        let _ = session.send_message(Role::User, "Msg2".to_string(), None).await;
        let _ = session.send_message(Role::User, "Msg3".to_string(), None).await;
        
        // Add a large message that should trigger trimming
        // 40 chars = (40/4).max(1) + 1 = 10 + 1 = 11 tokens
        let large_msg = "0123456789012345678901234567890123456789"; // 40 chars
        let _ = session.send_message(Role::User, large_msg.to_string(), None).await;

        // The client should have received fewer messages than we sent
        // because old messages should have been trimmed before transmission
        let message_count = client.get_last_message_count();
        
        // With max_tokens=20:
        // System prompt (3 tokens) + large message (11 tokens) = 14 tokens
        // We should have trimmed old messages to stay under 20
        // The last call should have sent: system prompt + some history + large message
        assert!(message_count > 0, "Should have sent at least the system prompt and new message");
        assert!(message_count < 6, "Should have trimmed some messages (system + 4 user + 4 assistant = 9 total before trim)");
        
        // Verify that conversation history exists
        assert!(!session.conversation_history.is_empty(), "Conversation history should not be empty");
    }

    #[tokio::test]
    async fn test_no_trimming_when_under_limit() {
        let client = Arc::new(MockClient::new("OK"));
        
        // Large max_tokens limit - no trimming should occur
        let mut session = LLMSession::new(
            client.clone(),
            "System".to_string(),
            10000,
        );

        // Add a few small messages
        let _ = session.send_message(Role::User, "Hi".to_string(), None).await;
        let _ = session.send_message(Role::User, "Hello".to_string(), None).await;

        // The last send should include: system prompt + first user message + first assistant response + second user message
        // = 1 system + 1 user + 1 assistant + 1 user = 4 messages
        let message_count = client.get_last_message_count();
        assert_eq!(message_count, 4, "Should have sent all messages without trimming");
    }

    #[test]
    fn test_estimate_token_count() {
        assert_eq!(estimate_token_count(""), 1); // min 1 token
        assert_eq!(estimate_token_count("test"), 1); // 4 chars = 1 token
        assert_eq!(estimate_token_count("12345678"), 2); // 8 chars = 2 tokens
        assert_eq!(estimate_token_count("123456789012"), 3); // 12 chars = 3 tokens
    }

    #[test]
    fn test_estimate_message_token_count() {
        let msg = Message {
            role: Role::User,
            content: "test".to_string(), // 4 chars = 1 token
        };
        // 1 (role) + 1 (content) = 2 tokens
        assert_eq!(estimate_message_token_count(&msg), 2);

        let long_msg = Message {
            role: Role::Assistant,
            content: "12345678".to_string(), // 8 chars = 2 tokens
        };
        // 1 (role) + 2 (content) = 3 tokens
        assert_eq!(estimate_message_token_count(&long_msg), 3);
    }
}
