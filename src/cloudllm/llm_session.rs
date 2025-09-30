//! The `llm_session` module manages a conversational session with an LLM,
//! handling not just message history and context pruning, but also
//! real token accounting (input vs. output) for cost estimates.
//!
//! **Key features:**
//! - **Pre-transmission trimming**: optimizes payload size by pruning before sending to LLM.
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
//! The session automatically trims oldest messages before transmission when cumulative tokens exceed the configured window.

use crate::client_wrapper;
use crate::cloudllm::client_wrapper::{ClientWrapper, Message, Role};
use openai_rust2 as openai_rust;
use std::sync::Arc;

/// A conversation session with an LLM, including:
///
/// - `client`: your `ClientWrapper` (e.g. `OpenAIClient`).
/// - `system_prompt`: the context-steering system message.
/// - `conversation_history`: all user & assistant messages (excluding system prompt).
/// - `cached_token_counts`: cached token estimates for each message in conversation_history.
/// - `max_tokens`: your configured context window size.
/// - `total_input_tokens`: sum of all prompt tokens sent so far.
/// - `total_output_tokens`: sum of all completion tokens received so far.
/// - `total_context_tokens`: shortcut for input + output totals.
/// - `total_token_count`: total tokens used in the current session.
pub struct LLMSession {
    client: Arc<dyn ClientWrapper>,
    system_prompt: Message,
    conversation_history: Vec<Message>,
    cached_token_counts: Vec<usize>,
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
            cached_token_counts: Vec::new(),
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

        // Cache the token count for the new message before adding it
        let message_token_count = estimate_message_token_count(&message);

        // Add the new message to the conversation history
        self.conversation_history.push(message);
        self.cached_token_counts.push(message_token_count);

        // Estimate total tokens before sending:
        // system prompt + all conversation history
        let system_prompt_tokens = estimate_message_token_count(&self.system_prompt);
        let mut estimated_total: usize = system_prompt_tokens;
        for msg in &self.conversation_history {
            estimated_total += estimate_message_token_count(msg);
        }

        // Trim oldest messages if estimated total exceeds max_tokens
        while estimated_total > self.max_tokens && !self.conversation_history.is_empty() {
            self.conversation_history.remove(0);
            if !self.cached_token_counts.is_empty() {
                let removed_tokens = self.cached_token_counts.remove(0);
                estimated_total = estimated_total.saturating_sub(removed_tokens);
            }
        }

        // Temporarily add the system prompt to the start of the conversation history
        self.conversation_history
            .insert(0, self.system_prompt.clone());

        // Send the messages to the LLM
        let response = self
            .client
            .send_message(&self.conversation_history, optional_search_parameters)
            .await?;

        // Remove the system prompt from the conversation history
        self.conversation_history.remove(0);

        // Add the LLM's response to the conversation history
        let response_token_count = estimate_message_token_count(&response);
        self.cached_token_counts.push(response_token_count);
        self.conversation_history.push(response.clone());

        // Update token counts from actual provider usage
        if let Some(usage) = self.client.get_last_usage().await{
            self.total_input_tokens = usage.input_tokens;
            self.total_output_tokens = usage.output_tokens;
            self.total_token_count = usage.total_tokens;

            // Trim again if actual usage exceeded max_tokens
            // This can happen if our estimate was off or if the response was large
            if self.total_token_count > self.max_tokens {
                let mut excess = self.total_token_count - self.max_tokens;
                
                // Remove oldest messages until we're back under the limit
                while excess > 0 && !self.conversation_history.is_empty() {
                    self.conversation_history.remove(0);
                    let removed = self.cached_token_counts.remove(0);
                    excess = excess.saturating_sub(removed);
                }
            }
        }

        // Return the last message, which is the assistant's response
        Ok(response)
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

    pub fn get_conversation_history(&self) -> &Vec<Message> {
        &self.conversation_history
    }

    pub fn get_cached_token_counts(&self) -> &Vec<usize> {
        &self.cached_token_counts
    }
}

/// Estimates the number of tokens in a string.
/// Uses an approximate formula: one token per 4 characters.
pub fn estimate_token_count(text: &str) -> usize {
    (text.len() / 4).max(1)
}

/// Estimates the number of tokens in a Message, including role annotations.
pub fn estimate_message_token_count(message: &Message) -> usize {
    // Assuming the role adds some fixed number of tokens, e.g., 1 token
    let role_token_count = 1;
    let content_token_count = estimate_token_count(&message.content);
    role_token_count + content_token_count
}
