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
//! ```rust,no_run
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
use bumpalo::Bump;
use openai_rust2::chat::{GrokTool, OpenAITool};
use std::sync::Arc;

pub struct LLMSession {
    /// Provider specific client used to execute messages.
    client: Arc<dyn ClientWrapper>,
    /// System priming message that is always prepended to requests.
    system_prompt: Message,
    /// Rolling conversation history excluding the system prompt.
    conversation_history: Vec<Message>,
    /// Cached token estimates for each entry in `conversation_history`.
    cached_token_counts: Vec<usize>,
    /// Hard limit on context window size in tokens.
    max_tokens: usize,
    /// Sum of prompt tokens consumed across the session (as reported by the provider).
    total_input_tokens: usize,
    /// Sum of completion tokens consumed across the session (as reported by the provider).
    total_output_tokens: usize,
    /// Total tokens consumed combining input and output counts.
    total_token_count: usize,
    /// Arena allocator used to back message content allocations without repeated heap traffic.
    arena: Bump,
    /// Scratch buffer reused when assembling the request payload sent to the provider.
    request_buffer: Vec<Message>,
}

impl LLMSession {
    /// Create a new session for the provided client, system prompt, and token budget.
    ///
    /// The `max_tokens` argument should reflect the effective context window of the underlying
    /// provider.  The session will proactively prune conversation history to remain within that
    /// limit.
    pub fn new(client: Arc<dyn ClientWrapper>, system_prompt: String, max_tokens: usize) -> Self {
        let arena = Bump::new();

        // Allocate system prompt in arena and create Arc<str> from it
        let system_prompt_str = arena.alloc_str(&system_prompt);
        let system_prompt_arc: Arc<str> = Arc::from(system_prompt_str);

        // Create the system prompt message
        let system_prompt_message = Message {
            role: Role::System,
            content: system_prompt_arc,
        };

        LLMSession {
            client,
            system_prompt: system_prompt_message,
            conversation_history: Vec::new(),
            cached_token_counts: Vec::new(),
            max_tokens,
            total_input_tokens: 0,
            total_output_tokens: 0,
            total_token_count: 0,
            arena,
            request_buffer: Vec::new(),
        }
    }

    /// Send a message, update the session history, and return the assistant reply.
    ///
    /// The method performs several steps automatically:
    ///
    /// 1. Append the caller supplied message to the history.
    /// 2. Estimate cumulative token usage and trim oldest exchanges if the soft limit would be
    ///    breached.
    /// 3. Dispatch the request via the wrapped [`ClientWrapper`].
    /// 4. Persist the response in the conversation history.
    /// 5. Pull provider reported token usage and update the session counters.
    /// 6. Perform a second pruning pass if the provider indicates the hard cap was exceeded.
    ///
    /// Inspect [`LLMSession::token_usage`] to read the cumulative accounting.
    pub async fn send_message(
        &mut self,
        role: Role,
        content: String,
        optional_grok_tools: Option<Vec<GrokTool>>,
        optional_openai_tools: Option<Vec<OpenAITool>>,
    ) -> Result<Message, Box<dyn std::error::Error>> {
        // Allocate message content in arena and create Arc<str>
        let content_str = self.arena.alloc_str(&content);
        let content_arc: Arc<str> = Arc::from(content_str);

        let message = Message {
            role,
            content: content_arc,
        };

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

        // Build request buffer by reusing the existing Vec to avoid allocation
        self.request_buffer.clear();
        self.request_buffer
            .reserve(1 + self.conversation_history.len());
        self.request_buffer.push(self.system_prompt.clone());
        self.request_buffer
            .extend_from_slice(&self.conversation_history);

        // Send the messages to the LLM
        let response = self
            .client
            .send_message(
                &self.request_buffer,
                optional_grok_tools,
                optional_openai_tools,
            )
            .await?;

        // Clone response for return before adding to history
        // This way we keep the owned response from client, push it to history (no clone),
        // and return a clone only for the caller
        let response_to_return = response.clone();

        // Cache token count and add the owned response to history
        let response_token_count = estimate_message_token_count(&response);
        self.cached_token_counts.push(response_token_count);
        self.conversation_history.push(response);

        // Update token counts from actual provider usage
        if let Some(usage) = self.client.get_last_usage().await {
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

        // Return the response (cloned earlier for caller)
        Ok(response_to_return)
    }

    /// Send a message and return a stream of partial responses when the provider supports it.
    ///
    /// The session keeps the optimistic trimming behaviour of [`LLMSession::send_message`] but
    /// does **not** automatically append the streamed chunks to the history.  Callers that wish
    /// to persist the streamed output should collect it and feed the result back through
    /// [`LLMSession::send_message`].
    ///
    /// Returning `Ok(None)` indicates that the wrapped client does not support streaming.
    pub async fn send_message_stream(
        &mut self,
        role: Role,
        content: String,
        optional_grok_tools: Option<Vec<GrokTool>>,
        optional_openai_tools: Option<Vec<OpenAITool>>,
    ) -> Result<Option<crate::client_wrapper::MessageChunkStream>, Box<dyn std::error::Error>> {
        // Allocate message content in arena and create Arc<str>
        let content_str = self.arena.alloc_str(&content);
        let content_arc: Arc<str> = Arc::from(content_str);

        let message = Message {
            role,
            content: content_arc,
        };

        // Cache the token count for the new message before adding it
        let message_token_count = estimate_message_token_count(&message);

        // Add the new message to the conversation history
        self.conversation_history.push(message);
        self.cached_token_counts.push(message_token_count);

        // Estimate total tokens before sending
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

        // Build request buffer
        self.request_buffer.clear();
        self.request_buffer
            .reserve(1 + self.conversation_history.len());
        self.request_buffer.push(self.system_prompt.clone());
        self.request_buffer
            .extend_from_slice(&self.conversation_history);

        // Get the streaming response
        let stream_result = self
            .client
            .send_message_stream(
                &self.request_buffer,
                optional_grok_tools,
                optional_openai_tools,
            )
            .await?;

        // If streaming is not supported, remove the message we added
        if stream_result.is_none() {
            self.conversation_history.pop();
            self.cached_token_counts.pop();
        }

        // Return the stream (caller will handle accumulating and adding to history if needed)
        Ok(stream_result)
    }

    /// Replace the system prompt that prefixes every request.
    ///
    /// Note that changing the system prompt does **not** attempt to re-estimate token usage –
    /// the provided prompt should respect the configured `max_tokens` limit.
    pub fn set_system_prompt(&mut self, prompt: String) {
        // Allocate prompt in arena and create Arc<str>
        let prompt_str = self.arena.alloc_str(&prompt);
        let prompt_arc: Arc<str> = Arc::from(prompt_str);

        self.system_prompt = Message {
            role: Role::System,
            content: prompt_arc,
        };
    }

    /// Return the system prompt content as a string slice.
    pub fn system_prompt_text(&self) -> &str {
        &self.system_prompt.content
    }

    /// Return the cumulative token usage statistics tracked for this session.
    pub fn token_usage(&self) -> client_wrapper::TokenUsage {
        client_wrapper::TokenUsage {
            input_tokens: self.total_input_tokens,
            output_tokens: self.total_output_tokens,
            total_tokens: self.total_token_count,
        }
    }

    /// Return the model identifier exposed by the underlying client.
    pub fn model_name(&self) -> String {
        self.client.model_name().to_string()
    }

    /// Ask the wrapped client for the most recent token usage report.
    pub async fn last_token_usage(&self) -> Option<client_wrapper::TokenUsage> {
        self.client.get_last_usage().await
    }

    /// Current maximum token budget configured for the session.
    pub fn get_max_tokens(&self) -> usize {
        self.max_tokens
    }

    /// Borrow the conversation history (excluding the system prompt).
    pub fn get_conversation_history(&self) -> &Vec<Message> {
        &self.conversation_history
    }

    /// Borrow the cached token counts for each entry in the conversation history.
    pub fn get_cached_token_counts(&self) -> &Vec<usize> {
        &self.cached_token_counts
    }

    /// Borrow the system prompt message currently in use.
    pub fn get_system_prompt(&self) -> &Message {
        &self.system_prompt
    }

    /// Borrow the underlying client.
    ///
    /// Useful for creating new sessions that share the same provider (e.g.
    /// [`Agent::fork`](crate::Agent::fork)).
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use cloudllm::LLMSession;
    /// use cloudllm::clients::openai::OpenAIClient;
    /// use std::sync::Arc;
    ///
    /// let session = LLMSession::new(
    ///     Arc::new(OpenAIClient::new_with_model_string("key", "gpt-4o")),
    ///     "system".into(),
    ///     8_192,
    /// );
    ///
    /// // Clone the client to create a sibling session
    /// let sibling = LLMSession::new(
    ///     session.client().clone(),
    ///     "different system prompt".into(),
    ///     4_096,
    /// );
    /// ```
    pub fn client(&self) -> &Arc<dyn ClientWrapper> {
        &self.client
    }

    /// Wipe conversation history and cached token counts.
    ///
    /// The system prompt is preserved. Cumulative token counters are **not**
    /// reset so lifetime accounting remains accurate.
    ///
    /// This is typically called by context strategies after persisting a
    /// compression summary, right before injecting the bootstrap prompt
    /// via [`inject_message`](LLMSession::inject_message).
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use cloudllm::LLMSession;
    /// use cloudllm::client_wrapper::Role;
    /// use cloudllm::clients::openai::OpenAIClient;
    /// use std::sync::Arc;
    ///
    /// let mut session = LLMSession::new(
    ///     Arc::new(OpenAIClient::new_with_model_string("key", "gpt-4o")),
    ///     "system".into(),
    ///     8_192,
    /// );
    ///
    /// // After many messages, clear and start fresh
    /// session.clear_history();
    /// assert_eq!(session.get_conversation_history().len(), 0);
    /// assert_eq!(session.estimated_history_tokens(), 0);
    /// ```
    pub fn clear_history(&mut self) {
        self.conversation_history.clear();
        self.cached_token_counts.clear();
    }

    /// Add a message to history without sending it to the LLM.
    ///
    /// The content is allocated in the session's arena and its token count
    /// is cached, just like messages added via
    /// [`send_message`](LLMSession::send_message).  Use this to inject
    /// [`ThoughtChain`](crate::ThoughtChain) context into a fresh or cleared
    /// session.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use cloudllm::LLMSession;
    /// use cloudllm::client_wrapper::Role;
    /// use cloudllm::clients::openai::OpenAIClient;
    /// use std::sync::Arc;
    ///
    /// let mut session = LLMSession::new(
    ///     Arc::new(OpenAIClient::new_with_model_string("key", "gpt-4o")),
    ///     "system".into(),
    ///     8_192,
    /// );
    ///
    /// session.inject_message(
    ///     Role::System,
    ///     "=== RESTORED CONTEXT ===\nPrevious findings...".into(),
    /// );
    /// assert_eq!(session.get_conversation_history().len(), 1);
    /// ```
    pub fn inject_message(&mut self, role: Role, content: String) {
        let content_str = self.arena.alloc_str(&content);
        let content_arc: Arc<str> = Arc::from(content_str);
        let message = Message {
            role,
            content: content_arc,
        };
        let token_count = estimate_message_token_count(&message);
        self.conversation_history.push(message);
        self.cached_token_counts.push(token_count);
    }

    /// Sum of estimated tokens across all messages currently in history.
    ///
    /// This is the value that [`ContextStrategy`](crate::context_strategy::ContextStrategy)
    /// implementations compare against [`get_max_tokens`](LLMSession::get_max_tokens)
    /// to decide when to compact.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use cloudllm::LLMSession;
    /// use cloudllm::client_wrapper::Role;
    /// use cloudllm::clients::openai::OpenAIClient;
    /// use std::sync::Arc;
    ///
    /// let mut session = LLMSession::new(
    ///     Arc::new(OpenAIClient::new_with_model_string("key", "gpt-4o")),
    ///     "system".into(),
    ///     8_192,
    /// );
    ///
    /// assert_eq!(session.estimated_history_tokens(), 0);
    /// session.inject_message(Role::User, "Hello world".into());
    /// assert!(session.estimated_history_tokens() > 0);
    /// ```
    pub fn estimated_history_tokens(&self) -> usize {
        self.cached_token_counts.iter().sum()
    }
}

/// Estimates the number of tokens in a string.
/// Uses the common approximation of one token per four UTF-8 characters with a floor of one.
pub fn estimate_token_count(text: &str) -> usize {
    (text.len() / 4).max(1)
}

/// Estimate tokens for a [`Message`] by combining role overhead and body length.
pub fn estimate_message_token_count(message: &Message) -> usize {
    // Assuming the role adds some fixed number of tokens, e.g., 1 token
    let role_token_count = 1;
    let content_token_count = estimate_token_count(&message.content);
    role_token_count + content_token_count
}
