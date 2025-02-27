//! The `llm_session` module encapsulates a conversational session with a Language Learning Model (LLM).
//! It provides the foundational tools necessary for real-time, back-and-forth interactions with the LLM,
//! ensuring that both the user's queries and the LLM's responses are managed and tracked efficiently within
//! specified token limits to comply with model constraints.
//!
//! At its core is the `LLMSession` structure, responsible for maintaining a running dialogue history
//! while adhering to the token limitations of the LLM. This allows for contextualized exchanges that build
//! upon previous interactions without exceeding the model's capacity. This session-centric design
//! enables developers to harness it for applications requiring dynamic conversations, such as chatbots,
//! virtual assistants, or interactive teaching tools.
//!
//! With methods like `send_message`, users can seamlessly communicate with the LLM, while utilities
//! like `set_system_prompt` offer ways to guide or pivot the direction of the conversation. The session
//! automatically manages the conversation history, trimming older messages as necessary to stay within
//! the token limits. In essence, this module bridges user inputs and sophisticated model responses,
//! orchestrating intelligent and coherent dialogues with the LLM.
//!
//! ## Example Usage
//!
//! `LLMSession` maintains a conversation history while interacting with the LLM, ensuring that each
//! exchange is contextualized. To use an OpenAI client wrapper as the client for a session, follow these steps:
//!
//! ### 1. Instantiation of OpenAIClient
//! Before creating an `LLMSession`, you first need an instance of `OpenAIClient`.
//! This requires your OpenAI secret key and the model name you want to utilize (e.g., "gpt-4").
//!
//! ```rust
//! use cloudllm::clients::openai::OpenAIClient;
//! let secret_key = "YOUR_OPENAI_SECRET_KEY";
//! let model_name = "gpt-4";
//! let openai_client = OpenAIClient::new_with_model_string(secret_key, model_name);
//! ```
//!
//! ### 2. Creating an LLMSession with OpenAIClient
//! Now, you can create an `LLMSession` by providing the `OpenAIClient` instance, a system prompt to set the context,
//! and the maximum number of tokens allowed in the conversation (including the system prompt).
//!
//! ```rust
//! use cloudllm::clients::openai::OpenAIClient;
//! use cloudllm::LLMSession;
//! let secret_key = "YOUR_OPENAI_SECRET_KEY";
//! let model_name = "gpt-4";
//! let openai_client = OpenAIClient::new_with_model_string(secret_key, model_name);
//! let system_prompt = "You are an AI assistant.";
//! let max_tokens = 8000; // Adjust based on the model's token limit
//! let mut session = LLMSession::new(openai_client, system_prompt.to_string(), max_tokens);
//! ```
//!
//! ### 3. Using the Session
//! With the session set up, you can send messages and maintain a conversation history. Each message sent
//! to the LLM via `send_message` gets appended to the session's history. The session automatically manages
//! the conversation history to ensure it doesn't exceed the token limit, trimming older messages as needed.
//! This ensures a consistent and coherent interaction over multiple message exchanges.
//!
//! ```rust ignore
//! use cloudllm::client_wrapper::Role;
//! use cloudllm::clients::openai::OpenAIClient;
//! use cloudllm::LLMSession;
//! let secret_key = "YOUR_OPENAI_SECRET_KEY";
//! let model_name = "gpt-4";
//! let openai_client = OpenAIClient::new_with_model_string(secret_key, model_name);
//! let system_prompt = "You are an AI assistant.";
//! let max_tokens = 8000; // Adjust based on the model's token limit
//! let mut session = LLMSession::new(openai_client, system_prompt.to_string(), max_tokens);
//! let user_message = "Hello, World!";
//! let response = session.send_message(Role::User, user_message.to_string()).await.unwrap();
//! println!("Assistant: {}", response.content);
//! ```
//!
//! The session's history grows with each interaction but remains within the token constraints of the LLM.
//! The `LLMSession` handles token limit management internally, so you don't need to manually truncate older parts
//! of the conversation.
//!
//! ### 4. Adjusting the System Prompt
//! You can update the system prompt during the session to change the conversation's context.
//!
//! ```rust ignore
//! let new_system_prompt = "You are a helpful assistant specialized in astronomy.";
//! session.set_system_prompt(new_system_prompt.to_string());
//! ```
//!
//! ### 5. Handling Token Limits
//! The `LLMSession` automatically ensures that the total number of tokens in the conversation history,
//! including the system prompt, does not exceed `max_tokens`. It does this by removing the oldest messages
//! when necessary. If the system prompt and a single message exceed `max_tokens`, you may need to reduce
//! their length or increase `max_tokens` (within the model's limitations).
//!
//! ```rust ignore
//! // Attempt to send a very long message
//! let long_message = "A".repeat(10000); // A long message exceeding token limits
//! match session.send_message(Role::User, long_message).await {
//!     Ok(response) => println!("Assistant: {}", response.content),
//!     Err(e) => eprintln!("Error: {}", e),
//! }
//! ```
//!
//! ## Notes
//!
//! - **Token Counting:** The session uses an approximate method to estimate the number of tokens, assuming
//!   one token per 4 characters. This approximation works reasonably well for English text but may not be exact.
//! - **Error Handling:** Ensure to handle potential errors, especially when exceeding token limits.
//! - **Customization:** You can adjust `max_tokens` based on the model's capabilities and your application's needs.
//!
//! ## Conclusion
//!
//! The `llm_session` module simplifies interactions with LLMs by managing the conversation context and
//! token limitations. By handling the intricacies of session management, it allows developers to focus
//! on building intelligent applications that leverage the power of language models.

use std::sync::Arc;

// src/llm_session.rs
use crate::cloudllm::client_wrapper::{ClientWrapper, Message, Role};

/// Represents a conversational session with an LLM (Language Learning Model).
///
/// `LLMSession` allows for real-time, back-and-forth interactions with the LLM while maintaining
/// a history of the conversation. This ensures that exchanges with the model are contextualized,
/// building upon previous interactions for a more coherent and intelligent dialogue.
///
/// # Fields
///
/// * `client`: The client that communicates with the LLM. It could be any implementation of the
///   `ClientWrapper` trait, like the `OpenAIClient` for interfacing with OpenAI.
///
/// * `system_prompt`: The system prompt that sets the context for the conversation, as a `Message`.
///
/// * `conversation_history`: A dynamic list that keeps the messages exchanged in the session,
///   excluding the system prompt.
///
/// * `max_tokens`: The maximum number of tokens allowed in the conversation history including the system prompt.
///
/// * `token_count`: The current total token count of the system prompt and conversation history.
///
pub struct LLMSession<T: ClientWrapper> {
    /// The client used for sending messages and communicating with the LLM.
    client: Arc<T>,
    /// The system prompt for the session as a `Message`.
    system_prompt: Message,
    /// A vector that keeps the conversation history excluding the system prompt.
    conversation_history: Vec<Message>,
    /// The maximum number of tokens allowed in the conversation.
    max_tokens: usize,
    /// The current total token count.
    token_count: usize,
}

impl<T: ClientWrapper> LLMSession<T> {
    /// Creates a new `LLMSession` with the given client and system prompt.
    /// Initializes the conversation history and sets a default maximum token limit.
    pub fn new(client: T, system_prompt: String, max_tokens: usize) -> Self {
        // Create the system prompt message
        let system_prompt_message = Message {
            role: Role::System,
            content: system_prompt,
        };
        // Count tokens in the system prompt message
        let system_prompt_tokens = count_message_tokens(&system_prompt_message);
        LLMSession {
            client: Arc::new(client),
            system_prompt: system_prompt_message,
            conversation_history: Vec::new(),
            max_tokens,
            token_count: system_prompt_tokens,
        }
    }

    /// Sends a message to the LLM and updates the conversation history.
    /// Maintains the conversation history within the specified token limit.
    /// Returns the response from the LLM.
    /// The `option_url_path` parameter allows for specifying a custom URL path for the request, this
    /// is needed for example in the GeminiClient implementation
    pub async fn send_message(
        &mut self,
        role: Role,
        content: String
    ) -> Result<Message, Box<dyn std::error::Error>> {
        let message = Message { role, content };

        // Count tokens in the new message
        let message_tokens = count_message_tokens(&message);

        // Add the message tokens to the total token count
        self.token_count += message_tokens;

        // Add the new message to the conversation history
        self.conversation_history.push(message);

        // Trim the conversation history to fit within the max_tokens limit
        self.trim_conversation_history();

        // Temporarily add the system prompt to the start of the conversation history
        self.conversation_history
            .insert(0, self.system_prompt.clone());

        // Send the messages to the LLM
        let response = self
            .client
            .send_message(self.conversation_history.clone())
            .await?;

        // Remove the system prompt from the conversation history
        self.conversation_history.remove(0);

        // Count tokens in the response
        let response_tokens = count_message_tokens(&response);

        // Add the response tokens to the total token count
        self.token_count += response_tokens;

        // Add the LLM's response to the conversation history
        self.conversation_history.push(response);

        // Trim the conversation history again after adding the response
        self.trim_conversation_history();

        // Return the last message, which is the assistant's response
        Ok(self.conversation_history.last().unwrap().clone())
    }

    /// Sets a new system prompt for the session.
    /// Updates the token count accordingly.
    pub fn set_system_prompt(&mut self, prompt: String) {
        // Update token count by subtracting old prompt tokens and adding new ones
        let old_prompt_tokens = count_message_tokens(&self.system_prompt);

        self.system_prompt = Message {
            role: Role::System,
            content: prompt,
        };

        let new_prompt_tokens = count_message_tokens(&self.system_prompt);

        self.token_count = self.token_count - old_prompt_tokens + new_prompt_tokens;
    }

    /// Trims the conversation history to ensure the total token count does not exceed max_tokens.
    fn trim_conversation_history(&mut self) {
        while self.token_count > self.max_tokens {
            if !self.conversation_history.is_empty() {
                let removed_message = self.conversation_history.remove(0);
                let removed_tokens = count_message_tokens(&removed_message);
                self.token_count -= removed_tokens;
            } else {
                // Cannot remove any more messages
                break;
            }
        }
    }
}

/// Estimates the number of tokens in a string.
/// Uses an approximate formula: one token per 4 characters.
fn count_tokens(text: &str) -> usize {
    (text.len() / 4).max(1)
}

/// Estimates the number of tokens in a Message, including role annotations if necessary.
fn count_message_tokens(message: &Message) -> usize {
    // Assuming the role adds some fixed number of tokens, e.g., 1 token
    let role_token_count = 1; // Adjust as needed
    let content_token_count = count_tokens(&message.content);
    role_token_count + content_token_count
}
