//! The `llm_session` module encapsulates a conversational session with a Language Learning Model (LLM). 
//! It provides the foundational tools necessary for real-time, back-and-forth interactions with the LLM,
//! ensuring that both the user's queries and the LLM's responses are managed and tracked efficiently.
//! 
//! At its core is the `LLMSession` structure, which is responsible for maintaining a running dialogue history,
//! allowing for contextualized exchanges that build upon previous interactions. This session-centric design 
//! means developers can harness it for applications that require dynamic conversations, such as chatbots,
//! virtual assistants, or interactive teaching tools.
//! 
//! With methods like `send_message`, users can seamlessly communicate with the LLM, while other utilities, 
//! such as `set_system_prompt`, offer ways to guide or pivot the direction of the conversation. In essence,
//! this module is the bridge between user inputs and sophisticated model responses, serving as the orchestrator 
//! for intelligent and coherent dialogues with the LLM.
//! `LLMSession` maintains a conversation history while interacting with the LLM (Language Learning Model).
//! To use an OpenAI client wrapper as the client for this session, follow these steps:
//!
//! ## An example
//!
//! `LLMSession` maintains a conversation history while interacting with the LLM (Language Learning Model).
//! To use an OpenAI client wrapper as the client for a session, follow these steps:
//!
//! 1. **Instantiation of OpenAIClient**: 
//! Before creating an LLMSession, you first need an instance of `OpenAIClient`. 
//! This requires your OpenAI secret key and the model name you want to utilize (e.g., "gpt-4").
//!
//! ```rust
//! use crate::cloudllm::clients::openai::OpenAIClient;
//! let secret_key = "YOUR_OPENAI_SECRET_KEY";
//! let model_name = "gpt-4";
//! let openai_client = OpenAIClient::new(secret_key, model_name);
//! ```
//!
//! 2. **Creating an LLMSession with OpenAIClient**: 
//! Now, you can create an `LLMSession` by providing the `OpenAIClient` instance and a system prompt to set the context.
//!
//! ```rust
//! use crate::cloudllm::llm_session::LLMSession;
//! let system_prompt = "You are an AI assistant.";
//! let session = LLMSession::new(openai_client, system_prompt.to_string());
//! ```
//!
//! 3. **Using the Session**: 
//! With the session set up, you can send messages and maintain a conversation history. Each message sent 
//! to the LLM via `send_message` gets appended to the session's history. This ensures a consistent and coherent 
//! interaction over multiple message exchanges.
//!
//! ```rust
//! let user_message = "Hello, World!";
//! let response = session.send_message(Role::User, user_message.to_string()).await.unwrap();
//! println!("Assistant: {}", response.content);
//! ```
//!
//! Keep in mind that the session's history grows with each interaction. Ensure to handle token limits or other 
//! constraints by potentially truncating older parts of the conversation if required.
//!
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
/// * `conversation_history`: A dynamic list that keeps track of messages exchanged in the session.
///   It holds both the user's queries and the LLM's responses, ensuring a contextualized conversation.
///
pub struct LLMSession<T: ClientWrapper> {
    /// The client used for sending messages and communicating with the LLM.
    client: Arc<T>,
    /// A vector that keeps a history of the conversation with the LLM.
    conversation_history: Vec<Message>,
}

impl<T: ClientWrapper> LLMSession<T> {
    pub fn new(client: T, system_prompt: String) -> Self {
        LLMSession {
            client: Arc::new(client),
            // for now doing it the OpenAI way here, but we should probably make this more generic
            conversation_history: vec![Message {
                role: Role::System,
                content: system_prompt.clone(),
            }],
        }
    }

    /// Send a message to the LLM, add it to the conversation history.
    /// Returns the last response from the LLM
    pub async fn send_message(&mut self, role: Role, content: String) -> Result<Message, Box<dyn std::error::Error>> {
        let message = Message {
            role,
            content,
        };

        // Update the conversation history
        self.conversation_history.push(message.clone());

        // Handle token limit or any other constraints by possibly truncating older parts of the conversation history if needed

        let response = self.client.send_message(self.conversation_history.clone()).await.unwrap();
        self.conversation_history.push(response.clone());

        // Add the LLM's responses to the conversation history
        Ok(response)
    }

    /// Set the system prompt for the session.
    pub fn set_system_prompt(&mut self, prompt: String) {
        self.conversation_history[0].content = prompt;
    }
}
