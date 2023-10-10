use std::sync::Arc;

// src/llm_session.rs
use crate::cloudllm::client_wrapper::{ClientWrapper, Message, Role};

pub struct LLMSession<T: ClientWrapper> {
    client: Arc<T>,
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
    pub async fn send_message(&mut self, model: &str, role: Role, content: String) -> Result<Message, Box<dyn std::error::Error>> {
        let message = Message {
            role,
            content,
        };

        // Update the conversation history
        self.conversation_history.push(message.clone());

        // Handle token limit or any other constraints by possibly truncating older parts of the conversation history if needed

        let response = self.client.send_message(model, self.conversation_history.clone()).await?;
        self.conversation_history.push(response.clone());

        // Add the LLM's responses to the conversation history
        Ok(response)
    }

    /// Set the system prompt for the session.
    pub fn set_system_prompt(&mut self, prompt: String) {
        self.conversation_history[0].content = prompt;
    }
}
