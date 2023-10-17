/// The `llm_session` module encapsulates a conversational session with a Language Learning Model (LLM). 
/// It provides the foundational tools necessary for real-time, back-and-forth interactions with the LLM,
/// ensuring that both the user's queries and the LLM's responses are managed and tracked efficiently.
/// 
/// At its core is the `LLMSession` structure, which is responsible for maintaining a running dialogue history,
/// allowing for contextualized exchanges that build upon previous interactions. This session-centric design 
/// means developers can harness it for applications that require dynamic conversations, such as chatbots,
/// virtual assistants, or interactive teaching tools.
/// 
/// With methods like `send_message`, users can seamlessly communicate with the LLM, while other utilities, 
/// such as `set_system_prompt`, offer ways to guide or pivot the direction of the conversation. In essence,
/// this module is the bridge between user inputs and sophisticated model responses, serving as the orchestrator 
/// for intelligent and coherent dialogues with the LLM.

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
