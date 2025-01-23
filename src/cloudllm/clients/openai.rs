/// The `OpenAIClient` struct provides an implementation of the `ClientWrapper` trait for OpenAI's ChatGPT.
/// This allows interactions with the OpenAI ChatGPT LLM REST API, abstracting the underlying details and 
/// providing a consistent interface for sending and receiving messages.
///
/// # Example
///
/// ```rust ignore
/// use cloudllm::clients::openai::OpenAIClient;
/// use cloudllm::client_wrapper::{ClientWrapper, Message, Role};
///
/// let secret_key = "YOUR_OPENAI_SECRET_KEY";
/// let model_name = "gpt-4";
///
/// let client = OpenAIClient::new(secret_key, model_name);
/// let system_prompt = "You are an AI assistant.";
/// let msg = Message { role: Role::User, content: "Hello, World!".to_string() };
/// 
/// let response = client.send_message(vec![Message { role: Role::System, content: system_prompt.to_string() }, msg]).await.unwrap();
/// println!("Assistant: {}", response.content);
/// ```
///
/// # Note
/// You will need to have the OpenAI API key and the desired model name (e.g., "gpt-4") to instantiate and use the client.
///
use std::error::Error;

use async_trait::async_trait;
use openai_rust2 as openai_rust;
use openai_rust::chat;

// src/openai.rs
use crate::cloudllm::client_wrapper::{ClientWrapper, Message, Role};

pub struct OpenAIClient {
    client: openai_rust::Client,
    model: String,
}

impl OpenAIClient {
    pub fn new(secret_key: &str, model_name: &str) -> Self {
        OpenAIClient {
            client: openai_rust::Client::new(secret_key),
            model: model_name.to_string(),
        }
    }

    pub fn new_with_base_url(secret_key: &str, model_name: &str, base_url: &str) -> Self {
        OpenAIClient {
            client: openai_rust::Client::new_with_base_url(secret_key, base_url),
            model: model_name.to_string(),
        }
    }
}

#[async_trait]
impl ClientWrapper for OpenAIClient {
    async fn send_message(
        &self,
        messages: Vec<Message>,
    ) -> Result<Message, Box<dyn Error>> {

        // Convert the provided messages into the format expected by openai_rust
        let formatted_messages = messages
            .into_iter()
            .map(|msg| chat::Message {
                role: match msg.role {
                    Role::System => "system".to_owned(),
                    Role::User => "user".to_owned(),
                    Role::Assistant => "assistant".to_owned(),
                    // Extend this match as new roles are added to the Role enum
                },
                content: msg.content,
            })
            .collect();

        let args = chat::ChatArguments::new(&self.model, formatted_messages);
        let res = self.client.create_chat(args).await?;
        Ok(Message {
            role: Role::Assistant,
            content: res.choices[0].message.content.clone(),
        })
    }
}

