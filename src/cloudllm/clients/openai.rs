use std::error::Error;

use async_trait::async_trait;
use openai_rust::chat;

// src/openai.rs
use crate::cloudllm::client_wrapper::{ClientWrapper, Message, Role};

pub struct OpenAIClient {
    client: openai_rust::Client,
}

impl OpenAIClient {
    pub fn new(secret_key: &str) -> Self {
        OpenAIClient {
            client: openai_rust::Client::new(secret_key)
        }
    }
}

#[async_trait]
impl ClientWrapper for OpenAIClient {
    async fn send_message(
        &self,
        model: &str,
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

        let args = chat::ChatArguments::new(model, formatted_messages);
        let res = self.client.create_chat(args).await?;
        Ok(Message {
            role: Role::Assistant,
            content: res.choices[0].message.content.clone(),
        })
    }
}

