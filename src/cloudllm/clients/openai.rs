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
use log::error;
use openai_rust::chat;
use openai_rust2 as openai_rust;

// src/openai.rs
use crate::cloudllm::client_wrapper::{ClientWrapper, Message, Role};

pub enum Model {
    GPT4o,           // input $2.5/1M tokens, cached input $1.25/1M tokens, output $10/1M tokens
    ChatGPT4oLatest, // latest used in ChatGPT
    GPt4oMini,
    O1,
    O1Mini,
    O1Preview,
    O3Mini,
    O4Mini,
    O4MiniHigh,
    O3,
    GPT4oRealtimePreview,
    GPT4oMiniRealtimePreview,
    GPT4oAudioPreview,
    GPT45Preview, // input $75/1M tokens, cached input $37.5/1M tokens, output $150/1M tokens
    GPT41,        // input $2/1M tokens, cached input $0.5/1M tokens, output $8/1M tokens
    GPT41Mini,    // input $0.4/1M tokens, cached input $0.1/1M tokens, output $1.6/1M tokens
    GPT41Nano,    // input $0.1/1M tokens, cached input $0.025/1M tokens, output $0.4/1M tokens
}

pub fn model_to_string(model: Model) -> String {
    match model {
        Model::GPT4o => "gpt-4o".to_string(),
        Model::ChatGPT4oLatest => "chatgpt-4o-latest".to_string(),
        Model::GPt4oMini => "gpt-4o-mini".to_string(),
        Model::O1 => "o1".to_string(),
        Model::O1Mini => "o1-mini".to_string(),
        Model::O1Preview => "o1-preview".to_string(),
        Model::O3Mini => "o3-mini".to_string(),
        Model::O4Mini => "o4-mini".to_string(),
        Model::O4MiniHigh => "o4-mini-high".to_string(),
        Model::O3 => "o3".to_string(),
        Model::GPT4oRealtimePreview => "gpt-4o-realtime-preview".to_string(),
        Model::GPT4oMiniRealtimePreview => "gpt-4o-mini-realtime-preview".to_string(),
        Model::GPT4oAudioPreview => "gpt-4o-audio-preview".to_string(),
        Model::GPT45Preview => "gpt-4.5-preview".to_string(),
        Model::GPT41 => "gpt-4.1".to_string(),
        Model::GPT41Mini => "gpt-4.1-mini".to_string(),
        Model::GPT41Nano => "gpt-4.1-nano".to_string(),
    }
}

pub struct OpenAIClient {
    client: openai_rust::Client,
    model: String,
}

impl OpenAIClient {
    pub fn new_with_model_string(secret_key: &str, model_name: &str) -> Self {
        OpenAIClient {
            client: openai_rust::Client::new(secret_key),
            model: model_name.to_string(),
        }
    }

    pub fn new_with_model_enum(secret_key: &str, model: Model) -> Self {
        Self::new_with_model_string(secret_key, &model_to_string(model))
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
    async fn send_message(&self, messages: Vec<Message>) -> Result<Message, Box<dyn Error>> {
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
        let url_path_string = "/v1/chat/completions".to_string();

        let res = self.client.create_chat(args, Some(url_path_string)).await;
        match res {
            Ok(response) => Ok(Message {
                role: Role::Assistant,
                content: response.choices[0].message.content.clone(),
            }),
            Err(err) => {
                error!("OpenAI API Error: {}", err); // Log the entire error
                Err(err.into()) // Convert the error to Box<dyn Error>
            }
        }
    }
}
