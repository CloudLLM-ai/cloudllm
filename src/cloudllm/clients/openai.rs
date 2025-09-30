//! The `OpenAIClient` struct implements `ClientWrapper` for OpenAI’s Chat API,
//! capturing both the assistant response and detailed token usage (input vs output)
//! for cost tracking.
//!
//! # Key Features
//!
//! - **send_message(...)**: unchanged signature; returns a `Message` as before.
//! - **Automatic Usage Capture**: stores the latest `TokenUsage` (input_tokens, output_tokens, total_tokens) internally.
//! - **Inspect Usage**: call `get_last_usage()` after `send_message()` to retrieve actual usage stats.
//!
//! # Example
//!
//! ```rust
//! use cloudllm::clients::openai::{OpenAIClient, Model};
//! use cloudllm::client_wrapper::{ClientWrapper, Message, Role};
//!
//! #[tokio::main]
//! async fn main() {
//!     // Initialize with your API key and model enum.
//!     let secret_key : String = std::env::var("OPEN_AI_SECRET").expect("OPEN_AI_SECRET not set");
//!     let client = OpenAIClient::new_with_model_enum(&secret_key, Model::GPT41Nano);
//!
//!     // Send system + user messages.
//!     let resp = client.send_message(vec![
//!         Message { role: Role::System,    content: "You are an assistant.".into() },
//!         Message { role: Role::User,      content: "Hello!".into() },
//!     ], None).await.unwrap();
//!     println!("Assistant: {}", resp.content);
//!
//!     // Then pull the real token usage.
//!     if let Some(usage) = client.get_last_usage() {
//!         println!(
//!             "Tokens — input: {}, output: {}, total: {}",
//!             usage.input_tokens, usage.output_tokens, usage.total_tokens
//!         );
//!     }
//! }
//! ```
//!
//! # Note
//!
//! Make sure `OPENAI_API_KEY` is set and pick a valid model name (e.g. `"gpt-4.1-nano"`).

use std::env;
use std::error::Error;

use async_trait::async_trait;
use log::{error, info};
use openai_rust::chat;
use openai_rust2 as openai_rust;

use crate::client_wrapper::TokenUsage;
use crate::clients::common::send_and_track;
use crate::cloudllm::client_wrapper::{ClientWrapper, Message, Role};
use std::sync::Mutex;
use tokio::runtime::Runtime;
use crate::clients::openai::Model::GPT5Nano;
use crate::LLMSession;

pub enum Model {
    GPT5,            // Higher Reasoning, Medium speed, Text+Image input, Text output; input $1.25/1M tokens, cached input $0.125/1M tokens, output $10/1M tokens
    GPT5Mini,        // High Reasoning, Fast speed, Text+Image input, Text output; input $0.25/1M tokens, cached input $0.025/1M tokens, output $2/1M tokens
    GPT5Nano,        // Average Reasoning, Very fast speed, Text+Image input, Text output; input $0.05/1M tokens, cached input $0.005/1M tokens, output $0.4/1M tokens
    GPT5ChatLatest,  // High Reasoning, Medium speed, Text+Image input, Text output; used in ChatGPT, input $1.25/1M tokens, cached input $0.125/1M tokens, output $10/1M tokens
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
        Model::GPT5 => "gpt-5".to_string(),
        Model::GPT5Mini => "gpt-5-mini".to_string(),
        Model::GPT5Nano => "gpt-5-nano".to_string(),
        Model::GPT5ChatLatest => "gpt-5-chat-latest".to_string(),
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
    token_usage: Mutex<Option<TokenUsage>>,
}

impl OpenAIClient {
    pub fn new_with_model_enum(secret_key: &str, model: Model) -> Self {
        Self::new_with_model_string(secret_key, &model_to_string(model))
    }

    pub fn new_with_model_string(secret_key: &str, model_name: &str) -> Self {
        OpenAIClient {
            client: openai_rust::Client::new(secret_key),
            model: model_name.to_string(),
            token_usage: Mutex::new(None),
        }
    }

    pub fn new_with_base_url(secret_key: &str, model_name: &str, base_url: &str) -> Self {
        OpenAIClient {
            client: openai_rust::Client::new_with_base_url(secret_key, base_url),
            model: model_name.to_string(),
            token_usage: Mutex::new(None),
        }
    }

    pub fn new_with_base_url_and_model_enum(
        secret_key: &str,
        model: Model,
        base_url: &str,
    ) -> Self {
        Self::new_with_base_url(secret_key, &model_to_string(model), base_url)
    }
}

#[async_trait]
impl ClientWrapper for OpenAIClient {
    async fn send_message(
        &self,
        messages: Vec<Message>,
        optional_search_parameters: Option<openai_rust::chat::SearchParameters>,
    ) -> Result<Message, Box<dyn Error>> {
        // Convert the provided messages into the format expected by openai_rust
        // Pre-allocate capacity to avoid reallocation
        let mut formatted_messages = Vec::with_capacity(messages.len());
        for msg in messages {
            formatted_messages.push(chat::Message {
                role: match msg.role {
                    Role::System => "system".to_owned(),
                    Role::User => "user".to_owned(),
                    Role::Assistant => "assistant".to_owned(),
                    // Extend this match as new roles are added to the Role enum
                },
                content: msg.content,
            });
        }

        let url_path_string = "/v1/chat/completions".to_string();

        let result = send_and_track(
            &self.client,
            &self.model,
            formatted_messages,
            Some(url_path_string),
            &self.token_usage,
            optional_search_parameters,
        )
        .await;

        match result {
            Ok(c) => Ok(Message {
                role: Role::Assistant,
                content: c,
            }),
            Err(_) => {
                error!(
                    "OpenAIClient::send_message(...): OpenAI API Error: {}",
                    "Error occurred while sending message"
                );
                Err("Error occurred while sending message".into())
            }
        }
    }

    fn usage_slot(&self) -> Option<&Mutex<Option<TokenUsage>>> {
        Some(&self.token_usage)
    }
}

#[test]
pub fn test_openai_client() {
    // initialize logger
    crate::init_logger();

    let secret_key = env::var("OPEN_AI_SECRET").expect("OPEN_AI_SECRET not set");
    let client = OpenAIClient::new_with_model_enum(&secret_key, GPT5Nano);
    let mut llm_session: LLMSession = LLMSession::new(
        std::sync::Arc::new(client),
        "You are a philosophy professor.".to_string(),
        1048576,
    );

    // Create a new Tokio runtime
    let rt = Runtime::new().unwrap();

    let response_message: Message = rt.block_on(async {
        let s = llm_session
            .send_message(
                Role::User,
                "If life is a game and you are not an NPC character, what can you while you play to benefit the higher consciousness of your avatar controller?"
                    .to_string(),
                None,
            )
            .await;

        match s {
            Ok(msg) => msg,
            Err(e) => {
                error!("Error: {}", e);
                Message {
                    role: Role::System,
                    content: format!("An error occurred: {:?}", e),
                }
            }
        }
    });

    info!("test_openai_client() response: {}", response_message.content);
}
