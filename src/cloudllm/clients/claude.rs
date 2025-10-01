use crate::client_wrapper::TokenUsage;
use crate::clients::openai::OpenAIClient;
use crate::{ClientWrapper, Message};
use async_trait::async_trait;
use openai_rust2 as openai_rust;
use std::error::Error;
use tokio::sync::Mutex;

pub struct ClaudeClient {
    delegate_client: OpenAIClient,
    model: String,
    token_usage: Mutex<Option<TokenUsage>>,
}

// Models available in Claude API as of jan.2025
pub enum Model {
    ClaudeOpus41,
    ClaudeOpus4,
    ClaudeSonnet4,
    ClaudeSonnet37,
    ClaudeHaiku35,
}

fn model_to_string(model: Model) -> String {
    match model {
        Model::ClaudeOpus41 => "claude-opus-4-1".to_string(),
        Model::ClaudeOpus4 => "claude-opus-4-0".to_string(),
        Model::ClaudeSonnet4 => "claude-sonnet-4-0".to_string(),
        Model::ClaudeSonnet37 => "claude-sonnet-3-7-sonnet-latest".to_string(),
        Model::ClaudeHaiku35 => "claude-haiku-3-5-haiku-latest".to_string(),
    }
}

impl ClaudeClient {
    pub fn new_with_model_enum(secret_key: &str, model: Model) -> Self {
        Self::new_with_model_str(secret_key, &model_to_string(model))
    }

    pub fn new_with_model_str(secret_key: &str, model_name: &str) -> Self {
        ClaudeClient {
            // we reuse the OpenAIClient for Claude and delegate the calls to it
            delegate_client: OpenAIClient::new_with_base_url(
                secret_key,
                model_name,
                "https://api.anthropic.com/v1",
            ),
            model: model_name.to_string(),
            token_usage: Mutex::new(None),
        }
    }

    pub fn new_with_base_url(secret_key: &str, model_name: &str, base_url: &str) -> Self {
        ClaudeClient {
            delegate_client: OpenAIClient::new_with_base_url(secret_key, model_name, base_url),
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
impl ClientWrapper for ClaudeClient {
    async fn send_message(
        &self,
        messages: &[Message],
        optional_search_parameters: Option<openai_rust::chat::SearchParameters>,
    ) -> Result<Message, Box<dyn Error>> {
        self.delegate_client
            .send_message(messages, optional_search_parameters)
            .await
    }

    fn usage_slot(&self) -> Option<&Mutex<Option<TokenUsage>>> {
        self.delegate_client.usage_slot()
    }
}