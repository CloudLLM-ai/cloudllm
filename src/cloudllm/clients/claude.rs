use crate::client_wrapper::TokenUsage;
use crate::clients::claude::Model::Claude35Sonnet20241022;
use crate::clients::openai::OpenAIClient;
use crate::{ClientWrapper, LLMSession, Message, Role};
use async_trait::async_trait;
use log::{error, info};
use openai_rust2 as openai_rust;
use openai_rust2::chat::SearchMode;
use std::env;
use std::error::Error;
use std::sync::Mutex;
use tokio::runtime::Runtime;

pub struct ClaudeClient {
    delegate_client: OpenAIClient,
    model: String,
    token_usage: Mutex<Option<TokenUsage>>,
}

// Models available in Claude API as of jan.2025
pub enum Model {
    Claude35Sonnet20241022,  // Latest Claude 3.5 Sonnet
    Claude35Haiku20241022,   // Latest Claude 3.5 Haiku
    Claude3Opus20240229,     // Claude 3 Opus
    Claude35Sonnet20240620,  // Previous Claude 3.5 Sonnet
    Claude3Sonnet20240229,   // Claude 3 Sonnet
    Claude3Haiku20240307,    // Claude 3 Haiku
}

fn model_to_string(model: Model) -> String {
    match model {
        Model::Claude35Sonnet20241022 => "claude-3-5-sonnet-20241022".to_string(),
        Model::Claude35Haiku20241022 => "claude-3-5-haiku-20241022".to_string(),
        Model::Claude3Opus20240229 => "claude-3-opus-20240229".to_string(),
        Model::Claude35Sonnet20240620 => "claude-3-5-sonnet-20240620".to_string(),
        Model::Claude3Sonnet20240229 => "claude-3-sonnet-20240229".to_string(),
        Model::Claude3Haiku20240307 => "claude-3-haiku-20240307".to_string(),
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
        messages: Vec<Message>,
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

#[test]
pub fn test_claude_client() {
    // initialize logger
    crate::init_logger();

    let secret_key = env::var("CLAUDE_API_KEY").expect("CLAUDE_API_KEY not set");
    let client = ClaudeClient::new_with_model_enum(&secret_key, Claude35Sonnet20241022);
    let mut llm_session: LLMSession = LLMSession::new(
        std::sync::Arc::new(client),
        "You are a helpful assistant.".to_string(),
        1048576,
    );

    // Create a new Tokio runtime
    let rt = Runtime::new().unwrap();

    let response_message: Message = rt.block_on(async {
        let s = llm_session
            .send_message(
                Role::User,
                "What is the capital of France?".to_string(),
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

    info!("test_claude_client() response: {}", response_message.content);
}