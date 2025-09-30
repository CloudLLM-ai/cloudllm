use crate::client_wrapper::TokenUsage;
use crate::clients::openai::OpenAIClient;
use crate::{ClientWrapper, Message};
use async_trait::async_trait;
use openai_rust2 as openai_rust;
use std::error::Error;
use std::sync::Mutex;

#[cfg(test)]
use {
    std::env,
    tokio::runtime::Runtime,
    crate::LLMSession,
    crate::Role,
    crate::clients::grok::Model::Grok4_0709,
    openai_rust2::chat::SearchMode,
    log::{error, info},
};

pub struct GrokClient {
    delegate_client: OpenAIClient,
    model: String,
    token_usage: Mutex<Option<TokenUsage>>,
}

// Models returned by the xAI API as of apr.14.2025
pub enum Model {
    Grok2,
    Grok2Latest,
    Grok21212,         // $2/MMT input $10/MMT output
    Grok3MiniFast, // $0.60/MMT input $4.00/MMT output
    Grok3Mini,     // $0.30/MMT input $0.50/MMT output
    Grok3Fast,     // $5/MMT input $25/MMT output
    Grok3,         // $3/MMT input $15/MMT output
    Grok4_0709,       // $3/MMT input $15/MMT output
    Grok4FastReasoning, // #$0.2/MMT input $0.50/MMT output
    Grok4FastNonReasoning, // #$0.2/MMT input $0.50/MMT output
    GrokCodeFast1, // #$0.2/MMT input $1.50/MMT output
}

fn model_to_string(model: Model) -> String {
    match model {
        Model::Grok2 => "grok-2".to_string(),
        Model::Grok2Latest => "grok-2-latest".to_string(),
        Model::Grok21212 => "grok-2-1212".to_string(),
        Model::Grok3MiniFast => "grok-3-mini-fast".to_string(),
        Model::Grok3Mini => "grok-3-mini".to_string(), // cheapest model
        Model::Grok3Fast => "grok-3-fast".to_string(),
        Model::Grok3 => "grok-3".to_string(),
        Model::Grok4_0709 => "grok-4-0709".to_string(),
        Model::Grok4FastReasoning => "grok-4-fast-reasoning".to_string(),
        Model::Grok4FastNonReasoning => "grok-4-fast-nonreasoning".to_string(),
        Model::GrokCodeFast1 => "grok-code-fast-1".to_string()
    }
}

impl GrokClient {
    pub fn new_with_model_enum(secret_key: &str, model: Model) -> Self {
        Self::new_with_model_str(secret_key, &model_to_string(model))
    }

    pub fn new_with_model_str(secret_key: &str, model_name: &str) -> Self {
        GrokClient {
            // we reuse the OpenAIClient for Grok and delegate the calls to it
            delegate_client: OpenAIClient::new_with_base_url(
                secret_key,
                model_name,
                "https://api.x.ai/v1",
            ),
            model: model_name.to_string(),
            token_usage: Mutex::new(None),
        }
    }

    pub fn new_with_base_url(secret_key: &str, model_name: &str, base_url: &str) -> Self {
        GrokClient {
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
impl ClientWrapper for GrokClient {
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
pub fn test_grok_client() {
    // initialize logger
    crate::init_logger();

    let secret_key = env::var("XAI_API_KEY").expect("XAI_API_KEY not set");
    let client = GrokClient::new_with_model_enum(&secret_key, Grok4_0709);
    let mut llm_session: LLMSession = LLMSession::new(
        std::sync::Arc::new(client),
        "You are a math professor.".to_string(),
        1048576,
    );

    // Create a new Tokio runtime
    let rt = Runtime::new().unwrap();

    let search_parameters =
        openai_rust::chat::SearchParameters::new(SearchMode::On).with_citations(true);

    let response_message: Message = rt.block_on(async {
        let s = llm_session
            .send_message(
                Role::User,
                "Using your Live search capabilities: What's the current price of Bitcoin?"
                    .to_string(),
                Some(search_parameters),
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

    info!("test_grok_client() response: {}", response_message.content);
}
