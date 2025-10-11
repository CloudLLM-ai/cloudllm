use crate::client_wrapper::TokenUsage;
use crate::clients::openai::OpenAIClient;
use crate::{ClientWrapper, Message};
use async_trait::async_trait;
use openai_rust2 as openai_rust;
use std::error::Error;
use tokio::sync::Mutex;

pub struct GrokClient {
    delegate_client: OpenAIClient,
    model: String,
}

// Models returned by the xAI API as of apr.14.2025
pub enum Model {
    Grok2,
    Grok2Latest,
    Grok21212,             // $2/MMT input $10/MMT output
    Grok3MiniFast,         // $0.60/MMT input $4.00/MMT output
    Grok3Mini,             // $0.30/MMT input $0.50/MMT output
    Grok3Fast,             // $5/MMT input $25/MMT output
    Grok3,                 // $3/MMT input $15/MMT output
    Grok4_0709,            // $3/MMT input $15/MMT output
    Grok4FastReasoning,    // #$0.2/MMT input $0.50/MMT output
    Grok4FastNonReasoning, // #$0.2/MMT input $0.50/MMT output
    GrokCodeFast1,         // #$0.2/MMT input $1.50/MMT output
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
        Model::GrokCodeFast1 => "grok-code-fast-1".to_string(),
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
        }
    }

    pub fn new_with_base_url(secret_key: &str, model_name: &str, base_url: &str) -> Self {
        GrokClient {
            delegate_client: OpenAIClient::new_with_base_url(secret_key, model_name, base_url),
            model: model_name.to_string(),
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
    fn model_name(&self) -> &str {
        &self.model
    }

    async fn send_message(
        &self,
        messages: &[Message],
        optional_search_parameters: Option<openai_rust::chat::SearchParameters>,
    ) -> Result<Message, Box<dyn Error>> {
        self.delegate_client
            .send_message(messages, optional_search_parameters)
            .await
    }

    fn send_message_stream<'a>(
        &'a self,
        messages: &'a [Message],
        optional_search_parameters: Option<openai_rust::chat::SearchParameters>,
    ) -> crate::client_wrapper::MessageStreamFuture<'a> {
        self.delegate_client
            .send_message_stream(messages, optional_search_parameters)
    }

    fn usage_slot(&self) -> Option<&Mutex<Option<TokenUsage>>> {
        self.delegate_client.usage_slot()
    }
}
