use crate::clients::grok::Model::Grok3MiniFastBeta;
use crate::clients::openai::OpenAIClient;
use crate::{ClientWrapper, LLMSession, Message, Role};
use async_trait::async_trait;
use log::{error, info};
use std::env;
use std::error::Error;
use tokio::runtime::Runtime;

pub struct GrokClient {
    client: OpenAIClient,
    model: String,
}

// Models returned by the xAI API as of apr.14.2025
pub enum Model {
    Grok2,
    Grok2Latest,
    Grok21212,         // $2/MMT input $10/MMT output
    Grok3MiniFastBeta, // $0.60/MMT input $4.00/MMT output
    Grok3MiniBeta,     // $0.30/MMT input $0.50/MMT output
    Grok3FastBeta,     // $5/MMT input $25/MMT output
    Grok3Beta,         // $3/MMT input $15/MMT output
}

fn model_to_string(model: Model) -> String {
    match model {
        Model::Grok2 => "grok-2".to_string(),
        Model::Grok2Latest => "grok-2-latest".to_string(),
        Model::Grok21212 => "grok-2-1212".to_string(),
        Model::Grok3MiniFastBeta => "grok-3-mini-fast-beta".to_string(),
        Model::Grok3MiniBeta => "grok-3-mini-beta".to_string(),
        Model::Grok3FastBeta => "grok-3-fast-beta".to_string(),
        Model::Grok3Beta => "grok-3-beta".to_string(),
    }
}

impl GrokClient {
    pub fn new_with_model_enum(secret_key: &str, model: Model) -> Self {
        Self::new_with_model_str(secret_key, &model_to_string(model))
    }

    pub fn new_with_model_str(secret_key: &str, model_name: &str) -> Self {
        GrokClient {
            client: OpenAIClient::new_with_base_url(secret_key, model_name, "https://api.x.ai/v1"),
            model: model_name.to_string(),
        }
    }
}

#[async_trait]
impl ClientWrapper for GrokClient {
    async fn send_message(&self, messages: Vec<Message>) -> Result<Message, Box<dyn Error>> {
        self.client.send_message(messages).await
    }
}

#[test]
pub fn test_grok_client() {
    // initialize logger
    crate::init_logger();

    let secret_key = env::var("XAI_API_KEY").expect("XAI_API_KEY not set");
    let client = GrokClient::new_with_model_enum(&secret_key, Grok3MiniFastBeta);
    let mut llm_session: LLMSession = LLMSession::new(
        std::sync::Arc::new(client),
        "You are a math professor.".to_string(),
        1048576,
    );

    // Create a new Tokio runtime
    let rt = Runtime::new().unwrap();

    let response_message: Message = rt.block_on(async {
        let s = llm_session
            .send_message(
                Role::User,
                "What is the square root of 16? What does the square of a root mean?".to_string(),
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
