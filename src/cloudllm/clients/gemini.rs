use crate::clients::openai::OpenAIClient;
use crate::{ClientWrapper, LLMSession, Message, Role};
use async_trait::async_trait;
use log::{error, info};
use std::env;
use std::error::Error;
use tokio::runtime::Runtime;

pub struct GeminiClient {
    client: OpenAIClient,
    model: String,
}

// Models returned by the API as of feb.23.2025
pub enum Model {
    Gemini20Flash,
    Gemini20FlashExp,
    Gemini20Flash001,
    Gemini20FlashLite001,
    Gemini20FlashThinking001,
    Gemini20FlashLitePreview,
    Gemini20FlashLitePreview0205,
    Gemini20ProExp,
    Gemini20ProExp0205,
    ChatBison001,
    TextBison001,
    TextBisonSafetyRecitationOff,
    TextBisonSafetyOff,
    TextBisonRecitationOff,
    EmbeddingGecko001,
    EvergreenCustom,
    Gemini10ProLatest,
    Gemini10Pro,
    GeminiPro,
    Gemini10Pro001,
    Gemini10ProVisionLatest,
    GeminiProVision,
    Gemini15ProLatest,
    Gemini15Pro001,
    Gemini15Pro002,
    Gemini15Pro,
    Gemini15FlashLatest,
    Gemini15Flash001,
    Gemini15Flash001Tuning,
    Gemini15Flash,
    Gemini15Flash002,
    Gemini15FlashDarkLaunch,
    GeminiTest23,
    Gemini15Flash8b,
    Gemini15Flash8b001,
    Gemini15Flash8bLatest,
    Gemini15Flash8bExp0827,
    Gemini15Flash8bExp0924,
    GeminiExp1206,
    GeminiToolTest,
    Gemini20FlashThinkingExp0121,
    Gemini20FlashThinkingExp,
    Gemini20FlashThinkingExp1219,
    Gemini20FlashThinkingExpNoThoughts,
    Gemini20Flash001BidiTest,
    Gemini20FlashAudiogenRev17,
    Gemini20FlashMmgenRev17,
    Gemini20FlashJarvis,
    Learnlm15ProExperimental,
    Embedding001,
    TextEmbedding004,
    Aqa,
    Imagen30Generate002,
    Imagen30Generate002Exp,
    ImageVerification001,
    Veo20Generate001,
}

pub fn model_to_string(model: Model) -> String {
    match model {
        Model::Gemini20Flash => "gemini-2.0-flash".to_string(),
        Model::Gemini20FlashExp => "gemini-2.0-flash-exp".to_string(),
        Model::Gemini20Flash001 => "gemini-2.0-flash-001".to_string(),
        Model::Gemini20FlashLite001 => "gemini-2.0-flash-lite-001".to_string(),
        Model::Gemini20FlashThinking001 => "gemini-2.0-flash-thinking-001".to_string(),
        Model::Gemini20FlashLitePreview => "gemini-2.0-flash-lite-preview".to_string(),
        Model::Gemini20FlashLitePreview0205 => "gemini-2.0-flash-lite-preview-02-05".to_string(),
        Model::Gemini20ProExp => "gemini-2.0-pro-exp".to_string(),
        Model::Gemini20ProExp0205 => "gemini-2.0-pro-exp-02-05".to_string(),
        Model::ChatBison001 => "chat-bison-001".to_string(),
        Model::TextBison001 => "text-bison-001".to_string(),
        Model::TextBisonSafetyRecitationOff => "text-bison-safety-recitation-off".to_string(),
        Model::TextBisonSafetyOff => "text-bison-safety-off".to_string(),
        Model::TextBisonRecitationOff => "text-bison-recitation-off".to_string(),
        Model::EmbeddingGecko001 => "embedding-gecko-001".to_string(),
        Model::EvergreenCustom => "evergreen-custom".to_string(),
        Model::Gemini10ProLatest => "gemini-1.0-pro-latest".to_string(),
        Model::Gemini10Pro => "gemini-1.0-pro".to_string(),
        Model::GeminiPro => "gemini-pro".to_string(),
        Model::Gemini10Pro001 => "gemini-1.0-pro-001".to_string(),
        Model::Gemini10ProVisionLatest => "gemini-1.0-pro-vision-latest".to_string(),
        Model::GeminiProVision => "gemini-pro-vision".to_string(),
        Model::Gemini15ProLatest => "gemini-1.5-pro-latest".to_string(),
        Model::Gemini15Pro001 => "gemini-1.5-pro-001".to_string(),
        Model::Gemini15Pro002 => "gemini-1.5-pro-002".to_string(),
        Model::Gemini15Pro => "gemini-1.5-pro".to_string(),
        Model::Gemini15FlashLatest => "gemini-1.5-flash-latest".to_string(),
        Model::Gemini15Flash001 => "gemini-1.5-flash-001".to_string(),
        Model::Gemini15Flash001Tuning => "gemini-1.5-flash-001-tuning".to_string(),
        Model::Gemini15Flash => "gemini-1.5-flash".to_string(),
        Model::Gemini15Flash002 => "gemini-1.5-flash-002".to_string(),
        Model::Gemini15FlashDarkLaunch => "gemini-1.5-flash-dark-launch".to_string(),
        Model::GeminiTest23 => "gemini-test-23".to_string(),
        Model::Gemini15Flash8b => "gemini-1.5-flash-8b".to_string(),
        Model::Gemini15Flash8b001 => "gemini-1.5-flash-8b-001".to_string(),
        Model::Gemini15Flash8bLatest => "gemini-1.5-flash-8b-latest".to_string(),
        Model::Gemini15Flash8bExp0827 => "gemini-1.5-flash-8b-exp-0827".to_string(),
        Model::Gemini15Flash8bExp0924 => "gemini-1.5-flash-8b-exp-0924".to_string(),
        Model::GeminiExp1206 => "gemini-exp-1206".to_string(),
        Model::GeminiToolTest => "gemini-tool-test".to_string(),
        Model::Gemini20FlashThinkingExp0121 => "gemini-2.0-flash-thinking-exp-01-21".to_string(),
        Model::Gemini20FlashThinkingExp => "gemini-2.0-flash-thinking-exp".to_string(),
        Model::Gemini20FlashThinkingExp1219 => "gemini-2.0-flash-thinking-exp-1219".to_string(),
        Model::Gemini20FlashThinkingExpNoThoughts => {
            "gemini-2.0-flash-thinking-exp-no-thoughts".to_string()
        }
        Model::Gemini20Flash001BidiTest => "gemini-2.0-flash-001-bidi-test".to_string(),
        Model::Gemini20FlashAudiogenRev17 => "gemini-2.0-flash-audiogen-rev17".to_string(),
        Model::Gemini20FlashMmgenRev17 => "gemini-2.0-flash-mmgen-rev17".to_string(),
        Model::Gemini20FlashJarvis => "gemini-2.0-flash-jarvis".to_string(),
        Model::Learnlm15ProExperimental => "learnlm-1.5-pro-experimental".to_string(),
        Model::Embedding001 => "embedding-001".to_string(),
        Model::TextEmbedding004 => "text-embedding-004".to_string(),
        Model::Aqa => "aqa".to_string(),
        Model::Imagen30Generate002 => "imagen-3.0-generate-002".to_string(),
        Model::Imagen30Generate002Exp => "imagen-3.0-generate-002-exp".to_string(),
        Model::ImageVerification001 => "image-verification-001".to_string(),
        Model::Veo20Generate001 => "veo-2.0-generate-001".to_string(),
    }
}

impl GeminiClient {
    pub fn new_with_model_enum(secret_key: &str, model: Model) -> Self {
        Self::new_with_model_str(secret_key, &model_to_string(model))
    }

    pub fn new_with_model_str(secret_key: &str, model_name: &str) -> Self {
        GeminiClient {
            client: OpenAIClient::new_with_base_url(
                secret_key,
                model_name,
                "https://generativelanguage.googleapis.com/v1beta/",
            ),
            model: model_name.to_string(),
        }
    }
}

#[async_trait]
impl ClientWrapper for GeminiClient {
    async fn send_message(
        &self,
        messages: Vec<Message>,
        opt_url_path: Option<String>,
    ) -> Result<Message, Box<dyn Error>> {
        self.client.send_message(messages, opt_url_path).await
    }
}

#[test]
pub fn test_gemini_client() {
    // initialize logger
    env_logger::try_init().expect("");

    let secret_key = env::var("GEMINI_API_KEY").expect("GEMINI_API_KEY not set");
    let client = GeminiClient::new_with_model_enum(&secret_key, Model::Gemini20Flash);
    assert_eq!(client.model, "gemini-2.0-flash");

    let mut llm_session: LLMSession<GeminiClient> =
        LLMSession::new(client, "You are a math professor.".to_string(), 1048576);

    // Create a new Tokio runtime
    let rt = Runtime::new().unwrap();

    let response_message: Message = rt.block_on(async {
        let s = llm_session
            .send_message(
                Role::User,
                "What is the square root of 16?".to_string(),
                Some("/v1beta/chat/completions".to_string()),
            )
            .await;

        match s {
            Ok(msg) => msg,
            Err(e) => {
                panic!("test_gemini_client Error: {}", e);
            }
        }
    });

    info!(
        "test_gemini_client() response: {}",
        response_message.content
    );
}
