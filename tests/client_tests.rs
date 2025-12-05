use cloudllm::clients::claude;
use cloudllm::clients::claude::ClaudeClient;
use cloudllm::clients::gemini;
use cloudllm::clients::gemini::GeminiClient;
use cloudllm::clients::grok;
use cloudllm::clients::grok::GrokClient;
use cloudllm::clients::openai;
use cloudllm::clients::openai::OpenAIClient;
use cloudllm::cloudllm::client_wrapper::Role;
use cloudllm::cloudllm::client_wrapper::Role::System;
use cloudllm::init_logger;
use cloudllm::LLMSession;
use cloudllm::Message;
use openai_rust2::chat::GrokTool;

#[test]
fn test_claude_client() {
    // initialize logger
    init_logger();

    let secret_key = std::env::var("CLAUDE_API_KEY").expect("CLAUDE_API_KEY not set");
    let client = ClaudeClient::new_with_model_enum(&secret_key, claude::Model::ClaudeSonnet4);
    let mut llm_session: crate::LLMSession = crate::LLMSession::new(
        std::sync::Arc::new(client),
        "You are a helpful assistant.".to_string(),
        1048576,
    );

    // Create a new Tokio runtime
    let rt = tokio::runtime::Runtime::new().unwrap();

    let response_message: Message = rt.block_on(async {
        let s = llm_session
            .send_message(
                crate::Role::User,
                "What is the capital of France?".to_string(),
                None,
            )
            .await;

        s.unwrap_or_else(|e| {
            log::error!("Error: {}", e);
            Message {
                role: System,
                content: format!("An error occurred: {:?}", e).into(),
            }
        })
    });

    log::info!(
        "test_claude_client() response: {}",
        response_message.content
    );
}

#[test]
fn test_gemini_client() {
    // initialize logger
    crate::init_logger();

    let secret_key = std::env::var("GEMINI_API_KEY").expect("GEMINI_API_KEY not set");
    let client = GeminiClient::new_with_model_enum(&secret_key, gemini::Model::Gemini20Flash);
    assert_eq!(client.model, "gemini-2.0-flash");

    let mut llm_session: crate::LLMSession = crate::LLMSession::new(
        std::sync::Arc::new(client),
        "You are a math professor.".to_string(),
        1048576,
    );

    // Create a new Tokio runtime
    let rt = tokio::runtime::Runtime::new().unwrap();

    let response_message: Message = rt.block_on(async {
        let s = llm_session
            .send_message(
                Role::User,
                "What is the square root of 16?".to_string(),
                None,
            )
            .await;

        match s {
            Ok(msg) => msg,
            Err(e) => {
                panic!("test_gemini_client Error: {}", e);
            }
        }
    });

    log::info!(
        "test_gemini_client() response: {}",
        response_message.content
    );
}

#[test]
pub fn test_grok_client() {
    // initialize logger
    crate::init_logger();

    let secret_key = std::env::var("XAI_API_KEY").expect("XAI_API_KEY not set");
    // Use grok-4-1-fast-reasoning which supports server_tools (web_search, x_search, etc.)
    let client = GrokClient::new_with_model_enum(&secret_key, grok::Model::Grok41FastReasoning);
    let mut llm_session: crate::LLMSession = crate::LLMSession::new(
        std::sync::Arc::new(client),
        "You are a helpful assistant with access to web search and X search.".to_string(),
        1048576,
    );

    // Create a new Tokio runtime
    let rt = tokio::runtime::Runtime::new().unwrap();

    // Use the new xAI Agent Tools API (replaces deprecated Live Search)
    // See: https://docs.x.ai/docs/guides/tools/overview
    let grok_tools = vec![
        GrokTool::web_search(),
        GrokTool::x_search(),
    ];

    let response_message: Message = rt.block_on(async {
        let s = llm_session
            .send_message(
                crate::Role::User,
                "What's the current price of Bitcoin? Search the web for the latest information."
                    .to_string(),
                Some(grok_tools),
            )
            .await;

        s.unwrap_or_else(|e| {
            log::error!("Error: {}", e);
            Message {
                role: crate::Role::System,
                content: format!("An error occurred: {:?}", e).into(),
            }
        })
    });

    log::info!("test_grok_client() response: {}", response_message.content);
}

#[cfg(test)]
#[test]
fn test_openai_client() {
    // initialize logger
    crate::init_logger();

    let secret_key = std::env::var("OPEN_AI_SECRET").expect("OPEN_AI_SECRET not set");
    let client = OpenAIClient::new_with_model_enum(&secret_key, openai::Model::GPT5Nano);
    let mut llm_session: crate::LLMSession = crate::LLMSession::new(
        std::sync::Arc::new(client),
        "You are a philosophy professor.".to_string(),
        1048576,
    );

    // Create a new Tokio runtime
    let rt = tokio::runtime::Runtime::new().unwrap();

    let response_message: Message = rt.block_on(async {
        let s = llm_session
            .send_message(
                Role::User,
                "If life is a game and you are not an NPC character, what can you while you play to benefit the higher consciousness of your avatar controller?"
                    .to_string(),
                None,
            )
            .await;

        s.unwrap_or_else(|e| {
            log::error!("Error: {}", e);
            Message {
                role: Role::System,
                content: format!("An error occurred: {:?}", e).into(),
            }
        })
    });

    log::info!(
        "test_openai_client() response: {}",
        response_message.content
    );
}
