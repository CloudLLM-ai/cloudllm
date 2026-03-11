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
use cloudllm::cloudllm::image_generation::{
    decode_base64, get_image_extension_from_base64, ImageGenerationClient, ImageGenerationOptions,
};
use cloudllm::init_logger;
use cloudllm::LLMSession;
use cloudllm::Message;

fn required_env_or_skip(key: &str) -> Option<String> {
    match std::env::var(key) {
        Ok(value) => Some(value),
        Err(_) => {
            log::info!("Skipping test because {} is not set", key);
            None
        }
    }
}

fn is_skippable_external_api_error(message: &str) -> bool {
    let normalized = message.to_ascii_lowercase();
    normalized.contains("error sending request")
        || normalized.contains("connection")
        || normalized.contains("dns")
        || normalized.contains("timed out")
        || normalized.contains("timeout")
        || normalized.contains("quota")
        || normalized.contains("resource_exhausted")
        || normalized.contains("rate limit")
        || normalized.contains("429")
        || normalized.contains("502")
        || normalized.contains("503")
        || normalized.contains("temporarily unavailable")
        || normalized.contains("service unavailable")
        || normalized.contains("overloaded")
}

#[test]
fn test_claude_client() {
    // initialize logger
    init_logger();

    let Some(secret_key) = required_env_or_skip("CLAUDE_API_KEY") else {
        return;
    };
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
                tool_calls: vec![],
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

    let Some(secret_key) = required_env_or_skip("GEMINI_API_KEY") else {
        return;
    };
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
                if is_skippable_external_api_error(&e.to_string()) {
                    log::info!("Skipping Gemini smoke test due to external API issue: {}", e);
                    Message {
                        role: System,
                        content: format!("Skipped Gemini smoke test due to external API issue: {e}")
                            .into(),
                        tool_calls: vec![],
                    }
                } else {
                    panic!("test_gemini_client Error: {}", e);
                }
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

    let Some(secret_key) = required_env_or_skip("XAI_API_KEY") else {
        return;
    };
    // Use grok-4-1-fast-reasoning which supports server_tools (web_search, x_search, etc.)
    let client = GrokClient::new_with_model_enum(&secret_key, grok::Model::Grok41FastReasoning);
    let mut llm_session: crate::LLMSession = crate::LLMSession::new(
        std::sync::Arc::new(client),
        "You are a helpful assistant with access to web search and X search.".to_string(),
        1048576,
    );

    // Create a new Tokio runtime
    let rt = tokio::runtime::Runtime::new().unwrap();

    let response_message: Message = rt.block_on(async {
        let s = llm_session
            .send_message(
                crate::Role::User,
                "What's the current price of Bitcoin? Search the web for the latest information."
                    .to_string(),
                None,
            )
            .await;

        s.unwrap_or_else(|e| {
            log::error!("Error: {}", e);
            Message {
                role: crate::Role::System,
                content: format!("An error occurred: {:?}", e).into(),
                tool_calls: vec![],
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

    let Some(secret_key) = required_env_or_skip("OPEN_AI_SECRET") else {
        return;
    };
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
                tool_calls: vec![],
            }
        })
    });

    log::info!(
        "test_openai_client() response: {}",
        response_message.content
    );
}

#[test]
fn test_gemini_image_generation() {
    init_logger();

    let Some(api_key) = required_env_or_skip("GEMINI_API_KEY") else {
        return;
    };
    let client = GeminiClient::new_with_model_string(&api_key, "gemini-2.5-flash-image");

    let rt = tokio::runtime::Runtime::new().unwrap();

    let response = rt.block_on(async {
        let options = ImageGenerationOptions {
            aspect_ratio: Some("16:9".to_string()),
            num_images: Some(1),
            response_format: Some("b64_json".to_string()),
        };

        client
            .generate_image(
                "A majestic mountain landscape at sunset with vibrant orange and purple clouds",
                options,
            )
            .await
    });

    match response {
        Ok(result) => {
            assert!(!result.images.is_empty(), "Expected at least one image");

            if let Some(b64_raw) = &result.images[0].b64_json {
                // Strip data URI prefix if present (e.g. "data:image/png;base64,...")
                let b64 = if let Some(pos) = b64_raw.find(",") {
                    &b64_raw[pos + 1..]
                } else {
                    b64_raw.as_str()
                };

                // Detect extension from the data URI mime type or from the raw bytes
                let ext = if b64_raw.contains("image/png") {
                    "png"
                } else if b64_raw.contains("image/jpeg") {
                    "jpg"
                } else if b64_raw.contains("image/webp") {
                    "webp"
                } else {
                    get_image_extension_from_base64(b64)
                };

                let filename = format!("gemini_landscape_test.{}", ext);
                let decoded = decode_base64(b64).expect("Failed to decode base64 image data");
                std::fs::write(&filename, &decoded).expect("Failed to write image file");
                log::info!("Saved Gemini-generated image to: {}", filename);

                // Verify the file was created and has content
                let metadata = std::fs::metadata(&filename).expect("Image file should exist");
                assert!(metadata.len() > 0, "Image file should not be empty");
                log::info!("Image file size: {} bytes", metadata.len());
            } else {
                panic!("Expected base64 image data but got none");
            }
        }
        Err(e) => {
            if is_skippable_external_api_error(&e.to_string()) {
                log::info!(
                    "Skipping Gemini image-generation smoke test due to external API issue: {}",
                    e
                );
                return;
            }
            panic!("test_gemini_image_generation failed: {}", e);
        }
    }
}
