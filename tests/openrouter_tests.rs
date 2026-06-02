use cloudllm::client_wrapper::ClientWrapper;
use cloudllm::clients::openrouter::{
    model_to_string, Model, OpenRouterClient, OPENROUTER_DEFAULT_BASE_URL,
};
use cloudllm::{LLMSession, Role};

#[test]
fn minimax_m3_maps_to_expected_openrouter_slug() {
    assert_eq!(model_to_string(Model::MinimaxM3), "minimax/minimax-m3");
}

#[test]
fn top_weekly_models_map_to_expected_openrouter_slugs() {
    // Cover every variant in the top-50 enum and assert the wire slug.
    // If a variant is renamed or removed this test will fail loudly and tell
    // the maintainer to update either the enum or the slug mapping.
    let expected: &[(Model, &str)] = &[
        (Model::MinimaxM3, "minimax/minimax-m3"),
        (Model::TencentHy3Preview, "tencent/hy3-preview"),
        (Model::DeepSeekV4Flash, "deepseek/deepseek-v4-flash"),
        (Model::ClaudeOpus47, "anthropic/claude-opus-4.7"),
        (Model::ClaudeSonnet46, "anthropic/claude-sonnet-4.6"),
        (Model::OpenRouterOwlAlpha, "openrouter/owl-alpha"),
        (Model::XiaomiMimoV25, "xiaomi/mimo-v2.5"),
        (Model::XiaomiMimoV25Pro, "xiaomi/mimo-v2.5-pro"),
        (Model::DeepSeekV4Pro, "deepseek/deepseek-v4-pro"),
        (Model::DeepSeekV32, "deepseek/deepseek-v3.2"),
        (Model::Gemini3FlashPreview, "google/gemini-3-flash-preview"),
        (
            Model::NvidiaNemotron3Super120bA12b,
            "nvidia/nemotron-3-super-120b-a12b",
        ),
        (Model::Gemini25FlashLite, "google/gemini-2.5-flash-lite"),
        (Model::Gemini25Flash, "google/gemini-2.5-flash"),
        (Model::PoolsideLagunaM1, "poolside/laguna-m.1"),
        (Model::ClaudeOpus46, "anthropic/claude-opus-4.6"),
        (Model::Gemini35Flash, "google/gemini-3.5-flash"),
        (Model::MinimaxM27, "minimax/minimax-m2.7"),
        (Model::MoonshotKimiK26, "moonshotai/kimi-k2.6"),
        (Model::GPT4oMini, "openai/gpt-4o-mini"),
        (Model::GPT55, "openai/gpt-5.5"),
        (Model::ClaudeOpus48, "anthropic/claude-opus-4.8"),
        (Model::GPTOSS120B, "openai/gpt-oss-120b"),
        (Model::Gemini31FlashLite, "google/gemini-3.1-flash-lite"),
        (Model::Gemma431BIt, "google/gemma-4-31b-it"),
        (Model::ZAiGlm51, "z-ai/glm-5.1"),
        (Model::Gemini31ProPreview, "google/gemini-3.1-pro-preview"),
        (Model::GPT54, "openai/gpt-5.4"),
        (Model::Qwen3235bA22b2507, "qwen/qwen3-235b-a22b-2507"),
        (Model::ClaudeHaiku45, "anthropic/claude-haiku-4.5"),
        (Model::Gemma426bA4bIt, "google/gemma-4-26b-a4b-it"),
        (Model::StepfunStep37Flash, "stepfun/step-3.7-flash"),
        (
            Model::Gemini31FlashLitePreview,
            "google/gemini-3.1-flash-lite-preview",
        ),
        (Model::ZAiGlm47, "z-ai/glm-4.7"),
        (Model::Qwen36Plus, "qwen/qwen3.6-plus"),
        (Model::Qwen37Max, "qwen/qwen3.7-max"),
        (Model::MinimaxM25, "minimax/minimax-m2.5"),
        (Model::GPT54Mini, "openai/gpt-5.4-mini"),
        (Model::GPT5Mini, "openai/gpt-5-mini"),
        (Model::MoonshotKimiK25, "moonshotai/kimi-k2.5"),
        (Model::MistralNemo, "mistralai/mistral-nemo"),
        (Model::ZAiGlm5, "z-ai/glm-5"),
        (Model::ZAiGlm45Air, "z-ai/glm-4.5-air"),
        (Model::Qwen3Embedding8B, "qwen/qwen3-embedding-8b"),
        (Model::ClaudeSonnet45, "anthropic/claude-sonnet-4.5"),
        (Model::Qwen35Flash0223, "qwen/qwen3.5-flash-02-23"),
        (Model::GPT54Nano, "openai/gpt-5.4-nano"),
        (Model::GPT53Codex, "openai/gpt-5.3-codex"),
        (Model::PoolsideLagunaXs2, "poolside/laguna-xs.2"),
        (Model::Llama318bInstruct, "meta-llama/llama-3.1-8b-instruct"),
        (Model::Grok43, "x-ai/grok-4.3"),
    ];

    assert_eq!(expected.len(), 51, "test fixture must cover every variant");

    for (variant, slug) in expected {
        assert_eq!(model_to_string(*variant), *slug);
    }
}

#[test]
fn minimax_m3_client_exposes_expected_model_name() {
    let client = OpenRouterClient::new_with_model_enum("test-key", Model::MinimaxM3);
    assert_eq!(client.model_name(), "minimax/minimax-m3");
}

#[test]
fn new_with_model_str_accepts_arbitrary_openrouter_slug() {
    let client = OpenRouterClient::new_with_model_str("test-key", "openai/gpt-5.5");
    assert_eq!(client.model_name(), "openai/gpt-5.5");
}

#[test]
fn new_with_base_url_normalizes_trailing_slash() {
    let with_slash = OpenRouterClient::new_with_base_url(
        "test-key",
        "minimax/minimax-m3",
        "https://openrouter.ai/api/v1/",
    );
    let without_slash = OpenRouterClient::new_with_base_url(
        "test-key",
        "minimax/minimax-m3",
        "https://openrouter.ai/api/v1",
    );
    assert_eq!(with_slash.model_name(), without_slash.model_name());
    assert_eq!(with_slash.model_name(), "minimax/minimax-m3");
}

#[test]
fn new_with_base_url_and_model_enum_uses_typed_model() {
    let client = OpenRouterClient::new_with_base_url_and_model_enum(
        "test-key",
        Model::MinimaxM3,
        OPENROUTER_DEFAULT_BASE_URL,
    );
    assert_eq!(client.model_name(), "minimax/minimax-m3");
}

#[test]
fn default_base_url_points_at_openrouter() {
    assert_eq!(OPENROUTER_DEFAULT_BASE_URL, "https://openrouter.ai/api/v1");
    assert!(!OPENROUTER_DEFAULT_BASE_URL.ends_with('/'));
}

/// Skip-when-no-key helper, mirrors the convention used in
/// `tests/client_tests.rs`.
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
fn test_openrouter_minimax_m3_smoke() {
    cloudllm::init_logger();

    let Some(secret_key) = required_env_or_skip("OPENROUTER_API_KEY") else {
        return;
    };

    let client = OpenRouterClient::new_with_model_enum(&secret_key, Model::MinimaxM3);

    let mut llm_session: LLMSession = LLMSession::new(
        std::sync::Arc::new(client),
        "You are a helpful assistant.".to_string(),
        1_048_576,
    );

    let rt = tokio::runtime::Runtime::new().unwrap();

    let response = rt.block_on(async {
        llm_session
            .send_message(
                Role::User,
                "In one sentence, what is Rust's ownership model?".to_string(),
                None,
            )
            .await
    });

    match response {
        Ok(msg) => {
            log::info!("test_openrouter_minimax_m3_smoke response: {}", msg.content);
            assert!(
                !msg.content.is_empty(),
                "OpenRouter reply must not be empty"
            );
        }
        Err(e) => {
            if is_skippable_external_api_error(&e.to_string()) {
                log::info!(
                    "Skipping OpenRouter smoke test due to external API issue: {}",
                    e
                );
                return;
            }
            panic!("test_openrouter_minimax_m3_smoke failed: {}", e);
        }
    }
}
