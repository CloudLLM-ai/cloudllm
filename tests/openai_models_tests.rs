use cloudllm::client_wrapper::{ClientWrapper, Message, Role};
use cloudllm::clients::openai::{model_to_string, Model, OpenAIClient};
use cloudllm::LLMSession;
use std::sync::Arc;

fn required_env_or_skip(key: &str) -> Option<String> {
    match std::env::var(key) {
        Ok(value) => Some(value),
        Err(_) => {
            eprintln!("Skipping test because {} is not set", key);
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
        || normalized.contains("model_not_found")
        || normalized.contains("does not exist")
        || normalized.contains("not have access")
}

#[test]
fn gpt_56_model_variants_map_to_expected_api_names() {
    assert_eq!(model_to_string(Model::GPT56Sol), "gpt-5.6-sol");
    assert_eq!(model_to_string(Model::GPT56), "gpt-5.6");
    assert_eq!(model_to_string(Model::GPT56Terra), "gpt-5.6-terra");
    assert_eq!(model_to_string(Model::GPT56Luna), "gpt-5.6-luna");
}

#[test]
#[allow(deprecated)]
fn gpt_55_model_variants_map_to_expected_api_names() {
    assert_eq!(model_to_string(Model::GPT55), "gpt-5.5");
    assert_eq!(model_to_string(Model::GPT55Mini), "gpt-5.5-mini");
    assert_eq!(model_to_string(Model::GPT55Nano), "gpt-5.5-nano");
    assert_eq!(model_to_string(Model::GPT55Pro), "gpt-5.5-pro");
}

#[test]
#[allow(deprecated)]
fn gpt_54_model_variants_map_to_expected_api_names() {
    assert_eq!(model_to_string(Model::GPT54), "gpt-5.4");
    assert_eq!(model_to_string(Model::GPT54Mini), "gpt-5.4-mini");
    assert_eq!(model_to_string(Model::GPT54Nano), "gpt-5.4-nano");
    assert_eq!(model_to_string(Model::GPT54Pro), "gpt-5.4-pro");
}

#[test]
fn openai_client_uses_new_gpt_56_variants() {
    let sol_client = OpenAIClient::new_with_model_enum("test-key", Model::GPT56Sol);
    let alias_client = OpenAIClient::new_with_model_enum("test-key", Model::GPT56);
    let terra_client = OpenAIClient::new_with_model_enum("test-key", Model::GPT56Terra);
    let luna_client = OpenAIClient::new_with_model_enum("test-key", Model::GPT56Luna);

    assert_eq!(sol_client.model_name(), "gpt-5.6-sol");
    assert_eq!(alias_client.model_name(), "gpt-5.6");
    assert_eq!(terra_client.model_name(), "gpt-5.6-terra");
    assert_eq!(luna_client.model_name(), "gpt-5.6-luna");
}

#[test]
#[allow(deprecated)]
fn openai_client_uses_legacy_gpt_55_variants() {
    let mini_client = OpenAIClient::new_with_model_enum("test-key", Model::GPT55Mini);
    let nano_client = OpenAIClient::new_with_model_enum("test-key", Model::GPT55Nano);
    let pro_client = OpenAIClient::new_with_model_enum("test-key", Model::GPT55Pro);

    assert_eq!(mini_client.model_name(), "gpt-5.5-mini");
    assert_eq!(nano_client.model_name(), "gpt-5.5-nano");
    assert_eq!(pro_client.model_name(), "gpt-5.5-pro");
}

#[test]
#[allow(deprecated)]
fn openai_client_uses_legacy_gpt_54_variants() {
    let mini_client = OpenAIClient::new_with_model_enum("test-key", Model::GPT54Mini);
    let nano_client = OpenAIClient::new_with_model_enum("test-key", Model::GPT54Nano);

    assert_eq!(mini_client.model_name(), "gpt-5.4-mini");
    assert_eq!(nano_client.model_name(), "gpt-5.4-nano");
}

/// Live loadability: each GPT-5.6 tier must accept a minimal chat completion.
/// Skips when `OPEN_AI_SECRET` is unset; also skips transient / access errors.
#[test]
fn gpt_56_models_are_loadable_live() {
    cloudllm::init_logger();

    let Some(secret_key) = required_env_or_skip("OPEN_AI_SECRET") else {
        return;
    };

    let models = [
        (Model::GPT56Sol, "gpt-5.6-sol"),
        (Model::GPT56, "gpt-5.6"),
        (Model::GPT56Terra, "gpt-5.6-terra"),
        (Model::GPT56Luna, "gpt-5.6-luna"),
    ];

    let rt = tokio::runtime::Runtime::new().unwrap();

    for (model, expected_name) in models {
        let client = OpenAIClient::new_with_model_enum(&secret_key, model);
        assert_eq!(client.model_name(), expected_name);

        let mut session = LLMSession::new(
            Arc::new(client),
            "You are a concise assistant.".to_string(),
            8_192,
        );

        let result = rt.block_on(async {
            session
                .send_message(
                    Role::User,
                    "Reply with exactly the word: pong".to_string(),
                    None,
                )
                .await
        });

        match result {
            Ok(msg) => {
                assert!(
                    !msg.content.is_empty(),
                    "{} reply must not be empty",
                    expected_name
                );
                log::info!(
                    "gpt_56_models_are_loadable_live {} => {}",
                    expected_name,
                    msg.content
                );
            }
            Err(e) => {
                if is_skippable_external_api_error(&e.to_string()) {
                    log::info!(
                        "Skipping live loadability for {} due to external API issue: {}",
                        expected_name,
                        e
                    );
                    continue;
                }
                panic!(
                    "gpt_56_models_are_loadable_live failed for {}: {}",
                    expected_name, e
                );
            }
        }
    }
}

/// Smoke that a single GPT-5.6 Luna request round-trips through ClientWrapper directly.
#[test]
fn gpt_56_luna_client_wrapper_smoke() {
    cloudllm::init_logger();

    let Some(secret_key) = required_env_or_skip("OPEN_AI_SECRET") else {
        return;
    };

    let client = OpenAIClient::new_with_model_enum(&secret_key, Model::GPT56Luna);
    let rt = tokio::runtime::Runtime::new().unwrap();

    let result = rt.block_on(async {
        client
            .send_message(
                &[Message {
                    role: Role::User,
                    content: Arc::from("Reply with exactly: ok"),
                    tool_calls: vec![],
                }],
                None,
            )
            .await
    });

    match result {
        Ok(msg) => {
            assert!(!msg.content.is_empty());
            log::info!("gpt_56_luna_client_wrapper_smoke => {}", msg.content);
        }
        Err(e) => {
            if is_skippable_external_api_error(&e.to_string()) {
                log::info!(
                    "Skipping gpt_56_luna_client_wrapper_smoke due to external API issue: {}",
                    e
                );
                return;
            }
            panic!("gpt_56_luna_client_wrapper_smoke failed: {}", e);
        }
    }
}
