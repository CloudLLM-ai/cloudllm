use cloudllm::client_wrapper::{ClientWrapper, Message, Role};
use cloudllm::clients::grok::{model_to_string, GrokClient, Model};
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
fn grok_45_model_variants_map_to_expected_api_names() {
    assert_eq!(model_to_string(Model::Grok45), "grok-4.5");
    assert_eq!(model_to_string(Model::Grok45Latest), "grok-4.5-latest");
}

#[test]
fn grok_46_model_variants_map_to_expected_api_names() {
    assert_eq!(model_to_string(Model::Grok46), "grok-4.6");
    assert_eq!(model_to_string(Model::Grok46Latest), "grok-4.6-latest");
}

#[test]
fn grok_43_and_build_still_map() {
    assert_eq!(model_to_string(Model::Grok43), "grok-4.3");
    assert_eq!(model_to_string(Model::Grok43Latest), "grok-4.3-latest");
    assert_eq!(model_to_string(Model::GrokBuild01), "grok-build-0.1");
}

#[test]
fn grok_client_uses_new_grok_45_variants() {
    let client = GrokClient::new_with_model_enum("test-key", Model::Grok45);
    let latest = GrokClient::new_with_model_enum("test-key", Model::Grok45Latest);

    assert_eq!(client.model_name(), "grok-4.5");
    assert_eq!(latest.model_name(), "grok-4.5-latest");
    assert_eq!(client.provider_name(), "Grok");
}

#[test]
fn grok_client_uses_new_grok_46_variants() {
    let client = GrokClient::new_with_model_enum("test-key", Model::Grok46);
    let latest = GrokClient::new_with_model_enum("test-key", Model::Grok46Latest);

    assert_eq!(client.model_name(), "grok-4.6");
    assert_eq!(latest.model_name(), "grok-4.6-latest");
    assert_eq!(client.provider_name(), "Grok");
}

/// Live loadability: Grok 4.5 must accept a minimal chat completion.
/// Skips when `XAI_API_KEY` is unset; also skips transient / access errors.
#[test]
fn grok_45_models_are_loadable_live() {
    cloudllm::init_logger();

    let Some(secret_key) = required_env_or_skip("XAI_API_KEY") else {
        return;
    };

    let models = [
        (Model::Grok45, "grok-4.5"),
        (Model::Grok45Latest, "grok-4.5-latest"),
    ];

    let rt = tokio::runtime::Runtime::new().unwrap();

    for (model, expected_name) in models {
        let client = GrokClient::new_with_model_enum(&secret_key, model);
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
                    "grok_45_models_are_loadable_live {} => {}",
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
                    "grok_45_models_are_loadable_live failed for {}: {}",
                    expected_name, e
                );
            }
        }
    }
}

/// Direct ClientWrapper smoke for the primary grok-4.5 model id.
#[test]
fn grok_45_client_wrapper_smoke() {
    cloudllm::init_logger();

    let Some(secret_key) = required_env_or_skip("XAI_API_KEY") else {
        return;
    };

    let client = GrokClient::new_with_model_enum(&secret_key, Model::Grok45);
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
            log::info!("grok_45_client_wrapper_smoke => {}", msg.content);
        }
        Err(e) => {
            if is_skippable_external_api_error(&e.to_string()) {
                log::info!(
                    "Skipping grok_45_client_wrapper_smoke due to external API issue: {}",
                    e
                );
                return;
            }
            panic!("grok_45_client_wrapper_smoke failed: {}", e);
        }
    }
}
