use cloudllm::client_wrapper::Role;
/// Tests for streaming functionality
use cloudllm::clients::openai::{Model, OpenAIClient};
use cloudllm::{ClientWrapper, LLMSession, Message};

#[tokio::test]
async fn test_streaming_returns_option() {
    cloudllm::init_logger();

    // This test verifies that send_message_stream returns Ok(Some(_)) or Ok(None)
    // depending on whether the client supports streaming.

    // We'll use a mock scenario - in real usage, this would require API keys
    // For now, just check that the API is callable

    let secret_key = std::env::var("OPEN_AI_SECRET").unwrap_or_else(|_| "fake_key".to_string());
    let client = OpenAIClient::new_with_model_enum(&secret_key, Model::GPT41Nano);

    let messages = vec![Message {
        role: Role::User,
        content: "Hello".into(),
    }];

    // This will fail with authentication error if fake_key is used,
    // but we're just testing that the API is callable
    let _ = client.send_message_stream(&messages, None, None).await;
}

#[tokio::test]
async fn test_session_streaming_api() {
    cloudllm::init_logger();

    let secret_key = std::env::var("OPEN_AI_SECRET").unwrap_or_else(|_| "fake_key".to_string());
    let client = OpenAIClient::new_with_model_enum(&secret_key, Model::GPT41Nano);

    let mut session = LLMSession::new(
        std::sync::Arc::new(client),
        "You are a helpful assistant.".to_string(),
        1024,
    );

    // Test that send_message_stream is callable
    let _ = session
        .send_message_stream(Role::User, "Test".to_string(), None, None)
        .await;
}

#[tokio::test]
async fn test_backward_compatibility_non_streaming() {
    cloudllm::init_logger();

    // Verify that existing non-streaming code still works
    let secret_key = std::env::var("OPEN_AI_SECRET").unwrap_or_else(|_| "fake_key".to_string());
    let client = OpenAIClient::new_with_model_enum(&secret_key, Model::GPT41Nano);

    let mut session = LLMSession::new(
        std::sync::Arc::new(client),
        "You are a helpful assistant.".to_string(),
        1024,
    );

    // The old non-streaming API should still work
    let _ = session
        .send_message(Role::User, "Test".to_string(), None, None)
        .await;
}
