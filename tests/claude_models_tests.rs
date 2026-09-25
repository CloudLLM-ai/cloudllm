use cloudllm::client_wrapper::ClientWrapper;
use cloudllm::clients::claude::{ClaudeClient, Model};

/// Verifies the latest Anthropic model variants select their official API IDs.
#[test]
fn latest_claude_models_use_official_api_ids() {
    let opus = ClaudeClient::new_with_model_enum("test-key", Model::ClaudeOpus55);
    let sonnet = ClaudeClient::new_with_model_enum("test-key", Model::ClaudeSonnet5);
    let fable = ClaudeClient::new_with_model_enum("test-key", Model::ClaudeFable51);

    assert_eq!(opus.model_name(), "claude-opus-5-5");
    assert_eq!(sonnet.model_name(), "claude-sonnet-5");
    assert_eq!(fable.model_name(), "claude-fable-5-1");
}
