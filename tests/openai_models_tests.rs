use cloudllm::client_wrapper::ClientWrapper;
use cloudllm::clients::openai::{model_to_string, Model, OpenAIClient};

#[test]
fn gpt_54_model_variants_map_to_expected_api_names() {
    assert_eq!(model_to_string(Model::GPT54), "gpt-5.4");
    assert_eq!(model_to_string(Model::GPT54Mini), "gpt-5.4-mini");
    assert_eq!(model_to_string(Model::GPT54Nano), "gpt-5.4-nano");
}

#[test]
fn openai_client_uses_new_gpt_54_variants() {
    let mini_client = OpenAIClient::new_with_model_enum("test-key", Model::GPT54Mini);
    let nano_client = OpenAIClient::new_with_model_enum("test-key", Model::GPT54Nano);

    assert_eq!(mini_client.model_name(), "gpt-5.4-mini");
    assert_eq!(nano_client.model_name(), "gpt-5.4-nano");
}
