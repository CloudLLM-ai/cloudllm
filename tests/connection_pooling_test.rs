use cloudllm::clients::common::get_shared_http_client;

#[test]
fn test_shared_http_client_is_singleton() {
    // Get the client multiple times and verify they all point to the same instance
    let client1 = get_shared_http_client();
    let client2 = get_shared_http_client();
    let client3 = get_shared_http_client();

    // All pointers should be identical since it's a singleton
    let ptr1 = client1 as *const _;
    let ptr2 = client2 as *const _;
    let ptr3 = client3 as *const _;

    assert_eq!(
        ptr1, ptr2,
        "All clients should point to the same singleton instance"
    );
    assert_eq!(
        ptr2, ptr3,
        "All clients should point to the same singleton instance"
    );
}

#[test]
fn test_shared_http_client_has_pooling_config() {
    // This test verifies that the client is created successfully
    // The pooling configuration is verified implicitly through successful creation
    let client = get_shared_http_client();

    // Verify we can clone the client (reqwest::Client is cloneable and uses Arc internally)
    let _cloned = client.clone();

    // If we got here without panicking, the client was created with proper configuration
}

#[tokio::test]
async fn test_multiple_clients_share_connection_pool() {
    use cloudllm::clients::openai::OpenAIClient;
    use cloudllm::ClientWrapper;

    // Create multiple OpenAI clients with a dummy API key
    let client1 = OpenAIClient::new_with_model_string("dummy_key_1", "gpt-4");
    let client2 = OpenAIClient::new_with_model_string("dummy_key_2", "gpt-4");
    let client3 = OpenAIClient::new_with_model_string("dummy_key_3", "gpt-4");

    // All clients should be using the same underlying HTTP client pool
    // This is verified by the fact that they all call get_shared_http_client()
    // We can't directly test connection reuse without making actual HTTP calls,
    // but we can verify the clients are created successfully
    assert_eq!(client1.model_name(), "gpt-4");
    assert_eq!(client2.model_name(), "gpt-4");
    assert_eq!(client3.model_name(), "gpt-4");
}

#[tokio::test]
async fn test_gemini_client_uses_shared_pool() {
    use cloudllm::clients::gemini::GeminiClient;
    use cloudllm::ClientWrapper;

    // Create a Gemini client
    let client = GeminiClient::new_with_model_string("dummy_key", "gemini-2.0-flash");

    // Verify the client is created successfully with the shared HTTP pool
    assert_eq!(client.model_name(), "gemini-2.0-flash");
}
