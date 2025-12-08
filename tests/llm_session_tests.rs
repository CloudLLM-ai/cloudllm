use async_trait::async_trait;
use cloudllm::client_wrapper;
use cloudllm::client_wrapper::{ClientWrapper, Message, Role, TokenUsage};
use cloudllm::cloudllm::llm_session;
use cloudllm::cloudllm::llm_session::estimate_message_token_count;
use cloudllm::LLMSession;
use openai_rust2 as openai_rust;
use std::sync::Arc;
use tokio::sync::Mutex;

// Mock client for testing
struct MockClient {
    usage: Mutex<Option<TokenUsage>>,
    response_content: String,
    last_message_count: Mutex<usize>,
}

impl MockClient {
    fn new(response_content: String) -> Self {
        Self {
            usage: Mutex::new(None),
            response_content,
            last_message_count: Mutex::new(0),
        }
    }

    async fn get_last_message_count(&self) -> usize {
        *self.last_message_count.lock().await
    }

    async fn set_usage(&self, input: usize, output: usize, total: usize) {
        let mut usage = self.usage.lock().await;
        *usage = Some(client_wrapper::TokenUsage {
            input_tokens: input,
            output_tokens: output,
            total_tokens: total,
        });
    }
}

#[async_trait]
impl ClientWrapper for MockClient {
    async fn send_message(
        &self,
        messages: &[Message],
        _optional_grok_tools: Option<Vec<openai_rust::chat::GrokTool>>,
        _optional_openai_tools: Option<Vec<openai_rust::chat::OpenAITool>>,
    ) -> Result<Message, Box<dyn std::error::Error>> {
        // Record how many messages were sent
        let mut count_guard = self.last_message_count.lock().await;
        *count_guard = messages.len();

        // Calculate token usage
        let mut input_tokens = 0;
        for msg in messages {
            input_tokens += estimate_message_token_count(msg);
        }
        let output_tokens = estimate_message_token_count(&Message {
            role: Role::Assistant,
            content: self.response_content.clone().into(),
        });

        let computed_usage = TokenUsage {
            input_tokens,
            output_tokens,
            total_tokens: input_tokens + output_tokens,
        };

        let mut usage_guard = self.usage.lock().await;
        if usage_guard.is_none() {
            *usage_guard = Some(computed_usage);
        }

        Ok(Message {
            role: Role::Assistant,
            content: self.response_content.clone().into(),
        })
    }

    fn model_name(&self) -> &str {
        "mock-model"
    }

    fn usage_slot(&self) -> Option<&Mutex<Option<TokenUsage>>> {
        Some(&self.usage)
    }
}

#[tokio::test]
async fn test_token_caching() {
    let mock_client = Arc::new(MockClient::new("Response".to_string()));
    let mut session = LLMSession::new(mock_client.clone(), "System prompt".to_string(), 1000);

    // Send a message
    let user_message = "Hello, this is a test message";
    mock_client.set_usage(100, 50, 150).await;

    let _ = session
        .send_message(Role::User, user_message.to_string(), None, None)
        .await;

    // Verify that both the user message and response have cached token counts
    assert_eq!(session.get_conversation_history().len(), 2); // User message + response
    assert_eq!(session.get_cached_token_counts().len(), 2); // Token counts for both messages

    // Verify token counts are cached correctly
    let expected_user_tokens = llm_session::estimate_message_token_count(&Message {
        role: Role::User,
        content: user_message.to_string().into(),
    });
    let expected_response_tokens = llm_session::estimate_message_token_count(&Message {
        role: Role::Assistant,
        content: "Response".to_string().into(),
    });

    assert_eq!(session.get_cached_token_counts()[0], expected_user_tokens);
    assert_eq!(
        session.get_cached_token_counts()[1],
        expected_response_tokens
    );
}

#[tokio::test]
async fn test_token_caching_with_trimming() {
    let mock_client = Arc::new(MockClient::new("Response".to_string()));
    let mut session = LLMSession::new(
        mock_client.clone(),
        "System prompt".to_string(),
        100, // Small max_tokens to trigger trimming
    );

    // Send first message
    mock_client.set_usage(50, 25, 75).await;
    let _ = session
        .send_message(Role::User, "First message".to_string(), None, None)
        .await;

    assert_eq!(session.get_conversation_history().len(), 2);
    assert_eq!(session.get_cached_token_counts().len(), 2);

    // Send second message with usage that exceeds max_tokens
    mock_client.set_usage(80, 40, 120).await; // Exceeds max_tokens of 100
    let _ = session
        .send_message(Role::User, "Second message".to_string(), None, None)
        .await;

    // Some messages should have been trimmed
    assert!(session.get_conversation_history().len() < 4); // Should have fewer than 4 messages
                                                           // cached_token_counts should match conversation_history length
    assert_eq!(
        session.get_conversation_history().len(),
        session.get_cached_token_counts().len()
    );
}

#[test]
fn test_estimate_token_count() {
    // Test basic token estimation (1 token per 4 characters)
    assert_eq!(llm_session::estimate_token_count("test"), 1);
    assert_eq!(
        llm_session::estimate_token_count("this is a longer test"),
        5
    );
    assert_eq!(llm_session::estimate_token_count(""), 1); // Minimum 1 token
}

#[test]
fn test_estimate_message_token_count() {
    let message = Message {
        role: Role::User,
        content: "test message".to_string().into(),
    };
    // "test message" = 12 characters = 3 tokens + 1 role token = 4 tokens
    assert_eq!(llm_session::estimate_message_token_count(&message), 4);
}

#[tokio::test]
async fn test_pre_transmission_trimming() {
    // Create a mock client
    let client = Arc::new(MockClient::new("Response".to_string()));

    // Create a session with a very small max_tokens limit
    // System prompt: "System" = (6/4).max(1) + 1 = 2 + 1 = 3 tokens
    let mut session = LLMSession::new(
        client.clone(),
        "System".to_string(),
        20, // Very small limit to trigger trimming
    );

    // Add several messages that exceed the limit
    // Each message with 4 chars = (4/4).max(1) + 1 = 1 + 1 = 2 tokens
    let _ = session
        .send_message(Role::User, "Msg1".to_string(), None, None)
        .await;
    let _ = session
        .send_message(Role::User, "Msg2".to_string(), None, None)
        .await;
    let _ = session
        .send_message(Role::User, "Msg3".to_string(), None, None)
        .await;

    // Add a large message that should trigger trimming
    // 40 chars = (40/4).max(1) + 1 = 10 + 1 = 11 tokens
    let large_msg = "0123456789012345678901234567890123456789"; // 40 chars
    let _ = session
        .send_message(Role::User, large_msg.to_string(), None, None)
        .await;

    // The client should have received fewer messages than we sent
    // because old messages should have been trimmed before transmission
    let message_count = client.get_last_message_count().await;

    // With max_tokens=20:
    // System prompt (3 tokens) + large message (11 tokens) = 14 tokens
    // We should have trimmed old messages to stay under 20
    // The last call should have sent: system prompt + some history + large message
    assert!(
        message_count > 0,
        "Should have sent at least the system prompt and new message"
    );
    assert!(
        message_count < 6,
        "Should have trimmed some messages (system + 4 user + 4 assistant = 9 total before trim)"
    );

    // Verify that conversation history exists
    assert!(
        !session.get_conversation_history().is_empty(),
        "Conversation history should not be empty"
    );
}

#[tokio::test]
async fn test_no_trimming_when_under_limit() {
    let client = Arc::new(MockClient::new("OK".to_string()));

    // Large max_tokens limit - no trimming should occur
    let mut session = LLMSession::new(client.clone(), "System".to_string(), 10000);

    // Add a few small messages
    let _ = session
        .send_message(Role::User, "Hi".to_string(), None, None)
        .await;
    let _ = session
        .send_message(Role::User, "Hello".to_string(), None, None)
        .await;

    // The last send should include: system prompt + first user message + first assistant response + second user message
    // = 1 system + 1 user + 1 assistant + 1 user = 4 messages
    let message_count = client.get_last_message_count().await;
    assert_eq!(
        message_count, 4,
        "Should have sent all messages without trimming"
    );
}

#[tokio::test]
async fn test_request_buffer_reuse() {
    // Test that the request buffer is reused correctly across multiple send_message calls
    let client = Arc::new(MockClient::new("Response".to_string()));
    let mut session = LLMSession::new(
        client.clone() as Arc<dyn ClientWrapper>,
        "System prompt".to_string(),
        10_000,
    );

    // Send first message
    let _ = session
        .send_message(Role::User, "First".to_string(), None, None)
        .await;

    // Should have sent: system prompt + user message = 2 messages
    let count1 = client.get_last_message_count().await;
    assert_eq!(count1, 2);

    // Send second message
    let _ = session
        .send_message(Role::User, "Second".to_string(), None, None)
        .await;

    // Should have sent: system prompt + first user + first assistant + second user = 4 messages
    let count2 = client.get_last_message_count().await;
    assert_eq!(count2, 4);

    // Send third message
    let _ = session
        .send_message(Role::User, "Third".to_string(), None, None)
        .await;

    // Should have sent: system prompt + all messages = 6 messages
    let count3 = client.get_last_message_count().await;
    assert_eq!(count3, 6);
}
