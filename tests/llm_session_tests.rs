use openai_rust2;
use openai_rust2 as openai_rust;
use cloudllm::client_wrapper;
use cloudllm::client_wrapper::{ClientWrapper, Message, Role, TokenUsage};
use cloudllm::LLMSession;
use cloudllm::cloudllm::llm_session;
use async_trait::async_trait;
use std::sync::Arc;
use tokio::sync::Mutex;

// Mock client for testing
struct MockClient {
    usage: Mutex<Option<client_wrapper::TokenUsage>>,
    response_content: String,
}

impl MockClient {
    fn new(response_content: String) -> Self {
        Self {
            usage: Mutex::new(None),
            response_content,
        }
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
        _messages: &[Message],
        _optional_search_parameters: Option<openai_rust::chat::SearchParameters>,
    ) -> Result<Message, Box<dyn std::error::Error>> {
        Ok(Message {
            role: Role::Assistant,
            content: self.response_content.clone().into(),
        })
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
        .send_message(Role::User, user_message.to_string(), None)
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
    assert_eq!(session.get_cached_token_counts()[1], expected_response_tokens);
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
        .send_message(Role::User, "First message".to_string(), None)
        .await;

    assert_eq!(session.get_conversation_history().len(), 2);
    assert_eq!(session.get_cached_token_counts().len(), 2);

    // Send second message with usage that exceeds max_tokens
    mock_client.set_usage(80, 40, 120).await; // Exceeds max_tokens of 100
    let _ = session
        .send_message(Role::User, "Second message".to_string(), None)
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
    assert_eq!(llm_session::estimate_token_count("this is a longer test"), 5);
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
