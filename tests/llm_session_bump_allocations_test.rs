use async_trait::async_trait;
use cloudllm::client_wrapper::{ClientWrapper, Message, Role, TokenUsage};
use cloudllm::LLMSession;
use openai_rust2 as openai_rust;
use std::sync::Arc;
use tokio::sync::Mutex;

// Mock client for testing
struct MockClient {
    usage: Mutex<Option<TokenUsage>>,
    response_content: String,
}

impl MockClient {
    fn new(response_content: String) -> Self {
        Self {
            usage: Mutex::new(None),
            response_content,
        }
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

    fn model_name(&self) -> &str {
        "mock-model"
    }

    fn usage_slot(&self) -> Option<&Mutex<Option<TokenUsage>>> {
        Some(&self.usage)
    }
}

#[tokio::test]
async fn test_arena_allocation() {
    let mock_client = Arc::new(MockClient::new("Mock response".to_string()));
    let mut session = LLMSession::new(mock_client, "Test system prompt".to_string(), 1000);

    // Send a message
    let result = session
        .send_message(Role::User, "Test user message".to_string(), None)
        .await;

    assert!(result.is_ok());
    let response = result.unwrap();
    assert_eq!(&*response.content, "Mock response");

    // Verify system prompt is allocated correctly
    assert_eq!(&*session.get_system_prompt().content, "Test system prompt");

    // Verify conversation history
    assert_eq!(session.get_conversation_history().len(), 2); // user message + assistant response
}

#[test]
fn test_set_system_prompt() {
    let mock_client = Arc::new(MockClient::new("Response".to_string()));
    let mut session = LLMSession::new(mock_client, "Initial prompt".to_string(), 1000);

    // Change system prompt
    session.set_system_prompt("Updated prompt".to_string());
    assert_eq!(&*session.get_system_prompt().content, "Updated prompt");
}

#[test]
fn test_message_content_is_arc_str() {
    // Verify that Message.content is Arc<str> and cloning is cheap
    let msg = Message {
        role: Role::User,
        content: Arc::from("Test message"),
    };

    let cloned = msg.clone();

    // Arc::ptr_eq checks if both Arcs point to the same allocation
    assert!(Arc::ptr_eq(&msg.content, &cloned.content));
}
