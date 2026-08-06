use async_trait::async_trait;
use cloudllm::client_wrapper::{ClientWrapper, Message, Role, TokenUsage, ToolDefinition};
use cloudllm::LLMSession;
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
        _tools: Option<Vec<ToolDefinition>>,
    ) -> Result<Message, Box<dyn std::error::Error>> {
        let mut usage = self.usage.lock().await;
        *usage = Some(TokenUsage {
            input_tokens: 10,
            output_tokens: 5,
            total_tokens: 15,
        });
        Ok(Message {
            role: Role::Assistant,
            content: self.response_content.clone().into(),
            tool_calls: vec![],
        })
    }

    fn model_name(&self) -> &str {
        "mock-model"
    }

    fn provider_name(&self) -> &str {
        "mock"
    }

    fn usage_slot(&self) -> Option<&Mutex<Option<TokenUsage>>> {
        Some(&self.usage)
    }
}

#[tokio::test]
async fn test_message_content_single_arc_allocation() {
    let mock_client = Arc::new(MockClient::new("Mock response".to_string()));
    let mut session = LLMSession::new(mock_client, "Test system prompt".to_string(), 1000);

    let result = session
        .send_message(Role::User, "Test user message".to_string(), None)
        .await;

    assert!(result.is_ok());
    let response = result.unwrap();
    assert_eq!(&*response.content, "Mock response");

    // Returned message and history share the same Arc allocation.
    let history = session.get_conversation_history();
    assert_eq!(history.len(), 2);
    assert!(Arc::ptr_eq(&response.content, &history[1].content));

    assert_eq!(&*session.get_system_prompt().content, "Test system prompt");
}

#[tokio::test]
async fn test_token_usage_accumulates_across_turns() {
    let mock_client = Arc::new(MockClient::new("ok".to_string()));
    let mut session = LLMSession::new(mock_client, "sys".to_string(), 10_000);

    session
        .send_message(Role::User, "first".to_string(), None)
        .await
        .unwrap();
    let after_one = session.token_usage();
    assert!(after_one.total_tokens > 0);

    session
        .send_message(Role::User, "second".to_string(), None)
        .await
        .unwrap();
    let after_two = session.token_usage();
    assert!(after_two.total_tokens >= after_one.total_tokens * 2
        || after_two.input_tokens > after_one.input_tokens);
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
        tool_calls: vec![],
    };

    let cloned = msg.clone();

    // Arc::ptr_eq checks if both Arcs point to the same allocation
    assert!(Arc::ptr_eq(&msg.content, &cloned.content));
}
