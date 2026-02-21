use async_trait::async_trait;
use cloudllm::client_wrapper::{ClientWrapper, Message, Role, TokenUsage, ToolDefinition};
use cloudllm::thought_chain::{ThoughtChain, ThoughtType};
use cloudllm::Agent;
use std::sync::Arc;
use tokio::sync::RwLock;

struct MockClient {
    response: String,
}

#[async_trait]
impl ClientWrapper for MockClient {
    async fn send_message(
        &self,
        _messages: &[Message],
        _tools: Option<Vec<ToolDefinition>>,
    ) -> Result<Message, Box<dyn std::error::Error>> {
        Ok(Message {
            role: Role::Assistant,
            content: Arc::from(self.response.as_str()),
            tool_calls: vec![],
        })
    }

    fn model_name(&self) -> &str {
        "mock"
    }

    async fn get_last_usage(&self) -> Option<TokenUsage> {
        None
    }
}

#[test]
fn test_agent_fork() {
    let client: Arc<dyn ClientWrapper> = Arc::new(MockClient {
        response: "test".to_string(),
    });

    let agent = Agent::new("original", "Original Agent", client)
        .with_expertise("Testing")
        .with_personality("Friendly")
        .with_metadata("key", "value");

    let forked = agent.fork();

    // Identity is cloned
    assert_eq!(forked.id, "original");
    assert_eq!(forked.name, "Original Agent");
    assert_eq!(forked.expertise, Some("Testing".to_string()));
    assert_eq!(forked.personality, Some("Friendly".to_string()));
    assert_eq!(forked.metadata.get("key"), Some(&"value".to_string()));
}

#[tokio::test]
async fn test_agent_runtime_tool_mutation() {
    use cloudllm::tool_protocol::ToolMetadata;
    use cloudllm::tool_protocols::CustomToolProtocol;

    let client: Arc<dyn ClientWrapper> = Arc::new(MockClient {
        response: "test".to_string(),
    });

    let protocol = Arc::new(CustomToolProtocol::new());
    protocol
        .register_tool(
            ToolMetadata::new("test_tool", "A test tool"),
            Arc::new(|_params| {
                Ok(cloudllm::tool_protocol::ToolResult::success(
                    serde_json::json!({"ok": true}),
                ))
            }),
        )
        .await;

    let agent = Agent::new("agent1", "Agent", client);

    // Initially empty
    let tools = agent.list_tools().await;
    assert!(tools.is_empty());

    // Add protocol
    agent.add_protocol("custom", protocol).await.unwrap();
    let tools = agent.list_tools().await;
    assert_eq!(tools, vec!["test_tool"]);

    // Remove protocol
    agent.remove_protocol("custom").await;
    let tools = agent.list_tools().await;
    assert!(tools.is_empty());
}

#[tokio::test]
async fn test_agent_with_thought_chain() {
    let dir = std::env::temp_dir().join(format!("cloudllm_agent_tc_test_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);

    let chain = ThoughtChain::open(&dir, "agent1", "Agent", None, None).unwrap();
    let chain = Arc::new(RwLock::new(chain));

    let client: Arc<dyn ClientWrapper> = Arc::new(MockClient {
        response: "test".to_string(),
    });

    let agent = Agent::new("agent1", "Agent", client).with_thought_chain(chain.clone());

    // Commit thoughts
    agent
        .commit(ThoughtType::Finding, "Found something")
        .await
        .unwrap();
    agent
        .commit(ThoughtType::Decision, "Decided to proceed")
        .await
        .unwrap();

    // Retrieve thoughts
    let entries = agent.thought_entries().await.unwrap();
    assert_eq!(entries.len(), 2);
    assert_eq!(entries[0].content, "Found something");
    assert_eq!(entries[1].thought_type, ThoughtType::Decision);

    let _ = std::fs::remove_dir_all(&dir);
}
