use async_trait::async_trait;
use cloudllm::client_wrapper::{
    ClientWrapper, Message, NativeToolCall, Role, TokenUsage, ToolDefinition,
};
use cloudllm::tool_protocol::{ToolMetadata, ToolRegistry, ToolResult};
use cloudllm::tool_protocols::CustomToolProtocol;
use cloudllm::Agent;
use mentisdb::{MentisDb, ThoughtType};
use serde_json::json;
use std::sync::{Arc, Mutex};
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
async fn test_agent_with_mentisdb() {
    let dir = std::env::temp_dir().join(format!("cloudllm_agent_tc_test_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);

    let chain = MentisDb::open(&dir, "agent1", "Agent", None, None).unwrap();
    let chain = Arc::new(RwLock::new(chain));

    let client: Arc<dyn ClientWrapper> = Arc::new(MockClient {
        response: "test".to_string(),
    });

    let agent = Agent::new("agent1", "Agent", client).with_mentisdb(chain.clone());

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

struct MultiToolMockClient {
    requests: Mutex<Vec<Vec<Message>>>,
}

#[async_trait]
impl ClientWrapper for MultiToolMockClient {
    async fn send_message(
        &self,
        messages: &[Message],
        _tools: Option<Vec<ToolDefinition>>,
    ) -> Result<Message, Box<dyn std::error::Error>> {
        let mut requests = self.requests.lock().unwrap();
        requests.push(messages.to_vec());

        match requests.len() {
            1 => Ok(Message {
                role: Role::Assistant,
                content: Arc::from(""),
                tool_calls: vec![
                    NativeToolCall {
                        id: "call_one".to_string(),
                        name: "tool_one".to_string(),
                        arguments: json!({ "value": "alpha" }),
                    },
                    NativeToolCall {
                        id: "call_two".to_string(),
                        name: "tool_two".to_string(),
                        arguments: json!({ "value": "beta" }),
                    },
                ],
            }),
            2 => Ok(Message {
                role: Role::Assistant,
                content: Arc::from("both tools handled"),
                tool_calls: vec![],
            }),
            n => panic!("unexpected mock call count: {}", n),
        }
    }

    fn model_name(&self) -> &str {
        "mock-native-tools"
    }

    async fn get_last_usage(&self) -> Option<TokenUsage> {
        None
    }
}

#[tokio::test]
async fn test_agent_send_handles_multiple_native_tool_calls() {
    let client = Arc::new(MultiToolMockClient {
        requests: Mutex::new(Vec::new()),
    });

    let protocol = Arc::new(CustomToolProtocol::new());
    protocol
        .register_tool(
            ToolMetadata::new("tool_one", "Returns the first mock result"),
            Arc::new(|params| {
                let value = params["value"].as_str().unwrap_or_default().to_string();
                Ok(ToolResult::success(
                    json!({ "tool": "tool_one", "value": value }),
                ))
            }),
        )
        .await;
    protocol
        .register_tool(
            ToolMetadata::new("tool_two", "Returns the second mock result"),
            Arc::new(|params| {
                let value = params["value"].as_str().unwrap_or_default().to_string();
                Ok(ToolResult::success(
                    json!({ "tool": "tool_two", "value": value }),
                ))
            }),
        )
        .await;

    let mut registry = ToolRegistry::new(protocol);
    registry.discover_tools_from_primary().await.unwrap();

    let mut agent = Agent::new(
        "agent-native-tools",
        "Native Tool Agent",
        client.clone() as Arc<dyn ClientWrapper>,
    )
    .with_tools(registry);

    let response = agent.send("Run both tools.").await.unwrap();
    assert_eq!(response.content, "both tools handled");

    let requests = client.requests.lock().unwrap();
    assert_eq!(requests.len(), 2);

    let second_request = &requests[1];
    assert!(matches!(second_request[0].role, Role::System));
    assert!(matches!(second_request[1].role, Role::User));
    assert!(matches!(second_request[2].role, Role::Assistant));
    assert_eq!(second_request[2].tool_calls.len(), 2);

    match &second_request[3].role {
        Role::Tool { call_id } => assert_eq!(call_id, "call_one"),
        other => panic!("expected first tool response, got {:?}", other),
    }
    match &second_request[4].role {
        Role::Tool { call_id } => assert_eq!(call_id, "call_two"),
        other => panic!("expected second tool response, got {:?}", other),
    }
}
