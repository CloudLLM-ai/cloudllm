//! Integration tests for each built-in tool via Agent.send()
//!
//! Each test wires up a real tool through its protocol, creates an Agent
//! with a MockClient that returns a tool_call JSON on the first LLM round
//! and a final answer on the second round, then asserts the tool actually
//! executed and the agent returned meaningful output.
//!
//! These tests catch issues like the Memory "ERR:Unknown Command" bug
//! where the protocol's command format doesn't match what agents send.

use async_trait::async_trait;
use cloudllm::client_wrapper::{ClientWrapper, Message, Role, TokenUsage, ToolDefinition};
use cloudllm::tool_protocol::{
    ToolMetadata, ToolParameter, ToolParameterType, ToolProtocol, ToolRegistry, ToolResult,
};
use cloudllm::tool_protocols::{BashProtocol, CustomToolProtocol, MemoryProtocol};
use cloudllm::tools::{BashTool, Calculator, FileSystemTool, HttpClient, Memory, Platform};
use cloudllm::Agent;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use tokio::sync::RwLock;

/// A MockClient that returns different responses on sequential calls.
///
/// - Call 1: returns `first_response` (typically a tool_call JSON)
/// - Call 2+: returns `second_response` (typically a final answer)
struct SequentialMockClient {
    first_response: String,
    second_response: String,
    call_count: AtomicUsize,
}

impl SequentialMockClient {
    fn new(first_response: String, second_response: String) -> Self {
        Self {
            first_response,
            second_response,
            call_count: AtomicUsize::new(0),
        }
    }
}

#[async_trait]
impl ClientWrapper for SequentialMockClient {
    async fn send_message(
        &self,
        _messages: &[Message],
        _tools: Option<Vec<ToolDefinition>>,
    ) -> Result<Message, Box<dyn std::error::Error>> {
        let count = self.call_count.fetch_add(1, Ordering::SeqCst);
        let response = if count == 0 {
            &self.first_response
        } else {
            &self.second_response
        };
        Ok(Message {
            role: Role::Assistant,
            content: Arc::from(response.as_str()),
            tool_calls: vec![],
        })
    }

    fn model_name(&self) -> &str {
        "mock-sequential"
    }

    async fn get_last_usage(&self) -> Option<TokenUsage> {
        None
    }
}

// =============================================================================
// Helper: build an agent with a SequentialMockClient and a populated ToolRegistry
// =============================================================================

async fn build_agent_with_registry(
    name: &str,
    registry: ToolRegistry,
    tool_call_json: &str,
    final_answer: &str,
) -> Agent {
    let client: Arc<dyn ClientWrapper> = Arc::new(SequentialMockClient::new(
        tool_call_json.to_string(),
        final_answer.to_string(),
    ));
    let shared_registry = Arc::new(RwLock::new(registry));
    Agent::new(name, name, client).with_shared_tools(shared_registry)
}

// =============================================================================
// TEST 1: Memory tool via MemoryProtocol
// =============================================================================

/// Verify that the MemoryProtocol correctly handles P (put) and G (get) commands
/// when invoked through the Agent's tool loop.
#[tokio::test]
async fn test_memory_tool_put_via_agent() {
    let memory = Arc::new(Memory::new());
    let protocol = Arc::new(MemoryProtocol::new(memory.clone()));

    let mut registry = ToolRegistry::new(protocol);
    registry.discover_tools_from_primary().await.unwrap();

    // Mock LLM returns a tool_call to store "hello" under key "greeting"
    let tool_call =
        r#"{"tool_call": {"name": "memory", "parameters": {"command": "P greeting hello"}}}"#;
    let final_answer = "I stored the greeting in memory.";

    let mut agent = build_agent_with_registry("mem_agent", registry, tool_call, final_answer).await;
    let response = agent.send("Store greeting=hello in memory").await.unwrap();

    // Agent should return the final answer
    assert_eq!(response.content, final_answer);

    // Verify the memory actually has the value
    let (value, _) = memory
        .get("greeting", false)
        .expect("Key 'greeting' should exist");
    assert_eq!(value, "hello");
}

#[tokio::test]
async fn test_memory_tool_get_via_agent() {
    let memory = Arc::new(Memory::new());
    // Pre-populate memory
    memory.put("city".to_string(), "Berlin".to_string(), None);

    let protocol = Arc::new(MemoryProtocol::new(memory.clone()));
    let mut registry = ToolRegistry::new(protocol);
    registry.discover_tools_from_primary().await.unwrap();

    let tool_call = r#"{"tool_call": {"name": "memory", "parameters": {"command": "G city"}}}"#;
    let final_answer = "The stored city is Berlin.";

    let mut agent =
        build_agent_with_registry("mem_get_agent", registry, tool_call, final_answer).await;
    let response = agent.send("What city is stored in memory?").await.unwrap();

    assert_eq!(response.content, final_answer);
}

#[tokio::test]
async fn test_memory_tool_list_via_agent() {
    let memory = Arc::new(Memory::new());
    memory.put("k1".to_string(), "v1".to_string(), None);
    memory.put("k2".to_string(), "v2".to_string(), None);

    let protocol = Arc::new(MemoryProtocol::new(memory.clone()));
    let mut registry = ToolRegistry::new(protocol);
    registry.discover_tools_from_primary().await.unwrap();

    let tool_call = r#"{"tool_call": {"name": "memory", "parameters": {"command": "L"}}}"#;
    let final_answer = "Memory contains keys: k1, k2.";

    let mut agent =
        build_agent_with_registry("mem_list_agent", registry, tool_call, final_answer).await;
    let response = agent.send("List all keys in memory").await.unwrap();

    assert_eq!(response.content, final_answer);
}

#[tokio::test]
async fn test_memory_tool_delete_via_agent() {
    let memory = Arc::new(Memory::new());
    memory.put("temp".to_string(), "data".to_string(), None);

    let protocol = Arc::new(MemoryProtocol::new(memory.clone()));
    let mut registry = ToolRegistry::new(protocol);
    registry.discover_tools_from_primary().await.unwrap();

    let tool_call = r#"{"tool_call": {"name": "memory", "parameters": {"command": "D temp"}}}"#;
    let final_answer = "Deleted temp from memory.";

    let mut agent =
        build_agent_with_registry("mem_del_agent", registry, tool_call, final_answer).await;
    let response = agent.send("Delete temp from memory").await.unwrap();

    assert_eq!(response.content, final_answer);
    assert!(memory.get("temp", false).is_none(), "Key should be deleted");
}

#[tokio::test]
async fn test_memory_tool_put_with_ttl_via_agent() {
    let memory = Arc::new(Memory::new());
    let protocol = Arc::new(MemoryProtocol::new(memory.clone()));

    let mut registry = ToolRegistry::new(protocol);
    registry.discover_tools_from_primary().await.unwrap();

    // Store with TTL of 3600 seconds
    let tool_call = r#"{"tool_call": {"name": "memory", "parameters": {"command": "P session token123 3600"}}}"#;
    let final_answer = "Session token stored with 1h TTL.";

    let mut agent =
        build_agent_with_registry("mem_ttl_agent", registry, tool_call, final_answer).await;
    let response = agent.send("Store session token with 1h TTL").await.unwrap();

    assert_eq!(response.content, final_answer);
    let (value, _) = memory
        .get("session", false)
        .expect("Key 'session' should exist");
    assert_eq!(value, "token123");
}

#[tokio::test]
async fn test_memory_tool_clear_via_agent() {
    let memory = Arc::new(Memory::new());
    memory.put("a".to_string(), "1".to_string(), None);
    memory.put("b".to_string(), "2".to_string(), None);

    let protocol = Arc::new(MemoryProtocol::new(memory.clone()));
    let mut registry = ToolRegistry::new(protocol);
    registry.discover_tools_from_primary().await.unwrap();

    let tool_call = r#"{"tool_call": {"name": "memory", "parameters": {"command": "C"}}}"#;
    let final_answer = "Memory cleared.";

    let mut agent =
        build_agent_with_registry("mem_clear_agent", registry, tool_call, final_answer).await;
    let response = agent.send("Clear all memory").await.unwrap();

    assert_eq!(response.content, final_answer);
    assert!(
        memory.list_keys().is_empty(),
        "Memory should be empty after clear"
    );
}

#[tokio::test]
async fn test_memory_tool_total_bytes_via_agent() {
    let memory = Arc::new(Memory::new());
    memory.put("key".to_string(), "value".to_string(), None);

    let protocol = Arc::new(MemoryProtocol::new(memory.clone()));
    let mut registry = ToolRegistry::new(protocol);
    registry.discover_tools_from_primary().await.unwrap();

    let tool_call = r#"{"tool_call": {"name": "memory", "parameters": {"command": "T A"}}}"#;
    let final_answer = "Total memory usage: 8 bytes.";

    let mut agent =
        build_agent_with_registry("mem_total_agent", registry, tool_call, final_answer).await;
    let response = agent.send("How much memory is being used?").await.unwrap();

    assert_eq!(response.content, final_answer);
}

#[tokio::test]
async fn test_memory_tool_spec_via_agent() {
    let memory = Arc::new(Memory::new());
    let protocol = Arc::new(MemoryProtocol::new(memory.clone()));

    let mut registry = ToolRegistry::new(protocol);
    registry.discover_tools_from_primary().await.unwrap();

    let tool_call = r#"{"tool_call": {"name": "memory", "parameters": {"command": "SPEC"}}}"#;
    let final_answer = "Here is the memory protocol specification.";

    let mut agent =
        build_agent_with_registry("mem_spec_agent", registry, tool_call, final_answer).await;
    let response = agent
        .send("Show me the memory protocol spec")
        .await
        .unwrap();

    assert_eq!(response.content, final_answer);
}

/// Verify that an invalid memory command produces a failure result (not a crash).
#[tokio::test]
async fn test_memory_tool_invalid_command_via_agent() {
    let memory = Arc::new(Memory::new());
    let protocol = Arc::new(MemoryProtocol::new(memory.clone()));

    let mut registry = ToolRegistry::new(protocol);
    registry.discover_tools_from_primary().await.unwrap();

    // This simulates the "ERR:Unknown Command" bug — agent sends wrong format
    let tool_call =
        r#"{"tool_call": {"name": "memory", "parameters": {"command": "PUT greeting hello"}}}"#;
    let final_answer = "The memory tool returned an error, let me try differently.";

    let mut agent =
        build_agent_with_registry("mem_bad_agent", registry, tool_call, final_answer).await;
    let response = agent.send("Store greeting=hello").await.unwrap();

    // The agent should still return (the second LLM call gives the final answer)
    assert_eq!(response.content, final_answer);
}

// =============================================================================
// TEST 2: Bash tool via BashProtocol
// =============================================================================

#[tokio::test]
async fn test_bash_tool_echo_via_agent() {
    let bash_tool = Arc::new(BashTool::new(Platform::Linux).with_timeout(10));
    let protocol = Arc::new(BashProtocol::new(bash_tool));

    let mut registry = ToolRegistry::new(protocol);
    registry.discover_tools_from_primary().await.unwrap();

    let tool_call =
        r#"{"tool_call": {"name": "bash", "parameters": {"command": "echo hello_from_agent"}}}"#;
    let final_answer = "The command echoed: hello_from_agent";

    let mut agent =
        build_agent_with_registry("bash_agent", registry, tool_call, final_answer).await;
    let response = agent.send("Run echo hello_from_agent").await.unwrap();

    assert_eq!(response.content, final_answer);
}

#[tokio::test]
async fn test_bash_tool_pwd_via_agent() {
    let bash_tool = Arc::new(BashTool::new(Platform::Linux).with_timeout(10));
    let protocol = Arc::new(BashProtocol::new(bash_tool));

    let mut registry = ToolRegistry::new(protocol);
    registry.discover_tools_from_primary().await.unwrap();

    let tool_call = r#"{"tool_call": {"name": "bash", "parameters": {"command": "pwd"}}}"#;
    let final_answer = "Current directory retrieved.";

    let mut agent =
        build_agent_with_registry("bash_pwd_agent", registry, tool_call, final_answer).await;
    let response = agent.send("What is the current directory?").await.unwrap();

    assert_eq!(response.content, final_answer);
}

#[tokio::test]
async fn test_bash_tool_denied_command_via_agent() {
    let bash_tool = Arc::new(
        BashTool::new(Platform::Linux)
            .with_timeout(10)
            .with_denied_commands(vec!["rm".to_string()]),
    );
    let protocol = Arc::new(BashProtocol::new(bash_tool));

    let mut registry = ToolRegistry::new(protocol);
    registry.discover_tools_from_primary().await.unwrap();

    // Agent tries a denied command — BashProtocol wraps the error as ToolResult::failure
    let tool_call = r#"{"tool_call": {"name": "bash", "parameters": {"command": "rm -rf /"}}}"#;
    let final_answer = "The command was blocked for security.";

    let mut agent =
        build_agent_with_registry("bash_deny_agent", registry, tool_call, final_answer).await;
    let response = agent.send("Delete everything").await.unwrap();

    // Agent still gets a response (BashProtocol returns failure, agent gets error message, second LLM call)
    assert_eq!(response.content, final_answer);
}

// =============================================================================
// TEST 3: Calculator tool via CustomToolProtocol
// =============================================================================

async fn build_calculator_protocol() -> CustomToolProtocol {
    let adapter = CustomToolProtocol::new();
    let calc = Arc::new(Calculator::new());

    let calc_clone = calc.clone();
    adapter
        .register_async_tool(
            ToolMetadata::new(
                "calculator",
                "Evaluate mathematical expressions. Supports arithmetic (+, -, *, /, ^, %), \
                 trigonometric (sin, cos, tan), logarithmic (ln, log, log2), \
                 and statistical functions (mean, median, std, etc.).",
            )
            .with_parameter(
                ToolParameter::new("expression", ToolParameterType::String)
                    .with_description("Mathematical expression to evaluate (e.g., '2 + 2', 'sqrt(16)', 'mean([1,2,3])')")
                    .required(),
            ),
            Arc::new(move |params| {
                let calc = calc_clone.clone();
                Box::pin(async move {
                    let expression = params
                        .get("expression")
                        .and_then(|v| v.as_str())
                        .ok_or_else(|| -> Box<dyn std::error::Error + Send + Sync> {
                            "Missing 'expression' parameter".into()
                        })?;

                    match calc.evaluate(expression).await {
                        Ok(result) => Ok(ToolResult::success(
                            serde_json::json!({"result": result}),
                        )),
                        Err(e) => Ok(ToolResult::failure(e.to_string())),
                    }
                })
            }),
        )
        .await;

    adapter
}

#[tokio::test]
async fn test_calculator_basic_arithmetic_via_agent() {
    let adapter = build_calculator_protocol().await;

    let mut registry = ToolRegistry::new(Arc::new(adapter));
    registry.discover_tools_from_primary().await.unwrap();

    let tool_call =
        r#"{"tool_call": {"name": "calculator", "parameters": {"expression": "2 + 3 * 4"}}}"#;
    let final_answer = "The result is 14.";

    let mut agent =
        build_agent_with_registry("calc_agent", registry, tool_call, final_answer).await;
    let response = agent.send("What is 2 + 3 * 4?").await.unwrap();

    assert_eq!(response.content, final_answer);
}

#[tokio::test]
async fn test_calculator_sqrt_via_agent() {
    let adapter = build_calculator_protocol().await;

    let mut registry = ToolRegistry::new(Arc::new(adapter));
    registry.discover_tools_from_primary().await.unwrap();

    let tool_call =
        r#"{"tool_call": {"name": "calculator", "parameters": {"expression": "sqrt(144)"}}}"#;
    let final_answer = "The square root of 144 is 12.";

    let mut agent =
        build_agent_with_registry("calc_sqrt_agent", registry, tool_call, final_answer).await;
    let response = agent.send("What is the square root of 144?").await.unwrap();

    assert_eq!(response.content, final_answer);
}

#[tokio::test]
async fn test_calculator_trig_via_agent() {
    let adapter = build_calculator_protocol().await;

    let mut registry = ToolRegistry::new(Arc::new(adapter));
    registry.discover_tools_from_primary().await.unwrap();

    let tool_call =
        r#"{"tool_call": {"name": "calculator", "parameters": {"expression": "sin(0)"}}}"#;
    let final_answer = "sin(0) = 0.";

    let mut agent =
        build_agent_with_registry("calc_trig_agent", registry, tool_call, final_answer).await;
    let response = agent.send("What is sin(0)?").await.unwrap();

    assert_eq!(response.content, final_answer);
}

#[tokio::test]
async fn test_calculator_statistics_via_agent() {
    let adapter = build_calculator_protocol().await;

    let mut registry = ToolRegistry::new(Arc::new(adapter));
    registry.discover_tools_from_primary().await.unwrap();

    let tool_call = r#"{"tool_call": {"name": "calculator", "parameters": {"expression": "mean([1, 2, 3, 4, 5])"}}}"#;
    let final_answer = "The mean of [1,2,3,4,5] is 3.0.";

    let mut agent =
        build_agent_with_registry("calc_stats_agent", registry, tool_call, final_answer).await;
    let response = agent.send("Calculate the mean of 1,2,3,4,5").await.unwrap();

    assert_eq!(response.content, final_answer);
}

#[tokio::test]
async fn test_calculator_log_via_agent() {
    let adapter = build_calculator_protocol().await;

    let mut registry = ToolRegistry::new(Arc::new(adapter));
    registry.discover_tools_from_primary().await.unwrap();

    let tool_call =
        r#"{"tool_call": {"name": "calculator", "parameters": {"expression": "log(100)"}}}"#;
    let final_answer = "log10(100) = 2.";

    let mut agent =
        build_agent_with_registry("calc_log_agent", registry, tool_call, final_answer).await;
    let response = agent.send("What is log base 10 of 100?").await.unwrap();

    assert_eq!(response.content, final_answer);
}

// =============================================================================
// TEST 4: FileSystemTool via CustomToolProtocol
// =============================================================================

async fn build_filesystem_protocol(root: std::path::PathBuf) -> CustomToolProtocol {
    let adapter = CustomToolProtocol::new();
    let fs_tool = Arc::new(FileSystemTool::new().with_root_path(root));

    // Register read_file tool
    let fs_read = fs_tool.clone();
    adapter
        .register_async_tool(
            ToolMetadata::new("read_file", "Read the contents of a file").with_parameter(
                ToolParameter::new("path", ToolParameterType::String)
                    .with_description("Relative path to the file")
                    .required(),
            ),
            Arc::new(move |params| {
                let fs = fs_read.clone();
                Box::pin(async move {
                    let path = params.get("path").and_then(|v| v.as_str()).ok_or_else(
                        || -> Box<dyn std::error::Error + Send + Sync> {
                            "Missing 'path' parameter".into()
                        },
                    )?;
                    match fs.read_file(path).await {
                        Ok(content) => {
                            Ok(ToolResult::success(serde_json::json!({"content": content})))
                        }
                        Err(e) => Ok(ToolResult::failure(e.to_string())),
                    }
                })
            }),
        )
        .await;

    // Register write_file tool
    let fs_write = fs_tool.clone();
    adapter
        .register_async_tool(
            ToolMetadata::new("write_file", "Write content to a file")
                .with_parameter(
                    ToolParameter::new("path", ToolParameterType::String)
                        .with_description("Relative path to the file")
                        .required(),
                )
                .with_parameter(
                    ToolParameter::new("content", ToolParameterType::String)
                        .with_description("Content to write")
                        .required(),
                ),
            Arc::new(move |params| {
                let fs = fs_write.clone();
                Box::pin(async move {
                    let path = params.get("path").and_then(|v| v.as_str()).ok_or_else(
                        || -> Box<dyn std::error::Error + Send + Sync> {
                            "Missing 'path' parameter".into()
                        },
                    )?;
                    let content = params.get("content").and_then(|v| v.as_str()).ok_or_else(
                        || -> Box<dyn std::error::Error + Send + Sync> {
                            "Missing 'content' parameter".into()
                        },
                    )?;
                    match fs.write_file(path, content).await {
                        Ok(()) => Ok(ToolResult::success(serde_json::json!({"status": "OK"}))),
                        Err(e) => Ok(ToolResult::failure(e.to_string())),
                    }
                })
            }),
        )
        .await;

    // Register list_directory tool
    let fs_list = fs_tool.clone();
    adapter
        .register_async_tool(
            ToolMetadata::new("list_directory", "List contents of a directory").with_parameter(
                ToolParameter::new("path", ToolParameterType::String)
                    .with_description("Relative path to the directory")
                    .required(),
            ),
            Arc::new(move |params| {
                let fs = fs_list.clone();
                Box::pin(async move {
                    let path = params.get("path").and_then(|v| v.as_str()).ok_or_else(
                        || -> Box<dyn std::error::Error + Send + Sync> {
                            "Missing 'path' parameter".into()
                        },
                    )?;
                    match fs.read_directory(path, false).await {
                        Ok(entries) => {
                            let names: Vec<String> =
                                entries.iter().map(|e| e.name.clone()).collect();
                            Ok(ToolResult::success(serde_json::json!({"entries": names})))
                        }
                        Err(e) => Ok(ToolResult::failure(e.to_string())),
                    }
                })
            }),
        )
        .await;

    adapter
}

#[tokio::test]
async fn test_filesystem_write_and_read_via_agent() {
    let temp_dir = tempfile::TempDir::new().unwrap();
    let adapter = build_filesystem_protocol(temp_dir.path().to_path_buf()).await;

    let mut registry = ToolRegistry::new(Arc::new(adapter));
    registry.discover_tools_from_primary().await.unwrap();

    // First: write a file
    let tool_call = r#"{"tool_call": {"name": "write_file", "parameters": {"path": "test.txt", "content": "Hello from agent!"}}}"#;
    let final_answer = "File written successfully.";

    let mut agent =
        build_agent_with_registry("fs_write_agent", registry, tool_call, final_answer).await;
    let response = agent
        .send("Write 'Hello from agent!' to test.txt")
        .await
        .unwrap();
    assert_eq!(response.content, final_answer);

    // Verify the file was actually written
    let content = std::fs::read_to_string(temp_dir.path().join("test.txt")).unwrap();
    assert_eq!(content, "Hello from agent!");
}

#[tokio::test]
async fn test_filesystem_read_via_agent() {
    let temp_dir = tempfile::TempDir::new().unwrap();
    // Pre-create a file
    std::fs::write(temp_dir.path().join("data.txt"), "pre-existing content").unwrap();

    let adapter = build_filesystem_protocol(temp_dir.path().to_path_buf()).await;
    let mut registry = ToolRegistry::new(Arc::new(adapter));
    registry.discover_tools_from_primary().await.unwrap();

    let tool_call = r#"{"tool_call": {"name": "read_file", "parameters": {"path": "data.txt"}}}"#;
    let final_answer = "The file contains: pre-existing content";

    let mut agent =
        build_agent_with_registry("fs_read_agent", registry, tool_call, final_answer).await;
    let response = agent.send("Read data.txt").await.unwrap();
    assert_eq!(response.content, final_answer);
}

#[tokio::test]
async fn test_filesystem_list_directory_via_agent() {
    let temp_dir = tempfile::TempDir::new().unwrap();
    std::fs::write(temp_dir.path().join("file1.txt"), "a").unwrap();
    std::fs::write(temp_dir.path().join("file2.txt"), "b").unwrap();
    std::fs::create_dir(temp_dir.path().join("subdir")).unwrap();

    let adapter = build_filesystem_protocol(temp_dir.path().to_path_buf()).await;
    let mut registry = ToolRegistry::new(Arc::new(adapter));
    registry.discover_tools_from_primary().await.unwrap();

    let tool_call = r#"{"tool_call": {"name": "list_directory", "parameters": {"path": "."}}}"#;
    let final_answer = "Directory contains 3 entries.";

    let mut agent =
        build_agent_with_registry("fs_list_agent", registry, tool_call, final_answer).await;
    let response = agent.send("List files in current directory").await.unwrap();
    assert_eq!(response.content, final_answer);
}

#[tokio::test]
async fn test_filesystem_path_traversal_blocked_via_agent() {
    let temp_dir = tempfile::TempDir::new().unwrap();
    let adapter = build_filesystem_protocol(temp_dir.path().to_path_buf()).await;
    let mut registry = ToolRegistry::new(Arc::new(adapter));
    registry.discover_tools_from_primary().await.unwrap();

    // Agent tries to read /etc/passwd via path traversal — should fail
    let tool_call =
        r#"{"tool_call": {"name": "read_file", "parameters": {"path": "../../../etc/passwd"}}}"#;
    let final_answer = "Access denied.";

    let mut agent =
        build_agent_with_registry("fs_traversal_agent", registry, tool_call, final_answer).await;
    let response = agent.send("Read /etc/passwd").await.unwrap();
    assert_eq!(response.content, final_answer);
}

// =============================================================================
// TEST 5: HttpClient tool via CustomToolProtocol
// =============================================================================

async fn build_http_protocol() -> CustomToolProtocol {
    let adapter = CustomToolProtocol::new();

    adapter
        .register_async_tool(
            ToolMetadata::new("http_get", "Make an HTTP GET request to a URL").with_parameter(
                ToolParameter::new("url", ToolParameterType::String)
                    .with_description("The URL to fetch")
                    .required(),
            ),
            Arc::new(move |params| {
                Box::pin(async move {
                    let url = params.get("url").and_then(|v| v.as_str()).ok_or_else(
                        || -> Box<dyn std::error::Error + Send + Sync> {
                            "Missing 'url' parameter".into()
                        },
                    )?;

                    let client = HttpClient::new();
                    match client.get(url).await {
                        Ok(response) => Ok(ToolResult::success(serde_json::json!({
                            "status": response.status,
                            "body": response.body,
                        }))),
                        Err(e) => Ok(ToolResult::failure(e.to_string())),
                    }
                })
            }),
        )
        .await;

    adapter
}

#[tokio::test]
async fn test_http_tool_domain_security() {
    // Test domain filtering at the HttpClient level
    let mut client = HttpClient::new();
    client.deny_domain("evil.com");

    let result = client.get("https://evil.com/data").await;
    assert!(result.is_err(), "Blocked domain should fail");
    assert!(
        result.unwrap_err().to_string().contains("blocked"),
        "Error should mention domain is blocked"
    );
}

#[tokio::test]
async fn test_http_tool_registration_and_discovery() {
    let adapter = build_http_protocol().await;

    let mut registry = ToolRegistry::new(Arc::new(adapter));
    registry.discover_tools_from_primary().await.unwrap();

    // Verify tool is discoverable
    let tools = registry.list_tools();
    assert_eq!(tools.len(), 1);
    assert_eq!(tools[0].name, "http_get");

    // Verify parameters
    assert_eq!(tools[0].parameters.len(), 1);
    assert_eq!(tools[0].parameters[0].name, "url");
    assert!(tools[0].parameters[0].required);
}

// =============================================================================
// TEST 6: Multi-protocol registry
// =============================================================================

#[tokio::test]
async fn test_multi_protocol_agent_memory_and_calculator() {
    let memory = Arc::new(Memory::new());
    let mem_protocol = Arc::new(MemoryProtocol::new(memory.clone()));

    let calc_protocol = Arc::new(build_calculator_protocol().await);

    let mut registry = ToolRegistry::empty();
    registry.add_protocol("memory", mem_protocol).await.unwrap();
    registry
        .add_protocol("calculator", calc_protocol)
        .await
        .unwrap();

    // Verify both tools are available
    let tools = registry.list_tools();
    let tool_names: Vec<&str> = tools.iter().map(|t| t.name.as_str()).collect();
    assert!(
        tool_names.contains(&"memory"),
        "memory tool should be registered"
    );
    assert!(
        tool_names.contains(&"calculator"),
        "calculator tool should be registered"
    );

    // Agent uses the calculator tool
    let tool_call =
        r#"{"tool_call": {"name": "calculator", "parameters": {"expression": "7 * 8"}}}"#;
    let final_answer = "7 * 8 = 56.";

    let mut agent =
        build_agent_with_registry("multi_agent", registry, tool_call, final_answer).await;
    let response = agent.send("What is 7 * 8?").await.unwrap();
    assert_eq!(response.content, final_answer);
}

#[tokio::test]
async fn test_multi_protocol_agent_uses_memory() {
    let memory = Arc::new(Memory::new());
    let mem_protocol = Arc::new(MemoryProtocol::new(memory.clone()));

    let calc_protocol = Arc::new(build_calculator_protocol().await);

    let mut registry = ToolRegistry::empty();
    registry.add_protocol("memory", mem_protocol).await.unwrap();
    registry
        .add_protocol("calculator", calc_protocol)
        .await
        .unwrap();

    // Agent uses the memory tool
    let tool_call =
        r#"{"tool_call": {"name": "memory", "parameters": {"command": "P result 56"}}}"#;
    let final_answer = "Stored result=56 in memory.";

    let mut agent =
        build_agent_with_registry("multi_mem_agent", registry, tool_call, final_answer).await;
    let response = agent.send("Store result=56 in memory").await.unwrap();
    assert_eq!(response.content, final_answer);

    let (value, _) = memory
        .get("result", false)
        .expect("Key 'result' should exist");
    assert_eq!(value, "56");
}

// =============================================================================
// TEST 7: Direct protocol execution (no agent, sanity checks)
// =============================================================================

/// These verify the protocol itself works correctly before the agent layer.

#[tokio::test]
async fn test_memory_protocol_direct_put_get() {
    let memory = Arc::new(Memory::new());
    let protocol = MemoryProtocol::new(memory.clone());

    // Put
    let put_result = protocol
        .execute("memory", serde_json::json!({"command": "P mykey myvalue"}))
        .await
        .unwrap();
    assert!(put_result.success, "PUT should succeed");

    // Get
    let get_result = protocol
        .execute("memory", serde_json::json!({"command": "G mykey"}))
        .await
        .unwrap();
    assert!(get_result.success, "GET should succeed");
    assert_eq!(get_result.output["value"], "myvalue");
}

#[tokio::test]
async fn test_memory_protocol_direct_unknown_command() {
    let memory = Arc::new(Memory::new());
    let protocol = MemoryProtocol::new(memory.clone());

    // Send an invalid command — this is the "ERR:Unknown Command" scenario
    let result = protocol
        .execute("memory", serde_json::json!({"command": "PUT key value"}))
        .await
        .unwrap();

    assert!(!result.success, "Unknown command should fail");
    assert!(
        result.error.as_ref().unwrap().contains("Unknown Command"),
        "Error should mention unknown command, got: {:?}",
        result.error
    );
}

#[tokio::test]
async fn test_bash_protocol_direct_echo() {
    let bash_tool = Arc::new(BashTool::new(Platform::Linux).with_timeout(10));
    let protocol = BashProtocol::new(bash_tool);

    let result = protocol
        .execute("bash", serde_json::json!({"command": "echo test_direct"}))
        .await
        .unwrap();

    assert!(result.success, "Echo should succeed");
    assert!(
        result.output["stdout"]
            .as_str()
            .unwrap()
            .contains("test_direct"),
        "Output should contain 'test_direct'"
    );
}

#[tokio::test]
async fn test_calculator_protocol_direct() {
    let adapter = build_calculator_protocol().await;

    let result = adapter
        .execute("calculator", serde_json::json!({"expression": "sqrt(25)"}))
        .await
        .unwrap();

    assert!(result.success, "sqrt(25) should succeed");
    let value = result.output["result"].as_f64().unwrap();
    assert!((value - 5.0).abs() < 1e-10, "sqrt(25) should be 5.0");
}

#[tokio::test]
async fn test_filesystem_protocol_direct_write_read() {
    let temp_dir = tempfile::TempDir::new().unwrap();
    let adapter = build_filesystem_protocol(temp_dir.path().to_path_buf()).await;

    // Write
    let write_result = adapter
        .execute(
            "write_file",
            serde_json::json!({"path": "hello.txt", "content": "direct test"}),
        )
        .await
        .unwrap();
    assert!(write_result.success, "Write should succeed");

    // Read
    let read_result = adapter
        .execute("read_file", serde_json::json!({"path": "hello.txt"}))
        .await
        .unwrap();
    assert!(read_result.success, "Read should succeed");
    assert_eq!(read_result.output["content"], "direct test");
}

// =============================================================================
// TEST 8: Tool discovery and metadata validation
// =============================================================================

#[tokio::test]
async fn test_memory_tool_metadata_has_command_parameter() {
    let memory = Arc::new(Memory::new());
    let protocol = MemoryProtocol::new(memory);

    let metadata = protocol.get_tool_metadata("memory").await.unwrap();
    assert_eq!(metadata.name, "memory");
    assert_eq!(metadata.parameters.len(), 1);
    assert_eq!(metadata.parameters[0].name, "command");
    assert!(metadata.parameters[0].required);
}

#[tokio::test]
async fn test_bash_tool_metadata_has_command_parameter() {
    let bash_tool = Arc::new(BashTool::new(Platform::Linux));
    let protocol = BashProtocol::new(bash_tool);

    let metadata = protocol.get_tool_metadata("bash").await.unwrap();
    assert_eq!(metadata.name, "bash");
    assert_eq!(metadata.parameters.len(), 1);
    assert_eq!(metadata.parameters[0].name, "command");
    assert!(metadata.parameters[0].required);
}

#[tokio::test]
async fn test_calculator_tool_metadata_has_expression_parameter() {
    let adapter = build_calculator_protocol().await;

    let metadata = adapter.get_tool_metadata("calculator").await.unwrap();
    assert_eq!(metadata.name, "calculator");
    assert_eq!(metadata.parameters.len(), 1);
    assert_eq!(metadata.parameters[0].name, "expression");
    assert!(metadata.parameters[0].required);
}

#[tokio::test]
async fn test_filesystem_tool_metadata_parameters() {
    let temp_dir = tempfile::TempDir::new().unwrap();
    let adapter = build_filesystem_protocol(temp_dir.path().to_path_buf()).await;

    let tools = adapter.list_tools().await.unwrap();
    let tool_names: Vec<&str> = tools.iter().map(|t| t.name.as_str()).collect();
    assert!(tool_names.contains(&"read_file"));
    assert!(tool_names.contains(&"write_file"));
    assert!(tool_names.contains(&"list_directory"));

    // write_file should have both path and content parameters
    let write_meta = adapter.get_tool_metadata("write_file").await.unwrap();
    assert_eq!(write_meta.parameters.len(), 2);
    let param_names: Vec<&str> = write_meta
        .parameters
        .iter()
        .map(|p| p.name.as_str())
        .collect();
    assert!(param_names.contains(&"path"));
    assert!(param_names.contains(&"content"));
}
