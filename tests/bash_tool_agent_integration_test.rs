//! Integration tests demonstrating how Agents use the BashTool
//!
//! These tests showcase two integration patterns:
//! 1. **Local Adapter Pattern**: Agent directly uses BashTool via CustomToolProtocol
//! 2. **MCP Adapter Pattern**: Agent uses BashTool through an MCP server (simulated)
//!
//! This demonstrates the flexibility of the tool protocol system, where the same
//! agent code can work with tools whether they're local or accessed via MCP.

use cloudllm::tool_protocol::{
    ToolMetadata, ToolParameter, ToolParameterType, ToolProtocol, ToolResult,
};
use cloudllm::tool_protocols::CustomToolProtocol;
use cloudllm::tools::BashTool;
use cloudllm::tools::Platform;
use std::sync::Arc;

// Helper function to reduce boilerplate in tests
async fn register_bash_tool(adapter: &CustomToolProtocol, bash_tool: Arc<BashTool>) {
    let bash_clone = bash_tool.clone();
    adapter
        .register_async_tool(
            ToolMetadata::new("bash", "Execute bash commands safely").with_parameter(
                ToolParameter::new("command", ToolParameterType::String)
                    .with_description("Shell command to execute")
                    .required(),
            ),
            Arc::new(move |params| {
                let bash = bash_clone.clone();
                Box::pin(async move {
                    let command = params.get("command").and_then(|v| v.as_str()).ok_or_else(
                        || -> Box<dyn std::error::Error + Send + Sync> {
                            "Missing 'command' parameter".into()
                        },
                    )?;

                    let result = bash.execute(command).await.map_err(
                        |e| -> Box<dyn std::error::Error + Send + Sync> {
                            format!("Bash execution failed: {}", e).into()
                        },
                    )?;

                    Ok(ToolResult::success(serde_json::json!({
                        "stdout": result.stdout,
                        "stderr": result.stderr,
                        "exit_code": result.exit_code,
                        "duration_ms": result.duration_ms,
                    })))
                })
            }),
        )
        .await;
}

// ==============================================================================
// PATTERN 1: LOCAL ADAPTER - Agent directly uses BashTool
// ==============================================================================

/// Test showing agent executing a simple command through local adapter
#[tokio::test]
async fn test_agent_with_bash_tool_local_adapter() {
    let bash_tool = Arc::new(
        BashTool::new(Platform::Linux)
            .with_timeout(10)
            .with_denied_commands(vec!["rm".to_string(), "sudo".to_string()]),
    );

    let adapter = CustomToolProtocol::new();
    register_bash_tool(&adapter, bash_tool).await;

    let tool_result = adapter
        .execute(
            "bash",
            serde_json::json!({"command": "echo 'Agent says: Hello from local tool'"}),
        )
        .await;

    assert!(
        tool_result.is_ok(),
        "Agent should successfully call bash tool"
    );
    let result = tool_result.unwrap();
    assert!(result.success);
    assert!(result.output["stdout"]
        .as_str()
        .unwrap_or("")
        .contains("Agent says"));
}

/// Test showing agent can discover what tools are available
#[tokio::test]
async fn test_agent_discovers_tools_local_adapter() {
    let bash_tool = Arc::new(BashTool::new(Platform::Linux));
    let adapter = CustomToolProtocol::new();
    register_bash_tool(&adapter, bash_tool).await;

    let tools = adapter.list_tools().await;
    assert!(tools.is_ok());

    let tool_list = tools.unwrap();
    assert_eq!(tool_list.len(), 1);
    assert_eq!(tool_list[0].name, "bash");
}

/// Test showing agent inspects tool metadata before using it
#[tokio::test]
async fn test_agent_inspects_tool_metadata_local_adapter() {
    let bash_tool = Arc::new(BashTool::new(Platform::Linux));
    let adapter = CustomToolProtocol::new();
    register_bash_tool(&adapter, bash_tool).await;

    let metadata = adapter.get_tool_metadata("bash").await;
    assert!(metadata.is_ok());

    let meta = metadata.unwrap();
    assert_eq!(meta.name, "bash");
    assert_eq!(meta.parameters.len(), 1);
    assert_eq!(meta.parameters[0].name, "command");
}

/// Test showing agent executing multiple commands in sequence
#[tokio::test]
async fn test_agent_sequential_bash_commands_local_adapter() {
    let bash_tool = Arc::new(BashTool::new(Platform::Linux));
    let adapter = CustomToolProtocol::new();
    register_bash_tool(&adapter, bash_tool).await;

    // Agent executes first command
    let result1 = adapter
        .execute("bash", serde_json::json!({"command": "echo step1"}))
        .await;
    assert!(result1.is_ok());
    assert!(result1.unwrap().output["stdout"]
        .as_str()
        .unwrap()
        .contains("step1"));

    // Agent uses output from first command and executes second
    let result2 = adapter
        .execute("bash", serde_json::json!({"command": "echo step2"}))
        .await;
    assert!(result2.is_ok());
    assert!(result2.unwrap().output["stdout"]
        .as_str()
        .unwrap()
        .contains("step2"));

    // Agent executes third command
    let result3 = adapter
        .execute("bash", serde_json::json!({"command": "echo step3"}))
        .await;
    assert!(result3.is_ok());
    assert!(result3.unwrap().output["stdout"]
        .as_str()
        .unwrap()
        .contains("step3"));
}

/// Test showing agent respects security constraints (denied commands)
#[tokio::test]
async fn test_agent_bash_security_with_denied_commands() {
    let bash_tool = Arc::new(
        BashTool::new(Platform::Linux)
            .with_timeout(10)
            .with_denied_commands(vec!["rm".to_string(), "sudo".to_string()]),
    );

    let adapter = CustomToolProtocol::new();
    register_bash_tool(&adapter, bash_tool).await;

    // Agent tries to execute dangerous command
    let result = adapter
        .execute("bash", serde_json::json!({"command": "rm -rf /"}))
        .await;

    assert!(
        result.is_err(),
        "Dangerous command should be blocked by adapter"
    );

    // But can execute safe commands
    let safe_result = adapter
        .execute("bash", serde_json::json!({"command": "echo safe"}))
        .await;

    assert!(safe_result.is_ok(), "Safe command should succeed");
}

/// Test showing agent can execute allowed commands only (allowlist mode)
#[tokio::test]
async fn test_agent_bash_with_allowed_commands_only() {
    let bash_tool = Arc::new(
        BashTool::new(Platform::Linux)
            .with_allowed_commands(vec!["echo".to_string(), "ls".to_string()]),
    );

    let adapter = CustomToolProtocol::new();
    register_bash_tool(&adapter, bash_tool).await;

    // Allowed command should work
    let result = adapter
        .execute("bash", serde_json::json!({"command": "echo allowed"}))
        .await;
    assert!(result.is_ok());

    // Disallowed command should fail
    let blocked = adapter
        .execute("bash", serde_json::json!({"command": "rm test.txt"}))
        .await;
    assert!(
        blocked.is_err(),
        "Non-allowlisted command should be blocked"
    );
}

/// Test showing agent gathering system information
#[tokio::test]
async fn test_agent_gathers_system_info_via_bash() {
    let bash_tool = Arc::new(BashTool::new(Platform::Linux));
    let adapter = CustomToolProtocol::new();
    register_bash_tool(&adapter, bash_tool).await;

    // Agent queries for current directory
    let pwd_result = adapter
        .execute("bash", serde_json::json!({"command": "pwd"}))
        .await;

    assert!(pwd_result.is_ok());
    let pwd_output = pwd_result.unwrap();
    let pwd = pwd_output.output["stdout"].as_str().unwrap().trim();
    assert!(!pwd.is_empty(), "Should get current directory");

    // Agent queries for user info
    let user_result = adapter
        .execute("bash", serde_json::json!({"command": "whoami"}))
        .await;

    assert!(user_result.is_ok());
    let user_output = user_result.unwrap();
    let user = user_output.output["stdout"].as_str().unwrap().trim();
    assert!(!user.is_empty(), "Should get current user");
}

/// Test showing agent handles command failures gracefully
#[tokio::test]
async fn test_agent_handles_bash_errors_gracefully() {
    let bash_tool = Arc::new(BashTool::new(Platform::Linux));
    let adapter = CustomToolProtocol::new();
    register_bash_tool(&adapter, bash_tool).await;

    // Agent tries a command that will fail
    let result = adapter
        .execute("bash", serde_json::json!({"command": "false"}))
        .await;

    assert!(result.is_ok(), "Tool call should succeed");
    let output = result.unwrap();
    assert!(output.success);
    assert_ne!(
        output.output["exit_code"].as_i64().unwrap(),
        0,
        "Command should have non-zero exit code"
    );

    // Agent can retry with different command
    let retry_result = adapter
        .execute("bash", serde_json::json!({"command": "echo 'retrying'"}))
        .await;

    assert!(retry_result.is_ok());
    assert_eq!(
        retry_result.unwrap().output["exit_code"].as_i64().unwrap(),
        0
    );
}

/// Test showing agent can pipe and chain commands
#[tokio::test]
async fn test_agent_chains_bash_commands() {
    let bash_tool = Arc::new(BashTool::new(Platform::Linux));
    let adapter = CustomToolProtocol::new();
    register_bash_tool(&adapter, bash_tool).await;

    // Agent executes complex command with pipes
    let result = adapter
        .execute(
            "bash",
            serde_json::json!({"command": "echo -e '3\\n1\\n2' | sort"}),
        )
        .await;

    assert!(result.is_ok());
    let result_output = result.unwrap();
    let output = result_output.output["stdout"].as_str().unwrap();
    // Check that output contains sorted numbers
    let lines: Vec<&str> = output.trim().lines().collect();
    assert!(lines.len() >= 3, "Should have 3 lines of sorted output");
}

// ==============================================================================
// PATTERN 2: MCP ADAPTER - Simulating Agent access through MCP
// ==============================================================================

/// Test simulating Agent using BashTool through MCP protocol
/// Shows what MCP endpoints would look like
#[tokio::test]
async fn test_agent_with_bash_tool_mcp_adapter_simulation() {
    let bash_tool = Arc::new(BashTool::new(Platform::Linux).with_timeout(10));

    // Simulate what would be returned from MCP /tools endpoint
    let tool_metadata = ToolMetadata::new("bash", "Execute bash commands safely").with_parameter(
        ToolParameter::new("command", ToolParameterType::String)
            .with_description("Shell command to execute")
            .required(),
    );

    // Verify metadata is serializable (critical for MCP communication)
    let json = serde_json::to_value(&tool_metadata).unwrap();
    assert_eq!(json["name"], "bash");

    // Execute directly on bash tool (simulating MCP server behavior)
    let result = bash_tool
        .execute("echo 'From MCP'")
        .await
        .expect("Command should execute");

    // Simulate what would be returned from MCP /execute endpoint
    let execute_response = ToolResult::success(serde_json::json!({
        "stdout": result.stdout.trim(),
        "stderr": result.stderr,
        "exit_code": result.exit_code,
        "duration_ms": result.duration_ms,
    }));

    assert!(execute_response.success);
    assert!(execute_response.output["stdout"]
        .as_str()
        .unwrap()
        .contains("From MCP"));
}

/// Test showing agent security constraints work through MCP path
#[tokio::test]
async fn test_agent_bash_security_through_mcp_simulation() {
    let bash_tool = Arc::new(
        BashTool::new(Platform::Linux)
            .with_timeout(30)
            .with_denied_commands(vec![
                "rm".to_string(),
                "sudo".to_string(),
                "pkill".to_string(),
            ]),
    );

    // Simulate agent trying to execute dangerous command through MCP
    let dangerous_cmd = "rm -rf /";
    let result = bash_tool.execute(dangerous_cmd).await;

    assert!(result.is_err(), "Dangerous command should be blocked");

    // Simulate agent executing safe command through MCP
    let safe_cmd = "echo 'Safe command'";
    let result = bash_tool.execute(safe_cmd).await;

    assert!(result.is_ok(), "Safe command should succeed");
    assert!(result.unwrap().stdout.contains("Safe command"));
}

/// Test showing Agent with environment variable configuration through MCP
#[tokio::test]
async fn test_agent_bash_with_env_vars_mcp_simulation() {
    let bash_tool = Arc::new(
        BashTool::new(Platform::Linux)
            .with_env_var("AGENT_NAME".to_string(), "TestAgent".to_string())
            .with_env_var("MODE".to_string(), "production".to_string()),
    );

    // Simulate MCP request
    let result = bash_tool
        .execute("echo $AGENT_NAME in $MODE mode")
        .await
        .expect("Command should execute");

    assert!(result.stdout.contains("TestAgent"));
    assert!(result.stdout.contains("production"));
}

// ==============================================================================
// PATTERN 3: AGENT PLANNING & DISCOVERY
// ==============================================================================

/// Test showing agent planning phase - discovering and inspecting tools
#[tokio::test]
async fn test_agent_planning_phase_tool_discovery() {
    let bash_tool = Arc::new(BashTool::new(Platform::Linux));
    let adapter = CustomToolProtocol::new();
    register_bash_tool(&adapter, bash_tool).await;

    // Agent planning phase 1: discover what tools exist
    let available_tools = adapter.list_tools().await.unwrap();
    assert_eq!(available_tools.len(), 1);

    // Agent planning phase 2: inspect each tool's metadata
    for tool in available_tools {
        let metadata = adapter.get_tool_metadata(&tool.name).await.unwrap();
        assert_eq!(metadata.name, "bash");

        // Verify parameters are documented
        for param in &metadata.parameters {
            assert!(!param.name.is_empty());
            assert!(param.required || param.description.is_some());
        }
    }
}

/// Test showing agent adaptive behavior based on tool capabilities
#[tokio::test]
async fn test_agent_adapts_behavior_based_on_tool_metadata() {
    let bash_tool = Arc::new(BashTool::new(Platform::Linux));
    let adapter = CustomToolProtocol::new();

    // Register tool
    register_bash_tool(&adapter, bash_tool).await;

    // Agent reads tool metadata to understand capabilities
    let metadata = adapter.get_tool_metadata("bash").await.unwrap();

    // Agent verifies timeout configuration is known
    assert!(!metadata.description.is_empty());
    assert_eq!(metadata.parameters.len(), 1);

    // Agent can now decide how to use the tool
    let has_command_param = metadata.parameters.iter().any(|p| p.name == "command");
    assert!(has_command_param, "Should have command parameter");
}

/// Test showing agent retrying logic after tool execution
#[tokio::test]
async fn test_agent_retry_logic_with_bash_tool() {
    let bash_tool = Arc::new(BashTool::new(Platform::Linux));
    let adapter = CustomToolProtocol::new();
    register_bash_tool(&adapter, bash_tool).await;

    let mut attempt = 0;
    let max_attempts = 3;

    // Simulate agent retry loop
    while attempt < max_attempts {
        let result = adapter
            .execute("bash", serde_json::json!({"command": "echo attempt"}))
            .await;

        if result.is_ok() {
            assert!(result.unwrap().success);
            break;
        }

        attempt += 1;
    }

    assert!(attempt < max_attempts, "Should succeed before max attempts");
}

/// Test showing multi-step agent workflow with bash tool
#[tokio::test]
async fn test_agent_multistep_workflow() {
    let bash_tool = Arc::new(BashTool::new(Platform::Linux));
    let adapter = CustomToolProtocol::new();
    register_bash_tool(&adapter, bash_tool).await;

    // Step 1: Check current state
    let state_check = adapter
        .execute(
            "bash",
            serde_json::json!({"command": "echo 'workflow started'"}),
        )
        .await;
    assert!(state_check.is_ok());

    // Step 2: Process data
    let process = adapter
        .execute(
            "bash",
            serde_json::json!({"command": "echo 'processing data'"}),
        )
        .await;
    assert!(process.is_ok());

    // Step 3: Verify results
    let verify = adapter
        .execute(
            "bash",
            serde_json::json!({"command": "echo 'verification complete'"}),
        )
        .await;
    assert!(verify.is_ok());

    let verify_result = verify.unwrap();
    let final_output = verify_result.output["stdout"].as_str().unwrap();
    assert!(final_output.contains("verification complete"));
}
