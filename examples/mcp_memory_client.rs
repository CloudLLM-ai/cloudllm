//! MCP Memory Client Example
//!
//! This example demonstrates how to use the McpMemoryProtocol to interact with
//! a remote Memory service via the Model Context Protocol (MCP).
//!
//! # Architecture
//!
//! The MCP Memory pattern consists of:
//! 1. A Memory service (typically running on a remote server)
//! 2. An MCP HTTP endpoint that exposes the Memory tool
//! 3. MCP Memory Clients that connect to the remote service
//!
//! # Prerequisites
//!
//! In a real deployment, you would need:
//! - A running MCP Memory Server (see mcp_memory_server example)
//! - Network connectivity to the server
//! - Proper configuration of the server endpoint

use cloudllm::tool_adapters::McpMemoryProtocol;
use cloudllm::tool_protocol::ToolProtocol;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    cloudllm::init_logger();

    println!("=== MCP Memory Client Example ===\n");

    // Create an MCP Memory Client pointing to a remote server
    // In a real scenario, this could be:
    // - http://memory-service.example.com:8080
    // - http://192.168.1.100:3000
    // - http://memory-cluster.region.cloud:443
    let client = McpMemoryProtocol::new("http://localhost:8080".to_string());

    println!("Client Configuration:");
    println!("  Endpoint: {}", client.endpoint());
    println!("  Protocol: {}\n", client.protocol_name());

    // Example 1: Store data with TTL
    println!("=== Example 1: Storing Data ===");
    println!("Command: P user:alice alice_data 3600");
    println!("This stores 'alice_data' for user 'alice' with 1-hour TTL\n");

    let store_params = serde_json::json!({
        "command": "P user:alice alice_data 3600"
    });

    match client.execute("memory", store_params).await {
        Ok(result) => {
            if result.success {
                println!("✓ Successfully stored data");
                println!("  Response: {}\n", result.output);
            } else {
                println!("✗ Failed to store data: {}\n", result.output);
            }
        }
        Err(e) => {
            println!("✗ Error executing command: {}", e);
            println!("  (This is expected - no MCP server is running)\n");
        }
    }

    // Example 2: Retrieve data
    println!("=== Example 2: Retrieving Data ===");
    println!("Command: G user:alice");
    println!("This retrieves data for key 'user:alice'\n");

    let get_params = serde_json::json!({
        "command": "G user:alice"
    });

    match client.execute("memory", get_params).await {
        Ok(result) => {
            if result.success {
                println!("✓ Successfully retrieved data");
                println!("  Response: {}\n", result.output);
            } else {
                println!("✗ Data not found: {}\n", result.output);
            }
        }
        Err(e) => {
            println!("✗ Error executing command: {}\n", e);
        }
    }

    // Example 3: List all keys with metadata
    println!("=== Example 3: Listing All Keys ===");
    println!("Command: L META");
    println!("This lists all stored keys with their metadata\n");

    let list_params = serde_json::json!({
        "command": "L META"
    });

    match client.execute("memory", list_params).await {
        Ok(result) => {
            if result.success {
                println!("✓ Successfully listed keys");
                println!("  Response: {}\n", result.output);
            } else {
                println!("✗ Failed to list keys: {}\n", result.output);
            }
        }
        Err(e) => {
            println!("✗ Error executing command: {}\n", e);
        }
    }

    // Example 4: Get memory statistics
    println!("=== Example 4: Memory Statistics ===");
    println!("Command: T A");
    println!("This retrieves total memory usage (all data)\n");

    let stats_params = serde_json::json!({
        "command": "T A"
    });

    match client.execute("memory", stats_params).await {
        Ok(result) => {
            if result.success {
                println!("✓ Successfully retrieved statistics");
                println!("  Response: {}\n", result.output);
            } else {
                println!("✗ Failed to get statistics: {}\n", result.output);
            }
        }
        Err(e) => {
            println!("✗ Error executing command: {}\n", e);
        }
    }

    // Example 5: Get tool metadata
    println!("=== Example 5: Tool Metadata ===");
    println!("Querying available tools from remote MCP server\n");

    match client.list_tools().await {
        Ok(tools) => {
            println!("✓ Successfully retrieved tool list");
            println!("  Available tools: {} ", tools.len());
            for tool in &tools {
                println!("    - {} ({})", tool.name, tool.description);
            }
            println!();
        }
        Err(e) => {
            println!("✗ Error retrieving tools: {}\n", e);
        }
    }

    // Example 6: Get specific tool metadata
    println!("=== Example 6: Specific Tool Metadata ===");
    println!("Querying metadata for 'memory' tool\n");

    match client.get_tool_metadata("memory").await {
        Ok(metadata) => {
            println!("✓ Successfully retrieved tool metadata");
            println!("  Tool: {}", metadata.name);
            println!("  Description: {}", metadata.description);
            println!("  Parameters:");
            for param in &metadata.parameters {
                println!(
                    "    - {} ({:?}) - {}",
                    param.name,
                    param.param_type,
                    param.description.as_deref().unwrap_or("no description")
                );
                if param.required {
                    println!("      [REQUIRED]");
                }
            }
            println!();
        }
        Err(e) => {
            println!("✗ Error retrieving metadata: {}\n", e);
        }
    }

    println!("=== Pattern: Distributed Agent Coordination ===\n");

    println!("Multi-Agent Scenario:");
    println!("  Agent A: Stores research findings at memory://research");
    println!("  Agent B: Reads findings and adds analysis at memory://analysis");
    println!("  Agent C: Reads both and makes decisions\n");

    println!("All agents connect to the same MCP Memory Server:");
    println!("  Agent A: McpMemoryProtocol::new(\"http://memory-server:8080\")");
    println!("  Agent B: McpMemoryProtocol::new(\"http://memory-server:8080\")");
    println!("  Agent C: McpMemoryProtocol::new(\"http://memory-server:8080\")\n");

    println!("Communication flow:");
    println!("  1. Agent A executes: \"P research important_findings 3600\"");
    println!("  2. Agent B executes: \"G research\"");
    println!("  3. Agent B executes: \"P analysis strategic_insights 3600\"");
    println!("  4. Agent C executes: \"L\" to see all stored information");
    println!("  5. Agent C executes: \"G analysis\" to read the analysis\n");

    println!("=== Configuration ===\n");

    println!("For different deployment scenarios:\n");

    println!("Local Development:");
    println!("  let client = McpMemoryProtocol::new(\"http://localhost:8080\".to_string());\n");

    println!("Private Network:");
    println!("  let client = McpMemoryProtocol::new(");
    println!("    \"http://192.168.1.100:3000\".to_string()");
    println!("  );\n");

    println!("Cloud Deployment:");
    println!("  let client = McpMemoryProtocol::new(");
    println!("    \"https://memory.example.com\".to_string()");
    println!("  );\n");

    println!("Custom Timeout (60 seconds):");
    println!("  let client = McpMemoryProtocol::with_timeout(");
    println!("    \"http://localhost:8080\".to_string(),");
    println!("    60");
    println!("  );\n");

    println!("=== Running This Example ===\n");

    println!("To test with a real server:");
    println!("  1. Start the MCP Memory Server (cargo run --example mcp_memory_server)");
    println!("  2. In another terminal: cargo run --example mcp_memory_client\n");

    println!("Note: This example shows the expected behavior and API design.");
    println!("Actual execution requires a running MCP Memory Server.\n");

    Ok(())
}
