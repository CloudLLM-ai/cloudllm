//! Example: Agent with Multiple MCP Servers
//!
//! This example demonstrates how an agent can connect to multiple MCP servers
//! using the enhanced ToolRegistry with multi-protocol support.
//!
//! The agent transparently accesses tools from multiple sources (local protocols,
//! remote MCP servers, etc.) as if they were all available locally.
//!
//! # Architecture
//!
//! ```text
//! Agent
//!   │
//!   └─ ToolRegistry (Multi-Protocol)
//!       ├─ Protocol: "local" (CustomToolProtocol)
//!       │   ├─ memory tool (put, get, list, delete)
//!       │   └─ bash tool (execute commands)
//!       │
//!       ├─ Protocol: "youtube" (McpClientProtocol)
//!       │   ├─ youtube_search tool (query videos)
//!       │   └─ youtube_get_transcript tool (get transcripts)
//!       │
//!       └─ Protocol: "github" (McpClientProtocol)
//!           ├─ github_search_repos tool
//!           └─ github_get_issues tool
//! ```
//!
//! # Usage
//!
//! This is a demonstration example showing how to build a multi-MCP agent.
//! In a real scenario, you would:
//!
//! 1. Start local MCP servers exposing their tools on HTTP endpoints
//! 2. Create an agent with an empty ToolRegistry
//! 3. Add each MCP server via `registry.add_protocol(name, endpoint)`
//! 4. Use the agent with all tools available transparently
//!
//! ```bash
//! # Terminal 1: Start the local MCP server
//! cargo run --example mcp_server_local
//!
//! # Terminal 2: Run this agent
//! cargo run --example multi_mcp_agent
//! ```

use std::sync::Arc;

use cloudllm::clients::openai::{Model, OpenAIClient};
use cloudllm::council::Agent;
use cloudllm::tool_protocol::ToolRegistry;
use cloudllm::tool_protocols::McpClientProtocol;

/// Example showing how to create an agent with multiple MCP servers
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    cloudllm::init_logger();

    println!("=== Multi-MCP Agent Example ===\n");

    // Step 1: Create the LLM client
    println!("Step 1: Creating OpenAI client...");
    let api_key = std::env::var("OPEN_AI_SECRET")?;
    let client = Arc::new(OpenAIClient::new_with_model_enum(&api_key, Model::GPT41Nano));

    // Step 2: Create an empty ToolRegistry for multi-protocol support
    println!("\nStep 2: Creating multi-protocol ToolRegistry...");
    let mut registry = ToolRegistry::empty();

    // Step 3: Add multiple MCP server protocols
    println!("Step 3: Registering MCP server protocols...\n");

    // Add local MCP server (running on port 8080)
    println!("  - Adding local MCP server (http://localhost:8080)");
    println!("    Expected tools: memory, bash");
    let local_protocol = Arc::new(McpClientProtocol::new(
        "http://localhost:8080".to_string(),
    ));
    match registry.add_protocol("local", local_protocol).await {
        Ok(_) => println!("    ✓ Local server connected"),
        Err(e) => println!("    ⚠ Could not connect to local server: {}", e),
    }

    // Add remote YouTube MCP server (running on port 8081)
    println!("\n  - Adding remote YouTube MCP server (http://youtube-mcp.example.com:8081)");
    println!("    Expected tools: youtube_search, youtube_get_transcript");
    let youtube_protocol = Arc::new(McpClientProtocol::new(
        "http://youtube-mcp.example.com:8081".to_string(),
    ));
    match registry.add_protocol("youtube", youtube_protocol).await {
        Ok(_) => println!("    ✓ YouTube server connected"),
        Err(e) => println!("    ⚠ Could not connect to YouTube server: {}", e),
    }

    // Add remote GitHub MCP server (running on port 8082)
    println!("\n  - Adding remote GitHub MCP server (http://github-mcp.example.com:8082)");
    println!("    Expected tools: github_search_repos, github_get_issues");
    let github_protocol = Arc::new(McpClientProtocol::new(
        "http://github-mcp.example.com:8082".to_string(),
    ));
    match registry.add_protocol("github", github_protocol).await {
        Ok(_) => println!("    ✓ GitHub server connected"),
        Err(e) => println!("    ⚠ Could not connect to GitHub server: {}", e),
    }

    // Step 4: Display available tools from all servers
    println!("\nStep 4: Listing available tools from all protocols...");
    let tools = registry.list_tools();
    if tools.is_empty() {
        println!("  No tools available (MCP servers may not be running)");
        println!("\n  To test this example, you would need to:");
        println!("    1. Start the local MCP server: cargo run --example mcp_server_local");
        println!("    2. Update the YouTube and GitHub server URLs to valid endpoints");
    } else {
        println!("  Available tools:");
        for tool_meta in tools {
            println!("    - {} ({})", tool_meta.name, tool_meta.description);
            if let Some(protocol) = registry.get_tool_protocol(&tool_meta.name) {
                println!("      [from: {}]", protocol);
            }
        }
    }

    // Step 5: Show protocol information
    println!("\nStep 5: Registered protocols:");
    for protocol_name in registry.list_protocols() {
        println!("  - {}", protocol_name);
    }

    // Step 6: Create Agent with access to all tools
    println!("\nStep 6: Creating agent with access to all tools...");
    let mut agent = Agent::new("research-agent", "Research Agent", client);

    agent = agent
        .with_expertise("Finding information using multiple sources")
        .with_personality("Curious and methodical");

    // Attach the multi-protocol registry to the agent
    agent = agent.with_tools(Arc::new(registry));

    // Step 7: Example agent interaction
    println!("\n=== Agent Capabilities ===\n");

    println!("The agent now has access to tools from multiple MCP servers:");
    println!();
    println!("From 'local' MCP server:");
    println!("  - Use 'memory' to store/retrieve research notes");
    println!("  - Use 'bash' to process data locally");
    println!();
    println!("From 'youtube' MCP server:");
    println!("  - Use 'youtube_search' to find videos");
    println!("  - Use 'youtube_get_transcript' to get video transcripts");
    println!();
    println!("From 'github' MCP server:");
    println!("  - Use 'github_search_repos' to find repositories");
    println!("  - Use 'github_get_issues' to retrieve issues");

    println!("\n=== Example Workflow ===\n");

    println!("Agent instruction: \"Search GitHub for Rust projects and summarize top issues\"\n");
    println!("The agent would then:");
    println!("  1. Call 'github_search_repos' (routed to 'github' server)");
    println!("  2. Retrieve repo details and call 'github_get_issues'");
    println!("  3. Process results and store summary in 'memory' tool (routed to 'local' server)");
    println!("  4. Return findings to the user");

    println!("\n=== How Tool Routing Works ===\n");

    println!("When the agent calls a tool:");
    println!("  1. Agent calls registry.execute_tool(tool_name, params)");
    println!("  2. Registry looks up which protocol owns this tool");
    println!("  3. Registry forwards execute() to the correct MCP server");
    println!("  4. Result is returned to agent transparently");

    println!("\nThis allows agents to seamlessly orchestrate across multiple sources!");

    Ok(())
}
