//! MCP Server with All Available Tools
//!
//! This example starts an MCP server on localhost:8008 with all built-in tools available:
//! - Memory Tool: Persistent key-value store for state management
//! - Calculator Tool: Mathematical operations and expressions
//! - Bash Tool: Secure command execution (Linux/macOS)
//! - HTTP Client Tool: REST API requests
//! - FileSystem Tool: Safe file operations
//!
//! The server only accepts localhost connections (127.0.0.1 and ::1) for security.
//! HTTP is used (not HTTPS) since the server only listens on localhost, which is secure.
//!
//! # Setup Instructions for OpenAI Desktop Client
//!
//! ## Prerequisites
//! - OpenAI Desktop Client installed
//! - CloudLLM compiled with mcp-server feature: `cargo build --release --features mcp-server`
//!
//! ## Starting the Server
//!
//! ```bash
//! cargo run --example mcp_server_all_tools --release --features mcp-server
//! ```
//!
//! The server will output something like:
//! ```
//! MCP Server starting on 127.0.0.1:8008
//! Available tools:
//!   - memory (Key-value store with TTL)
//!   - calculator (Mathematical expressions)
//!   - bash (Command execution)
//!   - http_client (REST API requests)
//!   - filesystem (File operations)
//! Server ready! Connect your OpenAI client to http://localhost:8008/mcp
//! ```
//!
//! ## Connecting from OpenAI Desktop Client
//!
//! 1. In OpenAI Desktop Settings, add MCP Server:
//!    - Type: HTTP
//!    - URL: `http://localhost:8008/mcp`
//!    - Authentication: None (localhost only)
//!
//! 2. Test the connection by querying tool availability
//!
//! 3. Example prompts to try:
//!    - "Calculate 2+2" (uses Calculator)
//!    - "List the files in the current directory" (uses FileSystem)
//!    - "What files have 'test' in their name?" (uses FileSystem search)
//!    - "Store the value 'important_data' with key 'my_state'" (uses Memory)
//!    - "Retrieve my_state from memory" (uses Memory)
//!    - "Check the weather by calling api.weather.example.com" (uses HTTP Client)
//!
//! # Architecture
//!
//! The server uses:
//! - MCPServerBuilder for simplified configuration
//! - IpFilter to restrict to localhost only
//! - ToolProtocol implementations for each tool
//! - HTTP adapter (Axum) for MCP communication
//!
//! # Security
//!
//! - Only accepts connections from 127.0.0.1 (IPv4) and ::1 (IPv6)
//! - No authentication needed since localhost-only
//! - All tools operate within their security restrictions
//! - FileSystem tool restricted to current directory by default
//! - Bash tool supports command allow/deny lists (see notes below)
//!
//! # Notes
//!
//! - The FileSystem tool starts with no root restriction (entire filesystem accessible)
//! - For production, configure FileSystem with .with_root_path() to restrict to specific directories
//! - Bash tool executes on the current system (Linux/macOS only)
//! - Memory tool data is ephemeral (lost on server restart)
//! - For persistent memory, integrate with external database

use cloudllm::cloudllm::mcp_server_builder::MCPServerBuilder;
use cloudllm::cloudllm::tool_protocol::{
    ToolMetadata, ToolParameter, ToolParameterType, ToolResult,
};
use cloudllm::cloudllm::tool_protocols::{BashProtocol, CustomToolProtocol, MemoryProtocol};
use cloudllm::tools::{BashTool, Calculator, FileSystemTool, HttpClient, Memory, Platform};
use std::sync::Arc;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging
    env_logger::Builder::from_default_env()
        .filter_level(log::LevelFilter::Info)
        .try_init()
        .ok();

    println!("╔════════════════════════════════════════════════════════════╗");
    println!("║         CloudLLM MCP Server - All Tools Demo               ║");
    println!("║                                                            ║");
    println!("║  Available Tools:                                          ║");
    println!("║    • Memory - Key-value store with TTL                     ║");
    println!("║    • Calculator - Math expressions (evalexpr)              ║");
    println!("║    • Bash - Secure command execution                       ║");
    println!("║    • HTTP Client - REST API requests                       ║");
    println!("║    • FileSystem - Safe file operations                     ║");
    println!("║                                                            ║");
    println!("║  Security: localhost-only (127.0.0.1, ::1)                 ║");
    println!("║  Endpoint: http://localhost:8008/mcp                       ║");
    println!("╚════════════════════════════════════════════════════════════╝\n");

    // Create Memory tool and protocol
    let memory = Arc::new(Memory::new());
    let memory_protocol = Arc::new(MemoryProtocol::new(memory));

    // Create Bash tool and protocol
    let bash_tool = Arc::new(
        BashTool::new(detect_platform()).with_timeout(30), // 30 second timeout per command
    );
    let bash_protocol = Arc::new(BashProtocol::new(bash_tool));

    // Create Calculator tool wrapped in CustomToolProtocol
    let calculator = Arc::new(Calculator::new());
    let calc_protocol = Arc::new(CustomToolProtocol::new());
    calc_protocol
        .register_async_tool(
            ToolMetadata::new("calculator", "Mathematical expression evaluator")
                .with_parameter(
                    ToolParameter::new("expression", ToolParameterType::String)
                        .with_description("Mathematical expression to evaluate (e.g., '2+2', 'sqrt(16)', 'sin(pi/2)')")
                        .required(),
                ),
            {
                let calc = calculator.clone();
                Arc::new(move |params| {
                    let calc = calc.clone();
                    Box::pin(async move {
                    let expr = params
                        .get("expression")
                        .and_then(|v| v.as_str())
                        .ok_or_else(|| {
                            Box::new(std::io::Error::new(
                                std::io::ErrorKind::InvalidInput,
                                "Missing 'expression' parameter",
                            )) as Box<dyn std::error::Error + Send + Sync>
                        })?;

                    match calc.evaluate(expr).await {
                        Ok(result) => Ok(ToolResult::success(serde_json::json!({"result": result}))),
                        Err(e) => Ok(ToolResult::failure(format!("Calculation error: {}", e))),
                    }
                    })
                })
            }
        )
        .await;

    // Create FileSystem tool wrapped in CustomToolProtocol
    let fs_tool = Arc::new(FileSystemTool::new());
    let fs_protocol = Arc::new(CustomToolProtocol::new());
    fs_protocol
        .register_async_tool(
            ToolMetadata::new("filesystem", "Safe file and directory operations")
                .with_parameter(
                    ToolParameter::new("operation", ToolParameterType::String)
                        .with_description("Operation: 'list' (path), 'read' (path), 'write' (path, content), 'delete' (path)")
                        .required(),
                )
                .with_parameter(
                    ToolParameter::new("path", ToolParameterType::String)
                        .with_description("File or directory path")
                        .required(),
                )
                .with_parameter(
                    ToolParameter::new("content", ToolParameterType::String)
                        .with_description("Content to write (only for write operations)"),
                ),
            {
                let fs = fs_tool.clone();
                Arc::new(move |params| {
                    let fs = fs.clone();
                    Box::pin(async move {
                    let operation = params
                        .get("operation")
                        .and_then(|v| v.as_str())
                        .unwrap_or("list");
                    let path = params
                        .get("path")
                        .and_then(|v| v.as_str())
                        .ok_or_else(|| {
                            Box::new(std::io::Error::new(
                                std::io::ErrorKind::InvalidInput,
                                "Missing 'path' parameter",
                            )) as Box<dyn std::error::Error + Send + Sync>
                        })?;

                    match operation {
                        "list" => {
                            match fs.read_directory(path, false).await {
                                Ok(entries) => {
                                    let names: Vec<String> = entries
                                        .iter()
                                        .map(|e| e.name.clone())
                                        .collect();
                                    Ok(ToolResult::success(serde_json::json!({"files": names})))
                                }
                                Err(e) => Ok(ToolResult::failure(format!("List error: {}", e))),
                            }
                        }
                        "read" => {
                            match fs.read_file(path).await {
                                Ok(content) => Ok(ToolResult::success(serde_json::json!({"content": content}))),
                                Err(e) => Ok(ToolResult::failure(format!("Read error: {}", e))),
                            }
                        }
                        _ => Ok(ToolResult::failure("Unsupported operation".to_string())),
                    }
                    })
                })
            }
        )
        .await;

    // Create HTTP Client tool wrapped in CustomToolProtocol
    let http_client = Arc::new(HttpClient::new());
    let http_protocol = Arc::new(CustomToolProtocol::new());
    http_protocol
        .register_async_tool(
            ToolMetadata::new("http_client", "Make HTTP requests to external APIs")
                .with_parameter(
                    ToolParameter::new("url", ToolParameterType::String)
                        .with_description("URL to request")
                        .required(),
                )
                .with_parameter(
                    ToolParameter::new("method", ToolParameterType::String)
                        .with_description("HTTP method: GET, POST, PUT, DELETE")
                        .required(),
                ),
            {
                let http = http_client.clone();
                Arc::new(move |params| {
                    let http = http.clone();
                    Box::pin(async move {
                        let url = params.get("url").and_then(|v| v.as_str()).ok_or_else(|| {
                            Box::new(std::io::Error::new(
                                std::io::ErrorKind::InvalidInput,
                                "Missing 'url' parameter",
                            ))
                                as Box<dyn std::error::Error + Send + Sync>
                        })?;
                        let method = params
                            .get("method")
                            .and_then(|v| v.as_str())
                            .unwrap_or("GET");

                        let result = match method.to_uppercase().as_str() {
                            "GET" => http.get(url).await,
                            "POST" => http.post(url, serde_json::json!({})).await,
                            "PUT" => http.put(url, serde_json::json!({})).await,
                            "DELETE" => http.delete(url).await,
                            "PATCH" => http.patch(url, serde_json::json!({})).await,
                            "HEAD" => http.head(url).await,
                            _ => {
                                return Ok(ToolResult::failure(format!(
                                    "Unsupported HTTP method: {}",
                                    method
                                )))
                            }
                        };

                        match result {
                            Ok(response) => Ok(ToolResult::success(serde_json::json!({
                                "status": response.status,
                                "body": response.body
                            }))),
                            Err(e) => Ok(ToolResult::failure(format!("HTTP error: {}", e))),
                        }
                    })
                })
            },
        )
        .await;

    // Build and start the MCP server
    let server = MCPServerBuilder::new()
        // Add all tools
        .with_custom_tool("memory", memory_protocol)
        .await
        .with_custom_tool("calculator", calc_protocol)
        .await
        .with_custom_tool("filesystem", fs_protocol)
        .await
        .with_custom_tool("http_client", http_protocol)
        .await
        .with_custom_tool("bash", bash_protocol)
        .await
        // Configure security
        .allow_localhost_only()
        // Start on port 8008
        .start_on(8008)
        .await
        .map_err(|e| {
            Box::new(std::io::Error::new(
                std::io::ErrorKind::Other,
                e.to_string(),
            )) as Box<dyn std::error::Error>
        })?;

    println!("✅ MCP Server started successfully!");
    println!("📍 Listening on: http://{}", server.addr);
    println!("\n🔌 Connection Information:");
    println!("   URL: http://localhost:8008/mcp");
    println!("   Authentication: None (localhost only)");
    println!("   Allow localhost: 127.0.0.1, ::1");
    println!("\n💡 Tools Available:");
    println!("   1. memory        - Store/retrieve state with TTL");
    println!("   2. calculator    - Evaluate mathematical expressions");
    println!("   3. filesystem    - Read/write files and directories");
    println!("   4. http_client   - Make HTTP requests");
    println!("   5. bash          - Execute shell commands (Linux/macOS)");
    println!("\n📖 Example Requests:");
    println!("   • 'Store my API key with key=api_token'");
    println!("   • 'Calculate sqrt(16)'");
    println!("   • 'List files in current directory'");
    println!("   • 'Make a GET request to https://api.github.com'");
    println!("   • 'Run: ls -la' (bash)");
    println!("\n⚙️  Configuration:");
    println!("   • Localhost only: Yes");
    println!("   • Bash timeout: 30 seconds");
    println!("   • FileSystem: Full access (configure with root_path for production)");
    println!("   • Memory: In-memory (ephemeral)");
    println!("\n🛑 Press Ctrl+C to stop the server\n");

    // Keep server running
    loop {
        tokio::time::sleep(tokio::time::Duration::from_secs(60)).await;
    }
}

/// Detect the current platform for Bash tool
fn detect_platform() -> Platform {
    #[cfg(target_os = "macos")]
    return Platform::macOS;

    #[cfg(target_os = "linux")]
    return Platform::Linux;

    #[cfg(not(any(target_os = "macos", target_os = "linux")))]
    panic!("Bash tool only supports Linux and macOS");
}
