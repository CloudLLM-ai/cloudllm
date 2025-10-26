# CloudLLM

<p align="center">
  <img src="https://github.com/CloudLLM-ai/cloudllm/blob/master/logo.png?raw=true" width="220" alt="CloudLLM logo" />
</p>

CloudLLM is a batteries-included Rust toolkit for working with remote Large Language Models.  It
provides:

* ergonomic provider clients built on a shared [`ClientWrapper`](https://docs.rs/cloudllm/latest/cloudllm/client_wrapper/index.html) trait,
* a stateful [`LLMSession`](https://docs.rs/cloudllm/latest/cloudllm/struct.LLMSession.html) that automates
  context trimming and token accounting,
* a multi-agent [`council`](https://docs.rs/cloudllm/latest/cloudllm/council/index.html) orchestration engine, and
* a protocol-agnostic tool interface with adapters for MCP and OpenAI function calling.

The entire public API is documented with _compilable_ examples—run `cargo doc --open` to browse the
crate-level manual.

---

## Installation

Add CloudLLM to your project:

```toml
[dependencies]
cloudllm = "0.5.0"
```

The crate targets `tokio` 1.x and Rust 1.70+.

---

## Quick start

### 1. Initialising a session

```rust,no_run
use std::sync::Arc;

use cloudllm::{init_logger, LLMSession, Role};
use cloudllm::clients::openai::{Model, OpenAIClient};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    init_logger();

    let api_key = std::env::var("OPEN_AI_SECRET")?;
    let client = OpenAIClient::new_with_model_enum(&api_key, Model::GPT41Nano);

    let mut session = LLMSession::new(
        Arc::new(client),
        "You write product update haikus.".to_owned(),
        8_192,
    );

    let reply = session
        .send_message(Role::User, "Announce the logging feature.".to_owned(), None)
        .await?;

    println!("Assistant: {}", reply.content);
    println!("Usage (tokens): {:?}", session.token_usage());
    Ok(())
}
```

### 2. Streaming tokens in real time

```rust,no_run
use cloudllm::{LLMSession, Role};
use cloudllm::clients::openai::{Model, OpenAIClient};
use futures_util::StreamExt;
use std::sync::Arc;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let api_key = std::env::var("OPEN_AI_SECRET")?;
    let client = Arc::new(OpenAIClient::new_with_model_enum(&api_key, Model::GPT41Mini));
    let mut session = LLMSession::new(client, "You think out loud.".into(), 16_000);

    if let Some(mut stream) = session
        .send_message_stream(Role::User, "Explain type erasure.".into(), None)
        .await? {
        while let Some(chunk) = stream.next().await {
            let chunk = chunk?;
            print!("{}", chunk.content);
            if let Some(reason) = chunk.finish_reason {
                println!("\n<terminated: {reason}>");
            }
        }
    }

    Ok(())
}
```

---

## Provider wrappers

CloudLLM ships wrappers for popular OpenAI-compatible services:

| Provider | Module | Notable constructors |
|----------|--------|----------------------|
| OpenAI   | `cloudllm::clients::openai`  | `OpenAIClient::new_with_model_enum`, `OpenAIClient::new_with_base_url` |
| Anthropic Claude | `cloudllm::clients::claude` | `ClaudeClient::new_with_model_enum` |
| Google Gemini | `cloudllm::clients::gemini` | `GeminiClient::new_with_model_enum` |
| xAI Grok | `cloudllm::clients::grok` | `GrokClient::new_with_model_enum` |

Providers share the [`ClientWrapper`](https://docs.rs/cloudllm/latest/cloudllm/client_wrapper/trait.ClientWrapper.html)
contract, so you can swap them without changing downstream code.

```rust,no_run
use cloudllm::ClientWrapper;
use cloudllm::clients::claude::{ClaudeClient, Model};
use cloudllm::client_wrapper::{Message, Role};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let key = std::env::var("ANTHROPIC_KEY")?;
    let claude = ClaudeClient::new_with_model_enum(&key, Model::ClaudeSonnet4);

    let response = claude
        .send_message(
            &[Message { role: Role::User, content: "Summarise rice fermentation.".into() }],
            None,
        )
        .await?;

    println!("{}", response.content);
    Ok(())
}
```

Every wrapper exposes token accounting via [`ClientWrapper::get_last_usage`](https://docs.rs/cloudllm/latest/cloudllm/client_wrapper/trait.ClientWrapper.html#method.get_last_usage).

---

## Tooling

CloudLLM provides a powerful, protocol-agnostic tool system that works seamlessly with agents and councils.
Tools enable agents to take actions beyond conversation—calculate values, query databases, call APIs, or
maintain state across sessions.

### Creating Tools

Define tools once, use them with any LLM provider:

```rust,no_run
use std::sync::Arc;

use cloudllm::tool_adapters::CustomToolAdapter;
use cloudllm::tool_protocol::{ToolMetadata, ToolParameter, ToolParameterType, ToolRegistry, ToolResult};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create a custom tool adapter
    let adapter = Arc::new(CustomToolAdapter::new());

    // Register a synchronous tool
    adapter
        .register_tool(
            ToolMetadata::new("area", "Compute the area of a rectangle")
                .with_parameter(
                    ToolParameter::new("width", ToolParameterType::Number)
                        .with_description("Width in units")
                        .required()
                )
                .with_parameter(
                    ToolParameter::new("height", ToolParameterType::Number)
                        .with_description("Height in units")
                        .required()
                ),
            Arc::new(|params| {
                let width = params["width"].as_f64().unwrap();
                let height = params["height"].as_f64().unwrap();
                Ok(ToolResult::success(serde_json::json!({"area": width * height})))
            }),
        )
        .await;

    // Create a registry and verify the tool is available
    let registry = ToolRegistry::new(adapter.clone());
    assert_eq!(registry.list_tools()[0].name, "area");
    Ok(())
}
```

### Using Tools with Agents

Agents can use tools to extend their capabilities:

```rust,no_run
use std::sync::Arc;

use cloudllm::council::Agent;
use cloudllm::clients::openai::{OpenAIClient, Model};
use cloudllm::tool_adapters::CustomToolAdapter;
use cloudllm::tool_protocol::{ToolMetadata, ToolParameter, ToolParameterType, ToolRegistry, ToolResult};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Set up tools
    let adapter = Arc::new(CustomToolAdapter::new());

    adapter.register_tool(
        ToolMetadata::new("add", "Add two numbers"),
        Arc::new(|params| {
            let a = params["a"].as_f64().unwrap_or(0.0);
            let b = params["b"].as_f64().unwrap_or(0.0);
            Ok(ToolResult::success(serde_json::json!({"result": a + b})))
        }),
    ).await;

    let registry = Arc::new(ToolRegistry::new(adapter));

    // Create an agent with tool access
    let agent = Agent::new(
        "calculator",
        "Calculator Agent",
        Arc::new(OpenAIClient::new_with_model_enum(
            &std::env::var("OPEN_AI_SECRET")?,
            Model::GPT41Mini
        )),
    )
    .with_expertise("Performs calculations")
    .with_tools(registry);

    println!("Agent has {} tools", agent.tool_registry.is_some() as u8);
    Ok(())
}
```

### Tool Adapters

Different tool adapters allow you to use the same tool definitions with different protocols:

#### 1. CustomToolAdapter (Simple Rust Functions)

Perfect for prototyping and simple use cases:

```rust,no_run
use std::sync::Arc;
use cloudllm::tool_adapters::CustomToolAdapter;
use cloudllm::tool_protocol::{ToolMetadata, ToolResult};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let adapter = Arc::new(CustomToolAdapter::new());

    // Synchronous tool
    adapter.register_tool(
        ToolMetadata::new("greet", "Greet someone"),
        Arc::new(|params| {
            let name = params["name"].as_str().unwrap_or("Friend");
            Ok(ToolResult::success(serde_json::json!({"greeting": format!("Hello, {}!", name)})))
        }),
    ).await;

    // Asynchronous tool
    adapter.register_async_tool(
        ToolMetadata::new("fetch", "Fetch data asynchronously"),
        Arc::new(|_params| {
            Box::pin(async {
                tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
                Ok(ToolResult::success(serde_json::json!({"data": "fetched"})))
            })
        }),
    ).await;

    Ok(())
}
```

#### 2. McpAdapter (Model Context Protocol)

For integration with MCP servers:

```rust,no_run
use std::sync::Arc;
use cloudllm::tool_adapters::McpAdapter;
use cloudllm::tool_protocol::ToolProtocol;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Connect to an MCP server
    let mut adapter = McpAdapter::new("http://localhost:8080/mcp".to_string());
    adapter.initialize().await?;

    // List available tools from the MCP server
    let tools = adapter.list_tools().await?;
    println!("Available tools: {}", tools.len());

    Ok(())
}
```

#### 3. OpenAIFunctionAdapter (OpenAI Function Calling)

For native OpenAI function calling format:

```rust,no_run
use std::sync::Arc;
use cloudllm::tool_adapters::OpenAIFunctionAdapter;
use cloudllm::tool_protocol::{ToolMetadata, ToolResult};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let adapter = Arc::new(OpenAIFunctionAdapter::new());

    adapter.register_function(
        ToolMetadata::new("search", "Search the web"),
        Arc::new(|params| {
            let query = params["query"].as_str().unwrap_or("");
            Box::pin(async move {
                Ok(ToolResult::success(serde_json::json!({
                    "results": ["result1", "result2"],
                    "query": query
                })))
            })
        }),
    ).await;

    // Get OpenAI-formatted functions
    let functions = adapter.get_openai_functions().await;
    println!("OpenAI functions: {:?}", functions);

    Ok(())
}
```

#### 4. MemoryToolAdapter (Persistent Agent State)

For maintaining state across sessions within a single process:

```rust,no_run
use std::sync::Arc;
use cloudllm::tools::Memory;
use cloudllm::tool_adapters::MemoryToolAdapter;
use cloudllm::tool_protocol::ToolRegistry;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create shared memory for persistence
    let memory = Arc::new(Memory::new());
    let adapter = Arc::new(MemoryToolAdapter::new(memory));
    let registry = Arc::new(ToolRegistry::new(adapter));

    // Execute memory operations
    let result = registry.execute_tool(
        "memory",
        serde_json::json!({"command": "P task_name ImportantTask 3600"}),
    ).await?;

    println!("Stored: {}", result.output);
    Ok(())
}
```

#### 5. McpMemoryClient (Distributed Agent Coordination)

For coordinating multiple agents across different processes or machines via a remote Memory service:

```rust,no_run
use cloudllm::tool_adapters::McpMemoryClient;
use cloudllm::tool_protocol::ToolProtocol;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Connect to a remote MCP Memory Server
    let client = McpMemoryClient::new("http://localhost:8080".to_string());

    // Store data on the remote server
    let store_result = client.execute(
        "memory",
        serde_json::json!({"command": "P agent_state research_complete 3600"})
    ).await?;

    // Retrieve data from the remote server
    let get_result = client.execute(
        "memory",
        serde_json::json!({"command": "G agent_state META"})
    ).await?;

    println!("Agent state: {}", get_result.output);
    Ok(())
}
```

**Use Cases:**
- **Agent Fleets**: Multiple agents sharing memory across network
- **Microservices**: Different services coordinating through shared Memory
- **Multi-Region**: Centralized memory for geographically distributed systems
- **Agent Clusters**: Coordinating state across a cluster of agent instances

The McpMemoryClient connects to a remote MCP Memory Server (which exposes a Memory instance via HTTP), allowing distributed agents to coordinate decisions and maintain shared state.

### Multi-Protocol ToolRegistry (Using Multiple MCP Servers)

CloudLLM supports agents connecting to multiple MCP servers simultaneously. Agents transparently access
tools from all connected sources as if they were available locally.

#### Basic Usage

```rust,no_run
use std::sync::Arc;
use cloudllm::council::Agent;
use cloudllm::clients::openai::{Model, OpenAIClient};
use cloudllm::tool_protocol::ToolRegistry;
use cloudllm::tool_protocols::McpClientProtocol;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create an empty multi-protocol registry
    let mut registry = ToolRegistry::empty();

    // Add local tools
    let local_tools = Arc::new(CustomToolProtocol::new());
    registry.add_protocol("local", local_tools).await?;

    // Add remote MCP servers
    let youtube = Arc::new(McpClientProtocol::new(
        "http://youtube-mcp:8081".to_string()
    ));
    registry.add_protocol("youtube", youtube).await?;

    let github = Arc::new(McpClientProtocol::new(
        "http://github-mcp:8082".to_string()
    ));
    registry.add_protocol("github", github).await?;

    // Create agent with all tools available
    let agent = Agent::new(
        "researcher",
        "Research Agent",
        Arc::new(OpenAIClient::new_with_model_enum(
            &std::env::var("OPEN_AI_SECRET")?,
            Model::GPT41Mini
        )),
    )
    .with_tools(Arc::new(registry));

    // Agent can now use tools from all three sources transparently!
    Ok(())
}
```

#### Key Features

- **Dynamic Protocol Registration**: Add/remove protocols at runtime via `add_protocol()` and `remove_protocol()`
- **Transparent Tool Routing**: Registry automatically routes tool calls to the correct protocol
- **Auto Tool Discovery**: Each protocol is queried for available tools when added
- **Backwards Compatible**: Existing single-protocol code continues to work unchanged

#### Single-Protocol Mode (Existing Code)

```rust,no_run
use std::sync::Arc;
use cloudllm::tool_protocol::ToolRegistry;
use cloudllm::tool_protocols::CustomToolProtocol;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let protocol = Arc::new(CustomToolProtocol::new());
    let mut registry = ToolRegistry::new(protocol);
    registry.discover_tools_from_primary().await?;
    Ok(())
}
```

### Built-in Tools

CloudLLM includes several production-ready tools that agents can use directly:

#### Calculator Tool

A fast, reliable scientific calculator for mathematical operations and statistical analysis. Perfect for agents that need to perform computations.

**Features:**
- Comprehensive arithmetic operations (`+`, `-`, `*`, `/`, `^`, `%`)
- Trigonometric functions (sin, cos, tan, csc, sec, cot, asin, acos, atan)
- Hyperbolic functions (sinh, cosh, tanh, csch, sech, coth)
- Logarithmic and exponential functions (ln, log, log2, exp)
- Statistical operations (mean, median, mode, std, stdpop, var, varpop, sum, count, min, max)
- Mathematical constants (pi, e)

**Usage Example:**

```rust,no_run
use cloudllm::tools::Calculator;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let calc = Calculator::new();

    // Arithmetic
    println!("{}", calc.evaluate("2 + 2 * 3").await?);  // 8.0

    // Trigonometry (radians)
    println!("{}", calc.evaluate("sin(pi/2)").await?);  // 1.0

    // Statistical functions
    println!("{}", calc.evaluate("mean([1, 2, 3, 4, 5])").await?);  // 3.0

    Ok(())
}
```

**More Examples:**
- `sqrt(16)` → 4.0
- `log(100)` → 2.0 (base 10)
- `std([1, 2, 3, 4, 5])` → 1.581 (sample standard deviation)
- `floor(3.7)` → 3.0

For comprehensive documentation, see [`Calculator` API docs](https://docs.rs/cloudllm/latest/cloudllm/tools/struct.Calculator.html).

#### Memory Tool

A persistent, TTL-aware key-value store for maintaining agent state across sessions. See [`Memory` API docs](https://docs.rs/cloudllm/latest/cloudllm/tools/struct.Memory.html).

#### HTTP Client Tool

A secure REST API client for calling external services with domain allowlist/blocklist protection. Perfect for agents that need to make HTTP requests to external APIs.

**Features:**
- All HTTP methods (GET, POST, PUT, DELETE, PATCH, HEAD)
- Domain security with allowlist/blocklist (blocklist takes precedence)
- Basic authentication and bearer token support
- Custom headers and query parameters with automatic URL encoding
- JSON response parsing
- Configurable request timeout and response size limits
- Thread-safe with connection pooling
- Builder pattern for chainable configuration

**Usage Example:**

```rust,no_run
use cloudllm::tools::HttpClient;
use std::time::Duration;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut client = HttpClient::new();

    // Security: only allow api.example.com
    client.allow_domain("api.example.com");

    // Configuration via builder pattern
    client
        .with_header("Authorization", "Bearer token123")
        .with_query_param("format", "json")
        .with_timeout(Duration::from_secs(30));

    // Make request
    let response = client.get("https://api.example.com/data").await?;

    // Check status and parse JSON
    if response.is_success() {
        let json_data = response.json()?;
        println!("Data: {}", json_data);
    }

    Ok(())
}
```

**Security Features:**

- **Allowlist**: Restrict requests to trusted domains only
- **Blocklist**: Explicitly block malicious domains
- **Precedence**: Blocklist always takes precedence over allowlist
- **No allowlist = All allowed**: Empty allowlist means any domain is allowed (unless in blocklist)

**More Examples:**
- Basic auth: `client.with_basic_auth("username", "password")`
- Custom header: `client.with_header("X-API-Key", "secret123")`
- Query params: `client.with_query_param("page", "1").with_query_param("limit", "50")`
- Size limit: `client.with_max_response_size(50 * 1024 * 1024)` (50MB)
- Short timeout: `client.with_timeout(Duration::from_secs(5))`

For comprehensive documentation and more examples, see [`HttpClient` API docs](https://docs.rs/cloudllm/latest/cloudllm/tools/struct.HttpClient.html) and run `cargo run --example http_client_example`.

##### Using HTTP Client Tool with Agents

The HTTP Client tool can be exposed to agents through the MCP protocol, allowing agents to make API calls autonomously. Here's how to set it up:

**Step 1: Create an MCP HTTP Server (expose via HTTP)**

First, create a simple MCP server that wraps the HTTP Client tool:

```rust,no_run
use std::sync::Arc;
use cloudllm::tools::HttpClient;
use cloudllm::tool_protocols::CustomToolProtocol;
use cloudllm::tool_protocol::{ToolMetadata, ToolParameter, ToolParameterType, ToolResult};
use serde_json::json;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create HTTP client with security settings
    let http_client = Arc::new(HttpClient::new());

    // Wrap it with CustomToolProtocol for agent usage
    let mut protocol = CustomToolProtocol::new();

    // Register HTTP GET tool
    let client = http_client.clone();
    protocol.register_async_tool(
        ToolMetadata::new("http_get", "Make an HTTP GET request to an API")
            .with_parameter(
                ToolParameter::new("url", ToolParameterType::String)
                    .with_description("The URL to fetch")
                    .required()
            )
            .with_parameter(
                ToolParameter::new("headers", ToolParameterType::Object)
                    .with_description("Optional headers as JSON object")
            ),
        Arc::new(move |params| {
            let client = client.clone();
            Box::pin(async move {
                let url = params["url"].as_str().ok_or("url parameter required")?;

                match client.get(url).await {
                    Ok(response) => {
                        if response.is_success() {
                            Ok(ToolResult::success(json!({
                                "status": response.status,
                                "body": response.body
                            })))
                        } else {
                            Ok(ToolResult::error(
                                format!("HTTP {}: {}", response.status, response.body)
                            ))
                        }
                    }
                    Err(e) => Ok(ToolResult::error(e.to_string()))
                }
            })
        })
    ).await;

    Ok(())
}
```

**Step 2: Create an Agent that Uses HTTP Client Tools**

```rust,no_run
use std::sync::Arc;
use cloudllm::council::Agent;
use cloudllm::clients::openai::{OpenAIClient, Model};
use cloudllm::tool_protocol::ToolRegistry;
use cloudllm::tool_protocols::CustomToolProtocol;
use cloudllm::tool_protocol::{ToolMetadata, ToolParameter, ToolParameterType, ToolResult};
use serde_json::json;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create protocol with HTTP tools
    let mut protocol = CustomToolProtocol::new();

    // Register GET tool
    protocol.register_async_tool(
        ToolMetadata::new("get_json_api", "Fetch JSON data from an API endpoint"),
        Arc::new(|params| {
            Box::pin(async move {
                let url = params["url"].as_str().unwrap_or("http://api.example.com");
                Ok(ToolResult::success(json!({"data": "sample"})))
            })
        })
    ).await;

    // Create tool registry
    let registry = Arc::new(ToolRegistry::new(Arc::new(protocol)));

    // Create agent with HTTP access
    let mut agent = Agent::new(
        "api-agent",
        "API Integration Agent",
        Arc::new(OpenAIClient::new_with_model_enum(
            &std::env::var("OPEN_AI_SECRET")?,
            Model::GPT41Mini
        )),
    )
    .with_expertise("Makes HTTP requests to external APIs")
    .with_tools(registry);

    // Agent can now make API calls!
    println!("Agent has HTTP tools available");
    Ok(())
}
```

**Step 3: Configure Agent System Prompt for HTTP Usage**

Teach the agent about available HTTP tools via the system prompt:

```
You have access to HTTP tools for making API calls:

1. get_json_api(url: string, headers?: object)
   - Fetches JSON data from an API endpoint
   - Returns: {status: number, body: string}
   - Security: Only allowed domains are accessible
   - Use this to fetch real-time data from external services

2. post_json_api(url: string, data: object, headers?: object)
   - Posts JSON data to an API endpoint
   - Use this to submit data to external services

Always check the response status before processing the body.
When calling APIs, include appropriate headers like Content-Type.
Never share authentication tokens in logs.
```

**Step 4: Multi-MCP Setup (Advanced)**

Combine HTTP Client with other tools via multiple MCP servers:

```rust,no_run
use std::sync::Arc;
use cloudllm::council::Agent;
use cloudllm::clients::openai::{OpenAIClient, Model};
use cloudllm::tool_protocol::ToolRegistry;
use cloudllm::tool_protocols::CustomToolProtocol;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create empty registry for multiple protocols
    let mut registry = ToolRegistry::empty();

    // Add HTTP tools locally
    let http_protocol = Arc::new(CustomToolProtocol::new());
    registry.add_protocol("http", http_protocol).await?;

    // Add memory tools locally
    let memory_protocol = Arc::new(CustomToolProtocol::new());
    registry.add_protocol("memory", memory_protocol).await?;

    // Connect to remote MCP servers
    use cloudllm::tool_protocols::McpClientProtocol;

    let github_mcp = Arc::new(McpClientProtocol::new(
        "http://localhost:8081".to_string()
    ));
    registry.add_protocol("github", github_mcp).await?;

    // Create agent with access to all tools
    let mut agent = Agent::new(
        "orchestrator",
        "Multi-Tool Orchestrator",
        Arc::new(OpenAIClient::new_with_model_enum(
            &std::env::var("OPEN_AI_SECRET")?,
            Model::GPT41Mini
        )),
    )
    .with_tools(Arc::new(registry));

    println!("Agent can now:");
    println!("  - Make HTTP API calls (http_*)");
    println!("  - Store/retrieve data in memory (memory_*)");
    println!("  - Interact with GitHub (github_*)");

    Ok(())
}
```

**Security Best Practices:**

1. **Domain Allowlist**: Configure HTTP clients with domain allowlists to prevent unauthorized requests
   ```rust
   let mut client = HttpClient::new();
   client.allow_domain("api.trusted-service.com");
   client.allow_domain("public-api.example.com");
   ```

2. **Deny Malicious Domains**: Use blocklists as a second layer
   ```rust
   client.deny_domain("malicious.attacker.com");
   ```

3. **Timeout Protection**: Set reasonable timeouts to prevent hanging requests
   ```rust
   use std::time::Duration;
   client.with_timeout(Duration::from_secs(30));
   ```

4. **Size Limits**: Limit response sizes to prevent memory exhaustion
   ```rust
   client.with_max_response_size(10 * 1024 * 1024); // 10MB
   ```

5. **Authentication**: Use appropriate auth methods when needed
   ```rust
   client.with_basic_auth("username", "password");
   // or
   client.with_header("Authorization", "Bearer your-token");
   ```

#### Bash Tool

Secure command execution on Linux and macOS with timeout and security controls. See [`BashTool` API docs](https://docs.rs/cloudllm/latest/cloudllm/tools/struct.BashTool.html).

### Creating Custom Protocol Adapters

Implement the [`ToolProtocol`](https://docs.rs/cloudllm/latest/cloudllm/tool_protocol/trait.ToolProtocol.html) trait to support new protocols:

```rust,no_run
use async_trait::async_trait;
use cloudllm::tool_protocol::{ToolMetadata, ToolProtocol, ToolResult};
use std::error::Error;

/// Example: Custom protocol adapter for a hypothetical service
pub struct MyCustomAdapter {
    // Your implementation
}

#[async_trait]
impl ToolProtocol for MyCustomAdapter {
    async fn execute(
        &self,
        tool_name: &str,
        parameters: serde_json::Value,
    ) -> Result<ToolResult, Box<dyn Error + Send + Sync>> {
        // Implement tool execution logic
        Ok(ToolResult::success(serde_json::json!({})))
    }

    async fn list_tools(&self) -> Result<Vec<ToolMetadata>, Box<dyn Error + Send + Sync>> {
        // Return available tools
        Ok(vec![])
    }

    async fn get_tool_metadata(
        &self,
        tool_name: &str,
    ) -> Result<ToolMetadata, Box<dyn Error + Send + Sync>> {
        // Return specific tool metadata
        Ok(ToolMetadata::new(tool_name, "Tool description"))
    }

    fn protocol_name(&self) -> &str {
        "my-custom-protocol"
    }
}
```

### Using Tools in Agent System Prompts

Teach agents about available tools via the system prompt:

```
You have access to the following tools:

1. Calculator (add, subtract, multiply)
   - Use for mathematical operations
   - Respond with: {"tool_call": {"name": "add", "parameters": {"a": 5, "b": 3}}}

2. Memory System
   - Store important information
   - Use command: P key value ttl
   - Retrieve with: G key META

Always use tools when they can help answer the user's question. After using a tool,
incorporate the result into your response.
```

### Best Practices for Tools

1. **Clear Names & Descriptions**: Make tool purposes obvious to LLMs
2. **Comprehensive Parameters**: Document all required and optional parameters
3. **Error Handling**: Return meaningful error messages in ToolResult
4. **Atomicity**: Each tool should do one thing well
5. **Documentation**: Include examples in tool descriptions
6. **Testing**: Test tool execution in isolation before integration

For more examples, see the `examples/` directory and run `cargo doc --open` for complete API documentation.

---

## Councils: multi-agent orchestration

The `council` module orchestrates conversations between agents built on any `ClientWrapper`.
Choose from parallel, round-robin, moderated, hierarchical, or debate modes.

```rust,no_run
use std::sync::Arc;

use cloudllm::council::{Agent, Council, CouncilMode};
use cloudllm::clients::openai::{Model, OpenAIClient};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let key = std::env::var("OPEN_AI_SECRET")?;

    let architect = Agent::new(
        "architect",
        "System Architect",
        Arc::new(OpenAIClient::new_with_model_enum(&key, Model::GPT4o)),
    )
    .with_expertise("Distributed systems")
    .with_personality("Pragmatic, direct");

    let tester = Agent::new(
        "qa",
        "QA Lead",
        Arc::new(OpenAIClient::new_with_model_enum(&key, Model::GPT41Mini)),
    )
    .with_expertise("Test automation")
    .with_personality("Sceptical, detail-oriented");

    let mut council = Council::new("design-review", "Deployment Review")
        .with_mode(CouncilMode::RoundRobin)
        .with_system_context("Collaboratively review the proposed architecture.");

    council.add_agent(architect)?;
    council.add_agent(tester)?;

    let outcome = council
        .discuss("Evaluate whether the blue/green rollout plan is sufficient.", 2)
        .await?;

    for msg in outcome.messages {
        if let Some(name) = msg.agent_name {
            println!("{name}: {}", msg.content);
        }
    }

    Ok(())
}
```

For a deep dive, read [`COUNCIL_TUTORIAL.md`](./COUNCIL_TUTORIAL.md) which walks through each
collaboration mode with progressively sophisticated examples.

---

## Examples

Clone the repository and run the provided examples:

```bash
export OPEN_AI_SECRET=...
export ANTHROPIC_KEY=...
export GEMINI_KEY=...
export XAI_KEY=...

cargo run --example interactive_session
cargo run --example streaming_session
cargo run --example council_demo
```

Each example corresponds to a module in the documentation so you can cross-reference the code with
explanations.

---

## Support & contributing

Issues and pull requests are welcome via [GitHub](https://github.com/CloudLLM-ai/cloudllm).
Please open focused pull requests against `main` and include tests or doc updates where relevant.

CloudLLM is released under the [MIT License](./LICENSE).

---

Happy orchestration! 🤖🤝🤖
