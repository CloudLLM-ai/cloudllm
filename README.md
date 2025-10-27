# CloudLLM

<p align="center">
  <img src="https://github.com/CloudLLM-ai/cloudllm/blob/master/logo.png?raw=true" width="220" alt="CloudLLM logo" />
</p>

CloudLLM is a batteries-included Rust toolkit for building intelligent agents with LLM integration,
multi-protocol tool support, and multi-agent orchestration. It provides:

* **Agents with Tools**: Create agents that connect to LLMs and execute actions through a flexible,
  multi-protocol tool system (local, remote MCP, Memory, custom protocols),
* **Server Deployment**: Easy standalone MCP server creation via [`MCPServerBuilder`](https://docs.rs/cloudllm/latest/cloudllm/mcp_server/struct.MCPServerBuilder.html)
  with HTTP, authentication, and IP filtering,
* **Flexible Tool Creation**: From simple Rust closures to advanced custom protocol implementations,
* **Stateful Sessions**: A [`LLMSession`](https://docs.rs/cloudllm/latest/cloudllm/struct.LLMSession.html) for
  managing conversation history with context trimming and token accounting,
* **Multi-Agent Orchestration**: A [`council`](https://docs.rs/cloudllm/latest/cloudllm/council/index.html) engine
  supporting Parallel, RoundRobin, Moderated, Hierarchical, and Debate collaboration patterns,
* **Provider Flexibility**: Unified [`ClientWrapper`](https://docs.rs/cloudllm/latest/cloudllm/client_wrapper/index.html)
  trait for OpenAI, Claude, Gemini, Grok, and custom OpenAI-compatible endpoints.

The entire public API is documented with _compilable_ examples—run `cargo doc --open` to browse the
crate-level manual.

---

## Installation

Add CloudLLM to your project:

```toml
[dependencies]
cloudllm = "0.6.0"
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

## LLMSession: Stateful Conversations (The Foundation)

LLMSession is the core building block—it maintains conversation history with automatic context trimming
and token accounting. Use it for simple stateful conversations with any LLM provider:

```rust,no_run
use std::sync::Arc;
use cloudllm::{LLMSession, Role};
use cloudllm::clients::openai::{OpenAIClient, Model};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = Arc::new(OpenAIClient::new_with_model_enum(
        &std::env::var("OPEN_AI_SECRET")?,
        Model::GPT41Mini
    ));

    let mut session = LLMSession::new(client, "You are helpful.".into(), 8_192);

    let reply = session
        .send_message(Role::User, "Tell me about Rust.".into(), None)
        .await?;

    println!("Assistant: {}", reply.content);
    println!("Tokens used: {:?}", session.token_usage());
    Ok(())
}
```

---

## Agents: Building Intelligent Workers with Tools

Agents extend LLMSession by adding identity, expertise, and optional tools. They're the primary way to build
sophisticated LLM interactions where you need the agent to take actions beyond conversation:

```rust,no_run
use std::sync::Arc;
use cloudllm::Agent;
use cloudllm::clients::openai::{OpenAIClient, Model};
use cloudllm::tool_protocol::ToolRegistry;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = Arc::new(OpenAIClient::new_with_model_enum(
        &std::env::var("OPEN_AI_SECRET")?,
        Model::GPT41Mini
    ));

    // Create agent with custom identity and expertise
    let agent = Agent::new("researcher", "Research Assistant", client)
        .with_expertise("Literature search and analysis")
        .with_personality("Thorough and methodical");

    // Agent is ready to execute actions!
    println!("Agent ready: {}", agent.name);
    Ok(())
}
```

---

## Tool Registry: Multi-Protocol Tool Access

Agents access tools through the `ToolRegistry`, which supports **multiple simultaneous protocols**. Use local tools, remote MCP servers, persistent Memory, or custom implementations—all transparently:

### Adding Tools to a Registry

```rust,no_run
use std::sync::Arc;
use cloudllm::tool_protocol::ToolRegistry;
use cloudllm::tool_protocols::{CustomToolProtocol, McpClientProtocol};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create empty registry for multiple protocols
    let mut registry = ToolRegistry::empty();

    // Add local tools (Rust closures)
    let local = Arc::new(CustomToolProtocol::new());
    registry.add_protocol("local", local).await?;

    // Add remote MCP servers
    let github = Arc::new(McpClientProtocol::new("http://localhost:8081".to_string()));
    registry.add_protocol("github", github).await?;

    let calculator = Arc::new(McpClientProtocol::new("http://localhost:8082".to_string()));
    registry.add_protocol("calculator", calculator).await?;

    // Agent using this registry accesses all tools transparently!
    Ok(())
}
```

**Key Benefits:**
- **Local + Remote**: Mix tools from different sources in a single agent
- **Transparent Routing**: Registry automatically routes calls to the correct protocol
- **Dynamic Management**: Add/remove protocols at runtime
- **Backward Compatible**: Existing single-protocol code still works

### Registry Modes

**Multi-Protocol (New agents):**
```rust
let mut registry = ToolRegistry::empty();
registry.add_protocol("name", protocol).await?;
```

**Single-Protocol (Existing code):**
```rust
let protocol = Arc::new(CustomToolProtocol::new());
let registry = ToolRegistry::new(protocol);
```

---

## Deploying Tool Servers with MCPServerBuilder

Create standalone MCP servers exposing tools over HTTP. Perfect for microservices, integration testing, or sharing tools across your infrastructure:

```rust,no_run
use std::sync::Arc;
use cloudllm::mcp_server::MCPServerBuilder;
use cloudllm::tool_protocols::CustomToolProtocol;
use cloudllm::tool_protocol::{ToolMetadata, ToolResult};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let protocol = Arc::new(CustomToolProtocol::new());

    // Register tools
    protocol.register_tool(
        ToolMetadata::new("calculator", "Evaluate math expressions"),
        Arc::new(|params| {
            let expr = params["expr"].as_str().unwrap_or("0");
            Ok(ToolResult::success(serde_json::json!({"result": 42.0})))
        }),
    ).await;

    // Deploy with security options
    MCPServerBuilder::new()
        .with_protocol("tools", protocol)
        .with_port(8080)
        .with_localhost_only()  // Only accept localhost
        .with_bearer_token("your-secret-token")  // Optional auth
        .build_and_serve()
        .await?;

    Ok(())
}
```

Available on the `mcp-server` feature. Other agents connect via `McpClientProtocol::new("http://localhost:8080")`.

---

## Creating Tools: Simple to Advanced

CloudLLM provides a powerful, protocol-agnostic tool system that works seamlessly with agents and councils.
Tools enable agents to take actions beyond conversation—calculate values, query databases, call APIs, or
maintain state across sessions.

### Simple Tool Creation: Rust Closures

Register Rust functions or closures as tools. Perfect for quick prototyping:

```rust,no_run
use std::sync::Arc;
use cloudllm::tool_protocols::CustomToolProtocol;
use cloudllm::tool_protocol::{ToolMetadata, ToolResult};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let protocol = Arc::new(CustomToolProtocol::new());

    // Synchronous tool
    protocol.register_tool(
        ToolMetadata::new("add", "Add two numbers"),
        Arc::new(|params| {
            let a = params["a"].as_f64().unwrap_or(0.0);
            let b = params["b"].as_f64().unwrap_or(0.0);
            Ok(ToolResult::success(serde_json::json!({"result": a + b})))
        }),
    ).await;

    // Asynchronous tool
    protocol.register_async_tool(
        ToolMetadata::new("fetch_url", "Fetch data from a URL"),
        Arc::new(|params| {
            Box::pin(async {
                let url = params["url"].as_str().unwrap_or("");
                // Perform async operation
                Ok(ToolResult::success(serde_json::json!({"url": url, "status": "ok"})))
            })
        }),
    ).await;

    Ok(())
}
```

### Advanced Tool Creation: Custom Protocol Implementation

For complex tools or external system integration, implement the `ToolProtocol` trait:

```rust,no_run
use async_trait::async_trait;
use cloudllm::tool_protocol::{ToolMetadata, ToolProtocol, ToolResult};
use std::error::Error;

pub struct DatabaseAdapter;

#[async_trait]
impl ToolProtocol for DatabaseAdapter {
    async fn execute(
        &self,
        tool_name: &str,
        parameters: serde_json::Value,
    ) -> Result<ToolResult, Box<dyn Error + Send + Sync>> {
        match tool_name {
            "query" => {
                let sql = parameters["sql"].as_str().unwrap_or("");
                // Execute actual database query
                Ok(ToolResult::success(serde_json::json!({"result": "data"})))
            }
            _ => Ok(ToolResult::error("Unknown tool".into()))
        }
    }

    async fn list_tools(&self) -> Result<Vec<ToolMetadata>, Box<dyn Error + Send + Sync>> {
        Ok(vec![ToolMetadata::new("query", "Execute SQL query")])
    }

    async fn get_tool_metadata(
        &self,
        tool_name: &str,
    ) -> Result<ToolMetadata, Box<dyn Error + Send + Sync>> {
        Ok(ToolMetadata::new(tool_name, "Database query tool"))
    }

    fn protocol_name(&self) -> &str {
        "database"
    }
}
```

### Using Tools with Agents

Agents use tools through a registry. Connect any tool source to an agent:

```rust,no_run
use std::sync::Arc;
use cloudllm::Agent;
use cloudllm::clients::openai::{OpenAIClient, Model};
use cloudllm::tool_protocols::CustomToolProtocol;
use cloudllm::tool_protocol::{ToolMetadata, ToolRegistry, ToolResult};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create tools
    let protocol = Arc::new(CustomToolProtocol::new());
    protocol.register_tool(
        ToolMetadata::new("add", "Add two numbers"),
        Arc::new(|params| {
            let a = params["a"].as_f64().unwrap_or(0.0);
            let b = params["b"].as_f64().unwrap_or(0.0);
            Ok(ToolResult::success(serde_json::json!({"result": a + b})))
        }),
    ).await;

    let registry = Arc::new(ToolRegistry::new(protocol));

    // Create agent with tool access
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

    println!("✓ Agent ready with tools");
    Ok(())
}
```

### Protocol Implementations

#### 1. CustomToolProtocol (Local Rust Functions)

Register local Rust closures or async functions as tools. Covered above under "Simple Tool Creation".

#### 2. McpClientProtocol (Remote MCP Servers)

Connect to remote MCP servers:

```rust,no_run
use std::sync::Arc;
use cloudllm::tool_protocols::McpClientProtocol;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Connect to an MCP server
    let protocol = Arc::new(McpClientProtocol::new("http://localhost:8080".to_string()));

    // List available tools from the MCP server
    let tools = protocol.list_tools().await?;
    println!("Available tools: {}", tools.len());

    Ok(())
}
```

#### 3. MemoryProtocol (Persistent Agent State)

For maintaining state across sessions within a single process:

```rust,no_run
use std::sync::Arc;
use cloudllm::tools::Memory;
use cloudllm::tool_protocols::MemoryProtocol;
use cloudllm::tool_protocol::ToolRegistry;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create shared memory for persistence
    let memory = Arc::new(Memory::new());
    let protocol = Arc::new(MemoryProtocol::new(memory));
    let registry = Arc::new(ToolRegistry::new(protocol));

    // Execute memory operations
    let result = registry.execute_tool(
        "memory",
        serde_json::json!({"command": "P task_name ImportantTask 3600"}),
    ).await?;

    println!("Stored: {}", result.output);
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

A persistent, TTL-aware key-value store for maintaining agent state across sessions. Perfect for single agents to track progress or multi-agent councils to coordinate decisions.

**Features:**
- Key-value storage with optional TTL (time-to-live) expiration
- Automatic background expiration of stale entries (1-second cleanup)
- Metadata tracking (creation timestamp, expiration time)
- Succinct protocol for LLM communication (token-efficient)
- Thread-safe shared access across agents
- Designed specifically for agent communication (not a general database)

**Basic Usage Example:**

```rust,no_run
use cloudllm::tools::Memory;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let memory = Memory::new();

    // Store data with 1-hour TTL
    memory.put("research_progress".to_string(), "Found 3 relevant papers".to_string(), Some(3600));

    // Retrieve data
    if let Some((value, metadata)) = memory.get("research_progress", true) {
        println!("Progress: {}", value);
        println!("Stored at: {:?}", metadata.unwrap().added_utc);
    }

    // List all stored keys
    let keys = memory.list_keys();
    println!("Active memories: {:?}", keys);

    // Store without expiration (permanent)
    memory.put("important_decision".to_string(), "Use approach A".to_string(), None);

    // Delete specific memory
    memory.delete("research_progress");

    // Clear all memories
    memory.clear();

    Ok(())
}
```

**Using with Agents via Tool Protocol:**

```rust,no_run
use std::sync::Arc;
use cloudllm::tools::Memory;
use cloudllm::tool_protocols::MemoryProtocol;
use cloudllm::tool_protocol::ToolRegistry;
use cloudllm::council::Agent;
use cloudllm::clients::openai::{OpenAIClient, Model};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create shared memory for agents
    let memory = Arc::new(Memory::new());

    // Wrap with protocol for agent usage
    let protocol = Arc::new(MemoryProtocol::new(memory.clone()));
    let registry = Arc::new(ToolRegistry::new(protocol));

    // Create agent with memory access
    let mut agent = Agent::new(
        "researcher",
        "Research Agent",
        Arc::new(OpenAIClient::new_with_model_enum(
            &std::env::var("OPEN_AI_SECRET")?,
            Model::GPT41Mini
        )),
    )
    .with_tools(registry);

    // Agent can now use memory via commands like:
    // "P research_state Gathering data TTL:7200"
    // "G research_state META"
    // "L"

    Ok(())
}
```

**Memory Protocol Commands (for agents):**

The Memory tool uses a token-efficient protocol designed for LLM communication:

| Command | Syntax | Example | Use Case |
|---------|--------|---------|----------|
| **Put** | `P <key> <value> [TTL:<seconds>]` | `P task_status InProgress TTL:3600` | Store state with 1-hour expiration |
| **Get** | `G <key> [META]` | `G task_status META` | Retrieve value + metadata |
| **List** | `L [META]` | `L META` | List all keys with metadata |
| **Delete** | `D <key>` | `D task_status` | Remove specific memory |
| **Clear** | `C` | `C` | Wipe all memories |
| **Spec** | `SPEC` | `SPEC` | Get protocol specification |

**Use Case Examples:**

1. **Single-Agent Progress Tracking:**
   ```
   Agent stores: "P document_checkpoint Page 247 TTL:86400"
   Later: "G document_checkpoint" → retrieves current progress
   ```

2. **Multi-Agent Council Coordination:**
   ```
   Agent A stores: "P decision_consensus Approved TTL:3600"
   Agent B reads: "G decision_consensus"
   Agent C confirms: "L" → sees what's been decided
   ```

3. **Session Recovery:**
   ```
   Before shutdown: "P session_state {full_context} TTL:604800" (1 week)
   After restart: "G session_state" → resume from checkpoint
   ```

4. **Audit Trail:**
   ```
   Store each decision: "P milestone_v1 Completed TTL:2592000" (30 days)
   Track progress: "L META" → see timestamp and TTL of each milestone
   ```

**Best Practices:**

1. **Use TTL wisely**: Temporary data (hours), permanent decisions (None)
2. **Clear old memories**: Call `C` or `D` to free space
3. **Descriptive keys**: Use clear, hierarchical names like `decision_inference_v2`
4. **Batch operations**: Use `L META` to understand stored state before updates
5. **Monitor expiration**: Check metadata to prevent unexpected data loss

**Multi-Agent Memory Sharing:**

```rust,no_run
use std::sync::Arc;
use cloudllm::tools::Memory;
use cloudllm::tool_protocols::MemoryProtocol;
use cloudllm::tool_protocol::ToolRegistry;
use cloudllm::council::{Agent, Council, CouncilMode};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create shared memory (all agents access same instance)
    let shared_memory = Arc::new(Memory::new());

    let protocol = Arc::new(MemoryProtocol::new(shared_memory));
    let registry = Arc::new(ToolRegistry::new(protocol));

    // Create council of agents
    let agent1 = Agent::new(...)
        .with_tools(registry.clone());

    let agent2 = Agent::new(...)
        .with_tools(registry.clone());

    // Both agents access same memory
    let mut council = Council::new("research", "Collaborative Research");
    council.add_agent(agent1)?;
    council.add_agent(agent2)?;

    // Agents can:
    // 1. Coordinate: Agent A stores findings, Agent B retrieves
    // 2. Consensus: Store decisions that others can see
    // 3. Progress: Track overall research advancement

    Ok(())
}
```

For comprehensive documentation and patterns, see [`Memory` API docs](https://docs.rs/cloudllm/latest/cloudllm/tools/struct.Memory.html).

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

Create an HTTP server that exposes the HTTP Client tool via MCP protocol. This server can be accessed by agents over the network:

```rust,no_run
use std::sync::Arc;
use cloudllm::tools::HttpClient;
use cloudllm::tool_protocols::CustomToolProtocol;
use cloudllm::tool_protocol::{ToolMetadata, ToolParameter, ToolParameterType, ToolResult, ToolRegistry};
use serde_json::json;
use axum::{
    extract::Json,
    routing::post,
    Router,
};
use std::net::SocketAddr;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create HTTP client with security settings
    let mut http_client = HttpClient::new();

    // Configure security: only allow specific domains
    http_client.allow_domain("api.github.com");
    http_client.allow_domain("api.example.com");
    http_client.allow_domain("jsonplaceholder.typicode.com");

    let http_client = Arc::new(http_client);

    // Wrap it with CustomToolProtocol for tool management
    let mut protocol = CustomToolProtocol::new();

    // Register HTTP GET tool
    let client = http_client.clone();
    protocol.register_async_tool(
        ToolMetadata::new("http_get", "Make an HTTP GET request to an API")
            .with_parameter(
                ToolParameter::new("url", ToolParameterType::String)
                    .with_description("The URL to fetch (must be from allowed domains)")
                    .required()
            )
            .with_parameter(
                ToolParameter::new("headers", ToolParameterType::Object)
                    .with_description("Optional custom headers as JSON object")
            ),
        Arc::new(move |params| {
            let client = client.clone();
            Box::pin(async move {
                let url = params["url"].as_str().ok_or("url parameter required")?;

                match client.get(url).await {
                    Ok(response) => {
                        if response.is_success() {
                            // Try to parse as JSON
                            match response.json() {
                                Ok(json_data) => {
                                    Ok(ToolResult::success(json!({
                                        "status": response.status,
                                        "data": json_data
                                    })))
                                }
                                Err(_) => {
                                    Ok(ToolResult::success(json!({
                                        "status": response.status,
                                        "body": response.body
                                    })))
                                }
                            }
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

    // Register HTTP POST tool
    let client = http_client.clone();
    protocol.register_async_tool(
        ToolMetadata::new("http_post", "Post JSON data to an API")
            .with_parameter(
                ToolParameter::new("url", ToolParameterType::String)
                    .with_description("The URL to POST to (must be from allowed domains)")
                    .required()
            )
            .with_parameter(
                ToolParameter::new("data", ToolParameterType::Object)
                    .with_description("JSON data to send")
                    .required()
            ),
        Arc::new(move |params| {
            let client = client.clone();
            Box::pin(async move {
                let url = params["url"].as_str().ok_or("url parameter required")?;
                let data = params["data"].clone();

                match client.post(url, data).await {
                    Ok(response) => {
                        if response.is_success() {
                            Ok(ToolResult::success(json!({
                                "status": response.status,
                                "message": "Data posted successfully"
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

    // Create tool registry
    let registry = Arc::new(ToolRegistry::new(Arc::new(protocol)));

    // Create HTTP server endpoints
    let registry_list = registry.clone();
    let registry_exec = registry.clone();

    let app = Router::new()
        // MCP standard: list available tools
        .route("/tools/list", post(move || {
            let reg = registry_list.clone();
            async move {
                let tools = reg.list_tools().await.unwrap_or_default();
                Json(json!({
                    "tools": tools
                }))
            }
        }))
        // MCP standard: execute a tool
        .route("/tools/execute", post(move |Json(payload): Json<serde_json::Value>| {
            let reg = registry_exec.clone();
            async move {
                let tool_name = payload["tool"].as_str().unwrap_or("");
                let params = payload["params"].clone();

                match reg.execute_tool(tool_name, params).await {
                    Ok(result) => Json(json!({"result": result})),
                    Err(e) => Json(json!({"error": e.to_string()}))
                }
            }
        }));

    // Start server
    let addr = SocketAddr::from(([127, 0, 0, 1], 8080));
    println!("🚀 MCP HTTP Server running on http://{}", addr);
    println!("📋 List tools: POST http://{}/tools/list", addr);
    println!("🔧 Execute tool: POST http://{}/tools/execute", addr);
    println!("✓ Allowed domains: api.github.com, api.example.com, jsonplaceholder.typicode.com");

    axum::Server::bind(&addr)
        .serve(app.into_make_service())
        .await?;

    Ok(())
}
```

**Add to Cargo.toml:**
```toml
axum = "0.7"
```

**Usage:**

Once running, other services/agents can call this MCP server:

```bash
# List available tools
curl -X POST http://localhost:8080/tools/list

# Use http_get tool
curl -X POST http://localhost:8080/tools/execute \
  -H "Content-Type: application/json" \
  -d '{
    "tool": "http_get",
    "params": {
      "url": "https://api.github.com/repos/CloudLLM-ai/cloudllm"
    }
  }'
```

This MCP server can now be referenced by agents using `McpClientProtocol::new("http://localhost:8080")`, allowing them to access HTTP capabilities securely and with domain restrictions.
```

**Step 2: Create an Agent that Uses HTTP Client Tools**

```rust,no_run
use std::sync::Arc;
use cloudllm::council::Agent;
use cloudllm::clients::openai::{OpenAIClient, Model};
use cloudllm::tool_protocol::ToolRegistry;
use cloudllm::tool_protocols::CustomToolProtocol;
use cloudllm::tool_protocol::{ToolMetadata, ToolParameter, ToolParameterType, ToolResult};
use cloudllm::tools::HttpClient;
use serde_json::json;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create HTTP client with security settings
    let mut http_client = HttpClient::new();

    // Configure security: only allow trusted domains
    http_client.allow_domain("api.github.com");
    http_client.allow_domain("api.example.com");

    // Configure authentication
    http_client.with_header("User-Agent", "CloudLLM-Agent/1.0");

    let http_client = Arc::new(http_client);

    // Wrap with CustomToolProtocol to expose to agents
    let mut protocol = CustomToolProtocol::new();

    // Register HTTP GET tool using the actual HttpClient
    let client = http_client.clone();
    protocol.register_async_tool(
        ToolMetadata::new("get_json_api", "Fetch JSON data from an API endpoint")
            .with_parameter(
                ToolParameter::new("url", ToolParameterType::String)
                    .with_description("The URL to fetch (must be from allowed domains)")
                    .required()
            )
            .with_parameter(
                ToolParameter::new("headers", ToolParameterType::Object)
                    .with_description("Optional custom headers")
            ),
        Arc::new(move |params| {
            let client = client.clone();
            Box::pin(async move {
                let url = params["url"]
                    .as_str()
                    .ok_or("url parameter is required")?;

                // Use the actual HttpClient to make the request
                match client.get(url).await {
                    Ok(response) => {
                        if response.is_success() {
                            // Try to parse as JSON
                            match response.json() {
                                Ok(json_data) => {
                                    Ok(ToolResult::success(json!({
                                        "status": response.status,
                                        "data": json_data
                                    })))
                                }
                                Err(_) => {
                                    // Not JSON, return raw body
                                    Ok(ToolResult::success(json!({
                                        "status": response.status,
                                        "body": response.body
                                    })))
                                }
                            }
                        } else {
                            Ok(ToolResult::error(format!(
                                "HTTP {} error: {}",
                                response.status, response.body
                            )))
                        }
                    }
                    Err(e) => Ok(ToolResult::error(format!(
                        "Request failed: {}",
                        e
                    )))
                }
            })
        })
    ).await;

    // Register HTTP POST tool for sending data
    let client = http_client.clone();
    protocol.register_async_tool(
        ToolMetadata::new("post_json_api", "Post JSON data to an API endpoint")
            .with_parameter(
                ToolParameter::new("url", ToolParameterType::String)
                    .with_description("The URL to POST to (must be from allowed domains)")
                    .required()
            )
            .with_parameter(
                ToolParameter::new("data", ToolParameterType::Object)
                    .with_description("JSON data to send")
                    .required()
            ),
        Arc::new(move |params| {
            let client = client.clone();
            Box::pin(async move {
                let url = params["url"]
                    .as_str()
                    .ok_or("url parameter is required")?;

                let data = params["data"].clone();

                // Use the actual HttpClient to POST
                match client.post(url, data).await {
                    Ok(response) => {
                        if response.is_success() {
                            Ok(ToolResult::success(json!({
                                "status": response.status,
                                "message": "Data posted successfully"
                            })))
                        } else {
                            Ok(ToolResult::error(format!(
                                "HTTP {} error: {}",
                                response.status, response.body
                            )))
                        }
                    }
                    Err(e) => Ok(ToolResult::error(format!(
                        "Request failed: {}",
                        e
                    )))
                }
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

    // Agent can now make authenticated, secure API calls!
    println!("✓ Agent configured with HTTP tools");
    println!("✓ Allowed domains: api.github.com, api.example.com");
    println!("✓ Agent can now GET and POST to these APIs");
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

#### File System Tool

Safe file and directory operations with path traversal protection and optional extension filtering. Perfect for agents that need to read, write, and manage files within designated directories.

**Key Features:**
- Read, write, append, and delete files
- Directory creation, listing, and recursive deletion
- File metadata retrieval (size, modification time, is_directory)
- File search with pattern matching
- Path traversal prevention (`../../../etc/passwd` is blocked)
- Optional file extension filtering for security
- Root path restriction for sandboxing

**Basic Usage:**

```rust,no_run
use cloudllm::tools::FileSystemTool;
use std::path::PathBuf;

// Create tool with root path restriction
let fs = FileSystemTool::new()
    .with_root_path(PathBuf::from("/home/user/documents"))
    .with_allowed_extensions(vec!["txt".to_string(), "md".to_string()]);

// Write a file
fs.write_file("notes.txt", "Important information").await?;

// Read a file
let content = fs.read_file("notes.txt").await?;

// List directory contents
let entries = fs.read_directory(".", false).await?;
for entry in entries {
    println!("{}: {} bytes", entry.name, entry.size);
}

// Get metadata
let metadata = fs.get_file_metadata("notes.txt").await?;
println!("Size: {} bytes, Modified: {}", metadata.size, metadata.modified);
```

**Security:**
- All paths are normalized to prevent traversal attacks
- Root path restriction ensures operations stay within designated directory
- Extension filtering can prevent execution of dangerous file types
- Works safely with untrusted input

For comprehensive documentation and examples, see the [`FileSystemTool` API docs](https://docs.rs/cloudllm/latest/cloudllm/tools/struct.FileSystemTool.html) and `examples/filesystem_example.rs`.

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
