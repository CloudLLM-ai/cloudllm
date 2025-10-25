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
cloudllm = "0.4.0"
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
