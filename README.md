# CloudLLM

<p align="center">
  <img src="https://github.com/CloudLLM-ai/cloudllm/blob/master/logo.png?raw=true" width="220" alt="CloudLLM logo" />
</p>

CloudLLM is a batteries-included Rust toolkit for building intelligent agents with LLM integration,
multi-protocol tool support, and multi-agent orchestration. It provides:

* **Agents with Tools**: Create agents that connect to LLMs and execute actions through a flexible,
  multi-protocol tool system (local, remote MCP, Memory, custom protocols),
* **Multi-Agent Orchestration**: An [`orchestration`](https://docs.rs/cloudllm/latest/cloudllm/orchestration/index.html) engine
  supporting Parallel, RoundRobin, Moderated, Hierarchical, Debate, and Ralph collaboration patterns,
* **Image Generation**: Unified image generation across OpenAI (DALL-E), Grok, and Google Gemini with the
  simplified `register_image_generation_tool()` helper,
* **Server Deployment**: Easy standalone MCP server creation via [`MCPServerBuilder`](https://docs.rs/cloudllm/latest/cloudllm/mcp_server/struct.MCPServerBuilder.html)
  with HTTP, authentication, and IP filtering,
* **Flexible Tool Creation**: From simple Rust closures to advanced custom protocol implementations,
* **Stateful Sessions**: A [`LLMSession`](https://docs.rs/cloudllm/latest/cloudllm/struct.LLMSession.html) for
  managing conversation history with context trimming and token accounting,
* **Provider Flexibility**: Unified [`ClientWrapper`](https://docs.rs/cloudllm/latest/cloudllm/client_wrapper/index.html)
  trait for OpenAI, Claude, Gemini, Grok, and custom OpenAI-compatible endpoints.

The entire public API is documented with _compilable_ examples—run `cargo doc --open` to browse the
crate-level manual.

---

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Multi-Agent Orchestration](#multi-agent-orchestration)
  - [Orchestration Modes](#orchestration-modes)
  - [Basic Example: RoundRobin](#basic-example-roundrobin)
  - [Ralph: Autonomous PRD-Driven Loop](#ralph-autonomous-prd-driven-loop)
- [Provider Wrappers](#provider-wrappers)
- [LLMSession: Stateful Conversations](#llmsession-stateful-conversations-the-foundation)
- [Agents: Building Intelligent Workers with Tools](#agents-building-intelligent-workers-with-tools)
- [Tool Registry: Multi-Protocol Tool Access](#tool-registry-multi-protocol-tool-access)
- [Deploying Tool Servers with MCPServerBuilder](#deploying-tool-servers-with-mcpserverbuilder)
- [Creating Tools: Simple to Advanced](#creating-tools-simple-to-advanced)
  - [Simple Tool Creation: Rust Closures](#simple-tool-creation-rust-closures)
  - [Advanced Tool Creation: Custom Protocol Implementation](#advanced-tool-creation-custom-protocol-implementation)
  - [Using Tools with Agents](#using-tools-with-agents)
  - [Protocol Implementations](#protocol-implementations)
  - [Built-in Tools](#built-in-tools)
- [Image Generation](#image-generation)
- [Examples](#examples)
- [Support & Contributing](#support--contributing)

---

## Installation

Add CloudLLM to your project:

```toml
[dependencies]
cloudllm = "0.8.0"
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

## Multi-Agent Orchestration

The [`orchestration`](https://docs.rs/cloudllm/latest/cloudllm/orchestration/index.html) module
coordinates conversations between multiple LLM agents. Each agent can have its own provider,
expertise, personality, and tool access. Choose from six collaboration patterns depending on your
use case.

### Orchestration Modes

| Mode | Description | Best For |
|------|-------------|----------|
| **Parallel** | All agents respond simultaneously; results are aggregated | Fast fan-out queries, getting diverse perspectives |
| **RoundRobin** | Agents take sequential turns, each building on previous responses | Iterative refinement, structured review |
| **Moderated** | Agents propose ideas, a moderator synthesizes the final answer | Consensus building, curated outputs |
| **Hierarchical** | Lead agent coordinates; specialists handle specific aspects | Complex tasks with delegation |
| **Debate** | Agents discuss and challenge until convergence is reached | Critical analysis, stress-testing ideas |
| **Ralph** | Autonomous iterative loop working through a PRD task list | Multi-step builds, code generation, structured project work |

### Basic Example: RoundRobin

```rust,no_run
use std::sync::Arc;

use cloudllm::orchestration::{Agent, Orchestration, OrchestrationMode};
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

    let mut orchestration = Orchestration::new("design-review", "Deployment Review")
        .with_mode(OrchestrationMode::RoundRobin)
        .with_system_context("Collaboratively review the proposed architecture.");

    orchestration.add_agent(architect)?;
    orchestration.add_agent(tester)?;

    let outcome = orchestration
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

### Ralph: Autonomous PRD-Driven Loop

**Ralph** (named after Ralph Wiggum) is an autonomous iterative orchestration mode where agents
work through a structured PRD (Product Requirements Document) task list. Each iteration presents
agents with the current task checklist. Agents signal completion by including
`[TASK_COMPLETE:task_id]` markers in their responses. The loop ends when all tasks are done or
`max_iterations` is reached.

Key features:
- **PRD-driven**: Structured `RalphTask` items with id, title, and description
- **Completion detection**: Agents include `[TASK_COMPLETE:task_id]` markers
- **Progress tracking**: `convergence_score` reports task completion fraction (0.0 to 1.0)
- **History trimming**: Conversation history is automatically trimmed to fit within `max_tokens`,
  keeping the most recent messages
- **Live progress**: `log::info!` output shows iteration progress, agent calls, and task completions

```rust,no_run
use std::sync::Arc;

use cloudllm::orchestration::{Orchestration, OrchestrationMode, RalphTask};
use cloudllm::clients::claude::{ClaudeClient, Model};
use cloudllm::Agent;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let key = std::env::var("ANTHROPIC_KEY")?;
    let make_client = || Arc::new(ClaudeClient::new_with_model_enum(&key, Model::ClaudeHaiku45));

    let frontend = Agent::new("frontend", "Frontend Dev", make_client())
        .with_expertise("HTML, CSS, Canvas");
    let backend = Agent::new("backend", "Backend Dev", make_client())
        .with_expertise("JavaScript, game logic");

    let tasks = vec![
        RalphTask::new("html",  "HTML Structure", "Create the HTML boilerplate and canvas"),
        RalphTask::new("loop",  "Game Loop",      "Implement requestAnimationFrame game loop"),
        RalphTask::new("input", "Controls",       "Add keyboard input for the paddle"),
    ];

    let mut orch = Orchestration::new("game-builder", "Game Builder")
        .with_mode(OrchestrationMode::Ralph {
            tasks,
            max_iterations: 5,
        })
        .with_system_context("Build a game. Output full HTML. Mark done with [TASK_COMPLETE:id].")
        .with_max_tokens(180_000);

    orch.add_agent(frontend)?;
    orch.add_agent(backend)?;

    let result = orch.discuss("Build a Pong game in a single index.html", 1).await?;

    println!("Iterations: {}",  result.round);
    println!("Complete: {}",    result.is_complete);
    println!("Progress: {:.0}%", result.convergence_score.unwrap_or(0.0) * 100.0);
    println!("Tokens: {}",     result.total_tokens_used);

    Ok(())
}
```

See `examples/breakout_game_ralph.rs` for a full working example that orchestrates 4 agents
through 10 PRD tasks to produce a complete Atari Breakout game with multi-hit bricks, powerups,
chiptune music, and collision sound effects.

For a deep dive into all collaboration modes, read
[`ORCHESTRATION_TUTORIAL.md`](./ORCHESTRATION_TUTORIAL.md).

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

CloudLLM provides a powerful, protocol-agnostic tool system that works seamlessly with agents and orchestrations.
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

    println!("Agent ready with tools");
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
- `sqrt(16)` -> 4.0
- `log(100)` -> 2.0 (base 10)
- `std([1, 2, 3, 4, 5])` -> 1.581 (sample standard deviation)
- `floor(3.7)` -> 3.0

For comprehensive documentation, see [`Calculator` API docs](https://docs.rs/cloudllm/latest/cloudllm/tools/struct.Calculator.html).

#### Memory Tool

A persistent, TTL-aware key-value store for maintaining agent state across sessions. Perfect for single agents to track progress or multi-agent orchestrations to coordinate decisions.

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
use cloudllm::orchestration::Agent;
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

**Multi-Agent Memory Sharing:**

```rust,no_run
use std::sync::Arc;
use cloudllm::tools::Memory;
use cloudllm::tool_protocols::MemoryProtocol;
use cloudllm::tool_protocol::ToolRegistry;
use cloudllm::orchestration::{Agent, Orchestration, OrchestrationMode};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create shared memory (all agents access same instance)
    let shared_memory = Arc::new(Memory::new());

    let protocol = Arc::new(MemoryProtocol::new(shared_memory));
    let registry = Arc::new(ToolRegistry::new(protocol));

    // Create orchestration of agents
    let agent1 = Agent::new(...)
        .with_tools(registry.clone());

    let agent2 = Agent::new(...)
        .with_tools(registry.clone());

    // Both agents access same memory
    let mut orchestration = Orchestration::new("research", "Collaborative Research");
    orchestration.add_agent(agent1)?;
    orchestration.add_agent(agent2)?;

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

**Security Best Practices:**
- **Domain Allowlist**: `client.allow_domain("api.trusted-service.com")`
- **Deny Malicious Domains**: `client.deny_domain("malicious.attacker.com")`
- **Timeout Protection**: `client.with_timeout(Duration::from_secs(30))`
- **Size Limits**: `client.with_max_response_size(10 * 1024 * 1024)` (10MB)
- **Authentication**: `client.with_basic_auth("user", "pass")` or `client.with_header("Authorization", "Bearer token")`

For comprehensive documentation, see [`HttpClient` API docs](https://docs.rs/cloudllm/latest/cloudllm/tools/struct.HttpClient.html) and `examples/http_client_example.rs`.

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

For comprehensive documentation, see the [`FileSystemTool` API docs](https://docs.rs/cloudllm/latest/cloudllm/tools/struct.FileSystemTool.html) and `examples/filesystem_example.rs`.

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

### Best Practices for Tools

1. **Clear Names & Descriptions**: Make tool purposes obvious to LLMs
2. **Comprehensive Parameters**: Document all required and optional parameters
3. **Error Handling**: Return meaningful error messages in ToolResult
4. **Atomicity**: Each tool should do one thing well
5. **Documentation**: Include examples in tool descriptions
6. **Testing**: Test tool execution in isolation before integration

For more examples, see the `examples/` directory and run `cargo doc --open` for complete API documentation.

---

## Image Generation

CloudLLM provides unified image generation across OpenAI, Grok, and Google Gemini. The new `register_image_generation_tool()` helper dramatically simplifies adding image generation capabilities to agents.

### Quick Start: Image Generation Tool

Register an image generation tool with a single line:

```rust,no_run
use std::sync::Arc;
use cloudllm::Agent;
use cloudllm::clients::openai::{OpenAIClient, Model};
use cloudllm::cloudllm::image_generation::register_image_generation_tool;
use cloudllm::cloudllm::{ImageGenerationProvider, new_image_generation_client};
use cloudllm::tool_protocols::CustomToolProtocol;
use cloudllm::tool_protocol::ToolRegistry;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let api_key = std::env::var("OPEN_AI_SECRET")?;

    // Create image generation client (choose provider: OpenAI, Grok, or Gemini)
    let image_client = new_image_generation_client(
        ImageGenerationProvider::OpenAI,
        &api_key,
    )?;

    // Create a tool protocol
    let protocol = Arc::new(CustomToolProtocol::new());

    // Register the image generation tool (much simpler than manual implementation!)
    let rt = tokio::runtime::Runtime::new()?;
    rt.block_on(register_image_generation_tool(&protocol, image_client.clone()))?;

    // Create agent with image generation capability
    let registry = Arc::new(ToolRegistry::new(protocol));

    let agent = Agent::new(
        "designer",
        "Creative Designer",
        Arc::new(OpenAIClient::new_with_model_enum(&api_key, Model::GPT41Mini)),
    )
    .with_tools(registry)
    .with_expertise("Creating visual content")
    .with_personality("Creative and detailed");

    println!("Agent created with image generation capability");
    Ok(())
}
```

### Supported Providers

| Provider | Model | Supported Ratios |
|----------|-------|------------------|
| OpenAI (DALL-E 3) | `gpt-image-1.5` | 1:1, 16:9, 4:3, 3:2, 9:16, 3:4, 2:3 |
| Grok Imagine | `grok-2-image-1212` | 1:1, 16:9, 4:3, 3:2, 9:16, 3:4, 2:3, and more |
| Google Gemini | `gemini-2.5-flash-image` | 1:1, 2:3, 3:2, 3:4, 4:3, 4:5, 5:4, 9:16, 16:9, 21:9 |

### Using Different Providers

```rust,no_run
use cloudllm::cloudllm::{ImageGenerationProvider, new_image_generation_client};

// OpenAI (realistic, high-quality)
let client = new_image_generation_client(
    ImageGenerationProvider::OpenAI,
    &std::env::var("OPEN_AI_SECRET")?,
)?;

// Grok (fast, creative)
let client = new_image_generation_client(
    ImageGenerationProvider::Grok,
    &std::env::var("XAI_KEY")?,
)?;

// Gemini (flexible aspect ratios)
let client = new_image_generation_client(
    ImageGenerationProvider::Gemini,
    &std::env::var("GEMINI_API_KEY")?,
)?;
```

### Parsing from Strings with FromStr

For dynamic provider selection from strings, use the `FromStr` trait:

```rust,no_run
use cloudllm::cloudllm::{ImageGenerationProvider, new_image_generation_client};
use std::str::FromStr;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let provider_name = "grok";  // From config, user input, etc.

    // Parse string to enum using FromStr trait
    let provider = ImageGenerationProvider::from_str(provider_name)?;

    // Create client with parsed provider
    let client = new_image_generation_client(
        provider,
        &std::env::var("XAI_KEY")?,
    )?;

    println!("Using provider: {}", provider.display_name());
    Ok(())
}
```

**Supported provider strings (case-insensitive):**
- `"openai"` -> OpenAI (DALL-E 3)
- `"grok"` -> Grok Imagine
- `"gemini"` -> Google Gemini

For comprehensive documentation, see the [`image_generation` module docs](https://docs.rs/cloudllm/latest/cloudllm/cloudllm/image_generation/index.html).

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
cargo run --example orchestration_demo
cargo run --example breakout_game_ralph
```

Each example corresponds to a module in the documentation so you can cross-reference the code with
explanations.

---

## Support & contributing

Issues and pull requests are welcome via [GitHub](https://github.com/CloudLLM-ai/cloudllm).
Please open focused pull requests against `main` and include tests or doc updates where relevant.

CloudLLM is released under the [MIT License](./LICENSE).

---

Happy orchestration!
