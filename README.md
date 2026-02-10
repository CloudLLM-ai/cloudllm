# CloudLLM

<p align="center">
  <img src="https://github.com/CloudLLM-ai/cloudllm/blob/master/logo.png?raw=true" width="220" alt="CloudLLM logo" />
</p>

CloudLLM is a batteries-included Rust toolkit for building intelligent agents with LLM integration,
multi-protocol tool support, and multi-agent orchestration. It provides:

* **Agents with Tools**: Create agents that connect to LLMs and execute actions through a flexible,
  multi-protocol tool system (local, remote MCP, Memory, custom protocols) with runtime hot-swapping,
* **Multi-Agent Orchestration**: An [`orchestration`](https://docs.rs/cloudllm/latest/cloudllm/orchestration/index.html) engine
  supporting Parallel, RoundRobin, Moderated, Hierarchical, Debate, and Ralph collaboration patterns,
* **ThoughtChain**: Persistent, SHA-256 hash-chained agent memory with back-references for graph-based
  context resolution and tamper-evident integrity verification,
* **Context Strategies**: Pluggable strategies for handling context window exhaustion — Trim,
  SelfCompression (LLM writes its own save file), and NoveltyAware (entropy-based trigger),
* **Image Generation**: Unified image generation across OpenAI (DALL-E), Grok, and Google Gemini with the
  simplified `register_image_generation_tool()` helper,
* **Server Deployment**: Easy standalone MCP server creation via [`MCPServerBuilder`](https://docs.rs/cloudllm/latest/cloudllm/mcp_server/struct.MCPServerBuilder.html)
  with HTTP, authentication, and IP filtering,
* **Flexible Tool Creation**: From simple Rust closures to advanced custom protocol implementations,
* **Event System**: Real-time observability via [`EventHandler`](https://docs.rs/cloudllm/latest/cloudllm/event/trait.EventHandler.html)
  callbacks for LLM round-trips, tool calls, task completions, and orchestration lifecycle,
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
  - [LLMSession — stateful conversation (OpenAI)](#1-llmsession--stateful-conversation-openai)
  - [Agent — identity + tools (Claude)](#2-agent--identity--tools-claude)
  - [Streaming tokens in real time (Grok)](#3-streaming-tokens-in-real-time-grok)
- [Multi-Agent Orchestration](#multi-agent-orchestration)
  - [Orchestration Modes](#orchestration-modes)
  - [Basic Example: RoundRobin](#basic-example-roundrobin)
  - [Ralph: Autonomous PRD-Driven Loop](#ralph-autonomous-prd-driven-loop)
- [Provider Wrappers](#provider-wrappers)
- [LLMSession: Stateful Conversations](#llmsession-stateful-conversations-the-foundation)
- [Agents: Building Intelligent Workers with Tools](#agents-building-intelligent-workers-with-tools)
- [ThoughtChain: Persistent Agent Memory](#thoughtchain-persistent-agent-memory)
- [Context Strategies: Managing Context Window Exhaustion](#context-strategies-managing-context-window-exhaustion)
- [Agent::fork() — Lightweight Copies for Parallel Execution](#agentfork--lightweight-copies-for-parallel-execution)
- [Runtime Tool Hot-Swapping](#runtime-tool-hot-swapping)
- [Event System: Real-Time Agent & Orchestration Observability](#event-system-real-time-agent--orchestration-observability)
  - [EventHandler Trait](#eventhandler-trait)
  - [AgentEvent Variants](#agentevent-variants)
  - [OrchestrationEvent Variants](#orchestrationevent-variants)
  - [Registering an Event Handler](#registering-an-event-handler)
  - [Full Example: Real-Time Progress Display](#full-example-real-time-progress-display)
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
cloudllm = "0.9.0"
```

The crate targets `tokio` 1.x and Rust 1.70+.

---

## Quick Start

CloudLLM has two core abstractions for talking to LLMs:

| Abstraction | What it is | When to use it |
|-------------|-----------|----------------|
| **LLMSession** | Stateful conversation wrapper around any `ClientWrapper`. Maintains rolling history with automatic context trimming and token accounting. | Simple chat bots, Q&A, any 1-on-1 conversation with an LLM. |
| **Agent** | Wraps LLMSession with an identity (name, expertise, personality), optional tools, persistent ThoughtChain memory, and pluggable context strategies. Can execute actions, not just converse. | Tool-using assistants, orchestrated multi-agent teams, autonomous workflows. |

Think of it this way: **LLMSession is the foundation; Agent builds on top of it.**

### 1. LLMSession — stateful conversation (OpenAI)

```rust,no_run
use std::sync::Arc;
use cloudllm::{LLMSession, Role};
use cloudllm::clients::openai::{Model, OpenAIClient};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = Arc::new(OpenAIClient::new_with_model_enum(
        &std::env::var("OPEN_AI_SECRET")?, Model::GPT41Mini,
    ));

    let mut session = LLMSession::new(client, "You are a concise tutor.".into(), 8_192);

    let reply = session
        .send_message(Role::User, "What is ownership in Rust?".into(), None, None)
        .await?;

    println!("{}", reply.content);
    println!("Tokens used: {:?}", session.token_usage());
    Ok(())
}
```

### 2. Agent — identity + tools (Claude)

An Agent wraps a client just like LLMSession, but adds a name, expertise, personality, and
(optionally) tools. Here the agent uses Anthropic Claude and can answer questions using its
personality and expertise context:

```rust,no_run
use std::sync::Arc;
use cloudllm::Agent;
use cloudllm::clients::claude::{ClaudeClient, Model};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let client = Arc::new(ClaudeClient::new_with_model_enum(
        &std::env::var("ANTHROPIC_KEY")?, Model::ClaudeHaiku45,
    ));

    let agent = Agent::new("tutor", "Rust Tutor", client)
        .with_expertise("Rust programming, ownership, lifetimes")
        .with_personality("Patient teacher who uses short analogies");

    // generate() sends a one-shot prompt through the agent's identity context
    let answer = agent
        .generate(
            "You are a helpful programming tutor.",
            "Explain borrowing vs cloning in two sentences.",
            &[],  // no prior conversation history
        )
        .await?;

    println!("{}", answer);
    Ok(())
}
```

### 3. Streaming tokens in real time (Grok)

Any `ClientWrapper` supports streaming. Here we use xAI Grok:

```rust,no_run
use cloudllm::{LLMSession, Role};
use cloudllm::clients::grok::{GrokClient, Model};
use futures_util::StreamExt;
use std::sync::Arc;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = Arc::new(GrokClient::new_with_model_enum(
        &std::env::var("XAI_KEY")?, Model::Grok3Mini,
    ));
    let mut session = LLMSession::new(client, "You think out loud.".into(), 16_000);

    if let Some(mut stream) = session
        .send_message_stream(Role::User, "Explain type erasure.".into(), None, None)
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
        .run("Evaluate whether the blue/green rollout plan is sufficient.", 2)
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
- **Live progress**: Event handler shows real-time iteration progress, LLM round-trips, tool calls, and task completions (see [Event System](#event-system-real-time-agent--orchestration-observability))

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

    let result = orch.run("Build a Pong game in a single index.html", 1).await?;

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
        .send_message(Role::User, "Tell me about Rust.".into(), None, None)
        .await?;

    println!("Assistant: {}", reply.content);
    println!("Tokens used: {:?}", session.token_usage());
    Ok(())
}
```

---

## Agents: Building Intelligent Workers with Tools

Agents extend LLMSession by adding identity, expertise, and optional tools. They're the primary
way to build sophisticated LLM interactions where you need the agent to take actions beyond
conversation.

The example below creates a single agent with **four tools** attached: the built-in Calculator,
a shared Memory store, image generation via OpenAI, and a custom Fibonacci tool — all on one
`CustomToolProtocol`:

```rust,no_run
use std::sync::Arc;
use cloudllm::Agent;
use cloudllm::clients::openai::{OpenAIClient, Model};
use cloudllm::tool_protocol::{ToolMetadata, ToolParameter, ToolParameterType, ToolResult, ToolRegistry};
use cloudllm::tool_protocols::{CustomToolProtocol, MemoryProtocol};
use cloudllm::tools::{Calculator, Memory};
use cloudllm::cloudllm::image_generation::register_image_generation_tool;
use cloudllm::cloudllm::{ImageGenerationProvider, new_image_generation_client};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let api_key = std::env::var("OPEN_AI_SECRET")?;

    let client = Arc::new(OpenAIClient::new_with_model_enum(&api_key, Model::GPT41Mini));

    // -- Tool protocol (all tools register here) ----------------------------
    let protocol = Arc::new(CustomToolProtocol::new());

    // 1. Calculator — wraps the built-in evaluator
    let calc = Calculator::new();
    protocol.register_async_tool(
        ToolMetadata::new("calculator", "Evaluate a math expression")
            .with_parameter(
                ToolParameter::new("expr", ToolParameterType::String)
                    .with_description("Math expression, e.g. sqrt(2) + mean([1,2,3])")
                    .required(),
            ),
        Arc::new(move |params| {
            let calc = calc.clone();
            Box::pin(async move {
                let expr = params["expr"].as_str().unwrap_or("0");
                match calc.evaluate(expr).await {
                    Ok(val) => Ok(ToolResult::success(serde_json::json!({ "result": val }))),
                    Err(e)  => Ok(ToolResult::failure(e.to_string())),
                }
            })
        }),
    ).await;

    // 2. Image generation — one-liner helper registers the tool
    let image_client = new_image_generation_client(ImageGenerationProvider::OpenAI, &api_key)?;
    register_image_generation_tool(&protocol, image_client).await?;

    // 3. Custom tool — Fibonacci sequence (sync closure, no boilerplate)
    protocol.register_tool(
        ToolMetadata::new("fibonacci", "Return the Nth Fibonacci number")
            .with_parameter(
                ToolParameter::new("n", ToolParameterType::Number)
                    .with_description("Index (0-based)")
                    .required(),
            ),
        Arc::new(|params| {
            let n = params["n"].as_u64().unwrap_or(0) as usize;
            let mut a: u64 = 0;
            let mut b: u64 = 1;
            for _ in 0..n {
                let tmp = a + b;
                a = b;
                b = tmp;
            }
            Ok(ToolResult::success(serde_json::json!({ "fib": a })))
        }),
    ).await;

    // -- Build the registry and the agent -----------------------------------
    // Memory lives in its own protocol so multiple agents can share it
    let memory = Arc::new(Memory::new());
    let mut registry = ToolRegistry::empty();
    registry.add_protocol("tools",  protocol).await?;
    registry.add_protocol("memory", Arc::new(MemoryProtocol::new(memory))).await?;

    let agent = Agent::new("assistant", "Research Assistant", client)
        .with_expertise("Math, memory, image generation, and Fibonacci numbers")
        .with_personality("Thorough and methodical")
        .with_tools(registry);

    println!("Agent '{}' ready with {} tools", agent.name, 4);
    Ok(())
}
```

**Key patterns shown above:**

| Pattern | Used For |
|---------|----------|
| `register_image_generation_tool()` | One-line built-in tool registration |
| `protocol.register_tool(metadata, closure)` | Sync custom tool (Fibonacci) |
| `protocol.register_async_tool(metadata, closure)` | Async tool wrapping a built-in (Calculator) |
| `MemoryProtocol::new(memory)` | Protocol wrapper for built-in Memory |
| `ToolRegistry::empty()` + `add_protocol()` | Multi-protocol registry |
| `agent.with_tools(registry)` | Attach tools to an agent |

---

## ThoughtChain: Persistent Agent Memory

[`ThoughtChain`](https://docs.rs/cloudllm/latest/cloudllm/thought_chain/struct.ThoughtChain.html) is an
append-only, SHA-256 hash-chained, disk-persisted log of agent thoughts. Each thought can carry
back-references to ancestor thoughts, forming a DAG that enables graph-based context resolution.

```text
ThoughtChain (.jsonl on disk)
  ├─ Thought #0  Finding      hash=abc1...   refs=[]
  ├─ Thought #1  Decision     hash=def2...   refs=[]      prev_hash=abc1...
  ├─ Thought #2  Finding      hash=789a...   refs=[]      prev_hash=def2...
  └─ Thought #3  Compression  hash=bcd3...   refs=[0, 2]  prev_hash=789a...
                                                ↑
                             resolve_context(3) walks refs → returns [#0, #2, #3]
```

```rust,no_run
use cloudllm::Agent;
use cloudllm::thought_chain::{ThoughtChain, ThoughtType};
use cloudllm::clients::openai::OpenAIClient;
use std::sync::Arc;
use std::path::PathBuf;
use tokio::sync::RwLock;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let chain = Arc::new(RwLock::new(
        ThoughtChain::open(&PathBuf::from("chains"), "analyst", "Analyst", Some("ML"), None)?
    ));

    let agent = Agent::new(
        "analyst", "Analyst",
        Arc::new(OpenAIClient::new_with_model_string(
            &std::env::var("OPEN_AI_SECRET")?, "gpt-4o",
        )),
    )
    .with_thought_chain(chain);

    // Record findings and decisions as the agent works
    agent.commit(ThoughtType::Finding, "Latency increased 3x after deploy").await?;
    agent.commit(ThoughtType::Decision, "Roll back to v2.3").await?;

    // Verify the hash chain is intact
    let entries = agent.thought_entries().await.unwrap();
    assert_eq!(entries.len(), 2);

    Ok(())
}
```

ThoughtChain files are newline-delimited JSON (`.jsonl`), one thought per line.
Use `ThoughtChain::verify_integrity()` to detect tampering, and
`ThoughtChain::resolve_context(index)` to reconstruct the minimal context
graph for any thought.

Resume a previously running agent from its chain:

```rust,no_run
use cloudllm::Agent;
use cloudllm::thought_chain::ThoughtChain;
use cloudllm::clients::openai::OpenAIClient;
use std::sync::Arc;
use std::path::PathBuf;
use tokio::sync::RwLock;

# fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
let chain = Arc::new(RwLock::new(
    ThoughtChain::open(&PathBuf::from("chains"), "analyst", "Analyst", Some("ML"), None)?
));

// Resume from the latest thought — context graph is injected into a fresh session
let agent = Agent::resume_from_latest(
    "analyst", "Analyst",
    Arc::new(OpenAIClient::new_with_model_string(
        &std::env::var("OPEN_AI_SECRET")?, "gpt-4o",
    )),
    128_000,
    chain,
)?;
# Ok(())
# }
```

---

## Context Strategies: Managing Context Window Exhaustion

The [`ContextStrategy`](https://docs.rs/cloudllm/latest/cloudllm/context_strategy/trait.ContextStrategy.html)
trait lets you plug in different policies for what happens when an agent's conversation history
approaches its token budget.

| Strategy | Trigger | Action |
|----------|---------|--------|
| **TrimStrategy** (default) | Token ratio > 0.85 | No-op — LLMSession's built-in trimming handles it |
| **SelfCompressionStrategy** | Token ratio > 0.80 | LLM writes a structured save file; persisted to ThoughtChain |
| **NoveltyAwareStrategy** | High pressure always; moderate pressure + low novelty | Delegates to inner strategy (typically SelfCompression) |

```rust,no_run
use cloudllm::Agent;
use cloudllm::context_strategy::{NoveltyAwareStrategy, SelfCompressionStrategy};
use cloudllm::clients::openai::OpenAIClient;
use std::sync::Arc;

let agent = Agent::new(
    "analyst", "Analyst",
    Arc::new(OpenAIClient::new_with_model_string("key", "gpt-4o")),
)
.context_collapse_strategy(Box::new(
    NoveltyAwareStrategy::new(Box::new(SelfCompressionStrategy::default()))
        .with_thresholds(0.85, 0.65, 0.25),
));
```

The strategy can also be swapped at runtime via `agent.set_context_collapse_strategy(...)`.

---

## Agent::fork() — Lightweight Copies for Parallel Execution

`Agent` is intentionally not `Clone` (its `LLMSession` contains a bumpalo arena).  Instead, use
`fork()` to create a lightweight copy that shares the same tool registry and thought chain (via
`Arc`) but has a **fresh, empty** session:

```rust,no_run
use cloudllm::Agent;
use cloudllm::clients::openai::OpenAIClient;
use std::sync::Arc;

let agent = Agent::new(
    "analyst", "Analyst",
    Arc::new(OpenAIClient::new_with_model_string("key", "gpt-4o")),
).with_expertise("Cloud Architecture");

// Fork for parallel execution
let forked = agent.fork();
assert_eq!(forked.id, agent.id);
assert_eq!(forked.expertise, agent.expertise);
// forked has an empty session but shares tools and thought chain
```

Orchestration modes (Parallel, Hierarchical) use `fork()` internally when they need
temporary per-task agents.

---

## Runtime Tool Hot-Swapping

The tool registry is wrapped in `Arc<RwLock<ToolRegistry>>`, allowing protocols to be added
or removed while an agent is running:

```rust,no_run
use cloudllm::Agent;
use cloudllm::tool_protocols::CustomToolProtocol;
use cloudllm::clients::openai::OpenAIClient;
use std::sync::Arc;

# async {
let agent = Agent::new(
    "a1", "Agent",
    Arc::new(OpenAIClient::new_with_model_string("key", "gpt-4o")),
);

// Add a protocol at runtime
agent.add_protocol("custom", Arc::new(CustomToolProtocol::new())).await.unwrap();

// List available tools
let tools = agent.list_tools().await;
println!("Tools: {:?}", tools);

// Remove it later
agent.remove_protocol("custom").await;
# };
```

For sharing a mutable registry across agents, use `with_shared_tools()`:

```rust,no_run
use cloudllm::Agent;
use cloudllm::tool_protocol::ToolRegistry;
use cloudllm::clients::openai::OpenAIClient;
use std::sync::Arc;
use tokio::sync::RwLock;

let shared = Arc::new(RwLock::new(ToolRegistry::empty()));
let client = Arc::new(OpenAIClient::new_with_model_string("key", "gpt-4o"));

let agent_a = Agent::new("a", "Agent A", client.clone())
    .with_shared_tools(shared.clone());
let agent_b = Agent::new("b", "Agent B", client)
    .with_shared_tools(shared.clone());
// Adding a protocol via agent_a is visible to agent_b
```

---

## Event System: Real-Time Agent & Orchestration Observability

The [`event`](https://docs.rs/cloudllm/latest/cloudllm/event/index.html) module provides
a callback-based observability layer for agents and orchestrations. Implement the
[`EventHandler`](https://docs.rs/cloudllm/latest/cloudllm/event/trait.EventHandler.html) trait
to receive real-time notifications about LLM round-trips, tool calls, task completions, and more.

This replaces guessing what's happening during long-running orchestrations — you'll see exactly
when each agent starts thinking, which tools it calls, and when the LLM responds.

### EventHandler Trait

```rust,no_run
use cloudllm::event::{AgentEvent, EventHandler, OrchestrationEvent};
use async_trait::async_trait;

struct MyHandler;

#[async_trait]
impl EventHandler for MyHandler {
    async fn on_agent_event(&self, event: &AgentEvent) {
        // Handle agent-level events (LLM calls, tool usage, etc.)
        println!("Agent: {:?}", event);
    }
    async fn on_orchestration_event(&self, event: &OrchestrationEvent) {
        // Handle orchestration-level events (rounds, task completion, etc.)
        println!("Orchestration: {:?}", event);
    }
}
```

Both methods have **default no-op implementations**, so you only need to override the events you
care about. For example, to only observe orchestration-level progress:

```rust,no_run
# use cloudllm::event::{EventHandler, OrchestrationEvent};
# use async_trait::async_trait;
struct ProgressLogger;

#[async_trait]
impl EventHandler for ProgressLogger {
    async fn on_orchestration_event(&self, event: &OrchestrationEvent) {
        match event {
            OrchestrationEvent::RunCompleted { rounds, total_tokens, is_complete, .. } => {
                println!("Done! {} rounds, {} tokens, complete={}", rounds, total_tokens, is_complete);
            }
            _ => {}
        }
    }
}
```

### AgentEvent Variants

Events emitted by an [`Agent`](https://docs.rs/cloudllm/latest/cloudllm/struct.Agent.html)
during its lifecycle. Every variant carries `agent_id` and `agent_name` for identification.

| Variant | Fields | When Emitted |
|---------|--------|--------------|
| **`SendStarted`** | `message_preview` | At the start of `send()` or `generate_with_tokens()` |
| **`SendCompleted`** | `tokens_used`, `tool_calls_made`, `response_length` | When `send()` or `generate_with_tokens()` finishes successfully |
| **`LLMCallStarted`** | `iteration` | Before each LLM round-trip (first call + each tool-loop follow-up) |
| **`LLMCallCompleted`** | `iteration`, `tokens_used`, `response_length` | After each LLM round-trip completes |
| **`ToolCallDetected`** | `tool_name`, `parameters`, `iteration` | When a tool call is parsed from the LLM response |
| **`ToolExecutionCompleted`** | `tool_name`, `parameters`, `success`, `error`, `iteration` | After a tool finishes executing |
| **`ToolMaxIterationsReached`** | _(none extra)_ | When the tool loop hits its iteration cap |
| **`ThoughtCommitted`** | `thought_type` | After a thought is appended to the ThoughtChain |
| **`ProtocolAdded`** | `protocol_name` | When a new tool protocol is added to the agent |
| **`ProtocolRemoved`** | `protocol_name` | When a tool protocol is removed |
| **`SystemPromptSet`** | _(none extra)_ | When the agent's system prompt is set or replaced |
| **`MessageReceived`** | _(none extra)_ | When a message is injected into the agent's session |
| **`Forked`** | _(none extra)_ | When `fork()` creates a lightweight copy (fresh session) |
| **`ForkedWithContext`** | _(none extra)_ | When `fork_with_context()` copies the agent with history |

The `LLMCallStarted`/`LLMCallCompleted` pair is especially useful for understanding latency —
during orchestration you'll see exactly when each agent is waiting on the LLM and when the
response arrives.

### OrchestrationEvent Variants

Events emitted by an
[`Orchestration`](https://docs.rs/cloudllm/latest/cloudllm/orchestration/struct.Orchestration.html)
during a `run()`. Each variant carries `orchestration_id` for identification.

| Variant | Fields | When Emitted |
|---------|--------|--------------|
| **`RunStarted`** | `orchestration_name`, `mode`, `agent_count` | At the start of `run()` |
| **`RunCompleted`** | `orchestration_name`, `rounds`, `total_tokens`, `is_complete` | When `run()` finishes |
| **`RoundStarted`** | `round` | At the start of each round/iteration |
| **`RoundCompleted`** | `round` | At the end of each round/iteration |
| **`AgentSelected`** | `agent_id`, `agent_name`, `reason` | When an agent is chosen to respond (Moderated, Hierarchical modes) |
| **`AgentResponded`** | `agent_id`, `agent_name`, `tokens_used`, `response_length` | After an agent responds successfully |
| **`AgentFailed`** | `agent_id`, `agent_name`, `error` | When an agent encounters an error |
| **`ConvergenceChecked`** | `round`, `score`, `threshold`, `converged` | After similarity check in Debate mode |
| **`RalphIterationStarted`** | `iteration`, `max_iterations`, `tasks_completed`, `tasks_total` | At the start of each RALPH iteration |
| **`RalphTaskCompleted`** | `agent_id`, `agent_name`, `task_ids`, `tasks_completed_total`, `tasks_total` | When a RALPH task is completed by an agent |

### Registering an Event Handler

Wrap your handler in `Arc` and register it via the builder pattern:

**On an Agent:**

```rust,no_run
use std::sync::Arc;
use cloudllm::Agent;
use cloudllm::event::EventHandler;
use cloudllm::clients::openai::OpenAIClient;

# fn example(handler: Arc<dyn EventHandler>) {
let agent = Agent::new("a1", "Agent", Arc::new(
    OpenAIClient::new_with_model_string("key", "gpt-4o"),
))
.with_event_handler(handler);  // builder pattern
# }
```

You can also set or replace the handler at runtime:

```rust,no_run
# use std::sync::Arc;
# use cloudllm::Agent;
# use cloudllm::event::EventHandler;
# use cloudllm::clients::openai::OpenAIClient;
# fn example(handler: Arc<dyn EventHandler>) {
# let mut agent = Agent::new("a1", "Agent", Arc::new(
#     OpenAIClient::new_with_model_string("key", "gpt-4o"),
# ));
agent.set_event_handler(handler);  // runtime mutation
# }
```

**On an Orchestration:**

```rust,no_run
use std::sync::Arc;
use cloudllm::orchestration::{Orchestration, OrchestrationMode};
use cloudllm::event::EventHandler;

# fn example(handler: Arc<dyn EventHandler>) {
let orchestration = Orchestration::new("id", "Name")
    .with_mode(OrchestrationMode::RoundRobin)
    .with_event_handler(handler);  // auto-propagates to agents added later
# }
```

When you register an event handler on an `Orchestration`, it is **automatically propagated** to
every agent added via `add_agent()`. This means agents emit their own `AgentEvent`s through the
same handler, giving you a unified stream of both agent-level and orchestration-level events.

### Full Example: Real-Time Progress Display

This example (adapted from `examples/breakout_game_ralph.rs`) shows a handler that tracks
elapsed time and pretty-prints events as they happen:

```rust,no_run
use async_trait::async_trait;
use cloudllm::event::{AgentEvent, EventHandler, OrchestrationEvent};
use std::time::Instant;
use std::sync::Arc;

struct ProgressHandler {
    start: Instant,
}

impl ProgressHandler {
    fn new() -> Self { Self { start: Instant::now() } }

    fn elapsed(&self) -> String {
        let secs = self.start.elapsed().as_secs();
        format!("{:02}:{:02}", secs / 60, secs % 60)
    }
}

#[async_trait]
impl EventHandler for ProgressHandler {
    async fn on_agent_event(&self, event: &AgentEvent) {
        match event {
            AgentEvent::SendStarted { agent_name, message_preview, .. } => {
                let preview = &message_preview[..80.min(message_preview.len())];
                println!("  [{}] >> {} thinking... ({}...)", self.elapsed(), agent_name, preview);
            }
            AgentEvent::SendCompleted { agent_name, tokens_used, response_length, tool_calls_made, .. } => {
                let tokens = tokens_used.as_ref().map(|u| u.total_tokens).unwrap_or(0);
                println!("  [{}] << {} responded ({} chars, {} tokens, {} tool calls)",
                    self.elapsed(), agent_name, response_length, tokens, tool_calls_made);
            }
            AgentEvent::LLMCallStarted { agent_name, iteration, .. } => {
                println!("  [{}]    {} sending to LLM (round {})...", self.elapsed(), agent_name, iteration);
            }
            AgentEvent::LLMCallCompleted { agent_name, iteration, tokens_used, response_length, .. } => {
                let tokens = tokens_used.as_ref()
                    .map(|u| format!("{} tokens", u.total_tokens))
                    .unwrap_or_else(|| "no token info".to_string());
                println!("  [{}]    {} LLM round {} complete ({} chars, {})",
                    self.elapsed(), agent_name, iteration, response_length, tokens);
            }
            AgentEvent::ToolCallDetected { agent_name, tool_name, parameters, iteration, .. } => {
                println!("  [{}]    {} calling tool '{}' (iter {}), params={}",
                    self.elapsed(), agent_name, tool_name, iteration,
                    serde_json::to_string(parameters).unwrap_or_default());
            }
            AgentEvent::ToolExecutionCompleted { agent_name, tool_name, success, error, .. } => {
                if *success {
                    println!("  [{}]    {} tool '{}' succeeded", self.elapsed(), agent_name, tool_name);
                } else {
                    println!("  [{}]    {} tool '{}' FAILED: {}",
                        self.elapsed(), agent_name, tool_name, error.as_deref().unwrap_or("unknown"));
                }
            }
            _ => {}
        }
    }

    async fn on_orchestration_event(&self, event: &OrchestrationEvent) {
        match event {
            OrchestrationEvent::RunStarted { orchestration_name, mode, agent_count, .. } => {
                println!("\n{}\n  {} — mode={}, agents={}\n{}",
                    "=".repeat(70), orchestration_name, mode, agent_count, "=".repeat(70));
            }
            OrchestrationEvent::RalphIterationStarted { iteration, max_iterations, tasks_completed, tasks_total, .. } => {
                println!("\n  RALPH Iteration {}/{} — {}/{} tasks complete",
                    iteration, max_iterations, tasks_completed, tasks_total);
            }
            OrchestrationEvent::RalphTaskCompleted { agent_name, task_ids, tasks_completed_total, tasks_total, .. } => {
                println!("  [{}] *** {} completed tasks: [{}] — progress: {}/{}",
                    self.elapsed(), agent_name, task_ids.join(", "), tasks_completed_total, tasks_total);
            }
            OrchestrationEvent::AgentFailed { agent_name, error, .. } => {
                println!("  [{}] !!! {} FAILED: {}", self.elapsed(), agent_name, error);
            }
            OrchestrationEvent::RunCompleted { rounds, total_tokens, is_complete, .. } => {
                println!("\n{}\n  Run complete — {} rounds, {} tokens, complete={}\n{}",
                    "=".repeat(70), rounds, total_tokens, is_complete, "=".repeat(70));
            }
            _ => {}
        }
    }
}

// Register on an orchestration (auto-propagates to all agents):
// let handler = Arc::new(ProgressHandler::new());
// let orchestration = Orchestration::new("id", "Name")
//     .with_event_handler(handler);
```

**Sample output during a RALPH run:**

```text
======================================================================
  Breakout Game RALPH Orchestration — mode=Ralph, agents=4
======================================================================

  RALPH Iteration 1/5 — 0/10 tasks complete
  [00:00] >> Game Architect thinking... (Build a complete Atari Breakout game...)
  [00:00]    Game Architect sending to LLM (round 1)...
  [00:22]    Game Architect LLM round 1 complete (8923 chars, 3241 tokens)
  [00:22]    Game Architect calling tool 'write_game_file' (iter 1), params={"filename":"breakout_game.html",...}
  [00:22]    Game Architect tool 'write_game_file' succeeded
  [00:22]    Game Architect sending to LLM (round 2)...
  [00:35]    Game Architect LLM round 2 complete (412 chars, 158 tokens)
  [00:35] << Game Architect responded (412 chars, 3399 tokens, 1 tool calls)
  [00:35] *** Game Architect completed tasks: [html_structure, game_loop] — progress: 2/10
  [00:35] >> Game Programmer thinking... (Build a complete Atari Breakout game...)
  ...
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
            _ => Ok(ToolResult::failure("Unknown tool".to_string()))
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

    let registry = ToolRegistry::new(protocol);

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
    let registry = ToolRegistry::new(protocol);

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
use cloudllm::Agent;
use cloudllm::clients::openai::{OpenAIClient, Model};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create shared memory for agents
    let memory = Arc::new(Memory::new());

    // Wrap with protocol for agent usage
    let protocol = Arc::new(MemoryProtocol::new(memory.clone()));
    let registry = ToolRegistry::new(protocol);

    // Create agent with memory access
    let agent = Agent::new(
        "researcher",
        "Research Agent",
        Arc::new(OpenAIClient::new_with_model_enum(
            &std::env::var("OPEN_AI_SECRET")?,
            Model::GPT41Mini
        )),
    )
    .with_tools(registry);

    // Agent can now use memory via commands like:
    // "P research_state Gathering data 7200"
    // "G research_state META"
    // "L"

    Ok(())
}
```

**Memory Protocol Commands (for agents):**

The Memory tool uses a token-efficient protocol designed for LLM communication:

| Command | Syntax | Example | Use Case |
|---------|--------|---------|----------|
| **Put** | `P <key> <value> [ttl_seconds]` | `P task_status InProgress 3600` | Store state with 1-hour expiration |
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
use cloudllm::{Agent, orchestration::{Orchestration, OrchestrationMode}};
use tokio::sync::RwLock;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create shared memory (all agents access same instance)
    let shared_memory = Arc::new(Memory::new());

    let protocol = Arc::new(MemoryProtocol::new(shared_memory));
    let shared_registry = Arc::new(RwLock::new(ToolRegistry::new(protocol)));

    // Create orchestration of agents — shared registry is visible to both
    let agent1 = Agent::new(...)
        .with_shared_tools(shared_registry.clone());

    let agent2 = Agent::new(...)
        .with_shared_tools(shared_registry.clone());

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
    let registry = ToolRegistry::new(protocol);

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
