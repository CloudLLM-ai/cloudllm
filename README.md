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

Describe tools once, reuse them everywhere:

```rust,no_run
use std::sync::Arc;

use cloudllm::tool_adapters::CustomToolAdapter;
use cloudllm::tool_protocol::{ToolMetadata, ToolParameter, ToolParameterType, ToolRegistry, ToolResult};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let adapter = Arc::new(CustomToolAdapter::new());

    adapter
        .register_tool(
            ToolMetadata::new("area", "Compute the area of a rectangle")
                .with_parameter(ToolParameter::new("width", ToolParameterType::Number).required())
                .with_parameter(ToolParameter::new("height", ToolParameterType::Number).required()),
            Arc::new(|params| {
                let width = params["width"].as_f64().unwrap();
                let height = params["height"].as_f64().unwrap();
                Ok(ToolResult::success(serde_json::json!({"area": width * height})))
            }),
        )
        .await;

    let registry = ToolRegistry::new(adapter.clone());
    assert_eq!(registry.list_tools()[0].name, "area");
    Ok(())
}
```

Adapters for the Model Context Protocol (`McpAdapter`) and OpenAI function calling (`OpenAIFunctionAdapter`)
are ready to use.  Implementors can create additional adapters by implementing
[`ToolProtocol`](https://docs.rs/cloudllm/latest/cloudllm/tool_protocol/trait.ToolProtocol.html).

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
