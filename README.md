

# CloudLLM

<img src="https://github.com/CloudLLM-ai/cloudllm/blob/master/logo.png?raw=true" width="256"/>

CloudLLM is a Rust library designed to seamlessly bridge applications with remote Language Learning Models (LLMs) across various platforms. With CloudLLM, you can integrate pay-as-you-go LLM APIs like OpenAI's and more, all under one unified abstraction for your app.

## Features

- **Unified Interface**: Interact with multiple LLMs using a single, consistent API.
- **Multi-Agent Councils**: Orchestrate multiple LLM agents in parallel, hierarchical, or debate modes for complex problem-solving.
- **Tool Protocol Abstraction**: Connect agents to tools via MCP, custom functions, or your own protocol adapters.
- **Streaming Support**: First-class streaming for real-time token delivery, dramatically reducing perceived latency.
- **Pay-as-you-go Integration**: Designed to work efficiently with pay-as-you-go LLM platforms.
- **Extendable**: Easily add new LLM platform clients as they emerge.
- **Asynchronous Support**: Built with async operations for non-blocking calls.

## Quick Start

### Simple LLM Session

```rust
use cloudllm::{LLMSession, Role};
use cloudllm::clients::openai::OpenAIClient;
use std::sync::Arc;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = OpenAIClient::new_with_model_string("your-api-key", "gpt-4o");
    let mut session = LLMSession::new(
        Arc::new(client),
        "You are a helpful assistant.".to_string(),
        8192
    );

    let response = session.send_message(
        Role::User,
        "What is Rust?".to_string(),
        None
    ).await?;

    println!("Assistant: {}", response.content);
    Ok(())
}
```

### Multi-Agent Council

```rust
use cloudllm::council::{Council, CouncilMode, Agent};
use cloudllm::clients::openai::OpenAIClient;
use std::sync::Arc;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create diverse agents
    let architect = Agent::new(
        "architect",
        "System Architect",
        Arc::new(OpenAIClient::new_with_model_string("key", "gpt-4o"))
    ).with_expertise("Distributed systems, scalability");

    let security = Agent::new(
        "security",
        "Security Expert",
        Arc::new(OpenAIClient::new_with_model_string("key", "gpt-4o"))
    ).with_expertise("Application security, threat modeling");

    // Create a council with parallel mode
    let mut council = Council::new("tech-council", "Technical Council")
        .with_mode(CouncilMode::Parallel)
        .with_max_tokens(8192);

    council.add_agent(architect)?;
    council.add_agent(security)?;

    // Get expert panel analysis
    let response = council.discuss(
        "How should we architect a payment processing system?",
        1
    ).await?;

    for msg in response.messages {
        if let Some(name) = msg.agent_name {
            println!("{}: {}\n", name, msg.content);
        }
    }

    Ok(())
}
```

### Agents with Tools

```rust
use cloudllm::council::Agent;
use cloudllm::tool_adapters::CustomToolAdapter;
use cloudllm::tool_protocol::{ToolMetadata, ToolParameter, ToolParameterType, ToolRegistry, ToolResult};
use std::sync::Arc;

// Create a tool adapter
let mut adapter = CustomToolAdapter::new();

// Register a custom tool
adapter.register_tool(
    ToolMetadata::new("calculate", "Performs calculations")
        .with_parameter(
            ToolParameter::new("expression", ToolParameterType::String)
                .with_description("Math expression")
                .required()
        ),
    Arc::new(|params| {
        // Tool implementation
        Ok(ToolResult::success(serde_json::json!({"result": 42})))
    })
).await;

// Create an agent with tools
let agent = Agent::new("analyst", "Data Analyst", client)
    .with_tools(Arc::new(ToolRegistry::new(Arc::new(adapter))));
```

## Installation

Add CloudLLM to your `Cargo.toml`:

```toml
[dependencies]
cloudllm = "0.3.0" # Use the latest version
```

## Supported LLM Platforms

- OpenAI
- Grok
- Gemini
- Claude
- AWS Bedrock (Coming Soon)
- ... and more to come!

## Council Modes

CloudLLM supports multiple collaboration patterns for multi-agent systems:

### Parallel Mode
All agents respond simultaneously. Best for independent analysis from multiple perspectives.

### RoundRobin Mode
Agents take turns, building on previous responses. Ideal for iterative refinement.

### Moderated Mode
One agent orchestrates the discussion, directing questions to appropriate experts.

### Hierarchical Mode
Multi-layer processing: workers → supervisors → executives. Perfect for complex problem decomposition.

### Debate Mode
Agents engage in discussion until reaching convergence. Great for exploring tradeoffs.

## Tool Integration

Agents can use tools through a flexible protocol abstraction:

- **MCP Adapter**: Standard Model Context Protocol support
- **Custom Adapter**: Register Rust functions as tools
- **OpenAI Functions**: Compatible with OpenAI function calling format
- **Extensible**: Implement your own ToolProtocol for any tool system

## Examples

Run the comprehensive demo to see all features in action:

```bash
export OPENAI_KEY=your_key_here
cargo run --example council_demo
```

Additional examples in the `examples/` directory:
- `interactive_session.rs` - Basic LLM session
- `streaming_example.rs` - Streaming responses
- `council_demo.rs` - Multi-agent councils with all modes
- `venezuela_regime_change_debate.rs` - Strategic debate with 5 specialized agents analyzing geopolitical scenarios
- `digimon_vs_pokemon_debate.rs` - Fun moderated debate between two experts arguing Digimon vs Pokemon

## Contributing

Contributions to CloudLLM are always welcome! Whether it's feature suggestions, bug reporting, or code improvements, all contributions are appreciated.

If you are to send a pull request, please make a separate branch out of `main`. Try to minimize the scope of your contribution to one issue per pull request.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

## Author

**Angel Leon**

---

[CloudLLM.ai](https://cloudllm.ai)
