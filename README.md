

# CloudLLM

<img src="https://github.com/CloudLLM-ai/cloudllm/blob/master/logo.png?raw=true" width="256"/>

CloudLLM is a Rust library designed to seamlessly bridge applications with remote Language Learning Models (LLMs) across various platforms. With CloudLLM, you can integrate pay-as-you-go LLM APIs like OpenAI's and more, all under one unified abstraction for your app.

## Features

- **Unified Interface**: Interact with multiple LLMs using a single, consistent API.
- **Multi-Participant Sessions**: Orchestrate conversations between multiple LLM clients with different roles and strategies (panels, hierarchies, round-robin discussions).
- **Streaming Support**: First-class streaming for real-time token delivery, dramatically reducing perceived latency.
- **Pay-as-you-go Integration**: Designed to work efficiently with pay-as-you-go LLM platforms.
- **Extendable**: Easily add new LLM platform clients as they emerge.
- **Asynchronous Support**: Built with async operations for non-blocking calls.

## Quick Start

```rust
// Example code on setting up a session and communicating with an LLM (this is just a placeholder for now).
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

## Usage

### Single LLM Session

For basic interactions with a single LLM:

```rust
use std::sync::Arc;
use cloudllm::client_wrapper::Role;
use cloudllm::clients::openai::{OpenAIClient, Model};
use cloudllm::LLMSession;

let client = Arc::new(OpenAIClient::new_with_model_enum(&api_key, Model::GPT4o));
let mut session = LLMSession::new(client, "You are a helpful assistant.".to_string(), 8192);

let response = session.send_message(
    Role::User,
    "What is Rust?".to_string(),
    None,
).await?;

println!("Assistant: {}", response.content);
```

### Multi-Participant Sessions

Create complex multi-agent systems with different orchestration strategies:

```rust
use cloudllm::multi_participant_session::{
    MultiParticipantSession, OrchestrationStrategy, ParticipantRole,
};

// Create a panel discussion
let mut session = MultiParticipantSession::new(
    "You are participating in an AI expert panel.".to_string(),
    8192,
    OrchestrationStrategy::ModeratorLed,
);

session.add_participant("Moderator", moderator_client, ParticipantRole::Moderator);
session.add_participant("Expert-1", expert1_client, ParticipantRole::Panelist);
session.add_participant("Expert-2", expert2_client, ParticipantRole::Panelist);

let responses = session.send_message(
    Role::User,
    "What are the key challenges in AI safety?".to_string(),
    None,
).await?;
```

Refer to the `examples/` directory for more detailed examples, including all orchestration strategies and use cases.

## Contributing

Contributions to CloudLLM are always welcome! Whether it's feature suggestions, bug reporting, or code improvements, all contributions are appreciated.

If you are to send a pull request, please make a separate branch out of `main`. Try to minimize the scope of your contribution to one issue per pull request.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

## Author

**Angel Leon**

---

[CloudLLM.ai](https://cloudllm.ai)
