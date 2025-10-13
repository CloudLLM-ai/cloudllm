

# CloudLLM

<img src="https://github.com/CloudLLM-ai/cloudllm/blob/master/logo.png?raw=true" width="256"/>

CloudLLM is a Rust library designed to seamlessly bridge applications with remote Language Learning Models (LLMs) across various platforms. With CloudLLM, you can integrate pay-as-you-go LLM APIs like OpenAI's and more, all under one unified abstraction for your app.

## Features

- **Unified Interface**: Interact with multiple LLMs using a single, consistent API.
- **Streaming Support**: First-class streaming for real-time token delivery, dramatically reducing perceived latency.
- **Pay-as-you-go Integration**: Designed to work efficiently with pay-as-you-go LLM platforms.
- **Extendable**: Easily add new LLM platform clients as they emerge.
- **Asynchronous Support**: Built with async operations for non-blocking calls.
- **Multi-LLM Councils**: Combine heterogeneous LLM clients into a structured conversation (moderators, panelists, observers) for richer deliberation.

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

Refer to the `examples/` directory to see how you can set up sessions and interact with various LLM platforms using CloudLLM.

### Multi-LLM Councils

CloudLLM now provides a `CouncilSession` abstraction for orchestrating round-robin conversations between multiple LLM clients. You can attach OpenAI, Grok, Gemini, Claude (or any other `ClientWrapper`) to the same session, assign them roles, and gather their perspectives in ordered rounds.

```rust
use std::sync::Arc;
use cloudllm::{CouncilRole, CouncilSession, Role};
use cloudllm::clients::openai::OpenAIClient;
use cloudllm::clients::grok::GrokClient;

let moderator = Arc::new(OpenAIClient::new("OPENAI_KEY", "gpt-4o"));
let panelist = Arc::new(GrokClient::new("GROK_KEY", "grok-2"));

let mut council = CouncilSession::new("You are DiarioBitcoin's expert editorial board.");
council.add_participant(moderator, CouncilRole::Moderator);
council.add_participant(panelist, CouncilRole::Panelist);

let round = council
    .send_message(Role::User, "¿Cómo explicamos el último halving a nuevos lectores?".into(), None)
    .await?;

for reply in round.replies {
    println!("{} => {}", reply.name, reply.message.content);
}
```

Use `ParticipantConfig` to provide custom display names, persona prompts, or per-model context window sizes, and `set_round_robin_order` to override the default moderator-first speaking sequence.

## Contributing

Contributions to CloudLLM are always welcome! Whether it's feature suggestions, bug reporting, or code improvements, all contributions are appreciated.

If you are to send a pull request, please make a separate branch out of `main`. Try to minimize the scope of your contribution to one issue per pull request.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

## Author

**Angel Leon**

---

[CloudLLM.ai](https://cloudllm.ai)
