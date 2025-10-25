//! # CloudLLM
//!
//! CloudLLM is a batteries-included toolkit for orchestrating conversations with remote
//! Large Language Models (LLMs).  It exposes a set of carefully layered abstractions that
//! let you:
//!
//! * compose provider-agnostic clients (`ClientWrapper` implementors such as
//!   [`OpenAIClient`](crate::clients::openai::OpenAIClient), [`ClaudeClient`](crate::clients::claude::ClaudeClient)),
//! * drive stateful conversations via [`LLMSession`],
//! * coordinate multi-agent discussions with the [`council`] module, and
//! * integrate structured tool calling through [`tool_protocol`] and
//!   [`tool_adapters`].
//!
//! The crate aims to provide documentation-quality examples for every public API.  These
//! examples are kept up to date and are written to compile under `cargo test --doc`.
//!
//! ## Feature Highlights
//!
//! ### Provider Abstraction
//!
//! Each cloud provider (OpenAI, Anthropic/Claude, Google Gemini, xAI Grok, and custom OpenAI-
//! compatible endpoints) is exposed as a `ClientWrapper` implementation.  All wrappers share
//! the same ergonomics for synchronous calls, streaming, and token accounting.
//!
//! ### Stateful Sessions
//!
//! [`LLMSession`] wraps a client to maintain a rolling conversation history.  It offers
//! predictive and post-hoc context trimming so you can respect provider token budgets while
//! still benefiting from long running conversations.
//!
//! ### Tooling & Councils
//!
//! The [`tool_protocol`] module defines a protocol-agnostic vocabulary for tools.  Concrete
//! adapters link that vocabulary to the Model Context Protocol (MCP), OpenAI function calling,
//! or simple Rust closures.  The [`council`] module then builds on top of those tools to
//! orchestrate multi-agent conversations across a variety of collaboration patterns.
//!
//! ## Getting Started
//!
//! ```rust,no_run
//! use cloudllm::clients::openai::{Model, OpenAIClient};
//! use cloudllm::client_wrapper::{ClientWrapper, Message, Role};
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     cloudllm::init_logger();
//!
//!     let api_key = std::env::var("OPEN_AI_SECRET")?;
//!     let client = OpenAIClient::new_with_model_enum(&api_key, Model::GPT41Nano);
//!
//!     let response = client
//!         .send_message(
//!             &[
//!                 Message { role: Role::System, content: "You are terse.".into() },
//!                 Message { role: Role::User, content: "Summarise CloudLLM in one sentence.".into() },
//!             ],
//!             None,
//!         )
//!         .await?;
//!
//!     println!("{}", response.content);
//!     Ok(())
//! }
//! ```
//!
//! Continue exploring the modules re-exported from the crate root for progressively richer
//! interaction patterns.

use std::sync::Once;

static INIT_LOGGER: Once = Once::new();

/// Initialise the global [`env_logger`] subscriber exactly once.
///
/// The helper is intentionally lightweight so that applications embedding CloudLLM can opt-in
/// to simple `RUST_LOG` driven diagnostics without having to choose a specific logging backend
/// upfront.
///
/// ```rust
/// cloudllm::init_logger();
/// log::info!("Logger is ready");
/// ```
pub fn init_logger() {
    INIT_LOGGER.call_once(|| {
        env_logger::init();
    });
}

// Import the top-level `cloudllm` module.
pub mod cloudllm;

// Re-exporting key items for easier external access.
pub use cloudllm::client_wrapper;
pub use cloudllm::client_wrapper::{
    ClientWrapper, Message, MessageChunk, MessageChunkStream, MessageStreamFuture, Role,
};
pub use cloudllm::clients;
pub use cloudllm::llm_session::LLMSession;

// Re-export tool protocol and council functionality
pub use cloudllm::council;
pub use cloudllm::tool_adapters;
pub use cloudllm::tool_protocol;
pub use cloudllm::tools;
