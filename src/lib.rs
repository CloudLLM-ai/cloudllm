//! # CloudLLM
//!
//! CloudLLM is a batteries-included Rust toolkit for orchestrating intelligent agents that
//! converse with remote Large Language Models and execute structured actions through tools.
//!
//! The crate provides carefully layered abstractions for:
//!
//! * **Agents with Tools**: [`Agent`] abstractions that connect to LLMs and execute actions
//!   through a flexible, multi-protocol tool system via [`tool_protocol::ToolRegistry`]
//! * **Tool Routing**: Local Rust functions, remote MCP servers, Memory persistence, or custom
//!   protocols all accessible through a unified interface
//! * **Server Deployment**: `MCPServerBuilder` (available on `mcp-server` feature) for easily deploying tool servers
//!   with HTTP support, authentication, and IP filtering
//! * **Stateful Conversations**: [`LLMSession`] for maintaining rolling conversation history
//!   with context trimming and token accounting
//! * **Multi-Agent Orchestration**: [`orchestration`] module for coordinating discussions across
//!   multiple agents with Parallel, RoundRobin, Moderated, Hierarchical, Debate, Ralph, or
//!   AnthropicAgentTeams patterns
//! * **Provider Flexibility**: [`ClientWrapper`] trait implemented for OpenAI, Anthropic Claude,
//!   Google Gemini, xAI Grok, and custom OpenAI-compatible endpoints
//!
//! The crate aims to provide documentation-quality examples for every public API.  These
//! examples are kept up to date and are written to compile under `cargo test --doc`.
//!
//! ## Core Concepts
//!
//! ### LLMSession: Stateful Conversations (The Foundation)
//!
//! [`LLMSession`] is the foundational abstraction—it wraps a client to maintain a rolling
//! conversation history with automatic context trimming and token accounting. Perfect for
//! simple use cases where you need persistent conversation state:
//!
//! ```rust,no_run
//! use std::sync::Arc;
//! use cloudllm::{LLMSession, Role};
//! use cloudllm::clients::openai::{OpenAIClient, Model};
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let client = Arc::new(OpenAIClient::new_with_model_enum(
//!         &std::env::var("OPEN_AI_SECRET")?,
//!         Model::GPT41Mini
//!     ));
//!
//!     let mut session = LLMSession::new(client, "You are helpful.".into(), 8_192);
//!
//!     let reply = session
//!         .send_message(Role::User, "Hello, how are you?".into(), None)
//!         .await?;
//!
//!     println!("Assistant: {}", reply.content);
//!     println!("Tokens used: {:?}", session.token_usage());
//!     Ok(())
//! }
//! ```
//!
//! ### Agents: The Heart of CloudLLM
//!
//! [`Agent`] extends [`LLMSession`] by adding identity, expertise, and optional tools. Agents are
//! the primary way to build sophisticated LLM interactions—they maintain personality and can
//! execute actions through tools:
//!
//! ```rust,no_run
//! use std::sync::Arc;
//! use cloudllm::Agent;
//! use cloudllm::clients::openai::{OpenAIClient, Model};
//! use cloudllm::tool_protocol::ToolRegistry;
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let client = Arc::new(OpenAIClient::new_with_model_enum(
//!         &std::env::var("OPEN_AI_SECRET")?,
//!         Model::GPT41Mini
//!     ));
//!
//!     let agent = Agent::new("assistant", "My AI Assistant", client)
//!         .with_expertise("Problem solving")
//!         .with_personality("Friendly and analytical");
//!
//!     // Agent is now ready to execute actions with tools!
//!     Ok(())
//! }
//! ```
//!
//! ### Tool Registry: Multi-Protocol Tool Access
//!
//! Agents access tools through the [`tool_protocol::ToolRegistry`], which supports **multiple
//! simultaneous protocols**. Register tools from different sources—local Rust functions, remote
//! MCP servers, persistent Memory, custom implementations—and agents access them transparently:
//!
//! - **Local Tools**: Rust closures and async functions via [`tool_protocols::CustomToolProtocol`]
//! - **Remote Tools**: HTTP-based MCP servers via [`tool_protocols::McpClientProtocol`]
//! - **Persistent Memory**: Key-value storage with TTL via [`tool_protocols::MemoryProtocol`]
//! - **Custom Protocols**: Implement [`tool_protocol::ToolProtocol`] for any system
//!
//! ```rust,no_run
//! use std::sync::Arc;
//! use cloudllm::tool_protocol::ToolRegistry;
//! use cloudllm::tool_protocols::CustomToolProtocol;
//!
//! # #[tokio::main]
//! # async fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let mut registry = ToolRegistry::empty();
//!
//! // Add local tools
//! let local = Arc::new(CustomToolProtocol::new());
//! let _ = registry.add_protocol("local", local).await;
//!
//! // Add remote MCP servers
//! use cloudllm::tool_protocols::McpClientProtocol;
//! let mcp_server = Arc::new(McpClientProtocol::new("http://localhost:8080".to_string()));
//! let _ = registry.add_protocol("remote", mcp_server).await;
//!
//! // Agent uses tools from both sources transparently!
//! # Ok(())
//! # }
//! ```
//!
//! ### Provider Abstraction
//!
//! Each cloud provider (OpenAI, Anthropic/Claude, Google Gemini, xAI Grok, and custom OpenAI-
//! compatible endpoints) is exposed as a `ClientWrapper` implementation.  All wrappers share
//! the same ergonomics for synchronous calls, streaming, and token accounting.
//!
//! ### Multi-Agent Orchestration
//!
//! The [`orchestration`] module orchestrates conversations between multiple agents across a variety
//! of collaboration patterns:
//! - **Parallel**: All agents respond simultaneously with aggregated results
//! - **RoundRobin**: Agents take sequential turns building on previous responses
//! - **Moderated**: Agents propose ideas, moderator synthesizes the answer
//! - **Hierarchical**: Lead agent coordinates, specialists handle specific aspects
//! - **Debate**: Agents discuss and challenge until convergence
//! - **Ralph**: Autonomous iterative loop working through a PRD task list
//! - **AnthropicAgentTeams**: Decentralized task-based coordination where agents autonomously
//!   discover, claim, and complete tasks from a shared pool with no central orchestrator
//!
//! ### Deploying Tool Servers with MCPServerBuilder
//!
//! Create standalone MCP servers that expose tools over HTTP with a simple builder API.
//! Perfect for microservices or sharing tool capabilities across the network:
//!
//! For a complete MCP server example with HTTP support, see the `examples/mcp_server_all_tools.rs`
//! example which demonstrates deploying all built-in tools via HTTP with authentication and IP filtering.
//!
//! MCPServerBuilder is available on the `mcp-server` feature (requires `axum` and `tower`).
//!
//! ### Creating Tools: Simple to Advanced
//!
//! Tools are the actions agents can take. CloudLLM supports multiple ways to create them:
//!
//! **Simple Approach: Rust Closures**
//!
//! Register any Rust function or async closure as a tool:
//!
//! ```rust,no_run
//! use cloudllm::tool_protocols::CustomToolProtocol;
//! use cloudllm::tool_protocol::{ToolMetadata, ToolResult};
//! use std::sync::Arc;
//!
//! # #[tokio::main]
//! # async fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let protocol = Arc::new(CustomToolProtocol::new());
//!
//! // Synchronous tool
//! protocol.register_tool(
//!     ToolMetadata::new("add", "Add two numbers"),
//!     Arc::new(|params| {
//!         let a = params["a"].as_f64().unwrap_or(0.0);
//!         let b = params["b"].as_f64().unwrap_or(0.0);
//!         Ok(ToolResult::success(serde_json::json!({"result": a + b})))
//!     }),
//! ).await;
//!
//! // Asynchronous tool
//! protocol.register_async_tool(
//!     ToolMetadata::new("fetch", "Fetch data from a URL"),
//!     Arc::new(|params| {
//!         Box::pin(async move {
//!             let url = params["url"].as_str().unwrap_or("");
//!             Ok(ToolResult::success(serde_json::json!({"url": url})))
//!         })
//!     }),
//! ).await;
//! # Ok(())
//! # }
//! ```
//!
//! **Advanced Approach: Custom Protocol Implementation**
//!
//! For complex tools or integration with external systems, implement [`tool_protocol::ToolProtocol`]:
//!
//! ```rust,no_run
//! use async_trait::async_trait;
//! use cloudllm::tool_protocol::{ToolMetadata, ToolProtocol, ToolResult};
//! use std::error::Error;
//!
//! pub struct DatabaseAdapter;
//!
//! #[async_trait]
//! impl ToolProtocol for DatabaseAdapter {
//!     async fn execute(
//!         &self,
//!         tool_name: &str,
//!         parameters: serde_json::Value,
//!     ) -> Result<ToolResult, Box<dyn Error + Send + Sync>> {
//!         match tool_name {
//!             "query" => {
//!                 let sql = parameters["sql"].as_str().unwrap_or("");
//!                 // Execute actual database query
//!                 Ok(ToolResult::success(serde_json::json!({"result": "data"})))
//!             }
//!             _ => Ok(ToolResult::failure("Unknown tool".into()))
//!         }
//!     }
//!
//!     async fn list_tools(&self) -> Result<Vec<ToolMetadata>, Box<dyn Error + Send + Sync>> {
//!         Ok(vec![ToolMetadata::new("query", "Execute SQL query")])
//!     }
//!
//!     async fn get_tool_metadata(
//!         &self,
//!         tool_name: &str,
//!     ) -> Result<ToolMetadata, Box<dyn Error + Send + Sync>> {
//!         Ok(ToolMetadata::new(tool_name, "Tool"))
//!     }
//!
//!     fn protocol_name(&self) -> &str {
//!         "database"
//!     }
//! }
//! ```
//!
//! **Built-in Tools: Ready to Use**
//!
//! CloudLLM provides several production-ready tools:
//! - [`tools::Calculator`] - Mathematical expressions and statistics
//! - [`tools::Memory`] - Persistent TTL-aware key-value store
//! - [`tools::HttpClient`] - Secure REST API calls with domain filtering
//! - [`tools::BashTool`] - Safe command execution with timeouts
//! - [`tools::FileSystemTool`] - Sandboxed file operations
//!
//! See the [`tools`] module for complete documentation on each tool.
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
pub use cloudllm::agent::Agent;
pub use cloudllm::client_wrapper;
pub use cloudllm::client_wrapper::{
    ClientWrapper, Message, MessageChunk, MessageChunkStream, MessageStreamFuture, Role,
};
pub use cloudllm::clients;
pub use cloudllm::config::CloudLLMConfig;
pub use cloudllm::context_strategy;
pub use cloudllm::context_strategy::{
    ContextStrategy, NoveltyAwareStrategy, SelfCompressionStrategy, TrimStrategy,
};
pub use cloudllm::llm_session::LLMSession;
pub use cloudllm::thought_chain;
pub use cloudllm::thought_chain::{Thought, ThoughtChain, ThoughtType};

// Re-export tool protocol and orchestration functionality
pub use cloudllm::event;
pub use cloudllm::event::{AgentEvent, EventHandler, McpEvent, OrchestrationEvent, PlannerEvent};
pub use cloudllm::mcp_server;
pub use cloudllm::orchestration;
pub use cloudllm::planner;
pub use cloudllm::planner::{
    BasicPlanner, MemoryEntry, MemoryStore, NoopMemory, NoopPolicy, NoopStream, Planner,
    PlannerContext, PlannerOutcome, PlannerResult, PolicyDecision, PolicyEngine, StreamSink,
    ToolCallRequest, UserMessage,
};
pub use cloudllm::tool_protocol;
pub use cloudllm::tool_protocols;
pub use cloudllm::tools;
