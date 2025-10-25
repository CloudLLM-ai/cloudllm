//! Internal module tree housing the building blocks exposed via `cloudllm`.
//!
//! This module organizes CloudLLM's core functionality:
//!
//! - **client_wrapper**: Trait definition for LLM provider implementations
//! - **clients**: Concrete implementations for OpenAI, Claude, Gemini, Grok, and custom endpoints
//! - **llm_session**: Stateful conversation management with context trimming
//! - **tool_protocol**: Protocol-agnostic tool interface and ToolRegistry for multi-protocol support
//! - **tool_protocols**: Concrete ToolProtocol implementations (Custom, MCP, Memory, OpenAI)
//! - **tools**: Built-in tools (Memory, Bash, etc.)
//! - **council**: Multi-agent orchestration system with 5 collaboration modes
//! - **mcp_server**: Unified MCP server for tool aggregation and routing

pub mod client_wrapper;
pub mod clients;
pub mod council;
pub mod llm_session;
pub mod mcp_server;
pub mod tool_protocol;
pub mod tool_protocols;
pub mod tools;

// Backwards compatibility: Re-export old name
#[deprecated(
    since = "0.5.0",
    note = "Use `tool_protocols` instead, these are ToolProtocol implementations"
)]
pub mod tool_adapters {
    pub use super::tool_protocols::*;
}

// Let's explicitly export LLMSession so we don't have to access it via cloudllm::llm_session::LLMSession
// and instead as cloudllm::LLMSession
pub use llm_session::LLMSession;
