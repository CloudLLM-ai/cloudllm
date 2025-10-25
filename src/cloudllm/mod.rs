//! Internal module tree housing the building blocks exposed via `cloudllm`.

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
