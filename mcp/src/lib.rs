//! Reusable MCP runtime primitives for tool discovery, execution, and HTTP serving.
//!
//! This crate contains the protocol-facing pieces that can be shared by multiple
//! higher-level projects. It intentionally avoids any dependency on `cloudllm`
//! so crates such as `thoughtchain` and `cloudllm` can both build on the same
//! MCP foundation without introducing circular dependencies.
#![warn(missing_docs)]

pub mod builder;
pub mod builder_utils;
pub mod client;
pub mod events;
pub mod http;
pub mod protocol;
pub mod resources;
pub mod server;
#[cfg(feature = "server")]
pub mod streamable_http;

pub use builder::MCPServerBuilder;
pub use builder_utils::{AuthConfig, IpFilter};
pub use client::McpClientProtocol;
pub use events::{McpEvent, McpEventHandler};
pub use http::{HttpServerAdapter, HttpServerConfig, HttpServerInstance};
pub use protocol::{
    Tool, ToolDefinition, ToolError, ToolMetadata, ToolParameter, ToolParameterType, ToolProtocol,
    ToolRegistry, ToolResult,
};
pub use resources::{ResourceError, ResourceMetadata, ResourceProtocol};
pub use server::UnifiedMcpServer;
#[cfg(feature = "server")]
pub use streamable_http::{
    streamable_http_router, StreamableHttpConfig, CURRENT_MCP_PROTOCOL_VERSION,
    SUPPORTED_MCP_PROTOCOL_VERSIONS,
};
