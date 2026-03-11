//! Compatibility re-exports for the shared MCP HTTP server layer.

#[cfg(feature = "mcp-server")]
pub use mcp::http::AxumHttpAdapter;
pub use mcp::http::{HttpServerAdapter, HttpServerConfig, HttpServerInstance};
