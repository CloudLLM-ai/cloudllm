//! Compatibility re-exports for the shared `mcp` protocol layer.
//!
//! CloudLLM continues to expose its tool protocol surface from this module, but
//! the underlying protocol primitives now live in the standalone `mcp` crate.

pub use mcp::protocol::{
    Tool, ToolDefinition, ToolError, ToolMetadata, ToolParameter, ToolParameterType, ToolProtocol,
    ToolRegistry, ToolResult,
};
