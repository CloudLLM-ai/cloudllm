//! Built-in Tool Implementations
//!
//! This module provides production-ready tools that agents can use to enhance their capabilities.
//! These tools can be used individually or composed together via the tool protocol system.
//!
//! # Available Tools
//!
//! - **Memory**: Persistent, TTL-aware key-value store for maintaining agent state across sessions
//!   - Succinct command protocol (P/G/L/D/C/T/SPEC)
//!   - Automatic background expiration
//!   - Thread-safe with full async support
//!   - Can be wrapped with MemoryProtocol for use in agents
//!
//! - **Bash**: Secure command execution on Linux and macOS
//!   - Cross-platform with configurable timeout
//!   - Security features: command allow/deny lists, working directory restrictions
//!   - Separate stdout/stderr capture with size limits
//!   - Full async/await support via tokio
//!
//! # Integration with Agents
//!
//! These tools can be exposed to agents through the tool protocol system:
//!
//! ```ignore
//! use cloudllm::tools::Memory;
//! use cloudllm::tool_protocols::MemoryProtocol;
//! use cloudllm::tool_protocol::ToolRegistry;
//! use std::sync::Arc;
//!
//! let memory = Arc::new(Memory::new());
//! let protocol = Arc::new(MemoryProtocol::new(memory));
//! let registry = Arc::new(ToolRegistry::new(protocol));
//! agent.with_tools(registry);
//! ```

pub mod bash;
pub mod memory;

pub use bash::{BashError, BashResult, BashTool, Platform};
pub use memory::{Memory, MemoryMetadata};
