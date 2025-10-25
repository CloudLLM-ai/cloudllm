//! Tool implementations for CloudLLM agents
//!
//! This module provides built-in tools that agents can use to enhance their capabilities.
//! Currently includes:
//! - Memory: A persistent, TTL-aware memory system for agents to store and retrieve information
//! - Bash: Secure command execution on Linux and macOS with safety controls

pub mod bash;
pub mod memory;

pub use bash::{BashError, BashResult, BashTool, Platform};
pub use memory::{Memory, MemoryMetadata};
