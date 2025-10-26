//! Built-in Tool Implementations
//!
//! This module provides production-ready tools that agents can use to enhance their capabilities.
//! These tools can be used individually or composed together via the tool protocol system.
//!
//! # Available Tools
//!
//! - **Calculator**: Fast, reliable scientific calculator with full mathematical operations
//!   - Comprehensive arithmetic, trigonometric, and logarithmic functions
//!   - Statistical operations on arrays (mean, median, mode, std, variance, etc.)
//!   - Support for all standard mathematical constants (pi, e)
//!   - Stateless and thread-safe for high-performance concurrent use
//!   - Can be wrapped with CalculatorProtocol for use in agents
//!
//! - **HTTP Client**: Secure REST API client for calling external services
//!   - All HTTP methods: GET, POST, PUT, DELETE, PATCH, HEAD, OPTIONS
//!   - JSON payloads and custom headers support
//!   - Domain allowlist/blocklist for security
//!   - Basic authentication and bearer token support
//!   - Configurable timeout and response size limits
//!   - Thread-safe with connection pooling
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
//! - **File System**: Safe file and directory operations with path restrictions
//!   - Read, write, append, delete files
//!   - List and manage directories recursively
//!   - Path traversal protection (`../../../etc/passwd` is blocked)
//!   - Optional file extension filtering
//!   - Root path restriction for sandboxing
//!   - File metadata access and search functionality
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
pub mod calculator;
pub mod filesystem;
pub mod http_client;
pub mod memory;

pub use bash::{BashError, BashResult, BashTool, Platform};
pub use calculator::{Calculator, CalculatorError, CalculatorResult};
pub use filesystem::{DirectoryEntry, FileMetadata, FileSystemError, FileSystemTool};
pub use http_client::{HttpClient, HttpClientError, HttpResponse};
pub use memory::{Memory, MemoryMetadata};
