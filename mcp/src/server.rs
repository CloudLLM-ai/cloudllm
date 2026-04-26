//! Unified MCP Server
//!
//! This module provides a concrete MCP server implementation that aggregates
//! multiple tools and implements the ToolProtocol trait, routing tool calls
//! to the appropriate underlying tool implementation.
//!
//! The server acts as a dispatcher that can be deployed as an HTTP service,
//! allowing multiple agents (local or remote) to access a unified set of tools
//! through a single ToolProtocol interface.
//!
//! # Architecture
//!
//! ```text
//! Multiple Tools (Memory, Bash, etc.)
//!         ↓
//! UnifiedMcpServer (implements ToolProtocol)
//!         ↓
//! HTTP Endpoints (GET /tools, POST /execute)
//!         ↓
//! Agents/Clients (via McpClientProtocol)
//! ```
//!
//! # Example
//!
//! ```ignore
//! use async_trait::async_trait;
//! use mcp::{ToolMetadata, ToolProtocol, ToolResult};
//! use mcp::UnifiedMcpServer;
//! use std::sync::Arc;
//!
//! struct MemoryProtocol;
//!
//! #[async_trait]
//! impl ToolProtocol for MemoryProtocol {
//!     async fn execute(
//!         &self,
//!         _tool_name: &str,
//!         _parameters: serde_json::Value,
//!     ) -> Result<ToolResult, Box<dyn std::error::Error + Send + Sync>> {
//!         Ok(ToolResult::success(serde_json::json!({"ok": true})))
//!     }
//!
//!     async fn list_tools(
//!         &self,
//!     ) -> Result<Vec<ToolMetadata>, Box<dyn std::error::Error + Send + Sync>> {
//!         Ok(vec![])
//!     }
//! }
//!
//! # async {
//! let memory_protocol = Arc::new(MemoryProtocol);
//!
//! let mut server = UnifiedMcpServer::new();
//! server.register_tool("memory", memory_protocol).await;
//!
//! // Now the server implements ToolProtocol and can route calls
//! let tools = server.list_tools().await.unwrap();
//! # };
//! ```

use crate::protocol::{ToolError, ToolMetadata, ToolProtocol, ToolResult};
use async_trait::async_trait;
use std::collections::HashMap;
use std::error::Error;
use std::sync::Arc;
use tokio::sync::RwLock;

/// A unified MCP server that aggregates multiple tools
///
/// The UnifiedMcpServer implements the ToolProtocol trait and routes
/// tool execution requests to the appropriate underlying tool protocol
/// implementation based on the tool name.
///
/// This allows a single server instance to expose multiple tools with
/// different implementations, making it suitable for deployment as an
/// MCP HTTP service that can be accessed by multiple agents.
///
/// # Thread Safety
///
/// The server is thread-safe and can be shared across multiple concurrent
/// tool executions using `Arc<UnifiedMcpServer>`.
#[derive(Clone)]
pub struct UnifiedMcpServer {
    /// Map of tool name to its ToolProtocol implementation
    tools: Arc<RwLock<HashMap<String, Arc<dyn ToolProtocol>>>>,
}

impl UnifiedMcpServer {
    /// Create a new empty unified MCP server
    pub fn new() -> Self {
        Self {
            tools: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Register a tool with the server
    ///
    /// # Arguments
    ///
    /// * `tool_name` - The identifier for the tool (e.g., "memory", "bash")
    /// * `protocol` - The ToolProtocol implementation for this tool
    ///
    /// # Example
    ///
    /// ```ignore
    /// use async_trait::async_trait;
    /// use mcp::{ToolMetadata, ToolProtocol, ToolResult, UnifiedMcpServer};
    /// use std::sync::Arc;
    ///
    /// struct MemoryProtocol;
    ///
    /// #[async_trait]
    /// impl ToolProtocol for MemoryProtocol {
    ///     async fn execute(
    ///         &self,
    ///         _tool_name: &str,
    ///         _parameters: serde_json::Value,
    ///     ) -> Result<ToolResult, Box<dyn std::error::Error + Send + Sync>> {
    ///         Ok(ToolResult::success(serde_json::json!({"ok": true})))
    ///     }
    ///
    ///     async fn list_tools(
    ///         &self,
    ///     ) -> Result<Vec<ToolMetadata>, Box<dyn std::error::Error + Send + Sync>> {
    ///         Ok(vec![])
    ///     }
    /// }
    ///
    /// # #[tokio::main]
    /// # async fn main() {
    /// let memory_protocol = Arc::new(MemoryProtocol);
    ///
    /// let mut server = UnifiedMcpServer::new();
    /// server.register_tool("memory", memory_protocol).await;
    /// # }
    /// ```
    pub async fn register_tool(&mut self, tool_name: &str, protocol: Arc<dyn ToolProtocol>) {
        let mut tools = self.tools.write().await;
        tools.insert(tool_name.to_string(), protocol);
    }

    /// Unregister a tool from the server
    pub async fn unregister_tool(&mut self, tool_name: &str) {
        let mut tools = self.tools.write().await;
        tools.remove(tool_name);
    }

    /// Check if a tool is registered
    pub async fn has_tool(&self, tool_name: &str) -> bool {
        let tools = self.tools.read().await;
        tools.contains_key(tool_name)
    }

    /// Get the number of registered tools
    pub async fn tool_count(&self) -> usize {
        let tools = self.tools.read().await;
        tools.len()
    }
}

impl Default for UnifiedMcpServer {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl ToolProtocol for UnifiedMcpServer {
    /// Execute a tool by routing to the appropriate protocol
    ///
    /// # Routing Logic
    ///
    /// 1. Look up the tool name in the registry
    /// 2. If found, delegate to that tool's protocol
    /// 3. If not found, return NotFound error
    async fn execute(
        &self,
        tool_name: &str,
        parameters: serde_json::Value,
    ) -> Result<ToolResult, Box<dyn Error + Send + Sync>> {
        let tools = self.tools.read().await;

        let protocol = tools.get(tool_name).cloned().ok_or_else(|| {
            Box::new(ToolError::NotFound(tool_name.to_string())) as Box<dyn Error + Send + Sync>
        })?;

        // Drop the read lock before executing to allow concurrent access
        drop(tools);

        // Route to the appropriate tool's protocol
        protocol.execute(tool_name, parameters).await
    }

    /// List all available tools across all registered protocols
    ///
    /// This aggregates tool metadata from all registered tool protocols.
    /// Each protocol is queried at most once even if multiple tool names
    /// are registered to the same protocol instance.
    async fn list_tools(&self) -> Result<Vec<ToolMetadata>, Box<dyn Error + Send + Sync>> {
        let tools = self.tools.read().await;

        // Deduplicate protocol instances by pointer so each protocol's list_tools()
        // is called at most once (multiple tool names may point to the same protocol).
        let mut seen: std::collections::HashSet<usize> = std::collections::HashSet::new();
        let protocols: Vec<Arc<dyn ToolProtocol>> = tools
            .values()
            .filter(|p| seen.insert(Arc::as_ptr(*p) as *const () as usize))
            .cloned()
            .collect();

        // Drop the read lock before making async calls
        drop(tools);

        let mut all_tools = Vec::new();

        for protocol in protocols {
            match protocol.list_tools().await {
                Ok(mut tool_list) => all_tools.append(&mut tool_list),
                Err(e) => {
                    // Log but continue - we want to return what we can
                    eprintln!("Error listing tools from protocol: {}", e);
                }
            }
        }

        Ok(all_tools)
    }

    /// Get metadata for a specific tool
    ///
    /// This searches across all registered protocols to find the tool.
    async fn get_tool_metadata(
        &self,
        tool_name: &str,
    ) -> Result<ToolMetadata, Box<dyn Error + Send + Sync>> {
        let all_tools = self.list_tools().await?;
        all_tools
            .into_iter()
            .find(|t| t.name == tool_name)
            .ok_or_else(|| {
                Box::new(ToolError::NotFound(tool_name.to_string())) as Box<dyn Error + Send + Sync>
            })
    }

    /// Protocol identifier
    fn protocol_name(&self) -> &str {
        "unified-mcp-server"
    }

    /// Initialize the server (initializes all registered protocols)
    async fn initialize(&mut self) -> Result<(), Box<dyn Error + Send + Sync>> {
        let _tools = self.tools.read().await;

        // Note: We can't call initialize on Arc<dyn ToolProtocol> since
        // it takes &mut self. This is a limitation of the current design.
        // Future: Consider a separate initialization registry or use Arc<Mutex<>>.

        Ok(())
    }

    /// Shutdown the server (shuts down all registered protocols)
    async fn shutdown(&mut self) -> Result<(), Box<dyn Error + Send + Sync>> {
        let _tools = self.tools.read().await;

        // Same limitation as initialize - we need Arc<Mutex<>> for protocols
        // that need shutdown handling.

        Ok(())
    }
}
