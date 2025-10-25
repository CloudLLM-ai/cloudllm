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
//! ```rust,no_run
//! use cloudllm::mcp_server::UnifiedMcpServer;
//! use cloudllm::tools::Memory;
//! use cloudllm::tool_protocols::MemoryProtocol;
//! use cloudllm::tool_protocol::ToolProtocol;
//! use std::sync::Arc;
//!
//! # async {
//! let memory = Arc::new(Memory::new());
//! let memory_protocol = Arc::new(MemoryProtocol::new(memory));
//!
//! let mut server = UnifiedMcpServer::new();
//! server.register_tool("memory", memory_protocol);
//!
//! // Now the server implements ToolProtocol and can route calls
//! let tools = server.list_tools().await.unwrap();
//! # };
//! ```

use crate::cloudllm::tool_protocol::{ToolError, ToolMetadata, ToolProtocol, ToolResult};
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
    /// ```rust,no_run
    /// use cloudllm::mcp_server::UnifiedMcpServer;
    /// use cloudllm::tools::Memory;
    /// use cloudllm::tool_protocols::MemoryProtocol;
    /// use std::sync::Arc;
    ///
    /// # #[tokio::main]
    /// # async fn main() {
    /// let memory = Arc::new(Memory::new());
    /// let memory_protocol = Arc::new(MemoryProtocol::new(memory));
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
    async fn list_tools(&self) -> Result<Vec<ToolMetadata>, Box<dyn Error + Send + Sync>> {
        let tools = self.tools.read().await;
        let protocols: Vec<Arc<dyn ToolProtocol>> = tools.values().cloned().collect();

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cloudllm::tool_protocol::ToolMetadata;

    /// Mock tool protocol for testing
    struct MockToolProtocol {
        name: String,
    }

    #[async_trait]
    impl ToolProtocol for MockToolProtocol {
        async fn execute(
            &self,
            tool_name: &str,
            _parameters: serde_json::Value,
        ) -> Result<ToolResult, Box<dyn Error + Send + Sync>> {
            Ok(ToolResult::success(serde_json::json!({
                "tool": tool_name,
                "source": &self.name
            })))
        }

        async fn list_tools(&self) -> Result<Vec<ToolMetadata>, Box<dyn Error + Send + Sync>> {
            Ok(vec![ToolMetadata::new(&self.name, "A mock tool")])
        }

        async fn get_tool_metadata(
            &self,
            tool_name: &str,
        ) -> Result<ToolMetadata, Box<dyn Error + Send + Sync>> {
            if tool_name == self.name {
                Ok(ToolMetadata::new(&self.name, "A mock tool"))
            } else {
                Err(Box::new(ToolError::NotFound(tool_name.to_string())))
            }
        }

        fn protocol_name(&self) -> &str {
            "mock"
        }
    }

    #[tokio::test]
    async fn test_unified_server_creation() {
        let server = UnifiedMcpServer::new();
        assert_eq!(server.tool_count().await, 0);
        assert_eq!(server.protocol_name(), "unified-mcp-server");
    }

    #[tokio::test]
    async fn test_register_single_tool() {
        let mut server = UnifiedMcpServer::new();
        let mock = Arc::new(MockToolProtocol {
            name: "test_tool".to_string(),
        });

        server.register_tool("test_tool", mock).await;
        assert_eq!(server.tool_count().await, 1);
        assert!(server.has_tool("test_tool").await);
    }

    #[tokio::test]
    async fn test_register_multiple_tools() {
        let mut server = UnifiedMcpServer::new();
        let mock1 = Arc::new(MockToolProtocol {
            name: "tool1".to_string(),
        });
        let mock2 = Arc::new(MockToolProtocol {
            name: "tool2".to_string(),
        });

        server.register_tool("tool1", mock1).await;
        server.register_tool("tool2", mock2).await;
        assert_eq!(server.tool_count().await, 2);
        assert!(server.has_tool("tool1").await);
        assert!(server.has_tool("tool2").await);
    }

    #[tokio::test]
    async fn test_execute_tool_routing() {
        let mut server = UnifiedMcpServer::new();
        let mock = Arc::new(MockToolProtocol {
            name: "router_test".to_string(),
        });

        server.register_tool("router_test", mock).await;

        let result = server.execute("router_test", serde_json::json!({})).await;

        assert!(result.is_ok());
        let tool_result = result.unwrap();
        assert!(tool_result.success);
        assert_eq!(tool_result.output["tool"], "router_test");
    }

    #[tokio::test]
    async fn test_execute_nonexistent_tool() {
        let server = UnifiedMcpServer::new();

        let result = server.execute("nonexistent", serde_json::json!({})).await;

        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("not found") || err.contains("NotFound"));
    }

    #[tokio::test]
    async fn test_list_tools_aggregation() {
        let mut server = UnifiedMcpServer::new();
        let mock1 = Arc::new(MockToolProtocol {
            name: "tool1".to_string(),
        });
        let mock2 = Arc::new(MockToolProtocol {
            name: "tool2".to_string(),
        });

        server.register_tool("tool1", mock1).await;
        server.register_tool("tool2", mock2).await;

        let tools = server.list_tools().await.unwrap();
        assert_eq!(tools.len(), 2);
        assert!(tools.iter().any(|t| t.name == "tool1"));
        assert!(tools.iter().any(|t| t.name == "tool2"));
    }

    #[tokio::test]
    async fn test_get_tool_metadata() {
        let mut server = UnifiedMcpServer::new();
        let mock = Arc::new(MockToolProtocol {
            name: "metadata_test".to_string(),
        });

        server.register_tool("metadata_test", mock).await;

        let metadata = server.get_tool_metadata("metadata_test").await;
        assert!(metadata.is_ok());
        assert_eq!(metadata.unwrap().name, "metadata_test");
    }

    #[tokio::test]
    async fn test_unregister_tool() {
        let mut server = UnifiedMcpServer::new();
        let mock = Arc::new(MockToolProtocol {
            name: "temp_tool".to_string(),
        });

        server.register_tool("temp_tool", mock).await;
        assert_eq!(server.tool_count().await, 1);

        server.unregister_tool("temp_tool").await;
        assert_eq!(server.tool_count().await, 0);
        assert!(!server.has_tool("temp_tool").await);
    }

    #[tokio::test]
    async fn test_default_constructor() {
        let server = UnifiedMcpServer::default();
        assert_eq!(server.tool_count().await, 0);
    }
}
