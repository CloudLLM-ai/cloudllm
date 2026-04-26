//! Integration tests for UnifiedMcpServer.

use async_trait::async_trait;
use mcp::protocol::{ToolError, ToolMetadata, ToolProtocol, ToolResult};
use mcp::UnifiedMcpServer;
use std::error::Error;
use std::sync::Arc;

/// Mock tool protocol for testing server routing.
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
