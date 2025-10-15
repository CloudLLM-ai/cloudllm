//! Tool Protocol Abstraction Layer
//!
//! This module provides a flexible abstraction for connecting agents to various tool protocols.
//! It supports multiple standards including MCP (Model Context Protocol), custom function calling,
//! and allows users to implement their own tool communication mechanisms.
//!
//! # Architecture
//!
//! ```text
//! Agent → ToolRegistry → ToolProtocol (trait) → [MCP | Custom | User-defined]
//! ```
//!
//! # Example
//!
//! ```rust,no_run
//! use cloudllm::tool_protocol::{ToolParameter, ToolParameterType};
//! use serde_json::json;
//!
//! // Define a tool parameter
//! let param = ToolParameter::new("expression", ToolParameterType::String)
//!     .with_description("The mathematical expression to evaluate")
//!     .required();
//! ```

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::error::Error;
use std::fmt;
use std::sync::Arc;

/// Represents the result of a tool execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolResult {
    /// Whether the tool execution was successful
    pub success: bool,
    /// The output data from the tool
    pub output: serde_json::Value,
    /// Optional error message if execution failed
    pub error: Option<String>,
    /// Metadata about the execution (timing, cost, etc.)
    pub metadata: HashMap<String, serde_json::Value>,
}

impl ToolResult {
    /// Convenience constructor for successful tool execution.
    pub fn success(output: serde_json::Value) -> Self {
        Self {
            success: true,
            output,
            error: None,
            metadata: HashMap::new(),
        }
    }

    /// Convenience constructor for failed tool execution.
    pub fn failure(error: String) -> Self {
        Self {
            success: false,
            output: serde_json::Value::Null,
            error: Some(error),
            metadata: HashMap::new(),
        }
    }

    /// Attach protocol or application specific metadata to the result.
    pub fn with_metadata(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        self.metadata.insert(key.into(), value);
        self
    }
}

/// Defines the type of a tool parameter
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum ToolParameterType {
    String,
    Number,
    Integer,
    Boolean,
    Array,
    Object,
}

/// Defines a parameter for a tool
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolParameter {
    pub name: String,
    #[serde(rename = "type")]
    pub param_type: ToolParameterType,
    pub description: Option<String>,
    pub required: bool,
    pub default: Option<serde_json::Value>,
    /// For array types, specifies the type of items
    pub items: Option<Box<ToolParameterType>>,
    /// For object types, specifies nested properties
    pub properties: Option<HashMap<String, ToolParameter>>,
}

impl ToolParameter {
    /// Define a new tool parameter with the provided name and type.
    pub fn new(name: impl Into<String>, param_type: ToolParameterType) -> Self {
        Self {
            name: name.into(),
            param_type,
            description: None,
            required: false,
            default: None,
            items: None,
            properties: None,
        }
    }

    /// Add a human readable description that will surface in generated schemas.
    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }

    /// Mark the argument as required.
    pub fn required(mut self) -> Self {
        self.required = true;
        self
    }

    /// Provide a default value that will be used when the LLM omits the parameter.
    pub fn with_default(mut self, default: serde_json::Value) -> Self {
        self.default = Some(default);
        self
    }

    /// For array parameters, declare the type of the contained items.
    pub fn with_items(mut self, item_type: ToolParameterType) -> Self {
        self.items = Some(Box::new(item_type));
        self
    }

    /// For object parameters, describe the nested properties.
    pub fn with_properties(mut self, properties: HashMap<String, ToolParameter>) -> Self {
        self.properties = Some(properties);
        self
    }
}

/// Metadata about a tool
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolMetadata {
    pub name: String,
    pub description: String,
    pub parameters: Vec<ToolParameter>,
    /// Additional metadata specific to the protocol
    pub protocol_metadata: HashMap<String, serde_json::Value>,
}

impl ToolMetadata {
    /// Create metadata with the supplied identifier and description.
    pub fn new(name: impl Into<String>, description: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            parameters: Vec::new(),
            protocol_metadata: HashMap::new(),
        }
    }

    /// Append a parameter definition to the tool metadata.
    pub fn with_parameter(mut self, param: ToolParameter) -> Self {
        self.parameters.push(param);
        self
    }

    /// Add protocol specific metadata (e.g. MCP capability flags).
    pub fn with_protocol_metadata(
        mut self,
        key: impl Into<String>,
        value: serde_json::Value,
    ) -> Self {
        self.protocol_metadata.insert(key.into(), value);
        self
    }
}

/// Trait for implementing tool execution protocols
#[async_trait]
pub trait ToolProtocol: Send + Sync {
    /// Execute a tool with the given parameters
    async fn execute(
        &self,
        tool_name: &str,
        parameters: serde_json::Value,
    ) -> Result<ToolResult, Box<dyn Error + Send + Sync>>;

    /// Get metadata about available tools
    async fn list_tools(&self) -> Result<Vec<ToolMetadata>, Box<dyn Error + Send + Sync>>;

    /// Get metadata about a specific tool
    async fn get_tool_metadata(
        &self,
        tool_name: &str,
    ) -> Result<ToolMetadata, Box<dyn Error + Send + Sync>>;

    /// Protocol identifier (e.g., "mcp", "custom", "openai-functions")
    fn protocol_name(&self) -> &str;

    /// Initialize/connect to the tool protocol
    async fn initialize(&mut self) -> Result<(), Box<dyn Error + Send + Sync>> {
        Ok(())
    }

    /// Cleanup/disconnect from the tool protocol
    async fn shutdown(&mut self) -> Result<(), Box<dyn Error + Send + Sync>> {
        Ok(())
    }
}

/// Error types for tool operations
#[derive(Debug, Clone)]
pub enum ToolError {
    /// Requested tool is not registered in the current registry/protocol.
    NotFound(String),
    /// Tool execution completed with an application level failure.
    ExecutionFailed(String),
    /// The provided JSON parameters failed validation or deserialization.
    InvalidParameters(String),
    /// A lower level protocol/transport error occurred.
    ProtocolError(String),
}

impl fmt::Display for ToolError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ToolError::NotFound(name) => write!(f, "Tool not found: {}", name),
            ToolError::ExecutionFailed(msg) => write!(f, "Tool execution failed: {}", msg),
            ToolError::InvalidParameters(msg) => write!(f, "Invalid parameters: {}", msg),
            ToolError::ProtocolError(msg) => write!(f, "Protocol error: {}", msg),
        }
    }
}

impl Error for ToolError {}

/// A tool that can be used by agents
pub struct Tool {
    /// Metadata describing the tool interface.
    metadata: ToolMetadata,
    /// Underlying protocol implementation that actually executes the tool.
    protocol: Arc<dyn ToolProtocol>,
}

impl Tool {
    /// Create a new tool bound to the supplied protocol implementation.
    pub fn new(
        name: impl Into<String>,
        description: impl Into<String>,
        protocol: Arc<dyn ToolProtocol>,
    ) -> Self {
        Self {
            metadata: ToolMetadata::new(name, description),
            protocol,
        }
    }

    /// Add a parameter definition to the tool builder.
    pub fn with_parameter(mut self, param: ToolParameter) -> Self {
        self.metadata.parameters.push(param);
        self
    }

    /// Attach protocol specific metadata to the tool builder.
    pub fn with_protocol_metadata(
        mut self,
        key: impl Into<String>,
        value: serde_json::Value,
    ) -> Self {
        self.metadata.protocol_metadata.insert(key.into(), value);
        self
    }

    /// Borrow the static metadata for the tool.
    pub fn metadata(&self) -> &ToolMetadata {
        &self.metadata
    }

    /// Execute the tool using the configured protocol.
    pub async fn execute(
        &self,
        parameters: serde_json::Value,
    ) -> Result<ToolResult, Box<dyn Error + Send + Sync>> {
        self.protocol.execute(&self.metadata.name, parameters).await
    }
}

/// Registry for managing tools available to agents
pub struct ToolRegistry {
    tools: HashMap<String, Tool>,
    protocol: Arc<dyn ToolProtocol>,
}

impl ToolRegistry {
    /// Build a registry powered by the provided protocol implementation.
    pub fn new(protocol: Arc<dyn ToolProtocol>) -> Self {
        Self {
            tools: HashMap::new(),
            protocol,
        }
    }

    /// Insert or replace a tool definition.
    pub fn add_tool(&mut self, tool: Tool) {
        self.tools.insert(tool.metadata.name.clone(), tool);
    }

    /// Remove a tool by name returning the owned entry if present.
    pub fn remove_tool(&mut self, name: &str) -> Option<Tool> {
        self.tools.remove(name)
    }

    /// Borrow a tool by name.
    pub fn get_tool(&self, name: &str) -> Option<&Tool> {
        self.tools.get(name)
    }

    /// List metadata for registered tools (iteration order follows the underlying map).
    pub fn list_tools(&self) -> Vec<&ToolMetadata> {
        self.tools.values().map(|t| &t.metadata).collect()
    }

    /// Execute a named tool with serialized parameters.
    pub async fn execute_tool(
        &self,
        tool_name: &str,
        parameters: serde_json::Value,
    ) -> Result<ToolResult, Box<dyn Error + Send + Sync>> {
        let tool = self
            .tools
            .get(tool_name)
            .ok_or_else(|| ToolError::NotFound(tool_name.to_string()))?;

        tool.execute(parameters).await
    }

    /// Borrow the registry level protocol implementation.
    pub fn protocol(&self) -> &Arc<dyn ToolProtocol> {
        &self.protocol
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct MockProtocol;

    #[async_trait]
    impl ToolProtocol for MockProtocol {
        async fn execute(
            &self,
            tool_name: &str,
            _parameters: serde_json::Value,
        ) -> Result<ToolResult, Box<dyn Error + Send + Sync>> {
            Ok(ToolResult::success(serde_json::json!({
                "tool": tool_name,
                "result": "mock_result"
            })))
        }

        async fn list_tools(&self) -> Result<Vec<ToolMetadata>, Box<dyn Error + Send + Sync>> {
            Ok(vec![])
        }

        async fn get_tool_metadata(
            &self,
            _tool_name: &str,
        ) -> Result<ToolMetadata, Box<dyn Error + Send + Sync>> {
            Ok(ToolMetadata::new("mock_tool", "A mock tool"))
        }

        fn protocol_name(&self) -> &str {
            "mock"
        }
    }

    #[test]
    fn test_tool_parameter_builder() {
        let param = ToolParameter::new("test_param", ToolParameterType::String)
            .with_description("A test parameter")
            .required()
            .with_default(serde_json::json!("default_value"));

        assert_eq!(param.name, "test_param");
        assert_eq!(param.param_type, ToolParameterType::String);
        assert_eq!(param.description, Some("A test parameter".to_string()));
        assert!(param.required);
        assert_eq!(param.default, Some(serde_json::json!("default_value")));
    }

    #[tokio::test]
    async fn test_tool_execution() {
        let protocol = Arc::new(MockProtocol);
        let tool = Tool::new("test_tool", "A test tool", protocol.clone());

        let result = tool.execute(serde_json::json!({})).await.unwrap();
        assert!(result.success);
        assert_eq!(result.output["tool"], "test_tool");
    }

    #[tokio::test]
    async fn test_tool_registry() {
        let protocol = Arc::new(MockProtocol);
        let mut registry = ToolRegistry::new(protocol.clone());

        let tool = Tool::new("calculator", "Performs calculations", protocol.clone());
        registry.add_tool(tool);

        assert!(registry.get_tool("calculator").is_some());
        assert_eq!(registry.list_tools().len(), 1);

        let result = registry
            .execute_tool("calculator", serde_json::json!({}))
            .await
            .unwrap();
        assert!(result.success);
    }
}
