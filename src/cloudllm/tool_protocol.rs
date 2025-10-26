//! Tool Protocol Abstraction Layer
//!
//! This module provides a flexible abstraction for connecting agents to various tool protocols.
//! It supports multiple standards including MCP (Model Context Protocol), custom function calling,
//! Memory persistence, and allows users to implement their own tool communication mechanisms.
//!
//! # Architecture
//!
//! **Single Protocol** (traditional):
//! ```text
//! Agent → ToolRegistry → ToolProtocol → Single Tool Source
//! ```
//!
//! **Multi-Protocol** (new in 0.5.0):
//! ```text
//! Agent → ToolRegistry → [Protocol1, Protocol2, Protocol3]
//!         (routing map)     ↓          ↓          ↓
//!                        Local      YouTube    GitHub
//!                        Tools      Server     Server
//! ```
//!
//! # Key Components
//!
//! - **ToolProtocol trait**: Define how tools are executed, discovered, and described
//! - **ToolRegistry**: Single or multi-protocol tool aggregation with transparent routing
//! - **ToolMetadata**: Tool identity, description, parameters
//! - **ToolParameter**: Type-safe parameter definitions with validation
//! - **ToolResult**: Structured tool execution results
//! - **Tool**: Runtime tool instance bound to a protocol
//!
//! # Single Protocol Example
//!
//! ```rust,no_run
//! use cloudllm::tool_protocol::ToolRegistry;
//! use cloudllm::tool_protocols::CustomToolProtocol;
//! use std::sync::Arc;
//!
//! #[tokio::main]
//! async fn main() {
//!     let protocol = Arc::new(CustomToolProtocol::new());
//!     let mut registry = ToolRegistry::new(protocol);
//!     let _ = registry.discover_tools_from_primary().await;
//! }
//! ```
//!
//! # Multi-Protocol Example
//!
//! ```rust,no_run
//! use cloudllm::tool_protocol::ToolRegistry;
//! use cloudllm::tool_protocols::{CustomToolProtocol, McpClientProtocol};
//! use std::sync::Arc;
//!
//! #[tokio::main]
//! async fn main() {
//!     let mut registry = ToolRegistry::empty();
//!     let _ = registry.add_protocol("local", Arc::new(CustomToolProtocol::new())).await;
//!     let _ = registry.add_protocol("youtube",
//!         Arc::new(McpClientProtocol::new("http://youtube-mcp:8081".to_string()))
//!     ).await;
//! }
//! ```

use crate::cloudllm::resource_protocol::ResourceMetadata;
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

    /// List available resources (MCP Resource support)
    ///
    /// Resources are application-provided contextual data that agents can read.
    /// This method is optional and defaults to returning an empty list.
    async fn list_resources(&self) -> Result<Vec<ResourceMetadata>, Box<dyn Error + Send + Sync>> {
        Ok(Vec::new())
    }

    /// Read the content of a resource by URI (MCP Resource support)
    ///
    /// This method is optional and defaults to returning NotFound.
    async fn read_resource(&self, uri: &str) -> Result<String, Box<dyn Error + Send + Sync>> {
        Err(format!("Resource not found: {}", uri).into())
    }

    /// Check if this protocol supports resources
    fn supports_resources(&self) -> bool {
        false
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
///
/// Supports single or multiple tool protocols, enabling agents to transparently
/// access tools from multiple sources (local functions, MCP servers, etc.)
///
/// # Single Protocol
///
/// ```rust,no_run
/// use cloudllm::tool_protocol::ToolRegistry;
/// use cloudllm::tool_protocols::CustomToolProtocol;
/// use std::sync::Arc;
///
/// let protocol = Arc::new(CustomToolProtocol::new());
/// let registry = ToolRegistry::new(protocol);
/// ```
///
/// # Multiple Protocols
///
/// ```rust,no_run
/// use cloudllm::tool_protocol::ToolRegistry;
/// use cloudllm::tool_protocols::{CustomToolProtocol, McpClientProtocol};
/// use std::sync::Arc;
///
/// # async {
/// let mut registry = ToolRegistry::empty();
///
/// // Add local tools
/// registry.add_protocol(
///     "local",
///     Arc::new(CustomToolProtocol::new())
/// ).await.ok();
///
/// // Add remote MCP server
/// registry.add_protocol(
///     "youtube",
///     Arc::new(McpClientProtocol::new("http://youtube-mcp:8081".to_string()))
/// ).await.ok();
///
/// // Agent transparently accesses both
/// # };
/// ```
pub struct ToolRegistry {
    /// All discovered tools from all protocols
    tools: HashMap<String, Tool>,
    /// Mapping of tool_name -> protocol_name for routing
    tool_to_protocol: HashMap<String, String>,
    /// All registered protocols
    protocols: HashMap<String, Arc<dyn ToolProtocol>>,
    /// Primary protocol (for backwards compatibility with single-protocol code)
    primary_protocol: Option<Arc<dyn ToolProtocol>>,
}

impl ToolRegistry {
    /// Build a registry powered by a single protocol implementation.
    ///
    /// This is the traditional single-protocol mode. Use `empty()` and `add_protocol()`
    /// for multi-protocol support.
    pub fn new(protocol: Arc<dyn ToolProtocol>) -> Self {
        Self {
            tools: HashMap::new(),
            tool_to_protocol: HashMap::new(),
            protocols: {
                let mut m = HashMap::new();
                m.insert("primary".to_string(), protocol.clone());
                m
            },
            primary_protocol: Some(protocol),
        }
    }

    /// Create an empty registry ready to accept multiple protocols.
    ///
    /// Use `add_protocol()` to register protocols.
    pub fn empty() -> Self {
        Self {
            tools: HashMap::new(),
            tool_to_protocol: HashMap::new(),
            protocols: HashMap::new(),
            primary_protocol: None,
        }
    }

    /// Register a protocol and discover its tools.
    ///
    /// # Arguments
    ///
    /// * `protocol_name` - Unique identifier for this protocol (e.g., "local", "youtube", "github")
    /// * `protocol` - The ToolProtocol implementation
    ///
    /// # Tool Discovery
    ///
    /// This method calls `protocol.list_tools()` to discover available tools
    /// and automatically registers them in the registry.
    ///
    /// # Conflicts
    ///
    /// If a tool with the same name already exists, it will be replaced.
    /// The new protocol's tool takes precedence.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use cloudllm::tool_protocol::ToolRegistry;
    /// use cloudllm::tool_protocols::McpClientProtocol;
    /// use std::sync::Arc;
    ///
    /// # async {
    /// let mut registry = ToolRegistry::empty();
    /// registry.add_protocol(
    ///     "memory_server",
    ///     Arc::new(McpClientProtocol::new("http://localhost:8080".to_string()))
    /// ).await?;
    /// # Ok::<(), Box<dyn std::error::Error + Send + Sync>>(())
    /// # };
    /// ```
    pub async fn add_protocol(
        &mut self,
        protocol_name: &str,
        protocol: Arc<dyn ToolProtocol>,
    ) -> Result<(), Box<dyn Error + Send + Sync>> {
        // Discover tools from this protocol
        let discovered_tools = protocol.list_tools().await?;

        // Register the protocol
        self.protocols
            .insert(protocol_name.to_string(), protocol.clone());

        // Register discovered tools
        for tool_meta in discovered_tools {
            let tool_name = tool_meta.name.clone();

            // Create a Tool that routes through this protocol
            let tool = Tool::new(
                tool_name.clone(),
                tool_meta.description.clone(),
                protocol.clone(),
            );

            // Copy over any additional parameters and metadata
            let mut tool = tool;
            for param in &tool_meta.parameters {
                tool = tool.with_parameter(param.clone());
            }
            for (key, value) in &tool_meta.protocol_metadata {
                tool = tool.with_protocol_metadata(key.clone(), value.clone());
            }

            // Register the tool and its routing
            self.tools.insert(tool_name.clone(), tool);
            self.tool_to_protocol
                .insert(tool_name, protocol_name.to_string());
        }

        Ok(())
    }

    /// Remove a protocol and all its tools from the registry.
    pub fn remove_protocol(&mut self, protocol_name: &str) {
        self.protocols.remove(protocol_name);

        // Collect tool names to remove
        let tools_to_remove: Vec<String> = self
            .tool_to_protocol
            .iter()
            .filter(|(_, pn)| *pn == protocol_name)
            .map(|(tn, _)| tn.clone())
            .collect();

        // Remove the tools
        for tool_name in tools_to_remove {
            self.tools.remove(&tool_name);
            self.tool_to_protocol.remove(&tool_name);
        }
    }

    /// Insert or replace a tool definition (for manual tool registration).
    pub fn add_tool(&mut self, tool: Tool) {
        self.tools.insert(tool.metadata.name.clone(), tool);
    }

    /// Remove a tool by name returning the owned entry if present.
    pub fn remove_tool(&mut self, name: &str) -> Option<Tool> {
        self.tool_to_protocol.remove(name);
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

    /// Discover tools from the primary protocol (for single-protocol registries).
    ///
    /// This is useful after registering tools with the protocol to populate the registry.
    /// For multi-protocol registries, use `add_protocol()` instead.
    pub async fn discover_tools_from_primary(
        &mut self,
    ) -> Result<(), Box<dyn Error + Send + Sync>> {
        if let Some(protocol) = &self.primary_protocol {
            let discovered_tools = protocol.list_tools().await?;
            for tool_meta in discovered_tools {
                let tool_name = tool_meta.name.clone();
                let tool = Tool::new(
                    tool_name.clone(),
                    tool_meta.description.clone(),
                    protocol.clone(),
                );

                // Copy over parameters and metadata
                let mut tool = tool;
                for param in &tool_meta.parameters {
                    tool = tool.with_parameter(param.clone());
                }
                for (key, value) in &tool_meta.protocol_metadata {
                    tool = tool.with_protocol_metadata(key.clone(), value.clone());
                }

                self.tools.insert(tool_name.clone(), tool);
                self.tool_to_protocol
                    .insert(tool_name, "primary".to_string());
            }
            Ok(())
        } else {
            Err("No primary protocol available".into())
        }
    }

    /// Get which protocol handles a specific tool.
    pub fn get_tool_protocol(&self, tool_name: &str) -> Option<&str> {
        self.tool_to_protocol.get(tool_name).map(|s| s.as_str())
    }

    /// Get all registered protocol names.
    pub fn list_protocols(&self) -> Vec<&str> {
        self.protocols.keys().map(|s| s.as_str()).collect()
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

    /// Borrow the primary protocol implementation (for single-protocol mode).
    ///
    /// Returns None if registry was created with `empty()` or has multiple protocols.
    pub fn protocol(&self) -> Option<&Arc<dyn ToolProtocol>> {
        self.primary_protocol.as_ref()
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

    #[tokio::test]
    async fn test_empty_registry_creation() {
        let registry = ToolRegistry::empty();
        assert_eq!(registry.list_tools().len(), 0);
        assert_eq!(registry.list_protocols().len(), 0);
        assert!(registry.protocol().is_none());
    }

    #[tokio::test]
    async fn test_add_single_protocol_to_empty_registry() {
        let protocol = Arc::new(MockProtocol);
        let mut registry = ToolRegistry::empty();

        // Add protocol
        registry
            .add_protocol("mock", protocol.clone())
            .await
            .unwrap();

        // Verify protocol was added
        assert_eq!(registry.list_protocols().len(), 1);
        assert!(registry.list_protocols().contains(&"mock"));
    }

    #[tokio::test]
    async fn test_add_multiple_protocols() {
        let protocol1 = Arc::new(MockProtocol);
        let protocol2 = Arc::new(MockProtocol);
        let mut registry = ToolRegistry::empty();

        // Add two protocols
        registry
            .add_protocol("protocol1", protocol1.clone())
            .await
            .unwrap();
        registry
            .add_protocol("protocol2", protocol2.clone())
            .await
            .unwrap();

        // Verify both protocols are registered
        assert_eq!(registry.list_protocols().len(), 2);
        assert!(registry.list_protocols().contains(&"protocol1"));
        assert!(registry.list_protocols().contains(&"protocol2"));
    }

    #[tokio::test]
    async fn test_remove_protocol() {
        let protocol = Arc::new(MockProtocol);
        let mut registry = ToolRegistry::empty();

        // Add protocol
        registry
            .add_protocol("protocol1", protocol.clone())
            .await
            .unwrap();
        assert_eq!(registry.list_protocols().len(), 1);

        // Remove protocol
        registry.remove_protocol("protocol1");
        assert_eq!(registry.list_protocols().len(), 0);
    }

    #[tokio::test]
    async fn test_get_tool_protocol() {
        let protocol = Arc::new(MockProtocol);
        let mut registry = ToolRegistry::empty();

        // Add protocol with a tool
        registry
            .add_protocol("local", protocol.clone())
            .await
            .unwrap();

        // Add a tool manually for testing
        let tool = Tool::new("calculator", "Performs calculations", protocol.clone());
        registry.add_tool(tool);
        registry
            .tool_to_protocol
            .insert("calculator".to_string(), "local".to_string());

        // Verify tool-to-protocol mapping
        assert_eq!(registry.get_tool_protocol("calculator"), Some("local"));
        assert_eq!(registry.get_tool_protocol("nonexistent"), None);
    }

    #[tokio::test]
    async fn test_remove_protocol_removes_tools() {
        let protocol = Arc::new(MockProtocol);
        let mut registry = ToolRegistry::empty();

        // Add protocol and tools
        registry
            .add_protocol("protocol1", protocol.clone())
            .await
            .unwrap();

        let tool1 = Tool::new("tool1", "First tool", protocol.clone());
        registry.add_tool(tool1);
        registry
            .tool_to_protocol
            .insert("tool1".to_string(), "protocol1".to_string());

        let tool2 = Tool::new("tool2", "Second tool", protocol.clone());
        registry.add_tool(tool2);
        registry
            .tool_to_protocol
            .insert("tool2".to_string(), "protocol1".to_string());

        assert_eq!(registry.list_tools().len(), 2);

        // Remove protocol
        registry.remove_protocol("protocol1");

        // Verify all tools from that protocol are removed
        assert_eq!(registry.list_tools().len(), 0);
        assert_eq!(registry.get_tool_protocol("tool1"), None);
        assert_eq!(registry.get_tool_protocol("tool2"), None);
    }

    #[tokio::test]
    async fn test_execute_tool_through_registry() {
        let protocol = Arc::new(MockProtocol);
        let mut registry = ToolRegistry::empty();

        // Add protocol
        registry
            .add_protocol("mock", protocol.clone())
            .await
            .unwrap();

        // Add and execute tool
        let tool = Tool::new("test_tool", "A test tool", protocol.clone());
        registry.add_tool(tool);

        let result = registry
            .execute_tool("test_tool", serde_json::json!({}))
            .await
            .unwrap();

        assert!(result.success);
        assert_eq!(result.output["tool"], "test_tool");
    }

    #[tokio::test]
    async fn test_backwards_compatibility_single_protocol() {
        let protocol = Arc::new(MockProtocol);
        let registry = ToolRegistry::new(protocol.clone());

        // Single-protocol registry should have primary_protocol set
        assert!(registry.protocol().is_some());
        assert_eq!(registry.list_protocols().len(), 1);
        assert!(registry.list_protocols().contains(&"primary"));
    }

    #[tokio::test]
    async fn test_discover_tools_from_primary() {
        struct TestProtocol {
            tools: Vec<ToolMetadata>,
        }

        #[async_trait]
        impl ToolProtocol for TestProtocol {
            async fn execute(
                &self,
                tool_name: &str,
                _parameters: serde_json::Value,
            ) -> Result<ToolResult, Box<dyn Error + Send + Sync>> {
                Ok(ToolResult::success(serde_json::json!({
                    "tool": tool_name,
                })))
            }

            async fn list_tools(&self) -> Result<Vec<ToolMetadata>, Box<dyn Error + Send + Sync>> {
                Ok(self.tools.clone())
            }

            async fn get_tool_metadata(
                &self,
                tool_name: &str,
            ) -> Result<ToolMetadata, Box<dyn Error + Send + Sync>> {
                self.tools
                    .iter()
                    .find(|t| t.name == tool_name)
                    .cloned()
                    .ok_or_else(|| "Tool not found".into())
            }

            fn protocol_name(&self) -> &str {
                "test"
            }
        }

        // Create protocol with tools
        let protocol = Arc::new(TestProtocol {
            tools: vec![
                ToolMetadata::new("tool1", "First tool"),
                ToolMetadata::new("tool2", "Second tool"),
            ],
        });

        let mut registry = ToolRegistry::new(protocol.clone());

        // Initially, registry has no tools (they haven't been discovered)
        assert_eq!(registry.list_tools().len(), 0);

        // Discover tools
        registry.discover_tools_from_primary().await.unwrap();

        // Now registry should have the tools
        assert_eq!(registry.list_tools().len(), 2);
        assert!(registry.get_tool("tool1").is_some());
        assert!(registry.get_tool("tool2").is_some());

        // Tools should be mapped to primary protocol
        assert_eq!(registry.get_tool_protocol("tool1"), Some("primary"));
        assert_eq!(registry.get_tool_protocol("tool2"), Some("primary"));
    }
}
