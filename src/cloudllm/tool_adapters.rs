//! Tool Protocol Adapters
//!
//! This module provides concrete implementations of the ToolProtocol trait
//! for various tool communication standards and custom implementations.

use crate::cloudllm::tool_protocol::{ToolError, ToolMetadata, ToolProtocol, ToolResult};
use async_trait::async_trait;
use serde_json::Value as JsonValue;
use std::collections::HashMap;
use std::error::Error;
use std::sync::Arc;
use tokio::sync::RwLock;

/// Type alias for custom tool function implementations
pub type ToolFunction =
    Arc<dyn Fn(JsonValue) -> Result<ToolResult, Box<dyn Error + Send + Sync>> + Send + Sync>;

/// Type alias for async custom tool function implementations
pub type AsyncToolFunction = Arc<
    dyn Fn(JsonValue) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<ToolResult, Box<dyn Error + Send + Sync>>> + Send>>
        + Send
        + Sync,
>;

/// Custom function-calling tool adapter
///
/// This adapter allows you to register Rust functions as tools that agents can use.
/// It's useful for quick prototyping and simple tool implementations.
///
/// # Example
///
/// ```rust,no_run
/// use cloudllm::tool_adapters::CustomToolAdapter;
/// use cloudllm::tool_protocol::{ToolResult, ToolMetadata, ToolParameter, ToolParameterType};
/// use std::sync::Arc;
///
/// let mut adapter = CustomToolAdapter::new();
///
/// // Register a synchronous tool
/// adapter.register_tool(
///     ToolMetadata::new("add", "Adds two numbers")
///         .with_parameter(
///             ToolParameter::new("a", ToolParameterType::Number).required()
///         )
///         .with_parameter(
///             ToolParameter::new("b", ToolParameterType::Number).required()
///         ),
///     Arc::new(|params| {
///         let a = params["a"].as_f64().unwrap_or(0.0);
///         let b = params["b"].as_f64().unwrap_or(0.0);
///         Ok(ToolResult::success(serde_json::json!({"result": a + b})))
///     })
/// );
/// ```
pub struct CustomToolAdapter {
    tools: Arc<RwLock<HashMap<String, ToolMetadata>>>,
    sync_functions: Arc<RwLock<HashMap<String, ToolFunction>>>,
    async_functions: Arc<RwLock<HashMap<String, AsyncToolFunction>>>,
}

impl CustomToolAdapter {
    pub fn new() -> Self {
        Self {
            tools: Arc::new(RwLock::new(HashMap::new())),
            sync_functions: Arc::new(RwLock::new(HashMap::new())),
            async_functions: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Register a synchronous tool function
    pub async fn register_tool(&self, metadata: ToolMetadata, function: ToolFunction) {
        let name = metadata.name.clone();
        self.tools.write().await.insert(name.clone(), metadata);
        self.sync_functions.write().await.insert(name, function);
    }

    /// Register an asynchronous tool function
    pub async fn register_async_tool(&self, metadata: ToolMetadata, function: AsyncToolFunction) {
        let name = metadata.name.clone();
        self.tools.write().await.insert(name.clone(), metadata);
        self.async_functions.write().await.insert(name, function);
    }

    /// Remove a tool from the adapter
    pub async fn unregister_tool(&self, name: &str) {
        self.tools.write().await.remove(name);
        self.sync_functions.write().await.remove(name);
        self.async_functions.write().await.remove(name);
    }
}

impl Default for CustomToolAdapter {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl ToolProtocol for CustomToolAdapter {
    async fn execute(
        &self,
        tool_name: &str,
        parameters: JsonValue,
    ) -> Result<ToolResult, Box<dyn Error + Send + Sync>> {
        // Try async functions first
        {
            let async_funcs = self.async_functions.read().await;
            if let Some(func) = async_funcs.get(tool_name) {
                return func(parameters).await;
            }
        }

        // Then try sync functions
        {
            let sync_funcs = self.sync_functions.read().await;
            if let Some(func) = sync_funcs.get(tool_name) {
                return func(parameters);
            }
        }

        Err(Box::new(ToolError::NotFound(tool_name.to_string())))
    }

    async fn list_tools(&self) -> Result<Vec<ToolMetadata>, Box<dyn Error + Send + Sync>> {
        let tools = self.tools.read().await;
        Ok(tools.values().cloned().collect())
    }

    async fn get_tool_metadata(
        &self,
        tool_name: &str,
    ) -> Result<ToolMetadata, Box<dyn Error + Send + Sync>> {
        let tools = self.tools.read().await;
        tools
            .get(tool_name)
            .cloned()
            .ok_or_else(|| Box::new(ToolError::NotFound(tool_name.to_string())) as Box<dyn Error + Send + Sync>)
    }

    fn protocol_name(&self) -> &str {
        "custom"
    }
}

/// MCP (Model Context Protocol) adapter
///
/// This adapter provides integration with the Model Context Protocol standard.
/// It allows agents to communicate with external tools and services using MCP.
///
/// # Example
///
/// ```rust,no_run
/// use cloudllm::tool_adapters::McpAdapter;
/// use cloudllm::tool_protocol::ToolProtocol;
///
/// # async {
/// let mut adapter = McpAdapter::new("http://localhost:8080/mcp".to_string());
/// adapter.initialize().await.unwrap();
/// # };
/// ```
pub struct McpAdapter {
    endpoint: String,
    client: reqwest::Client,
    tools_cache: Arc<RwLock<Option<Vec<ToolMetadata>>>>,
    cache_ttl_secs: u64,
    last_cache_refresh: Arc<RwLock<Option<std::time::Instant>>>,
}

impl McpAdapter {
    pub fn new(endpoint: String) -> Self {
        Self {
            endpoint,
            client: reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(30))
                .build()
                .expect("Failed to build HTTP client"),
            tools_cache: Arc::new(RwLock::new(None)),
            cache_ttl_secs: 300, // 5 minutes
            last_cache_refresh: Arc::new(RwLock::new(None)),
        }
    }

    pub fn with_timeout(mut self, timeout_secs: u64) -> Self {
        self.client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(timeout_secs))
            .build()
            .expect("Failed to build HTTP client");
        self
    }

    pub fn with_cache_ttl(mut self, ttl_secs: u64) -> Self {
        self.cache_ttl_secs = ttl_secs;
        self
    }

    async fn should_refresh_cache(&self) -> bool {
        let last_refresh = self.last_cache_refresh.read().await;
        match *last_refresh {
            None => true,
            Some(instant) => instant.elapsed().as_secs() > self.cache_ttl_secs,
        }
    }

    async fn refresh_cache(&self) -> Result<(), Box<dyn Error + Send + Sync>> {
        let response = self
            .client
            .get(format!("{}/tools", self.endpoint))
            .send()
            .await?;

        if !response.status().is_success() {
            return Err(Box::new(ToolError::ProtocolError(format!(
                "MCP server returned status: {}",
                response.status()
            ))));
        }

        let tools: Vec<ToolMetadata> = response.json().await?;
        *self.tools_cache.write().await = Some(tools);
        *self.last_cache_refresh.write().await = Some(std::time::Instant::now());

        Ok(())
    }
}

#[async_trait]
impl ToolProtocol for McpAdapter {
    async fn execute(
        &self,
        tool_name: &str,
        parameters: JsonValue,
    ) -> Result<ToolResult, Box<dyn Error + Send + Sync>> {
        let response = self
            .client
            .post(format!("{}/execute", self.endpoint))
            .json(&serde_json::json!({
                "tool": tool_name,
                "parameters": parameters
            }))
            .send()
            .await?;

        if !response.status().is_success() {
            return Err(Box::new(ToolError::ExecutionFailed(format!(
                "MCP server returned status: {}",
                response.status()
            ))));
        }

        let result: ToolResult = response.json().await?;
        Ok(result)
    }

    async fn list_tools(&self) -> Result<Vec<ToolMetadata>, Box<dyn Error + Send + Sync>> {
        if self.should_refresh_cache().await {
            self.refresh_cache().await?;
        }

        let cache = self.tools_cache.read().await;
        cache
            .as_ref()
            .cloned()
            .ok_or_else(|| Box::new(ToolError::ProtocolError("Tools cache not initialized".to_string())) as Box<dyn Error + Send + Sync>)
    }

    async fn get_tool_metadata(
        &self,
        tool_name: &str,
    ) -> Result<ToolMetadata, Box<dyn Error + Send + Sync>> {
        let tools = self.list_tools().await?;
        tools
            .into_iter()
            .find(|t| t.name == tool_name)
            .ok_or_else(|| Box::new(ToolError::NotFound(tool_name.to_string())) as Box<dyn Error + Send + Sync>)
    }

    fn protocol_name(&self) -> &str {
        "mcp"
    }

    async fn initialize(&mut self) -> Result<(), Box<dyn Error + Send + Sync>> {
        // Test connection and load initial tool list
        self.refresh_cache().await
    }

    async fn shutdown(&mut self) -> Result<(), Box<dyn Error + Send + Sync>> {
        // Clear cache
        *self.tools_cache.write().await = None;
        *self.last_cache_refresh.write().await = None;
        Ok(())
    }
}

/// OpenAI-style function calling adapter
///
/// This adapter formats tools in the OpenAI function calling format,
/// making it easy to integrate with OpenAI's function calling API.
pub struct OpenAIFunctionAdapter {
    tools: Arc<RwLock<HashMap<String, ToolMetadata>>>,
    functions: Arc<RwLock<HashMap<String, AsyncToolFunction>>>,
}

impl OpenAIFunctionAdapter {
    pub fn new() -> Self {
        Self {
            tools: Arc::new(RwLock::new(HashMap::new())),
            functions: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    pub async fn register_function(&self, metadata: ToolMetadata, function: AsyncToolFunction) {
        let name = metadata.name.clone();
        self.tools.write().await.insert(name.clone(), metadata);
        self.functions.write().await.insert(name, function);
    }

    /// Get tools in OpenAI function calling format
    pub async fn get_openai_functions(&self) -> Vec<JsonValue> {
        let tools = self.tools.read().await;
        tools
            .values()
            .map(|metadata| {
                let mut properties = serde_json::Map::new();
                let mut required = Vec::new();

                for param in &metadata.parameters {
                    properties.insert(
                        param.name.clone(),
                        serde_json::json!({
                            "type": param.param_type,
                            "description": param.description.as_deref().unwrap_or("")
                        }),
                    );

                    if param.required {
                        required.push(param.name.clone());
                    }
                }

                serde_json::json!({
                    "name": metadata.name,
                    "description": metadata.description,
                    "parameters": {
                        "type": "object",
                        "properties": properties,
                        "required": required
                    }
                })
            })
            .collect()
    }
}

impl Default for OpenAIFunctionAdapter {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl ToolProtocol for OpenAIFunctionAdapter {
    async fn execute(
        &self,
        tool_name: &str,
        parameters: JsonValue,
    ) -> Result<ToolResult, Box<dyn Error + Send + Sync>> {
        let functions = self.functions.read().await;
        let func = functions
            .get(tool_name)
            .ok_or_else(|| ToolError::NotFound(tool_name.to_string()))?;

        func(parameters).await
    }

    async fn list_tools(&self) -> Result<Vec<ToolMetadata>, Box<dyn Error + Send + Sync>> {
        let tools = self.tools.read().await;
        Ok(tools.values().cloned().collect())
    }

    async fn get_tool_metadata(
        &self,
        tool_name: &str,
    ) -> Result<ToolMetadata, Box<dyn Error + Send + Sync>> {
        let tools = self.tools.read().await;
        tools
            .get(tool_name)
            .cloned()
            .ok_or_else(|| Box::new(ToolError::NotFound(tool_name.to_string())) as Box<dyn Error + Send + Sync>)
    }

    fn protocol_name(&self) -> &str {
        "openai-functions"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cloudllm::tool_protocol::{ToolParameter, ToolParameterType};

    #[tokio::test]
    async fn test_custom_adapter_sync_tool() {
        let adapter = CustomToolAdapter::new();

        let metadata = ToolMetadata::new("add", "Adds two numbers")
            .with_parameter(ToolParameter::new("a", ToolParameterType::Number).required())
            .with_parameter(ToolParameter::new("b", ToolParameterType::Number).required());

        adapter
            .register_tool(
                metadata,
                Arc::new(|params| {
                    let a = params["a"].as_f64().unwrap_or(0.0);
                    let b = params["b"].as_f64().unwrap_or(0.0);
                    Ok(ToolResult::success(serde_json::json!({"result": a + b})))
                }),
            )
            .await;

        let result = adapter
            .execute("add", serde_json::json!({"a": 5.0, "b": 3.0}))
            .await
            .unwrap();

        assert!(result.success);
        assert_eq!(result.output["result"], 8.0);
    }

    #[tokio::test]
    async fn test_custom_adapter_async_tool() {
        let adapter = CustomToolAdapter::new();

        let metadata = ToolMetadata::new("fetch", "Fetches data asynchronously");

        adapter
            .register_async_tool(
                metadata,
                Arc::new(|_params| {
                    Box::pin(async {
                        tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
                        Ok(ToolResult::success(serde_json::json!({"data": "fetched"})))
                    })
                }),
            )
            .await;

        let result = adapter
            .execute("fetch", serde_json::json!({}))
            .await
            .unwrap();

        assert!(result.success);
        assert_eq!(result.output["data"], "fetched");
    }

    #[tokio::test]
    async fn test_custom_adapter_list_tools() {
        let adapter = CustomToolAdapter::new();

        let metadata1 = ToolMetadata::new("tool1", "First tool");
        let metadata2 = ToolMetadata::new("tool2", "Second tool");

        adapter
            .register_tool(
                metadata1,
                Arc::new(|_| Ok(ToolResult::success(serde_json::json!({})))),
            )
            .await;

        adapter
            .register_tool(
                metadata2,
                Arc::new(|_| Ok(ToolResult::success(serde_json::json!({})))),
            )
            .await;

        let tools = adapter.list_tools().await.unwrap();
        assert_eq!(tools.len(), 2);
    }

    #[tokio::test]
    async fn test_openai_function_adapter() {
        let adapter = OpenAIFunctionAdapter::new();

        let metadata = ToolMetadata::new("search", "Searches the web")
            .with_parameter(
                ToolParameter::new("query", ToolParameterType::String)
                    .with_description("The search query")
                    .required(),
            );

        adapter
            .register_function(
                metadata,
                Arc::new(|params| {
                    Box::pin(async move {
                        let query = params["query"].as_str().unwrap_or("");
                        Ok(ToolResult::success(serde_json::json!({
                            "results": [
                                {"title": "Result 1", "url": "http://example.com/1"},
                                {"title": "Result 2", "url": "http://example.com/2"}
                            ],
                            "query": query
                        })))
                    })
                }),
            )
            .await;

        let functions = adapter.get_openai_functions().await;
        assert_eq!(functions.len(), 1);
        assert_eq!(functions[0]["name"], "search");
        assert_eq!(functions[0]["description"], "Searches the web");

        let result = adapter
            .execute("search", serde_json::json!({"query": "rust programming"}))
            .await
            .unwrap();

        assert!(result.success);
        assert_eq!(result.output["query"], "rust programming");
    }
}
