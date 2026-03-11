//! HTTP client protocol for talking to MCP-compatible legacy tool servers.

use crate::events::{McpEvent, McpEventHandler};
use crate::protocol::{ToolError, ToolMetadata, ToolProtocol, ToolResult};
use async_trait::async_trait;
use serde_json::Value as JsonValue;
use std::error::Error;
use std::sync::Arc;
use tokio::sync::RwLock;

/// HTTP client adapter for MCP-compatible tool servers.
pub struct McpClientProtocol {
    endpoint: String,
    client: reqwest::Client,
    tools_cache: Arc<RwLock<Option<Vec<ToolMetadata>>>>,
    cache_ttl_secs: u64,
    last_cache_refresh: Arc<RwLock<Option<std::time::Instant>>>,
    event_handler: Option<Arc<dyn McpEventHandler>>,
}

impl McpClientProtocol {
    /// Create a new client for a remote endpoint.
    pub fn new(endpoint: String) -> Self {
        Self {
            endpoint,
            client: reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(30))
                .build()
                .expect("Failed to build HTTP client"),
            tools_cache: Arc::new(RwLock::new(None)),
            cache_ttl_secs: 300,
            last_cache_refresh: Arc::new(RwLock::new(None)),
            event_handler: None,
        }
    }

    /// Override the request timeout.
    pub fn with_timeout(mut self, timeout_secs: u64) -> Self {
        self.client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(timeout_secs))
            .build()
            .expect("Failed to build HTTP client");
        self
    }

    /// Override the metadata cache TTL.
    pub fn with_cache_ttl(mut self, ttl_secs: u64) -> Self {
        self.cache_ttl_secs = ttl_secs;
        self
    }

    /// Attach an event handler.
    pub fn with_event_handler(mut self, handler: Arc<dyn McpEventHandler>) -> Self {
        self.event_handler = Some(handler);
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
            .post(format!("{}/tools/list", self.endpoint))
            .json(&serde_json::json!({}))
            .send()
            .await?;

        if !response.status().is_success() {
            return Err(Box::new(ToolError::ProtocolError(format!(
                "MCP server returned status: {}",
                response.status()
            ))));
        }

        let body: serde_json::Value = response.json().await?;
        let tools: Vec<ToolMetadata> =
            if let Some(arr) = body.get("tools").and_then(|v| v.as_array()) {
                serde_json::from_value(serde_json::Value::Array(arr.clone())).map_err(|e| {
                    Box::new(ToolError::ProtocolError(format!(
                        "Failed to deserialize tool list from MCP server: {}",
                        e
                    ))) as Box<dyn Error + Send + Sync>
                })?
            } else {
                serde_json::from_value(body).map_err(|e| {
                    Box::new(ToolError::ProtocolError(format!(
                        "Failed to deserialize tool list from MCP server: {}",
                        e
                    ))) as Box<dyn Error + Send + Sync>
                })?
            };

        let tool_count = tools.len();
        let tool_names: Vec<String> = tools.iter().map(|t| t.name.clone()).collect();

        *self.tools_cache.write().await = Some(tools);
        *self.last_cache_refresh.write().await = Some(std::time::Instant::now());

        if let Some(ref eh) = self.event_handler {
            eh.on_mcp_event(&McpEvent::ToolsDiscovered {
                endpoint: self.endpoint.clone(),
                tool_count,
                tool_names,
            })
            .await;
        }

        Ok(())
    }
}

#[async_trait]
impl ToolProtocol for McpClientProtocol {
    async fn execute(
        &self,
        tool_name: &str,
        parameters: JsonValue,
    ) -> Result<ToolResult, Box<dyn Error + Send + Sync>> {
        if let Some(ref eh) = self.event_handler {
            eh.on_mcp_event(&McpEvent::RemoteToolCallStarted {
                endpoint: self.endpoint.clone(),
                tool_name: tool_name.to_string(),
                parameters: parameters.clone(),
            })
            .await;
        }

        let call_start = std::time::Instant::now();

        let response = self
            .client
            .post(format!("{}/tools/execute", self.endpoint))
            .json(&serde_json::json!({
                "tool": tool_name,
                "parameters": parameters
            }))
            .send()
            .await;

        match response {
            Err(e) => {
                let duration_ms = call_start.elapsed().as_millis() as u64;
                if let Some(ref eh) = self.event_handler {
                    eh.on_mcp_event(&McpEvent::ToolError {
                        source: self.endpoint.clone(),
                        tool_name: tool_name.to_string(),
                        error: e.to_string(),
                        duration_ms,
                    })
                    .await;
                }
                Err(Box::new(e))
            }
            Ok(resp) => {
                if !resp.status().is_success() {
                    let duration_ms = call_start.elapsed().as_millis() as u64;
                    let err_msg = format!("MCP server returned status: {}", resp.status());
                    if let Some(ref eh) = self.event_handler {
                        eh.on_mcp_event(&McpEvent::ToolError {
                            source: self.endpoint.clone(),
                            tool_name: tool_name.to_string(),
                            error: err_msg.clone(),
                            duration_ms,
                        })
                        .await;
                    }
                    return Err(Box::new(ToolError::ExecutionFailed(err_msg)));
                }

                let body: serde_json::Value = resp.json().await?;
                let result: ToolResult = if let Some(r) = body.get("result") {
                    serde_json::from_value(r.clone()).map_err(|e| {
                        Box::new(ToolError::ProtocolError(format!(
                            "Failed to deserialize tool result from MCP server: {}",
                            e
                        ))) as Box<dyn Error + Send + Sync>
                    })?
                } else {
                    serde_json::from_value(body).map_err(|e| {
                        Box::new(ToolError::ProtocolError(format!(
                            "Failed to deserialize tool result from MCP server: {}",
                            e
                        ))) as Box<dyn Error + Send + Sync>
                    })?
                };

                let duration_ms = call_start.elapsed().as_millis() as u64;
                if let Some(ref eh) = self.event_handler {
                    eh.on_mcp_event(&McpEvent::RemoteToolCallCompleted {
                        endpoint: self.endpoint.clone(),
                        tool_name: tool_name.to_string(),
                        success: result.success,
                        error: result.error.clone(),
                        duration_ms,
                    })
                    .await;
                }

                Ok(result)
            }
        }
    }

    async fn list_tools(&self) -> Result<Vec<ToolMetadata>, Box<dyn Error + Send + Sync>> {
        if self.should_refresh_cache().await {
            let had_cache = self.last_cache_refresh.read().await.is_some();
            if had_cache {
                if let Some(ref eh) = self.event_handler {
                    eh.on_mcp_event(&McpEvent::CacheExpired {
                        endpoint: self.endpoint.clone(),
                    })
                    .await;
                }
            }
            self.refresh_cache().await?;
        } else if let Some(ref eh) = self.event_handler {
            let cache = self.tools_cache.read().await;
            let count = cache.as_ref().map_or(0, |t| t.len());
            drop(cache);
            eh.on_mcp_event(&McpEvent::CacheHit {
                endpoint: self.endpoint.clone(),
                tool_count: count,
            })
            .await;
        }

        let cache = self.tools_cache.read().await;
        cache.as_ref().cloned().ok_or_else(|| {
            Box::new(ToolError::ProtocolError(
                "Tools cache not initialized".to_string(),
            )) as Box<dyn Error + Send + Sync>
        })
    }

    async fn get_tool_metadata(
        &self,
        tool_name: &str,
    ) -> Result<ToolMetadata, Box<dyn Error + Send + Sync>> {
        let tools = self.list_tools().await?;
        tools
            .into_iter()
            .find(|t| t.name == tool_name)
            .ok_or_else(|| {
                Box::new(ToolError::NotFound(tool_name.to_string())) as Box<dyn Error + Send + Sync>
            })
    }

    fn protocol_name(&self) -> &str {
        "mcp"
    }

    async fn initialize(&mut self) -> Result<(), Box<dyn Error + Send + Sync>> {
        self.refresh_cache().await?;

        let tool_count = {
            let cache = self.tools_cache.read().await;
            cache.as_ref().map_or(0, |t| t.len())
        };

        if let Some(ref eh) = self.event_handler {
            eh.on_mcp_event(&McpEvent::ConnectionInitialized {
                endpoint: self.endpoint.clone(),
                tool_count,
            })
            .await;
        }

        Ok(())
    }

    async fn shutdown(&mut self) -> Result<(), Box<dyn Error + Send + Sync>> {
        if let Some(ref eh) = self.event_handler {
            eh.on_mcp_event(&McpEvent::ConnectionClosed {
                endpoint: self.endpoint.clone(),
            })
            .await;
        }
        *self.tools_cache.write().await = None;
        *self.last_cache_refresh.write().await = None;
        Ok(())
    }
}
