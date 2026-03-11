//! CloudLLM convenience wrapper around the shared `mcp::MCPServerBuilder`.

use crate::cloudllm::event::{EventHandler, McpEvent};
use crate::cloudllm::mcp_http_adapter::{HttpServerAdapter, HttpServerInstance};
use crate::cloudllm::tool_protocol::ToolProtocol;
use crate::cloudllm::tool_protocols::{BashProtocol, MemoryProtocol};
use crate::cloudllm::tools::{BashTool, Memory, Platform};
use mcp::MCPServerBuilder as InnerBuilder;
use std::error::Error;
use std::net::SocketAddr;
use std::sync::Arc;

struct EventHandlerAdapter {
    inner: Arc<dyn EventHandler>,
}

#[async_trait::async_trait]
impl mcp::McpEventHandler for EventHandlerAdapter {
    async fn on_mcp_event(&self, event: &McpEvent) {
        self.inner.on_mcp_event(event).await;
    }
}

/// Builder for creating MCP servers with CloudLLM-specific convenience methods.
pub struct MCPServerBuilder {
    inner: InnerBuilder,
}

impl MCPServerBuilder {
    /// Create a new MCP server builder with default settings.
    pub fn new() -> Self {
        Self {
            inner: InnerBuilder::new(),
        }
    }

    /// Add the in-process memory tool.
    pub async fn with_memory_tool(self) -> Self {
        let memory = Arc::new(Memory::new());
        let protocol = Arc::new(MemoryProtocol::new(memory));
        self.with_custom_tool("memory", protocol).await
    }

    /// Add the bash tool.
    pub async fn with_bash_tool(self, platform: Platform, timeout_secs: u64) -> Self {
        let bash_tool = Arc::new(BashTool::new(platform).with_timeout(timeout_secs));
        let protocol = Arc::new(BashProtocol::new(bash_tool));
        self.with_custom_tool("bash", protocol).await
    }

    /// Add a custom tool protocol.
    pub async fn with_custom_tool(
        mut self,
        tool_name: &str,
        protocol: Arc<dyn ToolProtocol>,
    ) -> Self {
        self.inner = self.inner.with_custom_tool(tool_name, protocol).await;
        self
    }

    /// Require bearer token authentication.
    pub fn with_bearer_token(mut self, token: impl Into<String>) -> Self {
        self.inner = self.inner.with_bearer_token(token);
        self
    }

    /// Require basic authentication.
    pub fn with_basic_auth(
        mut self,
        username: impl Into<String>,
        password: impl Into<String>,
    ) -> Self {
        self.inner = self.inner.with_basic_auth(username, password);
        self
    }

    /// Allow a specific IP address.
    pub fn allow_ip(mut self, ip: &str) -> Result<Self, String> {
        self.inner = self.inner.allow_ip(ip)?;
        Ok(self)
    }

    /// Allow a CIDR block.
    pub fn allow_cidr(mut self, cidr: &str) -> Result<Self, String> {
        self.inner = self.inner.allow_cidr(cidr)?;
        Ok(self)
    }

    /// Restrict access to localhost only.
    pub fn allow_localhost_only(mut self) -> Self {
        self.inner = self.inner.allow_localhost_only();
        self
    }

    /// Override the HTTP adapter.
    pub fn with_adapter(mut self, adapter: Arc<dyn HttpServerAdapter>) -> Self {
        self.inner = self.inner.with_adapter(adapter);
        self
    }

    /// Attach a CloudLLM event handler.
    pub fn with_event_handler(mut self, handler: Arc<dyn EventHandler>) -> Self {
        self.inner = self
            .inner
            .with_event_handler(Arc::new(EventHandlerAdapter { inner: handler }));
        self
    }

    /// Start on localhost and the given port.
    pub async fn start_on(
        self,
        port: u16,
    ) -> Result<HttpServerInstance, Box<dyn Error + Send + Sync>> {
        self.inner.start_on(port).await
    }

    /// Start at an explicit socket address.
    pub async fn start_at(
        self,
        addr: SocketAddr,
    ) -> Result<HttpServerInstance, Box<dyn Error + Send + Sync>> {
        self.inner.start_at(addr).await
    }
}

impl Default for MCPServerBuilder {
    fn default() -> Self {
        Self::new()
    }
}
