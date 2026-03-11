//! Generic builder for MCP-compatible HTTP tool servers.

use crate::builder_utils::{AuthConfig, IpFilter};
use crate::events::McpEventHandler;
use crate::http::{HttpServerAdapter, HttpServerConfig, HttpServerInstance};
use crate::protocol::ToolProtocol;
use crate::server::UnifiedMcpServer;
use std::error::Error;
use std::net::SocketAddr;
use std::sync::Arc;

/// Builder for creating MCP-compatible HTTP servers with registered tools.
pub struct MCPServerBuilder {
    server: UnifiedMcpServer,
    ip_filter: IpFilter,
    auth: AuthConfig,
    adapter: Arc<dyn HttpServerAdapter>,
    event_handler: Option<Arc<dyn McpEventHandler>>,
}

impl MCPServerBuilder {
    /// Create a new empty builder using the default HTTP adapter.
    pub fn new() -> Self {
        Self {
            server: UnifiedMcpServer::new(),
            ip_filter: IpFilter::new(),
            auth: AuthConfig::None,
            adapter: Self::default_adapter(),
            event_handler: None,
        }
    }

    #[cfg(feature = "server")]
    fn default_adapter() -> Arc<dyn HttpServerAdapter> {
        Arc::new(crate::http::AxumHttpAdapter)
    }

    #[cfg(not(feature = "server"))]
    fn default_adapter() -> Arc<dyn HttpServerAdapter> {
        panic!(
            "{}",
            "MCPServerBuilder requires the 'server' feature to be enabled."
        )
    }

    /// Register a custom tool protocol under a tool name.
    pub async fn with_custom_tool(
        mut self,
        tool_name: &str,
        protocol: Arc<dyn ToolProtocol>,
    ) -> Self {
        self.server.register_tool(tool_name, protocol).await;
        self
    }

    /// Require bearer token authentication.
    pub fn with_bearer_token(mut self, token: impl Into<String>) -> Self {
        self.auth = AuthConfig::bearer(token);
        self
    }

    /// Require basic authentication.
    pub fn with_basic_auth(
        mut self,
        username: impl Into<String>,
        password: impl Into<String>,
    ) -> Self {
        self.auth = AuthConfig::basic(username, password);
        self
    }

    /// Allow a specific IP address.
    pub fn allow_ip(mut self, ip: &str) -> Result<Self, String> {
        self.ip_filter.allow(ip)?;
        Ok(self)
    }

    /// Allow a CIDR block.
    pub fn allow_cidr(mut self, cidr: &str) -> Result<Self, String> {
        self.ip_filter.allow(cidr)?;
        Ok(self)
    }

    /// Restrict access to localhost only.
    pub fn allow_localhost_only(mut self) -> Self {
        let _ = self.ip_filter.allow("127.0.0.1");
        let _ = self.ip_filter.allow("::1");
        self
    }

    /// Override the HTTP adapter.
    pub fn with_adapter(mut self, adapter: Arc<dyn HttpServerAdapter>) -> Self {
        self.adapter = adapter;
        self
    }

    /// Attach an MCP event handler.
    pub fn with_event_handler(mut self, handler: Arc<dyn McpEventHandler>) -> Self {
        self.event_handler = Some(handler);
        self
    }

    /// Start the server on the supplied port on localhost.
    pub async fn start_on(
        self,
        port: u16,
    ) -> Result<HttpServerInstance, Box<dyn Error + Send + Sync>> {
        self.start_at(SocketAddr::from(([127, 0, 0, 1], port)))
            .await
    }

    /// Start the server at an explicit socket address.
    pub async fn start_at(
        self,
        addr: SocketAddr,
    ) -> Result<HttpServerInstance, Box<dyn Error + Send + Sync>> {
        let bearer_token = match self.auth {
            AuthConfig::None => None,
            AuthConfig::Bearer(token) => Some(token),
            AuthConfig::Basic { .. } => {
                return Err("Basic auth is not supported by the generic MCP HTTP adapter".into())
            }
        };

        self.adapter
            .start(
                HttpServerConfig {
                    addr,
                    bearer_token,
                    ip_filter: self.ip_filter,
                    event_handler: self.event_handler,
                },
                Arc::new(self.server),
            )
            .await
    }
}

impl Default for MCPServerBuilder {
    fn default() -> Self {
        Self::new()
    }
}
