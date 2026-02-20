//! MCP Server Builder
//!
//! Simplifies creation and deployment of MCP servers with tools, security, and IP filtering.
//!
//! # Example
//!
//! ```rust,ignore
//! use cloudllm::mcp_server_builder::MCPServerBuilder;
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let builder = MCPServerBuilder::new();
//!
//!     // Add built-in tools
//!     let builder = builder.with_memory_tool().await;
//!
//!     // Configure security
//!     let builder = builder.allow_localhost_only();
//!     let builder = builder.with_bearer_token("my-secret-token");
//!
//!     // Start the server
//!     let server = builder.start_on(8080).await?;
//!     println!("Server running at {}", server.addr());
//!
//!     Ok(())
//! }
//! ```

use crate::cloudllm::event::EventHandler;
use crate::cloudllm::mcp_http_adapter::{HttpServerAdapter, HttpServerConfig, HttpServerInstance};
use crate::cloudllm::mcp_server::UnifiedMcpServer;
use crate::cloudllm::mcp_server_builder_utils::{AuthConfig, IpFilter};
use crate::cloudllm::tool_protocol::ToolProtocol;
use crate::cloudllm::tool_protocols::{BashProtocol, MemoryProtocol};
use crate::cloudllm::tools::{BashTool, Memory, Platform};
use std::error::Error;
use std::net::SocketAddr;
use std::sync::Arc;

/// Builder for creating MCP servers with simplified API
///
/// The MCPServerBuilder hides HTTP framework complexity and provides a
/// fluent API for configuring MCP servers with tools, authentication, and IP filtering.
pub struct MCPServerBuilder {
    /// The unified MCP server that aggregates all tools
    server: UnifiedMcpServer,
    /// IP filtering configuration
    ip_filter: IpFilter,
    /// Authentication configuration
    auth: AuthConfig,
    /// HTTP framework adapter (trait object for swappability)
    adapter: Arc<dyn HttpServerAdapter>,
    /// Optional event handler for MCP server lifecycle and request events
    event_handler: Option<Arc<dyn EventHandler>>,
}

impl MCPServerBuilder {
    /// Create a new MCP server builder with default settings
    ///
    /// By default:
    /// - No tools are registered
    /// - No IP filtering (allows all)
    /// - No authentication required
    /// - Uses Axum HTTP adapter (if "mcp-server" feature enabled)
    pub fn new() -> Self {
        Self {
            server: UnifiedMcpServer::new(),
            ip_filter: IpFilter::new(),
            auth: AuthConfig::None,
            adapter: Self::default_adapter(),
            event_handler: None,
        }
    }

    /// Get the default HTTP adapter
    #[cfg(feature = "mcp-server")]
    fn default_adapter() -> Arc<dyn HttpServerAdapter> {
        use crate::cloudllm::mcp_http_adapter::AxumHttpAdapter;
        Arc::new(AxumHttpAdapter)
    }

    /// Get the default HTTP adapter (stub when feature not enabled)
    #[cfg(not(feature = "mcp-server"))]
    fn default_adapter() -> Arc<dyn HttpServerAdapter> {
        panic!(
            "{}", "MCPServerBuilder requires 'mcp-server' feature to be enabled. \
             Add this to Cargo.toml: cloudllm = {{ version = \"...\", features = [\"mcp-server\"] }}"
        )
    }

    /// Add the Memory tool to this server
    pub async fn with_memory_tool(mut self) -> Self {
        let memory = Arc::new(Memory::new());
        let protocol = Arc::new(MemoryProtocol::new(memory));
        // Register tool with server
        let _ = self.server.register_tool("memory", protocol).await;
        self
    }

    /// Add the Bash tool to this server
    ///
    /// # Arguments
    ///
    /// * `platform` - The platform to run bash commands on (Linux or macOS)
    /// * `timeout_secs` - Maximum time in seconds for command execution
    pub async fn with_bash_tool(mut self, platform: Platform, timeout_secs: u64) -> Self {
        let bash_tool = Arc::new(BashTool::new(platform).with_timeout(timeout_secs));
        let protocol = Arc::new(BashProtocol::new(bash_tool));
        // Register tool with server
        let _ = self.server.register_tool("bash", protocol).await;
        self
    }

    /// Add a custom tool to the server
    ///
    /// # Arguments
    ///
    /// * `tool_name` - Unique name for the tool
    /// * `protocol` - The ToolProtocol implementation for this tool
    pub async fn with_custom_tool(
        mut self,
        tool_name: &str,
        protocol: Arc<dyn ToolProtocol>,
    ) -> Self {
        let _ = self.server.register_tool(tool_name, protocol).await;
        self
    }

    /// Set bearer token authentication
    ///
    /// Requires requests to include: `Authorization: Bearer <token>`
    pub fn with_bearer_token(mut self, token: impl Into<String>) -> Self {
        self.auth = AuthConfig::bearer(token);
        self
    }

    /// Set basic authentication
    ///
    /// Requires requests to include: `Authorization: Basic <base64(username:password)>`
    pub fn with_basic_auth(
        mut self,
        username: impl Into<String>,
        password: impl Into<String>,
    ) -> Self {
        self.auth = AuthConfig::basic(username, password);
        self
    }

    /// Allow a specific IP address
    ///
    /// # Arguments
    ///
    /// * `ip` - IP address (e.g., "127.0.0.1" or "::1")
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use cloudllm::mcp_server_builder::MCPServerBuilder;
    /// let builder = MCPServerBuilder::new()
    ///     .allow_ip("127.0.0.1")?
    ///     .allow_ip("::1")?;
    /// ```
    pub fn allow_ip(mut self, ip: &str) -> Result<Self, String> {
        self.ip_filter.allow(ip)?;
        Ok(self)
    }

    /// Allow a CIDR block
    ///
    /// # Arguments
    ///
    /// * `cidr` - CIDR block (e.g., "192.168.1.0/24" or "2001:db8::/32")
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use cloudllm::mcp_server_builder::MCPServerBuilder;
    /// let builder = MCPServerBuilder::new()
    ///     .allow_cidr("192.168.1.0/24")?
    ///     .allow_cidr("10.0.0.0/8")?;
    /// ```
    pub fn allow_cidr(mut self, cidr: &str) -> Result<Self, String> {
        self.ip_filter.allow(cidr)?;
        Ok(self)
    }

    /// Allow only localhost connections
    ///
    /// Convenience method that allows both IPv4 and IPv6 localhost:
    /// - 127.0.0.1
    /// - ::1
    pub fn allow_localhost_only(mut self) -> Self {
        let _ = self.ip_filter.allow("127.0.0.1");
        let _ = self.ip_filter.allow("::1");
        self
    }

    /// Set a custom HTTP server adapter
    ///
    /// This allows using a different HTTP framework (e.g., Actix, Warp, Rocket)
    /// instead of the default Axum adapter.
    pub fn with_adapter(mut self, adapter: Arc<dyn HttpServerAdapter>) -> Self {
        self.adapter = adapter;
        self
    }

    /// Attach an event handler to receive MCP server lifecycle and request events.
    ///
    /// When set, the handler will receive [`McpEvent`] variants for server startup,
    /// tool list requests, tool executions, and rejected connections.
    pub fn with_event_handler(mut self, handler: Arc<dyn EventHandler>) -> Self {
        self.event_handler = Some(handler);
        self
    }

    /// Start the MCP server on the specified port
    ///
    /// # Arguments
    ///
    /// * `port` - Port number to listen on (e.g., 8080)
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use cloudllm::mcp_server_builder::MCPServerBuilder;
    /// #[tokio::main]
    /// async fn main() -> Result<(), Box<dyn std::error::Error>> {
    ///     let server = MCPServerBuilder::new()
    ///         .start_on(8080)
    ///         .await?;
    ///     println!("Server at {}", server.addr());
    ///     Ok(())
    /// }
    /// ```
    pub async fn start_on(
        self,
        port: u16,
    ) -> Result<HttpServerInstance, Box<dyn Error + Send + Sync>> {
        let addr = SocketAddr::from(([127, 0, 0, 1], port));
        self.start_at(addr).await
    }

    /// Start the MCP server at the specified address
    ///
    /// # Arguments
    ///
    /// * `addr` - Socket address to bind to
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use cloudllm::mcp_server_builder::MCPServerBuilder;
    /// use std::net::{IpAddr, Ipv4Addr, SocketAddr};
    ///
    /// #[tokio::main]
    /// async fn main() -> Result<(), Box<dyn std::error::Error>> {
    ///     let addr = SocketAddr::new(IpAddr::V4(Ipv4Addr::new(0, 0, 0, 0)), 8080);
    ///     let server = MCPServerBuilder::new()
    ///         .start_at(addr)
    ///         .await?;
    ///     Ok(())
    /// }
    /// ```
    pub async fn start_at(
        self,
        addr: SocketAddr,
    ) -> Result<HttpServerInstance, Box<dyn Error + Send + Sync>> {
        // Build configuration
        let config = HttpServerConfig {
            addr,
            bearer_token: match self.auth {
                AuthConfig::Bearer(token) => Some(token),
                _ => None,
            },
            allowed_ips: Vec::new(), // TODO: Extract from ip_filter
            event_handler: self.event_handler,
        };

        // Start server using adapter
        self.adapter.start(config, Arc::new(self.server)).await
    }
}

impl Default for MCPServerBuilder {
    fn default() -> Self {
        Self::new()
    }
}
