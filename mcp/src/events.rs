//! MCP-specific event types and observer trait.

use async_trait::async_trait;

/// Events emitted by MCP servers and MCP HTTP clients.
#[derive(Debug, Clone)]
pub enum McpEvent {
    /// The HTTP server started listening on the supplied address.
    ServerStarted {
        /// Bound socket address.
        addr: String,
    },
    /// A tool-list request was received from a client.
    ToolListRequested {
        /// Client IP address.
        client_addr: String,
    },
    /// A tool-list response was returned to the client.
    ToolListReturned {
        /// Client IP address.
        client_addr: String,
        /// Number of tools returned.
        tool_count: usize,
    },
    /// A tool execution request was received from a client.
    ToolCallReceived {
        /// Client IP address.
        client_addr: String,
        /// Requested tool name.
        tool_name: String,
        /// Raw JSON parameters.
        parameters: serde_json::Value,
    },
    /// A server-side tool execution completed.
    ToolCallCompleted {
        /// Client IP address.
        client_addr: String,
        /// Tool name.
        tool_name: String,
        /// Whether execution succeeded.
        success: bool,
        /// Optional error message.
        error: Option<String>,
        /// Execution duration in milliseconds.
        duration_ms: u64,
    },
    /// A tool call failed at the transport or protocol layer.
    ToolError {
        /// Client IP or remote endpoint URL.
        source: String,
        /// Tool name.
        tool_name: String,
        /// Error message.
        error: String,
        /// Elapsed time in milliseconds.
        duration_ms: u64,
    },
    /// A request was rejected before tool execution.
    RequestRejected {
        /// Client IP address.
        client_addr: String,
        /// Human-readable rejection reason.
        reason: String,
    },
    /// An MCP client initialized successfully.
    ConnectionInitialized {
        /// Remote endpoint URL.
        endpoint: String,
        /// Number of discovered tools.
        tool_count: usize,
    },
    /// An MCP client connection was closed.
    ConnectionClosed {
        /// Remote endpoint URL.
        endpoint: String,
    },
    /// The tool cache was refreshed from the remote endpoint.
    ToolsDiscovered {
        /// Remote endpoint URL.
        endpoint: String,
        /// Number of discovered tools.
        tool_count: usize,
        /// Discovered tool names.
        tool_names: Vec<String>,
    },
    /// Cached tool metadata was used.
    CacheHit {
        /// Remote endpoint URL.
        endpoint: String,
        /// Number of cached tools.
        tool_count: usize,
    },
    /// Cached tool metadata expired.
    CacheExpired {
        /// Remote endpoint URL.
        endpoint: String,
    },
    /// A remote tool call is being dispatched.
    RemoteToolCallStarted {
        /// Remote endpoint URL.
        endpoint: String,
        /// Tool name.
        tool_name: String,
        /// JSON parameters.
        parameters: serde_json::Value,
    },
    /// A remote tool call completed.
    RemoteToolCallCompleted {
        /// Remote endpoint URL.
        endpoint: String,
        /// Tool name.
        tool_name: String,
        /// Whether the tool result succeeded.
        success: bool,
        /// Optional error message.
        error: Option<String>,
        /// Round-trip duration in milliseconds.
        duration_ms: u64,
    },
}

/// Narrow observer interface for MCP lifecycle events.
#[async_trait]
pub trait McpEventHandler: Send + Sync {
    /// Observe an MCP event.
    async fn on_mcp_event(&self, _event: &McpEvent) {}
}
