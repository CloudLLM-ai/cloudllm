//! HTTP Server Adapter for MCP
//!
//! This module defines a pluggable interface for HTTP servers that expose MCP protocols.
//! The adapter pattern allows different HTTP frameworks (axum, actix, warp, etc.) to be
//! swapped without changing the MCPServerBuilder API.
//!
//! # Design
//!
//! ```text
//! MCPServerBuilder
//!        ↓
//!   (configures)
//!        ↓
//! HttpServerAdapter (trait)
//!        ↓ (implements)
//!   ┌────┴────┬─────────────┐
//!   ↓         ↓             ↓
//! AxumAdapter ActixAdapter OtherAdapter
//! ```
//!
//! This allows users to swap HTTP frameworks without changing their code.

use crate::cloudllm::event::EventHandler;
#[cfg(feature = "mcp-server")]
use crate::cloudllm::event::McpEvent;
use crate::cloudllm::tool_protocol::ToolProtocol;
use std::error::Error;
use std::net::SocketAddr;
use std::sync::Arc;

/// Configuration for an HTTP MCP server
pub struct HttpServerConfig {
    /// Socket address to bind to (e.g., "127.0.0.1:8080")
    pub addr: SocketAddr,
    /// Optional bearer token for authentication
    pub bearer_token: Option<String>,
    /// Optional list of allowed IP addresses and CIDR blocks
    pub allowed_ips: Vec<String>,
    /// Optional event handler for MCP server lifecycle and request events
    pub event_handler: Option<Arc<dyn EventHandler>>,
}

impl Clone for HttpServerConfig {
    fn clone(&self) -> Self {
        Self {
            addr: self.addr,
            bearer_token: self.bearer_token.clone(),
            allowed_ips: self.allowed_ips.clone(),
            event_handler: self.event_handler.clone(),
        }
    }
}

impl std::fmt::Debug for HttpServerConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HttpServerConfig")
            .field("addr", &self.addr)
            .field("bearer_token", &self.bearer_token)
            .field("allowed_ips", &self.allowed_ips)
            .field("has_event_handler", &self.event_handler.is_some())
            .finish()
    }
}

/// A running HTTP server instance
pub struct HttpServerInstance {
    /// Socket address the server is listening on
    pub addr: SocketAddr,
    /// Handle for shutting down the server
    /// Type erased to allow different framework implementations
    shutdown_handle: Box<dyn std::any::Any + Send + Sync>,
}

impl HttpServerInstance {
    /// Create a new server instance with the given address and shutdown handle
    pub fn new(addr: SocketAddr, shutdown_handle: Box<dyn std::any::Any + Send + Sync>) -> Self {
        Self {
            addr,
            shutdown_handle,
        }
    }

    /// Get the server's socket address
    pub fn get_addr(&self) -> SocketAddr {
        self.addr
    }

    /// Get mutable reference to the shutdown handle for advanced usage
    pub fn shutdown_handle_mut(&mut self) -> &mut Box<dyn std::any::Any + Send + Sync> {
        &mut self.shutdown_handle
    }
}

/// Trait for HTTP server implementations
///
/// Implementations of this trait provide HTTP endpoints for MCP protocols.
/// Different HTTP frameworks can be swapped by implementing this trait.
#[async_trait::async_trait]
pub trait HttpServerAdapter: Send + Sync {
    /// Start the HTTP server with the given configuration and tool protocol
    ///
    /// # Arguments
    ///
    /// * `config` - Server configuration (address, auth, IP filtering)
    /// * `protocol` - The ToolProtocol implementation to expose
    ///
    /// # Endpoints
    ///
    /// The server must provide the following endpoints:
    /// - `POST /tools/list` - List all available tools from the protocol
    /// - `POST /tools/execute` - Execute a tool with given parameters
    /// - `POST /resources/list` - List all available resources (if protocol supports)
    /// - `POST /resources/read` - Read a resource by URI (if protocol supports)
    ///
    /// # Returns
    ///
    /// A running server instance, or an error if startup fails
    async fn start(
        &self,
        config: HttpServerConfig,
        protocol: Arc<dyn ToolProtocol>,
    ) -> Result<HttpServerInstance, Box<dyn Error + Send + Sync>>;

    /// Get the name of this adapter (for logging/debugging)
    fn name(&self) -> &str {
        "unknown"
    }
}

/// Default Axum-based HTTP server adapter
///
/// Provides a full MCP-compatible HTTP server using the Axum framework.
/// Only available when the "mcp-server" feature is enabled.
#[cfg(feature = "mcp-server")]
pub struct AxumHttpAdapter;

#[cfg(feature = "mcp-server")]
#[async_trait::async_trait]
impl HttpServerAdapter for AxumHttpAdapter {
    async fn start(
        &self,
        config: HttpServerConfig,
        protocol: Arc<dyn ToolProtocol>,
    ) -> Result<HttpServerInstance, Box<dyn Error + Send + Sync>> {
        use axum::{
            extract::ConnectInfo, http::StatusCode, response::IntoResponse, routing::post, Json,
            Router,
        };
        use serde_json::json;
        use std::net::IpAddr;
        use std::str::FromStr;
        use tokio::net::TcpListener;

        // Parse allowed IPs for filtering
        let allowed_ips: Vec<IpAddr> = config
            .allowed_ips
            .iter()
            .filter_map(|ip_str| {
                if let Ok(addr) = IpAddr::from_str(ip_str) {
                    return Some(addr);
                }
                // TODO: Handle CIDR notation in future enhancement
                None
            })
            .collect();

        let bearer_token = Arc::new(config.bearer_token.clone());
        let allowed_ips = Arc::new(allowed_ips);

        // Clone per-route: token, IPs, protocol, event handler
        let token_list = bearer_token.clone();
        let ips_list = allowed_ips.clone();
        let token_exec = bearer_token.clone();
        let ips_exec = allowed_ips.clone();
        let token_res_list = bearer_token.clone();
        let ips_res_list = allowed_ips.clone();
        let token_res_read = bearer_token.clone();
        let ips_res_read = allowed_ips.clone();

        let eh_list = config.event_handler.clone();
        let eh_exec = config.event_handler.clone();

        let protocol_list = protocol.clone();
        let protocol_exec = protocol.clone();
        let protocol_res_list = protocol.clone();
        let protocol_res_read = protocol.clone();

        // Build router with endpoints
        let app = Router::new()
            .route(
                "/tools/list",
                post(move |ConnectInfo(addr): ConnectInfo<SocketAddr>| {
                    let token = token_list.clone();
                    let allowed = ips_list.clone();
                    let proto = protocol_list.clone();
                    let eh = eh_list.clone();
                    async move {
                        // Check IP filtering
                        if !allowed.is_empty() && !allowed.contains(&addr.ip()) {
                            if let Some(ref handler) = eh {
                                handler
                                    .on_mcp_event(&McpEvent::RequestRejected {
                                        client_addr: addr.ip().to_string(),
                                        reason: "IP not allowed".to_string(),
                                    })
                                    .await;
                            }
                            return (
                                StatusCode::FORBIDDEN,
                                Json(json!({"error": "Access denied"})),
                            )
                                .into_response();
                        }

                        // Token validation placeholder
                        if let Some(_expected_token) = token.as_ref() {
                            // TODO: Validate Authorization header here
                        }

                        // Fire ToolListRequested
                        if let Some(ref handler) = eh {
                            handler
                                .on_mcp_event(&McpEvent::ToolListRequested {
                                    client_addr: addr.ip().to_string(),
                                })
                                .await;
                        }

                        match proto.list_tools().await {
                            Ok(tools) => {
                                let tool_count = tools.len();
                                if let Some(ref handler) = eh {
                                    handler
                                        .on_mcp_event(&McpEvent::ToolListReturned {
                                            client_addr: addr.ip().to_string(),
                                            tool_count,
                                        })
                                        .await;
                                }
                                (StatusCode::OK, Json(json!({"tools": tools}))).into_response()
                            }
                            Err(e) => (
                                StatusCode::INTERNAL_SERVER_ERROR,
                                Json(json!({"error": e.to_string()})),
                            )
                                .into_response(),
                        }
                    }
                }),
            )
            .route(
                "/tools/execute",
                post(
                    move |ConnectInfo(addr): ConnectInfo<SocketAddr>,
                          Json(payload): Json<serde_json::Value>| {
                        let token = token_exec.clone();
                        let allowed = ips_exec.clone();
                        let proto = protocol_exec.clone();
                        let eh = eh_exec.clone();
                        async move {
                            // Check IP filtering
                            if !allowed.is_empty() && !allowed.contains(&addr.ip()) {
                                if let Some(ref handler) = eh {
                                    handler
                                        .on_mcp_event(&McpEvent::RequestRejected {
                                            client_addr: addr.ip().to_string(),
                                            reason: "IP not allowed".to_string(),
                                        })
                                        .await;
                                }
                                return (
                                    StatusCode::FORBIDDEN,
                                    Json(json!({"error": "Access denied"})),
                                )
                                    .into_response();
                            }

                            // Token validation placeholder
                            if let Some(_expected_token) = token.as_ref() {
                                // TODO: Validate Authorization header here
                            }

                            let tool_name = payload["tool"].as_str().unwrap_or("").to_string();
                            let params = payload["parameters"].clone();

                            // Fire ToolCallReceived
                            if let Some(ref handler) = eh {
                                handler
                                    .on_mcp_event(&McpEvent::ToolCallReceived {
                                        client_addr: addr.ip().to_string(),
                                        tool_name: tool_name.clone(),
                                        parameters: params.clone(),
                                    })
                                    .await;
                            }

                            let exec_start = std::time::Instant::now();
                            match proto.execute(&tool_name, params).await {
                                Ok(result) => {
                                    let duration_ms = exec_start.elapsed().as_millis() as u64;
                                    let success = result.success;
                                    let error = result.error.clone();
                                    if let Some(ref handler) = eh {
                                        handler
                                            .on_mcp_event(&McpEvent::ToolCallCompleted {
                                                client_addr: addr.ip().to_string(),
                                                tool_name: tool_name.clone(),
                                                success,
                                                error,
                                                duration_ms,
                                            })
                                            .await;
                                    }
                                    (StatusCode::OK, Json(json!({"result": result})))
                                        .into_response()
                                }
                                Err(e) => {
                                    let duration_ms = exec_start.elapsed().as_millis() as u64;
                                    let err_msg = e.to_string();
                                    if let Some(ref handler) = eh {
                                        handler
                                            .on_mcp_event(&McpEvent::ToolError {
                                                source: addr.ip().to_string(),
                                                tool_name: tool_name.clone(),
                                                error: err_msg.clone(),
                                                duration_ms,
                                            })
                                            .await;
                                    }
                                    (
                                        StatusCode::BAD_REQUEST,
                                        Json(json!({"error": err_msg})),
                                    )
                                        .into_response()
                                }
                            }
                        }
                    },
                ),
            )
            .route(
                "/resources/list",
                post(move |ConnectInfo(addr): ConnectInfo<SocketAddr>| {
                    let token = token_res_list.clone();
                    let allowed = ips_res_list.clone();
                    let proto = protocol_res_list.clone();
                    async move {
                        // Check IP filtering
                        if !allowed.is_empty() && !allowed.contains(&addr.ip()) {
                            return (
                                StatusCode::FORBIDDEN,
                                Json(json!({"error": "Access denied"})),
                            )
                                .into_response();
                        }

                        // Token validation placeholder
                        if let Some(_expected_token) = token.as_ref() {
                            // TODO: Validate Authorization header here
                        }

                        if !proto.supports_resources() {
                            return (
                                StatusCode::NOT_IMPLEMENTED,
                                Json(json!({"error": "Resources not supported"})),
                            )
                                .into_response();
                        }

                        match proto.list_resources().await {
                            Ok(resources) => {
                                (StatusCode::OK, Json(json!({"resources": resources})))
                                    .into_response()
                            }
                            Err(e) => (
                                StatusCode::INTERNAL_SERVER_ERROR,
                                Json(json!({"error": e.to_string()})),
                            )
                                .into_response(),
                        }
                    }
                }),
            )
            .route(
                "/resources/read",
                post(
                    move |ConnectInfo(addr): ConnectInfo<SocketAddr>,
                          Json(payload): Json<serde_json::Value>| {
                        let token = token_res_read.clone();
                        let allowed = ips_res_read.clone();
                        let proto = protocol_res_read.clone();
                        async move {
                            // Check IP filtering
                            if !allowed.is_empty() && !allowed.contains(&addr.ip()) {
                                return (
                                    StatusCode::FORBIDDEN,
                                    Json(json!({"error": "Access denied"})),
                                )
                                    .into_response();
                            }

                            // Token validation placeholder
                            if let Some(_expected_token) = token.as_ref() {
                                // TODO: Validate Authorization header here
                            }

                            if !proto.supports_resources() {
                                return (
                                    StatusCode::NOT_IMPLEMENTED,
                                    Json(json!({"error": "Resources not supported"})),
                                )
                                    .into_response();
                            }

                            let uri = payload["uri"].as_str().unwrap_or("");

                            match proto.read_resource(uri).await {
                                Ok(content) => (
                                    StatusCode::OK,
                                    Json(json!({"uri": uri, "content": content})),
                                )
                                    .into_response(),
                                Err(e) => {
                                    (StatusCode::NOT_FOUND, Json(json!({"error": e.to_string()})))
                                        .into_response()
                                }
                            }
                        }
                    },
                ),
            )
            .into_make_service_with_connect_info::<SocketAddr>();

        // Bind and start server
        let listener = TcpListener::bind(config.addr).await?;
        let addr = listener.local_addr()?;

        // Fire ServerStarted event
        if let Some(ref handler) = config.event_handler {
            handler
                .on_mcp_event(&McpEvent::ServerStarted {
                    addr: addr.to_string(),
                })
                .await;
        }

        let server_handle = tokio::spawn(async move { axum::serve(listener, app).await });

        Ok(HttpServerInstance::new(addr, Box::new(server_handle)))
    }

    fn name(&self) -> &str {
        "axum"
    }
}
