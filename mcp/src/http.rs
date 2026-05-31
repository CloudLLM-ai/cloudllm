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

use crate::builder_utils::IpFilter;
use crate::events::McpEventHandler;
use crate::protocol::ToolProtocol;
use std::error::Error;
use std::net::SocketAddr;
use std::sync::Arc;

#[cfg(feature = "server")]
use axum::Router;

/// Configuration for an HTTP MCP server
pub struct HttpServerConfig {
    /// Socket address to bind to (e.g., "127.0.0.1:8080")
    pub addr: SocketAddr,
    /// Optional bearer token for authentication
    pub bearer_token: Option<String>,
    /// Optional dynamic bearer-token authorizer.
    ///
    /// When present, this authorizer is consulted after the static
    /// [`bearer_token`](Self::bearer_token) check. A request is accepted when
    /// either configured mechanism accepts the supplied bearer token.
    pub bearer_authorizer: Option<Arc<dyn BearerTokenAuthorizer>>,
    /// IP filter controlling which client addresses are allowed
    pub ip_filter: IpFilter,
    /// Optional event handler for MCP server lifecycle and request events
    pub event_handler: Option<Arc<dyn McpEventHandler>>,
}

impl Clone for HttpServerConfig {
    fn clone(&self) -> Self {
        Self {
            addr: self.addr,
            bearer_token: self.bearer_token.clone(),
            bearer_authorizer: self.bearer_authorizer.clone(),
            ip_filter: self.ip_filter.clone(),
            event_handler: self.event_handler.clone(),
        }
    }
}

impl std::fmt::Debug for HttpServerConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HttpServerConfig")
            .field("addr", &self.addr)
            .field("has_bearer_token", &self.bearer_token.is_some())
            .field("has_bearer_authorizer", &self.bearer_authorizer.is_some())
            .field("ip_filter", &self.ip_filter)
            .field("has_event_handler", &self.event_handler.is_some())
            .finish()
    }
}

/// Per-request context passed to a dynamic bearer-token authorizer.
///
/// The MCP runtime builds this value before dispatching a request to the
/// protocol implementation. Servers can inspect it to make authorization
/// decisions that depend on more than the raw token string.
///
/// # Payload shape
///
/// `payload` is deliberately transport-shaped:
///
/// - streamable HTTP JSON-RPC requests receive the `params` object, such as
///   `{ "name": "tool_name", "arguments": { ... } }` for `tools/call`.
/// - legacy `/tools/execute` requests receive the full request body, usually
///   `{ "tool": "tool_name", "parameters": { ... } }`.
/// - metadata-style routes such as `/tools/list` and `/resources/list` receive
///   `None` when there is no useful body to authorize.
///
/// Server crates keep ownership of policy. For example, a memory server can
/// inspect tool arguments inside `payload` and allow a token for one chain while
/// denying the same token for another.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BearerAuthContext {
    /// Client socket address reported by the HTTP framework.
    pub client_addr: SocketAddr,
    /// HTTP route that received the request.
    pub route: String,
    /// MCP method or legacy action being authorized.
    pub action: String,
    /// Parsed request payload or JSON-RPC params, when available.
    ///
    /// This value is cloned from the already-parsed request body so authorizers
    /// do not need to parse JSON a second time.
    pub payload: Option<serde_json::Value>,
}

/// Dynamic bearer-token authorization hook for MCP HTTP transports.
///
/// Implement this trait when a server needs revocable tokens, scoped access,
/// or token lookup from durable storage instead of a single static configured
/// secret.
///
/// # Examples
///
/// ```
/// use mcp::{BearerAuthContext, BearerTokenAuthorizer};
///
/// struct ToolListOnly;
///
/// impl BearerTokenAuthorizer for ToolListOnly {
///     fn authorize_bearer_token(&self, token: &str, context: &BearerAuthContext) -> bool {
///         token == "good-token" && context.action == "tools/list"
///     }
/// }
/// ```
pub trait BearerTokenAuthorizer: Send + Sync {
    /// Return `true` when a request without a bearer token should be allowed.
    ///
    /// The default is fail-closed. Override this only when the embedding server
    /// has an explicit runtime mode where unauthenticated requests are expected.
    fn allow_missing_bearer_token(&self, _context: &BearerAuthContext) -> bool {
        false
    }

    /// Return `true` when `token` is allowed for `context`.
    ///
    /// Implementations should avoid logging raw token values. If comparing
    /// against stored secrets, prefer constant-time hash comparison in the
    /// server crate.
    fn authorize_bearer_token(&self, token: &str, context: &BearerAuthContext) -> bool;
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

/// Build an Axum router that exposes a [`ToolProtocol`] over the shared HTTP MCP surface.
///
/// The returned router serves:
/// - `POST /tools/list`
/// - `POST /tools/execute`
/// - `POST /resources/list`
/// - `POST /resources/read`
///
/// This helper is useful when a crate wants to reuse the shared MCP transport
/// but still compose extra routes of its own, such as a `/health` endpoint.
#[cfg(feature = "server")]
pub fn axum_router(config: &HttpServerConfig, protocol: Arc<dyn ToolProtocol>) -> Router {
    use crate::events::McpEvent;
    use axum::{
        extract::ConnectInfo, http::HeaderMap, http::StatusCode, response::IntoResponse,
        routing::post, Json, Router,
    };
    use serde_json::json;
    use sha2::{Digest, Sha256};
    use subtle::ConstantTimeEq;

    fn bearer_from_headers(headers: &HeaderMap) -> Option<&str> {
        headers
            .get("Authorization")
            .and_then(|v| v.to_str().ok())
            .and_then(|v| v.strip_prefix("Bearer "))
    }

    /// Validate the Authorization header against static or dynamic bearer auth.
    ///
    /// Returns `true` when neither auth mechanism is configured (open server),
    /// when the supplied `Bearer <token>` matches the static token, or when the
    /// dynamic authorizer accepts it. Static comparison uses
    /// `subtle::ConstantTimeEq` on SHA-256 digests so the compiler cannot
    /// short-circuit the comparison and leak token length via timing.
    fn check_auth(
        expected_token: &Option<String>,
        authorizer: &Option<Arc<dyn BearerTokenAuthorizer>>,
        headers: &HeaderMap,
        context: BearerAuthContext,
    ) -> bool {
        if expected_token.is_none() && authorizer.is_none() {
            return true;
        }

        let Some(provided) = bearer_from_headers(headers) else {
            return authorizer
                .as_ref()
                .is_some_and(|auth| auth.allow_missing_bearer_token(&context));
        };

        if let Some(expected) = expected_token.as_deref() {
            let expected_hash = Sha256::digest(expected.as_bytes());
            let provided_hash = Sha256::digest(provided.as_bytes());
            if bool::from(expected_hash.ct_eq(&provided_hash)) {
                return true;
            }
        }

        authorizer
            .as_ref()
            .is_some_and(|auth| auth.authorize_bearer_token(provided, &context))
    }

    let bearer_token = Arc::new(config.bearer_token.clone());
    let bearer_authorizer = Arc::new(config.bearer_authorizer.clone());
    let ip_filter = Arc::new(config.ip_filter.clone());

    let token_list = bearer_token.clone();
    let authz_list = bearer_authorizer.clone();
    let ips_list = ip_filter.clone();
    let token_exec = bearer_token.clone();
    let authz_exec = bearer_authorizer.clone();
    let ips_exec = ip_filter.clone();
    let token_res_list = bearer_token.clone();
    let authz_res_list = bearer_authorizer.clone();
    let ips_res_list = ip_filter.clone();
    let token_res_read = bearer_token.clone();
    let authz_res_read = bearer_authorizer.clone();
    let ips_res_read = ip_filter.clone();

    let eh_list = config.event_handler.clone();
    let eh_exec = config.event_handler.clone();

    let protocol_list = protocol.clone();
    let protocol_exec = protocol.clone();
    let protocol_res_list = protocol.clone();
    let protocol_res_read = protocol.clone();

    Router::new()
        .route(
            "/tools/list",
            post(
                move |ConnectInfo(addr): ConnectInfo<SocketAddr>, headers: HeaderMap| {
                    let token = token_list.clone();
                    let authz = authz_list.clone();
                    let allowed = ips_list.clone();
                    let proto = protocol_list.clone();
                    let eh = eh_list.clone();
                    async move {
                        if !allowed.is_allowed(addr.ip()) {
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

                        if !check_auth(
                            &token,
                            &authz,
                            &headers,
                            BearerAuthContext {
                                client_addr: addr,
                                route: "/tools/list".to_string(),
                                action: "tools/list".to_string(),
                                payload: None,
                            },
                        ) {
                            return (
                                StatusCode::UNAUTHORIZED,
                                Json(json!({"error": "Unauthorized"})),
                            )
                                .into_response();
                        }

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
                },
            ),
        )
        .route(
            "/tools/execute",
            post(
                move |ConnectInfo(addr): ConnectInfo<SocketAddr>,
                      headers: HeaderMap,
                      Json(payload): Json<serde_json::Value>| {
                    let token = token_exec.clone();
                    let authz = authz_exec.clone();
                    let allowed = ips_exec.clone();
                    let proto = protocol_exec.clone();
                    let eh = eh_exec.clone();
                    async move {
                        if !allowed.is_allowed(addr.ip()) {
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

                        if !check_auth(
                            &token,
                            &authz,
                            &headers,
                            BearerAuthContext {
                                client_addr: addr,
                                route: "/tools/execute".to_string(),
                                action: "tools/execute".to_string(),
                                payload: Some(payload.clone()),
                            },
                        ) {
                            return (
                                StatusCode::UNAUTHORIZED,
                                Json(json!({"error": "Unauthorized"})),
                            )
                                .into_response();
                        }

                        let tool_name = payload["tool"].as_str().unwrap_or("").to_string();
                        let params = payload["parameters"].clone();

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
                                (StatusCode::OK, Json(json!({"result": result}))).into_response()
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
                                (StatusCode::BAD_REQUEST, Json(json!({"error": err_msg})))
                                    .into_response()
                            }
                        }
                    }
                },
            ),
        )
        .route(
            "/resources/list",
            post(
                move |ConnectInfo(addr): ConnectInfo<SocketAddr>, headers: HeaderMap| {
                    let token = token_res_list.clone();
                    let authz = authz_res_list.clone();
                    let allowed = ips_res_list.clone();
                    let proto = protocol_res_list.clone();
                    async move {
                        if !allowed.is_allowed(addr.ip()) {
                            return (
                                StatusCode::FORBIDDEN,
                                Json(json!({"error": "Access denied"})),
                            )
                                .into_response();
                        }

                        if !check_auth(
                            &token,
                            &authz,
                            &headers,
                            BearerAuthContext {
                                client_addr: addr,
                                route: "/resources/list".to_string(),
                                action: "resources/list".to_string(),
                                payload: None,
                            },
                        ) {
                            return (
                                StatusCode::UNAUTHORIZED,
                                Json(json!({"error": "Unauthorized"})),
                            )
                                .into_response();
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
                },
            ),
        )
        .route(
            "/resources/read",
            post(
                move |ConnectInfo(addr): ConnectInfo<SocketAddr>,
                      headers: HeaderMap,
                      Json(payload): Json<serde_json::Value>| {
                    let token = token_res_read.clone();
                    let authz = authz_res_read.clone();
                    let allowed = ips_res_read.clone();
                    let proto = protocol_res_read.clone();
                    async move {
                        if !allowed.is_allowed(addr.ip()) {
                            return (
                                StatusCode::FORBIDDEN,
                                Json(json!({"error": "Access denied"})),
                            )
                                .into_response();
                        }

                        if !check_auth(
                            &token,
                            &authz,
                            &headers,
                            BearerAuthContext {
                                client_addr: addr,
                                route: "/resources/read".to_string(),
                                action: "resources/read".to_string(),
                                payload: Some(payload.clone()),
                            },
                        ) {
                            return (
                                StatusCode::UNAUTHORIZED,
                                Json(json!({"error": "Unauthorized"})),
                            )
                                .into_response();
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
}

/// Default Axum-based HTTP server adapter
///
/// Provides a full MCP-compatible HTTP server using the Axum framework.
/// Only available when the `server` feature is enabled.
#[cfg(feature = "server")]
pub struct AxumHttpAdapter;

#[cfg(feature = "server")]
#[async_trait::async_trait]
impl HttpServerAdapter for AxumHttpAdapter {
    async fn start(
        &self,
        config: HttpServerConfig,
        protocol: Arc<dyn ToolProtocol>,
    ) -> Result<HttpServerInstance, Box<dyn Error + Send + Sync>> {
        use crate::events::McpEvent;
        use tokio::net::TcpListener;
        let app =
            axum_router(&config, protocol).into_make_service_with_connect_info::<SocketAddr>();

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
