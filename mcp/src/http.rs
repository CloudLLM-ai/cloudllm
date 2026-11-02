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

/// Standard error body returned when a request is rejected because of a missing
/// or invalid bearer token.
///
/// Returned in two shapes:
/// - `Legacy` — flat `{ "error": "Unauthorized", "message": "..." }` for the
///   `/tools/list`, `/tools/execute`, `/resources/list`, `/resources/read`
///   legacy endpoints so they remain a superset of the previous wire format.
/// - `Rpc` — the JSON-RPC 2.0 error object used by streamable-HTTP transports.
pub const BEARER_TOKEN_REQUIRED_MESSAGE: &str =
    "This MCP endpoint requires a Bearer token in the `Authorization: Bearer <token>` header. \
     Supply a valid token, or ask the server operator for one.";

/// Build a flat JSON body returned for bearer-token rejections on legacy
/// (`/tools/list`, `/tools/execute`, `/resources/list`, `/resources/read`)
/// endpoints.
#[cfg(feature = "server")]
fn bearer_token_required_body() -> serde_json::Value {
    serde_json::json!({
        "error": "Unauthorized",
        "message": BEARER_TOKEN_REQUIRED_MESSAGE,
        "error_description": BEARER_TOKEN_REQUIRED_MESSAGE,
        "hint": "Send `Authorization: Bearer <token>` with a token issued by the server operator.",
    })
}

/// Browser-facing request checks shared by both HTTP transports.
///
/// Neither check affects non-browser clients (they send no `Origin`, and a
/// loopback bind is only reachable from the local machine anyway); both exist
/// to stop a web page in the operator's browser from driving the server:
///
/// - **Origin**: a present `Origin` header must be a loopback origin unless
///   [`with_skip_origin_validation`](Self::with_skip_origin_validation) is set.
/// - **Host** (DNS-rebinding defence): when the server is bound to a loopback
///   address the `Host` header must name `localhost`, `127.0.0.1` or `[::1]`
///   (any port), unless an explicit allow-list is configured with
///   [`with_allowed_hosts`](Self::with_allowed_hosts), which then replaces the
///   default. A non-loopback bind with no allow-list performs no Host check.
///   Put your public host name(s) in the list when running behind a reverse
///   proxy on a loopback bind.
#[derive(Debug, Clone, Default)]
pub struct RequestGuard {
    allowed_hosts: Vec<String>,
    skip_origin_validation: bool,
}

impl RequestGuard {
    /// Default guard: loopback Origin check, loopback Host default on loopback binds.
    pub fn new() -> Self {
        Self::default()
    }

    /// Accept only these `Host` values (host names or IPs, port ignored,
    /// case-insensitive, IPv6 as `[::1]`). An empty list restores the default.
    pub fn with_allowed_hosts<I, S>(mut self, hosts: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        self.allowed_hosts = hosts
            .into_iter()
            .filter_map(|h| normalize_host(h.as_ref()))
            .collect();
        self
    }

    /// Skip the `Origin` check (for LAN deployments whose browser clients live
    /// on other origins).
    pub fn with_skip_origin_validation(mut self, skip: bool) -> Self {
        self.skip_origin_validation = skip;
        self
    }

    /// True when a request with this `Host` header may proceed.
    ///
    /// A missing header is accepted (HTTP/1.0 clients; browsers always send it).
    #[cfg(feature = "server")]
    pub(crate) fn host_allowed(&self, bind: SocketAddr, headers: &axum::http::HeaderMap) -> bool {
        let Some(raw) = headers.get("host") else {
            return true;
        };
        let Some(host) = raw.to_str().ok().and_then(normalize_host) else {
            return false;
        };
        if !self.allowed_hosts.is_empty() {
            return self.allowed_hosts.contains(&host);
        }
        !bind.ip().is_loopback() || matches!(host.as_str(), "localhost" | "127.0.0.1" | "[::1]")
    }

    /// True when a request with this `Origin` header may proceed.
    #[cfg(feature = "server")]
    pub(crate) fn origin_allowed(&self, headers: &axum::http::HeaderMap) -> bool {
        if self.skip_origin_validation {
            return true;
        }
        match headers.get("origin") {
            None => true,
            Some(v) => v.to_str().is_ok_and(origin_is_loopback),
        }
    }
}

/// Lower-case the host of a `host[:port]` authority (IPv6 keeps its brackets).
/// Returns `None` for anything that is not a plain authority.
fn normalize_host(authority: &str) -> Option<String> {
    let authority = authority.trim();
    if authority.is_empty() || authority.contains(['/', '@', '?', '#', ' ']) {
        return None;
    }
    let (host, port) = if authority.starts_with('[') {
        let end = authority.find(']')?;
        let tail = &authority[end + 1..];
        let port = match tail {
            "" => None,
            t => Some(t.strip_prefix(':')?),
        };
        (&authority[..=end], port)
    } else {
        match authority.split_once(':') {
            Some((h, p)) => (h, Some(p)),
            None => (authority, None),
        }
    };
    if host.is_empty() || host == "[]" {
        return None;
    }
    if let Some(p) = port {
        if p.is_empty() || !p.bytes().all(|b| b.is_ascii_digit()) || p.parse::<u16>().is_err() {
            return None;
        }
    }
    Some(host.to_ascii_lowercase())
}

/// True when `origin` is an `http(s)` origin whose host is exactly
/// `localhost`, `127.0.0.1` or `[::1]`, with an optional numeric port.
///
/// A browser `Origin` is `scheme "://" host [":" port]` with no userinfo,
/// path, query or fragment, so anything else is rejected. Matching the host
/// exactly (not as a string prefix) keeps `http://localhost.evil.com` and
/// `http://127.0.0.1.evil.com` out.
pub(crate) fn origin_is_loopback(origin: &str) -> bool {
    let Some(authority) = origin
        .strip_prefix("http://")
        .or_else(|| origin.strip_prefix("https://"))
    else {
        return false;
    };
    matches!(
        normalize_host(authority).as_deref(),
        Some("localhost" | "127.0.0.1" | "[::1]")
    )
}

/// Generate a short opaque id to correlate a generic 5xx response with the
/// server-side log line that carries the real error.
pub(crate) fn new_request_id() -> String {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0);
    format!(
        "req-{:x}-{:x}",
        nanos,
        COUNTER.fetch_add(1, Ordering::Relaxed)
    )
}

/// Log the real error server-side and return the generic body text for a 5xx.
pub(crate) fn internal_error_message(context: &str, error: &dyn std::fmt::Display) -> String {
    let id = new_request_id();
    log::error!("[cloudllm_mcp] internal error ({context}) request_id={id}: {error}");
    format!("Internal server error (request id {id})")
}

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

    /// Start the server with an explicit Origin / Host policy.
    ///
    /// The default implementation ignores `guard` and calls
    /// [`start`](Self::start), so existing adapters keep compiling; override it
    /// to honour the policy (the built-in Axum adapter does).
    async fn start_with_guard(
        &self,
        config: HttpServerConfig,
        protocol: Arc<dyn ToolProtocol>,
        _guard: RequestGuard,
    ) -> Result<HttpServerInstance, Box<dyn Error + Send + Sync>> {
        self.start(config, protocol).await
    }

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
    axum_router_with_guard(config, protocol, &RequestGuard::new())
}

/// Like [`axum_router`], with explicit Origin / Host policy (see [`RequestGuard`]).
///
/// Every route first applies the IP filter, then the Host and Origin checks,
/// then authentication.
#[cfg(feature = "server")]
pub fn axum_router_with_guard(
    config: &HttpServerConfig,
    protocol: Arc<dyn ToolProtocol>,
    guard: &RequestGuard,
) -> Router {
    use crate::builder_utils::static_token_matches;
    use crate::events::McpEvent;
    use axum::{
        extract::ConnectInfo, http::HeaderMap, http::StatusCode, response::IntoResponse,
        routing::post, Json, Router,
    };
    use serde_json::json;

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
            if static_token_matches(expected, provided) {
                return true;
            }
        }

        authorizer
            .as_ref()
            .is_some_and(|auth| auth.authorize_bearer_token(provided, &context))
    }

    /// Host / Origin rejection, shared by all four routes.
    fn guard_rejection(
        guard: &RequestGuard,
        bind: SocketAddr,
        headers: &HeaderMap,
    ) -> Option<axum::response::Response> {
        let message = if !guard.host_allowed(bind, headers) {
            "Forbidden host"
        } else if !guard.origin_allowed(headers) {
            "Forbidden origin"
        } else {
            return None;
        };
        Some((StatusCode::FORBIDDEN, Json(json!({"error": message}))).into_response())
    }

    let guard = Arc::new(guard.clone());
    let bind_addr = config.addr;
    let guard_list = guard.clone();
    let guard_exec = guard.clone();
    let guard_res_list = guard.clone();
    let guard_res_read = guard.clone();

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
                    let guard = guard_list.clone();
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

                        if let Some(rejection) = guard_rejection(&guard, bind_addr, &headers) {
                            return rejection;
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
                            return (StatusCode::UNAUTHORIZED, Json(bearer_token_required_body()))
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
                                Json(json!({"error": internal_error_message("tools/list", &e)})),
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
                    let guard = guard_exec.clone();
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

                        if let Some(rejection) = guard_rejection(&guard, bind_addr, &headers) {
                            return rejection;
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
                            return (StatusCode::UNAUTHORIZED, Json(bearer_token_required_body()))
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
                    let guard = guard_res_list.clone();
                    let proto = protocol_res_list.clone();
                    async move {
                        if !allowed.is_allowed(addr.ip()) {
                            return (
                                StatusCode::FORBIDDEN,
                                Json(json!({"error": "Access denied"})),
                            )
                                .into_response();
                        }

                        if let Some(rejection) = guard_rejection(&guard, bind_addr, &headers) {
                            return rejection;
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
                            return (StatusCode::UNAUTHORIZED, Json(bearer_token_required_body()))
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
                                Json(
                                    json!({"error": internal_error_message("resources/list", &e)}),
                                ),
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
                    let guard = guard_res_read.clone();
                    let proto = protocol_res_read.clone();
                    async move {
                        if !allowed.is_allowed(addr.ip()) {
                            return (
                                StatusCode::FORBIDDEN,
                                Json(json!({"error": "Access denied"})),
                            )
                                .into_response();
                        }

                        if let Some(rejection) = guard_rejection(&guard, bind_addr, &headers) {
                            return rejection;
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
                            return (StatusCode::UNAUTHORIZED, Json(bearer_token_required_body()))
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

/// Print a prominent startup warning when no authentication is configured.
///
/// An open server is a supported operator choice (the streamable transport
/// behaves the same way), but it must never be an accident: without a bearer
/// token or authorizer every client that can reach the socket may run tools.
pub fn warn_if_unauthenticated(config: &HttpServerConfig) {
    if config.bearer_token.is_some() || config.bearer_authorizer.is_some() {
        return;
    }
    let exposure = if config.addr.ip().is_loopback() {
        "reachable by every local process and, via a browser, by web pages \
         (Origin/Host checks apply)"
    } else {
        "reachable by every host that can connect to this address"
    };
    log::warn!(
        "[cloudllm_mcp] MCP server on {} has NO authentication configured; \
         tools are {}. Configure a bearer token or authorizer.",
        config.addr,
        exposure
    );
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
        self.start_with_guard(config, protocol, RequestGuard::new())
            .await
    }

    async fn start_with_guard(
        &self,
        config: HttpServerConfig,
        protocol: Arc<dyn ToolProtocol>,
        guard: RequestGuard,
    ) -> Result<HttpServerInstance, Box<dyn Error + Send + Sync>> {
        use crate::events::McpEvent;
        use tokio::net::TcpListener;
        warn_if_unauthenticated(&config);
        let app = axum_router_with_guard(&config, protocol, &guard)
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

#[cfg(test)]
mod log_tests {
    use super::internal_error_message;
    use std::sync::{Mutex, OnceLock};

    static RECORDS: OnceLock<Mutex<Vec<(log::Level, String)>>> = OnceLock::new();

    struct Capture;

    impl log::Log for Capture {
        fn enabled(&self, _: &log::Metadata) -> bool {
            true
        }
        fn log(&self, record: &log::Record) {
            if let Some(records) = RECORDS.get() {
                records
                    .lock()
                    .expect("capture lock")
                    .push((record.level(), record.args().to_string()));
            }
        }
        fn flush(&self) {}
    }

    /// The 5xx path must report the real error through the `log` facade at
    /// error level (not stderr) while the client-facing text carries only the
    /// request id.
    #[test]
    fn test_internal_error_is_logged_not_printed_and_not_leaked() {
        RECORDS.get_or_init(Default::default);
        log::set_logger(&Capture).expect("install capture logger");
        log::set_max_level(log::LevelFilter::Trace);

        let body = internal_error_message("test-ctx", &"secret backend failure");

        assert!(body.starts_with("Internal server error (request id req-"));
        assert!(!body.contains("secret backend failure"));
        let records = RECORDS.get().expect("records").lock().expect("lock");
        let (level, text) = records.last().expect("one record");
        assert_eq!(*level, log::Level::Error);
        assert!(text.contains("test-ctx") && text.contains("secret backend failure"));
        assert!(text.contains("request_id=req-"));
    }
}
