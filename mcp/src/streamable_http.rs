//! Streamable HTTP transport for standard MCP clients.
//!
//! This module implements the modern MCP transport over a single HTTP endpoint
//! that accepts JSON-RPC requests via `POST`. It returns either a single JSON
//! response (`application/json`) or, in more advanced servers, an SSE stream.
//! This implementation currently uses single-response JSON for compatibility
//! with standard MCP clients such as Codex and Claude Code.

use crate::builder_utils::IpFilter;
use crate::events::{McpEvent, McpEventHandler};
use crate::protocol::{ToolError, ToolProtocol};
use axum::extract::{ConnectInfo, Path, State};
use axum::http::{HeaderMap, HeaderValue, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use futures::stream::Stream;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::error::Error;
use std::net::SocketAddr;
use std::pin::Pin;
use std::sync::Arc;
use subtle::ConstantTimeEq;
use tokio::sync::broadcast;
use tokio_stream::StreamExt;

/// Current MCP protocol version supported by the streamable HTTP transport.
pub const CURRENT_MCP_PROTOCOL_VERSION: &str = "2025-11-25";

/// Legacy MCP protocol versions still accepted for compatibility.
pub const SUPPORTED_MCP_PROTOCOL_VERSIONS: &[&str] = &["2025-11-25", "2025-06-18", "2025-03-26", "2024-11-05"];

/// Configuration for a standard streamable HTTP MCP endpoint.
///
/// # Example
///
/// ```rust
/// use mcp::streamable_http::StreamableHttpConfig;
///
/// let config = StreamableHttpConfig::new("mentisdb", "0.1.0");
/// assert_eq!(config.endpoint_path, "/");
/// ```
#[derive(Debug, Clone)]
pub struct StreamableHttpConfig {
    /// MCP endpoint path that accepts `POST`, `GET`, and `DELETE`.
    pub endpoint_path: String,
    /// Protocol version advertised during initialization.
    pub protocol_version: String,
    /// Stable machine-readable server name.
    pub server_name: String,
    /// Optional human-friendly display title.
    pub server_title: Option<String>,
    /// Server version exposed during initialization.
    pub server_version: String,
    /// Optional instructions returned in the initialize result.
    pub instructions: Option<String>,
    /// When true, skip HTTP Origin header validation.
    /// Set this to `true` when deploying on a LAN where non-localhost
    /// origins need to connect (e.g., `http://192.168.1.x`).
    pub skip_origin_validation: bool,
}

impl StreamableHttpConfig {
    /// Build a streamable HTTP config with sane defaults.
    pub fn new(server_name: impl Into<String>, server_version: impl Into<String>) -> Self {
        Self {
            endpoint_path: "/".to_string(),
            protocol_version: CURRENT_MCP_PROTOCOL_VERSION.to_string(),
            server_name: server_name.into(),
            server_title: None,
            server_version: server_version.into(),
            instructions: None,
            skip_origin_validation: false,
        }
    }

    /// Override the endpoint path.
    pub fn with_endpoint_path(mut self, endpoint_path: impl Into<String>) -> Self {
        self.endpoint_path = endpoint_path.into();
        self
    }

    /// Override the reported protocol version.
    pub fn with_protocol_version(mut self, protocol_version: impl Into<String>) -> Self {
        self.protocol_version = protocol_version.into();
        self
    }

    /// Set a human-friendly server title.
    pub fn with_server_title(mut self, server_title: impl Into<String>) -> Self {
        self.server_title = Some(server_title.into());
        self
    }

    /// Set optional instructions returned during initialization.
    pub fn with_instructions(mut self, instructions: impl Into<String>) -> Self {
        self.instructions = Some(instructions.into());
        self
    }

    /// Skip HTTP Origin header validation for LAN deployments.
    pub fn with_skip_origin_validation(mut self, skip: bool) -> Self {
        self.skip_origin_validation = skip;
        self
    }
}

/// An SSE message following the text/event-stream format.
#[derive(Debug, Clone, Serialize)]
pub struct SseMessage {
    /// Optional event type (defaults to "message" for MCP).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub event: Option<String>,
    /// Optional unique message ID for replay/resume.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    /// The JSON-RPC payload or notification.
    pub data: Value,
}

impl SseMessage {
    /// Create a simple data-only SSE message (no event type or ID).
    pub fn data(data: Value) -> Self {
        Self {
            event: None,
            id: None,
            data,
        }
    }

    /// Create a typed SSE message with an event name.
    pub fn with_event(event: impl Into<String>, data: Value) -> Self {
        Self {
            event: Some(event.into()),
            id: None,
            data,
        }
    }

    /// Format as a proper text/event-stream line sequence.
    pub fn format(&self) -> String {
        let mut out = String::new();
        if let Some(ref event) = self.event {
            out.push_str(&format!("event: {}\n", event));
        }
        if let Some(ref id) = self.id {
            out.push_str(&format!("id: {}\n", id));
        }
        out.push_str(&format!("data: {}\n\n", serde_json::to_string(&self.data).unwrap_or_default()));
        out
    }
}

/// Broadcast channel wrapping McpEvent for SSE streaming.
#[derive(Clone)]
pub struct SseBroadcaster {
    sender: broadcast::Sender<SseMessage>,
}

impl SseBroadcaster {
    /// Create a new broadcaster with the given buffer size.
    pub fn new(buffer_size: usize) -> Self {
        let (sender, _) = broadcast::channel(buffer_size);
        Self { sender }
    }

    /// Subscribe to the event stream.
    pub fn subscribe(&self) -> broadcast::Receiver<SseMessage> {
        self.sender.subscribe()
    }

    /// Send an event to all subscribers.
    pub fn send(&self, message: SseMessage) {
        let _ = self.sender.send(message);
    }

    /// Convert an McpEvent into an SseMessage and broadcast it.
    pub fn broadcast_mcp_event(&self, event: &McpEvent) {
        let data = match event {
            McpEvent::ServerStarted { addr } => json!({
                "event": "server_started",
                "addr": addr
            }),
            McpEvent::ToolListRequested { client_addr } => json!({
                "event": "tool_list_requested",
                "client_addr": client_addr
            }),
            McpEvent::ToolListReturned { client_addr, tool_count } => json!({
                "event": "tool_list_returned",
                "client_addr": client_addr,
                "tool_count": tool_count
            }),
            McpEvent::ToolCallReceived { client_addr, tool_name, parameters } => json!({
                "event": "tool_call_received",
                "client_addr": client_addr,
                "tool_name": tool_name,
                "parameters": parameters
            }),
            McpEvent::ToolCallCompleted { client_addr, tool_name, success, error, duration_ms } => json!({
                "event": "tool_call_completed",
                "client_addr": client_addr,
                "tool_name": tool_name,
                "success": success,
                "error": error,
                "duration_ms": duration_ms
            }),
            McpEvent::ToolError { source, tool_name, error, duration_ms } => json!({
                "event": "tool_error",
                "source": source,
                "tool_name": tool_name,
                "error": error,
                "duration_ms": duration_ms
            }),
            McpEvent::RequestRejected { client_addr, reason } => json!({
                "event": "request_rejected",
                "client_addr": client_addr,
                "reason": reason
            }),
            McpEvent::ConnectionInitialized { endpoint, tool_count } => json!({
                "event": "connection_initialized",
                "endpoint": endpoint,
                "tool_count": tool_count
            }),
            McpEvent::ConnectionClosed { endpoint } => json!({
                "event": "connection_closed",
                "endpoint": endpoint
            }),
            McpEvent::ToolsDiscovered { endpoint, tool_count, tool_names } => json!({
                "event": "tools_discovered",
                "endpoint": endpoint,
                "tool_count": tool_count,
                "tool_names": tool_names
            }),
            McpEvent::CacheHit { endpoint, tool_count } => json!({
                "event": "cache_hit",
                "endpoint": endpoint,
                "tool_count": tool_count
            }),
            McpEvent::CacheExpired { endpoint } => json!({
                "event": "cache_expired",
                "endpoint": endpoint
            }),
            McpEvent::RemoteToolCallStarted { endpoint, tool_name, parameters } => json!({
                "event": "remote_tool_call_started",
                "endpoint": endpoint,
                "tool_name": tool_name,
                "parameters": parameters
            }),
            McpEvent::RemoteToolCallCompleted { endpoint, tool_name, success, error, duration_ms } => json!({
                "event": "remote_tool_call_completed",
                "endpoint": endpoint,
                "tool_name": tool_name,
                "success": success,
                "error": error,
                "duration_ms": duration_ms
            }),
        };
        self.send(SseMessage::with_event("mcp_event", data));
    }
}

/// An event handler that bridges to an SseBroadcaster.
#[derive(Clone)]
pub struct SseEventHandler {
    broadcaster: SseBroadcaster,
}

impl SseEventHandler {
    /// Create a new SSE event handler with a broadcaster of the given size.
    pub fn new(buffer_size: usize) -> (Self, SseBroadcaster) {
        let broadcaster = SseBroadcaster::new(buffer_size);
        (Self { broadcaster: broadcaster.clone() }, broadcaster)
    }
}

#[async_trait::async_trait]
impl McpEventHandler for SseEventHandler {
    async fn on_mcp_event(&self, event: &McpEvent) {
        self.broadcaster.broadcast_mcp_event(event);
    }
}

#[derive(Clone)]
struct StreamableHttpState {
    protocol: Arc<dyn ToolProtocol>,
    http_config: StreamableHttpRuntimeConfig,
    transport: StreamableHttpConfig,
    sse_broadcaster: Option<SseBroadcaster>,
}

#[derive(Clone)]
struct StreamableHttpRuntimeConfig {
    bearer_token: Option<String>,
    ip_filter: IpFilter,
    skip_origin_validation: bool,
}

#[derive(Debug, Deserialize)]
struct JsonRpcRequest {
    jsonrpc: String,
    id: Option<Value>,
    method: Option<String>,
    #[serde(default)]
    params: Option<Value>,
}

#[derive(Debug, Serialize)]
struct JsonRpcErrorObject {
    code: i32,
    message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    data: Option<Value>,
}

#[derive(Debug, Serialize)]
struct JsonRpcResponse {
    jsonrpc: &'static str,
    id: Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    result: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<JsonRpcErrorObject>,
}

/// Build a standard streamable HTTP MCP router.
///
/// The router serves a single MCP endpoint that accepts:
/// - `POST` for JSON-RPC requests and notifications
/// - `GET` returning an SSE stream when a broadcaster is configured, otherwise `405`
/// - `DELETE` returning `405 Method Not Allowed` because sessions are stateless
///
/// # Example
///
/// ```rust,no_run
/// use std::sync::Arc;
/// use mcp::http::HttpServerConfig;
/// use mcp::streamable_http::{streamable_http_router, StreamableHttpConfig};
/// use mcp::{IpFilter, ToolMetadata, ToolProtocol, ToolResult};
///
/// struct Demo;
///
/// #[async_trait::async_trait]
/// impl ToolProtocol for Demo {
///     async fn execute(
///         &self,
///         _tool_name: &str,
///         _parameters: serde_json::Value,
///     ) -> Result<ToolResult, Box<dyn std::error::Error + Send + Sync>> {
///         Ok(ToolResult::success(serde_json::json!({"ok": true})))
///     }
///
///     async fn list_tools(
///         &self,
///     ) -> Result<Vec<ToolMetadata>, Box<dyn std::error::Error + Send + Sync>> {
///         Ok(vec![ToolMetadata::new("demo", "Demo tool")])
///     }
///
///     async fn get_tool_metadata(
///         &self,
///         _tool_name: &str,
///     ) -> Result<ToolMetadata, Box<dyn std::error::Error + Send + Sync>> {
///         Ok(ToolMetadata::new("demo", "Demo tool"))
///     }
///
///     fn protocol_name(&self) -> &str {
///         "demo"
///     }
/// }
///
/// let router = streamable_http_router(
///     &HttpServerConfig {
///         addr: std::net::SocketAddr::from(([127, 0, 0, 1], 9471)),
///         bearer_token: None,
///         ip_filter: IpFilter::new(),
///         event_handler: None,
///     },
///     &StreamableHttpConfig::new("demo", "0.1.0"),
///     Arc::new(Demo),
/// );
/// let _ = router;
/// ```
pub fn streamable_http_router(
    http_config: &crate::http::HttpServerConfig,
    transport: &StreamableHttpConfig,
    protocol: Arc<dyn ToolProtocol>,
) -> Router {
    streamable_http_router_with_sse(http_config, transport, protocol, None)
}

/// Build a streamable HTTP MCP router with optional SSE support.
///
/// When `sse_broadcaster` is `Some`, the `GET` endpoint returns a
/// `text/event-stream` response that emits server-side events as they occur.
///
/// # Example
///
/// ```rust,no_run
/// use std::sync::Arc;
/// use mcp::http::HttpServerConfig;
/// use mcp::streamable_http::{streamable_http_router_with_sse, SseBroadcaster, StreamableHttpConfig};
/// use mcp::{IpFilter, ToolMetadata, ToolProtocol, ToolResult};
///
/// struct Demo;
///
/// #[async_trait::async_trait]
/// impl ToolProtocol for Demo {
///     async fn execute(
///         &self,
///         _tool_name: &str,
///         _parameters: serde_json::Value,
///     ) -> Result<ToolResult, Box<dyn std::error::Error + Send + Sync>> {
///         Ok(ToolResult::success(serde_json::json!({"ok": true})))
///     }
///
///     async fn list_tools(
///         &self,
///     ) -> Result<Vec<ToolMetadata>, Box<dyn std::error::Error + Send + Sync>> {
///         Ok(vec![ToolMetadata::new("demo", "Demo tool")])
///     }
///
///     async fn get_tool_metadata(
///         &self,
///         _tool_name: &str,
///     ) -> Result<ToolMetadata, Box<dyn std::error::Error + Send + Sync>> {
///         Ok(ToolMetadata::new("demo", "Demo tool"))
///     }
///
///     fn protocol_name(&self) -> &str {
///         "demo"
///     }
/// }
///
/// let broadcaster = SseBroadcaster::new(256);
/// let router = streamable_http_router_with_sse(
///     &HttpServerConfig {
///         addr: std::net::SocketAddr::from(([127, 0, 0, 1], 9471)),
///         bearer_token: None,
///         ip_filter: IpFilter::new(),
///         event_handler: None,
///     },
///     &StreamableHttpConfig::new("demo", "0.1.0"),
///     Arc::new(Demo),
///     Some(broadcaster),
/// );
/// let _ = router;
/// ```
pub fn streamable_http_router_with_sse(
    http_config: &crate::http::HttpServerConfig,
    transport: &StreamableHttpConfig,
    protocol: Arc<dyn ToolProtocol>,
    sse_broadcaster: Option<SseBroadcaster>,
) -> Router {
    let state = Arc::new(StreamableHttpState {
        protocol,
        http_config: StreamableHttpRuntimeConfig {
            bearer_token: http_config.bearer_token.clone(),
            ip_filter: http_config.ip_filter.clone(),
            skip_origin_validation: transport.skip_origin_validation,
        },
        transport: transport.clone(),
        sse_broadcaster,
    });

    Router::new()
        .route(
            transport.endpoint_path.as_str(),
            post(streamable_http_post_handler)
                .get(streamable_http_get_handler)
                .delete(streamable_http_delete_handler),
        )
        .route(
            "/{*rest}",
            get(method_not_allowed_handler)
                .post(method_not_allowed_handler)
                .delete(method_not_allowed_handler),
        )
        .with_state(state)
}

async fn streamable_http_post_handler(
    State(state): State<Arc<StreamableHttpState>>,
    ConnectInfo(addr): ConnectInfo<SocketAddr>,
    headers: HeaderMap,
    Json(message): Json<JsonRpcRequest>,
) -> Response {
    if !authorize(&state.http_config, &headers, addr.ip()) {
        return json_error_response(
            StatusCode::UNAUTHORIZED,
            None,
            -32001,
            "Unauthorized".to_string(),
            None,
        );
    }

    if !validate_origin(&state.http_config, &headers) {
        return json_error_response(
            StatusCode::FORBIDDEN,
            None,
            -32002,
            "Forbidden origin".to_string(),
            None,
        );
    }

    if let Some(protocol_version) = headers
        .get("MCP-Protocol-Version")
        .and_then(|v| v.to_str().ok())
    {
        if !SUPPORTED_MCP_PROTOCOL_VERSIONS.contains(&protocol_version) {
            return json_error_response(
                StatusCode::BAD_REQUEST,
                message.id.clone(),
                -32602,
                format!("Unsupported MCP protocol version: {}", protocol_version),
                None,
            );
        }
    }

    if message.jsonrpc != "2.0" {
        return json_error_response(
            StatusCode::BAD_REQUEST,
            message.id.clone(),
            -32600,
            "Invalid JSON-RPC version".to_string(),
            None,
        );
    }

    if message.method.is_none() {
        return StatusCode::ACCEPTED.into_response();
    }

    let method = message.method.as_deref().unwrap_or_default();

    if message.id.is_none() {
        if method == "notifications/initialized" {
            return StatusCode::ACCEPTED.into_response();
        }
        return StatusCode::ACCEPTED.into_response();
    }

    let id = message.id.clone().unwrap_or(Value::Null);
    let params = message.params.unwrap_or(Value::Object(Default::default()));

    match handle_jsonrpc_request(&state, method, params).await {
        Ok(result) => {
            if let Some(ref broadcaster) = state.sse_broadcaster {
                broadcast_jsonrpc_result(broadcaster, method, &result, &addr);
            }
            json_success_response(id, result)
        }
        Err((status, code, error_message, data)) => {
            if let Some(ref broadcaster) = state.sse_broadcaster {
                broadcast_jsonrpc_error(broadcaster, method, code, &error_message, &addr);
            }
            json_error_response(status, Some(id), code, error_message, data)
        }
    }
}

async fn streamable_http_get_handler(
    State(state): State<Arc<StreamableHttpState>>,
    ConnectInfo(_addr): ConnectInfo<SocketAddr>,
    _headers: HeaderMap,
) -> Response {
    let Some(broadcaster) = state.sse_broadcaster.clone() else {
        return (
            StatusCode::METHOD_NOT_ALLOWED,
            [("content-type", "application/json")],
            Json(json!({"error": "SSE stream is not enabled on this endpoint"})),
        )
            .into_response();
    };

    let stream = sse_event_stream(broadcaster);

    (
        StatusCode::OK,
        [
            ("content-type", "text/event-stream"),
            ("cache-control", "no-cache"),
            ("connection", "keep-alive"),
            (
                "MCP-Protocol-Version",
                CURRENT_MCP_PROTOCOL_VERSION,
            ),
        ],
        axum::body::Body::from_stream(stream),
    )
        .into_response()
}

fn sse_event_stream(
    broadcaster: SseBroadcaster,
) -> Pin<Box<dyn Stream<Item = Result<String, std::convert::Infallible>> + Send>> {
    let rx = broadcaster.subscribe();
    Box::pin(tokio_stream::wrappers::BroadcastStream::new(rx).then(|msg| async move {
        match msg {
            Ok(sse_msg) => Ok(sse_msg.format()),
            Err(_) => Ok(format!(
                "event: warning\ndata: {}\n\n",
                json!({"message": "SSE stream closed"})
            )),
        }
    }))
}

fn broadcast_jsonrpc_result(
    broadcaster: &SseBroadcaster,
    method: &str,
    result: &Value,
    addr: &SocketAddr,
) {
    let event_name = match method {
        "initialize" => "initialized",
        "ping" => "ping",
        "tools/list" => "tools_listed",
        "tools/call" => "tool_called",
        "resources/list" => "resources_listed",
        "resources/read" => "resource_read",
        _ => method,
    };
    broadcaster.send(SseMessage::with_event(
        event_name,
        json!({
            "method": method,
            "result": result,
            "client_addr": addr.to_string(),
        }),
    ));
}

fn broadcast_jsonrpc_error(
    broadcaster: &SseBroadcaster,
    method: &str,
    code: i32,
    message: &str,
    addr: &SocketAddr,
) {
    broadcaster.send(SseMessage::with_event(
        "error",
        json!({
            "method": method,
            "error": {
                "code": code,
                "message": message,
            },
            "client_addr": addr.to_string(),
        }),
    ));
}

async fn streamable_http_delete_handler(
    ConnectInfo(_addr): ConnectInfo<SocketAddr>,
    _headers: HeaderMap,
) -> Response {
    (
        StatusCode::METHOD_NOT_ALLOWED,
        [("content-type", "application/json")],
        Json(json!({"error": "Session termination is not implemented on this endpoint"})),
    )
        .into_response()
}

async fn method_not_allowed_handler(Path(_rest): Path<String>) -> Response {
    (
        StatusCode::METHOD_NOT_ALLOWED,
        [("content-type", "application/json")],
        Json(json!({"error": "Method not allowed"})),
    )
        .into_response()
}

async fn handle_jsonrpc_request(
    state: &StreamableHttpState,
    method: &str,
    params: Value,
) -> Result<Value, (StatusCode, i32, String, Option<Value>)> {
    match method {
        "initialize" => Ok(json!({
            "protocolVersion": negotiate_protocol_version(&state.transport.protocol_version),
            "capabilities": server_capabilities(state.protocol.supports_resources()),
            "serverInfo": server_info(&state.transport),
            "instructions": state.transport.instructions.clone(),
        })),
        "ping" => Ok(json!({})),
        "tools/list" => {
            let tools = state
                .protocol
                .list_tools()
                .await
                .map_err(server_error_from)?;
            Ok(json!({
                "tools": tools.into_iter().map(tool_to_mcp_json).collect::<Vec<_>>()
            }))
        }
        "tools/call" => {
            let object = params.as_object().ok_or_else(|| {
                (
                    StatusCode::BAD_REQUEST,
                    -32602,
                    "tools/call params must be an object".to_string(),
                    None,
                )
            })?;
            let tool_name = object.get("name").and_then(Value::as_str).ok_or_else(|| {
                (
                    StatusCode::BAD_REQUEST,
                    -32602,
                    "tools/call requires params.name".to_string(),
                    None,
                )
            })?;
            let arguments = object
                .get("arguments")
                .cloned()
                .unwrap_or_else(|| json!({}));
            let result = state
                .protocol
                .execute(tool_name, arguments)
                .await
                .map_err(tool_protocol_error_from)?;
            Ok(tool_result_to_mcp_json(result))
        }
        "resources/list" => {
            if !state.protocol.supports_resources() {
                return Err((
                    StatusCode::BAD_REQUEST,
                    -32601,
                    "resources/list is not supported".to_string(),
                    None,
                ));
            }
            let resources = state
                .protocol
                .list_resources()
                .await
                .map_err(server_error_from)?;
            Ok(json!({ "resources": resources }))
        }
        "resources/read" => {
            if !state.protocol.supports_resources() {
                return Err((
                    StatusCode::BAD_REQUEST,
                    -32601,
                    "resources/read is not supported".to_string(),
                    None,
                ));
            }
            let object = params.as_object().ok_or_else(|| {
                (
                    StatusCode::BAD_REQUEST,
                    -32602,
                    "resources/read params must be an object".to_string(),
                    None,
                )
            })?;
            let uri = object.get("uri").and_then(Value::as_str).ok_or_else(|| {
                (
                    StatusCode::BAD_REQUEST,
                    -32602,
                    "resources/read requires params.uri".to_string(),
                    None,
                )
            })?;
            let content = state
                .protocol
                .read_resource(uri)
                .await
                .map_err(server_error_from)?;
            Ok(json!({
                "contents": [
                    {
                        "uri": uri,
                        "text": content
                    }
                ]
            }))
        }
        _ => Err((
            StatusCode::BAD_REQUEST,
            -32601,
            format!("Method not found: {}", method),
            None,
        )),
    }
}

fn authorize(
    config: &StreamableHttpRuntimeConfig,
    headers: &HeaderMap,
    ip: std::net::IpAddr,
) -> bool {
    if !config.ip_filter.is_allowed(ip) {
        return false;
    }

    match config.bearer_token.as_deref() {
        None => true,
        Some(expected) => {
            let provided = headers
                .get("Authorization")
                .and_then(|v| v.to_str().ok())
                .and_then(|v| v.strip_prefix("Bearer "))
                .unwrap_or("");
            let expected_hash = Sha256::digest(expected.as_bytes());
            let provided_hash = Sha256::digest(provided.as_bytes());
            expected_hash.ct_eq(&provided_hash).into()
        }
    }
}

fn validate_origin(config: &StreamableHttpRuntimeConfig, headers: &HeaderMap) -> bool {
    if config.skip_origin_validation {
        return true;
    }

    let Some(origin) = headers.get("Origin").and_then(|v| v.to_str().ok()) else {
        return true;
    };

    origin.starts_with("http://127.0.0.1")
        || origin.starts_with("http://localhost")
        || origin.starts_with("http://[::1]")
        || origin.starts_with("https://127.0.0.1")
        || origin.starts_with("https://localhost")
        || origin.starts_with("https://[::1]")
}

fn negotiate_protocol_version(server_protocol_version: &str) -> &str {
    if SUPPORTED_MCP_PROTOCOL_VERSIONS.contains(&server_protocol_version) {
        server_protocol_version
    } else {
        CURRENT_MCP_PROTOCOL_VERSION
    }
}

fn server_capabilities(include_resources: bool) -> Value {
    let mut capabilities = serde_json::Map::new();
    capabilities.insert("tools".to_string(), json!({"listChanged": false}));
    if include_resources {
        capabilities.insert(
            "resources".to_string(),
            json!({"subscribe": false, "listChanged": false}),
        );
    }
    Value::Object(capabilities)
}

fn server_info(config: &StreamableHttpConfig) -> Value {
    let mut info = serde_json::Map::from_iter([
        (
            "name".to_string(),
            Value::String(config.server_name.clone()),
        ),
        (
            "version".to_string(),
            Value::String(config.server_version.clone()),
        ),
    ]);
    if let Some(title) = &config.server_title {
        info.insert("title".to_string(), Value::String(title.clone()));
    }
    Value::Object(info)
}

fn tool_to_mcp_json(tool: crate::protocol::ToolMetadata) -> Value {
    let definition = tool.to_tool_definition();
    let mut object = serde_json::Map::from_iter([
        ("name".to_string(), Value::String(definition.name)),
        (
            "description".to_string(),
            Value::String(definition.description),
        ),
        ("inputSchema".to_string(), definition.parameters_schema),
    ]);
    if let Some(title) = tool
        .protocol_metadata
        .get("title")
        .and_then(Value::as_str)
        .map(ToOwned::to_owned)
    {
        object.insert("title".to_string(), Value::String(title));
    }
    if let Some(output_schema) = tool.protocol_metadata.get("outputSchema") {
        object.insert("outputSchema".to_string(), output_schema.clone());
    }
    if let Some(annotations) = tool.protocol_metadata.get("annotations") {
        object.insert("annotations".to_string(), annotations.clone());
    }
    // 2025-11-25: execution object with taskSupport (defaults to "optional" for tools that
    // support async execution via the MCP task mechanism).
    object.insert(
        "execution".to_string(),
        json!({"taskSupport": "optional"}),
    );
    Value::Object(object)
}

fn tool_result_to_mcp_json(result: crate::protocol::ToolResult) -> Value {
    let text = if let Some(error) = &result.error {
        error.clone()
    } else if result.output.is_string() {
        result.output.as_str().unwrap_or_default().to_string()
    } else {
        serde_json::to_string_pretty(&result.output).unwrap_or_else(|_| result.output.to_string())
    };

    let mut object = serde_json::Map::from_iter([
        (
            "content".to_string(),
            Value::Array(vec![json!({
                "type": "text",
                "text": text
            })]),
        ),
        ("isError".to_string(), Value::Bool(!result.success)),
    ]);

    if result.output.is_object() {
        object.insert("structuredContent".to_string(), result.output);
    }

    Value::Object(object)
}

fn json_success_response(id: Value, result: Value) -> Response {
    let response = JsonRpcResponse {
        jsonrpc: "2.0",
        id,
        result: Some(result),
        error: None,
    };
    (
        StatusCode::OK,
        [
            ("content-type", HeaderValue::from_static("application/json")),
            (
                "MCP-Protocol-Version",
                HeaderValue::from_static(CURRENT_MCP_PROTOCOL_VERSION),
            ),
        ],
        Json(response),
    )
        .into_response()
}

fn json_error_response(
    status: StatusCode,
    id: Option<Value>,
    code: i32,
    message: String,
    data: Option<Value>,
) -> Response {
    let response = JsonRpcResponse {
        jsonrpc: "2.0",
        id: id.unwrap_or(Value::Null),
        result: None,
        error: Some(JsonRpcErrorObject {
            code,
            message,
            data,
        }),
    };
    (
        status,
        [
            ("content-type", HeaderValue::from_static("application/json")),
            (
                "MCP-Protocol-Version",
                HeaderValue::from_static(CURRENT_MCP_PROTOCOL_VERSION),
            ),
        ],
        Json(response),
    )
        .into_response()
}

fn server_error_from(
    error: Box<dyn Error + Send + Sync>,
) -> (StatusCode, i32, String, Option<Value>) {
    (
        StatusCode::INTERNAL_SERVER_ERROR,
        -32603,
        error.to_string(),
        None,
    )
}

fn tool_protocol_error_from(
    error: Box<dyn Error + Send + Sync>,
) -> (StatusCode, i32, String, Option<Value>) {
    if let Some(tool_error) = error.downcast_ref::<ToolError>() {
        match tool_error {
            ToolError::NotFound(message) => {
                (StatusCode::BAD_REQUEST, -32602, message.clone(), None)
            }
            ToolError::InvalidParameters(message) => {
                (StatusCode::BAD_REQUEST, -32602, message.clone(), None)
            }
            ToolError::ExecutionFailed(message) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                -32603,
                message.clone(),
                None,
            ),
            ToolError::ProtocolError(message) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                -32603,
                message.clone(),
                None,
            ),
        }
    } else {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            -32603,
            error.to_string(),
            None,
        )
    }
}
