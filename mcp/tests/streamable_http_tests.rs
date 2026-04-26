//! Integration tests for streamable HTTP transport and MCP 2025-11-25 protocol features.
//!
//! These tests require the `server` feature to be enabled since they exercise
//! the axum-based HTTP router.

#![cfg(feature = "server")]

use axum::body::to_bytes;
use axum::extract::connect_info::ConnectInfo;
use axum::http::{Method, Request, StatusCode};
use mcp::http::HttpServerConfig;
use mcp::streamable_http::{
    streamable_http_router, StreamableHttpConfig, CURRENT_MCP_PROTOCOL_VERSION,
    SUPPORTED_MCP_PROTOCOL_VERSIONS,
};
use mcp::{IpFilter, ToolMetadata, ToolProtocol, ToolResult};
use serde_json::json;
use std::sync::Arc;
use tower::ServiceExt;

// ── Helpers ───────────────────────────────────────────────────────────────

/// Minimal protocol that lists one tool and echoes back parameters.
struct EchoProtocol;

#[async_trait::async_trait]
impl ToolProtocol for EchoProtocol {
    async fn execute(
        &self,
        _tool_name: &str,
        parameters: serde_json::Value,
    ) -> Result<ToolResult, Box<dyn std::error::Error + Send + Sync>> {
        Ok(ToolResult::success(parameters))
    }

    async fn list_tools(
        &self,
    ) -> Result<Vec<ToolMetadata>, Box<dyn std::error::Error + Send + Sync>> {
        Ok(vec![
            ToolMetadata::new("echo", "Echo parameters back"),
        ])
    }

    async fn get_tool_metadata(
        &self,
        _tool_name: &str,
    ) -> Result<ToolMetadata, Box<dyn std::error::Error + Send + Sync>> {
        Ok(ToolMetadata::new("echo", "Echo parameters back"))
    }

    fn protocol_name(&self) -> &str {
        "echo"
    }
}

fn make_router(skip_origin: bool) -> axum::Router {
    let config = StreamableHttpConfig::new("test-server", "0.1.0")
        .with_skip_origin_validation(skip_origin);
    streamable_http_router(
        &HttpServerConfig {
            addr: std::net::SocketAddr::from(([127, 0, 0, 1], 0)),
            bearer_token: None,
            ip_filter: IpFilter::new(),
            event_handler: None,
        },
        &config,
        Arc::new(EchoProtocol),
    )
}

fn client_addr() -> std::net::SocketAddr {
    std::net::SocketAddr::from(([127, 0, 0, 1], 12345))
}

async fn post_json(
    router: &axum::Router,
    body: serde_json::Value,
) -> (StatusCode, String) {
    let mut req = Request::builder()
        .method(Method::POST)
        .uri("/")
        .header("content-type", "application/json")
        .body(axum::body::Body::from(body.to_string()))
        .unwrap();
    req.extensions_mut().insert(ConnectInfo(client_addr()));
    let response = router.clone().oneshot(req).await.unwrap();
    let status = response.status();
    let bytes = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    (status, String::from_utf8(bytes.to_vec()).unwrap())
}

async fn post_json_with_origin(
    router: &axum::Router,
    body: serde_json::Value,
    origin: &str,
) -> (StatusCode, String) {
    let mut req = Request::builder()
        .method(Method::POST)
        .uri("/")
        .header("content-type", "application/json")
        .header("origin", origin)
        .body(axum::body::Body::from(body.to_string()))
        .unwrap();
    req.extensions_mut().insert(ConnectInfo(client_addr()));
    let response = router.clone().oneshot(req).await.unwrap();
    let status = response.status();
    let bytes = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    (status, String::from_utf8(bytes.to_vec()).unwrap())
}

// ── Protocol version tests ────────────────────────────────────────────────

#[tokio::test]
async fn test_current_protocol_version_is_2025_11_25() {
    assert_eq!(CURRENT_MCP_PROTOCOL_VERSION, "2025-11-25");
}

#[tokio::test]
async fn test_supported_versions_includes_current_first() {
    assert_eq!(SUPPORTED_MCP_PROTOCOL_VERSIONS[0], "2025-11-25");
    assert!(SUPPORTED_MCP_PROTOCOL_VERSIONS.contains(&"2025-06-18"));
    assert!(SUPPORTED_MCP_PROTOCOL_VERSIONS.contains(&"2024-11-05"));
}

#[tokio::test]
async fn test_initialize_returns_current_protocol_version() {
    let router = make_router(false);
    let body = json!({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {}
    });
    let (status, text) = post_json(&router, body).await;
    assert_eq!(status, StatusCode::OK);
    let response: serde_json::Value = serde_json::from_str(&text).unwrap();
    assert_eq!(response["result"]["protocolVersion"], "2025-11-25");
    assert_eq!(response["result"]["serverInfo"]["name"], "test-server");
    assert!(response["result"]["capabilities"]["tools"].is_object());
}

#[tokio::test]
async fn test_initialize_with_legacy_client_version_accepted() {
    let router = make_router(false);
    let body = json!({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {}
    });
    let mut req = Request::builder()
        .method(Method::POST)
        .uri("/")
        .header("content-type", "application/json")
        .header("MCP-Protocol-Version", "2025-06-18")
        .body(axum::body::Body::from(body.to_string()))
        .unwrap();
    req.extensions_mut().insert(ConnectInfo(client_addr()));
    let response = router.clone().oneshot(req).await.unwrap();
    assert_eq!(response.status(), StatusCode::OK);
}

#[tokio::test]
async fn test_unsupported_protocol_version_returns_bad_request() {
    let router = make_router(false);
    let body = json!({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {}
    });
    let mut req = Request::builder()
        .method(Method::POST)
        .uri("/")
        .header("content-type", "application/json")
        .header("MCP-Protocol-Version", "2023-01-01")
        .body(axum::body::Body::from(body.to_string()))
        .unwrap();
    req.extensions_mut().insert(ConnectInfo(client_addr()));
    let response = router.clone().oneshot(req).await.unwrap();
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let bytes = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    let response: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert!(response["error"]["message"]
        .as_str()
        .unwrap()
        .contains("Unsupported MCP protocol version"));
}

// ── Origin validation tests ─────────────────────────────────────────────────

#[tokio::test]
async fn test_localhost_origin_allowed() {
    let router = make_router(false);
    let body = json!({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "ping",
        "params": {}
    });
    let (status, _) = post_json_with_origin(&router, body, "http://localhost:3000").await;
    assert_eq!(status, StatusCode::OK);
}

#[tokio::test]
async fn test_127_0_0_1_origin_allowed() {
    let router = make_router(false);
    let body = json!({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "ping",
        "params": {}
    });
    let (status, _) = post_json_with_origin(&router, body, "http://127.0.0.1:3000").await;
    assert_eq!(status, StatusCode::OK);
}

#[tokio::test]
async fn test_lan_origin_blocked_by_default() {
    let router = make_router(false);
    let body = json!({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "ping",
        "params": {}
    });
    let (status, text) =
        post_json_with_origin(&router, body, "http://192.168.1.50:3000").await;
    assert_eq!(status, StatusCode::FORBIDDEN);
    let response: serde_json::Value = serde_json::from_str(&text).unwrap();
    assert!(response["error"]["message"]
        .as_str()
        .unwrap()
        .contains("Forbidden origin"));
}

#[tokio::test]
async fn test_lan_origin_allowed_when_skip_validation_enabled() {
    let router = make_router(true);
    let body = json!({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "ping",
        "params": {}
    });
    let (status, _) =
        post_json_with_origin(&router, body, "http://192.168.1.50:3000").await;
    assert_eq!(status, StatusCode::OK);
}

#[tokio::test]
async fn test_missing_origin_header_allowed() {
    let router = make_router(false);
    let body = json!({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "ping",
        "params": {}
    });
    let (status, _) = post_json(&router, body).await;
    assert_eq!(status, StatusCode::OK);
}

// ── Tool listing with 2025-11-25 execution field ──────────────────────────

#[tokio::test]
async fn test_tools_list_includes_execution_task_support() {
    let router = make_router(false);
    let body = json!({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/list",
        "params": {}
    });
    let (status, text) = post_json(&router, body).await;
    assert_eq!(status, StatusCode::OK);
    let response: serde_json::Value = serde_json::from_str(&text).unwrap();
    let tools = response["result"]["tools"].as_array().unwrap();
    assert_eq!(tools.len(), 1);
    let tool = &tools[0];
    assert_eq!(tool["name"], "echo");
    assert!(tool["execution"].is_object());
    assert_eq!(tool["execution"]["taskSupport"], "optional");
}

// ── Response header tests ───────────────────────────────────────────────────

#[tokio::test]
async fn test_success_response_includes_mcp_protocol_version_header() {
    let router = make_router(false);
    let body = json!({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {}
    });
    let mut req = Request::builder()
        .method(Method::POST)
        .uri("/")
        .header("content-type", "application/json")
        .body(axum::body::Body::from(body.to_string()))
        .unwrap();
    req.extensions_mut().insert(ConnectInfo(client_addr()));
    let response = router.clone().oneshot(req).await.unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let version_header: Option<&str> = response
        .headers()
        .get("MCP-Protocol-Version")
        .and_then(|v| v.to_str().ok());
    assert_eq!(version_header, Some("2025-11-25"));
}

// ── Error handling tests ──────────────────────────────────────────────────

#[tokio::test]
async fn test_invalid_jsonrpc_version_returns_bad_request() {
    let router = make_router(false);
    let body = json!({
        "jsonrpc": "1.0",
        "id": 1,
        "method": "initialize",
        "params": {}
    });
    let (status, text) = post_json(&router, body).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    let response: serde_json::Value = serde_json::from_str(&text).unwrap();
    assert!(response["error"]["message"]
        .as_str()
        .unwrap()
        .contains("Invalid JSON-RPC version"));
}

#[tokio::test]
async fn test_notification_initialized_returns_accepted() {
    let router = make_router(false);
    let body = json!({
        "jsonrpc": "2.0",
        "method": "notifications/initialized"
    });
    let (status, _) = post_json(&router, body).await;
    assert_eq!(status, StatusCode::ACCEPTED);
}

#[tokio::test]
async fn test_get_returns_method_not_allowed() {
    let router = make_router(false);
    let mut req = Request::builder()
        .method(Method::GET)
        .uri("/")
        .body(axum::body::Body::empty())
        .unwrap();
    req.extensions_mut().insert(ConnectInfo(client_addr()));
    let response = router.clone().oneshot(req).await.unwrap();
    assert_eq!(response.status(), StatusCode::METHOD_NOT_ALLOWED);
}

#[tokio::test]
async fn test_delete_returns_method_not_allowed() {
    let router = make_router(false);
    let mut req = Request::builder()
        .method(Method::DELETE)
        .uri("/")
        .body(axum::body::Body::empty())
        .unwrap();
    req.extensions_mut().insert(ConnectInfo(client_addr()));
    let response = router.clone().oneshot(req).await.unwrap();
    assert_eq!(response.status(), StatusCode::METHOD_NOT_ALLOWED);
}

#[tokio::test]
async fn test_unknown_method_returns_bad_request() {
    let router = make_router(false);
    let body = json!({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "unknown/method",
        "params": {}
    });
    let (status, text) = post_json(&router, body).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    let response: serde_json::Value = serde_json::from_str(&text).unwrap();
    assert!(response["error"]["message"]
        .as_str()
        .unwrap()
        .contains("Method not found"));
}
