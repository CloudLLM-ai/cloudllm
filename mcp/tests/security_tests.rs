//! Security regression tests for the streamable HTTP transport.
//!
//! Each test pins a vulnerability that was found in a security review of the
//! crate. They exercise the real axum router in-process.

#![cfg(feature = "server")]

use axum::body::Body;
use axum::extract::connect_info::ConnectInfo;
use axum::http::{Method, Request, StatusCode};
use mcp::http::HttpServerConfig;
use mcp::streamable_http::{streamable_http_router_with_sse, SseBroadcaster, StreamableHttpConfig};
use mcp::{IpFilter, ToolMetadata, ToolProtocol, ToolResult};
use std::sync::Arc;
use tower::ServiceExt;

const TOKEN: &str = "sk-test-secret-token";

struct Echo;

#[async_trait::async_trait]
impl ToolProtocol for Echo {
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
        Ok(vec![ToolMetadata::new("echo", "Echo")])
    }

    async fn get_tool_metadata(
        &self,
        _tool_name: &str,
    ) -> Result<ToolMetadata, Box<dyn std::error::Error + Send + Sync>> {
        Ok(ToolMetadata::new("echo", "Echo"))
    }

    fn protocol_name(&self) -> &str {
        "echo"
    }
}

fn addr() -> std::net::SocketAddr {
    std::net::SocketAddr::from(([127, 0, 0, 1], 4242))
}

fn router(bearer: Option<&str>, with_sse: bool) -> (axum::Router, SseBroadcaster) {
    let broadcaster = SseBroadcaster::new(64);
    let router = streamable_http_router_with_sse(
        &HttpServerConfig {
            addr: addr(),
            bearer_token: bearer.map(str::to_string),
            bearer_authorizer: None,
            ip_filter: IpFilter::new(),
            event_handler: None,
        },
        &StreamableHttpConfig::new("t", "0.0.1"),
        Arc::new(Echo),
        with_sse.then(|| broadcaster.clone()),
    );
    (router, broadcaster)
}

async fn get(
    router: &axum::Router,
    auth: Option<&str>,
    origin: Option<&str>,
) -> axum::response::Response {
    let mut b = Request::builder().method(Method::GET).uri("/");
    if let Some(a) = auth {
        b = b.header("authorization", a);
    }
    if let Some(o) = origin {
        b = b.header("origin", o);
    }
    let mut req = b.body(Body::empty()).unwrap();
    req.extensions_mut().insert(ConnectInfo(addr()));
    router.clone().oneshot(req).await.unwrap()
}

#[tokio::test]
async fn sse_get_requires_bearer_token() {
    let (router, _b) = router(Some(TOKEN), true);

    assert_eq!(
        get(&router, None, None).await.status(),
        StatusCode::UNAUTHORIZED,
        "unauthenticated GET must not open the event stream"
    );
    assert_eq!(
        get(&router, Some("Bearer wrong"), None).await.status(),
        StatusCode::UNAUTHORIZED
    );
    let ok = get(&router, Some(&format!("Bearer {TOKEN}")), None).await;
    assert_eq!(ok.status(), StatusCode::OK);
    assert_eq!(ok.headers()["content-type"], "text/event-stream");
}

#[tokio::test]
async fn sse_get_validates_origin() {
    let (router, _b) = router(Some(TOKEN), true);
    let auth = format!("Bearer {TOKEN}");
    let resp = get(&router, Some(&auth), Some("http://evil.example")).await;
    assert_eq!(resp.status(), StatusCode::FORBIDDEN);
}

#[tokio::test]
async fn sse_get_without_broadcaster_still_authenticates_first() {
    let (router, _b) = router(Some(TOKEN), false);
    assert_eq!(
        get(&router, None, None).await.status(),
        StatusCode::UNAUTHORIZED
    );
}

#[tokio::test]
async fn sse_get_respects_ip_filter() {
    let mut filter = IpFilter::new();
    filter.allow("10.9.9.9").unwrap();
    let router = streamable_http_router_with_sse(
        &HttpServerConfig {
            addr: addr(),
            bearer_token: None,
            bearer_authorizer: None,
            ip_filter: filter,
            event_handler: None,
        },
        &StreamableHttpConfig::new("t", "0.0.1"),
        Arc::new(Echo),
        Some(SseBroadcaster::new(8)),
    );
    assert_eq!(
        get(&router, None, None).await.status(),
        StatusCode::UNAUTHORIZED
    );
}

// ── SSE payload minimisation ────────────────────────────────────────────────

use mcp::streamable_http::SsePayloadMode;
use mcp::McpEvent;
use serde_json::json;
use std::time::Duration;

const SECRET: &str = "sk_live_super_secret_value";

async fn post_tool_call(router: &axum::Router) {
    let mut req = Request::builder()
        .method(Method::POST)
        .uri("/")
        .header("content-type", "application/json")
        .body(Body::from(
            json!({
                "jsonrpc": "2.0", "id": 1, "method": "tools/call",
                "params": {"name": "echo", "arguments": {"api_key": SECRET}}
            })
            .to_string(),
        ))
        .unwrap();
    req.extensions_mut().insert(ConnectInfo(addr()));
    let resp = router.clone().oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
}

fn router_with(broadcaster: SseBroadcaster) -> axum::Router {
    streamable_http_router_with_sse(
        &HttpServerConfig {
            addr: addr(),
            bearer_token: None,
            bearer_authorizer: None,
            ip_filter: IpFilter::new(),
            event_handler: None,
        },
        &StreamableHttpConfig::new("t", "0.0.1"),
        Arc::new(Echo),
        Some(broadcaster),
    )
}

async fn next_text(rx: &mut tokio::sync::broadcast::Receiver<mcp::SseMessage>) -> String {
    let msg = tokio::time::timeout(Duration::from_millis(500), rx.recv())
        .await
        .expect("timeout")
        .expect("message");
    msg.data.to_string()
}

#[tokio::test]
async fn sse_default_does_not_broadcast_tool_results() {
    let b = SseBroadcaster::new(16);
    assert_eq!(b.payload_mode(), SsePayloadMode::Minimal);
    let mut rx = b.subscribe();
    post_tool_call(&router_with(b)).await;
    let text = next_text(&mut rx).await;
    assert!(!text.contains(SECRET), "tool result leaked to SSE: {text}");
    assert!(text.contains("tools/call"));
}

#[tokio::test]
async fn sse_default_does_not_broadcast_tool_parameters_or_errors() {
    let b = SseBroadcaster::new(16);
    let mut rx = b.subscribe();
    b.broadcast_mcp_event(&McpEvent::ToolCallReceived {
        client_addr: "1.2.3.4".into(),
        tool_name: "echo".into(),
        parameters: json!({"api_key": SECRET}),
    });
    b.broadcast_mcp_event(&McpEvent::ToolCallCompleted {
        client_addr: "1.2.3.4".into(),
        tool_name: "echo".into(),
        success: false,
        error: Some(format!("boom {SECRET}")),
        duration_ms: 1,
    });
    b.broadcast_mcp_event(&McpEvent::ToolError {
        source: "1.2.3.4".into(),
        tool_name: "echo".into(),
        error: format!("boom {SECRET}"),
        duration_ms: 1,
    });
    b.broadcast_mcp_event(&McpEvent::RemoteToolCallStarted {
        endpoint: "http://x".into(),
        tool_name: "echo".into(),
        parameters: json!({"api_key": SECRET}),
    });
    for _ in 0..4 {
        let text = next_text(&mut rx).await;
        assert!(!text.contains(SECRET), "leaked: {text}");
        assert!(
            text.contains("tool_name"),
            "tool name must still be reported"
        );
    }
}

#[tokio::test]
async fn sse_full_mode_is_an_explicit_opt_in() {
    let b = SseBroadcaster::new(16).with_payload_mode(SsePayloadMode::Full);
    let mut rx = b.subscribe();
    post_tool_call(&router_with(b)).await;
    assert!(next_text(&mut rx).await.contains(SECRET));
}

// ── Origin validation ───────────────────────────────────────────────────────

async fn post_with_origin(origin: &str) -> StatusCode {
    let (router, _b) = router(None, false);
    let mut req = Request::builder()
        .method(Method::POST)
        .uri("/")
        .header("content-type", "application/json")
        .header("origin", origin)
        .body(Body::from(
            json!({"jsonrpc": "2.0", "id": 1, "method": "ping"}).to_string(),
        ))
        .unwrap();
    req.extensions_mut().insert(ConnectInfo(addr()));
    router.oneshot(req).await.unwrap().status()
}

#[tokio::test]
async fn origin_must_match_loopback_host_exactly() {
    for evil in [
        "http://localhost.evil.com",
        "http://127.0.0.1.evil.com",
        "http://127.0.0.10",
        "https://localhost.evil.com:8443",
        "http://localhostevil.com",
        "http://localhost@evil.com",
        "http://localhost:80@evil.com",
        "http://localhost:evil",
        "http://[::1].evil.com",
        "http://localhost/",
        "ftp://localhost",
        "null",
        "",
    ] {
        assert_eq!(
            post_with_origin(evil).await,
            StatusCode::FORBIDDEN,
            "origin {evil:?} must be rejected"
        );
    }
}

#[tokio::test]
async fn loopback_origins_with_any_port_are_allowed() {
    for ok in [
        "http://localhost",
        "http://localhost:3000",
        "https://localhost:8443",
        "http://LOCALHOST:3000",
        "http://127.0.0.1",
        "http://127.0.0.1:9471",
        "https://127.0.0.1:1",
        "http://[::1]",
        "http://[::1]:5173",
    ] {
        assert_eq!(
            post_with_origin(ok).await,
            StatusCode::OK,
            "origin {ok:?} must be allowed"
        );
    }
}
