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
