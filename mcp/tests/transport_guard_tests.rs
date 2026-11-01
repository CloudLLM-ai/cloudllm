//! Security regression tests for Origin / Host validation on both HTTP
//! transports (legacy `/tools/*` routes and streamable HTTP).

#![cfg(feature = "server")]

use axum::body::Body;
use axum::extract::connect_info::ConnectInfo;
use axum::http::{Method, Request, StatusCode};
use mcp::http::{axum_router, HttpServerConfig};
use mcp::streamable_http::{streamable_http_router_with_sse, StreamableHttpConfig};
use mcp::{IpFilter, ToolMetadata, ToolProtocol, ToolResult};
use std::net::SocketAddr;
use std::sync::Arc;
use tower::ServiceExt;

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

fn client() -> SocketAddr {
    SocketAddr::from(([127, 0, 0, 1], 4242))
}

fn config(bind: SocketAddr) -> HttpServerConfig {
    HttpServerConfig {
        addr: bind,
        bearer_token: None,
        bearer_authorizer: None,
        ip_filter: IpFilter::new(),
        event_handler: None,
    }
}

fn loopback() -> SocketAddr {
    SocketAddr::from(([127, 0, 0, 1], 8080))
}

async fn send(
    router: &axum::Router,
    method: Method,
    uri: &str,
    headers: &[(&str, &str)],
) -> StatusCode {
    let mut b = Request::builder().method(method).uri(uri);
    for (k, v) in headers {
        b = b.header(*k, *v);
    }
    let mut req = b.body(Body::from("{}")).unwrap();
    req.extensions_mut().insert(ConnectInfo(client()));
    router.clone().oneshot(req).await.unwrap().status()
}

fn legacy(cfg: &HttpServerConfig) -> axum::Router {
    axum_router(cfg, Arc::new(Echo))
}

fn streamable(cfg: &HttpServerConfig) -> axum::Router {
    streamable_http_router_with_sse(
        cfg,
        &StreamableHttpConfig::new("t", "0.0.1"),
        Arc::new(Echo),
        None,
    )
}

const JSON: (&str, &str) = ("content-type", "application/json");

// ── Legacy routes: Origin ───────────────────────────────────────────────────

#[tokio::test]
async fn legacy_routes_reject_foreign_origin() {
    let router = legacy(&config(loopback()));
    for route in [
        "/tools/list",
        "/tools/execute",
        "/resources/list",
        "/resources/read",
    ] {
        let status = send(
            &router,
            Method::POST,
            route,
            &[JSON, ("origin", "http://evil.example")],
        )
        .await;
        assert_eq!(
            status,
            StatusCode::FORBIDDEN,
            "{route} accepted a foreign Origin"
        );
    }
}

#[tokio::test]
async fn legacy_routes_reject_lookalike_loopback_origin() {
    let router = legacy(&config(loopback()));
    let status = send(
        &router,
        Method::POST,
        "/tools/list",
        &[JSON, ("origin", "http://localhost.evil.com")],
    )
    .await;
    assert_eq!(status, StatusCode::FORBIDDEN);
}

#[tokio::test]
async fn legacy_routes_accept_loopback_and_absent_origin() {
    let router = legacy(&config(loopback()));
    assert_eq!(
        send(&router, Method::POST, "/tools/list", &[JSON]).await,
        StatusCode::OK,
        "non-browser clients send no Origin"
    );
    assert_eq!(
        send(
            &router,
            Method::POST,
            "/tools/list",
            &[JSON, ("origin", "http://localhost:3000")]
        )
        .await,
        StatusCode::OK
    );
}

// ── Host header (DNS rebinding) ─────────────────────────────────────────────

#[tokio::test]
async fn legacy_loopback_bind_host_policy() {
    let router = legacy(&config(loopback()));
    for host in [
        "attacker.example",
        "attacker.example:8080",
        "127.0.0.1.evil.com",
    ] {
        assert_eq!(
            send(
                &router,
                Method::POST,
                "/tools/list",
                &[JSON, ("host", host)]
            )
            .await,
            StatusCode::FORBIDDEN,
            "Host {host} must be rejected on a loopback bind"
        );
    }
    for host in [
        "localhost",
        "localhost:8080",
        "127.0.0.1:8080",
        "[::1]:8080",
        "LOCALHOST",
    ] {
        assert_eq!(
            send(
                &router,
                Method::POST,
                "/tools/list",
                &[JSON, ("host", host)]
            )
            .await,
            StatusCode::OK,
            "Host {host} must be accepted on a loopback bind"
        );
    }
}

async fn rpc_ping(router: &axum::Router, host: &str) -> StatusCode {
    let mut req = Request::builder()
        .method(Method::POST)
        .uri("/")
        .header("content-type", "application/json")
        .header("host", host)
        .body(Body::from(r#"{"jsonrpc":"2.0","id":1,"method":"ping"}"#))
        .unwrap();
    req.extensions_mut().insert(ConnectInfo(client()));
    router.clone().oneshot(req).await.unwrap().status()
}

#[tokio::test]
async fn streamable_loopback_bind_host_policy() {
    let router = streamable(&config(loopback()));
    assert_eq!(
        rpc_ping(&router, "attacker.example:8080").await,
        StatusCode::FORBIDDEN
    );
    assert_eq!(rpc_ping(&router, "localhost:8080").await, StatusCode::OK);
    assert_eq!(rpc_ping(&router, "127.0.0.1").await, StatusCode::OK);
}

#[tokio::test]
async fn non_loopback_bind_does_not_default_to_loopback_hosts() {
    let public = SocketAddr::from(([0, 0, 0, 0], 8080));
    let router = legacy(&config(public));
    assert_eq!(
        send(
            &router,
            Method::POST,
            "/tools/list",
            &[JSON, ("host", "mcp.example.com")]
        )
        .await,
        StatusCode::OK
    );
}

// ── Opt-in Host allow-list ──────────────────────────────────────────────────

#[tokio::test]
async fn allowed_hosts_replaces_loopback_default_on_both_transports() {
    use mcp::http::{axum_router_with_guard, RequestGuard};
    let cfg = config(loopback());
    let guard = RequestGuard::new().with_allowed_hosts(["mcp.example.com"]);
    let router = axum_router_with_guard(&cfg, Arc::new(Echo), &guard);
    for (host, want) in [
        ("mcp.example.com", StatusCode::OK),
        ("MCP.example.com:443", StatusCode::OK),
        ("localhost", StatusCode::FORBIDDEN),
        ("evil.example", StatusCode::FORBIDDEN),
    ] {
        assert_eq!(
            send(
                &router,
                Method::POST,
                "/tools/list",
                &[JSON, ("host", host)]
            )
            .await,
            want,
            "legacy Host {host}"
        );
    }

    let transport = StreamableHttpConfig::new("t", "0.0.1").with_allowed_hosts(["mcp.example.com"]);
    let router = streamable_http_router_with_sse(&cfg, &transport, Arc::new(Echo), None);
    assert_eq!(rpc_ping(&router, "mcp.example.com").await, StatusCode::OK);
    assert_eq!(rpc_ping(&router, "localhost").await, StatusCode::FORBIDDEN);
}

#[tokio::test]
async fn malformed_host_is_rejected_on_loopback_bind() {
    let router = legacy(&config(loopback()));
    for host in ["localhost:notaport", "user@localhost", "localhost/x", ""] {
        assert_eq!(
            send(
                &router,
                Method::POST,
                "/tools/list",
                &[JSON, ("host", host)]
            )
            .await,
            StatusCode::FORBIDDEN,
            "Host {host:?}"
        );
    }
}

// ── Error hygiene ───────────────────────────────────────────────────────────

struct Failing;

#[async_trait::async_trait]
impl ToolProtocol for Failing {
    async fn execute(
        &self,
        _t: &str,
        _p: serde_json::Value,
    ) -> Result<ToolResult, Box<dyn std::error::Error + Send + Sync>> {
        Err("postgres://user:hunter2@10.0.0.5/db connection refused".into())
    }
    async fn list_tools(
        &self,
    ) -> Result<Vec<ToolMetadata>, Box<dyn std::error::Error + Send + Sync>> {
        Err("open /var/lib/secret/tools.db: permission denied".into())
    }
    async fn get_tool_metadata(
        &self,
        _t: &str,
    ) -> Result<ToolMetadata, Box<dyn std::error::Error + Send + Sync>> {
        Err("nope".into())
    }
    fn protocol_name(&self) -> &str {
        "failing"
    }
}

async fn body_text(resp: axum::response::Response) -> (StatusCode, String) {
    let status = resp.status();
    let bytes = http_body_util::BodyExt::collect(resp.into_body())
        .await
        .unwrap()
        .to_bytes();
    (status, String::from_utf8_lossy(&bytes).into_owned())
}

#[tokio::test]
async fn internal_errors_are_not_echoed_to_clients() {
    let mut req = Request::builder()
        .method(Method::POST)
        .uri("/tools/list")
        .header("content-type", "application/json")
        .body(Body::from("{}"))
        .unwrap();
    req.extensions_mut().insert(ConnectInfo(client()));
    let router = axum_router(&config(loopback()), Arc::new(Failing));
    let (status, body) = body_text(router.oneshot(req).await.unwrap()).await;
    assert_eq!(status, StatusCode::INTERNAL_SERVER_ERROR);
    assert!(!body.contains("/var/lib/secret"), "leaked: {body}");
    assert!(body.contains("request id"), "no request id: {body}");

    let router = streamable_http_router_with_sse(
        &config(loopback()),
        &StreamableHttpConfig::new("t", "0.0.1"),
        Arc::new(Failing),
        None,
    );
    let mut req = Request::builder()
        .method(Method::POST)
        .uri("/")
        .header("content-type", "application/json")
        .body(Body::from(
            r#"{"jsonrpc":"2.0","id":1,"method":"tools/list"}"#,
        ))
        .unwrap();
    req.extensions_mut().insert(ConnectInfo(client()));
    let (status, body) = body_text(router.oneshot(req).await.unwrap()).await;
    assert_eq!(status, StatusCode::INTERNAL_SERVER_ERROR);
    assert!(!body.contains("/var/lib/secret"), "leaked: {body}");
}

#[tokio::test]
async fn unauthorized_text_is_not_product_specific() {
    let mut cfg = config(loopback());
    cfg.bearer_token = Some("tok".into());
    let router = legacy(&cfg);
    let mut req = Request::builder()
        .method(Method::POST)
        .uri("/tools/list")
        .header("content-type", "application/json")
        .body(Body::from("{}"))
        .unwrap();
    req.extensions_mut().insert(ConnectInfo(client()));
    let (status, body) = body_text(router.oneshot(req).await.unwrap()).await;
    assert_eq!(status, StatusCode::UNAUTHORIZED);
    assert!(!body.to_lowercase().contains("mentisdb"), "{body}");
}

// ── Streamable auth failures are reported ───────────────────────────────────

struct Recorder(std::sync::Mutex<Vec<String>>);

#[async_trait::async_trait]
impl mcp::McpEventHandler for Recorder {
    async fn on_mcp_event(&self, event: &mcp::McpEvent) {
        if let mcp::McpEvent::RequestRejected { reason, .. } = event {
            self.0.lock().unwrap().push(reason.clone());
        }
    }
}

#[tokio::test]
async fn streamable_rejections_emit_request_rejected() {
    let rec = Arc::new(Recorder(Default::default()));
    let mut cfg = config(loopback());
    cfg.bearer_token = Some("tok".into());
    cfg.event_handler = Some(rec.clone());
    let router = streamable(&cfg);
    assert_eq!(
        rpc_ping(&router, "localhost").await,
        StatusCode::UNAUTHORIZED
    );
    assert_eq!(
        rpc_ping(&router, "evil.example").await,
        StatusCode::FORBIDDEN
    );
    let seen = rec.0.lock().unwrap().clone();
    assert_eq!(seen, vec!["Unauthorized", "Host not allowed"]);
}

// ── SSE lag notice ──────────────────────────────────────────────────────────

#[tokio::test]
async fn sse_lag_notice_does_not_claim_stream_closed() {
    use mcp::streamable_http::SseBroadcaster;
    let b = SseBroadcaster::new(2);
    let router = streamable_http_router_with_sse(
        &config(loopback()),
        &StreamableHttpConfig::new("t", "0.0.1"),
        Arc::new(Echo),
        Some(b.clone()),
    );
    let mut req = Request::builder()
        .method(Method::GET)
        .uri("/")
        .body(Body::empty())
        .unwrap();
    req.extensions_mut().insert(ConnectInfo(client()));
    let resp = router.oneshot(req).await.unwrap();
    for i in 0..10 {
        b.send(mcp::SseMessage::data(serde_json::json!({ "n": i })));
    }
    let mut body = resp.into_body();
    let frame = tokio::time::timeout(
        std::time::Duration::from_millis(500),
        http_body_util::BodyExt::frame(&mut body),
    )
    .await
    .unwrap()
    .unwrap()
    .unwrap();
    let text = String::from_utf8_lossy(frame.data_ref().unwrap()).into_owned();
    assert!(text.contains("event: warning"), "{text}");
    assert!(!text.contains("closed"), "{text}");
    assert!(text.contains("lagged"), "{text}");
}
