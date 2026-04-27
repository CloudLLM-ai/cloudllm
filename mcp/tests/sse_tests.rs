//! Integration tests for SSE (Server-Sent Events) support.
//!
//! These tests require the `server` feature to be enabled since they exercise
//! the axum-based HTTP router with SSE streaming capabilities.

#![cfg(feature = "server")]

use axum::body::Body;
use axum::extract::connect_info::ConnectInfo;
use axum::http::{Method, Request, StatusCode};

use mcp::events::{McpEvent, McpEventHandler};
use mcp::http::HttpServerConfig;
use mcp::streamable_http::{
    streamable_http_router_with_sse, SseBroadcaster, SseEventHandler, SseMessage,
    StreamableHttpConfig, CURRENT_MCP_PROTOCOL_VERSION,
};
use mcp::{IpFilter, ToolMetadata, ToolProtocol, ToolResult};
use serde_json::json;
use std::sync::Arc;
use std::time::Duration;
use tokio_stream::StreamExt;
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

fn client_addr() -> std::net::SocketAddr {
    std::net::SocketAddr::from(([127, 0, 0, 1], 12345))
}

async fn read_sse_stream_with_timeout(body: Body, timeout_duration: Duration) -> Vec<u8> {
    let mut buf = Vec::new();
    let deadline = tokio::time::Instant::now() + timeout_duration;
    let mut body_stream = body.into_data_stream();
    loop {
        if tokio::time::Instant::now() >= deadline {
            break;
        }
        match tokio::time::timeout(Duration::from_millis(200), body_stream.next()).await {
            Ok(Some(Ok(data))) => {
                buf.extend_from_slice(&data);
            }
            Ok(Some(Err(_))) | Ok(None) | Err(_) => break,
        }
    }
    buf
}

fn make_router_with_sse() -> (axum::Router, SseBroadcaster) {
    let broadcaster = SseBroadcaster::new(256);
    let config = StreamableHttpConfig::new("test-server", "0.1.0");
    let router = streamable_http_router_with_sse(
        &HttpServerConfig {
            addr: std::net::SocketAddr::from(([127, 0, 0, 1], 0)),
            bearer_token: None,
            ip_filter: IpFilter::new(),
            event_handler: None,
        },
        &config,
        Arc::new(EchoProtocol),
        Some(broadcaster.clone()),
    );
    (router, broadcaster)
}

fn make_router_without_sse() -> axum::Router {
    let config = StreamableHttpConfig::new("test-server", "0.1.0");
    mcp::streamable_http::streamable_http_router(
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
    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX).await.unwrap();
    (status, String::from_utf8(bytes.to_vec()).unwrap())
}

// ── SseMessage unit tests ─────────────────────────────────────────────────

#[test]
fn test_sse_message_data_only() {
    let msg = SseMessage::data(json!({"key": "value"}));
    let formatted = msg.format();
    assert!(formatted.contains("data: {\"key\":\"value\"}"));
    assert!(!formatted.contains("event:"));
    assert!(!formatted.contains("id:"));
}

#[test]
fn test_sse_message_with_event() {
    let msg = SseMessage::with_event("tools_list", json!({"tools": []}));
    let formatted = msg.format();
    assert!(formatted.starts_with("event: tools_list\n"));
    assert!(formatted.contains("data: {\"tools\":[]}"));
}

#[test]
fn test_sse_message_format_multiple_lines() {
    let msg = SseMessage::with_event("test_event", json!({"a": 1, "b": 2}));
    let formatted = msg.format();
    let lines: Vec<&str> = formatted.lines().collect();
    assert_eq!(lines[0], "event: test_event");
    assert!(lines[1].starts_with("data:"));
    assert_eq!(lines[2], "");
}

#[test]
fn test_sse_message_serialization() {
    let msg = SseMessage::data(json!({"hello": "world"}));
    let serialized = serde_json::to_string(&msg).unwrap();
    assert!(serialized.contains("\"data\""));
    assert!(serialized.contains("\"hello\":\"world\""));
}

#[test]
fn test_sse_message_with_event_serialization() {
    let msg = SseMessage::with_event("my_event", json!({"x": 42}));
    let serialized = serde_json::to_string(&msg).unwrap();
    assert!(serialized.contains("\"event\":\"my_event\""));
    assert!(serialized.contains("\"x\":42"));
}

// ── SseBroadcaster unit tests ─────────────────────────────────────────────

#[tokio::test]
async fn test_broadcaster_send_and_receive() {
    let broadcaster = SseBroadcaster::new(16);
    let mut rx = broadcaster.subscribe();

    let msg = SseMessage::data(json!({"test": true}));
    broadcaster.send(msg);

    let received = tokio::time::timeout(Duration::from_millis(100), rx.recv())
        .await
        .expect("timeout waiting for message")
        .expect("message received");

    assert_eq!(received.data, json!({"test": true}));
    assert!(received.event.is_none());
}

#[tokio::test]
async fn test_broadcaster_multiple_subscribers() {
    let broadcaster = SseBroadcaster::new(16);
    let mut rx1 = broadcaster.subscribe();
    let mut rx2 = broadcaster.subscribe();

    broadcaster.send(SseMessage::with_event("broadcast_test", json!({"n": 1})));

    let msg1 = tokio::time::timeout(Duration::from_millis(100), rx1.recv())
        .await
        .expect("timeout")
        .expect("message");
    let msg2 = tokio::time::timeout(Duration::from_millis(100), rx2.recv())
        .await
        .expect("timeout")
        .expect("message");

    assert_eq!(msg1.data, json!({"n": 1}));
    assert_eq!(msg2.data, json!({"n": 1}));
    assert_eq!(msg1.event, Some("broadcast_test".to_string()));
    assert_eq!(msg2.event, Some("broadcast_test".to_string()));
}

#[tokio::test]
async fn test_broadcaster_clone_shares_channel() {
    let broadcaster = SseBroadcaster::new(16);
    let cloned = broadcaster.clone();
    let mut rx = broadcaster.subscribe();

    cloned.send(SseMessage::data(json!({"from_clone": true})));

    let msg = tokio::time::timeout(Duration::from_millis(100), rx.recv())
        .await
        .expect("timeout")
        .expect("message");

    assert_eq!(msg.data, json!({"from_clone": true}));
}

#[tokio::test]
async fn test_broadcaster_mcp_event_conversion() {
    let broadcaster = SseBroadcaster::new(16);
    let mut rx = broadcaster.subscribe();

    let event = McpEvent::ToolCallCompleted {
        client_addr: "127.0.0.1".to_string(),
        tool_name: "echo".to_string(),
        success: true,
        error: None,
        duration_ms: 42,
    };
    broadcaster.broadcast_mcp_event(&event);

    let msg = tokio::time::timeout(Duration::from_millis(100), rx.recv())
        .await
        .expect("timeout")
        .expect("message");

    assert_eq!(msg.event, Some("mcp_event".to_string()));
    assert_eq!(msg.data["event"], "tool_call_completed");
    assert_eq!(msg.data["tool_name"], "echo");
    assert_eq!(msg.data["success"], true);
    assert_eq!(msg.data["duration_ms"], 42);
}

#[tokio::test]
async fn test_broadcaster_all_event_types() {
    let broadcaster = SseBroadcaster::new(64);
    let mut rx = broadcaster.subscribe();

    let events = vec![
        McpEvent::ServerStarted { addr: "127.0.0.1:8080".to_string() },
        McpEvent::ToolListRequested { client_addr: "10.0.0.1".to_string() },
        McpEvent::ToolListReturned { client_addr: "10.0.0.1".to_string(), tool_count: 5 },
        McpEvent::ToolCallReceived {
            client_addr: "10.0.0.1".to_string(),
            tool_name: "test".to_string(),
            parameters: json!({"x": 1}),
        },
        McpEvent::ToolCallCompleted {
            client_addr: "10.0.0.1".to_string(),
            tool_name: "test".to_string(),
            success: true,
            error: None,
            duration_ms: 100,
        },
        McpEvent::ToolError {
            source: "10.0.0.1".to_string(),
            tool_name: "test".to_string(),
            error: "fail".to_string(),
            duration_ms: 50,
        },
        McpEvent::RequestRejected {
            client_addr: "10.0.0.1".to_string(),
            reason: "bad token".to_string(),
        },
        McpEvent::ConnectionInitialized {
            endpoint: "http://remote".to_string(),
            tool_count: 3,
        },
        McpEvent::ConnectionClosed {
            endpoint: "http://remote".to_string(),
        },
        McpEvent::ToolsDiscovered {
            endpoint: "http://remote".to_string(),
            tool_count: 3,
            tool_names: vec!["a".to_string(), "b".to_string()],
        },
        McpEvent::CacheHit {
            endpoint: "http://remote".to_string(),
            tool_count: 2,
        },
        McpEvent::CacheExpired {
            endpoint: "http://remote".to_string(),
        },
        McpEvent::RemoteToolCallStarted {
            endpoint: "http://remote".to_string(),
            tool_name: "remote_tool".to_string(),
            parameters: json!({}),
        },
        McpEvent::RemoteToolCallCompleted {
            endpoint: "http://remote".to_string(),
            tool_name: "remote_tool".to_string(),
            success: true,
            error: None,
            duration_ms: 200,
        },
    ];

    for event in &events {
        broadcaster.broadcast_mcp_event(event);
    }

    for _ in 0..events.len() {
        let msg = tokio::time::timeout(Duration::from_millis(500), rx.recv())
            .await
            .expect("timeout waiting for event")
            .expect("message received");
        assert_eq!(msg.event, Some("mcp_event".to_string()));
        assert!(msg.data["event"].is_string());
    }
}

#[tokio::test]
async fn test_broadcaster_lagged_messages() {
    let broadcaster = SseBroadcaster::new(4);
    let mut rx = broadcaster.subscribe();

    for i in 0..10 {
        broadcaster.send(SseMessage::data(json!({"i": i})));
    }

    match rx.try_recv() {
        Ok(msg) => panic!("should have gotten lagged error, got message: {:?}", msg),
        Err(tokio::sync::broadcast::error::TryRecvError::Lagged(n)) => {
            assert_eq!(n, 6);
        }
        Err(tokio::sync::broadcast::error::TryRecvError::Closed) => {
            panic!("should not be closed");
        }
        Err(tokio::sync::broadcast::error::TryRecvError::Empty) => {
            panic!("should not be empty");
        }
    }

    for expected_i in 6..10 {
        match rx.try_recv() {
            Ok(msg) => assert_eq!(msg.data["i"], expected_i),
            Err(e) => panic!("expected message {}, got error: {:?}", expected_i, e),
        }
    }
}

// ── SseEventHandler tests ─────────────────────────────────────────────────

#[tokio::test]
async fn test_sse_event_handler_propagates_events() {
    let (handler, broadcaster) = SseEventHandler::new(16);
    let mut rx = broadcaster.subscribe();

    let event = McpEvent::ServerStarted {
        addr: "0.0.0.0:3000".to_string(),
    };
    handler.on_mcp_event(&event).await;

    let msg = tokio::time::timeout(Duration::from_millis(100), rx.recv())
        .await
        .expect("timeout")
        .expect("message");

    assert_eq!(msg.event, Some("mcp_event".to_string()));
    assert_eq!(msg.data["event"], "server_started");
    assert_eq!(msg.data["addr"], "0.0.0.0:3000");
}

#[tokio::test]
async fn test_sse_event_handler_multiple_events() {
    let (handler, broadcaster) = SseEventHandler::new(32);
    let mut rx = broadcaster.subscribe();

    handler.on_mcp_event(&McpEvent::ToolListRequested {
        client_addr: "1.2.3.4".to_string(),
    }).await;

    handler.on_mcp_event(&McpEvent::ToolListReturned {
        client_addr: "1.2.3.4".to_string(),
        tool_count: 10,
    }).await;

    let msg1 = tokio::time::timeout(Duration::from_millis(100), rx.recv())
        .await
        .expect("timeout")
        .expect("message");
    let msg2 = tokio::time::timeout(Duration::from_millis(100), rx.recv())
        .await
        .expect("timeout")
        .expect("message");

    assert_eq!(msg1.data["event"], "tool_list_requested");
    assert_eq!(msg2.data["event"], "tool_list_returned");
    assert_eq!(msg2.data["tool_count"], 10);
}

// ── HTTP GET handler SSE stream tests ─────────────────────────────────────

#[tokio::test]
async fn test_get_without_sse_returns_method_not_allowed() {
    let router = make_router_without_sse();
    let mut req = Request::builder()
        .method(Method::GET)
        .uri("/")
        .body(axum::body::Body::empty())
        .unwrap();
    req.extensions_mut().insert(ConnectInfo(client_addr()));
    let response = router.clone().oneshot(req).await.unwrap();
    assert_eq!(response.status(), StatusCode::METHOD_NOT_ALLOWED);
    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX).await.unwrap();
    let body: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert!(body["error"].as_str().unwrap().contains("SSE stream is not enabled"));
}

#[tokio::test]
async fn test_get_with_sse_returns_event_stream() {
    let (router, _broadcaster) = make_router_with_sse();

    let mut req = Request::builder()
        .method(Method::GET)
        .uri("/")
        .header("accept", "text/event-stream")
        .body(axum::body::Body::empty())
        .unwrap();
    req.extensions_mut().insert(ConnectInfo(client_addr()));

    let response = router.clone().oneshot(req).await.unwrap();
    assert_eq!(response.status(), StatusCode::OK);

    let content_type = response.headers().get("content-type").unwrap().to_str().unwrap();
    assert_eq!(content_type, "text/event-stream");

    let cache_control = response.headers().get("cache-control").unwrap().to_str().unwrap();
    assert_eq!(cache_control, "no-cache");

    let connection = response.headers().get("connection").unwrap().to_str().unwrap();
    assert_eq!(connection, "keep-alive");

    let mcp_version = response.headers().get("MCP-Protocol-Version").unwrap().to_str().unwrap();
    assert_eq!(mcp_version, CURRENT_MCP_PROTOCOL_VERSION);
}

#[tokio::test]
async fn test_sse_stream_receives_broadcast_messages() {
    let (router, broadcaster) = make_router_with_sse();

    let mut req = Request::builder()
        .method(Method::GET)
        .uri("/")
        .header("accept", "text/event-stream")
        .body(axum::body::Body::empty())
        .unwrap();
    req.extensions_mut().insert(ConnectInfo(client_addr()));

    let response = router.clone().oneshot(req).await.unwrap();
    assert_eq!(response.status(), StatusCode::OK);

    broadcaster.send(SseMessage::with_event(
        "test_ping",
        json!({"ping": true}),
    ));

    let body = response.into_body();
    let buf = read_sse_stream_with_timeout(body, Duration::from_secs(2)).await;

    let text = String::from_utf8_lossy(&buf);
    assert!(text.contains("event: test_ping"));
    assert!(text.contains("\"ping\":true"));
    assert!(text.starts_with("event:"));
    assert!(text.contains("data:"));
}

#[tokio::test]
async fn test_sse_stream_receives_multiple_messages() {
    let (router, broadcaster) = make_router_with_sse();

    let mut req = Request::builder()
        .method(Method::GET)
        .uri("/")
        .header("accept", "text/event-stream")
        .body(axum::body::Body::empty())
        .unwrap();
    req.extensions_mut().insert(ConnectInfo(client_addr()));

    let response = router.clone().oneshot(req).await.unwrap();
    assert_eq!(response.status(), StatusCode::OK);

    broadcaster.send(SseMessage::with_event("msg1", json!({"n": 1})));
    broadcaster.send(SseMessage::with_event("msg2", json!({"n": 2})));
    broadcaster.send(SseMessage::with_event("msg3", json!({"n": 3})));

    let body = response.into_body();
    let buf = read_sse_stream_with_timeout(body, Duration::from_secs(2)).await;
    let text = String::from_utf8_lossy(&buf);

    let event_count = text.matches("event:").count();
    assert_eq!(event_count, 3);

    assert!(text.contains("event: msg1"));
    assert!(text.contains("event: msg2"));
    assert!(text.contains("event: msg3"));
    assert!(text.contains("\"n\":1"));
    assert!(text.contains("\"n\":2"));
    assert!(text.contains("\"n\":3"));
}

#[tokio::test]
async fn test_sse_stream_mcp_event_from_post_request() {
    let (router, broadcaster) = make_router_with_sse();
    let mut rx = broadcaster.subscribe();

    let body = json!({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/list",
        "params": {}
    });
    let (status, _) = post_json(&router, body).await;
    assert_eq!(status, StatusCode::OK);

    let msg = tokio::time::timeout(Duration::from_millis(500), rx.recv())
        .await
        .expect("timeout waiting for SSE event")
        .expect("message received");

    assert_eq!(msg.event, Some("tools_listed".to_string()));
    assert!(msg.data["result"]["tools"].is_array());
}

#[tokio::test]
async fn test_sse_stream_error_event_from_failed_request() {
    let (router, broadcaster) = make_router_with_sse();
    let mut rx = broadcaster.subscribe();

    let body = json!({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "nonexistent/method",
        "params": {}
    });
    let (status, _) = post_json(&router, body).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);

    let msg = tokio::time::timeout(Duration::from_millis(500), rx.recv())
        .await
        .expect("timeout waiting for SSE error event")
        .expect("message received");

    assert_eq!(msg.event, Some("error".to_string()));
    assert_eq!(msg.data["error"]["code"], -32601);
    assert!(msg.data["error"]["message"].as_str().unwrap().contains("Method not found"));
}

#[tokio::test]
async fn test_sse_stream_tool_call_event() {
    let (router, broadcaster) = make_router_with_sse();
    let mut rx = broadcaster.subscribe();

    let body = json!({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {
            "name": "echo",
            "arguments": {"message": "hello"}
        }
    });
    let (status, _) = post_json(&router, body).await;
    assert_eq!(status, StatusCode::OK);

    let msg = tokio::time::timeout(Duration::from_millis(500), rx.recv())
        .await
        .expect("timeout waiting for SSE event")
        .expect("message received");

    assert_eq!(msg.event, Some("tool_called".to_string()));
    assert_eq!(msg.data["method"], "tools/call");
}

#[tokio::test]
async fn test_sse_stream_initialize_event() {
    let (router, broadcaster) = make_router_with_sse();
    let mut rx = broadcaster.subscribe();

    let body = json!({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {}
    });
    let (status, _) = post_json(&router, body).await;
    assert_eq!(status, StatusCode::OK);

    let msg = tokio::time::timeout(Duration::from_millis(500), rx.recv())
        .await
        .expect("timeout waiting for SSE event")
        .expect("message received");

    assert_eq!(msg.event, Some("initialized".to_string()));
    assert_eq!(msg.data["result"]["protocolVersion"], "2025-11-25");
}

#[tokio::test]
async fn test_sse_stream_ping_event() {
    let (router, broadcaster) = make_router_with_sse();
    let mut rx = broadcaster.subscribe();

    let body = json!({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "ping",
        "params": {}
    });
    let (status, _) = post_json(&router, body).await;
    assert_eq!(status, StatusCode::OK);

    let msg = tokio::time::timeout(Duration::from_millis(500), rx.recv())
        .await
        .expect("timeout waiting for SSE event")
        .expect("message received");

    assert_eq!(msg.event, Some("ping".to_string()));
}

// ── SSE format compliance tests ───────────────────────────────────────────

#[test]
fn test_sse_format_contains_double_newline_terminator() {
    let msg = SseMessage::data(json!({"test": true}));
    let formatted = msg.format();
    assert!(formatted.ends_with("\n\n"));
}

#[test]
fn test_sse_format_event_line_prefix() {
    let msg = SseMessage::with_event("my_type", json!({}));
    let formatted = msg.format();
    assert!(formatted.starts_with("event: my_type\n"));
}

#[test]
fn test_sse_format_data_line_prefix() {
    let msg = SseMessage::data(json!({"key": "val"}));
    let formatted = msg.format();
    assert!(formatted.contains("data: "));
}

#[test]
fn test_sse_format_valid_json_in_data() {
    let msg = SseMessage::with_event("test", json!({"arr": [1,2,3], "obj": {"a": "b"}}));
    let formatted = msg.format();
    let data_line = formatted.lines().find(|l| l.starts_with("data: ")).expect("data line");
    let data_line = data_line.strip_prefix("data: ").expect("prefix");
    let parsed: serde_json::Value = serde_json::from_str(data_line).expect("data should be valid JSON");
    assert_eq!(parsed["arr"][0], 1);
    assert_eq!(parsed["obj"]["a"], "b");
}

// ── Concurrent subscriber tests ───────────────────────────────────────────

#[tokio::test]
async fn test_concurrent_subscribers_receive_same_messages() {
    let broadcaster = SseBroadcaster::new(64);
    let mut rx1 = broadcaster.subscribe();
    let mut rx2 = broadcaster.subscribe();
    let mut rx3 = broadcaster.subscribe();

    for i in 0..5 {
        broadcaster.send(SseMessage::with_event("concurrent", json!({"seq": i})));
    }

    for rx in [&mut rx1, &mut rx2, &mut rx3] {
        for expected_i in 0..5 {
            let msg = tokio::time::timeout(Duration::from_millis(500), rx.recv())
                .await
                .expect("timeout")
                .expect("message");
            assert_eq!(msg.data["seq"], expected_i);
            assert_eq!(msg.event, Some("concurrent".to_string()));
        }
    }
}

#[tokio::test]
async fn test_subscriber_late_join_does_not_miss_new_messages() {
    let broadcaster = SseBroadcaster::new(64);

    broadcaster.send(SseMessage::data(json!({"early": true})));

    let mut rx = broadcaster.subscribe();

    broadcaster.send(SseMessage::data(json!({"late": true})));

    let msg = tokio::time::timeout(Duration::from_millis(100), rx.recv())
        .await
        .expect("timeout")
        .expect("message");

    assert_eq!(msg.data, json!({"late": true}));
}

// ── SseMessage edge case tests ────────────────────────────────────────────

#[test]
fn test_sse_message_empty_object() {
    let msg = SseMessage::data(json!({}));
    let formatted = msg.format();
    assert!(formatted.contains("data: {}"));
}

#[test]
fn test_sse_message_null_value() {
    let msg = SseMessage::data(serde_json::Value::Null);
    let formatted = msg.format();
    assert!(formatted.contains("data: null"));
}

#[test]
fn test_sse_message_array_value() {
    let msg = SseMessage::data(json!([1, "two", true, null]));
    let formatted = msg.format();
    assert!(formatted.contains("data: [1,\"two\",true,null]"));
}

#[test]
fn test_sse_message_nested_object() {
    let msg = SseMessage::with_event(
        "nested",
        json!({"outer": {"inner": {"deep": "value"}}}),
    );
    let formatted = msg.format();
    assert!(formatted.contains("event: nested"));
    assert!(formatted.contains("\"deep\":\"value\""));
}

#[test]
fn test_sse_message_special_characters_in_event() {
    let msg = SseMessage::with_event("tool_call/completed", json!({}));
    let formatted = msg.format();
    assert!(formatted.contains("event: tool_call/completed"));
}

#[test]
fn test_sse_message_unicode_data() {
    let msg = SseMessage::data(json!({"text": "hello \u{4e16}\u{754c}"}));
    let formatted = msg.format();
    assert!(formatted.contains("hello"));
}

// ── SseBroadcaster edge case tests ────────────────────────────────────────

#[tokio::test]
async fn test_broadcaster_minimum_buffer() {
    let broadcaster = SseBroadcaster::new(1);
    let mut rx = broadcaster.subscribe();

    broadcaster.send(SseMessage::data(json!({"test": true})));

    let msg = tokio::time::timeout(Duration::from_millis(100), rx.recv())
        .await
        .expect("timeout")
        .expect("message");

    assert_eq!(msg.data, json!({"test": true}));
}

#[tokio::test]
async fn test_broadcaster_large_buffer() {
    let broadcaster = SseBroadcaster::new(10000);
    let mut rx = broadcaster.subscribe();

    for i in 0..100 {
        broadcaster.send(SseMessage::data(json!({"i": i})));
    }

    for expected_i in 0..100 {
        let msg = tokio::time::timeout(Duration::from_millis(100), rx.recv())
            .await
            .expect("timeout")
            .expect("message");
        assert_eq!(msg.data["i"], expected_i);
    }
}

#[tokio::test]
async fn test_broadcaster_send_after_all_dropped() {
    let broadcaster = SseBroadcaster::new(16);
    {
        let _rx = broadcaster.subscribe();
    }

    broadcaster.send(SseMessage::data(json!({"orphan": true})));
}

// ── SseEventHandler edge case tests ───────────────────────────────────────

#[tokio::test]
async fn test_sse_event_handler_handler_is_send_sync() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<SseEventHandler>();
}

#[tokio::test]
async fn test_sse_event_handler_broadcaster_is_clone() {
    let (_, broadcaster) = SseEventHandler::new(16);
    let _cloned = broadcaster.clone();
}

// ── Integration: POST triggers SSE broadcast ──────────────────────────────

#[tokio::test]
async fn test_post_tools_call_broadcasts_sse_event() {
    let (router, broadcaster) = make_router_with_sse();
    let mut rx = broadcaster.subscribe();

    let body = json!({
        "jsonrpc": "2.0",
        "id": 42,
        "method": "tools/call",
        "params": {
            "name": "echo",
            "arguments": {"payload": "test_data"}
        }
    });
    let (status, resp_text) = post_json(&router, body).await;
    assert_eq!(status, StatusCode::OK);

    let resp: serde_json::Value = serde_json::from_str(&resp_text).unwrap();
    assert_eq!(resp["id"], 42);

    let sse_msg = tokio::time::timeout(Duration::from_millis(500), rx.recv())
        .await
        .expect("timeout")
        .expect("message");

    assert_eq!(sse_msg.event, Some("tool_called".to_string()));
    assert_eq!(sse_msg.data["method"], "tools/call");
    assert!(sse_msg.data["client_addr"].as_str().unwrap().contains("127.0.0.1"));
}

#[tokio::test]
async fn test_post_unauthorized_still_broadcasts_error() {
    let (router, broadcaster) = make_router_with_sse();
    let mut rx = broadcaster.subscribe();

    let body = json!({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "nonexistent/method",
        "params": {}
    });
    let (status, _) = post_json(&router, body).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);

    let sse_msg = tokio::time::timeout(Duration::from_millis(500), rx.recv())
        .await
        .expect("timeout")
        .expect("message");

    assert_eq!(sse_msg.event, Some("error".to_string()));
    assert_eq!(sse_msg.data["error"]["code"], -32601);
}

#[tokio::test]
async fn test_post_resources_list_broadcasts() {
    let (router, broadcaster) = make_router_with_sse();
    let mut rx = broadcaster.subscribe();

    let body = json!({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "resources/list",
        "params": {}
    });
    let (status, _) = post_json(&router, body).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);

    let sse_msg = tokio::time::timeout(Duration::from_millis(500), rx.recv())
        .await
        .expect("timeout")
        .expect("message");

    assert_eq!(sse_msg.event, Some("error".to_string()));
}
