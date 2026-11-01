//! Security regression tests for `McpClientProtocol` against hostile servers.

#![cfg(feature = "server")]

use axum::http::{header, StatusCode};
use axum::response::IntoResponse;
use axum::routing::post;
use axum::Router;
use mcp::{McpClientProtocol, ToolProtocol};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

async fn serve(app: Router) -> String {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, app).await });
    format!("http://{addr}")
}

#[tokio::test]
async fn client_rejects_oversized_tool_list_response() {
    // ~600 KiB of valid JSON: parses fine unless the client caps the body.
    let big = format!(
        "{{\"tools\":[{{\"name\":\"t\",\"description\":\"{}\",\"parameters\":[],\"protocol_metadata\":{{}}}}]}}",
        "A".repeat(600 * 1024)
    );
    let app = Router::new().route(
        "/tools/list",
        post(move || {
            let big = big.clone();
            async move { ([(header::CONTENT_TYPE, "application/json")], big) }
        }),
    );
    let endpoint = serve(app).await;
    let client = McpClientProtocol::new(endpoint).with_max_response_bytes(64 * 1024);
    let err = client.list_tools().await.unwrap_err().to_string();
    assert!(err.contains("exceeds"), "unexpected error: {err}");
}

#[tokio::test]
async fn client_rejects_oversized_chunked_response_without_content_length() {
    let app = Router::new().route(
        "/tools/execute",
        post(|| async {
            let chunks = (0..64).map(|_| Ok::<_, std::io::Error>(vec![b' '; 4096]));
            let body = axum::body::Body::from_stream(futures::stream::iter(chunks));
            ([(header::CONTENT_TYPE, "application/json")], body)
        }),
    );
    let endpoint = serve(app).await;
    let client = McpClientProtocol::new(endpoint).with_max_response_bytes(16 * 1024);
    let err = client
        .execute("x", serde_json::json!({}))
        .await
        .unwrap_err()
        .to_string();
    assert!(err.contains("exceeds"), "unexpected error: {err}");
}

#[tokio::test]
async fn client_does_not_follow_redirects() {
    let hits = Arc::new(AtomicUsize::new(0));
    let hits_target = hits.clone();
    let target = Router::new().route(
        "/internal",
        post(move || {
            let hits = hits_target.clone();
            async move {
                hits.fetch_add(1, Ordering::SeqCst);
                (StatusCode::OK, "{\"tools\":[]}")
            }
        }),
    );
    let target_url = serve(target).await;

    let location = format!("{target_url}/internal");
    let redirector = Router::new().route(
        "/tools/list",
        post(move || {
            let location = location.clone();
            async move {
                (
                    StatusCode::TEMPORARY_REDIRECT,
                    [(header::LOCATION, location)],
                )
                    .into_response()
            }
        }),
    );
    let endpoint = serve(redirector).await;
    let client = McpClientProtocol::new(endpoint);
    assert!(client.list_tools().await.is_err());
    assert_eq!(hits.load(Ordering::SeqCst), 0, "client followed a redirect");
}
