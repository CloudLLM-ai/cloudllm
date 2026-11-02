//! Tests for the bounded response readers used by the LLM clients.
//!
//! A hostile or misconfigured endpoint must not be able to make the client
//! buffer an unbounded body in memory.

use cloudllm::clients::common::{read_body_text, read_error_body, BodyReadError};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;

/// Serve one raw HTTP response and return the client-side `reqwest::Response`.
async fn serve_once(raw_response: Vec<u8>) -> reqwest::Response {
    let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
    let addr = listener.local_addr().expect("addr");
    tokio::spawn(async move {
        let (mut sock, _) = listener.accept().await.expect("accept");
        let mut buf = [0u8; 1024];
        let _ = sock.read(&mut buf).await;
        // Ignore write errors: the client may hang up once it hits its cap.
        let _ = sock.write_all(&raw_response).await;
        let _ = sock.shutdown().await;
    });
    reqwest::Client::new()
        .get(format!("http://{addr}/"))
        .send()
        .await
        .expect("send")
}

fn with_content_length(body: &[u8]) -> Vec<u8> {
    let mut out = format!(
        "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        body.len()
    )
    .into_bytes();
    out.extend_from_slice(body);
    out
}

/// A body within the limit is returned verbatim.
#[tokio::test]
async fn test_read_body_text_returns_body_under_limit() {
    let resp = serve_once(with_content_length(b"hello world")).await;
    assert_eq!(
        read_body_text(resp, 1024).await.expect("body"),
        "hello world"
    );
}

/// A body whose `Content-Length` exceeds the limit is rejected up front.
#[tokio::test]
async fn test_read_body_text_rejects_declared_oversize() {
    let resp = serve_once(with_content_length(&vec![b'a'; 4096])).await;
    let err = read_body_text(resp, 1024).await.expect_err("too large");
    assert!(matches!(err, BodyReadError::TooLarge { limit: 1024 }));
}

/// A chunked body (no `Content-Length`) is cut off while streaming.
#[tokio::test]
async fn test_read_body_text_rejects_streamed_oversize() {
    let mut raw =
        b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\nConnection: close\r\n\r\n".to_vec();
    for _ in 0..8 {
        raw.extend_from_slice(b"200\r\n");
        raw.extend_from_slice(&vec![b'a'; 0x200]);
        raw.extend_from_slice(b"\r\n");
    }
    raw.extend_from_slice(b"0\r\n\r\n");
    let resp = serve_once(raw).await;
    let err = read_body_text(resp, 1024).await.expect_err("too large");
    assert!(matches!(err, BodyReadError::TooLarge { .. }));
}

/// Error bodies are truncated to the cap and marked, never rejected.
#[tokio::test]
async fn test_read_error_body_truncates_instead_of_failing() {
    let resp = serve_once(with_content_length(&vec![b'x'; 4096])).await;
    let text = read_error_body(resp, 100).await;
    assert_eq!(text.chars().filter(|c| *c == 'x').count(), 100);
    assert!(text.ends_with('…'));
}

/// A short error body is returned untouched.
#[tokio::test]
async fn test_read_error_body_keeps_short_body() {
    let resp = serve_once(with_content_length(b"bad request")).await;
    assert_eq!(read_error_body(resp, 100).await, "bad request");
}
