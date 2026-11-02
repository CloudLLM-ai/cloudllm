//! The Gemini image endpoint must receive the API key in the
//! `x-goog-api-key` header, never in the URL (URLs end up in reqwest error
//! messages and logs).

use cloudllm::clients::gemini::GeminiClient;
use cloudllm::cloudllm::image_generation::{ImageGenerationClient, ImageGenerationOptions};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;

/// Spawn a one-shot server that records the request head and returns a canned
/// Gemini image response. Returns the base URL and the captured head.
async fn spawn_capture_server() -> (String, tokio::sync::oneshot::Receiver<String>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
    let addr = listener.local_addr().expect("addr");
    let (tx, rx) = tokio::sync::oneshot::channel();
    tokio::spawn(async move {
        let (mut sock, _) = listener.accept().await.expect("accept");
        let mut buf = vec![0u8; 16 * 1024];
        let n = sock.read(&mut buf).await.expect("read");
        let _ = tx.send(String::from_utf8_lossy(&buf[..n]).into_owned());
        let body = r#"{"candidates":[{"content":{"parts":[{"inlineData":{"mimeType":"image/png","data":"QUJD"}}]}}]}"#;
        let resp = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
            body.len(),
            body
        );
        let _ = sock.write_all(resp.as_bytes()).await;
    });
    (format!("http://{addr}"), rx)
}

/// The key is sent as a header and the request line carries no `key=` query.
#[tokio::test]
async fn test_gemini_image_key_is_sent_in_header_not_url() {
    let (base, head_rx) = spawn_capture_server().await;
    let client = GeminiClient::new_with_base_url("TEST-KEY-123", "gemini-2.5-flash-image", &base);

    let resp = client
        .generate_image("a cat", ImageGenerationOptions::default())
        .await
        .expect("image response");
    assert_eq!(resp.images.len(), 1);

    let head = head_rx.await.expect("captured request");
    let request_line = head.lines().next().expect("request line");
    assert!(request_line.contains(":generateContent"));
    assert!(
        !request_line.contains("key="),
        "key leaked in URL: {}",
        request_line
    );
    assert!(!request_line.contains("TEST-KEY-123"));
    assert!(head
        .to_ascii_lowercase()
        .contains("x-goog-api-key: test-key-123"));
}
