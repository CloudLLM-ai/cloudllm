//! Tests for [`send_with_native_tools_and_headers`], the header-aware variant
//! of the blocking Chat Completions helper that provider wrappers such as the
//! GitHub Copilot client build on.

use cloudllm::client_wrapper::{Message, Role};
use cloudllm::clients::common::{get_shared_http_client, send_with_native_tools_and_headers};
use std::sync::Arc;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;
use tokio::sync::Mutex;

/// Accept one connection, return the raw request text, and answer with a
/// minimal Chat Completions body.
async fn serve_one(listener: TcpListener) -> String {
    let (mut socket, _) = listener.accept().await.expect("request should connect");
    let mut buffer = vec![0u8; 16 * 1024];
    let read = socket.read(&mut buffer).await.expect("request is readable");
    let body = r#"{"choices":[{"message":{"role":"assistant","content":"pong"}}],"usage":{"prompt_tokens":3,"completion_tokens":1}}"#;
    let response = format!(
        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
        body.len(),
        body
    );
    socket
        .write_all(response.as_bytes())
        .await
        .expect("response is writable");
    String::from_utf8_lossy(&buffer[..read]).into_owned()
}

/// Verifies that caller-supplied headers reach the wire next to the bearer
/// token, that an empty tool list sends no `tools` array, and that usage is
/// recorded.
#[tokio::test]
async fn extra_headers_are_sent_and_empty_tools_are_omitted() {
    let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
    let address = listener.local_addr().expect("address");
    let server = tokio::spawn(serve_one(listener));

    let usage = Mutex::new(None);
    let reply = send_with_native_tools_and_headers(
        &format!("http://{}/v1", address),
        "secret-token",
        "test-model",
        &[Message {
            role: Role::User,
            content: Arc::from("ping"),
            tool_calls: vec![],
        }],
        &[],
        &[("Editor-Version", "CloudLLM/test"), ("X-Custom", "yes")],
        get_shared_http_client(),
        &usage,
    )
    .await
    .expect("request should succeed");

    let request = server.await.expect("server task").to_ascii_lowercase();
    assert_eq!(reply.content.as_ref(), "pong");
    assert!(request.starts_with("post /v1/chat/completions "));
    assert!(request.contains("authorization: bearer secret-token"));
    assert!(request.contains("editor-version: cloudllm/test"));
    assert!(request.contains("x-custom: yes"));
    assert!(
        !request.contains("\"tools\""),
        "empty tools must be omitted"
    );
    let recorded = usage.lock().await.clone().expect("usage recorded");
    assert_eq!(recorded.input_tokens, 3);
    assert_eq!(recorded.output_tokens, 1);
}
