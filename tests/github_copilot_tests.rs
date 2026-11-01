//! Integration tests for the GitHub Copilot provider, run against a scripted
//! local HTTP server that stands in for github.com, api.github.com and the
//! Copilot API.

use cloudllm::clients::github_copilot::{CopilotAuthError, GitHubDeviceFlow};
use std::sync::{Arc, Mutex};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;

/// One canned answer: HTTP status and JSON body.
type Reply = (u16, String);

/// A recorded request: the request line, headers and body, lower-cased only in
/// the head so bodies stay comparable.
#[derive(Debug, Clone)]
struct Recorded {
    head: String,
    body: String,
}

/// A local HTTP server that answers each connection with the next reply in
/// its script and records what it received.
struct MockServer {
    base_url: String,
    requests: Arc<Mutex<Vec<Recorded>>>,
    task: tokio::task::JoinHandle<()>,
}

impl MockServer {
    /// Start a server that plays `script` in order, one reply per request.
    async fn start(script: Vec<Reply>) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
        let base_url = format!("http://{}", listener.local_addr().expect("address"));
        let requests = Arc::new(Mutex::new(Vec::new()));
        let recorded = Arc::clone(&requests);
        let task = tokio::spawn(async move {
            for (status, body) in script {
                let (mut socket, _) = listener.accept().await.expect("accept");
                let request = read_request(&mut socket).await;
                recorded.lock().expect("lock").push(request);
                let response = format!(
                    "HTTP/1.1 {status} X\r\nContent-Type: application/json\r\n\
                     Content-Length: {}\r\nConnection: close\r\n\r\n{body}",
                    body.len()
                );
                socket.write_all(response.as_bytes()).await.expect("write");
            }
        });
        Self {
            base_url,
            requests,
            task,
        }
    }

    /// Everything the server has received so far.
    fn requests(&self) -> Vec<Recorded> {
        self.requests.lock().expect("lock").clone()
    }
}

impl Drop for MockServer {
    fn drop(&mut self) {
        self.task.abort();
    }
}

/// Read one HTTP/1.1 request (head plus `Content-Length` body).
async fn read_request(socket: &mut tokio::net::TcpStream) -> Recorded {
    let mut data = Vec::new();
    let mut chunk = [0u8; 4096];
    loop {
        let read = socket.read(&mut chunk).await.expect("read");
        data.extend_from_slice(&chunk[..read]);
        let text = String::from_utf8_lossy(&data).into_owned();
        if let Some(split) = text.find("\r\n\r\n") {
            let head = text[..split].to_ascii_lowercase();
            let length = head
                .lines()
                .find_map(|line| line.strip_prefix("content-length:"))
                .and_then(|value| value.trim().parse::<usize>().ok())
                .unwrap_or(0);
            let body = &text[split + 4..];
            if body.len() >= length || read == 0 {
                return Recorded {
                    head,
                    body: body.to_owned(),
                };
            }
        } else if read == 0 {
            return Recorded {
                head: text,
                body: String::new(),
            };
        }
    }
}

fn device_code_reply() -> Reply {
    (
        200,
        r#"{"device_code":"dev-123","user_code":"WDJB-MJHT","verification_uri":"https://github.com/login/device","expires_in":900,"interval":0}"#.to_owned(),
    )
}

/// Verifies the whole device-flow login: the code request, one pending poll,
/// then the token, with the client ID and grant type on the wire.
#[tokio::test]
async fn device_flow_polls_until_the_user_approves() {
    let server = MockServer::start(vec![
        device_code_reply(),
        (200, r#"{"error":"authorization_pending"}"#.to_owned()),
        (
            200,
            r#"{"access_token":"gho_token","token_type":"bearer","scope":"read:user"}"#.to_owned(),
        ),
    ])
    .await;

    let flow = GitHubDeviceFlow::new("my-client-id").with_github_base_url(&server.base_url);
    let mut shown = None;
    let token = flow
        .authorize(|code| shown = Some(code.user_code.clone()))
        .await
        .expect("login should succeed");

    assert_eq!(token.secret(), "gho_token");
    assert_eq!(shown.as_deref(), Some("WDJB-MJHT"));
    let requests = server.requests();
    assert_eq!(requests.len(), 3);
    assert!(requests[0].head.starts_with("post /login/device/code "));
    assert!(requests[0].body.contains("client_id=my-client-id"));
    assert!(requests[0].body.contains("scope=read%3Auser"));
    assert!(requests[1]
        .head
        .starts_with("post /login/oauth/access_token "));
    assert!(requests[1].body.contains("device_code=dev-123"));
    assert!(requests[1]
        .body
        .contains("grant_type=urn%3Aietf%3Aparams%3Aoauth%3Agrant-type%3Adevice_code"));
}

/// Verifies that a user who declines ends the login with `AccessDenied`.
#[tokio::test]
async fn device_flow_reports_a_declined_login() {
    let server = MockServer::start(vec![
        device_code_reply(),
        (200, r#"{"error":"access_denied"}"#.to_owned()),
    ])
    .await;

    let flow = GitHubDeviceFlow::new("id").with_github_base_url(&server.base_url);
    let result = flow.authorize(|_| {}).await;
    assert!(matches!(result, Err(CopilotAuthError::AccessDenied)));
}

/// Verifies that an unknown client ID surfaces GitHub's OAuth error code
/// instead of a generic HTTP failure.
#[tokio::test]
async fn device_flow_surfaces_oauth_errors_from_the_code_request() {
    let server = MockServer::start(vec![(
        400,
        r#"{"error":"incorrect_client_credentials","error_description":"bad id"}"#.to_owned(),
    )])
    .await;

    let flow = GitHubDeviceFlow::new("nope").with_github_base_url(&server.base_url);
    match flow.request_code().await {
        Err(CopilotAuthError::OAuth { error, description }) => {
            assert_eq!(error, "incorrect_client_credentials");
            assert_eq!(description.as_deref(), Some("bad id"));
        }
        other => panic!("expected an OAuth error, got {:?}", other),
    }
}

/// Verifies that a non-JSON error page becomes `HttpStatus` with the body.
#[tokio::test]
async fn device_flow_reports_http_failures() {
    let server = MockServer::start(vec![(503, "upstream down".to_owned())]).await;

    let flow = GitHubDeviceFlow::new("id").with_github_base_url(&server.base_url);
    match flow.request_code().await {
        Err(CopilotAuthError::HttpStatus { status, body }) => {
            assert_eq!(status, 503);
            assert_eq!(body, "upstream down");
        }
        other => panic!("expected HttpStatus, got {:?}", other),
    }
}
