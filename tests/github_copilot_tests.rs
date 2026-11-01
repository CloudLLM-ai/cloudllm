//! Integration tests for the GitHub Copilot provider, run against a scripted
//! local HTTP server that stands in for github.com, api.github.com and the
//! Copilot API.

use cloudllm::client_wrapper::{ClientWrapper, Message, Role};
use cloudllm::clients::github_copilot::{
    CopilotAuthError, CopilotClient, CopilotIdentity, CopilotTokenProvider, GitHubDeviceFlow,
    GitHubOAuthToken,
};
use futures_util::StreamExt;
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime, UNIX_EPOCH};
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
        Self::start_with(|_| script).await
    }

    /// Start a server whose script is built from its own base URL, for
    /// replies that must point back at the server (a session token naming
    /// this server as the Copilot API host, say).
    async fn start_with(script: impl FnOnce(&str) -> Vec<Reply>) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
        let base_url = format!("http://{}", listener.local_addr().expect("address"));
        let script = script(&base_url);
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

/// Verifies that GitHub's answer to an unknown client ID (404 with a JSON
/// `error` that is not an OAuth code) is `HttpStatus`, not an OAuth error.
#[tokio::test]
async fn device_flow_reports_an_unknown_client_id_as_404() {
    let server = MockServer::start(vec![(404, r#"{"error":"Not Found"}"#.to_owned())]).await;

    let flow = GitHubDeviceFlow::new("unknown").with_github_base_url(&server.base_url);
    match flow.request_code().await {
        Err(CopilotAuthError::HttpStatus { status, .. }) => assert_eq!(status, 404),
        other => panic!("expected HttpStatus 404, got {:?}", other),
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

/// A session-token answer that expires `seconds_from_now` from now and points
/// at `api`.
fn session_token_reply(token: &str, seconds_from_now: u64, api: &str) -> Reply {
    let expires_at = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("clock after 1970")
        .as_secs()
        + seconds_from_now;
    (
        200,
        format!(
            r#"{{"token":"{token}","expires_at":{expires_at},"refresh_in":1500,"endpoints":{{"api":"{api}"}}}}"#
        ),
    )
}

/// Verifies that the GitHub token goes to the token endpoint with the
/// identity headers, and that a fresh session token is reused rather than
/// traded for again.
#[tokio::test]
async fn session_token_is_exchanged_once_and_cached() {
    let server = MockServer::start(vec![session_token_reply(
        "tid=one",
        1800,
        "https://api.individual.githubcopilot.com",
    )])
    .await;

    let provider = CopilotTokenProvider::from_github_token(GitHubOAuthToken::new("gho_abc"))
        .with_github_api_base_url(&server.base_url)
        .with_identity(CopilotIdentity::default().with_integration_id("cloudllm-test"));

    let first = provider.session_token().await.expect("first trade");
    let second = provider.session_token().await.expect("cached token");

    assert_eq!(first.secret(), "tid=one");
    assert_eq!(first, second);
    assert_eq!(
        first.api_base_url(),
        "https://api.individual.githubcopilot.com"
    );
    let requests = server.requests();
    assert_eq!(requests.len(), 1, "the second call must hit the cache");
    let head = &requests[0].head;
    assert!(head.starts_with("get /copilot_internal/v2/token "));
    assert!(head.contains("authorization: token gho_abc"));
    assert!(head.contains("editor-version: cloudllm/"));
    assert!(head.contains("copilot-integration-id: cloudllm-test"));
}

/// Verifies that a token inside the refresh margin is traded for a new one,
/// and that `invalidate` forces a trade even for a fresh token.
#[tokio::test]
async fn session_token_is_refreshed_near_expiry_and_after_invalidate() {
    let server = MockServer::start(vec![
        session_token_reply("tid=short", 30, "https://a"),
        session_token_reply("tid=long", 1800, "https://b"),
        session_token_reply("tid=after-invalidate", 1800, "https://c"),
    ])
    .await;

    let provider = CopilotTokenProvider::from_github_token(GitHubOAuthToken::new("gho"))
        .with_github_api_base_url(&server.base_url)
        .with_refresh_margin(Duration::from_secs(60));

    assert_eq!(
        provider.session_token().await.unwrap().secret(),
        "tid=short"
    );
    assert_eq!(provider.session_token().await.unwrap().secret(), "tid=long");
    assert_eq!(provider.session_token().await.unwrap().secret(), "tid=long");
    provider.invalidate().await;
    assert_eq!(
        provider.session_token().await.unwrap().secret(),
        "tid=after-invalidate"
    );
    assert_eq!(server.requests().len(), 3);
}

/// Verifies that GitHub refusing the trade (no Copilot seat) is reported as
/// `CopilotNotAvailable`, not as a generic HTTP error.
#[tokio::test]
async fn session_token_refusal_is_copilot_not_available() {
    let server = MockServer::start(vec![(403, r#"{"message":"no seat"}"#.to_owned())]).await;

    let provider = CopilotTokenProvider::from_github_token(GitHubOAuthToken::new("gho"))
        .with_github_api_base_url(&server.base_url);
    match provider.session_token().await {
        Err(CopilotAuthError::CopilotNotAvailable { status, body }) => {
            assert_eq!(status, 403);
            assert!(body.contains("no seat"));
        }
        other => panic!("expected CopilotNotAvailable, got {:?}", other),
    }
}

fn user(text: &str) -> Message {
    Message {
        role: Role::User,
        content: Arc::from(text),
        tool_calls: vec![],
    }
}

fn chat_reply(text: &str) -> Reply {
    (
        200,
        format!(
            r#"{{"choices":[{{"message":{{"role":"assistant","content":"{text}"}}}}],"usage":{{"prompt_tokens":7,"completion_tokens":2}}}}"#
        ),
    )
}

/// Verifies a chat round trip: the session token is traded for, then sent as
/// the bearer token to the API host the trade named, with the identity
/// headers and the model ID; the reply and usage come back.
#[tokio::test]
async fn copilot_client_chats_through_the_session_host() {
    // The token endpoint points the account back at the same mock server.
    let server = MockServer::start_with(|base| {
        vec![
            session_token_reply("tid=chat", 1800, base),
            chat_reply("hello from copilot"),
        ]
    })
    .await;

    let provider = Arc::new(
        CopilotTokenProvider::from_github_token(GitHubOAuthToken::new("gho"))
            .with_github_api_base_url(&server.base_url)
            .with_identity(CopilotIdentity::default().with_integration_id("cloudllm-test")),
    );
    let client = CopilotClient::new(provider, "gpt-4.1");
    let reply = client
        .send_message(&[user("hi")], None)
        .await
        .expect("chat should succeed");

    assert_eq!(reply.content.as_ref(), "hello from copilot");
    let usage = client.get_last_usage().await.expect("usage recorded");
    assert_eq!((usage.input_tokens, usage.output_tokens), (7, 2));
    let requests = server.requests();
    assert_eq!(requests.len(), 2);
    let chat = &requests[1];
    assert!(chat.head.starts_with("post /chat/completions "));
    assert!(chat.head.contains("authorization: bearer tid=chat"));
    assert!(chat.head.contains("copilot-integration-id: cloudllm-test"));
    assert!(chat.head.contains("editor-version: cloudllm/"));
    assert!(chat.body.contains(r#""model":"gpt-4.1""#));
}

/// Verifies that a 401 from the Copilot API discards the session token,
/// trades for a new one, and retries the request once with it.
#[tokio::test]
async fn copilot_client_refreshes_the_token_after_a_401() {
    let server = MockServer::start_with(|base| {
        vec![
            session_token_reply("tid=stale", 1800, base),
            (401, r#"{"error":"token expired"}"#.to_owned()),
            session_token_reply("tid=fresh", 1800, base),
            chat_reply("retried"),
        ]
    })
    .await;

    let provider = Arc::new(
        CopilotTokenProvider::from_github_token(GitHubOAuthToken::new("gho"))
            .with_github_api_base_url(&server.base_url),
    );
    let client = CopilotClient::new(provider, "gpt-4.1");
    let reply = client
        .send_message(&[user("hi")], None)
        .await
        .expect("the retry should succeed");

    assert_eq!(reply.content.as_ref(), "retried");
    let requests = server.requests();
    assert_eq!(requests.len(), 4);
    assert!(requests[1].head.contains("authorization: bearer tid=stale"));
    assert!(requests[2]
        .head
        .starts_with("get /copilot_internal/v2/token "));
    assert!(requests[3].head.contains("authorization: bearer tid=fresh"));
}

/// Verifies that a refused trade surfaces as `CopilotNotAvailable` through
/// the `ClientWrapper` error.
#[tokio::test]
async fn copilot_client_reports_a_missing_copilot_seat() {
    let server = MockServer::start(vec![(404, "Not Found".to_owned())]).await;

    let provider = Arc::new(
        CopilotTokenProvider::from_github_token(GitHubOAuthToken::new("gho"))
            .with_github_api_base_url(&server.base_url),
    );
    let client = CopilotClient::new(provider, "gpt-4.1");
    let error = match client.send_message(&[user("hi")], None).await {
        Ok(_) => panic!("no seat must fail"),
        Err(error) => error,
    };
    assert!(matches!(
        error.downcast_ref::<CopilotAuthError>(),
        Some(CopilotAuthError::CopilotNotAvailable { status: 404, .. })
    ));
}

/// Verifies that streaming sends the identity headers and yields the SSE
/// deltas in order.
#[tokio::test]
async fn copilot_client_streams_tokens() {
    let events = concat!(
        "data: {\"choices\":[{\"delta\":{\"content\":\"Hel\"}}]}\n\n",
        "data: {\"choices\":[{\"delta\":{\"content\":\"lo\"}}]}\n\n",
        "data: [DONE]\n\n"
    );
    let server = MockServer::start(vec![(200, events.to_owned())]).await;
    let provider = Arc::new(
        CopilotTokenProvider::with_static_token("tid=s", &server.base_url)
            .with_identity(CopilotIdentity::default().with_header("X-Initiator", "user")),
    );
    let client = CopilotClient::new(provider, "claude-sonnet-4");

    let messages = [user("hi")];
    let mut stream = client
        .send_message_stream(&messages, None)
        .await
        .expect("stream opens")
        .expect("Copilot streams");
    let mut text = String::new();
    while let Some(chunk) = stream.next().await {
        text.push_str(&chunk.expect("chunk").content);
    }

    assert_eq!(text, "Hello");
    let request = &server.requests()[0];
    assert!(request.head.contains("authorization: bearer tid=s"));
    assert!(request.head.contains("x-initiator: user"));
    assert!(request.body.contains(r#""stream":true"#));
}

/// Verifies that the model list is read from `GET /models` on the session
/// host.
#[tokio::test]
async fn list_models_reads_the_catalogue() {
    let server = MockServer::start(vec![(
        200,
        r#"{"data":[
            {"id":"gpt-4.1","vendor":"Azure OpenAI","capabilities":{"type":"chat","supports":{"tool_calls":true}}},
            {"id":"text-embedding-3-small","capabilities":{"type":"embeddings"}}
        ],"object":"list"}"#
            .to_owned(),
    )])
    .await;
    let provider = Arc::new(CopilotTokenProvider::with_static_token(
        "tid=m",
        &server.base_url,
    ));
    let client = CopilotClient::new(provider, "gpt-4.1");

    let models = client.list_models().await.expect("model list");
    let chat: Vec<&str> = models
        .iter()
        .filter(|model| model.is_chat())
        .map(|model| model.id.as_str())
        .collect();
    assert_eq!(chat, ["gpt-4.1"]);
    assert!(models[0].supports_tool_calls());
    assert!(server.requests()[0].head.starts_with("get /models "));
}
