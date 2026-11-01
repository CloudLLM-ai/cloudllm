//! Copilot session tokens: the short-lived credential the Copilot API accepts,
//! and the headers that identify your program to it.
//!
//! The Copilot API does not take the GitHub OAuth token from a device-flow
//! login directly. A client first trades it at
//! `GET https://api.github.com/copilot_internal/v2/token` for a **session
//! token**, which
//!
//! - lasts about 30 minutes,
//! - names the API host this account must use (individual, business and
//!   enterprise plans are served from different hosts), and
//! - is refused once the account loses its Copilot seat.
//!
//! [`CopilotTokenProvider`] does the trade, caches the result, and trades again
//! shortly before the session token expires, so callers just ask it for a
//! token on every request. One provider can serve any number of
//! `CopilotClient`s at once; put it in an [`Arc`](std::sync::Arc).
//!
//! # Example
//!
//! ```rust,no_run
//! use cloudllm::clients::github_copilot::{CopilotTokenProvider, GitHubOAuthToken};
//!
//! # #[tokio::main]
//! # async fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // A token saved from an earlier device-flow login.
//! let github_token = GitHubOAuthToken::new(std::env::var("GITHUB_COPILOT_OAUTH_TOKEN")?);
//! let provider = CopilotTokenProvider::from_github_token(github_token);
//!
//! let session = provider.session_token().await?;
//! println!("this account uses {}", session.api_base_url());
//! # Ok(())
//! # }
//! ```

use super::device_flow::GitHubOAuthToken;
use super::error::CopilotAuthError;
use crate::clients::common::get_shared_http_client;
use serde::Deserialize;
use std::fmt;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tokio::sync::Mutex;

/// Where GitHub serves its REST API, including the Copilot token endpoint.
pub const GITHUB_API_DEFAULT_BASE_URL: &str = "https://api.github.com";

/// The Copilot API host used when the token endpoint does not name one.
pub const COPILOT_DEFAULT_API_BASE_URL: &str = "https://api.githubcopilot.com";

/// How long before expiry [`CopilotTokenProvider`] trades for a new session
/// token, so a request never leaves with a token about to lapse.
pub const DEFAULT_REFRESH_MARGIN: Duration = Duration::from_secs(120);

/// The headers that tell the Copilot API which program is calling.
///
/// Copilot expects every request to name an editor and, for chat, an
/// integration. The defaults identify CloudLLM honestly:
///
/// | Header | Default |
/// |---|---|
/// | `Editor-Version` | `CloudLLM/<crate version>` |
/// | `Editor-Plugin-Version` | `cloudllm/<crate version>` |
/// | `User-Agent` | `CloudLLM/<crate version>` |
/// | `Copilot-Integration-Id` | not sent |
///
/// GitHub decides which integrations it accepts, and the Copilot API may
/// refuse chat requests that carry no `Copilot-Integration-Id`. Set the one
/// GitHub assigned to your integration with
/// [`CopilotIdentity::with_integration_id`]. Any other header the API wants
/// goes in with [`CopilotIdentity::with_header`].
///
/// # Examples
///
/// ```rust
/// use cloudllm::clients::github_copilot::CopilotIdentity;
///
/// let identity = CopilotIdentity::default()
///     .with_integration_id("my-integration")
///     .with_header("Openai-Intent", "conversation-panel");
///
/// let headers = identity.headers();
/// assert!(headers.contains(&("Copilot-Integration-Id", "my-integration")));
/// assert!(headers.contains(&("Openai-Intent", "conversation-panel")));
/// assert!(headers.iter().any(|(name, value)| *name == "Editor-Version" && value.starts_with("CloudLLM/")));
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CopilotIdentity {
    /// Sent as `Editor-Version`, for example `"CloudLLM/0.16.0"`.
    pub editor_version: String,
    /// Sent as `Editor-Plugin-Version`, for example `"cloudllm/0.16.0"`.
    pub editor_plugin_version: String,
    /// Sent as `User-Agent`.
    pub user_agent: String,
    /// Sent as `Copilot-Integration-Id` when set.
    pub integration_id: Option<String>,
    /// More headers to send on every Copilot request, as `(name, value)`.
    pub extra_headers: Vec<(String, String)>,
}

impl Default for CopilotIdentity {
    fn default() -> Self {
        let version = env!("CARGO_PKG_VERSION");
        Self {
            editor_version: format!("CloudLLM/{version}"),
            editor_plugin_version: format!("cloudllm/{version}"),
            user_agent: format!("CloudLLM/{version}"),
            integration_id: None,
            extra_headers: Vec::new(),
        }
    }
}

impl CopilotIdentity {
    /// Set `Editor-Version`.
    pub fn with_editor_version(self, value: impl Into<String>) -> Self {
        Self {
            editor_version: value.into(),
            ..self
        }
    }

    /// Set `Editor-Plugin-Version`.
    pub fn with_editor_plugin_version(self, value: impl Into<String>) -> Self {
        Self {
            editor_plugin_version: value.into(),
            ..self
        }
    }

    /// Set `User-Agent`.
    pub fn with_user_agent(self, value: impl Into<String>) -> Self {
        Self {
            user_agent: value.into(),
            ..self
        }
    }

    /// Send `Copilot-Integration-Id: <value>`.
    pub fn with_integration_id(self, value: impl Into<String>) -> Self {
        Self {
            integration_id: Some(value.into()),
            ..self
        }
    }

    /// Add one more header to every Copilot request.
    pub fn with_header(self, name: impl Into<String>, value: impl Into<String>) -> Self {
        let extra_headers = self
            .extra_headers
            .into_iter()
            .chain(std::iter::once((name.into(), value.into())))
            .collect();
        Self {
            extra_headers,
            ..self
        }
    }

    /// Every header this identity sends, as `(name, value)` pairs.
    pub fn headers(&self) -> Vec<(&str, &str)> {
        let fixed = vec![
            ("Editor-Version", self.editor_version.as_str()),
            ("Editor-Plugin-Version", self.editor_plugin_version.as_str()),
            ("User-Agent", self.user_agent.as_str()),
        ];
        let integration = self
            .integration_id
            .as_deref()
            .map(|id| ("Copilot-Integration-Id", id));
        let extra = self
            .extra_headers
            .iter()
            .map(|(name, value)| (name.as_str(), value.as_str()));
        fixed.into_iter().chain(integration).chain(extra).collect()
    }
}

/// A Copilot session token and the API host it is good for.
///
/// Get one from [`CopilotTokenProvider::session_token`]. `Debug` output hides
/// the token.
///
/// # Examples
///
/// ```rust
/// use cloudllm::clients::github_copilot::CopilotSessionToken;
/// use std::time::{Duration, SystemTime, UNIX_EPOCH};
///
/// let expires = UNIX_EPOCH + Duration::from_secs(2_000_000_000);
/// let token = CopilotSessionToken::new("tid=abc", "https://api.githubcopilot.com", Some(expires));
///
/// assert!(token.is_valid_at(UNIX_EPOCH + Duration::from_secs(1_999_999_000), Duration::from_secs(60)));
/// assert!(!token.is_valid_at(UNIX_EPOCH + Duration::from_secs(1_999_999_990), Duration::from_secs(60)));
/// ```
#[derive(Clone, PartialEq, Eq)]
pub struct CopilotSessionToken {
    token: String,
    api_base_url: String,
    expires_at: Option<SystemTime>,
}

impl CopilotSessionToken {
    /// Build a session token by hand. `expires_at` of `None` means it never
    /// expires as far as CloudLLM can tell. A trailing slash on
    /// `api_base_url` is ignored.
    pub fn new(
        token: impl Into<String>,
        api_base_url: impl AsRef<str>,
        expires_at: Option<SystemTime>,
    ) -> Self {
        Self {
            token: token.into(),
            api_base_url: api_base_url.as_ref().trim_end_matches('/').to_owned(),
            expires_at,
        }
    }

    /// The token itself, sent as `Authorization: Bearer <token>`.
    pub fn secret(&self) -> &str {
        &self.token
    }

    /// The Copilot API host for this account, without a trailing slash, for
    /// example `https://api.individual.githubcopilot.com`.
    pub fn api_base_url(&self) -> &str {
        &self.api_base_url
    }

    /// When the token stops working, if known.
    pub fn expires_at(&self) -> Option<SystemTime> {
        self.expires_at
    }

    /// Whether the token is still good at `now` with at least `margin` to
    /// spare. A token with no known expiry is always valid.
    pub fn is_valid_at(&self, now: SystemTime, margin: Duration) -> bool {
        self.expires_at
            .is_none_or(|expires_at| now + margin < expires_at)
    }
}

impl fmt::Debug for CopilotSessionToken {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CopilotSessionToken")
            .field("token", &"<redacted>")
            .field("api_base_url", &self.api_base_url)
            .field("expires_at", &self.expires_at)
            .finish()
    }
}

/// Where a [`CopilotTokenProvider`] gets its session tokens.
#[derive(Debug)]
enum TokenSource {
    /// Trade a GitHub OAuth token at the Copilot token endpoint.
    Exchange {
        github_token: GitHubOAuthToken,
        github_api_base_url: String,
    },
    /// Use one fixed token for every request.
    Static(CopilotSessionToken),
}

/// Hands out valid Copilot session tokens, trading for a new one when the
/// cached one is close to expiry.
///
/// Build it with [`CopilotTokenProvider::from_github_token`] (the usual case)
/// or [`CopilotTokenProvider::with_static_token`], wrap it in an
/// [`Arc`](std::sync::Arc), and share it between clients. Concurrent callers
/// wait for one trade instead of each starting their own.
///
/// It also carries the [`CopilotIdentity`] headers, since GitHub checks them on
/// the token endpoint as well as on the chat API.
#[derive(Debug)]
pub struct CopilotTokenProvider {
    source: TokenSource,
    identity: CopilotIdentity,
    refresh_margin: Duration,
    cached: Mutex<Option<CopilotSessionToken>>,
}

impl CopilotTokenProvider {
    /// A provider that trades `github_token` (from a
    /// [`GitHubDeviceFlow`](super::GitHubDeviceFlow) login, or one you stored)
    /// for session tokens at <https://api.github.com>.
    pub fn from_github_token(github_token: GitHubOAuthToken) -> Self {
        Self::with_source(TokenSource::Exchange {
            github_token,
            github_api_base_url: GITHUB_API_DEFAULT_BASE_URL.to_owned(),
        })
    }

    /// A provider that sends `token` to `api_base_url` on every request and
    /// never trades for a new one.
    ///
    /// Use it when you already hold a token the Copilot API accepts as is,
    /// for example one that another tool obtained, or when testing against a
    /// local server.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use cloudllm::clients::github_copilot::CopilotTokenProvider;
    ///
    /// # #[tokio::main]
    /// # async fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let provider = CopilotTokenProvider::with_static_token("tok", "http://127.0.0.1:8080/");
    /// let session = provider.session_token().await?;
    /// assert_eq!(session.secret(), "tok");
    /// assert_eq!(session.api_base_url(), "http://127.0.0.1:8080");
    /// # Ok(())
    /// # }
    /// ```
    pub fn with_static_token(token: impl Into<String>, api_base_url: impl AsRef<str>) -> Self {
        Self::with_source(TokenSource::Static(CopilotSessionToken::new(
            token,
            api_base_url,
            None,
        )))
    }

    fn with_source(source: TokenSource) -> Self {
        Self {
            source,
            identity: CopilotIdentity::default(),
            refresh_margin: DEFAULT_REFRESH_MARGIN,
            cached: Mutex::new(None),
        }
    }

    /// Trade at a different GitHub API host, such as GitHub Enterprise Server
    /// (`https://github.example.com/api/v3`) or a test server. Has no effect
    /// on a provider built with [`CopilotTokenProvider::with_static_token`].
    pub fn with_github_api_base_url(self, base_url: impl AsRef<str>) -> Self {
        let source = match self.source {
            TokenSource::Exchange { github_token, .. } => TokenSource::Exchange {
                github_token,
                github_api_base_url: base_url.as_ref().trim_end_matches('/').to_owned(),
            },
            fixed @ TokenSource::Static(_) => fixed,
        };
        Self { source, ..self }
    }

    /// Send `identity`'s headers instead of the defaults.
    pub fn with_identity(self, identity: CopilotIdentity) -> Self {
        Self { identity, ..self }
    }

    /// Trade for a new session token once fewer than `margin` remain on the
    /// cached one (default [`DEFAULT_REFRESH_MARGIN`]).
    pub fn with_refresh_margin(self, margin: Duration) -> Self {
        Self {
            refresh_margin: margin,
            ..self
        }
    }

    /// The headers sent with every Copilot request.
    pub fn identity(&self) -> &CopilotIdentity {
        &self.identity
    }

    /// A session token good for at least the refresh margin.
    ///
    /// Returns the cached token when it is still fresh; otherwise trades for
    /// a new one and caches it.
    ///
    /// # Errors
    ///
    /// [`CopilotAuthError::CopilotNotAvailable`] when GitHub refuses the trade
    /// (HTTP 401, 403 or 404: no Copilot seat, or the OAuth app may not use
    /// Copilot), [`CopilotAuthError::Transport`],
    /// [`CopilotAuthError::HttpStatus`] or
    /// [`CopilotAuthError::InvalidResponse`] otherwise.
    pub async fn session_token(&self) -> Result<CopilotSessionToken, CopilotAuthError> {
        let (github_token, github_api_base_url) = match &self.source {
            TokenSource::Static(token) => return Ok(token.clone()),
            TokenSource::Exchange {
                github_token,
                github_api_base_url,
            } => (github_token, github_api_base_url),
        };
        // Holding the lock across the trade makes concurrent callers wait for
        // one request instead of each sending their own.
        let mut cached = self.cached.lock().await;
        if let Some(token) = cached
            .as_ref()
            .filter(|token| token.is_valid_at(SystemTime::now(), self.refresh_margin))
        {
            return Ok(token.clone());
        }
        let fresh = self.exchange(github_token, github_api_base_url).await?;
        *cached = Some(fresh.clone());
        Ok(fresh)
    }

    /// Forget the cached session token so the next
    /// [`CopilotTokenProvider::session_token`] trades for a new one.
    ///
    /// `CopilotClient` calls this when the Copilot API
    /// answers 401, which happens when GitHub revokes a session token early.
    pub async fn invalidate(&self) {
        *self.cached.lock().await = None;
    }

    /// Trade `github_token` for a session token.
    async fn exchange(
        &self,
        github_token: &GitHubOAuthToken,
        github_api_base_url: &str,
    ) -> Result<CopilotSessionToken, CopilotAuthError> {
        let url = format!("{github_api_base_url}/copilot_internal/v2/token");
        let request = self.identity.headers().into_iter().fold(
            get_shared_http_client()
                .get(&url)
                .header("Authorization", format!("token {}", github_token.secret()))
                .header("Accept", "application/json"),
            |request, (name, value)| request.header(name, value),
        );
        let response = request.send().await?;
        let status = response.status();
        let text = response.text().await?;
        match status.as_u16() {
            200..=299 => parse_session_token(&text),
            401 | 403 | 404 => Err(CopilotAuthError::copilot_not_available(
                status.as_u16(),
                &text,
            )),
            other => Err(CopilotAuthError::http_status(other, &text)),
        }
    }
}

/// The JSON the Copilot token endpoint answers with (fields CloudLLM uses).
#[derive(Deserialize)]
struct SessionTokenResponse {
    token: String,
    #[serde(default)]
    expires_at: Option<u64>,
    #[serde(default)]
    endpoints: Option<SessionEndpoints>,
}

#[derive(Deserialize)]
struct SessionEndpoints {
    #[serde(default)]
    api: Option<String>,
}

/// Parse the token endpoint's answer, falling back to
/// [`COPILOT_DEFAULT_API_BASE_URL`] when it names no API host.
pub(crate) fn parse_session_token(text: &str) -> Result<CopilotSessionToken, CopilotAuthError> {
    let response: SessionTokenResponse = serde_json::from_str(text).map_err(|error| {
        CopilotAuthError::InvalidResponse(format!("Copilot token response: {error}"))
    })?;
    let api_base_url = response
        .endpoints
        .and_then(|endpoints| endpoints.api)
        .unwrap_or_else(|| COPILOT_DEFAULT_API_BASE_URL.to_owned());
    let expires_at = response
        .expires_at
        .map(|seconds| UNIX_EPOCH + Duration::from_secs(seconds));
    Ok(CopilotSessionToken::new(
        response.token,
        api_base_url,
        expires_at,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Verifies that the API host and expiry come from the token answer.
    #[test]
    fn parses_host_and_expiry() {
        let token = parse_session_token(
            r#"{"token":"tid=1","expires_at":1800000000,"refresh_in":1500,
                "endpoints":{"api":"https://api.individual.githubcopilot.com/"}}"#,
        )
        .unwrap();
        assert_eq!(token.secret(), "tid=1");
        assert_eq!(
            token.api_base_url(),
            "https://api.individual.githubcopilot.com"
        );
        assert_eq!(
            token.expires_at(),
            Some(UNIX_EPOCH + Duration::from_secs(1_800_000_000))
        );
    }

    /// Verifies the default API host when the answer names none.
    #[test]
    fn falls_back_to_the_default_host() {
        let token = parse_session_token(r#"{"token":"tid=1"}"#).unwrap();
        assert_eq!(token.api_base_url(), COPILOT_DEFAULT_API_BASE_URL);
        assert_eq!(token.expires_at(), None);
    }

    /// Verifies that an answer without a token is rejected.
    #[test]
    fn rejects_an_answer_without_a_token() {
        assert!(matches!(
            parse_session_token(r#"{"expires_at":1}"#),
            Err(CopilotAuthError::InvalidResponse(_))
        ));
    }

    /// Verifies that the integration ID header is only sent when set.
    #[test]
    fn integration_id_is_optional() {
        let identity = CopilotIdentity::default();
        assert!(!identity
            .headers()
            .iter()
            .any(|(name, _)| *name == "Copilot-Integration-Id"));
    }

    /// Verifies that `Debug` never prints the session token.
    #[test]
    fn session_token_debug_hides_the_secret() {
        let token = CopilotSessionToken::new("tid=secret", "https://x", None);
        assert!(!format!("{token:?}").contains("tid=secret"));
    }
}
