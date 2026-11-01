//! GitHub's OAuth device flow: sign a user in from a terminal, with no browser
//! redirect and no client secret.
//!
//! The flow has three steps:
//!
//! 1. [`GitHubDeviceFlow::request_code`] asks GitHub for a short `user_code`
//!    and a `verification_uri` (normally <https://github.com/login/device>).
//! 2. You show both to the user. They open the page, type the code, and
//!    approve your OAuth app.
//! 3. [`GitHubDeviceFlow::wait_for_token`] polls GitHub at the interval it asked
//!    for until the user approves, declines, or the code expires, and returns a
//!    [`GitHubOAuthToken`].
//!
//! [`GitHubDeviceFlow::authorize`] runs all three steps and calls your closure
//! with the code to display. GitHub documents the protocol in
//! [Authorizing OAuth apps: device flow](https://docs.github.com/en/apps/oauth-apps/building-oauth-apps/authorizing-oauth-apps#device-flow).
//!
//! # The `client_id`
//!
//! Every login names the OAuth app it is for. Register your own at
//! <https://github.com/settings/developers> ("New OAuth App", then tick
//! "Enable Device Flow") and pass its client ID to [`GitHubDeviceFlow::new`].
//! CloudLLM ships no default ID. Which Copilot models a token can reach can
//! depend on the app that minted it, so the choice stays with you.
//!
//! # Example
//!
//! ```rust,no_run
//! use cloudllm::clients::github_copilot::GitHubDeviceFlow;
//!
//! # #[tokio::main]
//! # async fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let flow = GitHubDeviceFlow::new("Iv1.your-oauth-app-id");
//! let token = flow
//!     .authorize(|code| {
//!         println!(
//!             "Open {} and enter the code {}",
//!             code.verification_uri, code.user_code
//!         );
//!     })
//!     .await?;
//!
//! // Store the token (in the OS keychain, say) so the user signs in only once.
//! let secret: &str = token.secret();
//! # let _ = secret;
//! # Ok(())
//! # }
//! ```

use super::error::CopilotAuthError;
use crate::clients::common::get_shared_http_client;
use serde::Deserialize;
use std::fmt;
use std::time::{Duration, Instant};

/// Where GitHub serves its OAuth endpoints.
pub const GITHUB_DEFAULT_BASE_URL: &str = "https://github.com";

/// The scope [`GitHubDeviceFlow::new`] asks for.
///
/// Copilot needs no repository access; `read:user` is the smallest scope the
/// Copilot token endpoint accepts.
pub const DEFAULT_SCOPE: &str = "read:user";

/// The grant type GitHub expects when polling for a device-flow token.
const DEVICE_GRANT_TYPE: &str = "urn:ietf:params:oauth:grant-type:device_code";

/// How much longer to wait between polls after GitHub answers `slow_down`,
/// when it does not say how long. RFC 8628 §3.5 fixes this at five seconds.
const SLOW_DOWN_STEP: Duration = Duration::from_secs(5);

/// The code a user types on GitHub, plus what CloudLLM needs to poll for the
/// token. Returned by [`GitHubDeviceFlow::request_code`].
///
/// Show [`DeviceCode::user_code`] and [`DeviceCode::verification_uri`] to the
/// user. Keep [`DeviceCode::device_code`] private: whoever holds it can collect
/// the token once the user approves. `Debug` output hides it.
#[derive(Clone, Deserialize, PartialEq, Eq)]
pub struct DeviceCode {
    /// The secret CloudLLM sends while polling. Do not show it to the user.
    pub device_code: String,
    /// The short code the user types, for example `"WDJB-MJHT"`.
    pub user_code: String,
    /// The page where the user types the code, normally
    /// `https://github.com/login/device`.
    pub verification_uri: String,
    /// Seconds from issue until the code expires (GitHub uses 900).
    pub expires_in: u64,
    /// The minimum number of seconds GitHub wants between polls.
    pub interval: u64,
}

impl DeviceCode {
    /// The minimum wait between polls, as a [`Duration`].
    ///
    /// # Examples
    ///
    /// ```rust
    /// use cloudllm::clients::github_copilot::DeviceCode;
    /// use std::time::Duration;
    ///
    /// let code = DeviceCode {
    ///     device_code: "secret".into(),
    ///     user_code: "WDJB-MJHT".into(),
    ///     verification_uri: "https://github.com/login/device".into(),
    ///     expires_in: 900,
    ///     interval: 5,
    /// };
    /// assert_eq!(code.poll_interval(), Duration::from_secs(5));
    /// assert_eq!(code.lifetime(), Duration::from_secs(900));
    /// ```
    pub fn poll_interval(&self) -> Duration {
        Duration::from_secs(self.interval)
    }

    /// How long the code stays valid after GitHub issued it.
    pub fn lifetime(&self) -> Duration {
        Duration::from_secs(self.expires_in)
    }
}

impl fmt::Debug for DeviceCode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DeviceCode")
            .field("device_code", &"<redacted>")
            .field("user_code", &self.user_code)
            .field("verification_uri", &self.verification_uri)
            .field("expires_in", &self.expires_in)
            .field("interval", &self.interval)
            .finish()
    }
}

/// A GitHub OAuth access token, the result of a device-flow login.
///
/// The token is long-lived: it stays valid until the user revokes the app at
/// <https://github.com/settings/applications>. Persist it with
/// [`GitHubOAuthToken::secret`] and rebuild it next time with
/// [`GitHubOAuthToken::new`], so the user signs in once.
///
/// `Debug` output hides the secret, so logging a token by accident leaks
/// nothing.
///
/// # Examples
///
/// ```rust
/// use cloudllm::clients::github_copilot::GitHubOAuthToken;
///
/// let token = GitHubOAuthToken::new("gho_example");
/// assert_eq!(token.secret(), "gho_example");
/// assert!(!format!("{token:?}").contains("gho_example"));
/// ```
#[derive(Clone, Deserialize, PartialEq, Eq)]
pub struct GitHubOAuthToken {
    access_token: String,
    #[serde(default)]
    token_type: Option<String>,
    #[serde(default)]
    scope: Option<String>,
}

impl GitHubOAuthToken {
    /// Wrap a token you stored earlier (or a token from another source, such
    /// as `gh auth token`).
    pub fn new(access_token: impl Into<String>) -> Self {
        Self {
            access_token: access_token.into(),
            token_type: None,
            scope: None,
        }
    }

    /// The token itself. Treat it like a password.
    pub fn secret(&self) -> &str {
        &self.access_token
    }

    /// The token type GitHub reported (`"bearer"`), if this token came from a
    /// login rather than [`GitHubOAuthToken::new`].
    pub fn token_type(&self) -> Option<&str> {
        self.token_type.as_deref()
    }

    /// The scopes GitHub granted, comma-separated, if this token came from a
    /// login rather than [`GitHubOAuthToken::new`].
    pub fn scope(&self) -> Option<&str> {
        self.scope.as_deref()
    }
}

impl fmt::Debug for GitHubOAuthToken {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("GitHubOAuthToken")
            .field("access_token", &"<redacted>")
            .field("token_type", &self.token_type)
            .field("scope", &self.scope)
            .finish()
    }
}

/// What one poll of the token endpoint found. Returned by
/// [`GitHubDeviceFlow::poll_once`].
///
/// [`GitHubDeviceFlow::wait_for_token`] handles these for you; use
/// `poll_once` only when you drive the polling loop yourself (from a UI event
/// loop, say).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PollOutcome {
    /// The user has not approved yet. Wait the poll interval and ask again.
    Pending,
    /// You polled too fast. Wait `interval` from now on.
    SlowDown {
        /// The new minimum wait between polls.
        interval: Duration,
    },
    /// The user approved. Here is the token.
    Authorized(GitHubOAuthToken),
}

/// A GitHub OAuth device-flow login for one OAuth app.
///
/// See the [module documentation](self) for the protocol and a full example.
#[derive(Debug, Clone)]
pub struct GitHubDeviceFlow {
    client_id: String,
    scope: String,
    github_base_url: String,
}

impl GitHubDeviceFlow {
    /// A login for the OAuth app `client_id`, asking for [`DEFAULT_SCOPE`]
    /// against <https://github.com>.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use cloudllm::clients::github_copilot::GitHubDeviceFlow;
    ///
    /// let flow = GitHubDeviceFlow::new("Iv1.your-oauth-app-id");
    /// assert_eq!(flow.client_id(), "Iv1.your-oauth-app-id");
    /// assert_eq!(flow.scope(), "read:user");
    /// ```
    pub fn new(client_id: impl Into<String>) -> Self {
        Self {
            client_id: client_id.into(),
            scope: DEFAULT_SCOPE.to_owned(),
            github_base_url: GITHUB_DEFAULT_BASE_URL.to_owned(),
        }
    }

    /// Ask for different scopes, space-separated (for example
    /// `"read:user user:email"`).
    pub fn with_scope(self, scope: impl Into<String>) -> Self {
        Self {
            scope: scope.into(),
            ..self
        }
    }

    /// Talk to a different GitHub host, such as a GitHub Enterprise Server
    /// (`https://github.example.com`) or a test server. A trailing slash is
    /// ignored.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use cloudllm::clients::github_copilot::GitHubDeviceFlow;
    ///
    /// let flow = GitHubDeviceFlow::new("id").with_github_base_url("https://ghe.example.com/");
    /// assert_eq!(flow.github_base_url(), "https://ghe.example.com");
    /// ```
    pub fn with_github_base_url(self, base_url: impl AsRef<str>) -> Self {
        Self {
            github_base_url: base_url.as_ref().trim_end_matches('/').to_owned(),
            ..self
        }
    }

    /// The OAuth app's client ID.
    pub fn client_id(&self) -> &str {
        &self.client_id
    }

    /// The scopes this login asks for.
    pub fn scope(&self) -> &str {
        &self.scope
    }

    /// The GitHub host this login talks to, without a trailing slash.
    pub fn github_base_url(&self) -> &str {
        &self.github_base_url
    }

    /// Step 1: ask GitHub for a device code and the code the user types.
    ///
    /// # Errors
    ///
    /// [`CopilotAuthError::Transport`] when GitHub cannot be reached,
    /// [`CopilotAuthError::OAuth`] when GitHub rejects the request (an unknown
    /// `client_id`, or an app with device flow turned off), and
    /// [`CopilotAuthError::HttpStatus`] or
    /// [`CopilotAuthError::InvalidResponse`] for other bad answers.
    pub async fn request_code(&self) -> Result<DeviceCode, CopilotAuthError> {
        let url = format!("{}/login/device/code", self.github_base_url);
        let form = [
            ("client_id", self.client_id.as_str()),
            ("scope", self.scope.as_str()),
        ];
        let body = post_form(&url, &form).await?;
        if let Some(error) = oauth_error(&body) {
            return Err(error);
        }
        serde_json::from_value(body).map_err(|error| {
            CopilotAuthError::InvalidResponse(format!("device code response: {error}"))
        })
    }

    /// Step 3, once: ask GitHub whether the user has approved `code` yet.
    ///
    /// # Errors
    ///
    /// [`CopilotAuthError::ExpiredToken`] when the code expired,
    /// [`CopilotAuthError::AccessDenied`] when the user declined, and the
    /// errors listed on [`GitHubDeviceFlow::request_code`].
    pub async fn poll_once(&self, code: &DeviceCode) -> Result<PollOutcome, CopilotAuthError> {
        let url = format!("{}/login/oauth/access_token", self.github_base_url);
        let form = [
            ("client_id", self.client_id.as_str()),
            ("device_code", code.device_code.as_str()),
            ("grant_type", DEVICE_GRANT_TYPE),
        ];
        parse_poll_response(post_form(&url, &form).await?, code.poll_interval())
    }

    /// Step 3: poll until the user approves `code`, then return the token.
    ///
    /// Waits [`DeviceCode::poll_interval`] before each poll and backs off
    /// further whenever GitHub answers `slow_down`, as the protocol requires.
    /// Gives up with [`CopilotAuthError::ExpiredToken`] once
    /// [`DeviceCode::lifetime`] has passed. The future can be dropped at any
    /// time to cancel the login.
    ///
    /// # Errors
    ///
    /// As for [`GitHubDeviceFlow::poll_once`].
    pub async fn wait_for_token(
        &self,
        code: &DeviceCode,
    ) -> Result<GitHubOAuthToken, CopilotAuthError> {
        let deadline = Instant::now() + code.lifetime();
        let mut interval = code.poll_interval();
        loop {
            if Instant::now() + interval > deadline {
                return Err(CopilotAuthError::ExpiredToken);
            }
            tokio::time::sleep(interval).await;
            match self.poll_once(code).await? {
                PollOutcome::Pending => {}
                PollOutcome::SlowDown { interval: slower } => interval = slower,
                PollOutcome::Authorized(token) => return Ok(token),
            }
        }
    }

    /// Run the whole login: request a code, hand it to `show_code` so you can
    /// display it, then wait for the user to approve.
    ///
    /// `show_code` runs once, before polling starts. Print the code, open a
    /// browser, copy it to the clipboard: whatever suits your program.
    ///
    /// # Errors
    ///
    /// As for [`GitHubDeviceFlow::request_code`] and
    /// [`GitHubDeviceFlow::wait_for_token`].
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use cloudllm::clients::github_copilot::GitHubDeviceFlow;
    ///
    /// # #[tokio::main]
    /// # async fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let token = GitHubDeviceFlow::new("Iv1.your-oauth-app-id")
    ///     .authorize(|code| {
    ///         eprintln!("Go to {} and type {}", code.verification_uri, code.user_code)
    ///     })
    ///     .await?;
    /// # let _ = token;
    /// # Ok(())
    /// # }
    /// ```
    pub async fn authorize<F>(&self, show_code: F) -> Result<GitHubOAuthToken, CopilotAuthError>
    where
        F: FnOnce(&DeviceCode),
    {
        let code = self.request_code().await?;
        show_code(&code);
        self.wait_for_token(&code).await
    }
}

/// POST a form to a GitHub OAuth endpoint and return the JSON answer.
///
/// GitHub answers OAuth errors such as `authorization_pending` with HTTP 200
/// and an `error` field, so a 200 body is returned for the caller to inspect.
/// A 4xx body that is JSON with an `error` field is returned the same way.
async fn post_form(
    url: &str,
    form: &[(&str, &str)],
) -> Result<serde_json::Value, CopilotAuthError> {
    let response = get_shared_http_client()
        .post(url)
        .header("Accept", "application/json")
        .form(form)
        .send()
        .await?;
    let status = response.status();
    let text = response.text().await?;
    match serde_json::from_str::<serde_json::Value>(&text) {
        Ok(body) if status.is_success() || body.get("error").is_some() => Ok(body),
        Err(error) if status.is_success() => Err(CopilotAuthError::InvalidResponse(format!(
            "not JSON ({error}): {text}"
        ))),
        _ => Err(CopilotAuthError::http_status(status.as_u16(), &text)),
    }
}

/// Turn an OAuth `error` field into the matching [`CopilotAuthError`].
///
/// Returns `None` when `body` has no `error` field. The two polling states
/// (`authorization_pending`, `slow_down`) are not errors and are handled by
/// [`parse_poll_response`] before this runs.
fn oauth_error(body: &serde_json::Value) -> Option<CopilotAuthError> {
    let error = body.get("error")?.as_str().unwrap_or("unknown_error");
    Some(match error {
        "expired_token" => CopilotAuthError::ExpiredToken,
        "access_denied" => CopilotAuthError::AccessDenied,
        other => CopilotAuthError::OAuth {
            error: other.to_owned(),
            description: body
                .get("error_description")
                .and_then(serde_json::Value::as_str)
                .map(str::to_owned),
        },
    })
}

/// Interpret one answer from the token endpoint.
///
/// `current_interval` is the wait in force; a `slow_down` without an
/// `interval` field adds [`SLOW_DOWN_STEP`] to it.
pub(crate) fn parse_poll_response(
    body: serde_json::Value,
    current_interval: Duration,
) -> Result<PollOutcome, CopilotAuthError> {
    match body.get("error").and_then(serde_json::Value::as_str) {
        Some("authorization_pending") => Ok(PollOutcome::Pending),
        Some("slow_down") => {
            let interval = body
                .get("interval")
                .and_then(serde_json::Value::as_u64)
                .map(Duration::from_secs)
                .unwrap_or(current_interval + SLOW_DOWN_STEP);
            Ok(PollOutcome::SlowDown { interval })
        }
        Some(_) => Err(oauth_error(&body).expect("body has an error field")),
        None => serde_json::from_value(body)
            .map(PollOutcome::Authorized)
            .map_err(|error| {
                CopilotAuthError::InvalidResponse(format!("access token response: {error}"))
            }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    const FIVE: Duration = Duration::from_secs(5);

    /// Verifies that `authorization_pending` keeps the login polling.
    #[test]
    fn pending_answer_keeps_polling() {
        let outcome = parse_poll_response(json!({"error": "authorization_pending"}), FIVE);
        assert_eq!(outcome.unwrap(), PollOutcome::Pending);
    }

    /// Verifies that `slow_down` adopts GitHub's interval when it sends one.
    #[test]
    fn slow_down_uses_the_interval_github_sends() {
        let outcome = parse_poll_response(json!({"error": "slow_down", "interval": 12}), FIVE);
        assert_eq!(
            outcome.unwrap(),
            PollOutcome::SlowDown {
                interval: Duration::from_secs(12)
            }
        );
    }

    /// Verifies that `slow_down` without an interval adds five seconds.
    #[test]
    fn slow_down_without_interval_adds_five_seconds() {
        let outcome = parse_poll_response(json!({"error": "slow_down"}), FIVE);
        assert_eq!(
            outcome.unwrap(),
            PollOutcome::SlowDown {
                interval: Duration::from_secs(10)
            }
        );
    }

    /// Verifies that a token answer yields the token and its metadata.
    #[test]
    fn token_answer_is_authorized() {
        let body = json!({"access_token": "gho_x", "token_type": "bearer", "scope": "read:user"});
        let PollOutcome::Authorized(token) = parse_poll_response(body, FIVE).unwrap() else {
            panic!("expected a token");
        };
        assert_eq!(token.secret(), "gho_x");
        assert_eq!(token.token_type(), Some("bearer"));
        assert_eq!(token.scope(), Some("read:user"));
    }

    /// Verifies that the terminal OAuth errors map to their own variants.
    #[test]
    fn terminal_errors_map_to_variants() {
        assert!(matches!(
            parse_poll_response(json!({"error": "expired_token"}), FIVE),
            Err(CopilotAuthError::ExpiredToken)
        ));
        assert!(matches!(
            parse_poll_response(json!({"error": "access_denied"}), FIVE),
            Err(CopilotAuthError::AccessDenied)
        ));
        let Err(CopilotAuthError::OAuth { error, description }) = parse_poll_response(
            json!({"error": "device_flow_disabled", "error_description": "off"}),
            FIVE,
        ) else {
            panic!("expected an OAuth error");
        };
        assert_eq!(error, "device_flow_disabled");
        assert_eq!(description.as_deref(), Some("off"));
    }

    /// Verifies that a 200 answer with neither a token nor an error is
    /// rejected instead of being mistaken for success.
    #[test]
    fn answer_without_token_or_error_is_invalid() {
        assert!(matches!(
            parse_poll_response(json!({"foo": 1}), FIVE),
            Err(CopilotAuthError::InvalidResponse(_))
        ));
    }

    /// Verifies that `Debug` never prints the device code.
    #[test]
    fn device_code_debug_hides_the_secret() {
        let code = DeviceCode {
            device_code: "very-secret".into(),
            user_code: "ABCD-EFGH".into(),
            verification_uri: "https://github.com/login/device".into(),
            expires_in: 900,
            interval: 5,
        };
        let printed = format!("{code:?}");
        assert!(!printed.contains("very-secret"));
        assert!(printed.contains("ABCD-EFGH"));
    }
}
