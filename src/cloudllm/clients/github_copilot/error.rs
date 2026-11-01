//! The error type shared by the GitHub Copilot login and token code.

use std::fmt;

/// The longest response body, in bytes, that an error message quotes.
///
/// GitHub error pages can be large HTML documents. Quoting only the start keeps
/// log lines readable and stops an error from carrying a whole page around.
const MAX_QUOTED_BODY: usize = 512;

/// Everything that can go wrong while signing in to GitHub or obtaining a
/// Copilot session token.
///
/// The variants separate the cases a caller handles differently:
///
/// - [`CopilotAuthError::ExpiredToken`] and [`CopilotAuthError::AccessDenied`]
///   end a device-flow login. Start a new login (or tell the user they
///   declined) instead of retrying.
/// - [`CopilotAuthError::CopilotNotAvailable`] means the GitHub account signed
///   in, but GitHub refused to issue a Copilot token: the account has no
///   Copilot seat, or the OAuth app's `client_id` is not allowed to use
///   Copilot.
/// - [`CopilotAuthError::Transport`] and [`CopilotAuthError::HttpStatus`] are
///   network and server failures, often worth retrying later.
///
/// # Examples
///
/// ```rust
/// use cloudllm::clients::github_copilot::CopilotAuthError;
///
/// fn advice(error: &CopilotAuthError) -> &'static str {
///     match error {
///         CopilotAuthError::ExpiredToken => "the code expired, start over",
///         CopilotAuthError::AccessDenied => "you declined the request",
///         CopilotAuthError::CopilotNotAvailable { .. } => "no Copilot access",
///         _ => "try again later",
///     }
/// }
///
/// assert_eq!(advice(&CopilotAuthError::AccessDenied), "you declined the request");
/// ```
#[derive(Debug)]
#[non_exhaustive]
pub enum CopilotAuthError {
    /// The HTTP request could not be sent or its body could not be read.
    Transport(reqwest::Error),
    /// The server answered with a status CloudLLM does not treat specially.
    HttpStatus {
        /// The HTTP status code.
        status: u16,
        /// The start of the response body (at most 512 bytes).
        body: String,
    },
    /// The server answered successfully, but not with the JSON CloudLLM
    /// expected (a missing field, or not JSON at all).
    InvalidResponse(String),
    /// The device code expired before the user approved it. Start a new login.
    ExpiredToken,
    /// The user pressed "Cancel" on the GitHub authorization page.
    AccessDenied,
    /// GitHub returned an OAuth error other than the ones above, such as
    /// `incorrect_client_credentials` (unknown `client_id`) or
    /// `device_flow_disabled` (the OAuth app has device flow turned off).
    OAuth {
        /// The OAuth `error` code, for example `"device_flow_disabled"`.
        error: String,
        /// GitHub's `error_description`, when present.
        description: Option<String>,
    },
    /// GitHub refused to issue a Copilot session token (HTTP 401, 403 or 404
    /// from the token endpoint).
    ///
    /// The GitHub token is valid, but the account has no Copilot access, or
    /// the OAuth app that minted the token may not use Copilot.
    CopilotNotAvailable {
        /// The HTTP status code.
        status: u16,
        /// The start of the response body (at most 512 bytes).
        body: String,
    },
}

impl CopilotAuthError {
    /// Build an [`CopilotAuthError::HttpStatus`], quoting at most 512 bytes of
    /// `body`.
    pub(crate) fn http_status(status: u16, body: &str) -> Self {
        Self::HttpStatus {
            status,
            body: quote_body(body),
        }
    }
}

/// Cut `body` to [`MAX_QUOTED_BODY`] bytes on a character boundary.
fn quote_body(body: &str) -> String {
    if body.len() <= MAX_QUOTED_BODY {
        return body.to_owned();
    }
    let cut = (0..=MAX_QUOTED_BODY)
        .rev()
        .find(|&index| body.is_char_boundary(index))
        .unwrap_or(0);
    format!("{}…", &body[..cut])
}

impl fmt::Display for CopilotAuthError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Transport(source) => write!(f, "GitHub request failed: {source}"),
            Self::HttpStatus { status, body } => {
                write!(f, "GitHub answered HTTP {status}: {body}")
            }
            Self::InvalidResponse(detail) => {
                write!(f, "unexpected response from GitHub: {detail}")
            }
            Self::ExpiredToken => write!(
                f,
                "the device code expired before it was approved; start a new login"
            ),
            Self::AccessDenied => write!(f, "the user denied the GitHub authorization request"),
            Self::OAuth { error, description } => match description {
                Some(description) => write!(f, "GitHub OAuth error {error}: {description}"),
                None => write!(f, "GitHub OAuth error {error}"),
            },
            Self::CopilotNotAvailable { status, body } => write!(
                f,
                "GitHub refused a Copilot token (HTTP {status}); the account may have no \
                 Copilot seat, or this OAuth client_id may not use Copilot: {body}"
            ),
        }
    }
}

impl std::error::Error for CopilotAuthError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Transport(source) => Some(source),
            _ => None,
        }
    }
}

impl From<reqwest::Error> for CopilotAuthError {
    fn from(source: reqwest::Error) -> Self {
        Self::Transport(source)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Verifies that long bodies are cut to the quote limit without splitting
    /// a multi-byte character.
    #[test]
    fn quote_body_truncates_on_a_char_boundary() {
        let body = "é".repeat(600);
        let quoted = quote_body(&body);
        assert!(quoted.ends_with('…'));
        assert!(quoted.len() <= MAX_QUOTED_BODY + '…'.len_utf8());
    }

    /// Verifies that short bodies are quoted unchanged.
    #[test]
    fn quote_body_keeps_short_bodies() {
        assert_eq!(quote_body("not found"), "not found");
    }
}
