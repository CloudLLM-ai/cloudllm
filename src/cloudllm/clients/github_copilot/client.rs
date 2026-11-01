//! [`CopilotClient`], the [`ClientWrapper`] that sends chat requests to GitHub
//! Copilot, and [`list_models`], which asks Copilot what a plan includes.

use super::error::CopilotAuthError;
use super::session_token::{CopilotSessionToken, CopilotTokenProvider};
use crate::client_wrapper::{MessageStreamFuture, TokenUsage, ToolDefinition};
use crate::clients::common::{get_shared_http_client, send_with_native_tools_and_headers};
use crate::clients::sse_stream::open_chat_completions_stream;
use crate::{ClientWrapper, Message};
use async_trait::async_trait;
use serde::Deserialize;
use std::error::Error;
use std::sync::Arc;
use tokio::sync::Mutex;

/// A [`ClientWrapper`] for GitHub Copilot's OpenAI-compatible Chat Completions
/// API.
///
/// Each request asks the shared [`CopilotTokenProvider`] for a session token,
/// so expiry and refresh never reach the caller. If Copilot answers 401 (GitHub
/// revoked the session token early), the client drops the cached token, trades
/// for a new one, and retries the request once.
///
/// Supported: plain chat, native tool calling, live streaming
/// ([`ClientWrapper::send_message_stream`]), and token usage. Model names are
/// Copilot's own IDs, such as `"gpt-4.1"` or `"claude-sonnet-4"`. They change
/// with GitHub's catalogue and differ between plans, so CloudLLM has no enum
/// for them. Call [`CopilotClient::list_models`] to see what your account
/// offers.
///
/// # Example
///
/// ```rust,no_run
/// use std::sync::Arc;
/// use cloudllm::client_wrapper::{ClientWrapper, Message, Role};
/// use cloudllm::clients::github_copilot::{
///     CopilotClient, CopilotTokenProvider, GitHubOAuthToken,
/// };
///
/// # #[tokio::main]
/// # async fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let github_token = GitHubOAuthToken::new(std::env::var("GITHUB_COPILOT_OAUTH_TOKEN")?);
/// let provider = Arc::new(CopilotTokenProvider::from_github_token(github_token));
///
/// let client = CopilotClient::new(provider, "gpt-4.1");
/// let reply = client
///     .send_message(
///         &[Message {
///             role: Role::User,
///             content: Arc::from("Explain borrowing in one sentence."),
///             tool_calls: vec![],
///         }],
///         None,
///     )
///     .await?;
/// println!("{}", reply.content);
/// # Ok(())
/// # }
/// ```
pub struct CopilotClient {
    provider: Arc<CopilotTokenProvider>,
    model: String,
    token_usage: Mutex<Option<TokenUsage>>,
}

impl CopilotClient {
    /// A client for `model` that authenticates through `provider`.
    ///
    /// Share one provider between clients for different models, so they reuse
    /// one session token.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use std::sync::Arc;
    /// use cloudllm::client_wrapper::ClientWrapper;
    /// use cloudllm::clients::github_copilot::{CopilotClient, CopilotTokenProvider};
    ///
    /// let provider = Arc::new(CopilotTokenProvider::with_static_token("tok", "http://127.0.0.1:1"));
    /// let fast = CopilotClient::new(Arc::clone(&provider), "gpt-4.1");
    /// let smart = CopilotClient::new(provider, "claude-sonnet-4");
    /// assert_eq!(fast.model_name(), "gpt-4.1");
    /// assert_eq!(smart.provider_name(), "GitHub Copilot");
    /// ```
    pub fn new(provider: Arc<CopilotTokenProvider>, model: impl Into<String>) -> Self {
        Self {
            provider,
            model: model.into(),
            token_usage: Mutex::new(None),
        }
    }

    /// The token provider this client uses.
    pub fn token_provider(&self) -> &Arc<CopilotTokenProvider> {
        &self.provider
    }

    /// The models this account can use. See [`list_models`].
    ///
    /// # Errors
    ///
    /// As for [`list_models`].
    pub async fn list_models(&self) -> Result<Vec<CopilotModel>, CopilotAuthError> {
        list_models(&self.provider).await
    }

    /// One blocking Chat Completions request with the given session token.
    ///
    /// Returns the error as text so no non-`Send` error is held across the
    /// retry's `await`.
    async fn send_once(
        &self,
        session: &CopilotSessionToken,
        messages: &[Message],
        tools: &[ToolDefinition],
    ) -> Result<Message, String> {
        send_with_native_tools_and_headers(
            session.api_base_url(),
            session.secret(),
            &self.model,
            messages,
            tools,
            &self.provider.identity().headers(),
            get_shared_http_client(),
            &self.token_usage,
        )
        .await
        .map_err(|error| error.to_string())
    }
}

/// Whether an error message reports an HTTP 401 from the Copilot API.
///
/// Both the blocking and the streaming helpers put `HTTP <status>` in their
/// error text.
fn is_unauthorized(error: &str) -> bool {
    error.match_indices("HTTP 401").any(|(index, matched)| {
        !error[index + matched.len()..].starts_with(|next: char| next.is_ascii_digit())
    })
}

#[async_trait]
impl ClientWrapper for CopilotClient {
    fn model_name(&self) -> &str {
        &self.model
    }

    fn provider_name(&self) -> &str {
        "GitHub Copilot"
    }

    /// Send one chat request; uses native tool calling when `tools` is
    /// non-empty.
    ///
    /// # Errors
    ///
    /// A [`CopilotAuthError`] when no session token can be had, otherwise the
    /// Copilot API's error (after one retry with a fresh token on 401).
    async fn send_message(
        &self,
        messages: &[Message],
        tools: Option<&[ToolDefinition]>,
    ) -> Result<Message, Box<dyn Error>> {
        let tools = tools.unwrap_or(&[]);
        let session = self.provider.session_token().await?;
        let first = self.send_once(&session, messages, tools).await;
        let result = match first {
            Err(error) if is_unauthorized(&error) => {
                self.provider.invalidate().await;
                let session = self.provider.session_token().await?;
                self.send_once(&session, messages, tools).await
            }
            other => other,
        };
        result.map_err(|error| {
            log::error!("CopilotClient::send_message: {}", error);
            error.into()
        })
    }

    /// Stream the reply token by token over Server-Sent Events.
    ///
    /// Retries once with a fresh session token if Copilot answers 401 before
    /// the stream starts.
    fn send_message_stream<'a>(
        &'a self,
        messages: &'a [Message],
        tools: Option<&[ToolDefinition]>,
    ) -> MessageStreamFuture<'a> {
        // `tools` may not outlive this call, so the future owns a copy.
        let tools: Option<Vec<ToolDefinition>> = tools.map(<[ToolDefinition]>::to_vec);
        Box::pin(async move {
            let open = |session: CopilotSessionToken| {
                open_chat_completions_stream(
                    session.api_base_url(),
                    session.secret(),
                    &self.model,
                    messages,
                    tools.as_deref(),
                    &self.provider.identity().headers(),
                )
            };
            let session = self.provider.session_token().await?;
            match open(session).await {
                Err(error) if is_unauthorized(&error.to_string()) => {
                    self.provider.invalidate().await;
                    let session = self.provider.session_token().await?;
                    open(session).await
                }
                other => other,
            }
        })
    }

    fn usage_slot(&self) -> Option<&Mutex<Option<TokenUsage>>> {
        Some(&self.token_usage)
    }
}

/// A model listed by the Copilot API's `GET /models`.
///
/// Only `id` is guaranteed; the rest is whatever GitHub reports and may be
/// missing for some models.
///
/// # Examples
///
/// ```rust
/// use cloudllm::clients::github_copilot::CopilotModel;
///
/// let model: CopilotModel = serde_json::from_str(r#"{
///     "id": "gpt-4.1",
///     "name": "GPT-4.1",
///     "vendor": "Azure OpenAI",
///     "model_picker_enabled": true,
///     "capabilities": {"type": "chat", "family": "gpt-4.1",
///                      "supports": {"tool_calls": true, "streaming": true}}
/// }"#).unwrap();
///
/// assert!(model.is_chat());
/// assert!(model.supports_tool_calls());
/// assert_eq!(model.vendor.as_deref(), Some("Azure OpenAI"));
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct CopilotModel {
    /// The ID to pass to [`CopilotClient::new`], for example `"gpt-4.1"`.
    pub id: String,
    /// A display name, for example `"GPT-4.1"`.
    #[serde(default)]
    pub name: Option<String>,
    /// Who serves the model, for example `"Anthropic"`.
    #[serde(default)]
    pub vendor: Option<String>,
    /// Whether GitHub offers the model in its own model pickers.
    #[serde(default)]
    pub model_picker_enabled: Option<bool>,
    /// Whether GitHub marks the model as a preview.
    #[serde(default)]
    pub preview: Option<bool>,
    /// What the model can do.
    #[serde(default)]
    pub capabilities: Option<CopilotModelCapabilities>,
}

impl CopilotModel {
    /// Whether this is a chat model (rather than, say, an embedding model).
    pub fn is_chat(&self) -> bool {
        self.capabilities
            .as_ref()
            .and_then(|capabilities| capabilities.kind.as_deref())
            == Some("chat")
    }

    /// Whether the model accepts native tool calls.
    pub fn supports_tool_calls(&self) -> bool {
        self.capabilities
            .as_ref()
            .and_then(|capabilities| capabilities.supports.as_ref())
            .and_then(|supports| supports.tool_calls)
            .unwrap_or(false)
    }
}

/// The `capabilities` block of a [`CopilotModel`].
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct CopilotModelCapabilities {
    /// The model kind, such as `"chat"` or `"embeddings"`.
    #[serde(default, rename = "type")]
    pub kind: Option<String>,
    /// The model family, for example `"claude-sonnet-4"`.
    #[serde(default)]
    pub family: Option<String>,
    /// Feature flags.
    #[serde(default)]
    pub supports: Option<CopilotModelSupports>,
}

/// Feature flags in [`CopilotModelCapabilities`].
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct CopilotModelSupports {
    /// Native tool calling.
    #[serde(default)]
    pub tool_calls: Option<bool>,
    /// Server-Sent Events streaming.
    #[serde(default)]
    pub streaming: Option<bool>,
}

/// The envelope `GET /models` answers with.
#[derive(Deserialize)]
struct ModelList {
    data: Vec<CopilotModel>,
}

/// Ask the Copilot API which models the signed-in account can use.
///
/// The answer depends on the plan (Free, Pro, Business, Enterprise), on the
/// organisation's policies, and possibly on the OAuth app that minted the
/// GitHub token. Filter with [`CopilotModel::is_chat`] for models that
/// [`CopilotClient`] can talk to.
///
/// # Errors
///
/// Any error from [`CopilotTokenProvider::session_token`];
/// [`CopilotAuthError::HttpStatus`] when the Copilot API refuses;
/// [`CopilotAuthError::InvalidResponse`] when the answer is not a model list.
///
/// # Example
///
/// ```rust,no_run
/// use cloudllm::clients::github_copilot::{list_models, CopilotTokenProvider, GitHubOAuthToken};
///
/// # #[tokio::main]
/// # async fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let provider = CopilotTokenProvider::from_github_token(GitHubOAuthToken::new(
///     std::env::var("GITHUB_COPILOT_OAUTH_TOKEN")?,
/// ));
/// for model in list_models(&provider).await?.iter().filter(|m| m.is_chat()) {
///     println!("{:<28} {}", model.id, model.vendor.as_deref().unwrap_or(""));
/// }
/// # Ok(())
/// # }
/// ```
pub async fn list_models(
    provider: &CopilotTokenProvider,
) -> Result<Vec<CopilotModel>, CopilotAuthError> {
    let session = provider.session_token().await?;
    let url = format!("{}/models", session.api_base_url());
    let request = provider.identity().headers().into_iter().fold(
        get_shared_http_client()
            .get(&url)
            .bearer_auth(session.secret())
            .header("Accept", "application/json"),
        |request, (name, value)| request.header(name, value),
    );
    let response = request.send().await?;
    let status = response.status();
    let text = response.text().await?;
    if !status.is_success() {
        return Err(CopilotAuthError::http_status(status.as_u16(), &text));
    }
    serde_json::from_str::<ModelList>(&text)
        .map(|list| list.data)
        .map_err(|error| CopilotAuthError::InvalidResponse(format!("model list: {error}")))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Verifies that only HTTP 401 errors trigger the token-refresh retry.
    #[test]
    fn only_401_counts_as_unauthorized() {
        assert!(is_unauthorized(
            "send_with_native_tools: HTTP 401 Unauthorized — bad token"
        ));
        assert!(is_unauthorized("stream HTTP 401 Unauthorized from x: y"));
        assert!(!is_unauthorized(
            "send_with_native_tools: HTTP 403 Forbidden"
        ));
        assert!(!is_unauthorized("HTTP 4010"));
    }

    /// Verifies that a model with no capabilities is neither chat nor
    /// tool-capable.
    #[test]
    fn bare_model_has_no_capabilities() {
        let model: CopilotModel = serde_json::from_str(r#"{"id":"x"}"#).unwrap();
        assert!(!model.is_chat());
        assert!(!model.supports_tool_calls());
    }
}
