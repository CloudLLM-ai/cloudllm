//! GitHub Copilot as a CloudLLM provider: sign in with a GitHub account and
//! use the models a Copilot plan includes.
//!
//! A Copilot subscription gives access to models from several vendors (GPT,
//! Claude, Gemini and others) behind one OpenAI-compatible API. The first step
//! is a GitHub OAuth token: the user signs in once with GitHub's device flow
//! ([`GitHubDeviceFlow`]). CloudLLM shows a short code, the user types it at
//! <https://github.com/login/device>, and GitHub hands back a long-lived token
//! you can store.
//!
//! The second step trades that token for a short-lived Copilot session token
//! ([`CopilotTokenProvider`]), which also names the API host for the account.
//! The provider caches the session token and trades again before it expires.
//!
//! [`CopilotClient`] then speaks Copilot's OpenAI-compatible Chat Completions
//! API as an ordinary [`ClientWrapper`](crate::client_wrapper::ClientWrapper),
//! so it works anywhere another CloudLLM client does: in an
//! [`LLMSession`](crate::LLMSession), an [`Agent`](crate::Agent), or an
//! orchestration.
//!
//! # Modules
//!
//! - [`client`]: [`CopilotClient`] and [`list_models`].
//! - [`device_flow`]: the GitHub OAuth device-flow login.
//! - [`session_token`]: session tokens and the [`CopilotIdentity`] headers.
//! - [`error`]: [`CopilotAuthError`], shared by every step.

pub mod client;
pub mod device_flow;
pub mod error;
pub mod session_token;

pub use client::{
    list_models, CopilotClient, CopilotModel, CopilotModelCapabilities, CopilotModelSupports,
};
pub use device_flow::{
    DeviceCode, GitHubDeviceFlow, GitHubOAuthToken, PollOutcome, DEFAULT_SCOPE,
    GITHUB_DEFAULT_BASE_URL,
};
pub use error::CopilotAuthError;
pub use session_token::{
    CopilotIdentity, CopilotSessionToken, CopilotTokenProvider, COPILOT_DEFAULT_API_BASE_URL,
    DEFAULT_REFRESH_MARGIN, GITHUB_API_DEFAULT_BASE_URL,
};
