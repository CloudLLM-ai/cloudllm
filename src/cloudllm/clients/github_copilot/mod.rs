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
//! # Modules
//!
//! - [`device_flow`]: the GitHub OAuth device-flow login.
//! - [`error`]: [`CopilotAuthError`], shared by every step.

pub mod device_flow;
pub mod error;

pub use device_flow::{
    DeviceCode, GitHubDeviceFlow, GitHubOAuthToken, PollOutcome, DEFAULT_SCOPE,
    GITHUB_DEFAULT_BASE_URL,
};
pub use error::CopilotAuthError;
