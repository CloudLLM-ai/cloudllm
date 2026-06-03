//! OpenRouter client wrapper, exposing the top-50 most popular models on
//! [openrouter.ai](https://openrouter.ai) through CloudLLM's [`ClientWrapper`]
//! interface.
//!
//! OpenRouter offers a single OpenAI-compatible Chat Completions endpoint at
//! `https://openrouter.ai/api/v1/chat/completions` that fronts hundreds of
//! upstream models (Anthropic, OpenAI, Google, xAI, DeepSeek, Meta, Mistral,
//! Moonshot, Z-AI, Qwen, MiniMax, and more).  This wrapper mirrors the
//! conventions established by [`crate::clients::grok::GrokClient`] and
//! [`crate::clients::gemini::GeminiClient`]: it owns its own
//! [`openai_rust2::Client`] and routes every request through the
//! OpenRouter-specific base URL + path, so the [`crate::clients::openai::OpenAIClient`]
//! (whose wire paths are hardcoded to `/v1/chat/completions`) is never
//! accidentally used against OpenRouter.
//!
//! # Authentication
//!
//! Set the `OPENROUTER_API_KEY` environment variable to a key obtained from
//! <https://openrouter.ai/keys>.  The constructor itself accepts a `&str` so
//! callers can pull the key from any source they prefer.
//!
//! # Example: chat with the MiniMax M3
//!
//! ```rust,no_run
//! use std::sync::Arc;
//!
//! use cloudllm::client_wrapper::{ClientWrapper, Message, Role};
//! use cloudllm::clients::openrouter::{Model, OpenRouterClient};
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let key = std::env::var("OPENROUTER_API_KEY")?;
//!     let client = OpenRouterClient::new_with_model_enum(&key, Model::MinimaxM3);
//!
//!     let reply = client
//!         .send_message(
//!             &[Message {
//!                 role: Role::User,
//!                 content: Arc::<str>::from("Name three real-world use cases for M3."),
//!                 tool_calls: vec![],
//!             }],
//!             None,
//!         )
//!         .await?;
//!     println!("OpenRouter reply: {}", reply.content);
//!     Ok(())
//! }
//! ```
//!
//! # Example: using an unlisted model via [`OpenRouterClient::new_with_model_str`]
//!
//! ```rust,no_run
//! use std::sync::Arc;
//!
//! use cloudllm::client_wrapper::{ClientWrapper, Message, Role};
//! use cloudllm::clients::openrouter::OpenRouterClient;
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let key = std::env::var("OPENROUTER_API_KEY")?;
//!     // Any `vendor/model` slug accepted by openrouter.ai works.
//!     let client = OpenRouterClient::new_with_model_str(&key, "openai/gpt-5.5");
//!
//!     let reply = client
//!         .send_message(
//!             &[Message {
//!                 role: Role::User,
//!                 content: Arc::<str>::from("What is Rust's ownership model?"),
//!                 tool_calls: vec![],
//!             }],
//!             None,
//!         )
//!         .await?;
//!     println!("{}", reply.content);
//!     Ok(())
//! }
//! ```
//!
//! # Notes
//!
//! - The `Model` enum below lists the top-50 most popular models on OpenRouter
//!   as of May 2026, plus [`Model::MinimaxM3`] (the M-series release CloudLLM
//!   is migrating towards because OpenAI costs are no longer sustainable).
//!   Use [`OpenRouterClient::new_with_model_str`] for anything not in the enum.
//! - OpenRouter's optional attribution headers (`HTTP-Referer`, `X-Title`)
//!   are not yet wired up in v1; add them in a follow-up if needed.

use crate::client_wrapper::{TokenUsage, ToolDefinition};
use crate::clients::common::{get_shared_http_client, send_and_track, send_with_native_tools};
use crate::{ClientWrapper, Message, Role};
use async_trait::async_trait;
use openai_rust::chat;
use openai_rust2 as openai_rust;
use std::error::Error;
use std::sync::Arc;
use tokio::sync::Mutex;

/// Default OpenRouter base URL.  OpenRouter exposes its OpenAI-compatible API
/// under `/api/v1`, so the chat completions path we hand to `openai-rust2` is
/// `/api/v1/chat/completions` (not the plain `/v1/chat/completions` used by
/// vanilla OpenAI).
pub const OPENROUTER_DEFAULT_BASE_URL: &str = "https://openrouter.ai/api/v1";

/// Top OpenRouter model identifiers, ordered roughly by weekly token share
/// (May 2026 snapshot) with [`Model::MinimaxM3`] promoted to the front because
/// it is the migration target for this crate.
///
/// The list is intentionally exhaustive at the top of the market: any model
/// not represented here can still be used through
/// [`OpenRouterClient::new_with_model_str`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[allow(non_camel_case_types)]
pub enum Model {
    /// `minimax/minimax-m3` – migration target.  Multimodal M3 release from
    /// MiniMax with a 1M-token context window.
    MinimaxM3,
    /// `tencent/hy3-preview` – Tencent's Mixture-of-Experts preview tuned for
    /// agentic workflows.
    TencentHy3Preview,
    /// `deepseek/deepseek-v4-flash` – DeepSeek v4 fast tier.
    DeepSeekV4Flash,
    /// `anthropic/claude-opus-4.7` – Anthropic Opus 4.7.
    ClaudeOpus47,
    /// `anthropic/claude-sonnet-4.6` – Anthropic Sonnet 4.6.
    ClaudeSonnet46,
    /// `openrouter/owl-alpha` – OpenRouter's own alpha model.
    OpenRouterOwlAlpha,
    /// `xiaomi/mimo-v2.5` – Xiaomi MiMo v2.5.
    XiaomiMimoV25,
    /// `xiaomi/mimo-v2.5-pro` – Xiaomi MiMo v2.5 Pro.
    XiaomiMimoV25Pro,
    /// `deepseek/deepseek-v4-pro` – DeepSeek v4 Pro.
    DeepSeekV4Pro,
    /// `deepseek/deepseek-v3.2` – DeepSeek v3.2.
    DeepSeekV32,
    /// `google/gemini-3-flash-preview` – Google Gemini 3 Flash (preview).
    Gemini3FlashPreview,
    /// `nvidia/nemotron-3-super-120b-a12b` – NVIDIA Nemotron 3 Super 120B.
    NvidiaNemotron3Super120bA12b,
    /// `google/gemini-2.5-flash-lite` – Google Gemini 2.5 Flash Lite.
    Gemini25FlashLite,
    /// `google/gemini-2.5-flash` – Google Gemini 2.5 Flash.
    Gemini25Flash,
    /// `poolside/laguna-m.1` – Poolside Laguna M.1.
    PoolsideLagunaM1,
    /// `anthropic/claude-opus-4.6` – Anthropic Opus 4.6.
    ClaudeOpus46,
    /// `google/gemini-3.5-flash` – Google Gemini 3.5 Flash.
    Gemini35Flash,
    /// `minimax/minimax-m2.7` – MiniMax M2.7.
    MinimaxM27,
    /// `moonshotai/kimi-k2.6` – Moonshot Kimi K2.6.
    MoonshotKimiK26,
    /// `openai/gpt-4o-mini` – OpenAI GPT-4o mini via OpenRouter.
    GPT4oMini,
    /// `openai/gpt-5.5` – OpenAI GPT-5.5 via OpenRouter.
    GPT55,
    /// `anthropic/claude-opus-4.8` – Anthropic Opus 4.8.
    ClaudeOpus48,
    /// `openai/gpt-oss-120b` – OpenAI GPT-OSS 120B.
    GPTOSS120B,
    /// `google/gemini-3.1-flash-lite` – Google Gemini 3.1 Flash Lite.
    Gemini31FlashLite,
    /// `google/gemma-4-31b-it` – Google Gemma 4 31B IT.
    Gemma431BIt,
    /// `z-ai/glm-5.1` – Z-AI GLM 5.1.
    ZAiGlm51,
    /// `google/gemini-3.1-pro-preview` – Google Gemini 3.1 Pro preview.
    Gemini31ProPreview,
    /// `openai/gpt-5.4` – OpenAI GPT-5.4 via OpenRouter.
    GPT54,
    /// `qwen/qwen3-235b-a22b-2507` – Qwen 3 235B A22B (July 2025 snapshot).
    Qwen3235bA22b2507,
    /// `anthropic/claude-haiku-4.5` – Anthropic Haiku 4.5.
    ClaudeHaiku45,
    /// `google/gemma-4-26b-a4b-it` – Google Gemma 4 26B A4B IT.
    Gemma426bA4bIt,
    /// `stepfun/step-3.7-flash` – StepFun Step 3.7 Flash.
    StepfunStep37Flash,
    /// `google/gemini-3.1-flash-lite-preview` – Google Gemini 3.1 Flash Lite preview.
    Gemini31FlashLitePreview,
    /// `z-ai/glm-4.7` – Z-AI GLM 4.7.
    ZAiGlm47,
    /// `qwen/qwen3.6-plus` – Qwen 3.6 Plus.
    Qwen36Plus,
    /// `qwen/qwen3.7-max` – Qwen 3.7 Max.
    Qwen37Max,
    /// `minimax/minimax-m2.5` – MiniMax M2.5.
    MinimaxM25,
    /// `openai/gpt-5.4-mini` – OpenAI GPT-5.4 mini via OpenRouter.
    GPT54Mini,
    /// `openai/gpt-5-mini` – OpenAI GPT-5 mini via OpenRouter.
    GPT5Mini,
    /// `moonshotai/kimi-k2.5` – Moonshot Kimi K2.5.
    MoonshotKimiK25,
    /// `mistralai/mistral-nemo` – Mistral Nemo.
    MistralNemo,
    /// `z-ai/glm-5` – Z-AI GLM 5.
    ZAiGlm5,
    /// `z-ai/glm-4.5-air` – Z-AI GLM 4.5 Air.
    ZAiGlm45Air,
    /// `qwen/qwen3-embedding-8b` – Qwen 3 Embedding 8B.
    Qwen3Embedding8B,
    /// `anthropic/claude-sonnet-4.5` – Anthropic Sonnet 4.5.
    ClaudeSonnet45,
    /// `qwen/qwen3.5-flash-02-23` – Qwen 3.5 Flash (Feb-23 snapshot).
    Qwen35Flash0223,
    /// `openai/gpt-5.4-nano` – OpenAI GPT-5.4 nano via OpenRouter.
    GPT54Nano,
    /// `openai/gpt-5.3-codex` – OpenAI GPT-5.3 Codex via OpenRouter.
    GPT53Codex,
    /// `poolside/laguna-xs.2` – Poolside Laguna XS.2.
    PoolsideLagunaXs2,
    /// `meta-llama/llama-3.1-8b-instruct` – Meta Llama 3.1 8B Instruct.
    Llama318bInstruct,
    /// `x-ai/grok-4.3` – xAI Grok 4.3 via OpenRouter.
    Grok43,
}

/// Convert a [`Model`] variant into its OpenRouter `vendor/model` slug.
pub fn model_to_string(model: Model) -> String {
    match model {
        Model::MinimaxM3 => "minimax/minimax-m3".to_string(),
        Model::TencentHy3Preview => "tencent/hy3-preview".to_string(),
        Model::DeepSeekV4Flash => "deepseek/deepseek-v4-flash".to_string(),
        Model::ClaudeOpus47 => "anthropic/claude-opus-4.7".to_string(),
        Model::ClaudeSonnet46 => "anthropic/claude-sonnet-4.6".to_string(),
        Model::OpenRouterOwlAlpha => "openrouter/owl-alpha".to_string(),
        Model::XiaomiMimoV25 => "xiaomi/mimo-v2.5".to_string(),
        Model::XiaomiMimoV25Pro => "xiaomi/mimo-v2.5-pro".to_string(),
        Model::DeepSeekV4Pro => "deepseek/deepseek-v4-pro".to_string(),
        Model::DeepSeekV32 => "deepseek/deepseek-v3.2".to_string(),
        Model::Gemini3FlashPreview => "google/gemini-3-flash-preview".to_string(),
        Model::NvidiaNemotron3Super120bA12b => "nvidia/nemotron-3-super-120b-a12b".to_string(),
        Model::Gemini25FlashLite => "google/gemini-2.5-flash-lite".to_string(),
        Model::Gemini25Flash => "google/gemini-2.5-flash".to_string(),
        Model::PoolsideLagunaM1 => "poolside/laguna-m.1".to_string(),
        Model::ClaudeOpus46 => "anthropic/claude-opus-4.6".to_string(),
        Model::Gemini35Flash => "google/gemini-3.5-flash".to_string(),
        Model::MinimaxM27 => "minimax/minimax-m2.7".to_string(),
        Model::MoonshotKimiK26 => "moonshotai/kimi-k2.6".to_string(),
        Model::GPT4oMini => "openai/gpt-4o-mini".to_string(),
        Model::GPT55 => "openai/gpt-5.5".to_string(),
        Model::ClaudeOpus48 => "anthropic/claude-opus-4.8".to_string(),
        Model::GPTOSS120B => "openai/gpt-oss-120b".to_string(),
        Model::Gemini31FlashLite => "google/gemini-3.1-flash-lite".to_string(),
        Model::Gemma431BIt => "google/gemma-4-31b-it".to_string(),
        Model::ZAiGlm51 => "z-ai/glm-5.1".to_string(),
        Model::Gemini31ProPreview => "google/gemini-3.1-pro-preview".to_string(),
        Model::GPT54 => "openai/gpt-5.4".to_string(),
        Model::Qwen3235bA22b2507 => "qwen/qwen3-235b-a22b-2507".to_string(),
        Model::ClaudeHaiku45 => "anthropic/claude-haiku-4.5".to_string(),
        Model::Gemma426bA4bIt => "google/gemma-4-26b-a4b-it".to_string(),
        Model::StepfunStep37Flash => "stepfun/step-3.7-flash".to_string(),
        Model::Gemini31FlashLitePreview => "google/gemini-3.1-flash-lite-preview".to_string(),
        Model::ZAiGlm47 => "z-ai/glm-4.7".to_string(),
        Model::Qwen36Plus => "qwen/qwen3.6-plus".to_string(),
        Model::Qwen37Max => "qwen/qwen3.7-max".to_string(),
        Model::MinimaxM25 => "minimax/minimax-m2.5".to_string(),
        Model::GPT54Mini => "openai/gpt-5.4-mini".to_string(),
        Model::GPT5Mini => "openai/gpt-5-mini".to_string(),
        Model::MoonshotKimiK25 => "moonshotai/kimi-k2.5".to_string(),
        Model::MistralNemo => "mistralai/mistral-nemo".to_string(),
        Model::ZAiGlm5 => "z-ai/glm-5".to_string(),
        Model::ZAiGlm45Air => "z-ai/glm-4.5-air".to_string(),
        Model::Qwen3Embedding8B => "qwen/qwen3-embedding-8b".to_string(),
        Model::ClaudeSonnet45 => "anthropic/claude-sonnet-4.5".to_string(),
        Model::Qwen35Flash0223 => "qwen/qwen3.5-flash-02-23".to_string(),
        Model::GPT54Nano => "openai/gpt-5.4-nano".to_string(),
        Model::GPT53Codex => "openai/gpt-5.3-codex".to_string(),
        Model::PoolsideLagunaXs2 => "poolside/laguna-xs.2".to_string(),
        Model::Llama318bInstruct => "meta-llama/llama-3.1-8b-instruct".to_string(),
        Model::Grok43 => "x-ai/grok-4.3".to_string(),
    }
}

/// OpenRouter client wrapper.  Holds its own [`openai_rust2::Client`] pointed at
/// the OpenRouter base URL so the wire path can be `/api/v1/chat/completions`
/// (vanilla OpenAI uses `/v1/chat/completions`).
pub struct OpenRouterClient {
    /// Underlying SDK client pointing at the OpenRouter REST endpoint.
    client: openai_rust::Client,
    /// Selected model slug, e.g. `"minimax/minimax-m3"`.
    model: String,
    /// Storage for the token usage returned by the most recent request.
    token_usage: Mutex<Option<TokenUsage>>,
    /// API key, kept for raw-`reqwest` calls (tool calling, future attribution headers).
    api_key: String,
    /// Normalized base URL, e.g. `"https://openrouter.ai/api/v1"` (no trailing slash).
    base_url: String,
}

impl OpenRouterClient {
    /// Construct a client from an API key and a strongly typed [`Model`].
    pub fn new_with_model_enum(secret_key: &str, model: Model) -> Self {
        Self::new_with_model_str(secret_key, &model_to_string(model))
    }

    /// Construct a client from an API key and an explicit OpenRouter model
    /// slug (e.g. `"minimax/minimax-m3"` or `"openai/gpt-5.5"`).
    ///
    /// Use this constructor for models that are not in the [`Model`] enum —
    /// OpenRouter adds new entries regularly and we cannot keep the enum
    /// perfectly in sync.
    pub fn new_with_model_str(secret_key: &str, model_name: &str) -> Self {
        Self::new_with_base_url(secret_key, model_name, OPENROUTER_DEFAULT_BASE_URL)
    }

    /// Construct a client targeting a custom OpenAI-compatible base URL.
    ///
    /// `base_url` should not have a trailing slash
    /// (e.g. `"https://openrouter.ai/api/v1"`).  Useful for self-hosted
    /// OpenRouter-compatible gateways.
    pub fn new_with_base_url(secret_key: &str, model_name: &str, base_url: &str) -> Self {
        let base_url_normalized = base_url.trim_end_matches('/');
        OpenRouterClient {
            client: openai_rust::Client::new_with_client_and_base_url(
                secret_key,
                get_shared_http_client().clone(),
                &format!("{}/", base_url_normalized),
            ),
            model: model_name.to_string(),
            token_usage: Mutex::new(None),
            api_key: secret_key.to_string(),
            base_url: base_url_normalized.to_string(),
        }
    }

    /// Convenience wrapper around [`OpenRouterClient::new_with_base_url`] for
    /// strongly typed models.
    pub fn new_with_base_url_and_model_enum(
        secret_key: &str,
        model: Model,
        base_url: &str,
    ) -> Self {
        Self::new_with_base_url(secret_key, &model_to_string(model), base_url)
    }
}

#[async_trait]
impl ClientWrapper for OpenRouterClient {
    fn model_name(&self) -> &str {
        &self.model
    }

    fn provider_name(&self) -> &str {
        "OpenRouter"
    }

    /// Send a chat completion, routing to native tool calling when `tools` is non-empty.
    ///
    /// When `tools` is `Some` and non-empty the request is forwarded to
    /// [`send_with_native_tools`](crate::clients::common::send_with_native_tools)
    /// (OpenRouter advertises native tool-calling support on the models
    /// exposed here).  Otherwise the standard Chat Completions endpoint is
    /// used with the OpenRouter-specific path.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use std::sync::Arc;
    /// use cloudllm::client_wrapper::{ClientWrapper, Message, Role};
    /// use cloudllm::clients::openrouter::{Model, OpenRouterClient};
    ///
    /// # #[tokio::main]
    /// # async fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let client = OpenRouterClient::new_with_model_enum(
    ///     &std::env::var("OPENROUTER_API_KEY")?,
    ///     Model::MinimaxM3,
    /// );
    /// let resp = client.send_message(
    ///     &[Message { role: Role::User, content: Arc::from("Hello"), tool_calls: vec![] }],
    ///     None,
    /// ).await?;
    /// println!("{}", resp.content);
    /// # Ok(())
    /// # }
    /// ```
    async fn send_message(
        &self,
        messages: &[Message],
        tools: Option<Vec<ToolDefinition>>,
    ) -> Result<Message, Box<dyn Error>> {
        if let Some(tool_defs) = tools.filter(|t| !t.is_empty()) {
            return send_with_native_tools(
                &self.base_url,
                &self.api_key,
                &self.model,
                messages,
                &tool_defs,
                get_shared_http_client(),
                &self.token_usage,
            )
            .await
            .map_err(|e| {
                if log::log_enabled!(log::Level::Error) {
                    log::error!("OpenRouterClient::send_message (native tools): {}", e);
                }
                e
            });
        }

        let mut formatted_messages = Vec::with_capacity(messages.len());
        for msg in messages {
            formatted_messages.push(chat::Message {
                role: match &msg.role {
                    Role::System => "system".to_owned(),
                    Role::User => "user".to_owned(),
                    Role::Assistant => "assistant".to_owned(),
                    Role::Tool { .. } => "tool".to_owned(),
                },
                content: msg.content.to_string(),
            });
        }

        let result = send_and_track(
            &self.client,
            &self.model,
            formatted_messages,
            Some("/api/v1/chat/completions".to_string()),
            &self.token_usage,
            None,
        )
        .await;

        match result {
            Ok(content) => Ok(Message {
                role: Role::Assistant,
                content: Arc::from(content.as_str()),
                tool_calls: vec![],
            }),
            Err(e) => {
                if log::log_enabled!(log::Level::Error) {
                    log::error!("OpenRouterClient::send_message: API Error: {}", e);
                }
                Err(e)
            }
        }
    }

    /// OpenRouter forwards streaming requests to the underlying provider.
    /// The request is issued against `/api/v1/chat/completions` with
    /// `stream=true` and the resulting SSE bytes are split into
    /// [`crate::client_wrapper::MessageChunk`]s using the same approach
    /// [`crate::clients::openai::OpenAIClient`] uses.
    fn send_message_stream<'a>(
        &'a self,
        messages: &'a [Message],
        _tools: Option<Vec<ToolDefinition>>,
    ) -> crate::client_wrapper::MessageStreamFuture<'a> {
        Box::pin(async move {
            let mut formatted_messages = Vec::with_capacity(messages.len());
            for msg in messages {
                formatted_messages.push(chat::Message {
                    role: match &msg.role {
                        Role::System => "system".to_owned(),
                        Role::User => "user".to_owned(),
                        Role::Assistant => "assistant".to_owned(),
                        Role::Tool { .. } => "tool".to_owned(),
                    },
                    content: msg.content.to_string(),
                });
            }

            let chat_arguments = chat::ChatArguments::new(&self.model, formatted_messages);
            let stream_result = self
                .client
                .create_chat_stream(chat_arguments, Some("/api/v1/chat/completions".to_string()))
                .await;

            match stream_result {
                Ok(mut chunk_stream) => {
                    use futures_util::StreamExt;

                    let mut chunks: Vec<
                        Result<crate::client_wrapper::MessageChunk, Box<dyn Error + Send>>,
                    > = Vec::new();

                    while let Some(chunk_result) = chunk_stream.next().await {
                        let message_chunk = match chunk_result {
                            Ok(chunk) => {
                                let content = chunk
                                    .choices
                                    .first()
                                    .and_then(|choice| choice.delta.content.clone())
                                    .unwrap_or_default();
                                let finish_reason = chunk
                                    .choices
                                    .first()
                                    .and_then(|choice| choice.finish_reason.clone());
                                Ok(crate::client_wrapper::MessageChunk {
                                    content,
                                    finish_reason,
                                })
                            }
                            Err(err) => {
                                if log::log_enabled!(log::Level::Error) {
                                    log::error!(
                                        "OpenRouterClient::send_message_stream: chunk error: {}",
                                        err
                                    );
                                }
                                Err(Box::new(crate::clients::common::StreamError(format!(
                                    "Stream chunk error: {}",
                                    err
                                ))) as Box<dyn Error + Send>)
                            }
                        };
                        chunks.push(message_chunk);
                    }

                    Ok(Some(crate::clients::common::chunks_to_stream(chunks)))
                }
                Err(err) => {
                    if log::log_enabled!(log::Level::Error) {
                        log::error!("OpenRouterClient::send_message_stream: API Error: {}", err);
                    }
                    Err(err.into())
                }
            }
        })
    }

    fn usage_slot(&self) -> Option<&Mutex<Option<TokenUsage>>> {
        Some(&self.token_usage)
    }
}
