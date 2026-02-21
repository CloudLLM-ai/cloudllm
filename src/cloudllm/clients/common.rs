//! Shared utilities used across provider client implementations.
//!
//! The helpers in this module are useful when implementing additional providers that expose an
//! OpenAI-compatible HTTP surface.  They provide a tuned [`reqwest`] client, convenience
//! functions for sending chat requests, and adapters for streaming responses.
//!
//! # Example: building a custom wrapper
//!
//! ```rust,no_run
//! use std::sync::Arc;
//!
//! use async_trait::async_trait;
//! use cloudllm::client_wrapper::{ClientWrapper, Message, Role, TokenUsage};
//! use cloudllm::clients::common::{get_shared_http_client, send_and_track};
//! use openai_rust2 as openai_rust;
//! use openai_rust2::chat::GrokTool;
//! use tokio::sync::Mutex;
//!
//! struct MyHostedClient {
//!     client: openai_rust::Client,
//!     model: String,
//!     usage: Mutex<Option<TokenUsage>>,
//! }
//!
//! impl MyHostedClient {
//!     fn new(key: &str, base_url: &str, model: &str) -> Self {
//!         Self {
//!             client: openai_rust::Client::new_with_client_and_base_url(
//!                 key,
//!                 get_shared_http_client().clone(),
//!                 base_url,
//!             ),
//!             model: model.to_owned(),
//!             usage: Mutex::new(None),
//!         }
//!     }
//! }
//!
//! #[async_trait]
//! impl ClientWrapper for MyHostedClient {
//!     fn model_name(&self) -> &str {
//!         &self.model
//!     }
//!
//!     async fn send_message(
//!         &self,
//!         messages: &[Message],
//!         optional_grok_tools: Option<Vec<GrokTool>>,
//!     ) -> Result<Message, Box<dyn std::error::Error>> {
//!         let formatted = messages
//!             .iter()
//!             .map(|msg| openai_rust::chat::Message {
//!                 role: match msg.role {
//!                     Role::System => "system".into(),
//!                     Role::User => "user".into(),
//!                     Role::Assistant => "assistant".into(),
//!                 },
//!                 content: msg.content.as_ref().to_owned(),
//!             })
//!             .collect();
//!
//!         let reply = send_and_track(
//!             &self.client,
//!             &self.model,
//!             formatted,
//!             Some("/v1/chat/completions".to_string()),
//!             &self.usage,
//!             optional_grok_tools,
//!         )
//!         .await?;
//!
//!         Ok(Message {
//!             role: Role::Assistant,
//!             content: Arc::<str>::from(reply),
//!         })
//!     }
//! }
//! ```
//!
//! The same helpers can be combined with [`chunks_to_stream`] to wire streaming support into the
//! custom client.

use crate::client_wrapper::{Message, MessageChunk, NativeToolCall, Role, TokenUsage, ToolDefinition};
use lazy_static::lazy_static;
use openai_rust::chat;
use openai_rust::chat::{
    GrokTool, OpenAIResponsesArguments, OpenAITool, ResponsesArguments, ResponsesMessage,
};
use openai_rust2 as openai_rust;
use std::error::Error;
use std::time::Duration;
use tokio::sync::Mutex;

lazy_static! {
    /// Shared HTTP client with persistent connection pooling.
    ///
    /// The single client instance keeps TLS sessions and DNS lookups warm which significantly
    /// reduces latency when many concurrent requests are issued to upstream providers.
    static ref SHARED_HTTP_CLIENT: reqwest::Client = {
        reqwest::ClientBuilder::new()
            .pool_idle_timeout(Some(Duration::from_secs(90)))
            .pool_max_idle_per_host(10)
            .tcp_keepalive(Some(Duration::from_secs(60)))
            .timeout(Duration::from_secs(300))
            .connect_timeout(Duration::from_secs(30))
            .build()
            .expect("Failed to build shared HTTP client")
    };
}

/// Borrow the lazily initialised shared [`reqwest::Client`].
///
/// The returned reference can be cloned and reused by individual client wrappers.
pub fn get_shared_http_client() -> &'static reqwest::Client {
    &SHARED_HTTP_CLIENT
}

/// Send a chat completion request, persist token usage, and surface the assistant content.
///
/// The helper captures the common logic shared by OpenAI-compatible endpoints (OpenAI, Anthropic
/// via the Claude proxy, Gemini, xAI Grok, etc.).
pub async fn send_and_track(
    api: &openai_rust::Client,
    model: &str,
    formatted_msgs: Vec<chat::Message>,
    url_path: Option<String>,
    usage_slot: &Mutex<Option<TokenUsage>>,
    optional_grok_tools: Option<Vec<GrokTool>>,
) -> Result<String, Box<dyn Error>> {
    let mut chat_arguments = chat::ChatArguments::new(model, formatted_msgs);

    if let Some(grok_tools) = optional_grok_tools {
        chat_arguments = chat_arguments.with_grok_tools(grok_tools);
    }

    let response = api.create_chat(chat_arguments, url_path).await;

    match response {
        Ok(response) => {
            let usage = TokenUsage {
                input_tokens: response.usage.prompt_tokens as usize,
                output_tokens: response.usage.completion_tokens as usize,
                total_tokens: response.usage.total_tokens as usize,
            };

            // Store it for get_last_usage()
            *usage_slot.lock().await = Some(usage);

            // Return the assistant’s content
            Ok(response.choices[0].message.content.clone())
        }
        Err(err) => {
            if log::log_enabled!(log::Level::Error) {
                log::error!(
                    "cloudllm::clients::common::send_and_track(...): OpenAI API Error: {}",
                    err
                );
            }
            Err(err.into()) // Convert the error to Box<dyn Error>
        }
    }
}

/// Send a request to xAI's Responses API (/v1/responses) with agentic tool calling.
///
/// This function is used when grok_tools are provided, as the Responses API uses
/// a different endpoint and request/response format than Chat Completions.
pub async fn send_and_track_responses(
    api: &openai_rust::Client,
    model: &str,
    formatted_msgs: Vec<chat::Message>,
    url_path: Option<String>,
    usage_slot: &Mutex<Option<TokenUsage>>,
    grok_tools: Vec<GrokTool>,
) -> Result<String, Box<dyn Error>> {
    // Convert chat messages to ResponsesMessage format
    let input: Vec<ResponsesMessage> = formatted_msgs
        .into_iter()
        .map(|msg| ResponsesMessage {
            role: msg.role,
            content: msg.content,
        })
        .collect();

    let args = ResponsesArguments::new(model, input).with_tools(grok_tools);

    let response = api.create_responses(args, url_path).await;

    match response {
        Ok(response) => {
            let usage = TokenUsage {
                input_tokens: response.usage.input_tokens as usize,
                output_tokens: response.usage.output_tokens as usize,
                total_tokens: response.usage.total_tokens as usize,
            };

            // Store it for get_last_usage()
            *usage_slot.lock().await = Some(usage);

            // Return the assistant's content
            Ok(response.get_text_content())
        }
        Err(err) => {
            if log::log_enabled!(log::Level::Error) {
                log::error!(
                    "cloudllm::clients::common::send_and_track_responses(...): xAI Responses API Error: {}",
                    err
                );
            }
            Err(err.into())
        }
    }
}

/// Send a request to OpenAI's Responses API (/v1/responses) with agentic tool calling.
///
/// This function is used when openai_tools are provided (web_search, file_search, code_interpreter),
/// as the Responses API uses a different endpoint and request/response format than Chat Completions.
pub async fn send_and_track_openai_responses(
    api: &openai_rust::Client,
    model: &str,
    formatted_msgs: Vec<chat::Message>,
    url_path: Option<String>,
    usage_slot: &Mutex<Option<TokenUsage>>,
    openai_tools: Vec<OpenAITool>,
) -> Result<String, Box<dyn Error>> {
    // Convert chat messages to ResponsesMessage format
    let input: Vec<ResponsesMessage> = formatted_msgs
        .into_iter()
        .map(|msg| ResponsesMessage {
            role: msg.role,
            content: msg.content,
        })
        .collect();

    let args = OpenAIResponsesArguments::new(model, input).with_tools(openai_tools);

    let response = api.create_openai_responses(args, url_path).await;

    match response {
        Ok(response) => {
            let usage = TokenUsage {
                input_tokens: response.usage.input_tokens as usize,
                output_tokens: response.usage.output_tokens as usize,
                total_tokens: response.usage.total_tokens as usize,
            };

            // Store it for get_last_usage()
            *usage_slot.lock().await = Some(usage);

            // Return the assistant's content (with citations extracted)
            Ok(response.get_text_content())
        }
        Err(err) => {
            if log::log_enabled!(log::Level::Error) {
                log::error!(
                    "cloudllm::clients::common::send_and_track_openai_responses(...): OpenAI Responses API Error: {}",
                    err
                );
            }
            Err(err.into())
        }
    }
}

/// Call the OpenAI-compatible Chat Completions endpoint with native tool definitions.
///
/// Posts to `{base_url}/chat/completions` with an `Authorization: Bearer {api_key}` header.
/// The response is parsed to extract the assistant content string and any tool calls the model
/// requested.  Token usage is persisted in `usage_slot` so callers can retrieve it via
/// [`ClientWrapper::get_last_usage`](crate::client_wrapper::ClientWrapper::get_last_usage).
///
/// Compatible with OpenAI, Anthropic Claude (via its OpenAI-compatible endpoint), xAI Grok, and
/// Google Gemini.
///
/// # Message serialisation
///
/// | [`Role`] variant | Wire representation |
/// |---|---|
/// | `System` | `{"role":"system","content":"..."}` |
/// | `User` | `{"role":"user","content":"..."}` |
/// | `Assistant` with tool_calls | `{"role":"assistant","content":null,"tool_calls":[...]}` |
/// | `Assistant` without tool_calls | `{"role":"assistant","content":"..."}` |
/// | `Tool { call_id }` | `{"role":"tool","tool_call_id":"<id>","content":"..."}` |
///
/// # Example
///
/// ```rust,no_run
/// use std::sync::Arc;
/// use tokio::sync::Mutex;
/// use cloudllm::client_wrapper::{Message, Role, ToolDefinition};
/// use cloudllm::clients::common::{get_shared_http_client, send_with_native_tools};
///
/// # #[tokio::main]
/// # async fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let usage = Mutex::new(None);
/// let tool = ToolDefinition {
///     name: "calculator".to_string(),
///     description: "Evaluates math".to_string(),
///     parameters_schema: serde_json::json!({"type":"object","properties":{}}),
/// };
/// let msg = Message {
///     role: Role::User,
///     content: Arc::from("What is 2+2?"),
///     tool_calls: vec![],
/// };
/// let reply = send_with_native_tools(
///     "https://api.openai.com/v1",
///     &std::env::var("OPEN_AI_SECRET")?,
///     "gpt-4.1-nano",
///     &[msg],
///     &[tool],
///     get_shared_http_client(),
///     &usage,
/// ).await?;
/// println!("{}", reply.content);
/// # Ok(())
/// # }
/// ```
pub async fn send_with_native_tools(
    base_url: &str,
    api_key: &str,
    model: &str,
    messages: &[Message],
    tools: &[ToolDefinition],
    http_client: &reqwest::Client,
    usage_slot: &Mutex<Option<TokenUsage>>,
) -> Result<Message, Box<dyn Error>> {
    // Serialise messages to OpenAI wire format
    let wire_messages: Vec<serde_json::Value> = messages
        .iter()
        .map(|msg| match &msg.role {
            Role::System => serde_json::json!({
                "role": "system",
                "content": msg.content.as_ref()
            }),
            Role::User => serde_json::json!({
                "role": "user",
                "content": msg.content.as_ref()
            }),
            Role::Assistant => {
                if msg.tool_calls.is_empty() {
                    serde_json::json!({
                        "role": "assistant",
                        "content": msg.content.as_ref()
                    })
                } else {
                    let tool_calls: Vec<serde_json::Value> = msg
                        .tool_calls
                        .iter()
                        .map(|tc| serde_json::json!({
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.name,
                                "arguments": serde_json::to_string(&tc.arguments)
                                    .unwrap_or_else(|_| "{}".to_string())
                            }
                        }))
                        .collect();
                    serde_json::json!({
                        "role": "assistant",
                        "content": serde_json::Value::Null,
                        "tool_calls": tool_calls
                    })
                }
            }
            Role::Tool { call_id } => serde_json::json!({
                "role": "tool",
                "tool_call_id": call_id,
                "content": msg.content.as_ref()
            }),
        })
        .collect();

    // Serialise tools array
    let wire_tools: Vec<serde_json::Value> = tools
        .iter()
        .map(|t| serde_json::json!({
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description,
                "parameters": t.parameters_schema
            }
        }))
        .collect();

    let body = serde_json::json!({
        "model": model,
        "messages": wire_messages,
        "tools": wire_tools
    });

    let url = format!("{}/chat/completions", base_url.trim_end_matches('/'));

    let resp = http_client
        .post(&url)
        .header("Authorization", format!("Bearer {}", api_key))
        .header("Content-Type", "application/json")
        .json(&body)
        .send()
        .await
        .map_err(|e| -> Box<dyn Error> { Box::new(e) })?;

    let status = resp.status();
    let text = resp.text().await.map_err(|e| -> Box<dyn Error> { Box::new(e) })?;

    if !status.is_success() {
        if log::log_enabled!(log::Level::Error) {
            log::error!(
                "send_with_native_tools: HTTP {} from {}: {}",
                status, url, text
            );
        }
        return Err(format!("send_with_native_tools: HTTP {} — {}", status, text).into());
    }

    let parsed: serde_json::Value = serde_json::from_str(&text)
        .map_err(|e| -> Box<dyn Error> { Box::new(e) })?;

    // Store token usage
    if let Some(usage_obj) = parsed.get("usage") {
        let input = usage_obj
            .get("prompt_tokens")
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as usize;
        let output = usage_obj
            .get("completion_tokens")
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as usize;
        *usage_slot.lock().await = Some(TokenUsage {
            input_tokens: input,
            output_tokens: output,
            total_tokens: input + output,
        });
    }

    // Extract message from choices[0].message
    let choice_msg = parsed
        .get("choices")
        .and_then(|c| c.get(0))
        .and_then(|c| c.get("message"))
        .ok_or_else(|| -> Box<dyn Error> { "send_with_native_tools: no choices in response".into() })?;

    let content: std::sync::Arc<str> = choice_msg
        .get("content")
        .and_then(|c| c.as_str())
        .map(|s| std::sync::Arc::from(s))
        .unwrap_or_else(|| std::sync::Arc::from(""));

    // Parse native tool calls if present
    let tool_calls: Vec<NativeToolCall> = choice_msg
        .get("tool_calls")
        .and_then(|tc| tc.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|tc| {
                    let id = tc.get("id")?.as_str()?.to_string();
                    let func = tc.get("function")?;
                    let name = func.get("name")?.as_str()?.to_string();
                    let args_str = func.get("arguments")?.as_str().unwrap_or("{}");
                    let arguments: serde_json::Value =
                        serde_json::from_str(args_str).unwrap_or(serde_json::Value::Object(
                            serde_json::Map::new(),
                        ));
                    Some(NativeToolCall { id, name, arguments })
                })
                .collect()
        })
        .unwrap_or_default();

    Ok(Message {
        role: Role::Assistant,
        content,
        tool_calls,
    })
}

/// Thin error wrapper used when streaming responses fail mid-flight.
#[derive(Debug, Clone)]
pub struct StreamError(pub String);

impl std::fmt::Display for StreamError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl Error for StreamError {}

/// Convert eagerly collected message chunks into a boxed stream suitable for [`ClientWrapper`](crate::client_wrapper::ClientWrapper)
/// implementations.
pub fn chunks_to_stream(
    chunks: Vec<Result<MessageChunk, Box<dyn Error + Send>>>,
) -> crate::client_wrapper::MessageChunkStream {
    let stream = futures_util::stream::iter(
        chunks
            .into_iter()
            .map(|r| r.map_err(|e| e as Box<dyn Error>)),
    );
    Box::pin(stream)
}
