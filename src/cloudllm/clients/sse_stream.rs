//! Live OpenAI-compatible Chat Completions SSE streaming.
//!
//! The historical `create_chat_stream` path collected every chunk into a `Vec` and only
//! then replayed them, so callers still waited for the full generation. This module
//! reads the HTTP body incrementally and yields [`MessageChunk`]s as SSE events arrive.
//!
//! Reasoning traces are pulled from several vendor field names so Grok, DeepSeek,
//! OpenRouter, and Gemini-compat thinking models all surface through the same channel.

use crate::client_wrapper::{
    Message, MessageChunk, MessageChunkStream, NativeToolCall, Role, StreamedToolCallDelta,
    TokenUsage, ToolDefinition,
};
use crate::clients::common::StreamError;
use futures_util::StreamExt;
use std::collections::BTreeMap;
use std::error::Error;
use std::pin::Pin;
use std::task::{Context, Poll};
use std::time::Duration;
use tokio::sync::mpsc;

/// Shared HTTP client **without** a request timeout.
///
/// Reasoning models regularly sit silent for many minutes before the first
/// visible token. The regular [`super::common::get_shared_http_client`] times
/// out at 300s, which would kill those runs.
fn stream_http_client() -> &'static reqwest::Client {
    use std::sync::OnceLock;
    static CLIENT: OnceLock<reqwest::Client> = OnceLock::new();
    CLIENT.get_or_init(|| {
        reqwest::ClientBuilder::new()
            .pool_idle_timeout(Some(Duration::from_secs(90)))
            .pool_max_idle_per_host(10)
            .tcp_keepalive(Some(Duration::from_secs(60)))
            .connect_timeout(Duration::from_secs(30))
            .build()
            .expect("Failed to build streaming HTTP client")
    })
}

/// Optional reasoning-effort override from the environment.
///
/// `CLOUDLLM_REASONING_EFFORT=low|medium|high|none` — `none` leaves the
/// provider default. Lower effort is the main lever for hour-long RALPH runs.
pub fn reasoning_effort_from_env() -> Option<String> {
    std::env::var("CLOUDLLM_REASONING_EFFORT")
        .ok()
        .map(|s| s.trim().to_ascii_lowercase())
        .filter(|s| matches!(s.as_str(), "low" | "medium" | "high" | "minimal" | "none"))
}

/// Heartbeat interval used by [`crate::Agent`] while an LLM call is in flight.
pub fn heartbeat_secs_from_env() -> u64 {
    std::env::var("CLOUDLLM_HEARTBEAT_SECS")
        .ok()
        .and_then(|s| s.parse().ok())
        .filter(|&n| n >= 2)
        .unwrap_or(10)
}

/// Which OpenAI-compatible dialect we are talking to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CompatibleProvider {
    OpenAi,
    OpenRouter,
    Xai,
    Gemini,
    Other,
}

impl CompatibleProvider {
    /// Infer the dialect from a Chat Completions base URL.
    pub fn from_base_url(url: &str) -> Self {
        let u = url.to_ascii_lowercase();
        if u.contains("openrouter.ai") {
            CompatibleProvider::OpenRouter
        } else if u.contains("x.ai") {
            CompatibleProvider::Xai
        } else if u.contains("googleapis.com") || u.contains("generativelanguage") {
            CompatibleProvider::Gemini
        } else if u.contains("openai.com") {
            CompatibleProvider::OpenAi
        } else {
            CompatibleProvider::Other
        }
    }
}

/// Serialise CloudLLM messages to the OpenAI Chat Completions wire format.
pub fn messages_to_openai_wire(messages: &[Message]) -> Vec<serde_json::Value> {
    messages
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
                        .map(|tc| {
                            serde_json::json!({
                                "id": tc.id,
                                "type": "function",
                                "function": {
                                    "name": tc.name,
                                    "arguments": serde_json::to_string(&tc.arguments)
                                        .unwrap_or_else(|_| "{}".to_string())
                                }
                            })
                        })
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
        .collect()
}

/// Serialise native tool definitions to the OpenAI `tools` array.
pub fn tools_to_openai_wire(tools: &[ToolDefinition]) -> Vec<serde_json::Value> {
    tools
        .iter()
        .map(|t| {
            serde_json::json!({
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters_schema
                }
            })
        })
        .collect()
}

/// Attach streaming + optional reasoning knobs to a Chat Completions body.
pub fn apply_stream_body_options(body: &mut serde_json::Value, provider: CompatibleProvider) {
    body["stream"] = serde_json::json!(true);
    // `stream_options` is OpenAI-family. Local OpenAI-compat servers (older
    // vLLM / llama.cpp) 400 on unknown fields; Agent can fall back, but the
    // planner historically could not — so only send it where it is known-good.
    match provider {
        CompatibleProvider::OpenAi
        | CompatibleProvider::OpenRouter
        | CompatibleProvider::Xai
        | CompatibleProvider::Gemini => {
            body["stream_options"] = serde_json::json!({ "include_usage": true });
        }
        CompatibleProvider::Other => {}
    }

    let effort = reasoning_effort_from_env();
    match provider {
        CompatibleProvider::OpenRouter => {
            if let Some(ref e) = effort {
                if e != "none" {
                    body["reasoning"] = serde_json::json!({
                        "effort": e,
                        "exclude": false
                    });
                }
            } else {
                body["include_reasoning"] = serde_json::json!(true);
                body["reasoning"] = serde_json::json!({ "exclude": false });
            }
        }
        CompatibleProvider::Xai | CompatibleProvider::OpenAi => {
            if let Some(ref e) = effort {
                if e != "none" {
                    body["reasoning_effort"] = serde_json::json!(e);
                }
            }
        }
        CompatibleProvider::Gemini | CompatibleProvider::Other => {
            if let Some(ref e) = effort {
                if e != "none" {
                    body["reasoning_effort"] = serde_json::json!(e);
                }
            }
        }
    }
}

/// Attach non-stream reasoning knobs (blocking `send_with_native_tools` path).
pub fn apply_blocking_reasoning_options(body: &mut serde_json::Value, base_url: &str) {
    let provider = CompatibleProvider::from_base_url(base_url);
    if let Some(effort) = reasoning_effort_from_env() {
        if effort != "none" {
            match provider {
                CompatibleProvider::OpenRouter => {
                    body["reasoning"] = serde_json::json!({
                        "effort": effort,
                        "exclude": false
                    });
                }
                _ => {
                    body["reasoning_effort"] = serde_json::json!(effort);
                }
            }
        }
    } else if provider == CompatibleProvider::OpenRouter {
        body["include_reasoning"] = serde_json::json!(true);
        body["reasoning"] = serde_json::json!({ "exclude": false });
    }
}

/// Build a Chat Completions request body (messages + optional tools).
pub fn chat_completions_body(
    model: &str,
    messages: &[Message],
    tools: Option<&[ToolDefinition]>,
) -> serde_json::Value {
    let mut body = serde_json::json!({
        "model": model,
        "messages": messages_to_openai_wire(messages),
    });
    if let Some(defs) = tools.filter(|t| !t.is_empty()) {
        body["tools"] = serde_json::json!(tools_to_openai_wire(defs));
    }
    body
}

/// Open a live Chat Completions SSE stream against `base_url/chat/completions`.
pub fn open_chat_completions_stream(
    base_url: &str,
    api_key: &str,
    model: &str,
    messages: &[Message],
    tools: Option<&[ToolDefinition]>,
    extra_headers: &[(&str, &str)],
) -> crate::client_wrapper::MessageStreamFuture<'static> {
    let url = format!("{}/chat/completions", base_url.trim_end_matches('/'));
    let mut body = chat_completions_body(model, messages, tools);
    apply_stream_body_options(&mut body, CompatibleProvider::from_base_url(base_url));
    let api_key = api_key.to_string();
    let extra: Vec<(String, String)> = extra_headers
        .iter()
        .map(|(k, v)| ((*k).to_string(), (*v).to_string()))
        .collect();

    Box::pin(async move {
        let mut req = stream_http_client()
            .post(&url)
            .header("Authorization", format!("Bearer {}", api_key))
            .header("Content-Type", "application/json")
            .header("Accept", "text/event-stream");
        for (k, v) in &extra {
            req = req.header(k.as_str(), v.as_str());
        }

        let resp = req.json(&body).send().await.map_err(|e| {
            Box::new(StreamError(format!("stream connect error: {}", e)))
                as Box<dyn Error + Send + Sync>
        })?;

        let status = resp.status();
        if !status.is_success() {
            let text = resp.text().await.unwrap_or_default();
            return Err(Box::new(StreamError(format!(
                "stream HTTP {} from {}: {}",
                status, url, text
            ))) as Box<dyn Error + Send + Sync>);
        }

        let (tx, rx) = mpsc::unbounded_channel::<Result<MessageChunk, String>>();
        tokio::spawn(async move {
            let mut byte_stream = resp.bytes_stream();
            let mut parser = SseParser::new();
            while let Some(item) = byte_stream.next().await {
                match item {
                    Ok(bytes) => {
                        for data in parser.push(&bytes) {
                            if !dispatch_sse_data(&tx, &data) {
                                return;
                            }
                        }
                    }
                    Err(e) => {
                        let _ = tx.send(Err(format!("stream body error: {}", e)));
                        return;
                    }
                }
            }
            // Providers sometimes close the socket without a trailing blank line
            // after the last `data:` event. Flush the leftover event so we do
            // not drop the final token / [DONE].
            for data in parser.finish() {
                if !dispatch_sse_data(&tx, &data) {
                    return;
                }
            }
        });

        Ok(Some(Box::pin(MpscChunkStream { rx }) as MessageChunkStream))
    })
}

/// Incremental SSE event splitter.
///
/// Line-oriented (not "scan for `\n\n`") so `\n`, `\r\n`, and mixed
/// `\n\r\n` / `\r\n\n` separators all dispatch correctly. Incomplete UTF-8
/// at a TCP boundary stays in `pending` until a newline arrives.
struct SseParser {
    pending: Vec<u8>,
    data_lines: Vec<String>,
}

impl SseParser {
    fn new() -> Self {
        Self {
            pending: Vec::new(),
            data_lines: Vec::new(),
        }
    }

    fn push(&mut self, bytes: &[u8]) -> Vec<String> {
        self.pending.extend_from_slice(bytes);
        let mut events = Vec::new();
        while let Some(line) = self.take_line() {
            if let Some(data) = self.ingest_line(&line) {
                events.push(data);
            }
        }
        events
    }

    /// Flush a trailing event that was not terminated by a blank line.
    fn finish(&mut self) -> Vec<String> {
        if !self.pending.is_empty() {
            let leftover = std::mem::take(&mut self.pending);
            let line = String::from_utf8_lossy(&leftover);
            let line = line.trim_end_matches(['\r', '\n']);
            if !line.is_empty() {
                let _ = self.ingest_line(line);
            }
        }
        self.take_event_data().into_iter().collect()
    }

    /// Returns `Some(data)` when `line` is a blank event separator.
    fn ingest_line(&mut self, line: &str) -> Option<String> {
        if line.is_empty() {
            return self.take_event_data();
        }
        if let Some(rest) = line.strip_prefix("data:") {
            // SSE spec: strip at most one leading U+0020 after `data:`.
            let rest = rest.strip_prefix(' ').unwrap_or(rest);
            self.data_lines.push(rest.to_string());
        }
        None
    }

    fn take_line(&mut self) -> Option<String> {
        let pos = self.pending.iter().position(|&b| b == b'\n')?;
        let mut line: Vec<u8> = self.pending.drain(..=pos).collect();
        line.pop(); // drop LF
        if line.last() == Some(&b'\r') {
            line.pop();
        }
        Some(String::from_utf8_lossy(&line).into_owned())
    }

    fn take_event_data(&mut self) -> Option<String> {
        if self.data_lines.is_empty() {
            None
        } else {
            Some(self.data_lines.drain(..).collect::<Vec<_>>().join("\n"))
        }
    }
}

/// Heartbeat ticker used while an LLM call is in flight.
///
/// `Skip` (not the default `Burst`) so a slow event handler cannot dump a
/// backlog of "still working" lines when it finally yields.
pub fn heartbeat_interval() -> tokio::time::Interval {
    let mut ticker = tokio::time::interval(Duration::from_secs(heartbeat_secs_from_env()));
    ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
    ticker
}

fn dispatch_sse_data(tx: &mpsc::UnboundedSender<Result<MessageChunk, String>>, data: &str) -> bool {
    if data.trim() == "[DONE]" {
        return false;
    }
    match parse_sse_event(data) {
        Ok(chunks) => {
            for chunk in chunks {
                if tx.send(Ok(chunk)).is_err() {
                    return false;
                }
            }
            true
        }
        Err(e) => {
            let _ = tx.send(Err(e));
            false
        }
    }
}

/// Parse one `data:` JSON payload into a [`MessageChunk`].
///
/// Returns `Ok(None)` for keep-alives / empty objects that carry no signal.
/// When a single SSE event carries **multiple** `tool_calls`, only the first
/// is returned here; use [`parse_sse_event`] to keep all of them.
pub fn parse_sse_data(data: &str) -> Result<Option<MessageChunk>, String> {
    Ok(parse_sse_event(data)?.into_iter().next())
}

/// Parse one `data:` JSON payload into every [`MessageChunk`] it represents.
///
/// OpenAI often puts several `tool_calls` (name + id for each index) in the
/// *first* delta. Emitting one chunk per tool-call fragment keeps
/// [`StreamAccumulator`] from dropping parallel calls.
///
/// Returns `Ok(vec![])` for keep-alives / empty objects.
/// Returns `Err` for malformed JSON **or** a top-level `error` object
/// (OpenAI/OpenRouter stream errors arrive as HTTP 200 + `data: {"error":...}`).
pub fn parse_sse_event(data: &str) -> Result<Vec<MessageChunk>, String> {
    let trimmed = data.trim();
    if trimmed.is_empty() || trimmed == "[DONE]" {
        return Ok(Vec::new());
    }
    let value: serde_json::Value =
        serde_json::from_str(trimmed).map_err(|e| format!("SSE JSON: {} ({})", e, trimmed))?;
    if let Some(err) = value.get("error") {
        return Err(format_sse_error(err));
    }
    Ok(chunks_from_completion_json(&value))
}

/// Extract [`MessageChunk`]s from a Chat Completions JSON object (stream or not).
pub fn chunk_from_completion_json(value: &serde_json::Value) -> Option<MessageChunk> {
    chunks_from_completion_json(value).into_iter().next()
}

fn chunks_from_completion_json(value: &serde_json::Value) -> Vec<MessageChunk> {
    let usage = value.get("usage").and_then(parse_usage);

    let choice = value.get("choices").and_then(|c| c.get(0));
    let delta = choice
        .and_then(|c| c.get("delta"))
        .or_else(|| choice.and_then(|c| c.get("message")));

    let mut content = String::new();
    let mut reasoning = String::new();
    let mut tool_deltas = Vec::new();
    let finish_reason = choice
        .and_then(|c| c.get("finish_reason"))
        .and_then(|v| v.as_str())
        .map(|s| s.to_string());

    if let Some(d) = delta {
        if let Some(s) = json_str(d.get("content")) {
            content.push_str(s);
        }
        for key in &[
            "reasoning",
            "reasoning_content",
            "thinking",
            "reasoning_text",
            "thought",
        ] {
            if let Some(s) = json_str(d.get(*key)) {
                reasoning.push_str(s);
            }
        }
        if let Some(arr) = d.get("reasoning_details").and_then(|a| a.as_array()) {
            for item in arr {
                if let Some(s) = json_str(item.get("text"))
                    .or_else(|| json_str(item.get("summary")))
                    .or_else(|| json_str(item.get("content")))
                {
                    reasoning.push_str(s);
                }
            }
        }
        if let Some(arr) = d.get("tool_calls").and_then(|a| a.as_array()) {
            tool_deltas.extend(arr.iter().map(tool_call_delta_from_json));
        }
    }

    if content.is_empty()
        && reasoning.is_empty()
        && tool_deltas.is_empty()
        && finish_reason.is_none()
        && usage.is_none()
    {
        return Vec::new();
    }

    if tool_deltas.is_empty() {
        return vec![MessageChunk {
            content,
            reasoning,
            finish_reason,
            tool_call_delta: None,
            usage,
        }];
    }

    // Attach visible text / usage to the first tool-call chunk so a consumer
    // that only looks at one MessageChunk per SSE event still sees content.
    let mut chunks = Vec::with_capacity(tool_deltas.len());
    for (i, tc) in tool_deltas.into_iter().enumerate() {
        if i == 0 {
            chunks.push(MessageChunk {
                content: content.clone(),
                reasoning: reasoning.clone(),
                finish_reason: finish_reason.clone(),
                tool_call_delta: Some(tc),
                usage: usage.clone(),
            });
        } else {
            chunks.push(MessageChunk {
                tool_call_delta: Some(tc),
                ..MessageChunk::default()
            });
        }
    }
    chunks
}

fn tool_call_delta_from_json(tc: &serde_json::Value) -> StreamedToolCallDelta {
    let index = tc.get("index").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
    let id = tc.get("id").and_then(|v| v.as_str()).map(|s| s.to_string());
    let func = tc.get("function");
    let name = func
        .and_then(|f| f.get("name"))
        .and_then(|v| v.as_str())
        .map(|s| s.to_string());
    let arguments = json_args_fragment(func.and_then(|f| f.get("arguments")));
    StreamedToolCallDelta {
        index,
        id,
        name,
        arguments,
    }
}

/// Serialise a tool-call `arguments` field that may be a JSON string *or* a
/// parsed object (some Gemini/OpenRouter compat paths send the latter).
fn json_args_fragment(v: Option<&serde_json::Value>) -> String {
    match v {
        Some(serde_json::Value::String(s)) => s.clone(),
        Some(other) if !other.is_null() => other.to_string(),
        _ => String::new(),
    }
}

fn format_sse_error(err: &serde_json::Value) -> String {
    if let Some(msg) = err.get("message").and_then(|m| m.as_str()) {
        let typ = err.get("type").and_then(|t| t.as_str()).unwrap_or("error");
        format!("SSE error ({}): {}", typ, msg)
    } else if let Some(s) = err.as_str() {
        format!("SSE error: {}", s)
    } else {
        format!("SSE error: {}", err)
    }
}

fn json_str(v: Option<&serde_json::Value>) -> Option<&str> {
    v.and_then(|x| x.as_str()).filter(|s| !s.is_empty())
}

fn parse_usage(v: &serde_json::Value) -> Option<TokenUsage> {
    let input = v
        .get("prompt_tokens")
        .or_else(|| v.get("input_tokens"))
        .and_then(|x| x.as_u64())
        .unwrap_or(0) as usize;
    let output = v
        .get("completion_tokens")
        .or_else(|| v.get("output_tokens"))
        .and_then(|x| x.as_u64())
        .unwrap_or(0) as usize;
    if input == 0 && output == 0 {
        if let Some(total) = v.get("total_tokens").and_then(|x| x.as_u64()) {
            return Some(TokenUsage {
                input_tokens: 0,
                output_tokens: 0,
                total_tokens: total as usize,
            });
        }
        return None;
    }
    Some(TokenUsage {
        input_tokens: input,
        output_tokens: output,
        total_tokens: input + output,
    })
}

/// Accumulates streamed chunks into a single assistant [`Message`].
#[derive(Default)]
pub struct StreamAccumulator {
    content: String,
    reasoning: String,
    tool_calls: BTreeMap<usize, (Option<String>, Option<String>, String)>,
    usage: Option<TokenUsage>,
}

impl StreamAccumulator {
    /// Create an empty accumulator.
    pub fn new() -> Self {
        Self::default()
    }

    /// Fold one streamed chunk into the running reply.
    pub fn apply(&mut self, chunk: &MessageChunk) {
        self.content.push_str(&chunk.content);
        self.reasoning.push_str(&chunk.reasoning);
        if let Some(tc) = &chunk.tool_call_delta {
            let entry = self
                .tool_calls
                .entry(tc.index)
                .or_insert((None, None, String::new()));
            if let Some(id) = &tc.id {
                entry.0 = Some(id.clone());
            }
            if let Some(name) = &tc.name {
                entry.1 = Some(name.clone());
            }
            entry.2.push_str(&tc.arguments);
        }
        if chunk.usage.is_some() {
            self.usage = chunk.usage.clone();
        }
    }

    /// Character length of the assembled reasoning trace.
    pub fn reasoning_len(&self) -> usize {
        self.reasoning.len()
    }

    /// Token usage reported by the provider, if any.
    pub fn usage(&self) -> Option<TokenUsage> {
        self.usage.clone()
    }

    /// Consume the accumulator and produce the assistant message.
    pub fn into_message(self) -> Message {
        let tool_calls = self
            .tool_calls
            .into_iter()
            .filter_map(|(index, (id, name, args))| {
                let name = name?;
                let arguments = serde_json::from_str(&args)
                    .unwrap_or_else(|_| serde_json::Value::Object(serde_json::Map::new()));
                Some(NativeToolCall {
                    id: id.unwrap_or_else(|| format!("call_{}", index)),
                    name,
                    arguments,
                })
            })
            .collect();
        Message {
            role: Role::Assistant,
            content: std::sync::Arc::from(self.content),
            tool_calls,
        }
    }
}

struct MpscChunkStream {
    rx: mpsc::UnboundedReceiver<Result<MessageChunk, String>>,
}

impl futures_util::Stream for MpscChunkStream {
    type Item = Result<MessageChunk, Box<dyn Error + Send + Sync>>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        match Pin::new(&mut self.rx).poll_recv(cx) {
            Poll::Ready(Some(Ok(c))) => Poll::Ready(Some(Ok(c))),
            Poll::Ready(Some(Err(e))) => Poll::Ready(Some(Err(
                Box::new(StreamError(e)) as Box<dyn Error + Send + Sync>
            ))),
            Poll::Ready(None) => Poll::Ready(None),
            Poll::Pending => Poll::Pending,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_openai_content_delta() {
        let raw = r#"{"choices":[{"delta":{"content":"Hello"},"finish_reason":null}]}"#;
        let chunk = parse_sse_data(raw).unwrap().unwrap();
        assert_eq!(chunk.content, "Hello");
        assert!(chunk.reasoning.is_empty());
    }

    #[test]
    fn parses_deepseek_reasoning_content() {
        let raw = r#"{"choices":[{"delta":{"reasoning_content":"I should"}}]}"#;
        let chunk = parse_sse_data(raw).unwrap().unwrap();
        assert_eq!(chunk.reasoning, "I should");
        assert!(chunk.content.is_empty());
    }

    #[test]
    fn parses_openrouter_reasoning_and_details() {
        let raw = r#"{
            "choices":[{
                "delta":{
                    "reasoning":"plan: ",
                    "reasoning_details":[{"text":"write maze"}]
                }
            }]
        }"#;
        let chunk = parse_sse_data(raw).unwrap().unwrap();
        assert_eq!(chunk.reasoning, "plan: write maze");
    }

    #[test]
    fn parses_tool_call_fragment() {
        let raw = r#"{
            "choices":[{
                "delta":{
                    "tool_calls":[{
                        "index":0,
                        "id":"call_1",
                        "function":{"name":"write_game_file","arguments":"{\"html\":"}
                    }]
                }
            }]
        }"#;
        let chunk = parse_sse_data(raw).unwrap().unwrap();
        let tc = chunk.tool_call_delta.unwrap();
        assert_eq!(tc.name.as_deref(), Some("write_game_file"));
        assert_eq!(tc.arguments, "{\"html\":");
    }

    #[test]
    fn parses_usage() {
        let raw = r#"{"choices":[],"usage":{"prompt_tokens":10,"completion_tokens":20}}"#;
        let chunk = parse_sse_data(raw).unwrap().unwrap();
        let u = chunk.usage.unwrap();
        assert_eq!(u.input_tokens, 10);
        assert_eq!(u.output_tokens, 20);
        assert_eq!(u.total_tokens, 30);
    }

    #[test]
    fn accumulator_joins_tool_args() {
        let mut acc = StreamAccumulator::new();
        acc.apply(&MessageChunk {
            tool_call_delta: Some(StreamedToolCallDelta {
                index: 0,
                id: Some("c1".into()),
                name: Some("mem".into()),
                arguments: "{\"k\":".into(),
            }),
            ..MessageChunk::default()
        });
        acc.apply(&MessageChunk {
            tool_call_delta: Some(StreamedToolCallDelta {
                index: 0,
                arguments: "\"v\"}".into(),
                ..StreamedToolCallDelta::default()
            }),
            ..MessageChunk::default()
        });
        let msg = acc.into_message();
        assert_eq!(msg.tool_calls[0].name, "mem");
        assert_eq!(msg.tool_calls[0].arguments["k"], "v");
    }

    #[test]
    fn sse_parser_splits_events() {
        let mut p = SseParser::new();
        let bytes = b"data: {\"choices\":[{\"delta\":{\"content\":\"A\"}}]}\n\ndata: [DONE]\n\n";
        let events = p.push(bytes);
        assert_eq!(events.len(), 2);
        assert_eq!(events[1].trim(), "[DONE]");
    }

    #[test]
    fn sse_parser_handles_crlf_and_mixed_separators() {
        let mut p = SseParser::new();
        let bytes = b"data: {\"a\":1}\r\n\r\ndata: {\"b\":2}\n\r\ndata: {\"c\":3}\r\n\n";
        let events = p.push(bytes);
        assert_eq!(events.len(), 3);
        assert_eq!(events[0], "{\"a\":1}");
        assert_eq!(events[1], "{\"b\":2}");
        assert_eq!(events[2], "{\"c\":3}");
    }

    #[test]
    fn sse_parser_flushes_trailing_event_without_blank_line() {
        let mut p = SseParser::new();
        assert!(p
            .push(b"data: {\"choices\":[{\"delta\":{\"content\":\"Z\"}}]}")
            .is_empty());
        let events = p.finish();
        assert_eq!(events.len(), 1);
        assert!(events[0].contains("\"Z\""));
    }

    #[test]
    fn sse_parser_ignores_comment_keepalives() {
        let mut p = SseParser::new();
        let events = p.push(b": OPENROUTER PROCESSING\n\ndata: {\"choices\":[]}\n\n");
        // empty choices + no usage → parse later as no chunk; the event itself is present
        // if it had a data: line. `{"choices":[]}` is a data payload.
        assert_eq!(events.len(), 1);
        assert_eq!(events[0], "{\"choices\":[]}");
    }

    #[test]
    fn sse_parser_joins_multiline_data() {
        let mut p = SseParser::new();
        let events = p.push(b"data: {\"foo\":\ndata: 1}\n\n");
        assert_eq!(events, vec!["{\"foo\":\n1}"]);
    }

    #[test]
    fn parse_sse_event_emits_every_tool_call_in_one_delta() {
        let raw = r#"{
            "choices":[{
                "delta":{
                    "tool_calls":[
                        {"index":0,"id":"call_a","function":{"name":"alpha","arguments":""}},
                        {"index":1,"id":"call_b","function":{"name":"beta","arguments":"{"}}
                    ]
                }
            }]
        }"#;
        let chunks = parse_sse_event(raw).unwrap();
        assert_eq!(chunks.len(), 2);
        assert_eq!(
            chunks[0].tool_call_delta.as_ref().unwrap().name.as_deref(),
            Some("alpha")
        );
        assert_eq!(
            chunks[1].tool_call_delta.as_ref().unwrap().name.as_deref(),
            Some("beta")
        );
        assert_eq!(chunks[1].tool_call_delta.as_ref().unwrap().arguments, "{");
    }

    #[test]
    fn accumulator_keeps_parallel_tool_calls_from_one_event() {
        let chunks = parse_sse_event(
            r#"{"choices":[{"delta":{"tool_calls":[
                {"index":0,"id":"c0","function":{"name":"a","arguments":"{}"}},
                {"index":1,"id":"c1","function":{"name":"b","arguments":"{\"x\":1}"}}
            ]}}]}"#,
        )
        .unwrap();
        let mut acc = StreamAccumulator::new();
        for c in &chunks {
            acc.apply(c);
        }
        let msg = acc.into_message();
        assert_eq!(msg.tool_calls.len(), 2);
        assert_eq!(msg.tool_calls[0].name, "a");
        assert_eq!(msg.tool_calls[1].name, "b");
        assert_eq!(msg.tool_calls[1].arguments["x"], 1);
    }

    #[test]
    fn parse_sse_rejects_error_payload() {
        let raw = r#"{"error":{"message":"The server had an error","type":"server_error"}}"#;
        let err = parse_sse_event(raw).unwrap_err();
        assert!(err.contains("server_error"), "{}", err);
        assert!(err.contains("The server had an error"), "{}", err);
    }

    #[test]
    fn tool_args_object_is_not_dropped() {
        let raw = r#"{"choices":[{"delta":{"tool_calls":[{
            "index":0,"id":"c1","function":{"name":"mem","arguments":{"k":"v"}}
        }]}}]}"#;
        let chunk = parse_sse_data(raw).unwrap().unwrap();
        let tc = chunk.tool_call_delta.unwrap();
        assert!(tc.arguments.contains("\"k\""));
        let mut acc = StreamAccumulator::new();
        acc.apply(&MessageChunk {
            tool_call_delta: Some(tc),
            ..MessageChunk::default()
        });
        let msg = acc.into_message();
        assert_eq!(msg.tool_calls[0].arguments["k"], "v");
    }

    #[test]
    fn missing_tool_id_gets_stable_fallback() {
        let mut acc = StreamAccumulator::new();
        acc.apply(&MessageChunk {
            tool_call_delta: Some(StreamedToolCallDelta {
                index: 3,
                name: Some("x".into()),
                arguments: "{}".into(),
                ..StreamedToolCallDelta::default()
            }),
            ..MessageChunk::default()
        });
        let msg = acc.into_message();
        assert_eq!(msg.tool_calls[0].id, "call_3");
    }

    #[test]
    fn provider_from_url() {
        assert_eq!(
            CompatibleProvider::from_base_url("https://openrouter.ai/api/v1"),
            CompatibleProvider::OpenRouter
        );
        assert_eq!(
            CompatibleProvider::from_base_url("https://api.x.ai/v1"),
            CompatibleProvider::Xai
        );
    }

    #[test]
    fn stream_options_omitted_for_unknown_providers() {
        let mut body = serde_json::json!({"model": "local"});
        apply_stream_body_options(&mut body, CompatibleProvider::Other);
        assert!(body.get("stream_options").is_none());
        assert_eq!(body["stream"], true);

        let mut openai = serde_json::json!({"model": "gpt"});
        apply_stream_body_options(&mut openai, CompatibleProvider::OpenAi);
        assert!(openai.get("stream_options").is_some());
    }
}
