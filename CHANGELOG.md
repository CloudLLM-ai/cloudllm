# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.11.1] — 2026-02-20

### Summary

Native function-calling support for all four major providers (OpenAI, Anthropic Claude, xAI
Grok, Google Gemini). Tool calls are now routed through the provider's structured API rather
than relying solely on brace-counted JSON text parsing, eliminating the multibyte-Unicode
panic vector and enabling parallel tool-call support in future releases.

### Added

- **`NativeToolCall` struct** (`client_wrapper`) — Carries the provider-assigned call ID, tool
  name, and parsed JSON arguments returned by a native function-calling response.
- **`ToolDefinition` struct** (`client_wrapper`) — JSON Schema descriptor for a tool (name,
  description, `parameters_schema`). Passed to `send_message` as `tools:
  Option<Vec<ToolDefinition>>`.
- **`Role::Tool { call_id }` variant** — New conversation role for tool-result messages.
  Serialises as `{"role":"tool","tool_call_id":"<id>","content":"..."}` in the OpenAI wire
  format, allowing providers to correlate results with the originating call.
- **`Message::tool_calls: Vec<NativeToolCall>`** field — Populated when the provider returns
  native function-calling results; empty `vec![]` otherwise (no breaking change for code that
  only reads `content`).
- **`send_with_native_tools()`** (`clients/common`) — reqwest-based helper that serialises a
  `tools` array and parses `NativeToolCall` objects from the response. Compatible with all
  OpenAI-compatible endpoints (OpenAI `/v1/chat/completions`, Anthropic
  `api.anthropic.com/v1/chat/completions`, xAI `/v1/chat/completions`, Gemini
  `/v1beta/chat/completions`).
- **`ToolParameter::to_json_schema()`** — Converts a parameter definition to a JSON Schema
  snippet for all six `ToolParameterType` variants (String, Number, Integer, Boolean, Array,
  Object), including recursive nested object/array support.
- **`ToolMetadata::to_tool_definition()`** — Builds a `ToolDefinition` from a `ToolMetadata`
  instance, collecting required/optional parameters into the standard JSON Schema `"required"`
  array.
- **`ToolRegistry::to_tool_definitions()`** — Returns all registered tools as a
  `Vec<ToolDefinition>` ready to pass to `send_message`. This is a **synchronous** method.
- **`NativeToolCall` and `ToolDefinition` re-exported** from the crate root (`lib.rs`).
- **New unit tests** in `tool_protocol.rs` for `to_json_schema()`, `to_tool_definition()`, and
  `to_tool_definitions()`.

### Changed

- **`ClientWrapper::send_message` signature** — `(messages, grok_tools, openai_tools)` →
  `(messages, tools: Option<Vec<ToolDefinition>>)`. All four provider implementations
  (`OpenAIClient`, `GrokClient`, `ClaudeClient`, `GeminiClient`) updated.
  - When `tools` is `Some` and non-empty: routes to `send_with_native_tools()`.
  - When `tools` is `None` or empty: falls through to the existing Chat Completions path.
- **`ClientWrapper::send_message_stream` signature** — Same parameter change. Streaming with
  native tools is out of scope; implementations return `Ok(None)`.
- **`LLMSession::send_message` signature** — `(role, content, grok, openai)` → `(role, content,
  tools: Option<Vec<ToolDefinition>>)`. Existing call sites pass `None` for plain turns.
- **`LLMSession::send_message_stream` signature** — Same update.
- **`Agent::send()`** — Now calls `registry.to_tool_definitions()` and passes the result to
  every `send_message` call. Primary tool detection uses `response.tool_calls`; text-parsing
  (`{"tool_call": {...}}`) remains as a fallback.
- **`Agent::generate_with_tokens()`** — Same native tool-calling wiring as `send()`.
- **`ToolCall` internal struct** — Gains `native_id: Option<String>` to distinguish native
  calls (result injected as `Role::Tool { call_id }`) from text-parsed calls (result injected
  as `Role::User`).
- **Breakout game examples** (`breakout_game_ralph.rs`, `breakout_game_agent_teams.rs`) — All
  agents upgraded from `ClaudeHaiku45` to `ClaudeSonnet46`.
- `context_strategy.rs` and `planner.rs` callers updated to the new 3-argument
  `send_message` signature.

### Deprecated

- `Agent::with_grok_tools()` / `with_openai_tools()` builder methods — Fields are retained for
  the xAI / OpenAI Responses API use-case (web_search, x_search, etc.) but are no longer
  forwarded through `send_message`. Providers that need the Responses API path should call
  `send_and_track_responses()` / `send_and_track_openai_responses()` directly.

### Fixed

- All clippy warnings resolved (`-D warnings` clean):
  - Redundant field name `tool_name: tool_name` → `tool_name`
  - Redundant closure `.map(|s| Arc::from(s))` → `.map(Arc::from)`
  - `io::Error::new(ErrorKind::Other, …)` → `io::Error::other(…)`
  - Manual `split_once` pattern in `tool_protocols.rs`
- All doc tests pass (176 passed, 0 failed, 68 ignored).
  - `clients/common.rs` module doc — updated `ClientWrapper` example to new 3-param signature.
  - `image_generation.rs` — fixed `cloudllm::image_generation` import paths (not re-exported
    from crate root; correct path is `cloudllm::cloudllm::image_generation`).
  - `planner.rs` — fixed `#[async_trait(?Send)]` annotation and `StreamSink` import in doc
    examples.
  - `orchestration.rs` — fixed `async` block structure in `AnthropicAgentTeams` doc example.

---

## [0.11.0] — 2026-02-14

### Summary

MCP event tracing, BashTool security hardening, and several critical/high vulnerability fixes.

### Added

- `McpEvent` enum and `on_mcp_event()` callback in `EventHandler`.
- MCP server event tracing and `list_tools` deduplication.
- `McpClientProtocol` event tracing and corrected endpoint paths.

### Fixed

- `parse_tool_call` panic on multibyte Unicode characters (brace-counter now uses
  `char_indices` on the UTF-8 slice, not byte offsets).
- BashTool: enforce timeout and `cwd_restriction`; 62 comprehensive tests added.
- BashTool: fix absolute-path bypass; enforce output size limit.
- 3 Critical and 4 High security vulnerabilities (see security advisory).

---

## [0.10.x and earlier]

See git log for changes prior to v0.11.0.
