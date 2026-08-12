//! Isolated tests for live OpenAI-compatible SSE parsing.

use cloudllm::client_wrapper::{MessageChunk, StreamedToolCallDelta};
use cloudllm::clients::sse_stream::{
    apply_stream_body_options, parse_sse_data, parse_sse_event, CompatibleProvider, SseParser,
    StreamAccumulator,
};

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
