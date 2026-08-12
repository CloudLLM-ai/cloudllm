//! Isolated tests for Chat Completions transport-error classification.

use cloudllm::clients::common::is_transient_llm_error;

#[test]
fn classifies_grok_transport_blip() {
    assert!(is_transient_llm_error(
        "error sending request for url (https://api.x.ai/v1/chat/completions)"
    ));
    assert!(is_transient_llm_error(
        "send_with_native_tools: HTTP 429 — rate"
    ));
    assert!(is_transient_llm_error("connection reset by peer"));
    assert!(!is_transient_llm_error(
        "send_with_native_tools: HTTP 400 — unknown field"
    ));
}
