# Streaming Support in CloudLLM

This document explains how to use the streaming feature in CloudLLM to receive LLM responses in real-time as tokens arrive.

## Overview

Streaming support allows you to display LLM responses incrementally as they are generated, rather than waiting for the complete response. This dramatically reduces perceived latency and provides a better user experience.

## Benefits

- **Reduced Perceived Latency**: Users see tokens appear immediately as the LLM generates them
- **Better UX**: The "typing" effect feels more responsive and natural
- **Easy to Use**: Similar API to non-streaming methods
- **Backward Compatible**: Existing code continues to work unchanged

## Basic Usage

### Using LLMSession

```rust
use cloudllm::clients::openai::{Model, OpenAIClient};
use cloudllm::client_wrapper::Role;
use cloudllm::LLMSession;
use futures_util::StreamExt;
use std::sync::Arc;

#[tokio::main]
async fn main() {
    let secret_key = std::env::var("OPEN_AI_SECRET").expect("OPEN_AI_SECRET not set");
    let client = OpenAIClient::new_with_model_enum(&secret_key, Model::GPT41Nano);
    
    let mut session = LLMSession::new(
        Arc::new(client),
        "You are a helpful assistant.".to_string(),
        8192,
    );

    // Send a message with streaming enabled
    match session
        .send_message_stream(Role::User, "Write a haiku about Rust.".to_string(), None)
        .await
    {
        Ok(Some(mut stream)) => {
            // Stream is available - process chunks as they arrive
            let mut full_response = String::new();
            
            while let Some(chunk_result) = stream.next().await {
                match chunk_result {
                    Ok(chunk) => {
                        // Display the incremental content
                        if !chunk.content.is_empty() {
                            print!("{}", chunk.content);
                            full_response.push_str(&chunk.content);
                        }
                        
                        // Check if streaming is complete
                        if let Some(reason) = chunk.finish_reason {
                            println!("\n[Finished: {}]", reason);
                        }
                    }
                    Err(e) => {
                        eprintln!("Error in stream: {}", e);
                        break;
                    }
                }
            }
            
            println!("\nReceived {} chars", full_response.len());
        }
        Ok(None) => {
            // Streaming not supported by this client
            println!("Streaming not available");
        }
        Err(e) => {
            eprintln!("Error: {}", e);
        }
    }
}
```

## MessageChunk Structure

Each chunk in the stream is a `MessageChunk` with:

- `content: String` - The incremental text content (may be empty)
- `finish_reason: Option<String>` - Indicates why streaming ended (e.g., "stop", "length")

## Important Notes

### Token Usage Tracking

⚠️ Token usage tracking is **not available** for streaming responses. If you need accurate token counts, use the non-streaming `send_message()` method instead.

### Conversation History

When using `LLMSession::send_message_stream()`:

1. The user message is automatically added to conversation history
2. The assistant response is **not** automatically added
3. You can manually add the accumulated response to history if needed:

```rust
// After collecting the full streamed response
if !full_response.is_empty() {
    session.send_message(Role::Assistant, full_response, None).await?;
}
```

### Supported Clients

- ✅ **OpenAIClient**: Full streaming support
- ✅ **GrokClient**: Full streaming support (delegates to OpenAI)
- ⏳ **Other clients**: Return `None` (not yet implemented)

You can check if a client supports streaming:

```rust
match client.send_message_stream(&messages, None).await? {
    Some(stream) => { /* streaming available */ }
    None => { /* fall back to non-streaming */ }
}
```

## Error Handling

Streaming can fail at two points:

1. **Initiation**: The request to start streaming fails
   ```rust
   .await? // Handle with standard error handling
   ```

2. **During streaming**: Individual chunks may fail
   ```rust
   match chunk_result {
       Ok(chunk) => { /* process chunk */ }
       Err(e) => { /* handle error */ }
   }
   ```

## Performance Considerations

- Streaming provides **better perceived performance** but may use slightly more bandwidth
- For very short responses, non-streaming might be faster
- For longer responses, streaming provides immediate feedback

## Complete Example

See `examples/streaming_example.rs` for a complete working example that demonstrates:

- Basic streaming usage
- Error handling
- Accumulating the full response
- Managing conversation history

## Migration from Non-Streaming

Existing code using `send_message()` continues to work without changes:

```rust
// Old code - still works!
let response = session.send_message(Role::User, "Hello".to_string(), None).await?;
```

To add streaming:

```rust
// New streaming code
if let Some(mut stream) = session.send_message_stream(Role::User, "Hello".to_string(), None).await? {
    while let Some(chunk_result) = stream.next().await {
        // Process chunks
    }
}
```

## Future Improvements

Planned enhancements:

- Token usage tracking for streaming responses
- Automatic conversation history management for streamed responses
- Streaming support for additional providers (Claude, Gemini, etc.)
