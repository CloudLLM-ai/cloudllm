# Streaming Support in CloudLLM

CloudLLM now provides first-class streaming support for real-time token delivery, dramatically reducing perceived latency in user interfaces.

## Overview

Streaming allows you to receive and display tokens as soon as they arrive from the LLM provider, rather than waiting for the complete response. This creates a better user experience as users can start reading the response immediately.

## Features

- **Low Latency**: Tokens are delivered as soon as they arrive from the provider
- **Compatible API**: Streaming methods work alongside existing non-streaming methods
- **Provider Support**: Works with OpenAI, Grok, Claude, and other providers that support streaming
- **Simple Interface**: Easy-to-use Stream interface with `MessageChunk` items

## Usage

### Basic Streaming with ClientWrapper

```rust
use cloudllm::client_wrapper::Role;
use cloudllm::clients::openai::OpenAIClient;
use cloudllm::ClientWrapper;
use futures_util::StreamExt;

#[tokio::main]
async fn main() {
    let client = OpenAIClient::new_with_model_enum(
        &secret_key,
        cloudllm::clients::openai::Model::GPT5Nano,
    );

    let messages = vec![
        cloudllm::client_wrapper::Message {
            role: Role::User,
            content: "Tell me a short story".to_string(),
        },
    ];

    // Get streaming response
    let mut stream = client.send_message_stream(messages, None).await.unwrap();
    
    // Process chunks as they arrive
    while let Some(chunk_result) = stream.next().await {
        match chunk_result {
            Ok(chunk) => {
                print!("{}", chunk.content);  // Display immediately!
                
                if chunk.is_final {
                    break;
                }
            }
            Err(e) => {
                eprintln!("Error: {}", e);
                break;
            }
        }
    }
}
```

### Streaming with LLMSession

```rust
use cloudllm::LLMSession;
use cloudllm::client_wrapper::Role;
use futures_util::StreamExt;

let mut session = LLMSession::new(
    std::sync::Arc::new(client),
    "You are a helpful assistant.".to_string(),
    4096
);

let mut stream = session.send_message_stream(
    Role::User,
    "Write a poem".to_string(),
    None,
).await.unwrap();

let mut full_response = String::new();
while let Some(chunk_result) = stream.next().await {
    let chunk = chunk_result.unwrap();
    print!("{}", chunk.content);
    full_response.push_str(&chunk.content);
}
```

## API Reference

### `MessageChunk`

Represents a chunk of streaming response:

```rust
pub struct MessageChunk {
    /// The incremental content in this chunk
    pub content: String,
    /// Whether this is the final chunk in the stream
    pub is_final: bool,
}
```

### `send_message_stream()`

Available on all `ClientWrapper` implementations:

```rust
async fn send_message_stream(
    &self,
    messages: Vec<Message>,
    optional_search_parameters: Option<SearchParameters>,
) -> Result<Pin<Box<dyn Stream<Item = Result<MessageChunk, SendError>>>>, Box<dyn Error>>
```

Also available on `LLMSession`:

```rust
pub async fn send_message_stream(
    &mut self,
    role: Role,
    content: String,
    optional_search_parameters: Option<SearchParameters>,
) -> Result<Pin<Box<dyn Stream<Item = Result<MessageChunk, SendError>>>>, Box<dyn Error>>
```

## Important Notes

### Token Usage Tracking

**Token usage tracking is NOT available for streaming responses.** The OpenAI streaming API does not provide usage information in real-time. If you need token usage tracking, use the non-streaming `send_message()` method instead.

### Conversation History with LLMSession

When using `send_message_stream()` with `LLMSession`, the assistant's response is **NOT** automatically added to the conversation history. If you want to maintain conversation context with streaming, you must:

1. Collect all chunks into a complete message
2. Manually add the message to the session's conversation history

Example:

```rust
let mut stream = session.send_message_stream(Role::User, prompt, None).await?;
let mut full_response = String::new();

while let Some(chunk) = stream.next().await {
    let chunk = chunk?;
    full_response.push_str(&chunk.content);
}

// Manually add to history if needed
// (Note: You'll need to access the internal conversation_history field,
//  or use send_message() for automatic history management)
```

For most use cases with conversation history, consider using the non-streaming `send_message()` method which handles history automatically.

### Thread Safety

The streaming API returns streams that are not `Send` due to limitations in the underlying HTTP client. This means:

- The stream must be consumed in the same task that creates it
- You cannot send the stream across threads
- This is typically not an issue for web servers or CLI applications

## Examples

See the examples directory for complete working examples:

- `examples/streaming_example.rs` - Basic streaming with ClientWrapper
- `examples/streaming_session_example.rs` - Streaming with LLMSession

Run examples:
```bash
OPEN_AI_SECRET=your-key cargo run --example streaming_example
OPEN_AI_SECRET=your-key cargo run --example streaming_session_example
```

## Supported Providers

All providers that delegate to OpenAI-compatible APIs support streaming:

- ✅ OpenAI (GPT-4, GPT-5, etc.)
- ✅ Grok (xAI)
- ✅ Claude (Anthropic) - via OpenAI-compatible endpoint
- ✅ Gemini - via OpenAI-compatible endpoint

## Benefits

### Dramatically Reduced Perceived Latency

Instead of waiting 5-10 seconds for a complete response, users see the assistant "typing" almost immediately - typically within 200-500ms. This creates a much more responsive and natural user experience.

### Better User Engagement

Users can start reading and processing the response while it's still being generated, leading to:
- Higher user satisfaction
- More natural conversation flow
- Better perceived performance

## Performance Comparison

**Without Streaming:**
```
User sends message -> [5-10 second wait] -> Complete response appears
Time to first token: 5000ms
```

**With Streaming:**
```
User sends message -> [200-500ms] -> First tokens appear -> More tokens arrive continuously
Time to first token: 300ms ⚡
```

The total time to receive the complete message is similar, but the user experience is dramatically better with streaming.
