# Interactive Streaming Session Example

This example demonstrates how to use CloudLLM's streaming support in an interactive chat session. It shows real-time token-by-token responses as the LLM generates them.

## Features

- **Real-time streaming**: See the assistant's response appear token by token as it's generated
- **Interactive chat**: Multi-turn conversations with conversation history management
- **Automatic fallback**: If streaming isn't supported, falls back to standard response mode
- **Token usage tracking**: Displays token usage after each response
- **Multi-line input**: Use `\end` on a separate line to submit your prompt

## Running the Example

### With OpenAI

```bash
OPEN_AI_SECRET=your-api-key cargo run --example interactive_streaming_session
```

Then uncomment the OpenAI client section in the code (lines 20-26) and comment out the other client sections.

### With Grok (xAI)

```bash
XAI_API_KEY=your-api-key cargo run --example interactive_streaming_session
```

The Grok client is enabled by default in the example.

### With Claude

```bash
CLAUDE_API_KEY=your-api-key cargo run --example interactive_streaming_session
```

Then uncomment the Claude client section in the code (lines 51-56) and comment out the other client sections.

### With Gemini

```bash
GEMINI_API_KEY=your-api-key cargo run --example interactive_streaming_session
```

Then uncomment the Gemini client section in the code (lines 28-35) and comment out the other client sections.

## How It Works

1. **Setup**: Creates an LLMSession with your chosen client and system prompt
2. **User Input**: Accepts multi-line input (type `\end` on a new line to submit)
3. **Streaming Response**: Sends the message with `send_message_stream()`
4. **Real-time Display**: Displays each token as it arrives from the LLM
5. **History Management**: Accumulates the full response and adds it to conversation history
6. **Token Tracking**: Shows token usage after each exchange

## Key Differences from Standard Session

### Standard (non-streaming) session:
```rust
let response = session
    .send_message(Role::User, user_input, None)
    .await?;
println!("Assistant: {}", response.content);
```

### Streaming session:
```rust
let stream_result = session
    .send_message_stream(Role::User, user_input, None)
    .await?;

if let Some(mut stream) = stream_result {
    let mut full_response = String::new();
    
    while let Some(chunk_result) = stream.next().await {
        let chunk = chunk_result?;
        print!("{}", chunk.content);  // Display immediately
        full_response.push_str(&chunk.content);
    }
    
    // Add accumulated response to history
    session.send_message(Role::Assistant, full_response, None).await?;
}
```

## Benefits

- **Reduced Perceived Latency**: Users see output immediately instead of waiting for the complete response
- **Better UX**: The "typing" effect feels more responsive and natural
- **Interactive Feel**: Makes the conversation feel more dynamic and engaging
- **Progress Feedback**: Users know the LLM is working and can start reading early

## Conversation History

The example properly maintains conversation history by:

1. User messages are automatically added by `send_message_stream()`
2. The assistant's streamed response is accumulated
3. The full response is added to history using `send_message(Role::Assistant, ...)`

This ensures subsequent messages have full context from previous exchanges.

## Token Usage Note

Token usage tracking is displayed after each response, but note that for streaming responses, the usage information may be less detailed than non-streaming responses. The session still tracks total token usage across the conversation.

## Example Session

```
=== CloudLLM Interactive Streaming Session ===

This example demonstrates real-time streaming responses.
You'll see the assistant's response appear token by token as it's generated.

Using model: grok-4-fast-reasoning
Max tokens: 1024


You [type '\end' in a separate line to submit prompt]:
Explain what makes Rust a great programming language in one sentence.
\end

Assistant (streaming): Rust is a great programming language because it combines memory safety with zero-cost abstractions, enabling developers to write fast and reliable systems without the pitfalls of manual memory management.

[Stream finished: stop | Chunks received: 42]
Token Usage: <input tokens: 145, output tokens: 42, total tokens: 187, max tokens: 1024>


You [type '\end' in a separate line to submit prompt]:
```

## Troubleshooting

### "Streaming not supported by this client"

If you see this message, it means the client you're using doesn't yet have streaming implemented. The example will automatically fall back to standard (non-streaming) mode.

### Token Usage Not Accurate

For streaming responses, detailed token usage may not be available immediately. The session tracks cumulative usage across the conversation.

### Connection Issues

Make sure your API key is valid and you have a stable internet connection. Streaming requires maintaining an open connection to the LLM provider.

## Code Structure

The example demonstrates:

- Setting up streaming-capable clients (OpenAI, Grok)
- Using `send_message_stream()` for streaming responses
- Processing stream chunks with `StreamExt::next()`
- Accumulating streamed content
- Managing conversation history with streaming
- Graceful fallback to non-streaming mode
- Error handling for stream errors

This example provides a complete template for building interactive streaming applications with CloudLLM.
