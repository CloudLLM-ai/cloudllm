use std::env;
use std::io::{self, Write};

use cloudllm::client_wrapper::Role;
use cloudllm::clients::grok::GrokClient;
use cloudllm::LLMSession;
use futures_util::StreamExt;

// Run from the root folder of the repo as follows:
// OPEN_AI_SECRET=your-open-ai-key-here cargo run --example interactive_streaming_session
// XAI_API_KEY=your-xai-key-here cargo run --example interactive_streaming_session
// CLAUDE_API_KEY=your-claude-key-here cargo run --example interactive_streaming_session

#[tokio::main]
async fn main() {
    println!("=== CloudLLM Interactive Streaming Session ===\n");
    println!("This example demonstrates real-time streaming responses.");
    println!("You'll see the assistant's response appear token by token as it's generated.\n");

    // // Read OPENAI_AI_SECRET from environment variable
    // let secret_key =
    //      env::var("OPEN_AI_SECRET").expect("Please set the OPEN_AI_SECRET environment variable!");
    // // Instantiate the OpenAI client
    // let client = cloudllm::clients::openai::OpenAIClient::new_with_model_enum(
    //     &secret_key,
    //     cloudllm::clients::openai::Model::GPT41Nano,
    // );

    // // Read GEMINI_API_KEY from the environment variable
    // let secret_key =
    //    env::var("GEMINI_API_KEY").expect("Please set the GEMINI_API_KEY environment variable!");
    // // Instantiate the Gemini client
    // let client = cloudllm::clients::gemini::GeminiClient::new_with_model_enum(
    //     &secret_key,
    //     cloudllm::clients::gemini::Model::Gemini25Flash,
    // );

    // Read the XAI_API_KEY from the environment variable
    let secret_key =
        env::var("XAI_API_KEY").expect("Please set the XAI_API_KEY environment variable!");
    // Instantiate the Grok client
    let client = GrokClient::new_with_model_enum(
        &secret_key,
        cloudllm::clients::grok::Model::Grok4FastReasoning,
    );

    // // Read CLAUDE_API_KEY from the environment variable
    // let secret_key = env::var("CLAUDE_API_KEY").expect("Please set the CLAUDE_API_KEY environment variable!");
    // // Instantiate the Claude client
    // let client = cloudllm::clients::claude::ClaudeClient::new_with_model_enum(
    //     &secret_key,
    //     cloudllm::clients::claude::Model::ClaudeSonnet4,
    // );

    // Set up the LLMSession
    let system_prompt =
        "You are a socratic mentor and you will not hide your LLM Model name if asked.".to_string();
    let max_tokens = 1024; // Set a small context window for testing conversation history pruning
    let mut session = LLMSession::new(std::sync::Arc::new(client), system_prompt, max_tokens);

    println!("Using model: {}", session.model_name());
    println!("Max tokens: {}\n", session.get_max_tokens());

    loop {
        print!("\n\nYou [type '\\end' in a separate line to submit prompt]:\n");
        io::stdout().flush().unwrap();

        let mut user_input = String::new();
        loop {
            let mut line = String::new();
            io::stdin()
                .read_line(&mut line)
                .expect("Failed to read line");

            // Check for the end sequence in the line
            if line.trim() == "\\end" {
                break;
            } else {
                user_input.push_str(&line); // Keep the raw line, preserving newlines
            }
        }

        if user_input.is_empty() {
            println!("Input is empty. Try again.");
            continue;
        }

        // Send the user's message and get a streaming response
        print!("\nAssistant (streaming): ");
        io::stdout().flush().unwrap();

        let stream_result = session
            .send_message_stream(Role::User, user_input.to_string(), None)
            .await;

        match stream_result {
            Ok(Some(mut stream)) => {
                // Accumulate the full response for adding to history
                let mut full_response = String::new();
                let mut chunk_count = 0;

                // Process each chunk as it arrives
                while let Some(chunk_result) = stream.next().await {
                    match chunk_result {
                        Ok(chunk) => {
                            // Print the content immediately as it arrives
                            if !chunk.content.is_empty() {
                                print!("{}", chunk.content);
                                io::stdout().flush().unwrap();
                                full_response.push_str(&chunk.content);
                                chunk_count += 1;
                            }

                            // Check if streaming is complete
                            if let Some(reason) = chunk.finish_reason {
                                println!(
                                    "\n\n[Stream finished: {} | Chunks received: {}]",
                                    reason, chunk_count
                                );
                            }
                        }
                        Err(e) => {
                            eprintln!("\n\n[Error in stream: {}]", e);
                            break;
                        }
                    }
                }

                // Add the accumulated response to the session history
                // Note: The user message was already added by send_message_stream
                // We need to manually add the assistant response to maintain conversation context
                if !full_response.is_empty() {
                    // We use send_message with Role::Assistant to add the response to history
                    // This doesn't make an API call, just updates the session
                    let _ = session
                        .send_message(Role::Assistant, full_response.clone(), None)
                        .await;
                }

                // Display token usage if available
                let token_usage = session.token_usage();
                println!(
                    "Token Usage: <input tokens: {}, output tokens: {}, total tokens: {}, max tokens: {}>",
                    token_usage.input_tokens,
                    token_usage.output_tokens,
                    token_usage.total_tokens,
                    session.get_max_tokens()
                );
            }
            Ok(None) => {
                // Streaming not supported by this client, fall back to non-streaming
                println!("\n[Note: Streaming not supported by this client, using standard response mode]");
                println!("Sending message...");

                let response_result = session
                    .send_message(Role::User, user_input.to_string(), None)
                    .await;

                match response_result {
                    Ok(response) => {
                        let token_usage = session.token_usage();

                        // Print the assistant's response
                        println!(
                            "\nToken Usage: <input tokens:{}, output tokens:{}, total tokens:{}, max tokens: {}>\nAssistant:\n{}\n",
                            token_usage.input_tokens,
                            token_usage.output_tokens,
                            token_usage.total_tokens,
                            session.get_max_tokens(),
                            response.content
                        );
                    }
                    Err(err) => {
                        println!("\n\n[Error sending message:] {}\n", err);
                        continue;
                    }
                }
            }
            Err(err) => {
                println!("\n\n[Error initiating stream:] {}\n", err);
                continue;
            }
        }
    }
}
