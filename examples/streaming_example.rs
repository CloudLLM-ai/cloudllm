use cloudllm::client_wrapper::Role;
/// Example demonstrating streaming support for LLM responses.
/// This example shows how to receive tokens as they arrive from the LLM,
/// providing a much better user experience with reduced perceived latency.
use cloudllm::clients::openai::{Model, OpenAIClient};
use cloudllm::{ClientWrapper, LLMSession};
use futures_util::StreamExt;
use std::io::{self, Write};
use std::sync::Arc;

#[tokio::main]
async fn main() {
    // Initialize logger
    cloudllm::init_logger();

    println!("=== CloudLLM Streaming Example ===\n");

    // Get API key from environment
    let secret_key = match std::env::var("OPEN_AI_SECRET") {
        Ok(key) => key,
        Err(_) => {
            eprintln!("Error: OPEN_AI_SECRET environment variable not set");
            eprintln!("Please set it with: export OPEN_AI_SECRET=your_api_key");
            std::process::exit(1);
        }
    };

    // Create OpenAI client with a fast model
    let client = OpenAIClient::new_with_model_enum(&secret_key, Model::GPT41Nano);
    println!("Using model: {}\n", client.model_name());

    // Create a session
    let mut session = LLMSession::new(
        Arc::new(client),
        "You are a helpful assistant. Keep responses concise.".to_string(),
        8192,
    );

    // Example 1: Streaming through LLMSession
    println!("Example 1: LLMSession streaming");
    println!("==================================\n");

    match session
        .send_message_stream(
            Role::User,
            "Write a haiku about Rust programming.".to_string(),
            None,
            None,
        )
        .await
    {
        Ok(Some(mut stream)) => {
            print!("Assistant (streaming): ");
            io::stdout().flush().unwrap();

            let mut full_response = String::new();
            while let Some(chunk_result) = stream.next().await {
                match chunk_result {
                    Ok(chunk) => {
                        if !chunk.content.is_empty() {
                            print!("{}", chunk.content);
                            io::stdout().flush().unwrap();
                            full_response.push_str(&chunk.content);
                        }

                        if let Some(reason) = chunk.finish_reason {
                            println!("\n[Finished: {}]", reason);
                        }
                    }
                    Err(e) => {
                        eprintln!("\nError in stream: {}", e);
                        break;
                    }
                }
            }

            // After collecting the full response, you can manually add it to history if needed
            println!("\nAccumulated response: {} chars", full_response.len());

            // Optionally add the streamed response to history for context
            if !full_response.is_empty() {
                // Note: The user message was already added; now add the assistant response
                let _ = session
                    .send_message(Role::Assistant, full_response, None, None)
                    .await;
            }
        }
        Ok(None) => {
            println!("Streaming not supported by this client\n");
        }
        Err(e) => {
            eprintln!("Error initiating stream: {}\n", e);
        }
    }

    println!("\n=== Streaming Example Complete ===");
}
