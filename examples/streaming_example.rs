use cloudllm::client_wrapper::Role;
/// Example demonstrating streaming support for LLM responses.
/// This example shows how to receive tokens as they arrive from the LLM,
/// providing a much better user experience with reduced perceived latency.
use cloudllm::clients::openai::{Model, OpenAIClient};
use cloudllm::clients::sse_stream::StreamAccumulator;
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
    let client = OpenAIClient::new_with_model_enum(&secret_key, Model::GPT5Nano);
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
        )
        .await
    {
        Ok(Some(mut stream)) => {
            print!("Assistant (streaming): ");
            io::stdout().flush().unwrap();

            let mut acc = StreamAccumulator::new();
            while let Some(chunk_result) = stream.next().await {
                match chunk_result {
                    Ok(chunk) => {
                        if !chunk.reasoning.is_empty() {
                            print!("\x1b[90m{}\x1b[0m", chunk.reasoning);
                            io::stdout().flush().unwrap();
                        }
                        if !chunk.content.is_empty() {
                            print!("{}", chunk.content);
                            io::stdout().flush().unwrap();
                        }

                        if let Some(reason) = chunk.finish_reason.clone() {
                            println!("\n[Finished: {}]", reason);
                        }
                        acc.apply(&chunk);
                    }
                    Err(e) => {
                        eprintln!("\nError in stream: {}", e);
                        break;
                    }
                }
            }

            let usage = acc.usage();
            let reply = acc.into_message();
            println!("\nAccumulated response: {} chars", reply.content.len());
            if reply.content.is_empty() && reply.tool_calls.is_empty() {
                session.rollback_last_message();
            } else {
                session.commit_streamed_reply(reply, usage).await;
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
