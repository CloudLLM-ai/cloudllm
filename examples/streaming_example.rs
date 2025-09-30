use std::env;
use std::io::{self, Write};

use cloudllm::client_wrapper::Role;
use cloudllm::clients::openai::OpenAIClient;
use cloudllm::ClientWrapper;
use futures_util::StreamExt;

// Run from the root folder of the repo as follows:
// OPEN_AI_SECRET=your-open-ai-key-here cargo run --example streaming_example

#[tokio::main]
async fn main() {
    // Read OPEN_AI_SECRET from environment variable
    let secret_key =
        env::var("OPEN_AI_SECRET").expect("Please set the OPEN_AI_SECRET environment variable!");

    // Instantiate the OpenAI client
    let client = OpenAIClient::new_with_model_enum(
        &secret_key,
        cloudllm::clients::openai::Model::GPT5Nano,
    );

    // Create messages
    let messages = vec![
        cloudllm::client_wrapper::Message {
            role: Role::System,
            content: "You are a helpful assistant.".to_string(),
        },
        cloudllm::client_wrapper::Message {
            role: Role::User,
            content: "Tell me a short story about a robot learning to paint.".to_string(),
        },
    ];

    println!("Streaming response from AI:");
    println!("----------------------------");

    // Send message and get streaming response
    match client.send_message_stream(messages, None).await {
        Ok(mut stream) => {
            let mut full_content = String::new();
            
            // Process chunks as they arrive
            while let Some(chunk_result) = stream.next().await {
                match chunk_result {
                    Ok(chunk) => {
                        // Print the chunk immediately (this is what gives us low latency!)
                        print!("{}", chunk.content);
                        io::stdout().flush().unwrap();
                        
                        full_content.push_str(&chunk.content);
                        
                        if chunk.is_final {
                            println!("\n\n[Stream complete]");
                            break;
                        }
                    }
                    Err(e) => {
                        eprintln!("\nError in stream: {}", e);
                        break;
                    }
                }
            }
            
            println!("\n----------------------------");
            println!("Total characters received: {}", full_content.len());
        }
        Err(e) => {
            eprintln!("Error starting stream: {}", e);
        }
    }
}
