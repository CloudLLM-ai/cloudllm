use std::env;
use std::io::{self, Write};

use cloudllm::client_wrapper::Role;
use cloudllm::clients::openai::OpenAIClient;
use cloudllm::LLMSession;
use futures_util::StreamExt;

// Run from the root folder of the repo as follows:
// OPEN_AI_SECRET=your-open-ai-key-here cargo run --example streaming_session_example

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

    // Create an LLM session
    let system_prompt = "You are a creative writing assistant.".to_string();
    let max_tokens = 4096;
    let mut session = LLMSession::new(std::sync::Arc::new(client), system_prompt, max_tokens);

    println!("Streaming Session Example");
    println!("========================\n");

    // First message with streaming
    println!("User: Write a haiku about coding\n");
    println!("Assistant (streaming):");
    
    match session
        .send_message_stream(
            Role::User,
            "Write a haiku about coding".to_string(),
            None,
        )
        .await
    {
        Ok(mut stream) => {
            let mut full_response = String::new();
            
            // Process chunks as they arrive
            while let Some(chunk_result) = stream.next().await {
                match chunk_result {
                    Ok(chunk) => {
                        // Print each chunk immediately for low latency display
                        print!("{}", chunk.content);
                        io::stdout().flush().unwrap();
                        
                        full_response.push_str(&chunk.content);
                        
                        if chunk.is_final {
                            break;
                        }
                    }
                    Err(e) => {
                        eprintln!("\nError in stream: {}", e);
                        break;
                    }
                }
            }
            
            println!("\n");
            println!("[Received {} characters]\n", full_response.len());
            
            // Note: In a real application, you'd want to add the assistant's response
            // to the session history manually if you're using streaming, like this:
            // session.conversation_history.push(Message {
            //     role: Role::Assistant,
            //     content: full_response,
            // });
        }
        Err(e) => {
            eprintln!("Error starting stream: {}", e);
        }
    }

    println!("---");
    println!("\nNote: With streaming, token usage tracking is not automatically updated.");
    println!("Use the non-streaming send_message() method if you need token tracking.");
}
