//! OpenAI Bitcoin Price Fetcher
//!
//! A simple example demonstrating real-time Bitcoin price lookup using OpenAI's
//! Responses API with the web_search tool.
//!
//! This example shows:
//! - Using OpenAI's web_search tool for real-time data
//! - Minimal setup required for tool-based queries
//! - Parsing and extracting information from web search results
//!
//! Run with:
//! ```bash
//! export OPEN_AI_SECRET=your_openai_key
//! cargo run --example openai_bitcoin_price_example
//! ```
//!
//! **Note**: This example requires:
//! - An OpenAI API key with access to gpt-5 or gpt-4o models
//! - These models support the Responses API with tool calling

use cloudllm::client_wrapper::{ClientWrapper, Message, Role};
use cloudllm::clients::openai::{Model, OpenAIClient};
use std::sync::Arc;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logger
    env_logger::Builder::from_default_env()
        .filter_level(log::LevelFilter::Info)
        .init();

    println!("\n{}", "=".repeat(70));
    println!("  Bitcoin Price Fetcher - Using OpenAI Web Search");
    println!("{}\n", "=".repeat(70));

    // Get API key from environment
    let openai_key =
        std::env::var("OPEN_AI_SECRET").expect("OPEN_AI_SECRET environment variable must be set");

    // Create OpenAI client with gpt-5 (supports Responses API)
    let client = Arc::new(OpenAIClient::new_with_model_enum(&openai_key, Model::GPT5));

    println!("✓ OpenAI client initialized");
    println!("✓ Model: {}\n", client.model_name());

    // Prepare the request with system context for financial data
    let messages = vec![
        Message {
            role: Role::System,
            content: Arc::from(
                "You are a financial analyst. When asked about cryptocurrency prices, \
                 search for the current market data and provide accurate, up-to-date prices. \
                 Include the source of your information and any relevant market context.",
            ),
            tool_calls: vec![],
        },
        Message {
            role: Role::User,
            content: Arc::from(
                "What is the current price of Bitcoin in USD? \
                 Please search for the latest price and provide context about recent price movements.",
            ),
            tool_calls: vec![],
        },
    ];

    println!("📊 Fetching current Bitcoin price...\n");

    // Send request (no ToolDefinition tools for this example)
    let response = client.send_message(&messages, None).await?;

    // Display the response
    println!("{}", "=".repeat(70));
    println!("Bitcoin Price Information:\n");
    println!("{}", response.content);
    println!("\n{}", "=".repeat(70));

    // Display token usage
    if let Some(usage) = client.get_last_usage().await {
        println!(
            "\n📈 Token Usage - Input: {}, Output: {}, Total: {}",
            usage.input_tokens, usage.output_tokens, usage.total_tokens
        );
    }

    println!("\n✓ Bitcoin price fetch completed successfully!\n");

    Ok(())
}
