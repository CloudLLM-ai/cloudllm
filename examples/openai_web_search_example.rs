//! OpenAI Web Search Example
//!
//! Demonstrates using the OpenAI Responses API with web_search tool.
//! This example shows how to:
//! - Use OpenAI tools (web_search, file_search, code_interpreter)
//! - Configure search context size (high/medium/low)
//! - Set geographic location filtering
//! - Integrate tools with the Client and LLMSession APIs
//!
//! Run with:
//! ```bash
//! export OPEN_AI_SECRET=your_openai_key
//! cargo run --example openai_web_search_example
//! ```
//!
//! **Note**: This example requires:
//! - An OpenAI API key with access to gpt-5 or gpt-4o models
//! - These models support the Responses API with tool calling
//! - Web search results are real-time and may vary

use cloudllm::client_wrapper::{ClientWrapper, Message, Role};
use cloudllm::clients::openai::{Model, OpenAIClient};
use cloudllm::LLMSession;
use openai_rust2::chat::{OpenAITool, UserLocation};
use std::sync::Arc;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logger for error tracking
    env_logger::Builder::from_default_env()
        .filter_level(log::LevelFilter::Info)
        .init();

    println!("\n{}", "=".repeat(80));
    println!("  OpenAI Web Search Example");
    println!("  Demonstrating Responses API with Tool Calling");
    println!("{}\n", "=".repeat(80));

    // Get API key from environment
    let openai_key = std::env::var("OPEN_AI_SECRET").expect("OPEN_AI_SECRET must be set");

    // Create OpenAI client with gpt-5 (supports Responses API)
    // Note: gpt-5 and gpt-4o are the recommended models for tool calling
    let client = OpenAIClient::new_with_model_enum(&openai_key, Model::GPT5);

    println!("✓ OpenAI client initialized with gpt-5");
    println!("✓ Model: {}\n", client.model_name());

    // Example 1: Basic web search without geographic filtering
    println!("{}", "=".repeat(80));
    println!("Example 1: Basic Web Search");
    println!("{}\n", "=".repeat(80));

    let messages_1 = vec![
        Message {
            role: Role::System,
            content: Arc::from("You are a helpful research assistant. Search the web for current information and provide a comprehensive answer."),
        },
        Message {
            role: Role::User,
            content: Arc::from("What are the latest developments in artificial intelligence this week? Provide a brief summary of the top 3 recent developments."),
        },
    ];

    // Create tools with basic web search (high context for detailed results)
    let tools_1 = vec![OpenAITool::web_search().with_search_context_size("high")];

    println!("Sending request with web_search tool (context: high)...\n");

    let response_1 = client
        .send_message(&messages_1, None, Some(tools_1))
        .await?;

    println!("Response:\n{}\n", response_1.content);

    if let Some(usage) = client.get_last_usage().await {
        println!(
            "Tokens - Input: {}, Output: {}, Total: {}\n",
            usage.input_tokens, usage.output_tokens, usage.total_tokens
        );
    }

    // Example 2: Web search with geographic filtering
    println!("{}", "=".repeat(80));
    println!("Example 2: Web Search with Geographic Filtering");
    println!("{}\n", "=".repeat(80));

    let messages_2 = vec![
        Message {
            role: Role::System,
            content: Arc::from(
                "You are a local events coordinator. Search for current events and news.",
            ),
        },
        Message {
            role: Role::User,
            content: Arc::from(
                "What are the major tech events happening in San Francisco this month?",
            ),
        },
    ];

    // Create tools with geographic filtering for US/San Francisco
    let location = UserLocation {
        country: Some("US".to_string()),
        city: Some("San Francisco".to_string()),
        region: Some("California".to_string()),
        timezone: Some("America/Los_Angeles".to_string()),
    };

    let tools_2 = vec![OpenAITool::web_search()
        .with_search_context_size("medium")
        .with_user_location(location)];

    println!("Sending request with geographic filtering (SF, CA)...\n");

    let response_2 = client
        .send_message(&messages_2, None, Some(tools_2))
        .await?;

    println!("Response:\n{}\n", response_2.content);

    if let Some(usage) = client.get_last_usage().await {
        println!(
            "Tokens - Input: {}, Output: {}, Total: {}\n",
            usage.input_tokens, usage.output_tokens, usage.total_tokens
        );
    }

    // Example 3: Using tools with LLMSession API
    println!("{}", "=".repeat(80));
    println!("Example 3: Web Search with LLMSession");
    println!("{}\n", "=".repeat(80));

    let session_client = Arc::new(OpenAIClient::new_with_model_enum(&openai_key, Model::GPT5));

    let mut session = LLMSession::new(
        session_client,
        "You are a helpful assistant with access to real-time web information.".to_string(),
        8192, // max context window
    );

    let web_search_tools = vec![OpenAITool::web_search().with_search_context_size("medium")];

    println!("Starting multi-turn conversation with web search capabilities...\n");

    // First message with web search
    let response = session
        .send_message(
            Role::User,
            "What is the current stock price of Tesla and what were the major news stories about Tesla this week?".to_string(),
            None, // no grok_tools
            Some(web_search_tools.clone()),
        )
        .await?;

    println!("Assistant: {}\n", response.content);

    let usage_1 = session.token_usage();
    println!(
        "Session tokens - Input: {}, Output: {}, Total: {}\n",
        usage_1.input_tokens, usage_1.output_tokens, usage_1.total_tokens
    );

    // Follow-up message (session maintains context)
    let follow_up = session
        .send_message(
            Role::User,
            "Based on that information, what do you think might happen to Tesla's stock price in the next quarter?".to_string(),
            None,
            Some(web_search_tools),
        )
        .await?;

    println!("Assistant: {}\n", follow_up.content);

    let usage_2 = session.token_usage();
    println!(
        "Session cumulative tokens - Input: {}, Output: {}, Total: {}\n",
        usage_2.input_tokens, usage_2.output_tokens, usage_2.total_tokens
    );

    // Summary
    println!("{}", "=".repeat(80));
    println!("Summary");
    println!("{}\n", "=".repeat(80));

    println!("✓ Web search tool working correctly");
    println!("✓ Geographic filtering functional");
    println!("✓ LLMSession multi-turn context maintained");
    println!("\nAll OpenAI Responses API examples completed successfully!");

    Ok(())
}
