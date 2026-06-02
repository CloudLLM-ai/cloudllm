//! OpenRouter + MiniMax M3 example
//!
//! Demonstrates how to wire CloudLLM's `LLMSession` to OpenRouter and the
//! MiniMax M3 model — useful when OpenAI costs are no longer sustainable and
//! you want to A/B test against a different provider with no code changes
//! beyond the constructor.
//!
//! Run with:
//! ```bash
//! export OPENROUTER_API_KEY=sk-or-...
//! cargo run --example openrouter_basic
//! ```

use std::sync::Arc;

use cloudllm::client_wrapper::{ClientWrapper, Role};
use cloudllm::clients::openrouter::{Model, OpenRouterClient};
use cloudllm::LLMSession;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::Builder::from_default_env()
        .filter_level(log::LevelFilter::Info)
        .init();

    println!("\n{}", "=".repeat(70));
    println!("  OpenRouter + MiniMax M3 — CloudLLM");
    println!("{}\n", "=".repeat(70));

    let api_key = std::env::var("OPENROUTER_API_KEY")
        .expect("OPENROUTER_API_KEY environment variable must be set");

    let client = Arc::new(OpenRouterClient::new_with_model_enum(
        &api_key,
        Model::MinimaxM3,
    ));
    println!("✓ OpenRouter client initialised");
    println!("✓ Model: {}\n", client.model_name());

    let mut session = LLMSession::new(
        client.clone(),
        "You are a concise technical writer. Reply in at most three sentences.".to_string(),
        1_048_576,
    );

    let prompt = "Explain the difference between `String` and `&str` in Rust.";
    let reply = session
        .send_message(Role::User, prompt.to_string(), None)
        .await?;

    println!("Prompt : {prompt}");
    println!("Reply  : {}\n", reply.content);

    if let Some(usage) = client.get_last_usage().await {
        println!(
            "Tokens — input: {}, output: {}, total: {}",
            usage.input_tokens, usage.output_tokens, usage.total_tokens
        );
    }

    Ok(())
}
