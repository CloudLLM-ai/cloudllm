//! GitHub Copilot example: sign in with the device flow, list the models your
//! plan includes, and stream a reply.
//!
//! On the first run the example signs you in. It prints a code and a URL, you
//! approve the OAuth app on GitHub, and it prints the token to export so later
//! runs skip the login.
//!
//! Run with:
//! ```bash
//! # Your OAuth app (https://github.com/settings/developers, "Enable Device Flow").
//! export GITHUB_COPILOT_CLIENT_ID=Iv1....
//! # Optional: the Copilot-Integration-Id GitHub accepts for your integration.
//! export GITHUB_COPILOT_INTEGRATION_ID=...
//! # Optional: skip the login with a token from an earlier run.
//! export GITHUB_COPILOT_OAUTH_TOKEN=gho_...
//! # Optional: the model to chat with (default: the first chat model listed).
//! export GITHUB_COPILOT_MODEL=gpt-4.1
//! cargo run --example github_copilot_chat
//! ```

use std::io::Write;
use std::sync::Arc;

use cloudllm::client_wrapper::{ClientWrapper, Message, Role};
use cloudllm::clients::github_copilot::{
    CopilotClient, CopilotIdentity, CopilotTokenProvider, GitHubDeviceFlow, GitHubOAuthToken,
};
use futures_util::StreamExt;

/// A stored token from `GITHUB_COPILOT_OAUTH_TOKEN`, or a fresh device-flow
/// login.
async fn github_token() -> Result<GitHubOAuthToken, Box<dyn std::error::Error>> {
    if let Ok(saved) = std::env::var("GITHUB_COPILOT_OAUTH_TOKEN") {
        return Ok(GitHubOAuthToken::new(saved));
    }
    let client_id = std::env::var("GITHUB_COPILOT_CLIENT_ID")
        .map_err(|_| "set GITHUB_COPILOT_CLIENT_ID to your OAuth app's client ID")?;
    let token = GitHubDeviceFlow::new(client_id)
        .authorize(|code| {
            println!(
                "Open {} and enter the code {}",
                code.verification_uri, code.user_code
            );
            println!("Waiting for approval...");
        })
        .await?;
    println!("\nSigned in. To skip the login next time:");
    println!("  export GITHUB_COPILOT_OAUTH_TOKEN={}\n", token.secret());
    Ok(token)
}

/// The default identity, plus `GITHUB_COPILOT_INTEGRATION_ID` when set.
fn identity() -> CopilotIdentity {
    match std::env::var("GITHUB_COPILOT_INTEGRATION_ID") {
        Ok(id) => CopilotIdentity::default().with_integration_id(id),
        Err(_) => CopilotIdentity::default(),
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let provider = Arc::new(
        CopilotTokenProvider::from_github_token(github_token().await?).with_identity(identity()),
    );

    let models = cloudllm::clients::github_copilot::list_models(&provider).await?;
    let chat_models: Vec<_> = models.iter().filter(|model| model.is_chat()).collect();
    println!("Chat models on this account:");
    for model in &chat_models {
        println!(
            "  {:<32} {}",
            model.id,
            model.vendor.as_deref().unwrap_or("")
        );
    }

    let model = match std::env::var("GITHUB_COPILOT_MODEL") {
        Ok(model) => model,
        Err(_) => chat_models
            .first()
            .map(|model| model.id.clone())
            .ok_or("Copilot listed no chat models for this account")?,
    };
    let client = CopilotClient::new(provider, model);
    println!("\n{} says:", client.model_name());

    let messages = [Message {
        role: Role::User,
        content: Arc::from("In two sentences, why does Rust have a borrow checker?"),
        tool_calls: vec![],
    }];
    // Stream errors are `Send + Sync` boxes; `to_string` lets `?` turn them
    // into this function's error type.
    let stream = client
        .send_message_stream(&messages, None)
        .await
        .map_err(|error| error.to_string())?;
    match stream {
        Some(mut stream) => {
            while let Some(chunk) = stream.next().await {
                print!("{}", chunk.map_err(|error| error.to_string())?.content);
                std::io::stdout().flush()?;
            }
            println!();
        }
        None => println!("{}", client.send_message(&messages, None).await?.content),
    }
    Ok(())
}
