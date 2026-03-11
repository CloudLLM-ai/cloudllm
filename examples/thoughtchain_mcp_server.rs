//! Run a dedicated ThoughtChain MCP server on localhost.
//!
//! This exposes ThoughtChain as a remote MCP tool source so agents can use
//! durable semantic memory without linking directly to the storage layer.

#[path = "support/thoughtchain_mcp.rs"]
mod thoughtchain_mcp;

use std::env;
use std::net::SocketAddr;

use thoughtchain_mcp::{
    default_thoughtchain_dir, start_thoughtchain_mcp_server, ThoughtChainMcpConfig,
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    cloudllm::init_logger();

    let chain_dir = env::var("THOUGHTCHAIN_DIR")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| default_thoughtchain_dir());
    let chain_key = env::var("THOUGHTCHAIN_DEFAULT_KEY")
        .unwrap_or_else(|_| "persistent-chat-agent".to_string());
    let port = env::var("THOUGHTCHAIN_PORT")
        .ok()
        .and_then(|value| value.parse::<u16>().ok())
        .unwrap_or(9471);

    let addr = SocketAddr::from(([127, 0, 0, 1], port));
    let server = start_thoughtchain_mcp_server(
        addr,
        ThoughtChainMcpConfig {
            chain_dir: chain_dir.clone(),
            default_chain_key: chain_key.clone(),
        },
    )
    .await?;

    println!(
        "ThoughtChain MCP server running at http://{}",
        server.get_addr()
    );
    println!("Chain directory: {}", chain_dir.display());
    println!("Default chain key: {}", chain_key);
    println!("Available tools:");
    println!("  - thoughtchain_bootstrap");
    println!("  - thoughtchain_append");
    println!("  - thoughtchain_search");
    println!("  - thoughtchain_recent_context");
    println!("  - thoughtchain_memory_markdown");
    println!("  - thoughtchain_head");

    tokio::signal::ctrl_c().await?;
    Ok(())
}
