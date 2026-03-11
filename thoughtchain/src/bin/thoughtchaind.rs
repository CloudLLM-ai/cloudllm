//! Standalone ThoughtChain daemon.
//!
//! This binary starts both:
//!
//! - an MCP server
//! - a REST server
//!
//! Configuration is read from environment variables:
//!
//! - `THOUGHTCHAIN_DIR`
//! - `THOUGHTCHAIN_DEFAULT_KEY`
//! - `THOUGHTCHAIN_BIND_HOST`
//! - `THOUGHTCHAIN_MCP_PORT`
//! - `THOUGHTCHAIN_REST_PORT`

use thoughtchain::server::{start_servers, ThoughtChainServerConfig};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let config = ThoughtChainServerConfig::from_env();
    let handles = start_servers(config.clone()).await?;

    println!("thoughtchaind running");
    println!("Chain directory: {}", config.service.chain_dir.display());
    println!("Default chain key: {}", config.service.default_chain_key);
    println!("MCP server:  http://{}", handles.mcp.local_addr());
    println!("REST server: http://{}", handles.rest.local_addr());

    tokio::signal::ctrl_c().await?;
    Ok(())
}
