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
//! - `THOUGHTCHAIN_DEFAULT_STORAGE_ADAPTER`
//! - `THOUGHTCHAIN_VERBOSE`
//! - `THOUGHTCHAIN_BIND_HOST`
//! - `THOUGHTCHAIN_MCP_PORT`
//! - `THOUGHTCHAIN_REST_PORT`
//! - `RUST_LOG`

use env_logger::Env;
use thoughtchain::server::{start_servers, ThoughtChainServerConfig};
use thoughtchain::{migrate_registered_chains, ThoughtChainMigrationEvent};

const THOUGHTCHAIN_BANNER: &str = r#"████████╗██╗  ██╗ ██████╗ ██╗   ██╗ ██████╗ ██╗  ██╗████████╗ ██████╗██╗  ██╗ █████╗ ██╗███╗   ██╗
╚══██╔══╝██║  ██║██╔═══██╗██║   ██║██╔════╝ ██║  ██║╚══██╔══╝██╔════╝██║  ██║██╔══██╗██║████╗  ██║
   ██║   ███████║██║   ██║██║   ██║██║  ███╗███████║   ██║   ██║     ███████║███████║██║██╔██╗ ██║
   ██║   ██╔══██║██║   ██║██║   ██║██║   ██║██╔══██║   ██║   ██║     ██╔══██║██╔══██║██║██║╚██╗██║
   ██║   ██║  ██║╚██████╔╝╚██████╔╝╚██████╔╝██║  ██║   ██║   ╚██████╗██║  ██║██║  ██║██║██║ ╚████║
   ╚═╝   ╚═╝  ╚═╝ ╚═════╝  ╚═════╝  ╚═════╝ ╚═╝  ╚═╝   ╚═╝    ╚═════╝╚═╝  ╚═╝╚═╝  ╚═╝╚═╝╚═╝  ╚═══╝"#;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    init_logger();
    let config = ThoughtChainServerConfig::from_env();

    println!("{}", THOUGHTCHAIN_BANNER);
    println!("thoughtchain v{}", env!("CARGO_PKG_VERSION"));
    println!("thoughtchaind starting");
    println!("Configuration:");
    print_env_var(
        "THOUGHTCHAIN_DIR",
        Some(config.service.chain_dir.display().to_string()),
    );
    print_env_var(
        "THOUGHTCHAIN_DEFAULT_KEY",
        Some(config.service.default_chain_key.clone()),
    );
    print_env_var(
        "THOUGHTCHAIN_DEFAULT_STORAGE_ADAPTER",
        Some(config.service.default_storage_adapter.to_string()),
    );
    print_env_var(
        "THOUGHTCHAIN_VERBOSE",
        Some(config.service.verbose.to_string()),
    );
    print_env_var(
        "THOUGHTCHAIN_BIND_HOST",
        Some(config.mcp_addr.ip().to_string()),
    );
    print_env_var(
        "THOUGHTCHAIN_MCP_PORT",
        Some(config.mcp_addr.port().to_string()),
    );
    print_env_var(
        "THOUGHTCHAIN_REST_PORT",
        Some(config.rest_addr.port().to_string()),
    );

    let migration_reports =
        migrate_registered_chains(&config.service.chain_dir, |event| match event {
            ThoughtChainMigrationEvent::Started {
                chain_key,
                from_version,
                to_version,
                current,
                total,
            } => println!(
                "{} Migrating chain {} from version {} to version {}",
                progress_bar(current, total),
                chain_key,
                from_version,
                to_version
            ),
            ThoughtChainMigrationEvent::Completed {
                chain_key,
                from_version,
                to_version,
                current,
                total,
            } => println!(
                "{} Migrated chain {} from version {} to version {}",
                progress_bar(current, total),
                chain_key,
                from_version,
                to_version
            ),
        })?;
    if migration_reports.is_empty() {
        println!("No chain migrations required.");
    }

    let handles = start_servers(config.clone()).await?;

    println!("thoughtchaind running");
    println!("Resolved endpoints:");
    println!("MCP server:  http://{}", handles.mcp.local_addr());
    println!("REST server: http://{}", handles.rest.local_addr());
    println!("Press Ctrl+C to stop.");

    tokio::signal::ctrl_c().await?;
    Ok(())
}

fn print_env_var(name: &str, effective_value: Option<String>) {
    match std::env::var(name) {
        Ok(raw_value) => println!(
            "  {name}={raw_value} (effective: {})",
            display_value(effective_value)
        ),
        Err(_) => println!(
            "  {name}=<unset> (effective default: {})",
            display_value(effective_value)
        ),
    }
}

fn display_value(value: Option<String>) -> String {
    value.unwrap_or_else(|| "<none>".to_string())
}

fn init_logger() {
    let mut builder = env_logger::Builder::from_env(Env::default().default_filter_or("info"));
    builder.format_timestamp_millis();
    let _ = builder.try_init();
}

fn progress_bar(current: usize, total: usize) -> String {
    let total = total.max(1);
    let current = current.min(total);
    let filled = ((current * 20) / total).min(20);
    format!(
        "[{}{}] {}/{}",
        "#".repeat(filled),
        "-".repeat(20 - filled),
        current,
        total
    )
}
