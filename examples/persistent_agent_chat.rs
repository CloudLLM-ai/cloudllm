//! Persistent CLI chat agent backed by ThoughtChain over MCP.
//!
//! By default this example starts a local ThoughtChain MCP server on an
//! ephemeral localhost port, then creates a GPT-5.4 CloudLLM agent with:
//! - remote ThoughtChain memory tools over MCP
//! - local memory, bash, HTTP, calculator, and filesystem tools
//!
//! The agent restores prior memory on startup and persists each completed turn
//! back into ThoughtChain so it can remember previous sessions.

#[path = "support/persistent_agent_tools.rs"]
mod persistent_agent_tools;

use std::env;
use std::io::{self, Write};
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;

use cloudllm::clients::openai::{Model, OpenAIClient};
use cloudllm::tool_protocol::ToolProtocol;
use cloudllm::Agent;
use persistent_agent_tools::build_persistent_agent_registry;
use serde_json::json;
use thoughtchain::server::{default_thoughtchain_dir, start_mcp_server, ThoughtChainServiceConfig};

const DEFAULT_CHAIN_KEY: &str = "persistent-chat-agent";

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    cloudllm::init_logger();

    let api_key = env::var("OPEN_AI_SECRET")
        .map_err(|_| "Please set OPEN_AI_SECRET to run persistent_agent_chat")?;

    let chain_key =
        env::var("THOUGHTCHAIN_CHAIN_KEY").unwrap_or_else(|_| DEFAULT_CHAIN_KEY.to_string());
    let chain_dir = env::var("THOUGHTCHAIN_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| default_thoughtchain_dir());
    let filesystem_root = env::var("CLOUDLLM_CHAT_FS_ROOT")
        .map(PathBuf::from)
        .unwrap_or(env::current_dir()?);

    let mut embedded_server = None;
    let thoughtchain_endpoint = if let Ok(endpoint) = env::var("THOUGHTCHAIN_MCP_ENDPOINT") {
        endpoint
    } else {
        let server = start_mcp_server(
            SocketAddr::from(([127, 0, 0, 1], 0)),
            ThoughtChainServiceConfig::new(
                chain_dir.clone(),
                chain_key.clone(),
                thoughtchain::StorageAdapterKind::Jsonl,
            ),
        )
        .await?;
        let endpoint = format!("http://{}", server.local_addr());
        embedded_server = Some(server);
        endpoint
    };

    let (registry, thoughtchain_protocol) =
        build_persistent_agent_registry(&thoughtchain_endpoint, filesystem_root.clone()).await?;

    bootstrap_chain(&thoughtchain_protocol, &chain_key).await?;
    let restored_memory = load_memory_markdown(&thoughtchain_protocol, &chain_key).await?;
    append_session_checkpoint(
        &thoughtchain_protocol,
        &chain_key,
        "Session started for the persistent CLI chat agent.",
    )
    .await?;

    let client = Arc::new(OpenAIClient::new_with_model_enum(&api_key, Model::GPT54));
    let mut agent = Agent::new("persistent-chat", "Persistent Chat Agent", client)
        .with_expertise(
            "Long-running user collaboration, durable memory management, coding, shell, HTTP, and file operations",
        )
        .with_personality("Direct, pragmatic, memory-aware, and concise")
        .with_tools(registry);

    let system_prompt = build_system_prompt(&chain_key, &filesystem_root, &restored_memory);
    agent.set_system_prompt(&system_prompt);

    println!("Persistent Agent Chat");
    println!("Model: gpt-5.4");
    println!("ThoughtChain MCP endpoint: {}", thoughtchain_endpoint);
    println!("ThoughtChain directory: {}", chain_dir.display());
    println!("ThoughtChain chain key: {}", chain_key);
    println!("Filesystem root: {}", filesystem_root.display());
    if embedded_server.is_some() {
        println!("ThoughtChain MCP server mode: embedded local server");
    } else {
        println!("ThoughtChain MCP server mode: external endpoint");
    }
    println!("Commands: /help, /tools, /memory, /recent, /search <text>, /remember <note>, /exit");
    println!("Input: press Enter to send. End a line with \\ to continue onto the next line.");

    loop {
        print!("\nYou:\n");
        io::stdout().flush()?;

        let user_input = read_continuation_input()?;
        let trimmed = user_input.trim();
        if trimmed.is_empty() {
            continue;
        }

        if trimmed == "/exit" {
            break;
        }
        if trimmed == "/help" {
            print_help();
            continue;
        }
        if trimmed == "/tools" {
            let tools = agent.list_tools().await;
            println!("Available tools:");
            for tool in tools {
                println!("  - {}", tool);
            }
            continue;
        }
        if trimmed == "/memory" {
            let markdown = load_memory_markdown(&thoughtchain_protocol, &chain_key).await?;
            println!("{markdown}");
            continue;
        }
        if trimmed == "/recent" {
            let result = thoughtchain_protocol
                .execute(
                    "thoughtchain_recent_context",
                    json!({"chain_key": chain_key, "last_n": 12}),
                )
                .await?;
            println!(
                "{}",
                result.output["prompt"]
                    .as_str()
                    .unwrap_or("(no recent context)")
            );
            continue;
        }
        if let Some(query) = trimmed.strip_prefix("/search ") {
            let result = thoughtchain_protocol
                .execute(
                    "thoughtchain_search",
                    json!({"chain_key": chain_key, "text": query, "limit": 8}),
                )
                .await?;
            println!("{}", serde_json::to_string_pretty(&result.output)?);
            continue;
        }
        if let Some(note) = trimmed.strip_prefix("/remember ") {
            let result = thoughtchain_protocol
                .execute(
                    "thoughtchain_append",
                    json!({
                        "chain_key": chain_key,
                        "thought_type": "Insight",
                        "role": "Memory",
                        "importance": 0.9,
                        "tags": ["manual-note"],
                        "content": note,
                    }),
                )
                .await?;
            println!("{}", serde_json::to_string_pretty(&result.output)?);
            continue;
        }

        println!("Assistant is thinking...");
        let response = agent.send(&user_input).await?;
        println!("\nAssistant:\n{}\n", response.content);

        persist_turn(
            &thoughtchain_protocol,
            &chain_key,
            &user_input,
            &response.content,
        )
        .await?;
    }

    println!("Session ended.");
    Ok(())
}

fn build_system_prompt(
    chain_key: &str,
    filesystem_root: &PathBuf,
    restored_memory: &str,
) -> String {
    format!(
        "You are a persistent GPT-5.4 powered CloudLLM agent in a terminal chat.\n\
Your durable memory lives in ThoughtChain and is exposed over MCP tools.\n\
Chain key: {chain_key}\n\
Filesystem root: {}\n\n\
Behavior rules:\n\
- Use thoughtchain_search when a user request may depend on prior sessions.\n\
- Use thoughtchain_append whenever you learn durable user preferences, constraints, decisions, plans, corrections, insights, or surprises.\n\
- Keep stored memories concise, factual, and semantically typed.\n\
- Do not store secrets unless the user explicitly asks you to remember them.\n\
- Use other tools normally for coding, shell, filesystem, HTTP, and calculations.\n\n\
Restored durable memory:\n{}\n",
        filesystem_root.display(),
        restored_memory
    )
}

fn read_continuation_input() -> Result<String, io::Error> {
    let mut user_input = String::new();
    loop {
        let mut line = String::new();
        io::stdin().read_line(&mut line)?;

        if let Some(stripped) = line.strip_suffix("\\\n") {
            user_input.push_str(stripped);
            user_input.push('\n');
            print!("> ");
            io::stdout().flush()?;
            continue;
        }
        if let Some(stripped) = line.strip_suffix("\\\r\n") {
            user_input.push_str(stripped);
            user_input.push('\n');
            print!("> ");
            io::stdout().flush()?;
            continue;
        }

        user_input.push_str(&line);
        break;
    }
    Ok(user_input)
}

fn print_help() {
    println!("Commands:");
    println!("  /help            Show this help");
    println!("  /tools           List tools available to the agent");
    println!("  /memory          Print MEMORY.md exported from ThoughtChain");
    println!("  /recent          Print recent ThoughtChain catch-up context");
    println!("  /search <text>   Search ThoughtChain memories by text");
    println!("  /remember <note> Store a manual durable memory");
    println!("  /exit            Quit the chat");
    println!("\nInput behavior:");
    println!("  Press Enter to send the current message");
    println!("  End a line with \\ to continue onto the next line");
}

async fn bootstrap_chain(
    thoughtchain_protocol: &Arc<cloudllm::tool_protocols::McpClientProtocol>,
    chain_key: &str,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    thoughtchain_protocol
        .execute(
            "thoughtchain_bootstrap",
            json!({
                "chain_key": chain_key,
                "content": "Bootstrap memory for the persistent CloudLLM CLI chat agent. Preserve durable user preferences, constraints, plans, decisions, insights, corrections, and summaries across sessions.",
                "importance": 1.0,
                "tags": ["bootstrap", "system"],
                "concepts": ["persistence", "semantic-memory", "cli-chat"]
            }),
        )
        .await?;
    Ok(())
}

async fn append_session_checkpoint(
    thoughtchain_protocol: &Arc<cloudllm::tool_protocols::McpClientProtocol>,
    chain_key: &str,
    content: &str,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    thoughtchain_protocol
        .execute(
            "thoughtchain_append",
            json!({
                "chain_key": chain_key,
                "thought_type": "Checkpoint",
                "role": "Checkpoint",
                "importance": 0.4,
                "tags": ["session"],
                "content": content,
            }),
        )
        .await?;
    Ok(())
}

async fn load_memory_markdown(
    thoughtchain_protocol: &Arc<cloudllm::tool_protocols::McpClientProtocol>,
    chain_key: &str,
) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
    let result = thoughtchain_protocol
        .execute(
            "thoughtchain_memory_markdown",
            json!({
                "chain_key": chain_key,
                "limit": 80,
            }),
        )
        .await?;

    Ok(result.output["markdown"]
        .as_str()
        .unwrap_or("# MEMORY\n\n")
        .to_string())
}

async fn persist_turn(
    thoughtchain_protocol: &Arc<cloudllm::tool_protocols::McpClientProtocol>,
    chain_key: &str,
    user_input: &str,
    assistant_output: &str,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let content = format!(
        "Conversation turn summary.\nUser: {}\nAssistant: {}",
        truncate_for_memory(user_input, 800),
        truncate_for_memory(assistant_output, 1200)
    );

    thoughtchain_protocol
        .execute(
            "thoughtchain_append",
            json!({
                "chain_key": chain_key,
                "thought_type": "Summary",
                "role": "Memory",
                "importance": 0.6,
                "tags": ["conversation-turn"],
                "concepts": ["session-memory"],
                "content": content,
            }),
        )
        .await?;

    Ok(())
}

fn truncate_for_memory(input: &str, max_chars: usize) -> String {
    let mut truncated = input.trim().chars().take(max_chars).collect::<String>();
    if input.trim().chars().count() > max_chars {
        truncated.push_str("...");
    }
    truncated
}
