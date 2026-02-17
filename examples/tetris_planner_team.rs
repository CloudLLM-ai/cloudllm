use std::fs;
use std::path::PathBuf;
use std::sync::Arc;

use cloudllm::clients::claude::{ClaudeClient, Model};
use cloudllm::planner::{
    BasicPlanner, NoopMemory, NoopPolicy, NoopStream, Planner, PlannerContext, UserMessage,
};
use cloudllm::tool_protocol::{
    ToolMetadata, ToolParameter, ToolParameterType, ToolRegistry, ToolResult,
};
use cloudllm::tool_protocols::CustomToolProtocol;
use cloudllm::LLMSession;

struct AgentTask {
    name: &'static str,
    system_prompt: &'static str,
    task_prompt: String,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let key = std::env::var("ANTHROPIC_KEY")?;
    let client: Arc<dyn cloudllm::ClientWrapper> =
        Arc::new(ClaudeClient::new_with_model_enum(&key, Model::ClaudeHaiku45));

    let output_path = std::env::current_dir()?.join("tetris_planner_output.html");
    if !output_path.exists() {
        fs::write(&output_path, "")?;
    }

    let protocol = Arc::new(CustomToolProtocol::new());
    protocol
        .register_tool(
            ToolMetadata::new("read_file", "Read a file from disk")
                .with_parameter(
                    ToolParameter::new("path", ToolParameterType::String)
                        .with_description("Path to the file")
                        .required(),
                ),
            Arc::new(move |params| {
                let path = params["path"].as_str().unwrap_or("");
                let content = fs::read_to_string(path).unwrap_or_default();
                Ok(ToolResult::success(serde_json::json!({
                    "path": path,
                    "content": content,
                    "exists": PathBuf::from(path).exists()
                })))
            }),
        )
        .await;

    protocol
        .register_tool(
            ToolMetadata::new("write_file", "Write content to a file")
                .with_parameter(
                    ToolParameter::new("path", ToolParameterType::String)
                        .with_description("Path to the file")
                        .required(),
                )
                .with_parameter(
                    ToolParameter::new("content", ToolParameterType::String)
                        .with_description("Full file contents")
                        .required(),
                ),
            Arc::new(move |params| {
                let path = params["path"].as_str().unwrap_or("");
                let content = params["content"].as_str().unwrap_or("");
                if path.is_empty() {
                    return Ok(ToolResult::failure("missing path".to_string()));
                }
                if let Err(err) = fs::write(path, content) {
                    return Ok(ToolResult::failure(err.to_string()));
                }
                Ok(ToolResult::success(serde_json::json!({
                    "path": path,
                    "bytes": content.len()
                })))
            }),
        )
        .await;

    let mut registry = ToolRegistry::new(protocol);
    registry.discover_tools_from_primary().await?;

    let tasks = vec![
        AgentTask {
            name: "Architect",
            system_prompt: "You design clear, minimal game systems.",
            task_prompt: format!(
                "Create a complete single-file HTML/CSS/JS Tetris game. \
                 Include: 10x20 grid, next-piece preview, scoring, hold piece, and \
                 keyboard controls. Use a clean NES-like visual style. \
                 Start from scratch if the file is empty. \
                 Read the current file with read_file, then write the full updated file \
                 to {} using write_file.",
                output_path.display()
            ),
        },
        AgentTask {
            name: "Gameplay",
            system_prompt: "You refine gameplay and animation details.",
            task_prompt: format!(
                "Enhance the Tetris game with smooth animations: soft drop glow, \
                 line-clear flash, and piece lock-in effect. Ensure scoring matches \
                 classic rules and the next-piece preview works reliably. \
                 Read the current file with read_file, then write the full updated file \
                 to {} using write_file.",
                output_path.display()
            ),
        },
        AgentTask {
            name: "Audio",
            system_prompt: "You craft minimalistic NES-style audio.",
            task_prompt: format!(
                "Add sound effects (rotate, drop, line clear, game over) and a minimal \
                 NES-inspired looping background tune using Web Audio. Keep it subtle \
                 and provide a toggle to mute. Read the current file with read_file, then \
                 write the full updated file to {} using write_file.",
                output_path.display()
            ),
        },
    ];

    let planner = BasicPlanner::new();
    for task in tasks {
        let mut session = LLMSession::new(client.clone(), task.system_prompt.to_string(), 128_000);
        let outcome = planner
            .plan(
                UserMessage::from(task.task_prompt.as_str()),
                PlannerContext {
                    session: &mut session,
                    tools: &registry,
                    policy: &NoopPolicy,
                    memory: &NoopMemory,
                    streamer: &NoopStream,
                    grok_tools: None,
                    openai_tools: None,
                },
            )
            .await?;
        println!("{} complete. {}", task.name, outcome.final_message);
    }

    println!("Tetris game written to: {}", output_path.display());
    Ok(())
}