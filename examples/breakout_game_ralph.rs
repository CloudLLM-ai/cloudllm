//! RALPH Orchestration Mode — Breakout Game Example
//!
//! This example demonstrates the RALPH (autonomous iterative loop) orchestration mode
//! by having multiple specialized agents collaborate to build a complete Atari Breakout
//! game in a single `index.html` file.
//!
//! RALPH works by repeatedly presenting agents with the same PRD task list. Agents see
//! accumulated work from previous iterations via conversation history and mark tasks
//! complete with `[TASK_COMPLETE:task_id]` markers. The loop ends when all tasks are
//! done or `max_iterations` is reached.
//!
//! ## Features
//!
//! - **BreakoutEventHandler**: Real-time pretty-printed event output
//! - **Shared Memory**: All agents share a Memory tool for coordination
//! - **write_game_file**: Custom tool that writes game files to disk
//!
//! ## Agents
//!
//! - **game-architect**: Designs HTML structure, CSS, Canvas setup
//! - **game-programmer**: Implements core game mechanics (physics, collision, rendering)
//! - **sound-designer**: Implements Atari 2600-style background music and SFX (Web Audio API)
//! - **powerup-engineer**: Implements all powerup systems
//!
//! ## PRD Tasks (10)
//!
//! 1. HTML boilerplate, canvas element, CSS styling
//! 2. requestAnimationFrame game loop, game state management
//! 3. Player paddle with keyboard input
//! 4. Ball movement, wall bouncing, paddle collision
//! 5. Brick grid with multi-hit bricks (different colors per HP)
//! 6. Ball-brick collision with brick destruction
//! 7. Atari 2600-style chiptune background music (Web Audio API oscillators)
//! 8. Sound effects on collisions
//! 9. Powerups: paddle length +10%, speed +10%, projectile shooting
//! 10. Spawn 2 extra balls powerup
//!
//! ## Running
//!
//! ```bash
//! export ANTHROPIC_KEY=your_key
//! cargo run --example breakout_game_ralph
//! ```
//!
//! The example writes the assembled game to `breakout_game.html` in the current directory.

use async_trait::async_trait;
use cloudllm::clients::claude::{ClaudeClient, Model};
use cloudllm::event::{AgentEvent, EventHandler, OrchestrationEvent};
use cloudllm::tool_protocol::{ToolMetadata, ToolParameter, ToolParameterType, ToolRegistry};
use cloudllm::tool_protocols::{CustomToolProtocol, MemoryProtocol};
use cloudllm::tools::Memory;
use cloudllm::{
    orchestration::{Orchestration, OrchestrationMode, RalphTask},
    Agent,
};
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::RwLock;

// ── Event Handler ──────────────────────────────────────────────────────────

/// Pretty-prints agent and orchestration events in real-time.
///
/// Implements [`EventHandler`] to provide a live progress display during
/// RALPH orchestration runs. Tracks elapsed time from construction and
/// formats each event as a timestamped line.
///
/// Handles the following events:
/// - `AgentEvent::SendStarted` / `SendCompleted` — agent generation lifecycle
/// - `AgentEvent::LLMCallStarted` / `LLMCallCompleted` — per-call LLM latency visibility
/// - `AgentEvent::ToolCallDetected` / `ToolExecutionCompleted` — tool usage tracking
/// - `OrchestrationEvent::RunStarted` / `RunCompleted` — orchestration banners
/// - `OrchestrationEvent::RalphIterationStarted` / `RalphTaskCompleted` — RALPH progress
/// - `OrchestrationEvent::AgentFailed` — error reporting
struct BreakoutEventHandler {
    /// Wall-clock instant captured at construction, used for elapsed time display.
    start: Instant,
}

impl BreakoutEventHandler {
    fn new() -> Self {
        Self {
            start: Instant::now(),
        }
    }

    fn elapsed_str(&self) -> String {
        let secs = self.start.elapsed().as_secs();
        format!("{:02}:{:02}", secs / 60, secs % 60)
    }
}

#[async_trait]
impl EventHandler for BreakoutEventHandler {
    async fn on_agent_event(&self, event: &AgentEvent) {
        match event {
            AgentEvent::SendStarted {
                agent_name,
                message_preview,
                ..
            } => {
                let preview_len = 80.min(message_preview.len());
                let preview_end = message_preview
                    .char_indices()
                    .nth(preview_len)
                    .map(|(i, _)| i)
                    .unwrap_or(message_preview.len());
                println!(
                    "  [{}] >> {} thinking... ({}...)",
                    self.elapsed_str(),
                    agent_name,
                    &message_preview[..preview_end]
                );
            }
            AgentEvent::SendCompleted {
                agent_name,
                tokens_used,
                response_length,
                tool_calls_made,
                ..
            } => {
                let tokens = tokens_used
                    .as_ref()
                    .map(|u| u.total_tokens)
                    .unwrap_or(0);
                println!(
                    "  [{}] << {} responded ({} chars, {} tokens, {} tool calls)",
                    self.elapsed_str(),
                    agent_name,
                    response_length,
                    tokens,
                    tool_calls_made
                );
            }
            AgentEvent::ToolCallDetected {
                agent_name,
                tool_name,
                parameters,
                iteration,
                ..
            } => {
                let params_str = serde_json::to_string(parameters).unwrap_or_default();
                println!(
                    "  [{}]    {} calling tool '{}' (iter {}) params={}",
                    self.elapsed_str(),
                    agent_name,
                    tool_name,
                    iteration,
                    params_str
                );
            }
            AgentEvent::ToolExecutionCompleted {
                agent_name,
                tool_name,
                parameters,
                success,
                error,
                ..
            } => {
                if *success {
                    println!(
                        "  [{}]    {} tool '{}' succeeded",
                        self.elapsed_str(),
                        agent_name,
                        tool_name
                    );
                } else {
                    let params_str = serde_json::to_string(parameters).unwrap_or_default();
                    println!(
                        "  [{}]    {} tool '{}' FAILED: {} | params={}",
                        self.elapsed_str(),
                        agent_name,
                        tool_name,
                        error.as_deref().unwrap_or("unknown"),
                        params_str
                    );
                }
            }
            AgentEvent::LLMCallStarted {
                agent_name,
                iteration,
                ..
            } => {
                println!(
                    "  [{}]    {} sending to LLM (round {})...",
                    self.elapsed_str(),
                    agent_name,
                    iteration
                );
            }
            AgentEvent::LLMCallCompleted {
                agent_name,
                iteration,
                tokens_used,
                response_length,
                ..
            } => {
                let tokens = tokens_used
                    .as_ref()
                    .map(|u| format!("{} tokens", u.total_tokens))
                    .unwrap_or_else(|| "no token info".to_string());
                println!(
                    "  [{}]    {} LLM round {} complete ({} chars, {})",
                    self.elapsed_str(),
                    agent_name,
                    iteration,
                    response_length,
                    tokens
                );
            }
            _ => {}
        }
    }

    async fn on_orchestration_event(&self, event: &OrchestrationEvent) {
        match event {
            OrchestrationEvent::RunStarted {
                orchestration_name,
                mode,
                agent_count,
                ..
            } => {
                println!();
                println!("{}", "=".repeat(80));
                println!(
                    "  {} — mode={}, agents={}",
                    orchestration_name, mode, agent_count
                );
                println!("{}", "=".repeat(80));
            }
            OrchestrationEvent::RalphIterationStarted {
                iteration,
                max_iterations,
                tasks_completed,
                tasks_total,
                ..
            } => {
                println!();
                println!("{}", "-".repeat(80));
                println!(
                    "  RALPH Iteration {}/{} — {}/{} tasks complete",
                    iteration, max_iterations, tasks_completed, tasks_total
                );
                println!("{}", "-".repeat(80));
            }
            OrchestrationEvent::RalphTaskCompleted {
                agent_name,
                task_ids,
                tasks_completed_total,
                tasks_total,
                ..
            } => {
                println!(
                    "  [{}] *** {} completed tasks: [{}] — progress: {}/{}",
                    self.elapsed_str(),
                    agent_name,
                    task_ids.join(", "),
                    tasks_completed_total,
                    tasks_total
                );
            }
            OrchestrationEvent::AgentFailed {
                agent_name, error, ..
            } => {
                println!(
                    "  [{}] !!! {} FAILED: {}",
                    self.elapsed_str(),
                    agent_name,
                    error
                );
            }
            OrchestrationEvent::RunCompleted {
                rounds,
                total_tokens,
                is_complete,
                ..
            } => {
                println!();
                println!("{}", "=".repeat(80));
                println!(
                    "  Run complete — {} rounds, {} tokens, complete={}",
                    rounds, total_tokens, is_complete
                );
                println!("{}", "=".repeat(80));
            }
            _ => {}
        }
    }
}

// ── Main ───────────────────────────────────────────────────────────────────

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    env_logger::Builder::from_default_env()
        .filter_level(log::LevelFilter::Info)
        .init();

    let api_key = match std::env::var("ANTHROPIC_KEY") {
        Ok(key) => key,
        Err(_) => {
            eprintln!("Error: Set ANTHROPIC_KEY environment variable.");
            std::process::exit(1);
        }
    };

    println!("\n{}", "=".repeat(80));
    println!("  RALPH Orchestration Mode — Atari Breakout Game Builder");
    println!("  Using model: Claude Haiku 4.5");
    println!("{}\n", "=".repeat(80));

    // ── Shared Memory + Custom Tools ──────────────────────────────────────
    // All agents share the same Memory store and custom tool protocol through
    // an Arc<RwLock<ToolRegistry>>. This allows agents to coordinate by reading
    // and writing to shared memory, and to write game files to disk.

    let memory = Arc::new(Memory::new());
    let memory_protocol = Arc::new(MemoryProtocol::new(memory.clone()));

    let custom_protocol = Arc::new(CustomToolProtocol::new());
    custom_protocol
        .register_tool(
            ToolMetadata::new("write_game_file", "Write content to a file on disk")
                .with_parameter(
                    ToolParameter::new("filename", ToolParameterType::String)
                        .with_description("The filename to write (e.g. 'breakout_game.html')"),
                )
                .with_parameter(
                    ToolParameter::new("content", ToolParameterType::String)
                        .with_description("The file content to write"),
                ),
            Arc::new(|params| {
                let filename = params["filename"]
                    .as_str()
                    .unwrap_or("output.html")
                    .to_string();
                let content = params["content"].as_str().unwrap_or("").to_string();
                std::fs::write(&filename, &content)?;
                Ok(cloudllm::tool_protocol::ToolResult::success(
                    serde_json::json!({"written": filename, "bytes": content.len()}),
                ))
            }),
        )
        .await;

    let mut shared_registry = ToolRegistry::empty();
    shared_registry
        .add_protocol("memory", memory_protocol)
        .await?;
    shared_registry
        .add_protocol("custom", custom_protocol)
        .await?;
    let shared_registry = Arc::new(RwLock::new(shared_registry));

    // ── Agents ──────────────────────────────────────────────────────────────

    let make_client = || {
        Arc::new(ClaudeClient::new_with_model_enum(
            &api_key,
            Model::ClaudeHaiku45,
        ))
    };

    let architect = Agent::new("game-architect", "Game Architect", make_client())
        .with_expertise("HTML5 structure, CSS layout, Canvas setup")
        .with_personality(
            "Meticulous front-end architect who produces clean, well-structured HTML/CSS.",
        )
        .with_shared_tools(shared_registry.clone());

    let programmer = Agent::new("game-programmer", "Game Programmer", make_client())
        .with_expertise("JavaScript game mechanics, physics, collision detection, rendering")
        .with_personality(
            "Seasoned game developer who writes tight, performant JavaScript game loops.",
        )
        .with_shared_tools(shared_registry.clone());

    let sound_designer = Agent::new("sound-designer", "Sound Designer", make_client())
        .with_expertise("Web Audio API, chiptune synthesis, oscillator-based sound effects")
        .with_personality(
            "Retro audio enthusiast who crafts authentic Atari 2600-era sounds with Web Audio API oscillators.",
        )
        .with_shared_tools(shared_registry.clone());

    let powerup_engineer = Agent::new("powerup-engineer", "Powerup Engineer", make_client())
        .with_expertise("Game powerup systems, spawn logic, timed effects")
        .with_personality(
            "Creative gameplay engineer who designs fun and balanced powerup mechanics.",
        )
        .with_shared_tools(shared_registry.clone());

    // ── PRD Tasks ───────────────────────────────────────────────────────────

    let tasks = vec![
        RalphTask::new(
            "html_structure",
            "HTML Structure",
            "Create the HTML boilerplate with a <canvas> element, CSS styling (centered canvas, \
             dark background, retro font), and all necessary <script> tags. Everything must be in \
             a single self-contained index.html.",
        ),
        RalphTask::new(
            "game_loop",
            "Game Loop",
            "Implement the requestAnimationFrame game loop with game state management \
             (menu, playing, paused, game_over). Include score tracking and lives display.",
        ),
        RalphTask::new(
            "paddle_control",
            "Paddle Control",
            "Implement the player paddle with left/right arrow key and A/D key input. \
             Paddle should be constrained to the canvas bounds.",
        ),
        RalphTask::new(
            "ball_physics",
            "Ball Physics",
            "Implement ball movement with velocity, wall bouncing (top, left, right), \
             paddle collision with angle reflection based on hit position, and bottom-of-screen \
             life loss.",
        ),
        RalphTask::new(
            "brick_layout",
            "Brick Layout",
            "Create a brick grid with multiple rows. Bricks have HP (1-3 hits) with different \
             colors per HP level (e.g., green=1HP, yellow=2HP, red=3HP). Display remaining HP visually.",
        ),
        RalphTask::new(
            "collision_detection",
            "Collision Detection",
            "Implement ball-brick collision detection. When a brick is hit, decrease its HP. \
             When HP reaches 0, destroy the brick and add score. Handle ball deflection on brick hit.",
        ),
        RalphTask::new(
            "background_music",
            "Background Music",
            "Implement Atari 2600-style chiptune background music using Web Audio API oscillators. \
             Use square and triangle waves to create a looping retro melody. Music should start on \
             game start and loop continuously.",
        ),
        RalphTask::new(
            "collision_sfx",
            "Collision Sound Effects",
            "Implement distinct sound effects for: ball-brick hit (high pitched blip), \
             ball-paddle hit (medium thud), ball-wall bounce (low click). Use Web Audio API \
             oscillators with short duration envelopes.",
        ),
        RalphTask::new(
            "powerups_basic",
            "Basic Powerups",
            "Implement powerups that drop from destroyed bricks (random chance): \
             paddle length +10% (green powerup), ball speed +10% (blue powerup), \
             projectile shooting with space bar (red powerup — fires 2 projectiles that deal \
             1 damage each to bricks). Powerups fall downward and are caught by the paddle.",
        ),
        RalphTask::new(
            "powerup_multiball",
            "Multiball Powerup",
            "Implement a multiball powerup (purple) that spawns 2 extra balls when collected. \
             Extra balls behave identically to the main ball. Game continues as long as at least \
             one ball remains. All balls interact with bricks and the paddle.",
        ),
    ];

    // ── Orchestration ───────────────────────────────────────────────────────

    let system_context = "\
You are collaborating with other specialized agents to build a complete Atari Breakout game \
in a single self-contained index.html file. All HTML, CSS, and JavaScript must be inline. \
Do NOT use external dependencies. Use the HTML5 Canvas API for rendering and the Web Audio API \
for sound. \n\n\
When you work on a task, output the COMPLETE updated index.html incorporating ALL previous work \
plus your additions. Never output partial snippets — always output the full file. \n\n\
When a task is fully implemented, include the marker [TASK_COMPLETE:task_id] at the end of your \
response (e.g., [TASK_COMPLETE:html_structure]). You may complete multiple tasks at once.\n\n\
You have access to shared Memory and a write_game_file tool:\n\
- Memory: Use PUT/GET/LIST commands to coordinate with other agents (e.g., store design decisions)\n\
- write_game_file: Write the game HTML to a file (filename + content parameters)";

    // Register the event handler on the orchestration. It will be
    // auto-propagated to each agent added via add_agent(), giving us
    // a unified stream of both OrchestrationEvents and AgentEvents.
    let event_handler = Arc::new(BreakoutEventHandler::new());

    let mut orchestration =
        Orchestration::new("breakout-builder", "Breakout Game RALPH Orchestration")
            .with_mode(OrchestrationMode::Ralph {
                tasks,
                max_iterations: 5,
            })
            .with_system_context(system_context)
            .with_max_tokens(180_000)
            .with_event_handler(event_handler);

    orchestration.add_agent(architect)?;
    orchestration.add_agent(programmer)?;
    orchestration.add_agent(sound_designer)?;
    orchestration.add_agent(powerup_engineer)?;

    // ── Run ─────────────────────────────────────────────────────────────────

    let prompt = "\
Build a complete Atari Breakout game in a single self-contained index.html. \
The game should feature: an HTML5 Canvas, a game loop with state management, \
keyboard-controlled paddle, ball physics with angle reflection, multi-hit bricks \
with color-coded HP, collision detection, Atari 2600-style chiptune background music, \
collision sound effects, powerups (paddle size, speed boost, projectile shooting), \
and a multiball powerup. Everything must work in a modern browser with no external dependencies.";

    println!("Starting RALPH orchestration with 4 agents and 10 PRD tasks...\n");

    let start = Instant::now();
    let response = orchestration.run(prompt, 1).await?;
    let elapsed = start.elapsed();

    // ── Results ─────────────────────────────────────────────────────────────

    let minutes = elapsed.as_secs() / 60;
    let seconds = elapsed.as_secs() % 60;

    println!("\n{}", "=".repeat(80));
    println!("  RALPH Results");
    println!("{}", "=".repeat(80));
    println!("  Iterations used : {}", response.round);
    println!("  All tasks done  : {}", response.is_complete);
    println!(
        "  Completion score: {:.0}%",
        response.convergence_score.unwrap_or(0.0) * 100.0
    );
    println!("  Total tokens    : {}", response.total_tokens_used);
    println!("  Messages        : {}", response.messages.len());
    println!("  Elapsed time    : {}m {}s", minutes, seconds);
    println!("{}\n", "=".repeat(80));

    // Print per-message summary
    for (i, msg) in response.messages.iter().enumerate() {
        let agent = msg.agent_name.as_deref().unwrap_or("unknown");
        let iteration = msg
            .metadata
            .get("iteration")
            .map(|s| s.as_str())
            .unwrap_or("?");
        let completed = msg
            .metadata
            .get("tasks_completed")
            .map(|s| s.as_str())
            .unwrap_or("-");
        let preview_len = 120.min(msg.content.len());
        let preview_end = msg
            .content
            .char_indices()
            .nth(preview_len)
            .map(|(i, _)| i)
            .unwrap_or(msg.content.len());
        println!(
            "  [{}] iter={} agent={:<20} tasks_completed={:<30} preview={}...",
            i + 1,
            iteration,
            agent,
            completed,
            &msg.content[..preview_end]
        );
    }

    // ── Memory Dump ────────────────────────────────────────────────────────

    let keys = memory.list_keys();
    if !keys.is_empty() {
        println!("\n{}", "-".repeat(80));
        println!("  Shared Memory ({} entries)", keys.len());
        println!("{}", "-".repeat(80));
        for key in &keys {
            if let Some((value, _)) = memory.get(key, false) {
                let preview_len = 120.min(value.len());
                let preview_end = value
                    .char_indices()
                    .nth(preview_len)
                    .map(|(i, _)| i)
                    .unwrap_or(value.len());
                println!("  {}: {}...", key, &value[..preview_end]);
            }
        }
    }

    // ── Extract HTML ───────────────────────────────────────────────────────

    // Extract the last message's content as the final HTML (the last agent output
    // should contain the most complete version of the file).
    if let Some(last_msg) = response.messages.last() {
        // Try to extract just the HTML from the response
        let html = extract_html(&last_msg.content);
        std::fs::write("breakout_game.html", &html)?;
        println!(
            "\nGame written to breakout_game.html ({} bytes)",
            html.len()
        );
        println!("Open it in a browser to play!");
    } else {
        println!("\nNo messages were generated. Check your API key and try again.");
    }

    Ok(())
}

/// Attempt to extract a self-contained HTML document from an LLM response.
/// Falls back to using the entire string if no `<!DOCTYPE` or `<html` tag is found.
fn extract_html(text: &str) -> String {
    // Look for the start of an HTML document
    let lower = text.to_lowercase();
    let start = lower
        .find("<!doctype")
        .or_else(|| lower.find("<html"))
        .unwrap_or(0);

    // Look for the closing </html> tag
    let end = lower
        .rfind("</html>")
        .map(|i| i + "</html>".len())
        .unwrap_or(text.len());

    text[start..end].to_string()
}
