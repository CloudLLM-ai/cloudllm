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
//! ## PRD Tasks (18)
//!
//! **Core Mechanics (1-6)**
//! 1. HTML boilerplate, canvas element, CSS styling, responsive sizing
//! 2. requestAnimationFrame game loop, game state management (MENU, PLAYING, PAUSED, GAME_OVER, LEVEL_COMPLETE)
//! 3. Player paddle with keyboard input (arrow keys), left/right bounds checking
//! 4. Ball movement, velocity vectors, wall bouncing (top, left, right), paddle collision with angle reflection
//! 5. Brick grid with multi-hit HP system (1-5 HP, color-coded: yellow=1, green=2, blue=3, orange=4, red=5)
//! 6. Ball-brick collision detection, brick HP damage system, score tracking, powerup drops
//!
//! **Audio System (7-8)**
//! 7. Atari 2600-style chiptune background music (Web Audio API oscillators), loop and mute controls
//! 8. Collision sound effects (brick, paddle, wall) with different pitches, powerup pickup sound, life earned sound
//!
//! **Powerup System (9-11)**
//! 9. Basic Powerups: paddle extension, speed boost (slow), projectile shooting (100 shots)
//! 10. Advanced Powerups: lava balls (destroy on impact), bomb mode (5 impacts), growth (50% size), mushroom (1UP)
//! 11. Multiball powerup: spawns 10 new balls from paddle position
//!
//! **Visual Effects (12-14)**
//! 12. Particle effects: fire particles (brick destruction), paddle jet particles (level complete animation), 1UP text displays
//! 13. Paddle 3D animation (screw effect with wing rotation) on level complete, with upward motion
//! 14. Level complete celebration animation with animated paddle and particle bursts
//!
//! **Advanced Mechanics (15-18)**
//! 15. Level progression system with 10+ procedural brick patterns (pyramid, diamond, checkerboard, wave, spiral, etc.)
//! 16. Dynamic brick HP and powerup scaling based on level difficulty
//! 17. Mobile touch/swipe controls with responsive canvas resizing on window change
//! 18. Score milestones for automatic 1UP awards, lives system, level persistence
//!
//! ## Running
//!
//! ```bash
//! export ANTHROPIC_API_KEY=your_key
//! cargo run --example breakout_game_ralph
//! ```
//!
//! The example writes the assembled game to `breakout_game_ralph.html` in the current directory.

use async_trait::async_trait;
use cloudllm::clients::claude::{ClaudeClient, Model};
use cloudllm::event::{AgentEvent, EventHandler, OrchestrationEvent};
use cloudllm::tool_protocol::{ToolMetadata, ToolParameter, ToolParameterType, ToolRegistry};
use cloudllm::tool_protocols::{BashProtocol, CustomToolProtocol, HttpClientProtocol, MemoryProtocol};
use cloudllm::tools::{BashTool, HttpClient, Memory, Platform};
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
                let tokens = tokens_used.as_ref().map(|u| u.total_tokens).unwrap_or(0);
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
                        .with_description("The filename to write (e.g. 'breakout_game_ralph.html')"),
                )
                .with_parameter(
                    ToolParameter::new("content", ToolParameterType::String)
                        .with_description("The file content to write"),
                ),
            Arc::new(|params| {
                let filename = params["filename"]
                    .as_str()
                    .unwrap_or("breakout_game_ralph.html")
                    .to_string();
                let content = params["content"].as_str().unwrap_or("").to_string();
                std::fs::write(&filename, &content)?;
                Ok(cloudllm::tool_protocol::ToolResult::success(
                    serde_json::json!({"written": filename, "bytes": content.len()}),
                ))
            }),
        )
        .await;

    // Set up Bash protocol for command execution
    // Auto-detect platform (Linux or macOS)
    #[cfg(target_os = "macos")]
    let bash_tool = Arc::new(BashTool::new(Platform::macOS));
    #[cfg(target_os = "linux")]
    let bash_tool = Arc::new(BashTool::new(Platform::Linux));
    #[cfg(not(any(target_os = "macos", target_os = "linux")))]
    let bash_tool = Arc::new(BashTool::new(Platform::Linux)); // Fallback

    let bash_protocol = Arc::new(BashProtocol::new(bash_tool));

    // Set up HTTP Client protocol for web requests
    let http_client = Arc::new(HttpClient::new());
    let http_protocol = Arc::new(HttpClientProtocol::new(http_client));

    // Create shared tool registry with all protocols
    let mut shared_registry = ToolRegistry::empty();
    shared_registry
        .add_protocol("memory", memory_protocol)
        .await?;
    shared_registry
        .add_protocol("custom", custom_protocol)
        .await?;
    shared_registry
        .add_protocol("bash", bash_protocol)
        .await?;
    shared_registry
        .add_protocol("http", http_protocol)
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
            "HTML Structure & Canvas Setup",
            "Create HTML boilerplate with <canvas> element (800x600), responsive CSS styling \
             (dark background #000000, retro font), centered game container, and touch control \
             buttons (left, right, fire). Implement canvas resizing on window resize.",
        ),
        RalphTask::new(
            "game_loop",
            "Game Loop & State Management",
            "Implement requestAnimationFrame game loop with 5 game states: MENU, PLAYING, PAUSED, \
             GAME_OVER, LEVEL_COMPLETE. Include score tracking, lives display (HUD), current level display, \
             powerup status indicator, and frame rate stability.",
        ),
        RalphTask::new(
            "paddle_control",
            "Paddle Control & Input",
            "Implement paddle movement with keyboard input (arrow keys, A/D) and mouse tracking. \
             Paddle constrained to canvas bounds. Implement pause/unpause with spacebar. Display paddle \
             width scaling visually.",
        ),
        RalphTask::new(
            "ball_physics",
            "Ball Physics & Collision",
            "Implement ball velocity vector, wall bouncing (top, left, right with perfect reflection), \
             paddle collision with angle reflection based on hit position (center vs edge), bottom-of-screen \
             life loss, ball speed clamping (min 3, max 8).",
        ),
        RalphTask::new(
            "brick_layout",
            "Brick Grid & HP System",
            "Create brick grid (11 columns x 5 rows) with multi-hit HP system (1-5 HP). Color-code by HP: \
             yellow=1HP, green=2HP, blue=3HP, orange=4HP, red=5HP. Display HP visually via text or color. \
             Support random powerup loot from bricks.",
        ),
        RalphTask::new(
            "collision_detection",
            "Ball-Brick Collision Detection",
            "Implement precise ball-brick collision detection with spatial hashing. On hit: decrease brick HP, \
             handle ball deflection (top/bottom vs left/right), award points (10 * maxHP), trigger destruction \
             when HP=0, spawn powerup drops with random type selection.",
        ),
        RalphTask::new(
            "background_music",
            "Background Music & Audio System",
            "Implement Atari 2600-style chiptune background music using Web Audio API oscillators (square & triangle waves). \
             Create looping melody that starts on game start, loops continuously, supports pause/resume, \
             mute button control, and volume slider.",
        ),
        RalphTask::new(
            "collision_sfx",
            "Sound Effects System",
            "Implement distinct Web Audio API sound effects: ball-brick collision (high pitched blip, 100-200ms), \
             ball-paddle collision (medium thud, 150-250ms), ball-wall bounce (low click, 50-100ms), \
             powerup pickup sound, life earned sound, and level complete fanfare.",
        ),
        RalphTask::new(
            "powerups_basic",
            "Basic Powerups System",
            "Implement 3 basic powerups dropping from destroyed bricks: paddle extension (extends width 20%), \
             speed boost (slows all balls to 50% speed for 30 seconds), projectile system (activates missile \
             firing with 100 shots, 4 damage per hit). Powerups fall, have collision detection with paddle.",
        ),
        RalphTask::new(
            "advanced_powerups",
            "Advanced Powerups",
            "Implement 4 advanced powerups: lava balls (balls destroy bricks on contact for 30 seconds, \
             yellow/orange trail), bomb mode (balls become bombs, destroy bricks in 5 impacts, black with \
             impact counter), growth (balls grow 50% larger, white border), mushroom (1UP award, red, triggers \
             life earned animation).",
        ),
        RalphTask::new(
            "projectile_missiles",
            "Projectile & Missile System",
            "Implement projectile firing from paddle (space bar while projectile powerup active). Projectiles \
             travel upward, have smoke trail particle effects, deal 1 damage per brick hit, can penetrate \
             multiple bricks. Draw missile cannons on paddle when active. Support up to 100 shots per powerup.",
        ),
        RalphTask::new(
            "multiball_powerup",
            "Multiball Powerup",
            "Implement multiball powerup (purple) that spawns 10 new balls at paddle position with varied angles. \
             New balls have 50% speed of current balls, behave identically to main ball (physics, collision, powerups). \
             Game continues with all active balls until all lost.",
        ),
        RalphTask::new(
            "particle_effects",
            "Particle Effects System",
            "Implement 3 particle systems: fire particles (brick destruction bursts, radial spread, decay over time), \
             paddle jet particles (level complete animation, upward spray from paddle wings), 1UP text displays \
             (floating score notifications with fade-out). Support particle physics (velocity, gravity, color, alpha).",
        ),
        RalphTask::new(
            "paddle_animation",
            "Paddle 3D Animation & Level Complete",
            "Implement 3D paddle screw effect animation on level complete: paddle flies upward with rotating \
             wings (4 full rotations), squash/stretch effect, glowing blue appearance. Add wing cannons visual \
             when projectiles active. Smooth animation over 3 seconds.",
        ),
        RalphTask::new(
            "level_system",
            "Level Progression & Patterns",
            "Implement level system with 10+ procedural brick patterns: level 1=classic grid, level 2+=pyramid, \
             diamond, checkerboard, stripe, wave, spiral, hourglass, cross, rings, random patterns. Use seeded \
             RNG for deterministic layouts. Increment level on all bricks cleared, award 1UP on level complete.",
        ),
        RalphTask::new(
            "brick_difficulty_scaling",
            "Brick HP & Difficulty Scaling",
            "Implement dynamic brick HP scaling by level (max HP increases with level, variable 1-5). Scale \
             brick HP based on row position (top=harder). Adjust powerup drop chances by level (reduce at high \
             levels). Increase brick density with level. Support seeded random for reproducible difficulty curves.",
        ),
        RalphTask::new(
            "mobile_controls",
            "Mobile Touch & Responsive Design",
            "Implement touch/swipe controls for mobile: touch-to-aim paddle movement, swipe for rapid movement, \
             buttons for fire (spacebar equivalent). Detect mobile device and show touch UI. Implement responsive \
             canvas resizing on window change. Support both portrait and landscape orientations.",
        ),
        RalphTask::new(
            "scoring_lives_persistence",
            "Scoring, Lives System & Level Persistence",
            "Implement score tracking with point awards (brick=10*HP, powerup=100-500). Implement lives system \
             (start with 3, lose 1 on ball lost). Automatic 1UP awards at score milestones (every 2500 points). \
             Persist level progress across lives. Display all stats in HUD (score, level, lives, powerup status).",
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
You have access to a comprehensive toolkit for coordination and development:\n\
- Memory (memory:*): Use PUT/GET/LIST commands to coordinate (e.g., store design decisions)\n\
- Bash (bash:*): Execute shell commands for file operations, git, testing, debugging\n\
- HTTP Client (http:*): Make web requests (http_get, http_post, http_put, http_delete, http_patch)\n\
- Custom Tools (custom:write_game_file): Write the game HTML to a file (filename + content parameters)";

    // Register the event handler on the orchestration. It will be
    // auto-propagated to each agent added via add_agent(), giving us
    // a unified stream of both OrchestrationEvents and AgentEvents.
    let event_handler = Arc::new(BreakoutEventHandler::new());

    let mut orchestration =
        Orchestration::new("breakout-builder", "Breakout Game RALPH Orchestration")
            .with_mode(OrchestrationMode::Ralph {
                tasks,
                max_iterations: 8,
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
Build a complete, feature-rich Atari Breakout game in a single self-contained index.html. \
The game should feature: 800x600 responsive Canvas with 5 game states (MENU, PLAYING, PAUSED, \
GAME_OVER, LEVEL_COMPLETE), paddle with keyboard/mouse/touch input, realistic ball physics with \
angle reflection, multi-hit bricks (1-5 HP, color-coded) with HP scaling by level and position, \
comprehensive collision detection, Atari 2600-style chiptune background music with mute/volume, \
distinct collision sound effects and powerup sounds, 8 powerup types (paddle, speed, projectile, \
lava, bomb, growth, mushroom, multiball), projectile system with smoke trails, particle effects \
(fire, jets, 1UP displays), 3D paddle animation on level complete, 10+ procedural brick patterns \
for 15+ levels, dynamic difficulty scaling, mobile touch controls, responsive canvas resizing, \
score milestones for automatic 1UP awards, and lives system. Everything must work in a modern \
browser with no external dependencies.";

    println!("Starting RALPH orchestration with 4 agents and 18 PRD tasks...\n");

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
        std::fs::write("breakout_game_ralph.html", &html)?;
        println!(
            "\nGame written to breakout_game_ralph.html ({} bytes)",
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
