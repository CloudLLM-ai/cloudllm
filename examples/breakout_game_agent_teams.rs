//! AnthropicAgentTeams Orchestration Mode — Breakout Game Example
//!
//! This example demonstrates the AnthropicAgentTeams (decentralized task-based) orchestration
//! mode by having multiple Claude Haiku 4.5 agents autonomously discover, claim, and complete
//! tasks from a shared Memory pool to build a complete Atari Breakout game in a single
//! `breakout_game.html` file.
//!
//! AnthropicAgentTeams works by:
//! 1. Initializing a task pool in Memory with keys: `teams:<pool_id>:unclaimed:<task_id>`
//! 2. Each iteration, agents autonomously:
//!    - LIST available unclaimed tasks
//!    - SELECT a task to work on
//!    - PUT a claim: `teams:<pool_id>:claimed:<task_id>`
//!    - WORK on the task (via LLM)
//!    - PUT the result: `teams:<pool_id>:completed:<task_id>`
//! 3. Orchestration tracks progress by querying Memory for completed tasks
//! 4. Loop terminates when all tasks completed or max_iterations reached
//!
//! ## Features
//!
//! - **Decentralized Coordination**: No central orchestrator; agents self-select work from task pool
//! - **Shared Memory**: All agents share a hierarchical Memory structure for coordination
//! - **TeamsEventHandler**: Real-time pretty-printed event output showing task claims and completions
//! - **write_game_file**: Custom tool that writes game files to disk
//!
//! ## Agents (All Claude Haiku 4.5)
//!
//! - **architect**: Discovers and works on HTML/CSS/canvas setup tasks
//! - **core-engineer**: Discovers and works on physics, collision, rendering tasks
//! - **audio-engineer**: Discovers and works on music and SFX tasks
//! - **features-engineer**: Discovers and works on powerups, effects, and advanced mechanics
//!
//! ## Task Pool (18 items)
//!
//! Tasks are organized in Memory with hierarchical keys:
//! - `teams:<pool_id>:unclaimed:<task_id>` — discovered by agents via LIST
//! - `teams:<pool_id>:claimed:<task_id>` — marked by agent when starting work
//! - `teams:<pool_id>:completed:<task_id>` — marked by agent when finished
//!
//! ### Core Mechanics (6 tasks)
//! - html_structure: HTML boilerplate, canvas, responsive CSS, touch controls
//! - game_loop: Game loop, 5 states, HUD with score/lives/level
//! - paddle_control: Keyboard, mouse, and touch input
//! - ball_physics: Velocity vectors, wall bouncing, paddle reflection
//! - brick_layout: Brick grid with 1-5 HP, color coding, loot system
//! - collision_detection: Ball-brick collision with HP damage and scoring
//!
//! ### Audio System (2 tasks)
//! - background_music: Atari 2600-style chiptune with loops and muting
//! - collision_sfx: Ball/paddle/wall collision sounds, powerup sound, 1UP sound
//!
//! ### Powerup System (3 tasks)
//! - powerups_basic: Paddle extension, speed boost, projectile system (100 shots)
//! - advanced_powerups: Lava balls, bomb mode, growth, mushroom (1UP)
//! - multiball_powerup: Spawns 10 new balls from paddle
//!
//! ### Visual Effects (3 tasks)
//! - particle_effects: Fire particles, paddle jet particles, 1UP text displays
//! - paddle_animation: 3D screw effect on level complete
//! - level_complete_anim: Celebration animation with upward paddle motion
//!
//! ### Advanced Mechanics (4 tasks)
//! - level_system: 10+ procedural patterns, 15+ levels
//! - brick_difficulty_scaling: HP scaling by level and position
//! - mobile_controls: Touch/swipe, responsive canvas resizing
//! - scoring_persistence: Score milestones, lives, 1UP awards
//!
//! ## Running
//!
//! ```bash
//! export ANTHROPIC_API_KEY=your_key
//! cargo run --example breakout_game_agent_teams
//! ```
//!
//! Expected runtime: 10-15 minutes (depending on LLM response times)
//! Expected cost: $2.00-$4.00 (Claude Haiku 4.5 is cost-effective)
//!
//! The example writes the assembled game to `breakout_game.html` in the current directory.

use async_trait::async_trait;
use cloudllm::clients::claude::{ClaudeClient, Model};
use cloudllm::event::{EventHandler, OrchestrationEvent};
use cloudllm::tool_protocol::{ToolMetadata, ToolParameter, ToolParameterType, ToolRegistry};
use cloudllm::tool_protocols::{BashProtocol, CustomToolProtocol, HttpClientProtocol, MemoryProtocol};
use cloudllm::tools::{BashTool, HttpClient, Memory, Platform};
use cloudllm::{
    orchestration::{Orchestration, OrchestrationMode, WorkItem},
    Agent,
};
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::RwLock;

// ── Event Handler ──────────────────────────────────────────────────────────

/// Pretty-prints orchestration and agent events in real-time.
///
/// Displays task claims, completions, and failures as they happen, providing
/// a live progress view of the decentralized task coordination.
struct TeamsEventHandler {
    start: Instant,
}

impl TeamsEventHandler {
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
impl EventHandler for TeamsEventHandler {
    async fn on_orchestration_event(&self, event: &OrchestrationEvent) {
        match event {
            OrchestrationEvent::RunStarted {
                orchestration_id: _,
                orchestration_name,
                mode,
                agent_count,
            } => {
                println!(
                    "\n🚀 {} — {} mode with {} agents",
                    orchestration_name, mode, agent_count
                );
            }
            OrchestrationEvent::RoundStarted {
                orchestration_id: _,
                round,
            } => {
                println!("\n📍 Iteration {} [{}]", round, self.elapsed_str());
            }
            OrchestrationEvent::AgentSelected {
                orchestration_id: _,
                agent_id: _,
                agent_name,
                reason,
            } => {
                println!("  → {} ({})", agent_name, reason);
            }
            OrchestrationEvent::TaskClaimed {
                orchestration_id: _,
                agent_id: _,
                agent_name,
                task_id,
            } => {
                println!("    ✋ {} claimed: {}", agent_name, task_id);
            }
            OrchestrationEvent::TaskCompleted {
                orchestration_id: _,
                agent_id: _,
                agent_name,
                task_id,
                result: _,
            } => {
                println!("    ✅ {} completed: {}", agent_name, task_id);
            }
            OrchestrationEvent::TaskFailed {
                orchestration_id: _,
                agent_id: _,
                agent_name,
                task_id,
                error,
            } => {
                println!("    ❌ {} failed on {}: {}", agent_name, task_id, error);
            }
            OrchestrationEvent::RoundCompleted { .. } => {
                // Less verbose
            }
            OrchestrationEvent::RunCompleted {
                orchestration_id: _,
                orchestration_name: _,
                rounds,
                total_tokens,
                is_complete,
            } => {
                println!(
                    "\n✨ Run completed in {} iterations, {} tokens, complete={}",
                    rounds, total_tokens, is_complete
                );
            }
            _ => {}
        }
    }
}

// ── Memory Setup Helper ────────────────────────────────────────────────────

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    println!("════════════════════════════════════════════════════════════════════════════════");
    println!("    Breakout Game — AnthropicAgentTeams (Decentralized Task Coordination)");
    println!("════════════════════════════════════════════════════════════════════════════════");

    // ── Task Pool ───────────────────────────────────────────────────────────────

    let tasks = vec![
        // Core Mechanics (6 tasks)
        WorkItem::new(
            "html_structure",
            "HTML Structure & Canvas Setup",
            "Create HTML boilerplate with <canvas> element (800x600), responsive CSS styling \
             (dark background #000000, retro font), centered game container, and touch control \
             buttons (left, right, fire). Implement canvas resizing on window resize.",
        ),
        WorkItem::new(
            "game_loop",
            "Game Loop & State Management",
            "Implement requestAnimationFrame game loop with 5 game states: MENU, PLAYING, PAUSED, \
             GAME_OVER, LEVEL_COMPLETE. Include score tracking, lives display (HUD), current level display, \
             powerup status indicator, and frame rate stability.",
        ),
        WorkItem::new(
            "paddle_control",
            "Paddle Control & Input",
            "Implement paddle movement with keyboard input (arrow keys, A/D) and mouse tracking. \
             Paddle constrained to canvas bounds. Implement pause/unpause with spacebar. Display paddle \
             width scaling visually.",
        ),
        WorkItem::new(
            "ball_physics",
            "Ball Physics & Collision",
            "Implement ball velocity vector, wall bouncing (top, left, right with perfect reflection), \
             paddle collision with angle reflection based on hit position (center vs edge), bottom-of-screen \
             life loss, ball speed clamping (min 3, max 8).",
        ),
        WorkItem::new(
            "brick_layout",
            "Brick Grid & HP System",
            "Create brick grid (11 columns x 5 rows) with multi-hit HP system (1-5 HP). Color-code by HP: \
             yellow=1HP, green=2HP, blue=3HP, orange=4HP, red=5HP. Display HP visually via text or color. \
             Support random powerup loot from bricks.",
        ),
        WorkItem::new(
            "collision_detection",
            "Ball-Brick Collision Detection",
            "Implement precise ball-brick collision detection with spatial hashing. On hit: decrease brick HP, \
             handle ball deflection (top/bottom vs left/right), award points (10 * maxHP), trigger destruction \
             when HP=0, spawn powerup drops with random type selection.",
        ),
        // Audio System (2 tasks)
        WorkItem::new(
            "background_music",
            "Background Music & Audio System",
            "Implement Atari 2600-style chiptune background music using Web Audio API oscillators (square & triangle waves). \
             Create looping melody that starts on game start, loops continuously, supports pause/resume, \
             mute button control, and volume slider.",
        ),
        WorkItem::new(
            "collision_sfx",
            "Sound Effects System",
            "Implement distinct Web Audio API sound effects: ball-brick collision (high pitched blip, 100-200ms), \
             ball-paddle collision (medium thud, 150-250ms), ball-wall bounce (low click, 50-100ms), \
             powerup pickup sound, life earned sound, and level complete fanfare.",
        ),
        // Powerup System (3 tasks)
        WorkItem::new(
            "powerups_basic",
            "Basic Powerups System",
            "Implement 3 basic powerups dropping from destroyed bricks: paddle extension (extends width 20%), \
             speed boost (slows all balls to 50% speed for 30 seconds), projectile system (activates missile \
             firing with 100 shots, 4 damage per hit). Powerups fall, have collision detection with paddle.",
        ),
        WorkItem::new(
            "advanced_powerups",
            "Advanced Powerups",
            "Implement 4 advanced powerups: lava balls (balls destroy bricks on contact for 30 seconds, \
             yellow/orange trail), bomb mode (balls become bombs, destroy bricks in 5 impacts, black with \
             impact counter), growth (balls grow 50% larger, white border), mushroom (1UP award, red, triggers \
             life earned animation).",
        ),
        WorkItem::new(
            "multiball_powerup",
            "Multiball Powerup",
            "Implement multiball powerup (purple) that spawns 10 new balls at paddle position with varied angles. \
             New balls have 50% speed of current balls, behave identically to main ball (physics, collision, powerups). \
             Game continues with all active balls until all lost.",
        ),
        // Visual Effects (3 tasks)
        WorkItem::new(
            "particle_effects",
            "Particle Effects System",
            "Implement 3 particle systems: fire particles (brick destruction bursts, radial spread, decay over time), \
             paddle jet particles (level complete animation, upward spray from paddle wings), 1UP text displays \
             (floating score notifications with fade-out). Support particle physics (velocity, gravity, color, alpha).",
        ),
        WorkItem::new(
            "paddle_animation",
            "Paddle 3D Animation & Level Complete",
            "Implement 3D paddle screw effect animation on level complete: paddle flies upward with rotating \
             wings (4 full rotations), squash/stretch effect, glowing blue appearance. Add wing cannons visual \
             when projectiles active. Smooth animation over 3 seconds.",
        ),
        WorkItem::new(
            "projectile_missiles",
            "Projectile & Missile System",
            "Implement projectile firing from paddle (space bar while projectile powerup active). Projectiles \
             travel upward, have smoke trail particle effects, deal 1 damage per brick hit, can penetrate \
             multiple bricks. Draw missile cannons on paddle when active. Support up to 100 shots per powerup.",
        ),
        // Advanced Mechanics (4 tasks)
        WorkItem::new(
            "level_system",
            "Level Progression & Patterns",
            "Implement level system with 10+ procedural brick patterns: level 1=classic grid, level 2+=pyramid, \
             diamond, checkerboard, stripe, wave, spiral, hourglass, cross, rings, random patterns. Use seeded \
             RNG for deterministic layouts. Increment level on all bricks cleared, award 1UP on level complete.",
        ),
        WorkItem::new(
            "brick_difficulty_scaling",
            "Brick HP & Difficulty Scaling",
            "Implement dynamic brick HP scaling by level (max HP increases with level, variable 1-5). Scale \
             brick HP based on row position (top=harder). Adjust powerup drop chances by level (reduce at high \
             levels). Increase brick density with level. Support seeded random for reproducible difficulty curves.",
        ),
        WorkItem::new(
            "mobile_controls",
            "Mobile Touch & Responsive Design",
            "Implement touch/swipe controls for mobile: touch-to-aim paddle movement, swipe for rapid movement, \
             buttons for fire (spacebar equivalent). Detect mobile device and show touch UI. Implement responsive \
             canvas resizing on window change. Support both portrait and landscape orientations.",
        ),
        WorkItem::new(
            "scoring_persistence",
            "Scoring, Lives System & Level Persistence",
            "Implement score tracking with point awards (brick=10*HP, powerup=100-500). Implement lives system \
             (start with 3, lose 1 on ball lost). Automatic 1UP awards at score milestones (every 2500 points). \
             Persist level progress across lives. Display all stats in HUD (score, level, lives, powerup status).",
        ),
    ];

    println!("\n📋 Task Pool: {} items", tasks.len());
    println!("   Core Mechanics (6), Audio (2), Powerups (3), Effects (3), Advanced (4)\n");

    // ── Shared Tools Setup ──────────────────────────────────────────────────────
    // All agents share access to a comprehensive toolkit including:
    // - Memory: For task pool coordination (teams:<pool_id>:*)
    // - Custom Tools: write_game_file for saving game to disk

    let memory = Arc::new(Memory::new());
    let memory_protocol = Arc::new(MemoryProtocol::new(memory.clone()));

    // Set up custom tools (write_game_file)
    let custom_protocol = Arc::new(CustomToolProtocol::new());
    custom_protocol
        .register_tool(
            ToolMetadata::new("write_game_file", "Write the complete game HTML to disk")
                .with_parameter(
                    ToolParameter::new("filename", ToolParameterType::String)
                        .with_description("The output filename (e.g., 'breakout_game_agent_teams.html')"),
                )
                .with_parameter(
                    ToolParameter::new("content", ToolParameterType::String)
                        .with_description("The complete HTML document with inline CSS and JavaScript"),
                ),
            Arc::new(|params| {
                let filename = params["filename"]
                    .as_str()
                    .unwrap_or("breakout_game_agent_teams.html")
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

    // ── Agents ──────────────────────────────────────────────────────────────────

    let claude_key = std::env::var("ANTHROPIC_API_KEY").unwrap_or_else(|_| "demo-key".to_string());

    // Factory for creating Claude Haiku 4.5 clients
    let make_client = || {
        Arc::new(ClaudeClient::new_with_model_enum(
            &claude_key,
            Model::ClaudeHaiku45,
        ))
    };

    // All agents use Claude Haiku 4.5 with shared access to all tools
    let architect = Agent::new(
        "architect",
        "Architect (Claude Haiku 4.5)",
        make_client(),
    )
    .with_expertise("HTML5 structure, CSS layout, Canvas setup, responsive design")
    .with_personality("Meticulous front-end architect who produces clean, well-structured HTML/CSS.")
    .with_shared_tools(shared_registry.clone());

    let core_engineer = Agent::new(
        "core-engineer",
        "Core Engineer (Claude Haiku 4.5)",
        make_client(),
    )
    .with_expertise("JavaScript game mechanics, physics, collision detection, rendering")
    .with_personality("Seasoned game developer who writes tight, performant JavaScript.")
    .with_shared_tools(shared_registry.clone());

    let audio_engineer = Agent::new(
        "audio-engineer",
        "Audio Engineer (Claude Haiku 4.5)",
        make_client(),
    )
    .with_expertise("Web Audio API, chiptune synthesis, oscillator-based sound effects")
    .with_personality("Retro audio enthusiast who crafts authentic Atari 2600-era sounds.")
    .with_shared_tools(shared_registry.clone());

    let features_engineer = Agent::new(
        "features-engineer",
        "Features Engineer (Claude Haiku 4.5)",
        make_client(),
    )
    .with_expertise("Game powerup systems, spawn logic, timed effects, animations")
    .with_personality("Creative gameplay engineer who designs fun and balanced mechanics.")
    .with_shared_tools(shared_registry.clone());

    // ── Orchestration ───────────────────────────────────────────────────────────

    let system_context = "\
You are a specialized agent in a decentralized team building a complete Atari Breakout game \
in a single self-contained index.html file. All HTML, CSS, and JavaScript must be inline. \
Do NOT use external dependencies. Use the HTML5 Canvas API for rendering and the Web Audio API for sound.\n\n\
Your role is to autonomously discover and claim tasks from a shared task pool, work on them, and \
report completion. You have access to a shared Memory tool that stores the task pool and your team's \
coordination state.\n\n\
TASK DISCOVERY & CLAIMING PROCESS:\n\
1. Use Memory LIST teams:<pool_id>:unclaimed:* to discover available unclaimed tasks\n\
2. For each task, use GET teams:<pool_id>:unclaimed:<task_id> to read the full description\n\
3. Select a task that matches your specialty and claim it via PUT teams:<pool_id>:claimed:<task_id> <your_agent_id>\n\
4. Work on the task autonomously (e.g., generate code, design patterns, or implementations)\n\
5. When complete, PUT the result to teams:<pool_id>:completed:<task_id> with your work\n\n\
IMPORTANT: Always output the COMPLETE updated index.html incorporating ALL previous work from \
other agents (retrieved via Memory GET) plus your additions. Never output partial snippets — \
always output the full file.\n\n\
You have access to a comprehensive toolkit for coordination and development:\n\
- Memory (memory:*): Use PUT/GET/LIST to discover tasks, claim work, store designs, coordinate\n\
- Bash (bash:*): Execute shell commands for file operations, git, testing, debugging\n\
- HTTP Client (http:*): Make web requests (http_get, http_post, http_put, http_delete, http_patch)\n\
- Custom Tools (custom:write_game_file): Write the final game HTML to disk when complete\n\
\n\
Memory Task Pool Keys:\n\
  teams:<pool_id>:unclaimed:<task_id> - Discover available work\n\
  teams:<pool_id>:claimed:<task_id> - Mark task as claimed\n\
  teams:<pool_id>:completed:<task_id> - Record completed work";

    let event_handler = Arc::new(TeamsEventHandler::new());

    let mut orchestration = Orchestration::new(
        "breakout-builder-teams",
        "Breakout Game AnthropicAgentTeams",
    )
    .with_mode(OrchestrationMode::AnthropicAgentTeams {
        pool_id: "breakout-pool-1".to_string(),
        tasks: tasks.clone(),
        max_iterations: 6,
    })
    .with_system_context(system_context)
    .with_max_tokens(200_000)
    .with_event_handler(event_handler);

    orchestration.add_agent(architect)?;
    orchestration.add_agent(core_engineer)?;
    orchestration.add_agent(audio_engineer)?;
    orchestration.add_agent(features_engineer)?;

    // ── Run ─────────────────────────────────────────────────────────────────────

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
browser with no external dependencies.\n\n\
The team will autonomously discover and claim tasks from the shared task pool via Memory. \
Each agent should claim 4-5 tasks matching their specialty. Work efficiently and coordinate \
via Memory to avoid conflicts. When complete, write the final game to breakout_game.html.";

    println!("👥 Team Members (All Claude Haiku 4.5):");
    println!("  1. Architect — HTML/CSS/canvas setup and styling");
    println!("  2. Core Engineer — Physics, collision, rendering");
    println!("  3. Audio Engineer — Music and sound effects");
    println!("  4. Features Engineer — Powerups, effects, advanced mechanics");

    println!("\n🎮 Building game with decentralized task coordination...\n");

    let start = Instant::now();
    let response = orchestration.run(prompt, 1).await?;
    let elapsed = start.elapsed();

    // ── Results ─────────────────────────────────────────────────────────────────

    let minutes = elapsed.as_secs() / 60;
    let seconds = elapsed.as_secs() % 60;

    println!("\n{}", "=".repeat(80));
    println!("  AnthropicAgentTeams Results");
    println!("{}", "=".repeat(80));
    println!("  Iterations used : {}", response.round);
    println!("  All tasks done  : {}", response.is_complete);
    println!(
        "  Completion %    : {:.0}%",
        response.convergence_score.unwrap_or(0.0) * 100.0
    );
    println!("  Total tokens    : {}", response.total_tokens_used);
    println!("  Messages        : {}", response.messages.len());
    println!("  Elapsed time    : {}m {}s", minutes, seconds);
    println!("{}\n", "=".repeat(80));

    // Print message summary
    for (_i, msg) in response.messages.iter().take(10).enumerate() {
        let agent = msg.agent_name.as_deref().unwrap_or("system");
        let preview = if msg.content.len() > 80 {
            format!("{}...", &msg.content[..80])
        } else {
            msg.content.to_string()
        };
        println!("  [{}]: {}", agent, preview);
    }

    if response.messages.len() > 10 {
        println!("  ... ({} more messages)", response.messages.len() - 10);
    }

    println!("\n✅ Build complete! Check breakout_game_agent_teams.html in current directory.");

    Ok(())
}
