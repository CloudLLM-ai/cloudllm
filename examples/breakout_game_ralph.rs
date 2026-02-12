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
use cloudllm::tool_protocols::{
    BashProtocol, CustomToolProtocol, HttpClientProtocol, MemoryProtocol,
};
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
                result,
                ..
            } => {
                if *success {
                    let result_preview = result
                        .as_ref()
                        .map(|r| {
                            let s = serde_json::to_string(r).unwrap_or_default();
                            if s.len() > 200 {
                                format!("{}...", &s[..200])
                            } else {
                                s
                            }
                        })
                        .unwrap_or_default();
                    println!(
                        "  [{}]    {} tool '{}' succeeded → {}",
                        self.elapsed_str(),
                        agent_name,
                        tool_name,
                        result_preview
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

    let api_key = match std::env::var("ANTHROPIC_API_KEY") {
        Ok(key) => key,
        Err(_) => {
            eprintln!("\n❌ Error: ANTHROPIC_API_KEY environment variable is not set.");
            eprintln!("\nThis example requires a valid Anthropic API key to run Claude agents.");
            eprintln!("\nTo fix this:");
            eprintln!("  1. Get your API key from https://console.anthropic.com/");
            eprintln!("  2. Set the environment variable:");
            eprintln!("     export ANTHROPIC_API_KEY=your-actual-key-here");
            eprintln!("  3. Run the example again:");
            eprintln!("     cargo run --example breakout_game_ralph");
            eprintln!("\nExpected runtime: 30-50 minutes (10 iterations × 4 agents × 2-3 min per LLM call)");
            eprintln!("Expected cost: $5.00-$8.00 (Claude Haiku 4.5 is cost-effective)\n");
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

    // ── Seed starter HTML skeleton ─────────────────────────────────────────
    let starter_html = r#"<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Atari Breakout - RALPH Edition</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { background: #000; display: flex; justify-content: center; align-items: center; min-height: 100vh; font-family: 'Courier New', monospace; color: #fff; }
        #gameContainer { text-align: center; }
        canvas { border: 2px solid #333; display: block; margin: 0 auto; background: #111; }
        #hud { margin-top: 10px; font-size: 14px; }
    </style>
</head>
<body>
    <div id="gameContainer">
        <canvas id="gameCanvas" width="800" height="600"></canvas>
        <div id="hud">SCORE: 0 | LEVEL: 1 | LIVES: 3</div>
    </div>
    <script>
        // === GAME STATE ===
        const canvas = document.getElementById('gameCanvas');
        const ctx = canvas.getContext('2d');
        const STATES = { MENU: 0, PLAYING: 1, PAUSED: 2, GAME_OVER: 3, LEVEL_COMPLETE: 4 };
        let gameState = STATES.MENU;
        let score = 0, lives = 3, level = 1;

        // === PADDLE ===
        const paddle = { x: 350, y: 560, width: 100, height: 12, speed: 7, color: '#4488ff' };
        let keys = {};
        document.addEventListener('keydown', e => keys[e.key] = true);
        document.addEventListener('keyup', e => keys[e.key] = false);

        // === BALL ===
        let balls = [{ x: 400, y: 300, dx: 4, dy: -4, radius: 6, color: '#fff' }];

        // === BRICKS ===
        const BRICK_COLORS = { 1: '#ffff00', 2: '#00ff00', 3: '#4488ff', 4: '#ff8800', 5: '#ff0000' };
        let bricks = [];
        function initBricks() {
            bricks = [];
            for (let r = 0; r < 5; r++) {
                for (let c = 0; c < 11; c++) {
                    let hp = Math.min(5, r + 1);
                    bricks.push({ x: 10 + c * 71, y: 50 + r * 28, width: 65, height: 22, hp: hp, maxHp: hp, alive: true });
                }
            }
        }
        initBricks();

        // === GAME LOOP ===
        function update() {
            if (gameState !== STATES.PLAYING) return;
            if (keys['ArrowLeft'] || keys['a']) paddle.x = Math.max(0, paddle.x - paddle.speed);
            if (keys['ArrowRight'] || keys['d']) paddle.x = Math.min(canvas.width - paddle.width, paddle.x + paddle.speed);
            for (let ball of balls) {
                ball.x += ball.dx; ball.y += ball.dy;
                if (ball.x - ball.radius < 0 || ball.x + ball.radius > canvas.width) ball.dx *= -1;
                if (ball.y - ball.radius < 0) ball.dy *= -1;
                if (ball.y + ball.radius > canvas.height) { lives--; ball.x = 400; ball.y = 300; ball.dy = -4; if (lives <= 0) gameState = STATES.GAME_OVER; }
                if (ball.dy > 0 && ball.y + ball.radius >= paddle.y && ball.x >= paddle.x && ball.x <= paddle.x + paddle.width) {
                    ball.dy *= -1; let hitPos = (ball.x - paddle.x) / paddle.width; ball.dx = 8 * (hitPos - 0.5);
                }
                for (let brick of bricks) {
                    if (!brick.alive) continue;
                    if (ball.x + ball.radius > brick.x && ball.x - ball.radius < brick.x + brick.width &&
                        ball.y + ball.radius > brick.y && ball.y - ball.radius < brick.y + brick.height) {
                        ball.dy *= -1; brick.hp--; score += 10;
                        if (brick.hp <= 0) brick.alive = false;
                    }
                }
            }
            if (bricks.every(b => !b.alive)) { level++; initBricks(); gameState = STATES.LEVEL_COMPLETE; }
            document.getElementById('hud').textContent = `SCORE: ${score} | LEVEL: ${level} | LIVES: ${lives}`;
        }

        function draw() {
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            if (gameState === STATES.MENU) { ctx.fillStyle = '#fff'; ctx.font = '36px Courier New'; ctx.fillText('ATARI BREAKOUT', 240, 280); ctx.font = '18px Courier New'; ctx.fillText('Click or press SPACE to start', 230, 330); return; }
            if (gameState === STATES.GAME_OVER) { ctx.fillStyle = '#f00'; ctx.font = '48px Courier New'; ctx.fillText('GAME OVER', 240, 300); ctx.font = '18px Courier New'; ctx.fillStyle = '#fff'; ctx.fillText(`Final Score: ${score}`, 310, 350); return; }
            if (gameState === STATES.LEVEL_COMPLETE) { ctx.fillStyle = '#0f0'; ctx.font = '36px Courier New'; ctx.fillText(`LEVEL ${level} COMPLETE!`, 220, 300); ctx.font = '18px Courier New'; ctx.fillStyle = '#fff'; ctx.fillText('Click or press SPACE to continue', 220, 350); return; }
            ctx.fillStyle = paddle.color; ctx.fillRect(paddle.x, paddle.y, paddle.width, paddle.height);
            for (let ball of balls) { ctx.beginPath(); ctx.arc(ball.x, ball.y, ball.radius, 0, Math.PI * 2); ctx.fillStyle = ball.color; ctx.fill(); }
            for (let brick of bricks) { if (!brick.alive) continue; ctx.fillStyle = BRICK_COLORS[brick.hp] || '#fff'; ctx.fillRect(brick.x, brick.y, brick.width, brick.height); ctx.strokeStyle = '#333'; ctx.strokeRect(brick.x, brick.y, brick.width, brick.height); }
        }

        function gameLoop() { update(); draw(); requestAnimationFrame(gameLoop); }
        document.addEventListener('click', () => { if (gameState === STATES.MENU || gameState === STATES.LEVEL_COMPLETE) gameState = STATES.PLAYING; });
        document.addEventListener('keydown', e => { if (e.code === 'Space') { if (gameState === STATES.MENU || gameState === STATES.LEVEL_COMPLETE) gameState = STATES.PLAYING; else if (gameState === STATES.PLAYING) gameState = STATES.PAUSED; else if (gameState === STATES.PAUSED) gameState = STATES.PLAYING; }});
        gameLoop();
    </script>
</body>
</html>"#;

    // Write starter to disk and Memory so agents can build on it
    std::fs::write("breakout_game_ralph.html", starter_html)?;
    memory.put(
        "current_game_html".to_string(),
        starter_html.to_string(),
        None,
    );
    println!(
        "📄 Starter HTML written to disk and Memory ({} bytes)\n",
        starter_html.len()
    );

    let memory_for_tool = memory.clone();
    let custom_protocol = Arc::new(CustomToolProtocol::new());
    custom_protocol
        .register_tool(
            ToolMetadata::new(
                "write_game_file",
                "Write the COMPLETE updated game HTML to disk AND save it to Memory. \
                 ALWAYS use this after making changes so other agents can build on your work.",
            )
            .with_parameter(
                ToolParameter::new("filename", ToolParameterType::String).with_description(
                    "The filename to write (e.g. 'breakout_game_ralph.html')",
                ),
            )
            .with_parameter(
                ToolParameter::new("content", ToolParameterType::String).with_description(
                    "The COMPLETE HTML document with ALL features implemented so far",
                ),
            ),
            Arc::new(move |params| {
                let filename = params["filename"]
                    .as_str()
                    .unwrap_or("breakout_game_ralph.html")
                    .to_string();
                let content = params["content"]
                    .as_str()
                    .unwrap_or("")
                    .replace("\\n", "\n")
                    .replace("\\t", "\t")
                    .replace("\\\"", "\"");
                let bytes = content.len();
                std::fs::write(&filename, &content)?;
                memory_for_tool.put("current_game_html".to_string(), content, None);
                Ok(cloudllm::tool_protocol::ToolResult::success(
                    serde_json::json!({"written": filename, "bytes": bytes, "also_saved_to_memory": "current_game_html"}),
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
    shared_registry.add_protocol("bash", bash_protocol).await?;
    shared_registry.add_protocol("http", http_protocol).await?;
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
in a single self-contained HTML file. All HTML, CSS, and JavaScript must be inline. \
Do NOT use external dependencies. Use the HTML5 Canvas API for rendering and the Web Audio API for sound.\n\n\
\
WORKFLOW — FOLLOW THESE STEPS EXACTLY:\n\
1. READ the current game from Memory: {\"command\": \"G current_game_html\"}\n\
2. MODIFY the HTML: add your feature implementation into the existing code\n\
3. WRITE the updated file using write_game_file with the COMPLETE modified HTML\n\
4. Include [TASK_COMPLETE:task_id] markers for completed tasks\n\n\
\
CRITICAL RULES:\n\
- ALWAYS start by reading current_game_html from Memory — never start from scratch\n\
- ALWAYS write back the COMPLETE file using write_game_file after your changes\n\
- The write_game_file tool saves to BOTH disk and Memory so other agents get your changes\n\
- NEVER output partial snippets. NEVER describe what you would do. Actually write the code.\n\
- Add your code into the existing <script> block, do not replace existing features\n\
- The game already has a working skeleton with paddle, ball, bricks, and game loop\n\n\
\
TOOLS AVAILABLE:\n\
- Memory (memory): {\"command\": \"G key\"} to read, {\"command\": \"P key value\"} to write, {\"command\": \"L\"} to list\n\
- write_game_file: {\"filename\": \"breakout_game_ralph.html\", \"content\": \"<!DOCTYPE html>...\"}\n\
- Bash (bash:*): Shell commands if needed\n\n\
\
Key Memory entries:\n\
  current_game_html — THE CURRENT COMPLETE GAME HTML (read this first, write back after changes)";

    // Register the event handler on the orchestration. It will be
    // auto-propagated to each agent added via add_agent(), giving us
    // a unified stream of both OrchestrationEvents and AgentEvents.
    let event_handler = Arc::new(BreakoutEventHandler::new());

    let mut orchestration =
        Orchestration::new("breakout-builder", "Breakout Game RALPH Orchestration")
            .with_mode(OrchestrationMode::Ralph {
                tasks,
                // 10 iterations: Each iteration all 4 agents get a turn. With RALPH's sequential model
                // and shared context, 10 rounds allows agents to build incrementally on each other's work
                max_iterations: 10,
            })
            .with_system_context(system_context)
            // 250k tokens per call allows full HTML output + agent reasoning with accumulated history
            .with_max_tokens(250_000)
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

    // ── Final HTML ─────────────────────────────────────────────────────────
    // Agents write to disk incrementally via write_game_file.
    // Check Memory for the latest version first, then fall back to messages.

    let final_html = if let Some((mem_html, _)) = memory.get("current_game_html", false) {
        if mem_html.len() > 1000 && mem_html.contains("<canvas") {
            let unescaped = mem_html
                .replace("\\n", "\n")
                .replace("\\t", "\t")
                .replace("\\\"", "\"");
            std::fs::write("breakout_game_ralph.html", &unescaped)?;
            println!(
                "\n✅ Game written from Memory to breakout_game_ralph.html ({} bytes)",
                unescaped.len()
            );
            Some(unescaped)
        } else {
            None
        }
    } else {
        None
    };

    if final_html.is_none() {
        // Fallback: try extracting from agent messages
        let mut game_html: Option<String> = None;
        for msg in response.messages.iter().rev() {
            let html = extract_html(&msg.content);
            if html.len() > 1000 && (html.contains("<canvas") || html.contains("canvas")) {
                game_html = Some(html);
                break;
            }
        }
        if let Some(html) = game_html {
            std::fs::write("breakout_game_ralph.html", &html)?;
            println!(
                "\n✅ Game extracted from messages to breakout_game_ralph.html ({} bytes)",
                html.len()
            );
        } else {
            let disk_size = std::fs::metadata("breakout_game_ralph.html")
                .map(|m| m.len())
                .unwrap_or(0);
            println!(
                "\n⚠️  Agents didn't write updates via write_game_file. Starter HTML on disk ({} bytes).",
                disk_size
            );
        }
    }
    println!("Open breakout_game_ralph.html in a browser to play!");

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

    let raw = &text[start..end];

    // LLM responses often contain literal escape sequences instead of real
    // whitespace characters.  Unescape them so the HTML renders correctly.
    raw.replace("\\n", "\n")
        .replace("\\t", "\t")
        .replace("\\\"", "\"")
}
