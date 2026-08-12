//! RALPH Orchestration Mode — Pac-Man Game Example (spec-only)
//!
//! This example demonstrates the RALPH (autonomous iterative loop) orchestration mode
//! by having multiple specialized agents collaborate to **implement from scratch** a
//! complete Pac-Man game in a single `pacman_game_ralph_grok_4_6.html` file.
//!
//! **Important:** this file is a **pure PRD / prompt harness**. It contains no game
//! source (no HTML, CSS, or JavaScript samples) — only product requirements, task
//! acceptance criteria, and orchestration wiring. Agents invent and write the entire
//! implementation. That is intentional for evaluating the coding model.
//!
//! RALPH works by repeatedly presenting agents with the same PRD task list. Agents see
//! accumulated work from previous iterations via conversation history and mark tasks
//! complete with `[TASK_COMPLETE:task_id]` markers. The loop ends when all tasks are
//! done or `max_iterations` is reached.
//!
//! ## Features
//!
//! - **PacManEventHandler**: Real-time pretty-printed event output
//! - **MentisDB durable memory** on the shared `cloudllm` chain (agent thoughts + run log)
//! - **Session Memory tool**: short-lived key/value coordination for the game page (`current_game_html`)
//! - **write_game_file**: custom tool that writes the game page to disk, session Memory, and MentisDB
//! - **xAI Grok 4.6 (native GrokClient)**: flagship coding/agentic model via xAI API
//!
//! ## Agents
//!
//! - **maze-architect**: maze layout, tiles, scoring HUD
//! - **pacman-programmer**: Pac-Man movement, dots/pellets, collision, lives
//! - **ghost-ai-engineer**: classic per-color ghost personalities (Blinky/Pinky/Inky/Clyde)
//! - **audio-vfx-designer**: chiptune audio, frightened mode, death/win polish
//!
//! ## PRD Tasks (highlights)
//!
//! **Core (1–6)** — maze, tile map, Pac-Man movement, dots/power pellets, score/lives
//! **Ghost AI (7–12)** — Blinky chase, Pinky ambush, Inky flank, Clyde scatter, modes & tunnel
//! **Polish (13–18)** — frightened/eaten states, fruit bonus, levels, audio, win/lose screens
//!
//! ## Running
//!
//! ```bash
//! export XAI_API_KEY=xai-...
//! cargo run --example pacman_game_ralph_grok_4_6
//! ```
//!
//! MentisDB chain key defaults to `cloudllm` under `mentisdbs/` (override with
//! `MENTISDB_DIR` / `MENTISDB_CHAIN_KEY`). Agents write the playable page to
//! `pacman_game_ralph_grok_4_6.html` in the current directory.

use async_trait::async_trait;
use cloudllm::clients::grok::{GrokClient, Model as GrokModel};
use cloudllm::event::{AgentEvent, EventHandler, OrchestrationEvent};
use cloudllm::tool_protocol::{ToolMetadata, ToolParameter, ToolParameterType, ToolRegistry};
use cloudllm::tool_protocols::{
    BashProtocol, CustomToolProtocol, HttpClientProtocol, MemoryProtocol,
};
use cloudllm::tools::{BashTool, HttpClient, Memory, Platform};
use cloudllm::{
    orchestration::{Orchestration, OrchestrationMode, RalphTask},
    Agent, CloudLLMConfig, MentisDb, ThoughtType,
};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::RwLock;

/// Default MentisDB chain for CloudLLM examples and project memory.
const MENTISDB_CHAIN_KEY: &str = "cloudllm";

/// Canonical playable deliverable written by agents and recovered at end-of-run.
const OUTPUT_HTML: &str = "pacman_game_ralph_grok_4_6.html";

/// Session Memory key holding the latest full game page source.
const MEMORY_GAME_KEY: &str = "current_game_html_grok_4_6";

// ── Event Handler ──────────────────────────────────────────────────────────

/// Pretty-prints agent and orchestration events in real-time for Pac-Man RALPH runs.
struct PacManEventHandler {
    start: Instant,
}

impl PacManEventHandler {
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
impl EventHandler for PacManEventHandler {
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
                                let end =
                                    s.char_indices().nth(200).map(|(i, _)| i).unwrap_or(s.len());
                                format!("{}...", &s[..end])
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

    let api_key = match std::env::var("XAI_API_KEY") {
        Ok(key) => key,
        Err(_) => {
            eprintln!("\n❌ Error: XAI_API_KEY environment variable is not set.");
            eprintln!("\nThis example requires an xAI API key for Grok 4.6.");
            eprintln!("\nTo fix this:");
            eprintln!("  1. Get your API key from https://console.x.ai/team/default/api-keys");
            eprintln!("  2. Set the environment variable:");
            eprintln!("     export XAI_API_KEY=xai-...");
            eprintln!("  3. Run the example again:");
            eprintln!("     cargo run --example pacman_game_ralph");
            eprintln!("\nModel: grok-4.6 via xAI");
            eprintln!("Expected runtime: 20-45 minutes (10 iterations × 4 agents)\n");
            std::process::exit(1);
        }
    };

    println!("\n{}", "=".repeat(80));
    println!("  RALPH Orchestration Mode — Classic Pac-Man Game Builder");
    println!("  Provider: xAI (Grok)");
    println!("  Model:    grok-4.6 (Grok 4.6, 500k ctx)");
    println!("{}\n", "=".repeat(80));

    // Never keep a stale deliverable from a previous run.
    if PathBuf::from(OUTPUT_HTML).exists() {
        std::fs::remove_file(OUTPUT_HTML)?;
        println!("🗑️  Removed existing {OUTPUT_HTML} (fresh run)");
    }

    // ── MentisDB durable memory (project chain: cloudllm) ───────────────────
    let mentisdb_dir = std::env::var("MENTISDB_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| CloudLLMConfig::default().mentisdb_dir);
    let chain_key =
        std::env::var("MENTISDB_CHAIN_KEY").unwrap_or_else(|_| MENTISDB_CHAIN_KEY.to_string());
    std::fs::create_dir_all(&mentisdb_dir)?;
    let mentisdb = Arc::new(RwLock::new(MentisDb::open_with_key(
        &mentisdb_dir,
        &chain_key,
    )?));
    {
        let mut db = mentisdb.write().await;
        db.append(
            "pacman-builder",
            ThoughtType::Plan,
            "Pac-Man RALPH run starting: four agents will implement a full classic Pac-Man page \
             from pure PRD specifications (no starter game source). Deliverable: \
             pacman_game_ralph_grok_4_6.html. Non-negotiables: fixed-rate simulation, moderate pacing, \
             Pac-Man never leaves the board, audible SFX after user input, restart without reload.",
        )?;
        db.append(
            "pacman-builder",
            ThoughtType::Constraint,
            "Playability constraints for this run: (1) fixed ~60 logic updates/sec independent of \
             display refresh; (2) moderate arcade speeds; (3) tunnel-only wrap / never disappear; \
             (4) audible pellet, power, eat-ghost, death SFX after first input; (5) space/click \
             restart without page reload.",
        )?;
    }
    println!(
        "🧠 MentisDB chain '{}' opened at {}",
        chain_key,
        mentisdb_dir.display()
    );

    // Session key/value Memory: agents coordinate the evolving game page here.
    // MentisDB holds durable run history; this store is the working buffer.
    let memory = Arc::new(Memory::new());
    let memory_protocol = Arc::new(MemoryProtocol::new(memory.clone()));
    memory.put(MEMORY_GAME_KEY.to_string(), String::new(), None);
    println!("📋 Spec-only PRD mode: agents implement the full game from requirements");
    println!("📄 Deliverable path: {OUTPUT_HTML} (must exist when the run finishes)\n");

    let memory_for_tool = memory.clone();
    let mentisdb_for_tool = mentisdb.clone();
    let chain_key_for_tool = chain_key.clone();
    let custom_protocol = Arc::new(CustomToolProtocol::new());
    custom_protocol
        .register_async_tool(
            ToolMetadata::new(
                "write_game_file",
                "MANDATORY deliverable tool. Write the COMPLETE playable game page to \
                 pacman_game_ralph_grok_4_6.html on disk, session Memory (current_game_html), and a \
                 MentisDB checkpoint. Call this every time you produce or update the game — \
                 a finished run without this file is a failed run. Content must be a full \
                 self-contained web page (not a snippet).",
            )
            .with_parameter(
                ToolParameter::new("filename", ToolParameterType::String).with_description(
                    "Ignored for path selection; the harness always writes pacman_game_ralph_grok_4_6.html",
                ),
            )
            .with_parameter(
                ToolParameter::new("content", ToolParameterType::String).with_description(
                    "The complete self-contained game page source with all features so far \
                     (must include a full document body, not an empty string)",
                ),
            ),
            Arc::new(move |params| {
                let memory_for_tool = memory_for_tool.clone();
                let mentisdb_for_tool = mentisdb_for_tool.clone();
                let chain_key_for_tool = chain_key_for_tool.clone();
                Box::pin(async move {
                    let content = normalize_page_source(params["content"].as_str().unwrap_or(""));
                    if !looks_like_game_page(&content) {
                        return Err(format!(
                            "write_game_file rejected content: need a full playable page \
                             (>=500 chars with html/doctype or game canvas/script). got {} bytes",
                            content.len()
                        )
                        .into());
                    }
                    // Always the canonical path so agents cannot scatter wrong filenames.
                    let filename = OUTPUT_HTML.to_string();
                    let bytes = content.len();
                    std::fs::write(&filename, &content)?;
                    memory_for_tool.put(MEMORY_GAME_KEY.to_string(), content.clone(), None);
                    {
                        let mut db = mentisdb_for_tool.write().await;
                        let note = format!(
                            "Game page written to '{filename}' ({bytes} bytes) on chain '{chain_key_for_tool}'. \
                             Session Memory key {MEMORY_GAME_KEY} updated for teammate agents."
                        );
                        if let Err(err) =
                            db.append("pacman-builder", ThoughtType::StateSnapshot, &note)
                        {
                            eprintln!(
                                "⚠️  MentisDB StateSnapshot append failed (chain '{}'): {err}",
                                chain_key_for_tool
                            );
                        }
                    }
                    println!("💾 write_game_file → {filename} ({bytes} bytes)");
                    Ok(cloudllm::tool_protocol::ToolResult::success(serde_json::json!({
                        "written": filename,
                        "bytes": bytes,
                        "session_memory_key": MEMORY_GAME_KEY,
                        "mentisdb_chain": chain_key_for_tool,
                    })))
                })
            }),
        )
        .await;

    #[cfg(target_os = "macos")]
    let bash_tool = Arc::new(BashTool::new(Platform::macOS));
    #[cfg(target_os = "linux")]
    let bash_tool = Arc::new(BashTool::new(Platform::Linux));
    #[cfg(not(any(target_os = "macos", target_os = "linux")))]
    let bash_tool = Arc::new(BashTool::new(Platform::Linux));

    let bash_protocol = Arc::new(BashProtocol::new(bash_tool));
    let http_client = Arc::new(HttpClient::new());
    let http_protocol = Arc::new(HttpClientProtocol::new(http_client));

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

    // ── Agents (xAI Grok 4.6) ───────────────────────────────────────────────

    let make_client = || Arc::new(GrokClient::new_with_model_enum(&api_key, GrokModel::Grok46));

    let architect = Agent::new("maze-architect", "Maze Architect", make_client())
        .with_expertise(
            "Classic Pac-Man maze layout, tile maps, corridors, tunnels, ghost house, HUD design",
        )
        .with_personality(
            "Pixel-perfect maze designer who replicates arcade proportions and clear UI.",
        )
        .with_shared_tools(shared_registry.clone())
        .with_mentisdb(mentisdb.clone());

    let programmer = Agent::new("pacman-programmer", "Pac-Man Programmer", make_client())
        .with_expertise(
            "Pac-Man movement, timing, board bounds, pellets, collisions, lives, level flow",
        )
        .with_personality(
            "Arcade systems programmer who keeps movement fair, paced, and always on the board.",
        )
        .with_shared_tools(shared_registry.clone())
        .with_mentisdb(mentisdb.clone());

    let ghost_ai = Agent::new("ghost-ai-engineer", "Ghost AI Engineer", make_client())
        .with_expertise(
            "Classic Namco ghost AI: Blinky chase, Pinky ambush, Inky flank, Clyde scatter, \
             scatter/chase cycles, frightened and eaten states, house release timing",
        )
        .with_personality(
            "Ghost-behavior historian who implements authentic per-color targeting and mode timing.",
        )
        .with_shared_tools(shared_registry.clone())
        .with_mentisdb(mentisdb.clone());

    let audio_vfx = Agent::new("audio-vfx-designer", "Audio & VFX Designer", make_client())
        .with_expertise(
            "Retro chiptune sound design and visual polish: pellet chomp, power-up, eat-ghost, \
             death, sirens, frightened flash, fruit, overlays",
        )
        .with_personality(
            "Retro arcade polish artist who ensures every pellet and ghost-eat is heard and felt.",
        )
        .with_shared_tools(shared_registry.clone())
        .with_mentisdb(mentisdb.clone());

    // ── PRD (specifications only — no implementation samples) ───────────────

    let tasks = vec![
        RalphTask::new(
            "maze_layout",
            "Classic Maze Layout & Tiles",
            "Design and implement a full classic-inspired Pac-Man maze using roughly 28 columns \
             by 31 rows of tiles, rendered at arcade-like pixel scale (about 16 pixels per tile). \
             Include solid walls, walkable corridors, a central ghost house with a door, left and \
             right side tunnels, and correct placement of ordinary dots and four power pellets. \
             Outside-maze margins and blank areas that are not corridors must be solid (not walkable). \
             The ghost-house door must block Pac-Man. Presentation should look crisp and arcade-like.",
        ),
        RalphTask::new(
            "game_states_hud",
            "Game States & HUD",
            "Support clear game states: ready, playing, paused, dying, game over, and win. \
             Always show score, level, and lives. Space pauses and unpauses during play. \
             Space or a click starts from ready and fully restarts after game over or win, \
             without requiring the player to reload the page.",
        ),
        RalphTask::new(
            "pacman_movement",
            "Pac-Man Movement & Turning",
            "Pac-Man moves on the grid with direction buffering at intersections (queued turns \
             apply when the path is free), a mouth chomp animation facing the current direction, \
             and smooth corridor travel. \
             Critical playability requirements: \
             (1) Simulation must advance at a fixed rate near sixty logic updates per second, \
             independent of display refresh rate. High-refresh displays must not make the game \
             twice as fast. \
             (2) Pac-Man travel speed must feel arcade-like and moderate — roughly one tile per \
             fraction of a second, not frantic. Avoid speeds that feel twice as fast as the arcade. \
             (3) Pac-Man must never leave the playable board or disappear. The only allowed exit \
             is wrapping through the designated side tunnels. Leaving the board on non-tunnel edges \
             must be impossible. Walls and the ghost-house door always block him. If he somehow \
             ends on a wall, return him to the starting spawn so he stays visible. \
             (4) Turns and wall collision must not let him clip through corners.",
        ),
        RalphTask::new(
            "dots_power_pellets",
            "Dots & Power Pellets",
            "Eating a dot scores ten points; eating a power pellet scores fifty. Track remaining \
             dots so the level can clear. A power pellet puts all eligible ghosts into frightened \
             mode. Each ordinary pellet must play an audible chomp sound; each power pellet must \
             play a distinct power-up sound. Silent pellet collection fails this requirement.",
        ),
        RalphTask::new(
            "lives_death",
            "Lives & Death Flow",
            "Colliding with a non-frightened ghost costs a life, plays a death sound, resets \
             actors to spawn (Pac-Man always reappears on the board — never invisible or off-map), \
             shows a short ready period, then continues. At zero lives, show game over with clear \
             instructions that space or click restarts. Restart must not require reloading the page. \
             A short death animation is optional but welcome.",
        ),
        RalphTask::new(
            "level_progression",
            "Level Progression",
            "When every dot is cleared, either advance the level (reset pellets, increase level \
             number, slightly raise ghost speed by a small percentage per level while staying \
             within the fair speed caps) or show a win state. Support multi-level replay. \
             Space or click continues or restarts after a win.",
        ),
        RalphTask::new(
            "ghost_blinky",
            "Blinky (Red) — Direct Chase",
            "Blinky targets Pac-Man's current tile while chasing. In scatter mode he targets the \
             top-right corner. He reverses direction when global mode switches. He leaves the \
             ghost house first, or starts already outside.",
        ),
        RalphTask::new(
            "ghost_pinky",
            "Pinky (Pink) — Ambush",
            "Pinky aims four tiles ahead of Pac-Man's facing direction. Include the classic \
             upward-facing quirk that also shifts the aim left by four tiles. In scatter mode \
             he targets the top-left corner.",
        ),
        RalphTask::new(
            "ghost_inky",
            "Inky (Cyan) — Flank with Blinky",
            "Inky uses a pivot two tiles ahead of Pac-Man, then doubles the vector from Blinky's \
             tile through that pivot. In scatter mode he targets the bottom-right corner. \
             Release him from the house after a delay following Blinky and Pinky.",
        ),
        RalphTask::new(
            "ghost_clyde",
            "Clyde (Orange) — Shy Chase/Scatter",
            "If Clyde's Euclidean distance to Pac-Man is greater than eight tiles, chase Pac-Man's \
             tile; otherwise head for the bottom-left scatter corner. He is released from the \
             house last.",
        ),
        RalphTask::new(
            "ghost_modes",
            "Scatter/Chase Mode Cycle",
            "Alternate global scatter and chase with arcade-like timings (for example about seven \
             seconds scatter, twenty chase, seven scatter, twenty chase, five scatter, then chase \
             for the rest of the level). Ghosts reverse when the global mode changes. Each ghost \
             has states covering house, scatter, chase, frightened, and eaten.",
        ),
        RalphTask::new(
            "frightened_eaten",
            "Frightened & Eaten States",
            "After a power pellet, ghosts turn blue (flash near the end of the timer), move more \
             slowly, reverse, and can be eaten for escalating points (two hundred, four hundred, \
             and so on). Eating a ghost plays a clear eat-ghost sound. Eaten ghosts show eyes only, \
             return home to the house, then respawn after a short house timer.",
        ),
        RalphTask::new(
            "ghost_pathfinding",
            "Ghost Pathfinding Rules",
            "At the center of a tile, each ghost chooses among legal exits (no reverse except \
             special cases such as mode switch or eaten return), minimizing Euclidean distance to \
             its target tile. Respect walls and door rules: enter the house only when eaten; leave \
             through the door when exiting the house. Ghost base speed must stay fair and slower \
             than a frantic chase — comparable to classic arcade pacing, not much faster.",
        ),
        RalphTask::new(
            "tunnel_slowdown",
            "Tunnel Behavior",
            "Pac-Man and ghosts may wrap only through the designated horizontal side tunnels. \
             Horizontal movement off the board on non-tunnel rows must be blocked as if by a wall; \
             vertical off-board is always blocked. Do not treat every off-edge sample as a wrap to \
             the opposite side of the maze (that causes characters to leave non-tunnel edges and \
             vanish). Optionally slow ghosts while they are inside tunnels. Pathfinding must remain \
             stable across a valid tunnel wrap.",
        ),
        RalphTask::new(
            "fruit_bonus",
            "Fruit Bonus",
            "Spawn a fruit inside the maze at classic-ish remaining-dot thresholds. Award bonus \
             points on pickup with an optional fruit sound. Fruit disappears after a timer. \
             Draw a distinct fruit appearance per level.",
        ),
        RalphTask::new(
            "audio_chiptune",
            "Chiptune Audio",
            "Provide retro-style sound effects that are actually audible after the first key or \
             click (browsers block sound until user interaction — unlock audio on that interaction \
             and keep it working for the rest of the session). \
             Required sounds: ordinary pellet chomp, distinct power-pellet cue, eat-ghost sting, \
             and death. Optional: short start jingle and chase or frightened siren. \
             Failures: a game that never produces sound after input, or pellet and ghost eats \
             that stay silent. If the environment blocks audio entirely, fail soft without crashing.",
        ),
        RalphTask::new(
            "vfx_polish",
            "Visual Polish",
            "Ghosts have recognizable bodies with eyes looking in their move direction. Frightened \
             mode flashes near expiry. Show brief score popups when eating ghosts. Ready, game over, \
             and win overlays are clear and readable. Optional on-screen direction buttons for touch.",
        ),
        RalphTask::new(
            "balance_feel",
            "Game Feel & Balance",
            "Tune speeds, frightened duration by level, house release timers, and collision sizes so \
             the game is fair and fun. Movement must stay moderate under fixed-rate simulation \
             (high-refresh displays must not accelerate gameplay). Pac-Man never leaves the maze or \
             vanishes. Space or click restarts after death or game over. All four ghosts clearly \
             show distinct personalities. Audio remains working after polish passes.",
        ),
    ];

    // Full product brief given to every agent. Pure requirements language — no sample markup
    // or implementation snippets. Agents invent the technology choices within the constraints.
    let system_context = "\
You are on a multi-agent team building a complete classic Pac-Man game from a product \
specification only. There is no starter implementation. You must design and implement the \
entire playable game yourselves and deliver it as one self-contained web page file named \
pacman_game_ralph_grok_4_6.html (all presentation and behavior inline; no external libraries, fonts, \
or assets).\n\n\
\
## Memory architecture\n\
- Durable project memory: MentisDB chain 'cloudllm' is attached to every agent for harness-level \
run history. The orchestration records plan, constraints, file-write snapshots, and a final \
summary automatically when you call write_game_file and when the run ends.\n\
- Session working buffer: Memory tool key current_game_html holds the latest full game page \
so teammates can read/modify/write within this run. Prefer this for coordinating the page itself.\n\
- write_game_file always updates disk, session Memory, and a MentisDB snapshot note.\n\n\
\
## Deliverable (non-negotiable)\n\
A single offline-playable page named pacman_game_ralph_grok_4_6.html that a human can open in a browser \
and play immediately. Every productive turn MUST call write_game_file with the complete page. \
Also keep the same content in session Memory under current_game_html. A run that ends without \
a valid game page on disk is a failed run.\n\n\
\
## Product vision\n\
A faithful, fair, classic Pac-Man experience: maze, pellets, lives, levels, four distinctly \
behaved ghosts, power pellets that reverse the hunt, optional fruit bonus, retro sound, and \
polished presentation.\n\n\
\
## Ghost personalities (must be visibly distinct)\n\
- Blinky (red): chases Pac-Man's current tile; scatters to the top-right; leaves the house first.\n\
- Pinky (pink): ambushes four tiles ahead of Pac-Man's facing; classic upward aim also shifts \
left four tiles; scatters to the top-left.\n\
- Inky (cyan): flanks using Blinky's position and a point two tiles ahead of Pac-Man; scatters \
to the bottom-right; delayed house release.\n\
- Clyde (orange): chases when farther than eight tiles from Pac-Man, otherwise heads to the \
bottom-left scatter corner; released last.\n\
- Global scatter and chase alternate on an arcade-like schedule; power pellets frighten ghosts; \
eaten ghosts return home as eyes and respawn.\n\n\
\
## Non-negotiable playability requirements\n\
Prior builds failed when these were ignored. Your build must satisfy all of them:\n\
1. Timing — Game logic advances at a fixed rate near sixty updates per second. Visual refresh \
may be higher, but logic must not run once per display frame alone, or high-refresh screens make \
everything far too fast.\n\
2. Pacing — Pac-Man and ghosts move at moderate arcade-like speeds. Gameplay must not feel \
frantic or twice as fast as the classic game.\n\
3. Board safety — Pac-Man never falls off the map or disappears. The only wrap is through the \
horizontal side tunnels. Non-tunnel edges act as walls. Blank outside-maze regions are solid. \
The ghost-house door blocks Pac-Man. Pac-Man remains visible on the board at all times; if \
needed, return him to the start spawn after death or a bad position.\n\
4. Restart — After death, game over, or win, the player restarts with space or click without \
reloading the page.\n\
5. Audio — After the first key or click, the game produces audible sound: pellet chomp, power \
pellet, eating a ghost, and death. Unlock browser audio on user interaction; a silent game after \
input is a failed deliverable.\n\n\
\
## Workflow\n\
1. Read Memory key current_game_html (it may be empty on the first turn — then create the full page).\n\
2. Implement or extend the complete page for your assigned PRD tasks.\n\
3. Write the entire page back with write_game_file (disk and Memory).\n\
4. Mark finished tasks with markers of the form [TASK_COMPLETE:task_id].\n\n\
\
## Collaboration rules\n\
- Always read current_game_html before editing so you build on teammates' work when present.\n\
- Always write the complete page via write_game_file — never end a turn with only prose or a snippet.\n\
- Never only describe what you would do — the playable page must land on disk through write_game_file.\n\
- Do not remove teammates' working features when adding yours.\n\
- No external network dependencies for gameplay assets.\n\
- If current_game_html is empty, you are responsible for creating the first full page immediately.\n\n\
\
## Tools\n\
- Memory: read, write, and list session keys (especially current_game_html).\n\
- write_game_file: write the complete game page to pacman_game_ralph_grok_4_6.html and session Memory; \
the harness also records a MentisDB snapshot on the project chain.\n\
- Shell tools if needed for local checks.\n";

    let event_handler = Arc::new(PacManEventHandler::new());

    let mut orchestration =
        Orchestration::new("pacman-builder", "Pac-Man Game RALPH Orchestration")
            .with_mode(OrchestrationMode::Ralph {
                tasks,
                max_iterations: 10,
            })
            .with_system_context(system_context)
            // Grok 4.6 supports ~500k context on xAI; apply via
            // Orchestration::with_max_tokens so add_agent sets each LLMSession budget.
            .with_max_tokens(500_000)
            .with_event_handler(event_handler);

    orchestration.add_agent(architect)?;
    orchestration.add_agent(programmer)?;
    orchestration.add_agent(ghost_ai)?;
    orchestration.add_agent(audio_vfx)?;

    let prompt = "\
Build a complete classic Pac-Man game from the product requirements alone. There is no starter \
code: the team authors the entire playable page.\n\n\
Scope: classic maze with tunnels and ghost house; Pac-Man grid movement with buffered turns; \
dots and power pellets; score, lives, and levels; four ghosts with authentic personalities \
(Blinky direct chase, Pinky ambush, Inky flank via Blinky, Clyde shy eight-tile rule); scatter \
and chase cycling; frightened mode and eaten return-to-house; fruit bonus; retro sound; polished \
presentation. One self-contained page, no external dependencies.\n\n\
Must-pass quality bar: fixed-rate simulation so high-refresh screens do not speed up the game; \
moderate arcade pacing; Pac-Man never leaves the board or disappears (tunnel wrap only); \
outside blanks and the house door are solid for Pac-Man; space or click restarts after game over \
without reloading; audible sounds for pellets, power pellets, eating ghosts, and death after the \
first user input.\n\n\
Every turn that advances the game MUST call write_game_file with the complete page so \
pacman_game_ralph_grok_4_6.html exists on disk. Coordinate through Memory key current_game_html. \
Complete as many PRD tasks as you can each turn. Leaving no playable file is unacceptable.";

    println!("Starting RALPH orchestration with 4 agents and 18 PRD tasks...\n");
    println!("Model: grok-4.6 via xAI (500k context budget)\n");

    let start = Instant::now();
    let response = orchestration.run(prompt, 1).await?;
    let elapsed = start.elapsed();

    let minutes = elapsed.as_secs() / 60;
    let seconds = elapsed.as_secs() % 60;

    println!("\n{}", "=".repeat(80));
    println!("  RALPH Results — Pac-Man");
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
            "  [{}] iter={} agent={:<22} tasks_completed={:<30} preview={}...",
            i + 1,
            iteration,
            agent,
            completed,
            &msg.content[..preview_end]
        );
    }

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
                println!(
                    "  {}: {} bytes, preview={}...",
                    key,
                    value.len(),
                    &value[..preview_end]
                );
            }
        }
    }

    // Recover / ensure the deliverable no matter how agents cooperated.
    let deliverable = ensure_game_deliverable(&memory, &response.messages)?;

    // Durable run summary on the MentisDB cloudllm chain
    {
        let mut db = mentisdb.write().await;
        let summary = format!(
            "Pac-Man RALPH run finished: iterations={}, all_tasks_done={}, tokens={}, elapsed={}m{}s, \
             messages={}, deliverable_bytes={}, output={}. MentisDB chain: {}.",
            response.round,
            response.is_complete,
            response.total_tokens_used,
            minutes,
            seconds,
            response.messages.len(),
            deliverable.len(),
            OUTPUT_HTML,
            chain_key
        );
        let thought_type = if response.is_complete {
            ThoughtType::TaskComplete
        } else {
            ThoughtType::Summary
        };
        if let Err(err) = db.append("pacman-builder", thought_type, &summary) {
            eprintln!("⚠️  Failed to append MentisDB run summary: {err}");
        } else {
            println!(
                "🧠 MentisDB run summary appended to chain '{}' ({} thoughts total)",
                chain_key,
                db.thoughts().len()
            );
        }
    }

    println!(
        "\n✅ Deliverable ready: {OUTPUT_HTML} ({} bytes)",
        deliverable.len()
    );
    println!("Open {OUTPUT_HTML} in a browser to play!");

    Ok(())
}

/// Unescape common tool-transport artifacts and strip outer code fences if present.
fn normalize_page_source(raw: &str) -> String {
    let mut s = raw
        .replace("\\n", "\n")
        .replace("\\t", "\t")
        .replace("\\\"", "\"")
        .replace("\\r", "");
    s = s.trim().to_string();

    // ```html ... ``` or ``` ... ```
    if s.starts_with("```") {
        if let Some(rest) = s.strip_prefix("```") {
            let rest = rest
                .strip_prefix("html")
                .or_else(|| rest.strip_prefix("HTML"))
                .unwrap_or(rest)
                .trim_start_matches(['\n', '\r', ' ']);
            if let Some(end) = rest.rfind("```") {
                s = rest[..end].trim().to_string();
            } else {
                s = rest.trim().to_string();
            }
        }
    }
    s
}

/// Heuristic: does this look like a real game page agents should have produced?
fn looks_like_game_page(content: &str) -> bool {
    if content.len() < 500 {
        return false;
    }
    let lower = content.to_lowercase();
    let has_doc = lower.contains("<!doctype") || lower.contains("<html");
    let has_game_surface = lower.contains("<canvas")
        || lower.contains("getcontext(")
        || lower.contains("pac-man")
        || lower.contains("pacman")
        || (lower.contains("<script") && lower.contains("requestanimationframe"));
    // Prefer full documents; still accept a large script-heavy page that is clearly the game.
    (has_doc && has_game_surface)
        || (has_game_surface && content.len() > 2_000 && lower.contains("<script"))
}

/// Extract the largest plausible HTML document from free-form agent text.
fn extract_html(text: &str) -> String {
    let normalized = normalize_page_source(text);
    let lower = normalized.to_lowercase();
    let start = lower
        .find("<!doctype")
        .or_else(|| lower.find("<html"))
        .unwrap_or(0);
    let end = lower
        .rfind("</html>")
        .map(|i| i + "</html>".len())
        .unwrap_or(normalized.len());
    if start >= end {
        return String::new();
    }
    normalized[start..end].to_string()
}

/// Ensure `pacman_game_ralph_grok_4_6.html` exists with a valid game page.
///
/// Recovery order:
/// 1. Disk file already written by `write_game_file` during the run
/// 2. Session Memory `current_game_html`
/// 3. Largest HTML document extractable from agent messages
///
/// Returns the final page bytes, or an error if nothing usable was produced.
fn ensure_game_deliverable(
    memory: &Memory,
    messages: &[cloudllm::orchestration::OrchestrationMessage],
) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
    // 1) Prefer a valid on-disk file produced mid-run.
    if let Ok(disk) = std::fs::read_to_string(OUTPUT_HTML) {
        let disk = normalize_page_source(&disk);
        if looks_like_game_page(&disk) {
            println!(
                "\n✅ Using on-disk {OUTPUT_HTML} from write_game_file ({} bytes)",
                disk.len()
            );
            return Ok(disk);
        }
        // Stale/partial — remove so we don't leave a bad file if recovery fails later.
        let _ = std::fs::remove_file(OUTPUT_HTML);
    }

    // 2) Session Memory working buffer.
    if let Some((mem_html, _)) = memory.get(MEMORY_GAME_KEY, false) {
        let page = normalize_page_source(&mem_html);
        if looks_like_game_page(&page) {
            std::fs::write(OUTPUT_HTML, &page)?;
            println!(
                "\n✅ Recovered {OUTPUT_HTML} from session Memory ({} bytes)",
                page.len()
            );
            return Ok(page);
        }
    }

    // 3) Scan agent messages newest-first for embedded documents.
    let mut best: Option<String> = None;
    for msg in messages.iter().rev() {
        let html = extract_html(&msg.content);
        if looks_like_game_page(&html) {
            let take = match &best {
                None => true,
                Some(prev) => html.len() > prev.len(),
            };
            if take {
                best = Some(html);
            }
        }
    }
    if let Some(page) = best {
        std::fs::write(OUTPUT_HTML, &page)?;
        println!(
            "\n✅ Recovered {OUTPUT_HTML} from agent messages ({} bytes)",
            page.len()
        );
        return Ok(page);
    }

    // 4) Hard failure — do not pretend the game exists.
    let mem_len = memory
        .get(MEMORY_GAME_KEY, false)
        .map(|(v, _)| v.len())
        .unwrap_or(0);
    Err(format!(
        "FATAL: {OUTPUT_HTML} was not produced.\n\
         Agents never wrote a valid full game page via write_game_file, Memory, or messages.\n\
         session Memory {MEMORY_GAME_KEY}: {mem_len} bytes; messages scanned: {}.\n\
         Re-run and ensure every agent turn ends with write_game_file(content=<full page>).",
        messages.len()
    )
    .into())
}
