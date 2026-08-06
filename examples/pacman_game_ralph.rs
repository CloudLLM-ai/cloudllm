//! RALPH Orchestration Mode — Pac-Man Game Example
//!
//! This example demonstrates the RALPH (autonomous iterative loop) orchestration mode
//! by having multiple specialized agents collaborate to build a complete Pac-Man
//! game in a single `pacman_game_ralph.html` file.
//!
//! RALPH works by repeatedly presenting agents with the same PRD task list. Agents see
//! accumulated work from previous iterations via conversation history and mark tasks
//! complete with `[TASK_COMPLETE:task_id]` markers. The loop ends when all tasks are
//! done or `max_iterations` is reached.
//!
//! ## Features
//!
//! - **PacManEventHandler**: Real-time pretty-printed event output
//! - **Shared Memory**: All agents share a Memory tool for coordination
//! - **write_game_file**: Custom tool that writes game files to disk
//! - **OpenRouter + DeepSeek V4 Flash**: cost-efficient coding model via OpenRouter
//!
//! ## Agents
//!
//! - **maze-architect**: HTML/CSS canvas maze layout, tiles, scoring HUD
//! - **pacman-programmer**: Pac-Man movement, dots/pellets, collision, lives
//! - **ghost-ai-engineer**: Classic per-color ghost personalities (Blinky/Pinky/Inky/Clyde)
//! - **audio-vfx-designer**: Chiptune audio, frightened mode, death/win animations
//!
//! ## PRD Tasks (highlights)
//!
//! **Core (1–6)** — canvas maze, tile map, Pac-Man movement, dots/power pellets, score/lives
//! **Ghost AI (7–12)** — Blinky chase, Pinky ambush, Inky flank, Clyde scatter, modes & tunnel
//! **Polish (13–18)** — frightened/eaten states, fruit bonus, levels, audio, win/lose screens
//!
//! ## Running
//!
//! ```bash
//! export OPENROUTER_API_KEY=sk-or-...
//! cargo run --example pacman_game_ralph
//! ```
//!
//! The example writes the assembled game to `pacman_game_ralph.html` in the current directory.

use async_trait::async_trait;
use cloudllm::clients::openrouter::{Model as OpenRouterModel, OpenRouterClient};
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

    let api_key = match std::env::var("OPENROUTER_API_KEY") {
        Ok(key) => key,
        Err(_) => {
            eprintln!("\n❌ Error: OPENROUTER_API_KEY environment variable is not set.");
            eprintln!("\nThis example requires an OpenRouter API key for DeepSeek V4 Flash.");
            eprintln!("\nTo fix this:");
            eprintln!("  1. Get your API key from https://openrouter.ai/keys");
            eprintln!("  2. Set the environment variable:");
            eprintln!("     export OPENROUTER_API_KEY=sk-or-...");
            eprintln!("  3. Run the example again:");
            eprintln!("     cargo run --example pacman_game_ralph");
            eprintln!("\nModel: deepseek/deepseek-v4-flash-0731 via OpenRouter");
            eprintln!("Expected runtime: 20-45 minutes (10 iterations × 4 agents)\n");
            std::process::exit(1);
        }
    };

    println!("\n{}", "=".repeat(80));
    println!("  RALPH Orchestration Mode — Classic Pac-Man Game Builder");
    println!("  Provider: OpenRouter");
    println!("  Model:    deepseek/deepseek-v4-flash-0731 (DeepSeek V4 Flash)");
    println!("{}\n", "=".repeat(80));

    let memory = Arc::new(Memory::new());
    let memory_protocol = Arc::new(MemoryProtocol::new(memory.clone()));

    // ── Seed starter HTML skeleton ─────────────────────────────────────────
    // A playable-ish maze shell so agents incrementally add ghosts, AI, audio.
    let starter_html = r#"<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Pac-Man — RALPH Edition</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            background: #000; color: #fff;
            font-family: 'Courier New', monospace;
            display: flex; flex-direction: column;
            align-items: center; justify-content: center;
            min-height: 100vh;
        }
        #hud {
            width: 448px; display: flex; justify-content: space-between;
            padding: 8px 4px; font-size: 16px; letter-spacing: 1px;
        }
        #hud .score { color: #fff; }
        #hud .lives { color: #ffcc00; }
        #hud .level { color: #00e5ff; }
        canvas {
            background: #000;
            border: 2px solid #2121de;
            image-rendering: pixelated;
            display: block;
        }
        #hint { margin-top: 10px; font-size: 12px; color: #888; text-align: center; }
        #overlay {
            position: absolute; width: 448px; height: 496px;
            display: flex; align-items: center; justify-content: center;
            pointer-events: none; font-size: 28px; color: #ffcc00;
            text-shadow: 0 0 8px #000;
        }
        #gameWrap { position: relative; }
    </style>
</head>
<body>
    <div id="hud">
        <span class="score">SCORE 0</span>
        <span class="level">LEVEL 1</span>
        <span class="lives">LIVES ★★★</span>
    </div>
    <div id="gameWrap">
        <canvas id="game" width="448" height="496"></canvas>
        <div id="overlay">READY!</div>
    </div>
    <div id="hint">ARROWS / WASD to move · SPACE pause · Classic ghost AI (Blinky · Pinky · Inky · Clyde)</div>
<script>
// ═══════════════════════════════════════════════════════════════════════════
// PAC-MAN SKELETON — agents extend this via RALPH (read Memory → modify → write)
// Tile size 16px. Maze is 28×31 tiles (classic arcade proportions).
// Ghost personality targets (to implement fully):
//   BLINKY (red)   — chase Pac-Man's tile directly
//   PINKY  (pink)  — ambush 4 tiles ahead of Pac-Man's facing
//   INKY   (cyan)  — flank using vector from Blinky through Pac-Man+2
//   CLYDE  (orange)— chase when far; scatter to corner when close (8 tiles)
// Modes cycle: SCATTER → CHASE → SCATTER → CHASE … ; power pellet → FRIGHTENED
// ═══════════════════════════════════════════════════════════════════════════
(() => {
  'use strict';
  const TILE = 16;
  const COLS = 28, ROWS = 31;
  const canvas = document.getElementById('game');
  const ctx = canvas.getContext('2d');
  const overlay = document.getElementById('overlay');
  const hudScore = document.querySelector('#hud .score');
  const hudLives = document.querySelector('#hud .lives');
  const hudLevel = document.querySelector('#hud .level');

  // 0=wall 1=empty 2=dot 3=power 4=ghost house door 5=tunnel
  // Simplified classic-inspired maze (agents should refine to full arcade layout)
  const RAW = [
    '############################',
    '#............##............#',
    '#.####.#####.##.#####.####.#',
    '#o####.#####.##.#####.####o#',
    '#.####.#####.##.#####.####.#',
    '#..........................#',
    '#.####.##.########.##.####.#',
    '#.####.##.########.##.####.#',
    '#......##....##....##......#',
    '######.##### ## #####.######',
    '     #.##### ## #####.#     ',
    '     #.##          ##.#     ',
    '     #.## ###--### ##.#     ',
    '######.## #      # ##.######',
    'T......=  #      #  =......T',
    '######.## #      # ##.######',
    '     #.## ######## ##.#     ',
    '     #.##          ##.#     ',
    '     #.## ######## ##.#     ',
    '######.## ######## ##.######',
    '#............##............#',
    '#.####.#####.##.#####.####.#',
    '#.####.#####.##.#####.####.#',
    '#o..##................##..o#',
    '###.##.##.########.##.##.###',
    '###.##.##.########.##.##.###',
    '#......##....##....##......#',
    '#.##########.##.##########.#',
    '#.##########.##.##########.#',
    '#..........................#',
    '############################',
  ];

  function parseMaze(rows) {
    const grid = [];
    let dots = 0;
    for (let y = 0; y < rows.length; y++) {
      const line = rows[y];
      const row = [];
      for (let x = 0; x < line.length; x++) {
        const ch = line[x];
        let t = 0;
        if (ch === '#') t = 0;
        else if (ch === '.') { t = 2; dots++; }
        else if (ch === 'o') { t = 3; dots++; }
        else if (ch === '-') t = 4;
        else if (ch === 'T' || ch === '=') t = 5;
        else t = 1;
        row.push(t);
      }
      grid.push(row);
    }
    return { grid, dots };
  }

  let { grid, dots: dotsLeft } = parseMaze(RAW);
  const DIRS = {
    LEFT:  { x: -1, y:  0 },
    RIGHT: { x:  1, y:  0 },
    UP:    { x:  0, y: -1 },
    DOWN:  { x:  0, y:  1 },
  };
  const OPP = { LEFT: 'RIGHT', RIGHT: 'LEFT', UP: 'DOWN', DOWN: 'UP' };

  const STATES = { READY: 0, PLAYING: 1, PAUSED: 2, DYING: 3, GAME_OVER: 4, WIN: 5 };
  let state = STATES.READY;
  let score = 0, lives = 3, level = 1, readyTimer = 120;
  let modeTimer = 0;
  // Classic-ish mode schedule in frames @ ~60fps: scatter 7s, chase 20s, scatter 7s, chase 20s, scatter 5s, chase forever
  const MODE_SCHEDULE = [
    { mode: 'SCATTER', frames: 7 * 60 },
    { mode: 'CHASE',   frames: 20 * 60 },
    { mode: 'SCATTER', frames: 7 * 60 },
    { mode: 'CHASE',   frames: 20 * 60 },
    { mode: 'SCATTER', frames: 5 * 60 },
    { mode: 'CHASE',   frames: 1e9 },
  ];
  let scheduleIdx = 0;
  let globalMode = 'SCATTER';
  let frightenedTimer = 0;

  const pac = {
    x: 14 * TILE, y: 23 * TILE + TILE / 2,
    dir: 'LEFT', nextDir: 'LEFT',
    speed: 1.5, mouth: 0, mouthDir: 1,
    radius: 7,
  };

  // Scatter corners (tile coords) — classic arcade corners
  const SCATTER = {
    blinky: { x: 25, y: 0 },
    pinky:  { x: 2,  y: 0 },
    inky:   { x: 27, y: 30 },
    clyde:  { x: 0,  y: 30 },
  };

  function makeGhost(name, color, tileX, tileY, scatter) {
    return {
      name, color,
      x: tileX * TILE + TILE / 2,
      y: tileY * TILE + TILE / 2,
      dir: 'LEFT',
      speed: 1.35,
      state: 'SCATTER', // SCATTER | CHASE | FRIGHTENED | EATEN | HOUSE
      scatter,
      houseTimer: name === 'blinky' ? 0 : (name === 'pinky' ? 30 : name === 'inky' ? 180 : 360),
      eaten: false,
      flash: false,
    };
  }

  let ghosts = [
    makeGhost('blinky', '#ff0000', 14, 11, SCATTER.blinky),
    makeGhost('pinky',  '#ffb8ff', 14, 14, SCATTER.pinky),
    makeGhost('inky',   '#00ffff', 12, 14, SCATTER.inky),
    makeGhost('clyde',  '#ffb852', 16, 14, SCATTER.clyde),
  ];

  const keys = {};
  addEventListener('keydown', e => {
    keys[e.key] = true;
    if (['ArrowUp','ArrowDown','ArrowLeft','ArrowRight',' '].includes(e.key)) e.preventDefault();
    if (e.key === 'ArrowLeft' || e.key === 'a') pac.nextDir = 'LEFT';
    if (e.key === 'ArrowRight' || e.key === 'd') pac.nextDir = 'RIGHT';
    if (e.key === 'ArrowUp' || e.key === 'w') pac.nextDir = 'UP';
    if (e.key === 'ArrowDown' || e.key === 's') pac.nextDir = 'DOWN';
    if (e.code === 'Space') {
      if (state === STATES.PLAYING) { state = STATES.PAUSED; overlay.textContent = 'PAUSED'; }
      else if (state === STATES.PAUSED) { state = STATES.PLAYING; overlay.textContent = ''; }
      else if (state === STATES.READY || state === STATES.GAME_OVER || state === STATES.WIN) startGame();
    }
  });
  addEventListener('keyup', e => { keys[e.key] = false; });
  canvas.addEventListener('click', () => {
    if (state === STATES.READY || state === STATES.GAME_OVER || state === STATES.WIN) startGame();
  });

  function startGame() {
    if (state === STATES.GAME_OVER || state === STATES.WIN) {
      ({ grid, dots: dotsLeft } = parseMaze(RAW));
      score = 0; lives = 3; level = 1;
    }
    resetActors();
    state = STATES.READY;
    readyTimer = 120;
    overlay.textContent = 'READY!';
    scheduleIdx = 0; modeTimer = 0; globalMode = 'SCATTER'; frightenedTimer = 0;
    updateHud();
  }

  function resetActors() {
    pac.x = 14 * TILE; pac.y = 23 * TILE + TILE / 2;
    pac.dir = 'LEFT'; pac.nextDir = 'LEFT';
    ghosts = [
      makeGhost('blinky', '#ff0000', 14, 11, SCATTER.blinky),
      makeGhost('pinky',  '#ffb8ff', 14, 14, SCATTER.pinky),
      makeGhost('inky',   '#00ffff', 12, 14, SCATTER.inky),
      makeGhost('clyde',  '#ffb852', 16, 14, SCATTER.clyde),
    ];
  }

  function updateHud() {
    hudScore.textContent = 'SCORE ' + score;
    hudLevel.textContent = 'LEVEL ' + level;
    hudLives.textContent = 'LIVES ' + '★'.repeat(Math.max(0, lives));
  }

  function tileAt(px, py) {
    let tx = Math.floor(px / TILE);
    let ty = Math.floor(py / TILE);
    // wrap tunnels
    if (tx < 0) tx = COLS - 1;
    if (tx >= COLS) tx = 0;
    if (ty < 0 || ty >= ROWS) return 0;
    return grid[ty][tx];
  }

  function isWallTile(t) { return t === 0; }

  function canMove(px, py, dir) {
    const d = DIRS[dir];
    const nx = px + d.x * pac.speed;
    const ny = py + d.y * pac.speed;
    // center-based collision against upcoming tile
    const lookX = px + d.x * (TILE / 2 + 1);
    const lookY = py + d.y * (TILE / 2 + 1);
    const t = tileAt(lookX, lookY);
    if (isWallTile(t) && t !== 4) return false;
    return true;
  }

  function centerOnTrack(entity) {
    // Snap to center of corridor for clean turns
    if (entity.dir === 'LEFT' || entity.dir === 'RIGHT') {
      entity.y = Math.round(entity.y / TILE) * TILE + TILE / 2;
    } else {
      entity.x = Math.round(entity.x / TILE) * TILE + TILE / 2;
    }
  }

  function atTileCenter(entity) {
    const cx = (Math.floor(entity.x / TILE) + 0.5) * TILE;
    const cy = (Math.floor(entity.y / TILE) + 0.5) * TILE;
    return Math.abs(entity.x - cx) < entity.speed && Math.abs(entity.y - cy) < entity.speed;
  }

  function wrapTunnel(entity) {
    if (entity.x < -TILE / 2) entity.x = COLS * TILE + TILE / 2;
    if (entity.x > COLS * TILE + TILE / 2) entity.x = -TILE / 2;
  }

  function pacTile() {
    return { x: Math.floor(pac.x / TILE), y: Math.floor(pac.y / TILE) };
  }

  // ── Ghost targeting (classic arcade rules) ───────────────────────────────
  function targetFor(g) {
    if (g.state === 'EATEN') return { x: 14, y: 14 }; // house
    if (g.state === 'FRIGHTENED') {
      // random-ish wander: pick current scatter with noise
      return {
        x: (g.scatter.x + ((Math.floor(pac.x) ^ Math.floor(g.y)) % 7)) % COLS,
        y: (g.scatter.y + ((Math.floor(pac.y) ^ Math.floor(g.x)) % 5)) % ROWS,
      };
    }
    if (g.state === 'SCATTER' || globalMode === 'SCATTER' && g.state !== 'CHASE') {
      // when globally scattering and not forced chase
      if (g.state === 'SCATTER' || globalMode === 'SCATTER') return g.scatter;
    }

    const pt = pacTile();
    const pd = DIRS[pac.dir] || DIRS.LEFT;

    if (g.name === 'blinky') {
      // Aggressive red: direct chase
      return { x: pt.x, y: pt.y };
    }
    if (g.name === 'pinky') {
      // Ambush: 4 tiles ahead of Pac-Man (classic bug: UP also shifts left 4)
      let tx = pt.x + pd.x * 4;
      let ty = pt.y + pd.y * 4;
      if (pac.dir === 'UP') tx -= 4; // original Namco overflow quirk
      return { x: tx, y: ty };
    }
    if (g.name === 'inky') {
      // Flank: vector from Blinky through (Pac + 2 ahead), doubled
      const blinky = ghosts.find(h => h.name === 'blinky');
      let pivotX = pt.x + pd.x * 2;
      let pivotY = pt.y + pd.y * 2;
      if (pac.dir === 'UP') pivotX -= 2;
      const bx = Math.floor(blinky.x / TILE);
      const by = Math.floor(blinky.y / TILE);
      return { x: pivotX * 2 - bx, y: pivotY * 2 - by };
    }
    if (g.name === 'clyde') {
      // Shy: chase if distance > 8 tiles, else scatter
      const gx = Math.floor(g.x / TILE), gy = Math.floor(g.y / TILE);
      const dist = Math.hypot(gx - pt.x, gy - pt.y);
      if (dist > 8) return { x: pt.x, y: pt.y };
      return g.scatter;
    }
    return pt;
  }

  function ghostChooseDir(g) {
    if (!atTileCenter(g) && g.state !== 'HOUSE') return;
    centerOnTrack(g);
    const gx = Math.floor(g.x / TILE), gy = Math.floor(g.y / TILE);
    const target = targetFor(g);
    const candidates = ['UP', 'LEFT', 'DOWN', 'RIGHT'].filter(dir => {
      if (dir === OPP[g.dir] && g.state !== 'EATEN') return false; // no reverse except eaten
      const d = DIRS[dir];
      const nx = gx + d.x, ny = gy + d.y;
      if (ny < 0 || ny >= ROWS) return false;
      let tx = nx;
      if (tx < 0) tx = COLS - 1;
      if (tx >= COLS) tx = 0;
      const tile = grid[ny][tx];
      if (isWallTile(tile)) return false;
      // no entering door unless eaten/leaving house
      if (tile === 4 && g.state !== 'EATEN' && g.state !== 'HOUSE') return false;
      return true;
    });
    if (!candidates.length) {
      g.dir = OPP[g.dir] || 'LEFT';
      return;
    }
    if (g.state === 'FRIGHTENED') {
      g.dir = candidates[Math.floor(Math.random() * candidates.length)];
      return;
    }
    // pick direction minimizing Euclidean distance to target (arcade tie-break: UP LEFT DOWN RIGHT)
    let best = candidates[0], bestD = Infinity;
    for (const dir of candidates) {
      const d = DIRS[dir];
      const nx = gx + d.x, ny = gy + d.y;
      const dist = (nx - target.x) * (nx - target.x) + (ny - target.y) * (ny - target.y);
      if (dist < bestD) { bestD = dist; best = dir; }
    }
    g.dir = best;
  }

  function updateModes() {
    if (frightenedTimer > 0) {
      frightenedTimer--;
      if (frightenedTimer === 0) {
        for (const g of ghosts) {
          if (g.state === 'FRIGHTENED') g.state = globalMode;
        }
      }
      return;
    }
    modeTimer++;
    const slot = MODE_SCHEDULE[scheduleIdx];
    if (modeTimer >= slot.frames) {
      modeTimer = 0;
      scheduleIdx = Math.min(scheduleIdx + 1, MODE_SCHEDULE.length - 1);
      globalMode = MODE_SCHEDULE[scheduleIdx].mode;
      for (const g of ghosts) {
        if (g.state === 'CHASE' || g.state === 'SCATTER') {
          g.state = globalMode;
          g.dir = OPP[g.dir] || g.dir; // reverse on mode switch
        }
      }
    }
  }

  function updatePac() {
    // try buffered turn at intersections
    if (pac.nextDir !== pac.dir && canMove(pac.x, pac.y, pac.nextDir)) {
      pac.dir = pac.nextDir;
      centerOnTrack(pac);
    }
    if (canMove(pac.x, pac.y, pac.dir)) {
      const d = DIRS[pac.dir];
      pac.x += d.x * pac.speed;
      pac.y += d.y * pac.speed;
      wrapTunnel(pac);
    } else {
      centerOnTrack(pac);
    }

    // eat dots
    const tx = Math.floor(pac.x / TILE), ty = Math.floor(pac.y / TILE);
    if (ty >= 0 && ty < ROWS && tx >= 0 && tx < COLS) {
      const t = grid[ty][tx];
      if (t === 2) {
        grid[ty][tx] = 1; dotsLeft--; score += 10; updateHud();
      } else if (t === 3) {
        grid[ty][tx] = 1; dotsLeft--; score += 50; updateHud();
        // frightened mode
        frightenedTimer = 6 * 60;
        for (const g of ghosts) {
          if (g.state !== 'EATEN' && g.state !== 'HOUSE') {
            g.state = 'FRIGHTENED';
            g.dir = OPP[g.dir] || g.dir;
          }
        }
      }
    }
    if (dotsLeft <= 0) {
      state = STATES.WIN;
      overlay.textContent = 'YOU WIN!';
    }
    pac.mouth += pac.mouthDir * 0.25;
    if (pac.mouth > 0.45 || pac.mouth < 0.05) pac.mouthDir *= -1;
  }

  function updateGhosts() {
    for (const g of ghosts) {
      if (g.state === 'HOUSE') {
        g.houseTimer--;
        // bob in house
        g.y = 14 * TILE + Math.sin(performance.now() / 200) * 2;
        if (g.houseTimer <= 0) {
          g.state = globalMode;
          g.x = 14 * TILE + TILE / 2;
          g.y = 11 * TILE + TILE / 2;
          g.dir = 'LEFT';
        }
        continue;
      }

      ghostChooseDir(g);
      const spd = g.state === 'FRIGHTENED' ? g.speed * 0.6
                : g.state === 'EATEN' ? g.speed * 2
                : g.speed * (1 + (level - 1) * 0.05);
      const d = DIRS[g.dir] || DIRS.LEFT;
      g.x += d.x * spd;
      g.y += d.y * spd;
      wrapTunnel(g);

      if (g.state === 'EATEN') {
        const dx = g.x - (14 * TILE + TILE / 2);
        const dy = g.y - (14 * TILE + TILE / 2);
        if (dx * dx + dy * dy < 36) {
          g.state = 'HOUSE';
          g.houseTimer = 60;
          g.x = 14 * TILE + TILE / 2;
          g.y = 14 * TILE + TILE / 2;
        }
      }

      // collision with Pac-Man
      const dist = Math.hypot(g.x - pac.x, g.y - pac.y);
      if (dist < TILE * 0.8) {
        if (g.state === 'FRIGHTENED') {
          g.state = 'EATEN';
          score += 200;
          updateHud();
        } else if (g.state !== 'EATEN' && g.state !== 'HOUSE') {
          lives--;
          updateHud();
          if (lives <= 0) {
            state = STATES.GAME_OVER;
            overlay.textContent = 'GAME OVER';
          } else {
            state = STATES.READY;
            readyTimer = 120;
            overlay.textContent = 'READY!';
            resetActors();
          }
        }
      }
    }
  }

  function drawMaze() {
    for (let y = 0; y < ROWS; y++) {
      for (let x = 0; x < COLS; x++) {
        const t = grid[y][x];
        const px = x * TILE, py = y * TILE;
        if (t === 0) {
          ctx.fillStyle = '#2121de';
          ctx.fillRect(px, py, TILE, TILE);
          ctx.fillStyle = '#000';
          ctx.fillRect(px + 2, py + 2, TILE - 4, TILE - 4);
        } else if (t === 2) {
          ctx.fillStyle = '#ffb897';
          ctx.beginPath();
          ctx.arc(px + TILE / 2, py + TILE / 2, 2, 0, Math.PI * 2);
          ctx.fill();
        } else if (t === 3) {
          ctx.fillStyle = '#ffb897';
          ctx.beginPath();
          ctx.arc(px + TILE / 2, py + TILE / 2, 5, 0, Math.PI * 2);
          ctx.fill();
        } else if (t === 4) {
          ctx.fillStyle = '#ffb8ff';
          ctx.fillRect(px, py + TILE / 2 - 1, TILE, 2);
        }
      }
    }
  }

  function drawPac() {
    const ang = { RIGHT: 0, DOWN: Math.PI / 2, LEFT: Math.PI, UP: -Math.PI / 2 }[pac.dir] || 0;
    ctx.save();
    ctx.translate(pac.x, pac.y);
    ctx.rotate(ang);
    ctx.fillStyle = '#ffff00';
    ctx.beginPath();
    ctx.moveTo(0, 0);
    ctx.arc(0, 0, pac.radius, pac.mouth, Math.PI * 2 - pac.mouth);
    ctx.closePath();
    ctx.fill();
    ctx.restore();
  }

  function drawGhost(g) {
    let color = g.color;
    if (g.state === 'FRIGHTENED') {
      color = (frightenedTimer < 120 && Math.floor(frightenedTimer / 10) % 2 === 0) ? '#fff' : '#2121de';
    } else if (g.state === 'EATEN') {
      // eyes only
      ctx.fillStyle = '#fff';
      ctx.beginPath(); ctx.arc(g.x - 3, g.y - 2, 2.5, 0, Math.PI * 2); ctx.fill();
      ctx.beginPath(); ctx.arc(g.x + 3, g.y - 2, 2.5, 0, Math.PI * 2); ctx.fill();
      ctx.fillStyle = '#00f';
      ctx.beginPath(); ctx.arc(g.x - 3, g.y - 2, 1, 0, Math.PI * 2); ctx.fill();
      ctx.beginPath(); ctx.arc(g.x + 3, g.y - 2, 1, 0, Math.PI * 2); ctx.fill();
      return;
    }
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.arc(g.x, g.y - 2, 7, Math.PI, 0);
    ctx.lineTo(g.x + 7, g.y + 6);
    for (let i = 0; i < 3; i++) {
      ctx.quadraticCurveTo(g.x + 7 - i * 4.6 - 2.3, g.y + 10, g.x + 7 - (i + 1) * 4.6, g.y + 6);
    }
    ctx.closePath();
    ctx.fill();
    // eyes
    ctx.fillStyle = '#fff';
    ctx.beginPath(); ctx.arc(g.x - 3, g.y - 3, 2.2, 0, Math.PI * 2); ctx.fill();
    ctx.beginPath(); ctx.arc(g.x + 3, g.y - 3, 2.2, 0, Math.PI * 2); ctx.fill();
    const ed = DIRS[g.dir] || DIRS.LEFT;
    ctx.fillStyle = '#00f';
    ctx.beginPath(); ctx.arc(g.x - 3 + ed.x, g.y - 3 + ed.y, 1, 0, Math.PI * 2); ctx.fill();
    ctx.beginPath(); ctx.arc(g.x + 3 + ed.x, g.y - 3 + ed.y, 1, 0, Math.PI * 2); ctx.fill();
  }

  function draw() {
    ctx.fillStyle = '#000';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    drawMaze();
    drawPac();
    for (const g of ghosts) drawGhost(g);
  }

  function tick() {
    if (state === STATES.READY) {
      readyTimer--;
      if (readyTimer <= 0) {
        state = STATES.PLAYING;
        overlay.textContent = '';
      }
    } else if (state === STATES.PLAYING) {
      updateModes();
      updatePac();
      updateGhosts();
    }
    draw();
    requestAnimationFrame(tick);
  }

  updateHud();
  startGame();
  tick();
})();
</script>
</body>
</html>"#;

    std::fs::write("pacman_game_ralph.html", starter_html)?;
    memory.put(
        "current_game_html".to_string(),
        starter_html.to_string(),
        None,
    );
    println!(
        "📄 Starter Pac-Man HTML written to disk and Memory ({} bytes)\n",
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
                    "The filename to write (e.g. 'pacman_game_ralph.html')",
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
                    .unwrap_or("pacman_game_ralph.html")
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

    // ── Agents (OpenRouter + DeepSeek V4 Flash) ─────────────────────────────

    let make_client = || {
        Arc::new(OpenRouterClient::new_with_model_enum(
            &api_key,
            OpenRouterModel::DeepSeekV4Flash,
        ))
    };

    let architect = Agent::new("maze-architect", "Maze Architect", make_client())
        .with_expertise("HTML5 canvas mazes, tile maps, classic Pac-Man layout, HUD/UI")
        .with_personality(
            "Pixel-perfect maze designer who replicates arcade proportions and clean tile rendering.",
        )
        .with_shared_tools(shared_registry.clone());

    let programmer = Agent::new("pacman-programmer", "Pac-Man Programmer", make_client())
        .with_expertise("Pac-Man movement, pellet eating, collisions, lives, level flow")
        .with_personality(
            "Arcade systems programmer who writes tight 60fps game loops and reliable collision.",
        )
        .with_shared_tools(shared_registry.clone());

    let ghost_ai = Agent::new("ghost-ai-engineer", "Ghost AI Engineer", make_client())
        .with_expertise(
            "Classic Namco ghost AI: Blinky chase, Pinky ambush, Inky flank, Clyde scatter, \
             scatter/chase mode cycles, frightened/eaten states, house release timers",
        )
        .with_personality(
            "Ghost-behavior historian who implements authentic per-color targeting rules and mode timing.",
        )
        .with_shared_tools(shared_registry.clone());

    let audio_vfx = Agent::new("audio-vfx-designer", "Audio & VFX Designer", make_client())
        .with_expertise("Web Audio chiptunes, death animation, frightened flashing, fruit bonuses")
        .with_personality(
            "Retro arcade polish artist who adds sirens, waka-waka, and satisfying feedback.",
        )
        .with_shared_tools(shared_registry.clone());

    // ── PRD Tasks ───────────────────────────────────────────────────────────

    let tasks = vec![
        RalphTask::new(
            "maze_layout",
            "Classic Maze Layout & Tiles",
            "Refine the maze to a full 28×31 classic-inspired Pac-Man layout with correct wall \
             thickness, ghost house with door, left/right tunnels, and accurate pellet placement. \
             Keep TILE=16 rendering crisp (pixelated).",
        ),
        RalphTask::new(
            "game_states_hud",
            "Game States & HUD",
            "Implement READY, PLAYING, PAUSED, DYING, GAME_OVER, WIN states with overlay text. \
             HUD shows SCORE, LEVEL, LIVES. SPACE toggles pause; click/SPACE starts from READY.",
        ),
        RalphTask::new(
            "pacman_movement",
            "Pac-Man Movement & Turning",
            "Grid-aligned movement with buffered turns at intersections, tunnel wrap, speed that \
             feels arcade-like, and mouth chomp animation tied to movement direction.",
        ),
        RalphTask::new(
            "dots_power_pellets",
            "Dots & Power Pellets",
            "Eat dots (+10) and power pellets (+50). Track remaining dots for level clear. \
             Power pellet triggers frightened mode for all eligible ghosts.",
        ),
        RalphTask::new(
            "lives_death",
            "Lives & Death Flow",
            "On ghost collision (non-frightened): lose a life, short READY reset of actors, \
             GAME OVER at 0 lives. Optional simple death animation before READY.",
        ),
        RalphTask::new(
            "level_progression",
            "Level Progression",
            "When all dots cleared: WIN or advance level (reset maze pellets, bump level, \
             slightly increase ghost speed). Support at least multi-level replay.",
        ),
        RalphTask::new(
            "ghost_blinky",
            "Blinky (Red) — Direct Chase",
            "Implement Blinky targeting: in CHASE mode target Pac-Man's current tile exactly. \
             In SCATTER target top-right corner. Reverse direction on mode switches. \
             Blinky leaves the house first (or starts outside).",
        ),
        RalphTask::new(
            "ghost_pinky",
            "Pinky (Pink) — Ambush",
            "Implement Pinky targeting: aim 4 tiles ahead of Pac-Man's facing direction. \
             Include the classic UP-direction offset quirk (also shift left 4 when Pac faces up). \
             SCATTER to top-left corner.",
        ),
        RalphTask::new(
            "ghost_inky",
            "Inky (Cyan) — Flank with Blinky",
            "Implement Inky targeting: take the tile 2 ahead of Pac-Man, then double the vector \
             from Blinky's tile through that pivot. SCATTER to bottom-right. Release after delay.",
        ),
        RalphTask::new(
            "ghost_clyde",
            "Clyde (Orange) — Shy Chase/Scatter",
            "Implement Clyde: if Euclidean distance to Pac-Man > 8 tiles, chase Pac-Man's tile; \
             otherwise target bottom-left scatter corner. House release last.",
        ),
        RalphTask::new(
            "ghost_modes",
            "Scatter/Chase Mode Cycle",
            "Global mode schedule alternating SCATTER and CHASE with arcade-like timings \
             (e.g. 7s/20s/7s/20s/5s then chase). Ghosts reverse direction when mode changes. \
             Per-ghost state machine: HOUSE, SCATTER, CHASE, FRIGHTENED, EATEN.",
        ),
        RalphTask::new(
            "frightened_eaten",
            "Frightened & Eaten States",
            "Power pellet: ghosts turn blue (flash near end), slow down, reverse, and can be eaten \
             for escalating points (200, 400, …). Eaten ghosts show eyes-only and path home to house, \
             then respawn after a short house timer.",
        ),
        RalphTask::new(
            "ghost_pathfinding",
            "Ghost Pathfinding Rules",
            "At tile centers, choose among legal exits (no reverse except special cases) minimizing \
             Euclidean distance to target tile. Respect walls and ghost-house door rules \
             (only enter when EATEN / leave when exiting HOUSE).",
        ),
        RalphTask::new(
            "tunnel_slowdown",
            "Tunnel Behavior",
            "Ghosts and Pac-Man wrap through side tunnels. Optionally slow ghosts in tunnels \
             like the arcade. Ensure pathfinding remains stable across wrap.",
        ),
        RalphTask::new(
            "fruit_bonus",
            "Fruit Bonus",
            "Spawn a fruit in the maze at classic-ish dot thresholds; award bonus points on pickup; \
             fruit disappears after a timer. Draw distinct fruit per level.",
        ),
        RalphTask::new(
            "audio_chiptune",
            "Chiptune Audio",
            "Web Audio API: start jingle, waka-ish chomp (or munch pulse), siren that pitches up \
             as dots decrease, frightened siren, eat-ghost blip, death sound. Mute-safe if AudioContext blocked.",
        ),
        RalphTask::new(
            "vfx_polish",
            "Visual Polish",
            "Ghost body shapes with eyes looking in move direction; frightened flash; score popups \
             when eating ghosts; smooth READY/GAME OVER/WIN overlays; optional touch buttons for mobile.",
        ),
        RalphTask::new(
            "balance_feel",
            "Game Feel & Balance",
            "Tune speeds, frightened duration by level, house release timers, and collision radii so \
             the game is fair and fun. Ensure all four ghosts visibly exhibit distinct behavior.",
        ),
    ];

    let system_context = "\
You are collaborating with other specialized agents to build a complete classic Pac-Man game \
in a single self-contained HTML file. All HTML, CSS, and JavaScript must be inline. \
Do NOT use external dependencies. Use the HTML5 Canvas API for rendering and the Web Audio API for sound.\n\n\
\
AUTHENTIC GHOST AI (must preserve / improve):\n\
- BLINKY (red): chase Pac-Man's tile directly\n\
- PINKY (pink): target 4 tiles ahead of Pac-Man (with classic UP quirk)\n\
- INKY (cyan): flank using Blinky's position and a pivot 2 tiles ahead of Pac-Man\n\
- CLYDE (orange): chase when far (>8 tiles), else scatter to corner\n\
- Global SCATTER/CHASE schedule; power pellets → FRIGHTENED; eaten → eyes return to house\n\n\
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
- Add your code into the existing <script> block; do not delete working ghost AI or maze code\n\
- The game already has a maze skeleton, Pac-Man, four ghosts with basic personality hooks, and a loop\n\n\
\
TOOLS AVAILABLE:\n\
- Memory (memory): {\"command\": \"G key\"} to read, {\"command\": \"P key value\"} to write, {\"command\": \"L\"} to list\n\
- write_game_file: {\"filename\": \"pacman_game_ralph.html\", \"content\": \"<!DOCTYPE html>...\"}\n\
- Bash (bash:*): Shell commands if needed\n\n\
\
Key Memory entries:\n\
  current_game_html — THE CURRENT COMPLETE GAME HTML (read this first, write back after changes)";

    let event_handler = Arc::new(PacManEventHandler::new());

    let mut orchestration =
        Orchestration::new("pacman-builder", "Pac-Man Game RALPH Orchestration")
            .with_mode(OrchestrationMode::Ralph {
                tasks,
                max_iterations: 10,
            })
            .with_system_context(system_context)
            .with_max_tokens(250_000)
            .with_event_handler(event_handler);

    orchestration.add_agent(architect)?;
    orchestration.add_agent(programmer)?;
    orchestration.add_agent(ghost_ai)?;
    orchestration.add_agent(audio_vfx)?;

    let prompt = "\
Build a complete, fully featured classic Pac-Man game in a single self-contained HTML file. \
Requirements: 28×31-style maze with tunnels and ghost house; Pac-Man grid movement with buffered \
turns; dots and power pellets; score/lives/levels; four ghosts with AUTHENTIC color personalities — \
Blinky (direct chase), Pinky (ambush 4 ahead), Inky (flank via Blinky), Clyde (shy 8-tile rule); \
scatter/chase mode cycling; frightened + eaten-eyes return-to-house; fruit bonus; chiptune audio; \
polished canvas rendering. No external dependencies. Prefer improving the existing skeleton over rewriting.";

    println!("Starting RALPH orchestration with 4 agents and 18 PRD tasks...\n");
    println!("Model: deepseek/deepseek-v4-flash-0731 via OpenRouter\n");

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
                println!("  {}: {}...", key, &value[..preview_end]);
            }
        }
    }

    let final_html = if let Some((mem_html, _)) = memory.get("current_game_html", false) {
        if mem_html.len() > 1000 && mem_html.contains("<canvas") {
            let unescaped = mem_html
                .replace("\\n", "\n")
                .replace("\\t", "\t")
                .replace("\\\"", "\"");
            std::fs::write("pacman_game_ralph.html", &unescaped)?;
            println!(
                "\n✅ Game written from Memory to pacman_game_ralph.html ({} bytes)",
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
        let mut game_html: Option<String> = None;
        for msg in response.messages.iter().rev() {
            let html = extract_html(&msg.content);
            if html.len() > 1000 && (html.contains("<canvas") || html.contains("canvas")) {
                game_html = Some(html);
                break;
            }
        }
        if let Some(html) = game_html {
            std::fs::write("pacman_game_ralph.html", &html)?;
            println!(
                "\n✅ Game extracted from messages to pacman_game_ralph.html ({} bytes)",
                html.len()
            );
        } else {
            let disk_size = std::fs::metadata("pacman_game_ralph.html")
                .map(|m| m.len())
                .unwrap_or(0);
            println!(
                "\n⚠️  Agents didn't write updates via write_game_file. Starter HTML on disk ({} bytes).",
                disk_size
            );
        }
    }
    println!("Open pacman_game_ralph.html in a browser to play!");

    Ok(())
}

/// Attempt to extract a self-contained HTML document from an LLM response.
fn extract_html(text: &str) -> String {
    let lower = text.to_lowercase();
    let start = lower
        .find("<!doctype")
        .or_else(|| lower.find("<html"))
        .unwrap_or(0);

    let end = lower
        .rfind("</html>")
        .map(|i| i + "</html>".len())
        .unwrap_or(text.len());

    let raw = &text[start..end];
    raw.replace("\\n", "\n")
        .replace("\\t", "\t")
        .replace("\\\"", "\"")
}
