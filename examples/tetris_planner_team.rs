use async_trait::async_trait;
use cloudllm::clients::claude::{ClaudeClient, Model};
use cloudllm::event::{AgentEvent, EventHandler, OrchestrationEvent, PlannerEvent};
use cloudllm::tool_protocol::{
    ToolMetadata, ToolParameter, ToolParameterType, ToolRegistry, ToolResult,
};
use cloudllm::tool_protocols::{CustomToolProtocol, MemoryProtocol};
use cloudllm::tools::Memory;
use cloudllm::{
    orchestration::{Orchestration, OrchestrationMode, RalphTask},
    Agent,
};
use serde_json::json;
use std::fs;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::RwLock;

struct TetrisEventHandler {
    start: Instant,
    wrote_file: Arc<AtomicBool>,
}

impl TetrisEventHandler {
    fn new(wrote_file: Arc<AtomicBool>) -> Self {
        Self {
            start: Instant::now(),
            wrote_file,
        }
    }

    fn elapsed(&self) -> String {
        let secs = self.start.elapsed().as_secs();
        format!("{:02}:{:02}", secs / 60, secs % 60)
    }

    fn log(&self, domain: &str, msg: impl AsRef<str>) {
        println!("[{}] [{}] {}", self.elapsed(), domain, msg.as_ref());
    }
}

#[async_trait]
impl EventHandler for TetrisEventHandler {
    async fn on_agent_event(&self, event: &AgentEvent) {
        match event {
            AgentEvent::SendStarted {
                agent_name,
                message_preview,
                ..
            } => {
                let preview = message_preview
                    .chars()
                    .take(100)
                    .collect::<String>();
                self.log("agent", format!("▶ {agent_name} starting turn: {preview}..."));
            }
            AgentEvent::LLMCallStarted {
                agent_name,
                iteration,
                ..
            } => {
                self.log("agent", format!("  ├─ {agent_name} LLM call #{iteration} started"));
            }
            AgentEvent::LLMCallCompleted {
                agent_name,
                iteration,
                response_length,
                tokens_used,
                ..
            } => {
                let tokens = tokens_used
                    .as_ref()
                    .map(|u| u.total_tokens)
                    .unwrap_or(0);
                self.log(
                    "agent",
                    format!("  ├─ {agent_name} LLM call #{iteration} done ({response_length} chars, {tokens} tokens)"),
                );
            }
            AgentEvent::ToolCallDetected {
                agent_name,
                tool_name,
                parameters,
                iteration,
                ..
            } => {
                let param_str = serde_json::to_string(&parameters)
                    .unwrap_or_else(|_| "??".to_string());
                let param_preview = if param_str.len() > 60 {
                    format!("{}...", &param_str[..60])
                } else {
                    param_str
                };
                self.log(
                    "agent",
                    format!("  ├─ {agent_name} tool call #{iteration}: {tool_name}({param_preview})"),
                );
            }
            AgentEvent::ToolExecutionCompleted {
                agent_name,
                tool_name,
                success,
                error,
                result,
                iteration,
                ..
            } => {
                if *success {
                    if tool_name == "write_tetris_file" {
                        self.wrote_file.store(true, Ordering::SeqCst);
                        if let Some(res) = result {
                            if let Some(bytes) = res.get("bytes") {
                                self.log(
                                    "agent",
                                    format!(
                                        "  ├─ ✅ {agent_name} wrote HTML file ({} bytes) [iter #{iteration}]",
                                        bytes.as_u64().unwrap_or(0)
                                    ),
                                );
                            }
                        }
                    } else {
                        self.log(
                            "agent",
                            format!("  ├─ ✅ {agent_name} tool '{tool_name}' succeeded [iter #{iteration}]"),
                        );
                    }
                } else {
                    self.log(
                        "agent",
                        format!(
                            "  ├─ ❌ {agent_name} tool '{tool_name}' FAILED [iter #{iteration}]: {}",
                            error.as_deref().unwrap_or("unknown error")
                        ),
                    );
                }
            }
            AgentEvent::SendCompleted {
                agent_name,
                response_length,
                tool_calls_made,
                tokens_used,
                ..
            } => {
                let tokens = tokens_used
                    .as_ref()
                    .map(|usage| usage.total_tokens)
                    .unwrap_or(0);
                self.log(
                    "agent",
                    format!(
                        "✓ {agent_name} completed ({response_length} chars, {tokens} tokens, {tool_calls_made} tool calls)"
                    ),
                );
            }
            AgentEvent::ToolMaxIterationsReached { agent_name, .. } => self.log(
                "agent",
                format!("❌ {agent_name} hit max tool iterations (tool loop stuck)"),
            ),
            AgentEvent::SystemPromptSet { agent_name, .. } => {
                self.log("agent", format!("📝 {agent_name} system prompt set"));
            }
            AgentEvent::MessageReceived { agent_name, .. } => {
                self.log("agent", format!("📨 {agent_name} received routed message"));
            }
            _ => {}
        }
    }

    async fn on_planner_event(&self, event: &PlannerEvent) {
        match event {
            PlannerEvent::TurnStarted {
                plan_id,
                message_preview,
            } => {
                let preview = message_preview
                    .chars()
                    .take(80)
                    .collect::<String>();
                self.log(
                    "planner",
                    format!("▶ Plan {}: {preview}...", plan_id),
                );
            }
            PlannerEvent::LLMCallStarted { iteration, .. } => {
                self.log("planner", format!("  ├─ LLM call #{iteration} started"));
            }
            PlannerEvent::LLMCallCompleted {
                iteration,
                response_length,
                ..
            } => {
                self.log(
                    "planner",
                    format!("  ├─ LLM call #{iteration} done ({response_length} chars)"),
                );
            }
            PlannerEvent::ToolCallDetected {
                tool_name,
                iteration,
                ..
            } => {
                self.log(
                    "planner",
                    format!("  ├─ Tool call #{iteration}: {tool_name}"),
                );
            }
            PlannerEvent::ToolExecutionCompleted {
                tool_name,
                success,
                error,
                iteration,
                ..
            } => {
                if *success {
                    self.log(
                        "planner",
                        format!("  ├─ ✅ {tool_name} succeeded [#{iteration}]"),
                    );
                } else {
                    self.log(
                        "planner",
                        format!(
                            "  ├─ ❌ {tool_name} FAILED [#{iteration}]: {}",
                            error.as_deref().unwrap_or("unknown")
                        ),
                    );
                }
            }
            PlannerEvent::TurnCompleted {
                tool_calls_made,
                response_length,
                tokens_used,
                ..
            } => {
                let tokens = tokens_used
                    .as_ref()
                    .map(|u| u.total_tokens)
                    .unwrap_or(0);
                self.log(
                    "planner",
                    format!(
                        "✓ Plan completed ({response_length} chars, {tokens} tokens, {tool_calls_made} tool calls)"
                    ),
                );
            }
            PlannerEvent::TurnErrored { error, .. } => {
                self.log("planner", format!("❌ Plan error: {error}"));
            }
            PlannerEvent::ToolMaxIterationsReached { .. } => {
                self.log("planner", format!("❌ Plan hit max tool iterations"));
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
            } => self.log(
                "orch",
                format!(
                    "🚀 Run started: {orchestration_name} [{mode}, {agent_count} agents]"
                ),
            ),
            OrchestrationEvent::RoundStarted { round, .. } => {
                self.log("orch", format!("── Round {round} ──"));
            }
            OrchestrationEvent::AgentSelected {
                agent_name,
                reason,
                ..
            } => {
                self.log(
                    "orch",
                    format!("  Agent selected: {agent_name} ({reason})"),
                );
            }
            OrchestrationEvent::AgentResponded {
                agent_name,
                response_length,
                tokens_used,
                ..
            } => {
                let tokens = tokens_used
                    .as_ref()
                    .map(|u| u.total_tokens)
                    .unwrap_or(0);
                self.log(
                    "orch",
                    format!(
                        "  {agent_name} responded ({response_length} chars, {tokens} tokens)"
                    ),
                );
            }
            OrchestrationEvent::AgentFailed {
                agent_name,
                error,
                ..
            } => {
                self.log(
                    "orch",
                    format!("  ❌ {agent_name} failed: {error}"),
                );
            }
            OrchestrationEvent::RoundCompleted { round, .. } => {
                self.log("orch", format!("── Round {round} complete ──"));
            }
            OrchestrationEvent::RalphIterationStarted {
                iteration,
                max_iterations,
                tasks_completed,
                tasks_total,
                ..
            } => self.log(
                "orch",
                format!(
                    "🔄 RALPH iteration {iteration}/{max_iterations} ({tasks_completed}/{tasks_total} tasks done)"
                ),
            ),
            OrchestrationEvent::RalphTaskCompleted {
                agent_name,
                task_ids,
                tasks_completed_total,
                tasks_total,
                ..
            } => self.log(
                "orch",
                format!(
                    "  ✅ {agent_name} completed: {} → {tasks_completed_total}/{tasks_total}",
                    task_ids.join(", ")
                ),
            ),
            OrchestrationEvent::RunCompleted {
                rounds,
                total_tokens,
                is_complete,
                ..
            } => {
                let status = if *is_complete { "✅" } else { "⚠️" };
                self.log(
                    "orch",
                    format!(
                        "{status} Run finished: {rounds} iterations, {total_tokens} tokens, complete={is_complete}"
                    ),
                );
            }
            _ => {}
        }
    }
}

fn normalize_generated_html(raw: &str) -> String {
    raw.replace("\\n", "\n")
        .replace("\\t", "\t")
        .replace("\\\"", "\"")
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let api_key =
        match std::env::var("ANTHROPIC_API_KEY").or_else(|_| std::env::var("ANTHROPIC_KEY")) {
            Ok(key) => key,
            Err(_) => {
                eprintln!("Missing ANTHROPIC_API_KEY (or ANTHROPIC_KEY) environment variable.");
                eprintln!("Example usage:");
                eprintln!("  export ANTHROPIC_API_KEY=sk-ant-...");
                eprintln!("  cargo run --example tetris_planner_team");
                std::process::exit(1);
            }
        };

    println!("\n╔════════════════════════════════════════════════════════════════╗");
    println!("║         TETRIS BUILDER — RALPH Orchestration Demo             ║");
    println!("║               Claude Sonnet 4.6 Agent Team                     ║");
    println!("╚════════════════════════════════════════════════════════════════╝\n");

    println!("📐 ORCHESTRATION SETUP:");
    println!("  Mode: RALPH (Iterative task-based coordination)");
    println!("  Max Iterations: 8");
    println!("  Agents: 4 specialists (researcher, architect, programmer, tester)");
    println!("  Model: Claude Sonnet 4.6");
    println!("  Shared Tools: Memory, file I/O");
    println!();
    println!("🎯 PROCESS:");
    println!("  1. Agents read current HTML from Memory");
    println!("  2. Each agent works on assigned task");
    println!("  3. Agent writes updated HTML via write_tetris_file");
    println!("  4. Mark [TASK_COMPLETE:task_id] when done");
    println!("  5. Repeat until all 4 tasks complete");
    println!();
    println!("📊 TASKS:");
    println!("  1. board_engine  — Board state, pieces (SRS), canvas shell");
    println!("  2. gameplay_loop — Game loop, gravity, input, scoring, hold");
    println!("  3. rendering_ui  — Draw board, ghost, next/hold panels, legend");
    println!("  4. polish_audio  — Web Audio effects, mute toggle, start screen");
    println!();
    println!("═══════════════════════════════════════════════════════════════\n");

    let output_path = std::env::current_dir()?.join("tetris_planner_output.html");
    let starter_html = r#"<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8" />
    <title>OpenClaw Tetris</title>
    <style>
        body { background: #111; color: #fafafa; font-family: "Press Start 2P", monospace; margin: 0; display:flex; align-items:center; justify-content:center; min-height:100vh; }
        #app { display:flex; gap:32px; align-items:flex-start; }
        canvas { background:#1b1b1b; border:4px solid #303030; box-shadow:0 0 20px rgba(0,0,0,0.6); }
        .panel { text-transform:uppercase; letter-spacing:0.08em; }
        h1 { font-size:18px; margin-bottom:16px; text-align:center; }
    </style>
</head>
<body>
    <div id="app">
        <canvas id="playfield" width="320" height="640"></canvas>
        <div class="panel">
            <h1>OpenClaw Tetris</h1>
            <p>This file is a starting point. The agent team will replace it with a full implementation.</p>
        </div>
    </div>
    <script>
        console.log("Starter Tetris shell loaded.");
    </script>
</body>
</html>
"#;

    // Always start fresh — delete any previous output so agents can't accidentally
    // inherit a stale or broken game from a prior run.
    if output_path.exists() {
        fs::remove_file(&output_path)?;
        println!("🗑️  Cleared previous output: {}", output_path.display());
    }
    fs::write(&output_path, starter_html)?;
    let baseline_html = starter_html.to_string();

    let memory = Arc::new(Memory::new());
    memory.put(
        "tetris_current_html".to_string(),
        baseline_html.clone(),
        None,
    );

    println!(
        "📄 Fresh game shell written at {} ({} bytes)",
        output_path.display(),
        baseline_html.len()
    );

    let memory_protocol = Arc::new(MemoryProtocol::new(memory.clone()));
    let custom_protocol = Arc::new(CustomToolProtocol::new());

    custom_protocol
        .register_tool(
            ToolMetadata::new("read_file", "Read a UTF-8 text file from disk (returns content)")
                .with_parameter(
                    ToolParameter::new("path", ToolParameterType::String)
                        .with_description("Absolute or relative file path to read")
                        .required(),
                ),
            Arc::new(|params| {
                let path = params
                    .get("path")
                    .and_then(|v| v.as_str())
                    .unwrap_or("");
                if path.is_empty() {
                    return Ok(ToolResult::failure("path parameter is required and cannot be empty".to_string()));
                }
                match fs::read_to_string(path) {
                    Ok(content) => {
                        eprintln!("[read_file] Read {} bytes from {}", content.len(), path);
                        Ok(ToolResult::success(json!({
                            "path": path,
                            "content": content,
                            "bytes": content.len(),
                        })))
                    }
                    Err(err) => {
                        eprintln!("[read_file] ERROR reading {}: {}", path, err);
                        Ok(ToolResult::failure(format!("Failed to read {}: {}", path, err)))
                    }
                }
            }),
        )
        .await;

    let default_path = Arc::new(output_path.to_string_lossy().into_owned());
    let memory_for_tool = memory.clone();
    let write_tool_path = default_path.clone();
    custom_protocol
        .register_tool(
            ToolMetadata::new(
                "write_tetris_file",
                "Write the COMPLETE Tetris HTML/CSS/JS bundle to disk AND memory (CRITICAL: must include full document)"
            )
            .with_parameter(
                ToolParameter::new("path", ToolParameterType::String)
                    .with_description("Output path (defaults to tetris_planner_output.html if not specified)"),
            )
            .with_parameter(
                ToolParameter::new("content", ToolParameterType::String)
                    .with_description("Complete, valid HTML document including DOCTYPE, html, head (with style), body (with canvas and script)")
                    .required(),
            ),
            Arc::new(move |params| {
                // Extract parameters safely
                let path_value = params
                    .get("path")
                    .and_then(|v| v.as_str())
                    .filter(|s| !s.is_empty())
                    .unwrap_or(write_tool_path.as_str());

                let raw_content = match params.get("content").and_then(|v| v.as_str()) {
                    Some(s) => s,
                    None => {
                        eprintln!("[write_tetris_file] ERROR: content parameter is missing or null");
                        return Ok(ToolResult::failure(
                            "content parameter is required and must contain the complete HTML".to_string(),
                        ));
                    }
                };

                if raw_content.trim().is_empty() {
                    eprintln!("[write_tetris_file] ERROR: content is empty");
                    return Ok(ToolResult::failure(
                        "content cannot be empty - must contain complete HTML document".to_string(),
                    ));
                }

                // Normalize HTML (fix escape sequences)
                let normalized = normalize_generated_html(raw_content);

                // Validate it looks like HTML
                if !normalized.contains("<html") && !normalized.contains("<HTML") {
                    eprintln!("[write_tetris_file] WARNING: content doesn't look like HTML (no <html tag)");
                }

                // Create parent directories if needed
                let target_path = PathBuf::from(path_value);
                if let Some(parent) = target_path.parent() {
                    if parent.as_os_str().len() > 0 {
                        if let Err(err) = fs::create_dir_all(parent) {
                            eprintln!("[write_tetris_file] ERROR creating parent dir: {}", err);
                            return Ok(ToolResult::failure(format!("Failed to create parent directory: {}", err)));
                        }
                    }
                }

                // Write to disk
                let byte_count = normalized.len();
                match fs::write(&target_path, normalized.as_bytes()) {
                    Ok(_) => {
                        eprintln!("[write_tetris_file] ✅ Wrote {} bytes to {}", byte_count, target_path.display());

                        // Also store in Memory for agent coordination
                        memory_for_tool.put("tetris_current_html".to_string(), normalized.clone(), None);
                        eprintln!("[write_tetris_file] ✅ Updated Memory key 'tetris_current_html'");

                        Ok(ToolResult::success(json!({
                            "path": target_path.to_string_lossy().to_string(),
                            "bytes": byte_count,
                            "in_memory": true,
                        })))
                    }
                    Err(err) => {
                        eprintln!("[write_tetris_file] ERROR writing to disk: {}", err);
                        Ok(ToolResult::failure(format!("Failed to write file: {}", err)))
                    }
                }
            }),
        )
        .await;

    let mut shared_registry = ToolRegistry::empty();

    println!("\n📋 Registering tool protocols...");
    println!("  ├─ Memory protocol (GET/PUT/LIST for shared state)");
    shared_registry
        .add_protocol("memory", memory_protocol)
        .await?;

    println!("  ├─ Custom protocol (read_file, write_tetris_file)");
    shared_registry
        .add_protocol("custom", custom_protocol)
        .await?;

    let tool_list = shared_registry.list_tools();
    println!("  └─ Done: {} tools available", tool_list.len());

    println!();
    println!("🔍 TOOL DESCRIPTIONS (exactly as agents will see them):");
    println!("══════════════════════════════════════════════════════════════════════");
    for (i, tool) in tool_list.iter().enumerate() {
        // Truncate long descriptions at the first newline for the header line
        let desc_summary = tool.description.lines().next().unwrap_or(&tool.description);
        println!("[{}] {} — {}", i + 1, tool.name, desc_summary);
        if !tool.parameters.is_empty() {
            println!("    Parameters:");
            for param in &tool.parameters {
                let req = if param.required { " [REQUIRED]" } else { "" };
                let type_str = format!("{:?}", param.param_type).to_lowercase();
                let desc = param.description.as_deref().unwrap_or("(no description)");
                println!("      • {} ({}){}", param.name, type_str, req);
                println!("        {}", desc);
            }
        }
        // Print full multi-line description for the memory tool since it teaches the protocol
        if tool.name == "memory" && tool.description.contains('\n') {
            println!("    Full description:");
            for line in tool.description.lines().skip(1) {
                println!("      {}", line);
            }
        }
        println!();
    }
    println!("══════════════════════════════════════════════════════════════════════");
    println!();

    let shared_registry = Arc::new(RwLock::new(shared_registry));

    let make_client = || {
        Arc::new(ClaudeClient::new_with_model_enum(
            &api_key,
            Model::ClaudeSonnet46,
        )) as Arc<dyn cloudllm::ClientWrapper>
    };

    let researcher = Agent::new("tetris-researcher", "Gameplay Researcher", make_client())
        .with_expertise("Canonical Tetris rules, NES/SNES reference behavior")
        .with_personality("Meticulous archivist who cites classic implementations.")
        .with_shared_tools(shared_registry.clone());

    let architect = Agent::new("tetris-architect", "System Architect", make_client())
        .with_expertise("HTML5 layout, Canvas rendering, component structure")
        .with_personality("Clean, methodical layout engineer.")
        .with_shared_tools(shared_registry.clone());

    let programmer = Agent::new("tetris-programmer", "Gameplay Programmer", make_client())
        .with_expertise("JavaScript game loops, collision detection, rotation systems")
        .with_personality("Fast iteration gameplay engineer.")
        .with_shared_tools(shared_registry.clone());

    let playtester = Agent::new("tetris-playtester", "QA & Polish", make_client())
        .with_expertise("UX polish, accessibility, instructions, audio balancing")
        .with_personality("Enthusiastic playtester with an ear for detail.")
        .with_shared_tools(shared_registry.clone());

    let system_context = r#"You are a Tetris builder agent. Your output is tool call JSON, not text.

══════════════════════════════════════════════════════════════════════════
YOUR ENTIRE RESPONSE FOR EACH TASK FOLLOWS ONE OF THESE TWO PATTERNS:
══════════════════════════════════════════════════════════════════════════

PATTERN A — Write file directly (use when you have enough context):

  {"tool_call": {"name": "write_tetris_file", "parameters": {"content": "<!DOCTYPE html>...FULL HTML..."}}}

  [TASK_COMPLETE:task_id]

PATTERN B — Read prior state first, then write (use when extending previous work):

  {"tool_call": {"name": "memory", "parameters": {"command": "G", "key": "tetris_current_html"}}}

  ... wait for tool result, then immediately: ...

  {"tool_call": {"name": "write_tetris_file", "parameters": {"content": "<!DOCTYPE html>...UPDATED HTML..."}}}

  [TASK_COMPLETE:task_id]

══════════════════════════════════════════════════════════════════════════
ABSOLUTE RULES:
══════════════════════════════════════════════════════════════════════════

1. Your response begins with a {"tool_call": ...} line. No text before it.
2. After a memory GET result, your next output is IMMEDIATELY write_tetris_file.
3. HTML, JS, or CSS typed anywhere outside a tool_call parameter is IGNORED.
4. write_tetris_file MUST be called before [TASK_COMPLETE:X] is valid.
5. No markdown, no code fences, no explanations — only tool call JSON.

══════════════════════════════════════════════════════════════════════════
TETRIS REQUIREMENTS (for reference):
══════════════════════════════════════════════════════════════════════════

Single HTML file: DOCTYPE + head(style) + body(canvas + side panels) + script
- 7-bag randomizer, SRS rotation + wall kicks, hold slot (C key)
- Next queue (3 pieces), ghost piece, gravity + level (level = lines/10)
- Controls: ←/→ move, ↑/Z rotate, ↓ soft drop, Space hard drop, P pause
- Scoring: 100/300/500/800 × level; game over when spawn blocked
- Start/pause overlay; dark retro theme; score/level/lines display"#.to_string();

    let write_flag = Arc::new(AtomicBool::new(false));
    let event_handler = Arc::new(TetrisEventHandler::new(write_flag.clone()));

    let tasks = vec![
        RalphTask::new(
            "board_engine",
            "Board state, pieces, and canvas shell",
            "Write a complete Tetris HTML file from scratch with: \
             PIECES array (all 7 tetrominoes, SRS rotation states, hex colors), \
             Board class (10×20 grid, collision detection, line-clear), \
             Bag7 randomizer, canvas id='playfield' 320×640, drawBoard() stub. \
             Call write_tetris_file with the COMPLETE HTML. \
             After it succeeds write [TASK_COMPLETE:board_engine].",
        ),
        RalphTask::new(
            "gameplay_loop",
            "Game loop, input, gravity, and scoring",
            "Read current HTML from memory first (key: tetris_current_html). \
             Then extend the script — keep all existing code, add: \
             requestAnimationFrame loop with delta-time gravity, \
             keyboard handler (←/→ move, ↑/Z rotate, ↓ soft drop, Space hard drop, C hold, P pause), \
             lock delay 500ms, hold slot (swap once per piece), \
             scoring (single=100 double=300 triple=500 tetris=800 × level), \
             level up every 10 lines, game-over detection, start/pause canvas overlay. \
             Call write_tetris_file with the COMPLETE updated HTML. \
             After it succeeds write [TASK_COMPLETE:gameplay_loop].",
        ),
        RalphTask::new(
            "rendering_ui",
            "Rendering all canvases and side panels",
            "Read current HTML from memory first (key: tetris_current_html). \
             Then extend the script — keep all existing code, add: \
             colored board cells (piece color or #1b1b1b for empty), \
             draw active piece, ghost piece (projected drop position), \
             next-piece panel (3 upcoming), hold panel, \
             score/level/lines elements updated each frame, keyboard legend panel. \
             Call write_tetris_file with the COMPLETE updated HTML. \
             After it succeeds write [TASK_COMPLETE:rendering_ui].",
        ),
        RalphTask::new(
            "polish_audio",
            "Polish: audio, instructions, and final QA",
            "Read current HTML from memory first (key: tetris_current_html). \
             Then extend the script — keep all existing code, add: \
             Web Audio sound effects (line-clear beep, lock tick, game-over buzz), \
             mute toggle button, 'Press Space to start' on-screen prompt, \
             fix any JS syntax errors, ensure game loop starts on page load. \
             Call write_tetris_file with the COMPLETE final HTML. \
             After it succeeds write [TASK_COMPLETE:polish_audio].",
        ),
    ];

    let mut orchestration = Orchestration::new("tetris-ralph", "RALPH Tetris Build")
        .with_mode(OrchestrationMode::Ralph {
            tasks,
            max_iterations: 8,
        })
        .with_system_context(system_context)
        .with_max_tokens(32_000)
        .with_event_handler(event_handler.clone());

    orchestration.add_agent(researcher)?;
    orchestration.add_agent(architect)?;
    orchestration.add_agent(programmer)?;
    orchestration.add_agent(playtester)?;

    let prompt = "\
Build a fully playable Tetris game as a single HTML file. \
IMPORTANT: You MUST call write_tetris_file to save your code — code in response text is ignored. \
Each task has explicit steps; follow them exactly. Do NOT mark [TASK_COMPLETE:X] unless you called write_tetris_file this turn. \
Classic mechanics required: SRS rotation, 7-bag randomizer, hold slot, ghost piece, scoring, level progression, game over.";

    println!("Starting RALPH orchestration with 4 agents and 4 PRD tasks...\n");

    let start = Instant::now();
    let response = orchestration.run(prompt, 1).await?;
    let elapsed = start.elapsed();

    println!("\n╔════════════════════════════════════════════════════════════════╗");
    println!("║                     RALPH RUN SUMMARY                         ║");
    println!("╚════════════════════════════════════════════════════════════════╝");
    println!();
    println!("📊 RESULTS:");
    println!("  Iterations Executed : {}", response.round);
    println!(
        "  Task Completion Rate : {:.0}% ({:.1}/4 tasks)",
        response.convergence_score.unwrap_or(0.0) * 100.0,
        response.convergence_score.unwrap_or(0.0) * 4.0
    );
    println!("  Total Tokens Used   : {}", response.total_tokens_used);
    println!(
        "  Elapsed Time        : {}m {}s",
        elapsed.as_secs() / 60,
        elapsed.as_secs() % 60
    );
    println!("  Status              : {}", if response.is_complete { "✅ COMPLETE" } else { "⏳ INCOMPLETE" });
    println!();

    // Show which agents did what (messages use display names, not IDs)
    println!("📝 AGENT ACTIVITY:");
    let mut agents_worked = std::collections::HashSet::new();
    for msg in &response.messages {
        if let Some(agent_name) = &msg.agent_name {
            agents_worked.insert(agent_name.clone());
        }
    }
    for (display_name, id) in &[
        ("Gameplay Researcher", "tetris-researcher"),
        ("System Architect",    "tetris-architect"),
        ("Gameplay Programmer", "tetris-programmer"),
        ("QA & Polish",         "tetris-playtester"),
    ] {
        let marker = if agents_worked.contains(*display_name) { "✓" } else { "✗" };
        println!("  {} {} ({})", marker, display_name, id);
    }
    println!();

    // Show detailed turn breakdown
    println!("📋 DETAILED TURNS:");
    for (idx, msg) in response.messages.iter().enumerate() {
        let agent = msg.agent_name.as_deref().unwrap_or("unknown");
        let iteration = msg
            .metadata
            .get("iteration")
            .map(|s| s.as_str())
            .unwrap_or("?");
        let tasks_completed = msg
            .metadata
            .get("tasks_completed")
            .map(|s| s.as_str())
            .unwrap_or("");

        if !tasks_completed.is_empty() {
            println!(
                "  [{:02}] iter={} agent={:<20} completed={}",
                idx + 1,
                iteration,
                agent,
                tasks_completed
            );
        } else {
            println!(
                "  [{:02}] iter={} agent={:<20} (no completion marker)",
                idx + 1,
                iteration,
                agent
            );
        }
    }
    println!();

    // ── File status ──────────────────────────────────────────────────────────
    println!("💾 FILE STATUS:");
    let file_written = write_flag.load(Ordering::SeqCst);

    // Primary path: write_tetris_file was called properly.
    let saved_html: Option<String> = if file_written {
        println!("  ✅ write_tetris_file was called (proper tool flow)");
        memory
            .get("tetris_current_html", false)
            .map(|(v, _)| v)
            .filter(|h| h.contains("<canvas"))
    } else {
        // Rescue path: agents wrote HTML as response text instead of calling the tool.
        // Scan all agent messages for the largest complete HTML document.
        println!("  ⚠️  write_tetris_file was NOT called — scanning response text for HTML...");
        let mut best: Option<String> = None;
        for msg in &response.messages {
            let content = msg.content.as_ref();
            // Find the first <!DOCTYPE html> or <html in the message
            let start_idx = content
                .find("<!DOCTYPE html>")
                .or_else(|| content.find("<!doctype html>"))
                .or_else(|| content.find("<html"));
            if let Some(start) = start_idx {
                let slice = &content[start..];
                // Take everything up to and including the last </html>
                if let Some(end_off) = slice.rfind("</html>") {
                    let candidate = &slice[..end_off + "</html>".len()];
                    if candidate.contains("<script") {
                        // Keep the longest one (most complete implementation)
                        if best.as_ref().map_or(0, |b: &String| b.len()) < candidate.len() {
                            best = Some(candidate.to_string());
                        }
                    }
                }
            }
        }
        if best.is_some() {
            println!("  🔧 Extracted HTML from response text (rescue mode)");
        } else {
            println!("  ❌ No usable HTML found in any agent response");
        }
        best
    };

    let mut final_file_written = file_written;
    if let Some(html) = saved_html {
        let normalized = normalize_generated_html(&html);
        let lines = normalized.lines().count();
        let has_gameloop =
            normalized.contains("requestAnimationFrame") || normalized.contains("gameLoop");
        let has_pieces = normalized.contains("pieces") || normalized.contains("PIECES");
        println!("  {} lines, game_loop={}, pieces={}", lines, has_gameloop, has_pieces);
        fs::write(&output_path, normalized.as_bytes())?;
        println!("  ✅ Written {} bytes to {}", normalized.len(), output_path.display());
        final_file_written = true;
    } else if !file_written {
        println!("  ❌ Nothing to save — agents produced no usable HTML in this run");
    }
    println!();

    println!("🎮 NEXT STEPS:");
    if final_file_written {
        println!("  ✅ Open {} in a web browser", output_path.display());
        if !file_written {
            println!("     (Rescued from response text — tool flow did not work as intended)");
        }
    } else {
        println!("  ❌ No output produced. Check event log above for per-agent tool call counts.");
        println!("     Context: {}k tokens used across {} iterations", response.total_tokens_used / 1000, response.round);
    }
    println!();
    Ok(())
}
