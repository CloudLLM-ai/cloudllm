//! Shared live-progress console for long-running examples and agent runs.
//!
//! Reasoning models (Grok 4.6, DeepSeek V4 Pro, o-series, …) can sit silent for
//! minutes while they think. [`LiveConsoleHandler`] prints:
//!
//! * **heartbeats** while an LLM call is in flight, including the phase
//!   (`waiting for first token`, `reasoning`, `generating`, `blocked`)
//! * **reasoning traces** in dark gray (ANSI 90)
//! * tool calls, RALPH task progress, and run banners
//!
//! Attach it once on an [`Orchestration`](crate::orchestration::Orchestration)
//! or [`Agent`](crate::Agent):
//!
//! ```rust,no_run
//! use std::sync::Arc;
//! use cloudllm::live_console::LiveConsoleHandler;
//! use cloudllm::orchestration::Orchestration;
//!
//! let orch = Orchestration::new("demo", "Demo")
//!     .with_event_handler(Arc::new(LiveConsoleHandler::new()));
//! # let _ = orch;
//! ```
//!
//! # Environment
//!
//! | Variable | Effect |
//! | --- | --- |
//! | `CLOUDLLM_NO_COLOR=1` | Disable ANSI colors |
//! | `CLOUDLLM_STREAM_REASONING=full\|compact\|off` | How to print reasoning (default `full`) |
//! | `CLOUDLLM_STREAM_CONTENT=full\|compact\|off` | How to print visible tokens (default `compact`) |
//! | `CLOUDLLM_HEARTBEAT_SECS` | Heartbeat interval used by `Agent` (default `10`) |
//! | `CLOUDLLM_REASONING_EFFORT=low\|medium\|high` | Provider effort hint (can cut hour-long runs) |
//! | `MENTISDB_DIR` | Local chain directory (default `mentisdbs/`). **No daemon.** |
//! | `MENTISDB_CHAIN_KEY` | Chain name (example default is usually `cloudllm`) |

use crate::event::{AgentEvent, EventHandler, OrchestrationEvent, PlannerEvent};
use crate::{CloudLLMConfig, MentisDb};
use async_trait::async_trait;
use std::io::{self, IsTerminal, Write};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Instant;
use tokio::sync::RwLock;

const ANSI_RESET: &str = "\x1b[0m";
const ANSI_DARK_GRAY: &str = "\x1b[90m";
const ANSI_DIM: &str = "\x1b[2m";
const ANSI_CYAN: &str = "\x1b[36m";
const ANSI_RED: &str = "\x1b[31m";
const ANSI_YELLOW: &str = "\x1b[33m";

/// How streamed text is written to the terminal.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum StreamPrint {
    Full,
    Compact,
    Off,
}

impl StreamPrint {
    fn from_env(var: &str, default: StreamPrint) -> Self {
        match std::env::var(var)
            .ok()
            .map(|s| s.trim().to_ascii_lowercase())
            .as_deref()
        {
            Some("full") => StreamPrint::Full,
            Some("compact") => StreamPrint::Compact,
            Some("off") | Some("none") | Some("0") => StreamPrint::Off,
            _ => default,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum StreamKind {
    Idle,
    Reasoning,
    Content,
}

struct PrinterState {
    kind: StreamKind,
    reasoning_chars: usize,
    content_chars: usize,
    reasoning_shown: usize,
    content_shown: usize,
}

impl Default for PrinterState {
    fn default() -> Self {
        Self {
            kind: StreamKind::Idle,
            reasoning_chars: 0,
            content_chars: 0,
            reasoning_shown: 0,
            content_shown: 0,
        }
    }
}

/// In-process MentisDB handle opened by a RALPH / game example.
///
/// This is **not** a network client. [`MentisDb::open_with_key`] writes
/// `.tcbin` files under [`Self::dir`]. No `mentisdbd` process is required.
pub struct EmbeddedMentisDb {
    /// Directory that holds the chain files.
    pub dir: PathBuf,
    /// Chain key passed to [`MentisDb::open_with_key`].
    pub chain_key: String,
    /// Shared chain used by every agent in the run.
    pub db: Arc<RwLock<MentisDb>>,
}

/// Pretty-prints agent, planner, and orchestration events to stdout.
///
/// Designed for hour-scale RALPH / team runs where the hold-up is almost
/// always a single in-flight LLM call (often hidden reasoning).
pub struct LiveConsoleHandler {
    start: Instant,
    color: bool,
    reasoning: StreamPrint,
    content: StreamPrint,
    compact_limit: usize,
    state: Mutex<PrinterState>,
}

impl Default for LiveConsoleHandler {
    fn default() -> Self {
        Self::new()
    }
}

impl LiveConsoleHandler {
    /// Construct a handler that colors output when stdout is a TTY.
    pub fn new() -> Self {
        let no_color = std::env::var("CLOUDLLM_NO_COLOR")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false);
        Self {
            start: Instant::now(),
            color: !no_color && io::stdout().is_terminal(),
            reasoning: StreamPrint::from_env("CLOUDLLM_STREAM_REASONING", StreamPrint::Full),
            content: StreamPrint::from_env("CLOUDLLM_STREAM_CONTENT", StreamPrint::Compact),
            compact_limit: 8_192,
            state: Mutex::new(PrinterState::default()),
        }
    }

    /// Force plain (no ANSI) output.
    pub fn without_color(mut self) -> Self {
        self.color = false;
        self
    }

    /// Print the environment knobs that change live progress, reasoning traces,
    /// and run speed.
    ///
    /// Call this at the start of long RALPH / team examples so a run log shows
    /// what was actually set (versus the default).
    pub fn print_env_knobs() {
        println!("  Environment knobs (export before launch to change this run):");
        print_knob(
            "CLOUDLLM_REASONING_EFFORT",
            env_raw("CLOUDLLM_REASONING_EFFORT"),
            "unset → provider default",
            "low|medium|high|none  (low = faster, less thinking)",
        );
        print_knob(
            "CLOUDLLM_HEARTBEAT_SECS",
            env_raw("CLOUDLLM_HEARTBEAT_SECS"),
            "10",
            "seconds between \"still working\" lines (min 2)",
        );
        print_knob(
            "CLOUDLLM_STREAM_REASONING",
            env_raw("CLOUDLLM_STREAM_REASONING"),
            "full",
            "full|compact|off  (dark-gray thinking traces)",
        );
        print_knob(
            "CLOUDLLM_STREAM_CONTENT",
            env_raw("CLOUDLLM_STREAM_CONTENT"),
            "compact",
            "full|compact|off  (visible tokens; compact avoids huge HTML dumps)",
        );
        print_knob(
            "CLOUDLLM_NO_COLOR",
            env_raw("CLOUDLLM_NO_COLOR"),
            "unset → color if TTY",
            "1 to disable ANSI colors",
        );
        print_knob(
            "MENTISDB_DIR",
            env_raw("MENTISDB_DIR"),
            "mentisdbs/",
            "local files only — examples do not start or need mentisdbd",
        );
        print_knob(
            "MENTISDB_CHAIN_KEY",
            env_raw("MENTISDB_CHAIN_KEY"),
            "cloudllm",
            "embedded chain name under MENTISDB_DIR",
        );
        print_knob(
            "CLOUDLLM_HTTP_RETRIES",
            env_raw("CLOUDLLM_HTTP_RETRIES"),
            "3",
            "retries for transient Chat Completions blips (timeout/reset/429/5xx)",
        );
        println!();
    }

    /// Open the example MentisDB chain as **local files** (no daemon).
    ///
    /// RALPH examples are self-contained: `cargo run --example …` writes
    /// `mentisdbs/<chain>.tcbin` in the working directory. If the directory
    /// cannot be created or the chain cannot be opened (corrupt file, lock
    /// held by another process), this prints a hard-abort message and
    /// returns `Err` so `main` exits before any LLM spend.
    pub fn open_embedded_mentisdb(
        default_chain_key: &str,
    ) -> Result<EmbeddedMentisDb, Box<dyn std::error::Error + Send + Sync>> {
        let dir = std::env::var("MENTISDB_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|_| CloudLLMConfig::default().mentisdb_dir);
        let chain_key = std::env::var("MENTISDB_CHAIN_KEY")
            .unwrap_or_else(|_| default_chain_key.to_string());
        Self::open_embedded_mentisdb_at(dir, chain_key)
    }

    /// Open an embedded chain at an explicit directory (used by tests).
    pub fn open_embedded_mentisdb_at(
        dir: PathBuf,
        chain_key: String,
    ) -> Result<EmbeddedMentisDb, Box<dyn std::error::Error + Send + Sync>> {
        if let Err(err) = std::fs::create_dir_all(&dir) {
            return Err(abort_embedded_mentisdb(&dir, &chain_key, &err.to_string()));
        }
        match MentisDb::open_with_key(&dir, &chain_key) {
            Ok(db) => {
                println!(
                    "🧠 MentisDB embedded chain '{}' at {} (local files; no mentisdbd)",
                    chain_key,
                    dir.display()
                );
                Ok(EmbeddedMentisDb {
                    dir,
                    chain_key,
                    db: Arc::new(RwLock::new(db)),
                })
            }
            Err(err) => Err(abort_embedded_mentisdb(&dir, &chain_key, &err.to_string())),
        }
    }

    fn elapsed_str(&self) -> String {
        let secs = self.start.elapsed().as_secs();
        format!("{:02}:{:02}", secs / 60, secs % 60)
    }

    fn paint(&self, code: &str, text: &str) -> String {
        if self.color {
            format!("{}{}{}", code, text, ANSI_RESET)
        } else {
            text.to_string()
        }
    }

    fn finish_stream_line(&self, state: &mut PrinterState) {
        if state.kind != StreamKind::Idle {
            println!();
            let _ = io::stdout().flush();
            state.kind = StreamKind::Idle;
        }
    }

    fn lock_state(&self) -> std::sync::MutexGuard<'_, PrinterState> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn write_stream(&self, kind: StreamKind, text: &str) {
        if text.is_empty() || kind == StreamKind::Idle {
            return;
        }
        let mut state = self.lock_state();
        let mode = match kind {
            StreamKind::Reasoning => self.reasoning,
            StreamKind::Content => self.content,
            StreamKind::Idle => return,
        };
        let color = match kind {
            StreamKind::Reasoning => ANSI_DARK_GRAY,
            _ => "",
        };
        let label = match kind {
            StreamKind::Reasoning => "reasoning",
            StreamKind::Content => "content",
            StreamKind::Idle => "",
        };

        match kind {
            StreamKind::Reasoning => state.reasoning_chars += text.len(),
            StreamKind::Content => state.content_chars += text.len(),
            StreamKind::Idle => {}
        }

        if mode == StreamPrint::Off {
            return;
        }

        let shown = match kind {
            StreamKind::Reasoning => state.reasoning_shown,
            StreamKind::Content => state.content_shown,
            StreamKind::Idle => 0,
        };
        let total = match kind {
            StreamKind::Reasoning => state.reasoning_chars,
            StreamKind::Content => state.content_chars,
            StreamKind::Idle => 0,
        };

        if mode == StreamPrint::Compact && shown >= self.compact_limit {
            if total % 4096 < text.len() {
                self.finish_stream_line(&mut state);
                println!(
                    "  [{}]    · {} {}",
                    self.elapsed_str(),
                    label,
                    self.paint(
                        ANSI_DIM,
                        &format!("(+{} chars, total {})", text.len(), total)
                    )
                );
            }
            return;
        }

        if state.kind != kind {
            if state.kind != StreamKind::Idle {
                println!();
            }
            print!(
                "  [{}]    · {} ",
                self.elapsed_str(),
                self.paint(ANSI_DIM, label)
            );
            state.kind = kind;
        }

        let take = if mode == StreamPrint::Compact {
            self.compact_limit.saturating_sub(shown).min(text.len())
        } else {
            text.len()
        };
        // `take` is a byte budget. Falling back to the whole `text` when the
        // cut is mid-character would dump a huge HTML payload in compact mode.
        let slice = utf8_prefix(text, take);
        if self.color && !color.is_empty() {
            print!("{}{}{}", color, slice, ANSI_RESET);
        } else {
            print!("{}", slice);
        }
        match kind {
            StreamKind::Reasoning => state.reasoning_shown += slice.len(),
            StreamKind::Content => state.content_shown += slice.len(),
            StreamKind::Idle => {}
        }
        if mode == StreamPrint::Compact && slice.len() < text.len() {
            print!(
                "{}",
                self.paint(
                    ANSI_DIM,
                    " … (truncated; set CLOUDLLM_STREAM_CONTENT=full or CLOUDLLM_STREAM_REASONING=full)"
                )
            );
        }
        let _ = io::stdout().flush();
    }

    fn reset_stream_counts(&self) {
        let mut state = self.lock_state();
        self.finish_stream_line(&mut state);
        *state = PrinterState::default();
    }

    /// First `max_chars` of `text`, with `...` if truncated.
    pub fn preview(text: &str, max_chars: usize) -> String {
        let mut it = text.chars();
        let taken: String = it.by_ref().take(max_chars).collect();
        if it.next().is_some() {
            format!("{}...", taken)
        } else {
            taken
        }
    }
}

#[async_trait]
impl EventHandler for LiveConsoleHandler {
    async fn on_agent_event(&self, event: &AgentEvent) {
        match event {
            AgentEvent::SendStarted {
                agent_name,
                message_preview,
                ..
            } => {
                self.reset_stream_counts();
                println!(
                    "  [{}] >> {} thinking... ({})",
                    self.elapsed_str(),
                    agent_name,
                    Self::preview(message_preview, 80)
                );
            }
            AgentEvent::LLMCallStarted {
                agent_name,
                iteration,
                ..
            } => {
                self.reset_stream_counts();
                println!(
                    "  [{}]    {} sending to LLM (round {}) — waiting for first token...",
                    self.elapsed_str(),
                    agent_name,
                    iteration
                );
            }
            AgentEvent::LLMReasoningDelta { text, .. } => {
                self.write_stream(StreamKind::Reasoning, text);
            }
            AgentEvent::LLMContentDelta { text, .. } => {
                self.write_stream(StreamKind::Content, text);
            }
            AgentEvent::LLMWaiting {
                agent_name,
                iteration,
                elapsed_secs,
                seconds_since_last_token,
                phase,
                ..
            } => {
                {
                    let mut state = self.lock_state();
                    self.finish_stream_line(&mut state);
                }
                let hint = if phase == "waiting for first token" {
                    " (model may be reasoning privately — not all providers stream traces)"
                } else {
                    ""
                };
                println!(
                    "  [{}]    {} still working — {} on LLM round {}, {}s since last token (phase: {}){}",
                    self.elapsed_str(),
                    agent_name,
                    format_duration(*elapsed_secs),
                    iteration,
                    seconds_since_last_token,
                    self.paint(ANSI_YELLOW, phase),
                    self.paint(ANSI_DIM, hint)
                );
            }
            AgentEvent::LLMCallCompleted {
                agent_name,
                iteration,
                tokens_used,
                response_length,
                ..
            } => {
                {
                    let mut state = self.lock_state();
                    self.finish_stream_line(&mut state);
                    if state.reasoning_chars > 0 || state.content_chars > 0 {
                        println!(
                            "  [{}]    {} stream totals: {} reasoning chars, {} content chars",
                            self.elapsed_str(),
                            agent_name,
                            state.reasoning_chars,
                            state.content_chars
                        );
                    }
                }
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
            AgentEvent::SendCompleted {
                agent_name,
                tokens_used,
                response_length,
                tool_calls_made,
                ..
            } => {
                {
                    let mut state = self.lock_state();
                    self.finish_stream_line(&mut state);
                }
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
                {
                    let mut state = self.lock_state();
                    self.finish_stream_line(&mut state);
                }
                let params = serde_json::to_string(parameters).unwrap_or_default();
                println!(
                    "  [{}]    {} calling tool '{}' (iter {}) {}",
                    self.elapsed_str(),
                    agent_name,
                    self.paint(ANSI_CYAN, tool_name),
                    iteration,
                    Self::preview(&params, 160)
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
                {
                    let mut state = self.lock_state();
                    self.finish_stream_line(&mut state);
                }
                if *success {
                    let preview = result
                        .as_ref()
                        .map(|r| {
                            let s = serde_json::to_string(r).unwrap_or_default();
                            Self::preview(&s, 160)
                        })
                        .unwrap_or_default();
                    println!(
                        "  [{}]    {} tool '{}' succeeded [iter {}] → {}",
                        self.elapsed_str(),
                        agent_name,
                        self.paint(ANSI_CYAN, tool_name),
                        iteration,
                        preview
                    );
                } else {
                    println!(
                        "  [{}]    {} tool '{}' {} [iter {}]: {}",
                        self.elapsed_str(),
                        agent_name,
                        tool_name,
                        self.paint(ANSI_RED, "FAILED"),
                        iteration,
                        error.as_deref().unwrap_or("unknown")
                    );
                }
            }
            AgentEvent::ToolMaxIterationsReached { agent_name, .. } => {
                println!(
                    "  [{}]    {} {}",
                    self.elapsed_str(),
                    agent_name,
                    self.paint(ANSI_RED, "hit max tool iterations")
                );
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
                println!(
                    "  [{}] >> planner {} starting ({})",
                    self.elapsed_str(),
                    Self::preview(plan_id, 8),
                    Self::preview(message_preview, 80)
                );
            }
            PlannerEvent::LLMCallStarted { iteration, .. } => {
                self.reset_stream_counts();
                println!(
                    "  [{}]    planner sending to LLM (round {}) — waiting for first token...",
                    self.elapsed_str(),
                    iteration
                );
            }
            PlannerEvent::LLMReasoningDelta { text, .. } => {
                self.write_stream(StreamKind::Reasoning, text);
            }
            PlannerEvent::PartialOutputChunk { chunk, .. } => {
                self.write_stream(StreamKind::Content, chunk);
            }
            PlannerEvent::LLMWaiting {
                iteration,
                elapsed_secs,
                seconds_since_last_token,
                phase,
                ..
            } => {
                {
                    let mut state = self.lock_state();
                    self.finish_stream_line(&mut state);
                }
                println!(
                    "  [{}]    planner still working — {} on LLM round {}, {}s since last token (phase: {})",
                    self.elapsed_str(),
                    format_duration(*elapsed_secs),
                    iteration,
                    seconds_since_last_token,
                    self.paint(ANSI_YELLOW, phase)
                );
            }
            PlannerEvent::LLMCallCompleted {
                iteration,
                response_length,
                ..
            } => {
                {
                    let mut state = self.lock_state();
                    self.finish_stream_line(&mut state);
                }
                println!(
                    "  [{}]    planner LLM round {} complete ({} chars)",
                    self.elapsed_str(),
                    iteration,
                    response_length
                );
            }
            PlannerEvent::TurnCompleted {
                response_length,
                tool_calls_made,
                ..
            } => {
                println!(
                    "  [{}] << planner done ({} chars, {} tool calls)",
                    self.elapsed_str(),
                    response_length,
                    tool_calls_made
                );
            }
            PlannerEvent::TurnErrored { error, .. } => {
                println!(
                    "  [{}] !!! planner {}",
                    self.elapsed_str(),
                    self.paint(ANSI_RED, error)
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
            OrchestrationEvent::RoundStarted { round, .. } => {
                println!("  [{}] -- round {} --", self.elapsed_str(), round);
            }
            OrchestrationEvent::AgentSelected {
                agent_name, reason, ..
            } => {
                println!("  [{}] -> {} ({})", self.elapsed_str(), agent_name, reason);
            }
            OrchestrationEvent::TaskClaimed {
                agent_name,
                task_id,
                ..
            } => {
                println!(
                    "  [{}]    {} claimed task {}",
                    self.elapsed_str(),
                    agent_name,
                    task_id
                );
            }
            OrchestrationEvent::TaskCompleted {
                agent_name,
                task_id,
                ..
            } => {
                println!(
                    "  [{}]    {} completed task {}",
                    self.elapsed_str(),
                    agent_name,
                    task_id
                );
            }
            OrchestrationEvent::AgentFailed {
                agent_name, error, ..
            } => {
                println!(
                    "  [{}] !!! {} FAILED: {}",
                    self.elapsed_str(),
                    agent_name,
                    self.paint(ANSI_RED, error)
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

fn abort_embedded_mentisdb(
    dir: &std::path::Path,
    chain_key: &str,
    error: &str,
) -> Box<dyn std::error::Error + Send + Sync> {
    eprintln!();
    eprintln!("❌ MentisDB could not be opened — aborting before any LLM calls.");
    eprintln!("   This example is self-contained: it uses the MentisDB *library*");
    eprintln!("   and writes local files. It does NOT start or connect to mentisdbd.");
    eprintln!("   Directory : {}", dir.display());
    eprintln!("   Chain key : {}", chain_key);
    eprintln!("   Error     : {}", error);
    eprintln!();
    eprintln!("   If this is a lock error, another process already has the chain open.");
    eprintln!("   Otherwise check that the working directory is writable.");
    eprintln!();
    format!(
        "embedded MentisDB open failed (dir={}, chain={}): {}",
        dir.display(),
        chain_key,
        error
    )
    .into()
}

fn env_raw(name: &str) -> Option<String> {
    std::env::var(name)
        .ok()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
}

fn print_knob(name: &str, set: Option<String>, default_label: &str, hint: &str) {
    match set {
        Some(value) => {
            println!("    {:<28} = {}  ({})", name, value, hint);
        }
        None => {
            println!("    {:<28} = (unset, {})  ({})", name, default_label, hint);
        }
    }
}

/// Take up to `max_bytes` from `text` without splitting a UTF-8 character.
pub fn utf8_prefix(text: &str, max_bytes: usize) -> &str {
    if max_bytes >= text.len() {
        return text;
    }
    let mut end = max_bytes;
    while end > 0 && !text.is_char_boundary(end) {
        end -= 1;
    }
    &text[..end]
}

/// Format a duration as `9s` or `1m 15s`.
pub fn format_duration(secs: u64) -> String {
    if secs < 60 {
        format!("{}s", secs)
    } else {
        format!("{}m {:02}s", secs / 60, secs % 60)
    }
}
