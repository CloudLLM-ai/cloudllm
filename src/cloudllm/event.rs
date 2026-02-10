//! Agent and Orchestration event system.
//!
//! Provides a callback-based observability layer for agents and orchestrations.
//! Implement [`EventHandler`] to receive real-time notifications about:
//!
//! - **LLM round-trips**: When each agent sends to and receives from its LLM
//! - **Tool operations**: Tool call detection, execution outcomes, iteration limits
//! - **ThoughtChain**: Thought commits to persistent memory
//! - **Tool mutations**: Protocol additions and removals at runtime
//! - **Agent lifecycle**: Fork, system prompt changes, message injection
//! - **Orchestration lifecycle**: Run start/end, round boundaries, agent selection
//! - **Mode-specific**: Debate convergence checks, RALPH iteration/task progress
//!
//! # Architecture
//!
//! Events flow through a single [`EventHandler`] trait with two methods:
//! - [`on_agent_event`](EventHandler::on_agent_event) — receives [`AgentEvent`]s from individual agents
//! - [`on_orchestration_event`](EventHandler::on_orchestration_event) — receives [`OrchestrationEvent`]s from the orchestration engine
//!
//! Both methods have default no-op implementations, so you only override what
//! you care about. The handler is wrapped in `Arc<dyn EventHandler>` and shared
//! across agents — when registered on an [`Orchestration`](crate::orchestration::Orchestration)
//! via [`with_event_handler`](crate::orchestration::Orchestration::with_event_handler),
//! it is automatically propagated to every agent added via
//! [`add_agent`](crate::orchestration::Orchestration::add_agent).
//!
//! # Example
//!
//! ```rust,no_run
//! use cloudllm::event::{AgentEvent, EventHandler, OrchestrationEvent};
//! use async_trait::async_trait;
//! use std::sync::Arc;
//!
//! struct MyHandler;
//!
//! #[async_trait]
//! impl EventHandler for MyHandler {
//!     async fn on_agent_event(&self, event: &AgentEvent) {
//!         match event {
//!             AgentEvent::LLMCallStarted { agent_name, iteration, .. } => {
//!                 println!("{} calling LLM (round {})...", agent_name, iteration);
//!             }
//!             AgentEvent::LLMCallCompleted { agent_name, iteration, response_length, .. } => {
//!                 println!("{} LLM round {} done ({} chars)", agent_name, iteration, response_length);
//!             }
//!             _ => {}
//!         }
//!     }
//!     async fn on_orchestration_event(&self, event: &OrchestrationEvent) {
//!         println!("Orchestration: {:?}", event);
//!     }
//! }
//! ```

use crate::client_wrapper::TokenUsage;
use crate::cloudllm::thought_chain::ThoughtType;
use async_trait::async_trait;

/// Events emitted by an [`Agent`](crate::Agent) during its lifecycle.
///
/// Every variant carries `agent_id` and `agent_name` so handlers can identify
/// the source agent without external state. Events are emitted from within
/// [`Agent::send`](crate::Agent::send) and
/// [`Agent::generate_with_tokens`](crate::Agent::generate_with_tokens) as
/// well as from mutation methods like `commit()`, `add_protocol()`, `fork()`, etc.
///
/// # Event Flow (during a typical `send()` call)
///
/// ```text
/// SendStarted
///   └─ LLMCallStarted { iteration: 1 }
///   └─ LLMCallCompleted { iteration: 1 }
///   └─ (if tool call detected in response)
///       ├─ ToolCallDetected { iteration: 1 }
///       ├─ ToolExecutionCompleted { iteration: 1 }
///       ├─ LLMCallStarted { iteration: 2 }
///       └─ LLMCallCompleted { iteration: 2 }
///   └─ (loop continues until no tool call or max iterations)
/// SendCompleted
/// ```
#[derive(Debug, Clone)]
pub enum AgentEvent {
    // ── Generation lifecycle ──────────────────────────────────────────────

    /// Fired at the start of [`Agent::send`](crate::Agent::send) or
    /// [`Agent::generate_with_tokens`](crate::Agent::generate_with_tokens).
    ///
    /// Use this to log when an agent begins working on a prompt.
    SendStarted {
        /// Stable identifier of the agent (e.g. `"game-architect"`).
        agent_id: String,
        /// Human-readable display name (e.g. `"Game Architect"`).
        agent_name: String,
        /// First ~120 characters of the user message, useful for logging.
        message_preview: String,
    },

    /// Fired when `send()` or `generate_with_tokens()` returns successfully.
    ///
    /// This is the bookend to [`SendStarted`](AgentEvent::SendStarted). The
    /// `tool_calls_made` field tells you how many tool iterations occurred
    /// within this generation cycle.
    SendCompleted {
        /// Stable identifier of the agent.
        agent_id: String,
        /// Human-readable display name.
        agent_name: String,
        /// Cumulative token usage across all LLM calls in this generation,
        /// or `None` if the provider did not report usage.
        tokens_used: Option<TokenUsage>,
        /// Number of tool calls that were executed during this generation.
        /// Zero means the LLM responded without requesting any tools.
        tool_calls_made: usize,
        /// Character length of the final response text.
        response_length: usize,
    },

    /// Fired **before** each LLM round-trip inside the tool loop.
    ///
    /// Iteration 1 is the initial LLM call. Subsequent iterations are
    /// follow-up calls after tool results have been injected. Pair with
    /// [`LLMCallCompleted`](AgentEvent::LLMCallCompleted) to measure
    /// per-call latency.
    LLMCallStarted {
        /// Stable identifier of the agent.
        agent_id: String,
        /// Human-readable display name.
        agent_name: String,
        /// 1-based iteration counter (1 = initial call, 2+ = tool follow-ups).
        iteration: usize,
    },

    /// Fired **after** each LLM round-trip completes.
    ///
    /// Use the time delta between `LLMCallStarted` and this event to
    /// identify slow LLM responses (the main source of latency in
    /// orchestrations).
    LLMCallCompleted {
        /// Stable identifier of the agent.
        agent_id: String,
        /// Human-readable display name.
        agent_name: String,
        /// 1-based iteration counter matching the corresponding `LLMCallStarted`.
        iteration: usize,
        /// Cumulative token usage up to and including this call.
        tokens_used: Option<TokenUsage>,
        /// Character length of this specific LLM response.
        response_length: usize,
    },

    // ── Tool operations ──────────────────────────────────────────────────

    /// A tool call was parsed from the LLM response.
    ///
    /// Emitted after the agent's `parse_tool_call()` successfully extracts
    /// a `{"tool_call": {"name": "...", "parameters": {...}}}` JSON fragment
    /// from the LLM output. The `parameters` field contains the raw JSON
    /// arguments that will be passed to the tool protocol.
    ToolCallDetected {
        /// Stable identifier of the agent.
        agent_id: String,
        /// Human-readable display name.
        agent_name: String,
        /// Name of the tool being invoked (e.g. `"memory"`, `"calculator"`).
        tool_name: String,
        /// Raw JSON parameters extracted from the LLM's tool call request.
        parameters: serde_json::Value,
        /// 1-based tool iteration (1 = first tool call in this generation).
        iteration: usize,
    },

    /// A tool finished executing (success or failure).
    ///
    /// Emitted after the `ToolRegistry::execute_tool` call returns. Check
    /// `success` to determine the outcome; on failure `error` contains the
    /// error message.
    ToolExecutionCompleted {
        /// Stable identifier of the agent.
        agent_id: String,
        /// Human-readable display name.
        agent_name: String,
        /// Name of the tool that was executed.
        tool_name: String,
        /// The parameters that were passed to the tool (same as in `ToolCallDetected`).
        parameters: serde_json::Value,
        /// `true` if the tool executed without error.
        success: bool,
        /// Error message if the tool failed, `None` on success.
        error: Option<String>,
        /// 1-based tool iteration matching the corresponding `ToolCallDetected`.
        iteration: usize,
    },

    /// The tool loop hit its iteration cap (currently 5).
    ///
    /// Emitted when the agent's internal tool-call loop reaches its maximum
    /// number of iterations. The response will include a
    /// `"[Warning: Maximum tool iterations reached]"` suffix. This typically
    /// indicates a misbehaving LLM that keeps requesting tool calls in a loop.
    ToolMaxIterationsReached {
        /// Stable identifier of the agent.
        agent_id: String,
        /// Human-readable display name.
        agent_name: String,
    },

    // ── ThoughtChain ─────────────────────────────────────────────────────

    /// A thought was appended to the agent's [`ThoughtChain`](crate::thought_chain::ThoughtChain).
    ///
    /// Emitted after a successful [`Agent::commit`](crate::Agent::commit) call.
    ThoughtCommitted {
        /// Stable identifier of the agent.
        agent_id: String,
        /// Human-readable display name.
        agent_name: String,
        /// Type of thought that was committed (Finding, Decision, Compression, etc.).
        thought_type: ThoughtType,
    },

    // ── Tool mutations ───────────────────────────────────────────────────

    /// A new protocol was added to the agent's tool registry at runtime.
    ///
    /// Emitted after a successful [`Agent::add_protocol`](crate::Agent::add_protocol) call.
    ProtocolAdded {
        /// Stable identifier of the agent.
        agent_id: String,
        /// Human-readable display name.
        agent_name: String,
        /// Name under which the protocol was registered (e.g. `"memory"`, `"custom"`).
        protocol_name: String,
    },

    /// A protocol was removed from the agent's tool registry at runtime.
    ///
    /// Emitted after [`Agent::remove_protocol`](crate::Agent::remove_protocol) completes.
    ProtocolRemoved {
        /// Stable identifier of the agent.
        agent_id: String,
        /// Human-readable display name.
        agent_name: String,
        /// Name of the protocol that was removed.
        protocol_name: String,
    },

    // ── Session / Hub-routing ────────────────────────────────────────────

    /// The agent's system prompt was set or replaced.
    ///
    /// Emitted from [`Agent::set_system_prompt`](crate::Agent::set_system_prompt).
    /// In orchestration, this fires once per agent during the `setup_agent_prompts()`
    /// phase at the start of each mode's execution.
    SystemPromptSet {
        /// Stable identifier of the agent.
        agent_id: String,
        /// Human-readable display name.
        agent_name: String,
    },

    /// A message was injected into the agent's session history.
    ///
    /// Emitted from [`Agent::receive_message`](crate::Agent::receive_message).
    /// In orchestration, this fires when the hub routes another agent's response
    /// into this agent's context before its turn.
    MessageReceived {
        /// Stable identifier of the agent.
        agent_id: String,
        /// Human-readable display name.
        agent_name: String,
    },

    // ── Lifecycle ────────────────────────────────────────────────────────

    /// The agent was forked via [`Agent::fork`](crate::Agent::fork).
    ///
    /// The forked agent shares tools and thought chain via `Arc` but has a
    /// fresh, empty session. Used internally by Parallel and Hierarchical modes.
    Forked {
        /// Stable identifier of the agent (same for original and fork).
        agent_id: String,
        /// Human-readable display name.
        agent_name: String,
    },

    /// The agent was forked with context carried forward via
    /// [`Agent::fork_with_context`](crate::Agent::fork_with_context).
    ///
    /// Unlike [`Forked`](AgentEvent::Forked), the new agent's session contains
    /// a copy of the original's system prompt and conversation history.
    ForkedWithContext {
        /// Stable identifier of the agent (same for original and fork).
        agent_id: String,
        /// Human-readable display name.
        agent_name: String,
    },
}

/// Events emitted by an [`Orchestration`](crate::orchestration::Orchestration)
/// during a [`run()`](crate::orchestration::Orchestration::run) call.
///
/// Every variant carries an `orchestration_id` for identification. These events
/// provide coarse-grained progress visibility into the orchestration lifecycle,
/// while the [`AgentEvent`]s emitted by individual agents provide fine-grained
/// visibility into each agent's LLM calls and tool usage.
///
/// # Event Flow (RoundRobin example with 2 agents, 1 round)
///
/// ```text
/// RunStarted { mode: "RoundRobin", agent_count: 2 }
///   └─ RoundStarted { round: 1 }
///       ├─ AgentSelected { agent: "Alice", reason: "RoundRobin turn" }
///       ├─ AgentResponded { agent: "Alice", response_length: 1234 }
///       ├─ AgentSelected { agent: "Bob", reason: "RoundRobin turn" }
///       └─ AgentResponded { agent: "Bob", response_length: 567 }
///   └─ RoundCompleted { round: 1 }
/// RunCompleted { rounds: 1, is_complete: true }
/// ```
///
/// # Event Flow (RALPH example)
///
/// ```text
/// RunStarted { mode: "Ralph", agent_count: 4 }
///   └─ RalphIterationStarted { iteration: 1, tasks_completed: 0, tasks_total: 10 }
///   └─ RoundStarted { round: 1 }
///       ├─ AgentResponded { agent: "Architect" }
///       ├─ RalphTaskCompleted { agent: "Architect", task_ids: ["html"], progress: 1/10 }
///       ├─ AgentResponded { agent: "Programmer" }
///       └─ RalphTaskCompleted { agent: "Programmer", task_ids: ["game_loop", "paddle"], progress: 3/10 }
///   └─ RoundCompleted { round: 1 }
///   └─ RalphIterationStarted { iteration: 2, tasks_completed: 3, tasks_total: 10 }
///   ...
/// RunCompleted { rounds: N, is_complete: true/false }
/// ```
#[derive(Debug, Clone)]
pub enum OrchestrationEvent {
    /// The orchestration run has started.
    ///
    /// Emitted once at the top of [`Orchestration::run`](crate::orchestration::Orchestration::run),
    /// before any agents are called. Use this to display a banner or initialise
    /// timing state.
    RunStarted {
        /// Stable identifier of the orchestration.
        orchestration_id: String,
        /// Human-readable orchestration name.
        orchestration_name: String,
        /// Active mode name (e.g. `"Parallel"`, `"RoundRobin"`, `"Ralph"`).
        mode: String,
        /// Number of agents registered when the run started.
        agent_count: usize,
    },

    /// The orchestration run has completed (successfully or after hitting limits).
    ///
    /// Emitted once at the end of `run()`, after all rounds/iterations are done.
    /// Pair with [`RunStarted`](OrchestrationEvent::RunStarted) to measure total
    /// orchestration wall-clock time.
    RunCompleted {
        /// Stable identifier of the orchestration.
        orchestration_id: String,
        /// Human-readable orchestration name.
        orchestration_name: String,
        /// Number of rounds/iterations actually executed.
        rounds: usize,
        /// Approximate total tokens consumed across all agents and rounds.
        total_tokens: usize,
        /// Whether the orchestration's completion condition was met.
        /// `false` for RALPH if not all tasks were finished.
        is_complete: bool,
    },

    /// A new round (or iteration) is beginning.
    ///
    /// Emitted at the top of each round loop in every mode. In Hierarchical
    /// mode, each layer counts as a "round".
    RoundStarted {
        /// Stable identifier of the orchestration.
        orchestration_id: String,
        /// 1-based round number.
        round: usize,
    },

    /// A round (or iteration) has completed.
    ///
    /// Emitted at the end of each round loop, after all agents in that round
    /// have responded (or failed).
    RoundCompleted {
        /// Stable identifier of the orchestration.
        orchestration_id: String,
        /// 1-based round number matching the corresponding `RoundStarted`.
        round: usize,
    },

    /// An agent was selected to respond next.
    ///
    /// Emitted in modes that have explicit agent selection: RoundRobin (before
    /// each agent's turn), Moderated (after the moderator picks an expert),
    /// Hierarchical (before each layer's agents), and Debate (before each
    /// debater's turn). The `reason` field describes why this agent was chosen.
    AgentSelected {
        /// Stable identifier of the orchestration.
        orchestration_id: String,
        /// Stable identifier of the selected agent.
        agent_id: String,
        /// Human-readable display name of the selected agent.
        agent_name: String,
        /// Human-readable explanation (e.g. `"RoundRobin turn"`, `"Hierarchical layer 2"`).
        reason: String,
    },

    /// An agent responded successfully to its prompt.
    ///
    /// Emitted after an agent's `send()` call returns `Ok`. The `tokens_used`
    /// field reflects that agent's token consumption for this single response.
    AgentResponded {
        /// Stable identifier of the orchestration.
        orchestration_id: String,
        /// Stable identifier of the responding agent.
        agent_id: String,
        /// Human-readable display name.
        agent_name: String,
        /// Token usage reported by the agent for this response.
        tokens_used: Option<TokenUsage>,
        /// Character length of the agent's response.
        response_length: usize,
    },

    /// An agent encountered an error during its `send()` call.
    ///
    /// The agent's response is lost for this round, but the orchestration
    /// continues with remaining agents. Previously these errors were bare
    /// `eprintln!()` calls; now they surface as structured events.
    AgentFailed {
        /// Stable identifier of the orchestration.
        orchestration_id: String,
        /// Stable identifier of the agent that failed.
        agent_id: String,
        /// Human-readable display name.
        agent_name: String,
        /// The error message (from the `Box<dyn Error>` chain).
        error: String,
    },

    /// Convergence was checked at the end of a Debate round.
    ///
    /// The `score` is the average Jaccard similarity between corresponding
    /// agent messages in the current vs. previous round. When `score >= threshold`,
    /// the debate terminates early.
    ConvergenceChecked {
        /// Stable identifier of the orchestration.
        orchestration_id: String,
        /// 1-based round in which convergence was checked.
        round: usize,
        /// Calculated Jaccard similarity score (`0.0..=1.0`).
        score: f32,
        /// The threshold that `score` must meet or exceed for convergence.
        threshold: f32,
        /// `true` if `score >= threshold` (debate will terminate after this round).
        converged: bool,
    },

    /// A RALPH iteration is starting.
    ///
    /// Emitted at the top of each RALPH iteration loop, before any agents are
    /// called for that iteration. Use `tasks_completed` / `tasks_total` to
    /// display a progress indicator.
    RalphIterationStarted {
        /// Stable identifier of the orchestration.
        orchestration_id: String,
        /// 1-based iteration number.
        iteration: usize,
        /// Maximum number of iterations configured for this RALPH run.
        max_iterations: usize,
        /// Number of PRD tasks completed so far (before this iteration).
        tasks_completed: usize,
        /// Total number of PRD tasks in the checklist.
        tasks_total: usize,
    },

    /// One or more RALPH tasks were completed by an agent.
    ///
    /// Emitted after parsing `[TASK_COMPLETE:id]` markers from an agent's
    /// response. Only includes task IDs that match valid PRD task IDs and
    /// were not previously completed. An agent may complete multiple tasks
    /// in a single response.
    RalphTaskCompleted {
        /// Stable identifier of the orchestration.
        orchestration_id: String,
        /// Stable identifier of the agent that completed the task(s).
        agent_id: String,
        /// Human-readable display name.
        agent_name: String,
        /// List of newly completed task IDs from this response.
        task_ids: Vec<String>,
        /// Total number of tasks now completed (including these new ones).
        tasks_completed_total: usize,
        /// Total number of PRD tasks in the checklist.
        tasks_total: usize,
    },
}

/// Trait for receiving agent and orchestration events.
///
/// Both methods have **default no-op implementations**, so you only need to
/// override the events you care about. For example, if you only want
/// orchestration-level progress, implement only `on_orchestration_event`.
///
/// # Thread Safety
///
/// The `Send + Sync` bound allows the handler to be shared across agents
/// and tokio tasks via `Arc<dyn EventHandler>`. Make sure any internal state
/// uses appropriate synchronization (e.g., `AtomicUsize`, `Mutex`).
///
/// # Registration
///
/// - **On an Agent**: [`Agent::with_event_handler`](crate::Agent::with_event_handler)
///   (builder) or [`Agent::set_event_handler`](crate::Agent::set_event_handler) (runtime).
/// - **On an Orchestration**: [`Orchestration::with_event_handler`](crate::orchestration::Orchestration::with_event_handler).
///   The handler is **automatically propagated** to every agent added via
///   [`add_agent`](crate::orchestration::Orchestration::add_agent), giving you
///   a unified stream of both agent-level and orchestration-level events.
///
/// # Example: Minimal Logger
///
/// ```rust,no_run
/// use cloudllm::event::{AgentEvent, EventHandler, OrchestrationEvent};
/// use async_trait::async_trait;
///
/// struct Logger;
///
/// #[async_trait]
/// impl EventHandler for Logger {
///     async fn on_agent_event(&self, event: &AgentEvent) {
///         println!("{:?}", event);
///     }
/// }
/// ```
///
/// # Example: Progress Tracker with Timing
///
/// ```rust,no_run
/// use cloudllm::event::{AgentEvent, EventHandler, OrchestrationEvent};
/// use async_trait::async_trait;
/// use std::time::Instant;
///
/// struct ProgressTracker { start: Instant }
///
/// impl ProgressTracker {
///     fn new() -> Self { Self { start: Instant::now() } }
///     fn elapsed(&self) -> String {
///         let s = self.start.elapsed().as_secs();
///         format!("{:02}:{:02}", s / 60, s % 60)
///     }
/// }
///
/// #[async_trait]
/// impl EventHandler for ProgressTracker {
///     async fn on_agent_event(&self, event: &AgentEvent) {
///         match event {
///             AgentEvent::LLMCallStarted { agent_name, iteration, .. } => {
///                 println!("[{}] {} calling LLM (round {})...", self.elapsed(), agent_name, iteration);
///             }
///             AgentEvent::LLMCallCompleted { agent_name, iteration, response_length, .. } => {
///                 println!("[{}] {} LLM round {} done ({} chars)", self.elapsed(), agent_name, iteration, response_length);
///             }
///             _ => {}
///         }
///     }
///     async fn on_orchestration_event(&self, event: &OrchestrationEvent) {
///         match event {
///             OrchestrationEvent::RunCompleted { rounds, total_tokens, is_complete, .. } => {
///                 println!("[{}] Done! {} rounds, {} tokens, complete={}", self.elapsed(), rounds, total_tokens, is_complete);
///             }
///             _ => {}
///         }
///     }
/// }
/// ```
#[async_trait]
pub trait EventHandler: Send + Sync {
    /// Called when an agent emits an event.
    ///
    /// Receives a reference to the [`AgentEvent`]. The default implementation
    /// is a no-op. Override this to observe LLM calls, tool usage, and
    /// other agent lifecycle events.
    async fn on_agent_event(&self, _event: &AgentEvent) {}

    /// Called when an orchestration emits an event.
    ///
    /// Receives a reference to the [`OrchestrationEvent`]. The default
    /// implementation is a no-op. Override this to observe round boundaries,
    /// agent selection, convergence checks, and RALPH task progress.
    async fn on_orchestration_event(&self, _event: &OrchestrationEvent) {}
}
