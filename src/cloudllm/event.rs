//! Agent and Orchestration event system.
//!
//! Provides observable events emitted during agent generation cycles and
//! orchestration runs. Implement [`EventHandler`] to receive real-time
//! notifications about tool calls, LLM round-trips, task completion, and more.
//!
//! # Example
//!
//! ```rust,no_run
//! use cloudllm::event::{AgentEvent, EventHandler, OrchestrationEvent};
//! use async_trait::async_trait;
//!
//! struct MyHandler;
//!
//! #[async_trait]
//! impl EventHandler for MyHandler {
//!     async fn on_agent_event(&self, event: &AgentEvent) {
//!         println!("Agent event: {:?}", event);
//!     }
//!     async fn on_orchestration_event(&self, event: &OrchestrationEvent) {
//!         println!("Orchestration event: {:?}", event);
//!     }
//! }
//! ```

use crate::client_wrapper::TokenUsage;
use crate::cloudllm::thought_chain::ThoughtType;
use async_trait::async_trait;

/// Events emitted by an [`Agent`](crate::Agent) during its lifecycle.
#[derive(Debug, Clone)]
pub enum AgentEvent {
    // -- Generation lifecycle --
    /// Fired when `send()` or `generate_with_tokens()` begins.
    SendStarted {
        agent_id: String,
        agent_name: String,
        message_preview: String,
    },
    /// Fired when `send()` or `generate_with_tokens()` returns successfully.
    SendCompleted {
        agent_id: String,
        agent_name: String,
        tokens_used: Option<TokenUsage>,
        tool_calls_made: usize,
        response_length: usize,
    },
    /// Fired before each LLM round-trip inside the tool loop.
    LLMCallStarted {
        agent_id: String,
        agent_name: String,
        iteration: usize,
    },
    /// Fired after each LLM round-trip completes.
    LLMCallCompleted {
        agent_id: String,
        agent_name: String,
        iteration: usize,
        tokens_used: Option<TokenUsage>,
        response_length: usize,
    },

    // -- Tool operations --
    /// A tool call was parsed from the LLM response.
    ToolCallDetected {
        agent_id: String,
        agent_name: String,
        tool_name: String,
        parameters: serde_json::Value,
        iteration: usize,
    },
    /// A tool finished executing.
    ToolExecutionCompleted {
        agent_id: String,
        agent_name: String,
        tool_name: String,
        parameters: serde_json::Value,
        success: bool,
        error: Option<String>,
        iteration: usize,
    },
    /// The tool loop hit its iteration cap.
    ToolMaxIterationsReached {
        agent_id: String,
        agent_name: String,
    },

    // -- ThoughtChain --
    /// A thought was appended to the agent's chain.
    ThoughtCommitted {
        agent_id: String,
        agent_name: String,
        thought_type: ThoughtType,
    },

    // -- Tool mutations --
    /// A new protocol was added to the agent's tool registry.
    ProtocolAdded {
        agent_id: String,
        agent_name: String,
        protocol_name: String,
    },
    /// A protocol was removed from the agent's tool registry.
    ProtocolRemoved {
        agent_id: String,
        agent_name: String,
        protocol_name: String,
    },

    // -- Session / Hub-routing --
    /// The agent's system prompt was set or replaced.
    SystemPromptSet {
        agent_id: String,
        agent_name: String,
    },
    /// A message was injected into the agent's session history.
    MessageReceived {
        agent_id: String,
        agent_name: String,
    },

    // -- Lifecycle --
    /// The agent was forked (fresh session).
    Forked {
        agent_id: String,
        agent_name: String,
    },
    /// The agent was forked with context carried forward.
    ForkedWithContext {
        agent_id: String,
        agent_name: String,
    },
}

/// Events emitted by an [`Orchestration`](crate::orchestration::Orchestration) during a run.
#[derive(Debug, Clone)]
pub enum OrchestrationEvent {
    /// The orchestration run has started.
    RunStarted {
        orchestration_id: String,
        orchestration_name: String,
        mode: String,
        agent_count: usize,
    },
    /// The orchestration run has completed.
    RunCompleted {
        orchestration_id: String,
        orchestration_name: String,
        rounds: usize,
        total_tokens: usize,
        is_complete: bool,
    },
    /// A new round/iteration is beginning.
    RoundStarted {
        orchestration_id: String,
        round: usize,
    },
    /// A round/iteration has completed.
    RoundCompleted {
        orchestration_id: String,
        round: usize,
    },
    /// An agent was selected to respond (Moderated, Hierarchical).
    AgentSelected {
        orchestration_id: String,
        agent_id: String,
        agent_name: String,
        reason: String,
    },
    /// An agent responded successfully.
    AgentResponded {
        orchestration_id: String,
        agent_id: String,
        agent_name: String,
        tokens_used: Option<TokenUsage>,
        response_length: usize,
    },
    /// An agent encountered an error.
    AgentFailed {
        orchestration_id: String,
        agent_id: String,
        agent_name: String,
        error: String,
    },
    /// Convergence was checked (Debate mode).
    ConvergenceChecked {
        orchestration_id: String,
        round: usize,
        score: f32,
        threshold: f32,
        converged: bool,
    },
    /// A RALPH iteration is starting.
    RalphIterationStarted {
        orchestration_id: String,
        iteration: usize,
        max_iterations: usize,
        tasks_completed: usize,
        tasks_total: usize,
    },
    /// A RALPH task was completed by an agent.
    RalphTaskCompleted {
        orchestration_id: String,
        agent_id: String,
        agent_name: String,
        task_ids: Vec<String>,
        tasks_completed_total: usize,
        tasks_total: usize,
    },
}

/// Trait for receiving agent and orchestration events.
///
/// Both methods have default no-op implementations so users only need to
/// override the events they care about.
///
/// # Example
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
#[async_trait]
pub trait EventHandler: Send + Sync {
    /// Called when an agent emits an event.
    async fn on_agent_event(&self, _event: &AgentEvent) {}
    /// Called when an orchestration emits an event.
    async fn on_orchestration_event(&self, _event: &OrchestrationEvent) {}
}
