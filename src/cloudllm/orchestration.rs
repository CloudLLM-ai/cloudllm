//! Multi-Agent Orchestration System
//!
//! This module provides abstractions for orchestrating multiple LLM agents in various
//! collaboration patterns. Each agent can have its own LLM provider, expertise, personality,
//! and access to tools (single or multi-protocol).
//!
//! # Collaboration Modes
//!
//! - **Parallel**: All agents process the prompt simultaneously, responses are aggregated
//! - **RoundRobin**: Agents take sequential turns, each building on previous responses
//! - **Moderated**: Agents propose ideas, a moderator synthesizes the final answer
//! - **Hierarchical**: Lead agent coordinates, specialists handle specific aspects
//! - **Debate**: Agents discuss and challenge each other until convergence is reached
//! - **Ralph**: Autonomous iterative loop that works through a PRD task list
//! - **AnthropicAgentTeams** ⭐: Decentralized task-based coordination with no central orchestrator
//!   Agents autonomously discover, claim, and complete tasks from a shared pool via Memory
//!
//! # Architecture
//!
//! ```text
//! Orchestration (orchestration engine)
//!   ├─ EventHandler (shared — receives OrchestrationEvents + AgentEvents)
//!   ├─ Agent 1 (OpenAI GPT-4)
//!   │   ├─ Tools: Local + YouTube MCP Server
//!   │   ├─ Expertise: "Video Analysis"
//!   │   └─ EventHandler ← auto-propagated from Orchestration
//!   │
//!   ├─ Agent 2 (Claude)
//!   │   ├─ Tools: Local + GitHub MCP Server
//!   │   ├─ Expertise: "Code Architecture"
//!   │   └─ EventHandler ← auto-propagated from Orchestration
//!   │
//!   └─ Agent 3 (Grok)
//!       ├─ Tools: Memory Protocol
//!       ├─ Expertise: "System Coordination"
//!       └─ EventHandler ← auto-propagated from Orchestration
//! ```
//!
//! # Event System
//!
//! Attach an [`EventHandler`](crate::event::EventHandler) via
//! [`with_event_handler`](Orchestration::with_event_handler) to receive
//! real-time [`OrchestrationEvent`](crate::event::OrchestrationEvent)s
//! (run lifecycle, round boundaries, RALPH task progress) and
//! [`AgentEvent`](crate::event::AgentEvent)s (LLM calls, tool usage) through
//! a single callback. The handler is automatically propagated to all agents
//! when they are added via [`add_agent`](Orchestration::add_agent). See the
//! [`event`](crate::event) module for the full list of events and examples.
//!
//! # Hub-Routed Sessions
//!
//! Each agent maintains its own [`LLMSession`](crate::LLMSession). The orchestration
//! engine acts as a hub: it selectively routes messages between agents by injecting
//! relevant prior responses into each agent's session via
//! [`receive_message`](crate::Agent::receive_message). Per-agent cursors track
//! which messages each agent has already seen, preventing duplicate injection.
//!
//! # Tool Integration
//!
//! Agents can access tools from multiple protocols simultaneously via
//! [`ToolRegistry`](crate::tool_protocol::ToolRegistry). Share a tool registry
//! across agents using [`with_shared_tools`](crate::Agent::with_shared_tools) —
//! all agents will see the same tools and can coordinate via shared state
//! (e.g., [`Memory`](crate::tools::Memory)).
//!
//! # Example
//!
//! ```rust,no_run
//! use cloudllm::{Agent, orchestration::{Orchestration, OrchestrationMode}};
//! use cloudllm::clients::openai::OpenAIClient;
//! use std::sync::Arc;
//!
//! # async {
//! let agent = Agent::new(
//!     "analyst",
//!     "Technical Analyst",
//!     Arc::new(OpenAIClient::new_with_model_string("key", "gpt-4o"))
//! );
//!
//! let mut orchestration = Orchestration::new("tech-team", "Technical Advisory Orchestration")
//!     .with_mode(OrchestrationMode::Parallel)
//!     .with_max_tokens(8192);
//!
//! orchestration.add_agent(agent).unwrap();
//!
//! let response = orchestration.run("How should we architect this system?", 1).await.unwrap();
//! # };
//! ```

use crate::client_wrapper::Role;
use crate::cloudllm::agent::Agent;
use crate::cloudllm::event::{EventHandler, OrchestrationEvent};
use chrono::{DateTime, Utc};
use std::collections::{HashMap, HashSet};
use std::error::Error;
use std::fmt;
use std::sync::Arc;

/// A task in a RALPH PRD (Product Requirements Document).
///
/// Each `RalphTask` represents a discrete work item that agents iterate on until complete.
/// During orchestration, agents signal completion by including `[TASK_COMPLETE:id]` markers
/// in their responses. The orchestration engine validates the ID against known tasks and
/// tracks progress accordingly.
///
/// # Examples
///
/// ```
/// use cloudllm::orchestration::RalphTask;
///
/// let task = RalphTask::new(
///     "auth_module",
///     "Authentication Module",
///     "Implement JWT-based login with refresh tokens",
/// );
///
/// assert_eq!(task.id, "auth_module");
/// assert_eq!(task.title, "Authentication Module");
/// ```
#[derive(Debug, Clone)]
pub struct RalphTask {
    /// Unique identifier used in `[TASK_COMPLETE:id]` markers.
    ///
    /// Keep IDs short, lowercase, and free of whitespace so agents can emit them
    /// reliably (e.g., `"html_structure"`, `"game_loop"`).
    pub id: String,

    /// Human-readable title displayed in the PRD checklist shown to agents.
    pub title: String,

    /// Detailed description of what the task entails.
    ///
    /// This text is included verbatim in the checklist prompt sent to agents on every
    /// iteration, so it should be specific enough for the agent to know when the work
    /// is done.
    pub description: String,
}

/// A work item in an AnthropicAgentTeams task pool.
///
/// Each `WorkItem` represents a discrete, independently-completable task that agents
/// autonomously discover and claim from a shared Memory pool. Work items are the
/// fundamental unit of coordination in decentralized agent teams.
///
/// # Design Principles
///
/// - **Atomic**: Each task is small enough to complete in one agent turn (~2 minutes)
/// - **Independent**: Minimal dependencies between tasks (agents claim one at a time)
/// - **Clear**: Description and criteria must be understandable by LLMs
/// - **Ordered**: Earlier tasks in the vec are presented first (set task order explicitly)
///
/// # Structure
///
/// Work items have three components:
///
/// | Field | Purpose | Example |
/// |-------|---------|---------|
/// | `id` | Unique task identifier | `"research_nmn"` |
/// | `description` | 1-2 sentence task brief | `"Research phase — NMN+ mechanisms"` |
/// | `acceptance_criteria` | Success condition for agents | `"Summarize NAD+ boosting pathways"` |
///
/// # Memory Storage
///
/// When added to an orchestration, each WorkItem is represented in Memory as:
/// - **Unclaimed**: `teams:<pool_id>:unclaimed:<task_id>` → `description + "\n\nAcceptance criteria: " + acceptance_criteria`
/// - **Claimed**: `teams:<pool_id>:claimed:<task_id>` → `<agent_id>:<timestamp>` (when agent starts work)
/// - **Completed**: `teams:<pool_id>:completed:<task_id>` → `<result>` (when work is done)
///
/// # Best Practices for Effective Tasks
///
/// **Good Task IDs** (What to use):
/// - `research_phase`, `analysis_block`, `write_summary`, `peer_review`
/// - Short, lowercase, underscores between words
/// - Easily parseable by LLMs (no special chars, no spaces)
///
/// **Avoid**:
/// - `task1`, `step_1_a_i` (too generic or complex)
/// - `Research Phase` (uppercase, spaces break LLM parsing)
/// - `task-with-hyphens` (hyphens can confuse tokenizers)
///
/// **Good Descriptions** (What to use):
/// - `"Research phase — identify NAD+ pathways and mitochondrial functions"`
/// - `"Analysis phase — synthesize 3-5 key biological mechanisms"`
/// - 1-2 sentences, specific enough for agent to start work
///
/// **Avoid**:
/// - `"Do research"` (too vague)
/// - `"Research phase. Analysis phase. Write it up. Review it."` (should be separate tasks)
/// - `"Implement X, which depends on Y, which depends on Z"` (hide dependencies)
///
/// **Good Acceptance Criteria** (What to use):
/// - `"Identify 5+ peer-reviewed sources; summarize in 2-3 paragraphs"`
/// - `"Extract 3-5 key biological themes; map relationships between them"`
/// - Specific, measurable, achievable in a single agent turn
///
/// **Avoid**:
/// - `"Do it well"` (not measurable)
/// - `"Complete by next Tuesday"` (time-based, not work-based)
/// - `"Until perfect"` (subjective, unbounded)
///
/// # Example Task Pools
///
/// **Research Project (8 tasks)**:
/// ```text
/// 1. research_sources    — Research phase      — Find 5+ sources
/// 2. analyze_findings    — Analysis phase      — Extract 3-5 themes
/// 3. research_background — Deep research      — Explore historical context
/// 4. analyze_mechanisms  — Mechanism analysis  — Map causal relationships
/// 5. write_summary       — Writing phase      — Draft 2-3 page summary
/// 6. create_outline      — Structure phase    — Organize findings into sections
/// 7. synthesis_report    — Synthesis phase    — Synthesize all findings
/// 8. final_review        — Quality review     — Peer review for accuracy
/// ```
///
/// **Code Review Project (6 tasks)**:
/// ```text
/// 1. review_architecture — Architecture review   — Assess design patterns
/// 2. review_tests        — Test coverage review  — Identify gaps
/// 3. review_performance  — Performance review    — Flag bottlenecks
/// 4. review_security     — Security review       — Identify vulnerabilities
/// 5. summarize_issues    — Issue synthesis      — Consolidate findings
/// 6. final_recommendations — Recommendations    — Suggest top 3-5 changes
/// ```
///
/// # Task Ordering & Dependencies
///
/// When inserting tasks into a `WorkItem` vector, consider:
///
/// - **Phase ordering**: Research → Analysis → Writing → Review
/// - **Logical grouping**: Related tasks nearby (helps agents understand context)
/// - **Complexity gradient**: Simple tasks early (give agents quick wins)
/// - **Explicit dependencies**: If task B needs task A done first, mention it in description
///
/// # Examples
///
/// ```
/// use cloudllm::orchestration::WorkItem;
///
/// // Simple research task
/// let task = WorkItem::new(
///     "research_nmn",
///     "Research phase — NMN+ mechanisms and pathways",
///     "Gather and summarize current scientific literature on NAD+ boosting, \
///      NMN metabolism, mitochondrial function, and sirtuins activation",
/// );
///
/// assert_eq!(task.id, "research_nmn");
/// assert!(task.description.contains("NMN+"));
/// assert!(task.acceptance_criteria.contains("NAD+"));
///
/// // Analysis task
/// let analysis = WorkItem::new(
///     "analyze_longevity",
///     "Analysis phase — longevity effects and clinical outcomes",
///     "Synthesize findings on aging reversal, lifespan extension, \
///      and key biomarkers of rejuvenation (NAD+ levels, cellular senescence)",
/// );
///
/// // Create a small task pool
/// let tasks = vec![task, analysis];
/// assert_eq!(tasks.len(), 2);
/// ```
#[derive(Debug, Clone)]
pub struct WorkItem {
    /// Unique identifier for this task (used in Memory key prefixes).
    ///
    /// - **Format**: lowercase, underscores between words, no spaces or special chars
    /// - **Length**: 10-30 characters (short enough to embed in Memory keys)
    /// - **Semantic**: Descriptive enough that agents understand the work
    /// - **Examples**: `"research_nmn"`, `"analyze_findings"`, `"write_summary"`, `"peer_review"`
    pub id: String,

    /// Human-readable task description shown to agents when they discover available work.
    ///
    /// This text is stored in Memory at `teams:<pool_id>:unclaimed:<task_id>` and
    /// becomes the primary signal for agent task selection. Should be concise but specific.
    ///
    /// - **Length**: 1-2 sentences; 50-150 characters
    /// - **Content**: Task category + brief objective
    /// - **Example**: `"Research phase — identify NAD+ pathways and mechanisms"`
    /// - **Pattern**: `"<Phase> — <Objective>"`
    pub description: String,

    /// Acceptance criteria or success condition for this task.
    ///
    /// Guidance for agents on how to know when the work is complete. Agents use this
    /// to evaluate their own output and decide when to report completion.
    ///
    /// - **Specificity**: Measurable, not subjective ("5+ sources" not "thorough research")
    /// - **Length**: 1-3 sentences; 100-300 characters
    /// - **Constraints**: Achievable in a single agent turn (~2 minutes max)
    /// - **Examples**:
    ///   - `"Identify 5+ peer-reviewed sources; summarize in 2-3 paragraphs"`
    ///   - `"Extract 3-5 key biological mechanisms; map relationships between them"`
    ///   - `"Review code for security vulnerabilities; identify top 3 issues with impact levels"`
    pub acceptance_criteria: String,
}

impl WorkItem {
    /// Create a new work item with the given identifier, description, and acceptance criteria.
    ///
    /// All three parameters accept anything that implements `Into<String>`, so you
    /// can pass `&str`, `String`, or other convertible types.
    ///
    /// # Best Practices
    ///
    /// 1. **Use clear, semantic IDs**: `"research_nmn"` instead of `"task1"`
    /// 2. **Keep descriptions short**: 1-2 sentences, 50-150 chars
    /// 3. **Make criteria measurable**: `"5+ sources"` instead of `"do it well"`
    /// 4. **Order tasks logically**: Research → Analysis → Writing → Review
    /// 5. **Design for independence**: Each task should be claimable and completable alone
    ///
    /// # Examples
    ///
    /// ```
    /// use cloudllm::orchestration::WorkItem;
    ///
    /// // Research task
    /// let research = WorkItem::new(
    ///     "research_nmn",
    ///     "Research phase — NMN+ mechanisms",
    ///     "Summarize NAD+ boosting pathways in 2-3 paragraphs",
    /// );
    /// assert_eq!(research.id, "research_nmn");
    ///
    /// // Analysis task
    /// let analysis = WorkItem::new(
    ///     "analyze_findings",
    ///     "Analysis phase — synthesize research",
    ///     "Extract 3-5 key themes; identify relationships",
    /// );
    ///
    /// // Create task pool
    /// let tasks = vec![research, analysis];
    /// assert_eq!(tasks.len(), 2);
    /// ```
    pub fn new(
        id: impl Into<String>,
        description: impl Into<String>,
        acceptance_criteria: impl Into<String>,
    ) -> Self {
        Self {
            id: id.into(),
            description: description.into(),
            acceptance_criteria: acceptance_criteria.into(),
        }
    }
}

impl RalphTask {
    /// Create a new PRD task with the given identifier, title, and description.
    ///
    /// All three parameters accept anything that implements `Into<String>`, so you
    /// can pass `&str`, `String`, or other convertible types.
    ///
    /// # Examples
    ///
    /// ```
    /// use cloudllm::orchestration::RalphTask;
    ///
    /// // From &str literals
    /// let task = RalphTask::new("db_schema", "Database Schema", "Design the tables");
    ///
    /// // From owned Strings
    /// let id = String::from("api_routes");
    /// let task = RalphTask::new(id, "API Routes", "Implement REST endpoints");
    /// ```
    pub fn new(
        id: impl Into<String>,
        title: impl Into<String>,
        description: impl Into<String>,
    ) -> Self {
        Self {
            id: id.into(),
            title: title.into(),
            description: description.into(),
        }
    }
}

/// Collaboration modes that control how agents interact during [`Orchestration::run`].
///
/// Each variant produces different communication patterns and termination semantics.
/// Choose the mode that best fits your use-case:
///
/// | Mode | Pattern | Termination | Best For |
/// |------|---------|-------------|----------|
/// | `Parallel` | All agents respond at once | Fixed rounds | Independent opinions |
/// | `RoundRobin` | Agents take turns sequentially | Fixed rounds | Sequential refinement |
/// | `Moderated` | Moderator picks who speaks | Fixed rounds | Expert selection |
/// | `Hierarchical` | Layer-by-layer processing | All layers done | Pipeline architectures |
/// | `Debate` | Agents challenge each other | Convergence or max rounds | Consensus building |
/// | `Ralph` | Iterative PRD task loop | All tasks done or max iterations | Checklist completion |
/// | `AnthropicAgentTeams` | Autonomous task claiming | All tasks done or max iterations | Large task pools |
///
/// # Examples
///
/// ```
/// use cloudllm::orchestration::{OrchestrationMode, RalphTask, WorkItem};
///
/// // Simple parallel — every agent answers independently
/// let mode = OrchestrationMode::Parallel;
///
/// // Debate with convergence detection
/// let mode = OrchestrationMode::Debate {
///     max_rounds: 5,
///     convergence_threshold: Some(0.75),
/// };
///
/// // RALPH — agents work through a PRD checklist
/// let mode = OrchestrationMode::Ralph {
///     tasks: vec![
///         RalphTask::new("step1", "Step 1", "Do the first thing"),
///         RalphTask::new("step2", "Step 2", "Do the second thing"),
///     ],
///     max_iterations: 3,
/// };
///
/// // AnthropicAgentTeams — decentralized task coordination
/// let mode = OrchestrationMode::AnthropicAgentTeams {
///     pool_id: "research-2024".to_string(),
///     tasks: vec![
///         WorkItem::new("task1", "Research phase", "Find 5 sources"),
///         WorkItem::new("task2", "Analysis phase", "Synthesize findings"),
///     ],
///     max_iterations: 10,
/// };
/// ```
#[derive(Debug, Clone)]
pub enum OrchestrationMode {
    /// All agents respond in parallel to each prompt.
    ///
    /// Every registered agent receives the same prompt simultaneously via `tokio::spawn`.
    /// The `rounds` parameter passed to [`Orchestration::run`] controls how many
    /// parallel sweeps are executed.
    Parallel,

    /// Agents take turns responding in sequence (round-robin order).
    ///
    /// Each agent sees the accumulated responses from agents that spoke before it
    /// in the current and previous rounds. The `rounds` parameter controls the
    /// number of full cycles through all agents.
    RoundRobin,

    /// A designated moderator agent selects which expert speaks each round.
    ///
    /// The moderator is asked to pick from the available experts. The chosen agent
    /// then responds in context of the ongoing discussion.
    Moderated {
        /// Agent ID of the moderator. Must match an agent registered via
        /// [`Orchestration::add_agent`].
        moderator_id: String,
    },

    /// Layer-by-layer processing where each layer's output feeds the next.
    ///
    /// Agents within the same layer run in parallel. The synthesised output of
    /// one layer becomes the input prompt for the next. Useful for pipelines
    /// like "research -> analyse -> summarise".
    Hierarchical {
        /// Ordered list of layers. Each inner `Vec` contains the agent IDs that
        /// belong to that layer. Layer 0 runs first, then layer 1, and so on.
        layers: Vec<Vec<String>>,
    },

    /// Agents argue and refine positions until their responses converge.
    ///
    /// Convergence is measured via Jaccard similarity on word sets between
    /// consecutive rounds. The loop terminates early when the similarity
    /// score meets or exceeds the threshold.
    Debate {
        /// Upper bound on the number of debate rounds.
        max_rounds: usize,

        /// Jaccard similarity threshold (`0.0..=1.0`) at which the debate is
        /// considered converged. Defaults to `0.75` when `None`.
        convergence_threshold: Option<f32>,
    },

    /// RALPH: Autonomous iterative loop — agents work through a PRD task list
    /// until all tasks are marked complete or `max_iterations` is reached.
    ///
    /// On each iteration every agent sees a checklist of `[x]` / `[ ]` tasks
    /// and is instructed to work on the next incomplete one. Agents signal
    /// completion by including `[TASK_COMPLETE:task_id]` in their response.
    /// The `convergence_score` in the response reflects the fraction of tasks
    /// completed (`0.0..=1.0`).
    Ralph {
        /// The PRD checklist. Each [`RalphTask`] has an `id` that agents
        /// reference in their `[TASK_COMPLETE:id]` markers.
        tasks: Vec<RalphTask>,

        /// Maximum number of full iterations (one pass through all agents per
        /// iteration). Acts as a cost ceiling — the loop may terminate earlier
        /// if all tasks are completed.
        max_iterations: usize,
    },

    /// Anthropic Agent Teams: Decentralized task-based coordination (no central orchestrator).
    ///
    /// Inspired by Anthropic's agent coordination methodology (C compiler project), this mode
    /// enables **autonomous multi-agent cooperation** through a shared Memory pool. Each agent
    /// self-selects work by discovering unclaimed tasks, claiming them atomically via Memory,
    /// completing the work, and reporting results — all without a central coordinator.
    ///
    /// # Key Characteristics
    ///
    /// - **No Central Orchestrator**: Every agent has equal autonomy; no manager or moderator
    /// - **Atomic Task Claiming**: Memory PUT ensures only one agent claims a task (single-threaded)
    /// - **Self-Discovery**: Agents use Memory LIST to find available work
    /// - **Scalable**: Add more agents or tasks without architectural changes
    /// - **Mixed Providers**: Works seamlessly with different LLM providers (OpenAI, Claude, etc.)
    /// - **Event-Driven Progress**: TaskClaimed, TaskCompleted, TaskFailed events for monitoring
    ///
    /// # Memory Coordination Scheme
    ///
    /// Task state is stored in Memory with hierarchical keys:
    ///
    /// ```text
    /// teams:<pool_id>:unclaimed:<task_id>     → task description + criteria (discoverable)
    /// teams:<pool_id>:claimed:<task_id>       → "<agent_id>:<timestamp>" (who's working)
    /// teams:<pool_id>:completed:<task_id>     → "<result_json>" (task finished)
    /// teams:<pool_id>:metadata                → pool configuration + timestamp
    /// teams:<pool_id>:stats                   → progress counters (completed, failed, total)
    /// ```
    ///
    /// # Task Lifecycle
    ///
    /// Each task transitions through states:
    /// 1. **Unclaimed** → Agent discovers it via `LIST teams:<pool_id>:unclaimed:*`
    /// 2. **Claimed** → Agent atomically PUT's to `teams:<pool_id>:claimed:<task_id>`
    /// 3. **In Progress** → Agent works on the task using LLM
    /// 4. **Completed** → Agent PUT's result to `teams:<pool_id>:completed:<task_id>`
    ///
    /// # Runtime Behavior
    ///
    /// **Per Iteration:**
    /// - Each iteration calls all agents sequentially (in registration order)
    /// - Every agent gets a prompt listing available unclaimed tasks
    /// - Agent decides autonomously whether to claim a task or wait
    /// - Agent completes work and reports via simulated Memory operations
    /// - Orchestration detects claimed/completed tasks and emits events
    ///
    /// **Termination:**
    /// - Stops when all tasks are completed (best case)
    /// - Or when `max_iterations` is reached (safety ceiling)
    /// - Estimated runtime: ~1-5 minutes for 8 tasks with 4 agents (depends on LLM latency)
    ///
    /// # Best Practices
    ///
    /// **Task Design:**
    /// - Keep task IDs short, lowercase, underscore-separated (e.g., `research_phase`)
    /// - Write clear descriptions (1-2 sentences) that LLMs can parse
    /// - Define acceptance criteria explicitly (what does "done" mean?)
    /// - Group related tasks into logical phases
    ///
    /// **Agent Mix:**
    /// - Combine specialized agents: researcher, analyst, writer, reviewer
    /// - Mix LLM providers for cost optimization (fast Haiku for easy tasks, smart Sonnet for complex)
    /// - Consider agent count vs. task pool size (4 agents per 8 tasks is ideal)
    ///
    /// **Iteration Tuning:**
    /// - `max_iterations = task_count / agent_count + 2` is a good starting heuristic
    /// - For 8 tasks + 4 agents: `8 / 4 + 2 = 4` iterations
    /// - Increase if tasks are complex or interdependent
    ///
    /// **Monitoring:**
    /// - Attach an EventHandler to track TaskClaimed, TaskCompleted, TaskFailed events
    /// - Watch the convergence_score (0.0 = no progress, 1.0 = all tasks done)
    /// - Log iteration boundaries to diagnose bottlenecks
    ///
    /// # Example: Research Team with 4 Agents and 8 Tasks
    ///
    /// ```rust,no_run
    /// use cloudllm::{Agent, orchestration::{Orchestration, OrchestrationMode, WorkItem}};
    /// use cloudllm::clients::openai::OpenAIClient;
    /// use cloudllm::clients::claude::{ClaudeClient, Model};
    /// use std::sync::Arc;
    ///
    /// # async {
    /// // Define task pool (8 research tasks)
    /// let tasks = vec![
    ///     WorkItem::new("research_nmn", "NMN+ research", "Summarize NAD+ boosting mechanisms"),
    ///     WorkItem::new("analyze_longevity", "Longevity analysis", "Identify 3-5 aging reversal markers"),
    ///     WorkItem::new("research_alzheimers", "Alzheimer's research", "Document amyloid-beta pathology"),
    ///     WorkItem::new("analyze_protection", "Neuroprotection analysis", "Map NAD+ restoration benefits"),
    ///     WorkItem::new("memory_recovery", "Memory recovery research", "Find evidence for cognitive restoration"),
    ///     WorkItem::new("clinical_integration", "Clinical analysis", "Assess therapeutic feasibility"),
    ///     WorkItem::new("synthesis_report", "Report writing", "Synthesize findings into 3-4 page report"),
    ///     WorkItem::new("final_review", "Quality review", "Peer review for accuracy and completeness"),
    /// ];
    ///
    /// // Create mixed-provider agents (4 specialists)
    /// let openai_key = std::env::var("OPENAI_API_KEY").unwrap();
    /// let anthropic_key = std::env::var("ANTHROPIC_API_KEY").unwrap();
    ///
    /// let researcher = Agent::new(
    ///     "researcher",
    ///     "Research Agent (GPT)",
    ///     Arc::new(OpenAIClient::new_with_model_string(&openai_key, "gpt-4o-mini")),
    /// );
    ///
    /// let analyst = Agent::new(
    ///     "analyst",
    ///     "Analysis Agent (Claude Haiku)",
    ///     Arc::new(ClaudeClient::new_with_model_enum(&anthropic_key, Model::ClaudeHaiku45)),
    /// );
    ///
    /// // ... create writer and reviewer agents similarly
    ///
    /// // Create orchestration with AnthropicAgentTeams mode
    /// let mut orchestration = Orchestration::new("research-team", "NMN+ Research Team")
    ///     .with_mode(OrchestrationMode::AnthropicAgentTeams {
    ///         pool_id: "nmn-research-2024".to_string(),
    ///         tasks,
    ///         max_iterations: 4,  // Safety ceiling
    ///     })
    ///     .with_system_context(
    ///         "You are a specialized researcher in a coordinated team. \
    ///          Autonomously claim and complete tasks from the shared pool. \
    ///          Work collaboratively and focus on scientific accuracy."
    ///     )
    ///     .with_max_tokens(4096);
    ///
    /// orchestration.add_agent(researcher)?;
    /// orchestration.add_agent(analyst)?;
    /// // ... add other agents
    ///
    /// // Run the team coordination
    /// let prompt = "Prepare a comprehensive report on NMN+ for Alzheimer's recovery";
    /// let response = orchestration.run(prompt, 1).await?;
    ///
    /// println!("Completed: {}/{} tasks",
    ///     (response.convergence_score.unwrap_or(0.0) * response.messages.len() as f32) as usize,
    ///     tasks.len()
    /// );
    /// # Ok::<(), Box<dyn std::error::Error + Send + Sync>>(())
    /// # }
    /// ```
    ///
    /// # Comparison with RALPH
    ///
    /// Both modes are iterative, but differ in coordination:
    ///
    /// | Aspect | AnthropicAgentTeams | RALPH |
    /// |--------|-------------------|-------|
    /// | **Orchestrator** | None (decentralized) | Implicit (centralized) |
    /// | **Task Selection** | Agent-autonomous via Memory | Orchestration-assigned (checklist) |
    /// | **Coordination** | Memory keys | Completion markers `[TASK_COMPLETE:id]` |
    /// | **Scalability** | Better for large pools | Better for small checklists |
    /// | **Complexity** | More agent autonomy | More orchestration control |
    ///
    /// Choose **AnthropicAgentTeams** if:
    /// - You have >8 tasks (scales better than RALPH's centralized checklist)
    /// - You want agents to self-select work (more natural collaboration)
    /// - You're mixing different LLM providers (simpler coordination)
    /// - You prioritize agent autonomy over orchestration control
    ///
    /// Choose **RALPH** if:
    /// - You have a fixed checklist (<8 items)
    /// - You need strict orchestration control over task assignment
    /// - You want completion markers embedded in agent responses
    ///
    /// # Common Pitfalls & Solutions
    ///
    /// **Pitfall**: Tasks not getting completed
    /// - **Cause**: max_iterations too low or task descriptions unclear
    /// - **Solution**: Increase max_iterations to 2x task count; make descriptions specific
    ///
    /// **Pitfall**: Same agent always claims the same tasks
    /// - **Cause**: Agent preferences in task selection logic (LLM behavior)
    /// - **Solution**: Vary system prompts per agent; randomize task order in descriptions
    ///
    /// **Pitfall**: High token usage / long runtime
    /// - **Cause**: max_iterations too high or agents taking unnecessary turns
    /// - **Solution**: Monitor via events; use fast models (Haiku) for simple tasks
    ///
    /// **Pitfall**: No progress (convergence_score stuck at low value)
    /// - **Cause**: Agents failing to detect task completion or claim work
    /// - **Solution**: Check system prompt clarity; add logging to task descriptions
    AnthropicAgentTeams {
        /// Unique identifier for this task pool (used in Memory key prefixes).
        ///
        /// This ID scopes all Memory operations, allowing multiple independent pools
        /// to coexist. Use descriptive names: `"research-2024"`, `"code-review-batch-1"`.
        pool_id: String,

        /// Tasks to be completed. Each [`WorkItem`] has an id, description, and acceptance criteria.
        ///
        /// Tasks are presented to agents in order each iteration. Agent IDs and descriptions
        /// should be concise and LLM-friendly. Acceptance criteria guide agents on success.
        /// **Recommended**: 5-20 tasks per pool; 4-8 agents.
        tasks: Vec<WorkItem>,

        /// Maximum iterations (one pass through all agents per iteration).
        ///
        /// Acts as a safety ceiling to prevent infinite loops. Typical heuristic:
        /// `max_iterations = (task_count / agent_count) + buffer`.
        /// For 8 tasks and 4 agents: `max_iterations = 3 + 1 = 4`.
        /// Loop terminates early if all tasks are completed.
        /// **Estimated runtime**: ~30 seconds per iteration (with 4 agents, 1-3 sec LLM calls each).
        max_iterations: usize,
    },
}

/// A single message produced during an orchestration discussion.
///
/// Every agent response, user prompt, and system directive flowing through an
/// [`Orchestration`] is captured as an `OrchestrationMessage`. The struct carries
/// identity and timing information alongside the text, making it easy to replay
/// or audit a multi-agent conversation.
///
/// # Examples
///
/// ```
/// use cloudllm::orchestration::OrchestrationMessage;
/// use cloudllm::Role;
///
/// // System / user message (no agent identity)
/// let user_msg = OrchestrationMessage::new(Role::User, "What is 2+2?");
/// assert!(user_msg.agent_id.is_none());
///
/// // Agent message with metadata
/// let agent_msg = OrchestrationMessage::from_agent("calc", "Calculator", "4")
///     .with_metadata("round", "1");
/// assert_eq!(agent_msg.agent_id.as_deref(), Some("calc"));
/// assert_eq!(agent_msg.metadata.get("round").unwrap(), "1");
/// ```
#[derive(Debug, Clone)]
pub struct OrchestrationMessage {
    /// UTC timestamp recorded when the message was created.
    pub timestamp: DateTime<Utc>,

    /// Unique identifier of the agent that produced this message, or `None` for
    /// system / user messages that have no agent origin.
    pub agent_id: Option<String>,

    /// Human-readable display name of the contributing agent, or `None` for
    /// non-agent messages.
    pub agent_name: Option<String>,

    /// Conversation role — typically [`Role::User`] for prompts or
    /// [`Role::Assistant`] for agent responses.
    pub role: Role,

    /// The message body. Stored as `Arc<str>` so cloning messages is cheap.
    pub content: Arc<str>,

    /// Free-form key-value metadata attached to the message.
    ///
    /// Built-in modes populate well-known keys:
    /// - `"round"` / `"iteration"` — the round or iteration number
    /// - `"layer"` — the hierarchical layer index
    /// - `"moderator"` — moderator agent id (Moderated mode)
    /// - `"tasks_completed"` — comma-separated task ids (Ralph mode)
    pub metadata: HashMap<String, String>,
}

impl OrchestrationMessage {
    /// Create a message with the given role and content but no agent identity.
    ///
    /// Use this for user prompts or system directives that originate outside of
    /// any agent.
    ///
    /// # Examples
    ///
    /// ```
    /// use cloudllm::orchestration::OrchestrationMessage;
    /// use cloudllm::Role;
    ///
    /// let msg = OrchestrationMessage::new(Role::User, "Summarise this document");
    /// assert!(msg.agent_id.is_none());
    /// assert!(msg.content.contains("Summarise"));
    /// ```
    pub fn new(role: Role, content: impl Into<String>) -> Self {
        Self {
            timestamp: Utc::now(),
            agent_id: None,
            agent_name: None,
            role,
            content: Arc::from(content.into().as_str()),
            metadata: HashMap::new(),
        }
    }

    /// Create an assistant-role message attributed to a specific agent.
    ///
    /// This is the constructor used internally whenever an agent produces a
    /// response during orchestration.
    ///
    /// # Examples
    ///
    /// ```
    /// use cloudllm::orchestration::OrchestrationMessage;
    ///
    /// let msg = OrchestrationMessage::from_agent(
    ///     "researcher",
    ///     "Research Agent",
    ///     "The capital of France is Paris.",
    /// );
    /// assert_eq!(msg.agent_name.as_deref(), Some("Research Agent"));
    /// ```
    pub fn from_agent(
        agent_id: impl Into<String>,
        agent_name: impl Into<String>,
        content: impl Into<String>,
    ) -> Self {
        Self {
            timestamp: Utc::now(),
            agent_id: Some(agent_id.into()),
            agent_name: Some(agent_name.into()),
            role: Role::Assistant,
            content: Arc::from(content.into().as_str()),
            metadata: HashMap::new(),
        }
    }

    /// Attach a key-value metadata pair to this message (builder pattern).
    ///
    /// Multiple calls can be chained to attach several entries.
    ///
    /// # Examples
    ///
    /// ```
    /// use cloudllm::orchestration::OrchestrationMessage;
    ///
    /// let msg = OrchestrationMessage::from_agent("a1", "Agent 1", "Hello")
    ///     .with_metadata("round", "2")
    ///     .with_metadata("source", "debate");
    ///
    /// assert_eq!(msg.metadata.len(), 2);
    /// assert_eq!(msg.metadata["source"], "debate");
    /// ```
    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }
}

/// The result of an [`Orchestration::run`] call.
///
/// Contains every message produced during the discussion together with summary
/// metrics that let callers assess whether the orchestration reached its goal.
///
/// # Examples
///
/// ```rust,no_run
/// # async {
/// # use cloudllm::orchestration::{Orchestration, OrchestrationMode};
/// # use cloudllm::Agent;
/// # use cloudllm::clients::openai::OpenAIClient;
/// # use std::sync::Arc;
/// # let client = Arc::new(OpenAIClient::new_with_model_string("key", "gpt-4o"));
/// # let agent = Agent::new("a", "A", client);
/// # let mut orch = Orchestration::new("id", "name").with_mode(OrchestrationMode::Parallel);
/// # orch.add_agent(agent).unwrap();
/// let response = orch.run("Hello", 1).await.unwrap();
///
/// println!("Rounds: {}", response.round);
/// println!("Complete: {}", response.is_complete);
/// println!("Tokens: {}", response.total_tokens_used);
///
/// for msg in &response.messages {
///     let who = msg.agent_name.as_deref().unwrap_or("system");
///     println!("[{}]: {}", who, msg.content);
/// }
/// # };
/// ```
#[derive(Debug)]
pub struct OrchestrationResponse {
    /// Every [`OrchestrationMessage`] generated during the discussion, in
    /// chronological order.
    pub messages: Vec<OrchestrationMessage>,

    /// Number of rounds (or iterations) that were actually executed.
    ///
    /// For fixed-round modes (Parallel, RoundRobin, Moderated) this equals the
    /// `rounds` argument. For Debate it may be less than `max_rounds` if
    /// convergence was reached early. For Ralph it reflects the number of
    /// iterations completed before all tasks were done or the cap was hit.
    pub round: usize,

    /// Whether the orchestration reached its natural completion condition.
    ///
    /// - **Parallel / RoundRobin / Moderated / Hierarchical**: always `true`.
    /// - **Debate**: `true` when agents converged *or* `max_rounds` was reached.
    /// - **Ralph**: `true` only when *every* task was marked complete.
    pub is_complete: bool,

    /// Mode-specific progress metric in the range `0.0..=1.0`, or `None` when
    /// the mode does not compute one.
    ///
    /// - **Debate**: Jaccard similarity between the last two rounds of responses.
    /// - **Ralph**: fraction of PRD tasks completed (`completed / total`).
    /// - **Other modes**: `None`.
    pub convergence_score: Option<f32>,

    /// Approximate total tokens consumed across all agents and all rounds.
    ///
    /// Accumulated from the `TokenUsage` reported by each agent's underlying
    /// LLM client. If a client does not report usage the contribution is zero.
    pub total_tokens_used: usize,
}

/// Errors that can occur during orchestration configuration or execution.
///
/// These are returned from [`Orchestration::add_agent`] and
/// [`Orchestration::run`] (boxed as `Box<dyn Error + Send + Sync>`).
///
/// # Examples
///
/// ```
/// use cloudllm::orchestration::OrchestrationError;
///
/// let err = OrchestrationError::AgentNotFound("missing-agent".into());
/// assert_eq!(err.to_string(), "Agent not found: missing-agent");
/// ```
#[derive(Debug, Clone)]
pub enum OrchestrationError {
    /// An agent ID referenced in the mode configuration (e.g., the moderator in
    /// [`OrchestrationMode::Moderated`]) does not match any registered agent.
    AgentNotFound(String),

    /// The mode configuration is structurally invalid (e.g., empty layer list in
    /// Hierarchical mode).
    InvalidMode(String),

    /// A runtime failure occurred while gathering agent responses (e.g., a
    /// `tokio::spawn` join error or a duplicate agent ID on insertion).
    ExecutionFailed(String),

    /// [`Orchestration::run`] was called before any agents were added.
    NoAgents,
}

impl fmt::Display for OrchestrationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            OrchestrationError::AgentNotFound(id) => write!(f, "Agent not found: {}", id),
            OrchestrationError::InvalidMode(msg) => write!(f, "Invalid mode: {}", msg),
            OrchestrationError::ExecutionFailed(msg) => write!(f, "Execution failed: {}", msg),
            OrchestrationError::NoAgents => write!(f, "No agents in orchestration"),
        }
    }
}

impl Error for OrchestrationError {}

/// The orchestration engine that coordinates multiple [`Agent`]s in a chosen
/// [`OrchestrationMode`].
///
/// An `Orchestration` owns a set of agents, a collaboration mode, and a running
/// conversation history. Call [`Orchestration::run`] to execute a multi-agent
/// conversation and receive an [`OrchestrationResponse`].
///
/// # Examples
///
/// ```rust,no_run
/// use cloudllm::{Agent, orchestration::{Orchestration, OrchestrationMode}};
/// use cloudllm::clients::openai::OpenAIClient;
/// use std::sync::Arc;
///
/// # async {
/// let client = || Arc::new(OpenAIClient::new_with_model_string("key", "gpt-4o"));
///
/// let mut orch = Orchestration::new("team", "My Team")
///     .with_mode(OrchestrationMode::RoundRobin)
///     .with_system_context("You are expert engineers.")
///     .with_max_tokens(16384);
///
/// orch.add_agent(Agent::new("alice", "Alice", client())).unwrap();
/// orch.add_agent(Agent::new("bob", "Bob", client())).unwrap();
///
/// let result = orch.run("Design a REST API", 2).await.unwrap();
/// println!("{} messages over {} rounds", result.messages.len(), result.round);
/// # };
/// ```
pub struct Orchestration {
    /// Stable identifier used for logging, metrics, and external integrations.
    pub id: String,

    /// Human-readable name of this orchestration.
    pub name: String,

    /// Registered agents keyed by their [`Agent::id`].
    agents: HashMap<String, Agent>,

    /// Agent IDs in insertion order. Determines the iteration sequence for
    /// round-robin, debate, and Ralph modes.
    agent_order: Vec<String>,

    /// The active collaboration strategy. Set via [`Orchestration::with_mode`].
    mode: OrchestrationMode,

    /// Running conversation history shared across rounds. Each agent response
    /// and user prompt is appended here so subsequent agents see prior context.
    conversation_history: Vec<OrchestrationMessage>,

    /// System-level context string prepended to every agent call.
    /// Override with [`Orchestration::with_system_context`].
    system_context: String,

    /// Soft token budget forwarded to agents for context trimming.
    /// Override with [`Orchestration::with_max_tokens`].
    max_tokens: usize,

    /// Per-agent cursor tracking the last message index each agent has seen.
    /// Used by hub-routing to avoid re-injecting messages agents already have.
    agent_message_cursors: HashMap<String, usize>,

    /// Optional event handler for real-time observability. When set, the orchestration
    /// emits [`OrchestrationEvent`]s during `run()` and propagates the handler to
    /// agents added via `add_agent()` so their [`AgentEvent`](crate::event::AgentEvent)s
    /// also flow through the same callback.
    event_handler: Option<Arc<dyn EventHandler>>,
}

impl Orchestration {
    /// Create an orchestration with the provided identifiers.
    ///
    /// Defaults to [`OrchestrationMode::Parallel`], a generic system context, and
    /// an 8 192-token budget. Use the `with_*` builder methods to customise.
    ///
    /// # Examples
    ///
    /// ```
    /// use cloudllm::orchestration::Orchestration;
    ///
    /// let orch = Orchestration::new("qa-team", "QA Review Team");
    /// assert_eq!(orch.id, "qa-team");
    /// assert_eq!(orch.name, "QA Review Team");
    /// ```
    pub fn new(id: impl Into<String>, name: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            agents: HashMap::new(),
            agent_order: Vec::new(),
            mode: OrchestrationMode::Parallel,
            conversation_history: Vec::new(),
            system_context: String::from(
                "You are participating in a collaborative discussion with other AI agents.",
            ),
            max_tokens: 8192,
            agent_message_cursors: HashMap::new(),
            event_handler: None,
        }
    }

    /// Select the collaboration mode used during [`Orchestration::run`] (builder pattern).
    ///
    /// # Examples
    ///
    /// ```
    /// use cloudllm::orchestration::{Orchestration, OrchestrationMode};
    ///
    /// let orch = Orchestration::new("id", "name")
    ///     .with_mode(OrchestrationMode::Debate {
    ///         max_rounds: 3,
    ///         convergence_threshold: Some(0.8),
    ///     });
    /// ```
    pub fn with_mode(mut self, mode: OrchestrationMode) -> Self {
        self.mode = mode;
        self
    }

    /// Override the default system context prompt shared across agents (builder pattern).
    ///
    /// This string is passed as the system prompt in every LLM call made during
    /// [`Orchestration::run`].
    ///
    /// # Examples
    ///
    /// ```
    /// use cloudllm::orchestration::Orchestration;
    ///
    /// let orch = Orchestration::new("id", "name")
    ///     .with_system_context("You are senior Rust engineers reviewing a PR.");
    /// ```
    pub fn with_system_context(mut self, context: impl Into<String>) -> Self {
        self.system_context = context.into();
        self
    }

    /// Override the soft token budget used for context trimming (builder pattern).
    ///
    /// The budget is forwarded to each agent's LLM call so that overly-long
    /// conversation histories can be trimmed before transmission.
    ///
    /// # Examples
    ///
    /// ```
    /// use cloudllm::orchestration::Orchestration;
    ///
    /// let orch = Orchestration::new("id", "name")
    ///     .with_max_tokens(32768);
    /// ```
    pub fn with_max_tokens(mut self, max_tokens: usize) -> Self {
        self.max_tokens = max_tokens;
        self
    }

    /// Attach an [`EventHandler`] for orchestration and agent lifecycle events (builder pattern).
    ///
    /// The handler receives [`OrchestrationEvent`]s directly from the orchestration
    /// engine (run start/end, round boundaries, RALPH progress, etc.). Additionally,
    /// the handler is **automatically propagated** to every agent added via
    /// [`add_agent`](Orchestration::add_agent), so those agents will also emit
    /// their [`AgentEvent`](crate::event::AgentEvent)s through the same handler.
    /// This gives you a unified stream of both orchestration-level and agent-level
    /// events through a single callback.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use cloudllm::orchestration::{Orchestration, OrchestrationMode};
    /// use cloudllm::event::{EventHandler, OrchestrationEvent};
    /// use async_trait::async_trait;
    /// use std::sync::Arc;
    ///
    /// struct MyHandler;
    /// #[async_trait]
    /// impl EventHandler for MyHandler {
    ///     async fn on_orchestration_event(&self, event: &OrchestrationEvent) {
    ///         println!("{:?}", event);
    ///     }
    /// }
    ///
    /// let orch = Orchestration::new("id", "name")
    ///     .with_mode(OrchestrationMode::RoundRobin)
    ///     .with_event_handler(Arc::new(MyHandler));
    /// ```
    pub fn with_event_handler(mut self, handler: Arc<dyn EventHandler>) -> Self {
        self.event_handler = Some(handler);
        self
    }

    /// Emit an [`OrchestrationEvent`] to the registered handler.
    ///
    /// If no handler is registered, this is a no-op. Called throughout the
    /// execution methods to signal run start/end, round boundaries, agent
    /// selection/response/failure, convergence checks, and RALPH progress.
    async fn emit(&self, event: OrchestrationEvent) {
        if let Some(handler) = &self.event_handler {
            handler.on_orchestration_event(&event).await;
        }
    }

    /// Return a human-readable name for the current [`OrchestrationMode`].
    ///
    /// Used in [`OrchestrationEvent::RunStarted`] to populate the `mode` field.
    fn mode_name(&self) -> &'static str {
        match &self.mode {
            OrchestrationMode::Parallel => "Parallel",
            OrchestrationMode::RoundRobin => "RoundRobin",
            OrchestrationMode::Moderated { .. } => "Moderated",
            OrchestrationMode::Hierarchical { .. } => "Hierarchical",
            OrchestrationMode::Debate { .. } => "Debate",
            OrchestrationMode::Ralph { .. } => "Ralph",
            OrchestrationMode::AnthropicAgentTeams { .. } => "AnthropicAgentTeams",
        }
    }

    /// Register a new agent with the orchestration.
    ///
    /// Returns an error if an agent with the same [`Agent::id`] is already
    /// registered. The insertion order determines the round-robin sequence
    /// used by RoundRobin, Debate, and Ralph modes.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use cloudllm::{Agent, orchestration::Orchestration};
    /// use cloudllm::clients::openai::OpenAIClient;
    /// use std::sync::Arc;
    ///
    /// let mut orch = Orchestration::new("id", "name");
    /// let client = Arc::new(OpenAIClient::new_with_model_string("key", "gpt-4o"));
    ///
    /// orch.add_agent(Agent::new("analyst", "Analyst", client)).unwrap();
    ///
    /// // Duplicate ID is an error
    /// # let client2 = Arc::new(OpenAIClient::new_with_model_string("key", "gpt-4o"));
    /// assert!(orch.add_agent(Agent::new("analyst", "Analyst 2", client2)).is_err());
    /// ```
    pub fn add_agent(&mut self, mut agent: Agent) -> Result<(), OrchestrationError> {
        let id = agent.id.clone();
        if self.agents.contains_key(&id) {
            return Err(OrchestrationError::ExecutionFailed(format!(
                "Agent with id '{}' already exists",
                id
            )));
        }

        // Propagate the orchestration's event handler to the agent so that
        // AgentEvents (LLM calls, tool usage, etc.) flow through the same
        // handler as OrchestrationEvents, giving the user a unified stream.
        if let Some(handler) = &self.event_handler {
            agent.set_event_handler(Arc::clone(handler));
        }

        self.agent_order.push(id.clone());
        self.agents.insert(id, agent);
        Ok(())
    }

    /// Remove and return an agent by its identifier.
    ///
    /// Returns `None` if no agent with the given ID exists. Removing an agent
    /// also removes it from the round-robin order.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// # use cloudllm::{Agent, orchestration::Orchestration};
    /// # use cloudllm::clients::openai::OpenAIClient;
    /// # use std::sync::Arc;
    /// let mut orch = Orchestration::new("id", "name");
    /// # let client = Arc::new(OpenAIClient::new_with_model_string("key", "gpt-4o"));
    /// # orch.add_agent(Agent::new("a1", "Agent", client)).unwrap();
    ///
    /// let removed = orch.remove_agent("a1");
    /// assert!(removed.is_some());
    /// assert!(orch.remove_agent("a1").is_none()); // already gone
    /// ```
    pub fn remove_agent(&mut self, id: &str) -> Option<Agent> {
        self.agent_order.retain(|aid| aid != id);
        self.agents.remove(id)
    }

    /// Borrow a registered agent by its identifier.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// # use cloudllm::{Agent, orchestration::Orchestration};
    /// # use cloudllm::clients::openai::OpenAIClient;
    /// # use std::sync::Arc;
    /// let mut orch = Orchestration::new("id", "name");
    /// # let client = Arc::new(OpenAIClient::new_with_model_string("key", "gpt-4o"));
    /// # orch.add_agent(Agent::new("a1", "Agent 1", client)).unwrap();
    ///
    /// if let Some(agent) = orch.get_agent("a1") {
    ///     println!("Found agent: {}", agent.name);
    /// }
    /// ```
    pub fn get_agent(&self, id: &str) -> Option<&Agent> {
        self.agents.get(id)
    }

    /// List agents in their insertion order.
    ///
    /// The returned order matches the round-robin sequence used by RoundRobin,
    /// Debate, and Ralph modes.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// # use cloudllm::{Agent, orchestration::Orchestration};
    /// # use cloudllm::clients::openai::OpenAIClient;
    /// # use std::sync::Arc;
    /// let mut orch = Orchestration::new("id", "name");
    /// # let c = || Arc::new(OpenAIClient::new_with_model_string("key", "gpt-4o"));
    /// # orch.add_agent(Agent::new("a", "Alice", c())).unwrap();
    /// # orch.add_agent(Agent::new("b", "Bob", c())).unwrap();
    ///
    /// for agent in orch.list_agents() {
    ///     println!("{}: {}", agent.id, agent.name);
    /// }
    /// ```
    pub fn list_agents(&self) -> Vec<&Agent> {
        self.agent_order
            .iter()
            .filter_map(|id| self.agents.get(id))
            .collect()
    }

    /// Execute a multi-agent discussion according to the configured [`OrchestrationMode`].
    ///
    /// The `prompt` is broadcast to all agents according to the active mode.
    ///
    /// # Parameters
    ///
    /// - `prompt` — The user question or task description.
    /// - `rounds` — Number of iterations for fixed-round modes (Parallel, RoundRobin,
    ///   Moderated). Ignored by Hierarchical (which runs once per layer), Debate (uses
    ///   `max_rounds`), and Ralph (uses `max_iterations`).
    ///
    /// # Errors
    ///
    /// Returns [`OrchestrationError::NoAgents`] if no agents have been registered.
    /// May also surface errors from individual agent LLM calls or tokio task joins.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// # use cloudllm::{Agent, orchestration::{Orchestration, OrchestrationMode}};
    /// # use cloudllm::clients::openai::OpenAIClient;
    /// # use std::sync::Arc;
    /// # async {
    /// # let c = Arc::new(OpenAIClient::new_with_model_string("key", "gpt-4o"));
    /// # let mut orch = Orchestration::new("id", "name");
    /// # orch.add_agent(Agent::new("a", "A", c)).unwrap();
    /// let response = orch.run("Summarise this paper", 2).await?;
    /// assert!(response.is_complete);
    /// # Ok::<(), Box<dyn std::error::Error + Send + Sync>>(())
    /// # };
    /// ```
    pub async fn run(
        &mut self,
        prompt: &str,
        rounds: usize,
    ) -> Result<OrchestrationResponse, Box<dyn Error + Send + Sync>> {
        if self.agents.is_empty() {
            return Err(Box::new(OrchestrationError::NoAgents));
        }

        self.emit(OrchestrationEvent::RunStarted {
            orchestration_id: self.id.clone(),
            orchestration_name: self.name.clone(),
            mode: self.mode_name().to_string(),
            agent_count: self.agents.len(),
        })
        .await;

        // Add user message to history
        self.conversation_history
            .push(OrchestrationMessage::new(Role::User, prompt));

        // Clone mode to avoid borrow issues
        let mode = self.mode.clone();

        let result = match mode {
            OrchestrationMode::Parallel => self.execute_parallel(prompt, rounds).await,
            OrchestrationMode::RoundRobin => self.execute_round_robin(prompt, rounds).await,
            OrchestrationMode::Moderated { moderator_id } => {
                self.execute_moderated(prompt, rounds, &moderator_id).await
            }
            OrchestrationMode::Hierarchical { layers } => {
                self.execute_hierarchical(prompt, &layers).await
            }
            OrchestrationMode::Debate {
                max_rounds,
                convergence_threshold,
            } => {
                self.execute_debate(prompt, max_rounds, convergence_threshold)
                    .await
            }
            OrchestrationMode::Ralph {
                tasks,
                max_iterations,
            } => self.execute_ralph(prompt, &tasks, max_iterations).await,
            OrchestrationMode::AnthropicAgentTeams {
                pool_id,
                tasks,
                max_iterations,
            } => {
                self.execute_anthropic_agent_teams(prompt, &pool_id, &tasks, max_iterations)
                    .await
            }
        };

        if let Ok(ref response) = result {
            self.emit(OrchestrationEvent::RunCompleted {
                orchestration_id: self.id.clone(),
                orchestration_name: self.name.clone(),
                rounds: response.round,
                total_tokens: response.total_tokens_used,
                is_complete: response.is_complete,
            })
            .await;
        }

        result
    }

    /// Initialize all agents' system prompts using the orchestration's system context.
    ///
    /// Called at the start of each execution mode so every agent has its
    /// augmented system prompt configured before any messages are routed.
    fn setup_agent_prompts(&mut self) {
        for agent in self.agents.values_mut() {
            agent.set_system_prompt(&self.system_context);
        }
    }

    /// Execute parallel mode: all agents respond simultaneously.
    ///
    /// For each round, every agent is forked into a separate `tokio` task.
    /// Each fork receives only the system prompt and the user prompt via
    /// hub-routing — no full history broadcast.
    ///
    /// # Events Emitted
    ///
    /// Per round: `RoundStarted`, then `AgentResponded` (or `AgentFailed`)
    /// for each agent, then `RoundCompleted`.
    async fn execute_parallel(
        &mut self,
        prompt: &str,
        rounds: usize,
    ) -> Result<OrchestrationResponse, Box<dyn Error + Send + Sync>> {
        self.setup_agent_prompts();
        let mut all_messages = Vec::new();
        let mut total_tokens = 0;

        for round_num in 0..rounds {
            self.emit(OrchestrationEvent::RoundStarted {
                orchestration_id: self.id.clone(),
                round: round_num + 1,
            })
            .await;

            let mut round_messages = Vec::new();

            // Spawn all agent tasks in parallel
            let mut tasks = Vec::new();
            let prompt_owned = prompt.to_string();
            let system_context = self.system_context.clone();

            for agent_id in &self.agent_order {
                let agent = self.agents.get(agent_id).unwrap();
                let mut temp_agent = agent.fork();
                temp_agent.set_system_prompt(&system_context);
                let prompt_clone = prompt_owned.clone();

                tasks.push(tokio::spawn(async move {
                    let result = temp_agent.send(&prompt_clone).await;
                    (temp_agent.id.clone(), temp_agent.name.clone(), result)
                }));
            }

            // Collect results
            for task in tasks {
                let (agent_id, agent_name, result) = task.await.map_err(|e| {
                    Box::new(OrchestrationError::ExecutionFailed(format!(
                        "Task join error: {}",
                        e
                    ))) as Box<dyn Error + Send + Sync>
                })?;

                match result {
                    Ok(agent_response) => {
                        if let Some(usage) = &agent_response.tokens_used {
                            total_tokens += usage.total_tokens;
                        }

                        self.emit(OrchestrationEvent::AgentResponded {
                            orchestration_id: self.id.clone(),
                            agent_id: agent_id.clone(),
                            agent_name: agent_name.clone(),
                            tokens_used: agent_response.tokens_used.clone(),
                            response_length: agent_response.content.len(),
                        })
                        .await;

                        let msg = OrchestrationMessage::from_agent(
                            agent_id,
                            agent_name,
                            agent_response.content,
                        );
                        round_messages.push(msg.clone());
                        self.conversation_history.push(msg);
                    }
                    Err(e) => {
                        self.emit(OrchestrationEvent::AgentFailed {
                            orchestration_id: self.id.clone(),
                            agent_id: agent_id.clone(),
                            agent_name: agent_name.clone(),
                            error: e.to_string(),
                        })
                        .await;
                    }
                }
            }

            self.emit(OrchestrationEvent::RoundCompleted {
                orchestration_id: self.id.clone(),
                round: round_num + 1,
            })
            .await;

            all_messages.extend(round_messages);
        }

        Ok(OrchestrationResponse {
            messages: all_messages,
            round: rounds,
            is_complete: true,
            convergence_score: None,
            total_tokens_used: total_tokens,
        })
    }

    /// Execute round-robin mode: agents take turns sequentially.
    ///
    /// Each agent has its own session. The hub routes only prior agents'
    /// responses into the current agent's session before it generates.
    /// No prompt augmentation duplication — each message is injected once.
    ///
    /// # Events Emitted
    ///
    /// Per round: `RoundStarted`, then for each agent: `AgentSelected` →
    /// `AgentResponded` (or `AgentFailed`), then `RoundCompleted`.
    async fn execute_round_robin(
        &mut self,
        prompt: &str,
        rounds: usize,
    ) -> Result<OrchestrationResponse, Box<dyn Error + Send + Sync>> {
        self.setup_agent_prompts();
        let mut all_messages: Vec<OrchestrationMessage> = Vec::new();
        let mut total_tokens = 0;

        for round_num in 0..rounds {
            self.emit(OrchestrationEvent::RoundStarted {
                orchestration_id: self.id.clone(),
                round: round_num + 1,
            })
            .await;

            for agent_id in self.agent_order.clone() {
                let mut agent = self.agents.remove(&agent_id).unwrap();

                // Route only NEW messages this agent hasn't seen yet
                let cursor = self.agent_message_cursors.get(&agent_id).copied().unwrap_or(0);
                for msg in &all_messages[cursor..] {
                    if let Some(name) = &msg.agent_name {
                        agent.receive_message(
                            Role::Assistant,
                            format!("[{}]: {}", name, msg.content),
                        );
                    }
                }
                self.agent_message_cursors.insert(agent_id.clone(), all_messages.len());

                self.emit(OrchestrationEvent::AgentSelected {
                    orchestration_id: self.id.clone(),
                    agent_id: agent_id.clone(),
                    agent_name: agent.name.clone(),
                    reason: "RoundRobin turn".to_string(),
                })
                .await;

                let result = agent.send(prompt).await;

                // Re-insert agent before handling result
                let agent_name = agent.name.clone();
                self.agents.insert(agent_id.clone(), agent);

                match result {
                    Ok(agent_response) => {
                        if let Some(usage) = &agent_response.tokens_used {
                            total_tokens += usage.total_tokens;
                        }

                        self.emit(OrchestrationEvent::AgentResponded {
                            orchestration_id: self.id.clone(),
                            agent_id: agent_id.clone(),
                            agent_name: agent_name.clone(),
                            tokens_used: agent_response.tokens_used.clone(),
                            response_length: agent_response.content.len(),
                        })
                        .await;

                        let msg = OrchestrationMessage::from_agent(
                            &agent_id,
                            &agent_name,
                            agent_response.content,
                        );
                        all_messages.push(msg.clone());
                        self.conversation_history.push(msg);
                    }
                    Err(e) => {
                        self.emit(OrchestrationEvent::AgentFailed {
                            orchestration_id: self.id.clone(),
                            agent_id: agent_id.clone(),
                            agent_name: agent_name.clone(),
                            error: e.to_string(),
                        })
                        .await;
                    }
                }
            }

            self.emit(OrchestrationEvent::RoundCompleted {
                orchestration_id: self.id.clone(),
                round: round_num + 1,
            })
            .await;
        }

        Ok(OrchestrationResponse {
            messages: all_messages,
            round: rounds,
            is_complete: true,
            convergence_score: None,
            total_tokens_used: total_tokens,
        })
    }

    /// Execute moderated mode: a moderator agent picks which expert speaks each round.
    ///
    /// The moderator and each expert use their own sessions. The hub routes
    /// discussion messages selectively — the moderator sees everything, while
    /// the chosen expert receives only the messages it hasn't seen yet.
    ///
    /// # Events Emitted
    ///
    /// The moderator's `send()` call is not surfaced as an `AgentSelected` event
    /// (it's an internal selection step). The selected expert's response triggers
    /// `AgentResponded` (or `AgentFailed`).
    async fn execute_moderated(
        &mut self,
        prompt: &str,
        rounds: usize,
        moderator_id: &str,
    ) -> Result<OrchestrationResponse, Box<dyn Error + Send + Sync>> {
        if !self.agents.contains_key(moderator_id) {
            return Err(Box::new(OrchestrationError::AgentNotFound(moderator_id.to_string())));
        }

        self.setup_agent_prompts();

        // Set up moderator with its own system prompt
        {
            let moderator = self.agents.get_mut(moderator_id).unwrap();
            moderator.set_system_prompt(
                "You are a moderator. Your job is to select the most appropriate expert to answer each question. \
                 Ensure both sides get fair representation by alternating between different experts.",
            );
        }

        let mut all_messages: Vec<OrchestrationMessage> = Vec::new();
        let mut total_tokens = 0;

        let expert_names: String = self
            .agents
            .values()
            .filter(|a| a.id != moderator_id)
            .map(|a| a.name.as_str())
            .collect::<Vec<_>>()
            .join(", ");

        for round_num in 0..rounds {
            // Build moderator prompt
            let moderator_prompt = if round_num == 0 {
                format!(
                    "{}\n\nAvailable experts: {}\n\nWhich expert should address this question? \
                     Respond with ONLY the expert name.",
                    prompt, expert_names
                )
            } else {
                format!(
                    "Based on the discussion so far, who should speak next to continue the debate?\
                     \n\nAvailable experts: {}\n\nWhich expert should address this question? \
                     Respond with ONLY the expert name.",
                    expert_names
                )
            };

            // Remove moderator, route new messages, call send, re-insert
            let mut moderator = self.agents.remove(moderator_id).unwrap();
            let mod_cursor = self.agent_message_cursors.get(moderator_id).copied().unwrap_or(0);
            for msg in &all_messages[mod_cursor..] {
                if let Some(name) = &msg.agent_name {
                    moderator.receive_message(
                        Role::Assistant,
                        format!("[{}]: {}", name, msg.content),
                    );
                }
            }
            self.agent_message_cursors.insert(moderator_id.to_string(), all_messages.len());

            let moderator_result = moderator.send(&moderator_prompt).await?;
            let selection = moderator_result.content.clone();
            if let Some(usage) = moderator_result.tokens_used {
                total_tokens += usage.total_tokens;
            }
            self.agents.insert(moderator_id.to_string(), moderator);

            // Find the selected agent (fuzzy match on name)
            let selected_id = self
                .agents
                .iter()
                .find(|(id, a)| {
                    id.as_str() != moderator_id
                        && selection.to_lowercase().contains(&a.name.to_lowercase())
                })
                .map(|(id, _)| id.clone())
                .or_else(|| {
                    self.agents
                        .keys()
                        .find(|id| id.as_str() != moderator_id)
                        .cloned()
                });

            if let Some(agent_id) = selected_id {
                let mut agent = self.agents.remove(&agent_id).unwrap();

                // Route new messages to this expert
                let cursor = self.agent_message_cursors.get(&agent_id).copied().unwrap_or(0);
                for msg in &all_messages[cursor..] {
                    if let Some(name) = &msg.agent_name {
                        agent.receive_message(
                            Role::Assistant,
                            format!("[{}]: {}", name, msg.content),
                        );
                    }
                }
                self.agent_message_cursors.insert(agent_id.clone(), all_messages.len());

                let agent_result = agent.send(prompt).await?;
                let agent_name = agent.name.clone();
                self.agents.insert(agent_id.clone(), agent);

                if let Some(usage) = agent_result.tokens_used {
                    total_tokens += usage.total_tokens;
                }

                let msg = OrchestrationMessage::from_agent(
                    &agent_id,
                    &agent_name,
                    agent_result.content,
                )
                .with_metadata("moderator", moderator_id.to_string())
                .with_metadata("round", round_num.to_string());

                all_messages.push(msg.clone());
                self.conversation_history.push(msg);
            }
        }

        Ok(OrchestrationResponse {
            messages: all_messages,
            round: rounds,
            is_complete: true,
            convergence_score: None,
            total_tokens_used: total_tokens,
        })
    }

    /// Execute hierarchical mode: agents are arranged in ordered layers.
    ///
    /// All agents within a single layer run in parallel via `fork()` + `send()`.
    /// Layer N agents receive only the synthesised output from layer N-1 —
    /// no full history broadcast.
    ///
    /// # Events Emitted
    ///
    /// Per layer: `RoundStarted`, then `AgentSelected` → `AgentResponded`
    /// (or `AgentFailed`) for each agent in the layer, then `RoundCompleted`.
    /// Each layer counts as one "round".
    async fn execute_hierarchical(
        &mut self,
        prompt: &str,
        layers: &[Vec<String>],
    ) -> Result<OrchestrationResponse, Box<dyn Error + Send + Sync>> {
        self.setup_agent_prompts();
        let mut all_messages = Vec::new();
        let mut layer_input = prompt.to_string();
        let mut total_tokens = 0;
        let system_context = self.system_context.clone();

        for (layer_idx, layer_agent_ids) in layers.iter().enumerate() {
            self.emit(OrchestrationEvent::RoundStarted {
                orchestration_id: self.id.clone(),
                round: layer_idx + 1,
            })
            .await;

            let mut layer_messages = Vec::new();

            // All agents in this layer work in parallel
            let mut tasks = Vec::new();

            for agent_id in layer_agent_ids {
                let agent = self
                    .agents
                    .get(agent_id)
                    .ok_or_else(|| OrchestrationError::AgentNotFound(agent_id.clone()))?;

                self.emit(OrchestrationEvent::AgentSelected {
                    orchestration_id: self.id.clone(),
                    agent_id: agent_id.clone(),
                    agent_name: agent.name.clone(),
                    reason: format!("Hierarchical layer {}", layer_idx),
                })
                .await;

                let mut temp_agent = agent.fork();
                temp_agent.set_system_prompt(&system_context);
                let current_prompt = layer_input.clone();

                tasks.push(tokio::spawn(async move {
                    let result = temp_agent.send(&current_prompt).await;
                    (temp_agent.id.clone(), temp_agent.name.clone(), result)
                }));
            }

            // Collect layer results
            for task in tasks {
                let (agent_id, agent_name, result) = task.await.map_err(|e| {
                    Box::new(OrchestrationError::ExecutionFailed(format!(
                        "Task join error: {}",
                        e
                    ))) as Box<dyn Error + Send + Sync>
                })?;

                match result {
                    Ok(agent_response) => {
                        if let Some(usage) = &agent_response.tokens_used {
                            total_tokens += usage.total_tokens;
                        }

                        self.emit(OrchestrationEvent::AgentResponded {
                            orchestration_id: self.id.clone(),
                            agent_id: agent_id.clone(),
                            agent_name: agent_name.clone(),
                            tokens_used: agent_response.tokens_used.clone(),
                            response_length: agent_response.content.len(),
                        })
                        .await;

                        let msg = OrchestrationMessage::from_agent(
                            agent_id,
                            agent_name,
                            agent_response.content,
                        )
                        .with_metadata("layer", layer_idx.to_string());
                        layer_messages.push(msg.clone());
                        self.conversation_history.push(msg);
                    }
                    Err(e) => {
                        self.emit(OrchestrationEvent::AgentFailed {
                            orchestration_id: self.id.clone(),
                            agent_id: agent_id.clone(),
                            agent_name: agent_name.clone(),
                            error: e.to_string(),
                        })
                        .await;
                    }
                }
            }

            self.emit(OrchestrationEvent::RoundCompleted {
                orchestration_id: self.id.clone(),
                round: layer_idx + 1,
            })
            .await;

            // Synthesize layer results for next layer
            if layer_idx < layers.len() - 1 {
                layer_input = format!(
                    "Original task: {}\n\nLayer {} results:\n{}",
                    prompt,
                    layer_idx,
                    layer_messages
                        .iter()
                        .map(|m| format!("{}: {}", m.agent_name.as_ref().unwrap(), m.content))
                        .collect::<Vec<_>>()
                        .join("\n\n")
                );
            }

            all_messages.extend(layer_messages);
        }

        Ok(OrchestrationResponse {
            messages: all_messages,
            round: layers.len(),
            is_complete: true,
            convergence_score: None,
            total_tokens_used: total_tokens,
        })
    }

    /// Execute debate mode: agents argue in rounds until their positions converge.
    ///
    /// Each agent maintains its own session. The hub injects only the latest
    /// round's arguments from OTHER agents — no prompt augmentation duplication.
    ///
    /// # Events Emitted
    ///
    /// Per round: `RoundStarted`, then `AgentSelected` → `AgentResponded`
    /// (or `AgentFailed`) for each agent, then `ConvergenceChecked` (after
    /// round 1+), then `RoundCompleted`. The debate terminates early when
    /// `ConvergenceChecked` reports `converged: true`.
    async fn execute_debate(
        &mut self,
        prompt: &str,
        max_rounds: usize,
        convergence_threshold: Option<f32>,
    ) -> Result<OrchestrationResponse, Box<dyn Error + Send + Sync>> {
        self.setup_agent_prompts();
        let mut all_messages: Vec<OrchestrationMessage> = Vec::new();
        let threshold = convergence_threshold.unwrap_or(0.75);
        let mut converged = false;
        let mut final_convergence_score = None;
        let mut actual_rounds = 0;
        let mut total_tokens = 0;

        for round in 0..max_rounds {
            actual_rounds = round + 1;

            self.emit(OrchestrationEvent::RoundStarted {
                orchestration_id: self.id.clone(),
                round: actual_rounds,
            })
            .await;

            let mut round_messages = Vec::new();

            for agent_id in self.agent_order.clone() {
                let mut agent = self.agents.remove(&agent_id).unwrap();

                // Route only NEW messages this agent hasn't seen
                let cursor = self.agent_message_cursors.get(&agent_id).copied().unwrap_or(0);
                for msg in &all_messages[cursor..] {
                    if let Some(name) = &msg.agent_name {
                        agent.receive_message(
                            Role::Assistant,
                            format!("[{}]: {}", name, msg.content),
                        );
                    }
                }
                self.agent_message_cursors.insert(agent_id.clone(), all_messages.len() + round_messages.len());

                let debate_prompt = format!(
                    "Round {} of debate: {}\n\n\
                     Consider the arguments presented and provide your position. \
                     Acknowledge strong points and challenge weak ones.",
                    round + 1,
                    prompt
                );

                self.emit(OrchestrationEvent::AgentSelected {
                    orchestration_id: self.id.clone(),
                    agent_id: agent_id.clone(),
                    agent_name: agent.name.clone(),
                    reason: format!("Debate round {}", actual_rounds),
                })
                .await;

                let result = agent.send(&debate_prompt).await;
                let agent_name = agent.name.clone();
                self.agents.insert(agent_id.clone(), agent);

                match result {
                    Ok(agent_response) => {
                        if let Some(usage) = &agent_response.tokens_used {
                            total_tokens += usage.total_tokens;
                        }

                        self.emit(OrchestrationEvent::AgentResponded {
                            orchestration_id: self.id.clone(),
                            agent_id: agent_id.clone(),
                            agent_name: agent_name.clone(),
                            tokens_used: agent_response.tokens_used.clone(),
                            response_length: agent_response.content.len(),
                        })
                        .await;

                        let msg = OrchestrationMessage::from_agent(
                            &agent_id,
                            &agent_name,
                            agent_response.content,
                        )
                        .with_metadata("round", round.to_string());
                        round_messages.push(msg.clone());
                        self.conversation_history.push(msg);
                    }
                    Err(e) => {
                        self.emit(OrchestrationEvent::AgentFailed {
                            orchestration_id: self.id.clone(),
                            agent_id: agent_id.clone(),
                            agent_name: agent_name.clone(),
                            error: e.to_string(),
                        })
                        .await;
                    }
                }
            }

            // Check for convergence after the first round
            if round > 0 && !round_messages.is_empty() {
                let convergence_score =
                    self.calculate_convergence_score(&all_messages, &round_messages);
                final_convergence_score = Some(convergence_score);

                let did_converge = convergence_score >= threshold;
                self.emit(OrchestrationEvent::ConvergenceChecked {
                    orchestration_id: self.id.clone(),
                    round: actual_rounds,
                    score: convergence_score,
                    threshold,
                    converged: did_converge,
                })
                .await;

                if did_converge {
                    converged = true;
                    all_messages.extend(round_messages);
                    self.emit(OrchestrationEvent::RoundCompleted {
                        orchestration_id: self.id.clone(),
                        round: actual_rounds,
                    })
                    .await;
                    break;
                }
            }

            self.emit(OrchestrationEvent::RoundCompleted {
                orchestration_id: self.id.clone(),
                round: actual_rounds,
            })
            .await;

            all_messages.extend(round_messages);
        }

        Ok(OrchestrationResponse {
            messages: all_messages,
            round: actual_rounds,
            is_complete: converged || actual_rounds >= max_rounds,
            convergence_score: final_convergence_score,
            total_tokens_used: total_tokens,
        })
    }

    /// Execute RALPH mode: autonomous iterative loop through a PRD task list.
    ///
    /// Each agent maintains its own session across iterations. The hub injects
    /// only the iteration prompt (task checklist + instructions). Previous
    /// iteration responses from OTHER agents are selectively injected.
    /// LLMSession's built-in trimming handles context overflow.
    ///
    /// # Events Emitted
    ///
    /// Per iteration: `RalphIterationStarted`, `RoundStarted`, then
    /// `AgentResponded` (or `AgentFailed`) + `RalphTaskCompleted` (if tasks
    /// were completed) for each agent, then `RoundCompleted`. The loop ends
    /// when all tasks are done or `max_iterations` is reached.
    async fn execute_ralph(
        &mut self,
        prompt: &str,
        tasks: &[RalphTask],
        max_iterations: usize,
    ) -> Result<OrchestrationResponse, Box<dyn Error + Send + Sync>> {
        // Edge case: no tasks means immediate completion
        if tasks.is_empty() {
            return Ok(OrchestrationResponse {
                messages: Vec::new(),
                round: 0,
                is_complete: true,
                convergence_score: Some(1.0),
                total_tokens_used: 0,
            });
        }

        self.setup_agent_prompts();

        let mut completed_tasks: HashSet<String> = HashSet::new();
        let mut all_messages: Vec<OrchestrationMessage> = Vec::new();
        let mut total_tokens: usize = 0;
        let mut actual_iterations: usize = 0;

        let task_ids: HashSet<String> = tasks.iter().map(|t| t.id.clone()).collect();

        for iteration in 0..max_iterations {
            actual_iterations = iteration + 1;

            self.emit(OrchestrationEvent::RalphIterationStarted {
                orchestration_id: self.id.clone(),
                iteration: actual_iterations,
                max_iterations,
                tasks_completed: completed_tasks.len(),
                tasks_total: tasks.len(),
            })
            .await;

            self.emit(OrchestrationEvent::RoundStarted {
                orchestration_id: self.id.clone(),
                round: actual_iterations,
            })
            .await;

            log::info!(
                "RALPH iteration {}/{} — {}/{} tasks complete",
                actual_iterations,
                max_iterations,
                completed_tasks.len(),
                tasks.len()
            );

            // Build task status checklist
            let mut checklist = String::new();
            for task in tasks {
                if completed_tasks.contains(&task.id) {
                    checklist.push_str(&format!(
                        "- [x] {} — {}\n",
                        task.title, task.description
                    ));
                } else {
                    checklist.push_str(&format!(
                        "- [ ] {} — {}\n",
                        task.title, task.description
                    ));
                }
            }

            // Build iteration prompt
            let iteration_prompt = format!(
                "=== RALPH Iteration {}/{} ===\n\n\
                 ## Original Request\n{}\n\n\
                 ## PRD Task Status\n{}\n\
                 ## Instructions\n\
                 Work on the next incomplete task. When done, include [TASK_COMPLETE:task_id].\n\
                 You may complete multiple tasks in a single response.",
                actual_iterations, max_iterations, prompt, checklist
            );

            // Each agent responds sequentially (round-robin within iteration)
            for agent_id in self.agent_order.clone() {
                let mut agent = self.agents.remove(&agent_id).unwrap();
                log::info!("  Calling agent '{}' ({})...", agent.name, agent.id);

                // Route only NEW messages from other agents
                let cursor = self.agent_message_cursors.get(&agent_id).copied().unwrap_or(0);
                for msg in &all_messages[cursor..] {
                    if let Some(name) = &msg.agent_name {
                        agent.receive_message(
                            Role::Assistant,
                            format!("[{}]: {}", name, msg.content),
                        );
                    }
                }
                self.agent_message_cursors.insert(agent_id.clone(), all_messages.len());

                let result = agent.send(&iteration_prompt).await;

                let agent_name_clone = agent.name.clone();
                self.agents.insert(agent_id.clone(), agent);

                match result {
                    Ok(agent_response) => {
                        let tokens_this_call = agent_response
                            .tokens_used
                            .as_ref()
                            .map(|u| u.total_tokens)
                            .unwrap_or(0);
                        if let Some(usage) = &agent_response.tokens_used {
                            total_tokens += usage.total_tokens;
                        }

                        self.emit(OrchestrationEvent::AgentResponded {
                            orchestration_id: self.id.clone(),
                            agent_id: agent_id.clone(),
                            agent_name: agent_name_clone.clone(),
                            tokens_used: agent_response.tokens_used.clone(),
                            response_length: agent_response.content.len(),
                        })
                        .await;

                        log::info!(
                            "  Agent '{}' responded ({} chars, {} tokens)",
                            agent_name_clone,
                            agent_response.content.len(),
                            tokens_this_call
                        );

                        // Parse completions from response
                        let newly_completed =
                            Self::parse_ralph_completions(&agent_response.content);

                        // Validate and insert
                        let mut valid_completions = Vec::new();
                        for id in &newly_completed {
                            if task_ids.contains(id) {
                                completed_tasks.insert(id.clone());
                                valid_completions.push(id.clone());
                            }
                        }

                        if !valid_completions.is_empty() {
                            self.emit(OrchestrationEvent::RalphTaskCompleted {
                                orchestration_id: self.id.clone(),
                                agent_id: agent_id.clone(),
                                agent_name: agent_name_clone.clone(),
                                task_ids: valid_completions.clone(),
                                tasks_completed_total: completed_tasks.len(),
                                tasks_total: tasks.len(),
                            })
                            .await;

                            log::info!(
                                "  Tasks completed: [{}] — progress: {}/{}",
                                valid_completions.join(", "),
                                completed_tasks.len(),
                                tasks.len()
                            );
                        }

                        let mut msg = OrchestrationMessage::from_agent(
                            &agent_id,
                            &agent_name_clone,
                            agent_response.content,
                        )
                        .with_metadata("iteration", actual_iterations.to_string());

                        if !valid_completions.is_empty() {
                            msg = msg.with_metadata(
                                "tasks_completed",
                                valid_completions.join(","),
                            );
                        }

                        all_messages.push(msg.clone());
                        self.conversation_history.push(msg);
                    }
                    Err(e) => {
                        self.emit(OrchestrationEvent::AgentFailed {
                            orchestration_id: self.id.clone(),
                            agent_id: agent_id.clone(),
                            agent_name: agent_name_clone.clone(),
                            error: e.to_string(),
                        })
                        .await;
                    }
                }
            }

            self.emit(OrchestrationEvent::RoundCompleted {
                orchestration_id: self.id.clone(),
                round: actual_iterations,
            })
            .await;

            // Check termination
            if completed_tasks.len() == tasks.len() {
                log::info!(
                    "All {}/{} tasks complete — stopping after iteration {}",
                    completed_tasks.len(),
                    tasks.len(),
                    actual_iterations
                );
                break;
            }
        }

        let total = tasks.len() as f32;
        let completed = completed_tasks.len() as f32;

        Ok(OrchestrationResponse {
            messages: all_messages,
            round: actual_iterations,
            is_complete: completed_tasks.len() == tasks.len(),
            convergence_score: Some(completed / total),
            total_tokens_used: total_tokens,
        })
    }

    /// Scan a string for `[TASK_COMPLETE:xxx]` markers, returning the task IDs found.
    ///
    /// Uses simple string scanning (no regex). Multiple markers in the same
    /// response are supported — the agent may complete several tasks at once.
    ///
    /// # Examples (internal)
    ///
    /// ```text
    /// Input:  "Done! [TASK_COMPLETE:auth] and [TASK_COMPLETE:db_schema]"
    /// Output: ["auth", "db_schema"]
    /// ```
    fn parse_ralph_completions(text: &str) -> Vec<String> {
        let mut results = Vec::new();
        let marker = "[TASK_COMPLETE:";
        let mut search_from = 0;
        while let Some(start) = text[search_from..].find(marker) {
            let abs_start = search_from + start + marker.len();
            if let Some(end) = text[abs_start..].find(']') {
                let id = text[abs_start..abs_start + end].trim().to_string();
                if !id.is_empty() {
                    results.push(id);
                }
                search_from = abs_start + end + 1;
            } else {
                break;
            }
        }
        results
    }

    /// Execute AnthropicAgentTeams mode: decentralized task coordination via shared Memory.
    ///
    /// Agents autonomously discover and claim tasks from a Memory pool. No central
    /// orchestrator — each agent uses Memory LIST to find unclaimed tasks,
    /// PUT to claim them, work on them, and PUT results when done.
    ///
    /// # Memory Key Scheme
    ///
    /// - `teams:<pool_id>:unclaimed:<task_id>` — task description + acceptance criteria
    /// - `teams:<pool_id>:claimed:<task_id>` — `<agent_id>:<timestamp>` (who's working)
    /// - `teams:<pool_id>:completed:<task_id>` — result JSON (task finished)
    /// - `teams:<pool_id>:metadata` — pool metadata (total tasks, created_at)
    /// - `teams:<pool_id>:stats` — stats (tasks_completed, tasks_failed)
    ///
    /// # Events Emitted
    ///
    /// Per iteration: `RoundStarted`, then `AgentSelected` + `TaskClaimed` (if new task),
    /// `TaskCompleted` or `TaskFailed` for each agent, then `RoundCompleted`.
    async fn execute_anthropic_agent_teams(
        &mut self,
        prompt: &str,
        pool_id: &str,
        tasks: &[WorkItem],
        max_iterations: usize,
    ) -> Result<OrchestrationResponse, Box<dyn Error + Send + Sync>> {
        // Edge case: no tasks means immediate completion
        if tasks.is_empty() {
            return Ok(OrchestrationResponse {
                messages: Vec::new(),
                round: 0,
                is_complete: true,
                convergence_score: Some(1.0),
                total_tokens_used: 0,
            });
        }

        self.setup_agent_prompts();

        let mut completed_tasks: HashSet<String> = HashSet::new();
        let mut claimed_tasks: HashMap<String, String> = HashMap::new(); // task_id -> agent_id
        let mut all_messages: Vec<OrchestrationMessage> = Vec::new();
        let mut total_tokens: usize = 0;
        let mut actual_iterations: usize = 0;

        let task_ids: HashSet<String> = tasks.iter().map(|t| t.id.clone()).collect();

        // Initialize Memory with task pool (mock — in real code agents would have Memory tool)
        log::info!(
            "AnthropicAgentTeams pool '{}' initialized with {} tasks",
            pool_id,
            tasks.len()
        );

        for iteration in 0..max_iterations {
            actual_iterations = iteration + 1;

            self.emit(OrchestrationEvent::RoundStarted {
                orchestration_id: self.id.clone(),
                round: actual_iterations,
            })
            .await;

            log::info!(
                "AnthropicAgentTeams iteration {}/{} — {}/{} tasks complete",
                actual_iterations,
                max_iterations,
                completed_tasks.len(),
                tasks.len()
            );

            // Build iteration prompt with available tasks
            let mut available_tasks = String::new();
            for task in tasks {
                if !completed_tasks.contains(&task.id) && !claimed_tasks.contains_key(&task.id) {
                    available_tasks.push_str(&format!("- {} — {}\n", task.id, task.description));
                }
            }

            let iteration_prompt = if available_tasks.is_empty() {
                format!(
                    "=== AnthropicAgentTeams Iteration {}/{} ===\n\n\
                     ## Original Request\n{}\n\n\
                     ## Task Status\n\
                     All available tasks have been claimed or completed.\n\
                     Completed tasks: {}/{}\n\
                     If you previously claimed a task, please report your result.",
                    actual_iterations, max_iterations, prompt, completed_tasks.len(), tasks.len()
                )
            } else {
                format!(
                    "=== AnthropicAgentTeams Iteration {}/{} ===\n\n\
                     ## Original Request\n{}\n\n\
                     ## Available Tasks\n{}\n\
                     ## Instructions\n\
                     Discover an unclaimed task from the Memory pool using the LIST and GET commands.\n\
                     Claim it by writing to Memory: PUT teams:{}:claimed:<task_id> <your_id>\n\
                     Complete the task and report: PUT teams:{}:completed:<task_id> {{\"result\": \"...\"}}\n\
                     Progress: {}/{}",
                    actual_iterations, max_iterations, prompt, available_tasks, pool_id, pool_id,
                    completed_tasks.len(),
                    tasks.len()
                )
            };

            // Each agent responds sequentially
            for agent_id in self.agent_order.clone() {
                let mut agent = self.agents.remove(&agent_id).unwrap();

                // Route only NEW messages from other agents
                let cursor = self.agent_message_cursors.get(&agent_id).copied().unwrap_or(0);
                for msg in &all_messages[cursor..] {
                    if let Some(name) = &msg.agent_name {
                        agent.receive_message(
                            Role::Assistant,
                            format!("[{}]: {}", name, msg.content),
                        );
                    }
                }
                self.agent_message_cursors.insert(agent_id.clone(), all_messages.len());

                self.emit(OrchestrationEvent::AgentSelected {
                    orchestration_id: self.id.clone(),
                    agent_id: agent_id.clone(),
                    agent_name: agent.name.clone(),
                    reason: format!("AnthropicAgentTeams iteration {}", actual_iterations),
                })
                .await;

                let result = agent.send(&iteration_prompt).await;

                let agent_name_clone = agent.name.clone();
                self.agents.insert(agent_id.clone(), agent);

                match result {
                    Ok(agent_response) => {
                        if let Some(usage) = &agent_response.tokens_used {
                            total_tokens += usage.total_tokens;
                        }

                        self.emit(OrchestrationEvent::AgentResponded {
                            orchestration_id: self.id.clone(),
                            agent_id: agent_id.clone(),
                            agent_name: agent_name_clone.clone(),
                            tokens_used: agent_response.tokens_used.clone(),
                            response_length: agent_response.content.len(),
                        })
                        .await;

                        // Parse agent response for task claims and completions
                        // In real implementation, agents would use Memory tool;
                        // here we simulate by looking for patterns in the response
                        let response_lower = agent_response.content.to_lowercase();

                        // Detect claimed task: if agent mentions a task_id
                        for task_id in &task_ids {
                            if response_lower.contains(task_id)
                                && !claimed_tasks.contains_key(task_id)
                                && !completed_tasks.contains(task_id)
                            {
                                claimed_tasks.insert(task_id.clone(), agent_id.clone());

                                self.emit(OrchestrationEvent::TaskClaimed {
                                    orchestration_id: self.id.clone(),
                                    agent_id: agent_id.clone(),
                                    agent_name: agent_name_clone.clone(),
                                    task_id: task_id.clone(),
                                })
                                .await;

                                log::info!(
                                    "  Agent '{}' claimed task '{}'",
                                    agent_name_clone,
                                    task_id
                                );
                                break; // One task per iteration
                            }
                        }

                        // Detect completed task: if response mentions completion
                        let is_completed = response_lower.contains("complete")
                            || response_lower.contains("done")
                            || response_lower.contains("finished");

                        if is_completed {
                            for task_id in &task_ids {
                                if claimed_tasks.get(task_id) == Some(&agent_id)
                                    && !completed_tasks.contains(task_id)
                                {
                                    completed_tasks.insert(task_id.clone());
                                    claimed_tasks.remove(task_id);

                                    self.emit(OrchestrationEvent::TaskCompleted {
                                        orchestration_id: self.id.clone(),
                                        agent_id: agent_id.clone(),
                                        agent_name: agent_name_clone.clone(),
                                        task_id: task_id.clone(),
                                        result: agent_response.content.clone(),
                                    })
                                    .await;

                                    log::info!(
                                        "  Agent '{}' completed task '{}'",
                                        agent_name_clone,
                                        task_id
                                    );
                                    break;
                                }
                            }
                        }

                        let msg = OrchestrationMessage::from_agent(
                            &agent_id,
                            &agent_name_clone,
                            agent_response.content,
                        )
                        .with_metadata("iteration", actual_iterations.to_string());

                        all_messages.push(msg.clone());
                        self.conversation_history.push(msg);
                    }
                    Err(e) => {
                        self.emit(OrchestrationEvent::AgentFailed {
                            orchestration_id: self.id.clone(),
                            agent_id: agent_id.clone(),
                            agent_name: agent_name_clone.clone(),
                            error: e.to_string(),
                        })
                        .await;

                        // Emit TaskFailed if agent had claimed a task
                        for (task_id, claiming_agent) in &claimed_tasks {
                            if claiming_agent == &agent_id {
                                self.emit(OrchestrationEvent::TaskFailed {
                                    orchestration_id: self.id.clone(),
                                    agent_id: agent_id.clone(),
                                    agent_name: agent_name_clone.clone(),
                                    task_id: task_id.clone(),
                                    error: format!("Agent failed: {}", e),
                                })
                                .await;
                            }
                        }
                    }
                }
            }

            self.emit(OrchestrationEvent::RoundCompleted {
                orchestration_id: self.id.clone(),
                round: actual_iterations,
            })
            .await;

            // Check termination
            if completed_tasks.len() == tasks.len() {
                log::info!(
                    "All {}/{} tasks complete — stopping after iteration {}",
                    completed_tasks.len(),
                    tasks.len(),
                    actual_iterations
                );
                break;
            }
        }

        let total = tasks.len() as f32;
        let completed = completed_tasks.len() as f32;

        Ok(OrchestrationResponse {
            messages: all_messages,
            round: actual_iterations,
            is_complete: completed_tasks.len() == tasks.len(),
            convergence_score: Some(completed / total),
            total_tokens_used: total_tokens,
        })
    }

    /// Calculate convergence score between the current and previous round of messages.
    ///
    /// Computes the average Jaccard similarity (word-set overlap) across
    /// corresponding agent messages from two consecutive rounds. A score of
    /// `1.0` means the responses are identical at the word level; `0.0` means
    /// zero overlap.
    fn calculate_convergence_score(
        &self,
        all_messages: &[OrchestrationMessage],
        current_round: &[OrchestrationMessage],
    ) -> f32 {
        // Get messages from the previous round
        let num_agents = self.agents.len();
        let previous_round: Vec<_> = all_messages.iter().rev().take(num_agents).rev().collect();

        if previous_round.len() != current_round.len() {
            return 0.0;
        }

        // Calculate average Jaccard similarity between corresponding agents' messages
        let mut total_similarity = 0.0;
        let mut comparison_count = 0;

        for i in 0..previous_round.len() {
            if let (Some(prev_msg), Some(curr_msg)) = (previous_round.get(i), current_round.get(i))
            {
                let similarity = self.jaccard_similarity(&prev_msg.content, &curr_msg.content);
                total_similarity += similarity;
                comparison_count += 1;
            }
        }

        if comparison_count > 0 {
            total_similarity / comparison_count as f32
        } else {
            0.0
        }
    }

    /// Calculate Jaccard similarity between two texts based on normalised word sets.
    ///
    /// Words shorter than 3 characters are ignored to reduce noise from articles
    /// and prepositions. Both inputs are lowercased before tokenisation.
    /// Returns `1.0` when both texts are empty, `0.0` when only one is empty.
    fn jaccard_similarity(&self, text1: &str, text2: &str) -> f32 {
        use std::collections::HashSet;

        // Normalize and tokenize both texts into word sets
        let words1: HashSet<String> = text1
            .to_lowercase()
            .split_whitespace()
            .filter(|w| w.len() > 2) // Ignore very short words
            .map(|w| w.trim_matches(|c: char| !c.is_alphanumeric()).to_string())
            .filter(|w| !w.is_empty())
            .collect();

        let words2: HashSet<String> = text2
            .to_lowercase()
            .split_whitespace()
            .filter(|w| w.len() > 2)
            .map(|w| w.trim_matches(|c: char| !c.is_alphanumeric()).to_string())
            .filter(|w| !w.is_empty())
            .collect();

        if words1.is_empty() && words2.is_empty() {
            return 1.0; // Both empty, consider them identical
        }

        if words1.is_empty() || words2.is_empty() {
            return 0.0; // One empty, no similarity
        }

        // Jaccard similarity = |intersection| / |union|
        let intersection_size = words1.intersection(&words2).count();
        let union_size = words1.union(&words2).count();

        intersection_size as f32 / union_size as f32
    }

    /// Return a slice of all messages accumulated since the orchestration was
    /// created (or since the last [`Orchestration::clear_history`] call).
    ///
    /// This includes the initial user prompt(s) and every agent response across
    /// all rounds.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// # use cloudllm::{Agent, orchestration::{Orchestration, OrchestrationMode}};
    /// # use cloudllm::clients::openai::OpenAIClient;
    /// # use std::sync::Arc;
    /// # async {
    /// # let c = Arc::new(OpenAIClient::new_with_model_string("key", "gpt-4o"));
    /// # let mut orch = Orchestration::new("id", "name");
    /// # orch.add_agent(Agent::new("a", "A", c)).unwrap();
    /// let _ = orch.run("Hello", 1).await?;
    ///
    /// let history = orch.get_conversation_history();
    /// println!("{} messages in history", history.len());
    /// # Ok::<(), Box<dyn std::error::Error + Send + Sync>>(())
    /// # };
    /// ```
    pub fn get_conversation_history(&self) -> &[OrchestrationMessage] {
        &self.conversation_history
    }

    /// Remove all historical messages, resetting the orchestration state.
    ///
    /// Call this between unrelated discussions when you want to reuse the same
    /// `Orchestration` instance without carrying over prior context.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// # use cloudllm::{Agent, orchestration::{Orchestration, OrchestrationMode}};
    /// # use cloudllm::clients::openai::OpenAIClient;
    /// # use std::sync::Arc;
    /// # async {
    /// # let c = Arc::new(OpenAIClient::new_with_model_string("key", "gpt-4o"));
    /// # let mut orch = Orchestration::new("id", "name");
    /// # orch.add_agent(Agent::new("a", "A", c)).unwrap();
    /// let _ = orch.run("First topic", 1).await?;
    /// orch.clear_history();
    ///
    /// // Start fresh — agents will not see "First topic" responses
    /// let _ = orch.run("Second topic", 1).await?;
    /// # Ok::<(), Box<dyn std::error::Error + Send + Sync>>(())
    /// # };
    /// ```
    pub fn clear_history(&mut self) {
        self.conversation_history.clear();
        self.agent_message_cursors.clear();
    }
}

