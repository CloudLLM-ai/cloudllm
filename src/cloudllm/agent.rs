//! Agent System
//!
//! This module provides the core [`Agent`] struct that represents an LLM-powered agent
//! with identity, expertise, personality, optional tool access, and real-time event
//! observability.
//!
//! Agents are the fundamental building blocks for LLM applications in CloudLLM and can be used:
//! - Standalone for single-agent interactions
//! - In orchestrations for multi-agent orchestration patterns
//! - In custom workflows for specialized use cases
//!
//! # Core Components
//!
//! - **Agent**: Represents an LLM agent with identity and capabilities
//! - **LLMSession**: Each agent wraps its own session with rolling history and token tracking
//! - **Tool Access**: Agents can be granted access to local or remote tools via [`ToolRegistry`](crate::tool_protocol::ToolRegistry)
//! - **ThoughtChain**: Optional persistent, hash-chained memory for findings/decisions
//! - **ContextStrategy**: Pluggable strategy for handling context window exhaustion
//! - **EventHandler**: Optional callback for real-time observability of LLM calls, tool usage, and lifecycle events
//! - **Expertise & Personality**: Optional attributes for behavior customization
//! - **Metadata**: Arbitrary key-value pairs for domain-specific extensions
//!
//! # Event System
//!
//! Agents emit [`AgentEvent`](crate::event::AgentEvent)s during their lifecycle. Attach
//! an [`EventHandler`](crate::event::EventHandler) via [`with_event_handler`](Agent::with_event_handler)
//! or [`set_event_handler`](Agent::set_event_handler) to receive real-time notifications
//! about LLM round-trips, tool calls, thought commits, and more. See the
//! [`event`](crate::event) module for the full list of events and examples.
//!
//! # Example
//!
//! ```rust,no_run
//! use cloudllm::Agent;
//! use cloudllm::clients::openai::OpenAIClient;
//! use std::sync::Arc;
//!
//! # async {
//! let agent = Agent::new(
//!     "analyst",
//!     "Technical Analyst",
//!     Arc::new(OpenAIClient::new_with_model_string("key", "gpt-4o"))
//! )
//! .with_expertise("Cloud Architecture")
//! .with_personality("Direct and analytical");
//!
//! // Use agent in your application...
//! # };
//! ```

use crate::client_wrapper::{ClientWrapper, Message, Role, TokenUsage};
use crate::cloudllm::context_strategy::{ContextStrategy, TrimStrategy};
use crate::cloudllm::event::{AgentEvent, EventHandler, PlannerEvent};
use crate::cloudllm::llm_session::LLMSession;
use crate::cloudllm::thought_chain::{Thought, ThoughtChain, ThoughtType};
use crate::cloudllm::tool_protocol::{ToolProtocol, ToolRegistry};
use openai_rust2::chat::{GrokTool, OpenAITool};
use std::collections::HashMap;
use std::error::Error;
use std::io;
use std::sync::Arc;
use tokio::sync::RwLock;

/// Internal representation of a parsed tool call extracted from an LLM response.
///
/// The agent's `parse_tool_call()` method scans LLM output for JSON fragments
/// matching `{"tool_call": {"name": "...", "parameters": {...}}}` and returns
/// this struct. The `name` is used to route the call through the
/// [`ToolRegistry`](crate::tool_protocol::ToolRegistry), and `parameters` is
/// the raw JSON payload forwarded to the tool protocol's `execute()` method.
#[derive(Debug, Clone)]
struct ToolCall {
    /// Name of the tool to execute (e.g. `"memory"`, `"calculator"`, `"write_game_file"`).
    name: String,
    /// Raw JSON parameters extracted from the LLM's tool call request.
    parameters: serde_json::Value,
}

/// Response body returned after asking an agent to generate content.
///
/// Wraps both the final text output and optional token-usage accounting.
/// When the agent makes multiple tool calls during a single generation
/// cycle, the `tokens_used` field aggregates usage across all LLM
/// round-trips.
#[derive(Debug, Clone)]
pub struct AgentResponse {
    /// Final message content produced across tool iterations.
    pub content: String,
    /// Optional token usage aggregated across all tool iterations.
    pub tokens_used: Option<TokenUsage>,
}

/// Represents an agent with identity, expertise, optional tool access, and
/// event observability.
///
/// Agents are LLM-powered entities that can:
/// - Generate responses based on system prompts and user messages
/// - Access tools through a [`ToolRegistry`] (single or multi-protocol)
/// - Maintain per-agent conversation memory via [`LLMSession`]
/// - Persist findings and decisions via [`ThoughtChain`]
/// - Handle context window exhaustion via pluggable [`ContextStrategy`]
/// - Emit [`AgentEvent`]s for real-time observability of LLM calls and tool usage
/// - Be orchestrated by [`Orchestration`](crate::orchestration::Orchestration) or used independently
pub struct Agent {
    // ---- Identity (public, unchanged) ----
    /// Stable identifier referenced inside orchestration coordination.
    pub id: String,
    /// Human-readable display name for logging and UI surfaces.
    pub name: String,
    /// Free-form description of the agent's strengths that will be embedded into prompts.
    pub expertise: Option<String>,
    /// Persona hints that help diversify the tone of generated responses.
    pub personality: Option<String>,
    /// Arbitrary metadata associated with the agent (e.g. department, region).
    pub metadata: HashMap<String, String>,

    // ---- Session (replaces raw Arc<dyn ClientWrapper>) ----
    session: LLMSession,

    // ---- Tools (now behind Arc<RwLock<_>> for runtime mutation) ----
    tool_registry: Arc<RwLock<ToolRegistry>>,
    grok_tools: Vec<GrokTool>,
    openai_tools: Vec<OpenAITool>,

    // ---- Context management (new) ----
    context_strategy: Box<dyn ContextStrategy>,
    thought_chain: Option<Arc<RwLock<ThoughtChain>>>,

    /// Optional event handler for real-time observability. When set, the agent
    /// emits [`AgentEvent`]s during `send()`, `generate_with_tokens()`, `commit()`,
    /// `add_protocol()`, `remove_protocol()`, `fork()`, `set_system_prompt()`,
    /// and `receive_message()`.
    event_handler: Option<Arc<dyn EventHandler>>,
}

impl Agent {
    /// Create a new agent with the mandatory identity information.
    ///
    /// Internally creates an [`LLMSession`] with the provided client, an empty
    /// system prompt, and a 128k token budget. Tools default to an empty
    /// [`ToolRegistry`] and the context strategy defaults to [`TrimStrategy`].
    pub fn new(
        id: impl Into<String>,
        name: impl Into<String>,
        client: Arc<dyn ClientWrapper>,
    ) -> Self {
        let session = LLMSession::new(client, String::new(), 128_000);
        Self {
            id: id.into(),
            name: name.into(),
            expertise: None,
            personality: None,
            metadata: HashMap::new(),
            session,
            tool_registry: Arc::new(RwLock::new(ToolRegistry::empty())),
            grok_tools: Vec::new(),
            openai_tools: Vec::new(),
            context_strategy: Box::new(TrimStrategy::default()),
            thought_chain: None,
            event_handler: None,
        }
    }

    /// Attach a brief description of the agent's domain expertise.
    pub fn with_expertise(mut self, expertise: impl Into<String>) -> Self {
        self.expertise = Some(expertise.into());
        self
    }

    /// Attach a personality descriptor used to diversify prompts.
    pub fn with_personality(mut self, personality: impl Into<String>) -> Self {
        self.personality = Some(personality.into());
        self
    }

    /// Add arbitrary metadata to the agent definition.
    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }

    /// Override the default token budget (builder pattern).
    ///
    /// Recreates the internal [`LLMSession`] with the new budget while keeping
    /// the same client.  History is reset (the session starts empty).
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use cloudllm::Agent;
    /// use cloudllm::clients::openai::OpenAIClient;
    /// use std::sync::Arc;
    ///
    /// let agent = Agent::new(
    ///     "a1", "Agent",
    ///     Arc::new(OpenAIClient::new_with_model_string("key", "gpt-4o")),
    /// )
    /// .with_max_tokens(32_000); // 32k instead of the default 128k
    /// ```
    pub fn with_max_tokens(mut self, max_tokens: usize) -> Self {
        let client = self.session.client().clone();
        self.session = LLMSession::new(client, String::new(), max_tokens);
        self
    }

    /// Grant the agent access to a registry of tools.
    ///
    /// Takes ownership of the registry and wraps it in `Arc<RwLock<_>>`.
    ///
    /// # Example: Single Protocol
    ///
    /// ```ignore
    /// let registry = ToolRegistry::new(
    ///     Arc::new(CustomToolProtocol::new())
    /// );
    /// agent.with_tools(registry);
    /// ```
    ///
    /// # Example: Multiple Protocols
    ///
    /// ```ignore
    /// let mut registry = ToolRegistry::empty();
    /// registry.add_protocol("local", Arc::new(local_protocol)).await?;
    /// registry.add_protocol("youtube", Arc::new(youtube_mcp)).await?;
    /// agent.with_tools(registry);
    /// ```
    pub fn with_tools(mut self, registry: ToolRegistry) -> Self {
        self.tool_registry = Arc::new(RwLock::new(registry));
        self
    }

    /// Share a mutable tool registry across multiple agents.
    ///
    /// This allows runtime mutations (add/remove protocols) to be visible
    /// to all agents sharing the same registry.  Use this when agents in an
    /// orchestration need to see the same tool set and react to hot-swaps.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use cloudllm::Agent;
    /// use cloudllm::tool_protocol::ToolRegistry;
    /// use cloudllm::clients::openai::OpenAIClient;
    /// use std::sync::Arc;
    /// use tokio::sync::RwLock;
    ///
    /// let shared = Arc::new(RwLock::new(ToolRegistry::empty()));
    ///
    /// let client = Arc::new(OpenAIClient::new_with_model_string("key", "gpt-4o"));
    /// let agent_a = Agent::new("a", "Agent A", client.clone())
    ///     .with_shared_tools(shared.clone());
    /// let agent_b = Agent::new("b", "Agent B", client)
    ///     .with_shared_tools(shared.clone());
    ///
    /// // Adding a protocol via agent_a is visible to agent_b
    /// ```
    pub fn with_shared_tools(mut self, registry: Arc<RwLock<ToolRegistry>>) -> Self {
        self.tool_registry = registry;
        self
    }

    /// Forward xAI server-side tools (web_search, x_search, etc.) to the underlying client.
    /// Only supported by Grok clients.
    pub fn with_grok_tools(mut self, grok_tools: Vec<GrokTool>) -> Self {
        self.grok_tools = grok_tools;
        self
    }

    /// Forward OpenAI server-side tools (web_search, file_search, code_interpreter) to the underlying client.
    /// Only supported by OpenAI clients.
    pub fn with_openai_tools(mut self, openai_tools: Vec<OpenAITool>) -> Self {
        self.openai_tools = openai_tools;
        self
    }

    /// Set the context collapse strategy (builder pattern).
    ///
    /// The strategy determines when and how the agent compacts its conversation
    /// history.  See the [`context_strategy`](crate::context_strategy) module
    /// for available implementations.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use cloudllm::Agent;
    /// use cloudllm::context_strategy::{NoveltyAwareStrategy, SelfCompressionStrategy};
    /// use cloudllm::clients::openai::OpenAIClient;
    /// use std::sync::Arc;
    ///
    /// let agent = Agent::new(
    ///     "a1", "Agent",
    ///     Arc::new(OpenAIClient::new_with_model_string("key", "gpt-4o")),
    /// )
    /// .context_collapse_strategy(Box::new(
    ///     NoveltyAwareStrategy::new(Box::new(SelfCompressionStrategy::default()))
    /// ));
    /// ```
    pub fn context_collapse_strategy(mut self, strategy: Box<dyn ContextStrategy>) -> Self {
        self.context_strategy = strategy;
        self
    }

    /// Replace the context collapse strategy at runtime.
    ///
    /// Unlike [`context_collapse_strategy`](Agent::context_collapse_strategy)
    /// (which consumes `self`), this takes `&mut self` so the strategy can be
    /// swapped on a live agent.
    pub fn set_context_collapse_strategy(&mut self, strategy: Box<dyn ContextStrategy>) {
        self.context_strategy = strategy;
    }

    /// Attach a [`ThoughtChain`] for persistent memory (builder pattern).
    ///
    /// Once attached, the agent can record findings and decisions via
    /// [`commit`](Agent::commit), and context strategies like
    /// [`SelfCompressionStrategy`](crate::context_strategy::SelfCompressionStrategy)
    /// will persist compression summaries to the chain automatically.
    ///
    /// The chain is wrapped in `Arc<RwLock<_>>` so it can be shared across
    /// forked agents or accessed concurrently.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use cloudllm::Agent;
    /// use cloudllm::thought_chain::ThoughtChain;
    /// use cloudllm::clients::openai::OpenAIClient;
    /// use std::sync::Arc;
    /// use std::path::PathBuf;
    /// use tokio::sync::RwLock;
    ///
    /// let chain = Arc::new(RwLock::new(
    ///     ThoughtChain::open(
    ///         &PathBuf::from("chains"), "a1", "Agent", Some("ML"), None,
    ///     ).unwrap()
    /// ));
    ///
    /// let agent = Agent::new(
    ///     "a1", "Agent",
    ///     Arc::new(OpenAIClient::new_with_model_string("key", "gpt-4o")),
    /// )
    /// .with_thought_chain(chain);
    /// ```
    pub fn with_thought_chain(mut self, chain: Arc<RwLock<ThoughtChain>>) -> Self {
        self.thought_chain = Some(chain);
        self
    }

    /// Attach an [`EventHandler`] that will receive lifecycle events (builder pattern).
    ///
    /// The handler receives [`AgentEvent`]s for LLM calls, tool usage, thought
    /// commits, protocol mutations, fork operations, and session changes.
    /// When this agent is added to an [`Orchestration`](crate::orchestration::Orchestration)
    /// via `add_agent()`, the orchestration's handler (if any) will override this one.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use cloudllm::Agent;
    /// use cloudllm::event::{AgentEvent, EventHandler};
    /// use cloudllm::clients::openai::OpenAIClient;
    /// use async_trait::async_trait;
    /// use std::sync::Arc;
    ///
    /// struct MyHandler;
    /// #[async_trait]
    /// impl EventHandler for MyHandler {
    ///     async fn on_agent_event(&self, event: &AgentEvent) {
    ///         println!("{:?}", event);
    ///     }
    /// }
    ///
    /// let agent = Agent::new("a1", "Agent", Arc::new(
    ///     OpenAIClient::new_with_model_string("key", "gpt-4o"),
    /// ))
    /// .with_event_handler(Arc::new(MyHandler));
    /// ```
    pub fn with_event_handler(mut self, handler: Arc<dyn EventHandler>) -> Self {
        self.event_handler = Some(handler);
        self
    }

    /// Set or replace the event handler at runtime.
    ///
    /// Unlike [`with_event_handler`](Agent::with_event_handler) (which consumes `self`
    /// in the builder chain), this takes `&mut self` so the handler can be attached
    /// to a live agent. Used internally by [`Orchestration::add_agent`](crate::orchestration::Orchestration::add_agent)
    /// to propagate the orchestration's handler to each agent.
    pub fn set_event_handler(&mut self, handler: Arc<dyn EventHandler>) {
        self.event_handler = Some(handler);
    }

    /// Emit an [`AgentEvent`] to the registered handler (async context).
    ///
    /// If no handler is registered, this is a no-op. Called from async methods
    /// like `send()`, `generate_with_tokens()`, `commit()`, `add_protocol()`,
    /// and `remove_protocol()`.
    async fn emit(&self, event: AgentEvent) {
        if let Some(handler) = &self.event_handler {
            handler.on_agent_event(&event).await;
        }
    }

    /// Emit an [`AgentEvent`] from a non-async (synchronous) context.
    ///
    /// Spawns a detached tokio task to call the async handler. Used by
    /// synchronous methods like `fork()`, `fork_with_context()`,
    /// `set_system_prompt()`, and `receive_message()` that cannot `.await`.
    /// The event delivery is fire-and-forget.
    fn emit_sync(&self, event: AgentEvent) {
        if let Some(handler) = &self.event_handler {
            let handler = Arc::clone(handler);
            tokio::spawn(async move {
                handler.on_agent_event(&event).await;
            });
        }
    }

    /// Emit a [`PlannerEvent`] to the registered handler (async context).
    ///
    /// Mirrors [`emit`](Agent::emit) but fires planner events so that agent
    /// turns also appear in the `[planner]` event stream alongside RALPH
    /// orchestration. If no handler is registered, this is a no-op.
    async fn emit_planner(&self, event: PlannerEvent) {
        if let Some(handler) = &self.event_handler {
            handler.on_planner_event(&event).await;
        }
    }

    // ---- Runtime tool mutation ----

    /// Add a new tool protocol at runtime.
    ///
    /// The protocol is discovered (its tools are listed) and then registered
    /// under `name`.  If the agent's tool registry is shared via
    /// [`with_shared_tools`](Agent::with_shared_tools), the new protocol is
    /// immediately visible to all agents sharing the same registry.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use cloudllm::Agent;
    /// use cloudllm::tool_protocols::CustomToolProtocol;
    /// use cloudllm::clients::openai::OpenAIClient;
    /// use std::sync::Arc;
    ///
    /// # async {
    /// let agent = Agent::new(
    ///     "a1", "Agent",
    ///     Arc::new(OpenAIClient::new_with_model_string("key", "gpt-4o")),
    /// );
    ///
    /// agent.add_protocol("custom", Arc::new(CustomToolProtocol::new())).await.unwrap();
    /// assert!(!agent.list_tools().await.is_empty());
    /// # };
    /// ```
    pub async fn add_protocol(
        &self,
        name: &str,
        protocol: Arc<dyn ToolProtocol>,
    ) -> Result<(), Box<dyn Error + Send + Sync>> {
        let mut registry = self.tool_registry.write().await;
        let result = registry.add_protocol(name, protocol).await;
        if result.is_ok() {
            self.emit(AgentEvent::ProtocolAdded {
                agent_id: self.id.clone(),
                agent_name: self.name.clone(),
                protocol_name: name.to_string(),
            })
            .await;
        }
        result
    }

    /// Remove a tool protocol at runtime.
    ///
    /// All tools registered under `name` are removed. If the protocol name
    /// does not exist, this is a no-op.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// # use cloudllm::Agent;
    /// # use cloudllm::clients::openai::OpenAIClient;
    /// # use std::sync::Arc;
    /// # async {
    /// # let agent = Agent::new("a1", "Agent", Arc::new(OpenAIClient::new_with_model_string("key", "gpt-4o")));
    /// agent.remove_protocol("custom").await;
    /// # };
    /// ```
    pub async fn remove_protocol(&self, name: &str) {
        let mut registry = self.tool_registry.write().await;
        registry.remove_protocol(name);
        self.emit(AgentEvent::ProtocolRemoved {
            agent_id: self.id.clone(),
            agent_name: self.name.clone(),
            protocol_name: name.to_string(),
        })
        .await;
    }

    /// List all tool names currently available to this agent.
    ///
    /// Returns the name of every tool across all registered protocols.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// # use cloudllm::Agent;
    /// # use cloudllm::clients::openai::OpenAIClient;
    /// # use std::sync::Arc;
    /// # async {
    /// # let agent = Agent::new("a1", "Agent", Arc::new(OpenAIClient::new_with_model_string("key", "gpt-4o")));
    /// let tools = agent.list_tools().await;
    /// for name in &tools {
    ///     println!("Available: {}", name);
    /// }
    /// # };
    /// ```
    pub async fn list_tools(&self) -> Vec<String> {
        let registry = self.tool_registry.read().await;
        registry
            .list_tools()
            .iter()
            .map(|m| m.name.clone())
            .collect()
    }

    // ---- ThoughtChain convenience methods ----

    /// Append a thought to this agent's [`ThoughtChain`].
    ///
    /// This is a convenience wrapper that acquires a write lock on the chain
    /// and calls [`ThoughtChain::append`].  If no chain is attached, the call
    /// is a silent no-op.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use cloudllm::Agent;
    /// use cloudllm::thought_chain::{ThoughtChain, ThoughtType};
    /// use cloudllm::clients::openai::OpenAIClient;
    /// use std::sync::Arc;
    /// use std::path::PathBuf;
    /// use tokio::sync::RwLock;
    ///
    /// # async {
    /// let chain = Arc::new(RwLock::new(
    ///     ThoughtChain::open(&PathBuf::from("/tmp/ch"), "a1", "Agent", None, None).unwrap()
    /// ));
    /// let agent = Agent::new(
    ///     "a1", "Agent",
    ///     Arc::new(OpenAIClient::new_with_model_string("key", "gpt-4o")),
    /// ).with_thought_chain(chain);
    ///
    /// agent.commit(ThoughtType::Finding, "Latency increased 3x").await.unwrap();
    /// agent.commit(ThoughtType::Decision, "Enable caching").await.unwrap();
    ///
    /// let entries = agent.thought_entries().await.unwrap();
    /// assert_eq!(entries.len(), 2);
    /// # };
    /// ```
    pub async fn commit(
        &self,
        entry_type: ThoughtType,
        content: impl Into<String>,
    ) -> io::Result<()> {
        if let Some(chain) = &self.thought_chain {
            let thought_type = entry_type.clone();
            let mut chain = chain.write().await;
            chain.append(&self.id, entry_type, &content.into())?;
            self.emit(AgentEvent::ThoughtCommitted {
                agent_id: self.id.clone(),
                agent_name: self.name.clone(),
                thought_type,
            })
            .await;
        }
        Ok(())
    }

    /// Return a snapshot of all thoughts in this agent's chain.
    ///
    /// Returns `None` if no [`ThoughtChain`] is attached, or `Some(vec)` with
    /// cloned thoughts otherwise.
    pub async fn thought_entries(&self) -> Option<Vec<Thought>> {
        if let Some(chain) = &self.thought_chain {
            let chain = chain.read().await;
            Some(chain.thoughts().to_vec())
        } else {
            None
        }
    }

    // ---- Resume constructors ----

    /// Resume an agent from a specific thought in an existing chain.
    ///
    /// Resolves the context graph at `thought_index` via
    /// [`ThoughtChain::resolve_context`] and injects the resulting bootstrap
    /// prompt into a fresh [`LLMSession`].  The agent starts with the
    /// critical reasoning context already in its history, ready to continue
    /// where it left off.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use cloudllm::Agent;
    /// use cloudllm::thought_chain::{ThoughtChain, ThoughtType};
    /// use cloudllm::clients::openai::OpenAIClient;
    /// use std::sync::Arc;
    /// use std::path::PathBuf;
    /// use tokio::sync::RwLock;
    ///
    /// // Assume a chain was previously populated
    /// let chain = Arc::new(RwLock::new(
    ///     ThoughtChain::open(
    ///         &PathBuf::from("chains"), "a1", "Agent", Some("ML"), None,
    ///     ).unwrap()
    /// ));
    ///
    /// let agent = Agent::resume_from_chain(
    ///     "a1", "Agent",
    ///     Arc::new(OpenAIClient::new_with_model_string("key", "gpt-4o")),
    ///     128_000,
    ///     chain,
    ///     5, // resume from thought #5
    /// ).unwrap();
    /// ```
    pub fn resume_from_chain(
        id: impl Into<String>,
        name: impl Into<String>,
        client: Arc<dyn ClientWrapper>,
        max_tokens: usize,
        chain: Arc<RwLock<ThoughtChain>>,
        thought_index: u64,
    ) -> Result<Self, Box<dyn Error + Send + Sync>> {
        let id = id.into();
        let name = name.into();
        let mut session = LLMSession::new(client, String::new(), max_tokens);

        // We need to block briefly to read the chain — this runs during construction
        let chain_guard = chain.try_read().map_err(|_| {
            Box::new(io::Error::other("ThoughtChain is locked")) as Box<dyn Error + Send + Sync>
        })?;
        let bootstrap = chain_guard.to_bootstrap_prompt(thought_index);
        drop(chain_guard);

        if !bootstrap.is_empty() {
            session.inject_message(Role::System, bootstrap);
        }

        Ok(Self {
            id,
            name,
            expertise: None,
            personality: None,
            metadata: HashMap::new(),
            session,
            tool_registry: Arc::new(RwLock::new(ToolRegistry::empty())),
            grok_tools: Vec::new(),
            openai_tools: Vec::new(),
            context_strategy: Box::new(TrimStrategy::default()),
            thought_chain: Some(chain),
            event_handler: None,
        })
    }

    /// Resume an agent from the latest thought in an existing chain.
    ///
    /// Convenience wrapper around [`resume_from_chain`](Agent::resume_from_chain)
    /// that automatically targets the last thought in the chain.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use cloudllm::Agent;
    /// use cloudllm::thought_chain::ThoughtChain;
    /// use cloudllm::clients::openai::OpenAIClient;
    /// use std::sync::Arc;
    /// use std::path::PathBuf;
    /// use tokio::sync::RwLock;
    ///
    /// let chain = Arc::new(RwLock::new(
    ///     ThoughtChain::open(
    ///         &PathBuf::from("chains"), "a1", "Agent", None, None,
    ///     ).unwrap()
    /// ));
    ///
    /// let agent = Agent::resume_from_latest(
    ///     "a1", "Agent",
    ///     Arc::new(OpenAIClient::new_with_model_string("key", "gpt-4o")),
    ///     128_000,
    ///     chain,
    /// ).unwrap();
    /// ```
    pub fn resume_from_latest(
        id: impl Into<String>,
        name: impl Into<String>,
        client: Arc<dyn ClientWrapper>,
        max_tokens: usize,
        chain: Arc<RwLock<ThoughtChain>>,
    ) -> Result<Self, Box<dyn Error + Send + Sync>> {
        let last_index = {
            let guard = chain.try_read().map_err(|_| {
                Box::new(io::Error::other("ThoughtChain is locked")) as Box<dyn Error + Send + Sync>
            })?;
            guard.thoughts().last().map(|t| t.index).unwrap_or(0)
        };
        Self::resume_from_chain(id, name, client, max_tokens, chain, last_index)
    }

    // ---- fork() — replaces Clone for parallel execution ----

    /// Create a lightweight copy for parallel execution.
    ///
    /// The fork shares the same tool registry and thought chain (via `Arc`)
    /// but has a **fresh, empty** [`LLMSession`] backed by the same client.
    /// Identity fields (`id`, `name`, `expertise`, `personality`, `metadata`)
    /// are cloned.  The context strategy is reset to [`TrimStrategy`] since
    /// forked agents are typically short-lived.
    ///
    /// This replaces `Clone` — `Agent` is intentionally not `Clone` because
    /// cloning a full `LLMSession` (with its bumpalo arena) would be expensive
    /// and semantically misleading.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use cloudllm::Agent;
    /// use cloudllm::clients::openai::OpenAIClient;
    /// use std::sync::Arc;
    ///
    /// let agent = Agent::new(
    ///     "analyst", "Analyst",
    ///     Arc::new(OpenAIClient::new_with_model_string("key", "gpt-4o")),
    /// ).with_expertise("Cloud Architecture");
    ///
    /// // Fork for parallel execution — identity is preserved
    /// let forked = agent.fork();
    /// assert_eq!(forked.id, agent.id);
    /// assert_eq!(forked.expertise, agent.expertise);
    /// ```
    pub fn fork(&self) -> Self {
        let client = self.session.client().clone();
        let max_tokens = self.session.get_max_tokens();
        self.emit_sync(AgentEvent::Forked {
            agent_id: self.id.clone(),
            agent_name: self.name.clone(),
        });
        Self {
            id: self.id.clone(),
            name: self.name.clone(),
            expertise: self.expertise.clone(),
            personality: self.personality.clone(),
            metadata: self.metadata.clone(),
            session: LLMSession::new(client, String::new(), max_tokens),
            tool_registry: Arc::clone(&self.tool_registry),
            grok_tools: self.grok_tools.clone(),
            openai_tools: self.openai_tools.clone(),
            context_strategy: Box::new(TrimStrategy::default()),
            thought_chain: self.thought_chain.clone(),
            event_handler: self.event_handler.clone(),
        }
    }

    /// Create a lightweight copy that also carries forward session context.
    ///
    /// Like [`fork`](Agent::fork), the clone shares tool registry and thought
    /// chain via `Arc`, but additionally copies the current system prompt and
    /// conversation history into the new session. Use this when a parallel
    /// task needs the accumulated context (e.g., later rounds of an
    /// orchestration).
    pub fn fork_with_context(&self) -> Self {
        let client = self.session.client().clone();
        let max_tokens = self.session.get_max_tokens();
        let mut session = LLMSession::new(client, String::new(), max_tokens);

        // Copy system prompt
        session.set_system_prompt(self.session.system_prompt_text().to_string());

        // Copy conversation history
        for msg in self.session.get_conversation_history() {
            session.inject_message(msg.role.clone(), msg.content.to_string());
        }

        self.emit_sync(AgentEvent::ForkedWithContext {
            agent_id: self.id.clone(),
            agent_name: self.name.clone(),
        });
        Self {
            id: self.id.clone(),
            name: self.name.clone(),
            expertise: self.expertise.clone(),
            personality: self.personality.clone(),
            metadata: self.metadata.clone(),
            session,
            tool_registry: Arc::clone(&self.tool_registry),
            grok_tools: self.grok_tools.clone(),
            openai_tools: self.openai_tools.clone(),
            context_strategy: Box::new(TrimStrategy::default()),
            thought_chain: self.thought_chain.clone(),
            event_handler: self.event_handler.clone(),
        }
    }

    // ---- Session-based methods for hub-routed orchestration ----

    /// Set the agent's LLMSession system prompt, augmented with expertise and personality.
    ///
    /// Called by orchestration modes during setup so each agent has its system
    /// prompt configured once before generation begins.
    pub fn set_system_prompt(&mut self, base_prompt: &str) {
        let augmented = self.augment_system_prompt(base_prompt);
        self.session.set_system_prompt(augmented);
        self.emit_sync(AgentEvent::SystemPromptSet {
            agent_id: self.id.clone(),
            agent_name: self.name.clone(),
        });
    }

    /// Inject a message into this agent's session history without sending to the LLM.
    ///
    /// Used by orchestration hub-routing to feed specific messages (e.g., other
    /// agents' responses) into this agent's context before calling [`send`](Agent::send).
    pub fn receive_message(&mut self, role: Role, content: String) {
        self.session.inject_message(role, content);
        self.emit_sync(AgentEvent::MessageReceived {
            agent_id: self.id.clone(),
            agent_name: self.name.clone(),
        });
    }

    /// Return the number of messages in this agent's session history.
    ///
    /// Useful for orchestration to check whether the agent has been initialized.
    pub fn session_history_len(&self) -> usize {
        self.session.get_conversation_history().len()
    }

    /// Send a message using the agent's own session history.
    ///
    /// This is the primary method used by orchestration modes. Unlike
    /// [`generate_with_tokens`](Agent::generate_with_tokens) which takes an
    /// external conversation history, this method relies on the session's
    /// accumulated messages (populated via [`receive_message`] and prior
    /// `send` calls). The session handles system prompt, history, and
    /// auto-trimming automatically.
    ///
    /// # Tool Loop
    ///
    /// After the initial LLM call, the method checks whether the response
    /// contains a tool call (`{"tool_call": {"name": "...", "parameters": {...}}}`).
    /// If so, the tool is executed via the [`ToolRegistry`], the result is
    /// fed back into the session as a follow-up message, and the LLM is
    /// called again. This loop runs for up to 5 iterations.
    ///
    /// # Events Emitted
    ///
    /// The following [`AgentEvent`]s are emitted during `send()` (in order):
    /// 1. [`SendStarted`](AgentEvent::SendStarted) — at entry
    /// 2. [`LLMCallStarted`](AgentEvent::LLMCallStarted) — before each LLM call
    /// 3. [`LLMCallCompleted`](AgentEvent::LLMCallCompleted) — after each LLM call
    /// 4. [`ToolCallDetected`](AgentEvent::ToolCallDetected) — when a tool call is parsed
    /// 5. [`ToolExecutionCompleted`](AgentEvent::ToolExecutionCompleted) — after tool execution
    /// 6. [`ToolMaxIterationsReached`](AgentEvent::ToolMaxIterationsReached) — if the loop cap is hit
    /// 7. [`SendCompleted`](AgentEvent::SendCompleted) — at exit
    pub async fn send(
        &mut self,
        user_message: &str,
    ) -> Result<AgentResponse, Box<dyn Error + Send + Sync>> {
        let preview_len = 120.min(user_message.len());
        let preview_end = user_message
            .char_indices()
            .nth(preview_len)
            .map(|(i, _)| i)
            .unwrap_or(user_message.len());
        let message_preview = user_message[..preview_end].to_string();
        self.emit(AgentEvent::SendStarted {
            agent_id: self.id.clone(),
            agent_name: self.name.clone(),
            message_preview: message_preview.clone(),
        })
        .await;
        self.emit_planner(PlannerEvent::TurnStarted {
            plan_id: self.id.clone(),
            message_preview,
        })
        .await;

        // Build tool description string to append to user message
        let mut message_with_tools = user_message.to_string();
        {
            let registry = self.tool_registry.read().await;
            let tools = registry.list_tools();
            if !tools.is_empty() {
                message_with_tools.push_str("\n\nYou have access to the following tools:\n");
                for tool_metadata in tools {
                    message_with_tools.push_str(&format!(
                        "- {}: {}\n",
                        tool_metadata.name, tool_metadata.description
                    ));
                    if !tool_metadata.parameters.is_empty() {
                        message_with_tools.push_str("  Parameters:\n");
                        for param in &tool_metadata.parameters {
                            message_with_tools.push_str(&format!(
                                "    - {} ({:?}): {}\n",
                                param.name,
                                param.param_type,
                                param.description.as_deref().unwrap_or("No description")
                            ));
                        }
                    }
                }
                message_with_tools.push_str(
                    "\nTo use a tool, respond with a JSON object in the following format:\n\
                     {\"tool_call\": {\"name\": \"tool_name\", \"parameters\": {...}}}\n\
                     After tool execution, I'll provide the result and you can continue.\n",
                );
            }
        }

        let grok_tools = if self.grok_tools.is_empty() {
            None
        } else {
            Some(self.grok_tools.clone())
        };
        let openai_tools = if self.openai_tools.is_empty() {
            None
        } else {
            Some(self.openai_tools.clone())
        };

        // Tool execution loop
        let max_tool_iterations = 5;
        let mut tool_iteration = 0;
        let mut total_input_tokens = 0;
        let mut total_output_tokens = 0;
        let mut total_tokens = 0;

        // First call uses the user message
        self.emit(AgentEvent::LLMCallStarted {
            agent_id: self.id.clone(),
            agent_name: self.name.clone(),
            iteration: 1,
        })
        .await;
        self.emit_planner(PlannerEvent::LLMCallStarted {
            plan_id: self.id.clone(),
            iteration: 1,
        })
        .await;

        let response = self
            .session
            .send_message(
                Role::User,
                message_with_tools,
                grok_tools.clone(),
                openai_tools.clone(),
            )
            .await
            .map_err(|e| {
                Box::new(crate::orchestration::OrchestrationError::ExecutionFailed(
                    e.to_string(),
                )) as Box<dyn Error + Send + Sync>
            })?;

        if let Some(usage) = self.session.client().get_last_usage().await {
            total_input_tokens += usage.input_tokens;
            total_output_tokens += usage.output_tokens;
            total_tokens += usage.total_tokens;
        }

        let first_response_length = response.content.len();
        self.emit(AgentEvent::LLMCallCompleted {
            agent_id: self.id.clone(),
            agent_name: self.name.clone(),
            iteration: 1,
            tokens_used: if total_tokens > 0 {
                Some(TokenUsage {
                    input_tokens: total_input_tokens,
                    output_tokens: total_output_tokens,
                    total_tokens,
                })
            } else {
                None
            },
            response_length: first_response_length,
        })
        .await;
        self.emit_planner(PlannerEvent::LLMCallCompleted {
            plan_id: self.id.clone(),
            iteration: 1,
            response_length: first_response_length,
        })
        .await;

        let mut current_response = response.content.to_string();

        loop {
            let tool_call = self.parse_tool_call(&current_response);
            if let Some(tool_call) = tool_call {
                if tool_iteration >= max_tool_iterations {
                    self.emit(AgentEvent::ToolMaxIterationsReached {
                        agent_id: self.id.clone(),
                        agent_name: self.name.clone(),
                    })
                    .await;
                    self.emit_planner(PlannerEvent::ToolMaxIterationsReached {
                        plan_id: self.id.clone(),
                    })
                    .await;
                    current_response = format!(
                        "{}\n\n[Warning: Maximum tool iterations reached]",
                        current_response
                    );
                    break;
                }
                tool_iteration += 1;

                let tool_params_snapshot = tool_call.parameters.clone();
                let tool_name = tool_call.name.clone();

                self.emit(AgentEvent::ToolCallDetected {
                    agent_id: self.id.clone(),
                    agent_name: self.name.clone(),
                    tool_name: tool_name.clone(),
                    parameters: tool_params_snapshot.clone(),
                    iteration: tool_iteration,
                })
                .await;
                self.emit_planner(PlannerEvent::ToolCallDetected {
                    plan_id: self.id.clone(),
                    tool_name: tool_name.clone(),
                    parameters: tool_params_snapshot.clone(),
                    iteration: tool_iteration,
                })
                .await;

                // Execute the tool
                let tool_result = {
                    let registry = self.tool_registry.read().await;
                    registry
                        .execute_tool(&tool_call.name, tool_call.parameters)
                        .await
                };

                let (tool_result_message, tool_success, tool_error, tool_output) =
                    match &tool_result {
                        Ok(result) => {
                            if result.success {
                                (
                                    format!(
                                        "Tool '{}' executed successfully. Result: {}",
                                        tool_call.name,
                                        serde_json::to_string_pretty(&result.output)
                                            .unwrap_or_else(|_| format!("{:?}", result.output))
                                    ),
                                    true,
                                    None,
                                    Some(result.output.clone()),
                                )
                            } else {
                                let err = result
                                    .error
                                    .clone()
                                    .unwrap_or_else(|| "Unknown error".to_string());
                                (
                                    format!("Tool '{}' failed. Error: {}", tool_call.name, err),
                                    false,
                                    Some(err),
                                    None,
                                )
                            }
                        }
                        Err(e) => (
                            format!("Tool execution error: {}", e),
                            false,
                            Some(e.to_string()),
                            None,
                        ),
                    };

                self.emit(AgentEvent::ToolExecutionCompleted {
                    agent_id: self.id.clone(),
                    agent_name: self.name.clone(),
                    tool_name: tool_call.name.clone(),
                    parameters: tool_params_snapshot.clone(),
                    success: tool_success,
                    error: tool_error.clone(),
                    result: tool_output.clone(),
                    iteration: tool_iteration,
                })
                .await;
                self.emit_planner(PlannerEvent::ToolExecutionCompleted {
                    plan_id: self.id.clone(),
                    tool_name: tool_name,
                    parameters: tool_params_snapshot,
                    success: tool_success,
                    error: tool_error,
                    result: tool_output,
                    iteration: tool_iteration,
                })
                .await;

                // Send tool result back through session
                let next_iteration = tool_iteration + 1;
                self.emit(AgentEvent::LLMCallStarted {
                    agent_id: self.id.clone(),
                    agent_name: self.name.clone(),
                    iteration: next_iteration,
                })
                .await;
                self.emit_planner(PlannerEvent::LLMCallStarted {
                    plan_id: self.id.clone(),
                    iteration: next_iteration,
                })
                .await;

                let follow_up = self
                    .session
                    .send_message(
                        Role::User,
                        tool_result_message,
                        grok_tools.clone(),
                        openai_tools.clone(),
                    )
                    .await
                    .map_err(|e| {
                        Box::new(crate::orchestration::OrchestrationError::ExecutionFailed(
                            e.to_string(),
                        )) as Box<dyn Error + Send + Sync>
                    })?;

                if let Some(usage) = self.session.client().get_last_usage().await {
                    total_input_tokens += usage.input_tokens;
                    total_output_tokens += usage.output_tokens;
                    total_tokens += usage.total_tokens;
                }

                let follow_up_response_length = follow_up.content.len();
                self.emit(AgentEvent::LLMCallCompleted {
                    agent_id: self.id.clone(),
                    agent_name: self.name.clone(),
                    iteration: next_iteration,
                    tokens_used: if total_tokens > 0 {
                        Some(TokenUsage {
                            input_tokens: total_input_tokens,
                            output_tokens: total_output_tokens,
                            total_tokens,
                        })
                    } else {
                        None
                    },
                    response_length: follow_up_response_length,
                })
                .await;
                self.emit_planner(PlannerEvent::LLMCallCompleted {
                    plan_id: self.id.clone(),
                    iteration: next_iteration,
                    response_length: follow_up_response_length,
                })
                .await;

                current_response = follow_up.content.to_string();
            } else {
                break;
            }
        }

        let tokens_used = if total_tokens > 0 {
            Some(TokenUsage {
                input_tokens: total_input_tokens,
                output_tokens: total_output_tokens,
                total_tokens,
            })
        } else {
            None
        };

        let final_response_length = current_response.len();
        self.emit(AgentEvent::SendCompleted {
            agent_id: self.id.clone(),
            agent_name: self.name.clone(),
            tokens_used: tokens_used.clone(),
            tool_calls_made: tool_iteration,
            response_length: final_response_length,
        })
        .await;
        self.emit_planner(PlannerEvent::TurnCompleted {
            plan_id: self.id.clone(),
            tokens_used: tokens_used.clone(),
            response_length: final_response_length,
            tool_calls_made: tool_iteration,
        })
        .await;

        Ok(AgentResponse {
            content: current_response,
            tokens_used,
        })
    }

    // ---- Accessor for client ----

    /// Borrow the underlying [`ClientWrapper`] from the session.
    ///
    /// Useful for creating new sessions or agents that share the same LLM
    /// provider connection.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use cloudllm::Agent;
    /// use cloudllm::LLMSession;
    /// use cloudllm::clients::openai::OpenAIClient;
    /// use std::sync::Arc;
    ///
    /// let agent = Agent::new(
    ///     "a1", "Agent",
    ///     Arc::new(OpenAIClient::new_with_model_string("key", "gpt-4o")),
    /// );
    ///
    /// // Create a standalone session sharing the same provider
    /// let session = LLMSession::new(
    ///     agent.client().clone(),
    ///     "system prompt".into(),
    ///     8_192,
    /// );
    /// ```
    pub fn client(&self) -> &Arc<dyn ClientWrapper> {
        self.session.client()
    }

    /// Generate the system prompt augmented with the agent's expertise and personality.
    fn augment_system_prompt(&self, base_prompt: &str) -> String {
        let mut prompt = String::new();

        prompt.push_str(&format!("You are {}.\n", self.name));

        if let Some(expertise) = &self.expertise {
            prompt.push_str(&format!("Your expertise: {}\n", expertise));
        }

        if let Some(personality) = &self.personality {
            prompt.push_str(&format!("Your approach: {}\n", personality));
        }

        prompt.push('\n');
        prompt.push_str(base_prompt);

        prompt
    }

    /// Send a message to the backing model and capture the response plus token usage.
    ///
    /// Unlike [`send`](Agent::send) which uses the agent's own session, this
    /// method takes an explicit system prompt and conversation history. Each
    /// call builds a fresh message array — there is no session state carried
    /// between calls. This makes it suitable for one-shot interactions or
    /// when you manage conversation history externally.
    ///
    /// The tool loop (up to 5 iterations) works identically to `send()` and
    /// emits the same [`AgentEvent`]s.
    ///
    /// # Parameters
    ///
    /// - `system_prompt` — Base system prompt (augmented with the agent's
    ///   expertise and personality automatically).
    /// - `user_message` — The user's question or instruction.
    /// - `conversation_history` — Prior messages to include as context.
    ///
    /// # Returns
    ///
    /// An [`AgentResponse`] containing the final text and cumulative token usage.
    pub async fn generate_with_tokens(
        &self,
        system_prompt: &str,
        user_message: &str,
        conversation_history: &[crate::orchestration::OrchestrationMessage],
    ) -> Result<AgentResponse, Box<dyn Error + Send + Sync>> {
        let gwt_preview_len = 120.min(user_message.len());
        let gwt_preview_end = user_message
            .char_indices()
            .nth(gwt_preview_len)
            .map(|(i, _)| i)
            .unwrap_or(user_message.len());
        self.emit(AgentEvent::SendStarted {
            agent_id: self.id.clone(),
            agent_name: self.name.clone(),
            message_preview: user_message[..gwt_preview_end].to_string(),
        })
        .await;

        let augmented_system = self.augment_system_prompt(system_prompt);

        // Build message array
        let mut messages = Vec::new();

        // System message with tool information if available
        let mut system_with_tools = augmented_system.clone();
        {
            let registry = self.tool_registry.read().await;
            let tools = registry.list_tools();
            if !tools.is_empty() {
                system_with_tools.push_str("\n\nYou have access to the following tools:\n");
                for tool_metadata in tools {
                    system_with_tools.push_str(&format!(
                        "- {}: {}\n",
                        tool_metadata.name, tool_metadata.description
                    ));
                    if !tool_metadata.parameters.is_empty() {
                        system_with_tools.push_str("  Parameters:\n");
                        for param in &tool_metadata.parameters {
                            system_with_tools.push_str(&format!(
                                "    - {} ({:?}): {}\n",
                                param.name,
                                param.param_type,
                                param.description.as_deref().unwrap_or("No description")
                            ));
                        }
                    }
                }
                system_with_tools.push_str(
                    "\nTo use a tool, respond with a JSON object in the following format:\n\
                     {\"tool_call\": {\"name\": \"tool_name\", \"parameters\": {...}}}\n\
                     After tool execution, I'll provide the result and you can continue.\n",
                );
            }
        }

        messages.push(Message {
            role: Role::System,
            content: Arc::from(system_with_tools.as_str()),
        });

        // Add conversation history
        for msg in conversation_history {
            messages.push(Message {
                role: msg.role.clone(),
                content: msg.content.clone(),
            });
        }

        // Add current user message
        messages.push(Message {
            role: Role::User,
            content: Arc::from(user_message),
        });

        // Tool execution loop - allow up to 5 tool calls to prevent infinite loops
        let max_tool_iterations = 5;
        let mut tool_iteration = 0;
        let final_response;

        // Track cumulative token usage across all LLM calls (including tool iterations)
        let mut total_input_tokens = 0;
        let mut total_output_tokens = 0;
        let mut total_tokens = 0;

        let mut gwt_llm_iteration: usize = 0;

        loop {
            gwt_llm_iteration += 1;

            // Send to LLM
            let grok_tools = if self.grok_tools.is_empty() {
                None
            } else {
                Some(self.grok_tools.clone())
            };
            let openai_tools = if self.openai_tools.is_empty() {
                None
            } else {
                Some(self.openai_tools.clone())
            };

            self.emit(AgentEvent::LLMCallStarted {
                agent_id: self.id.clone(),
                agent_name: self.name.clone(),
                iteration: gwt_llm_iteration,
            })
            .await;

            let response = self
                .session
                .client()
                .send_message(&messages, grok_tools, openai_tools)
                .await
                .map_err(|e| {
                    Box::new(crate::orchestration::OrchestrationError::ExecutionFailed(
                        e.to_string(),
                    )) as Box<dyn Error + Send + Sync>
                })?;

            // Track token usage from this call
            if let Some(usage) = self.session.client().get_last_usage().await {
                total_input_tokens += usage.input_tokens;
                total_output_tokens += usage.output_tokens;
                total_tokens += usage.total_tokens;
            }

            let current_response = response.content.to_string();

            self.emit(AgentEvent::LLMCallCompleted {
                agent_id: self.id.clone(),
                agent_name: self.name.clone(),
                iteration: gwt_llm_iteration,
                tokens_used: if total_tokens > 0 {
                    Some(TokenUsage {
                        input_tokens: total_input_tokens,
                        output_tokens: total_output_tokens,
                        total_tokens,
                    })
                } else {
                    None
                },
                response_length: current_response.len(),
            })
            .await;

            // Check if we have tools and if the response contains a tool call
            let tool_call = self.parse_tool_call(&current_response);
            if let Some(tool_call) = tool_call {
                if tool_iteration >= max_tool_iterations {
                    self.emit(AgentEvent::ToolMaxIterationsReached {
                        agent_id: self.id.clone(),
                        agent_name: self.name.clone(),
                    })
                    .await;
                    // Max iterations reached, return with warning
                    final_response = format!(
                        "{}\n\n[Warning: Maximum tool iterations reached]",
                        current_response
                    );
                    break;
                }

                tool_iteration += 1;

                let gwt_params_snapshot = tool_call.parameters.clone();

                self.emit(AgentEvent::ToolCallDetected {
                    agent_id: self.id.clone(),
                    agent_name: self.name.clone(),
                    tool_name: tool_call.name.clone(),
                    parameters: gwt_params_snapshot.clone(),
                    iteration: tool_iteration,
                })
                .await;

                // Execute the tool via the registry
                let tool_result = {
                    let registry = self.tool_registry.read().await;
                    registry
                        .execute_tool(&tool_call.name, tool_call.parameters)
                        .await
                };

                // Add assistant's tool call to messages
                messages.push(Message {
                    role: Role::Assistant,
                    content: response.content.clone(),
                });

                // Add tool result to messages
                let (tool_result_message, gwt_tool_success, gwt_tool_error, gwt_tool_output) =
                    match &tool_result {
                        Ok(result) => {
                            if result.success {
                                (
                                    format!(
                                        "Tool '{}' executed successfully. Result: {}",
                                        tool_call.name,
                                        serde_json::to_string_pretty(&result.output)
                                            .unwrap_or_else(|_| format!("{:?}", result.output))
                                    ),
                                    true,
                                    None,
                                    Some(result.output.clone()),
                                )
                            } else {
                                let err = result
                                    .error
                                    .clone()
                                    .unwrap_or_else(|| "Unknown error".to_string());
                                (
                                    format!("Tool '{}' failed. Error: {}", tool_call.name, err),
                                    false,
                                    Some(err),
                                    None,
                                )
                            }
                        }
                        Err(e) => (
                            format!("Tool execution error: {}", e),
                            false,
                            Some(e.to_string()),
                            None,
                        ),
                    };

                self.emit(AgentEvent::ToolExecutionCompleted {
                    agent_id: self.id.clone(),
                    agent_name: self.name.clone(),
                    tool_name: tool_call.name.clone(),
                    parameters: gwt_params_snapshot,
                    success: gwt_tool_success,
                    error: gwt_tool_error,
                    result: gwt_tool_output,
                    iteration: tool_iteration,
                })
                .await;

                messages.push(Message {
                    role: Role::User,
                    content: Arc::from(tool_result_message.as_str()),
                });

                // Continue loop to get next response
                continue;
            } else {
                // No tool call found, return the response
                final_response = current_response;
                break;
            }
        }

        let tokens_used = if total_tokens > 0 {
            Some(TokenUsage {
                input_tokens: total_input_tokens,
                output_tokens: total_output_tokens,
                total_tokens,
            })
        } else {
            None
        };

        self.emit(AgentEvent::SendCompleted {
            agent_id: self.id.clone(),
            agent_name: self.name.clone(),
            tokens_used: tokens_used.clone(),
            tool_calls_made: tool_iteration,
            response_length: final_response.len(),
        })
        .await;

        Ok(AgentResponse {
            content: final_response,
            tokens_used,
        })
    }

    /// Convenience wrapper around [`generate_with_tokens`](Agent::generate_with_tokens)
    /// that discards token-usage data and returns only the response text.
    ///
    /// Useful for quick one-shot interactions where you don't need to track
    /// token consumption. All events are still emitted normally.
    pub async fn generate(
        &self,
        system_prompt: &str,
        user_message: &str,
        conversation_history: &[crate::orchestration::OrchestrationMessage],
    ) -> Result<String, Box<dyn Error + Send + Sync>> {
        let response = self
            .generate_with_tokens(system_prompt, user_message, conversation_history)
            .await?;
        Ok(response.content)
    }

    /// Parse a tool call from an LLM response.
    ///
    /// Scans the response text for a JSON fragment matching the pattern:
    /// ```json
    /// {"tool_call": {"name": "tool_name", "parameters": {...}}}
    /// ```
    ///
    /// The method uses brace-counting to find the matching closing `}` rather
    /// than parsing the entire response as JSON. This handles the common case
    /// where the LLM wraps the tool call in surrounding prose.
    ///
    /// Returns `Some(ToolCall)` if a valid tool call is found, `None` otherwise.
    /// Only the *first* tool call in the response is extracted.
    fn parse_tool_call(&self, response: &str) -> Option<ToolCall> {
        // Locate the start of the tool_call JSON fragment
        if let Some(start_idx) = response.find("{\"tool_call\"") {
            // Use brace-counting to find the matching closing brace
            let mut brace_count = 0;
            let mut end_idx = start_idx;
            let chars: Vec<char> = response.chars().collect();

            for (i, ch) in chars.iter().enumerate().skip(start_idx) {
                if *ch == '{' {
                    brace_count += 1;
                } else if *ch == '}' {
                    brace_count -= 1;
                    if brace_count == 0 {
                        end_idx = i + 1;
                        break;
                    }
                }
            }

            if end_idx > start_idx {
                let json_str = &response[start_idx..end_idx];
                if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(json_str) {
                    if let Some(tool_call_obj) = parsed.get("tool_call") {
                        if let (Some(name), Some(parameters)) = (
                            tool_call_obj.get("name").and_then(|v| v.as_str()),
                            tool_call_obj.get("parameters"),
                        ) {
                            return Some(ToolCall {
                                name: name.to_string(),
                                parameters: parameters.clone(),
                            });
                        }
                    }
                }
            }
        }

        None
    }
}

// SAFETY: Agent is Send + Sync because:
// - All public methods that access mutable state use proper synchronization (Arc<RwLock>)
// - LLMSession's arena (bumpalo::Bump) makes it !Sync, but generate_with_tokens only
//   accesses session.client() (which is Arc<dyn ClientWrapper>) — never mutating the arena
// - Box<dyn ContextStrategy> is bounded by Send + Sync
// - The Bump allocator is never accessed across thread boundaries through &self methods
unsafe impl Send for Agent {}
unsafe impl Sync for Agent {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_agent_creation() {
        use crate::clients::openai::OpenAIClient;

        let agent = Agent::new(
            "test-agent",
            "Test Agent",
            Arc::new(OpenAIClient::new_with_model_string("test-key", "gpt-4o")),
        );

        assert_eq!(agent.id, "test-agent");
        assert_eq!(agent.name, "Test Agent");
        assert!(agent.expertise.is_none());
        assert!(agent.personality.is_none());
    }

    #[test]
    fn test_agent_builder_pattern() {
        use crate::clients::openai::OpenAIClient;

        let agent = Agent::new(
            "analyst",
            "Technical Analyst",
            Arc::new(OpenAIClient::new_with_model_string("test-key", "gpt-4o")),
        )
        .with_expertise("Cloud Architecture")
        .with_personality("Direct and analytical")
        .with_metadata("department", "Engineering");

        assert_eq!(agent.expertise, Some("Cloud Architecture".to_string()));
        assert_eq!(agent.personality, Some("Direct and analytical".to_string()));
        assert_eq!(
            agent.metadata.get("department"),
            Some(&"Engineering".to_string())
        );
    }
}
