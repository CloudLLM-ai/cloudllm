//! Agent System
//!
//! This module provides the core `Agent` struct that represents an LLM-powered agent
//! with identity, expertise, personality, and optional tool access.
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
//! - **Tool Access**: Agents can be granted access to local or remote tools via ToolRegistry
//! - **ThoughtChain**: Optional persistent, hash-chained memory for findings/decisions
//! - **ContextStrategy**: Pluggable strategy for handling context window exhaustion
//! - **Expertise & Personality**: Optional attributes for behavior customization
//! - **Metadata**: Arbitrary key-value pairs for domain-specific extensions
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
use crate::cloudllm::llm_session::LLMSession;
use crate::cloudllm::thought_chain::{Thought, ThoughtChain, ThoughtType};
use crate::cloudllm::tool_protocol::{ToolProtocol, ToolRegistry};
use openai_rust2::chat::{GrokTool, OpenAITool};
use std::collections::HashMap;
use std::error::Error;
use std::io;
use std::sync::Arc;
use tokio::sync::RwLock;

/// Parsed representation of a tool call emitted by an agent.
#[derive(Debug, Clone)]
struct ToolCall {
    /// Name of the tool to execute.
    name: String,
    /// JSON payload describing the arguments.
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

/// Represents an agent with identity, expertise, and optional tool access.
///
/// Agents are LLM-powered entities that can:
/// - Generate responses based on system prompts and user messages
/// - Access tools through a ToolRegistry (single or multi-protocol)
/// - Maintain per-agent conversation memory via LLMSession
/// - Persist findings and decisions via ThoughtChain
/// - Handle context window exhaustion via pluggable ContextStrategy
/// - Be orchestrated by orchestrations or used independently
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
        registry.add_protocol(name, protocol).await
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
            let mut chain = chain.write().await;
            chain.append(&self.id, entry_type, &content.into())?;
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
            Box::new(io::Error::other(
                "ThoughtChain is locked",
            )) as Box<dyn Error + Send + Sync>
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
                Box::new(io::Error::other(
                    "ThoughtChain is locked",
                )) as Box<dyn Error + Send + Sync>
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
        }
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
    /// This is used internally by orchestrations and can be used for direct agent interaction.
    pub async fn generate_with_tokens(
        &self,
        system_prompt: &str,
        user_message: &str,
        conversation_history: &[crate::orchestration::OrchestrationMessage],
    ) -> Result<AgentResponse, Box<dyn Error + Send + Sync>> {
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

        loop {
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

            // Check if we have tools and if the response contains a tool call
            let tool_call = self.parse_tool_call(&current_response);
            if let Some(tool_call) = tool_call {
                if tool_iteration >= max_tool_iterations {
                    // Max iterations reached, return with warning
                    final_response = format!(
                        "{}\n\n[Warning: Maximum tool iterations reached]",
                        current_response
                    );
                    break;
                }

                tool_iteration += 1;

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
                let tool_result_message = match tool_result {
                    Ok(result) => {
                        if result.success {
                            format!(
                                "Tool '{}' executed successfully. Result: {}",
                                tool_call.name,
                                serde_json::to_string_pretty(&result.output)
                                    .unwrap_or_else(|_| format!("{:?}", result.output))
                            )
                        } else {
                            format!(
                                "Tool '{}' failed. Error: {}",
                                tool_call.name,
                                result.error.unwrap_or_else(|| "Unknown error".to_string())
                            )
                        }
                    }
                    Err(e) => format!("Tool execution error: {}", e),
                };

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

        Ok(AgentResponse {
            content: final_response,
            tokens_used,
        })
    }

    /// Convenience wrapper around `generate_with_tokens` that discards usage data.
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

    /// Parse a tool call emitted by a model response.
    ///
    /// The method looks for JSON fragments in the format
    /// `{ "tool_call": { "name": "...", "parameters": { ... }}}`.
    fn parse_tool_call(&self, response: &str) -> Option<ToolCall> {
        // Try to find JSON object in the response
        // Look for the pattern {"tool_call": ...}
        if let Some(start_idx) = response.find("{\"tool_call\"") {
            // Find the matching closing brace
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
