//! Multi-Agent Council System
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
//!
//! # Architecture
//!
//! ```text
//! Council (orchestration engine)
//!   ├─ Agent 1 (OpenAI GPT-4)
//!   │   ├─ Tools: Local + YouTube MCP Server
//!   │   └─ Expertise: "Video Analysis"
//!   │
//!   ├─ Agent 2 (Claude)
//!   │   ├─ Tools: Local + GitHub MCP Server
//!   │   └─ Expertise: "Code Architecture"
//!   │
//!   └─ Agent 3 (Grok)
//!       ├─ Tools: Memory Protocol
//!       └─ Expertise: "System Coordination"
//! ```
//!
//! # Tool Integration
//!
//! Starting in 0.5.0, agents can access tools from multiple protocols simultaneously.
//! This enables rich multi-source interaction patterns in councils.
//!
//! # Example
//!
//! ```rust,no_run
//! use cloudllm::{Agent, council::{Council, CouncilMode}};
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
//! let mut council = Council::new("tech-council", "Technical Advisory Council")
//!     .with_mode(CouncilMode::Parallel)
//!     .with_max_tokens(8192);
//!
//! council.add_agent(agent).unwrap();
//!
//! let response = council.discuss("How should we architect this system?", 1).await.unwrap();
//! # };
//! ```

use crate::client_wrapper::Role;
use crate::cloudllm::agent::Agent;
use chrono::{DateTime, Utc};
use std::collections::HashMap;
use std::error::Error;
use std::fmt;
use std::sync::Arc;

/// Collaboration modes for councils
#[derive(Debug, Clone)]
pub enum CouncilMode {
    /// All agents respond in parallel to each prompt
    Parallel,
    /// Agents take turns responding in sequence
    RoundRobin,
    /// One moderator agent orchestrates the discussion
    Moderated { moderator_id: String },
    /// Hierarchical: workers submit to supervisors who submit to executives
    Hierarchical { layers: Vec<Vec<String>> },
    /// Debate: agents respond to each other until convergence
    Debate {
        max_rounds: usize,
        convergence_threshold: Option<f32>,
    },
}

/// A message in a council discussion
#[derive(Debug, Clone)]
pub struct CouncilMessage {
    /// Timestamp the message was added to the conversation.
    pub timestamp: DateTime<Utc>,
    /// Identifier of the agent that generated the message (if any).
    pub agent_id: Option<String>,
    /// Display name of the contributing agent (if any).
    pub agent_name: Option<String>,
    /// Message role (system/user/assistant).
    pub role: Role,
    /// Message body.
    pub content: Arc<str>,
    /// Arbitrary metadata associated with the message.
    pub metadata: HashMap<String, String>,
}

impl CouncilMessage {
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

    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }
}

/// Response from a council discussion
#[derive(Debug)]
pub struct CouncilResponse {
    /// Messages generated during the discussion.
    pub messages: Vec<CouncilMessage>,
    /// Number of rounds executed.
    pub round: usize,
    /// Whether the council completed according to the selected mode's termination criteria.
    pub is_complete: bool,
    /// Optional convergence metric for debate mode.
    pub convergence_score: Option<f32>,
    /// Approximate total tokens consumed across all agents.
    pub total_tokens_used: usize,
}

/// Error types for council operations
#[derive(Debug, Clone)]
pub enum CouncilError {
    /// Requested agent identifier could not be found.
    AgentNotFound(String),
    /// Invalid configuration encountered for the selected collaboration mode.
    InvalidMode(String),
    /// Underlying execution error surfaced while gathering responses.
    ExecutionFailed(String),
    /// Attempt to run a council action without any members.
    NoAgents,
}

impl fmt::Display for CouncilError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CouncilError::AgentNotFound(id) => write!(f, "Agent not found: {}", id),
            CouncilError::InvalidMode(msg) => write!(f, "Invalid mode: {}", msg),
            CouncilError::ExecutionFailed(msg) => write!(f, "Execution failed: {}", msg),
            CouncilError::NoAgents => write!(f, "No agents in council"),
        }
    }
}

impl Error for CouncilError {}

/// A council managing multiple agents in various collaboration modes
pub struct Council {
    /// Stable identifier for integrations and logging.
    pub id: String,
    /// Human readable name of the council.
    pub name: String,
    /// Storage for registered agents keyed by identifier.
    agents: HashMap<String, Agent>,
    /// Insertion order of agents used for deterministic iteration.
    agent_order: Vec<String>, // Preserve insertion order for round-robin
    /// Collaboration strategy used by the council.
    mode: CouncilMode,
    /// Ongoing conversation history maintained across rounds.
    conversation_history: Vec<CouncilMessage>,
    /// Global system context shared by all agents.
    system_context: String,
    /// Soft token budget used for pre-trimming.
    max_tokens: usize,
}

impl Council {
    /// Create a council with the provided identifiers and default to [`CouncilMode::Parallel`].
    pub fn new(id: impl Into<String>, name: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            agents: HashMap::new(),
            agent_order: Vec::new(),
            mode: CouncilMode::Parallel,
            conversation_history: Vec::new(),
            system_context: String::from(
                "You are participating in a collaborative discussion with other AI agents.",
            ),
            max_tokens: 8192,
        }
    }

    /// Select the collaboration mode the council will use during [`Council::discuss`].
    pub fn with_mode(mut self, mode: CouncilMode) -> Self {
        self.mode = mode;
        self
    }

    /// Override the default system context prompt shared across agents.
    pub fn with_system_context(mut self, context: impl Into<String>) -> Self {
        self.system_context = context.into();
        self
    }

    /// Override the soft token budget used for context trimming.
    pub fn with_max_tokens(mut self, max_tokens: usize) -> Self {
        self.max_tokens = max_tokens;
        self
    }

    /// Register a new agent with the council.
    pub fn add_agent(&mut self, agent: Agent) -> Result<(), CouncilError> {
        let id = agent.id.clone();
        if self.agents.contains_key(&id) {
            return Err(CouncilError::ExecutionFailed(format!(
                "Agent with id '{}' already exists",
                id
            )));
        }
        self.agent_order.push(id.clone());
        self.agents.insert(id, agent);
        Ok(())
    }

    /// Remove and return an agent by identifier.
    pub fn remove_agent(&mut self, id: &str) -> Option<Agent> {
        self.agent_order.retain(|aid| aid != id);
        self.agents.remove(id)
    }

    /// Borrow an agent by identifier.
    pub fn get_agent(&self, id: &str) -> Option<&Agent> {
        self.agents.get(id)
    }

    /// List agents in their insertion order.
    pub fn list_agents(&self) -> Vec<&Agent> {
        self.agent_order
            .iter()
            .filter_map(|id| self.agents.get(id))
            .collect()
    }

    /// Execute a discussion according to the configured [`CouncilMode`].
    ///
    /// The `prompt` is broadcast to the council according to the active mode.  The `rounds`
    /// parameter controls how many iterations to run for deterministic modes (parallel and
    /// round-robin).  Other modes interpret the value as a safety bound.
    pub async fn discuss(
        &mut self,
        prompt: &str,
        rounds: usize,
    ) -> Result<CouncilResponse, Box<dyn Error + Send + Sync>> {
        if self.agents.is_empty() {
            return Err(Box::new(CouncilError::NoAgents));
        }

        // Add user message to history
        self.conversation_history
            .push(CouncilMessage::new(Role::User, prompt));

        // Clone mode to avoid borrow issues
        let mode = self.mode.clone();

        match mode {
            CouncilMode::Parallel => self.execute_parallel(prompt, rounds).await,
            CouncilMode::RoundRobin => self.execute_round_robin(prompt, rounds).await,
            CouncilMode::Moderated { moderator_id } => {
                self.execute_moderated(prompt, rounds, &moderator_id).await
            }
            CouncilMode::Hierarchical { layers } => {
                self.execute_hierarchical(prompt, &layers).await
            }
            CouncilMode::Debate {
                max_rounds,
                convergence_threshold,
            } => {
                self.execute_debate(prompt, max_rounds, convergence_threshold)
                    .await
            }
        }
    }

    /// Execute parallel mode: all agents respond simultaneously
    async fn execute_parallel(
        &mut self,
        prompt: &str,
        rounds: usize,
    ) -> Result<CouncilResponse, Box<dyn Error + Send + Sync>> {
        let mut all_messages = Vec::new();
        let mut total_tokens = 0;

        for _round in 0..rounds {
            let mut round_messages = Vec::new();

            // Spawn all agent tasks in parallel
            let mut tasks = Vec::new();
            let prompt_owned = prompt.to_string(); // Convert to owned string

            for agent_id in &self.agent_order {
                let agent = self.agents.get(agent_id).unwrap();
                let system_prompt = self.system_context.clone();
                let history = self.conversation_history.clone();
                let agent_id = agent.id.clone();
                let agent_name = agent.name.clone();
                let client = agent.client.clone();
                let expertise = agent.expertise.clone();
                let personality = agent.personality.clone();
                let search_parameters = agent.search_parameters.clone();
                let tool_registry = agent.tool_registry.clone();
                let metadata = agent.metadata.clone();
                let prompt_clone = prompt_owned.clone();

                // Create temporary agent for task
                let temp_agent = Agent {
                    id: agent_id.clone(),
                    name: agent_name.clone(),
                    client: client.clone(),
                    expertise: expertise.clone(),
                    personality: personality.clone(),
                    metadata,
                    tool_registry,
                    search_parameters,
                };

                tasks.push(tokio::spawn(async move {
                    let result = temp_agent
                        .generate_with_tokens(&system_prompt, &prompt_clone, &history)
                        .await;
                    (agent_id, agent_name, result)
                }));
            }

            // Collect results
            for task in tasks {
                let (agent_id, agent_name, result) = task.await.map_err(|e| {
                    Box::new(CouncilError::ExecutionFailed(format!(
                        "Task join error: {}",
                        e
                    ))) as Box<dyn Error + Send + Sync>
                })?;

                match result {
                    Ok(agent_response) => {
                        // Track tokens
                        if let Some(usage) = agent_response.tokens_used {
                            total_tokens += usage.total_tokens;
                        }

                        let msg = CouncilMessage::from_agent(
                            agent_id,
                            agent_name,
                            agent_response.content,
                        );
                        round_messages.push(msg.clone());
                        self.conversation_history.push(msg);
                    }
                    Err(e) => {
                        eprintln!("Agent {} failed: {}", agent_id, e);
                    }
                }
            }

            all_messages.extend(round_messages);
        }

        Ok(CouncilResponse {
            messages: all_messages,
            round: rounds,
            is_complete: true,
            convergence_score: None,
            total_tokens_used: total_tokens,
        })
    }

    /// Execute round-robin mode: agents take turns
    async fn execute_round_robin(
        &mut self,
        prompt: &str,
        rounds: usize,
    ) -> Result<CouncilResponse, Box<dyn Error + Send + Sync>> {
        let mut all_messages: Vec<CouncilMessage> = Vec::new();
        let mut total_tokens = 0;

        for round in 0..rounds {
            for agent_id in self.agent_order.clone() {
                let agent = self.agents.get(&agent_id).unwrap();

                // Build context including what others have said
                let mut round_prompt = prompt.to_string();
                if round > 0 || !all_messages.is_empty() {
                    round_prompt.push_str("\n\nPrevious responses from other agents:\n");
                    for msg in &all_messages {
                        if let Some(name) = &msg.agent_name {
                            round_prompt.push_str(&format!("{}: {}\n\n", name, msg.content));
                        }
                    }
                }

                let result = agent
                    .generate_with_tokens(
                        &self.system_context,
                        &round_prompt,
                        &self.conversation_history,
                    )
                    .await;

                match result {
                    Ok(agent_response) => {
                        // Track tokens
                        if let Some(usage) = agent_response.tokens_used {
                            total_tokens += usage.total_tokens;
                        }

                        let msg = CouncilMessage::from_agent(
                            agent.id.clone(),
                            agent.name.clone(),
                            agent_response.content,
                        );
                        all_messages.push(msg.clone());
                        self.conversation_history.push(msg);
                    }
                    Err(e) => {
                        eprintln!("Agent {} failed: {}", agent_id, e);
                    }
                }
            }
        }

        Ok(CouncilResponse {
            messages: all_messages,
            round: rounds,
            is_complete: true,
            convergence_score: None,
            total_tokens_used: total_tokens,
        })
    }

    /// Execute moderated mode: moderator directs the discussion
    async fn execute_moderated(
        &mut self,
        prompt: &str,
        rounds: usize,
        moderator_id: &str,
    ) -> Result<CouncilResponse, Box<dyn Error + Send + Sync>> {
        let moderator = self
            .agents
            .get(moderator_id)
            .ok_or_else(|| CouncilError::AgentNotFound(moderator_id.to_string()))?;

        let mut all_messages: Vec<CouncilMessage> = Vec::new();
        let mut total_tokens = 0;

        for round_num in 0..rounds {
            // Build context for moderator including conversation so far
            let mut moderator_prompt = String::new();

            if round_num == 0 {
                // First round: use original prompt
                moderator_prompt.push_str(prompt);
            } else {
                // Subsequent rounds: include what's been said so far
                moderator_prompt.push_str(&format!("Original topic: {}\n\n", prompt));
                moderator_prompt.push_str("Discussion so far:\n");
                for msg in &all_messages {
                    if let Some(name) = &msg.agent_name {
                        moderator_prompt.push_str(&format!("{}: {}\n\n", name, msg.content));
                    }
                }
                moderator_prompt.push_str(
                    "Based on the discussion so far, who should speak next to continue the debate?",
                );
            }

            moderator_prompt.push_str(&format!(
                "\n\nAvailable experts: {}\n\nWhich expert should address this question? \
                 Respond with ONLY the expert name.",
                self.list_agents()
                    .iter()
                    .filter(|a| a.id != moderator_id)
                    .map(|a| a.name.as_str())
                    .collect::<Vec<_>>()
                    .join(", ")
            ));

            let moderator_result = moderator
                .generate_with_tokens(
                    "You are a moderator. Your job is to select the most appropriate expert to answer each question. \
                     Ensure both sides get fair representation by alternating between different experts.",
                    &moderator_prompt,
                    &self.conversation_history,
                )
                .await?;

            // Extract content before consuming tokens_used
            let selection = moderator_result.content.clone();

            // Track moderator tokens
            if let Some(usage) = moderator_result.tokens_used {
                total_tokens += usage.total_tokens;
            }

            // Find the selected agent (fuzzy match on name)
            let selected_agent = self
                .agents
                .values()
                .find(|a| {
                    a.id != moderator_id
                        && selection.to_lowercase().contains(&a.name.to_lowercase())
                })
                .or_else(|| self.agents.values().find(|a| a.id != moderator_id));

            if let Some(agent) = selected_agent {
                // Build context for the speaking agent
                let mut agent_prompt = String::new();
                if all_messages.is_empty() {
                    agent_prompt = prompt.to_string();
                } else {
                    agent_prompt.push_str(&format!("Topic: {}\n\n", prompt));
                    agent_prompt.push_str("Discussion so far:\n");
                    for msg in &all_messages {
                        if let Some(name) = &msg.agent_name {
                            agent_prompt.push_str(&format!("{}: {}\n\n", name, msg.content));
                        }
                    }
                    agent_prompt.push_str("Now it's your turn to respond.");
                }

                let agent_result = agent
                    .generate_with_tokens(
                        &self.system_context,
                        &agent_prompt,
                        &self.conversation_history,
                    )
                    .await?;

                // Track agent tokens
                if let Some(usage) = agent_result.tokens_used {
                    total_tokens += usage.total_tokens;
                }

                let msg = CouncilMessage::from_agent(
                    agent.id.clone(),
                    agent.name.clone(),
                    agent_result.content,
                )
                .with_metadata("moderator", moderator_id.to_string())
                .with_metadata("round", round_num.to_string());

                all_messages.push(msg.clone());
                self.conversation_history.push(msg);
            }
        }

        Ok(CouncilResponse {
            messages: all_messages,
            round: rounds,
            is_complete: true,
            convergence_score: None,
            total_tokens_used: total_tokens,
        })
    }

    /// Execute hierarchical mode: layer by layer processing
    async fn execute_hierarchical(
        &mut self,
        prompt: &str,
        layers: &[Vec<String>],
    ) -> Result<CouncilResponse, Box<dyn Error + Send + Sync>> {
        let mut all_messages = Vec::new();
        let mut layer_results = prompt.to_string();
        let mut total_tokens = 0;

        for (layer_idx, layer_agent_ids) in layers.iter().enumerate() {
            let mut layer_messages = Vec::new();

            // All agents in this layer work in parallel
            let mut tasks = Vec::new();

            for agent_id in layer_agent_ids {
                let agent = self
                    .agents
                    .get(agent_id)
                    .ok_or_else(|| CouncilError::AgentNotFound(agent_id.clone()))?;

                let system_prompt = self.system_context.clone();
                let history = self.conversation_history.clone();
                let current_prompt = layer_results.clone();
                let agent_id = agent.id.clone();
                let agent_name = agent.name.clone();
                let client = agent.client.clone();
                let expertise = agent.expertise.clone();
                let personality = agent.personality.clone();
                let search_parameters = agent.search_parameters.clone();
                let tool_registry = agent.tool_registry.clone();
                let metadata = agent.metadata.clone();

                let temp_agent = Agent {
                    id: agent_id.clone(),
                    name: agent_name.clone(),
                    client: client.clone(),
                    expertise: expertise.clone(),
                    personality: personality.clone(),
                    metadata,
                    tool_registry,
                    search_parameters,
                };

                tasks.push(tokio::spawn(async move {
                    let result = temp_agent
                        .generate_with_tokens(&system_prompt, &current_prompt, &history)
                        .await;
                    (agent_id, agent_name, result)
                }));
            }

            // Collect layer results
            for task in tasks {
                let (agent_id, agent_name, result) = task.await.map_err(|e| {
                    Box::new(CouncilError::ExecutionFailed(format!(
                        "Task join error: {}",
                        e
                    ))) as Box<dyn Error + Send + Sync>
                })?;

                match result {
                    Ok(agent_response) => {
                        // Track tokens
                        if let Some(usage) = agent_response.tokens_used {
                            total_tokens += usage.total_tokens;
                        }

                        let msg = CouncilMessage::from_agent(
                            agent_id,
                            agent_name,
                            agent_response.content,
                        )
                        .with_metadata("layer", layer_idx.to_string());
                        layer_messages.push(msg.clone());
                        self.conversation_history.push(msg);
                    }
                    Err(e) => {
                        eprintln!("Agent {} failed: {}", agent_id, e);
                    }
                }
            }

            // Synthesize layer results for next layer
            if layer_idx < layers.len() - 1 {
                layer_results = format!(
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

        Ok(CouncilResponse {
            messages: all_messages,
            round: layers.len(),
            is_complete: true,
            convergence_score: None,
            total_tokens_used: total_tokens,
        })
    }

    /// Execute debate mode: agents iterate until convergence
    async fn execute_debate(
        &mut self,
        prompt: &str,
        max_rounds: usize,
        convergence_threshold: Option<f32>,
    ) -> Result<CouncilResponse, Box<dyn Error + Send + Sync>> {
        let mut all_messages: Vec<CouncilMessage> = Vec::new();
        let threshold = convergence_threshold.unwrap_or(0.75); // Default: 75% similarity
        let mut converged = false;
        let mut final_convergence_score = None;
        let mut actual_rounds = 0;
        let mut total_tokens = 0;

        for round in 0..max_rounds {
            actual_rounds = round + 1;
            let mut round_messages = Vec::new();

            for agent_id in self.agent_order.clone() {
                let agent = self.agents.get(&agent_id).unwrap();

                let mut debate_prompt = format!("Round {} of debate: {}\n\n", round + 1, prompt);

                if !all_messages.is_empty() {
                    debate_prompt.push_str("Previous arguments:\n");
                    for msg in all_messages.iter().rev().take(self.agents.len() * 2) {
                        if let Some(name) = &msg.agent_name {
                            debate_prompt.push_str(&format!("{}: {}\n\n", name, msg.content));
                        }
                    }
                }

                debate_prompt.push_str(
                    "Consider the arguments presented and provide your position. \
                     Acknowledge strong points and challenge weak ones.",
                );

                let result = agent
                    .generate_with_tokens(
                        &self.system_context,
                        &debate_prompt,
                        &self.conversation_history,
                    )
                    .await;

                match result {
                    Ok(agent_response) => {
                        // Track tokens
                        if let Some(usage) = agent_response.tokens_used {
                            total_tokens += usage.total_tokens;
                        }

                        let msg = CouncilMessage::from_agent(
                            agent.id.clone(),
                            agent.name.clone(),
                            agent_response.content,
                        )
                        .with_metadata("round", round.to_string());
                        round_messages.push(msg.clone());
                        self.conversation_history.push(msg);
                    }
                    Err(e) => {
                        eprintln!("Agent {} failed: {}", agent_id, e);
                    }
                }
            }

            // Check for convergence after the first round
            if round > 0 && !round_messages.is_empty() {
                let convergence_score =
                    self.calculate_convergence_score(&all_messages, &round_messages);
                final_convergence_score = Some(convergence_score);

                if convergence_score >= threshold {
                    converged = true;
                    // Add the round messages before breaking
                    all_messages.extend(round_messages);
                    break;
                }
            }

            all_messages.extend(round_messages);
        }

        Ok(CouncilResponse {
            messages: all_messages,
            round: actual_rounds,
            is_complete: converged || actual_rounds >= max_rounds,
            convergence_score: final_convergence_score,
            total_tokens_used: total_tokens,
        })
    }

    /// Calculate convergence score between current and previous round messages
    /// Uses Jaccard similarity on word sets to detect when agents' positions converge
    fn calculate_convergence_score(
        &self,
        all_messages: &[CouncilMessage],
        current_round: &[CouncilMessage],
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

    /// Calculate Jaccard similarity between two texts based on word sets
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

    pub fn get_conversation_history(&self) -> &[CouncilMessage] {
        &self.conversation_history
    }

    /// Remove all historical messages, resetting the council state.
    pub fn clear_history(&mut self) {
        self.conversation_history.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::client_wrapper::{Message, TokenUsage};
    use async_trait::async_trait;

    struct MockClient {
        name: String,
        response: String,
    }

    #[async_trait]
    impl crate::client_wrapper::ClientWrapper for MockClient {
        async fn send_message(
            &self,
            _messages: &[Message],
            _optional_search_parameters: Option<openai_rust2::chat::SearchParameters>,
        ) -> Result<Message, Box<dyn std::error::Error>> {
            Ok(Message {
                role: Role::Assistant,
                content: Arc::from(self.response.as_str()),
            })
        }

        fn model_name(&self) -> &str {
            &self.name
        }

        async fn get_last_usage(&self) -> Option<TokenUsage> {
            None
        }
    }

    #[tokio::test]
    async fn test_agent_creation() {
        let client = Arc::new(MockClient {
            name: "mock".to_string(),
            response: "test response".to_string(),
        });

        let agent = Agent::new("agent1", "Test Agent", client)
            .with_expertise("Testing")
            .with_personality("Thorough and detail-oriented");

        assert_eq!(agent.id, "agent1");
        assert_eq!(agent.name, "Test Agent");
        assert_eq!(agent.expertise, Some("Testing".to_string()));
    }

    #[tokio::test]
    async fn test_council_parallel_mode() {
        let agent1 = Agent::new(
            "agent1",
            "Agent 1",
            Arc::new(MockClient {
                name: "mock1".to_string(),
                response: "Response from agent 1".to_string(),
            }),
        );

        let agent2 = Agent::new(
            "agent2",
            "Agent 2",
            Arc::new(MockClient {
                name: "mock2".to_string(),
                response: "Response from agent 2".to_string(),
            }),
        );

        let mut council =
            Council::new("test-council", "Test Council").with_mode(CouncilMode::Parallel);

        council.add_agent(agent1).unwrap();
        council.add_agent(agent2).unwrap();

        let response = council.discuss("Test question", 1).await.unwrap();

        assert_eq!(response.messages.len(), 2);
        assert!(response.is_complete);
    }

    #[tokio::test]
    async fn test_council_round_robin_mode() {
        let agent1 = Agent::new(
            "agent1",
            "Agent 1",
            Arc::new(MockClient {
                name: "mock1".to_string(),
                response: "First agent response".to_string(),
            }),
        );

        let agent2 = Agent::new(
            "agent2",
            "Agent 2",
            Arc::new(MockClient {
                name: "mock2".to_string(),
                response: "Second agent response".to_string(),
            }),
        );

        let mut council =
            Council::new("test-council", "Test Council").with_mode(CouncilMode::RoundRobin);

        council.add_agent(agent1).unwrap();
        council.add_agent(agent2).unwrap();

        let response = council.discuss("Test question", 2).await.unwrap();

        assert_eq!(response.messages.len(), 4); // 2 agents * 2 rounds
        assert!(response.is_complete);
    }

    #[tokio::test]
    async fn test_agent_with_tool_execution() {
        use crate::tool_protocol::{ToolMetadata, ToolParameter, ToolParameterType, ToolResult};
        use crate::tool_protocols::CustomToolProtocol;
        use tokio::sync::Mutex as TokioMutex;

        // Create a custom tool adapter
        let adapter = CustomToolProtocol::new();

        // Register a simple calculator tool
        adapter
            .register_tool(
                ToolMetadata::new("add", "Adds two numbers")
                    .with_parameter(ToolParameter::new("a", ToolParameterType::Number).required())
                    .with_parameter(ToolParameter::new("b", ToolParameterType::Number).required()),
                Arc::new(|params| {
                    let a = params["a"].as_f64().unwrap_or(0.0);
                    let b = params["b"].as_f64().unwrap_or(0.0);
                    Ok(ToolResult::success(serde_json::json!({"sum": a + b})))
                }),
            )
            .await;

        let mut registry = crate::tool_protocol::ToolRegistry::new(Arc::new(adapter));
        // Discover tools from the adapter
        registry.discover_tools_from_primary().await.unwrap();
        let registry = Arc::new(registry);

        // Create a mock client that will respond with a tool call
        struct ToolCallingMockClient {
            call_count: Arc<TokioMutex<usize>>,
        }

        #[async_trait]
        impl crate::client_wrapper::ClientWrapper for ToolCallingMockClient {
            async fn send_message(
                &self,
                messages: &[Message],
                _optional_search_parameters: Option<openai_rust2::chat::SearchParameters>,
            ) -> Result<Message, Box<dyn std::error::Error>> {
                let mut count = self.call_count.lock().await;
                *count += 1;

                // First call: return a tool call
                // Second call: return final response
                let response = if *count == 1 {
                    // Check that system message includes tool information
                    let system_msg = &messages[0];
                    // The system message should contain the tool name and description
                    let system_content = system_msg.content.as_ref();
                    if !system_content.contains("add")
                        || !system_content.contains("Adds two numbers")
                    {
                        panic!(
                            "System message doesn't contain tool information. Content:\n{}",
                            system_content
                        );
                    }

                    // Return tool call
                    r#"{"tool_call": {"name": "add", "parameters": {"a": 5, "b": 3}}}"#
                } else {
                    // Verify tool result was provided
                    let last_msg = messages.last().unwrap();
                    let last_content = last_msg.content.as_ref();
                    if !last_content.contains("Tool 'add' executed successfully") {
                        panic!(
                            "Last message doesn't contain tool result. Content:\n{}",
                            last_content
                        );
                    }

                    "The sum is 8"
                };

                Ok(Message {
                    role: Role::Assistant,
                    content: Arc::from(response),
                })
            }

            fn model_name(&self) -> &str {
                "tool-mock"
            }

            async fn get_last_usage(&self) -> Option<TokenUsage> {
                None
            }
        }

        let agent = Agent::new(
            "calculator",
            "Calculator Agent",
            Arc::new(ToolCallingMockClient {
                call_count: Arc::new(TokioMutex::new(0)),
            }),
        )
        .with_tools(registry);

        let response = agent
            .generate("You are a helpful assistant", "What is 5 + 3?", &[])
            .await
            .unwrap();

        assert_eq!(response, "The sum is 8");
    }

    #[tokio::test]
    async fn test_debate_mode_convergence() {
        use tokio::sync::Mutex as TokioMutex;

        // Mock client that returns increasingly similar responses
        struct ConvergingMockClient {
            call_count: Arc<TokioMutex<usize>>,
            agent_id: String,
        }

        #[async_trait]
        impl crate::client_wrapper::ClientWrapper for ConvergingMockClient {
            async fn send_message(
                &self,
                _messages: &[Message],
                _optional_search_parameters: Option<openai_rust2::chat::SearchParameters>,
            ) -> Result<Message, Box<dyn std::error::Error>> {
                let mut count = self.call_count.lock().await;
                *count += 1;

                // Simulate agents converging on a solution over multiple rounds
                let response = match *count {
                    1 => format!("Agent {}: I think we should use approach A", self.agent_id),
                    2 => format!(
                        "Agent {}: Approach A seems reasonable but needs refinement",
                        self.agent_id
                    ),
                    3 => format!(
                        "Agent {}: After consideration approach A with refinement is best solution",
                        self.agent_id
                    ),
                    _ => format!(
                        "Agent {}: I agree approach A with refinement is the best solution",
                        self.agent_id
                    ),
                };

                Ok(Message {
                    role: Role::Assistant,
                    content: Arc::from(response.as_str()),
                })
            }

            fn model_name(&self) -> &str {
                "converging-mock"
            }

            async fn get_last_usage(&self) -> Option<TokenUsage> {
                None
            }
        }

        let agent1 = Agent::new(
            "agent1",
            "Agent 1",
            Arc::new(ConvergingMockClient {
                call_count: Arc::new(TokioMutex::new(0)),
                agent_id: "1".to_string(),
            }),
        );

        let agent2 = Agent::new(
            "agent2",
            "Agent 2",
            Arc::new(ConvergingMockClient {
                call_count: Arc::new(TokioMutex::new(0)),
                agent_id: "2".to_string(),
            }),
        );

        let mut council =
            Council::new("debate-council", "Debate Council").with_mode(CouncilMode::Debate {
                max_rounds: 5,
                convergence_threshold: Some(0.6), // 60% similarity threshold
            });

        council.add_agent(agent1).unwrap();
        council.add_agent(agent2).unwrap();

        let response = council
            .discuss("What approach should we use?", 5)
            .await
            .unwrap();

        // Should converge before max rounds (5)
        assert!(response.round < 5);
        assert!(response.is_complete);

        // Should have a convergence score
        assert!(response.convergence_score.is_some());
        let score = response.convergence_score.unwrap();
        assert!(score >= 0.6, "Convergence score {} should be >= 0.6", score);
    }
}
