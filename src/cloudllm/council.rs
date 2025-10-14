//! Multi-Agent Council System
//!
//! This module provides abstractions for orchestrating multiple LLM agents in various
//! collaboration patterns including parallel execution, round-robin discussion, moderated
//! panels, hierarchical problem solving, and debate-style convergence.
//!
//! # Architecture
//!
//! ```text
//! Council
//!   ├─ Agent 1 (OpenAI GPT-4) + Tools
//!   ├─ Agent 2 (Claude) + Tools
//!   └─ Agent 3 (Grok) + Tools
//! ```
//!
//! # Example
//!
//! ```rust,no_run
//! use cloudllm::council::{Council, CouncilMode, Agent};
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

use crate::client_wrapper::{ClientWrapper, Message, Role, TokenUsage};
use crate::cloudllm::tool_protocol::ToolRegistry;
use chrono::{DateTime, Utc};
use std::collections::HashMap;
use std::error::Error;
use std::fmt;
use std::sync::Arc;

/// Represents a parsed tool call from an LLM response
#[derive(Debug, Clone)]
struct ToolCall {
    name: String,
    parameters: serde_json::Value,
}

/// Response from an agent generation including token usage
#[derive(Debug, Clone)]
struct AgentResponse {
    content: String,
    tokens_used: Option<TokenUsage>,
}

/// Represents an agent with identity, expertise, and optional tool access
pub struct Agent {
    pub id: String,
    pub name: String,
    pub client: Arc<dyn ClientWrapper>,
    pub expertise: Option<String>,
    pub personality: Option<String>,
    pub metadata: HashMap<String, String>,
    pub tool_registry: Option<Arc<ToolRegistry>>,
}

impl Agent {
    pub fn new(
        id: impl Into<String>,
        name: impl Into<String>,
        client: Arc<dyn ClientWrapper>,
    ) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            client,
            expertise: None,
            personality: None,
            metadata: HashMap::new(),
            tool_registry: None,
        }
    }

    pub fn with_expertise(mut self, expertise: impl Into<String>) -> Self {
        self.expertise = Some(expertise.into());
        self
    }

    pub fn with_personality(mut self, personality: impl Into<String>) -> Self {
        self.personality = Some(personality.into());
        self
    }

    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }

    pub fn with_tools(mut self, registry: Arc<ToolRegistry>) -> Self {
        self.tool_registry = Some(registry);
        self
    }

    /// Generate the system prompt augmented with agent's expertise and personality
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

    /// Send a message and get a response from this agent with token tracking
    async fn generate_with_tokens(
        &self,
        system_prompt: &str,
        user_message: &str,
        conversation_history: &[CouncilMessage],
    ) -> Result<AgentResponse, Box<dyn Error + Send + Sync>> {
        let augmented_system = self.augment_system_prompt(system_prompt);

        // Build message array
        let mut messages = Vec::new();

        // System message with tool information if available
        let mut system_with_tools = augmented_system.clone();
        if let Some(registry) = &self.tool_registry {
            // Get tools from the protocol
            if let Ok(tools) = registry.protocol().list_tools().await {
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
            let response = self
                .client
                .send_message(&messages, None)
                .await
                .map_err(|e| {
                    Box::new(CouncilError::ExecutionFailed(e.to_string()))
                        as Box<dyn Error + Send + Sync>
                })?;

            // Track token usage from this call
            if let Some(usage) = self.client.get_last_usage().await {
                total_input_tokens += usage.input_tokens;
                total_output_tokens += usage.output_tokens;
                total_tokens += usage.total_tokens;
            }

            let current_response = response.content.to_string();

            // Check if we have tools and if the response contains a tool call
            if let Some(registry) = &self.tool_registry {
                if let Some(tool_call) = self.parse_tool_call(&current_response) {
                    if tool_iteration >= max_tool_iterations {
                        // Max iterations reached, return with warning
                        final_response = format!(
                            "{}\n\n[Warning: Maximum tool iterations reached]",
                            current_response
                        );
                        break;
                    }

                    tool_iteration += 1;

                    // Execute the tool via the protocol
                    let tool_result = registry
                        .protocol()
                        .execute(&tool_call.name, tool_call.parameters)
                        .await;

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
                                    serde_json::to_string_pretty(&result.output).unwrap_or_else(
                                        |_| format!("{:?}", result.output)
                                    )
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
            } else {
                // No tools available, return the response
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

    /// Send a message and get a response from this agent (convenience wrapper)
    pub async fn generate(
        &self,
        system_prompt: &str,
        user_message: &str,
        conversation_history: &[CouncilMessage],
    ) -> Result<String, Box<dyn Error + Send + Sync>> {
        let response = self.generate_with_tokens(system_prompt, user_message, conversation_history).await?;
        Ok(response.content)
    }

    /// Parse tool call from LLM response
    /// Looks for JSON in format: {"tool_call": {"name": "tool_name", "parameters": {...}}}
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
    pub timestamp: DateTime<Utc>,
    pub agent_id: Option<String>,
    pub agent_name: Option<String>,
    pub role: Role,
    pub content: Arc<str>,
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
    pub messages: Vec<CouncilMessage>,
    pub round: usize,
    pub is_complete: bool,
    pub convergence_score: Option<f32>,
    pub total_tokens_used: usize,
}

/// Error types for council operations
#[derive(Debug, Clone)]
pub enum CouncilError {
    AgentNotFound(String),
    InvalidMode(String),
    ExecutionFailed(String),
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
    pub id: String,
    pub name: String,
    agents: HashMap<String, Agent>,
    agent_order: Vec<String>, // Preserve insertion order for round-robin
    mode: CouncilMode,
    conversation_history: Vec<CouncilMessage>,
    system_context: String,
    max_tokens: usize,
}

impl Council {
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

    pub fn with_mode(mut self, mode: CouncilMode) -> Self {
        self.mode = mode;
        self
    }

    pub fn with_system_context(mut self, context: impl Into<String>) -> Self {
        self.system_context = context.into();
        self
    }

    pub fn with_max_tokens(mut self, max_tokens: usize) -> Self {
        self.max_tokens = max_tokens;
        self
    }

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

    pub fn remove_agent(&mut self, id: &str) -> Option<Agent> {
        self.agent_order.retain(|aid| aid != id);
        self.agents.remove(id)
    }

    pub fn get_agent(&self, id: &str) -> Option<&Agent> {
        self.agents.get(id)
    }

    pub fn list_agents(&self) -> Vec<&Agent> {
        self.agent_order
            .iter()
            .filter_map(|id| self.agents.get(id))
            .collect()
    }

    /// Main entry point for council discussions
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
                let prompt_clone = prompt_owned.clone();

                // Create temporary agent for task
                let temp_agent = Agent {
                    id: agent_id.clone(),
                    name: agent_name.clone(),
                    client: client.clone(),
                    expertise: expertise.clone(),
                    personality: personality.clone(),
                    metadata: HashMap::new(),
                    tool_registry: None,
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

                        let msg = CouncilMessage::from_agent(agent_id, agent_name, agent_response.content);
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
                    .generate_with_tokens(&self.system_context, &round_prompt, &self.conversation_history)
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

        let mut all_messages = Vec::new();
        let mut total_tokens = 0;

        for _round in 0..rounds {
            // Moderator decides who should speak next
            let moderator_prompt = format!(
                "{}\n\nAvailable experts: {}\n\nWhich expert should address this question? \
                 Respond with ONLY the expert name.",
                prompt,
                self.list_agents()
                    .iter()
                    .filter(|a| a.id != moderator_id)
                    .map(|a| a.name.as_str())
                    .collect::<Vec<_>>()
                    .join(", ")
            );

            let moderator_result = moderator
                .generate_with_tokens(
                    "You are a moderator. Your job is to select the most appropriate expert to answer each question.",
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
                let agent_result = agent
                    .generate_with_tokens(&self.system_context, prompt, &self.conversation_history)
                    .await?;

                // Track agent tokens
                if let Some(usage) = agent_result.tokens_used {
                    total_tokens += usage.total_tokens;
                }

                let msg = CouncilMessage::from_agent(agent.id.clone(), agent.name.clone(), agent_result.content)
                    .with_metadata("moderator", moderator_id.to_string());

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
                let agent = self.agents.get(agent_id).ok_or_else(|| {
                    CouncilError::AgentNotFound(agent_id.clone())
                })?;

                let system_prompt = self.system_context.clone();
                let history = self.conversation_history.clone();
                let current_prompt = layer_results.clone();
                let agent_id = agent.id.clone();
                let agent_name = agent.name.clone();
                let client = agent.client.clone();
                let expertise = agent.expertise.clone();
                let personality = agent.personality.clone();

                let temp_agent = Agent {
                    id: agent_id.clone(),
                    name: agent_name.clone(),
                    client: client.clone(),
                    expertise: expertise.clone(),
                    personality: personality.clone(),
                    metadata: HashMap::new(),
                    tool_registry: None,
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

                        let msg = CouncilMessage::from_agent(agent_id, agent_name, agent_response.content)
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
                        .map(|m| format!(
                            "{}: {}",
                            m.agent_name.as_ref().unwrap(),
                            m.content
                        ))
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
                    .generate_with_tokens(&self.system_context, &debate_prompt, &self.conversation_history)
                    .await;

                match result {
                    Ok(agent_response) => {
                        // Track tokens
                        if let Some(usage) = agent_response.tokens_used {
                            total_tokens += usage.total_tokens;
                        }

                        let msg =
                            CouncilMessage::from_agent(agent.id.clone(), agent.name.clone(), agent_response.content)
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
                let convergence_score = self.calculate_convergence_score(&all_messages, &round_messages);
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
        let previous_round: Vec<_> = all_messages
            .iter()
            .rev()
            .take(num_agents)
            .rev()
            .collect();

        if previous_round.len() != current_round.len() {
            return 0.0;
        }

        // Calculate average Jaccard similarity between corresponding agents' messages
        let mut total_similarity = 0.0;
        let mut comparison_count = 0;

        for i in 0..previous_round.len() {
            if let (Some(prev_msg), Some(curr_msg)) = (
                previous_round.get(i),
                current_round.get(i),
            ) {
                let similarity = self.jaccard_similarity(
                    &prev_msg.content,
                    &curr_msg.content,
                );
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

    pub fn clear_history(&mut self) {
        self.conversation_history.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::client_wrapper::TokenUsage;
    use async_trait::async_trait;

    struct MockClient {
        name: String,
        response: String,
    }

    #[async_trait]
    impl ClientWrapper for MockClient {
        async fn send_message(
            &self,
            _messages: &[Message],
            _optional_search_parameters: Option<openai_rust2::chat::SearchParameters>,
        ) -> Result<Message, Box<dyn Error>> {
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

        let mut council = Council::new("test-council", "Test Council")
            .with_mode(CouncilMode::Parallel);

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

        let mut council = Council::new("test-council", "Test Council")
            .with_mode(CouncilMode::RoundRobin);

        council.add_agent(agent1).unwrap();
        council.add_agent(agent2).unwrap();

        let response = council.discuss("Test question", 2).await.unwrap();

        assert_eq!(response.messages.len(), 4); // 2 agents * 2 rounds
        assert!(response.is_complete);
    }

    #[tokio::test]
    async fn test_agent_with_tool_execution() {
        use crate::tool_adapters::CustomToolAdapter;
        use crate::tool_protocol::{ToolMetadata, ToolParameter, ToolParameterType, ToolResult};
        use tokio::sync::Mutex as TokioMutex;

        // Create a custom tool adapter
        let adapter = CustomToolAdapter::new();

        // Register a simple calculator tool
        adapter
            .register_tool(
                ToolMetadata::new("add", "Adds two numbers")
                    .with_parameter(
                        ToolParameter::new("a", ToolParameterType::Number).required(),
                    )
                    .with_parameter(
                        ToolParameter::new("b", ToolParameterType::Number).required(),
                    ),
                Arc::new(|params| {
                    let a = params["a"].as_f64().unwrap_or(0.0);
                    let b = params["b"].as_f64().unwrap_or(0.0);
                    Ok(ToolResult::success(serde_json::json!({"sum": a + b})))
                }),
            )
            .await;

        let registry = Arc::new(ToolRegistry::new(Arc::new(adapter)));

        // Create a mock client that will respond with a tool call
        struct ToolCallingMockClient {
            call_count: Arc<TokioMutex<usize>>,
        }

        #[async_trait]
        impl ClientWrapper for ToolCallingMockClient {
            async fn send_message(
                &self,
                messages: &[Message],
                _optional_search_parameters: Option<openai_rust2::chat::SearchParameters>,
            ) -> Result<Message, Box<dyn Error>> {
                let mut count = self.call_count.lock().await;
                *count += 1;

                // First call: return a tool call
                // Second call: return final response
                let response = if *count == 1 {
                    // Check that system message includes tool information
                    let system_msg = &messages[0];
                    // The system message should contain the tool name and description
                    let system_content = system_msg.content.as_ref();
                    if !system_content.contains("add") || !system_content.contains("Adds two numbers") {
                        panic!("System message doesn't contain tool information. Content:\n{}", system_content);
                    }

                    // Return tool call
                    r#"{"tool_call": {"name": "add", "parameters": {"a": 5, "b": 3}}}"#
                } else {
                    // Verify tool result was provided
                    let last_msg = messages.last().unwrap();
                    let last_content = last_msg.content.as_ref();
                    if !last_content.contains("Tool 'add' executed successfully") {
                        panic!("Last message doesn't contain tool result. Content:\n{}", last_content);
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
        impl ClientWrapper for ConvergingMockClient {
            async fn send_message(
                &self,
                _messages: &[Message],
                _optional_search_parameters: Option<openai_rust2::chat::SearchParameters>,
            ) -> Result<Message, Box<dyn Error>> {
                let mut count = self.call_count.lock().await;
                *count += 1;

                // Simulate agents converging on a solution over multiple rounds
                let response = match *count {
                    1 => format!("Agent {}: I think we should use approach A", self.agent_id),
                    2 => format!("Agent {}: Approach A seems reasonable but needs refinement", self.agent_id),
                    3 => format!("Agent {}: After consideration approach A with refinement is best solution", self.agent_id),
                    _ => format!("Agent {}: I agree approach A with refinement is the best solution", self.agent_id),
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

        let mut council = Council::new("debate-council", "Debate Council")
            .with_mode(CouncilMode::Debate {
                max_rounds: 5,
                convergence_threshold: Some(0.6), // 60% similarity threshold
            });

        council.add_agent(agent1).unwrap();
        council.add_agent(agent2).unwrap();

        let response = council.discuss("What approach should we use?", 5).await.unwrap();

        // Should converge before max rounds (5)
        assert!(response.round < 5);
        assert!(response.is_complete);

        // Should have a convergence score
        assert!(response.convergence_score.is_some());
        let score = response.convergence_score.unwrap();
        assert!(score >= 0.6, "Convergence score {} should be >= 0.6", score);
    }
}
