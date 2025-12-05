//! Agent System
//!
//! This module provides the core `Agent` struct that represents an LLM-powered agent
//! with identity, expertise, personality, and optional tool access.
//!
//! Agents are the fundamental building blocks for LLM applications in CloudLLM and can be used:
//! - Standalone for single-agent interactions
//! - In councils for multi-agent orchestration patterns
//! - In custom workflows for specialized use cases
//!
//! # Core Components
//!
//! - **Agent**: Represents an LLM agent with identity and capabilities
//! - **Tool Access**: Agents can be granted access to local or remote tools via ToolRegistry
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
use crate::cloudllm::tool_protocol::ToolRegistry;
use openai_rust2::chat::SearchParameters;
use std::collections::HashMap;
use std::error::Error;
use std::sync::Arc;

/// Parsed representation of a tool call emitted by an agent.
#[derive(Debug, Clone)]
struct ToolCall {
    /// Name of the tool to execute.
    name: String,
    /// JSON payload describing the arguments.
    parameters: serde_json::Value,
}

/// Response body returned after asking an agent to generate content.
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
/// - Maintain state through expertise and personality attributes
/// - Be orchestrated by councils or used independently
#[derive(Clone)]
pub struct Agent {
    /// Stable identifier referenced inside council orchestration.
    pub id: String,
    /// Human-readable display name for logging and UI surfaces.
    pub name: String,
    /// Underlying client used to communicate with the model backing this agent.
    pub client: Arc<dyn ClientWrapper>,
    /// Free-form description of the agent's strengths that will be embedded into prompts.
    pub expertise: Option<String>,
    /// Persona hints that help diversify the tone of generated responses.
    pub personality: Option<String>,
    /// Arbitrary metadata associated with the agent (e.g. department, region).
    pub metadata: HashMap<String, String>,
    /// Optional registry of tools the agent may invoke during generation.
    ///
    /// The registry supports both single-protocol (for backward compatibility)
    /// and multi-protocol (multiple MCP servers) modes. Tools are transparently
    /// routed to the appropriate protocol based on tool ownership.
    pub tool_registry: Option<Arc<ToolRegistry>>,
    /// Optional vector search configuration to forward to compatible providers.
    pub search_parameters: Option<SearchParameters>,
}

impl Agent {
    /// Create a new agent with the mandatory identity information.
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
            search_parameters: None,
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

    /// Grant the agent access to a registry of tools.
    ///
    /// The registry can contain tools from a single protocol or from multiple
    /// protocols (local, MCP servers, etc.). Tools are transparently routed to
    /// the appropriate protocol when executed.
    ///
    /// # Example: Single Protocol
    ///
    /// ```ignore
    /// let registry = Arc::new(ToolRegistry::new(
    ///     Arc::new(CustomToolProtocol::new())
    /// ));
    /// agent.with_tools(registry);
    /// ```
    ///
    /// # Example: Multiple Protocols
    ///
    /// ```ignore
    /// let mut registry = ToolRegistry::empty();
    /// registry.add_protocol("local", Arc::new(local_protocol)).await?;
    /// registry.add_protocol("youtube", Arc::new(youtube_mcp)).await?;
    /// agent.with_tools(Arc::new(registry));
    /// ```
    pub fn with_tools(mut self, registry: Arc<ToolRegistry>) -> Self {
        self.tool_registry = Some(registry);
        self
    }

    /// Forward search parameters to the underlying client wrapper if supported.
    pub fn with_search_parameters(mut self, search_parameters: SearchParameters) -> Self {
        self.search_parameters = Some(search_parameters);
        self
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
    /// This is used internally by councils and can be used for direct agent interaction.
    pub async fn generate_with_tokens(
        &self,
        system_prompt: &str,
        user_message: &str,
        conversation_history: &[crate::council::CouncilMessage],
    ) -> Result<AgentResponse, Box<dyn Error + Send + Sync>> {
        let augmented_system = self.augment_system_prompt(system_prompt);

        // Build message array
        let mut messages = Vec::new();

        // System message with tool information if available
        let mut system_with_tools = augmented_system.clone();
        if let Some(registry) = &self.tool_registry {
            // Get tools from the registry (works for both single and multi-protocol)
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
            let search_parameters = self.search_parameters.clone();
            let response = self
                .client
                .send_message(&messages, search_parameters)
                .await
                .map_err(|e| {
                    Box::new(crate::council::CouncilError::ExecutionFailed(e.to_string()))
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

                    // Execute the tool via the registry
                    let tool_result = registry
                        .execute_tool(&tool_call.name, tool_call.parameters)
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

    /// Convenience wrapper around `generate_with_tokens` that discards usage data.
    pub async fn generate(
        &self,
        system_prompt: &str,
        user_message: &str,
        conversation_history: &[crate::council::CouncilMessage],
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
        assert!(agent.tool_registry.is_none());
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
