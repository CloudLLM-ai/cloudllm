//! Multi-participant LLM session management for orchestrating conversations between multiple LLMs.
//!
//! This module enables complex multi-agent scenarios such as:
//! - Panel discussions with moderators
//! - Round-robin conversations between multiple LLMs
//! - Hierarchical agent structures (councils, work groups, evaluators)
//! - Expert panels with different models and roles
//!
//! ## Example
//!
//! ```rust,no_run
//! use std::sync::Arc;
//! use cloudllm::client_wrapper::Role;
//! use cloudllm::clients::openai::OpenAIClient;
//! use cloudllm::multi_participant_session::{MultiParticipantSession, ParticipantRole, OrchestrationStrategy};
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! // Create different LLM clients
//! let openai_client = Arc::new(OpenAIClient::new("api_key", "gpt-4"));
//! let grok_client = Arc::new(OpenAIClient::new("api_key", "grok-model"));
//!
//! // Create a multi-participant session with a moderator
//! let mut session = MultiParticipantSession::new(
//!     "You are coordinating a discussion about AI safety.".to_string(),
//!     8192,
//!     OrchestrationStrategy::ModeratorLed,
//! );
//!
//! // Add participants with different roles
//! session.add_participant("GPT-4", openai_client, ParticipantRole::Moderator);
//! session.add_participant("Grok", grok_client, ParticipantRole::Panelist);
//!
//! // Send a message and get responses from all participants
//! let responses = session.send_message(
//!     Role::User,
//!     "What are the key challenges in AI alignment?".to_string(),
//!     None,
//! ).await?;
//!
//! for response in responses {
//!     println!("{}: {}", response.participant_name, response.content);
//! }
//! # Ok(())
//! # }
//! ```

use crate::client_wrapper::{ClientWrapper, Message, Role, TokenUsage};
use bumpalo::Bump;
use openai_rust2 as openai_rust;
use std::collections::HashMap;
use std::sync::Arc;

/// Defines the role of a participant in a multi-participant session.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ParticipantRole {
    /// The participant acts as a moderator, guiding the conversation and synthesizing responses.
    Moderator,
    /// A regular participant that contributes to the discussion.
    Panelist,
    /// An observer that receives all messages but doesn't actively respond in round-robin.
    Observer,
    /// An evaluator that assesses responses from other participants.
    Evaluator,
    /// A supervisor that oversees the work of other participants.
    Supervisor,
    /// A worker that performs specific tasks assigned by supervisors.
    Worker,
}

/// Strategy for orchestrating message flow between participants.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum OrchestrationStrategy {
    /// Messages are sent to all participants simultaneously.
    Broadcast,
    /// Messages are sent to participants in round-robin fashion.
    RoundRobin,
    /// Only the moderator receives and responds to messages initially, then distributes to others.
    ModeratorLed,
    /// Hierarchical: supervisors coordinate workers, then synthesize results.
    Hierarchical,
    /// Custom ordering based on participant priority.
    Custom,
}

/// Represents a participant in a multi-participant session.
pub struct Participant {
    /// Unique identifier/name for the participant.
    pub name: String,
    /// The LLM client wrapper for this participant.
    pub client: Arc<dyn ClientWrapper>,
    /// The role of this participant.
    pub role: ParticipantRole,
    /// Individual conversation history for this participant.
    pub conversation_history: Vec<Message>,
    /// Cached token counts for this participant's messages.
    pub cached_token_counts: Vec<usize>,
    /// Priority for custom orchestration (higher = earlier in sequence).
    pub priority: i32,
}

/// Response from a single participant.
#[derive(Clone)]
pub struct ParticipantResponse {
    /// Name of the participant who generated this response.
    pub participant_name: String,
    /// The role of the participant.
    pub participant_role: ParticipantRole,
    /// The actual message content.
    pub content: Arc<str>,
    /// Token usage for this specific response.
    pub token_usage: Option<TokenUsage>,
}

/// A multi-participant LLM session that orchestrates conversations between multiple LLM clients.
pub struct MultiParticipantSession {
    /// System prompt that applies to all participants.
    system_prompt: Message,
    /// Map of participant name to participant data.
    participants: HashMap<String, Participant>,
    /// Maximum tokens per participant.
    max_tokens: usize,
    /// Strategy for orchestrating message flow.
    orchestration_strategy: OrchestrationStrategy,
    /// Arena for efficient string allocation.
    arena: Bump,
    /// Shared conversation history visible to all participants.
    shared_history: Vec<Message>,
    /// Order of participants for round-robin or custom strategies.
    participant_order: Vec<String>,
}

impl MultiParticipantSession {
    /// Creates a new multi-participant session.
    ///
    /// # Arguments
    /// * `system_prompt` - The system prompt applied to all participants
    /// * `max_tokens` - Maximum token limit per participant
    /// * `orchestration_strategy` - How to orchestrate message flow
    pub fn new(
        system_prompt: String,
        max_tokens: usize,
        orchestration_strategy: OrchestrationStrategy,
    ) -> Self {
        let arena = Bump::new();
        let system_prompt_str = arena.alloc_str(&system_prompt);
        let system_prompt_arc: Arc<str> = Arc::from(system_prompt_str);

        let system_prompt_message = Message {
            role: Role::System,
            content: system_prompt_arc,
        };

        MultiParticipantSession {
            system_prompt: system_prompt_message,
            participants: HashMap::new(),
            max_tokens,
            orchestration_strategy,
            arena,
            shared_history: Vec::new(),
            participant_order: Vec::new(),
        }
    }

    /// Adds a participant to the session.
    ///
    /// # Arguments
    /// * `name` - Unique identifier for the participant
    /// * `client` - The LLM client wrapper
    /// * `role` - The role of this participant
    pub fn add_participant(
        &mut self,
        name: impl Into<String>,
        client: Arc<dyn ClientWrapper>,
        role: ParticipantRole,
    ) {
        let name = name.into();
        let participant = Participant {
            name: name.clone(),
            client,
            role,
            conversation_history: Vec::new(),
            cached_token_counts: Vec::new(),
            priority: 0,
        };

        self.participants.insert(name.clone(), participant);
        self.participant_order.push(name);
    }

    /// Adds a participant with a specific priority for custom orchestration.
    pub fn add_participant_with_priority(
        &mut self,
        name: impl Into<String>,
        client: Arc<dyn ClientWrapper>,
        role: ParticipantRole,
        priority: i32,
    ) {
        let name = name.into();
        let participant = Participant {
            name: name.clone(),
            client,
            role,
            conversation_history: Vec::new(),
            cached_token_counts: Vec::new(),
            priority,
        };

        self.participants.insert(name.clone(), participant);
        self.participant_order.push(name);

        // Sort by priority if using custom strategy
        if self.orchestration_strategy == OrchestrationStrategy::Custom {
            // Collect priorities into a temporary map to avoid borrow issues
            let priorities: HashMap<String, i32> = self
                .participants
                .iter()
                .map(|(name, p)| (name.clone(), p.priority))
                .collect();

            self.participant_order.sort_by(|a, b| {
                let a_priority = priorities.get(a).copied().unwrap_or(0);
                let b_priority = priorities.get(b).copied().unwrap_or(0);
                b_priority.cmp(&a_priority) // Higher priority first
            });
        }
    }

    /// Removes a participant from the session.
    pub fn remove_participant(&mut self, name: &str) -> Option<Participant> {
        self.participant_order.retain(|n| n != name);
        self.participants.remove(name)
    }

    /// Gets a reference to a participant by name.
    pub fn get_participant(&self, name: &str) -> Option<&Participant> {
        self.participants.get(name)
    }

    /// Gets a mutable reference to a participant by name.
    pub fn get_participant_mut(&mut self, name: &str) -> Option<&mut Participant> {
        self.participants.get_mut(name)
    }

    /// Lists all participant names.
    pub fn list_participants(&self) -> Vec<String> {
        self.participant_order.clone()
    }

    /// Gets the orchestration strategy.
    pub fn orchestration_strategy(&self) -> &OrchestrationStrategy {
        &self.orchestration_strategy
    }

    /// Changes the orchestration strategy.
    pub fn set_orchestration_strategy(&mut self, strategy: OrchestrationStrategy) {
        self.orchestration_strategy = strategy;
    }

    /// Sends a message to participants according to the orchestration strategy.
    ///
    /// Returns a vector of responses from participants, where the order and number
    /// of responses depend on the orchestration strategy.
    pub async fn send_message(
        &mut self,
        role: Role,
        content: String,
        optional_search_parameters: Option<openai_rust::chat::SearchParameters>,
    ) -> Result<Vec<ParticipantResponse>, Box<dyn std::error::Error>> {
        // Allocate message content in arena
        let content_str = self.arena.alloc_str(&content);
        let content_arc: Arc<str> = Arc::from(content_str);

        let message = Message {
            role: role.clone(),
            content: content_arc,
        };

        // Add to shared history
        self.shared_history.push(message.clone());

        // Route messages based on orchestration strategy
        match self.orchestration_strategy {
            OrchestrationStrategy::Broadcast => {
                self.broadcast_message(message, optional_search_parameters)
                    .await
            }
            OrchestrationStrategy::RoundRobin => {
                self.round_robin_message(message, optional_search_parameters)
                    .await
            }
            OrchestrationStrategy::ModeratorLed => {
                self.moderator_led_message(message, optional_search_parameters)
                    .await
            }
            OrchestrationStrategy::Hierarchical => {
                self.hierarchical_message(message, optional_search_parameters)
                    .await
            }
            OrchestrationStrategy::Custom => {
                self.custom_order_message(message, optional_search_parameters)
                    .await
            }
        }
    }

    /// Broadcasts a message to all participants simultaneously.
    async fn broadcast_message(
        &mut self,
        message: Message,
        optional_search_parameters: Option<openai_rust::chat::SearchParameters>,
    ) -> Result<Vec<ParticipantResponse>, Box<dyn std::error::Error>> {
        let mut responses = Vec::new();

        for participant_name in self.participant_order.clone() {
            if let Some(participant) = self.participants.get_mut(&participant_name) {
                // Add message to participant's history
                participant.conversation_history.push(message.clone());

                // Build request with system prompt + history
                let mut request_messages = vec![self.system_prompt.clone()];
                request_messages.extend_from_slice(&participant.conversation_history);

                // Send to participant
                match participant
                    .client
                    .send_message(&request_messages, optional_search_parameters.clone())
                    .await
                {
                    Ok(response) => {
                        let token_usage = participant.client.get_last_usage().await;

                        // Add response to participant's history
                        participant.conversation_history.push(response.clone());

                        responses.push(ParticipantResponse {
                            participant_name: participant.name.clone(),
                            participant_role: participant.role.clone(),
                            content: response.content,
                            token_usage,
                        });
                    }
                    Err(e) => {
                        eprintln!("Error from participant {}: {}", participant.name, e);
                    }
                }
            }
        }

        Ok(responses)
    }

    /// Sends message to participants in round-robin fashion.
    async fn round_robin_message(
        &mut self,
        message: Message,
        optional_search_parameters: Option<openai_rust::chat::SearchParameters>,
    ) -> Result<Vec<ParticipantResponse>, Box<dyn std::error::Error>> {
        let mut responses = Vec::new();
        let mut accumulated_context = message.clone();

        for participant_name in self.participant_order.clone() {
            if let Some(participant) = self.participants.get_mut(&participant_name) {
                // Add accumulated context to participant's history
                participant
                    .conversation_history
                    .push(accumulated_context.clone());

                // Build request
                let mut request_messages = vec![self.system_prompt.clone()];
                request_messages.extend_from_slice(&participant.conversation_history);

                // Send to participant
                match participant
                    .client
                    .send_message(&request_messages, optional_search_parameters.clone())
                    .await
                {
                    Ok(response) => {
                        let token_usage = participant.client.get_last_usage().await;

                        // Add response to participant's history
                        participant.conversation_history.push(response.clone());

                        // Create a new message that includes this participant's response
                        let context_str = self
                            .arena
                            .alloc_str(&format!("{}: {}", participant.name, response.content));
                        let context_arc: Arc<str> = Arc::from(context_str);
                        accumulated_context = Message {
                            role: Role::Assistant,
                            content: context_arc,
                        };

                        responses.push(ParticipantResponse {
                            participant_name: participant.name.clone(),
                            participant_role: participant.role.clone(),
                            content: response.content,
                            token_usage,
                        });
                    }
                    Err(e) => {
                        eprintln!("Error from participant {}: {}", participant.name, e);
                    }
                }
            }
        }

        Ok(responses)
    }

    /// Moderator-led message flow: moderator responds first, then others.
    async fn moderator_led_message(
        &mut self,
        message: Message,
        optional_search_parameters: Option<openai_rust::chat::SearchParameters>,
    ) -> Result<Vec<ParticipantResponse>, Box<dyn std::error::Error>> {
        let mut responses = Vec::new();

        // Find moderator
        let moderator_name = self
            .participant_order
            .iter()
            .find(|name| {
                self.participants
                    .get(*name)
                    .map(|p| p.role == ParticipantRole::Moderator)
                    .unwrap_or(false)
            })
            .cloned();

        // Get moderator response first
        let moderator_response = if let Some(mod_name) = moderator_name {
            if let Some(moderator) = self.participants.get_mut(&mod_name) {
                moderator.conversation_history.push(message.clone());

                let mut request_messages = vec![self.system_prompt.clone()];
                request_messages.extend_from_slice(&moderator.conversation_history);

                match moderator
                    .client
                    .send_message(&request_messages, optional_search_parameters.clone())
                    .await
                {
                    Ok(response) => {
                        let token_usage = moderator.client.get_last_usage().await;
                        moderator.conversation_history.push(response.clone());

                        let participant_response = ParticipantResponse {
                            participant_name: moderator.name.clone(),
                            participant_role: moderator.role.clone(),
                            content: response.content.clone(),
                            token_usage,
                        };
                        responses.push(participant_response);
                        Some(response)
                    }
                    Err(e) => {
                        eprintln!("Error from moderator: {}", e);
                        None
                    }
                }
            } else {
                None
            }
        } else {
            None
        };

        // Now get responses from other participants with moderator's response as context
        for participant_name in self.participant_order.clone() {
            if let Some(participant) = self.participants.get_mut(&participant_name) {
                if participant.role == ParticipantRole::Moderator {
                    continue; // Skip moderator, already handled
                }

                // Add original message
                participant.conversation_history.push(message.clone());

                // Add moderator's response if available
                if let Some(ref mod_response) = moderator_response {
                    participant.conversation_history.push(mod_response.clone());
                }

                let mut request_messages = vec![self.system_prompt.clone()];
                request_messages.extend_from_slice(&participant.conversation_history);

                match participant
                    .client
                    .send_message(&request_messages, optional_search_parameters.clone())
                    .await
                {
                    Ok(response) => {
                        let token_usage = participant.client.get_last_usage().await;
                        participant.conversation_history.push(response.clone());

                        responses.push(ParticipantResponse {
                            participant_name: participant.name.clone(),
                            participant_role: participant.role.clone(),
                            content: response.content,
                            token_usage,
                        });
                    }
                    Err(e) => {
                        eprintln!("Error from participant {}: {}", participant.name, e);
                    }
                }
            }
        }

        Ok(responses)
    }

    /// Hierarchical message flow: supervisors coordinate workers, then synthesize results.
    async fn hierarchical_message(
        &mut self,
        message: Message,
        optional_search_parameters: Option<openai_rust::chat::SearchParameters>,
    ) -> Result<Vec<ParticipantResponse>, Box<dyn std::error::Error>> {
        let mut responses = Vec::new();

        // Phase 1: Workers process the message
        let mut worker_responses = Vec::new();
        for participant_name in self.participant_order.clone() {
            if let Some(participant) = self.participants.get_mut(&participant_name) {
                if participant.role == ParticipantRole::Worker {
                    participant.conversation_history.push(message.clone());

                    let mut request_messages = vec![self.system_prompt.clone()];
                    request_messages.extend_from_slice(&participant.conversation_history);

                    match participant
                        .client
                        .send_message(&request_messages, optional_search_parameters.clone())
                        .await
                    {
                        Ok(response) => {
                            let token_usage = participant.client.get_last_usage().await;
                            participant.conversation_history.push(response.clone());

                            let participant_response = ParticipantResponse {
                                participant_name: participant.name.clone(),
                                participant_role: participant.role.clone(),
                                content: response.content.clone(),
                                token_usage,
                            };
                            worker_responses.push(participant_response.clone());
                            responses.push(participant_response);
                        }
                        Err(e) => {
                            eprintln!("Error from worker {}: {}", participant.name, e);
                        }
                    }
                }
            }
        }

        // Phase 2: Supervisors synthesize worker results
        if !worker_responses.is_empty() {
            // Build context with all worker responses
            let worker_context = worker_responses
                .iter()
                .map(|r| format!("{} ({}): {}", r.participant_name, "Worker", r.content))
                .collect::<Vec<_>>()
                .join("\n\n");

            let context_str = self.arena.alloc_str(&worker_context);
            let context_arc: Arc<str> = Arc::from(context_str);
            let synthesis_message = Message {
                role: Role::Assistant,
                content: context_arc,
            };

            for participant_name in self.participant_order.clone() {
                if let Some(participant) = self.participants.get_mut(&participant_name) {
                    if participant.role == ParticipantRole::Supervisor {
                        participant.conversation_history.push(message.clone());
                        participant
                            .conversation_history
                            .push(synthesis_message.clone());

                        let mut request_messages = vec![self.system_prompt.clone()];
                        request_messages.extend_from_slice(&participant.conversation_history);

                        match participant
                            .client
                            .send_message(&request_messages, optional_search_parameters.clone())
                            .await
                        {
                            Ok(response) => {
                                let token_usage = participant.client.get_last_usage().await;
                                participant.conversation_history.push(response.clone());

                                responses.push(ParticipantResponse {
                                    participant_name: participant.name.clone(),
                                    participant_role: participant.role.clone(),
                                    content: response.content,
                                    token_usage,
                                });
                            }
                            Err(e) => {
                                eprintln!("Error from supervisor {}: {}", participant.name, e);
                            }
                        }
                    }
                }
            }
        }

        Ok(responses)
    }

    /// Custom order message flow based on participant priority.
    async fn custom_order_message(
        &mut self,
        message: Message,
        optional_search_parameters: Option<openai_rust::chat::SearchParameters>,
    ) -> Result<Vec<ParticipantResponse>, Box<dyn std::error::Error>> {
        // Similar to round-robin but uses participant_order which is already sorted by priority
        self.round_robin_message(message, optional_search_parameters)
            .await
    }

    /// Gets the shared conversation history.
    pub fn shared_history(&self) -> &Vec<Message> {
        &self.shared_history
    }

    /// Gets aggregated token usage across all participants.
    pub fn total_token_usage(&self) -> TokenUsage {
        let total_input = 0;
        let total_output = 0;

        // This is a simplified version - in practice, you might want to track this more carefully
        TokenUsage {
            input_tokens: total_input,
            output_tokens: total_output,
            total_tokens: total_input + total_output,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use tokio::sync::Mutex;

    struct MockClient {
        name: String,
        response: String,
        usage: Mutex<Option<TokenUsage>>,
    }

    impl MockClient {
        fn new(name: &str, response: &str) -> Self {
            Self {
                name: name.to_string(),
                response: response.to_string(),
                usage: Mutex::new(Some(TokenUsage {
                    input_tokens: 10,
                    output_tokens: 5,
                    total_tokens: 15,
                })),
            }
        }
    }

    #[async_trait]
    impl ClientWrapper for MockClient {
        async fn send_message(
            &self,
            _messages: &[Message],
            _optional_search_parameters: Option<openai_rust::chat::SearchParameters>,
        ) -> Result<Message, Box<dyn std::error::Error>> {
            Ok(Message {
                role: Role::Assistant,
                content: Arc::from(self.response.as_str()),
            })
        }

        fn model_name(&self) -> &str {
            &self.name
        }

        fn usage_slot(&self) -> Option<&Mutex<Option<TokenUsage>>> {
            Some(&self.usage)
        }
    }

    #[tokio::test]
    async fn test_add_participant() {
        let client = Arc::new(MockClient::new("test-model", "response"));
        let mut session = MultiParticipantSession::new(
            "System prompt".to_string(),
            1000,
            OrchestrationStrategy::Broadcast,
        );

        session.add_participant("Alice", client, ParticipantRole::Panelist);
        assert_eq!(session.list_participants().len(), 1);
        assert!(session.get_participant("Alice").is_some());
    }

    #[tokio::test]
    async fn test_remove_participant() {
        let client = Arc::new(MockClient::new("test-model", "response"));
        let mut session = MultiParticipantSession::new(
            "System prompt".to_string(),
            1000,
            OrchestrationStrategy::Broadcast,
        );

        session.add_participant("Alice", client, ParticipantRole::Panelist);
        assert_eq!(session.list_participants().len(), 1);

        session.remove_participant("Alice");
        assert_eq!(session.list_participants().len(), 0);
    }

    #[tokio::test]
    async fn test_broadcast_strategy() {
        let client1 = Arc::new(MockClient::new("model1", "Response from client 1"));
        let client2 = Arc::new(MockClient::new("model2", "Response from client 2"));

        let mut session = MultiParticipantSession::new(
            "System prompt".to_string(),
            1000,
            OrchestrationStrategy::Broadcast,
        );

        session.add_participant("Client1", client1, ParticipantRole::Panelist);
        session.add_participant("Client2", client2, ParticipantRole::Panelist);

        let responses = session
            .send_message(Role::User, "Test message".to_string(), None)
            .await
            .unwrap();

        assert_eq!(responses.len(), 2);
        assert!(responses.iter().any(|r| r.participant_name == "Client1"));
        assert!(responses.iter().any(|r| r.participant_name == "Client2"));
    }

    #[tokio::test]
    async fn test_moderator_led_strategy() {
        let moderator = Arc::new(MockClient::new("moderator", "Moderator response"));
        let panelist = Arc::new(MockClient::new("panelist", "Panelist response"));

        let mut session = MultiParticipantSession::new(
            "System prompt".to_string(),
            1000,
            OrchestrationStrategy::ModeratorLed,
        );

        session.add_participant("Moderator", moderator, ParticipantRole::Moderator);
        session.add_participant("Panelist", panelist, ParticipantRole::Panelist);

        let responses = session
            .send_message(Role::User, "Test message".to_string(), None)
            .await
            .unwrap();

        assert_eq!(responses.len(), 2);
        // Moderator should respond first
        assert_eq!(responses[0].participant_name, "Moderator");
        assert_eq!(responses[0].participant_role, ParticipantRole::Moderator);
    }

    #[tokio::test]
    async fn test_hierarchical_strategy() {
        let worker1 = Arc::new(MockClient::new("worker1", "Worker 1 response"));
        let worker2 = Arc::new(MockClient::new("worker2", "Worker 2 response"));
        let supervisor = Arc::new(MockClient::new("supervisor", "Supervisor synthesis"));

        let mut session = MultiParticipantSession::new(
            "System prompt".to_string(),
            1000,
            OrchestrationStrategy::Hierarchical,
        );

        session.add_participant("Worker1", worker1, ParticipantRole::Worker);
        session.add_participant("Worker2", worker2, ParticipantRole::Worker);
        session.add_participant("Supervisor", supervisor, ParticipantRole::Supervisor);

        let responses = session
            .send_message(Role::User, "Test message".to_string(), None)
            .await
            .unwrap();

        // Should have 2 worker responses + 1 supervisor response
        assert_eq!(responses.len(), 3);

        let worker_count = responses
            .iter()
            .filter(|r| r.participant_role == ParticipantRole::Worker)
            .count();
        let supervisor_count = responses
            .iter()
            .filter(|r| r.participant_role == ParticipantRole::Supervisor)
            .count();

        assert_eq!(worker_count, 2);
        assert_eq!(supervisor_count, 1);
    }

    #[tokio::test]
    async fn test_priority_ordering() {
        let client1 = Arc::new(MockClient::new("low", "Low priority"));
        let client2 = Arc::new(MockClient::new("high", "High priority"));

        let mut session = MultiParticipantSession::new(
            "System prompt".to_string(),
            1000,
            OrchestrationStrategy::Custom,
        );

        session.add_participant_with_priority("Low", client1, ParticipantRole::Panelist, 1);
        session.add_participant_with_priority("High", client2, ParticipantRole::Panelist, 10);

        let order = session.list_participants();
        assert_eq!(order[0], "High"); // Higher priority comes first
        assert_eq!(order[1], "Low");
    }
}
