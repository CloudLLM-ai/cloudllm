// Integration tests for multi-participant sessions
use async_trait::async_trait;
use cloudllm::client_wrapper::{ClientWrapper, Message, Role, TokenUsage};
use cloudllm::multi_participant_session::{
    MultiParticipantSession, OrchestrationStrategy, ParticipantRole,
};
use openai_rust2 as openai_rust;
use std::sync::Arc;
use tokio::sync::Mutex;

// Mock client for testing
struct TestClient {
    name: String,
    response_prefix: String,
    usage: Mutex<Option<TokenUsage>>,
}

impl TestClient {
    fn new(name: &str, response_prefix: &str) -> Self {
        Self {
            name: name.to_string(),
            response_prefix: response_prefix.to_string(),
            usage: Mutex::new(Some(TokenUsage {
                input_tokens: 10,
                output_tokens: 5,
                total_tokens: 15,
            })),
        }
    }
}

#[async_trait]
impl ClientWrapper for TestClient {
    async fn send_message(
        &self,
        messages: &[Message],
        _optional_search_parameters: Option<openai_rust::chat::SearchParameters>,
    ) -> Result<Message, Box<dyn std::error::Error>> {
        let response_text = format!(
            "{} response to {} messages",
            self.response_prefix,
            messages.len()
        );
        Ok(Message {
            role: Role::Assistant,
            content: Arc::from(response_text.as_str()),
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
async fn test_broadcast_all_participants_respond() {
    let client1 = Arc::new(TestClient::new("client1", "Client1"));
    let client2 = Arc::new(TestClient::new("client2", "Client2"));
    let client3 = Arc::new(TestClient::new("client3", "Client3"));

    let mut session = MultiParticipantSession::new(
        "Test system prompt".to_string(),
        1000,
        OrchestrationStrategy::Broadcast,
    );

    session.add_participant("Alice", client1, ParticipantRole::Panelist);
    session.add_participant("Bob", client2, ParticipantRole::Panelist);
    session.add_participant("Carol", client3, ParticipantRole::Panelist);

    let responses = session
        .send_message(Role::User, "Test message".to_string(), None)
        .await
        .unwrap();

    assert_eq!(responses.len(), 3, "Should have 3 responses");
    assert!(responses.iter().any(|r| r.participant_name == "Alice"));
    assert!(responses.iter().any(|r| r.participant_name == "Bob"));
    assert!(responses.iter().any(|r| r.participant_name == "Carol"));

    // All should have token usage
    for response in &responses {
        assert!(response.token_usage.is_some());
    }
}

#[tokio::test]
async fn test_round_robin_sequential_context() {
    let client1 = Arc::new(TestClient::new("client1", "First"));
    let client2 = Arc::new(TestClient::new("client2", "Second"));

    let mut session = MultiParticipantSession::new(
        "System".to_string(),
        1000,
        OrchestrationStrategy::RoundRobin,
    );

    session.add_participant("First", client1, ParticipantRole::Panelist);
    session.add_participant("Second", client2, ParticipantRole::Panelist);

    let responses = session
        .send_message(Role::User, "Question".to_string(), None)
        .await
        .unwrap();

    assert_eq!(responses.len(), 2);

    // First participant should see: system + user message = 2 messages
    // Second participant should see: system + accumulated context from first = 2 messages
    // Both should have responses
    assert_eq!(responses[0].participant_name, "First");
    assert_eq!(responses[1].participant_name, "Second");
}

#[tokio::test]
async fn test_moderator_led_flow() {
    let moderator = Arc::new(TestClient::new("mod", "Moderator"));
    let panelist1 = Arc::new(TestClient::new("pan1", "Panelist1"));
    let panelist2 = Arc::new(TestClient::new("pan2", "Panelist2"));

    let mut session = MultiParticipantSession::new(
        "System".to_string(),
        1000,
        OrchestrationStrategy::ModeratorLed,
    );

    session.add_participant("Moderator", moderator, ParticipantRole::Moderator);
    session.add_participant("Panelist1", panelist1, ParticipantRole::Panelist);
    session.add_participant("Panelist2", panelist2, ParticipantRole::Panelist);

    let responses = session
        .send_message(Role::User, "Question".to_string(), None)
        .await
        .unwrap();

    assert_eq!(responses.len(), 3);

    // First response should be from moderator
    assert_eq!(responses[0].participant_name, "Moderator");
    assert_eq!(responses[0].participant_role, ParticipantRole::Moderator);

    // Other responses should be from panelists
    let panelist_responses: Vec<_> = responses
        .iter()
        .filter(|r| r.participant_role == ParticipantRole::Panelist)
        .collect();
    assert_eq!(panelist_responses.len(), 2);
}

#[tokio::test]
async fn test_hierarchical_worker_supervisor_flow() {
    let worker1 = Arc::new(TestClient::new("w1", "Worker1"));
    let worker2 = Arc::new(TestClient::new("w2", "Worker2"));
    let supervisor = Arc::new(TestClient::new("sup", "Supervisor"));

    let mut session = MultiParticipantSession::new(
        "System".to_string(),
        1000,
        OrchestrationStrategy::Hierarchical,
    );

    session.add_participant("Worker1", worker1, ParticipantRole::Worker);
    session.add_participant("Worker2", worker2, ParticipantRole::Worker);
    session.add_participant("Supervisor", supervisor, ParticipantRole::Supervisor);

    let responses = session
        .send_message(Role::User, "Task".to_string(), None)
        .await
        .unwrap();

    // Should have 2 worker responses + 1 supervisor response
    assert_eq!(responses.len(), 3);

    let worker_responses: Vec<_> = responses
        .iter()
        .filter(|r| r.participant_role == ParticipantRole::Worker)
        .collect();
    let supervisor_responses: Vec<_> = responses
        .iter()
        .filter(|r| r.participant_role == ParticipantRole::Supervisor)
        .collect();

    assert_eq!(worker_responses.len(), 2);
    assert_eq!(supervisor_responses.len(), 1);
}

#[tokio::test]
async fn test_custom_priority_ordering() {
    let low = Arc::new(TestClient::new("low", "Low"));
    let high = Arc::new(TestClient::new("high", "High"));
    let medium = Arc::new(TestClient::new("med", "Medium"));

    let mut session =
        MultiParticipantSession::new("System".to_string(), 1000, OrchestrationStrategy::Custom);

    // Add in random order with different priorities
    session.add_participant_with_priority("Low", low, ParticipantRole::Panelist, 1);
    session.add_participant_with_priority("High", high, ParticipantRole::Panelist, 10);
    session.add_participant_with_priority("Medium", medium, ParticipantRole::Panelist, 5);

    let order = session.list_participants();
    assert_eq!(order[0], "High", "Highest priority should be first");
    assert_eq!(order[1], "Medium", "Medium priority should be second");
    assert_eq!(order[2], "Low", "Lowest priority should be last");

    let responses = session
        .send_message(Role::User, "Question".to_string(), None)
        .await
        .unwrap();

    // Responses should be in priority order
    assert_eq!(responses[0].participant_name, "High");
    assert_eq!(responses[1].participant_name, "Medium");
    assert_eq!(responses[2].participant_name, "Low");
}

#[tokio::test]
async fn test_participant_management() {
    let client1 = Arc::new(TestClient::new("c1", "Client1"));
    let client2 = Arc::new(TestClient::new("c2", "Client2"));

    let mut session =
        MultiParticipantSession::new("System".to_string(), 1000, OrchestrationStrategy::Broadcast);

    // Add participants
    session.add_participant("P1", client1.clone(), ParticipantRole::Panelist);
    session.add_participant("P2", client2.clone(), ParticipantRole::Panelist);

    assert_eq!(session.list_participants().len(), 2);
    assert!(session.get_participant("P1").is_some());
    assert!(session.get_participant("P2").is_some());

    // Remove one
    let removed = session.remove_participant("P1");
    assert!(removed.is_some());
    assert_eq!(session.list_participants().len(), 1);
    assert!(session.get_participant("P1").is_none());

    // Send message to remaining participant
    let responses = session
        .send_message(Role::User, "Test".to_string(), None)
        .await
        .unwrap();

    assert_eq!(responses.len(), 1);
    assert_eq!(responses[0].participant_name, "P2");
}

#[tokio::test]
async fn test_strategy_switching() {
    let client1 = Arc::new(TestClient::new("c1", "Client1"));
    let client2 = Arc::new(TestClient::new("c2", "Client2"));

    let mut session =
        MultiParticipantSession::new("System".to_string(), 1000, OrchestrationStrategy::Broadcast);

    session.add_participant("P1", client1, ParticipantRole::Panelist);
    session.add_participant("P2", client2, ParticipantRole::Panelist);

    assert_eq!(
        *session.orchestration_strategy(),
        OrchestrationStrategy::Broadcast
    );

    // Switch strategy
    session.set_orchestration_strategy(OrchestrationStrategy::RoundRobin);
    assert_eq!(
        *session.orchestration_strategy(),
        OrchestrationStrategy::RoundRobin
    );

    // Should still work with new strategy
    let responses = session
        .send_message(Role::User, "Test".to_string(), None)
        .await
        .unwrap();

    assert_eq!(responses.len(), 2);
}

#[tokio::test]
async fn test_multiple_rounds_conversation() {
    let client1 = Arc::new(TestClient::new("c1", "Expert1"));
    let client2 = Arc::new(TestClient::new("c2", "Expert2"));

    let mut session = MultiParticipantSession::new(
        "You are experts discussing a topic.".to_string(),
        1000,
        OrchestrationStrategy::Broadcast,
    );

    session.add_participant("Expert1", client1, ParticipantRole::Panelist);
    session.add_participant("Expert2", client2, ParticipantRole::Panelist);

    // First round
    let responses1 = session
        .send_message(Role::User, "Question 1".to_string(), None)
        .await
        .unwrap();
    assert_eq!(responses1.len(), 2);

    // Second round
    let responses2 = session
        .send_message(Role::User, "Question 2".to_string(), None)
        .await
        .unwrap();
    assert_eq!(responses2.len(), 2);

    // Third round
    let responses3 = session
        .send_message(Role::User, "Question 3".to_string(), None)
        .await
        .unwrap();
    assert_eq!(responses3.len(), 2);

    // Participants should have accumulated conversation history
    // Each participant should have: Q1, R1, Q2, R2, Q3, R3 = 6 messages
    let participant = session.get_participant("Expert1").unwrap();
    assert!(participant.conversation_history.len() >= 6);
}

#[tokio::test]
async fn test_observer_role_in_hierarchy() {
    let observer = Arc::new(TestClient::new("obs", "Observer"));
    let worker = Arc::new(TestClient::new("work", "Worker"));
    let supervisor = Arc::new(TestClient::new("sup", "Supervisor"));

    let mut session = MultiParticipantSession::new(
        "System".to_string(),
        1000,
        OrchestrationStrategy::Hierarchical,
    );

    session.add_participant("Observer", observer, ParticipantRole::Observer);
    session.add_participant("Worker", worker, ParticipantRole::Worker);
    session.add_participant("Supervisor", supervisor, ParticipantRole::Supervisor);

    let responses = session
        .send_message(Role::User, "Task".to_string(), None)
        .await
        .unwrap();

    // Observer shouldn't respond in hierarchical (only workers and supervisors do)
    // Should have 1 worker + 1 supervisor = 2 responses
    assert_eq!(responses.len(), 2);
    assert!(!responses.iter().any(|r| r.participant_name == "Observer"));
}
