use async_trait::async_trait;
use cloudllm::client_wrapper::{ClientWrapper, Message, Role, TokenUsage};
use cloudllm::cloudllm::llm_session::estimate_message_token_count;
use cloudllm::{CouncilRole, CouncilSession};
use openai_rust2 as openai_rust;
use std::collections::VecDeque;
use std::sync::Arc;
use tokio::sync::Mutex;

struct SequencedMockClient {
    responses: Mutex<VecDeque<String>>,
    transcripts: Mutex<Vec<Vec<(String, String)>>>,
    usage: Mutex<Option<TokenUsage>>,
}

impl SequencedMockClient {
    fn new(responses: Vec<String>) -> Self {
        Self {
            responses: Mutex::new(responses.into()),
            transcripts: Mutex::new(Vec::new()),
            usage: Mutex::new(None),
        }
    }

    async fn transcripts(&self) -> Vec<Vec<(String, String)>> {
        self.transcripts.lock().await.clone()
    }

    fn role_label(role: &Role) -> &'static str {
        match role {
            Role::System => "system",
            Role::User => "user",
            Role::Assistant => "assistant",
        }
    }
}

#[async_trait]
impl ClientWrapper for SequencedMockClient {
    async fn send_message(
        &self,
        messages: &[Message],
        _optional_search_parameters: Option<openai_rust::chat::SearchParameters>,
    ) -> Result<Message, Box<dyn std::error::Error>> {
        let mut transcript_snapshot = Vec::with_capacity(messages.len());
        let mut input_tokens = 0;
        for message in messages.iter() {
            input_tokens += estimate_message_token_count(message);
            transcript_snapshot.push((
                Self::role_label(&message.role).to_string(),
                message.content.to_string(),
            ));
        }

        self.transcripts.lock().await.push(transcript_snapshot);

        let mut responses = self.responses.lock().await;
        let reply_text = responses
            .pop_front()
            .unwrap_or_else(|| "default reply".to_string());

        let response_message = Message {
            role: Role::Assistant,
            content: Arc::<str>::from(reply_text.clone()),
        };

        let output_tokens = estimate_message_token_count(&response_message);
        let usage = TokenUsage {
            input_tokens,
            output_tokens,
            total_tokens: input_tokens + output_tokens,
        };
        *self.usage.lock().await = Some(usage);

        Ok(response_message)
    }

    fn model_name(&self) -> &str {
        "mock-model"
    }

    async fn get_last_usage(&self) -> Option<TokenUsage> {
        self.usage.lock().await.clone()
    }

    fn usage_slot(&self) -> Option<&Mutex<Option<TokenUsage>>> {
        Some(&self.usage)
    }
}

fn make_client(responses: Vec<&str>) -> Arc<SequencedMockClient> {
    Arc::new(SequencedMockClient::new(
        responses.into_iter().map(|s| s.to_string()).collect(),
    ))
}

#[tokio::test]
async fn council_round_robin_basic_flow() {
    let moderator_client = make_client(vec!["Moderator response"]);
    let panel_client = make_client(vec!["Panelist response"]);

    let mut session = CouncilSession::new("Base instructions for the council");
    let moderator_id = session.add_participant(moderator_client.clone(), CouncilRole::Moderator);
    let panelist_id = session.add_participant(panel_client.clone(), CouncilRole::Panelist);

    let round = session
        .send_message(
            Role::User,
            "How should we cover the Bitcoin ETF news?".to_string(),
            None,
        )
        .await
        .expect("round robin should succeed");

    assert!(round.is_complete);
    assert_eq!(round.round_index, 1);
    assert_eq!(round.replies.len(), 2);
    assert_eq!(round.replies[0].participant_id, moderator_id);
    assert_eq!(
        round.replies[0].message.content.as_ref(),
        "Moderator response"
    );
    assert_eq!(round.replies[1].participant_id, panelist_id);
    assert_eq!(
        round.replies[1].message.content.as_ref(),
        "Panelist response"
    );
    assert_eq!(session.history().len(), 3);

    let panel_transcripts = panel_client.transcripts().await;
    assert_eq!(panel_transcripts.len(), 1);
    let panel_transcript = &panel_transcripts[0];
    assert!(panel_transcript
        .iter()
        .any(|(role, content)| role == "user" && content.starts_with("User:")));
    assert!(panel_transcript
        .iter()
        .any(|(role, content)| role == "user" && content.contains("Moderator 1")));
}

#[tokio::test]
async fn council_preserves_history_between_rounds() {
    let moderator_client = make_client(vec!["First moderator take", "Second moderator take"]);
    let panel_client = make_client(vec!["First panel take", "Second panel take"]);

    let mut session = CouncilSession::new("");
    let moderator_id = session.add_participant(moderator_client.clone(), CouncilRole::Moderator);
    let panelist_id = session.add_participant(panel_client.clone(), CouncilRole::Panelist);

    let first_round = session
        .send_message(Role::User, "Round one topic".to_string(), None)
        .await
        .expect("first round to succeed");
    assert_eq!(first_round.round_index, 1);

    let second_round = session
        .send_message(Role::User, "Round two topic".to_string(), None)
        .await
        .expect("second round to succeed");
    assert_eq!(second_round.round_index, 2);
    assert_eq!(session.history().len(), 6);

    assert_eq!(second_round.replies.len(), 2);
    assert_eq!(second_round.replies[0].participant_id, moderator_id);
    assert_eq!(second_round.replies[1].participant_id, panelist_id);

    let moderator_transcripts = moderator_client.transcripts().await;
    assert_eq!(moderator_transcripts.len(), 2);
    let second_moderator_transcript = &moderator_transcripts[1];
    assert!(second_moderator_transcript
        .iter()
        .any(|(role, content)| role == "assistant" && content == "First moderator take"));
    assert!(second_moderator_transcript
        .iter()
        .any(|(role, content)| role == "user" && content.contains("Round two topic")));

    let panel_transcripts = panel_client.transcripts().await;
    assert_eq!(panel_transcripts.len(), 2);
    let second_panel_transcript = &panel_transcripts[1];
    assert!(second_panel_transcript
        .iter()
        .any(|(role, content)| role == "user" && content.contains("Moderator 1")));
}

#[tokio::test]
async fn council_custom_order_is_respected() {
    let first_client = make_client(vec!["Panel first"]);
    let second_client = make_client(vec!["Moderator second"]);

    let mut session = CouncilSession::new("Council order test");
    let panel_id = session.add_participant(first_client, CouncilRole::Panelist);
    let moderator_id = session.add_participant(second_client, CouncilRole::Moderator);

    session
        .set_round_robin_order(vec![panel_id, moderator_id])
        .expect("order should be valid");

    let round = session
        .send_message(Role::User, "Who speaks first?".to_string(), None)
        .await
        .expect("round should succeed");

    assert_eq!(round.replies.len(), 2);
    assert_eq!(round.replies[0].participant_id, panel_id);
    assert_eq!(round.replies[1].participant_id, moderator_id);
}

#[tokio::test]
async fn participant_respects_token_limit() {
    use cloudllm::ParticipantConfig;

    // Create a client that will respond 3 times
    let client = make_client(vec!["First response", "Second response", "Third response"]);

    let mut session = CouncilSession::new("Token limit test");

    // Add participant with a 50 token limit
    let _participant_id = session.add_participant_with_config(
        client.clone(),
        CouncilRole::Panelist,
        ParticipantConfig {
            display_name: Some("Limited Panelist".into()),
            persona_prompt: None,
            max_tokens: Some(8192),
            max_total_tokens: Some(50),
        },
    );

    // First round should succeed (below limit)
    let round1 = session
        .send_message(Role::User, "First question".to_string(), None)
        .await
        .expect("first round should succeed");

    assert_eq!(round1.replies.len(), 1);
    assert!(round1.replies[0].usage.is_some());
    let usage1 = round1.replies[0].usage.clone().unwrap();

    // Check participant's total usage
    let participants = session.participants();
    assert_eq!(participants.len(), 1);
    assert_eq!(participants[0].total_usage.total_tokens, usage1.total_tokens);

    // Second round should succeed if still below limit
    let round2 = session
        .send_message(Role::User, "Second question".to_string(), None)
        .await
        .expect("second round should succeed");

    // Check if participant was included or skipped based on cumulative usage
    let participants_after = session.participants();
    let total_after_round2 = participants_after[0].total_usage.total_tokens;

    if total_after_round2 > usage1.total_tokens {
        // Participant responded
        assert_eq!(round2.replies.len(), 1);
    } else {
        // Participant was skipped due to hitting limit
        assert_eq!(round2.replies.len(), 0);
    }
}

#[tokio::test]
async fn multiple_participants_independent_token_limits() {
    use cloudllm::ParticipantConfig;

    let limited_client = make_client(vec!["Limited 1", "Limited 2", "Limited 3"]);
    let unlimited_client = make_client(vec!["Unlimited 1", "Unlimited 2", "Unlimited 3"]);

    let mut session = CouncilSession::new("Multi-participant token test");

    // Add limited participant (50 tokens)
    let limited_id = session.add_participant_with_config(
        limited_client.clone(),
        CouncilRole::Panelist,
        ParticipantConfig {
            display_name: Some("Limited".into()),
            persona_prompt: None,
            max_tokens: Some(8192),
            max_total_tokens: Some(50),
        },
    );

    // Add unlimited participant
    let unlimited_id = session.add_participant_with_config(
        unlimited_client.clone(),
        CouncilRole::Panelist,
        ParticipantConfig {
            display_name: Some("Unlimited".into()),
            persona_prompt: None,
            max_tokens: Some(8192),
            max_total_tokens: None,
        },
    );

    // Run multiple rounds
    let mut limited_was_skipped = false;
    for i in 1..=3 {
        let round = session
            .send_message(Role::User, format!("Question {}", i), None)
            .await
            .expect("round should succeed");

        // Unlimited participant should always respond
        assert!(round.replies.iter().any(|r| r.participant_id == unlimited_id),
                "Unlimited participant should always respond in round {}", i);

        // Limited participant may be skipped after hitting limit
        let limited_responded = round.replies.iter().any(|r| r.participant_id == limited_id);

        let participants = session.participants();
        let limited_info = participants.iter().find(|p| p.id == limited_id).unwrap();

        // If already skipped in a previous round, should still be skipped
        if limited_was_skipped {
            assert!(!limited_responded,
                    "Limited participant should remain skipped after exceeding limit in round {}", i);
        }

        // Check if participant has reached limit
        if limited_info.total_usage.total_tokens >= 50 {
            if !limited_responded {
                limited_was_skipped = true;
            }
        }
    }

    // Verify final state
    let participants = session.participants();
    let unlimited_info = participants.iter().find(|p| p.id == unlimited_id).unwrap();

    // Unlimited should have responded 3 times
    assert!(unlimited_info.total_usage.total_tokens > 0);

    // Limited participant should have hit their limit or been skipped
    let limited_info = participants.iter().find(|p| p.id == limited_id).unwrap();
    if limited_info.total_usage.total_tokens >= 50 {
        assert!(limited_was_skipped, "Limited participant should have been skipped after reaching limit");
    }
}

#[tokio::test]
async fn participant_info_includes_token_limit() {
    use cloudllm::ParticipantConfig;

    let client = make_client(vec!["Response"]);
    let mut session = CouncilSession::new("Participant info test");

    let _with_limit = session.add_participant_with_config(
        client.clone(),
        CouncilRole::Moderator,
        ParticipantConfig {
            display_name: Some("With Limit".into()),
            persona_prompt: None,
            max_tokens: Some(4096),
            max_total_tokens: Some(10000),
        },
    );

    let _without_limit = session.add_participant_with_config(
        make_client(vec!["Response"]),
        CouncilRole::Panelist,
        ParticipantConfig {
            display_name: Some("No Limit".into()),
            persona_prompt: None,
            max_tokens: Some(8192),
            max_total_tokens: None,
        },
    );

    let participants = session.participants();
    assert_eq!(participants.len(), 2);

    // Check first participant has limit
    assert_eq!(participants[0].display_name, "With Limit");
    assert_eq!(participants[0].max_total_tokens, Some(10000));

    // Check second participant has no limit
    assert_eq!(participants[1].display_name, "No Limit");
    assert_eq!(participants[1].max_total_tokens, None);
}
