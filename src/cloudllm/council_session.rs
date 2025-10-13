use crate::client_wrapper::{ClientWrapper, Message, Role, TokenUsage};
use crate::cloudllm::llm_session::estimate_message_token_count;
use openai_rust2 as openai_rust;
use std::collections::HashSet;
use std::fmt;
use std::sync::Arc;

const DEFAULT_MAX_TOKENS: usize = 16_384;

#[derive(Clone, Debug, PartialEq, Eq, Hash, Copy)]
pub struct ParticipantId(pub usize);

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CouncilRole {
    Moderator,
    Panelist,
    Observer,
    Custom(String),
}

impl CouncilRole {
    fn label(&self) -> String {
        match self {
            CouncilRole::Moderator => "Moderator".to_string(),
            CouncilRole::Panelist => "Panelist".to_string(),
            CouncilRole::Observer => "Observer".to_string(),
            CouncilRole::Custom(name) => name.clone(),
        }
    }
}

#[derive(Debug)]
pub struct CouncilError {
    details: String,
}

impl CouncilError {
    fn new(details: impl Into<String>) -> Self {
        Self {
            details: details.into(),
        }
    }
}

impl fmt::Display for CouncilError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.details)
    }
}

impl std::error::Error for CouncilError {}

#[derive(Clone, Debug)]
pub struct ParticipantConfig {
    pub display_name: Option<String>,
    pub persona_prompt: Option<String>,
    pub max_tokens: Option<usize>,
}

impl Default for ParticipantConfig {
    fn default() -> Self {
        Self {
            display_name: None,
            persona_prompt: None,
            max_tokens: None,
        }
    }
}

#[derive(Clone, Debug)]
pub enum CouncilSpeaker {
    User,
    Participant(ParticipantId),
}

#[derive(Clone, Debug)]
pub struct CouncilDialogueTurn {
    pub speaker: CouncilSpeaker,
    pub content: Arc<str>,
}

#[derive(Clone)]
pub struct ParticipantReply {
    pub participant_id: ParticipantId,
    pub role: CouncilRole,
    pub name: String,
    pub message: Message,
    pub usage: Option<TokenUsage>,
}

#[derive(Clone)]
pub struct CouncilRoundResponse {
    pub user_message: Arc<str>,
    pub replies: Vec<ParticipantReply>,
    pub pending_participants: Vec<ParticipantId>,
    pub round_index: usize,
    pub is_complete: bool,
}

impl CouncilRoundResponse {
    pub fn is_waiting(&self) -> bool {
        !self.pending_participants.is_empty()
    }
}

#[derive(Clone, Debug)]
pub struct CouncilParticipantInfo {
    pub id: ParticipantId,
    pub role: CouncilRole,
    pub display_name: String,
    pub model_name: String,
    pub max_tokens: usize,
    pub total_usage: TokenUsage,
}

struct Participant {
    id: ParticipantId,
    role: CouncilRole,
    display_name: String,
    model_name: String,
    label: String,
    system_prompt: Message,
    client: Arc<dyn ClientWrapper>,
    max_tokens: usize,
    total_usage: TokenUsage,
}

impl Participant {
    fn info(&self) -> CouncilParticipantInfo {
        CouncilParticipantInfo {
            id: self.id,
            role: self.role.clone(),
            display_name: self.display_name.clone(),
            model_name: self.model_name.clone(),
            max_tokens: self.max_tokens,
            total_usage: self.total_usage.clone(),
        }
    }
}

pub struct CouncilSession {
    base_system_prompt: Arc<str>,
    participants: Vec<Participant>,
    history: Vec<CouncilDialogueTurn>,
    speaking_order: Option<Vec<ParticipantId>>,
    default_max_tokens: usize,
    rounds_completed: usize,
    total_usage: TokenUsage,
}

impl CouncilSession {
    pub fn new(system_prompt: impl Into<String>) -> Self {
        let prompt_input = system_prompt.into();
        let prompt = if prompt_input.trim().is_empty() {
            Arc::<str>::from(
                "You are part of a multi-LLM council. You will receive transcripts labelled with the speaker's name. Each response should be thoughtful, actionable, and aware of the prior discussion.",
            )
        } else {
            Arc::<str>::from(prompt_input)
        };

        Self {
            base_system_prompt: prompt,
            participants: Vec::new(),
            history: Vec::new(),
            speaking_order: None,
            default_max_tokens: DEFAULT_MAX_TOKENS,
            rounds_completed: 0,
            total_usage: TokenUsage {
                input_tokens: 0,
                output_tokens: 0,
                total_tokens: 0,
            },
        }
    }

    pub fn set_default_participant_max_tokens(&mut self, tokens: usize) {
        self.default_max_tokens = tokens.max(1);
    }

    pub fn add_participant(
        &mut self,
        client: Arc<dyn ClientWrapper>,
        role: CouncilRole,
    ) -> ParticipantId {
        self.add_participant_with_config(client, role, ParticipantConfig::default())
    }

    pub fn add_participant_with_config(
        &mut self,
        client: Arc<dyn ClientWrapper>,
        role: CouncilRole,
        config: ParticipantConfig,
    ) -> ParticipantId {
        let id = ParticipantId(self.participants.len());
        let role_label = role.label();
        let model_name = client.model_name().to_string();
        let ordinal = self.participants.len() + 1;

        let display_name = config
            .display_name
            .unwrap_or_else(|| format!("{} {}", role_label, ordinal));

        let persona = config.persona_prompt.unwrap_or_default();

        let mut system_prompt = String::with_capacity(
            self.base_system_prompt.len()
                + role_label.len()
                + display_name.len()
                + persona.len()
                + 160,
        );
        system_prompt.push_str(&self.base_system_prompt);
        system_prompt.push_str("\n\nYou are ");
        system_prompt.push_str(&role_label.to_lowercase());
        system_prompt.push_str(" identified as ");
        system_prompt.push_str(&display_name);
        if !persona.trim().is_empty() {
            system_prompt.push_str(". ");
            system_prompt.push_str(&persona);
        }
        system_prompt.push_str(
            "\n\nRespond only when prompted. Reference previous remarks when helpful and keep the conversation focused on solving the user's request.",
        );

        let system_prompt = Message {
            role: Role::System,
            content: Arc::<str>::from(system_prompt),
        };

        let participant = Participant {
            id,
            role,
            display_name: display_name.clone(),
            model_name: model_name.clone(),
            label: format!("{} [{}]", display_name, model_name),
            system_prompt,
            client,
            max_tokens: config.max_tokens.unwrap_or(self.default_max_tokens),
            total_usage: TokenUsage {
                input_tokens: 0,
                output_tokens: 0,
                total_tokens: 0,
            },
        };

        self.participants.push(participant);
        id
    }

    pub fn set_round_robin_order(
        &mut self,
        order: Vec<ParticipantId>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        if order.is_empty() {
            return Err(Box::new(CouncilError::new(
                "speaking order cannot be empty",
            )));
        }

        for participant_id in &order {
            if participant_id.0 >= self.participants.len() {
                return Err(Box::new(CouncilError::new(format!(
                    "invalid participant id {} in order",
                    participant_id.0
                ))));
            }
        }

        self.speaking_order = Some(order);
        Ok(())
    }

    pub fn participants(&self) -> Vec<CouncilParticipantInfo> {
        self.participants.iter().map(|p| p.info()).collect()
    }

    pub fn history(&self) -> &[CouncilDialogueTurn] {
        &self.history
    }

    pub fn total_usage(&self) -> TokenUsage {
        self.total_usage.clone()
    }

    pub async fn send_message(
        &mut self,
        role: Role,
        content: String,
        optional_search_parameters: Option<openai_rust::chat::SearchParameters>,
    ) -> Result<CouncilRoundResponse, Box<dyn std::error::Error>> {
        match role {
            Role::User => {
                if self.participants.is_empty() {
                    return Err(Box::new(CouncilError::new(
                        "no participants registered for council session",
                    )));
                }
                self.run_round(content, optional_search_parameters).await
            }
            _ => Err(Box::new(CouncilError::new(
                "council session send_message currently supports Role::User only",
            ))),
        }
    }

    async fn run_round(
        &mut self,
        content: String,
        optional_search_parameters: Option<openai_rust::chat::SearchParameters>,
    ) -> Result<CouncilRoundResponse, Box<dyn std::error::Error>> {
        let user_arc: Arc<str> = Arc::<str>::from(content);
        self.history.push(CouncilDialogueTurn {
            speaker: CouncilSpeaker::User,
            content: user_arc.clone(),
        });

        let speaking_order = self.current_order_indices();
        if speaking_order.is_empty() {
            return Err(Box::new(CouncilError::new(
                "speaking order resolved to empty sequence",
            )));
        }

        let mut replies = Vec::with_capacity(speaking_order.len());

        for participant_index in speaking_order {
            let messages = self.build_messages_for_participant(participant_index);
            let client = self.participants[participant_index].client.clone();
            let response = client
                .send_message(&messages, optional_search_parameters.clone())
                .await?;

            self.history.push(CouncilDialogueTurn {
                speaker: CouncilSpeaker::Participant(self.participants[participant_index].id),
                content: response.content.clone(),
            });

            let usage = client.get_last_usage().await;
            if let Some(usage_metrics) = usage.clone() {
                self.total_usage.input_tokens += usage_metrics.input_tokens;
                self.total_usage.output_tokens += usage_metrics.output_tokens;
                self.total_usage.total_tokens += usage_metrics.total_tokens;
                let totals = &mut self.participants[participant_index].total_usage;
                totals.input_tokens += usage_metrics.input_tokens;
                totals.output_tokens += usage_metrics.output_tokens;
                totals.total_tokens += usage_metrics.total_tokens;
            }

            replies.push(ParticipantReply {
                participant_id: self.participants[participant_index].id,
                role: self.participants[participant_index].role.clone(),
                name: self.participants[participant_index].display_name.clone(),
                message: response,
                usage,
            });
        }

        self.rounds_completed += 1;

        Ok(CouncilRoundResponse {
            user_message: user_arc,
            replies,
            pending_participants: Vec::new(),
            round_index: self.rounds_completed,
            is_complete: true,
        })
    }

    fn current_order_indices(&self) -> Vec<usize> {
        if let Some(explicit_order) = &self.speaking_order {
            let mut seen = HashSet::new();
            let mut indices = Vec::with_capacity(self.participants.len());
            for participant_id in explicit_order {
                if participant_id.0 < self.participants.len() && seen.insert(participant_id.0) {
                    indices.push(participant_id.0);
                }
            }
            for idx in 0..self.participants.len() {
                if seen.insert(idx) {
                    indices.push(idx);
                }
            }
            return indices;
        }

        let mut moderators = Vec::new();
        let mut panelists = Vec::new();
        let mut others = Vec::new();

        for (idx, participant) in self.participants.iter().enumerate() {
            match participant.role {
                CouncilRole::Moderator => moderators.push(idx),
                CouncilRole::Panelist => panelists.push(idx),
                _ => others.push(idx),
            }
        }

        moderators
            .into_iter()
            .chain(panelists)
            .chain(others)
            .collect()
    }

    fn build_messages_for_participant(&self, participant_index: usize) -> Vec<Message> {
        let participant = &self.participants[participant_index];
        let mut messages = Vec::with_capacity(self.history.len() + 1);
        messages.push(participant.system_prompt.clone());

        for turn in &self.history {
            let (role, content_string) = self.format_turn_for_participant(turn, participant_index);
            messages.push(Message {
                role,
                content: Arc::<str>::from(content_string),
            });
        }

        self.trim_to_max_tokens(messages, participant.max_tokens)
    }

    fn format_turn_for_participant(
        &self,
        turn: &CouncilDialogueTurn,
        participant_index: usize,
    ) -> (Role, String) {
        match turn.speaker {
            CouncilSpeaker::User => (Role::User, format!("User: {}", turn.content)),
            CouncilSpeaker::Participant(id) => {
                if id.0 == participant_index {
                    (Role::Assistant, turn.content.to_string())
                } else {
                    let label = &self.participants[id.0].label;
                    (Role::User, format!("{}: {}", label, turn.content))
                }
            }
        }
    }

    fn trim_to_max_tokens(&self, messages: Vec<Message>, max_tokens: usize) -> Vec<Message> {
        if messages.is_empty() {
            return messages;
        }

        if max_tokens == 0 {
            return messages;
        }

        let mut iter = messages.into_iter();
        let system_message = match iter.next() {
            Some(message) => message,
            None => return Vec::new(),
        };

        let mut trimmed: Vec<Message> = Vec::new();
        trimmed.push(system_message.clone());

        let mut accumulated = estimate_message_token_count(&system_message);
        let recent: Vec<Message> = iter.collect();

        for message in recent.into_iter().rev() {
            let tokens = estimate_message_token_count(&message);
            if accumulated + tokens <= max_tokens || trimmed.len() == 1 {
                accumulated += tokens;
                trimmed.push(message);
            }
        }

        if trimmed.len() > 1 {
            let system = trimmed.remove(0);
            trimmed.reverse();
            trimmed.insert(0, system);
        }

        trimmed
    }
}
