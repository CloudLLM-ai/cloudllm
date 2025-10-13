// src/cloudllm/mod.rs

pub mod client_wrapper;
pub mod clients;
pub mod council_session;
pub mod llm_session;

// Let's explicitly export LLMSession so we don't have to access it via cloudllm::llm_session::LLMSession
// and instead as cloudllm::LLMSession
pub use council_session::{
    CouncilDialogueTurn, CouncilError, CouncilParticipantInfo, CouncilRole, CouncilRoundResponse,
    CouncilSession, CouncilSpeaker, ParticipantConfig, ParticipantId, ParticipantReply,
};
pub use llm_session::LLMSession;
