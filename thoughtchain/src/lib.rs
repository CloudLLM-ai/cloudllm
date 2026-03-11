//! Semantic, hash-chained memory for long-running agents.
//!
//! `thoughtchain` provides an append-only, adapter-backed memory log for
//! durable, queryable cognitive state. Thoughts are timestamped, hash-chained,
//! typed, optionally connected to prior thoughts, and exportable as prompts or
//! Markdown memory snapshots. The current default backend is JSONL, but the
//! chain model is intentionally independent from any single storage format.
#![warn(missing_docs)]

#[cfg(feature = "server")]
pub mod server;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{HashMap, HashSet};
use std::fmt;
use std::fs::{self, OpenOptions};
use std::io::{self, BufRead, BufReader, Read, Write};
use std::path::{Path, PathBuf};
use std::str::FromStr;
use uuid::Uuid;

/// Persistence interface for ThoughtChain storage backends.
///
/// Storage adapters are responsible only for durable read and append
/// operations. The in-memory chain model, hashing, querying, and replay logic
/// remain inside [`ThoughtChain`].
///
/// # Example
///
/// ```
/// use std::path::PathBuf;
/// use thoughtchain::{JsonlStorageAdapter, StorageAdapter};
///
/// let adapter = JsonlStorageAdapter::for_chain_key(PathBuf::from("/tmp/tc_store"), "demo");
/// let location = adapter.storage_location();
///
/// assert!(location.ends_with(".jsonl"));
/// ```
pub trait StorageAdapter: Send + Sync {
    /// Load all persisted thoughts in order.
    fn load_thoughts(&self) -> io::Result<Vec<Thought>>;

    /// Persist a newly appended thought.
    fn append_thought(&self, thought: &Thought) -> io::Result<()>;

    /// Return a human-readable storage location or descriptor.
    fn storage_location(&self) -> String;
}

/// Supported durable storage formats for ThoughtChain.
///
/// This enum lets applications select a backend without hard-coding a concrete
/// adapter type. `Jsonl` remains the most human-inspectable option, while
/// `Binary` stores length-prefixed serialized thoughts for more compact and
/// efficient loading.
///
/// # Example
///
/// ```
/// use std::str::FromStr;
/// use thoughtchain::StorageAdapterKind;
///
/// let kind = StorageAdapterKind::from_str("binary").unwrap();
/// assert_eq!(kind.as_str(), "binary");
/// ```
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash, Default)]
pub enum StorageAdapterKind {
    /// Newline-delimited JSON storage.
    #[default]
    Jsonl,
    /// Length-prefixed binary serialization of `Thought` records.
    Binary,
}

impl StorageAdapterKind {
    /// Return the stable lowercase name of this adapter kind.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Jsonl => "jsonl",
            Self::Binary => "binary",
        }
    }

    /// Return the file extension used by this adapter kind.
    pub fn file_extension(self) -> &'static str {
        match self {
            Self::Jsonl => "jsonl",
            Self::Binary => "tcbin",
        }
    }

    /// Create a boxed storage adapter for a durable chain key.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use std::path::PathBuf;
    /// use thoughtchain::{StorageAdapter, StorageAdapterKind};
    ///
    /// let adapter = StorageAdapterKind::Binary
    ///     .for_chain_key(PathBuf::from("/tmp/tc_kind"), "demo");
    /// assert!(adapter.storage_location().ends_with(".tcbin"));
    /// ```
    pub fn for_chain_key<P: AsRef<Path>>(
        self,
        chain_dir: P,
        chain_key: &str,
    ) -> Box<dyn StorageAdapter> {
        match self {
            Self::Jsonl => Box::new(JsonlStorageAdapter::for_chain_key(chain_dir, chain_key)),
            Self::Binary => Box::new(BinaryStorageAdapter::for_chain_key(chain_dir, chain_key)),
        }
    }
}

impl fmt::Display for StorageAdapterKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for StorageAdapterKind {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.trim().to_ascii_lowercase().as_str() {
            "jsonl" => Ok(Self::Jsonl),
            "binary" => Ok(Self::Binary),
            other => Err(format!(
                "Unsupported ThoughtChain storage adapter '{other}'. Expected 'jsonl' or 'binary'"
            )),
        }
    }
}

/// Append-only JSONL storage adapter for ThoughtChain.
///
/// This is the default storage backend used by [`ThoughtChain::open`] and
/// [`ThoughtChain::open_with_key`].
///
/// # Example
///
/// ```rust,no_run
/// use std::path::PathBuf;
/// use thoughtchain::{JsonlStorageAdapter, StorageAdapter};
///
/// let adapter = JsonlStorageAdapter::for_chain_key(PathBuf::from("/tmp/tc_jsonl"), "agent-memory");
/// assert!(adapter.storage_location().ends_with(".jsonl"));
/// ```
#[derive(Debug, Clone)]
pub struct JsonlStorageAdapter {
    file_path: PathBuf,
}

impl JsonlStorageAdapter {
    /// Create a JSONL adapter for an explicit file path.
    pub fn new(file_path: PathBuf) -> Self {
        Self { file_path }
    }

    /// Create a JSONL adapter using the stable ThoughtChain filename for a chain key.
    pub fn for_chain_key<P: AsRef<Path>>(chain_dir: P, chain_key: &str) -> Self {
        let file_path = chain_dir
            .as_ref()
            .join(chain_filename(chain_key, "", None, None));
        Self::new(file_path)
    }

    /// Return the underlying JSONL path.
    pub fn file_path(&self) -> &PathBuf {
        &self.file_path
    }
}

impl StorageAdapter for JsonlStorageAdapter {
    fn load_thoughts(&self) -> io::Result<Vec<Thought>> {
        if !self.file_path.exists() {
            return Ok(Vec::new());
        }

        let file = fs::File::open(&self.file_path)?;
        let reader = BufReader::new(file);
        let mut entries = Vec::new();
        for line in reader.lines() {
            let line = line?;
            if line.trim().is_empty() {
                continue;
            }
            let thought: Thought = serde_json::from_str(&line).map_err(|e| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("Failed to parse thought: {e}"),
                )
            })?;
            entries.push(thought);
        }
        Ok(entries)
    }

    fn append_thought(&self, thought: &Thought) -> io::Result<()> {
        persist_jsonl_thought(&self.file_path, thought)
    }

    fn storage_location(&self) -> String {
        self.file_path.display().to_string()
    }
}

/// Append-only binary storage adapter for ThoughtChain.
///
/// Each record is stored as a length-prefixed serialized [`Thought`], which
/// keeps append operations simple while avoiding JSON parse overhead on reload.
///
/// # Example
///
/// ```rust,no_run
/// use std::path::PathBuf;
/// use thoughtchain::{BinaryStorageAdapter, StorageAdapter};
///
/// let adapter = BinaryStorageAdapter::for_chain_key(PathBuf::from("/tmp/tc_bin"), "agent-memory");
/// assert!(adapter.storage_location().ends_with(".tcbin"));
/// ```
#[derive(Debug, Clone)]
pub struct BinaryStorageAdapter {
    file_path: PathBuf,
}

impl BinaryStorageAdapter {
    /// Create a binary adapter for an explicit file path.
    pub fn new(file_path: PathBuf) -> Self {
        Self { file_path }
    }

    /// Create a binary adapter using the stable ThoughtChain filename for a chain key.
    pub fn for_chain_key<P: AsRef<Path>>(chain_dir: P, chain_key: &str) -> Self {
        let file_path = chain_dir.as_ref().join(chain_storage_filename(
            chain_key,
            StorageAdapterKind::Binary,
        ));
        Self::new(file_path)
    }

    /// Return the underlying binary path.
    pub fn file_path(&self) -> &PathBuf {
        &self.file_path
    }
}

impl StorageAdapter for BinaryStorageAdapter {
    fn load_thoughts(&self) -> io::Result<Vec<Thought>> {
        load_binary_thoughts(&self.file_path)
    }

    fn append_thought(&self, thought: &Thought) -> io::Result<()> {
        persist_binary_thought(&self.file_path, thought)
    }

    fn storage_location(&self) -> String {
        self.file_path.display().to_string()
    }
}

/// Semantic category describing what changed in the agent's internal model.
///
/// `ThoughtType` is intentionally semantic rather than operational. For example,
/// `Summary` describes the meaning of the thought, while
/// [`ThoughtRole::Compression`] captures why it was emitted.
///
/// # Example
///
/// ```
/// use thoughtchain::ThoughtType;
///
/// let thought_type = ThoughtType::Constraint;
/// let json = serde_json::to_string(&thought_type).unwrap();
///
/// assert_eq!(json, "\"Constraint\"");
/// ```
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum ThoughtType {
    /// A user's stated preference changed or became explicit.
    PreferenceUpdate,
    /// A durable characteristic of the user was learned.
    UserTrait,
    /// The agent's model of its relationship with the user changed.
    RelationshipUpdate,
    /// A concrete observation was recorded.
    Finding,
    /// A higher-level synthesis or realization was recorded.
    Insight,
    /// A factual piece of information was learned.
    FactLearned,
    /// A recurring pattern was detected across events or interactions.
    PatternDetected,
    /// A tentative explanation or prediction was formed.
    Hypothesis,
    /// The agent recorded an error in its prior reasoning or action.
    Mistake,
    /// The agent recorded the corrected version of a prior mistake.
    Correction,
    /// A durable lesson or operating heuristic was distilled from prior work.
    LessonLearned,
    /// A previously trusted assumption was invalidated.
    AssumptionInvalidated,
    /// A requirement or hard limit was identified.
    Constraint,
    /// A plan for future work was created or updated.
    Plan,
    /// A smaller unit of work was carved out from a broader plan.
    Subgoal,
    /// A concrete choice was made.
    Decision,
    /// The agent changed its overall approach.
    StrategyShift,
    /// An open-ended curiosity or line of exploration was recorded.
    Wonder,
    /// An unresolved question was recorded.
    Question,
    /// A possible future direction or design concept was proposed.
    Idea,
    /// An experiment or trial was proposed or executed.
    Experiment,
    /// A meaningful action was performed.
    ActionTaken,
    /// A task or milestone was completed.
    TaskComplete,
    /// A checkpoint suitable for resumption was recorded.
    Checkpoint,
    /// A broader snapshot of current state was recorded.
    StateSnapshot,
    /// Work or context was explicitly handed to another actor.
    Handoff,
    /// A summary view of prior thoughts was recorded.
    Summary,
    /// An unexpected outcome or mismatch was observed.
    Surprise,
}

/// Operational role of a thought inside the system.
///
/// Roles answer how a thought is being used by the system, which lets callers
/// distinguish semantic meaning from lifecycle mechanics.
///
/// # Example
///
/// ```
/// use thoughtchain::ThoughtRole;
///
/// assert_eq!(ThoughtRole::default(), ThoughtRole::Memory);
/// ```
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash, Default)]
pub enum ThoughtRole {
    /// Durable long-term memory.
    #[default]
    Memory,
    /// Shorter-lived or more speculative working memory.
    WorkingMemory,
    /// A synthesized summary role.
    Summary,
    /// A role emitted during context compression.
    Compression,
    /// A role intended primarily for resumption checkpoints.
    Checkpoint,
    /// A role intended for handoff to another actor or process.
    Handoff,
    /// A role intended mainly for traceability or audit logs.
    Audit,
    /// A role emitted during deliberate post-incident or post-struggle reflection.
    Retrospective,
}

/// Why a thought points to another thought.
///
/// # Example
///
/// ```
/// use thoughtchain::ThoughtRelationKind;
///
/// assert_eq!(ThoughtRelationKind::Corrects as u8, ThoughtRelationKind::Corrects as u8);
/// ```
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum ThoughtRelationKind {
    /// A general back-reference.
    References,
    /// The source thought summarizes the target thought.
    Summarizes,
    /// The source thought corrects the target thought.
    Corrects,
    /// The source thought invalidates the target thought.
    Invalidates,
    /// The source thought was caused by the target thought.
    CausedBy,
    /// The source thought supports the target thought.
    Supports,
    /// The source thought contradicts the target thought.
    Contradicts,
    /// The source thought was derived from the target thought.
    DerivedFrom,
    /// The source thought continues the work or state of the target thought.
    ContinuesFrom,
    /// A generic semantic relation exists between source and target.
    RelatedTo,
}

/// Typed edge in the thought graph.
///
/// A relation explains why one thought points to another thought. This avoids
/// a common misconception: not every link is just a generic "reference". A
/// later thought may correct, summarize, support, or continue an earlier one,
/// and that semantic meaning matters during replay, inspection, and retrieval.
///
/// `ThoughtRelation` is more expressive than raw `refs`. Use `refs` when a
/// simple positional backlink is enough. Use relations when the meaning of the
/// link should survive into downstream tools, summaries, and audits.
///
/// # Example
///
/// ```
/// use thoughtchain::{ThoughtRelation, ThoughtRelationKind};
/// use uuid::Uuid;
///
/// let relation = ThoughtRelation {
///     kind: ThoughtRelationKind::Supports,
///     target_id: Uuid::nil(),
/// };
///
/// assert_eq!(relation.kind, ThoughtRelationKind::Supports);
/// ```
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ThoughtRelation {
    /// Semantic meaning of the edge.
    pub kind: ThoughtRelationKind,
    /// Stable id of the target thought.
    pub target_id: Uuid,
}

/// Builder-like input struct used to append rich thoughts.
///
/// `ThoughtInput` is the caller-authored description of a memory to be
/// committed. It is not yet part of the durable chain. Callers use it to say
/// what the thought means, how important it is, which earlier thoughts it
/// refers to, and which optional metadata should accompany it.
///
/// ThoughtChain then turns that input into a persisted [`Thought`] by adding
/// the system-managed fields that callers should not forge directly, such as:
///
/// - the stable thought `id`
/// - the chain `index`
/// - the commit `timestamp`
/// - the writer `agent_id`
/// - the `prev_hash`
/// - the final `hash`
///
/// In short:
///
/// - `ThoughtInput` is the proposed memory payload
/// - [`Thought`] is the committed memory record
///
/// Use `ThoughtInput` when you want richer metadata than the simple
/// [`ThoughtChain::append`] helper allows.
///
/// # Example
///
/// ```
/// use thoughtchain::{ThoughtInput, ThoughtRole, ThoughtType};
///
/// let input = ThoughtInput::new(ThoughtType::Insight, "Rate limiting is the real bottleneck.")
///     .with_role(ThoughtRole::Summary)
///     .with_importance(0.9)
///     .with_tags(["api", "performance"]);
///
/// assert_eq!(input.thought_type, ThoughtType::Insight);
/// assert_eq!(input.role, ThoughtRole::Summary);
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThoughtInput {
    /// Optional session identifier associated with the thought.
    ///
    /// This groups related thoughts from one run without changing the chain's
    /// durable identity.
    pub session_id: Option<Uuid>,
    /// Optional human-readable name of the producing agent.
    ///
    /// This is display metadata. The stable producer identity is `agent_id`,
    /// which is supplied separately when appending.
    pub agent_name: Option<String>,
    /// Optional owner or grouping label for the producing agent.
    ///
    /// Useful for shared chains, tenants, or human ownership models.
    pub agent_owner: Option<String>,
    /// Semantic meaning of the thought.
    ///
    /// This answers "what kind of memory is this?"
    pub thought_type: ThoughtType,
    /// Operational role played by this thought.
    ///
    /// This answers "why is the system emitting or using this memory?"
    pub role: ThoughtRole,
    /// Primary human-readable content.
    ///
    /// This should be a durable memory statement, not hidden chain-of-thought.
    pub content: String,
    /// Optional confidence score between `0.0` and `1.0`.
    ///
    /// Use this when the content is uncertain or speculative.
    pub confidence: Option<f32>,
    /// Importance score between `0.0` and `1.0`.
    ///
    /// Higher values indicate memories that should matter more during
    /// retrieval, summarization, or pruning.
    pub importance: f32,
    /// Free-form tags for retrieval.
    ///
    /// Tags are lightweight labels supplied by the caller.
    pub tags: Vec<String>,
    /// Concept labels or semantic anchors for retrieval.
    ///
    /// Concepts are intended to be more semantic and reusable than ad hoc
    /// tags, though both can coexist.
    pub concepts: Vec<String>,
    /// Back-references to prior thought indices.
    ///
    /// These are compact positional links into the same chain.
    pub refs: Vec<u64>,
    /// Typed graph relations to prior thoughts.
    ///
    /// These preserve the meaning of the link, not just the fact that a link
    /// exists.
    pub relations: Vec<ThoughtRelation>,
}

impl ThoughtInput {
    /// Create a new input with default metadata.
    ///
    /// Defaults:
    /// - `role`: [`ThoughtRole::Memory`]
    /// - `importance`: `0.5`
    /// - `confidence`: `None`
    ///
    /// # Example
    ///
    /// ```
    /// use thoughtchain::{ThoughtInput, ThoughtRole, ThoughtType};
    ///
    /// let input = ThoughtInput::new(ThoughtType::Plan, "Build a query index first.");
    ///
    /// assert_eq!(input.role, ThoughtRole::Memory);
    /// assert_eq!(input.importance, 0.5);
    /// ```
    pub fn new(thought_type: ThoughtType, content: impl Into<String>) -> Self {
        Self {
            session_id: None,
            agent_name: None,
            agent_owner: None,
            thought_type,
            role: ThoughtRole::Memory,
            content: content.into(),
            confidence: None,
            importance: 0.5,
            tags: Vec::new(),
            concepts: Vec::new(),
            refs: Vec::new(),
            relations: Vec::new(),
        }
    }

    /// Attach a session identifier to the thought.
    pub fn with_session_id(mut self, session_id: Uuid) -> Self {
        self.session_id = Some(session_id);
        self
    }

    /// Attach a human-readable agent name to the thought.
    pub fn with_agent_name(mut self, agent_name: impl Into<String>) -> Self {
        self.agent_name = Some(agent_name.into());
        self
    }

    /// Attach an owner or grouping label to the thought.
    pub fn with_agent_owner(mut self, agent_owner: impl Into<String>) -> Self {
        self.agent_owner = Some(agent_owner.into());
        self
    }

    /// Override the operational role of the thought.
    pub fn with_role(mut self, role: ThoughtRole) -> Self {
        self.role = role;
        self
    }

    /// Attach a confidence score between `0.0` and `1.0`.
    pub fn with_confidence(mut self, confidence: f32) -> Self {
        self.confidence = Some(confidence.clamp(0.0, 1.0));
        self
    }

    /// Attach an importance score between `0.0` and `1.0`.
    pub fn with_importance(mut self, importance: f32) -> Self {
        self.importance = importance.clamp(0.0, 1.0);
        self
    }

    /// Replace the thought's tags.
    pub fn with_tags<I, S>(mut self, tags: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.tags = tags.into_iter().map(Into::into).collect();
        self
    }

    /// Replace the thought's concept labels.
    pub fn with_concepts<I, S>(mut self, concepts: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.concepts = concepts.into_iter().map(Into::into).collect();
        self
    }

    /// Add back-references to prior thought indices.
    pub fn with_refs(mut self, refs: Vec<u64>) -> Self {
        self.refs = refs;
        self
    }

    /// Add typed graph relations to prior thoughts.
    pub fn with_relations(mut self, relations: Vec<ThoughtRelation>) -> Self {
        self.relations = relations;
        self
    }
}

/// A single durable thought record.
///
/// `Thought` is the committed record that ThoughtChain stores and returns. It
/// contains the semantic memory payload together with the fields required for
/// ordering, attribution, and integrity verification.
///
/// A caller typically does not construct this type directly. Instead, the
/// caller provides a [`ThoughtInput`], and ThoughtChain produces a `Thought`
/// with system-managed fields filled in. This distinction prevents accidental
/// confusion between "memory content proposed by an agent" and "memory record
/// accepted into the chain".
///
/// # Example
///
/// ```rust,no_run
/// use std::path::PathBuf;
/// use thoughtchain::{ThoughtChain, ThoughtType};
///
/// # fn main() -> std::io::Result<()> {
/// let mut chain = ThoughtChain::open(&PathBuf::from("/tmp/tc_doc"), "agent1", "Agent", None, None)?;
/// let thought = chain.append("agent1", ThoughtType::Finding, "The cache hit rate is 97%.")?;
///
/// assert_eq!(thought.index, 0);
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Thought {
    /// Stable unique identifier for this thought.
    ///
    /// This is the canonical target for future semantic relations.
    pub id: Uuid,
    /// Zero-based position within the chain.
    ///
    /// This reflects append order inside one chain. It is not a global ID.
    pub index: u64,
    /// UTC timestamp when the thought was recorded.
    ///
    /// Assigned at commit time by ThoughtChain.
    pub timestamp: DateTime<Utc>,
    /// Optional session identifier associated with the thought.
    pub session_id: Option<Uuid>,
    /// Stable identifier of the producing agent.
    ///
    /// This answers who wrote the record in a shared chain.
    pub agent_id: String,
    /// Human-readable name of the producing agent.
    ///
    /// Friendly display label associated with `agent_id`.
    #[serde(default)]
    pub agent_name: String,
    /// Optional owner or grouping label for the producing agent.
    #[serde(default)]
    pub agent_owner: Option<String>,
    /// Semantic meaning of the thought.
    pub thought_type: ThoughtType,
    /// Operational role played by this thought.
    pub role: ThoughtRole,
    /// Primary human-readable content.
    pub content: String,
    /// Optional confidence score between `0.0` and `1.0`.
    pub confidence: Option<f32>,
    /// Importance score between `0.0` and `1.0`.
    pub importance: f32,
    /// Free-form tags for retrieval.
    pub tags: Vec<String>,
    /// Concept labels or semantic anchors for retrieval.
    pub concepts: Vec<String>,
    /// Back-references to prior thought indices.
    pub refs: Vec<u64>,
    /// Typed graph relations to prior thoughts.
    pub relations: Vec<ThoughtRelation>,
    /// Hash of the previous thought in the chain.
    ///
    /// This links the record to the prior committed chain state.
    pub prev_hash: String,
    /// SHA-256 hash of this thought's canonical contents.
    ///
    /// This is the record's integrity fingerprint.
    pub hash: String,
}

/// Retrieval filter for semantic memory queries.
///
/// `ThoughtQuery` lets callers ask for slices of memory without replaying the
/// entire chain.
///
/// `ThoughtQuery` is read-only. It does not create or modify thoughts. Its job
/// is to select already-committed [`Thought`] records by semantic type,
/// operational role, agent identity, tags, concepts, text, confidence,
/// importance, and time range.
///
/// The relationship between the three main data shapes is:
///
/// - `ThoughtInput`: proposed memory to append
/// - `Thought`: committed durable memory
/// - `ThoughtQuery`: retrieval filter over committed memory
///
/// # Example
///
/// ```
/// use thoughtchain::{ThoughtQuery, ThoughtType};
///
/// let query = ThoughtQuery::new()
///     .with_types(vec![ThoughtType::Decision, ThoughtType::Constraint])
///     .with_min_importance(0.8);
///
/// assert!(query.min_importance.is_some());
/// ```
#[derive(Debug, Clone, Default)]
pub struct ThoughtQuery {
    /// Semantic thought types to match.
    pub thought_types: Option<Vec<ThoughtType>>,
    /// Operational roles to match.
    pub roles: Option<Vec<ThoughtRole>>,
    /// Agent ids to match.
    pub agent_ids: Option<Vec<String>>,
    /// Agent names to match.
    pub agent_names: Option<Vec<String>>,
    /// Agent owners to match.
    pub agent_owners: Option<Vec<String>>,
    /// Match if any tag matches.
    pub tags_any: Vec<String>,
    /// Match if any concept matches.
    pub concepts_any: Vec<String>,
    /// Text filter applied to content, tags, and concepts.
    pub text_contains: Option<String>,
    /// Minimum importance threshold.
    pub min_importance: Option<f32>,
    /// Minimum confidence threshold.
    pub min_confidence: Option<f32>,
    /// Start of the timestamp window, inclusive.
    pub since: Option<DateTime<Utc>>,
    /// End of the timestamp window, inclusive.
    pub until: Option<DateTime<Utc>>,
    /// Maximum number of thoughts to return.
    pub limit: Option<usize>,
}

impl ThoughtQuery {
    /// Create an empty query that matches every thought.
    pub fn new() -> Self {
        Self::default()
    }

    /// Limit matches to the provided semantic thought types.
    pub fn with_types(mut self, thought_types: Vec<ThoughtType>) -> Self {
        self.thought_types = Some(thought_types);
        self
    }

    /// Limit matches to the provided thought roles.
    pub fn with_roles(mut self, roles: Vec<ThoughtRole>) -> Self {
        self.roles = Some(roles);
        self
    }

    /// Limit matches to the provided agent identifiers.
    pub fn with_agent_ids<I, S>(mut self, agent_ids: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.agent_ids = Some(agent_ids.into_iter().map(Into::into).collect());
        self
    }

    /// Limit matches to the provided agent names.
    pub fn with_agent_names<I, S>(mut self, agent_names: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.agent_names = Some(agent_names.into_iter().map(Into::into).collect());
        self
    }

    /// Limit matches to the provided agent owner labels.
    pub fn with_agent_owners<I, S>(mut self, agent_owners: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.agent_owners = Some(agent_owners.into_iter().map(Into::into).collect());
        self
    }

    /// Match thoughts that have at least one of the provided tags.
    pub fn with_tags_any<I, S>(mut self, tags: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.tags_any = tags.into_iter().map(Into::into).collect();
        self
    }

    /// Match thoughts that have at least one of the provided concepts.
    pub fn with_concepts_any<I, S>(mut self, concepts: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.concepts_any = concepts.into_iter().map(Into::into).collect();
        self
    }

    /// Match thoughts whose content, tags, or concepts contain the provided text.
    pub fn with_text(mut self, text: impl Into<String>) -> Self {
        self.text_contains = Some(text.into());
        self
    }

    /// Only match thoughts whose importance is at least this value.
    pub fn with_min_importance(mut self, importance: f32) -> Self {
        self.min_importance = Some(importance.clamp(0.0, 1.0));
        self
    }

    /// Only match thoughts whose confidence is at least this value.
    pub fn with_min_confidence(mut self, confidence: f32) -> Self {
        self.min_confidence = Some(confidence.clamp(0.0, 1.0));
        self
    }

    /// Only match thoughts at or after the given timestamp.
    pub fn with_since(mut self, since: DateTime<Utc>) -> Self {
        self.since = Some(since);
        self
    }

    /// Only match thoughts at or before the given timestamp.
    pub fn with_until(mut self, until: DateTime<Utc>) -> Self {
        self.until = Some(until);
        self
    }

    /// Limit the number of returned thoughts.
    pub fn with_limit(mut self, limit: usize) -> Self {
        self.limit = Some(limit);
        self
    }

    fn matches(&self, thought: &Thought) -> bool {
        if let Some(types) = &self.thought_types {
            if !types.contains(&thought.thought_type) {
                return false;
            }
        }

        if let Some(roles) = &self.roles {
            if !roles.contains(&thought.role) {
                return false;
            }
        }

        if let Some(agent_ids) = &self.agent_ids {
            if !agent_ids
                .iter()
                .any(|agent_id| agent_id == &thought.agent_id)
            {
                return false;
            }
        }

        if let Some(agent_names) = &self.agent_names {
            if !agent_names
                .iter()
                .any(|agent_name| equals_case_insensitive(&thought.agent_name, agent_name))
            {
                return false;
            }
        }

        if let Some(agent_owners) = &self.agent_owners {
            let Some(owner) = &thought.agent_owner else {
                return false;
            };
            if !agent_owners
                .iter()
                .any(|agent_owner| equals_case_insensitive(owner, agent_owner))
            {
                return false;
            }
        }

        if let Some(min_importance) = self.min_importance {
            if thought.importance < min_importance {
                return false;
            }
        }

        if let Some(min_confidence) = self.min_confidence {
            match thought.confidence {
                Some(confidence) if confidence >= min_confidence => {}
                _ => return false,
            }
        }

        if let Some(since) = self.since {
            if thought.timestamp < since {
                return false;
            }
        }

        if let Some(until) = self.until {
            if thought.timestamp > until {
                return false;
            }
        }

        if !self.tags_any.is_empty()
            && !self
                .tags_any
                .iter()
                .any(|tag| contains_case_insensitive(&thought.tags, tag))
        {
            return false;
        }

        if !self.concepts_any.is_empty()
            && !self
                .concepts_any
                .iter()
                .any(|concept| contains_case_insensitive(&thought.concepts, concept))
        {
            return false;
        }

        if let Some(text) = &self.text_contains {
            let needle = text.to_lowercase();
            if !thought.content.to_lowercase().contains(&needle)
                && !thought.agent_id.to_lowercase().contains(&needle)
                && !thought.agent_name.to_lowercase().contains(&needle)
                && !thought
                    .agent_owner
                    .as_ref()
                    .map(|owner| owner.to_lowercase().contains(&needle))
                    .unwrap_or(false)
                && !thought
                    .tags
                    .iter()
                    .any(|tag| tag.to_lowercase().contains(&needle))
                && !thought
                    .concepts
                    .iter()
                    .any(|concept| concept.to_lowercase().contains(&needle))
            {
                return false;
            }
        }

        true
    }
}

/// Append-only, hash-chained semantic memory store.
///
/// `ThoughtChain` stores thoughts in memory and persists them through a
/// [`StorageAdapter`]. Every record includes a SHA-256 hash of its canonical
/// contents plus the previous record hash, making offline tampering
/// detectable. The default backend is newline-delimited JSON via
/// [`JsonlStorageAdapter`].
///
/// # Example
///
/// ```rust,no_run
/// use std::path::PathBuf;
/// use thoughtchain::{ThoughtChain, ThoughtType};
///
/// # fn main() -> std::io::Result<()> {
/// let mut chain = ThoughtChain::open(&PathBuf::from("/tmp/tc_chain"), "researcher", "Researcher", None, None)?;
/// chain.append("researcher", ThoughtType::FactLearned, "The corpus contains 4 million rows.")?;
///
/// assert!(chain.verify_integrity());
/// # Ok(())
/// # }
/// ```
pub struct ThoughtChain {
    thoughts: Vec<Thought>,
    id_to_index: HashMap<Uuid, usize>,
    storage: Box<dyn StorageAdapter>,
    auto_flush: bool,
}

impl ThoughtChain {
    /// Open or create a chain using the agent id as the durable storage key.
    ///
    /// The additional identity parameters are accepted for compatibility with
    /// `cloudllm`, but storage identity is now derived from `agent_id` so
    /// changing an agent's profile does not fork its memory file.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use std::path::PathBuf;
    /// use thoughtchain::ThoughtChain;
    ///
    /// # fn main() -> std::io::Result<()> {
    /// let chain = ThoughtChain::open(&PathBuf::from("/tmp/tc_open"), "agent1", "Agent", None, None)?;
    /// assert!(chain.thoughts().is_empty());
    /// # Ok(())
    /// # }
    /// ```
    pub fn open(
        chain_dir: &PathBuf,
        agent_id: &str,
        _agent_name: &str,
        _expertise: Option<&str>,
        _personality: Option<&str>,
    ) -> io::Result<Self> {
        Self::open_with_key(chain_dir, agent_id)
    }

    /// Open or create a chain using a caller-provided storage adapter.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use std::path::PathBuf;
    /// use thoughtchain::{JsonlStorageAdapter, ThoughtChain};
    ///
    /// # fn main() -> std::io::Result<()> {
    /// let adapter = JsonlStorageAdapter::for_chain_key(PathBuf::from("/tmp/tc_custom"), "project-memory");
    /// let chain = ThoughtChain::open_with_storage(Box::new(adapter))?;
    /// assert!(chain.thoughts().is_empty());
    /// # Ok(())
    /// # }
    /// ```
    pub fn open_with_storage(storage: Box<dyn StorageAdapter>) -> io::Result<Self> {
        let thoughts = storage.load_thoughts()?;

        let mut id_to_index = HashMap::new();
        for (position, thought) in thoughts.iter().enumerate() {
            if thought.index != position as u64 {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!(
                        "Thought index {} does not match position {}",
                        thought.index, position
                    ),
                ));
            }
            id_to_index.insert(thought.id, position);
        }

        let chain = Self {
            thoughts,
            id_to_index,
            storage,
            auto_flush: true,
        };

        if !chain.verify_integrity() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Thought chain integrity verification failed",
            ));
        }

        Ok(chain)
    }

    /// Open or create a chain using an explicit stable chain key.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use std::path::PathBuf;
    /// use thoughtchain::ThoughtChain;
    ///
    /// # fn main() -> std::io::Result<()> {
    /// let chain = ThoughtChain::open_with_key(PathBuf::from("/tmp/tc_key"), "project-memory")?;
    /// assert!(chain.thoughts().is_empty());
    /// # Ok(())
    /// # }
    /// ```
    pub fn open_with_key<P: AsRef<Path>>(chain_dir: P, chain_key: &str) -> io::Result<Self> {
        fs::create_dir_all(chain_dir.as_ref())?;
        Self::open_with_storage(Box::new(JsonlStorageAdapter::for_chain_key(
            chain_dir, chain_key,
        )))
    }

    /// Append a simple thought with default metadata and no references.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use std::path::PathBuf;
    /// use thoughtchain::{ThoughtChain, ThoughtType};
    ///
    /// # fn main() -> std::io::Result<()> {
    /// let mut chain = ThoughtChain::open(&PathBuf::from("/tmp/tc_append"), "agent1", "Agent", None, None)?;
    /// let thought = chain.append("agent1", ThoughtType::Decision, "Use SQLite for local state.")?;
    ///
    /// assert_eq!(thought.index, 0);
    /// # Ok(())
    /// # }
    /// ```
    pub fn append(
        &mut self,
        agent_id: &str,
        thought_type: ThoughtType,
        content: &str,
    ) -> io::Result<&Thought> {
        self.append_thought(agent_id, ThoughtInput::new(thought_type, content))
    }

    /// Append a simple thought that references prior thought indices.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use std::path::PathBuf;
    /// use thoughtchain::{ThoughtChain, ThoughtType};
    ///
    /// # fn main() -> std::io::Result<()> {
    /// let mut chain = ThoughtChain::open(&PathBuf::from("/tmp/tc_refs"), "agent1", "Agent", None, None)?;
    /// chain.append("agent1", ThoughtType::Finding, "Observed rising latency.")?;
    /// let summary = chain.append_with_refs("agent1", ThoughtType::Summary, "Latency issue captured.", vec![0])?;
    ///
    /// assert_eq!(summary.refs, vec![0]);
    /// # Ok(())
    /// # }
    /// ```
    pub fn append_with_refs(
        &mut self,
        agent_id: &str,
        thought_type: ThoughtType,
        content: &str,
        refs: Vec<u64>,
    ) -> io::Result<&Thought> {
        self.append_thought(
            agent_id,
            ThoughtInput::new(thought_type, content).with_refs(refs),
        )
    }

    /// Append a rich thought with semantic metadata, tags, concepts, and relations.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use std::path::PathBuf;
    /// use thoughtchain::{ThoughtChain, ThoughtInput, ThoughtRole, ThoughtType};
    ///
    /// # fn main() -> std::io::Result<()> {
    /// let mut chain = ThoughtChain::open(&PathBuf::from("/tmp/tc_rich"), "agent1", "Agent", None, None)?;
    /// let input = ThoughtInput::new(ThoughtType::Constraint, "The system must work offline.")
    ///     .with_role(ThoughtRole::Checkpoint)
    ///     .with_importance(0.95)
    ///     .with_tags(["offline", "ops"]);
    /// chain.append_thought("agent1", input)?;
    ///
    /// assert_eq!(chain.thoughts().len(), 1);
    /// # Ok(())
    /// # }
    /// ```
    pub fn append_thought(
        &mut self,
        agent_id: &str,
        mut input: ThoughtInput,
    ) -> io::Result<&Thought> {
        validate_refs(&self.thoughts, &input.refs)?;

        let mut relations = input.relations.clone();
        for &reference_index in &input.refs {
            if let Some(target) = self.thoughts.get(reference_index as usize) {
                relations.push(ThoughtRelation {
                    kind: ThoughtRelationKind::References,
                    target_id: target.id,
                });
            }
        }
        dedupe_relations(&mut relations);

        let index = self.thoughts.len() as u64;
        let prev_hash = self
            .thoughts
            .last()
            .map(|thought| thought.hash.clone())
            .unwrap_or_default();
        let timestamp = Utc::now();
        input.importance = input.importance.clamp(0.0, 1.0);
        let thought = Thought {
            id: Uuid::new_v4(),
            index,
            timestamp,
            session_id: input.session_id,
            agent_id: agent_id.to_string(),
            agent_name: input
                .agent_name
                .take()
                .map(|name| name.trim().to_string())
                .filter(|name| !name.is_empty())
                .unwrap_or_else(|| agent_id.to_string()),
            agent_owner: input
                .agent_owner
                .take()
                .map(|owner| owner.trim().to_string())
                .filter(|owner| !owner.is_empty()),
            thought_type: input.thought_type,
            role: input.role,
            content: input.content,
            confidence: input.confidence.map(|value| value.clamp(0.0, 1.0)),
            importance: input.importance,
            tags: normalize_strings(input.tags),
            concepts: normalize_strings(input.concepts),
            refs: input.refs,
            relations,
            prev_hash,
            hash: String::new(),
        };

        let hash = compute_thought_hash(&thought);
        let thought = Thought { hash, ..thought };

        if self.auto_flush {
            self.storage.append_thought(&thought)?;
        }

        self.id_to_index.insert(thought.id, self.thoughts.len());
        self.thoughts.push(thought.clone());
        Ok(self.thoughts.last().unwrap())
    }

    /// Verify the entire hash chain and sequence invariants.
    ///
    /// Returns `false` if:
    /// - any `index` does not match its position
    /// - any `prev_hash` does not match the previous thought hash
    /// - any thought hash does not match its recomputed canonical hash
    pub fn verify_integrity(&self) -> bool {
        let mut prev_hash = String::new();
        for (position, thought) in self.thoughts.iter().enumerate() {
            if thought.index != position as u64 {
                return false;
            }
            if thought.prev_hash != prev_hash {
                return false;
            }
            if thought.hash != compute_thought_hash(thought) {
                return false;
            }
            prev_hash = thought.hash.clone();
        }
        true
    }

    /// Resolve all context reachable from the target thought index.
    ///
    /// Traversal follows both explicit `refs` and typed relations.
    pub fn resolve_context(&self, target_index: u64) -> Vec<&Thought> {
        let Some(target) = self.thoughts.get(target_index as usize) else {
            return Vec::new();
        };
        self.resolve_context_by_id(target.id)
    }

    /// Resolve all context reachable from the target thought id.
    pub fn resolve_context_by_id(&self, target_id: Uuid) -> Vec<&Thought> {
        let mut visited = HashSet::new();
        let mut stack = vec![target_id];

        while let Some(id) = stack.pop() {
            if !visited.insert(id) {
                continue;
            }

            if let Some(&position) = self.id_to_index.get(&id) {
                let thought = &self.thoughts[position];
                for relation in &thought.relations {
                    if !visited.contains(&relation.target_id) {
                        stack.push(relation.target_id);
                    }
                }
                for &reference_index in &thought.refs {
                    if let Some(reference) = self.thoughts.get(reference_index as usize) {
                        if !visited.contains(&reference.id) {
                            stack.push(reference.id);
                        }
                    }
                }
            }
        }

        let mut resolved: Vec<&Thought> = visited
            .into_iter()
            .filter_map(|id| self.id_to_index.get(&id).copied())
            .map(|position| &self.thoughts[position])
            .collect();
        resolved.sort_by_key(|thought| thought.index);
        resolved
    }

    /// Query the chain using semantic filters.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use std::path::PathBuf;
    /// use thoughtchain::{ThoughtChain, ThoughtQuery, ThoughtType};
    ///
    /// # fn main() -> std::io::Result<()> {
    /// let mut chain = ThoughtChain::open(&PathBuf::from("/tmp/tc_query"), "agent1", "Agent", None, None)?;
    /// chain.append("agent1", ThoughtType::Decision, "Use SQLite for local state.")?;
    ///
    /// let results = chain.query(&ThoughtQuery::new().with_types(vec![ThoughtType::Decision]));
    /// assert_eq!(results.len(), 1);
    /// # Ok(())
    /// # }
    /// ```
    pub fn query(&self, query: &ThoughtQuery) -> Vec<&Thought> {
        let mut results: Vec<&Thought> = self
            .thoughts
            .iter()
            .filter(|thought| query.matches(thought))
            .collect();

        if let Some(limit) = query.limit {
            if results.len() > limit {
                results = results[results.len() - limit..].to_vec();
            }
        }

        results
    }

    /// Convenience helper to find thoughts related to a concept string.
    pub fn related_to_concept(&self, concept: &str, limit: usize) -> Vec<&Thought> {
        self.query(
            &ThoughtQuery::new()
                .with_concepts_any([concept])
                .with_limit(limit),
        )
    }

    /// Render a context reconstruction prompt for a target thought.
    pub fn to_bootstrap_prompt(&self, target_index: u64) -> String {
        let resolved = self.resolve_context(target_index);
        if resolved.is_empty() {
            return String::new();
        }

        let mut prompt = String::from("=== RESTORED CONTEXT (from ThoughtChain) ===\n\n");
        for thought in resolved {
            prompt.push_str(&format!(
                "[#{}] {:?} / {:?} ({})\n{}\n",
                thought.index,
                thought.thought_type,
                thought.role,
                agent_label(thought),
                thought.content
            ));
            if let Some(confidence) = thought.confidence {
                prompt.push_str(&format!("  confidence: {:.2}\n", confidence));
            }
            prompt.push_str(&format!("  importance: {:.2}\n", thought.importance));
            if !thought.tags.is_empty() {
                prompt.push_str(&format!("  tags: {}\n", thought.tags.join(", ")));
            }
            if !thought.concepts.is_empty() {
                prompt.push_str(&format!("  concepts: {}\n", thought.concepts.join(", ")));
            }
            if !thought.refs.is_empty() {
                prompt.push_str(&format!("  refs: {:?}\n", thought.refs));
            }
        }
        prompt.push_str("\n=== END RESTORED CONTEXT ===\n");
        prompt
    }

    /// Render the last `n` thoughts as a lightweight catch-up prompt.
    pub fn to_catchup_prompt(&self, last_n: usize) -> String {
        let start = self.thoughts.len().saturating_sub(last_n);
        let tail = &self.thoughts[start..];
        if tail.is_empty() {
            return String::new();
        }

        let mut prompt = String::from("=== RECENT CONTEXT ===\n\n");
        for thought in tail {
            prompt.push_str(&format!(
                "[#{}] {:?} / {:?} ({}) {}\n",
                thought.index,
                thought.thought_type,
                thought.role,
                agent_label(thought),
                thought.content
            ));
        }
        prompt.push_str("\n=== END RECENT CONTEXT ===\n");
        prompt
    }

    /// Export a Markdown memory view.
    ///
    /// This is suitable for generating a `MEMORY.md`-style summary from a full
    /// chain or a queried subset of thoughts.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use std::path::PathBuf;
    /// use thoughtchain::{ThoughtChain, ThoughtType};
    ///
    /// # fn main() -> std::io::Result<()> {
    /// let mut chain = ThoughtChain::open(&PathBuf::from("/tmp/tc_md"), "agent1", "Agent", None, None)?;
    /// chain.append("agent1", ThoughtType::PreferenceUpdate, "User prefers concise Markdown.")?;
    ///
    /// let markdown = chain.to_memory_markdown(None);
    /// assert!(markdown.contains("# MEMORY"));
    /// # Ok(())
    /// # }
    /// ```
    pub fn to_memory_markdown(&self, query: Option<&ThoughtQuery>) -> String {
        let thoughts = query
            .map(|query| self.query(query))
            .unwrap_or_else(|| self.thoughts.iter().collect());

        let mut markdown = String::from("# MEMORY\n\n");
        markdown.push_str(&format!(
            "Generated from `{}` with {} thought(s).\n\n",
            self.storage.storage_location(),
            thoughts.len()
        ));

        append_memory_section(
            &mut markdown,
            "Identity",
            &thoughts,
            &[
                ThoughtType::PreferenceUpdate,
                ThoughtType::UserTrait,
                ThoughtType::RelationshipUpdate,
            ],
        );
        append_memory_section(
            &mut markdown,
            "Knowledge",
            &thoughts,
            &[
                ThoughtType::Finding,
                ThoughtType::Insight,
                ThoughtType::FactLearned,
                ThoughtType::PatternDetected,
                ThoughtType::Hypothesis,
                ThoughtType::Surprise,
            ],
        );
        append_memory_section(
            &mut markdown,
            "Constraints And Decisions",
            &thoughts,
            &[
                ThoughtType::Constraint,
                ThoughtType::Plan,
                ThoughtType::Subgoal,
                ThoughtType::Decision,
                ThoughtType::StrategyShift,
            ],
        );
        append_memory_section(
            &mut markdown,
            "Corrections",
            &thoughts,
            &[
                ThoughtType::Mistake,
                ThoughtType::Correction,
                ThoughtType::LessonLearned,
                ThoughtType::AssumptionInvalidated,
            ],
        );
        append_memory_section(
            &mut markdown,
            "Open Threads",
            &thoughts,
            &[
                ThoughtType::Wonder,
                ThoughtType::Question,
                ThoughtType::Idea,
                ThoughtType::Experiment,
            ],
        );
        append_memory_section(
            &mut markdown,
            "Execution State",
            &thoughts,
            &[
                ThoughtType::ActionTaken,
                ThoughtType::TaskComplete,
                ThoughtType::Checkpoint,
                ThoughtType::StateSnapshot,
                ThoughtType::Handoff,
                ThoughtType::Summary,
            ],
        );

        markdown
    }

    /// Return all thoughts in chronological order.
    pub fn thoughts(&self) -> &[Thought] {
        &self.thoughts
    }

    /// Return the current head hash of the chain, if any.
    pub fn head_hash(&self) -> Option<&str> {
        self.thoughts.last().map(|thought| thought.hash.as_str())
    }

    /// Return a human-readable description of the underlying storage location.
    pub fn storage_location(&self) -> String {
        self.storage.storage_location()
    }

    /// Enable or disable immediate persistence on append.
    pub fn set_auto_flush(&mut self, auto_flush: bool) {
        self.auto_flush = auto_flush;
    }
}

/// Stable filename derived from a chain key rather than mutable agent profile data.
///
/// # Example
///
/// ```
/// use thoughtchain::chain_filename;
///
/// let a = chain_filename("agent1", "Researcher", Some("rust"), Some("careful"));
/// let b = chain_filename("agent1", "Different", Some("go"), Some("direct"));
/// let c = chain_filename("agent2", "Researcher", Some("rust"), Some("careful"));
///
/// assert_eq!(a, b);
/// assert_ne!(a, c);
/// ```
pub fn chain_filename(
    chain_key: &str,
    _agent_name: &str,
    _expertise: Option<&str>,
    _personality: Option<&str>,
) -> String {
    chain_storage_filename(chain_key, StorageAdapterKind::Jsonl)
}

/// Stable filename derived from a chain key and storage adapter kind.
///
/// # Example
///
/// ```
/// use thoughtchain::{chain_storage_filename, StorageAdapterKind};
///
/// let jsonl = chain_storage_filename("agent1", StorageAdapterKind::Jsonl);
/// let binary = chain_storage_filename("agent1", StorageAdapterKind::Binary);
///
/// assert!(jsonl.ends_with(".jsonl"));
/// assert!(binary.ends_with(".tcbin"));
/// ```
pub fn chain_storage_filename(chain_key: &str, kind: StorageAdapterKind) -> String {
    let mut hasher = Sha256::new();
    hasher.update(chain_key.as_bytes());
    let digest = format!("{:x}", hasher.finalize());
    let fingerprint = &digest[..16];

    let safe_key: String = chain_key
        .chars()
        .map(|character| {
            if character.is_alphanumeric() || character == '-' || character == '_' {
                character
            } else {
                '_'
            }
        })
        .collect();

    format!("{safe_key}-{fingerprint}.{}", kind.file_extension())
}

/// Recover the stable chain key portion from a ThoughtChain storage filename.
///
/// This reverses the filename convention used by [`chain_storage_filename`]
/// and returns the durable chain key prefix as stored in the filename. The
/// returned value matches the filename-safe key, so callers should treat it as
/// the persisted chain identifier rather than as an exact reconstruction of an
/// arbitrary original input string.
///
/// # Example
///
/// ```
/// use thoughtchain::{chain_key_from_storage_filename, chain_storage_filename, StorageAdapterKind};
///
/// let filename = chain_storage_filename("borganism-brain", StorageAdapterKind::Jsonl);
/// let chain_key = chain_key_from_storage_filename(&filename).unwrap();
///
/// assert_eq!(chain_key, "borganism-brain");
/// ```
pub fn chain_key_from_storage_filename(filename: &str) -> Option<String> {
    let (stem, extension) = filename.rsplit_once('.')?;
    if extension != StorageAdapterKind::Jsonl.file_extension()
        && extension != StorageAdapterKind::Binary.file_extension()
    {
        return None;
    }

    let (chain_key, fingerprint) = stem.rsplit_once('-')?;
    if fingerprint.len() != 16
        || !fingerprint
            .chars()
            .all(|character| character.is_ascii_hexdigit())
    {
        return None;
    }

    Some(chain_key.to_string())
}

fn append_memory_section(
    markdown: &mut String,
    title: &str,
    thoughts: &[&Thought],
    types: &[ThoughtType],
) {
    let items: Vec<&Thought> = thoughts
        .iter()
        .copied()
        .filter(|thought| types.contains(&thought.thought_type))
        .collect();
    if items.is_empty() {
        return;
    }

    markdown.push_str(&format!("## {title}\n\n"));
    for thought in items {
        markdown.push_str(&format!(
            "- [#{}] {:?}: {}",
            thought.index, thought.thought_type, thought.content
        ));
        let mut metadata = Vec::new();
        metadata.push(format!("agent {}", agent_label(thought)));
        if thought.role != ThoughtRole::Memory {
            metadata.push(format!("role {:?}", thought.role));
        }
        metadata.push(format!("importance {:.2}", thought.importance));
        if let Some(confidence) = thought.confidence {
            metadata.push(format!("confidence {:.2}", confidence));
        }
        if !thought.tags.is_empty() {
            metadata.push(format!("tags {}", thought.tags.join(", ")));
        }
        if !metadata.is_empty() {
            markdown.push_str(&format!(" ({})", metadata.join("; ")));
        }
        markdown.push('\n');
    }
    markdown.push('\n');
}

fn contains_case_insensitive(values: &[String], needle: &str) -> bool {
    let needle = needle.to_lowercase();
    values
        .iter()
        .any(|value| value.to_lowercase() == needle || value.to_lowercase().contains(&needle))
}

fn equals_case_insensitive(value: &str, needle: &str) -> bool {
    value.eq_ignore_ascii_case(needle)
}

fn agent_label(thought: &Thought) -> String {
    let mut label =
        if thought.agent_name.trim().is_empty() || thought.agent_name == thought.agent_id {
            thought.agent_id.clone()
        } else {
            format!("{} [{}]", thought.agent_name, thought.agent_id)
        };

    if let Some(owner) = &thought.agent_owner {
        if !owner.trim().is_empty() {
            label.push_str(&format!(" owned by {}", owner));
        }
    }

    label
}

fn normalize_strings(values: Vec<String>) -> Vec<String> {
    let mut seen = HashSet::new();
    let mut normalized = Vec::new();
    for value in values {
        let trimmed = value.trim();
        if trimmed.is_empty() {
            continue;
        }
        let key = trimmed.to_lowercase();
        if seen.insert(key) {
            normalized.push(trimmed.to_string());
        }
    }
    normalized
}

fn validate_refs(thoughts: &[Thought], refs: &[u64]) -> io::Result<()> {
    for &reference in refs {
        if reference as usize >= thoughts.len() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("Thought reference {reference} does not exist"),
            ));
        }
    }
    Ok(())
}

fn dedupe_relations(relations: &mut Vec<ThoughtRelation>) {
    let mut seen = HashSet::new();
    relations.retain(|relation| seen.insert((relation.kind, relation.target_id)));
}

fn persist_jsonl_thought(file_path: &Path, thought: &Thought) -> io::Result<()> {
    if let Some(parent) = file_path.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(file_path)?;
    let json = serde_json::to_string(thought)
        .map_err(|error| io::Error::other(format!("Failed to serialize thought: {error}")))?;
    writeln!(file, "{json}")?;
    Ok(())
}

fn load_binary_thoughts(file_path: &Path) -> io::Result<Vec<Thought>> {
    if !file_path.exists() {
        return Ok(Vec::new());
    }

    let mut file = fs::File::open(file_path)?;
    let mut thoughts = Vec::new();

    loop {
        let mut length_bytes = [0_u8; 8];
        match file.read_exact(&mut length_bytes) {
            Ok(()) => {}
            Err(error) if error.kind() == io::ErrorKind::UnexpectedEof => break,
            Err(error) => return Err(error),
        }

        let length = u64::from_le_bytes(length_bytes) as usize;
        let mut payload = vec![0_u8; length];
        file.read_exact(&mut payload)?;
        let (thought, _bytes_read): (Thought, usize) =
            bincode::serde::decode_from_slice(&payload, bincode::config::standard()).map_err(
                |error| {
                    io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!("Failed to deserialize binary thought: {error}"),
                    )
                },
            )?;
        thoughts.push(thought);
    }

    Ok(thoughts)
}

fn persist_binary_thought(file_path: &Path, thought: &Thought) -> io::Result<()> {
    let payload = bincode::serde::encode_to_vec(thought, bincode::config::standard())
        .map_err(|error| io::Error::other(format!("Failed to serialize thought: {error}")))?;
    if let Some(parent) = file_path.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(file_path)?;
    file.write_all(&(payload.len() as u64).to_le_bytes())?;
    file.write_all(&payload)?;
    Ok(())
}

fn compute_thought_hash(thought: &Thought) -> String {
    #[derive(Serialize)]
    struct CanonicalThought<'a> {
        id: Uuid,
        index: u64,
        timestamp: &'a DateTime<Utc>,
        session_id: Option<Uuid>,
        agent_id: &'a str,
        agent_name: &'a str,
        agent_owner: Option<&'a str>,
        thought_type: ThoughtType,
        role: ThoughtRole,
        content: &'a str,
        confidence: Option<f32>,
        importance: f32,
        tags: &'a [String],
        concepts: &'a [String],
        refs: &'a [u64],
        relations: &'a [ThoughtRelation],
        prev_hash: &'a str,
    }

    let canonical = CanonicalThought {
        id: thought.id,
        index: thought.index,
        timestamp: &thought.timestamp,
        session_id: thought.session_id,
        agent_id: &thought.agent_id,
        agent_name: &thought.agent_name,
        agent_owner: thought.agent_owner.as_deref(),
        thought_type: thought.thought_type,
        role: thought.role,
        content: &thought.content,
        confidence: thought.confidence,
        importance: thought.importance,
        tags: &thought.tags,
        concepts: &thought.concepts,
        refs: &thought.refs,
        relations: &thought.relations,
        prev_hash: &thought.prev_hash,
    };

    let bytes = serde_json::to_vec(&canonical).unwrap_or_default();
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    format!("{:x}", hasher.finalize())
}
