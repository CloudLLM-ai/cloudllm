//! Pluggable strategies for handling context window exhaustion.
//!
//! The [`ContextStrategy`] trait defines how an agent should respond when its
//! conversation history approaches the token budget. Three implementations are
//! provided out of the box:
//!
//! - [`TrimStrategy`] (default): relies on [`LLMSession`]'s built-in oldest-first
//!   trimming — `compact()` is a no-op.
//! - [`SelfCompressionStrategy`]: asks the backing LLM to write a structured
//!   save-file that is persisted to the agent's [`ThoughtChain`] and then
//!   injected back as a bootstrap prompt after clearing the session.
//! - [`NoveltyAwareStrategy`]: wraps another strategy and uses an entropy
//!   heuristic (unique n-gram ratio) to avoid compressing when the conversation
//!   is still producing novel content.
//!
//! # Architecture
//!
//! ```text
//! Agent
//!   ├─ LLMSession (tracks estimated_history_tokens / max_tokens)
//!   └─ ContextStrategy
//!        ├─ should_compact(&session) → bool    // "is it time?"
//!        └─ compact(&mut session, chain, id)   // "do it"
//! ```
//!
//! # Wiring a Strategy into an Agent
//!
//! ```rust,no_run
//! use cloudllm::Agent;
//! use cloudllm::context_strategy::SelfCompressionStrategy;
//! use cloudllm::clients::openai::OpenAIClient;
//! use std::sync::Arc;
//!
//! let agent = Agent::new(
//!     "analyst", "Analyst",
//!     Arc::new(OpenAIClient::new_with_model_string("key", "gpt-4o")),
//! )
//! .context_collapse_strategy(Box::new(SelfCompressionStrategy::default()));
//! ```

use crate::cloudllm::llm_session::LLMSession;
use crate::cloudllm::thought_chain::{ThoughtChain, ThoughtType};
use crate::client_wrapper::Role;
use async_trait::async_trait;
use std::collections::HashSet;
use std::error::Error;
use std::sync::Arc;
use tokio::sync::RwLock;

/// Trait for pluggable context-window management strategies.
///
/// Implementors decide **when** to compact ([`should_compact`](ContextStrategy::should_compact))
/// and **how** to compact ([`compact`](ContextStrategy::compact)).  The agent
/// calls these at appropriate points during generation.
///
/// # Implementing a Custom Strategy
///
/// ```rust,no_run
/// use cloudllm::context_strategy::ContextStrategy;
/// use cloudllm::LLMSession;
/// use cloudllm::thought_chain::ThoughtChain;
/// use async_trait::async_trait;
/// use std::sync::Arc;
/// use tokio::sync::RwLock;
///
/// struct MyStrategy;
///
/// #[async_trait]
/// impl ContextStrategy for MyStrategy {
///     fn should_compact(&self, session: &LLMSession) -> bool {
///         let ratio = session.estimated_history_tokens() as f64
///             / session.get_max_tokens() as f64;
///         ratio > 0.75
///     }
///
///     async fn compact(
///         &self,
///         session: &mut LLMSession,
///         _chain: &Option<Arc<RwLock<ThoughtChain>>>,
///         _agent_id: &str,
///     ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
///         session.clear_history();
///         Ok(())
///     }
///
///     fn name(&self) -> &str { "MyStrategy" }
/// }
/// ```
#[async_trait]
pub trait ContextStrategy: Send + Sync {
    /// Return `true` when the session's history is large enough to warrant
    /// compaction.
    fn should_compact(&self, session: &LLMSession) -> bool;

    /// Perform the actual compaction.
    ///
    /// Strategies that need the [`ThoughtChain`] (e.g. [`SelfCompressionStrategy`])
    /// use it; strategies that don't (e.g. [`TrimStrategy`]) ignore it.
    async fn compact(
        &self,
        session: &mut LLMSession,
        thought_chain: &Option<Arc<RwLock<ThoughtChain>>>,
        agent_id: &str,
    ) -> Result<(), Box<dyn Error + Send + Sync>>;

    /// Human-readable name for logging and diagnostics.
    fn name(&self) -> &str;
}

// ---------------------------------------------------------------------------
// TrimStrategy
// ---------------------------------------------------------------------------

/// Default strategy: delegates entirely to [`LLMSession`]'s built-in
/// oldest-first trimming. [`compact`](ContextStrategy::compact) is a no-op.
///
/// This is the simplest and cheapest strategy — it never makes an extra LLM
/// call.  When [`should_compact`](ContextStrategy::should_compact) returns
/// `true`, the agent knows that `LLMSession::send_message` will
/// automatically prune oldest messages to stay within the token budget.
///
/// # Example
///
/// ```rust
/// use cloudllm::context_strategy::TrimStrategy;
///
/// // Default threshold is 0.85
/// let strategy = TrimStrategy::default();
/// assert!((strategy.threshold - 0.85).abs() < f64::EPSILON);
///
/// // Custom threshold
/// let aggressive = TrimStrategy::new(0.60);
/// assert!((aggressive.threshold - 0.60).abs() < f64::EPSILON);
/// ```
pub struct TrimStrategy {
    /// Ratio of `estimated_history_tokens / max_tokens` above which
    /// `should_compact` returns `true`. Default: `0.85`.
    pub threshold: f64,
}

impl Default for TrimStrategy {
    /// Create a `TrimStrategy` with a threshold of `0.85`.
    fn default() -> Self {
        Self { threshold: 0.85 }
    }
}

impl TrimStrategy {
    /// Create a `TrimStrategy` with a custom threshold.
    ///
    /// The `threshold` is a ratio (0.0 – 1.0) of estimated history tokens to
    /// the session's max tokens.  Lower values trigger earlier.
    pub fn new(threshold: f64) -> Self {
        Self { threshold }
    }
}

#[async_trait]
impl ContextStrategy for TrimStrategy {
    fn should_compact(&self, session: &LLMSession) -> bool {
        let max = session.get_max_tokens();
        if max == 0 {
            return false;
        }
        let ratio = session.estimated_history_tokens() as f64 / max as f64;
        ratio > self.threshold
    }

    async fn compact(
        &self,
        _session: &mut LLMSession,
        _thought_chain: &Option<Arc<RwLock<ThoughtChain>>>,
        _agent_id: &str,
    ) -> Result<(), Box<dyn Error + Send + Sync>> {
        // No-op — LLMSession's built-in trimming handles it.
        Ok(())
    }

    fn name(&self) -> &str {
        "TrimStrategy"
    }
}

// ---------------------------------------------------------------------------
// SelfCompressionStrategy
// ---------------------------------------------------------------------------

/// "LLM writes its own save file."
///
/// When token pressure exceeds the threshold the strategy:
///
/// 1. Sends a compression prompt asking the LLM to produce a structured summary
///    covering key findings, decisions, current state, open questions, and next steps.
/// 2. Parses `REFS:` lines from the response via [`parse_refs`].
/// 3. Appends a [`Compression`](crate::thought_chain::ThoughtType::Compression) thought
///    to the [`ThoughtChain`] with back-references.
/// 4. Clears the session history.
/// 5. Injects the resolved bootstrap prompt from the ThoughtChain into the fresh session.
///
/// If no [`ThoughtChain`] is attached, the summary is injected directly as a
/// system message (no disk persistence).
///
/// # Example
///
/// ```rust,no_run
/// use cloudllm::Agent;
/// use cloudllm::context_strategy::SelfCompressionStrategy;
/// use cloudllm::thought_chain::ThoughtChain;
/// use cloudllm::clients::openai::OpenAIClient;
/// use std::sync::Arc;
/// use std::path::PathBuf;
/// use tokio::sync::RwLock;
///
/// # async {
/// let chain = Arc::new(RwLock::new(
///     ThoughtChain::open(&PathBuf::from("/tmp/chains"), "a1", "Agent", None, None).unwrap()
/// ));
///
/// let agent = Agent::new(
///     "a1", "Agent",
///     Arc::new(OpenAIClient::new_with_model_string("key", "gpt-4o")),
/// )
/// .with_thought_chain(chain)
/// .context_collapse_strategy(Box::new(SelfCompressionStrategy::default()));
/// # };
/// ```
pub struct SelfCompressionStrategy {
    /// Token-pressure ratio above which compaction triggers. Default: `0.80`.
    pub threshold: f64,
}

impl Default for SelfCompressionStrategy {
    /// Create a `SelfCompressionStrategy` with a threshold of `0.80`.
    fn default() -> Self {
        Self { threshold: 0.80 }
    }
}

impl SelfCompressionStrategy {
    /// Create a `SelfCompressionStrategy` with a custom threshold.
    ///
    /// The `threshold` is a ratio (0.0 – 1.0) of estimated history tokens to
    /// the session's max tokens.
    pub fn new(threshold: f64) -> Self {
        Self { threshold }
    }
}

#[async_trait]
impl ContextStrategy for SelfCompressionStrategy {
    fn should_compact(&self, session: &LLMSession) -> bool {
        let max = session.get_max_tokens();
        if max == 0 {
            return false;
        }
        let ratio = session.estimated_history_tokens() as f64 / max as f64;
        ratio > self.threshold
    }

    async fn compact(
        &self,
        session: &mut LLMSession,
        thought_chain: &Option<Arc<RwLock<ThoughtChain>>>,
        agent_id: &str,
    ) -> Result<(), Box<dyn Error + Send + Sync>> {
        let compression_prompt = "\
You are compressing your conversation memory into a structured save file. \
Write a concise summary covering:\n\
1. Key Findings\n\
2. Decisions Made\n\
3. Current Task State\n\
4. Open Questions\n\
5. Next Steps\n\
6. Essential Prior Thoughts (reference by index if available)\n\n\
If you reference prior thought indices, include a line: REFS: 150, 200\n\
Be concise but preserve all critical information.";

        let response = session
            .send_message(
                Role::User,
                compression_prompt.to_string(),
                None,
                None,
            )
            .await
            .map_err(|e| -> Box<dyn Error + Send + Sync> {
                Box::new(std::io::Error::other(e.to_string()))
            })?;

        let summary = response.content.to_string();
        let refs = parse_refs(&summary);

        // Persist to ThoughtChain if available
        if let Some(chain) = thought_chain {
            let mut chain = chain.write().await;
            chain.append_with_refs(agent_id, ThoughtType::Compression, &summary, refs)?;

            // Clear session and inject bootstrap
            session.clear_history();
            let last_idx = chain.thoughts().last().map(|t| t.index).unwrap_or(0);
            let bootstrap = chain.to_bootstrap_prompt(last_idx);
            if !bootstrap.is_empty() {
                session.inject_message(Role::System, bootstrap);
            }
        } else {
            // Without ThoughtChain, just clear and inject the summary
            session.clear_history();
            session.inject_message(Role::System, summary);
        }

        Ok(())
    }

    fn name(&self) -> &str {
        "SelfCompressionStrategy"
    }
}

// ---------------------------------------------------------------------------
// NoveltyAwareStrategy
// ---------------------------------------------------------------------------

/// Entropy-heuristic wrapper that only triggers compression when the
/// conversation has low novelty.
///
/// At high token pressure (above `high_threshold`), compression always fires.
/// At moderate pressure (between `moderate_threshold` and `high_threshold`),
/// it only fires when the unique bigram ratio between recent messages and
/// prior history is below `novelty_threshold` — meaning the conversation
/// is mostly rehashing old content and compressing won't lose much
/// information.
///
/// The actual compression work is delegated to an inner [`ContextStrategy`],
/// typically a [`SelfCompressionStrategy`].
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
///         .with_thresholds(0.85, 0.65, 0.25),
/// ));
/// ```
pub struct NoveltyAwareStrategy {
    /// High pressure threshold — always compress above this. Default: `0.90`.
    pub high_threshold: f64,
    /// Moderate pressure threshold. Default: `0.70`.
    pub moderate_threshold: f64,
    /// Minimum novelty ratio to *skip* compression at moderate pressure. Default: `0.30`.
    pub novelty_threshold: f64,
    /// Number of recent messages to consider for novelty estimation. Default: `4`.
    pub recent_window: usize,
    /// Inner strategy that performs the actual compression.
    pub inner: Box<dyn ContextStrategy>,
}

impl NoveltyAwareStrategy {
    /// Create a `NoveltyAwareStrategy` wrapping the given inner strategy.
    ///
    /// Uses default thresholds: `high=0.90`, `moderate=0.70`, `novelty=0.30`,
    /// `recent_window=4`.
    pub fn new(inner: Box<dyn ContextStrategy>) -> Self {
        Self {
            high_threshold: 0.90,
            moderate_threshold: 0.70,
            novelty_threshold: 0.30,
            recent_window: 4,
            inner,
        }
    }

    /// Override the default thresholds (builder pattern).
    ///
    /// - `high`: token pressure ratio above which compression always fires.
    /// - `moderate`: token pressure ratio above which novelty is checked.
    /// - `novelty`: bigram novelty ratio below which compression fires at moderate pressure.
    pub fn with_thresholds(
        mut self,
        high: f64,
        moderate: f64,
        novelty: f64,
    ) -> Self {
        self.high_threshold = high;
        self.moderate_threshold = moderate;
        self.novelty_threshold = novelty;
        self
    }

    /// Estimate novelty as the ratio of unique bigrams in the recent window
    /// that do NOT appear in the prior history.
    ///
    /// Returns `1.0` (fully novel) when there's insufficient history to compare.
    fn estimate_novelty(&self, session: &LLMSession) -> f64 {
        let history = session.get_conversation_history();
        if history.len() < 2 {
            return 1.0; // not enough data — assume novel
        }

        let split = history.len().saturating_sub(self.recent_window);
        let prior = &history[..split];
        let recent = &history[split..];

        let prior_ngrams = extract_bigrams_from_messages(prior);
        let recent_ngrams = extract_bigrams_from_messages(recent);

        if recent_ngrams.is_empty() {
            return 1.0;
        }

        let novel_count = recent_ngrams
            .iter()
            .filter(|ng| !prior_ngrams.contains(*ng))
            .count();

        novel_count as f64 / recent_ngrams.len() as f64
    }
}

#[async_trait]
impl ContextStrategy for NoveltyAwareStrategy {
    fn should_compact(&self, session: &LLMSession) -> bool {
        let max = session.get_max_tokens();
        if max == 0 {
            return false;
        }
        let ratio = session.estimated_history_tokens() as f64 / max as f64;

        if ratio > self.high_threshold {
            return true;
        }

        if ratio > self.moderate_threshold {
            let novelty = self.estimate_novelty(session);
            return novelty < self.novelty_threshold;
        }

        false
    }

    async fn compact(
        &self,
        session: &mut LLMSession,
        thought_chain: &Option<Arc<RwLock<ThoughtChain>>>,
        agent_id: &str,
    ) -> Result<(), Box<dyn Error + Send + Sync>> {
        self.inner.compact(session, thought_chain, agent_id).await
    }

    fn name(&self) -> &str {
        "NoveltyAwareStrategy"
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Parse `REFS: 150, 200` lines from a compression response.
///
/// Scans each line of `content` for a `REFS:` prefix and parses the
/// comma-separated integers that follow.  Returns the first match found,
/// or an empty vec if no `REFS:` line is present.
///
/// This is used by [`SelfCompressionStrategy`] to extract back-references
/// from the LLM's compression output.
///
/// # Example
///
/// ```rust
/// use cloudllm::context_strategy::parse_refs;
///
/// assert_eq!(parse_refs("Summary text\nREFS: 10, 25, 42\nMore text"), vec![10, 25, 42]);
/// assert_eq!(parse_refs("No refs here"), Vec::<u64>::new());
/// assert_eq!(parse_refs("REFS: 0"), vec![0]);
/// assert_eq!(parse_refs("REFS: bad, 5, also_bad"), vec![5]); // non-numeric skipped
/// ```
pub fn parse_refs(content: &str) -> Vec<u64> {
    for line in content.lines() {
        let trimmed = line.trim();
        if let Some(rest) = trimmed.strip_prefix("REFS:") {
            return rest
                .split(',')
                .filter_map(|s| s.trim().parse::<u64>().ok())
                .collect();
        }
    }
    vec![]
}

/// Extract unique bigrams (pairs of consecutive words) from message contents.
///
/// All words are lowercased before forming bigrams to make the comparison
/// case-insensitive.  Used by [`NoveltyAwareStrategy::estimate_novelty`].
fn extract_bigrams_from_messages(
    messages: &[crate::client_wrapper::Message],
) -> HashSet<String> {
    let mut bigrams = HashSet::new();
    for msg in messages {
        let words: Vec<&str> = msg.content.split_whitespace().collect();
        for pair in words.windows(2) {
            bigrams.insert(format!("{} {}", pair[0].to_lowercase(), pair[1].to_lowercase()));
        }
    }
    bigrams
}
