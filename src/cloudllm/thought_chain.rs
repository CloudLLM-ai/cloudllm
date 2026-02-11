//! Persistent, hash-chained agent memory.
//!
//! [`ThoughtChain`] is an append-only log of agent findings, decisions, compressions,
//! and checkpoints. Each [`Thought`] is SHA-256 hash-chained to the previous entry and
//! can carry back-references (`refs`) to important ancestor thoughts, enabling
//! graph-based context resolution.
//!
//! Thoughts are persisted as newline-delimited JSON (`.jsonl`) — one [`Thought`] per
//! line, append-only. The filename is derived from the agent's identity fingerprint
//! for collision resistance.
//!
//! # Architecture
//!
//! ```text
//! ThoughtChain (.jsonl on disk)
//!   ├─ Thought #0  Finding      hash=abc1...   refs=[]
//!   ├─ Thought #1  Decision     hash=def2...   refs=[]      prev_hash=abc1...
//!   ├─ Thought #2  Finding      hash=789a...   refs=[]      prev_hash=def2...
//!   └─ Thought #3  Compression  hash=bcd3...   refs=[0, 2]  prev_hash=789a...
//!                                                 ↑
//!                              resolve_context(3) walks refs → returns [#0, #2, #3]
//! ```
//!
//! # Disk Format
//!
//! Each `.jsonl` file contains one JSON-serialized [`Thought`] per line:
//!
//! ```text
//! {"index":0,"timestamp":"2025-07-01T12:00:00Z","agent_id":"analyst","thought_type":"Finding","content":"Found X","refs":[],"prev_hash":"","hash":"abc1..."}
//! {"index":1,"timestamp":"2025-07-01T12:01:00Z","agent_id":"analyst","thought_type":"Decision","content":"Will do Y","refs":[0],"prev_hash":"abc1...","hash":"def2..."}
//! ```
//!
//! Files are append-only; corruption of earlier lines is detectable via
//! [`ThoughtChain::verify_integrity`].
//!
//! # Example
//!
//! ```rust,no_run
//! use cloudllm::thought_chain::{ThoughtChain, ThoughtType};
//! use std::path::PathBuf;
//!
//! # fn main() -> std::io::Result<()> {
//! let mut chain = ThoughtChain::open(
//!     &PathBuf::from("thought_chains"),
//!     "analyst",
//!     "Technical Analyst",
//!     Some("Cloud Architecture"),
//!     None,
//! )?;
//!
//! // Record a finding and a decision that references it
//! chain.append("analyst", ThoughtType::Finding, "Discovered memory leak in service X")?;
//! chain.append_with_refs("analyst", ThoughtType::Decision, "Will fix via pooling", vec![0])?;
//!
//! // Verify the hash chain is intact
//! assert!(chain.verify_integrity());
//!
//! // Resolve context for thought #1 — walks refs to include #0
//! let context = chain.resolve_context(1);
//! assert_eq!(context.len(), 2); // [#0, #1]
//! # Ok(())
//! # }
//! ```

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashSet;
use std::fs::{self, OpenOptions};
use std::io::{self, BufRead, BufReader, Write};
use std::path::PathBuf;

/// Classification of a thought entry.
///
/// Each variant represents a semantic category that helps strategies and
/// tooling understand the *purpose* of a thought without parsing its content.
///
/// # Example
///
/// ```rust
/// use cloudllm::thought_chain::ThoughtType;
///
/// let t = ThoughtType::Finding;
/// // ThoughtType is Serialize/Deserialize — round-trips through JSON cleanly.
/// let json = serde_json::to_string(&t).unwrap();
/// assert_eq!(json, "\"Finding\"");
/// ```
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum ThoughtType {
    /// A factual observation or discovery.
    Finding,
    /// A deliberate choice made by the agent.
    Decision,
    /// Signals that a specific task has been completed.
    TaskComplete,
    /// A snapshot of agent state for resumption.
    Checkpoint,
    /// An open question the agent needs answered.
    Question,
    /// The agent hands off work to another agent.
    Handoff,
    /// A compressed summary replacing older thoughts.
    Compression,
    /// A compression triggered during idle time.
    IdleCompression,
}

/// A single entry in a [`ThoughtChain`].
///
/// Each thought captures *what* an agent observed or decided, *when* it happened,
/// and *which prior thoughts* it depends on.  The `hash` and `prev_hash` fields
/// form a SHA-256 chain that makes post-hoc tampering detectable.
///
/// # Fields
///
/// | Field | Purpose |
/// |-------|---------|
/// | `index` | Zero-based position — monotonically increasing within a chain |
/// | `timestamp` | UTC wall-clock time when the thought was recorded |
/// | `agent_id` | Which agent produced it (useful in multi-agent chains) |
/// | `thought_type` | Semantic classification ([`ThoughtType`]) |
/// | `content` | Free-form text of the thought |
/// | `refs` | Indices of ancestor thoughts this one depends on |
/// | `prev_hash` | SHA-256 hex of the immediately preceding thought |
/// | `hash` | SHA-256 hex of this thought's canonical representation |
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Thought {
    /// Zero-based position in the chain.
    pub index: u64,
    /// When this thought was recorded.
    pub timestamp: DateTime<Utc>,
    /// Which agent produced this thought.
    pub agent_id: String,
    /// Classification of the thought.
    pub thought_type: ThoughtType,
    /// Free-form content of the thought.
    pub content: String,
    /// Back-references to important ancestor thought indices.
    pub refs: Vec<u64>,
    /// SHA-256 hex digest of the previous thought (empty string for the first entry).
    pub prev_hash: String,
    /// SHA-256 hex digest of this thought's canonical representation.
    pub hash: String,
}

/// Append-only, SHA-256 hash-chained, disk-persisted log of agent thoughts.
///
/// A `ThoughtChain` owns an in-memory `Vec<Thought>` mirrored to a `.jsonl`
/// file on disk.  New thoughts are appended atomically (one JSON line per
/// thought), and the SHA-256 hash chain ensures that any modification of
/// earlier entries is detectable via [`ThoughtChain::verify_integrity`].
///
/// Back-references (`refs`) allow individual thoughts to point at ancestors,
/// forming a DAG that [`ThoughtChain::resolve_context`] traverses to
/// reconstruct the minimal context needed for a given thought.
///
/// # Example
///
/// ```rust,no_run
/// use cloudllm::thought_chain::{ThoughtChain, ThoughtType};
/// use std::path::PathBuf;
///
/// # fn main() -> std::io::Result<()> {
/// let dir = PathBuf::from("/tmp/tc_example");
/// let mut chain = ThoughtChain::open(&dir, "bot", "Bot", None, None)?;
///
/// chain.append("bot", ThoughtType::Finding, "Service latency spiked")?;
/// chain.append("bot", ThoughtType::Decision, "Scale horizontally")?;
///
/// assert_eq!(chain.thoughts().len(), 2);
/// assert!(chain.verify_integrity());
/// # Ok(())
/// # }
/// ```
pub struct ThoughtChain {
    thoughts: Vec<Thought>,
    file_path: PathBuf,
    auto_flush: bool,
}

impl ThoughtChain {
    /// Open an existing chain or create a new one.
    ///
    /// The filename is derived from a SHA-256 fingerprint of the agent's identity
    /// attributes (via [`chain_filename`]), providing collision resistance across
    /// agents with the same id but different configurations.
    ///
    /// If the `.jsonl` file already exists, all previously persisted thoughts are
    /// loaded back into memory and the hash chain is ready for further appending.
    /// If the file does not exist, a new empty chain is created.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use cloudllm::thought_chain::{ThoughtChain, ThoughtType};
    /// use std::path::PathBuf;
    ///
    /// # fn main() -> std::io::Result<()> {
    /// // First run — creates the .jsonl file
    /// let dir = PathBuf::from("/tmp/my_chains");
    /// let mut chain = ThoughtChain::open(
    ///     &dir,
    ///     "researcher",
    ///     "Deep Researcher",
    ///     Some("Machine Learning"),
    ///     Some("Thorough and methodical"),
    /// )?;
    /// chain.append("researcher", ThoughtType::Finding, "Found anomaly in dataset")?;
    /// drop(chain);
    ///
    /// // Second run — loads the existing chain from disk
    /// let chain = ThoughtChain::open(
    ///     &dir,
    ///     "researcher",
    ///     "Deep Researcher",
    ///     Some("Machine Learning"),
    ///     Some("Thorough and methodical"),
    /// )?;
    /// assert_eq!(chain.thoughts().len(), 1);
    /// # Ok(())
    /// # }
    /// ```
    pub fn open(
        chain_dir: &PathBuf,
        agent_id: &str,
        agent_name: &str,
        expertise: Option<&str>,
        personality: Option<&str>,
    ) -> io::Result<Self> {
        fs::create_dir_all(chain_dir)?;

        let filename = chain_filename(agent_id, agent_name, expertise, personality);
        let file_path = chain_dir.join(filename);

        let thoughts = if file_path.exists() {
            let file = fs::File::open(&file_path)?;
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
                        format!("Failed to parse thought: {}", e),
                    )
                })?;
                entries.push(thought);
            }
            entries
        } else {
            Vec::new()
        };

        Ok(Self {
            thoughts,
            file_path,
            auto_flush: true,
        })
    }

    /// Append a thought with no back-references.
    ///
    /// This is the most common way to record a thought — it appends a new entry,
    /// computes its SHA-256 hash chained to the previous entry, and (when
    /// `auto_flush` is enabled) writes it to disk immediately.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use cloudllm::thought_chain::{ThoughtChain, ThoughtType};
    /// # fn main() -> std::io::Result<()> {
    /// # let dir = std::path::PathBuf::from("/tmp/tc_append");
    /// let mut chain = ThoughtChain::open(&dir, "a1", "Agent", None, None)?;
    ///
    /// let thought = chain.append("a1", ThoughtType::Finding, "CPU usage at 95%")?;
    /// assert_eq!(thought.index, 0);
    /// assert!(thought.refs.is_empty());
    /// # Ok(())
    /// # }
    /// ```
    pub fn append(
        &mut self,
        agent_id: &str,
        thought_type: ThoughtType,
        content: &str,
    ) -> io::Result<&Thought> {
        self.append_with_refs(agent_id, thought_type, content, vec![])
    }

    /// Append a thought with explicit back-references to ancestor indices.
    ///
    /// Back-references create edges in a DAG that [`ThoughtChain::resolve_context`]
    /// can traverse.  A `Compression` thought typically references the findings
    /// it summarizes so that the resolved context graph remains self-contained.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use cloudllm::thought_chain::{ThoughtChain, ThoughtType};
    /// # fn main() -> std::io::Result<()> {
    /// # let dir = std::path::PathBuf::from("/tmp/tc_refs");
    /// let mut chain = ThoughtChain::open(&dir, "a1", "Agent", None, None)?;
    ///
    /// chain.append("a1", ThoughtType::Finding, "Finding A")?;     // #0
    /// chain.append("a1", ThoughtType::Finding, "Finding B")?;     // #1
    /// chain.append("a1", ThoughtType::Decision, "Unrelated")?;    // #2
    ///
    /// // Compression summarizes findings #0 and #1
    /// let compression = chain.append_with_refs(
    ///     "a1",
    ///     ThoughtType::Compression,
    ///     "Summary of A and B",
    ///     vec![0, 1],
    /// )?;
    /// assert_eq!(compression.refs, vec![0, 1]);
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
        let index = self.thoughts.len() as u64;
        let prev_hash = self
            .thoughts
            .last()
            .map(|t| t.hash.clone())
            .unwrap_or_default();

        let timestamp = Utc::now();
        let hash = compute_thought_hash(
            index,
            &timestamp,
            agent_id,
            &thought_type,
            content,
            &refs,
            &prev_hash,
        );

        let thought = Thought {
            index,
            timestamp,
            agent_id: agent_id.to_string(),
            thought_type,
            content: content.to_string(),
            refs,
            prev_hash,
            hash,
        };

        if self.auto_flush {
            let mut file = OpenOptions::new()
                .create(true)
                .append(true)
                .open(&self.file_path)?;
            let json = serde_json::to_string(&thought)
                .map_err(|e| io::Error::other(format!("Failed to serialize thought: {}", e)))?;
            writeln!(file, "{}", json)?;
        }

        self.thoughts.push(thought);
        Ok(self.thoughts.last().unwrap())
    }

    /// Walk the chain and verify that every hash matches its recomputed value.
    ///
    /// Returns `true` if every thought's `prev_hash` matches the preceding
    /// thought's `hash`, and every thought's `hash` matches the SHA-256 digest
    /// recomputed from its canonical fields.  Returns `false` on the first
    /// mismatch.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use cloudllm::thought_chain::{ThoughtChain, ThoughtType};
    /// # fn main() -> std::io::Result<()> {
    /// # let dir = std::path::PathBuf::from("/tmp/tc_integrity");
    /// let mut chain = ThoughtChain::open(&dir, "a1", "Agent", None, None)?;
    /// chain.append("a1", ThoughtType::Finding, "Entry 1")?;
    /// chain.append("a1", ThoughtType::Finding, "Entry 2")?;
    ///
    /// assert!(chain.verify_integrity());
    /// # Ok(())
    /// # }
    /// ```
    pub fn verify_integrity(&self) -> bool {
        let mut prev_hash = String::new();
        for thought in &self.thoughts {
            if thought.prev_hash != prev_hash {
                return false;
            }
            let expected = compute_thought_hash(
                thought.index,
                &thought.timestamp,
                &thought.agent_id,
                &thought.thought_type,
                &thought.content,
                &thought.refs,
                &thought.prev_hash,
            );
            if thought.hash != expected {
                return false;
            }
            prev_hash = thought.hash.clone();
        }
        true
    }

    /// Resolve the context graph for a target thought via DFS through refs.
    ///
    /// Returns a deduplicated, chronologically sorted slice of thoughts that
    /// the target thought transitively depends on, including the target itself.
    /// This is the primary mechanism for reconstructing an agent's reasoning
    /// history from a [`Compression`](ThoughtType::Compression) node without
    /// replaying the entire chain.
    ///
    /// # Algorithm
    ///
    /// 1. Start with `target_index` on the stack.
    /// 2. Pop an index, mark it visited, push all its `refs` that haven't been visited.
    /// 3. Repeat until the stack is empty.
    /// 4. Return visited thoughts sorted by index (chronological order).
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use cloudllm::thought_chain::{ThoughtChain, ThoughtType};
    /// # fn main() -> std::io::Result<()> {
    /// # let dir = std::path::PathBuf::from("/tmp/tc_resolve");
    /// let mut chain = ThoughtChain::open(&dir, "a1", "Agent", None, None)?;
    ///
    /// chain.append("a1", ThoughtType::Finding, "Base finding")?;        // #0
    /// chain.append("a1", ThoughtType::Decision, "Unrelated")?;          // #1
    /// chain.append("a1", ThoughtType::Finding, "Important discovery")?; // #2
    /// chain.append_with_refs(
    ///     "a1", ThoughtType::Compression, "Summary", vec![0, 2],
    /// )?;                                                                // #3
    ///
    /// let resolved = chain.resolve_context(3);
    /// let indices: Vec<u64> = resolved.iter().map(|t| t.index).collect();
    /// assert_eq!(indices, vec![0, 2, 3]); // skips #1 because it's not referenced
    /// # Ok(())
    /// # }
    /// ```
    pub fn resolve_context(&self, target_index: u64) -> Vec<&Thought> {
        let mut visited = HashSet::new();
        let mut stack = vec![target_index];

        while let Some(idx) = stack.pop() {
            if visited.contains(&idx) {
                continue;
            }
            if let Some(thought) = self.thoughts.get(idx as usize) {
                visited.insert(idx);
                for &r in &thought.refs {
                    if !visited.contains(&r) {
                        stack.push(r);
                    }
                }
            }
        }

        let mut result: Vec<&Thought> = self
            .thoughts
            .iter()
            .filter(|t| visited.contains(&t.index))
            .collect();
        result.sort_by_key(|t| t.index);
        result
    }

    /// Render the resolved context graph as a layered bootstrap prompt.
    ///
    /// Calls [`resolve_context`](ThoughtChain::resolve_context) for `target_index`
    /// and formats the result as a human-readable prompt wrapped in
    /// `=== RESTORED CONTEXT ===` markers.  This prompt is typically injected
    /// into a fresh [`LLMSession`](crate::LLMSession) after a context collapse
    /// so the agent can resume with its critical reasoning intact.
    ///
    /// Returns an empty string if no thoughts are resolved.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use cloudllm::thought_chain::{ThoughtChain, ThoughtType};
    /// # fn main() -> std::io::Result<()> {
    /// # let dir = std::path::PathBuf::from("/tmp/tc_boot");
    /// let mut chain = ThoughtChain::open(&dir, "a1", "Agent", None, None)?;
    /// chain.append("a1", ThoughtType::Finding, "Key insight")?;
    /// chain.append_with_refs(
    ///     "a1", ThoughtType::Compression, "Compressed view", vec![0],
    /// )?;
    ///
    /// let prompt = chain.to_bootstrap_prompt(1);
    /// assert!(prompt.contains("RESTORED CONTEXT"));
    /// assert!(prompt.contains("Key insight"));
    /// assert!(prompt.contains("Compressed view"));
    /// # Ok(())
    /// # }
    /// ```
    pub fn to_bootstrap_prompt(&self, target_index: u64) -> String {
        let resolved = self.resolve_context(target_index);
        if resolved.is_empty() {
            return String::new();
        }

        let mut prompt = String::from("=== RESTORED CONTEXT (from ThoughtChain) ===\n\n");
        for thought in &resolved {
            prompt.push_str(&format!(
                "[#{}] {:?} ({}): {}\n",
                thought.index, thought.thought_type, thought.agent_id, thought.content
            ));
            if !thought.refs.is_empty() {
                prompt.push_str(&format!("  refs: {:?}\n", thought.refs));
            }
        }
        prompt.push_str("\n=== END RESTORED CONTEXT ===\n");
        prompt
    }

    /// Render the last `N` thoughts as a catch-up prompt.
    ///
    /// Unlike [`to_bootstrap_prompt`](ThoughtChain::to_bootstrap_prompt) which
    /// walks the reference graph, this method simply takes the tail of the chain
    /// and formats it.  Useful for giving a newly forked agent a quick summary
    /// of recent activity without full graph resolution.
    ///
    /// Returns an empty string if the chain is empty.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use cloudllm::thought_chain::{ThoughtChain, ThoughtType};
    /// # fn main() -> std::io::Result<()> {
    /// # let dir = std::path::PathBuf::from("/tmp/tc_catchup");
    /// let mut chain = ThoughtChain::open(&dir, "a1", "Agent", None, None)?;
    /// chain.append("a1", ThoughtType::Finding, "Old finding")?;
    /// chain.append("a1", ThoughtType::Decision, "Recent decision")?;
    /// chain.append("a1", ThoughtType::TaskComplete, "Done")?;
    ///
    /// let prompt = chain.to_catchup_prompt(2);
    /// assert!(prompt.contains("RECENT CONTEXT"));
    /// assert!(!prompt.contains("Old finding"));   // only last 2
    /// assert!(prompt.contains("Recent decision"));
    /// assert!(prompt.contains("Done"));
    /// # Ok(())
    /// # }
    /// ```
    pub fn to_catchup_prompt(&self, last_n: usize) -> String {
        let start = self.thoughts.len().saturating_sub(last_n);
        let tail = &self.thoughts[start..];
        if tail.is_empty() {
            return String::new();
        }

        let mut prompt = String::from("=== RECENT CONTEXT ===\n\n");
        for thought in tail {
            prompt.push_str(&format!(
                "[#{}] {:?} ({}): {}\n",
                thought.index, thought.thought_type, thought.agent_id, thought.content
            ));
        }
        prompt.push_str("\n=== END RECENT CONTEXT ===\n");
        prompt
    }

    /// Return all thoughts in the chain.
    ///
    /// The returned slice is in chronological order (sorted by `index`).
    pub fn thoughts(&self) -> &[Thought] {
        &self.thoughts
    }

    /// Return the file path used for persistence.
    ///
    /// The path is derived from the agent identity fingerprint during
    /// [`open`](ThoughtChain::open) and does not change for the lifetime of
    /// the chain.
    pub fn file_path(&self) -> &PathBuf {
        &self.file_path
    }

    /// Control whether appended thoughts are immediately flushed to disk.
    ///
    /// When `true` (the default), each [`append`](ThoughtChain::append) or
    /// [`append_with_refs`](ThoughtChain::append_with_refs) call writes one
    /// JSON line immediately.  Set to `false` for batch operations where you
    /// want to control when I/O happens — but note that unflushed thoughts
    /// will be lost if the process crashes.
    pub fn set_auto_flush(&mut self, auto_flush: bool) {
        self.auto_flush = auto_flush;
    }
}

/// Compute the SHA-256 hex digest for a thought's canonical fields.
///
/// The canonical representation is: `index|timestamp|agent_id|thought_type|content|refs|prev_hash`
/// joined by pipe characters.  This deterministic format ensures that any
/// change to any field produces a different hash.
fn compute_thought_hash(
    index: u64,
    timestamp: &DateTime<Utc>,
    agent_id: &str,
    thought_type: &ThoughtType,
    content: &str,
    refs: &[u64],
    prev_hash: &str,
) -> String {
    let type_str = serde_json::to_string(thought_type).unwrap_or_default();
    let refs_str = refs
        .iter()
        .map(|r| r.to_string())
        .collect::<Vec<_>>()
        .join(",");
    let canonical = format!(
        "{}|{}|{}|{}|{}|{}|{}",
        index,
        timestamp.to_rfc3339(),
        agent_id,
        type_str,
        content,
        refs_str,
        prev_hash
    );
    let mut hasher = Sha256::new();
    hasher.update(canonical.as_bytes());
    format!("{:x}", hasher.finalize())
}

/// Derive a collision-resistant filename from the agent's identity attributes.
///
/// Format: `{safe_id}-{sha256(id|name|expertise|personality)[..16]}.jsonl`
///
/// The agent id is sanitized for filesystem safety (non-alphanumeric characters
/// other than `-` and `_` are replaced with `_`).  The 16-hex-char fingerprint
/// is derived from the full identity string `"id|name|expertise|personality"`,
/// giving ~2^64 collision resistance.
///
/// # Example
///
/// ```rust
/// use cloudllm::thought_chain::chain_filename;
///
/// let f1 = chain_filename("agent-1", "Analyst", Some("ML"), None);
/// let f2 = chain_filename("agent-1", "Analyst", Some("ML"), None);
/// assert_eq!(f1, f2); // deterministic
///
/// let f3 = chain_filename("agent-1", "Analyst", Some("NLP"), None);
/// assert_ne!(f1, f3); // different expertise → different filename
/// ```
pub fn chain_filename(
    agent_id: &str,
    agent_name: &str,
    expertise: Option<&str>,
    personality: Option<&str>,
) -> String {
    let identity = format!(
        "{}|{}|{}|{}",
        agent_id,
        agent_name,
        expertise.unwrap_or(""),
        personality.unwrap_or("")
    );
    let mut hasher = Sha256::new();
    hasher.update(identity.as_bytes());
    let digest = format!("{:x}", hasher.finalize());
    let fingerprint = &digest[..16];

    // Sanitize agent_id for filesystem safety
    let safe_id: String = agent_id
        .chars()
        .map(|c| {
            if c.is_alphanumeric() || c == '-' || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect();

    format!("{}-{}.jsonl", safe_id, fingerprint)
}
