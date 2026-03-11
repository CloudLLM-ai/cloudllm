use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use thoughtchain::{
    chain_filename, chain_key_from_storage_filename, chain_storage_filename,
    BinaryStorageAdapter, StorageAdapter, StorageAdapterKind, Thought, ThoughtChain, ThoughtInput,
    ThoughtQuery, ThoughtRelation, ThoughtRelationKind, ThoughtRole, ThoughtType,
};
use uuid::Uuid;

static TEST_COUNTER: AtomicU64 = AtomicU64::new(0);

fn unique_chain_dir() -> PathBuf {
    let n = TEST_COUNTER.fetch_add(1, Ordering::SeqCst);
    let dir = std::env::temp_dir().join(format!("thoughtchain_test_{}_{}", std::process::id(), n));
    let _ = std::fs::remove_dir_all(&dir);
    dir
}

#[test]
fn append_and_reload_preserves_semantic_metadata() {
    let dir = unique_chain_dir();
    let session_id = Uuid::new_v4();

    {
        let mut chain = ThoughtChain::open(&dir, "agent1", "Analyst", Some("rust"), None).unwrap();
        chain
            .append_thought(
                "agent1",
                ThoughtInput::new(
                    ThoughtType::Insight,
                    "The bottleneck is cache invalidation.",
                )
                .with_session_id(session_id)
                .with_agent_name("Analyst")
                .with_agent_owner("cloudllm")
                .with_importance(0.95)
                .with_confidence(0.8)
                .with_tags(["performance", "cache"])
                .with_concepts(["latency", "cache invalidation"]),
            )
            .unwrap();
    }

    let chain = ThoughtChain::open(
        &dir,
        "agent1",
        "Analyst",
        Some("different"),
        Some("changed"),
    )
    .unwrap();
    assert_eq!(chain.thoughts().len(), 1);
    let thought = &chain.thoughts()[0];
    assert_eq!(thought.session_id, Some(session_id));
    assert_eq!(thought.thought_type, ThoughtType::Insight);
    assert_eq!(thought.role, ThoughtRole::Memory);
    assert_eq!(thought.agent_id, "agent1");
    assert_eq!(thought.agent_name, "Analyst");
    assert_eq!(thought.agent_owner.as_deref(), Some("cloudllm"));
    assert_eq!(thought.tags, vec!["performance", "cache"]);
    assert_eq!(thought.concepts, vec!["latency", "cache invalidation"]);

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn resolve_context_follows_refs_and_relations() {
    let dir = unique_chain_dir();
    let mut chain = ThoughtChain::open(&dir, "agent1", "Analyst", Some("data"), None).unwrap();

    let base_id = chain
        .append(
            "agent1",
            ThoughtType::FactLearned,
            "The dataset has 4 million rows.",
        )
        .unwrap()
        .id;
    chain
        .append_thought(
            "agent1",
            ThoughtInput::new(
                ThoughtType::Hypothesis,
                "Failures may come from stale partitions.",
            )
            .with_relations(vec![ThoughtRelation {
                kind: ThoughtRelationKind::DerivedFrom,
                target_id: base_id,
            }]),
        )
        .unwrap();
    chain
        .append_with_refs(
            "agent1",
            ThoughtType::Summary,
            "Important memory snapshot",
            vec![1],
        )
        .unwrap();

    let resolved = chain.resolve_context(2);
    let indices: Vec<u64> = resolved.iter().map(|thought| thought.index).collect();
    assert_eq!(indices, vec![0, 1, 2]);

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn query_filters_by_type_tag_and_text() {
    let dir = unique_chain_dir();
    let mut chain = ThoughtChain::open(&dir, "agent1", "Analyst", Some("memory"), None).unwrap();

    chain
        .append_thought(
            "agent1",
            ThoughtInput::new(
                ThoughtType::Constraint,
                "Memory must survive session resets.",
            )
            .with_importance(0.9)
            .with_tags(["durability"])
            .with_concepts(["persistence"]),
        )
        .unwrap();
    chain
        .append_thought(
            "agent1",
            ThoughtInput::new(ThoughtType::Idea, "Consider vector search later.")
                .with_importance(0.4)
                .with_tags(["retrieval"]),
        )
        .unwrap();

    let results = chain.query(
        &ThoughtQuery::new()
            .with_types(vec![ThoughtType::Constraint])
            .with_tags_any(["durability"])
            .with_text("survive"),
    );
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].thought_type, ThoughtType::Constraint);

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn query_filters_retrospectives_and_lesson_learned() {
    let dir = unique_chain_dir();
    let mut chain = ThoughtChain::open_with_key(&dir, "shared-project").unwrap();

    chain
        .append_thought(
            "astro",
            ThoughtInput::new(
                ThoughtType::LessonLearned,
                "When native tool calls return multiple tool invocations, resolve all of them before the next model round-trip.",
            )
            .with_agent_name("Astro")
            .with_role(ThoughtRole::Retrospective)
            .with_tags(["tools", "openai"])
            .with_concepts(["multi-tool call handling"]),
        )
        .unwrap();
    chain
        .append_thought(
            "astro",
            ThoughtInput::new(
                ThoughtType::Decision,
                "Keep the shared MCP runtime in the standalone mcp crate.",
            )
            .with_agent_name("Astro"),
        )
        .unwrap();

    let results = chain.query(
        &ThoughtQuery::new()
            .with_types(vec![ThoughtType::LessonLearned])
            .with_roles(vec![ThoughtRole::Retrospective]),
    );
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].thought_type, ThoughtType::LessonLearned);
    assert_eq!(results[0].role, ThoughtRole::Retrospective);

    let _ = std::fs::remove_dir_all(&dir);
}

#[derive(Clone)]
struct MemoryStorageAdapter {
    location: String,
    thoughts: Arc<Mutex<Vec<Thought>>>,
}

impl MemoryStorageAdapter {
    fn new(location: impl Into<String>) -> Self {
        Self {
            location: location.into(),
            thoughts: Arc::new(Mutex::new(Vec::new())),
        }
    }
}

impl StorageAdapter for MemoryStorageAdapter {
    fn load_thoughts(&self) -> std::io::Result<Vec<Thought>> {
        Ok(self.thoughts.lock().unwrap().clone())
    }

    fn append_thought(&self, thought: &Thought) -> std::io::Result<()> {
        self.thoughts.lock().unwrap().push(thought.clone());
        Ok(())
    }

    fn storage_location(&self) -> String {
        self.location.clone()
    }
}

#[test]
fn custom_storage_adapter_can_back_a_chain() {
    let adapter = MemoryStorageAdapter::new("memory://test");
    let mut chain = ThoughtChain::open_with_storage(Box::new(adapter.clone())).unwrap();
    chain
        .append(
            "agent1",
            ThoughtType::Checkpoint,
            "Adapter-backed thought persisted.",
        )
        .unwrap();
    assert_eq!(chain.storage_location(), "memory://test");

    let reloaded = ThoughtChain::open_with_storage(Box::new(adapter)).unwrap();
    assert_eq!(reloaded.thoughts().len(), 1);
    assert_eq!(
        reloaded.thoughts()[0].content,
        "Adapter-backed thought persisted."
    );
}

#[test]
fn binary_storage_adapter_persists_and_reloads() {
    let dir = unique_chain_dir();
    let adapter = BinaryStorageAdapter::for_chain_key(&dir, "binary-demo");
    let expected_path = dir.join(thoughtchain::chain_storage_filename(
        "binary-demo",
        StorageAdapterKind::Binary,
    ));

    let mut chain = ThoughtChain::open_with_storage(Box::new(adapter.clone())).unwrap();
    chain
        .append(
            "agent1",
            ThoughtType::Checkpoint,
            "Persist this in the binary adapter.",
        )
        .unwrap();

    let reloaded = ThoughtChain::open_with_storage(Box::new(adapter)).unwrap();
    assert_eq!(reloaded.thoughts().len(), 1);
    assert_eq!(
        reloaded.thoughts()[0].content,
        "Persist this in the binary adapter."
    );
    assert_eq!(
        reloaded.storage_location(),
        expected_path.display().to_string()
    );

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn shared_chain_queries_can_filter_by_agent_identity() {
    let dir = unique_chain_dir();
    let mut chain = ThoughtChain::open_with_key(&dir, "shared-project").unwrap();

    chain
        .append_thought(
            "agent-alpha",
            ThoughtInput::new(ThoughtType::Insight, "Rate limiting is upstream.")
                .with_agent_name("Planner")
                .with_agent_owner("team-red"),
        )
        .unwrap();
    chain
        .append_thought(
            "agent-beta",
            ThoughtInput::new(ThoughtType::Decision, "Use backoff and retry windows.")
                .with_agent_name("Executor")
                .with_agent_owner("team-blue"),
        )
        .unwrap();

    let by_name = chain.query(&ThoughtQuery::new().with_agent_names(["Planner"]));
    assert_eq!(by_name.len(), 1);
    assert_eq!(by_name[0].agent_id, "agent-alpha");

    let by_owner = chain.query(&ThoughtQuery::new().with_agent_owners(["team-blue"]));
    assert_eq!(by_owner.len(), 1);
    assert_eq!(by_owner[0].agent_name, "Executor");

    let by_text = chain.query(&ThoughtQuery::new().with_text("team-red"));
    assert_eq!(by_text.len(), 1);
    assert_eq!(by_text[0].agent_name, "Planner");

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn memory_markdown_groups_thoughts_into_sections() {
    let dir = unique_chain_dir();
    let mut chain = ThoughtChain::open(&dir, "agent1", "Analyst", Some("memory"), None).unwrap();

    chain
        .append(
            "agent1",
            ThoughtType::PreferenceUpdate,
            "User prefers short Markdown outputs.",
        )
        .unwrap();
    chain
        .append(
            "agent1",
            ThoughtType::Decision,
            "Use SQLite for local memory indexing.",
        )
        .unwrap();
    chain
        .append(
            "agent1",
            ThoughtType::Wonder,
            "Would concept embeddings improve retrieval quality?",
        )
        .unwrap();
    chain
        .append(
            "agent1",
            ThoughtType::Question,
            "Should embeddings be optional?",
        )
        .unwrap();
    chain
        .append_thought(
            "agent1",
            ThoughtInput::new(
                ThoughtType::LessonLearned,
                "When a fix takes multiple failed passes, store the final operating rule for the next agent.",
            )
            .with_role(ThoughtRole::Retrospective),
        )
        .unwrap();

    let markdown = chain.to_memory_markdown(None);
    assert!(markdown.contains("# MEMORY"));
    assert!(markdown.contains("## Identity"));
    assert!(markdown.contains("## Constraints And Decisions"));
    assert!(markdown.contains("## Corrections"));
    assert!(markdown.contains("## Open Threads"));
    assert!(markdown.contains("User prefers short Markdown outputs."));
    assert!(markdown.contains("Would concept embeddings improve retrieval quality?"));
    assert!(markdown.contains("role Retrospective"));
    assert!(markdown.contains("When a fix takes multiple failed passes"));
    assert!(markdown.contains("agent agent1"));

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn filename_depends_only_on_chain_key() {
    let first = chain_filename("agent1", "Analyst", Some("rust"), Some("friendly"));
    let second = chain_filename("agent1", "Different", Some("go"), Some("severe"));
    let third = chain_filename("agent2", "Analyst", Some("rust"), Some("friendly"));

    assert_eq!(first, second);
    assert_ne!(first, third);
}

#[test]
fn chain_key_can_be_recovered_from_storage_filename() {
    let filename = chain_storage_filename("borganism-brain", StorageAdapterKind::Binary);
    let recovered = chain_key_from_storage_filename(&filename).unwrap();

    assert_eq!(recovered, "borganism-brain");
    assert!(chain_key_from_storage_filename("not-a-thoughtchain-file.txt").is_none());
}
