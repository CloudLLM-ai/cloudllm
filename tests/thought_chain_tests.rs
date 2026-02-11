use cloudllm::thought_chain::{chain_filename, ThoughtChain, ThoughtType};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

static TEST_COUNTER: AtomicU64 = AtomicU64::new(0);

fn unique_chain_dir() -> PathBuf {
    let n = TEST_COUNTER.fetch_add(1, Ordering::SeqCst);
    let dir = std::env::temp_dir().join(format!("cloudllm_tc_test_{}_{}", std::process::id(), n));
    let _ = std::fs::remove_dir_all(&dir);
    dir
}

#[test]
fn test_thought_chain_create_and_append() {
    let dir = unique_chain_dir();
    let mut chain = ThoughtChain::open(&dir, "agent1", "Analyst", Some("data"), None).unwrap();

    chain
        .append("agent1", ThoughtType::Finding, "Found pattern A")
        .unwrap();
    chain
        .append("agent1", ThoughtType::Decision, "Will use approach B")
        .unwrap();
    chain
        .append("agent1", ThoughtType::TaskComplete, "Done with step 1")
        .unwrap();

    assert_eq!(chain.thoughts().len(), 3);
    assert_eq!(chain.thoughts()[0].content, "Found pattern A");
    assert_eq!(chain.thoughts()[1].thought_type, ThoughtType::Decision);
    assert_eq!(chain.thoughts()[2].index, 2);

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn test_thought_chain_disk_persistence() {
    let dir = unique_chain_dir();

    // Write and drop
    {
        let mut chain = ThoughtChain::open(
            &dir,
            "persist_agent",
            "Persist Analyst",
            Some("persist_data"),
            None,
        )
        .unwrap();
        chain
            .append("persist_agent", ThoughtType::Finding, "Persisted finding")
            .unwrap();
        chain
            .append("persist_agent", ThoughtType::Checkpoint, "Checkpoint 1")
            .unwrap();
    }

    // Reopen
    let chain = ThoughtChain::open(
        &dir,
        "persist_agent",
        "Persist Analyst",
        Some("persist_data"),
        None,
    )
    .unwrap();
    assert_eq!(chain.thoughts().len(), 2);
    assert_eq!(chain.thoughts()[0].content, "Persisted finding");
    assert_eq!(chain.thoughts()[1].content, "Checkpoint 1");

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn test_thought_chain_hash_integrity() {
    let dir = unique_chain_dir();
    let mut chain = ThoughtChain::open(&dir, "agent1", "Analyst", Some("data"), None).unwrap();

    chain
        .append("agent1", ThoughtType::Finding, "Entry 1")
        .unwrap();
    chain
        .append("agent1", ThoughtType::Finding, "Entry 2")
        .unwrap();
    chain
        .append("agent1", ThoughtType::Finding, "Entry 3")
        .unwrap();

    assert!(chain.verify_integrity());

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn test_thought_chain_resolve_context() {
    let dir = unique_chain_dir();
    let mut chain = ThoughtChain::open(&dir, "agent1", "Analyst", Some("data"), None).unwrap();

    // #0: Finding
    chain
        .append("agent1", ThoughtType::Finding, "Base finding")
        .unwrap();
    // #1: Decision
    chain
        .append("agent1", ThoughtType::Decision, "Unrelated decision")
        .unwrap();
    // #2: Finding
    chain
        .append("agent1", ThoughtType::Finding, "Important discovery")
        .unwrap();
    // #3: Decision (no refs)
    chain
        .append("agent1", ThoughtType::Decision, "Another decision")
        .unwrap();
    // #4: Compression referencing #0 and #2
    chain
        .append_with_refs(
            "agent1",
            ThoughtType::Compression,
            "Summary of findings",
            vec![0, 2],
        )
        .unwrap();

    let resolved = chain.resolve_context(4);
    let indices: Vec<u64> = resolved.iter().map(|t| t.index).collect();
    assert_eq!(indices, vec![0, 2, 4]);

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn test_thought_chain_filename_determinism() {
    // Same attributes -> same filename
    let f1 = chain_filename("id1", "Name", Some("exp"), Some("pers"));
    let f2 = chain_filename("id1", "Name", Some("exp"), Some("pers"));
    assert_eq!(f1, f2);

    // Different attributes -> different filename
    let f3 = chain_filename("id1", "Name", Some("different"), Some("pers"));
    assert_ne!(f1, f3);

    // Different id -> different filename
    let f4 = chain_filename("id2", "Name", Some("exp"), Some("pers"));
    assert_ne!(f1, f4);
}

#[test]
fn test_thought_chain_bootstrap_prompt() {
    let dir = unique_chain_dir();
    let mut chain = ThoughtChain::open(&dir, "agent1", "Analyst", Some("data"), None).unwrap();

    chain
        .append("agent1", ThoughtType::Finding, "Key insight")
        .unwrap();
    chain
        .append_with_refs(
            "agent1",
            ThoughtType::Compression,
            "Compressed view",
            vec![0],
        )
        .unwrap();

    let prompt = chain.to_bootstrap_prompt(1);
    assert!(prompt.contains("RESTORED CONTEXT"));
    assert!(prompt.contains("Key insight"));
    assert!(prompt.contains("Compressed view"));
    assert!(prompt.contains("[#0]"));
    assert!(prompt.contains("[#1]"));

    let _ = std::fs::remove_dir_all(&dir);
}
