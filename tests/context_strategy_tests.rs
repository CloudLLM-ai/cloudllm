use cloudllm::context_strategy::parse_refs;

#[test]
fn test_self_compression_parse_refs_basic() {
    let content = "Here is the summary.\nREFS: 150, 200\nEnd.";
    let refs = parse_refs(content);
    assert_eq!(refs, vec![150, 200]);
}

#[test]
fn test_self_compression_parse_refs_with_spaces() {
    let content = "Summary text.\n  REFS:  10 , 20 , 30  \nMore text.";
    let refs = parse_refs(content);
    assert_eq!(refs, vec![10, 20, 30]);
}

#[test]
fn test_self_compression_parse_refs_single() {
    let content = "REFS: 42";
    let refs = parse_refs(content);
    assert_eq!(refs, vec![42]);
}

#[test]
fn test_self_compression_parse_refs_missing() {
    let content = "No refs here.\nJust normal text.";
    let refs = parse_refs(content);
    assert!(refs.is_empty());
}

#[test]
fn test_self_compression_parse_refs_invalid_numbers() {
    let content = "REFS: abc, 10, xyz, 20";
    let refs = parse_refs(content);
    assert_eq!(refs, vec![10, 20]);
}
