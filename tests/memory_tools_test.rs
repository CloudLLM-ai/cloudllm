use cloudllm::tools::Memory;

#[tokio::test]
async fn test_memory_put_and_get() {
    let memory = Memory::new();

    memory.put("key1".to_string(), "value1".to_string(), None);
    let result = memory.get("key1", false);

    assert_eq!(result, Some(("value1".to_string(), None)));
}

#[tokio::test]
async fn test_memory_put_and_get_with_metadata() {
    let memory = Memory::new();

    memory.put("key1".to_string(), "value1".to_string(), None);
    let result = memory.get("key1", true);

    assert!(result.is_some());
    let (value, metadata) = result.unwrap();
    assert_eq!(value, "value1");
    assert!(metadata.is_some());
    assert_eq!(metadata.unwrap().expires_in, None);
}

#[tokio::test]
async fn test_memory_put_with_ttl_expiration() {
    let memory = Memory::new();

    // Put with 1 second TTL
    memory.put("key1".to_string(), "value1".to_string(), Some(1));

    // Should exist immediately
    assert!(memory.get("key1", false).is_some());

    // Wait for expiration
    tokio::time::sleep(std::time::Duration::from_secs(2)).await;

    // Should be expired now
    assert!(memory.get("key1", false).is_none());
}

#[tokio::test]
async fn test_memory_delete() {
    let memory = Memory::new();

    memory.put("key1".to_string(), "value1".to_string(), None);
    assert!(memory.delete("key1"));
    assert!(memory.get("key1", false).is_none());
}

#[tokio::test]
async fn test_memory_delete_nonexistent() {
    let memory = Memory::new();
    assert!(!memory.delete("nonexistent"));
}

#[tokio::test]
async fn test_memory_list_keys() {
    let memory = Memory::new();

    memory.put("key1".to_string(), "value1".to_string(), None);
    memory.put("key2".to_string(), "value2".to_string(), None);
    memory.put("key3".to_string(), "value3".to_string(), None);

    let keys = memory.list_keys();
    assert_eq!(keys.len(), 3);
    assert!(keys.contains(&"key1".to_string()));
    assert!(keys.contains(&"key2".to_string()));
    assert!(keys.contains(&"key3".to_string()));
}

#[tokio::test]
async fn test_memory_clear() {
    let memory = Memory::new();

    memory.put("key1".to_string(), "value1".to_string(), None);
    memory.put("key2".to_string(), "value2".to_string(), None);

    memory.clear();

    let keys = memory.list_keys();
    assert_eq!(keys.len(), 0);
}

#[tokio::test]
async fn test_memory_get_total_bytes() {
    let memory = Memory::new();

    memory.put("key1".to_string(), "value1".to_string(), None);
    memory.put("key2".to_string(), "value2".to_string(), None);

    let (total, keys_size, values_size) = memory.get_total_bytes_stored();

    // key1 (4) + value1 (6) + key2 (4) + value2 (6) = 20
    assert_eq!(total, 20);
    assert_eq!(keys_size, 8); // 4 + 4
    assert_eq!(values_size, 12); // 6 + 6
}

#[tokio::test]
async fn test_memory_list_keys_excludes_expired() {
    let memory = Memory::new();

    memory.put("key1".to_string(), "value1".to_string(), None);
    memory.put("key2".to_string(), "value2".to_string(), Some(1)); // Will expire

    assert_eq!(memory.list_keys().len(), 2);

    tokio::time::sleep(std::time::Duration::from_secs(2)).await;

    // key2 should be expired and removed
    let keys = memory.list_keys();
    assert_eq!(keys.len(), 1);
    assert_eq!(keys[0], "key1");
}

#[test]
fn test_memory_protocol_spec_not_empty() {
    let spec = Memory::get_protocol_spec();
    assert!(!spec.is_empty());
    assert!(spec.contains("Put (P)"));
    assert!(spec.contains("Get (G)"));
    assert!(spec.contains("List Keys (L)"));
    assert!(spec.contains("Delete (D)"));
    assert!(spec.contains("Clear (C)"));
    assert!(spec.contains("Total Bytes (T)"));
}

#[tokio::test]
async fn test_memory_multiple_operations_sequence() {
    let memory = Memory::new();

    // Store multiple items
    memory.put("name".to_string(), "Assistant".to_string(), None);
    memory.put("task".to_string(), "Summarize document".to_string(), None);
    memory.put(
        "milestone".to_string(),
        "Half complete".to_string(),
        Some(3600),
    ); // 1 hour

    // Verify all exist
    assert_eq!(memory.list_keys().len(), 3);

    // Retrieve with metadata
    let (value, metadata) = memory.get("milestone", true).unwrap();
    assert_eq!(value, "Half complete");
    assert!(metadata.unwrap().expires_in.is_some());

    // Delete one
    memory.delete("task");
    assert_eq!(memory.list_keys().len(), 2);

    // Check total bytes
    let (total, _, _) = memory.get_total_bytes_stored();
    assert!(total > 0);

    // Clear all
    memory.clear();
    assert_eq!(memory.list_keys().len(), 0);
}
