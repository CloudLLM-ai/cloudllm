//! Documentation tests for Memory tool and MemoryToolAdapter
//!
//! These tests verify the examples in the documentation work correctly

use cloudllm::tool_adapters::MemoryToolAdapter;
use cloudllm::tool_protocol::ToolProtocol;
use cloudllm::tools::Memory;
use std::sync::Arc;

#[tokio::test]
async fn test_memory_basic_operations() {
    let memory = Memory::new();

    // Store data
    memory.put(
        "task_name".to_string(),
        "Document_Summary".to_string(),
        Some(3600),
    );

    // Retrieve data
    if let Some((value, metadata)) = memory.get("task_name", true) {
        assert_eq!(value, "Document_Summary");
        assert!(metadata.is_some());
        assert_eq!(metadata.unwrap().expires_in, Some(3600));
    } else {
        panic!("Expected to find task_name in memory");
    }

    // List all keys
    let keys = memory.list_keys();
    assert!(keys.contains(&"task_name".to_string()));

    // Delete key
    assert!(memory.delete("task_name"));

    // Clear all
    memory.clear();
    assert_eq!(memory.list_keys().len(), 0);
}

#[tokio::test]
async fn test_memory_tool_protocol_adapter() {
    let memory = Arc::new(Memory::new());
    let adapter = Arc::new(MemoryToolAdapter::new(memory));

    // Execute via adapter directly - Put operation
    let result = adapter
        .execute(
            "memory",
            serde_json::json!({"command": "P task_name Document_Summary 3600"}),
        )
        .await;

    assert!(result.is_ok(), "Put command should succeed");
    let result = result.unwrap();
    assert!(result.success);
}

#[tokio::test]
async fn test_memory_adapter_get_operation() {
    let memory = Arc::new(Memory::new());

    // Store directly in memory
    memory.put("key".to_string(), "value".to_string(), None);

    // Retrieve via adapter
    let adapter = Arc::new(MemoryToolAdapter::new(memory));

    let result = adapter
        .execute("memory", serde_json::json!({"command": "G key"}))
        .await;

    assert!(result.is_ok());
    let result = result.unwrap();
    assert!(result.success);
    assert_eq!(result.output["value"], "value");
}

#[tokio::test]
async fn test_memory_adapter_list_operation() {
    let memory = Arc::new(Memory::new());

    // Store multiple items
    memory.put("key1".to_string(), "value1".to_string(), None);
    memory.put("key2".to_string(), "value2".to_string(), None);

    let adapter = Arc::new(MemoryToolAdapter::new(memory));

    let result = adapter
        .execute("memory", serde_json::json!({"command": "L"}))
        .await;

    assert!(result.is_ok());
    let result = result.unwrap();
    assert!(result.success);

    let keys = result.output["keys"].as_array().unwrap();
    assert_eq!(keys.len(), 2);
}

#[tokio::test]
async fn test_memory_adapter_spec_command() {
    let memory = Arc::new(Memory::new());
    let adapter = Arc::new(MemoryToolAdapter::new(memory));

    let result = adapter
        .execute("memory", serde_json::json!({"command": "SPEC"}))
        .await;

    assert!(result.is_ok());
    let result = result.unwrap();
    assert!(result.success);

    let spec = result.output["specification"].as_str().unwrap();
    assert!(spec.contains("Put (P)"));
    assert!(spec.contains("Get (G)"));
}

#[tokio::test]
async fn test_memory_adapter_with_ttl() {
    let memory = Arc::new(Memory::new());
    let adapter = Arc::new(MemoryToolAdapter::new(memory));

    // Store with TTL
    let result = adapter
        .execute(
            "memory",
            serde_json::json!({"command": "P temp_data temporary_value 60"}),
        )
        .await;

    assert!(result.is_ok());
    assert!(result.unwrap().success);

    // Verify it's there
    let result = adapter
        .execute("memory", serde_json::json!({"command": "G temp_data META"}))
        .await;

    assert!(result.is_ok());
    let result = result.unwrap();
    assert_eq!(result.output["value"], "temporary_value");
    assert_eq!(result.output["expires_in"], 60);
}
