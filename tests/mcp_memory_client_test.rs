use cloudllm::tool_adapters::McpMemoryProtocol;
use cloudllm::tool_protocol::ToolProtocol;

#[tokio::test]
async fn test_mcp_memory_client_creation() {
    let client = McpMemoryProtocol::new("http://localhost:8080".to_string());
    assert_eq!(client.endpoint(), "http://localhost:8080");
    assert_eq!(client.protocol_name(), "mcp-memory-client");
}

#[tokio::test]
async fn test_mcp_memory_client_with_custom_timeout() {
    let client = McpMemoryProtocol::with_timeout("http://127.0.0.1:3000".to_string(), 60);
    assert_eq!(client.endpoint(), "http://127.0.0.1:3000");
    assert_eq!(client.protocol_name(), "mcp-memory-client");
}

#[tokio::test]
async fn test_mcp_memory_client_clone() {
    let client = McpMemoryProtocol::new("http://localhost:8080".to_string());
    let cloned = client.clone();
    assert_eq!(cloned.endpoint(), client.endpoint());
    assert_eq!(cloned.protocol_name(), client.protocol_name());
}

#[tokio::test]
async fn test_mcp_memory_client_different_endpoints() {
    let client1 = McpMemoryProtocol::new("http://localhost:8080".to_string());
    let client2 = McpMemoryProtocol::new("http://192.168.1.100:3000".to_string());
    let client3 = McpMemoryProtocol::new("http://0.0.0.0:5000".to_string());

    assert_eq!(client1.endpoint(), "http://localhost:8080");
    assert_eq!(client2.endpoint(), "http://192.168.1.100:3000");
    assert_eq!(client3.endpoint(), "http://0.0.0.0:5000");
}

#[tokio::test]
async fn test_mcp_memory_client_timeout_values() {
    let client30 = McpMemoryProtocol::with_timeout("http://localhost:8080".to_string(), 30);
    let client60 = McpMemoryProtocol::with_timeout("http://localhost:8080".to_string(), 60);
    let client120 = McpMemoryProtocol::with_timeout("http://localhost:8080".to_string(), 120);

    assert_eq!(client30.endpoint(), "http://localhost:8080");
    assert_eq!(client60.endpoint(), "http://localhost:8080");
    assert_eq!(client120.endpoint(), "http://localhost:8080");
}

#[tokio::test]
async fn test_mcp_memory_client_protocol_consistency() {
    let clients = vec![
        McpMemoryProtocol::new("http://localhost:8080".to_string()),
        McpMemoryProtocol::with_timeout("http://localhost:9000".to_string(), 45),
        McpMemoryProtocol::new("http://127.0.0.1:5000".to_string()),
    ];

    for client in clients {
        assert_eq!(client.protocol_name(), "mcp-memory-client");
    }
}
