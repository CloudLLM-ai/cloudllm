use cloudllm::tool_protocol::{ToolMetadata, ToolParameter, ToolParameterType, ToolProtocol};
use cloudllm::tool_protocols::{CustomToolProtocol, OpenAIFunctionsProtocol};
use std::sync::Arc;

#[tokio::test]
async fn test_custom_adapter_sync_tool() {
    let adapter = CustomToolProtocol::new();

    let metadata = ToolMetadata::new("add", "Adds two numbers")
        .with_parameter(ToolParameter::new("a", ToolParameterType::Number).required())
        .with_parameter(ToolParameter::new("b", ToolParameterType::Number).required());

    adapter
        .register_tool(
            metadata,
            Arc::new(|params| {
                let a = params["a"].as_f64().unwrap_or(0.0);
                let b = params["b"].as_f64().unwrap_or(0.0);
                Ok(cloudllm::tool_protocol::ToolResult::success(
                    serde_json::json!({"result": a + b}),
                ))
            }),
        )
        .await;

    let result = adapter
        .execute("add", serde_json::json!({"a": 5.0, "b": 3.0}))
        .await
        .unwrap();

    assert!(result.success);
    assert_eq!(result.output["result"], 8.0);
}

#[tokio::test]
async fn test_custom_adapter_async_tool() {
    let adapter = CustomToolProtocol::new();

    let metadata = ToolMetadata::new("fetch", "Fetches data asynchronously");

    adapter
        .register_async_tool(
            metadata,
            Arc::new(|_params| {
                Box::pin(async {
                    tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
                    Ok(cloudllm::tool_protocol::ToolResult::success(
                        serde_json::json!({"data": "fetched"}),
                    ))
                })
            }),
        )
        .await;

    let result = adapter
        .execute("fetch", serde_json::json!({}))
        .await
        .unwrap();

    assert!(result.success);
    assert_eq!(result.output["data"], "fetched");
}

#[tokio::test]
async fn test_custom_adapter_list_tools() {
    let adapter = CustomToolProtocol::new();

    let metadata1 = ToolMetadata::new("tool1", "First tool");
    let metadata2 = ToolMetadata::new("tool2", "Second tool");

    adapter
        .register_tool(
            metadata1,
            Arc::new(|_| {
                Ok(cloudllm::tool_protocol::ToolResult::success(
                    serde_json::json!({}),
                ))
            }),
        )
        .await;

    adapter
        .register_tool(
            metadata2,
            Arc::new(|_| {
                Ok(cloudllm::tool_protocol::ToolResult::success(
                    serde_json::json!({}),
                ))
            }),
        )
        .await;

    let tools = adapter.list_tools().await.unwrap();
    assert_eq!(tools.len(), 2);
}

#[tokio::test]
async fn test_openai_function_adapter() {
    let adapter = OpenAIFunctionsProtocol::new();

    let metadata = ToolMetadata::new("search", "Searches the web").with_parameter(
        ToolParameter::new("query", ToolParameterType::String)
            .with_description("The search query")
            .required(),
    );

    adapter
        .register_function(
            metadata,
            Arc::new(|params| {
                Box::pin(async move {
                    let query = params["query"].as_str().unwrap_or("");
                    Ok(cloudllm::tool_protocol::ToolResult::success(
                        serde_json::json!({
                            "results": [
                                {"title": "Result 1", "url": "http://example.com/1"},
                                {"title": "Result 2", "url": "http://example.com/2"}
                            ],
                            "query": query
                        }),
                    ))
                })
            }),
        )
        .await;

    let functions = adapter.get_openai_functions().await;
    assert_eq!(functions.len(), 1);
    assert_eq!(functions[0]["name"], "search");
    assert_eq!(functions[0]["description"], "Searches the web");

    let result = adapter
        .execute("search", serde_json::json!({"query": "rust programming"}))
        .await
        .unwrap();

    assert!(result.success);
    assert_eq!(result.output["query"], "rust programming");
}
