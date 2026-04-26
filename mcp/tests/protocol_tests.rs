//! Integration tests for MCP protocol types: ToolMetadata, ToolParameter, ToolRegistry, ToolResult.

use async_trait::async_trait;
use mcp::protocol::{
    Tool, ToolMetadata, ToolParameter, ToolParameterType, ToolProtocol, ToolRegistry, ToolResult,
};
use std::collections::HashMap;
use std::sync::Arc;

/// Mock protocol for testing registry and tool routing.
struct MockProtocol;

#[async_trait]
impl ToolProtocol for MockProtocol {
    async fn execute(
        &self,
        tool_name: &str,
        _parameters: serde_json::Value,
    ) -> Result<ToolResult, Box<dyn std::error::Error + Send + Sync>> {
        Ok(ToolResult::success(serde_json::json!({
            "tool": tool_name,
            "result": "mock_result"
        })))
    }

    async fn list_tools(
        &self,
    ) -> Result<Vec<ToolMetadata>, Box<dyn std::error::Error + Send + Sync>> {
        Ok(vec![])
    }

    async fn get_tool_metadata(
        &self,
        _tool_name: &str,
    ) -> Result<ToolMetadata, Box<dyn std::error::Error + Send + Sync>> {
        Ok(ToolMetadata::new("mock_tool", "A mock tool"))
    }

    fn protocol_name(&self) -> &str {
        "mock"
    }
}

#[tokio::test]
async fn test_tool_parameter_builder() {
    let param = ToolParameter::new("test_param", ToolParameterType::String)
        .with_description("A test parameter")
        .required()
        .with_default(serde_json::json!("default_value"));

    assert_eq!(param.name, "test_param");
    assert_eq!(param.param_type, ToolParameterType::String);
    assert_eq!(param.description, Some("A test parameter".to_string()));
    assert!(param.required);
    assert_eq!(param.default, Some(serde_json::json!("default_value")));
}

#[tokio::test]
async fn test_tool_execution() {
    let protocol = Arc::new(MockProtocol);
    let tool = Tool::new("test_tool", "A test tool", protocol.clone());

    let result = tool.execute(serde_json::json!({})).await.unwrap();
    assert!(result.success);
    assert_eq!(result.output["tool"], "test_tool");
}

#[tokio::test]
async fn test_tool_registry() {
    let protocol = Arc::new(MockProtocol);
    let mut registry = ToolRegistry::new(protocol.clone());

    let tool = Tool::new("calculator", "Performs calculations", protocol.clone());
    registry.add_tool(tool);

    assert!(registry.get_tool("calculator").is_some());
    assert_eq!(registry.list_tools().len(), 1);

    let result = registry
        .execute_tool("calculator", serde_json::json!({}))
        .await
        .unwrap();
    assert!(result.success);
}

#[tokio::test]
async fn test_empty_registry_creation() {
    let registry = ToolRegistry::empty();
    assert_eq!(registry.list_tools().len(), 0);
    assert_eq!(registry.list_protocols().len(), 0);
    assert!(registry.protocol().is_none());
}

#[tokio::test]
async fn test_add_single_protocol_to_empty_registry() {
    let protocol = Arc::new(MockProtocol);
    let mut registry = ToolRegistry::empty();

    registry
        .add_protocol("mock", protocol.clone())
        .await
        .unwrap();

    assert_eq!(registry.list_protocols().len(), 1);
    assert!(registry.list_protocols().contains(&"mock"));
}

#[tokio::test]
async fn test_add_multiple_protocols() {
    let protocol1 = Arc::new(MockProtocol);
    let protocol2 = Arc::new(MockProtocol);
    let mut registry = ToolRegistry::empty();

    registry
        .add_protocol("protocol1", protocol1.clone())
        .await
        .unwrap();
    registry
        .add_protocol("protocol2", protocol2.clone())
        .await
        .unwrap();

    assert_eq!(registry.list_protocols().len(), 2);
    assert!(registry.list_protocols().contains(&"protocol1"));
    assert!(registry.list_protocols().contains(&"protocol2"));
}

#[tokio::test]
async fn test_remove_protocol() {
    let protocol = Arc::new(MockProtocol);
    let mut registry = ToolRegistry::empty();

    registry
        .add_protocol("protocol1", protocol.clone())
        .await
        .unwrap();
    assert_eq!(registry.list_protocols().len(), 1);

    registry.remove_protocol("protocol1");
    assert_eq!(registry.list_protocols().len(), 0);
}

#[tokio::test]
async fn test_execute_tool_through_registry() {
    let protocol = Arc::new(MockProtocol);
    let mut registry = ToolRegistry::empty();

    registry
        .add_protocol("mock", protocol.clone())
        .await
        .unwrap();

    let tool = Tool::new("test_tool", "A test tool", protocol.clone());
    registry.add_tool(tool);

    let result = registry
        .execute_tool("test_tool", serde_json::json!({}))
        .await
        .unwrap();

    assert!(result.success);
    assert_eq!(result.output["tool"], "test_tool");
}

#[tokio::test]
async fn test_backwards_compatibility_single_protocol() {
    let protocol = Arc::new(MockProtocol);
    let registry = ToolRegistry::new(protocol.clone());

    assert!(registry.protocol().is_some());
    assert_eq!(registry.list_protocols().len(), 1);
    assert!(registry.list_protocols().contains(&"primary"));
}

#[tokio::test]
async fn test_discover_tools_from_primary() {
    struct TestProtocol {
        tools: Vec<ToolMetadata>,
    }

    #[async_trait]
    impl ToolProtocol for TestProtocol {
        async fn execute(
            &self,
            tool_name: &str,
            _parameters: serde_json::Value,
        ) -> Result<ToolResult, Box<dyn std::error::Error + Send + Sync>> {
            Ok(ToolResult::success(serde_json::json!({
                "tool": tool_name,
            })))
        }

        async fn list_tools(
            &self,
        ) -> Result<Vec<ToolMetadata>, Box<dyn std::error::Error + Send + Sync>> {
            Ok(self.tools.clone())
        }

        async fn get_tool_metadata(
            &self,
            tool_name: &str,
        ) -> Result<ToolMetadata, Box<dyn std::error::Error + Send + Sync>> {
            self.tools
                .iter()
                .find(|t| t.name == tool_name)
                .cloned()
                .ok_or_else(|| "Tool not found".into())
        }

        fn protocol_name(&self) -> &str {
            "test"
        }
    }

    let protocol = Arc::new(TestProtocol {
        tools: vec![
            ToolMetadata::new("tool1", "First tool"),
            ToolMetadata::new("tool2", "Second tool"),
        ],
    });

    let mut registry = ToolRegistry::new(protocol.clone());

    assert_eq!(registry.list_tools().len(), 0);

    registry.discover_tools_from_primary().await.unwrap();

    assert_eq!(registry.list_tools().len(), 2);
    assert!(registry.get_tool("tool1").is_some());
    assert!(registry.get_tool("tool2").is_some());
}

// ──────────────────────────────────────────────────────────────────────────
// JSON Schema generation tests (native tool-calling support, v0.11.1)
// ──────────────────────────────────────────────────────────────────────────

#[test]
fn test_to_json_schema_string() {
    let param =
        ToolParameter::new("q", ToolParameterType::String).with_description("Search query");
    let schema = param.to_json_schema();
    assert_eq!(schema["type"], "string");
    assert_eq!(schema["description"], "Search query");
}

#[test]
fn test_to_json_schema_number() {
    let param = ToolParameter::new("value", ToolParameterType::Number);
    let schema = param.to_json_schema();
    assert_eq!(schema["type"], "number");
    assert!(schema.get("description").is_none());
}

#[test]
fn test_to_json_schema_integer() {
    let schema = ToolParameter::new("n", ToolParameterType::Integer).to_json_schema();
    assert_eq!(schema["type"], "integer");
}

#[test]
fn test_to_json_schema_boolean() {
    let schema = ToolParameter::new("flag", ToolParameterType::Boolean).to_json_schema();
    assert_eq!(schema["type"], "boolean");
}

#[test]
fn test_to_json_schema_array_with_items() {
    let param = ToolParameter::new("ids", ToolParameterType::Array)
        .with_items(ToolParameterType::Integer);
    let schema = param.to_json_schema();
    assert_eq!(schema["type"], "array");
    assert_eq!(schema["items"]["type"], "integer");
}

#[test]
fn test_to_json_schema_array_without_items() {
    let schema = ToolParameter::new("items", ToolParameterType::Array).to_json_schema();
    assert_eq!(schema["type"], "array");
    assert!(schema.get("items").is_some());
}

#[test]
fn test_to_json_schema_object_with_properties() {
    let mut props = HashMap::new();
    props.insert(
        "name".to_string(),
        ToolParameter::new("name", ToolParameterType::String)
            .with_description("Person's name")
            .required(),
    );
    props.insert(
        "age".to_string(),
        ToolParameter::new("age", ToolParameterType::Integer),
    );
    let param = ToolParameter::new("person", ToolParameterType::Object).with_properties(props);
    let schema = param.to_json_schema();
    assert_eq!(schema["type"], "object");
    assert!(schema["properties"]["name"].is_object());
    assert!(schema["properties"]["age"].is_object());
    let required = schema["required"].as_array().unwrap();
    assert!(required.iter().any(|v| v.as_str() == Some("name")));
    assert!(!required.iter().any(|v| v.as_str() == Some("age")));
}

#[test]
fn test_to_tool_definition_roundtrip() {
    let meta = ToolMetadata::new("calculator", "Evaluates a math expression")
        .with_parameter(
            ToolParameter::new("expression", ToolParameterType::String)
                .with_description("The expression")
                .required(),
        )
        .with_parameter(
            ToolParameter::new("precision", ToolParameterType::Integer)
                .with_description("Decimal places"),
        );

    let def = meta.to_tool_definition();
    assert_eq!(def.name, "calculator");
    assert_eq!(def.description, "Evaluates a math expression");
    assert_eq!(def.parameters_schema["type"], "object");
    assert!(def.parameters_schema["properties"]["expression"].is_object());
    assert!(def.parameters_schema["properties"]["precision"].is_object());

    let required = def.parameters_schema["required"].as_array().unwrap();
    assert!(required.iter().any(|v| v.as_str() == Some("expression")));
    assert!(!required.iter().any(|v| v.as_str() == Some("precision")));
}

#[test]
fn test_to_tool_definitions_collects_all() {
    let protocol = Arc::new(MockProtocol);
    let mut registry = ToolRegistry::empty();

    let tool_a = Tool::new("tool_a", "First tool", protocol.clone());
    registry.add_tool(tool_a);
    let tool_b = Tool::new("tool_b", "Second tool", protocol.clone());
    registry.add_tool(tool_b);

    let defs = registry.to_tool_definitions();
    assert_eq!(defs.len(), 2);
    let names: Vec<&str> = defs.iter().map(|d| d.name.as_str()).collect();
    assert!(names.contains(&"tool_a"));
    assert!(names.contains(&"tool_b"));
}

#[test]
fn test_to_tool_definitions_empty_registry() {
    let registry = ToolRegistry::empty();
    let defs = registry.to_tool_definitions();
    assert!(defs.is_empty());
}
