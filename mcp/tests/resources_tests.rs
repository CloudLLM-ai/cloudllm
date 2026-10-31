//! Integration tests for MCP resource types: ResourceMetadata serialization.

use mcp::ResourceMetadata;
use serde_json::json;

#[test]
fn test_resource_serializes_to_mcp_resource_shape() {
    let resource = ResourceMetadata::new("mentisdb://skill/core", "Operating skill")
        .with_mime_type("text/markdown");
    let value = serde_json::to_value(&resource).unwrap();

    assert_eq!(value["uri"], "mentisdb://skill/core");
    assert_eq!(value["name"], "mentisdb://skill/core");
    assert_eq!(value["mimeType"], "text/markdown");
    assert!(value.get("mime_type").is_none());
    assert!(value.get("title").is_none());
}

#[test]
fn test_resource_name_and_title_builders() {
    let resource = ResourceMetadata::new("file:///config.yaml", "Configuration")
        .with_name("config")
        .with_title("Application configuration");
    let value = serde_json::to_value(&resource).unwrap();

    assert_eq!(value["name"], "config");
    assert_eq!(value["title"], "Application configuration");
    assert!(value.get("mimeType").is_none());
}

#[test]
fn test_resource_deserializes_legacy_payload() {
    let legacy = json!({
        "uri": "file:///config.yaml",
        "description": "Configuration",
        "mime_type": "text/yaml",
        "metadata": {}
    });
    let resource: ResourceMetadata = serde_json::from_value(legacy).unwrap();

    assert_eq!(resource.mime_type.as_deref(), Some("text/yaml"));
    assert_eq!(resource.name, "");
    assert!(resource.title.is_none());
}
