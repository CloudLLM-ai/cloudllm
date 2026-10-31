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
    // A missing name falls back to the URI, as `ResourceMetadata::new` does.
    assert_eq!(resource.name, "file:///config.yaml");
    assert!(resource.title.is_none());
}

/// A resource from another, spec-compliant server carries only the keys the
/// spec defines; `description` and `metadata` are optional there.
#[test]
fn test_resource_deserializes_minimal_spec_payload() {
    let spec = json!({ "uri": "file:///a.txt", "name": "a" });
    let resource: ResourceMetadata = serde_json::from_value(spec).unwrap();

    assert_eq!(resource.uri, "file:///a.txt");
    assert_eq!(resource.name, "a");
    assert_eq!(resource.description, "");
    assert!(resource.mime_type.is_none());
    assert!(resource.metadata.is_empty());
}

/// Regression: re-serializing a legacy payload (no `name`, `mime_type` key)
/// must produce a spec-valid resource, never an empty `name`.
#[test]
fn test_legacy_payload_round_trips_to_spec_shape() {
    let legacy = json!({
        "uri": "mentisdb://skill/core",
        "description": "Operating skill",
        "mime_type": "text/markdown",
        "metadata": {}
    });
    let resource: ResourceMetadata = serde_json::from_value(legacy).unwrap();
    let wire = serde_json::to_value(&resource).unwrap();

    assert_eq!(wire["name"], "mentisdb://skill/core");
    assert_eq!(wire["mimeType"], "text/markdown");
    assert!(wire.get("mime_type").is_none());

    let again: ResourceMetadata = serde_json::from_value(wire.clone()).unwrap();
    assert_eq!(serde_json::to_value(&again).unwrap(), wire);
}

/// A payload without the required `uri` is rejected.
#[test]
fn test_resource_without_uri_is_rejected() {
    let missing = json!({ "name": "a", "description": "no uri" });
    assert!(serde_json::from_value::<ResourceMetadata>(missing).is_err());
}
