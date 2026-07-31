//! Resource Protocol Abstraction
//!
//! This module provides support for MCP Resources - application-provided contextual data
//! that agents can read and reference.
//!
//! Resources complement Tools:
//! - **Tools**: Model-controlled actions (agent decides when to invoke)
//! - **Resources**: Application-controlled context (app provides to agent)
//!
//! # Architecture
//!
//! ```text
//! Agent → ResourceProtocol → Resource URIs
//!                         → Read Resource Content
//! ```
//!
//! # Example
//!
//! ```rust,ignore
//! use cloudllm::resource_protocol::{ResourceMetadata, ResourceProtocol};
//! use std::sync::Arc;
//!
//! struct MyResourceProtocol;
//!
//! #[async_trait::async_trait]
//! impl ResourceProtocol for MyResourceProtocol {
//!     async fn list_resources(&self) -> Result<Vec<ResourceMetadata>, Box<dyn std::error::Error + Send + Sync>> {
//!         Ok(vec![
//!             ResourceMetadata::new("file:///config.yaml", "Application configuration"),
//!             ResourceMetadata::new("schema:///database", "Database schema"),
//!         ])
//!     }
//!
//!     async fn read_resource(&self, uri: &str) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
//!         match uri {
//!             "file:///config.yaml" => Ok("...".to_string()),
//!             _ => Err("Not found".into()),
//!         }
//!     }
//! }
//! ```

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::error::Error;

/// Metadata describing a resource.
///
/// Wire JSON follows the MCP resource schema: required `name`, camelCase
/// `mimeType`, and `_meta` for free-form metadata. Rust field names stay
/// idiomatic (`mime_type`, `metadata`); serde renames handle the wire form.
/// Deserialize also accepts the legacy snake_case keys (`mime_type`,
/// `metadata`) so older payloads still round-trip during upgrades.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceMetadata {
    /// Programmatic resource name (MCP `BaseMetadata.name`, required).
    ///
    /// Examples: `"core"`, `"config.yaml"`. Defaults to the final URI path
    /// segment when constructed via [`ResourceMetadata::new`].
    pub name: String,
    /// Unique resource identifier (URI)
    /// Examples: "file:///config.yaml", "schema:///users", "db:///schema.sql"
    pub uri: String,
    /// Human-readable description of the resource
    pub description: String,
    /// Optional MIME type of the resource content (wire: `mimeType`)
    #[serde(
        default,
        rename = "mimeType",
        alias = "mime_type",
        skip_serializing_if = "Option::is_none"
    )]
    pub mime_type: Option<String>,
    /// Additional metadata (wire: `_meta`)
    #[serde(default, rename = "_meta", alias = "metadata")]
    pub metadata: HashMap<String, serde_json::Value>,
}

impl ResourceMetadata {
    /// Create a new resource with URI and description.
    ///
    /// `name` is derived from the final path segment of `uri` (e.g.
    /// `mentisdb://skill/core` → `"core"`). Use [`Self::with_name`] to set an
    /// explicit MCP name when the URI segment is not the desired name.
    pub fn new(uri: impl Into<String>, description: impl Into<String>) -> Self {
        let uri = uri.into();
        let name = default_name_from_uri(&uri);
        Self {
            name,
            uri,
            description: description.into(),
            mime_type: None,
            metadata: HashMap::new(),
        }
    }

    /// Override the MCP `name` field (required on the wire).
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    /// Set the MIME type for this resource
    pub fn with_mime_type(mut self, mime_type: impl Into<String>) -> Self {
        self.mime_type = Some(mime_type.into());
        self
    }

    /// Add metadata to the resource
    pub fn with_metadata(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        self.metadata.insert(key.into(), value);
        self
    }
}

/// Derive a default MCP resource name from a URI's final path segment.
fn default_name_from_uri(uri: &str) -> String {
    let trimmed = uri.trim_end_matches('/');
    trimmed
        .rsplit('/')
        .next()
        .filter(|segment| !segment.is_empty())
        .unwrap_or("resource")
        .to_string()
}

/// Trait for implementing resource protocols
///
/// Resources are application-provided contextual data that agents can read.
/// Unlike tools (which perform actions), resources provide information.
#[async_trait]
pub trait ResourceProtocol: Send + Sync {
    /// List all available resources
    async fn list_resources(&self) -> Result<Vec<ResourceMetadata>, Box<dyn Error + Send + Sync>>;

    /// Read the content of a resource by URI
    async fn read_resource(&self, uri: &str) -> Result<String, Box<dyn Error + Send + Sync>>;

    /// Protocol identifier (e.g., "mcp", "custom")
    fn protocol_name(&self) -> &str {
        "resource"
    }

    /// Initialize/connect to the resource protocol (optional)
    async fn initialize(&mut self) -> Result<(), Box<dyn Error + Send + Sync>> {
        Ok(())
    }

    /// Cleanup/disconnect from the resource protocol (optional)
    async fn shutdown(&mut self) -> Result<(), Box<dyn Error + Send + Sync>> {
        Ok(())
    }
}

/// Error types for resource operations
#[derive(Debug, Clone)]
pub enum ResourceError {
    /// Requested resource is not available
    NotFound(String),
    /// Permission denied reading this resource
    PermissionDenied(String),
    /// Invalid URI format
    InvalidUri(String),
    /// Protocol error
    ProtocolError(String),
}

impl std::fmt::Display for ResourceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ResourceError::NotFound(uri) => write!(f, "Resource not found: {}", uri),
            ResourceError::PermissionDenied(uri) => write!(f, "Permission denied: {}", uri),
            ResourceError::InvalidUri(uri) => write!(f, "Invalid URI: {}", uri),
            ResourceError::ProtocolError(msg) => write!(f, "Protocol error: {}", msg),
        }
    }
}

impl std::error::Error for ResourceError {}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{json, Value};

    #[test]
    fn new_derives_name_from_uri_path_segment() {
        let resource = ResourceMetadata::new(
            "mentisdb://skill/core",
            "MentisDB operating skill",
        )
        .with_mime_type("text/markdown")
        .with_metadata("recommended_first", json!(true));

        assert_eq!(resource.name, "core");
        assert_eq!(resource.uri, "mentisdb://skill/core");
        assert_eq!(resource.mime_type.as_deref(), Some("text/markdown"));
    }

    #[test]
    fn with_name_overrides_derived_name() {
        let resource = ResourceMetadata::new("file:///config.yaml", "config").with_name("app-config");
        assert_eq!(resource.name, "app-config");
    }

    #[test]
    fn serialize_uses_mcp_wire_keys() {
        let resource = ResourceMetadata::new("mentisdb://skill/core", "skill")
            .with_name("core")
            .with_mime_type("text/markdown")
            .with_metadata("recommended_first", json!(true));

        let value = serde_json::to_value(&resource).expect("serialize");
        assert_eq!(value["name"], "core");
        assert_eq!(value["uri"], "mentisdb://skill/core");
        assert_eq!(value["mimeType"], "text/markdown");
        assert_eq!(value["_meta"]["recommended_first"], true);
        assert!(value.get("mime_type").is_none());
        assert!(value.get("metadata").is_none());
    }

    #[test]
    fn deserialize_accepts_legacy_snake_case_keys() {
        let legacy = json!({
            "name": "core",
            "uri": "mentisdb://skill/core",
            "description": "skill",
            "mime_type": "text/markdown",
            "metadata": {"priority": 1}
        });
        let resource: ResourceMetadata =
            serde_json::from_value(legacy).expect("deserialize legacy");
        assert_eq!(resource.name, "core");
        assert_eq!(resource.mime_type.as_deref(), Some("text/markdown"));
        assert_eq!(resource.metadata.get("priority"), Some(&json!(1)));
    }

    #[test]
    fn deserialize_accepts_mcp_wire_keys() {
        let wire = json!({
            "name": "core",
            "uri": "mentisdb://skill/core",
            "description": "skill",
            "mimeType": "text/markdown",
            "_meta": {"recommended_first": true}
        });
        let resource: ResourceMetadata = serde_json::from_value(wire).expect("deserialize wire");
        assert_eq!(resource.mime_type.as_deref(), Some("text/markdown"));
        assert_eq!(
            resource.metadata.get("recommended_first"),
            Some(&Value::Bool(true))
        );
    }
}
