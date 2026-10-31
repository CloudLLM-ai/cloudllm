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

/// Metadata describing a resource
///
/// Serializes to the MCP `Resource` shape returned by `resources/list`:
/// `uri` and `name` are always present, `mimeType` uses the spec's camelCase
/// key, and `title` / `mimeType` are omitted when unset. Deserialization also
/// accepts the legacy `mime_type` key and a missing `name`.
///
/// # Example
///
/// ```rust
/// use mcp::ResourceMetadata;
///
/// let resource = ResourceMetadata::new("file:///config.yaml", "Application configuration")
///     .with_name("config")
///     .with_title("Config")
///     .with_mime_type("text/yaml");
///
/// let wire = serde_json::to_value(&resource).unwrap();
/// assert_eq!(wire["uri"], "file:///config.yaml");
/// assert_eq!(wire["name"], "config");
/// assert_eq!(wire["title"], "Config");
/// assert_eq!(wire["mimeType"], "text/yaml");
///
/// // Without `with_name`, the name defaults to the URI.
/// let plain = ResourceMetadata::new("schema:///users", "User table schema");
/// assert_eq!(plain.name, "schema:///users");
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ResourceMetadata {
    /// Unique resource identifier (URI)
    /// Examples: "file:///config.yaml", "schema:///users", "db:///schema.sql"
    pub uri: String,
    /// Programmatic name of the resource, required by the MCP spec.
    /// [`ResourceMetadata::new`] defaults it to the URI.
    #[serde(default)]
    pub name: String,
    /// Optional human-readable display name
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
    /// Human-readable description of the resource
    pub description: String,
    /// Optional MIME type of the resource content
    #[serde(default, alias = "mime_type", skip_serializing_if = "Option::is_none")]
    pub mime_type: Option<String>,
    /// Additional metadata
    pub metadata: HashMap<String, serde_json::Value>,
}

impl ResourceMetadata {
    /// Create a new resource with URI and description.
    ///
    /// The resource `name` defaults to the URI; override it with
    /// [`ResourceMetadata::with_name`].
    pub fn new(uri: impl Into<String>, description: impl Into<String>) -> Self {
        let uri = uri.into();
        Self {
            name: uri.clone(),
            uri,
            title: None,
            description: description.into(),
            mime_type: None,
            metadata: HashMap::new(),
        }
    }

    /// Set the programmatic name for this resource
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    /// Set the human-readable display title for this resource
    pub fn with_title(mut self, title: impl Into<String>) -> Self {
        self.title = Some(title.into());
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
