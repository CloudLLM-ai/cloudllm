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
//! ```rust,no_run
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
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceMetadata {
    /// Unique resource identifier (URI)
    /// Examples: "file:///config.yaml", "schema:///users", "db:///schema.sql"
    pub uri: String,
    /// Human-readable description of the resource
    pub description: String,
    /// Optional MIME type of the resource content
    pub mime_type: Option<String>,
    /// Additional metadata
    pub metadata: HashMap<String, serde_json::Value>,
}

impl ResourceMetadata {
    /// Create a new resource with URI and description
    pub fn new(uri: impl Into<String>, description: impl Into<String>) -> Self {
        Self {
            uri: uri.into(),
            description: description.into(),
            mime_type: None,
            metadata: HashMap::new(),
        }
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
