//! Agent Memory Tool
//!
//! This module provides a persistent, time-aware memory system for agents to maintain
//! state across multiple LLM calls and coordinate in multi-agent scenarios.
//!
//! # Features
//!
//! - **Key-value storage** with optional TTL (time-to-live) expiration
//! - **Automatic background expiration** of stale entries (1-second cleanup interval)
//! - **Metadata tracking** (creation timestamp, expiration time)
//! - **Succinct protocol** for LLM communication (token-efficient)
//! - **Thread-safe** shared access via Arc<Mutex<...>>
//!
//! # Design Principles
//!
//! - **Token-efficient**: Uses minimal tokens in LLM prompts
//! - **LLM-friendly**: Simple command syntax that models learn quickly
//! - **Automatic**: Handles expiration without manual intervention
//! - **Stateful**: Enables agents to maintain context across sessions
//!
//! # Common Use Cases
//!
//! 1. **Single-Agent Progress Tracking**: Store document metadata, checkpoints, and recovery instructions
//! 2. **Multi-Agent Coordination**: Shared memory for orchestrations to track decisions and build consensus
//! 3. **Session Recovery**: Save state to resume interrupted work
//! 4. **Audit Trails**: Maintain records of decisions and milestones
//!
//! # Examples
//!
//! ## Basic Operations
//!
//! ```ignore
//! use cloudllm::tools::Memory;
//!
//! let memory = Memory::new();
//!
//! // Store data
//! memory.put("task_name".to_string(), "Document_Summary".to_string(), Some(3600));
//!
//! // Retrieve data
//! if let Some((value, metadata)) = memory.get("task_name", true) {
//!     println!("Task: {}, Stored at: {:?}", value, metadata.unwrap().added_utc);
//! }
//!
//! // List all keys
//! let keys = memory.list_keys();
//! println!("Stored keys: {:?}", keys);
//!
//! // Delete key
//! memory.delete("task_name");
//!
//! // Clear all
//! memory.clear();
//! ```
//!
//! ## With Tool Protocol
//!
//! ```ignore
//! use cloudllm::tools::Memory;
//! use cloudllm::tool_protocols::MemoryProtocol;
//! use cloudllm::tool_protocol::ToolRegistry;
//! use std::sync::Arc;
//!
//! let memory = Arc::new(Memory::new());
//! let adapter = Arc::new(MemoryProtocol::new(memory));
//! let registry = Arc::new(ToolRegistry::new(adapter));
//!
//! // Execute via protocol
//! let result = registry.execute_tool("memory",
//!     serde_json::json!({"command": "P task_name Document_Summary 3600"})).await?;
//! ```
//!
//! See `examples/MEMORY_TOOL_GUIDE.md` for comprehensive documentation and patterns.

use chrono::{DateTime, Utc};
use std::collections::{BTreeMap, HashMap};
use std::sync::{Arc, Mutex};
use tokio::time::{self, Duration};

/// Metadata associated with a stored key-value pair
#[derive(Debug, Clone, PartialEq)]
pub struct MemoryMetadata {
    /// UTC timestamp when the entry was created
    pub added_utc: DateTime<Utc>,
    /// Time-to-live in seconds, None if entry never expires
    pub expires_in: Option<u64>,
}

impl MemoryMetadata {
    /// Create new metadata with optional expiration
    fn new(expires_in: Option<u64>) -> Self {
        Self {
            added_utc: Utc::now(),
            expires_in,
        }
    }

    /// Check if the metadata has expired
    fn is_expired(&self) -> bool {
        if let Some(ttl) = self.expires_in {
            let expiration_time = self.added_utc + chrono::Duration::seconds(ttl as i64);
            Utc::now() > expiration_time
        } else {
            false
        }
    }

    /// Check if this entry has an expiration
    fn is_expireable(&self) -> bool {
        self.expires_in.is_some()
    }
}

/// Agent Memory System
///
/// A TTL-aware key-value store designed for agent state management.
/// Supports storing snapshots, instructions, and other important data
/// that should persist across sessions.
#[derive(Debug)]
pub struct Memory {
    store: Arc<Mutex<HashMap<String, (String, MemoryMetadata)>>>,
    // Maps expiration time to a list of keys that expire at that time
    expiring_timestamps_2_keys: Arc<Mutex<BTreeMap<DateTime<Utc>, Vec<String>>>>,
}

impl Memory {
    /// Create a new Memory instance
    ///
    /// Spawns a background task that periodically evicts expired entries every 1 second.
    /// This ensures that TTL-based cleanup happens automatically without manual intervention.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use cloudllm::tools::Memory;
    ///
    /// let memory = Memory::new();
    /// memory.put("key".to_string(), "value".to_string(), None);
    /// assert!(memory.get("key", false).is_some());
    /// ```
    pub fn new() -> Self {
        let store: Arc<Mutex<HashMap<String, (String, MemoryMetadata)>>> =
            Arc::new(Mutex::new(HashMap::new()));

        let expiring_timestamps_2_keys = Arc::new(Mutex::new(BTreeMap::new()));

        // Spawn background task for expiration management
        let expiration_store = Arc::clone(&store);
        tokio::spawn(async move {
            loop {
                time::sleep(Duration::from_secs(1)).await;
                let mut store = expiration_store.lock().unwrap();
                let keys_to_remove: Vec<_> = store
                    .iter()
                    .filter_map(|(key, (_, metadata))| {
                        if metadata.is_expired() {
                            Some(key.clone())
                        } else {
                            None
                        }
                    })
                    .collect();

                for key in keys_to_remove {
                    store.remove(&key);
                }
            }
        });

        Self {
            store,
            expiring_timestamps_2_keys,
        }
    }

    /// Returns a concise, accurate protocol specification for agents.
    ///
    /// Use this in system prompts so agents know how to interact with the
    /// memory tool. The format exactly matches what `MemoryProtocol::execute`
    /// accepts — no length prefixes, plain whitespace-separated tokens.
    pub fn get_protocol_spec() -> String {
        r#"Memory tool quick-reference (tool name: "memory"):

COMMAND FIELD FORMATS — two styles are accepted:
  Style A (inline): {"command": "G mykey"}
  Style B (split):  {"command": "G", "key": "mykey"}

COMMANDS:
  G <key>              Read a value.      → {"value": "..."}  or ERR:NOT_FOUND
  P <key> <value>      Write a value.     → {"status": "OK"}
  P <key> <value> <s>  Write with TTL s.  → {"status": "OK"}
  L                    List all keys.     → {"keys": [...]}
  D <key>              Delete a key.      → {"status": "OK"}  or ERR:NOT_FOUND
  C                    Clear everything.  → {"status": "OK"}
  T A                  Total byte usage.  → {"total_bytes": N}

ALIASES: GET=G, PUT=P, LIST=L, DELETE=D, CLEAR=C

EXAMPLES:
  {"command": "G tetris_current_html"}
  {"command": "GET", "key": "tetris_current_html"}
  {"command": "P", "key": "notes", "value": "step1done"}
  {"command": "L"}

NOTE: P stores single-token values (no spaces). To persist HTML use write_tetris_file."#
            .to_string()
    }

    /// Put a key-value pair with optional TTL (in seconds)
    pub fn put(&self, key: String, value: String, ttl: Option<u64>) {
        let metadata = MemoryMetadata::new(ttl);
        let mut store = self.store.lock().unwrap();
        let mut expiring_timestamps_2_keys = self.expiring_timestamps_2_keys.lock().unwrap();

        // Calculate expiration time
        let expiration_time = if metadata.is_expireable() {
            let ttl_seconds = metadata.expires_in.unwrap_or(0);
            Some(metadata.added_utc + chrono::Duration::seconds(ttl_seconds as i64))
        } else {
            None
        };

        // Store the key-value pair
        store.insert(key.clone(), (value, metadata.clone()));

        // Track expiration time if TTL is set
        if let Some(exp_time) = expiration_time {
            expiring_timestamps_2_keys
                .entry(exp_time)
                .or_default()
                .push(key);
        }
    }

    /// Get the value and optionally metadata for a key
    pub fn get(
        &self,
        key: &str,
        include_metadata: bool,
    ) -> Option<(String, Option<MemoryMetadata>)> {
        self.evict_expired_keys();
        let store = self.store.lock().unwrap();
        if let Some((value, metadata)) = store.get(key) {
            if metadata.is_expired() {
                None
            } else if include_metadata {
                Some((value.clone(), Some(metadata.clone())))
            } else {
                Some((value.clone(), None))
            }
        } else {
            None
        }
    }

    /// Delete a key
    pub fn delete(&self, key: &str) -> bool {
        let mut store = self.store.lock().unwrap();
        store.remove(key).is_some()
    }

    /// List all non-expired keys
    pub fn list_keys(&self) -> Vec<String> {
        self.evict_expired_keys();
        let store = self.store.lock().unwrap();
        store
            .iter()
            .filter(|(_, (_, metadata))| !metadata.is_expired())
            .map(|(key, _)| key.clone())
            .collect()
    }

    /// Clear all keys
    pub fn clear(&self) {
        let mut store = self.store.lock().unwrap();
        store.clear();
    }

    /// Get the total size of stored data in bytes
    ///
    /// Returns (total_bytes, keys_bytes, values_bytes)
    pub fn get_total_bytes_stored(&self) -> (usize, usize, usize) {
        let store = self.store.lock().unwrap();
        let mut total = 0;
        let mut keys_size = 0;
        let mut values_size = 0;

        for (key, (value, _)) in store.iter() {
            keys_size += key.len();
            values_size += value.len();
            total += key.len() + value.len();
        }

        (total, keys_size, values_size)
    }

    /// Evict expired keys from both the main store and the expiration index
    fn evict_expired_keys(&self) {
        let now = Utc::now();

        let mut expiring_timestamps_2_keys = self.expiring_timestamps_2_keys.lock().unwrap();
        let mut store = self.store.lock().unwrap();

        // Remove expired keys from both structures
        expiring_timestamps_2_keys.retain(|&expiry, keys| {
            if expiry <= now {
                // Remove all keys that expired at this time
                for key in keys {
                    store.remove(key);
                }
                false
            } else {
                true
            }
        });
    }
}

impl Default for Memory {
    fn default() -> Self {
        Self::new()
    }
}
