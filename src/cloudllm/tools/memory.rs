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
//! 2. **Multi-Agent Coordination**: Shared memory for councils to track decisions and build consensus
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

    /// Returns the protocol specification for agents
    ///
    /// This should be included in the system prompt to teach the LLM
    /// about the memory interface.
    pub fn get_protocol_spec() -> String {
        r#"
Memory Module Protocol Specification:
--------------------------------------
The memory module supports the following commands for persistent state management:

1. Put (P): Store a key-value pair with optional TTL.
   Syntax: P <key_length>:<key><value_length>:<value>[TTL:<ttl_seconds>]\n
   Example (no expiration): P 3:foo7:barbaz\n
   Example (with expiration): P 3:foo7:barbazTTL:3600\n
   Response: OK

2. Get (G): Retrieve a value for a key, optionally with metadata.
   Syntax: G <key_length>:<key>[META]\n
   Example (without metadata): G 3:foo\n
   Example (with metadata): G 3:fooMETA\n
   Response (with metadata): 7:barbaz|added_utc:2024-11-25T14:30:00Z|expires_in:3600\n
   Response (without metadata): 7:barbaz\n

3. List Keys (L): List all stored keys, optionally with metadata.
   Syntax: L[META]\n
   Example (keys only): L\n
   Example (with metadata): LMETA\n
   Response: 2:foo|added_utc:2024-11-25T14:30:00Z|expires_in:3600,bar\n

4. Delete (D): Remove a specific key.
   Syntax: D <key_length>:<key>\n
   Example: D 3:foo\n
   Response: OK or ERR:NOT_FOUND

5. Clear (C): Delete all stored keys.
   Syntax: C\n
   Response: OK

6. Total Bytes (T): Get total memory usage.
   Syntax: T <scope>\n
   Scopes: A (all), K (keys only), V (values only)
   Example: T A\n
   Response: 512\n

Notes:
- Metadata includes `added_utc` (timestamp) and `expires_in` (TTL in seconds).
- Expired keys are automatically removed by the memory module.
- The module prioritizes efficiency and minimal token usage.

This protocol is designed to minimize token consumption and ensure accurate memory interactions.
"#
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
