//! MentisDB-backed `memory` tool — drop-in replacement for [`super::tool_protocols::MemoryProtocol`].
//!
//! Same agent-facing command language (`G`/`P`/`L`/`D`/`C`) so RALPH prompts do not
//! change, but values are durable StateSnapshot thoughts on the embedded chain
//! instead of a process-local HashMap that vanishes when the example exits.

use crate::cloudllm::tool_protocol::{
    ToolError, ToolMetadata, ToolParameter, ToolParameterType, ToolProtocol, ToolResult,
};
use async_trait::async_trait;
use mentisdb::{MentisDb, ThoughtInput, ThoughtType};
use serde_json::Value as JsonValue;
use std::collections::HashMap;
use std::error::Error;
use std::sync::{Arc, Mutex};
use tokio::sync::RwLock;

const KV_TAG: &str = "memory-kv";
const TOMBSTONE_TAG: &str = "memory-kv-deleted";

/// Durable key/value store implemented as MentisDB snapshots.
///
/// Latest-write-wins is applied in a process cache hydrated from the chain so
/// GET of a large game page does not rescan every thought.
pub struct MentisDbMemoryProtocol {
    db: Arc<RwLock<MentisDb>>,
    writer_id: String,
    cache: Mutex<HashMap<String, String>>,
}

impl MentisDbMemoryProtocol {
    /// Bind the `memory` tool to an already-open embedded chain.
    pub fn new(db: Arc<RwLock<MentisDb>>, writer_id: impl Into<String>) -> Self {
        let writer_id = writer_id.into();
        let cache = {
            let db = db.try_read().ok();
            db.map(|guard| hydrate_cache(&guard)).unwrap_or_default()
        };
        Self {
            db,
            writer_id,
            cache: Mutex::new(cache),
        }
    }

    /// Latest value for `key`, if any (cache; hydrated from the chain at open).
    pub fn get_value(&self, key: &str) -> Option<String> {
        self.cache
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .get(key)
            .cloned()
    }

    /// Keys currently holding a value.
    pub fn list_keys(&self) -> Vec<String> {
        let mut keys: Vec<String> = self
            .cache
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .keys()
            .cloned()
            .collect();
        keys.sort();
        keys
    }

    /// Update the cache immediately and append if the chain lock is free.
    ///
    /// Used by sync `write_game_file` closures. Prefer [`put_value`] when async.
    pub fn put_value_sync(&self, key: &str, value: &str) {
        self.cache
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .insert(key.to_string(), value.to_string());
        if let Ok(mut db) = self.db.try_write() {
            let input = ThoughtInput::new(
                ThoughtType::StateSnapshot,
                format!("KV {}\n{}", key, value),
            )
            .with_tags([KV_TAG, &format!("kv:{}", key)]);
            let _ = db.append_thought(&self.writer_id, input);
        }
    }

    /// Persist `key`=`value` as a MentisDB snapshot and update the cache.
    pub async fn put_value(
        &self,
        key: &str,
        value: &str,
    ) -> Result<(), Box<dyn Error + Send + Sync>> {
        {
            let mut db = self.db.write().await;
            let input = ThoughtInput::new(
                ThoughtType::StateSnapshot,
                format!("KV {}\n{}", key, value),
            )
            .with_tags([KV_TAG, &format!("kv:{}", key)]);
            db.append_thought(&self.writer_id, input)
                .map_err(|e| -> Box<dyn Error + Send + Sync> { e.to_string().into() })?;
        }
        self.cache
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .insert(key.to_string(), value.to_string());
        Ok(())
    }

    async fn delete_value(&self, key: &str) -> Result<bool, Box<dyn Error + Send + Sync>> {
        let existed = self
            .cache
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .remove(key)
            .is_some();
        if existed {
            let mut db = self.db.write().await;
            let input = ThoughtInput::new(ThoughtType::StateSnapshot, format!("KV DELETE {}", key))
                .with_tags([KV_TAG, TOMBSTONE_TAG, &format!("kv:{}", key)]);
            db.append_thought(&self.writer_id, input)
                .map_err(|e| -> Box<dyn Error + Send + Sync> { e.to_string().into() })?;
        }
        Ok(existed)
    }

    fn memory_tool_metadata(&self) -> ToolMetadata {
        ToolMetadata::new(
            "memory",
            "Durable MentisDB key-value store (embedded chain; no daemon). \
             Same commands as the old in-process Memory tool, but values survive \
             the process and are visible to every agent on this chain.\n\
             \n\
             READ:  {\"command\": \"G mykey\"}  or  {\"command\": \"GET\", \"key\": \"mykey\"}\n\
             WRITE: {\"command\": \"P\", \"key\": \"mykey\", \"value\": \"...\"}  \
             (prefer split-param for large HTML)\n\
             LIST:  {\"command\": \"L\"}\n\
             DELETE: {\"command\": \"D mykey\"}\n\
             CLEAR: {\"command\": \"C\"}\n\
             \n\
             For the full game page prefer write_game_file, which also stores the \
             page under the configured MentisDB key.",
        )
        .with_parameter(
            ToolParameter::new("command", ToolParameterType::String)
                .with_description(
                    "Command: 'G key', 'P key value', 'L', 'D key', 'C'. \
                     Full-word GET/PUT/LIST/DELETE/CLEAR accepted.",
                )
                .required(),
        )
        .with_parameter(
            ToolParameter::new("key", ToolParameterType::String)
                .with_description("Key when using split-parameter GET/PUT/DELETE."),
        )
        .with_parameter(
            ToolParameter::new("value", ToolParameterType::String)
                .with_description("Value for PUT when using split-parameter style."),
        )
    }
}

fn hydrate_cache(db: &MentisDb) -> HashMap<String, String> {
    let mut cache = HashMap::new();
    for thought in db.thoughts() {
        if !thought.tags.iter().any(|t| t == KV_TAG) {
            continue;
        }
        let Some(key) = thought
            .tags
            .iter()
            .find_map(|t| t.strip_prefix("kv:"))
            .map(|s| s.to_string())
        else {
            continue;
        };
        if thought.tags.iter().any(|t| t == TOMBSTONE_TAG) {
            cache.remove(&key);
            continue;
        }
        if let Some(value) = thought.content.strip_prefix("KV ") {
            if let Some((_, rest)) = value.split_once('\n') {
                cache.insert(key, rest.to_string());
            }
        }
    }
    cache
}

fn normalize_memory_command(parameters: &JsonValue) -> Result<String, ToolError> {
    let raw_command = parameters
        .get("command")
        .and_then(|v| v.as_str())
        .ok_or_else(|| {
            ToolError::InvalidParameters(
                "Missing 'command' field. Use e.g. {\"command\": \"G mykey\"}".to_string(),
            )
        })?;
    let normalised_verb = match raw_command.split_whitespace().next().unwrap_or("") {
        "GET" => "G",
        "PUT" => "P",
        "LIST" => "L",
        "DELETE" => "D",
        "CLEAR" => "C",
        other => other,
    };
    let key_param = parameters.get("key").and_then(|v| v.as_str()).unwrap_or("");
    let value_param = parameters
        .get("value")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    let rest = raw_command
        .split_once(char::is_whitespace)
        .map(|x| x.1)
        .unwrap_or("")
        .trim();
    Ok(if !key_param.is_empty() {
        if !value_param.is_empty() {
            format!("{} {} {}", normalised_verb, key_param, value_param)
        } else {
            format!("{} {}", normalised_verb, key_param)
        }
    } else if rest.is_empty() {
        normalised_verb.to_string()
    } else {
        format!("{} {}", normalised_verb, rest)
    })
}

#[async_trait]
impl ToolProtocol for MentisDbMemoryProtocol {
    async fn execute(
        &self,
        tool_name: &str,
        parameters: JsonValue,
    ) -> Result<ToolResult, Box<dyn Error + Send + Sync>> {
        if tool_name != "memory" {
            return Err(Box::new(ToolError::NotFound(tool_name.to_string())));
        }
        let command = normalize_memory_command(&parameters)?;
        let mut parts = command.splitn(3, char::is_whitespace);
        let verb = parts.next().unwrap_or("");
        let key = parts.next().unwrap_or("");
        let value = parts.next().unwrap_or("");

        match verb {
            "P" => {
                if key.is_empty() || value.is_empty() {
                    return Ok(ToolResult::failure("ERR:Invalid PUT Syntax".into()));
                }
                self.put_value(key, value).await?;
                Ok(ToolResult::success(serde_json::json!({"status": "OK"})))
            }
            "G" => {
                if key.is_empty() {
                    return Ok(ToolResult::failure("ERR:Invalid GET Syntax".into()));
                }
                match self.get_value(key) {
                    Some(v) => Ok(ToolResult::success(serde_json::json!({"value": v}))),
                    None => Ok(ToolResult::failure("ERR:NOT_FOUND".into())),
                }
            }
            "L" => Ok(ToolResult::success(serde_json::json!({
                "keys": self.list_keys()
            }))),
            "D" => {
                if key.is_empty() {
                    return Ok(ToolResult::failure("ERR:Invalid DELETE Syntax".into()));
                }
                if self.delete_value(key).await? {
                    Ok(ToolResult::success(serde_json::json!({"status": "OK"})))
                } else {
                    Ok(ToolResult::failure("ERR:NOT_FOUND".into()))
                }
            }
            "C" => {
                let keys = self.list_keys();
                for k in keys {
                    let _ = self.delete_value(&k).await;
                }
                Ok(ToolResult::success(serde_json::json!({"status": "OK"})))
            }
            _ => Ok(ToolResult::failure("ERR:Invalid Command".into())),
        }
    }

    async fn list_tools(&self) -> Result<Vec<ToolMetadata>, Box<dyn Error + Send + Sync>> {
        Ok(vec![self.memory_tool_metadata()])
    }

    async fn get_tool_metadata(
        &self,
        tool_name: &str,
    ) -> Result<ToolMetadata, Box<dyn Error + Send + Sync>> {
        if tool_name != "memory" {
            return Err(Box::new(ToolError::NotFound(tool_name.to_string())));
        }
        Ok(self.memory_tool_metadata())
    }

    fn protocol_name(&self) -> &str {
        "mentisdb-memory"
    }
}
