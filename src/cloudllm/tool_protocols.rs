//! Tool Protocol Implementations
//!
//! This module provides concrete implementations of the ToolProtocol trait
//! for various tool communication standards and transports.
//!
//! Each struct is a complete implementation of ToolProtocol, representing a different
//! way to communicate with tools. These implementations can be used individually or
//! combined in a multi-protocol setup via ToolRegistry.
//!
//! # Available Implementations
//!
//! - **CustomToolProtocol**: Direct Rust function calls (sync and async)
//! - **McpClientProtocol**: HTTP client for remote MCP servers
//! - **MemoryProtocol**: TTL-aware in-process memory store with succinct protocol
//! - **OpenAIFunctionProtocol**: OpenAI-compatible function calling format
//! - **McpMemoryClient**: HTTP client for remote Memory servers (distributed coordination)
//!
//! # Usage Patterns
//!
//! ## Single Protocol
//!
//! ```ignore
//! let protocol = Arc::new(CustomToolProtocol::new());
//! let registry = ToolRegistry::new(protocol);
//! ```
//!
//! ## Multiple Protocols (New in 0.5.0)
//!
//! ```ignore
//! let mut registry = ToolRegistry::empty();
//! registry.add_protocol("local", Arc::new(CustomToolProtocol::new())).await?;
//! registry.add_protocol("mcp", Arc::new(McpClientProtocol::new(url))).await?;
//! ```

use crate::cloudllm::tool_protocol::{
    ToolError, ToolMetadata, ToolParameter, ToolParameterType, ToolProtocol, ToolResult,
};
use async_trait::async_trait;
use serde_json::Value as JsonValue;
use std::collections::HashMap;
use std::error::Error;
use std::sync::Arc;
use tokio::sync::RwLock;

/// Type alias for synchronous tool functions exposed via the custom adapter.
pub type ToolFunction =
    Arc<dyn Fn(JsonValue) -> Result<ToolResult, Box<dyn Error + Send + Sync>> + Send + Sync>;

/// Type alias for asynchronous tool functions exposed via the custom adapter.
pub type AsyncToolFunction = Arc<
    dyn Fn(
            JsonValue,
        ) -> std::pin::Pin<
            Box<
                dyn std::future::Future<Output = Result<ToolResult, Box<dyn Error + Send + Sync>>>
                    + Send,
            >,
        > + Send
        + Sync,
>;

/// Custom function-calling tool adapter
///
/// This adapter allows you to register Rust functions as tools that agents can use.
/// It's useful for quick prototyping and simple tool implementations.
///
/// # Example
///
/// ```rust,no_run
/// use cloudllm::tool_protocols::CustomToolProtocol;
/// use cloudllm::tool_protocol::{ToolResult, ToolMetadata, ToolParameter, ToolParameterType};
/// use std::sync::Arc;
///
/// let mut adapter = CustomToolProtocol::new();
///
/// // Register a synchronous tool
/// adapter.register_tool(
///     ToolMetadata::new("add", "Adds two numbers")
///         .with_parameter(
///             ToolParameter::new("a", ToolParameterType::Number).required()
///         )
///         .with_parameter(
///             ToolParameter::new("b", ToolParameterType::Number).required()
///         ),
///     Arc::new(|params| {
///         let a = params["a"].as_f64().unwrap_or(0.0);
///         let b = params["b"].as_f64().unwrap_or(0.0);
///         Ok(ToolResult::success(serde_json::json!({"result": a + b})))
///     })
/// );
/// ```
pub struct CustomToolProtocol {
    tools: Arc<RwLock<HashMap<String, ToolMetadata>>>,
    sync_functions: Arc<RwLock<HashMap<String, ToolFunction>>>,
    async_functions: Arc<RwLock<HashMap<String, AsyncToolFunction>>>,
}

impl CustomToolProtocol {
    /// Create an empty adapter ready to accept new tool registrations.
    pub fn new() -> Self {
        Self {
            tools: Arc::new(RwLock::new(HashMap::new())),
            sync_functions: Arc::new(RwLock::new(HashMap::new())),
            async_functions: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Register a synchronous tool function.
    ///
    /// Subsequent calls will overwrite any existing tool with the same name.
    pub async fn register_tool(&self, metadata: ToolMetadata, function: ToolFunction) {
        let name = metadata.name.clone();
        self.tools.write().await.insert(name.clone(), metadata);
        self.sync_functions.write().await.insert(name, function);
    }

    /// Register an asynchronous tool function.
    pub async fn register_async_tool(&self, metadata: ToolMetadata, function: AsyncToolFunction) {
        let name = metadata.name.clone();
        self.tools.write().await.insert(name.clone(), metadata);
        self.async_functions.write().await.insert(name, function);
    }

    /// Remove a tool from the adapter.
    pub async fn unregister_tool(&self, name: &str) {
        self.tools.write().await.remove(name);
        self.sync_functions.write().await.remove(name);
        self.async_functions.write().await.remove(name);
    }
}

impl Default for CustomToolProtocol {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl ToolProtocol for CustomToolProtocol {
    async fn execute(
        &self,
        tool_name: &str,
        parameters: JsonValue,
    ) -> Result<ToolResult, Box<dyn Error + Send + Sync>> {
        // Try async functions first
        {
            let async_funcs = self.async_functions.read().await;
            if let Some(func) = async_funcs.get(tool_name) {
                return func(parameters).await;
            }
        }

        // Then try sync functions
        {
            let sync_funcs = self.sync_functions.read().await;
            if let Some(func) = sync_funcs.get(tool_name) {
                return func(parameters);
            }
        }

        Err(Box::new(ToolError::NotFound(tool_name.to_string())))
    }

    async fn list_tools(&self) -> Result<Vec<ToolMetadata>, Box<dyn Error + Send + Sync>> {
        let tools = self.tools.read().await;
        Ok(tools.values().cloned().collect())
    }

    async fn get_tool_metadata(
        &self,
        tool_name: &str,
    ) -> Result<ToolMetadata, Box<dyn Error + Send + Sync>> {
        let tools = self.tools.read().await;
        tools.get(tool_name).cloned().ok_or_else(|| {
            Box::new(ToolError::NotFound(tool_name.to_string())) as Box<dyn Error + Send + Sync>
        })
    }

    fn protocol_name(&self) -> &str {
        "custom"
    }
}

/// MCP (Model Context Protocol) adapter
///
/// This adapter provides integration with the Model Context Protocol standard.
/// It allows agents to communicate with external tools and services using MCP.
///
/// # Example
///
/// ```rust,no_run
/// use cloudllm::tool_protocols::McpClientProtocol;
/// use cloudllm::tool_protocol::ToolProtocol;
///
/// # async {
/// let mut adapter = McpClientProtocol::new("http://localhost:8080/mcp".to_string());
/// adapter.initialize().await.unwrap();
/// # };
/// ```
pub struct McpClientProtocol {
    endpoint: String,
    client: reqwest::Client,
    tools_cache: Arc<RwLock<Option<Vec<ToolMetadata>>>>,
    cache_ttl_secs: u64,
    last_cache_refresh: Arc<RwLock<Option<std::time::Instant>>>,
}

impl McpClientProtocol {
    /// Create an adapter that fetches tool metadata and executes calls against a MCP HTTP relay.
    pub fn new(endpoint: String) -> Self {
        Self {
            endpoint,
            client: reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(30))
                .build()
                .expect("Failed to build HTTP client"),
            tools_cache: Arc::new(RwLock::new(None)),
            cache_ttl_secs: 300, // 5 minutes
            last_cache_refresh: Arc::new(RwLock::new(None)),
        }
    }

    /// Override the default request timeout for subsequent HTTP calls.
    pub fn with_timeout(mut self, timeout_secs: u64) -> Self {
        self.client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(timeout_secs))
            .build()
            .expect("Failed to build HTTP client");
        self
    }

    /// Override the cache TTL (in seconds) for the tool metadata snapshot.
    pub fn with_cache_ttl(mut self, ttl_secs: u64) -> Self {
        self.cache_ttl_secs = ttl_secs;
        self
    }

    async fn should_refresh_cache(&self) -> bool {
        let last_refresh = self.last_cache_refresh.read().await;
        match *last_refresh {
            None => true,
            Some(instant) => instant.elapsed().as_secs() > self.cache_ttl_secs,
        }
    }

    async fn refresh_cache(&self) -> Result<(), Box<dyn Error + Send + Sync>> {
        let response = self
            .client
            .get(format!("{}/tools", self.endpoint))
            .send()
            .await?;

        if !response.status().is_success() {
            return Err(Box::new(ToolError::ProtocolError(format!(
                "MCP server returned status: {}",
                response.status()
            ))));
        }

        let tools: Vec<ToolMetadata> = response.json().await?;
        *self.tools_cache.write().await = Some(tools);
        *self.last_cache_refresh.write().await = Some(std::time::Instant::now());

        Ok(())
    }
}

#[async_trait]
impl ToolProtocol for McpClientProtocol {
    async fn execute(
        &self,
        tool_name: &str,
        parameters: JsonValue,
    ) -> Result<ToolResult, Box<dyn Error + Send + Sync>> {
        let response = self
            .client
            .post(format!("{}/execute", self.endpoint))
            .json(&serde_json::json!({
                "tool": tool_name,
                "parameters": parameters
            }))
            .send()
            .await?;

        if !response.status().is_success() {
            return Err(Box::new(ToolError::ExecutionFailed(format!(
                "MCP server returned status: {}",
                response.status()
            ))));
        }

        let result: ToolResult = response.json().await?;
        Ok(result)
    }

    async fn list_tools(&self) -> Result<Vec<ToolMetadata>, Box<dyn Error + Send + Sync>> {
        if self.should_refresh_cache().await {
            self.refresh_cache().await?;
        }

        let cache = self.tools_cache.read().await;
        cache.as_ref().cloned().ok_or_else(|| {
            Box::new(ToolError::ProtocolError(
                "Tools cache not initialized".to_string(),
            )) as Box<dyn Error + Send + Sync>
        })
    }

    async fn get_tool_metadata(
        &self,
        tool_name: &str,
    ) -> Result<ToolMetadata, Box<dyn Error + Send + Sync>> {
        let tools = self.list_tools().await?;
        tools
            .into_iter()
            .find(|t| t.name == tool_name)
            .ok_or_else(|| {
                Box::new(ToolError::NotFound(tool_name.to_string())) as Box<dyn Error + Send + Sync>
            })
    }

    fn protocol_name(&self) -> &str {
        "mcp"
    }

    async fn initialize(&mut self) -> Result<(), Box<dyn Error + Send + Sync>> {
        // Test connection and load initial tool list
        self.refresh_cache().await
    }

    async fn shutdown(&mut self) -> Result<(), Box<dyn Error + Send + Sync>> {
        // Clear cache
        *self.tools_cache.write().await = None;
        *self.last_cache_refresh.write().await = None;
        Ok(())
    }
}

/// OpenAI-style function calling adapter
///
/// This adapter formats tools in the OpenAI function calling format,
/// making it easy to integrate with OpenAI's function calling API.
pub struct OpenAIFunctionsProtocol {
    tools: Arc<RwLock<HashMap<String, ToolMetadata>>>,
    functions: Arc<RwLock<HashMap<String, AsyncToolFunction>>>,
}

impl OpenAIFunctionsProtocol {
    /// Create an adapter that exposes tools using the OpenAI function calling interface.
    pub fn new() -> Self {
        Self {
            tools: Arc::new(RwLock::new(HashMap::new())),
            functions: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Register a new async function that can be invoked by the protocol.
    pub async fn register_function(&self, metadata: ToolMetadata, function: AsyncToolFunction) {
        let name = metadata.name.clone();
        self.tools.write().await.insert(name.clone(), metadata);
        self.functions.write().await.insert(name, function);
    }

    /// Render registered tools into the JSON structure expected by OpenAI's function calling API.
    pub async fn get_openai_functions(&self) -> Vec<JsonValue> {
        let tools = self.tools.read().await;
        tools
            .values()
            .map(|metadata| {
                let mut properties = serde_json::Map::new();
                let mut required = Vec::new();

                for param in &metadata.parameters {
                    properties.insert(
                        param.name.clone(),
                        serde_json::json!({
                            "type": param.param_type,
                            "description": param.description.as_deref().unwrap_or("")
                        }),
                    );

                    if param.required {
                        required.push(param.name.clone());
                    }
                }

                serde_json::json!({
                    "name": metadata.name,
                    "description": metadata.description,
                    "parameters": {
                        "type": "object",
                        "properties": properties,
                        "required": required
                    }
                })
            })
            .collect()
    }
}

impl Default for OpenAIFunctionsProtocol {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl ToolProtocol for OpenAIFunctionsProtocol {
    async fn execute(
        &self,
        tool_name: &str,
        parameters: JsonValue,
    ) -> Result<ToolResult, Box<dyn Error + Send + Sync>> {
        let functions = self.functions.read().await;
        let func = functions
            .get(tool_name)
            .ok_or_else(|| ToolError::NotFound(tool_name.to_string()))?;

        func(parameters).await
    }

    async fn list_tools(&self) -> Result<Vec<ToolMetadata>, Box<dyn Error + Send + Sync>> {
        let tools = self.tools.read().await;
        Ok(tools.values().cloned().collect())
    }

    async fn get_tool_metadata(
        &self,
        tool_name: &str,
    ) -> Result<ToolMetadata, Box<dyn Error + Send + Sync>> {
        let tools = self.tools.read().await;
        tools.get(tool_name).cloned().ok_or_else(|| {
            Box::new(ToolError::NotFound(tool_name.to_string())) as Box<dyn Error + Send + Sync>
        })
    }

    fn protocol_name(&self) -> &str {
        "openai-functions"
    }
}

/// Memory Tool Adapter
///
/// Provides integration with the CloudLLM Memory tool using a succinct, token-efficient protocol.
/// The protocol is designed to minimize token usage while being easily understood by LLMs.
///
/// This adapter implements the [`ToolProtocol`] trait, allowing Memory to work with
/// CloudLLM's agent and orchestration systems. It translates between LLM tool calls and Memory
/// operations.
///
/// # Protocol Commands
///
/// The adapter supports the following commands (sent via tool parameters):
/// - `P <key> <value> [ttl]` - Put (store) a value
/// - `G <key> [META]` - Get (retrieve) a value
/// - `L [META]` - List all keys
/// - `D <key>` - Delete a key
/// - `C` - Clear all keys
/// - `T <scope>` - Total bytes (scope: A=all, K=keys, V=values)
/// - `SPEC` - Get protocol specification
///
/// # Usage with Agents
///
/// ```ignore
/// use cloudllm::tools::Memory;
/// use cloudllm::tool_protocols::MemoryProtocol;
/// use cloudllm::tool_protocol::ToolRegistry;
/// use cloudllm::Agent;
/// use std::sync::Arc;
///
/// let memory = Arc::new(Memory::new());
/// let adapter = Arc::new(MemoryProtocol::new(memory));
/// let registry = Arc::new(ToolRegistry::new(adapter));
///
/// let agent = Agent::new("analyzer", "Analyzer", client)
///     .with_tools(registry);
/// ```
///
/// # Usage with Orchestrations
///
/// For multi-agent coordination, create a shared Memory:
///
/// ```ignore
/// let shared_memory = Arc::new(Memory::new());
/// let shared_adapter = Arc::new(MemoryProtocol::new(shared_memory));
/// let shared_registry = Arc::new(ToolRegistry::new(shared_adapter));
///
/// let agent1 = Agent::new("a1", "Agent 1", client1).with_tools(shared_registry.clone());
/// let agent2 = Agent::new("a2", "Agent 2", client2).with_tools(shared_registry.clone());
/// ```
///
/// # System Prompt Integration
///
/// Include this in your system prompt to teach agents how to use Memory:
///
/// ```text
/// You have access to a memory tool for persistent state management.
/// Commands:
/// - Store: {"tool_call": {"name": "memory", "parameters": {"command": "P key value ttl"}}}
/// - Retrieve: {"tool_call": {"name": "memory", "parameters": {"command": "G key"}}}
/// - List: {"tool_call": {"name": "memory", "parameters": {"command": "L META"}}}
/// - Get spec: {"tool_call": {"name": "memory", "parameters": {"command": "SPEC"}}}
/// ```
///
/// See [`crate::tools::Memory`] for more details and `examples/MEMORY_TOOL_GUIDE.md` for comprehensive usage guide.
pub struct MemoryProtocol {
    memory: Arc<crate::tools::Memory>,
}

impl MemoryProtocol {
    /// Create a new memory tool adapter bound to a Memory instance
    ///
    /// The adapter will delegate all tool calls to the provided Memory instance.
    /// The Memory instance is typically wrapped in Arc for shared access across agents.
    ///
    /// # Arguments
    ///
    /// * `memory` - The Memory instance to use for storage operations
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use cloudllm::tools::Memory;
    /// use cloudllm::tool_protocols::MemoryProtocol;
    /// use std::sync::Arc;
    ///
    /// let memory = Arc::new(Memory::new());
    /// let adapter = MemoryProtocol::new(memory);
    /// ```
    pub fn new(memory: Arc<crate::tools::Memory>) -> Self {
        Self { memory }
    }
}

#[async_trait]
impl ToolProtocol for MemoryProtocol {
    async fn execute(
        &self,
        tool_name: &str,
        parameters: JsonValue,
    ) -> Result<ToolResult, Box<dyn Error + Send + Sync>> {
        // This adapter only handles the "memory" tool
        if tool_name != "memory" {
            return Err(Box::new(ToolError::NotFound(tool_name.to_string())));
        }

        // Extract the command string.  Agents may pass:
        //   (a) a single "command" field that includes the key/value:
        //         {"command": "G tetris_current_html"}
        //   (b) separate "command"+"key" (and optionally "value") fields:
        //         {"command": "G", "key": "tetris_current_html"}
        //         {"command": "GET", "key": "tetris_current_html"}
        //         {"command": "P", "key": "mykey", "value": "myval"}
        //
        // We normalise both forms into a single command string before parsing.
        let raw_command = parameters
            .get("command")
            .and_then(|v| v.as_str())
            .ok_or_else(|| {
                ToolError::InvalidParameters(
                    "Missing 'command' field. Use e.g. {\"command\": \"G mykey\"} or \
                     {\"command\": \"G\", \"key\": \"mykey\"}"
                        .to_string(),
                )
            })?;

        // Normalise full-word aliases to single-letter codes so the parser
        // works regardless of which form the agent uses.
        let normalised_verb = match raw_command.split_whitespace().next().unwrap_or("") {
            "GET"    => "G",
            "PUT"    => "P",
            "LIST"   => "L",
            "DELETE" => "D",
            "CLEAR"  => "C",
            other    => other,
        };

        // Reconstruct the command string, optionally appending key/value from
        // separate JSON fields when the agent chose the split-parameter form.
        let key_param   = parameters.get("key").and_then(|v| v.as_str()).unwrap_or("");
        let value_param = parameters.get("value").and_then(|v| v.as_str()).unwrap_or("");

        // Rest of the original command after stripping the leading verb token.
        let rest_of_command = raw_command
            .splitn(2, char::is_whitespace)
            .nth(1)
            .unwrap_or("")
            .trim();

        // Build the final canonical command: "<VERB> [key [value]]"
        let command = if !key_param.is_empty() {
            // Agent used the split-parameter style.
            if !value_param.is_empty() {
                format!("{} {} {}", normalised_verb, key_param, value_param)
            } else {
                format!("{} {}", normalised_verb, key_param)
            }
        } else if rest_of_command.is_empty() {
            // Just the verb with no inline key (e.g. "L" or "C")
            normalised_verb.to_string()
        } else {
            // Agent used the inline style, verb already normalised.
            format!("{} {}", normalised_verb, rest_of_command)
        };

        let result = self.process_memory_command(&command);
        Ok(result)
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
        "memory"
    }
}

impl MemoryProtocol {
    /// Build the ToolMetadata for the memory tool with a comprehensive self-contained description.
    ///
    /// The description teaches agents exactly how to call the tool — both the compact
    /// single-letter commands and the more readable full-word aliases are accepted.
    fn memory_tool_metadata(&self) -> ToolMetadata {
        ToolMetadata::new(
            "memory",
            "Shared key-value store for agent coordination. \
             Pass a 'command' string. Supported commands:\n\
             \n\
             READ a value:\n\
               {\"command\": \"G mykey\"}                  — get key 'mykey'\n\
               {\"command\": \"GET\", \"key\": \"mykey\"}  — same, split-param style\n\
             \n\
             WRITE a value (small values only; for large content use write_tetris_file):\n\
               {\"command\": \"P mykey myvalue\"}                           — put without TTL\n\
               {\"command\": \"P\", \"key\": \"mykey\", \"value\": \"val\"} — same, split-param style\n\
               {\"command\": \"P mykey myvalue 3600\"}                      — put with 1-hour TTL\n\
             \n\
             LIST all keys:\n\
               {\"command\": \"L\"}     — list keys\n\
               {\"command\": \"LIST\"}  — same\n\
             \n\
             DELETE a key:\n\
               {\"command\": \"D mykey\"}                   — delete key\n\
               {\"command\": \"DELETE\", \"key\": \"mykey\"} — same\n\
             \n\
             CLEAR everything:\n\
               {\"command\": \"C\"}  or  {\"command\": \"CLEAR\"}\n\
             \n\
             NOTE: PUT stores values as a single token — use write_tetris_file to persist HTML.",
        )
        .with_parameter(
            ToolParameter::new("command", ToolParameterType::String)
                .with_description(
                    "Command string: 'G key', 'P key value [ttl]', 'L', 'D key', 'C'. \
                     Full-word aliases (GET/PUT/LIST/DELETE/CLEAR) also accepted.",
                )
                .required(),
        )
        .with_parameter(
            ToolParameter::new("key", ToolParameterType::String)
                .with_description(
                    "Key for GET/PUT/DELETE when using split-parameter style \
                     instead of embedding the key in the command string.",
                ),
        )
        .with_parameter(
            ToolParameter::new("value", ToolParameterType::String)
                .with_description(
                    "Value for PUT when using split-parameter style. \
                     Must be a single token (no spaces). Use write_tetris_file for HTML content.",
                ),
        )
    }

    /// Process a normalised memory protocol command string.
    ///
    /// Commands (case-sensitive single-letter form after normalisation):
    ///   G key [META]   — get value (optionally with metadata)
    ///   P key value [ttl_secs] — put value
    ///   L [META]       — list keys
    ///   D key          — delete key
    ///   C              — clear all
    ///   T A|K|V        — byte usage (All / Keys / Values)
    ///   SPEC           — return raw protocol spec string
    fn process_memory_command(&self, command: &str) -> ToolResult {
        let parts: Vec<&str> = command.split_whitespace().collect();
        if parts.is_empty() {
            return ToolResult::failure("ERR:Invalid Command".to_string());
        }

        match parts[0] {
            "P" => {
                if parts.len() < 3 {
                    return ToolResult::failure("ERR:Invalid PUT Syntax".to_string());
                }
                let key = parts[1];
                let value = parts[2];
                let ttl = parts.get(3).and_then(|ttl| ttl.parse::<u64>().ok());

                self.memory.put(key.to_string(), value.to_string(), ttl);
                ToolResult::success(serde_json::json!({"status": "OK"}))
            }
            "G" => {
                if parts.len() < 2 {
                    return ToolResult::failure("ERR:Invalid GET Syntax".to_string());
                }
                let key = parts[1];
                let include_metadata = parts.get(2) == Some(&"META");

                match self.memory.get(key, include_metadata) {
                    Some((value, Some(metadata))) => ToolResult::success(serde_json::json!({
                        "value": value,
                        "added_utc": metadata.added_utc.to_rfc3339(),
                        "expires_in": metadata.expires_in
                    })),
                    Some((value, None)) => ToolResult::success(serde_json::json!({
                        "value": value
                    })),
                    None => ToolResult::failure("ERR:NOT_FOUND".to_string()),
                }
            }
            "L" => {
                let include_metadata = parts.get(1) == Some(&"META");
                let keys = self.memory.list_keys();

                if include_metadata {
                    let mut list = Vec::new();
                    for key in &keys {
                        if let Some((_, metadata)) = self.memory.get(key, true) {
                            let metadata = metadata.unwrap();
                            list.push(serde_json::json!({
                                "key": key,
                                "added_utc": metadata.added_utc.to_rfc3339(),
                                "expires_in": metadata.expires_in
                            }));
                        }
                    }
                    ToolResult::success(serde_json::json!({
                        "keys": list
                    }))
                } else {
                    ToolResult::success(serde_json::json!({
                        "keys": keys
                    }))
                }
            }
            "D" => {
                if parts.len() < 2 {
                    return ToolResult::failure("ERR:Invalid DELETE Syntax".to_string());
                }
                let key = parts[1];
                if self.memory.delete(key) {
                    ToolResult::success(serde_json::json!({"status": "OK"}))
                } else {
                    ToolResult::failure("ERR:NOT_FOUND".to_string())
                }
            }
            "C" => {
                self.memory.clear();
                ToolResult::success(serde_json::json!({"status": "OK"}))
            }
            "T" => {
                if parts.len() < 2 {
                    return ToolResult::failure("ERR:Invalid TOTAL Syntax".to_string());
                }
                match parts[1] {
                    "A" => {
                        let (total, _, _) = self.memory.get_total_bytes_stored();
                        ToolResult::success(serde_json::json!({"total_bytes": total}))
                    }
                    "K" => {
                        let (_, keys_size, _) = self.memory.get_total_bytes_stored();
                        ToolResult::success(serde_json::json!({"keys_bytes": keys_size}))
                    }
                    "V" => {
                        let (_, _, values_size) = self.memory.get_total_bytes_stored();
                        ToolResult::success(serde_json::json!({"values_bytes": values_size}))
                    }
                    _ => ToolResult::failure("ERR:Invalid TOTAL Scope".to_string()),
                }
            }
            "SPEC" => ToolResult::success(serde_json::json!({
                "specification": crate::tools::Memory::get_protocol_spec()
            })),
            _ => ToolResult::failure("ERR:Unknown Command".to_string()),
        }
    }
}

/// MCP (Model Context Protocol) Client for Memory Tool
///
/// Provides a client-side adapter that connects to a remote Memory service via MCP protocol.
/// This enables distributed agents to interact with a centralized memory store.
///
/// The MCP Memory Client is designed to work with MCP servers that expose the Memory tool,
/// allowing agents across different processes or machines to share persistent state.
///
/// # Architecture
///
/// The MCP Memory Client interacts with a remote MCP server through standard HTTP:
/// - `GET /tools` - Discover available tools
/// - `POST /execute` - Execute memory commands on the remote server
///
/// # Example
///
/// ```rust,no_run
/// use cloudllm::tool_protocols::McpMemoryProtocol;
/// use cloudllm::tool_protocol::ToolProtocol;
///
/// # async {
/// let client = McpMemoryProtocol::new("http://localhost:8080".to_string());
/// let result = client.execute("memory",
///     serde_json::json!({"command": "P my_key my_value 3600"})).await;
/// # };
/// ```
///
/// # Use Cases
///
/// - **Distributed Agents**: Multiple agents sharing memory across network
/// - **Microservices**: Different services coordinating through shared Memory
/// - **Multi-Region Deployments**: Centralized memory for geographically distributed systems
/// - **Agent Clusters**: Coordinating state across a cluster of agent instances
#[derive(Clone)]
pub struct McpMemoryProtocol {
    /// The base URL of the remote MCP server
    /// Example: "http://localhost:8080"
    endpoint: String,
    /// HTTP client for making requests to the remote server
    client: reqwest::Client,
}

impl McpMemoryProtocol {
    /// Create a new MCP Memory Client connecting to a remote server
    ///
    /// # Arguments
    ///
    /// * `endpoint` - The base URL of the MCP server (e.g., "http://localhost:8080")
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use cloudllm::tool_protocols::McpMemoryProtocol;
    ///
    /// let client = McpMemoryProtocol::new("http://localhost:8080".to_string());
    /// let client_with_custom_port = McpMemoryProtocol::new("http://192.168.1.100:3000".to_string());
    /// ```
    pub fn new(endpoint: String) -> Self {
        Self {
            endpoint,
            client: reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(30))
                .build()
                .expect("Failed to build HTTP client"),
        }
    }

    /// Create an MCP Memory Client with a custom timeout
    ///
    /// # Arguments
    ///
    /// * `endpoint` - The base URL of the MCP server
    /// * `timeout_secs` - Custom timeout in seconds for HTTP requests
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use cloudllm::tool_protocols::McpMemoryProtocol;
    ///
    /// let client = McpMemoryProtocol::with_timeout(
    ///     "http://localhost:8080".to_string(),
    ///     60
    /// );
    /// ```
    pub fn with_timeout(endpoint: String, timeout_secs: u64) -> Self {
        Self {
            endpoint,
            client: reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(timeout_secs))
                .build()
                .expect("Failed to build HTTP client"),
        }
    }

    /// Get the endpoint this client is connected to
    pub fn endpoint(&self) -> &str {
        &self.endpoint
    }

    /// Helper to make HTTP requests to the MCP server
    async fn send_request<T: serde::de::DeserializeOwned>(
        &self,
        path: &str,
        method: &str,
        body: Option<JsonValue>,
    ) -> Result<T, Box<dyn Error + Send + Sync>> {
        let url = format!("{}{}", self.endpoint, path);
        let response = match method {
            "GET" => self.client.get(&url).send().await?,
            "POST" => {
                let mut req = self.client.post(&url);
                if let Some(b) = body {
                    req = req.json(&b);
                }
                req.send().await?
            }
            _ => {
                return Err(Box::new(std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    format!("Unsupported HTTP method: {}", method),
                )) as Box<dyn Error + Send + Sync>)
            }
        };

        if !response.status().is_success() {
            return Err(Box::new(ToolError::ProtocolError(format!(
                "MCP server returned status: {}",
                response.status()
            ))));
        }

        Ok(response.json().await?)
    }
}

#[async_trait]
impl ToolProtocol for McpMemoryProtocol {
    async fn execute(
        &self,
        tool_name: &str,
        parameters: JsonValue,
    ) -> Result<ToolResult, Box<dyn Error + Send + Sync>> {
        let request = serde_json::json!({
            "tool": tool_name,
            "parameters": parameters
        });

        self.send_request("/execute", "POST", Some(request)).await
    }

    async fn list_tools(&self) -> Result<Vec<ToolMetadata>, Box<dyn Error + Send + Sync>> {
        self.send_request("/tools", "GET", None).await
    }

    async fn get_tool_metadata(
        &self,
        tool_name: &str,
    ) -> Result<ToolMetadata, Box<dyn Error + Send + Sync>> {
        let tools = self.list_tools().await?;
        tools
            .into_iter()
            .find(|t| t.name == tool_name)
            .ok_or_else(|| {
                Box::new(ToolError::NotFound(tool_name.to_string())) as Box<dyn Error + Send + Sync>
            })
    }

    fn protocol_name(&self) -> &str {
        "mcp-memory-client"
    }
}

/// Bash Tool Protocol
///
/// Wraps the BashTool to implement the ToolProtocol interface,
/// enabling Bash command execution within the MCP system.
///
/// # Example
///
/// ```rust,no_run
/// use cloudllm::tools::{BashTool, Platform};
/// use cloudllm::tool_protocols::BashProtocol;
/// use std::sync::Arc;
///
/// let bash_tool = Arc::new(BashTool::new(Platform::Linux).with_timeout(30));
/// let protocol = BashProtocol::new(bash_tool);
/// ```
pub struct BashProtocol {
    /// The underlying Bash tool implementation
    bash_tool: Arc<crate::cloudllm::tools::BashTool>,
}

impl BashProtocol {
    /// Create a new Bash protocol wrapping a BashTool
    pub fn new(bash_tool: Arc<crate::cloudllm::tools::BashTool>) -> Self {
        Self { bash_tool }
    }
}

#[async_trait]
impl ToolProtocol for BashProtocol {
    async fn execute(
        &self,
        _tool_name: &str,
        parameters: JsonValue,
    ) -> Result<ToolResult, Box<dyn Error + Send + Sync>> {
        let command = parameters["command"]
            .as_str()
            .ok_or("'command' parameter is required")?;

        match self.bash_tool.execute(command).await {
            Ok(result) => Ok(ToolResult::success(serde_json::json!({
                "stdout": result.stdout,
                "stderr": result.stderr,
                "exit_code": result.exit_code,
                "duration_ms": result.duration_ms
            }))),
            Err(e) => Ok(ToolResult::failure(e.to_string())),
        }
    }

    async fn list_tools(&self) -> Result<Vec<ToolMetadata>, Box<dyn Error + Send + Sync>> {
        Ok(vec![ToolMetadata::new(
            "bash",
            "Execute bash commands on the system (Linux/macOS only)",
        )
        .with_parameter(
            ToolParameter::new("command", ToolParameterType::String)
                .with_description("Bash command to execute (e.g., 'ls -la', 'pwd')")
                .required(),
        )])
    }

    async fn get_tool_metadata(
        &self,
        tool_name: &str,
    ) -> Result<ToolMetadata, Box<dyn Error + Send + Sync>> {
        if tool_name == "bash" {
            Ok(ToolMetadata::new(
                "bash",
                "Execute bash commands on the system (Linux/macOS only)",
            )
            .with_parameter(
                ToolParameter::new("command", ToolParameterType::String)
                    .with_description("Bash command to execute (e.g., 'ls -la', 'pwd')")
                    .required(),
            ))
        } else {
            Err(Box::new(ToolError::NotFound(tool_name.to_string())))
        }
    }

    fn protocol_name(&self) -> &str {
        "bash"
    }
}

/// HTTP Client Protocol for making web requests
///
/// Wraps an [`HttpClient`](crate::cloudllm::tools::HttpClient) to expose HTTP operations
/// (GET, POST, PUT, DELETE, PATCH) as discoverable tools through the [`ToolProtocol`] interface.
///
/// This protocol enables agents to make web requests for research, data fetching, API calls,
/// and coordination with remote services. All HTTP tools return consistent response structures
/// with status, body, and headers.
///
/// # Tools Provided
///
/// - **`http_get`**: Make HTTP GET requests to fetch data
/// - **`http_post`**: Make HTTP POST requests to send data (with optional JSON body)
/// - **`http_put`**: Make HTTP PUT requests to update resources (with optional JSON body)
/// - **`http_delete`**: Make HTTP DELETE requests to remove resources
/// - **`http_patch`**: Make HTTP PATCH requests to partially update resources (with optional JSON body)
///
/// # Tool Parameters
///
/// All HTTP tools require:
/// - **`url`** (string, required): The endpoint to request (e.g., `"https://api.example.com/data"`)
///
/// POST/PUT/PATCH tools additionally support:
/// - **`body`** (string, optional): JSON body to send in the request
///
/// # Response Format
///
/// All HTTP tools return a consistent JSON response:
///
/// ```json
/// {
///   "status": 200,
///   "body": "response content",
///   "headers": {
///     "content-type": "application/json",
///     "content-length": "1234"
///   }
/// }
/// ```
///
/// # Examples
///
/// ## Basic Setup
///
/// ```ignore
/// use cloudllm::tools::HttpClient;
/// use cloudllm::tool_protocols::HttpClientProtocol;
/// use std::sync::Arc;
///
/// let http_client = Arc::new(HttpClient::new());
/// let http_protocol = Arc::new(HttpClientProtocol::new(http_client));
/// ```
///
/// ## With ToolRegistry
///
/// ```ignore
/// use cloudllm::tool_protocol::ToolRegistry;
/// use cloudllm::tools::HttpClient;
/// use cloudllm::tool_protocols::HttpClientProtocol;
/// use std::sync::Arc;
///
/// let http_client = Arc::new(HttpClient::new());
/// let http_protocol = Arc::new(HttpClientProtocol::new(http_client));
///
/// let mut registry = ToolRegistry::empty();
/// registry
///     .add_protocol("http", http_protocol)
///     .await?;
/// ```
///
/// ## Agent Usage Examples
///
/// ### GET Request (Fetch Data)
/// ```json
/// {
///   "tool": "http_get",
///   "parameters": {
///     "url": "https://api.github.com/repos/anthropics/anthropic-sdk-python"
///   }
/// }
/// ```
///
/// ### POST Request (Send Data)
/// ```json
/// {
///   "tool": "http_post",
///   "parameters": {
///     "url": "https://api.example.com/data",
///     "body": "{\"name\": \"test\", \"value\": 42}"
///   }
/// }
/// ```
///
/// ### PUT Request (Update Resource)
/// ```json
/// {
///   "tool": "http_put",
///   "parameters": {
///     "url": "https://api.example.com/users/123",
///     "body": "{\"status\": \"active\"}"
///   }
/// }
/// ```
///
/// ### DELETE Request (Remove Resource)
/// ```json
/// {
///   "tool": "http_delete",
///   "parameters": {
///     "url": "https://api.example.com/users/123"
///   }
/// }
/// ```
///
/// # Security Considerations
///
/// The underlying [`HttpClient`](crate::cloudllm::tools::HttpClient) supports:
/// - Domain allowlist/blocklist for restricting which URLs can be accessed
/// - Configurable timeouts (default 30 seconds)
/// - Response size limits (default 10MB)
/// - Automatic HTTPS enforcement
///
/// # Integration with Agents
///
/// Agents can discover available HTTP tools automatically:
///
/// ```ignore
/// // Agent discovers tools
/// let tools = protocol.list_tools().await?;
/// // Returns: [http_get, http_post, http_put, http_delete, http_patch]
///
/// // Agent makes HTTP request
/// let result = protocol.execute("http_get", json!({"url": "..."})).await?;
/// ```
pub struct HttpClientProtocol {
    /// The underlying HTTP client implementation
    http_client: Arc<crate::cloudllm::tools::HttpClient>,
}

impl HttpClientProtocol {
    /// Create a new HTTP protocol wrapping an HttpClient
    pub fn new(http_client: Arc<crate::cloudllm::tools::HttpClient>) -> Self {
        Self { http_client }
    }
}

#[async_trait]
impl ToolProtocol for HttpClientProtocol {
    async fn execute(
        &self,
        tool_name: &str,
        parameters: JsonValue,
    ) -> Result<ToolResult, Box<dyn Error + Send + Sync>> {
        let url = parameters["url"]
            .as_str()
            .ok_or("'url' parameter is required")?;

        match tool_name {
            "http_get" => match self.http_client.get(url).await {
                Ok(response) => Ok(ToolResult::success(serde_json::json!({
                    "status": response.status,
                    "body": response.body,
                    "headers": response.headers,
                }))),
                Err(e) => Ok(ToolResult::failure(e.to_string())),
            },
            "http_post" => {
                let body = parameters.get("body").cloned().unwrap_or(JsonValue::Null);
                match self.http_client.post(url, body).await {
                    Ok(response) => Ok(ToolResult::success(serde_json::json!({
                        "status": response.status,
                        "body": response.body,
                        "headers": response.headers,
                    }))),
                    Err(e) => Ok(ToolResult::failure(e.to_string())),
                }
            }
            "http_put" => {
                let body = parameters.get("body").cloned().unwrap_or(JsonValue::Null);
                match self.http_client.put(url, body).await {
                    Ok(response) => Ok(ToolResult::success(serde_json::json!({
                        "status": response.status,
                        "body": response.body,
                        "headers": response.headers,
                    }))),
                    Err(e) => Ok(ToolResult::failure(e.to_string())),
                }
            }
            "http_delete" => match self.http_client.delete(url).await {
                Ok(response) => Ok(ToolResult::success(serde_json::json!({
                    "status": response.status,
                    "body": response.body,
                    "headers": response.headers,
                }))),
                Err(e) => Ok(ToolResult::failure(e.to_string())),
            },
            "http_patch" => {
                let body = parameters.get("body").cloned().unwrap_or(JsonValue::Null);
                match self.http_client.patch(url, body).await {
                    Ok(response) => Ok(ToolResult::success(serde_json::json!({
                        "status": response.status,
                        "body": response.body,
                        "headers": response.headers,
                    }))),
                    Err(e) => Ok(ToolResult::failure(e.to_string())),
                }
            }
            _ => Err(Box::new(ToolError::NotFound(tool_name.to_string()))),
        }
    }

    async fn list_tools(&self) -> Result<Vec<ToolMetadata>, Box<dyn Error + Send + Sync>> {
        Ok(vec![
            ToolMetadata::new("http_get", "Make an HTTP GET request").with_parameter(
                ToolParameter::new("url", ToolParameterType::String)
                    .with_description("The URL to request (e.g., 'https://api.example.com/data')")
                    .required(),
            ),
            ToolMetadata::new("http_post", "Make an HTTP POST request")
                .with_parameter(
                    ToolParameter::new("url", ToolParameterType::String)
                        .with_description("The URL to request")
                        .required(),
                )
                .with_parameter(
                    ToolParameter::new("body", ToolParameterType::String)
                        .with_description("JSON body to send (optional)"),
                ),
            ToolMetadata::new("http_put", "Make an HTTP PUT request")
                .with_parameter(
                    ToolParameter::new("url", ToolParameterType::String)
                        .with_description("The URL to request")
                        .required(),
                )
                .with_parameter(
                    ToolParameter::new("body", ToolParameterType::String)
                        .with_description("JSON body to send (optional)"),
                ),
            ToolMetadata::new("http_delete", "Make an HTTP DELETE request").with_parameter(
                ToolParameter::new("url", ToolParameterType::String)
                    .with_description("The URL to request")
                    .required(),
            ),
            ToolMetadata::new("http_patch", "Make an HTTP PATCH request")
                .with_parameter(
                    ToolParameter::new("url", ToolParameterType::String)
                        .with_description("The URL to request")
                        .required(),
                )
                .with_parameter(
                    ToolParameter::new("body", ToolParameterType::String)
                        .with_description("JSON body to send (optional)"),
                ),
        ])
    }

    async fn get_tool_metadata(
        &self,
        tool_name: &str,
    ) -> Result<ToolMetadata, Box<dyn Error + Send + Sync>> {
        match tool_name {
            "http_get" => Ok(ToolMetadata::new("http_get", "Make an HTTP GET request")
                .with_parameter(
                    ToolParameter::new("url", ToolParameterType::String)
                        .with_description(
                            "The URL to request (e.g., 'https://api.example.com/data')",
                        )
                        .required(),
                )),
            "http_post" => Ok(ToolMetadata::new("http_post", "Make an HTTP POST request")
                .with_parameter(
                    ToolParameter::new("url", ToolParameterType::String)
                        .with_description("The URL to request")
                        .required(),
                )
                .with_parameter(
                    ToolParameter::new("body", ToolParameterType::String)
                        .with_description("JSON body to send (optional)"),
                )),
            "http_put" => Ok(ToolMetadata::new("http_put", "Make an HTTP PUT request")
                .with_parameter(
                    ToolParameter::new("url", ToolParameterType::String)
                        .with_description("The URL to request")
                        .required(),
                )
                .with_parameter(
                    ToolParameter::new("body", ToolParameterType::String)
                        .with_description("JSON body to send (optional)"),
                )),
            "http_delete" => Ok(
                ToolMetadata::new("http_delete", "Make an HTTP DELETE request").with_parameter(
                    ToolParameter::new("url", ToolParameterType::String)
                        .with_description("The URL to request")
                        .required(),
                ),
            ),
            "http_patch" => Ok(
                ToolMetadata::new("http_patch", "Make an HTTP PATCH request")
                    .with_parameter(
                        ToolParameter::new("url", ToolParameterType::String)
                            .with_description("The URL to request")
                            .required(),
                    )
                    .with_parameter(
                        ToolParameter::new("body", ToolParameterType::String)
                            .with_description("JSON body to send (optional)"),
                    ),
            ),
            _ => Err(Box::new(ToolError::NotFound(tool_name.to_string()))),
        }
    }

    fn protocol_name(&self) -> &str {
        "http"
    }
}
