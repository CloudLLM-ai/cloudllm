#![allow(dead_code)]

use cloudllm::cloudllm::mcp_http_adapter::HttpServerInstance;
use cloudllm::cloudllm::mcp_server_builder::MCPServerBuilder;
use cloudllm::tool_protocol::{
    ToolError, ToolMetadata, ToolParameter, ToolParameterType, ToolProtocol, ToolRegistry,
    ToolResult,
};
use cloudllm::tool_protocols::{
    BashProtocol, CustomToolProtocol, HttpClientProtocol, McpClientProtocol, MemoryProtocol,
};
use cloudllm::tools::{BashTool, Calculator, FileSystemTool, HttpClient, Memory, Platform};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::error::Error;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;
use thoughtchain::{Thought, ThoughtChain, ThoughtInput, ThoughtQuery, ThoughtRole, ThoughtType};
use tokio::sync::RwLock;

pub struct ThoughtChainMcpConfig {
    pub chain_dir: PathBuf,
    pub default_chain_key: String,
}

#[derive(Clone)]
pub struct ThoughtChainProtocol {
    chain_dir: PathBuf,
    default_chain_key: String,
    chains: Arc<RwLock<HashMap<String, Arc<RwLock<ThoughtChain>>>>>,
}

impl ThoughtChainProtocol {
    pub fn new(config: ThoughtChainMcpConfig) -> Self {
        Self {
            chain_dir: config.chain_dir,
            default_chain_key: config.default_chain_key,
            chains: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    async fn get_chain(
        &self,
        chain_key: Option<&str>,
    ) -> Result<Arc<RwLock<ThoughtChain>>, Box<dyn Error + Send + Sync>> {
        let chain_key = chain_key.unwrap_or(&self.default_chain_key).to_string();

        if let Some(existing) = self.chains.read().await.get(&chain_key).cloned() {
            return Ok(existing);
        }

        let chain = Arc::new(RwLock::new(ThoughtChain::open_with_key(
            &self.chain_dir,
            &chain_key,
        )?));

        let mut chains = self.chains.write().await;
        let entry = chains.entry(chain_key).or_insert_with(|| chain.clone());
        Ok(entry.clone())
    }

    fn tool_metadata() -> Vec<ToolMetadata> {
        vec![
            ToolMetadata::new(
                "thoughtchain_bootstrap",
                "Ensure a thought chain exists and initialize it the first time with a bootstrap memory.",
            )
            .with_parameter(
                ToolParameter::new("chain_key", ToolParameterType::String)
                    .with_description("Optional durable chain key. Defaults to the server's default chain."),
            )
            .with_parameter(
                ToolParameter::new("content", ToolParameterType::String)
                    .with_description("Bootstrap summary to store if the chain is empty.")
                    .required(),
            )
            .with_parameter(
                ToolParameter::new("importance", ToolParameterType::Number)
                    .with_description("Optional importance score between 0.0 and 1.0."),
            )
            .with_parameter(
                ToolParameter::new("tags", ToolParameterType::Array)
                    .with_description("Optional tags for the bootstrap memory.")
                    .with_items(ToolParameterType::String),
            )
            .with_parameter(
                ToolParameter::new("concepts", ToolParameterType::Array)
                    .with_description("Optional concepts for the bootstrap memory.")
                    .with_items(ToolParameterType::String),
            ),
            ToolMetadata::new(
                "thoughtchain_append",
                "Append a durable semantic memory to ThoughtChain. Use exact ThoughtType names like PreferenceUpdate, Constraint, Decision, Insight, Question, Summary, Mistake, or Correction.",
            )
            .with_parameter(
                ToolParameter::new("chain_key", ToolParameterType::String)
                    .with_description("Optional durable chain key."),
            )
            .with_parameter(
                ToolParameter::new("thought_type", ToolParameterType::String)
                    .with_description("Semantic type of the thought.")
                    .required(),
            )
            .with_parameter(
                ToolParameter::new("content", ToolParameterType::String)
                    .with_description("Concise durable memory content.")
                    .required(),
            )
            .with_parameter(
                ToolParameter::new("role", ToolParameterType::String)
                    .with_description("Optional thought role such as Memory, Summary, Compression, Checkpoint, or Handoff."),
            )
            .with_parameter(
                ToolParameter::new("importance", ToolParameterType::Number)
                    .with_description("Optional importance score between 0.0 and 1.0."),
            )
            .with_parameter(
                ToolParameter::new("confidence", ToolParameterType::Number)
                    .with_description("Optional confidence score between 0.0 and 1.0."),
            )
            .with_parameter(
                ToolParameter::new("tags", ToolParameterType::Array)
                    .with_description("Optional tags.")
                    .with_items(ToolParameterType::String),
            )
            .with_parameter(
                ToolParameter::new("concepts", ToolParameterType::Array)
                    .with_description("Optional semantic concepts.")
                    .with_items(ToolParameterType::String),
            )
            .with_parameter(
                ToolParameter::new("refs", ToolParameterType::Array)
                    .with_description("Optional referenced thought indices.")
                    .with_items(ToolParameterType::Integer),
            ),
            ToolMetadata::new(
                "thoughtchain_search",
                "Search durable memories by text, type, role, tags, concepts, and importance.",
            )
            .with_parameter(
                ToolParameter::new("chain_key", ToolParameterType::String)
                    .with_description("Optional durable chain key."),
            )
            .with_parameter(
                ToolParameter::new("text", ToolParameterType::String)
                    .with_description("Optional text filter applied to content, tags, and concepts."),
            )
            .with_parameter(
                ToolParameter::new("thought_types", ToolParameterType::Array)
                    .with_description("Optional list of ThoughtType names.")
                    .with_items(ToolParameterType::String),
            )
            .with_parameter(
                ToolParameter::new("roles", ToolParameterType::Array)
                    .with_description("Optional list of ThoughtRole names.")
                    .with_items(ToolParameterType::String),
            )
            .with_parameter(
                ToolParameter::new("tags_any", ToolParameterType::Array)
                    .with_description("Optional tags to match.")
                    .with_items(ToolParameterType::String),
            )
            .with_parameter(
                ToolParameter::new("concepts_any", ToolParameterType::Array)
                    .with_description("Optional concepts to match.")
                    .with_items(ToolParameterType::String),
            )
            .with_parameter(
                ToolParameter::new("min_importance", ToolParameterType::Number)
                    .with_description("Optional minimum importance threshold."),
            )
            .with_parameter(
                ToolParameter::new("limit", ToolParameterType::Integer)
                    .with_description("Optional maximum number of results."),
            ),
            ToolMetadata::new(
                "thoughtchain_recent_context",
                "Render recent ThoughtChain context as a prompt snippet suitable for resuming work.",
            )
            .with_parameter(
                ToolParameter::new("chain_key", ToolParameterType::String)
                    .with_description("Optional durable chain key."),
            )
            .with_parameter(
                ToolParameter::new("last_n", ToolParameterType::Integer)
                    .with_description("How many recent thoughts to include."),
            ),
            ToolMetadata::new(
                "thoughtchain_memory_markdown",
                "Export a MEMORY.md style Markdown summary from ThoughtChain.",
            )
            .with_parameter(
                ToolParameter::new("chain_key", ToolParameterType::String)
                    .with_description("Optional durable chain key."),
            )
            .with_parameter(
                ToolParameter::new("text", ToolParameterType::String)
                    .with_description("Optional text filter."),
            )
            .with_parameter(
                ToolParameter::new("thought_types", ToolParameterType::Array)
                    .with_description("Optional list of ThoughtType names.")
                    .with_items(ToolParameterType::String),
            )
            .with_parameter(
                ToolParameter::new("roles", ToolParameterType::Array)
                    .with_description("Optional list of ThoughtRole names.")
                    .with_items(ToolParameterType::String),
            )
            .with_parameter(
                ToolParameter::new("tags_any", ToolParameterType::Array)
                    .with_description("Optional tags to match.")
                    .with_items(ToolParameterType::String),
            )
            .with_parameter(
                ToolParameter::new("concepts_any", ToolParameterType::Array)
                    .with_description("Optional concepts to match.")
                    .with_items(ToolParameterType::String),
            )
            .with_parameter(
                ToolParameter::new("min_importance", ToolParameterType::Number)
                    .with_description("Optional minimum importance threshold."),
            )
            .with_parameter(
                ToolParameter::new("limit", ToolParameterType::Integer)
                    .with_description("Optional maximum number of thoughts."),
            ),
            ToolMetadata::new(
                "thoughtchain_head",
                "Return head metadata for a ThoughtChain including length, latest thought, and head hash.",
            )
            .with_parameter(
                ToolParameter::new("chain_key", ToolParameterType::String)
                    .with_description("Optional durable chain key."),
            ),
        ]
    }
}

#[async_trait::async_trait]
impl ToolProtocol for ThoughtChainProtocol {
    async fn execute(
        &self,
        tool_name: &str,
        parameters: Value,
    ) -> Result<ToolResult, Box<dyn Error + Send + Sync>> {
        let chain_key = parameters.get("chain_key").and_then(Value::as_str);
        let chain = self.get_chain(chain_key).await?;

        match tool_name {
            "thoughtchain_bootstrap" => {
                let content = required_string(&parameters, "content")?;
                let importance = optional_f32(&parameters, "importance").unwrap_or(1.0);
                let tags = optional_string_vec(&parameters, "tags");
                let concepts = optional_string_vec(&parameters, "concepts");

                let mut chain = chain.write().await;
                let bootstrapped = if chain.thoughts().is_empty() {
                    let input = ThoughtInput::new(ThoughtType::Summary, content)
                        .with_role(ThoughtRole::Checkpoint)
                        .with_importance(importance)
                        .with_tags(tags)
                        .with_concepts(concepts);
                    chain.append_thought(chain_key.unwrap_or(&self.default_chain_key), input)?;
                    true
                } else {
                    false
                };

                Ok(ToolResult::success(json!({
                    "bootstrapped": bootstrapped,
                    "thought_count": chain.thoughts().len(),
                    "head_hash": chain.head_hash(),
                })))
            }
            "thoughtchain_append" => {
                let thought_type =
                    parse_thought_type(&required_string(&parameters, "thought_type")?)?;
                let role = match parameters.get("role").and_then(Value::as_str) {
                    Some(role) => parse_thought_role(role)?,
                    None => ThoughtRole::Memory,
                };
                let content = required_string(&parameters, "content")?;
                let importance = optional_f32(&parameters, "importance").unwrap_or(0.5);
                let confidence = optional_f32(&parameters, "confidence");
                let tags = optional_string_vec(&parameters, "tags");
                let concepts = optional_string_vec(&parameters, "concepts");
                let refs = optional_u64_vec(&parameters, "refs");

                let mut input = ThoughtInput::new(thought_type, content)
                    .with_role(role)
                    .with_importance(importance)
                    .with_tags(tags)
                    .with_concepts(concepts)
                    .with_refs(refs);
                if let Some(confidence) = confidence {
                    input = input.with_confidence(confidence);
                }

                let mut chain = chain.write().await;
                let thought =
                    chain.append_thought(chain_key.unwrap_or(&self.default_chain_key), input)?;
                Ok(ToolResult::success(json!({
                    "thought": thought_to_json(thought),
                    "head_hash": chain.head_hash(),
                })))
            }
            "thoughtchain_search" => {
                let chain = chain.read().await;
                let query = build_query(&parameters)?;
                let thoughts = chain
                    .query(&query)
                    .into_iter()
                    .map(thought_to_json)
                    .collect::<Vec<_>>();
                Ok(ToolResult::success(json!({ "thoughts": thoughts })))
            }
            "thoughtchain_recent_context" => {
                let chain = chain.read().await;
                let last_n = parameters
                    .get("last_n")
                    .and_then(Value::as_u64)
                    .unwrap_or(12) as usize;
                Ok(ToolResult::success(json!({
                    "prompt": chain.to_catchup_prompt(last_n),
                })))
            }
            "thoughtchain_memory_markdown" => {
                let chain = chain.read().await;
                let query = build_query(&parameters)?;
                let markdown = if query_is_empty(&query) {
                    chain.to_memory_markdown(None)
                } else {
                    chain.to_memory_markdown(Some(&query))
                };
                Ok(ToolResult::success(json!({ "markdown": markdown })))
            }
            "thoughtchain_head" => {
                let chain = chain.read().await;
                Ok(ToolResult::success(json!({
                    "chain_key": chain_key.unwrap_or(&self.default_chain_key),
                    "thought_count": chain.thoughts().len(),
                    "head_hash": chain.head_hash(),
                    "latest_thought": chain.thoughts().last().map(thought_to_json),
                    "integrity_ok": chain.verify_integrity(),
                    "file_path": chain.file_path().display().to_string(),
                })))
            }
            _ => Err(Box::new(ToolError::NotFound(tool_name.to_string()))),
        }
    }

    async fn list_tools(&self) -> Result<Vec<ToolMetadata>, Box<dyn Error + Send + Sync>> {
        Ok(Self::tool_metadata())
    }

    async fn get_tool_metadata(
        &self,
        tool_name: &str,
    ) -> Result<ToolMetadata, Box<dyn Error + Send + Sync>> {
        Self::tool_metadata()
            .into_iter()
            .find(|tool| tool.name == tool_name)
            .ok_or_else(|| {
                Box::new(ToolError::NotFound(tool_name.to_string())) as Box<dyn Error + Send + Sync>
            })
    }

    fn protocol_name(&self) -> &str {
        "thoughtchain"
    }
}

pub async fn start_thoughtchain_mcp_server(
    addr: SocketAddr,
    config: ThoughtChainMcpConfig,
) -> Result<HttpServerInstance, Box<dyn Error + Send + Sync>> {
    let protocol = Arc::new(ThoughtChainProtocol::new(config));

    let builder = MCPServerBuilder::new().allow_localhost_only();
    let builder = builder
        .with_custom_tool("thoughtchain_bootstrap", protocol.clone())
        .await;
    let builder = builder
        .with_custom_tool("thoughtchain_append", protocol.clone())
        .await;
    let builder = builder
        .with_custom_tool("thoughtchain_search", protocol.clone())
        .await;
    let builder = builder
        .with_custom_tool("thoughtchain_recent_context", protocol.clone())
        .await;
    let builder = builder
        .with_custom_tool("thoughtchain_memory_markdown", protocol.clone())
        .await;
    let builder = builder
        .with_custom_tool("thoughtchain_head", protocol.clone())
        .await;

    builder.start_at(addr).await
}

pub async fn build_persistent_agent_registry(
    thoughtchain_endpoint: &str,
    filesystem_root: PathBuf,
) -> Result<(ToolRegistry, Arc<McpClientProtocol>), Box<dyn Error + Send + Sync>> {
    let thoughtchain_protocol =
        Arc::new(McpClientProtocol::new(thoughtchain_endpoint.to_string()).with_cache_ttl(30));

    let mut registry = ToolRegistry::empty();
    registry
        .add_protocol("thoughtchain", thoughtchain_protocol.clone())
        .await?;

    let memory = Arc::new(Memory::new());
    registry
        .add_protocol("memory", Arc::new(MemoryProtocol::new(memory)))
        .await?;

    let bash_tool = Arc::new(BashTool::new(detect_platform()).with_timeout(30));
    registry
        .add_protocol("bash", Arc::new(BashProtocol::new(bash_tool)))
        .await?;

    let http_client = Arc::new(HttpClient::new());
    registry
        .add_protocol("http", Arc::new(HttpClientProtocol::new(http_client)))
        .await?;

    let custom = Arc::new(CustomToolProtocol::new());
    register_calculator_tool(custom.clone()).await;
    register_filesystem_tools(custom.clone(), filesystem_root).await;
    registry.add_protocol("custom", custom).await?;

    Ok((registry, thoughtchain_protocol))
}

async fn register_calculator_tool(protocol: Arc<CustomToolProtocol>) {
    let calculator = Arc::new(Calculator::new());
    protocol
        .register_async_tool(
            ToolMetadata::new("calculator", "Evaluate a mathematical expression.").with_parameter(
                ToolParameter::new("expression", ToolParameterType::String)
                    .with_description("Expression to evaluate, e.g. 'sqrt(16) + mean([1,2,3])'.")
                    .required(),
            ),
            Arc::new(move |params| {
                let calculator = calculator.clone();
                Box::pin(async move {
                    let expression = required_string(&params, "expression")?;
                    match calculator.evaluate(&expression).await {
                        Ok(result) => Ok(ToolResult::success(json!({ "result": result }))),
                        Err(error) => Ok(ToolResult::failure(error.to_string())),
                    }
                })
            }),
        )
        .await;
}

async fn register_filesystem_tools(protocol: Arc<CustomToolProtocol>, filesystem_root: PathBuf) {
    let filesystem = Arc::new(FileSystemTool::new().with_root_path(filesystem_root.clone()));

    protocol
        .register_async_tool(
            ToolMetadata::new(
                "read_file",
                "Read a text file inside the configured workspace root.",
            )
            .with_parameter(
                ToolParameter::new("path", ToolParameterType::String)
                    .with_description("Relative path to the file.")
                    .required(),
            ),
            Arc::new(move |params| {
                let filesystem = filesystem.clone();
                Box::pin(async move {
                    let path = required_string(&params, "path")?;
                    match filesystem.read_file(&path).await {
                        Ok(content) => Ok(ToolResult::success(json!({ "content": content }))),
                        Err(error) => Ok(ToolResult::failure(error.to_string())),
                    }
                })
            }),
        )
        .await;

    let filesystem = Arc::new(FileSystemTool::new().with_root_path(filesystem_root.clone()));
    protocol
        .register_async_tool(
            ToolMetadata::new(
                "write_file",
                "Write a text file inside the configured workspace root.",
            )
            .with_parameter(
                ToolParameter::new("path", ToolParameterType::String)
                    .with_description("Relative path to the file.")
                    .required(),
            )
            .with_parameter(
                ToolParameter::new("content", ToolParameterType::String)
                    .with_description("Text content to write.")
                    .required(),
            ),
            Arc::new(move |params| {
                let filesystem = filesystem.clone();
                Box::pin(async move {
                    let path = required_string(&params, "path")?;
                    let content = required_string(&params, "content")?;
                    match filesystem.write_file(&path, &content).await {
                        Ok(()) => Ok(ToolResult::success(json!({ "status": "OK" }))),
                        Err(error) => Ok(ToolResult::failure(error.to_string())),
                    }
                })
            }),
        )
        .await;

    let filesystem = Arc::new(FileSystemTool::new().with_root_path(filesystem_root.clone()));
    protocol
        .register_async_tool(
            ToolMetadata::new(
                "append_file",
                "Append text to a file inside the configured workspace root.",
            )
            .with_parameter(
                ToolParameter::new("path", ToolParameterType::String)
                    .with_description("Relative path to the file.")
                    .required(),
            )
            .with_parameter(
                ToolParameter::new("content", ToolParameterType::String)
                    .with_description("Text content to append.")
                    .required(),
            ),
            Arc::new(move |params| {
                let filesystem = filesystem.clone();
                Box::pin(async move {
                    let path = required_string(&params, "path")?;
                    let content = required_string(&params, "content")?;
                    match filesystem.append_file(&path, &content).await {
                        Ok(()) => Ok(ToolResult::success(json!({ "status": "OK" }))),
                        Err(error) => Ok(ToolResult::failure(error.to_string())),
                    }
                })
            }),
        )
        .await;

    let filesystem = Arc::new(FileSystemTool::new().with_root_path(filesystem_root.clone()));
    protocol
        .register_async_tool(
            ToolMetadata::new(
                "list_directory",
                "List a directory inside the configured workspace root.",
            )
            .with_parameter(
                ToolParameter::new("path", ToolParameterType::String)
                    .with_description("Relative path to the directory.")
                    .required(),
            )
            .with_parameter(
                ToolParameter::new("recursive", ToolParameterType::Boolean)
                    .with_description("Whether to recurse into subdirectories."),
            ),
            Arc::new(move |params| {
                let filesystem = filesystem.clone();
                Box::pin(async move {
                    let path = required_string(&params, "path")?;
                    let recursive = params
                        .get("recursive")
                        .and_then(Value::as_bool)
                        .unwrap_or(false);
                    match filesystem.read_directory(&path, recursive).await {
                        Ok(entries) => Ok(ToolResult::success(json!({
                            "entries": entries
                                .iter()
                                .map(directory_entry_to_json)
                                .collect::<Vec<_>>()
                        }))),
                        Err(error) => Ok(ToolResult::failure(error.to_string())),
                    }
                })
            }),
        )
        .await;

    let filesystem = Arc::new(FileSystemTool::new().with_root_path(filesystem_root.clone()));
    protocol
        .register_async_tool(
            ToolMetadata::new(
                "file_metadata",
                "Return file metadata inside the configured workspace root.",
            )
            .with_parameter(
                ToolParameter::new("path", ToolParameterType::String)
                    .with_description("Relative path to the file.")
                    .required(),
            ),
            Arc::new(move |params| {
                let filesystem = filesystem.clone();
                Box::pin(async move {
                    let path = required_string(&params, "path")?;
                    match filesystem.get_file_metadata(&path).await {
                        Ok(metadata) => Ok(ToolResult::success(json!({
                            "metadata": file_metadata_to_json(&metadata)
                        }))),
                        Err(error) => Ok(ToolResult::failure(error.to_string())),
                    }
                })
            }),
        )
        .await;

    let filesystem = Arc::new(FileSystemTool::new().with_root_path(filesystem_root.clone()));
    protocol
        .register_async_tool(
            ToolMetadata::new(
                "create_directory",
                "Create a directory inside the configured workspace root.",
            )
            .with_parameter(
                ToolParameter::new("path", ToolParameterType::String)
                    .with_description("Relative path to the directory.")
                    .required(),
            ),
            Arc::new(move |params| {
                let filesystem = filesystem.clone();
                Box::pin(async move {
                    let path = required_string(&params, "path")?;
                    match filesystem.create_directory(&path).await {
                        Ok(()) => Ok(ToolResult::success(json!({ "status": "OK" }))),
                        Err(error) => Ok(ToolResult::failure(error.to_string())),
                    }
                })
            }),
        )
        .await;

    let filesystem = Arc::new(FileSystemTool::new().with_root_path(filesystem_root.clone()));
    protocol
        .register_async_tool(
            ToolMetadata::new(
                "delete_file",
                "Delete a file inside the configured workspace root.",
            )
            .with_parameter(
                ToolParameter::new("path", ToolParameterType::String)
                    .with_description("Relative path to the file.")
                    .required(),
            ),
            Arc::new(move |params| {
                let filesystem = filesystem.clone();
                Box::pin(async move {
                    let path = required_string(&params, "path")?;
                    match filesystem.delete_file(&path).await {
                        Ok(()) => Ok(ToolResult::success(json!({ "status": "OK" }))),
                        Err(error) => Ok(ToolResult::failure(error.to_string())),
                    }
                })
            }),
        )
        .await;

    let filesystem = Arc::new(FileSystemTool::new().with_root_path(filesystem_root.clone()));
    protocol
        .register_async_tool(
            ToolMetadata::new(
                "delete_directory",
                "Delete a directory inside the configured workspace root.",
            )
            .with_parameter(
                ToolParameter::new("path", ToolParameterType::String)
                    .with_description("Relative path to the directory.")
                    .required(),
            ),
            Arc::new(move |params| {
                let filesystem = filesystem.clone();
                Box::pin(async move {
                    let path = required_string(&params, "path")?;
                    match filesystem.delete_directory(&path).await {
                        Ok(()) => Ok(ToolResult::success(json!({ "status": "OK" }))),
                        Err(error) => Ok(ToolResult::failure(error.to_string())),
                    }
                })
            }),
        )
        .await;

    let filesystem = Arc::new(FileSystemTool::new().with_root_path(filesystem_root.clone()));
    protocol
        .register_async_tool(
            ToolMetadata::new(
                "search_files",
                "Search for files by substring within the configured workspace root.",
            )
            .with_parameter(
                ToolParameter::new("directory", ToolParameterType::String)
                    .with_description("Relative path to the directory to search.")
                    .required(),
            )
            .with_parameter(
                ToolParameter::new("pattern", ToolParameterType::String)
                    .with_description("Substring to search for in file names.")
                    .required(),
            ),
            Arc::new(move |params| {
                let filesystem = filesystem.clone();
                Box::pin(async move {
                    let directory = required_string(&params, "directory")?;
                    let pattern = required_string(&params, "pattern")?;
                    match filesystem.search_files(&directory, &pattern).await {
                        Ok(entries) => Ok(ToolResult::success(json!({
                            "entries": entries
                                .iter()
                                .map(directory_entry_to_json)
                                .collect::<Vec<_>>()
                        }))),
                        Err(error) => Ok(ToolResult::failure(error.to_string())),
                    }
                })
            }),
        )
        .await;

    let filesystem = Arc::new(FileSystemTool::new().with_root_path(filesystem_root.clone()));
    protocol
        .register_async_tool(
            ToolMetadata::new(
                "file_exists",
                "Check whether a path exists inside the configured workspace root.",
            )
            .with_parameter(
                ToolParameter::new("path", ToolParameterType::String)
                    .with_description("Relative path to check.")
                    .required(),
            ),
            Arc::new(move |params| {
                let filesystem = filesystem.clone();
                Box::pin(async move {
                    let path = required_string(&params, "path")?;
                    match filesystem.file_exists(&path).await {
                        Ok(exists) => Ok(ToolResult::success(json!({ "exists": exists }))),
                        Err(error) => Ok(ToolResult::failure(error.to_string())),
                    }
                })
            }),
        )
        .await;

    let filesystem = Arc::new(FileSystemTool::new().with_root_path(filesystem_root));
    protocol
        .register_async_tool(
            ToolMetadata::new(
                "file_size",
                "Return a file size in bytes inside the configured workspace root.",
            )
            .with_parameter(
                ToolParameter::new("path", ToolParameterType::String)
                    .with_description("Relative path to the file.")
                    .required(),
            ),
            Arc::new(move |params| {
                let filesystem = filesystem.clone();
                Box::pin(async move {
                    let path = required_string(&params, "path")?;
                    match filesystem.get_file_size(&path).await {
                        Ok(size) => Ok(ToolResult::success(json!({ "size": size }))),
                        Err(error) => Ok(ToolResult::failure(error.to_string())),
                    }
                })
            }),
        )
        .await;
}

pub fn default_thoughtchain_dir() -> PathBuf {
    if let Some(home) = std::env::var_os("HOME") {
        PathBuf::from(home).join(".cloudllm").join("thoughtchain")
    } else {
        std::env::current_dir()
            .unwrap_or_else(|_| PathBuf::from("."))
            .join(".cloudllm")
            .join("thoughtchain")
    }
}

fn thought_to_json(thought: &Thought) -> Value {
    serde_json::to_value(thought).unwrap_or_else(|_| {
        json!({
            "index": thought.index,
            "content": thought.content,
        })
    })
}

fn directory_entry_to_json(entry: &cloudllm::tools::DirectoryEntry) -> Value {
    json!({
        "name": entry.name,
        "is_directory": entry.is_directory,
        "size": entry.size,
    })
}

fn file_metadata_to_json(metadata: &cloudllm::tools::FileMetadata) -> Value {
    json!({
        "name": metadata.name,
        "path": metadata.path,
        "size": metadata.size,
        "is_directory": metadata.is_directory,
        "modified": metadata.modified,
    })
}

fn query_is_empty(query: &ThoughtQuery) -> bool {
    query.thought_types.is_none()
        && query.roles.is_none()
        && query.agent_ids.is_none()
        && query.tags_any.is_empty()
        && query.concepts_any.is_empty()
        && query.text_contains.is_none()
        && query.min_importance.is_none()
        && query.min_confidence.is_none()
        && query.since.is_none()
        && query.until.is_none()
        && query.limit.is_none()
}

fn build_query(parameters: &Value) -> Result<ThoughtQuery, Box<dyn Error + Send + Sync>> {
    let mut query = ThoughtQuery::new();

    if let Some(text) = parameters.get("text").and_then(Value::as_str) {
        query = query.with_text(text.to_string());
    }
    if let Some(min_importance) = optional_f32(parameters, "min_importance") {
        query = query.with_min_importance(min_importance);
    }
    if let Some(limit) = parameters.get("limit").and_then(Value::as_u64) {
        query = query.with_limit(limit as usize);
    }

    let thought_types = optional_string_vec(parameters, "thought_types")
        .into_iter()
        .map(|value| parse_thought_type(&value))
        .collect::<Result<Vec<_>, _>>()?;
    if !thought_types.is_empty() {
        query = query.with_types(thought_types);
    }

    let roles = optional_string_vec(parameters, "roles")
        .into_iter()
        .map(|value| parse_thought_role(&value))
        .collect::<Result<Vec<_>, _>>()?;
    if !roles.is_empty() {
        query = query.with_roles(roles);
    }

    let tags = optional_string_vec(parameters, "tags_any");
    if !tags.is_empty() {
        query = query.with_tags_any(tags);
    }

    let concepts = optional_string_vec(parameters, "concepts_any");
    if !concepts.is_empty() {
        query = query.with_concepts_any(concepts);
    }

    Ok(query)
}

fn required_string(parameters: &Value, key: &str) -> Result<String, Box<dyn Error + Send + Sync>> {
    parameters
        .get(key)
        .and_then(Value::as_str)
        .map(ToOwned::to_owned)
        .ok_or_else(|| {
            Box::new(ToolError::InvalidParameters(format!(
                "'{key}' parameter is required"
            ))) as Box<dyn Error + Send + Sync>
        })
}

fn optional_string_vec(parameters: &Value, key: &str) -> Vec<String> {
    parameters
        .get(key)
        .and_then(Value::as_array)
        .map(|values| {
            values
                .iter()
                .filter_map(Value::as_str)
                .map(ToOwned::to_owned)
                .collect()
        })
        .unwrap_or_default()
}

fn optional_u64_vec(parameters: &Value, key: &str) -> Vec<u64> {
    parameters
        .get(key)
        .and_then(Value::as_array)
        .map(|values| values.iter().filter_map(Value::as_u64).collect())
        .unwrap_or_default()
}

fn optional_f32(parameters: &Value, key: &str) -> Option<f32> {
    parameters
        .get(key)
        .and_then(Value::as_f64)
        .map(|value| value as f32)
}

fn detect_platform() -> Platform {
    #[cfg(target_os = "macos")]
    {
        Platform::macOS
    }
    #[cfg(not(target_os = "macos"))]
    {
        Platform::Linux
    }
}

fn parse_thought_type(input: &str) -> Result<ThoughtType, Box<dyn Error + Send + Sync>> {
    let normalized = normalize_label(input);
    let thought_type = match normalized.as_str() {
        "preferenceupdate" => ThoughtType::PreferenceUpdate,
        "usertrait" => ThoughtType::UserTrait,
        "relationshipupdate" => ThoughtType::RelationshipUpdate,
        "finding" => ThoughtType::Finding,
        "insight" => ThoughtType::Insight,
        "factlearned" => ThoughtType::FactLearned,
        "patterndetected" => ThoughtType::PatternDetected,
        "hypothesis" => ThoughtType::Hypothesis,
        "mistake" => ThoughtType::Mistake,
        "correction" => ThoughtType::Correction,
        "assumptioninvalidated" => ThoughtType::AssumptionInvalidated,
        "constraint" => ThoughtType::Constraint,
        "plan" => ThoughtType::Plan,
        "subgoal" => ThoughtType::Subgoal,
        "decision" => ThoughtType::Decision,
        "strategyshift" => ThoughtType::StrategyShift,
        "question" => ThoughtType::Question,
        "idea" => ThoughtType::Idea,
        "experiment" => ThoughtType::Experiment,
        "actiontaken" => ThoughtType::ActionTaken,
        "taskcomplete" => ThoughtType::TaskComplete,
        "checkpoint" => ThoughtType::Checkpoint,
        "statesnapshot" => ThoughtType::StateSnapshot,
        "handoff" => ThoughtType::Handoff,
        "summary" => ThoughtType::Summary,
        "surprise" => ThoughtType::Surprise,
        _ => {
            return Err(Box::new(ToolError::InvalidParameters(format!(
                "Unknown ThoughtType '{input}'"
            ))))
        }
    };
    Ok(thought_type)
}

fn parse_thought_role(input: &str) -> Result<ThoughtRole, Box<dyn Error + Send + Sync>> {
    let role = match normalize_label(input).as_str() {
        "memory" => ThoughtRole::Memory,
        "workingmemory" => ThoughtRole::WorkingMemory,
        "summary" => ThoughtRole::Summary,
        "compression" => ThoughtRole::Compression,
        "checkpoint" => ThoughtRole::Checkpoint,
        "handoff" => ThoughtRole::Handoff,
        "audit" => ThoughtRole::Audit,
        _ => {
            return Err(Box::new(ToolError::InvalidParameters(format!(
                "Unknown ThoughtRole '{input}'"
            ))))
        }
    };
    Ok(role)
}

fn normalize_label(input: &str) -> String {
    input
        .chars()
        .filter(|character| character.is_ascii_alphanumeric())
        .flat_map(|character| character.to_lowercase())
        .collect()
}
