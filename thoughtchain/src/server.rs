//! HTTP servers for exposing ThoughtChain as MCP and REST services.
//!
//! This module keeps the server implementation inside the `thoughtchain` crate
//! so other projects can run ThoughtChain as an independent long-running
//! process without depending on `cloudllm`.
//!
//! The MCP surface is wire-compatible with CloudLLM's `McpClientProtocol`:
//!
//! - `POST /tools/list`
//! - `POST /tools/execute`
//!
//! The REST surface exposes ThoughtChain operations directly:
//!
//! - `GET /health`
//! - `POST /v1/bootstrap`
//! - `POST /v1/thoughts`
//! - `POST /v1/search`
//! - `POST /v1/recent-context`
//! - `POST /v1/memory-markdown`
//! - `POST /v1/head`

use crate::{Thought, ThoughtChain, ThoughtInput, ThoughtQuery, ThoughtRole, ThoughtType};
use axum::extract::State;
use axum::http::StatusCode;
use axum::routing::{get, post};
use axum::{Json, Router};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::error::Error;
use std::net::{IpAddr, SocketAddr};
use std::path::PathBuf;
use std::sync::Arc;
use tokio::net::TcpListener;
use tokio::sync::{oneshot, RwLock};

/// Configuration shared by ThoughtChain server variants.
///
/// # Example
///
/// ```rust,no_run
/// use std::path::PathBuf;
/// use thoughtchain::server::ThoughtChainServiceConfig;
///
/// let config = ThoughtChainServiceConfig::new(
///     PathBuf::from("/tmp/thoughtchain"),
///     "persistent-chat-agent",
/// );
/// assert_eq!(config.default_chain_key, "persistent-chat-agent");
/// ```
#[derive(Debug, Clone)]
pub struct ThoughtChainServiceConfig {
    /// Directory containing chain storage files for the default JSONL adapter.
    pub chain_dir: PathBuf,
    /// Default chain key used when requests omit `chain_key`.
    pub default_chain_key: String,
}

impl ThoughtChainServiceConfig {
    /// Create a new service configuration.
    pub fn new(chain_dir: PathBuf, default_chain_key: impl Into<String>) -> Self {
        Self {
            chain_dir,
            default_chain_key: default_chain_key.into(),
        }
    }
}

/// Runtime configuration for the standalone `thoughtchaind` process.
///
/// Environment variables:
///
/// - `THOUGHTCHAIN_DIR`
/// - `THOUGHTCHAIN_DEFAULT_KEY`
/// - `THOUGHTCHAIN_BIND_HOST`
/// - `THOUGHTCHAIN_MCP_PORT`
/// - `THOUGHTCHAIN_REST_PORT`
///
/// # Example
///
/// ```rust,no_run
/// use thoughtchain::server::ThoughtChainServerConfig;
///
/// let config = ThoughtChainServerConfig::from_env();
/// assert!(config.mcp_addr.port() > 0);
/// ```
#[derive(Debug, Clone)]
pub struct ThoughtChainServerConfig {
    /// Shared storage configuration for both HTTP servers.
    pub service: ThoughtChainServiceConfig,
    /// Socket address to bind the MCP server to.
    pub mcp_addr: SocketAddr,
    /// Socket address to bind the REST server to.
    pub rest_addr: SocketAddr,
}

impl ThoughtChainServerConfig {
    /// Build a server configuration from environment variables.
    pub fn from_env() -> Self {
        let bind_host = std::env::var("THOUGHTCHAIN_BIND_HOST")
            .ok()
            .and_then(|value| value.parse::<IpAddr>().ok())
            .unwrap_or(IpAddr::from([127, 0, 0, 1]));
        let mcp_port = env_u16("THOUGHTCHAIN_MCP_PORT").unwrap_or(9471);
        let rest_port = env_u16("THOUGHTCHAIN_REST_PORT").unwrap_or(9472);

        Self {
            service: ThoughtChainServiceConfig::new(
                std::env::var("THOUGHTCHAIN_DIR")
                    .map(PathBuf::from)
                    .unwrap_or_else(|_| default_thoughtchain_dir()),
                std::env::var("THOUGHTCHAIN_DEFAULT_KEY")
                    .unwrap_or_else(|_| "persistent-chat-agent".to_string()),
            ),
            mcp_addr: SocketAddr::new(bind_host, mcp_port),
            rest_addr: SocketAddr::new(bind_host, rest_port),
        }
    }
}

/// Handle to a running HTTP server.
///
/// # Example
///
/// ```rust,no_run
/// use std::net::SocketAddr;
/// use thoughtchain::server::ServerHandle;
///
/// let addr = SocketAddr::from(([127, 0, 0, 1], 8080));
/// let (tx, _rx) = tokio::sync::oneshot::channel();
/// let handle = ServerHandle::new(addr, tx);
/// assert_eq!(handle.local_addr(), addr);
/// ```
#[derive(Debug)]
pub struct ServerHandle {
    addr: SocketAddr,
    shutdown_tx: Option<oneshot::Sender<()>>,
}

impl ServerHandle {
    /// Create a new server handle.
    pub fn new(addr: SocketAddr, shutdown_tx: oneshot::Sender<()>) -> Self {
        Self {
            addr,
            shutdown_tx: Some(shutdown_tx),
        }
    }

    /// Return the bound socket address.
    pub fn local_addr(&self) -> SocketAddr {
        self.addr
    }

    /// Request graceful shutdown of the server.
    pub fn shutdown(&mut self) -> Result<(), Box<dyn Error + Send + Sync>> {
        if let Some(tx) = self.shutdown_tx.take() {
            tx.send(())
                .map_err(|_| "server shutdown signal could not be delivered".into())
        } else {
            Ok(())
        }
    }
}

/// Handles for a running ThoughtChain MCP and REST server pair.
///
/// # Example
///
/// ```rust,no_run
/// use thoughtchain::server::{start_servers, ThoughtChainServerConfig};
///
/// # #[tokio::main]
/// # async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
/// let config = ThoughtChainServerConfig::from_env();
/// let handles = start_servers(config).await?;
/// println!("MCP: {}", handles.mcp.local_addr());
/// println!("REST: {}", handles.rest.local_addr());
/// # Ok(())
/// # }
/// ```
#[derive(Debug)]
pub struct ThoughtChainServerHandles {
    /// Running MCP server handle.
    pub mcp: ServerHandle,
    /// Running REST server handle.
    pub rest: ServerHandle,
}

/// Return the default on-disk ThoughtChain directory.
///
/// The default is `$HOME/.cloudllm/thoughtchain` when `HOME` is available,
/// otherwise `./.cloudllm/thoughtchain`.
///
/// # Example
///
/// ```
/// use thoughtchain::server::default_thoughtchain_dir;
///
/// let dir = default_thoughtchain_dir();
/// assert!(dir.ends_with("thoughtchain"));
/// ```
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

/// Start a standalone ThoughtChain MCP server.
///
/// The returned server is wire-compatible with CloudLLM's
/// `McpClientProtocol`.
///
/// # Example
///
/// ```rust,no_run
/// use std::net::SocketAddr;
/// use std::path::PathBuf;
/// use thoughtchain::server::{start_mcp_server, ThoughtChainServiceConfig};
///
/// # #[tokio::main]
/// # async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
/// let config = ThoughtChainServiceConfig::new(PathBuf::from("/tmp/tc"), "agent-memory");
/// let server = start_mcp_server(SocketAddr::from(([127, 0, 0, 1], 0)), config).await?;
/// println!("{}", server.local_addr());
/// # Ok(())
/// # }
/// ```
pub async fn start_mcp_server(
    addr: SocketAddr,
    config: ThoughtChainServiceConfig,
) -> Result<ServerHandle, Box<dyn Error + Send + Sync>> {
    start_router(addr, mcp_router(config)).await
}

/// Start a standalone ThoughtChain REST server.
///
/// # Example
///
/// ```rust,no_run
/// use std::net::SocketAddr;
/// use std::path::PathBuf;
/// use thoughtchain::server::{start_rest_server, ThoughtChainServiceConfig};
///
/// # #[tokio::main]
/// # async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
/// let config = ThoughtChainServiceConfig::new(PathBuf::from("/tmp/tc"), "agent-memory");
/// let server = start_rest_server(SocketAddr::from(([127, 0, 0, 1], 0)), config).await?;
/// println!("{}", server.local_addr());
/// # Ok(())
/// # }
/// ```
pub async fn start_rest_server(
    addr: SocketAddr,
    config: ThoughtChainServiceConfig,
) -> Result<ServerHandle, Box<dyn Error + Send + Sync>> {
    start_router(addr, rest_router(config)).await
}

/// Start both the MCP and REST servers for `thoughtchaind`.
pub async fn start_servers(
    config: ThoughtChainServerConfig,
) -> Result<ThoughtChainServerHandles, Box<dyn Error + Send + Sync>> {
    let mcp = start_mcp_server(config.mcp_addr, config.service.clone()).await?;
    let rest = start_rest_server(config.rest_addr, config.service).await?;
    Ok(ThoughtChainServerHandles { mcp, rest })
}

/// Build the MCP router without binding a socket.
///
/// This is useful for embedding the service inside another process or testing
/// the HTTP contract in-process.
pub fn mcp_router(config: ThoughtChainServiceConfig) -> Router {
    let service = Arc::new(ThoughtChainService::new(config));
    Router::new()
        .route("/health", get(health_handler))
        .route("/tools/list", post(mcp_list_tools_handler))
        .route("/tools/execute", post(mcp_execute_handler))
        .with_state(service)
}

/// Build the REST router without binding a socket.
///
/// This is useful for embedding the service inside another process or testing
/// the HTTP contract in-process.
pub fn rest_router(config: ThoughtChainServiceConfig) -> Router {
    let service = Arc::new(ThoughtChainService::new(config));
    Router::new()
        .route("/health", get(health_handler))
        .route("/v1/bootstrap", post(rest_bootstrap_handler))
        .route("/v1/thoughts", post(rest_append_handler))
        .route("/v1/search", post(rest_search_handler))
        .route("/v1/recent-context", post(rest_recent_context_handler))
        .route("/v1/memory-markdown", post(rest_memory_markdown_handler))
        .route("/v1/head", post(rest_head_handler))
        .with_state(service)
}

#[derive(Clone)]
struct ThoughtChainService {
    config: ThoughtChainServiceConfig,
    chains: Arc<RwLock<HashMap<String, Arc<RwLock<ThoughtChain>>>>>,
}

impl ThoughtChainService {
    fn new(config: ThoughtChainServiceConfig) -> Self {
        Self {
            config,
            chains: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    async fn get_chain(
        &self,
        chain_key: Option<&str>,
    ) -> Result<Arc<RwLock<ThoughtChain>>, Box<dyn Error + Send + Sync>> {
        let chain_key = chain_key
            .unwrap_or(&self.config.default_chain_key)
            .to_string();

        if let Some(existing) = self.chains.read().await.get(&chain_key).cloned() {
            return Ok(existing);
        }

        let chain = Arc::new(RwLock::new(ThoughtChain::open_with_key(
            &self.config.chain_dir,
            &chain_key,
        )?));

        let mut chains = self.chains.write().await;
        let entry = chains.entry(chain_key).or_insert_with(|| chain.clone());
        Ok(entry.clone())
    }

    async fn bootstrap(
        &self,
        request: BootstrapRequest,
    ) -> Result<BootstrapResponse, Box<dyn Error + Send + Sync>> {
        let chain = self.get_chain(request.chain_key.as_deref()).await?;
        let mut chain = chain.write().await;
        let bootstrapped = if chain.thoughts().is_empty() {
            let (agent_id, agent_name, agent_owner) = self.resolve_agent_identity(
                request.chain_key.as_deref(),
                request.agent_id.as_deref(),
                request.agent_name.as_deref(),
                request.agent_owner.as_deref(),
                "system",
                "ThoughtChain",
            );
            let input = ThoughtInput::new(ThoughtType::Summary, request.content)
                .with_agent_name(agent_name)
                .with_role(ThoughtRole::Checkpoint)
                .with_importance(request.importance.unwrap_or(1.0))
                .with_tags(request.tags.unwrap_or_default())
                .with_concepts(request.concepts.unwrap_or_default());
            let input = if let Some(agent_owner) = agent_owner {
                input.with_agent_owner(agent_owner)
            } else {
                input
            };
            chain.append_thought(&agent_id, input)?;
            true
        } else {
            false
        };

        Ok(BootstrapResponse {
            bootstrapped,
            thought_count: chain.thoughts().len(),
            head_hash: chain.head_hash().map(ToOwned::to_owned),
        })
    }

    async fn append(
        &self,
        request: AppendThoughtRequest,
    ) -> Result<AppendThoughtResponse, Box<dyn Error + Send + Sync>> {
        let chain = self.get_chain(request.chain_key.as_deref()).await?;
        let mut chain = chain.write().await;

        let thought_type = parse_thought_type(&request.thought_type)?;
        let role = request
            .role
            .as_deref()
            .map(parse_thought_role)
            .transpose()?
            .unwrap_or(ThoughtRole::Memory);
        let fallback_agent_id = self.resolve_chain_key(request.chain_key.as_deref());
        let (agent_id, agent_name, agent_owner) = self.resolve_agent_identity(
            request.chain_key.as_deref(),
            request.agent_id.as_deref(),
            request.agent_name.as_deref(),
            request.agent_owner.as_deref(),
            &fallback_agent_id,
            &fallback_agent_id,
        );

        let mut input = ThoughtInput::new(thought_type, request.content)
            .with_agent_name(agent_name)
            .with_role(role)
            .with_importance(request.importance.unwrap_or(0.5))
            .with_tags(request.tags.unwrap_or_default())
            .with_concepts(request.concepts.unwrap_or_default())
            .with_refs(request.refs.unwrap_or_default());
        if let Some(agent_owner) = agent_owner {
            input = input.with_agent_owner(agent_owner);
        }
        if let Some(confidence) = request.confidence {
            input = input.with_confidence(confidence);
        }

        let thought = chain.append_thought(&agent_id, input)?;
        Ok(AppendThoughtResponse {
            thought: thought_to_json(thought),
            head_hash: chain.head_hash().map(ToOwned::to_owned),
        })
    }

    async fn search(
        &self,
        request: SearchRequest,
    ) -> Result<SearchResponse, Box<dyn Error + Send + Sync>> {
        let chain = self.get_chain(request.chain_key.as_deref()).await?;
        let chain = chain.read().await;
        let query = build_query(&request)?;
        let thoughts = chain
            .query(&query)
            .into_iter()
            .map(thought_to_json)
            .collect::<Vec<_>>();
        Ok(SearchResponse { thoughts })
    }

    async fn recent_context(
        &self,
        request: RecentContextRequest,
    ) -> Result<RecentContextResponse, Box<dyn Error + Send + Sync>> {
        let chain = self.get_chain(request.chain_key.as_deref()).await?;
        let chain = chain.read().await;
        Ok(RecentContextResponse {
            prompt: chain.to_catchup_prompt(request.last_n.unwrap_or(12)),
        })
    }

    async fn memory_markdown(
        &self,
        request: MemoryMarkdownRequest,
    ) -> Result<MemoryMarkdownResponse, Box<dyn Error + Send + Sync>> {
        let chain = self.get_chain(request.chain_key.as_deref()).await?;
        let chain = chain.read().await;
        let query = build_markdown_query(&request)?;
        let markdown = if query_is_empty(&query) {
            chain.to_memory_markdown(None)
        } else {
            chain.to_memory_markdown(Some(&query))
        };
        Ok(MemoryMarkdownResponse { markdown })
    }

    async fn head(
        &self,
        request: ChainHeadRequest,
    ) -> Result<HeadResponse, Box<dyn Error + Send + Sync>> {
        let chain = self.get_chain(request.chain_key.as_deref()).await?;
        let chain = chain.read().await;
        Ok(HeadResponse {
            chain_key: self.resolve_chain_key(request.chain_key.as_deref()),
            thought_count: chain.thoughts().len(),
            head_hash: chain.head_hash().map(ToOwned::to_owned),
            latest_thought: chain.thoughts().last().map(thought_to_json),
            integrity_ok: chain.verify_integrity(),
            storage_location: chain.storage_location(),
        })
    }

    fn resolve_chain_key(&self, chain_key: Option<&str>) -> String {
        chain_key
            .unwrap_or(&self.config.default_chain_key)
            .to_string()
    }

    fn resolve_agent_identity(
        &self,
        chain_key: Option<&str>,
        agent_id: Option<&str>,
        agent_name: Option<&str>,
        agent_owner: Option<&str>,
        default_agent_id: &str,
        default_agent_name: &str,
    ) -> (String, String, Option<String>) {
        let fallback_agent_id = if default_agent_id.is_empty() {
            self.resolve_chain_key(chain_key)
        } else {
            default_agent_id.to_string()
        };
        let resolved_agent_id = agent_id
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(ToOwned::to_owned)
            .unwrap_or(fallback_agent_id);
        let resolved_agent_name = agent_name
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(ToOwned::to_owned)
            .unwrap_or_else(|| {
                if default_agent_name.is_empty() {
                    resolved_agent_id.clone()
                } else {
                    default_agent_name.to_string()
                }
            });
        let resolved_agent_owner = agent_owner
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(ToOwned::to_owned);

        (resolved_agent_id, resolved_agent_name, resolved_agent_owner)
    }
}

#[derive(Debug, Deserialize)]
struct McpExecuteRequest {
    tool: String,
    #[serde(default)]
    parameters: Value,
}

#[derive(Debug, Deserialize)]
struct BootstrapRequest {
    chain_key: Option<String>,
    agent_id: Option<String>,
    agent_name: Option<String>,
    agent_owner: Option<String>,
    content: String,
    importance: Option<f32>,
    tags: Option<Vec<String>>,
    concepts: Option<Vec<String>>,
}

#[derive(Debug, Serialize)]
struct BootstrapResponse {
    bootstrapped: bool,
    thought_count: usize,
    head_hash: Option<String>,
}

#[derive(Debug, Deserialize)]
struct AppendThoughtRequest {
    chain_key: Option<String>,
    agent_id: Option<String>,
    agent_name: Option<String>,
    agent_owner: Option<String>,
    thought_type: String,
    content: String,
    role: Option<String>,
    importance: Option<f32>,
    confidence: Option<f32>,
    tags: Option<Vec<String>>,
    concepts: Option<Vec<String>>,
    refs: Option<Vec<u64>>,
}

#[derive(Debug, Serialize)]
struct AppendThoughtResponse {
    thought: Value,
    head_hash: Option<String>,
}

#[derive(Debug, Deserialize, Default)]
struct SearchRequest {
    chain_key: Option<String>,
    text: Option<String>,
    thought_types: Option<Vec<String>>,
    roles: Option<Vec<String>>,
    tags_any: Option<Vec<String>>,
    concepts_any: Option<Vec<String>>,
    agent_ids: Option<Vec<String>>,
    agent_names: Option<Vec<String>>,
    agent_owners: Option<Vec<String>>,
    min_importance: Option<f32>,
    min_confidence: Option<f32>,
    since: Option<DateTime<Utc>>,
    until: Option<DateTime<Utc>>,
    limit: Option<usize>,
}

#[derive(Debug, Serialize)]
struct SearchResponse {
    thoughts: Vec<Value>,
}

#[derive(Debug, Deserialize)]
struct RecentContextRequest {
    chain_key: Option<String>,
    last_n: Option<usize>,
}

#[derive(Debug, Serialize)]
struct RecentContextResponse {
    prompt: String,
}

#[derive(Debug, Deserialize, Default)]
struct MemoryMarkdownRequest {
    chain_key: Option<String>,
    text: Option<String>,
    thought_types: Option<Vec<String>>,
    roles: Option<Vec<String>>,
    tags_any: Option<Vec<String>>,
    concepts_any: Option<Vec<String>>,
    agent_ids: Option<Vec<String>>,
    agent_names: Option<Vec<String>>,
    agent_owners: Option<Vec<String>>,
    min_importance: Option<f32>,
    min_confidence: Option<f32>,
    since: Option<DateTime<Utc>>,
    until: Option<DateTime<Utc>>,
    limit: Option<usize>,
}

#[derive(Debug, Serialize)]
struct MemoryMarkdownResponse {
    markdown: String,
}

#[derive(Debug, Deserialize, Default)]
struct ChainHeadRequest {
    chain_key: Option<String>,
}

#[derive(Debug, Serialize)]
struct HeadResponse {
    chain_key: String,
    thought_count: usize,
    head_hash: Option<String>,
    latest_thought: Option<Value>,
    integrity_ok: bool,
    storage_location: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct McpToolResult {
    success: bool,
    output: Value,
    error: Option<String>,
    metadata: HashMap<String, Value>,
}

impl McpToolResult {
    fn success(output: Value) -> Self {
        Self {
            success: true,
            output,
            error: None,
            metadata: HashMap::new(),
        }
    }

    fn failure(message: impl Into<String>) -> Self {
        Self {
            success: false,
            output: Value::Null,
            error: Some(message.into()),
            metadata: HashMap::new(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
struct McpToolParameterType(String);

#[derive(Debug, Clone, Serialize, Deserialize)]
struct McpToolParameter {
    name: String,
    #[serde(rename = "type")]
    param_type: McpToolParameterType,
    description: Option<String>,
    required: bool,
    default: Option<Value>,
    items: Option<Box<McpToolParameterType>>,
    properties: Option<HashMap<String, McpToolParameter>>,
}

impl McpToolParameter {
    fn string(name: &str, description: &str, required: bool) -> Self {
        Self {
            name: name.to_string(),
            param_type: McpToolParameterType("string".to_string()),
            description: Some(description.to_string()),
            required,
            default: None,
            items: None,
            properties: None,
        }
    }

    fn number(name: &str, description: &str) -> Self {
        Self {
            name: name.to_string(),
            param_type: McpToolParameterType("number".to_string()),
            description: Some(description.to_string()),
            required: false,
            default: None,
            items: None,
            properties: None,
        }
    }

    fn integer(name: &str, description: &str) -> Self {
        Self {
            name: name.to_string(),
            param_type: McpToolParameterType("integer".to_string()),
            description: Some(description.to_string()),
            required: false,
            default: None,
            items: None,
            properties: None,
        }
    }

    fn array(name: &str, description: &str, item_type: &str) -> Self {
        Self {
            name: name.to_string(),
            param_type: McpToolParameterType("array".to_string()),
            description: Some(description.to_string()),
            required: false,
            default: None,
            items: Some(Box::new(McpToolParameterType(item_type.to_string()))),
            properties: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct McpToolMetadata {
    name: String,
    description: String,
    parameters: Vec<McpToolParameter>,
    protocol_metadata: HashMap<String, Value>,
}

async fn start_router(
    addr: SocketAddr,
    router: Router,
) -> Result<ServerHandle, Box<dyn Error + Send + Sync>> {
    let listener = TcpListener::bind(addr).await?;
    let local_addr = listener.local_addr()?;
    let (shutdown_tx, shutdown_rx) = oneshot::channel();

    tokio::spawn(async move {
        let _ = axum::serve(listener, router.into_make_service())
            .with_graceful_shutdown(async move {
                let _ = shutdown_rx.await;
            })
            .await;
    });

    Ok(ServerHandle::new(local_addr, shutdown_tx))
}

async fn health_handler() -> Json<Value> {
    Json(json!({
        "status": "ok",
        "service": "thoughtchain"
    }))
}

async fn mcp_list_tools_handler() -> Json<Value> {
    Json(json!({ "tools": mcp_tool_metadata() }))
}

async fn mcp_execute_handler(
    State(service): State<Arc<ThoughtChainService>>,
    Json(request): Json<McpExecuteRequest>,
) -> (StatusCode, Json<Value>) {
    let result = match request.tool.as_str() {
        "thoughtchain_bootstrap" => {
            parse_and_call(request.parameters, |request| service.bootstrap(request)).await
        }
        "thoughtchain_append" => {
            parse_and_call(request.parameters, |request| service.append(request)).await
        }
        "thoughtchain_search" => {
            parse_and_call(request.parameters, |request| service.search(request)).await
        }
        "thoughtchain_recent_context" => {
            parse_and_call(request.parameters, |request| {
                service.recent_context(request)
            })
            .await
        }
        "thoughtchain_memory_markdown" => {
            parse_and_call(request.parameters, |request| {
                service.memory_markdown(request)
            })
            .await
        }
        "thoughtchain_head" => {
            parse_and_call(request.parameters, |request| service.head(request)).await
        }
        _ => Err(format!("Unknown tool '{}'", request.tool).into()),
    };

    match result {
        Ok(output) => (
            StatusCode::OK,
            Json(json!({ "result": McpToolResult::success(output) })),
        ),
        Err(error) => (
            StatusCode::BAD_REQUEST,
            Json(json!({ "result": McpToolResult::failure(error.to_string()) })),
        ),
    }
}

async fn rest_bootstrap_handler(
    State(service): State<Arc<ThoughtChainService>>,
    Json(request): Json<BootstrapRequest>,
) -> Result<Json<BootstrapResponse>, (StatusCode, Json<Value>)> {
    service_call(service.bootstrap(request).await)
}

async fn rest_append_handler(
    State(service): State<Arc<ThoughtChainService>>,
    Json(request): Json<AppendThoughtRequest>,
) -> Result<Json<AppendThoughtResponse>, (StatusCode, Json<Value>)> {
    service_call(service.append(request).await)
}

async fn rest_search_handler(
    State(service): State<Arc<ThoughtChainService>>,
    Json(request): Json<SearchRequest>,
) -> Result<Json<SearchResponse>, (StatusCode, Json<Value>)> {
    service_call(service.search(request).await)
}

async fn rest_recent_context_handler(
    State(service): State<Arc<ThoughtChainService>>,
    Json(request): Json<RecentContextRequest>,
) -> Result<Json<RecentContextResponse>, (StatusCode, Json<Value>)> {
    service_call(service.recent_context(request).await)
}

async fn rest_memory_markdown_handler(
    State(service): State<Arc<ThoughtChainService>>,
    Json(request): Json<MemoryMarkdownRequest>,
) -> Result<Json<MemoryMarkdownResponse>, (StatusCode, Json<Value>)> {
    service_call(service.memory_markdown(request).await)
}

async fn rest_head_handler(
    State(service): State<Arc<ThoughtChainService>>,
    Json(request): Json<ChainHeadRequest>,
) -> Result<Json<HeadResponse>, (StatusCode, Json<Value>)> {
    service_call(service.head(request).await)
}

async fn parse_and_call<T, O, F, Fut>(
    parameters: Value,
    f: F,
) -> Result<Value, Box<dyn Error + Send + Sync>>
where
    T: for<'de> Deserialize<'de>,
    O: Serialize,
    F: FnOnce(T) -> Fut,
    Fut: std::future::Future<Output = Result<O, Box<dyn Error + Send + Sync>>>,
{
    let request = serde_json::from_value::<T>(parameters)?;
    Ok(serde_json::to_value(f(request).await?)?)
}

fn service_call<T: Serialize>(
    result: Result<T, Box<dyn Error + Send + Sync>>,
) -> Result<Json<T>, (StatusCode, Json<Value>)> {
    result.map(Json).map_err(|error| {
        (
            StatusCode::BAD_REQUEST,
            Json(json!({ "error": error.to_string() })),
        )
    })
}

fn mcp_tool_metadata() -> Vec<McpToolMetadata> {
    vec![
        McpToolMetadata {
            name: "thoughtchain_bootstrap".to_string(),
            description:
                "Ensure a thought chain exists and initialize it the first time with a bootstrap memory."
                    .to_string(),
            parameters: vec![
                McpToolParameter::string("chain_key", "Optional durable chain key. Defaults to the server's default chain.", false),
                McpToolParameter::string("agent_id", "Optional producing agent id. Defaults to 'system' for bootstrap.", false),
                McpToolParameter::string("agent_name", "Optional producing agent name.", false),
                McpToolParameter::string("agent_owner", "Optional producing agent owner or tenant label.", false),
                McpToolParameter::string("content", "Bootstrap summary to store if the chain is empty.", true),
                McpToolParameter::number("importance", "Optional importance score between 0.0 and 1.0."),
                McpToolParameter::array("tags", "Optional tags for the bootstrap memory.", "string"),
                McpToolParameter::array("concepts", "Optional concepts for the bootstrap memory.", "string"),
            ],
            protocol_metadata: HashMap::new(),
        },
        McpToolMetadata {
            name: "thoughtchain_append".to_string(),
            description:
                "Append a durable semantic memory to ThoughtChain. Use exact ThoughtType names like PreferenceUpdate, Constraint, Decision, Insight, Wonder, Question, Summary, Mistake, or Correction."
                    .to_string(),
            parameters: vec![
                McpToolParameter::string("chain_key", "Optional durable chain key.", false),
                McpToolParameter::string("agent_id", "Optional producing agent id. Defaults to the chain key when omitted.", false),
                McpToolParameter::string("agent_name", "Optional producing agent name.", false),
                McpToolParameter::string("agent_owner", "Optional producing agent owner or tenant label.", false),
                McpToolParameter::string("thought_type", "Semantic type of the thought.", true),
                McpToolParameter::string("content", "Concise durable memory content.", true),
                McpToolParameter::string("role", "Optional thought role such as Memory, Summary, Compression, Checkpoint, or Handoff.", false),
                McpToolParameter::number("importance", "Optional importance score between 0.0 and 1.0."),
                McpToolParameter::number("confidence", "Optional confidence score between 0.0 and 1.0."),
                McpToolParameter::array("tags", "Optional tags.", "string"),
                McpToolParameter::array("concepts", "Optional semantic concepts.", "string"),
                McpToolParameter::array("refs", "Optional referenced thought indices.", "integer"),
            ],
            protocol_metadata: HashMap::new(),
        },
        McpToolMetadata {
            name: "thoughtchain_search".to_string(),
            description:
                "Search durable memories by text, type, role, tags, concepts, and importance."
                    .to_string(),
            parameters: vec![
                McpToolParameter::string("chain_key", "Optional durable chain key.", false),
                McpToolParameter::string("text", "Optional text filter applied to content, tags, and concepts.", false),
                McpToolParameter::array("thought_types", "Optional list of ThoughtType names.", "string"),
                McpToolParameter::array("roles", "Optional list of ThoughtRole names.", "string"),
                McpToolParameter::array("tags_any", "Optional tags to match.", "string"),
                McpToolParameter::array("concepts_any", "Optional concepts to match.", "string"),
                McpToolParameter::array("agent_ids", "Optional producing agent ids to match.", "string"),
                McpToolParameter::array("agent_names", "Optional producing agent names to match.", "string"),
                McpToolParameter::array("agent_owners", "Optional producing agent owners to match.", "string"),
                McpToolParameter::number("min_importance", "Optional minimum importance threshold."),
                McpToolParameter::integer("limit", "Optional maximum number of results."),
            ],
            protocol_metadata: HashMap::new(),
        },
        McpToolMetadata {
            name: "thoughtchain_recent_context".to_string(),
            description:
                "Render recent ThoughtChain context as a prompt snippet suitable for resuming work."
                    .to_string(),
            parameters: vec![
                McpToolParameter::string("chain_key", "Optional durable chain key.", false),
                McpToolParameter::integer("last_n", "How many recent thoughts to include."),
            ],
            protocol_metadata: HashMap::new(),
        },
        McpToolMetadata {
            name: "thoughtchain_memory_markdown".to_string(),
            description: "Export a MEMORY.md style Markdown summary from ThoughtChain.".to_string(),
            parameters: vec![
                McpToolParameter::string("chain_key", "Optional durable chain key.", false),
                McpToolParameter::string("text", "Optional text filter.", false),
                McpToolParameter::array("thought_types", "Optional list of ThoughtType names.", "string"),
                McpToolParameter::array("roles", "Optional list of ThoughtRole names.", "string"),
                McpToolParameter::array("tags_any", "Optional tags to match.", "string"),
                McpToolParameter::array("concepts_any", "Optional concepts to match.", "string"),
                McpToolParameter::array("agent_ids", "Optional producing agent ids to match.", "string"),
                McpToolParameter::array("agent_names", "Optional producing agent names to match.", "string"),
                McpToolParameter::array("agent_owners", "Optional producing agent owners to match.", "string"),
                McpToolParameter::number("min_importance", "Optional minimum importance threshold."),
                McpToolParameter::integer("limit", "Optional maximum number of thoughts."),
            ],
            protocol_metadata: HashMap::new(),
        },
        McpToolMetadata {
            name: "thoughtchain_head".to_string(),
            description:
                "Return head metadata for a ThoughtChain including length, latest thought, and head hash."
                    .to_string(),
            parameters: vec![McpToolParameter::string(
                "chain_key",
                "Optional durable chain key.",
                false,
            )],
            protocol_metadata: HashMap::new(),
        },
    ]
}

fn build_query(request: &SearchRequest) -> Result<ThoughtQuery, Box<dyn Error + Send + Sync>> {
    let mut query = ThoughtQuery::new();

    if let Some(text) = &request.text {
        query = query.with_text(text.clone());
    }
    if let Some(min_importance) = request.min_importance {
        query = query.with_min_importance(min_importance);
    }
    if let Some(min_confidence) = request.min_confidence {
        query = query.with_min_confidence(min_confidence);
    }
    if let Some(limit) = request.limit {
        query = query.with_limit(limit);
    }
    if let Some(since) = request.since {
        query = query.with_since(since);
    }
    if let Some(until) = request.until {
        query = query.with_until(until);
    }

    if let Some(thought_types) = &request.thought_types {
        query = query.with_types(
            thought_types
                .iter()
                .map(|value| parse_thought_type(value))
                .collect::<Result<Vec<_>, _>>()?,
        );
    }
    if let Some(roles) = &request.roles {
        query = query.with_roles(
            roles
                .iter()
                .map(|value| parse_thought_role(value))
                .collect::<Result<Vec<_>, _>>()?,
        );
    }
    if let Some(tags_any) = &request.tags_any {
        query = query.with_tags_any(tags_any.clone());
    }
    if let Some(concepts_any) = &request.concepts_any {
        query = query.with_concepts_any(concepts_any.clone());
    }
    if let Some(agent_ids) = &request.agent_ids {
        query = query.with_agent_ids(agent_ids.clone());
    }
    if let Some(agent_names) = &request.agent_names {
        query = query.with_agent_names(agent_names.clone());
    }
    if let Some(agent_owners) = &request.agent_owners {
        query = query.with_agent_owners(agent_owners.clone());
    }

    Ok(query)
}

fn build_markdown_query(
    request: &MemoryMarkdownRequest,
) -> Result<ThoughtQuery, Box<dyn Error + Send + Sync>> {
    build_query(&SearchRequest {
        chain_key: request.chain_key.clone(),
        text: request.text.clone(),
        thought_types: request.thought_types.clone(),
        roles: request.roles.clone(),
        tags_any: request.tags_any.clone(),
        concepts_any: request.concepts_any.clone(),
        agent_ids: request.agent_ids.clone(),
        agent_names: request.agent_names.clone(),
        agent_owners: request.agent_owners.clone(),
        min_importance: request.min_importance,
        min_confidence: request.min_confidence,
        since: request.since,
        until: request.until,
        limit: request.limit,
    })
}

fn query_is_empty(query: &ThoughtQuery) -> bool {
    query.thought_types.is_none()
        && query.roles.is_none()
        && query.agent_ids.is_none()
        && query.agent_names.is_none()
        && query.agent_owners.is_none()
        && query.tags_any.is_empty()
        && query.concepts_any.is_empty()
        && query.text_contains.is_none()
        && query.min_importance.is_none()
        && query.min_confidence.is_none()
        && query.since.is_none()
        && query.until.is_none()
        && query.limit.is_none()
}

fn parse_thought_type(input: &str) -> Result<ThoughtType, Box<dyn Error + Send + Sync>> {
    let thought_type = match normalize_label(input).as_str() {
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
        "wonder" => ThoughtType::Wonder,
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
        _ => return Err(format!("Unknown ThoughtType '{input}'").into()),
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
        _ => return Err(format!("Unknown ThoughtRole '{input}'").into()),
    };

    Ok(role)
}

fn thought_to_json(thought: &Thought) -> Value {
    serde_json::to_value(thought).unwrap_or_else(|_| {
        json!({
            "index": thought.index,
            "content": thought.content,
        })
    })
}

fn normalize_label(input: &str) -> String {
    input
        .chars()
        .filter(|character| character.is_ascii_alphanumeric())
        .collect::<String>()
        .to_lowercase()
}

fn env_u16(key: &str) -> Option<u16> {
    std::env::var(key)
        .ok()
        .and_then(|value| value.parse::<u16>().ok())
}
