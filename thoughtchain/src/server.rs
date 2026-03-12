//! HTTP servers for exposing ThoughtChain as MCP and REST services.
//!
//! This module keeps the server implementation inside the `thoughtchain` crate
//! so other projects can run ThoughtChain as an independent long-running
//! process without depending on `cloudllm`.
//!
//! The MCP surface includes both:
//!
//! - standard streamable HTTP MCP at `POST /`
//! - legacy CloudLLM-compatible endpoints:
//!   - `POST /tools/list`
//!   - `POST /tools/execute`
//!
//! The REST surface exposes ThoughtChain operations directly:
//!
//! - `GET /health`
//! - `POST /v1/bootstrap`
//! - `POST /v1/thoughts`
//! - `POST /v1/retrospectives`
//! - `POST /v1/search`
//! - `POST /v1/recent-context`
//! - `POST /v1/memory-markdown`
//! - `POST /v1/head`
//! - `GET /v1/chains`
//! - `POST /v1/agents`

use crate::{
    load_registered_chains, AgentRecord, AgentStatus, PublicKeyAlgorithm, StorageAdapterKind,
    Thought, ThoughtChain, ThoughtInput, ThoughtQuery, ThoughtRole, ThoughtType,
    THOUGHTCHAIN_CURRENT_VERSION,
};
use async_trait::async_trait;
use axum::extract::State;
use axum::http::StatusCode;
use axum::routing::{get, post};
use axum::{Json, Router};
use chrono::{DateTime, Utc};
use mcp::http::axum_router as shared_mcp_router;
use mcp::{
    streamable_http_router, HttpServerConfig, IpFilter, StreamableHttpConfig, ToolError,
    ToolMetadata, ToolParameter, ToolParameterType, ToolProtocol, ToolResult,
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::error::Error;
use std::io;
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
/// use thoughtchain::StorageAdapterKind;
/// use thoughtchain::server::ThoughtChainServiceConfig;
///
/// let config = ThoughtChainServiceConfig::new(
///     PathBuf::from("/tmp/thoughtchain"),
///     "borganism-brain",
///     StorageAdapterKind::Jsonl,
/// );
/// assert_eq!(config.default_chain_key, "borganism-brain");
/// assert!(!config.verbose);
/// ```
#[derive(Debug, Clone)]
pub struct ThoughtChainServiceConfig {
    /// Directory containing chain storage files.
    pub chain_dir: PathBuf,
    /// Default chain key used when requests omit `chain_key`.
    pub default_chain_key: String,
    /// Default storage adapter used when creating new chains.
    pub default_storage_adapter: StorageAdapterKind,
    /// When true, log each ThoughtChain read or write interaction to stderr.
    pub verbose: bool,
}

impl ThoughtChainServiceConfig {
    /// Create a new service configuration.
    pub fn new(
        chain_dir: PathBuf,
        default_chain_key: impl Into<String>,
        default_storage_adapter: StorageAdapterKind,
    ) -> Self {
        Self {
            chain_dir,
            default_chain_key: default_chain_key.into(),
            default_storage_adapter,
            verbose: false,
        }
    }

    /// Enable or disable verbose interaction logging for the service.
    pub fn with_verbose(mut self, verbose: bool) -> Self {
        self.verbose = verbose;
        self
    }
}

/// Runtime configuration for the standalone `thoughtchaind` process.
///
/// Environment variables:
///
/// - `THOUGHTCHAIN_DIR`
/// - `THOUGHTCHAIN_DEFAULT_KEY`
/// - `THOUGHTCHAIN_DEFAULT_STORAGE_ADAPTER`
/// - `THOUGHTCHAIN_VERBOSE`
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
        let storage_adapter = std::env::var("THOUGHTCHAIN_DEFAULT_STORAGE_ADAPTER")
            .or_else(|_| std::env::var("THOUGHTCHAIN_STORAGE_ADAPTER"))
            .ok()
            .map(|value| value.parse().unwrap_or(StorageAdapterKind::Binary))
            .unwrap_or(StorageAdapterKind::Binary);
        let verbose = env_bool("THOUGHTCHAIN_VERBOSE").unwrap_or(false);
        let mcp_port = env_u16("THOUGHTCHAIN_MCP_PORT").unwrap_or(9471);
        let rest_port = env_u16("THOUGHTCHAIN_REST_PORT").unwrap_or(9472);

        Self {
            service: ThoughtChainServiceConfig::new(
                std::env::var("THOUGHTCHAIN_DIR")
                    .map(PathBuf::from)
                    .unwrap_or_else(|_| default_thoughtchain_dir()),
                std::env::var("THOUGHTCHAIN_DEFAULT_KEY")
                    .unwrap_or_else(|_| "borganism-brain".to_string()),
                storage_adapter,
            )
            .with_verbose(verbose),
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
/// The returned server exposes both standard MCP and the legacy
/// CloudLLM-compatible MCP HTTP endpoints.
///
/// # Example
///
/// ```rust,no_run
/// use std::net::SocketAddr;
/// use std::path::PathBuf;
/// use thoughtchain::StorageAdapterKind;
/// use thoughtchain::server::{start_mcp_server, ThoughtChainServiceConfig};
///
/// # #[tokio::main]
/// # async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
/// let config = ThoughtChainServiceConfig::new(
///     PathBuf::from("/tmp/tc"),
///     "agent-memory",
///     StorageAdapterKind::Jsonl,
/// );
/// let server = start_mcp_server(SocketAddr::from(([127, 0, 0, 1], 0)), config).await?;
/// println!("{}", server.local_addr());
/// # Ok(())
/// # }
/// ```
pub async fn start_mcp_server(
    addr: SocketAddr,
    config: ThoughtChainServiceConfig,
) -> Result<ServerHandle, Box<dyn Error + Send + Sync>> {
    let service = Arc::new(ThoughtChainService::new(config));
    start_router(addr, standard_and_legacy_mcp_router(service, addr)).await
}

/// Start a standalone ThoughtChain REST server.
///
/// # Example
///
/// ```rust,no_run
/// use std::net::SocketAddr;
/// use std::path::PathBuf;
/// use thoughtchain::StorageAdapterKind;
/// use thoughtchain::server::{start_rest_server, ThoughtChainServiceConfig};
///
/// # #[tokio::main]
/// # async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
/// let config = ThoughtChainServiceConfig::new(
///     PathBuf::from("/tmp/tc"),
///     "agent-memory",
///     StorageAdapterKind::Jsonl,
/// );
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

/// Build a standard streamable HTTP MCP router without binding a socket.
///
/// This exposes the modern MCP root endpoint used by remote MCP clients such as
/// Codex and Claude Code. It is primarily useful for testing and embedding.
pub fn standard_mcp_router(config: ThoughtChainServiceConfig) -> Router {
    let service = Arc::new(ThoughtChainService::new(config));
    standard_mcp_only_router(service, SocketAddr::from(([127, 0, 0, 1], 0)))
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
        .route(
            "/v1/retrospectives",
            post(rest_append_retrospective_handler),
        )
        .route("/v1/search", post(rest_search_handler))
        .route("/v1/recent-context", post(rest_recent_context_handler))
        .route("/v1/memory-markdown", post(rest_memory_markdown_handler))
        .route("/v1/head", post(rest_head_handler))
        .route("/v1/chains", get(rest_list_chains_handler))
        .route("/v1/agents", post(rest_list_agents_handler))
        .route("/v1/agent", post(rest_get_agent_handler))
        .route("/v1/agent-registry", post(rest_list_agent_registry_handler))
        .route("/v1/agents/upsert", post(rest_upsert_agent_handler))
        .route(
            "/v1/agents/description",
            post(rest_set_agent_description_handler),
        )
        .route("/v1/agents/aliases", post(rest_add_agent_alias_handler))
        .route("/v1/agents/keys", post(rest_add_agent_key_handler))
        .route(
            "/v1/agents/keys/revoke",
            post(rest_revoke_agent_key_handler),
        )
        .route("/v1/agents/disable", post(rest_disable_agent_handler))
        .with_state(service)
}

#[derive(Clone)]
struct ThoughtChainService {
    config: ThoughtChainServiceConfig,
    chains: Arc<RwLock<HashMap<String, Arc<RwLock<ThoughtChain>>>>>,
}

#[derive(Clone)]
struct ThoughtChainMcpProtocol {
    service: Arc<ThoughtChainService>,
}

impl ThoughtChainMcpProtocol {
    fn new(service: Arc<ThoughtChainService>) -> Self {
        Self { service }
    }
}

fn standard_and_legacy_mcp_router(service: Arc<ThoughtChainService>, addr: SocketAddr) -> Router {
    standard_mcp_only_router(service.clone(), addr).merge(shared_mcp_router(
        &HttpServerConfig {
            addr,
            bearer_token: None,
            ip_filter: IpFilter::new(),
            event_handler: None,
        },
        Arc::new(ThoughtChainMcpProtocol::new(service)),
    ))
}

fn standard_mcp_only_router(service: Arc<ThoughtChainService>, addr: SocketAddr) -> Router {
    Router::new()
        .route("/health", get(health_handler))
        .merge(streamable_http_router(
            &HttpServerConfig {
                addr,
                bearer_token: None,
                ip_filter: IpFilter::new(),
                event_handler: None,
            },
            &StreamableHttpConfig::new("thoughtchain", env!("CARGO_PKG_VERSION"))
                .with_server_title("ThoughtChain")
                .with_instructions(
                    "ThoughtChain provides semantic, append-only memory tools for durable agent context, memory search, handoff, and auditability.",
                ),
            Arc::new(ThoughtChainMcpProtocol::new(service)),
        ))
}

#[async_trait]
impl ToolProtocol for ThoughtChainMcpProtocol {
    async fn execute(
        &self,
        tool_name: &str,
        parameters: Value,
    ) -> Result<ToolResult, Box<dyn Error + Send + Sync>> {
        let output = match tool_name {
            "thoughtchain_bootstrap" => {
                parse_and_call(parameters, |request| self.service.bootstrap(request)).await
            }
            "thoughtchain_append" => {
                parse_and_call(parameters, |request| self.service.append(request)).await
            }
            "thoughtchain_append_retrospective" => {
                parse_and_call(parameters, |request| {
                    self.service.append_retrospective(request)
                })
                .await
            }
            "thoughtchain_search" => {
                parse_and_call(parameters, |request| self.service.search(request)).await
            }
            "thoughtchain_list_chains" => self.service.list_chains_json().await,
            "thoughtchain_list_agents" => {
                parse_and_call(parameters, |request| self.service.list_agents(request)).await
            }
            "thoughtchain_get_agent" => {
                parse_and_call(parameters, |request| self.service.get_agent(request)).await
            }
            "thoughtchain_list_agent_registry" => {
                parse_and_call(parameters, |request| self.service.list_agent_registry(request)).await
            }
            "thoughtchain_upsert_agent" => {
                parse_and_call(parameters, |request| self.service.upsert_agent(request)).await
            }
            "thoughtchain_set_agent_description" => {
                parse_and_call(parameters, |request| {
                    self.service.set_agent_description(request)
                })
                .await
            }
            "thoughtchain_add_agent_alias" => {
                parse_and_call(parameters, |request| self.service.add_agent_alias(request)).await
            }
            "thoughtchain_add_agent_key" => {
                parse_and_call(parameters, |request| self.service.add_agent_key(request)).await
            }
            "thoughtchain_revoke_agent_key" => {
                parse_and_call(parameters, |request| self.service.revoke_agent_key(request)).await
            }
            "thoughtchain_disable_agent" => {
                parse_and_call(parameters, |request| self.service.disable_agent(request)).await
            }
            "thoughtchain_recent_context" => {
                parse_and_call(parameters, |request| self.service.recent_context(request)).await
            }
            "thoughtchain_memory_markdown" => {
                parse_and_call(parameters, |request| self.service.memory_markdown(request)).await
            }
            "thoughtchain_head" => {
                parse_and_call(parameters, |request| self.service.head(request)).await
            }
            _ => {
                return Err(Box::new(ToolError::NotFound(tool_name.to_string())));
            }
        }?;

        Ok(ToolResult::success(output))
    }

    async fn list_tools(&self) -> Result<Vec<ToolMetadata>, Box<dyn Error + Send + Sync>> {
        Ok(mcp_tool_metadata())
    }

    async fn get_tool_metadata(
        &self,
        tool_name: &str,
    ) -> Result<ToolMetadata, Box<dyn Error + Send + Sync>> {
        mcp_tool_metadata()
            .into_iter()
            .find(|tool| tool.name == tool_name)
            .ok_or_else(|| Box::new(ToolError::NotFound(tool_name.to_string())) as _)
    }

    fn protocol_name(&self) -> &str {
        "thoughtchain"
    }
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
        storage_adapter: Option<StorageAdapterKind>,
    ) -> Result<Arc<RwLock<ThoughtChain>>, Box<dyn Error + Send + Sync>> {
        let chain_key = chain_key
            .unwrap_or(&self.config.default_chain_key)
            .to_string();

        if let Some(existing) = self.chains.read().await.get(&chain_key).cloned() {
            return Ok(existing);
        }

        let chain = Arc::new(RwLock::new(ThoughtChain::open_with_key_and_storage_kind(
            &self.config.chain_dir,
            &chain_key,
            storage_adapter.unwrap_or(self.config.default_storage_adapter),
        )?));

        let mut chains = self.chains.write().await;
        let entry = chains.entry(chain_key).or_insert_with(|| chain.clone());
        Ok(entry.clone())
    }

    async fn bootstrap(
        &self,
        request: BootstrapRequest,
    ) -> Result<BootstrapResponse, Box<dyn Error + Send + Sync>> {
        let chain_key = self.resolve_chain_key(request.chain_key.as_deref());
        let storage_adapter = request
            .storage_adapter
            .as_deref()
            .map(parse_storage_adapter_kind)
            .transpose()?;
        let chain = self.get_chain(Some(&chain_key), storage_adapter).await?;
        let mut chain = chain.write().await;
        let bootstrapped = if chain.thoughts().is_empty() {
            let (agent_id, agent_name, agent_owner) = self.resolve_agent_identity(
                Some(&chain_key),
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
            let thought = chain.append_thought(&agent_id, input)?.clone();
            self.log_interaction(InteractionLogEntry {
                access: "write",
                operation: "bootstrap",
                chain_key: chain_key.clone(),
                metadata: InteractionMetadata::from_chain_thought(&chain, &thought),
                result_count: Some(1),
                note: Some("bootstrapped=true".to_string()),
            });
            true
        } else {
            self.log_interaction(InteractionLogEntry {
                access: "write",
                operation: "bootstrap",
                chain_key: chain_key.clone(),
                metadata: InteractionMetadata::default(),
                result_count: Some(chain.thoughts().len()),
                note: Some("bootstrapped=false".to_string()),
            });
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
        let chain_key = self.resolve_chain_key(request.chain_key.as_deref());
        let chain = self.get_chain(Some(&chain_key), None).await?;
        let mut chain = chain.write().await;

        let thought_type = parse_thought_type(&request.thought_type)?;
        let role = request
            .role
            .as_deref()
            .map(parse_thought_role)
            .transpose()?
            .unwrap_or(ThoughtRole::Memory);
        let fallback_agent_id = chain_key.clone();
        let (agent_id, agent_name, agent_owner) = self.resolve_agent_identity(
            Some(&chain_key),
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
        if let Some(signing_key_id) = request.signing_key_id {
            input = input.with_signing_key_id(signing_key_id);
        }
        if let Some(thought_signature) = request.thought_signature {
            input = input.with_thought_signature(thought_signature);
        }
        if let Some(confidence) = request.confidence {
            input = input.with_confidence(confidence);
        }

        let thought = chain.append_thought(&agent_id, input)?.clone();
        self.log_interaction(InteractionLogEntry {
            access: "write",
            operation: "append",
            chain_key,
            metadata: InteractionMetadata::from_chain_thought(&chain, &thought),
            result_count: Some(1),
            note: None,
        });
        Ok(AppendThoughtResponse {
            thought: thought_to_json(&chain, &thought),
            head_hash: chain.head_hash().map(ToOwned::to_owned),
        })
    }

    async fn append_retrospective(
        &self,
        request: AppendRetrospectiveRequest,
    ) -> Result<AppendThoughtResponse, Box<dyn Error + Send + Sync>> {
        let chain_key = self.resolve_chain_key(request.chain_key.as_deref());
        let chain = self.get_chain(Some(&chain_key), None).await?;
        let mut chain = chain.write().await;

        let thought_type = request
            .thought_type
            .as_deref()
            .map(parse_thought_type)
            .transpose()?
            .unwrap_or(ThoughtType::LessonLearned);
        let fallback_agent_id = chain_key.clone();
        let (agent_id, agent_name, agent_owner) = self.resolve_agent_identity(
            Some(&chain_key),
            request.agent_id.as_deref(),
            request.agent_name.as_deref(),
            request.agent_owner.as_deref(),
            &fallback_agent_id,
            &fallback_agent_id,
        );

        let mut input = ThoughtInput::new(thought_type, request.content)
            .with_agent_name(agent_name)
            .with_role(ThoughtRole::Retrospective)
            .with_importance(request.importance.unwrap_or(0.7))
            .with_tags(request.tags.unwrap_or_default())
            .with_concepts(request.concepts.unwrap_or_default())
            .with_refs(request.refs.unwrap_or_default());
        if let Some(agent_owner) = agent_owner {
            input = input.with_agent_owner(agent_owner);
        }
        if let Some(signing_key_id) = request.signing_key_id {
            input = input.with_signing_key_id(signing_key_id);
        }
        if let Some(thought_signature) = request.thought_signature {
            input = input.with_thought_signature(thought_signature);
        }
        if let Some(confidence) = request.confidence {
            input = input.with_confidence(confidence);
        }

        let thought = chain.append_thought(&agent_id, input)?.clone();
        self.log_interaction(InteractionLogEntry {
            access: "write",
            operation: "append_retrospective",
            chain_key,
            metadata: InteractionMetadata::from_chain_thought(&chain, &thought),
            result_count: Some(1),
            note: None,
        });
        Ok(AppendThoughtResponse {
            thought: thought_to_json(&chain, &thought),
            head_hash: chain.head_hash().map(ToOwned::to_owned),
        })
    }

    async fn search(
        &self,
        request: SearchRequest,
    ) -> Result<SearchResponse, Box<dyn Error + Send + Sync>> {
        let chain_key = self.resolve_chain_key(request.chain_key.as_deref());
        let chain = self.get_chain(Some(&chain_key), None).await?;
        let chain = chain.read().await;
        let query = build_query(&request)?;
        let matched = chain.query(&query);
        self.log_interaction(InteractionLogEntry {
            access: "read",
            operation: "search",
            chain_key,
            metadata: InteractionMetadata::from_chain_thoughts(&chain, matched.iter().copied()),
            result_count: Some(matched.len()),
            note: None,
        });
        let thoughts = matched
            .into_iter()
            .map(|thought| thought_to_json(&chain, thought))
            .collect::<Vec<_>>();
        Ok(SearchResponse { thoughts })
    }

    async fn list_chains_json(&self) -> Result<Value, Box<dyn Error + Send + Sync>> {
        Ok(serde_json::to_value(self.list_chains().await?)?)
    }

    async fn list_chains(&self) -> Result<ListChainsResponse, Box<dyn Error + Send + Sync>> {
        let mut chain_keys = BTreeSet::new();
        let registry = load_registered_chains(&self.config.chain_dir)?;
        chain_keys.extend(registry.chains.keys().cloned());

        let mut chains_by_key: BTreeMap<String, ChainSummary> = registry
            .chains
            .values()
            .map(|entry| {
                (
                    entry.chain_key.clone(),
                    ChainSummary {
                        chain_key: entry.chain_key.clone(),
                        version: entry.version,
                        storage_adapter: entry.storage_adapter.to_string(),
                        thought_count: entry.thought_count,
                        agent_count: entry.agent_count,
                        storage_location: entry.storage_location.clone(),
                    },
                )
            })
            .collect();

        let open_chains: Vec<(String, Arc<RwLock<ThoughtChain>>)> = self
            .chains
            .read()
            .await
            .iter()
            .map(|(chain_key, chain)| (chain_key.clone(), Arc::clone(chain)))
            .collect();

        for (chain_key, chain) in open_chains {
            chain_keys.insert(chain_key.clone());
            let chain = chain.read().await;
            let storage_location = chain.storage_location();
            chains_by_key
                .entry(chain_key.clone())
                .and_modify(|summary| {
                    summary.version = THOUGHTCHAIN_CURRENT_VERSION;
                    summary.thought_count = chain.thoughts().len() as u64;
                    summary.agent_count = chain.agent_registry().agents.len();
                    summary.storage_location = storage_location.clone();
                })
                .or_insert_with(|| ChainSummary {
                    chain_key: chain_key.clone(),
                    version: THOUGHTCHAIN_CURRENT_VERSION,
                    storage_adapter: infer_storage_adapter_name(&storage_location),
                    thought_count: chain.thoughts().len() as u64,
                    agent_count: chain.agent_registry().agents.len(),
                    storage_location: storage_location.clone(),
                });
        }

        let chains = chains_by_key.into_values().collect();

        let response = ListChainsResponse {
            default_chain_key: self.config.default_chain_key.clone(),
            chain_keys: chain_keys.into_iter().collect(),
            chains,
        };
        self.log_interaction(InteractionLogEntry {
            access: "read",
            operation: "list_chains",
            chain_key: "<all>".to_string(),
            metadata: InteractionMetadata::default(),
            result_count: Some(response.chain_keys.len()),
            note: None,
        });
        Ok(response)
    }

    async fn list_agents(
        &self,
        request: ListAgentsRequest,
    ) -> Result<ListAgentsResponse, Box<dyn Error + Send + Sync>> {
        let chain_key = self.resolve_chain_key(request.chain_key.as_deref());
        let chain = self.get_chain(Some(&chain_key), None).await?;
        let chain = chain.read().await;
        let agents = chain
            .agent_registry()
            .agents
            .values()
            .map(|record| AgentIdentitySummary {
                agent_id: record.agent_id.clone(),
                agent_name: record.display_name.clone(),
                agent_owner: record.owner.clone(),
            })
            .collect();

        self.log_interaction(InteractionLogEntry {
            access: "read",
            operation: "list_agents",
            chain_key: chain_key.clone(),
            metadata: InteractionMetadata::from_chain_thoughts(&chain, chain.thoughts().iter()),
            result_count: Some(chain.agent_registry().agents.len()),
            note: None,
        });
        Ok(ListAgentsResponse { chain_key, agents })
    }

    async fn get_agent(
        &self,
        request: GetAgentRequest,
    ) -> Result<AgentRecordResponse, Box<dyn Error + Send + Sync>> {
        let chain_key = self.resolve_chain_key(request.chain_key.as_deref());
        let chain = self.get_chain(Some(&chain_key), None).await?;
        let chain = chain.read().await;
        let agent = chain.get_agent(&request.agent_id).cloned().ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::NotFound,
                format!(
                    "No agent '{}' is registered in chain '{}'",
                    request.agent_id, chain_key
                ),
            )
        })?;
        self.log_interaction(InteractionLogEntry {
            access: "read",
            operation: "get_agent",
            chain_key: chain_key.clone(),
            metadata: InteractionMetadata::default(),
            result_count: Some(1),
            note: Some(format!("agent_id={}", request.agent_id)),
        });
        Ok(AgentRecordResponse { chain_key, agent })
    }

    async fn list_agent_registry(
        &self,
        request: ListAgentRegistryRequest,
    ) -> Result<AgentRegistryResponse, Box<dyn Error + Send + Sync>> {
        let chain_key = self.resolve_chain_key(request.chain_key.as_deref());
        let chain = self.get_chain(Some(&chain_key), None).await?;
        let chain = chain.read().await;
        let agents = chain
            .list_agent_registry()
            .into_iter()
            .cloned()
            .collect::<Vec<_>>();
        self.log_interaction(InteractionLogEntry {
            access: "read",
            operation: "list_agent_registry",
            chain_key: chain_key.clone(),
            metadata: InteractionMetadata::default(),
            result_count: Some(agents.len()),
            note: None,
        });
        Ok(AgentRegistryResponse { chain_key, agents })
    }

    async fn upsert_agent(
        &self,
        request: UpsertAgentRequest,
    ) -> Result<AgentRecordResponse, Box<dyn Error + Send + Sync>> {
        let chain_key = self.resolve_chain_key(request.chain_key.as_deref());
        let chain = self.get_chain(Some(&chain_key), None).await?;
        let mut chain = chain.write().await;
        let status = request
            .status
            .as_deref()
            .map(parse_agent_status)
            .transpose()?;
        let agent = chain.upsert_agent(
            &request.agent_id,
            request.display_name.as_deref(),
            request.agent_owner.as_deref(),
            request.description.as_deref(),
            status,
        )?;
        self.log_interaction(InteractionLogEntry {
            access: "write",
            operation: "upsert_agent",
            chain_key: chain_key.clone(),
            metadata: InteractionMetadata::default(),
            result_count: Some(1),
            note: Some(format!("agent_id={}", request.agent_id)),
        });
        Ok(AgentRecordResponse { chain_key, agent })
    }

    async fn set_agent_description(
        &self,
        request: SetAgentDescriptionRequest,
    ) -> Result<AgentRecordResponse, Box<dyn Error + Send + Sync>> {
        let chain_key = self.resolve_chain_key(request.chain_key.as_deref());
        let chain = self.get_chain(Some(&chain_key), None).await?;
        let mut chain = chain.write().await;
        let agent = chain.set_agent_description(&request.agent_id, request.description.as_deref())?;
        self.log_interaction(InteractionLogEntry {
            access: "write",
            operation: "set_agent_description",
            chain_key: chain_key.clone(),
            metadata: InteractionMetadata::default(),
            result_count: Some(1),
            note: Some(format!("agent_id={}", request.agent_id)),
        });
        Ok(AgentRecordResponse { chain_key, agent })
    }

    async fn add_agent_alias(
        &self,
        request: AddAgentAliasRequest,
    ) -> Result<AgentRecordResponse, Box<dyn Error + Send + Sync>> {
        let chain_key = self.resolve_chain_key(request.chain_key.as_deref());
        let chain = self.get_chain(Some(&chain_key), None).await?;
        let mut chain = chain.write().await;
        let agent = chain.add_agent_alias(&request.agent_id, &request.alias)?;
        self.log_interaction(InteractionLogEntry {
            access: "write",
            operation: "add_agent_alias",
            chain_key: chain_key.clone(),
            metadata: InteractionMetadata::default(),
            result_count: Some(1),
            note: Some(format!("agent_id={}", request.agent_id)),
        });
        Ok(AgentRecordResponse { chain_key, agent })
    }

    async fn add_agent_key(
        &self,
        request: AddAgentKeyRequest,
    ) -> Result<AgentRecordResponse, Box<dyn Error + Send + Sync>> {
        let chain_key = self.resolve_chain_key(request.chain_key.as_deref());
        let chain = self.get_chain(Some(&chain_key), None).await?;
        let mut chain = chain.write().await;
        let algorithm = parse_public_key_algorithm(&request.algorithm)?;
        let agent = chain.add_agent_key(
            &request.agent_id,
            &request.key_id,
            algorithm,
            request.public_key_bytes,
        )?;
        self.log_interaction(InteractionLogEntry {
            access: "write",
            operation: "add_agent_key",
            chain_key: chain_key.clone(),
            metadata: InteractionMetadata::default(),
            result_count: Some(1),
            note: Some(format!("agent_id={}", request.agent_id)),
        });
        Ok(AgentRecordResponse { chain_key, agent })
    }

    async fn revoke_agent_key(
        &self,
        request: RevokeAgentKeyRequest,
    ) -> Result<AgentRecordResponse, Box<dyn Error + Send + Sync>> {
        let chain_key = self.resolve_chain_key(request.chain_key.as_deref());
        let chain = self.get_chain(Some(&chain_key), None).await?;
        let mut chain = chain.write().await;
        let agent = chain.revoke_agent_key(&request.agent_id, &request.key_id)?;
        self.log_interaction(InteractionLogEntry {
            access: "write",
            operation: "revoke_agent_key",
            chain_key: chain_key.clone(),
            metadata: InteractionMetadata::default(),
            result_count: Some(1),
            note: Some(format!("agent_id={}", request.agent_id)),
        });
        Ok(AgentRecordResponse { chain_key, agent })
    }

    async fn disable_agent(
        &self,
        request: DisableAgentRequest,
    ) -> Result<AgentRecordResponse, Box<dyn Error + Send + Sync>> {
        let chain_key = self.resolve_chain_key(request.chain_key.as_deref());
        let chain = self.get_chain(Some(&chain_key), None).await?;
        let mut chain = chain.write().await;
        let agent = chain.disable_agent(&request.agent_id)?;
        self.log_interaction(InteractionLogEntry {
            access: "write",
            operation: "disable_agent",
            chain_key: chain_key.clone(),
            metadata: InteractionMetadata::default(),
            result_count: Some(1),
            note: Some(format!("agent_id={}", request.agent_id)),
        });
        Ok(AgentRecordResponse { chain_key, agent })
    }

    async fn recent_context(
        &self,
        request: RecentContextRequest,
    ) -> Result<RecentContextResponse, Box<dyn Error + Send + Sync>> {
        let chain_key = self.resolve_chain_key(request.chain_key.as_deref());
        let chain = self.get_chain(Some(&chain_key), None).await?;
        let chain = chain.read().await;
        let last_n = request.last_n.unwrap_or(12);
        let start = chain.thoughts().len().saturating_sub(last_n);
        let tail = &chain.thoughts()[start..];
        self.log_interaction(InteractionLogEntry {
            access: "read",
            operation: "recent_context",
            chain_key,
            metadata: InteractionMetadata::from_chain_thoughts(&chain, tail.iter()),
            result_count: Some(tail.len()),
            note: Some(format!("last_n={last_n}")),
        });
        Ok(RecentContextResponse {
            prompt: chain.to_catchup_prompt(last_n),
        })
    }

    async fn memory_markdown(
        &self,
        request: MemoryMarkdownRequest,
    ) -> Result<MemoryMarkdownResponse, Box<dyn Error + Send + Sync>> {
        let chain_key = self.resolve_chain_key(request.chain_key.as_deref());
        let chain = self.get_chain(Some(&chain_key), None).await?;
        let chain = chain.read().await;
        let query = build_markdown_query(&request)?;
        let matched = if query_is_empty(&query) {
            chain.thoughts().iter().collect::<Vec<_>>()
        } else {
            chain.query(&query)
        };
        self.log_interaction(InteractionLogEntry {
            access: "read",
            operation: "memory_markdown",
            chain_key,
            metadata: InteractionMetadata::from_chain_thoughts(&chain, matched.iter().copied()),
            result_count: Some(matched.len()),
            note: None,
        });
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
        let chain_key = self.resolve_chain_key(request.chain_key.as_deref());
        let chain = self.get_chain(Some(&chain_key), None).await?;
        let chain = chain.read().await;
        self.log_interaction(InteractionLogEntry {
            access: "read",
            operation: "head",
            chain_key: chain_key.clone(),
            metadata: chain
                .thoughts()
                .last()
                .map(|thought| InteractionMetadata::from_chain_thought(&chain, thought))
                .unwrap_or_default(),
            result_count: Some(chain.thoughts().len()),
            note: None,
        });
        Ok(HeadResponse {
            chain_key,
            thought_count: chain.thoughts().len(),
            head_hash: chain.head_hash().map(ToOwned::to_owned),
            latest_thought: chain
                .thoughts()
                .last()
                .map(|thought| thought_to_json(&chain, thought)),
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

    fn log_interaction(&self, entry: InteractionLogEntry) {
        if !self.config.verbose {
            return;
        }

        log::info!(target: "thoughtchain::interaction", "{}", format_interaction_log_entry(&entry));
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
struct InteractionMetadata {
    agent_ids: Vec<String>,
    agent_names: Vec<String>,
    thought_types: Vec<String>,
    roles: Vec<String>,
    tags: Vec<String>,
    concepts: Vec<String>,
}

impl InteractionMetadata {
    fn from_chain_thought(chain: &ThoughtChain, thought: &Thought) -> Self {
        Self::from_chain_thoughts(chain, std::iter::once(thought))
    }

    fn from_chain_thoughts<'a, I>(chain: &ThoughtChain, thoughts: I) -> Self
    where
        I: IntoIterator<Item = &'a Thought>,
    {
        let mut agent_ids = BTreeSet::new();
        let mut agent_names = BTreeSet::new();
        let mut thought_types = BTreeSet::new();
        let mut roles = BTreeSet::new();
        let mut tags = BTreeSet::new();
        let mut concepts = BTreeSet::new();

        for thought in thoughts {
            agent_ids.insert(thought.agent_id.clone());
            if let Some(agent_name) = chain
                .agent_registry()
                .agents
                .get(&thought.agent_id)
                .map(|record| record.display_name.clone())
                .filter(|value| !value.trim().is_empty())
            {
                agent_names.insert(agent_name);
            }
            thought_types.insert(format!("{:?}", thought.thought_type));
            roles.insert(format!("{:?}", thought.role));
            tags.extend(thought.tags.iter().cloned());
            concepts.extend(thought.concepts.iter().cloned());
        }

        Self {
            agent_ids: agent_ids.into_iter().collect(),
            agent_names: agent_names.into_iter().collect(),
            thought_types: thought_types.into_iter().collect(),
            roles: roles.into_iter().collect(),
            tags: tags.into_iter().collect(),
            concepts: concepts.into_iter().collect(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct InteractionLogEntry {
    access: &'static str,
    operation: &'static str,
    chain_key: String,
    metadata: InteractionMetadata,
    result_count: Option<usize>,
    note: Option<String>,
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
    storage_adapter: Option<String>,
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
    signing_key_id: Option<String>,
    thought_signature: Option<Vec<u8>>,
    thought_type: String,
    content: String,
    role: Option<String>,
    importance: Option<f32>,
    confidence: Option<f32>,
    tags: Option<Vec<String>>,
    concepts: Option<Vec<String>>,
    refs: Option<Vec<u64>>,
}

#[derive(Debug, Deserialize)]
struct AppendRetrospectiveRequest {
    chain_key: Option<String>,
    agent_id: Option<String>,
    agent_name: Option<String>,
    agent_owner: Option<String>,
    signing_key_id: Option<String>,
    thought_signature: Option<Vec<u8>>,
    thought_type: Option<String>,
    content: String,
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

#[derive(Debug, Deserialize, Default)]
struct ListAgentsRequest {
    chain_key: Option<String>,
}

#[derive(Debug, Deserialize)]
struct GetAgentRequest {
    chain_key: Option<String>,
    agent_id: String,
}

#[derive(Debug, Deserialize, Default)]
struct ListAgentRegistryRequest {
    chain_key: Option<String>,
}

#[derive(Debug, Deserialize)]
struct UpsertAgentRequest {
    chain_key: Option<String>,
    agent_id: String,
    display_name: Option<String>,
    agent_owner: Option<String>,
    description: Option<String>,
    status: Option<String>,
}

#[derive(Debug, Deserialize)]
struct SetAgentDescriptionRequest {
    chain_key: Option<String>,
    agent_id: String,
    description: Option<String>,
}

#[derive(Debug, Deserialize)]
struct AddAgentAliasRequest {
    chain_key: Option<String>,
    agent_id: String,
    alias: String,
}

#[derive(Debug, Deserialize)]
struct AddAgentKeyRequest {
    chain_key: Option<String>,
    agent_id: String,
    key_id: String,
    algorithm: String,
    public_key_bytes: Vec<u8>,
}

#[derive(Debug, Deserialize)]
struct RevokeAgentKeyRequest {
    chain_key: Option<String>,
    agent_id: String,
    key_id: String,
}

#[derive(Debug, Deserialize)]
struct DisableAgentRequest {
    chain_key: Option<String>,
    agent_id: String,
}

#[derive(Debug, Serialize)]
struct ListChainsResponse {
    default_chain_key: String,
    chain_keys: Vec<String>,
    chains: Vec<ChainSummary>,
}

#[derive(Debug, Serialize)]
struct ChainSummary {
    chain_key: String,
    version: u32,
    storage_adapter: String,
    thought_count: u64,
    agent_count: usize,
    storage_location: String,
}

#[derive(Debug, Serialize, PartialEq, Eq)]
struct AgentIdentitySummary {
    agent_id: String,
    agent_name: String,
    agent_owner: Option<String>,
}

#[derive(Debug, Serialize)]
struct ListAgentsResponse {
    chain_key: String,
    agents: Vec<AgentIdentitySummary>,
}

#[derive(Debug, Serialize)]
struct AgentRecordResponse {
    chain_key: String,
    agent: AgentRecord,
}

#[derive(Debug, Serialize)]
struct AgentRegistryResponse {
    chain_key: String,
    agents: Vec<AgentRecord>,
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

async fn start_router(
    addr: SocketAddr,
    router: Router,
) -> Result<ServerHandle, Box<dyn Error + Send + Sync>> {
    let listener = TcpListener::bind(addr).await?;
    let local_addr = listener.local_addr()?;
    let (shutdown_tx, shutdown_rx) = oneshot::channel();

    tokio::spawn(async move {
        let _ = axum::serve(
            listener,
            router.into_make_service_with_connect_info::<SocketAddr>(),
        )
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
    let protocol = ThoughtChainMcpProtocol::new(service);

    match protocol.execute(&request.tool, request.parameters).await {
        Ok(result) => (StatusCode::OK, Json(json!({ "result": result }))),
        Err(error) => (
            StatusCode::BAD_REQUEST,
            Json(json!({ "result": ToolResult::failure(error.to_string()) })),
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

async fn rest_append_retrospective_handler(
    State(service): State<Arc<ThoughtChainService>>,
    Json(request): Json<AppendRetrospectiveRequest>,
) -> Result<Json<AppendThoughtResponse>, (StatusCode, Json<Value>)> {
    service_call(service.append_retrospective(request).await)
}

async fn rest_search_handler(
    State(service): State<Arc<ThoughtChainService>>,
    Json(request): Json<SearchRequest>,
) -> Result<Json<SearchResponse>, (StatusCode, Json<Value>)> {
    service_call(service.search(request).await)
}

async fn rest_list_chains_handler(
    State(service): State<Arc<ThoughtChainService>>,
) -> Result<Json<ListChainsResponse>, (StatusCode, Json<Value>)> {
    service_call(service.list_chains().await)
}

async fn rest_list_agents_handler(
    State(service): State<Arc<ThoughtChainService>>,
    Json(request): Json<ListAgentsRequest>,
) -> Result<Json<ListAgentsResponse>, (StatusCode, Json<Value>)> {
    service_call(service.list_agents(request).await)
}

async fn rest_get_agent_handler(
    State(service): State<Arc<ThoughtChainService>>,
    Json(request): Json<GetAgentRequest>,
) -> Result<Json<AgentRecordResponse>, (StatusCode, Json<Value>)> {
    service_call(service.get_agent(request).await)
}

async fn rest_list_agent_registry_handler(
    State(service): State<Arc<ThoughtChainService>>,
    Json(request): Json<ListAgentRegistryRequest>,
) -> Result<Json<AgentRegistryResponse>, (StatusCode, Json<Value>)> {
    service_call(service.list_agent_registry(request).await)
}

async fn rest_upsert_agent_handler(
    State(service): State<Arc<ThoughtChainService>>,
    Json(request): Json<UpsertAgentRequest>,
) -> Result<Json<AgentRecordResponse>, (StatusCode, Json<Value>)> {
    service_call(service.upsert_agent(request).await)
}

async fn rest_set_agent_description_handler(
    State(service): State<Arc<ThoughtChainService>>,
    Json(request): Json<SetAgentDescriptionRequest>,
) -> Result<Json<AgentRecordResponse>, (StatusCode, Json<Value>)> {
    service_call(service.set_agent_description(request).await)
}

async fn rest_add_agent_alias_handler(
    State(service): State<Arc<ThoughtChainService>>,
    Json(request): Json<AddAgentAliasRequest>,
) -> Result<Json<AgentRecordResponse>, (StatusCode, Json<Value>)> {
    service_call(service.add_agent_alias(request).await)
}

async fn rest_add_agent_key_handler(
    State(service): State<Arc<ThoughtChainService>>,
    Json(request): Json<AddAgentKeyRequest>,
) -> Result<Json<AgentRecordResponse>, (StatusCode, Json<Value>)> {
    service_call(service.add_agent_key(request).await)
}

async fn rest_revoke_agent_key_handler(
    State(service): State<Arc<ThoughtChainService>>,
    Json(request): Json<RevokeAgentKeyRequest>,
) -> Result<Json<AgentRecordResponse>, (StatusCode, Json<Value>)> {
    service_call(service.revoke_agent_key(request).await)
}

async fn rest_disable_agent_handler(
    State(service): State<Arc<ThoughtChainService>>,
    Json(request): Json<DisableAgentRequest>,
) -> Result<Json<AgentRecordResponse>, (StatusCode, Json<Value>)> {
    service_call(service.disable_agent(request).await)
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

fn mcp_tool_metadata() -> Vec<ToolMetadata> {
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
            ToolParameter::new("agent_id", ToolParameterType::String)
                .with_description("Optional producing agent id. Defaults to 'system' for bootstrap."),
        )
        .with_parameter(
            ToolParameter::new("agent_name", ToolParameterType::String)
                .with_description("Optional producing agent name."),
        )
        .with_parameter(
            ToolParameter::new("agent_owner", ToolParameterType::String)
                .with_description("Optional producing agent owner or tenant label."),
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
            "Append a durable semantic memory to ThoughtChain. Use exact ThoughtType names like PreferenceUpdate, Constraint, Decision, Insight, Wonder, Question, Summary, Mistake, or Correction.",
        )
        .with_parameter(ToolParameter::new("chain_key", ToolParameterType::String).with_description("Optional durable chain key."))
        .with_parameter(ToolParameter::new("agent_id", ToolParameterType::String).with_description("Optional producing agent id. Defaults to the chain key when omitted."))
        .with_parameter(ToolParameter::new("agent_name", ToolParameterType::String).with_description("Optional producing agent name."))
        .with_parameter(ToolParameter::new("agent_owner", ToolParameterType::String).with_description("Optional producing agent owner or tenant label."))
        .with_parameter(ToolParameter::new("thought_type", ToolParameterType::String).with_description("Semantic type of the thought.").required())
        .with_parameter(ToolParameter::new("content", ToolParameterType::String).with_description("Concise durable memory content.").required())
        .with_parameter(ToolParameter::new("role", ToolParameterType::String).with_description("Optional thought role such as Memory, Summary, Compression, Checkpoint, or Handoff."))
        .with_parameter(ToolParameter::new("importance", ToolParameterType::Number).with_description("Optional importance score between 0.0 and 1.0."))
        .with_parameter(ToolParameter::new("confidence", ToolParameterType::Number).with_description("Optional confidence score between 0.0 and 1.0."))
        .with_parameter(ToolParameter::new("tags", ToolParameterType::Array).with_description("Optional tags.").with_items(ToolParameterType::String))
        .with_parameter(ToolParameter::new("concepts", ToolParameterType::Array).with_description("Optional semantic concepts.").with_items(ToolParameterType::String))
        .with_parameter(ToolParameter::new("refs", ToolParameterType::Array).with_description("Optional referenced thought indices.").with_items(ToolParameterType::Integer))
        .with_parameter(ToolParameter::new("signing_key_id", ToolParameterType::String).with_description("Optional key id used to verify the detached thought signature."))
        .with_parameter(ToolParameter::new("thought_signature", ToolParameterType::Array).with_description("Optional detached signature bytes for the signable thought payload.").with_items(ToolParameterType::Integer)),
        ToolMetadata::new(
            "thoughtchain_append_retrospective",
            "Append a guided retrospective memory after a hard failure, repeated snag, or non-obvious fix. Prefer this over thoughtchain_append when you want future agents to avoid repeating the same struggle. This tool defaults to ThoughtType LessonLearned and always records the thought with role Retrospective.",
        )
        .with_parameter(ToolParameter::new("chain_key", ToolParameterType::String).with_description("Optional durable chain key."))
        .with_parameter(ToolParameter::new("agent_id", ToolParameterType::String).with_description("Optional producing agent id. Defaults to the chain key when omitted."))
        .with_parameter(ToolParameter::new("agent_name", ToolParameterType::String).with_description("Optional producing agent name."))
        .with_parameter(ToolParameter::new("agent_owner", ToolParameterType::String).with_description("Optional producing agent owner or tenant label."))
        .with_parameter(ToolParameter::new("thought_type", ToolParameterType::String).with_description("Optional retrospective thought type. Defaults to LessonLearned. Useful alternatives include Mistake, Correction, AssumptionInvalidated, StrategyShift, Insight, or Summary."))
        .with_parameter(ToolParameter::new("content", ToolParameterType::String).with_description("Concise lesson, correction, or operating guidance distilled from the struggle.").required())
        .with_parameter(ToolParameter::new("importance", ToolParameterType::Number).with_description("Optional importance score between 0.0 and 1.0. Defaults to 0.7."))
        .with_parameter(ToolParameter::new("confidence", ToolParameterType::Number).with_description("Optional confidence score between 0.0 and 1.0."))
        .with_parameter(ToolParameter::new("tags", ToolParameterType::Array).with_description("Optional tags.").with_items(ToolParameterType::String))
        .with_parameter(ToolParameter::new("concepts", ToolParameterType::Array).with_description("Optional semantic concepts.").with_items(ToolParameterType::String))
        .with_parameter(ToolParameter::new("refs", ToolParameterType::Array).with_description("Optional referenced thought indices, such as the mistake, correction, or earlier checkpoint that motivated the lesson.").with_items(ToolParameterType::Integer))
        .with_parameter(ToolParameter::new("signing_key_id", ToolParameterType::String).with_description("Optional key id used to verify the detached thought signature."))
        .with_parameter(ToolParameter::new("thought_signature", ToolParameterType::Array).with_description("Optional detached signature bytes for the signable thought payload.").with_items(ToolParameterType::Integer)),
        ToolMetadata::new(
            "thoughtchain_search",
            "Search durable memories by text, type, role, tags, concepts, and importance.",
        )
        .with_parameter(ToolParameter::new("chain_key", ToolParameterType::String).with_description("Optional durable chain key."))
        .with_parameter(ToolParameter::new("text", ToolParameterType::String).with_description("Optional text filter applied to content, tags, and concepts."))
        .with_parameter(ToolParameter::new("thought_types", ToolParameterType::Array).with_description("Optional list of ThoughtType names.").with_items(ToolParameterType::String))
        .with_parameter(ToolParameter::new("roles", ToolParameterType::Array).with_description("Optional list of ThoughtRole names.").with_items(ToolParameterType::String))
        .with_parameter(ToolParameter::new("tags_any", ToolParameterType::Array).with_description("Optional tags to match.").with_items(ToolParameterType::String))
        .with_parameter(ToolParameter::new("concepts_any", ToolParameterType::Array).with_description("Optional concepts to match.").with_items(ToolParameterType::String))
        .with_parameter(ToolParameter::new("agent_ids", ToolParameterType::Array).with_description("Optional producing agent ids to match.").with_items(ToolParameterType::String))
        .with_parameter(ToolParameter::new("agent_names", ToolParameterType::Array).with_description("Optional producing agent names to match.").with_items(ToolParameterType::String))
        .with_parameter(ToolParameter::new("agent_owners", ToolParameterType::Array).with_description("Optional producing agent owners to match.").with_items(ToolParameterType::String))
        .with_parameter(ToolParameter::new("min_importance", ToolParameterType::Number).with_description("Optional minimum importance threshold."))
        .with_parameter(ToolParameter::new("min_confidence", ToolParameterType::Number).with_description("Optional minimum confidence threshold."))
        .with_parameter(ToolParameter::new("since", ToolParameterType::String).with_description("Optional RFC 3339 lower timestamp bound."))
        .with_parameter(ToolParameter::new("until", ToolParameterType::String).with_description("Optional RFC 3339 upper timestamp bound."))
        .with_parameter(ToolParameter::new("limit", ToolParameterType::Integer).with_description("Optional maximum number of results.")),
        ToolMetadata::new(
            "thoughtchain_list_chains",
            "List the durable chain keys currently available in ThoughtChain storage, together with the server default chain key.",
        ),
        ToolMetadata::new(
            "thoughtchain_list_agents",
            "List the distinct agent identities that have written to a particular chain key. Use this to discover participating agents on a shared brain before filtering searches by agent.",
        )
        .with_parameter(ToolParameter::new("chain_key", ToolParameterType::String).with_description("Optional durable chain key. Defaults to the server default chain.")),
        ToolMetadata::new(
            "thoughtchain_get_agent",
            "Return the full registry record for one agent in a chain, including description, aliases, public keys, status, and per-chain activity metadata.",
        )
        .with_parameter(ToolParameter::new("chain_key", ToolParameterType::String).with_description("Optional durable chain key. Defaults to the server default chain."))
        .with_parameter(ToolParameter::new("agent_id", ToolParameterType::String).with_description("Stable agent id to retrieve.").required()),
        ToolMetadata::new(
            "thoughtchain_list_agent_registry",
            "Return the full per-chain agent registry, including descriptions, aliases, public keys, status, and per-chain activity metadata for every registered agent.",
        )
        .with_parameter(ToolParameter::new("chain_key", ToolParameterType::String).with_description("Optional durable chain key. Defaults to the server default chain.")),
        ToolMetadata::new(
            "thoughtchain_upsert_agent",
            "Create or update one agent registry record so a chain can track agent metadata even before the agent writes thoughts.",
        )
        .with_parameter(ToolParameter::new("chain_key", ToolParameterType::String).with_description("Optional durable chain key. Defaults to the server default chain."))
        .with_parameter(ToolParameter::new("agent_id", ToolParameterType::String).with_description("Stable agent id to create or update.").required())
        .with_parameter(ToolParameter::new("display_name", ToolParameterType::String).with_description("Optional friendly display name for the agent."))
        .with_parameter(ToolParameter::new("agent_owner", ToolParameterType::String).with_description("Optional owner, tenant, or grouping label for the agent."))
        .with_parameter(ToolParameter::new("description", ToolParameterType::String).with_description("Optional free-form description of what the agent does."))
        .with_parameter(ToolParameter::new("status", ToolParameterType::String).with_description("Optional lifecycle status. Supported values: active, revoked.")),
        ToolMetadata::new(
            "thoughtchain_set_agent_description",
            "Set or clear the free-form description for one registered agent.",
        )
        .with_parameter(ToolParameter::new("chain_key", ToolParameterType::String).with_description("Optional durable chain key. Defaults to the server default chain."))
        .with_parameter(ToolParameter::new("agent_id", ToolParameterType::String).with_description("Stable agent id to update.").required())
        .with_parameter(ToolParameter::new("description", ToolParameterType::String).with_description("Description to store. Omit or use an empty string to clear.")),
        ToolMetadata::new(
            "thoughtchain_add_agent_alias",
            "Add one historical or alternate alias to a registered agent.",
        )
        .with_parameter(ToolParameter::new("chain_key", ToolParameterType::String).with_description("Optional durable chain key. Defaults to the server default chain."))
        .with_parameter(ToolParameter::new("agent_id", ToolParameterType::String).with_description("Stable agent id to update.").required())
        .with_parameter(ToolParameter::new("alias", ToolParameterType::String).with_description("Alias to add to the agent record.").required()),
        ToolMetadata::new(
            "thoughtchain_add_agent_key",
            "Add or replace one public verification key on a registered agent. This is the intended path for future signed-thought workflows.",
        )
        .with_parameter(ToolParameter::new("chain_key", ToolParameterType::String).with_description("Optional durable chain key. Defaults to the server default chain."))
        .with_parameter(ToolParameter::new("agent_id", ToolParameterType::String).with_description("Stable agent id to update.").required())
        .with_parameter(ToolParameter::new("key_id", ToolParameterType::String).with_description("Stable identifier for the public key.").required())
        .with_parameter(ToolParameter::new("algorithm", ToolParameterType::String).with_description("Public-key algorithm. Currently supported: ed25519.").required())
        .with_parameter(ToolParameter::new("public_key_bytes", ToolParameterType::Array).with_description("Raw public-key bytes.").with_items(ToolParameterType::Integer).required()),
        ToolMetadata::new(
            "thoughtchain_revoke_agent_key",
            "Mark one previously registered public key as revoked for a given agent.",
        )
        .with_parameter(ToolParameter::new("chain_key", ToolParameterType::String).with_description("Optional durable chain key. Defaults to the server default chain."))
        .with_parameter(ToolParameter::new("agent_id", ToolParameterType::String).with_description("Stable agent id to update.").required())
        .with_parameter(ToolParameter::new("key_id", ToolParameterType::String).with_description("Stable identifier for the public key to revoke.").required()),
        ToolMetadata::new(
            "thoughtchain_disable_agent",
            "Disable one agent by marking its registry status as revoked.",
        )
        .with_parameter(ToolParameter::new("chain_key", ToolParameterType::String).with_description("Optional durable chain key. Defaults to the server default chain."))
        .with_parameter(ToolParameter::new("agent_id", ToolParameterType::String).with_description("Stable agent id to disable.").required()),
        ToolMetadata::new(
            "thoughtchain_recent_context",
            "Render recent ThoughtChain context as a prompt snippet suitable for resuming work.",
        )
        .with_parameter(ToolParameter::new("chain_key", ToolParameterType::String).with_description("Optional durable chain key."))
        .with_parameter(ToolParameter::new("last_n", ToolParameterType::Integer).with_description("How many recent thoughts to include.")),
        ToolMetadata::new(
            "thoughtchain_memory_markdown",
            "Export a MEMORY.md style Markdown summary from ThoughtChain.",
        )
        .with_parameter(ToolParameter::new("chain_key", ToolParameterType::String).with_description("Optional durable chain key."))
        .with_parameter(ToolParameter::new("text", ToolParameterType::String).with_description("Optional text filter."))
        .with_parameter(ToolParameter::new("thought_types", ToolParameterType::Array).with_description("Optional list of ThoughtType names.").with_items(ToolParameterType::String))
        .with_parameter(ToolParameter::new("roles", ToolParameterType::Array).with_description("Optional list of ThoughtRole names.").with_items(ToolParameterType::String))
        .with_parameter(ToolParameter::new("tags_any", ToolParameterType::Array).with_description("Optional tags to match.").with_items(ToolParameterType::String))
        .with_parameter(ToolParameter::new("concepts_any", ToolParameterType::Array).with_description("Optional concepts to match.").with_items(ToolParameterType::String))
        .with_parameter(ToolParameter::new("agent_ids", ToolParameterType::Array).with_description("Optional producing agent ids to match.").with_items(ToolParameterType::String))
        .with_parameter(ToolParameter::new("agent_names", ToolParameterType::Array).with_description("Optional producing agent names to match.").with_items(ToolParameterType::String))
        .with_parameter(ToolParameter::new("agent_owners", ToolParameterType::Array).with_description("Optional producing agent owners to match.").with_items(ToolParameterType::String))
        .with_parameter(ToolParameter::new("min_importance", ToolParameterType::Number).with_description("Optional minimum importance threshold."))
        .with_parameter(ToolParameter::new("min_confidence", ToolParameterType::Number).with_description("Optional minimum confidence threshold."))
        .with_parameter(ToolParameter::new("since", ToolParameterType::String).with_description("Optional RFC 3339 lower timestamp bound."))
        .with_parameter(ToolParameter::new("until", ToolParameterType::String).with_description("Optional RFC 3339 upper timestamp bound."))
        .with_parameter(ToolParameter::new("limit", ToolParameterType::Integer).with_description("Optional maximum number of thoughts.")),
        ToolMetadata::new(
            "thoughtchain_head",
            "Return head metadata for a ThoughtChain including length, latest thought, and head hash.",
        )
        .with_parameter(ToolParameter::new("chain_key", ToolParameterType::String).with_description("Optional durable chain key.")),
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
        "lessonlearned" => ThoughtType::LessonLearned,
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
        "retrospective" => ThoughtRole::Retrospective,
        _ => return Err(format!("Unknown ThoughtRole '{input}'").into()),
    };

    Ok(role)
}

fn parse_storage_adapter_kind(
    input: &str,
) -> Result<StorageAdapterKind, Box<dyn Error + Send + Sync>> {
    input
        .parse::<StorageAdapterKind>()
        .map_err(|error| error.into())
}

fn parse_agent_status(input: &str) -> Result<AgentStatus, Box<dyn Error + Send + Sync>> {
    input.parse::<AgentStatus>().map_err(|error| error.into())
}

fn parse_public_key_algorithm(
    input: &str,
) -> Result<PublicKeyAlgorithm, Box<dyn Error + Send + Sync>> {
    input
        .parse::<PublicKeyAlgorithm>()
        .map_err(|error| error.into())
}

fn infer_storage_adapter_name(storage_location: &str) -> String {
    if storage_location.ends_with(".tcbin") {
        StorageAdapterKind::Binary.to_string()
    } else if storage_location.ends_with(".jsonl") {
        StorageAdapterKind::Jsonl.to_string()
    } else {
        "unknown".to_string()
    }
}

fn thought_to_json(chain: &ThoughtChain, thought: &Thought) -> Value {
    chain.thought_json(thought)
}

fn normalize_label(input: &str) -> String {
    input
        .chars()
        .filter(|character| character.is_ascii_alphanumeric())
        .collect::<String>()
        .to_lowercase()
}

fn format_interaction_log_entry(entry: &InteractionLogEntry) -> String {
    let metadata = &entry.metadata;
    let mut log_line = format!(
        "[thoughtchaind] access={} op={} chain={} result_count={} agent_ids={} agent_names={} thought_types={} roles={} tags={} concepts={}",
        entry.access,
        entry.operation,
        entry.chain_key,
        entry
            .result_count
            .map(|count| count.to_string())
            .unwrap_or_else(|| "-".to_string()),
        summarize_values(&metadata.agent_ids),
        summarize_values(&metadata.agent_names),
        summarize_values(&metadata.thought_types),
        summarize_values(&metadata.roles),
        summarize_values(&metadata.tags),
        summarize_values(&metadata.concepts),
    );

    if let Some(note) = &entry.note {
        log_line.push_str(" note=");
        log_line.push_str(note);
    }

    log_line
}

fn summarize_values(values: &[String]) -> String {
    const MAX_ITEMS: usize = 8;

    if values.is_empty() {
        return "-".to_string();
    }

    if values.len() <= MAX_ITEMS {
        return values.join(",");
    }

    format!(
        "{}...(+{} more)",
        values[..MAX_ITEMS].join(","),
        values.len() - MAX_ITEMS
    )
}

fn env_u16(key: &str) -> Option<u16> {
    std::env::var(key)
        .ok()
        .and_then(|value| value.parse::<u16>().ok())
}

fn env_bool(key: &str) -> Option<bool> {
    std::env::var(key)
        .ok()
        .and_then(|value| parse_bool_flag(&value))
}

fn parse_bool_flag(value: &str) -> Option<bool> {
    match value.trim().to_ascii_lowercase().as_str() {
        "1" | "true" => Some(true),
        "0" | "false" => Some(false),
        _ => None,
    }
}
