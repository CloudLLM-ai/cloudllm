//! Planner abstraction for orchestrating a single agent turn.
//!
//! A Planner coordinates context assembly, tool usage, policy checks, and streaming
//! output for a single agent turn while keeping the orchestration logic separate
//! from the `Agent` identity and session model.
//!
//! # Example: end-to-end planner usage
//!
//! ```rust,no_run
//! use std::sync::Arc;
//!
//! use cloudllm::clients::openai::{Model, OpenAIClient};
//! use cloudllm::tool_protocol::{ToolMetadata, ToolRegistry, ToolResult};
//! use cloudllm::tool_protocols::CustomToolProtocol;
//! use cloudllm::planner::{
//!     BasicPlanner, NoopMemory, NoopPolicy, NoopStream, Planner, PlannerContext, UserMessage,
//! };
//! use cloudllm::LLMSession;
//!
//! # #[tokio::main]
//! # async fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let client = Arc::new(OpenAIClient::new_with_model_enum(
//!     &std::env::var("OPEN_AI_SECRET")?,
//!     Model::GPT41Mini,
//! ));
//!
//! let mut session = LLMSession::new(client, "You are concise.".into(), 16_000);
//!
//! let protocol = Arc::new(CustomToolProtocol::new());
//! protocol
//!     .register_tool(
//!         ToolMetadata::new("add", "Add two numbers"),
//!         Arc::new(|params| {
//!             let a = params["a"].as_f64().unwrap_or(0.0);
//!             let b = params["b"].as_f64().unwrap_or(0.0);
//!             Ok(ToolResult::success(serde_json::json!({ "result": a + b })))
//!         }),
//!     )
//!     .await;
//!
//! let mut tools = ToolRegistry::new(protocol);
//! tools.discover_tools_from_primary().await?;
//!
//! let planner = BasicPlanner::new();
//! let outcome = planner
//!     .plan(
//!         UserMessage::from("What is 2+2? Use the add tool."),
//!         PlannerContext {
//!             session: &mut session,
//!             tools: &tools,
//!             policy: &NoopPolicy,
//!             memory: &NoopMemory,
//!             streamer: &NoopStream,
//!             grok_tools: None,
//!             openai_tools: None,
//!             event_handler: None,
//!         },
//!     )
//!     .await?;
//!
//! println!("{}", outcome.final_message);
//! # Ok(())
//! # }
//! ```

use crate::client_wrapper::{Role, TokenUsage};
use crate::cloudllm::event::{EventHandler, PlannerEvent};
use crate::cloudllm::llm_session::LLMSession;
use crate::cloudllm::tool_protocol::{ToolRegistry, ToolResult};
use async_trait::async_trait;
use openai_rust2::chat::{GrokTool, OpenAITool};
use serde_json::Value;
use std::collections::HashMap;
use std::error::Error;
use std::io;
use uuid::Uuid;

pub type PlannerResult<T> = Result<T, Box<dyn Error + Send + Sync>>;

/// Represents the new user input for a planner turn.
///
/// This is the minimal input structure a planner needs to operate. More elaborate
/// systems can extend or wrap it, but the planner only requires the message content.
///
/// # Example
///
/// ```rust
/// use cloudllm::planner::UserMessage;
///
/// let input = UserMessage::from("Summarize this for me");
/// assert!(input.content.contains("Summarize"));
/// ```
#[derive(Debug, Clone)]
pub struct UserMessage {
    /// The raw user-provided content for the planner turn.
    pub content: String,
}

/// Convert an owned string into a [`UserMessage`].
///
/// # Example
///
/// ```rust
/// use cloudllm::planner::UserMessage;
///
/// let input = UserMessage::from("Hello".to_string());
/// assert_eq!(input.content, "Hello");
/// ```
impl From<String> for UserMessage {
    fn from(content: String) -> Self {
        Self { content }
    }
}

/// Convert a string slice into a [`UserMessage`].
///
/// # Example
///
/// ```rust
/// use cloudllm::planner::UserMessage;
///
/// let input = UserMessage::from("Hello");
/// assert_eq!(input.content, "Hello");
/// ```
impl From<&str> for UserMessage {
    fn from(content: &str) -> Self {
        Self {
            content: content.to_string(),
        }
    }
}

/// Represents a memory item retrieved for the current turn.
///
/// Planners can use this to add semantic recall or persisted context to the
/// prompt they send to the model.
///
/// # Example
///
/// ```rust
/// use cloudllm::planner::MemoryEntry;
///
/// let entry = MemoryEntry::new("User prefers short answers")
///     .with_metadata("source", "preferences");
/// assert_eq!(entry.metadata.get("source").unwrap(), "preferences");
/// ```
#[derive(Debug, Clone)]
pub struct MemoryEntry {
    /// The memory content to be injected into the prompt.
    pub content: String,
    /// Optional metadata such as source, timestamp, or tags.
    pub metadata: HashMap<String, String>,
}

impl MemoryEntry {
    /// Create a new memory entry with plain content.
    ///
    /// # Example
    ///
    /// ```rust
    /// use cloudllm::planner::MemoryEntry;
    ///
    /// let entry = MemoryEntry::new("Pinned note");
    /// assert_eq!(entry.content, "Pinned note");
    /// ```
    pub fn new(content: impl Into<String>) -> Self {
        Self {
            content: content.into(),
            metadata: HashMap::new(),
        }
    }

    /// Attach metadata to the memory entry (builder pattern).
    ///
    /// # Example
    ///
    /// ```rust
    /// use cloudllm::planner::MemoryEntry;
    ///
    /// let entry = MemoryEntry::new("Pinned note")
    ///     .with_metadata("source", "manual");
    /// assert_eq!(entry.metadata.get("source").unwrap(), "manual");
    /// ```
    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }
}

/// Represents a tool call requested by the model.
///
/// Planners should validate this request with a [`PolicyEngine`] before executing.
///
/// # Example
///
/// ```rust
/// use cloudllm::planner::ToolCallRequest;
///
/// let call = ToolCallRequest {
///     name: "calculator".to_string(),
///     parameters: serde_json::json!({"expr": "2+2"}),
/// };
/// assert_eq!(call.name, "calculator");
/// ```
#[derive(Debug, Clone)]
pub struct ToolCallRequest {
    /// Tool name to execute (must exist in the registry).
    pub name: String,
    /// JSON parameters for the tool execution.
    pub parameters: Value,
}

/// Decision returned by a [`PolicyEngine`] for tool usage.
///
/// `Deny` should be treated as a hard stop for the tool call.
///
/// # Example
///
/// ```rust
/// use cloudllm::planner::PolicyDecision;
///
/// let decision = PolicyDecision::Deny("not allowed".to_string());
/// match decision {
///     PolicyDecision::Allow => unreachable!(),
///     PolicyDecision::Deny(reason) => assert_eq!(reason, "not allowed"),
/// }
/// ```
#[derive(Debug, Clone)]
pub enum PolicyDecision {
    Allow,
    Deny(String),
}

/// Policy hook for tool authorization.
///
/// Implementations can enforce allow/deny rules, rate limits, or capability scopes.
///
/// # Example
///
/// ```rust
/// use async_trait::async_trait;
/// use cloudllm::planner::{PolicyDecision, PolicyEngine, ToolCallRequest};
///
/// struct DenyAll;
///
/// #[async_trait]
/// impl PolicyEngine for DenyAll {
///     async fn allow_tool_call(
///         &self,
///         _call: &ToolCallRequest,
///     ) -> Result<PolicyDecision, Box<dyn std::error::Error + Send + Sync>> {
///         Ok(PolicyDecision::Deny("blocked".to_string()))
///     }
/// }
/// ```
#[async_trait]
pub trait PolicyEngine: Send + Sync {
    /// Decide whether a tool call should be allowed.
    ///
    /// # Parameters
    ///
    /// * `call` - The parsed tool call request from the model.
    ///
    /// # Example
    ///
    /// ```rust
    /// use async_trait::async_trait;
    /// use cloudllm::planner::{PolicyDecision, PolicyEngine, ToolCallRequest};
    ///
    /// struct AllowAll;
    ///
    /// #[async_trait]
    /// impl PolicyEngine for AllowAll {
    ///     async fn allow_tool_call(
    ///         &self,
    ///         _call: &ToolCallRequest,
    ///     ) -> Result<PolicyDecision, Box<dyn std::error::Error + Send + Sync>> {
    ///         Ok(PolicyDecision::Allow)
    ///     }
    /// }
    /// ```
    async fn allow_tool_call(&self, call: &ToolCallRequest) -> PlannerResult<PolicyDecision>;
}

/// Default policy implementation that allows all tool calls.
pub struct NoopPolicy;

#[async_trait]
impl PolicyEngine for NoopPolicy {
    /// Always returns [`PolicyDecision::Allow`].
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use cloudllm::planner::{NoopPolicy, PolicyEngine, PolicyDecision, ToolCallRequest};
    ///
    /// # async fn example() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    /// let policy = NoopPolicy;
    /// let decision = policy
    ///     .allow_tool_call(&ToolCallRequest {
    ///         name: "calculator".to_string(),
    ///         parameters: serde_json::json!({"expr": "2+2"}),
    ///     })
    ///     .await?;
    /// assert!(matches!(decision, PolicyDecision::Allow));
    /// # Ok(())
    /// # }
    /// ```
    async fn allow_tool_call(&self, _call: &ToolCallRequest) -> PlannerResult<PolicyDecision> {
        Ok(PolicyDecision::Allow)
    }
}

/// Abstraction for semantic memory retrieval and persistence.
///
/// Most applications will inject a concrete implementation that fetches relevant
/// memories for the current user message and optionally persists new memories.
///
/// # Example
///
/// ```rust
/// use cloudllm::planner::{MemoryEntry, MemoryStore, UserMessage};
///
/// struct StaticMemory;
///
/// impl MemoryStore for StaticMemory {
///     fn retrieve(
///         &self,
///         _input: &UserMessage,
///     ) -> Result<Vec<MemoryEntry>, Box<dyn std::error::Error + Send + Sync>> {
///         Ok(vec![MemoryEntry::new("Pinned note")])
///     }
/// }
/// ```
pub trait MemoryStore: Send + Sync {
    /// Retrieve memory entries relevant to the current user input.
    ///
    /// # Parameters
    ///
    /// * `input` - The current user message.
    ///
    /// # Example
    ///
    /// ```rust
    /// use cloudllm::planner::{MemoryEntry, MemoryStore, UserMessage};
    ///
    /// struct StaticMemory;
    ///
    /// impl MemoryStore for StaticMemory {
    ///     fn retrieve(
    ///         &self,
    ///         _input: &UserMessage,
    ///     ) -> Result<Vec<MemoryEntry>, Box<dyn std::error::Error + Send + Sync>> {
    ///         Ok(vec![MemoryEntry::new("Pinned note")])
    ///     }
    /// }
    /// ```
    fn retrieve(&self, _input: &UserMessage) -> PlannerResult<Vec<MemoryEntry>> {
        Ok(Vec::new())
    }

    /// Persist a memory entry for future turns.
    ///
    /// # Parameters
    ///
    /// * `entry` - The memory entry to store.
    ///
    /// # Example
    ///
    /// ```rust
    /// use cloudllm::planner::{MemoryEntry, MemoryStore};
    ///
    /// struct NoopStore;
    ///
    /// impl MemoryStore for NoopStore {
    ///     fn write(
    ///         &self,
    ///         _entry: MemoryEntry,
    ///     ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    ///         Ok(())
    ///     }
    /// }
    /// ```
    fn write(&self, _entry: MemoryEntry) -> PlannerResult<()> {
        Ok(())
    }
}

/// Default memory implementation that returns no entries and does not persist.
pub struct NoopMemory;

impl MemoryStore for NoopMemory {}

/// Streaming hooks for planner output.
///
/// Implementations can forward tool progress or final content to a UI, websocket,
/// or logging sink.
///
/// # Example
///
/// ```rust
/// use async_trait::async_trait;
/// use cloudllm::planner::StreamSink;
/// use cloudllm::tool_protocol::ToolResult;
///
/// struct Logger;
///
/// #[async_trait]
/// impl StreamSink for Logger {
///     async fn on_tool_start(
///         &self,
///         name: &str,
///         _parameters: &serde_json::Value,
///     ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
///         println!("tool start: {name}");
///         Ok(())
///     }
///
///     async fn on_tool_end(
///         &self,
///         name: &str,
///         _result: &ToolResult,
///     ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
///         println!("tool end: {name}");
///         Ok(())
///     }
///
///     async fn on_final(
///         &self,
///         content: &str,
///     ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
///         println!("final: {content}");
///         Ok(())
///     }
/// }
/// ```
#[async_trait]
pub trait StreamSink: Send + Sync {
    /// Notify that a tool execution is about to start.
    ///
    /// # Parameters
    ///
    /// * `name` - The tool name being executed.
    /// * `parameters` - The raw JSON parameters provided by the model.
    ///
    /// # Example
    ///
    /// ```rust
    /// use async_trait::async_trait;
    /// use cloudllm::planner::StreamSink;
    ///
    /// struct Logger;
    ///
    /// #[async_trait]
    /// impl StreamSink for Logger {
    ///     async fn on_tool_start(
    ///         &self,
    ///         name: &str,
    ///         _parameters: &serde_json::Value,
    ///     ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    ///         println!("tool start: {name}");
    ///         Ok(())
    ///     }
    /// }
    /// ```
    async fn on_tool_start(&self, _name: &str, _parameters: &Value) -> PlannerResult<()> {
        Ok(())
    }

    /// Notify that a tool execution has completed.
    ///
    /// # Parameters
    ///
    /// * `name` - The tool name that finished execution.
    /// * `result` - The tool result object.
    ///
    /// # Example
    ///
    /// ```rust
    /// use async_trait::async_trait;
    /// use cloudllm::planner::StreamSink;
    /// use cloudllm::tool_protocol::ToolResult;
    ///
    /// struct Logger;
    ///
    /// #[async_trait]
    /// impl StreamSink for Logger {
    ///     async fn on_tool_end(
    ///         &self,
    ///         name: &str,
    ///         _result: &ToolResult,
    ///     ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    ///         println!("tool end: {name}");
    ///         Ok(())
    ///     }
    /// }
    /// ```
    async fn on_tool_end(&self, _name: &str, _result: &ToolResult) -> PlannerResult<()> {
        Ok(())
    }

    /// Notify the final response content for this turn.
    ///
    /// # Parameters
    ///
    /// * `content` - The final assistant response content.
    ///
    /// # Example
    ///
    /// ```rust
    /// use async_trait::async_trait;
    /// use cloudllm::planner::StreamSink;
    ///
    /// struct Logger;
    ///
    /// #[async_trait]
    /// impl StreamSink for Logger {
    ///     async fn on_final(
    ///         &self,
    ///         content: &str,
    ///     ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    ///         println!("final: {content}");
    ///         Ok(())
    /// }
    /// ```
    async fn on_final(&self, _content: &str) -> PlannerResult<()> {
        Ok(())
    }
}

/// Default stream sink that emits nothing.
pub struct NoopStream;

#[async_trait]
impl StreamSink for NoopStream {
    /// No-op tool start hook.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use cloudllm::planner::NoopStream;
    ///
    /// # async fn example() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    /// let sink = NoopStream;
    /// sink.on_tool_start("tool", &serde_json::json!({})).await?;
    /// # Ok(())
    /// # }
    /// ```
    async fn on_tool_start(&self, _name: &str, _parameters: &Value) -> PlannerResult<()> {
        Ok(())
    }

    /// No-op tool end hook.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use cloudllm::planner::NoopStream;
    /// use cloudllm::tool_protocol::ToolResult;
    ///
    /// # async fn example() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    /// let sink = NoopStream;
    /// sink
    ///     .on_tool_end("tool", &ToolResult::success(serde_json::json!({})))
    ///     .await?;
    /// # Ok(())
    /// # }
    /// ```
    async fn on_tool_end(&self, _name: &str, _result: &ToolResult) -> PlannerResult<()> {
        Ok(())
    }

    /// No-op final output hook.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use cloudllm::planner::NoopStream;
    ///
    /// # async fn example() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    /// let sink = NoopStream;
    /// sink.on_final("done").await?;
    /// # Ok(())
    /// # }
    /// ```
    async fn on_final(&self, _content: &str) -> PlannerResult<()> {
        Ok(())
    }
}

/// Planner inputs bundled for a single turn.
///
/// This is typically constructed at call time and passed to a planner implementation.
///
/// # Example
///
/// ```rust,no_run
/// use cloudllm::planner::{NoopMemory, NoopPolicy, NoopStream, PlannerContext};
/// use cloudllm::tool_protocol::ToolRegistry;
/// use cloudllm::LLMSession;
///
/// fn example<'a>(session: &'a mut LLMSession, tools: &'a ToolRegistry) -> PlannerContext<'a> {
///     PlannerContext {
///         session,
///         tools,
///         policy: &NoopPolicy,
///         memory: &NoopMemory,
///         streamer: &NoopStream,
///         grok_tools: None,
///         openai_tools: None,
///         event_handler: None,
///     }
/// }
/// ```
pub struct PlannerContext<'a> {
    /// Mutable session used to send messages and maintain history.
    pub session: &'a mut LLMSession,
    /// Registry of tools available to the planner.
    pub tools: &'a ToolRegistry,
    /// Policy implementation that authorizes tool calls.
    pub policy: &'a dyn PolicyEngine,
    /// Memory store used for retrieval and persistence.
    pub memory: &'a dyn MemoryStore,
    /// Streaming sink for tool and final output updates.
    pub streamer: &'a dyn StreamSink,
    /// Optional Grok tools to pass through to the provider client.
    pub grok_tools: Option<Vec<GrokTool>>,
    /// Optional OpenAI tools to pass through to the provider client.
    pub openai_tools: Option<Vec<OpenAITool>>,
    /// Optional event handler for planner lifecycle events.
    pub event_handler: Option<&'a dyn EventHandler>,
}

/// Result structure returned after a planner turn.
///
/// Includes the final text, tool call results, any memory writes, and
/// optional token usage.
///
/// # Example
///
/// ```rust
/// use cloudllm::planner::PlannerOutcome;
///
/// let outcome = PlannerOutcome {
///     final_message: "ok".to_string(),
///     tool_calls: Vec::new(),
///     memory_writes: Vec::new(),
///     tokens_used: None,
/// };
/// assert_eq!(outcome.final_message, "ok");
/// ```
#[derive(Debug, Clone)]
pub struct PlannerOutcome {
    /// Final response content after tool loops complete.
    pub final_message: String,
    /// Tool results captured during the loop.
    pub tool_calls: Vec<ToolResult>,
    /// Memory entries written during the turn.
    pub memory_writes: Vec<MemoryEntry>,
    /// Aggregated token usage for the last provider call.
    pub tokens_used: Option<TokenUsage>,
}

/// Planner trait that executes one full agent turn.
///
/// Implementations control how input, memory, tools, and streaming are combined.
///
/// # Example
///
/// ```rust,no_run
/// use async_trait::async_trait;
/// use cloudllm::planner::{Planner, PlannerContext, PlannerOutcome, UserMessage};
///
/// struct EchoPlanner;
///
/// #[async_trait]
/// impl Planner for EchoPlanner {
///     async fn plan(
///         &self,
///         input: UserMessage,
///         _ctx: PlannerContext<'_>,
///     ) -> Result<PlannerOutcome, Box<dyn std::error::Error + Send + Sync>> {
///         Ok(PlannerOutcome {
///             final_message: input.content,
///             tool_calls: Vec::new(),
///             memory_writes: Vec::new(),
///             tokens_used: None,
///         })
///     }
/// }
/// ```
#[async_trait(?Send)]
pub trait Planner: Send + Sync {
    /// Execute a single planner turn.
    ///
    /// # Parameters
    ///
    /// * `input` - The user message for this turn.
    /// * `ctx` - Planner context containing session, tools, policy, and memory.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use async_trait::async_trait;
    /// use cloudllm::planner::{Planner, PlannerContext, PlannerOutcome, UserMessage};
    ///
    /// struct EchoPlanner;
    ///
    /// #[async_trait]
    /// impl Planner for EchoPlanner {
    ///     async fn plan(
    ///         &self,
    ///         input: UserMessage,
    ///         _ctx: PlannerContext<'_>,
    ///     ) -> Result<PlannerOutcome, Box<dyn std::error::Error + Send + Sync>> {
    ///         Ok(PlannerOutcome {
    ///             final_message: input.content,
    ///             tool_calls: Vec::new(),
    ///             memory_writes: Vec::new(),
    ///             tokens_used: None,
    ///         })
    ///     }
    /// }
    /// ```
    async fn plan(
        &self,
        input: UserMessage,
        ctx: PlannerContext<'_>,
    ) -> PlannerResult<PlannerOutcome>;
}

/// Default planner implementation.
///
/// The `BasicPlanner` mirrors the current agent tool loop: it appends a tool
/// prompt, executes tool calls, and feeds results back into the session.
///
/// # Example
///
/// ```rust
/// use cloudllm::planner::BasicPlanner;
///
/// let planner = BasicPlanner::new().with_max_tool_iterations(3);
/// ```
pub struct BasicPlanner {
    /// Maximum number of tool iterations before aborting.
    max_tool_iterations: usize,
}

impl Default for BasicPlanner {
    /// Build a `BasicPlanner` with default settings.
    ///
    /// # Example
    ///
    /// ```rust
    /// use cloudllm::planner::BasicPlanner;
    ///
    /// let planner = BasicPlanner::default();
    /// ```
    fn default() -> Self {
        Self {
            max_tool_iterations: 5,
        }
    }
}

impl BasicPlanner {
    /// Create a new planner with default settings.
    ///
    /// # Example
    ///
    /// ```rust
    /// use cloudllm::planner::BasicPlanner;
    ///
    /// let planner = BasicPlanner::new();
    /// ```
    pub fn new() -> Self {
        Self::default()
    }

    /// Override the maximum number of tool iterations in the loop.
    ///
    /// # Example
    ///
    /// ```rust
    /// use cloudllm::planner::BasicPlanner;
    ///
    /// let planner = BasicPlanner::new().with_max_tool_iterations(2);
    /// ```
    pub fn with_max_tool_iterations(mut self, max_tool_iterations: usize) -> Self {
        self.max_tool_iterations = max_tool_iterations;
        self
    }
}

#[async_trait(?Send)]
impl Planner for BasicPlanner {
    /// Execute a planner turn using the default tool loop semantics.
    ///
    /// This implementation mirrors the built-in `Agent` tool loop: it appends tool
    /// descriptions to the user message, checks for tool calls, executes tools, and
    /// feeds tool results back into the session until a final response is produced
    /// or the iteration cap is reached.
    async fn plan(
        &self,
        input: UserMessage,
        ctx: PlannerContext<'_>,
    ) -> PlannerResult<PlannerOutcome> {
        let mut tool_calls = Vec::new();
        let memory_writes = Vec::new();
        let plan_id = Uuid::new_v4().to_string();
        let event_handler = ctx.event_handler;

        if let Some(handler) = event_handler {
            let start = PlannerEvent::TurnStarted {
                plan_id: plan_id.clone(),
                message_preview: preview_message(&input.content),
            };
            handler.on_planner_event(&start).await;
        }

        let memory_entries = ctx.memory.retrieve(&input)?;

        let mut message = build_memory_prompt(&input.content, &memory_entries);
        message = append_tool_prompt(message, ctx.tools);

        if let Some(handler) = event_handler {
            let started = PlannerEvent::LLMCallStarted {
                plan_id: plan_id.clone(),
                iteration: 1,
            };
            handler.on_planner_event(&started).await;
        }

        let mut response = match ctx
            .session
            .send_message(
                Role::User,
                message,
                ctx.grok_tools.clone(),
                ctx.openai_tools.clone(),
            )
            .await
        {
            Ok(resp) => resp,
            Err(err) => {
                if let Some(handler) = event_handler {
                    let event = PlannerEvent::TurnErrored {
                        plan_id: plan_id.clone(),
                        error: err.to_string(),
                    };
                    handler.on_planner_event(&event).await;
                }
                return Err(map_session_error(err));
            }
        };

        if let Some(handler) = event_handler {
            let completed = PlannerEvent::LLMCallCompleted {
                plan_id: plan_id.clone(),
                iteration: 1,
                response_length: response.content.len(),
            };
            handler.on_planner_event(&completed).await;

            let chunk = PlannerEvent::PartialOutputChunk {
                plan_id: plan_id.clone(),
                iteration: 1,
                chunk: response.content.to_string(),
            };
            handler.on_planner_event(&chunk).await;
        }

        let mut current_response = response.content.to_string();
        let mut tool_iteration = 0;

        loop {
            let tool_call = parse_tool_call(&current_response);
            let Some(tool_call) = tool_call else {
                break;
            };

            let iteration_index = tool_iteration + 1;

            if let Some(handler) = event_handler {
                let event = PlannerEvent::ToolCallDetected {
                    plan_id: plan_id.clone(),
                    tool_name: tool_call.name.clone(),
                    parameters: tool_call.parameters.clone(),
                    iteration: iteration_index,
                };
                handler.on_planner_event(&event).await;
            }

            if tool_iteration >= self.max_tool_iterations {
                if let Some(handler) = event_handler {
                    let event = PlannerEvent::ToolMaxIterationsReached {
                        plan_id: plan_id.clone(),
                    };
                    handler.on_planner_event(&event).await;
                }
                current_response = format!(
                    "{}\n\n[Warning: Maximum tool iterations reached]",
                    current_response
                );
                break;
            }

            tool_iteration += 1;
            let policy_decision = ctx.policy.allow_tool_call(&tool_call).await?;
            if let PolicyDecision::Deny(reason) = policy_decision {
                if let Some(handler) = event_handler {
                    let event = PlannerEvent::ToolExecutionCompleted {
                        plan_id: plan_id.clone(),
                        tool_name: tool_call.name.clone(),
                        parameters: tool_call.parameters.clone(),
                        success: false,
                        error: Some(format!("policy denied: {}", reason)),
                        result: None,
                        iteration: iteration_index,
                    };
                    handler.on_planner_event(&event).await;
                }
                current_response = format!("Tool call denied: {}", reason);
                break;
            }

            let tool_params_snapshot = tool_call.parameters.clone();

            ctx.streamer
                .on_tool_start(&tool_call.name, &tool_params_snapshot)
                .await?;

            let tool_result = ctx
                .tools
                .execute_tool(&tool_call.name, tool_params_snapshot.clone())
                .await;

            let (tool_result_message, resolved_tool_result) = match &tool_result {
                Ok(result) => {
                    let message = if result.success {
                        format!(
                            "Tool '{}' executed successfully. Result: {}",
                            tool_call.name,
                            serde_json::to_string_pretty(&result.output)
                                .unwrap_or_else(|_| format!("{:?}", result.output))
                        )
                    } else {
                        let err = result
                            .error
                            .clone()
                            .unwrap_or_else(|| "Unknown error".to_string());
                        format!("Tool '{}' failed. Error: {}", tool_call.name, err)
                    };
                    (message, result.clone())
                }
                Err(err) => (
                    format!("Tool execution error: {}", err),
                    ToolResult::failure(err.to_string()),
                ),
            };

            if let Some(handler) = event_handler {
                let event = PlannerEvent::ToolExecutionCompleted {
                    plan_id: plan_id.clone(),
                    tool_name: tool_call.name.clone(),
                    parameters: tool_params_snapshot.clone(),
                    success: resolved_tool_result.success,
                    error: resolved_tool_result.error.clone(),
                    result: if resolved_tool_result.success {
                        Some(resolved_tool_result.output.clone())
                    } else {
                        None
                    },
                    iteration: iteration_index,
                };
                handler.on_planner_event(&event).await;
            }

            ctx.streamer
                .on_tool_end(&tool_call.name, &resolved_tool_result)
                .await?;

            tool_calls.push(resolved_tool_result.clone());

            if let Some(handler) = event_handler {
                let started = PlannerEvent::LLMCallStarted {
                    plan_id: plan_id.clone(),
                    iteration: tool_iteration + 1,
                };
                handler.on_planner_event(&started).await;
            }

            response = match ctx
                .session
                .send_message(
                    Role::User,
                    tool_result_message,
                    ctx.grok_tools.clone(),
                    ctx.openai_tools.clone(),
                )
                .await
            {
                Ok(resp) => resp,
                Err(err) => {
                    if let Some(handler) = event_handler {
                        let event = PlannerEvent::TurnErrored {
                            plan_id: plan_id.clone(),
                            error: err.to_string(),
                        };
                        handler.on_planner_event(&event).await;
                    }
                    return Err(map_session_error(err));
                }
            };

            if let Some(handler) = event_handler {
                let completed = PlannerEvent::LLMCallCompleted {
                    plan_id: plan_id.clone(),
                    iteration: tool_iteration + 1,
                    response_length: response.content.len(),
                };
                handler.on_planner_event(&completed).await;

                let chunk = PlannerEvent::PartialOutputChunk {
                    plan_id: plan_id.clone(),
                    iteration: tool_iteration + 1,
                    chunk: response.content.to_string(),
                };
                handler.on_planner_event(&chunk).await;
            }

            current_response = response.content.to_string();
        }

        if let Err(err) = ctx.streamer.on_final(&current_response).await {
            if let Some(handler) = event_handler {
                let event = PlannerEvent::TurnErrored {
                    plan_id: plan_id.clone(),
                    error: err.to_string(),
                };
                handler.on_planner_event(&event).await;
            }
            return Err(err);
        }

        let tokens_used = ctx.session.last_token_usage().await;

        if let Some(handler) = event_handler {
            let event = PlannerEvent::TurnCompleted {
                plan_id: plan_id.clone(),
                tokens_used: tokens_used.clone(),
                response_length: current_response.len(),
                tool_calls_made: tool_calls.len(),
            };
            handler.on_planner_event(&event).await;
        }

        Ok(PlannerOutcome {
            final_message: current_response,
            tool_calls,
            memory_writes,
            tokens_used,
        })
    }
}

/// Produce a truncated preview of the user message for event payloads.
///
/// Returns the first N characters of the message (respecting UTF-8 boundaries).
/// If the message is longer than the limit, an ellipsis is appended. Removes
/// newlines for cleaner logging.
///
/// # Parameters
///
/// * `text` - The full message text.
///
/// # Returns
///
/// A truncated, newline-free preview suitable for logging (≤ 120 chars + "...").
fn preview_message(text: &str) -> String {
    const MAX_PREVIEW_CHARS: usize = 120;

    // Normalize: remove newlines for cleaner logging
    let normalized = text.replace('\n', " ").replace('\r', "");

    // Truncate at character boundary
    let mut chars = normalized.chars();
    let preview: String = chars.by_ref().take(MAX_PREVIEW_CHARS).collect();

    // Append ellipsis if truncated
    if chars.next().is_some() {
        format!("{preview}...")
    } else {
        preview
    }
}

/// Build a prompt that prepends retrieved memory to the user message.
///
/// Formats memory entries with clear section headers and maintains separation
/// from the main user message. If no entries are provided, returns the message unchanged.
///
/// # Parameters
///
/// * `user_message` - The raw user input text.
/// * `entries` - Memory entries to prepend (typically from semantic retrieval).
///
/// # Returns
///
/// A formatted prompt with memory context prepended, or just the user message if empty.
///
/// # Example
///
/// ```ignore
/// use cloudllm::planner::MemoryEntry;
///
/// let entries = vec![
///     MemoryEntry::new("Remember this").with_metadata("source", "session"),
///     MemoryEntry::new("Also recall that").with_metadata("source", "manual"),
/// ];
/// let prompt = cloudllm::planner::build_memory_prompt("What should I do?", &entries);
/// assert!(prompt.contains("Relevant memory:"));
/// assert!(prompt.contains("Remember this"));
/// assert!(prompt.contains("What should I do?"));
/// ```
fn build_memory_prompt(user_message: &str, entries: &[MemoryEntry]) -> String {
    if entries.is_empty() {
        return user_message.to_string();
    }

    let mut prompt = String::new();
    prompt.push_str("═══════════════════════════════════════════════════════════\n");
    prompt.push_str("RELEVANT MEMORY (context from prior work):\n");
    prompt.push_str("═══════════════════════════════════════════════════════════\n\n");

    for (idx, entry) in entries.iter().enumerate() {
        prompt.push_str(&format!("[Memory {}] {}\n", idx + 1, entry.content));
        if !entry.metadata.is_empty() {
            for (key, value) in &entry.metadata {
                prompt.push_str(&format!("  ({}: {})\n", key, value));
            }
        }
        prompt.push('\n');
    }

    prompt.push_str("═══════════════════════════════════════════════════════════\n");
    prompt.push_str("NEW TASK:\n");
    prompt.push_str("═══════════════════════════════════════════════════════════\n\n");
    prompt.push_str(user_message);

    prompt
}

/// Append tool descriptions and usage instructions to a user message.
///
/// Formats available tools with clear section headers, parameter details, and
/// provides explicit JSON format examples. If no tools are available, returns
/// the message unchanged.
///
/// # Parameters
///
/// * `message` - The prompt to augment with tool information.
/// * `tools` - Tool registry containing all available tools and their metadata.
///
/// # Returns
///
/// The augmented message with tool list and format instructions, or unchanged
/// message if no tools are registered.
///
/// # Example
///
/// ```ignore
/// use std::sync::Arc;
/// use cloudllm::tool_protocol::{ToolMetadata, ToolRegistry};
/// use cloudllm::tool_protocols::CustomToolProtocol;
///
/// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
/// let protocol = Arc::new(CustomToolProtocol::new());
/// protocol
///     .register_tool(
///         ToolMetadata::new("calculator", "Do math operations")
///             .with_parameter(cloudllm::tool_protocol::ToolParameter::new(
///                 "expr",
///                 cloudllm::tool_protocol::ToolParameterType::String
///             ).with_description("Math expression").required()),
///         Arc::new(|params| Ok(cloudllm::tool_protocol::ToolResult::success(serde_json::json!({"result": 5})))),
///     )
///     .await;
///
/// let mut registry = ToolRegistry::new(protocol);
/// registry.discover_tools_from_primary().await?;
/// let prompt = cloudllm::planner::append_tool_prompt("Calculate 2+3".to_string(), &registry);
/// assert!(prompt.contains("calculator"));
/// assert!(prompt.contains("tool_call"));
/// # Ok(())
/// # }
/// ```
fn append_tool_prompt(mut message: String, tools: &ToolRegistry) -> String {
    let tool_list = tools.list_tools();
    if tool_list.is_empty() {
        return message;
    }

    message.push_str("\n\n");
    message.push_str("═══════════════════════════════════════════════════════════\n");
    message.push_str(&format!("AVAILABLE TOOLS ({} total):\n", tool_list.len()));
    message.push_str("═══════════════════════════════════════════════════════════\n\n");

    for (idx, tool_metadata) in tool_list.iter().enumerate() {
        message.push_str(&format!(
            "[{}] {}\n    {}\n",
            idx + 1,
            tool_metadata.name,
            tool_metadata.description
        ));

        if !tool_metadata.parameters.is_empty() {
            message.push_str("    Parameters:\n");
            for param in &tool_metadata.parameters {
                let required = if param.required { " [REQUIRED]" } else { "" };
                message.push_str(&format!(
                    "      • {} ({}){}\n        {}\n",
                    param.name,
                    format!("{:?}", param.param_type).to_lowercase(),
                    required,
                    param.description.as_deref().unwrap_or("(no description)")
                ));
            }
        }
        message.push('\n');
    }

    message.push_str("═══════════════════════════════════════════════════════════\n");
    message.push_str("TOOL USAGE FORMAT:\n");
    message.push_str("═══════════════════════════════════════════════════════════\n\n");
    message.push_str("To invoke a tool, respond with EXACTLY this JSON structure:\n\n");
    message.push_str("  {\"tool_call\": {\"name\": \"tool_name\", \"parameters\": {\"param1\": \"value1\", \"param2\": \"value2\"}}}\n\n");
    message.push_str("Examples:\n");
    message.push_str("  {\"tool_call\": {\"name\": \"calculator\", \"parameters\": {\"expr\": \"2+2\"}}}\n");
    message.push_str("  {\"tool_call\": {\"name\": \"read_file\", \"parameters\": {\"path\": \"/home/user/data.txt\"}}}\n\n");
    message.push_str("After I execute the tool, I'll provide the result and you can continue working.\n");
    message.push_str("You may call multiple tools sequentially in a single response.\n");

    message
}

/// Parse the first JSON tool call from a response string.
///
/// Scans the response for the JSON tool call marker `{"tool_call":` and extracts
/// the first valid tool call found. Supports tool calls anywhere in the response
/// (before, after, or mixed with other text).
///
/// Returns `None` if:
/// - No `{"tool_call":` marker is found
/// - No closing `}` is found after the marker
/// - The extracted JSON is malformed
/// - Required fields (`name` or `parameters`) are missing
///
/// # Parameters
///
/// * `response` - The complete model response text to scan.
///
/// # Returns
///
/// The first valid `ToolCallRequest` found, or `None` if parsing fails.
///
/// # Example
///
/// ```ignore
/// use cloudllm::planner::ToolCallRequest;
///
/// // Tool call in the middle of text
/// let response = "I will now calculate this.\n{\"tool_call\": {\"name\": \"calculator\", \"parameters\": {\"expr\": \"2+2\"}}}\nThe result will help.";
/// let parsed = cloudllm::planner::parse_tool_call(response).unwrap();
/// assert_eq!(parsed.name, "calculator");
///
/// // No tool call (returns None)
/// let no_call = "Just talking about tools";
/// assert!(cloudllm::planner::parse_tool_call(no_call).is_none());
/// ```
fn parse_tool_call(response: &str) -> Option<ToolCallRequest> {
    // Find the tool call marker
    let marker = "{\"tool_call\"";
    let start_idx = response.find(marker)?;

    // Find the closing brace (naive approach — assumes balanced JSON)
    let end_idx = response[start_idx..].rfind('}')?;
    let abs_end_idx = start_idx + end_idx;

    if abs_end_idx <= start_idx {
        return None;
    }

    // Extract and parse the JSON
    let tool_json = &response[start_idx..=abs_end_idx];

    // Try to parse as JSON
    let parsed: Value = match serde_json::from_str(tool_json) {
        Ok(v) => v,
        Err(_) => return None,
    };

    // Extract tool_call object
    let tool_call_obj = parsed.get("tool_call")?;

    // Extract name (required)
    let name = tool_call_obj
        .get("name")?
        .as_str()?
        .trim()
        .to_string();

    if name.is_empty() {
        return None;
    }

    // Extract parameters (required, may be empty object)
    let parameters = tool_call_obj.get("parameters")?.clone();

    Some(ToolCallRequest { name, parameters })
}

/// Convert session errors into Send + Sync errors for planner callers.
///
/// Wraps any error type (which may not be Send + Sync) into a standard io::Error
/// that implements the required traits. This is necessary because planners are
/// used in async contexts where error types must be Send + Sync.
///
/// # Parameters
///
/// * `err` - The original session error (any Box<dyn Error> type).
///
/// # Returns
///
/// A Send + Sync error suitable for propagation through async code.
fn map_session_error(err: Box<dyn Error>) -> Box<dyn Error + Send + Sync> {
    Box::new(io::Error::new(io::ErrorKind::Other, err.to_string()))
}
