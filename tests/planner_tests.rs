use async_trait::async_trait;
use cloudllm::client_wrapper::{ClientWrapper, Message, Role, TokenUsage, ToolDefinition};
use cloudllm::event::{EventHandler, PlannerEvent};
use cloudllm::planner::{
    BasicPlanner, MemoryEntry, MemoryStore, NoopMemory, NoopPolicy, NoopStream, Planner,
    PlannerContext, PolicyDecision, PolicyEngine, StreamSink, UserMessage,
};
use cloudllm::tool_protocol::{
    ToolMetadata, ToolParameter, ToolParameterType, ToolRegistry, ToolResult,
};
use cloudllm::tool_protocols::CustomToolProtocol;
use cloudllm::LLMSession;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use tokio::sync::Mutex;

struct SequentialMockClient {
    responses: Vec<String>,
    call_count: AtomicUsize,
}

impl SequentialMockClient {
    fn new(responses: Vec<String>) -> Self {
        Self {
            responses,
            call_count: AtomicUsize::new(0),
        }
    }
}

#[async_trait]
impl ClientWrapper for SequentialMockClient {
    async fn send_message(
        &self,
        _messages: &[Message],
        _tools: Option<Vec<ToolDefinition>>,
    ) -> Result<Message, Box<dyn std::error::Error>> {
        let index = self.call_count.fetch_add(1, Ordering::SeqCst);
        let response = self
            .responses
            .get(index)
            .or_else(|| self.responses.last())
            .ok_or("missing mock response")?;
        Ok(Message {
            role: Role::Assistant,
            content: Arc::from(response.as_str()),
            tool_calls: vec![],
        })
    }

    fn model_name(&self) -> &str {
        "mock-sequential"
    }

    async fn get_last_usage(&self) -> Option<TokenUsage> {
        None
    }
}

struct DenyPolicy;

#[async_trait]
impl PolicyEngine for DenyPolicy {
    async fn allow_tool_call(
        &self,
        _call: &cloudllm::planner::ToolCallRequest,
    ) -> Result<PolicyDecision, Box<dyn std::error::Error + Send + Sync>> {
        Ok(PolicyDecision::Deny("blocked".to_string()))
    }
}

struct RecordingMemory {
    called: Arc<AtomicBool>,
}

impl MemoryStore for RecordingMemory {
    fn retrieve(
        &self,
        _input: &UserMessage,
    ) -> Result<Vec<MemoryEntry>, Box<dyn std::error::Error + Send + Sync>> {
        self.called.store(true, Ordering::SeqCst);
        Ok(vec![MemoryEntry::new("Remember this")])
    }
}

struct InspectingClient {
    saw_memory: Arc<AtomicBool>,
}

#[async_trait]
impl ClientWrapper for InspectingClient {
    async fn send_message(
        &self,
        messages: &[Message],
        _tools: Option<Vec<ToolDefinition>>,
    ) -> Result<Message, Box<dyn std::error::Error>> {
        let found = messages.iter().any(|message| {
            message.content.contains("Relevant memory") && message.content.contains("Remember this")
        });
        self.saw_memory.store(found, Ordering::SeqCst);
        if !found {
            return Err("memory not injected".into());
        }
        Ok(Message {
            role: Role::Assistant,
            content: Arc::from("OK"),
            tool_calls: vec![],
        })
    }

    fn model_name(&self) -> &str {
        "mock-inspecting"
    }
}

struct CountingStream {
    tool_start: AtomicUsize,
    tool_end: AtomicUsize,
    finals: AtomicUsize,
}

#[async_trait]
impl StreamSink for CountingStream {
    async fn on_tool_start(
        &self,
        _name: &str,
        _parameters: &serde_json::Value,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        self.tool_start.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }

    async fn on_tool_end(
        &self,
        _name: &str,
        _result: &ToolResult,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        self.tool_end.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }

    async fn on_final(
        &self,
        _content: &str,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        self.finals.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }
}

struct RecordingHandler {
    events: Arc<Mutex<Vec<PlannerEvent>>>,
}

impl RecordingHandler {
    fn new() -> Self {
        Self {
            events: Arc::new(Mutex::new(Vec::new())),
        }
    }
}

#[async_trait]
impl EventHandler for RecordingHandler {
    async fn on_planner_event(&self, event: &PlannerEvent) {
        let mut lock = self.events.lock().await;
        lock.push(event.clone());
    }
}

async fn build_registry(exec_count: Arc<AtomicUsize>) -> ToolRegistry {
    let protocol = Arc::new(CustomToolProtocol::new());
    protocol
        .register_tool(
            ToolMetadata::new("add", "Add two numbers")
                .with_parameter(
                    ToolParameter::new("a", ToolParameterType::Number)
                        .with_description("First number")
                        .required(),
                )
                .with_parameter(
                    ToolParameter::new("b", ToolParameterType::Number)
                        .with_description("Second number")
                        .required(),
                ),
            Arc::new(move |params| {
                exec_count.fetch_add(1, Ordering::SeqCst);
                let a = params["a"].as_f64().unwrap_or(0.0);
                let b = params["b"].as_f64().unwrap_or(0.0);
                Ok(ToolResult::success(serde_json::json!({ "result": a + b })))
            }),
        )
        .await;

    let mut registry = ToolRegistry::new(protocol);
    registry.discover_tools_from_primary().await.unwrap();
    registry
}

#[tokio::test]
async fn planner_executes_tool_loop_and_streams_events() {
    let exec_count = Arc::new(AtomicUsize::new(0));
    let registry = build_registry(exec_count.clone()).await;
    let responses = vec![
        r#"{"tool_call": {"name": "add", "parameters": {"a": 2, "b": 3}}}"#.to_string(),
        "Done".to_string(),
    ];

    let client: Arc<dyn ClientWrapper> = Arc::new(SequentialMockClient::new(responses));
    let mut session = LLMSession::new(client, "You are precise.".into(), 8_000);
    let stream = CountingStream {
        tool_start: AtomicUsize::new(0),
        tool_end: AtomicUsize::new(0),
        finals: AtomicUsize::new(0),
    };

    let planner = BasicPlanner::new();
    let outcome = planner
        .plan(
            UserMessage::from("Add 2+3"),
            PlannerContext {
                session: &mut session,
                tools: &registry,
                policy: &NoopPolicy,
                memory: &NoopMemory,
                streamer: &stream,
                grok_tools: None,
                openai_tools: None,
                event_handler: None,
            },
        )
        .await
        .unwrap();

    assert_eq!(outcome.final_message, "Done");
    assert_eq!(outcome.tool_calls.len(), 1);
    assert!(outcome.tool_calls[0].success);
    assert_eq!(exec_count.load(Ordering::SeqCst), 1);
    assert_eq!(stream.tool_start.load(Ordering::SeqCst), 1);
    assert_eq!(stream.tool_end.load(Ordering::SeqCst), 1);
    assert_eq!(stream.finals.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn planner_emits_planner_events() {
    let exec_count = Arc::new(AtomicUsize::new(0));
    let registry = build_registry(exec_count.clone()).await;
    let responses = vec![
        r#"{"tool_call": {"name": "add", "parameters": {"a": 2, "b": 3}}}"#.to_string(),
        "Done".to_string(),
    ];

    let client: Arc<dyn ClientWrapper> = Arc::new(SequentialMockClient::new(responses));
    let mut session = LLMSession::new(client, "You are precise.".into(), 8_000);
    let handler = Arc::new(RecordingHandler::new());
    let events_handle = handler.events.clone();

    let planner = BasicPlanner::new();
    let outcome = planner
        .plan(
            UserMessage::from("Add 2+3"),
            PlannerContext {
                session: &mut session,
                tools: &registry,
                policy: &NoopPolicy,
                memory: &NoopMemory,
                streamer: &NoopStream,
                grok_tools: None,
                openai_tools: None,
                event_handler: Some(handler.as_ref()),
            },
        )
        .await
        .unwrap();

    assert_eq!(outcome.final_message, "Done");

    let events = events_handle.lock().await.clone();
    assert!(!events.is_empty());
    assert!(matches!(
        events.first(),
        Some(PlannerEvent::TurnStarted { .. })
    ));
    assert!(events
        .iter()
        .any(|event| matches!(event, PlannerEvent::ToolCallDetected { .. })));
    assert!(events.iter().any(|event| matches!(
        event,
        PlannerEvent::ToolExecutionCompleted { success: true, .. }
    )));
    assert!(matches!(
        events.last(),
        Some(PlannerEvent::TurnCompleted { .. })
    ));
}

#[tokio::test]
async fn planner_denies_tool_call_via_policy() {
    let exec_count = Arc::new(AtomicUsize::new(0));
    let registry = build_registry(exec_count.clone()).await;
    let responses =
        vec![r#"{"tool_call": {"name": "add", "parameters": {"a": 1, "b": 1}}}"#.to_string()];

    let client: Arc<dyn ClientWrapper> = Arc::new(SequentialMockClient::new(responses));
    let mut session = LLMSession::new(client, "You are precise.".into(), 8_000);
    let planner = BasicPlanner::new();
    let outcome = planner
        .plan(
            UserMessage::from("Add 1+1"),
            PlannerContext {
                session: &mut session,
                tools: &registry,
                policy: &DenyPolicy,
                memory: &NoopMemory,
                streamer: &NoopStream,
                grok_tools: None,
                openai_tools: None,
                event_handler: None,
            },
        )
        .await
        .unwrap();

    assert!(outcome.final_message.contains("Tool call denied"));
    assert!(outcome.tool_calls.is_empty());
    assert_eq!(exec_count.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn planner_includes_memory_in_prompt() {
    let saw_memory = Arc::new(AtomicBool::new(false));
    let client: Arc<dyn ClientWrapper> = Arc::new(InspectingClient {
        saw_memory: saw_memory.clone(),
    });
    let mut session = LLMSession::new(client, "You are precise.".into(), 8_000);
    let registry = ToolRegistry::empty();
    let memory_called = Arc::new(AtomicBool::new(false));
    let memory = RecordingMemory {
        called: memory_called.clone(),
    };

    let planner = BasicPlanner::new();
    let outcome = planner
        .plan(
            UserMessage::from("Check memory"),
            PlannerContext {
                session: &mut session,
                tools: &registry,
                policy: &NoopPolicy,
                memory: &memory,
                streamer: &NoopStream,
                grok_tools: None,
                openai_tools: None,
                event_handler: None,
            },
        )
        .await
        .unwrap();

    assert_eq!(outcome.final_message, "OK");
    assert!(saw_memory.load(Ordering::SeqCst));
    assert!(memory_called.load(Ordering::SeqCst));
}
