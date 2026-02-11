use async_trait::async_trait;
use cloudllm::client_wrapper::{ClientWrapper, Message, Role, TokenUsage};
use cloudllm::orchestration::{Orchestration, OrchestrationMode, RalphTask};
use cloudllm::Agent;
use openai_rust2 as openai_rust;
use std::sync::Arc;

struct MockClient {
    name: String,
    response: String,
}

#[async_trait]
impl ClientWrapper for MockClient {
    async fn send_message(
        &self,
        _messages: &[Message],
        _optional_grok_tools: Option<Vec<openai_rust::chat::GrokTool>>,
        _optional_openai_tools: Option<Vec<openai_rust::chat::OpenAITool>>,
    ) -> Result<Message, Box<dyn std::error::Error>> {
        Ok(Message {
            role: Role::Assistant,
            content: Arc::from(self.response.as_str()),
        })
    }

    fn model_name(&self) -> &str {
        &self.name
    }

    async fn get_last_usage(&self) -> Option<TokenUsage> {
        None
    }
}

#[tokio::test]
async fn test_agent_creation() {
    let client = Arc::new(MockClient {
        name: "mock".to_string(),
        response: "test response".to_string(),
    });

    let agent = Agent::new("agent1", "Test Agent", client)
        .with_expertise("Testing")
        .with_personality("Thorough and detail-oriented");

    assert_eq!(agent.id, "agent1");
    assert_eq!(agent.name, "Test Agent");
    assert_eq!(agent.expertise, Some("Testing".to_string()));
}

#[tokio::test]
async fn test_orchestration_parallel_mode() {
    let agent1 = Agent::new(
        "agent1",
        "Agent 1",
        Arc::new(MockClient {
            name: "mock1".to_string(),
            response: "Response from agent 1".to_string(),
        }),
    );

    let agent2 = Agent::new(
        "agent2",
        "Agent 2",
        Arc::new(MockClient {
            name: "mock2".to_string(),
            response: "Response from agent 2".to_string(),
        }),
    );

    let mut orchestration = Orchestration::new("test-orchestration", "Test Orchestration")
        .with_mode(OrchestrationMode::Parallel);

    orchestration.add_agent(agent1).unwrap();
    orchestration.add_agent(agent2).unwrap();

    let response = orchestration.run("Test question", 1).await.unwrap();

    assert_eq!(response.messages.len(), 2);
    assert!(response.is_complete);
}

#[tokio::test]
async fn test_orchestration_round_robin_mode() {
    let agent1 = Agent::new(
        "agent1",
        "Agent 1",
        Arc::new(MockClient {
            name: "mock1".to_string(),
            response: "First agent response".to_string(),
        }),
    );

    let agent2 = Agent::new(
        "agent2",
        "Agent 2",
        Arc::new(MockClient {
            name: "mock2".to_string(),
            response: "Second agent response".to_string(),
        }),
    );

    let mut orchestration = Orchestration::new("test-orchestration", "Test Orchestration")
        .with_mode(OrchestrationMode::RoundRobin);

    orchestration.add_agent(agent1).unwrap();
    orchestration.add_agent(agent2).unwrap();

    let response = orchestration.run("Test question", 2).await.unwrap();

    assert_eq!(response.messages.len(), 4); // 2 agents * 2 rounds
    assert!(response.is_complete);
}

#[tokio::test]
async fn test_agent_with_tool_execution() {
    use cloudllm::tool_protocol::{
        ToolMetadata, ToolParameter, ToolParameterType, ToolRegistry, ToolResult,
    };
    use cloudllm::tool_protocols::CustomToolProtocol;
    use tokio::sync::Mutex as TokioMutex;

    // Create a custom tool adapter
    let adapter = CustomToolProtocol::new();

    // Register a simple calculator tool
    adapter
        .register_tool(
            ToolMetadata::new("add", "Adds two numbers")
                .with_parameter(ToolParameter::new("a", ToolParameterType::Number).required())
                .with_parameter(ToolParameter::new("b", ToolParameterType::Number).required()),
            Arc::new(|params| {
                let a = params["a"].as_f64().unwrap_or(0.0);
                let b = params["b"].as_f64().unwrap_or(0.0);
                Ok(ToolResult::success(serde_json::json!({"sum": a + b})))
            }),
        )
        .await;

    let mut registry = ToolRegistry::new(Arc::new(adapter));
    // Discover tools from the adapter
    registry.discover_tools_from_primary().await.unwrap();

    // Create a mock client that will respond with a tool call
    struct ToolCallingMockClient {
        call_count: Arc<TokioMutex<usize>>,
    }

    #[async_trait]
    impl ClientWrapper for ToolCallingMockClient {
        async fn send_message(
            &self,
            messages: &[Message],
            _optional_grok_tools: Option<Vec<openai_rust::chat::GrokTool>>,
            _optional_openai_tools: Option<Vec<openai_rust::chat::OpenAITool>>,
        ) -> Result<Message, Box<dyn std::error::Error>> {
            let mut count = self.call_count.lock().await;
            *count += 1;

            // First call: return a tool call
            // Second call: return final response
            let response = if *count == 1 {
                // Check that system message includes tool information
                let system_msg = &messages[0];
                // The system message should contain the tool name and description
                let system_content = system_msg.content.as_ref();
                if !system_content.contains("add") || !system_content.contains("Adds two numbers") {
                    panic!(
                        "System message doesn't contain tool information. Content:\n{}",
                        system_content
                    );
                }

                // Return tool call
                r#"{"tool_call": {"name": "add", "parameters": {"a": 5, "b": 3}}}"#
            } else {
                // Verify tool result was provided
                let last_msg = messages.last().unwrap();
                let last_content = last_msg.content.as_ref();
                if !last_content.contains("Tool 'add' executed successfully") {
                    panic!(
                        "Last message doesn't contain tool result. Content:\n{}",
                        last_content
                    );
                }

                "The sum is 8"
            };

            Ok(Message {
                role: Role::Assistant,
                content: Arc::from(response),
            })
        }

        fn model_name(&self) -> &str {
            "tool-mock"
        }

        async fn get_last_usage(&self) -> Option<TokenUsage> {
            None
        }
    }

    let agent = Agent::new(
        "calculator",
        "Calculator Agent",
        Arc::new(ToolCallingMockClient {
            call_count: Arc::new(TokioMutex::new(0)),
        }),
    )
    .with_tools(registry);

    let response = agent
        .generate("You are a helpful assistant", "What is 5 + 3?", &[])
        .await
        .unwrap();

    assert_eq!(response, "The sum is 8");
}

#[tokio::test]
async fn test_debate_mode_convergence() {
    use tokio::sync::Mutex as TokioMutex;

    // Mock client that returns increasingly similar responses
    struct ConvergingMockClient {
        call_count: Arc<TokioMutex<usize>>,
        agent_id: String,
    }

    #[async_trait]
    impl ClientWrapper for ConvergingMockClient {
        async fn send_message(
            &self,
            _messages: &[Message],
            _optional_grok_tools: Option<Vec<openai_rust::chat::GrokTool>>,
            _optional_openai_tools: Option<Vec<openai_rust::chat::OpenAITool>>,
        ) -> Result<Message, Box<dyn std::error::Error>> {
            let mut count = self.call_count.lock().await;
            *count += 1;

            // Simulate agents converging on a solution over multiple rounds
            let response = match *count {
                1 => format!("Agent {}: I think we should use approach A", self.agent_id),
                2 => format!(
                    "Agent {}: Approach A seems reasonable but needs refinement",
                    self.agent_id
                ),
                3 => format!(
                    "Agent {}: After consideration approach A with refinement is best solution",
                    self.agent_id
                ),
                _ => format!(
                    "Agent {}: I agree approach A with refinement is the best solution",
                    self.agent_id
                ),
            };

            Ok(Message {
                role: Role::Assistant,
                content: Arc::from(response.as_str()),
            })
        }

        fn model_name(&self) -> &str {
            "converging-mock"
        }

        async fn get_last_usage(&self) -> Option<TokenUsage> {
            None
        }
    }

    let agent1 = Agent::new(
        "agent1",
        "Agent 1",
        Arc::new(ConvergingMockClient {
            call_count: Arc::new(TokioMutex::new(0)),
            agent_id: "1".to_string(),
        }),
    );

    let agent2 = Agent::new(
        "agent2",
        "Agent 2",
        Arc::new(ConvergingMockClient {
            call_count: Arc::new(TokioMutex::new(0)),
            agent_id: "2".to_string(),
        }),
    );

    let mut orchestration = Orchestration::new("debate-orchestration", "Debate Orchestration")
        .with_mode(OrchestrationMode::Debate {
            max_rounds: 5,
            convergence_threshold: Some(0.6), // 60% similarity threshold
        });

    orchestration.add_agent(agent1).unwrap();
    orchestration.add_agent(agent2).unwrap();

    let response = orchestration
        .run("What approach should we use?", 5)
        .await
        .unwrap();

    // Should converge before max rounds (5)
    assert!(response.round < 5);
    assert!(response.is_complete);

    // Should have a convergence score
    assert!(response.convergence_score.is_some());
    let score = response.convergence_score.unwrap();
    assert!(score >= 0.6, "Convergence score {} should be >= 0.6", score);
}

#[tokio::test]
async fn test_ralph_mode_completion() {
    use tokio::sync::Mutex as TokioMutex;

    // Mock client that completes task1 on first call, task2 on second call
    struct RalphCompletingClient {
        call_count: Arc<TokioMutex<usize>>,
    }

    #[async_trait]
    impl ClientWrapper for RalphCompletingClient {
        async fn send_message(
            &self,
            _messages: &[Message],
            _optional_grok_tools: Option<Vec<openai_rust::chat::GrokTool>>,
            _optional_openai_tools: Option<Vec<openai_rust::chat::OpenAITool>>,
        ) -> Result<Message, Box<dyn std::error::Error>> {
            let mut count = self.call_count.lock().await;
            *count += 1;

            let response = match *count {
                1 => "I've implemented the HTML structure. [TASK_COMPLETE:task1]".to_string(),
                _ => "Game loop is done. [TASK_COMPLETE:task2]".to_string(),
            };

            Ok(Message {
                role: Role::Assistant,
                content: Arc::from(response.as_str()),
            })
        }

        fn model_name(&self) -> &str {
            "ralph-mock"
        }

        async fn get_last_usage(&self) -> Option<TokenUsage> {
            None
        }
    }

    let agent = Agent::new(
        "builder",
        "Builder Agent",
        Arc::new(RalphCompletingClient {
            call_count: Arc::new(TokioMutex::new(0)),
        }),
    );

    let tasks = vec![
        RalphTask::new("task1", "HTML Structure", "Create the HTML boilerplate"),
        RalphTask::new("task2", "Game Loop", "Implement the game loop"),
    ];

    let mut orchestration =
        Orchestration::new("ralph-test", "Ralph Test").with_mode(OrchestrationMode::Ralph {
            tasks,
            max_iterations: 5,
        });

    orchestration.add_agent(agent).unwrap();

    let response = orchestration.run("Build a breakout game", 1).await.unwrap();

    assert!(response.is_complete);
    assert_eq!(response.convergence_score, Some(1.0));
    assert!(response.round <= 5);
}

#[tokio::test]
async fn test_ralph_mode_max_iterations() {
    // Mock client that never emits completion markers
    let agent = Agent::new(
        "lazy",
        "Lazy Agent",
        Arc::new(MockClient {
            name: "lazy-mock".to_string(),
            response: "I'm working on it but not done yet.".to_string(),
        }),
    );

    let tasks = vec![
        RalphTask::new("task1", "Task One", "Do something"),
        RalphTask::new("task2", "Task Two", "Do something else"),
    ];

    let max_iterations = 3;

    let mut orchestration = Orchestration::new("ralph-max-test", "Ralph Max Test").with_mode(
        OrchestrationMode::Ralph {
            tasks,
            max_iterations,
        },
    );

    orchestration.add_agent(agent).unwrap();

    let response = orchestration.run("Do the tasks", 1).await.unwrap();

    assert!(!response.is_complete);
    assert_eq!(response.round, max_iterations);
    assert_eq!(response.convergence_score.unwrap(), 0.0);
}

#[tokio::test]
async fn test_ralph_mode_empty_tasks() {
    let agent = Agent::new(
        "agent1",
        "Agent 1",
        Arc::new(MockClient {
            name: "mock".to_string(),
            response: "test".to_string(),
        }),
    );

    let mut orchestration =
        Orchestration::new("ralph-empty", "Ralph Empty").with_mode(OrchestrationMode::Ralph {
            tasks: vec![],
            max_iterations: 5,
        });

    orchestration.add_agent(agent).unwrap();

    let response = orchestration.run("Do nothing", 1).await.unwrap();

    assert!(response.is_complete);
    assert_eq!(response.convergence_score, Some(1.0));
    assert_eq!(response.round, 0);
    assert!(response.messages.is_empty());
}

#[tokio::test]
async fn test_agent_send_uses_session() {
    let client = Arc::new(MockClient {
        name: "mock".to_string(),
        response: "session response".to_string(),
    });

    let mut agent = Agent::new("agent1", "Test Agent", client);
    agent.set_system_prompt("You are a helpful assistant.");

    let result = agent.send("Hello").await.unwrap();
    assert_eq!(result.content, "session response");

    // After send, session should have history (user message + assistant response)
    assert!(agent.session_history_len() >= 2);
}

#[tokio::test]
async fn test_agent_receive_message() {
    let client = Arc::new(MockClient {
        name: "mock".to_string(),
        response: "test".to_string(),
    });

    let mut agent = Agent::new("agent1", "Test Agent", client);
    assert_eq!(agent.session_history_len(), 0);

    agent.receive_message(Role::Assistant, "[Agent 2]: Some context".to_string());
    assert_eq!(agent.session_history_len(), 1);

    agent.receive_message(Role::Assistant, "[Agent 3]: More context".to_string());
    assert_eq!(agent.session_history_len(), 2);
}

#[tokio::test]
async fn test_hub_routing_no_duplication() {
    use tokio::sync::Mutex as TokioMutex;

    /// Mock client that counts messages received per call
    struct CountingMockClient {
        message_counts: Arc<TokioMutex<Vec<usize>>>,
    }

    #[async_trait]
    impl ClientWrapper for CountingMockClient {
        async fn send_message(
            &self,
            messages: &[Message],
            _optional_grok_tools: Option<Vec<openai_rust::chat::GrokTool>>,
            _optional_openai_tools: Option<Vec<openai_rust::chat::OpenAITool>>,
        ) -> Result<Message, Box<dyn std::error::Error>> {
            let mut counts = self.message_counts.lock().await;
            counts.push(messages.len());
            Ok(Message {
                role: Role::Assistant,
                content: Arc::from("response"),
            })
        }

        fn model_name(&self) -> &str {
            "counting-mock"
        }

        async fn get_last_usage(&self) -> Option<TokenUsage> {
            None
        }
    }

    let counts1 = Arc::new(TokioMutex::new(Vec::new()));
    let counts2 = Arc::new(TokioMutex::new(Vec::new()));

    let agent1 = Agent::new(
        "agent1",
        "Agent 1",
        Arc::new(CountingMockClient {
            message_counts: counts1.clone(),
        }),
    );

    let agent2 = Agent::new(
        "agent2",
        "Agent 2",
        Arc::new(CountingMockClient {
            message_counts: counts2.clone(),
        }),
    );

    let mut orchestration =
        Orchestration::new("test", "Test").with_mode(OrchestrationMode::RoundRobin);

    orchestration.add_agent(agent1).unwrap();
    orchestration.add_agent(agent2).unwrap();

    let response = orchestration.run("Test question", 2).await.unwrap();
    assert_eq!(response.messages.len(), 4); // 2 agents * 2 rounds

    // Verify agent1's message counts grow incrementally (not full broadcast)
    let c1 = counts1.lock().await;
    let c2 = counts2.lock().await;

    // Round 1: agent1 sees system + user = 2 messages
    // Round 1: agent2 sees system + agent1's response + user = 3 messages
    // Round 2: agent1 sees prior + new messages + user (grows incrementally)
    // Round 2: agent2 sees prior + new messages + user (grows incrementally)
    //
    // Key assertion: agent2's first call should NOT receive the entire
    // conversation_history duplicated (which would be much larger)
    assert!(
        c1.len() >= 2,
        "Agent 1 should have been called at least 2 times"
    );
    assert!(
        c2.len() >= 2,
        "Agent 2 should have been called at least 2 times"
    );

    // In the old broadcast mode, agent2's second call would receive ALL messages
    // from ALL prior rounds plus the full conversation_history. With hub-routing,
    // each call receives only the session's accumulated messages.
    // Verify the message count is reasonable (not exploding)
    for &count in c1.iter() {
        assert!(
            count < 20,
            "Agent 1 message count {} should be reasonable",
            count
        );
    }
    for &count in c2.iter() {
        assert!(
            count < 20,
            "Agent 2 message count {} should be reasonable",
            count
        );
    }
}

#[tokio::test]
async fn test_agent_fork_with_context() {
    let client = Arc::new(MockClient {
        name: "mock".to_string(),
        response: "forked response".to_string(),
    });

    let mut agent = Agent::new("agent1", "Test Agent", client);
    agent.set_system_prompt("You are a helpful assistant.");
    agent.receive_message(Role::User, "Hello".to_string());
    agent.receive_message(Role::Assistant, "Hi there!".to_string());
    assert_eq!(agent.session_history_len(), 2);

    let forked = agent.fork_with_context();
    assert_eq!(forked.session_history_len(), 2);
    assert_eq!(forked.id, "agent1");
    assert_eq!(forked.name, "Test Agent");

    // Regular fork should have empty session
    let regular_fork = agent.fork();
    assert_eq!(regular_fork.session_history_len(), 0);
}
