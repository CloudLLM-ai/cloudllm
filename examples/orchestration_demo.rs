//! Comprehensive Orchestration Demo
//!
//! This example demonstrates all Orchestration modes and tool integration features:
//! - Parallel mode: Multiple agents respond simultaneously
//! - RoundRobin mode: Agents take turns
//! - Moderated mode: One agent orchestrates others
//! - Hierarchical mode: Layered problem solving
//! - Debate mode: Iterative convergence
//! - Tool usage: Agents with custom tools
//!
//! To run this example, you need to set environment variables for your LLM API keys:
//! export OPENAI_KEY=your_openai_key
//! export ANTHROPIC_KEY=your_anthropic_key (optional, for Claude)
//! export XAI_KEY=your_xai_key (optional, for Grok)
//!
//! Then run: cargo run --example orchestration_demo

use cloudllm::clients::openai::OpenAIClient;
use cloudllm::tool_protocol::{
    ToolMetadata, ToolParameter, ToolParameterType, ToolRegistry, ToolResult,
};
use cloudllm::tool_protocols::CustomToolProtocol;
use cloudllm::{
    orchestration::{Orchestration, OrchestrationMode},
    Agent,
};
use std::sync::Arc;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== CloudLLM Orchestration Demonstration ===\n");

    // Setup: Create a simple calculator tool
    let tool_adapter = CustomToolProtocol::new();

    tool_adapter
        .register_tool(
            ToolMetadata::new("calculate", "Performs basic mathematical calculations")
                .with_parameter(
                    ToolParameter::new("expression", ToolParameterType::String)
                        .with_description("Mathematical expression (e.g., '2 + 2', '10 * 5')")
                        .required(),
                ),
            Arc::new(|params| {
                let expr = params["expression"].as_str().unwrap_or("");
                // Simple calculator implementation
                let result = if expr.contains('+') {
                    let parts: Vec<&str> = expr.split('+').collect();
                    let a: f64 = parts[0].trim().parse().unwrap_or(0.0);
                    let b: f64 = parts[1].trim().parse().unwrap_or(0.0);
                    a + b
                } else if expr.contains('*') {
                    let parts: Vec<&str> = expr.split('*').collect();
                    let a: f64 = parts[0].trim().parse().unwrap_or(0.0);
                    let b: f64 = parts[1].trim().parse().unwrap_or(0.0);
                    a * b
                } else {
                    0.0
                };
                Ok(ToolResult::success(serde_json::json!({"result": result})))
            }),
        )
        .await;

    let tool_registry = ToolRegistry::new(Arc::new(tool_adapter));

    // Get API key from environment
    let openai_key = std::env::var("OPENAI_KEY").unwrap_or_else(|_| {
        eprintln!("Warning: OPENAI_KEY not set. Using placeholder.");
        "placeholder_key".to_string()
    });

    // Demo 1: Parallel Mode - Expert Panel
    println!("=== DEMO 1: Parallel Mode (Expert Panel) ===");
    println!("Three experts analyze a problem simultaneously\n");

    let agent1 = Agent::new(
        "architect",
        "System Architect",
        Arc::new(OpenAIClient::new_with_model_string(
            &openai_key,
            "gpt-4o-mini",
        )),
    )
    .with_expertise("Distributed systems, scalability, microservices architecture")
    .with_personality("Analytical, focuses on long-term maintainability");

    let agent2 = Agent::new(
        "security",
        "Security Expert",
        Arc::new(OpenAIClient::new_with_model_string(
            &openai_key,
            "gpt-4o-mini",
        )),
    )
    .with_expertise("Application security, threat modeling, secure coding practices")
    .with_personality("Cautious, emphasizes security-first design");

    let agent3 = Agent::new(
        "performance",
        "Performance Engineer",
        Arc::new(OpenAIClient::new_with_model_string(
            &openai_key,
            "gpt-4o-mini",
        )),
    )
    .with_expertise("Performance optimization, caching strategies, database tuning")
    .with_personality("Data-driven, focuses on metrics and benchmarks")
    .with_tools(tool_registry);

    let mut parallel_orchestration = Orchestration::new("expert-panel", "Technical Expert Panel")
        .with_mode(OrchestrationMode::Parallel)
        .with_system_context(
            "You are participating in a technical panel. Provide concise, expert analysis from your domain.",
        )
        .with_max_tokens(4096);

    parallel_orchestration.add_agent(agent1)?;
    parallel_orchestration.add_agent(agent2)?;
    parallel_orchestration.add_agent(agent3)?;

    let question = "We're building a payment processing system that needs to handle 10,000 transactions per second. What are the key considerations?";

    println!("Question: {}\n", question);

    match parallel_orchestration.run(question, 1).await {
        Ok(response) => {
            for msg in response.messages {
                if let Some(name) = msg.agent_name {
                    println!("--- {} ---", name);
                    println!("{}\n", msg.content);
                }
            }
        }
        Err(e) => eprintln!(
            "Parallel mode error: {}. This is expected if OPENAI_KEY is not set.",
            e
        ),
    }

    // Demo 2: Round-Robin Mode - Iterative Discussion
    println!("\n=== DEMO 2: Round-Robin Mode (Iterative Discussion) ===");
    println!("Agents take turns, building on each other's responses\n");

    let agent_a = Agent::new(
        "frontend",
        "Frontend Developer",
        Arc::new(OpenAIClient::new_with_model_string(
            &openai_key,
            "gpt-4o-mini",
        )),
    )
    .with_expertise("React, TypeScript, UI/UX design")
    .with_personality("User-focused, pragmatic");

    let agent_b = Agent::new(
        "backend",
        "Backend Developer",
        Arc::new(OpenAIClient::new_with_model_string(
            &openai_key,
            "gpt-4o-mini",
        )),
    )
    .with_expertise("Node.js, databases, API design")
    .with_personality("Systematic, detail-oriented");

    let mut roundrobin_orchestration = Orchestration::new("dev-team", "Development Team")
        .with_mode(OrchestrationMode::RoundRobin)
        .with_system_context(
            "You are on a development team. Listen to your teammates and build on their ideas. Keep responses brief.",
        );

    roundrobin_orchestration.add_agent(agent_a)?;
    roundrobin_orchestration.add_agent(agent_b)?;

    let task = "Design a real-time notification system for a chat application.";

    println!("Task: {}\n", task);

    match roundrobin_orchestration.run(task, 2).await {
        Ok(response) => {
            for (i, msg) in response.messages.iter().enumerate() {
                if let Some(name) = &msg.agent_name {
                    println!("[Turn {}] {} says:", i + 1, name);
                    println!("{}\n", msg.content);
                }
            }
        }
        Err(e) => eprintln!(
            "Round-robin mode error: {}. This is expected if OPENAI_KEY is not set.",
            e
        ),
    }

    // Demo 3: Hierarchical Mode - Multi-Layer Problem Solving
    println!("\n=== DEMO 3: Hierarchical Mode (Multi-Layer Problem Solving) ===");
    println!("Workers analyze, supervisors synthesize, executive decides\n");

    let worker1 = Agent::new(
        "db-analyst",
        "Database Analyst",
        Arc::new(OpenAIClient::new_with_model_string(
            &openai_key,
            "gpt-4o-mini",
        )),
    );

    let worker2 = Agent::new(
        "api-analyst",
        "API Analyst",
        Arc::new(OpenAIClient::new_with_model_string(
            &openai_key,
            "gpt-4o-mini",
        )),
    );

    let supervisor = Agent::new(
        "tech-lead",
        "Technical Lead",
        Arc::new(OpenAIClient::new_with_model_string(
            &openai_key,
            "gpt-4o-mini",
        )),
    )
    .with_expertise("System integration, architectural decisions");

    let mut hierarchical_orchestration = Orchestration::new("project-team", "Project Team Hierarchy")
        .with_mode(OrchestrationMode::Hierarchical {
            layers: vec![
                vec!["db-analyst".to_string(), "api-analyst".to_string()],
                vec!["tech-lead".to_string()],
            ],
        })
        .with_system_context("Analyze the problem from your perspective. Be concise.");

    hierarchical_orchestration.add_agent(worker1)?;
    hierarchical_orchestration.add_agent(worker2)?;
    hierarchical_orchestration.add_agent(supervisor)?;

    let problem = "We need to migrate from a monolithic database to a microservices architecture.";

    println!("Problem: {}\n", problem);

    match hierarchical_orchestration.run(problem, 1).await {
        Ok(response) => {
            for msg in response.messages {
                let layer = msg.metadata.get("layer").map(|s| s.as_str()).unwrap_or("0");
                if let Some(name) = msg.agent_name {
                    println!("[Layer {}] {} reports:", layer, name);
                    println!("{}\n", msg.content);
                }
            }
        }
        Err(e) => eprintln!(
            "Hierarchical mode error: {}. This is expected if OPENAI_KEY is not set.",
            e
        ),
    }

    // Demo 4: Debate Mode - Convergence Through Discussion
    println!("\n=== DEMO 4: Debate Mode (Convergence Through Discussion) ===");
    println!("Agents debate until reaching consensus\n");

    let debater1 = Agent::new(
        "optimist",
        "Optimistic Engineer",
        Arc::new(OpenAIClient::new_with_model_string(
            &openai_key,
            "gpt-4o-mini",
        )),
    )
    .with_personality("Innovative, focuses on opportunities");

    let debater2 = Agent::new(
        "realist",
        "Pragmatic Engineer",
        Arc::new(OpenAIClient::new_with_model_string(
            &openai_key,
            "gpt-4o-mini",
        )),
    )
    .with_personality("Practical, considers constraints");

    let mut debate_orchestration = Orchestration::new("debate-team", "Technical Debate")
        .with_mode(OrchestrationMode::Debate {
            max_rounds: 2,
            convergence_threshold: Some(0.8),
        })
        .with_system_context(
            "Engage in constructive debate. Challenge ideas respectfully and find common ground. Keep responses concise.",
        );

    debate_orchestration.add_agent(debater1)?;
    debate_orchestration.add_agent(debater2)?;

    let topic = "Should we use TypeScript or JavaScript for our new project?";

    println!("Debate Topic: {}\n", topic);

    match debate_orchestration.run(topic, 1).await {
        Ok(response) => {
            for msg in response.messages {
                let round = msg.metadata.get("round").map(|s| s.as_str()).unwrap_or("0");
                if let Some(name) = msg.agent_name {
                    println!("[Round {}] {}: {}\n", round, name, msg.content);
                }
            }
        }
        Err(e) => eprintln!(
            "Debate mode error: {}. This is expected if OPENAI_KEY is not set.",
            e
        ),
    }

    println!("\n=== Orchestration Demonstration Complete ===");
    println!("\nKey Takeaways:");
    println!("✓ Parallel: Fast, independent analysis");
    println!("✓ Round-Robin: Iterative, builds context");
    println!("✓ Hierarchical: Structured, multi-level synthesis");
    println!("✓ Debate: Collaborative convergence");
    println!("✓ Tools: Agents can use external capabilities");
    println!("\nMix and match modes, models, and tools to solve complex problems!");

    Ok(())
}
