//! Anthropic Agent Teams Orchestration Example
//!
//! Demonstrates decentralized task-based coordination with no central orchestrator.
//! Four agents with **mixed LLM providers** (OpenAI + Claude Haiku 4.5) autonomously
//! discover and claim tasks from a shared Memory pool, coordinate their work, and
//! report completion.
//!
//! # Team Composition (Mixed Providers)
//!
//! - **Agent 1 (Researcher)**: OpenAI GPT — finds and summarizes sources
//! - **Agent 2 (Analyst)**: Claude Haiku 4.5 — synthesizes findings into themes
//! - **Agent 3 (Writer)**: OpenAI GPT — drafts clear documentation
//! - **Agent 4 (Reviewer)**: Claude Haiku 4.5 — ensures quality and completeness
//!
//! # Task Pool
//!
//! 8 work items spanning research, analysis, writing, and review.
//! Agents use the Memory tool to:
//! 1. LIST unclaimed tasks
//! 2. GET task descriptions
//! 3. PUT claim: `teams:<pool_id>:claimed:<task_id>`
//! 4. Work on the task
//! 5. PUT result: `teams:<pool_id>:completed:<task_id>`
//!
//! # Event Handling
//!
//! All events (task claims, completions, failures) flow through a single
//! event handler that displays progress in real-time.
//!
//! # Environment Variables
//!
//! - `OPENAI_API_KEY` — API key for OpenAI (optional, defaults to "demo-key")
//! - `OPENAI_MODEL` — OpenAI model (optional, defaults to "gpt-4o-mini")
//! - `ANTHROPIC_API_KEY` — API key for Claude (optional, defaults to "demo-key")

use async_trait::async_trait;
use cloudllm::clients::claude::{ClaudeClient, Model};
use cloudllm::clients::openai::OpenAIClient;
use cloudllm::event::{EventHandler, OrchestrationEvent};
use cloudllm::orchestration::{Orchestration, OrchestrationMode, WorkItem};
use cloudllm::Agent;
use std::sync::Arc;

/// Simple event handler that logs orchestration events to stdout
struct TeamsEventHandler;

#[async_trait]
impl EventHandler for TeamsEventHandler {
    async fn on_orchestration_event(&self, event: &OrchestrationEvent) {
        match event {
            OrchestrationEvent::RunStarted {
                orchestration_id,
                orchestration_name,
                mode,
                agent_count,
            } => {
                println!(
                    "\n🚀 {} (ID: {}) — {} mode with {} agents",
                    orchestration_name, orchestration_id, mode, agent_count
                );
            }
            OrchestrationEvent::RoundStarted {
                orchestration_id: _,
                round,
            } => {
                println!("\n📍 Iteration {}", round);
            }
            OrchestrationEvent::AgentSelected {
                orchestration_id: _,
                agent_id: _,
                agent_name,
                reason,
            } => {
                println!("  → {} ({})", agent_name, reason);
            }
            OrchestrationEvent::TaskClaimed {
                orchestration_id: _,
                agent_id: _,
                agent_name,
                task_id,
            } => {
                println!("    ✋ {} claimed task: {}", agent_name, task_id);
            }
            OrchestrationEvent::TaskCompleted {
                orchestration_id: _,
                agent_id: _,
                agent_name,
                task_id,
                result: _,
            } => {
                println!("    ✅ {} completed: {}", agent_name, task_id);
            }
            OrchestrationEvent::TaskFailed {
                orchestration_id: _,
                agent_id: _,
                agent_name,
                task_id,
                error,
            } => {
                println!("    ❌ {} failed on {}: {}", agent_name, task_id, error);
            }
            OrchestrationEvent::RoundCompleted { .. } => {
                // Round completion is less verbose
            }
            OrchestrationEvent::RunCompleted {
                orchestration_id: _,
                orchestration_name: _,
                rounds,
                total_tokens,
                is_complete,
            } => {
                println!(
                    "\n✨ Run completed in {} iterations, {} tokens, complete={}",
                    rounds, total_tokens, is_complete
                );
            }
            _ => {
                // Ignore other events
            }
        }
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    // Initialize logging
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    println!("═══════════════════════════════════════════════════════════════");
    println!("    Anthropic Agent Teams: Decentralized Task Coordination");
    println!("═══════════════════════════════════════════════════════════════");

    // Create task pool
    let tasks = vec![
        WorkItem::new(
            "research_nmn",
            "Research phase — NMN+ mechanisms and pathways",
            "Gather and summarize current scientific literature on NAD+ boosting, \
             NMN metabolism, mitochondrial function, and sirtuins activation",
        ),
        WorkItem::new(
            "analyze_longevity",
            "Analysis phase — longevity effects and clinical outcomes",
            "Synthesize findings on aging reversal, lifespan extension, \
             and key biomarkers of rejuvenation (NAD+ levels, cellular senescence)",
        ),
        WorkItem::new(
            "research_alzheimers",
            "Research phase — Alzheimer's pathology and neurodegeneration",
            "Find and summarize evidence on amyloid-beta, tau tangles, \
             neuroinflammation, and cognitive decline in Alzheimer's disease",
        ),
        WorkItem::new(
            "analyze_neuroprotection",
            "Analysis phase — NMN+ neuroprotective mechanisms",
            "Analyze how NAD+ restoration protects neurons, supports mitochondrial energy, \
             reduces neuroinflammation, and may prevent amyloid accumulation",
        ),
        WorkItem::new(
            "memory_recovery",
            "Research phase — memory recovery and neuroplasticity",
            "Investigate evidence for memory restoration, synaptic plasticity recovery, \
             and cognitive function restoration in Alzheimer's models",
        ),
        WorkItem::new(
            "clinical_integration",
            "Analysis phase — clinical feasibility and dosing",
            "Assess therapeutic potential, optimal dosing protocols, bioavailability, \
             and safety profile of NMN+ in human trials",
        ),
        WorkItem::new(
            "synthesis_report",
            "Writing phase — comprehensive synthesis and recommendations",
            "Draft 3-4 page report synthesizing all findings with clear conclusions \
             and future research directions",
        ),
        WorkItem::new(
            "final_review",
            "Quality review — accuracy, completeness, and impact",
            "Peer review for scientific accuracy, identify gaps, suggest improvements, \
             ensure evidence-based conclusions",
        ),
    ];

    println!("\n📋 Task Pool: {} items", tasks.len());
    for (i, task) in tasks.iter().enumerate() {
        println!("  {}. {} ({})", i + 1, task.description, task.id);
    }

    // Create agents with mixed LLM providers
    let openai_model = std::env::var("OPENAI_MODEL").unwrap_or_else(|_| "gpt-4o-mini".to_string());
    let openai_key = std::env::var("OPENAI_API_KEY").unwrap_or_else(|_| "demo-key".to_string());
    let claude_key = std::env::var("ANTHROPIC_API_KEY").unwrap_or_else(|_| "demo-key".to_string());

    // Agent 1: Research Agent (OpenAI)
    let agent1 = Agent::new(
        "researcher",
        "Research Agent (GPT)",
        Arc::new(OpenAIClient::new_with_model_string(
            &openai_key,
            &openai_model,
        )),
    );

    // Agent 2: Analysis Agent (Claude Haiku 4.5)
    let agent2 = Agent::new(
        "analyst",
        "Analysis Agent (Claude Haiku 4.5)",
        Arc::new(ClaudeClient::new_with_model_enum(
            &claude_key,
            Model::ClaudeHaiku45,
        )),
    );

    // Agent 3: Writing Agent (OpenAI)
    let agent3 = Agent::new(
        "writer",
        "Writing Agent (GPT)",
        Arc::new(OpenAIClient::new_with_model_string(
            &openai_key,
            &openai_model,
        )),
    );

    // Agent 4: Review Agent (Claude Haiku 4.5)
    let agent4 = Agent::new(
        "reviewer",
        "Review Agent (Claude Haiku 4.5)",
        Arc::new(ClaudeClient::new_with_model_enum(
            &claude_key,
            Model::ClaudeHaiku45,
        )),
    );

    // Create orchestration
    let mut orchestration =
        Orchestration::new("teams-demo", "AnthropicAgentTeams Demo: 4 Agents, 8 Tasks")
            .with_mode(OrchestrationMode::AnthropicAgentTeams {
                pool_id: "demo-pool-1".to_string(),
                tasks: tasks.clone(),
                max_iterations: 4,
            })
            .with_system_context(
                "You are a specialized agent in a coordinated team. \
         Your role is to claim unclaimed tasks from a shared task pool and complete them. \
         Work autonomously and collaboratively. Focus on quality and clear communication.",
            )
            .with_max_tokens(4096)
            .with_event_handler(Arc::new(TeamsEventHandler));

    // Register agents
    orchestration.add_agent(agent1)?;
    orchestration.add_agent(agent2)?;
    orchestration.add_agent(agent3)?;
    orchestration.add_agent(agent4)?;

    println!("\n👥 Team Members (Mixed LLM Providers):");
    println!("  1. Research Agent (GPT) — specialist in finding and summarizing sources");
    println!("  2. Analysis Agent (Claude Haiku 4.5) — synthesizes findings into themes");
    println!("  3. Writing Agent (GPT) — drafts clear, concise documentation");
    println!("  4. Review Agent (Claude Haiku 4.5) — ensures quality and completeness");

    // Run orchestration
    let user_prompt = "Prepare a comprehensive scientific report on NMN+ (Nicotinamide Mononucleotide) \
                        for longevity and its potential effects on Alzheimer's disease, specifically \
                        focusing on memory recovery and reversal of cognitive decline. The team should: \
                        (1) research NMN+ mechanisms and its effects on longevity, \
                        (2) analyze how it protects against Alzheimer's pathology, \
                        (3) investigate memory recovery and neuroplasticity restoration, and \
                        (4) synthesize findings into an evidence-based clinical perspective. \
                        Each agent claims tasks autonomously from the shared pool.";

    println!("\n📝 User Prompt:\n  {}\n", user_prompt);

    let response = orchestration.run(user_prompt, 1).await?;

    // Print results
    println!("\n📊 Orchestration Results:");
    println!("  Iterations: {}", response.round);
    println!("  Is Complete: {}", response.is_complete);
    println!(
        "  Convergence Score: {:.0}%",
        response.convergence_score.unwrap_or(0.0) * 100.0
    );
    println!("  Total Tokens: {}", response.total_tokens_used);

    println!(
        "\n💬 Conversation History ({} messages):",
        response.messages.len()
    );
    for msg in response.messages.iter().take(5) {
        let source = msg.agent_name.as_deref().unwrap_or("system");
        let preview = if msg.content.len() > 100 {
            format!("{}...", &msg.content[..100])
        } else {
            msg.content.to_string()
        };
        println!("  [{}]: {}", source, preview);
    }

    if response.messages.len() > 5 {
        println!("  ... ({} more messages)", response.messages.len() - 5);
    }

    println!("\n✅ Orchestration complete!");

    Ok(())
}
