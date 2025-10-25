//! Council Discussion with Shared Memory Example
//!
//! This example demonstrates how a council of agents can use a shared Memory tool to:
//! - Coordinate their work on a complex problem
//! - Store decisions and consensus points
//! - Track the discussion state for recovery
//! - Maintain session state across multiple rounds
//!
//! The shared memory acts as a "chalkboard" that all agents can read and write to,
//! enabling sophisticated coordination patterns.

use cloudllm::clients::openai::{Model, OpenAIClient};
use cloudllm::council::{Agent, Council, CouncilMode};
use cloudllm::tool_adapters::MemoryToolAdapter;
use cloudllm::tool_protocol::ToolRegistry;
use cloudllm::tools::Memory;
use std::sync::Arc;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    cloudllm::init_logger();

    let api_key = std::env::var("OPEN_AI_SECRET").unwrap_or_else(|_| "sk-test".to_string());

    // Create shared memory for the council
    let shared_memory = Arc::new(Memory::new());

    // Create memory adapter and registry - ALL AGENTS WILL SHARE THIS
    let memory_adapter = Arc::new(MemoryToolAdapter::new(shared_memory.clone()));
    let shared_registry = Arc::new(ToolRegistry::new(memory_adapter));

    // Create three agents with different roles, all with access to shared memory
    let analyst = Agent::new(
        "analyst",
        "Data Analyst",
        Arc::new(OpenAIClient::new_with_model_enum(
            &api_key,
            Model::GPT41Nano,
        )),
    )
    .with_expertise("Analyzes data patterns and identifies key metrics")
    .with_personality("Quantitative, focused on numbers and trends")
    .with_tools(shared_registry.clone());

    let strategist = Agent::new(
        "strategist",
        "Strategic Advisor",
        Arc::new(OpenAIClient::new_with_model_enum(
            &api_key,
            Model::GPT41Nano,
        )),
    )
    .with_expertise("Creates strategic plans and identifies opportunities")
    .with_personality("Visionary, thinks about long-term implications")
    .with_tools(shared_registry.clone());

    let implementer = Agent::new(
        "implementer",
        "Implementation Expert",
        Arc::new(OpenAIClient::new_with_model_enum(
            &api_key,
            Model::GPT41Nano,
        )),
    )
    .with_expertise("Focuses on practical execution and feasibility")
    .with_personality("Pragmatic, detail-oriented, risk-aware")
    .with_tools(shared_registry.clone());

    // Create a council with these agents
    let mut council = Council::new("decision-council", "Strategic Decision Council")
        .with_mode(CouncilMode::RoundRobin)
        .with_system_context(
            "You are part of a council making strategic decisions. \
             You have access to SHARED MEMORY where the council records:\
             - Key findings and insights from other agents\
             - Consensus points and decisions\
             - Action items and next steps\n\n\
             Use the memory tool liberally to:\
             1. Store your analysis and recommendations\
             2. Review what others have stored\
             3. Contribute to shared decision-making\n\n\
             Memory commands:\
             - Save finding: {{\"tool_call\": {{\"name\": \"memory\", \"parameters\": {{\"command\": \"P analyst_finding Finding_Description 3600\"}}}}}}\
             - List all shared findings: {{\"tool_call\": {{\"name\": \"memory\", \"parameters\": {{\"command\": \"L\"}}}}}}\
             - Retrieve specific finding: {{\"tool_call\": {{\"name\": \"memory\", \"parameters\": {{\"command\": \"G analyst_finding META\"}}}}}}\
             - Record consensus: {{\"tool_call\": {{\"name\": \"memory\", \"parameters\": {{\"command\": \"P consensus Decision_or_Agreement 3600\"}}}}}}"
        )
        .with_max_tokens(8192);

    council.add_agent(analyst)?;
    council.add_agent(strategist)?;
    council.add_agent(implementer)?;

    println!("=== Council Configuration ===");
    println!("Council: {}", council.name);
    println!("Mode: Round-Robin discussion");
    println!("Agents: {}", council.list_agents().len());
    for agent in council.list_agents() {
        println!("  - {} ({})", agent.name, agent.id);
        println!("    Has shared memory: {}", agent.tool_registry.is_some());
    }

    println!("\n=== Problem Statement ===");
    let problem = "We need to decide on a new technology platform for our company. \
                   The analysts should present data on options, \
                   the strategist should evaluate strategic fit, \
                   and the implementer should assess feasibility. \
                   Use the shared memory to coordinate your recommendations.";

    println!("{}\n", problem);

    println!("=== How the Council Uses Shared Memory ===");
    println!("Round 1: Data Analyst");
    println!("  → Analyzes platforms and stores findings in shared memory");
    println!("  → Records metrics: \"P platform_comparison Option_A_Best_performance 3600\"");
    println!();
    println!("Round 2: Strategic Advisor");
    println!("  → Retrieves analyst's findings: \"G platform_comparison\"");
    println!(
        "  → Adds strategic perspective to memory: \"P strategy_fit Aligns_with_growth_plan 3600\""
    );
    println!();
    println!("Round 3: Implementation Expert");
    println!("  → Reviews both previous analyses");
    println!("  → Stores feasibility assessment: \"P implementation_plan 6_month_rollout 3600\"");
    println!("  → Records consensus decision: \"P final_recommendation Option_A_selected 3600\"");

    println!("\n=== Shared Memory Lifecycle ===");

    // Pre-populate some data to simulate agent interactions
    println!("Simulating council discussion with shared memory...\n");

    // Simulate analyst storing findings
    println!("Analyst stores initial findings...");
    shared_memory.put(
        "analyst_findings".to_string(),
        "Platform_A: 99.9% uptime, 50ms latency, 2x cost".to_string(),
        Some(3600),
    );
    shared_memory.put(
        "analyst_preference".to_string(),
        "Platform_A has best performance metrics".to_string(),
        Some(3600),
    );

    // Simulate strategist reading and adding
    println!("Strategist reviews findings and adds strategy...");
    let analyst_pref = shared_memory.get("analyst_preference", false);
    println!("  Read from memory: {:?}", analyst_pref);

    shared_memory.put(
        "strategic_fit".to_string(),
        "Platform_A aligns with 5-year growth strategy".to_string(),
        Some(3600),
    );

    // Simulate implementer reviewing and making decision
    println!("Implementer reviews all insights and decides...");
    let all_keys = shared_memory.list_keys();
    println!(
        "  Council memory now contains {} items: {:?}",
        all_keys.len(),
        all_keys
    );

    shared_memory.put(
        "implementation_feasible".to_string(),
        "Yes: 6-month rollout plan ready".to_string(),
        Some(3600),
    );
    shared_memory.put(
        "council_decision".to_string(),
        "CONSENSUS: Proceed with Platform_A immediately".to_string(),
        Some(7200), // Longer TTL for final decision
    );

    println!("\n=== Final Shared Memory State ===");
    let (total, _, _) = shared_memory.get_total_bytes_stored();
    println!("Total memory used: {} bytes", total);
    println!("All stored entries:");
    for key in shared_memory.list_keys() {
        if let Some((value, metadata)) = shared_memory.get(&key, true) {
            let meta = metadata.unwrap();
            println!(
                "  {} = \"{}\" (expires in {} seconds)",
                key,
                value,
                meta.expires_in.unwrap_or(0)
            );
        }
    }

    println!("\n=== Key Benefits for Council Discussions ===");
    println!("✓ All agents access the same facts and previous decisions");
    println!("✓ Enables sophisticated consensus-building");
    println!("✓ Natural way to track multi-round discussions");
    println!("✓ Minimal token overhead thanks to succinct protocol");
    println!("✓ Automatic cleanup of stale decisions via TTL");
    println!("✓ Audit trail of how decisions were reached");

    println!("\n=== Session Recovery ===");
    println!("If the council is interrupted and restarted:");
    println!("  - All agents can review the full discussion history");
    println!("  - Consensus points are preserved");
    println!("  - Council can resume from where it left off");
    println!("  - No need to re-analyze or repeat discussions");

    Ok(())
}
