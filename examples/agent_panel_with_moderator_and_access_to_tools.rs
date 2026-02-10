//! Four-Agent Panel with Moderator and Shared Tools
//!
//! This example demonstrates a sophisticated multi-agent system using the Orchestration API
//! to estimate global CO₂ emissions from Bitcoin mining. It showcases:
//!
//! - **Parallel execution**: Three workers run simultaneously via Orchestration.Parallel mode
//! - **Two iterative rounds**: Independent work (Round 1) → Moderator feedback → Revisions (Round 2)
//! - **Shared tools**: Memory (KV store), Calculator (mathematical operations)
//! - **Agent autonomy**: LLM models decide which tools to use based on tasks
//! - **Structured outputs**: JSON Memory keys with proper units and ISO-8601 timestamps
//!
//! Workers operate in parallel during each round, storing results in shared Memory.
//! The moderator then validates, provides feedback, and synthesizes the final report.

use cloudllm::clients::grok::GrokClient;
use cloudllm::clients::openai::{Model, OpenAIClient};
use cloudllm::tool_protocol::ToolRegistry;
use cloudllm::tool_protocol::{ToolMetadata, ToolParameter, ToolParameterType, ToolResult};
use cloudllm::tool_protocols::{CustomToolProtocol, MemoryProtocol};
use cloudllm::tools::Memory;
use cloudllm::{
    orchestration::{Orchestration, OrchestrationMode},
    Agent,
};
use std::collections::HashMap;
use std::sync::Arc;

#[allow(dead_code)]
struct PanelWorkflow {
    memory: Arc<Memory>,
    memory_protocol: Arc<MemoryProtocol>,
    custom_protocol: Arc<CustomToolProtocol>,
    worker_a: Agent,
    worker_b: Agent,
    worker_c: Agent,
    moderator: Agent,
}

impl PanelWorkflow {
    async fn new(
        api_key_grok: &str,
        api_key_openai: &str,
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let memory = Arc::new(Memory::new());

        // Create shared Memory tool access for all agents
        let memory_protocol = Arc::new(MemoryProtocol::new(memory.clone()));

        // Create a custom tool protocol with Calculator tool
        let custom_protocol = Arc::new(CustomToolProtocol::new());

        // Register Calculator tool using the actual Calculator implementation
        use cloudllm::tools::Calculator;
        let calculator = Arc::new(Calculator::new());
        let calculator_clone = calculator.clone();
        custom_protocol
            .register_tool(
                ToolMetadata::new(
                    "calculator",
                    "Performs mathematical calculations with arbitrary precision",
                )
                .with_parameter(
                    ToolParameter::new("expression", ToolParameterType::String)
                        .with_description(
                            "Mathematical expression to evaluate (e.g., '650*1e6*25*24/1000')",
                        )
                        .required(),
                ),
                Arc::new(move |params| {
                    let expr = params["expression"].as_str().unwrap_or("0");
                    // Use the actual Calculator tool to evaluate the expression
                    // Note: We use tokio::runtime to block on the async call from a sync context
                    let result =
                        tokio::runtime::Handle::current().block_on(calculator_clone.evaluate(expr));
                    match result {
                        Ok(value) => Ok(ToolResult {
                            success: true,
                            output: serde_json::json!({ "result": value, "expression": expr }),
                            error: None,
                            metadata: HashMap::new(),
                        }),
                        Err(e) => Ok(ToolResult {
                            success: false,
                            output: serde_json::json!({ "expression": expr }),
                            error: Some(format!("Calculation error: {}", e)),
                            metadata: HashMap::new(),
                        }),
                    }
                }),
            )
            .await;

        // Worker A: Data Collector (uses Memory)
        let registry_a = ToolRegistry::new(memory_protocol.clone());
        let worker_a = Agent::new(
            "worker_a",
            "Data Collector",
            Arc::new(GrokClient::new_with_model_str(api_key_grok, "grok-4")),
        )
        .with_expertise(
            "Fetches current global Bitcoin hashrate and energy efficiency metrics from primary sources"
        )
        .with_personality("Meticulous, data-driven, focuses on source quality and recency")
        .with_tools(registry_a);

        // Worker B: Energy Analyst (uses Calculator + Memory)
        let registry_b = ToolRegistry::new(custom_protocol.clone());
        let worker_b = Agent::new(
            "worker_b",
            "Energy Analyst",
            Arc::new(GrokClient::new_with_model_str(api_key_grok, "grok-4")),
        )
        .with_expertise(
            "Converts hashrate and efficiency data into daily energy consumption (kWh/day) using precise unit conversion"
        )
        .with_personality("Systematic, careful with unit conversions, emphasizes precision in calculations")
        .with_tools(registry_b);

        // Worker C: Emissions Analyst (uses Memory + Calculator)
        let registry_c = ToolRegistry::new(memory_protocol.clone());
        let worker_c = Agent::new(
            "worker_c",
            "Emissions Analyst",
            Arc::new(GrokClient::new_with_model_str(api_key_grok, "grok-4")),
        )
        .with_expertise(
            "Fetches global electricity emission factors and computes CO₂ emissions in tons per day"
        )
        .with_personality("Thorough, questions assumptions, provides uncertainty bounds")
        .with_tools(registry_c);

        // Moderator: Verifier & Integrator (uses Memory)
        let moderator_registry = ToolRegistry::new(memory_protocol.clone());
        let moderator = Agent::new(
            "moderator",
            "Verifier & Integrator",
            Arc::new(OpenAIClient::new_with_model_enum(
                api_key_openai,
                Model::GPT41Nano,
            )),
        )
        .with_expertise(
            "Validates data recency, unit coherence, magnitude sanity; integrates and audits multi-agent findings"
        )
        .with_personality("Skeptical, audit-focused, demands reproducibility and clarity")
        .with_tools(moderator_registry);

        Ok(Self {
            memory,
            memory_protocol,
            custom_protocol,
            worker_a,
            worker_b,
            worker_c,
            moderator,
        })
    }

    async fn run_round_1(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        println!("\n╔════════════════════════════════════════════════════════════════╗");
        println!("║              ROUND 1: PARALLEL INDEPENDENT WORK               ║");
        println!("╚════════════════════════════════════════════════════════════════╝\n");

        println!("🔄 Launching three workers in PARALLEL...\n");

        // Create an orchestration for parallel worker execution
        let mut orchestration = Orchestration::new("workers-panel", "Worker Analysis Panel")
            .with_mode(OrchestrationMode::Parallel)
            .with_max_tokens(4096);

        orchestration.add_agent(self.worker_a.fork())?;
        orchestration.add_agent(self.worker_b.fork())?;
        orchestration.add_agent(self.worker_c.fork())?;

        let prompt = r#"You are part of a three-agent research team analyzing Bitcoin mining CO₂ emissions.

WORKER A (Data Collector): Research and fetch CURRENT global Bitcoin metrics:
- Global hashrate in EH/s (exahashes per second)
- Energy efficiency in J/TH (joules per terahash)
Store findings in Memory with keys: r1/source.hashrate, r1/source.energy_per_ths

WORKER B (Energy Analyst): Read Worker A's findings, calculate daily energy:
- TH/s = EH/s × 10^6
- Power (W) = TH/s × (J/TH)
- kWh/day = Power (W) × 24 / 1000
Use the Calculator tool for math. Store in Memory: r1/energy.kwh_per_day

WORKER C (Emissions Analyst): Research global electricity emission factor:
- Find kgCO2/kWh (current global average)
- Read r1/energy.kwh_per_day from Memory
- Calculate: CO₂ tons/day = kWh/day × kgCO2/kWh / 1000
Store in Memory: r1/emissions.tons_per_day, r1/source.co2_factor

ALL WORKERS: Use Memory tool to store findings with:
- Exact values and source URLs
- ISO-8601 timestamps
- Any uncertainty notes

Execute your task independently. Do NOT wait for others.
"#;

        let response = orchestration.run(prompt, 1).await?;

        println!("✓ Round 1 complete. All workers completed in parallel:");
        for msg in &response.messages {
            if let Some(agent_id) = &msg.agent_id {
                println!(
                    "\n  📌 {}: {}",
                    agent_id,
                    &msg.content[..100.min(msg.content.len())]
                );
            }
        }

        println!("\n✓ All worker findings stored in Memory under r1/* namespace.\n");
        Ok(())
    }

    async fn run_moderator_review_round_1(
        &self,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        println!("\n╔════════════════════════════════════════════════════════════════╗");
        println!("║               MODERATOR ROUND 1: VALIDATION & FEEDBACK        ║");
        println!("╚════════════════════════════════════════════════════════════════╝\n");

        let task_moderator = r#"
You are the moderator validating Round 1 findings stored in Memory. Your task:

1. Read ALL r1/source.* and r1/energy.* and r1/emissions.* values from Memory
2. Validate:
   - Timestamps ≤48 hours old? ✓
   - Units coherent? (EH/s → TH/s → W → kWh/day) ✓
   - Magnitudes reasonable? (Bitcoin hashrate ~600-800 EH/s typical)
   - Primary sources used (not blogs)? ✓
   - Raw data and trails present? ✓

3. Generate feedback for each worker:
   - Issues found (if any)
   - Specific actions for Round 2
   - Permission grants for Round 2 read/write

4. Store feedback/r1 in Memory as JSON with per-worker entries

Allow Round-2: Workers may now read r1/source.* and r1/energy.* for validation.
"#;

        let response = self
            .moderator
            .generate(
                "You are validating agent work. Use Memory to read findings and store feedback.",
                task_moderator,
                &[],
            )
            .await?;

        println!("✓ Moderator feedback generated:\n{}\n", response);
        Ok(())
    }

    async fn run_round_2(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        println!("\n╔════════════════════════════════════════════════════════════════╗");
        println!("║          ROUND 2: PARALLEL REVISIONS WITH BOUNDS              ║");
        println!("╚════════════════════════════════════════════════════════════════╝\n");

        println!("🔄 Launching three workers in PARALLEL for revisions...\n");

        // Create orchestration for parallel Round 2 execution
        let mut orchestration = Orchestration::new("workers-r2", "Worker Revision Panel")
            .with_mode(OrchestrationMode::Parallel)
            .with_max_tokens(4096);

        orchestration.add_agent(self.worker_a.fork())?;
        orchestration.add_agent(self.worker_b.fork())?;
        orchestration.add_agent(self.worker_c.fork())?;

        let prompt = r#"
Round 2: Refine your Round 1 estimates with uncertainty bounds.

1. Read feedback/r1 from Memory to see moderator comments
2. Read your r1/* entries and refine them with bounds
3. Provide:
   - Central value (your best estimate)
   - Low bound (conservative)
   - High bound (optimistic)
   - Confidence score (0-1)

WORKER A: Store r2/source.hashrate.v2 and r2/source.energy_per_ths.v2 with bounds
WORKER B: Store r2/energy.kwh_per_day.v2 with low/mid/high calculations using Calculator
WORKER C: Store r2/source.co2_factor.v2 and r2/emissions.tons_per_day.v2 with ranges

ALL: Use ISO-8601 timestamps, include sources and confidence scores.

Execute in parallel. Store results using Memory tool with .v2 suffix.
"#;

        let response = orchestration.run(prompt, 1).await?;

        println!("✓ Round 2 complete. All workers revised estimates in parallel:");
        for msg in &response.messages {
            if let Some(agent_id) = &msg.agent_id {
                println!(
                    "\n  📌 {}: {}",
                    agent_id,
                    &msg.content[..100.min(msg.content.len())]
                );
            }
        }

        println!("\n✓ All refined estimates stored in Memory under r2/*.v2 namespace.\n");
        Ok(())
    }

    async fn run_moderator_finalization(
        &self,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        println!("\n╔════════════════════════════════════════════════════════════════╗");
        println!("║          MODERATOR ROUND 2: FINALIZATION & SYNTHESIS          ║");
        println!("╚════════════════════════════════════════════════════════════════╝\n");

        let task_final = r#"
Final task: Synthesize all Round 2 findings and create integrated report.

1. Read all r2/*.v2 entries from Memory (refined Round 2 data)
2. Assemble final/report with:
   - Central values: hashrate_ehs, energy_per_ths_j, energy_kwh_per_day, co2_factor_kg_per_kwh, emissions_tons_per_day
   - Ranges: energy_kwh_per_day {low, high}, emissions_tons_per_day {low, high}
   - Assumptions list (5 key assumptions about this analysis)
   - Sources: array of {metric, url}
   - Confidence: weighted average (0-1)
   - Timestamp: ISO-8601

3. Create final/summary - concise one-liner:
   "Estimated ≈XXXk tCO₂/day (±YY% range), using H=XXX EH/s, η=XX J/TH, EF=X.XX kgCO₂/kWh; confidence ZZ%."

4. Create meta/current - canonical version pointers:
   {"hashrate": "r2/source.hashrate.v2", "energy_per_ths": "r2/source.energy_per_ths.v2", ...}

Store all three in Memory (no TTL for final outputs).
"#;

        let response = self
            .moderator
            .generate(
                "You have Memory access. Create final integrated report from Round 2 data.",
                task_final,
                &[],
            )
            .await?;

        println!("✓ Moderator finalized:\n{}\n", response);
        Ok(())
    }

    async fn dump_memory_state(&self) {
        println!("\n╔════════════════════════════════════════════════════════════════╗");
        println!("║              MEMORY STATE (All Stored Keys & Values)           ║");
        println!("╚════════════════════════════════════════════════════════════════╝\n");

        let keys = self.memory.list_keys();
        println!("Total keys stored: {}\n", keys.len());

        for key in keys {
            if let Some((value, _metadata)) = self.memory.get(&key, false) {
                println!("📌 {}", key);
                // Truncate very long values
                let display_value = if value.len() > 400 {
                    let mut end = 400;
                    while !value.is_char_boundary(end) {
                        end -= 1;
                    }
                    format!("{}...", &value[..end])
                } else {
                    value
                };
                println!("   {}\n", display_value);
            }
        }
    }
}

#[tokio::main]
async fn main() {
    cloudllm::init_logger();

    let api_key_grok =
        std::env::var("XAI_API_KEY").unwrap_or_else(|_| "xai-placeholder".to_string());
    let api_key_openai =
        std::env::var("OPENAI_API_KEY").unwrap_or_else(|_| "sk-placeholder".to_string());

    println!("\n╔════════════════════════════════════════════════════════════════╗");
    println!("║    FOUR-AGENT PANEL WITH MODERATOR & PARALLEL EXECUTION      ║");
    println!("║     Estimating Global CO₂ from Bitcoin Mining (tons/day)      ║");
    println!("╚════════════════════════════════════════════════════════════════╝");

    println!("\n📋 PANEL CONFIGURATION:");
    println!("   Workers (Grok-based, RUN IN PARALLEL):");
    println!("   ├─ Worker A (Data Collector): Researches hashrate & efficiency");
    println!("   ├─ Worker B (Energy Analyst): Calculates kWh/day");
    println!("   └─ Worker C (Emissions Analyst): Estimates CO₂/day");
    println!("   ");
    println!("   Moderator (OpenAI GPT-4.1): Validates, provides feedback, synthesizes output");
    println!("   ");
    println!("   Shared Tools: Memory (KV store), Calculator (math operations)");
    println!("   ");
    println!("   Execution: Orchestration.Parallel for workers, single agent for moderator");
    println!("   Each round: workers run simultaneously, then moderator reviews.");

    match PanelWorkflow::new(&api_key_grok, &api_key_openai).await {
        Ok(panel) => {
            // Execute two-round workflow
            if let Err(e) = panel.run_round_1().await {
                eprintln!("Error in Round 1: {}", e);
                return;
            }

            if let Err(e) = panel.run_moderator_review_round_1().await {
                eprintln!("Error in Moderator Round 1 review: {}", e);
                return;
            }

            if let Err(e) = panel.run_round_2().await {
                eprintln!("Error in Round 2: {}", e);
                return;
            }

            if let Err(e) = panel.run_moderator_finalization().await {
                eprintln!("Error in Moderator finalization: {}", e);
                return;
            }

            // Dump all memory state
            panel.dump_memory_state().await;

            println!("\n✅ WORKFLOW COMPLETE");
            println!("All outputs stored in Memory KV store under namespaces:");
            println!("   r1/* — Round 1 independent analyses");
            println!("   r2/* — Round 2 refined estimates with bounds (.v2 suffix)");
            println!("   final/* — Final report and summary");
            println!("   meta/* — Canonical version pointers");
            println!("   feedback/* — Moderator feedback");
        }
        Err(e) => {
            eprintln!("Failed to initialize panel workflow: {}", e);
        }
    }
}
