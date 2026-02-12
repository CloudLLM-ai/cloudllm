# Multi-Agent Orchestration Tutorial: A Practical Cookbook

## Introduction

This tutorial demonstrates how to build multi-agent AI systems using CloudLLM's Orchestration framework. We'll progress through six collaboration patterns from simple to complex, with a focus on understanding **costs, runtime expectations, and real-world tradeoffs**.

**⚠️ Cost & Runtime Warning**: This tutorial emphasizes cost implications because multi-agent orchestrations can run up bills quickly. We provide concrete examples with token estimates and timing for each mode.

---

## Quick Reference: Modes by Complexity & Cost

| Mode | Complexity | Est. Runtime | Est. Cost (4 agents) | Best For | ⚠️ Cost Risk |
|------|-----------|--------------|---------------------|----------|-------------|
| **AnthropicAgentTeams** | ★★★★★ | 2-5 min | $0.30-$1.00 | Large task pools | HIGH if max_iterations too high |
| **RALPH** | ★★★☆☆ | 3-20 min | $0.40-$9.00 | Checklist completion | MEDIUM (controlled iterations) |
| **Debate** | ★★★★☆ | 5-15 min | $0.60-$2.00 | Consensus building | **VERY HIGH** (exponential with rounds) |
| **Parallel** | ★☆☆☆☆ | 10-20 sec | $0.10-$0.30 | Independent opinions | LOW |
| **RoundRobin** | ★★☆☆☆ | 20-60 sec | $0.15-$0.50 | Sequential refinement | LOW-MEDIUM |
| **Moderated** | ★★★☆☆ | 30-90 sec | $0.20-$0.60 | Q&A sessions | MEDIUM |
| **Hierarchical** | ★★★★☆ | 1-3 min | $0.25-$0.80 | Multi-level problems | MEDIUM |

---

# MODE 1: AnthropicAgentTeams — Decentralized Task Coordination

## Overview

**AnthropicAgentTeams** is a **completely decentralized** orchestration mode where agents autonomously discover, claim, and complete tasks from a shared pool with **no central orchestrator**. This is the most powerful mode for large, complex projects but also the easiest to over-run and waste money.

**Key Insight**: Instead of the orchestration engine assigning tasks (like RALPH), agents use Memory to coordinate work peer-to-peer. This enables true autonomous multi-agent teams.

### ⚠️ COST WARNING

- **Per Iteration Cost**: ~$0.05-$0.15 per agent (4 agents = $0.20-$0.60/iteration)
- **Default Settings**: 4 iterations × 8 tasks = 16-32 LLM calls
- **Worst Case**: Setting `max_iterations: 100` with 4 agents = **3200 LLM calls** = **$1000+** in costs
- **How to Avoid**: Always cap `max_iterations` to ~2-3x your task count. For 8 tasks with 4 agents: use `max_iterations: 5` max.

### Runtime Expectations

- **Best case**: All tasks claimed and completed → ~2-3 minutes
- **Average case**: Agents work through pool → ~3-5 minutes
- **Worst case**: Poor task design, many retries → 10+ minutes

### Example: Research Team with NMN+ Study (8 Tasks)

```rust
use cloudllm::{
    Agent,
    orchestration::{Orchestration, OrchestrationMode, WorkItem},
    clients::openai::OpenAIClient,
    clients::claude::{ClaudeClient, Model},
    event::{EventHandler, OrchestrationEvent},
};
use async_trait::async_trait;
use std::sync::Arc;

/// Event handler for cost monitoring
struct CostTracker {
    iteration: std::sync::atomic::AtomicUsize,
}

#[async_trait]
impl EventHandler for CostTracker {
    async fn on_orchestration_event(&self, event: &OrchestrationEvent) {
        match event {
            OrchestrationEvent::RoundStarted { round, .. } => {
                println!("📍 Iteration {} starting...", round);
            }
            OrchestrationEvent::TaskClaimed {
                agent_name,
                task_id,
                ..
            } => {
                println!("  ✋ {} claimed: {}", agent_name, task_id);
            }
            OrchestrationEvent::TaskCompleted {
                agent_name,
                task_id,
                ..
            } => {
                println!("  ✅ {} completed: {}", agent_name, task_id);
            }
            OrchestrationEvent::RoundCompleted { .. } => {
                println!("  Cost for this iteration: ~$0.30-$0.50");
            }
            _ => {}
        }
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Define task pool
    let tasks = vec![
        WorkItem::new(
            "research_nmn",
            "Research phase — NMN+ mechanisms",
            "Summarize NAD+ pathways, mitochondrial function, sirtuins in 2-3 paragraphs",
        ),
        WorkItem::new(
            "analyze_longevity",
            "Analysis phase — longevity mechanisms",
            "Extract 3-5 key aging reversal pathways; estimate lifespan impact",
        ),
        WorkItem::new(
            "research_alzheimers",
            "Research phase — Alzheimer's pathology",
            "Document amyloid-beta, tau tangles, neuroinflammation; summarize in 2 paragraphs",
        ),
        WorkItem::new(
            "analyze_neuroprotection",
            "Analysis phase — neuroprotective mechanisms",
            "Map how NAD+ restoration combats neurodegeneration (5+ specific mechanisms)",
        ),
        WorkItem::new(
            "memory_recovery",
            "Research phase — memory recovery evidence",
            "Find 3+ studies showing cognitive restoration in AD models; summarize findings",
        ),
        WorkItem::new(
            "clinical_integration",
            "Analysis phase — clinical feasibility",
            "Assess dosing, bioavailability, safety profile; recommend next clinical trial",
        ),
        WorkItem::new(
            "synthesis_report",
            "Writing phase — comprehensive synthesis",
            "Write 3-4 page executive report integrating all findings with clear conclusions",
        ),
        WorkItem::new(
            "final_review",
            "Quality review — peer review assessment",
            "Review report for accuracy, completeness, evidence quality; suggest improvements",
        ),
    ];

    println!("═══════════════════════════════════════════════════════");
    println!("   NMN+ Research Team — AnthropicAgentTeams Mode");
    println!("═══════════════════════════════════════════════════════\n");

    println!("⚠️  COST ESTIMATE:");
    println!("  - 8 tasks × 4 agents = max 32 LLM calls");
    println!("  - At $0.05-0.10/call = $1.60-$3.20 total");
    println!("  - Runtime: ~3-5 minutes\n");

    // Create agents with mixed providers
    let openai_key = std::env::var("OPENAI_API_KEY")?;
    let anthropic_key = std::env::var("ANTHROPIC_API_KEY")?;

    let researcher = Agent::new(
        "researcher",
        "Research Agent (GPT-4o-mini)",
        Arc::new(OpenAIClient::new_with_model_string(&openai_key, "gpt-4o-mini")),
    );

    let analyst = Agent::new(
        "analyst",
        "Analysis Agent (Claude Haiku 4.5)",
        Arc::new(ClaudeClient::new_with_model_enum(&anthropic_key, Model::ClaudeHaiku45)),
    );

    let writer = Agent::new(
        "writer",
        "Writing Agent (GPT-4o-mini)",
        Arc::new(OpenAIClient::new_with_model_string(&openai_key, "gpt-4o-mini")),
    );

    let reviewer = Agent::new(
        "reviewer",
        "Review Agent (Claude Haiku 4.5)",
        Arc::new(ClaudeClient::new_with_model_enum(&anthropic_key, Model::ClaudeHaiku45)),
    );

    // ⚠️ CRITICAL: max_iterations calculation
    // Formula: (task_count / agent_count) * 1.5, capped at 5
    // 8 tasks / 4 agents = 2 * 1.5 = 3, use 4 for safety
    let max_iterations = 4;  // DO NOT SET TO 100!

    let mut orchestration = Orchestration::new(
        "nmn-research-team",
        "NMN+ & Alzheimer's Research Team",
    )
    .with_mode(OrchestrationMode::AnthropicAgentTeams {
        pool_id: "nmn-study-2024".to_string(),
        tasks: tasks.clone(),
        max_iterations,
    })
    .with_system_context(
        "You are a specialized researcher in a coordinated team. \
         Autonomously claim tasks from the shared pool and complete them thoroughly. \
         Build on previous agents' work when relevant. Focus on scientific accuracy \
         and clear communication. When done, report completion.",
    )
    .with_max_tokens(4096)
    .with_event_handler(Arc::new(CostTracker {
        iteration: std::sync::atomic::AtomicUsize::new(0),
    }));

    orchestration.add_agent(researcher)?;
    orchestration.add_agent(analyst)?;
    orchestration.add_agent(writer)?;
    orchestration.add_agent(reviewer)?;

    // Run orchestration
    let prompt = "Prepare a comprehensive scientific report on NMN+ for longevity and \
                   Alzheimer's disease recovery, with specific focus on memory restoration. \
                   The team will autonomously work through the 8 research tasks.";

    println!("👥 Team Members:");
    println!("  1. Researcher (GPT) — finds and summarizes sources");
    println!("  2. Analyst (Claude Haiku) — synthesizes findings");
    println!("  3. Writer (GPT) — drafts comprehensive report");
    println!("  4. Reviewer (Claude Haiku) — ensures quality\n");

    println!("⏱️  Starting orchestration...");

    let start = std::time::Instant::now();
    let response = orchestration.run(prompt, 1).await?;
    let elapsed = start.elapsed();

    println!("\n✨ RESULTS:");
    println!("  ├─ Iterations completed: {}", response.round);
    println!("  ├─ Tasks completed: {:.0}%", response.convergence_score.unwrap_or(0.0) * 100.0);
    println!("  ├─ Total time: {:.1}s", elapsed.as_secs_f32());
    println!("  ├─ Total tokens: {}", response.total_tokens_used);
    println!("  └─ Estimated cost: ${:.2}", (response.total_tokens_used as f64) * 0.00001);

    // Print sample messages
    println!("\n📝 Sample outputs:");
    for (i, msg) in response.messages.iter().take(3).enumerate() {
        if let Some(name) = &msg.agent_name {
            let preview = if msg.content.len() > 200 {
                format!("{}...", &msg.content[..200])
            } else {
                msg.content.to_string()
            };
            println!("  {}. [{}]: {}", i + 1, name, preview);
        }
    }

    Ok(())
}
```

### Key Parameters to Tune

```rust
// ✅ GOOD: Controls cost effectively
max_iterations: 4,           // 8 tasks ÷ 4 agents × 1.5 buffer = ~4 iterations
with_max_tokens(4096),       // Prevents runaway responses

// ❌ BAD: Will waste money
max_iterations: 100,         // Could run for 30+ minutes, $50+ cost
max_iterations: 50,          // Excessive iterations for 8 tasks
with_max_tokens(32768),      // Allows 100KB responses per agent
```

### Best Practices for AnthropicAgentTeams

1. **Task Design**: Keep task IDs short (`research_nmn` not `research_phase_1_nanoparticle_nmn_mechanism`)
2. **Iteration Cap**: `max_iterations = ceil(task_count / agent_count) + 1`
3. **Agent Count**: 3-6 agents per 8-15 tasks (more agents = more parallelism but higher cost)
4. **Monitoring**: Use event handler to detect stuck agents (same task claimed repeatedly)
5. **Early Exit**: If convergence_score reaches 1.0 before max_iterations, orchestration stops automatically
6. **Starter Content + Read-Modify-Write**: For file-producing tasks (e.g., building an HTML game), seed a working starter to disk and Memory (`current_game_html` key) before `run()`. Instruct agents to READ from Memory, MODIFY, and WRITE back via a custom tool that saves to both disk and Memory. See `examples/breakout_game_agent_teams.rs`.

### ⚠️ When AnthropicAgentTeams Gets Expensive

These scenarios can waste $100+:

```rust
// ❌ TOO MANY ITERATIONS
max_iterations: 50,      // Even if tasks complete in 5, runs all 50
tasks: vec![...], // 20 tasks
                         // Result: 50 × 4 agents × 5-10 calls = 1000-2000 calls = $10-50

// ❌ AMBIGUOUS TASKS
WorkItem::new("task1", "Do research", "Complete the task"),  // Agents don't know what "done" is
                         // Result: Agents keep claiming same task, never marking complete

// ❌ TOO MANY AGENTS FOR TASK POOL
max_iterations: 20,
tasks: vec![3_items], // 3 tasks
                         // Result: 4 agents all working on same 3 tasks repeatedly

// ✅ CORRECT
max_iterations: 2,       // 3 tasks ÷ 4 agents + buffer = 2 iterations
tasks: vec![...],
with_max_tokens(4096),   // Reasonable response length
```

---

# MODE 2: RALPH — Iterative Checklist with Agent Turn-Taking

## Overview

**RALPH** (Requirements Addressing Progressive Lite Heuristic) is for problems that can be broken into a **fixed checklist** of tasks. Unlike AnthropicAgentTeams, the orchestration engine manages the task list and agents signal completion via response markers.

**Best For**: Step-by-step project completion where tasks are clearly sequential or grouped.

### ⚠️ COST WARNING

- **Per Iteration**: ~$0.05-$0.15 per agent
- **Typical Cost**: 3-5 iterations × 3-4 agents = $0.45-$2.00
- **Risk**: Setting too high max_iterations for simple tasks
- **How to Avoid**: Monitor completion markers in responses; stop if no progress for 2 iterations

### Runtime Expectations

- **Simple checklist (5 items, 3 agents)**: 2-3 minutes, $0.30-$0.60
- **Medium checklist (10 items, 4 agents)**: 10-20 minutes, $3-9
- **Complex checklist (15+ items)**: 30-80 minutes, $5-10+

### Example: Breakout Game Implementation (18 Tasks)

The full breakout game examples use a **starter HTML + read-modify-write** pattern:

1. **Seed a working starter**: Before orchestration starts, a ~4KB working breakout game skeleton (paddle, ball, bricks, game loop) is written to disk and stored in Memory under `current_game_html`.
2. **Read-Modify-Write loop**: Each agent reads the current HTML from Memory (`G current_game_html`), modifies it to implement their assigned feature, then writes the updated HTML back via the `write_game_file` tool (which persists to both disk and Memory).
3. **Post-run recovery**: After orchestration completes, the code checks Memory first for the latest HTML, falls back to message extraction, then to the starter on disk.

This ensures every agent builds incrementally on the team's cumulative work and there is always a playable game on disk.

```rust
use cloudllm::{
    Agent,
    orchestration::{Orchestration, OrchestrationMode, RalphTask},
    clients::claude::{ClaudeClient, Model},
};
use std::sync::Arc;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("═══════════════════════════════════════════════════════");
    println!("   Breakout Game Implementation — RALPH Mode");
    println!("═══════════════════════════════════════════════════════\n");

    println!("⚠️  COST ESTIMATE:");
    println!("  - 18 tasks × 4 agents × ~5 iterations = many LLM calls");
    println!("  - At $0.05-0.15/call = $3-$9 total");
    println!("  - Runtime: ~10-20 minutes\n");

    // Define task checklist (18 tasks covering core mechanics, audio, powerups, etc.)
    let tasks = vec![
        RalphTask::new("html_structure", "HTML Structure", "Canvas element and game container"),
        RalphTask::new("game_states", "Game States", "MENU, PLAYING, PAUSED, GAME_OVER, LEVEL_COMPLETE"),
        RalphTask::new("paddle_control", "Paddle Control", "Keyboard and touch controls for paddle"),
        RalphTask::new("ball_physics", "Ball Physics", "Movement, angle reflection, boundary collision"),
        RalphTask::new("brick_grid", "Brick Grid", "Multi-hit bricks (1-5 HP) with color coding"),
        RalphTask::new("collision", "Collision Detection", "Ball-paddle, ball-brick, ball-wall"),
        RalphTask::new("scoring", "Score System", "Points, lives, level progression"),
        RalphTask::new("audio_engine", "Audio Engine", "Web Audio API chiptune music and SFX"),
        RalphTask::new("powerup_system", "Powerup System", "8 powerup types: paddle, speed, lava, etc."),
        RalphTask::new("particle_effects", "Particle Effects", "Fire bursts, paddle jets, 1UP displays"),
        RalphTask::new("brick_patterns", "Brick Patterns", "10+ procedural patterns per level"),
        RalphTask::new("difficulty", "Difficulty Scaling", "Dynamic difficulty by level"),
        RalphTask::new("mobile_controls", "Mobile Controls", "Touch/swipe with responsive canvas"),
        // ... (18 tasks total — see examples/breakout_game_ralph.rs for full list)
    ];

    let anthropic_key = std::env::var("ANTHROPIC_API_KEY")?;
    let make_client = || Arc::new(ClaudeClient::new_with_model_enum(&anthropic_key, Model::ClaudeSonnet45));

    let architect = Agent::new("architect", "Game Architect", make_client());
    let programmer = Agent::new("programmer", "Implementation Specialist", make_client());
    let sound_dev = Agent::new("sound", "Sound Designer", make_client());
    let powerup_dev = Agent::new("powerup", "Powerup Engineer", make_client());

    // Seed starter HTML to disk and Memory before orchestration
    // (see breakout_game_ralph.rs for full starter HTML and Memory setup)

    let mut orchestration = Orchestration::new("breakout-game", "Atari Breakout Implementation")
        .with_mode(OrchestrationMode::Ralph {
            tasks: tasks.clone(),
            max_iterations: 10,  // ⚠️ Safety cap (18 tasks / 4 agents + buffer)
        })
        .with_system_context(
            "You are implementing an Atari Breakout game in a single HTML file. \
             WORKFLOW: 1) READ current_game_html from Memory, 2) MODIFY it to \
             implement your assigned task, 3) WRITE back via write_game_file. \
             Mark done with [TASK_COMPLETE:task_id]. NEVER start from scratch.",
        )
        .with_max_tokens(180_000);

    orchestration.add_agent(architect)?;
    orchestration.add_agent(programmer)?;
    orchestration.add_agent(sound_dev)?;
    orchestration.add_agent(powerup_dev)?;

    let response = orchestration.run("Build an Atari Breakout game", 1).await?;

    println!("Iterations: {}", response.round);
    println!("Progress: {:.0}%", response.convergence_score.unwrap_or(0.0) * 100.0);

    // Post-run: check Memory first for latest HTML, then messages, then starter on disk
    Ok(())
}
```

### RALPH vs. AnthropicAgentTeams: Decision Matrix

| Scenario | Use RALPH | Use AnthropicAgentTeams |
|----------|-----------|------------------------|
| < 8 tasks | ✅ Yes | ❌ No (overkill) |
| 8-20 tasks | ✅ Maybe | ✅ Yes (better) |
| 20+ tasks | ❌ No | ✅ Yes (scales better) |
| Tasks are sequential | ✅ Yes | ✅ Yes (but looser) |
| Need tight orchestration control | ✅ Yes | ❌ No |
| Want agent autonomy | ❌ No | ✅ Yes |
| Building a game/app | ✅ Yes | ✅ Yes (both work) |
| Research/analysis project | ❌ No | ✅ Yes |

---

# MODE 3: Debate — Consensus Through Adversarial Refinement

## Overview

**Debate** mode has agents argue positions and refine their stances based on counterarguments. Agents continue until they reach **convergence** (word-set similarity) or hit max_rounds.

**Best For**: Contested decisions, exploring tradeoff spaces, stress-testing assumptions.

### ⚠️ COST WARNING — THIS ONE IS EXPENSIVE

- **Per Round**: ~$0.10-$0.30 per agent (5 agents = $0.50-$1.50/round)
- **Typical Run**: 3-5 rounds = $1.50-$7.50
- **Worst Case**: 5 agents × 10 rounds = **$5-15** easily
- **Exponential Risk**: Each extra round doubles cost. Going from 3 to 5 rounds = +$1.50-$3.00
- **How to Avoid**: Start with `max_rounds: 3`, increase only if needed; set `convergence_threshold: 0.70` (looser = fewer rounds)

### Runtime Expectations

- **Fast debate (2-3 rounds)**: 3-5 minutes
- **Medium debate (4-5 rounds)**: 6-10 minutes
- **Long debate (6+ rounds)**: 12+ minutes, **$10+ cost**

### Example: Carbon Pricing Debate (5 Positions)

```rust
use cloudllm::{
    Agent,
    orchestration::{Orchestration, OrchestrationMode},
    clients::openai::OpenAIClient,
    clients::claude::{ClaudeClient, Model},
    clients::gemini::GeminiClient,
};
use std::sync::Arc;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("═══════════════════════════════════════════════════════");
    println!("   Carbon Pricing Debate — Debate Mode");
    println!("═══════════════════════════════════════════════════════\n");

    println!("⚠️  COST WARNING (THIS IS EXPENSIVE!):");
    println!("  - 5 agents × 3 rounds minimum = 15 LLM calls");
    println!("  - Per-call cost: $0.03-0.10");
    println!("  - Estimated total: $0.45-$1.50");
    println!("  - But if agents don't converge, can go to 5 rounds = $0.75-$2.50");
    println!("  - Worst case (no convergence, 10 rounds): $1.50-$5.00\n");

    println!("⏱️  ESTIMATED TIME: 4-10 minutes (watch the clock!)\n");

    // Create agents with distinct perspectives
    let openai_key = std::env::var("OPENAI_API_KEY")?;
    let anthropic_key = std::env::var("ANTHROPIC_API_KEY")?;
    let gemini_key = std::env::var("GEMINI_API_KEY")?;

    let optimist = Agent::new(
        "market-optimist",
        "Dr. Chen (Market Optimist)",
        Arc::new(OpenAIClient::new_with_model_string(&openai_key, "gpt-4o")),
    )
    .with_expertise("Market mechanisms, technology cost curves, innovation economics")
    .with_personality(
        "Believes technology curves will make carbon capture cost-effective. \
         Advocates low carbon price ($25-50/ton) with strong R&D support.",
    );

    let hawk = Agent::new(
        "climate-hawk",
        "Dr. Andersson (Climate Emergency Advocate)",
        Arc::new(ClaudeClient::new_with_model_enum(&anthropic_key, Model::ClaudeSonnet45)),
    )
    .with_expertise("Climate science, tipping points, social cost of carbon")
    .with_personality(
        "Emphasizes climate urgency and intergenerational justice. \
         Advocates high carbon price ($150-200/ton) to reflect true social cost.",
    );

    let pragmatist = Agent::new(
        "pragmatist",
        "Dr. Patel (Economic Pragmatist)",
        Arc::new(GeminiClient::new_with_model_string(&gemini_key, "gemini-1.5-pro")),
    )
    .with_expertise("Development economics, political feasibility, policy design")
    .with_personality(
        "Balances climate urgency with political reality. \
         Advocates moderate, escalating carbon price ($50-100/ton, rising $5/year).",
    );

    let industry = Agent::new(
        "industry-realist",
        "Dr. Mueller (Industrial Engineer)",
        Arc::new(OpenAIClient::new_with_model_string(&openai_key, "gpt-4o-mini")),
    )
    .with_expertise("Industrial capital investment, competitiveness, carbon leakage")
    .with_personality(
        "Represents industry constraints. Warns high prices cause carbon leakage. \
         Advocates $30-60/ton with competitiveness safeguards.",
    );

    let analyst = Agent::new(
        "systems-analyst",
        "Dr. Okonkwo (Systems Analyst)",
        Arc::new(ClaudeClient::new_with_model_enum(&anthropic_key, Model::ClaudeHaiku45)),
    )
    .with_expertise("Policy modeling, feedback loops, unintended consequences")
    .with_personality(
        "Analyzes second- and third-order effects. Seeks price that optimizes \
         multiple objectives: climate action, economic efficiency, equity.",
    );

    // Create orchestration
    let mut orchestration = Orchestration::new("carbon-pricing-debate", "Carbon Pricing Policy Debate")
        .with_mode(OrchestrationMode::Debate {
            max_rounds: 4,                      // ⚠️ CRITICAL: Cap at 4, not 10!
            convergence_threshold: Some(0.70), // Higher threshold = earlier convergence = lower cost
        })
        .with_system_context(
            "You are a policy expert in a rigorous debate. Argue your position with evidence. \
             Acknowledge valid points from others. Seek common ground where possible. \
             Aim for robust consensus, not groupthink.",
        )
        .with_max_tokens(6144);

    orchestration.add_agent(optimist)?;
    orchestration.add_agent(hawk)?;
    orchestration.add_agent(pragmatist)?;
    orchestration.add_agent(industry)?;
    orchestration.add_agent(analyst)?;

    let prompt = "What carbon price ($/ton CO2) should be implemented globally? \
                  Consider: CCS costs ($50-150/ton), social cost of carbon ($75-200/ton), \
                  political feasibility, industrial competitiveness, climate urgency.";

    println!("🎙️  Debate participants: 5 agents with distinct perspectives");
    println!("📊 Max rounds: 4 (prevents runaway costs)");
    println!("⏱️  Starting debate...\n");

    let start = std::time::Instant::now();
    let response = orchestration.run(prompt, 1).await?;
    let elapsed = start.elapsed();

    println!("\n✨ DEBATE RESULTS:");
    println!("  ├─ Rounds completed: {}", response.round);
    println!("  ├─ Converged: {}", response.is_complete);
    if let Some(score) = response.convergence_score {
        println!("  ├─ Convergence score: {:.1}%", score * 100.0);
    }
    println!("  ├─ Time: {:.1}s", elapsed.as_secs_f32());
    println!("  ├─ Tokens: {}", response.total_tokens_used);
    println!("  └─ Cost: ${:.2}", (response.total_tokens_used as f64) * 0.00002);

    println!("\n💡 Interpretation:");
    if response.is_complete {
        println!("  ✅ Agents converged to consensus position");
    } else {
        println!("  ⚠️  Max rounds reached without full convergence (diverse views remain)");
    }

    // Show final positions
    println!("\n📄 Final positions (last 2 messages):");
    for msg in response.messages.iter().rev().take(2) {
        if let Some(name) = &msg.agent_name {
            let preview = if msg.content.len() > 250 {
                format!("{}...", &msg.content[..250])
            } else {
                msg.content.clone()
            };
            println!("\n  [{}]: {}", name, preview);
        }
    }

    Ok(())
}
```

### Debate Convergence Tuning

**The convergence_threshold parameter controls cost directly:**

```rust
// ❌ COSTS $5+: Requires high agreement to stop
OrchestrationMode::Debate {
    max_rounds: 10,
    convergence_threshold: Some(0.95),  // Need 95% similarity = many rounds
}

// ✅ COSTS $1-2: Balanced
OrchestrationMode::Debate {
    max_rounds: 5,
    convergence_threshold: Some(0.70),  // 70% similar = stops sooner
}

// ✅ COSTS $0.50: Loose consensus
OrchestrationMode::Debate {
    max_rounds: 3,
    convergence_threshold: Some(0.60),  // 60% = stops very quickly
}
```

---

# MODE 4: Parallel — Independent Expert Analysis

## Overview

**Parallel** mode is the **cheapest and fastest** — all agents respond simultaneously to the same prompt, with no interaction.

**Best For**: Independent opinions, quick polls, parallel processing.

### Cost Profile

- **Cost**: $0.05-$0.15 per agent, regardless of rounds
- **Time**: 15-30 seconds for most responses
- **Example**: 4 agents, 1 round = $0.20-$0.60, 30 seconds

### Example

```rust
let mut orchestration = Orchestration::new("parallel-demo", "Parallel Analysis")
    .with_mode(OrchestrationMode::Parallel);

// Add agents...

let response = orchestration.run(
    "Analyze these three carbon capture technologies independently. \
     1) Direct Air Capture, 2) Point Source Capture, 3) Ocean-based capture",
    1
).await?;

println!("Completed in 30 seconds, cost $0.25");
```

---

# MODE 5: Round-Robin — Sequential Deliberation

## Overview

Each agent speaks in turn, building on previous agents' responses. Useful for brainstorming, iterative refinement, and getting sequential perspectives.

**Best For**: Creative collaboration, iterative problem-solving, building consensus gradually.

### Cost Profile

- **Cost**: $0.10-$0.40 per round (4 agents × 2 rounds = $0.20-$0.80)
- **Time**: 30-90 seconds per round

### Example

```rust
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let claude_key = std::env::var("ANTHROPIC_API_KEY")?;

    let analyst1 = Agent::new(
        "analyst1",
        "Data Analyst",
        Arc::new(ClaudeClient::new_with_model_enum(&claude_key, Model::ClaudeHaiku45)),
    );

    let analyst2 = Agent::new(
        "analyst2",
        "Business Strategist",
        Arc::new(ClaudeClient::new_with_model_enum(&claude_key, Model::ClaudeHaiku45)),
    );

    let analyst3 = Agent::new(
        "analyst3",
        "Risk Manager",
        Arc::new(ClaudeClient::new_with_model_enum(&claude_key, Model::ClaudeHaiku45)),
    );

    let mut orchestration = Orchestration::new("roundrobin-demo", "Market Analysis Round-Robin")
        .with_mode(OrchestrationMode::RoundRobin { max_rounds: 3 });

    orchestration.add_agent(analyst1)?;
    orchestration.add_agent(analyst2)?;
    orchestration.add_agent(analyst3)?;

    let response = orchestration.run(
        "Analyze the investment potential of electric vehicle manufacturers. \
         Analyst1: Present market data and trends. \
         Analyst2: Build on that with strategic insights. \
         Analyst3: Then address risks and mitigations.",
        1
    ).await?;

    println!("Round-Robin completed in {} rounds, {} tokens", response.round, response.total_tokens_used);

    Ok(())
}
```

---

# MODE 6: Moderated — Expert Routing

## Overview

A moderator agent receives the prompt and decides which experts to consult. Experts only respond when asked by the moderator, optimizing token usage.

**Best For**: Complex questions requiring selective expert consultation, reducing unnecessary API calls.

### Cost Profile

- **Cost**: $0.15-$0.60 per run (moderator + selected experts only)
- **Time**: 45-120 seconds
- **Best for**: Q&A sessions, dynamic problem routing

### Example

```rust
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let claude_key = std::env::var("ANTHROPIC_API_KEY")?;

    let moderator = Agent::new(
        "moderator",
        "Interview Moderator",
        Arc::new(ClaudeClient::new_with_model_enum(&claude_key, Model::ClaudeHaiku45)),
    )
    .with_expertise("Directing technical interviews and routing questions to specialists");

    let systems_expert = Agent::new(
        "systems_expert",
        "Systems Design Expert",
        Arc::new(ClaudeClient::new_with_model_enum(&claude_key, Model::ClaudeHaiku45)),
    )
    .with_expertise("Large-scale systems architecture, scalability, distributed systems");

    let algo_expert = Agent::new(
        "algo_expert",
        "Algorithms Expert",
        Arc::new(ClaudeClient::new_with_model_enum(&claude_key, Model::ClaudeHaiku45)),
    )
    .with_expertise("Algorithm design, time/space complexity, advanced data structures");

    let mut orchestration = Orchestration::new("moderated-demo", "Technical Interview")
        .with_mode(OrchestrationMode::Moderated {
            moderator_id: "moderator".to_string(),
            respondent_ids: vec!["systems_expert".to_string(), "algo_expert".to_string()],
        });

    orchestration.add_agent(moderator)?;
    orchestration.add_agent(systems_expert)?;
    orchestration.add_agent(algo_expert)?;

    let response = orchestration.run(
        "We're building a real-time recommendation system. \
         Question 1: How should we design the system architecture? \
         Question 2: What algorithms would optimize matching speed?",
        1
    ).await?;

    println!("Moderated run: {} tokens (only moderator + selected experts called)", response.total_tokens_used);

    Ok(())
}
```

---

# MODE 7: Hierarchical — Multi-Layer Decision Making

## Overview

Multi-layer processing: Workers generate initial analysis, Supervisors review and synthesize, Executives make final decisions. Each layer's output feeds into the next.

**Best For**: Complex organizational decisions, multi-stage refinement, hierarchical problem decomposition.

### Cost Profile

- **Cost**: $0.25-$0.80 per run
- **Time**: 1-3 minutes

### Example

```rust
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let claude_key = std::env::var("ANTHROPIC_API_KEY")?;

    // Layer 1: Workers (specialists gather information)
    let researcher1 = Agent::new(
        "researcher1",
        "Market Researcher",
        Arc::new(ClaudeClient::new_with_model_enum(&claude_key, Model::ClaudeHaiku45)),
    )
    .with_expertise("Market analysis, customer trends, competitive landscape");

    let researcher2 = Agent::new(
        "researcher2",
        "Technical Researcher",
        Arc::new(ClaudeClient::new_with_model_enum(&claude_key, Model::ClaudeHaiku45)),
    )
    .with_expertise("Technology feasibility, implementation challenges, engineering effort");

    // Layer 2: Supervisors (synthesize and prioritize)
    let product_lead = Agent::new(
        "product_lead",
        "Product Manager",
        Arc::new(ClaudeClient::new_with_model_enum(&claude_key, Model::ClaudeHaiku45)),
    )
    .with_expertise("Product strategy, feature prioritization, user impact");

    // Layer 3: Executive (final decision)
    let ceo = Agent::new(
        "ceo",
        "CEO",
        Arc::new(ClaudeClient::new_with_model_enum(&claude_key, Model::ClaudeHaiku45)),
    )
    .with_expertise("Business strategy, resource allocation, long-term vision");

    let mut orchestration = Orchestration::new("hierarchical-demo", "Product Feature Decision")
        .with_mode(OrchestrationMode::Hierarchical {
            layers: vec![
                vec!["researcher1".to_string(), "researcher2".to_string()],  // Layer 1: Workers
                vec!["product_lead".to_string()],                             // Layer 2: Supervisor
                vec!["ceo".to_string()],                                      // Layer 3: Executive
            ],
        });

    orchestration.add_agent(researcher1)?;
    orchestration.add_agent(researcher2)?;
    orchestration.add_agent(product_lead)?;
    orchestration.add_agent(ceo)?;

    let response = orchestration.run(
        "Should we invest in building an AI-powered personalization engine? \
         Workers: Analyze market demand, technical complexity, implementation timeline. \
         Product: Synthesize findings, prioritize requirements, estimate ROI. \
         CEO: Make final strategic decision with full context.",
        1
    ).await?;

    println!("Hierarchical decision: {} tokens over {} rounds", response.total_tokens_used, response.round);

    Ok(())
}
```

---

## Cost Comparison Summary

| Mode | 4 Agents, 1 Round | Notes |
|------|------------------|-------|
| Parallel | $0.20-$0.60 | Fastest, cheapest |
| RoundRobin | $0.30-$0.80 | 2-3 rounds recommended |
| Moderated | $0.25-$0.70 | Dynamic routing |
| Hierarchical | $0.35-$0.90 | Multi-layer synthesis |
| RALPH | $0.40-$1.20 | Per iteration |
| Debate | $0.50-$2.00 | ⚠️ Varies by convergence |
| AnthropicAgentTeams | $0.30-$1.00 | Per iteration |

---

## Avoiding Expensive Mistakes

### ❌ Mistake #1: Infinite Debate

```rust
// BAD: No cap on rounds
OrchestrationMode::Debate {
    max_rounds: 1000,  // Agents keep arguing, $50+ cost
    convergence_threshold: Some(0.99),  // Convergence never reached
}
```

**Fix**: Cap at 3-5 rounds, set convergence to 0.65-0.75

### ❌ Mistake #2: Too Many Iterations

```rust
// BAD: Excessive iterations for small task pool
max_iterations: 100,   // 100 × 4 agents = 400+ calls
tasks: vec![...],      // Only 5 tasks!
```

**Fix**: Use formula `ceil(task_count / agent_count) + buffer`

### ❌ Mistake #3: Oversized Token Budget

```rust
// BAD: Allows 100KB responses per agent
with_max_tokens(32768),  // 4 agents × 32K tokens = runaway costs
```

**Fix**: Use 4096-8192 for normal tasks

### ✅ Best Practice: Always Monitor

```rust
let response = orchestration.run(prompt, rounds).await?;

// Print cost before accepting results
let estimated_cost = (response.total_tokens_used as f64) * 0.00002;
println!("Cost: ${:.2}", estimated_cost);

if estimated_cost > 5.0 {
    eprintln!("⚠️  WARNING: High cost run. Review mode parameters.");
}
```

---

## Complete Multi-Mode Pipeline Example

```rust
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 Multi-Stage Orchestration Pipeline");
    println!("   Stage 1: Parallel analysis ($0.30)");
    println!("   Stage 2: Debate for selection ($1.50)");
    println!("   Stage 3: Hierarchical planning ($0.50)");
    println!("   Total estimate: $2.30\n");

    // STAGE 1: Parallel independent analysis
    let mut stage1 = Orchestration::new("stage1", "Tech Assessment")
        .with_mode(OrchestrationMode::Parallel);

    stage1.add_agent(Agent::new("tech1", "DAC Expert", ...))?;
    stage1.add_agent(Agent::new("tech2", "Point Source Expert", ...))?;

    let result1 = stage1.run("Evaluate your assigned technology", 1).await?;
    println!("Stage 1: ${:.2}", (result1.total_tokens_used as f64) * 0.00002);

    // STAGE 2: Debate to select winner
    let mut stage2 = Orchestration::new("stage2", "Technology Selection")
        .with_mode(OrchestrationMode::Debate {
            max_rounds: 3,
            convergence_threshold: Some(0.70),
        });

    stage2.add_agent(Agent::new("advocate1", "DAC Advocate", ...))?;
    stage2.add_agent(Agent::new("advocate2", "Point Source Advocate", ...))?;

    let result2 = stage2.run("Argue for your preferred technology", 1).await?;
    println!("Stage 2: ${:.2}", (result2.total_tokens_used as f64) * 0.00002);

    // STAGE 3: Hierarchical deployment planning
    let mut stage3 = Orchestration::new("stage3", "Deployment Planning")
        .with_mode(OrchestrationMode::Hierarchical {
            layers: vec![
                vec!["regional1", "regional2"],
                vec!["executive"],
            ],
        });

    // Add agents...

    let result3 = stage3.run("Create deployment strategy", 1).await?;
    println!("Stage 3: ${:.2}", (result3.total_tokens_used as f64) * 0.00002);

    let total = result1.total_tokens_used + result2.total_tokens_used + result3.total_tokens_used;
    println!("\nTotal tokens: {}", total);
    println!("Total cost: ${:.2}", (total as f64) * 0.00002);

    Ok(())
}
```

---

## Key Takeaways

1. **Parallel is cheapest** (~$0.30, 30 sec) — use when agents don't need to interact
2. **RALPH is predictable** (~$0.50-$1.00/iteration) — use for fixed checklists
3. **Debate is expensive** (~$1.50-$5.00) — always cap rounds and set convergence threshold
4. **AnthropicAgentTeams is powerful but risks** — cap `max_iterations` strictly
5. **Always monitor tokens** — $0.00002 per token means 50K tokens = $1, 100K tokens = $2
6. **Start conservative** — begin with low iteration counts, increase only if needed

Happy orchestrating! 🤖🤝🤖
