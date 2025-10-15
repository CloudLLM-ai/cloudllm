# Multi-Agent Council Tutorial: A Practical Cookbook

## Introduction

This tutorial demonstrates how to build multi-agent AI systems using CloudLLM's Council framework. We'll progress from simple to complex collaboration patterns, solving increasingly difficult problems with teams of AI agents from different providers (OpenAI, Claude, Gemini, Grok).

**The Challenge**: Throughout this tutorial, we'll tackle a pressing scientific problem: **designing an optimal carbon capture and storage (CCS) strategy** to combat climate change. This is a real-world problem with known solutions, allowing us to evaluate how well our AI councils converge on optimal approaches.

## Prerequisites

```rust
use cloudllm::{
    council::{Agent, Council, CouncilMode},
    clients::{
        openai::OpenAIClient,
        claude::ClaudeClient,
        gemini::GeminiClient,
        grok::GrokClient,
    },
};
use std::sync::Arc;

// Set your API keys
let openai_key = std::env::var("OPENAI_KEY").expect("OPENAI_KEY not set");
let anthropic_key = std::env::var("ANTHROPIC_KEY").expect("ANTHROPIC_KEY not set");
let gemini_key = std::env::var("GEMINI_KEY").expect("GEMINI_KEY not set");
let xai_key = std::env::var("XAI_KEY").expect("XAI_KEY not set");
```

---

## Recipe 1: Parallel Mode - Independent Expert Analysis

**Use Case**: When you need multiple independent perspectives on a problem without agents influencing each other.

**Problem**: Evaluate the three main carbon capture technologies: Direct Air Capture (DAC), Point Source Capture, and Ocean-based capture.

### The Council

```rust
async fn parallel_carbon_capture_analysis() -> Result<(), Box<dyn std::error::Error>> {
    // Create specialized agents with different AI providers
    let agent_chemistry = Agent::new(
        "chemistry-expert",
        "Dr. Chen (Chemistry Specialist)",
        Arc::new(ClaudeClient::new_with_model_str(
            &anthropic_key,
            "claude-3-5-sonnet-20241022"
        ))
    )
    .with_expertise("Chemical engineering, carbon chemistry, catalysis")
    .with_personality("Analytical, detail-oriented, focuses on molecular-level processes");

    let agent_economics = Agent::new(
        "economics-expert",
        "Dr. Martinez (Environmental Economist)",
        Arc::new(OpenAIClient::new_with_model_string(
            &openai_key,
            "gpt-4o"
        ))
    )
    .with_expertise("Cost-benefit analysis, carbon markets, policy economics")
    .with_personality("Pragmatic, data-driven, focuses on scalability and ROI");

    let agent_engineering = Agent::new(
        "engineering-expert",
        "Dr. Patel (Process Engineer)",
        Arc::new(GeminiClient::new_with_model_string(
            &gemini_key,
            "gemini-1.5-pro"
        ))
    )
    .with_expertise("Industrial processes, energy efficiency, systems integration")
    .with_personality("Practical, systems-thinking, focuses on implementation challenges");

    // Build the council
    let mut council = Council::new(
        "carbon-capture-council",
        "Carbon Capture Technology Assessment Council"
    )
    .with_mode(CouncilMode::Parallel)
    .with_system_context(
        "You are part of an expert panel evaluating carbon capture technologies. \
         Provide your independent analysis based on your domain expertise."
    );

    council.add_agent(agent_chemistry)?;
    council.add_agent(agent_economics)?;
    council.add_agent(agent_engineering)?;

    // Execute parallel analysis
    let response = council.discuss(
        "Analyze the three main carbon capture technologies (DAC, Point Source, Ocean-based) \
         and identify the most promising approach for immediate deployment. Consider: \
         1) Technical maturity, 2) Cost per ton CO2, 3) Scalability, 4) Environmental impact.",
        1  // One round
    ).await?;

    // Review results
    println!("=== PARALLEL ANALYSIS RESULTS ===\n");
    for msg in &response.messages {
        if let Some(name) = &msg.agent_name {
            println!("--- {} ---", name);
            println!("{}\n", msg.content);
        }
    }

    println!("Total tokens used: {}", response.total_tokens_used);

    Ok(())
}
```

**Expected Outcome**: Three independent analyses that can be compared side-by-side. Each expert provides their perspective without being influenced by others. Chemistry expert focuses on capture efficiency, economist on cost-effectiveness, engineer on practical deployment challenges.

**Best For**:
- Initial problem exploration
- Diverse viewpoint gathering
- Avoiding groupthink
- Fast parallel processing

---

## Recipe 2: Round-Robin Mode - Sequential Deliberation

**Use Case**: When agents should build upon each other's insights in a structured sequence.

**Problem**: Design a comprehensive carbon capture deployment strategy, where each expert adds their layer of analysis.

### The Council

```rust
async fn round_robin_deployment_strategy() -> Result<(), Box<dyn std::error::Error>> {
    // Create a 4-agent council with specific sequencing
    let agent_scientist = Agent::new(
        "climate-scientist",
        "Dr. Thompson (Climate Scientist)",
        Arc::new(ClaudeClient::new_with_model_str(
            &anthropic_key,
            "claude-3-5-sonnet-20241022"
        ))
    )
    .with_expertise("Climate modeling, carbon cycles, atmospheric science")
    .with_personality("Evidence-based, urgent but measured, focuses on climate impact");

    let agent_engineer = Agent::new(
        "systems-engineer",
        "Dr. Kim (Systems Engineer)",
        Arc::new(GeminiClient::new_with_model_string(
            &gemini_key,
            "gemini-1.5-pro"
        ))
    )
    .with_expertise("Large-scale infrastructure, grid integration, logistics")
    .with_personality("Methodical, risk-aware, focuses on feasibility");

    let agent_economist = Agent::new(
        "policy-economist",
        "Dr. Rodriguez (Policy & Economics)",
        Arc::new(OpenAIClient::new_with_model_string(
            &openai_key,
            "gpt-4o"
        ))
    )
    .with_expertise("Carbon pricing, government incentives, international cooperation")
    .with_personality("Strategic, diplomatic, focuses on policy mechanisms");

    let agent_innovator = Agent::new(
        "tech-innovator",
        "Dr. Zhang (Innovation Strategist)",
        Arc::new(GrokClient::new_with_model_str(
            &xai_key,
            "grok-beta"
        ))
    )
    .with_expertise("Emerging technologies, R&D acceleration, moonshot thinking")
    .with_personality("Optimistic, forward-thinking, challenges assumptions");

    let mut council = Council::new(
        "deployment-council",
        "Carbon Capture Deployment Strategy Council"
    )
    .with_mode(CouncilMode::RoundRobin)
    .with_system_context(
        "You are collaboratively designing a global carbon capture deployment strategy. \
         Build upon previous experts' insights while adding your unique perspective."
    );

    // Order matters in Round-Robin!
    council.add_agent(agent_scientist)?;  // Sets the scientific foundation
    council.add_agent(agent_engineer)?;   // Adds engineering reality
    council.add_agent(agent_economist)?;  // Layers in policy/economics
    council.add_agent(agent_innovator)?;  // Challenges with innovation

    // Run 2 rounds - each agent speaks twice
    let response = council.discuss(
        "Design a 10-year global deployment strategy for carbon capture to remove \
         5 gigatons CO2/year by 2035. Address: \
         1) Technology selection and phasing, \
         2) Infrastructure requirements, \
         3) Financing mechanisms, \
         4) Innovation acceleration.",
        2
    ).await?;

    // Display sequential discussion
    println!("=== ROUND-ROBIN STRATEGY DEVELOPMENT ===\n");
    let mut current_round = 0;
    for msg in &response.messages {
        if let Some(round_str) = msg.metadata.get("round") {
            let round: usize = round_str.parse().unwrap_or(0);
            if round != current_round {
                current_round = round;
                println!("\n╔════════════════════════════════════╗");
                println!("║         ROUND {}                    ║", round + 1);
                println!("╚════════════════════════════════════╝\n");
            }
        }

        if let Some(name) = &msg.agent_name {
            println!(">>> {} <<<", name);
            println!("{}\n", msg.content);
        }
    }

    println!("Total tokens used: {}", response.total_tokens_used);

    Ok(())
}
```

**Expected Outcome**: A layered, comprehensive strategy where each expert builds on the previous insights. Round 1 establishes the foundation, Round 2 refines and integrates. You'll see how Dr. Kim references Dr. Thompson's climate urgency, how Dr. Rodriguez builds financing around Kim's infrastructure needs, and how Dr. Zhang proposes innovations to accelerate the timeline.

**Best For**:
- Building complex, layered solutions
- Ensuring all perspectives are heard in order
- Creating comprehensive strategies
- Educational content (seeing reasoning progression)

---

## Recipe 3: Moderated Mode - Expert Panel with Chair

**Use Case**: When you have a moderator who should intelligently route questions to the most qualified expert.

**Problem**: Answer technical questions about carbon capture implementation, with a moderator selecting the right expert for each query.

### The Council

```rust
async fn moderated_qa_session() -> Result<(), Box<dyn std::error::Error>> {
    // Create the moderator
    let moderator = Agent::new(
        "moderator",
        "Dr. Sarah Wilson (Panel Chair)",
        Arc::new(OpenAIClient::new_with_model_string(
            &openai_key,
            "gpt-4o"
        ))
    )
    .with_expertise("Carbon capture overview, interdisciplinary coordination")
    .with_personality("Diplomatic, organized, excellent at matching questions to expertise");

    // Create specialized experts
    let agent_chemical = Agent::new(
        "chemical-expert",
        "Dr. Liu (Chemical Processes)",
        Arc::new(ClaudeClient::new_with_model_str(
            &anthropic_key,
            "claude-3-5-sonnet-20241022"
        ))
    )
    .with_expertise("Amine solvents, sorbent materials, reaction kinetics")
    .with_personality("Highly technical, precise with chemistry details");

    let agent_energy = Agent::new(
        "energy-expert",
        "Dr. Okafor (Energy Systems)",
        Arc::new(GeminiClient::new_with_model_string(
            &gemini_key,
            "gemini-1.5-pro"
        ))
    )
    .with_expertise("Energy requirements, heat integration, renewable coupling")
    .with_personality("Quantitative, focuses on energy efficiency and sustainability");

    let agent_storage = Agent::new(
        "storage-expert",
        "Dr. Bjorn (Geological Storage)",
        Arc::new(GrokClient::new_with_model_str(
            &xai_key,
            "grok-beta"
        ))
    )
    .with_expertise("CO2 sequestration, reservoir characterization, long-term monitoring")
    .with_personality("Safety-focused, experienced with subsurface engineering");

    let agent_lifecycle = Agent::new(
        "lifecycle-expert",
        "Dr. Sharma (Lifecycle Assessment)",
        Arc::new(OpenAIClient::new_with_model_string(
            &openai_key,
            "gpt-4o-mini"
        ))
    )
    .with_expertise("Full lifecycle analysis, net carbon accounting, environmental impact")
    .with_personality("Holistic thinker, considers entire system impacts");

    let mut council = Council::new(
        "moderated-qa-council",
        "Carbon Capture Q&A Panel"
    )
    .with_mode(CouncilMode::Moderated {
        moderator_id: "moderator".to_string()
    })
    .with_system_context(
        "You are participating in a technical Q&A session about carbon capture technology."
    );

    council.add_agent(moderator)?;
    council.add_agent(agent_chemical)?;
    council.add_agent(agent_energy)?;
    council.add_agent(agent_storage)?;
    council.add_agent(agent_lifecycle)?;

    // Ask a complex question requiring expert knowledge
    let response = council.discuss(
        "We're considering a 1 MT/year direct air capture facility powered by geothermal energy \
         in Iceland, storing CO2 in basalt formations. What are the key technical challenges and \
         is the net carbon balance truly negative when accounting for construction and operations?",
        3  // Let moderator route to 3 different experts
    ).await?;

    println!("=== MODERATED EXPERT Q&A ===\n");
    for msg in &response.messages {
        if let Some(name) = &msg.agent_name {
            if let Some(moderator_id) = msg.metadata.get("moderator") {
                println!("[Selected by {}]", moderator_id);
            }
            println!(">>> {} <<<", name);
            println!("{}\n", msg.content);
        }
    }

    println!("Total tokens used: {}", response.total_tokens_used);

    Ok(())
}
```

**Expected Outcome**: The moderator intelligently routes the multi-part question to appropriate experts. Energy expert discusses geothermal coupling, storage expert analyzes basalt mineralization potential, and lifecycle expert provides the net carbon accounting. The moderator may ask follow-ups to specific experts.

**Best For**:
- Q&A sessions
- Dynamic problem routing
- Efficient use of specialized expertise
- Interactive consultations

---

## Recipe 4: Hierarchical Mode - Multi-Layer Problem Solving

**Use Case**: When you need worker-level analysis, supervisor synthesis, and executive decision-making.

**Problem**: Evaluate and select the optimal carbon capture technology portfolio for three different regions with different constraints.

### The Council

```rust
async fn hierarchical_technology_selection() -> Result<(), Box<dyn std::error::Error>> {
    // === LAYER 1: Regional Analysis Workers ===

    let worker_north_america = Agent::new(
        "worker-na",
        "Analysis Team: North America",
        Arc::new(OpenAIClient::new_with_model_string(&openai_key, "gpt-4o-mini"))
    )
    .with_expertise("North American energy infrastructure, policy landscape, industrial base")
    .with_personality("Detail-oriented, region-specific knowledge");

    let worker_europe = Agent::new(
        "worker-eu",
        "Analysis Team: Europe",
        Arc::new(ClaudeClient::new_with_model_str(&anthropic_key, "claude-3-haiku-20240307"))
    )
    .with_expertise("European carbon markets, renewable integration, environmental regulations")
    .with_personality("Compliance-focused, sustainability-driven");

    let worker_asia = Agent::new(
        "worker-asia",
        "Analysis Team: Asia-Pacific",
        Arc::new(GeminiClient::new_with_model_string(&gemini_key, "gemini-1.5-flash"))
    )
    .with_expertise("Rapid industrialization, coal infrastructure, emerging technology adoption")
    .with_personality("Growth-oriented, pragmatic about constraints");

    // === LAYER 2: Domain Supervisors ===

    let supervisor_tech = Agent::new(
        "supervisor-tech",
        "Technical Supervisor",
        Arc::new(ClaudeClient::new_with_model_str(&anthropic_key, "claude-3-5-sonnet-20241022"))
    )
    .with_expertise("Technology assessment, comparative analysis, technical feasibility")
    .with_personality("Synthesizes technical details, identifies patterns across regions");

    let supervisor_business = Agent::new(
        "supervisor-business",
        "Business Supervisor",
        Arc::new(GeminiClient::new_with_model_string(&gemini_key, "gemini-1.5-pro"))
    )
    .with_expertise("Investment analysis, market dynamics, commercial viability")
    .with_personality("ROI-focused, risk assessment, market opportunities");

    // === LAYER 3: Executive Decision Maker ===

    let executive = Agent::new(
        "executive",
        "Chief Strategy Officer",
        Arc::new(OpenAIClient::new_with_model_string(&openai_key, "gpt-4o"))
    )
    .with_expertise("Strategic planning, portfolio management, resource allocation")
    .with_personality("Decisive, balances multiple objectives, long-term vision");

    let mut council = Council::new(
        "hierarchical-council",
        "Global Carbon Capture Portfolio Selection"
    )
    .with_mode(CouncilMode::Hierarchical {
        layers: vec![
            // Layer 1: Regional workers (parallel)
            vec![
                "worker-na".to_string(),
                "worker-eu".to_string(),
                "worker-asia".to_string(),
            ],
            // Layer 2: Domain supervisors (parallel)
            vec![
                "supervisor-tech".to_string(),
                "supervisor-business".to_string(),
            ],
            // Layer 3: Executive (single decision maker)
            vec!["executive".to_string()],
        ],
    })
    .with_system_context(
        "You are part of a hierarchical decision-making process for global carbon capture deployment."
    );

    // Add all agents
    council.add_agent(worker_north_america)?;
    council.add_agent(worker_europe)?;
    council.add_agent(worker_asia)?;
    council.add_agent(supervisor_tech)?;
    council.add_agent(supervisor_business)?;
    council.add_agent(executive)?;

    let response = council.discuss(
        "Evaluate carbon capture technology options for deployment in: \
         1) North America (abundant natural gas, existing industrial CO2 sources), \
         2) Europe (strong renewables, carbon pricing, limited storage), \
         3) Asia-Pacific (coal-heavy, rapid growth, cost sensitivity). \
         \
         Recommend a technology portfolio for each region that maximizes CO2 removal \
         while minimizing cost and risk. Consider: Point Source Capture, Direct Air Capture, \
         and Bioenergy with CCS (BECCS).",
        1
    ).await?;

    println!("=== HIERARCHICAL DECISION PROCESS ===\n");

    for msg in &response.messages {
        if let Some(layer_str) = msg.metadata.get("layer") {
            let layer: usize = layer_str.parse().unwrap_or(0);
            let layer_name = match layer {
                0 => "LAYER 1: Regional Analysis",
                1 => "LAYER 2: Domain Supervision",
                2 => "LAYER 3: Executive Decision",
                _ => "Unknown Layer",
            };

            println!("\n╔═══════════════════════════════════╗");
            println!("║  {}  ║", layer_name);
            println!("╚═══════════════════════════════════╝\n");
        }

        if let Some(name) = &msg.agent_name {
            println!(">>> {} <<<", name);
            println!("{}\n", msg.content);
        }
    }

    println!("Total tokens used: {}", response.total_tokens_used);

    Ok(())
}
```

**Expected Outcome**:
- **Layer 1**: Three regional teams provide detailed analysis of constraints and opportunities
- **Layer 2**: Supervisors synthesize the regional inputs - tech supervisor evaluates feasibility across regions, business supervisor assesses commercial viability
- **Layer 3**: Executive makes final portfolio allocation decision based on synthesized analysis

This mimics real organizational decision-making with clear information flow up the hierarchy.

**Best For**:
- Complex multi-region/multi-domain problems
- Organizational decision simulation
- Problems requiring both detail and synthesis
- Resource allocation decisions

---

## Recipe 5: Debate Mode - Adversarial Refinement with Convergence

**Use Case**: When you need agents to challenge each other's assumptions and converge on the most robust solution through argumentation.

**Problem**: Determine the optimal carbon price needed to make carbon capture economically viable. This is contentious with no single answer - perfect for debate.

### The Council

```rust
async fn debate_carbon_pricing() -> Result<(), Box<dyn std::error::Error>> {
    // Create agents with genuinely different perspectives

    let agent_market_optimist = Agent::new(
        "market-optimist",
        "Dr. Chen (Market Optimist)",
        Arc::new(OpenAIClient::new_with_model_string(&openai_key, "gpt-4o"))
    )
    .with_expertise("Market mechanisms, technological learning curves, innovation economics")
    .with_personality(
        "Believes in market efficiency and technology cost reductions. \
         Argues for moderate carbon prices with strong R&D support. \
         Optimistic about breakthrough technologies."
    );

    let agent_climate_hawk = Agent::new(
        "climate-hawk",
        "Dr. Andersson (Climate Emergency Advocate)",
        Arc::new(ClaudeClient::new_with_model_str(
            &anthropic_key,
            "claude-3-5-sonnet-20241022"
        ))
    )
    .with_expertise("Climate science, tipping points, urgency of action")
    .with_personality(
        "Emphasizes the urgency of climate crisis and social cost of carbon. \
         Advocates for high carbon prices to reflect true environmental cost. \
         Focuses on moral imperative and intergenerational justice."
    );

    let agent_pragmatist = Agent::new(
        "pragmatist",
        "Dr. Patel (Economic Pragmatist)",
        Arc::new(GeminiClient::new_with_model_string(&gemini_key, "gemini-1.5-pro"))
    )
    .with_expertise("Development economics, political feasibility, transition planning")
    .with_personality(
         "Balances climate urgency with economic reality and political feasibility. \
         Advocates for gradual, predictable carbon price escalation. \
         Concerned about economic disruption and public acceptance."
    );

    let agent_industry_realist = Agent::new(
        "industry-realist",
        "Dr. Mueller (Industrial Engineer)",
        Arc::new(GrokClient::new_with_model_str(&xai_key, "grok-beta"))
    )
    .with_expertise("Industrial processes, capital investment cycles, competitiveness")
    .with_personality(
        "Represents industry perspective and capital constraints. \
         Argues for carbon prices aligned with technology readiness and investment cycles. \
         Warns against policies that cause carbon leakage or economic damage."
    );

    let agent_systems_thinker = Agent::new(
        "systems-thinker",
        "Dr. Okonkwo (Systems Analyst)",
        Arc::new(OpenAIClient::new_with_model_string(&openai_key, "gpt-4o"))
    )
    .with_expertise("Systems dynamics, policy modeling, feedback loops")
    .with_personality(
        "Analyzes feedback loops and system effects. \
         Seeks carbon price that optimizes multiple objectives simultaneously. \
         Challenges simplistic arguments from all sides."
    );

    let mut council = Council::new(
        "debate-council",
        "Carbon Pricing Debate Council"
    )
    .with_mode(CouncilMode::Debate {
        max_rounds: 5,
        convergence_threshold: Some(0.65),  // 65% similarity triggers convergence
    })
    .with_system_context(
        "You are participating in a rigorous debate on carbon pricing policy. \
         Challenge weak arguments, acknowledge strong points, and refine your position \
         based on evidence presented by others. Seek truth through dialectic."
    );

    council.add_agent(agent_market_optimist)?;
    council.add_agent(agent_climate_hawk)?;
    council.add_agent(agent_pragmatist)?;
    council.add_agent(agent_industry_realist)?;
    council.add_agent(agent_systems_thinker)?;

    let response = council.discuss(
        "What carbon price ($/ton CO2) should be implemented globally to make carbon capture \
         and storage economically competitive while being politically and economically feasible? \
         \
         Consider: \
         - Current CCS costs ($50-150/ton depending on technology) \
         - Social cost of carbon ($50-200/ton depending on discount rate) \
         - Political feasibility and public acceptance \
         - Impact on industrial competitiveness \
         - Technology learning curves and R&D incentives \
         - Timeline for climate targets (net-zero by 2050) \
         \
         Justify your position with evidence and respond to counterarguments.",
        5
    ).await?;

    println!("=== CARBON PRICING DEBATE ===\n");

    let mut current_round = 0;
    for msg in &response.messages {
        if let Some(round_str) = msg.metadata.get("round") {
            let round: usize = round_str.parse().unwrap_or(0);
            if round != current_round {
                current_round = round;
                println!("\n");
                println!("╔═══════════════════════════════════════════════════╗");
                println!("║              DEBATE ROUND {}                       ║", round + 1);
                println!("╚═══════════════════════════════════════════════════╝");
                println!();
            }
        }

        if let Some(name) = &msg.agent_name {
            println!("┌─ {} ─┐", name);
            println!("{}", msg.content);
            println!("└─────────────────────────────────────────────────────┘\n");
        }
    }

    println!("\n=== DEBATE OUTCOME ===");
    println!("Rounds completed: {}", response.round);
    println!("Converged: {}", response.is_complete);
    if let Some(score) = response.convergence_score {
        println!("Convergence score: {:.2}%", score * 100.0);
    }
    println!("Total tokens used: {}", response.total_tokens_used);

    Ok(())
}
```

**Expected Outcome**:
- **Round 1**: Agents stake out initial positions ranging from $50-$200/ton
- **Round 2-3**: Agents challenge each other's assumptions. Climate hawk criticizes optimist's timeline, pragmatist questions hawk's political feasibility, industry realist highlights competitiveness concerns
- **Round 4-5**: Positions begin to converge as agents acknowledge valid points. May settle around $80-120/ton with gradual escalation
- **Convergence**: Debate terminates early if agents reach >65% similarity in their arguments

The debate mode is powerful because it surfaces and resolves conflicts through argumentation rather than averaging.

**Best For**:
- Contested decisions with no clear answer
- Exploring tradeoff spaces
- Stress-testing assumptions
- Finding robust consensus through dialectic

---

## Advanced: Combining Tools with Agents

All council modes support tool-augmented agents. Here's an example with real calculations:

```rust
use cloudllm::{
    tool_adapters::CustomToolAdapter,
    tool_protocol::{ToolRegistry, ToolMetadata, ToolParameter, ToolParameterType, ToolResult},
};

async fn council_with_tools() -> Result<(), Box<dyn std::error::Error>> {
    // Create a calculator tool for carbon accounting
    let mut adapter = CustomToolAdapter::new();

    adapter.register_tool(
        ToolMetadata::new("calculate_ccs_cost", "Calculate total cost of CCS deployment")
            .with_parameter(
                ToolParameter::new("capacity_mt_per_year", ToolParameterType::Number)
                    .with_description("Capture capacity in megatons CO2 per year")
                    .required()
            )
            .with_parameter(
                ToolParameter::new("cost_per_ton", ToolParameterType::Number)
                    .with_description("Cost per ton of CO2 captured")
                    .required()
            )
            .with_parameter(
                ToolParameter::new("years", ToolParameterType::Number)
                    .with_description("Number of years of operation")
                    .required()
            ),
        Arc::new(|params| {
            let capacity = params["capacity_mt_per_year"].as_f64().unwrap_or(0.0);
            let cost_per_ton = params["cost_per_ton"].as_f64().unwrap_or(0.0);
            let years = params["years"].as_f64().unwrap_or(0.0);

            let total_co2 = capacity * years;
            let total_cost = total_co2 * cost_per_ton * 1_000_000.0; // MT to tons
            let annual_cost = total_cost / years;

            Ok(ToolResult::success(serde_json::json!({
                "total_co2_removed_tons": total_co2 * 1_000_000.0,
                "total_cost_usd": total_cost,
                "annual_cost_usd": annual_cost,
                "cost_per_ton_usd": cost_per_ton
            })))
        })
    ).await;

    let registry = Arc::new(ToolRegistry::new(Arc::new(adapter)));

    // Create agent with tools
    let agent_analyst = Agent::new(
        "analyst",
        "Carbon Economics Analyst",
        Arc::new(OpenAIClient::new_with_model_string(&openai_key, "gpt-4o"))
    )
    .with_expertise("Carbon economics, cost analysis, financial modeling")
    .with_tools(registry);

    let mut council = Council::new("analysis-council", "CCS Cost Analysis")
        .with_mode(CouncilMode::Parallel);

    council.add_agent(agent_analyst)?;

    let response = council.discuss(
        "Calculate the total cost of deploying 5 MT/year carbon capture capacity \
         over 20 years at $100/ton. Use the calculate_ccs_cost tool.",
        1
    ).await?;

    println!("=== TOOL-AUGMENTED ANALYSIS ===\n");
    for msg in &response.messages {
        if let Some(name) = &msg.agent_name {
            println!("--- {} ---", name);
            println!("{}\n", msg.content);
        }
    }

    Ok(())
}
```

The agent will:
1. See the tool is available in its system prompt
2. Respond with: `{"tool_call": {"name": "calculate_ccs_cost", "parameters": {"capacity_mt_per_year": 5, "cost_per_ton": 100, "years": 20}}}`
3. Tool executes automatically
4. Agent receives result and formulates final response

---

## Best Practices

### 1. **Choosing the Right Mode**

| Mode | Use When | Avoid When |
|------|----------|-----------|
| **Parallel** | Need independent viewpoints, speed critical | Agents should build on each other |
| **RoundRobin** | Building layered solutions, clear expertise order | Need debate or routing |
| **Moderated** | Dynamic Q&A, varied question types | All questions suit same expert |
| **Hierarchical** | Complex multi-level problems, org simulation | Flat problem structure |
| **Debate** | Contested decisions, need robust consensus | Clear optimal solution exists |

### 2. **Agent Design Tips**

- **Expertise**: Be specific and actionable ("chemical kinetics of amine solvents" not "chemistry")
- **Personality**: Give distinct perspectives, not just different knowledge
- **Provider diversity**: Mix Claude (analytical), GPT-4 (balanced), Gemini (creative), Grok (contrarian)
- **Agent names**: Use realistic names and titles for better role-playing

### 3. **Prompt Engineering for Councils**

Good council prompts:
- ✅ Are specific and measurable
- ✅ Require multiple perspectives
- ✅ Have constrained solution spaces
- ✅ Provide context and constraints

Poor council prompts:
- ❌ Are too open-ended
- ❌ Can be answered by one expert
- ❌ Lack success criteria
- ❌ Are purely factual lookups

### 4. **Token Management**

```rust
// Monitor token usage
let response = council.discuss(prompt, rounds).await?;
println!("Tokens used: {}", response.total_tokens_used);

// For expensive debates, limit rounds
CouncilMode::Debate {
    max_rounds: 3,  // Lower for cost control
    convergence_threshold: Some(0.70)  // Higher = earlier convergence = lower cost
}
```

### 5. **Convergence Tuning**

The Jaccard similarity threshold controls debate termination:
- **0.50-0.60**: Very different positions can converge (loose consensus)
- **0.65-0.75**: Moderate agreement needed (recommended)
- **0.80-0.90**: Strong agreement needed (strict consensus)
- **0.95+**: Near-identical responses (potentially groupthink)

---

## Complete Example: Full Pipeline

Here's a complete program combining multiple modes:

```rust
use cloudllm::{
    council::{Agent, Council, CouncilMode},
    clients::{openai::OpenAIClient, claude::ClaudeClient, gemini::GeminiClient, grok::GrokClient},
};
use std::sync::Arc;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load API keys
    let openai_key = std::env::var("OPENAI_KEY")?;
    let anthropic_key = std::env::var("ANTHROPIC_KEY")?;
    let gemini_key = std::env::var("GEMINI_KEY")?;
    let xai_key = std::env::var("XAI_KEY")?;

    println!("╔════════════════════════════════════════════════════════╗");
    println!("║  Carbon Capture Strategy: Multi-Mode Council Pipeline  ║");
    println!("╚════════════════════════════════════════════════════════╝\n");

    // STAGE 1: Parallel analysis of technologies
    println!("📊 STAGE 1: Independent Technology Assessment (Parallel Mode)\n");

    let mut stage1_council = Council::new("stage1", "Tech Assessment")
        .with_mode(CouncilMode::Parallel);

    stage1_council.add_agent(Agent::new(
        "tech1", "Technology Analyst A",
        Arc::new(ClaudeClient::new_with_model_str(&anthropic_key, "claude-3-5-sonnet-20241022"))
    ).with_expertise("Direct Air Capture"))?;

    stage1_council.add_agent(Agent::new(
        "tech2", "Technology Analyst B",
        Arc::new(OpenAIClient::new_with_model_string(&openai_key, "gpt-4o"))
    ).with_expertise("Point Source Capture"))?;

    let stage1_result = stage1_council.discuss(
        "Evaluate your assigned carbon capture technology. Provide: \
         1) Readiness level (TRL 1-9), 2) Current cost, 3) Key challenges.",
        1
    ).await?;

    for msg in &stage1_result.messages {
        if let Some(name) = &msg.agent_name {
            println!("✓ {}: {}\n", name, msg.content.chars().take(150).collect::<String>());
        }
    }

    // STAGE 2: Debate to select optimal approach
    println!("\n💬 STAGE 2: Technology Selection Debate (Debate Mode)\n");

    let mut stage2_council = Council::new("stage2", "Selection Debate")
        .with_mode(CouncilMode::Debate { max_rounds: 3, convergence_threshold: Some(0.70) });

    stage2_council.add_agent(Agent::new(
        "advocate1", "DAC Advocate",
        Arc::new(GeminiClient::new_with_model_string(&gemini_key, "gemini-1.5-pro"))
    ))?;

    stage2_council.add_agent(Agent::new(
        "advocate2", "Point Source Advocate",
        Arc::new(GrokClient::new_with_model_str(&xai_key, "grok-beta"))
    ))?;

    let stage2_result = stage2_council.discuss(
        "Based on the stage 1 assessment, argue for your preferred technology. \
         Consider cost, scalability, and timeline to 2035.",
        3
    ).await?;

    println!("Debate completed in {} rounds", stage2_result.round);
    if let Some(score) = stage2_result.convergence_score {
        println!("Convergence: {:.1}%\n", score * 100.0);
    }

    // STAGE 3: Hierarchical deployment planning
    println!("🏗️  STAGE 3: Deployment Strategy (Hierarchical Mode)\n");

    let mut stage3_council = Council::new("stage3", "Deployment Planning")
        .with_mode(CouncilMode::Hierarchical {
            layers: vec![
                vec!["regional1".to_string(), "regional2".to_string()],
                vec!["executive".to_string()],
            ],
        });

    stage3_council.add_agent(Agent::new(
        "regional1", "Regional Planner US",
        Arc::new(OpenAIClient::new_with_model_string(&openai_key, "gpt-4o-mini"))
    ))?;

    stage3_council.add_agent(Agent::new(
        "regional2", "Regional Planner EU",
        Arc::new(ClaudeClient::new_with_model_str(&anthropic_key, "claude-3-haiku-20240307"))
    ))?;

    stage3_council.add_agent(Agent::new(
        "executive", "Strategy Director",
        Arc::new(OpenAIClient::new_with_model_string(&openai_key, "gpt-4o"))
    ))?;

    let stage3_result = stage3_council.discuss(
        "Create a 5-year deployment roadmap for the selected technology \
         in US and EU markets. Executives synthesize into unified strategy.",
        1
    ).await?;

    println!("✓ Deployment strategy completed\n");

    // FINAL SUMMARY
    println!("╔════════════════════════════════════════╗");
    println!("║           PIPELINE COMPLETE             ║");
    println!("╚════════════════════════════════════════╝\n");

    let total_tokens = stage1_result.total_tokens_used
        + stage2_result.total_tokens_used
        + stage3_result.total_tokens_used;

    println!("Total tokens used: {}", total_tokens);
    println!("Estimated cost (GPT-4o): ${:.2}", (total_tokens as f64) * 0.00001);

    Ok(())
}
```

---

## Troubleshooting

### Problem: Agents give similar responses
**Solution**: Increase personality/perspective differences, use different model providers, add expertise specificity

### Problem: Debate doesn't converge
**Solution**: Lower convergence threshold, increase max_rounds, or ensure agents have common ground

### Problem: High token costs
**Solution**: Use smaller models for workers, limit debate rounds, use parallel instead of round-robin for independent tasks

### Problem: Agents ignore each other in Round-Robin
**Solution**: Strengthen system context to emphasize building on previous responses, increase rounds

---

## Conclusion

You now have five powerful patterns for multi-agent collaboration:

1. **Parallel**: Fast, independent analysis
2. **RoundRobin**: Sequential, layered deliberation
3. **Moderated**: Dynamic expert routing
4. **Hierarchical**: Multi-level organizational decision-making
5. **Debate**: Adversarial convergence on robust solutions

Each mode excels in different scenarios. The carbon capture examples demonstrate how these patterns can tackle complex, real-world problems requiring multiple perspectives and deep domain expertise.

**Next Steps**:
- Experiment with agent personality variations
- Add custom tools for domain-specific calculations
- Combine modes in multi-stage pipelines
- Try other pressing problems: pandemic response, space mission planning, economic policy, etc.

Happy orchestrating! 🤖🤝🤖
