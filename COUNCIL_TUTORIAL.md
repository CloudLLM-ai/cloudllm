# Council Cookbook: Orchestrating Multi-LLM Councils for Scientific Impact

This cookbook-style tutorial shows how to combine multiple LLM providers inside a `CouncilSession` to tackle high-stakes scientific problems where benchmark answers are already established. Each recipe emphasizes a different council pattern—round-robin panels, moderator-steered reviews, and multi-round consensus building—while showcasing how to wire together the OpenAI, Claude, Gemini, and Grok client wrappers provided by this crate.

## Prerequisites

- **Rust toolchain** with async support (`tokio` runtime) and a project that depends on this crate.
- **API keys** exported as environment variables:
  - `OPENAI_API_KEY`
  - `ANTHROPIC_API_KEY`
  - `GEMINI_API_KEY`
  - `XAI_API_KEY`
- `cloudllm::init_logger()` is optional but recommended when you want HTTP diagnostics from the shared client pool.

> **Hint:** The council stores total token usage across rounds. This makes it convenient to keep scientific investigations within a defined research budget.

---

## Recipe 1 — Round-Robin Cross-Check of IPCC Mitigation Pathways

**Goal:** Validate that a proposed climate mitigation plan matches the IPCC AR6 1.5 °C pathway (net-zero CO₂ by 2050 and ~45 % reductions from 2010 baselines by 2030).

**Why it matters:** Cross-verifying the headline numbers of net-zero transition plans guards against optimistic modelling that could derail real-world decarbonization.

**Council design:**

- Moderator powered by OpenAI GPT-4o (coordinates the panel).
- Climate policy specialist using Claude Sonnet.
- Energy systems analyst using Gemini 2.0 Pro Experimental.

```rust
use std::error::Error;
use std::sync::Arc;

use cloudllm::client_wrapper::Role;
use cloudllm::clients::claude::{ClaudeClient, Model as ClaudeModel};
use cloudllm::clients::gemini::{GeminiClient, Model as GeminiModel};
use cloudllm::clients::openai::{OpenAIClient, Model as OpenAIModel};
use cloudllm::{CouncilRole, CouncilSession, ParticipantConfig};

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    cloudllm::init_logger();

    let openai_key = std::env::var("OPENAI_API_KEY")?;
    let anthropic_key = std::env::var("ANTHROPIC_API_KEY")?;
    let gemini_key = std::env::var("GEMINI_API_KEY")?;

    let mut council = CouncilSession::new(
        "You are reviewing mitigation strategies against the IPCC AR6 1.5 °C carbon budget. \
         Cite peer-reviewed sources, surface inconsistencies, and ensure all numbers are self-consistent.",
    );

    let moderator = council.add_participant_with_config(
        Arc::new(OpenAIClient::new_with_model_enum(&openai_key, OpenAIModel::GPT4o)),
        CouncilRole::Moderator,
        ParticipantConfig {
            display_name: Some("IPCC Rapporteur (GPT-4o)".into()),
            persona_prompt: Some("Coordinate the panel, ask for sourced figures, and summarize areas of agreement.".into()),
            max_tokens: Some(4096),
        },
    );

    let _policy = council.add_participant_with_config(
        Arc::new(ClaudeClient::new_with_model_enum(&anthropic_key, ClaudeModel::ClaudeSonnet4)),
        CouncilRole::Panelist,
        ParticipantConfig {
            display_name: Some("Climate Policy Analyst (Claude)".into()),
            persona_prompt: Some("Focus on nationally determined contributions, carbon pricing trajectories, and equity.".into()),
            max_tokens: Some(4096),
        },
    );

    let _energy = council.add_participant_with_config(
        Arc::new(GeminiClient::new_with_model_enum(&gemini_key, GeminiModel::Gemini20ProExp)),
        CouncilRole::Panelist,
        ParticipantConfig {
            display_name: Some("Energy Systems Modeler (Gemini)".into()),
            persona_prompt: Some("Validate electrification rates, negative-emission capacity, and renewable build-out assumptions.".into()),
            max_tokens: Some(4096),
        },
    );

    council.set_round_robin_order(vec![moderator])?;

    let first_round = council
        .send_message(
            Role::User,
            "Confirm that the proposed pathway meets the IPCC AR6 SR1.5 requirement of \
             a 45% global CO₂ reduction from 2010 levels by 2030 and net-zero by 2050. \
             Cross-check energy demand assumptions against the REMIND-MAgPIE SSP1-1.9 scenario.".to_string(),
            None,
        )
        .await?;

    println!("Round {}", first_round.round_index);
    for reply in first_round.replies {
        println!("{}\n{}\n", reply.name, reply.message.content);
        if let Some(usage) = reply.usage {
            println!(
                "Token usage → input: {} output: {} total: {}\n",
                usage.input_tokens, usage.output_tokens, usage.total_tokens
            );
        }
    }

    Ok(())
}
```

**What to expect:** The moderator should restate the canonical AR6 numbers (≥ 45 % cuts by 2030, net-zero by mid-century) and reconcile panel comments. Deviations signal where your mitigation plan diverges from the published pathway.

---

## Recipe 2 — Moderator-Steered Oversight for Vaccine Cold-Chain Scale-Up

**Goal:** Detect practical flaws in a national mRNA vaccine deployment plan that must maintain −70 °C cold-chain integrity through a 30-day distribution cycle.

**Known baseline:** WHO guidance documents show that maintaining −60 °C to −80 °C is mandatory for current mRNA vaccines, and dry-ice replenishment becomes critical after five days in transit.

**Council design:**

- Logistics moderator (OpenAI GPT-4o) sets the cadence.
- Grok agent focuses on real-time supply routing.
- Gemini agent checks thermodynamic feasibility.
- Claude agent reviews regulatory readiness.

```rust
use std::error::Error;
use std::sync::Arc;

use cloudllm::client_wrapper::Role;
use cloudllm::clients::grok::{GrokClient, Model as GrokModel};
use cloudllm::clients::gemini::{GeminiClient, Model as GeminiModel};
use cloudllm::clients::openai::{OpenAIClient, Model as OpenAIModel};
use cloudllm::clients::claude::{ClaudeClient, Model as ClaudeModel};
use cloudllm::{CouncilRole, CouncilSession, ParticipantConfig, ParticipantId};

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let openai_key = std::env::var("OPENAI_API_KEY")?;
    let xai_key = std::env::var("XAI_API_KEY")?;
    let gemini_key = std::env::var("GEMINI_API_KEY")?;
    let anthropic_key = std::env::var("ANTHROPIC_API_KEY")?;

    let mut council = CouncilSession::new(
        "You are auditing an mRNA vaccine program. Flag any plan that violates WHO cold-chain tolerances.",
    );

    let moderator = council.add_participant_with_config(
        Arc::new(OpenAIClient::new_with_model_enum(&openai_key, OpenAIModel::GPT4o)),
        CouncilRole::Moderator,
        ParticipantConfig {
            display_name: Some("Distribution Moderator".into()),
            persona_prompt: Some("Enforce logistics checklists and request corrective actions.".into()),
            max_tokens: Some(3072),
        },
    );

    let grok = council.add_participant_with_config(
        Arc::new(GrokClient::new_with_model_enum(&xai_key, GrokModel::Grok21212)),
        CouncilRole::Panelist,
        ParticipantConfig {
            display_name: Some("Routing Optimizer (Grok)".into()),
            persona_prompt: Some("Stress-test transport legs, border crossings, and fleet capacity.".into()),
            max_tokens: Some(2048),
        },
    );

    let gemini = council.add_participant_with_config(
        Arc::new(GeminiClient::new_with_model_enum(&gemini_key, GeminiModel::Gemini20FlashThinking001)),
        CouncilRole::Panelist,
        ParticipantConfig {
            display_name: Some("Thermal Engineer (Gemini)".into()),
            persona_prompt: Some("Verify holdover times, phase-change materials, and dry-ice resupply windows.".into()),
            max_tokens: Some(2048),
        },
    );

    let claude = council.add_participant_with_config(
        Arc::new(ClaudeClient::new_with_model_enum(&anthropic_key, ClaudeModel::ClaudeHaiku35)),
        CouncilRole::Observer,
        ParticipantConfig {
            display_name: Some("Regulatory Compliance (Claude)".into()),
            persona_prompt: Some("Only speak when the moderator invites you to evaluate adverse-event monitoring or pharmacovigilance.".into()),
            max_tokens: Some(1024),
        },
    );

    council.set_round_robin_order(vec![moderator, gemini, grok, claude])?;

    let assessment = council
        .send_message(
            Role::User,
            "Audit this plan: maintain −70 °C stability for 12 million doses over 30 days. \
             Dry-ice replenishment is scheduled every seven days, and refrigerated last-mile trucks operate at +4 °C.".to_string(),
            None,
        )
        .await?;

    for reply in assessment.replies {
        println!("{}\n{}\n", reply.name, reply.message.content);
    }

    Ok(())
}
```

**Interpretation tip:** Expect the Gemini thermal engineer to flag that seven-day dry-ice cycles breach the validated five-day replenishment window. The moderator should force corrective steps; if not, you know where to refine prompts or escalation logic.

---

## Recipe 3 — Multi-Round Consensus on Malaria Elimination Coverage Targets

**Goal:** Drive the council to converge on the long-standing WHO recommendation that ≥ 80 % insecticide-treated net (ITN) coverage is required to push malaria’s basic reproduction number (R₀) below 1 in high-transmission settings.

**Why it matters:** Rapid consensus on intervention thresholds accelerates program design when case numbers spike after vector resistance or humanitarian disruptions.

**Council design:**

- Claude Sonnet acts as an epidemiological modeller.
- OpenAI O1 evaluates R₀ sensitivity.
- Gemini 2.5 Flash surfaces implementation pitfalls.
- Grok 3 Mini synthesizes a take-home action memo.

The council runs two rounds: the first to present analyses, the second to reconcile numbers and agree on the ≥ 80 % coverage target that has been validated in WHO’s 2023 World Malaria Report.

```rust
use std::error::Error;
use std::sync::Arc;

use cloudllm::client_wrapper::Role;
use cloudllm::clients::claude::{ClaudeClient, Model as ClaudeModel};
use cloudllm::clients::gemini::{GeminiClient, Model as GeminiModel};
use cloudllm::clients::grok::{GrokClient, Model as GrokModel};
use cloudllm::clients::openai::{OpenAIClient, Model as OpenAIModel};
use cloudllm::{CouncilRole, CouncilSession, ParticipantConfig};

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let anthropic_key = std::env::var("ANTHROPIC_API_KEY")?;
    let openai_key = std::env::var("OPENAI_API_KEY")?;
    let gemini_key = std::env::var("GEMINI_API_KEY")?;
    let xai_key = std::env::var("XAI_API_KEY")?;

    let mut council = CouncilSession::new(
        "Estimate malaria R₀ under varying ITN and indoor residual spraying coverage. \
         When consensus emerges, draft a policy memo highlighting the ≥80% ITN threshold documented by WHO (2023).",
    );

    council.add_participant_with_config(
        Arc::new(ClaudeClient::new_with_model_enum(&anthropic_key, ClaudeModel::ClaudeSonnet4)),
        CouncilRole::Panelist,
        ParticipantConfig {
            display_name: Some("Epidemiological Modeler (Claude)".into()),
            persona_prompt: Some("Use Ross–Macdonald dynamics and cite observational studies from Tanzania and Ghana.".into()),
            max_tokens: Some(3584),
        },
    );

    council.add_participant_with_config(
        Arc::new(OpenAIClient::new_with_model_enum(&openai_key, OpenAIModel::O1)),
        CouncilRole::Panelist,
        ParticipantConfig {
            display_name: Some("R₀ Sensitivity Analyst (OpenAI O1)".into()),
            persona_prompt: Some("Quantify how ITN usage alters mosquito biting rate b and human infectious period 1/γ.".into()),
            max_tokens: Some(3584),
        },
    );

    council.add_participant_with_config(
        Arc::new(GeminiClient::new_with_model_enum(&gemini_key, GeminiModel::Gemini25Flash)),
        CouncilRole::Panelist,
        ParticipantConfig {
            display_name: Some("Operations Lead (Gemini)".into()),
            persona_prompt: Some("Highlight supply constraints, pyrethroid resistance, and community uptake barriers.".into()),
            max_tokens: Some(2048),
        },
    );

    council.add_participant_with_config(
        Arc::new(GrokClient::new_with_model_enum(&xai_key, GrokModel::Grok3Mini)),
        CouncilRole::Panelist,
        ParticipantConfig {
            display_name: Some("Policy Synthesizer (Grok)".into()),
            persona_prompt: Some("Track agreements and draft an action memo once consensus is achieved.".into()),
            max_tokens: Some(2048),
        },
    );

    let round_one = council
        .send_message(
            Role::User,
            "Round 1: Estimate R₀ for Plasmodium falciparum when ITN coverage rises from 60% to 80%. \
             Assume baseline R₀ = 3.2 with 60% coverage and historical field data showing ~40% biting reduction per 20 percentage-point increase.".to_string(),
            None,
        )
        .await?;

    println!("After round {} the council token budget stands at {} tokens.", round_one.round_index, council.total_usage().total_tokens);

    let round_two = council
        .send_message(
            Role::User,
            "Round 2: Reconcile your numbers and confirm whether ≥80% ITN coverage is sufficient to push R₀ below 1. \
             Draft the key policy message for health ministers overseeing high-transmission Sahel districts.".to_string(),
            None,
        )
        .await?;

    for reply in round_two.replies {
        println!("{}\n{}\n", reply.name, reply.message.content);
    }

    Ok(())
}
```

**Validation cue:** The second round should explicitly cite that reducing the mosquito biting rate by ≥ 40 % drops R₀ below one (e.g., from 3.2 to ≈ 0.96), consistent with WHO field trials. Any divergent result indicates the panel failed to integrate partners’ calculations.

---

## Recipe 4 — Evidence Reconciliation for Long-Duration Energy Storage Targets

**Goal:** Ensure a national grid reliability plan aligns with the Net-Zero Emissions by 2050 (IEA 2023) benchmark of 6–8 TWh of long-duration storage by 2030 for high-renewable systems.

**Council design:**

- Gemini 2.0 Flash runs techno-economic projections.
- OpenAI GPT-5 Mini checks capital expenditure curves.
- Claude Haiku challenges assumptions from a regulatory standpoint.
- Grok 4 Fast Reasoning produces a final compliance checklist.

This recipe shows how to interpret aggregate council usage and selectively re-run agents when new data arrives.

```rust
use std::error::Error;
use std::sync::Arc;

use cloudllm::client_wrapper::Role;
use cloudllm::clients::gemini::{GeminiClient, Model as GeminiModel};
use cloudllm::clients::openai::{OpenAIClient, Model as OpenAIModel};
use cloudllm::clients::claude::{ClaudeClient, Model as ClaudeModel};
use cloudllm::clients::grok::{GrokClient, Model as GrokModel};
use cloudllm::{CouncilRole, CouncilSession, ParticipantConfig, ParticipantId, ParticipantReply};

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let gemini_key = std::env::var("GEMINI_API_KEY")?;
    let openai_key = std::env::var("OPENAI_API_KEY")?;
    let anthropic_key = std::env::var("ANTHROPIC_API_KEY")?;
    let xai_key = std::env::var("XAI_API_KEY")?;

    let mut council = CouncilSession::new(
        "Check whether the storage roadmap reaches at least 6 TWh of long-duration storage by 2030 \
         as recommended in the IEA 2023 Net-Zero scenario. Flag if capex assumptions deviate from \$150/kWh by 2030.",
    );

    let gemini_id = council.add_participant_with_config(
        Arc::new(GeminiClient::new_with_model_enum(&gemini_key, GeminiModel::Gemini20Flash)),
        CouncilRole::Panelist,
        ParticipantConfig {
            display_name: Some("Techno-Economic Analyst (Gemini)".into()),
            persona_prompt: Some("Model storage build-out using learning rates and cite the IEA 2023 data tables.".into()),
            max_tokens: Some(4096),
        },
    );

    council.add_participant_with_config(
        Arc::new(OpenAIClient::new_with_model_enum(&openai_key, OpenAIModel::GPT5Mini)),
        CouncilRole::Panelist,
        ParticipantConfig {
            display_name: Some("Capex Reviewer (GPT-5 Mini)".into()),
            persona_prompt: Some("Audit cost curves and ensure \$150/kWh is achievable by 2030 under an 18% learning rate.".into()),
            max_tokens: Some(4096),
        },
    );

    council.add_participant_with_config(
        Arc::new(ClaudeClient::new_with_model_enum(&anthropic_key, ClaudeModel::ClaudeHaiku35)),
        CouncilRole::Panelist,
        ParticipantConfig {
            display_name: Some("Regulatory Advocate (Claude Haiku)".into()),
            persona_prompt: Some("Check permitting timelines and environmental justice safeguards.".into()),
            max_tokens: Some(2048),
        },
    );

    let grok_id = council.add_participant_with_config(
        Arc::new(GrokClient::new_with_model_enum(&xai_key, GrokModel::Grok4FastReasoning)),
        CouncilRole::Moderator,
        ParticipantConfig {
            display_name: Some("Scenario Synthesizer (Grok)".into()),
            persona_prompt: Some("Summarize points of agreement and list violations of the IEA benchmarks.".into()),
            max_tokens: Some(2048),
        },
    );

    council.set_round_robin_order(vec![grok_id, gemini_id])?;

    let round = council
        .send_message(
            Role::User,
            "Assess this roadmap: 4 TWh pumped hydro + 1.5 TWh flow batteries + 0.4 TWh gravity storage by 2030. \
             Capex trajectory assumes \$210/kWh in 2025 falling 18% with each doubling.".to_string(),
            None,
        )
        .await?;

    println!("Council used {} total tokens in this round.", council.total_usage().total_tokens);

    // Example: re-query a single participant if you need an updated projection
    if let Some(ref reply) = round
        .replies
        .iter()
        .find(|r| r.name.contains("Techno-Economic Analyst"))
    {
        println!("Gemini baseline analysis:\n{}\n", reply.message.content);
        if let Some(usage) = reply.usage.clone() {
            println!("Gemini tokens → {}", usage.total_tokens);
        }
    }

    Ok(())
}
```

**Expected output:** The council should flag that the plan reaches only 5.9 TWh (below the 6 TWh floor) and that the assumed learning rate barely attains the \$150/kWh target, requiring either additional capacity or accelerated cost declines.

---

## Troubleshooting Checklist

- **Order drift:** If specialists answer before the moderator, confirm the vector passed to `set_round_robin_order` lists moderator IDs first.
- **Token overruns:** Use `ParticipantConfig::max_tokens` to cap history length per participant, or reset the council when conversations grow too long.
- **Streaming needs:** The `GrokClient` wrapper already forwards `send_message_stream`; integrate it directly if you require live transcript dashboards before council-level streaming lands in the API.

---

## Where to Go Next

- Combine councils recursively: treat one `CouncilSession` as an agent inside a higher-level session to separate research and policy review.
- Add automated verification: parse each `ParticipantReply` and feed quantitative claims into domain-specific validators (e.g., comparing published mitigation pathways or epidemiological parameters).
- Instrument telemetry: persist `CouncilRoundResponse` objects to trace how token budgets translate into decision quality over time.

Armed with these recipes, you can orchestrate multi-provider, multi-persona panels that cross-check critical scientific policies against well-established benchmarks—giving teams higher confidence before they act.
