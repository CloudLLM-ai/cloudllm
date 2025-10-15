//! Venezuela Regime Change Debate Example
//!
//! This example demonstrates the Council API with debate mode and multi-agent collaboration.
//! A panel of Grok-4 analysts debates various scenarios for addressing the Venezuelan political
//! crisis, analyzing:
//! - Military intervention options
//! - Diplomatic and economic pressure strategies
//! - Covert operations possibilities
//! - Second-order effects on regional stability
//! - Transition of power to democratically elected leadership
//!
//! The agents include:
//! - Military Strategist (analyzes military options)
//! - Diplomatic Expert (evaluates diplomatic approaches)
//! - Intelligence Analyst (assesses covert operations)
//! - Economist (analyzes economic/sanctions impact)
//! - Regional Expert (second-order effects analysis)
//!
//! Run with:
//! ```bash
//! export XAI_API_KEY=your_key
//! cargo run --example venezuela_regime_change_debate
//! ```
//!
//! **Note**: This example takes 2-5 minutes to complete as it makes sequential API calls
//! to 5 different agents across multiple debate rounds. The Council API does not currently
//! support streaming or real-time progress updates during execution.

use chrono::{Duration, Utc};
use cloudllm::clients::grok::{GrokClient, Model as GrokModel};
use cloudllm::council::{Agent, Council, CouncilMode};
use openai_rust2::chat::{SearchMode, SearchParameters};
use std::error::Error as StdError;
use std::sync::Arc;

#[tokio::main]
async fn main() -> Result<(), Box<dyn StdError>> {
    // Initialize logger
    env_logger::Builder::from_default_env()
        .filter_level(log::LevelFilter::Info)
        .init();

    // Get API key from environment
    let xai_key = std::env::var("XAI_API_KEY").expect("XAI_API_KEY must be set");

    println!("\n{}", "=".repeat(80));
    println!("  Venezuela Regime Change Strategy Debate");
    println!("  Demonstrating Multi-Agent Council Collaboration");
    println!("{}\n", "=".repeat(80));

    let now = Utc::now();
    let time_today_str = now.format("%Y-%m-%d").to_string();
    let time_yesterday_str = (now - Duration::days(1)).format("%Y-%m-%d").to_string();

    let base_search_parameters = SearchParameters::new(SearchMode::On)
        .with_citations(true)
        .with_date_range_str(time_yesterday_str.clone(), time_today_str.clone());

    // Create agents with different expertises
    println!("Setting up council agents...\n");

    let military_strategist = Agent::new(
        "military",
        "Gen. Military Strategist",
        Arc::new(GrokClient::new_with_model_enum(
            &xai_key,
            GrokModel::Grok4_0709,
        )),
    )
    .with_expertise("Retired 4-star general with extensive experience in Latin America")
    .with_personality(
        "Analyze military intervention scenarios including force requirements, logistics, \
         potential casualties, international coalition building, and rules of engagement. \
         Be realistic about costs, risks, and likelihood of success.",
    )
    .with_search_parameters(base_search_parameters.clone());

    let diplomat = Agent::new(
        "diplomat",
        "Ambassador Diplomatic Expert",
        Arc::new(GrokClient::new_with_model_enum(
            &xai_key,
            GrokModel::Grok4_0709,
        )),
    )
    .with_expertise("Former US Ambassador with deep expertise in Latin American diplomacy")
    .with_personality(
        "Evaluate diplomatic strategies including multilateral pressure, recognition of \
         opposition leaders, negotiated transitions, and regional coalition building through \
         OAS and Lima Group. Consider Russia/China interests.",
    )
    .with_search_parameters(base_search_parameters.clone());

    let intelligence_analyst = Agent::new(
        "intelligence",
        "Intelligence Analyst",
        Arc::new(GrokClient::new_with_model_enum(
            &xai_key,
            GrokModel::Grok4_0709,
        )),
    )
    .with_expertise("Senior intelligence analyst specializing in covert operations")
    .with_personality(
        "Assess covert action possibilities including support for opposition groups, \
         information operations, cyber operations, and targeted actions. Consider \
         HUMINT networks, technical capabilities, deniability, and blowback risks.",
    )
    .with_search_parameters(base_search_parameters.clone());

    let economist = Agent::new(
        "economist",
        "Economic Sanctions Expert",
        Arc::new(GrokClient::new_with_model_enum(
            &xai_key,
            GrokModel::Grok4_0709,
        )),
    )
    .with_expertise("Economist specializing in sanctions and economic warfare")
    .with_personality(
        "Analyze economic pressure strategies including oil sector sanctions, \
         financial system isolation, secondary sanctions, humanitarian exemptions, \
         and economic reconstruction plans. Evaluate effectiveness and humanitarian impact.",
    )
    .with_search_parameters(base_search_parameters.clone());

    let regional_expert = Agent::new(
        "regional",
        "Regional Stability Analyst",
        Arc::new(GrokClient::new_with_model_enum(
            &xai_key,
            GrokModel::Grok4_0709,
        )),
    )
    .with_expertise("Expert in Latin American regional dynamics and second-order effects")
    .with_personality(
        "Analyze second-order effects and regional impacts including refugee flows \
         to Colombia/Brazil, drug trafficking dynamics, regional political stability, \
         reactions from leftist governments, oil market impacts, and long-term \
         democratization prospects.",
    )
    .with_search_parameters(base_search_parameters.clone());

    // Create council in Debate mode with convergence detection
    // Using fewer rounds (3) for faster execution - increase to 5+ for deeper analysis
    let mut council = Council::new("venezuela-council", "Venezuela Strategy Council")
        .with_mode(CouncilMode::Debate {
            max_rounds: 3,                     // Reduced from 5 for faster testing
            convergence_threshold: Some(0.65), // 65% similarity to converge
        })
        .with_system_context(
            "You are participating in a high-level strategic discussion about addressing \
             the Venezuelan political crisis. Be professional, analytical, and evidence-based. \
             Consider both ethical implications and practical realities. Engage constructively \
             with other experts' perspectives.",
        );

    // Add all agents
    council.add_agent(military_strategist)?;
    council.add_agent(diplomat)?;
    council.add_agent(intelligence_analyst)?;
    council.add_agent(economist)?;
    council.add_agent(regional_expert)?;

    println!(
        "Council configured with {} agents\n",
        council.list_agents().len()
    );

    // Initial strategic question
    let question =
        "We need to develop a comprehensive strategy assessment for addressing the Venezuelan \
         political crisis and supporting democratic transition. The democratically elected \
         opposition leader has been recognized by many nations but cannot assume power due to \
         Nicolás Maduro's continued control.\n\n\
         Please analyze the following scenarios in order of feasibility:\n\
         1. Diplomatic and economic pressure (sanctions escalation, regional coalitions)\n\
         2. Support for internal opposition movements (covert support, information operations)\n\
         3. Limited military intervention (naval blockade, no-fly zone, special operations)\n\
         4. Full military intervention (regime change operation)\n\n\
         For each scenario, evaluate:\n\
         - Likelihood of success\n\
         - Costs (human, financial, political)\n\
         - Second-order effects (regional stability, migration, oil markets)\n\
         - International support/opposition\n\
         - Timeline to democratic transition\n\
         - Risks and failure modes\n\n\
         Let's start with the most feasible approaches and work our way through the options.";

    println!("Question:\n{}\n", question);
    println!("{}\n", "=".repeat(80));

    // Start the debate - Note: This may take several minutes as 5 agents respond sequentially
    // The Council API doesn't currently support streaming or progress callbacks
    println!("Starting debate (max 3 rounds, converges at 65% similarity)...");
    println!("⚠️  This will take 2-5 minutes as each agent generates a response.");
    println!("   With 5 agents × 3 rounds = up to 15 sequential API calls\n");

    // Show a progress indicator
    print!("⏳ Processing: ");
    use std::io::{self, Write};
    io::stdout().flush().unwrap();

    let response = council
        .discuss(question, 3)
        .await
        .map_err(|e| format!("Council discussion failed: {}", e))?;

    println!("✅ Complete!\n");

    // Print all responses with better formatting
    let mut current_round = None;
    for message in response.messages.iter() {
        if let Some(agent_name) = &message.agent_name {
            let round = message
                .metadata
                .get("round")
                .and_then(|s| s.parse::<usize>().ok())
                .map(|r| r + 1);

            // Print round header if this is a new round
            if current_round != round {
                if current_round.is_some() {
                    println!("\n");
                }
                println!("{}", "=".repeat(80));
                println!("  ROUND {}", round.unwrap_or(0));
                println!("{}", "=".repeat(80));
                current_round = round;
            }

            println!("\n{}", "-".repeat(80));
            println!(
                "🗣️  {} ({})",
                agent_name,
                message.agent_id.as_ref().unwrap_or(&"unknown".to_string())
            );
            println!("{}", "-".repeat(80));

            // Truncate very long responses for readability
            let content = message.content.as_ref();
            if content.len() > 1500 {
                println!(
                    "{}...\n[Response truncated - {} chars total]",
                    &content[..1500],
                    content.len()
                );
            } else {
                println!("{}", content);
            }
        }
    }

    // Print summary
    println!("\n{}", "=".repeat(80));
    println!("DEBATE SUMMARY");
    println!("{}\n", "=".repeat(80));

    println!("Total rounds: {}", response.round);
    println!("Converged: {}", response.is_complete);
    if let Some(score) = response.convergence_score {
        println!("Final convergence score: {:.2}%", score * 100.0);
    }
    println!("Total tokens used: {}", response.total_tokens_used);
    println!("Total messages: {}", response.messages.len());

    if response.is_complete && response.convergence_score.is_some() {
        println!("\nThe agents reached consensus on the Venezuelan strategy.");
    } else {
        println!("\nThe debate reached maximum rounds without full convergence.");
        println!("Further discussion rounds could refine the strategy.");
    }

    Ok(())
}
