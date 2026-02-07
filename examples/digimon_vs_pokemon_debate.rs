//! Digimon vs Pokemon Debate Example
//!
//! This example demonstrates the Orchestration API in Moderated mode with a fun debate
//! between two experts arguing about which franchise is better: Digimon or Pokemon.
//!
//! The debate features:
//! - A neutral moderator who guides the discussion
//! - A Digimon enthusiast who champions Digimon
//! - A Pokemon expert who advocates for Pokemon
//!
//! Topics covered:
//! - Story depth and character development
//! - Gameplay mechanics and strategy
//! - Cultural impact and longevity
//! - Animation quality and world-building
//! - Innovation and evolution of the franchises
//!
//! Run with:
//! ```bash
//! export OPENAI_API_KEY=your_key
//! export ANTHROPIC_API_KEY=your_key
//! cargo run --example digimon_vs_pokemon_debate
//! ```
//!
//! **Note**: This example takes 1-3 minutes as it makes sequential API calls.
//! The Moderated mode means the moderator selects speakers, creating a more
//! dynamic conversation flow.

use cloudllm::clients::claude::ClaudeClient;
use cloudllm::clients::openai::OpenAIClient;
use cloudllm::{
    orchestration::{Orchestration, OrchestrationMode},
    Agent,
};
use std::error::Error as StdError;
use std::sync::Arc;

#[tokio::main]
async fn main() -> Result<(), Box<dyn StdError>> {
    // Initialize logger
    env_logger::Builder::from_default_env()
        .filter_level(log::LevelFilter::Info)
        .init();

    // Get API keys from environment
    let openai_key = std::env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY must be set");
    let anthropic_key = std::env::var("ANTHROPIC_API_KEY").expect("ANTHROPIC_API_KEY must be set");

    println!("\n{}", "=".repeat(80));
    println!("  🎮 THE GREAT DEBATE: Digimon vs Pokemon 🎮");
    println!("  Demonstrating Moderated Orchestration Mode");
    println!("{}\n", "=".repeat(80));

    // Create the moderator
    println!("Setting up debate panel...\n");

    let moderator = Agent::new(
        "moderator",
        "Debate Moderator",
        Arc::new(OpenAIClient::new_with_model_string(&openai_key, "gpt-4o")),
    )
    .with_expertise("Professional debate moderator with deep knowledge of anime and gaming culture")
    .with_personality(
        "You are a neutral, professional moderator. Your role is to:\n\
         - Ask probing questions that get to the heart of each franchise's strengths\n\
         - Ensure both sides get fair representation\n\
         - Highlight interesting points of comparison\n\
         - Keep the debate civil and fun\n\
         - Summarize key arguments\n\
         Don't take sides, but do encourage specific examples and concrete comparisons.",
    );

    // Create the Digimon expert
    let digimon_expert = Agent::new(
        "digimon",
        "Digimon Enthusiast",
        Arc::new(ClaudeClient::new_with_model_str(&anthropic_key, "claude-sonnet-4-20250514"))
    )
    .with_expertise(
        "Digimon superfan who has watched all seasons, played all games, and read the manga"
    )
    .with_personality(
        "You are passionate about Digimon and believe it's superior to Pokemon. Your key arguments:\n\
         - MUCH deeper storytelling and character development\n\
         - More mature themes (death, sacrifice, growth)\n\
         - Evolution is emotional and story-driven, not just level grinding\n\
         - Digimon partners have distinct personalities and relationships\n\
         - Adventure 01/02 had incredible character arcs\n\
         - The Digital World is more creative and varied\n\
         Be enthusiastic but back up claims with specific examples from the shows/games."
    );

    // Create the Pokemon expert
    let pokemon_expert = Agent::new(
        "pokemon",
        "Pokemon Master",
        Arc::new(ClaudeClient::new_with_model_str(
            &anthropic_key,
            "claude-sonnet-4-20250514",
        )),
    )
    .with_expertise(
        "Pokemon expert with encyclopedic knowledge of all generations, competitive play, and lore",
    )
    .with_personality(
        "You believe Pokemon is the superior franchise. Your key arguments:\n\
         - More polished, consistent quality across decades\n\
         - Superior game design with deeper strategy (competitive scene)\n\
         - 1000+ Pokemon with incredible variety and design\n\
         - Stronger brand recognition and cultural impact\n\
         - Better world-building with consistent lore\n\
         - Games are more replayable and mechanically deep\n\
         - Music is iconic (Lavender Town, Champion themes, etc.)\n\
         Be confident but acknowledge Digimon's strengths. Use specific examples.",
    );

    // Create orchestration in Moderated mode
    let mut orchestration = Orchestration::new("anime-debate", "The Great Digimon vs Pokemon Debate")
        .with_mode(OrchestrationMode::Moderated {
            moderator_id: "moderator".to_string(),
        })
        .with_system_context(
            "This is a friendly but spirited debate about two beloved anime franchises. \
             Be respectful, enthusiastic, and back up your arguments with specific examples. \
             The goal is to have a fun, informative discussion that fans of both sides can enjoy.",
        );

    // Add all agents
    orchestration.add_agent(moderator)?;
    orchestration.add_agent(digimon_expert)?;
    orchestration.add_agent(pokemon_expert)?;

    println!("✅ Debate panel assembled:");
    println!("   🎭 Moderator: Debate Moderator (gpt-4o)");
    println!("   🐉 Digimon Side: Digimon Enthusiast (claude-sonnet-4)");
    println!("   ⚡ Pokemon Side: Pokemon Master (claude-sonnet-4)\n");

    // Opening question
    let opening_question = "Welcome to the great debate: Digimon vs Pokemon! \n\n\
         Today we're settling the age-old question - which franchise is truly better? \n\n\
         Let's start with the fundamentals. I want each side to make their opening argument \
         covering these key areas:\n\
         1. Story and narrative quality\n\
         2. Character development and relationships\n\
         3. Gameplay and mechanics (for the games)\n\
         4. Cultural impact and longevity\n\
         5. Innovation and uniqueness\n\n\
         Let's begin with the Digimon enthusiast - make your case for why Digimon is superior!";

    println!("🎤 Opening Question:\n{}\n", opening_question);
    println!("{}\n", "=".repeat(80));

    // Show progress indicator
    println!("⏳ Starting moderated debate (5 rounds)...");
    println!("   The moderator will direct questions to each expert.\n");

    use std::io::{self, Write};
    print!("🎭 Processing: ");
    io::stdout().flush().unwrap();

    // Run debate for 5 rounds
    let response = orchestration
        .discuss(opening_question, 5)
        .await
        .map_err(|e| format!("Debate failed: {}", e))?;

    println!("✅ Complete!\n");

    // Print all exchanges with nice formatting
    let mut exchange_num = 0;
    for message in response.messages.iter() {
        if let Some(agent_name) = &message.agent_name {
            exchange_num += 1;

            // Choose emoji based on speaker
            let emoji = match agent_name.as_str() {
                "Debate Moderator" => "🎭",
                "Digimon Enthusiast" => "🐉",
                "Pokemon Master" => "⚡",
                _ => "💬",
            };

            println!("\n{}", "-".repeat(80));
            println!(
                "{} Exchange {} - {} speaks:",
                emoji, exchange_num, agent_name
            );
            println!("{}", "-".repeat(80));

            // Show full response for this shorter, fun debate
            let content = message.content.as_ref();
            println!("{}\n", content);
        }
    }

    // Print summary
    println!("\n{}", "=".repeat(80));
    println!("📊 DEBATE STATISTICS");
    println!("{}\n", "=".repeat(80));

    println!("Total exchanges: {}", exchange_num);
    println!("Rounds completed: {}", response.round);
    println!("Total messages: {}", response.messages.len());

    // Count who spoke more
    let mut digimon_count = 0;
    let mut pokemon_count = 0;
    let mut moderator_count = 0;

    for message in &response.messages {
        if let Some(agent_name) = &message.agent_name {
            match agent_name.as_str() {
                "Digimon Enthusiast" => digimon_count += 1,
                "Pokemon Master" => pokemon_count += 1,
                "Debate Moderator" => moderator_count += 1,
                _ => {}
            }
        }
    }

    println!("\nParticipation:");
    println!("  🐉 Digimon Enthusiast: {} contributions", digimon_count);
    println!("  ⚡ Pokemon Master: {} contributions", pokemon_count);
    println!("  🎭 Moderator: {} interventions", moderator_count);

    println!("\n{}", "=".repeat(80));
    println!("🎮 Thanks for watching the debate! Both franchises are amazing in their own ways.");
    println!("{}", "=".repeat(80));

    Ok(())
}
