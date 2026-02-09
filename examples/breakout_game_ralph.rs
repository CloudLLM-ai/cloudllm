//! RALPH Orchestration Mode — Breakout Game Example
//!
//! This example demonstrates the RALPH (autonomous iterative loop) orchestration mode
//! by having multiple specialized agents collaborate to build a complete Atari Breakout
//! game in a single `index.html` file.
//!
//! RALPH works by repeatedly presenting agents with the same PRD task list. Agents see
//! accumulated work from previous iterations via conversation history and mark tasks
//! complete with `[TASK_COMPLETE:task_id]` markers. The loop ends when all tasks are
//! done or `max_iterations` is reached.
//!
//! ## Agents
//!
//! - **game-architect**: Designs HTML structure, CSS, Canvas setup
//! - **game-programmer**: Implements core game mechanics (physics, collision, rendering)
//! - **sound-designer**: Implements Atari 2600-style background music and SFX (Web Audio API)
//! - **powerup-engineer**: Implements all powerup systems
//!
//! ## PRD Tasks (10)
//!
//! 1. HTML boilerplate, canvas element, CSS styling
//! 2. requestAnimationFrame game loop, game state management
//! 3. Player paddle with keyboard input
//! 4. Ball movement, wall bouncing, paddle collision
//! 5. Brick grid with multi-hit bricks (different colors per HP)
//! 6. Ball-brick collision with brick destruction
//! 7. Atari 2600-style chiptune background music (Web Audio API oscillators)
//! 8. Sound effects on collisions
//! 9. Powerups: paddle length +10%, speed +10%, projectile shooting
//! 10. Spawn 2 extra balls powerup
//!
//! ## Running
//!
//! ```bash
//! export ANTHROPIC_KEY=your_key
//! cargo run --example breakout_game_ralph
//! ```
//!
//! The example writes the assembled game to `breakout_game.html` in the current directory.

use cloudllm::clients::claude::{ClaudeClient, Model};
use cloudllm::{
    orchestration::{Orchestration, OrchestrationMode, RalphTask},
    Agent,
};
use std::sync::Arc;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    env_logger::Builder::from_default_env()
        .filter_level(log::LevelFilter::Info)
        .init();

    let api_key = match std::env::var("ANTHROPIC_KEY") {
        Ok(key) => key,
        Err(_) => {
            eprintln!("Error: Set ANTHROPIC_KEY environment variable.");
            std::process::exit(1);
        }
    };

    println!("\n{}", "=".repeat(80));
    println!("  RALPH Orchestration Mode — Atari Breakout Game Builder");
    println!("  Using model: Claude Haiku 4.5");
    println!("{}\n", "=".repeat(80));

    // ── Agents ──────────────────────────────────────────────────────────────

    let make_client = || {
        Arc::new(ClaudeClient::new_with_model_enum(&api_key, Model::ClaudeHaiku45))
    };

    let architect = Agent::new("game-architect", "Game Architect", make_client())
        .with_expertise("HTML5 structure, CSS layout, Canvas setup")
        .with_personality(
            "Meticulous front-end architect who produces clean, well-structured HTML/CSS.",
        );

    let programmer = Agent::new("game-programmer", "Game Programmer", make_client())
        .with_expertise("JavaScript game mechanics, physics, collision detection, rendering")
        .with_personality(
            "Seasoned game developer who writes tight, performant JavaScript game loops.",
        );

    let sound_designer = Agent::new("sound-designer", "Sound Designer", make_client())
        .with_expertise("Web Audio API, chiptune synthesis, oscillator-based sound effects")
        .with_personality(
            "Retro audio enthusiast who crafts authentic Atari 2600-era sounds with Web Audio API oscillators.",
        );

    let powerup_engineer = Agent::new("powerup-engineer", "Powerup Engineer", make_client())
        .with_expertise("Game powerup systems, spawn logic, timed effects")
        .with_personality(
            "Creative gameplay engineer who designs fun and balanced powerup mechanics.",
        );

    // ── PRD Tasks ───────────────────────────────────────────────────────────

    let tasks = vec![
        RalphTask::new(
            "html_structure",
            "HTML Structure",
            "Create the HTML boilerplate with a <canvas> element, CSS styling (centered canvas, \
             dark background, retro font), and all necessary <script> tags. Everything must be in \
             a single self-contained index.html.",
        ),
        RalphTask::new(
            "game_loop",
            "Game Loop",
            "Implement the requestAnimationFrame game loop with game state management \
             (menu, playing, paused, game_over). Include score tracking and lives display.",
        ),
        RalphTask::new(
            "paddle_control",
            "Paddle Control",
            "Implement the player paddle with left/right arrow key and A/D key input. \
             Paddle should be constrained to the canvas bounds.",
        ),
        RalphTask::new(
            "ball_physics",
            "Ball Physics",
            "Implement ball movement with velocity, wall bouncing (top, left, right), \
             paddle collision with angle reflection based on hit position, and bottom-of-screen \
             life loss.",
        ),
        RalphTask::new(
            "brick_layout",
            "Brick Layout",
            "Create a brick grid with multiple rows. Bricks have HP (1-3 hits) with different \
             colors per HP level (e.g., green=1HP, yellow=2HP, red=3HP). Display remaining HP visually.",
        ),
        RalphTask::new(
            "collision_detection",
            "Collision Detection",
            "Implement ball-brick collision detection. When a brick is hit, decrease its HP. \
             When HP reaches 0, destroy the brick and add score. Handle ball deflection on brick hit.",
        ),
        RalphTask::new(
            "background_music",
            "Background Music",
            "Implement Atari 2600-style chiptune background music using Web Audio API oscillators. \
             Use square and triangle waves to create a looping retro melody. Music should start on \
             game start and loop continuously.",
        ),
        RalphTask::new(
            "collision_sfx",
            "Collision Sound Effects",
            "Implement distinct sound effects for: ball-brick hit (high pitched blip), \
             ball-paddle hit (medium thud), ball-wall bounce (low click). Use Web Audio API \
             oscillators with short duration envelopes.",
        ),
        RalphTask::new(
            "powerups_basic",
            "Basic Powerups",
            "Implement powerups that drop from destroyed bricks (random chance): \
             paddle length +10% (green powerup), ball speed +10% (blue powerup), \
             projectile shooting with space bar (red powerup — fires 2 projectiles that deal \
             1 damage each to bricks). Powerups fall downward and are caught by the paddle.",
        ),
        RalphTask::new(
            "powerup_multiball",
            "Multiball Powerup",
            "Implement a multiball powerup (purple) that spawns 2 extra balls when collected. \
             Extra balls behave identically to the main ball. Game continues as long as at least \
             one ball remains. All balls interact with bricks and the paddle.",
        ),
    ];

    // ── Orchestration ───────────────────────────────────────────────────────

    let system_context = "\
You are collaborating with other specialized agents to build a complete Atari Breakout game \
in a single self-contained index.html file. All HTML, CSS, and JavaScript must be inline. \
Do NOT use external dependencies. Use the HTML5 Canvas API for rendering and the Web Audio API \
for sound. \n\n\
When you work on a task, output the COMPLETE updated index.html incorporating ALL previous work \
plus your additions. Never output partial snippets — always output the full file. \n\n\
When a task is fully implemented, include the marker [TASK_COMPLETE:task_id] at the end of your \
response (e.g., [TASK_COMPLETE:html_structure]). You may complete multiple tasks at once.";

    let mut orchestration =
        Orchestration::new("breakout-builder", "Breakout Game RALPH Orchestration")
            .with_mode(OrchestrationMode::Ralph {
                tasks,
                max_iterations: 5,
            })
            .with_system_context(system_context)
            .with_max_tokens(180_000);

    orchestration.add_agent(architect)?;
    orchestration.add_agent(programmer)?;
    orchestration.add_agent(sound_designer)?;
    orchestration.add_agent(powerup_engineer)?;

    // ── Run ─────────────────────────────────────────────────────────────────

    let prompt = "\
Build a complete Atari Breakout game in a single self-contained index.html. \
The game should feature: an HTML5 Canvas, a game loop with state management, \
keyboard-controlled paddle, ball physics with angle reflection, multi-hit bricks \
with color-coded HP, collision detection, Atari 2600-style chiptune background music, \
collision sound effects, powerups (paddle size, speed boost, projectile shooting), \
and a multiball powerup. Everything must work in a modern browser with no external dependencies.";

    println!("Starting RALPH orchestration with 4 agents and 10 PRD tasks...\n");

    let response = orchestration.discuss(prompt, 1).await?;

    // ── Results ─────────────────────────────────────────────────────────────

    println!("\n{}", "=".repeat(80));
    println!("  RALPH Results");
    println!("{}", "=".repeat(80));
    println!("  Iterations used : {}", response.round);
    println!("  All tasks done  : {}", response.is_complete);
    println!(
        "  Completion score: {:.0}%",
        response.convergence_score.unwrap_or(0.0) * 100.0
    );
    println!("  Total tokens    : {}", response.total_tokens_used);
    println!("  Messages        : {}", response.messages.len());
    println!("{}\n", "=".repeat(80));

    // Print per-message summary
    for (i, msg) in response.messages.iter().enumerate() {
        let agent = msg.agent_name.as_deref().unwrap_or("unknown");
        let iteration = msg.metadata.get("iteration").map(|s| s.as_str()).unwrap_or("?");
        let completed = msg.metadata.get("tasks_completed").map(|s| s.as_str()).unwrap_or("-");
        let preview_len = 120.min(msg.content.len());
        let preview_end = msg
            .content
            .char_indices()
            .nth(preview_len)
            .map(|(i, _)| i)
            .unwrap_or(msg.content.len());
        println!(
            "  [{}] iter={} agent={:<20} tasks_completed={:<30} preview={}...",
            i + 1,
            iteration,
            agent,
            completed,
            &msg.content[..preview_end]
        );
    }

    // Extract the last message's content as the final HTML (the last agent output
    // should contain the most complete version of the file).
    if let Some(last_msg) = response.messages.last() {
        // Try to extract just the HTML from the response
        let html = extract_html(&last_msg.content);
        std::fs::write("breakout_game.html", &html)?;
        println!("\nGame written to breakout_game.html ({} bytes)", html.len());
        println!("Open it in a browser to play!");
    } else {
        println!("\nNo messages were generated. Check your API key and try again.");
    }

    Ok(())
}

/// Attempt to extract a self-contained HTML document from an LLM response.
/// Falls back to using the entire string if no `<!DOCTYPE` or `<html` tag is found.
fn extract_html(text: &str) -> String {
    // Look for the start of an HTML document
    let lower = text.to_lowercase();
    let start = lower
        .find("<!doctype")
        .or_else(|| lower.find("<html"))
        .unwrap_or(0);

    // Look for the closing </html> tag
    let end = lower
        .rfind("</html>")
        .map(|i| i + "</html>".len())
        .unwrap_or(text.len());

    text[start..end].to_string()
}
