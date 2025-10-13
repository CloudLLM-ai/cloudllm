// This example demonstrates how to use MultiParticipantSession to orchestrate
// conversations between multiple LLM clients with different roles and strategies.
//
// To run this example, you need API keys for the providers you want to use:
// - OPEN_AI_SECRET for OpenAI
// - XAI_API_KEY for Grok
// - GEMINI_API_KEY for Gemini
//
// Example usage:
// ```
// export OPEN_AI_SECRET="your-openai-key"
// export XAI_API_KEY="your-grok-key"
// cargo run --example multi_participant_example
// ```

use cloudllm::client_wrapper::Role;
use cloudllm::clients::openai::{Model, OpenAIClient};
use cloudllm::multi_participant_session::{
    MultiParticipantSession, OrchestrationStrategy, ParticipantRole,
};
use std::sync::Arc;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging
    cloudllm::init_logger();

    println!("=== CloudLLM Multi-Participant Session Example ===\n");

    // Example 1: Broadcast Strategy - All participants respond simultaneously
    println!("\n--- Example 1: Broadcast Strategy ---");
    broadcast_example().await?;

    // Example 2: Round-Robin Strategy - Participants respond sequentially, each seeing previous responses
    println!("\n--- Example 2: Round-Robin Strategy ---");
    round_robin_example().await?;

    // Example 3: Moderator-Led Strategy - Moderator responds first, then others
    println!("\n--- Example 3: Moderator-Led Strategy ---");
    moderator_led_example().await?;

    // Example 4: Hierarchical Strategy - Workers process task, supervisors synthesize
    println!("\n--- Example 4: Hierarchical Strategy ---");
    hierarchical_example().await?;

    // Example 5: Custom Priority Strategy - Participants respond in priority order
    println!("\n--- Example 5: Custom Priority Strategy ---");
    custom_priority_example().await?;

    println!("\n=== All examples completed successfully! ===");
    Ok(())
}

// Example 1: Broadcast - All participants respond to the same prompt simultaneously
async fn broadcast_example() -> Result<(), Box<dyn std::error::Error>> {
    let openai_key = std::env::var("OPEN_AI_SECRET")
        .unwrap_or_else(|_| "demo-key-please-set-OPEN_AI_SECRET".to_string());

    // Create two different OpenAI clients with different models
    let gpt4_client = Arc::new(OpenAIClient::new_with_model_enum(
        &openai_key,
        Model::GPt4oMini,
    ));
    let gpt4o_client = Arc::new(OpenAIClient::new_with_model_enum(
        &openai_key,
        Model::GPT4o,
    ));

    // Create session with broadcast strategy
    let mut session = MultiParticipantSession::new(
        "You are participating in a panel discussion about AI technology.".to_string(),
        8192,
        OrchestrationStrategy::Broadcast,
    );

    // Add participants
    session.add_participant("GPT-4o-mini", gpt4_client, ParticipantRole::Panelist);
    session.add_participant("GPT-4o", gpt4o_client, ParticipantRole::Panelist);

    println!("Participants: {:?}", session.list_participants());
    println!("Asking: 'What is the most exciting development in AI recently?'");

    // Send message - all participants will respond
    let responses = session
        .send_message(
            Role::User,
            "In one sentence, what is the most exciting development in AI recently?".to_string(),
            None,
        )
        .await?;

    println!("\nResponses received: {}", responses.len());
    for response in responses {
        println!(
            "\n{} ({:?}):",
            response.participant_name, response.participant_role
        );
        println!("{}", response.content);
        if let Some(usage) = response.token_usage {
            println!(
                "  [Tokens - Input: {}, Output: {}, Total: {}]",
                usage.input_tokens, usage.output_tokens, usage.total_tokens
            );
        }
    }

    Ok(())
}

// Example 2: Round-Robin - Participants respond sequentially, each seeing previous responses
async fn round_robin_example() -> Result<(), Box<dyn std::error::Error>> {
    let openai_key = std::env::var("OPEN_AI_SECRET")
        .unwrap_or_else(|_| "demo-key-please-set-OPEN_AI_SECRET".to_string());

    let client1 = Arc::new(OpenAIClient::new_with_model_enum(
        &openai_key,
        Model::GPt4oMini,
    ));
    let client2 = Arc::new(OpenAIClient::new_with_model_enum(
        &openai_key,
        Model::GPt4oMini,
    ));

    let mut session = MultiParticipantSession::new(
        "You are participating in a sequential discussion where you build on previous responses."
            .to_string(),
        8192,
        OrchestrationStrategy::RoundRobin,
    );

    session.add_participant("First Responder", client1, ParticipantRole::Panelist);
    session.add_participant("Second Responder", client2, ParticipantRole::Panelist);

    println!("Participants: {:?}", session.list_participants());
    println!("Asking: 'What are the benefits of renewable energy?'");

    let responses = session
        .send_message(
            Role::User,
            "What are the benefits of renewable energy? (one sentence)".to_string(),
            None,
        )
        .await?;

    println!("\nSequential responses:");
    for (i, response) in responses.iter().enumerate() {
        println!("\n{}. {} ({:?}):", i + 1, response.participant_name, response.participant_role);
        println!("{}", response.content);
    }

    Ok(())
}

// Example 3: Moderator-Led - Moderator responds first, others respond to moderator's framing
async fn moderator_led_example() -> Result<(), Box<dyn std::error::Error>> {
    let openai_key = std::env::var("OPEN_AI_SECRET")
        .unwrap_or_else(|_| "demo-key-please-set-OPEN_AI_SECRET".to_string());

    let moderator_client = Arc::new(OpenAIClient::new_with_model_enum(
        &openai_key,
        Model::GPT4o,
    ));
    let panelist1 = Arc::new(OpenAIClient::new_with_model_enum(
        &openai_key,
        Model::GPt4oMini,
    ));
    let panelist2 = Arc::new(OpenAIClient::new_with_model_enum(
        &openai_key,
        Model::GPt4oMini,
    ));

    let mut session = MultiParticipantSession::new(
        "You are in a panel discussion. The moderator frames questions, panelists respond."
            .to_string(),
        8192,
        OrchestrationStrategy::ModeratorLed,
    );

    session.add_participant("Moderator", moderator_client, ParticipantRole::Moderator);
    session.add_participant("Tech Expert", panelist1, ParticipantRole::Panelist);
    session.add_participant("Business Expert", panelist2, ParticipantRole::Panelist);

    println!("Participants: {:?}", session.list_participants());
    println!("Topic: 'The future of remote work'");

    let responses = session
        .send_message(
            Role::User,
            "Let's discuss the future of remote work. (brief responses)".to_string(),
            None,
        )
        .await?;

    println!("\nModerated discussion:");
    for response in responses {
        println!("\n{} ({:?}):", response.participant_name, response.participant_role);
        println!("{}", response.content);
    }

    Ok(())
}

// Example 4: Hierarchical - Workers do tasks, supervisor synthesizes results
async fn hierarchical_example() -> Result<(), Box<dyn std::error::Error>> {
    let openai_key = std::env::var("OPEN_AI_SECRET")
        .unwrap_or_else(|_| "demo-key-please-set-OPEN_AI_SECRET".to_string());

    let worker1 = Arc::new(OpenAIClient::new_with_model_enum(
        &openai_key,
        Model::GPt4oMini,
    ));
    let worker2 = Arc::new(OpenAIClient::new_with_model_enum(
        &openai_key,
        Model::GPt4oMini,
    ));
    let supervisor = Arc::new(OpenAIClient::new_with_model_enum(
        &openai_key,
        Model::GPT4o,
    ));

    let mut session = MultiParticipantSession::new(
        "Workers provide analysis, supervisors synthesize findings.".to_string(),
        8192,
        OrchestrationStrategy::Hierarchical,
    );

    session.add_participant("Data Analyst", worker1, ParticipantRole::Worker);
    session.add_participant("Market Researcher", worker2, ParticipantRole::Worker);
    session.add_participant("Senior Strategist", supervisor, ParticipantRole::Supervisor);

    println!("Participants: {:?}", session.list_participants());
    println!("Task: Analyze market trends");

    let responses = session
        .send_message(
            Role::User,
            "Provide one key insight about current tech market trends.".to_string(),
            None,
        )
        .await?;

    println!("\nHierarchical workflow results:");
    println!("\nWorker Analyses:");
    for response in responses.iter().filter(|r| r.participant_role == ParticipantRole::Worker) {
        println!("\n  {} (Worker):", response.participant_name);
        println!("  {}", response.content);
    }

    println!("\nSupervisor Synthesis:");
    for response in responses.iter().filter(|r| r.participant_role == ParticipantRole::Supervisor) {
        println!("\n  {} (Supervisor):", response.participant_name);
        println!("  {}", response.content);
    }

    Ok(())
}

// Example 5: Custom Priority - Participants respond in priority order
async fn custom_priority_example() -> Result<(), Box<dyn std::error::Error>> {
    let openai_key = std::env::var("OPEN_AI_SECRET")
        .unwrap_or_else(|_| "demo-key-please-set-OPEN_AI_SECRET".to_string());

    let high_priority = Arc::new(OpenAIClient::new_with_model_enum(
        &openai_key,
        Model::GPT4o,
    ));
    let medium_priority = Arc::new(OpenAIClient::new_with_model_enum(
        &openai_key,
        Model::GPt4oMini,
    ));
    let low_priority = Arc::new(OpenAIClient::new_with_model_enum(
        &openai_key,
        Model::GPt4oMini,
    ));

    let mut session = MultiParticipantSession::new(
        "Respond in order of priority/expertise.".to_string(),
        8192,
        OrchestrationStrategy::Custom,
    );

    // Add with different priorities (higher number = higher priority)
    session.add_participant_with_priority(
        "Junior Analyst",
        low_priority,
        ParticipantRole::Panelist,
        1,
    );
    session.add_participant_with_priority(
        "Senior Consultant",
        medium_priority,
        ParticipantRole::Panelist,
        5,
    );
    session.add_participant_with_priority(
        "Chief Expert",
        high_priority,
        ParticipantRole::Panelist,
        10,
    );

    println!("Participants in priority order: {:?}", session.list_participants());
    println!("Question: 'What's the best approach?'");

    let responses = session
        .send_message(
            Role::User,
            "What's the best approach to solving complex problems? (one sentence)".to_string(),
            None,
        )
        .await?;

    println!("\nResponses in priority order:");
    for (i, response) in responses.iter().enumerate() {
        println!("\n{}. {}:", i + 1, response.participant_name);
        println!("   {}", response.content);
    }

    Ok(())
}
