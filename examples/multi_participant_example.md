# Multi-Participant LLM Sessions

The `MultiParticipantSession` feature enables orchestrating conversations between multiple LLM clients, allowing you to create complex multi-agent systems, expert panels, and hierarchical AI architectures.

## Overview

Multi-participant sessions allow you to:
- Create panel discussions with different LLM models
- Implement hierarchical agent structures (supervisors, workers, evaluators)
- Build round-robin conversations where agents build on each other's responses
- Design custom agent topologies with priority-based execution
- Experiment with different orchestration strategies

## Key Concepts

### Participant Roles

Participants can have different roles that affect how they interact in the session:

- **`Moderator`**: Guides the conversation and synthesizes responses
- **`Panelist`**: Regular participant that contributes to discussions
- **`Observer`**: Receives all messages but doesn't actively respond
- **`Evaluator`**: Assesses responses from other participants
- **`Supervisor`**: Oversees worker output and provides synthesis
- **`Worker`**: Performs specific tasks assigned by supervisors

### Orchestration Strategies

The session supports multiple strategies for coordinating message flow:

- **`Broadcast`**: All participants receive and respond to messages simultaneously
- **`RoundRobin`**: Participants respond sequentially, each seeing previous responses
- **`ModeratorLed`**: Moderator responds first, then distributes to others
- **`Hierarchical`**: Workers process tasks, supervisors synthesize results
- **`Custom`**: Participants respond in priority order (user-defined)

## Basic Usage

```rust
use cloudllm::client_wrapper::Role;
use cloudllm::clients::openai::OpenAIClient;
use cloudllm::multi_participant_session::{
    MultiParticipantSession, OrchestrationStrategy, ParticipantRole,
};
use std::sync::Arc;

// Create different LLM clients
let client1 = Arc::new(OpenAIClient::new("api_key", "gpt-4"));
let client2 = Arc::new(OpenAIClient::new("api_key", "gpt-4o"));

// Create a session with broadcast strategy
let mut session = MultiParticipantSession::new(
    "You are participating in an AI panel discussion.".to_string(),
    8192,  // max tokens
    OrchestrationStrategy::Broadcast,
);

// Add participants
session.add_participant("Expert-1", client1, ParticipantRole::Panelist);
session.add_participant("Expert-2", client2, ParticipantRole::Panelist);

// Send a message and get responses from all participants
let responses = session.send_message(
    Role::User,
    "What are the key challenges in AI alignment?".to_string(),
    None,
).await?;

// Process responses
for response in responses {
    println!("{}: {}", response.participant_name, response.content);
    if let Some(usage) = response.token_usage {
        println!("Tokens used: {}", usage.total_tokens);
    }
}
```

## Orchestration Strategy Examples

### 1. Broadcast Strategy

All participants respond to the same prompt simultaneously. Best for gathering diverse perspectives on a single question.

```rust
let mut session = MultiParticipantSession::new(
    "You are an expert panelist.".to_string(),
    8192,
    OrchestrationStrategy::Broadcast,
);

session.add_participant("GPT-4", client1, ParticipantRole::Panelist);
session.add_participant("Claude", client2, ParticipantRole::Panelist);
session.add_participant("Gemini", client3, ParticipantRole::Panelist);

// All three will respond independently to the same question
let responses = session.send_message(
    Role::User,
    "What is the most important AI development in 2024?".to_string(),
    None,
).await?;
```

### 2. Round-Robin Strategy

Participants respond sequentially, with each seeing the previous participant's response. Creates a flowing conversation.

```rust
let mut session = MultiParticipantSession::new(
    "Build on the previous response.".to_string(),
    8192,
    OrchestrationStrategy::RoundRobin,
);

session.add_participant("Analyst-1", client1, ParticipantRole::Panelist);
session.add_participant("Analyst-2", client2, ParticipantRole::Panelist);
session.add_participant("Analyst-3", client3, ParticipantRole::Panelist);

// Each analyst will see and build upon previous analysts' responses
let responses = session.send_message(
    Role::User,
    "Analyze the impact of AI on education.".to_string(),
    None,
).await?;
```

### 3. Moderator-Led Strategy

A moderator frames the discussion first, then other participants respond to the moderator's framing.

```rust
let mut session = MultiParticipantSession::new(
    "You are in a moderated panel.".to_string(),
    8192,
    OrchestrationStrategy::ModeratorLed,
);

session.add_participant("Moderator", moderator_client, ParticipantRole::Moderator);
session.add_participant("Expert-1", expert1, ParticipantRole::Panelist);
session.add_participant("Expert-2", expert2, ParticipantRole::Panelist);

// Moderator responds first, then experts respond to moderator's framing
let responses = session.send_message(
    Role::User,
    "Discuss the ethics of AI in healthcare.".to_string(),
    None,
).await?;
```

### 4. Hierarchical Strategy

Workers process the task independently, then supervisors synthesize their findings.

```rust
let mut session = MultiParticipantSession::new(
    "Workers analyze, supervisors synthesize.".to_string(),
    8192,
    OrchestrationStrategy::Hierarchical,
);

session.add_participant("Data-Worker", worker1, ParticipantRole::Worker);
session.add_participant("Research-Worker", worker2, ParticipantRole::Worker);
session.add_participant("Lead-Supervisor", supervisor, ParticipantRole::Supervisor);

// Workers analyze first, supervisor sees all worker outputs and synthesizes
let responses = session.send_message(
    Role::User,
    "Analyze market trends in renewable energy.".to_string(),
    None,
).await?;

// Filter responses by role
let worker_analyses: Vec<_> = responses.iter()
    .filter(|r| r.participant_role == ParticipantRole::Worker)
    .collect();
    
let supervisor_synthesis: Vec<_> = responses.iter()
    .filter(|r| r.participant_role == ParticipantRole::Supervisor)
    .collect();
```

### 5. Custom Priority Strategy

Participants respond in order of priority (higher priority responds first).

```rust
let mut session = MultiParticipantSession::new(
    "Respond by priority level.".to_string(),
    8192,
    OrchestrationStrategy::Custom,
);

// Add with different priorities (higher = earlier in order)
session.add_participant_with_priority("Junior", junior_client, ParticipantRole::Panelist, 1);
session.add_participant_with_priority("Senior", senior_client, ParticipantRole::Panelist, 5);
session.add_participant_with_priority("Principal", principal_client, ParticipantRole::Panelist, 10);

// Responses will be in order: Principal -> Senior -> Junior
let responses = session.send_message(
    Role::User,
    "What's the best architecture for this system?".to_string(),
    None,
).await?;
```

## Advanced Usage

### Managing Participants

```rust
// Add a participant
session.add_participant("Agent-1", client, ParticipantRole::Panelist);

// Add with priority
session.add_participant_with_priority("Agent-2", client, ParticipantRole::Evaluator, 5);

// List all participants
let participants = session.list_participants();

// Get a specific participant
if let Some(participant) = session.get_participant("Agent-1") {
    println!("Found: {}", participant.name);
}

// Remove a participant
session.remove_participant("Agent-1");
```

### Changing Strategy Mid-Session

```rust
// Start with broadcast
let mut session = MultiParticipantSession::new(
    "System prompt".to_string(),
    8192,
    OrchestrationStrategy::Broadcast,
);

// ... add participants and have some conversations ...

// Switch to round-robin for deeper discussion
session.set_orchestration_strategy(OrchestrationStrategy::RoundRobin);

// Continue with new strategy
let responses = session.send_message(
    Role::User,
    "Let's dive deeper into this topic.".to_string(),
    None,
).await?;
```

### Accessing Participant Information

```rust
let responses = session.send_message(/* ... */).await?;

for response in responses {
    println!("Participant: {}", response.participant_name);
    println!("Role: {:?}", response.participant_role);
    println!("Content: {}", response.content);
    
    if let Some(usage) = response.token_usage {
        println!("Input tokens: {}", usage.input_tokens);
        println!("Output tokens: {}", usage.output_tokens);
        println!("Total tokens: {}", usage.total_tokens);
    }
}
```

## Use Cases

### 1. Expert Panel Discussion

Create a panel of AI experts with different specializations:

```rust
let mut session = MultiParticipantSession::new(
    "You are an expert panelist discussing AI safety.".to_string(),
    8192,
    OrchestrationStrategy::ModeratorLed,
);

session.add_participant("Safety-Expert", safety_expert, ParticipantRole::Moderator);
session.add_participant("Ethics-Expert", ethics_expert, ParticipantRole::Panelist);
session.add_participant("Tech-Expert", tech_expert, ParticipantRole::Panelist);
```

### 2. Research Analysis Pipeline

Multiple workers analyze different aspects, supervisor synthesizes:

```rust
let mut session = MultiParticipantSession::new(
    "Analyze different aspects of the research paper.".to_string(),
    8192,
    OrchestrationStrategy::Hierarchical,
);

session.add_participant("Methodology-Analyst", worker1, ParticipantRole::Worker);
session.add_participant("Results-Analyst", worker2, ParticipantRole::Worker);
session.add_participant("Implications-Analyst", worker3, ParticipantRole::Worker);
session.add_participant("Senior-Researcher", supervisor, ParticipantRole::Supervisor);
```

### 3. Collaborative Problem Solving

Agents build on each other's ideas:

```rust
let mut session = MultiParticipantSession::new(
    "Collaborate to solve this problem, building on previous ideas.".to_string(),
    8192,
    OrchestrationStrategy::RoundRobin,
);

session.add_participant("Creative-Agent", creative, ParticipantRole::Panelist);
session.add_participant("Analytical-Agent", analytical, ParticipantRole::Panelist);
session.add_participant("Practical-Agent", practical, ParticipantRole::Panelist);
```

### 4. Multi-Model Consensus

Get opinions from multiple models and evaluate them:

```rust
let mut session = MultiParticipantSession::new(
    "Provide your best answer to the question.".to_string(),
    8192,
    OrchestrationStrategy::Broadcast,
);

// Add multiple different models
session.add_participant("GPT-4", gpt4, ParticipantRole::Panelist);
session.add_participant("Claude", claude, ParticipantRole::Panelist);
session.add_participant("Gemini", gemini, ParticipantRole::Panelist);
session.add_participant("Evaluator", evaluator_model, ParticipantRole::Evaluator);
```

## Running the Example

To run the provided example:

```bash
# Set your API keys
export OPEN_AI_SECRET="your-openai-api-key"
export XAI_API_KEY="your-grok-api-key"  # optional
export GEMINI_API_KEY="your-gemini-api-key"  # optional

# Run the example
cargo run --example multi_participant_example
```

The example demonstrates all orchestration strategies and shows how responses differ based on the strategy used.

## Best Practices

1. **Choose the Right Strategy**: Match the orchestration strategy to your use case
   - Use `Broadcast` for independent perspectives
   - Use `RoundRobin` for building on ideas
   - Use `ModeratorLed` for structured discussions
   - Use `Hierarchical` for complex analysis pipelines

2. **Monitor Token Usage**: Each participant accumulates tokens, so monitor usage:
   ```rust
   for response in responses {
       if let Some(usage) = response.token_usage {
           println!("Participant {} used {} tokens", 
               response.participant_name, usage.total_tokens);
       }
   }
   ```

3. **Role Assignment**: Assign appropriate roles to leverage strategy-specific behavior
   - Moderators control the flow in `ModeratorLed`
   - Workers and Supervisors have specific behaviors in `Hierarchical`
   - Priorities matter in `Custom` strategy

4. **System Prompts**: Craft system prompts that work for all participants or customize per participant through their individual `LLMSession` if needed

5. **Error Handling**: The session continues even if one participant fails:
   ```rust
   let responses = session.send_message(/* ... */).await?;
   // Check responses.len() to see how many succeeded
   ```

## Performance Considerations

- **Broadcast** sends requests in parallel (fastest for getting multiple opinions)
- **RoundRobin** is sequential (slowest but builds context)
- **ModeratorLed** has two phases (moderator first, then parallel to others)
- **Hierarchical** has two phases (workers parallel, then supervisors)
- **Custom** follows priority order (similar to RoundRobin)

Each participant maintains its own conversation history, allowing for context-aware responses in subsequent interactions.
