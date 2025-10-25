//! Memory Session with Snapshots Example
//!
//! This example demonstrates how an agent can use the Memory tool to:
//! - Store important information about the task
//! - Create snapshots of progress at key milestones
//! - Save instructions that could be useful if the session restarts
//! - Track the state of a multi-step process
//!
//! The memory tool uses a token-efficient protocol that minimizes the overhead
//! of storing and retrieving session state.

use cloudllm::clients::openai::{Model, OpenAIClient};
use cloudllm::council::Agent;
use cloudllm::tool_protocol::ToolRegistry;
use cloudllm::tool_protocols::MemoryProtocol;
use cloudllm::tools::Memory;
use std::sync::Arc;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    cloudllm::init_logger();

    let api_key = std::env::var("OPEN_AI_SECRET").unwrap_or_else(|_| "sk-test".to_string());

    // Initialize the memory system
    let memory = Arc::new(Memory::new());

    // Create the memory tool adapter with the succinct protocol
    let memory_adapter = Arc::new(MemoryProtocol::new(memory.clone()));

    // Create a tool registry with the memory tool
    let registry = Arc::new(ToolRegistry::new(memory_adapter));

    // Create an OpenAI client
    let client = Arc::new(OpenAIClient::new_with_model_enum(
        &api_key,
        Model::GPT41Nano,
    ));

    // Create an agent with the memory tool registry
    let agent = Agent::new("summarizer", "Document Summarization Agent", client)
        .with_expertise(
            "Skilled at creating concise summaries of documents while preserving key information",
        )
        .with_personality("Methodical and detail-oriented, saves progress and creates checkpoints")
        .with_tools(registry);

    println!("=== Agent Configuration ===");
    println!("Agent ID: {}", agent.id);
    println!("Agent Name: {}", agent.name);
    println!(
        "Expertise: {}",
        agent.expertise.as_ref().unwrap_or(&"None".to_string())
    );
    println!("Has memory tool: {}", agent.tool_registry.is_some());

    // System prompt that teaches the agent about memory
    let system_prompt = format!(
        "You are a document summarization specialist.\n\n\
         You have access to a MEMORY tool for storing important information across sessions.\n\n\
         IMPORTANT: Always use the memory tool to:\n\
         1. Store the document name and type at the start\n\
         2. Create snapshots of your progress at key milestones\n\
         3. Save recovery instructions if needed\n\n\
         Memory Protocol Commands:\n\
         - Store: {{\"tool_call\": {{\"name\": \"memory\", \"parameters\": {{\"command\": \"P key value [ttl_seconds]\"}}}}}}\n\
         - Retrieve: {{\"tool_call\": {{\"name\": \"memory\", \"parameters\": {{\"command\": \"G key\"}}}}}}\n\
         - List: {{\"tool_call\": {{\"name\": \"memory\", \"parameters\": {{\"command\": \"L\"}}}}}}\n\
         - Delete: {{\"tool_call\": {{\"name\": \"memory\", \"parameters\": {{\"command\": \"D key\"}}}}}}\n\n\
         Protocol Specification:\n\
         {}\n\n\
         Example usage:\n\
         - Save document info: {{\"tool_call\": {{\"name\": \"memory\", \"parameters\": {{\"command\": \"P doc_name Important_Report 3600\"}}}}}}\n\
         - Save progress: {{\"tool_call\": {{\"name\": \"memory\", \"parameters\": {{\"command\": \"P milestone Section1_Complete 3600\"}}}}}}\n\
         - Retrieve progress: {{\"tool_call\": {{\"name\": \"memory\", \"parameters\": {{\"command\": \"G milestone\"}}}}}}\n",
        Memory::get_protocol_spec()
    );

    println!("\n=== System Prompt (Teaching Agent About Memory) ===");
    println!("{}\n", system_prompt);

    // Example user prompt that would trigger memory usage
    let user_prompt = "I have a 50-page technical document about cloud architecture. \
                       Please summarize it section by section, and use the memory tool to \
                       store your progress so that if you need to continue later, you can resume \
                       from where you left off. Start by storing the document metadata.";

    println!("=== User Prompt ===");
    println!("{}\n", user_prompt);

    println!("=== How The Agent Would Respond ===");
    println!("The agent would:");
    println!(
        "1. Store document metadata using memory: P doc_name Technical_Document_50_pages 3600"
    );
    println!("2. Process sections and create snapshots: P progress Section_1-10_Complete 3600");
    println!("3. Save recovery instructions: P recovery_point Continue_from_section_11 3600");
    println!("4. Query memory to verify stored state: G progress");
    println!("5. List all memory entries when needed: L META");

    println!("\n=== Direct Memory Operations Demo ===");

    // Demonstrate memory operations
    if let Some(tool_registry) = &agent.tool_registry {
        // Store initial task information
        println!("\nStoring document metadata...");
        match tool_registry
            .execute_tool(
                "memory",
                serde_json::json!({
                    "command": "P doc_name CloudArchitecture_50pages 3600"
                }),
            )
            .await
        {
            Ok(put_result) => println!("Result: {}", put_result.output),
            Err(e) => println!("Error: {}", e),
        }

        // Store first milestone
        println!("\nStoring first milestone...");
        match tool_registry
            .execute_tool(
                "memory",
                serde_json::json!({
                    "command": "P milestone Sections_1_to_10_summarized 3600"
                }),
            )
            .await
        {
            Ok(milestone_result) => println!("Result: {}", milestone_result.output),
            Err(e) => println!("Error: {}", e),
        }

        // Store recovery checkpoint
        println!("\nStoring recovery checkpoint...");
        match tool_registry
            .execute_tool(
                "memory",
                serde_json::json!({
                    "command": "P recovery_checkpoint Resume_from_section_11 3600"
                }),
            )
            .await
        {
            Ok(checkpoint_result) => println!("Result: {}", checkpoint_result.output),
            Err(e) => println!("Error: {}", e),
        }

        // List all stored entries
        println!("\nListing all memory entries...");
        match tool_registry
            .execute_tool(
                "memory",
                serde_json::json!({
                    "command": "L META"
                }),
            )
            .await
        {
            Ok(list_result) => {
                if let Ok(json_str) = serde_json::to_string_pretty(&list_result.output) {
                    println!("All entries: {}", json_str);
                }
            }
            Err(e) => println!("Error: {}", e),
        }

        // Retrieve a specific entry
        println!("\nRetrieving specific entry (milestone)...");
        match tool_registry
            .execute_tool(
                "memory",
                serde_json::json!({
                    "command": "G milestone META"
                }),
            )
            .await
        {
            Ok(get_result) => {
                if let Ok(json_str) = serde_json::to_string_pretty(&get_result.output) {
                    println!("Retrieved: {}", json_str);
                }
            }
            Err(e) => println!("Error: {}", e),
        }

        // Get memory usage
        println!("\nMemory usage statistics...");
        match tool_registry
            .execute_tool(
                "memory",
                serde_json::json!({
                    "command": "T A"
                }),
            )
            .await
        {
            Ok(usage_result) => println!("Total bytes used: {}", usage_result.output),
            Err(e) => println!("Error: {}", e),
        }
    }

    println!("\n=== Session Recovery Example ===");
    println!("If the session restarts, the agent would:");
    println!("1. Retrieve the document metadata: G doc_name");
    println!("2. Get the recovery checkpoint: G recovery_checkpoint");
    println!("3. Resume summarization from section 11");
    println!("4. Continue creating new snapshots as it progresses");

    println!("\n=== Key Benefits of Memory Tool ===");
    println!("✓ Token-efficient protocol minimizes LLM overhead");
    println!("✓ Automatic TTL-based expiration prevents stale data");
    println!("✓ Simple command syntax that LLMs learn quickly");
    println!("✓ Enables stateful agents that can recover from interruptions");
    println!("✓ Tracks multi-step processes without losing context");

    Ok(())
}
