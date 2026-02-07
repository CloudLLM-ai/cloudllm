# CloudLLM Memory Tool Guide

The Memory tool is a persistent, TTL-aware key-value store designed for agent state management. It enables agents to maintain context across sessions, create recovery checkpoints, and coordinate shared state in multi-agent scenarios.

## Overview

The Memory tool provides:
- **Persistent Storage**: Key-value pairs survive across LLM calls
- **TTL-Based Expiration**: Automatic cleanup of old data
- **Metadata Tracking**: Know when data was stored and when it expires
- **Token Efficiency**: Minimal overhead in LLM prompts
- **Easy Integration**: Works with single agents or orchestrations

## Core Concepts

### 1. Single-Agent Memory Usage

An agent can use Memory to:
- Store task metadata and progress
- Create snapshots at milestones
- Save recovery instructions for session restart
- Track state of multi-step processes

### 2. Multi-Agent Shared Memory

Multiple agents in an Orchestration can share Memory to:
- Coordinate decisions across a team
- Build on each other's analysis
- Maintain a shared record of conclusions
- Track discussion state for recovery

### 3. TTL (Time-To-Live)

Each memory entry can have an expiration time:
- Entries without TTL persist indefinitely
- Entries with TTL are automatically deleted when expired
- Default cleanup interval: 1 second

## Setup

### Basic Usage

```rust
use cloudllm::tools::Memory;
use cloudllm::tool_adapters::MemoryToolAdapter;
use cloudllm::tool_protocol::ToolRegistry;
use std::sync::Arc;

// Create memory instance
let memory = Arc::new(Memory::new());

// Create adapter for tool protocol
let adapter = Arc::new(MemoryToolAdapter::new(memory));

// Create tool registry
let registry = Arc::new(ToolRegistry::new(adapter));
```

### With an Agent

```rust
use cloudllm::Agent;

let agent = Agent::new("analyzer", "Data Analyzer", client)
    .with_tools(registry);
```

### With an Orchestration

```rust
use cloudllm::orchestration::Orchestration;

let mut orchestration = Orchestration::new("orchestration", "Analysis Orchestration")
    .with_mode(OrchestrationMode::RoundRobin);

// Create shared memory
let shared_memory = Arc::new(Memory::new());
let shared_adapter = Arc::new(MemoryToolAdapter::new(shared_memory));
let shared_registry = Arc::new(ToolRegistry::new(shared_adapter));

// All agents get the same registry
let agent1 = Agent::new("a1", "Analyst 1", client1).with_tools(shared_registry.clone());
let agent2 = Agent::new("a2", "Analyst 2", client2).with_tools(shared_registry.clone());

orchestration.add_agent(agent1)?;
orchestration.add_agent(agent2)?;
```

## Protocol Reference

The Memory tool uses a simple, token-efficient protocol. Agents communicate via tool calls:

```json
{
  "tool_call": {
    "name": "memory",
    "parameters": {
      "command": "P doc_name Important_Report 3600"
    }
  }
}
```

### Commands

#### Put (P): Store Data
Store a key-value pair with optional TTL.

```
P <key> <value> [ttl_seconds]
```

**Examples:**
```
P doc_name My_Document            # No expiration
P milestone Step_1_Complete 3600  # Expires in 1 hour
P counter 42 60                   # Expires in 60 seconds
```

#### Get (G): Retrieve Data
Retrieve a value with optional metadata.

```
G <key> [META]
```

**Examples:**
```
G doc_name        # Get value only
G milestone META  # Get value with creation time and TTL info
```

**Response:**
```json
{
  "value": "Important_Report"
}
```

Or with metadata:
```json
{
  "value": "Step_1_Complete",
  "added_utc": "2024-11-25T14:30:00Z",
  "expires_in": 3600
}
```

#### List (L): List All Keys
List all stored keys with optional metadata.

```
L [META]
```

**Examples:**
```
L      # List keys only
L META # List with metadata
```

**Response:**
```json
{
  "keys": ["doc_name", "milestone", "checkpoint"]
}
```

#### Delete (D): Remove Data
Remove a specific key from memory.

```
D <key>
```

**Example:**
```
D old_checkpoint
```

#### Clear (C): Clear All Data
Remove all stored entries.

```
C
```

#### Total Bytes (T): Check Memory Usage
Get total memory consumption.

```
T <scope>
```

Scopes:
- `A` (all): Total bytes
- `K` (keys): Bytes used by keys
- `V` (values): Bytes used by values

**Examples:**
```
T A  # Total usage
T K  # Keys size
T V  # Values size
```

**Response:**
```json
{
  "total_bytes": 512
}
```

#### SPEC: Get Protocol Specification
Request the protocol specification (useful for LLM learning).

```
SPEC
```

## System Prompt Examples

### Single Agent with Memory

```
You are a document summarization assistant.

You have access to a memory system for tracking your progress.
Use it to:
1. Store document metadata when you start
2. Create snapshots after each major section
3. Save recovery checkpoints for continuation

Memory Protocol:
- Store: {"tool_call": {"name": "memory", "parameters": {"command": "P key value ttl"}}}
- Retrieve: {"tool_call": {"name": "memory", "parameters": {"command": "G key META"}}}
- List: {"tool_call": {"name": "memory", "parameters": {"command": "L META"}}}

Example workflow:
1. Start: P doc_name TechReport_50pages 7200
2. Progress: P current_section Section_3_of_10 7200
3. Checkpoint: P recovery_point Resume_at_section_4 7200
4. Check progress: G current_section META
```

### Orchestration with Shared Memory

```
You are part of a strategic decision orchestration.

All orchestration members share a memory system. Use it to:
1. Record your analysis and findings
2. Review other members' contributions
3. Build consensus decisions

Memory is shared across all agents. One agent's storage is visible to all.

Example:
- Analyst stores: P analysis_findings Data_shows_trend_up 3600
- Strategist reads: G analysis_findings META
- Implementer records: P decision Proceed_with_plan 3600
```

## Usage Examples

### Example 1: Document Processing with Snapshots

```rust
// Agent stores document metadata
registry.execute_tool("memory",
    serde_json::json!({
        "command": "P doc_name TechArticle_25pages 7200"
    })).await?;

// Agent updates progress
registry.execute_tool("memory",
    serde_json::json!({
        "command": "P current_section Pages_1_to_5_summarized 7200"
    })).await?;

// Agent checks current status
let result = registry.execute_tool("memory",
    serde_json::json!({
        "command": "G current_section"
    })).await?;
println!("Progress: {}", result.output["value"]);

// Agent saves recovery point
registry.execute_tool("memory",
    serde_json::json!({
        "command": "P recovery_point Resume_at_page_6 7200"
    })).await?;
```

### Example 2: Orchestration Decision Tracking

```rust
// All agents share this memory
let shared_memory = Arc::new(Memory::new());
let shared_adapter = Arc::new(MemoryToolAdapter::new(shared_memory));
let shared_registry = Arc::new(ToolRegistry::new(shared_adapter));

// Analyst stores findings
shared_registry.execute_tool("memory",
    serde_json::json!({
        "command": "P analyst_findings Option_A_has_best_ROI 3600"
    })).await?;

// Strategist reviews and adds perspective
let findings = shared_registry.execute_tool("memory",
    serde_json::json!({
        "command": "G analyst_findings"
    })).await?;

shared_registry.execute_tool("memory",
    serde_json::json!({
        "command": "P strategy_assessment Aligns_with_5year_plan 3600"
    })).await?;

// Implementer makes decision
let strategy = shared_registry.execute_tool("memory",
    serde_json::json!({
        "command": "G strategy_assessment"
    })).await?;

shared_registry.execute_tool("memory",
    serde_json::json!({
        "command": "P final_decision CONSENSUS_on_Option_A 7200"
    })).await?;

// Review all stored decisions
let all = shared_registry.execute_tool("memory",
    serde_json::json!({
        "command": "L META"
    })).await?;
println!("Decision record: {}", serde_json::to_string_pretty(&all.output)?);
```

### Example 3: TTL-Based Cleanup

```rust
// Store temporary checkpoint that expires in 60 seconds
registry.execute_tool("memory",
    serde_json::json!({
        "command": "P temp_state Processing 60"
    })).await?;

// Store permanent record that never expires
registry.execute_tool("memory",
    serde_json::json!({
        "command": "P final_result Completed"
    })).await?;

// After 61 seconds, temp_state is automatically deleted
tokio::time::sleep(Duration::from_secs(61)).await;

let keys = registry.list_tools().await?;
// final_result still exists, temp_state is gone
```

## Best Practices

### 1. Naming Conventions
Use descriptive, understandable names for keys:
```
✓ P current_milestone Section_2_Complete
✓ P recovery_checkpoint Resume_at_section_3
✗ P m2  # Too vague
✗ P x 42
```

### 2. TTL Strategy
- **Short TTL (60-300s)**: Temporary state during processing
- **Medium TTL (600-3600s)**: Progress checkpoints
- **No TTL**: Important decisions and final results

### 3. Key Organization
Use prefixes to organize related data:
```
P task_description Summarize_report
P task_status InProgress
P task_section Current_page_5
P task_recovery Resume_page_6

G task_description  # Easy to retrieve related items
L                   # All task_ keys visible together
```

### 4. Memory Efficiency
Keep values concise to minimize token usage:
```
✓ P status Done              # 4 tokens
✗ P status Task is complete  # 6 tokens
```

### 5. Coordination in Teams
Use consistent naming for shared memory:
```
// All agents know to look for these keys
P analyst_findings ...
P strategist_perspective ...
P implementer_assessment ...
P team_decision ...
```

## Performance Considerations

- **Storage**: No practical limit (system RAM dependent)
- **Lookup**: O(1) for Get, O(n) for List
- **Expiration**: Checked on each Get/List, background cleanup every 1 second
- **Token Cost**: Minimal - simple key-value operations

## Common Patterns

### Pattern 1: Resumable Tasks
```rust
// Start
P task_id unique_123 7200
P progress Step_1 7200

// Mid-session check recovery
if interrupted {
    let progress = G progress;
    // Resume from progress
}
```

### Pattern 2: Voting/Consensus
```rust
// Agent 1 votes
P vote_agent1 option_A 3600

// Agent 2 votes
P vote_agent2 option_A 3600

// Agent 3 tallies
L  # See all votes
// Majority is option_A
P consensus option_A 3600
```

### Pattern 3: Hierarchical Decisions
```rust
// Workers analyze
P worker1_analysis Finding_X 1800
P worker2_analysis Finding_Y 1800

// Supervisor aggregates
L META
P supervisor_summary Both_X_and_Y_important 3600

// Executive decides
G supervisor_summary
P executive_decision Proceed_with_both 7200
```

## Troubleshooting

### Data Not Found
- Check key spelling and case sensitivity
- Verify TTL hasn't expired: `G key META`
- List all keys: `L META`

### Session Recovery
- Save recovery checkpoint before processing: `P recovery_point <state>`
- On restart, retrieve: `G recovery_point`
- Resume from saved point

### Memory Growing Too Large
- Review TTLs - add expiration for temporary data
- Use `T A` to check total usage
- Clear old entries: `D old_key` or `C` to clear all

## Full Working Examples

See the complete examples in the `examples/` directory:

1. **`memory_session_with_snapshots.rs`**: Single agent using memory for progress tracking
2. **`orchestration_with_memory.rs`**: Multiple agents coordinating via shared memory

Run examples:
```bash
cargo run --example memory_session_with_snapshots
cargo run --example orchestration_with_memory
```

## API Reference

### Memory Struct

```rust
impl Memory {
    pub fn new() -> Self
    pub fn put(&self, key: String, value: String, ttl: Option<u64>)
    pub fn get(&self, key: &str, include_metadata: bool) -> Option<(String, Option<MemoryMetadata>)>
    pub fn delete(&self, key: &str) -> bool
    pub fn list_keys(&self) -> Vec<String>
    pub fn clear(&self)
    pub fn get_total_bytes_stored(&self) -> (usize, usize, usize)
    pub fn get_protocol_spec() -> String
}
```

### MemoryToolAdapter Struct

```rust
impl MemoryToolAdapter {
    pub fn new(memory: Arc<Memory>) -> Self
    fn process_memory_command(&self, command: &str) -> ToolResult
}

impl ToolProtocol for MemoryToolAdapter {
    async fn execute(
        &self,
        tool_name: &str,
        parameters: serde_json::Value,
    ) -> Result<ToolResult, Box<dyn Error + Send + Sync>>

    async fn list_tools(&self) -> Result<Vec<ToolMetadata>, Box<dyn Error + Send + Sync>>
    async fn get_tool_metadata(&self, tool_name: &str) -> Result<ToolMetadata, Box<dyn Error + Send + Sync>>
    fn protocol_name(&self) -> &str
}
```

### MemoryMetadata Struct

```rust
pub struct MemoryMetadata {
    pub added_utc: DateTime<Utc>,
    pub expires_in: Option<u64>,
}
```

## Integration with Other CloudLLM Features

The Memory tool works seamlessly with:

- **Agents**: Each agent can have a memory registry
- **Orchestrations**: All orchestration members can share memory
- **Sessions**: Memory persists across LLMSession calls
- **Tool Registry**: Memory is registered like any other tool
- **Custom Protocols**: Can be adapted for other communication protocols
