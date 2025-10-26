# Multi-Protocol Agent Architecture Diagram

## Complete System Overview

This diagram shows how an Agent can register and use tools from:
1. **Local tools** (no MCP server needed, using succinct protocol)
2. **Multiple remote MCP servers** (HTTP-based)

```
┌──────────────────────────────────────────────────────────────────────────┐
│                                                                           │
│                          🤖 AGENT                                         │
│                    (OpenAI, Claude, Grok, etc.)                          │
│                                                                           │
│  "Find trending Rust repos on GitHub and save summary to memory"         │
│                                                                           │
└────────────────────────┬────────────────────────────────────────────────┘
                         │
                         │ sends prompt & tool requests
                         │
        ┌────────────────▼───────────────┐
        │   receives responses & results │
        │                                 │
        │  - Tool execution results       │
        │  - Tool availability list       │
        │  - Tool metadata                │
        │                                 │
        └────────────────┬────────────────┘
                         │
                         │ delegates tool operations to
                         │
        ┌────────────────▼────────────────────────────────────────────┐
        │                                                              │
        │                  TOOL REGISTRY                              │
        │                                                              │
        │  Mode: Multi-Protocol (0.5.0+)                             │
        │  ═══════════════════════════════════════════════════════   │
        │                                                              │
        │  Connected Protocols:                                       │
        │  ├─ local       → CustomToolProtocol + MemoryProtocol      │
        │  ├─ github      → McpClientProtocol (HTTP)                 │
        │  ├─ youtube     → McpClientProtocol (HTTP)                 │
        │  └─ slack       → McpClientProtocol (HTTP)                 │
        │                                                              │
        │  Tool Registry (auto-discovered):                           │
        │  ├─ memory (P/G/L/D/C/T/SPEC)      → local                │
        │  ├─ bash (commands)                 → local                │
        │  ├─ github_search_repos             → github               │
        │  ├─ github_get_issues               → github               │
        │  ├─ youtube_search                  → youtube              │
        │  ├─ youtube_get_transcript          → youtube              │
        │  ├─ slack_send_message              → slack                │
        │  └─ slack_create_thread             → slack                │
        │                                                              │
        │  Routing Map (tool_name → protocol):                        │
        │  ╔═══════════════════════════════════════════════════════╗ │
        │  ║  memory → local                                       ║ │
        │  ║  bash → local                                         ║ │
        │  ║  github_* → github                                    ║ │
        │  ║  youtube_* → youtube                                  ║ │
        │  ║  slack_* → slack                                      ║ │
        │  ╚═══════════════════════════════════════════════════════╝ │
        │                                                              │
        └────────────┬────────────────┬───────────────┬───────────────┘
                     │                │               │
         ┌───────────▼──┐  ┌──────────▼──┐  ┌────────▼──────┐
         │ TOOL ROUTER  │  │ TOOL ROUTER  │  │ TOOL ROUTER   │
         │   (Local)    │  │   (GitHub)   │  │  (YouTube)    │
         │              │  │              │  │               │
         │ Decision:    │  │ Decision:    │  │ Decision:     │
         │ "Is tool in  │  │ "Is tool in  │  │ "Is tool in   │
         │  our map?"   │  │  our map?"   │  │  our map?"     │
         └──────────────┘  └──────────────┘  └────────────────┘
                     │                │               │
         ┌───────────▼──┐  ┌──────────▼──┐  ┌────────▼──────┐
         │              │  │              │  │               │
         │   LOCAL      │  │   GITHUB MCP │  │ YOUTUBE MCP   │
         │   TOOLS      │  │   SERVER     │  │ SERVER        │
         │   (in-proc)  │  │   (HTTP)     │  │ (HTTP)        │
         │              │  │              │  │               │
         │ ┌──────────┐ │  │ ┌──────────┐ │  │ ┌──────────┐ │
         │ │ Memory   │ │  │ │ GitHub   │ │  │ │ YouTube  │ │
         │ │ Protocol │ │  │ │ Tools    │ │  │ │ Tools    │ │
         │ │          │ │  │ │          │ │  │ │          │ │
         │ │ Succinct │ │  │ │ search_  │ │  │ │ search   │ │
         │ │ Protocol │ │  │ │ repos    │ │  │ │          │ │
         │ │ P/G/L... │ │  │ │          │ │  │ │ get_     │ │
         │ │          │ │  │ │ get_     │ │  │ │ transcript
         │ ├──────────┤ │  │ │ issues   │ │  │ │          │ │
         │ │ Bash     │ │  │ │          │ │  │ └──────────┘ │
         │ │ Tool     │ │  │ │ etc.     │ │  │              │
         │ │          │ │  │ │          │ │  │ Routed via   │
         │ │ execute  │ │  │ │          │ │  │ HTTP POST    │
         │ │ commands │ │  │ │ Routed   │ │  │ /execute     │
         │ │          │ │  │ │ via HTTP │ │  │              │
         │ └──────────┘ │  │ │ POST     │ │  │              │
         │              │  │ │ /execute │ │  │              │
         │ NO NETWORK   │  │ │          │ │  │ NETWORK I/O  │
         │ LATENCY!     │  │ │ NETWORK  │ │  │ (50-200ms)   │
         │              │  │ │ I/O      │ │  │              │
         │ Succinct     │  │ │ (50-200  │ │  │ Tool Result: │
         │ Protocol:    │  │ │ ms)      │ │  │ ┌──────────┐ │
         │ ┌──────────┐ │  │ │          │ │  │ │ JSON:    │ │
         │ │P key val │ │  │ │ Tool     │ │  │ │ {videos: │ │
         │ │  [ttl]   │ │  │ │ Result:  │ │  │ │  [{...}, │ │
         │ │          │ │  │ │ ┌──────┐ │ │  │ │   {...}] │ │
         │ │G key META│ │  │ │ │ JSON:│ │ │  │ │ }        │ │
         │ │          │ │  │ │ │{repos│ │ │  │ │          │ │
         │ │L         │ │  │ │ │ [{..}│ │ │  │ │ Task:    │ │
         │ │          │ │  │ │ │ {...}│ │ │  │ │ Extract  │ │
         │ │D key     │ │  │ │ │]}   │ │ │  │ │ video    │ │
         │ │          │ │  │ │ └──────┘ │ │  │ │ data     │ │
         │ │C         │ │  │ │          │ │  │ │ from     │ │
         │ │          │ │  │ │ Task:    │ │  │ │ results  │ │
         │ │T         │ │  │ │ Find     │ │  │ │          │ │
         │ │          │ │  │ │ repos by │ │  │ └──────────┘ │
         │ │SPEC key  │ │  │ │ keyword  │ │  │              │
         │ │          │ │  │ │ "rust"   │ │  │              │
         │ └──────────┘ │  │ │          │ │  │              │
         │              │  │ └──────────┘ │  │              │
         │              │  │              │  │              │
         │ Task:        │  │              │  │              │
         │ Store key-   │  │              │  │              │
         │ value pairs  │  │ Task:        │  │              │
         │ with TTL     │  │ Query GitHub │  │              │
         │              │  │ GraphQL API  │  │              │
         │              │  │              │  │              │
         └──────────────┘  └──────────────┘  └────────────────┘
```

## Data Flow Example

### Agent Task
**"Find trending Rust repos, fetch top 5 issue titles, and save summary to memory"**

### Step-by-Step Execution

```
1. AGENT MESSAGE RECEPTION
   ├─ LLM generates tool calls
   ├─ Detects multiple tool calls needed:
   │  ├─ github_search_repos("rust", sort: "stars")
   │  ├─ memory("P repos_found [data] 3600")
   │  ├─ github_get_issues("repository_id")
   │  └─ memory("P top_issues [data] 3600")
   │
   └─ Sends all to ToolRegistry

2. TOOL REGISTRY RECEIVES CALLS
   ├─ github_search_repos
   │  ├─ Lookup: "github_search_repos" → route to "github" protocol
   │  ├─ McpClientProtocol::execute()
   │  ├─ HTTP POST http://github-mcp:8082/execute
   │  │  {
   │  │    "tool": "github_search_repos",
   │  │    "parameters": {"query": "rust", "sort": "stars"}
   │  │  }
   │  ├─ Wait for response (~150ms network latency)
   │  └─ Return: { repos: [{name, url, stars}, ...] }
   │
   ├─ memory("P repos_found ...")
   │  ├─ Lookup: "memory" → route to "local" protocol
   │  ├─ MemoryProtocol::execute()
   │  ├─ In-process, NO network latency
   │  ├─ Store key "repos_found" with TTL 3600s
   │  └─ Return: { success: true }
   │
   ├─ github_get_issues
   │  ├─ Lookup: "github_get_issues" → route to "github" protocol
   │  ├─ McpClientProtocol::execute()
   │  ├─ HTTP POST http://github-mcp:8082/execute
   │  └─ Return: { issues: [{title, number}, ...] }
   │
   └─ memory("P top_issues ...")
      ├─ Lookup: "memory" → route to "local" protocol
      ├─ MemoryProtocol::execute()
      ├─ In-process, NO network latency
      └─ Return: { success: true }

3. RESULTS AGGREGATED
   └─ All results returned to agent in single batch
      {
        "tool_results": [
          {"tool": "github_search_repos", "result": {...}, "time_ms": 145},
          {"tool": "memory_put_1", "result": {...}, "time_ms": 2},
          {"tool": "github_get_issues", "result": {...}, "time_ms": 156},
          {"tool": "memory_put_2", "result": {...}, "time_ms": 1}
        ]
      }

4. AGENT PROCESSES RESULTS
   └─ Continues conversation with LLM using results
      "Based on the GitHub search results and issues, here's a summary..."
```

## Code Implementation

### Creating the Multi-Protocol Registry

```rust
use cloudllm::tool_protocol::ToolRegistry;
use cloudllm::tool_protocols::{CustomToolProtocol, McpClientProtocol, MemoryProtocol};
use cloudllm::tools::Memory;
use std::sync::Arc;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Step 1: Create empty registry for multi-protocol mode
    let mut registry = ToolRegistry::empty();

    // Step 2: Add LOCAL tools (no MCP server needed!)
    // ═════════════════════════════════════════════════
    let memory = Arc::new(Memory::new());
    let memory_protocol = Arc::new(MemoryProtocol::new(memory));
    registry.add_protocol("local", memory_protocol).await?;

    // The "local" protocol now provides:
    // - memory (succinct protocol: P/G/L/D/C/T/SPEC commands)
    // - bash (if we added it)

    // Step 3: Add GITHUB MCP SERVER (HTTP-based)
    // ════════════════════════════════════════════
    let github_mcp = Arc::new(McpClientProtocol::new(
        "http://github-mcp.example.com:8082".to_string()
    ));
    registry.add_protocol("github", github_mcp).await?;

    // The "github" protocol now provides:
    // - github_search_repos
    // - github_get_issues
    // - github_get_pull_requests
    // - etc.

    // Step 4: Add YOUTUBE MCP SERVER (HTTP-based)
    // ═══════════════════════════════════════════════
    let youtube_mcp = Arc::new(McpClientProtocol::new(
        "http://youtube-mcp.example.com:8081".to_string()
    ));
    registry.add_protocol("youtube", youtube_mcp).await?;

    // The "youtube" protocol now provides:
    // - youtube_search
    // - youtube_get_transcript
    // - youtube_get_stats
    // - etc.

    // Step 5: Add SLACK MCP SERVER (HTTP-based)
    // ═════════════════════════════════════════════
    let slack_mcp = Arc::new(McpClientProtocol::new(
        "http://slack-mcp.example.com:8083".to_string()
    ));
    registry.add_protocol("slack", slack_mcp).await?;

    // The "slack" protocol now provides:
    // - slack_send_message
    // - slack_create_thread
    // - slack_update_channel
    // - etc.

    // Step 6: Attach to Agent
    // ═══════════════════════════════════════════
    let agent = Agent::new("researcher", "Research Agent", client)
        .with_expertise("Finding information from multiple sources")
        .with_personality("Thorough and methodical")
        .with_tools(Arc::new(registry));  // All tools now available!

    // Step 7: Agent can now use ANY tool transparently!
    // ═════════════════════════════════════════════════════
    agent.send_message(
        "Find trending Rust repos on GitHub, save to memory, \
         get transcripts from top Rust videos on YouTube, \
         and post summary to Slack"
    ).await?;

    // What happens internally:
    // 1. Registry discovers all tools from all 4 protocols
    // 2. Agent sees: ~15 tools from different sources
    // 3. Agent calls tools transparently:
    //    - Local memory tool: NO network latency (~1-2ms)
    //    - GitHub tools: HTTP to remote server (~150ms)
    //    - YouTube tools: HTTP to remote server (~150ms)
    //    - Slack tools: HTTP to remote server (~100ms)
    // 4. All results aggregated and returned to agent
    // 5. Agent continues conversation naturally

    Ok(())
}
```

## Protocol Comparison

```
┌─────────────────────────────────────────────────────────────────────┐
│                    LOCAL PROTOCOL CHARACTERISTICS                   │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│ ✓ Succinct Protocol (P/G/L/D/C/T/SPEC)                             │
│   - P key value [ttl]        → PUT (store with TTL)                │
│   - G key META               → GET (retrieve)                      │
│   - L                        → LIST all keys                       │
│   - D key                    → DELETE key                          │
│   - C                        → CLEAR all                           │
│   - T                        → TTL info                            │
│   - SPEC key                 → Get key specification               │
│                                                                     │
│ ✓ In-Process Execution                                             │
│   - No network I/O needed                                          │
│   - Latency: ~1-2ms (vs 100-200ms over network)                    │
│   - Perfect for state management & caching                         │
│                                                                     │
│ ✓ Thread-Safe                                                      │
│   - Uses Arc<RwLock<>> for concurrent access                       │
│   - Multiple agents can access same memory                         │
│                                                                     │
│ ✓ TTL-Aware                                                        │
│   - Automatic background expiration                                │
│   - Set-and-forget semantics                                       │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                   REMOTE MCP PROTOCOL CHARACTERISTICS               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│ ✓ HTTP-Based Communication                                         │
│   - Uses standard HTTP POST for tool calls                         │
│   - JSON request/response format                                   │
│   - Supports any tool implementation                               │
│                                                                     │
│ ✓ Network Latency                                                  │
│   - Typically 50-200ms per request                                 │
│   - Depends on network conditions & server response time           │
│   - Consider for non-critical or asynchronous operations           │
│                                                                     │
│ ✓ Scalability                                                      │
│   - Run MCP server on separate machine/container                   │
│   - Multiple agents can query same server                          │
│   - Independent server scaling                                     │
│                                                                     │
│ ✓ Tool Variety                                                     │
│   - GitHub: repositories, issues, PRs                              │
│   - YouTube: search, transcripts, metadata                         │
│   - Slack: messaging, channels, threads                            │
│   - Custom: any HTTP-based tool                                    │
│                                                                     │
│ ✓ Distributed                                                      │
│   - Servers can run anywhere (cloud, on-prem, edge)               │
│   - Agent queries multiple distributed services                    │
│   - Central coordination via agent                                 │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

## Performance Characteristics

### Latency Breakdown

```
Agent Message to Results (Example: Query GitHub + Save to Memory)

┌────────────────────────────────────────────────────────────────┐
│ OPERATION: github_search_repos                                 │
│                                                                │
│ Timeline:                                                      │
│ ├─ Lookup routing: "github_search_repos" → "github" : 0.1ms  │
│ ├─ Network request prep: 0.5ms                               │
│ ├─ HTTP POST to github-mcp:8082 : 0.3ms                      │
│ ├─ Server processing: 120ms                                  │
│ ├─ HTTP response transmission: 5ms                           │
│ ├─ Parse JSON response: 2ms                                  │
│ └─ Return to agent: 0.1ms                                    │
│                                                                │
│ TOTAL: ~128ms                                                 │
└────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────┐
│ OPERATION: memory("P search_results ...")                      │
│                                                                │
│ Timeline:                                                      │
│ ├─ Lookup routing: "memory" → "local" : 0.1ms               │
│ ├─ Execute in-process: 1.5ms                                 │
│ │  ├─ Parse succinct protocol: 0.3ms                         │
│ │  ├─ Store in HashMap: 0.9ms                                │
│ │  ├─ Set TTL timer: 0.3ms                                   │
│ │  └─ Return result: 0.1ms                                   │
│ └─ Return to agent: 0.1ms                                    │
│                                                                │
│ TOTAL: ~1.7ms                                                 │
└────────────────────────────────────────────────────────────────┘

Savings: 128ms - 1.7ms = ~126ms faster for local operations!
```

## Tool Discovery

### When Registry is Created

```rust
// Single protocol (traditional)
let registry = ToolRegistry::new(protocol);
// Tools NOT automatically discovered
// Call: registry.discover_tools_from_primary().await?;

// Multi-protocol (new in 0.5.0)
let mut registry = ToolRegistry::empty();
registry.add_protocol("local", local_proto).await?;
//     ↓
//     └─ Automatically calls local_proto.list_tools()
//        └─ Returns: ["memory", "bash"]
//           └─ Registered in registry

registry.add_protocol("github", github_proto).await?;
//     ↓
//     └─ Automatically calls github_proto.list_tools()
//        └─ Returns: ["github_search_repos", "github_get_issues", ...]
//           └─ Registered in registry

registry.add_protocol("youtube", youtube_proto).await?;
//     ↓
//     └─ Automatically calls youtube_proto.list_tools()
//        └─ Returns: ["youtube_search", "youtube_get_transcript", ...]
//           └─ Registered in registry
```

## Summary

The multi-protocol agent architecture enables:

1. **Local Tools** - Fast, in-process tools using succinct protocols (0.5-2ms latency)
2. **Remote Tools** - Distributed MCP servers for external integrations (50-200ms latency)
3. **Transparent Routing** - Agent doesn't care where tool comes from
4. **Automatic Discovery** - Tools auto-discovered on protocol registration
5. **Unified Interface** - ToolRegistry handles all routing and execution
6. **Scalability** - Add/remove protocols dynamically at runtime

This enables powerful multi-source agent orchestration where agents can seamlessly coordinate across local and remote services!
