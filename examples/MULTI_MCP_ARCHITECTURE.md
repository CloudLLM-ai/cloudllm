# Multi-MCP Agent Architecture

## Vision

An agent that transparently accesses tools from multiple MCP servers as if they were all available locally.

```
┌─────────────────────────────────────────────────────────────────┐
│ Agent (using OpenAI, Claude, etc.)                              │
│                                                                  │
│  "Search YouTube for Rust tutorials and save results to memory" │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         │ has
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│ ToolRegistry                                                     │
│                                                                  │
│ Powered by: MultiMcpClientProtocol                              │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         │ routes through
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│ MultiMcpClientProtocol                                          │
│                                                                  │
│  Available MCP Servers:                                         │
│  ├─ "local"  → http://localhost:8080                           │
│  └─ "youtube" → http://youtube-mcp.example.com:8081            │
│                                                                  │
│  Tool Registry:                                                 │
│  ├─ memory → local                                              │
│  ├─ bash → local                                                │
│  ├─ youtube_search → youtube                                    │
│  └─ youtube_get_transcript → youtube                            │
└────────────────────────┬────────────────────────────────────────┘
                         │
         ┌───────────────┼───────────────┐
         │               │               │
         ▼               ▼               ▼
    ┌────────┐      ┌────────┐     ┌──────────┐
    │ Local  │      │ Local  │     │ YouTube  │
    │ MCP    │      │ MCP    │     │   MCP    │
    │Server  │      │Server  │     │  Server  │
    │        │      │        │     │          │
    │memory  │      │ bash   │     │youtube_  │
    │tool    │      │tool    │     │search    │
    │        │      │        │     │tool      │
    └────────┘      └────────┘     └──────────┘
```

## Implementation Flow

### 1. Create MultiMcpClientProtocol

```rust
let multi_mcp = MultiMcpClientProtocol::new();
```

**State after creation:**
- `servers`: empty HashMap
- `clients`: empty HashMap
- `tool_to_server`: empty HashMap
- `all_tools`: None

---

### 2. Register Local MCP Server

```rust
multi_mcp.add_mcp_server(
    "local",
    "http://localhost:8080".to_string()
).await?;
```

**What happens:**
1. Store server URL in `servers`
2. Create `McpClientProtocol` for that URL
3. Call `client.list_tools()` → returns `["memory", "bash"]`
4. Build `tool_to_server` mapping:
   - `"memory"` → `"local"`
   - `"bash"` → `"local"`
5. Cache tools in `all_tools`

**State after:**
```
servers = {
  "local": "http://localhost:8080"
}

clients = {
  "local": McpClientProtocol(...)
}

tool_to_server = {
  "memory": "local",
  "bash": "local"
}

all_tools = [
  ToolMetadata { name: "memory", ... },
  ToolMetadata { name: "bash", ... }
]
```

---

### 3. Register Remote YouTube MCP Server

```rust
multi_mcp.add_mcp_server(
    "youtube",
    "http://youtube-mcp.example.com:8081".to_string()
).await?;
```

**What happens:**
1. Store server URL
2. Create `McpClientProtocol` for that URL
3. Call `client.list_tools()` → returns `["youtube_search", "youtube_get_transcript"]`
4. Update `tool_to_server`:
   - `"youtube_search"` → `"youtube"`
   - `"youtube_get_transcript"` → `"youtube"`
5. Append tools to `all_tools`

**State after:**
```
servers = {
  "local": "http://localhost:8080",
  "youtube": "http://youtube-mcp.example.com:8081"
}

tool_to_server = {
  "memory": "local",
  "bash": "local",
  "youtube_search": "youtube",
  "youtube_get_transcript": "youtube"
}

all_tools = [
  ToolMetadata { name: "memory", ... },
  ToolMetadata { name: "bash", ... },
  ToolMetadata { name: "youtube_search", ... },
  ToolMetadata { name: "youtube_get_transcript", ... }
]
```

---

### 4. Agent Lists Available Tools

```rust
let tools = multi_mcp.list_tools().await?;
// Returns all 4 tools as if they were one registry
```

**Output:**
```
Available tools:
  - memory (Store and retrieve key-value data)
  - bash (Execute bash commands)
  - youtube_search (Search YouTube videos)
  - youtube_get_transcript (Get video transcript)
```

---

### 5. Agent Calls a Tool

**Scenario A: Agent calls `memory` tool**

```rust
multi_mcp.execute("memory", json!({
    "command": "P search_results [...results...]"
})).await?
```

**Internal flow:**
1. Look up `"memory"` in `tool_to_server` → returns `"local"`
2. Get client for `"local"` server
3. Call `client.execute("memory", params).await`
4. Local MCP server handles it
5. Return result to agent

**Scenario B: Agent calls `youtube_search` tool**

```rust
multi_mcp.execute("youtube_search", json!({
    "query": "Rust programming tutorial"
})).await?
```

**Internal flow:**
1. Look up `"youtube_search"` in `tool_to_server` → returns `"youtube"`
2. Get client for `"youtube"` server
3. Call `client.execute("youtube_search", params).await`
4. Remote YouTube MCP server handles it
5. Return results (videos) to agent

---

## Complete Agent Workflow

```rust
// Step 1: Create multi-MCP protocol
let multi_mcp = MultiMcpClientProtocol::new();

// Step 2: Register servers
multi_mcp.add_mcp_server("local", "http://localhost:8080").await?;
multi_mcp.add_mcp_server("youtube", "http://youtube-mcp:8081").await?;

// Step 3: Create registry
let registry = Arc::new(ToolRegistry::new(Arc::new(multi_mcp)));

// Step 4: Create agent
let mut agent = Agent::new(
    "researcher",
    "Research Assistant",
    openai_client
);
agent.tool_registry = Some(registry);

// Step 5: Agent uses tools transparently
agent.send_message(
    "Find the top 3 Rust tutorials on YouTube and save them to memory",
    None
).await?;

// Agent will:
// 1. Call youtube_search (routed to YouTube server)
// 2. Call memory PUT (routed to local server)
// 3. Return findings to user
```

---

## Benefits

1. **Transparency**: Agent code doesn't know about multiple servers
2. **Composition**: Add/remove servers dynamically
3. **Fault tolerance**: One server down doesn't crash agent (optional: skip that server)
4. **Scalability**: Add as many MCP servers as needed
5. **Type Safety**: Tool discovery and execution still type-safe

---

## Evolution

### Phase 1: Core (Implement First)
- `MultiMcpClientProtocol` struct
- `add_mcp_server()` method
- `tool_to_server` routing
- Basic error handling

### Phase 2: Improvements
- Server health checks
- Tool name conflict resolution (e.g., if two servers have same tool)
- Caching strategies
- Timeout handling per server
- Metrics (which tools called, latency, errors)

### Phase 3: Advanced
- Server priority (if tool exists on multiple servers, use preferred one)
- Failover (if primary server down, try backup)
- Tool aliasing (expose "search" as alias for "youtube_search")
- Rate limiting per server
- Load balancing

---

## Example Use Cases

### Use Case 1: Local + Cloud Tools

```
Local MCP Server:           Cloud MCP Server:
├─ filesystem              ├─ openai_api
├─ sqlite_db               ├─ stripe_payments
└─ bash                    └─ slack_notifications
```

Agent can orchestrate across both seamlessly.

### Use Case 2: Multiple Third-Party Services

```
YouTube MCP Server:         GitHub MCP Server:        Slack MCP Server:
├─ youtube_search          ├─ get_repo              ├─ send_message
├─ youtube_transcript      ├─ list_issues           ├─ create_channel
└─ youtube_stats           └─ create_pr             └─ update_thread
```

Agent monitors GitHub, posts updates to Slack, links to YouTube content.

### Use Case 3: Cascading Tool Execution

```
Agent request: "Find trending Rust content and summarize latest discussions"

1. Call youtube_search (remote) → finds video
2. Call youtube_get_transcript (remote) → gets transcript
3. Call bash (local) → processes text with NLP
4. Call memory (local) → stores summary
5. Call slack_notify (remote) → posts summary to team
```

All through one agent interface!
