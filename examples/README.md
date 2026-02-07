# CloudLLM Examples

Runnable demonstrations of CloudLLM patterns, tools, and multi-agent orchestration.

All examples use `cargo run --example <name>` and can run with placeholder API keys (no actual API calls made).

## Core Examples

### Agent Panel with Moderator & Tools
**File**: `agent_panel_with_moderator_and_access_to_tools.rs` ⭐ **NEW**

Four-agent system estimating global CO₂ emissions from Bitcoin mining with a two-round feedback loop.

**Demonstrates**:
- Four specialized workers (Data Collector, Energy Analyst, Emissions Analyst) + ChatGPT-5 Moderator
- Two-round iterative workflow with moderator feedback
- Shared Memory tool for coordination and audit trails
- Structured JSON outputs with units, sources, and uncertainty bounds
- Round-1 independence rule and Round-2 integration

**Setup**:
```bash
export GROK_API_KEY=xai-...
export OPENAI_API_KEY=sk-...
cargo run --example agent_panel_with_moderator_and_access_to_tools
```

**Output**: Memory KV store with 15+ keys organized in namespaces (r1/*, r2/*, final/*, meta/*, feedback/*)

---

## Orchestration & Multi-Agent Examples

### Orchestration Demo
**File**: `orchestration_demo.rs`

Comprehensive showcase of all orchestration collaboration modes.

**Demonstrates**:
- **Parallel mode**: Multiple agents respond simultaneously
- **RoundRobin mode**: Agents take turns sequentially
- **Moderated mode**: One agent orchestrates others
- **Hierarchical mode**: Layered problem-solving (workers → supervisors → executives)
- **Debate mode**: Iterative convergence with max rounds
- Custom tool registration and usage

**Setup**:
```bash
export OPENAI_API_KEY=sk-...
cargo run --example orchestration_demo
```

---

### Digimon vs Pokemon Debate
**File**: `digimon_vs_pokemon_debate.rs`

Fun debate example showcasing Moderated orchestration mode with three agents.

**Demonstrates**:
- Moderated orchestration with explicit moderator selection
- Multi-turn debate with narrative structure
- Different LLM providers (OpenAI, Anthropic)
- Topic-based agent personas

**Setup**:
```bash
export OPENAI_API_KEY=sk-...
export ANTHROPIC_API_KEY=claude-...
cargo run --example digimon_vs_pokemon_debate
```

**Note**: Takes 1-3 minutes due to sequential API calls

---

### Venezuela Regime Change Debate
**File**: `venezuela_regime_change_debate.rs`

Complex geopolitical debate showcasing orchestration negotiation patterns.

**Demonstrates**:
- Multi-round debate with convergence logic
- Competing strategic perspectives
- Debate mode for reaching consensus

---

### Orchestration with Memory
**File**: `orchestration_with_memory.rs`

Multi-agent orchestration coordinating through shared Memory tool.

**Demonstrates**:
- Shared Memory as coordination layer between agents
- Orchestration discussion with persistent decision-making
- Reading and writing from shared state
- Three-agent round-robin discussion

**Setup**:
```bash
export OPENAI_API_KEY=sk-...
cargo run --example orchestration_with_memory
```

---

## Tool Integration Examples

### Calculator Example
**File**: `calculator_example.rs`

Demonstrates the scientific Calculator tool with 30+ supported operations.

**Demonstrates**:
- Arithmetic, trigonometry, logarithms, statistics
- Expression evaluation with proper order of operations
- Error handling for invalid expressions

**Operations Supported**:
- Basic: `+`, `-`, `*`, `/`, `^`, `%`
- Trig: `sin()`, `cos()`, `tan()`, `asin()`, `acos()`, `atan()`
- Hyperbolic: `sinh()`, `cosh()`, `tanh()`
- Logarithmic: `ln()`, `log()`, `log2()`, `exp()`
- Statistical: `mean()`, `median()`, `std()`, `var()`, `sum()`, `count()`, `min()`, `max()`

---

### HTTP Client Example
**File**: `http_client_example.rs`

Demonstrates REST API client with security controls and all HTTP methods.

**Demonstrates**:
- GET, POST, PUT, DELETE, PATCH, HEAD methods
- Query parameters and custom headers
- Basic auth and Bearer token support
- Domain allowlist/blocklist security
- JSON payload handling
- Status code and response validation

**Features**:
- Automatic connection pooling
- Configurable timeouts (default 30s)
- Response size limits (default 10MB)
- Automatic HTTPS enforcement

---

### Bash Tool Example
**File**: `bash_tool_basic.rs`

Demonstrates secure cross-platform command execution.

**Demonstrates**:
- Command allowlist/denylist
- Timeout configuration (default 30s)
- Separate stdout/stderr capture
- Exit code handling
- Environment variable support
- Cross-platform (Linux/macOS)

**Allowed Commands**: `curl`, `wget`, `jq`, `sed`, `awk`, `bc`, `sha256sum`, `date`

---

### Memory Session with Snapshots
**File**: `memory_session_with_snapshots.rs`

Single-agent example using Memory for state persistence and recovery.

**Demonstrates**:
- TTL-aware key-value storage
- Session snapshots for recovery
- Automatic background expiration
- Metadata tracking (creation time, TTL)

**Succinct Protocol Commands**:
- `P key value ttl` — Put (store)
- `G key [META]` — Get (retrieve, optionally with metadata)
- `L` — List all keys
- `D key` — Delete
- `C` — Clear all
- `T key` — Get TTL status

---

## MCP & Remote Tool Examples

### Multi-MCP Agent
**File**: `multi_mcp_agent.rs`

Shows agent accessing tools from multiple MCP servers simultaneously.

**Demonstrates**:
- Multi-protocol ToolRegistry setup
- Adding protocols dynamically with `registry.add_protocol()`
- Tool routing to appropriate protocol
- Transparent tool discovery from multiple sources

**Architecture**:
```
Agent → ToolRegistry (Multi-Protocol)
  ├─ Local Protocol: memory, bash
  ├─ YouTube MCP: youtube_search, youtube_get_transcript
  └─ GitHub MCP: github_search_repos, github_get_issues
```

---

### MCP Memory Client
**File**: `mcp_memory_client.rs`

HTTP-based client for remote Memory servers.

**Demonstrates**:
- Client-server architecture for Memory
- HTTP PUT, GET, DELETE operations
- Connection pooling and timeout handling
- Remote agent coordination

**Setup**:
```bash
# Terminal 1: Start MCP Memory Server
cargo run --example mcp_memory_server

# Terminal 2: Run client
cargo run --example mcp_memory_client
```

---

### MCP Memory Server
**File**: `mcp_memory_server.rs`

HTTP server exposing Memory as a remote MCP service.

**Demonstrates**:
- axum HTTP server setup
- `/tools/list` and `/tools/execute` endpoints
- Memory protocol over HTTP
- Server deployment patterns

**Listens on**: `http://127.0.0.1:8082`

---

## Session & Streaming Examples

### Interactive Session
**File**: `interactive_session.rs`

REPL-style multi-turn conversation with a single LLM agent.

**Demonstrates**:
- LLMSession for maintaining conversation history
- Multi-turn context management
- Streaming message support
- Token budget tracking

**Setup**:
```bash
export XAI_API_KEY=xai-...
cargo run --example interactive_session
```

**Type**: Fully interactive with stdin/stdout

---

### Streaming Example
**File**: `streaming_example.rs`

Demonstrates token streaming for real-time response display.

**Demonstrates**:
- Streaming API responses
- Real-time token output
- Async stream handling
- Progressive rendering

---

### Interactive Streaming Session
**File**: `interactive_streaming_session.rs`

Combines REPL and streaming for live-updating multi-turn sessions.

**Demonstrates**:
- Interactive input with streaming responses
- Real-time message rendering
- Conversation history with streaming

---

## Quick Reference

### Run Any Example
```bash
# Basic execution
cargo run --example <name>

# With environment variables
OPENAI_API_KEY=sk-... cargo run --example orchestration_demo

# With logging
RUST_LOG=debug cargo run --example orchestration_with_memory

# Build all examples
cargo build --examples
```

### Environment Variables

| Variable | Used By | Example |
|----------|---------|---------|
| `OPENAI_API_KEY` | OpenAI-based examples | orchestration_demo, digimon_vs_pokemon_debate |
| `GROK_API_KEY` | Grok-based examples | agent_panel_with_moderator_and_access_to_tools, interactive_session |
| `ANTHROPIC_API_KEY` | Claude-based examples | digimon_vs_pokemon_debate |
| `GEMINI_API_KEY` | Gemini-based examples | (if enabled) |
| `RUST_LOG` | All examples | debug, info, trace |

### API Key Format
- **OpenAI**: `sk-...`
- **Grok (xAI)**: `xai-...`
- **Anthropic**: `claude-...`
- **Gemini**: API key from Google Cloud

All examples run without actual API keys (default to placeholders), but won't make external API calls without valid credentials.

---

## Example Patterns

### Pattern 1: Single Agent with Tools
```rust
let agent = Agent::new("id", "name", client)
    .with_tools(Arc::new(registry));
```

### Pattern 2: Multi-Agent Orchestration
```rust
let mut orchestration = Orchestration::new("id", "name")
    .with_mode(OrchestrationMode::RoundRobin);
orchestration.add_agent(agent1)?;
orchestration.add_agent(agent2)?;
let response = orchestration.discuss("prompt", num_rounds).await?;
```

### Pattern 3: Shared Memory Coordination
```rust
let memory = Arc::new(Memory::new());
let registry = Arc::new(ToolRegistry::new(
    Arc::new(MemoryProtocol::new(memory.clone()))
));
// All agents share the same memory registry
```

---

## Testing Examples

All examples can be tested without external APIs:
```bash
cargo test --doc           # Doc tests
cargo build --examples     # Compilation check
cargo clippy --example <name>  # Lint check
```

---

## Performance Notes

- **Local tools** (Memory, Calculator, Bash): < 1ms
- **MCP server tools**: 100-200ms (over HTTP)
- **LLM API calls**: 1-10 seconds (depends on model and token count)
- **Multi-agent orchestrations**: Linear in number of agents × rounds

---

**Last Updated**: Current session
**Total Examples**: 15
**Prerequisite Knowledge**: Rust async/await, tokio, serde_json
