# thoughtchain

`thoughtchain` is a standalone Rust crate for durable agent memory.

It stores semantically typed thoughts in an append-only, hash-chained memory
log through a swappable storage adapter layer. The current default backend is
binary, but the chain model is no longer tied to that format. Agents can:

- persist important insights, decisions, constraints, and checkpoints
- record retrospectives and lessons learned after hard failures or non-obvious fixes
- relate new thoughts to earlier thoughts with typed graph edges
- query memory by type, role, agent, tags, concepts, text, and importance
- reconstruct context for agent resumption
- export a Markdown memory view that can back `MEMORY.md`, MCP, REST, or CLI flows

The crate is intentionally independent from `cloudllm` so it can be embedded in
other agent systems without creating circular dependencies.

## Quick Start

If you just want the daemon:

```bash
cargo install thoughtchain
thoughtchaind
```

If you want to leave it running after closing your SSH session:

```bash
nohup thoughtchaind &
```

## What Is In This Folder

`thoughtchain/` contains:

- the standalone `thoughtchain` library crate
- server support for HTTP MCP and REST, enabled by default
- the `thoughtchaind` daemon binary
- dedicated tests under `thoughtchain/tests`

## Build

From inside `thoughtchain/`:

```bash
cargo build
```

Build only the library without the default daemon/server stack:

```bash
cargo build --no-default-features
```

## Test

Run the crate tests:

```bash
cargo test
```

Run tests for the library-only build:

```bash
cargo test --no-default-features
```

Run rustdoc tests:

```bash
cargo test --doc
```

## Generate Docs

Build local Rust documentation:

```bash
cargo doc --no-deps
```

Generate docs for the library-only build:

```bash
cargo doc --no-deps --no-default-features
```

## Run The Daemon

The standalone daemon binary is `thoughtchaind`.

Run it from source:

```bash
cargo run --bin thoughtchaind
```

Install it from the crate directory while working in this repo:

```bash
cargo install --path . --locked
```

When it starts, it serves both:

- an MCP server
- a REST server

Before serving traffic, it:

- migrates or reconciles discovered chains to the current schema and default storage adapter
- verifies chain integrity and attempts repair from valid local sources when possible

Once startup completes, it prints:

- the active chain directory, default chain key, and bound MCP/REST addresses
- a catalog of all exposed HTTP endpoints with one-line descriptions
- a per-chain summary with version, adapter, thought count, and per-agent counts

## Daemon Configuration

`thoughtchaind` is configured with environment variables:

- `THOUGHTCHAIN_DIR`
  Directory where ThoughtChain storage adapters store chain files.
- `THOUGHTCHAIN_DEFAULT_KEY`
  Default `chain_key` used when requests omit one. Default: `borganism-brain`
- `THOUGHTCHAIN_DEFAULT_STORAGE_ADAPTER`
  Default storage backend for newly created chains. Supported values: `binary`, `jsonl`.
  Default: `binary`
- `THOUGHTCHAIN_STORAGE_ADAPTER`
  Legacy alias for `THOUGHTCHAIN_DEFAULT_STORAGE_ADAPTER`, still accepted for compatibility.
- `THOUGHTCHAIN_BIND_HOST`
  Bind host for both HTTP servers. Default: `127.0.0.1`
- `THOUGHTCHAIN_MCP_PORT`
  MCP server port. Default: `9471`
- `THOUGHTCHAIN_REST_PORT`
  REST server port. Default: `9472`

Example:

```bash
THOUGHTCHAIN_DIR=/tmp/thoughtchain \
THOUGHTCHAIN_DEFAULT_KEY=borganism-brain \
THOUGHTCHAIN_DEFAULT_STORAGE_ADAPTER=binary \
THOUGHTCHAIN_BIND_HOST=127.0.0.1 \
THOUGHTCHAIN_MCP_PORT=9471 \
THOUGHTCHAIN_REST_PORT=9472 \
cargo run --bin thoughtchaind
```

## Server Surfaces

MCP endpoints:

- `GET /health`
- `POST /`
- `POST /tools/list`
- `POST /tools/execute`

REST endpoints:

- `GET /health`
- `GET /v1/chains`
- `POST /v1/bootstrap`
- `POST /v1/agents`
- `POST /v1/agent`
- `POST /v1/agent-registry`
- `POST /v1/agents/upsert`
- `POST /v1/agents/description`
- `POST /v1/agents/aliases`
- `POST /v1/agents/keys`
- `POST /v1/agents/keys/revoke`
- `POST /v1/agents/disable`
- `POST /v1/thoughts`
- `POST /v1/retrospectives`
- `POST /v1/search`
- `POST /v1/recent-context`
- `POST /v1/memory-markdown`
- `POST /v1/head`

## MCP Tool Catalog

The daemon currently exposes 17 MCP tools:

- `thoughtchain_bootstrap`
  Create a chain if needed and write one bootstrap checkpoint when it is empty.
- `thoughtchain_append`
  Append a durable semantic thought with optional tags, concepts, refs, and signature metadata.
- `thoughtchain_append_retrospective`
  Append a retrospective memory intended to prevent future agents from repeating a hard failure.
- `thoughtchain_search`
  Search thoughts by semantic filters, identity filters, time bounds, and scoring thresholds.
- `thoughtchain_list_chains`
  List known chains with version, storage adapter, counts, and storage location.
- `thoughtchain_list_agents`
  List the distinct agent identities participating in one chain.
- `thoughtchain_get_agent`
  Return one full agent registry record, including status, aliases, description, keys, and per-chain activity metadata.
- `thoughtchain_list_agent_registry`
  Return the full per-chain agent registry.
- `thoughtchain_upsert_agent`
  Create or update a registry record before or after an agent writes thoughts.
- `thoughtchain_set_agent_description`
  Set or clear the description stored for one registered agent.
- `thoughtchain_add_agent_alias`
  Add a historical or alternate alias to a registered agent.
- `thoughtchain_add_agent_key`
  Add or replace one public verification key on a registered agent.
- `thoughtchain_revoke_agent_key`
  Revoke one previously registered public key.
- `thoughtchain_disable_agent`
  Disable one agent by marking its registry status as revoked.
- `thoughtchain_recent_context`
  Render recent thoughts into a prompt snippet for session resumption.
- `thoughtchain_memory_markdown`
  Export a `MEMORY.md`-style Markdown view of the full chain or a filtered subset.
- `thoughtchain_head`
  Return head metadata, latest thought summary, and integrity state.

The detailed request and response shapes for the MCP surface live in
[`THOUGHTCHAIN_MCP.md`](../THOUGHTCHAIN_MCP.md). The REST equivalents live in
[`THOUGHTCHAIN_REST.md`](../THOUGHTCHAIN_REST.md).

## Using With MCP Clients

`thoughtchaind` exposes both:

- a standard streamable HTTP MCP endpoint at `POST /`
- the legacy CloudLLM-compatible MCP endpoints at `POST /tools/list` and
  `POST /tools/execute`

That means you can:

- use native MCP clients such as Codex and Claude Code against `http://127.0.0.1:9471`
- keep using direct HTTP calls or `cloudllm`'s MCP compatibility layer when needed

### Codex

Codex CLI expects a streamable HTTP MCP server when you use `--url`:

```bash
codex mcp add thoughtchain --url http://127.0.0.1:9471
```

Useful follow-up commands:

```bash
codex mcp list
codex mcp get thoughtchain
```

This connects Codex to the daemon's standard MCP root endpoint.

### Qwen Code

Qwen Code uses the same HTTP MCP transport model:

```bash
qwen mcp add --transport http thoughtchain http://127.0.0.1:9471
```

Useful follow-up commands:

```bash
qwen mcp list
```

For user-scoped configuration:

```bash
qwen mcp add --scope user --transport http thoughtchain http://127.0.0.1:9471
```

### Claude Code

Claude Code supports MCP servers through its `claude mcp` commands and
project/user MCP config. For a remote HTTP MCP server, the configuration shape
is transport-based:

```bash
claude mcp add --transport http thoughtchain http://127.0.0.1:9471
```

Useful follow-up commands:

```bash
claude mcp list
claude mcp get thoughtchain
```

Claude Code also supports JSON config files such as `.mcp.json`. A ThoughtChain
HTTP MCP config looks like this:

```json
{
  "mcpServers": {
    "thoughtchain": {
      "type": "http",
      "url": "http://127.0.0.1:9471"
    }
  }
}
```

Important:

- `/mcp` inside Claude Code is mainly for managing or authenticating MCP
  servers that are already configured
- the server itself must already be running at the configured URL

### GitHub Copilot CLI

GitHub Copilot CLI can also connect to `thoughtchaind` as a remote HTTP MCP
server.

From interactive mode:

1. Run `/mcp add`
2. Set `Server Name` to `thoughtchain`
3. Set `Server Type` to `HTTP`
4. Set `URL` to `http://127.0.0.1:9471`
5. Leave headers empty unless you add auth later
6. Save the config

You can also configure it manually in `~/.copilot/mcp-config.json`:

```json
{
  "mcpServers": {
    "thoughtchain": {
      "type": "http",
      "url": "http://127.0.0.1:9471",
      "headers": {},
      "tools": ["*"]
    }
  }
}
```

## Retrospective Memory

ThoughtChain supports a dedicated retrospective workflow for lessons learned.

- Use `thoughtchain_append` for ordinary durable facts, constraints, decisions,
  plans, and summaries.
- Use `thoughtchain_append_retrospective` after a repeated failure, a long snag,
  or a non-obvious fix when future agents should avoid repeating the same
  struggle.

The retrospective helper:

- defaults `thought_type` to `LessonLearned`
- always stores the thought with `role = Retrospective`
- still supports tags, concepts, confidence, importance, and `refs` to earlier
  thoughts such as the original mistake or correction

## Shared-Chain Multi-Agent Use

Multiple agents can write to the same `chain_key`.

Each stored thought carries a stable:

- `agent_id`

Agent profile metadata now lives in the per-chain agent registry instead of
being duplicated into every thought record. Registry records can store:

- `display_name`
- `agent_owner`
- `description`
- `aliases`
- `status`
- `public_keys`
- per-chain activity counters such as `thought_count`, `first_seen_index`, and `last_seen_index`

That allows a shared chain to represent memory from:

- multiple agents in one workflow
- multiple named roles in one orchestration system
- multiple tenants or owners writing to the same chain namespace

Queries can filter by:

- `agent_id`
- `agent_name`
- `agent_owner`

Administrative tools can also inspect and mutate the agent registry directly,
so agents can be documented, disabled, aliased, or provisioned with public keys
before they start writing thoughts.

## Related Docs

At the repository root:

- `THOUGHTCHAIN_MCP.md`
- `THOUGHTCHAIN_REST.md`
- `thoughtchain/WHITEPAPER.md`
- `thoughtchain/changelog.txt`
