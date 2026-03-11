# ThoughtChain REST

`thoughtchain` can also be exposed as a plain REST service for agents, services, CLIs, and orchestration systems that do not want to speak MCP.

This document describes the current REST interface implemented in `thoughtchain/src/server.rs`.

## Purpose

The REST API gives a caller durable, append-only semantic memory with:

- semantic `thought_type`
- operational `role`
- timestamps
- importance and confidence scoring
- tags and concepts
- integrity verification through a hash chain
- resumable prompt rendering
- `MEMORY.md`-style export

The basic usage pattern is:

1. bootstrap a chain once
2. append durable thoughts as work progresses
3. search the chain before making decisions
4. render recent context when resuming work
5. export Markdown when a human-readable memory view is needed

## Running The Server

The standalone daemon is `thoughtchaind`.

Example:

```bash
cargo run -p thoughtchain --features server --bin thoughtchaind
```

Environment variables:

- `THOUGHTCHAIN_DIR`
  Directory where the default JSONL storage adapter stores chain files.
- `THOUGHTCHAIN_DEFAULT_KEY`
  Default `chain_key` used when a request omits one.
- `THOUGHTCHAIN_BIND_HOST`
  Bind host for both HTTP servers. Default: `127.0.0.1`
- `THOUGHTCHAIN_MCP_PORT`
  MCP server port. Default: `9471`
- `THOUGHTCHAIN_REST_PORT`
  REST server port. Default: `9472`

By default, the REST base URL is:

```text
http://127.0.0.1:9472
```

## Chain Model

Every memory belongs to a `chain_key`.

- If `chain_key` is omitted, the server uses its configured default chain.
- Each chain is stored through a pluggable storage adapter.
- The current daemon uses the JSONL storage adapter by default.
- The server verifies chain integrity when it opens a chain.

For a remote client, `chain_key` is the durable identity of the memory stream.

Examples:

- one chain per long-running agent
- one chain per user
- one chain per project
- one chain per workflow or orchestration pipeline

## Endpoints

### `GET /health`

Simple service health check.

Response:

```json
{
  "status": "ok",
  "service": "thoughtchain"
}
```

### `POST /v1/bootstrap`

Creates the chain if needed and writes a bootstrap thought only when the chain is empty.

Request body:

- `chain_key: string` optional
- `agent_id: string` optional
- `agent_name: string` optional
- `agent_owner: string` optional
- `content: string` required
- `importance: number` optional
- `tags: string[]` optional
- `concepts: string[]` optional

Behavior:

- if the chain is empty, one thought is appended
- that thought is stored as:
  - `thought_type = Summary`
  - `role = Checkpoint`
- if `agent_id` is omitted, bootstrap uses a system producer identity
- if the chain already contains thoughts, nothing is overwritten

Response body:

- `bootstrapped: boolean`
- `thought_count: integer`
- `head_hash: string | null`

Example:

```bash
curl -s http://127.0.0.1:9472/v1/bootstrap \
  -H 'content-type: application/json' \
  -d '{
    "chain_key": "persistent-chat-agent",
    "agent_id": "bootstrap",
    "agent_name": "Bootstrap",
    "agent_owner": "cloudllm",
    "content": "Bootstrap memory for a long-running coding agent. Preserve user preferences, constraints, plans, corrections, and summaries across sessions.",
    "importance": 1.0,
    "tags": ["bootstrap", "system"],
    "concepts": ["persistence", "semantic-memory"]
  }'
```

Example response:

```json
{
  "bootstrapped": true,
  "thought_count": 1,
  "head_hash": "7e1c..."
}
```

### `POST /v1/thoughts`

Appends a durable thought.

Request body:

- `chain_key: string` optional
- `agent_id: string` optional
- `agent_name: string` optional
- `agent_owner: string` optional
- `thought_type: string` required
- `content: string` required
- `role: string` optional
- `importance: number` optional
- `confidence: number` optional
- `tags: string[]` optional
- `concepts: string[]` optional
- `refs: integer[]` optional

Notes:

- `importance` is clamped to `0.0..=1.0`
- `confidence` is clamped to `0.0..=1.0`
- if `role` is omitted, the server defaults it to `Memory`
- if `agent_id` is omitted, it defaults to the current `chain_key`
- if `agent_name` is omitted, it defaults to `agent_id`
- `refs` points to prior thought indices in the same chain

Response body:

- `thought: object`
- `head_hash: string | null`

The returned `thought` is the full stored thought object, including fields such as:

- `index`
- `id`
- `agent_id`
- `agent_name`
- `agent_owner`
- `timestamp`
- `thought_type`
- `role`
- `content`
- `confidence`
- `importance`
- `tags`
- `concepts`
- `refs`
- `relations`
- `hash`
- `prev_hash`

Supported `thought_type` values:

- `PreferenceUpdate`
- `UserTrait`
- `RelationshipUpdate`
- `Finding`
- `Insight`
- `FactLearned`
- `PatternDetected`
- `Hypothesis`
- `Mistake`
- `Correction`
- `AssumptionInvalidated`
- `Constraint`
- `Plan`
- `Subgoal`
- `Decision`
- `StrategyShift`
- `Wonder`
- `Question`
- `Idea`
- `Experiment`
- `ActionTaken`
- `TaskComplete`
- `Checkpoint`
- `StateSnapshot`
- `Handoff`
- `Summary`
- `Surprise`

Supported `role` values:

- `Memory`
- `WorkingMemory`
- `Summary`
- `Compression`
- `Checkpoint`
- `Handoff`
- `Audit`

Example:

```bash
curl -s http://127.0.0.1:9472/v1/thoughts \
  -H 'content-type: application/json' \
  -d '{
    "chain_key": "persistent-chat-agent",
    "agent_id": "agent-42",
    "agent_name": "Planner",
    "agent_owner": "ops-team",
    "thought_type": "Constraint",
    "role": "Memory",
    "importance": 0.95,
    "confidence": 0.98,
    "tags": ["security", "ops"],
    "concepts": ["no-external-api", "offline-mode"],
    "content": "This deployment path must work without external APIs."
  }'
```

When should a thought use `refs`?

- when a `Correction` fixes an earlier `Mistake`
- when a `Lesson`-like `Insight` was learned from a prior `Mistake`
- when a `Summary` compresses earlier thoughts
- when a `Decision` was made because of a prior `Constraint` or `Finding`
- when a `Handoff` continues work from a prior `Checkpoint`

Concrete example:

1. append a `Mistake` at index `12`
2. later append a `Correction` with `refs: [12]`
3. later append an `Insight` with `refs: [12]` or `refs: [12, 13]`

That gives future agents a direct path from failure to correction to durable lesson.

### `POST /v1/search`

Queries the chain for relevant memories.

Request body:

- `chain_key: string` optional
- `text: string` optional
- `thought_types: string[]` optional
- `roles: string[]` optional
- `tags_any: string[]` optional
- `concepts_any: string[]` optional
- `agent_ids: string[]` optional
- `agent_names: string[]` optional
- `agent_owners: string[]` optional
- `min_importance: number` optional
- `min_confidence: number` optional
- `since: string` optional, RFC 3339 timestamp
- `until: string` optional, RFC 3339 timestamp
- `limit: integer` optional

Response body:

- `thoughts: object[]`

Typical use:

- retrieve preferences before responding to a user
- retrieve constraints before planning
- retrieve prior mistakes before retrying similar work
- search for all thoughts related to `rust`, `memory`, `rate limiting`, or `embeddings`
- filter to thoughts produced by a specific agent, agent name, or owner/tenant
- filter to a time window for one session or incident

Example:

```bash
curl -s http://127.0.0.1:9472/v1/search \
  -H 'content-type: application/json' \
  -d '{
    "chain_key": "persistent-chat-agent",
    "text": "rate limit",
    "agent_names": ["Planner"],
    "thought_types": ["Insight", "Mistake", "Correction"],
    "min_importance": 0.7,
    "limit": 8
  }'
```

Example response:

```json
{
  "thoughts": [
    {
      "index": 8,
      "thought_type": "Mistake",
      "content": "Incorrectly blamed database latency; the real issue was API rate limiting."
    },
    {
      "index": 9,
      "thought_type": "Correction",
      "content": "Shift debugging focus to upstream API throttling."
    }
  ]
}
```

### `POST /v1/recent-context`

Renders recent thoughts as a prompt snippet suitable for resuming work.

Request body:

- `chain_key: string` optional
- `last_n: integer` optional, default `12`

Response body:

- `prompt: string`

Typical use:

- beginning of a new session
- preloading a worker before it continues a task
- resuming after a model or process restart

Example:

```bash
curl -s http://127.0.0.1:9472/v1/recent-context \
  -H 'content-type: application/json' \
  -d '{
    "chain_key": "persistent-chat-agent",
    "last_n": 10
  }'
```

### `POST /v1/memory-markdown`

Exports the full chain, or a filtered subset, as `MEMORY.md`-style Markdown.

Request body:

- `chain_key: string` optional
- `text: string` optional
- `thought_types: string[]` optional
- `roles: string[]` optional
- `tags_any: string[]` optional
- `concepts_any: string[]` optional
- `agent_ids: string[]` optional
- `agent_names: string[]` optional
- `agent_owners: string[]` optional
- `min_importance: number` optional
- `min_confidence: number` optional
- `since: string` optional, RFC 3339 timestamp
- `until: string` optional, RFC 3339 timestamp
- `limit: integer` optional

Response body:

- `markdown: string`

Typical use:

- render a human-readable state snapshot
- export memory into a repo artifact or operator dashboard
- generate a concise memory handoff for a future agent session

Example:

```bash
curl -s http://127.0.0.1:9472/v1/memory-markdown \
  -H 'content-type: application/json' \
  -d '{
    "chain_key": "persistent-chat-agent",
    "thought_types": ["PreferenceUpdate", "Constraint", "Decision", "Summary"],
    "min_importance": 0.75
  }'
```

### `POST /v1/head`

Returns chain head metadata.

Request body:

- `chain_key: string` optional

Response body:

- `chain_key: string`
- `thought_count: integer`
- `head_hash: string | null`
- `latest_thought: object | null`
- `integrity_ok: boolean`
- `storage_location: string`

Typical use:

- check whether a chain exists
- inspect the current head without doing a broader search
- verify integrity before starting a new agent session

Example:

```bash
curl -s http://127.0.0.1:9472/v1/head \
  -H 'content-type: application/json' \
  -d '{
    "chain_key": "persistent-chat-agent"
  }'
```

## Error Format

Successful REST calls return normal JSON responses with HTTP `200`.

Validation or execution failures return HTTP `400` with:

```json
{
  "error": "human-readable message"
}
```

Examples:

- unknown `thought_type`
- unknown `role`
- malformed JSON
- invalid timestamp format
- integrity or storage errors when opening a chain

## Recommended Sequence

For a long-running agent:

1. `POST /v1/bootstrap`
   Write the initial purpose of the chain if it is empty.
2. `POST /v1/head`
   Inspect whether there is prior memory.
3. `POST /v1/recent-context`
   Load a compact resume prompt into the next model session.
4. `POST /v1/search`
   Pull relevant preferences, constraints, plans, mistakes, or summaries before acting.
5. `POST /v1/thoughts`
   Append durable thoughts during meaningful checkpoints.
6. `POST /v1/memory-markdown`
   Export a human-readable summary when needed.

For a multi-agent pipeline:

1. one agent appends `Findings`, `Mistakes`, `Corrections`, and `Decisions`
2. a later agent searches by concept, type, and time range
3. the later agent appends `Insights`, `StrategyShift`, or `Handoff`
4. operators export a Markdown memory view for review

## What The REST API Does Not Yet Expose

The core `thoughtchain` crate supports richer internal structures than the REST append endpoint currently exposes.

Today, the REST append API does not accept:

- `session_id`
- typed `relations`

The stored `Thought` objects can still contain those fields, but remote callers currently append through the simpler `refs`-based interface.

That means:

- `refs` should be used today for causal or corrective links
- typed relation submission can be added later without changing the basic chain model

## Relationship Between REST And MCP

The REST and MCP services expose the same core ThoughtChain operations.

- REST is better for services, scripts, dashboards, and generic HTTP clients.
- MCP is better when an agent framework wants memory to appear as callable tools.

The corresponding MCP contract is documented in `THOUGHTCHAIN_MCP.md`.
