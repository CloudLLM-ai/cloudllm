# thoughtchain

`thoughtchain` is a standalone Rust crate for durable agent memory.

It stores semantically typed thoughts in an append-only, hash-chained memory
log through a swappable storage adapter layer. The current default backend is
JSONL, but the chain model is no longer tied to that format. Agents can:

- persist important insights, decisions, constraints, and checkpoints
- relate new thoughts to earlier thoughts with typed graph edges
- query memory by type, role, agent, tags, concepts, text, and importance
- reconstruct context for agent resumption
- export a Markdown memory view that can back `MEMORY.md`, MCP, REST, or CLI flows

The crate is intentionally independent from `cloudllm` so it can be embedded in
other agent systems without creating circular dependencies.

## What Is In This Folder

`thoughtchain/` contains:

- the standalone `thoughtchain` library crate
- an optional `server` feature for HTTP MCP and REST servers
- the `thoughtchaind` daemon binary
- dedicated tests under `thoughtchain/tests`

## Build

From inside `thoughtchain/`:

```bash
cargo build
```

Build with server support:

```bash
cargo build --features server
```

## Test

Run the crate tests:

```bash
cargo test
```

Run tests including the server feature:

```bash
cargo test --features server
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

Include the server API docs:

```bash
cargo doc --no-deps --features server
```

## Run The Daemon

The standalone daemon binary is `thoughtchaind`.

Run it with the server feature enabled:

```bash
cargo run --features server --bin thoughtchaind
```

When it starts, it serves both:

- an MCP server
- a REST server

It prints the active chain directory, default chain key, and bound MCP/REST addresses on startup.

## Daemon Configuration

`thoughtchaind` is configured with environment variables:

- `THOUGHTCHAIN_DIR`
  Directory where the default JSONL storage adapter stores chain files.
- `THOUGHTCHAIN_DEFAULT_KEY`
  Default `chain_key` used when requests omit one.
- `THOUGHTCHAIN_BIND_HOST`
  Bind host for both HTTP servers. Default: `127.0.0.1`
- `THOUGHTCHAIN_MCP_PORT`
  MCP server port. Default: `9471`
- `THOUGHTCHAIN_REST_PORT`
  REST server port. Default: `9472`

Example:

```bash
THOUGHTCHAIN_DIR=/tmp/thoughtchain \
THOUGHTCHAIN_DEFAULT_KEY=shared-project \
THOUGHTCHAIN_BIND_HOST=127.0.0.1 \
THOUGHTCHAIN_MCP_PORT=9471 \
THOUGHTCHAIN_REST_PORT=9472 \
cargo run --features server --bin thoughtchaind
```

## Server Surfaces

MCP endpoints:

- `GET /health`
- `POST /tools/list`
- `POST /tools/execute`

REST endpoints:

- `GET /health`
- `POST /v1/bootstrap`
- `POST /v1/thoughts`
- `POST /v1/search`
- `POST /v1/recent-context`
- `POST /v1/memory-markdown`
- `POST /v1/head`

## Shared-Chain Multi-Agent Use

Multiple agents can write to the same `chain_key`.

Each stored thought carries:

- `agent_id`
- `agent_name`
- optional `agent_owner`

That allows a shared chain to represent memory from:

- multiple agents in one workflow
- multiple named roles in one orchestration system
- multiple tenants or owners writing to the same chain namespace

Queries can filter by:

- `agent_id`
- `agent_name`
- `agent_owner`

## Related Docs

At the repository root:

- `THOUGHTCHAIN_MCP.md`
- `THOUGHTCHAIN_REST.md`
- `thoughtchain/changelog.txt`
