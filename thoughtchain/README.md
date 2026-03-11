# thoughtchain

`thoughtchain` is a standalone Rust crate for durable agent memory.

It stores semantically typed thoughts in an append-only, hash-chained `.jsonl`
log so agents can:

- persist important insights, decisions, constraints, and checkpoints
- relate new thoughts to earlier thoughts with typed graph edges
- query memory by type, role, agent, tags, concepts, text, and importance
- reconstruct context for agent resumption
- export a Markdown memory view that can back `MEMORY.md`, MCP, REST, or CLI flows

The crate is intentionally independent from `cloudllm` so it can be embedded in
other agent systems without creating circular dependencies.
