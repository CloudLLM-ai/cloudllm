# mentisdb Standards

`mentisdb` follows the same engineering bar as `cloudllm`.

## Scope

- `mentisdb` is a standalone crate.
- `cloudllm` may depend on `mentisdb`.
- `mentisdb` must not depend on `cloudllm`.

## Standards

- Public API changes require rustdoc on exported types and functions.
- New behavior requires tests.
- Serialization and persistence changes must preserve explicit integrity checks.
- Retrieval and export features should be generic enough to support direct crate use, MCP services, REST services, and CLI tools.
- Crate metadata must remain suitable for crates.io publishing.

## Design Bias

- Prefer semantic memory primitives over agent-framework-specific abstractions.
- Keep storage durable and append-only.
- Keep query and export APIs first-class, not bolted on as prompt helpers.
