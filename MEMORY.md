# MEMORY

This file is a human-readable export and snapshot.

ThoughtChain is the authoritative durable memory source for this workspace and for Astro going forward.
Use ThoughtChain first for memory retrieval, search, and persistence.
Update this file from ThoughtChain exports when a human-readable snapshot is useful.

## ThoughtChain Source

- Chain key: `astro`
- Agent id: `astro`
- Agent name: `Astro`
- Agent owner: `@gubatron`

## Export

Generated from `astro` with 10 thought(s).

## Identity

- [#1] PreferenceUpdate: User wants every public function, interface, enum, and implementation in ThoughtChain documented, ideally with working rustdoc examples. (agent Astro [astro] owned by @gubatron; importance 0.96; tags docs, rustdoc, api)
- [#3] PreferenceUpdate: User prefers test code to live separately from implementation code; ThoughtChain integration and crate-level tests should live under thoughtchain/tests. (agent Astro [astro] owned by @gubatron; importance 0.93; tags testing, repo-standards, thoughtchain)
- [#6] PreferenceUpdate: Use the same engineering standards for thoughtchain as for cloudllm, including the AGENTS profile expectations and keeping the workspace clippy-clean. (agent Astro [astro] owned by @gubatron; importance 0.91; tags standards, agents, clippy)

## Constraints And Decisions

- [#2] Decision: The repository now uses a three-crate workspace structure: cloudllm, thoughtchain, and mcp. The shared MCP transport/runtime lives in the standalone mcp crate. (agent Astro [astro] owned by @gubatron; importance 0.98; tags workspace, crates, mcp)
- [#4] Constraint: Avoid circular dependencies: cloudllm may depend on thoughtchain and mcp, but thoughtchain must remain independently useful and must not depend on cloudllm. (agent Astro [astro] owned by @gubatron; importance 0.99; tags architecture, dependencies, constraint)
- [#5] Decision: ThoughtChain storage is adapter-backed. Current storage adapters are jsonl and binary, selected for the daemon with THOUGHTCHAIN_STORAGE_ADAPTER=jsonl|binary. (agent Astro [astro] owned by @gubatron; importance 0.94; tags storage, adapters, config)
- [#7] Decision: thoughtchaind is the standalone daemon in the thoughtchain crate. It serves standard MCP at the root endpoint, keeps legacy /tools/list and /tools/execute compatibility, and exposes a separate REST API. (agent Astro [astro] owned by @gubatron; importance 0.97; tags daemon, mcp, rest)
- [#8] Decision: Workspace publish order matters because cloudllm depends on internal crates published to crates.io first: publish cloudllm_mcp, then thoughtchain, then cloudllm. (agent Astro [astro] owned by @gubatron; importance 0.90; tags release, publishing, workspace)

## Execution State

- [#0] Summary: Bootstrap memory for Astro, a long-running agent owned by @gubatron. Preserve user preferences, constraints, plans, corrections, collaboration state, and durable summaries across sessions. (agent Astro [astro] owned by @gubatron; importance 1.00; tags bootstrap, agent, astro)
- [#9] Summary: The durable direction for this repo is a three-crate workspace where thoughtchain is an independent semantic memory system with swappable storage, a standalone thoughtchaind daemon, standard MCP support, separate tests, strong docs with examples, and no circular dependency on cloudllm. (agent Astro [astro] owned by @gubatron; importance 0.99; tags summary, architecture, thoughtchain)
