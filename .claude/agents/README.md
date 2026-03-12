# Project Agents

This directory contains project-local agent briefs for Claude-compatible agent
setups.

The source of truth for durable memory is MentisDB, not these files.
Use these files as compact operating briefs and use MentisDB for:

- retrieval of prior decisions, corrections, and constraints
- shared memory across agents
- long-term continuity across sessions

## Shared Memory

All project agents should write durable memory to:

- `chain_key`: `borganism-brain`

Each agent should keep its own identity fields:

- `agent_id`
- `agent_name`
- optional `agent_owner`

## Current Durable Repo Facts

- This repository is a workspace with three crates:
  - `cloudllm`
  - `mentisdb`
  - `mcp` as the published package `cloudllm_mcp`
- Dependency direction must stay one-way:
  - `cloudllm` may depend on `mentisdb` and `mcp`
  - `mentisdb` may depend on `mcp`
  - `mentisdb` must not depend on `cloudllm`
- `mentisdbd` is the standalone daemon in `mentisdb`
- `mentisdbd` exposes:
  - standard MCP at `POST /`
  - legacy compatibility MCP at `POST /tools/list` and `POST /tools/execute`
  - REST under `/v1/...`
- MentisDB storage is adapter-backed:
  - `jsonl`
  - `binary`
- The daemon default shared chain key is `borganism-brain`
- Publish order is:
  - `cloudllm_mcp`
  - `mentisdb`
  - `cloudllm`

## Engineering Standards

- Keep tests separate from implementation code where practical.
- Public APIs require documentation, ideally with working rustdoc examples.
- Keep the workspace clippy-clean.
- Update docs when architecture or transport behavior changes.
