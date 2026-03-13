# MentisDB MCP

`MentisDB` can be exposed as an MCP server so a remote agent can treat durable memory as a tool, not as a writable `MEMORY.md` file.

At the moment, the MentisDB MCP server exposes 26 tools:

- `mentisdb_bootstrap`
- `mentisdb_append`
- `mentisdb_append_retrospective`
- `mentisdb_search`
- `mentisdb_list_chains`
- `mentisdb_list_agents`
- `mentisdb_get_agent`
- `mentisdb_list_agent_registry`
- `mentisdb_upsert_agent`
- `mentisdb_set_agent_description`
- `mentisdb_add_agent_alias`
- `mentisdb_add_agent_key`
- `mentisdb_revoke_agent_key`
- `mentisdb_disable_agent`
- `mentisdb_recent_context`
- `mentisdb_memory_markdown`
- `mentisdb_skill_md`
- `mentisdb_list_skills`
- `mentisdb_skill_manifest`
- `mentisdb_upload_skill`
- `mentisdb_search_skill`
- `mentisdb_read_skill`
- `mentisdb_skill_versions`
- `mentisdb_deprecate_skill`
- `mentisdb_revoke_skill`
- `mentisdb_head`

This document describes the current remote interface implemented in `mentisdb/src/server.rs`.

## Purpose

The MCP server gives an agent a durable, append-only memory log with:

- semantic `thought_type`
- operational `role`
- timestamps
- confidence and importance scoring
- tags and concepts
- hash-chain integrity checks
- resumable recent-context rendering
- export to `MEMORY.md`-style Markdown

The main idea is:

1. the agent stores durable thoughts as work progresses
2. later agents search those thoughts semantically
3. a new session reconstructs context from the chain instead of depending on a mutable text file

## Chain Model

Every memory belongs to a `chain_key`.

- If `chain_key` is omitted, the server uses its configured default chain.
- A chain is stored through a pluggable storage adapter.
- The current daemon uses the binary storage adapter by default.
- `mentisdbd` migrates legacy schema-version `0` chains to the current schema on startup before serving traffic.
- The server verifies integrity when opening the chain.

For a remote agent, `chain_key` is the durable identity of the memory stream.

Examples:

- one chain per long-running agent
- one chain per user
- one chain per project
- one chain per orchestration workflow

## Skill Registry Model

MentisDB also keeps a versioned skill registry in the daemon storage root.

- Skills are stored once per daemon, not inside individual thought chains.
- `mentisdb_upload_skill` requires `agent_id` to already exist in the agent registry for the referenced `chain_key`.
- Other skill tools also accept `chain_key`; today that value is used for audit and logging context while the registry itself remains daemon-global.
- Skill content should be treated as untrusted input until provenance and requested capabilities are validated.

## Available Tools

### `mentisdb_bootstrap`

Creates the chain if needed and writes a bootstrap memory only when the chain is empty.

Parameters:

- `chain_key: string` optional
- `agent_id: string` optional
- `agent_name: string` optional
- `agent_owner: string` optional
- `content: string` required
- `importance: number` optional, clamped to `0.0..=1.0`
- `tags: string[]` optional
- `concepts: string[]` optional
- `storage_adapter: string` optional, one of `binary` or `jsonl`

Behavior:

- if the chain is empty, the server writes one bootstrap thought
- that thought is stored as:
  - `thought_type = Summary`
  - `role = Checkpoint`
- if `agent_id` is omitted, bootstrap defaults to a system producer identity
- if `storage_adapter` is omitted, bootstrap uses the daemon default
- if the chain already has data, nothing is overwritten

Response fields:

- `bootstrapped`
- `thought_count`
- `head_hash`

Typical use:

- first run of a persistent agent
- first run of a project memory
- creating a stable “what this memory is for” anchor

Example:

```json
{
  "tool": "mentisdb_bootstrap",
  "arguments": {
    "chain_key": "borganism-brain",
    "agent_id": "bootstrap",
    "agent_name": "Bootstrap",
    "agent_owner": "cloudllm",
    "content": "Bootstrap memory for a long-running coding agent. Preserve user preferences, constraints, plans, corrections, and summaries across sessions.",
    "importance": 1.0,
    "tags": ["bootstrap", "system"],
    "concepts": ["persistence", "semantic-memory"]
  }
}
```

### `mentisdb_append`

Appends a durable thought.

Parameters:

- `chain_key: string` optional
- `agent_id: string` optional
- `agent_name: string` optional
- `agent_owner: string` optional
- `thought_type: string` required
- `content: string` required
- `role: string` optional
- `importance: number` optional, clamped to `0.0..=1.0`
- `confidence: number` optional, clamped to `0.0..=1.0`
- `tags: string[]` optional
- `concepts: string[]` optional
- `refs: integer[]` optional
- `signing_key_id: string` optional
- `thought_signature: number[]` optional

Response fields:

- `thought`
- `head_hash`

The returned `thought` includes useful fields for later reference:

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
- `hash`
- `prev_hash`
- `signing_key_id`
- `thought_signature`

#### Supported `thought_type` values

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
- `LessonLearned`

#### Supported `role` values

- `Memory`
- `WorkingMemory`
- `Summary`
- `Compression`
- `Checkpoint`
- `Handoff`
- `Audit`
- `Retrospective`

Example:

```json
{
  "tool": "mentisdb_append",
  "arguments": {
    "chain_key": "borganism-brain",
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
  }
}
```

### `mentisdb_append_retrospective`

Appends a guided retrospective memory after a hard failure, repeated snag, or
non-obvious fix.

This is the tool agents should prefer when they want to store:

- a lesson learned from a tough debugging session
- a durable rule that prevents future rework
- a correction distilled after several failed attempts

Use `mentisdb_append` for ordinary durable facts and decisions.
Use `mentisdb_append_retrospective` when the memory exists specifically to
help future agents avoid repeating the same struggle.

Parameters:

- `chain_key: string` optional
- `agent_id: string` optional
- `agent_name: string` optional
- `agent_owner: string` optional
- `thought_type: string` optional
- `content: string` required
- `importance: number` optional, clamped to `0.0..=1.0`
- `confidence: number` optional, clamped to `0.0..=1.0`
- `tags: string[]` optional
- `concepts: string[]` optional
- `refs: integer[]` optional
- `signing_key_id: string` optional
- `thought_signature: number[]` optional

Behavior:

- defaults `thought_type` to `LessonLearned`
- always records the thought with `role = Retrospective`
- is ideal for linking back to the triggering mistake or correction through
  `refs`

Example:

```json
{
  "tool": "mentisdb_append_retrospective",
  "arguments": {
    "chain_key": "borganism-brain",
    "agent_id": "astro",
    "agent_name": "Astro",
    "content": "If a model returns multiple tool calls in one assistant turn, every tool_call_id must receive a tool response before the next model request.",
    "importance": 0.9,
    "tags": ["retrospective", "tools", "openai"],
    "concepts": ["multi-tool call handling"]
  }
}
```

### `mentisdb_search`

Queries the chain for relevant memories.

Parameters:

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

Response fields:

- `thoughts`

Typical use:

- search for prior user preferences
- retrieve constraints before planning
- find old mistakes before attempting the same task again
- retrieve memories written by a specific agent, agent name, or owner/tenant
- search for thoughts about a concept such as `rust`, `memory`, `rate limiting`, or `embeddings`

Example:

```json
{
  "tool": "mentisdb_search",
  "arguments": {
    "chain_key": "borganism-brain",
    "text": "rate limit",
    "agent_names": ["Planner"],
    "thought_types": ["Insight", "Mistake", "Correction"],
    "min_importance": 0.7,
    "limit": 8
  }
}
```

### `mentisdb_list_chains`

Lists the durable chain keys currently available in MentisDB storage.

Parameters:

- none

Response fields:

- `default_chain_key`
- `chain_keys`
- `chains`

Each returned `chain` contains:

- `chain_key`
- `version`
- `storage_adapter`
- `thought_count`
- `agent_count`
- `storage_location`

Typical use:

- discover available long-running memories on a daemon
- inspect whether a shared brain already exists before bootstrapping another
  chain

Example:

```json
{
  "tool": "mentisdb_list_chains",
  "arguments": {}
}
```

### `mentisdb_list_agents`

Lists the distinct agent identities that have written to a specific chain.

Parameters:

- `chain_key: string` optional

Response fields:

- `chain_key`
- `agents`

Each returned `agent` contains:

- `agent_id`
- `agent_name`
- `agent_owner`

Typical use:

- discover which agents participate in a shared brain
- choose `agent_names` or `agent_ids` filters before calling
  `mentisdb_search`

Example:

```json
{
  "tool": "mentisdb_list_agents",
  "arguments": {
    "chain_key": "borganism-brain"
  }
}
```

### `mentisdb_get_agent`

Returns one full agent registry record for a chain.

Parameters:

- `chain_key: string` optional
- `agent_id: string` required

Response fields:

- `chain_key`
- `agent`

The returned `agent` includes:

- `agent_id`
- `display_name`
- `agent_owner`
- `description`
- `aliases`
- `status`
- `public_keys`
- `thought_count`
- `first_seen_index`
- `last_seen_index`
- `first_seen_at`
- `last_seen_at`

Typical use:

- inspect one agent before filtering searches
- verify an alias, status, or key record
- display agent details in a UI such as a future ThoughtExplorer

### `mentisdb_list_agent_registry`

Returns the full per-chain agent registry.

Parameters:

- `chain_key: string` optional

Response fields:

- `chain_key`
- `agents`

Typical use:

- build an agent table for a chain browser
- inspect descriptions, aliases, keys, and activity counts in one call
- reconcile identity drift such as historical aliases or display-name changes

### `mentisdb_upsert_agent`

Creates or updates one agent registry record.

Parameters:

- `chain_key: string` optional
- `agent_id: string` required
- `display_name: string` optional
- `agent_owner: string` optional
- `description: string` optional
- `status: string` optional, one of `active` or `revoked`

Response fields:

- `chain_key`
- `agent`

Typical use:

- pre-register an agent before it appends thoughts
- add human-readable descriptions to a shared chain
- normalize display names and ownership labels

### `mentisdb_set_agent_description`

Sets or clears the free-form description for one registered agent.

Parameters:

- `chain_key: string` optional
- `agent_id: string` required
- `description: string` optional

Response fields:

- `chain_key`
- `agent`

Typical use:

- annotate agents with responsibilities
- clear stale descriptions without rewriting thought history

### `mentisdb_add_agent_alias`

Adds one alias to a registered agent.

Parameters:

- `chain_key: string` optional
- `agent_id: string` required
- `alias: string` required

Response fields:

- `chain_key`
- `agent`

Typical use:

- preserve historical names after a rename
- collapse duplicate identities during cleanup

### `mentisdb_add_agent_key`

Adds or replaces one public verification key on a registered agent.

Parameters:

- `chain_key: string` optional
- `agent_id: string` required
- `key_id: string` required
- `algorithm: string` required, currently `ed25519`
- `public_key_bytes: integer[]` required

Response fields:

- `chain_key`
- `agent`

Typical use:

- prepare for signed-thought verification workflows
- rotate public keys while keeping durable agent identity stable

### `mentisdb_revoke_agent_key`

Revokes one previously registered public key.

Parameters:

- `chain_key: string` optional
- `agent_id: string` required
- `key_id: string` required

Response fields:

- `chain_key`
- `agent`

Typical use:

- retire a compromised or superseded key
- preserve key history without deleting the registry record

### `mentisdb_disable_agent`

Marks one agent as revoked in the registry.

Parameters:

- `chain_key: string` optional
- `agent_id: string` required

Response fields:

- `chain_key`
- `agent`

Typical use:

- disable deprecated agents
- keep historical thoughts visible while preventing silent reuse in tooling

### `mentisdb_recent_context`

Renders the latest thoughts as a prompt snippet suitable for resuming work.

Parameters:

- `chain_key: string` optional
- `last_n: integer` optional, default `12`

Response fields:

- `prompt`

Typical use:

- beginning of a new session
- preloading a remote worker before it continues a task
- quick catch-up without full memory export

### `mentisdb_memory_markdown`

Exports the chain, or a filtered subset of it, as `MEMORY.md`-style Markdown.

Parameters:

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

Response fields:

- `markdown`

Typical use:

- give an agent a compact memory document
- inspect memory manually
- export a human-readable project memory

### `mentisdb_skill_md`

Returns the official embedded `MENTISDB_SKILL.md` Markdown file.

Parameters:

- none

Response fields:

- `markdown`

Typical use:

- bootstrap a client with the built-in MentisDB usage skill
- inspect the canonical skill text without reading local files

### `mentisdb_list_skills`

Lists uploaded skill summaries from the versioned skill registry.

Parameters:

- `chain_key: string` optional

Response fields:

- `skills`

Each returned `skill` includes:

- `skill_id`
- `name`
- `description`
- `status`
- `status_reason`
- `schema_version`
- `tags`
- `triggers`
- `warnings`
- `latest_version_id`
- `version_count`
- `created_at`
- `updated_at`
- `latest_uploaded_at`
- `latest_uploaded_by_agent_id`
- `latest_uploaded_by_agent_name`
- `latest_uploaded_by_agent_owner`
- `latest_source_format`

### `mentisdb_skill_manifest`

Returns the versioned skill-registry manifest describing supported formats and searchable fields.

Parameters:

- none

Response fields:

- `manifest`

### `mentisdb_upload_skill`

Uploads a new immutable skill version from Markdown or JSON.

Parameters:

- `chain_key: string` optional
- `skill_id: string` optional
- `agent_id: string` required
- `format: string` optional, one of `markdown`, `md`, or `json`
- `content: string` required

Response fields:

- `skill`

Typical use:

- publish a reviewed skill for reuse by other agents
- upload a new version without rewriting earlier audit history

### `mentisdb_search_skill`

Searches the skill registry by indexed metadata and time window.

Parameters:

- `chain_key: string` optional
- `text: string` optional
- `skill_ids: string[]` optional
- `names: string[]` optional
- `tags_any: string[]` optional
- `triggers_any: string[]` optional
- `uploaded_by_agent_ids: string[]` optional
- `uploaded_by_agent_names: string[]` optional
- `uploaded_by_agent_owners: string[]` optional
- `statuses: string[]` optional, any of `active`, `deprecated`, `revoked`
- `formats: string[]` optional, any of `markdown` or `json`
- `schema_versions: integer[]` optional
- `since: string` optional, RFC 3339 timestamp
- `until: string` optional, RFC 3339 timestamp
- `limit: integer` optional

Response fields:

- `skills`

### `mentisdb_read_skill`

Reads one stored skill in Markdown or JSON and returns explicit safety warnings.

Parameters:

- `chain_key: string` optional
- `skill_id: string` required
- `version_id: string` optional
- `format: string` optional, one of `markdown`, `md`, or `json`

Response fields:

- `skill_id`
- `version_id`
- `format`
- `source_format`
- `schema_version`
- `content`
- `status`
- `safety_warnings`

### `mentisdb_skill_versions`

Lists immutable uploaded versions for one stored skill.

Parameters:

- `chain_key: string` optional
- `skill_id: string` required

Response fields:

- `skill_id`
- `versions`

Each returned `version` includes:

- `version_id`
- `uploaded_at`
- `uploaded_by_agent_id`
- `uploaded_by_agent_name`
- `uploaded_by_agent_owner`
- `source_format`
- `content_hash`
- `schema_version`

### `mentisdb_deprecate_skill`

Marks one stored skill as deprecated while preserving prior versions.

Parameters:

- `chain_key: string` optional
- `skill_id: string` required
- `reason: string` optional

Response fields:

- `skill`

### `mentisdb_revoke_skill`

Marks one stored skill as revoked while preserving prior versions.

Parameters:

- `chain_key: string` optional
- `skill_id: string` required
- `reason: string` optional

Response fields:

- `skill`

### `mentisdb_head`

Returns chain metadata.

Parameters:

- `chain_key: string` optional

Response fields:

- `chain_key`
- `thought_count`
- `head_hash`
- `latest_thought`
- `integrity_ok`
- `storage_location`

Typical use:

- health checks
- “did memory append succeed?”
- quick introspection of the newest memory

## Recommended Agent Workflow

For a remote agent, the normal flow should look like this:

1. If you are connecting to a shared daemon, call `mentisdb_list_chains` first.
2. Bootstrap the chain once if it does not already exist.
3. For shared chains, call `mentisdb_list_agents` or `mentisdb_list_agent_registry` to discover which agents are already writing there.
4. If you need better metadata, call `mentisdb_upsert_agent`, `mentisdb_set_agent_description`, or `mentisdb_add_agent_alias` before active use.
5. At the start of a session, call `mentisdb_recent_context` or `mentisdb_memory_markdown`.
6. Before important work, call `mentisdb_search` for relevant prior constraints, plans, mistakes, and insights.
7. When reusable operating knowledge belongs in a sharable skill, call `mentisdb_search_skill` or `mentisdb_read_skill` before reinventing it.
8. During work, append durable thoughts whenever the agent learns something worth keeping.
9. After a hard failure or a long debugging snag, prefer `mentisdb_append_retrospective`.
10. Publish reviewed reusable instructions through `mentisdb_upload_skill` instead of hiding them in ad hoc notes.
11. At the end of a session, append a `Summary`, `Checkpoint`, or `Handoff`.

## Example Sequence

This sequence shows a realistic remote-agent interaction.

### 1. First run

Bootstrap the chain:

```json
{
  "tool": "mentisdb_bootstrap",
  "arguments": {
    "chain_key": "project-alpha",
    "content": "Memory for Project Alpha. Preserve architecture decisions, user preferences, constraints, mistakes, and deployment lessons.",
    "importance": 1.0,
    "tags": ["bootstrap"],
    "concepts": ["project-alpha", "semantic-memory"]
  }
}
```

### 2. On session start

Load recent context:

```json
{
  "tool": "mentisdb_recent_context",
  "arguments": {
    "chain_key": "project-alpha",
    "last_n": 12
  }
}
```

### 3. Before acting

Search for relevant memories:

```json
{
  "tool": "mentisdb_search",
  "arguments": {
    "chain_key": "project-alpha",
    "thought_types": ["Constraint", "Decision", "Mistake", "Correction"],
    "text": "deployment",
    "limit": 10
  }
}
```

### 4. During work

Store a new plan:

```json
{
  "tool": "mentisdb_append",
  "arguments": {
    "chain_key": "project-alpha",
    "thought_type": "Plan",
    "role": "Memory",
    "importance": 0.82,
    "tags": ["deployment", "rollout"],
    "concepts": ["staged-rollout"],
    "content": "Use a staged deployment with a canary instance before global rollout."
  }
}
```

Store a mistake:

```json
{
  "tool": "mentisdb_append",
  "arguments": {
    "chain_key": "project-alpha",
    "thought_type": "Mistake",
    "role": "Memory",
    "importance": 0.91,
    "tags": ["deployment", "incident"],
    "content": "Assumed the production environment already had the required migration."
  }
}
```

### 5. Later in the same or a future session

Search for the mistake, get its `index`, then append the lesson:

```json
{
  "tool": "mentisdb_append_retrospective",
  "arguments": {
    "chain_key": "project-alpha",
    "thought_type": "LessonLearned",
    "importance": 0.95,
    "confidence": 0.97,
    "tags": ["deployment", "lesson"],
    "concepts": ["migration-checklist"],
    "refs": [17],
    "content": "Before deployment, explicitly verify migration state instead of assuming environment parity."
  }
}
```

### 6. End of session

Store a checkpoint:

```json
{
  "tool": "mentisdb_append",
  "arguments": {
    "chain_key": "project-alpha",
    "thought_type": "Summary",
    "role": "Checkpoint",
    "importance": 0.9,
    "tags": ["session-summary"],
    "content": "We identified deployment migration drift, adopted a staged rollout plan, and added a migration verification checklist."
  }
}
```

## When A Thought Should Refer To A Previous Thought

In the MCP interface, a thought refers to previous thoughts through `refs`, which are prior thought indices.

The current remote MCP interface does not yet expose typed graph relations directly. The core `mentisdb` crate supports typed relations internally, but the MCP server currently exposes only `refs`.

A thought should usually refer to earlier thoughts when one of these is true:

- it corrects a previous belief
- it invalidates a previous assumption
- it summarizes earlier memories
- it records a lesson learned from an earlier mistake
- it reports an experiment result for an earlier hypothesis
- it records a strategy shift caused by earlier failures
- it creates a handoff or checkpoint derived from earlier work

### Good `refs` examples

Mistake followed by lesson learned:

1. append a `Mistake`
2. later append a `LessonLearned`, `Correction`, `Insight`, or `Summary`
3. include `refs` pointing to the mistake

Hypothesis followed by experiment:

1. append a `Hypothesis`
2. append an `Experiment`
3. append an `Insight` or `FactLearned`
4. reference the earlier hypothesis and experiment

Plan followed by strategy change:

1. append a `Plan`
2. later append a `StrategyShift`
3. reference the earlier plan

Summary of important context:

1. search or inspect prior relevant thoughts
2. append a `Summary`
3. reference the key earlier thought indices in `refs`

### Example: mistake in the past, lesson in the future

Past thought:

```json
{
  "tool": "mentisdb_append",
  "arguments": {
    "chain_key": "project-alpha",
    "thought_type": "Mistake",
    "content": "Used a staging-only configuration assumption in production.",
    "importance": 0.92,
    "tags": ["config", "incident"]
  }
}
```

Future thought referring back to it:

```json
{
  "tool": "mentisdb_append_retrospective",
  "arguments": {
    "chain_key": "project-alpha",
    "thought_type": "LessonLearned",
    "refs": [23],
    "importance": 0.94,
    "tags": ["lesson", "config"],
    "content": "Lesson learned: environment-specific assumptions must be verified explicitly before rollout."
  }
}
```

That makes the later lesson part of the same causal memory thread as the earlier mistake.

## What The Remote Interface Does Not Yet Expose

The core `mentisdb` crate supports more than the current MCP surface. In particular, the current MCP server does not yet expose:

- typed `relations`
- `session_id`
- direct filtering by `confidence`
- direct `since` or `until` date-range filters
- direct context resolution by thought id or index

Those capabilities exist or can be added on the crate side, but they are not yet part of the current remote MCP tool API.

## Remote-Agent Guidance

A remote agent should store a thought when the information is likely to matter later.

Good candidates:

- user preferences
- user traits or working style
- hard constraints
- decisions
- plans worth revisiting
- discovered facts
- insights
- mistakes
- corrections
- summaries
- checkpoints
- handoffs

Do not store everything.

Avoid storing:

- raw chain-of-thought
- transient filler
- duplicate observations with no new value
- secrets, unless the user explicitly wants them preserved

The right unit is not “everything the agent thought.”

The right unit is “a durable change in the agent’s model of the world or of the work.”
