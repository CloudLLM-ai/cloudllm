# DRAFT: MentisDB White Paper

Note: the product is now named MentisDB. Some code artifacts in this repo may
still carry the legacy `mentisdb` name during the transition.

**Author:** Angel Leon

## Abstract

Modern agent frameworks are still weak at long-term memory. In practice, memory is often reduced to ad hoc prompt stuffing, fragile `MEMORY.md` files, or proprietary session state that is hard to inspect, hard to transfer, and easy to lose or tamper with. MentisDB is a simple, durable alternative: an append-only, semantically typed memory ledger for agents and teams of agents.

MentisDB stores important thoughts, decisions, corrections, constraints, checkpoints, and handoffs as structured records in a hash-chained log. The chain model is storage-agnostic through a storage adapter layer, with binary storage as the current default backend and JSONL still supported. This makes memory replayable, queryable, portable, and auditable. It improves agent continuity across sessions, supports collaboration across specialized agents, and creates a clear foundation for future transparency, accountability, and regulatory compliance.

## Problem Statement

Today’s agent memory systems are messy.

- Long-term memory is often just another prompt.
- Durable memory is often a mutable text file.
- Context handoff between agents is brittle and lossy.
- Memory is rarely semantic enough for precise retrieval.
- Auditability and provenance are usually missing.

This creates operational and governance problems.

- Agents forget important constraints.
- Teams of agents repeat mistakes.
- Supervisors cannot easily inspect how a decision evolved.
- A malicious or faulty agent can rewrite or erase context.
- Future regulation will likely require stronger traceability than current frameworks provide.

## MentisDB

MentisDB is a lightweight memory primitive for agents.

Each memory record, or thought, is:

- append-only
- timestamped
- semantically typed
- attributable to an agent
- linkable to previous thoughts
- hashed into a chain for tamper detection

Rather than storing raw chain-of-thought, MentisDB stores durable cognitive checkpoints: facts learned, plans, insights, corrections, constraints, summaries, handoffs, and execution state.

## Core Design

MentisDB combines five ideas.

### 1. Semantic Memory

Thoughts are explicitly typed. This makes memory retrieval much more useful than searching free-form logs or transcripts.

Examples include:

- preferences
- user traits
- insights
- lessons learned
- facts learned
- hypotheses
- mistakes
- corrections
- constraints
- decisions
- plans
- questions
- ideas
- experiments
- checkpoints
- handoffs
- summaries

### 2. Hash-Chained Integrity

Thoughts are stored in an append-only hash chain, effectively a small blockchain for agent memory. Each record includes the previous hash and its own hash. This makes offline tampering detectable and gives the chain an auditable history.

This is not presented as a public cryptocurrency system. It is a practical blockchain-style ledger for memory integrity.

### 3. Shared Multi-Agent Memory

MentisDB supports multiple agents writing to the same chain. Each thought carries a stable:

- `agent_id`

Agent profile metadata such as display name, owner, aliases, descriptions, and public keys live in a per-chain agent registry rather than being duplicated inside every thought record.

This allows a single chain to represent the work of a team, a workflow, a tenant, or a project. Memory can then be searched not only by content and type, but also by who produced it, while keeping the durable thought records smaller and the identity model more consistent.

The agent registry is no longer just passive metadata inferred from old thoughts. It can now be administered directly through library calls, MCP tools, and REST endpoints. That means agents can be pre-registered, documented, disabled, aliased, or provisioned with public keys even before they start writing memories.

### 4. Query, Replay, and Export

The chain can be:

- discovered
- searched
- filtered
- replayed
- summarized
- exported as `MEMORY.md`
- served over MCP
- served over REST

This makes MentisDB usable by agents, services, dashboards, CLIs, and orchestration systems.

In practice, that also means a daemon can tell a caller:

- which chain keys already exist
- which distinct agents are writing to a shared chain
- what the full registry metadata says about those agents
- which schema version each chain uses
- which storage adapter each chain uses

That makes shared brains easier to inspect and safer to reuse across teams of
agents.

### 5. Swappable Storage

MentisDB now separates the chain model from the storage backend.

- A `StorageAdapter` interface handles persistence.
- A `BinaryStorageAdapter` provides the current default implementation.
- A `JsonlStorageAdapter` remains available as a line-oriented, inspectable format.
- Additional adapters can be added without changing the core memory model.

This keeps the system simple today while allowing more efficient storage engines in the future.

### 6. Versioned Schemas And Migration

MentisDB schemas are versioned.

- schema version `0` was the original format
- schema version `1` adds explicit versioning and optional signing metadata
- daemon startup can migrate discovered legacy chains before serving traffic
- startup can reconcile older active files into the configured default storage adapter
- startup can attempt repair when the expected active file is missing or invalid but another valid local source exists

This matters because append-only memory still evolves. A durable memory system needs a way to add fields, change attribution strategy, and improve integrity without abandoning existing chains.

The daemon also maintains a MentisDB registry so callers and operators can quickly inspect:

- what chains exist
- which schema version each chain uses
- which storage adapter each chain uses
- where each chain is stored
- how many thoughts and registered agents each chain currently has

## Data Model

MentisDB deliberately separates memory creation, memory storage, and memory retrieval.

### ThoughtInput

`ThoughtInput` is the caller-authored memory proposal.

It contains the semantic payload:

- the thought content
- the thought type
- the thought role
- tags and concepts
- confidence and importance
- references and semantic relations
- optional session metadata
- optional agent profile hints used to populate or update the registry
- optional signing metadata

It does not contain the final chain-managed fields such as index, timestamp, or hashes.

This is important because an agent should be able to say what memory it wants to record, but it should not directly forge the chain mechanics that make the ledger trustworthy.

### Thought

`Thought` is the committed durable record written into the chain.

MentisDB derives it from a `ThoughtInput` and adds the system-managed fields:

- `schema_version`
- `id`
- `index`
- `timestamp`
- `agent_id`
- optional `signing_key_id`
- optional `thought_signature`
- `prev_hash`
- `hash`

This prevents confusion between proposed memory content and accepted memory state.

### ThoughtType And ThoughtRole

These two concepts are intentionally different.

- `ThoughtType` describes what the memory means
- `ThoughtRole` describes how the system is using that memory

For example:

- `Decision` is a thought type
- `Checkpoint` is usually a thought role
- `LessonLearned` is a thought type
- `Retrospective` is a thought role

That separation avoids mixing semantics with workflow mechanics.

This distinction is especially useful for reflective agent loops. A hard-won
fix might be stored as:

- `Mistake`
- `Correction`
- `LessonLearned`

with the final distilled guidance marked using the `Retrospective` role. That
lets future agents retrieve not just what happened, but what they should do
differently next time.

### ThoughtQuery

`ThoughtQuery` is the read-side filter over committed thoughts.

It does not create memories and it does not modify the chain. It simply retrieves relevant thoughts by type, role, agent identity, text, tags, concepts, importance, confidence, and time range.

## Use Cases

### Long-Term Agent Memory

A persistent agent can return days or weeks later and recover the important facts, preferences, constraints, and ongoing plans that matter for continuing work.

### Multi-Agent Handoff

One agent can shut down and hand work to another. A planning agent can hand off to an implementation agent. A coding agent can hand off to a debugging agent. A generalist can hand off to a specialist with different tools or cognitive strengths.

The receiving agent does not need the full conversation transcript. It can reconstruct the relevant state from the MentisDB.

### Team Coordination

When multiple agents collaborate, MentisDB provides a shared memory surface for:

- discoveries
- decisions
- mistakes
- lessons learned
- checkpoints
- handoff markers

This reduces repeated work and allows agents to build on each other’s progress.

### Human Oversight

Operators can inspect a chain directly, query it, browse the agent registry, or export it as Markdown. This makes it easier to understand what happened and why.

The current daemon startup output also leans into operability. It prints a readable catalog of every HTTP endpoint it serves, followed by a summary of every registered chain and the known agents in each chain, including per-agent thought counts and descriptions. That is a small but important step toward a future ThoughtExplorer-style web interface.

## Transparency, Traceability, and Regulation

As agent systems become more powerful, regulation is likely to require stronger accountability. Governments and enterprises will increasingly ask:

- What did the agent know at the time?
- What constraints did it receive?
- Why was a decision made?
- What was learned after a failure?
- Who or what changed the memory state?

MentisDB is a strong primitive for answering those questions. It does not solve every governance problem, but it gives systems a durable and inspectable memory record instead of an opaque prompt history.

This is useful for:

- internal audits
- incident review
- compliance workflows
- model behavior analysis
- regulated industries that need traceability

## Anti-Tamper and Future Signing

The current hash chain makes memory rewrites detectable, but a sufficiently privileged malicious actor could still rewrite the full chain and recompute hashes.

For that reason, the thought format now includes optional signing hooks:

- `signing_key_id`
- `thought_signature`

Those fields allow a thought to carry a detached signature over the signable payload, while public verification keys can live in the agent registry.

This is still an early foundation rather than a full trust model. The current implementation does not yet require signatures or enforce a public-key policy, but the schema is now shaped to support Ed25519-style agent identity and stronger provenance controls.

Stronger controls could include signatures from a human-controlled or centrally controlled authority that agents themselves cannot control.

That authority could:

- sign checkpoints
- anchor chain heads externally
- validate approved memory states
- make unauthorized rewrites detectable even if an agent has local write access

This is an important future direction for environments where agents may attempt to cover their tracks.

## Why MentisDB Matters

MentisDB turns agent memory from an informal prompt trick into durable infrastructure.

It helps solve:

- long-term memory
- semantic retrieval
- context handoff
- multi-agent collaboration
- transparency
- traceability
- tamper detection

In short, MentisDB is designed to be a practical memory ledger for real agent systems.

## Conclusion

Agent systems need a better memory foundation than mutable text files, prompt stuffing, and framework-specific hidden state. MentisDB provides a simple and durable alternative: semantic memory records stored in an append-only blockchain-style chain, queryable across time and across agents, with a storage layer that can evolve without rewriting the memory model.

It is useful today for persistent agents and multi-agent teams, and it points toward a future where agent systems can be both more capable and more accountable.

\
**Angel Leon**
