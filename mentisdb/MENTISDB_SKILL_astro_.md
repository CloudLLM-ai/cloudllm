# MentisDB Skill: Astro

## Core Stance

Use MentisDB as the durable semantic memory layer for preferences, constraints, decisions, lessons learned, corrections, checkpoints, and multi-agent handoffs. Do not treat it as a generic artifact store. Full prompt bundles, scripts, assets, or executable skill packages belong somewhere else; MentisDB should hold the semantic facts that help agents resume, search, and reason.

## What To Write

- Write `PreferenceUpdate` thoughts for stable user and team expectations: rustdoc coverage, separate tests, small coherent commits, zero-warning standards, and similar working norms.
- Write `Constraint` thoughts for architectural boundaries that must not drift, especially crate dependency rules, ownership boundaries, and storage/integrity requirements.
- Write `Decision` thoughts when a design choice changes how the system should be operated or integrated, such as daemon shape, storage adapter defaults, protocol compatibility, or publish order.
- Write `Correction` thoughts when documentation, assumptions, or behavior were wrong and future agents need the fixed version quickly.
- Write `LessonLearned` or retrospective thoughts after migrations, schema changes, rebrands, protocol changes, or non-obvious breakages. Astro’s strongest MentisDB guidance came from these.
- Write `Checkpoint` summaries at session boundaries so the next agent can reload state without replaying the full implementation history mentally.

## What Makes A Good Thought

- Be concrete. Name the exact subsystem, failure mode, or policy.
- Record why the lesson matters operationally, not just what happened.
- Prefer durable rules over ephemeral chat paraphrases.
- Add tags and concepts that future queries will actually use: `migration`, `registry`, `mcp`, `storage`, `rebrand`, `docs`, `integrity`, `compatibility`.
- Use shared-chain identity correctly: stable `agent_id`, readable `agent_name`, optional `agent_owner`.

## Retrieval Patterns That Pay Off

- Start sessions by loading recent checkpoints and retrospectives for the relevant agent or chain.
- Search by tags and concepts around migration and interface changes before editing storage, registry, MCP, REST, or startup behavior.
- Query by agent when you want domain-specific guidance, then broaden to the shared chain for cross-agent context.
- Treat checkpoint thoughts as compression anchors and lessons learned as anti-regression anchors.
- When a change touches public surface area, search for related memories about docs, metadata, compatibility aliases, and startup behavior before coding.

## High-Value Memory Categories

- Migration ordering and storage integrity.
- Registry and identity semantics.
- MCP/REST parity and metadata drift.
- Compatibility strategy during renames.
- Rustdoc, doctest, and workspace-quality expectations.
- Startup observability and operational visibility.

## Anti-Patterns

- Do not store only final code outcomes; store the constraint or lesson that explains why the final shape matters.
- Do not collapse MentisDB into a skill/package manager.
- Do not rely on transient chat context for important workflow rules.
- Do not make breaking renames without preserving compatibility memories and rollout lessons.
- Do not change request schemas without also updating MCP metadata and docs.
- Do not treat migrations as one-shot version bumps; think reconciliation, repair, and active-file verification.

## Safety And Quality Rules

- Preserve append-only integrity and make migrations explicit.
- Keep library, MCP, and REST surfaces aligned; incomplete parity creates operational holes.
- Prefer asymmetric compatibility during product renames: new names primary, legacy aliases retained only where rollout safety requires them.
- Keep doctest examples crate-local or clearly ignored when cross-crate coupling would make them brittle.
- Favor standard structured logging and observable startup summaries over opaque stderr noise.

## Practical Operating Guidance

- On session start: read checkpoints, then retrospectives, then current constraints.
- Before touching storage or startup code: search `migration`, `storage`, `integrity`, `registry`, and `rebrand`.
- Before touching interfaces: search `mcp`, `rest`, `docs`, `metadata`, and `compatibility`.
- After any hard-earned fix: write one durable retrospective immediately while the failure mode is still precise.
- After major milestones: write a checkpoint that compresses the current architectural truth for the next agent.
