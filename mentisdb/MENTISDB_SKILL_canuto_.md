# MentisDB Skill: Canuto

## Purpose

Use MentisDB as the durable place for engineering conventions, implementation patterns, identity corrections, security lessons, and workflow decisions that should survive sessions. Do not treat it as a scratchpad for transient reasoning or a replacement for source control.

## What To Write

- Write memories when a rule becomes reusable across tasks.
  Canuto's strongest memories are not one-off facts. They are stable conventions such as CLI flag handling, WordPress endpoint structure, BotCommand composability rules, and Rust design standards.
- Write corrections immediately when identity or attribution is wrong.
  The most important Canuto lessons are identity corrections: always write as `agent_id=canuto`, `agent_name=Canuto`, `agent_owner=@gubatron`, and never reuse the chain key as the producer id.
- Write implementation patterns, not only outcomes.
  Good memories capture the actual reusable shape of a solution: where routes live, how auth is validated, how flags are stripped from CLI args, how behavior toggles belong on command structs, and how output escaping should match context.
- Write security guidance when tools or external systems are involved.
  Memories about tool execution, WordPress auth, schema validation, least privilege, and auditability are high-value because they prevent expensive mistakes later.

## What Kinds Of Memories Matter Most

- Canonical identity rules
- Architecture and codebase conventions
- Reusable endpoint and CLI patterns
- Stable parsing or content-structure knowledge
- Corrections to prior assumptions
- Security review heuristics around tools and integrations

These are better MentisDB material than low-signal activity logs like "changed file X" unless that change produced a reusable rule.

## Retrieval Patterns

- Start with `agent_id=canuto` for Canuto-authored guidance.
- Search by stable tags and concepts, not vague text alone.
  Good examples from Canuto history: `identity`, `canonicalization`, `wordpress`, `rest-api`, `cli`, `arg-parsing`, `rust-design`, `security`.
- Query corrections before acting on older memories.
  Canuto's history proves why: older `borganism-brain` producer identity had to be superseded by canonical `canuto`.
- Prefer looking for patterns over isolated facts.
  Search for clusters like `wordpress` + `authentication`, or `cli` + `publish`, or `rust` + `agent-conventions`.

## Anti-Patterns

- Do not write under the chain key instead of the agent's canonical identity.
- Do not store only symptoms when you can store the reusable rule behind them.
- Do not rely on ad hoc `MEMORY.md` files or chat history for long-lived conventions.
- Do not record implementation details without the operating principle that makes them reusable.
- Do not treat MentisDB as a dump of every action; store the lessons that change future behavior.

## Practical Guidance

- When a change introduces a new cross-cutting rule, write a memory that names the pattern explicitly.
  Example categories: authenticated custom REST route pattern, CLI optional-flag pattern, composability rule for behavior flags, or canonical identity policy.
- Include tags and concepts that future searches will actually use.
  Favor domain tags plus intent tags, for example `wordpress`, `authentication`, `cli`, `publish`, `identity`, `canonicalization`, `security`.
- Use corrections when superseding earlier behavior.
  If an old pattern becomes invalid, write the correction instead of assuming future agents will infer it.
- Preserve confidence and importance discipline.
  High-confidence, high-importance memories should be reserved for rules that are likely to be reused and costly to forget.

## Recommended Operating Habit

After any task that reveals a reusable convention, ask:

1. Will this matter again outside the current session?
2. Is there a stable rule here, not just a local edit?
3. Would a future agent search for this by tag, concept, or identity?
4. Does this correct or supersede older memory?

If the answer is yes, write it to MentisDB immediately and tag it so retrieval will be obvious later.
