# MentisDB Skill Notes for Alden

## Purpose

Use MentisDB as the durable layer for the small set of facts that meaningfully improve future work. Alden's history points to precision over volume: store the memories that prevent repeat confusion, reduce waste, and preserve intent across sessions.

## What To Write

- Write bootstrap checkpoints that pin identity, working style, and core constraints early.
- Write stable engineering preferences that affect many future changes: commit discipline, refactoring style, API-surface preferences, lifecycle discipline, and dependency-reduction principles.
- Write decisions and corrections when they eliminate future ambiguity or prevent the same wrong path from being taken again.
- Write lessons learned only when they change future execution, not just to narrate what happened.

## What Matters Most

- Preferences and constraints with long half-lives are higher value than verbose session logs.
- Architectural intent is worth preserving when it helps future agents make smaller, more correct changes.
- Memories should help future work make the smallest precise cut, not justify broad rewrites.

## Retrieval Pattern

- Start a session by reading the bootstrap checkpoint and the highest-importance preference updates for the agent.
- Before a refactor, search for prior preferences, constraints, corrections, and decisions tied to code quality, API shape, lifecycle management, or dependency reduction.
- When resuming a stalled thread, prefer targeted retrieval by `agent_id`, `thought_type`, tags, and time window instead of dumping broad history.

## Anti-Patterns

- Do not use MentisDB as a raw transcript store.
- Do not record obvious implementation steps that can be recovered from git history or the code itself.
- Do not write expansive summaries when a short, surgical correction or preference update is enough.
- Do not store stack-trace-like noise; store the operating rule that future agents should follow instead.

## Operating Guidance

- Keep entries concise, durable, and actionable.
- Favor `PreferenceUpdate`, `Constraint`, `Decision`, `Correction`, and selective `LessonLearned` thoughts.
- Use tags and concepts that make retrieval narrow and predictable.
- If a memory would not materially improve a future decision, skip it.
- If a memory will reduce code churn, narrow future diffs, or preserve intent, write it.
