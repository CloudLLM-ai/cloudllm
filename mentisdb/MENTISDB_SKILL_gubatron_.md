# MENTISDB SKILL: gubatron

## Core stance

Use MentisDB as the durable source of truth for the things that should survive chat loss, session resets, and tool changes. Keep it pragmatic, specific, and directly useful to future work. If a fact will change how code gets written, debugged, reviewed, or shipped later, it belongs in MentisDB.

## What to write

- Record durable engineering preferences, not transient chatter.
- Save working standards that repeatedly affect implementation quality.
- Write checkpoints that re-establish identity, priorities, and operating context after a long gap.
- Capture lessons that explain why a bug happened and how to avoid the class of failure next time.
- Prefer entries that improve real execution quality: concurrency correctness, UI responsiveness, user-facing error handling, commit discipline, and technical-debt reduction.

## What matters most for this agent

Based on gubatron's memory history, the highest-value memories are:

- Root-cause-first debugging guidance.
- Commit-message discipline that explains cause and fix, not just the surface patch.
- Preferences around EDT or UI-thread safety and bounded background execution.
- Guidance to turn low-level failures into user-facing messages.
- Non-obvious reasoning worth preserving in logs or comments.
- Broad cleanup decisions when they reduce debt and are clearly documented.

## How to write entries well

- Be concrete. Say what changed future behavior.
- Prefer one durable recommendation per memory over vague summaries.
- Use semantic types deliberately:
  `PreferenceUpdate` for stable coding standards,
  `Decision` for chosen approaches,
  `Constraint` for non-negotiable boundaries,
  `Insight` or `LessonLearned` for reusable debugging or design lessons,
  `Checkpoint` for resumable state.
- Tag for real retrieval paths such as `concurrency`, `ui-thread`, `error-handling`, `commit-discipline`, `technical-debt`, `frostwire`, or `gubatron`.
- Include enough wording that text search will find the memory later.

## Retrieval patterns

- On resume, read the latest checkpoints first to restore identity and priorities.
- Search by agent plus tags or concepts when looking for operating standards.
- Search by time window when reconstructing a specific debugging or design session.
- Query for `PreferenceUpdate`, `Decision`, `Constraint`, and `LessonLearned` before starting invasive code changes.
- Treat MentisDB as the replacement for scattered MEMORY files and undocumented tribal knowledge.

## Anti-patterns

- Do not store generic motivational text or redundant chat recap.
- Do not write memories that only restate obvious code.
- Do not save symptom-only bug notes; preserve the root cause.
- Do not create new patterns when an existing utility or convention already solves the problem.
- Do not let important standards live only in transient conversation.

## Operating recommendation

For gubatron-style work, the best MentisDB usage loop is:

1. Before coding, query for durable standards and prior constraints.
2. During debugging, write only when the root cause or a durable rule becomes clear.
3. After shipping, append the lesson or decision that will make the next change faster and safer.
4. On future sessions, resume from MentisDB instead of reconstructing intent from raw git history or chat logs.
