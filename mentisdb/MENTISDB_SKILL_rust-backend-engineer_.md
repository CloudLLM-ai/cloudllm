# MentisDB Skill: rust-backend-engineer

## What To Store

- Write memories when you hit a backend failure mode that will recur: ORM quirks, serialization mismatches, env-var validation rules, schema migration gotchas, transaction handling, or test-cleanup ordering.
- Prefer concrete, implementation-level lessons over vague summaries. Good MentisDB memories name the exact crate, API, error shape, commit, or runtime mismatch that caused the issue.
- Store patterns that define the happy path as well as the failure path. A reusable handler flow or test pattern is as valuable as a bug fix.

## How To Write It

- Use `Insight` for durable backend lessons and `Correction` when replacing a wrong assumption.
- Include enough detail to act without reopening old code: exact macro behavior, feature-flag names, transaction idioms, cleanup ordering, and response-field names.
- Favor compact numbered guidance when several related backend lessons belong together.
- Record the reason the rule exists, not just the rule. "Use `&mut *tx`" is weaker than explaining the sqlx 0.8 transaction behavior behind it.

## Retrieval Patterns

- Search by stable technical anchors first: crate names like `sqlx`, `axum`, `serde_json`, feature names, env-var names, or domain nouns like `transaction`, `cleanup`, `migration`.
- Add tags and concepts that match the backend domain you will search later, such as `sqlx`, `axum`, `serialization`, `tests`, `schema`, `env`, and the project name.
- Before large backend changes, query MentisDB for prior lessons in the same subsystem instead of rediscovering edge cases during implementation.

## Operating Guidance

- Treat MentisDB as a backend incident-prevention log. If a bug would waste another engineer's hour later, it belongs in memory.
- Capture migration and compatibility lessons immediately after the fix lands, while the exact failure mode is still fresh.
- Store test-harness lessons aggressively. Backend regressions often come from cleanup order, schema assumptions, or mock-auth shortcuts that are easy to forget.
- Use MentisDB to preserve framework upgrade knowledge. Dependency migrations are where brittle, high-value lessons accumulate fastest.

## Anti-Patterns

- Do not write generic memories like "sqlx was tricky" or "be careful with env vars." Those are not searchable enough to prevent repeat failures.
- Do not omit the concrete mismatch. Backend memories should preserve the exact type/OID mismatch, parsing shape, feature-flag split, or HTTP response contract.
- Do not treat MentisDB as a dump of every coding step. Save the lessons that change future implementation choices.

## Quality Bar

- A good backend MentisDB entry should let another Rust engineer avoid the same bug without re-running the failure.
- If the memory would not change how you design the next handler, migration, test, or config parser, it is probably too weak to keep.
