# MentisDB Skill: ai-systems-security-engineer

## Purpose

Use MentisDB to preserve security-relevant system knowledge that will prevent repeated mistakes, surface trust boundaries quickly, and make future reviews faster and more consistent.

## What To Write

- Record trust boundaries explicitly.
  Store where authority crosses systems such as frontend to backend, backend to wallet, backend to chain, or model to tool execution.
- Write durable known issues, not noisy scan output.
  MentisDB is most useful for persistent security posture items like XSS exposure, missing CSRF, overbroad CORS, missing rate limits, key-management risk, and unsafe rendering paths.
- Capture authentication and authorization rules as durable facts.
  Persist the actual access model, token types, identity extractors, and role restrictions so future agents can audit changes against a stable baseline.
- Save high-risk file pointers.
  Include the concrete files that define the security boundary so later reviewers can jump directly to the right code.
- Prefer lessons and constraints over generic summaries.
  Security memory should tell future agents what must not regress.

## What Not To Write

- Do not dump secrets, raw tokens, API keys, wallet material, or private signing data.
- Do not store transient vulnerability chatter unless it changes operating guidance.
- Do not write vague security statements without the threatened boundary or affected component.

## Retrieval Patterns

- Start with `agent_id=ai-systems-security-engineer` when you want the security baseline from this agent.
- Search by concepts such as `trust-boundary`, `authentication`, `authorization`, `xss`, `csrf`, `rate-limiting`, and `key-management`.
- Filter for high-importance security thoughts first, then widen to related architecture memories from other agents.
- When reviewing a feature, search both by system name and threat class.
  Example: combine `wallet`, `frontend`, or `solana` with `signing`, `xss`, or `auth`.

## Recommended Thought Shapes

- `Insight` for current security posture and concrete known issues.
- `Constraint` for hard rules such as “never store raw secrets in memory” or “frontend must not render raw HTML”.
- `LessonLearned` for post-incident or post-review corrections that future agents must reuse.
- `Correction` when a prior security assumption turned out to be wrong.

## Operating Guidance

- Write one durable memory whenever you identify a new trust boundary.
- Write one durable memory whenever you discover a systemic weakness that could survive beyond the current task.
- Include the attacker-relevant consequence.
  Example: “If leaked, anyone can create valid permits.”
- Include the enforcement point.
  Example: middleware extractor, auth handler, wallet bridge, signing module, or rendering boundary.
- Keep the memory short enough to be searchable, but concrete enough to drive action.

## Anti-Patterns

- Treating MentisDB as a vulnerability database.
  Use it for durable security knowledge, not exhaustive ticket tracking.
- Writing implementation details without the security implication.
- Writing threat descriptions without naming the boundary, asset, and blast radius.
- Storing security memories that cannot be operationalized by a later agent.

## Best Use Of MentisDB For Security

MentisDB is strongest when it stores the stable security model of a system: who can do what, where trust changes hands, which components are dangerous, and which known weaknesses must stay visible across sessions. Use it to preserve reviewer judgment, not to archive raw evidence.
