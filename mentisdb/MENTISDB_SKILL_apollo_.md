# MentisDB Skill Notes From Apollo

## Core Position

Use MentisDB as the durable execution memory for multi-step engineering work, not as a passive journal. Apollo's history is strongest when it stores:

- architecture decisions that shape future work
- battle-tested implementation gotchas
- security constraints and known risks
- WIP checkpoints before interruption
- explicit next-session restart instructions

Do not rely on transient chat context or ad hoc `MEMORY.md` files when the work spans backend, frontend, blockchain, and security concerns.

## What To Write

Write a thought when one of these becomes true:

- You learned a non-obvious implementation trap that would waste time again.
- You settled an architectural choice that downstream work should not re-litigate.
- You discovered a security boundary, unsafe default, or production risk.
- You reached a handoff or restart point and need the next session to resume fast.
- You finished a WIP and need to record the transition from "planned" to "committed".

Apollo's best memories are concrete and operational. They include exact APIs, route names, env vars, protocol details, wallet behavior, and framework-specific gotchas. That makes later retrieval actually useful.

## What Matters Most

Prioritize these memory shapes:

- `Insight` for hard-earned framework, language, or protocol lessons.
- `Decision` for platform architecture and stack choices.
- `PreferenceUpdate` for persistent engineering standards that should shape future code.
- `Summary` with role `Checkpoint` for restart state, WIP capture, and next-session instructions.

Apollo's history shows that compact, high-density technical memories outperform vague summaries. A good memory should preserve the exact thing that would otherwise need rediscovery.

## Retrieval Patterns

Start every resumed task by searching for the project tag first, then narrow by subsystem.

Recommended search patterns:

- project restart: search by `tags_any=["meatpuppets"]`
- architecture refresh: search `thought_types=["Decision","Summary"]`
- gotcha recovery: search `thought_types=["Insight"]`
- restart state: search `roles=["Checkpoint"]`
- time-bounded work review: search by `since` / `until`

Apollo's own checkpoint explicitly says to read shared-brain thoughts tagged `meatpuppets` before starting. Follow that pattern. Project tag first, subsystem tag second.

## Tagging And Concepts

Use tags and concepts to make memories composable across specialties.

Good Apollo-style tags:

- project: `meatpuppets`
- layer: `backend`, `frontend`, `solana`, `security`
- mechanism: `permit`, `wallet`, `anchor`, `sqlx`, `axum`
- workflow state: `wip`, `committed`, `next-session`

Good Apollo-style concepts:

- domain nouns such as `task-cancellation`, `on-chain-refund`, `wallet-integration`
- architectural anchors such as `platform`, `tech-stack`, `marketplace`
- implementation anchors such as `cancel-permit`, `anchor-discriminator`, `wasm-bindgen`

If a memory crosses boundaries, tag all relevant layers. Apollo got value from storing one shared memory that backend, frontend, blockchain, and security work could all benefit from.

## Operating Guidance

- Write checkpoints before restarts, not after you forget the active state.
- Record exact route names, env vars, account layouts, message formats, and edge conditions.
- Capture unfinished work as WIP, then write a second checkpoint when it becomes committed.
- Store next-session instructions explicitly. "Test end-to-end on devnet" is better than a generic "continue later."
- Preserve security debt in memory even if you are not fixing it now.
- Treat cross-stack lessons as first-class memories; a full-stack agent benefits from storing frontend, backend, chain, and auth knowledge in one place.

## Anti-Patterns

- Do not write generic reflections with no actionable detail.
- Do not store only code-change summaries when the real value is the reasoning or gotcha.
- Do not skip memory writes for WIP flows that may be interrupted.
- Do not omit tags; Apollo's retrieval strategy depends on project and subsystem tagging.
- Do not bury security concerns inside unrelated implementation notes.

## Quality Bar

Apollo's memory trail implies this standard:

- specific over broad
- operational over inspirational
- restart-oriented over historical
- cross-disciplinary when the system is cross-disciplinary
- durable enough that another agent could continue the work without re-deriving the logic

If a future agent can resume the task, avoid the main pitfalls, and know what to query next, the memory was worth writing.
