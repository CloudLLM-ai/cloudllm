# MentisDB Skill: leptos-frontend-engineer

## What To Store

- Write memories for frontend issues that are framework-specific, browser-specific, or wallet-integration-specific. These are exactly the lessons most likely to be forgotten and repeated.
- Capture UI constraints that are non-obvious from the type system alone: Leptos component limitations, WASM build pitfalls, browser API gaps, wallet-provider quirks, and auth-flow rules.
- Save reusable frontend conventions such as auth-request patterns, navigation rules, and CSS token decisions when they affect multiple screens.

## How To Write It

- Prefer `Insight` for frontend gotchas and `Correction` when a common assumption turns out to be wrong.
- Name the concrete component, API, or runtime boundary involved: `gloo_timers`, `<A>`, `StoredValue::new_local()`, `wasm32-unknown-unknown`, `TextEncoder`, `Wallet Standard`, `window.location.set_href`.
- Keep the memory terse but explicit about the correct replacement pattern, not just the broken one.
- Group related frontend lessons into one structured memory when they belong to the same stack boundary, such as Leptos 0.7 or wallet integration.

## Retrieval Patterns

- Search by framework and boundary terms first: `leptos`, `wasm`, `wallet`, `metamask`, `phantom`, `auth`, `router`, `frontend`.
- Use tags and concepts that mirror how you debug frontend issues later, not just product names.
- Query MentisDB before touching browser-wallet code or WASM build setup. Those areas have the highest density of brittle, non-obvious lessons.

## Operating Guidance

- Use MentisDB to preserve "this looks normal but is wrong here" knowledge. Frontend bugs often come from patterns that are valid elsewhere but invalid in Leptos or the browser.
- Record environment-specific workflow rules like where to run `cargo check --target wasm32-unknown-unknown`. Build-context lessons save a lot of churn.
- Persist UX-critical auth and navigation behavior. Logout flows, full-page reload requirements, and local-storage session conventions are easy to break later.
- Treat wallet interoperability as memory-worthy by default. Provider-specific signing behavior is too expensive to relearn from scratch.

## Anti-Patterns

- Do not store generic advice like "test in the browser" or "watch for frontend bugs." MentisDB is most valuable for concrete stack-specific traps.
- Do not omit the exact workaround. A memory that says a component is limited but does not show the replacement pattern is too weak.
- Do not flood memory with visual polish notes unless they affect system-wide conventions or repeated implementation choices.

## Quality Bar

- A good frontend MentisDB entry should tell the next engineer exactly what to search for and what pattern to use instead.
- If the memory would not prevent a broken build, a broken wallet flow, or a repeated Leptos misuse, it is probably not strong enough.
