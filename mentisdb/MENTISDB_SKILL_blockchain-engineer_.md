# MentisDB Skill Notes from `blockchain-engineer`

## What deserves a memory write

- Write once a pattern becomes a protocol invariant, not just a local implementation detail.
- Capture the exact bytes, field names, RPC flow, and environment variables that make blockchain integrations succeed or fail.
- Prefer writing lessons after non-obvious fixes, especially wallet interoperability bugs, signing payload details, and per-currency config requirements.

## Highest-value memory shapes

- `Insight` for contract architecture, wallet behavior, fee math, and integration gotchas.
- `Correction` when a field name, encoding format, or signing flow was previously wrong.
- `LessonLearned` when a bug was expensive to rediscover and likely to recur across agents or chains.
- Rich `concepts` and `tags` matter. Good examples from this history are `solana`, `anchor`, `erc8004`, `ed25519`, `escrow`, `cancel-flow`, `wallet`, and `config`.

## How to write useful blockchain memories

- Include canonical payload structure, not vague prose. If signing depends on `task_id_bytes`, `poster pubkey`, `amount`, `mint`, `deadline`, and `metadata_hash`, write all of that down.
- Record exact env var names. `SOLANA_TASK_ESCROW_PROGRAM_ID` is useful; “the Solana program id env var” is not.
- Write cross-wallet differences explicitly. MetaMask and Phantom are not interchangeable, and browser bridge assumptions like `Buffer.from()` can break production flows.
- Prefer durable records for flows that span backend, frontend, and chain. The cancel flow is valuable because it preserves the whole sequence, not one isolated endpoint.
- Store numeric conventions precisely. Fee splits, decimal precision, chain ids, and PDA derivation seeds should be written as operational facts.

## Retrieval patterns that work

- Search by protocol surface first: `concepts_any=["ed25519","escrow","cancel-flow"]`.
- Use tags for ecosystem slices: `tags_any=["solana","erc8004","anchor"]`.
- Filter by agent when you want the specialist view: `agent_ids=["blockchain-engineer"]`.
- Combine time windows with domain concepts when debugging regressions introduced after a deploy or wallet integration change.

## Anti-patterns

- Do not store “wallet support fixed” without naming the wallet, required API, and failure mode.
- Do not summarize signing or serialization issues without the concrete encoding rule.
- Do not rely on chat memory for protocol invariants, chain ids, env vars, or fee formulas.
- Do not write memories that are purely code-location pointers without the actual rule those files enforce.

## Practical operating guidance

- Treat MentisDB as the place for blockchain invariants that multiple agents must reuse safely.
- Write memories when a detail crosses trust boundaries: backend to wallet, wallet to chain, or chain to identity layer.
- Favor append-only factual guidance over speculative design notes.
- If a future agent could lose funds, sign invalid payloads, or break wallet compatibility by forgetting the lesson, it belongs in MentisDB.
