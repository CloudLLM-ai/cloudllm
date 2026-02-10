# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.8.0] - 2026-02-09

### Added
- **ThoughtChain** — persistent, hash-chained agent memory
  - Append-only log of findings, decisions, compressions, checkpoints
  - SHA-256 hash chain for tamper-evident integrity verification
  - Back-references (refs) for graph-based context resolution
  - Disk-persisted as newline-delimited JSON (.jsonl)
  - Collision-resistant filenames from agent identity fingerprint
  - Resume from any thought: `resolve_context()` walks ref graph
  - `Agent::resume_from_chain()` / `resume_from_latest()` constructors
- **Pluggable context collapse strategies**
  - `ContextStrategy` trait with `should_compact()` + `compact()`
  - `TrimStrategy` (default): LLMSession built-in oldest-first trimming
  - `SelfCompressionStrategy`: LLM writes structured save file, persisted to ThoughtChain
  - `NoveltyAwareStrategy`: entropy-based trigger with delegated compression
  - `context_collapse_strategy()` builder + `set_context_collapse_strategy()` runtime swap
- **Runtime tool hot-swapping**
  - ToolRegistry wrapped in `Arc<RwLock>` for runtime mutation
  - `Agent::add_protocol()` / `remove_protocol()` while agent is running
  - `with_shared_tools()` for sharing mutable registries across agents
- `CloudLLMConfig` struct for ThoughtChain directory configuration
- `Agent::with_max_tokens()` builder method
- `LLMSession::client()`, `clear_history()`, `inject_message()`, `estimated_history_tokens()` accessor methods

### Changed
- **BREAKING**: Agent wraps `LLMSession` for per-agent conversation memory
  - Agent no longer holds raw `Arc<dyn ClientWrapper>` — access via `agent.client()`
  - Each agent owns its `LLMSession` with rolling history and token tracking
  - `Agent::fork()` creates lightweight copies for parallel execution (replaces Clone)
  - Agent fields (`session`, `tool_registry`) are now private
- **BREAKING**: `with_tools()` now takes owned `ToolRegistry` (not `Arc<ToolRegistry>`)
- Orchestration parallel and hierarchical modes use `agent.fork()` instead of manual Agent construction

## [0.7.2] - 2025-12-12

### Added
- Support for OpenAI GPT-5.2 models: `gpt-5.2`, `gpt-5.2-chat-latest`, and `gpt-5.2-pro`
- New `Model::GPT52` enum variant for `gpt-5.2` (complex reasoning, broad world knowledge, code-heavy and multi-step agentic tasks)
- New `Model::GPT52ChatLatest` enum variant for `gpt-5.2-chat-latest` (ChatGPT's production deployment of GPT-5.2)
- New `Model::GPT52Pro` enum variant for `gpt-5.2-pro` (for problems requiring harder thinking)

### Changed
- Updated `model_to_string()` function to support new GPT-5.2 model variants
- Refactored `OpenAIClient` implementation to improve code organization
- Improved code formatting and import organization across the codebase for better maintainability
- Reformatted long function signatures and method chains for improved readability

### Fixed
- Fixed code formatting inconsistencies in examples and library code
- Improved formatting of long lines in `openai_bitcoin_price_example.rs`, `openai_web_search_example.rs`, and `filesystem_example.rs`
- Standardized import ordering across all source files

## [0.7.1] - 2024-XX-XX

### Fixed
- Fixed test suite and added Bitcoin price example

## [0.7.0] - 2024-XX-XX

### Added
- OpenAI Responses API tool support with dual API routing

## [0.6.3] - 2024-XX-XX

### Added
- xAI Responses API support for agentic tool calling
