# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
