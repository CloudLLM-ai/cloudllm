# Changelog

All notable changes to this project will be documented in this file.

## [0.15.4] - 2026-05-06

### Added

- **Grok client model enum expanded up to Grok 4.3**.
  - Added `Model::Grok4` (`grok-4`).
  - Added `Model::Grok4Latest` (`grok-4-latest`).
  - Added `Model::Grok4Fast` (`grok-4-fast`).
  - Added `Model::Grok43` (`grok-4.3`).
  - Added `Model::Grok43Latest` (`grok-4.3-latest`).
  - Updated enum doc comment to "May 2026".

### Fixed

- **Updated image generation test assertion** to match the `gpt-image-2` default introduced in 0.15.3.

## [0.15.3] - 2026-04-28

### Changed

- **OpenAI image generation now uses `gpt-image-2` by default** (upgraded from `gpt-image-1.5`).
  - Added `ImageModel::GPTImage2` variant with string mapping to `"gpt-image-2"`.
  - Updated `OpenAIClient::generate_image()` to default to `gpt-image-2`.
  - Updated `model_name()` to return `"gpt-image-2"`.
  - Updated display name from `"OpenAI (DALL-E 3)"` to `"OpenAI (gpt-image-2)"`.
