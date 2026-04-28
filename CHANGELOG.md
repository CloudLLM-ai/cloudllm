# Changelog

All notable changes to this project will be documented in this file.

## [0.15.3] - 2026-04-28

### Changed

- **OpenAI image generation now uses `gpt-image-2` by default** (upgraded from `gpt-image-1.5`).
  - Added `ImageModel::GPTImage2` variant with string mapping to `"gpt-image-2"`.
  - Updated `OpenAIClient::generate_image()` to default to `gpt-image-2`.
  - Updated `model_name()` to return `"gpt-image-2"`.
  - Updated display name from `"OpenAI (DALL-E 3)"` to `"OpenAI (gpt-image-2)"`.
