//! Offline checks of the Gemini model names (no API key, no network).

use cloudllm::clients::gemini::{model_to_string, Model};

/// `gemini-2.5-flash` is no longer served to new users (HTTP 404, Nov 2026); the current
/// Flash models must map to the exact names the API lists.
#[test]
fn current_flash_models_map_to_api_names() {
    assert_eq!(model_to_string(Model::Gemini38Flash), "gemini-3.8-flash");
    assert_eq!(model_to_string(Model::Gemini37Flash), "gemini-3.7-flash");
    assert_eq!(model_to_string(Model::Gemini36Flash), "gemini-3.6-flash");
}
