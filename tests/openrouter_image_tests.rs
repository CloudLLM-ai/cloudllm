// OpenRouter image generation (ImageGenerationProvider::OpenRouter).
//
// Offline tests cover the request body, response parsing, the factory and
// data-URI decoding. The live test needs OPENROUTER_API_KEY (skipped when
// unset) and costs about USD 0.03 per run.

use cloudllm::clients::openrouter::{
    build_openrouter_image_body, extract_openrouter_images, OPENROUTER_DEFAULT_IMAGE_MODEL,
};
use cloudllm::cloudllm::image_generation::{
    decode_base64, get_image_extension_from_base64, ImageGenerationOptions,
};
use cloudllm::cloudllm::{
    new_image_generation_client, new_image_generation_client_from_str,
    new_image_generation_client_with_model, ImageGenerationProvider,
};
use serde_json::json;
use std::str::FromStr;

#[test]
fn body_requests_image_modality_and_aspect_ratio() {
    let body =
        build_openrouter_image_body("google/gemini-3.1-flash-lite-image", "a cat", Some("16:9"));
    assert_eq!(body["model"], "google/gemini-3.1-flash-lite-image");
    assert_eq!(body["messages"][0]["role"], "user");
    assert_eq!(body["messages"][0]["content"], "a cat");
    assert_eq!(body["modalities"], json!(["image", "text"]));
    assert_eq!(body["image_config"]["aspect_ratio"], "16:9");
    assert_eq!(body["usage"]["include"], true);
}

#[test]
fn body_omits_image_config_without_aspect_ratio() {
    assert!(build_openrouter_image_body("m", "p", None)
        .get("image_config")
        .is_none());
    assert!(build_openrouter_image_body("m", "p", Some("  "))
        .get("image_config")
        .is_none());
}

#[test]
fn extracts_data_uri_and_https_images() {
    let body = json!({
        "choices": [{ "finish_reason": "stop", "message": {
            "content": "",
            "images": [
                { "type": "image_url", "image_url": { "url": "data:image/jpeg;base64,/9j/4AAQ" } },
                { "type": "image_url", "image_url": { "url": "https://cdn.example/x.png" } },
                { "type": "image_url", "image_url": { "url": "" } }
            ]
        }}]
    });
    let images = extract_openrouter_images(&body).unwrap();
    assert_eq!(images.len(), 2);
    assert_eq!(
        images[0].b64_json.as_deref(),
        Some("data:image/jpeg;base64,/9j/4AAQ")
    );
    assert!(images[0].url.is_none());
    assert_eq!(images[1].url.as_deref(), Some("https://cdn.example/x.png"));
}

#[test]
fn no_image_is_an_error_that_quotes_the_model_text() {
    let body = json!({ "choices": [{ "finish_reason": "stop",
        "message": { "content": "I can't draw that." } }] });
    let err = extract_openrouter_images(&body).unwrap_err();
    assert!(err.contains("no image"), "{}", err);
    assert!(err.contains("I can't draw that."), "{}", err);
}

#[test]
fn api_error_object_is_reported() {
    let body = json!({ "error": { "code": 402, "message": "Insufficient credits" } });
    let err = extract_openrouter_images(&body).unwrap_err();
    assert!(err.contains("Insufficient credits"), "{}", err);
}

#[test]
fn provider_parses_and_names() {
    let provider = ImageGenerationProvider::from_str("OpenRouter").unwrap();
    assert_eq!(provider, ImageGenerationProvider::OpenRouter);
    assert_eq!(provider.as_str(), "openrouter");
    assert_eq!(provider.display_name(), "OpenRouter");
}

#[test]
fn factories_pick_the_model() {
    let client = new_image_generation_client(ImageGenerationProvider::OpenRouter, "k").unwrap();
    assert_eq!(client.model_name(), OPENROUTER_DEFAULT_IMAGE_MODEL);
    let client = new_image_generation_client_from_str("openrouter", "k").unwrap();
    assert_eq!(client.model_name(), OPENROUTER_DEFAULT_IMAGE_MODEL);
    let client = new_image_generation_client_with_model(
        ImageGenerationProvider::OpenRouter,
        "k",
        "openai/gpt-5-image-mini",
    )
    .unwrap();
    assert_eq!(client.model_name(), "openai/gpt-5-image-mini");
    assert!(
        new_image_generation_client_with_model(ImageGenerationProvider::OpenRouter, "k", " ")
            .is_err()
    );
}

#[test]
fn base64_helpers_accept_data_uris() {
    // "Hello World"
    assert_eq!(
        decode_base64("data:text/plain;base64,SGVsbG8gV29ybGQ=").unwrap(),
        b"Hello World"
    );
    assert_eq!(decode_base64("SGVsbG8gV29ybGQ=").unwrap(), b"Hello World");
    assert_eq!(
        get_image_extension_from_base64("data:image/jpeg;base64,/9j/4AAQ"),
        "jpg"
    );
    assert_eq!(
        get_image_extension_from_base64("data:image/png;base64,iVBORw0KGgo"),
        "png"
    );
}

#[tokio::test]
async fn live_openrouter_image_generation() {
    let Ok(key) = std::env::var("OPENROUTER_API_KEY") else {
        eprintln!("skipping: OPENROUTER_API_KEY not set");
        return;
    };
    let client = new_image_generation_client(ImageGenerationProvider::OpenRouter, &key).unwrap();
    let response = client
        .generate_image(
            "A minimalist flat illustration of a lighthouse at dawn, no text.",
            ImageGenerationOptions {
                aspect_ratio: Some("16:9".to_string()),
                num_images: Some(1),
                ..Default::default()
            },
        )
        .await
        .expect("OpenRouter image generation failed");
    assert_eq!(response.images.len(), 1);
    let b64 = response.images[0]
        .b64_json
        .as_deref()
        .expect("expected a data URI");
    let bytes = decode_base64(b64).unwrap();
    assert!(
        bytes.len() > 10_000,
        "image too small: {} bytes",
        bytes.len()
    );
    assert!(matches!(
        get_image_extension_from_base64(b64),
        "jpg" | "png" | "webp"
    ));
}
