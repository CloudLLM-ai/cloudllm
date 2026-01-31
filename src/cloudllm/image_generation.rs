//! Image generation trait for CloudLLM providers.
//!
//! This module defines the [`ImageGenerationClient`] trait that allows different
//! LLM providers (OpenAI, Grok, Gemini) to generate images from text prompts.
//!
//! # Overview
//!
//! The image generation API provides a unified interface for generating images across
//! multiple providers:
//! - **OpenAI**: DALL-E 3 with support for various sizes and quality levels
//! - **Grok**: Grok Imagine API with fast image generation
//! - **Gemini**: Google's image generation with aspect ratio control
//!
//! # Basic Example
//!
//! ```rust,no_run
//! use cloudllm::image_generation::{ImageGenerationClient, ImageGenerationOptions};
//! use std::sync::Arc;
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     // Create a client (e.g., OpenAI)
//!     let client: Arc<dyn ImageGenerationClient> = /* ... */;
//!
//!     let options = ImageGenerationOptions {
//!         aspect_ratio: Some("16:9".to_string()),
//!         num_images: Some(1),
//!         response_format: Some("url".to_string()),
//!     };
//!
//!     let response = client.generate_image(
//!         "A futuristic city at sunset",
//!         options,
//!     ).await?;
//!
//!     for image in response.images {
//!         if let Some(url) = image.url {
//!             println!("Generated image: {}", url);
//!         }
//!     }
//!     Ok(())
//! }
//! ```
//!
//! # Handling Different Response Formats
//!
//! ```rust,no_run
//! use cloudllm::image_generation::{ImageGenerationClient, ImageGenerationOptions};
//! use std::sync::Arc;
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let client: Arc<dyn ImageGenerationClient> = /* ... */;
//!
//!     // Request Base64-encoded images instead of URLs
//!     let options = ImageGenerationOptions {
//!         aspect_ratio: None,
//!         num_images: Some(1),
//!         response_format: Some("b64_json".to_string()),
//!     };
//!
//!     let response = client.generate_image(
//!         "An oil painting of mountains",
//!         options,
//!     ).await?;
//!
//!     for image in response.images {
//!         if let Some(base64_data) = image.b64_json {
//!             println!("Base64 image data: {}...", &base64_data[..50]);
//!             // Can save to file or process directly
//!         }
//!     }
//!     Ok(())
//! }
//! ```
//!
//! # Landscape Mode with Aspect Ratios
//!
//! ```rust,no_run
//! use cloudllm::image_generation::{ImageGenerationClient, ImageGenerationOptions};
//! use std::sync::Arc;
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let client: Arc<dyn ImageGenerationClient> = /* ... */;
//!
//!     let options = ImageGenerationOptions {
//!         aspect_ratio: Some("16:9".to_string()), // Landscape
//!         num_images: Some(1),
//!         response_format: Some("url".to_string()),
//!     };
//!
//!     let response = client.generate_image(
//!         "Wide panoramic view of the Grand Canyon",
//!         options,
//!     ).await?;
//!
//!     println!("Generated {} images", response.images.len());
//!     Ok(())
//! }
//! ```
//!
//! # Multiple Images
//!
//! ```rust,no_run
//! use cloudllm::image_generation::{ImageGenerationClient, ImageGenerationOptions};
//! use std::sync::Arc;
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let client: Arc<dyn ImageGenerationClient> = /* ... */;
//!
//!     let options = ImageGenerationOptions {
//!         aspect_ratio: None,
//!         num_images: Some(4), // Generate 4 variations
//!         response_format: Some("url".to_string()),
//!     };
//!
//!     let response = client.generate_image(
//!         "A robot chef cooking pasta",
//!         options,
//!     ).await?;
//!
//!     for (idx, image) in response.images.iter().enumerate() {
//!         if let Some(url) = &image.url {
//!             println!("Variation {}: {}", idx + 1, url);
//!         }
//!     }
//!     Ok(())
//! }
//! ```
//!
//! # Accessing Revised Prompts (Grok-specific)
//!
//! ```rust,no_run
//! use cloudllm::image_generation::{ImageGenerationClient, ImageGenerationOptions};
//! use std::sync::Arc;
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let client: Arc<dyn ImageGenerationClient> = /* ... */;
//!
//!     let options = ImageGenerationOptions::default();
//!
//!     let response = client.generate_image(
//!         "A surreal landscape",
//!         options,
//!     ).await?;
//!
//!     // Some providers (like Grok) return the revised/completed prompt
//!     if let Some(revised) = &response.revised_prompt {
//!         println!("Revised prompt: {}", revised);
//!     }
//!     Ok(())
//! }
//! ```
//!
//! # Agent Integration with the Helper Function
//!
//! The `register_image_generation_tool()` helper dramatically simplifies adding image
//! generation to agents. Instead of 50+ lines of boilerplate, register a tool in one line:
//!
//! ```rust,no_run
//! use std::sync::Arc;
//! use cloudllm::Agent;
//! use cloudllm::clients::openai::{OpenAIClient, Model};
//! use cloudllm::image_generation::register_image_generation_tool;
//! use cloudllm::cloudllm::{ImageGenerationProvider, new_image_generation_client};
//! use cloudllm::tool_protocols::CustomToolProtocol;
//! use cloudllm::tool_protocol::ToolRegistry;
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let api_key = std::env::var("OPEN_AI_SECRET")?;
//!
//!     // Create image generation client
//!     let image_client = new_image_generation_client(
//!         ImageGenerationProvider::OpenAI,
//!         &api_key,
//!     )?;
//!
//!     // Create tool protocol
//!     let protocol = Arc::new(CustomToolProtocol::new());
//!
//!     // Register image generation tool in ONE LINE!
//!     let rt = tokio::runtime::Runtime::new()?;
//!     rt.block_on(register_image_generation_tool(&protocol, image_client))?;
//!
//!     // Create agent with image generation capability
//!     let registry = Arc::new(ToolRegistry::new(protocol));
//!     let agent = Agent::new(
//!         "designer",
//!         "Creative Designer",
//!         Arc::new(OpenAIClient::new_with_model_enum(&api_key, Model::GPT41Mini)),
//!     )
//!     .with_tools(registry);
//!
//!     println!("✓ Agent can now generate images!");
//!     Ok(())
//! }
//! ```
//!
//! The helper handles:
//! - Tool metadata and parameter definitions
//! - Async closure implementation
//! - Response parsing (URL and Base64 formats)
//! - Error handling and formatting
//! - Tool registration

use async_trait::async_trait;
use std::error::Error;

/// Configuration options for image generation.
///
/// All fields are optional and will use provider defaults if not specified.
///
/// # Examples
///
/// Minimal options with defaults:
/// ```
/// use cloudllm::image_generation::ImageGenerationOptions;
///
/// let options = ImageGenerationOptions {
///     aspect_ratio: None,
///     num_images: None,
///     response_format: None,
/// };
/// ```
///
/// Custom landscape format:
/// ```
/// use cloudllm::image_generation::ImageGenerationOptions;
///
/// let options = ImageGenerationOptions {
///     aspect_ratio: Some("16:9".to_string()),
///     num_images: Some(2),
///     response_format: Some("url".to_string()),
/// };
/// ```
#[derive(Clone, Debug, Default)]
pub struct ImageGenerationOptions {
    /// Aspect ratio for the generated image (e.g., "16:9", "4:3", "1:1").
    /// Supported ratios vary by provider:
    /// - OpenAI: Maps to standard sizes (1024x1024, 1024x1536, 1536x1024)
    /// - Grok: Fixed output, aspect ratio is ignored
    /// - Gemini: Supports 1:1, 2:3, 3:2, 3:4, 4:3, 4:5, 5:4, 9:16, 16:9, 21:9
    ///
    /// If `None`, the provider's default (typically square) is used.
    pub aspect_ratio: Option<String>,

    /// Number of images to generate (1-10).
    /// Most providers default to 1 if not specified.
    /// Note: Higher numbers may increase API costs.
    pub num_images: Option<u32>,

    /// Response format: "url" or "b64_json".
    /// - "url": Returns direct image URLs (default for most providers)
    /// - "b64_json": Returns Base64-encoded image data for direct embedding
    ///
    /// If `None`, the provider's default ("url" for most) is used.
    pub response_format: Option<String>,
}

/// A single generated image with optional URL or Base64 encoding.
///
/// Exactly one of `url` or `b64_json` will be populated depending on the
/// response format requested in [`ImageGenerationOptions`].
///
/// # Examples
///
/// Accessing a URL-based image:
/// ```
/// use cloudllm::image_generation::ImageData;
///
/// let image = ImageData {
///     url: Some("https://example.com/image.png".to_string()),
///     b64_json: None,
/// };
///
/// if let Some(url) = image.url {
///     println!("Download from: {}", url);
/// }
/// ```
///
/// Processing Base64 data:
/// ```
/// use cloudllm::image_generation::ImageData;
///
/// let image = ImageData {
///     url: None,
///     b64_json: Some("iVBORw0KGgoAAAANS...".to_string()),
/// };
///
/// if let Some(b64) = image.b64_json {
///     // Decode and save to file
///     let decoded = base64::decode(&b64).unwrap();
///     std::fs::write("image.png", decoded).unwrap();
/// }
/// ```
#[derive(Clone, Debug)]
pub struct ImageData {
    /// The URL to the generated image (if response_format="url").
    /// URLs are typically valid for 1 hour.
    pub url: Option<String>,

    /// The Base64-encoded image data (if response_format="b64_json").
    /// Includes the data URI prefix (e.g., "data:image/png;base64,...")
    pub b64_json: Option<String>,
}

/// Response from an image generation request.
///
/// Contains the list of generated images and optional metadata from the provider.
///
/// # Examples
///
/// Iterating over generated images:
/// ```
/// use cloudllm::image_generation::{ImageGenerationResponse, ImageData};
///
/// let response = ImageGenerationResponse {
///     images: vec![
///         ImageData { url: Some("url1".to_string()), b64_json: None },
///         ImageData { url: Some("url2".to_string()), b64_json: None },
///     ],
///     revised_prompt: None,
/// };
///
/// for (idx, image) in response.images.iter().enumerate() {
///     println!("Image {}: {:?}", idx + 1, image.url);
/// }
/// ```
///
/// Checking for revised prompt:
/// ```
/// use cloudllm::image_generation::{ImageGenerationResponse, ImageData};
///
/// let response = ImageGenerationResponse {
///     images: vec![ImageData { url: Some("url".to_string()), b64_json: None }],
///     revised_prompt: Some("A majestic eagle in flight over mountains".to_string()),
/// };
///
/// if let Some(revised) = response.revised_prompt {
///     println!("Provider modified prompt to: {}", revised);
/// }
/// ```
#[derive(Clone, Debug)]
pub struct ImageGenerationResponse {
    /// List of generated images.
    /// Length will match the `num_images` requested in [`ImageGenerationOptions`].
    pub images: Vec<ImageData>,

    /// The revised/completed prompt used for generation (if provided by provider).
    /// Only set by providers like Grok that enhance or expand the original prompt.
    /// Other providers may leave this as `None`.
    pub revised_prompt: Option<String>,
}

/// Determine image file extension from base64 data by inspecting the magic bytes.
///
/// This is useful when you have base64-encoded image data and need to determine
/// the file format to save it with the correct extension.
///
/// # Examples
///
/// ```
/// use cloudllm::image_generation::get_image_extension_from_base64;
///
/// // PNG magic bytes in base64
/// let b64_png = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==";
/// assert_eq!(get_image_extension_from_base64(b64_png), "png");
///
/// // JPEG magic bytes in base64
/// let b64_jpg = "/9j/4AAQSkZJRgABAQEAYABgAAD/2wBDA==";
/// assert_eq!(get_image_extension_from_base64(b64_jpg), "jpg");
///
/// // Unknown format
/// let b64_unknown = "aW52YWxpZCBpbWFnZSBkYXRh";
/// assert_eq!(get_image_extension_from_base64(b64_unknown), "bin");
/// ```
pub fn get_image_extension_from_base64(b64_data: &str) -> &str {
    if b64_data.starts_with("iVBORw0KG") {
        "png"
    } else if b64_data.starts_with("/9j/") {
        "jpg"
    } else if b64_data.starts_with("UklGRi") {
        "webp"
    } else {
        "bin" // fallback for unknown format
    }
}

/// Decode a base64 string to bytes.
///
/// This is useful when you receive base64-encoded image data and need to decode it
/// to save to a file or process it further.
///
/// # Examples
///
/// ```
/// use cloudllm::image_generation::decode_base64;
///
/// let b64 = "SGVsbG8gV29ybGQ="; // "Hello World" in base64
/// let decoded = decode_base64(b64).expect("Failed to decode");
/// assert_eq!(decoded, b"Hello World");
/// ```
pub fn decode_base64(input: &str) -> Result<Vec<u8>, String> {
    let bytes = input.as_bytes().iter().fold(Vec::new(), |mut acc, &b| {
        let val = match b {
            b'A'..=b'Z' => b - b'A',
            b'a'..=b'z' => b - b'a' + 26,
            b'0'..=b'9' => b - b'0' + 52,
            b'+' => 62,
            b'/' => 63,
            b'=' => return acc,
            _ => return acc,
        };
        acc.push(val);
        acc
    });

    let mut result = Vec::new();
    for chunk in bytes.chunks(4) {
        if chunk.len() >= 2 {
            result.push((chunk[0] << 2) | (chunk[1] >> 4));
        }
        if chunk.len() >= 3 {
            result.push((chunk[1] << 4) | (chunk[2] >> 2));
        }
        if chunk.len() >= 4 {
            result.push((chunk[2] << 6) | chunk[3]);
        }
    }

    Ok(result)
}

/// Trait for LLM providers that support image generation.
///
/// Implementors handle provider-specific details like API endpoints, authentication,
/// and response parsing. Users interact through this unified trait interface.
///
/// # Implementing the trait
///
/// Providers must implement two methods:
/// - `generate_image`: The core image generation logic
/// - `model_name`: Returns the name of the underlying model
///
/// # Error handling
///
/// Errors should be converted to `Box<dyn Error + Send + Sync>` for consistency.
/// Common errors include:
/// - Invalid API key
/// - Rate limiting
/// - Invalid prompt content
/// - Unsupported aspect ratios or formats
#[async_trait]
pub trait ImageGenerationClient: Send + Sync {
    /// Generate images from a text prompt.
    ///
    /// This is the primary method for creating images. The provider handles
    /// authentication, endpoint routing, and response parsing internally.
    ///
    /// # Arguments
    ///
    /// * `prompt` - The text description of the image to generate.
    ///   Longer, more detailed prompts typically result in better images.
    ///   Consider including style, mood, lighting, and composition details.
    ///
    /// * `options` - Configuration options for image generation.
    ///   Providers will use their defaults for any `None` fields.
    ///
    /// # Returns
    ///
    /// A [`ImageGenerationResponse`] containing the generated images on success,
    /// or an error describing what went wrong.
    ///
    /// # Errors
    ///
    /// Returns errors for:
    /// - Network failures
    /// - Authentication issues
    /// - Invalid or unsafe prompts
    /// - Rate limiting
    /// - Unsupported options for the provider
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use cloudllm::image_generation::{ImageGenerationClient, ImageGenerationOptions};
    /// use std::sync::Arc;
    ///
    /// #[tokio::main]
    /// async fn main() -> Result<(), Box<dyn std::error::Error>> {
    ///     let client: Arc<dyn ImageGenerationClient> = /* ... */;
    ///
    ///     let prompt = "A serene Japanese garden with a koi pond, \
    ///                   morning light filtering through cherry blossoms";
    ///     let options = ImageGenerationOptions {
    ///         aspect_ratio: Some("4:3".to_string()),
    ///         num_images: Some(1),
    ///         response_format: Some("url".to_string()),
    ///     };
    ///
    ///     match client.generate_image(prompt, options).await {
    ///         Ok(response) => {
    ///             for image in response.images {
    ///                 if let Some(url) = image.url {
    ///                     println!("Success: {}", url);
    ///                 }
    ///             }
    ///         }
    ///         Err(e) => eprintln!("Failed to generate image: {}", e),
    ///     }
    ///     Ok(())
    /// }
    /// ```
    async fn generate_image(
        &self,
        prompt: &str,
        options: ImageGenerationOptions,
    ) -> Result<ImageGenerationResponse, Box<dyn Error + Send + Sync>>;

    /// Get the name of the model being used for image generation.
    ///
    /// Returns the provider's model identifier (e.g., "dall-e-3", "grok-2-image",
    /// "gemini-2.5-flash-image").
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use cloudllm::image_generation::ImageGenerationClient;
    /// use std::sync::Arc;
    ///
    /// #[tokio::main]
    /// async fn main() -> Result<(), Box<dyn std::error::Error>> {
    ///     let client: Arc<dyn ImageGenerationClient> = /* ... */;
    ///     println!("Using model: {}", client.model_name());
    ///     Ok(())
    /// }
    /// ```
    fn model_name(&self) -> &str;
}

/// Simplified helper to register an image generation tool with a tool protocol.
///
/// This helper eliminates boilerplate when adding image generation to agents.
/// It creates the tool metadata and async closure handler automatically.
///
/// # Arguments
///
/// * `protocol` - The CustomToolProtocol to register the tool with
/// * `image_client` - The ImageGenerationClient to use for image generation
///
/// # Example
///
/// ```rust,no_run
/// use cloudllm::image_generation::register_image_generation_tool;
/// use cloudllm::cloudllm::{ImageGenerationProvider, new_image_generation_client};
/// use cloudllm::tool_protocols::CustomToolProtocol;
/// use cloudllm::tool_protocol::ToolRegistry;
/// use cloudllm::Agent;
/// use cloudllm::clients::openai::{OpenAIClient, Model};
/// use std::sync::Arc;
///
/// #[tokio::main]
/// async fn main() -> Result<(), Box<dyn std::error::Error>> {
///     let api_key = std::env::var("OPEN_AI_SECRET")?;
///
///     // Create image generation client
///     let image_client = new_image_generation_client(
///         ImageGenerationProvider::OpenAI,
///         &api_key,
///     )?;
///
///     // Create protocol and register image tool
///     let protocol = Arc::new(CustomToolProtocol::new());
///     let rt = tokio::runtime::Runtime::new()?;
///     rt.block_on(register_image_generation_tool(&protocol, image_client))?;
///
///     // Create agent with image generation capability
///     let registry = Arc::new(ToolRegistry::new(protocol));
///     let agent = Agent::new(
///         "designer",
///         "Creative Designer",
///         Arc::new(OpenAIClient::new_with_model_enum(&api_key, Model::GPT41Mini)),
///     )
///     .with_tools(registry);
///
///     // Agent can now generate images!
///     println!("✓ Agent ready with image generation tool");
///     Ok(())
/// }
/// ```
///
/// # Tool Parameters
///
/// The registered tool accepts:
/// - `prompt` (string, required): The image description to generate
/// - `aspect_ratio` (string, optional): Aspect ratio like "16:9", "4:3", "1:1"
///
/// # Returns
///
/// The tool returns a JSON object with:
/// - `url`: Image URL (if response_format was "url")
/// - `b64_json`: Base64 image data (if response_format was "b64_json")
/// - `model`: The image generation model used
/// - `format`: Response format ("url" or "base64")
/// - `success`: Boolean success indicator
pub async fn register_image_generation_tool(
    protocol: &std::sync::Arc<crate::tool_protocols::CustomToolProtocol>,
    image_client: std::sync::Arc<dyn ImageGenerationClient>,
) -> Result<(), Box<dyn std::error::Error>> {
    use crate::tool_protocol::{ToolMetadata, ToolParameter, ToolParameterType, ToolResult};
    use serde_json::json;

    protocol
        .register_async_tool(
            ToolMetadata::new("generate_image", "Generate an image from a text prompt")
                .with_parameter(
                    ToolParameter::new("prompt", ToolParameterType::String)
                        .with_description("The image prompt to generate")
                        .required(),
                )
                .with_parameter(
                    ToolParameter::new("aspect_ratio", ToolParameterType::String)
                        .with_description("Aspect ratio like 16:9, 4:3, 1:1"),
                ),
            std::sync::Arc::new(move |params| {
                let client = image_client.clone();
                Box::pin(async move {
                    let prompt = params["prompt"]
                        .as_str()
                        .ok_or("prompt parameter required")?;

                    match client
                        .generate_image(
                            prompt,
                            ImageGenerationOptions {
                                aspect_ratio: params
                                    .get("aspect_ratio")
                                    .and_then(|v| v.as_str())
                                    .map(|s| s.to_string()),
                                num_images: Some(1),
                                response_format: Some("url".to_string()),
                            },
                        )
                        .await
                    {
                        Ok(response) => {
                            if let Some(image) = response.images.first() {
                                if let Some(url) = &image.url {
                                    Ok(ToolResult::success(json!({
                                        "url": url,
                                        "model": client.model_name(),
                                        "format": "url",
                                        "success": true
                                    })))
                                } else if let Some(b64) = &image.b64_json {
                                    Ok(ToolResult::success(json!({
                                        "b64_json": b64,
                                        "model": client.model_name(),
                                        "format": "base64",
                                        "success": true
                                    })))
                                } else {
                                    Ok(ToolResult::failure(
                                        "No URL or base64 data in image response".to_string(),
                                    ))
                                }
                            } else {
                                Ok(ToolResult::failure("No images were generated".to_string()))
                            }
                        }
                        Err(e) => Ok(ToolResult::failure(e.to_string())),
                    }
                })
            }),
        )
        .await;

    Ok(())
}
