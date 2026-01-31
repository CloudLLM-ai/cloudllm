//! Internal module tree housing the building blocks exposed via `cloudllm`.
//!
//! This module organizes CloudLLM's core functionality:
//!
//! - **agent**: Core Agent struct for LLM-powered entities
//! - **client_wrapper**: Trait definition for LLM provider implementations
//! - **clients**: Concrete implementations for OpenAI, Claude, Gemini, Grok, and custom endpoints
//! - **image_generation**: Image generation trait for creating images from prompts
//! - **llm_session**: Stateful conversation management with context trimming
//! - **tool_protocol**: Protocol-agnostic tool interface and ToolRegistry for multi-protocol support
//! - **tool_protocols**: Concrete ToolProtocol implementations (Custom, MCP, Memory, OpenAI)
//! - **resource_protocol**: MCP Resource support for application-provided context
//! - **tools**: Built-in tools (Memory, Bash, HTTP Client, etc.)
//! - **council**: Multi-agent orchestration system with 5 collaboration modes
//! - **mcp_server**: Unified MCP server for tool aggregation and routing

pub mod agent;
pub mod client_wrapper;
pub mod clients;
pub mod council;
pub mod image_generation;
pub mod llm_session;
pub mod mcp_http_adapter;
pub mod mcp_server;
pub mod mcp_server_builder;
pub mod mcp_server_builder_utils;
pub mod resource_protocol;
pub mod tool_protocol;
pub mod tool_protocols;
pub mod tools;

// Core exports for easy access
pub use agent::Agent;
pub use image_generation::{
    decode_base64, get_image_extension_from_base64, ImageData, ImageGenerationClient,
    ImageGenerationOptions, ImageGenerationResponse,
};
pub use llm_session::LLMSession;

use std::sync::Arc;

/// Image generation provider enum with type-safe provider selection.
///
/// This enum provides compile-time type safety when selecting image generation providers,
/// eliminating the possibility of typos that would occur with string-based selection.
///
/// # Examples
///
/// ```rust,no_run
/// use cloudllm::cloudllm::{ImageGenerationProvider, new_image_generation_client};
///
/// let client = new_image_generation_client(
///     ImageGenerationProvider::OpenAI,
///     "your-api-key",
/// ).expect("Failed to create client");
///
/// println!("Using: {}", client.model_name());
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImageGenerationProvider {
    /// OpenAI's DALL-E 3 for high-quality image generation with aspect ratio support
    OpenAI,

    /// xAI's Grok Imagine API for fast image generation
    Grok,

    /// Google's Gemini image generation with advanced aspect ratio control
    Gemini,
}

use std::error::Error;
use std::fmt;
use std::str::FromStr;

#[derive(Debug, Clone)]
pub struct ImageGenerationProviderError(String);

impl fmt::Display for ImageGenerationProviderError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl Error for ImageGenerationProviderError {}

impl ImageGenerationProvider {
    /// Convert the provider enum to its string representation.
    ///
    /// # Examples
    ///
    /// ```
    /// use cloudllm::cloudllm::ImageGenerationProvider;
    ///
    /// assert_eq!(ImageGenerationProvider::OpenAI.as_str(), "openai");
    /// assert_eq!(ImageGenerationProvider::Grok.as_str(), "grok");
    /// assert_eq!(ImageGenerationProvider::Gemini.as_str(), "gemini");
    /// ```
    pub fn as_str(&self) -> &'static str {
        match self {
            ImageGenerationProvider::OpenAI => "openai",
            ImageGenerationProvider::Grok => "grok",
            ImageGenerationProvider::Gemini => "gemini",
        }
    }

    /// Get a human-readable name for the provider.
    ///
    /// # Examples
    ///
    /// ```
    /// use cloudllm::cloudllm::ImageGenerationProvider;
    ///
    /// assert_eq!(ImageGenerationProvider::OpenAI.display_name(), "OpenAI (DALL-E 3)");
    /// ```
    pub fn display_name(&self) -> &'static str {
        match self {
            ImageGenerationProvider::OpenAI => "OpenAI (DALL-E 3)",
            ImageGenerationProvider::Grok => "Grok Imagine",
            ImageGenerationProvider::Gemini => "Google Gemini",
        }
    }
}

/// Standard library [`FromStr`] trait implementation for string-to-enum conversion.
///
/// Allows using the `from_str()` method from the `FromStr` trait to parse
/// provider names. This is the recommended way to convert strings to the enum.
///
/// # Supported Strings
///
/// - `"openai"` (case-insensitive) → [`ImageGenerationProvider::OpenAI`]
/// - `"grok"` (case-insensitive) → [`ImageGenerationProvider::Grok`]
/// - `"gemini"` (case-insensitive) → [`ImageGenerationProvider::Gemini`]
///
/// # Examples
///
/// ```
/// use cloudllm::cloudllm::ImageGenerationProvider;
/// use std::str::FromStr;
///
/// // Parse from string using FromStr trait
/// let provider = ImageGenerationProvider::from_str("openai").unwrap();
/// assert_eq!(provider, ImageGenerationProvider::OpenAI);
///
/// // Case-insensitive
/// let provider = ImageGenerationProvider::from_str("GROK").unwrap();
/// assert_eq!(provider, ImageGenerationProvider::Grok);
///
/// // Invalid provider returns error
/// assert!(ImageGenerationProvider::from_str("invalid").is_err());
/// ```
impl FromStr for ImageGenerationProvider {
    type Err = ImageGenerationProviderError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "openai" => Ok(ImageGenerationProvider::OpenAI),
            "grok" => Ok(ImageGenerationProvider::Grok),
            "gemini" => Ok(ImageGenerationProvider::Gemini),
            _ => Err(ImageGenerationProviderError(format!(
                "Unknown image generation provider '{}'. Supported providers: openai, grok, gemini",
                s
            ))),
        }
    }
}

/// Factory function to create image generation clients using enum-based provider selection.
///
/// This is the type-safe way to create image generation clients. It uses the
/// [`ImageGenerationProvider`] enum to ensure compile-time correctness and avoid
/// typos that would occur with string-based selection.
///
/// # Arguments
///
/// * `provider` - The image generation provider enum
/// * `api_key` - The API key for the provider
///
/// # Returns
///
/// An `Arc` to a boxed `ImageGenerationClient` trait object that can be used
/// to generate images.
///
/// # Examples
///
/// Creating an OpenAI image generation client:
/// ```rust,no_run
/// use cloudllm::cloudllm::{ImageGenerationProvider, new_image_generation_client};
///
/// let client = new_image_generation_client(
///     ImageGenerationProvider::OpenAI,
///     "your-openai-api-key",
/// ).expect("Failed to create OpenAI client");
///
/// println!("Using model: {}", client.model_name());
/// ```
///
/// Creating a Grok image generation client:
/// ```rust,no_run
/// use cloudllm::cloudllm::{ImageGenerationProvider, new_image_generation_client};
///
/// let client = new_image_generation_client(
///     ImageGenerationProvider::Grok,
///     "your-xai-key",
/// ).expect("Failed to create Grok client");
///
/// println!("Using model: {}", client.model_name());
/// ```
///
/// Creating a Gemini image generation client:
/// ```rust,no_run
/// use cloudllm::cloudllm::{ImageGenerationProvider, new_image_generation_client};
///
/// let client = new_image_generation_client(
///     ImageGenerationProvider::Gemini,
///     "your-gemini-key",
/// ).expect("Failed to create Gemini client");
///
/// println!("Using model: {}", client.model_name());
/// ```
pub fn new_image_generation_client(
    provider: ImageGenerationProvider,
    api_key: &str,
) -> Result<Arc<dyn ImageGenerationClient>, String> {
    match provider {
        ImageGenerationProvider::OpenAI => {
            let client = clients::openai::OpenAIClient::new_with_model_string(
                api_key,
                "gpt-4o-mini", // This is just for the text model; image uses DALL-E 3
            );
            Ok(Arc::new(client))
        }
        ImageGenerationProvider::Grok => {
            let client = clients::grok::GrokClient::new_with_model_str(api_key, "grok-3-mini");
            Ok(Arc::new(client))
        }
        ImageGenerationProvider::Gemini => {
            let client =
                clients::gemini::GeminiClient::new_with_model_string(api_key, "gemini-2.5-flash");
            Ok(Arc::new(client))
        }
    }
}

/// Factory function to create image generation clients from a string provider name.
///
/// This function is a fallback for when the provider is determined at runtime.
/// For compile-time type safety, prefer using [`new_image_generation_client`] with
/// the [`ImageGenerationProvider`] enum.
///
/// # Arguments
///
/// * `provider` - The provider name as a string: "openai", "grok", or "gemini"
/// * `api_key` - The API key for the provider
///
/// # Returns
///
/// An `Arc` to a boxed `ImageGenerationClient` trait object, or an error if the
/// provider name is not recognized.
///
/// # Examples
///
/// ```rust,no_run
/// use cloudllm::cloudllm::new_image_generation_client_from_str;
///
/// let client = new_image_generation_client_from_str(
///     "openai",
///     "your-api-key",
/// ).expect("Failed to create client");
///
/// println!("Using: {}", client.model_name());
/// ```
pub fn new_image_generation_client_from_str(
    provider: &str,
    api_key: &str,
) -> Result<Arc<dyn ImageGenerationClient>, String> {
    let parsed_provider = ImageGenerationProvider::from_str(provider).map_err(|e| e.to_string())?;
    new_image_generation_client(parsed_provider, api_key)
}
