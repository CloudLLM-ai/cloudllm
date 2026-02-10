//! Configuration for CloudLLM.
//!
//! Provides the [`CloudLLMConfig`] struct for configuring
//! [`ThoughtChain`](crate::ThoughtChain) storage and other global settings.
//! Users construct this manually — no file parsing dependencies are required.
//!
//! # Example
//!
//! ```rust
//! use cloudllm::CloudLLMConfig;
//! use std::path::PathBuf;
//!
//! // Use the default ("thought_chains" in the current directory)
//! let config = CloudLLMConfig::default();
//!
//! // Or specify a custom directory
//! let config = CloudLLMConfig {
//!     thought_chain_dir: PathBuf::from("/var/data/agent_chains"),
//! };
//! ```

use std::path::PathBuf;

/// Global configuration for CloudLLM features.
///
/// This struct is intentionally minimal and users construct it however they want.
/// No TOML, YAML, or other config-file parsing dependencies are introduced.
///
/// # Example
///
/// ```rust
/// use cloudllm::CloudLLMConfig;
/// use std::path::PathBuf;
///
/// let config = CloudLLMConfig {
///     thought_chain_dir: PathBuf::from("/tmp/my_chains"),
/// };
/// ```
pub struct CloudLLMConfig {
    /// Directory where [`ThoughtChain`](crate::ThoughtChain) `.jsonl` files
    /// are stored.  Passed to [`ThoughtChain::open`](crate::ThoughtChain::open)
    /// as the `chain_dir` argument.
    pub thought_chain_dir: PathBuf,
}

impl Default for CloudLLMConfig {
    /// Create a config pointing at `"thought_chains"` in the current working
    /// directory.
    ///
    /// # Example
    ///
    /// ```rust
    /// use cloudllm::CloudLLMConfig;
    /// use std::path::PathBuf;
    ///
    /// let config = CloudLLMConfig::default();
    /// assert_eq!(config.thought_chain_dir, PathBuf::from("thought_chains"));
    /// ```
    fn default() -> Self {
        Self {
            thought_chain_dir: PathBuf::from("thought_chains"),
        }
    }
}
