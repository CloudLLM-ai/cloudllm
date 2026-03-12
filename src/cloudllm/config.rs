//! Configuration for CloudLLM.
//!
//! Provides the [`CloudLLMConfig`] struct for configuring
//! [`MentisDb`](crate::MentisDb) storage configuration and other global settings.
//! Users construct this manually — no file parsing dependencies are required.
//!
//! # Example
//!
//! ```rust
//! use cloudllm::CloudLLMConfig;
//! use std::path::PathBuf;
//!
//! // Use the default ("mentisdbs" in the current directory)
//! let config = CloudLLMConfig::default();
//!
//! // Or specify a custom directory
//! let config = CloudLLMConfig {
//!     mentisdb_dir: PathBuf::from("/var/data/agent_chains"),
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
///     mentisdb_dir: PathBuf::from("/tmp/my_chains"),
/// };
/// ```
pub struct CloudLLMConfig {
    /// Directory where the default MentisDb JSONL storage adapter stores
    /// chain files.
    ///
    /// Passed to [`MentisDb::open`](crate::MentisDb::open) as the
    /// `chain_dir` argument.
    pub mentisdb_dir: PathBuf,
}

impl Default for CloudLLMConfig {
    /// Create a config pointing at `"mentisdbs"` in the current working
    /// directory.
    ///
    /// # Example
    ///
    /// ```rust
    /// use cloudllm::CloudLLMConfig;
    /// use std::path::PathBuf;
    ///
    /// let config = CloudLLMConfig::default();
    /// assert_eq!(config.mentisdb_dir, PathBuf::from("mentisdbs"));
    /// ```
    fn default() -> Self {
        Self {
            mentisdb_dir: PathBuf::from("mentisdbs"),
        }
    }
}
