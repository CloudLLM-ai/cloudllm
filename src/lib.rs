// src/lib.rs

// Import the top-level `cloudllm` module.
pub mod cloudllm;

// If you want to provide direct access (without having to navigate through the whole hierarchy) to certain types or functionalities at the crate level, you can use re-exports:

// Re-exporting key items for easier external access.
pub use cloudllm::client_wrapper::{ClientWrapper, Message, Role};
pub use cloudllm::llm_session::LLMSession;
// If you wish, you can also re-export specific clients or functionalities from the `clients` submodule:
// pub use cloudllm::clients::openai;
