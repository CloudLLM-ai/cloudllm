//! # CloudLLM Library
//!
//! `cloudllm` provides an abstracted interface for seamless interactions with Language Learning Models (LLMs) like OpenAI's ChatGPT.
//! At its core, it offers tools for real-time, back-and-forth conversations, encapsulating these interactions within sessions 
//! that maintain dialogue history for contextualized and coherent exchanges.
//!
//! ## Current Functionality
//!
//! - **Session Management**: With the `LLMSession`, you can engage in dynamic conversations, sending user or system messages 
//!   to the LLM and receiving contextual responses. The session also maintains a history of interactions, ensuring that 
//!   subsequent messages can build upon previous ones for a richer conversational experience.
//!
//! - **Client Wrappers**: The library supports a modular approach where different LLM providers can be integrated 
//!   via client wrappers. For example, `OpenAIClient` serves as a client for OpenAI's ChatGPT, abstracting the interaction 
//!   specifics and presenting a unified interface.
//!
//! ## The Road Ahead: LLM-VM Architecture
//!
//! The library is poised to evolve into a more sophisticated toolset with the introduction of the "LLM-VM" architecture.
//! This design envisions empowering the remote LLMs with local computing capabilities, effectively turning the client 
//! into a virtual machine (VM) for the LLM.
//!
//! Here's a sneak peek into what's coming:
//!
//! - **Instruction Set for LLMs**: The LLM-VM architecture will expose a set of instructions that the remote LLM can invoke 
//!   on the local client. This could range from simple I/O commands to more complex operations.
//!
//! - **State Management**: Local clients will be capable of maintaining state on behalf of the LLM. Think of key/value databases 
//!   or in-memory hashmaps, where the LLM can store and retrieve data without relying solely on its internal memory.
//!
//! - **Offloading Computations**: Certain operations, like arithmetical calculations, might be better executed locally than by 
//!   the LLM. The LLM-VM setup will allow the LLM to offload such tasks to the client, ensuring faster and more accurate results.
//!
//! With these advancements, `cloudllm` will not only be a bridge for dialogues but also a robust platform that enhances 
//! the capabilities of LLMs, opening doors to more interactive and intelligent applications.
//!

// src/lib.rs

// Import the top-level `cloudllm` module.
pub mod cloudllm;

// If you want to provide direct access (without having to navigate through the whole hierarchy) to certain types or functionalities at the crate level, you can use re-exports:

// Re-exporting key items for easier external access.
pub use cloudllm::client_wrapper;
pub use cloudllm::client_wrapper::{ClientWrapper, Message, Role};
pub use cloudllm::llm_session::LLMSession;
// If you wish, you can also re-export specific clients or functionalities from the `clients` submodule:
// pub use cloudllm::clients::openai;
pub use cloudllm::clients;
