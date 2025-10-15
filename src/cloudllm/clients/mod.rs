//! Provider specific [`ClientWrapper`](crate::client_wrapper::ClientWrapper) implementations.
//!
//! Each submodule offers a concrete client that speaks a particular vendor's API while
//! conforming to the uniform CloudLLM contract.

pub mod common;

pub mod claude;
pub mod gemini;
pub mod grok;
pub mod openai;
