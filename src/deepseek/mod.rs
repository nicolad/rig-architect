//! DeepSeek API client module
//!
//! This module provides a comprehensive client for the DeepSeek API with support for:
//! - Chat completions with context caching
//! - Function calling with strict mode
//! - Agent-like conversation management
//! - Proper error handling and logging

pub mod client;
pub mod types;

pub use client::{DeepSeekClient, Agent};
pub use types::*;
