//! AI Architecture Improver Library
//!
//! This library provides the core components for the AI Architecture Improver,
//! including the DeepSeek API client and configuration management.

pub mod config;
pub mod deepseek;

// Re-export commonly used items
pub use config::Config;
pub use deepseek::DeepSeekClient;
