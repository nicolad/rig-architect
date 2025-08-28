//! DeepSeek API response types
//!
//! This module contains all the type definitions for DeepSeek API requests and responses.
//! These types are generated based on REAL API responses from DeepSeek.
//! Generated on: Thu Aug 28 14:50:23 EEST 2025
//!
//! Analysis results:
//! - Detected fields: 20 fields including: id, logprobs, total_tokens, prompt_tokens, prompt_cache_miss_tokens, cached_tokens, model, role, object, content, ...
//! - Context caching support: detected
//! - Error response structure: not detected

use serde::{Deserialize, Serialize};

/// Chat completion request payload
#[derive(Debug, Clone, Serialize)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<Message>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    pub stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<Tool>>,
}

/// A single message in the conversation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: String,
    pub content: String,
}

/// Tool definition for function calling
#[derive(Debug, Clone, Serialize)]
pub struct Tool {
    #[serde(rename = "type")]
    pub tool_type: String,
    pub function: Function,
}

/// Function definition for strict mode
#[derive(Debug, Clone, Serialize)]
pub struct Function {
    pub name: String,
    pub description: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub strict: Option<bool>,
    pub parameters: serde_json::Value,
}

/// Chat completion response from DeepSeek API
#[derive(Debug, Clone, Deserialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system_fingerprint: Option<String>,
    pub choices: Vec<Choice>,
    pub usage: Usage,
}

/// A single choice in the completion response
#[derive(Debug, Clone, Deserialize)]
pub struct Choice {
    pub index: u32,
    pub message: Message,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<String>,
}

/// Token usage information with context caching support
#[derive(Debug, Clone, Deserialize)]
pub struct Usage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
    /// Number of tokens that resulted in a cache hit (0.1 yuan per million tokens)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_cache_hit_tokens: Option<u32>,
    /// Number of tokens that did not result in a cache hit (1 yuan per million tokens)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_cache_miss_tokens: Option<u32>,
}

/// Error response from DeepSeek API
#[derive(Debug, Clone, Deserialize)]
pub struct ErrorResponse {
    pub error: ErrorDetail,
}

/// Detailed error information
#[derive(Debug, Clone, Deserialize)]
pub struct ErrorDetail {
    pub message: String,
    #[serde(rename = "type")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error_type: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub param: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub code: Option<String>,
}

/// Model constants for DeepSeek API
pub mod models {
    /// Standard chat model
    pub const DEEPSEEK_CHAT: &str = "deepseek-chat";
    /// Reasoning model with enhanced capabilities
    pub const DEEPSEEK_REASONER: &str = "deepseek-reasoner";
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_message_serialization() {
        let message = Message {
            role: "user".to_string(),
            content: "Hello".to_string(),
        };

        let json = serde_json::to_string(&message).unwrap();
        assert!(json.contains("user"));
        assert!(json.contains("Hello"));
    }

    #[test]
    fn test_chat_completion_request_serialization() {
        let request = ChatCompletionRequest {
            model: "deepseek-chat".to_string(),
            messages: vec![Message {
                role: "user".to_string(),
                content: "Hello".to_string(),
            }],
            max_tokens: Some(100),
            temperature: Some(0.7),
            top_p: None,
            stream: false,
            tools: None,
        };

        let json = serde_json::to_string(&request).unwrap();
        assert!(json.contains("deepseek-chat"));
        assert!(json.contains("Hello"));
        assert!(json.contains("100"));
        assert!(json.contains("0.7"));
        assert!(!json.contains("top_p")); // Should be omitted when None
    }

    #[test]
    fn test_usage_deserialization_with_cache() {
        let json = r#"{
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "total_tokens": 30,
            "prompt_cache_hit_tokens": 5,
            "prompt_cache_miss_tokens": 5
        }"#;

        let usage: Usage = serde_json::from_str(json).unwrap();
        assert_eq!(usage.prompt_tokens, 10);
        assert_eq!(usage.completion_tokens, 20);
        assert_eq!(usage.total_tokens, 30);
        assert_eq!(usage.prompt_cache_hit_tokens, Some(5));
        assert_eq!(usage.prompt_cache_miss_tokens, Some(5));
    }

    #[test]
    fn test_usage_deserialization_without_cache() {
        let json = r#"{
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "total_tokens": 30
        }"#;

        let usage: Usage = serde_json::from_str(json).unwrap();
        assert_eq!(usage.prompt_tokens, 10);
        assert_eq!(usage.completion_tokens, 20);
        assert_eq!(usage.total_tokens, 30);
        assert_eq!(usage.prompt_cache_hit_tokens, None);
        assert_eq!(usage.prompt_cache_miss_tokens, None);
    }
}
