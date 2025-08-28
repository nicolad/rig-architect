#!/usr/bin/env cargo +nightly -Zscript
//! Type generator for DeepSeek API responses
//! 
//! This script generates Rust types based on example DeepSeek API responses.
//! It creates properly typed structs that can be used in the client.

fn main() {
    println!("🔧 Generating DeepSeek API types...");
    
    // Generate the types.rs file content
    let types_content = generate_types_file();
    
    println!("Generated types.rs content:");
    println!("{}", types_content);
    
    // Write to file
    std::fs::write("src/deepseek/types.rs", types_content)
        .expect("Failed to write types.rs");
    
    println!("✅ Types generated successfully in src/deepseek/types.rs");
}

fn generate_types_file() -> String {
    r#"//! DeepSeek API response types
//!
//! This module contains all the type definitions for DeepSeek API requests and responses.
//! These types are generated based on the actual API specification and responses.

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
            messages: vec![
                Message {
                    role: "user".to_string(),
                    content: "Hello".to_string(),
                }
            ],
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
"#.to_string()
}

use serde::{Deserialize, Serialize};
use serde_json::{Value, Map};
use std::collections::HashSet;
use std::env;
use std::fs;

#[derive(Debug)]
struct TypeGenerator {
    seen_types: HashSet<String>,
    generated_types: Vec<String>,
}

impl TypeGenerator {
    fn new() -> Self {
        Self {
            seen_types: HashSet::new(),
            generated_types: Vec::new(),
        }
    }

    fn generate_type_from_json(&mut self, json: &Value, type_name: &str) -> String {
        match json {
            Value::Object(map) => self.generate_struct(map, type_name),
            Value::Array(arr) if !arr.is_empty() => {
                // For arrays, generate type based on first element
                let element_type = self.infer_type(&arr[0], &format!("{}Item", type_name));
                format!("Vec<{}>", element_type)
            }
            _ => self.infer_primitive_type(json),
        }
    }

    fn generate_struct(&mut self, map: &Map<String, Value>, struct_name: &str) -> String {
        if self.seen_types.contains(struct_name) {
            return struct_name.to_string();
        }

        self.seen_types.insert(struct_name.to_string());

        let mut fields = Vec::new();
        let mut field_types = Vec::new();

        for (key, value) in map {
            let field_name = self.sanitize_field_name(key);
            let field_type = self.infer_type(value, &self.capitalize(&field_name));
            
            // Handle optional fields
            let is_optional = self.should_be_optional(&field_name);
            let final_type = if is_optional {
                format!("Option<{}>", field_type)
            } else {
                field_type
            };

            let serde_attr = if field_name != *key {
                format!("    #[serde(rename = \"{}\")]\n", key)
            } else {
                String::new()
            };

            fields.push(format!("{}    pub {}: {},", serde_attr, field_name, final_type));
            field_types.push((field_name, final_type));
        }

        let struct_def = format!(
            "#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]\npub struct {} {{\n{}\n}}",
            struct_name,
            fields.join("\n")
        );

        self.generated_types.push(struct_def.clone());
        struct_name.to_string()
    }

    fn infer_type(&mut self, value: &Value, suggested_name: &str) -> String {
        match value {
            Value::Null => "Option<Value>".to_string(),
            Value::Bool(_) => "bool".to_string(),
            Value::Number(n) => {
                if n.is_f64() {
                    "f64".to_string()
                } else if n.is_i64() {
                    "i64".to_string()
                } else {
                    "u64".to_string()
                }
            }
            Value::String(_) => "String".to_string(),
            Value::Array(arr) => {
                if arr.is_empty() {
                    "Vec<Value>".to_string()
                } else {
                    let element_type = self.infer_type(&arr[0], &format!("{}Item", suggested_name));
                    format!("Vec<{}>", element_type)
                }
            }
            Value::Object(map) => {
                self.generate_struct(map, suggested_name)
            }
        }
    }

    fn infer_primitive_type(&self, value: &Value) -> String {
        match value {
            Value::Null => "Option<Value>".to_string(),
            Value::Bool(_) => "bool".to_string(),
            Value::Number(n) => {
                if n.is_f64() {
                    "f64".to_string()
                } else if n.is_i64() {
                    "i64".to_string()
                } else {
                    "u64".to_string()
                }
            }
            Value::String(_) => "String".to_string(),
            Value::Array(_) => "Vec<Value>".to_string(),
            Value::Object(_) => "Value".to_string(),
        }
    }

    fn sanitize_field_name(&self, name: &str) -> String {
        let mut result = String::new();
        let mut chars = name.chars().peekable();
        
        // Handle leading numbers
        if chars.peek().map_or(false, |c| c.is_ascii_digit()) {
            result.push('_');
        }

        for ch in chars {
            match ch {
                'a'..='z' | 'A'..='Z' | '0'..='9' => result.push(ch.to_ascii_lowercase()),
                '_' | '-' => result.push('_'),
                _ => result.push('_'),
            }
        }

        // Handle Rust keywords
        match result.as_str() {
            "type" => "type_".to_string(),
            "match" => "match_".to_string(),
            "use" => "use_".to_string(),
            "mod" => "mod_".to_string(),
            "fn" => "fn_".to_string(),
            "struct" => "struct_".to_string(),
            "enum" => "enum_".to_string(),
            "impl" => "impl_".to_string(),
            "trait" => "trait_".to_string(),
            "const" => "const_".to_string(),
            "static" => "static_".to_string(),
            "async" => "async_".to_string(),
            "await" => "await_".to_string(),
            _ => result,
        }
    }

    fn capitalize(&self, s: &str) -> String {
        let mut chars = s.chars();
        match chars.next() {
            None => String::new(),
            Some(first) => first.to_uppercase().collect::<String>() + chars.as_str(),
        }
    }

    fn should_be_optional(&self, field_name: &str) -> bool {
        // Fields that are commonly optional in API responses
        matches!(field_name, 
            "finish_reason" | "logprobs" | "system_fingerprint" | 
            "prompt_cache_hit_tokens" | "prompt_cache_miss_tokens" |
            "tools" | "tool_calls" | "function_call"
        )
    }

    fn generate_output(&self) -> String {
        let mut output = String::new();
        
        output.push_str("//! Generated types for DeepSeek API responses\n");
        output.push_str("//! \n");
        output.push_str("//! This file is auto-generated by scripts/generate_types.rs\n");
        output.push_str("//! Do not edit manually - run the generator script instead.\n\n");
        output.push_str("use serde::{Deserialize, Serialize};\n");
        output.push_str("use serde_json::Value;\n\n");

        for type_def in &self.generated_types {
            output.push_str(&type_def);
            output.push_str("\n\n");
        }

        output
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    dotenvy::dotenv().ok();

    let api_key = env::var("DEEPSEEK_API_KEY")
        .expect("DEEPSEEK_API_KEY environment variable must be set");

    println!("🔍 Generating types from DeepSeek API responses...");

    let client = reqwest::Client::new();
    let mut generator = TypeGenerator::new();

    // Sample requests to generate comprehensive types
    let test_cases = vec![
        ("ChatCompletionRequest", serde_json::json!({
            "model": "deepseek-reasoner",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello!"}
            ],
            "max_tokens": 100,
            "temperature": 0.7,
            "stream": false
        })),
    ];

    for (type_name, request_body) in test_cases {
        println!("📡 Making API call for {}...", type_name);

        let response = client
            .post("https://api.deepseek.com/beta/v1/chat/completions")
            .header("Authorization", format!("Bearer {}", api_key))
            .header("Content-Type", "application/json")
            .json(&request_body)
            .send()
            .await?;

        let response_text = response.text().await?;
        
        match serde_json::from_str::<Value>(&response_text) {
            Ok(json) => {
                println!("✅ Received response for {}", type_name);
                
                // Generate response type
                let response_type_name = format!("{}Response", type_name.replace("Request", ""));
                generator.generate_type_from_json(&json, &response_type_name);
                
                // Also generate request type for completeness
                generator.generate_type_from_json(&request_body, type_name);
            }
            Err(e) => {
                eprintln!("❌ Failed to parse JSON response for {}: {}", type_name, e);
                eprintln!("Raw response: {}", response_text);
            }
        }
    }

    // Generate additional known types manually for comprehensive coverage
    let usage_json = serde_json::json!({
        "prompt_tokens": 10,
        "completion_tokens": 20,
        "total_tokens": 30,
        "prompt_cache_hit_tokens": 5,
        "prompt_cache_miss_tokens": 5
    });
    generator.generate_type_from_json(&usage_json, "Usage");

    let choice_json = serde_json::json!({
        "index": 0,
        "message": {
            "role": "assistant",
            "content": "Hello! How can I help you today?"
        },
        "finish_reason": "stop"
    });
    generator.generate_type_from_json(&choice_json, "Choice");

    let message_json = serde_json::json!({
        "role": "assistant",
        "content": "Hello! How can I help you today?"
    });
    generator.generate_type_from_json(&message_json, "Message");

    let tool_json = serde_json::json!({
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather information",
            "strict": true,
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string"}
                }
            }
        }
    });
    generator.generate_type_from_json(&tool_json, "Tool");

    // Write generated types to file
    let output = generator.generate_output();
    let output_path = "src/deepseek/generated_types.rs";
    
    fs::write(output_path, output)?;
    
    println!("✅ Generated types written to {}", output_path);
    println!("🔧 Generated {} types", generator.generated_types.len());
    
    // Print summary
    println!("\n📋 Generated types:");
    for type_name in &generator.seen_types {
        println!("  - {}", type_name);
    }

    Ok(())
}
