//! DeepSeek API client implementation
//!
//! This module provides a direct API client for DeepSeek's chat completion API,
//! bypassing external dependencies for better control and reliability.

use anyhow::{anyhow, Context, Result};
use serde::{Deserialize, Serialize};
use std::time::Duration;
use super::types::*;

/// DeepSeek API client
pub struct DeepSeekClient {
    client: reqwest::Client,
    api_key: String,
    base_url: String,
}

/// Response choice
#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct Choice {
    index: u32,
    message: Message,
    finish_reason: Option<String>,
}

/// Token usage information with context caching support
#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct Usage {
    prompt_tokens: u32,
    completion_tokens: u32,
    total_tokens: u32,
    /// Number of tokens that resulted in a cache hit (0.1 yuan per million tokens)
    prompt_cache_hit_tokens: Option<u32>,
    /// Number of tokens that did not result in a cache hit (1 yuan per million tokens)
    prompt_cache_miss_tokens: Option<u32>,
}

/// Error response from DeepSeek API
#[derive(Debug, Deserialize)]
struct ErrorResponse {
    error: ErrorDetail,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct ErrorDetail {
    message: String,
    #[serde(rename = "type")]
    error_type: Option<String>,
    code: Option<String>,
}

impl DeepSeekClient {
    /// Create a new DeepSeek client with robust configuration
    pub fn new(api_key: &str) -> Self {
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(180)) // Increased timeout for large responses
            .connect_timeout(Duration::from_secs(30)) // Connection timeout
            .read_timeout(Duration::from_secs(120)) // Read timeout for response body
            .tcp_keepalive(Duration::from_secs(60)) // Keep connections alive
            .pool_idle_timeout(Duration::from_secs(90)) // Pool management
            .pool_max_idle_per_host(2) // Limit idle connections
            .user_agent("ai-architect/1.0") // Identify our client
            .build()
            .expect("Failed to create HTTP client");

        Self {
            api_key: api_key.to_string(),
            base_url: "https://api.deepseek.com/beta/v1".to_string(), // Use beta endpoint for strict mode
            client,
        }
    }

    /// Send a chat completion request
    pub async fn chat_completion(
        &self,
        model: &str,
        messages: Vec<(String, String)>, // (role, content) pairs
        max_tokens: Option<u32>,
        temperature: Option<f32>,
    ) -> Result<String> {
        self.chat_completion_with_tools(model, messages, max_tokens, temperature, None)
            .await
    }

    /// Send a chat completion request with tools (strict mode support)
    pub async fn chat_completion_with_tools(
        &self,
        model: &str,
        messages: Vec<(String, String)>, // (role, content) pairs
        max_tokens: Option<u32>,
        temperature: Option<f32>,
        tools: Option<Vec<Tool>>,
    ) -> Result<String> {
        let messages: Vec<Message> = messages
            .into_iter()
            .map(|(role, content)| Message { role, content })
            .collect();

        let request = ChatCompletionRequest {
            model: model.to_string(),
            messages,
            max_tokens,
            temperature,
            top_p: None,
            stream: false,
            tools, // Use tools parameter
        };

        let url = format!("{}/chat/completions", self.base_url);

        let response = self
            .client
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&request)
            .send()
            .await
            .context("Failed to send request to DeepSeek API")?;

        let status = response.status();
        let response_text = response
            .text()
            .await
            .context("Failed to read response body")?;

        if !status.is_success() {
            // Try to parse error response
            if let Ok(error_response) = serde_json::from_str::<ErrorResponse>(&response_text) {
                return Err(anyhow!(
                    "DeepSeek API error: {} ({})",
                    error_response.error.message,
                    status
                ));
            } else {
                return Err(anyhow!(
                    "DeepSeek API request failed with status {}: {}",
                    status,
                    response_text
                ));
            }
        }

        let completion: ChatCompletionResponse = serde_json::from_str(&response_text)
            .context("Failed to parse chat completion response")?;

        // Log context caching information if available
        if let (Some(cache_hit), Some(cache_miss)) = (
            completion.usage.prompt_cache_hit_tokens,
            completion.usage.prompt_cache_miss_tokens,
        ) {
            tracing::info!(
                "Context cache stats: {} hit tokens, {} miss tokens, cache efficiency: {:.1}%",
                cache_hit,
                cache_miss,
                if cache_hit + cache_miss > 0 {
                    (cache_hit as f32 / (cache_hit + cache_miss) as f32) * 100.0
                } else {
                    0.0
                }
            );
        }

        if completion.choices.is_empty() {
            return Err(anyhow!("No choices returned from DeepSeek API"));
        }

        Ok(completion.choices[0].message.content.clone())
    }

    /// Simple prompt method for quick interactions
    #[allow(dead_code)]
    pub async fn prompt(
        &self,
        model: &str,
        system_prompt: &str,
        user_prompt: &str,
    ) -> Result<String> {
        let messages = vec![
            ("system".to_string(), system_prompt.to_string()),
            ("user".to_string(), user_prompt.to_string()),
        ];
        self.chat_completion(model, messages, Some(4000), Some(0.7))
            .await
    }

    /// Create a tool with strict mode enabled
    #[allow(dead_code)]
    pub fn create_strict_tool(
        name: &str,
        description: &str,
        parameters: serde_json::Value,
    ) -> Tool {
        Tool {
            tool_type: "function".to_string(),
            function: Function {
                name: name.to_string(),
                description: description.to_string(),
                strict: Some(true), // Enable strict mode
                parameters,
            },
        }
    }

    /// Agent-like interface for maintaining conversation context
    pub fn agent(&self, model: &str) -> Agent<'_> {
        Agent::new(self, model)
    }

    /// Create a cache-optimized agent for architecture analysis with few-shot examples
    /// This leverages DeepSeek's context caching to reduce costs for repeated architectural patterns
    pub fn architecture_agent(&self, model: &str) -> Agent<'_> {
        Agent::new(self, model)
            .with_few_shot_examples(vec![
                (
                    "Analyze this Rust code for potential improvements: `fn process_data(data: Vec<String>) -> Vec<String> { data.iter().map(|s| s.to_uppercase()).collect() }`".to_string(),
                    "**Improvement**: Use iterator adapters more efficiently and consider borrowing:\n```rust\nfn process_data(data: &[String]) -> Vec<String> {\n    data.iter().map(|s| s.to_uppercase()).collect()\n}\n```\n**Benefits**: Avoids unnecessary ownership transfer, more flexible API.".to_string(),
                ),
                (
                    "What tiny architectural improvement can be made here: `struct Config { pub host: String, pub port: u16 }`".to_string(),
                    "**Improvement**: Add validation and builder pattern:\n```rust\nstruct Config {\n    host: String,\n    port: u16,\n}\n\nimpl Config {\n    pub fn new(host: String, port: u16) -> Result<Self, ConfigError> {\n        if host.is_empty() { return Err(ConfigError::EmptyHost); }\n        Ok(Self { host, port })\n    }\n}\n```\n**Benefits**: Encapsulation, validation, error handling.".to_string(),
                ),
            ])
    }
}

/// Agent wrapper for maintaining conversation state with context caching optimization
pub struct Agent<'a> {
    client: &'a DeepSeekClient,
    model: String,
    system_prompt: Option<String>,
    messages: Vec<Message>,
    /// Base context that should be preserved for cache hits
    base_context: Vec<Message>,
}

impl<'a> Agent<'a> {
    fn new(client: &'a DeepSeekClient, model: &str) -> Self {
        Self {
            client,
            model: model.to_string(),
            system_prompt: None,
            messages: Vec::new(),
            base_context: Vec::new(),
        }
    }

    /// Set the system prompt (preamble)
    pub fn preamble(mut self, prompt: &str) -> Self {
        self.system_prompt = Some(prompt.to_string());
        self
    }

    /// Set base context for cache optimization
    /// This context will be preserved across conversations to maximize cache hits
    #[allow(dead_code)]
    pub fn with_base_context(mut self, context_messages: Vec<(String, String)>) -> Self {
        self.base_context = context_messages
            .into_iter()
            .map(|(role, content)| Message { role, content })
            .collect();
        self
    }

    /// Add few-shot examples to base context for cache-optimized learning
    pub fn with_few_shot_examples(mut self, examples: Vec<(String, String)>) -> Self {
        for (user_msg, assistant_msg) in examples {
            self.base_context.push(Message {
                role: "user".to_string(),
                content: user_msg,
            });
            self.base_context.push(Message {
                role: "assistant".to_string(),
                content: assistant_msg,
            });
        }
        self
    }

    /// Build the agent (returns self for compatibility)
    pub fn build(self) -> Self {
        self
    }

    /// Send a prompt and get a response (optimized for context caching)
    pub async fn prompt(&mut self, user_message: &str) -> Result<String> {
        // Prepare messages for API call with consistent prefix ordering for cache optimization
        let mut api_messages = Vec::new();

        // 1. Add system prompt if set (should be first for cache consistency)
        if let Some(ref system_prompt) = self.system_prompt {
            api_messages.push(("system".to_string(), system_prompt.clone()));
        }

        // 2. Add base context (few-shot examples, etc.) for cache optimization
        for msg in &self.base_context {
            api_messages.push((msg.role.clone(), msg.content.clone()));
        }

        // 3. Add conversation history (maintaining order for prefix matching)
        for msg in &self.messages {
            api_messages.push((msg.role.clone(), msg.content.clone()));
        }

        // 4. Add current user message
        api_messages.push(("user".to_string(), user_message.to_string()));

        // Send to API - DeepSeek will automatically cache the prefix
        let response = self
            .client
            .chat_completion(&self.model, api_messages, Some(4000), Some(0.7))
            .await?;

        // Update conversation history (not base context, to preserve cache-friendly prefix)
        self.messages.push(Message {
            role: "user".to_string(),
            content: user_message.to_string(),
        });
        self.messages.push(Message {
            role: "assistant".to_string(),
            content: response.clone(),
        });

        Ok(response)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_client_creation() {
        let client = DeepSeekClient::new("test-api-key");
        assert_eq!(client.api_key, "test-api-key");
        assert_eq!(client.base_url, "https://api.deepseek.com/beta/v1");
    }

    #[test]
    fn test_agent_creation() {
        let client = DeepSeekClient::new("test-key");
        let agent = client.agent(models::DEEPSEEK_CHAT);
        assert_eq!(agent.model, models::DEEPSEEK_CHAT);
    }

    #[test]
    fn test_architecture_agent_creation() {
        let client = DeepSeekClient::new("test-key");
        let agent = client.architecture_agent(models::DEEPSEEK_REASONER);
        assert_eq!(agent.model, models::DEEPSEEK_REASONER);
        // Architecture agent should have base context for cache optimization
        assert!(!agent.base_context.is_empty());
    }
}
