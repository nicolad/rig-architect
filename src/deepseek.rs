//! DeepSeek API client implementation
//!
//! This module provides a direct API client for DeepSeek's chat completion API,
//! bypassing external dependencies for better control and reliability.
//!
//! Features:
//! - Custom HTTP client using reqwest
//! - Support for DeepSeek's beta endpoint with strict mode
//! - Function calling with strict JSON schema validation
//! - Agent-like interface for conversation management

use anyhow::{anyhow, Context, Result};
use serde::{Deserialize, Serialize};
use std::time::Duration;

/// DeepSeek API client
pub struct DeepSeekClient {
    client: reqwest::Client,
    api_key: String,
    base_url: String,
}

/// Chat completion request
#[derive(Debug, Serialize)]
struct ChatCompletionRequest {
    model: String,
    messages: Vec<Message>,
    max_tokens: Option<u32>,
    temperature: Option<f32>,
    top_p: Option<f32>,
    stream: bool,
    tools: Option<Vec<Tool>>,
}

/// Tool definition for function calling
#[derive(Debug, Serialize)]
pub struct Tool {
    #[serde(rename = "type")]
    tool_type: String,
    function: Function,
}

/// Function definition for strict mode
#[derive(Debug, Serialize)]
pub struct Function {
    name: String,
    description: String,
    strict: Option<bool>,
    parameters: serde_json::Value,
}

/// Chat message
#[derive(Debug, Serialize, Deserialize)]
struct Message {
    role: String,
    content: String,
}

/// Chat completion response
#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct ChatCompletionResponse {
    id: String,
    object: String,
    created: u64,
    model: String,
    choices: Vec<Choice>,
    usage: Usage,
}

/// Response choice
#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct Choice {
    index: u32,
    message: Message,
    finish_reason: Option<String>,
}

/// Token usage information
#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct Usage {
    prompt_tokens: u32,
    completion_tokens: u32,
    total_tokens: u32,
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

/// DeepSeek model constants
pub mod models {
    #[allow(dead_code)]
    pub const DEEPSEEK_CHAT: &str = "deepseek-chat";
    pub const DEEPSEEK_REASONER: &str = "deepseek-reasoner";
}

impl DeepSeekClient {
    /// Create a new DeepSeek client
    pub fn new(api_key: &str) -> Self {
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(120))
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
}

/// Agent wrapper for maintaining conversation state
pub struct Agent<'a> {
    client: &'a DeepSeekClient,
    model: String,
    system_prompt: Option<String>,
    messages: Vec<Message>,
}

impl<'a> Agent<'a> {
    fn new(client: &'a DeepSeekClient, model: &str) -> Self {
        Self {
            client,
            model: model.to_string(),
            system_prompt: None,
            messages: Vec::new(),
        }
    }

    /// Set the system prompt (preamble)
    pub fn preamble(mut self, prompt: &str) -> Self {
        self.system_prompt = Some(prompt.to_string());
        self
    }

    /// Build the agent (returns self for compatibility)
    pub fn build(self) -> Self {
        self
    }

    /// Send a prompt and get a response
    pub async fn prompt(&mut self, user_message: &str) -> Result<String> {
        // Prepare messages for API call
        let mut api_messages = Vec::new();

        // Add system prompt if set
        if let Some(ref system_prompt) = self.system_prompt {
            api_messages.push(("system".to_string(), system_prompt.clone()));
        }

        // Add conversation history
        for msg in &self.messages {
            api_messages.push((msg.role.clone(), msg.content.clone()));
        }

        // Add current user message
        api_messages.push(("user".to_string(), user_message.to_string()));

        // Send to API
        let response = self
            .client
            .chat_completion(&self.model, api_messages, Some(4000), Some(0.7))
            .await?;

        // Update conversation history
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
}
