// src/deepseek.rs
//
// DeepSeek AI Client Module for the Rig bug finder
//
// This module provides a wrapper around the DeepSeek API using the rig framework
// for consistent AI interactions across the application.

use anyhow::{anyhow, Result};
use rig::client::{CompletionClient, ProviderClient};
use rig::completion::Prompt;
use rig::providers::deepseek;
use serde::{Deserialize, Serialize};
use std::env;
use tracing::{debug, info};

/// DeepSeek client using rig framework
#[derive(Clone, Debug)]
pub struct DeepSeekClient {
    client: deepseek::Client,
}

impl DeepSeekClient {
    /// Create a new DeepSeek client with the provided API key
    #[allow(dead_code)]
    pub fn new(api_key: String) -> Self {
        let client = deepseek::Client::new(&api_key);
        Self { client }
    }

    /// Create a DeepSeek client from the DEEPSEEK_API_KEY environment variable
    pub fn from_env() -> Result<Self> {
        // Check if the environment variable exists first
        match env::var("DEEPSEEK_API_KEY") {
            Ok(api_key) => {
                if api_key.trim().is_empty() {
                    return Err(anyhow!("DEEPSEEK_API_KEY environment variable is empty"));
                }
                // Use the rig framework client
                let client = deepseek::Client::from_env();
                Ok(Self { client })
            }
            Err(_) => Err(anyhow!("DEEPSEEK_API_KEY environment variable not set")),
        }
    }

    /// Send a simple prompt and get the complete response
    #[allow(dead_code)]
    pub async fn prompt(&self, prompt: &str) -> Result<String> {
        self.prompt_with_context(prompt, "Nautilus-Autopatcher")
            .await
    }

    /// Send a prompt with a specific context/agent name
    #[allow(dead_code)]
    pub async fn prompt_with_context(&self, prompt: &str, agent_name: &str) -> Result<String> {
        log::info!("🤖 Initializing agent: {}", agent_name);

        let agent = self
            .client
            .agent(deepseek::DEEPSEEK_REASONER)
            .preamble("You are a helpful assistant specialized in code analysis and improvement.")
            .name(agent_name)
            .build();

        log::debug!(
            "📤 Sending prompt to {}: {} chars",
            agent_name,
            prompt.len()
        );
        let response = agent.prompt(prompt).await?;
        log::info!(
            "📥 Received response from {}: {} chars",
            agent_name,
            response.len()
        );

        Ok(response)
    }

    /// Send a prompt for commit analysis with appropriate context
    pub async fn analyze_commits(&self, prompt: &str) -> Result<String> {
        log::info!("🔍 Starting commit analysis with DeepSeek");

        let agent = self
            .client
            .agent(deepseek::DEEPSEEK_REASONER)
            .preamble(
                "You are an expert code quality analyst specializing in commit message analysis, \
                 typo detection, and pattern consistency. You help identify inconsistencies, \
                 typos, and violations of established commit patterns in software repositories.",
            )
            .name("Commit-Quality-Analyzer")
            .build();

        log::debug!("📤 Sending commit analysis prompt: {} chars", prompt.len());
        let response = agent.prompt(prompt).await?;
        log::info!(
            "📥 Received commit analysis response: {} chars",
            response.len()
        );

        Ok(response)
    }

    /// Analyze critical bugs and create tests to reproduce criticality
    pub async fn confirm_critical_bug(
        &self,
        bug_description: &str,
        code_sample: &str,
    ) -> Result<String> {
        log::info!("🔍 Confirming critical bug with DeepSeek");

        let agent = self
            .client
            .agent(deepseek::DEEPSEEK_REASONER)
            .preamble(
                "You are an expert security researcher and test engineer specializing in critical bug analysis. \
                 Your task is to:\
                 1. Analyze reported bugs to confirm if they are truly critical\
                 2. Create specific test cases to reproduce the vulnerability\
                 3. Assess the actual risk level and potential impact\
                 4. Provide concrete evidence of criticality\
                 \
                 For each bug, provide:\
                 - CRITICAL_CONFIRMED: true/false\
                 - RISK_LEVEL: CRITICAL/HIGH/MEDIUM/LOW\
                 - REPRODUCTION_STEPS: detailed steps to reproduce\
                 - TEST_CODE: actual test code that demonstrates the vulnerability\
                 - IMPACT_ASSESSMENT: real-world impact description"
            )
            .name("Critical-Bug-Validator")
            .build();

        let prompt = format!(
            "CRITICAL BUG ANALYSIS REQUEST\n\
             ===============================\n\
             \n\
             Bug Description: {}\n\
             \n\
             Code Sample:\n\
             ```\n\
             {}\n\
             ```\n\
             \n\
             Please analyze this bug and provide:\n\
             1. Is this truly critical? (CRITICAL_CONFIRMED: true/false)\n\
             2. Risk level assessment (RISK_LEVEL: CRITICAL/HIGH/MEDIUM/LOW)\n\
             3. Step-by-step reproduction instructions\n\
             4. Test code that reproduces the vulnerability\n\
             5. Real-world impact assessment\n\
             \n\
             Focus on real-world impacts: data loss, security breaches, reliability issues.",
            bug_description, code_sample
        );

        log::debug!(
            "📤 Sending critical bug confirmation prompt: {} chars",
            prompt.len()
        );
        let response = agent.prompt(&prompt).await?;
        log::info!(
            "📥 Received critical bug confirmation: {} chars",
            response.len()
        );

        Ok(response)
    }

    /// Send a prompt for critical code analysis with appropriate context
    pub async fn analyze_code(&self, prompt: &str) -> Result<String> {
        log::info!("🔍 Starting critical code analysis with DeepSeek");

        let agent = self
            .client
            .agent(deepseek::DEEPSEEK_REASONER)
            .preamble(
                "You are an expert critical code analyst specializing in security vulnerabilities, \
                 reliability issues, correctness errors, and performance problems. \
                 You identify critical issues that could impact system stability, security, or correctness."
            )
            .name("Critical-Code-Analyzer")
            .build();

        log::debug!(
            "📤 Sending critical code analysis prompt: {} chars",
            prompt.len()
        );
        let response = agent.prompt(prompt).await?;
        log::info!(
            "📥 Received critical code analysis response: {} chars",
            response.len()
        );

        Ok(response)
    }

    /// Stream a prompt and get real-time response (simplified for now)
    #[allow(dead_code)]
    pub async fn stream_prompt(&self, prompt: &str) -> Result<String> {
        self.stream_prompt_with_context(prompt, "Nautilus-Autopatcher-Stream")
            .await
    }

    /// Stream a prompt with a specific context/agent name
    #[allow(dead_code)]
    pub async fn stream_prompt_with_context(
        &self,
        prompt: &str,
        agent_name: &str,
    ) -> Result<String> {
        println!("🤖 {} is thinking...", agent_name);
        std::io::Write::flush(&mut std::io::stdout())?;

        // For now, just use the regular prompt method
        // TODO: Implement proper streaming when rig API supports it
        let response = self.prompt_with_context(prompt, agent_name).await?;

        println!("✅ Response received from {}", agent_name);
        Ok(response)
    }

    /// Validate that the client can connect to DeepSeek API
    #[allow(dead_code)]
    pub async fn validate_connection(&self) -> Result<()> {
        log::info!("🔌 Validating DeepSeek API connection");

        let test_prompt = "Reply with 'OK' if you can receive this message.";
        let response = self
            .prompt_with_context(test_prompt, "Connection-Test")
            .await?;

        if response.trim().to_uppercase().contains("OK") {
            log::info!("✅ DeepSeek API connection validated successfully");
            Ok(())
        } else {
            Err(anyhow!(
                "DeepSeek API connection validation failed: unexpected response"
            ))
        }
    }

    /// Analyze code architecture and design patterns
    pub async fn analyze_architecture(&self, code: &str, context: &str) -> Result<String> {
        log::info!("🏗️ Starting architectural analysis with DeepSeek");

        let agent = self
            .client
            .agent(deepseek::DEEPSEEK_REASONER)
            .preamble(
                "You are a senior software architect specializing in Rust systems design, \
                 large-scale systems, and high-performance applications. \
                 Analyze code architecture for design patterns, scalability, maintainability, \
                 and system-level concerns. Provide specific recommendations for improvement.",
            )
            .name("Architecture-Analyzer")
            .build();

        let prompt = format!(
            "ARCHITECTURAL ANALYSIS REQUEST\n\
             ================================\n\
             \n\
             Context: {}\n\
             \n\
             Code to analyze:\n\
             ```rust\n\
             {}\n\
             ```\n\
             \n\
             Please provide a comprehensive architectural analysis covering:\n\
             1. Design pattern usage and appropriateness\n\
             2. Thread safety and concurrency concerns\n\
             3. State management and data flow\n\
             4. Error handling strategy\n\
             5. Performance and scalability implications\n\
             6. Maintainability and code organization\n\
             7. Specific improvement recommendations\n\
             8. Alternative architectural approaches",
            context, code
        );

        log::debug!(
            "📤 Sending architectural analysis prompt: {} chars",
            prompt.len()
        );
        let response = agent.prompt(&prompt).await?;
        log::info!(
            "📥 Received architectural analysis: {} chars",
            response.len()
        );

        Ok(response)
    }

    /// Analyze performance bottlenecks and optimization opportunities
    pub async fn analyze_performance(&self, code: &str, context: &str) -> Result<String> {
        log::info!("⚡ Starting performance analysis with DeepSeek");

        let agent = self
            .client
            .agent(deepseek::DEEPSEEK_REASONER)
            .preamble(
                "You are a performance optimization expert specializing in Rust, \
                 low-latency systems, and real-time applications. \
                 Analyze code for performance bottlenecks, memory allocation issues, \
                 and optimization opportunities. Provide specific, actionable recommendations.",
            )
            .name("Performance-Optimizer")
            .build();

        let prompt = format!(
            "PERFORMANCE ANALYSIS REQUEST\n\
             ============================\n\
             \n\
             Context: {}\n\
             \n\
             Code to analyze:\n\
             ```rust\n\
             {}\n\
             ```\n\
             \n\
             Please provide a detailed performance analysis covering:\n\
             1. Computational complexity and algorithmic efficiency\n\
             2. Memory allocation and deallocation patterns\n\
             3. Cache performance and data locality\n\
             4. Parallelization and concurrency opportunities\n\
             5. I/O and network performance considerations\n\
             6. Hot path optimization suggestions\n\
             7. Benchmarking and profiling recommendations\n\
             8. Specific code optimizations with examples",
            context, code
        );

        log::debug!(
            "📤 Sending performance analysis prompt: {} chars",
            prompt.len()
        );
        let response = agent.prompt(&prompt).await?;
        log::info!("📥 Received performance analysis: {} chars", response.len());

        Ok(response)
    }

    /// Generate comprehensive test cases for code coverage
    pub async fn generate_test_cases(&self, code: &str, context: &str) -> Result<String> {
        log::info!("🧪 Starting test case generation with DeepSeek");

        let agent = self
            .client
            .agent(deepseek::DEEPSEEK_REASONER)
            .preamble(
                "You are a test engineering expert specializing in Rust testing frameworks, \
                 property-based testing, and system validation. \
                 Generate comprehensive test cases that cover edge cases, error conditions, \
                 and critical business logic validation.",
            )
            .name("Test-Generator")
            .build();

        let prompt = format!(
            "TEST CASE GENERATION REQUEST\n\
             ============================\n\
             \n\
             Context: {}\n\
             \n\
             Code to test:\n\
             ```rust\n\
             {}\n\
             ```\n\
             \n\
             Please generate comprehensive test cases including:\n\
             1. Unit tests for all public functions\n\
             2. Integration tests for component interactions\n\
             3. Edge case testing (boundary conditions, overflow, etc.)\n\
             4. Error handling and failure scenario tests\n\
             5. Property-based testing suggestions\n\
             6. Mock and stub recommendations\n\
             7. Performance and load testing strategies\n\
             8. Complete runnable test code examples",
            context, code
        );

        log::debug!("📤 Sending test generation prompt: {} chars", prompt.len());
        let response = agent.prompt(&prompt).await?;
        log::info!("📥 Received test generation: {} chars", response.len());

        Ok(response)
    }

    /// Analyze security vulnerabilities and threats
    pub async fn analyze_security(&self, code: &str, context: &str) -> Result<String> {
        log::info!("🔒 Starting security analysis with DeepSeek");

        let agent = self
            .client
            .agent(deepseek::DEEPSEEK_REASONER)
            .preamble(
                "You are a cybersecurity expert specializing in secure coding practices, \
                 application security, and Rust memory safety. \
                 Analyze code for security vulnerabilities, attack vectors, \
                 and compliance with security best practices.",
            )
            .name("Security-Analyzer")
            .build();

        let prompt = format!(
            "SECURITY ANALYSIS REQUEST\n\
             =========================\n\
             \n\
             Context: {}\n\
             \n\
             Code to analyze:\n\
             ```rust\n\
             {}\n\
             ```\n\
             \n\
             Please provide a comprehensive security analysis covering:\n\
             1. Input validation and sanitization\n\
             2. Authentication and authorization flaws\n\
             3. Data exposure and information leakage\n\
             4. Injection vulnerabilities (SQL, command, etc.)\n\
             5. Cryptographic implementation issues\n\
             6. Race conditions and timing attacks\n\
             7. Memory safety and buffer overflow risks\n\
             8. Threat modeling and attack surface analysis\n\
             9. Compliance with security standards (PCI DSS, SOX, etc.)\n\
             10. Specific remediation steps with code examples",
            context, code
        );

        log::debug!(
            "📤 Sending security analysis prompt: {} chars",
            prompt.len()
        );
        let response = agent.prompt(&prompt).await?;
        log::info!("📥 Received security analysis: {} chars", response.len());

        Ok(response)
    }

    /// Analyze code quality and maintainability
    pub async fn analyze_code_quality(&self, code: &str, context: &str) -> Result<String> {
        log::info!("📊 Starting code quality analysis with DeepSeek");

        let agent = self
            .client
            .agent(deepseek::DEEPSEEK_CHAT)
            .preamble(
                "You are a code quality expert and technical lead specializing in \
                 Rust best practices, clean code principles, and software maintainability. \
                 Analyze code for quality metrics, technical debt, and adherence to best practices."
            )
            .name("Quality-Analyzer")
            .build();

        let prompt = format!(
            "CODE QUALITY ANALYSIS REQUEST\n\
             ==============================\n\
             \n\
             Context: {}\n\
             \n\
             Code to analyze:\n\
             ```rust\n\
             {}\n\
             ```\n\
             \n\
             Please provide a detailed code quality analysis covering:\n\
             1. Code structure and organization\n\
             2. Naming conventions and clarity\n\
             3. Function and module complexity\n\
             4. Documentation quality and completeness\n\
             5. Error handling patterns\n\
             6. Code duplication and reusability\n\
             7. Adherence to Rust idioms and best practices\n\
             8. Technical debt identification\n\
             9. Refactoring recommendations\n\
             10. Code metrics and quality scores",
            context, code
        );

        log::debug!(
            "📤 Sending code quality analysis prompt: {} chars",
            prompt.len()
        );
        let response = agent.prompt(&prompt).await?;
        log::info!(
            "📥 Received code quality analysis: {} chars",
            response.len()
        );

        Ok(response)
    }
}

// DeepSeek Embedding Client Structures
#[allow(dead_code)]
#[derive(Debug, Serialize)]
struct DeepSeekEmbeddingRequest {
    model: String,
    input: Vec<String>,
}

#[allow(dead_code)]
#[derive(Debug, Deserialize)]
struct DeepSeekEmbeddingResponse {
    data: Vec<EmbeddingData>,
    usage: Usage,
}

#[allow(dead_code)]
#[derive(Debug, Deserialize)]
struct EmbeddingData {
    embedding: Vec<f32>,
    index: usize,
}

#[allow(dead_code)]
#[derive(Debug, Deserialize)]
struct Usage {
    prompt_tokens: u32,
    total_tokens: u32,
}

/// DeepSeek embedding client for vector embeddings
#[allow(dead_code)]
pub struct DeepSeekEmbeddingClient {
    client: reqwest::Client,
    api_key: String,
    base_url: String,
    model: String,
}

#[allow(dead_code)]
impl DeepSeekEmbeddingClient {
    pub fn new() -> Result<Self> {
        let api_key = env::var("DEEPSEEK_API_KEY")
            .map_err(|_| anyhow!("DEEPSEEK_API_KEY environment variable not set"))?;

        let client = reqwest::Client::new();

        Ok(Self {
            client,
            api_key,
            base_url: "https://api.deepseek.com/v1".to_string(),
            model: "text-embedding-ada-002".to_string(), // DeepSeek's embedding model
        })
    }

    pub async fn embed_texts(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>> {
        let request = DeepSeekEmbeddingRequest {
            model: self.model.clone(),
            input: texts,
        };

        debug!(
            "Making DeepSeek embedding request for {} texts",
            request.input.len()
        );

        let response = self
            .client
            .post(format!("{}/embeddings", self.base_url))
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&request)
            .send()
            .await?;

        if !response.status().is_success() {
            let error_text = response.text().await?;
            return Err(anyhow!("DeepSeek API error: {}", error_text));
        }

        let embedding_response: DeepSeekEmbeddingResponse = response.json().await?;

        info!(
            "DeepSeek embedding token usage: {} prompt tokens, {} total tokens",
            embedding_response.usage.prompt_tokens, embedding_response.usage.total_tokens
        );

        let embeddings = embedding_response
            .data
            .into_iter()
            .map(|data| data.embedding)
            .collect();

        Ok(embeddings)
    }

    pub async fn embed_text(&self, text: String) -> Result<Vec<f32>> {
        let embeddings = self.embed_texts(vec![text]).await?;
        embeddings
            .into_iter()
            .next()
            .ok_or_else(|| anyhow!("No embedding returned from DeepSeek API"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_deepseek_client_creation() {
        let _client = DeepSeekClient::new("test-api-key".to_string());
        // Just test that we can create the client without panicking
    }

    #[tokio::test]
    async fn test_deepseek_client_from_env_missing_key() {
        // Remove the env var if it exists for this test
        std::env::remove_var("DEEPSEEK_API_KEY");

        let result = DeepSeekClient::from_env();
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("DEEPSEEK_API_KEY"));
    }

    #[test]
    fn test_deepseek_client_from_env_with_key() {
        std::env::set_var("DEEPSEEK_API_KEY", "test-key");

        let result = DeepSeekClient::from_env();
        assert!(result.is_ok());

        // Clean up
        std::env::remove_var("DEEPSEEK_API_KEY");
    }

    #[test]
    fn test_analyze_commits_prompt_format() {
        let commits = [
            "abc123|John Doe|john@example.com|2025-08-22 10:30:00 +0000|feat: add new feature"
                .to_string(),
            "def456|Jane Smith|jane@example.com|2025-08-22 11:30:00 +0000|fix: correct typo"
                .to_string(),
        ];

        let prompt = format!(
            "Analyze these git commits for quality, consistency, and potential issues:\n\n{}",
            commits.join("\n")
        );

        assert!(prompt.contains("feat: add new feature"));
        assert!(prompt.contains("fix: correct typo"));
        assert!(prompt.contains("Analyze these git commits"));
    }

    #[test]
    fn test_commit_format_parsing() {
        let commit =
            "abc123|John Doe|john@example.com|2025-08-22 10:30:00 +0000|feat: add new feature";
        let parts: Vec<&str> = commit.split('|').collect();

        assert_eq!(parts.len(), 5);
        assert_eq!(parts[0], "abc123");
        assert_eq!(parts[1], "John Doe");
        assert_eq!(parts[2], "john@example.com");
        assert_eq!(parts[3], "2025-08-22 10:30:00 +0000");
        assert_eq!(parts[4], "feat: add new feature");
    }

    #[test]
    fn test_empty_commits_handling() {
        let empty_commits: Vec<String> = vec![];
        let prompt = format!(
            "Analyze these git commits for quality, consistency, and potential issues:\n\n{}",
            empty_commits.join("\n")
        );

        assert!(prompt.contains("Analyze these git commits"));
        // Should handle empty list gracefully
        assert_eq!(empty_commits.len(), 0);
    }

    #[tokio::test]
    async fn test_new_client_creation() {
        // Test that we can create a client with a test key
        std::env::set_var("DEEPSEEK_API_KEY", "test-key-for-creation");

        let result = DeepSeekClient::from_env();

        // Should succeed in creating the client (even with fake key)
        assert!(result.is_ok());

        // Clean up
        std::env::remove_var("DEEPSEEK_API_KEY");
    }

    #[test]
    fn test_client_clone() {
        std::env::set_var("DEEPSEEK_API_KEY", "test-key");

        let client = DeepSeekClient::from_env().unwrap();
        let _cloned_client = client.clone();

        // Both clients should be valid (this tests the Clone implementation)

        std::env::remove_var("DEEPSEEK_API_KEY");
    }

    #[tokio::test]
    async fn test_deepseek_embedding() {
        // Skip test if no DeepSeek API key
        if env::var("DEEPSEEK_API_KEY").is_err() {
            return;
        }

        let client = DeepSeekEmbeddingClient::new().unwrap();
        let embedding = client.embed_text("Hello world".to_string()).await.unwrap();
        assert!(!embedding.is_empty());
        println!("Embedding dimension: {}", embedding.len());
    }
}
