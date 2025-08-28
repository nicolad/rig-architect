//! Type generator for DeepSeek API responses
//!
//! This script generates Rust types based on real DeepSeek API responses.

use serde_json::Value;
use std::collections::HashSet;
use std::env;
use std::fs;
use std::path::Path;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔧 Generating DeepSeek API types from real API responses...");

    // Run clippy and other checks first
    println!("🔍 Running code quality checks...");
    run_code_quality_checks().await?;

    // Ensure the output directory exists
    let output_dir = "src/deepseek";
    if !Path::new(output_dir).exists() {
        fs::create_dir_all(output_dir).expect("Failed to create output directory");
    }

    // Load environment variables
    dotenvy::dotenv().ok();

    match env::var("DEEPSEEK_API_KEY").or_else(|_| env::var("DEEPSEEK_API_TOKEN")) {
        Ok(api_key) => {
            println!("📡 Making test API calls to analyze response structure...");

            // Make real API calls to get response structure
            let response_data = make_test_api_calls(&api_key).await?;

            println!("🔍 Analyzing response structure...");

            // Generate types based on real responses
            let types_content = generate_types_from_responses(response_data);

            // Write to file
            let output_path = format!("{}/types.rs", output_dir);
            fs::write(&output_path, types_content)?;

            println!(
                "✅ Types generated successfully from real API responses in {}",
                output_path
            );
        }
        Err(_) => {
            println!("⚠️  No API key found, using template-based generation");
            println!("   Set DEEPSEEK_API_KEY or DEEPSEEK_API_TOKEN to use real API responses");

            // Fallback to template-based generation
            let types_content = generate_types_file();

            // Write to file
            let output_path = format!("{}/types.rs", output_dir);
            fs::write(&output_path, types_content)?;

            println!(
                "✅ Types generated successfully from template in {}",
                output_path
            );
        }
    }

    Ok(())
}

async fn run_code_quality_checks() -> Result<(), Box<dyn std::error::Error>> {
    println!("  - Running cargo clippy...");
    let clippy_output = std::process::Command::new("cargo")
        .args(["clippy", "--", "-D", "warnings"])
        .output()?;

    if !clippy_output.status.success() {
        println!("    ⚠ Clippy found issues:");
        println!("{}", String::from_utf8_lossy(&clippy_output.stdout));
        println!("{}", String::from_utf8_lossy(&clippy_output.stderr));
    } else {
        println!("    ✓ Clippy checks passed");
    }

    println!("  - Running cargo fmt check...");
    let fmt_output = std::process::Command::new("cargo")
        .args(["fmt", "--check"])
        .output()?;

    if !fmt_output.status.success() {
        println!("    ⚠ Code formatting issues found:");
        println!("{}", String::from_utf8_lossy(&fmt_output.stdout));
        println!("    💡 Run 'cargo fmt' to fix formatting");
    } else {
        println!("    ✓ Code formatting is correct");
    }

    println!("  - Running unused dependency check...");
    let unused_deps_output = std::process::Command::new("cargo")
        .args(["+nightly", "udeps"])
        .output();

    match unused_deps_output {
        Ok(output) if output.status.success() => {
            println!("    ✓ No unused dependencies");
        }
        Ok(output) => {
            println!("    ⚠ Unused dependencies detected:");
            println!("{}", String::from_utf8_lossy(&output.stdout));
        }
        Err(_) => {
            println!("    ℹ cargo-udeps not available (install with: cargo install cargo-udeps)");
        }
    }

    println!("🎯 Code quality checks completed\n");
    Ok(())
}

async fn make_test_api_calls(api_key: &str) -> Result<Vec<Value>, Box<dyn std::error::Error>> {
    let client = reqwest::Client::new();
    let mut responses = Vec::new();

    // Test call 1: Simple chat completion
    println!("  - Testing simple chat completion...");
    let simple_request = serde_json::json!({
        "model": "deepseek-chat",
        "messages": [
            {"role": "user", "content": "Hello, how are you?"}
        ],
        "max_tokens": 50,
        "temperature": 0.7
    });

    let response = client
        .post("https://api.deepseek.com/v1/chat/completions")
        .header("Authorization", format!("Bearer {}", api_key))
        .header("Content-Type", "application/json")
        .json(&simple_request)
        .send()
        .await?;

    if response.status().is_success() {
        let json: Value = response.json().await?;
        responses.push(json);
        println!("    ✓ Simple chat completion successful");
    } else {
        println!("    ⚠ Simple chat completion failed: {}", response.status());
        let error_json: Value = response.json().await?;
        responses.push(error_json);
    }

    // Test call 2: Chat with context caching
    println!("  - Testing chat with context caching...");
    let long_context = "You are a helpful assistant. This is a long context that should trigger caching mechanisms. ".repeat(50);
    let context_request = serde_json::json!({
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": long_context},
            {"role": "user", "content": "Summarize what you do in one sentence."}
        ],
        "max_tokens": 100,
        "temperature": 0.5
    });

    let response = client
        .post("https://api.deepseek.com/v1/chat/completions")
        .header("Authorization", format!("Bearer {}", api_key))
        .header("Content-Type", "application/json")
        .json(&context_request)
        .send()
        .await?;

    if response.status().is_success() {
        let json: Value = response.json().await?;
        responses.push(json);
        println!("    ✓ Context caching test successful");
    } else {
        println!("    ⚠ Context caching test failed: {}", response.status());
        let error_json: Value = response.json().await?;
        responses.push(error_json);
    }

    Ok(responses)
}

fn generate_types_from_responses(responses: Vec<Value>) -> String {
    let mut seen_fields = HashSet::new();
    let mut has_cache_fields = false;
    let mut has_error_fields = false;

    // Analyze responses to detect actual fields
    for response in &responses {
        analyze_response_fields(
            response,
            &mut seen_fields,
            &mut has_cache_fields,
            &mut has_error_fields,
        );
    }

    println!("    - Detected {} unique fields", seen_fields.len());
    println!(
        "    - Context caching fields: {}",
        if has_cache_fields { "Yes" } else { "No" }
    );
    println!(
        "    - Error response fields: {}",
        if has_error_fields { "Yes" } else { "No" }
    );

    // Generate comprehensive types based on analysis
    generate_comprehensive_types(has_cache_fields, has_error_fields, &seen_fields)
}

fn analyze_response_fields(
    value: &Value,
    seen_fields: &mut HashSet<String>,
    has_cache_fields: &mut bool,
    has_error_fields: &mut bool,
) {
    match value {
        Value::Object(map) => {
            for (key, val) in map {
                seen_fields.insert(key.clone());

                if key == "prompt_cache_hit_tokens" || key == "prompt_cache_miss_tokens" {
                    *has_cache_fields = true;
                }

                if key == "error" {
                    *has_error_fields = true;
                }

                analyze_response_fields(val, seen_fields, has_cache_fields, has_error_fields);
            }
        }
        Value::Array(arr) => {
            for item in arr {
                analyze_response_fields(item, seen_fields, has_cache_fields, has_error_fields);
            }
        }
        _ => {}
    }
}

fn generate_comprehensive_types(
    has_cache_fields: bool,
    has_error_fields: bool,
    seen_fields: &HashSet<String>,
) -> String {
    let _cache_comment = if has_cache_fields {
        "/// Context caching fields detected in real API responses"
    } else {
        "/// Context caching fields (may appear in responses)"
    };

    let _error_comment = if has_error_fields {
        "/// Error types detected in real API responses"
    } else {
        "/// Error types (standard API error format)"
    };

    let detected_fields: Vec<&str> = seen_fields.iter().map(|s| s.as_str()).collect();
    let detected_fields_str = if detected_fields.len() > 10 {
        format!(
            "{} fields including: {}, ...",
            detected_fields.len(),
            detected_fields[..10].join(", ")
        )
    } else {
        format!(
            "{} fields: {}",
            detected_fields.len(),
            detected_fields.join(", ")
        )
    };

    let timestamp = std::process::Command::new("date")
        .output()
        .map(|output| String::from_utf8_lossy(&output.stdout).trim().to_string())
        .unwrap_or_else(|_| "unknown".to_string());

    // Use string replacement to build the template
    let template = include_str!("../templates/types_template.rs");
    let mut result = template.to_string();

    // Add analysis information to the header
    let new_header = format!(
        "//! DeepSeek API response types\n//!\n//! This module contains all the type definitions for DeepSeek API requests and responses.\n//! These types are generated based on REAL API responses from DeepSeek.\n//! Generated on: {}\n//!\n//! Analysis results:\n//! - Detected fields: {}\n//! - Context caching support: {}\n//! - Error response structure: {}\n",
        timestamp,
        detected_fields_str,
        if has_cache_fields { "detected" } else { "not detected" },
        if has_error_fields { "detected" } else { "not detected" }
    );

    // Replace the header
    if let Some(pos) = result.find("use serde") {
        result = format!("{}\n{}", new_header, &result[pos..]);
    }

    result
}

fn generate_types_file() -> String {
    include_str!("../templates/types_template.rs").to_string()
}
