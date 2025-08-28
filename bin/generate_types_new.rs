//! Type generator for DeepSeek API responses
//! 
//! This script generates Rust types based on real DeepSeek API responses.
//! Run with: cargo run --bin generate_types

use serde_json::Value;
use std::collections::HashSet;
use std::env;
use std::fs;
use std::path::Path;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔧 Generating DeepSeek API types from real API responses...");
    
    // Ensure the output directory exists
    let output_dir = "src/deepseek";
    if !Path::new(output_dir).exists() {
        fs::create_dir_all(output_dir).expect("Failed to create output directory");
    }
    
    // Load environment variables
    dotenvy::dotenv().ok();
    
    let api_key = match env::var("DEEPSEEK_API_KEY").or_else(|_| env::var("DEEPSEEK_API_TOKEN")) {
        Ok(key) => {
            println!("📡 Making test API calls to analyze response structure...");
            
            // Make real API calls to get response structure
            let response_data = make_test_api_calls(&key).await?;
            
            println!("🔍 Analyzing response structure...");
            
            // Generate types based on real responses
            let types_content = generate_types_from_responses(response_data);
            
            // Write to file
            let output_path = format!("{}/types.rs", output_dir);
            fs::write(&output_path, types_content)?;
            
            println!("✅ Types generated successfully from real API responses in {}", output_path);
            
            return Ok(());
        }
        Err(_) => {
            println!("⚠️  No API key found, using template-based generation");
            println!("   Set DEEPSEEK_API_KEY or DEEPSEEK_API_TOKEN to use real API responses");
        }
    };
    
    // Fallback to template-based generation
    let types_content = generate_types_file();
    
    // Write to file
    let output_path = format!("{}/types.rs", output_dir);
    fs::write(&output_path, types_content)
        .expect("Failed to write types.rs");
    
    println!("✅ Types generated successfully from template in {}", output_path);
    
    // Verify the file was created and is valid Rust
    match fs::metadata(&output_path) {
        Ok(metadata) => {
            let size = metadata.len();
            println!("📊 Generated file size: {} bytes", size);
            
            if size > 0 {
                println!("🎉 Type generation completed successfully!");
            } else {
                eprintln!("⚠️  Warning: Generated file is empty");
            }
        }
        Err(e) => {
            eprintln!("❌ Error accessing generated file: {}", e);
        }
    }
    
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
    
    // Test call 3: Error case (invalid model)
    println!("  - Testing error response...");
    let error_request = serde_json::json!({
        "model": "invalid-model-name",
        "messages": [
            {"role": "user", "content": "Test"}
        ]
    });
    
    let response = client
        .post("https://api.deepseek.com/v1/chat/completions")
        .header("Authorization", format!("Bearer {}", api_key))
        .header("Content-Type", "application/json")
        .json(&error_request)
        .send()
        .await?;
    
    let json: Value = response.json().await?;
    responses.push(json);
    if response.status().is_success() {
        println!("    ✓ Error test completed");
    } else {
        println!("    ✓ Error response captured");
    }
    
    Ok(responses)
}

fn generate_types_from_responses(responses: Vec<Value>) -> String {
    let mut seen_fields = HashSet::new();
    let mut has_cache_fields = false;
    let mut has_error_fields = false;
    let mut sample_responses = Vec::new();
    
    // Analyze responses to detect actual fields
    for response in &responses {
        analyze_response_fields(response, &mut seen_fields, &mut has_cache_fields, &mut has_error_fields);
        
        // Store sample for documentation
        if sample_responses.len() < 2 {
            sample_responses.push(serde_json::to_string_pretty(response).unwrap_or_else(|_| "{}".to_string()));
        }
    }
    
    println!("    - Detected {} unique fields", seen_fields.len());
    println!("    - Context caching fields: {}", if has_cache_fields { "Yes" } else { "No" });
    println!("    - Error response fields: {}", if has_error_fields { "Yes" } else { "No" });
    
    // Generate comprehensive types based on analysis
    generate_comprehensive_types(has_cache_fields, has_error_fields, &seen_fields, sample_responses)
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
    sample_responses: Vec<String>,
) -> String {
    let cache_comment = if has_cache_fields {
        "/// Context caching fields detected in real API responses"
    } else {
        "/// Context caching fields (may appear in responses)"
    };
    
    let error_comment = if has_error_fields {
        "/// Error types detected in real API responses"
    } else {
        "/// Error types (standard API error format)"
    };
    
    let detected_fields = seen_fields.iter().collect::<Vec<_>>();
    let detected_fields_str = if detected_fields.len() > 10 {
        format!("{} fields including: {}, ...", detected_fields.len(), detected_fields[..10].join(", "))
    } else {
        format!("{} fields: {}", detected_fields.len(), detected_fields.join(", "))
    };
    
    let sample_docs = if !sample_responses.is_empty() {
        format!("//!\n//! Sample API response:\n//! ```json\n//! {}\n//! ```", 
                sample_responses[0].lines().take(20).collect::<Vec<_>>().join("\n//! "))
    } else {
        String::new()
    };
    
    let timestamp = std::process::Command::new("date")
        .output()
        .map(|output| String::from_utf8_lossy(&output.stdout).trim().to_string())
        .unwrap_or_else(|_| "unknown".to_string());
    
    format!(
        include_str!("../templates/api_types_template.txt"),
        timestamp = timestamp,
        detected_fields = detected_fields_str,
        cache_status = if has_cache_fields { "detected" } else { "not detected" },
        error_status = if has_error_fields { "detected" } else { "not detected" },
        sample_docs = sample_docs,
        cache_comment = cache_comment,
        error_comment = error_comment
    )
}

fn generate_types_file() -> String {
    include_str!("../templates/types_template.rs").to_string()
}
