//! MCP (Model Context Protocol) Server for Rig Analyzer
//!
//! This module implements an MCP server using rmcp with DeepSeek and FastEmbed integration
//! for code analysis and vector similarity search.

use crate::Config;
use rmcp::ServiceExt;
use std::future::Future;
use std::path::Path;
use std::sync::Arc;
use tokio::fs as async_fs;

use rig::{
    client::{CompletionClient, ProviderClient},
    completion::Prompt,
    providers::deepseek,
};
use rmcp::{
    handler::server::{router::tool::ToolRouter, tool::Parameters},
    model::*,
    schemars,
    service::RequestContext,
    tool, tool_handler, tool_router, RoleServer, ServerHandler,
};
use serde_json::json;
use tokio::sync::Mutex;

use hyper_util::{
    rt::{TokioExecutor, TokioIo},
    server::conn::auto::Builder,
    service::TowerToHyperService,
};
use rmcp::transport::streamable_http_server::{
    session::local::LocalSessionManager, StreamableHttpService,
};

use crate::deepseek::DeepSeekClient;
use crate::vector_store::VectorStoreManager;
use anyhow::Result;

#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
pub struct SimilaritySearchRequest {
    pub query: String,
    pub limit: Option<usize>,
}

#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
pub struct CodeAnalysisRequest {
    pub code: String,
    pub language: Option<String>,
}

#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
pub struct BugConfirmationRequest {
    pub bug_description: String,
    pub code_sample: String,
}

#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
#[allow(dead_code)]
pub struct AdapterAnalysisRequest {
    pub adapter_name: String,
    pub adapter_path: Option<String>,
    pub analysis_type: String, // "security", "performance", "compatibility"
}

#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
pub struct BugStoreRequest {
    pub bug_id: String,
    pub severity: String,
    pub description: String,
    pub adapter_name: Option<String>,
    pub code_sample: Option<String>,
    pub fix_suggestion: Option<String>,
}

#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
pub struct BugListRequest {
    pub severity_filter: Option<String>,
    pub adapter_filter: Option<String>,
    pub include_file_details: Option<bool>,
}

#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
pub struct BugDetailRequest {
    pub bug_id: String,
}

#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
pub struct FileReadRequest {
    pub file_path: String,
}

#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
pub struct FileWriteRequest {
    pub file_path: String,
    pub content: String,
}

#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
pub struct DirectoryListRequest {
    pub directory_path: String,
}

#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
pub struct AdapterFileRequest {
    pub adapter_name: String,
}

#[derive(Clone)]
pub struct NautilusMcpServer {
    pub vector_store: Arc<Mutex<Option<VectorStoreManager>>>,
    pub deepseek_client: Arc<Mutex<Option<DeepSeekClient>>>,
    tool_router: ToolRouter<NautilusMcpServer>,
}

impl Default for NautilusMcpServer {
    fn default() -> Self {
        Self::new()
    }
}

#[tool_router]
impl NautilusMcpServer {
    pub fn new() -> Self {
        Self {
            vector_store: Arc::new(Mutex::new(None)),
            deepseek_client: Arc::new(Mutex::new(None)),
            tool_router: Self::tool_router(),
        }
    }

    pub async fn initialize_services(&self) -> Result<()> {
        // Initialize vector store with FastEmbed
        if let Ok(vector_store) = VectorStoreManager::new().await {
            let mut vs = self.vector_store.lock().await;
            *vs = Some(vector_store);
            tracing::info!("✅ Vector store initialized with FastEmbed");
        } else {
            tracing::warn!("⚠️ Failed to initialize vector store");
        }

        // Initialize DeepSeek client
        if let Ok(deepseek_client) = DeepSeekClient::from_env() {
            let mut ds = self.deepseek_client.lock().await;
            *ds = Some(deepseek_client);
            tracing::info!("✅ DeepSeek client initialized");
        } else {
            tracing::warn!("⚠️ Failed to initialize DeepSeek client (DEEPSEEK_API_KEY required)");
        }

        Ok(())
    }

    fn _create_resource_text(&self, uri: &str, name: &str) -> Resource {
        RawResource::new(uri, name.to_string()).no_annotation()
    }

    #[tool(description = "Search for similar bug patterns using FastEmbed vector embeddings")]
    async fn similarity_search(
        &self,
        Parameters(SimilaritySearchRequest { query, limit }): Parameters<SimilaritySearchRequest>,
    ) -> Result<CallToolResult, ErrorData> {
        let search_limit = limit.unwrap_or(5);

        let vector_store = self.vector_store.lock().await;
        if let Some(vs) = vector_store.as_ref() {
            match vs.similarity_search(&query, search_limit).await {
                Ok(results) => {
                    let response = if results.is_empty() {
                        format!("No similar bug patterns found for query: '{}'", query)
                    } else {
                        format!(
                            "Found {} similar bug patterns for query '{}':\n{}",
                            results.len(),
                            query,
                            serde_json::to_string_pretty(&results).unwrap_or_default()
                        )
                    };

                    Ok(CallToolResult::success(vec![Content::text(response)]))
                }
                Err(e) => Ok(CallToolResult::success(vec![Content::text(format!(
                    "Error performing vector similarity search: {}",
                    e
                ))])),
            }
        } else {
            Ok(CallToolResult::success(vec![Content::text(
                "Vector store not available".to_string(),
            )]))
        }
    }

    #[tool(description = "Analyze code for security vulnerabilities and issues using DeepSeek AI")]
    async fn analyze_code(
        &self,
        Parameters(CodeAnalysisRequest { code, language }): Parameters<CodeAnalysisRequest>,
    ) -> Result<CallToolResult, ErrorData> {
        let deepseek_client = self.deepseek_client.lock().await;
        if let Some(client) = deepseek_client.as_ref() {
            let lang = language.unwrap_or_else(|| "unknown".to_string());
            let prompt = format!(
                "Analyze this {} code for security vulnerabilities, performance issues, and potential bugs:\n\n```{}\n{}\n```\n\nProvide a detailed analysis including:\n1. Security vulnerabilities\n2. Performance issues\n3. Code quality problems\n4. Recommended fixes",
                lang, lang, code
            );

            match client.analyze_code(&prompt).await {
                Ok(analysis) => Ok(CallToolResult::success(vec![Content::text(format!(
                    "🔍 DeepSeek Code Analysis:\n\n{}",
                    analysis
                ))])),
                Err(e) => Ok(CallToolResult::success(vec![Content::text(format!(
                    "❌ Code analysis failed: {}",
                    e
                ))])),
            }
        } else {
            Ok(CallToolResult::success(vec![Content::text(
                "⚠️ DeepSeek client not available (API key not set)".to_string(),
            )]))
        }
    }

    #[tool(description = "Confirm if a bug is critical using DeepSeek AI analysis")]
    async fn confirm_critical_bug(
        &self,
        Parameters(BugConfirmationRequest {
            bug_description,
            code_sample,
        }): Parameters<BugConfirmationRequest>,
    ) -> Result<CallToolResult, ErrorData> {
        let deepseek_client = self.deepseek_client.lock().await;
        if let Some(client) = deepseek_client.as_ref() {
            match client
                .confirm_critical_bug(&bug_description, &code_sample)
                .await
            {
                Ok(analysis) => Ok(CallToolResult::success(vec![Content::text(format!(
                    "🤖 DeepSeek Critical Bug Analysis:\n\n{}",
                    analysis
                ))])),
                Err(e) => Ok(CallToolResult::success(vec![Content::text(format!(
                    "❌ Bug confirmation failed: {}",
                    e
                ))])),
            }
        } else {
            Ok(CallToolResult::success(vec![Content::text(
                "⚠️ DeepSeek client not available (API key not set)".to_string(),
            )]))
        }
    }

    #[tool(description = "Get server status and available capabilities")]
    async fn get_status(&self) -> Result<CallToolResult, ErrorData> {
        let vector_store_status = {
            let vs = self.vector_store.lock().await;
            if vs.is_some() {
                "✅ Available (FastEmbed)"
            } else {
                "❌ Not available"
            }
        };

        let deepseek_status = {
            let ds = self.deepseek_client.lock().await;
            if ds.is_some() {
                "✅ Available"
            } else {
                "❌ Not available"
            }
        };

        let status = format!(
            "🔧 Rig Analyzer MCP Server Status:\n\
             📊 Vector Store: {}\n\
             🤖 DeepSeek Client: {}\n\
             🛠️ Available Tools:\n\
             - similarity_search: Search bug patterns with FastEmbed\n\
             - analyze_code: Analyze code with DeepSeek AI\n\
             - confirm_critical_bug: Validate critical bugs\n\
             - get_status: Get server status\n\
             - read_file: Read file contents\n\
             - write_file: Write content to file\n\
             - list_directory: List directory contents\n\
             - read_adapter: Read adapter source files\n\
             - store_bug: Store bug analysis to JSON file\n\
             - list_bugs: List all stored bugs with file locations and metadata\n\
             - get_bug_details: Get detailed information about a specific bug",
            vector_store_status, deepseek_status
        );

        Ok(CallToolResult::success(vec![Content::text(status)]))
    }

    #[tool(description = "Read contents of a file")]
    async fn read_file(
        &self,
        Parameters(FileReadRequest { file_path }): Parameters<FileReadRequest>,
    ) -> Result<CallToolResult, ErrorData> {
        // Security check: only allow access to certain directories
        let allowed_paths = [
            "nautilus_trader/adapters/",
            "nautilus-trader-rig/",
            "examples/",
            "crates/adapters/",
        ];

        let is_allowed = allowed_paths
            .iter()
            .any(|allowed| file_path.starts_with(allowed));

        if !is_allowed {
            return Ok(CallToolResult::success(vec![Content::text(format!(
                "❌ Access denied: File path '{}' is not in allowed directories",
                file_path
            ))]));
        }

        match async_fs::read_to_string(&file_path).await {
            Ok(content) => Ok(CallToolResult::success(vec![Content::text(format!(
                "📄 File: {}\n\n```\n{}\n```",
                file_path, content
            ))])),
            Err(e) => Ok(CallToolResult::success(vec![Content::text(format!(
                "❌ Failed to read file '{}': {}",
                file_path, e
            ))])),
        }
    }

    #[tool(description = "Write content to a file")]
    async fn write_file(
        &self,
        Parameters(FileWriteRequest { file_path, content }): Parameters<FileWriteRequest>,
    ) -> Result<CallToolResult, ErrorData> {
        // Security check: only allow writing to bugs directory and certain areas
    let allowed_write_paths = [Config::BUGS_DIRECTORY, "nautilus-trader-rig/logs/"];

        let is_allowed = allowed_write_paths
            .iter()
            .any(|allowed| file_path.starts_with(allowed));

        if !is_allowed {
            return Ok(CallToolResult::success(vec![Content::text(format!(
                "❌ Write access denied: File path '{}' is not in allowed write directories",
                file_path
            ))]));
        }

        // Ensure parent directory exists
        if let Some(parent) = Path::new(&file_path).parent() {
            if let Err(e) = async_fs::create_dir_all(parent).await {
                return Ok(CallToolResult::success(vec![Content::text(format!(
                    "❌ Failed to create directory '{}': {}",
                    parent.display(),
                    e
                ))]));
            }
        }

        match async_fs::write(&file_path, &content).await {
            Ok(_) => Ok(CallToolResult::success(vec![Content::text(format!(
                "✅ Successfully wrote to file: {}",
                file_path
            ))])),
            Err(e) => Ok(CallToolResult::success(vec![Content::text(format!(
                "❌ Failed to write file '{}': {}",
                file_path, e
            ))])),
        }
    }

    #[tool(description = "List contents of a directory")]
    async fn list_directory(
        &self,
        Parameters(DirectoryListRequest { directory_path }): Parameters<DirectoryListRequest>,
    ) -> Result<CallToolResult, ErrorData> {
        // Security check: only allow access to certain directories
        let allowed_paths = [
            "nautilus_trader/adapters/",
            "nautilus-trader-rig/",
            "examples/",
            "crates/adapters/",
        ];

        let is_allowed = allowed_paths
            .iter()
            .any(|allowed| directory_path.starts_with(allowed));

        if !is_allowed {
            return Ok(CallToolResult::success(vec![Content::text(format!(
                "❌ Access denied: Directory path '{}' is not in allowed directories",
                directory_path
            ))]));
        }

        match async_fs::read_dir(&directory_path).await {
            Ok(mut entries) => {
                let mut files = Vec::new();
                let mut dirs = Vec::new();

                while let Ok(Some(entry)) = entries.next_entry().await {
                    let path = entry.path();
                    let name = path
                        .file_name()
                        .unwrap_or_default()
                        .to_string_lossy()
                        .to_string();

                    if path.is_dir() {
                        dirs.push(format!("📁 {}/", name));
                    } else {
                        files.push(format!("📄 {}", name));
                    }
                }

                dirs.sort();
                files.sort();

                let mut contents = dirs;
                contents.extend(files);

                let result = if contents.is_empty() {
                    format!("📂 Directory '{}' is empty", directory_path)
                } else {
                    format!(
                        "📂 Directory '{}' contents:\n{}",
                        directory_path,
                        contents.join("\n")
                    )
                };

                Ok(CallToolResult::success(vec![Content::text(result)]))
            }
            Err(e) => Ok(CallToolResult::success(vec![Content::text(format!(
                "❌ Failed to list directory '{}': {}",
                directory_path, e
            ))])),
        }
    }

    #[tool(description = "Read Rust adapter source files from configured adapters directory")]
    async fn read_adapter(
        &self,
        Parameters(AdapterFileRequest { adapter_name }): Parameters<AdapterFileRequest>,
    ) -> Result<CallToolResult, ErrorData> {
        tracing::info!(
            "🌐 MCP read_adapter: Processing request for adapter: {}",
            adapter_name
        );

        // Try multiple Rust adapter directories
        let rust_directories = Config::all_rust_adapter_directories();
        let mut found_files = Vec::new();
        let mut processed_files = Vec::new();

        for rust_dir in rust_directories {
            let adapter_path = rust_dir.join(adapter_name.to_lowercase());
            tracing::debug!("📁 Checking adapter directory: {:?}", adapter_path);

            if adapter_path.exists() {
                tracing::info!("📁 Found adapter directory: {:?}", adapter_path);
                match async_fs::read_dir(&adapter_path).await {
                    Ok(mut entries) => {
                        while let Ok(Some(entry)) = entries.next_entry().await {
                            let path = entry.path();
                            let file_name = path.file_name().unwrap_or_default().to_string_lossy();
                            tracing::debug!("📄 Examining file: {}", file_name);

                            if let Some(ext) = path.extension() {
                                // Use configured Rust extensions
                                let ext_str = ext.to_string_lossy();
                                if Config::rust_extensions().iter().any(|&allowed_ext| {
                                    ext_str == allowed_ext.trim_start_matches('.')
                                }) {
                                    tracing::info!("📄 Processing Rust file: {:?}", path);
                                    processed_files.push(path.display().to_string());

                                    if let Ok(content) = async_fs::read_to_string(&path).await {
                                        tracing::debug!(
                                            "✅ Successfully read file: {} ({} bytes)",
                                            file_name,
                                            content.len()
                                        );
                                        found_files.push(format!(
                                            "📄 File: {}\n```rust\n{}\n```\n",
                                            path.display(),
                                            content
                                        ));
                                    } else {
                                        tracing::warn!(
                                            "❌ Failed to read file content: {}",
                                            file_name
                                        );
                                    }
                                } else {
                                    tracing::debug!("⏭️ Skipping non-Rust file: {}", file_name);
                                }
                            } else {
                                tracing::debug!(
                                    "⏭️ Skipping file without extension: {}",
                                    file_name
                                );
                            }
                        }
                    }
                    Err(e) => {
                        tracing::warn!("❌ Failed to read directory {:?}: {}", adapter_path, e);
                        continue; // Try next directory
                    }
                }
            } else {
                tracing::debug!("❌ Adapter directory not found: {:?}", adapter_path);
            }
        }

        tracing::info!(
            "📊 Summary: Found {} Rust files for adapter '{}'",
            found_files.len(),
            adapter_name
        );
        for file in &processed_files {
            tracing::info!("   📄 {}", file);
        }

        if !found_files.is_empty() {
            return Ok(CallToolResult::success(vec![Content::text(format!(
                "🦀 Rust Adapter '{}' source files:\n\n{}",
                adapter_name,
                found_files.join("\n")
            ))]));
        }

        Ok(CallToolResult::success(vec![Content::text(format!(
            "❌ No Rust adapter files found for '{}'",
            adapter_name
        ))]))
    }

    #[tool(description = "List all stored bugs with exact file locations and detailed metadata")]
    async fn list_bugs(
        &self,
        Parameters(BugListRequest {
            severity_filter,
            adapter_filter,
            include_file_details,
        }): Parameters<BugListRequest>,
    ) -> Result<CallToolResult, ErrorData> {
        let bugs_dir = std::path::Path::new(Config::BUGS_DIRECTORY);
        let include_details = include_file_details.unwrap_or(true);

        if !bugs_dir.exists() {
            return Ok(CallToolResult::success(vec![Content::text(
                "📁 No bugs directory found. No bugs have been stored yet.".to_string(),
            )]));
        }

        let mut bug_files = Vec::new();
        let mut summary_files = Vec::new();

        // Read all JSON files in bugs directory
        if let Ok(entries) = async_fs::read_dir(bugs_dir).await {
            let mut entries = entries;
            while let Ok(Some(entry)) = entries.next_entry().await {
                let path = entry.path();
                if let Some(extension) = path.extension() {
                    if extension == "json" {
                        let filename = path.file_name().unwrap_or_default().to_string_lossy();
                        if filename.starts_with("analysis_summary_") {
                            summary_files.push(path);
                        } else if filename.starts_with("AUTO_BUG_") || filename.contains("_BUG_") {
                            bug_files.push(path);
                        }
                    }
                }
            }
        }

        if bug_files.is_empty() && summary_files.is_empty() {
            return Ok(CallToolResult::success(vec![Content::text(
                "📭 No bug reports found in the bugs directory.".to_string(),
            )]));
        }

        let mut response_parts = Vec::new();
        let mut matching_bugs = Vec::new();

        // Process individual bug files
        for bug_file in &bug_files {
            if let Ok(content) = async_fs::read_to_string(bug_file).await {
                if let Ok(bug_data) = serde_json::from_str::<serde_json::Value>(&content) {
                    // Apply filters
                    let mut include_bug = true;

                    if let Some(ref sev_filter) = severity_filter {
                        if let Some(severity) = bug_data.get("severity").and_then(|s| s.as_str()) {
                            if !severity.to_lowercase().contains(&sev_filter.to_lowercase()) {
                                include_bug = false;
                            }
                        }
                    }

                    if let Some(ref adapter_filter) = adapter_filter {
                        if let Some(adapter) = bug_data.get("adapter_name").and_then(|a| a.as_str())
                        {
                            if !adapter
                                .to_lowercase()
                                .contains(&adapter_filter.to_lowercase())
                            {
                                include_bug = false;
                            }
                        }
                    }

                    if include_bug {
                        matching_bugs.push((bug_file.clone(), bug_data));
                    }
                }
            }
        }

        if matching_bugs.is_empty() {
            response_parts.push("🔍 No bugs match the specified filters.".to_string());
        } else {
            response_parts.push(format!("🐛 Found {} matching bugs:\n", matching_bugs.len()));

            for (i, (bug_file, bug_data)) in matching_bugs.iter().enumerate() {
                let bug_id = bug_data
                    .get("bug_id")
                    .and_then(|s| s.as_str())
                    .unwrap_or("unknown");
                let severity = bug_data
                    .get("severity")
                    .and_then(|s| s.as_str())
                    .unwrap_or("unknown");
                let description = bug_data
                    .get("description")
                    .and_then(|s| s.as_str())
                    .unwrap_or("No description");
                let adapter = bug_data
                    .get("adapter_name")
                    .and_then(|s| s.as_str())
                    .unwrap_or("unknown");
                let timestamp = bug_data
                    .get("timestamp")
                    .and_then(|s| s.as_str())
                    .unwrap_or("unknown");

                response_parts.push(format!(
                    "\n{}. 🆔 Bug ID: {}\n   ⚠️ Severity: {}\n   📦 Adapter: {}\n   ⏰ Timestamp: {}\n   📄 File: {}",
                    i + 1, bug_id, severity, adapter, timestamp, bug_file.display()
                ));

                if include_details {
                    // Include file location details if available
                    if let Some(file_location) = bug_data.get("file_location") {
                        response_parts.push("   📍 File Location Details:".to_string());

                        if let Some(details) = file_location.get("details") {
                            if let Some(abs_path) =
                                details.get("absolute_path").and_then(|s| s.as_str())
                            {
                                response_parts
                                    .push(format!("      📂 Absolute Path: {}", abs_path));
                            }
                            if let Some(rel_path) =
                                details.get("relative_path").and_then(|s| s.as_str())
                            {
                                response_parts
                                    .push(format!("      📁 Relative Path: {}", rel_path));
                            }
                            if let Some(filename) = details.get("filename").and_then(|s| s.as_str())
                            {
                                response_parts.push(format!("      📄 Filename: {}", filename));
                            }
                            if let Some(size) =
                                details.get("file_size_bytes").and_then(|s| s.as_u64())
                            {
                                response_parts.push(format!("      📏 File Size: {} bytes", size));
                            }
                        }

                        if let Some(source_loc) = file_location.get("source_location") {
                            if let Some(line_num) = source_loc
                                .get("approximate_line_number")
                                .and_then(|s| s.as_u64())
                            {
                                response_parts
                                    .push(format!("      📍 Approximate Line: {}", line_num));
                            }
                        }
                    }

                    // Include workspace info if available
                    if let Some(workspace) = bug_data.get("workspace_info") {
                        if let Some(branch) = workspace.get("branch").and_then(|s| s.as_str()) {
                            response_parts.push(format!("      🌿 Git Branch: {}", branch));
                        }
                        if let Some(commit) = workspace.get("commit_hash").and_then(|s| s.as_str())
                        {
                            response_parts.push(format!("      🔗 Commit: {}", &commit[..8]));
                        }
                    }
                }

                response_parts.push(format!(
                    "   📝 Description: {}",
                    if description.len() > 100 {
                        format!("{}...", &description[..100])
                    } else {
                        description.to_string()
                    }
                ));
            }
        }

        // Add summary information about analysis runs
        if !summary_files.is_empty() {
            response_parts.push(format!(
                "\n📊 Analysis Summary Files: {} found",
                summary_files.len()
            ));

            // Show the most recent summary
            if let Some(latest_summary) = summary_files.last() {
                if let Ok(content) = async_fs::read_to_string(latest_summary).await {
                    if let Ok(summary_data) = serde_json::from_str::<serde_json::Value>(&content) {
                        if let Some(summary) = summary_data.get("analysis_summary") {
                            response_parts.push("\n📋 Latest Analysis Summary:".to_string());
                            if let Some(total) = summary
                                .get("total_files_discovered")
                                .and_then(|s| s.as_u64())
                            {
                                response_parts
                                    .push(format!("   📁 Total Files Discovered: {}", total));
                            }
                            if let Some(analyzed) =
                                summary.get("files_analyzed").and_then(|s| s.as_u64())
                            {
                                response_parts.push(format!("   🔍 Files Analyzed: {}", analyzed));
                            }
                            if let Some(bugs) = summary.get("bugs_found").and_then(|s| s.as_u64()) {
                                response_parts.push(format!("   🐛 Bugs Found: {}", bugs));
                            }
                            if let Some(timestamp) =
                                summary.get("analysis_timestamp").and_then(|s| s.as_str())
                            {
                                response_parts.push(format!("   ⏰ Analysis Time: {}", timestamp));
                            }
                        }
                    }
                }
            }
        }

        Ok(CallToolResult::success(vec![Content::text(
            response_parts.join("\n"),
        )]))
    }

    #[tool(
        description = "Get detailed information about a specific bug including exact file location and fix suggestions"
    )]
    async fn get_bug_details(
        &self,
        Parameters(BugDetailRequest { bug_id }): Parameters<BugDetailRequest>,
    ) -> Result<CallToolResult, ErrorData> {
        let bugs_dir = std::path::Path::new(Config::BUGS_DIRECTORY);

        if !bugs_dir.exists() {
            return Ok(CallToolResult::success(vec![Content::text(
                "📁 No bugs directory found.".to_string(),
            )]));
        }

        // Search for bug file by ID
        let mut found_bug_file = None;
        if let Ok(entries) = async_fs::read_dir(bugs_dir).await {
            let mut entries = entries;
            while let Ok(Some(entry)) = entries.next_entry().await {
                let path = entry.path();
                if let Some(extension) = path.extension() {
                    if extension == "json" {
                        let filename = path.file_name().unwrap_or_default().to_string_lossy();
                        if filename.contains(&bug_id) {
                            found_bug_file = Some(path);
                            break;
                        }
                    }
                }
            }
        }

        let Some(bug_file) = found_bug_file else {
            return Ok(CallToolResult::success(vec![Content::text(format!(
                "❌ Bug with ID '{}' not found.",
                bug_id
            ))]));
        };

        let content = match async_fs::read_to_string(&bug_file).await {
            Ok(content) => content,
            Err(e) => {
                return Ok(CallToolResult::success(vec![Content::text(format!(
                    "❌ Failed to read bug file: {}",
                    e
                ))]));
            }
        };

        let bug_data: serde_json::Value = match serde_json::from_str(&content) {
            Ok(data) => data,
            Err(e) => {
                return Ok(CallToolResult::success(vec![Content::text(format!(
                    "❌ Failed to parse bug file: {}",
                    e
                ))]));
            }
        };

        let mut response_parts = Vec::new();

        // Basic bug information
        response_parts.push(format!("🐛 Bug Details for ID: {}", bug_id));
        response_parts.push("=".repeat(50));

        if let Some(severity) = bug_data.get("severity").and_then(|s| s.as_str()) {
            response_parts.push(format!("⚠️  Severity: {}", severity));
        }

        if let Some(adapter) = bug_data.get("adapter_name").and_then(|s| s.as_str()) {
            response_parts.push(format!("📦 Adapter: {}", adapter));
        }

        if let Some(timestamp) = bug_data.get("timestamp").and_then(|s| s.as_str()) {
            response_parts.push(format!("⏰ Timestamp: {}", timestamp));
        }

        if let Some(context) = bug_data.get("analysis_context").and_then(|s| s.as_str()) {
            response_parts.push(format!("🔍 Analysis Context: {}", context));
        }

        response_parts.push("".to_string());

        // Description
        if let Some(description) = bug_data.get("description").and_then(|s| s.as_str()) {
            response_parts.push("📝 Description:".to_string());
            response_parts.push(format!("   {}", description));
            response_parts.push("".to_string());
        }

        // File location details
        if let Some(file_location) = bug_data.get("file_location") {
            response_parts.push("📍 File Location:".to_string());

            if let Some(details) = file_location.get("details") {
                if let Some(abs_path) = details.get("absolute_path").and_then(|s| s.as_str()) {
                    response_parts.push(format!("   📂 Absolute Path: {}", abs_path));
                }
                if let Some(rel_path) = details.get("relative_path").and_then(|s| s.as_str()) {
                    response_parts.push(format!("   📁 Relative Path: {}", rel_path));
                }
                if let Some(filename) = details.get("filename").and_then(|s| s.as_str()) {
                    response_parts.push(format!("   📄 Filename: {}", filename));
                }
                if let Some(directory) = details.get("directory").and_then(|s| s.as_str()) {
                    response_parts.push(format!("   📂 Directory: {}", directory));
                }
                if let Some(size) = details.get("file_size_bytes").and_then(|s| s.as_u64()) {
                    response_parts.push(format!("   📏 File Size: {} bytes", size));
                }
                if let Some(modified) = details
                    .get("last_modified_timestamp")
                    .and_then(|s| s.as_u64())
                {
                    response_parts.push(format!(
                        "   🕒 Last Modified: {} (Unix timestamp)",
                        modified
                    ));
                }
            }

            if let Some(source_loc) = file_location.get("source_location") {
                if let Some(line_num) = source_loc
                    .get("approximate_line_number")
                    .and_then(|s| s.as_u64())
                {
                    response_parts.push(format!("   📍 Approximate Line Number: {}", line_num));
                }
                if let Some(method) = source_loc
                    .get("context_extraction_method")
                    .and_then(|s| s.as_str())
                {
                    response_parts.push(format!("   🔍 Location Method: {}", method));
                }
            }

            response_parts.push("".to_string());
        }

        // Workspace information
        if let Some(workspace) = bug_data.get("workspace_info") {
            response_parts.push("🌿 Workspace Information:".to_string());
            if let Some(repo) = workspace.get("repository").and_then(|s| s.as_str()) {
                response_parts.push(format!("   📚 Repository: {}", repo));
            }
            if let Some(branch) = workspace.get("branch").and_then(|s| s.as_str()) {
                response_parts.push(format!("   🌿 Git Branch: {}", branch));
            }
            if let Some(commit) = workspace.get("commit_hash").and_then(|s| s.as_str()) {
                response_parts.push(format!("   🔗 Commit Hash: {}", commit));
            }
            response_parts.push("".to_string());
        }

        // Code sample
        if let Some(code_sample) = bug_data.get("code_sample").and_then(|s| s.as_str()) {
            if !code_sample.is_empty() && code_sample != "See file content" {
                response_parts.push("💻 Code Sample:".to_string());
                response_parts.push("```rust".to_string());
                response_parts.push(code_sample.to_string());
                response_parts.push("```".to_string());
                response_parts.push("".to_string());
            }
        }

        // Fix suggestion
        if let Some(fix_suggestion) = bug_data.get("fix_suggestion").and_then(|s| s.as_str()) {
            if !fix_suggestion.is_empty() && fix_suggestion != "Manual review required" {
                response_parts.push("🔧 Fix Suggestion:".to_string());
                response_parts.push(format!("   {}", fix_suggestion));
                response_parts.push("".to_string());
            }
        }

        // Additional metadata
        if let Some(metadata) = bug_data.get("mcp_metadata") {
            response_parts.push("🔧 Technical Metadata:".to_string());
            if let Some(version) = metadata.get("tool_version").and_then(|s| s.as_str()) {
                response_parts.push(format!("   🛠️ Tool Version: {}", version));
            }
            if let Some(method) = metadata.get("submission_method").and_then(|s| s.as_str()) {
                response_parts.push(format!("   📤 Submission Method: {}", method));
            }
        }

        response_parts.push("".to_string());
        response_parts.push(format!("📄 Bug file location: {}", bug_file.display()));

        Ok(CallToolResult::success(vec![Content::text(
            response_parts.join("\n"),
        )]))
    }

    #[tool(
        description = "Store bug analysis as JSON file in bugs directory with enhanced file location tracking"
    )]
    async fn store_bug(
        &self,
        Parameters(BugStoreRequest {
            bug_id,
            severity,
            description,
            adapter_name,
            code_sample,
            fix_suggestion,
        }): Parameters<BugStoreRequest>,
    ) -> Result<CallToolResult, ErrorData> {
        let timestamp = chrono::Utc::now().format("%Y%m%d_%H%M%S").to_string();
        let adapter_suffix = adapter_name
            .as_ref()
            .map(|s| format!("_{}", s))
            .unwrap_or_default();
        let filename = format!(
            "{}/{}{}_{}.json",
            Config::BUGS_DIRECTORY,
            bug_id,
            adapter_suffix,
            timestamp
        );

        // Enhanced file location tracking (similar to store_bug_internal)
        let mut workspace_info = serde_json::Map::new();
        workspace_info.insert(
            "repository".to_string(),
            serde_json::Value::String("nautilus_trader".to_string()),
        );

        // Try to get git information
        if let Ok(output) = tokio::process::Command::new("git")
            .args(["branch", "--show-current"])
            .output()
            .await
        {
            if output.status.success() {
                if let Ok(branch) = String::from_utf8(output.stdout) {
                    workspace_info.insert(
                        "branch".to_string(),
                        serde_json::Value::String(branch.trim().to_string()),
                    );
                }
            }
        }

        if let Ok(output) = tokio::process::Command::new("git")
            .args(["rev-parse", "HEAD"])
            .output()
            .await
        {
            if output.status.success() {
                if let Ok(commit) = String::from_utf8(output.stdout) {
                    workspace_info.insert(
                        "commit_hash".to_string(),
                        serde_json::Value::String(commit.trim().to_string()),
                    );
                }
            }
        }

        let bug_data = serde_json::json!({
            "bug_id": bug_id,
            "severity": severity,
            "description": description,
            "adapter_name": adapter_name,
            "code_sample": code_sample,
            "fix_suggestion": fix_suggestion,
            "timestamp": timestamp,
            "analysis_context": "Submitted via MCP interface",
            "workspace_info": workspace_info,
            "mcp_metadata": {
                "tool_version": "nautilus-trader-rig-mcp",
                "submission_method": "store_bug_tool",
                "enhanced_tracking": true
            }
        });

        match async_fs::write(&filename, serde_json::to_string_pretty(&bug_data).unwrap()).await {
            Ok(_) => {
                let response_msg = format!(
                    "✅ Bug stored successfully: {}\n📊 Enhanced tracking enabled with workspace metadata\n🔍 Bug ID: {}\n⚠️ Severity: {}",
                    filename, bug_id, severity
                );
                Ok(CallToolResult::success(vec![Content::text(response_msg)]))
            }
            Err(e) => Ok(CallToolResult::success(vec![Content::text(format!(
                "❌ Failed to store bug: {}",
                e
            ))])),
        }
    }
}

#[tool_handler]
impl ServerHandler for NautilusMcpServer {
    fn get_info(&self) -> ServerInfo {
        ServerInfo {
            protocol_version: ProtocolVersion::V_2024_11_05,
            capabilities: ServerCapabilities::builder()
                .enable_resources()
                .enable_tools()
                .build(),
            server_info: Implementation::from_build_env(),
            instructions: Some("This server provides AI-powered trading system analysis tools using DeepSeek and FastEmbed. Available tools: similarity_search for finding similar bug patterns, analyze_code for security analysis, confirm_critical_bug for bug validation, and get_status for server information.".to_string()),
        }
    }

    async fn list_resources(
        &self,
        _request: Option<PaginatedRequestParam>,
        _: RequestContext<RoleServer>,
    ) -> Result<ListResourcesResult, ErrorData> {
        Ok(ListResourcesResult {
            resources: vec![
                self._create_resource_text("nautilus://vector_store", "Bug Pattern Vector Store"),
                self._create_resource_text("nautilus://deepseek_client", "DeepSeek AI Client"),
                self._create_resource_text("nautilus://analysis_results", "Analysis Results"),
            ],
            next_cursor: None,
        })
    }

    async fn read_resource(
        &self,
        ReadResourceRequestParam { uri }: ReadResourceRequestParam,
        _: RequestContext<RoleServer>,
    ) -> Result<ReadResourceResult, ErrorData> {
        match uri.as_str() {
            "nautilus://vector_store" => {
                let vs = self.vector_store.lock().await;
                let content = if vs.is_some() {
                    "Vector Store Status: Active\nType: FastEmbed with SQLite\nCapabilities: Local embeddings, similarity search, bug pattern storage"
                } else {
                    "Vector Store Status: Inactive"
                };
                Ok(ReadResourceResult {
                    contents: vec![ResourceContents::text(content, uri)],
                })
            }
            "nautilus://deepseek_client" => {
                let ds = self.deepseek_client.lock().await;
                let content = if ds.is_some() {
                    "DeepSeek Client Status: Active\nCapabilities: Code analysis, bug confirmation, critical vulnerability assessment"
                } else {
                    "DeepSeek Client Status: Inactive (API key required)"
                };
                Ok(ReadResourceResult {
                    contents: vec![ResourceContents::text(content, uri)],
                })
            }
            "nautilus://analysis_results" => {
                let content = "Analysis Results Repository\n\nThis resource provides access to stored analysis results and bug patterns from the Nautilus Trader Rig system.";
                Ok(ReadResourceResult {
                    contents: vec![ResourceContents::text(content, uri)],
                })
            }
            _ => Err(ErrorData::resource_not_found(
                "resource_not_found",
                Some(json!({
                    "uri": uri
                })),
            )),
        }
    }

    async fn list_resource_templates(
        &self,
        _request: Option<PaginatedRequestParam>,
        _: RequestContext<RoleServer>,
    ) -> Result<ListResourceTemplatesResult, ErrorData> {
        Ok(ListResourceTemplatesResult {
            next_cursor: None,
            resource_templates: Vec::new(),
        })
    }

    async fn initialize(
        &self,
        _request: InitializeRequestParam,
        context: RequestContext<RoleServer>,
    ) -> Result<InitializeResult, ErrorData> {
        if let Some(http_request_part) = context.extensions.get::<axum::http::request::Parts>() {
            let initialize_headers = &http_request_part.headers;
            let initialize_uri = &http_request_part.uri;
            tracing::info!(?initialize_headers, %initialize_uri, "initialize from http server");
        }

        // Initialize services asynchronously
        tokio::spawn({
            let server = self.clone();
            async move {
                if let Err(e) = server.initialize_services().await {
                    tracing::error!("Failed to initialize services: {}", e);
                }
            }
        });

        Ok(self.get_info())
    }
}

pub async fn run_mcp_server() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();

    tracing::info!("🚀 Starting Rig Analyzer MCP Server...");

    let service = TowerToHyperService::new(StreamableHttpService::new(
        || Ok(NautilusMcpServer::new()),
        LocalSessionManager::default().into(),
        Default::default(),
    ));
    let listener = tokio::net::TcpListener::bind("localhost:8080").await?;

    tracing::info!("🌐 MCP Server listening on http://localhost:8080");

    tokio::spawn({
        let service = service.clone();
        async move {
            loop {
                tokio::select! {
                    _ = tokio::signal::ctrl_c() => {
                        println!("Received Ctrl+C, shutting down MCP server");
                        break;
                    }
                    accept = listener.accept() => {
                        match accept {
                            Ok((stream, addr)) => {
                                tracing::debug!("New connection from: {}", addr);
                                let io = TokioIo::new(stream);
                                let service = service.clone();

                                tokio::spawn(async move {
                                    if let Err(e) = Builder::new(TokioExecutor::default())
                                        .serve_connection(io, service)
                                        .await
                                    {
                                        tracing::error!("Connection error: {e:?}");
                                    }
                                });
                            }
                            Err(e) => {
                                tracing::error!("Accept error: {e:?}");
                            }
                        }
                    }
                }
            }
        }
    });

    Ok(())
}

#[allow(dead_code)]
pub async fn test_mcp_client() -> anyhow::Result<()> {
    let transport =
        rmcp::transport::StreamableHttpClientTransport::from_uri("http://localhost:8080");

    let client_info = ClientInfo {
        protocol_version: Default::default(),
        capabilities: ClientCapabilities::default(),
        client_info: Implementation {
            name: "rig-analyzer".to_string(),
            version: "0.1.0".to_string(),
        },
    };

    let client = client_info.serve(transport).await.inspect_err(|e| {
        tracing::error!("client error: {:?}", e);
    })?;

    // Initialize
    let server_info = client.peer_info();
    tracing::info!("Connected to MCP server: {server_info:#?}");

    // List tools
    let tools: Vec<Tool> = client.list_tools(Default::default()).await?.tools;
    tracing::info!(
        "Available tools: {:?}",
        tools.iter().map(|t| &t.name).collect::<Vec<_>>()
    );

    // Test with DeepSeek if available
    let deepseek_client = deepseek::Client::from_env();
    let agent = deepseek_client
    .agent(deepseek::DEEPSEEK_REASONER)
    .preamble("You are a helpful assistant with access to Rig Analyzer MCP tools for code analysis.")
        .build();

    let res = agent
        .prompt("Search for authentication vulnerabilities and analyze them")
        .await?;

    println!("DeepSeek response: {res}");

    Ok(())
}
